# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core.docsets import DocSetStore
from dgml_core.errors import (
    AuthError,
    GroundedConfigInvalid,
    GroundedConfigMissing,
    SchemaGenerationFailed,
    SchemaNotFound,
    ValuesExtractionFailed,
)
from dgml_core.extraction_schema import parse_rnc
from dgml_core.extraction_xml import dgml_xml_to_values
from dgml_core.grounded import (
    DEFAULT_MAX_TOOL_ITERS,
    GroundedConfig,
    extract_values,
    generate_schema,
    get_page_words,
    load_grounded_config,
)
from dgml_core.models import FileRecord
from dgml_core.storage import Workspace, write_json_atomic

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_SCHEMA_MODEL = "anthropic/claude-opus-4-7"
DEFAULT_VALUES_MODEL = "gemini/gemini-2.5-pro"

# Extraction schemas are RNC at rest. A lowercase `title` tag keeps the
# engine's JSON-Schema property key "title" (matching the mocked LLM output).
_TITLE_RNC = """\
namespace dg = "http://dgml.io/ns/dg#"
namespace docset = "http://www.dgml.io/ws/Test"

start =
  element dg:chunk {
    (text | title)*
  }

title =
  element docset:title {
    text
  }
"""


def _write_grounded_config(workspace: Workspace, section: dict[str, object]) -> None:
    workspace.config_path.write_text(
        json.dumps({"grounded": section}, indent=2) + "\n",
        encoding="utf-8",
    )


def _seed_file(
    workspace: Workspace,
    file_id: str,
    *,
    pdf_bytes: bytes = b"%PDF-1.4 fake\n",
    page_count: int = 1,
    filename: str = "doc.pdf",
) -> Path:
    """Create a minimal file record + a placeholder PDF on disk.

    Returns the source PDF path. The PDF doesn't need to be valid for
    these tests — `extract_values`/`generate_schema` read the bytes and
    hand them to a mocked litellm, which never inspects them.
    """
    record = FileRecord(
        id=file_id,
        original_path=f"/fake/{filename}",
        original_filename=filename,
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=page_count,
        text_mode="digital",
    )
    workspace.file_dir(file_id).mkdir(parents=True, exist_ok=True)
    write_json_atomic(workspace.file_json_path(file_id), record.to_json())
    pdf_path = workspace.file_dir(file_id) / filename
    pdf_path.write_bytes(pdf_bytes)
    return pdf_path


def _seed_page_text(
    workspace: Workspace,
    file_id: str,
    page: int,
    *,
    width: int = 1000,
    height: int = 1000,
    words: list[dict[str, Any]] | None = None,
) -> None:
    """Drop a ``page_text/page_N.json`` so :func:`get_page_words` has data."""
    if words is None:
        # Two trivial words. Boxes are integer image pixels [left, top,
        # right, bottom] throughout, so assertions read straight off them.
        words = [
            {"t": "Hello", "l": [100, 210, 182, 242]},
            {"t": "world", "l": [190, 210, 290, 242]},
        ]
    workspace.file_text_dir(file_id).mkdir(parents=True, exist_ok=True)
    payload = {
        "file_id": file_id,
        "page": page,
        "width": width,
        "height": height,
        "words": words,
    }
    write_json_atomic(workspace.file_text_dir(file_id) / f"page_{page}.json", payload)


def _seed_page_image(workspace: Workspace, file_id: str, page: int) -> None:
    """Drop a minimal PNG so phase-3 ``image_path.exists()`` passes.
    Bytes never reach a real decoder — litellm is mocked in these tests."""
    img_dir = workspace.file_dir(file_id) / "page_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"page_{page}.png").write_bytes(b"\x89PNG\r\n\x1a\n")


def _tool_call_response(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str = "call_1",
    cost_usd: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> SimpleNamespace:
    """A litellm-shaped completion response with one tool call.

    Cost/token fields are optional — when set they get plumbed through
    the same attributes :func:`dgml_core.usage.extract_cost_and_tokens` reads
    in production, so tests can lock telemetry math."""
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    msg = SimpleNamespace(content=None, tool_calls=[call])
    response = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    if cost_usd is not None:
        response._hidden_params = {"response_cost": cost_usd}
    if prompt_tokens is not None or completion_tokens is not None:
        total = (prompt_tokens or 0) + (completion_tokens or 0)
        response.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
        )
    return response


def _no_tool_call_response() -> SimpleNamespace:
    msg = SimpleNamespace(content="I have no tools.", tool_calls=[])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ---------------------------------------------------------------------------
# load_grounded_config
# ---------------------------------------------------------------------------


def test_load_config_missing_when_no_config_file(workspace: Workspace) -> None:
    with pytest.raises(GroundedConfigMissing):
        load_grounded_config(workspace)


def test_load_config_missing_when_no_grounded_section(workspace: Workspace) -> None:
    workspace.config_path.write_text(json.dumps({"ocr": {}}), encoding="utf-8")
    with pytest.raises(GroundedConfigMissing):
        load_grounded_config(workspace)


def test_load_config_invalid_when_not_object(workspace: Workspace) -> None:
    workspace.config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(GroundedConfigInvalid):
        load_grounded_config(workspace)


def test_load_config_invalid_when_grounded_not_object(workspace: Workspace) -> None:
    workspace.config_path.write_text(json.dumps({"grounded": "azure"}), encoding="utf-8")
    with pytest.raises(GroundedConfigInvalid):
        load_grounded_config(workspace)


def test_load_config_rejects_missing_models(workspace: Workspace) -> None:
    _write_grounded_config(workspace, {"values_model": DEFAULT_VALUES_MODEL})
    with pytest.raises(GroundedConfigInvalid):
        load_grounded_config(workspace)

    _write_grounded_config(workspace, {"schema_model": DEFAULT_SCHEMA_MODEL})
    with pytest.raises(GroundedConfigInvalid):
        load_grounded_config(workspace)


def test_load_config_rejects_non_positive_max_iters(workspace: Workspace) -> None:
    _write_grounded_config(
        workspace,
        {
            "schema_model": DEFAULT_SCHEMA_MODEL,
            "values_model": DEFAULT_VALUES_MODEL,
            "max_tool_iters": 0,
        },
    )
    with pytest.raises(GroundedConfigInvalid):
        load_grounded_config(workspace)


def test_load_config_defaults(workspace: Workspace) -> None:
    _write_grounded_config(
        workspace,
        {"schema_model": DEFAULT_SCHEMA_MODEL, "values_model": DEFAULT_VALUES_MODEL},
    )
    config = load_grounded_config(workspace)
    assert config.schema_model == DEFAULT_SCHEMA_MODEL
    assert config.values_model == DEFAULT_VALUES_MODEL
    assert config.schema_api_key_env is None
    assert config.values_api_key_env is None
    assert config.max_tool_iters == DEFAULT_MAX_TOOL_ITERS


def test_load_config_rejects_empty_api_key_env(workspace: Workspace) -> None:
    _write_grounded_config(
        workspace,
        {
            "schema_model": DEFAULT_SCHEMA_MODEL,
            "values_model": DEFAULT_VALUES_MODEL,
            "schema_api_key_env": "",
        },
    )
    with pytest.raises(GroundedConfigInvalid):
        load_grounded_config(workspace)


# ---------------------------------------------------------------------------
# get_page_words (no LLM)
# ---------------------------------------------------------------------------


def test_get_page_words_returns_pixel_coords(workspace: Workspace) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    _seed_page_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        width=2000,
        height=4000,
        words=[{"t": "Hi", "l": [500, 1000, 700, 1200]}],
    )
    out = get_page_words(workspace, "f1aaaaaaaaaa", page=1)
    assert out["page"] == 1
    assert out["total_words"] == 1
    word = out["words"][0]
    assert word["idx"] == 0
    assert word["text"] == "Hi"
    # Boxes pass through as integer image pixels [left, top, right, bottom].
    assert word["location"] == {
        "page_number": 1,
        "bounding_box": [500, 1000, 700, 1200],
    }


def test_get_page_words_slice(workspace: Workspace) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    _seed_page_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            {"t": "a", "l": [0, 0, 10, 10]},
            {"t": "b", "l": [10, 0, 20, 10]},
            {"t": "c", "l": [20, 0, 30, 10]},
            {"t": "d", "l": [30, 0, 40, 10]},
        ],
    )
    out = get_page_words(workspace, "f1aaaaaaaaaa", page=1, start_idx=1, end_idx=3)
    # total_words reflects the file, not the slice — needed for the LLM
    # to know whether to ask for more.
    assert out["total_words"] == 4
    assert [w["text"] for w in out["words"]] == ["b", "c"]
    assert [w["idx"] for w in out["words"]] == [1, 2]


def test_get_page_words_missing_page_raises(workspace: Workspace) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    from dgml_core.errors import FileNotFound

    with pytest.raises(FileNotFound):
        get_page_words(workspace, "f1aaaaaaaaaa", page=99)


def test_get_page_words_rejects_zero_page(workspace: Workspace) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    with pytest.raises(ValueError):
        get_page_words(workspace, "f1aaaaaaaaaa", page=0)


# ---------------------------------------------------------------------------
# generate_schema
# ---------------------------------------------------------------------------


# A minimal valid typed field tree the model might submit, and a richer one that
# exercises datatypes / collections. generate_schema now renders these to RNC.
_MIN_FIELDS = [{"name": "title", "kind": "field", "datatype": "text"}]
_TYPED_FIELDS = [
    {"name": "due_date", "kind": "field", "datatype": "date"},
    {
        "name": "line_items",
        "kind": "collection",
        "item": {
            "name": "line_item",
            "kind": "container",
            "fields": [
                {"name": "description", "kind": "field", "datatype": "text"},
                {"name": "amount", "kind": "field", "datatype": "decimal"},
            ],
        },
    },
]


def test_generate_schema_returns_typed_rnc(workspace: Workspace) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    response = _tool_call_response("submit_schema", {"fields": _TYPED_FIELDS})
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", return_value=response) as mock_completion:
        rnc = generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="Invoice")
    # The field tree is rendered straight to at-rest RNC — datatypes preserved,
    # no grounded_field JSON Schema in between.
    assert isinstance(rnc, str)
    assert 'namespace docset = "http://dgml.io/' in rnc
    assert "element docset:DueDate {\n    xsd:date" in rnc
    assert "element docset:Amount {\n    xsd:decimal" in rnc
    assert "LineItem*" in rnc  # collection expanded to a repeated item
    # It round-trips through the parser (i.e. it is valid RNC).
    from dgml_core.extraction_schema import parse_rnc

    assert [t.name for t in parse_rnc(rnc).roots] == ["DueDate", "LineItems"]
    # tool_choice forced to submit_schema, the PDF was passed inline.
    _, kwargs = mock_completion.call_args
    assert kwargs["model"] == DEFAULT_SCHEMA_MODEL
    assert kwargs["tool_choice"]["function"]["name"] == "submit_schema"
    user_content = kwargs["messages"][1]["content"]
    assert any(c.get("type") == "file" for c in user_content)


def test_generate_schema_no_tool_call_errors(workspace: Workspace) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", return_value=_no_tool_call_response()):
        with pytest.raises(SchemaGenerationFailed):
            generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D")


def test_generate_schema_wrong_tool_errors(workspace: Workspace) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    response = _tool_call_response("not_the_right_tool", {"fields": []})
    with patch("litellm.completion", return_value=response):
        with pytest.raises(SchemaGenerationFailed):
            generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D")


def test_generate_schema_non_list_fields_errors(workspace: Workspace) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    response = _tool_call_response("submit_schema", {"fields": "not a list"})
    with patch("litellm.completion", return_value=response):
        with pytest.raises(SchemaGenerationFailed):
            generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D")


def test_generate_schema_invalid_field_tree_errors(workspace: Workspace) -> None:
    """A malformed field tree (bad datatype) surfaces as SchemaGenerationFailed,
    not a raw SchemaInvalid leaking out of the render step."""
    _seed_file(workspace, "f1aaaaaaaaaa")
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    bad = [{"name": "x", "kind": "field", "datatype": "not-a-type"}]
    response = _tool_call_response("submit_schema", {"fields": bad})
    with patch("litellm.completion", return_value=response):
        with pytest.raises(SchemaGenerationFailed):
            generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D")


def test_generate_schema_provider_exception_wrapped(workspace: Workspace) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", side_effect=RuntimeError("network down")):
        with pytest.raises(SchemaGenerationFailed) as exc:
            generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D")
    assert "network down" in str(exc.value)


def test_generate_schema_api_key_resolved(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    config = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL,
        values_model=DEFAULT_VALUES_MODEL,
        schema_api_key_env="MY_ANTHROPIC_KEY",
    )
    monkeypatch.setenv("MY_ANTHROPIC_KEY", "sk-test")
    response = _tool_call_response("submit_schema", {"fields": _MIN_FIELDS})
    with patch("litellm.completion", return_value=response) as mock_completion:
        generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D")
    _, kwargs = mock_completion.call_args
    assert kwargs["api_key"] == "sk-test"


def test_generate_schema_omits_reasoning_for_anthropic(workspace: Workspace) -> None:
    """Anthropic rejects extended-thinking + forced tool_choice. The schema
    generator forces tool_choice → must NOT send `reasoning_effort` for
    Anthropic-routed models. Other providers still get it.
    """
    _seed_file(workspace, "f1aaaaaaaaaa")
    response = _tool_call_response("submit_schema", {"fields": _MIN_FIELDS})

    # Anthropic — reasoning_effort must be stripped.
    anthropic_cfg = GroundedConfig(
        schema_model="anthropic/claude-opus-4-7",
        values_model=DEFAULT_VALUES_MODEL,
    )
    with patch("litellm.completion", return_value=response) as m:
        generate_schema(workspace, ["f1aaaaaaaaaa"], config=anthropic_cfg, docset_name="D")
    assert "reasoning_effort" not in m.call_args.kwargs

    # Gemini — kept.
    gemini_cfg = GroundedConfig(
        schema_model="gemini/gemini-2.5-pro",
        values_model=DEFAULT_VALUES_MODEL,
    )
    with patch("litellm.completion", return_value=response) as m:
        generate_schema(workspace, ["f1aaaaaaaaaa"], config=gemini_cfg, docset_name="D")
    assert m.call_args.kwargs["reasoning_effort"] == "high"


def test_extract_values_drops_temperature_for_anthropic(
    workspace: Workspace,
) -> None:
    """Anthropic-routed models never get ``temperature``: newer Claude
    models reject it as deprecated, and older ones only accept 1 with
    thinking enabled (which phase 1's auto tool_choice keeps on). Gemini
    keeps both knobs."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)
    phase1_values = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}
    response = _tool_call_response("submit_values", {"values": phase1_values})

    anthropic_cfg = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL, values_model="anthropic/claude-sonnet-5"
    )
    with patch("litellm.completion", return_value=response) as m:
        extract_values(workspace, ds_id, fid, config=anthropic_cfg)
    phase1_kwargs = m.call_args_list[0].kwargs
    assert phase1_kwargs["reasoning_effort"] == "high"
    assert "temperature" not in phase1_kwargs

    gemini_cfg = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL, values_model="gemini/gemini-2.5-pro"
    )
    with patch("litellm.completion", return_value=response) as m:
        extract_values(workspace, ds_id, fid, config=gemini_cfg)
    phase1_kwargs = m.call_args_list[0].kwargs
    assert phase1_kwargs["reasoning_effort"] == "high"
    assert phase1_kwargs["temperature"] == 0.0


def test_generate_schema_rejects_empty_file_list(workspace: Workspace) -> None:
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion") as mock_completion:
        with pytest.raises(SchemaGenerationFailed):
            generate_schema(workspace, [], config=config, docset_name="D")
    mock_completion.assert_not_called()


def test_generate_schema_sends_all_files(workspace: Workspace) -> None:
    """All attached PDFs land in the user-message content blocks."""
    _seed_file(workspace, "f1aaaaaaaaaa", pdf_bytes=b"%PDF-1.4 one\n")
    _seed_file(workspace, "f2aaaaaaaaaa", pdf_bytes=b"%PDF-1.4 two\n")
    _seed_file(workspace, "f3aaaaaaaaaa", pdf_bytes=b"%PDF-1.4 three\n")
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    response = _tool_call_response("submit_schema", {"fields": _MIN_FIELDS})
    with patch("litellm.completion", return_value=response) as mock_completion:
        generate_schema(
            workspace,
            ["f1aaaaaaaaaa", "f2aaaaaaaaaa", "f3aaaaaaaaaa"],
            config=config,
            docset_name="D",
        )
    _, kwargs = mock_completion.call_args
    user_content = kwargs["messages"][1]["content"]
    file_blocks = [c for c in user_content if c.get("type") == "file"]
    assert len(file_blocks) == 3
    # Prompt mentions the file count.
    text_blocks = [c for c in user_content if c.get("type") == "text"]
    assert any("3 attached PDFs" in c["text"] for c in text_blocks)


def test_generate_schema_literal_api_key_used_directly(workspace: Workspace) -> None:
    """A literal `schema_api_key` in config is sent verbatim to litellm,
    without going through os.environ."""
    _seed_file(workspace, "f1aaaaaaaaaa")
    config = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL,
        values_model=DEFAULT_VALUES_MODEL,
        schema_api_key="sk-direct-literal",
    )
    response = _tool_call_response("submit_schema", {"fields": _MIN_FIELDS})
    with patch("litellm.completion", return_value=response) as mock_completion:
        generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D")
    _, kwargs = mock_completion.call_args
    assert kwargs["api_key"] == "sk-direct-literal"


def test_load_config_rejects_both_literal_and_env_for_same_side(
    workspace: Workspace,
) -> None:
    _write_grounded_config(
        workspace,
        {
            "schema_model": DEFAULT_SCHEMA_MODEL,
            "values_model": DEFAULT_VALUES_MODEL,
            "schema_api_key": "sk-direct",
            "schema_api_key_env": "ANTHROPIC_API_KEY",
        },
    )
    with pytest.raises(GroundedConfigInvalid):
        load_grounded_config(workspace)


def test_load_config_accepts_literal_keys(workspace: Workspace) -> None:
    _write_grounded_config(
        workspace,
        {
            "schema_model": DEFAULT_SCHEMA_MODEL,
            "values_model": DEFAULT_VALUES_MODEL,
            "schema_api_key": "sk-ant-direct",
            "values_api_key": "g-direct",
        },
    )
    cfg = load_grounded_config(workspace)
    assert cfg.schema_api_key == "sk-ant-direct"
    assert cfg.values_api_key == "g-direct"
    assert cfg.schema_api_key_env is None
    assert cfg.values_api_key_env is None


def test_generate_schema_api_key_env_unset_raises_auth_error(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    monkeypatch.delenv("MY_ANTHROPIC_KEY", raising=False)
    config = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL,
        values_model=DEFAULT_VALUES_MODEL,
        schema_api_key_env="MY_ANTHROPIC_KEY",
    )
    with pytest.raises(AuthError):
        generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D")


# ---------------------------------------------------------------------------
# extract_values
# ---------------------------------------------------------------------------


def _seed_docset_with_schema(workspace: Workspace, file_id: str) -> tuple[str, str]:
    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _TITLE_RNC)
    store.add_file(ds.id, file_id)
    return ds.id, _TITLE_RNC


def test_extract_values_direct_submit(workspace: Workspace) -> None:
    """Phase 1 LLM submits text+page; phase 2 matcher finds the text in
    the seeded OCR words and fills in a bbox in code. No phase-3 call."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)  # "Hello", "world"
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    phase1_values = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": phase1_values}),
    ) as mock_completion:
        result = extract_values(workspace, ds_id, fid, config=config)

    # Only phase 1 calls the LLM — phase 2 was code, phase 3 had nothing to do.
    assert mock_completion.call_count == 1
    # Boxes are integer image pixels [left, top, right, bottom]: the span
    # "Hello world" unions the two seeded words (l=100..182, 190..290;
    # top=210, bottom=242) → left=100, top=210, right=290, bottom=242.
    title = result.values["title"]
    assert title["text"] == "Hello world"
    assert title["locations"] == [{"page_number": 1, "bounding_box": [100, 210, 290, 242]}]
    # Persisted as a dg:extraction element in the file's core <stem>.dgml.xml
    # (no separate file). With no prior document tree, mode is "extraction".
    assert result.mode == "extraction"
    assert result.xml_path == workspace.file_dgml_xml_path(ds_id, fid, "doc")
    xml = result.xml_path.read_text(encoding="utf-8")
    assert "<dg:extraction>" in xml
    vocab = parse_rnc(DocSetStore(workspace).get_schema(ds_id))
    assert dgml_xml_to_values(xml, vocab=vocab) == result.values


def test_extract_values_full_extraction_embeds_in_existing_tree(workspace: Workspace) -> None:
    """When the file's core <stem>.dgml.xml already exists (generate ran),
    extraction embeds a dg:extraction sibling and preserves the tree."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    # Simulate a prior `docset generate`: a core file with a document tree.
    core = workspace.file_dgml_xml_path(ds_id, fid, "doc")
    core.parent.mkdir(parents=True, exist_ok=True)
    core.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">\n'
        "  <dg:chunk>the generated document tree</dg:chunk>\n"
        "</dg:chunk>\n",
        encoding="utf-8",
    )

    phase1_values = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": phase1_values}),
    ):
        result = extract_values(workspace, ds_id, fid, config=config)

    assert result.mode == "full-extraction"
    assert result.xml_path == core
    xml = core.read_text(encoding="utf-8")
    assert "the generated document tree" in xml  # tree preserved
    assert xml.count("<dg:extraction>") == 1  # extraction added once


def test_extract_values_phase3_resolves_unmatched_via_llm(workspace: Workspace) -> None:
    """When phase 2 can't find the text in OCR, phase 3 sends the
    page + the unmatched id list to the LLM and patches the returned
    bbox into the values tree."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)  # only contains "Hello", "world"
    _seed_page_image(workspace, fid, 1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    # Phase 1: text NOT in OCR words → phase 2 leaves it unmatched.
    phase1_values = {"title": {"text": "Goodnight", "locations": [{"page_number": 1}]}}
    # Phase 3: the model returns a pixel bbox [left, top, right, bottom]
    # keyed by the id the matcher assigned.
    phase3_args = {"locations": [{"id": "a", "bounding_boxes": [[100, 56, 200, 76]]}]}
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        side_effect=[
            _tool_call_response("submit_values", {"values": phase1_values}, call_id="p1"),
            _tool_call_response("submit_locations", phase3_args, call_id="p3"),
        ],
    ) as mock_completion:
        result = extract_values(workspace, ds_id, fid, config=config)

    assert mock_completion.call_count == 2
    # Phase 3 spec: tool_choice forces submit_locations, with the ids
    # restricted to what the matcher actually couldn't resolve.
    phase3_kwargs = mock_completion.call_args_list[1].kwargs
    assert phase3_kwargs["tool_choice"]["function"]["name"] == "submit_locations"
    submit_tool = phase3_kwargs["tools"][0]
    id_enum = submit_tool["function"]["parameters"]["properties"]["locations"]["items"][
        "properties"
    ]["id"]["enum"]
    assert id_enum == ["a"]  # one unmatched item ⇒ one id
    # Final values carry the phase-3 bbox patched in.
    title = result.values["title"]
    assert title["locations"] == [{"page_number": 1, "bounding_box": [100, 56, 200, 76]}]


def test_extract_values_phase3_merges_costs_across_parallel_pages(
    workspace: Workspace,
) -> None:
    """Phase 3 spawns one LLM call per page in a ThreadPoolExecutor and
    merges per-page totals after the join. This test seeds two pages of
    unmatched items, gives each phase-3 call a distinct mocked cost, and
    asserts the merged total equals the sum — locking the merge math
    that the single-page tests never exercise."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid, page_count=2)
    _seed_page_text(workspace, fid, page=1)  # "Hello", "world"
    _seed_page_text(workspace, fid, page=2)  # "Hello", "world"
    _seed_page_image(workspace, fid, 1)
    _seed_page_image(workspace, fid, 2)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    # Two phase-1 values, one per page, neither resolvable by phase 2.
    phase1_values = {
        "title": {"text": "Goodnight", "locations": [{"page_number": 1}]},
        "subtitle": {"text": "Farewell", "locations": [{"page_number": 2}]},
    }
    # Phase 3 patches id 'a' on each page. Same id is fine because the
    # id namespace resets per page (matching.py:551).
    phase3_args = {"locations": [{"id": "a", "bounding_boxes": [[10.0, 20.0, 30.0, 40.0]]}]}
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        side_effect=[
            _tool_call_response(
                "submit_values",
                {"values": phase1_values},
                call_id="p1",
                cost_usd=0.01,
                prompt_tokens=100,
                completion_tokens=50,
            ),
            _tool_call_response(
                "submit_locations",
                phase3_args,
                call_id="p3a",
                cost_usd=0.02,
                prompt_tokens=200,
                completion_tokens=10,
            ),
            _tool_call_response(
                "submit_locations",
                phase3_args,
                call_id="p3b",
                cost_usd=0.04,
                prompt_tokens=300,
                completion_tokens=15,
            ),
        ],
    ) as mock_completion:
        extract_values(workspace, ds_id, fid, config=config)

    assert mock_completion.call_count == 3
    stats = json.loads(
        workspace.docset_file_extraction_stats_path(ds_id, fid).read_text(encoding="utf-8")
    )
    # Phase 3 ran two parallel page-calls; merged cost == sum of both.
    assert stats["phases"]["phase3"]["page_calls"] == 2
    assert stats["phases"]["phase3"]["cost_usd"] == pytest.approx(0.06)
    assert stats["phases"]["phase3"]["prompt_tokens"] == 500
    assert stats["phases"]["phase3"]["completion_tokens"] == 25
    assert stats["phases"]["phase3"]["total_tokens"] == 525
    # Phase 1's cost stays separate from phase 3's.
    assert stats["phases"]["phase1"]["cost_usd"] == pytest.approx(0.01)


def test_extract_values_writes_stats_file(workspace: Workspace) -> None:
    """Every successful extract_values writes extraction_stats.json with
    per-phase timings and match counts. The UX reads this directly."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    phase1_values = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": phase1_values}),
    ):
        extract_values(workspace, ds_id, fid, config=config)

    stats_path = workspace.docset_file_extraction_stats_path(ds_id, fid)
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    # Lock the top-level shape — this file is read by the UX
    # (StatsPanel) and is part of the on-disk surface.
    assert set(stats.keys()) == {
        "completed_at",
        "model",
        "outcome",
        "error",
        "phases",
        "matching",
        "phase1_layout",
    }
    assert stats["outcome"] == "ok"
    assert stats["error"] is None
    # Phase 2 matched the only location; phase 3 not needed.
    assert stats["matching"] == {
        "total_locations": 1,
        "matched_phase2": 1,
        "matched_phase3": 0,
        "unmatched": 0,
        "computed_fields": 0,
        "dropped_refs": 0,
    }
    # Per-phase shape. Phase 2 has no LLM, so no cost/token fields.
    assert set(stats["phases"].keys()) == {"phase1", "phase2", "phase3"}
    assert set(stats["phases"]["phase1"].keys()) == {
        "duration_s",
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    assert set(stats["phases"]["phase2"].keys()) == {"duration_s"}
    assert set(stats["phases"]["phase3"].keys()) == {
        "duration_s",
        "page_calls",
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    assert stats["phases"]["phase3"]["page_calls"] == 0
    assert "duration_s" in stats["phases"]["phase2"]
    assert stats["phases"]["phase3"]["page_calls"] == 0


def test_extract_values_write_stats_false_suppresses_file(workspace: Workspace) -> None:
    """write_stats=False (set by the CLI unless --debug) skips the
    extraction_stats.json sidecar; extracted.dgml.xml is still written."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    phase1_values = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": phase1_values}),
    ):
        extract_values(workspace, ds_id, fid, config=config, write_stats=False)

    assert not workspace.docset_file_extraction_stats_path(ds_id, fid).exists()
    assert workspace.file_dgml_xml_path(ds_id, fid, "doc").exists()


def test_extract_values_no_tool_call_errors(workspace: Workspace) -> None:
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", return_value=_no_tool_call_response()):
        with pytest.raises(ValuesExtractionFailed):
            extract_values(workspace, ds_id, fid, config=config)


def test_extract_values_unknown_tool_errors(workspace: Workspace) -> None:
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("frobnicate", {}),
    ):
        with pytest.raises(ValuesExtractionFailed, match="unknown tool"):
            extract_values(workspace, ds_id, fid, config=config)


def test_extract_values_submit_without_values_errors(workspace: Workspace) -> None:
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {}),
    ):
        with pytest.raises(ValuesExtractionFailed, match="values"):
            extract_values(workspace, ds_id, fid, config=config)


def test_extract_values_max_iters_exceeded(workspace: Workspace) -> None:
    """Phase 1's tool loop accepts ``get_page_words`` as a continuation
    even though it isn't in the published tools list (defensive). If the
    model keeps calling it instead of submitting, the loop bails after
    ``max_tool_iters``."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    config = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL,
        values_model=DEFAULT_VALUES_MODEL,
        max_tool_iters=3,
    )
    with patch(
        "litellm.completion",
        side_effect=[
            _tool_call_response("get_page_words", {"page": 1}, call_id=f"c{i}") for i in range(3)
        ],
    ):
        with pytest.raises(ValuesExtractionFailed, match="max_tool_iters"):
            extract_values(workspace, ds_id, fid, config=config)


def test_generate_schema_records_usage_on_success(workspace: Workspace) -> None:
    from dgml_core.usage import read_events

    _seed_file(workspace, "f1aaaaaaaaaa")
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    response = _tool_call_response("submit_schema", {"fields": _MIN_FIELDS})
    response._hidden_params = {"response_cost": 0.012}
    response.usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200, total_tokens=1200)
    with patch("litellm.completion", return_value=response):
        generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D", debug=True)

    events = read_events(workspace)
    assert len(events) == 1
    e = events[0]
    assert e["operation"] == "schema_generate"
    assert e["model"] == DEFAULT_SCHEMA_MODEL
    assert e["cost_usd"] == 0.012
    assert e["total_tokens"] == 1200
    assert e["outcome"] == "ok"
    assert e["error"] is None
    assert e["context"]["from_file_ids"] == ["f1aaaaaaaaaa"]


def test_generate_schema_records_usage_on_provider_exception(workspace: Workspace) -> None:
    from dgml_core.usage import read_events

    _seed_file(workspace, "f1aaaaaaaaaa")
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", side_effect=RuntimeError("network down")):
        with pytest.raises(SchemaGenerationFailed):
            generate_schema(workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D", debug=True)

    events = read_events(workspace)
    assert len(events) == 1
    e = events[0]
    assert e["operation"] == "schema_generate"
    assert e["outcome"] == "error"
    assert "network down" in (e["error"] or "")
    assert e["cost_usd"] is None  # no response → no cost data


def test_generate_schema_no_usage_recording_without_debug(workspace: Workspace) -> None:
    """Usage recording is gated on --debug: a normal (non-debug) schema
    generation writes no usage.jsonl row."""
    from dgml_core.usage import read_events

    _seed_file(workspace, "f1aaaaaaaaaa")
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    response = _tool_call_response("submit_schema", {"fields": _MIN_FIELDS})
    response._hidden_params = {"response_cost": 0.012}
    response.usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200, total_tokens=1200)
    with patch("litellm.completion", return_value=response):
        generate_schema(
            workspace, ["f1aaaaaaaaaa"], config=config, docset_name="D"
        )  # debug defaults False

    assert read_events(workspace) == []


def test_extract_values_records_one_event_summing_across_turns(workspace: Workspace) -> None:
    """A 3-phase extraction produces a SINGLE usage event summing
    cost/tokens across phase 1 + phase 3 (phase 2 is code-only)."""
    from dgml_core.usage import read_events

    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)  # only "Hello" + "world"
    _seed_page_image(workspace, fid, 1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    # Phase 1 returns text not in OCR → phase 2 leaves it unmatched →
    # phase 3 runs and supplies the bbox.
    phase1_values = {"title": {"text": "Goodnight", "locations": [{"page_number": 1}]}}
    p1 = _tool_call_response("submit_values", {"values": phase1_values}, call_id="p1")
    p1._hidden_params = {"response_cost": 0.002}
    p1.usage = SimpleNamespace(prompt_tokens=200, completion_tokens=30, total_tokens=230)
    p3 = _tool_call_response(
        "submit_locations",
        {"locations": [{"id": "a", "bounding_boxes": [[1, 2, 3, 4]]}]},
        call_id="p3",
    )
    p3._hidden_params = {"response_cost": 0.003}
    p3.usage = SimpleNamespace(prompt_tokens=600, completion_tokens=80, total_tokens=680)

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", side_effect=[p1, p3]):
        extract_values(workspace, ds_id, fid, config=config, debug=True)

    events = read_events(workspace)
    assert len(events) == 1
    e = events[0]
    assert e["operation"] == "extract_values"
    assert e["model"] == DEFAULT_VALUES_MODEL
    assert e["cost_usd"] == 0.005  # phase 1 + phase 3
    assert e["prompt_tokens"] == 800
    assert e["completion_tokens"] == 110
    assert e["total_tokens"] == 910
    assert e["outcome"] == "ok"
    assert e["context"]["file_id"] == fid
    assert e["context"]["docset_id"] == ds_id


def test_extract_values_records_error_event_when_max_iters_exceeded(
    workspace: Workspace,
) -> None:
    from dgml_core.usage import read_events

    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    # Phase 1 loops on get_page_words and never submits.
    r = _tool_call_response("get_page_words", {"page": 1}, call_id="cX")
    r._hidden_params = {"response_cost": 0.001}
    r.usage = SimpleNamespace(prompt_tokens=100, completion_tokens=5, total_tokens=105)
    config = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL,
        values_model=DEFAULT_VALUES_MODEL,
        max_tool_iters=3,
    )
    with patch("litellm.completion", side_effect=[r, r, r]):
        with pytest.raises(ValuesExtractionFailed, match="max_tool_iters"):
            extract_values(workspace, ds_id, fid, config=config, debug=True)
    events = read_events(workspace)
    assert len(events) == 1
    e = events[0]
    assert e["outcome"] == "error"
    assert e["cost_usd"] == 0.003  # 3 phase-1 calls x 0.001 — partial cost still recorded
    assert "max_tool_iters" in (e["error"] or "")


def test_expand_refs_inlines_definitions(workspace: Workspace) -> None:
    """A schema with `$ref` pointers gets flattened so the resulting
    spec is self-contained — what we hand to a tool-call validator
    can't rely on every provider resolving $ref the same way."""
    from dgml_core.grounded import _expand_refs

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": {
            "grounded_field": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "locations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "page_number": {"type": "integer"},
                                "bounding_box": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                            },
                            "required": ["page_number", "bounding_box"],
                        },
                    },
                },
                "required": ["text", "locations"],
            }
        },
        "type": "object",
        "properties": {
            "title": {"$ref": "#/definitions/grounded_field"},
            "lines": {
                "type": "array",
                "items": {"$ref": "#/definitions/grounded_field"},
            },
        },
    }
    expanded = _expand_refs(schema)
    # The `$schema` and `definitions` blocks are stripped because the
    # refs are now inlined.
    assert "$schema" not in expanded
    assert "definitions" not in expanded
    # No "$ref" anywhere in the output — fully self-contained.
    assert "$ref" not in json.dumps(expanded)
    # The grounded_field body landed at every reference site.
    title = expanded["properties"]["title"]
    assert title["type"] == "object"
    assert "page_number" in title["properties"]["locations"]["items"]["properties"]
    lines_item = expanded["properties"]["lines"]["items"]
    assert lines_item["type"] == "object"


def test_expand_refs_passthrough_when_no_refs() -> None:
    """No `$ref`s and no `definitions` to drop, but the object still
    picks up `additionalProperties: false` from the strict-objects
    rule (which is the point — extra-property typos like
    `bounding_2_box` get rejected by the provider even on schemas
    that never used $ref)."""
    from dgml_core.grounded import _expand_refs

    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert _expand_refs(schema) == {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": False,
    }


def test_expand_refs_forces_additional_properties_false() -> None:
    """Constrained objects get ``additionalProperties: false`` so the
    provider's tool-call validator rejects unknown property names
    (`_page_number`, `bounding_2_box`, etc.). Open-ended objects
    (no `properties`, or with an explicit `additionalProperties`)
    are left alone."""
    from dgml_core.grounded import _expand_refs

    schema = {
        "type": "object",
        "properties": {
            "a": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
            "b": {
                # operator-declared open-ended object — should NOT be tightened.
                "type": "object",
                "additionalProperties": True,
            },
            "c": {
                # No `properties` declared at all — also not tightened.
                "type": "object",
            },
        },
    }
    expanded = _expand_refs(schema)
    assert expanded["additionalProperties"] is False  # top-level constrained
    assert expanded["properties"]["a"]["additionalProperties"] is False
    # Already specified — leave it as the operator wrote it.
    assert expanded["properties"]["b"]["additionalProperties"] is True
    # No properties block → no tightening (open map style).
    assert "additionalProperties" not in expanded["properties"]["c"]


def test_expand_refs_chained_definitions() -> None:
    """A definition that itself references another definition expands
    transitively — chains collapse to a fully inlined tree."""
    from dgml_core.grounded import _expand_refs

    schema = {
        "definitions": {
            "inner": {"type": "string"},
            "outer": {
                "type": "object",
                "properties": {"v": {"$ref": "#/definitions/inner"}},
            },
        },
        "type": "object",
        "properties": {"a": {"$ref": "#/definitions/outer"}},
    }
    expanded = _expand_refs(schema)
    assert expanded["properties"]["a"]["properties"]["v"] == {"type": "string"}


def test_extract_values_phase1_submit_tool_strips_bbox(workspace: Workspace) -> None:
    """Phase 1's ``submit_values`` tool inlines the expanded docset schema
    with ``bounding_box`` stripped from ``grounded_field.locations`` — the
    provider validates only ``page_number`` at the tool-call layer, since
    phase 2 (code) is what attaches the bbox."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    # Seed OCR words containing "hi" so phase 2 matches and we don't
    # have to mock a phase-3 call.
    _seed_page_text(workspace, fid, page=1, words=[{"t": "hi", "l": [10, 20, 30, 40]}])

    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _TITLE_RNC)
    store.add_file(ds.id, fid)

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    phase1_values = {"title": {"text": "hi", "locations": [{"page_number": 1}]}}
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": phase1_values}),
    ) as mock_completion:
        extract_values(workspace, ds.id, fid, config=config)

    assert mock_completion.call_count == 1  # phase 2 matched ⇒ no phase 3
    phase1_tools = mock_completion.call_args_list[0].kwargs["tools"]
    submit_tool = next(t for t in phase1_tools if t["function"]["name"] == "submit_values")
    values_param = submit_tool["function"]["parameters"]["properties"]["values"]
    # Schema is expanded and self-contained — no $ref left.
    assert "definitions" not in values_param
    assert "$ref" not in json.dumps(values_param)
    # The leaf is the grounded/computed union; the grounded branch comes first.
    grounded_branch, computed_branch = values_param["properties"]["title"]["anyOf"]
    assert "derived_from" in computed_branch["properties"]
    # And crucially: bounding_box is gone from the grounded branch's locations[].
    location_props = grounded_branch["properties"]["locations"]["items"]["properties"]
    assert "page_number" in location_props
    assert "bounding_box" not in location_props


def test_extract_values_propagates_schema_not_found(workspace: Workspace) -> None:
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    store = DocSetStore(workspace)
    ds = store.create(name="No schema")
    store.add_file(ds.id, fid)

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion") as mock_completion:
        with pytest.raises(SchemaNotFound):
            extract_values(workspace, ds.id, fid, config=config)
    # We must not call the LLM if there's nothing to extract against.
    mock_completion.assert_not_called()


# ---------------------------------------------------------------------------
# computed (reasoned) fields — spec §7/§13


_COMPUTED_RNC = """\
namespace dg = "http://dgml.io/ns/dg#"
namespace docset = "http://www.dgml.io/ws/Test"

start =
  element dg:chunk {
    (text | title | word_count)*
  }

title =
  element docset:title {
    text
  }

## Prompt: Compute as the number of words in the title
word_count =
  element docset:word_count {
    xsd:integer
  }
"""


def test_extract_values_computed_field_end_to_end(workspace: Workspace) -> None:
    """A computed leaf flows through untouched by phases 2/3 (it carries no
    locations to ground), serializes with the spec's computed attribute set
    (dg:origin="computed", dg:value, dg:itemprop/dg:href + xml:id on the
    source), and is counted separately in extraction_stats.json."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)  # "Hello", "world"
    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _COMPUTED_RNC)
    store.add_file(ds.id, fid)

    phase1_values = {
        "title": {"text": "Hello world", "locations": [{"page_number": 1}]},
        "word_count": {
            "text": "2",
            "value": "2",
            "computed": True,
            "derived_from": ["title"],
        },
    }
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": phase1_values}),
    ) as mock_completion:
        result = extract_values(workspace, ds.id, fid, config=config)

    # One LLM call: phase 2 matched the grounded leaf; the computed leaf
    # never became an unmatched item, so no phase 3.
    assert mock_completion.call_count == 1
    assert result.values["word_count"] == phase1_values["word_count"]

    xml = result.xml_path.read_text(encoding="utf-8")
    assert 'dg:origin="computed"' in xml
    assert 'xsi:type="integer" dg:value="2"' in xml
    assert 'dg:itemprop="computedFrom"' in xml
    assert 'dg:href="#title"' in xml
    assert 'xml:id="title"' in xml
    # Round-trip through the persisted XML reproduces the values tree.
    vocab = parse_rnc(DocSetStore(workspace).get_schema(ds.id))
    assert dgml_xml_to_values(xml, vocab=vocab) == result.values

    stats = json.loads(
        workspace.docset_file_extraction_stats_path(ds.id, fid).read_text(encoding="utf-8")
    )
    assert stats["matching"] == {
        "total_locations": 1,
        "matched_phase2": 1,
        "matched_phase3": 0,
        "unmatched": 0,
        "computed_fields": 1,
        "dropped_refs": 0,
    }


def test_extract_values_counts_dropped_refs_in_stats(workspace: Workspace) -> None:
    """A computed leaf whose derived_from references values that were never
    extracted still lands in the XML, but the unresolvable entries are
    counted in stats so the incomplete provenance is visible."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _COMPUTED_RNC)
    store.add_file(ds.id, fid)

    phase1_values = {
        "title": {"text": "Hello world", "locations": [{"page_number": 1}]},
        "word_count": {
            "text": "2",
            "value": "2",
            "computed": True,
            # One resolvable ref, two that dangle (never extracted / malformed).
            "derived_from": ["title", "subtitle", "not a [valid path"],
        },
    }
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": phase1_values}),
    ):
        extract_values(workspace, ds.id, fid, config=config)

    stats = json.loads(
        workspace.docset_file_extraction_stats_path(ds.id, fid).read_text(encoding="utf-8")
    )
    assert stats["matching"]["computed_fields"] == 1
    assert stats["matching"]["dropped_refs"] == 2
