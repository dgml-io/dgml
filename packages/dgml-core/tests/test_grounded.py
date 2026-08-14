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
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core import layout
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
    _SCHEMA_TREE_MAX_DEPTH,
    DEFAULT_MAX_TOOL_ITERS,
    GroundedConfig,
    _field_node_schema,
    _submit_schema_tool,
    extract_values,
    generate_schema,
    get_page_words,
    load_grounded_config,
)
from dgml_core.models import FileRecord
from dgml_core.storage import Workspace

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
    from .conftest import write_config

    write_config(workspace, {"grounded": section})


def _seed_file(
    workspace: Workspace,
    file_id: str,
    *,
    pdf_bytes: bytes = b"%PDF-1.4 fake\n",
    page_count: int = 1,
    filename: str = "doc.pdf",
) -> None:
    """Create a minimal file record + a placeholder source PDF in the store.

    The PDF doesn't need to be valid for these tests — `extract_values` /
    `generate_schema` read the bytes and hand them to a mocked litellm, which
    never inspects them.
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
    workspace.docs.put_doc("files", file_id, record.to_json())
    workspace.blobs.put_blob(layout.file_source_key(file_id, filename), pdf_bytes)


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
    payload = {
        "file_id": file_id,
        "page": page,
        "width": width,
        "height": height,
        "words": words,
    }
    workspace.blobs.put_blob(layout.file_page_text_key(file_id, page), json.dumps(payload).encode())


def _seed_page_image(workspace: Workspace, file_id: str, page: int) -> None:
    """Drop a minimal PNG so phase-3 ``image_path.exists()`` passes.
    Bytes never reach a real decoder — litellm is mocked in these tests."""
    workspace.blobs.put_blob(layout.file_page_image_key(file_id, page), b"\x89PNG\r\n\x1a\n")


def _tool_call_response(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str = "call_1",
    cost_usd: float | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
) -> SimpleNamespace:
    """A litellm-shaped completion response with one tool call.

    Cost/token fields are optional — when set they get plumbed through
    the same attributes :func:`dgml_core.usage.extract_cost_and_tokens` reads
    in production, so tests can lock telemetry math. Cache counters use the
    litellm source names (``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``)."""
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    msg = SimpleNamespace(content=None, tool_calls=[call])
    response = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    if cost_usd is not None:
        response._hidden_params = {"response_cost": cost_usd}
    has_cache = cache_read_tokens is not None or cache_creation_tokens is not None
    if prompt_tokens is not None or completion_tokens is not None or has_cache:
        total = (prompt_tokens or 0) + (completion_tokens or 0)
        response.usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
        )
        if cache_read_tokens is not None:
            response.usage.cache_read_input_tokens = cache_read_tokens
        if cache_creation_tokens is not None:
            response.usage.cache_creation_input_tokens = cache_creation_tokens
    return response


def _truncated_response() -> SimpleNamespace:
    """A turn that stopped with finish_reason='length' mid-tool-call."""
    call = SimpleNamespace(
        id="trunc",
        function=SimpleNamespace(name="submit_values", arguments='{"values": {"Bi'),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[call]), finish_reason="length"
            )
        ]
    )


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
    # No [grounded] section and no [models] tiers → nothing resolves the models.
    workspace.config_path.write_text("[ocr]\n", encoding="utf-8")
    with pytest.raises(GroundedConfigMissing):
        load_grounded_config(workspace)


def test_load_config_invalid_when_malformed_toml(workspace: Workspace) -> None:
    from dgml_core.errors import CorruptMetadata

    workspace.config_path.write_text("this is = not = valid toml", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        load_grounded_config(workspace)


def test_load_config_invalid_when_grounded_not_object(workspace: Workspace) -> None:
    from dgml_core.errors import CorruptMetadata

    workspace.config_path.write_text('grounded = "azure"\n', encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        load_grounded_config(workspace)


def test_load_config_missing_model_without_tier(workspace: Workspace) -> None:
    # One model set on the section, the other neither overridden nor tiered.
    _write_grounded_config(workspace, {"values_model": DEFAULT_VALUES_MODEL})
    with pytest.raises(GroundedConfigMissing):
        load_grounded_config(workspace)

    _write_grounded_config(workspace, {"schema_model": DEFAULT_SCHEMA_MODEL})
    with pytest.raises(GroundedConfigMissing):
        load_grounded_config(workspace)


def test_load_config_models_from_tiers(workspace: Workspace) -> None:
    # No [grounded] section: schema ← expert tier, values ← advanced tier.
    from .conftest import write_config

    write_config(
        workspace,
        {"models": {"advanced": "gemini/gemini-2.5-pro", "expert": "anthropic/claude-opus-4-8"}},
    )
    config = load_grounded_config(workspace)
    assert config.schema_model == "anthropic/claude-opus-4-8"
    assert config.values_model == "gemini/gemini-2.5-pro"


def test_load_config_section_overrides_tier(workspace: Workspace) -> None:
    from .conftest import write_config

    write_config(
        workspace,
        {
            "models": {"advanced": "gemini/gemini-2.5-pro", "expert": "anthropic/claude-opus-4-8"},
            "grounded": {"values_model": "openai/gpt-5"},
        },
    )
    config = load_grounded_config(workspace)
    assert config.schema_model == "anthropic/claude-opus-4-8"  # tier
    assert config.values_model == "openai/gpt-5"  # override wins


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


def test_submit_schema_tool_declares_the_node_shape() -> None:
    """The field-tree item must be a *populated* object schema.

    A bare ``{"type": "object"}`` is a closed, empty object: providers that
    constrain-decode tool arguments (Gemini) then return ``{"fields": [{}, …]}``
    and generation dies in the parser with "missing a non-empty 'name'". Assert
    the declared keys are exactly the ones ``_field_node_to_tag`` reads, so the
    two can't drift.
    """
    item = _submit_schema_tool()["function"]["parameters"]["properties"]["fields"]["items"]
    # Assert it is the wired-up builder, so the node schema can't be improved
    # while the tool quietly keeps handing out something else.
    assert item == _field_node_schema(_SCHEMA_TREE_MAX_DEPTH)
    assert set(item["properties"]) == {
        "name",
        "kind",
        "datatype",
        "description",
        "example",
        "prompt",
        "fields",
        "item",
    }
    assert item["required"] == ["name", "kind"]
    assert item["properties"]["kind"]["enum"] == ["field", "container", "collection"]
    assert "text" in item["properties"]["datatype"]["enum"]


def test_field_node_schema_bottoms_out_at_leaves() -> None:
    """At the deepest level there is no ``fields``/``item`` slot left, so a
    container could only be childless — narrow ``kind`` to a leaf instead."""
    floor = _field_node_schema(0)
    assert floor["properties"]["kind"]["enum"] == ["field"]
    assert "fields" not in floor["properties"] and "item" not in floor["properties"]

    nested = _field_node_schema(1)
    assert nested["properties"]["fields"]["items"] == floor
    # Nesting is inlined, not $ref'd — provider adapters resolve $ref inconsistently.
    assert "$ref" not in json.dumps(nested)


def test_declared_datatype_enum_can_express_every_accepted_datatype() -> None:
    """``datatype`` is an enum, so a constrained decoder can only emit what it
    lists. ``_normalize_datatype`` also accepts ``"string"``, ``""`` and
    ``xsd:``-prefixed spellings — assert each of those normalizes to a value the
    enum *can* express, i.e. the narrowing costs redundant spellings, not types.
    """
    from dgml_core.extraction_schema import FIELD_DATATYPES, _normalize_datatype

    enum = _field_node_schema(0)["properties"]["datatype"]["enum"]
    aliases = ["", "string", "text", *(f"xsd:{d}" for d in sorted(FIELD_DATATYPES))]
    for alias in aliases:
        normalized = _normalize_datatype(alias, tag_name="X")
        # None is the untyped default, whose canonical enum spelling is "text".
        assert ("text" if normalized is None else normalized) in enum, alias


def test_declared_schema_accepts_a_collection_inside_a_collection() -> None:
    """A collection's ``item`` is a wrapper the parser flattens, so it must not
    cost a depth level: otherwise the prompt's recommended style (describe a
    collection via an explicit ``item``) bottoms out early and a nested
    collection comes back childless — which the parser renders as an empty
    element instead of failing.
    """
    node = _submit_schema_tool()["function"]["parameters"]["properties"]["fields"]["items"]
    # Walk the declared schema along `collection.item -> fields[] -> collection.item
    # -> fields[]` — the shape of an invoice whose line items each carry charges.
    # Every hop has to exist, or a constrained decoder cannot emit that tree.
    inner_item = node["properties"]["item"]
    inner_child = inner_item["properties"]["fields"]["items"]
    nested_item = inner_child["properties"]["item"]
    assert nested_item["properties"]["fields"]["items"]["properties"]["name"]

    # `item` inside `item` is what bounds the recursion — it must not be declared.
    assert "item" not in inner_item["properties"]


def test_submit_schema_tool_stays_within_geminis_decoder_budget() -> None:
    """Gemini rejects an over-large constrained-decoding schema with "too many
    states for serving". Measured against gemini-2.5-pro: the current schema
    (~13 KB) is served, one more level of depth (~27 KB) is not. Guard the size
    here so raising the depth fails in CI rather than in front of a user."""
    assert len(json.dumps(_submit_schema_tool())) < 20_000


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
    assert result.xml_key == layout.dgml_xml_key(ds_id, fid, "doc")
    xml = workspace.blobs.get_blob(result.xml_key).decode("utf-8")
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
    workspace.blobs.put_blob(
        layout.dgml_xml_key(ds_id, fid, "doc"),
        b'<?xml version="1.0" encoding="utf-8"?>\n'
        b'<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">\n'
        b"  <dg:chunk>the generated document tree</dg:chunk>\n"
        b"</dg:chunk>\n",
    )

    phase1_values = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": phase1_values}),
    ):
        result = extract_values(workspace, ds_id, fid, config=config)

    assert result.mode == "full-extraction"
    assert result.xml_key == layout.dgml_xml_key(ds_id, fid, "doc")
    xml = workspace.blobs.get_blob(layout.dgml_xml_key(ds_id, fid, "doc")).decode("utf-8")
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


def test_extract_values_phase3_words_are_toon_encoded(
    workspace: Workspace,
) -> None:
    """Phase 3 unconditionally renders the OCR word listing as the compact TOON
    table under the TOON system prompt — no env flag involved.

    The words reach the model as ``words[N]{idx,text,left,top,right,bottom}:``
    plus one row per word (never the old ``json.dumps(words, indent=2)`` array),
    the system prompt documents that TOON table, and the ``submit_locations``
    response contract is unchanged (the mocked reply patches the bbox in)."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)  # "Hello", "world"
    _seed_page_image(workspace, fid, 1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)

    phase1_values = {"title": {"text": "Goodnight", "locations": [{"page_number": 1}]}}
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

    messages = mock_completion.call_args_list[1].kwargs["messages"]
    system_text = messages[0]["content"]
    user_text = messages[1]["content"][0]["text"]

    # The word listing is the compact TOON table, and the JSON block is gone.
    assert "words[2]{idx,text,left,top,right,bottom}:" in user_text
    assert "TOON table" in system_text
    assert json.dumps(get_page_words(workspace, fid, 1)["words"], indent=2) not in user_text

    # Response contract unchanged: the bbox is patched in from submit_locations.
    assert result.values["title"]["locations"] == [
        {"page_number": 1, "bounding_box": [100, 56, 200, 76]}
    ]


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
    stats = workspace.docs.get_doc("extraction_stats", f"{ds_id}/{fid}")
    assert stats is not None
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

    stats = workspace.docs.get_doc("extraction_stats", f"{ds_id}/{fid}")
    assert stats is not None
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
        "phase1_tool_schema",
    }
    assert stats["phase1_tool_schema"] == "inlined"
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
        "unnormalized_enum_values": 0,
        "derivations_checked": 0,
        "derivations_mismatched": 0,
        "invariants_checked": 0,
        "invariants_violated": 0,
        "invariant_violations": [],
    }
    # Per-phase shape. Phase 2 has no LLM, so no cost/token fields.
    assert set(stats["phases"].keys()) == {"phase1", "phase2", "phase3"}
    assert set(stats["phases"]["phase1"].keys()) == {
        "duration_s",
        "chunk_calls",
        "truncated_retries",
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
    }
    assert stats["phases"]["phase1"]["chunk_calls"] == 1
    assert stats["phases"]["phase1"]["truncated_retries"] == 0
    assert set(stats["phases"]["phase2"].keys()) == {"duration_s"}
    assert set(stats["phases"]["phase3"].keys()) == {
        "duration_s",
        "page_calls",
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
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

    assert workspace.docs.get_doc("extraction_stats", f"{ds_id}/{fid}") is None
    assert workspace.blobs.blob_exists(layout.dgml_xml_key(ds_id, fid, "doc"))


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
    with ``bounding_box`` stripped from ``extracted_value.locations`` — the
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
    # The leaf is the merged extracted_value shape — grounded and computed
    # provenance share one object (no anyOf union).
    leaf = values_param["properties"]["title"]
    assert "derived_from" in leaf["properties"]
    assert "computed" in leaf["properties"]
    assert "value" in leaf["properties"]
    # And crucially: bounding_box is gone from the leaf's locations[].
    location_props = leaf["properties"]["locations"]["items"]["properties"]
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

    xml = workspace.blobs.get_blob(result.xml_key).decode("utf-8")
    assert 'dg:origin="computed"' in xml
    assert 'xsi:type="integer" dg:value="2"' in xml
    assert 'dg:itemprop="computedFrom"' in xml
    assert 'dg:href="#title"' in xml
    assert 'xml:id="title"' in xml
    # Round-trip through the persisted XML reproduces the values tree.
    vocab = parse_rnc(DocSetStore(workspace).get_schema(ds.id))
    assert dgml_xml_to_values(xml, vocab=vocab) == result.values

    stats = workspace.docs.get_doc("extraction_stats", f"{ds.id}/{fid}")
    assert stats is not None
    assert stats["matching"] == {
        "total_locations": 1,
        "matched_phase2": 1,
        "matched_phase3": 0,
        "unmatched": 0,
        "computed_fields": 1,
        "dropped_refs": 0,
        "unnormalized_enum_values": 0,
        "derivations_checked": 0,
        "derivations_mismatched": 0,
        "invariants_checked": 0,
        "invariants_violated": 0,
        "invariant_violations": [],
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

    stats = workspace.docs.get_doc("extraction_stats", f"{ds.id}/{fid}")
    assert stats is not None
    assert stats["matching"]["computed_fields"] == 1
    assert stats["matching"]["dropped_refs"] == 2


# ── Merged-leaf provenance normalization & enum tool-schema specialization ───


def test_normalize_leaf_provenance_grounded_wins_over_computed_markers() -> None:
    from dgml_core.grounded import _normalize_leaf_provenance

    values = {
        "Title": {
            "text": "hi",
            "computed": True,
            "derived_from": ["Other"],
            "locations": [{"page_number": 1}],
        }
    }
    _normalize_leaf_provenance(values)
    assert values["Title"] == {"text": "hi", "locations": [{"page_number": 1}]}


def test_normalize_leaf_provenance_computed_drops_empty_locations() -> None:
    from dgml_core.grounded import _normalize_leaf_provenance

    values = {
        "Total": {"text": "$5", "value": "5", "derived_from": ["Items[0].Amount"], "locations": []}
    }
    _normalize_leaf_provenance(values)
    assert values["Total"] == {
        "text": "$5",
        "value": "5",
        "computed": True,
        "derived_from": ["Items[0].Amount"],
    }


def test_normalize_leaf_provenance_filters_and_defaults() -> None:
    from dgml_core.grounded import _normalize_leaf_provenance

    values: dict[str, Any] = {
        "A": {"text": "x", "value": "", "locations": [{"page_number": 1}]},  # empty value dropped
        "B": {"text": "y", "computed": True, "derived_from": ["ok", 7]},  # non-str ref filtered
        "C": {"text": "z", "locations": []},  # unusable locations removed
        "Nested": [{"D": {"text": "w", "locations": [{"page_number": 2}]}}],
    }
    _normalize_leaf_provenance(values)
    assert values["A"] == {"text": "x", "locations": [{"page_number": 1}]}
    assert values["B"] == {"text": "y", "computed": True, "derived_from": ["ok"]}
    assert values["C"] == {"text": "z"}
    assert values["Nested"][0]["D"]["locations"] == [{"page_number": 2}]


def test_expand_refs_specializes_value_enum_and_carries_annotations() -> None:
    from dgml_core.extraction_schema import rnc_to_json_schema
    from dgml_core.grounded import _expand_refs

    rnc = (
        'namespace docset = "http://dgml.io/x/y"\n\n'
        "## Commodity being billed\n"
        "## Prompt: Classify by the service section heading\n"
        "MeterType =\n"
        "  element docset:MeterType {\n"
        '    ( "electric" | "water" )\n  }\n'
    )
    expanded = _expand_refs(rnc_to_json_schema(rnc))
    leaf = expanded["properties"]["MeterType"]
    # The enum lands on the leaf's `value` so providers constrain-decode it.
    assert leaf["properties"]["value"]["enum"] == ["electric", "water"]
    # Annotation sidecars survive expansion (guidance reaches the tool schema).
    assert leaf["prompt"] == "Classify by the service section heading"
    assert "value_enum" not in leaf
    # And the object is still tightened.
    assert leaf["additionalProperties"] is False


def test_extract_values_injects_docset_guidance(workspace: Workspace) -> None:
    """When the docset has extraction-guidance.md set, its text is appended to
    the phase-1 user prompt (after the schema, before the PDF block)."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1, words=[{"t": "hi", "l": [10, 20, 30, 40]}])

    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _TITLE_RNC)
    store.set_guidance(ds.id, "Classify charges by behavior, not by name.")
    store.add_file(ds.id, fid)

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    phase1_values = {"title": {"text": "hi", "locations": [{"page_number": 1}]}}
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": phase1_values}),
    ) as mock_completion:
        extract_values(workspace, ds.id, fid, config=config)

    messages = mock_completion.call_args_list[0].kwargs["messages"]
    user_text = messages[1]["content"][0]["text"]
    assert "DOCSET GUIDANCE" in user_text
    assert "Classify charges by behavior, not by name." in user_text
    # Guidance follows the schema (stable prefix for prompt caching).
    assert user_text.index("SCHEMA:") < user_text.index("DOCSET GUIDANCE")
    # And not set → not present.
    store.clear_guidance(ds.id)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": phase1_values}),
    ) as mock_completion:
        extract_values(workspace, ds.id, fid, config=config)
    user_text = mock_completion.call_args_list[0].kwargs["messages"][1]["content"][0]["text"]
    assert "DOCSET GUIDANCE" not in user_text


# ── Tier-3 robustness: annotation stripping, permissive fallback, truncation ─


def test_strip_annotations_drops_prose_keeps_enum() -> None:
    from dgml_core.extraction_schema import rnc_to_json_schema
    from dgml_core.grounded import _expand_refs, _strip_annotations

    rnc = (
        'namespace docset = "http://dgml.io/x/y"\n\n'
        "## Commodity being billed\n"
        "## Example: Electric Service\n"
        "## Prompt: Classify by the service section heading\n"
        "MeterType =\n"
        "  element docset:MeterType {\n"
        '    ( "electric" | "water" )\n  }\n'
    )
    stripped = _strip_annotations(_expand_refs(rnc_to_json_schema(rnc)))
    leaf = stripped["properties"]["MeterType"]
    assert "prompt" not in leaf and "example" not in leaf and "description" not in leaf
    assert "description" not in leaf["properties"]["text"]
    # Structure and the load-bearing enum survive.
    assert leaf["properties"]["value"]["enum"] == ["electric", "water"]
    assert leaf["additionalProperties"] is False
    assert "required" in leaf


def test_prune_to_vocabulary_drops_unknown_keys() -> None:
    from dgml_core.extraction_schema import parse_rnc
    from dgml_core.grounded import _prune_to_vocabulary

    rnc = (
        'namespace docset = "http://dgml.io/x/y"\n\n'
        "Bill =\n"
        "  element docset:Bill {\n"
        "    (text | Total | Items)*\n  }\n\n"
        "Total =\n"
        "  element docset:Total {\n"
        "    xsd:decimal\n  }\n\n"
        "Items =\n"
        "  element docset:Items {\n"
        "    Item*\n  }\n\n"
        "Item =\n"
        "  element docset:Item {\n"
        "    (text | Name)*\n  }\n\n"
        "Name =\n"
        "  element docset:Name {\n"
        "    text\n  }\n"
    )
    vocab = parse_rnc(rnc)
    values = {
        "Bill": {
            "Total": {"text": "$5", "locations": [{"page_number": 1}]},
            "Fabricated": {"text": "x"},  # not in the vocabulary
            "Items": [
                {"Name": {"text": "a", "locations": [{"page_number": 1}]}, "Extra": 1},
                "not-a-dict",  # wrong shape
            ],
        },
        "Unknown": {"text": "y"},
    }
    pruned = _prune_to_vocabulary(values, vocab)
    assert pruned == {
        "Bill": {
            "Total": {"text": "$5", "locations": [{"page_number": 1}]},
            "Items": [{"Name": {"text": "a", "locations": [{"page_number": 1}]}}],
        }
    }


def test_extract_values_permissive_fallback_on_too_many_states(workspace: Workspace) -> None:
    """A Gemini 'too many states for serving' rejection retries phase 1 once
    with a permissive object parameter, prunes the result against the
    vocabulary, and records the mode in extraction_stats.json."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1, words=[{"t": "hi", "l": [10, 20, 30, 40]}])

    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _TITLE_RNC)
    store.add_file(ds.id, fid)

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    phase1_values = {
        "title": {"text": "hi", "locations": [{"page_number": 1}]},
        "bogus_key": {"text": "fabricated"},
    }
    with patch(
        "litellm.completion",
        side_effect=[
            Exception("BadRequestError: The specified schema produces too many states for serving"),
            _tool_call_response("submit_values", {"values": phase1_values}),
        ],
    ) as mock_completion:
        result = extract_values(workspace, ds.id, fid, config=config)

    assert mock_completion.call_count == 2
    # Retry used the permissive values parameter (no inlined properties).
    retry_tools = mock_completion.call_args_list[1].kwargs["tools"]
    submit_tool = next(t for t in retry_tools if t["function"]["name"] == "submit_values")
    values_param = submit_tool["function"]["parameters"]["properties"]["values"]
    assert "properties" not in values_param
    # Fabricated key pruned code-side.
    assert "bogus_key" not in result.values
    assert result.values["title"]["text"] == "hi"

    stats = workspace.docs.get_doc("extraction_stats", f"{ds.id}/{fid}")
    assert stats is not None
    assert stats["phase1_tool_schema"] == "permissive"


def test_extract_values_other_provider_errors_do_not_fall_back(workspace: Workspace) -> None:
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _TITLE_RNC)
    store.add_file(ds.id, fid)

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        side_effect=Exception("BadRequestError: something unrelated"),
    ) as mock_completion:
        with pytest.raises(ValuesExtractionFailed, match="something unrelated"):
            extract_values(workspace, ds.id, fid, config=config)
    assert mock_completion.call_count == 1


def test_extract_values_truncated_output_reports_length(workspace: Workspace) -> None:
    """finish_reason == 'length' surfaces as an explicit truncation error, not
    an opaque malformed-JSON one."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _TITLE_RNC)
    store.add_file(ds.id, fid)

    call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="submit_values", arguments='{"values": {"tr'),
    )
    msg = SimpleNamespace(content=None, tool_calls=[call])
    truncated = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")])

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", return_value=truncated):
        with pytest.raises(ValuesExtractionFailed, match="truncated"):
            extract_values(workspace, ds.id, fid, config=config)


def test_pdf_cached_only_in_chunked_mode(workspace: Workspace) -> None:
    """The per-file PDF earns a cache breakpoint only where a read can happen.

    Chunked mode is several turns over one unchanged prefix, so turns after the
    first read the document back. On the ordinary single-turn path a breakpoint
    could only ever pay the write premium, so the document stays untagged and
    the only markers are the transport's system message and the cross-file
    schema block.
    """
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1, words=[{"t": "hi", "l": [10, 20, 30, 40]}])
    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _TITLE_RNC)
    store.add_file(ds.id, fid)

    values = {"title": {"text": "hi", "locations": [{"page_number": 1}]}}
    config = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL, values_model="anthropic/claude-sonnet-5"
    )

    def pdf_cached(call: Any) -> bool:
        blocks = call.kwargs["messages"][1]["content"]
        pdf = next(b for b in blocks if b.get("type") == "file")
        return "cache_control" in pdf

    # Single shot: schema block + system only.
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": values}),
    ) as m:
        extract_values(workspace, ds.id, fid, config=config)
    assert pdf_cached(m.call_args_list[0]) is False
    assert _cache_control_paths(m.call_args_list[0].kwargs["messages"]) == [
        "[0].content[0]",
        "[1].content[0]",
    ]

    # After a truncation, the chunked retry tags the document too.
    with patch(
        "litellm.completion",
        side_effect=[
            _truncated_response(),
            _tool_call_response("submit_values", {"values": values}),
        ],
    ) as m:
        extract_values(workspace, ds.id, fid, config=config)
    assert pdf_cached(m.call_args_list[0]) is False
    assert pdf_cached(m.call_args_list[1]) is True


def test_merge_values_semantics() -> None:
    from dgml_core.grounded import _merge_values

    base = {
        "Title": {"text": "old", "locations": [{"page_number": 1}]},
        "Items": [{"A": {"text": "1"}}],
        "Group": {"X": {"text": "x"}},
    }
    extra = {
        "Title": {"text": "new", "locations": [{"page_number": 2}]},  # leaf replaces
        "Items": [{"A": {"text": "2"}}],  # lists concatenate
        "Group": {"Y": {"text": "y"}},  # non-leaf dicts merge
    }
    merged = _merge_values(base, extra)
    assert merged["Title"]["text"] == "new"
    assert [e["A"]["text"] for e in merged["Items"]] == ["1", "2"]
    assert set(merged["Group"]) == {"X", "Y"}


def test_apply_append_entries_paths_and_errors() -> None:
    from dgml_core.grounded import _apply_append_entries

    assert "error" in _apply_append_entries(None, {"path": "Items", "entries": []})
    acc = {"values": {"Items": [{"A": {"text": "1"}}], "Scalar": {"text": "s"}}}
    assert "error" in _apply_append_entries(acc, {"entries": []})  # no path
    assert "error" in _apply_append_entries(acc, {"path": "Items"})  # no entries
    assert "error" in _apply_append_entries(acc, {"path": "Nope", "entries": []})
    assert "error" in _apply_append_entries(acc, {"path": "Scalar", "entries": []})
    ok = _apply_append_entries(acc, {"path": "Items", "entries": [{"A": {"text": "2"}}]})
    assert ok == {"recorded": 1, "total_entries": 2}
    assert len(acc["values"]["Items"]) == 2


_CHUNK_RNC = """\
namespace docset = "http://dgml.io/x/chunky"

Bill =
  element docset:Bill {
    (text | Total | Items)*
  }

Total =
  element docset:Total {
    xsd:decimal
  }

Items =
  element docset:Items {
    Item*
  }

Item =
  element docset:Item {
    (text | Name)*
  }

Name =
  element docset:Name {
    text
  }
"""


def test_extract_values_chunked_submission_merges_and_prunes(workspace: Workspace) -> None:
    """submit_values(done=false) + append_entries(...) + append_entries(done=true)
    merge into one values tree; fabricated keys in appended entries are pruned
    (appends bypass provider-side validation); stats record the chunk calls."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(
        workspace,
        fid,
        page=1,
        words=[
            {"t": "a", "l": [10, 20, 30, 40]},
            {"t": "b", "l": [50, 20, 70, 40]},
            {"t": "$3", "l": [90, 20, 120, 40]},
        ],
    )
    store = DocSetStore(workspace)
    ds = store.create(name="Chunky")
    store.set_schema(ds.id, _CHUNK_RNC)
    store.add_file(ds.id, fid)

    first = {
        "Bill": {
            "Total": {"text": "$3", "value": "3", "locations": [{"page_number": 1}]},
            "Items": [{"Name": {"text": "a", "locations": [{"page_number": 1}]}}],
        }
    }
    responses = [
        # Chunked mode is reachable only after a truncated attempt — that is
        # what puts append_entries on the tool list in the first place.
        _truncated_response(),
        _tool_call_response("submit_values", {"values": first, "done": False}, call_id="c1"),
        _tool_call_response(
            "append_entries",
            {
                "path": "Bill.Items",
                "entries": [
                    {
                        "Name": {"text": "b", "locations": [{"page_number": 1}]},
                        "Bogus": {"text": "x"},
                    }
                ],
            },
            call_id="c2",
        ),
        _tool_call_response(
            "append_entries", {"path": "Bill.Items", "entries": [], "done": True}, call_id="c3"
        ),
    ]
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", side_effect=responses) as mock_completion:
        result = extract_values(workspace, ds.id, fid, config=config)

    assert mock_completion.call_count == 4  # truncated attempt + 3 chunked calls
    items = result.values["Bill"]["Items"]
    assert [e["Name"]["text"] for e in items] == ["a", "b"]
    assert "Bogus" not in items[1]  # pruned — appends bypass API-layer validation
    # Non-terminal calls were acknowledged with tool results.
    final_messages = mock_completion.call_args_list[3].kwargs["messages"]
    tool_msgs = [m for m in final_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert "recorded" in tool_msgs[0]["content"]

    stats = workspace.docs.get_doc("extraction_stats", f"{ds.id}/{fid}")
    assert stats is not None
    assert stats["phases"]["phase1"]["chunk_calls"] == 3
    assert stats["phases"]["phase1"]["truncated_retries"] == 1


def test_extract_values_append_error_lets_model_correct(workspace: Workspace) -> None:
    """A bad append path is answered with an error tool result (not a run
    failure); the model corrects and completes."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1, words=[{"t": "a", "l": [10, 20, 30, 40]}])
    store = DocSetStore(workspace)
    ds = store.create(name="Chunky")
    store.set_schema(ds.id, _CHUNK_RNC)
    store.add_file(ds.id, fid)

    first = {"Bill": {"Items": [{"Name": {"text": "a", "locations": [{"page_number": 1}]}}]}}
    responses = [
        _truncated_response(),  # enables chunked mode
        _tool_call_response("submit_values", {"values": first, "done": False}, call_id="c1"),
        _tool_call_response(
            "append_entries", {"path": "Wrong.Path", "entries": [], "done": True}, call_id="c2"
        ),
        _tool_call_response(
            "append_entries", {"path": "Bill.Items", "entries": [], "done": True}, call_id="c3"
        ),
    ]
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", side_effect=responses) as mock_completion:
        result = extract_values(workspace, ds.id, fid, config=config)
    assert mock_completion.call_count == 4
    # The bad-path call got an error tool result and did NOT end the run.
    final_messages = mock_completion.call_args_list[3].kwargs["messages"]
    errs = [m for m in final_messages if m.get("role") == "tool" and "error" in m["content"]]
    assert len(errs) == 1
    assert result.values["Bill"]["Items"][0]["Name"]["text"] == "a"


def test_extract_values_truncation_retries_with_chunk_directive(workspace: Workspace) -> None:
    """First attempt truncates (finish_reason='length'); the retry carries the
    mandatory chunking directive and succeeds chunked."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1, words=[{"t": "a", "l": [10, 20, 30, 40]}])
    store = DocSetStore(workspace)
    ds = store.create(name="Chunky")
    store.set_schema(ds.id, _CHUNK_RNC)
    store.add_file(ds.id, fid)

    call = SimpleNamespace(
        id="t1", function=SimpleNamespace(name="submit_values", arguments='{"values": {"Bi')
    )
    truncated = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[call]), finish_reason="length"
            )
        ]
    )
    first = {"Bill": {"Items": [{"Name": {"text": "a", "locations": [{"page_number": 1}]}}]}}
    responses = [
        truncated,
        _tool_call_response("submit_values", {"values": first, "done": False}, call_id="c1"),
        _tool_call_response(
            "append_entries", {"path": "Bill.Items", "entries": [], "done": True}, call_id="c2"
        ),
    ]
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", side_effect=responses) as mock_completion:
        result = extract_values(workspace, ds.id, fid, config=config)

    assert mock_completion.call_count == 3

    # First attempt: no directive. Retry: directive present in the user text.
    def user_text(call: Any) -> str:
        blocks = call.kwargs["messages"][1]["content"]
        return "\n".join(b["text"] for b in blocks if b.get("type") == "text")

    assert "MANDATORY CHUNKED SUBMISSION" not in user_text(mock_completion.call_args_list[0])
    assert "MANDATORY CHUNKED SUBMISSION" in user_text(mock_completion.call_args_list[1])
    # The directive rides its own block so the cached schema block is unchanged
    # between the two attempts — the retry can still read the entry the first
    # attempt wrote.
    first_blocks = mock_completion.call_args_list[0].kwargs["messages"][1]["content"]
    retry_blocks = mock_completion.call_args_list[1].kwargs["messages"][1]["content"]
    assert first_blocks[0]["text"] == retry_blocks[0]["text"]
    assert result.values["Bill"]["Items"][0]["Name"]["text"] == "a"

    stats = workspace.docs.get_doc("extraction_stats", f"{ds.id}/{fid}")
    assert stats is not None
    assert stats["phases"]["phase1"]["truncated_retries"] == 1
    assert stats["phases"]["phase1"]["chunk_calls"] == 2


def test_chunking_tools_are_offered_only_after_truncation(workspace: Workspace) -> None:
    """The chunked protocol is an escalation, not a standing option: a normal
    run must not be offered append_entries or a `done` flag it has no reason
    to use, and the retry must be offered both."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1, words=[{"t": "hi", "l": [10, 20, 30, 40]}])
    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _TITLE_RNC)
    store.add_file(ds.id, fid)

    values = {"title": {"text": "hi", "locations": [{"page_number": 1}]}}
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)

    def tool_names(call: Any) -> set[str]:
        return {t["function"]["name"] for t in call.kwargs["tools"]}

    def submit_params(call: Any) -> set[str]:
        tools = call.kwargs["tools"]
        submit = next(t for t in tools if t["function"]["name"] == "submit_values")
        return set(submit["function"]["parameters"]["properties"])

    # Ordinary run: single-shot tool set only.
    with patch(
        "litellm.completion",
        return_value=_tool_call_response("submit_values", {"values": values}),
    ) as mock_completion:
        extract_values(workspace, ds.id, fid, config=config)
    call = mock_completion.call_args_list[0]
    # NOTE: get_page_words is never offered — see the dead-dispatch finding in
    # _run_extract_loop; phase 1 currently exposes submit_values alone.
    assert tool_names(call) == {"submit_values"}
    assert submit_params(call) == {"values", "layout"}  # no `done`

    # After a truncation, the retry carries the chunked protocol.
    with patch(
        "litellm.completion",
        side_effect=[
            _truncated_response(),
            _tool_call_response("submit_values", {"values": values}),
        ],
    ) as mock_completion:
        extract_values(workspace, ds.id, fid, config=config)
    first, retry = mock_completion.call_args_list[0], mock_completion.call_args_list[1]
    assert "append_entries" not in tool_names(first)
    assert "append_entries" in tool_names(retry)
    assert "done" in submit_params(retry)


def test_append_entries_rejected_when_chunking_is_off(workspace: Workspace) -> None:
    """Gating the tool is the primary defense; the loop also refuses the call
    outright, so 'appends ran' always implies chunked mode (and therefore
    code-side pruning of the unvalidated batches)."""
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    store = DocSetStore(workspace)
    ds = store.create(name="Test")
    store.set_schema(ds.id, _TITLE_RNC)
    store.add_file(ds.id, fid)

    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch(
        "litellm.completion",
        return_value=_tool_call_response(
            "append_entries", {"path": "title", "entries": [], "done": True}
        ),
    ):
        with pytest.raises(ValuesExtractionFailed, match="unknown tool"):
            extract_values(workspace, ds.id, fid, config=config)


def _cache_control_paths(obj: Any, path: str = "") -> list[str]:
    """Every location a ``cache_control`` key appears at, for assertions."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "cache_control":
                found.append(path)
            else:
                found.extend(_cache_control_paths(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(_cache_control_paths(v, f"{path}[{i}]"))
    return found


def test_phase1_caches_by_default(workspace: Workspace) -> None:
    """Phase 1 marks its stable prefix with no configuration required.

    The system prompt and the schema block ahead of the per-file PDF are
    byte-identical for every file in a docset, so each file after the first
    reads that prefix. This is on unconditionally for Anthropic values models —
    the test pins that so a refactor cannot silently drop the marker.
    """
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)
    values = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}
    response = _tool_call_response("submit_values", {"values": values})
    config = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL, values_model="anthropic/claude-sonnet-4-6"
    )
    with patch("litellm.completion", return_value=response) as m:
        extract_values(workspace, ds_id, fid, config=config)
    # System message (tagged by the transport) plus the schema-text block.
    assert _cache_control_paths(m.call_args_list[0].kwargs["messages"]) == [
        "[0].content[0]",
        "[1].content[0]",
    ]


def test_phase1_schema_block_not_cached_off_anthropic(workspace: Workspace) -> None:
    """The phase-1 schema block is Anthropic-gated, like the transport's marker.

    ``cache_control`` is Anthropic-specific. ``call_with_tools`` guards its own
    system-message marker with ``is_anthropic_model``; the schema-text block in
    ``extract_values`` must do the same, or a Gemini values model would get an
    Anthropic-only key injected into its user content.
    """
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    _seed_page_text(workspace, fid, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)
    values = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}
    response = _tool_call_response("submit_values", {"values": values})

    gemini_cfg = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL, values_model="gemini/gemini-2.5-pro"
    )
    with patch("litellm.completion", return_value=response) as m:
        extract_values(workspace, ds_id, fid, config=gemini_cfg)
    assert _cache_control_paths(m.call_args_list[0].kwargs["messages"]) == []

    anthropic_cfg = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL, values_model="anthropic/claude-sonnet-4-6"
    )
    with patch("litellm.completion", return_value=response) as m:
        extract_values(workspace, ds_id, fid, config=anthropic_cfg)
    # System message (tagged by the transport) plus the schema-text block.
    assert _cache_control_paths(m.call_args_list[0].kwargs["messages"]) == [
        "[0].content[0]",
        "[1].content[0]",
    ]


def test_schema_gen_never_cached(workspace: Workspace) -> None:
    """Schema generation carries no cache marker.

    Its cacheable prefix (tools + a short system prompt) is below the provider's
    minimum cacheable length, so a breakpoint creates no entry — and schema-gen
    runs once per docset, so there is no reuse to capture regardless.
    """
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid)
    response = _tool_call_response("submit_schema", {"fields": _MIN_FIELDS})
    config = GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with patch("litellm.completion", return_value=response) as m:
        generate_schema(workspace, [fid], config=config, docset_name="D")
    assert _cache_control_paths(m.call_args_list[0].kwargs["messages"]) == []


def test_phase3_never_cached(workspace: Workspace) -> None:
    """Phase-3 page calls carry no cache marker.

    ``_submit_locations_tool`` is built from the page's own unmatched ids, so the
    tools+system prefix differs on every call and no read can ever match; the
    calls also run in parallel, and the system prompt alone is under the minimum
    cacheable prefix. A marker could only cost the write premium.
    """
    fid = "f1aaaaaaaaaa"
    _seed_file(workspace, fid, page_count=1)
    _seed_page_text(workspace, fid, page=1, words=[{"t": "Hello", "l": [0, 0, 10, 10]}])
    _seed_page_image(workspace, fid, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, fid)
    # A phase-1 value whose text will not match page_text, forcing phase 3.
    phase1_values = {"title": {"text": "Unmatchable zzzz", "locations": [{"page_number": 1}]}}
    config = GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL, values_model="anthropic/claude-sonnet-4-6"
    )
    with patch(
        "litellm.completion",
        side_effect=[
            _tool_call_response("submit_values", {"values": phase1_values}, call_id="p1"),
            _tool_call_response(
                "submit_locations",
                {"locations": [{"id": "a", "bounding_boxes": [[1.0, 2.0, 3.0, 4.0]]}]},
                call_id="p3",
            ),
        ],
    ) as m:
        extract_values(workspace, ds_id, fid, config=config)
    assert len(m.call_args_list) >= 2, "phase 3 did not run; test would be vacuous"
    for call in m.call_args_list[1:]:
        assert _cache_control_paths(call.kwargs["messages"]) == []
