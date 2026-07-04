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
from dgml_core.classification import (
    DEFAULT_MAX_PAGES,
    ClassificationConfig,
    ClassificationDecision,
    classify_file,
    load_classification_config,
    propose_new_docset_for_files,
)
from dgml_core.docsets import DocSetStore
from dgml_core.errors import (
    AuthError,
    ClassificationConfigInvalid,
    ClassificationConfigMissing,
    ClassificationFailed,
)
from dgml_core.models import FileRecord
from dgml_core.storage import Workspace, write_json_atomic
from dgml_core.utils import gather_file_pages

from .conftest import write_classification_config

DEFAULT_TEST_MODEL = "gemini/gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_file(workspace: Workspace, file_id: str, *, filename: str = "doc.pdf") -> None:
    """Materialize a minimal File record on disk without ingesting a PDF.

    Most classification tests don't need real page rendering; they need the
    File record (so :class:`FileStore.get` works) and an optional page-images
    directory which the test populates explicitly.
    """
    record = FileRecord(
        id=file_id,
        original_path=f"/fake/{filename}",
        original_filename=filename,
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=1,
        text_mode="digital",
    )
    workspace.file_dir(file_id).mkdir(parents=True, exist_ok=True)
    write_json_atomic(workspace.file_json_path(file_id), record.to_json())


def _seed_page_image(workspace: Workspace, file_id: str, page: int, content: bytes) -> None:
    workspace.file_pages_dir(file_id).mkdir(parents=True, exist_ok=True)
    (workspace.file_pages_dir(file_id) / f"page_{page}.png").write_bytes(content)


def _tool_call_response(name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    """Build a litellm.completion response stub with one tool call.

    litellm returns OpenAI-compatible objects regardless of provider, so the
    attribute path ``response.choices[0].message.tool_calls[0].function`` is
    stable. SimpleNamespace gives attribute access without spec'ing a Mock.
    """
    call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))])


def _empty_tool_calls_response() -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))])


# ---------------------------------------------------------------------------
# load_classification_config
# ---------------------------------------------------------------------------


def test_load_config_missing_when_no_config_file(workspace: Workspace) -> None:
    with pytest.raises(ClassificationConfigMissing):
        load_classification_config(workspace)


def test_load_config_missing_when_no_classification_section(workspace: Workspace) -> None:
    workspace.config_path.write_text(json.dumps({"ocr": {}}), encoding="utf-8")
    with pytest.raises(ClassificationConfigMissing):
        load_classification_config(workspace)


def test_load_config_invalid_json(workspace: Workspace) -> None:
    workspace.config_path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(ClassificationConfigInvalid):
        load_classification_config(workspace)


def test_load_config_root_not_object(workspace: Workspace) -> None:
    workspace.config_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ClassificationConfigInvalid):
        load_classification_config(workspace)


def test_load_config_section_not_object(workspace: Workspace) -> None:
    workspace.config_path.write_text(
        json.dumps({"classification": "gemini/gemini-3.1-flash-lite"}), encoding="utf-8"
    )
    with pytest.raises(ClassificationConfigInvalid):
        load_classification_config(workspace)


def test_load_config_happy_minimal(workspace: Workspace) -> None:
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})
    cfg = load_classification_config(workspace)
    assert cfg.model == DEFAULT_TEST_MODEL
    assert cfg.max_pages == DEFAULT_MAX_PAGES  # default is 3
    assert cfg.api_key is None
    assert cfg.api_key_env is None


def test_load_config_literal_api_key(workspace: Workspace) -> None:
    write_classification_config(
        workspace,
        {"model": DEFAULT_TEST_MODEL, "api_key": "literal-test-key"},
    )
    cfg = load_classification_config(workspace)
    assert cfg.api_key == "literal-test-key"
    assert cfg.api_key_env is None


def test_load_config_rejects_both_api_key_and_env(workspace: Workspace) -> None:
    write_classification_config(
        workspace,
        {
            "model": DEFAULT_TEST_MODEL,
            "api_key": "literal",
            "api_key_env": "GEMINI_API_KEY",
        },
    )
    with pytest.raises(ClassificationConfigInvalid, match=r"api_key.*api_key_env"):
        load_classification_config(workspace)


def test_load_config_api_key_empty_rejected(workspace: Workspace) -> None:
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL, "api_key": ""})
    with pytest.raises(ClassificationConfigInvalid, match="api_key"):
        load_classification_config(workspace)


def test_load_config_happy_full(workspace: Workspace) -> None:
    write_classification_config(
        workspace,
        {
            "model": DEFAULT_TEST_MODEL,
            "max_pages": 5,
            "api_key_env": "GEMINI_API_KEY",
        },
    )
    cfg = load_classification_config(workspace)
    assert cfg.model == DEFAULT_TEST_MODEL
    assert cfg.max_pages == 5
    assert cfg.api_key_env == "GEMINI_API_KEY"


def test_load_config_missing_model(workspace: Workspace) -> None:
    write_classification_config(workspace, {"max_pages": 2})
    with pytest.raises(ClassificationConfigInvalid, match="model"):
        load_classification_config(workspace)


def test_load_config_empty_model(workspace: Workspace) -> None:
    write_classification_config(workspace, {"model": "  "})
    with pytest.raises(ClassificationConfigInvalid, match="model"):
        load_classification_config(workspace)


def test_load_config_max_pages_zero(workspace: Workspace) -> None:
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL, "max_pages": 0})
    with pytest.raises(ClassificationConfigInvalid, match="max_pages"):
        load_classification_config(workspace)


def test_load_config_max_pages_bool_rejected(workspace: Workspace) -> None:
    """Python's bool is a subclass of int; reject it explicitly so
    `"max_pages": true` doesn't silently mean 1.
    """
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL, "max_pages": True})
    with pytest.raises(ClassificationConfigInvalid, match="max_pages"):
        load_classification_config(workspace)


def test_load_config_api_key_env_empty_string(workspace: Workspace) -> None:
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL, "api_key_env": ""})
    with pytest.raises(ClassificationConfigInvalid, match="api_key_env"):
        load_classification_config(workspace)


# ---------------------------------------------------------------------------
# gather_file_pages
# ---------------------------------------------------------------------------


def test_gather_pages_empty_when_no_dir(workspace: Workspace) -> None:
    _seed_file(workspace, "abc123")
    assert gather_file_pages(workspace, "abc123", max_pages=3) == []


def test_gather_pages_respects_max(workspace: Workspace) -> None:
    _seed_file(workspace, "abc123")
    for i in range(1, 6):
        _seed_page_image(workspace, "abc123", i, f"page{i}".encode())
    pages = gather_file_pages(workspace, "abc123", max_pages=3)
    assert pages == [b"page1", b"page2", b"page3"]


def test_gather_pages_returns_all_when_fewer_than_max(workspace: Workspace) -> None:
    _seed_file(workspace, "abc123")
    _seed_page_image(workspace, "abc123", 1, b"only-page")
    assert gather_file_pages(workspace, "abc123", max_pages=10) == [b"only-page"]


# ---------------------------------------------------------------------------
# classify_file
# ---------------------------------------------------------------------------


def _seed_for_classify(workspace: Workspace) -> tuple[str, str]:
    """Common setup: one docset with one file (so the prompt has context),
    one new file ready to be classified. Returns (existing_docset_id, new_file_id).
    """
    docset = DocSetStore(workspace).create(
        name="Invoices",
        description="vendor invoices",
        key_questions=[
            "What is the vendor name?",
            "What is the invoice total?",
            "What is the invoice date?",
        ],
    )
    docset = DocSetStore(workspace).create(
        name="Invoices",
        description="vendor invoices",
        key_questions=[
            "What is the vendor name?",
            "What is the invoice total?",
            "What is the invoice date?",
        ],
    )
    _seed_file(workspace, "existingfid", filename="invoice-acme.pdf")
    DocSetStore(workspace).add_file(docset.id, "existingfid")

    _seed_file(workspace, "newfid", filename="incoming.pdf")
    _seed_page_image(workspace, "newfid", 1, b"\x89PNG\r\n\x1a\nfake-png")
    return docset.id, "newfid"


_DEFAULT_NEW_QUESTIONS = [
    "What is the PO number?",
    "What is the buyer's name?",
    "What is the order total?",
]


def _create_new_args(
    name: str = "Purchase Orders",
    description: str = "vendor POs",
    key_questions: list[str] | None = None,
) -> dict[str, Any]:
    chosen = key_questions if key_questions is not None else _DEFAULT_NEW_QUESTIONS
    return {
        "name": name,
        "description": description,
        "key_questions": list(chosen),
    }


def test_classify_file_existing_decision(workspace: Workspace) -> None:
    existing_id, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response("assign_to_existing_docset", {"docset_id": existing_id})

    with patch("litellm.completion", return_value=response) as mock_completion:
        decision = classify_file(workspace, new_id, config=cfg)

    assert decision == ClassificationDecision(decision="existing", existing_docset_id=existing_id)
    # api_key_env was unset → no api_key kwarg passed; litellm uses its own
    # per-provider env var lookup.
    call_kwargs = mock_completion.call_args.kwargs
    assert "api_key" not in call_kwargs
    assert call_kwargs["model"] == DEFAULT_TEST_MODEL
    assert call_kwargs["tool_choice"] == "required"


def test_classify_file_new_decision(workspace: Workspace) -> None:
    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response("create_new_docset", _create_new_args())

    with patch("litellm.completion", return_value=response):
        decision = classify_file(workspace, new_id, config=cfg)

    assert decision == ClassificationDecision(
        decision="new",
        new_name="Purchase Orders",
        new_description="vendor POs",
        new_key_questions=tuple(_DEFAULT_NEW_QUESTIONS),
    )


def test_classify_file_no_page_images(workspace: Workspace) -> None:
    _seed_file(workspace, "nopagesfid")  # no page_images directory
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    with patch("litellm.completion") as mock_completion:
        with pytest.raises(ClassificationFailed, match="no page images"):
            classify_file(workspace, "nopagesfid", config=cfg)
    mock_completion.assert_not_called()


def test_classify_file_provider_exception_wrapped(workspace: Workspace) -> None:
    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    with patch("litellm.completion", side_effect=RuntimeError("network boom")):
        with pytest.raises(ClassificationFailed, match="RuntimeError: network boom"):
            classify_file(workspace, new_id, config=cfg)


def test_classify_file_empty_tool_calls(workspace: Workspace) -> None:
    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    with patch("litellm.completion", return_value=_empty_tool_calls_response()):
        with pytest.raises(ClassificationFailed, match="no tool calls"):
            classify_file(workspace, new_id, config=cfg)


def test_classify_file_unknown_tool_name(workspace: Workspace) -> None:
    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response("delete_everything", {})
    with patch("litellm.completion", return_value=response):
        with pytest.raises(ClassificationFailed, match="unexpected tool name"):
            classify_file(workspace, new_id, config=cfg)


def test_classify_file_unknown_docset_id(workspace: Workspace) -> None:
    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response("assign_to_existing_docset", {"docset_id": "not-a-real-id"})
    with patch("litellm.completion", return_value=response):
        with pytest.raises(ClassificationFailed, match="unknown docset_id"):
            classify_file(workspace, new_id, config=cfg)


def test_classify_file_missing_required_arg(workspace: Workspace) -> None:
    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response("create_new_docset", {"name": "x"})  # no description
    with patch("litellm.completion", return_value=response):
        with pytest.raises(ClassificationFailed, match="description"):
            classify_file(workspace, new_id, config=cfg)


def test_classify_file_missing_key_questions_fails(workspace: Workspace) -> None:
    """create_new_docset must include key_questions — these define the
    DocSet for future classifications and aren't optional."""
    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response("create_new_docset", {"name": "x", "description": "y"})
    with patch("litellm.completion", return_value=response):
        with pytest.raises(ClassificationFailed, match="key_questions"):
            classify_file(workspace, new_id, config=cfg)


def test_classify_file_empty_key_questions_fails(workspace: Workspace) -> None:
    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response(
        "create_new_docset",
        {"name": "x", "description": "y", "key_questions": []},
    )
    with patch("litellm.completion", return_value=response):
        with pytest.raises(ClassificationFailed, match="key_questions"):
            classify_file(workspace, new_id, config=cfg)


def test_classify_file_key_questions_strips_blanks(workspace: Workspace) -> None:
    """Whitespace-only entries are silently dropped, but at least one
    non-empty question must remain or classification fails."""
    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response(
        "create_new_docset",
        {
            "name": "x",
            "description": "y",
            "key_questions": ["  ", "What is the date?", "  "],
        },
    )
    with patch("litellm.completion", return_value=response):
        decision = classify_file(workspace, new_id, config=cfg)
    assert decision.new_key_questions == ("What is the date?",)


def test_classify_file_prompt_lists_existing_key_questions(workspace: Workspace) -> None:
    """When existing DocSets have key_questions, the prompt must surface them
    so the LLM can apply the schema-shareability criterion."""
    existing_id, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response("assign_to_existing_docset", {"docset_id": existing_id})

    with patch("litellm.completion", return_value=response) as mock_completion:
        classify_file(workspace, new_id, config=cfg)

    content = mock_completion.call_args.kwargs["messages"][0]["content"]
    prompt_text = next(c["text"] for c in content if c["type"] == "text")
    # Each of the seeded key questions appears verbatim in the prompt.
    expected_qs = [
        "What is the vendor name?",
        "What is the invoice total?",
        "What is the invoice date?",
    ]
    for q in expected_qs:
        assert q in prompt_text
    # And the prompt frames the criterion in extraction-schema terms.
    assert "extraction schema" in prompt_text or "key questions" in prompt_text


def test_classify_file_malformed_json_arguments(workspace: Workspace) -> None:
    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    call = SimpleNamespace(
        function=SimpleNamespace(name="assign_to_existing_docset", arguments="{not json")
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))]
    )
    with patch("litellm.completion", return_value=response):
        with pytest.raises(ClassificationFailed, match="not valid JSON"):
            classify_file(workspace, new_id, config=cfg)


def test_classify_file_api_key_env_resolved(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, new_id = _seed_for_classify(workspace)
    monkeypatch.setenv("MY_CUSTOM_LLM_KEY", "sk-test-value")
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL, api_key_env="MY_CUSTOM_LLM_KEY")
    response = _tool_call_response("create_new_docset", _create_new_args())

    with patch("litellm.completion", return_value=response) as mock_completion:
        classify_file(workspace, new_id, config=cfg)

    assert mock_completion.call_args.kwargs["api_key"] == "sk-test-value"


def test_classify_file_api_key_env_unset_raises_auth_error(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, new_id = _seed_for_classify(workspace)
    monkeypatch.delenv("MY_CUSTOM_LLM_KEY", raising=False)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL, api_key_env="MY_CUSTOM_LLM_KEY")
    with patch("litellm.completion") as mock_completion:
        with pytest.raises(AuthError, match="MY_CUSTOM_LLM_KEY"):
            classify_file(workspace, new_id, config=cfg)
    mock_completion.assert_not_called()


def test_classify_file_records_usage_on_success(workspace: Workspace) -> None:
    from dgml_core.usage import read_events

    _, new_id = _seed_for_classify(workspace)
    response = _tool_call_response("create_new_docset", _create_new_args())
    response._hidden_params = {"response_cost": 0.0007}
    response.usage = SimpleNamespace(prompt_tokens=400, completion_tokens=30, total_tokens=430)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    with patch("litellm.completion", return_value=response):
        classify_file(workspace, new_id, config=cfg, debug=True)

    events = read_events(workspace)
    assert len(events) == 1
    e = events[0]
    assert e["operation"] == "classify"
    assert e["model"] == DEFAULT_TEST_MODEL
    assert e["cost_usd"] == 0.0007
    assert e["prompt_tokens"] == 400
    assert e["outcome"] == "ok"
    assert e["context"]["file_ids"] == [new_id]


def test_classify_file_records_usage_on_provider_exception(workspace: Workspace) -> None:
    from dgml_core.usage import read_events

    _, new_id = _seed_for_classify(workspace)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    with patch("litellm.completion", side_effect=RuntimeError("boom")):
        with pytest.raises(ClassificationFailed):
            classify_file(workspace, new_id, config=cfg, debug=True)
    events = read_events(workspace)
    assert len(events) == 1
    assert events[0]["outcome"] == "error"
    assert "boom" in (events[0]["error"] or "")


def test_classify_file_no_usage_recording_without_debug(workspace: Workspace) -> None:
    """Usage recording is gated on --debug: a normal (non-debug) classify
    writes no usage.jsonl row."""
    from dgml_core.usage import read_events

    _, new_id = _seed_for_classify(workspace)
    response = _tool_call_response("create_new_docset", _create_new_args())
    response._hidden_params = {"response_cost": 0.0007}
    response.usage = SimpleNamespace(prompt_tokens=400, completion_tokens=30, total_tokens=430)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    with patch("litellm.completion", return_value=response):
        classify_file(workspace, new_id, config=cfg)  # debug defaults False

    assert read_events(workspace) == []


def test_classify_file_literal_api_key_sent_directly(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A literal `api_key` is sent verbatim to litellm, bypassing
    os.environ entirely."""
    _, new_id = _seed_for_classify(workspace)
    # Make doubly sure: even if the env path were taken, this name isn't set.
    monkeypatch.delenv("ANY_NAME", raising=False)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL, api_key="sk-direct-literal")
    response = _tool_call_response("create_new_docset", _create_new_args())

    with patch("litellm.completion", return_value=response) as mock_completion:
        classify_file(workspace, new_id, config=cfg)

    assert mock_completion.call_args.kwargs["api_key"] == "sk-direct-literal"


def test_classify_file_no_existing_docsets_forces_new(workspace: Workspace) -> None:
    """When the workspace has no DocSets, the LLM must call create_new_docset.
    The prompt and tool schema still need to render correctly with an empty list.
    """
    _seed_file(workspace, "lonefid", filename="thing.pdf")
    _seed_page_image(workspace, "lonefid", 1, b"\xff\xd8\xff\xe0fake")
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response(
        "create_new_docset",
        _create_new_args(name="Standalone Things", description="one-off docs"),
    )

    with patch("litellm.completion", return_value=response):
        decision = classify_file(workspace, "lonefid", config=cfg)

    assert decision.decision == "new"
    assert decision.new_name == "Standalone Things"
    assert decision.new_key_questions == tuple(_DEFAULT_NEW_QUESTIONS)


# ---------------------------------------------------------------------------
# propose_new_docset_for_files
# ---------------------------------------------------------------------------


def test_propose_new_docset_returns_name_and_description(workspace: Workspace) -> None:
    _seed_file(workspace, "fid1", filename="po.pdf")
    _seed_page_image(workspace, "fid1", 1, b"\xff\xd8\xff\xe0fake")
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response(
        "create_new_docset",
        _create_new_args(name="Purchase Orders", description="vendor POs"),
    )

    with patch("litellm.completion", return_value=response) as mock_completion:
        decision = propose_new_docset_for_files(workspace, ["fid1"], config=cfg)

    assert decision == ClassificationDecision(
        decision="new",
        new_name="Purchase Orders",
        new_description="vendor POs",
        new_key_questions=tuple(_DEFAULT_NEW_QUESTIONS),
    )
    # Only the create-new tool is offered; the LLM is not given the assign tool.
    tools = mock_completion.call_args.kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "create_new_docset"
    assert mock_completion.call_args.kwargs["tool_choice"] == "required"


def test_propose_new_docset_aggregates_pages_across_files(workspace: Workspace) -> None:
    """When multiple files are passed, pages from each (up to ``max_pages`` per
    file) are bundled into one LLM call so the model sees the cluster as a
    whole."""
    _seed_file(workspace, "a")
    _seed_page_image(workspace, "a", 1, b"\xff\xd8\xff\xe0AAA")
    _seed_page_image(workspace, "a", 2, b"\xff\xd8\xff\xe0AAB")
    _seed_file(workspace, "b")
    _seed_page_image(workspace, "b", 1, b"\xff\xd8\xff\xe0BBA")
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL, max_pages=2)
    response = _tool_call_response(
        "create_new_docset",
        _create_new_args(name="Mixed Stuff", description="varied docs"),
    )

    with patch("litellm.completion", return_value=response) as mock_completion:
        propose_new_docset_for_files(workspace, ["a", "b"], config=cfg)

    content = mock_completion.call_args.kwargs["messages"][0]["content"]
    image_entries = [c for c in content if c["type"] == "image_url"]
    # 2 pages from file 'a' + 1 page from file 'b' = 3 total.
    assert len(image_entries) == 3


def test_propose_new_docset_strips_whitespace(workspace: Workspace) -> None:
    _seed_file(workspace, "fid2")
    _seed_page_image(workspace, "fid2", 1, b"\xff\xd8\xff\xe0fake")
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response(
        "create_new_docset",
        _create_new_args(name="  Padded Name  ", description="  padded desc  "),
    )
    with patch("litellm.completion", return_value=response):
        decision = propose_new_docset_for_files(workspace, ["fid2"], config=cfg)
    assert (decision.new_name, decision.new_description) == ("Padded Name", "padded desc")


def test_propose_new_docset_rejects_assign_tool(workspace: Workspace) -> None:
    """If the LLM somehow returns assign_to_existing_docset (it shouldn't, since
    that tool isn't offered), surface it as ClassificationFailed."""
    _seed_file(workspace, "fid3")
    _seed_page_image(workspace, "fid3", 1, b"\xff\xd8\xff\xe0fake")
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    response = _tool_call_response("assign_to_existing_docset", {"docset_id": "whatever"})
    with patch("litellm.completion", return_value=response):
        with pytest.raises(ClassificationFailed, match="unexpected tool name"):
            propose_new_docset_for_files(workspace, ["fid3"], config=cfg)


def test_propose_new_docset_no_page_images(workspace: Workspace) -> None:
    _seed_file(workspace, "nopagesfid")  # no page_images directory
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)
    with patch("litellm.completion") as mock_completion:
        with pytest.raises(ClassificationFailed, match="no page images"):
            propose_new_docset_for_files(workspace, ["nopagesfid"], config=cfg)
    mock_completion.assert_not_called()


def test_propose_new_docset_api_key_env_unset_raises_auth_error(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_file(workspace, "fid4")
    _seed_page_image(workspace, "fid4", 1, b"\xff\xd8\xff\xe0fake")
    monkeypatch.delenv("MY_CUSTOM_LLM_KEY", raising=False)
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL, api_key_env="MY_CUSTOM_LLM_KEY")
    with patch("litellm.completion") as mock_completion:
        with pytest.raises(AuthError, match="MY_CUSTOM_LLM_KEY"):
            propose_new_docset_for_files(workspace, ["fid4"], config=cfg)
    mock_completion.assert_not_called()
