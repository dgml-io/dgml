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

"""Value extraction as request generators: ``phase1_steps`` (the tool loop that
submits values) and ``phase3_page_steps`` (one page of box locating)."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest
from dgml_core import grounded
from dgml_core.errors import ValuesExtractionFailed
from dgml_core.extraction_schema import parse_rnc
from dgml_core.grounded import GroundedConfig, extract_values
from dgml_core.matching import UnmatchedItem, parse_path
from dgml_core.storage import Workspace
from dgml_core.usage import read_events

from .test_grounded import (
    _CHUNK_RNC,
    DEFAULT_VALUES_MODEL,
    _seed_docset_with_schema,
    _seed_file,
    _seed_page_image,
    _seed_page_text,
    _tool_call_response,
    _truncated_response,
)

FID = "f1aaaaaaaaaa"
SONNET = "anthropic/claude-sonnet-4-5"
EPHEMERAL = {"type": "ephemeral"}


def submit(values: dict[str, Any], **kw: Any) -> Any:
    return _tool_call_response("submit_values", {"values": values}, **kw)


def title(text: str) -> dict[str, Any]:
    return {"title": {"text": text, "locations": [{"page_number": 1}]}}


def tool_names(request: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in request["tools"]]


def run_extraction(ws: Workspace, capture_kwargs: Any, replies: list[Any], model: str) -> Any:
    _seed_file(ws, FID)
    _seed_page_text(ws, FID, page=1)  # "Hello", "world"
    _seed_page_image(ws, FID, 1)
    ds_id, _ = _seed_docset_with_schema(ws, FID)
    captured = capture_kwargs(replies)
    cfg = GroundedConfig(schema_model=model, values_model=model)
    extract_values(ws, ds_id, FID, config=cfg, record_usage=True, write_stats=True)
    return captured.kwargs


# -- requests, through the sync path -----------------------------------------


def test_gemini_phase1_request_requires_a_tool_with_no_cache_markers_and_medium_effort(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    (request,) = run_extraction(
        workspace, capture_kwargs, [submit(title("Hello world"))], DEFAULT_VALUES_MODEL
    )
    system, user = request["messages"]
    assert isinstance(system["content"], str)
    assert [block["type"] for block in user["content"]] == ["text", "file"]
    assert not any("cache_control" in block for block in user["content"])
    assert request["reasoning_effort"] == "medium"
    assert request["tool_choice"] == "required"
    assert tool_names(request) == ["submit_values"]


def test_anthropic_phase1_caches_system_and_schema_then_replays_tool_turns(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    page_words = _tool_call_response("get_page_words", {"page": 1}, call_id="w1")
    first, second = run_extraction(
        workspace, capture_kwargs, [page_words, submit(title("Hello world"))], SONNET
    )
    assert first["messages"][0]["content"][0]["cache_control"] == EPHEMERAL
    schema_text, document = first["messages"][1]["content"]
    assert schema_text["cache_control"] == EPHEMERAL
    assert "cache_control" not in document

    assistant, tool = second["messages"][2:]
    assert assistant["tool_calls"][0]["id"] == "w1"
    assert (tool["role"], tool["tool_call_id"], tool["name"]) == ("tool", "w1", "get_page_words")
    assert '"text": "Hello"' in tool["content"]


def test_phase3_page_request_forces_submit_locations_with_high_effort(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    boxes = {"locations": [{"id": "a", "bounding_boxes": [[100, 56, 200, 76]]}]}
    _, page = run_extraction(
        workspace,
        capture_kwargs,
        [submit(title("Goodnight")), _tool_call_response("submit_locations", boxes)],
        DEFAULT_VALUES_MODEL,
    )
    assert page["tool_choice"] == {"type": "function", "function": {"name": "submit_locations"}}
    assert page["reasoning_effort"] == "high"
    assert [block["type"] for block in page["messages"][1]["content"]] == ["text", "image_url"]


def test_phase1_and_phase3_share_one_usage_row(workspace: Workspace, capture_kwargs: Any) -> None:
    boxes = {"locations": [{"id": "a", "bounding_boxes": [[100, 56, 200, 76]]}]}
    run_extraction(
        workspace,
        capture_kwargs,
        [
            submit(title("Goodnight"), cost_usd=0.01, prompt_tokens=100, completion_tokens=50),
            _tool_call_response("submit_locations", boxes, cost_usd=0.02, prompt_tokens=200),
        ],
        DEFAULT_VALUES_MODEL,
    )
    (row,) = read_events(workspace)
    assert (row["operation"], row["outcome"], row["tier"]) == ("extract_values", "ok", "standard")
    assert (row["cost_usd"], row["prompt_tokens"]) == (0.01 + 0.02, 300)


# -- the generators, driven by hand ------------------------------------------


def phase1(ws: Workspace, run: grounded.Phase1Result, *, model: str = DEFAULT_VALUES_MODEL) -> Any:
    _seed_file(ws, FID)
    return grounded.phase1_steps(
        workspace=ws,
        file_id=FID,
        run=run,
        pdf_bytes=b"%PDF",
        messages_factory=lambda *, chunked, permissive, pdf, part: [
            {"role": "system", "content": "S"},
            {
                "role": "user",
                "content": [{"type": "text", "text": f"chunked={chunked} permissive={permissive}"}],
            },
        ],
        tool_schema={"type": "object"},
        vocab=parse_rnc(_CHUNK_RNC),
        model=model,
        api_key=None,
        api_base=None,
        max_tool_iters=5,
    )


def test_done_false_asks_again_with_the_acknowledgement(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = phase1(workspace, run)
    next(gen)
    chunk = {"Bill": {"Items": [{"Name": {"text": "a"}}]}}
    again = gen.send(_tool_call_response("submit_values", {"values": chunk, "done": False}))
    assert [m["role"] for m in again["messages"]] == ["system", "user", "assistant", "tool"]
    assert "recorded" in again["messages"][-1]["content"]

    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": {"Bill": {}}, "done": True}))
    assert run.chunk_calls == 2


def test_truncation_restarts_chunked_and_a_second_truncation_fails(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = phase1(workspace, run)
    next(gen)
    retry = gen.send(_truncated_response())
    assert retry["messages"][1]["content"][0]["text"] == "chunked=True permissive=False"
    assert tool_names(retry) == ["submit_values", "append_entries"]
    with pytest.raises(ValuesExtractionFailed, match="truncated too"):
        gen.send(_truncated_response())


TOO_MANY_STATES = RuntimeError("400 INVALID_ARGUMENT: too many states for serving")
BILL = {"Bill": {"Items": [{"Name": {"text": "a"}}]}}


def permissive_phase1(ws: Workspace, run: grounded.Phase1Result) -> Any:
    """A phase-1 generator on its permissive retry: the provider refused the
    inlined schema and the request with the permissive tool set is pending."""
    gen = phase1(ws, run)
    next(gen)
    gen.throw(TOO_MANY_STATES)
    return gen


def test_too_many_states_falls_back_to_the_permissive_schema(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = phase1(workspace, run)
    next(gen)
    retry = gen.throw(TOO_MANY_STATES)
    submit_tool = retry["tools"][0]["function"]
    values_param = submit_tool["parameters"]["properties"]["values"]
    assert values_param == grounded._PERMISSIVE_VALUES_PARAM
    # JSON text, not an open object: Gemini (the provider that trips this
    # fallback) can only leave an object with no declared keys empty.
    assert values_param["type"] == "string"
    # The factory is told, so the retry's prompt can carry the permissive block.
    assert retry["messages"][1]["content"][0]["text"] == "chunked=False permissive=True"
    with pytest.raises(ValuesExtractionFailed, match="too many states"):
        gen.throw(TOO_MANY_STATES)


GEMINI = "gemini/gemini-3.1-pro-preview"
# litellm's wording for a Gemini HTTP 500, as the attempt wraps it.
GEMINI_500 = RuntimeError(
    "litellm.InternalServerError: GeminiException InternalServerError - "
    '{"error": {"code": 500, "message": "An internal error has occurred. Please retry '
    'or report in https://developers.generativeai.google/guide/troubleshooting", '
    '"status": "INTERNAL"}}'
)


def test_a_gemini_500_on_the_inlined_schema_falls_back_to_the_permissive_one(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    """Gemini answers some oversized inlined schemas with a bare 500 rather
    than the 400 that names the overflow; on a Gemini model that takes the
    same permissive retry, once. A second 500 fails the file as itself. The
    warning names the cause in one line; the provider's JSON body is DEBUG."""
    run = grounded.Phase1Result()
    gen = phase1(workspace, run, model=GEMINI)
    next(gen)
    with caplog.at_level(logging.DEBUG, logger="dgml_core"):
        retry = gen.throw(GEMINI_500)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "HTTP 500 INTERNAL" in warnings[0]
    assert "\n" not in warnings[0] and "developers.generativeai.google" not in warnings[0]
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("developers.generativeai.google" in m for m in debug)
    values_param = retry["tools"][0]["function"]["parameters"]["properties"]["values"]
    assert values_param == grounded._PERMISSIVE_VALUES_PARAM
    assert retry["messages"][1]["content"][0]["text"] == "chunked=False permissive=True"
    assert run.tool_schema_mode == grounded.ToolSchemaMode.PERMISSIVE
    with pytest.raises(ValuesExtractionFailed, match="GeminiException InternalServerError"):
        gen.throw(GEMINI_500)


def test_a_500_from_another_provider_does_not_fall_back(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = phase1(workspace, run, model="openai/gpt-5.4")
    next(gen)
    with pytest.raises(ValuesExtractionFailed, match="InternalServerError"):
        gen.throw(RuntimeError("litellm.InternalServerError: OpenAIException InternalServerError"))
    assert run.tool_schema_mode == grounded.ToolSchemaMode.INLINED


def test_other_gemini_errors_do_not_fall_back(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = phase1(workspace, run, model=GEMINI)
    next(gen)
    with pytest.raises(ValuesExtractionFailed, match="ServiceUnavailable"):
        gen.throw(RuntimeError("litellm.ServiceUnavailableError: GeminiException - 503"))
    assert run.tool_schema_mode == grounded.ToolSchemaMode.INLINED


@pytest.mark.parametrize(
    ("model", "text", "expected"),
    [
        (GEMINI, str(GEMINI_500), True),
        ("vertex_ai/gemini-3.1-pro-preview", "VertexAIException InternalServerError - 500", True),
        ("anthropic/claude-sonnet-5", str(GEMINI_500), False),
        (GEMINI, "litellm.InternalServerError: OpenAIException InternalServerError", False),
        (GEMINI, "litellm.RateLimitError: GeminiException - 429", False),
        (GEMINI, "litellm.BadRequestError: GeminiException - too many states", False),
    ],
)
def test_is_gemini_internal_error(model: str, text: str, expected: bool) -> None:
    assert grounded._is_gemini_internal_error(RuntimeError(text), model) is expected


def test_an_empty_submission_logs_the_models_reason_at_info(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    """A model that finds nothing usually says why in text beside the tool
    call; the loop discards text, so that reason is logged at INFO with the
    empty submission (the run stays a legal nothing-found)."""
    run = grounded.Phase1Result()
    gen = phase1(workspace, run)
    next(gen)
    reply = _tool_call_response("submit_values", {"values": {}})
    reply.choices[0].message.content = (
        "This document is a Master Services Agreement, not a utility bill.\n"
        "None of the schema fields are applicable."
    )
    with caplog.at_level(logging.INFO, logger="dgml_core"), pytest.raises(StopIteration):
        gen.send(reply)
    assert run.values == {} and run.refusal is None
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any(
        FID in m and "submitted no values" in m and "not a utility bill. None of" in m
        for m in infos
    )


def test_an_empty_submission_without_text_is_still_logged(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    run = grounded.Phase1Result()
    gen = phase1(workspace, run)
    next(gen)
    with caplog.at_level(logging.INFO, logger="dgml_core"), pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": {"Bill": None}}))
    assert any("submitted no values and gave no reason" in r.getMessage() for r in caplog.records)


def test_a_non_empty_submission_logs_nothing_about_empty_values(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    run = grounded.Phase1Result()
    gen = phase1(workspace, run)
    next(gen)
    with caplog.at_level(logging.INFO, logger="dgml_core"), pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": BILL}))
    assert not any("submitted no values" in r.getMessage() for r in caplog.records)


def test_permissive_values_arrive_as_json_text_and_are_decoded(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = permissive_phase1(workspace, run)
    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": json.dumps(BILL)}))
    assert run.values == BILL
    assert run.refusal is None
    assert run.counters["envelope_repairs"] == 0


def test_permissive_values_text_wrapping_the_envelope_is_still_repaired(
    workspace: Workspace,
) -> None:
    """The text is decoded before the envelope repair reads it, so a tree
    the model wrapped in the tool's own envelope inside the string is
    unwrapped like a nested object would be, layout included."""
    run = grounded.Phase1Result()
    gen = permissive_phase1(workspace, run)
    layout = {"Items": {"kind": "table"}}
    text = json.dumps({"values": BILL, "layout": layout})
    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": text}))
    assert run.values == BILL
    assert run.layout == layout
    assert run.counters["envelope_repairs"] == 1


def test_permissive_values_text_that_is_not_an_object_is_a_correctable_error(
    workspace: Workspace,
) -> None:
    """The model hand-writes the tree inside a string on this path, so one
    slip is answered with an error tool result and a resend, not a failed
    file after a full round-trip of the PDF (as append_entries does)."""
    run = grounded.Phase1Result()
    gen = permissive_phase1(workspace, run)
    again = gen.send(_tool_call_response("submit_values", {"values": "Bill: a"}))
    reply = json.loads(again["messages"][-1]["content"])
    assert again["messages"][-1]["role"] == "tool"
    assert "JSON text of one object" in reply["error"]
    assert "Bill: a" in reply["error"]
    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": json.dumps(BILL)}))
    assert run.values == BILL
    assert run.refusal is None


def test_permissive_values_text_in_a_code_fence_is_decoded(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = permissive_phase1(workspace, run)
    fenced = "```json\n" + json.dumps(BILL, indent=2) + "\n```"
    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": fenced}))
    assert run.values == BILL
    assert run.counters["envelope_repairs"] == 0


@pytest.mark.parametrize(
    ("text", "inner"),
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}```', '{"a": 1}'),
        ('  ```JSON \r\n{"a":\n 1}\r\n```  ', '{"a":\n 1}'),
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}', '```json\n{"a": 1}'),
        ("```json\n```", ""),
    ],
)
def test_strip_code_fence(text: str, inner: str) -> None:
    assert grounded._strip_code_fence(text) == inner


def test_permissive_values_sent_as_an_object_anyway_are_taken_as_is(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = permissive_phase1(workspace, run)
    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": BILL}))
    assert run.values == BILL


def test_permissive_empty_tree_is_flagged_for_refusal(workspace: Workspace) -> None:
    """Under the inlined schema `{}` is a legal nothing-found. Under the
    permissive one it is also what a decoder that cannot fill the fallback
    parameter emits (how this path returned no values on Gemini), so the
    generator flags it and the caller refuses to write it as success."""
    run = grounded.Phase1Result()
    gen = permissive_phase1(workspace, run)
    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": "{}"}))
    assert run.values == {}
    assert run.refusal is not None
    assert "empty values tree under the permissive tool schema" in run.refusal
    assert DEFAULT_VALUES_MODEL in run.refusal


def test_permissive_all_null_roots_stay_a_legal_nothing_found(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = permissive_phase1(workspace, run)
    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": json.dumps({"Bill": None})}))
    assert run.refusal is None


def test_permissive_chunked_entries_arrive_as_json_text(workspace: Workspace) -> None:
    """A truncated permissive attempt restarts chunked with the permissive
    continuation tool: `entries` is JSON text of the array, decoded on the
    way in; text that is not an array is a correctable tool error."""
    run = grounded.Phase1Result()
    gen = permissive_phase1(workspace, run)
    retry = gen.send(_truncated_response())
    assert tool_names(retry) == ["submit_values", "append_entries"]
    append_tool = retry["tools"][1]["function"]
    assert append_tool["parameters"]["properties"]["entries"]["type"] == "string"

    first = {"values": json.dumps(BILL), "done": False}
    again = gen.send(_tool_call_response("submit_values", first))
    assert "recorded" in again["messages"][-1]["content"]

    bad = {"path": "Bill.Items", "entries": "[{oops", "done": True}
    again = gen.send(_tool_call_response("append_entries", bad))
    assert "JSON text of an array" in again["messages"][-1]["content"]

    more = {"path": "Bill.Items", "entries": json.dumps([{"Name": {"text": "b"}}]), "done": True}
    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("append_entries", more))
    assert [e["Name"]["text"] for e in run.values["Bill"]["Items"]] == ["a", "b"]
    assert run.tool_schema_mode == grounded.ToolSchemaMode.PERMISSIVE and run.chunked


def test_permissive_fallback_is_warned_once_per_model_then_logged_per_file(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="dgml_core"):
        for _ in range(2):
            permissive_phase1(workspace, grounded.Phase1Result())
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert DEFAULT_VALUES_MODEL in warnings[0]
    assert "too many states" in warnings[0]
    assert "permissive tool schema" in warnings[0]
    infos = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "permissive tool schema" in r.getMessage()
    ]
    assert len(infos) == 1 and FID in infos[0]


def test_a_nested_envelope_is_unwrapped_and_counted(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = phase1(workspace, run)
    next(gen)
    bill = {"Bill": {"Items": [{"Name": {"text": "a"}}]}}
    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", {"values": {"values": bill}}))
    assert run.values == bill
    assert run.counters == {"envelope_repairs": 1, "no_tool_call_retries": 0}
    assert run.refusal is None


def test_a_tree_that_fits_no_root_is_flagged_not_raised(workspace: Workspace) -> None:
    """The generator finishes and records the attempt; refusing is the
    caller's job, after it has counted the tool calls and kept the layout."""
    run = grounded.Phase1Result()
    gen = phase1(workspace, run)
    next(gen)
    layout = {"Invoice": {"kind": "free_form"}}
    reply = {"values": {"Invoice": {"text": "a"}}, "layout": layout}
    with pytest.raises(StopIteration):
        gen.send(_tool_call_response("submit_values", reply))
    assert run.layout == layout and run.counters == {
        "envelope_repairs": 0,
        "no_tool_call_retries": 0,
    }
    assert run.refusal is not None and "got keys 'Invoice'" in run.refusal


def test_sync_extraction_repairs_and_refuses_like_the_steps(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    """Through ``extract_values``: the repair lands in the stats, and the
    refused run's usage row still counts phase 1's ``get_page_words`` call."""
    from dgml_core import layout as ws_layout

    def stats(ds_id: str) -> Any:
        doc = workspace.docs.get_doc(
            ws_layout.Collection.EXTRACTION_STATS, ws_layout.pair_id(ds_id, FID)
        )
        assert doc is not None
        return doc

    nested = submit({"values": title("Hello world")})
    run_extraction(workspace, capture_kwargs, [nested], DEFAULT_VALUES_MODEL)
    ds_id = read_events(workspace)[0]["context"]["docset_id"]
    assert stats(ds_id)["phases"]["phase1"]["envelope_repairs"] == 1

    lookup = _tool_call_response("get_page_words", {"page": 1}, call_id="w1")
    capture_kwargs([lookup, submit({"Invoice": {"text": "x"}})])
    cfg = GroundedConfig(schema_model=DEFAULT_VALUES_MODEL, values_model=DEFAULT_VALUES_MODEL)
    with pytest.raises(ValuesExtractionFailed, match="nothing that fits the schema"):
        extract_values(workspace, ds_id, FID, config=cfg, record_usage=True, write_stats=True)
    row = read_events(workspace)[-1]
    assert (row["outcome"], row["context"]["tool_calls"]) == ("error", 1)
    assert stats(ds_id)["outcome"] == "error"


def phase3(ws: Workspace) -> Any:
    _seed_file(ws, FID)
    _seed_page_text(ws, FID, page=1)
    _seed_page_image(ws, FID, 1)
    path = parse_path("title")
    assert path is not None
    return grounded.phase3_page_steps(
        workspace=ws,
        file_id=FID,
        page_number=1,
        items=[UnmatchedItem(id="a", path=path, text="Goodnight", page_number=1)],
        values=title("Goodnight"),
        model=DEFAULT_VALUES_MODEL,
        api_key=None,
        api_base=None,
        max_tool_iters=3,
        totals=None,
        reasoning_effort="high",
    )


def test_thrown_errors_keep_their_sync_messages(workspace: Workspace) -> None:
    gen = phase1(workspace, grounded.Phase1Result())
    next(gen)
    with pytest.raises(
        ValuesExtractionFailed,
        match=(
            rf"^extraction call failed: RuntimeError: x \(model '{DEFAULT_VALUES_MODEL}', "
            r"api key not passed\)$"
        ),
    ):
        gen.throw(RuntimeError("x"))

    page = phase3(workspace)
    next(page)
    with pytest.raises(ValuesExtractionFailed, match=r"^phase 3 page 1 call failed: Runtime"):
        page.throw(RuntimeError("x"))


def test_a_page_reply_without_choices_is_a_page_failure(workspace: Workspace) -> None:
    page = phase3(workspace)
    next(page)
    with pytest.raises(ValuesExtractionFailed, match="phase 3 page 1 call failed"):
        page.send(SimpleNamespace(choices=[]))
