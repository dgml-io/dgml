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
    extract_values(ws, ds_id, FID, config=cfg, debug=True)
    return captured.kwargs


# -- requests, through the sync path -----------------------------------------


def test_gemini_phase1_request_has_no_cache_markers_and_medium_effort(
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
    assert "tool_choice" not in request
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
    assert (row["operation"], row["outcome"]) == ("extract_values", "ok")
    assert (row["cost_usd"], row["prompt_tokens"]) == (0.01 + 0.02, 300)


# -- the generators, driven by hand ------------------------------------------


def phase1(ws: Workspace, run: grounded.Phase1Result) -> Any:
    _seed_file(ws, FID)
    return grounded.phase1_steps(
        workspace=ws,
        file_id=FID,
        run=run,
        messages_factory=lambda *, chunked: [
            {"role": "system", "content": "S"},
            {"role": "user", "content": [{"type": "text", "text": f"chunked={chunked}"}]},
        ],
        tool_schema={"type": "object"},
        vocab=parse_rnc(_CHUNK_RNC),
        model=DEFAULT_VALUES_MODEL,
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
    assert retry["messages"][1]["content"][0]["text"] == "chunked=True"
    assert tool_names(retry) == ["submit_values", "append_entries"]
    with pytest.raises(ValuesExtractionFailed, match="truncated too"):
        gen.send(_truncated_response())


def test_too_many_states_falls_back_to_the_permissive_schema(workspace: Workspace) -> None:
    run = grounded.Phase1Result()
    gen = phase1(workspace, run)
    next(gen)
    refusal = RuntimeError("400 INVALID_ARGUMENT: too many states for serving")
    retry = gen.throw(refusal)
    submit_tool = retry["tools"][0]["function"]
    assert submit_tool["parameters"]["properties"]["values"] == grounded._PERMISSIVE_VALUES_PARAM
    with pytest.raises(ValuesExtractionFailed, match="too many states"):
        gen.throw(refusal)


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
    )


def test_thrown_errors_keep_their_sync_messages(workspace: Workspace) -> None:
    gen = phase1(workspace, grounded.Phase1Result())
    next(gen)
    with pytest.raises(ValuesExtractionFailed, match=r"^extraction call failed: RuntimeError: x$"):
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
