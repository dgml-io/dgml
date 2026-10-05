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

"""The ``steps_*`` request generators and the ``drive`` loop in ``dgml_core.llm``.

The sync wrappers (``call``, ``call_continued``, ``call_with_refinement``,
``call_with_tools``) now drive these generators. These tests pin the request
each one builds and the control flow around it.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from dgml_core import llm
from dgml_core.storage import Workspace
from dgml_core.usage import read_events

from .conftest import FakeLLMResponse

HAIKU = "anthropic/claude-haiku-4-5"
EPHEMERAL = {"type": "ephemeral"}
PDF = b"%PDF-1.4\n%%EOF"
TOOLS = [{"type": "function", "function": {"name": "submit", "parameters": {"type": "object"}}}]
FORCED = {"type": "function", "function": {"name": "submit"}}


def text(value: str = "U") -> list[dict[str, Any]]:
    return [{"type": "text", "text": value}]


def roles(request: dict[str, Any]) -> list[str]:
    return [m["role"] for m in request["messages"]]


def finish(gen: Generator[Any, Any, Any], response: Any) -> Any:
    with pytest.raises(StopIteration) as done:
        gen.send(response)
    return done.value.value


# -- request shape ------------------------------------------------------------


def test_call_marks_only_the_static_system_prefix_for_caching(capture_kwargs: Any) -> None:
    captured = capture_kwargs([FakeLLMResponse("hi")])
    cfg = llm.LLMConfig(model=HAIKU, temperature=0.0)
    out = llm.call(cfg, system_prompt=("STATIC", "DYNAMIC"), user_content=text(), cache=True)

    assert out == "hi"
    assert captured.kwargs == [
        {
            "model": HAIKU,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "STATIC", "cache_control": EPHEMERAL},
                        {"type": "text", "text": "DYNAMIC"},
                    ],
                },
                {"role": "user", "content": text()},
            ],
            "max_tokens": 16000,
        }
    ]


def test_call_on_openai_sends_no_cache_markers(capture_kwargs: Any) -> None:
    captured = capture_kwargs([FakeLLMResponse("hi")])
    cfg = llm.LLMConfig(model="gpt-4o", temperature=0.0, timeout=30.0)
    content = llm.build_user_content(instruction_text="read", pdf_bytes=PDF)
    llm.call(cfg, system_prompt="SYS", user_content=content, cache=True)

    (request,) = captured.kwargs
    assert request["messages"][0] == {"role": "system", "content": "SYS"}
    assert request["messages"][1]["content"][1]["type"] == "file"
    assert "cache_control" not in request["messages"][1]["content"][1]
    assert (request["temperature"], request["timeout"]) == (0.0, 30.0)


def test_continued_prefills_the_partial_on_anthropic(capture_kwargs: Any) -> None:
    captured = capture_kwargs(
        [FakeLLMResponse("part-1", finish_reason="length"), FakeLLMResponse("part-2")]
    )
    cfg = llm.LLMConfig(model=HAIKU, max_tokens=32000)
    content = llm.build_user_content(instruction_text="transcribe", pdf_bytes=PDF)
    out = llm.call_continued(cfg, system_prompt="SYS", user_content=content, cache=True)

    assert out == "part-1part-2"
    first, second = captured.kwargs
    assert first["messages"][0]["content"][0]["cache_control"] == EPHEMERAL
    assert first["messages"][1]["content"][-1]["cache_control"] == EPHEMERAL  # the document
    assert second["messages"] == [*first["messages"], {"role": "assistant", "content": "part-1"}]
    assert second["max_tokens"] == 32000


@pytest.mark.parametrize(
    "cfg",
    [
        llm.LLMConfig(model="openai/gpt-5.4"),
        llm.LLMConfig(model="gemini/gemini-2.5-pro"),
        # Extended thinking rejects prefill, so Anthropic asks explicitly too.
        llm.LLMConfig(model=HAIKU, reasoning_effort="medium"),
    ],
)
def test_continued_asks_in_a_user_turn_where_prefill_is_unsupported(
    capture_kwargs: Any, cfg: llm.LLMConfig
) -> None:
    captured = capture_kwargs(
        [FakeLLMResponse("part-1", finish_reason="length"), FakeLLMResponse("part-2")]
    )
    assert llm.call_continued(cfg, system_prompt="SYS", user_content=text()) == "part-1part-2"

    second = captured.kwargs[1]
    assert roles(second) == ["system", "user", "assistant", "user"]
    assert second["messages"][2]["content"] == "part-1"
    assert second["messages"][3]["content"].startswith("Your previous reply was cut off")
    assert second.get("reasoning_effort") == cfg.reasoning_effort


def test_continued_stops_after_max_rounds(capture_kwargs: Any) -> None:
    captured = capture_kwargs(lambda _kw: FakeLLMResponse("x", finish_reason="length"))
    cfg = llm.LLMConfig(model=HAIKU)
    assert llm.call_continued(cfg, system_prompt="S", user_content=text(), max_rounds=2) == "xx"
    assert len(captured.kwargs) == 2


def test_refinement_sends_the_draft_back_with_cached_system_and_listing(
    capture_kwargs: Any,
) -> None:
    captured = capture_kwargs([FakeLLMResponse("DRAFT"), FakeLLMResponse("REFINED")])
    out = llm.call_with_refinement(
        llm.LLMConfig(model="anthropic/claude-sonnet-4-6"),
        system_prompt="SYS",
        user_content=text("LISTING"),
        refine_instruction=text("complete it"),
        cache=True,
    )

    assert out == ("DRAFT", "REFINED")
    second = captured.kwargs[1]["messages"]
    assert second[0]["content"][0]["cache_control"] == EPHEMERAL
    assert second[1]["content"] == [{"type": "text", "text": "LISTING", "cache_control": EPHEMERAL}]
    assert second[2:] == [
        {"role": "assistant", "content": "DRAFT"},
        {"role": "user", "content": text("complete it")},
    ]


def test_forced_tool_choice_drops_reasoning_effort_on_anthropic(capture_kwargs: Any) -> None:
    captured = capture_kwargs([FakeLLMResponse("", tool_calls=[{"id": "c1"}])])
    cfg = llm.LLMConfig(model=HAIKU, max_tokens=None, reasoning_effort="high")
    result = llm.call_with_tools(
        cfg,
        messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "go"}],
        tools=TOOLS,
        tool_choice=FORCED,
        cache=True,
    )

    assert result.tool_calls == [{"id": "c1"}]
    (request,) = captured.kwargs
    assert request["tool_choice"] == FORCED
    assert "reasoning_effort" not in request
    assert request["messages"][0]["content"][0]["cache_control"] == EPHEMERAL


def test_auto_tool_choice_keeps_reasoning_effort(capture_kwargs: Any) -> None:
    captured = capture_kwargs([FakeLLMResponse("ok")])
    cfg = llm.LLMConfig(model="gpt-4o", max_tokens=None, reasoning_effort="high")
    llm.call_with_tools(cfg, messages=[{"role": "user", "content": "go"}], tools=TOOLS)

    (request,) = captured.kwargs
    assert request["reasoning_effort"] == "high"
    assert "tool_choice" not in request


def test_sync_wrapper_sends_exactly_what_its_generator_yields(capture_kwargs: Any) -> None:
    captured = capture_kwargs([FakeLLMResponse("D"), FakeLLMResponse("R")])
    cfg = llm.LLMConfig(model=HAIKU)
    kw: dict[str, Any] = dict(
        system_prompt="S", user_content=text("L"), refine_instruction=text("go"), cache=True
    )
    llm.call_with_refinement(cfg, **kw)

    gen = llm.steps_with_refinement(cfg, **kw)
    assert captured.kwargs == [next(gen), gen.send(FakeLLMResponse("D"))]


# -- usage rows ---------------------------------------------------------------


def debug_config(tmp_path: Path, **kw: Any) -> tuple[llm.LLMConfig, Workspace]:
    ws = Workspace(root=tmp_path)
    return llm.LLMConfig(model=HAIKU, workspace=ws, debug=True, operation="t", **kw), ws


def test_a_continued_call_writes_one_row_for_all_its_rounds(
    capture_kwargs: Any, tmp_path: Path
) -> None:
    capture_kwargs(
        [
            FakeLLMResponse(
                "a", finish_reason="length", cost=0.01, prompt_tokens=100, completion_tokens=50,
                cache_read_tokens=40,
            ),
            FakeLLMResponse("b", cost=0.02, prompt_tokens=150, completion_tokens=25),
        ]
    )  # fmt: skip
    cfg, ws = debug_config(tmp_path, context={"doc": "a.pdf"})
    llm.call_continued(cfg, system_prompt="S", user_content=text())

    (row,) = read_events(ws)
    assert isinstance(row.pop("at"), str) and isinstance(row.pop("duration_s"), float)
    assert row == {
        "operation": "t",
        "model": HAIKU,
        "cost_usd": 0.01 + 0.02,
        "prompt_tokens": 250,
        "completion_tokens": 75,
        "total_tokens": 325,
        "cache_read_tokens": 40,
        "cache_creation_tokens": 0,
        "outcome": "ok",
        "context": {"doc": "a.pdf"},
        "error": None,
        "tier": "standard",
    }


def test_a_failed_call_writes_an_error_row_with_the_spent_rounds(
    capture_kwargs: Any, tmp_path: Path
) -> None:
    replies = [FakeLLMResponse("a", finish_reason="length", cost=0.01, prompt_tokens=10)]

    def flaky(_kwargs: dict[str, Any]) -> Any:
        if not replies:
            raise RuntimeError("boom")
        return replies.pop()

    capture_kwargs(flaky)
    cfg, ws = debug_config(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        llm.call_continued(cfg, system_prompt="S", user_content=text())

    (row,) = read_events(ws)
    assert (row["outcome"], row["error"]) == ("error", "RuntimeError: boom")
    assert (row["cost_usd"], row["prompt_tokens"]) == (0.01, 10)


def test_usage_folds_per_wrapper_call_like_the_inline_wrappers(tmp_path: Path) -> None:
    """One ``call`` then a two-round ``call_continued`` in one scope: the row is
    0.1 + (0.2 + 0.3), not (0.1 + 0.2) + 0.3, which differ in the last digit."""
    cfg, ws = debug_config(tmp_path)

    def two_wrappers() -> Generator[dict[str, Any], Any, None]:
        yield from llm.steps_call(cfg, system_prompt="S", user_content=text())
        yield from llm.steps_continued(cfg, system_prompt="S", user_content=text())

    replies = [
        FakeLLMResponse("a", cost=0.1),
        FakeLLMResponse("b", finish_reason="length", cost=0.2),
        FakeLLMResponse("c", cost=0.3),
    ]
    llm.drive(two_wrappers(), cfg, lambda _step: replies.pop(0))
    (row,) = read_events(ws)
    assert row["cost_usd"] == 0.1 + (0.2 + 0.3) != (0.1 + 0.2) + 0.3


def test_hand_driven_generators_write_no_rows(tmp_path: Path) -> None:
    cfg, ws = debug_config(tmp_path)
    gen = llm.steps_call(cfg, system_prompt="S", user_content=text())
    next(gen)
    assert finish(gen, FakeLLMResponse("hi", cost=0.5)) == "hi"
    assert read_events(ws) == []


# -- errors thrown into the generator ----------------------------------------


def test_drive_throws_executor_errors_into_the_generator_which_may_retry() -> None:
    cfg = llm.LLMConfig(model="gpt-4o")
    caught: list[str] = []

    def retry_once() -> Generator[dict[str, Any], Any, str]:
        try:
            return (yield from llm.steps_call(cfg, system_prompt="S", user_content=text()))
        except ValueError as exc:
            caught.append(str(exc))
            return (yield from llm.steps_call(cfg, system_prompt="S", user_content=text()))

    outcomes: list[Any] = [ValueError("transient"), FakeLLMResponse("ok")]

    def execute(_step: dict[str, Any]) -> Any:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert llm.drive(retry_once(), cfg, execute) == "ok"
    assert caught == ["transient"]


def test_drive_reraises_an_unhandled_executor_error_and_closes_the_generator() -> None:
    import inspect

    cfg = llm.LLMConfig(model="gpt-4o")
    gen = llm.steps_call(cfg, system_prompt="S", user_content=text())

    def execute(_step: dict[str, Any]) -> Any:
        raise ValueError("no network")

    with pytest.raises(ValueError, match="no network"):
        llm.drive(gen, cfg, execute)
    assert inspect.getgeneratorstate(gen) == inspect.GEN_CLOSED


def test_a_reply_without_choices_raises_empty_model_response() -> None:
    from dgml_core.errors import EmptyModelResponse

    gen = llm.steps_call(llm.LLMConfig(model="gpt-4o"), system_prompt="S", user_content=text())
    next(gen)
    with pytest.raises(EmptyModelResponse, match="no choices"):
        gen.send({"choices": []})
