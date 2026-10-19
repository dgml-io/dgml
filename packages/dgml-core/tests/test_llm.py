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

"""Tests for the `dgml_core.llm` call helpers."""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, cast

import httpx
import litellm
import pytest
from dgml_core import llm
from dgml_core.errors import EmptyModelResponse
from dgml_core.usage import extract_cost_and_tokens


@pytest.fixture(autouse=True)
def _deterministic_stream_cost(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Price streams locally because the sandbox cannot download LiteLLM's table."""
    calls: list[Any] = []

    def completion_cost(*, completion_response: Any, **_kwargs: Any) -> float:
        usage = getattr(completion_response, "usage", None)
        calls.append(usage)
        if usage is None:
            raise ValueError("usage is required")
        return float(usage.prompt_tokens * 1e-6 + usage.completion_tokens * 2e-6)

    monkeypatch.setattr("litellm.completion_cost", completion_cost)
    return calls


def _resp(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}]}


def test_litellm_debug_banner_suppressed() -> None:
    """Importing dgml.llm silences LiteLLM's stdout 'Give Feedback' banner.

    LiteLLM prints that banner to stdout on every exception map (including
    transient errors we retry), which would corrupt the JSON-on-stdout CLI
    contract. The module sets the flag at import time.
    """
    assert litellm.suppress_debug_info is True


def test_completion_with_retry_keeps_stdout_clean(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Anything a completion writes to stdout becomes a WARNING record on
    ``dgml_core.llm`` — stdout stays clean for the JSON payload, and nothing
    reaches a real stream unless the caller routes it (the CLI shows WARNINGs
    by default)."""

    def chatty_completion(**kwargs: Any) -> dict[str, Any]:
        print("LiteLLM noise on stdout")  # simulating the chatty dependency
        return _resp("OK")

    monkeypatch.setattr("litellm.completion", chatty_completion)

    with caplog.at_level(logging.WARNING, logger="dgml_core.llm"):
        result = llm._completion_with_retry({"model": "gpt-4o", "messages": []})

    assert result == _resp("OK")
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout stays clean for the JSON payload
    assert captured.err == ""  # no logging configured -> no stream output
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("LiteLLM noise on stdout" in r.getMessage() for r in warned)


def test_quiet_stdout_non_lifo_exits_restore_stdout(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Overlapping guards exiting out of order still restore the real stdout.

    LLM calls run in thread pools (grounded phase 3, parallel transcription,
    style page workers) and ``sys.stdout`` is process-global. The guard is
    refcounted — first in saves the real stdout, last out restores it — where
    a naive per-call ``redirect_stdout`` would re-install a dead sink on the
    second exit and swallow every later print for the rest of the process."""
    real = sys.stdout
    a_in, b_in, a_out = threading.Event(), threading.Event(), threading.Event()

    def thread_a() -> None:
        with llm._quiet_stdout():
            a_in.set()
            assert b_in.wait(5)
            print("captured while both guards active")
        a_out.set()  # A exits FIRST — non-LIFO

    def thread_b() -> None:
        assert a_in.wait(5)
        with llm._quiet_stdout():
            b_in.set()
            assert a_out.wait(5)

    with caplog.at_level(logging.WARNING, logger="dgml_core.llm"):
        ta, tb = threading.Thread(target=thread_a), threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join(10)
        tb.join(10)

    assert sys.stdout is real  # the whole point: not a stale sink
    assert any("captured while both guards active" in r.getMessage() for r in caplog.records)


def test_quiet_stdout_only_captures_the_calling_thread(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Concurrent completions never capture another thread's stdout.

    ``sys.stdout`` is process-global. While any pool thread sat inside a
    completion, a process-wide swap also captured the main thread, so the
    CLI's final JSON could become a WARNING record instead of reaching stdout.
    Only the calling thread's writes may be diverted."""
    from concurrent.futures import ThreadPoolExecutor

    workers = 4
    inside = threading.Barrier(workers + 1)
    printed = threading.Event()

    def chatty_completion(**kwargs: Any) -> dict[str, Any]:
        print("worker noise")
        inside.wait(timeout=5)  # every worker is mid-call ...
        printed.wait(timeout=5)  # ... while the main thread writes
        print("more worker noise")
        return _resp("OK")

    monkeypatch.setattr("litellm.completion", chatty_completion)
    original = sys.stdout

    with caplog.at_level(logging.WARNING, logger="dgml_core.llm"):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(llm._completion_with_retry, {"model": "gpt-4o", "messages": []})
                for _ in range(workers)
            ]
            inside.wait(timeout=5)
            print('{"ok": true}')
            printed.set()
            results = [f.result(timeout=5) for f in futures]
        print("after")

    assert results == [_resp("OK")] * workers
    assert capsys.readouterr().out == '{"ok": true}\nafter\n'
    messages = [r.getMessage() for r in caplog.records]
    assert sum("worker noise" in m for m in messages) == 2 * workers
    assert not any('{"ok": true}' in m for m in messages)
    assert sys.stdout is original


def test_quiet_stdout_survives_a_handler_that_writes_to_stdout(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Pathological embedder config: a handler that resolves ``sys.stdout`` at
    emit time writes INTO the sink while the sink is logging. The sink's lock
    is a leaf lock (released before emitting) and lines carrying the capture
    marker are dropped on any thread, so this must neither deadlock nor echo
    unboundedly — and the original stray line is still recorded exactly once."""

    class _StdoutAtEmit(logging.StreamHandler):  # type: ignore[type-arg]
        @property
        def stream(self) -> Any:
            return sys.stdout

        @stream.setter
        def stream(self, _value: Any) -> None:
            pass

    log = logging.getLogger("dgml_core.llm")
    handler = _StdoutAtEmit()
    handler.setLevel(logging.WARNING)
    log.addHandler(handler)
    done = threading.Event()

    def scenario() -> None:
        with llm._quiet_stdout():
            print("stray line")
        done.set()

    try:
        with caplog.at_level(logging.WARNING, logger="dgml_core.llm"):
            worker = threading.Thread(target=scenario, daemon=True)
            worker.start()
            assert done.wait(10), "deadlock: sink lock held while a handler wrote back into it"
            worker.join(10)
    finally:
        log.removeHandler(handler)

    hits = [r for r in caplog.records if "stray line" in r.getMessage()]
    assert len(hits) == 1  # the real line once; the handler's echo dropped


def test_stray_stdout_partial_line_is_flushed_as_a_record(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A write with no trailing newline still surfaces: the sink flushes its
    buffer when the guarded call exits, so nothing a dependency wrote is lost."""

    def chatty_completion(**kwargs: Any) -> dict[str, Any]:
        sys.stdout.write("partial line, no newline")
        return _resp("OK")

    monkeypatch.setattr("litellm.completion", chatty_completion)

    with caplog.at_level(logging.WARNING, logger="dgml_core.llm"):
        llm._completion_with_retry({"model": "gpt-4o", "messages": []})

    assert any("partial line, no newline" in r.getMessage() for r in caplog.records)


def test_completion_with_retry_redirects_only_during_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The redirect is scoped to the call and restores sys.stdout afterward."""
    original = sys.stdout
    monkeypatch.setattr("litellm.completion", lambda **kwargs: _resp("OK"))

    llm._completion_with_retry({"model": "gpt-4o", "messages": []})

    assert sys.stdout is original


def test_completion_with_retry_retries_empty_choices_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful response with an empty `choices` list (an intermittent
    Gemini glitch) is retried rather than passed through to a downstream
    `choices[0]` IndexError; the first non-empty response is returned."""
    from types import SimpleNamespace

    calls = {"n": 0}

    def flaky_completion(**kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] < 3:
            return SimpleNamespace(choices=[])  # empty candidate — transient
        return SimpleNamespace(choices=[SimpleNamespace(message="ok")])

    monkeypatch.setattr("litellm.completion", flaky_completion)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)  # no backoff wait

    result = llm._completion_with_retry({"model": "gemini/gemini-2.5-pro", "messages": []})

    assert calls["n"] == 3
    assert result.choices[0].message == "ok"


def test_completion_with_retry_raises_on_persistent_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every attempt comes back empty, a clear EmptyModelResponse is raised
    (carrying the model id) instead of a bare IndexError."""
    from types import SimpleNamespace

    from dgml_core.errors import EmptyModelResponse

    calls = {"n": 0}

    def always_empty(**kwargs: Any) -> Any:
        calls["n"] += 1
        return SimpleNamespace(choices=[])

    monkeypatch.setattr("litellm.completion", always_empty)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    with pytest.raises(EmptyModelResponse, match="no choices"):
        llm._completion_with_retry(
            {"model": "gemini/gemini-2.5-pro", "messages": []}, max_retries=3
        )

    assert calls["n"] == 3  # exhausted every attempt before raising


def test_completion_with_retry_retries_dict_shaped_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The empty-choices guard also fires for dict-shaped responses ({"choices": []}),
    not just objects — exercises the isinstance(response, dict) fallback on the empty path."""
    calls = {"n": 0}

    def flaky(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        return {"choices": []} if calls["n"] < 2 else _resp("OK")

    monkeypatch.setattr("litellm.completion", flaky)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    result = llm._completion_with_retry({"model": "gemini/gemini-2.5-pro", "messages": []})

    assert calls["n"] == 2
    assert result == _resp("OK")


def test_completion_with_retry_retries_transient_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient network error is retried and then succeeds — guards the
    `continue` in the except branch (without it, `response` is unbound and the
    empty-choices guard would raise UnboundLocalError)."""
    calls = {"n": 0}

    def flaky(**kwargs: Any) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("503 Service Unavailable: overloaded")
        return _resp("OK")

    monkeypatch.setattr("litellm.completion", flaky)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    result = llm._completion_with_retry({"model": "gpt-4o", "messages": []})

    assert calls["n"] == 2
    assert result == _resp("OK")


def test_call_with_refinement_replays_draft_in_second_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request 1 drafts; request 2 replays (user, assistant=draft, refine)."""
    seen: list[list[dict[str, Any]]] = []
    replies = iter([_resp("DRAFT"), _resp("REFINED")])

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs["messages"])
        return next(replies)

    monkeypatch.setattr("litellm.completion", fake_completion)

    draft, refined = llm.call_with_refinement(
        llm.LLMConfig(model="anthropic/claude-haiku-4-5"),
        system_prompt="SYS",
        user_content=[{"type": "text", "text": "LISTING"}],
        refine_instruction=[{"type": "text", "text": "complete it"}],
    )

    assert (draft, refined) == ("DRAFT", "REFINED")
    assert len(seen) == 2
    # Request 1: system + the listing.
    assert [m["role"] for m in seen[0]] == ["system", "user"]
    # Request 2: same prefix, then the model's own draft, then the refine ask.
    assert [m["role"] for m in seen[1]] == ["system", "user", "assistant", "user"]
    assert seen[1][2]["content"] == "DRAFT"
    assert seen[1][3]["content"] == [{"type": "text", "text": "complete it"}]


def test_call_with_refinement_marks_cache_on_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cache=True tags the system prefix and the last user block for Anthropic."""
    seen: list[list[dict[str, Any]]] = []
    replies = iter([_resp("D"), _resp("R")])

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs["messages"])
        return next(replies)

    monkeypatch.setattr("litellm.completion", fake_completion)

    llm.call_with_refinement(
        llm.LLMConfig(model="anthropic/claude-haiku-4-5"),
        system_prompt="SYS",
        user_content=[{"type": "text", "text": "LISTING"}],
        refine_instruction=[{"type": "text", "text": "complete it"}],
        cache=True,
    )

    sys_msg, user_msg = seen[0][0], seen[0][1]
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert user_msg["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_call_with_refinement_no_cache_markers_off_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Anthropic providers get plain content even with cache=True."""
    seen: list[list[dict[str, Any]]] = []
    replies = iter([_resp("D"), _resp("R")])

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs["messages"])
        return next(replies)

    monkeypatch.setattr("litellm.completion", fake_completion)

    llm.call_with_refinement(
        llm.LLMConfig(model="gpt-4o"),
        system_prompt="SYS",
        user_content=[{"type": "text", "text": "LISTING"}],
        refine_instruction=[{"type": "text", "text": "complete it"}],
        cache=True,
    )

    assert seen[0][0]["content"] == "SYS"  # plain string, no cache blocks
    assert "cache_control" not in seen[0][1]["content"][-1]


def _tool_resp() -> Any:
    """Attribute-accessible fake response for call_with_tools (no tool call)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ]
    )


class _FixtureStream(list[Any]):
    """Yielded chunks plus LiteLLM-shaped raw provider chunks."""

    def __init__(self, yielded: list[Any], raw: list[Any]) -> None:
        super().__init__(yielded)
        self.chunks = raw


def _tool_stream(*, content: str = "working", include_usage: bool = True) -> _FixtureStream:
    """Real LiteLLM chunks for one text delta and one fragmented tool call."""
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponseStream,
        StreamingChoices,
        Usage,
    )

    chunks = [
        ModelResponseStream(
            id="stream-1",
            model="gpt-4o",
            choices=[StreamingChoices(index=0, delta=Delta(content=content))],
        ),
        ModelResponseStream(
            id="stream-1",
            model="gpt-4o",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=0,
                                id="call-1",
                                type="function",
                                function=Function(name="submit_values"),
                            )
                        ]
                    ),
                )
            ],
        ),
        ModelResponseStream(
            id="stream-1",
            model="gpt-4o",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=0,
                                function=Function(arguments='{"values":'),
                            )
                        ]
                    ),
                )
            ],
        ),
        ModelResponseStream(
            id="stream-1",
            model="gpt-4o",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=0,
                                function=Function(arguments='{"title":"hello"}}'),
                            )
                        ]
                    ),
                    finish_reason="tool_calls",
                )
            ],
        ),
    ]
    raw_chunks = [chunk.model_copy(deep=True) for chunk in chunks]
    if include_usage:
        cast(Any, raw_chunks[-1]).usage = Usage(
            prompt_tokens=21, completion_tokens=8, total_tokens=29
        )
    return _FixtureStream(chunks, raw_chunks)


def _finished_empty_stream(
    finish_reason: str, *, refusal: str | None = None, include_usage: bool = True
) -> _FixtureStream:
    """One empty-content choice that still represents a completed model turn."""
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

    chunk = ModelResponseStream(
        id="finished-stream",
        model="gpt-4o",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(content=None, refusal=refusal),
                finish_reason=finish_reason,
            )
        ],
    )
    raw = chunk.model_copy(deep=True)
    if include_usage:
        cast(Any, raw).usage = Usage(prompt_tokens=90, completion_tokens=32, total_tokens=122)
    return _FixtureStream([chunk], [raw])


def test_stream_call_accepts_a_complete_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A layer that ignores ``stream=True`` (a test double, a proxy) returns a
    complete response; it is used as it is, never iterated. A litellm
    ModelResponse is iterable (pydantic fields) and is excluded by type."""
    import litellm as _litellm

    complete = _resp("done")
    monkeypatch.setattr("litellm.completion", lambda **kwargs: complete)
    out = llm._completion_with_retry({"model": "gpt-4o", "stream": True, "messages": []})
    assert out is complete
    assert llm._is_stream(complete) is False
    assert llm._is_stream(_litellm.ModelResponse(choices=[])) is False
    assert llm._is_stream(iter(())) is True
    assert llm._is_stream({"choices": []}) is False


@pytest.mark.parametrize(
    ("model", "api_base", "stream", "expected_options"),
    [
        ("gpt-4o", None, True, {"include_usage": True}),
        ("azure/gpt-4o", None, True, None),
        ("openai/x", "https://example.test/v1", True, None),
        ("anthropic/claude-haiku-4-5", None, True, None),
        ("gemini/gemini-2.5-pro", None, True, None),
        ("gpt-4o", None, False, None),
    ],
)
def test_build_completion_kwargs_stream_fields(
    model: str,
    api_base: str | None,
    stream: bool,
    expected_options: dict[str, bool] | None,
) -> None:
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model=model, api_base=api_base, stream=stream),
        messages=[{"role": "user", "content": "hi"}],
    )

    assert kwargs.get("stream") is (True if stream else None)
    if expected_options is None:
        assert "stream_options" not in kwargs
    else:
        assert kwargs["stream_options"] == expected_options


@pytest.mark.parametrize(
    ("config", "expected_stream", "expected_timeout"),
    [
        (llm.LLMConfig(model="gpt-4o", timeout=1800), True, 600.0),
        (llm.LLMConfig(model="gpt-4o", timeout=60), False, 60),
        (llm.LLMConfig(model="gpt-4o", stream=True), True, 600.0),
        (llm.LLMConfig(model="gpt-4o", timeout=1800, stream=False), False, 1800),
    ],
)
def test_build_completion_kwargs_stream_default_and_timeout(
    config: llm.LLMConfig, expected_stream: bool, expected_timeout: float
) -> None:
    kwargs = llm._build_completion_kwargs(config, messages=[{"role": "user", "content": "hi"}])

    assert kwargs.get("stream") is (True if expected_stream else None)
    assert kwargs["timeout"] == expected_timeout


def test_extra_stream_false_controls_all_derived_fields() -> None:
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="gpt-4o", timeout=1800, extra={"stream": False}),
        messages=[{"role": "user", "content": "hi"}],
    )

    assert "stream" not in kwargs
    assert "stream_options" not in kwargs
    assert kwargs["timeout"] == 1800


def test_extra_stream_options_survive_for_custom_api_base() -> None:
    options = {"include_usage": True}
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(
            model="openai/my-model",
            api_base="https://example.test/v1",
            stream=True,
            extra={"stream_options": options},
        ),
        messages=[{"role": "user", "content": "hi"}],
    )

    assert kwargs["stream_options"] == options


def test_call_with_tools_rebuilds_streamed_text_tool_call_usage_and_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    monkeypatch.setattr("litellm.completion", lambda **_kwargs: _tool_stream())

    result = llm.call_with_tools(
        llm.LLMConfig(model="gpt-4o", stream=True),
        messages=[{"role": "user", "content": "extract"}],
        tools=[{"type": "function", "function": {"name": "submit_values"}}],
    )

    assert result.content == "working"
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "submit_values"
    assert json.loads(tool_call.function.arguments) == {"values": {"title": "hello"}}
    assert result.usage["prompt_tokens"] == 21
    assert result.usage["completion_tokens"] == 8
    assert result.usage["total_tokens"] == 29
    assert result.usage["cost_usd"] == pytest.approx(0.000037)


def test_stream_without_usage_does_not_estimate_tokens_from_messages(
    monkeypatch: pytest.MonkeyPatch,
    _deterministic_stream_cost: list[Any],
) -> None:
    monkeypatch.setattr(
        "litellm.completion",
        lambda **_kwargs: _tool_stream(include_usage=False),
    )

    result = llm.call_with_tools(
        llm.LLMConfig(model="gpt-4o", stream=True),
        messages=[{"role": "user", "content": "do not count this prompt"}],
        tools=[{"type": "function", "function": {"name": "submit_values"}}],
    )

    assert result.usage["prompt_tokens"] is None
    assert result.usage["completion_tokens"] is None
    assert result.usage["total_tokens"] is None
    assert result.usage["cost_usd"] is None
    assert _deterministic_stream_cost == []


def test_consume_stream_uses_raw_chunk_usage_and_cache_tokens(
    _deterministic_stream_cost: list[Any],
) -> None:
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

    chunk = ModelResponseStream(
        id="anthropic-stream",
        model="anthropic/claude-sonnet-4-5",
        choices=[StreamingChoices(index=0, delta=Delta(content="done"))],
    )
    cast(Any, chunk).usage = Usage(
        prompt_tokens=1200,
        completion_tokens=300,
        total_tokens=1500,
        cache_read_input_tokens=1000,
    )

    rebuilt = llm._consume_stream(iter([chunk]), model="anthropic/claude-sonnet-4-5")

    assert rebuilt is not None
    usage = extract_cost_and_tokens(rebuilt)
    assert usage["prompt_tokens"] == 1200
    assert usage["completion_tokens"] == 300
    assert usage["total_tokens"] == 1500
    assert usage["cache_read_tokens"] == 1000
    assert usage["cost_usd"] == pytest.approx(0.0018)
    assert len(_deterministic_stream_cost) == 1


def test_consume_stream_rejects_zero_raw_usage(
    _deterministic_stream_cost: list[Any],
) -> None:
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage

    chunk = ModelResponseStream(
        id="mock-stream",
        model="anthropic/claude-sonnet-4-5",
        choices=[StreamingChoices(index=0, delta=Delta(content="hello"))],
    )
    cast(Any, chunk).usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    rebuilt = llm._consume_stream(iter([chunk]), model="anthropic/claude-sonnet-4-5")

    assert rebuilt is not None
    usage = extract_cost_and_tokens(rebuilt)
    assert usage["prompt_tokens"] is None
    assert usage["completion_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["cost_usd"] is None
    assert _deterministic_stream_cost == []


def test_consume_stream_converts_mapping_usage_for_extraction(
    _deterministic_stream_cost: list[Any],
) -> None:
    yielded = _tool_stream(include_usage=False)
    raw = [chunk.model_copy(deep=True) for chunk in yielded]
    cast(Any, raw[-1]).usage = {
        "prompt_tokens": 31,
        "completion_tokens": 4,
        "total_tokens": 35,
        "cache_read_input_tokens": 20,
    }
    stream = _FixtureStream(list(yielded), raw)

    rebuilt = llm._consume_stream(stream, model="gpt-4o")

    assert rebuilt is not None
    assert extract_cost_and_tokens(rebuilt) == {
        "cost_usd": pytest.approx(0.000039),
        "prompt_tokens": 31,
        "completion_tokens": 4,
        "total_tokens": 35,
        "cache_read_tokens": 20,
        "cache_creation_tokens": 0,
    }
    assert len(_deterministic_stream_cost) == 1


def _sse_body(frames: list[dict[str, Any]], *, event_types: bool = False) -> bytes:
    import json

    if event_types:
        return "".join(
            f"event: {frame['type']}\ndata: {json.dumps(frame)}\n\n" for frame in frames
        ).encode()
    return (
        "".join(f"data: {json.dumps(frame)}\r\n\r\n" for frame in frames).encode()
        + b"data: [DONE]\r\n\r\n"
    )


def _openai_mock_response(*, usage: dict[str, int] | None) -> Any:
    import openai

    base = {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    frames: list[dict[str, Any]] = [
        dict(
            base,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hello"},
                    "finish_reason": None,
                }
            ],
        ),
        dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}]),
    ]
    if usage is not None:
        frames.append(dict(base, choices=[], usage=usage))
    body = _sse_body(frames)
    client = openai.OpenAI(
        api_key="k",
        base_url="https://example.test/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, content=body
                )
            )
        ),
    )
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(
            model="openai/my-model",
            api_base="https://example.test/v1",
            api_key="k",
            stream=True,
            extra={"client": client, "stream_options": {"include_usage": True}},
        ),
        messages=[{"role": "user", "content": "word " * 5000}],
    )
    return llm._completion_with_retry(kwargs)


def test_openai_compatible_stream_without_provider_usage_is_unknown(
    _deterministic_stream_cost: list[Any],
) -> None:
    rebuilt = _openai_mock_response(usage=None)

    usage = extract_cost_and_tokens(rebuilt)
    assert usage["prompt_tokens"] is None
    assert usage["completion_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["cost_usd"] is None
    assert _deterministic_stream_cost == []


def test_openai_compatible_stream_uses_provider_usage(
    _deterministic_stream_cost: list[Any],
) -> None:
    rebuilt = _openai_mock_response(
        usage={"prompt_tokens": 77, "completion_tokens": 2, "total_tokens": 79}
    )

    usage = extract_cost_and_tokens(rebuilt)
    assert usage["prompt_tokens"] == 77
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 79
    assert usage["cost_usd"] == pytest.approx(0.000081)
    assert len(_deterministic_stream_cost) == 1


def test_gemini_stream_keeps_provider_cache_usage(
    _deterministic_stream_cost: list[Any],
) -> None:
    import json

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    usage_metadata = {
        "promptTokenCount": 100000,
        "candidatesTokenCount": 40,
        "totalTokenCount": 100340,
        "thoughtsTokenCount": 300,
        "cachedContentTokenCount": 90000,
    }
    frames: list[dict[str, Any]] = [
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "hello"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": usage_metadata,
        }
    ]
    body = "".join(f"data: {json.dumps(frame)}\r\n\r\n" for frame in frames).encode()
    client = HTTPHandler(
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, content=body
                )
            )
        )
    )

    rebuilt = llm._completion_with_retry(
        {
            "model": "gemini/gemini-2.5-pro",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
            "client": client,
            "api_key": "k",
        }
    )

    usage = extract_cost_and_tokens(rebuilt)
    assert usage["prompt_tokens"] == 100000
    assert usage["completion_tokens"] == 340
    assert usage["total_tokens"] == 100340
    assert usage["cache_read_tokens"] == 90000
    assert len(_deterministic_stream_cost) == 1


def test_anthropic_stream_keeps_provider_usage_and_cache_detail(
    _deterministic_stream_cost: list[Any],
) -> None:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    frames: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [],
                "stop_reason": None,
                "usage": {
                    "input_tokens": 5000,
                    "output_tokens": 1,
                    "cache_read_input_tokens": 20000,
                    "cache_creation_input_tokens": 0,
                },
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hello"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ]
    body = _sse_body(frames, event_types=True)
    client = HTTPHandler(
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, content=body
                )
            )
        )
    )

    rebuilt = llm._completion_with_retry(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
            "client": client,
            "api_key": "k",
        }
    )

    usage = extract_cost_and_tokens(rebuilt)
    assert usage["prompt_tokens"] == 25000
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 25007
    assert usage["cache_read_tokens"] == 20000
    assert len(_deterministic_stream_cost) == 1


def test_consume_real_litellm_mock_stream_has_no_fabricated_usage(
    _deterministic_stream_cost: list[Any],
) -> None:
    stream = litellm.completion(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "user", "content": "say hello"}],
        mock_response="hello there",
        stream=True,
    )

    rebuilt = llm._consume_stream(stream, model="anthropic/claude-sonnet-4-5")

    assert rebuilt is not None
    assert rebuilt.choices[0].message.content == "hello there"
    usage = extract_cost_and_tokens(rebuilt)
    assert usage["prompt_tokens"] is None
    assert usage["completion_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["cost_usd"] is None
    assert _deterministic_stream_cost == []


def test_consume_stream_normalizes_gemini_parallel_tool_call_indices() -> None:
    import json

    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponseStream,
        StreamingChoices,
    )

    def chunk(
        *, call_id: str | None = None, name: str | None = None, arguments: str | None = None
    ) -> Any:
        return ModelResponseStream(
            id="gemini-stream",
            model="gemini/gemini-2.5-pro",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                index=0,
                                id=call_id,
                                type="function",
                                function=Function(name=name, arguments=arguments),
                            )
                        ]
                    ),
                )
            ],
        )

    chunks = [
        chunk(call_id="call-words", name="get_page_words"),
        chunk(arguments='{"page"'),
        chunk(arguments=": 2}"),
        chunk(call_id="call-submit", name="submit_values"),
        chunk(arguments='{"values": {"title"'),
        chunk(arguments=': "hello"}}'),
    ]
    rebuilt = llm._consume_stream(iter(chunks), model="gemini/gemini-2.5-pro")

    assert rebuilt is not None
    tool_calls = rebuilt.choices[0].message.tool_calls
    assert [call.function.name for call in tool_calls] == ["get_page_words", "submit_values"]
    assert json.loads(tool_calls[0].function.arguments) == {"page": 2}
    assert json.loads(tool_calls[1].function.arguments) == {"values": {"title": "hello"}}


def _tool_delta_chunk(
    *,
    provider_index: int,
    choice_index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> Any:
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponseStream,
        StreamingChoices,
    )

    return ModelResponseStream(
        id="tool-stream",
        model="gpt-4o",
        choices=[
            StreamingChoices(
                index=choice_index,
                delta=Delta(
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            index=provider_index,
                            id=call_id,
                            type="function",
                            function=Function(name=name, arguments=arguments),
                        )
                    ]
                ),
            )
        ],
    )


def test_consume_stream_preserves_interleaved_provider_indices() -> None:
    chunks = [
        _tool_delta_chunk(provider_index=0, call_id="call-a", name="first"),
        _tool_delta_chunk(provider_index=1, call_id="call-b", name="second"),
        _tool_delta_chunk(provider_index=0, arguments='{"a":'),
        _tool_delta_chunk(provider_index=1, arguments='{"b":'),
        _tool_delta_chunk(provider_index=0, arguments="1}"),
        _tool_delta_chunk(provider_index=1, arguments="2}"),
    ]

    rebuilt = llm._consume_stream(iter(chunks), model="gpt-4o")

    assert rebuilt is not None
    calls = rebuilt.choices[0].message.tool_calls
    assert [(call.id, call.function.name) for call in calls] == [
        ("call-a", "first"),
        ("call-b", "second"),
    ]
    assert [call.function.arguments for call in calls] == ['{"a":1}', '{"b":2}']


def test_consume_stream_repeated_id_continues_same_call() -> None:
    chunks = [
        _tool_delta_chunk(provider_index=0, call_id="call-a", name="first"),
        _tool_delta_chunk(provider_index=0, call_id="call-a", arguments='{"a":1}'),
    ]

    rebuilt = llm._consume_stream(iter(chunks), model="gpt-4o")

    assert rebuilt is not None
    calls = rebuilt.choices[0].message.tool_calls
    assert len(calls) == 1
    assert calls[0].function.name == "first"
    assert calls[0].function.arguments == '{"a":1}'


def test_consume_stream_id_only_then_name_only_is_one_call() -> None:
    chunks = [
        _tool_delta_chunk(provider_index=0, call_id="call-a"),
        _tool_delta_chunk(provider_index=0, name="first"),
        _tool_delta_chunk(provider_index=0, arguments="{}"),
    ]

    rebuilt = llm._consume_stream(iter(chunks), model="gpt-4o")

    assert rebuilt is not None
    calls = rebuilt.choices[0].message.tool_calls
    assert len(calls) == 1
    assert calls[0].id == "call-a"
    assert calls[0].function.name == "first"
    assert calls[0].function.arguments == "{}"


def test_consume_stream_name_then_id_adopts_one_call() -> None:
    chunks = [
        _tool_delta_chunk(provider_index=0, name="first"),
        _tool_delta_chunk(provider_index=0, call_id="call-a", arguments="{}"),
    ]

    rebuilt = llm._consume_stream(iter(chunks), model="gpt-4o")

    assert rebuilt is not None
    calls = rebuilt.choices[0].message.tool_calls
    assert len(calls) == 1
    assert calls[0].id == "call-a"
    assert calls[0].function.name == "first"
    assert calls[0].function.arguments == "{}"


def test_consume_stream_joins_fragmented_name_at_one_provider_index() -> None:
    chunks = [
        _tool_delta_chunk(provider_index=0, call_id="call-a"),
        _tool_delta_chunk(provider_index=0, name="submit_"),
        _tool_delta_chunk(provider_index=0, name="values", arguments="{}"),
    ]

    rebuilt = llm._consume_stream(iter(chunks), model="gpt-4o")

    assert rebuilt is not None
    calls = rebuilt.choices[0].message.tool_calls
    assert len(calls) == 1
    assert calls[0].function.name == "submit_values"
    assert calls[0].function.arguments == "{}"


def test_consume_stream_keeps_provider_indices_separate_by_choice() -> None:
    chunks = [
        _tool_delta_chunk(provider_index=0, choice_index=0, name="first", arguments="{}"),
        _tool_delta_chunk(provider_index=0, choice_index=1, name="second", arguments="{}"),
    ]

    llm._normalize_tool_call_indices(chunks)

    assert chunks[0].choices[0].delta.tool_calls[0].index == 0
    assert chunks[1].choices[0].delta.tool_calls[0].index == 0


def test_consume_stream_preserves_anthropic_reasoning_delta() -> None:
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    chunk = ModelResponseStream(
        id="anthropic-stream",
        model="anthropic/claude-sonnet-4-6",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(reasoning_content="careful reasoning", content="answer"),
            )
        ],
    )
    rebuilt = llm._consume_stream(iter([chunk]), model="anthropic/claude-sonnet-4-6")

    assert rebuilt is not None
    assert rebuilt.choices[0].message.reasoning_content == "careful reasoning"


def test_stream_disconnect_during_iteration_is_retried(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = 0

    def broken_stream() -> Any:
        chunks = _tool_stream(content="first")
        yield chunks[0]
        yield chunks[1]
        raise httpx.ReadTimeout("")

    def completion(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return broken_stream() if calls == 1 else _tool_stream(content="second")

    monkeypatch.setattr("litellm.completion", completion)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    result = llm.call_with_tools(
        llm.LLMConfig(model="gpt-4o", stream=True),
        messages=[{"role": "user", "content": "extract"}],
        tools=[{"type": "function", "function": {"name": "submit_values"}}],
    )

    assert calls == 2
    assert result.content == "second"
    assert any(
        "stream aborted for model 'gpt-4o' on attempt 1 after 2 chunks" in record.getMessage()
        and "ReadTimeout" in record.getMessage()
        and "no usage recorded; retrying" in record.getMessage()
        for record in caplog.records
    )


def test_stream_disconnect_attempts_provider_iterator_close_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class ProviderIterator:
        closed = False

        def close(self) -> None:
            self.closed = True

    provider = ProviderIterator()

    class BrokenStream:
        completion_stream = provider

        def __iter__(self) -> Any:
            yield _tool_stream(include_usage=False)[0]
            raise httpx.ReadTimeout("")

    def completion(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return BrokenStream() if calls == 1 else _tool_stream(content="second")

    monkeypatch.setattr("litellm.completion", completion)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    rebuilt = llm._completion_with_retry({"model": "gpt-4o", "messages": [], "stream": True})

    assert rebuilt.choices[0].message.content == "second"
    assert provider.closed is True
    assert calls == 2


def test_stream_disconnect_retries_when_provider_iterator_has_no_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class ProviderIterator:
        pass

    class BrokenStream:
        completion_stream = ProviderIterator()

        def __iter__(self) -> Any:
            yield _tool_stream(include_usage=False)[0]
            raise httpx.ReadTimeout("")

    def completion(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return BrokenStream() if calls == 1 else _tool_stream(content="second")

    monkeypatch.setattr("litellm.completion", completion)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    rebuilt = llm._completion_with_retry({"model": "gpt-4o", "messages": [], "stream": True})

    assert rebuilt.choices[0].message.content == "second"
    assert calls == 2


def test_bare_timeout_from_unstreamed_call_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def completion(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError()
        return _resp("done")

    monkeypatch.setattr("litellm.completion", completion)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    assert llm._completion_with_retry({"model": "gpt-4o", "messages": []}) == _resp("done")
    assert calls == 2


def test_non_transient_value_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def completion(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise ValueError("bad")

    monkeypatch.setattr("litellm.completion", completion)

    with pytest.raises(ValueError, match="bad"):
        llm._completion_with_retry({"model": "gpt-4o", "messages": []})
    assert calls == 1


def test_timeout_wrapping_stream_deadline_is_not_transient() -> None:
    try:
        raise llm.StreamDeadlineExceeded("deadline")
    except llm.StreamDeadlineExceeded as deadline:
        wrapped = TimeoutError("outer timeout")
        wrapped.__cause__ = deadline

    assert llm._is_transient(wrapped) is False


def test_stream_total_deadline_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = 0
    now = 0.0

    def completion(**_kwargs: Any) -> Any:
        nonlocal calls, now
        calls += 1
        now = 2.0
        return iter(_tool_stream(include_usage=False)[:1])

    monkeypatch.setattr("litellm.completion", completion)
    monkeypatch.setattr("time.monotonic", lambda: now)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(TimeoutError, match=r"stream exceeded the 1 s deadline after 1 chunks"):
        llm._completion_with_retry(
            {"model": "gpt-4o", "messages": [], "stream": True},
            stream_deadline_s=1,
        )
    assert calls == 1
    assert any(
        "StreamDeadlineExceeded" in record.getMessage() and record.getMessage().endswith("aborting")
        for record in caplog.records
    )


def test_zero_chunk_stream_checks_deadline_after_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    now = 0.0

    def completion(**_kwargs: Any) -> Any:
        nonlocal calls, now
        calls += 1
        now = 2.0
        return iter(())

    monkeypatch.setattr("litellm.completion", completion)
    monkeypatch.setattr("time.monotonic", lambda: now)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(llm.StreamDeadlineExceeded, match="after 0 chunks"):
        llm._completion_with_retry(
            {"model": "gpt-4o", "messages": [], "stream": True},
            stream_deadline_s=1,
        )

    assert calls == 1


def test_read_timeout_after_stream_deadline_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    now = 0.0

    def stalled_stream() -> Any:
        nonlocal now
        now = 2.0
        raise httpx.ReadTimeout("stalled")
        yield

    def completion(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return stalled_stream()

    monkeypatch.setattr("litellm.completion", completion)
    monkeypatch.setattr("time.monotonic", lambda: now)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(llm.StreamDeadlineExceeded, match="after 0 chunks") as exc_info:
        llm._completion_with_retry(
            {"model": "gpt-4o", "messages": [], "stream": True},
            stream_deadline_s=1,
        )

    assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)
    assert calls == 1


def test_stream_options_rejection_retries_same_attempt_without_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> Any:
        seen.append(kwargs)
        if len(seen) == 1:
            raise Exception("Unrecognized request argument supplied: stream_options")
        return _tool_stream()

    monkeypatch.setattr("litellm.completion", completion)

    result = llm._completion_with_retry(
        {
            "model": "gpt-4o",
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        max_retries=1,
    )

    assert result.choices[0].message.content == "working"
    assert "stream_options" in seen[0]
    assert "stream_options" not in seen[1]


def test_final_stream_abort_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken_stream() -> Any:
        yield _tool_stream(include_usage=False)[0]
        raise httpx.RemoteProtocolError("")

    monkeypatch.setattr("litellm.completion", lambda **_kwargs: broken_stream())
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(httpx.RemoteProtocolError):
        llm._completion_with_retry(
            {"model": "gpt-4o", "messages": [], "stream": True}, max_retries=2
        )

    messages = [
        record.getMessage() for record in caplog.records if "stream aborted" in record.getMessage()
    ]
    assert len(messages) == 2
    assert "attempt 2 after 1 chunks" in messages[-1]
    assert "RemoteProtocolError" in messages[-1]
    assert messages[-1].endswith("aborting")


def test_empty_stream_retries_then_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def completion(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return iter(())

    monkeypatch.setattr("litellm.completion", completion)
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(EmptyModelResponse, match="model returned no choices after 3 attempts"):
        llm._completion_with_retry({"model": "gpt-4o", "messages": [], "stream": True})

    assert calls == 3


def test_streamed_refusal_with_finish_reason_is_returned() -> None:
    rebuilt = llm._consume_stream(
        _finished_empty_stream("stop", refusal="cannot comply"), model="gpt-4o"
    )

    assert rebuilt is not None
    assert rebuilt.choices[0].finish_reason == "stop"
    assert rebuilt.choices[0].message.refusal == "cannot comply"


def test_empty_content_length_choice_is_returned_once_with_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def completion(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return _finished_empty_stream("length")

    monkeypatch.setattr("litellm.completion", completion)

    result = llm.call_with_tools(
        llm.LLMConfig(model="gpt-4o", stream=True),
        messages=[{"role": "user", "content": "extract"}],
        tools=[{"type": "function", "function": {"name": "submit_values"}}],
    )

    assert calls == 1
    assert result.content == ""
    assert result.finish_reason == "length"
    assert result.usage["prompt_tokens"] == 90
    assert result.usage["completion_tokens"] == 32
    assert result.usage["total_tokens"] == 122


def test_empty_content_stop_choice_is_returned() -> None:
    rebuilt = llm._consume_stream(_finished_empty_stream("stop"), model="gpt-4o")

    assert rebuilt is not None
    assert rebuilt.choices[0].message.content == ""
    assert rebuilt.choices[0].finish_reason == "stop"


def test_gemini_usage_only_stream_retries_then_raises_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    requests = 0
    body = (
        "data: "
        + json.dumps({"usageMetadata": {"promptTokenCount": 500, "totalTokenCount": 500}})
        + "\r\n\r\n"
    ).encode()

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(EmptyModelResponse, match="model returned no choices after 3 attempts"):
        llm._completion_with_retry(
            {
                "model": "gemini/gemini-2.5-pro",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
                "client": client,
                "api_key": "k",
            }
        )

    assert requests == 3


def _extraction_messages() -> list[dict[str, Any]]:
    """A phase-1-shaped tool-call message list: string system + [schema, PDF]."""
    return [
        {"role": "system", "content": "SYS"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "SCHEMA"},
                {"type": "file", "file": {"file_data": "data:application/pdf;base64,AAA"}},
            ],
        },
    ]


def test_call_with_tools_marks_system_cache_on_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cache=True tags the system message for Anthropic; volatile user content
    (the per-file PDF) is left untouched."""
    seen: list[list[dict[str, Any]]] = []

    def fake_completion(**kwargs: Any) -> Any:
        seen.append(kwargs["messages"])
        return _tool_resp()

    monkeypatch.setattr("litellm.completion", fake_completion)

    llm.call_with_tools(
        llm.LLMConfig(model="anthropic/claude-haiku-4-5"),
        messages=_extraction_messages(),
        tools=[{"type": "function", "function": {"name": "t"}}],
        cache=True,
    )

    sys_msg, user_msg = seen[0][0], seen[0][1]
    # System message is wrapped into a cacheable text block.
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
    # The per-file PDF block is never tagged by call_with_tools.
    assert "cache_control" not in user_msg["content"][-1]


def test_call_with_tools_preserves_caller_marked_stable_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stable user block the caller pre-tagged (e.g. the docset schema text)
    passes through untouched alongside the system marker."""
    seen: list[list[dict[str, Any]]] = []

    def fake_completion(**kwargs: Any) -> Any:
        seen.append(kwargs["messages"])
        return _tool_resp()

    monkeypatch.setattr("litellm.completion", fake_completion)

    messages = _extraction_messages()
    messages[1]["content"][0]["cache_control"] = {"type": "ephemeral"}  # schema text

    llm.call_with_tools(
        llm.LLMConfig(model="anthropic/claude-haiku-4-5"),
        messages=messages,
        tools=[{"type": "function", "function": {"name": "t"}}],
        cache=True,
    )

    user_msg = seen[0][1]
    assert seen[0][0]["content"][0]["cache_control"] == {"type": "ephemeral"}  # system
    assert user_msg["content"][0]["cache_control"] == {"type": "ephemeral"}  # schema kept
    assert "cache_control" not in user_msg["content"][-1]  # PDF still untagged


def test_call_with_tools_no_cache_markers_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cache=False leaves the system message as a plain string (uncached path)."""
    seen: list[list[dict[str, Any]]] = []

    def fake_completion(**kwargs: Any) -> Any:
        seen.append(kwargs["messages"])
        return _tool_resp()

    monkeypatch.setattr("litellm.completion", fake_completion)

    llm.call_with_tools(
        llm.LLMConfig(model="anthropic/claude-haiku-4-5"),
        messages=_extraction_messages(),
        tools=[{"type": "function", "function": {"name": "t"}}],
        cache=False,
    )

    assert seen[0][0]["content"] == "SYS"  # untouched plain string
    assert "cache_control" not in seen[0][1]["content"][0]


def test_call_with_tools_no_cache_markers_off_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-Anthropic providers get plain content even with cache=True (implicit
    provider caching)."""
    seen: list[list[dict[str, Any]]] = []

    def fake_completion(**kwargs: Any) -> Any:
        seen.append(kwargs["messages"])
        return _tool_resp()

    monkeypatch.setattr("litellm.completion", fake_completion)

    llm.call_with_tools(
        llm.LLMConfig(model="gpt-4o"),
        messages=_extraction_messages(),
        tools=[{"type": "function", "function": {"name": "t"}}],
        cache=True,
    )

    assert seen[0][0]["content"] == "SYS"  # plain string, no cache blocks


def _obj_resp(content: str, finish: str) -> Any:
    """Attribute-accessible fake response (call_continued reads .choices[*])."""
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)]
    )


def test_call_continued_stitches_length_truncations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A length-truncated reply is continued via assistant prefill and stitched.

    No real LLM call — litellm.completion is mocked.
    """
    import json

    chunks = [
        ('{"continues": "", "blocks": [{"structure": "p", "text": "a"}', "length"),
        (', {"structure": "p", "text": "b"}]}', "stop"),
    ]
    seen: list[list[dict[str, Any]]] = []

    def fake_completion(**kwargs: Any) -> Any:
        seen.append(kwargs["messages"])
        return _obj_resp(*chunks[len(seen) - 1])

    monkeypatch.setattr("litellm.completion", fake_completion)

    out = llm.call_continued(
        llm.LLMConfig(model="anthropic/claude-haiku-4-5"),
        system_prompt="SYS",
        user_content=[{"type": "text", "text": "U"}],
    )

    # The two chunks concatenate into one valid JSON document.
    assert json.loads(out)["blocks"] == [
        {"structure": "p", "text": "a"},
        {"structure": "p", "text": "b"},
    ]
    # Exactly two calls; the second replays the partial as an assistant prefill.
    assert len(seen) == 2
    assert [m["role"] for m in seen[0]] == ["system", "user"]
    assert [m["role"] for m in seen[1]] == ["system", "user", "assistant"]
    assert seen[1][-1]["content"] == chunks[0][0]


def test_call_continued_single_call_when_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untruncated reply costs exactly one call (no continuation)."""
    seen: list[Any] = []

    def fake_completion(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return _obj_resp('{"blocks": []}', "stop")

    monkeypatch.setattr("litellm.completion", fake_completion)
    out = llm.call_continued(
        llm.LLMConfig(model="anthropic/claude-haiku-4-5"),
        system_prompt="SYS",
        user_content=[{"type": "text", "text": "U"}],
    )
    assert out == '{"blocks": []}'
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# OpenAI routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "openai/gpt-4.1",
        "openai/o4-mini",
        "gpt-5.4",
        "o3",
    ],
)
def test_is_openai_model_matches_openai_routed_ids(model: str) -> None:
    assert llm.is_openai_model(model) is True
    # The two provider predicates must never both claim a model — the cache and
    # prefill rules key off them independently.
    assert llm.is_anthropic_model(model) is False


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-5",
        "gemini/gemini-2.5-pro",
        "ollama/llama3",
        # Azure deployments of the same models are a different endpoint with
        # their own quirks; nothing here has been validated against them, so
        # they must not be silently routed down the OpenAI path.
        "azure/gpt-5.4",
    ],
)
def test_is_openai_model_rejects_other_providers(model: str) -> None:
    assert llm.is_openai_model(model) is False


def test_prefill_supported_on_anthropic_but_not_openai_or_gemini() -> None:
    assert llm.supports_assistant_prefill("anthropic/claude-haiku-4-5") is True
    # Gemini refuses a trailing model turn outright ("Requests ending with a
    # model turn are not supported", 400) — see supports_assistant_prefill and
    # test_prefill_support.py for what the earlier assumption cost.
    assert llm.supports_assistant_prefill("gemini/gemini-2.5-flash") is False
    # Unrecognized ids (self-hosted, proxied) keep the prefill path — it is the
    # natural behaviour of an OpenAI-compatible server that just concatenates
    # the messages into a prompt.
    assert llm.supports_assistant_prefill("ollama/llama3") is True
    assert llm.supports_assistant_prefill("openai/gpt-5.4") is False


def test_call_continued_asks_openai_to_continue_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On OpenAI the partial gets an explicit continuation turn, not a bare prefill.

    OpenAI reads a trailing assistant message as a *finished* turn and answers
    afresh, so the Anthropic prefill trick re-emits the reply from the top and
    the concatenation is garbage. The wrapper therefore appends a user turn
    telling the model to resume. Without it this test's second call would look
    identical to the Anthropic one.
    """
    import json

    chunks = [
        ('{"continues": "", "blocks": [{"structure": "p", "text": "a"}', "length"),
        (', {"structure": "p", "text": "b"}]}', "stop"),
    ]
    seen: list[list[dict[str, Any]]] = []

    def fake_completion(**kwargs: Any) -> Any:
        seen.append(kwargs["messages"])
        return _obj_resp(*chunks[len(seen) - 1])

    monkeypatch.setattr("litellm.completion", fake_completion)

    out = llm.call_continued(
        llm.LLMConfig(model="openai/gpt-5.4"),
        system_prompt="SYS",
        user_content=[{"type": "text", "text": "U"}],
    )

    assert json.loads(out)["blocks"] == [
        {"structure": "p", "text": "a"},
        {"structure": "p", "text": "b"},
    ]
    assert len(seen) == 2
    assert [m["role"] for m in seen[0]] == ["system", "user"]
    # The partial is still shown, followed by the instruction to resume it.
    assert [m["role"] for m in seen[1]] == ["system", "user", "assistant", "user"]
    assert seen[1][-2]["content"] == chunks[0][0]
    assert "Continue it from exactly where it stopped" in seen[1][-1]["content"]


def test_call_continued_sends_no_cache_markers_to_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cache=True` is a no-op on OpenAI — litellm would reject the markers.

    OpenAI caches stable prefixes implicitly, so there is nothing to ask for.
    """
    seen: list[list[dict[str, Any]]] = []

    def fake_completion(**kwargs: Any) -> Any:
        seen.append(kwargs["messages"])
        return _obj_resp('{"blocks": []}', "stop")

    monkeypatch.setattr("litellm.completion", fake_completion)
    llm.call_continued(
        llm.LLMConfig(model="openai/gpt-5.4"),
        system_prompt=("STATIC", "DYNAMIC"),
        user_content=[{"type": "text", "text": "U"}],
        cache=True,
    )
    dumped = repr(seen)
    assert "cache_control" not in dumped
    # The two halves of the split system prompt are simply concatenated.
    assert seen[0][0]["content"] == "STATIC\nDYNAMIC"


# ---------------------------------------------------------------------------
# Auto-recording of usage from the call layer (gated on record_usage via the config)
# ---------------------------------------------------------------------------


class _PricedResp(dict):  # type: ignore[type-arg]
    """A response that is both subscriptable (``call`` reads ``["choices"]``)
    and attribute-accessible (``extract_cost_and_tokens`` reads ``.usage`` /
    ``._hidden_params``)."""

    def __init__(self, text: str, *, cost: float, tokens: int) -> None:
        from types import SimpleNamespace

        super().__init__(choices=[{"message": {"content": text}}])
        self._hidden_params = {"response_cost": cost}
        self.usage = SimpleNamespace(
            prompt_tokens=tokens, completion_tokens=tokens, total_tokens=tokens * 2
        )


def _tmp_workspace(tmp_path: Any) -> Any:
    from dgml_core.storage import Workspace

    return Workspace(root=tmp_path)


def test_call_auto_records_one_row_with_record_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from dgml_core.usage import read_events

    ws = _tmp_workspace(tmp_path)
    monkeypatch.setattr("litellm.completion", lambda **k: _PricedResp("hi", cost=0.01, tokens=100))
    cfg = llm.LLMConfig(
        model="gpt-4o", workspace=ws, record_usage=True, operation="unit_test", context={"k": "v"}
    )

    out = llm.call(cfg, system_prompt="SYS", user_content=[{"type": "text", "text": "U"}])
    assert out == "hi"

    events = read_events(ws)
    assert len(events) == 1
    assert events[0]["operation"] == "unit_test"
    assert events[0]["cost_usd"] == 0.01
    assert events[0]["total_tokens"] == 200
    assert events[0]["outcome"] == "ok"
    assert events[0]["context"] == {"k": "v"}


def test_call_records_nothing_without_record_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from dgml_core.usage import read_events

    ws = _tmp_workspace(tmp_path)
    monkeypatch.setattr("litellm.completion", lambda **k: _PricedResp("hi", cost=0.01, tokens=100))
    # workspace set but record_usage False (the default) → gated off.
    cfg = llm.LLMConfig(model="gpt-4o", workspace=ws, operation="unit_test")

    llm.call(cfg, system_prompt="SYS", user_content=[{"type": "text", "text": "U"}])
    assert read_events(ws) == []


def test_record_usage_for_aggregates_calls_into_one_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Multiple calls made inside a record_usage_for scope produce ONE
    aggregated row rather than one row per call."""
    from dgml_core.usage import read_events

    ws = _tmp_workspace(tmp_path)
    monkeypatch.setattr("litellm.completion", lambda **k: _PricedResp("x", cost=0.01, tokens=100))
    cfg = llm.LLMConfig(model="gpt-4o", workspace=ws, record_usage=True, operation="agg")

    with llm.record_usage_for(cfg):
        for _ in range(3):
            llm.call(cfg, system_prompt="S", user_content=[{"type": "text", "text": "U"}])

    events = read_events(ws)
    assert len(events) == 1
    assert events[0]["operation"] == "agg"
    assert events[0]["cost_usd"] == pytest.approx(0.03)  # 3x 0.01, summed
    assert events[0]["total_tokens"] == 600  # 3x 200


# ---------------------------------------------------------------------------
# Up-front model validation (_require_supported_model)
# ---------------------------------------------------------------------------


def test_unrecognized_model_raises_model_not_supported() -> None:
    """A model litellm doesn't map is rejected up front with a clear,
    model-focused error — not a confusing downstream param error."""
    from dgml_core.errors import ModelNotSupported

    cfg = llm.LLMConfig(model="gemini/gemini-9.9-nonexistent")
    with pytest.raises(ModelNotSupported, match="not a recognized model id"):
        llm._build_completion_kwargs(cfg, messages=[{"role": "user", "content": "hi"}])


def test_recognized_model_passes_validation() -> None:
    cfg = llm.LLMConfig(model="anthropic/claude-haiku-4-5")
    kwargs = llm._build_completion_kwargs(cfg, messages=[{"role": "user", "content": "hi"}])
    assert kwargs["model"] == "anthropic/claude-haiku-4-5"


def test_api_base_skips_model_validation() -> None:
    """A custom endpoint (proxy / self-hosted) serves models litellm has no
    metadata for, so the existence check is skipped when api_base is set."""
    cfg = llm.LLMConfig(model="my-local/whatever-model", api_base="http://localhost:11434")
    kwargs = llm._build_completion_kwargs(cfg, messages=[{"role": "user", "content": "hi"}])
    assert kwargs["model"] == "my-local/whatever-model"
    assert kwargs["api_base"] == "http://localhost:11434"


# ── Model-aware output-token ceiling ────────────────────────────────────────


def test_model_max_output_tokens_reads_litellm_metadata() -> None:
    # Real metadata: frontier models allow more output than older ones.
    assert llm.model_max_output_tokens("anthropic/claude-sonnet-5") == 128000
    assert llm.model_max_output_tokens("anthropic/claude-haiku-4-5") == 64000
    # Unknown id and custom api_base both fall back to "caller keeps its own".
    assert llm.model_max_output_tokens("not-a-real-model-xyz") is None
    assert llm.model_max_output_tokens("anthropic/claude-sonnet-5", "http://localhost:1234") is None


def test_output_tokens_clamped_to_model_ceiling() -> None:
    """A request for more output than the model allows is a provider 400 —
    the builder lowers it to the model's own ceiling instead."""
    msgs = [{"role": "user", "content": "hi"}]
    # Haiku caps at 64K: an ask for 128K is clamped down.
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-haiku-4-5", max_completion_tokens=128000),
        messages=msgs,
    )
    assert kwargs["max_completion_tokens"] == 64000
    # Sonnet 5 allows the full 128K — passed through untouched.
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-sonnet-5", max_completion_tokens=128000),
        messages=msgs,
    )
    assert kwargs["max_completion_tokens"] == 128000
    # Clamping only ever lowers: a modest ask is never raised.
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-sonnet-5", max_completion_tokens=4096),
        messages=msgs,
    )
    assert kwargs["max_completion_tokens"] == 4096
    # max_tokens follows the same rule.
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-haiku-4-5", max_tokens=128000), messages=msgs
    )
    assert kwargs["max_tokens"] == 64000


def test_rate_limits_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 says "not now", not "never". The pipeline runs documents
    concurrently, so hitting one is expected; before this it raised on the first
    try and the caller silently lost that document's links."""
    attempts: list[int] = []

    def flaky(**_kwargs: Any) -> Any:
        attempts.append(1)
        if len(attempts) < 3:
            raise Exception("litellm.RateLimitError: 429 Too Many Requests")
        return type("R", (), {"choices": [object()]})()

    monkeypatch.setattr(litellm, "completion", flaky)
    monkeypatch.setattr("dgml_core.llm.time.sleep", lambda _s: None)
    llm._completion_with_retry({"model": "claude-sonnet-4-5"})
    assert len(attempts) == 3


def test_non_transient_errors_still_raise_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding rate limits to the retry set must not turn a bad API key into
    three slow attempts."""
    attempts: list[int] = []

    def bad_key(**_kwargs: Any) -> Any:
        attempts.append(1)
        raise Exception("AuthenticationError: invalid x-api-key")

    monkeypatch.setattr(litellm, "completion", bad_key)
    with pytest.raises(Exception, match="AuthenticationError"):
        llm._completion_with_retry({"model": "claude-sonnet-4-5"})
    assert len(attempts) == 1


# ── Anthropic extended thinking (LLMConfig.thinking) ────────────────────────


def test_thinking_sent_to_anthropic_models() -> None:
    """The mode reaches the wire as Anthropic's `thinking` object."""
    msgs = [{"role": "user", "content": "hi"}]
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-sonnet-5", thinking="disabled"), messages=msgs
    )
    assert kwargs["thinking"] == {"type": "disabled"}
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-sonnet-5", thinking="adaptive"), messages=msgs
    )
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_thinking_omitted_when_unset() -> None:
    """Unset means "say nothing", which leaves the model's own default in
    force — deliberately NOT the same as "disabled"."""
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-sonnet-5"),
        messages=[{"role": "user", "content": "hi"}],
    )
    assert "thinking" not in kwargs


def test_thinking_not_sent_to_non_anthropic_models() -> None:
    """The field is Anthropic-shaped; other providers spell reasoning their
    own way and reject it."""
    for model in ("gemini/gemini-2.5-pro", "openai/gpt-5"):
        kwargs = llm._build_completion_kwargs(
            llm.LLMConfig(model=model, thinking="disabled"),
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "thinking" not in kwargs, model


def test_reasoning_effort_wins_over_thinking() -> None:
    """litellm translates reasoning_effort into the same Anthropic field, so
    sending both would contradict itself. The caller's explicit effort wins."""
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(
            model="anthropic/claude-sonnet-5", reasoning_effort="high", thinking="disabled"
        ),
        messages=[{"role": "user", "content": "hi"}],
    )
    assert kwargs["reasoning_effort"] == "high"
    assert "thinking" not in kwargs


def test_thinking_still_sent_when_reasoning_effort_was_dropped() -> None:
    """A forced tool_choice drops reasoning_effort for Anthropic, which frees
    the field — so the configured mode applies rather than silently vanishing."""
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(
            model="anthropic/claude-sonnet-5", reasoning_effort="high", thinking="disabled"
        ),
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice={"type": "function", "function": {"name": "f"}},
    )
    assert "reasoning_effort" not in kwargs
    assert kwargs["thinking"] == {"type": "disabled"}


def test_invalid_thinking_mode_rejected_at_construction() -> None:
    """A typo is a config error; catching it here names the value instead of
    letting the provider answer 400 mid-run."""
    with pytest.raises(ValueError, match="thinking must be one of"):
        llm.LLMConfig(model="anthropic/claude-sonnet-5", thinking="enabled")


class _Reply:
    """A minimal litellm-shaped response.

    Both access styles are supported because the two entry points differ:
    :func:`llm.call` subscripts the response, :func:`llm.call_continued` reads
    attributes.
    """

    def __init__(self, content: str | None, finish_reason: str, **extra: Any) -> None:
        from types import SimpleNamespace

        message = SimpleNamespace(content=content, tool_calls=None, **extra)
        self.choices = [SimpleNamespace(message=message, finish_reason=finish_reason)]
        self.usage = None
        self._mapping = {"choices": [{"message": {"content": content}}]}

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]


def _reply(content: str | None, finish_reason: str, **extra: Any) -> Any:
    return _Reply(content, finish_reason, **extra)


@pytest.mark.parametrize("thinking,expect_prefill", [("disabled", True), ("adaptive", False)])
def test_prefill_on_continuation_follows_thinking_mode(
    monkeypatch: pytest.MonkeyPatch, thinking: str, expect_prefill: bool
) -> None:
    """Anthropic rejects a prefilled assistant turn while thinking is on, so a
    truncated reply may only be continued by prefill when thinking is off.
    Turning thinking off is what makes the cheap continuation path usable."""
    seen: list[list[dict[str, Any]]] = []
    replies = [_reply("part one", "length"), _reply(" and two", "stop")]

    def fake(**kwargs: Any) -> Any:
        seen.append(kwargs["messages"])
        return replies[len(seen) - 1]

    monkeypatch.setattr(litellm, "completion", fake)
    out = llm.call_continued(
        llm.LLMConfig(model="anthropic/claude-haiku-4-5", thinking=thinking),
        system_prompt="s",
        user_content=[{"type": "text", "text": "u"}],
    )
    assert out == "part one and two"
    roles = [m["role"] for m in seen[1]]
    # With prefill the partial is the LAST turn; without it a user turn has to
    # follow and ask for the rest.
    assert (roles[-1] == "assistant") is expect_prefill


def test_call_raises_with_a_reason_when_the_reply_has_no_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reasoning-only reply used to be cast to str and pushed downstream,
    where it read as unparseable model output. Name the cause instead."""
    from dgml_core.errors import EmptyModelResponse

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **_k: _reply(None, "stop", reasoning_content="thought about it"),
    )
    with pytest.raises(EmptyModelResponse) as exc:
        llm.call(
            llm.LLMConfig(model="anthropic/claude-sonnet-5"),
            system_prompt="s",
            user_content=[{"type": "text", "text": "u"}],
        )
    assert "reasoning only" in str(exc.value)
    assert "finish_reason='stop'" in str(exc.value)


def test_call_continued_raises_when_every_round_is_textless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The budget went entirely into reasoning. Returning "" here sent an empty
    transcription window on as if the document itself were empty."""
    from dgml_core.errors import EmptyModelResponse

    monkeypatch.setattr(
        litellm, "completion", lambda **_k: _reply(None, "length", reasoning_content="thinking")
    )
    with pytest.raises(EmptyModelResponse, match="no message content"):
        llm.call_continued(
            llm.LLMConfig(model="anthropic/claude-sonnet-5"),
            system_prompt="s",
            user_content=[{"type": "text", "text": "u"}],
        )
