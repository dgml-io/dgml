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

import contextlib
import logging
import sys
import threading
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from dgml_core import llm


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
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
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
    caplog.set_level(logging.INFO, logger="dgml_core.llm")

    result = llm._completion_with_retry({"model": "gemini/gemini-3.1-pro-preview", "messages": []})

    assert calls["n"] == 3
    assert result.choices[0].message == "ok"
    # Each retry costs a paid call and the backoff, so each is logged, like a
    # raised error's retry.
    assert [r.getMessage() for r in caplog.records if r.name == "dgml_core.llm"] == [
        "model returned no choices (model 'gemini/gemini-3.1-pro-preview'); retry 1 of 2 in 2s",
        "model returned no choices (model 'gemini/gemini-3.1-pro-preview'); retry 2 of 2 in 4s",
    ]


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
            {"model": "gemini/gemini-3.1-pro-preview", "messages": []}, max_retries=3
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

    result = llm._completion_with_retry({"model": "gemini/gemini-3.1-pro-preview", "messages": []})

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


def _server_error(request_id: str | None = "req_abc123") -> Exception:
    """A LiteLLM 500 as the OpenAI provider raises it, headers included."""
    import httpx

    headers = {"x-request-id": request_id} if request_id else {}
    return litellm.exceptions.InternalServerError(
        message="InternalServerError: OpenAIException - Internal Server Error",
        llm_provider="openai",
        model="gpt-4o",
        response=httpx.Response(
            500, headers=headers, request=httpx.Request("POST", "https://api.openai.com")
        ),
        litellm_debug_info="\nModel: gpt-4o\nMessages: `[{'content': 'SECRET DOCUMENT TEXT'}]`",
    )


def test_completion_with_retry_logs_each_retry_and_counts_attempts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing(**kwargs: Any) -> Any:
        raise _server_error()

    monkeypatch.setattr("litellm.completion", failing)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    caplog.set_level(logging.INFO, logger="dgml_core.llm")

    with pytest.raises(litellm.exceptions.InternalServerError) as raised:
        llm._completion_with_retry(
            {
                "model": "gpt-4o",
                "api_base": "https://proxy.example.com/v1",
                "api_key": "sk-SECRET",
                "messages": [],
            }
        )

    retries = [r.getMessage() for r in caplog.records if r.name == "dgml_core.llm"]
    context = (
        "model 'gpt-4o' at https://proxy.example.com, api key passed, HTTP 500, "
        "request id req_abc123"
    )
    assert retries == [
        f"model call failed (InternalServerError, {context}); retry 1 of 2 in 2s",
        f"model call failed (InternalServerError, {context}, 2 attempts); retry 2 of 2 in 4s",
    ]
    assert llm.provider_error_context(raised.value, model="gpt-4o") == (
        "model 'gpt-4o', HTTP 500, request id req_abc123, 3 attempts"
    )


def test_completion_with_retry_logs_a_missing_key_as_not_passed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The kwargs leave out an empty key and None alike, so the retry line and
    # the final error (built from the config) must word both the same.
    attempts = iter([_server_error(), _resp("ok")])

    def flaky(**kwargs: Any) -> Any:
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("litellm.completion", flaky)
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)
    caplog.set_level(logging.INFO, logger="dgml_core.llm")

    llm._completion_with_retry({"model": "gpt-4o", "messages": []})

    [retry] = [r.getMessage() for r in caplog.records if r.name == "dgml_core.llm"]
    assert "api key not passed" in retry
    assert "None" not in retry and "empty" not in retry


def test_completion_with_retry_does_not_log_a_non_transient_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing(**kwargs: Any) -> Any:
        raise ValueError("bad request")

    monkeypatch.setattr("litellm.completion", failing)
    caplog.set_level(logging.DEBUG, logger="dgml_core.llm")

    with pytest.raises(ValueError) as raised:
        llm._completion_with_retry({"model": "gpt-4o", "messages": []})

    assert [r for r in caplog.records if r.name == "dgml_core.llm"] == []
    # One attempt is not worth mentioning.
    assert llm.provider_error_context(raised.value, model="gpt-4o") == "model 'gpt-4o'"


def test_provider_error_context_leaves_out_prompt_and_credentials() -> None:
    context = llm.provider_error_context(
        _server_error(),
        model="gpt-4o",
        api_base="https://user:hunter2@gw.example.com:8443/tok_SECRET/v1?api-key=sk-SECRET",
        api_key="sk-SECRET-key-value",
    )

    assert context == (
        "model 'gpt-4o' at https://gw.example.com:8443, api key passed, HTTP 500, "
        "request id req_abc123"
    )
    assert "SECRET" not in context and "hunter2" not in context


@pytest.mark.parametrize(
    ("api_key", "state"),
    [
        ("sk-x", "api key passed"),
        ("", "api key not passed"),
        (None, "api key not passed"),
    ],
)
def test_provider_error_context_reports_api_key_state(api_key: str | None, state: str) -> None:
    assert llm.provider_error_context(RuntimeError("x"), api_key=api_key) == state


def test_provider_error_context_leaves_out_an_unreported_api_key() -> None:
    # Not passing api_key at all is "unknown", not "no key".
    assert "api key" not in llm.provider_error_context(RuntimeError("x"), model="m")


def test_describe_call_failure_appends_the_calls_context() -> None:
    config = llm.LLMConfig(
        model="gpt-4o", api_key="sk-SECRET", api_base="https://proxy.example.com/v1"
    )

    assert llm.describe_call_failure(_server_error(), config) == (
        "InternalServerError: litellm.InternalServerError: InternalServerError: "
        "OpenAIException - Internal Server Error "
        "(model 'gpt-4o' at https://proxy.example.com, api key passed, HTTP 500, "
        "request id req_abc123)"
    )


def test_provider_error_context_reads_the_request_id_litellm_keeps_aside() -> None:
    # As LiteLLM raises an Anthropic error (seen live): exc.response is a
    # stand-in with no headers; the real ones are set on the exception after.
    import httpx

    exc = litellm.exceptions.AuthenticationError(
        message="AnthropicException - invalid x-api-key",
        llm_provider="anthropic",
        model="claude-sonnet-5",
        response=httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com")),
    )
    exc.litellm_response_headers = httpx.Headers(  # type: ignore[attr-defined]
        {"request-id": "req_011live", "content-type": "application/json"}
    )

    assert llm.provider_error_context(exc, model="anthropic/claude-sonnet-5") == (
        "model 'anthropic/claude-sonnet-5', HTTP 401, request id req_011live"
    )


def test_provider_error_context_survives_an_exception_whose_attributes_raise() -> None:
    # Runs inside except blocks: an error of its own would replace the provider's.
    class Unreadable(Exception):
        @property
        def response(self) -> Any:
            raise RuntimeError("no response attached")

    assert llm.provider_error_context(Unreadable(), model="m") == "model 'm'"


def test_completion_with_retry_keeps_an_exception_that_refuses_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Recording the attempt count must never replace the provider's error.
    class Locked(Exception):
        def __setattr__(self, name: str, value: Any) -> None:
            if not name.startswith("__"):
                raise AttributeError(f"{name} is read-only")
            super().__setattr__(name, value)

    def failing(**kwargs: Any) -> Any:
        raise Locked("bad request")

    monkeypatch.setattr("litellm.completion", failing)

    with pytest.raises(Locked):
        llm._completion_with_retry({"model": "gpt-4o", "messages": []})


def test_provider_error_context_with_nothing_known() -> None:
    assert llm.provider_error_context(RuntimeError("x")) == ""
    assert llm.provider_error_context(_server_error(request_id=None)) == "HTTP 500"
    assert (
        llm.provider_error_context(RuntimeError("x"), model="m", api_base="not a url")
        == "model 'm' at <unparseable api_base>"
    )


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
        llm.LLMConfig(model="anthropic/claude-haiku-5-5"),
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
        llm.LLMConfig(model="anthropic/claude-haiku-5-5"),
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
        llm.LLMConfig(model="anthropic/claude-haiku-5-5"),
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
        llm.LLMConfig(model="anthropic/claude-haiku-5-5"),
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
        llm.LLMConfig(model="anthropic/claude-haiku-5-5"),
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
        llm.LLMConfig(model="anthropic/claude-sonnet-4-6"),
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
        llm.LLMConfig(model="anthropic/claude-haiku-5-5"),
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
        "gemini/gemini-3.1-pro-preview",
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
    assert llm.supports_assistant_prefill("anthropic/claude-sonnet-4-6") is True
    # The current Opus and Sonnet refuse a prefilled turn (measured, 400).
    assert llm.supports_assistant_prefill("anthropic/claude-opus-5-5") is False
    assert llm.supports_assistant_prefill("anthropic/claude-sonnet-5-5") is False
    # Gemini refuses a trailing model turn outright ("Requests ending with a
    # model turn are not supported", 400) — see supports_assistant_prefill and
    # test_prefill_support.py for what the earlier assumption cost.
    assert llm.supports_assistant_prefill("gemini/gemini-3.8-flash") is False
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
    cfg = llm.LLMConfig(model="anthropic/claude-haiku-5-5")
    kwargs = llm._build_completion_kwargs(cfg, messages=[{"role": "user", "content": "hi"}])
    assert kwargs["model"] == "anthropic/claude-haiku-5-5"


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
    assert llm.model_max_output_tokens("anthropic/claude-haiku-5-5") == 128000
    # Unknown id and custom api_base both fall back to "caller keeps its own".
    assert llm.model_max_output_tokens("not-a-real-model-xyz") is None
    assert llm.model_max_output_tokens("anthropic/claude-sonnet-5", "http://localhost:1234") is None


def test_output_tokens_clamped_to_model_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A request for more output than the model allows is a provider 400 —
    the builder lowers it to the model's own ceiling instead."""
    msgs = [{"role": "user", "content": "hi"}]
    # Fake a 64K ceiling rather than pin a real model that has one: every such
    # model is an older release that will drop out of litellm's metadata.
    small = "anthropic/claude-haiku-5-5"
    real_info = litellm.get_model_info

    def fake_info(model: str, *args: Any, **kwargs: Any) -> Any:
        info = real_info(model, *args, **kwargs)
        return {**info, "max_output_tokens": 64000} if model == small else info

    monkeypatch.setattr(litellm, "get_model_info", fake_info)
    # A 64K model asked for 128K is clamped down.
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model=small, max_completion_tokens=128000),
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
        llm.LLMConfig(model=small, max_tokens=128000), messages=msgs
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


def test_between_tools_is_sonnet_55_off_switch() -> None:
    """Claude Sonnet 5.5 rejects `{"type": "disabled"}`; `between_tools` is how it
    turns thinking off for a plain text reply, and it goes on the wire verbatim
    (type only, no other field)."""
    msgs = [{"role": "user", "content": "hi"}]
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-sonnet-5-5", thinking="between_tools"), messages=msgs
    )
    assert kwargs["thinking"] == {"type": "between_tools"}


def test_disabled_thinking_becomes_between_tools_on_sonnet_55() -> None:
    """Sonnet 5.5 answers `{"type": "disabled"}` with a 400 naming
    `between_tools` as its off switch, so a config that says "disabled" — the
    generation default — must reach it as `between_tools`."""
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-sonnet-5-5", thinking="disabled"),
        messages=[{"role": "user", "content": "hi"}],
    )
    assert kwargs["thinking"] == {"type": "between_tools"}


@pytest.mark.parametrize("mode", ["disabled", "between_tools"])
def test_thinking_off_becomes_lowest_effort_on_opus_55(mode: str) -> None:
    """Opus 5.5 rejects both off switches; adaptive thinking at the lowest
    effort is the nearest request it accepts."""
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-opus-5-5", thinking=mode),
        messages=[{"role": "user", "content": "hi"}],
    )
    assert kwargs["reasoning_effort"] == "low"
    assert "thinking" not in kwargs


def test_adaptive_thinking_unchanged_on_opus_55() -> None:
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-opus-5-5", thinking="adaptive"),
        messages=[{"role": "user", "content": "hi"}],
    )
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert "reasoning_effort" not in kwargs


_TWO_TOOLS = [
    {"type": "function", "function": {"name": "lookup", "parameters": {}}},
    {"type": "function", "function": {"name": "submit", "parameters": {}}},
]


@pytest.mark.parametrize("model", ["anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5"])
def test_named_tool_choice_becomes_auto_over_that_tool_alone(model: str) -> None:
    """These models answer any forced tool_choice with a 400. Sending "auto"
    over only the forced function keeps that call the model's only option, and
    the choice no longer being forced lets reasoning_effort through."""
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model=model, reasoning_effort="high"),
        messages=[{"role": "user", "content": "hi"}],
        tools=_TWO_TOOLS,
        tool_choice={"type": "function", "function": {"name": "submit"}},
    )
    assert kwargs["tool_choice"] == "auto"
    assert [t["function"]["name"] for t in kwargs["tools"]] == ["submit"]
    assert kwargs["reasoning_effort"] == "high"


def test_required_tool_choice_becomes_auto_over_every_tool() -> None:
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-opus-5-5"),
        messages=[{"role": "user", "content": "hi"}],
        tools=_TWO_TOOLS,
        tool_choice="required",
    )
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["tools"] == _TWO_TOOLS


def test_forced_tool_choice_kept_where_supported() -> None:
    """Only the models that reject forcing lose it."""
    for model in ("anthropic/claude-opus-5", "anthropic/claude-haiku-5-5", "openai/gpt-5.4"):
        kwargs = llm._build_completion_kwargs(
            llm.LLMConfig(model=model),
            messages=[{"role": "user", "content": "hi"}],
            tools=_TWO_TOOLS,
            tool_choice={"type": "function", "function": {"name": "submit"}},
        )
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "submit"}}, model
        assert kwargs["tools"] == _TWO_TOOLS, model


def test_opus_55_lowest_effort_reaches_the_wire_as_adaptive_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Opus 5.5 off-switch substitute relies on litellm translating
    `reasoning_effort` into adaptive thinking plus an effort, not into the
    `budget_tokens` form Opus 5.5 rejects. Capture the request body litellm
    would send to pin that."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    sent: list[dict[str, Any]] = []

    def capture(self: Any, url: str, data: Any = None, json: Any = None, **_kw: Any) -> Any:
        import json as _json

        sent.append(json if json is not None else _json.loads(data))
        raise RuntimeError("captured, not sent")

    monkeypatch.setattr(HTTPHandler, "post", capture)
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model="anthropic/claude-opus-5-5", api_key="k", thinking="disabled"),
        messages=[{"role": "user", "content": "hi"}],
        tools=_TWO_TOOLS,
        tool_choice={"type": "function", "function": {"name": "submit"}},
    )
    # litellm re-raises the capture as its own error type; the body is the check.
    with contextlib.suppress(Exception):
        litellm.completion(**kwargs, max_retries=0)
    (body,) = sent
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "low"}
    assert body["tool_choice"] == {"type": "auto"}
    assert [t["name"] for t in body["tools"]] == ["submit"]


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
    for model in ("gemini/gemini-3.1-pro-preview", "openai/gpt-5"):
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
        llm.LLMConfig(model="anthropic/claude-sonnet-4-6", thinking=thinking),
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


@pytest.mark.parametrize(
    "api_base",
    ["http://[::1", "https://h:notaport", "https://h:99999", "not a url"],
)
def test_redacted_api_base_reports_a_malformed_url(api_base: str) -> None:
    assert llm._redacted_api_base(api_base) == "<unparseable api_base>"


def test_provider_error_context_ignores_a_non_string_api_base() -> None:
    # Typed str | None; anything else is left out rather than parsed.
    context = llm.provider_error_context(
        RuntimeError("x"),
        model="m",
        api_base=123,  # type: ignore[arg-type]
    )
    assert context == "model 'm'"


def test_provider_error_context_reads_only_well_typed_fields() -> None:
    """Fields that are missing, None, or not the expected type are left out."""
    exc = RuntimeError("x")
    exc.status_code = None  # type: ignore[attr-defined]
    exc.response = SimpleNamespace(headers=None)  # type: ignore[attr-defined]
    assert llm.provider_error_context(exc) == ""

    exc.response = SimpleNamespace(headers=["x-request-id"])  # type: ignore[attr-defined]
    assert llm.provider_error_context(exc) == ""

    exc.response = SimpleNamespace(headers={"x-request-id": None})  # type: ignore[attr-defined]
    assert llm.provider_error_context(exc) == ""


def test_build_user_content_labels_each_image_by_its_own_format() -> None:
    """A page image is a JPEG or a PNG, whichever encoding was smaller; each
    image block must carry the MIME type of the bytes it holds."""
    jpeg = b"\xff\xd8\xff\xe0 jpeg body"
    png = b"\x89PNG\r\n\x1a\n png body"
    content = llm.build_user_content(instruction_text="x", images=[jpeg, png])
    urls = [block["image_url"]["url"] for block in content if block["type"] == "image_url"]
    assert urls[0].startswith("data:image/jpeg;base64,")
    assert urls[1].startswith("data:image/png;base64,")


def test_build_user_content_names_the_inline_pdf() -> None:
    """The inline PDF block carries a filename (see `_PDF_FILENAME`)."""
    content = llm.build_user_content(instruction_text="read", pdf_bytes=b"%PDF-1.4 x")
    assert [block["type"] for block in content] == ["text", "file"]
    assert content[1]["file"] == {
        "file_data": "data:application/pdf;base64,JVBERi0xLjQgeA==",
        "filename": "document.pdf",
    }
