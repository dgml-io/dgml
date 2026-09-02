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

import sys
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
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Anything a completion writes to stdout is redirected to stderr."""

    def chatty_completion(**kwargs: Any) -> dict[str, Any]:
        print("LiteLLM noise on stdout")  # simulating the chatty dependency
        return _resp("OK")

    monkeypatch.setattr("litellm.completion", chatty_completion)

    result = llm._completion_with_retry({"model": "gpt-4o", "messages": []})

    assert result == _resp("OK")
    captured = capsys.readouterr()
    assert captured.out == ""  # stdout stays clean for the JSON payload
    assert "LiteLLM noise on stdout" in captured.err


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


def test_prefill_supported_everywhere_except_openai() -> None:
    assert llm.supports_assistant_prefill("anthropic/claude-haiku-4-5") is True
    assert llm.supports_assistant_prefill("gemini/gemini-2.5-flash") is True
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
# Auto-recording of usage from the call layer (gated on --debug via the config)
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


def test_call_auto_records_one_row_under_debug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from dgml_core.usage import read_events

    ws = _tmp_workspace(tmp_path)
    monkeypatch.setattr("litellm.completion", lambda **k: _PricedResp("hi", cost=0.01, tokens=100))
    cfg = llm.LLMConfig(
        model="gpt-4o", workspace=ws, debug=True, operation="unit_test", context={"k": "v"}
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


def test_call_records_nothing_without_debug(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from dgml_core.usage import read_events

    ws = _tmp_workspace(tmp_path)
    monkeypatch.setattr("litellm.completion", lambda **k: _PricedResp("hi", cost=0.01, tokens=100))
    # workspace set but debug False → gated off.
    cfg = llm.LLMConfig(model="gpt-4o", workspace=ws, debug=False, operation="unit_test")

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
    cfg = llm.LLMConfig(model="gpt-4o", workspace=ws, debug=True, operation="agg")

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
