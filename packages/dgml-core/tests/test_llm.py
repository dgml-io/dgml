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
