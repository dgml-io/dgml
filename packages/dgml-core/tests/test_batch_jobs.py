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

"""Batch job mode (:mod:`dgml_core.batch.jobs`): digest, response codec,
replaying executor, billing, and the synchronous recorder seam.

A job is exercised the way the CLI drives it: open a session, run a wave,
close the session with how the run ended, then open a new session on the same
job id (a resumed run) against the same FakeBackend instance — the stand-in
for a provider whose batches outlive the process that submitted them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core import layout, llm
from dgml_core.batch import (
    BatchItemError,
    BatchJobStore,
    FakeBackend,
    ReplayExecutor,
    active_session,
    fake_model_response,
    list_jobs,
    request_digest,
    start_session,
)
from dgml_core.batch.executor import TIER_MARKER
from dgml_core.batch.jobs import (
    RECORD_COLLECTED,
    RECORD_OPEN,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    JobSession,
    dump_response,
    load_response,
)
from dgml_core.errors import BatchJobInvalid, BatchJobNotFound, BatchPending
from dgml_core.storage import Workspace
from dgml_core.usage import (
    BILLED_MARKER,
    TIER_BATCH,
    UsageEvent,
    extract_cost_and_tokens,
    read_events,
    record_usage,
)

SECRET = "sk-DGML-TEST-SECRET-0123456789"


def _kwargs(text: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": "anthropic/claude-haiku-4-5",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 64,
        **extra,
    }


def _answer(request: Any) -> Any:
    text = request.kwargs["messages"][0]["content"]
    return fake_model_response(
        f"reply:{text}",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        cost=0.25,
    )


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root=root)


@pytest.fixture(autouse=True)
def _no_leaked_session() -> Iterator[None]:
    yield
    session = active_session()
    if session is not None:  # a failing test must not poison the next one
        session.close(None)
    assert llm._SYNC_RECORDER is None


def _open(ws: Workspace, job_id: str | None = None, *, wait: bool = False) -> JobSession:
    return start_session(ws, command="test run", argv=["test", "run"], job_id=job_id, wait=wait)


def _executor(backend: FakeBackend, session: JobSession) -> ReplayExecutor:
    return ReplayExecutor(
        backend,
        session=session,
        model="anthropic/claude-haiku-4-5",
        api_base=None,
        poll_interval_s=0,
        sleep=lambda _s: None,
    )


def _texts(responses: dict[str, Any]) -> dict[str, str]:
    return {cid: str(r.choices[0].message.content) for cid, r in responses.items()}


# ---- digest -------------------------------------------------------------------


def test_digest_ignores_credentials_and_transport_but_not_content() -> None:
    base = _kwargs("hello")
    same = {**base, "api_key": SECRET, "timeout": 30.0}
    assert request_digest(same) == request_digest(base)
    headers_a = {**base, "extra_headers": {"anthropic-beta": "x", "x-api-key": "k1"}}
    headers_b = {**base, "extra_headers": {"anthropic-beta": "x", "Authorization": "k2"}}
    assert request_digest(headers_a) == request_digest(headers_b)
    assert request_digest(_kwargs("hello!")) != request_digest(base)
    assert request_digest({**base, "model": "openai/gpt-5.4"}) != request_digest(base)
    assert request_digest({**base, "max_tokens": 65}) != request_digest(base)  # semantic


def test_digest_is_the_same_in_another_process() -> None:
    """Canonical JSON, not Python hashing: a fresh interpreter with a different
    hash seed gets the same digest (resume happens in another process)."""
    kwargs = _kwargs("hello", tools=[{"type": "function", "function": {"name": "f"}}])
    script = (
        "import json, sys; from dgml_core.batch.jobs import request_digest; "
        "print(request_digest(json.loads(sys.argv[1])))"
    )
    digests = set()
    for seed in ("1", "2"):
        out = subprocess.run(
            [sys.executable, "-c", script, json.dumps(kwargs)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        digests.add(out.stdout.strip())
    assert digests == {request_digest(kwargs)}


# ---- response codec -------------------------------------------------------------


def _assert_round_trip(response: Any) -> None:
    data = dump_response(response)
    assert data is not None
    back = load_response(data)
    orig, got = response.choices[0], back.choices[0]
    assert got.message.content == orig.message.content
    assert got.finish_reason == orig.finish_reason
    orig_calls = orig.message.tool_calls or []
    got_calls = got.message.tool_calls or []
    assert [(c.function.name, c.function.arguments) for c in got_calls] == [
        (c.function.name, c.function.arguments) for c in orig_calls
    ]
    assert extract_cost_and_tokens(back) == extract_cost_and_tokens(response)
    assert back._hidden_params.get(TIER_MARKER) == response._hidden_params.get(TIER_MARKER)
    # Item access, as the text wrappers read responses, survives too.
    assert back["choices"][0]["message"]["content"] == orig.message.content


def test_fake_response_round_trips_with_tool_calls_cache_counters_and_markers() -> None:
    response = fake_model_response(
        "",
        finish_reason="tool_calls",
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "submit", "arguments": "{}"}}
        ],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cache_read_input_tokens": 7,
            "cache_creation_input_tokens": 3,
        },
        cost=0.5,
    )
    response._hidden_params[TIER_MARKER] = TIER_BATCH
    _assert_round_trip(response)
    assert (
        extract_cost_and_tokens(load_response(dump_response(response) or b""))["cache_read_tokens"]
        == 7
    )


def test_each_backends_decoded_shape_round_trips() -> None:
    from dgml_core.batch.anthropic import AnthropicBatchBackend

    from .test_batch_anthropic import TEXT_MESSAGE, TOOL_MESSAGE

    anthropic = AnthropicBatchBackend(api_key="test-key")
    decoded = [
        anthropic.decode_message(TEXT_MESSAGE),
        anthropic.decode_message(TOOL_MESSAGE),
    ]
    for response in decoded:
        response._hidden_params[TIER_MARKER] = TIER_BATCH
        _assert_round_trip(response)


def test_only_accounting_fields_of_hidden_params_are_stored() -> None:
    response = fake_model_response("x", cost=0.1)
    response._hidden_params["additional_headers"] = {"x-api-key": SECRET}
    response._hidden_params["api_key"] = SECRET
    data = dump_response(response)
    assert data is not None and SECRET.encode() not in data
    assert set(json.loads(data)["hidden"]) == {"response_cost"}


def test_a_non_model_response_is_not_stored() -> None:
    assert dump_response(object()) is None


# ---- sessions: pending, resume, replay --------------------------------------------


def test_no_wait_pends_after_submission_and_resume_collects_without_resubmitting(
    ws: Workspace,
) -> None:
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=1)
    wave = {"u0_a-1": _kwargs("a"), "u1_b-1": _kwargs("b")}

    session = _open(ws)
    with pytest.raises(BatchPending) as pending:
        _executor(backend, session).run_wave(wave)
    assert pending.value.job_id == session.job_id
    assert pending.value.submitted_batches == 1
    assert pending.value.requests_in_flight == 2
    session.close(pending.value)
    manifest = BatchJobStore(ws, session.job_id).load()
    assert manifest.status == STATUS_PENDING
    assert [r["state"] for r in manifest.provider_batches] == [RECORD_OPEN]

    resumed = _open(ws, session.job_id)
    assert resumed.resumed
    executor = _executor(backend, resumed)
    out = executor.run_wave(wave)
    resumed.close(None)
    assert _texts(out) == {"u0_a-1": "reply:a", "u1_b-1": "reply:b"}
    assert len(backend.submitted) == 1  # collected, never resubmitted
    manifest = BatchJobStore(ws, session.job_id).load()
    assert manifest.status == STATUS_COMPLETED
    assert [r["state"] for r in manifest.provider_batches] == [RECORD_COLLECTED]


def test_stored_responses_replay_with_no_provider_call(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic")
    wave = {"x-1": _kwargs("a")}
    first = _open(ws, wait=True)
    _executor(backend, first).run_wave(wave)
    first.close(BatchPending(first.job_id, submitted_batches=0, requests_in_flight=0))

    again = _open(ws, first.job_id, wait=True)
    executor = _executor(backend, again)
    out = executor.run_wave(wave)
    again.close(None)
    assert _texts(out) == {"x-1": "reply:a"}
    assert len(backend.submitted) == 1
    assert executor.stats.replayed == 1
    assert executor.stats.to_json()["replayed"] == 1


def test_replayed_zero_is_absent_from_stats_json(ws: Workspace) -> None:
    session = _open(ws, wait=True)
    executor = _executor(FakeBackend(_answer, provider="anthropic"), session)
    executor.run_wave({"x-1": _kwargs("a")})
    session.close(None)
    assert "replayed" not in executor.stats.to_json()


def test_identical_requests_keep_separate_responses_by_occurrence(ws: Workspace) -> None:
    calls = {"n": 0}

    def numbered(_request: Any) -> Any:
        calls["n"] += 1
        return fake_model_response(f"reply-{calls['n']}")

    backend = FakeBackend(numbered, provider="anthropic")
    wave = {"a-1": _kwargs("same"), "b-1": _kwargs("same")}
    first = _open(ws, wait=True)
    got = _texts(_executor(backend, first).run_wave(wave))
    first.close(BatchPending(first.job_id, submitted_batches=0, requests_in_flight=0))
    again = _open(ws, first.job_id, wait=True)
    replayed = _texts(_executor(backend, again).run_wave(wave))
    again.close(None)
    assert replayed == got and len(set(got.values())) == 2


def test_a_blocking_run_killed_after_submit_resumes_onto_the_same_batch(ws: Workspace) -> None:
    """The crash case: the batch was accepted, the process died while waiting.
    The record was persisted at submission, so the resume polls it."""

    class Dies(FakeBackend):
        dead = True

        def poll(self, job: Any) -> Any:
            if Dies.dead:
                raise KeyboardInterrupt  # the process being killed mid-wait
            return super().poll(job)

    backend = Dies(_answer, provider="anthropic")
    wave = {"x-1": _kwargs("a")}
    session = _open(ws, wait=True)
    with pytest.raises(KeyboardInterrupt) as interrupted:
        _executor(backend, session).run_wave(wave)
    session.close(interrupted.value)
    assert BatchJobStore(ws, session.job_id).load().status == STATUS_FAILED

    Dies.dead = False
    resumed = _open(ws, session.job_id, wait=True)
    out = _executor(backend, resumed).run_wave(wave)
    resumed.close(None)
    assert _texts(out) == {"x-1": "reply:a"}
    assert len(backend.submitted) == 1


def test_a_changed_input_is_submitted_fresh_not_replayed(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic")
    first = _open(ws, wait=True)
    _executor(backend, first).run_wave({"x-1": _kwargs("old page")})
    first.close(BatchPending(first.job_id, submitted_batches=0, requests_in_flight=0))

    again = _open(ws, first.job_id, wait=True)
    out = _executor(backend, again).run_wave({"x-1": _kwargs("new page")})
    again.close(None)
    assert _texts(out) == {"x-1": "reply:new page"}  # never the stale stored reply
    assert len(backend.submitted) == 2


def test_a_dropped_batch_is_resubmitted_on_resume(ws: Workspace) -> None:
    """`dgml batch cancel` marks open batches dropped; the next resume submits
    those requests again, at batch price — never a silent sync fallback."""
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=5)
    session = _open(ws)
    with pytest.raises(BatchPending) as pending:
        _executor(backend, session).run_wave({"x-1": _kwargs("a")})
    session.close(pending.value)
    store = BatchJobStore(ws, session.job_id)
    manifest = store.load()
    manifest.provider_batches[0]["state"] = "dropped"
    store.save(manifest)

    fresh = FakeBackend(_answer, provider="anthropic")
    resumed = _open(ws, session.job_id, wait=True)
    executor = _executor(fresh, resumed)
    out = executor.run_wave({"x-1": _kwargs("a")})
    resumed.close(None)
    assert _texts(out) == {"x-1": "reply:a"}
    assert len(fresh.submitted) == 1 and executor.stats.sync_fallbacks == 0


def test_expired_items_are_resubmitted_then_recorded(ws: Workspace) -> None:
    seen: dict[str, int] = {}

    def expire_once(request: Any) -> Any:
        text = request.kwargs["messages"][0]["content"]
        seen[text] = seen.get(text, 0) + 1
        if seen[text] == 1:
            return BatchItemError(request.custom_id, "expired", "24h passed")
        return _answer(request)

    backend = FakeBackend(expire_once, provider="anthropic")
    session = _open(ws, wait=True)
    out = _executor(backend, session).run_wave({"x-1": _kwargs("a")})
    session.close(None)
    assert _texts(out) == {"x-1": "reply:a"}
    assert len(backend.submitted) == 2


def test_no_api_key_ever_reaches_the_store(ws: Workspace) -> None:
    def leaky(request: Any) -> Any:
        response = _answer(request)
        response._hidden_params["additional_headers"] = {"x-api-key": SECRET}
        return response

    backend = FakeBackend(leaky, provider="anthropic")
    session = _open(ws, wait=True)
    _executor(backend, session).run_wave({"x-1": _kwargs("a", api_key=SECRET)})
    session.close(None)
    for path in (ws.root / layout.BATCHES_DIR).rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), path


# ---- billing ----------------------------------------------------------------------


def _row(ws: Workspace, response: Any) -> None:
    totals = extract_cost_and_tokens(response)
    record_usage(
        ws,
        UsageEvent(
            at="t",
            operation="transcribe",
            model="m",
            cost_usd=totals["cost_usd"],
            prompt_tokens=totals["prompt_tokens"],
            completion_tokens=totals["completion_tokens"],
            total_tokens=totals["total_tokens"],
            duration_s=0.0,
            outcome="ok",
        ),
    )


def test_a_pending_run_writes_no_rows_and_the_completing_run_bills_everything_once(
    ws: Workspace,
) -> None:
    backend = FakeBackend(_answer, provider="anthropic")
    first = _open(ws, wait=True)
    _row(ws, _executor(backend, first).run_wave({"x-1": _kwargs("a")})["x-1"])
    first.close(BatchPending(first.job_id, submitted_batches=0, requests_in_flight=0))
    assert read_events(ws) == []  # paused: dropped, nothing billed

    final = _open(ws, first.job_id, wait=True)
    replayed = _executor(backend, final).run_wave({"x-1": _kwargs("a")})["x-1"]
    _row(ws, replayed)
    final.close(None)
    rows = read_events(ws)
    assert [r["cost_usd"] for r in rows] == [0.25]  # billed once, in the final run


def test_an_interrupted_runs_rows_are_kept_and_never_billed_again(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic")
    first = _open(ws, wait=True)
    _row(ws, _executor(backend, first).run_wave({"x-1": _kwargs("a")})["x-1"])
    first.close(KeyboardInterrupt())
    assert [r["cost_usd"] for r in read_events(ws)] == [0.25]

    again = _open(ws, first.job_id, wait=True)
    replayed = _executor(backend, again).run_wave({"x-1": _kwargs("a")})["x-1"]
    assert replayed._hidden_params[BILLED_MARKER] is True
    _row(ws, replayed)
    again.close(None)
    assert [r["cost_usd"] for r in read_events(ws)] == [0.25, 0.0]


# ---- the synchronous recorder seam ------------------------------------------------


def test_sync_calls_are_recorded_and_replayed(ws: Workspace) -> None:
    calls: list[dict[str, Any]] = []

    def completion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return fake_model_response("sync reply", cost=0.5)

    config = llm.LLMConfig(model="anthropic/claude-haiku-4-5")
    with patch("litellm.completion", side_effect=completion):
        first = _open(ws, wait=True)
        assert llm.call(config, system_prompt="S", user_content=[{"type": "text", "text": "u"}])
        first.close(BatchPending(first.job_id, submitted_batches=0, requests_in_flight=0))
        again = _open(ws, first.job_id, wait=True)
        text = llm.call(config, system_prompt="S", user_content=[{"type": "text", "text": "u"}])
        again.close(None)
    assert text == "sync reply"
    assert len(calls) == 1  # the resumed run replayed it


def test_without_a_session_the_sync_path_is_untouched() -> None:
    assert llm._SYNC_RECORDER is None
    with patch("litellm.completion", return_value=fake_model_response("plain")):
        out = llm.call(
            llm.LLMConfig(model="anthropic/claude-haiku-4-5"),
            system_prompt="S",
            user_content=[{"type": "text", "text": "u"}],
        )
    assert out == "plain"


# ---- job validation and listing ----------------------------------------------------


def _completed_job(ws: Workspace) -> str:
    """A --no-wait job that did real work and completed (so it is kept, trimmed)."""
    session = _open(ws, wait=False)
    _executor(FakeBackend(_answer, provider="anthropic"), session).run_wave({"x-1": _kwargs("a")})
    session.close(None)
    return session.job_id


def test_resume_checks_the_job_exists_matches_and_is_not_complete(ws: Workspace) -> None:
    with pytest.raises(BatchJobNotFound):
        _open(ws, "bj_000000000000")
    done = _completed_job(ws)
    with pytest.raises(BatchJobInvalid, match="already completed"):
        _open(ws, done)
    other = _open(ws, wait=True)
    other.close(BatchPending(other.job_id, submitted_batches=0, requests_in_flight=0))
    with pytest.raises(BatchJobInvalid, match="was created by"):
        start_session(ws, command="another", argv=[], job_id=other.job_id, wait=True)


def test_a_fresh_job_rejected_before_any_request_leaves_nothing_behind(ws: Workspace) -> None:
    session = _open(ws, wait=True)
    session.close(RuntimeError("pre-flight said no"))
    assert list_jobs(ws) == []
    assert not (ws.root / layout.BATCHES_DIR / session.job_id).exists()


def test_list_jobs_newest_first(ws: Workspace) -> None:
    ids = [_completed_job(ws) for _ in range(2)]
    listed = [m.job_id for m in list_jobs(ws)]
    assert set(listed) == set(ids) and len(listed) == 2
