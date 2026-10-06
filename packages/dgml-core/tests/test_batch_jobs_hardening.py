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

"""Batch job mode, executor hardening: batch-level rejection bisects, an
uncertain create is recorded and never resubmitted, provider artifacts are
released and cleaned up (never for a paused batch), the polling deadline
follows the provider's expiry window, and admission-capped stages replay
deterministically."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import pytest
from dgml_core import llm
from dgml_core.batch import (
    BatchRequest,
    FakeBackend,
    ReplayExecutor,
    Unit,
    active_session,
    fake_model_response,
    run_stage,
    start_session,
)
from dgml_core.batch.executor import POLL_MARGIN_S
from dgml_core.batch.jobs import (
    RECORD_COLLECTED,
    RECORD_DROPPED,
    RECORD_OPEN,
    RECORD_UNCERTAIN,
    STATUS_FAILED,
    BatchJobStore,
    JobSession,
)
from dgml_core.batch.types import BatchRejected, BatchSubmitUncertain
from dgml_core.errors import BatchExecutionFailed, BatchJobInvalid, BatchPending
from dgml_core.storage import Workspace

MODEL = "anthropic/claude-haiku-4-5"


def _kwargs(text: str) -> dict[str, Any]:
    return {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 64}


def _answer(request: BatchRequest) -> Any:
    return fake_model_response(f"reply:{request.kwargs['messages'][0]['content']}", cost=0.25)


def _text(response: Any) -> str:
    return str(response.choices[0].message.content)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root=root)


@pytest.fixture(autouse=True)
def _no_leaked_session() -> Iterator[None]:
    yield
    session = active_session()
    if session is not None:
        session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))
    assert llm._SYNC_RECORDER is None


def _open(ws: Workspace, job_id: str | None = None, *, wait: bool = False) -> JobSession:
    return start_session(ws, command="test run", argv=["test", "run"], job_id=job_id, wait=wait)


def _executor(backend: FakeBackend, session: JobSession, **kw: Any) -> ReplayExecutor:
    kw.setdefault("poll_interval_s", 0)
    return ReplayExecutor(
        backend, session=session, model=MODEL, api_base=None, sleep=lambda _s: None, **kw
    )


def _pause(session: JobSession) -> None:
    session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))


def _no_sync(kwargs: dict[str, Any], **_kw: Any) -> Any:
    raise AssertionError(f"unexpected synchronous call: {kwargs}")


# ---- bisection -----------------------------------------------------------------


def test_job_mode_bisects_a_batch_rejected_at_submit(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "_completion_attempts", _no_sync)

    def too_big(batch: list[BatchRequest]) -> Exception | None:
        return BatchRejected("too many enqueued tokens") if len(batch) > 1 else None

    backend = FakeBackend(_answer, provider="anthropic", fail_submit=too_big)
    session = _open(ws, wait=True)
    ex = _executor(backend, session)
    wave = {"x-1": _kwargs("a"), "y-1": _kwargs("b")}

    out = ex.run_wave(wave)

    assert _text(out["x-1"]) == "reply:a" and _text(out["y-1"]) == "reply:b"
    assert [len(b) for b in backend.attempted] == [2, 1, 1]
    records = session.manifest.provider_batches
    assert [r["state"] for r in records] == [RECORD_COLLECTED, RECORD_COLLECTED]
    assert ex.stats.to_json()["bisections"] == 1
    session.close(None)


def test_job_mode_bisects_an_accepted_batch_failed_at_batch_level(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "_completion_attempts", _no_sync)
    backend = FakeBackend(_answer, provider="anthropic", reject_batch=lambda b: len(b) > 1)
    session = _open(ws, wait=True)
    ex = _executor(backend, session)

    out = ex.run_wave({"x-1": _kwargs("a"), "y-1": _kwargs("b"), "z-1": _kwargs("c")})

    assert [_text(out[c]) for c in ("x-1", "y-1", "z-1")] == ["reply:a", "reply:b", "reply:c"]
    # 3 rejected → 1 + (2 rejected → 1 + 1).
    assert [len(b) for b in backend.submitted] == [3, 1, 2, 1, 1]
    assert ex.stats.bisections == 2
    assert all(r["state"] == RECORD_COLLECTED for r in session.manifest.provider_batches)
    session.close(None)


def test_bisected_halves_survive_a_pause_and_are_collected_on_resume(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A --no-wait run: the rejected batch ends, its halves are submitted and
    still running → pause. The resume collects the halves (never resubmits)."""
    monkeypatch.setattr(llm, "_completion_attempts", _no_sync)
    backend = FakeBackend(
        _answer, provider="anthropic", polls_until_ended=1, reject_batch=lambda b: len(b) > 1
    )
    wave = {"x-1": _kwargs("a"), "y-1": _kwargs("b")}

    session = _open(ws)
    with pytest.raises(BatchPending):
        _executor(backend, session).run_wave(wave)  # batch 1 still running
    _pause(session)
    assert backend.cleaned == []  # nothing collected yet → nothing cleaned

    resumed = _open(ws, session.job_id)
    with pytest.raises(BatchPending):
        _executor(backend, resumed).run_wave(wave)  # batch 1 rejected → halves running
    _pause(resumed)
    assert [len(b) for b in backend.submitted] == [2, 1, 1]
    assert backend.cleaned == ["fake_batch_0001"]  # the halves were left open

    third = _open(ws, session.job_id)
    with pytest.raises(BatchPending):
        _executor(backend, third).run_wave(wave)  # the halves' first poll: running
    _pause(third)

    final = _open(ws, session.job_id)
    out = _executor(backend, final).run_wave(wave)
    final.close(None)
    assert _text(out["x-1"]) == "reply:a" and _text(out["y-1"]) == "reply:b"
    assert len(backend.submitted) == 3  # nothing resubmitted
    assert sorted(backend.cleaned) == ["fake_batch_0001", "fake_batch_0002", "fake_batch_0003"]


# ---- uncertain create ----------------------------------------------------------


def test_job_mode_uncertain_create_is_recorded_and_fails_the_run(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm, "_completion_attempts", _no_sync)

    def second_times_out(batch: list[BatchRequest]) -> Exception | None:
        if batch[0].custom_id == "y-1":
            return BatchSubmitUncertain("read timeout after send")
        return None

    backend = FakeBackend(
        _answer, provider="anthropic", max_requests=1, fail_submit=second_times_out
    )
    session = _open(ws, wait=True)
    with pytest.raises(BatchExecutionFailed, match="may exist"):
        _executor(backend, session).run_wave({"x-1": _kwargs("a"), "y-1": _kwargs("b")})

    assert len(backend.attempted) == 2  # never resubmitted
    assert backend.canceled == ["fake_batch_0001"]
    stored = BatchJobStore(ws, session.job_id).load()
    states = [r["state"] for r in stored.provider_batches]
    assert states == [RECORD_DROPPED, RECORD_UNCERTAIN]
    uncertain = stored.provider_batches[1]
    assert "may exist" in uncertain["error"]
    assert uncertain["job"]["job_id"].startswith("uncertain_")
    assert list(uncertain["keys"]) == ["y-1"]

    # Even when a caller soft-fails the stage, the run ends failed with it.
    session.close(None)
    manifest = BatchJobStore(ws, session.job_id).load()
    assert manifest.status == STATUS_FAILED
    assert manifest.error is not None and "may exist" in manifest.error


def test_uncertain_record_is_never_settled_or_polled(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic", fail_submit=BatchSubmitUncertain("503"))
    session = _open(ws, wait=True)
    with pytest.raises(BatchExecutionFailed):
        _executor(backend, session).run_wave({"x-1": _kwargs("a")})
    assert session.manifest.open_records() == []
    session.close(None)
    assert backend.polls == 0 and backend.canceled == []


# ---- release / cleanup ---------------------------------------------------------


def test_job_mode_releases_and_cleans_up_collected_batches(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic", max_requests=1)
    session = _open(ws, wait=True)
    _executor(backend, session).run_wave({"x-1": _kwargs("a"), "y-1": _kwargs("b")})
    session.close(None)
    assert backend.released == [("x-1",), ("y-1",)]
    assert backend.cleaned == ["fake_batch_0001", "fake_batch_0002"]


def test_a_paused_batch_is_not_cleaned_up_until_collected(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=1)
    wave = {"x-1": _kwargs("a")}
    session = _open(ws)
    with pytest.raises(BatchPending):
        _executor(backend, session).run_wave(wave)
    _pause(session)
    assert backend.released == [("x-1",)] and backend.cleaned == []
    assert [r["state"] for r in session.manifest.provider_batches] == [RECORD_OPEN]

    resumed = _open(ws, session.job_id)
    _executor(backend, resumed).run_wave(wave)
    resumed.close(None)
    assert backend.cleaned == ["fake_batch_0001"]


def test_settling_at_close_cleans_up_what_it_collects(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=1)
    session = _open(ws)
    with pytest.raises(BatchPending):
        _executor(backend, session).run_wave({"x-1": _kwargs("a")})
    # The run ends some other way than pausing: the (now ended) batch is
    # collected by settling, and cleaned up once its responses are stored.
    session.close(RuntimeError("the command failed later"))
    assert backend.cleaned == ["fake_batch_0001"]


# ---- polling deadline ------------------------------------------------------------


def test_replay_executor_derives_its_deadline_from_the_backend(ws: Workspace) -> None:
    session = _open(ws, wait=True)
    backend = FakeBackend(_answer, provider="fake", max_wait_s=48 * 3600)
    assert _executor(backend, session).max_poll_s == 48 * 3600 + POLL_MARGIN_S
    assert _executor(backend, session, max_poll_s=5.0).max_poll_s == 5.0
    session.close(None)


def _unit(name: str, cfg: llm.LLMConfig, steps: int) -> Unit:
    def gen() -> Generator[dict[str, Any], Any, list[str]]:
        texts: list[str] = []
        for n in range(steps):
            response = yield _kwargs(f"{name}-{n}")
            texts.append(_text(response))
        return texts

    return Unit(name, cfg, gen())


def _uncertain_job(ws: Workspace) -> tuple[str, FakeBackend]:
    backend = FakeBackend(_answer, provider="anthropic", fail_submit=BatchSubmitUncertain("503"))
    session = _open(ws, wait=True)
    with pytest.raises(BatchExecutionFailed):
        _executor(backend, session).run_wave({"x-1": _kwargs("a")})
    session.close(None)
    return session.job_id, backend


@pytest.mark.parametrize("wait", [True, False], ids=["blocking", "no-wait"])
def test_a_job_with_an_uncertain_create_refuses_to_run_again(ws: Workspace, wait: bool) -> None:
    job_id, backend = _uncertain_job(ws)

    with pytest.raises(BatchJobInvalid, match=f"dgml batch cancel {job_id}") as info:
        _open(ws, job_id, wait=wait)
    assert "may have created those batches" in str(info.value)
    assert info.value.code == "BATCH_JOB_INVALID"
    assert len(backend.attempted) == 1  # nothing resubmitted
    assert active_session() is None
    # Refused before taking the lease or counting a run.
    store = BatchJobStore(ws, job_id)
    assert store.lease_holder() is None and store.load().runs == 1


def test_an_acknowledged_uncertain_create_is_resubmitted_on_resume(ws: Workspace) -> None:
    job_id, _backend = _uncertain_job(ws)
    # What `dgml batch cancel` does to the record.
    store = BatchJobStore(ws, job_id)
    manifest = store.load()
    manifest.provider_batches[0]["state"] = RECORD_DROPPED
    store.save(manifest, merge=False)

    healthy = FakeBackend(_answer, provider="anthropic")
    resumed = _open(ws, job_id, wait=True)
    out = _executor(healthy, resumed).run_wave({"x-1": _kwargs("a")})
    resumed.close(None)
    assert _text(out["x-1"]) == "reply:a"
    assert len(healthy.submitted) == 1


# ---- a stage outage leaves the job failed and resumable ---------------------------


def _two_units() -> list[Unit]:
    cfg = llm.LLMConfig(model=MODEL, operation="transcribe")
    return [_unit("short", cfg, 1), _unit("long", cfg, 2)]


def _second_wave_fails(batch: list[BatchRequest]) -> Exception | None:
    contents = {r.kwargs["messages"][0]["content"] for r in batch}
    return RuntimeError("provider outage") if "long-1" in contents else None


@pytest.mark.parametrize("wait", [True, False], ids=["blocking", "no-wait"])
def test_a_stage_outage_ends_the_job_failed_with_its_responses_kept(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, wait: bool
) -> None:
    monkeypatch.setattr(llm, "_completion_attempts", _no_sync)
    broken = FakeBackend(_answer, provider="anthropic", fail_submit=_second_wave_fails)
    session = _open(ws, wait=wait)

    out = run_stage(_two_units(), _executor(broken, session), stage="t")

    # The stage soft-failed per unit — the caller sees outcomes, not a raise ...
    assert out["short"].ok and out["short"].result == ["reply:short-0"]
    assert out["long"].stage_error and isinstance(out["long"].error, BatchExecutionFailed)
    assert session.stage_failures and "provider outage" in session.stage_failures[0]
    # ... yet the job ends failed, not completed: nothing trimmed, still resumable.
    session.close(None)
    manifest = BatchJobStore(ws, session.job_id).load()
    assert manifest.status == STATUS_FAILED
    assert manifest.error is not None and "a batch stage failed" in manifest.error
    assert manifest.provider_batches  # not trimmed to a summary

    healthy = FakeBackend(_answer, provider="anthropic")
    resumed = _open(ws, session.job_id, wait=True)
    again = run_stage(_two_units(), _executor(healthy, resumed), stage="t")
    resumed.close(None)
    assert again["long"].result == ["reply:long-0", "reply:long-1"]
    # Only the request the outage lost is paid for again.
    sent = [r.kwargs["messages"][0]["content"] for b in healthy.submitted for r in b]
    assert sent == ["long-1"]
    assert BatchJobStore(ws, session.job_id).load().status == "completed"


def test_a_clean_stage_still_completes_the_job(ws: Workspace) -> None:
    session = _open(ws, wait=True)
    run_stage(_two_units(), _executor(FakeBackend(_answer, provider="anthropic"), session))
    assert session.stage_failures == []
    session.close(None)
    assert not BatchJobStore(ws, session.job_id).exists()  # silent job: deleted


@pytest.mark.parametrize("failure", ["uncertain", "throttled"])
def test_job_mode_failed_wave_releases_the_encodings_it_never_submitted(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """X1 (F9 in job mode): ReplayExecutor has its own submit loop. Planning
    encoded every batch of the wave; when a create fails the wave mid-plan
    (uncertain, or throttled), the batches never submitted must not stay
    cached in the backend."""
    from dgml_core.batch.types import BatchThrottled

    monkeypatch.setattr(llm, "_completion_attempts", _no_sync)
    error: Exception = (
        BatchSubmitUncertain("read timeout after send")
        if failure == "uncertain"
        else BatchThrottled("HTTP 429")
    )

    def second_fails(batch: list[BatchRequest]) -> Exception | None:
        return error if batch[0].custom_id == "y-1" else None

    backend = FakeBackend(_answer, provider="anthropic", max_requests=1, fail_submit=second_fails)
    session = _open(ws, wait=True)
    wave = {"x-1": _kwargs("a"), "y-1": _kwargs("b"), "z-1": _kwargs("c"), "w-1": _kwargs("d")}
    with pytest.raises(BatchExecutionFailed):
        _executor(backend, session).run_wave(wave)
    assert [b[0].custom_id for b in backend.attempted] == ["x-1", "y-1"]  # z, w never sent
    released = sorted(cid for batch in backend.released for cid in batch)
    assert released == sorted(wave)
    session.close(None)
