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

"""The run-level batch deadline (``--batch-deadline``).

Every scenario runs on a :class:`FakeBackend` and an injected wall clock that
only the executor's ``sleep`` advances, so "the deadline passes mid-wave" is
exact, not timing-dependent.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core import llm
from dgml_core.batch import (
    TIER_MARKER,
    BatchDeadline,
    BatchExecutor,
    BatchRequest,
    FakeBackend,
    ReplayExecutor,
    active_session,
    fake_model_response,
    parse_duration,
    register_backend,
    resume_would_wait,
    start_session,
)
from dgml_core.batch import registry as batch_registry
from dgml_core.batch.jobs import (
    RECORD_COLLECTED,
    RECORD_DROPPED,
    RECORD_SETTLING,
    STATUS_COMPLETED,
    BatchJobStore,
    JobSession,
    cancel_job,
    job_status,
    prune_jobs,
)
from dgml_core.errors import BatchExecutionFailed, BatchJobInvalid, BatchPending
from dgml_core.storage import Workspace
from dgml_core.usage import TIER_BATCH, TIER_STANDARD

T0 = 1_800_000_000.0  # 2027-01-15T08:00:00Z


class Clock:
    """A wall clock that moves only when told to (the executor's sleep)."""

    def __init__(self, now: float = T0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _kwargs(text: str) -> dict[str, Any]:
    return {
        "model": "anthropic/claude-haiku-4-5",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 64,
    }


def _answer(request: BatchRequest) -> Any:
    text = request.kwargs["messages"][0]["content"]
    return fake_model_response(f"batch:{text}", cost=0.25)


def _text(response: Any) -> str:
    return str(response.choices[0].message.content)


def _tier(response: Any) -> str:
    return str(response._hidden_params[TIER_MARKER])


class _Sync:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, kwargs: dict[str, Any]) -> Any:
        text = kwargs["messages"][0]["content"]
        self.calls.append(text)
        return fake_model_response(f"sync:{text}", cost=0.5)


def _wave(*texts: str) -> dict[str, dict[str, Any]]:
    return {f"u_{t}": _kwargs(t) for t in texts}


def _executor(
    backend: FakeBackend, clock: Clock, deadline: BatchDeadline | None, **kw: Any
) -> tuple[BatchExecutor, _Sync]:
    sync = _Sync()
    return (
        BatchExecutor(
            backend,
            sync_execute=sync,
            poll_interval_s=60.0,
            sleep=clock.sleep,
            deadline=deadline,
            **kw,
        ),
        sync,
    )


# ---- durations ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("90m", 5400.0), ("6h", 21600.0), ("1d", 86400.0), ("45s", 45.0), ("5400", 5400.0)],
)
def test_parse_duration(text: str, seconds: float) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "abc", "10w", "-5m", "0", "1h30m"])
def test_parse_duration_rejects_non_durations(text: str) -> None:
    with pytest.raises(ValueError):
        parse_duration(text)


# ---- one executor, no job ---------------------------------------------------------------


def test_deadline_mid_wave_cancels_collects_partial_results_and_runs_the_rest_sync(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = Clock()
    # Still running for 5 polls; a and b are processed at the first one.
    backend = FakeBackend(
        _answer,
        polls_until_ended=5,
        processed_early=lambda r: r.custom_id in ("u_a", "u_b"),
    )
    deadline = BatchDeadline(at=T0 + 100, clock=clock)
    ex, sync = _executor(backend, clock, deadline)

    with caplog.at_level(logging.WARNING, logger="dgml_core.batch.deadline"):
        out = ex.run_wave(_wave("a", "b", "c", "d"))

    assert {cid: _text(r) for cid, r in out.items()} == {
        "u_a": "batch:a",
        "u_b": "batch:b",
        "u_c": "sync:c",
        "u_d": "sync:d",
    }
    assert _tier(out["u_a"]) == TIER_BATCH and _tier(out["u_c"]) == TIER_STANDARD
    assert sorted(sync.calls) == ["c", "d"]
    assert backend.canceled == ["fake_batch_0001"]
    assert len(backend.submitted) == 1  # the canceled requests were never resubmitted
    # The last sleep was cut short so the deadline round ran right at it.
    assert clock.sleeps[:2] == [60.0, 40.0]
    assert deadline.to_json() == {
        "at": "2027-01-15T08:01:40Z",
        "expired": True,
        "canceled_batches": 1,
        "collected_after_cancel": 2,
        "sync_after_deadline": 2,
        "possibly_double_billed": 0,
    }
    # Cost: two batch results (0.25) and two standard calls (0.5).
    assert ex.stats.cost_json() == {"cost_usd": 1.5, "standard_cost_usd": 2.0, "saved_usd": 0.5}
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "batch deadline" in warnings[0].getMessage()


def test_after_the_deadline_later_waves_run_synchronously_without_submitting() -> None:
    clock = Clock()
    backend = FakeBackend(_answer)
    deadline = BatchDeadline(at=T0 + 100, clock=clock)
    ex, sync = _executor(backend, clock, deadline)

    first = ex.run_wave(_wave("a"))
    assert _tier(first["u_a"]) == TIER_BATCH and not deadline.fired
    clock.now = T0 + 101
    second = ex.run_wave(_wave("b", "c"))

    assert {cid: _text(r) for cid, r in second.items()} == {"u_b": "sync:b", "u_c": "sync:c"}
    assert len(backend.submitted) == 1 and sorted(sync.calls) == ["b", "c"]
    assert deadline.to_json()["sync_after_deadline"] == 2
    assert deadline.to_json()["canceled_batches"] == 0


def test_post_deadline_sync_requests_run_concurrently() -> None:
    """After the deadline a wave runs synchronously on a worker pool, as the
    synchronous pipeline would (``sync_workers``), not one request at a time:
    a deadline is set because time is short, so the rest must not run N times
    slower than plain sync. Each fake call blocks until all three are in
    flight; run serially, the first would time out at the barrier."""
    clock = Clock()
    backend = FakeBackend(_answer)
    deadline = BatchDeadline(at=T0 - 1, clock=clock)
    barrier = threading.Barrier(3, timeout=5)
    sync = _Sync()

    def together(kwargs: dict[str, Any]) -> Any:
        barrier.wait()
        return sync(kwargs)

    ex = BatchExecutor(
        backend, sync_execute=together, sleep=clock.sleep, deadline=deadline, sync_workers=3
    )

    out = ex.run_wave(_wave("a", "b", "c"))

    assert list(out) == ["u_a", "u_b", "u_c"]  # wave order, whatever finished first
    assert {cid: _text(r) for cid, r in out.items()} == {
        "u_a": "sync:a",
        "u_b": "sync:b",
        "u_c": "sync:c",
    }
    assert backend.submitted == []
    assert ex.stats.sync_fallbacks == 3 and deadline.sync_after_deadline == 3


def test_sync_workers_one_runs_fallbacks_serially_in_wave_order() -> None:
    clock = Clock()
    deadline = BatchDeadline(at=T0 - 1, clock=clock)
    ex, sync = _executor(FakeBackend(_answer), clock, deadline, sync_workers=1)

    ex.run_wave(_wave("c", "a", "b"))

    assert sync.calls == ["c", "a", "b"]


def test_a_cancel_that_does_not_settle_runs_everything_synchronously() -> None:
    clock = Clock()
    backend = FakeBackend(
        _answer,
        polls_until_ended=50,
        processed_early=lambda r: True,
        cancel_settle_polls=10_000,
    )
    deadline = BatchDeadline(at=T0 + 30, clock=clock)
    ex, sync = _executor(backend, clock, deadline)
    ex.cancel_settle_s = 20.0

    out = ex.run_wave(_wave("a", "b"))

    assert {_text(r) for r in out.values()} == {"sync:a", "sync:b"}
    assert sorted(sync.calls) == ["a", "b"]
    assert deadline.canceled_batches == 1 and deadline.collected_after_cancel == 0
    assert clock.now - (T0 + 30) <= 30  # bounded: the settle wait did not run on


def test_the_earlier_max_poll_s_still_fails_the_wave() -> None:
    clock = Clock()
    backend = FakeBackend(_answer, polls_until_ended=5)
    ex, _sync = _executor(backend, clock, BatchDeadline(at=T0 + 10_000, clock=clock), max_poll_s=0)
    with pytest.raises(BatchExecutionFailed, match="not finished after"):
        ex.run_wave(_wave("a"))


def test_no_deadline_leaves_polling_exactly_as_before() -> None:
    clock = Clock()
    backend = FakeBackend(_answer, polls_until_ended=2)
    ex, sync = _executor(backend, clock, None)
    out = ex.run_wave(_wave("a"))
    assert _tier(out["u_a"]) == TIER_BATCH and sync.calls == []
    assert clock.sleeps == [60.0, 60.0]
    assert backend.canceled == []


# ---- job mode -----------------------------------------------------------------------------


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root=root)


@pytest.fixture(autouse=True)
def _no_leaked_session() -> Iterator[None]:
    saved = dict(batch_registry._REGISTRY)
    yield
    batch_registry._REGISTRY.clear()
    batch_registry._REGISTRY.update(saved)
    session = active_session()
    if session is not None:
        session.close(None)
    assert llm._SYNC_RECORDER is None


_ARGV = ["test", "run", "--batch", "--no-wait"]


def _open(
    ws: Workspace,
    clock: Clock,
    job_id: str | None = None,
    *,
    wait: bool = False,
    deadline_s: float | None = None,
) -> JobSession:
    return start_session(
        ws,
        command="test run",
        argv=_ARGV,
        job_id=job_id,
        wait=wait,
        deadline_s=deadline_s,
        clock=clock,
    )


def _replay(backend: FakeBackend, session: JobSession, clock: Clock) -> ReplayExecutor:
    return ReplayExecutor(
        backend,
        session=session,
        model="anthropic/claude-haiku-4-5",
        api_base=None,
        poll_interval_s=60.0,
        sleep=clock.sleep,
    )


def _litellm_sync(calls: list[str]) -> Any:
    def completion(**kwargs: Any) -> Any:
        calls.append(kwargs["messages"][0]["content"])
        return fake_model_response(f"sync:{kwargs['messages'][0]['content']}", cost=0.5)

    return patch("litellm.completion", side_effect=completion)


def test_the_deadline_is_stored_on_the_first_run_and_honored_by_every_resume(
    ws: Workspace,
) -> None:
    clock = Clock()
    first = _open(ws, clock, deadline_s=3600)
    assert first.deadline is not None and first.deadline.at == T0 + 3600
    first.close(BatchPending(first.job_id, submitted_batches=0, requests_in_flight=0))
    stored = BatchJobStore(ws, first.job_id).load().deadline
    assert stored is not None
    assert stored["at"] == "2027-01-15T09:00:00Z" and stored["seconds"] == 3600

    clock.now += 1800
    # The same flag again (what `batch resume` replays from the argv): the
    # stored instant, not one recomputed from now.
    again = _open(ws, clock, first.job_id, deadline_s=3600)
    assert again.deadline is not None and again.deadline.at == T0 + 3600
    again.close(BatchPending(again.job_id, submitted_batches=0, requests_in_flight=0))
    # No flag: still the job's deadline.
    plain = _open(ws, clock, first.job_id)
    assert plain.deadline is not None and plain.deadline.at == T0 + 3600
    plain.close(BatchPending(plain.job_id, submitted_batches=0, requests_in_flight=0))
    # A different one is refused, before the job is touched.
    with pytest.raises(BatchJobInvalid, match="already has a batch deadline"):
        _open(ws, clock, first.job_id, deadline_s=7200)
    assert job_status(ws, first.job_id)["deadline"] == {
        "at": "2027-01-15T09:00:00Z",
        "expired": False,
    }


def test_a_job_without_a_deadline_gets_one_on_the_first_resume_that_sets_it(
    ws: Workspace,
) -> None:
    clock = Clock()
    first = _open(ws, clock)
    assert first.deadline is None
    first.close(BatchPending(first.job_id, submitted_batches=0, requests_in_flight=0))
    assert "deadline" not in job_status(ws, first.job_id)
    clock.now += 100
    later = _open(ws, clock, first.job_id, deadline_s=60)
    assert later.deadline is not None and later.deadline.at == T0 + 160
    later.close(BatchPending(later.job_id, submitted_batches=0, requests_in_flight=0))


def test_pause_then_resume_after_the_deadline_cancels_collects_and_finishes_sync(
    ws: Workspace,
) -> None:
    clock = Clock()
    backend = FakeBackend(
        _answer,
        provider="anthropic",
        polls_until_ended=5,
        processed_early=lambda r: r.kwargs["messages"][0]["content"] == "a",
    )
    register_backend("anthropic", lambda _cfg: backend)  # what `batch status` polls
    wave = _wave("a", "b")

    first = _open(ws, clock, deadline_s=3600)
    with pytest.raises(BatchPending) as pending:
        _replay(backend, first, clock).run_wave(wave)
    first.close(pending.value)
    job_id = first.job_id
    # Before the deadline a resume would only pause again: short-circuited.
    assert resume_would_wait(ws, job_id) is not None

    clock.now = T0 + 3601
    with patch("dgml_core.batch.deadline.wall_clock", clock):
        # Past it, the short-circuit is off: the resume must run to finish.
        assert resume_would_wait(ws, job_id) is None
        assert job_status(ws, job_id)["deadline"]["expired"] is True

    calls: list[str] = []
    with _litellm_sync(calls):
        resumed = _open(ws, clock, job_id)
        ex = _replay(backend, resumed, clock)
        out = ex.run_wave(wave)
        keys = dict(ex._keys)
        manifest = resumed.manifest
        # Both responses are in the job's store: a later replay serves them.
        assert all(resumed.store.has_response(k) for k in keys.values())
        resumed.close(None)

    assert {cid: _text(r) for cid, r in out.items()} == {"u_a": "batch:a", "u_b": "sync:b"}
    assert calls == ["b"]
    assert len(backend.submitted) == 1 and backend.canceled == ["fake_batch_0001"]
    assert [r["state"] for r in manifest.provider_batches] == [RECORD_COLLECTED]
    assert resumed.deadline_json() == {
        "at": "2027-01-15T09:00:00Z",
        "expired": True,
        "canceled_batches": 1,
        "collected_after_cancel": 1,
        "sync_after_deadline": 1,
        "possibly_double_billed": 0,
    }
    assert BatchJobStore(ws, job_id).load().status == STATUS_COMPLETED


def test_a_post_deadline_sync_response_is_replayed_by_a_later_run(ws: Workspace) -> None:
    clock = Clock()
    backend = FakeBackend(_answer, provider="anthropic")
    first = _open(ws, clock, wait=True, deadline_s=10)
    clock.now += 11
    calls: list[str] = []
    with _litellm_sync(calls):
        out = _replay(backend, first, clock).run_wave(_wave("a"))
    assert _text(out["u_a"]) == "sync:a" and calls == ["a"]
    # The run fails later for another reason; the job stays, resumable.
    first.close(RuntimeError("boom"))

    with _litellm_sync(calls):
        again = _open(ws, clock, first.job_id, wait=True)
        ex = _replay(backend, again, clock)
        out = ex.run_wave(_wave("a"))
        again.close(None)
    assert _text(out["u_a"]) == "sync:a"
    assert calls == ["a"]  # replayed, not called again
    assert ex.stats.replayed == 1 and backend.submitted == []
    # Job-wide: the first run's synchronous call is still counted.
    assert again.deadline_json() is not None
    assert again.deadline_json()["sync_after_deadline"] == 1  # type: ignore[index]


def _unsettling(**kw: Any) -> FakeBackend:
    """An OpenAI-like backend: every request is processed while the batch
    runs (and keeps its result after a cancel), and a cancel stays
    ``cancelling`` until a test sets ``cancel_settle_polls = 0``."""
    return FakeBackend(
        _answer,
        provider="anthropic",
        polls_until_ended=50,
        processed_early=lambda r: True,
        cancel_settle_polls=10_000,
        cancel_settle_s=60.0,
        **kw,
    )


def _settling_job(ws: Workspace, clock: Clock, backend: FakeBackend) -> tuple[str, list[str]]:
    """A non-silent job whose one batch was canceled at the deadline and did
    not settle: its run failed (resumable) with the record ``settling``."""
    first = _open(ws, clock, deadline_s=100)
    first.close(BatchPending(first.job_id, submitted_batches=0, requests_in_flight=0))
    calls: list[str] = []
    with _litellm_sync(calls):
        run = _open(ws, clock, first.job_id, wait=True)
        out = _replay(backend, run, clock).run_wave(_wave("a", "b"))
        run.close(RuntimeError("a later step failed"))
    assert {_text(r) for r in out.values()} == {"sync:a", "sync:b"}
    return first.job_id, calls


def test_each_backend_declares_its_measured_cancel_settle_wait() -> None:
    from dgml_core.batch.anthropic import AnthropicBatchBackend
    from dgml_core.batch.deadline import CANCEL_SETTLE_S, cancel_settle_s
    from dgml_core.batch.gemini import GeminiBatchBackend
    from dgml_core.batch.openai import OpenAIBatchBackend

    assert AnthropicBatchBackend.cancel_settle_s == 450.0
    assert cancel_settle_s(AnthropicBatchBackend) == 450.0
    # Gemini ends a canceled job within seconds (measured 3-7 s): wait 60 s.
    assert GeminiBatchBackend.cancel_settle_s == 60.0
    assert cancel_settle_s(GeminiBatchBackend) == 60.0
    # OpenAI settles on ~5-minute sweeps and keeps processing meanwhile: two sweeps.
    assert OpenAIBatchBackend.cancel_settle_s == 660.0
    assert cancel_settle_s(OpenAIBatchBackend) == 660.0
    # A backend without one (or with a nonsense value) keeps the old default.
    assert cancel_settle_s(object()) == CANCEL_SETTLE_S == 180.0
    backend = FakeBackend(_answer)
    backend.cancel_settle_s = 0.0
    assert cancel_settle_s(backend) == 180.0
    ex, _sync = _executor(FakeBackend(_answer, cancel_settle_s=77.0), Clock(), None)
    assert ex.cancel_settle_s == 77.0


def test_a_slow_cancel_within_the_backends_settle_wait_is_collected() -> None:
    clock = Clock()
    # Settles after 25 polls (~250 s): past the old fixed 180 s, inside 300 s.
    backend = FakeBackend(
        _answer,
        polls_until_ended=50,
        processed_early=lambda r: True,
        cancel_settle_polls=25,
        cancel_settle_s=300.0,
    )
    deadline = BatchDeadline(at=T0 + 100, clock=clock)
    ex, sync = _executor(backend, clock, deadline)

    out = ex.run_wave(_wave("a", "b"))

    assert {_text(r) for r in out.values()} == {"batch:a", "batch:b"}
    assert sync.calls == []
    assert deadline.collected_after_cancel == 2 and deadline.possibly_double_billed == 0
    assert backend.cleaned == ["fake_batch_0001"]
    # The settle wait polls every 10 s — not hammering, not the 60 s interval.
    after = clock.sleeps[2:]
    assert after and set(after) == {10.0}


def test_a_cancel_that_never_settles_is_counted_possibly_double_billed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = Clock()
    backend = _unsettling()
    deadline = BatchDeadline(at=T0 + 100, clock=clock)
    ex, sync = _executor(backend, clock, deadline)

    with caplog.at_level(logging.WARNING, logger="dgml_core.batch"):
        out = ex.run_wave(_wave("a", "b"))

    assert {_text(r) for r in out.values()} == {"sync:a", "sync:b"}
    assert sorted(sync.calls) == ["a", "b"]
    assert deadline.to_json()["possibly_double_billed"] == 2
    assert deadline.sync_after_deadline == 2 and deadline.collected_after_cancel == 0
    # Waited the backend's 60 s, no longer.
    assert clock.now - (T0 + 100) == pytest.approx(60.0, abs=10.0)
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "dgml_core.batch.executor"
    ]
    assert len(warned) == 1
    assert "fake_batch_0001" in warned[0] and "2 request(s)" in warned[0]
    assert "billed twice" in warned[0]
    assert backend.cleaned == []  # an unsettled batch cannot be cleaned up


def test_an_unsettled_cancel_is_kept_settling_and_no_late_result_replaces_the_sync_one(
    ws: Workspace,
) -> None:
    clock = Clock()
    backend = _unsettling()
    job_id, calls = _settling_job(ws, clock, backend)
    assert sorted(calls) == ["a", "b"]

    manifest = BatchJobStore(ws, job_id).load()
    (record,) = manifest.provider_batches
    assert record["state"] == RECORD_SETTLING and record["possibly_double_billed"] == 2
    assert "not settled after 60s" in record["settling_reason"]
    assert manifest.open_records() == []  # never polled or collected as open
    assert manifest.deadline is not None
    assert manifest.deadline["runs"]["2"]["possibly_double_billed"] == 2

    # The batch settles; a resume replays the synchronous responses (never
    # the late batch results), and records what the late batch billed.
    backend.cancel_settle_polls = 0
    with _litellm_sync(calls):
        resumed = _open(ws, clock, job_id, wait=True)
        ex = _replay(backend, resumed, clock)
        out = ex.run_wave(_wave("a", "b"))
        resumed.close(None)

    assert {cid: _text(r) for cid, r in out.items()} == {"u_a": "sync:a", "u_b": "sync:b"}
    assert sorted(calls) == ["a", "b"]  # not called again
    assert ex.stats.replayed == 2 and len(backend.submitted) == 1
    assert backend.cleaned == ["fake_batch_0001"]
    block = resumed.deadline_json()
    assert block is not None
    assert block["possibly_double_billed"] == 2  # job-wide, from the earlier run
    assert block["settling_batches"] == 0
    assert block["late_billed"] == 2 and block["late_billed_usd"] == 0.5
    (record,) = BatchJobStore(ws, job_id).load().provider_batches
    assert record["state"] == RECORD_DROPPED and record["cleanup"] == "done"
    assert record["late_billed"] == {"requests": 2, "cost_usd": 0.5}


def test_status_reports_a_settled_late_cost_read_only_and_prune_keeps_the_job(
    ws: Workspace,
) -> None:
    clock = Clock()
    backend = _unsettling()
    register_backend("anthropic", lambda _cfg: backend)  # what the commands poll
    job_id, _calls = _settling_job(ws, clock, backend)

    # Still canceling: prune keeps the job (the only record of the batch).
    pruned = prune_jobs(ws)
    assert job_id in pruned["kept"] and pruned["settling"] == [job_id]
    status = job_status(ws, job_id)
    (entry,) = status["batches"]
    assert entry["state"] == RECORD_SETTLING and entry["done"] is False
    assert status["deadline"]["settling_batches"] == 1
    assert status["deadline"]["late_billed"] == 0

    backend.cancel_settle_polls = 0
    status = job_status(ws, job_id)
    (entry,) = status["batches"]
    assert entry["done"] is True and entry["late_billed"] == {"requests": 2, "cost_usd": 0.5}
    assert status["deadline"]["late_billed"] == 2
    assert status["deadline"]["settling_batches"] == 0
    # Read-only: nothing stored, nothing cleaned up.
    assert BatchJobStore(ws, job_id).load().provider_batches[0]["state"] == RECORD_SETTLING
    assert backend.cleaned == []

    # `batch cancel` records it for good and cleans the batch up.
    payload = cancel_job(ws, job_id)
    (entry,) = payload["batches"]
    assert entry["state"] == RECORD_DROPPED and entry["cleanup"] == "done"
    assert entry["late_billed"] == {"requests": 2, "cost_usd": 0.5}
    assert backend.cleaned == ["fake_batch_0001"]
    assert prune_jobs(ws)["deleted"] == [job_id]


def test_a_silent_job_is_kept_while_its_deadline_cancel_is_settling(ws: Workspace) -> None:
    clock = Clock()
    backend = _unsettling()
    register_backend("anthropic", lambda _cfg: backend)
    calls: list[str] = []
    with _litellm_sync(calls):
        session = _open(ws, clock, wait=True, deadline_s=100)  # a plain blocking run
        _replay(backend, session, clock).run_wave(_wave("a"))
        session.close(None)
    assert session.deadline_json() is not None
    assert session.deadline_json()["possibly_double_billed"] == 1  # type: ignore[index]
    manifest = BatchJobStore(ws, session.job_id).load()
    assert manifest.status == STATUS_COMPLETED
    (record,) = manifest.provider_batches
    # Trimmed, but the settling record is kept whole for reconciling.
    assert record["state"] == RECORD_SETTLING and record["job"]["custom_ids"]
    backend.cancel_settle_polls = 0
    assert prune_jobs(ws)["deleted"] == [session.job_id]
    assert backend.cleaned == ["fake_batch_0001"]


def test_a_silent_job_whose_cancels_settled_is_still_deleted(ws: Workspace) -> None:
    clock = Clock()
    backend = FakeBackend(
        _answer,
        provider="anthropic",
        polls_until_ended=50,
        processed_early=lambda r: True,
        cancel_settle_polls=2,
    )
    with _litellm_sync([]):
        session = _open(ws, clock, wait=True, deadline_s=100)
        out = _replay(backend, session, clock).run_wave(_wave("a"))
        session.close(None)
    assert _text(out["u_a"]) == "batch:a"
    assert session.deadline_json()["possibly_double_billed"] == 0  # type: ignore[index]
    assert "settling_batches" not in session.deadline_json()  # type: ignore[operator]
    assert not BatchJobStore(ws, session.job_id).exists()


# ---- F3: the settle wait is a time budget, whatever the poll interval ----------------


def _never_settling() -> FakeBackend:
    """Runs past any deadline here; a cancel never settles (60 s budget)."""
    return FakeBackend(
        _answer,
        provider="anthropic",
        polls_until_ended=1_000_000,
        processed_early=lambda r: True,
        cancel_settle_polls=1_000_000,
        cancel_settle_s=60.0,
    )


@pytest.mark.parametrize("poll_interval_s", [0.5, 1.0, 3.0, 10.0, 60.0])
def test_the_settle_wait_lasts_cancel_settle_s_whatever_the_poll_interval(
    poll_interval_s: float,
) -> None:
    clock = Clock()
    backend = _never_settling()  # cancel_settle_s = 60
    deadline = BatchDeadline(at=T0 + 100, clock=clock)
    ex = BatchExecutor(
        backend,
        sync_execute=_Sync(),
        poll_interval_s=poll_interval_s,
        sleep=clock.sleep,
        deadline=deadline,
    )

    ex.run_wave(_wave("a", "b"))

    # The cancel happened at the deadline; the wait then ran its full budget.
    waited = clock.now - (T0 + 100)
    assert 60.0 <= waited <= 60.0 + max(poll_interval_s, 10.0)
    assert deadline.possibly_double_billed == 2


def test_the_settle_wait_never_overshoots_its_budget() -> None:
    clock = Clock()
    backend = _never_settling()
    deadline = BatchDeadline(at=T0 + 100, clock=clock)
    ex = BatchExecutor(
        backend, sync_execute=_Sync(), poll_interval_s=7.0, sleep=clock.sleep, deadline=deadline
    )
    ex.cancel_settle_s = 25.0

    ex.run_wave(_wave("a"))

    assert clock.now - (T0 + 100) == pytest.approx(25.0)


def test_the_job_mode_settle_wait_lasts_cancel_settle_s_with_a_short_poll_interval(
    ws: Workspace,
) -> None:
    clock = Clock()
    backend = _never_settling()
    with _litellm_sync([]):
        session = _open(ws, clock, wait=True, deadline_s=100)
        ex = ReplayExecutor(
            backend,
            session=session,
            model="anthropic/claude-haiku-4-5",
            api_base=None,
            poll_interval_s=1.0,
            sleep=clock.sleep,
        )
        ex.run_wave(_wave("a"))
        session.close(None)
    deadline_at = T0 + 100
    assert clock.now - deadline_at >= 60.0
    manifest = BatchJobStore(ws, session.job_id).load()
    (record,) = manifest.provider_batches
    assert record["state"] == RECORD_SETTLING


# ---- F10: sync_after_deadline counts only what the deadline sent to sync ------------


def test_sync_after_deadline_counts_only_deadline_caused_fallbacks() -> None:
    """A request that falls back for its own reason (it cannot be encoded, the
    provider called it invalid) would have run synchronously anyway; only the
    requests the deadline took off the batch path count."""

    class _BadEncode(FakeBackend):
        def encode(self, request: BatchRequest) -> dict[str, Any]:
            if request.custom_id == "u_bad":
                raise ValueError("cannot encode")
            return super().encode(request)

    def answer(request: BatchRequest) -> Any:
        if request.custom_id == "u_invalid":
            from dgml_core.batch import BatchItemError

            return BatchItemError("u_invalid", "invalid", "schema rejected")
        return _answer(request)

    clock = Clock()
    backend = _BadEncode(
        answer,
        polls_until_ended=5,
        processed_early=lambda r: r.custom_id == "u_invalid",
        cancel_settle_s=60.0,
    )
    deadline = BatchDeadline(at=T0 + 100, clock=clock)
    ex, sync = _executor(backend, clock, deadline)

    ex.run_wave(_wave("bad", "invalid", "late"))

    # bad (encode) and invalid (its own batch error) are not the deadline's doing.
    assert sorted(sync.calls) == ["bad", "invalid", "late"]
    assert deadline.sync_after_deadline == 1
    # A later wave, after the deadline, runs sync entirely because of it.
    ex.run_wave(_wave("next1", "next2"))
    assert deadline.sync_after_deadline == 3
    # A wave below min_wave_size would have run sync with or without it.
    ex.min_wave_size = 5
    ex.run_wave(_wave("tiny"))
    assert deadline.sync_after_deadline == 3


# ---- F4: a failed cancel or collect at the deadline is counted and warned ------------


class _CancelFails(FakeBackend):
    def cancel(self, job: Any) -> None:
        self.canceled.append(job.job_id)
        raise RuntimeError("cancel endpoint down")


class _CollectFailsAfterCancel(FakeBackend):
    def results(self, job: Any) -> Any:
        if job.job_id in self.canceled:
            raise RuntimeError("results endpoint down")
        return super().results(job)


def _still_running(cls: type[FakeBackend]) -> FakeBackend:
    return cls(_answer, provider="anthropic", polls_until_ended=1_000_000, cancel_settle_s=60.0)


def _double_billed_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "dgml_core.batch.executor"
    ]


@pytest.mark.parametrize("cls", [_CancelFails, _CollectFailsAfterCancel])
def test_a_failed_cancel_or_collect_at_the_deadline_counts_possibly_double_billed(
    cls: type[FakeBackend], caplog: pytest.LogCaptureFixture
) -> None:
    clock = Clock()
    backend = _still_running(cls)
    deadline = BatchDeadline(at=T0 + 100, clock=clock)
    ex, _sync = _executor(backend, clock, deadline)

    with caplog.at_level(logging.WARNING, logger="dgml_core.batch"):
        out = ex.run_wave(_wave("a", "b"))

    assert {_text(r) for r in out.values()} == {"sync:a", "sync:b"}
    assert deadline.possibly_double_billed == 2
    (warned,) = _double_billed_warnings(caplog)
    assert "fake_batch_0001" in warned and "2 request(s)" in warned
    assert "billed twice" in warned


@pytest.mark.parametrize("cls", [_CancelFails, _CollectFailsAfterCancel])
def test_a_failed_cancel_or_collect_at_the_deadline_keeps_the_job_record_settling(
    cls: type[FakeBackend], ws: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    clock = Clock()
    backend = _still_running(cls)
    with _litellm_sync([]), caplog.at_level(logging.WARNING, logger="dgml_core.batch"):
        session = _open(ws, clock, wait=True, deadline_s=100)
        out = _replay(backend, session, clock).run_wave(_wave("a", "b"))
        session.close(None)

    assert {_text(r) for r in out.values()} == {"sync:a", "sync:b"}
    manifest = BatchJobStore(ws, session.job_id).load()
    (record,) = manifest.provider_batches
    assert record["state"] == RECORD_SETTLING
    assert record["possibly_double_billed"] == 2
    assert session.deadline_json()["possibly_double_billed"] == 2  # type: ignore[index]
    assert len(_double_billed_warnings(caplog)) == 1
