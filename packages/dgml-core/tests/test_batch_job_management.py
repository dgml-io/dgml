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

"""Job management as a library (:mod:`dgml_core.batch.jobs`): status, the
resume short-circuit, cancel, delete, prune, unlock and list — the calls
``dgml batch`` makes, exercised without the CLI.

A job is paused the way a ``--no-wait`` run pauses: a session submits one wave
to a :class:`FakeBackend` registered as the provider (so ``record_backend``
resolves the same instance a later call polls), and closes pending.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from dgml_core.batch import (
    BatchJobStore,
    FakeBackend,
    ReplayExecutor,
    active_session,
    cancel_job,
    delete_job,
    fake_model_response,
    job_status,
    list_job_summaries,
    prune_jobs,
    register_backend,
    resume_would_wait,
    start_session,
    unlock_job,
)
from dgml_core.batch import registry as batch_registry
from dgml_core.errors import (
    BatchExecutionFailed,
    BatchJobInvalid,
    BatchJobNotFound,
    BatchPending,
)
from dgml_core.storage import Workspace

_MODEL = "anthropic/claude-haiku-4-5"


def _answer(request: Any) -> Any:
    return fake_model_response(
        "ok", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, cost=0.1
    )


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root=root)


@pytest.fixture(autouse=True)
def _registry() -> Iterator[None]:
    saved = dict(batch_registry._REGISTRY)
    try:
        yield
    finally:
        session = active_session()
        if session is not None:
            session.close(None)
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


def _install(backend: FakeBackend) -> FakeBackend:
    register_backend("anthropic", lambda _cfg: backend)
    return backend


def _paused(
    ws: Workspace, backend: FakeBackend, *, argv: tuple[str, ...] = ("run", "--no-wait")
) -> str:
    """A pending job with one open provider batch of two requests."""
    session = start_session(ws, command="test run", argv=list(argv), job_id=None, wait=False)
    executor = ReplayExecutor(
        backend,
        session=session,
        model=_MODEL,
        api_base=None,
        poll_interval_s=0,
        sleep=lambda _s: None,
    )
    wave = {
        "a-1": {"model": _MODEL, "messages": [{"role": "user", "content": "a"}]},
        "b-1": {"model": _MODEL, "messages": [{"role": "user", "content": "b"}]},
    }
    with pytest.raises(BatchPending) as pending:
        executor.run_wave(wave)
    session.close(pending.value)
    return session.job_id


def _manifest_bytes(ws: Workspace, job_id: str) -> bytes:
    from dgml_core import layout

    return ws.blobs.get_blob(layout.batch_job_manifest_key(job_id))


# ---- status -------------------------------------------------------------------------


def test_status_polls_read_only_and_derives_ready(ws: Workspace) -> None:
    backend = _install(FakeBackend(_answer, provider="anthropic", polls_until_ended=2))
    job_id = _paused(ws, backend)  # poll 1: running
    before = _manifest_bytes(ws, job_id)

    first = job_status(ws, job_id)  # poll 2: running
    assert first["status"] == "pending" and first["lease"] is None
    (entry,) = first["batches"]
    assert entry["state"] == "open" and entry["done"] is False and entry["requests"] == 2
    assert job_status(ws, job_id)["status"] == "ready"  # poll 3: ended
    assert _manifest_bytes(ws, job_id) == before  # nothing written
    (listed,) = list_job_summaries(ws)
    assert listed["status"] == "pending"  # stored, not derived
    assert list(listed) == [
        "job_id",
        "command",
        "status",
        "created_at",
        "updated_at",
        "runs",
        "requests_in_flight",
        "error",
    ]


def test_status_reports_a_poll_failure_on_the_batch(ws: Workspace) -> None:
    class Broken(FakeBackend):
        def poll(self, job: Any) -> Any:
            raise RuntimeError("provider down")

    job_id = _paused(ws, _install(FakeBackend(_answer, provider="anthropic", polls_until_ended=5)))
    _install(Broken(_answer, provider="anthropic"))
    status = job_status(ws, job_id)
    assert "provider down" in status["batches"][0]["error"]
    assert status["status"] == "pending"


def test_unknown_jobs_are_not_found(ws: Workspace) -> None:
    for call in (job_status, resume_would_wait, cancel_job, delete_job, unlock_job):
        with pytest.raises(BatchJobNotFound):
            call(ws, "bj_000000000000")


# ---- the resume short-circuit ---------------------------------------------------------


def test_resume_would_wait_while_no_batch_has_ended(ws: Workspace) -> None:
    backend = _install(FakeBackend(_answer, provider="anthropic", polls_until_ended=3))
    job_id = _paused(ws, backend)  # poll 1
    before = _manifest_bytes(ws, job_id)
    waiting = resume_would_wait(ws, job_id)  # poll 2: still running
    assert waiting == {
        "job_id": job_id,
        "status": "pending",
        "command": "test run",
        "submitted_batches": 1,
        "requests_in_flight": 2,
        "resume": f"dgml batch resume {job_id}",
    }
    assert _manifest_bytes(ws, job_id) == before  # read-only
    assert resume_would_wait(ws, job_id) is not None  # poll 3: still running
    assert resume_would_wait(ws, job_id) is None  # poll 4: ended — re-run it
    assert len(backend.submitted) == 1


def test_resume_would_wait_reports_the_jobs_deadline(ws: Workspace) -> None:
    backend = _install(FakeBackend(_answer, provider="anthropic", polls_until_ended=50))
    job_id = _paused(ws, backend)
    store = BatchJobStore(ws, job_id)
    manifest = store.load()
    manifest.deadline = {"at": "2999-01-01T00:00:00Z", "seconds": 3600.0, "runs": {}}
    store.save(manifest)
    waiting = resume_would_wait(ws, job_id)
    assert waiting is not None
    # Appended after the pre-deadline keys; the same block `batch status` shows.
    assert list(waiting)[-1] == "deadline"
    assert waiting["deadline"] == {"at": "2999-01-01T00:00:00Z", "expired": False}
    assert waiting["deadline"] == job_status(ws, job_id)["deadline"]


def test_resume_reruns_a_blocking_job_even_while_its_batch_runs(ws: Workspace) -> None:
    """No --no-wait: the re-run waits on the batch instead of pausing."""
    backend = _install(FakeBackend(_answer, provider="anthropic", polls_until_ended=5))
    job_id = _paused(ws, backend, argv=("run",))
    assert resume_would_wait(ws, job_id) is None


def test_resume_reruns_when_leased_polling_fails_or_not_pending(ws: Workspace) -> None:
    backend = _install(FakeBackend(_answer, provider="anthropic", polls_until_ended=50))
    job_id = _paused(ws, backend)
    store = BatchJobStore(ws, job_id)

    # Another process holds the lease: the re-run reports BATCH_JOB_BUSY.
    store.acquire_lease("elsewhere")
    assert resume_would_wait(ws, job_id) is None
    store.break_lease()
    assert resume_would_wait(ws, job_id) is not None

    # A poll that fails: the re-run decides (and reports) what to do.
    class Broken(FakeBackend):
        def poll(self, job: Any) -> Any:
            raise RuntimeError("provider down")

    _install(Broken(_answer, provider="anthropic"))
    assert resume_would_wait(ws, job_id) is None
    _install(backend)

    # Canceled (failed, nothing open): resubmitting is the re-run's job.
    cancel_job(ws, job_id)
    assert resume_would_wait(ws, job_id) is None


# ---- cancel / delete / unlock / prune -------------------------------------------------


def test_cancel_drops_open_batches_and_fails_the_job(ws: Workspace) -> None:
    fail = {"on": True}

    class Backend(FakeBackend):
        def cancel(self, job: Any) -> None:
            if fail["on"]:
                raise RuntimeError("cancel refused")
            super().cancel(job)

    backend = _install(Backend(_answer, provider="anthropic", polls_until_ended=5))
    job_id = _paused(ws, backend)

    refused = cancel_job(ws, job_id)
    assert refused["canceled"] is False and refused["status"] == "pending"
    assert refused["batches"][0]["state"] == "open"
    assert "cancel refused" in refused["batches"][0]["error"]

    fail["on"] = False
    canceled = cancel_job(ws, job_id)
    assert canceled["canceled"] is True and canceled["status"] == "failed"
    assert canceled["error"] == "canceled with `dgml batch cancel`"
    assert canceled["batches"][0]["state"] == "dropped"
    assert backend.canceled
    assert BatchJobStore(ws, job_id).lease_info() is None  # lease released


class _FinishedBeforeCancel(FakeBackend):
    """A provider that refuses to cancel a batch that has already ended, as
    Anthropic does (HTTP 400 "cannot be canceled because it has already
    finished processing")."""

    def cancel(self, job: Any) -> None:
        raise RuntimeError(
            f"HTTP 400: Batch {job.job_id} cannot be canceled because it has already "
            "finished processing"
        )


def test_cancel_collects_a_batch_that_ended_before_its_cancel(ws: Workspace) -> None:
    """Live (Anthropic, 2026-10-01): a batch that ended between the last poll
    and `batch cancel` refused the cancel, the record stayed `open` with an
    error and `canceled: false` on every retry, and `delete --force` could
    never succeed. An ended batch has nothing to cancel: collect it instead."""
    backend = _install(_FinishedBeforeCancel(_answer, provider="anthropic", polls_until_ended=1))
    job_id = _paused(ws, backend)  # the pausing run's one poll saw it running

    payload = cancel_job(ws, job_id)
    assert payload["canceled"] is True and payload["status"] == "failed"
    (entry,) = payload["batches"]
    assert entry["state"] == "collected" and "error" not in entry
    store = BatchJobStore(ws, job_id)
    assert not store.load().open_records()
    assert len(store.response_keys()) == 2  # paid results kept for a resume
    assert backend.cleaned == [entry["batch_id"]]


def test_a_batch_collected_at_cancel_counts_in_the_jobs_cost(ws: Workspace) -> None:
    """The results `batch cancel` collects from an ended batch were billed: they
    count in the job's totals (``batch_ok``, ``cost_usd``) like a batch a run
    collects. A resume only replays them, and replayed responses are never
    counted, so uncounted at the cancel they would be missing for good."""
    backend = _install(_FinishedBeforeCancel(_answer, provider="anthropic", polls_until_ended=1))
    job_id = _paused(ws, backend)

    cancel_job(ws, job_id)

    manifest = BatchJobStore(ws, job_id).load()
    key = manifest.provider_batches[0]["stats_key"]
    runs = list(manifest.run_stats.get(key, {}).values())
    assert len(runs) == 1  # folded into the run that submitted it, not one more run
    assert sum(int(r.get("batch_ok", 0)) for r in runs) == 2
    assert sum(float(r.get("cost_usd") or 0) for r in runs) == pytest.approx(0.2)
    assert sum(float(r.get("standard_cost_usd") or 0) for r in runs) == pytest.approx(0.4)


# A batch's results count in the job's totals exactly once, whichever of a run
# or `batch cancel` collected them and wherever a run was killed around it.

_WAVE = {
    "a-1": {"model": _MODEL, "messages": [{"role": "user", "content": "a"}]},
    "b-1": {"model": _MODEL, "messages": [{"role": "user", "content": "b"}]},
}


def _replay_executor(backend: FakeBackend, session: Any) -> ReplayExecutor:
    return ReplayExecutor(
        backend, session=session, model=_MODEL, api_base=None, poll_interval_s=0,
        sleep=lambda _s: None,
    )  # fmt: skip


def _kill(ws: Workspace, session: Any) -> None:
    """``kill -9``: no close, no further manifest write; the lease is broken."""
    from dgml_core import llm, usage
    from dgml_core.batch import jobs as jobs_mod

    session._heartbeat_stop.set()
    jobs_mod._ACTIVE = None
    usage._BUFFER = None
    llm._SYNC_RECORDER = None
    BatchJobStore(ws, session.job_id).break_lease()


def _resume_totals(ws: Workspace, backend: FakeBackend, job_id: str) -> dict[str, Any]:
    """A final resume that replays the whole wave; the job-wide stats block."""
    session = start_session(ws, command="test run", argv=["run"], job_id=job_id, wait=False)
    executor = _replay_executor(backend, session)
    executor.run_wave(_WAVE)
    stats = executor.stats.to_json()
    session.close(None)
    return stats


def _assert_counted_once(stats: dict[str, Any], backend: FakeBackend) -> None:
    assert stats["this_run"]["replayed"] == 2  # nothing re-collected, nothing resubmitted
    assert stats["batch_ok"] == 2
    assert stats["cost_usd"] == pytest.approx(0.2)
    assert stats["standard_cost_usd"] == pytest.approx(0.4)
    assert stats["sync_fallbacks"] == 0 and stats["batches"] == 1
    assert len(backend.submitted) == 1


def test_a_batch_collected_at_cancel_is_counted_once_after_a_resume(ws: Workspace) -> None:
    backend = _install(_FinishedBeforeCancel(_answer, provider="anthropic", polls_until_ended=1))
    job_id = _paused(ws, backend)
    assert cancel_job(ws, job_id)["batches"][0]["state"] == "collected"

    _assert_counted_once(_resume_totals(ws, backend, job_id), backend)


def test_a_run_killed_after_collecting_is_not_recounted_by_cancel(ws: Workspace) -> None:
    """The run that collected the batch counted it and recorded it collected
    before it died; `batch cancel` then has nothing open to collect, so it adds
    nothing, and the resume replays without counting."""
    backend = _install(_FinishedBeforeCancel(_answer, provider="anthropic", polls_until_ended=1))
    job_id = _paused(ws, backend)
    collector = start_session(ws, command="test run", argv=["run"], job_id=job_id, wait=False)
    _replay_executor(backend, collector).run_wave(_WAVE)
    _kill(ws, collector)

    before = BatchJobStore(ws, job_id).load().run_stats
    payload = cancel_job(ws, job_id)
    assert payload["batches"][0]["state"] == "collected" and payload["canceled"] is True
    assert BatchJobStore(ws, job_id).load().run_stats == before  # cancel added nothing

    _assert_counted_once(_resume_totals(ws, backend, job_id), backend)


def test_a_run_killed_before_recording_its_collect_is_counted_by_cancel(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run collected (stored the responses, counted them in memory) but
    died before the manifest write that records the batch collected: none of
    its counts reached the manifest and the record is still open, so `batch
    cancel` collects and counts it — once."""
    from dgml_core.batch.jobs import JobSession

    backend = _install(_FinishedBeforeCancel(_answer, provider="anthropic", polls_until_ended=1))
    job_id = _paused(ws, backend)
    collector = start_session(ws, command="test run", argv=["run"], job_id=job_id, wait=False)

    class Killed(BaseException):
        pass

    def die(_self: Any) -> None:
        raise Killed

    monkeypatch.setattr(JobSession, "persist", die)
    with pytest.raises(Killed):
        _replay_executor(backend, collector).run_wave(_WAVE)
    monkeypatch.undo()
    _kill(ws, collector)
    stored = BatchJobStore(ws, job_id).load()
    assert len(stored.open_records()) == 1
    assert len(BatchJobStore(ws, job_id).response_keys()) == 2  # it did collect
    assert backend.cleaned == []
    assert all(
        int(r.get("batch_ok", 0)) == 0 for rs in stored.run_stats.values() for r in rs.values()
    )

    assert cancel_job(ws, job_id)["batches"][0]["state"] == "collected"
    _assert_counted_once(_resume_totals(ws, backend, job_id), backend)


def test_a_batch_collected_at_a_runs_close_is_counted_once_after_a_resume(
    ws: Workspace,
) -> None:
    """A run that ends (other than by pausing) with a batch still open settles
    it: the batch had ended, so the close collects it and stores its responses
    for a resume. Those were billed, so they count in the job's totals there —
    ``batch_ok`` as well as the cost — and the resume only replays them."""
    backend = _install(_FinishedBeforeCancel(_answer, provider="anthropic", polls_until_ended=1))
    job_id = _paused(ws, backend)
    closer = start_session(ws, command="test run", argv=["run"], job_id=job_id, wait=False)
    closer.close(RuntimeError("the run failed before it waited on the batch"))
    stored = BatchJobStore(ws, job_id).load()
    assert [r["state"] for r in stored.provider_batches] == ["collected"]

    _assert_counted_once(_resume_totals(ws, backend, job_id), backend)


def test_delete_force_accepts_a_batch_that_ended_before_its_cancel(ws: Workspace) -> None:
    backend = _install(_FinishedBeforeCancel(_answer, provider="anthropic", polls_until_ended=1))
    job_id = _paused(ws, backend)
    deleted = delete_job(ws, job_id, force=True)
    assert deleted["deleted"] is True and deleted["batches"][0]["state"] == "collected"
    assert not BatchJobStore(ws, job_id).exists()


def test_delete_refuses_open_batches_unless_forced(ws: Workspace) -> None:
    backend = _install(FakeBackend(_answer, provider="anthropic", polls_until_ended=5))
    job_id = _paused(ws, backend)
    with pytest.raises(BatchJobInvalid):
        delete_job(ws, job_id)
    assert BatchJobStore(ws, job_id).exists()
    deleted = delete_job(ws, job_id, force=True)
    assert deleted["deleted"] is True and deleted["batches"][0]["state"] == "dropped"
    assert not BatchJobStore(ws, job_id).exists()


def test_delete_keeps_the_job_when_a_batch_cannot_be_canceled(ws: Workspace) -> None:
    class Backend(FakeBackend):
        def cancel(self, job: Any) -> None:
            raise RuntimeError("cancel refused")

    job_id = _paused(ws, _install(Backend(_answer, provider="anthropic", polls_until_ended=5)))
    with pytest.raises(BatchExecutionFailed):
        delete_job(ws, job_id, force=True)
    assert BatchJobStore(ws, job_id).load().open_records()


def test_unlock_breaks_any_lease(ws: Workspace) -> None:
    job_id = _paused(ws, _install(FakeBackend(_answer, provider="anthropic", polls_until_ended=5)))
    assert unlock_job(ws, job_id) == {"job_id": job_id, "unlocked": False}
    BatchJobStore(ws, job_id).acquire_lease("elsewhere")
    assert unlock_job(ws, job_id) == {"job_id": job_id, "unlocked": True}


def test_prune_deletes_only_finished_unleased_jobs(ws: Workspace) -> None:
    backend = _install(FakeBackend(_answer, provider="anthropic", polls_until_ended=5))
    pending = _paused(ws, backend)
    failed = _paused(ws, backend)
    cancel_job(ws, failed)  # failed, nothing open
    too_recent = prune_jobs(ws, older_than_days=1)
    assert too_recent["deleted"] == [] and sorted(too_recent["kept"]) == sorted([failed, pending])
    BatchJobStore(ws, failed).acquire_lease("elsewhere")
    assert prune_jobs(ws)["deleted"] == []
    BatchJobStore(ws, failed).break_lease()
    assert prune_jobs(ws) == {"deleted": [failed], "kept": [pending]}


# ---- provider-side cleanup of canceled batches ------------------------------------


class _SlowCancel(FakeBackend):
    """A provider whose canceled batch stays ``canceling`` (reported running)
    for *settle_polls* polls after the cancel, as an Anthropic batch does."""

    def __init__(self, *args: Any, settle_polls: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.settle_polls = settle_polls
        self.fail_cleanup: Exception | None = None

    def poll(self, job: Any) -> Any:
        from dgml_core.batch.types import BatchState, BatchStatus

        if job.job_id in self.canceled and self.settle_polls > 0:
            self.settle_polls -= 1
            return BatchStatus(state=BatchState.RUNNING)
        return super().poll(job)

    def cleanup(self, job: Any) -> None:
        if self.fail_cleanup is not None:
            raise self.fail_cleanup
        super().cleanup(job)


def _slow(settle_polls: int, *, polls_until_ended: int = 5) -> _SlowCancel:
    backend = _SlowCancel(
        _answer,
        provider="anthropic",
        polls_until_ended=polls_until_ended,
        settle_polls=settle_polls,
    )
    _install(backend)
    return backend


def test_cancel_deletes_a_batch_whose_cancel_settled_at_once(ws: Workspace) -> None:
    backend = _slow(0)
    job_id = _paused(ws, backend)
    payload = cancel_job(ws, job_id)
    (entry,) = payload["batches"]
    assert entry["state"] == "dropped" and entry["cleanup"] == "done"
    assert backend.cleaned == backend.canceled
    (record,) = BatchJobStore(ws, job_id).load().provider_batches
    assert record["cleanup"] == "done"


def test_a_still_canceling_batch_is_cleaned_by_a_later_prune_not_by_status(
    ws: Workspace,
) -> None:
    backend = _slow(3)
    job_id = _paused(ws, backend)
    (entry,) = cancel_job(ws, job_id)["batches"]
    assert entry["cleanup"] == "pending"  # canceling: not deletable yet
    assert backend.cleaned == []
    store = BatchJobStore(ws, job_id)
    assert store.load().provider_batches[0]["cleanup"] == "pending"

    # status only reports the owed cleanup: read-only, it deletes nothing.
    (entry,) = job_status(ws, job_id)["batches"]
    assert entry["cleanup"] == "pending"
    assert backend.cleaned == []

    # prune keeps the job while its batch is still owed a cleanup...
    for _ in range(2):
        assert prune_jobs(ws) == {"deleted": [], "kept": [job_id], "cleanup_pending": [job_id]}
    assert backend.cleaned == []
    # ...and deletes it once the cleanup went through.
    assert prune_jobs(ws) == {"deleted": [job_id], "kept": []}
    assert backend.cleaned == backend.canceled


def test_cancel_retries_an_owed_cleanup_and_reports_a_failure(ws: Workspace) -> None:
    backend = _slow(1)
    job_id = _paused(ws, backend)
    assert cancel_job(ws, job_id)["batches"][0]["cleanup"] == "pending"
    backend.fail_cleanup = RuntimeError("HTTP 500: boom")
    (entry,) = cancel_job(ws, job_id)["batches"]  # the job has no open batch now
    assert entry["cleanup"].startswith("cleanup failed:") and "boom" in entry["cleanup"]
    assert BatchJobStore(ws, job_id).load().provider_batches[0]["cleanup"] == "pending"
    backend.fail_cleanup = None
    (entry,) = cancel_job(ws, job_id)["batches"]
    assert entry["cleanup"] == "done"
    assert BatchJobStore(ws, job_id).load().provider_batches[0]["cleanup"] == "done"
    assert job_status(ws, job_id)["batches"][0]["cleanup"] == "done"


def test_a_resumed_run_cleans_a_batch_an_earlier_cancel_left_canceling(ws: Workspace) -> None:
    backend = _slow(1)
    job_id = _paused(ws, backend)
    cancel_job(ws, job_id)
    assert backend.cleaned == []
    session = start_session(ws, command="test run", argv=["run"], job_id=job_id, wait=True)
    session.close(None)
    assert backend.cleaned == backend.canceled
    (record,) = BatchJobStore(ws, job_id).load().provider_batches
    assert record["cleanup"] == "done"
