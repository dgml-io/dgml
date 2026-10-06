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

"""Batch-job durability: what a job records before it touches the provider.

Covers a provider batch deleted before its record said so (and a batch the
provider no longer knows, :class:`BatchNotFound`), usage rows written only once
the billed keys they stand for are durable, ``prune`` under the job's lease,
a read-only ``batch status``, lease renewal that never revives a lost lease,
and usage rows for responses a job paid for but never used.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import patch

import pytest
from dgml_core import layout, llm, usage
from dgml_core.batch import (
    BatchJobStore,
    FakeBackend,
    ReplayExecutor,
    active_session,
    cancel_job,
    delete_job,
    fake_model_response,
    job_status,
    prune_jobs,
    register_backend,
    start_session,
)
from dgml_core.batch import jobs as jobs_mod
from dgml_core.batch import registry as batch_registry
from dgml_core.batch.jobs import (
    RECORD_COLLECTED,
    RECORD_DROPPED,
    STATUS_COMPLETED,
    JobSession,
    unlock_job,
)
from dgml_core.batch.types import BatchJob, BatchNotFound, BatchRequest
from dgml_core.errors import BatchJobBusy, BatchJobLeaseLost, BatchPending
from dgml_core.storage import Workspace
from dgml_core.usage import UsageEvent, extract_cost_and_tokens, read_events, record_usage

_MODEL = "anthropic/claude-haiku-4-5"


def _kwargs(text: str) -> dict[str, Any]:
    return {"model": _MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 64}


def _answer(request: Any) -> Any:
    text = request.kwargs["messages"][0]["content"]
    return fake_model_response(
        f"reply:{text}",
        usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        cost=0.25,
    )


class _GoneBackend(FakeBackend):
    """A fake whose ``cleanup`` really deletes the batch: afterwards (or once
    :meth:`forget` drops it) poll, results and cancel raise
    :class:`BatchNotFound`, as a provider's 404 does."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.before_poll: Any = None

    def forget(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def cleanup(self, job: BatchJob) -> None:
        super().cleanup(job)
        self.forget(job.job_id)

    def poll(self, job: BatchJob) -> Any:
        if self.before_poll is not None:
            self.before_poll(job)
        return super().poll(job)

    def _state(self, job: BatchJob) -> Any:
        try:
            return self._jobs[job.job_id]
        except KeyError:
            raise BatchNotFound(f"HTTP 404: batch {job.job_id} not found") from None


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "ws"
    root.mkdir()
    return Workspace(root=root)


@pytest.fixture(autouse=True)
def _isolation() -> Iterator[None]:
    saved = dict(batch_registry._REGISTRY)
    try:
        yield
    finally:
        session = active_session()
        if session is not None:
            session.close(None)
        jobs_mod._ACTIVE = None
        usage._BUFFER = None
        llm._SYNC_RECORDER = None
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


_B = TypeVar("_B", bound=FakeBackend)


def _install(backend: _B) -> _B:
    register_backend("anthropic", lambda _cfg: backend)
    return backend


def _open(ws: Workspace, job_id: str | None = None, *, wait: bool = True) -> JobSession:
    argv = ["test", "run"] if wait else ["test", "run", "--no-wait"]
    return start_session(ws, command="test run", argv=argv, job_id=job_id, wait=wait)


def _executor(backend: FakeBackend, session: JobSession) -> ReplayExecutor:
    return ReplayExecutor(
        backend,
        session=session,
        model=_MODEL,
        api_base=None,
        poll_interval_s=0,
        sleep=lambda _s: None,
    )


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


_WAVE = {"a-1": _kwargs("a"), "b-1": _kwargs("b")}


def _paused(ws: Workspace, backend: FakeBackend) -> str:
    """A pending job with one open provider batch carrying ``_WAVE``."""
    session = _open(ws, wait=False)
    with pytest.raises(BatchPending) as pending:
        _executor(backend, session).run_wave(_WAVE)
    session.close(pending.value)
    return session.job_id


# ---- J1: a batch is deleted at the provider only after its record says so ------------


def test_j1_a_run_fenced_mid_wait_leaves_a_job_its_resume_can_finish(ws: Workspace) -> None:
    backend = _install(_GoneBackend(_answer, provider="anthropic", polls_until_ended=1))
    first = _open(ws)
    store = BatchJobStore(ws, first.job_id)

    def takeover(_job: BatchJob) -> None:
        # The batch ends at this poll; meanwhile an operator unlocked the run
        # and another runner took the job.
        if backend.polls == 1:
            unlock_job(ws, first.job_id)
            store.acquire_lease("intruder")

    backend.before_poll = takeover
    with pytest.raises(BatchJobLeaseLost) as lost:
        _executor(backend, first).run_wave(_WAVE)
    first.close(lost.value)
    backend.before_poll = None

    # The intruder goes away; the job is resumed.
    store.break_lease()
    second = _open(ws, first.job_id)
    served = _executor(backend, second).run_wave(_WAVE)
    second.close(None)
    assert {cid: r.choices[0].message.content for cid, r in served.items()} == {
        "a-1": "reply:a",
        "b-1": "reply:b",
    }
    assert len(backend.submitted) == 1  # nothing paid for twice
    assert store.load().status == STATUS_COMPLETED


def test_j1_a_gone_open_batch_whose_responses_are_stored_counts_as_collected(
    ws: Workspace,
) -> None:
    backend = _install(_GoneBackend(_answer, provider="anthropic", polls_until_ended=5))
    job_id = _paused(ws, backend)
    store = BatchJobStore(ws, job_id)
    (record,) = store.load().provider_batches
    # Its responses reached the store, then the batch was deleted at the provider.
    job = BatchJob.from_json(record["job"])
    for provider_cid, key in record["keys"].items():
        req = next(r for r in backend.submitted[0] if r.custom_id == provider_cid)
        store.put_response(key, _answer(req))
    backend.forget(job.job_id)

    session = _open(ws, job_id)
    served = _executor(backend, session).run_wave(_WAVE)
    session.close(None)
    assert set(served) == {"a-1", "b-1"}
    manifest = store.load()
    assert manifest.status == STATUS_COMPLETED, manifest.error
    assert manifest.provider_batches[0]["state"] == RECORD_COLLECTED
    assert len(backend.submitted) == 1


def test_j1_a_gone_open_batch_without_stored_responses_is_dropped_and_rerun(
    ws: Workspace,
) -> None:
    backend = _install(_GoneBackend(_answer, provider="anthropic", polls_until_ended=5))
    job_id = _paused(ws, backend)
    store = BatchJobStore(ws, job_id)
    gone_id = store.load().provider_batches[0]["job"]["job_id"]
    backend.forget(gone_id)
    backend._polls_until_ended = 0

    session = _open(ws, job_id)
    served = _executor(backend, session).run_wave(_WAVE)
    session.close(None)
    assert {cid: r.choices[0].message.content for cid, r in served.items()} == {
        "a-1": "reply:a",
        "b-1": "reply:b",
    }
    assert len(backend.submitted) == 2  # re-run as a new batch
    manifest = store.load()
    assert manifest.status == STATUS_COMPLETED, manifest.error
    gone = next(r for r in manifest.provider_batches if r["job"]["job_id"] == gone_id)
    assert gone["state"] == RECORD_DROPPED and gone["cleanup"] == "done"


def test_j1_cancel_and_delete_force_accept_a_gone_open_batch(ws: Workspace) -> None:
    backend = _install(_GoneBackend(_answer, provider="anthropic", polls_until_ended=5))
    job_id = _paused(ws, backend)
    backend.forget(BatchJobStore(ws, job_id).load().provider_batches[0]["job"]["job_id"])
    (entry,) = job_status(ws, job_id)["batches"]
    assert entry["done"] is True and entry["gone"] is True
    payload = cancel_job(ws, job_id)
    assert payload["canceled"] is True
    assert payload["batches"][0]["state"] == RECORD_DROPPED

    job_id = _paused(ws, backend)
    backend.forget(BatchJobStore(ws, job_id).load().provider_batches[0]["job"]["job_id"])
    assert delete_job(ws, job_id, force=True)["deleted"] is True
    assert not BatchJobStore(ws, job_id).exists()


def test_j1_settling_a_gone_open_batch_at_close_does_not_fail_the_job(ws: Workspace) -> None:
    backend = _install(_GoneBackend(_answer, provider="anthropic", polls_until_ended=5))
    job_id = _paused(ws, backend)
    store = BatchJobStore(ws, job_id)
    backend.forget(store.load().provider_batches[0]["job"]["job_id"])
    # A resume that no longer asks for those requests: the record is settled at close.
    session = _open(ws, job_id)
    session.close(None)
    manifest = store.load()
    (record,) = manifest.provider_batches
    assert record["state"] == RECORD_DROPPED
    assert "no longer exists" in record["dropped_reason"]
    assert backend.canceled == []


def test_j2_a_lease_lost_while_settling_never_bills_a_response_twice(ws: Workspace) -> None:
    backend = _install(FakeBackend(_answer, provider="anthropic"))
    first = _open(ws, wait=False)  # a kept (non-silent) job
    store = BatchJobStore(ws, first.job_id)
    _row(ws, _executor(backend, first).run_wave({"a-1": _kwargs("a")})["a-1"])

    real_settle = first._settle_open_records

    def settle_then_lose_the_lease() -> list[str]:
        # The settling network calls take long enough for the lease to go.
        out = real_settle()
        unlock_job(ws, first.job_id)
        store.acquire_lease("intruder")
        return out

    first._settle_open_records = settle_then_lose_the_lease  # type: ignore[method-assign]
    first.close(None)
    rows = [r["cost_usd"] for r in read_events(ws)]
    billed = store.load().billed
    # Rows written iff the keys they bill are recorded as billed.
    assert (rows, bool(billed)) in (([], False), ([0.25], True))

    store.break_lease()
    second = _open(ws, first.job_id)
    _row(ws, _executor(backend, second).run_wave({"a-1": _kwargs("a")})["a-1"])
    second.close(None)
    assert len(backend.submitted) == 1
    assert sum(r["cost_usd"] for r in read_events(ws)) == 0.25  # billed exactly once


def test_j2_a_run_fenced_before_it_ends_writes_no_rows(ws: Workspace) -> None:
    backend = _install(FakeBackend(_answer, provider="anthropic"))
    first = _open(ws)
    store = BatchJobStore(ws, first.job_id)
    _row(ws, _executor(backend, first).run_wave({"a-1": _kwargs("a")})["a-1"])
    unlock_job(ws, first.job_id)
    store.acquire_lease("intruder")
    first.close(None)
    assert read_events(ws) == []
    assert store.load().billed == []


# ---- J3: prune holds each job's lease --------------------------------------------


def test_j3_prune_never_deletes_a_job_a_resume_just_took(ws: Workspace) -> None:
    backend = _install(
        _GoneBackend(_answer, provider="anthropic", polls_until_ended=5, cancel_settle_polls=1)
    )
    job_id = _paused(ws, backend)
    assert cancel_job(ws, job_id)["batches"][0]["cleanup"] == "pending"
    store = BatchJobStore(ws, job_id)
    resumed: list[JobSession] = []
    refused: list[BaseException] = []

    def resume_meanwhile(_job: BatchJob) -> None:
        # While prune talks to the provider, a resume of the job starts.
        if resumed or refused:
            return
        try:
            resumed.append(_open(ws, job_id))
        except BatchJobBusy as exc:
            refused.append(exc)

    backend.before_poll = resume_meanwhile
    prune_jobs(ws)
    backend.before_poll = None
    if resumed:
        exists = store.exists()
        resumed[0]._heartbeat_stop.set()
        jobs_mod._ACTIVE = None
        usage._BUFFER = None
        llm._SYNC_RECORDER = None
        assert exists, "prune deleted a job a resume held the lease of"
    assert refused, "the resume should have been refused while prune held the lease"


def test_j3_prune_keeps_a_job_whose_lease_is_held(ws: Workspace) -> None:
    backend = _install(_GoneBackend(_answer, provider="anthropic", polls_until_ended=5))
    job_id = _paused(ws, backend)
    cancel_job(ws, job_id)
    BatchJobStore(ws, job_id).acquire_lease("someone")
    assert prune_jobs(ws) == {"deleted": [], "kept": [job_id]}
    assert BatchJobStore(ws, job_id).exists()


# ---- J4: batch status is strictly read-only ----------------------------------------


def test_j4_status_reports_an_owed_cleanup_but_never_deletes_the_batch(ws: Workspace) -> None:
    backend = _install(
        _GoneBackend(_answer, provider="anthropic", polls_until_ended=5, cancel_settle_polls=1)
    )
    job_id = _paused(ws, backend)
    assert cancel_job(ws, job_id)["batches"][0]["cleanup"] == "pending"
    before = BatchJobStore(ws, job_id).load().to_json()
    for _ in range(2):  # the cancel has settled by now: a cleanup would go through
        (entry,) = job_status(ws, job_id)["batches"]
        assert entry["state"] == RECORD_DROPPED and entry["cleanup"] == "pending"
    assert backend.cleaned == []  # nothing deleted at the provider
    assert BatchJobStore(ws, job_id).load().to_json() == before
    # The owed cleanup still happens, in cancel (or prune, or a resume).
    assert cancel_job(ws, job_id)["batches"][0]["cleanup"] == "done"
    assert backend.cleaned == backend.canceled


# ---- J5: a renewal never revives a lost lease --------------------------------------


def test_j5_a_renewal_that_races_a_takeover_reports_the_lease_lost(ws: Workspace) -> None:
    store = BatchJobStore(ws, "bj_race")
    store.acquire_lease("runner")
    lease_key = layout.batch_job_lease_key("bj_race")
    real_put = ws.blobs.put_blob

    def put_then_taken(key: str, data: bytes) -> None:
        real_put(key, data)
        if key == lease_key and b'"runner"' in data:
            # Another runner's acquire lands right after this renewal's write.
            real_put(key, b'{"owner": "new-runner", "expires": 9999999999}')

    with patch.object(ws.blobs, "put_blob", side_effect=put_then_taken):
        assert store.renew_lease("runner") is False
    assert store.lease_info()["held_by"] == "new-runner"  # type: ignore[index]


def test_j5_a_renewal_never_writes_a_lease_that_is_gone_or_someone_elses(
    ws: Workspace,
) -> None:
    store = BatchJobStore(ws, "bj_gone")
    assert store.renew_lease("runner") is False
    assert store.lease_info() is None  # nothing revived
    store.acquire_lease("other")
    assert store.renew_lease("runner") is False
    assert store.lease_info()["held_by"] == "other"  # type: ignore[index]
    store.break_lease()
    store.acquire_lease("runner", ttl_s=-1)  # expired, still ours to renew
    assert store.renew_lease("runner") is True


# ---- J6: responses paid for but never used still reach usage.jsonl ------------------


@pytest.mark.parametrize("wait", [True, False], ids=["silent", "kept"])
def test_j6_a_completed_job_bills_the_responses_it_never_used(ws: Workspace, wait: bool) -> None:
    backend = _install(FakeBackend(_answer, provider="anthropic"))
    session = _open(ws, wait=wait)
    _row(ws, _executor(backend, session).run_wave({"a-1": _kwargs("a")})["a-1"])
    # A batch whose result no run of the job ever asks for (a superseded or
    # unwanted request): collected when the run ends, never used.
    extra = BatchRequest("x-1", _kwargs("x"))
    job = backend.submit([extra])
    session.add_record(
        job, model=_MODEL, api_base=None, keys={"x-1": session.next_key(extra.kwargs)}
    )
    session.close(None)

    rows = read_events(ws)
    assert [r["cost_usd"] for r in rows] == [0.25, 0.25]
    used, unused = rows
    assert used["context"] == {}
    assert unused["operation"] == usage.OPERATION_BATCH_UNUSED
    assert unused["context"] == {"unused": True, "batch_job": session.job_id}
    assert unused["tier"] == usage.TIER_BATCH
    assert unused["model"] == _MODEL
    assert unused["prompt_tokens"] == 10


def test_j6_a_used_response_is_never_also_billed_as_unused(ws: Workspace) -> None:
    backend = _install(FakeBackend(_answer, provider="anthropic", polls_until_ended=1))
    job_id = _paused(ws, backend)
    session = _open(ws, job_id)
    ex = _executor(backend, session)
    for response in ex.run_wave(_WAVE).values():
        _row(ws, response)
    session.close(None)
    assert [r["cost_usd"] for r in read_events(ws)] == [0.25, 0.25]
    assert all(not r["context"] for r in read_events(ws))
