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

"""Lease fencing: one runner drives a job at a time, even after an ``unlock``.

A live runner whose lease is broken (``dgml batch unlock``) or taken over by
another process must find out and stop — before its next provider submission,
synchronous call or manifest write — rather than keep driving the job next to
the new holder. Two sessions of one job are simulated in one process by
swapping the process-global active session (a second process in reality).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core import llm, usage
from dgml_core.batch import (
    BatchJobStore,
    FakeBackend,
    ReplayExecutor,
    active_session,
    fake_model_response,
    start_session,
)
from dgml_core.batch import jobs as jobs_mod
from dgml_core.batch.jobs import (
    RECORD_COLLECTED,
    RECORD_OPEN,
    STATUS_COMPLETED,
    STATUS_PENDING,
    JobSession,
    unlock_job,
)
from dgml_core.errors import BatchJobBusy, BatchJobLeaseLost, BatchPending
from dgml_core.storage import Workspace
from dgml_core.usage import (
    UsageEvent,
    extract_cost_and_tokens,
    read_events,
    record_usage,
)


def _kwargs(text: str) -> dict[str, Any]:
    return {
        "model": "anthropic/claude-haiku-4-5",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 64,
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
    if session is not None:
        session.close(None)
    jobs_mod._ACTIVE = None
    assert llm._SYNC_RECORDER is None


def _open(ws: Workspace, job_id: str | None = None, *, wait: bool = True) -> JobSession:
    return start_session(ws, command="test run", argv=["test", "run"], job_id=job_id, wait=wait)


@contextmanager
def _other_process() -> Iterator[None]:
    """Run the body as a second process would: with no active session of its
    own (the first session stays open and is restored afterwards)."""
    first = jobs_mod._ACTIVE
    recorder = llm._SYNC_RECORDER
    rows = usage._BUFFER
    jobs_mod._ACTIVE = None
    usage._BUFFER = None
    try:
        yield
    finally:
        jobs_mod._ACTIVE = first
        llm._SYNC_RECORDER = recorder
        usage._BUFFER = rows


def _switch_to(session: JobSession) -> None:
    """Make *session* (opened in :func:`_other_process`) this process's own,
    once the first session has closed."""
    jobs_mod._ACTIVE = session
    llm._SYNC_RECORDER = session.record_sync
    usage._BUFFER = session._usage_held


def _executor(backend: FakeBackend, session: JobSession) -> ReplayExecutor:
    return ReplayExecutor(
        backend,
        session=session,
        model="anthropic/claude-haiku-4-5",
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


def test_a_live_lease_still_refuses_a_second_session(ws: Workspace) -> None:
    first = _open(ws)
    with _other_process(), pytest.raises(BatchJobBusy):
        _open(ws, first.job_id)
    first.close(None)


def test_unlock_mid_run_fences_the_runner_and_the_new_holder_finishes_the_job(
    ws: Workspace,
) -> None:
    backend = FakeBackend(_answer, provider="anthropic")
    first = _open(ws)
    a = _executor(backend, first)
    _row(ws, a.run_wave({"w1-1": _kwargs("one")})["w1-1"])
    assert len(backend.submitted) == 1

    # The operator unlocks the live run, and a resume takes the job over.
    assert unlock_job(ws, first.job_id)["unlocked"] is True
    with _other_process():
        second = _open(ws, first.job_id)
    store = BatchJobStore(ws, first.job_id)
    assert store.lease_info()["held_by"] == second.owner  # type: ignore[index]

    # The first runner reaches its next wave: nothing is submitted.
    with pytest.raises(BatchJobLeaseLost) as lost:
        a.run_wave({"w2-1": _kwargs("two")})
    assert lost.value.code == "BATCH_JOB_BUSY"
    assert "lost the job's lease" in str(lost.value)
    assert len(backend.submitted) == 1
    before = store.load().to_json()
    first.close(lost.value)
    # Fenced: no status written, nothing settled, the holder's lease kept, no
    # rows billed by the fenced run.
    after = store.load().to_json()
    assert {k: v for k, v in after.items() if k != "updated_at"} == {
        k: v for k, v in before.items() if k != "updated_at"
    }
    assert store.lease_info()["held_by"] == second.owner  # type: ignore[index]
    assert read_events(ws) == []

    # The holder replays wave 1 from the store and submits wave 2 once.
    _switch_to(second)
    b = _executor(backend, second)
    _row(ws, b.run_wave({"w1-1": _kwargs("one")})["w1-1"])
    _row(ws, b.run_wave({"w2-1": _kwargs("two")})["w2-1"])
    second.close(None)
    assert len(backend.submitted) == 2
    assert store.load().status == STATUS_COMPLETED
    assert [r["cost_usd"] for r in read_events(ws)] == [0.25, 0.25]  # each billed once


def test_the_heartbeat_notices_a_broken_lease_and_never_takes_it_back(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs_mod, "LEASE_HEARTBEAT_S", 0.01)
    session = _open(ws)
    store = BatchJobStore(ws, session.job_id)
    store.break_lease()
    deadline = time.monotonic() + 5
    while session.lease_lost is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.lease_lost is not None
    assert store.lease_info() is None  # not re-created by the renewal
    session.close(None)
    assert store.load().status == STATUS_PENDING  # the fenced close wrote nothing


def test_a_blocking_wait_stops_when_the_lease_is_taken_and_leaves_the_batch_open(
    ws: Workspace,
) -> None:
    session = _open(ws)
    store = BatchJobStore(ws, session.job_id)

    class TakenWhileWaiting(FakeBackend):
        def poll(self, job: Any) -> Any:
            if store.owns_lease(session.owner):
                store.break_lease()
                store.acquire_lease("another-process")
            return super().poll(job)

    backend = TakenWhileWaiting(_answer, provider="anthropic", polls_until_ended=50)
    with pytest.raises(BatchJobLeaseLost):
        _executor(backend, session).run_wave({"x-1": _kwargs("a")})
    session.close(session.lease_lost)
    assert backend.canceled == []  # the holder's batch to collect, not canceled
    (record,) = store.load().provider_batches
    assert record["state"] == RECORD_OPEN
    assert store.lease_info()["held_by"] == "another-process"  # type: ignore[index]

    store.break_lease()  # the other process is done with it
    resumed = _open(ws, session.job_id)
    out = _executor(backend, resumed).run_wave({"x-1": _kwargs("a")})
    resumed.close(None)
    assert str(out["x-1"].choices[0].message.content) == "reply:a"
    assert len(backend.submitted) == 1  # collected, never resubmitted
    assert [r["state"] for r in store.load().provider_batches] == [RECORD_COLLECTED]


def test_a_batch_created_as_the_lease_is_lost_is_still_recorded(ws: Workspace) -> None:
    session = _open(ws, wait=False)
    store = BatchJobStore(ws, session.job_id)

    class LostDuringSubmit(FakeBackend):
        def submit(self, requests: Any) -> Any:
            job = super().submit(requests)
            store.break_lease()  # unlocked while the create was in flight
            return job

    backend = LostDuringSubmit(_answer, provider="anthropic", polls_until_ended=5)
    with pytest.raises(BatchJobLeaseLost):
        _executor(backend, session).run_wave({"x-1": _kwargs("a"), "y-1": _kwargs("b")})
    session.close(session.lease_lost)
    (record,) = store.load().provider_batches
    assert record["state"] == RECORD_OPEN and len(record["keys"]) == 2
    assert len(backend.submitted) == 1

    resumed = _open(ws, session.job_id, wait=False)
    with pytest.raises(BatchPending):
        _executor(backend, resumed).run_wave({"x-1": _kwargs("a"), "y-1": _kwargs("b")})
    assert len(backend.submitted) == 1  # the recorded batch is polled, not resubmitted
    resumed.close(BatchPending(resumed.job_id, submitted_batches=1, requests_in_flight=2))


def test_a_fenced_run_makes_no_synchronous_call(ws: Workspace) -> None:
    session = _open(ws)
    BatchJobStore(ws, session.job_id).break_lease()
    with (
        patch("litellm.completion", side_effect=AssertionError("no call expected")),
        pytest.raises(BatchJobLeaseLost),
    ):
        llm._completion_with_retry(_kwargs("a"))
    session.close(session.lease_lost)


def test_a_run_that_loses_the_lease_after_its_last_call_still_ends_fenced(
    ws: Workspace,
) -> None:
    backend = FakeBackend(_answer, provider="anthropic")
    session = _open(ws)
    _row(ws, _executor(backend, session).run_wave({"x-1": _kwargs("a")})["x-1"])
    store = BatchJobStore(ws, session.job_id)
    store.break_lease()
    store.acquire_lease("another-process")
    session.close(None)
    assert session.lease_lost is not None
    assert store.exists()  # a silent job is not deleted by a fenced run
    assert read_events(ws) == []  # its rows dropped: the holder bills them
    assert store.lease_info()["held_by"] == "another-process"  # type: ignore[index]
