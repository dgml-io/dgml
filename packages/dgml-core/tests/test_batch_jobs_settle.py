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

"""Batch job mode: settling open batches when a run ends, the
job lease, merge-on-save, retention, credential references, single writes,
and the synchronous fallback before a pause."""

from __future__ import annotations

import json
import time
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
    start_session,
)
from dgml_core.batch.jobs import (
    RECORD_DROPPED,
    RECORD_OPEN,
    STATUS_COMPLETED,
    STATUS_FAILED,
    JobSession,
    Manifest,
    credential_ref,
    resolve_credential,
)
from dgml_core.batch.types import BatchJob, BatchState, BatchStatus
from dgml_core.errors import BatchJobBusy, BatchPending
from dgml_core.storage import Workspace

SECRET = "sk-DGML-REVIEW-SECRET-987654321"


def _kwargs(text: str) -> dict[str, Any]:
    return {
        "model": "anthropic/claude-haiku-4-5",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 64,
    }


def _answer(request: Any) -> Any:
    return fake_model_response(f"reply:{request.kwargs['messages'][0]['content']}", cost=0.25)


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


def _executor(backend: FakeBackend, session: JobSession) -> ReplayExecutor:
    return ReplayExecutor(
        backend,
        session=session,
        model="anthropic/claude-haiku-4-5",
        api_base=None,
        poll_interval_s=0,
        sleep=lambda _s: None,
    )


def _pause(session: JobSession) -> None:
    session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))


# ---- 1. a run never ends with a provider batch left open --------------------------


class _PollFails(FakeBackend):
    def poll(self, job: Any) -> Any:
        raise RuntimeError("provider unreachable")


def test_a_run_that_ends_with_an_unpollable_batch_cancels_it_and_fails(
    ws: Workspace,
) -> None:
    backend = _PollFails(_answer, provider="anthropic")
    session = _open(ws, wait=True)
    with pytest.raises(Exception, match="polling batch"):
        _executor(backend, session).run_wave({"x-1": _kwargs("a")})
    session.close(None)  # the caller soft-failed and the command "succeeded"
    assert backend.canceled  # it stops billing
    manifest = BatchJobStore(ws, session.job_id).load()
    assert manifest.status == STATUS_FAILED and "canceled" in (manifest.error or "")
    assert [r["state"] for r in manifest.provider_batches] == [RECORD_DROPPED]

    good = FakeBackend(_answer, provider="anthropic")
    resumed = _open(ws, session.job_id, wait=True)  # failed is resumable
    out = _executor(good, resumed).run_wave({"x-1": _kwargs("a")})
    resumed.close(None)
    assert out["x-1"].choices[0].message.content == "reply:a"
    assert len(good.submitted) == 1  # resubmitted at batch price


def test_a_finished_open_batch_is_collected_when_the_run_ends(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=1)
    session = _open(ws, wait=False)
    with pytest.raises(BatchPending):
        _executor(backend, session).run_wave({"x-1": _kwargs("a")})
    # The command swallowed nothing and finished anyway (simulated): the batch
    # has ended by now, so closing collects it rather than canceling it.
    session.close(None)
    manifest = BatchJobStore(ws, session.job_id).load()
    assert not backend.canceled
    assert manifest.status == STATUS_COMPLETED


# ---- 4. the lease and merge-on-save -------------------------------------------------


def test_a_second_runner_on_the_same_job_is_refused(ws: Workspace) -> None:
    session = _open(ws, wait=False)
    _pause(session)
    store = BatchJobStore(ws, session.job_id)
    store.acquire_lease("another-process")
    with pytest.raises(BatchJobBusy):
        _open(ws, session.job_id)
    store.release_lease("another-process")
    again = _open(ws, session.job_id)
    _pause(again)


def test_an_expired_lease_does_not_block(ws: Workspace) -> None:
    session = _open(ws, wait=False)
    _pause(session)
    store = BatchJobStore(ws, session.job_id)
    store.acquire_lease("dead-process", ttl_s=-1)  # already expired
    again = _open(ws, session.job_id)
    _pause(again)


def test_a_stale_manifest_save_never_drops_another_runs_record(ws: Workspace) -> None:
    session = _open(ws, wait=False)
    store = BatchJobStore(ws, session.job_id)
    stale = store.load()  # read before the batch is recorded
    session.add_record(
        BatchJob(provider="anthropic", job_id="pb_1", custom_ids=("a",)),
        model="anthropic/claude-haiku-4-5",
        api_base=None,
        keys={"a": "k-1"},
    )
    stale.billed = ["k-old"]
    store.save(stale)  # a writer that never saw pb_1
    merged = store.load()
    assert [r["job"]["job_id"] for r in merged.provider_batches] == ["pb_1"]
    assert "k-old" in merged.billed
    _pause(session)


# ---- 5. retention -----------------------------------------------------------------


def test_a_silent_blocking_job_is_deleted_when_it_finishes(ws: Workspace) -> None:
    session = _open(ws, wait=True)
    _executor(FakeBackend(_answer, provider="anthropic"), session).run_wave({"x-1": _kwargs("a")})
    session.close(None)
    assert list_jobs(ws) == []
    assert not (ws.root / layout.BATCHES_DIR / session.job_id).exists()


def test_a_completed_job_keeps_only_a_summary(ws: Workspace) -> None:
    session = _open(ws, wait=False)
    _executor(FakeBackend(_answer, provider="anthropic"), session).run_wave({"x-1": _kwargs("a")})
    session.close(None)
    job_dir = ws.root / layout.BATCHES_DIR / session.job_id
    assert not (job_dir / "responses").exists() or not any((job_dir / "responses").iterdir())
    (manifest,) = list_jobs(ws)
    assert manifest.status == STATUS_COMPLETED and manifest.billed == []
    assert all("keys" not in r and r["requests"] == 1 for r in manifest.provider_batches)


def test_a_no_op_run_leaves_nothing(ws: Workspace) -> None:
    for wait in (True, False):
        session = _open(ws, wait=wait)
        session.close(None)
    assert list_jobs(ws) == []


# ---- 3. credential references -------------------------------------------------------


def test_the_credential_reference_resolves_through_the_env_var_name(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DGML_REVIEW_CUSTOM_KEY", SECRET)
    ref = credential_ref("grounded", "values", "DGML_REVIEW_CUSTOM_KEY")
    assert resolve_credential(ws, ref) == SECRET
    session = _open(ws, wait=False)
    session.add_record(
        BatchJob(provider="anthropic", job_id="pb_1", custom_ids=("a",)),
        model="anthropic/claude-haiku-4-5",
        api_base=None,
        keys={"a": "k-1"},
        credential=ref,
    )
    _pause(session)
    for path in (ws.root / layout.BATCHES_DIR).rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes()
    record = BatchJobStore(ws, session.job_id).load().provider_batches[0]
    assert record["credential"] == {
        "section": "grounded",
        "field": "values",
        "env": "DGML_REVIEW_CUSTOM_KEY",
    }


# ---- 6. each batch-served response is written once ---------------------------------


def test_batch_served_responses_are_written_once(ws: Workspace) -> None:
    writes: list[str] = []
    real = BatchJobStore.put_response

    def counting(self: BatchJobStore, key: str, response: Any) -> bool:
        writes.append(key)
        return real(self, key, response)

    session = _open(ws, wait=True)
    with patch.object(BatchJobStore, "put_response", counting):
        _executor(FakeBackend(_answer, provider="anthropic"), session).run_wave(
            {"x-1": _kwargs("a"), "y-1": _kwargs("b")}
        )
    _pause(session)
    assert len(writes) == 2 and len(set(writes)) == 2


# ---- 7. rewinding many inputs writes the manifest once ------------------------------


def test_rewinding_many_inputs_persists_the_manifest_once(ws: Workspace) -> None:
    session = _open(ws, wait=False)
    saves: list[int] = []
    real = BatchJobStore.save

    def counting(self: BatchJobStore, manifest: Manifest, *, merge: bool = True) -> None:
        saves.append(1)
        real(self, manifest, merge=merge)

    with patch.object(BatchJobStore, "save", counting):
        for n in range(50):
            session.rewind_presence(f"doc{n}", present_now=False)
            session.rewind_input(f"in{n}", lambda: None)
        session.persist()
    assert len(saves) == 1
    _pause(session)


# ---- 9. failures of a collected batch run synchronously before a pause ---------------


class _SecondStaysOpen(FakeBackend):
    """Two single-request batches: the first ends at once, the second keeps
    running until ``finish_second`` is set."""

    finish_second = False

    def poll(self, job: Any) -> Any:
        if job.job_id.endswith("0002") and not _SecondStaysOpen.finish_second:
            return BatchStatus(state=BatchState.RUNNING, processing=1)
        return super().poll(job)


def test_failed_items_of_a_collected_batch_are_served_before_pausing(ws: Workspace) -> None:
    def script(request: Any) -> Any:
        if request.kwargs["messages"][0]["content"] == "a":
            return BatchItemError(request.custom_id, "invalid", "provider rejected it")
        return _answer(request)

    backend = _SecondStaysOpen(script, provider="anthropic", max_requests=1)
    wave = {"x-1": _kwargs("a"), "y-1": _kwargs("b")}
    sync_calls: list[dict[str, Any]] = []

    def sync(kwargs: dict[str, Any], **_kw: Any) -> Any:
        sync_calls.append(kwargs)
        return fake_model_response("sync:a", cost=0.5)

    with patch.object(llm, "_completion_attempts", sync):
        session = _open(ws, wait=False)
        with pytest.raises(BatchPending):
            _executor(backend, session).run_wave(wave)
        session.close(BatchPending(session.job_id, submitted_batches=1, requests_in_flight=1))
        assert [k["messages"][0]["content"] for k in sync_calls] == ["a"]

        _SecondStaysOpen.finish_second = True
        resumed = _open(ws, session.job_id, wait=False)
        out = _executor(backend, resumed).run_wave(wave)
        resumed.close(None)
    assert out["x-1"].choices[0].message.content == "sync:a"  # replayed, not resent
    assert out["y-1"].choices[0].message.content == "reply:b"
    assert len(backend.submitted) == 2  # no third batch for the failed item
    assert len(sync_calls) == 1


def test_lease_expiry_is_documented_in_seconds() -> None:
    from dgml_core.batch.jobs import LEASE_HEARTBEAT_S, LEASE_TTL_S

    assert LEASE_HEARTBEAT_S < LEASE_TTL_S / 3


def test_lease_file_holds_no_secret(ws: Workspace) -> None:
    session = _open(ws, wait=False)
    lease = json.loads((ws.root / layout.batch_job_lease_key(session.job_id)).read_text())
    assert set(lease) == {"owner", "expires"} and lease["expires"] > time.time()
    _pause(session)
    assert not (ws.root / layout.batch_job_lease_key(session.job_id)).exists()


def test_open_record_is_kept_open_on_interrupt(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=5)
    session = _open(ws, wait=False)
    with pytest.raises(BatchPending):
        _executor(backend, session).run_wave({"x-1": _kwargs("a")})
    session.close(KeyboardInterrupt())
    manifest = BatchJobStore(ws, session.job_id).load()
    assert [r["state"] for r in manifest.provider_batches] == [RECORD_OPEN]
    assert not backend.canceled
