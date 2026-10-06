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

"""Batch jobs must replay the same request bytes on every run.

The live failure: a PDF slice is not byte-reproducible across processes
(ghostscript stamps a random ``/ID``), so each resume rebuilt the first
transcription wave with new digests and submitted it again — one paid batch
per resume. These tests pin the three defenses: recorded slices, cancellation
of superseded batches, and the nondeterminism guard.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from dgml_core import llm
from dgml_core.batch import (
    BatchJobStore,
    FakeBackend,
    ReplayExecutor,
    active_session,
    fake_model_response,
    start_session,
)
from dgml_core.batch.jobs import RECORD_DROPPED, RECORD_OPEN, JobSession
from dgml_core.errors import BatchJobNondeterministic, BatchPending
from dgml_core.generation import document as document_mod
from dgml_core.generation import transcribe as transcribe_mod
from dgml_core.storage import Workspace

from .conftest import _write_text_pdf, needs_gs


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


def _open(ws: Workspace, job_id: str | None = None) -> JobSession:
    return start_session(ws, command="test run", argv=["test", "run"], job_id=job_id, wait=False)


def _executor(
    backend: FakeBackend, session: JobSession, stage: str = "transcribe"
) -> ReplayExecutor:
    ex = ReplayExecutor(
        backend,
        session=session,
        model="anthropic/claude-haiku-4-5",
        api_base=None,
        poll_interval_s=0,
        sleep=lambda _s: None,
    )
    ex.stage = stage
    return ex


def _pause(session: JobSession) -> None:
    session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))


# ---- 1. slices are recorded and replayed -------------------------------------------

_SLICE_SCRIPT = """
import hashlib, sys
from pathlib import Path
from dgml_core.batch import start_session
from dgml_core.errors import BatchPending
from dgml_core.generation import transcribe
from dgml_core.storage import Workspace

ws = Workspace(root=Path(sys.argv[1]))
pdf = Path(sys.argv[2]).read_bytes()
job_id = sys.argv[3] if len(sys.argv) > 3 else None
session = start_session(ws, command="t", argv=[], job_id=job_id, wait=False)
data = transcribe._window_slice(pdf, [0, 1], pdf_config=None, total=3)
print(session.job_id, hashlib.sha256(data).hexdigest())
session.close(BatchPending(session.job_id, submitted_batches=0, requests_in_flight=0))
"""


def _run_slice(*args: str) -> tuple[str, str]:
    out = subprocess.run(
        [sys.executable, "-c", _SLICE_SCRIPT, *args],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "random"},
    )
    job_id, digest = out.stdout.split()
    return job_id, digest


@needs_gs
def test_a_real_pdf_slice_is_identical_across_processes_in_job_mode(
    ws: Workspace, tmp_path: Path
) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_text_pdf(pdf, ["one", "two", "three"])
    # The premise: outside a job, slicing twice gives different bytes.
    plain = {
        hashlib.sha256(document_mod.slice_pdf(pdf.read_bytes(), [0, 1], total_pages=3)).hexdigest()
        for _ in range(2)
    }
    job_id, first = _run_slice(str(ws.root), str(pdf))
    _, second = _run_slice(str(ws.root), str(pdf), job_id)
    assert first == second
    if len(plain) == 1:  # pragma: no cover - a reproducible engine needs no pinning
        pytest.skip("this ghostscript already writes reproducible slices")


def test_without_a_job_the_slicer_is_called_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[int]] = []

    def fake(pdf: bytes, pages: list[int], **_kw: Any) -> bytes:
        calls.append(pages)
        return b"slice"

    monkeypatch.setattr(document_mod, "slice_pdf", fake)
    assert transcribe_mod._window_slice(b"%PDF", [0], pdf_config=None, total=1) == b"slice"
    assert transcribe_mod._window_slice(b"%PDF", [0], pdf_config=None, total=1) == b"slice"
    assert len(calls) == 2  # no recording outside a job


def test_in_a_job_a_drifting_slicer_is_called_once_per_window(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    counter = iter(range(1000))
    monkeypatch.setattr(
        document_mod, "slice_pdf", lambda _pdf, _pages, **_kw: f"slice-{next(counter)}".encode()
    )
    session = _open(ws)
    first = transcribe_mod._window_slice(b"%PDF", [0, 1], pdf_config=None, total=2)
    _pause(session)
    resumed = _open(ws, session.job_id)
    again = transcribe_mod._window_slice(b"%PDF", [0, 1], pdf_config=None, total=2)
    other = transcribe_mod._window_slice(b"%PDF", [1], pdf_config=None, total=2)
    _pause(resumed)
    assert again == first
    assert other != first  # a different window is its own slice


# ---- 2. superseded batches are canceled ----------------------------------------------


def test_a_changed_request_cancels_the_batch_it_supersedes(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=5, max_requests=1)
    session = _open(ws)
    with pytest.raises(BatchPending):
        _executor(backend, session).run_wave({"a-1": _kwargs("a"), "b-1": _kwargs("b")})
    _pause(session)
    assert len(backend.submitted) == 2

    resumed = _open(ws, session.job_id)
    with pytest.raises(BatchPending):
        # b unchanged (still in flight); a's input really changed.
        _executor(backend, resumed).run_wave({"a-1": _kwargs("a CHANGED"), "b-1": _kwargs("b")})
    _pause(resumed)
    records = BatchJobStore(ws, session.job_id).load().provider_batches
    by_ids = {tuple(r["keys"]): r["state"] for r in records}
    assert len(backend.submitted) == 3  # only the changed request is new
    assert backend.canceled == [records[0]["job"]["job_id"]]
    assert by_ids[("a-1",)] in (RECORD_DROPPED, RECORD_OPEN)
    assert sum(1 for r in records if r["state"] == RECORD_DROPPED) == 1
    assert sum(1 for r in records if r["state"] == RECORD_OPEN) == 2  # b and new a


# ---- 3. the nondeterminism guard -----------------------------------------------------


def test_a_wave_that_matches_nothing_while_its_positions_are_in_flight_stops(
    ws: Workspace,
) -> None:
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=5)
    session = _open(ws)
    with pytest.raises(BatchPending):
        _executor(backend, session).run_wave({"a-1": _kwargs("a"), "b-1": _kwargs("b")})
    _pause(session)

    resumed = _open(ws, session.job_id)
    with pytest.raises(BatchJobNondeterministic, match="transcribe"):
        _executor(backend, resumed).run_wave(
            {"a-1": _kwargs("a drifted"), "b-1": _kwargs("b drifted")}
        )
    resumed.close(None)
    assert len(backend.submitted) == 1  # nothing new was paid for
    assert not backend.canceled  # the in-flight batch is kept
    manifest = BatchJobStore(ws, session.job_id).load()
    assert manifest.status == "failed" and "BatchJobNondeterministic" in (manifest.error or "")
    assert [r["state"] for r in manifest.provider_batches] == [RECORD_OPEN]


def test_once_the_guard_fired_the_session_sends_nothing_more(ws: Workspace) -> None:
    """The guard escapes `except Exception` soft-fail handlers, and the session
    refuses every provider call afterwards — any executor's wave (even one of
    fresh, never-seen requests) and any synchronous call — so a caller that
    kept going could not pay for anything."""
    assert not issubclass(BatchJobNondeterministic, Exception)
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=5)
    session = _open(ws)
    with pytest.raises(BatchPending):
        _executor(backend, session).run_wave({"a-1": _kwargs("a")})
    _pause(session)

    resumed = _open(ws, session.job_id)
    with pytest.raises(BatchJobNondeterministic):
        try:
            _executor(backend, resumed).run_wave({"a-1": _kwargs("a drifted")})
        except Exception:  # a per-file soft-fail handler
            pytest.fail("the guard must not be an Exception")
    with pytest.raises(BatchJobNondeterministic):
        _executor(backend, resumed, stage="links").run_wave({"z-1": _kwargs("brand new")})
    live: list[dict[str, Any]] = []
    with pytest.raises(BatchJobNondeterministic):
        resumed.record_sync(_kwargs("sync"), lambda kw: live.append(kw))
    assert live == [] and len(backend.submitted) == 1
    resumed.close(resumed.nondeterministic)
    assert BatchJobStore(ws, session.job_id).load().status == "failed"


def test_the_documented_recovery_is_cancel_then_resume(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=5)
    session = _open(ws)
    with pytest.raises(BatchPending):
        _executor(backend, session).run_wave({"a-1": _kwargs("a")})
    _pause(session)
    store = BatchJobStore(ws, session.job_id)
    manifest = store.load()
    manifest.provider_batches[0]["state"] = RECORD_DROPPED  # `dgml batch cancel`
    store.save(manifest, merge=False)

    fresh = FakeBackend(_answer, provider="anthropic")
    resumed = _open(ws, session.job_id)
    out = _executor(fresh, resumed).run_wave({"a-1": _kwargs("a changed on purpose")})
    resumed.close(None)
    assert out["a-1"].choices[0].message.content == "reply:a changed on purpose"


# ---- 4. job-wide stats ----------------------------------------------------------------


def test_the_completing_runs_stats_cover_the_whole_job(ws: Workspace) -> None:
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=1)
    wave = {"a-1": _kwargs("a"), "b-1": _kwargs("b")}
    session = _open(ws)
    first = _executor(backend, session)
    with pytest.raises(BatchPending):
        first.run_wave(wave)
    _pause(session)
    assert first.stats.to_json()["batches"] == 1

    resumed = _open(ws, session.job_id)
    final = _executor(backend, resumed)
    final.run_wave(wave)
    stats = final.stats.to_json()
    resumed.close(None)
    assert stats["batches"] == 1 and stats["batch_ids"] == ["fake_batch_0001"]
    assert stats["batch_ok"] == 2  # collected by the second run
    assert stats["runs"] == 2
    assert stats["this_run"]["batches"] == 0
    assert stats["requests"] == 2  # the job's requests, not the sum of replays


def _crash(ws: Workspace, session: JobSession) -> None:
    """Simulate ``kill -9``: the process stops with no close and no further
    manifest write; its lease stays behind until someone breaks it."""
    from dgml_core import usage
    from dgml_core.batch import jobs as jobs_mod

    session._heartbeat_stop.set()
    jobs_mod._ACTIVE = None
    usage._BUFFER = None
    llm._SYNC_RECORDER = None
    BatchJobStore(ws, session.job_id).break_lease()


def test_a_run_killed_after_collecting_keeps_its_batch_ok_in_the_job_totals(
    ws: Workspace,
) -> None:
    # A resume collects the transcription batch,
    # then is killed mid-labeling. The batch is recorded collected (and
    # deleted at the provider) and its cost kept, but its `batch_ok` was only
    # counted after that last manifest write, so the completing run reported
    # transcribe `batch_ok: 0` — 10 requests served by a batch, counted nowhere.
    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=1)
    wave = {"a-1": _kwargs("a"), "b-1": _kwargs("b")}
    session = _open(ws)
    with pytest.raises(BatchPending):
        _executor(backend, session).run_wave(wave)
    _pause(session)

    collector = _open(ws, session.job_id)
    _executor(backend, collector).run_wave(wave)  # collects the batch ...
    _crash(ws, collector)  # ... and dies before any later manifest write

    final_session = _open(ws, session.job_id)
    final = _executor(backend, final_session)
    final.run_wave(wave)  # replayed from the store
    stats = final.stats.to_json()
    final_session.close(None)
    assert stats["this_run"]["replayed"] == 2
    assert stats["batch_ok"] == 2  # served by the batch the killed run collected
    assert stats["cost_usd"] == 0.5  # the cost was never lost; batch_ok must match it


def test_a_job_over_several_resumes_costs_what_a_blocking_run_costs(ws: Workspace) -> None:
    wave = {"a-1": _kwargs("a"), "b-1": _kwargs("b")}
    blocking = start_session(ws, command="test run", argv=[], job_id=None, wait=True)
    ref = _executor(FakeBackend(_answer, provider="anthropic"), blocking)
    ref.run_wave(wave)
    expected = ref.stats.to_json()
    blocking.close(None)
    assert expected["cost_usd"] == 0.5 and expected["standard_cost_usd"] == 1.0

    backend = FakeBackend(_answer, provider="anthropic", polls_until_ended=2)
    session = _open(ws)
    for _ in range(2):  # submit, then one resume that still finds it running
        with pytest.raises(BatchPending):
            _executor(backend, session).run_wave(wave)
        _pause(session)
        session = _open(ws, session.job_id)
    _executor(backend, session).run_wave(wave)  # collects
    _pause(session)  # stop after collecting: the responses are now stored

    final_session = _open(ws, session.job_id)
    final = _executor(backend, final_session)
    final.run_wave(wave)  # everything replayed from the store
    stats = final.stats.to_json()
    final_session.close(None)
    assert final.stats.run_json()["cost_usd"] == 0.0  # replayed responses add 0
    assert stats["this_run"]["cost_usd"] == 0.0
    for name in ("cost_usd", "standard_cost_usd", "saved_usd"):
        assert stats[name] == expected[name]


def test_a_single_run_job_reports_the_plain_block(ws: Workspace) -> None:
    session = start_session(ws, command="test run", argv=[], job_id=None, wait=True)
    ex = _executor(FakeBackend(_answer, provider="anthropic"), session)
    ex.run_wave({"a-1": _kwargs("a")})
    stats = ex.stats.to_json()
    session.close(None)
    assert "runs" not in stats and "this_run" not in stats
