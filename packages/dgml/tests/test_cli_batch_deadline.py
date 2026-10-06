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

"""``--batch-deadline`` through the CLI: flag validation, the ``batch.deadline``
payload block, the ``--no-wait`` pause payload, and a job resumed after its
deadline.

The wall clock the deadline reads (``dgml_core.batch.deadline.wall_clock``)
is patched to a value the test moves, so "after the deadline" is exact.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core.batch import FakeBackend, list_jobs, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.storage import Workspace

from .test_cli import _read_stderr, _read_stdout
from .test_cli_batch_extraction import _seed_extraction
from .test_cli_batch_generate import _seed as _seed_generate
from .test_cli_batch_jobs import (
    _Provider,
    _values_answer,
    _ws_args,
    pdf_stubs,  # noqa: F401  (fixture)
    provider,  # noqa: F401  (fixture)
)

_POLL = ["--batch-poll-interval", "0.01"]
T0 = 1_800_000_000.0  # 2027-01-15T08:00:00Z
_FIDS = ["fddl00000001", "fddl00000002"]


class _Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Iterator[_Clock]:
    c = _Clock()
    with patch("dgml_core.batch.deadline.wall_clock", c):
        yield c


@pytest.fixture
def slow_values() -> Iterator[FakeBackend]:
    """A values provider whose batches never end on their own; the first
    file's request is processed early (so a cancel collects it)."""
    saved = dict(batch_registry._REGISTRY)
    backend = FakeBackend(
        lambda request: _values_answer(request.kwargs),
        provider="anthropic",
        polls_until_ended=10_000,
        processed_early=lambda r: _FIDS[0] in r.custom_id,
    )
    register_backend("anthropic", lambda _cfg: backend)
    try:
        yield backend
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


def _extract(ws: Path, ds_id: str, *extra: str) -> list[str]:
    return _ws_args(ws) + ["extraction", "extract", ds_id, "--all", *extra]


def _sync_values(calls: list[Any]) -> Any:
    def completion(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _values_answer(kwargs)

    return patch("litellm.completion", side_effect=completion)


# ---- flag validation ----------------------------------------------------------------


def test_the_deadline_needs_batch_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, _FIDS[:1])
    assert main(_extract(ws, ds_id, "--batch-deadline", "1h")) == 1
    error = _read_stderr(capsys)["error"]
    assert error["code"] == "BATCH_JOB_INVALID"
    assert "--batch-deadline" in error["message"]


def test_every_batch_command_takes_the_flag_and_rejects_it_without_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, _FIDS[:1])
    gen_ws, gen_id = _seed_generate(tmp_path / "gen", capsys)
    assert main(_ws_args(gen_ws) + ["docset", "generate", gen_id, "--batch-deadline", "1h"]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "BATCH_JOB_INVALID"
    for argv in (
        ["extraction", "generate-schema", ds_id, "--batch-deadline", "1h"],
        ["file", "add", str(tmp_path / "missing.pdf"), "--batch-deadline", "1h"],
    ):
        assert main(_ws_args(ws) + argv) == 1, argv
        error = _read_stderr(capsys)["error"]
        assert error["code"] == "BATCH_JOB_INVALID", (argv, error)
        assert "--batch-deadline" in error["message"]


def test_a_bad_duration_is_an_invalid_argument(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], slow_values: FakeBackend
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, _FIDS[:1])
    assert main(_extract(ws, ds_id, "--batch", "--batch-deadline", "soon")) == 1
    assert _read_stderr(capsys)["error"]["code"] == "INVALID_ARGUMENT"
    assert slow_values.submitted == []
    assert list_jobs(Workspace(root=ws)) == []


# ---- payload ------------------------------------------------------------------------


def test_no_deadline_no_block_and_an_unreached_one_reports_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,  # noqa: F811
    clock: _Clock,
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, _FIDS)
    provider.install("anthropic", _values_answer, polls=0)
    assert main(_extract(ws, ds_id, "--batch", *_POLL)) == 0
    assert "deadline" not in _read_stdout(capsys)["batch"]

    assert main(_extract(ws, ds_id, "--batch", *_POLL, "--batch-deadline", "90m")) == 0
    assert _read_stdout(capsys)["batch"]["deadline"] == {
        "at": "2027-01-15T09:30:00Z",
        "expired": False,
        "canceled_batches": 0,
        "collected_after_cancel": 0,
        "sync_after_deadline": 0,
        "possibly_double_billed": 0,
    }


# ---- the --no-wait pause payload ------------------------------------------------------

_PAUSE_KEYS = ["job_id", "status", "command", "submitted_batches", "requests_in_flight", "resume"]


def test_a_paused_job_with_a_deadline_reports_it_on_pause_and_on_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], slow_values: FakeBackend, clock: _Clock
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, _FIDS)
    argv = _extract(ws, ds_id, "--batch", *_POLL, "--no-wait", "--batch-deadline", "1h")
    assert main(argv) == 0
    paused = _read_stdout(capsys)["batch_job"]
    # Appended after the old keys (additive), same shape `batch status` shows.
    assert list(paused) == [*_PAUSE_KEYS, "deadline"]
    assert paused["deadline"] == {"at": "2027-01-15T09:00:00Z", "expired": False}

    # The resume short-circuit reports it through the same builder.
    assert main(_ws_args(ws) + ["batch", "resume", paused["job_id"]]) == 0
    waiting = _read_stdout(capsys)["batch_job"]
    assert waiting == paused
    assert main(_ws_args(ws) + ["batch", "status", paused["job_id"]]) == 0
    assert _read_stdout(capsys)["deadline"] == paused["deadline"]


def test_a_paused_job_without_a_deadline_keeps_the_exact_old_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], slow_values: FakeBackend
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, _FIDS)
    assert main(_extract(ws, ds_id, "--batch", *_POLL, "--no-wait")) == 0
    paused = _read_stdout(capsys)["batch_job"]
    job_id = paused["job_id"]
    expected = {
        "job_id": job_id,
        "status": "pending",
        "command": paused["command"],
        "submitted_batches": 1,
        "requests_in_flight": 2,
        "resume": f"dgml batch resume {job_id}",
    }
    assert paused == expected and list(paused) == _PAUSE_KEYS

    assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
    waiting = _read_stdout(capsys)["batch_job"]
    assert waiting == expected and list(waiting) == _PAUSE_KEYS


# ---- a job resumed after its deadline -----------------------------------------------------


def test_a_job_resumed_after_its_deadline_cancels_collects_and_finishes_sync(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], slow_values: FakeBackend, clock: _Clock
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, _FIDS)
    argv = _extract(ws, ds_id, "--batch", *_POLL, "--no-wait", "--batch-deadline", "1h")
    assert main(argv) == 0
    job_id = _read_stdout(capsys)["batch_job"]["job_id"]

    assert main(_ws_args(ws) + ["batch", "status", job_id]) == 0
    status = _read_stdout(capsys)
    assert status["deadline"] == {"at": "2027-01-15T09:00:00Z", "expired": False}
    # Before the deadline, a resume of a still-running job only reports it.
    assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
    assert _read_stdout(capsys)["batch_job"]["status"] == "pending"
    # Changing the job's deadline on a resume is refused.
    assert main(argv[:-1] + ["2h", "--job", job_id]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "BATCH_JOB_INVALID"

    clock.now = T0 + 3601
    assert main(_ws_args(ws) + ["batch", "status", job_id]) == 0
    assert _read_stdout(capsys)["deadline"]["expired"] is True
    calls: list[Any] = []
    with _sync_values(calls):
        # Past the deadline the short-circuit is off: the resume finishes.
        assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
    final = _read_stdout(capsys)
    assert final["summary"] == {"total": 2, "ok": 2, "failed": 0}
    assert final["batch"]["deadline"] == {
        "at": "2027-01-15T09:00:00Z",
        "expired": True,
        "canceled_batches": 1,
        "collected_after_cancel": 1,
        "sync_after_deadline": 1,
        "possibly_double_billed": 0,
    }
    assert len(calls) == 1 and len(slow_values.submitted) == 1
    assert slow_values.canceled == ["fake_batch_0001"]
    # One request at batch price, one at standard: costs stay consistent.
    block = final["batch"]
    assert block["sync_fallbacks"] == 1 and block["batch_ok"] == 1
    assert block["saved_usd"] == pytest.approx(block["standard_cost_usd"] - block["cost_usd"])
    (manifest,) = list_jobs(Workspace(root=ws))
    assert manifest.status == "completed"
