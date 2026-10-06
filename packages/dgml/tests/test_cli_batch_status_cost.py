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

"""`dgml batch status` is read-only (no lease, no writes) and reports the
lease; every `--batch` payload reports what its batched requests cost."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core import layout
from dgml_core.batch.jobs import BatchJobStore
from dgml_core.storage import Workspace

from .test_cli import _read_stderr, _read_stdout
from .test_cli_batch_extraction import _seed_extraction
from .test_cli_batch_jobs import (
    _drive_job,
    _extract_argv,
    _generate_argv,
    _generation_answer,
    _Provider,
    _seed_generate_ws,
    _values_answer,
    _ws_args,
)
from .test_cli_batch_jobs import pdf_stubs as _pdf_stubs_fixture
from .test_cli_batch_jobs import provider as _provider_fixture

provider = _provider_fixture
pdf_stubs = _pdf_stubs_fixture


def _paused_extraction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> tuple[Path, str]:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    provider.install("anthropic", _values_answer, polls=5)
    assert main(_extract_argv(ws, ds_id, "--no-wait")) == 0
    return ws, _read_stdout(capsys)["batch_job"]["job_id"]


def _manifest_bytes(ws: Path, job_id: str) -> bytes:
    return (ws / layout.batch_job_manifest_key(job_id)).read_bytes()


# ---- 1. status is read-only -------------------------------------------------------------


def test_status_works_while_another_process_holds_the_lease(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, job_id = _paused_extraction(tmp_path, capsys, provider)
    store = BatchJobStore(Workspace(root=ws), job_id)
    store.acquire_lease("running-elsewhere")
    before = _manifest_bytes(ws, job_id)
    lease_before = (ws / layout.batch_job_lease_key(job_id)).read_bytes()

    assert main(_ws_args(ws) + ["batch", "status", job_id]) == 0
    status = _read_stdout(capsys)
    assert status["lease"]["held_by"] == "running-elsewhere"
    assert status["lease"]["stale"] is False and status["lease"]["expires_at"]
    assert status["batches"][0]["done"] is False
    assert status["status"] == "pending"
    assert _manifest_bytes(ws, job_id) == before  # nothing written
    assert (ws / layout.batch_job_lease_key(job_id)).read_bytes() == lease_before

    # The lease still guards the job: a resume is refused.
    assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "BATCH_JOB_BUSY"


def test_status_right_after_a_crash_reports_a_stale_lease(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, job_id = _paused_extraction(tmp_path, capsys, provider)
    BatchJobStore(Workspace(root=ws), job_id).acquire_lease("dead-pid", ttl_s=-1)
    assert main(_ws_args(ws) + ["batch", "status", job_id]) == 0
    lease = _read_stdout(capsys)["lease"]
    assert lease["held_by"] == "dead-pid" and lease["stale"] is True


def test_status_reports_no_lease_and_derives_ready_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, job_id = _paused_extraction(tmp_path, capsys, provider)
    before = _manifest_bytes(ws, job_id)
    statuses = []
    for _ in range(5):  # polls_until_ended=5, one poll already made by the run
        assert main(_ws_args(ws) + ["batch", "status", job_id]) == 0
        payload = _read_stdout(capsys)
        assert payload["lease"] is None
        statuses.append(payload["status"])
    assert statuses[-1] == "ready" and statuses[0] == "pending"
    assert _manifest_bytes(ws, job_id) == before
    assert main(_ws_args(ws) + ["batch", "list"]) == 0
    assert _read_stdout(capsys)["jobs"][0]["status"] == "pending"  # stored, not derived


# ---- 2. cost fields ----------------------------------------------------------------------


def test_blocking_extraction_reports_the_batch_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001", "fjob00000002"])
    provider.install("anthropic", _values_answer, polls=0)
    assert main(_extract_argv(ws, ds_id)) == 0
    block = _read_stdout(capsys)["batch"]
    assert block["cost_usd"] == pytest.approx(2 * 0.02)  # two batch responses
    assert block["standard_cost_usd"] == pytest.approx(2 * 0.04)
    assert block["saved_usd"] == pytest.approx(0.04)


def test_a_resumed_generate_job_reports_the_blocking_runs_cost(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
) -> None:
    def cost_fields(block: dict[str, Any]) -> dict[str, float]:
        return {k: block[k] for k in ("cost_usd", "standard_cost_usd", "saved_usd")}

    ws_b, did_b = _seed_generate_ws(tmp_path / "blocking", capsys)
    provider.install("anthropic", _generation_answer, polls=0)
    with patch("litellm.completion", side_effect=lambda **kw: _generation_answer(kw)):
        assert main(_generate_argv(ws_b, did_b)) == 0
    blocking = _read_stdout(capsys)["batch"]["stages"]

    ws_j, did_j = _seed_generate_ws(tmp_path / "job", capsys)
    provider.install("anthropic", _generation_answer, polls=1)
    with patch("litellm.completion", side_effect=lambda **kw: _generation_answer(kw)):
        assert main(_generate_argv(ws_j, did_j, "--no-wait")) == 0
        pendings, final = _drive_job(capsys, ws_j, _read_stdout(capsys))
    stages = final["batch"]["stages"]
    assert len(pendings) > 1
    for stage in ("transcribe", "links"):
        assert cost_fields(stages[stage]) == pytest.approx(cost_fields(blocking[stage]))
        assert stages[stage]["cost_usd"] > 0
        assert stages[stage]["this_run"]["cost_usd"] < stages[stage]["cost_usd"]
