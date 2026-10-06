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

"""A run that loses its job's lease mid-run (``dgml batch unlock`` of a live
run, then another process taking the job) ends with one outcome: a
``BATCH_JOB_BUSY`` error envelope, exit 1, nothing on stdout — and the batch
it had just created stays recorded, so the holder collects it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from dgml.cli import main
from dgml_core.batch import BatchJobStore, active_session, list_jobs
from dgml_core.storage import Workspace

from .test_cli import _read_stdout
from .test_cli_batch_extraction import _seed_extraction
from .test_cli_batch_jobs import (
    _extract_argv,
    _Provider,
    _values_answer,
    _ws_args,
    provider,  # noqa: F401  (fixture)
)


def test_a_run_whose_lease_is_taken_mid_run_reports_busy_and_submits_nothing_more(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,  # noqa: F811
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001", "fjob00000002"])
    backend = provider.install("anthropic", _values_answer, polls=1)
    submit = backend.submit

    def taken_while_submitting(requests: Any) -> Any:
        job = submit(requests)
        session = active_session()
        assert session is not None
        store = BatchJobStore(Workspace(root=ws), session.job_id)
        store.break_lease()  # `dgml batch unlock` of this live run...
        store.acquire_lease("another-process")  # ...and a resume took the job
        return job

    backend.submit = taken_while_submitting  # type: ignore[method-assign]
    rc = main(_extract_argv(ws, ds_id, "--no-wait"))
    out = capsys.readouterr()
    assert rc == 1
    assert out.out == ""  # one outcome: no payload alongside the error
    start = max(i for i, c in enumerate(out.err) if c == "{" and (i == 0 or out.err[i - 1] == "\n"))
    error = json.loads(out.err[start:])["error"]
    assert error["code"] == "BATCH_JOB_BUSY"
    assert "lost the job's lease" in error["message"]
    (manifest,) = list_jobs(Workspace(root=ws))
    job_id = manifest.job_id
    assert error["details"] == {
        "lease_lost": True,
        "batch": {"job": {"job_id": job_id, "resume": f"dgml batch resume {job_id}"}},
    }
    assert len(backend.submitted) == 1
    assert [r["state"] for r in manifest.provider_batches] == ["open"]
    store = BatchJobStore(Workspace(root=ws), job_id)
    assert store.lease_info()["held_by"] == "another-process"  # type: ignore[index]

    # The other process finishes with the job; a resume collects the recorded
    # batch instead of submitting it again.
    backend.submit = submit  # type: ignore[method-assign]
    store.break_lease()
    for _ in range(5):
        assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
        if "batch_job" not in _read_stdout(capsys):
            break
    assert len(backend.submitted) == 1
