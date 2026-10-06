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

"""``dgml batch resume`` short-circuits while the job's wave is still open.

Resuming a ``--no-wait`` job whose provider batches are all still running
used to re-run the whole command only to pause again. It now polls those
batches first (read-only, like ``batch status``) and, when none has ended,
prints the pending payload a pausing run prints without re-running anything.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core import layout

from .test_cli import _read_stdout
from .test_cli_batch_extraction import _seed_extraction
from .test_cli_batch_jobs import _drive_job, _extract_argv, _Provider, _values_answer, _ws_args
from .test_cli_batch_jobs import provider as _provider_fixture

provider = _provider_fixture


def _manifest(ws: Path, job_id: str) -> bytes:
    return (ws / layout.batch_job_manifest_key(job_id)).read_bytes()


def test_resume_while_no_batch_has_ended_reports_pending_without_rerunning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001", "fjob00000002"])
    backend = provider.install("anthropic", _values_answer, polls=3)
    assert main(_extract_argv(ws, ds_id, "--no-wait")) == 0  # poll 1
    first = _read_stdout(capsys)
    job_id = first["batch_job"]["job_id"]
    before = _manifest(ws, job_id)

    with patch("litellm.completion", side_effect=AssertionError("nothing may run")):
        for _ in range(2):  # polls 2 and 3: still running
            assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
            assert _read_stdout(capsys) == first  # the same pending payload
    assert _manifest(ws, job_id) == before  # no run: nothing written, runs unchanged
    assert len(backend.submitted) == 1

    # Poll 4 sees the batch ended: the command re-runs and finishes the job.
    pendings, final = _drive_job(capsys, ws, first)
    assert pendings == [first]
    assert final["summary"] == {"total": 2, "ok": 2, "failed": 0}
    assert len(backend.submitted) == 1

    # A completed job is not short-circuited: resuming it re-runs nothing and
    # is refused as the docs say (BATCH_JOB_INVALID), with no payload.
    with patch("litellm.completion", side_effect=AssertionError("nothing may run")):
        assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert '"code": "BATCH_JOB_INVALID"' in out.err and "already completed" in out.err
    assert len(backend.submitted) == 1
