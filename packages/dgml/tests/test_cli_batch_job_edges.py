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

"""Batch job CLI edges: resuming an argv that no longer parses, abbreviated
flags in a stored argv, and a job whose original directory is gone."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core.batch import start_session
from dgml_core.storage import Workspace

from .test_cli import _init_ws, _read_stderr, _read_stdout, _ws_args
from .test_cli_batch_extraction import _seed_extraction
from .test_cli_batch_jobs import _Provider, _values_answer
from .test_cli_batch_jobs import provider as _provider_fixture

provider = _provider_fixture
_POLL = ["--batch-poll-interval", "0.01"]


def _pending_job(ws: Path, argv: list[str], *, cwd: str, command: str = "docset generate") -> str:
    """A ``--no-wait`` style job left pending, recording *argv* and *cwd*."""
    session = start_session(
        Workspace.open(str(ws)), command=command, argv=argv, job_id=None, wait=False, cwd=cwd
    )
    session.close(session.pending_signal())
    return session.job_id


# ---- C1: resume of an argv that no longer parses -----------------------------------


def test_resume_of_an_unparseable_stored_argv_is_a_structured_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    job_id = _pending_job(ws, ["--custom"], cwd=str(tmp_path))

    rc = main(["--workspace", str(ws), "batch", "resume", job_id])
    assert rc == 1
    err = _read_stderr(capsys)["error"]
    assert err["code"] == "BATCH_JOB_INVALID"
    assert job_id in err["message"] and "--custom" in err["message"]


# ---- abbreviated flags in a job's argv are normalized ------------------------------


def test_canonical_argv_expands_abbreviations_per_subcommand() -> None:
    from dgml.cli import _canonical_argv

    argv = ["--verb", "--form=text", "extraction", "extract", "ds", "--al", "--no-w"]
    assert _canonical_argv(argv) == [
        "--verbose",
        "--format=text",
        "extraction",
        "extract",
        "ds",
        "--all",
        "--no-wait",
    ]
    # A value that looks like a subcommand is a value; unknown / ambiguous
    # abbreviations (`--works`: --workspace or --workspace-config) are left
    # alone for argparse to judge.
    assert _canonical_argv(["--form", "batch", "--custom", "--works", "x"]) == [
        "--format",
        "batch",
        "--custom",
        "--works",
        "x",
    ]
    assert _canonical_argv(["batch", "resume", "bj_1", "--verb"]) == [
        "batch",
        "resume",
        "bj_1",
        "--verbose",
    ]


def test_a_job_started_with_an_abbreviated_no_wait_stores_the_canonical_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    from dgml_core.batch.jobs import BatchJobStore

    from .test_cli_batch_jobs import _extract_argv

    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    provider.install("anthropic", _values_answer, polls=3)
    assert main(_extract_argv(ws, ds_id, "--no-w")) == 0
    first = _read_stdout(capsys)
    job_id = first["batch_job"]["job_id"]
    argv = BatchJobStore(Workspace(root=ws), job_id).load().argv
    assert "--no-wait" in argv and "--no-w" not in argv
    # So the resume short-circuit sees it: still running, nothing re-run.
    with patch("litellm.completion", side_effect=AssertionError("nothing may run")):
        assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
    assert _read_stdout(capsys) == first
    assert len(provider.backends["anthropic"].submitted) == 1


def test_resume_strips_an_abbreviated_format_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    provider.install("anthropic", _values_answer, polls=0)
    stored = ["--workspace", str(ws), "--form", "text"]
    stored += ["extraction", "extract", ds_id, "--all", "--batch", *_POLL]
    job_id = _pending_job(ws, stored, cwd=str(tmp_path), command="extraction extract")
    with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
        assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
    final = _read_stdout(capsys)  # JSON: the resume's format, not the stored `--form text`
    assert final["summary"] == {"total": 1, "ok": 1, "failed": 0}


# ---- a job whose original directory is gone -------------------------------------


def test_resume_refuses_a_job_whose_directory_is_gone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    gone = tmp_path / "gone"
    gone.mkdir()
    job_id = _pending_job(ws, ["docset", "generate", "ds", "--batch"], cwd=str(gone))
    gone.rmdir()
    with patch("litellm.completion", side_effect=AssertionError("nothing may run")):
        assert main(["--workspace", str(ws), "batch", "resume", job_id]) == 1
    err = _read_stderr(capsys)["error"]
    assert err["code"] == "BATCH_JOB_INVALID"
    assert job_id in err["message"] and str(gone) in err["message"]
