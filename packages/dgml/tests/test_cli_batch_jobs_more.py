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

"""Batch job mode through the CLI: stranded batches, relative
paths on resume, stored credential references, the lease, retention
(`batch delete` / `batch prune`), and batch mode from config."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core import layout
from dgml_core.batch import FakeBackend, list_jobs, register_backend
from dgml_core.batch.jobs import BatchJobStore
from dgml_core.storage import Workspace

from .conftest import dump_toml, needs_gs
from .test_cli import _read_stderr, _read_stdout, _write_ws_config
from .test_cli_batch_extraction import _seed_classify_ws, _seed_extraction
from .test_cli_batch_jobs import (
    _drive_job,
    _extract_argv,
    _generate_argv,
    _generation_answer,
    _Provider,
    _seed_generate_ws,
    _tool_reply,
    _values_answer,
    _ws_args,
)
from .test_cli_batch_jobs import pdf_stubs as _pdf_stubs_fixture
from .test_cli_batch_jobs import provider as _provider_fixture

# Reuse the job tests' fixtures under their own names.
provider = _provider_fixture
pdf_stubs = _pdf_stubs_fixture

SECRET = "sk-DGML-CLI-REVIEW-SECRET-24680"


class _PollFails(FakeBackend):
    def poll(self, job: Any) -> Any:
        raise RuntimeError("provider unreachable while polling")


def _job(ws: Path) -> Any:
    (manifest,) = list_jobs(Workspace(root=ws))
    return manifest


# ---- 1. no stranded paid batch ---------------------------------------------------


def test_a_poll_failure_cancels_the_batch_fails_the_job_and_resume_resubmits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    failing = _PollFails(lambda r: _values_answer(r.kwargs), provider="anthropic")
    register_backend("anthropic", lambda _cfg: failing)
    assert main(_extract_argv(ws, ds_id)) == 0  # per-file soft failure, exit 0
    first = _read_stdout(capsys)
    assert first["results"][0]["status"] == "failed"
    assert failing.canceled  # the paid batch was not left running
    job = _job(ws)
    assert job.status == "failed" and "canceled" in job.error

    good = provider.install("anthropic", _values_answer, polls=0)
    assert main(_ws_args(ws) + ["batch", "resume", job.job_id]) == 0
    final = _read_stdout(capsys)
    assert final["summary"] == {"total": 1, "ok": 1, "failed": 0}
    assert len(good.submitted) == 1


# ---- 2. relative paths resolve from where the job was started ------------------------


@needs_gs
def test_resume_from_another_directory_resolves_relative_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, contracts, _, src = _seed_classify_ws(tmp_path, capsys, "rel")

    def answer(kwargs: dict[str, Any]) -> Any:
        return _tool_reply("assign_to_existing_docset", {"docset_id": contracts})

    provider.install("anthropic", answer, polls=1)
    monkeypatch.chdir(src.parent)
    argv = _ws_args(ws) + [
        "file",
        "add",
        f"./{src.name}",
        "--auto-classify",
        "existing",
        "--batch",
        "--batch-poll-interval",
        "0.01",
        "--no-wait",
    ]
    assert main(argv) == 0
    first = _read_stdout(capsys)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _pendings, final = _drive_job(capsys, ws, first)
    assert final["summary"]["added"] == 2
    assert Path.cwd() == elsewhere  # restored after the replay


def test_resume_from_another_directory_keeps_a_relative_schema_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, did = _seed_generate_ws(tmp_path / "job", capsys)
    (tmp_path / "tags.txt").write_text("Greeting\n", encoding="utf-8")
    provider.install("anthropic", _generation_answer, polls=1)
    monkeypatch.chdir(tmp_path)
    with patch("litellm.completion", side_effect=lambda **kw: _generation_answer(kw)):
        assert main(_generate_argv(ws, did, "--schema-path", "tags.txt", "--no-wait")) == 0
        first = _read_stdout(capsys)
        monkeypatch.chdir(tmp_path / "job")
        _pendings, final = _drive_job(capsys, ws, first)
    assert final["summary"]["converted"] == 2


# ---- 3. stored credential references ------------------------------------------------


def test_status_and_cancel_use_the_workspace_key_and_report_cancel_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dgml_core.batch.registry as batch_registry

    monkeypatch.setenv("DGML_REVIEW_VALUES_KEY", SECRET)
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    Workspace(root=ws).config_path.write_text(
        dump_toml(
            {
                "grounded": {
                    "schema_model": "anthropic/claude-opus-4-7",
                    "values_model": "anthropic/claude-sonnet-4-6",
                    "values_api_key_env": "DGML_REVIEW_VALUES_KEY",
                }
            }
        ),
        encoding="utf-8",
    )
    keys_seen: list[str | None] = []
    fail_cancel = {"on": True}

    class Backend(FakeBackend):
        def cancel(self, job: Any) -> None:
            if fail_cancel["on"]:
                raise RuntimeError("cancel refused")
            super().cancel(job)

    shared = Backend(lambda r: _values_answer(r.kwargs), provider="anthropic", polls_until_ended=5)

    def factory(cfg: Any) -> FakeBackend:
        keys_seen.append(cfg.api_key)
        return shared

    saved = dict(batch_registry._REGISTRY)
    register_backend("anthropic", factory)
    try:
        assert main(_extract_argv(ws, ds_id, "--no-wait")) == 0
        job_id = _read_stdout(capsys)["batch_job"]["job_id"]
        keys_seen.clear()
        assert main(_ws_args(ws) + ["batch", "status", job_id]) == 0
        _read_stdout(capsys)
        assert keys_seen == [SECRET]  # the configured key, not the provider default

        assert main(_ws_args(ws) + ["batch", "cancel", job_id]) == 0
        failed_cancel = _read_stdout(capsys)
        assert failed_cancel["canceled"] is False
        assert failed_cancel["batches"][0]["state"] == "open"
        assert "cancel refused" in failed_cancel["batches"][0]["error"]
        assert failed_cancel["status"] != "failed"

        fail_cancel["on"] = False
        assert main(_ws_args(ws) + ["batch", "cancel", job_id]) == 0
        canceled = _read_stdout(capsys)
        assert canceled["canceled"] is True and canceled["status"] == "failed"
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)
    for path in (ws / layout.BATCHES_DIR).rglob("*"):
        if path.is_file():
            assert SECRET.encode() not in path.read_bytes(), path


# ---- 4. one runner per job ------------------------------------------------------------


def test_a_leased_job_is_busy_and_unlock_breaks_the_lease(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    provider.install("anthropic", _values_answer, polls=5)
    assert main(_extract_argv(ws, ds_id, "--no-wait")) == 0
    job_id = _read_stdout(capsys)["batch_job"]["job_id"]
    BatchJobStore(Workspace(root=ws), job_id).acquire_lease("some-other-process")
    for sub in ("resume", "cancel", "delete"):
        assert main(_ws_args(ws) + ["batch", sub, job_id]) == 1
        assert _read_stderr(capsys)["error"]["code"] == "BATCH_JOB_BUSY"
    # status is read-only and needs no lease.
    assert main(_ws_args(ws) + ["batch", "status", job_id]) == 0
    assert _read_stdout(capsys)["lease"]["held_by"] == "some-other-process"
    assert main(_ws_args(ws) + ["batch", "unlock", job_id]) == 0
    assert _read_stdout(capsys)["unlocked"] is True
    assert main(_ws_args(ws) + ["batch", "status", job_id]) == 0
    _read_stdout(capsys)


# ---- 5. retention ------------------------------------------------------------------------


def test_delete_refuses_open_batches_unless_forced_and_prune_removes_finished(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    backend = provider.install("anthropic", _values_answer, polls=5)
    assert main(_extract_argv(ws, ds_id, "--no-wait")) == 0
    open_job = _read_stdout(capsys)["batch_job"]["job_id"]
    assert main(_ws_args(ws) + ["batch", "delete", open_job]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "BATCH_JOB_INVALID"
    assert main(_ws_args(ws) + ["batch", "delete", open_job, "--force"]) == 0
    assert _read_stdout(capsys)["deleted"] is True and backend.canceled
    assert list_jobs(Workspace(root=ws)) == []

    # A completed --no-wait job keeps a summary until pruned.
    provider.install("anthropic", _values_answer, polls=0)
    assert main(_extract_argv(ws, ds_id, "--no-wait")) == 0
    assert "batch_job" not in _read_stdout(capsys)
    assert [m.status for m in list_jobs(Workspace(root=ws))] == ["completed"]
    assert main(_ws_args(ws) + ["batch", "prune", "--older-than", "1"]) == 0
    assert _read_stdout(capsys)["deleted"] == []  # too recent
    assert main(_ws_args(ws) + ["batch", "prune"]) == 0
    assert len(_read_stdout(capsys)["deleted"]) == 1
    assert list_jobs(Workspace(root=ws)) == []


def test_a_no_op_batch_run_leaves_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, [])
    provider.install("anthropic", _values_answer, polls=0)
    for extra in ([], ["--no-wait"]):
        assert main(_extract_argv(ws, ds_id, *extra)) == 0
        assert _read_stdout(capsys)["summary"]["total"] == 0
    assert list_jobs(Workspace(root=ws)) == []


# ---- 8. batch mode from config is recorded explicitly --------------------------------


def test_batch_mode_from_config_survives_a_config_flip_before_resume(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
) -> None:
    ws, did = _seed_generate_ws(tmp_path, capsys)
    generation = {
        "model": "anthropic/claude-haiku-4-5",
        "label_model": "anthropic/claude-sonnet-4-6",
    }
    _write_ws_config(ws, {"generation": {**generation, "batch": True}})
    provider.install("anthropic", _generation_answer, polls=1)
    argv = [a for a in _generate_argv(ws, did, "--no-wait") if a != "--batch"]
    with patch("litellm.completion", side_effect=lambda **kw: _generation_answer(kw)):
        assert main(argv) == 0
        first = _read_stdout(capsys)
        assert "--batch" in _job(ws).argv
        _write_ws_config(ws, {"generation": {**generation, "batch": False}})
        _pendings, final = _drive_job(capsys, ws, first)
    assert final["summary"]["converted"] == 2
