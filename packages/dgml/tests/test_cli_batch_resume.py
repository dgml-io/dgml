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

import sys
import tomllib
import types
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core import layout
from dgml_core.workspaces_local import LocalDirWorkspacesStore
from dgml_core.workspaces_resolve import default_workspaces_store

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


class _ConfigElsewhereStore(LocalDirWorkspacesStore):
    """A store of workspaces that, like the MongoDB one, keeps each config away from
    the workspace's data: nothing at ``<root>/config.toml`` names the workspace, so it
    opens only by its id."""

    data_roots: ClassVar[dict[str, Path]] = {}

    def _config_path(self, workspace_id: str) -> Path:
        return self.root / "configs" / f"{workspace_id}.toml"

    def workspace_root(self, workspace_id: str) -> Path:
        return self.data_roots[workspace_id]

    def config_file(self, workspace_id: str) -> Path | None:
        return None


def test_resume_reopens_a_store_held_workspace_by_its_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    config = ws / layout.CONFIG_FILE
    wid = tomllib.loads(config.read_text(encoding="utf-8"))["workspace"]["workspace_id"]
    # Hand the workspace's config to the store: its data root keeps none.
    module = types.ModuleType("_config_elsewhere_store")
    module.Store = _ConfigElsewhereStore  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(_ConfigElsewhereStore, "data_roots", {wid: ws})
    user_config = tmp_path / "xdg-home" / "dgml" / "config.toml"
    user_config.parent.mkdir(parents=True, exist_ok=True)
    user_config.write_text(
        f'[workspaces]\nprovider = "{module.__name__}:Store"\nroot = "{tmp_path / "store"}"\n',
        encoding="utf-8",
    )
    default_workspaces_store.cache_clear()
    default_workspaces_store().create_config(wid, config.read_text(encoding="utf-8"))
    config.unlink()

    provider.install("anthropic", _values_answer, polls=1)
    argv = ["--workspace", wid, *_extract_argv(ws, ds_id, "--no-wait")[2:]]
    assert main(argv) == 0
    job_id = _read_stdout(capsys)["batch_job"]["job_id"]
    with patch("litellm.completion", side_effect=AssertionError("no full-price fallback")):
        for _ in range(5):
            assert main(["--workspace", wid, "batch", "resume", job_id]) == 0
            out = _read_stdout(capsys)
            if "batch_job" not in out:
                break
    assert out["summary"] == {"total": 1, "ok": 1, "failed": 0}
