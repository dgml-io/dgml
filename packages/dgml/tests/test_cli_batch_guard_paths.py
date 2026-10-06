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

"""The one-outcome guard on every batch command path.

Every command that runs in batch job mode opens its session through
``_start_batch_job`` (which holds the run's output until the session closes),
so a resume the nondeterminism guard stops has exactly one outcome — its error
envelope naming the job, exit 1, nothing on stdout — whichever stage it stops
in. The stages here are the ones whose orchestration lives in ``dgml_core``
(transcription and the link stage of ``docset generate``, classification and
the auto-extraction of ``file add``, ``extraction extract`` and
``extraction generate-schema``): any ``except Exception`` soft-fail block there
must let the guard (a ``BaseException``) through.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core.batch import jobs as jobs_mod
from dgml_core.batch import list_jobs
from dgml_core.generation import label as label_mod
from dgml_core.generation import links as links_mod
from dgml_core.storage import Workspace

from .test_cli import _read_stdout
from .test_cli_batch_extraction import _seed_classify_ws, _seed_extraction, _set_schema
from .test_cli_batch_jobs import (
    _PAGES,
    _extract_argv,
    _generate_argv,
    _generation_answer,
    _Provider,
    _seed_generate_ws,
    _system,
    _tool_reply,
    _values_answer,
    _ws_args,
    pdf_stubs,  # noqa: F401  (fixture)
    provider,  # noqa: F401  (fixture)
)
from .test_cli_batch_schema import _install as _install_schema
from .test_cli_batch_schema import _seed as _seed_schema

_NO_SYNC = AssertionError("no sync call expected")


def _generate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake: _Provider
) -> tuple[Path, list[str], Any]:
    ws, did = _seed_generate_ws(tmp_path, capsys)
    fake.install("anthropic", _generation_answer, polls=1)
    # Labeling is open-vocabulary: it batches one document per wave.
    return ws, _generate_argv(ws, did, "--no-wait"), lambda **kw: _generation_answer(kw)


def _file_add(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake: _Provider
) -> tuple[Path, list[str], Any]:
    ws, contracts, _, src = _seed_classify_ws(tmp_path, capsys, "ws", grounded=True)
    _set_schema(ws, tmp_path, contracts, capsys)

    def answer(kwargs: dict[str, Any]) -> Any:
        name = kwargs["tools"][0]["function"]["name"]
        if name == "assign_to_existing_docset":
            return _tool_reply(name, {"docset_id": contracts})
        return _values_answer(kwargs)

    fake.install("anthropic", answer, polls=1)
    argv = _ws_args(ws) + [
        "file",
        "add",
        str(src),
        "--auto-classify",
        "existing",
        "--batch",
        "--batch-poll-interval",
        "0.01",
        "--no-wait",
    ]
    return ws, argv, _NO_SYNC


def _extract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake: _Provider
) -> tuple[Path, list[str], Any]:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001", "fjob00000002"])
    fake.install("anthropic", _values_answer, polls=1)
    return ws, _extract_argv(ws, ds_id, "--no-wait"), _NO_SYNC


def _generate_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake: _Provider
) -> tuple[Path, list[str], Any]:
    fake.backends["schema"] = _install_schema(polls=1)
    ws, ds_id = _seed_schema(tmp_path, capsys)
    argv = _ws_args(ws) + ["extraction", "generate-schema", ds_id, "--batch", "--no-wait"]
    return ws, argv, _NO_SYNC


def _always(_kwargs: dict[str, Any]) -> bool:
    return True


def _is_link(kwargs: dict[str, Any]) -> bool:
    return _system(kwargs) in (links_mod.SYSTEM_PROMPT, links_mod.VERIFY_SYSTEM_PROMPT)


def _is_plan(kwargs: dict[str, Any]) -> bool:
    return _system(kwargs) == label_mod.PLAN_SYSTEM_PROMPT


def _is_label(kwargs: dict[str, Any]) -> bool:
    return _system(kwargs) == label_mod.SYSTEM_PROMPT


def _is_values(kwargs: dict[str, Any]) -> bool:
    return bool(kwargs["tools"][0]["function"]["name"] != "assign_to_existing_docset")


_Setup = Callable[[Path, pytest.CaptureFixture[str], _Provider], tuple[Path, list[str], Any]]


@pytest.mark.parametrize(
    ("setup", "command", "resumes", "drifts", "stage"),
    [
        (_generate, "docset generate", 0, _always, "transcribe"),
        # Three transcription waves, then roster planning's draft
        # (dgml_core.generation.single_calls through pipeline's runner).
        (_generate, "docset generate", 3, _is_plan, "plan"),
        # ... its refine wave, then one labeling wave per document
        # (pipeline._per_document_labeler) ...
        (_generate, "docset generate", 5, _is_label, "label"),
        # ... then dgml_core.generation.link_stage's.
        (_generate, "docset generate", 5 + len(_PAGES), _is_link, "links"),
        (_file_add, "file add", 0, _always, "classification"),  # dgml_core.auto_classification
        (
            _file_add,
            "file add",
            1,
            _is_values,
            "extraction phase 1",
        ),  # the auto-extraction after it
        (
            _extract,
            "extraction extract",
            0,
            _always,
            "extraction phase 1",
        ),  # grounded.extract_values_many
        (_generate_schema, "extraction generate-schema", 0, _always, "schema"),
    ],
    ids=[
        "generate-transcribe",
        "generate-plan",
        "generate-label",
        "generate-links",
        "add-classify",
        "add-extract",
        "extract",
        "schema",
    ],
)
def test_the_drift_guard_has_one_outcome_on_every_batch_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,  # noqa: F811
    pdf_stubs: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    setup: _Setup,
    command: str,
    resumes: int,
    drifts: Callable[[dict[str, Any]], bool],
    stage: str,
) -> None:
    ws, argv, sync = setup(tmp_path, capsys, provider)
    with patch("litellm.completion", side_effect=sync):
        assert main(argv) == 0
        pending = _read_stdout(capsys)["batch_job"]
        job_id = pending["job_id"]
        assert pending["command"] == command
        for _ in range(resumes):  # advance to the wave under test
            assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
            assert _read_stdout(capsys)["batch_job"]["job_id"] == job_id
    submitted = {name: len(b.submitted) for name, b in provider.backends.items()}

    digest = jobs_mod.request_digest

    def drifting(kwargs: Any) -> str:  # the wave under test rebuilds differently
        return digest(kwargs) + ("-drifted" if drifts(kwargs) else "")

    monkeypatch.setattr(jobs_mod, "request_digest", drifting)
    with patch("litellm.completion", side_effect=sync):
        rc = main(_ws_args(ws) + ["batch", "resume", job_id])
    out = capsys.readouterr()
    assert rc == 1
    assert out.out == ""  # one outcome: no payload alongside the error
    start = max(i for i, c in enumerate(out.err) if c == "{" and (i == 0 or out.err[i - 1] == "\n"))
    error = json.loads(out.err[start:])["error"]
    assert error["code"] == "BATCH_JOB_NONDETERMINISTIC"
    assert f", stage {stage}:" in error["message"]  # stopped in the wave under test
    assert error["details"]["batch"]["job"] == {
        "job_id": job_id,
        "status": "failed",
        "resume": f"dgml batch resume {job_id}",
    }
    # Nothing new was paid for.
    assert {name: len(b.submitted) for name, b in provider.backends.items()} == submitted
    (manifest,) = list_jobs(Workspace(root=ws))
    assert manifest.status == "failed"
    assert "BatchJobNondeterministic" in (manifest.error or "")
