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

"""The live bleed: every resume re-submitted the first transcription wave,
because a PDF slice comes out different in every process. Reproduced here with
a slicer that returns new bytes on every call."""

from __future__ import annotations

from itertools import count
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core.batch import list_jobs
from dgml_core.generation import document as document_mod
from dgml_core.storage import Workspace

from .test_cli_batch_jobs import (
    _drive_job,
    _generate_argv,
    _generation_answer,
    _Provider,
    _seed_generate_ws,
)
from .test_cli_batch_jobs import pdf_stubs as _pdf_stubs_fixture
from .test_cli_batch_jobs import provider as _provider_fixture

provider = _provider_fixture
pdf_stubs = _pdf_stubs_fixture


def test_resumes_with_a_drifting_slicer_submit_each_wave_once(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serial = count()

    def drifting(_pdf: bytes, pages: list[int], **_kw: Any) -> bytes:
        # Same pages, different bytes every call — ghostscript's random /ID.
        return bytes(pages) + f"/ID {next(serial)}".encode()

    monkeypatch.setattr(document_mod, "slice_pdf", drifting)
    ws, did = _seed_generate_ws(tmp_path, capsys)
    backend = provider.install("anthropic", _generation_answer, polls=1)
    with patch("litellm.completion", side_effect=lambda **kw: _generation_answer(kw)):
        assert main(_generate_argv(ws, did, "--no-wait")) == 0
        first = _read(capsys)
        pendings, final = _drive_job(capsys, ws, first)

    # 3 transcription waves (the 3-page doc) + roster planning's draft and
    # refine + the link pass's propose and verify, each once.
    assert len(backend.submitted) == 3 + 2 + 2
    assert len(pendings) == 3 + 2 + 2
    records = list_jobs(Workspace(root=ws))[0].provider_batches
    assert all(r["state"] == "collected" for r in records)

    stages = final["batch"]["stages"]
    assert stages["transcribe"]["batches"] == 3  # job-wide, not the last run's 0
    assert stages["transcribe"]["runs"] == len(pendings) + 1
    assert stages["links"]["batches"] == 2
    assert stages["plan"]["batches"] == 2


def _read(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    import json

    return json.loads(capsys.readouterr().out)  # type: ignore[no-any-return]
