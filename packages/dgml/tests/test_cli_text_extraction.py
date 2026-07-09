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

"""CLI-level integration tests for text-extraction modes.

The pure text-extraction unit tests live in
``dgml-core``'s ``tests/test_text_extraction.py``; this file exercises the same
behaviour through the ``dgml`` CLI surface (``file add --text-mode …``,
``check``) and the JSON/stderr error envelopes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from dgml.cli import main
from dgml_core.storage import Workspace


def _ws_args(ws: Path) -> list[str]:
    return ["--workspace", str(ws)]


def _read_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)  # type: ignore[no-any-return]


def _read_stderr(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().err)  # type: ignore[no-any-return]


def test_check_surfaces_text_extraction_failed_permanent(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    Workspace(root=ws).init()
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    assert rc == 0
    add_payload = _read_stdout(capsys)
    assert add_payload["text_extraction_error"] is not None

    rc = main(_ws_args(ws) + ["check"])
    assert rc == 2
    report = _read_stdout(capsys)
    kinds = {i["kind"] for i in report["issues"]}
    assert "text_extraction_failed_permanent" in kinds


def test_file_add_ocr_mode_with_invalid_config_errors(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An *invalid* OCR config blocks `file add` before any filesystem state
    is created, so the workspace stays clean. (A missing config is not
    an error: it falls back to the zero-config on-device macOS provider.)"""
    ws = tmp_path / "ws"
    Workspace(root=ws).init()
    # Azure provider with no endpoint → OcrConfigInvalid at load time.
    (ws / "config.json").write_text(json.dumps({"ocr": {"provider": "azure"}}), encoding="utf-8")
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["file", "add", str(text_pdf), "--text-mode", "ocr"])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "OCR_CONFIG_INVALID"

    # No file record was created on the rejected add.
    files_dir = ws / "files"
    assert not any(files_dir.iterdir())


def test_file_add_hybrid_mode_with_invalid_config_errors(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hybrid mode runs OCR per page; like `--text-mode ocr`, an invalid
    workspace OCR config blocks the add before any filesystem state is
    created."""
    ws = tmp_path / "ws"
    Workspace(root=ws).init()
    (ws / "config.json").write_text(json.dumps({"ocr": {"provider": "azure"}}), encoding="utf-8")
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["file", "add", str(text_pdf), "--text-mode", "hybrid"])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "OCR_CONFIG_INVALID"

    files_dir = ws / "files"
    assert not any(files_dir.iterdir())
