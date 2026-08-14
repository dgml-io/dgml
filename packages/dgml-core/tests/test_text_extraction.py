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

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dgml_core import layout
from dgml_core.consistency import check_workspace
from dgml_core.errors import load_recorded_errors
from dgml_core.files import FileStore
from dgml_core.pages import DEFAULT_DPI
from dgml_core.storage import Workspace
from dgml_core.text_extraction import (
    TextMode,
    extract_text_digital,
)

from .conftest import PAGE_HEIGHT_PTS, PAGE_WIDTH_PTS


def _page_texts(ws: Workspace, file_id: str) -> list[str]:
    """page_text blob keys for ``file_id`` — the store analogue of globbing
    ``page_*.json`` in the workspace's page-text dir."""
    return [k for k in ws.blobs.list_blobs(layout.file_text_prefix(file_id)) if k.endswith(".json")]


def test_extract_text_digital_writes_per_page_json(tmp_path: Path, text_pdf: Path) -> None:
    out_dir = tmp_path / "page_text"
    result = extract_text_digital(text_pdf, out_dir, file_id="testfile1234")

    assert result.pages_written == 2
    assert result.pages_with_words == 2
    assert result.total_words >= 4  # "Hello", "World", "Second", "Page", "Text"

    page1 = json.loads((out_dir / "page_1.json").read_text())
    page2 = json.loads((out_dir / "page_2.json").read_text())

    expected_w = round(PAGE_WIDTH_PTS * DEFAULT_DPI / 72)
    expected_h = round(PAGE_HEIGHT_PTS * DEFAULT_DPI / 72)
    for page in (page1, page2):
        assert page["file_id"] == "testfile1234"
        assert page["width"] == expected_w
        assert page["height"] == expected_h
        assert page["words"]
        for word in page["words"]:
            assert isinstance(word["t"], str) and word["t"]
            assert len(word["l"]) == 4
            left, top, right, bottom = word["l"]
            assert all(isinstance(v, int) for v in word["l"])
            assert 0 <= left < right <= expected_w
            assert 0 <= top < bottom <= expected_h

    assert page1["page"] == 1
    assert page2["page"] == 2
    words_p1 = [w["t"] for w in page1["words"]]
    assert "Hello" in words_p1
    assert "World" in words_p1


def test_extract_text_digital_scales_boxes_with_dpi(tmp_path: Path, text_pdf: Path) -> None:
    """Word boxes live in page-image pixel space, so they must follow the render dpi.

    Halving the dpi has to halve the page dimensions *and* every coordinate. Boxes
    left in 300-dpi space would sit off the right/bottom edge of a 150-dpi page
    image, breaking dg:origin grounding and — because the hybrid merge finds
    overlaps by comparing digital boxes against OCR boxes read from the real
    image — dropping every digital-only region as assumed-invisible.
    """
    at_300 = extract_text_digital(text_pdf, tmp_path / "at300", file_id="f", dpi=300)
    at_150 = extract_text_digital(text_pdf, tmp_path / "at150", file_id="f", dpi=150)
    assert at_300.total_words == at_150.total_words

    page_300 = json.loads((tmp_path / "at300" / "page_1.json").read_text())
    page_150 = json.loads((tmp_path / "at150" / "page_1.json").read_text())

    assert page_150["width"] == round(PAGE_WIDTH_PTS * 150 / 72)
    assert page_150["height"] == round(PAGE_HEIGHT_PTS * 150 / 72)
    assert page_150["width"] * 2 == page_300["width"]

    for w300, w150 in zip(page_300["words"], page_150["words"], strict=True):
        assert w300["t"] == w150["t"]
        # Rounding to whole pixels allows 1px of slack per coordinate.
        for c300, c150 in zip(w300["l"], w150["l"], strict=True):
            assert abs(c300 / 2 - c150) <= 1, (w300["t"], w300["l"], w150["l"])
        # Every box stays inside its own page.
        assert w150["l"][2] <= page_150["width"]
        assert w150["l"][3] <= page_150["height"]


def test_extract_text_digital_writes_compact_json(tmp_path: Path, text_pdf: Path) -> None:
    """Per-page JSON must be one-line/no-pretty-print to keep large workspaces small."""
    out_dir = tmp_path / "page_text"
    extract_text_digital(text_pdf, out_dir, file_id="testfile1234")
    body = (out_dir / "page_1.json").read_text()
    # One newline at end of file, no internal newlines or indentation.
    assert body.count("\n") == 1
    assert ", " not in body  # compact separators
    assert ": " not in body


def test_extract_text_digital_clears_stale_files(tmp_path: Path, text_pdf: Path) -> None:
    out_dir = tmp_path / "page_text"
    out_dir.mkdir()
    stale = out_dir / "page_99.json"
    stale.write_text("{}")
    extract_text_digital(text_pdf, out_dir, file_id="x")
    assert not stale.exists()


def test_file_add_digital_default_extracts_text(workspace: Workspace, text_pdf: Path) -> None:
    pytest.importorskip("pdfminer")
    # Use FileStore directly so we don't depend on Ghostscript for this test.
    # _render_pages will record a page_render_error if gs is missing; text
    # extraction is independent.
    result = FileStore(workspace).add(text_pdf)
    assert result.created is True
    assert result.record.text_mode == TextMode.DIGITAL.value
    assert result.text_extraction_error is None
    assert result.text_extraction is not None
    assert result.text_extraction["mode"] == TextMode.DIGITAL.value
    assert result.text_extraction["pages_with_words"] == 2

    assert len(_page_texts(workspace, result.record.id)) == 2


def test_file_add_partial_empty_pdf_soft_fails_non_permanent(
    workspace: Workspace, mixed_pdf: Path
) -> None:
    """When one page has text and another doesn't, ingestion soft-fails
    with a *non-permanent* error so ``dgml check`` can re-attempt later
    (e.g., after pdfminer is upgraded)."""
    result = FileStore(workspace).add(mixed_pdf)
    assert result.created is True
    assert result.text_extraction_error is not None
    assert "1/2 pages had no extractable digital text" in result.text_extraction_error
    assert result.text_extraction is not None
    assert result.text_extraction["pages_with_words"] == 1
    assert result.text_extraction["pages_written"] == 2

    errors = load_recorded_errors(workspace, result.record.id)
    text_errs = [e for e in errors if e.operation == "text_extraction"]
    assert text_errs and all(not e.permanent for e in text_errs)


def test_check_partial_empty_re_extract_records_non_permanent(
    workspace: Workspace, mixed_pdf: Path
) -> None:
    """When ``dgml check`` re-extracts a partial-empty PDF, the partial-empty
    signal must be re-recorded (not silently reported as 'repaired')."""
    f = FileStore(workspace).add(mixed_pdf)
    # Drop the page_text JSONs so check is forced to re-extract.
    workspace.blobs.delete_blobs(layout.file_text_prefix(f.record.id))

    report = check_workspace(workspace)
    issues = [i for i in report.issues if i.kind == "page_text_count_mismatch"]
    assert issues, report.to_json()
    # Re-extraction restored the JSONs but the file is still degraded —
    # do NOT report this as `repaired`.
    assert not any(i.repaired for i in issues)
    assert any("had no extractable digital text" in i.message for i in issues)

    errors = load_recorded_errors(workspace, f.record.id)
    text_errs = [e for e in errors if e.operation == "text_extraction"]
    assert text_errs and any(not e.permanent for e in text_errs)


def test_file_add_blank_pdf_soft_fails_text_extraction(
    workspace: Workspace, sample_pdf: Path
) -> None:
    result = FileStore(workspace).add(sample_pdf)
    assert result.created is True
    assert result.text_extraction_error is not None
    assert "no digital text" in result.text_extraction_error.lower()

    errors = load_recorded_errors(workspace, result.record.id)
    text_errs = [e for e in errors if e.operation == "text_extraction"]
    assert text_errs
    assert any(e.permanent for e in text_errs)


def test_check_retry_errors_re_runs_text_extraction(workspace: Workspace, sample_pdf: Path) -> None:
    f = FileStore(workspace).add(sample_pdf)
    # Confirm permanent text-extraction error was recorded.
    errs = load_recorded_errors(workspace, f.record.id)
    assert any(e.operation == "text_extraction" and e.permanent for e in errs)

    # Without --retry-errors: the permanent marker blocks re-attempts.
    report = check_workspace(workspace)
    kinds = [i.kind for i in report.issues]
    assert "text_extraction_failed_permanent" in kinds

    # Delete the page_text JSONs to simulate missing extraction output and
    # force the consistency check to attempt re-extraction.
    workspace.blobs.delete_blobs(layout.file_text_prefix(f.record.id))

    # With --retry-errors: the marker is cleared; re-extraction runs and (for
    # a blank PDF) re-records the same permanent failure.
    report = check_workspace(workspace, retry_errors=True)
    kinds = [i.kind for i in report.issues]
    assert "text_extraction_failed" in kinds


def test_check_repairs_missing_page_text_for_digital_pdf(
    workspace: Workspace, text_pdf: Path
) -> None:
    """When page_text JSONs are missing but the PDF has digital text,
    --retry-errors should re-extract and mark the issue repaired."""
    f = FileStore(workspace).add(text_pdf)
    assert f.text_extraction_error is None
    workspace.blobs.delete_blobs(layout.file_text_prefix(f.record.id))

    report = check_workspace(workspace)
    repaired = [i for i in report.issues if i.kind == "page_text_count_mismatch" and i.repaired]
    assert repaired, report.to_json()
    assert len(_page_texts(workspace, f.record.id)) == 2


def test_check_re_extracts_at_the_files_own_dpi(workspace: Workspace, text_pdf: Path) -> None:
    """Repairs must reproduce the file's geometry, not today's default.

    A file added at 150 dpi has 150-dpi page images on disk. Re-extracting its
    text at the 300-dpi default would leave page_text/ boxes pointing outside
    those images — a silent grounding break introduced by the repair itself.
    """
    f = FileStore(workspace).add(text_pdf, dpi=150)
    assert f.record.page_image_dpi == 150
    before = workspace.read_page_text(f.record.id, 1)
    assert before is not None
    assert before["width"] == round(PAGE_WIDTH_PTS * 150 / 72)

    workspace.blobs.delete_blobs(layout.file_text_prefix(f.record.id))
    check_workspace(workspace)

    after = workspace.read_page_text(f.record.id, 1)
    assert after is not None
    assert after["width"] == before["width"]
    assert after["height"] == before["height"]
    assert after["words"] == before["words"]


def test_check_falls_back_to_the_default_dpi_for_legacy_records(
    workspace: Workspace, text_pdf: Path
) -> None:
    """Records written before page_image_dpi existed mean "the default was in force"."""
    f = FileStore(workspace).add(text_pdf)
    data = workspace.docs.get_doc("files", f.record.id)
    assert data is not None
    del data["page_image_dpi"]
    workspace.docs.put_doc("files", f.record.id, data)

    workspace.blobs.delete_blobs(layout.file_text_prefix(f.record.id))
    check_workspace(workspace)

    page = workspace.read_page_text(f.record.id, 1)
    assert page is not None
    assert page["width"] == round(PAGE_WIDTH_PTS * DEFAULT_DPI / 72)
