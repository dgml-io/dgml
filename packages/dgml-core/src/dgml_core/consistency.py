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

"""Workspace consistency check with persistent error recording."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import (
    AuthError,
    CorruptMetadata,
    DgmlError,
    GhostscriptNotFound,
    OcrFailed,
    PageRenderFailed,
    RecordedError,
    TextExtractionFailed,
    append_recorded_error,
    clear_recorded_errors,
    load_recorded_errors,
    now_iso,
)
from .hashing import sha256_file
from .hybrid import extract_text_hybrid
from .ocr import extract_text_ocr, load_ocr_config
from .pages import PAGE_GLOB, render_pages
from .storage import Workspace, read_json
from .text_extraction import (
    PAGE_TEXT_GLOB,
    ExtractDigitalResult,
    TextMode,
    classify_extraction_outcome,
    extract_text_digital,
)
from .text_extraction_config import load_text_extraction_config


@dataclass
class Issue:
    kind: str
    target_type: str  # "file" | "docset"
    target_id: str
    message: str
    repaired: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "message": self.message,
            "repaired": self.repaired,
        }


@dataclass
class CheckReport:
    issues: list[Issue] = field(default_factory=list)
    files_checked: int = 0
    docsets_checked: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "files_checked": self.files_checked,
            "docsets_checked": self.docsets_checked,
            "issue_count": len(self.issues),
            "issues": [i.to_json() for i in self.issues],
        }

    @property
    def ok(self) -> bool:
        return not self.issues


def check_workspace(
    ws: Workspace, *, retry_errors: bool = False, verbose: bool = False
) -> CheckReport:
    """Validate the on-disk workspace; repair fixable issues where safe.

    With ``retry_errors=True``, recorded permanent errors are cleared before
    checking so that previously-failed operations are re-attempted.
    ``verbose`` is forwarded to hybrid re-extraction (the only path that
    currently produces optional stderr diagnostics).
    """
    report = CheckReport()

    if ws.files_dir.exists():
        for entry in sorted(ws.files_dir.iterdir()):
            if not entry.is_dir():
                continue
            report.files_checked += 1
            _check_file(ws, entry.name, retry_errors=retry_errors, verbose=verbose, report=report)

    if ws.docsets_dir.exists():
        for entry in sorted(ws.docsets_dir.iterdir()):
            if not entry.is_dir():
                continue
            report.docsets_checked += 1
            _check_docset(ws, entry.name, report=report)

    return report


def _check_file(
    ws: Workspace,
    file_id: str,
    *,
    retry_errors: bool,
    verbose: bool,
    report: CheckReport,
) -> None:
    file_dir = ws.file_dir(file_id)
    json_path = ws.file_json_path(file_id)
    errors_path = ws.file_errors_path(file_id)

    if retry_errors:
        clear_recorded_errors(errors_path)

    if not json_path.exists():
        report.issues.append(
            Issue(
                kind="missing_metadata",
                target_type="file",
                target_id=file_id,
                message="file.json missing",
            )
        )
        return

    try:
        record_data = read_json(json_path)
    except CorruptMetadata as exc:
        report.issues.append(
            Issue(
                kind="corrupt_metadata",
                target_type="file",
                target_id=file_id,
                message=str(exc),
            )
        )
        return

    sha = record_data.get("sha256")
    page_count: int | None = record_data.get("page_count")
    original_filename = record_data.get("original_filename")

    if not original_filename:
        report.issues.append(
            Issue(
                kind="corrupt_metadata",
                target_type="file",
                target_id=file_id,
                message="file.json is missing 'original_filename'",
            )
        )
        return

    pdf_path = file_dir / original_filename
    if not pdf_path.exists():
        report.issues.append(
            Issue(
                kind="missing_pdf",
                target_type="file",
                target_id=file_id,
                message=f"PDF '{original_filename}' missing from file directory",
            )
        )
        return

    if sha:
        actual_sha = sha256_file(pdf_path)
        if actual_sha != sha:
            report.issues.append(
                Issue(
                    kind="hash_mismatch",
                    target_type="file",
                    target_id=file_id,
                    message="stored sha256 does not match actual content",
                )
            )

    recorded = load_recorded_errors(errors_path)
    permanent_ops = {e.operation for e in recorded if e.permanent}

    pages_dir = ws.file_pages_dir(file_id)
    rendered_pages = sorted(pages_dir.glob(PAGE_GLOB)) if pages_dir.exists() else []

    expected: int | None
    # A stored ``page_count`` of 0 is never legitimate — every PDF has at
    # least one page. pdfminer's page-tree walk can silently yield 0 for PDFs
    # that ghostscript still renders fine, and that bogus 0 then gets persisted
    # at add time. Treat 0 the same as "no stored count" so we recover the true
    # count from the rendered pages rather than trusting the 0 as authoritative.
    if page_count:
        expected = page_count
    elif "pdf_page_count" in permanent_ops:
        # We previously failed to read the page count; don't keep retrying.
        report.issues.append(
            Issue(
                kind="pdf_unreadable_permanent",
                target_type="file",
                target_id=file_id,
                message="page count previously failed to read; not retried without --retry-errors",
            )
        )
        return
    else:
        # No reliable stored page count (the original add couldn't parse one,
        # or stored a bogus 0, though page rendering may still have succeeded).
        # Recover the count from the page images already on disk — ghostscript
        # renders one image per page, so the rendered set is authoritative —
        # rather than re-parsing the PDF. If nothing is on disk yet, attempt a
        # render to recover it: the count may have failed while rendering would
        # still succeed.
        expected = len(rendered_pages)
        if not expected:
            recovered = _recover_missing_pages(
                pdf_path=pdf_path,
                pages_dir=pages_dir,
                errors_path=errors_path,
                permanent_ops=permanent_ops,
                file_id=file_id,
                report=report,
            )
            if not recovered:
                return  # issue already recorded by the helper
            expected = recovered
            rendered_pages = sorted(pages_dir.glob(PAGE_GLOB))

    _check_page_rendering(
        pdf_path=pdf_path,
        pages_dir=pages_dir,
        rendered_pages=rendered_pages,
        expected=expected,
        errors_path=errors_path,
        permanent_ops=permanent_ops,
        file_id=file_id,
        report=report,
    )

    text_mode = record_data.get("text_mode")
    if text_mode in (TextMode.DIGITAL.value, TextMode.OCR.value, TextMode.HYBRID.value):
        # Permanent ops set is captured *before* page-rendering may have added
        # a render_pages permanent error this run; that's fine — text
        # extraction is independent of page rendering and we want to refresh
        # the set for the text-extraction check.
        permanent_ops = {e.operation for e in load_recorded_errors(errors_path) if e.permanent}
        _check_text_extraction(
            ws=ws,
            pdf_path=pdf_path,
            text_dir=ws.file_text_dir(file_id),
            expected=expected,
            errors_path=errors_path,
            permanent_ops=permanent_ops,
            file_id=file_id,
            text_mode=text_mode,
            verbose=verbose,
            report=report,
        )


def _recover_missing_pages(
    *,
    pdf_path: Path,
    pages_dir: Path,
    errors_path: Path,
    permanent_ops: set[str],
    file_id: str,
    report: CheckReport,
) -> int:
    """Recover a file whose stored page count is unknown/bogus and which has
    no rendered pages on disk, by attempting a fresh render.

    Ghostscript is the authority on how many pages a PDF has, so a successful
    render establishes the true count. Returns the number of pages rendered,
    or 0 if it could not be recovered — in which case an explanatory ``Issue``
    has already been appended to ``report``.
    """
    if "render_pages" in permanent_ops:
        report.issues.append(
            Issue(
                kind="page_render_failed_permanent",
                target_type="file",
                target_id=file_id,
                message="page rendering previously failed permanently; no pages on disk",
            )
        )
        return 0

    try:
        actual = render_pages(pdf_path, pages_dir)
    except (GhostscriptNotFound, PageRenderFailed) as exc:
        append_recorded_error(
            errors_path,
            RecordedError(
                operation="render_pages",
                message=str(exc),
                occurred_at=now_iso(),
                permanent=True,
            ),
        )
        report.issues.append(
            Issue(
                kind="page_render_failed",
                target_type="file",
                target_id=file_id,
                message=str(exc),
            )
        )
        return 0

    if not actual:
        report.issues.append(
            Issue(
                kind="pdf_unreadable",
                target_type="file",
                target_id=file_id,
                message="page count unavailable and ghostscript rendered no pages",
            )
        )
        return 0

    report.issues.append(
        Issue(
            kind="page_count_mismatch",
            target_type="file",
            target_id=file_id,
            message=f"recovered {actual} pages (stored count was unavailable)",
            repaired=True,
        )
    )
    return actual


def _check_page_rendering(
    *,
    pdf_path: Path,
    pages_dir: Path,
    rendered_pages: list[Path],
    expected: int,
    errors_path: Path,
    permanent_ops: set[str],
    file_id: str,
    report: CheckReport,
) -> None:
    if len(rendered_pages) == expected:
        return

    if "render_pages" in permanent_ops:
        report.issues.append(
            Issue(
                kind="page_render_failed_permanent",
                target_type="file",
                target_id=file_id,
                message=(
                    f"page rendering previously failed permanently; have "
                    f"{len(rendered_pages)}/{expected} pages"
                ),
            )
        )
        return

    try:
        actual = render_pages(pdf_path, pages_dir)
    except (GhostscriptNotFound, PageRenderFailed) as exc:
        append_recorded_error(
            errors_path,
            RecordedError(
                operation="render_pages",
                message=str(exc),
                occurred_at=now_iso(),
                permanent=True,
            ),
        )
        report.issues.append(
            Issue(
                kind="page_render_failed",
                target_type="file",
                target_id=file_id,
                message=str(exc),
            )
        )
        return

    if actual != expected:
        append_recorded_error(
            errors_path,
            RecordedError(
                operation="render_pages",
                message=f"rendered {actual}, expected {expected}",
                occurred_at=now_iso(),
                permanent=False,
            ),
        )
        report.issues.append(
            Issue(
                kind="page_count_mismatch",
                target_type="file",
                target_id=file_id,
                message=f"rendered {actual} pages, expected {expected}",
            )
        )
    else:
        report.issues.append(
            Issue(
                kind="page_count_mismatch",
                target_type="file",
                target_id=file_id,
                message=f"re-rendered {actual} pages",
                repaired=True,
            )
        )


def _check_text_extraction(
    *,
    ws: Workspace,
    pdf_path: Path,
    text_dir: Path,
    expected: int,
    errors_path: Path,
    permanent_ops: set[str],
    file_id: str,
    text_mode: str,
    verbose: bool,
    report: CheckReport,
) -> None:
    text_files = sorted(text_dir.glob(PAGE_TEXT_GLOB)) if text_dir.exists() else []

    corrupt = [p for p in text_files if not _is_valid_text_json(p)]
    for p in corrupt:
        report.issues.append(
            Issue(
                kind="page_text_corrupt",
                target_type="file",
                target_id=file_id,
                message=f"{p.name} is not valid JSON",
            )
        )

    # A permanent text_extraction error means the file is in a known-bad
    # state (e.g. no digital text, OCR auth failure). Surface it
    # unconditionally — the existence of page_text JSONs (possibly empty)
    # doesn't mean the file is healthy.
    if "text_extraction" in permanent_ops:
        report.issues.append(
            Issue(
                kind="text_extraction_failed_permanent",
                target_type="file",
                target_id=file_id,
                message=(
                    f"text extraction previously failed permanently; have "
                    f"{len(text_files) - len(corrupt)}/{expected} valid page_text files"
                ),
            )
        )
        return

    if not corrupt and len(text_files) == expected:
        return

    try:
        result = _reextract(ws, pdf_path, text_dir, file_id, text_mode, verbose=verbose)
    except (TextExtractionFailed, OcrFailed, AuthError, DgmlError) as exc:
        append_recorded_error(
            errors_path,
            RecordedError(
                operation="text_extraction",
                message=str(exc),
                occurred_at=now_iso(),
                permanent=True,
            ),
        )
        report.issues.append(
            Issue(
                kind="text_extraction_failed",
                target_type="file",
                target_id=file_id,
                message=str(exc),
            )
        )
        return

    outcome = classify_extraction_outcome(result, expected)
    if outcome.message is None:
        report.issues.append(
            Issue(
                kind="page_text_count_mismatch",
                target_type="file",
                target_id=file_id,
                message=f"re-extracted text for {result.pages_written} pages",
                repaired=True,
            )
        )
        return

    append_recorded_error(
        errors_path,
        RecordedError(
            operation="text_extraction",
            message=outcome.message,
            occurred_at=now_iso(),
            permanent=outcome.permanent,
        ),
    )
    # Permanent outcomes are reported as text_extraction_failed (matches the
    # add-time vocabulary); transient ones as page_text_count_mismatch.
    report.issues.append(
        Issue(
            kind="text_extraction_failed" if outcome.permanent else "page_text_count_mismatch",
            target_type="file",
            target_id=file_id,
            message=outcome.message,
        )
    )


def _reextract(
    ws: Workspace,
    pdf_path: Path,
    text_dir: Path,
    file_id: str,
    text_mode: str,
    *,
    verbose: bool = False,
) -> ExtractDigitalResult:
    """Re-extract text for ``file_id`` using whichever mode it was added with."""
    if text_mode == TextMode.OCR.value:
        config = load_ocr_config(ws)
        return extract_text_ocr(
            pdf_path,
            text_dir,
            file_id=file_id,
            page_images_dir=ws.file_pages_dir(file_id),
            config=config,
        )
    if text_mode == TextMode.HYBRID.value:
        config = load_ocr_config(ws)
        text_extraction_config = load_text_extraction_config(ws)
        return extract_text_hybrid(
            pdf_path,
            text_dir,
            file_id=file_id,
            page_images_dir=ws.file_pages_dir(file_id),
            config=config,
            text_extraction_config=text_extraction_config,
            workspace=ws,
            verbose=verbose,
        )
    return extract_text_digital(pdf_path, text_dir, file_id=file_id)


def _is_valid_text_json(path: Path) -> bool:
    try:
        data = read_json(path)
    except CorruptMetadata:
        return False
    return isinstance(data, dict) and "words" in data and "page" in data


def _check_docset(ws: Workspace, docset_id: str, *, report: CheckReport) -> None:
    json_path = ws.docset_json_path(docset_id)
    if not json_path.exists():
        report.issues.append(
            Issue(
                kind="missing_metadata",
                target_type="docset",
                target_id=docset_id,
                message="docset.json missing",
            )
        )
        return

    try:
        read_json(json_path)
    except CorruptMetadata as exc:
        report.issues.append(
            Issue(
                kind="corrupt_metadata",
                target_type="docset",
                target_id=docset_id,
                message=str(exc),
            )
        )
        return

    files_dir = ws.docset_files_dir(docset_id)
    if not files_dir.exists():
        return
    for ref in sorted(files_dir.iterdir()):
        if not ref.is_dir():
            continue
        if not ws.file_dir(ref.name).exists():
            report.issues.append(
                Issue(
                    kind="dangling_file_reference",
                    target_type="docset",
                    target_id=docset_id,
                    message=f"references missing file '{ref.name}'",
                )
            )
            continue
        _check_computed_attribution(ref, docset_id=docset_id, file_id=ref.name, report=report)


def _check_computed_attribution(
    ref_dir: Path, *, docset_id: str, file_id: str, report: CheckReport
) -> None:
    """Flag ``dg:origin="computed"`` elements with no ``dg:href`` in the
    file's DGML XML.

    A computed field's value is verifiable only by walking its ``dg:href``
    sources and recomputing the derivation (spec §13); one with no sources is
    an unauditable claim — usually the model derived it from document content
    the schema never extracted. Malformed XML is skipped here: XML validity
    is owned by the generation/extraction writers, not this check."""
    from .extraction_xml import unattributed_computed_fields

    for xml_path in sorted(ref_dir.glob("*.dgml.xml")):
        try:
            tags = unattributed_computed_fields(xml_path.read_bytes())
        except Exception:
            continue
        if tags:
            report.issues.append(
                Issue(
                    kind="computed_field_unattributed",
                    target_type="docset",
                    target_id=docset_id,
                    message=(
                        f"file '{file_id}' {xml_path.name}: computed element(s) with no "
                        f"dg:href sources: {', '.join(sorted(set(tags)))}"
                    ),
                )
            )
