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

import json
from dataclasses import dataclass, field
from typing import Any

from . import layout
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
from .hybrid import extract_text_hybrid
from .ocr import extract_text_ocr, load_ocr_config
from .pages import DEFAULT_DPI, render_pages
from .storage import Workspace
from .text_extraction import (
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


def _entity_ids(ws: Workspace, collection: str, blob_prefix: str) -> list[str]:
    """Every entity id in the workspace, store-natively.

    The union of two sources: ids with a readable manifest (``find_docs``), and
    *blob-orphans* — ids that have blobs under ``blob_prefix`` but whose manifest
    is missing or unreadable (so ``find_docs`` skipped them). The per-entity
    check then resolves each id's manifest and flags a missing/corrupt one. This
    is the store analogue of the old ``iterdir`` scan: an entity "exists" when it
    has a manifest *or* artifacts, not when a bare directory is present (a
    concept no remote store has)."""
    ids = {str(doc["id"]) for doc in ws.docs.find_docs(collection, {})}
    for key in ws.blobs.list_blobs(blob_prefix):
        segment = key[len(blob_prefix) :].split("/", 1)[0]
        if segment:
            ids.add(segment)
    return sorted(ids)


def check_workspace(
    ws: Workspace, *, retry_errors: bool = False, verbose: bool = False, debug: bool = False
) -> CheckReport:
    """Validate the workspace; repair fixable issues where safe.

    With ``retry_errors=True``, recorded permanent errors are cleared before
    checking so that previously-failed operations are re-attempted.
    ``verbose`` is forwarded to hybrid re-extraction (the only path that
    currently produces optional stderr diagnostics). ``debug`` is likewise
    forwarded so a hybrid LLM merge during re-extraction records its usage
    telemetry (only the hybrid path issues LLM calls).
    """
    report = CheckReport()

    for file_id in _entity_ids(ws, "files", "files/"):
        report.files_checked += 1
        _check_file(
            ws,
            file_id,
            retry_errors=retry_errors,
            verbose=verbose,
            debug=debug,
            report=report,
        )

    for docset_id in _entity_ids(ws, "docsets", "docsets/"):
        report.docsets_checked += 1
        _check_docset(ws, docset_id, report=report)

    return report


def _check_file(
    ws: Workspace,
    file_id: str,
    *,
    retry_errors: bool,
    verbose: bool,
    debug: bool,
    report: CheckReport,
) -> None:
    if retry_errors:
        clear_recorded_errors(ws, file_id)

    try:
        record_data = ws.docs.get_doc(layout.Collection.FILES, file_id)
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

    if record_data is None:
        report.issues.append(
            Issue(
                kind="missing_metadata",
                target_type="file",
                target_id=file_id,
                message="file.json missing",
            )
        )
        return

    sha = record_data.get("sha256")
    page_count: int | None = record_data.get("page_count")
    original_filename = record_data.get("original_filename")
    dpi = _recorded_dpi(record_data)

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

    source_key = layout.file_source_key(file_id, original_filename)
    if not ws.blobs.blob_exists(source_key):
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
        actual_sha = ws.blobs.sha256_blob(source_key)
        if actual_sha != sha:
            report.issues.append(
                Issue(
                    kind="hash_mismatch",
                    target_type="file",
                    target_id=file_id,
                    message="stored sha256 does not match actual content",
                )
            )

    recorded = load_recorded_errors(ws, file_id)
    permanent_ops = {e.operation for e in recorded if e.permanent}

    pages_prefix = layout.file_pages_prefix(file_id)
    rendered = len(ws.blobs.list_blobs(pages_prefix))

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
        # Recover the count from the page images already stored — ghostscript
        # renders one image per page, so the rendered set is authoritative —
        # rather than re-parsing the PDF. If nothing is stored yet, attempt a
        # render to recover it: the count may have failed while rendering would
        # still succeed.
        expected = rendered
        if not expected:
            recovered = _recover_missing_pages(
                ws=ws,
                source_key=source_key,
                pages_prefix=pages_prefix,
                permanent_ops=permanent_ops,
                file_id=file_id,
                dpi=dpi,
                report=report,
            )
            if not recovered:
                return  # issue already recorded by the helper
            expected = recovered
            rendered = len(ws.blobs.list_blobs(pages_prefix))

    _check_page_rendering(
        ws=ws,
        source_key=source_key,
        pages_prefix=pages_prefix,
        rendered=rendered,
        expected=expected,
        permanent_ops=permanent_ops,
        file_id=file_id,
        dpi=dpi,
        report=report,
    )

    text_mode = record_data.get("text_mode")
    if text_mode in (TextMode.DIGITAL.value, TextMode.OCR.value, TextMode.HYBRID.value):
        # Permanent ops set is captured *before* page-rendering may have added
        # a render_pages permanent error this run; that's fine — text
        # extraction is independent of page rendering and we want to refresh
        # the set for the text-extraction check.
        permanent_ops = {e.operation for e in load_recorded_errors(ws, file_id) if e.permanent}
        _check_text_extraction(
            ws=ws,
            source_key=source_key,
            expected=expected,
            permanent_ops=permanent_ops,
            file_id=file_id,
            text_mode=text_mode,
            dpi=dpi,
            verbose=verbose,
            debug=debug,
            report=report,
        )


def _render(ws: Workspace, source_key: str, pages_prefix: str, dpi: int) -> int:
    """Render the source PDF's page images through the store.

    Materialize the source to a real path (ghostscript needs one) and render
    into a store-backed staging directory; ``render_pages`` clears stale images
    itself. ``dpi`` reproduces the file's existing render resolution so repaired
    pages stay aligned with the ``page_text/`` boxes already stored. Returns the
    page count."""
    with ws.blobs.materialize(source_key) as pdf_path, ws.blobs.staged_write(pages_prefix) as tmp:
        return render_pages(pdf_path, tmp, dpi=dpi)


def _recorded_dpi(record_data: dict[str, Any]) -> int:
    """The DPI this file's pages were rendered at, per its own record.

    Every repair below has to reproduce the file's *existing* geometry rather
    than today's default: a re-render at a different DPI would leave the page
    images disagreeing with the ``page_text/`` boxes already on disk, and a
    re-extract at a different DPI would do the mirror image. ``page_image_dpi``
    is absent on records written before it existed and ``null`` when no pages
    were ever produced, both of which mean "the default was in force".
    """
    recorded = record_data.get("page_image_dpi")
    if isinstance(recorded, int) and not isinstance(recorded, bool) and recorded > 0:
        return recorded
    return DEFAULT_DPI


def _recover_missing_pages(
    *,
    ws: Workspace,
    source_key: str,
    pages_prefix: str,
    permanent_ops: set[str],
    file_id: str,
    dpi: int,
    report: CheckReport,
) -> int:
    """Recover a file whose stored page count is unknown/bogus and which has
    no rendered pages stored, by attempting a fresh render.

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
        actual = _render(ws, source_key, pages_prefix, dpi)
    except (GhostscriptNotFound, PageRenderFailed) as exc:
        append_recorded_error(
            ws,
            file_id,
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
    ws: Workspace,
    source_key: str,
    pages_prefix: str,
    rendered: int,
    expected: int,
    permanent_ops: set[str],
    file_id: str,
    dpi: int,
    report: CheckReport,
) -> None:
    if rendered == expected:
        return

    if "render_pages" in permanent_ops:
        report.issues.append(
            Issue(
                kind="page_render_failed_permanent",
                target_type="file",
                target_id=file_id,
                message=(
                    f"page rendering previously failed permanently; have "
                    f"{rendered}/{expected} pages"
                ),
            )
        )
        return

    try:
        actual = _render(ws, source_key, pages_prefix, dpi)
    except (GhostscriptNotFound, PageRenderFailed) as exc:
        append_recorded_error(
            ws,
            file_id,
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
            ws,
            file_id,
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
    source_key: str,
    expected: int,
    permanent_ops: set[str],
    file_id: str,
    text_mode: str,
    dpi: int,
    verbose: bool,
    debug: bool,
    report: CheckReport,
) -> None:
    text_keys = ws.blobs.list_blobs(layout.file_text_prefix(file_id))
    corrupt = [k for k in text_keys if not _is_valid_text_json(ws, k)]
    for k in corrupt:
        report.issues.append(
            Issue(
                kind="page_text_corrupt",
                target_type="file",
                target_id=file_id,
                message=f"{k.rsplit('/', 1)[-1]} is not valid JSON",
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
                    f"{len(text_keys) - len(corrupt)}/{expected} valid page_text files"
                ),
            )
        )
        return

    if not corrupt and len(text_keys) == expected:
        return

    try:
        result = _reextract(
            ws, source_key, file_id, text_mode, dpi=dpi, verbose=verbose, debug=debug
        )
    except (TextExtractionFailed, OcrFailed, AuthError, DgmlError) as exc:
        append_recorded_error(
            ws,
            file_id,
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
        ws,
        file_id,
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
    source_key: str,
    file_id: str,
    text_mode: str,
    *,
    dpi: int = DEFAULT_DPI,
    verbose: bool = False,
    debug: bool = False,
) -> ExtractDigitalResult:
    """Re-extract text for ``file_id`` using whichever mode it was added with.

    The source PDF is materialized for the extractors, page_text is written into
    a store-backed staging dir, and OCR/hybrid read the file's page images from a
    materialized copy — the same store bridges the file-add path uses.

    ``dpi`` is the file's *own* render resolution, not today's default: digital
    word boxes are written in page-image pixel space, so re-extracting at a
    different value would leave ``page_text/`` disagreeing with the
    ``page_images/`` already on disk. The pure-OCR path reads those images
    directly and so needs no dpi."""
    text_prefix = layout.file_text_prefix(file_id)
    pages_prefix = layout.file_pages_prefix(file_id)
    with (
        ws.blobs.materialize(source_key) as pdf_path,
        ws.blobs.staged_write(text_prefix) as text_dir,
    ):
        if text_mode == TextMode.OCR.value:
            config = load_ocr_config(ws)
            with ws.blobs.materialize_dir(pages_prefix) as pages_dir:
                return extract_text_ocr(
                    pdf_path,
                    text_dir,
                    file_id=file_id,
                    page_images_dir=pages_dir,
                    config=config,
                )
        if text_mode == TextMode.HYBRID.value:
            config = load_ocr_config(ws)
            text_extraction_config = load_text_extraction_config(ws)
            with ws.blobs.materialize_dir(pages_prefix) as pages_dir:
                return extract_text_hybrid(
                    pdf_path,
                    text_dir,
                    file_id=file_id,
                    page_images_dir=pages_dir,
                    config=config,
                    text_extraction_config=text_extraction_config,
                    workspace=ws,
                    dpi=dpi,
                    verbose=verbose,
                    debug=debug,
                )
        return extract_text_digital(pdf_path, text_dir, file_id=file_id, dpi=dpi)


def _is_valid_text_json(ws: Workspace, key: str) -> bool:
    try:
        data = json.loads(ws.blobs.get_blob(key))
    except (ValueError, FileNotFoundError):
        return False
    return isinstance(data, dict) and "words" in data and "page" in data


def _file_present(ws: Workspace, file_id: str) -> bool:
    """Whether ``file_id`` is a real entity — a readable/corrupt manifest, or any
    artifacts. Used by the dangling-reference check so an assignment to a file
    that has a manifest (even a broken one) or stored blobs is not "dangling"
    (its own corruption is reported by :func:`_check_file`)."""
    try:
        if ws.docs.get_doc(layout.Collection.FILES, file_id) is not None:
            return True
    except CorruptMetadata:
        return True
    return bool(ws.blobs.list_blobs(layout.file_prefix(file_id)))


def _check_docset(ws: Workspace, docset_id: str, *, report: CheckReport) -> None:
    try:
        record_data = ws.docs.get_doc(layout.Collection.DOCSETS, docset_id)
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

    if record_data is None:
        report.issues.append(
            Issue(
                kind="missing_metadata",
                target_type="docset",
                target_id=docset_id,
                message="docset.json missing",
            )
        )
        return

    for assignment in ws.docs.find_docs(layout.Collection.ASSIGNMENTS, {"docset_id": docset_id}):
        file_id = str(assignment["file_id"])
        if not _file_present(ws, file_id):
            report.issues.append(
                Issue(
                    kind="dangling_file_reference",
                    target_type="docset",
                    target_id=docset_id,
                    message=f"references missing file '{file_id}'",
                )
            )
            continue
        _check_computed_attribution(ws, docset_id=docset_id, file_id=file_id, report=report)


def _check_computed_attribution(
    ws: Workspace, *, docset_id: str, file_id: str, report: CheckReport
) -> None:
    """Flag ``dg:origin="computed"`` elements with no ``dg:href`` in the
    file's DGML XML.

    A computed field's value is verifiable only by walking its ``dg:href``
    sources and recomputing the derivation (spec §13); one with no sources is
    an unauditable claim — usually the model derived it from document content
    the schema never extracted. Malformed XML is skipped here: XML validity
    is owned by the generation/extraction writers, not this check."""
    from .extraction_xml import unattributed_computed_fields

    for key in sorted(ws.blobs.list_blobs(layout.docset_pair_prefix(docset_id, file_id))):
        if not key.endswith(".dgml.xml"):
            continue
        try:
            tags = unattributed_computed_fields(ws.blobs.get_blob(key))
        except Exception:
            continue
        if tags:
            report.issues.append(
                Issue(
                    kind="computed_field_unattributed",
                    target_type="docset",
                    target_id=docset_id,
                    message=(
                        f"file '{file_id}' {key.rsplit('/', 1)[-1]}: computed element(s) with no "
                        f"dg:href sources: {', '.join(sorted(set(tags)))}"
                    ),
                )
            )
