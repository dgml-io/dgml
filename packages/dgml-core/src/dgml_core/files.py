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

"""File CRUD operations."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .conversion import (
    convert_to_pdf_bytes,
    converter_name_for_path,
    family_for_suffix,
    load_conversion_config,
)
from .errors import (
    AuthError,
    ConflictError,
    CorruptMetadata,
    DgmlError,
    FileNotFound,
    InvalidArgument,
    InvalidPDF,
    OcrFailed,
    PageRenderFailed,
    RecordedError,
    TextExtractionFailed,
    UnsupportedFileType,
    append_recorded_error,
    now_iso,
)
from .hashing import sha256_file
from .hybrid import extract_text_hybrid
from .ids import new_id
from .models import FileRecord
from .ocr import extract_text_ocr, load_ocr_config
from .pages import DEFAULT_DPI, RENDERER_NAME, pdf_page_count, render_pages
from .storage import Workspace, read_json, write_json_atomic
from .text_extraction import TextMode, classify_extraction_outcome, extract_text_digital
from .text_extraction_config import load_text_extraction_config

PDF_MAGIC = b"%PDF-"


class ConflictPolicy(StrEnum):
    """How :meth:`FileStore.add` reacts to an existing duplicate."""

    ERROR = "error"  # Default — refuse and raise.
    SKIP = "skip"  # Return the existing record, do nothing.
    REPLACE = "replace"  # On path-conflict: delete old, add new.
    DUPLICATE = "duplicate"  # Always create a new record.


@dataclass
class AddFileResult:
    record: FileRecord
    created: bool  # False if an existing record was returned.
    conflict_kind: str | None = None  # "hash" | "path" | None.
    page_render_error: str | None = None
    page_count_error: str | None = None
    text_extraction_error: str | None = None
    conversion_error: str | None = None
    text_extraction: dict[str, Any] | None = field(default=None)
    note: str | None = None


def _validate_pdf(path: Path) -> None:
    """Validate that a ``.pdf`` source has the PDF magic header.

    The suffix is already known to be ``.pdf`` by the caller
    (:meth:`FileStore._validate_source`), which routes non-PDF sources to the
    converter path; this only guards against a mislabeled/corrupt PDF.
    """
    with path.open("rb") as fh:
        magic = fh.read(len(PDF_MAGIC))
    if magic != PDF_MAGIC:
        raise InvalidPDF(f"{path} does not start with the PDF magic header")


class FileStore:
    """CRUD for files in a workspace."""

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    def list_all(self) -> list[FileRecord]:
        if not self.ws.files_dir.exists():
            return []
        records: list[FileRecord] = []
        for entry in sorted(self.ws.files_dir.iterdir()):
            if not entry.is_dir():
                continue
            json_path = entry / "file.json"
            if not json_path.exists():
                continue
            try:
                data = read_json(json_path)
            except CorruptMetadata:
                continue
            records.append(FileRecord.from_json(data))
        return records

    def get(self, file_id: str) -> FileRecord:
        if not file_id.strip():
            raise InvalidArgument("file id must not be empty")
        json_path = self.ws.file_json_path(file_id)
        if not json_path.exists():
            raise FileNotFound(f"file '{file_id}' not found")
        return FileRecord.from_json(read_json(json_path))

    def _find_conflicts(
        self, sha256: str, original_path: str
    ) -> tuple[FileRecord | None, FileRecord | None]:
        """Single-pass scan for both hash- and path-conflicts."""
        same_hash: FileRecord | None = None
        same_path: FileRecord | None = None
        for record in self.list_all():
            if same_hash is None and record.sha256 == sha256:
                same_hash = record
            if same_path is None and record.original_path == original_path:
                same_path = record
            if same_hash is not None and same_path is not None:
                break
        return same_hash, same_path

    def add(
        self,
        source_path: Path,
        *,
        on_conflict: ConflictPolicy = ConflictPolicy.ERROR,
        text_mode: TextMode = TextMode.DIGITAL,
        verbose: bool = False,
        debug: bool = False,
    ) -> AddFileResult:
        if text_mode in (TextMode.OCR, TextMode.HYBRID):
            # Validate OCR config *before* touching the filesystem so a
            # rejected add leaves the workspace untouched. Hybrid needs OCR
            # too — it runs digital + OCR per page and merges the results.
            load_ocr_config(self.ws)

        source_path = Path(source_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFound(f"source file does not exist: {source_path}")
        self._validate_source(source_path)

        digest = sha256_file(source_path)
        original_path = self._relative_original_path(source_path)
        same_hash, same_path = self._find_conflicts(digest, original_path)

        if same_hash is not None:
            if on_conflict is ConflictPolicy.ERROR:
                raise ConflictError(
                    f"a file with identical content already exists as '{same_hash.id}'",
                    kind="hash",
                    existing_id=same_hash.id,
                )
            if on_conflict is ConflictPolicy.SKIP:
                return AddFileResult(
                    record=same_hash,
                    created=False,
                    conflict_kind="hash",
                    note="existing record returned (identical content)",
                )
            if on_conflict is ConflictPolicy.REPLACE:
                return AddFileResult(
                    record=same_hash,
                    created=False,
                    conflict_kind="hash",
                    note="replace is a no-op when content is identical; existing record returned",
                )
            # DUPLICATE — fall through and create a new record.
        elif same_path is not None:
            if on_conflict is ConflictPolicy.ERROR:
                raise ConflictError(
                    f"a different file with the same source path already exists as "
                    f"'{same_path.id}'",
                    kind="path",
                    existing_id=same_path.id,
                )
            if on_conflict is ConflictPolicy.SKIP:
                return AddFileResult(
                    record=same_path,
                    created=False,
                    conflict_kind="path",
                    note="existing record returned (same source path, different content)",
                )
            if on_conflict is ConflictPolicy.REPLACE:
                self.delete(same_path.id)
            # DUPLICATE — fall through.

        return self._create_record(
            source_path,
            digest,
            original_path=original_path,
            conflict_kind=("hash" if same_hash else "path" if same_path else None),
            text_mode=text_mode,
            verbose=verbose,
            debug=debug,
        )

    def _relative_original_path(self, source_path: Path) -> str:
        """The source's location as a path relative to the workspace root.

        Storing it relative (e.g. ``../files/report.pdf``) keeps a workspace
        portable: it can be moved or checked into a repo on another machine
        and ``original_path`` still points at the source alongside it. Falls
        back to the absolute path only when no relative path exists (a
        different drive on Windows), which ``os.path.relpath`` signals with
        ``ValueError``.
        """
        try:
            return os.path.relpath(source_path, self.ws.root)
        except ValueError:
            return str(source_path)

    def _validate_source(self, source_path: Path) -> None:
        """Reject a source the workspace can't ingest, before any filesystem work.

        A ``.pdf`` must have the PDF magic header. A convertible source
        (docx/xlsx/…) is accepted only if its format family has a converter
        configured in the workspace ``conversion`` config; otherwise it is an
        :class:`UnsupportedFileType`. There is no default converter.
        """
        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            _validate_pdf(source_path)
            return
        family = family_for_suffix(suffix)
        if family is None:
            raise UnsupportedFileType(
                f"unsupported file type '{suffix or '<no extension>'}' "
                "(supported: .pdf, plus .docx/.doc/.xlsx/.xls with a converter configured)"
            )
        # Resolving the config validates it and proves a converter is wired up
        # for this family — but does not yet construct it (no binary/SDK touch).
        if family not in load_conversion_config(self.ws):
            raise UnsupportedFileType(
                f"no converter configured for '{suffix}'; set conversion.{family}.provider "
                "in config.json (see the translators-pdf package for ready-made converters)"
            )

    def _create_record(
        self,
        source_path: Path,
        digest: str,
        *,
        original_path: str,
        conflict_kind: str | None,
        text_mode: TextMode,
        verbose: bool = False,
        debug: bool = False,
    ) -> AddFileResult:
        file_id = new_id()
        file_dir = self.ws.file_dir(file_id)
        file_dir.mkdir(parents=True, exist_ok=False)
        # The original source is stored under its own name. A convertible
        # source is converted to a PDF here (persisted alongside it as
        # `<stem>.pdf` by _ensure_pdf) to drive page rendering / count / text
        # extraction; generation later reuses that same persisted PDF.
        dest_source = file_dir / source_path.name
        shutil.copy2(source_path, dest_source)

        pdf_path, conversion_error, pdf_converter = self._ensure_pdf(dest_source, file_id)
        if pdf_path is None:
            record = FileRecord(
                id=file_id,
                original_path=original_path,
                original_filename=source_path.name,
                sha256=digest,
                added_at=now_iso(),
                page_count=None,
                text_mode=text_mode.value,
                pdf_converter=pdf_converter,
            )
            write_json_atomic(self.ws.file_json_path(file_id), record.to_json())
            return AddFileResult(
                record=record,
                created=True,
                conflict_kind=conflict_kind,
                conversion_error=conversion_error,
            )

        page_count, page_count_error = self._safe_page_count(pdf_path, file_id)
        page_render_error = self._render_pages(pdf_path, file_id, expected=page_count)
        text_extraction_error, text_summary = self._extract_text(
            pdf_path,
            file_id,
            text_mode=text_mode,
            page_count=page_count,
            verbose=verbose,
            debug=debug,
        )

        record = FileRecord(
            id=file_id,
            original_path=original_path,
            original_filename=source_path.name,
            sha256=digest,
            added_at=now_iso(),
            page_count=page_count,
            text_mode=text_mode.value,
            page_image_dpi=DEFAULT_DPI,
            page_image_renderer=RENDERER_NAME,
            pdf_converter=pdf_converter,
        )
        write_json_atomic(self.ws.file_json_path(file_id), record.to_json())
        return AddFileResult(
            record=record,
            created=True,
            conflict_kind=conflict_kind,
            page_render_error=page_render_error,
            page_count_error=page_count_error,
            text_extraction_error=text_extraction_error,
            text_extraction=text_summary,
        )

    def _ensure_pdf(
        self, stored_source: Path, file_id: str
    ) -> tuple[Path | None, str | None, str | None]:
        """Return ``(pdf_path, error, converter_name)`` for the stored source.

        For a ``.pdf`` source this is the stored file itself. For a convertible
        source it runs the configured converter and **persists** the resulting
        PDF alongside the original at ``<stem>.pdf`` in the file directory. That
        persisted PDF is what page rendering / count / text extraction run on
        here, and what generation later reuses (see
        :func:`dgml_core.generation.document.load_document_as_pdf`) — so the document
        is converted exactly once, and the bytes the page images were rendered
        from are byte-identical to those generation slices.

        The third element is the converter's name (``None`` for a ``.pdf``
        source), recorded on the file regardless of whether the conversion
        ultimately succeeded so a failed convert still names what was tried.

        On conversion failure a permanent error is recorded and
        ``(None, message, converter_name)`` is returned so the file record is
        still created (consistent with the page-render / text soft-fail pattern).
        """
        if stored_source.suffix.lower() == ".pdf":
            return stored_source, None, None

        converters = load_conversion_config(self.ws)
        converter_name = converter_name_for_path(stored_source, converters)
        try:
            pdf_bytes = convert_to_pdf_bytes(stored_source, converters)
        except DgmlError as exc:
            message = str(exc)
            append_recorded_error(
                self.ws.file_errors_path(file_id),
                RecordedError(
                    operation="convert_to_pdf",
                    message=message,
                    occurred_at=now_iso(),
                    permanent=True,
                ),
            )
            return None, message, converter_name

        pdf_path = stored_source.with_suffix(".pdf")
        pdf_path.write_bytes(pdf_bytes)
        return pdf_path, None, converter_name

    def _safe_page_count(self, pdf_path: Path, file_id: str) -> tuple[int | None, str | None]:
        """Read the PDF's page count. On failure, record a permanent error
        and return ``(None, message)`` so the file record is still created."""
        try:
            return pdf_page_count(pdf_path), None
        except Exception as exc:  # pdfminer can raise a variety of errors.
            message = f"could not read PDF page count: {type(exc).__name__}: {exc}"
            append_recorded_error(
                self.ws.file_errors_path(file_id),
                RecordedError(
                    operation="pdf_page_count",
                    message=message,
                    occurred_at=now_iso(),
                    permanent=True,
                ),
            )
            return None, message

    def _render_pages(self, pdf_path: Path, file_id: str, *, expected: int | None) -> str | None:
        """Render pages, recording errors. Returns a human-readable error
        message on failure or partial success, or ``None`` on full success."""
        try:
            rendered = render_pages(pdf_path, self.ws.file_pages_dir(file_id))
        except PageRenderFailed as exc:
            append_recorded_error(
                self.ws.file_errors_path(file_id),
                RecordedError(
                    operation="render_pages",
                    message=str(exc),
                    occurred_at=now_iso(),
                    permanent=True,
                ),
            )
            return str(exc)

        if expected is not None and rendered != expected:
            message = f"rendered {rendered} pages, PDF reports {expected}"
            append_recorded_error(
                self.ws.file_errors_path(file_id),
                RecordedError(
                    operation="render_pages",
                    message=message,
                    occurred_at=now_iso(),
                    permanent=False,
                ),
            )
            return message

        return None

    def _extract_text(
        self,
        pdf_path: Path,
        file_id: str,
        *,
        text_mode: TextMode,
        page_count: int | None,
        verbose: bool = False,
        debug: bool = False,
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Run text extraction for ``text_mode`` and record any failure.

        Returns ``(error_message_or_None, summary_dict_or_None)``. Follows the
        same soft-fail pattern as :meth:`_render_pages`: hard failures (no
        digital text, OCR API error, auth failure) are recorded as permanent
        errors; partial-extraction (some pages empty) is recorded as a
        non-permanent error so the next ``dgml check`` retries without
        ``--retry-errors``.
        """
        if text_mode is TextMode.DIGITAL:
            return self._extract_text_digital(pdf_path, file_id, page_count=page_count)
        if text_mode is TextMode.OCR:
            return self._extract_text_ocr(pdf_path, file_id, page_count=page_count)
        if text_mode is TextMode.HYBRID:
            return self._extract_text_hybrid(
                pdf_path, file_id, page_count=page_count, verbose=verbose, debug=debug
            )
        return None, None

    def _extract_text_digital(
        self,
        pdf_path: Path,
        file_id: str,
        *,
        page_count: int | None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        try:
            result = extract_text_digital(pdf_path, self.ws.file_text_dir(file_id), file_id=file_id)
        except TextExtractionFailed as exc:
            return self._record_text_failure(file_id, str(exc), permanent=True), None
        return self._classify_and_record(result, file_id, page_count, mode_label="digital")

    def _extract_text_ocr(
        self,
        pdf_path: Path,
        file_id: str,
        *,
        page_count: int | None,
    ) -> tuple[str | None, dict[str, Any] | None]:
        try:
            config = load_ocr_config(self.ws)
        except DgmlError as exc:
            # OcrConfigMissing / OcrConfigInvalid: permanent — workspace
            # config has to be fixed before retrying.
            return self._record_text_failure(file_id, str(exc), permanent=True), None

        try:
            result = extract_text_ocr(
                pdf_path,
                self.ws.file_text_dir(file_id),
                file_id=file_id,
                page_images_dir=self.ws.file_pages_dir(file_id),
                config=config,
            )
        except (OcrFailed, AuthError) as exc:
            # Provider/auth failures are recorded as permanent — re-running
            # without changing config or credentials won't help. `dgml check
            # --retry-errors` is the recovery path once the user fixes them.
            return self._record_text_failure(file_id, str(exc), permanent=True), None

        return self._classify_and_record(result, file_id, page_count, mode_label="ocr")

    def _extract_text_hybrid(
        self,
        pdf_path: Path,
        file_id: str,
        *,
        page_count: int | None,
        verbose: bool = False,
        debug: bool = False,
    ) -> tuple[str | None, dict[str, Any] | None]:
        try:
            config = load_ocr_config(self.ws)
            text_extraction_config = load_text_extraction_config(self.ws)
        except DgmlError as exc:
            return self._record_text_failure(file_id, str(exc), permanent=True), None

        try:
            result = extract_text_hybrid(
                pdf_path,
                self.ws.file_text_dir(file_id),
                file_id=file_id,
                page_images_dir=self.ws.file_pages_dir(file_id),
                config=config,
                text_extraction_config=text_extraction_config,
                workspace=self.ws,
                verbose=verbose,
                debug=debug,
            )
        except (OcrFailed, AuthError) as exc:
            return self._record_text_failure(file_id, str(exc), permanent=True), None

        return self._classify_and_record(result, file_id, page_count, mode_label="hybrid")

    def _classify_and_record(
        self,
        result: Any,
        file_id: str,
        page_count: int | None,
        *,
        mode_label: str,
    ) -> tuple[str | None, dict[str, Any] | None]:
        summary = result.to_summary()
        # The summary's ``mode`` field defaults to "digital" since
        # ExtractDigitalResult is shared; rewrite it for OCR runs so the
        # CLI/test_cli payload reflects how the text was actually produced.
        summary["mode"] = mode_label
        outcome = classify_extraction_outcome(result, page_count)
        if outcome.message is None:
            return None, summary

        self._record_text_failure(file_id, outcome.message, permanent=outcome.permanent)
        return outcome.message, summary

    def _record_text_failure(self, file_id: str, message: str, *, permanent: bool) -> str:
        append_recorded_error(
            self.ws.file_errors_path(file_id),
            RecordedError(
                operation="text_extraction",
                message=message,
                occurred_at=now_iso(),
                permanent=permanent,
            ),
        )
        return message

    def delete(self, file_id: str) -> None:
        if not file_id.strip():
            raise InvalidArgument("file id must not be empty")
        if not self.ws.file_dir(file_id).exists():
            raise FileNotFound(f"file '{file_id}' not found")
        if self.ws.docsets_dir.exists():
            for docset_dir in self.ws.docsets_dir.iterdir():
                ref = docset_dir / "files" / file_id
                if ref.exists():
                    shutil.rmtree(ref)
        shutil.rmtree(self.ws.file_dir(file_id))
