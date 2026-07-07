"""Document loading: PDF / convertible source → per-page-window slices."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dgml_core.conversion import ConverterConfig, convert_to_pdf_bytes
from dgml_core.pages import extract_pdf_pages


@dataclass
class PageSlice:
    """A subset of pages from the source document, packaged as a standalone PDF."""

    page_indices: list[int]
    pdf_bytes: bytes

    @property
    def first_page(self) -> int:
        return self.page_indices[0]

    @property
    def last_page(self) -> int:
        return self.page_indices[-1]


def load_pdf(path: Path) -> bytes:
    return Path(path).read_bytes()


def load_document_as_pdf(
    path: Path,
    *,
    converters: Mapping[str, ConverterConfig],
) -> bytes:
    """Return PDF bytes for a supported input.

    ``.pdf`` is loaded directly. A convertible source (docx/xlsx/…) is dispatched
    to the converter configured for its format family in ``converters`` (resolved
    from the workspace ``conversion`` config). A family with no configured
    converter — or an unknown extension — raises :class:`UnsupportedFileType`;
    there is no default converter.
    """
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    # Reuse the PDF persisted at ingest time (sibling `<stem>.pdf`), if present,
    # so the document is converted exactly once and what we slice here is
    # byte-identical to what the workspace page images were rendered from.
    # Falls through to on-demand conversion for files added before conversions
    # were persisted, or non-workspace inputs.
    converted = path.with_suffix(".pdf")
    if converted.exists():
        return load_pdf(converted)
    return convert_to_pdf_bytes(path, converters)


def slice_pdf(pdf_bytes: bytes, page_indices: list[int]) -> bytes:
    """Extract the given 0-based page indices into a new PDF and return its bytes.

    Slicing goes through ghostscript's ``pdfwrite`` device (see
    :func:`dgml.pages.extract_pdf_pages`); no Python PDF library is involved.
    """
    with tempfile.TemporaryDirectory(prefix="dgml-slice-") as tmp:
        tmpdir = Path(tmp)
        src = tmpdir / "in.pdf"
        out = tmpdir / "out.pdf"
        src.write_bytes(pdf_bytes)
        extract_pdf_pages(src, out, [i + 1 for i in page_indices])
        return out.read_bytes()


def iter_windows(total_pages: int, window_size: int, overlap: int) -> list[list[int]]:
    """Yield page-index lists for each window.

    First window: pages [0, window_size).
    Subsequent windows: start `overlap` pages earlier so the model sees continuity.
    """
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be in [0, window_size)")

    windows: list[list[int]] = []
    start = 0
    while start < total_pages:
        end = min(start + window_size, total_pages)
        windows.append(list(range(start, end)))
        if end >= total_pages:
            break
        start = end - overlap
    return windows
