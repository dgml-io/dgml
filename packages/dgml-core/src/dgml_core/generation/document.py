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

"""Document loading: PDF / convertible source → per-page-window slices."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dgml_core.conversion import ConverterConfig, convert_to_pdf_bytes
from dgml_core.pages import PAGE_FILENAME_TEMPLATE, PAGE_GLOB, extract_pdf_pages


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


def count_pages_for(
    name: str,
    *,
    page_images_dirs: Mapping[str, Path] | None = None,
) -> int:
    """Page count for one document, from the pre-rendered workspace images.

    In a workspace ghostscript already rendered one PNG per page at file-add
    time, so the page count is implied by the files in ``page_images/`` — we
    count those instead of re-parsing the PDF. Raises if no page images are
    available for ``name``.
    """
    if page_images_dirs is not None and name in page_images_dirs:
        rendered = len(list(page_images_dirs[name].glob(PAGE_GLOB)))
        if rendered:
            return rendered
    raise ValueError(f"No page images found for {name!r}")


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


def sample_page_indices(total_pages: int, max_sample: int) -> list[int]:
    """Pick up to `max_sample` representative page indices spanning the document.

    For documents at or below the budget every page is returned. Otherwise the
    first and last pages are always included and the rest are spread evenly.
    """
    if max_sample <= 0:
        return []
    if total_pages <= max_sample:
        return list(range(total_pages))
    if max_sample == 1:
        return [0]
    step = (total_pages - 1) / (max_sample - 1)
    return sorted({round(i * step) for i in range(max_sample)})


def resolve_page_image_paths(
    name: str,
    total_pages: int,
    *,
    page_images_dirs: Mapping[str, Path] | None,
) -> list[Path]:
    """Return one PNG path per page (0-indexed list, parallel to PDF pages).

    Builds the path list from the pre-rendered workspace images: pages were
    rendered at ``file add`` time and never re-rasterized here. Raises if no
    page images are available for ``name``.
    """
    if page_images_dirs is not None and name in page_images_dirs:
        ws_dir = page_images_dirs[name]
        return [ws_dir / (PAGE_FILENAME_TEMPLATE % (i + 1)) for i in range(total_pages)]
    raise ValueError(f"No page images found for {name!r}")
