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

"""The PDFium engine: page rendering and page slicing, in-process.

An in-process alternative to the ghostscript subprocess: no system binary to
install, just ``pip install dgml[pdfium]``. pypdfium2 is (Apache-2.0 OR
BSD-3-Clause) and PDFium itself is BSD-3-Clause, both on the
direct-dependency allow-list; Pillow (MIT-CMU) writes the PNGs.

Rendering must correct two PDFium behaviours to stay interchangeable with
ghostscript — it sizes canvases from the CropBox, and rounds up in float —
see :meth:`Pypdfium2Renderer._render_page`. Slicing needs no such correction:
``import_pages`` copies page objects structurally, preserving the text layer.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .errors import EngineNotAvailable, PageRenderFailed, PdfSliceFailed
from .pages import (
    PAGE_FILENAME_TEMPLATE,
    PageRenderer,
    PdfConfig,
    PdfSlicer,
    pdf_page_count,
)

# PDF user space is 72 points per inch; PDFium takes a scale factor, not a dpi.
_PDF_POINTS_PER_INCH = 72.0


def _has_area(box: tuple[float, float, float, float] | None) -> bool:
    """Is ``box`` a usable page box — present and of positive area?"""
    if not box or len(box) != 4:
        return False
    left, bottom, right, top = box
    return right > left and top > bottom


def _true_page_count(pdf_path: Path, *, fallback: int) -> int:
    """Authoritative page count via pdfminer's page-tree walk.

    Falls back to ``fallback`` (PDFium's own count) when pdfminer cannot parse
    the file at all: PDFium opened it, so refusing to render would lose more
    than it protects.
    """
    try:
        return pdf_page_count(pdf_path)
    except Exception:
        return fallback


def _import_pdfium() -> Any:
    """Import and return ``pypdfium2``, or raise with an install hint.

    Shared by both capabilities: availability is a property of the engine, not
    of the operation, so a missing package reports identically whether the
    caller was rendering or slicing.
    """
    try:
        import pypdfium2
    except ImportError as exc:
        raise EngineNotAvailable(
            "pdf.provider is 'pypdfium2' but the pypdfium2 package is "
            "not installed — install it with: pip install dgml[pdfium]"
        ) from exc
    return pypdfium2


class Pypdfium2Renderer(PageRenderer):
    """Rasterize pages in-process with PDFium."""

    def __init__(self, config: PdfConfig) -> None:
        self._pdfium = _import_pdfium()
        # Pillow writes the PNGs (pypdfium2's to_pil()); it ships with the same
        # extra, so its absence means a partial/hand-rolled install.
        try:
            import PIL.Image  # noqa: F401
        except ImportError as exc:
            raise EngineNotAvailable(
                "pdf.provider is 'pypdfium2' but Pillow is not installed — "
                "install it with: pip install dgml[pdfium]"
            ) from exc

    def render(self, pdf_path: Path, output_dir: Path, *, dpi: int) -> None:
        scale = dpi / _PDF_POINTS_PER_INCH
        try:
            pdf = self._pdfium.PdfDocument(str(pdf_path))
        except Exception as exc:  # PdfiumError, or OSError on unreadable input
            raise PageRenderFailed(f"pypdfium2 could not open {pdf_path.name}: {exc}") from exc
        true_page_count: int | None = None
        try:
            for index in range(len(pdf)):
                page_num = index + 1
                try:
                    self._render_page(pdf[index], output_dir, page_num, dpi=dpi, scale=scale)
                except Exception as exc:
                    # PDFium's page count comes from the catalog's ``/Count``,
                    # which ``pdf_page_count`` deliberately distrusts (it walks
                    # the page tree instead). A failure here may simply be that
                    # ``/Count`` overstated reality and we have run past the
                    # last real page — where ghostscript renders the pages that
                    # do exist rather than failing the document. Consult the
                    # authoritative count only now, so a well-formed PDF never
                    # pays for the extra parse.
                    if true_page_count is None:
                        true_page_count = _true_page_count(pdf_path, fallback=len(pdf))
                    if page_num > true_page_count:
                        break
                    raise PageRenderFailed(
                        f"pypdfium2 failed rendering page {page_num} of {pdf_path.name}: {exc}"
                    ) from exc
        finally:
            pdf.close()

    def _render_page(
        self, page: Any, output_dir: Path, page_num: int, *, dpi: int, scale: float
    ) -> None:
        """Rasterize one page to ``page_<n>.png`` at exactly ``dpi``."""
        # PDFium sizes its canvas from the CropBox, but ghostscript rasterizes
        # the MediaBox and ``page_text/`` word boxes are MediaBox-relative. Left
        # alone, a PDF whose CropBox trims the MediaBox renders at a different
        # scale and origin than every box that points into it. Force the
        # MediaBox so renderer, ghostscript and page_text all agree. A
        # degenerate MediaBox is ignored rather than installed as a zero-area
        # CropBox, which would make the page unrenderable.
        mediabox = page.get_mediabox()
        if _has_area(mediabox) and page.get_cropbox() != mediabox:
            page.set_cropbox(*mediabox)

        # get_size() is already rotation-aware (a /Rotate 90 page reports
        # swapped dimensions), so the target is computed from it directly.
        width_pts, height_pts = page.get_size()
        target_w = round(width_pts * dpi / _PDF_POINTS_PER_INCH)
        target_h = round(height_pts * dpi / _PDF_POINTS_PER_INCH)

        # PDFium computes its canvas as ceil(pts * scale) in binary float, so an
        # exactly-integral size overshoots by a pixel: 792pt at 300dpi is
        # 3300.0000000000005, which ceils to 3301. Trim the overshoot via
        # PDFium's own crop rather than cropping the decoded image: `crop` is
        # given in points and scaled by the same ceil, so eps points removes
        # exactly one pixel. Doing it here avoids copying a large bitmap through
        # Pillow — and avoids having to lift Pillow's decompression-bomb
        # ceiling, which would otherwise reject legitimately large pages.
        over_w = math.ceil(width_pts * scale) - target_w
        over_h = math.ceil(height_pts * scale) - target_h
        # crop is (left, bottom, right, top); trimming right/bottom keeps the
        # page anchored at the top-left, which is the page_text/ origin.
        crop = (
            0.0,
            over_h / scale if over_h > 0 else 0.0,
            over_w / scale if over_w > 0 else 0.0,
            0.0,
        )
        bitmap = page.render(scale=scale, crop=crop)
        bitmap.to_pil().save(output_dir / (PAGE_FILENAME_TEMPLATE % page_num), format="PNG")


class Pypdfium2Slicer(PdfSlicer):
    """Extract pages in-process with PDFium.

    ``import_pages`` copies page objects into a fresh document rather than
    re-encoding them, so the text layer and any embedded images survive
    untouched. That is higher fidelity than ghostscript's ``pdfwrite``, which
    re-encodes — but it also means an image-heavy slice keeps the originals and
    can be roughly twice the size of the ghostscript equivalent. The slice is
    sent to the model as a PDF attachment, so that is a payload-size tradeoff,
    not just a disk one.
    """

    def __init__(self, config: PdfConfig) -> None:
        self._pdfium = _import_pdfium()

    def slice(self, pdf_bytes: bytes, page_numbers: Sequence[int]) -> bytes:
        try:
            source = self._pdfium.PdfDocument(pdf_bytes)
        except Exception as exc:
            raise PdfSliceFailed(f"pypdfium2 could not open the PDF to slice: {exc}") from exc
        out = io.BytesIO()
        try:
            new = self._pdfium.PdfDocument.new()
            try:
                # import_pages takes 0-based indices; page_numbers are 1-based
                # and already bounds-checked and ordered by slice_pages().
                new.import_pages(source, [n - 1 for n in page_numbers])
                new.save(out)
            finally:
                new.close()
        except PdfSliceFailed:
            raise
        except Exception as exc:
            raise PdfSliceFailed(
                f"pypdfium2 failed slicing pages {list(page_numbers)}: {exc}"
            ) from exc
        finally:
            source.close()
        return out.getvalue()
