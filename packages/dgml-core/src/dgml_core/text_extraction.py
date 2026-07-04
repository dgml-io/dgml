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

"""Digital text extraction from PDFs.

Walks pdfminer.six's page layout, groups characters into whitespace-separated
words, and emits one compact JSON file per page in the file's ``page_text/``
directory. Coordinates are converted from PDF points (origin bottom-left) to
image pixels matching the corresponding ``page_images/page_N.png`` render.

OCR and hybrid extraction live in :mod:`dgml.ocr` and :mod:`dgml.hybrid` and
share the per-page JSON shape emitted here so downstream consumers don't have
to care which mode produced the words.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Any

from .errors import TextExtractionFailed
from .pages import DEFAULT_DPI
from .style import fontname_is_bold, fontname_is_italic, rgb_to_named

PAGE_TEXT_FILENAME = "page_{page}.json"
PAGE_TEXT_GLOB = "page_*.json"


def split_word_into_tokens(
    text: str, bbox: tuple[int, int, int, int]
) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Split a word at every alnum/non-alnum boundary, estimating each
    token's bbox by uniform character-width proration of the input bbox.

    ``"(75"`` becomes ``[("(", ...), ("75", ...)]``; ``"INC."`` becomes
    ``[("INC", ...), (".", ...)]``. This lets downstream span search
    match the alphanumeric core directly without fuzzing across attached
    punctuation — and keeps the punctuation tokens (with their own
    bboxes) in the stream so spans that legitimately include them still
    work. Used by the OCR providers (which only get word-level boxes)
    and by post-processing of page_text. The digital path
    has true char-level coordinates and splits accurately at extraction
    time, not via this proration."""
    if not text:
        return [(text, bbox)]
    # Find runs of same-class characters (alnum vs non-alnum, excluding
    # whitespace which never appears inside a word here).
    cuts: list[int] = [0]
    for i in range(1, len(text)):
        if text[i].isalnum() != text[i - 1].isalnum():
            cuts.append(i)
    cuts.append(len(text))
    if len(cuts) == 2:
        return [(text, bbox)]
    left, top, right, bottom = bbox
    width = right - left
    n = len(text)
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for k in range(len(cuts) - 1):
        s, e = cuts[k], cuts[k + 1]
        tk_left = left + round(width * s / n)
        tk_right = left + round(width * e / n)
        if tk_right <= tk_left:
            tk_right = tk_left + 1
        out.append((text[s:e], (tk_left, top, tk_right, bottom)))
    return out


class TextMode(StrEnum):
    """How text should be extracted at file-add time.

    ``DIGITAL`` uses pdfminer.six on the PDF. ``OCR`` runs the cloud
    provider configured in ``<workspace>/config.json``. ``HYBRID`` runs
    both and merges them by bounding-box overlap (OCR wins on conflict);
    see :mod:`dgml.hybrid`.
    """

    DIGITAL = "digital"
    OCR = "ocr"
    HYBRID = "hybrid"


@dataclass
class ExtractDigitalResult:
    pages_written: int
    pages_with_words: int
    total_words: int

    def to_summary(self) -> dict[str, Any]:
        return {
            "mode": TextMode.DIGITAL.value,
            "pages_written": self.pages_written,
            "pages_with_words": self.pages_with_words,
            "total_words": self.total_words,
        }


@dataclass
class ExtractionOutcome:
    """Health classification for an :class:`ExtractDigitalResult`.

    ``message`` is ``None`` when the result is fully healthy. Otherwise it
    describes what's wrong, and ``permanent`` says whether re-running
    pdfminer could plausibly improve things (``False`` for transient
    conditions like page-count drift, ``True`` for "this PDF will never
    yield digital text" outcomes that consistency check should not retry).
    """

    message: str | None
    permanent: bool = False


def classify_extraction_outcome(
    result: ExtractDigitalResult, expected_page_count: int | None
) -> ExtractionOutcome:
    """Single source of truth for "what does this extraction result mean."

    Used by both :func:`FileStore._extract_text` at add time and the
    consistency check at re-extract time so the two paths stay aligned.
    """
    if result.pages_written == 0 or result.pages_with_words == 0:
        msg = (
            f"no digital text found on any of {result.pages_written} pages"
            if result.pages_written
            else "no pages were processed by pdfminer"
        )
        return ExtractionOutcome(message=msg, permanent=True)

    empty_pages = result.pages_written - result.pages_with_words
    if empty_pages:
        return ExtractionOutcome(
            message=(f"{empty_pages}/{result.pages_written} pages had no extractable digital text"),
            permanent=False,
        )

    if expected_page_count is not None and result.pages_written != expected_page_count:
        return ExtractionOutcome(
            message=(
                f"extracted text for {result.pages_written} pages, expected {expected_page_count}"
            ),
            permanent=False,
        )

    return ExtractionOutcome(message=None)


def extract_text_digital(
    pdf_path: Path,
    output_dir: Path,
    *,
    file_id: str,
) -> ExtractDigitalResult:
    """Extract digital text from each PDF page and write per-page JSON.

    One ``page_N.json`` is written per page, where ``N`` is 1-based to align
    with ``page_images/page_N.png``. Stale ``page_*.json`` files in
    ``output_dir`` are removed first so retries don't leave orphans.

    Raises :class:`TextExtractionFailed` if pdfminer.six cannot parse the PDF.
    A successful run with zero words on every page returns a result with
    ``pages_with_words == 0`` — the caller decides how to treat that.
    """
    # Lazy import so a missing pdfminer install fails with a clear, actionable
    # error path rather than a module-load-time ImportError.
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LAParams
    except ImportError as exc:
        raise TextExtractionFailed(
            f"pdfminer.six is required for digital text extraction: {exc}"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob(PAGE_TEXT_GLOB):
        existing.unlink()

    pages_written = 0
    pages_with_words = 0
    total_words = 0

    try:
        # ``all_texts=True`` makes pdfminer run line-grouping on text inside
        # Form XObjects (``LTFigure``) too. Without it, brochure-style PDFs that
        # lay body text inside figures yield only loose ``LTChar``s with no
        # enclosing ``LTTextLine`` — which ``_iter_words`` walks past — so we'd
        # capture only page-chrome (e.g. "Page: 1 of 6") and report the page as
        # empty.
        page_layouts = extract_pages(str(pdf_path), laparams=LAParams(all_texts=True))
        for page_num, page_layout in enumerate(page_layouts, start=1):
            # ``LTPage.bbox`` is ``(x0, y0, x1, y1)`` in PDF user-space points.
            x0, y0, x1, y1 = page_layout.bbox
            page_w_pts = float(x1 - x0)
            page_h_pts = float(y1 - y0)
            width_px = max(0, round(page_w_pts * DEFAULT_DPI / 72))
            height_px = max(0, round(page_h_pts * DEFAULT_DPI / 72))

            words: list[dict[str, Any]] = []
            for text, (x0, y0, x1, y1), style in _iter_words(page_layout):
                box = _pts_box_to_pixel_lt_rb(
                    x0, y0, x1, y1, page_h_pts=page_h_pts, dpi=DEFAULT_DPI
                )
                if box is None:
                    continue
                word: dict[str, Any] = {"t": text, "l": list(box)}
                # ``s`` carries observed per-word style facts (font weight,
                # slant, size in points, dominant CSS-named color). ``sz`` is
                # recorded for every sized word — essentially every digital
                # word — so ``s`` lands on nearly all of them; ``b``/``i``/``c``
                # appear only when non-default. Omitted only when no style fact
                # is observed at all (e.g. a run with no sized glyphs), so OCR
                # words (which carry no style) produce no ``s`` key.
                if style:
                    word["s"] = style
                words.append(word)

            page_json: dict[str, Any] = {
                "file_id": file_id,
                "page": page_num,
                "width": width_px,
                "height": height_px,
                "words": words,
            }
            out_path = output_dir / PAGE_TEXT_FILENAME.format(page=page_num)
            # Compact one-line JSON — `page_text/` is per-page so files stay
            # small; pretty-printing would bloat workspaces with thousands of
            # pages without helping consumers (parse with `jq`, not eyeballs).
            out_path.write_text(
                json.dumps(page_json, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            pages_written += 1
            if words:
                pages_with_words += 1
                total_words += len(words)
    except TextExtractionFailed:
        raise
    except Exception as exc:
        raise TextExtractionFailed(
            f"pdfminer failed to parse PDF: {type(exc).__name__}: {exc}"
        ) from exc

    return ExtractDigitalResult(
        pages_written=pages_written,
        pages_with_words=pages_with_words,
        total_words=total_words,
    )


def _iter_words(
    page_layout: Any,
) -> Iterator[tuple[str, tuple[float, float, float, float], dict[str, Any] | None]]:
    """Yield ``(text, (x0, y0, x1, y1), style)`` for each token on a page.

    A token is a maximal run of same-class ``LTChar``s within the same
    ``LTTextLine`` — alphanumeric runs and non-alphanumeric runs split at
    every transition, so ``"(75"`` yields ``"("`` and ``"75"`` separately.
    Whitespace and pdfminer-synthesized ``LTAnno`` annotations end the
    current token. Coords are PDF points (origin bottom-left); per-token
    bboxes are exact since each ``LTChar`` carries its own glyph extent.
    ``style`` is the per-word style dict from :func:`_word_style` (or
    ``None``)."""
    from pdfminer.layout import LTChar, LTTextLine

    def _emit(
        chars: list[Any],
    ) -> Iterator[tuple[str, tuple[float, float, float, float], dict[str, Any] | None]]:
        if not chars:
            return
        text = "".join(c.get_text() for c in chars)
        x0 = min(c.x0 for c in chars)
        y0 = min(c.y0 for c in chars)
        x1 = max(c.x1 for c in chars)
        y1 = max(c.y1 for c in chars)
        yield text, (x0, y0, x1, y1), _word_style(chars)

    for line in _walk(page_layout, LTTextLine):
        current: list[Any] = []
        current_class: bool | None = None  # True=alnum, False=non-alnum
        for ch in line:
            if not isinstance(ch, LTChar):
                yield from _emit(current)
                current = []
                current_class = None
                continue
            text = ch.get_text()
            if not text or text.isspace():
                yield from _emit(current)
                current = []
                current_class = None
                continue
            cls = text.isalnum()
            if current_class is None or cls == current_class:
                current.append(ch)
                current_class = cls
            else:
                yield from _emit(current)
                current = [ch]
                current_class = cls
        yield from _emit(current)


def _walk(node: Any, target_type: type) -> Iterator[Any]:
    """Yield descendants of ``node`` that are instances of ``target_type``."""
    if isinstance(node, target_type):
        yield node
        return
    children = getattr(node, "_objs", None)
    if children is None:
        try:
            children = list(node)
        except TypeError:
            return
    for child in children:
        yield from _walk(child, target_type)


def _pts_box_to_pixel_lt_rb(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    page_h_pts: float,
    dpi: int,
) -> tuple[int, int, int, int] | None:
    """Convert a PDF-points bbox (origin bottom-left) to image-pixel
    ``(left, top, right, bottom)`` ints (origin top-left). Returns ``None``
    if the resulting box is degenerate (zero width or height).
    """
    scale = dpi / 72.0
    left = round(x0 * scale)
    right = round(x1 * scale)
    # PDF y is bottom-up; image y is top-down. y1 (PDF top) -> image top.
    top = round((page_h_pts - y1) * scale)
    bottom = round((page_h_pts - y0) * scale)
    left = max(0, left)
    top = max(0, top)
    right = max(left, right)
    bottom = max(top, bottom)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _word_style(chars: list[Any]) -> dict[str, Any] | None:
    """Summarize a word's ``LTChar`` run into observed style facts.

    Bold/italic are per-word majorities over the run's chars (the final
    chunk-level majority is taken at grounding, weighting each word by its
    character count). ``sz`` is the run's median glyph size in PDF points
    (the page baseline and ``em`` bucketing are resolved later). ``c`` is the
    run's dominant fill color, already snapped to a CSS named color (omitted
    for near-black default text). ``b``/``i``/``c`` appear only when that
    non-default formatting is observed, but ``sz`` is recorded for *every*
    word with sized glyphs — which is essentially every pdfminer ``LTChar``
    run — so in practice the dict is non-empty (and ``s`` present) on nearly
    all digital words. That is deliberate: the page-baseline vote in grounding
    (:func:`dgml_core.xml_grounding._page_baseline`) is the char-weighted mode
    of word sizes, so it needs body-size words carrying ``sz`` in the tally.
    Returns ``None`` only for a run with no chars (or none carrying a size)."""
    if not chars:
        return None
    n = len(chars)
    bold = sum(1 for c in chars if fontname_is_bold(getattr(c, "fontname", None)))
    italic = sum(1 for c in chars if fontname_is_italic(getattr(c, "fontname", None)))
    sizes = [float(sz) for c in chars if (sz := getattr(c, "size", 0.0))]
    colors = [rgb for c in chars if (rgb := _char_rgb(c)) is not None]

    style: dict[str, Any] = {}
    if bold * 2 > n:
        style["b"] = 1
    if italic * 2 > n:
        style["i"] = 1
    if sizes:
        style["sz"] = round(median(sizes), 1)
    if colors:
        dominant = Counter(colors).most_common(1)[0][0]
        named = rgb_to_named(dominant)
        if named:
            style["c"] = named
    return style or None


def _char_rgb(ch: Any) -> tuple[int, int, int] | None:
    """The non-stroking (fill) color of a glyph as 0-255 RGB, or ``None``."""
    gs = getattr(ch, "graphicstate", None)
    color = getattr(gs, "ncolor", None) if gs is not None else None
    return _normalize_pdf_color(color)


def _normalize_pdf_color(color: Any) -> tuple[int, int, int] | None:
    """Normalize a pdfminer color (gray float, RGB tuple, or CMYK tuple, with
    components in 0-1) to a 0-255 RGB triple. Returns ``None`` for unknown
    shapes."""
    if color is None:
        return None
    if isinstance(color, (int, float)):
        v = _clamp255(float(color))
        return (v, v, v)
    if isinstance(color, (tuple, list)):
        if len(color) == 1:
            v = _clamp255(float(color[0]))
            return (v, v, v)
        if len(color) == 3:
            r, g, b = (float(x) for x in color)
            return (_clamp255(r), _clamp255(g), _clamp255(b))
        if len(color) == 4:
            c, m, y, k = (float(x) for x in color)
            r = _clamp255(1.0 - min(1.0, c + k))
            g = _clamp255(1.0 - min(1.0, m + k))
            b = _clamp255(1.0 - min(1.0, y + k))
            return (r, g, b)
    return None


def _clamp255(v: float) -> int:
    """Scale a PDF color component (0-1) to 0-255, clamping out-of-range
    inputs. Values already above 1 are treated as 0-255 and just clamped."""
    if v <= 1.0:
        v = v * 255.0
    return max(0, min(255, round(v)))
