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

"""Words for the values filled into PDF form fields.

A filled PDF form (AcroForm) keeps each field's value in the field dictionary
(``/V``) and draws it with the widget annotation's appearance stream
(``/AP``). The appearance is not part of the page content, so the pdfminer
page walk in :func:`dgml_core.text_extraction.extract_text_digital` never
sees it. The page image ghostscript renders and the model do see it.

:func:`form_field_words` gives, per page, the words a field draws. The rules:

- Only text fields (``/FT /Tx``) and combo boxes (``/FT /Ch`` with the Combo
  field flag) with a non-empty ``/V``. A combo box draws its one chosen
  option, like a text field. Check boxes, radio buttons and list boxes (which
  draw several options) are not read, and neither are password fields (their
  appearance hides the value).
- Only widgets the page image shows: a field name (``/T``) on the widget or
  on a parent field (ghostscript draws no widget without one), a ``/Rect``
  with area, an ``/AP /N`` appearance, the Print annotation flag set and the
  Hidden flag not set. Ghostscript renders page images as a printer does, and
  a printer draws an annotation only when its Print flag is set.
- The words come from the appearance stream, interpreted by pdfminer with the
  same page transform as the page content, so each word has its true glyph
  box (alignment, comb fields, multiple lines and rotated pages included).
  A word whose centre is outside the widget rectangle is dropped: viewers
  clip it.
- When the appearance draws glyphs that pdfminer cannot map to text
  (``(cid:N)``), the words of ``/V`` are spread over the widget rectangle
  instead; for a combo box whose ``/Opt`` pairs an export value with a display
  text, the display text is what is spread. When the appearance draws no
  glyphs, nothing is added: nothing of the value shows on the page.
- The words carry no style facts (no ``s``). Grounding takes a page's body
  font size from the sizes of its words and sizes each element's
  ``dg:style`` against it; the font of a filled-in value is the form's, so it
  must not move that baseline. A filled and a blank copy of a form give the
  same styles.

:func:`merge_form_field_words` then skips a field whose text the page already
has inside the widget rectangle (a form that was also flattened into the page
content), so no value is added twice, and puts each field's words in reading
order: right after the nearest label on the same line, else after the nearest
line of page content above it in the same column, else at the end of the page.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Annotation flags (PDF 32000-1, table 165).
_FLAG_HIDDEN = 1 << 1
_FLAG_PRINT = 1 << 2
# Text-field flags (table 228): the field can hold more than one line; the
# field is a password field, whose appearance does not show the value.
_FF_MULTILINE = 1 << 12
_FF_PASSWORD = 1 << 13
# Choice-field flag (table 230): a combo box (one option shown); without it
# the field is a list box, whose appearance draws several options.
_FF_COMBO = 1 << 17
# A word whose centre is this many points outside the widget rectangle is
# still kept (glyph boxes include side bearings and the font descent).
_CLIP_TOLERANCE_PTS = 2.0
# Bounds the walk up the field /Parent chain (a malformed cycle).
_MAX_PARENT_DEPTH = 32
# Slack, as a share of the field word's height, in the reading-order tests of
# :meth:`_PageWords.predecessor`: a label may end slightly past the value's
# left edge, and the line above may reach slightly below the value's top
# (glyph boxes include the font's ascent and descent).
_ORDER_TOLERANCE = 0.25
# Widest gap between two words of one run of text on a line, as a share of the
# word height: a word space is about a quarter of it, a column gap is wider.
_RUN_GAP = 0.5

PtsBox = tuple[float, float, float, float]
Matrix = tuple[float, float, float, float, float, float]
_IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


@dataclass
class FieldWords:
    """The words one form field draws on one page.

    ``rect`` and every word box are in the device space the page content
    words use (PDF points after the page transform, origin bottom-left).
    ``prorated`` is ``True`` when the boxes are spread over ``rect`` from
    ``/V`` because the appearance stream had no readable text."""

    rect: PtsBox
    words: list[tuple[str, PtsBox]]
    prorated: bool = False


def form_field_words(pdf_path: Path) -> dict[int, list[FieldWords]]:
    """Return ``{page_number: [FieldWords, ...]}`` for the filled, visible text
    fields and combo boxes of ``pdf_path`` (1-based pages, same order as
    pdfminer's page walk).

    A PDF without an ``/AcroForm`` returns ``{}`` without reading any page.
    Form values are an addition to the page text, never a reason for text
    extraction to fail: a widget that cannot be read is skipped (logged at
    DEBUG), a parse error ends the walk with what was read so far, and a
    document where either happened logs one WARNING."""
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdftypes import resolve1

    out: dict[int, list[FieldWords]] = {}
    skipped = 0
    try:
        with open(pdf_path, "rb") as fh:
            doc = PDFDocument(PDFParser(fh))
            acroform = resolve1(doc.catalog.get("AcroForm"))
            if not isinstance(acroform, dict) or not resolve1(acroform.get("Fields")):
                return out
            default_resources = resolve1(acroform.get("DR"))
            renderer = _AppearanceRenderer()
            for page_num, page in enumerate(PDFPage.create_pages(doc), start=1):
                annots = resolve1(page.annots)
                if not isinstance(annots, list) or not annots:
                    continue
                ctm = _page_ctm(page)
                for ref in annots:
                    try:
                        fw = _widget_words(resolve1(ref), page, ctm, renderer, default_resources)
                    except Exception as exc:
                        # One broken widget costs that widget only; a fresh
                        # renderer drops any state the failed one left behind.
                        skipped += 1
                        logger.debug(
                            "form field skipped on page %d of %s: %s: %s",
                            page_num,
                            pdf_path.name,
                            type(exc).__name__,
                            exc,
                        )
                        renderer = _AppearanceRenderer()
                        continue
                    if fw is not None:
                        out.setdefault(page_num, []).append(fw)
    except Exception as exc:
        logger.warning(
            "form field values of %s read up to a parse error (%d widgets skipped before it): "
            "%s: %s",
            pdf_path.name,
            skipped,
            type(exc).__name__,
            exc,
        )
        return out
    if skipped:
        logger.warning(
            "%d form field widgets of %s could not be read and were skipped",
            skipped,
            pdf_path.name,
        )
    return out


def merge_form_field_words(
    words: list[dict[str, Any]],
    fields: list[FieldWords],
    *,
    page_h_pts: float,
    scale_x: float,
    scale_y: float,
) -> int:
    """Insert the words of ``fields`` into one page's ``words`` (the page_text
    JSON shape, page content words in pdfminer's reading order) and return how
    many were added. ``page_h_pts``, ``scale_x`` and ``scale_y`` place the
    field boxes in the page image's pixels, as the content words were placed.

    A field is skipped when the words already on the page whose centre lies
    in the field rectangle spell the field's text (letters and digits,
    case-insensitive).

    The words of a field stay together, right after the page content word
    that precedes the field in reading order (:meth:`_PageWords.predecessor`),
    or at the end of the page when no content word does. Fields that follow
    the same word go top to bottom, left to right on a line. The page content
    words keep their order; the list is changed in place."""
    from .text_extraction import _pts_box_to_pixel_lt_rb

    index = _PageWords(words)
    added = 0
    # Content-word index -> blocks of field words to insert after it; -1 is
    # the end of the page.
    after: dict[int, list[list[dict[str, Any]]]] = {}
    for field in fields:
        rect_px = _pts_box_to_pixel_lt_rb(
            *field.rect, page_h_pts=page_h_pts, scale_x=scale_x, scale_y=scale_y
        )
        if rect_px is None:
            continue
        field_text = _alnum("".join(text for text, _ in field.words))
        if not field_text or field_text in index.text_in(rect_px):
            continue
        block: list[dict[str, Any]] = []
        for text, box_pts in field.words:
            box = _pts_box_to_pixel_lt_rb(
                *box_pts, page_h_pts=page_h_pts, scale_x=scale_x, scale_y=scale_y
            )
            if box is not None:
                block.append({"t": text, "l": list(box)})
        if not block:
            continue
        index.add(block)
        added += len(block)
        after.setdefault(index.predecessor(block[0]["l"]), []).append(block)
    if not added:
        return 0
    merged: list[dict[str, Any]] = []
    for i, w in enumerate(words):
        merged.append(w)
        for block in _reading_order(after.get(i, [])):
            merged.extend(block)
    for block in _reading_order(after.get(-1, [])):
        merged.extend(block)
    words[:] = merged
    return added


class _PageWords:
    """One page's words, sorted by their vertical position, so that each field
    looks only at the words near it rather than at every word of the page (a
    form page can hold hundreds of fields and thousands of words).

    ``predecessor`` looks at the page content words only; ``text_in`` also
    sees the field words ``add`` gave it (a field drawn twice at one place is
    added once)."""

    def __init__(self, words: list[dict[str, Any]]) -> None:
        self.words = words
        boxes = [w["l"] for w in words]
        self.tallest = max((box[3] - box[1] for box in boxes), default=0)
        self.by_top: list[tuple[float, int]] = sorted((box[1], i) for i, box in enumerate(boxes))
        self.tops = [top for top, _ in self.by_top]
        self.by_bottom: list[tuple[float, int]] = sorted((box[3], i) for i, box in enumerate(boxes))
        self.bottoms = [bottom for bottom, _ in self.by_bottom]
        # Vertical centres, sorted, with (page order, word): the content words
        # first, then the field words in the order they are added.
        by_centre = sorted(((box[1] + box[3]) / 2, i) for i, box in enumerate(boxes))
        self.centres = [centre for centre, _ in by_centre]
        self.at_centre: list[tuple[int, dict[str, Any]]] = [(i, words[i]) for _, i in by_centre]

    def add(self, block: list[dict[str, Any]]) -> None:
        """Make the field words of ``block`` visible to :meth:`text_in`."""
        for w in block:
            centre = (w["l"][1] + w["l"][3]) / 2
            at = bisect_right(self.centres, centre)
            self.centres.insert(at, centre)
            self.at_centre.insert(at, (len(self.at_centre), w))

    def text_in(self, rect: tuple[int, int, int, int]) -> str:
        """The letters and digits, lowercase and in page order, of the words
        whose centre lies in ``rect`` (pixels, ``[left, top, right, bottom]``)."""
        left, top, right, bottom = rect
        lo, hi = bisect_left(self.centres, top), bisect_right(self.centres, bottom)
        inside = sorted(
            (order, str(w.get("t", "")))
            for order, w in self.at_centre[lo:hi]
            if left <= (w["l"][0] + w["l"][2]) / 2 <= right
        )
        return _alnum("".join(text for _, text in inside))

    def predecessor(self, first: list[int]) -> int:
        """Index of the page content word that a field's words follow in
        reading order, or -1 for none. ``first`` is the pixel box ``[left,
        top, right, bottom]`` of the field's first word, and ``h`` is its
        height.

        1. Same line: of the words whose vertical overlap with ``first`` is
           at least half the smaller height and whose right edge is at most
           ``first``'s left + ``h / 4``, the nearest one (the largest right
           edge): the label just before the value.
        2. Else above: of the words that end above ``first`` (bottom at most
           its top + ``h / 4``) and overlap it horizontally (the same column),
           the nearest one (the largest bottom): the label over the box. Then
           the words that continue it on its line (:func:`_continues`), so
           that a label of several words is not split.
        3. Else -1: the field goes at the end of the page.

        A tie goes to the word that comes last in the page's reading order."""
        left, top, right, bottom = first
        tolerance = (bottom - top) * _ORDER_TOLERANCE
        words = self.words
        best = -1
        # A word on the same line overlaps ``first`` vertically: its top is
        # above ``first``'s bottom, and at most the tallest word height above
        # ``first``'s top.
        lo, hi = bisect_left(self.tops, top - self.tallest), bisect_right(self.tops, bottom)
        for _, i in self.by_top[lo:hi]:
            box = words[i]["l"]
            if box[2] <= left + tolerance and _same_line(first, box):
                if best < 0 or (box[2], i) > (words[best]["l"][2], best):
                    best = i
        if best >= 0:
            return best
        # The words that end above ``first``, the lowest first.
        nearest_bottom: float | None = None
        for k in range(bisect_right(self.bottoms, top + tolerance) - 1, -1, -1):
            w_bottom, i = self.by_bottom[k]
            if nearest_bottom is not None and w_bottom < nearest_bottom:
                break
            box = words[i]["l"]
            if min(right, box[2]) > max(left, box[0]) and not _same_line(first, box):
                nearest_bottom = w_bottom
                best = max(best, i)
        while 0 <= best < len(words) - 1 and _continues(words[best]["l"], words[best + 1]["l"]):
            best += 1
        return best


def _continues(a: list[int], b: list[int]) -> bool:
    """``b`` is the next word after ``a`` in a run of text on one line: same
    line, and its left edge is at most half a word height after ``a``'s right
    edge (a word space; a wider gap starts a new column)."""
    height = a[3] - a[1]
    return _same_line(a, b) and a[2] - height * _ORDER_TOLERANCE <= b[0] <= a[2] + height * _RUN_GAP


def _reading_order(blocks: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """``blocks`` (field words that follow the same content word) top to
    bottom, and left to right among blocks whose first words share a line."""
    if len(blocks) < 2:
        return blocks
    lines: list[list[list[dict[str, Any]]]] = []
    for block in sorted(blocks, key=lambda b: (b[0]["l"][1], b[0]["l"][0])):
        if lines and _same_line(lines[-1][0][0]["l"], block[0]["l"]):
            lines[-1].append(block)
        else:
            lines.append([block])
    return [b for line in lines for b in sorted(line, key=lambda b: b[0]["l"][0])]


def _same_line(a: list[int], b: list[int]) -> bool:
    """The boxes overlap vertically by at least half the smaller height."""
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap >= 0.5 * min(a[3] - a[1], b[3] - b[1])


class _AppearanceRenderer:
    """Interprets widget appearance streams with pdfminer, one at a time."""

    def __init__(self) -> None:
        from pdfminer.converter import PDFPageAggregator
        from pdfminer.layout import LAParams
        from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager

        rsrcmgr = PDFResourceManager(caching=True)
        self.device = PDFPageAggregator(rsrcmgr, laparams=LAParams(all_texts=True))
        self.interpreter = PDFPageInterpreter(rsrcmgr, self.device)

    def render(self, page: Any, page_ctm: Matrix, stream: Any, resources: Any, ctm: Matrix) -> Any:
        """Lay out ``stream`` under ``ctm``; returns the pdfminer ``LTPage``."""
        self.device.begin_page(page, page_ctm)
        self.interpreter.render_contents(resources, [stream], ctm=ctm)
        self.device.end_page(page)
        return self.device.get_result()


def _widget_words(
    annot: Any,
    page: Any,
    page_ctm: Matrix,
    renderer: _AppearanceRenderer,
    default_resources: Any,
) -> FieldWords | None:
    from pdfminer.pdftypes import PDFStream, resolve1
    from pdfminer.utils import mult_matrix

    from .text_extraction import _iter_words

    if not isinstance(annot, dict) or _name(annot.get("Subtype")) != "Widget":
        return None
    # Ghostscript draws no widget whose field has no name: no /T on the
    # widget or on any parent field.
    if _inherited(annot, "T") is None:
        return None
    field_type = _name(_inherited(annot, "FT"))
    ff = _inherited(annot, "Ff")
    ff = ff if isinstance(ff, int) else 0
    if field_type == "Ch":
        if not ff & _FF_COMBO:
            return None  # a list box: its appearance draws several options
    elif field_type != "Tx" or ff & _FF_PASSWORD:
        return None  # not a text field, or a password field (the value is hidden)
    value = _text(_inherited(annot, "V"))
    if not value.strip():
        return None
    flags = resolve1(annot.get("F"))
    if not isinstance(flags, int) or not flags & _FLAG_PRINT or flags & _FLAG_HIDDEN:
        return None
    rect = _rect(resolve1(annot.get("Rect")))
    if rect is None:
        return None
    ap = resolve1(annot.get("AP"))
    normal = resolve1(ap.get("N")) if isinstance(ap, dict) else None
    if isinstance(normal, dict):  # appearance sub-dictionary keyed by state
        normal = resolve1(normal.get(_name(annot.get("AS")) or ""))
    if not isinstance(normal, PDFStream):
        return None

    # PDF 32000-1 12.5.5: the appearance BBox, transformed by its Matrix, is
    # mapped onto Rect by a scale and a translation (A); the page transform
    # comes last, as for any page content.
    bbox = _rect(resolve1(normal.get("BBox")))
    if bbox is None:
        return None
    matrix = _IDENTITY
    matrix_raw = resolve1(normal.get("Matrix"))
    if isinstance(matrix_raw, list) and len(matrix_raw) == 6:
        m0, m1, m2, m3, m4, m5 = (float(resolve1(v)) for v in matrix_raw)
        matrix = (m0, m1, m2, m3, m4, m5)
    tx0, ty0, tx1, ty1 = _bound(matrix, bbox)
    if tx1 - tx0 <= 0 or ty1 - ty0 <= 0:
        return None
    sx = (rect[2] - rect[0]) / (tx1 - tx0)
    sy = (rect[3] - rect[1]) / (ty1 - ty0)
    a: Matrix = (sx, 0.0, 0.0, sy, rect[0] - tx0 * sx, rect[1] - ty0 * sy)
    ctm = mult_matrix(mult_matrix(matrix, a), page_ctm)
    resources = resolve1(normal.get("Resources")) or default_resources or {}

    layout = renderer.render(page, page_ctm, normal, resources, ctm)
    device_rect = _bound(page_ctm, rect)
    words = [(t, box) for t, box, _ in _iter_words(layout) if _center_in_pts(box, device_rect)]
    if words and not any("(cid:" in t for t, _ in words):
        return FieldWords(rect=device_rect, words=words)
    if not words:
        return None
    # Glyphs are drawn but their text is unreadable: spread /V over the rect.
    if field_type == "Ch":
        value = _display_text(annot, value)
    multiline = bool(ff & _FF_MULTILINE)
    return FieldWords(
        rect=device_rect, words=_prorate(value, device_rect, multiline=multiline), prorated=True
    )


def _display_text(annot: Any, value: str) -> str:
    """The text a combo box shows for ``value``: ``/V`` holds the export value,
    and an ``/Opt`` entry ``[export, display]`` names the text drawn for it. An
    option given as a single string is both, so ``value`` is returned."""
    from pdfminer.pdftypes import resolve1

    options = _inherited(annot, "Opt")
    if not isinstance(options, list):
        return value
    for option in options:
        pair = resolve1(option)
        if isinstance(pair, list) and len(pair) == 2:
            export, display = (_text(resolve1(v)) for v in pair)
            if export == value and display.strip():
                return display
    return value


def _prorate(value: str, rect: PtsBox, *, multiline: bool) -> list[tuple[str, PtsBox]]:
    """Boxes for the words of ``value`` by character-count proration over
    ``rect``: one line, or one line per line break for a multiline field."""
    from .text_extraction import split_word_into_tokens

    text = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()] if multiline else [" ".join(text.split())]
    x0, y0, x1, y1 = rect
    line_h = (y1 - y0) / max(len(lines), 1)
    out: list[tuple[str, PtsBox]] = []
    for i, line in enumerate(lines):
        tokens = line.split()
        n = max(sum(len(t) for t in tokens) + len(tokens) - 1, 1)
        top = y1 - i * line_h
        pos = 0
        for tok in tokens:
            left = x0 + (x1 - x0) * pos / n
            right = x0 + (x1 - x0) * (pos + len(tok)) / n
            # split_word_into_tokens works on integer boxes; use centi-points.
            box_c = (
                round(left * 100),
                round((top - line_h) * 100),
                round(right * 100),
                round(top * 100),
            )
            for part, (pl, pb, pr, pt) in split_word_into_tokens(tok, box_c):
                out.append((part, (pl / 100, pb / 100, pr / 100, pt / 100)))
            pos += len(tok) + 1
    return out


def _page_ctm(page: Any) -> Matrix:
    """The page transform pdfminer's ``PDFPageInterpreter.process_page`` uses,
    so form words land in the same space as the page content words."""
    x0, y0, x1, y1 = (float(v) for v in page.mediabox)
    rotate = page.rotate
    if rotate == 90:
        return (0.0, -1.0, 1.0, 0.0, -y0, x1)
    if rotate == 180:
        return (-1.0, 0.0, 0.0, -1.0, x1, y1)
    if rotate == 270:
        return (0.0, 1.0, -1.0, 0.0, y1, -x0)
    return (1.0, 0.0, 0.0, 1.0, -x0, -y0)


def _bound(m: Matrix, rect: PtsBox) -> PtsBox:
    """The smallest upright box that holds ``rect`` transformed by ``m``."""
    from pdfminer.utils import apply_matrix_pt

    x0, y0, x1, y1 = rect
    corners = [apply_matrix_pt(m, p) for p in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _inherited(field: Any, key: str) -> Any:
    """``field[key]``, else the nearest ``/Parent``'s (inheritable field keys)."""
    from pdfminer.pdftypes import resolve1

    node = field
    for _ in range(_MAX_PARENT_DEPTH):
        if not isinstance(node, dict):
            return None
        if key in node:
            return resolve1(node[key])
        node = resolve1(node.get("Parent"))
    return None


def _name(obj: Any) -> str | None:
    from pdfminer.pdftypes import resolve1

    name = getattr(resolve1(obj), "name", None)
    return name if isinstance(name, str) else None


def _text(obj: Any) -> str:
    from pdfminer.utils import decode_text

    if isinstance(obj, bytes):
        return decode_text(obj)
    return obj if isinstance(obj, str) else ""


def _rect(obj: Any) -> PtsBox | None:
    from pdfminer.pdftypes import resolve1

    if not isinstance(obj, list) or len(obj) != 4:
        return None
    try:
        a, b, c, d = (float(resolve1(v)) for v in obj)
    except (TypeError, ValueError):
        return None
    x0, x1 = min(a, c), max(a, c)
    y0, y1 = min(b, d), max(b, d)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    return x0, y0, x1, y1


def _center_in_pts(box: PtsBox, rect: PtsBox) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    t = _CLIP_TOLERANCE_PTS
    return rect[0] - t <= cx <= rect[2] + t and rect[1] - t <= cy <= rect[3] + t


def _alnum(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())
