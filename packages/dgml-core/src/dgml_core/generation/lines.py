"""Pass A over the text layer: structure over numbered lines (``Plan B``).

For a document with a text layer, the model does not copy text. The window's
pages are cut into visual lines, numbered ``L1..Ln`` across the window, and
shown to the model next to the page images. The model answers with the same
typed block lines as the text format (``H2``, ``P``, ``I``, ``R``, ``F``,
``O``, ``C``) but every text field is a line reference -- ``L12``, ``L13-L19``
for a span, ``L20+L21`` for a table row wrapped over two lines -- and an ``X``
line marks page decorations (page numbers, running headers). The block text is
copied from the text layer, table cells come from the words' coordinates, and
a line the model did not account for is filled in as a plain block, so every
line of the page ends up in exactly one block.

What this buys over copying: the reply is a tenth of the size (ids instead of
words), there is no text to hallucinate or shorten, grounding is a lookup, and
the completeness check is ``ids assigned / ids shown`` rather than a word
recall. Pages without a usable text layer (scans) fall back to the text format
window by window; see :func:`dgml_core.generation.transcribe.transcribe_document`.
"""

from __future__ import annotations

import itertools
import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from dgml_core.generation.completeness import (
    _CELL_GAP_HEIGHTS,
    _core,
    _group_lines,
    _is_figure,
    _merge_split_words,
    _OTok,
)
from dgml_core.textmatch import PageDims, Word

# A line reference: "L12", a span "L12-L19" (also an en dash, also "L12-19"),
# or wrapped-row parts "L20+L21". Anything else in a text field is literal
# text the model chose to write (a chart description, an enumerator).
_REF_PART_RE = re.compile(r"^\s*L(\d+)\s*(?:[-\u2013]\s*L?(\d+))?\s*$", re.I)
# A page with fewer extractable words than this is treated as a scan.
MIN_PAGE_WORDS = 20
# What the model may mark as a page decoration with X: a short line within
# this many lines of a page's top or bottom, at most this many per page.
# Anything else marked X is content the model wanted to skip, and is filled.
_DECORATION_MAX_WORDS = 10
_DECORATION_EDGE_LINES = 3
_DECORATION_MAX_PER_PAGE = 3


@dataclass
class TextLine:
    """One visual line of a page, in the window's numbering."""

    id: int  # 1-based, unique within the window
    page: int  # 1-based page number
    toks: list[_OTok]  # merged tokens, left to right
    edge: bool = False  # within _DECORATION_EDGE_LINES of the page's top/bottom
    repeated: bool = False  # same words on another page of the window

    def may_be_decoration(self) -> bool:
        """A page number (a short line that is mostly a number) or a running
        header/footer (a short edge line repeated on another page)."""
        if not self.edge or self.n_words > _DECORATION_MAX_WORDS:
            return False
        if self.repeated:
            return True
        cores = [t.core for t in self.toks if t.core]
        numeric = [c for c in cores if _is_figure(c) or re.fullmatch(r"[ivxlc]+", c)]
        return len(cores) <= 3 and bool(numeric) and len(cores) - len(numeric) <= 1

    @property
    def height(self) -> float:
        return statistics.median(max(1, t.bottom - t.top) for t in self.toks)

    @property
    def n_words(self) -> int:
        return len(self.toks)

    @property
    def bold(self) -> bool:
        return all(t.bold for t in self.toks)

    def cells(self) -> list[str]:
        """Cell texts: runs of tokens separated by a gap wider than a cell gap."""
        h = self.height
        groups: list[list[_OTok]] = [[self.toks[0]]]
        for prev, t in itertools.pairwise(self.toks):
            if t.left - prev.right > _CELL_GAP_HEIGHTS * h:
                groups.append([t])
            else:
                groups[-1].append(t)
        return [" ".join(t.text for t in g) for g in groups]

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.toks)

    @property
    def has_figure(self) -> bool:
        return any(_is_figure(t.core) for t in self.toks)


# Two-column prose: a gap at least this many line heights wide, with at least
# this many words on each side, is a column break; a gutter is an x that such
# gaps share on enough of the page's long lines.
_GUTTER_GAP_HEIGHTS = 2.0
_GUTTER_MIN_WORDS_EACH_SIDE = 4
_GUTTER_MIN_SHARE = 0.3


def _gutter_x(lines: list[list[_OTok]]) -> float | None:
    """The x of a two-column gutter on this page, or None.

    A visual line that reads across both columns has one wide gap with real
    text on both sides. When enough of the page's long lines share such a gap
    at the same x, that x is a gutter and those lines are two lines each. A
    table does not qualify: its gaps separate short cells, not runs of
    prose."""
    long_lines = [ln for ln in lines if len(ln) >= 2 * _GUTTER_MIN_WORDS_EACH_SIDE]
    if len(long_lines) < 4:
        return None
    spans: list[tuple[float, float]] = []
    for ln in long_lines:
        h = statistics.median(max(1, t.bottom - t.top) for t in ln)
        wide = [
            (i, prev.right, t.left)
            for i, (prev, t) in enumerate(itertools.pairwise(ln), start=1)
            if t.left - prev.right > _GUTTER_GAP_HEIGHTS * h
        ]
        # A column break is ONE wide gap with prose on both sides; a table
        # row has several (one per figure column) and figures on the right.
        if len(wide) != 1:
            continue
        i, a, b = wide[0]
        right_toks = ln[i:]
        figures = sum(1 for t in right_toks if _is_figure(t.core))
        if (
            i >= _GUTTER_MIN_WORDS_EACH_SIDE
            and len(right_toks) >= _GUTTER_MIN_WORDS_EACH_SIDE
            and figures * 2 < len(right_toks)
        ):
            spans.append((a, b))
    if len(spans) < max(4, _GUTTER_MIN_SHARE * len(long_lines)):
        return None
    # The x crossed by the most spans: try each span's midpoint.
    best_x: float | None = None
    best_n = 0
    for lo, hi in spans:
        x = (lo + hi) / 2
        n = sum(1 for a2, b2 in spans if a2 <= x <= b2)
        if n > best_n:
            best_x, best_n = x, n
    if best_n < max(4, _GUTTER_MIN_SHARE * len(long_lines)):
        return None
    return best_x


def _split_columns(lines: list[list[_OTok]]) -> list[list[_OTok]]:
    """Cut two-column lines at the page's gutter and read the left column
    before the right. Lines that do not cross the gutter (full-width
    headings, tables) stay whole and keep their place."""
    gx = _gutter_x(lines)
    if gx is None:
        return lines
    out: list[list[_OTok]] = []
    lefts: list[list[_OTok]] = []
    rights: list[list[_OTok]] = []

    def flush() -> None:
        out.extend(lefts)
        out.extend(rights)
        lefts.clear()
        rights.clear()

    def table_like(ln: list[_OTok]) -> bool:
        """Several wide gaps, or a figure-heavy side: a table row, kept whole."""
        h = statistics.median(max(1, t.bottom - t.top) for t in ln)
        gaps = sum(
            1 for a, b in itertools.pairwise(ln) if b.left - a.right > _GUTTER_GAP_HEIGHTS * h
        )
        if gaps >= 3:
            return True
        left = [t for t in ln if t.right <= gx]
        right = [t for t in ln if t.left >= gx]
        for side in (left, right):
            if side and sum(1 for t in side if _is_figure(t.core)) * 2 >= len(side):
                return True
        return False

    for ln in lines:
        left = [t for t in ln if t.right <= gx]
        right = [t for t in ln if t.left >= gx]
        if left and right and len(left) + len(right) == len(ln) and not table_like(ln):
            lefts.append(left)
            rights.append(right)
        elif not right and lefts:
            # A short line entirely in the left column continues it.
            lefts.append(ln)
        elif not left and lefts:
            rights.append(ln)
        else:
            flush()
            out.append(ln)
    flush()
    return out


def page_lines(words: list[Word], page: int, first_id: int) -> list[TextLine]:
    """Visual lines of one page, numbered from *first_id*; two-column pages
    are read column by column."""
    toks: list[_OTok] = []
    for group in _merge_split_words(words):
        text = "".join(w.text for w in group)
        if text.strip():
            toks.append(_OTok(page, group, _core(text)))
    out: list[TextLine] = []
    grouped = _split_columns(_group_lines(toks))
    n = len(grouped)
    for k, ln in enumerate(grouped):
        edge = k < _DECORATION_EDGE_LINES or k >= n - _DECORATION_EDGE_LINES
        out.append(TextLine(id=first_id + k, page=page, toks=ln, edge=edge))
    return out


def document_vocabulary(pages: dict[int, tuple[list[Word], PageDims]]) -> set[str]:
    """Core forms of every word in the document's text layer."""
    return {c for ws, _ in pages.values() for w in ws if (c := _core(w.text))}


def window_lines(
    pages: dict[int, tuple[list[Word], PageDims]], page_numbers: list[int]
) -> list[TextLine]:
    """The numbered lines of a window's pages (1-based page numbers), in order."""
    out: list[TextLine] = []
    for p in page_numbers:
        words = pages.get(p, ([], PageDims(1, 1)))[0]
        out.extend(page_lines(words, p, len(out) + 1))
    seen: dict[str, set[int]] = {}
    for ln in out:
        if ln.edge:
            seen.setdefault(" ".join(t.core for t in ln.toks if t.core), set()).add(ln.page)
    for ln in out:
        if ln.edge and len(seen.get(" ".join(t.core for t in ln.toks if t.core), ())) >= 2:
            ln.repeated = True
    return out


def render_listing(lines: list[TextLine]) -> str:
    """The numbered listing shown to the model: ``L7<TAB>text`` per line, cells
    separated by ``|``, a marker line at each page start."""
    parts: list[str] = []
    page = None
    for ln in lines:
        if ln.page != page:
            page = ln.page
            parts.append(f"--- page {page} ---")
        cells = ln.cells()
        parts.append(f"L{ln.id}\t" + (" | ".join(cells) if len(cells) > 1 else cells[0]))
    return "\n".join(parts)


# ---- Reply -> payload ---------------------------------------------------------


def parse_refs(text: str) -> list[int] | None:
    """Line ids named by a field, or None when the field is literal text."""
    ids: list[int] = []
    for part in text.split("+"):
        m = _REF_PART_RE.match(part)
        if not m:
            return None
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a:
            a, b = b, a
        ids.extend(range(a, b + 1))
    return ids or None


def _join_lines(texts: list[str], vocab: set[str] | None = None) -> str:
    """Join line texts into running text, mending end-of-line hyphenation.

    "pro-" + "vider" is mended to "provider" only when that word occurs on
    its own somewhere in the document (*vocab*, core forms); "wholly-" +
    "owned" keeps its hyphen, since "whollyowned" occurs nowhere and the
    page's own tokens are "wholly-" and "owned". Without a vocabulary the
    hyphen is always mended."""
    out = ""
    for t in texts:
        if not out:
            out = t
        elif out.endswith("-") and t[:1].islower():
            head, tail = out[:-1].rsplit(" ", 1)[-1], t.split(" ", 1)[0]
            if vocab is None or _core(head + tail) in vocab:
                out = out[:-1] + t
            else:
                out = out + t
        else:
            out = f"{out} {t}"
    return out


def _strip_lim(text: str, lim: str) -> str:
    """Drop a printed enumerator the model put in ``lim`` from the line text."""
    lim = lim.strip()
    if lim and text.startswith(lim):
        rest = text[len(lim) :]
        if not rest or rest[0].isspace():
            return rest.lstrip()
    return text


def _row_cells(lines: list[TextLine]) -> list[str]:
    """Cells of a row given its line(s); a wrapped row merges cell-wise."""
    cells = lines[0].cells()
    for ln in lines[1:]:
        more = ln.cells()
        if len(more) == 1:
            cells[0] = f"{cells[0]} {more[0]}".strip()
        elif len(more) == len(cells):
            cells = [f"{a} {b}".strip() for a, b in zip(cells, more, strict=True)]
        else:
            cells = cells + more
    return cells


def _field_parts(lines: list[TextLine]) -> tuple[str, str]:
    """Label and value of a form line: split at the first cell gap, else at
    the first colon, else label only."""
    cells = _row_cells(lines)
    if len(cells) >= 2:
        return cells[0], " ".join(cells[1:])
    text = cells[0]
    if ":" in text:
        label, value = text.split(":", 1)
        return label.strip() + ":", value.strip()
    return text, ""


@dataclass
class LinesResult:
    payload: dict[str, Any]
    assigned: set[int] = field(default_factory=set)
    decorations: set[int] = field(default_factory=set)
    unknown_refs: int = 0
    malformed: int = 0
    literal_fields: int = 0
    restarts: int = 0
    demoted_continuations: int = 0

    def recall(self, lines: list[TextLine]) -> float:
        """Share of the window's words the model accounted for (a decoration
        counts as accounted for)."""
        total = sum(ln.n_words for ln in lines)
        if not total:
            return 1.0
        done = self.assigned | self.decorations
        return sum(ln.n_words for ln in lines if ln.id in done) / total

    def unassigned(self, lines: list[TextLine]) -> list[TextLine]:
        done = self.assigned | self.decorations
        return [ln for ln in lines if ln.id not in done]


_BLOCK_LINE_RE = re.compile(r"^(?:H\d*|[PIRFOXC])(?:\t|<TAB>)")


def build_payload(reply: str, lines: list[TextLine], vocab: set[str] | None = None) -> LinesResult:
    """Decode a structure-over-lines reply into the JSON-shaped window payload
    (``{"continues": str, "blocks": [...]}``) that ``parse_block`` consumes.

    A model sometimes abandons a draft mid-reply ("Wait, that was wrong.
    Restarting:") and answers again. The reply is cut into segments at such
    prose lines and the segment accounting for the most words is kept, so a
    discarded draft does not win by coming first. Each block dict also
    carries ``"lines"``: the ids it was built from, so the caller can place
    fallback blocks for the lines nobody claimed.
    """
    segments: list[list[str]] = [[]]
    prose = 0
    for raw in reply.splitlines():
        if not raw.strip():
            continue
        if _BLOCK_LINE_RE.match(raw):
            segments[-1].append(raw)
        else:
            prose += 1
            if segments[-1]:
                segments.append([])
    segments = [seg for seg in segments if seg] or [[]]
    best: LinesResult | None = None
    for seg in segments:
        res = _build_segment(seg, lines, vocab)
        res.malformed += prose
        if best is None or res.recall(lines) > best.recall(lines):
            best = res
    assert best is not None
    best.restarts = max(0, len(segments) - 1)
    return best


def _build_segment(
    block_lines: list[str], lines: list[TextLine], vocab: set[str] | None = None
) -> LinesResult:
    by_id = {ln.id: ln for ln in lines}
    res = LinesResult(payload={"continues": "", "blocks": []})
    blocks: list[dict[str, Any]] = res.payload["blocks"]

    core_of = {ln.id: " ".join(t.core for t in ln.toks if t.core) for ln in lines}

    def claim_by_text(literal: str) -> list[TextLine]:
        """Lines whose text the model typed out instead of referencing: an
        unassigned line, or a run of up to four, whose words equal the
        literal's. Claimed so the fallback fill does not add them again."""
        want = " ".join(c for c in (_core(t) for t in literal.split()) if c)
        if len(want) < 4:
            return []
        for ln in lines:
            if ln.id in res.assigned or not core_of[ln.id] or not want.startswith(core_of[ln.id]):
                continue
            run, acc = [ln], core_of[ln.id]
            while acc != want and len(run) < 4:
                nxt = by_id.get(run[-1].id + 1)
                if nxt is None or nxt.id in res.assigned or not core_of[nxt.id]:
                    break
                acc = f"{acc} {core_of[nxt.id]}"
                if not want.startswith(acc):
                    break
                run.append(nxt)
            if acc == want:
                for x in run:
                    res.assigned.add(x.id)
                return run
        return []

    def take(ref_text: str) -> tuple[list[TextLine], str | None]:
        """Lines named by a field, or (None, literal) when it is text."""
        ids = parse_refs(ref_text)
        if ids is None:
            res.literal_fields += 1
            claimed = claim_by_text(ref_text)
            if claimed:
                return claimed, None
            return [], ref_text
        got: list[TextLine] = []
        for i in ids:
            ln = by_id.get(i)
            if ln is None or i in res.assigned:
                res.unknown_refs += 1
                continue
            res.assigned.add(i)
            got.append(ln)
        return got, None

    def text_of(ref_text: str, lim: str = "") -> tuple[str, list[int]]:
        got, literal = take(ref_text)
        if literal is not None:
            return literal, []
        return _strip_lim(_join_lines([ln.text for ln in got], vocab), lim), [ln.id for ln in got]

    def lim_of(field_text: str) -> str:
        """A lim is printed text; a model that wrote a line ref there meant
        that line's text."""
        if parse_refs(field_text):
            got, _ = take(field_text)
            return " ".join(ln.text for ln in got)
        return field_text

    for raw in block_lines:
        line = raw.rstrip("\r")
        if "\t" not in line and "<TAB>" in line:
            line = line.replace("<TAB>", "\t")
        fields = line.split("\t")
        sigil, rest = fields[0].strip(), [f.strip() for f in fields[1:]]
        tail = "\t".join(rest)
        m = re.match(r"^H(\d*)$", sigil)
        if m:
            lim = lim_of(rest[0]) if rest else ""
            text, ids = text_of("\t".join(rest[1:]), lim)
            blocks.append(
                {
                    "structure": "heading",
                    "level": int(m.group(1)) if m.group(1) else 1,
                    "lim": lim,
                    "text": text,
                    "lines": ids,
                }
            )
        elif sigil == "P":
            text, ids = text_of(tail)
            blocks.append({"structure": "p", "text": text, "lines": ids})
        elif sigil == "I":
            lim = lim_of(rest[0]) if rest else ""
            text, ids = text_of("\t".join(rest[1:]), lim)
            blocks.append({"structure": "item", "lim": lim, "text": text, "lines": ids})
        elif sigil == "R":
            got, _literal = take(tail) if len(rest) == 1 else ([], None)
            if got and "+" in tail and len(got) <= 3:
                # A row wrapped over lines: "R L20+L21".
                blocks.append(
                    {"structure": "row", "cells": _row_cells(got), "lines": [ln.id for ln in got]}
                )
            elif got:
                # "R L20-L35" is a table, not a row: one row per line.
                for ln in got:
                    blocks.append({"structure": "row", "cells": ln.cells(), "lines": [ln.id]})
            elif rest:
                # Cells written out, or several cell fields each a ref.
                cells: list[str] = []
                ids = []
                for c in rest:
                    t, i = text_of(c)
                    cells.append(t)
                    ids.extend(i)
                blocks.append({"structure": "row", "cells": cells, "lines": ids})
        elif sigil == "F":
            lim = rest[0] if rest else ""
            body = rest[1:]
            if len(body) == 1 and parse_refs(body[0]):
                got, _ = take(body[0])
                label, value = _field_parts(got) if got else ("", "")
                ids = [ln.id for ln in got]
            else:
                label, i1 = text_of(body[0] if body else "")
                value, i2 = text_of("\t".join(body[1:]))
                ids = i1 + i2
            blocks.append(
                {"structure": "field", "lim": lim, "label": label, "value": value, "lines": ids}
            )
        elif sigil == "O":
            lim = rest[0] if rest else ""
            label, i1 = text_of(rest[1] if len(rest) > 1 else "")
            value, i2 = text_of(rest[2] if len(rest) > 2 else "")
            options: list[str] = []
            checked: list[str] = []
            ids = i1 + i2
            for tok in rest[3:]:
                starred = tok.startswith("*")
                body_t, i3 = text_of(tok[1:] if starred else tok)
                ids.extend(i3)
                options.append(body_t)
                if starred:
                    checked.append(body_t)
            blocks.append(
                {
                    "structure": "field",
                    "lim": lim,
                    "label": label,
                    "value": value,
                    "options": options,
                    "checked": checked,
                    "lines": ids,
                }
            )
        elif sigil == "C":
            # A continuation finishes a sentence cut by the window boundary,
            # so it starts mid-sentence. Lines that open with a capital or a
            # list mark are a new block the model mislabelled; keep them as
            # a paragraph rather than gluing them onto the previous window.
            ids = parse_refs(tail) or []
            first = by_id.get(ids[0]) if ids else None
            head = first.text.lstrip()[:1] if first is not None else ""
            if first is not None and (head.isupper() or head in "•-([" or len(ids) > 8):
                text, got_ids = text_of(tail)
                blocks.append({"structure": "p", "text": text, "lines": got_ids})
                res.demoted_continuations += 1
            else:
                text, _ids = text_of(tail)
                res.payload["continues"] = text
        elif sigil == "X":
            for lid in parse_refs(tail) or []:
                dln = by_id.get(lid)
                if dln is None or lid in res.assigned or not dln.may_be_decoration():
                    continue
                per_page = sum(1 for j in res.decorations if by_id[j].page == dln.page)
                if per_page < _DECORATION_MAX_PER_PAGE:
                    res.decorations.add(lid)
        else:
            res.malformed += 1
    return res


# ---- Lines nobody claimed ------------------------------------------------------


def fallback_block_dicts(
    lines: list[TextLine], vocab: set[str] | None = None
) -> list[dict[str, Any]]:
    """Plain blocks for unclaimed lines, the same rules as the completeness
    pass's fallback: a line with cells and a figure is a row, a short bold
    line a heading, consecutive other lines one paragraph."""
    out: list[dict[str, Any]] = []
    para: list[TextLine] = []

    def flush() -> None:
        nonlocal para
        if para:
            out.append(
                {
                    "structure": "p",
                    "text": _join_lines([ln.text for ln in para], vocab),
                    "lines": [ln.id for ln in para],
                    "fallback": True,
                }
            )
            para = []

    for ln in lines:
        cells = ln.cells()
        if len(cells) >= 2 and ln.has_figure:
            flush()
            out.append({"structure": "row", "cells": cells, "lines": [ln.id], "fallback": True})
        elif ln.n_words <= 8 and ln.bold and not ln.text.rstrip().endswith("."):
            flush()
            out.append(
                {
                    "structure": "heading",
                    "level": 3,
                    "text": ln.text,
                    "lines": [ln.id],
                    "fallback": True,
                }
            )
        else:
            if para and ln.id != para[-1].id + 1:
                flush()
            para.append(ln)
    flush()
    return out


def complete_payload(res: LinesResult, lines: list[TextLine], vocab: set[str] | None = None) -> int:
    """Insert fallback blocks for the window's unclaimed lines into the payload
    at their reading position (after the last block whose lines precede
    them). Returns how many lines were filled."""
    missing = res.unassigned(lines)
    if not missing:
        return 0
    blocks: list[dict[str, Any]] = res.payload["blocks"]
    fallback = fallback_block_dicts(missing, vocab)
    # Position by the first line id of each fallback block.
    maxima = [max(b.get("lines") or [0]) for b in blocks]
    out: list[dict[str, Any]] = []
    fi = 0
    for b, mx in zip(blocks, maxima, strict=True):
        while fi < len(fallback) and fallback[fi]["lines"][0] < min(b.get("lines") or [10**9]):
            out.append(fallback[fi])
            fi += 1
        out.append(b)
        _ = mx
    out.extend(fallback[fi:])
    res.payload["blocks"] = out
    for ln in missing:
        res.assigned.add(ln.id)
    return len(missing)


def order_by_lines(res: LinesResult) -> int:
    """Put the window's blocks in the page's reading order (by first line id;
    a block without lines keeps its place after the previous one). The
    listing is in reading order, so a block the model emitted out of turn is
    a slip, not a decision. Returns how many blocks moved."""
    blocks: list[dict[str, Any]] = res.payload["blocks"]
    keyed: list[tuple[int, int, dict[str, Any]]] = []
    last = 0
    for i, b in enumerate(blocks):
        ids = b.get("lines") or []
        if ids:
            last = min(ids)
        keyed.append((last, i, b))
    ordered = [b for _, _, b in sorted(keyed, key=lambda t: (t[0], t[1]))]
    moved = sum(1 for a, b in zip(blocks, ordered, strict=True) if a is not b)
    res.payload["blocks"] = ordered
    return moved


def merge_results(base: LinesResult, extra: LinesResult) -> None:
    """Fold the blocks of a rescue call (made over the lines *base* left
    unaccounted for) into *base*, each at its reading position."""
    blocks: list[dict[str, Any]] = base.payload["blocks"]
    new = [b for b in extra.payload["blocks"] if b.get("lines")]
    if not new:
        return
    out: list[dict[str, Any]] = []
    ni = 0
    for b in blocks:
        first = min(b.get("lines") or [10**9])
        while ni < len(new) and min(new[ni]["lines"]) < first:
            out.append(new[ni])
            ni += 1
        out.append(b)
    out.extend(new[ni:])
    base.payload["blocks"] = out
    base.assigned |= extra.assigned
    base.decorations |= extra.decorations
    if extra.payload.get("continues") and not base.payload.get("continues"):
        base.payload["continues"] = extra.payload["continues"]


def insert_page_blocks(
    res: LinesResult, lines: list[TextLine], page: int, block_dicts: list[dict[str, Any]]
) -> None:
    """Splice blocks transcribed for *page* (no text layer, text-format call)
    into the payload at that page's place: before the first block whose lines
    sit on a later page."""
    if not block_dicts:
        return
    page_of = {ln.id: ln.page for ln in lines}
    blocks: list[dict[str, Any]] = res.payload["blocks"]
    at = len(blocks)
    for i, b in enumerate(blocks):
        ids = b.get("lines") or []
        if ids and min(page_of.get(x, 0) for x in ids) > page:
            at = i
            break
    for b in block_dicts:
        b["page"] = page
    res.payload["blocks"] = blocks[:at] + list(block_dicts) + blocks[at:]


def describe_unassigned(lines: list[TextLine]) -> str:
    """Compact id list for a retry prompt: ``L4, L9-L12, L40``."""
    ids = sorted(ln.id for ln in lines)
    if not ids:
        return ""
    spans: list[str] = []
    start = prev = ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        spans.append(f"L{start}" if start == prev else f"L{start}-L{prev}")
        start = prev = i
    spans.append(f"L{start}" if start == prev else f"L{start}-L{prev}")
    return ", ".join(spans)
