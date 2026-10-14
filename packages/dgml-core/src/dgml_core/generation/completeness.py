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

"""Completeness of a transcription against the page text, by inverse grounding.

Pass A asks a model to transcribe pages. The model sometimes skips a page, a
table or a paragraph, and no retry recovers it: a window that lost one page
in ten still covers ~90% of the window's words and passes the window gate.
This pass makes the result complete by construction:

1. **Align** the transcribed blocks against the page words with the grounding
   aligner (:mod:`dgml_core.xml_grounding`): unique-n-gram anchors, a
   longest-increasing-subsequence pass, SequenceMatcher inside the windows;
   then an order-free pass that matches each block's remaining tokens to
   unclaimed words with the same core inside the block's neighbourhood, so a
   table the text layer reads column by column is claimed whatever order the
   model wrote its rows in. Each page word can be claimed by at most one
   block token, so a lost page's words stay unclaimed even when the same
   words occur elsewhere.
2. **Localise** the unclaimed words, with their boxes, into regions: a whole
   page, a block within a page, or noise (page furniture, which is ignored).
3. **Fill** each region with plain blocks built from the text layer itself --
   lines by baseline, rows by column gaps -- at the region's reading-order
   position. No model call: what the page says is what the page says. A
   caller may pass ``rescue`` to let the model re-transcribe a whole-page
   region first; its blocks replace the fallback only if they claim the
   region's words, so a rescue cannot be a hallucination either.
4. **Drop** blocks that claim nothing: a block repeating a claimed block is a
   duplicate (window overlap); a block whose words are on no page is
   fabricated.
5. **Report**, per page, what the model covered, what was filled in, and what
   was dropped -- so a document's completeness is a number, not a hope.

Alignment is local and takes seconds; the only model spend is the optional
whole-page rescue.
"""

from __future__ import annotations

import itertools
import json
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dgml_core.generation.blocks import Block
from dgml_core.textmatch import PageDims, Word, core_token, fuzzy_norm, reorder_words_by_cell
from dgml_core.xml_grounding import _align_window

# A region smaller than this is page furniture (a page number, a running
# header, a stray figure caption) unless it carries figures.
_REGION_MIN_WORDS = 8
_REGION_MIN_FIGURES = 2
# A region holding this share of a page's content words is the page itself.
_PAGE_REGION_SHARE = 0.7
# Cells on one line split where the gap exceeds this many word heights;
# words within a cell sit about a third of a height apart.
_CELL_GAP_HEIGHTS = 1.5
# Lines further apart than this many line heights start a new region.
_REGION_GAP_HEIGHTS = 1.6
# A block claiming under this share of its words after the local pass is
# suspect; a duplicate must repeat a claimed block within this many blocks
# (a page's worth), a fabricated paragraph must be this long and claim less
# than this share.
_UNCLAIMED_MAX_SHARE = 0.2
_DUPLICATE_WINDOW = 80
_FABRICATED_MIN_TOKENS = 8
_FABRICATED_MAX_SHARE = 0.1
# A whole-page rescue is accepted when it claims at least this share of the
# region's words; otherwise the text-layer fallback stands.
_RESCUE_MIN_CLAIM = 0.9


def _core(tok: str) -> str:
    """The aligner's token key, NFKC-normalised first: PDF text layers carry
    ligatures ('proﬁt') where a model writes plain letters, and the two must
    meet on the same core."""
    return core_token(unicodedata.normalize("NFKC", tok))


def _is_figure(core: str) -> bool:
    return any(ch.isdigit() for ch in core)


# ---- Page words -------------------------------------------------------------


def load_page_words_dir(page_text_dir: Path | str) -> dict[int, tuple[list[Word], PageDims]]:
    """Per-page words from a ``page_text/`` directory, in cell-aware reading
    order (the order grounding uses), keyed by 1-based page number."""
    out: dict[int, tuple[list[Word], PageDims]] = {}
    for pf in sorted(Path(page_text_dir).glob("page_*.json")):
        m = re.search(r"page_(\d+)", pf.stem)
        if not m:
            continue
        page = int(m.group(1))
        payload = json.loads(pf.read_text(encoding="utf-8"))
        dims = PageDims(int(payload.get("width") or 1000), int(payload.get("height") or 1000))
        words: list[Word] = []
        boxed = True
        for i, w in enumerate(payload.get("words", []) or []):
            box = w.get("l")
            if not box:
                boxed = False
                box = (0, 0, 0, 0)
            s = w.get("s") or {}
            left, top, right, bottom = box
            words.append(
                Word(
                    idx=i,
                    text=w["t"],
                    text_norm=fuzzy_norm(w["t"]),
                    left=round(left),
                    top=round(top),
                    right=round(right),
                    bottom=round(bottom),
                    bold=bool(s.get("b")),
                    italic=bool(s.get("i")),
                    size=float(s["sz"]) if s.get("sz") is not None else None,
                )
            )
        out[page] = (reorder_words_by_cell(words) if boxed and words else words, dims)
    return out


# ---- Alignment ----------------------------------------------------------------


# The text layer splits punctuation off into words of its own -- "1" "," "217",
# "U" "." "S" ".", "(" "1,234" ")" -- at zero gap, while a real word space is a
# fifth of a word height or more. Touching pieces with punctuation on one side
# are one token to the model ("1,217") and are merged back before alignment.
_TOUCH_GAP_HEIGHTS = 0.12


def _pure_punct(text: str) -> bool:
    t = text.strip()
    return bool(t) and not any(ch.isalnum() for ch in t)


def _merge_split_words(words: list[Word]) -> list[list[Word]]:
    groups: list[list[Word]] = []
    for w in words:
        if groups:
            prev = groups[-1][-1]
            h = max(1, prev.bottom - prev.top)
            boxed = prev.right > prev.left
            same_line = min(prev.bottom, w.bottom) - max(prev.top, w.top) > 0.5 * h
            touching = -0.2 * h <= w.left - prev.right <= _TOUCH_GAP_HEIGHTS * h
            if boxed and same_line and touching and (_pure_punct(prev.text) or _pure_punct(w.text)):
                groups[-1].append(w)
                continue
        groups.append([w])
    return groups


@dataclass
class _OTok:
    """One page token in the alignment stream: a word, or touching pieces the
    text layer split that the model would write as one."""

    page: int
    words: list[Word]
    core: str

    @property
    def text(self) -> str:
        return "".join(w.text for w in self.words) if len(self.words) > 1 else self.words[0].text

    @property
    def idx(self) -> int:
        return self.words[0].idx

    @property
    def left(self) -> int:
        return min(w.left for w in self.words)

    @property
    def right(self) -> int:
        return max(w.right for w in self.words)

    @property
    def top(self) -> int:
        return min(w.top for w in self.words)

    @property
    def bottom(self) -> int:
        return max(w.bottom for w in self.words)

    @property
    def y_center(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def bold(self) -> bool:
        return all(w.bold for w in self.words)


@dataclass
class Alignment:
    otoks: list[_OTok]
    page_start: dict[int, int]  # page -> index of its first otok
    claimed: dict[int, int]  # otok index -> block index
    block_tokens: list[int]  # per block: content tokens
    block_claimed: list[int]  # per block: tokens that found a word
    block_last: list[int]  # per block: last claimed otok index (-1 if none)

    def page_words(self, page: int) -> list[int]:
        """otok indices of one page."""
        lo = self.page_start.get(page, 0)
        hi = next((s for p, s in sorted(self.page_start.items()) if p > page), len(self.otoks))
        return list(range(lo, hi))


def _ocr_stream(
    pages: dict[int, tuple[list[Word], PageDims]],
) -> tuple[list[_OTok], dict[int, int]]:
    otoks: list[_OTok] = []
    page_start: dict[int, int] = {}
    for page in sorted(pages):
        page_start[page] = len(otoks)
        for group in _merge_split_words(pages[page][0]):
            c = _core("".join(w.text for w in group))
            if c:
                otoks.append(_OTok(page, group, c))
    return otoks, page_start


def align_blocks(
    blocks: list[Block],
    pages: dict[int, tuple[list[Word], PageDims]],
    *,
    order_free: bool = True,
) -> Alignment:
    """Align block text (document order) against the page words (reading order).
    Each page word is claimed by at most one block token. With *order_free*
    off only the ordered diff pass runs -- the view the duplicate check
    needs, since the order-free pass lets a repeated block scavenge stray
    words and look partly claimed."""
    otoks, page_start = _ocr_stream(pages)
    ocores = [t.core for t in otoks]
    xcores: list[str] = []
    xblock: list[int] = []
    block_tokens = [0] * len(blocks)
    for bi, b in enumerate(blocks):
        for tok in b.flat_text().split():
            c = _core(tok)
            if c:
                xcores.append(c)
                xblock.append(bi)
                block_tokens[bi] += 1
    pairs: dict[int, int] = {}
    stats = {"anchor_candidates": 0, "anchors_used": 0, "windows_abandoned": 0}
    if xcores and ocores:
        _align_window(xcores, ocores, 0, len(xcores), 0, len(ocores), pairs, stats)
    claimed: dict[int, int] = {}
    xclaimed: set[int] = set()
    block_claimed = [0] * len(blocks)
    block_first = [len(otoks)] * len(blocks)
    block_last = [-1] * len(blocks)
    for xi, oi in pairs.items():
        if oi in claimed:
            continue
        bi = xblock[xi]
        claimed[oi] = bi
        xclaimed.add(xi)
        block_claimed[bi] += 1
        block_first[bi] = min(block_first[bi], oi)
        block_last[bi] = max(block_last[bi], oi)

    # Pass 2 -- order-free, locality-bounded. SequenceMatcher cannot claim a
    # table the text layer reads column by column when the model wrote rows
    # (grounding proper has later passes for exactly this). Each block's
    # remaining tokens are matched to unclaimed words with the same core inside
    # the block's neighbourhood: from where the previous aligned block ended to
    # where the next one begins, widened to whole pages. A page no block was
    # aligned to is reached only by blocks sandwiched between its neighbours,
    # which is what keeps a genuinely lost page unclaimed.
    if not order_free:
        return Alignment(otoks, page_start, claimed, block_tokens, block_claimed, block_last)
    # Only a block the diff credited with most of its words fences a
    # neighbourhood: a row whose labels repeat on an earlier table can be
    # credited a word or two there, and must not trap its neighbours.
    n = len(blocks)
    anchored = [block_tokens[i] > 0 and block_claimed[i] / block_tokens[i] >= 0.5 for i in range(n)]
    prev_last = [-1] * n
    cur = -1
    for i in range(n):
        prev_last[i] = cur
        if anchored[i]:
            cur = block_last[i]
    next_first = [len(otoks)] * n
    cur = len(otoks)
    for i in range(n - 1, -1, -1):
        next_first[i] = cur
        if anchored[i]:
            cur = block_first[i]
    starts = sorted(page_start.values()) + [len(otoks)]

    def page_bounds(lo: int, hi: int, *, include_hi_page: bool) -> tuple[int, int]:
        """Widen [lo, hi) to page starts. `hi` is where the next aligned block
        begins -- very often the first word of a page. The narrow form stops
        there; the wide form takes that page too, because a table or footnote
        the model wrote before the page's first paragraph sits on it."""
        lo = max(0, min(lo, len(otoks) - 1))
        hi = max(lo + 1, min(hi, len(otoks)))
        plo = max(st for st in starts if st <= lo)
        after = [st for st in starts if (st > hi if include_hi_page else st >= hi)]
        phi = min(after) if after else len(otoks)
        return plo, phi

    pending: dict[int, Counter[str]] = {}
    for xi, c in enumerate(xcores):
        if xi not in xclaimed:
            pending.setdefault(xblock[xi], Counter())[c] += 1
    # Narrow neighbourhoods first, so every block claims what sits between
    # its aligned neighbours before anyone may reach into the next page;
    # then, for what is still pending, the neighbourhood widened by a page.
    for include_hi_page in (False, True):
        for bi, need in pending.items():
            if not +need:
                continue
            lo = block_first[bi] if block_first[bi] < len(otoks) else prev_last[bi] + 1
            hi = max(block_last[bi] + 1, next_first[bi])
            lo, hi = page_bounds(min(lo, prev_last[bi] + 1), hi, include_hi_page=include_hi_page)
            for oi in range(lo, hi):
                if oi in claimed:
                    continue
                c = otoks[oi].core
                if need.get(c, 0) > 0:
                    need[c] -= 1
                    claimed[oi] = bi
                    block_claimed[bi] += 1
                    block_first[bi] = min(block_first[bi], oi)
                    block_last[bi] = max(block_last[bi], oi)
    return Alignment(otoks, page_start, claimed, block_tokens, block_claimed, block_last)


# ---- Dropping duplicates and fabrications -----------------------------------


def _dropped_entry(
    blocks: list[Block], al: Alignment, i: int, share: float, reason: str
) -> dict[str, Any]:
    b = blocks[i]
    return {
        "id": b.id,
        "structure": b.structure,
        "reason": reason,
        "share": round(share, 2),
        "tokens": al.block_tokens[i],
        "text": b.flat_text()[:160],
    }


def _shares(al: Alignment) -> list[float]:
    return [c / n if n else 1.0 for n, c in zip(al.block_tokens, al.block_claimed, strict=True)]


def _drop_duplicates(
    blocks: list[Block], al: Alignment, window_starts: list[int] | None = None
) -> tuple[list[Block], list[dict[str, Any]]]:
    """Remove repeated blocks -- what window overlap and retry ladders produce.

    With *window_starts* (the block index each transcription window begins
    at) the overlap is known: a window's leading blocks that repeat, word for
    word or as a part of, the previous window's trailing blocks are the
    overlap page transcribed twice, and go. Without it, two blocks with the
    same words within a page's worth of blocks are a duplicate pair when one
    of them aligned nothing in order: the copy the ORDERED alignment credited
    less goes, the later one on a tie (a column-wise table credits neither).
    A short block inside a longer neighbour is dropped only when it claimed
    almost nothing and the neighbour claimed most of its own. *al* must come
    from ``align_blocks(..., order_free=False)``.
    """
    shares = _shares(al)
    cores = [" ".join(_core(t) for t in b.flat_text().split() if _core(t)) for b in blocks]
    gone: set[int] = set()
    if window_starts:
        bounds = sorted(set(window_starts)) + [len(blocks)]
        page_of = [al.otoks[t].page if t >= 0 else None for t in al.block_last]
        for w in range(1, len(bounds) - 1):
            # The overlap page is the last page the previous window's blocks
            # aligned to; its transcription there runs from the first block
            # credited on that page to the window's end.
            prev = range(bounds[w - 1], bounds[w])
            last_pages = [page_of[j] for j in prev if page_of[j] is not None and shares[j] >= 0.5]
            if not last_pages:
                continue
            overlap_page = max(last_pages)
            first_on_page = next(j for j in prev if page_of[j] == overlap_page)
            trail_cores = [cores[j] for j in range(first_on_page, bounds[w]) if cores[j]]
            # The new window's transcription of the overlap page ends where
            # its first block credited on a later page begins.
            for i in range(bounds[w], min(bounds[w + 1], bounds[w] + 120)):
                if shares[i] >= 0.5 and page_of[i] is not None and page_of[i] > overlap_page:
                    break
                if al.block_tokens[i] < 3 or not cores[i] or shares[i] >= 0.5:
                    continue
                long = al.block_tokens[i] >= _FABRICATED_MIN_TOKENS
                if any(cores[i] == c or (long and cores[i] in c) for c in trail_cores):
                    gone.add(i)
    if not window_starts:
        for i, (n_tok, sh) in enumerate(zip(al.block_tokens, shares, strict=True)):
            if i in gone or n_tok < 3 or not cores[i] or sh >= _UNCLAIMED_MAX_SHARE:
                continue
            lo, hi = max(0, i - _DUPLICATE_WINDOW), min(len(blocks), i + _DUPLICATE_WINDOW + 1)
            if any(
                j != i and j not in gone and cores[j] == cores[i] and (shares[j], -j) >= (sh, -i)
                for j in range(lo, hi)
            ):
                gone.add(i)
    keep = [b for i, b in enumerate(blocks) if i not in gone]
    dropped = [_dropped_entry(blocks, al, i, shares[i], "duplicate") for i in sorted(gone)]
    return keep, dropped


def _drop_unplaced(blocks: list[Block], al: Alignment) -> tuple[list[Block], list[dict[str, Any]]]:
    """Judge the blocks the FULL alignment could not place.

    One whose words all sit inside a neighbour that claimed most of its own
    is a duplicate (a retry re-emitting part of a window). A long non-row
    block whose distinctive words (5+ letters, or figures) mostly occur
    nowhere in the text layer is fabricated -- a block the aligner merely
    failed to place still has its words on some page. Rows are never dropped
    as fabricated: a figure row wrongly deleted costs more than one kept
    twice.
    """
    shares = _shares(al)
    cores = [" ".join(_core(t) for t in b.flat_text().split() if _core(t)) for b in blocks]
    doc_cores = {t.core for t in al.otoks}
    keep: list[Block] = []
    dropped: list[dict[str, Any]] = []
    for i, (b, n_tok, sh) in enumerate(zip(blocks, al.block_tokens, shares, strict=True)):
        if n_tok >= 3 and sh < _UNCLAIMED_MAX_SHARE and cores[i]:
            lo, hi = max(0, i - _DUPLICATE_WINDOW), min(len(blocks), i + _DUPLICATE_WINDOW + 1)
            if any(j != i and shares[j] >= 0.5 and cores[i] in cores[j] for j in range(lo, hi)):
                dropped.append(_dropped_entry(blocks, al, i, sh, "duplicate"))
                continue
        if b.structure != "row" and n_tok >= _FABRICATED_MIN_TOKENS and sh < _FABRICATED_MAX_SHARE:
            distinct = [
                c for c in (_core(t) for t in b.flat_text().split()) if len(c) >= 5 or _is_figure(c)
            ]
            if len(distinct) >= 3 and sum(c in doc_cores for c in distinct) / len(distinct) < 0.5:
                dropped.append(_dropped_entry(blocks, al, i, sh, "fabricated"))
                continue
        keep.append(b)
    return keep, dropped


# ---- Regions ------------------------------------------------------------------


@dataclass
class Region:
    page: int
    kind: str  # "page" | "block"
    lines: list[list[_OTok]]  # unclaimed words, grouped by visual line
    pos: int  # otok index of the first word, for placement
    source: str = "text-layer"  # or "model-rescue"
    blocks: list[Block] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return sum(len(ln) for ln in self.lines)

    @property
    def n_figures(self) -> int:
        return sum(_is_figure(t.core) for ln in self.lines for t in ln)

    @property
    def bbox(self) -> list[int]:
        ts = [t for ln in self.lines for t in ln]
        return [
            min(t.left for t in ts),
            min(t.top for t in ts),
            max(t.right for t in ts),
            max(t.bottom for t in ts),
        ]

    @property
    def first_words(self) -> str:
        return " ".join(t.text for t in self.lines[0][:8])


def _group_lines(toks: list[_OTok]) -> list[list[_OTok]]:
    """Visual lines: words whose vertical centres fall within the line's height."""
    if not toks:
        return []
    ordered = sorted(toks, key=lambda t: (t.top, t.left))
    h = statistics.median(max(1, t.bottom - t.top) for t in ordered)
    lines: list[list[_OTok]] = []
    for t in ordered:
        if lines and abs(t.y_center - lines[-1][0].y_center) <= 0.6 * h:
            lines[-1].append(t)
        else:
            lines.append([t])
    for ln in lines:
        ln.sort(key=lambda t: t.left)
    return lines


def find_regions(al: Alignment, pages: dict[int, tuple[list[Word], PageDims]]) -> list[Region]:
    regions: list[Region] = []
    for page in sorted(pages):
        idxs = al.page_words(page)
        unclaimed = [i for i in idxs if i not in al.claimed]
        if not unclaimed:
            continue
        lines = _group_lines([al.otoks[i] for i in unclaimed])
        h = statistics.median(max(1, t.bottom - t.top) for ln in lines for t in ln)
        groups: list[list[list[_OTok]]] = []
        for ln in lines:
            if groups and ln[0].top - groups[-1][-1][0].bottom <= _REGION_GAP_HEIGHTS * h:
                groups[-1].append(ln)
            else:
                groups.append([ln])
        n_page = len(idxs)
        for g in groups:
            n = sum(len(ln) for ln in g)
            nf = sum(_is_figure(t.core) for ln in g for t in ln)
            if n < _REGION_MIN_WORDS and nf < _REGION_MIN_FIGURES:
                continue
            # position = the smallest otok index among the region's words
            members = {id(t) for ln in g for t in ln}
            pos = min(i for i in unclaimed if id(al.otoks[i]) in members)
            kind = "page" if n >= _PAGE_REGION_SHARE * n_page else "block"
            regions.append(Region(page=page, kind=kind, lines=g, pos=pos))
    return regions


# ---- Fallback blocks from the text layer --------------------------------------


def fallback_blocks(region: Region, prefix: str) -> list[Block]:
    """Plain blocks for a region: a line with column gaps and a figure becomes a
    row; a short bold line a heading; consecutive other lines one paragraph."""
    out: list[Block] = []
    para: list[str] = []
    n = 0

    def flush() -> None:
        nonlocal para, n
        if para:
            n += 1
            out.append(Block(id=f"{prefix}{n:03d}", structure="p", text=" ".join(para)))
            para = []

    for ln in region.lines:
        h = statistics.median(max(1, t.bottom - t.top) for t in ln)
        cells: list[list[_OTok]] = [[ln[0]]]
        for prev, t in itertools.pairwise(ln):
            if t.left - prev.right > _CELL_GAP_HEIGHTS * h:
                cells.append([t])
            else:
                cells[-1].append(t)
        texts = [" ".join(t.text for t in c) for c in cells]
        if len(cells) >= 2 and any(_is_figure(t.core) for t in ln):
            flush()
            n += 1
            out.append(Block(id=f"{prefix}{n:03d}", structure="row", cells=texts))
        elif len(ln) <= 8 and all(t.bold for t in ln) and not texts[-1].rstrip().endswith("."):
            flush()
            n += 1
            out.append(
                Block(id=f"{prefix}{n:03d}", structure="heading", level=3, text=" ".join(texts))
            )
        else:
            para.append(" ".join(texts))
    flush()
    return out


# ---- The pass -----------------------------------------------------------------


def _insert_positions(blocks: list[Block], al: Alignment, regions: list[Region]) -> list[int]:
    """For each region, the block index after which its blocks go: the last
    block whose claimed words precede the region in reading order."""
    last = -1
    cursor: list[int] = []
    for bl in al.block_last:
        if bl >= 0:
            last = bl
        cursor.append(last)
    out = []
    for r in regions:
        after = -1
        for i, c in enumerate(cursor):
            if c < r.pos and c >= 0:
                after = i
        out.append(after)
    return out


def _claim_share(blocks: list[Block], al_page: Alignment, region: Region) -> float:
    """Share of a region's words that *blocks* claim when aligned to its page."""
    want = {(t.page, t.idx) for ln in region.lines for t in ln}
    got = sum(
        1 for oi in al_page.claimed if (al_page.otoks[oi].page, al_page.otoks[oi].idx) in want
    )
    return got / max(1, len(want))


def ensure_complete(
    blocks: list[Block],
    page_text_dir: Path | str,
    *,
    rescue: Callable[[int], list[Block] | None] | None = None,
    log: Callable[[str], None] = lambda _m: None,
    id_prefix: str = "f",
    window_starts: list[int] | None = None,
) -> tuple[list[Block], dict[str, Any]]:
    """Return the blocks made complete against the page text, and a report.

    ``rescue(page_index)`` (0-based) may re-transcribe a whole-page region with
    the model; its blocks are used only if they claim the region's words.
    ``window_starts`` -- the index in *blocks* at which each transcription
    window's output begins -- lets overlap duplicates be removed by position.
    """
    pages = load_page_words_dir(page_text_dir)
    if not pages or not blocks:
        return blocks, {"skipped": "no page text" if not pages else "no blocks"}

    blocks, dropped = _drop_duplicates(
        blocks, align_blocks(blocks, pages, order_free=False), window_starts
    )
    al = align_blocks(blocks, pages)
    model_covered = len(al.claimed)
    n_content = len(al.otoks)
    blocks, late = _drop_unplaced(blocks, al)
    dropped += late
    fabricated = sum(1 for d in dropped if d["reason"] == "fabricated")
    dups = len(dropped) - fabricated
    if late:
        al = align_blocks(blocks, pages)
    regions = find_regions(al, pages)

    # Fill regions. Whole pages may go to the model first; everything else, and
    # any rescue that fails to claim its page, comes from the text layer.
    for r in regions:
        got: list[Block] | None = None
        if r.kind == "page" and rescue is not None:
            got = rescue(r.page - 1)
            if got:
                sub = {r.page: pages[r.page]}
                probe = Region(r.page, r.kind, r.lines, 0)
                share = _claim_share(got, align_blocks(got, sub), probe)
                if share >= _RESCUE_MIN_CLAIM:
                    r.source = "model-rescue"
                else:
                    log(
                        f"page {r.page}: rescue claimed only {share:.0%} of the missing words; "
                        f"using the text layer"
                    )
                    got = None
        r.blocks = got or fallback_blocks(r, f"{id_prefix}{r.page:03d}_")

    positions = _insert_positions(blocks, al, regions)
    out: list[Block] = []
    by_after: dict[int, list[Block]] = {}
    for r, after in zip(regions, positions, strict=True):
        by_after.setdefault(after, []).extend(r.blocks)
    out.extend(by_after.get(-1, []))
    for i, b in enumerate(blocks):
        out.append(b)
        out.extend(by_after.get(i, []))

    # Verify the result the same way, so the report describes what was kept.
    final = align_blocks(out, pages) if regions else al
    per_page = []
    for page in sorted(pages):
        idxs = al.page_words(page)
        figs = [i for i in idxs if _is_figure(al.otoks[i].core)]
        fidx = final.page_words(page)
        per_page.append(
            {
                "page": page,
                "words": len(idxs),
                "claimed_by_model": sum(1 for i in idxs if i in al.claimed),
                "filled": sum(r.n_words for r in regions if r.page == page),
                "unclaimed_after": sum(1 for i in fidx if i not in final.claimed),
                "figures": len(figs),
                "unclaimed_figures_after": sum(
                    1 for i in fidx if _is_figure(final.otoks[i].core) and i not in final.claimed
                ),
            }
        )
    report = {
        "pages": per_page,
        "regions": [
            {
                "page": r.page,
                "kind": r.kind,
                "words": r.n_words,
                "figures": r.n_figures,
                "bbox": r.bbox,
                "source": r.source,
                "blocks": len(r.blocks),
                "begins": r.first_words,
            }
            for r in regions
        ],
        "dropped": {"duplicates": dups, "fabricated": fabricated},
        "dropped_blocks": dropped,
        "content_words": n_content,
        "model_coverage_pct": round(100 * model_covered / n_content, 1) if n_content else 100.0,
        "final_coverage_pct": (
            round(100 * len(final.claimed) / n_content, 1) if n_content else 100.0
        ),
    }
    filled_pages = sorted({r.page for r in regions})
    log(
        f"completeness: model covered {report['model_coverage_pct']}% of page words; "
        f"{len(regions)} region(s) filled"
        + (f" on page(s) {', '.join(map(str, filled_pages))}" if filled_pages else "")
        + f"; dropped {dups} duplicate, {fabricated} fabricated block(s); "
        f"final coverage {report['final_coverage_pct']}%"
    )
    return out, report
