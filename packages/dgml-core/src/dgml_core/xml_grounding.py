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

"""Ground a generated DGML XML document against its source page OCR.

``dgml docset generate`` produces ``<stem>.dgml.xml`` — the whole
document's text as a semantic XML tree in natural reading order. This
module aligns that tree against the workspace's OCR words
(``page_text/page_N.json``) and adds a ``dg:origin`` attribute to every
element whose subtree grounded — leaf elements, mixed-content parents,
and pure containers alike. Generation runs grounding in place so the canonical
``<stem>.dgml.xml`` already carries ``dg:origin``; callers can also
write the annotated tree to a separate path (the default
``<stem>.dgml.grounded.xml`` sibling) by passing ``output_path``.

Unlike grounded value extraction (:mod:`dgml.matching`), there are no
page anchors to start from. Both streams, however, claim to be "the
document in reading order":

- stream A — the XML's text nodes in document order;
- stream B — the OCR words, pages concatenated 1..N in cell-aware
  reading order.

So grounding is a *sequence alignment* problem, solved patience-diff
style:

1. **Anchor.** N-grams of normalized tokens that occur exactly once in
   each stream are candidate sync points; a longest-increasing-
   subsequence pass drops the ones that violate monotonic order
   (multi-column wobble, OCR serialization quirks). Anchors partition
   both streams into small aligned windows — this manufactures the
   page hint that value extraction gets for free.
2. **Align windows.** Each window is aligned token-by-token with
   ``difflib.SequenceMatcher`` (windows still too large recurse with
   within-window-unique anchors before falling back). Headers/footers
   the XML skipped become OCR-side gaps; LLM-dropped or hallucinated
   text becomes XML-side gaps — both are tolerated for free.
3. **Recover gaps.** Unmatched token runs bounded by matched neighbors
   are compared against the corresponding OCR gap with the
   character-class weighted similarity from :mod:`dgml.textmatch`
   (punctuation/case nearly free, digit edits expensive) — catching
   OCR letter slips and joined/split words without ever mis-grounding
   a number.
4. **Rescue duplicates.** Text nodes still fully unmatched (repeated
   boilerplate the aligner could only consume once — cover-page
   address blocks, running headers) get a direct span search on the
   pages bracketing their aligned neighbors, reusing already-claimed
   words.
5. **Ground by row context.** Nodes only their sibling window can
   place: punctuation-only cells (a ``?`` in a comparison table —
   invisible to passes 1-4) ground to the matching word nearest their
   row; interleaved multi-line table cells (whose words the OCR
   stream shuffles line-by-line with a touching neighbor cell's, so
   no contiguous span exists) ground to the in-order subsequence of
   unclaimed row-window words that forms one connected cell; and
   letters+digits text whose digits disagree with OCR (a date the
   generator transcribed as "Jun 18" where the page reads "Jun 16")
   grounds to the digit-masked-equal span in its row — the location
   is not in question there, only the digits.
6. **Assemble stranded cells.** Whatever remains gets one last
   page-wide subsequence assembly with the same safety gates as pass
   5's interleave handling (exact cores, unclaimed words, one-cell
   coherence). This is for the doubly-unlucky cell: interleaved (no
   contiguous span anywhere) AND with every window pointing away —
   two tables sharing header labels serialize transposed vs. the
   page, so the cell's siblings all mis-ground onto the other table.
   A longest-prefix variant grounds label-plus-value text whose
   trailing value the page never rendered as words (chart-only
   numbers), committed as a partial.

Boxes aggregate bottom-up: an element's region is the union of its
subtree's matched words. Elements with text-node children (leaves and
mixed-content parents) emit one box per visual line on each page (the
CSS ``getClientRects()`` model); pure containers (all-element children
— sections, lists, tables, rows, the root) emit one union box per page
(the ``getBoundingClientRect()`` model), since repeating every
descendant's line boxes up each ancestor would bloat the document.
Each box is ``<page> <x1> <y1> <x2> <y2>``
(space-separated) in integer image pixels (top-left origin, 300 dpi,
relative to ``page_images/page_N.png``), boxes ``"; "``-separated, e.g.::

    dg:origin="3 307 367 1098 428; 4 307 254 1093 376"

A ``<stem>.dgml.grounding_stats.json`` sidecar reports match rates so
generation gaps (DGML doesn't always cover the full document) stay
visible.
"""

from __future__ import annotations

import difflib
import re
from bisect import bisect_right
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .style_config import StyleConfig

from .errors import FileNotFound, GroundingFailed, now_iso
from .storage import Workspace, write_json_atomic
from .style import INHERITED_DEFAULTS, INHERITED_PROPERTIES, build_style, size_to_em
from .style import _parse_declarations as parse_style_declarations
from .textmatch import (
    PageDims,
    Word,
    core_token,
    find_fuzzy_spans,
    find_spans,
    find_spans_lenient,
    fuzzy_norm,
    line_groups,
    load_page_words,
    similarity,
    words_form_one_cell,
)

DG_NAMESPACE = "http://dgml.io/ns/dg#"

# An element's boxes are emitted only when at least this share of its
# subtree's tokens grounded — a box derived from a stray word or two
# would point somewhere misleading rather than at the element.
_EMIT_MIN_FRACTION = 0.5

# Gap recovery accepts an (unmatched run ↔ OCR gap) pairing only above
# this weighted similarity. Conservative on purpose: the weighted
# distance makes digit differences expensive, so dates/amounts can't
# silently ground to a lookalike, but OCR letter slips and rejoined
# hyphenations clear the bar.
_RECOVERY_SIM_THRESHOLD = 0.75

# An OCR gap more than ~2x the unmatched run (plus slack for dropped
# list markers and footers) is not "the same text, mangled" — skip.


def _recovery_max_gap(run_len: int) -> int:
    return 2 * run_len + 6


# Window-size cap (in DP cells) for direct SequenceMatcher alignment.
# Above it we recurse with within-window anchors; windows that stay
# above it with no anchors at all are abandoned (their tokens simply
# stay unmatched — never mis-grounded).
_SM_MAX_CELLS = 4_000_000

# How many pages on each side of the expected position the duplicate-
# rescue span search may look at.
_RESCUE_PAGE_SLACK = 1

# Extra reading-order words searched on each side of a punctuation-only
# node's context window. Table rows serialize in slightly different
# cell orders in OCR (cell-top sort jitter) than in the XML, so a '?'
# often sits just outside the strict context interval. Six words stays
# within the row's neighborhood; the nearest-match rule and claim
# tracking keep adjacent rows' identical cells from swapping.
_PUNCT_WINDOW_SLACK = 6

# The document-order fallback window (used when a punctuation node has
# no grounded siblings) is refused above this size — a giant window
# means the surrounding region never grounded, and placing a '?'
# against no anchor at all would be a guess, not a match.
_PUNCT_MAX_FALLBACK_WINDOW = 80

# Reading-order slack around a partial segment's own matched words when
# completing it along its own visual line — wide enough to skip an
# intervening wrapped label line, small enough to stay in the block.
_SAME_LINE_SLACK = 24

# Subsequence assembly is refused above this window size (in reading-
# order words). The pass exists for table-region interleaves — a row or
# table spans at most a few hundred words — and the scan is quadratic
# in the window, so a flat parent whose "sibling window" is half the
# document must not be searched (same posture as _SM_MAX_CELLS).
_ASSEMBLE_MAX_WINDOW = 400

# Out-of-band relocation only considers cell-sized segments; paragraphs
# are never a stranded table-cell duplicate (same posture as
# _PRUNE_MAX_TOKENS).
_RELOCATE_MAX_TOKENS = 8

# Rescue-assembly guards: only multi-token segments with enough
# character mass to be distinctive on a whole page (a two-letter pair
# could frankenmatch), and only pages small enough for the quadratic
# subsequence scan.
_RESCUE_ASSEMBLE_MIN_CHARS = 6
_RESCUE_ASSEMBLE_MAX_PAGE_WORDS = 2000


# ---- Data shapes -----------------------------------------------------------


@dataclass
class _TextSeg:
    """One XML text node: an element's ``.text`` or a child's ``.tail``.

    ``owner`` is the element whose *content* the text is — for a tail,
    that's the parent of the element carrying the tail attribute, which
    is exactly the mixed-content element the user reads the text inside.
    """

    owner: Any  # lxml element
    raw: str
    token_start: int = 0  # index of first token in the XML token stream
    n_tokens: int = 0
    matched_tokens: int = 0
    matched_words: list[tuple[int, Word]] = field(default_factory=list)  # (page, word)
    # Set when an assembly pass placed the words deliberately (exact
    # cores, coherence-checked) — outlier pruning must not second-guess
    # a wide-gap layout that assembly accepted on purpose.
    pinned: bool = False


@dataclass(frozen=True)
class _OTok:
    """One OCR word in the alignment stream (empty-core words excluded)."""

    page: int
    word: Word


@dataclass(frozen=True)
class GroundingResult:
    output_path: Path
    # None when the stats sidecar wasn't written (write_stats=False); the stats
    # dict is still returned in-memory regardless.
    stats_path: Path | None
    stats: dict[str, Any]


# ---- XML side --------------------------------------------------------------


_BARE_AMP_RE = re.compile(r"&(?!(?:#\d+|#x[\da-fA-F]+|[A-Za-z]\w*);)")


def _parse_xml(xml_path: Path) -> Any:
    """Parse to an lxml tree, strict first, recover on failure (same
    posture as the coverage module — generated XML occasionally carries
    a bare ``&``)."""
    from lxml import etree  # type: ignore[import-untyped]

    raw = xml_path.read_text(encoding="utf-8")
    cleaned = _BARE_AMP_RE.sub("&amp;", raw)
    try:
        return etree.fromstring(cleaned.encode("utf-8"))
    except etree.XMLSyntaxError:
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        root = etree.fromstring(cleaned.encode("utf-8"), parser=parser)
        if root is None:
            raise GroundingFailed(f"could not parse XML at {xml_path}") from None
        return root


def _collect_segments(root: Any) -> list[_TextSeg]:
    """Every non-whitespace text node in document order, with its owning
    element. Comments / processing instructions are skipped (their tag
    is not a string in lxml)."""
    segs: list[_TextSeg] = []

    def walk(el: Any) -> None:
        if el.text and el.text.strip():
            segs.append(_TextSeg(owner=el, raw=el.text))
        for child in el:
            if isinstance(child.tag, str):
                walk(child)
            if child.tail and child.tail.strip():
                segs.append(_TextSeg(owner=el, raw=child.tail))

    walk(root)
    return segs


# ---- OCR side --------------------------------------------------------------


def _list_pages(workspace: Workspace, file_id: str) -> list[int]:
    text_dir = workspace.file_text_dir(file_id)
    pages = sorted(
        int(m.group(1))
        for p in text_dir.glob("page_*.json")
        if (m := re.fullmatch(r"page_(\d+)", p.stem)) is not None
    )
    if not pages:
        raise FileNotFound(
            f"no page_text for file '{file_id}' (expected page_*.json under {text_dir}); "
            "was the file added with --text-mode digital, ocr, or hybrid?"
        )
    return pages


# ---- Anchoring + alignment -------------------------------------------------


def _lis_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Longest increasing subsequence of ``(x, y)`` pairs, strictly
    increasing in both coordinates. Input must be sorted by ``x``
    (strictly — unique x positions). O(n log n) patience sorting."""
    if not pairs:
        return []
    tails: list[int] = []  # y of smallest tail for each LIS length
    back: list[int] = [0] * len(pairs)
    idx_at_len: list[int] = []
    for i, (_x, y) in enumerate(pairs):
        pos = bisect_right(tails, y - 1)  # strictly increasing in y
        if pos == len(tails):
            tails.append(y)
            idx_at_len.append(i)
        else:
            tails[pos] = y
            idx_at_len[pos] = i
        back[i] = idx_at_len[pos - 1] if pos > 0 else -1
    out: list[tuple[int, int]] = []
    i = idx_at_len[len(tails) - 1]
    while i >= 0:
        out.append(pairs[i])
        i = back[i]
    out.reverse()
    return out


def _unique_anchors(
    xcores: list[str],
    ocores: list[str],
    xlo: int,
    xhi: int,
    olo: int,
    ohi: int,
) -> list[tuple[int, int, int]]:
    """Anchor candidates within the window: ``(xpos, opos, n)`` n-grams
    occurring exactly once in each stream slice. Longest n first; the
    first n that yields any anchors wins (longer n-grams are rarer and
    therefore safer sync points)."""
    for n in (3, 2, 1):
        if xhi - xlo < n or ohi - olo < n:
            continue
        xpos: dict[tuple[str, ...], int] = {}
        xdup: set[tuple[str, ...]] = set()
        for i in range(xlo, xhi - n + 1):
            g = tuple(xcores[i : i + n])
            if g in xdup:
                continue
            if g in xpos:
                del xpos[g]
                xdup.add(g)
            else:
                xpos[g] = i
        if not xpos:
            continue
        opos: dict[tuple[str, ...], int] = {}
        odup: set[tuple[str, ...]] = set()
        for j in range(olo, ohi - n + 1):
            g = tuple(ocores[j : j + n])
            if g in odup or g not in xpos:
                continue
            if g in opos:
                del opos[g]
                odup.add(g)
            else:
                opos[g] = j
        anchors = sorted((xpos[g], opos[g], n) for g in opos)
        if anchors:
            return anchors
    return []


# Windows whose shorter side is at or below this many tokens align
# directly with SequenceMatcher; larger windows anchor-and-recurse
# first. Anchoring at EVERY level (not just above the cell cap) is what
# keeps repetitive regions honest: a feature-comparison table is wall-
# to-wall Yes/No, and one whole-region LCS will happily slide a run of
# cells across a row boundary — the unique row labels only pin the rows
# if they're actually used as anchors.
_SM_DIRECT_MAX_TOKENS = 24


def _sm_align(
    xcores: list[str],
    ocores: list[str],
    xlo: int,
    xhi: int,
    olo: int,
    ohi: int,
    pairs: dict[int, int],
) -> None:
    sm = difflib.SequenceMatcher(None, xcores[xlo:xhi], ocores[olo:ohi], autojunk=False)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            pairs[xlo + blk.a + k] = olo + blk.b + k


def _align_window(
    xcores: list[str],
    ocores: list[str],
    xlo: int,
    xhi: int,
    olo: int,
    ohi: int,
    pairs: dict[int, int],
    stats: dict[str, int],
    depth: int = 0,
) -> None:
    """Patience-diff recursion: anchor, split, and SequenceMatcher the
    small windows. Matches land in ``pairs`` (xml token idx → ocr token
    idx)."""
    if xlo >= xhi or olo >= ohi:
        return
    nx, no = xhi - xlo, ohi - olo
    if nx <= _SM_DIRECT_MAX_TOKENS or no <= _SM_DIRECT_MAX_TOKENS or depth > 24:
        if nx * no > _SM_MAX_CELLS:
            stats["windows_abandoned"] += 1
            return
        _sm_align(xcores, ocores, xlo, xhi, olo, ohi, pairs)
        return
    anchors = _unique_anchors(xcores, ocores, xlo, xhi, olo, ohi)
    stats["anchor_candidates"] += len(anchors)
    kept = _lis_pairs([(a, o) for a, o, _n in anchors])
    kept_set = set(kept)
    chain = [(a, o, n) for a, o, n in anchors if (a, o) in kept_set]
    stats["anchors_used"] += len(chain)
    if not chain:
        # No sync points in the window at all — fall back to one direct
        # diff if it fits, else give up (never guess).
        if nx * no > _SM_MAX_CELLS:
            stats["windows_abandoned"] += 1
            return
        _sm_align(xcores, ocores, xlo, xhi, olo, ohi, pairs)
        return
    px, po = xlo, olo
    for ax, ao, n in chain:
        if ax < px or ao < po:  # overlap with the previous anchor's tail
            continue
        _align_window(xcores, ocores, px, ax, po, ao, pairs, stats, depth + 1)
        for k in range(n):
            pairs[ax + k] = ao + k
        px, po = ax + n, ao + n
    _align_window(xcores, ocores, px, xhi, po, ohi, pairs, stats, depth + 1)


# ---- The grounding pipeline ------------------------------------------------


def ground_dgml_xml(
    workspace: Workspace,
    file_id: str,
    xml_path: Path,
    output_path: Path | None = None,
    *,
    force: bool = False,
    write_stats: bool = True,
    debug: bool = False,
) -> GroundingResult:
    """Annotate the DGML XML at ``xml_path`` with ``dg:origin`` boxes,
    grounding against ``file_id``'s page OCR, and write the result.

    ``output_path`` defaults to a ``<stem>.dgml.grounded.xml`` sibling;
    pass ``output_path=xml_path`` (with ``force=True``) to ground in
    place — what ``dgml docset generate`` does so the canonical
    ``<stem>.dgml.xml`` carries the boxes. When ``write_stats`` is set
    (the default), a ``<stem>.dgml.grounding_stats.json`` sidecar with
    match-rate telemetry is written next to the output; the returned
    ``GroundingResult.stats`` is always populated regardless.

    With ``write_stats=False`` the ``<stem>.dgml.grounding_stats.json``
    sidecar is not written (the CLI suppresses it unless ``--debug``); the
    grounded XML is still produced and the stats dict is still returned
    in-memory, but ``GroundingResult.stats_path`` is ``None``.

    ``debug`` gates ``usage.jsonl`` recording for the LLM-backed image-based
    ``dg:style`` pass (OCR files with a ``style`` config) — like every other
    LLM path, no ``--debug`` means no usage rows. Deterministic grounding does
    no LLM work, so this is a no-op otherwise.

    Raises :class:`FileNotFound` if the workspace has no ``page_text``
    for the file, :class:`GroundingFailed` if the XML cannot be parsed,
    and :class:`GroundingFailed` if the output exists and ``force`` is
    not set (callers usually check first and report "skipped").
    """
    from lxml import etree

    out_path = output_path or grounded_output_path(xml_path)
    stats_path = out_path.with_name(_stats_name(xml_path))
    if out_path.exists() and not force:
        raise GroundingFailed(f"{out_path} already exists (use --force to regenerate)")

    root = _parse_xml(xml_path)
    # Grounding owns dg:style — clear any from a prior run so re-grounding is
    # idempotent (an element that no longer has observable style must lose the
    # attribute, not keep a stale value).
    _clear_attr(root, _dg_attr_name(root, "style"))
    segs = _collect_segments(root)

    # OCR stream: pages concatenated in order, cell-aware reading order
    # within each page (load_page_words reorders), empty-core words
    # excluded from the *alignment* stream but kept for box building.
    pages = _list_pages(workspace, file_id)
    page_words: dict[int, list[Word]] = {}
    page_dims: dict[int, PageDims] = {}
    page_baselines: dict[int, float] = {}
    otoks: list[_OTok] = []
    for page in pages:
        words, dims = load_page_words(workspace, file_id, page)
        page_words[page] = words
        page_dims[page] = dims
        baseline = _page_baseline(words)
        if baseline:
            page_baselines[page] = baseline
        for w in words:
            if core_token(w.text):
                otoks.append(_OTok(page=page, word=w))
    ocores = [core_token(t.word.text) for t in otoks]

    # XML stream: whitespace-split tokens of every segment, empty-core
    # tokens excluded (pure punctuation grounds nothing by itself).
    xtoks_seg: list[int] = []  # token idx -> segment idx
    xtoks_raw: list[str] = []
    for si, seg in enumerate(segs):
        seg.token_start = len(xtoks_raw)
        for tok in seg.raw.split():
            if core_token(tok):
                xtoks_seg.append(si)
                xtoks_raw.append(tok)
        seg.n_tokens = len(xtoks_raw) - seg.token_start
    xcores = [core_token(t) for t in xtoks_raw]

    align_stats = {"anchor_candidates": 0, "anchors_used": 0, "windows_abandoned": 0}
    pairs: dict[int, int] = {}
    _align_window(xcores, ocores, 0, len(xcores), 0, len(ocores), pairs, align_stats)

    matched_tokens = _commit_pairs(segs, xtoks_seg, otoks, pairs)
    recovered = _recover_gaps(segs, xtoks_seg, xtoks_raw, otoks, pairs)
    rescued = _rescue_unmatched_segs(segs, xtoks_seg, otoks, pairs, page_words, page_dims)
    relocated = _arbitrate_sibling_overlaps(segs, page_words)
    relocated += _relocate_column_outliers(segs, page_words, page_dims)
    punct_grounded, shape_tokens, interleaved_tokens, band_relocations = _ground_row_context_segs(
        segs, page_words
    )
    assembled_tokens = _assemble_stranded_segs(segs, otoks, pairs, page_words)
    # Absorb BEFORE pruning: a cell's own unmatched punctuation ("Cash
    # & Liquid" — the '&' is invisible to alignment) sits between its
    # matched words, and pruning would read that occupied space as an
    # outlier gap and drop the word on the far side of it.
    absorbed_fragments = _absorb_adjacent_fragments(segs, page_words)
    pruned_words = _prune_outlier_words(segs, page_dims)

    # dg:style policy. The deterministic path derives style from digital glyph
    # facts; OCR files carry none. An OCR file's dg:style therefore comes solely
    # from the opt-in image-based path below, and only when the workspace
    # configures a `style` section — so an OCR file with no style config gets no
    # dg:style at all (matching storage-layout.md / cli-reference.md / SKILL.md).
    # A digital/hybrid file always may. load_style_config validates the section
    # on every grounding run (raising StyleConfigInvalid), OCR or not.
    from .style_config import load_style_config

    is_ocr = _is_ocr_file(workspace, file_id)
    style_config = load_style_config(workspace)
    emit_style = not is_ocr or style_config is not None
    annotated, containers_annotated = _annotate_tree(
        root, segs, page_dims, page_baselines, emit_style=emit_style
    )

    # For opted-in OCR files, fill dg:style from the page images via an LLM
    # (best-effort). A no-op for digital/hybrid files and unconfigured workspaces.
    _maybe_style_from_image(
        workspace, file_id, root, config=style_config, is_ocr=is_ocr, debug=debug
    )

    # Drop dg:style declarations a child merely inherits from an ancestor, so an
    # inheriting property (color, font-*, …) lands only where it first appears —
    # the most specific element that introduces it — not restated down the tree.
    _suppress_inherited_style(root, _dg_attr_name(root, "style"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(etree.tostring(root, encoding="utf-8", xml_declaration=True))

    total_tokens = len(xtoks_raw)
    # Punctuation-only segments have n_tokens == 0; they count as
    # grounded once the positional pass attached a word.
    grounded_segs = sum(1 for s in segs if s.matched_words and s.matched_tokens == s.n_tokens)
    partial_segs = sum(1 for s in segs if 0 < s.matched_tokens < s.n_tokens)
    ungrounded_segs = sum(1 for s in segs if s.n_tokens and not s.matched_tokens)
    total_matched = sum(s.matched_tokens for s in segs)
    stats: dict[str, Any] = {
        "completed_at": now_iso(),
        "source": str(xml_path),
        "output": str(out_path),
        "file_id": file_id,
        "pages": len(pages),
        "ocr_words": sum(len(v) for v in page_words.values()),
        "xml_tokens": total_tokens,
        "matched_tokens": total_matched,
        "matched_token_pct": round(total_matched / total_tokens * 100, 1) if total_tokens else 0.0,
        "aligned_tokens": matched_tokens,
        "recovered_tokens": recovered,
        "rescued_tokens": rescued,
        "punct_grounded_nodes": punct_grounded,
        # Tokens grounded by in-order subsequence assembly — the cell's
        # words interleave with a neighboring cell's in the OCR stream
        # (adjacent multi-line cells merged by the reading-order cell
        # builder), so no contiguous span exists.
        "interleaved_cell_tokens": interleaved_tokens,
        # Tokens grounded by digit-masked row-context matching — the
        # generator transcribed a digit differently than OCR read it.
        # Worth auditing: the BOX is right, the XML text may not be.
        "shape_matched_tokens": shape_tokens,
        # Tokens grounded by the last-resort page-wide assembly — a
        # stranded interleaved cell every windowed pass pointed away
        # from (see _assemble_stranded_segs).
        "assembled_tokens": assembled_tokens,
        # Segments moved off a sibling-contained duplicate onto the
        # unclaimed identical copy (charge-code vs memo-prefix).
        "sibling_relocations": relocated,
        # Fully-matched cells moved off a duplicate outside their row
        # band onto the unclaimed copy assembled inside it (a data-row
        # amount stranded on the totals row's identical amount).
        "band_relocations": band_relocations,
        # Box-cleanup telemetry: stray words dropped from cell-sized
        # segments, and punctuation fragments pulled into their boxes.
        "pruned_words": pruned_words,
        "absorbed_fragments": absorbed_fragments,
        "text_nodes": {
            "total": len(segs),
            "grounded": grounded_segs,
            "partial": partial_segs,
            "ungrounded": ungrounded_segs,
            # Punctuation-only nodes the positional pass couldn't place
            # (plus glue text like the ", " between inline elements).
            "no_content": len(segs) - grounded_segs - partial_segs - ungrounded_segs,
        },
        "elements_annotated": annotated,
        # Of those, pure containers (no direct text nodes) annotated
        # with per-page union boxes derived from their subtrees.
        "containers_annotated": containers_annotated,
        "anchors": align_stats,
        "top_ungrounded": _top_ungrounded(root, segs),
    }
    if write_stats:
        write_json_atomic(stats_path, stats)
    return GroundingResult(
        output_path=out_path,
        stats_path=stats_path if write_stats else None,
        stats=stats,
    )


def grounded_output_path(xml_path: Path) -> Path:
    """``X.dgml.xml`` → ``X.dgml.grounded.xml`` (suffix-aware so a bare
    ``X.xml`` still gets a sensible ``X.grounded.xml``)."""
    name = xml_path.name
    if name.endswith(".xml"):
        return xml_path.with_name(name[: -len(".xml")] + ".grounded.xml")
    return xml_path.with_name(name + ".grounded.xml")


def _stats_name(xml_path: Path) -> str:
    name = xml_path.name
    if name.endswith(".xml"):
        return name[: -len(".xml")] + ".grounding_stats.json"
    return name + ".grounding_stats.json"


# ---- Match commitment, gap recovery, duplicate rescue ----------------------


def _commit_pairs(
    segs: list[_TextSeg],
    xtoks_seg: list[int],
    otoks: list[_OTok],
    pairs: dict[int, int],
) -> int:
    """Write alignment pairs into their segments' matched lists."""
    for xi, oi in pairs.items():
        seg = segs[xtoks_seg[xi]]
        ot = otoks[oi]
        seg.matched_tokens += 1
        seg.matched_words.append((ot.page, ot.word))
    return len(pairs)


def _recover_gaps(
    segs: list[_TextSeg],
    xtoks_seg: list[int],
    xtoks_raw: list[str],
    otoks: list[_OTok],
    pairs: dict[int, int],
) -> int:
    """Fuzzy-match unmatched token runs against the OCR gap their
    matched neighbors bracket.

    The aligner only pairs tokens whose cores are *equal*; an OCR letter
    slip ("Aqreement"), a rejoined hyphenation ("Agree- ment"), or a
    split word leaves a run of XML tokens unmatched even though the
    right words sit exactly in the corresponding OCR gap. The weighted
    similarity confirms the pairing without risking numeric content
    (digit edits are expensive by design)."""
    if not pairs:
        return 0
    sorted_x = sorted(pairs)
    recovered = 0
    n = len(xtoks_raw)
    i = 0
    while i < n:
        if i in pairs:
            i += 1
            continue
        run_start = i
        while i < n and i not in pairs:
            i += 1
        run_end = i  # exclusive
        # Bracketing matched neighbors (global, may sit in other segments).
        k = bisect_right(sorted_x, run_start) - 1
        if k < 0:
            continue  # leading run — no left bracket
        if sorted_x[-1] < run_end:
            continue  # trailing run — no right bracket
        prev_x = sorted_x[k]
        next_x = sorted_x[bisect_right(sorted_x, run_end - 1)]
        gap_lo = pairs[prev_x] + 1
        gap_hi = pairs[next_x]  # exclusive
        run_len = run_end - run_start
        if gap_hi <= gap_lo or (gap_hi - gap_lo) > _recovery_max_gap(run_len):
            continue
        run_text = " ".join(xtoks_raw[run_start:run_end])
        gap_text = " ".join(otoks[j].word.text for j in range(gap_lo, gap_hi))
        if similarity(run_text, gap_text) < _RECOVERY_SIM_THRESHOLD:
            continue
        # Accept: every token in the run counts as matched. A run can
        # span segment boundaries (adjacent table cells whose words all
        # slipped), so the gap words must be DISTRIBUTED across the
        # run's tokens — giving every segment the whole gap smears one
        # cell's box across its neighbors' columns. Exact word↔token
        # pairing is unknowable here (the words didn't core-match, and
        # OCR may have split or merged them), so each gap word goes to
        # the run token whose cumulative character interval contains the
        # word's midpoint — order-preserving and split/merge tolerant.
        # Lengths are measured on ``core_token`` for BOTH sides: gap
        # words are alignment-stream tokens (alphanumeric cores by
        # construction), while run tokens may carry punctuation the
        # stream never sees — fuzzy lengths would skew the boundary
        # (observed: OCR shattering "($25.00) $1,063.97" into char
        # fragments put the "1" one cell early).
        run_tokens = list(range(run_start, run_end))
        tok_lens = [max(1, len(core_token(xtoks_raw[x]))) for x in run_tokens]
        gap_words = [(otoks[j].page, otoks[j].word) for j in range(gap_lo, gap_hi)]
        word_lens = [max(1, len(core_token(w.text))) for _p, w in gap_words]
        total_t = sum(tok_lens)
        total_w = sum(word_lens)
        tok_bounds: list[tuple[float, float]] = []
        acc = 0.0
        for length in tok_lens:
            tok_bounds.append((acc, acc + length))
            acc += length
        acc_w = 0.0
        for (page, w), length in zip(gap_words, word_lens, strict=True):
            mid = (acc_w + length / 2) / total_w * total_t
            acc_w += length
            ti = len(run_tokens) - 1
            for t, (b_lo, b_hi) in enumerate(tok_bounds):
                if b_lo <= mid < b_hi:
                    ti = t
                    break
            segs[xtoks_seg[run_tokens[ti]]].matched_words.append((page, w))
        for x in run_tokens:
            segs[xtoks_seg[x]].matched_tokens += 1
        recovered += run_len
    return recovered


def _reading_index(
    page_words: dict[int, list[Word]],
) -> tuple[dict[tuple[int, int], int], dict[int, int], list[tuple[int, Word]]]:
    """Global reading-order positions over the full per-page word lists:
    each word's index within its (reordered) page, cumulative page
    offsets, and the flattened ``(page, word)`` stream. ``Word.idx`` is
    the original OCR index — unique per page — so it keys the maps."""
    pos_in_page: dict[tuple[int, int], int] = {}
    page_offset: dict[int, int] = {}
    flat: list[tuple[int, Word]] = []
    cum = 0
    for page in sorted(page_words):
        page_offset[page] = cum
        for i, w in enumerate(page_words[page]):
            pos_in_page[(page, w.idx)] = i
            flat.append((page, w))
        cum += len(page_words[page])
    return pos_in_page, page_offset, flat


def _claimed_words(segs: list[_TextSeg]) -> set[tuple[int, int]]:
    return {(page, w.idx) for seg in segs for page, w in seg.matched_words}


# Cells (short segments) should occupy one place on the page; box-word
# pruning keeps only the dominant x-cluster. The gap threshold sits
# between intra-cell word spacing (~1% of page width) and column
# gutters (~3% and up in dense ledgers).
_PRUNE_MAX_TOKENS = 4
_PRUNE_X_GAP_PCT = 2.5

# A punctuation fragment is absorbed into an adjacent box when its
# horizontal gap to a matched word is within this share of word height.
_ABSORB_GAP_FACTOR = 0.8


def _content_count(words: list[Word]) -> int:
    """Words with an alphanumeric core — punctuation fragments excluded."""
    return sum(1 for w in words if core_token(w.text))


def _prune_outlier_words(segs: list[_TextSeg], page_dims: dict[int, PageDims]) -> int:
    """Drop stray box words from short (cell-sized) segments.

    Gap recovery distributes shattered-OCR words by character share,
    and a fragment can land one cell over; dot leaders at the page
    margin can ride along in a recovery gap. A table cell occupies ONE
    place *per line*, so when the matched words on a visual line form a
    dominant x-cluster plus outliers, the outliers are attribution
    noise — keep the cluster, tighten the box. Clustering is per line
    on purpose: a short inline span that wraps legitimately jumps from
    line-end to line-start ("Class-A Office ⏎ and Laboratory"), and its
    continuation words are layout, not noise. Long segments
    (paragraphs) span the page legitimately and are left alone."""
    pruned = 0
    for seg in segs:
        if not seg.n_tokens or seg.n_tokens > _PRUNE_MAX_TOKENS or seg.pinned:
            continue
        if len(seg.matched_words) < 2:
            continue
        by_page: dict[int, list[Word]] = {}
        for p, w in seg.matched_words:
            by_page.setdefault(p, []).append(w)
        keep: set[tuple[int, int]] = set()
        for p, page_ws in by_page.items():
            for line in line_groups(page_ws):
                line_ws = sorted(line, key=lambda w: w.left)
                clusters: list[list[Word]] = [[line_ws[0]]]
                for w in line_ws[1:]:
                    gap_pct = (w.left - clusters[-1][-1].right) / page_dims[p].width * 100
                    if gap_pct > _PRUNE_X_GAP_PCT:
                        clusters.append([w])
                    else:
                        clusters[-1].append(w)
                # Dominance is judged on CONTENT words only: absorbed
                # punctuation fragments ride along in clusters (they
                # bridge gaps, which is why absorption runs first) but
                # must not pad an outlier cluster into a tie.
                best = max(clusters, key=_content_count)
                if len(clusters) == 1 or _content_count(best) * 2 <= _content_count(line_ws):
                    # single cluster, or no dominant one — leave the line alone
                    keep.update((p, w.idx) for w in line_ws)
                else:
                    keep.update((p, w.idx) for w in best)
        kept = [(p, w) for p, w in seg.matched_words if (p, w.idx) in keep]
        pruned += len(seg.matched_words) - len(kept)
        seg.matched_words = kept
    return pruned


def _absorb_adjacent_fragments(
    segs: list[_TextSeg],
    page_words: dict[int, list[Word]],
) -> int:
    """Pull unclaimed punctuation-only fragments into the box of the
    word they visually belong to.

    Shattering OCR splits ``($1,272.66)`` into fragments; the alignment
    stream carries only the alphanumeric cores, so the ``($``/``$``/
    ``)`` pieces are never matched and the rendered box clips the
    currency symbol. Any unclaimed fragment with no alphanumeric
    content that is reading-order adjacent AND same-line touching-
    distance from a matched word joins that word's segment. Numerals
    never move (they carry content), and distance gating keeps a cell's
    closing paren from absorbing the next column's opening one."""
    pos_in_page, page_offset, flat_words = _reading_index(page_words)
    claimed = _claimed_words(segs)
    absorbed = 0
    for seg in segs:
        if not seg.matched_words:
            continue
        # A fragment must be plausible for THIS segment's text: 'Code'
        # ("Utilities") never absorbs the charge's '$', and the charge
        # ("$38.70") never absorbs the balance's '(' — whichever is
        # geometrically closer.
        seg_chars = set(fuzzy_norm(seg.raw))
        added = True
        while added:
            added = False
            for p, w in list(seg.matched_words):
                gp = page_offset[p] + pos_in_page[(p, w.idx)]
                for ngp in (gp - 1, gp + 1):
                    if not 0 <= ngp < len(flat_words):
                        continue
                    np_, nw = flat_words[ngp]
                    if np_ != p or (np_, nw.idx) in claimed:
                        continue
                    if any(ch.isalnum() for ch in nw.text):
                        continue
                    if not set(fuzzy_norm(nw.text)) <= seg_chars:
                        continue
                    y_overlap = min(w.bottom, nw.bottom) - max(w.top, nw.top)
                    if y_overlap < 0.5 * min(w.height, nw.height):
                        continue
                    gap = max(w.left, nw.left) - min(w.right, nw.right)
                    if gap > _ABSORB_GAP_FACTOR * max(w.height, nw.height):
                        continue
                    seg.matched_words.append((np_, nw))
                    claimed.add((np_, nw.idx))
                    absorbed += 1
                    added = True
    return absorbed


def _arbitrate_sibling_overlaps(
    segs: list[_TextSeg],
    page_words: dict[int, list[Word]],
) -> int:
    """Relocate a fully-matched segment whose words sit inside a
    *sibling's* matched span when an unclaimed exact copy of its text
    exists in the row window.

    Sibling cells occupy disjoint page regions, so containment is a red
    flag: a charge-code cell whose text reappears verbatim as the memo
    prefix can get aligned onto the memo copy (both are textually valid
    matches), leaving its own cell's words unclaimed. The unclaimed
    identical span is the arbiter — if the page really shows the text
    twice and one copy is free, the contained segment moves there.
    Correctly-placed duplicates don't trip this (their regions are
    disjoint), and absent an unclaimed copy nothing moves."""
    pos_in_page, page_offset, flat_words = _reading_index(page_words)
    claimed = _claimed_words(segs)

    def gpos(p: int, w: Word) -> int:
        return page_offset[p] + pos_in_page[(p, w.idx)]

    by_parent: dict[Any, list[_TextSeg]] = {}
    for s in segs:
        parent = s.owner.getparent()
        if parent is not None:
            by_parent.setdefault(parent, []).append(s)

    relocated = 0
    for group in by_parent.values():
        if len(group) < 2:
            continue
        ranges: list[tuple[_TextSeg, int, int]] = []
        for s in group:
            if not s.matched_words:
                continue
            ps = [gpos(p, w) for p, w in s.matched_words]
            ranges.append((s, min(ps), max(ps)))
        if len(ranges) < 2:
            continue
        win_lo = max(0, min(lo for _s, lo, _hi in ranges) - _PUNCT_WINDOW_SLACK)
        win_hi = min(len(flat_words), max(hi for _s, _lo, hi in ranges) + 1 + _PUNCT_WINDOW_SLACK)
        for s, s_lo, s_hi in ranges:
            if not s.n_tokens or s.matched_tokens < s.n_tokens:
                continue  # only confidently-matched segments move
            contained = any(
                o is not s and o_lo <= s_lo and s_hi <= o_hi and (o_hi - o_lo) > (s_hi - s_lo)
                for o, o_lo, o_hi in ranges
            )
            if not contained:
                continue
            target = fuzzy_norm(s.raw)
            if not target:
                continue
            span = _find_unclaimed_span(flat_words, win_lo, win_hi, target, claimed)
            if span is None:
                continue
            old = {(p, w.idx) for p, w in s.matched_words}
            claimed.difference_update(old)
            s.matched_words = list(span)
            claimed.update((p, w.idx) for p, w in span)
            relocated += 1
    return relocated


def _find_unclaimed_span(
    flat_words: list[tuple[int, Word]],
    lo: int,
    hi: int,
    target: str,
    claimed: set[tuple[int, int]],
    accept: Callable[[list[tuple[int, Word]]], bool] | None = None,
) -> list[tuple[int, Word]] | None:
    """Leftmost contiguous span in ``flat_words[lo:hi]`` whose joined
    ``fuzzy_norm`` equals ``target`` with every word unclaimed.
    Empty-norm words (stray punctuation) are transparent. ``accept``
    optionally filters complete spans (e.g. by column position)."""
    for start in range(lo, hi):
        if not fuzzy_norm(flat_words[start][1].text):
            continue
        acc = ""
        span: list[tuple[int, Word]] = []
        for gp in range(start, hi):
            page, w = flat_words[gp]
            fn = fuzzy_norm(w.text)
            if not fn:
                continue
            acc += fn
            if not target.startswith(acc):
                break
            if (page, w.idx) in claimed:
                break
            span.append((page, w))
            if len(acc) == len(target):
                if accept is None or accept(span):
                    return span
                break
    return None


# A cell whose box left-edge deviates from its column's median by more
# than this many percent of page width is a column outlier; relocation
# targets must land back within the band.
_COLUMN_BAND_PCT = 4.0
# Columns vote only when enough same-tag sibling-row cells agree.
_COLUMN_MIN_MEMBERS = 4


def _relocate_column_outliers(
    segs: list[_TextSeg],
    page_words: dict[int, list[Word]],
    page_dims: dict[int, PageDims],
) -> int:
    """Cross-row column consensus: in a table, same-tag cells across
    sibling rows share an x band. A fully-matched cell sitting far from
    its column's median is suspicious even when the match is textually
    perfect — a charge-code whose text reappears verbatim as the memo
    prefix can be aligned onto the memo copy while its own cell sits
    unclaimed (stream order can't tell the copies apart; the column
    can). Relocate when an unclaimed exact copy of the cell's text
    exists in the row window *within the column band* — otherwise leave
    it alone (no copy, no move)."""
    pos_in_page, page_offset, flat_words = _reading_index(page_words)
    claimed = _claimed_words(segs)

    def gpos(p: int, w: Word) -> int:
        return page_offset[p] + pos_in_page[(p, w.idx)]

    def xleft(s: _TextSeg) -> float:
        return min(w.left / page_dims[p].width * 100 for p, w in s.matched_words)

    # Columns: cells grouped by (table element, cell tag), where a cell
    # is an element whose parent (the row) has a parent (the table).
    columns: dict[tuple[Any, str], list[_TextSeg]] = {}
    row_pool: dict[Any, list[int]] = {}
    for s in segs:
        if not s.matched_words:
            continue
        owner = s.owner
        if not isinstance(owner.tag, str):
            continue
        row = owner.getparent()
        if row is None:
            continue
        row_pool.setdefault(row, []).extend(gpos(p, w) for p, w in s.matched_words)
        table = row.getparent()
        if table is not None and s.n_tokens:
            columns.setdefault((table, owner.tag), []).append(s)

    relocated = 0
    for members in columns.values():
        if len(members) < _COLUMN_MIN_MEMBERS:
            continue
        xs = sorted(xleft(s) for s in members)
        median_x = xs[len(xs) // 2]
        for s in members:
            if s.matched_tokens < s.n_tokens:
                continue
            if abs(xleft(s) - median_x) <= _COLUMN_BAND_PCT:
                continue
            pool = row_pool.get(s.owner.getparent(), [])
            if not pool:
                continue
            lo = max(0, min(pool) - _PUNCT_WINDOW_SLACK)
            hi = min(len(flat_words), max(pool) + 1 + _PUNCT_WINDOW_SLACK)
            target = fuzzy_norm(s.raw)
            if not target:
                continue

            def in_band(span: list[tuple[int, Word]], median: float = median_x) -> bool:
                x = min(w.left / page_dims[p].width * 100 for p, w in span)
                return abs(x - median) <= _COLUMN_BAND_PCT

            span = _find_unclaimed_span(flat_words, lo, hi, target, claimed, accept=in_band)
            if span is None:
                # Exact copy not found — the cell's own words may be
                # OCR-mangled ("BIII" for "Bill"), which is exactly why
                # alignment preferred the duplicate elsewhere. Run the
                # fuzzy span search (letter-slip tolerant, digit-
                # protecting, unambiguous-only) over the UNCLAIMED
                # window words only: the claimed duplicate isn't a
                # relocation candidate, and leaving it in would make
                # the search look ambiguous and refuse. Filtering can
                # splice distant words together, so the mapped-back
                # span must be contiguous (small gaps = transparent
                # punctuation fragments only).
                positions = [
                    g
                    for g in range(lo, hi)
                    if (flat_words[g][0], flat_words[g][1].idx) not in claimed
                ]
                window_words = [flat_words[g][1] for g in positions]
                for fs, fe in find_fuzzy_spans(s.raw, window_words):
                    gps = positions[fs:fe]
                    if max(gps) - min(gps) > (fe - fs) + 2:
                        continue  # spliced across removed words — reject
                    cand = [flat_words[g] for g in gps]
                    if in_band(cand):
                        span = cand
                    break
            if span is None:
                continue
            old = {(p, w.idx) for p, w in s.matched_words}
            claimed.difference_update(old)
            s.matched_words = list(span)
            claimed.update((p, w.idx) for p, w in span)
            relocated += 1
    return relocated


def _one_visual_line(picked: list[tuple[int, Word]]) -> bool:
    """All picked words on a single visual line of one page."""
    pages = {p for p, _w in picked}
    if len(pages) != 1:
        return False
    return len(line_groups([w for _p, w in picked])) == 1


def _one_spatial_cell(picked: list[tuple[int, Word]]) -> bool:
    return words_form_one_cell([w for _p, w in picked])


def _assemble_subsequence(
    flat_words: list[tuple[int, Word]],
    lo: int,
    hi: int,
    cores: list[str],
    claimed: set[tuple[int, int]],
    own: set[tuple[int, int]],
    coherent: Callable[[list[tuple[int, Word]]], bool] | None = None,
    max_window: int = _ASSEMBLE_MAX_WINDOW,
) -> list[tuple[int, Word]] | None:
    """In-order subsequence of unclaimed window words whose cores
    assemble ``cores``, accepted only when the picked words satisfy
    ``coherent`` — by default, forming one spatially-connected cell on a
    single page. One token core may be consumed by a *consecutive* run
    of words (OCR shatters "3.5%" into ``3 . 5 %``); words may be
    skipped freely between cores (that's the interleave) but never
    inside one. Every position that can start the first core is tried;
    within a start, each core takes the leftmost run. Windows larger
    than ``max_window`` are refused."""
    if hi - lo > max_window:
        return None
    if not cores:
        return None
    if coherent is None:
        coherent = _one_spatial_cell

    def eligible(gp: int, page0: int) -> str | None:
        """The word's core, or None when it can't participate: off
        the start page, or claimed by another element. Empty-core
        words (bare punctuation) return '' — transparent inside a
        run because the alignment stream never sees them either."""
        p, w = flat_words[gp]
        if p != page0:
            return None
        if (p, w.idx) in claimed and (p, w.idx) not in own:
            return None
        return core_token(w.text)

    def consume(
        gp: int, page0: int, c: str, scan: bool = True
    ) -> tuple[int, list[tuple[int, Word]]] | None:
        """Leftmost consecutive run at-or-after ``gp`` whose
        concatenated cores equal ``c``; returns (next position,
        run words). With ``scan=False`` the run must begin exactly
        at ``gp`` (used by the outer start loop, which otherwise
        would rescan the whole window from every start)."""
        for start in range(gp, gp + 1 if not scan else hi):
            wc = eligible(start, page0)
            if not wc or not c.startswith(wc):
                continue
            acc = wc
            run = [flat_words[start]]
            j = start + 1
            while len(acc) < len(c) and j < hi:
                nc = eligible(j, page0)
                if nc is None:
                    break
                j += 1
                if not nc:
                    continue  # transparent punctuation fragment
                if not c.startswith(acc + nc):
                    break
                acc += nc
                run.append(flat_words[j - 1])
            if acc == c:
                return j, run
        return None

    for start in range(lo, hi):
        page0 = flat_words[start][0]
        first = consume(start, page0, cores[0], scan=False)
        if first is None:
            continue  # only positions that *start* the first core
        gp, picked = first
        ok = True
        for c in cores[1:]:
            nxt = consume(gp, page0, c)
            if nxt is None:
                ok = False
                break
            gp = nxt[0]
            picked.extend(nxt[1])
        if not ok:
            continue
        if not coherent(picked):
            continue
        return picked
    return None


def _ground_row_context_segs(
    segs: list[_TextSeg],
    page_words: dict[int, list[Word]],
) -> tuple[int, int, int, int]:
    """Ground the segments only their *row context* can place: nodes
    invisible or refused by every earlier pass, pinned positionally by
    the words their element siblings already grounded — plus a
    relocation check for fully-matched cells stranded on a duplicate
    outside their row band (see ``relocate_out_of_band``).

    Three kinds land here:

    - **Punctuation-only nodes** ('?' table cells, standalone dashes):
      no alphanumeric token cores, so alignment never sees them. The
      single window word with equal normalized text grounds them.
    - **Interleaved multi-line cells**: adjacent multi-line table cells
      whose words nearly touch on one line get merged into a single
      component by the reading-order cell builder, so their lines
      interleave in the OCR stream ("Monthly Base Annual Base / Rent
      (M$) Rent (M$)") and neither cell's text is ever *contiguous* —
      every span-based pass refuses. The cell's words are still all
      present in the row window, in order; an in-order subsequence of
      unclaimed window words whose cores equal the node's token cores
      grounds it, but only when the picked words form ONE spatially-
      connected cell on the page (scattered lookalikes assembled from
      unrelated cells fail that check and are refused). Document-order
      processing plus leftmost-greedy picking keeps interleaved sibling
      columns honest: the row's first cell claims the row's first
      matching words, the second the second.
    - **Digit-discrepant text** (a date the generator transcribed as
      "Jun 18, 2025" where the page reads "Jun 16, 2025"): alignment,
      recovery, and rescue all correctly refuse digit mismatches when
      *searching*, but inside an already-grounded row the location is
      not in question — only the digits are. A contiguous window span
      whose digit-masked core equals the node's grounds it. Letters
      must match exactly; pure-numeric text (amounts, IDs) is
      deliberately excluded, since without letters to pin identity a
      masked match could put one amount in another's column.

    The window is tried in order of reliability: the gpos span of the
    sibling pool (same parent: for a table cell, its row — immune to
    document-order pathologies, since rescued rows count too), then the
    interval between the nearest grounded segments in document order
    (median word position, refused above ``_PUNCT_MAX_FALLBACK_WINDOW``).
    Within a window the nearest match to the center wins, preferring
    unclaimed words; claimed words are an accepted fallback because
    duplicated XML content (a table the generator emitted twice) should
    point at the one place the page shows it.

    Returns ``(punct_nodes_grounded, shape_matched_tokens,
    interleaved_cell_tokens, band_relocations)``."""
    pos_in_page, page_offset, flat_words = _reading_index(page_words)
    claimed = _claimed_words(segs)

    # Words claimed by two or more segments when this pass starts. Two
    # elements sharing the same words is the signature of a duplicate
    # collision (the page shows the text twice but the span search
    # could only find one copy) — the out-of-band relocation below
    # requires it, so a correctly-placed segment that merely sits far
    # from its siblings is never yanked onto a lookalike.
    claim_counts: dict[tuple[int, int], int] = {}
    for s in segs:
        for p, w in s.matched_words:
            key = (p, w.idx)
            claim_counts[key] = claim_counts.get(key, 0) + 1
    shared_words = {key for key, n in claim_counts.items() if n > 1}

    def gpos(p: int, w: Word) -> int:
        return page_offset[p] + pos_in_page[(p, w.idx)]

    # Each segment's words contribute to the sibling-window pool of its
    # owner element AND its owner's parent — so a '?' cell looking up
    # its parent (the row) sees both the row's own text and the other
    # cells' text. Keyed by the element itself, NOT id(): lxml elements
    # are proxies, and only a held reference (the dict key) keeps the
    # proxy alive so later ``getparent()`` calls resolve to it.
    ctx_words: dict[Any, list[int]] = {}
    for s in segs:
        if not s.matched_words:
            continue
        positions = [gpos(p, w) for p, w in s.matched_words]
        owner = s.owner
        ctx_words.setdefault(owner, []).extend(positions)
        parent = owner.getparent()
        if parent is not None:
            ctx_words.setdefault(parent, []).extend(positions)

    def search(lo: int, hi: int, center: float, target: str) -> tuple[int, Word] | None:
        best_un: tuple[float, int, Word] | None = None
        best_cl: tuple[float, int, Word] | None = None
        for gp in range(lo, hi):
            page, w = flat_words[gp]
            if fuzzy_norm(w.text) != target:
                continue
            d = abs(gp - center)
            if (page, w.idx) in claimed:
                if best_cl is None or d < best_cl[0]:
                    best_cl = (d, page, w)
            elif best_un is None or d < best_un[0]:
                best_un = (d, page, w)
        best = best_un or best_cl
        return (best[1], best[2]) if best else None

    def masked_shape(s: str) -> str:
        """Digit-masked, case-folded, punctuation-preserving shape key.
        Built on ``fuzzy_norm`` (not ``core_token``) deliberately: the
        currency symbol is load-bearing — ``$22.85`` must not share a
        shape with the year ``2025`` just because both are four digits."""
        return re.sub(r"\d", "0", fuzzy_norm(s).lower())

    def shape_candidates(
        lo: int, hi: int, target: str, own: set[tuple[int, int]]
    ) -> list[tuple[float, float, list[tuple[int, Word]]]]:
        """All contiguous spans in the window whose digit-masked core
        equals ``target`` (already masked), as ``(claimed_share, start,
        span)``. Words with empty cores (stray punctuation fragments)
        are transparent within a span. ``own`` is the searching
        segment's current words — they must not make its own (correct)
        location look taken when scoring."""
        candidates: list[tuple[float, float, list[tuple[int, Word]]]] = []
        for start in range(lo, hi):
            if not masked_shape(flat_words[start][1].text):
                continue  # spans start on a content word, not whitespace noise
            acc = ""
            span: list[tuple[int, Word]] = []
            for gp in range(start, hi):
                page, w = flat_words[gp]
                mc = masked_shape(w.text)
                if not mc:
                    continue
                acc += mc
                if not target.startswith(acc):
                    break
                span.append((page, w))
                if len(acc) == len(target):
                    claimed_n = sum(
                        1 for p, sw in span if (p, sw.idx) in claimed and (p, sw.idx) not in own
                    )
                    candidates.append((claimed_n / len(span), float(start), span))
                    break
        return candidates

    def pick_mixed(
        candidates: list[tuple[float, float, list[tuple[int, Word]]]],
    ) -> list[tuple[int, Word]] | None:
        if not candidates:
            return None
        # Unclaimed first, then LEFTMOST (reading order) — segments are
        # processed in document order, so the row's first date cell
        # takes the row's first date span and the second the second.
        # Center distance swapped same-shape sibling columns.
        return min(candidates, key=lambda c: (c[0], c[1]))[2]

    def same_line_pick_safe(seg: _TextSeg, span: list[tuple[int, Word]]) -> bool:
        """Refuse a same-line completion whose pick skipped over a
        *claimed twin* — a word with the same core, owned by another
        element, sitting between the segment's anchor and the pick.
        The nearest copy being taken means the farther pick likely
        belongs to a neighboring cell on the same row line ("Total |
        1,500 | 1,500": if this cell's own 1,500 went to someone else,
        grabbing the other column's copy would box the wrong value)."""
        own_keys = {(p, w.idx) for p, w in seg.matched_words}
        own_pos = [gpos(p, w) for p, w in seg.matched_words]
        for p, w in span:
            if (p, w.idx) in own_keys:
                continue
            core = core_token(w.text)
            if not core:
                continue
            g = gpos(p, w)
            anchor = min(own_pos, key=lambda x: abs(x - g))
            for gp in range(min(anchor, g) + 1, max(anchor, g)):
                p2, w2 = flat_words[gp]
                key2 = (p2, w2.idx)
                if key2 in own_keys or key2 not in claimed:
                    continue
                if core_token(w2.text) == core:
                    return False
        return True

    def assemble_interleaved(
        lo: int,
        hi: int,
        seg: _TextSeg,
        own: set[tuple[int, int]],
        coherent: Callable[[list[tuple[int, Word]]], bool] | None = None,
    ) -> list[tuple[int, Word]] | None:
        """See :func:`_assemble_subsequence` — this binds the pass's
        reading-order stream and claim state to the segment's cores."""
        cores = [c for t in seg.raw.split() if (c := core_token(t))]
        return _assemble_subsequence(flat_words, lo, hi, cores, claimed, own, coherent)

    def pick_numeric(
        candidates: list[tuple[float, float, list[tuple[int, Word]]]],
    ) -> list[tuple[int, Word]] | None:
        # Pure-numeric text has no letters to pin identity, so the bar
        # is higher: prefer single-OCR-word spans (a transaction id or
        # amount is one word on the page; multi-word digit runs can be
        # frankenspans assembled from unrelated cells), and require
        # EXACTLY ONE unclaimed candidate — uniqueness in the row is
        # the identity anchor. Two same-shape amounts (charge vs
        # balance) stay unmatched rather than guessed at.
        pool = [c for c in candidates if len(c[2]) == 1] or candidates
        unclaimed = [c for c in pool if c[0] == 0.0]
        if len(unclaimed) != 1:
            return None
        return unclaimed[0][2]

    def median(values: list[int]) -> float:
        vs = sorted(values)
        return float(vs[len(vs) // 2])

    # The fixed slack alone misses a row whose LEADING cells are the
    # unmatched ones: the sibling pool starts at the first matched cell
    # and the words being searched for sit just before it. Widen each
    # parent's window by its own unmatched-token deficit (bounded).
    parent_deficit: dict[Any, int] = {}
    for s in segs:
        deficit = (s.n_tokens - s.matched_tokens) if s.n_tokens else (0 if s.matched_words else 1)
        if deficit <= 0:
            continue
        parent = s.owner.getparent()
        if parent is not None:
            parent_deficit[parent] = parent_deficit.get(parent, 0) + deficit

    def context_window(i: int, seg: _TextSeg) -> tuple[int, int, float] | None:
        parent = seg.owner.getparent()
        sib_positions = ctx_words.get(parent, []) if parent is not None else []
        if sib_positions:
            slack = _PUNCT_WINDOW_SLACK + min(24, parent_deficit.get(parent, 0))
            lo = max(0, min(sib_positions) - slack)
            hi = min(len(flat_words), max(sib_positions) + 1 + slack)
            return lo, hi, (lo + hi) / 2
        prev_pos = 0.0
        for j in range(i - 1, -1, -1):
            if segs[j].matched_words:
                prev_pos = median([gpos(p, w) for p, w in segs[j].matched_words]) + 1
                break
        next_pos = float(len(flat_words))
        for j in range(i + 1, len(segs)):
            if segs[j].matched_words:
                next_pos = median([gpos(p, w) for p, w in segs[j].matched_words])
                break
        lo = int(max(0, min(prev_pos, next_pos) - _PUNCT_WINDOW_SLACK))
        hi = int(min(len(flat_words), max(prev_pos, next_pos) + _PUNCT_WINDOW_SLACK))
        if hi - lo > _PUNCT_MAX_FALLBACK_WINDOW:
            return None
        return lo, hi, (prev_pos + next_pos) / 2

    def grandparent_window(seg: _TextSeg) -> tuple[int, int] | None:
        """Window over the grounded words of the parent's siblings —
        for a table cell, the whole table's span."""
        parent = seg.owner.getparent()
        grand = parent.getparent() if parent is not None else None
        if grand is None:
            return None
        pool: list[int] = []
        for sib in grand:
            if isinstance(sib.tag, str):
                pool.extend(ctx_words.get(sib, []))
        if not pool:
            return None
        lo = max(0, min(pool) - _PUNCT_WINDOW_SLACK)
        hi = min(len(flat_words), max(pool) + 1 + _PUNCT_WINDOW_SLACK)
        return lo, hi

    def note_ctx(seg: _TextSeg, span: list[tuple[int, Word]]) -> None:
        """Feed a fresh grounding back into the context pools, so later
        segments' windows (in document order) see it — a header cell
        grounded by this pass is exactly the anchor its column's data
        cells need when a merged neighbor dragged them out of their own
        row band."""
        positions = [gpos(p, w) for p, w in span]
        owner = seg.owner
        ctx_words.setdefault(owner, []).extend(positions)
        parent = owner.getparent()
        if parent is not None:
            ctx_words.setdefault(parent, []).extend(positions)

    def drop_ctx(seg: _TextSeg, span: list[tuple[int, Word]]) -> None:
        """Retract abandoned words from the context pools when a pass
        replaces a segment's placement — stale positions would stretch
        later siblings' windows toward the abandoned duplicate region,
        inviting them to assemble on the freed copy."""
        positions = [gpos(p, w) for p, w in span]
        owner = seg.owner
        for key in (owner, owner.getparent()):
            pool = ctx_words.get(key) if key is not None else None
            if not pool:
                continue
            for pos in positions:
                try:
                    pool.remove(pos)
                except ValueError:
                    pass

    by_parent: dict[Any, list[_TextSeg]] = {}
    for s in segs:
        p = s.owner.getparent()
        if p is not None:
            by_parent.setdefault(p, []).append(s)

    def relocate_out_of_band(seg: _TextSeg) -> bool:
        """A fully-matched cell stranded on a duplicate: when its whole
        text was swallowed into a merged neighbor cell (no contiguous
        span), the span search could only ever find the page's *other*
        copy (a totals row repeating the row's amount) — and the cell
        that legitimately owns that copy claims the very same words.
        Three gates keep correctly-placed segments alone: cell-sized
        only, every word also claimed by another segment (the duplicate
        collision), and the words entirely outside the row band. If an
        unclaimed coherent copy then assembles inside the row window,
        that copy is the cell's."""
        if seg.n_tokens > _RELOCATE_MAX_TOKENS:
            return False
        if not all((p, w.idx) in shared_words for p, w in seg.matched_words):
            return False
        parent = seg.owner.getparent()
        if parent is None:
            return False
        sib_pos = [
            gpos(p, w)
            for s in by_parent.get(parent, [])
            if s is not seg
            for p, w in s.matched_words
        ]
        if not sib_pos:
            return False
        lo = max(0, min(sib_pos) - _PUNCT_WINDOW_SLACK)
        hi = min(len(flat_words), max(sib_pos) + 1 + _PUNCT_WINDOW_SLACK)
        if any(lo <= gpos(p, w) < hi for p, w in seg.matched_words):
            return False  # touches its row band — placement is plausible
        span = assemble_interleaved(lo, hi, seg, set())
        if span is None:
            return False
        claimed.difference_update((p, w.idx) for p, w in seg.matched_words)
        drop_ctx(seg, seg.matched_words)
        seg.matched_words = list(span)
        seg.pinned = True
        claimed.update((p, w.idx) for p, w in span)
        note_ctx(seg, span)
        return True

    punct_grounded = 0
    shape_tokens = 0
    interleaved_tokens = 0
    band_relocations = 0
    for i, seg in enumerate(segs):
        if seg.n_tokens == 0:
            if seg.matched_words:
                continue
            # Punctuation-only node: single-word exact match.
            target = fuzzy_norm(seg.raw)
            if not target:
                continue
            win = context_window(i, seg)
            if win is None:
                continue
            found = search(*win, target)
            if found is not None:
                page, w = found
                seg.matched_words.append((page, w))
                claimed.add((page, w.idx))
                note_ctx(seg, [(page, w)])
                punct_grounded += 1
            continue
        if seg.matched_tokens == seg.n_tokens:
            if seg.n_tokens >= 2 and seg.matched_words and relocate_out_of_band(seg):
                band_relocations += 1
            continue
        win = context_window(i, seg)
        if win is None:
            continue
        old = {(p, w.idx) for p, w in seg.matched_words}
        lo, hi, _center = win
        # Interleaved-cell assembly first: it demands exact core
        # equality on every token, so when it fires it is stronger
        # evidence than the digit-masked shape match below. Single-token
        # segments are excluded — one word grounds nothing a plain span
        # search couldn't, and the coherence check is vacuous for it.
        if seg.n_tokens >= 2:
            span = assemble_interleaved(lo, hi, seg, old)
            if span is None:
                # A merged neighbor can drag the cell's words into a
                # different row *band* (a data cell unioned with the
                # header cell above it serializes with the header rows),
                # putting them outside the sibling window entirely. The
                # enclosing element's window (for a table cell: the
                # whole table) still bounds the search; exact cores plus
                # the one-cell coherence gate carry the safety there.
                gwin = grandparent_window(seg)
                if gwin is not None:
                    span = assemble_interleaved(gwin[0], gwin[1], seg, old)
            if span is None and seg.matched_words:
                # Same-line completion for a partial: the words already
                # matched pin a visual line, and the rest of the text
                # may sit further along it past a gap too wide for cell
                # connectivity (a signatory name typeset as "Tarek
                # <wide gap> Vance-Vargas" under a signature rule). The
                # shared line replaces the one-cell requirement.
                own_pos = [gpos(p, w) for p, w in seg.matched_words]
                lo2 = max(0, min(own_pos) - _SAME_LINE_SLACK)
                hi2 = min(len(flat_words), max(own_pos) + 1 + _SAME_LINE_SLACK)
                span = assemble_interleaved(lo2, hi2, seg, old, coherent=_one_visual_line)
                if span is not None and not same_line_pick_safe(seg, span):
                    span = None
            if span is not None:
                claimed.difference_update(old)
                drop_ctx(seg, seg.matched_words)
                interleaved_tokens += seg.n_tokens - seg.matched_tokens
                seg.matched_words = list(span)
                seg.matched_tokens = seg.n_tokens
                seg.pinned = True
                claimed.update((p, w.idx) for p, w in span)
                note_ctx(seg, span)
                continue
        # Shape rescue: an unmatched or partially-matched digit-bearing
        # segment (a date or transaction id the generator transcribed
        # with a wrong digit) whose row context is grounded. Digits are
        # masked — the location evidence (same row, same shape)
        # outweighs the digit discrepancy, and grounding answers
        # "where", not "what". A partial's stray words (the matching
        # pieces alignment did pair, possibly with another row's
        # identical fragments) are replaced wholesale by the clean
        # span. Mixed letters+digits text ("Jun 18, 2025") has its
        # letters as the identity anchor; pure-numeric text (ids,
        # amounts) instead requires a unique candidate — see
        # ``pick_numeric``.
        core = core_token(seg.raw)
        if not any(c.isdigit() for c in core):
            continue
        candidates = shape_candidates(lo, hi, masked_shape(seg.raw), old)
        if any(c.isalpha() for c in core):
            span = pick_mixed(candidates)
        else:
            span = pick_numeric(candidates)
        if span is not None:
            claimed.difference_update(old)
            drop_ctx(seg, seg.matched_words)
            shape_tokens += seg.n_tokens - seg.matched_tokens
            seg.matched_words = list(span)
            seg.matched_tokens = seg.n_tokens
            claimed.update((p, w.idx) for p, w in span)
            note_ctx(seg, span)
    return punct_grounded, shape_tokens, interleaved_tokens, band_relocations


def _page_assembly_candidates(
    seg: _TextSeg,
    lo_page: int,
    hi_page: int,
    page_words: dict[int, list[Word]],
    claimed: set[tuple[int, int]],
    own: set[tuple[int, int]],
) -> tuple[list[tuple[int, list[tuple[int, Word]]]], int]:
    """Page-wide one-cell subsequence assemblies for a segment that no
    contiguous span search could place, as ``(candidates, cores_matched)``.

    When the full token list assembles nowhere, progressively shorter
    *prefixes* are tried (longest wins, never below 3 cores or ~60% of
    the segment) — for label-plus-value text whose trailing value the
    page never rendered as words (a chart-only number). The caller
    commits such a match as a partial."""
    cores = [c for t in seg.raw.split() if (c := core_token(t))]
    n = len(cores)
    if n < 2 or sum(len(c) for c in cores) < _RESCUE_ASSEMBLE_MIN_CHARS:
        return [], 0
    min_k = n if n < 4 else max(3, (3 * n + 4) // 5)
    for k in range(n, min_k - 1, -1):
        found: list[tuple[int, list[tuple[int, Word]]]] = []
        for page in range(lo_page, hi_page + 1):
            words = page_words.get(page, [])
            if not words or len(words) > _RESCUE_ASSEMBLE_MAX_PAGE_WORDS:
                continue
            flat = [(page, w) for w in words]
            picked = _assemble_subsequence(
                flat, 0, len(flat), cores[:k], claimed, own, max_window=len(flat)
            )
            if picked is not None:
                found.append((page, picked))
        if found:
            return found, k
    return [], 0


def _assemble_stranded_segs(
    segs: list[_TextSeg],
    otoks: list[_OTok],
    pairs: dict[int, int],
    page_words: dict[int, list[Word]],
) -> int:
    """Last-resort page-wide subsequence assembly for segments every
    earlier pass left (fully or partly) unmatched. Returns the number of
    tokens grounded.

    The case this exists for: a wrapped multi-line table cell whose
    neighbor columns nearly touch gets merged into one reading-order
    component, so its words interleave with the neighbors' line-by-line
    and *no page offers a contiguous span* — and its sibling cells all
    mis-grounded onto a duplicate table elsewhere (identical header
    labels two tables share), so the row-context windows point away from
    the true location. Page-wide assembly with the row-context pass's
    own safety gates — exact core equality, unclaimed words only,
    one-cell coherence on a single page — finds the words where they
    actually sit. Runs AFTER the row-context pass so in-window
    interleaves keep their richer sibling-window semantics; this pass
    only sees the stranded remainder. Candidate pages and the expected
    stream position come from the segment's aligned neighbors, same as
    duplicate rescue."""
    if not pairs:
        return 0
    sorted_x = sorted(pairs)
    pos_in_page, page_offset, flat_words = _reading_index(page_words)
    cum = len(flat_words)

    def gpos(page: int, word: Word) -> float:
        return page_offset[page] + pos_in_page[(page, word.idx)]

    claimed = _claimed_words(segs)
    grounded = 0
    for seg in segs:
        if not seg.n_tokens or seg.matched_tokens == seg.n_tokens:
            continue
        lo_tok = seg.token_start
        hi_tok = seg.token_start + seg.n_tokens - 1
        k = bisect_right(sorted_x, lo_tok) - 1
        prev_ot = otoks[pairs[sorted_x[k]]] if k >= 0 else None
        k2 = bisect_right(sorted_x, hi_tok)
        next_ot = otoks[pairs[sorted_x[k2]]] if k2 < len(sorted_x) else None
        prev_page = prev_ot.page if prev_ot else min(page_words)
        next_page = next_ot.page if next_ot else max(page_words)
        lo_page = max(min(page_words), min(prev_page, next_page) - _RESCUE_PAGE_SLACK)
        hi_page = min(max(page_words), max(prev_page, next_page) + _RESCUE_PAGE_SLACK)
        prev_gpos = gpos(prev_ot.page, prev_ot.word) if prev_ot else 0.0
        next_gpos = gpos(next_ot.page, next_ot.word) if next_ot else float(cum)
        expected_gpos = (prev_gpos + next_gpos) / 2

        own = {(p, w.idx) for p, w in seg.matched_words}
        assembled, n_cores = _page_assembly_candidates(
            seg, lo_page, hi_page, page_words, claimed, own
        )
        if not assembled or n_cores <= seg.matched_tokens:
            continue

        def assembly_distance(
            cand: tuple[int, list[tuple[int, Word]]],
            expected: float = expected_gpos,
        ) -> float:
            _page, picked = cand
            mid = sum(gpos(p, w) for p, w in picked) / len(picked)
            return abs(mid - expected)

        _best_page, picked = min(assembled, key=assembly_distance)
        claimed.difference_update(own)
        grounded += n_cores - seg.matched_tokens
        seg.matched_tokens = n_cores
        seg.matched_words = list(picked)
        seg.pinned = True
        claimed.update((p, w.idx) for p, w in picked)
    return grounded


def _rescue_unmatched_segs(
    segs: list[_TextSeg],
    xtoks_seg: list[int],
    otoks: list[_OTok],
    pairs: dict[int, int],
    page_words: dict[int, list[Word]],
    page_dims: dict[int, PageDims],
) -> int:
    """Span-search rescue for segments the aligner matched nothing of.

    Two distinct cases land here, and the candidate scoring serves both:

    - Content the XML legitimately repeats (cover-page address blocks
      emitted twice, running headers). The aligner consumes each OCR
      word at most once, so later copies are fully unmatched and the
      only span the page offers is already claimed — claimed words are
      fair game, because duplicated XML content *should* point at the
      one place the page shows it.
    - Short text that fell out of the monotonic alignment because its
      neighborhood is locally reordered (a one-word table cell whose
      row serializes differently in DGML than in OCR). Here the page
      often shows the same word elsewhere too (e.g. a fee-table label
      "Software" vs. "Software" in the SOW prose above it), and the
      *unclaimed* occurrence is the right one — the claimed occurrence
      already belongs to the element the aligner gave it to.

    Hence the score: prefer the span with the smallest share of
    already-claimed words, then the one closest to where the segment's
    aligned neighbors say it should sit in the reading-order stream
    (page distance alone can't break a same-page tie)."""
    if not pairs:
        return 0
    sorted_x = sorted(pairs)
    pos_in_page, page_offset, flat_words = _reading_index(page_words)
    cum = len(flat_words)

    def gpos(page: int, word: Word) -> float:
        return page_offset[page] + pos_in_page[(page, word.idx)]

    # Words already grounded by the aligner / gap recovery; updated as
    # rescues commit so two duplicate segments don't pile onto one span
    # while another equally-good span sits unclaimed.
    claimed = _claimed_words(segs)

    rescued = 0
    for seg in segs:
        # Any token-bearing segment that didn't fully match is a rescue
        # candidate — including *partially* matched ones. Duplicated XML
        # content can lose most of its tokens to an anchor that gave the
        # page occurrence to the other copy, leaving a stray word or two
        # misattributed; a clean whole-text span replaces that.
        if not seg.n_tokens or seg.matched_tokens == seg.n_tokens:
            continue
        # Expected position from the nearest aligned tokens around the
        # segment's token span.
        lo_tok = seg.token_start
        hi_tok = seg.token_start + seg.n_tokens - 1
        k = bisect_right(sorted_x, lo_tok) - 1
        prev_ot = otoks[pairs[sorted_x[k]]] if k >= 0 else None
        k2 = bisect_right(sorted_x, hi_tok)
        next_ot = otoks[pairs[sorted_x[k2]]] if k2 < len(sorted_x) else None
        prev_page = prev_ot.page if prev_ot else min(page_words)
        next_page = next_ot.page if next_ot else max(page_words)
        lo_page = max(min(page_words), min(prev_page, next_page) - _RESCUE_PAGE_SLACK)
        hi_page = min(max(page_words), max(prev_page, next_page) + _RESCUE_PAGE_SLACK)
        prev_gpos = gpos(prev_ot.page, prev_ot.word) if prev_ot else 0.0
        next_gpos = gpos(next_ot.page, next_ot.word) if next_ot else float(cum)
        expected_gpos = (prev_gpos + next_gpos) / 2

        # The segment's own (possibly misattributed) words must not make
        # their location look "taken" when scoring its candidates.
        own = {(p, w.idx) for p, w in seg.matched_words}

        candidates: list[tuple[int, tuple[int, int]]] = []  # (page, span)
        for page in range(lo_page, hi_page + 1):
            words = page_words.get(page, [])
            spans = (
                find_spans(seg.raw, words)
                or find_fuzzy_spans(seg.raw, words)
                # Boundary-punctuation leniency: a "$" the page tokenizer
                # glued into "($", a "$" or minus sign the page never
                # rendered as its own word, a trailing "Label —"
                # separator dash.
                or find_spans_lenient(seg.raw, words)
            )
            candidates.extend((page, s) for s in spans)
        if not candidates:
            continue  # no clean span — keep whatever partial match exists

        def score(
            cand: tuple[int, tuple[int, int]],
            expected: float = expected_gpos,
            own: set[tuple[int, int]] = own,
        ) -> tuple[float, float]:
            page, (start, end) = cand
            span_words = page_words[page][start:end]
            claimed_share = sum(
                1 for w in span_words if (page, w.idx) in claimed and (page, w.idx) not in own
            ) / max(1, len(span_words))
            mid = page_offset[page] + (start + end) / 2
            return (claimed_share, abs(mid - expected))

        page, span = min(candidates, key=score)
        span_words = page_words[page][span[0] : span[1]]
        rescued += seg.n_tokens - seg.matched_tokens
        claimed.difference_update(own)
        seg.matched_tokens = seg.n_tokens
        seg.matched_words = [(page, w) for w in span_words]
        claimed.update((page, w.idx) for w in span_words)
    return rescued


# ---- Attribute emission ----------------------------------------------------


def _annotate_tree(
    root: Any,
    segs: list[_TextSeg],
    page_dims: dict[int, PageDims],
    page_baselines: dict[int, float],
    *,
    emit_style: bool = True,
) -> tuple[int, int]:
    """Set the ``dg:origin`` attribute on every element whose subtree
    grounded well enough, and ``dg:style`` where observable formatting is
    evident. Elements with text-node children get one ``dg:origin`` box per
    visual line; pure containers (all-element children — sections, lists,
    tables, rows) get one union box per page, since repeating every
    descendant's line boxes up each ancestor would bloat the document without
    adding information. Returns ``(total_annotated, containers_annotated)``.

    ``dg:style`` is taken from the element's *own* direct text only (so a parent
    doesn't inherit a child's bold/size and the style lands at the most specific
    element); pure containers, having no direct text, get none. ``emit_style``
    gates deterministic ``dg:style`` only: when ``False`` (OCR files with no
    image-based ``style`` config), no ``dg:style`` is written — the all-caps
    ``text-transform`` reading is text-derived and would otherwise leak onto OCR
    output that the docs promise is unstyled. ``dg:origin`` is emitted
    regardless."""
    # Per-element direct segments.
    own_segs: dict[Any, list[_TextSeg]] = {}
    for seg in segs:
        own_segs.setdefault(seg.owner, []).append(seg)

    attr_name = _dg_attr_name(root, "origin")
    style_attr = _dg_attr_name(root, "style")

    # Bottom-up accumulation of (tokens, matched, words) per subtree.
    annotated = 0
    containers = 0
    subtree: dict[Any, tuple[int, int, list[tuple[int, Word]]]] = {}

    def walk(el: Any) -> tuple[int, int, list[tuple[int, Word]]]:
        tokens = 0
        matched = 0
        words: list[tuple[int, Word]] = []
        for seg in own_segs.get(el, []):
            tokens += seg.n_tokens
            matched += seg.matched_tokens
            words.extend(seg.matched_words)
        for child in el:
            if not isinstance(child.tag, str):
                continue
            ct, cm, cw = walk(child)
            tokens += ct
            matched += cm
            words.extend(cw)
        subtree[el] = (tokens, matched, words)
        return tokens, matched, words

    walk(root)

    for el, (tokens, matched, words) in subtree.items():
        if not words:
            continue
        # tokens == 0 with words present is a punctuation-only element
        # ('?' table cell) grounded by the positional pass — emit it;
        # the fraction gate only applies where there are tokens to gate.
        if tokens and matched / tokens < _EMIT_MIN_FRACTION:
            continue
        if el in own_segs:
            value = _format_boxes(words)
        else:
            value = _format_page_boxes(words)
            containers += 1
        if value:
            el.set(attr_name, value)
            annotated += 1
        # dg:style from the element's OWN direct text/words — font facts from the
        # matched words, text-transform from the element's own text. Only elements
        # with direct text (in own_segs) are styled; pure containers have none.
        if emit_style and el in own_segs:
            own_words = [pw for seg in own_segs[el] for pw in seg.matched_words]
            own_text = "".join(seg.raw for seg in own_segs[el])
            style_value = _aggregate_style(own_words, page_baselines, own_text=own_text)
            if style_value:
                el.set(style_attr, style_value)
    return annotated, containers


def _is_ocr_file(workspace: Workspace, file_id: str) -> bool:
    """Whether the file's recorded ``text_mode`` is ``ocr`` — the authoritative
    gate for the image-based ``dg:style`` path (digital/hybrid get
    deterministic style instead). Missing/unreadable records → ``False``."""
    from .files import FileStore
    from .text_extraction import TextMode

    try:
        record = FileStore(workspace).get(file_id)
    except Exception:
        return False
    return record.text_mode == TextMode.OCR.value


def _maybe_style_from_image(
    workspace: Workspace,
    file_id: str,
    root: Any,
    *,
    config: StyleConfig | None,
    is_ocr: bool,
    debug: bool = False,
) -> None:
    """Run the image-based ``dg:style`` pass when the workspace has a ``style``
    config section (``config`` is the pre-loaded, already-validated section, or
    ``None``) *and* the file was extracted with OCR (``is_ocr``). A no-op
    otherwise, so the default deterministic path stays free of any LLM
    dependency. All LLM machinery is imported lazily here.

    This is **best-effort**: it only adds ``dg:style`` on top of the
    deterministic grounding already computed into ``root``. A runtime failure
    (missing/rejected credential, provider or network error) is swallowed —
    exactly as per-page failures are inside
    :func:`dgml_core.style_llm.annotate_style_from_image` — so the caller still
    writes the file with its ``dg:origin`` boxes intact rather than losing the
    whole grounding to a style-only problem. ``debug`` gates the per-page
    ``usage.jsonl`` recording done in the call layer."""
    if config is None or not is_ocr:
        return

    from .style_config import resolve_api_key

    try:
        from .llm import LLMConfig
        from .style_llm import annotate_style_from_image

        llm_config = LLMConfig(
            model=config.model,
            api_base=config.api_base,
            api_key=resolve_api_key(config),
            max_tokens=config.max_tokens,
        )
        annotate_style_from_image(
            workspace,
            file_id,
            root,
            config=llm_config,
            style_attr=_dg_attr_name(root, "style"),
            origin_attr=_dg_attr_name(root, "origin"),
            debug=debug,
        )
    except Exception:
        # Best-effort enhancement — never let it discard the grounding.
        return


def _suppress_inherited_style(root: Any, style_attr: str) -> None:
    """Remove ``dg:style`` declarations a descendant only inherits from an
    ancestor, in place. Walks top-down carrying the inherited value of each
    CSS-inheriting property; on each element, any declaration equal to what it
    would inherit is dropped (it is redundant — ``dg:style`` is copied verbatim
    into an HTML ``style`` attribute, where these properties inherit). Keeps a
    style at the most specific element that *introduces* or *changes* it, and
    strips the attribute entirely when nothing distinctive remains.

    The walk is seeded with :data:`INHERITED_DEFAULTS` (the CSS initial values),
    so this pass is also what elides default declarations: a root-level
    ``font-weight: normal`` matches the seeded default and is dropped, while the
    same ``normal`` under a ``bold`` ancestor differs from the inherited value
    and is kept — the override that :func:`build_style` deliberately does not
    strip. This unifies "drop defaults" and "suppress inherited" into one
    inheritance-aware step.

    Non-inheriting properties (``text-decoration``, ``background-color``) are
    never suppressed — a descendant repeating them is meaningful."""

    def walk(el: Any, inherited: dict[str, str]) -> None:
        value = el.get(style_attr)
        child_inherited = inherited
        if value is not None:
            decls = parse_style_declarations(value)
            kept = {
                prop: val
                for prop, val in decls.items()
                if not (prop in INHERITED_PROPERTIES and inherited.get(prop) == val)
            }
            rebuilt = build_style(kept)
            if rebuilt:
                el.set(style_attr, rebuilt)
            else:
                del el.attrib[style_attr]
            # Descendants inherit this element's inheriting values (whether we
            # kept or dropped them as redundant — the effective value is the same).
            child_inherited = {**inherited}
            child_inherited.update({p: v for p, v in decls.items() if p in INHERITED_PROPERTIES})
        for child in el:
            if isinstance(child.tag, str):
                walk(child, child_inherited)

    walk(root, dict(INHERITED_DEFAULTS))


def _clear_attr(root: Any, attr_name: str) -> None:
    """Remove ``attr_name`` from every element in the tree, in place."""
    for el in root.iter():
        if isinstance(el.tag, str) and el.get(attr_name) is not None:
            del el.attrib[attr_name]


def _dg_attr_name(root: Any, local: str) -> str:
    """A ``dg:``-prefixed attribute name (e.g. ``origin``, ``style``) qualified
    to whatever URI the document binds the ``dg`` prefix to — the
    ``dgml.io`` scheme on generated DGML, or the ``docugami.com``
    namespace — so the serialized attribute reuses the document's own ``dg``
    prefix. Bare ``local`` for namespace-free XML (e.g. a
    ``--no-semantic-transform`` artifact) so we don't force declarations into a
    plain document."""
    nsmap = getattr(root, "nsmap", {}) or {}
    dg_uri = nsmap.get("dg")
    if dg_uri:
        return f"{{{dg_uri}}}{local}"
    return local


def _page_baseline(words: list[Word]) -> float | None:
    """The page's "normal" body font size = the char-weighted mode of word
    sizes (rounded to 0.1pt). ``font-size`` in ``dg:style`` is an ``em`` bucket
    relative to this. Returns ``None`` when no word carries a size."""
    counts: Counter[float] = Counter()
    for w in words:
        if w.size:
            counts[round(w.size, 1)] += max(1, len(w.text))
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _aggregate_style(
    words: list[tuple[int, Word]],
    page_baselines: dict[int, float],
    *,
    own_text: str,
) -> str:
    """Build an element's ``dg:style`` from its own matched words and text.

    font-weight/font-style/color are char-weighted majorities over the words;
    font-size is the char-weighted representative size bucketed against the
    word's page baseline; text-transform is the all-caps test on ``own_text``.
    ``text-align`` is deliberately not derived here — page-relative geometry
    can't reliably tell right-aligned text from a left-aligned column, so it is
    left to the OCR image path (:mod:`dgml.style_llm`).

    When the words carry font facts (digital text), the *observed* value is
    emitted including the default — ``font-weight: normal``, ``font-size: 1em``,
    ``text-transform: none`` — not just the non-default one. That is what lets a
    plain child override a bold/large/uppercase ancestor;
    :func:`_suppress_inherited_style` then prunes every copy that merely restates
    the inherited value, so the output stays sparse. OCR words carry no font
    facts, so font-weight/font-style/font-size are left unset (the image path
    speaks instead). Returns ``""`` when nothing observable survives."""
    decls: dict[str, str] = {}

    total = 0
    bold = 0
    italic = 0
    weighted_size = 0.0
    size_weight = 0
    font_facts = 0  # weight of words carrying any font fact (size/weight/slant/color)
    colors: Counter[str] = Counter()
    for _page, w in words:
        weight = max(1, len(w.text))
        total += weight
        if w.bold:
            bold += weight
        if w.italic:
            italic += weight
        if w.size:
            weighted_size += w.size * weight
            size_weight += weight
        if w.color:
            colors[w.color] += weight
        if w.size or w.bold or w.italic or w.color:
            font_facts += weight

    # ``font_facts`` marks *digital* words: they carry glyph facts (size, and
    # weight/slant/color when non-default), whereas OCR words carry none. Only
    # assert font-weight/font-style when we actually have those facts — and then
    # emit the *observed* value including the "normal" default, so a plain child
    # can override a bold ancestor (_suppress_inherited_style prunes the copies
    # that merely restate the inherited value). Staying silent for OCR is what
    # lets the image path's reading survive the base-wins merge_styles.
    if font_facts:
        decls["font-weight"] = "bold" if bold * 2 > total else "normal"
        decls["font-style"] = "italic" if italic * 2 > total else "normal"
    if size_weight:
        rep = weighted_size / size_weight
        page = Counter(p for p, _ in words).most_common(1)[0][0]
        baseline = page_baselines.get(page)
        if baseline:
            # Emit "1em" (body size) too, for the same override reason; only when
            # the baseline is known, so a missing baseline stays absent rather
            # than falsely asserting body size.
            decls["font-size"] = size_to_em(rep, baseline) or "1em"
    if colors:
        # Char-weighted *majority*, like bold/italic above — not mere plurality.
        # ``total`` includes the black/near-black words that never enter
        # ``colors`` (rgb_to_named returns None for them), so a mostly-black
        # paragraph with one stray colored word stays uncolored.
        top_color, top_weight = colors.most_common(1)[0]
        if top_weight * 2 > total:
            decls["color"] = top_color

    # text-transform is text-derived, so it applies to OCR too. A positive
    # all-caps reading is always asserted. The "none" override is emitted only
    # for digital text: for OCR the image path owns text-transform (it can see
    # capitalize/none) and the base-wins merge would otherwise clobber it.
    if _is_all_caps(own_text):
        decls["text-transform"] = "uppercase"
    elif font_facts and own_text.strip():
        decls["text-transform"] = "none"

    return build_style(decls)


# Letters that are cased; an all-caps run needs enough of them to read as a
# styling choice rather than an incidental acronym ("USA", "ID").
_MIN_CAPS_LETTERS = 4


def _is_all_caps(text: str) -> bool:
    """Whether ``text`` is observably uppercased — every cased letter is upper
    and there are at least :data:`_MIN_CAPS_LETTERS` of them."""
    cased = [c for c in text if c.isalpha()]
    if len(cased) < _MIN_CAPS_LETTERS:
        return False
    return all(c.isupper() for c in cased)


def _format_boxes(words: list[tuple[int, Word]]) -> str:
    """Box list in the project-wide pixel convention: each box is
    ``<page> <x1> <y1> <x2> <y2>`` (space-separated) in integer image
    pixels (top-left origin), boxes ``"; "``-separated, pages in order. One
    box per visual line on each page — the ``getClientRects()`` analogue,
    uniform for leaves and mixed-content parents alike (a parent's lines
    cover its whole subtree, so its boxes read like the paragraph the
    user sees; consumers wanting the ``getBoundingClientRect()`` form can
    union per page)."""
    by_page: dict[int, list[Word]] = {}
    for page, w in words:
        by_page.setdefault(page, []).append(w)
    parts: list[str] = []
    for page in sorted(by_page):
        for group in line_groups(by_page[page]):
            left = min(w.left for w in group)
            top = min(w.top for w in group)
            right = max(w.right for w in group)
            bottom = max(w.bottom for w in group)
            parts.append(f"{page} {left} {top} {right} {bottom}")
    return "; ".join(parts)


def _format_page_boxes(words: list[tuple[int, Word]]) -> str:
    """Box list with one *union* box per page — the
    ``getBoundingClientRect()`` analogue, same pixel convention as
    :func:`_format_boxes`. Used for pure containers, whose region is
    "everything my subtree covers" rather than a run of text lines."""
    by_page: dict[int, list[Word]] = {}
    for page, w in words:
        by_page.setdefault(page, []).append(w)
    parts: list[str] = []
    for page in sorted(by_page):
        group = by_page[page]
        left = min(w.left for w in group)
        top = min(w.top for w in group)
        right = max(w.right for w in group)
        bottom = max(w.bottom for w in group)
        parts.append(f"{page} {left} {top} {right} {bottom}")
    return "; ".join(parts)


def _top_ungrounded(root: Any, segs: list[_TextSeg], limit: int = 20) -> list[dict[str, Any]]:
    """The largest ungrounded text nodes, for the stats sidecar."""
    tree = root.getroottree()
    misses = [s for s in segs if s.n_tokens and not s.matched_tokens]
    misses.sort(key=lambda s: -s.n_tokens)
    out: list[dict[str, Any]] = []
    for s in misses[:limit]:
        snippet = " ".join(s.raw.split())
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        out.append({"path": tree.getpath(s.owner), "tokens": s.n_tokens, "text": snippet})
    return out


__all__ = [
    "DG_NAMESPACE",
    "GroundingResult",
    "ground_dgml_xml",
    "grounded_output_path",
]
