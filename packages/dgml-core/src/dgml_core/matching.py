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

"""Code-side matching of phase-1 extracted text to OCR word boxes.

Phase 1 of grounded extraction returns ``{text, locations: [{page_number}]}``
for each leaf — the value and which page(s) it lives on, but no bbox. This
module is the first non-LLM pass at assigning bboxes: for each
``(leaf, page)`` pair, find contiguous spans of OCR words on the page whose
joined text matches the value, and commit the match when it's unambiguous.

**Disambiguation by row context.** When a leaf's text has multiple
candidate spans on the page (a date that appears in several columns of a
ledger row, say), look at already-matched siblings in the same direct
parent of the values tree. Sibling bboxes anchor the row's y position;
the only candidate within ~1.5 line heights of that band wins. If the
anchor doesn't single one out, the row anchor falls through to column
anchor.

**Disambiguation by column context.** Same-row collisions (post_date,
due_date and the prefix of created_on all reading "Jun 16, 2025") all
sit on the same y-band so row anchor can't break the tie. The column
pass infers, from peers in the same array, where a given field's column
sits on the x-axis. The inference is alignment-aware: it picks whichever
of {xmin, x_center, xmax} has the tightest spread across resolved peers,
so left-aligned text columns, right-aligned currency columns, and
centered date columns all work without a hint. If no edge clusters
tightly enough, the slot isn't a column and the heuristic declines — a
free-form list of signatures positioned anywhere on the page falls
through to phase 3 untouched.

**Coordinate space.** Bounding boxes are integer image pixels
``[left, top, right, bottom]`` (top-left origin) everywhere they are
*stored* — that is the project-wide convention, the shape OCR words
arrive in from ``page_text/page_N.json`` and the shape this module emits
into ``grounded_field`` locations (see :func:`_span_to_locations`).

The disambiguation heuristics, however, reason in page-relative
normalized 0-1000 units: row-band slack, column-cluster radii, and the
tie-break tolerances are all tuned as fractions of the page so they hold
across page sizes (column anchors in particular aggregate matches across
pages). So whenever a heuristic reads a stored sibling box it normalizes
on the fly via that box's page dims — see :func:`_bbox_norm`. Pixels are
the boundary representation; normalized 0-1000 is the internal scratch
space.
"""

from __future__ import annotations

import copy
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .errors import FileNotFound
from .storage import Workspace
from .textmatch import (
    PageDims as _PageDims,
)
from .textmatch import (
    Word as _Word,
)
from .textmatch import (
    find_fuzzy_spans as _find_fuzzy_spans,
)
from .textmatch import (
    find_spans as _find_spans,
)
from .textmatch import (
    line_groups as _line_groups,
)
from .textmatch import (
    load_page_words as _load_page_words,
)
from .textmatch import (
    span_overlaps_any as _span_overlaps_any,
)

LeafPath = tuple[Any, ...]


# ---- Data shapes ----------------------------------------------------------


@dataclass(frozen=True)
class UnmatchedItem:
    """A (leaf, page) pair the matcher couldn't pin down.

    ``id`` is short and page-unique so the phase-3 LLM can refer to the
    item without re-emitting its full path or text. Single letters are
    typically one token in modern tokenizers — the point of giving each
    item a stable ID is to ask the model for ``{id → bbox}`` pairs that
    we patch back into the values tree in code, instead of asking the
    model to echo the whole structure."""

    id: str
    path: LeafPath
    text: str
    page_number: int


@dataclass
class MatchStats:
    matched_locations: int = 0  # phase-1 page entries we resolved here
    unmatched_locations: int = 0  # entries left for phase 3
    total_locations: int = 0
    duration_s: float = 0.0


@dataclass
class Phase2Result:
    """Output of :func:`run_phase2_matching`.

    ``values`` is a fresh copy of the phase-1 tree with bounding boxes
    filled in for every location the matcher resolved. Unresolved
    locations are kept as ``{page_number}`` entries (no bbox) so the
    caller can identify what phase 3 owes us. ``unmatched`` enumerates
    those, with stable ids the phase-3 LLM can address.
    """

    values: dict[str, Any]
    unmatched: list[UnmatchedItem]
    stats: MatchStats


# ---- Span → grounded location conversion ----------------------------------


def _span_to_locations(
    span: tuple[int, int],
    page_words: list[_Word],
    dims: _PageDims,
    page_number: int,
) -> list[dict[str, Any]]:
    """Convert a matched span to grounded_field location entries (one per
    visual line). Bboxes are integer image pixels ``[left, top, right,
    bottom]`` — the project-wide convention. ``dims`` is accepted for a
    uniform signature with the other span helpers but is unused because
    boxes stay in pixel space."""
    del dims  # boxes are emitted in pixels; no normalization needed
    start, end = span
    span_words = page_words[start:end]
    locs: list[dict[str, Any]] = []
    for line in _line_groups(span_words):
        left = min(w.left for w in line)
        top = min(w.top for w in line)
        right = max(w.right for w in line)
        bottom = max(w.bottom for w in line)
        locs.append(
            {
                "page_number": page_number,
                "bounding_box": [left, top, right, bottom],
            }
        )
    return locs


def _bbox_norm(loc: dict[str, Any], dims: _PageDims) -> tuple[float, float, float, float]:
    """Normalize a stored pixel ``[left, top, right, bottom]`` bbox to the
    internal ``(ymin, xmin, ymax, xmax)`` 0-1000 space the disambiguation
    heuristics reason in (see the module docstring). ``dims`` are the
    bbox's own page dimensions in pixels."""
    left, top, right, bottom = loc["bounding_box"]
    h = dims.height or 1
    w = dims.width or 1
    return (
        (top / h) * 1000,
        (left / w) * 1000,
        (bottom / h) * 1000,
        (right / w) * 1000,
    )


# ---- Walking helpers ------------------------------------------------------


def is_computed_leaf(value: dict[str, Any]) -> bool:
    """True for a computed_field-shaped leaf: a value derived by reasoning
    (``computed``/``derived_from`` keys) rather than read off the page
    (``locations``). Computed leaves carry no page locations, so the
    matcher and phase 3 never touch them."""
    return "locations" not in value and ("computed" in value or "derived_from" in value)


def _walk_leaves(values: Any, path: LeafPath = ()) -> Iterable[tuple[LeafPath, dict[str, Any]]]:
    """Yield ``(path, leaf)`` for every grounded_field-shaped leaf.

    A leaf is recognized by having both ``text`` and ``locations`` keys
    on a dict. Walking stops descending once a leaf is recognized so
    nested dicts inside the locations array don't confuse the walker.
    Computed leaves (see :func:`is_computed_leaf`) are skipped entirely —
    they have nothing to ground."""
    if isinstance(values, dict):
        if "text" in values and "locations" in values:
            yield path, values
            return
        if "text" in values and is_computed_leaf(values):
            return
        for k, v in values.items():
            yield from _walk_leaves(v, path + (k,))
    elif isinstance(values, list):
        for i, v in enumerate(values):
            yield from _walk_leaves(v, path + (i,))


def walk_computed_leaves(
    values: Any, path: LeafPath = ()
) -> Iterable[tuple[LeafPath, dict[str, Any]]]:
    """Yield ``(path, leaf)`` for every computed_field-shaped leaf — the
    complement of :func:`_walk_leaves` over the same tree."""
    if isinstance(values, dict):
        if "text" in values and "locations" in values:
            return
        if "text" in values and is_computed_leaf(values):
            yield path, values
            return
        for k, v in values.items():
            yield from walk_computed_leaves(v, path + (k,))
    elif isinstance(values, list):
        for i, v in enumerate(values):
            yield from walk_computed_leaves(v, path + (i,))


def _get_at_path(values: dict[str, Any], path: LeafPath) -> Any:
    cur: Any = values
    for seg in path:
        cur = cur[seg]
    return cur


def path_to_str(path: LeafPath) -> str:
    """Render a path as ``a.b[0].c`` for human/LLM consumption."""
    parts: list[str] = []
    for seg in path:
        if isinstance(seg, int):
            parts.append(f"[{seg}]")
        else:
            parts.append(f".{seg}" if parts else str(seg))
    return "".join(parts)


_PATH_SEG_RE = re.compile(r"\.?([^.\[\]]+)|\[(\d+)\]")


def parse_path(text: str) -> LeafPath | None:
    """Inverse of :func:`path_to_str` — ``"a.b[0].c"`` → ``("a", "b", 0, "c")``.

    Computed leaves reference their sources with these dotted paths
    (``derived_from``); this parses them back for resolution against the
    values tree. Returns ``None`` for anything malformed rather than
    raising — a model-fumbled path is dropped, not fatal."""
    text = text.strip()
    if not text:
        return None
    segs: list[Any] = []
    pos = 0
    while pos < len(text):
        m = _PATH_SEG_RE.match(text, pos)
        if m is None:
            return None
        segs.append(int(m.group(2)) if m.group(2) is not None else m.group(1))
        pos = m.end()
    return tuple(segs)


def get_at_path(values: dict[str, Any], path: LeafPath) -> Any:
    """Resolve *path* against a values tree; ``None`` when it dangles."""
    cur: Any = values
    for seg in path:
        if isinstance(cur, dict) and isinstance(seg, str):
            cur = cur.get(seg)
        elif isinstance(cur, list) and isinstance(seg, int) and 0 <= seg < len(cur):
            cur = cur[seg]
        else:
            return None
    return cur


# ---- Phase 2 entry point --------------------------------------------------


@dataclass
class _Task:
    """One ``(leaf, page)`` matching slot.

    ``loc_index`` is the slot's position in the leaf's phase-1
    locations array — needed so we can rebuild the array in the right
    order at the end (matches expand into N entries if the text wraps
    across visual lines).

    ``matched_span`` records the OCR-word span that was committed, so
    the row-coherence audit can release it from ``claimed_spans`` when
    it demotes the task back to unmatched."""

    path: LeafPath
    page_number: int
    loc_index: int
    text: str
    candidates: list[tuple[int, int]] = field(default_factory=list)
    matched_locations: list[dict[str, Any]] | None = None
    matched_span: tuple[int, int] | None = None


def run_phase2_matching(
    workspace: Workspace,
    file_id: str,
    phase1_values: dict[str, Any],
    *,
    layout: dict[str, Any] | None = None,
) -> Phase2Result:
    """Top-level phase-2 entry. See module docstring for the algorithm.

    ``layout`` is the optional phase-1-emitted layout descriptor (see
    :func:`dgml.grounded.extract_values` for the prompt that produces
    it). When provided, the matcher runs a final pass after row + column
    anchors: for each array's row, it sorts unresolved tasks by their
    layout column index and assigns same-row candidates left-to-right,
    skipping x positions already occupied by resolved siblings. This
    handles the pathological case where every value in a column
    collides (e.g. ledger rows where ``post_date == due_date`` for
    every transaction), which prevents column-anchor bootstrap."""
    started = time.monotonic()
    out_values = copy.deepcopy(phase1_values)

    word_cache: dict[int, tuple[list[_Word], _PageDims]] = {}

    tasks: list[_Task] = []
    for path, leaf in _walk_leaves(out_values):
        text = leaf.get("text")
        locs = leaf.get("locations")
        if not isinstance(text, str) or not isinstance(locs, list):
            continue
        for i, loc in enumerate(locs):
            if not isinstance(loc, dict):
                continue
            pn = loc.get("page_number")
            if not isinstance(pn, int):
                continue
            tasks.append(_Task(path=path, page_number=pn, loc_index=i, text=text))

    if not tasks:
        return Phase2Result(
            values=out_values,
            unmatched=[],
            stats=MatchStats(duration_s=round(time.monotonic() - started, 4)),
        )

    # Find candidate spans per task. A page with no OCR (e.g. file added
    # without --text-mode) just yields empty candidates and falls through
    # to phase 3. Exact matching runs first; only when it finds nothing do
    # we fall back to fuzzy (character-class weighted) matching, which adds
    # recall for OCR boundary noise (a trailing colon, a line-break hyphen)
    # without ever overriding or reordering an exact match.
    for t in tasks:
        if t.page_number not in word_cache:
            try:
                word_cache[t.page_number] = _load_page_words(workspace, file_id, t.page_number)
            except FileNotFound:
                word_cache[t.page_number] = ([], _PageDims(1, 1))
        words, _ = word_cache[t.page_number]
        t.candidates = _find_spans(t.text, words)
        if not t.candidates:
            t.candidates = _find_fuzzy_spans(t.text, words)

    # Run the disambiguation loop, then a row-coherence audit; if the
    # audit demotes any off-row match, re-run disambiguation with the
    # band updated. Bounded by ``MAX_AUDIT_PASSES`` so we can't ping
    # back and forth indefinitely on pathological data.
    #
    # ``claimed_spans`` tracks every span the matcher has awarded so a
    # second task with overlapping candidates can't take the same one.
    # Keyed by page because spans are word-index ranges over a *page's*
    # OCR list — span (139, 140) on page 2 and page 3 refer to
    # different words, so they must NOT collide in this set.
    # ``_audit_row_coherence`` releases spans back into the pool when
    # it demotes the owning task.
    claimed_spans: dict[int, set[tuple[int, int]]] = {}
    audit_iters_remaining = _MAX_AUDIT_PASSES
    while True:
        _run_disambiguation_loop(
            tasks=tasks,
            word_cache=word_cache,
            claimed_spans=claimed_spans,
            layout=layout,
        )
        if audit_iters_remaining <= 0:
            break
        if not _audit_row_coherence(tasks, word_cache, claimed_spans, layout):
            break
        audit_iters_remaining -= 1

    # Materialize: rewrite each leaf's locations array so matched entries
    # expand into their per-line locations, unmatched entries stay as
    # page-only placeholders.
    matched_count = 0
    by_leaf: dict[LeafPath, list[_Task]] = {}
    for t in tasks:
        by_leaf.setdefault(t.path, []).append(t)
    for path, ts in by_leaf.items():
        leaf = _get_at_path(out_values, path)
        new_locs: list[dict[str, Any]] = []
        for t in sorted(ts, key=lambda t: t.loc_index):
            if t.matched_locations is not None:
                new_locs.extend(t.matched_locations)
                matched_count += 1
            else:
                new_locs.append({"page_number": t.page_number})
        leaf["locations"] = new_locs

    # Phase-3 ids reset per page so each phase-3 call has a clean
    # namespace (single letters stay one token, mostly).
    unmatched: list[UnmatchedItem] = []
    page_counters: dict[int, int] = {}
    for t in tasks:
        if t.matched_locations is not None:
            continue
        cnt = page_counters.get(t.page_number, 0)
        page_counters[t.page_number] = cnt + 1
        unmatched.append(
            UnmatchedItem(
                id=_short_id(cnt),
                path=t.path,
                text=t.text,
                page_number=t.page_number,
            )
        )

    return Phase2Result(
        values=out_values,
        unmatched=unmatched,
        stats=MatchStats(
            matched_locations=matched_count,
            unmatched_locations=len(unmatched),
            total_locations=len(tasks),
            duration_s=round(time.monotonic() - started, 4),
        ),
    )


# ---- Row-band disambiguation ----------------------------------------------
#
# Previously every disambiguation pass used its own y-tolerance (1.5 line
# heights for the sibling pass, 3 line heights for the layout pass). For
# ledgers with tight ~10-unit row spacing that's loose enough to accept
# candidates from the row above or below — which is what the matcher was
# routinely doing for repeated date values across rows.
#
# The row band replaces those ad-hoc tolerances with a single concept:
# for each row (each array entry), the y-band is the actual union of
# every matched sibling's bbox plus a small per-page slack. A candidate
# must fall inside the band to be considered for that row. This is
# automatically the right size — single-line rows get tight bands,
# rows with wrapped cells get larger bands without us hand-tuning a
# tolerance.


def _row_path(path: LeafPath) -> LeafPath | None:
    """Path identifying the 'row' a leaf belongs to: the path up to and
    including the rightmost array index. ``('transactions', 5, 'post_date')``
    → ``('transactions', 5)``.

    Returns ``None`` for leaves with no array index in their path. A row
    band exists to keep one array row's cells aligned on a visual line; a
    leaf with no array peers has no such row and must not inherit a band.
    The earlier ``path[:-1]`` fallback made every top-level scalar share
    the empty pseudo-row ``()`` — so the first such field to match set a
    y-band that drove every other top-level field's lone candidate to
    zero, sending otherwise-unambiguous values to phase 3."""
    for i in range(len(path) - 1, -1, -1):
        if isinstance(path[i], int):
            return path[: i + 1]
    return None


def _build_row_bands(
    tasks: list[_Task],
    word_cache: dict[int, tuple[list[_Word], _PageDims]],
) -> dict[tuple[LeafPath, int], tuple[float, float]]:
    """Per-row y-band ``(ymin, ymax)`` in normalized 0-1000, built from
    every currently matched sibling's bbox. Slack is applied at
    check-time, not here, so the band stays a pure data extent."""
    bands: dict[tuple[LeafPath, int], tuple[float, float]] = {}
    for s in tasks:
        if s.matched_locations is None:
            continue
        rp = _row_path(s.path)
        if rp is None:
            continue
        key = (rp, s.page_number)
        _, dims = word_cache[s.page_number]
        for loc in s.matched_locations:
            ymin, _, ymax, _ = _bbox_norm(loc, dims)
            cur = bands.get(key)
            if cur is None:
                bands[key] = (ymin, ymax)
            else:
                bands[key] = (min(cur[0], ymin), max(cur[1], ymax))
    return bands


def _row_band_slack_norm(words: list[_Word], dims: _PageDims) -> float:
    """Per-page slack for row-band membership: roughly a third of a
    median word height, but never under 5 normalized units. Small
    enough to reject the next row over, generous enough to absorb OCR
    jitter on the row's own siblings."""
    if dims.height <= 0 or not words:
        return 5.0
    heights = sorted(w.height for w in words if w.height > 0)
    median_h = heights[len(heights) // 2] if heights else 12.0
    return max((median_h * 0.3 / dims.height) * 1000, 5.0)


def _candidate_y_center_norm(
    span: tuple[int, int],
    words: list[_Word],
    dims: _PageDims,
) -> float | None:
    """y_center of a candidate span in normalized 0-1000 space."""
    if dims.height <= 0:
        return None
    start, end = span
    cand_words = words[start:end]
    if not cand_words:
        return None
    ymin = min(w.top for w in cand_words)
    ymax = max(w.bottom for w in cand_words)
    return ((ymin + ymax) / 2 / dims.height) * 1000


def _candidate_in_row_band(
    span: tuple[int, int],
    words: list[_Word],
    dims: _PageDims,
    band: tuple[float, float] | None,
    slack: float,
) -> bool:
    """True if ``span``'s y_center sits inside ``band`` (with slack).
    When the band is None — no sibling yet matched on this row — the
    filter is permissive: the candidate hasn't been disqualified by
    row geometry, only by whatever else the caller chooses to enforce."""
    if band is None:
        return True
    y = _candidate_y_center_norm(span, words, dims)
    if y is None:
        return False
    return band[0] - slack <= y <= band[1] + slack


# ---- Column-anchor disambiguation -----------------------------------------
#
# When a leaf has multiple same-row candidates that the sibling-row anchor
# can't break, look at how the same leaf landed in *other* entries of the
# same array. If those peer matches cluster tightly in x, the cluster is
# empirically a column — use its position to pick the right candidate.
#
# Three alignment styles show up in practice and we don't know which a
# given column uses, so the inference picks per-column: whichever of
# ``xmin`` / ``x_center`` / ``xmax`` has the tightest spread across peer
# matches becomes that column's alignment edge. Left-aligned text columns
# share ``xmin``; right-aligned currency columns share ``xmax``; centered
# columns share ``x_center``. Free-form arrays (a signatures list at
# scattered positions) blow past the spread threshold on all three edges
# and the heuristic correctly declines to anchor.


# Threshold for "tight enough to be a column" — about 3% of the
# normalized 0-1000 page width. Real columns spread by a few units
# (font kerning, OCR jitter); free-form blocks spread by 100+ units.
# 30 sits well inside the gap.
_COLUMN_SPREAD_LIMIT = 30.0

# Minimum tolerance for matching a candidate to a column anchor.
# A perfectly-aligned column still needs slack for the candidate's own
# pixel-jitter when its edge is compared to the median.
_COLUMN_TOL_MIN = 10.0

# Cap on candidates an unmatched task may contribute to the row-coherence
# audit's evidence (see _audit_row_coherence). Tasks with more candidates
# are too ambiguous to vote on their row's y — including all of them lets
# a column of repeated values drown out the matched-task evidence.
_AUDIT_MAX_CANDIDATE_EVIDENCE = 3


@dataclass(frozen=True)
class _ColumnAnchor:
    """Column position inferred from confirmed peer matches.

    ``edge`` records which of the three candidate alignments fit best;
    callers compare the same edge of their candidate to ``value`` (both
    in normalized 0-1000 space)."""

    edge: str  # 'xmin' | 'x_center' | 'xmax'
    value: float
    tol: float


def _column_key(path: LeafPath) -> LeafPath | None:
    """Identify the 'column' a leaf belongs to: same path with every
    array index replaced by a ``*`` sentinel. ``transactions[0].post_date``
    and ``transactions[5].post_date`` share key
    ``("transactions", "*", "post_date")``.

    Returns ``None`` for paths with no array index — column-anchor
    inference only applies when the schema actually has peers."""
    if not any(isinstance(seg, int) for seg in path):
        return None
    return tuple("*" if isinstance(seg, int) else seg for seg in path)


_COLUMN_CLUSTER_RADIUS = 8.0  # normalized 0-1000 units — words within a
# column rarely differ in xmin by more than a few units. A typical
# column gap is much wider (30+).
_COLUMN_MAJORITY_THRESHOLD = 0.4  # the dominant cluster must hold at least
# this fraction of peer matches to anchor — otherwise we have two
# legitimate clusters (the field shows up in two columns due to OCR
# errors poisoning the bootstrap) and the anchor would be ambiguous.


def _build_column_anchors(
    tasks: list[_Task],
    word_cache: dict[int, tuple[list[_Word], _PageDims]],
    layout: dict[str, Any] | None = None,
) -> dict[LeafPath, _ColumnAnchor]:
    """Group resolved tasks by column key, pick the tightest of
    ``{xmin, x_center, xmax}`` as the alignment edge, and emit an anchor
    whose value is the centroid of the *dominant cluster* of peer matches.

    Column anchors only build for arrays the phase-1 ``layout`` hint
    marks ``kind: "table"``. Layout classification is phase 1's job — it
    sees the page and knows whether an array is a grid. Guessing from
    x-clustering alone would misfire: list-shaped (bulleted, vertically
    stacked) arrays trigger the same x-cluster signal as a real column
    and a global anchor would then reject the array's items in other
    sections at a different indent. Trusting the explicit hint eliminates
    that misallocation; arrays without ``kind:"table"`` resolve via the
    row band and the reading-order fallback (which scope per-element
    rather than across the page).

    Why dominant-cluster instead of plain median: when OCR errors push
    a fraction of early phase-2 single-candidate matches to the wrong
    column (e.g. due_date forced to Post col because Due col reads
    ``Jun 01. 2025``), a global median splits the difference and the
    resulting anchor is wrong for *every* row. Clustering with a tight
    radius lets the majority of correctly-placed peers dominate; a
    minority of misplaced peers shows up as a separate cluster and is
    ignored. If neither cluster commands a clear majority, we decline
    the anchor (no signal is better than a contaminated signal —
    layout pass takes over)."""
    by_col: dict[LeafPath, list[tuple[float, float, float]]] = {}
    for t in tasks:
        if t.matched_locations is None:
            continue
        key = _column_key(t.path)
        if key is None:
            continue
        # Only build column anchors for arrays phase 1 marks as tables.
        if not _is_table_array(t.path, layout):
            continue
        _, dims = word_cache[t.page_number]
        for loc in t.matched_locations:
            _, xmin, _, xmax = _bbox_norm(loc, dims)
            by_col.setdefault(key, []).append((xmin, (xmin + xmax) / 2, xmax))

    anchors: dict[LeafPath, _ColumnAnchor] = {}
    for key, samples in by_col.items():
        if len(samples) < 2:
            continue
        best_edge: tuple[str, list[float], float] | None = None
        for edge_name, idx in [("xmin", 0), ("x_center", 1), ("xmax", 2)]:
            xs = sorted(s[idx] for s in samples)
            cluster = _dominant_cluster(xs, _COLUMN_CLUSTER_RADIUS)
            # A "cluster" of one observation isn't a cluster — wait
            # for at least two peers to agree before trusting an
            # anchor. This prevents the scattered/free-form layout
            # case from spuriously latching onto the first match.
            if len(cluster) < 2:
                continue
            score = len(cluster) / len(samples)
            if score < _COLUMN_MAJORITY_THRESHOLD:
                continue
            spread = cluster[-1] - cluster[0]
            # Prefer the edge whose dominant cluster is tightest AND
            # captures the most peers — ties broken by spread (smaller
            # is better) then by majority share (bigger is better).
            if best_edge is None or (spread, -score) < (
                best_edge[2],
                -len(best_edge[1]) / len(samples),
            ):
                best_edge = (edge_name, cluster, spread)
        if best_edge is None:
            # No edge had a clear majority cluster — anchor unreliable.
            continue
        edge_name, cluster, spread = best_edge
        median = cluster[len(cluster) // 2]
        anchors[key] = _ColumnAnchor(
            edge=edge_name,
            value=median,
            tol=max(_COLUMN_TOL_MIN, spread * 2),
        )
    return anchors


def _dominant_cluster(sorted_xs: list[float], radius: float) -> list[float]:
    """Largest gap-merge cluster of ``sorted_xs``. Two adjacent values
    join the same cluster when their gap is ``≤ radius``."""
    if not sorted_xs:
        return []
    clusters: list[list[float]] = [[sorted_xs[0]]]
    for x in sorted_xs[1:]:
        if x - clusters[-1][-1] <= radius:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return max(clusters, key=len)


def _candidate_in_column_anchor(
    span: tuple[int, int],
    words: list[_Word],
    dims: _PageDims,
    anchor: _ColumnAnchor,
) -> bool:
    """True if the candidate's alignment edge sits within the column
    anchor's tolerance. Mirrors the geometry the anchor was built from:
    ``anchor.edge`` decides whether to compare xmin / x_center / xmax,
    everything in normalized 0-1000 space.

    Used as a *filter* (not just a tiebreaker) so a single-candidate
    match in a row whose only text-match happens to land in another
    column gets rejected — better to leave it for phase 3 than commit
    to the wrong column."""
    if dims.width <= 0 or not words:
        return False
    start, end = span
    cand_words = words[start:end]
    if not cand_words:
        return False
    xmin_px = min(w.left for w in cand_words)
    xmax_px = max(w.right for w in cand_words)
    edge_px: float
    if anchor.edge == "xmin":
        edge_px = xmin_px
    elif anchor.edge == "xmax":
        edge_px = xmax_px
    else:
        edge_px = (xmin_px + xmax_px) / 2
    edge_norm = (edge_px / dims.width) * 1000
    return abs(edge_norm - anchor.value) <= anchor.tol


# ---- Layout-driven row assignment -----------------------------------------
#
# When every value of a field collides with a same-row neighbour (e.g. a
# ledger where every transaction has post_date == due_date), neither the
# row anchor (every candidate is on the same y-band) nor the column
# anchor (no peer ever resolves to bootstrap a column) can break the tie.
# Phase 1 has already seen the page, though, so we ask it to emit a
# light-weight layout descriptor: per-array, which fields appear as
# columns and in what left-to-right order.
#
# With that hint, each row's unresolved tasks can be assigned to the
# remaining x positions in column order — sorting candidates by x and
# excluding any already claimed by resolved siblings. This is what
# unblocks the "all values collide" pathology.


def _array_layout_key(path: LeafPath) -> str | None:
    """Layout key for a leaf path: the dotted path of the enclosing
    array(s). ``('transactions', 0, 'post_date')`` → ``'transactions'``;
    ``('co', 'contacts', 3, 'name')`` → ``'co.contacts'``;
    ``('outer', 0, 'inner', 2, 'field')`` → ``'outer.inner'`` (array of
    arrays). Returns ``None`` for paths that don't traverse any array."""
    parts: list[str] = []
    saw_index = False
    for seg in path:
        if isinstance(seg, int):
            saw_index = True
            continue
        parts.append(seg)
    if not saw_index:
        return None
    # Drop the trailing leaf field so the key names the enclosing array,
    # not the field itself: ``co.contacts.name`` would be a different
    # thing from ``co.contacts``.
    if parts:
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _is_free_form_array(path: LeafPath, layout: dict[str, Any]) -> bool:
    """True if ``path`` sits inside an array the layout hint marks
    ``free_form`` — a block-structured array (parties, signatures) whose
    elements stack across multiple lines, as opposed to a single-line
    ``table`` row."""
    key = _array_layout_key(path)
    if key is None:
        return False
    entry = layout.get(key)
    return isinstance(entry, dict) and entry.get("kind") == "free_form"


def _is_table_array(path: LeafPath, layout: dict[str, Any] | None) -> bool:
    """True if ``path`` sits inside an array the layout hint marks
    ``kind: "table"``. Used to gate column-x and row-coherence heuristics
    on phase 1's explicit classification rather than phase 2 guessing
    from geometry — list-shaped arrays look columnar (left-aligned bullet
    items) but aren't, and column inference there misallocates."""
    if layout is None:
        return False
    key = _array_layout_key(path)
    if key is None:
        return False
    entry = layout.get(key)
    return isinstance(entry, dict) and entry.get("kind") == "table"


def _disambiguate_by_layout(
    tasks: list[_Task],
    layout: dict[str, Any],
    word_cache: dict[int, tuple[list[_Word], _PageDims]],
    claimed_spans: dict[int, set[tuple[int, int]]],
) -> bool:
    """For each row whose enclosing array has a ``kind: table`` layout,
    assign that row's unresolved tasks to their layout columns. Mutates
    ``tasks`` in place. Returns True iff any task got resolved.

    Also drops the unresolved-tasks-with-≤1-candidate filter once a
    column anchor is in play: with single-candidate matches already
    bound by the column-anchor filter in the main loop, the layout
    pass needs to see those tasks too so an OCR-misspelled cell whose
    only candidate is in the wrong column doesn't slip through here."""
    # Build column anchors once so we can apply the same column-
    # anchor filter the main loop uses, preventing layout pass from
    # awarding a wrong-column candidate just because it's the only one.
    column_anchors = _build_column_anchors(tasks, word_cache, layout)

    # Group unresolved tasks by (array_key, row_path).
    rows: dict[tuple[str, LeafPath], list[_Task]] = {}
    for t in tasks:
        if t.matched_locations is not None:
            continue
        if len(t.candidates) < 1:
            continue
        key = _array_layout_key(t.path)
        if key is None:
            continue
        entry = layout.get(key)
        if not isinstance(entry, dict) or entry.get("kind") != "table":
            continue
        # row_path is everything up to and including the array index.
        # For ``('transactions', 0, 'post_date')`` that's
        # ``('transactions', 0)`` — used to find sibling tasks on the
        # same row.
        for i, seg in enumerate(t.path):
            if isinstance(seg, int):
                row_path = t.path[: i + 1]
                break
        else:
            continue
        rows.setdefault((key, row_path), []).append(t)

    progress = False
    for (array_key, row_path), unresolved in rows.items():
        columns = layout[array_key].get("columns")
        if not isinstance(columns, list):
            continue
        row_assignments = _assign_row_by_columns(
            unresolved=unresolved,
            row_path=row_path,
            all_tasks=tasks,
            layout_columns=columns,
            word_cache=word_cache,
            claimed_spans=claimed_spans,
            column_anchors=column_anchors,
        )
        for task, span in row_assignments:
            words, dims = word_cache[task.page_number]
            task.matched_locations = _span_to_locations(span, words, dims, task.page_number)
            task.matched_span = span
            claimed_spans.setdefault(task.page_number, set()).add(span)
            progress = True
    return progress


def _assign_row_by_columns(
    *,
    unresolved: list[_Task],
    row_path: LeafPath,
    all_tasks: list[_Task],
    layout_columns: list[str],
    word_cache: dict[int, tuple[list[_Word], _PageDims]],
    claimed_spans: dict[int, set[tuple[int, int]]],
    column_anchors: dict[LeafPath, _ColumnAnchor],
) -> list[tuple[_Task, tuple[int, int]]]:
    """For one row's unresolved tasks, pick a candidate span per task
    by sorting unresolved-task candidates left-to-right and matching
    against the layout's column order.

    Returns a list of ``(task, chosen_span)`` pairs — empty if the
    row can't be assigned cleanly. The assignment requires:

    - every unresolved task's field appears in ``layout_columns``;
    - after excluding x positions already covered by resolved siblings
      on the same row, exactly as many distinct x positions remain as
      there are unresolved tasks.

    When either condition fails the row is skipped — better to defer
    to phase 3 than collapse two tasks onto one cell."""
    if not unresolved:
        return []
    page_number = unresolved[0].page_number
    words, dims = word_cache[page_number]
    if dims.width <= 0 or not words:
        return []

    # Index unresolved tasks by their layout column index.
    indexed: list[tuple[int, _Task]] = []
    for t in unresolved:
        field = t.path[-1]
        if not isinstance(field, str):
            return []
        if field not in layout_columns:
            return []
        indexed.append((layout_columns.index(field), t))
    indexed.sort(key=lambda p: p[0])

    # Resolved siblings on the same row supply x ranges that are
    # already claimed (and a y centroid that tells us which row we're
    # on — without it we'd happily pick a Post-col candidate from a
    # different row that happens to share the same text).
    resolved_ranges: list[tuple[float, float]] = []
    row_y_centers: list[float] = []
    for s in all_tasks:
        if s.matched_locations is None:
            continue
        if s.page_number != page_number:
            continue
        if len(s.path) < len(row_path):
            continue
        if s.path[: len(row_path)] != row_path:
            continue
        for loc in s.matched_locations:
            ymin, xmin, ymax, xmax = _bbox_norm(loc, dims)
            resolved_ranges.append((xmin, xmax))
            row_y_centers.append((ymin + ymax) / 2)

    if not row_y_centers:
        # No resolved sibling = no y anchor for this row. Bail; the
        # column anchor in the outer loop is a safer bet than guessing
        # which page-wide occurrence belongs to this row.
        return []
    row_y_norm = sum(row_y_centers) / len(row_y_centers)
    heights = sorted(w.height for w in words if w.height > 0)
    median_h = heights[len(heights) // 2] if heights else 12.0
    # Row tolerance covers wrapped cells: a 3-line cell can be ~3
    # line-heights tall, so allow ~3 median heights either way.
    y_tol = max((median_h * 3.0 / dims.height) * 1000, 15.0)

    def _norm_x_center(span: tuple[int, int]) -> float | None:
        start, end = span
        cand_words = words[start:end]
        if not cand_words:
            return None
        xmin = min(w.left for w in cand_words)
        xmax = max(w.right for w in cand_words)
        return ((xmin + xmax) / 2 / dims.width) * 1000

    def _norm_y_center(span: tuple[int, int]) -> float | None:
        start, end = span
        cand_words = words[start:end]
        if not cand_words:
            return None
        ymin = min(w.top for w in cand_words)
        ymax = max(w.bottom for w in cand_words)
        return ((ymin + ymax) / 2 / dims.height) * 1000

    def _claimed(x_norm: float) -> bool:
        return any(lo <= x_norm <= hi for lo, hi in resolved_ranges)

    # Build the row's candidate x-position pool by deduplicating
    # candidates that sit at the same x (a candidate may appear in
    # multiple tasks' lists when text collides). For each task we keep
    # a list of (cluster_idx, span) pairs keyed by position in
    # ``indexed`` so we can hand back the right span per task.
    dedup_tol = 3.0  # normalized 0-1000; tighter than a typical column gap
    cluster_xs: list[float] = []
    per_task_cands: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in indexed]
    for k, (_ci, t) in enumerate(indexed):
        page_claimed = claimed_spans.get(t.page_number, set())
        # Same column-anchor filter as the main disambiguation loop —
        # without it, a task whose only candidate sits in another
        # column (e.g. ``charge_code`` matched against the
        # description-col prefix because OCR misspelled the real
        # charge cell) would get awarded its lone candidate here.
        anchor = column_anchors.get(_column_key(t.path) or ())
        for span in t.candidates:
            if span in page_claimed:
                continue
            x = _norm_x_center(span)
            if x is None or _claimed(x):
                continue
            y = _norm_y_center(span)
            if y is None or abs(y - row_y_norm) > y_tol:
                # Candidate from a different row that shares this
                # text — not for us.
                continue
            if anchor is not None and not _candidate_in_column_anchor(span, words, dims, anchor):
                # Wrong-column candidate; defer to phase 3 rather than
                # commit to the only available x.
                continue
            cluster_idx: int | None = None
            for i, cx in enumerate(cluster_xs):
                if abs(cx - x) <= dedup_tol:
                    cluster_idx = i
                    break
            if cluster_idx is None:
                cluster_xs.append(x)
                cluster_idx = len(cluster_xs) - 1
            per_task_cands[k].append((cluster_idx, span))

    order = sorted(range(len(cluster_xs)), key=lambda i: cluster_xs[i])
    new_index = {old: new for new, old in enumerate(order)}
    per_task_cands = [[(new_index[ci], sp) for ci, sp in pairs] for pairs in per_task_cands]

    if len(cluster_xs) < len(indexed):
        return []

    # Invariant: layout_columns is provided by phase 1 in VISUAL
    # left-to-right order (the prompt at grounded.py:_LAYOUT_INSTRUCTIONS
    # enforces this). So the K-th unresolved task in layout order should
    # have its candidate at the K-th surviving cluster in visual order.
    # If a vision model ever returns columns in non-visual order we fall
    # through to return [] here and defer the row to phase 3 — that's
    # the conservative choice the docstring promises.
    assignments: list[tuple[_Task, tuple[int, int]]] = []
    for assign_idx, (_layout_idx, t) in enumerate(indexed):
        match = next(
            (sp for ci, sp in per_task_cands[assign_idx] if ci == assign_idx),
            None,
        )
        if match is None:
            return []
        assignments.append((t, match))
    return assignments


def _disambiguate_reading_order(
    tasks: list[_Task],
    word_cache: dict[int, tuple[list[_Word], _PageDims]],
    claimed_spans: dict[int, set[tuple[int, int]]],
    layout: dict[str, Any] | None = None,
) -> bool:
    """Reading-order fallback for non-tabular (article/list) arrays.

    Tables are resolved by column geometry; article-shape arrays — a
    program-of-study's course lists, say — stack one *record* per region,
    fields inline in reading order with no shared column (the credits sit
    at the end of each item at a different x every time). For a field
    whose column never anchored (its values don't cluster in x), restrict
    its candidates to the vertical slice of its own array element —
    bounded above by the element's already-matched fields and below by
    the next element — and commit when exactly one candidate lands in that
    slice. A value like ``5 CR`` that collides 20x across the page is
    unique inside one course's slice. Fields whose column *did* anchor are
    left to column logic, so real ledgers are untouched.

    Returns True iff any task got resolved."""
    column_anchors = _build_column_anchors(tasks, word_cache, layout)

    # Group tasks by (enclosing-array path, page), then by element.
    groups: dict[tuple[LeafPath, int], dict[LeafPath, list[_Task]]] = {}
    for t in tasks:
        rp = _row_path(t.path)
        if rp is None or not isinstance(rp[-1], int):
            continue
        elems = groups.setdefault((rp[:-1], t.page_number), {})
        elems.setdefault(rp, []).append(t)

    progress = False

    # Seed unanchored elements. A field with a single page-wide candidate
    # is unambiguous; if its element has nothing matched yet, a
    # (possibly mis-scoped, cross-section) column anchor shouldn't veto
    # it. Elements that already anchored on some field are left alone, so
    # a genuine wrong-column candidate in an otherwise-resolved row stays
    # rejected by the column filter.
    for (_array_path, page), by_elem in groups.items():
        words, dims = word_cache[page]
        if not words or dims.height <= 0:
            continue
        for ets in by_elem.values():
            if any(t.matched_locations is not None for t in ets):
                continue
            page_claimed = claimed_spans.setdefault(page, set())
            for t in ets:
                live = [c for c in t.candidates if c not in page_claimed]
                if len(live) == 1:
                    span = live[0]
                    t.matched_locations = _span_to_locations(span, words, dims, page)
                    t.matched_span = span
                    page_claimed.add(span)
                    progress = True

    for (_array_path, page), by_elem in groups.items():
        words, dims = word_cache[page]
        if not words or dims.height <= 0:
            continue
        # Each element's top y, from whatever fields already matched.
        elem_top: dict[LeafPath, float] = {}
        for ek, ets in by_elem.items():
            ys = [
                ((nb := _bbox_norm(loc, dims))[0] + nb[2]) / 2
                for t in ets
                if t.matched_locations is not None
                for loc in t.matched_locations
            ]
            if ys:
                elem_top[ek] = min(ys)
        if not elem_top:
            continue  # no anchored element in this array on this page

        ordered = sorted(elem_top.items(), key=lambda kv: kv[1])
        slack = _row_band_slack_norm(words, dims)
        for i, (ek, top_y) in enumerate(ordered):
            # Region bounds use the nearest neighbor with a *distinct* y,
            # not just the previous/next index. Sibling elements on the
            # same visual line (an "or"-group: ``X or Y or Z`` packed onto
            # one bullet) all anchor at the same y; without this, each
            # one's downward region collapses to zero and the wrap-line
            # content below the group falls into nobody's region. Going
            # upward, the boundary is the midpoint with the previous
            # distinct anchor (or page top); going downward, the next
            # distinct anchor.
            region_top = max(0.0, top_y - slack)
            for j in range(i - 1, -1, -1):
                if ordered[j][1] < top_y - slack:
                    region_top = (ordered[j][1] + top_y) / 2
                    break
            region_bot = 1000.0 + slack
            for j in range(i + 1, len(ordered)):
                if ordered[j][1] > top_y + slack:
                    region_bot = ordered[j][1]
                    break
            # Proximity window around the element's own anchor — a couple
            # line-heights' worth, capturing wrap-line content of the
            # element but not bullets several lines below. Used as a
            # secondary filter when the broad region itself produces
            # multiple candidates (e.g. an article-shape page where the
            # phase-1 schema places the next sibling on another page, so
            # the broad region runs to the page bottom but the field's
            # real location is one line below its anchor).
            heights = sorted(w.height for w in words if w.height > 0)
            median_h = heights[len(heights) // 2] if heights else 12.0
            line_h = median_h / dims.height * 1000 if dims.height > 0 else 12.0
            proximity = max(line_h * 2, 10.0)
            for t in by_elem[ek]:
                if t.matched_locations is not None:
                    continue
                if column_anchors.get(_column_key(t.path) or ()) is not None:
                    continue  # a real column governs this field — leave it
                page_claimed = claimed_spans.get(page, set())
                in_region = [
                    c
                    for c in t.candidates
                    if c not in page_claimed
                    and (yc := _candidate_y_center_norm(c, words, dims)) is not None
                    and region_top <= yc < region_bot
                ]
                chosen: tuple[int, int] | None = None
                if len(in_region) == 1:
                    chosen = in_region[0]
                elif len(in_region) > 1:
                    # Multiple region candidates — narrow to those within
                    # a couple line-heights of this element's anchor.
                    near = [
                        c
                        for c in in_region
                        if (yc := _candidate_y_center_norm(c, words, dims)) is not None
                        and abs(yc - top_y) <= proximity
                    ]
                    if len(near) == 1:
                        chosen = near[0]
                if chosen is not None:
                    t.matched_locations = _span_to_locations(chosen, words, dims, page)
                    t.matched_span = chosen
                    claimed_spans.setdefault(page, set()).add(chosen)
                    progress = True
    return progress


def _element_region(
    t: _Task,
    tasks: list[_Task],
    word_cache: dict[int, tuple[list[_Word], _PageDims]],
) -> tuple[float, float] | None:
    """The reading-order vertical region of ``t``'s array element.

    Bounded above and below by the nearest *distinct-y* neighboring
    element anchors in the same enclosing array (siblings on the same
    visual line — "or"-groups — are skipped, so their wrap line below
    falls inside everyone's region). Returns ``None`` when ``t`` isn't
    in an array, when its element isn't anchored on the page yet, or
    when the page has no usable words."""
    rp = _row_path(t.path)
    if rp is None or not isinstance(rp[-1], int):
        return None
    if t.page_number not in word_cache:
        return None
    words, dims = word_cache[t.page_number]
    if not words or dims.height <= 0:
        return None
    array_path = rp[:-1]
    elem_top: dict[LeafPath, float] = {}
    for s in tasks:
        srp = _row_path(s.path)
        if srp is None or not isinstance(srp[-1], int):
            continue
        if srp[:-1] != array_path or s.page_number != t.page_number:
            continue
        if s.matched_locations is None:
            continue
        for loc in s.matched_locations:
            ymin, _, ymax, _ = _bbox_norm(loc, dims)
            y = (ymin + ymax) / 2
            cur = elem_top.get(srp)
            if cur is None or y < cur:
                elem_top[srp] = y
    if rp not in elem_top:
        return None
    ordered = sorted(elem_top.items(), key=lambda kv: kv[1])
    idx = next(i for i, (k, _) in enumerate(ordered) if k == rp)
    top_y = ordered[idx][1]
    slack = _row_band_slack_norm(words, dims)
    region_top = max(0.0, top_y - slack)
    for j in range(idx - 1, -1, -1):
        if ordered[j][1] < top_y - slack:
            region_top = (ordered[j][1] + top_y) / 2
            break
    region_bot = 1000.0 + slack
    for j in range(idx + 1, len(ordered)):
        if ordered[j][1] > top_y + slack:
            region_bot = ordered[j][1]
            break
    return region_top, region_bot


def _disambiguate_share_duplicate_text(
    tasks: list[_Task],
    word_cache: dict[int, tuple[list[_Word], _PageDims]],
) -> bool:
    """Let two schema entries with the *same text* share one page span.

    Documents render "or"-group alternatives — say two course rows that
    share a single ``5 CR`` token — once, but the schema represents each
    alternative as its own array element with the credit value
    duplicated. The claimed-span discipline that protects against
    wrong-column double-grabs leaves the second task unmatched once the
    first claims the only render. This pass relaxes that for the case
    we know is safe: an unmatched task adopts an already-matched task's
    location only when both have *exactly* the same text **and** the
    matched task's bbox falls inside the unmatched task's reading-order
    region (between the nearest distinct-y element anchors above and
    below). Using the broader region — not the tight row band — captures
    the multi-line case where the shared cell sits on the wrap line
    below the "or"-group's first line. The same-text check keeps
    wrong-column captures (different text by construction) out of scope.
    Returns True iff any task got resolved."""
    matched_by_text: dict[tuple[str, int], list[_Task]] = {}
    for t in tasks:
        if t.matched_locations is None:
            continue
        matched_by_text.setdefault((t.text, t.page_number), []).append(t)
    if not matched_by_text:
        return False

    progress = False
    for t in tasks:
        if t.matched_locations is not None:
            continue
        region = _element_region(t, tasks, word_cache)
        if region is None:
            continue
        region_top, region_bot = region
        _, dims = word_cache[t.page_number]
        viable: list[_Task] = []
        for m in matched_by_text.get((t.text, t.page_number), ()):
            for loc in m.matched_locations or ():
                ymin, _, ymax, _ = _bbox_norm(loc, dims)
                yc = (ymin + ymax) / 2
                if region_top <= yc < region_bot:
                    viable.append(m)
                    break
        if len(viable) == 1:
            m = viable[0]
            assert m.matched_locations is not None
            t.matched_locations = [dict(loc) for loc in m.matched_locations]
            t.matched_span = m.matched_span
            progress = True
    return progress


# ---- Phase-2 outer loop + row-coherence audit -----------------------------
#
# The audit pass catches matches that slipped through the up-front
# disambiguation but visibly belong to a different row — e.g. a
# ``charge_amount`` snapped to the monthly subtotal at the page bottom,
# or a ``post_date`` dragged a row up because the same date string was
# the only candidate after a sibling's OCR-misread blocked others. After
# every disambiguation pass converges, the audit recomputes each row's
# y-band from its matches and demotes any leaf whose bbox sits outside.
# A demoted leaf goes back to the pool with its span released, and we
# rerun disambiguation; the cycle is bounded by ``_MAX_AUDIT_PASSES``.

_MAX_AUDIT_PASSES = 3


def _run_disambiguation_loop(
    *,
    tasks: list[_Task],
    word_cache: dict[int, tuple[list[_Word], _PageDims]],
    claimed_spans: dict[int, set[tuple[int, int]]],
    layout: dict[str, Any] | None,
) -> None:
    """Iterate single-candidate / column-anchor / layout disambiguation
    until no task changes.

    Row identity is enforced via the row-band filter applied before
    each task's chosen pass. The band is updated *inline* as matches
    accumulate so a task processed later in the same iteration sees
    its sibling's bbox and rejects candidates from other rows — this
    is the difference between "match all rows fine" and "first leaf's
    OCR-misread candidate locks every subsequent leaf to the wrong
    row."
    """
    progress = True
    while progress:
        progress = False
        column_anchors = _build_column_anchors(tasks, word_cache, layout)
        row_bands = _build_row_bands(tasks, word_cache)
        resolved_spans_by_row = _build_resolved_spans_by_row(tasks)
        for t in tasks:
            if t.matched_locations is not None:
                continue
            words, dims = word_cache[t.page_number]
            slack = _row_band_slack_norm(words, dims)
            rp = _row_path(t.path)
            band_key = (rp, t.page_number) if rp is not None else None
            band = row_bands.get(band_key) if band_key is not None else None
            sibling_spans = resolved_spans_by_row.get(band_key, ()) if band_key is not None else ()
            # Block-structured arrays (the layout hint calls them
            # ``free_form``) stack one element's fields across several
            # lines — a party block with its address on the line below
            # the name, a signature block with the name above the title.
            # The single-line row band, built from whichever field
            # matched first, would reject every sibling on a different
            # line. Drop the y-band for those arrays; the claimed-span
            # and sibling-overlap checks still prevent double assignment,
            # and tabular arrays keep their band intact.
            if band is not None and layout and _is_free_form_array(t.path, layout):
                band = None
            page_claimed = claimed_spans.get(t.page_number, set())
            live = [
                c
                for c in t.candidates
                if c not in page_claimed
                and not _span_overlaps_any(c, sibling_spans)
                and _candidate_in_row_band(c, words, dims, band, slack)
            ]
            # Column anchor filter: when a column has settled into a
            # known x position from prior matches, every candidate for
            # tasks in that column must agree with it — including
            # single-candidate matches. Otherwise an OCR-typo'd cell
            # (e.g. ``Utilitles`` in the charge_code column) leaves
            # only its row's *description*-column prefix as a candidate,
            # and a naive single-candidate match would lock the field
            # to the wrong column. With this filter, those go to phase
            # 3 (which has the page image and can pick visually).
            anchor = column_anchors.get(_column_key(t.path) or ())
            if anchor is not None:
                live = [c for c in live if _candidate_in_column_anchor(c, words, dims, anchor)]
            chosen: tuple[int, int] | None = None
            if len(live) == 1:
                chosen = live[0]
            # Multi-candidate after band+sibling+column-anchor filters
            # is genuinely ambiguous on x and y both — leave for the
            # layout pass to handle.
            if chosen is not None:
                t.matched_locations = _span_to_locations(chosen, words, dims, t.page_number)
                t.matched_span = chosen
                claimed_spans.setdefault(t.page_number, set()).add(chosen)
                # Incremental updates: later tasks in this same
                # iteration see the band shrink (locks them to the
                # same row), the sibling-span list grow (can't reuse
                # words), and the column anchor settle (rejects
                # late-matching candidates that drift outside the
                # column's now-confirmed x position).
                if band_key is not None:
                    for loc in t.matched_locations:
                        ymin, _, ymax, _ = _bbox_norm(loc, dims)
                        cur = row_bands.get(band_key)
                        if cur is None:
                            row_bands[band_key] = (ymin, ymax)
                        else:
                            row_bands[band_key] = (min(cur[0], ymin), max(cur[1], ymax))
                    resolved_spans_by_row.setdefault(band_key, []).append(chosen)
                column_anchors = _build_column_anchors(tasks, word_cache, layout)
                progress = True
        if not progress and layout:
            # Anchor passes converged with tasks still pending. Try the
            # layout-driven row assignment as a tiebreaker.
            if _disambiguate_by_layout(tasks, layout, word_cache, claimed_spans):
                progress = True
        if not progress:
            # Last resort: reading-order resolution for non-tabular
            # (article/list) arrays, where column geometry doesn't apply.
            if _disambiguate_reading_order(tasks, word_cache, claimed_spans, layout):
                progress = True
        if not progress:
            # And finally: schema-duplicate text (an "or"-group's credits
            # rendered once but stored twice in the schema) shares a
            # single page span — both entries ground to the same place.
            if _disambiguate_share_duplicate_text(tasks, word_cache):
                progress = True


def _build_resolved_spans_by_row(
    tasks: list[_Task],
) -> dict[tuple[LeafPath, int], list[tuple[int, int]]]:
    """Per-row list of every already-claimed span. Used by the live
    filter so a task can't claim a candidate that's a sub-span of an
    already-resolved sibling on the same row — that's the same trap
    where ``charge_code='Payment'`` would grab the ``Payment`` token
    *inside* a description like ``eCheck Payment ID ...`` instead of
    the standalone token in its own column."""
    out: dict[tuple[LeafPath, int], list[tuple[int, int]]] = {}
    for s in tasks:
        if s.matched_span is None:
            continue
        rp = _row_path(s.path)
        if rp is None:
            continue
        out.setdefault((rp, s.page_number), []).append(s.matched_span)
    return out


def _audit_row_coherence(
    tasks: list[_Task],
    word_cache: dict[int, tuple[list[_Word], _PageDims]],
    claimed_spans: dict[int, set[tuple[int, int]]],
    layout: dict[str, Any] | None = None,
) -> bool:
    """Demote matched leaves whose y disagrees with the row's majority.

    For each row, collect y-evidence from every task: the matched
    bboxes' y-centers, plus the y-centers of every candidate of any
    *unmatched* task in the row. Cluster these (1D, simple
    gap-merging). The biggest cluster wins as the row's "true" y.
    Any matched leaf whose y sits outside that cluster's slack band
    gets demoted.

    The candidate-evidence trick is what catches the
    "first-task-picked-the-wrong-row" failure: if a row's lone
    matched leaf is at y=600 but every other task's only candidate
    sits at y=100, the cluster pass weights three votes for y=100
    against one for y=600 and demotes the y=600 match. The audit
    is conservative when the row has a true tie (two clusters of
    equal size) — better to leave both than guess wrong."""
    if not tasks:
        return False

    # Bucket every y observation by (row_path, page). Only tasks in
    # arrays the phase-1 layout marks ``kind: "table"`` contribute —
    # the audit's single-line row-coherence model assumes a real row,
    # which is exactly what tables have. For free_form / list / null
    # arrays an element's siblings legitimately sit on different lines
    # (a heading line, a body line, a wrap line below), so a dominant-y
    # cluster computed across them would demote correctly-placed
    # matches just for not sharing the heading's y.
    #
    # An unmatched task contributes its candidate y's only when it has
    # few candidates: a value with one or two places it could live is
    # informative about its row's y; a value like ``5 CR`` with 20
    # candidates strewn down a column would otherwise blanket the page
    # with votes and fabricate a fake "dominant" cluster from the column
    # itself.
    row_evidence: dict[tuple[LeafPath, int], list[float]] = {}
    for t in tasks:
        rp = _row_path(t.path)
        if rp is None:
            continue
        if not _is_table_array(t.path, layout):
            continue
        key = (rp, t.page_number)
        words, dims = word_cache[t.page_number]
        if t.matched_locations is not None:
            for loc in t.matched_locations:
                ymin, _, ymax, _ = _bbox_norm(loc, dims)
                row_evidence.setdefault(key, []).append((ymin + ymax) / 2)
        elif len(t.candidates) <= _AUDIT_MAX_CANDIDATE_EVIDENCE:
            page_claimed = claimed_spans.get(t.page_number, set())
            for span in t.candidates:
                if span in page_claimed:
                    continue
                y = _candidate_y_center_norm(span, words, dims)
                if y is not None:
                    row_evidence.setdefault(key, []).append(y)

    # For each row, find the dominant y cluster (if unambiguous).
    dominant_y: dict[tuple[LeafPath, int], float] = {}
    for key, ys in row_evidence.items():
        if not ys:
            continue
        words, dims = word_cache[key[1]]
        slack = _row_band_slack_norm(words, dims)
        # Cluster merge radius: ~3x per-row slack, so adjacent rows
        # stay distinct (typical row spacing is well above this)
        # but a single cell's wrapped lines still merge.
        cluster_radius = slack * 3
        sorted_ys = sorted(ys)
        clusters: list[list[float]] = []
        for y in sorted_ys:
            if clusters and y - clusters[-1][-1] <= cluster_radius:
                clusters[-1].append(y)
            else:
                clusters.append([y])
        biggest = max(clusters, key=len)
        tied = sum(1 for c in clusters if len(c) == len(biggest))
        if tied > 1:
            # No clear majority — uncertain which y is the row's
            # truth. Leave it for phase 3 to sort out.
            continue
        dominant_y[key] = sum(biggest) / len(biggest)

    # Demote any matched leaf whose y sits far from its row's dominant y.
    demoted_any = False
    for t in tasks:
        if t.matched_locations is None:
            continue
        rp = _row_path(t.path)
        if rp is None:
            continue
        # Block-structured (free_form) array elements legitimately span
        # several lines, so "off the row's dominant line" is not evidence
        # of a mismatch here — it's the normal shape. Skip them; demoting
        # would strip the only candidate (see _demote) and lose the match.
        if layout and _is_free_form_array(t.path, layout):
            continue
        key = (rp, t.page_number)
        dom = dominant_y.get(key)
        if dom is None:
            continue
        words, dims = word_cache[t.page_number]
        slack = _row_band_slack_norm(words, dims)
        radius = slack * 3
        # A leaf with multiple matched locations is intrinsically wrapped
        # content (a long paragraph, an "or"-group that spills onto the
        # next line). Its y-extent is not a row coordinate; the audit's
        # single-line row-coherence model doesn't apply to it, so skip
        # demotion. Single-location leaves still get the row-y check —
        # that's where the audit earns its keep, catching a value snapped
        # to the wrong row.
        if len(t.matched_locations) > 1:
            continue
        ymin, _, ymax, _ = _bbox_norm(t.matched_locations[0], dims)
        y_center = (ymin + ymax) / 2
        if abs(y_center - dom) > radius:
            _demote(t, claimed_spans)
            demoted_any = True
    return demoted_any


def _demote(t: _Task, claimed_spans: dict[int, set[tuple[int, int]]]) -> None:
    """Release a task's match back to the unmatched pool.

    Two things happen: the claimed span is released (so another task
    that wants it can take it), and the offending span is removed from
    ``t.candidates`` so the next disambiguation iteration can't
    immediately re-pick the same wrong location. Without that pruning
    a task with one candidate would oscillate forever between matched
    and demoted."""
    if t.matched_span is not None:
        page_set = claimed_spans.get(t.page_number)
        if page_set is not None:
            page_set.discard(t.matched_span)
        t.candidates = [c for c in t.candidates if c != t.matched_span]
    t.matched_span = None
    t.matched_locations = None


def _short_id(n: int) -> str:
    """Map non-negative ints to lowercase letter codes: 0→'a', 25→'z',
    26→'aa', 27→'ab', ...

    Each id is a single token in most tokenizers for n < 26, two tokens
    beyond that — small ids keep phase-3 prompts compact since the model
    only needs to echo the id, not the full path."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    out = ""
    n += 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = letters[rem] + out
    return out


__all__ = [
    "MatchStats",
    "Phase2Result",
    "UnmatchedItem",
    "path_to_str",
    "run_phase2_matching",
]
