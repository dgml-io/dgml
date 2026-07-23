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

"""Shared text-to-OCR matching primitives.

Home of the word-level machinery used by both grounded value matching
(:mod:`dgml.matching`) and whole-document DGML XML grounding
(:mod:`dgml.xml_grounding`):

- :class:`Word` / :class:`PageDims` — one OCR word with its integer
  pixel bbox, and the page dimensions in pixels.
- :func:`load_page_words` — read ``page_text/page_N.json`` and reorder
  words into cell-aware reading order.
- :func:`find_spans` — exact (modulo OCR-confusable punctuation)
  contiguous span search.
- :func:`find_fuzzy_spans` — recall fallback built on a character-class
  weighted edit distance (punctuation nearly free, letters one unit,
  digits expensive).
- :func:`line_groups` — split matched words into visual lines for
  per-line bboxes.

Everything here is geometry- and string-level: no knowledge of the
grounded-values tree or the DGML XML tree. Callers own how matches map
back to their own structures.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import FileNotFound
from .storage import Workspace, read_json

# ---- Data shapes -----------------------------------------------------------


@dataclass(frozen=True)
class Word:
    """One OCR word in image-pixel space (top-left origin), integer pixels.

    The style fields (``bold``/``italic``/``size``/``color``) carry observed
    formatting from the ``"s"`` object in ``page_text`` when present — the
    digital (and digital-derived hybrid) path populates them; OCR words leave
    them at their defaults. ``dg:style`` aggregation in :mod:`dgml.xml_grounding`
    weights each word's facts by its character count."""

    idx: int  # original OCR index, preserved for stability
    text: str
    text_norm: str  # whitespace-collapsed, punctuation-canonicalized form
    left: int
    top: int
    right: int
    bottom: int
    bold: bool = False
    italic: bool = False
    size: float | None = None  # glyph size in PDF points, if observed
    color: str | None = None  # dominant CSS named color, if observed

    @property
    def y_center(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class PageDims:
    width: int
    height: int


# ---- Normalization ----------------------------------------------------------

# OCR routinely confuses or drops minor punctuation that's not
# load-bearing for word identity. We canonicalize each ambiguous form
# in the comparison key used by :func:`find_spans` so a date column
# read as ``Jun 01. 2025`` (period substituted for comma) OR
# ``Jun 16 2025`` (comma dropped) still matches an extracted
# ``Jun 01, 2025``. Substituted digits or dropped letters still fail
# to match — by design: those are large OCR differences that should
# reach a smarter fallback (or stay unmatched) with a clear signal
# that no text-grounding exists.
_OCR_PUNCT_FUZZY = str.maketrans(
    {
        # Comma and period: dropped or swapped equivalently. Removing
        # them in canonical form covers both ``Jun 01. 2025`` (swap)
        # and ``Jun 01 2025`` (drop) without us having to enumerate
        # OCR error modes.
        ",": "",
        ".": "",
        # Semicolon and colon swap but stay distinct from absence:
        # ``:`` is load-bearing for timestamps like ``03:45`` and
        # bare-word "03 45" would otherwise collide.
        ";": ":",
        "\u2014": "-",  # em dash -> hyphen
        "\u2013": "-",  # en dash -> hyphen
        # Minus/hyphen codepoint variants: a digital PDF's negative
        # amount carries U+2212 while OCR (or another extractor) reads
        # ASCII "-" for the same glyph; word identity must not hinge
        # on which codepoint the source happened to use.
        "\u2212": "-",  # minus sign -> hyphen
        "\u2010": "-",  # hyphen -> hyphen-minus
        "\u2011": "-",  # non-breaking hyphen -> hyphen-minus
        "\uff0d": "-",  # fullwidth hyphen-minus -> hyphen-minus
        "\u2018": "'",  # left single quote -> straight
        "\u2019": "'",  # right single quote -> straight
        "\u201c": '"',  # left double quote -> straight
        "\u201d": '"',  # right double quote -> straight
    }
)


def fuzzy_norm(s: str) -> str:
    """Whitespace-collapsed + OCR-confusable-punctuation-canonicalized
    form of ``s``. The single comparison key used by :func:`find_spans`
    — every word's ``text_norm`` is computed this way at load time."""
    return "".join(s.split()).translate(_OCR_PUNCT_FUZZY)


def core_token(s: str) -> str:
    """Alphanumeric-lowercase core of a token. Used as the fuzzy-seeding
    key so a value embedded in a heading with attached punctuation or a
    different case — ``"(75 CREDITS)"`` on the page vs the extracted
    ``"75 CREDITS"`` — still finds its anchor. Seeding only has to
    *locate* the region; the weighted distance (which treats punctuation
    and case as near-free) confirms the actual match."""
    return "".join(ch.lower() for ch in s if ch.isalnum())


# ---- Page-text loading ----------------------------------------------------


def load_page_words(
    workspace: Workspace, file_id: str, page_number: int
) -> tuple[list[Word], PageDims]:
    """Load OCR words and reorder them into cell-aware reading order.

    Why reorder: providers like Azure / Textract serialize a page in a
    pass that often takes the first visual line of every cell across a
    row, *then* the second line of every cell, etc. A cell whose text
    wraps (a narrow date column reading "Apr" / "29," / "2025" on
    three lines) ends up with its three words interleaved with other
    cells' words in the OCR stream. :func:`find_spans` requires
    contiguous-in-OCR-order words to match, so the wrapped cell would
    never resolve.

    The fix: group spatially-connected words into cells via 2D
    connectivity (:func:`_build_cells`) and emit them in
    cell-then-(top-to-bottom, left-to-right) order. Within each cell
    every word is adjacent, so span search finds wrapped text the
    same way it finds single-line text. Word ``.idx`` is preserved so
    callers that care about OCR identity still see the original
    index."""
    text_path = workspace.file_text_dir(file_id) / f"page_{page_number}.json"
    if not text_path.exists():
        raise FileNotFound(
            f"no page_text for file '{file_id}' page {page_number} (expected at {text_path})"
        )
    payload = read_json(text_path)
    dims = PageDims(int(payload["width"]), int(payload["height"]))
    raw_words = payload.get("words", [])
    loaded: list[Word] = []
    for i, w in enumerate(raw_words):
        left, top, right, bottom = w["l"]
        s = w.get("s") or {}
        size = s.get("sz")
        loaded.append(
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
                size=float(size) if size is not None else None,
                color=s.get("c"),
            )
        )
    return reorder_words_by_cell(loaded), dims


# Characters a decorative rule / leader is drawn with. A word of ≥4 of
# these (and nothing else) is a graphic line, not text — signature
# rules, dot leaders, row separators.
# Underscore, hyphen, equals, dots (incl. middle dot U+00B7), and
# en/em/horizontal-bar dashes (U+2013/U+2014/U+2015).
_RULE_CHARS = frozenset("_-=." + "\u00b7\u2013\u2014\u2015")


def _is_rule_word(text: str) -> bool:
    t = "".join(text.split())
    return len(t) >= 4 and all(ch in _RULE_CHARS for ch in t)


def _build_cells(words: list[Word]) -> list[list[int]]:
    """Cluster word indices into cells via 2D spatial connectivity.

    Two words are unified when they are either:

    - on the same visual line (y-overlap > 0) and horizontally adjacent
      (gap <= ~0.8 * median word height — wide enough to absorb the
      space between words, narrow enough that column gutters reject); or
    - in the same column (x-overlap >= ~0.2 * median height) and on
      adjacent lines (vertical gap <= ~0.8 * median height — wrapped
      text within a cell leads at well under one word height, while
      the padding between table *rows* runs a full height or more; a
      looser bound was observed unioning header cells with the data
      row below them, which interleaves both cells' lines in the
      reading order).

    A "cell" is the resulting connected component. Wrapped narrow
    columns connect vertically; one-line wide cells connect
    horizontally; distinct cells in the same row stay separate.

    Rule words — long runs of underscores/dashes/dots (signature lines,
    dot leaders, separators) — are page *decoration*, not text, and
    never union with anything: a signature rule x-overlaps the typed
    name on the line below it and would otherwise capture that name
    into the rule's own (earlier) band, reordering "Julian Kase" as
    "Kase … Julian" in the stream."""
    n = len(words)
    if n == 0:
        return []
    heights = sorted(w.height for w in words if w.height > 0)
    median_h = heights[len(heights) // 2] if heights else 30.0
    h_gap_tol = max(median_h * 0.8, 6.0)
    v_gap_tol = max(median_h * 0.8, 8.0)
    x_overlap_min = max(median_h * 0.2, 2.0)
    rule = [_is_rule_word(w.text) for w in words]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # O(N^2) is fine for pages with a few hundred words; the per-pair
    # check is cheap. Spatial indexing would help on much larger pages.
    for i in range(n):
        if rule[i]:
            continue
        wi = words[i]
        for j in range(i + 1, n):
            if rule[j]:
                continue
            wj = words[j]
            # "Same visual line" needs *substantial* y-overlap, not just
            # touching: in dense tables adjacent rows' word boxes can
            # graze each other vertically, and a >0 test unions a cell
            # with the row below it (observed: a Yes/No column cell
            # absorbing the next row's cell, which then scrambles
            # reading order for the whole row).
            y_overlap = min(wi.bottom, wj.bottom) - max(wi.top, wj.top)
            if y_overlap > 0.5 * min(wi.height, wj.height):
                gap = max(wi.left, wj.left) - min(wi.right, wj.right)
                if gap <= h_gap_tol:
                    union(i, j)
                    continue
            x_overlap = min(wi.right, wj.right) - max(wi.left, wj.left)
            if x_overlap >= x_overlap_min:
                v_gap = max(wi.top, wj.top) - min(wi.bottom, wj.bottom)
                if 0 < v_gap <= v_gap_tol:
                    union(i, j)

    cells: dict[int, list[int]] = {}
    for i in range(n):
        cells.setdefault(find(i), []).append(i)
    return list(cells.values())


def words_form_one_cell(words: list[Word]) -> bool:
    """True when ``words`` form a single spatially-connected cell under
    the same 2D connectivity :func:`_build_cells` uses for reading-order
    cell building. Grounding uses this to validate that a set of words
    assembled *non-contiguously* from the OCR stream (an interleaved
    multi-line table cell) still reads as one place on the page."""
    return len(_build_cells(words)) <= 1


def reorder_words_by_cell(words: list[Word]) -> list[Word]:
    """Emit ``words`` in cell-then-(visual-line, left) order.

    Cells are sorted into reading order by *(row band, left)*: cells
    whose tops sit within half a median word height of each other share
    a band, bands run top-to-bottom, and cells within a band run
    left-to-right. Within a cell, words are grouped into *visual lines*
    (y-center proximity) and each line is sorted left-to-right.

    Naive sorting by ``(top, left)`` doesn't work: OCR can report two
    words on the same visual line with sub-pixel-different tops
    (1402 vs 1403), which makes the higher-top word win the sort even
    if it's to the right of its line-mate. ``find_spans`` then can't
    match a phrase like "Pet Deposit" because the OCR words appear in
    reverse order. The same jitter applies to whole cells — a table
    row's cells must come out left-to-right, not in top-jitter order —
    which is what the band grouping fixes."""
    cells = _build_cells(words)
    if not cells:
        return []
    heights = sorted(w.height for w in words if w.height > 0)
    median_h = heights[len(heights) // 2] if heights else 12.0
    band_tol = max(median_h * 0.5, 2.0)

    keyed = sorted((min(words[i].top for i in c), min(words[i].left for i in c), c) for c in cells)
    banded: list[tuple[int, float, list[int]]] = []
    band_idx = 0
    band_top: float | None = None
    for top, left, cell in keyed:
        if band_top is None or top - band_top > band_tol:
            band_idx += 1
            band_top = top
        banded.append((band_idx, left, cell))
    banded.sort(key=lambda t: (t[0], t[1]))

    out: list[Word] = []
    for _band, _left, cell in banded:
        cell_words = [words[i] for i in cell]
        for line in line_groups(cell_words):
            out.extend(sorted(line, key=lambda w: w.left))
    return out


# ---- Span search ----------------------------------------------------------


def find_spans(text: str, page_words: list[Word]) -> list[tuple[int, int]]:
    """All ``(start, end_exclusive)`` OCR-word spans whose joined text
    matches ``text`` modulo whitespace AND minor OCR-confusable
    punctuation (comma↔period, en/em-dash↔hyphen, smart↔plain quotes).

    The leniency is what catches the ``Jun 01. 2025`` Due-col cell
    when the extracted value reads ``Jun 01, 2025`` — without it the
    only page-level match would be the (correctly comma'd) Post-col
    cell on the same row, and the matcher would pin due_date to the
    wrong column. The fuzzy step doesn't widen to character
    substitutions that change meaning (digits, letters, dropped
    characters) — those still legitimately stay unmatched here.

    The scan is O(N * max-span-length) but bails out early as soon
    as the accumulated prefix diverges from the target."""
    target = fuzzy_norm(text)
    if not target:
        return []
    spans: list[tuple[int, int]] = []
    n = len(page_words)
    for start in range(n):
        # Skip starts whose word has empty norm — pure-punctuation OCR
        # tokens (a stray ``·`` or comma that fuzzy-norms to nothing)
        # otherwise create a phantom span starting one slot earlier than
        # the real one, duplicating every match and leaving the field
        # falsely ambiguous when downstream disambiguation expects spans
        # to correspond to distinct locations.
        if not page_words[start].text_norm:
            continue
        acc = ""
        for end in range(start, n):
            acc += page_words[end].text_norm
            if not target.startswith(acc):
                break
            if len(acc) == len(target):
                spans.append((start, end + 1))
                break
    return spans


def _strip_boundary_punct(s: str, *, leading: bool, trailing: bool) -> str:
    """Drop non-alphanumeric characters from the requested boundary of
    an already-normalized string. Interior characters are untouched."""
    start, end = 0, len(s)
    if leading:
        while start < end and not s[start].isalnum():
            start += 1
    if trailing:
        while end > start and not s[end - 1].isalnum():
            end -= 1
    return s[start:end]


def find_spans_lenient(text: str, page_words: list[Word]) -> list[tuple[int, int]]:
    """Boundary-punctuation-lenient fallback for :func:`find_spans`.

    Exact span search requires every normalized character to line up, so
    two *boundary* artifacts defeat it even when the content words match
    perfectly:

    - the page glues punctuation onto the span's first word (an amount
      typeset as ``($14.77`` tokenizes to ``($`` + digits — the run of
      same-class characters stays one token), hiding the ``$`` the
      target starts with;
    - the target carries boundary punctuation the page simply doesn't
      have (a ``$`` the generator added from column context, a
      ``Label —`` separator the page renders as whitespace or a column
      gap).

    Both are boundary effects, so the leniency stays there: interior
    punctuation must still match exactly, and letters/digits are never
    trimmed — a changed digit still refuses, same as the exact search.
    Target variants are tried strictest-first (untrimmed, then leading /
    trailing / both boundaries trimmed) and the first variant yielding
    any spans wins; within every variant the span's first page word may
    shed leading punctuation ("($" matching as "$"). Callers should
    treat the result exactly like :func:`find_spans` output — candidate
    locations to be scored — but only consult this after both the exact
    and fuzzy searches came back empty."""
    target = fuzzy_norm(text)
    if not target:
        return []
    variants = [target]
    for leading, trailing in ((True, False), (False, True), (True, True)):
        v = _strip_boundary_punct(target, leading=leading, trailing=trailing)
        if v and v not in variants:
            variants.append(v)
    for tgt in variants:
        spans: list[tuple[int, int]] = []
        n = len(page_words)
        for start in range(n):
            w0 = page_words[start].text_norm
            if not w0:
                continue
            # The first word may shed leading punctuation one char at a
            # time — "($" participates as "$"; letters/digits stop the
            # shedding so content is never silently dropped.
            inits = [w0]
            s = w0
            while s and not s[0].isalnum():
                s = s[1:]
                if s and s not in inits:
                    inits.append(s)
            seen_end: set[int] = set()
            for init in inits:
                if not tgt.startswith(init):
                    continue
                if len(init) == len(tgt):
                    if start + 1 not in seen_end:
                        spans.append((start, start + 1))
                        seen_end.add(start + 1)
                    continue
                acc = init
                for end in range(start + 1, n):
                    acc += page_words[end].text_norm
                    if not tgt.startswith(acc):
                        break
                    if len(acc) == len(tgt):
                        if end + 1 not in seen_end:
                            spans.append((start, end + 1))
                            seen_end.add(end + 1)
                        break
        if spans:
            return spans
    return []


def span_overlaps_any(
    span: tuple[int, int],
    others: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> bool:
    """True if ``span`` shares any word index with any span in
    ``others``. Spans are ``(start, end_exclusive)`` ranges over the
    reordered word list, so word-index overlap and bbox sub-region
    overlap mean the same thing."""
    a, b = span
    for c, d in others:
        if not (b <= c or a >= d):
            return True
    return False


# ---- Character-class weighted edit distance -------------------------------
#
# Exact span matching is equality against a fixed punctuation
# normalization table (:func:`fuzzy_norm`), so one stray character — a
# trailing colon, a line-break hyphen, an OCR letter slip — defeats an
# otherwise perfect match. The generalization is a similarity score
# whose edit costs depend on the *class* of character involved rather
# than an enumerated list of confusables:
#
#   - punctuation / whitespace edits are nearly free — the colon, hyphen,
#     and quote cases all collapse into this one rule;
#   - a letter substitution costs a full unit (OCR confuses letters, but
#     it shouldn't take many before it's a different word);
#   - case-only differences are cheap but non-zero, so an exact-case span
#     still scores strictly higher than a case-variant one;
#   - DIGIT edits are expensive, so a changed, inserted, or dropped digit
#     tanks the score and numbers / dates / money / zips stay distinct.
#
# :func:`similarity` normalizes the cost by the longer string's length,
# so the same single-character difference is fatal in a short numeric
# field but negligible in a long clause — exactly the behavior we want.
# The precision guardrail tests pin the digit/word behavior so loosening
# recall later can't quietly cost correctness.

PUNCT_EDIT_COST = 0.1
CASE_EDIT_COST = 0.1
LETTER_EDIT_COST = 1.0
DIGIT_EDIT_COST = 3.0


def char_class(ch: str) -> str:
    """One of ``digit`` / ``alpha`` / ``space`` / ``punct``."""
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "alpha"
    if ch.isspace():
        return "space"
    return "punct"


def sub_cost(a: str, b: str) -> float:
    """Cost of substituting ``a`` with ``b`` (0 if identical)."""
    if a == b:
        return 0.0
    ca, cb = char_class(a), char_class(b)
    # Any digit involved in a non-identical edit is expensive — this is
    # what keeps ``1001``/``1000`` and ``5/6``/``5/16`` apart, and it
    # deliberately does *not* bridge OCR digit/letter shape confusions
    # (``O``/``0``, ``l``/``1``): numeric integrity wins over that recall.
    if ca == "digit" or cb == "digit":
        return DIGIT_EDIT_COST
    if ca == "alpha" and cb == "alpha":
        return CASE_EDIT_COST if a.lower() == b.lower() else LETTER_EDIT_COST
    if ca in ("punct", "space") and cb in ("punct", "space"):
        return PUNCT_EDIT_COST
    return LETTER_EDIT_COST


def indel_cost(ch: str) -> float:
    """Cost of inserting or deleting ``ch``."""
    cls = char_class(ch)
    if cls == "digit":
        return DIGIT_EDIT_COST
    if cls in ("punct", "space"):
        return PUNCT_EDIT_COST
    return LETTER_EDIT_COST


def _classify_string(s: str) -> tuple[list[int], list[float], list[str]]:
    """Per-character class code (0 digit, 1 alpha, 2 punct/space), indel
    cost, and lowercased char — precomputed once so the DP inner loop
    does no per-cell function calls."""
    codes: list[int] = []
    indels: list[float] = []
    lowers: list[str] = []
    for ch in s:
        if ch.isdigit():
            codes.append(0)
            indels.append(DIGIT_EDIT_COST)
        elif ch.isalpha():
            codes.append(1)
            indels.append(LETTER_EDIT_COST)
        else:  # punctuation or whitespace
            codes.append(2)
            indels.append(PUNCT_EDIT_COST)
        lowers.append(ch.lower())
    return codes, indels, lowers


def weighted_edit_distance(
    a: str, b: str, *, band: int | None = None, budget: float | None = None
) -> float:
    """Levenshtein distance with class-dependent op costs. The inner loop
    inlines the same costs :func:`sub_cost` / :func:`indel_cost` define
    (kept in sync; those remain the readable spec and are unit-tested).

    ``band`` restricts the DP to cells within ``band`` of the diagonal —
    an optimal alignment of two near-length strings never strays far from
    it, so this is exact for the matches we care about and only ever
    *over*-estimates otherwise (safe: it can reject a far-off span, never
    accept a wrong one). ``budget`` enables early-abandon: once a whole
    row's best cost exceeds it, the final cost will too, so we bail. Both
    default off, giving an exact full DP (what the unit tests pin)."""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0:
        return sum(indel_cost(c) for c in b)
    if lb == 0:
        return sum(indel_cost(c) for c in a)
    if band is None:
        band = max(la, lb)
    big = float("inf")
    b_codes, b_indels, b_lowers = _classify_string(b)
    a_codes, a_indels, a_lowers = _classify_string(a)

    prev = [big] * (lb + 1)
    prev[0] = 0.0
    acc = 0.0
    for j in range(1, min(lb, band) + 1):
        acc += b_indels[j - 1]
        prev[j] = acc

    for i in range(1, la + 1):
        ai_code = a_codes[i - 1]
        ai_indel = a_indels[i - 1]
        ai_low = a_lowers[i - 1]
        jlo = max(1, i - band)
        jhi = min(lb, i + band)
        cur = [big] * (lb + 1)
        if jlo == 1:
            cur[0] = prev[0] + ai_indel
        row_min = cur[0]
        for j in range(jlo, jhi + 1):
            bj_code = b_codes[j - 1]
            # Substitution cost, inlined (mirrors sub_cost).
            if ai_low == b_lowers[j - 1] and ai_code == bj_code:
                # identical, or same-class case-only difference
                sub = 0.0 if a[i - 1] == b[j - 1] else CASE_EDIT_COST
            elif ai_code == 0 or bj_code == 0:
                sub = DIGIT_EDIT_COST
            elif ai_code == 1 and bj_code == 1:
                sub = LETTER_EDIT_COST
            elif ai_code == 2 and bj_code == 2:
                sub = PUNCT_EDIT_COST
            else:
                sub = LETTER_EDIT_COST
            v = prev[j - 1] + sub
            d = prev[j] + ai_indel
            if d < v:
                v = d
            ins = cur[j - 1] + b_indels[j - 1]
            if ins < v:
                v = ins
            cur[j] = v
            if v < row_min:
                row_min = v
        if budget is not None and row_min > budget:
            return big
        prev = cur
    return prev[lb]


def similarity(a: str, b: str) -> float:
    """Length-normalized similarity in ``[0, 1]``: ``1.0`` for identical
    strings, falling toward ``0`` as weighted edit cost approaches the
    longer string's length. Clamped at 0 (digit edits can exceed it)."""
    if not a and not b:
        return 1.0
    cost = weighted_edit_distance(a, b)
    return max(0.0, 1.0 - cost / max(len(a), len(b)))


# Acceptance bar for a fuzzy span. Tuned with the cost model so the
# precision guardrail pairs (one-digit, dropped-negation, prefix-only)
# fall below it while boundary noise (trailing colon, line-break hyphen,
# case) clears it. See tests/test_similarity.py.
FUZZY_SIM_THRESHOLD = 0.9


_FUZZY_WORD_WINDOW = 0.35  # span word-count may differ from target by this share
_FUZZY_ANCHOR_SLACK = 2  # start-position slack around an anchor occurrence
_FUZZY_BAND_PAD = 8  # DP band padding beyond the two strings' length gap
_FUZZY_MAX_ANCHOR_OCC = 4  # if the rarest target token recurs more than this,
# the target has no distinctive seed on the page — defer rather than scan
# (and DP) the many candidate locations it would generate.


def find_fuzzy_spans(
    text: str, page_words: list[Word], threshold: float = FUZZY_SIM_THRESHOLD
) -> list[tuple[int, int]]:
    """Recall fallback for when :func:`find_spans` finds no exact span.

    Scores contiguous word spans against the target with
    :func:`similarity` (character-class weighted) and commits the single
    best-scoring location only when it is *unambiguous* — no other
    above-threshold span sits at a different (non-overlapping) place on
    the page. Otherwise returns ``[]`` and the caller falls back to its
    own recovery path, same as an exact miss.

    To stay fast on pages of repetitive prose (where every window has a
    similar character *and* common-word profile), candidate spans are
    seeded by **anchor**: the target's rarest word-token on this page.
    Only spans positioned so that anchor lines up (± a little slack) get
    the O(len²) distance computed — typically a handful per page. A
    mangled boundary token (``party:`` vs ``party``) just isn't chosen as
    the anchor; a distinctive interior word seeds the search instead."""
    target = fuzzy_norm(text)
    target_tokens = [t for t in (fuzzy_norm(tok) for tok in text.split()) if t]
    if not target or not target_tokens:
        return []
    target_len = len(target)
    n_tokens = len(target_tokens)
    lo_words = max(1, int(n_tokens * (1 - _FUZZY_WORD_WINDOW)))
    hi_words = int(n_tokens * (1 + _FUZZY_WORD_WINDOW)) + 1

    norms = [w.text_norm for w in page_words]
    n = len(norms)
    # Seed on the alphanumeric-lowercase core of each token (see
    # core_token) so punctuation- or case-variant tokens still anchor.
    positions: dict[str, list[int]] = {}
    for i, w in enumerate(norms):
        core = core_token(w)
        if core:
            positions.setdefault(core, []).append(i)
    target_cores = [core_token(t) for t in target_tokens]

    # Anchor = the target token present on the page that occurs least
    # often (ties broken toward the longest, most distinctive token). If
    # no target token appears on the page at all there is nothing to seed
    # from — defer.
    present = {c for c in target_cores if c and c in positions}
    if not present:
        return []
    anchor = min(present, key=lambda c: (len(positions[c]), -len(c)))
    if len(positions[anchor]) > _FUZZY_MAX_ANCHOR_OCC:
        return []  # no distinctive anchor — can't localize cheaply or confidently
    anchor_word_idx = target_cores.index(anchor)

    starts: set[int] = set()
    for pos in positions[anchor]:
        base = pos - anchor_word_idx
        for delta in range(-_FUZZY_ANCHOR_SLACK, _FUZZY_ANCHOR_SLACK + 1):
            s = base + delta
            if 0 <= s < n:
                starts.add(s)

    scored: list[tuple[float, tuple[int, int]]] = []
    for start in starts:
        for end in range(max(start, start + lo_words - 1), min(n, start + hi_words)):
            acc = "".join(norms[start : end + 1])
            la = len(acc)
            # Wider on the upper bound so short heading-embedded values
            # like ``75 CREDITS`` (9 chars) still match against
            # ``(75 CREDITS)`` (11 chars on the page) — a tighter cap
            # cut off real matches just because added boundary punctuation
            # nudged length past 1.2x. The band-DP keeps cost bounded.
            if la > target_len * 1.4:
                break  # only grows from here; this start is exhausted
            if la < target_len * 0.7:
                continue
            # Budget + band let the DP abandon hopeless spans early and
            # skip cells far off the diagonal — the hot-path optimization.
            maxlen = la if la > target_len else target_len
            budget = (1.0 - threshold) * maxlen
            band = abs(la - target_len) + _FUZZY_BAND_PAD
            cost = weighted_edit_distance(acc, target, band=band, budget=budget)
            if cost <= budget:
                scored.append((1.0 - cost / maxlen, (start, end + 1)))

    if not scored:
        return []
    # Best = highest score, then closest to target length, then leftmost.
    scored.sort(key=lambda s: (-s[0], abs(s[1][1] - s[1][0]), s[1][0]))
    best_span = scored[0][1]
    # Ambiguous if any above-threshold span sits at a different location.
    if any(not span_overlaps_any(span, [best_span]) for _sim, span in scored):
        return []
    return [best_span]


# ---- Visual-line splitting -------------------------------------------------


def line_groups(words: list[Word]) -> list[list[Word]]:
    """Group words into visual lines using y-center proximity.

    Tolerance is half a median word height — tight enough that adjacent
    lines don't merge, loose enough to absorb OCR-y drift on tall
    characters. Words are visited in (top, left) reading order so groups
    come back in document order."""
    if not words:
        return []
    heights = sorted(w.height for w in words if w.height > 0)
    median_h = heights[len(heights) // 2] if heights else 12.0
    tol = max(median_h * 0.5, 2.0)
    groups: list[list[Word]] = []
    current: list[Word] = []
    cur_y: float | None = None
    for w in sorted(words, key=lambda w: (w.top, w.left)):
        if cur_y is None or abs(w.y_center - cur_y) <= tol:
            current.append(w)
            cur_y = w.y_center if cur_y is None else (cur_y + w.y_center) / 2
        else:
            groups.append(current)
            current = [w]
            cur_y = w.y_center
    if current:
        groups.append(current)
    return groups
