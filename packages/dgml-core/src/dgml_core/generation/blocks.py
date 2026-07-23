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

"""Typed block model and the deterministic flat→tree builder.

The model never emits a tree. It emits a FLAT sequence of typed blocks —
headings with levels, paragraphs, list items, table rows, form fields — and
the tree (sections, lists, tables, forms) is derived here with plain code.
That single decision is what deletes the seam problem: a flat list has no
open elements, so a window boundary cannot leave anything dangling.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Structures the model may emit (flat). Everything else is coerced to "p".
FLAT_STRUCTURES = {"heading", "p", "item", "row", "field"}

# Concepts are PascalCase role names (`DefinitionOfTerm`) — they become XML
# element names in the final dgml, so they must be valid XML Names.
_CONCEPT_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")

# Structural/discourse words carry no semantics — the block's `structure`
# field already says what shape the content has. A concept ending in one of
# these is a common source of spurious tag variation — the same role labeled
# with and without a structural suffix; stripping is deterministic and
# replay-safe. A concept that is ONLY structural words normalizes to ''
# (i.e. unlabeled).
_STRUCTURAL_SUFFIX_RE = re.compile(
    r"(Sections?|Subsections?|SubSections?|Clauses?|Paragraphs?|Items?|Lists?|"
    r"Headings?|Titles?|Texts?|Blocks?|Contents?|Body|Lines?|Rows?|Entry|"
    r"Entries|Notes?|Statements?|Details?|Intro(?:duction)?|Conclusion|"
    r"Summary|Tables?)\d*$"
)

# Leaked prompt prefix: "ConceptClientName" → "ClientName" ("Conception" survives).
_CONCEPT_PREFIX_RE = re.compile(r"^Concept(?=[A-Z_])")

# A real concept name never approaches this length (longest observed legit tag
# is ~43 chars). Anything longer is garbage — a str()-ified payload fragment or
# a whole sentence — and coining a tag from it pollutes the XML and the schema.
_MAX_CONCEPT_CHARS = 64


def sanitize_concept(raw: str) -> str:
    """Coerce a concept label to PascalCase and strip structural suffixes.

    Accepts any input convention (kebab-case, spaced, already-Pascal) so
    cached runs and model variations normalize to one format. Trailing
    structural/role words and ordinals are removed (`PaymentTermsClause` →
    `PaymentTerms`, `Paragraph2` → ''): structure lives on the block, never
    in the concept.
    """
    parts = [w for w in _CONCEPT_SPLIT_RE.split(raw.strip()) if w]
    pascal = "".join(w[0].upper() + w[1:] if w[0].isalpha() else w for w in parts)
    prev = None
    while prev != pascal:
        prev = pascal
        pascal = _CONCEPT_PREFIX_RE.sub("", pascal)
        pascal = _STRUCTURAL_SUFFIX_RE.sub("", pascal)
        pascal = re.sub(r"\d+$", "", pascal)
    if pascal and not (pascal[0].isalpha() or pascal[0] == "_"):
        pascal = f"_{pascal}"
    if len(pascal) > _MAX_CONCEPT_CHARS:
        return ""
    return pascal


@dataclass
class Span:
    """Inline entity: a [start, end) character span of a block's text."""

    start: int
    end: int
    concept: str


@dataclass
class Block:
    """One flat transcription unit.

    `structure` is one of FLAT_STRUCTURES. `level` applies to headings
    (1 = top). `cells` applies to rows; `label`/`value` to form fields; all
    other content lives in `text`. `concept`/`role`/`entities` are empty
    until the labeling pass fills them (Option I: staying empty is legal).
    """

    id: str
    structure: str
    text: str = ""
    level: int = 1
    lim: str = ""
    # Concept carried by the lim itself (e.g. a date used as the list marker
    # of an itinerary day). Set by apply_labels when an entity quote matches
    # the lim rather than the block text; the renderer tags the lim with it.
    lim_concept: str = ""
    cells: list[str] = field(default_factory=list)
    label: str = ""
    # Field blocks only: a printed CHOICE GROUP (checkboxes/radio). `options`
    # are the printed choice labels in reading order; `checked` the subset
    # whose box carries a mark. The mark character itself (X, tick) is never
    # content. `value` remains the single-selection convenience.
    options: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    # Field blocks only: inline entity spans WITHIN the label text. A packed
    # "label" often carries real values (a code and a name in one line); the
    # renderer wraps each span so they stay tagged. Unlike value spans there is
    # no whole-vs-partial conflict — the block concept wraps the VALUE, never
    # the label.
    label_entities: list[Span] = field(default_factory=list)
    value: str = ""
    concept: str = ""
    value_concept: str = ""
    role: str = ""
    entities: list[Span] = field(default_factory=list)
    # Row blocks only: per-column concept (parallel to `cells`) and the
    # table-level concept shared by every row of the table. Filled by the
    # labeling pass and made consistent by the propagation pass — a record
    # table's columns repeat across rows (propagated uniform); a key-value
    # table's columns vary (kept per-row).
    cell_concepts: list[str] = field(default_factory=list)
    # Row blocks only: inline entity spans WITHIN each cell's text (parallel to
    # `cells`). A single physical cell may hold several values (e.g. a part
    # number and a description in one column); the labeler tags them inline and
    # the renderer wraps each span, so sub-cell values survive even when the
    # transcribed cell count disagrees with the labeler's column model.
    cell_entities: list[list[Span]] = field(default_factory=list)
    group_concept: str = ""
    # Row blocks only, set by propagate_table_consistency. `header_row` marks a
    # demoted printed-title row so the renderer emits its cells as `ColumnHeader`
    # structure-td elements; `kv_table` marks the rows of a key-value run (no
    # column concept repeats across data rows) so the renderer emits a uniform
    # generic td with the value concept as an inner span — a stable td localname
    # (no per-column tag collision) that still keeps the leaf concept tag.
    header_row: bool = False
    kv_table: bool = False

    def flat_text(self) -> str:
        """Every character this block contributes to the document."""
        if self.structure == "row":
            return " ".join(self.cells)
        if self.structure == "field":
            return " ".join(p for p in (self.lim, self.label, self.value) if p)
        return f"{self.lim} {self.text}".strip() if self.lim else self.text


def parse_block(raw: dict[str, Any], block_id: str) -> Block | None:
    """Validate/coerce one model-emitted block dict; ``None`` drops it."""
    structure = str(raw.get("structure", "")).strip().lower()
    if structure not in FLAT_STRUCTURES:
        structure = "p"
    text = str(raw.get("text", "") or "")
    cells = [str(c) for c in raw.get("cells", []) or []]
    label = str(raw.get("label", "") or "")
    value = str(raw.get("value", "") or "")
    options = [str(o).strip() for o in raw.get("options", []) or [] if str(o).strip()]
    checked = [str(c).strip() for c in raw.get("checked", []) or [] if str(c).strip()]
    if options:
        # a selection can only be one of the PRINTED choices — a checked
        # entry outside the options is a hallucinated mark and is dropped
        checked = [c for c in checked if c in options]
    if not value and len(checked) == 1:
        value = checked[0]
    # A lim alone is content: numbered-but-untitled headings ("6.4.1" + body)
    # must survive, or the sub-section they open is silently flattened.
    lim = str(raw.get("lim", "") or "").strip()
    if not text and not cells and not (label or value) and not options and not lim:
        return None
    try:
        level = max(1, min(6, int(raw.get("level", 1))))
    except (TypeError, ValueError):
        level = 1
    return Block(
        id=block_id,
        structure=structure,
        text=text,
        level=level,
        lim=lim,
        cells=cells,
        label=label,
        value=value,
        options=options,
        checked=checked,
    )


# ── post-transcription normalization ─────────────────────────────────────────
# Two deterministic passes that remove the model's per-run degrees of freedom
# where the printed page already encodes the answer. Both are no-ops on input
# that is already consistent.

_DOTTED_LIM_RE = re.compile(r"^\d+(?:\.\d+)*\.?$")


def anchor_heading_levels(blocks: list[Block]) -> None:
    """Make dotted-numbered heading levels a pure function of the printed lim.

    The model assigns ``level`` per window, so the same numbering scheme can
    land at different depths in different windows/runs ("2.1" as level 2 here,
    level 3 there) — and since build_tree derives ALL <sec> nesting from the
    levels, that flips the tree. The printed enumerator is copied verbatim and
    encodes depth exactly: the first dotted-numeric heading fixes the scheme's
    base level, and every other dotted heading sits at base + (extra dotted
    components). Headings without a dotted-numeric lim are untouched.
    """
    base: int | None = None
    for b in blocks:
        if b.structure != "heading" or not _DOTTED_LIM_RE.match(b.lim):
            continue
        depth = b.lim.rstrip(".").count(".")  # "2" -> 0, "2.1" -> 1, "6.4.1" -> 2
        if base is None:
            base = max(1, b.level - depth)
        # Deliberately NOT capped at the prompt's 1-6 scale: an anchored level
        # is exact, and capping would flatten sub-clauses deeper than 6 into
        # siblings of their parents. 12 only bounds pathological lims.
        b.level = min(12, base + depth)


# Paragraph-initial enumerators that mark a list entry when they appear as a
# sequential series: "(a) ...", "(i) ...", "(1) ...", "1) ...". Conservative on
# purpose — bare "a." or "1." starts are too common in prose to convert.
_ENUM_RES: list[tuple[str, re.Pattern[str]]] = [
    ("alpha", re.compile(r"^\(([a-z])\)\s*")),
    ("roman", re.compile(r"^\(([ivxlc]+)\)\s*")),
    ("digit", re.compile(r"^\((\d+)\)\s*")),
    ("digit", re.compile(r"^(\d+)\)\s*")),
]
_ROMAN_VAL = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def _enum_ordinal(family: str, marker: str) -> int:
    if family == "alpha":
        return ord(marker) - ord("a") + 1
    if family == "digit":
        return int(marker)
    total = 0
    for ch, nxt in zip(marker, marker[1:] + " ", strict=True):
        v = _ROMAN_VAL[ch]
        total += -v if nxt in _ROMAN_VAL and _ROMAN_VAL[nxt] > v else v
    return total


def normalize_enumerated_paragraphs(blocks: list[Block]) -> None:
    """Turn sequential enumerated ``p`` runs into ``item`` blocks.

    "(a) ...", "(b) ..." emitted as paragraphs in one run and as list items in
    another is the single biggest structural flip between runs. A run of >= 2
    CONSECUTIVE p blocks whose texts start with the same enumerator family in
    strictly increasing sequence is unambiguous — convert each to an item and
    lift the printed marker into ``lim``. Isolated markers (a lone "(a) ..."
    paragraph, prose cross-references) never match the series requirement.
    """
    run: list[tuple[Block, str, str]] = []  # (block, stripped text, printed marker)
    run_family, run_ordinal = "", 0

    def flush() -> None:
        if len(run) >= 2:
            for blk, stripped, marker in run:
                blk.structure = "item"
                blk.lim = marker
                blk.text = stripped
        run.clear()

    for b in blocks:
        candidates = []
        if b.structure == "p" and not b.lim:
            for family, rx in _ENUM_RES:
                m = rx.match(b.text)
                if m:
                    candidates.append((family, m.group(1), m.group(0)))
        if not candidates:
            flush()
            continue
        # "(i)" parses as alpha AND roman, "(c)" as alpha and roman-100: prefer
        # whichever continues the current series; otherwise roman for "i" (the
        # usual roman list opener), first match for everything else.
        matched = next(
            (
                c
                for c in candidates
                if run and c[0] == run_family and _enum_ordinal(c[0], c[1]) == run_ordinal + 1
            ),
            None,
        )
        if matched is None:
            matched = next(
                (c for c in candidates if c[0] == "roman" and c[1] == "i"), candidates[0]
            )
        family, marker, prefix = matched
        ordinal = _enum_ordinal(family, marker)
        if run and not (family == run_family and ordinal == run_ordinal + 1):
            flush()
        run_family, run_ordinal = family, ordinal
        run.append((b, b.text[len(prefix) :], prefix.strip()))
    flush()


# ── flat → tree ──────────────────────────────────────────────────────────────


@dataclass
class Node:
    """Tree node produced by `build_tree` and consumed by the renderer.

    `kind` is the derived structural shape: section | h | p | list | li |
    table | tr | form | fld. Leaf content references the originating Block
    so labels/entities applied to blocks surface in the rendered output.
    """

    kind: str
    block: Block | None = None
    children: list[Node] = field(default_factory=list)
    # For a synthesized entity-container section that has no heading child, the
    # container concept it should render under (see build_tree's parent_map).
    concept: str = ""


def build_tree(blocks: list[Block], parent_map: Mapping[str, str] | None = None) -> Node:
    """Derive the document tree from the flat block sequence.

    - a heading of level N opens a section nested under the most recent
      heading of level < N (a stack, exactly like Markdown);
    - consecutive `item` blocks become one list;
    - consecutive `row` blocks become one table;
    - consecutive `field` blocks become one form;
    - when *parent_map* maps a leaf concept to a container concept, consecutive
      leaf blocks (``p``/``field``/``item``) whose concept shares one container
      are wrapped in a synthesized ``section`` node carrying that container
      concept — the "entity" grouping (e.g. a buyer's name/address/phone become
      one ``BuyerInformation`` section). Rows keep the table path; their own
      ``concept`` already rides on the ``tr``.

    Pure code, no heuristics beyond run-grouping — the same input always
    yields the same tree, and any tree it yields is well-formed.
    """
    pm = parent_map or {}
    root = Node(kind="doc")
    # Stack of (level, section-node); root acts as level 0.
    stack: list[tuple[int, Node]] = [(0, root)]

    def container() -> Node:
        return stack[-1][1]

    run_kind: str | None = None
    run_node: Node | None = None
    ent_node: Node | None = None  # current synthesized entity-container section
    ent_key: str | None = None

    def end_run() -> None:
        nonlocal run_kind, run_node
        run_kind, run_node = None, None

    def end_entity() -> None:
        nonlocal ent_node, ent_key
        ent_node, ent_key = None, None

    ent_child = {"p": "p", "item": "li", "field": "fld"}
    for block in blocks:
        if block.structure == "heading":
            end_run()
            end_entity()
            while stack[-1][0] >= block.level:
                stack.pop()
            section = Node(kind="section")
            section.children.append(Node(kind="h", block=block))
            container().children.append(section)
            stack.append((block.level, section))
            continue

        # Entity grouping: a leaf whose concept has a container parent joins a
        # synthesized section for that container (rows are excluded — a row's
        # concept already renders on its <tr>).
        fam = pm.get(block.concept) if block.concept else None
        if fam and block.structure in ent_child:
            end_run()
            if ent_key != fam or ent_node is None:
                end_entity()
                ent_node = Node(kind="section", concept=fam)
                container().children.append(ent_node)
                ent_key = fam
            ent_node.children.append(Node(kind=ent_child[block.structure], block=block))
            continue
        end_entity()

        group_for = {"item": "list", "row": "table", "field": "form"}
        child_kind = {"item": "li", "row": "tr", "field": "fld"}
        if block.structure in group_for:
            wanted = group_for[block.structure]
            if run_kind != wanted or run_node is None:
                run_node = Node(kind=wanted)
                container().children.append(run_node)
                run_kind = wanted
            run_node.children.append(Node(kind=child_kind[block.structure], block=block))
            continue

        end_run()
        container().children.append(Node(kind="p", block=block))

    return root
