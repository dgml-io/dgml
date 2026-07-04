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
# these is the suffix-conflation the format analysis measured at ~70% of all
# tag variation; stripping is deterministic and replay-safe. A concept that
# is ONLY structural words normalizes to '' (i.e. unlabeled).
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
    if not text and not cells and not (label or value):
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
        lim=str(raw.get("lim", "") or "").strip(),
        cells=cells,
        label=label,
        value=value,
    )


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
