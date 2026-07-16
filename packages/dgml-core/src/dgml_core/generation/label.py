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

"""Pass B — semantic labeling with a shared concept roster.

Documents are labeled in bounded chunks (per document, split further when
very large), with the **concept roster accumulated so far** fed into every
subsequent call: "these concepts exist — reuse them verbatim where the role
matches". Cross-document consistency comes from the roster, while each call
stays small enough that the model can label *densely* instead of
cherry-picking a handful of values.

Density contract: every heading block MUST receive a section-role concept;
repeating items/rows/fields receive their shared role; only connective prose may stay
unlabeled. Option I still applies — boilerplate paragraphs with no
queryable role legitimately get nothing — but "skip almost everything" is
not a permitted reading anymore.

Text is never touched; entities are VERBATIM QUOTES of the value, located
in the block's text by the pipeline — the model never counts positions.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dgml_core import llm
from dgml_core.generation.blocks import Block, Node, Span, build_tree, sanitize_concept
from dgml_core.generation.prompts import get as prompt
from dgml_core.generation.schema import VALID_KINDS, Schema, SchemaTag
from dgml_core.generation.transcribe import cache_write, loads_tolerant, strip_fences

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

_SNIPPET_CHARS = 160
# Blocks per labeling call. Bounds both the listing the model must read and
# the JSON it must write, so density doesn't collapse under an output cap.
_MAX_BLOCKS_PER_CALL = 1200
_ROSTER_MAX_ENTRIES = 400
# A chunk labeled far below this fraction is treated as a failed/truncated call and retried once.
_MIN_LABELED_FRACTION = 0.3
# When a block and an inline value carry the SAME concept and the value is a
# proper substring, the block is a thin "label: value" wrapper if the non-value
# remainder is no longer than this — the concept is really the value's kind, so
# it moves onto the value and the wrapper becomes dg:chunk. A longer remainder is
# a substantive clause that keeps its own role concept.
_THIN_WRAPPER_CHARS = 24
# Distinct example values collected per concept for schema.json (1 or many).
_SCHEMA_MAX_EXAMPLES = 3
# Currency amounts and percentages, un-anchored, for deterministic value isolation.
_VALUE_SCAN_RE = re.compile(
    r"[$€£¥₹]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?"
    r"|\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?[$€£¥₹]"
    r"|\d+(?:\.\d+)?\s?%"
)


@dataclass
class RosterEntry:
    """One concept in the labeling vocabulary (the roster).

    ``description`` is the planned/seeded role description; ``examples`` are
    observed VERBATIM values (never invented — filled from labeled blocks or a
    seed schema's curated examples). ``kind``/``parent`` mirror the schema.json
    fields when known, so seeding and re-deriving the schema is lossless.
    ``confirmed`` marks entries with evidence — a seed (authoritative by
    construction) or an observation in this run — and drives the two-tier
    prompt rendering. ``frozen`` pins schema-seeded entries: observations never
    mutate them, so a seeded run's rendered roster stays byte-stable (which is
    what lets the roster prompt block cache across calls).
    """

    description: str = ""
    examples: list[str] = field(default_factory=list)
    kind: str = ""  # "", or one of schema.VALID_KINDS
    parent: str = ""
    confirmed: bool = False
    frozen: bool = False


# Observed example values kept per roster entry / shown per rendered line.
_ROSTER_EXAMPLE_CHARS = 60
_ROSTER_RENDER_EXAMPLES = 2
# Kind marker shown on confirmed roster lines ("inline" reads better as value).
_KIND_LABELS = {"section": "section", "row": "row", "inline": "value"}


def _needs_label(block: Block) -> bool:
    """True for an unlabeled heading — a section that should carry a concept.

    A heading names its section's role; leaving one unlabeled drops a whole
    section's concept. This is the coverage gap worth a retry; items and
    prose are not forced.
    """
    return not block.concept and not block.value_concept and block.structure == "heading"


SYSTEM_PROMPT = prompt("label_system")


def render_block_listing(doc_name: str, blocks: list[Block]) -> str:
    """One-line-per-block input for one labeling call — FULL block text.

    The labeler must see the complete text: entities are verbatim quotes, so
    any truncation here is a hard ceiling on what can be labeled (an earlier
    snippet cap made values past the cut impossible to quote).
    """
    lines: list[str] = [f"== {doc_name} =="]
    for b in blocks:
        head = f"{b.id} {b.structure}"
        if b.lim:
            head += f" [{b.lim}]"
        lines.append(f"{head}: {b.flat_text()}")
    return "\n".join(lines)


def _roster_line(concept: str, entry: RosterEntry, *, confirmed: bool) -> str:
    """One rendered roster line: name, kind marker, role, observed examples."""
    line = f"- {concept}"
    if confirmed and entry.kind in _KIND_LABELS:
        line += f" [{_KIND_LABELS[entry.kind]}]"
    if entry.description:
        line += f" — {entry.description[:100]}"
    if confirmed and entry.examples:
        shown = "; ".join(f'"{ex}"' for ex in entry.examples[:_ROSTER_RENDER_EXAMPLES])
        line += f" (seen: {shown})"
    return line


def render_roster(roster: Mapping[str, RosterEntry]) -> str:
    """The concept roster, shown to every labeling call — two tiers.

    CONFIRMED concepts (seeded, or observed in this run) render rich — kind
    marker, role description, observed example values — under the strong
    reuse-verbatim intro. PLANNED concepts (proposed by the roster planner but
    not yet observed) render as name + description only, under a softer intro.
    The split gives the labeler an honest confidence signal instead of one
    uniform list; descriptions and examples are never conflated (the old
    single-string roster rendered planned descriptions inside an "e.g." hint).
    Confirmed entries fill the size cap first.
    """
    confirmed = [(n, e) for n, e in roster.items() if e.confirmed]
    planned = [(n, e) for n, e in roster.items() if not e.confirmed]
    lines: list[str] = []
    budget = _ROSTER_MAX_ENTRIES
    if confirmed:
        lines.append(prompt("roster_intro"))
        lines.extend(_roster_line(n, e, confirmed=True) for n, e in confirmed[:budget])
        budget -= len(confirmed[:budget])
    if planned and budget > 0:
        if lines:
            lines.append("")
        lines.append(prompt("roster_planned_intro"))
        lines.extend(_roster_line(n, e, confirmed=False) for n, e in planned[:budget])
    return "\n".join(lines)


def _roster_content_blocks(
    roster: Mapping[str, RosterEntry], *, model: str
) -> list[dict[str, Any]]:
    """The rendered roster as a user-content block, marked cacheable.

    Rendered once per document and reused by every call for that document
    (chunks and the section retry), the block is byte-identical across calls
    while the roster is unchanged — so on Anthropic models a ``cache_control``
    marker lets the provider replay it instead of re-reading it. Seeded runs
    (frozen entries) hit for the whole batch; unseeded runs hit whenever no new
    concept/example landed between calls. Non-Anthropic providers cache stable
    prefixes implicitly, so the marker is omitted (litellm would reject it).
    """
    text = render_roster(roster)
    if not text:
        return []
    block: dict[str, Any] = {"type": "text", "text": text}
    if llm.is_anthropic_model(model):
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def _parse_labels_json(raw: str) -> dict[str, Any]:
    match = _JSON_FENCE_RE.search(raw)
    cleaned = (match.group(1) if match else raw).strip()
    out = loads_tolerant(cleaned)
    return out if isinstance(out, dict) else {}


def _locate_quote(text: str, raw_span: Mapping[str, Any]) -> tuple[int, int] | None:
    """Resolve a verbatim entity quote to a [start, end) span in *text*.

    The model COPIES the value; the pipeline does the arithmetic — models
    cannot count positions reliably (the earlier offset contract cut values
    mid-token in live runs). The quote must occur verbatim; ``occurrence``
    (1-based) picks among repeats, defaulting to the first.
    """
    quote = str(raw_span.get("quote", "") or "")
    if not quote:
        return None
    try:
        occurrence = max(1, int(raw_span.get("occurrence", 1)))
    except (TypeError, ValueError):
        occurrence = 1
    start = -1
    for _ in range(occurrence):
        start = text.find(quote, start + 1)
        if start == -1:
            return None
    return start, start + len(quote)


def apply_labels(
    blocks: list[Block],
    labels: Mapping[str, Any],
    *,
    doc_name: str = "",
) -> list[str]:
    """Apply a validated subset of one call's labels in place.

    Returns warnings for everything dropped. Quotes that don't occur
    verbatim, mid-token offset spans, overlapping spans, and unknown ids are
    discarded — labeling can only annotate, never corrupt.
    """
    warnings: list[str] = []
    by_id = {b.id: b for b in blocks}
    for block_id, payload in labels.items():
        block = by_id.get(str(block_id))
        if block is None or not isinstance(payload, Mapping):
            warnings.append(f"{doc_name}: unknown block {block_id!r}; dropped")
            continue
        raw_concept = str(payload.get("concept", "") or "")
        concept = sanitize_concept(raw_concept)
        if concept:
            block.concept = concept
        elif raw_concept.strip():
            warnings.append(
                f"{doc_name}/{block_id}: concept {raw_concept!r} is purely structural; dropped"
            )
        if block.structure == "heading":
            # A value-heading is flagged by an entity quoting the whole heading
            # text: the heading text IS the value, so its kind tags the heading.
            whole = next(
                (
                    e
                    for e in payload.get("entities", []) or []
                    if isinstance(e, Mapping)
                    and str(e.get("quote", "") or "").strip() == block.text.strip()
                ),
                None,
            )
            value_concept = sanitize_concept(str(whole.get("concept", "") or "")) if whole else ""
            if value_concept:
                block.value_concept = value_concept
                if value_concept == block.concept:
                    block.concept = ""
        # An entity quote that IS the list marker (a date or number used as the
        # item's label) can never be located in text/value — resolve it for
        # EVERY structure by carrying its concept onto the lim. Exact token
        # match only: a partial quote would overstate the lim's value.
        if block.lim and not block.lim_concept:
            lim_tokens = block.lim.split()
            for raw_span in payload.get("entities", []) or []:
                if not isinstance(raw_span, Mapping):
                    continue
                lim_concept = sanitize_concept(str(raw_span.get("concept", "") or ""))
                if lim_concept and str(raw_span.get("quote", "") or "").split() == lim_tokens:
                    block.lim_concept = lim_concept
                    break
        # Row entities live inside individual cells (block.text is empty for a
        # row), so they are resolved per-cell below, not here. Field (key-value)
        # blocks are also skipped: their value lives in block.value (block.text
        # is empty) and the renderer wraps that value with block.concept — it
        # never emits field entity spans. Running the loop here would only let
        # the thin-wrapper branch below mis-fire on the empty text
        # (len("") - len(quote) <= THIN_WRAPPER) and erase block.concept, which
        # silently dropped the tag on every labelled key-value field.
        if block.structure not in ("row", "field"):
            spans: list[Span] = []
            # Does this block hold inline values of concepts OTHER than its own?
            # If so and the block concept merely duplicates one of its values,
            # the labeler named a multi-value container after one of its values
            # (e.g. a SellerAddress block that also holds VendorId/OrgName/Phone)
            # — the block must become a dg:chunk container, not a leaf wrapper.
            other_concepts = {
                sanitize_concept(str(e.get("concept", "") or ""))
                for e in (payload.get("entities", []) or [])
                if isinstance(e, Mapping)
            }
            has_other_concept = bool(other_concepts - {concept, ""})
            for raw_span in payload.get("entities", []) or []:
                if not isinstance(raw_span, Mapping):
                    continue
                span_concept = sanitize_concept(str(raw_span.get("concept", "") or ""))
                if not span_concept:
                    continue
                if (
                    block.structure == "heading"
                    and block.value_concept
                    and str(raw_span.get("quote", "") or "").strip() == block.text.strip()
                ):
                    continue
                if concept and span_concept == concept:
                    quote = str(raw_span.get("quote", "") or "").strip()
                    if quote and quote == block.text.strip():
                        # Whole-block value: the block IS the value (a cell or
                        # line that is only the value). Keep the one concept on
                        # the block and add no inner span.
                        continue
                    if quote and len(block.text.strip()) - len(quote) <= _THIN_WRAPPER_CHARS:
                        # Thin "label: value" line: the block concept is really
                        # the value's kind. Move it onto the value and leave the
                        # wrapper unlabeled, so the value is isolated (not the
                        # whole line) and the wrapper renders as dg:chunk.
                        block.concept = ""
                    elif has_other_concept:
                        # Multi-value container the labeler named after one of
                        # its values: demote the block to dg:chunk and KEEP this
                        # value as an inline leaf (fall through), so a leaf
                        # concept never wraps other leaves (the *Address/*Address
                        # nesting bug). All sibling values render as leaves.
                        block.concept = ""
                    else:
                        # Substantive single-role clause: the value's concept
                        # must be MORE SPECIFIC than the block's role. The model
                        # failed to differentiate; drop the duplicate rather than
                        # smear the block concept over the value.
                        warnings.append(
                            f"{doc_name}/{block_id}: entity concept equals the block "
                            f"concept ({span_concept!r}); dropped"
                        )
                        continue
                located = _locate_quote(block.text, raw_span)
                if located is None:
                    # A quote equal to the lim was already resolved by the lim
                    # pre-pass above — skip it here without a warning.
                    q_tokens = str(raw_span.get("quote", "") or "").split()
                    if q_tokens and block.lim and q_tokens == block.lim.split():
                        continue
                    warnings.append(
                        f"{doc_name}/{block_id}: entity quote not found verbatim; dropped"
                    )
                    continue
                spans.append(Span(start=located[0], end=located[1], concept=span_concept))
            spans.sort(key=lambda s: s.start)
            kept: list[Span] = []
            for span in spans:
                if kept and span.start < kept[-1].end:
                    warnings.append(f"{doc_name}/{block_id}: overlapping span dropped")
                    continue
                kept.append(span)
            if kept:
                block.entities = kept

        # Row blocks carry a table-level concept, per-column concepts, and
        # inline per-cell entities. Two complementary signals:
        #   - `cells` (positional, one concept per column) drives whole-cell
        #     column tags and cross-row consistency — trusted only when its
        #     length matches the transcribed cell count (a count disagreement
        #     means the labeler's column model differs from the physical split,
        #     so positional alignment would be wrong).
        #   - `entities` (verbatim quotes) are resolved INSIDE each cell, so a
        #     single physical cell holding several values is split into inline
        #     concept spans regardless of the column count. The renderer prefers
        #     a split cell over its positional concept, so a count mismatch no
        #     longer discards the row's labels.
        if block.structure == "row":
            group = sanitize_concept(str(payload.get("table", "") or ""))
            if group:
                block.group_concept = group
            raw_cells = payload.get("cells", []) or []
            # The labeler occasionally answers a cell with an OBJECT — a
            # {"concept": ..., "entities": [...]} payload per cell — instead of
            # a plain concept string. Read the concept and fold the nested
            # entities into the entity pool (they are exactly the packed-cell
            # sub-value shape resolved below); str()-ifying the dict would coin
            # a garbage mega-concept that pollutes the XML and the schema.
            cell_names: list[str] = []
            row_entities: list[Any] = list(payload.get("entities", []) or [])
            for raw_cell in raw_cells:
                if isinstance(raw_cell, Mapping):
                    cell_names.append(str(raw_cell.get("concept", "") or ""))
                    nested = raw_cell.get("entities")
                    if isinstance(nested, list):
                        row_entities.extend(nested)
                else:
                    cell_names.append(str(raw_cell))
            if raw_cells and len(raw_cells) == len(block.cells):
                # Positional concepts align with the physical cells — trust them
                # (one concept per column, consistent across rows). Entities are
                # STILL resolved: a matching column model does not mean every
                # cell holds a single value — a packed cell's sub-values (a
                # quantity inside a description, an original price beside the
                # discounted one) arrive as entity quotes. Only PARTIAL spans
                # are kept: a whole-cell span either duplicates the positional
                # concept or contradicts it, and the positional column model
                # wins whole-cell (cross-row consistency).
                block.cell_concepts = [sanitize_concept(c) for c in cell_names]
                resolved = _resolve_cell_entities(
                    block, row_entities, concept, doc_name, block_id, []
                )
                block.cell_entities = [
                    [
                        sp
                        for sp in spans
                        if (cell := block.cells[i])
                        and cell.strip() != cell[sp.start : sp.end].strip()
                    ]
                    for i, spans in enumerate(resolved)
                ]
                if not any(block.cell_entities):
                    block.cell_entities = []
            else:
                # Count disagreement (or no cells): the labeler's column model
                # differs from the physical split — a single cell packs several
                # values. Fall back to inline per-cell entities, which survive the
                # mismatch and split a multi-value cell into concept spans.
                if raw_cells:
                    warnings.append(
                        f"{doc_name}/{block_id}: {len(raw_cells)} cell concept(s) "
                        f"!= {len(block.cells)} cells; using inline cell entities instead"
                    )
                block.cell_entities = _resolve_cell_entities(
                    block, row_entities, concept, doc_name, block_id, warnings
                )
        # Field (key-value) blocks: sub-values packed inside the value (a name
        # beside a code in one "label: value" line) arrive as entity quotes.
        # Only PARTIAL spans are kept — the renderer wraps the WHOLE value with
        # the block concept, so a whole-value span either duplicates it or
        # contradicts it, and the block concept wins whole-value (the same rule
        # as positional concepts on table cells).
        elif block.structure == "field":
            field_spans: list[Span] = []
            label_spans: list[Span] = []
            for raw_span in payload.get("entities", []) or []:
                if not isinstance(raw_span, Mapping):
                    continue
                span_concept = sanitize_concept(str(raw_span.get("concept", "") or ""))
                quote = str(raw_span.get("quote", "") or "")
                if not span_concept or not quote:
                    continue
                if span_concept == block.concept and block.value:
                    continue  # duplicate: the concept already wraps the (non-empty) value
                if block.value:
                    if quote.strip() == block.value.strip():
                        continue  # whole-value: the block concept wins
                    pos = _find_verbatim(block.value, quote, 0)
                    if pos != -1:
                        field_spans.append(
                            Span(start=pos, end=pos + len(quote), concept=span_concept)
                        )
                        continue
                # A packed "label" often carries real values (a code and a name
                # in one line). Whole and partial label quotes are both kept —
                # the block concept wraps the VALUE, never the label, so there
                # is no precedence conflict on this side.
                if block.label:
                    pos = _find_verbatim(block.label, quote, 0)
                    if pos != -1:
                        label_spans.append(
                            Span(start=pos, end=pos + len(quote), concept=span_concept)
                        )
            for raw_spans, attr in ((field_spans, "entities"), (label_spans, "label_entities")):
                raw_spans.sort(key=lambda s: s.start)
                kept_spans: list[Span] = []
                for sp in raw_spans:
                    if kept_spans and sp.start < kept_spans[-1].end:
                        warnings.append(f"{doc_name}/{block_id}: overlapping field span dropped")
                        continue
                    kept_spans.append(sp)
                if kept_spans:
                    setattr(block, attr, kept_spans)
    return warnings


def _find_verbatim(haystack: str, needle: str, start: int) -> int:
    """First index of *needle* at/after *start* not embedded in a larger token.

    A short value (e.g. a single digit) must not match inside a larger token;
    require the match to be flanked by non-alphanumeric characters (or edges).
    """
    i = haystack.find(needle, start)
    while i != -1:
        before = haystack[i - 1] if i > 0 else ""
        after = haystack[i + len(needle)] if i + len(needle) < len(haystack) else ""
        if not (before.isalnum() or after.isalnum()):
            return i
        i = haystack.find(needle, i + 1)
    return -1


def _resolve_cell_entities(
    block: Block,
    raw_entities: list[Any],
    block_concept: str,
    doc_name: str,
    block_id: str,
    warnings: list[str],
) -> list[list[Span]]:
    """Place each model entity quote inside the cell that contains it.

    Entities are emitted in column order, so a left-to-right sweep assigns each
    quote to the current-or-later cell (handling repeated values like a "1" that
    appears in two columns). Several entities may land in one cell — that is a
    multi-value cell the renderer splits into inline concept spans.
    """
    per_cell: list[list[Span]] = [[] for _ in block.cells]
    cursor = [0] * len(block.cells)  # per-cell search start (verbatim, in order)
    min_cell = 0
    for raw_span in raw_entities:
        if not isinstance(raw_span, Mapping):
            continue
        span_concept = sanitize_concept(str(raw_span.get("concept", "") or ""))
        if not span_concept or span_concept == block_concept:
            continue
        quote = str(raw_span.get("quote", "") or "")
        if not quote:
            continue
        for j in range(min_cell, len(block.cells)):
            pos = _find_verbatim(block.cells[j], quote, cursor[j])
            if pos != -1:
                per_cell[j].append(Span(start=pos, end=pos + len(quote), concept=span_concept))
                cursor[j] = pos + len(quote)
                min_cell = j
                break
        else:
            warnings.append(f"{doc_name}/{block_id}: cell entity {quote!r} not found; dropped")
    for cell_spans in per_cell:
        cell_spans.sort(key=lambda s: s.start)
    return per_cell


def _table_runs(blocks: list[Block]) -> list[list[Block]]:
    """Maximal runs of consecutive row blocks — one per derived table."""
    runs: list[list[Block]] = []
    current: list[Block] = []
    for b in blocks:
        if b.structure == "row":
            current.append(b)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _most_common(values: list[str]) -> str:
    """Most frequent non-empty value, or '' (ties broken by first seen)."""
    counts = Counter(v for v in values if v)
    return counts.most_common(1)[0][0] if counts else ""


def _effective_cell_concepts(block: Block) -> list[str]:
    """The concept ``to_semantic`` would actually tag on each cell.

    Mirrors the renderer's ``cell_concept or whole`` rule: the positional
    ``cell_concepts[i]`` when present, else a whole-cell entity concept (a lone
    span covering the entire cell). Used by the key-value classification so a
    data table whose column roles arrived as whole-cell ENTITIES
    (``cell_concepts`` empty) is still recognized as a data table — not
    mis-classified key-value just because the positional slot is unset.
    """
    out: list[str] = []
    for i, cell in enumerate(block.cells):
        cc = block.cell_concepts[i] if i < len(block.cell_concepts) else ""
        if not cc:
            ents = block.cell_entities[i] if i < len(block.cell_entities) else []
            if (
                len(ents) == 1
                and cell.strip()
                and cell[ents[0].start : ents[0].end].strip() == cell.strip()
            ):
                cc = ents[0].concept
        out.append(cc)
    return out


def propagate_table_consistency(blocks: list[Block]) -> None:
    """Make each table's row/column concepts consistent (record vs key-value).

    Record-vs-key-value is inferred from the model's own per-cell labels, not
    a separate classification: a column whose concept REPEATS across rows is
    a record column → its concept is propagated to every row's cell in that
    position; a column whose concepts vary is key-value → left per-row. The
    table-level concept and a repeating row concept are propagated the same
    way. No concept is invented — only copied where the evidence shows the
    role is uniform.
    """
    for run in _table_runs(blocks):
        if len(run) < 2:
            continue
        # A HEADER ROW labeled as data poisons everything downstream: its cells
        # (the printed column titles) become the column concepts' first VALUES,
        # shifting every real value's position, and they vote in the uniform-
        # column election below. Detect it by type disagreement: in every
        # column whose data cells are value-like (digit-bearing) across the
        # rest of the run, a header's cell is a bare word. Demote such a first
        # row to structure: clear its concepts so it renders as plain cells.
        first = run[0]
        eligible = votes = 0
        for col, cell in enumerate(first.cells):
            col_cells = [
                b.cells[col].strip() for b in run[1:] if col < len(b.cells) and b.cells[col].strip()
            ]
            if len(col_cells) < 2:
                continue
            digitish = sum(any(ch.isdigit() for ch in c) for c in col_cells)
            if digitish / len(col_cells) >= 0.8:
                eligible += 1
                if not any(ch.isdigit() for ch in cell):
                    votes += 1
        header: Block | None = None
        detected = eligible >= 2 and votes == eligible
        if detected:
            header = first
            # Cell concepts are always poison on a header row (column titles
            # becoming the columns' first values). The row-level concept is
            # kept ONLY when it differs from the data rows' shared concept —
            # a distinct role (e.g. "...TableHeader") is deliberate labeling;
            # sharing the data rows' concept just means the labeler treated
            # the header as one more data row, and that tag would wrap header
            # text in a data role.
            data_concept = _most_common([b.concept for b in run[1:]])
            if first.concept and first.concept == data_concept:
                first.concept = ""
            first.cell_concepts = []
            first.cell_entities = []
            # Remember this row is a demoted printed-title row so the renderer can
            # emit its cells as ColumnHeader structure-td elements (gold's
            # <ColumnHeader structure="td"> shape) instead of anonymous dg:chunk
            # tds, which the recall check counts as a headerless table.
            first.header_row = True
        # Table name: the one the model gave (first non-empty), shared by all.
        group = _most_common([b.group_concept for b in run])
        if group:
            for b in run:
                b.group_concept = group
        # Row concept: propagate only if it repeats (uniform record rows).
        row_concept = _most_common([b.concept for b in run])
        if row_concept and sum(b.concept == row_concept for b in run) >= 2:
            for b in run:
                if not b.concept and b is not header:
                    b.concept = row_concept
        # Columns: per position, propagate a concept that repeats across rows.
        # Data-row-aware column election: the demoted header and any non-full-
        # width rows (banner/spanning/subtotal rows carry FEWER cells — faithful
        # structure, never padded) are dropped from the vote so they can't poison
        # it, and a run in which no column concept repeats across the data rows is
        # a KEY-VALUE run whose rows are marked kv_table and left per-row (the
        # renderer emits a uniform td with the value concept as an inner span).
        election = run
        data_rows = [b for b in run if b is not header and b.cells]
        if data_rows:
            modal = Counter(len(b.cells) for b in data_rows).most_common(1)[0][0]
            # "Full-width" DATA rows, matching table_recall's own definition
            # (len > modal/2): rows with far fewer cells are banner/span rows,
            # but ragged transcription can leave a real data row a cell or two
            # off the mode, so a strict ==modal filter would wrongly drop the
            # rows that actually carry the column concepts (mis-flagging the
            # table key-value). Excludes true spans without losing data rows.
            election = [b for b in data_rows if len(b.cells) > modal / 2] or data_rows
            # KEY-VALUE run: no EFFECTIVE column concept repeats across the
            # data rows (each row is its own label→value). The effective
            # concept is what the renderer tags — positional cell concept OR
            # a whole-cell entity — so a data table whose roles arrived as
            # whole-cell entities is NOT mis-classified key-value.
            eff = [_effective_cell_concepts(b) for b in election]
            w = max((len(e) for e in eff), default=0)
            repeats = False
            for col in range(w):
                cc = [e[col] for e in eff if col < len(e) and e[col]]
                u = _most_common(cc)
                if u and sum(c == u for c in cc) >= 2:
                    repeats = True
                    break
            # Require >= 3 full-width data rows before calling a run key-value:
            # with only 2 rows "no column concept repeats" is unreliable (a
            # small or slightly mis-aligned DATA table trips it), and
            # mis-rendering a data table as key-value moves its values a level
            # deeper and costs F1. 3 rows is the same floor table_recall uses
            # for its row-level table checks.
            if not repeats and len(election) >= 3:
                for b in run:
                    b.kv_table = True
                continue
        width = max(len(b.cell_concepts) for b in election)
        for col in range(width):
            col_concepts = [b.cell_concepts[col] for b in election if col < len(b.cell_concepts)]
            uniform = _most_common(col_concepts)
            if uniform and sum(c == uniform for c in col_concepts) >= 2:
                for b in run:
                    if b is header:
                        continue  # a demoted header row stays plain cells
                    if col < len(b.cells):
                        while len(b.cell_concepts) < len(b.cells):
                            b.cell_concepts.append("")
                        if col < len(b.cell_entities) and any(
                            b.cells[col].strip() == b.cells[col][s.start : s.end].strip()
                            for s in b.cell_entities[col]
                        ):
                            # A WHOLE-cell entity is direct row-local evidence
                            # of this cell's role — a propagated column concept
                            # is an inference and must not bury it (the renderer
                            # prefers the positional concept). Partial spans are
                            # sub-values, not the cell's role: they compose with
                            # a column concept (tagged td + inner spans), so
                            # they don't block the fill.
                            continue
                        if not b.cell_concepts[col]:
                            b.cell_concepts[col] = uniform


# Short list items below this length are treated as a uniform group (one
# shared concept); longer, prose-like items keep their per-item concept.
_SHORT_ITEM_CHARS = 80


def _list_runs(blocks: list[Block]) -> list[list[Block]]:
    runs: list[list[Block]] = []
    current: list[Block] = []
    for b in blocks:
        if b.structure == "item":
            current.append(b)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def propagate_list_consistency(blocks: list[Block]) -> None:
    """Give a uniform short-item list one shared concept.

    A list whose items are short (median length below the threshold) names a
    repeating role — the most-common item concept is propagated to all items.
    Prose-length items keep their individual concepts (they carry specific
    meaning), matching the row-vs-clause distinction for tables.
    """
    for run in _list_runs(blocks):
        if len(run) < 2:
            continue
        lengths = sorted(len(b.text) for b in run)
        median = lengths[len(lengths) // 2]
        if median > _SHORT_ITEM_CHARS:
            continue  # prose items — keep per-item concepts
        shared = _most_common([b.concept for b in run])
        if shared and sum(b.concept == shared for b in run) >= 2:
            for b in run:
                if not b.concept:
                    b.concept = shared


PLAN_SYSTEM_PROMPT = prompt("plan_system")

# Block structures that define the document's role skeleton — what the
# concept planner needs to see from every document.
_SKELETON_STRUCTURES = {"heading", "field", "row"}
# Effectively "the whole document" for real corpora (observed skeletons run
# 25-420 lines; Underlease-style docs sat just past the old 400 cap and were
# truncated) while still bounding a pathological 500-page manual.
_PLAN_MAX_LINES_PER_DOC = 2000
# Max documents whose skeleton feeds the single roster-planning call. Depth
# beats breadth: planning sees FEW documents IN FULL rather than many
# truncated ones (same worst-case line budget as the old 20 x 400). The
# roster only needs the RECURRING roles of a same-kind docset; every document
# is still fully labeled in Pass B (which extends the roster).
_PLAN_MAX_DOCS = 4
# Unseeded runs label this many documents (the largest, same sort as planning)
# as a PILOT first; their observed evidence promotes the planned roster to
# confirmed before the rest of the batch labels against it. Seeded runs skip
# staging — the seed is already confirmed.
_PILOT_MAX_DOCS = 5


def render_skeleton_listing(docs: Mapping[str, list[Block]]) -> str:
    """Compact cross-document skeleton: headings/fields/rows of every doc.

    The first paragraph after each heading is included too — value-kind
    concepts (names, addresses, dates, amounts) can only be planned if the
    planner sees at least one actual value per section, not just headings.
    """
    lines: list[str] = []
    for name, blocks in docs.items():
        lines.append(f"== {name} ==")
        count = 0
        skipped = 0
        prev_structure = ""
        for b in blocks:
            keep = b.structure in _SKELETON_STRUCTURES or (
                # first item of each list run, and the first paragraph after a
                # heading (the section's leading value line)
                (b.structure in ("item", "p") and prev_structure == "heading")
                or (b.structure == "item" and prev_structure != "item")
            )
            prev_structure = b.structure
            if not keep:
                continue
            if count >= _PLAN_MAX_LINES_PER_DOC:
                skipped += 1
                continue
            head = f"{b.structure}" + (f" [{b.lim}]" if b.lim else "")
            lines.append(f"{head}: {b.flat_text()[:_SNIPPET_CHARS]}")
            count += 1
        if skipped:
            # No silent caps: an unseen skeleton chunk must be visible in the
            # listing itself, not discovered later as missing concepts.
            lines.append(f"… ({skipped} further skeleton line(s) omitted for size)")
    return "\n".join(lines)


def plan_concept_roster(
    docs: Mapping[str, list[Block]],
    *,
    config: llm.LLMConfig,
    cache_dir: Path | str | None = None,
    debug: bool = False,
    log: Callable[[str], None] = lambda _m: None,
    refine: bool = True,
) -> dict[str, str]:
    """ONE call over every document's skeleton → the shared concept roster.

    This is the cross-document step: concepts are derived from the roles the
    documents demonstrably share (every heading of every document is visible
    side by side), then the per-document labeling applies them. Best-effort —
    a failure returns an empty roster and labeling proceeds roster-free.

    With ``refine=True`` (default) a grounded second turn asks the model to ADD
    recurring roles it missed in the draft (add-only — no merge/rename), which
    improves roster completeness and cross-run stability at the cost of one
    extra call. Set ``refine=False`` to skip it.
    """
    # Cap the planning input. Sample the LARGEST documents (most blocks): a
    # richer skeleton seeds a more complete roster, so the biggest docs cover
    # the most recurring roles from the fewest samples. Ties broken by name for
    # determinism. Pass B still labels every document.
    # TODO: smarter sampling — cluster documents by skeleton shape and sample
    # per cluster, so a heterogeneous docset's minority layouts are represented
    # instead of being crowded out by the largest documents.
    planning_docs = docs
    if len(docs) > _PLAN_MAX_DOCS:
        chosen = sorted(docs.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:_PLAN_MAX_DOCS]
        planning_docs = dict(chosen)
        log(
            f"Pass B.1: sampling {_PLAN_MAX_DOCS} of {len(docs)} docs "
            "(largest skeletons) for roster planning"
        )

    log(f"Pass B.1: planning concept roster from {len(planning_docs)} doc skeleton(s)...")
    listing = render_skeleton_listing(planning_docs)
    cache_write(cache_dir, "plan_roster_input.txt", listing, debug=debug)
    try:
        if refine:
            # Two-turn grounded build: draft the roster, then have the model
            # ADD recurring roles it missed, grounded on the same skeletons.
            # Add-only — no synonym merging — so it can only raise recall.
            draft_raw, raw = llm.call_with_refinement(
                config,
                system_prompt=PLAN_SYSTEM_PROMPT,
                user_content=[{"type": "text", "text": listing}],
                refine_instruction=[{"type": "text", "text": prompt("roster_complete")}],
                cache=True,
            )
            cache_write(
                cache_dir, "plan_roster_draft_raw.json", strip_fences(draft_raw), debug=debug
            )
        else:
            # Single-call roster: the draft only, without the add-only completion turn.
            raw = llm.call(
                config,
                system_prompt=PLAN_SYSTEM_PROMPT,
                user_content=[{"type": "text", "text": listing}],
                cache=True,
            )
        cache_write(cache_dir, "plan_roster_raw.json", strip_fences(raw), debug=debug)
        payload = _parse_labels_json(raw)
    except Exception as exc:
        log(f"[label] roster planning failed ({exc}); labeling proceeds without it")
        return {}
    roster: dict[str, str] = {}
    for name, description in (payload.get("concepts", {}) or {}).items():
        concept = sanitize_concept(str(name))
        if concept:
            # Keep the FULL description — it becomes the schema.json `role`.
            # render_roster truncates for the compact in-prompt listing.
            roster[concept] = str(description)
    log(f"Pass B.1: planned {len(roster)} shared concept(s)")
    return roster


def wrap_detected_values(blocks: list[Block]) -> None:
    """Isolate currency/percentage values the labeler left bare in block text.

    Empty-concept spans render as typed ``dg:chunk`` value elements, so the
    value is independently extractable without inventing a semantic role.
    """
    for b in blocks:
        if not b.text:
            continue
        spans = list(b.entities)
        occupied = [(s.start, s.end) for s in spans]
        for m in _VALUE_SCAN_RE.finditer(b.text):
            s, e = m.start(), m.end()
            if any(s < oe and os < e for os, oe in occupied):
                continue
            spans.append(Span(start=s, end=e, concept=""))
            occupied.append((s, e))
        spans.sort(key=lambda sp: sp.start)
        kept: list[Span] = []
        for sp in spans:
            if kept and sp.start < kept[-1].end:
                continue
            kept.append(sp)
        b.entities = kept


def _chunks(blocks: list[Block]) -> list[list[Block]]:
    return [
        blocks[i : i + _MAX_BLOCKS_PER_CALL] for i in range(0, len(blocks), _MAX_BLOCKS_PER_CALL)
    ]


def _observe(roster: dict[str, RosterEntry], concept: str, kind: str, example: str) -> None:
    """Record one observed use of *concept*: example, kind, confirmation.

    Planned entries keep their description AND gain observed verbatim examples;
    coined concepts enter with an example and no description (described later by
    ``describe_concepts``). Frozen (schema-seeded) entries are never mutated —
    the seed is authoritative and its rendered text stays cache-stable.
    """
    if not concept:
        return
    entry = roster.setdefault(concept, RosterEntry())
    if entry.frozen:
        return
    entry.confirmed = True
    if kind and not entry.kind:
        entry.kind = kind
    ex = example.strip()[:_ROSTER_EXAMPLE_CHARS]
    if ex and ex not in entry.examples and len(entry.examples) < _SCHEMA_MAX_EXAMPLES:
        entry.examples.append(ex)


def _update_roster(roster: dict[str, RosterEntry], blocks: list[Block]) -> None:
    """Enrich the roster from labeled blocks (kind mapping mirrors derive_schema)."""
    for b in blocks:
        if b.concept:
            if b.structure == "row":
                _observe(roster, b.concept, "row", "")
            elif b.structure == "field":
                _observe(roster, b.concept, "inline", b.value)
            else:
                _observe(roster, b.concept, "section", b.flat_text())
        if b.value_concept:
            _observe(roster, b.value_concept, "inline", b.text)
        if b.group_concept:
            _observe(roster, b.group_concept, "section", "")
        if b.lim_concept:
            _observe(roster, b.lim_concept, "inline", b.lim)
        for span in b.entities:
            _observe(roster, span.concept, "inline", b.text[span.start : span.end])
        for span in b.label_entities:
            _observe(roster, span.concept, "inline", b.label[span.start : span.end])
        for i, cell_concept in enumerate(b.cell_concepts):
            if cell_concept:
                _observe(roster, cell_concept, "inline", b.cells[i] if i < len(b.cells) else "")
        for i, spans in enumerate(b.cell_entities):
            cell = b.cells[i] if i < len(b.cells) else ""
            for span in spans:
                _observe(roster, span.concept, "inline", cell[span.start : span.end])


def describe_concepts(
    concepts: Mapping[str, str],
    *,
    config: llm.LLMConfig,
    cache_dir: Path | str | None = None,
    debug: bool = False,
    log: Callable[[str], None] = lambda _m: None,
) -> dict[str, str]:
    """One LLM call → a real one-line role description for each concept.

    *concepts* maps a concept name to an observed example value. Used for concepts
    coined DURING labeling (the planner only described the roles it foresaw), so
    their schema ``role`` is a description rather than a humanized name. The model
    is told to describe the field's role, never the example value. Best-effort: a
    failure (or an unparseable reply) returns ``{}`` and the caller falls back to
    the humanized name.
    """
    if not concepts:
        return {}
    lines = ["Concepts to describe (name — example value):"]
    for name, example in concepts.items():
        ex = str(example).strip()[:80]
        lines.append(f"- {name}" + (f' — "{ex}"' if ex else ""))
    try:
        raw = llm.call(
            config,
            system_prompt=prompt("describe_concepts"),
            user_content=[{"type": "text", "text": "\n".join(lines)}],
            cache=True,
        )
        cache_write(cache_dir, "describe_concepts_raw.json", strip_fences(raw), debug=debug)
        payload = _parse_labels_json(raw)
    except Exception as exc:  # description is best-effort; humanized names cover the gap
        log(f"[label] concept description failed ({exc}); using humanized names")
        return {}
    out: dict[str, str] = {}
    for name, desc in (payload.get("descriptions", {}) or {}).items():
        concept = sanitize_concept(str(name))
        text = str(desc).strip()
        if concept and text:
            out[concept] = text
    return out


_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _humanize_concept(name: str) -> str:
    """Readable fallback description from a PascalCase concept name.

    ``ClientAddress`` -> ``Client Address``; ``SOWFees`` -> ``SOW Fees``. Used as
    the schema ``role`` for concepts the planner never described, so the role is
    a description and never a leaked value.
    """
    words = _CAMEL_SPLIT_RE.sub(" ", name).split()
    return " ".join(words) if words else name


def derive_schema(
    docs: Mapping[str, list[Block]],
    roster: Mapping[str, RosterEntry],
    descriptions: Mapping[str, str],
) -> Schema:
    """Build a v1-format ``Schema`` from the already-labeled blocks.

    Pure derivation — no LLM call and no renaming: the concept names stay
    exactly as the pipeline produced them. Each field is filled from what the
    run already knows:

    - ``role``         — the planned roster description (or an observed snippet);
    - ``example``      — the first observed value/text for the concept;
    - ``kind``         — section (containers/clauses), row (table rows), or
                         inline (atomic values: entities, value-headings, cells);
    - ``parent_role``  — the nearest enclosing concept in the labeled tree.

    Observation wins; a roster entry's seeded ``kind``/``examples``/``parent``
    fill the gaps for concepts NOT observed in *docs* (e.g. an incremental run
    whose new documents never use an old tag), so re-deriving the schema never
    degrades fields a previous run established.
    """
    # Examples must be internally consistent per concept: an inline concept's
    # examples are observed VALUES, a section concept's examples are how the
    # section announces itself (its heading). Body snippets of blocks labeled
    # with a section concept are kept in a fallback bucket, used only when the
    # concept never appears as a heading — never mixed with heading examples.
    examples: dict[str, list[str]] = {}
    body_examples: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}
    parents: dict[str, str] = {}

    def _squash(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _caption_like(name: str, text: str) -> bool:
        """'Effective Date' is the CAPTION of EffectiveDate, not a value."""
        t = _squash(text)
        return bool(t) and (t == _squash(name) or t == _squash(_humanize_concept(name)))

    def record(raw: str, kind: str, parent: str, example: str, *, body: bool = False) -> str:
        name = sanitize_concept(raw)
        if not name:
            return ""
        kinds.setdefault(name, kind)
        parents.setdefault(name, parent)
        ex = (example or "").strip()[:80]
        if ex and not (kind == "inline" and _caption_like(name, ex)):
            store = body_examples if body else examples
            seen = store.setdefault(name, [])
            if ex not in seen and len(seen) < _SCHEMA_MAX_EXAMPLES:
                seen.append(ex)
        return name

    def walk(node: Node, parent_concept: str) -> None:
        if node.kind == "section":
            head = (
                node.children[0].block if node.children and node.children[0].kind == "h" else None
            )
            sec = ""
            if head is not None and head.concept:
                sec = record(head.concept, "section", parent_concept, head.text)
            if head is not None and head.value_concept:
                record(head.value_concept, "inline", sec or parent_concept, head.text)
            if head is not None:
                for sp in head.entities:
                    value = head.text[sp.start : sp.end]
                    record(sp.concept, "inline", sec or parent_concept, value)
            for child in node.children:
                walk(child, sec or parent_concept)
            return
        block = node.block
        if node.kind in ("p", "li") and block is not None:
            own = ""
            if block.concept:
                # body snippet: fallback example only (see buckets above)
                own = record(block.concept, "section", parent_concept, block.text, body=True)
            for sp in block.entities:
                value = block.text[sp.start : sp.end]
                record(sp.concept, "inline", own or parent_concept, value)
        elif node.kind == "tr" and block is not None:
            grp = ""
            if block.group_concept:
                grp = record(block.group_concept, "section", parent_concept, "")
            row = ""
            if block.concept:
                row = record(block.concept, "row", grp or parent_concept, "")
            for i, cc in enumerate(block.cell_concepts):
                cell = block.cells[i] if i < len(block.cells) else ""
                record(cc, "inline", row or grp or parent_concept, cell)
        for child in node.children:
            walk(child, parent_concept)

    for blocks in docs.values():
        tree = build_tree(blocks)
        for child in tree.children:
            walk(child, "")

    schema = Schema()
    for name in sorted(set(roster) | set(kinds)):
        entry = roster.get(name)
        exs = (
            examples.get(name) or body_examples.get(name) or (list(entry.examples) if entry else [])
        )
        kind = kinds.get(name) or (entry.kind if entry else "") or "inline"
        parent = parents.get(name) or (entry.parent if entry else "")
        schema.add(
            SchemaTag(
                name=name,
                role=str(descriptions.get(name) or _humanize_concept(name)),
                kind=kind,
                example=exs[0] if exs else "",
                examples=exs,
                parent_role=parent,
            )
        )
    return schema


def _seed_entries_from_schema(schema: Schema) -> dict[str, RosterEntry]:
    """Roster entries from an exported schema — full fidelity, frozen.

    Names/parents are normalized through ``sanitize_concept`` so a hand-edited
    schema meets the labeling vocabulary's naming rules; ``kind`` is kept only
    when valid. Entries are confirmed (the seed is authoritative) and frozen
    (observations never mutate them, keeping the rendered roster cache-stable
    for the whole batch).
    """
    roster: dict[str, RosterEntry] = {}
    for tag in schema.tags.values():
        name = sanitize_concept(tag.name)
        if not name:
            continue
        roster[name] = RosterEntry(
            description=str(tag.role or ""),
            examples=[
                str(ex).strip()[:_ROSTER_EXAMPLE_CHARS]
                for ex in (tag.examples or ([tag.example] if tag.example else []))
            ][:_SCHEMA_MAX_EXAMPLES],
            kind=tag.kind if tag.kind in VALID_KINDS else "",
            parent=sanitize_concept(tag.parent_role or ""),
            confirmed=True,
            frozen=True,
        )
    return roster


def _label_one_document(
    doc_name: str,
    blocks: list[Block],
    roster: dict[str, RosterEntry],
    *,
    config: llm.LLMConfig,
    cache_dir: Path | str | None,
    debug: bool,
    log: Callable[[str], None],
) -> list[str]:
    """Label one document's blocks against (and into) the shared roster."""
    warnings: list[str] = []
    stem = Path(doc_name).stem
    # One roster snapshot per document: every call for this document reuses the
    # same rendered block, so it stays byte-stable (and cacheable) across the
    # document's chunks and its section retry.
    roster_blocks = _roster_content_blocks(roster, model=config.model)
    for chunk_idx, chunk in enumerate(_chunks(blocks)):
        listing = render_block_listing(doc_name, chunk)
        user_content = [*roster_blocks, {"type": "text", "text": listing}]
        user_text = "\n\n".join(str(part["text"]) for part in user_content)
        cache_write(
            cache_dir, f"label_{stem}_c{chunk_idx + 1:02d}_input.txt", user_text, debug=debug
        )
        for attempt in range(2):
            try:
                raw = llm.call(
                    config,
                    system_prompt=SYSTEM_PROMPT,
                    user_content=user_content,
                    cache=True,
                )
                # Functional file the next run reloads — written regardless of --debug.
                cache_write(
                    cache_dir,
                    f"label_{stem}_c{chunk_idx + 1:02d}_raw.json",
                    strip_fences(raw),
                    debug=True,
                )
                payload = _parse_labels_json(raw)
                warnings.extend(
                    apply_labels(chunk, payload.get("labels", {}) or {}, doc_name=doc_name)
                )
            except Exception as exc:  # labeling must never lose the transcription
                msg = f"labeling failed for {doc_name} chunk {chunk_idx + 1}: {exc}"
                log(f"[label] {msg}")
                if attempt:
                    warnings.append(msg)
                continue
            labeled = sum(1 for b in chunk if b.concept)
            if attempt or labeled >= len(chunk) * _MIN_LABELED_FRACTION:
                break
            log(
                f"[label] {doc_name} chunk {chunk_idx + 1} under-labeled "
                f"({labeled}/{len(chunk)}); retrying"
            )
        _update_roster(roster, chunk)
    # Force coverage of untagged sections: an unlabeled heading drops a
    # whole section's concept — meaningful information left untagged.
    # Re-label just those blocks (at most one extra call, only when gaps
    # exist); the model still supplies the concept name.
    missing = [b for b in blocks if _needs_label(b)]
    if missing:
        listing = prompt("section_retry") + "\n\n" + render_block_listing(doc_name, missing)
        user_content = [*roster_blocks, {"type": "text", "text": listing}]
        user_text = "\n\n".join(str(part["text"]) for part in user_content)
        cache_write(cache_dir, f"label_{stem}_section_retry_input.txt", user_text, debug=debug)
        try:
            raw = llm.call(
                config,
                system_prompt=SYSTEM_PROMPT,
                user_content=user_content,
                cache=True,
            )
            cache_write(
                cache_dir,
                f"label_{stem}_section_retry_raw.json",
                strip_fences(raw),
                debug=debug,
            )
            warnings.extend(
                apply_labels(
                    missing, _parse_labels_json(raw).get("labels", {}) or {}, doc_name=doc_name
                )
            )
            _update_roster(roster, missing)
        except Exception as exc:
            warnings.append(f"section retry failed for {doc_name}: {exc}")
        recovered = sum(1 for b in missing if b.concept)
        log(f"Pass B: {doc_name}: section retry labeled {recovered}/{len(missing)}")
    # Enforce the consistency the model can't be trusted to hold: uniform
    # table columns/rows and uniform short-list-item roles.
    propagate_table_consistency(blocks)
    propagate_list_consistency(blocks)
    wrap_detected_values(blocks)
    labeled = sum(1 for b in blocks if b.concept)
    log(f"Pass B: {doc_name}: {labeled}/{len(blocks)} block(s) labeled")
    return warnings


def _promote_pilot(
    docs: Mapping[str, list[Block]],
    pilot: list[str],
    roster: dict[str, RosterEntry],
    descriptions: Mapping[str, str],
    log: Callable[[str], None],
) -> None:
    """Fold the pilot documents' observations into the roster.

    Kind, examples, and confirmation already accumulated via ``_update_roster``
    while the pilot labeled; what only the labeled TREE knows is each concept's
    parent. One pure ``derive_schema`` call (no LLM) recovers it, so schema.json
    keeps hierarchy even for concepts the later documents never nest — and the
    remaining documents label against a confirmed vocabulary.
    """
    schema = derive_schema({name: docs[name] for name in pilot}, roster, descriptions)
    for tag in schema.tags.values():
        entry = roster.get(tag.name)
        if entry is not None and not entry.frozen and not entry.parent:
            entry.parent = tag.parent_role
    confirmed = sum(1 for e in roster.values() if e.confirmed)
    log(f"Pass B: pilot confirmed {confirmed}/{len(roster)} concept(s)")


def label_documents(
    docs: Mapping[str, list[Block]],
    *,
    config: llm.LLMConfig,
    cache_dir: Path | str | None = None,
    debug: bool = False,
    log: Callable[[str], None] = lambda _m: None,
    roster_refine: bool = True,
    roster_seed: Mapping[str, str] | None = None,
    schema_seed: Schema | None = None,
) -> list[str]:
    """Label every document, chunked, carrying the roster between calls.

    Best-effort per chunk: a failed call leaves that chunk unlabeled (still a
    valid, renderable document) and is reported as a warning; later chunks
    proceed with whatever roster exists so far.

    *schema_seed* (from a user-supplied ``--schema-path`` or the docset's own
    ``schema.json`` on an incremental run) is used as the starting roster with
    FULL fidelity — role descriptions, curated examples, kind, hierarchy —
    INSTEAD of the planning call (Pass B.1). This makes the vocabulary
    deterministic and authoritative; per-document labeling still extends it for
    roles the seed does not cover. *roster_seed* is the legacy flat
    ``{concept: description}`` seed (``cache/concept_roster.json``) — same
    semantics, no examples/kind/hierarchy; ignored when *schema_seed* is given.

    Unseeded runs are STAGED: after planning, the largest documents label
    first (a pilot), their observations promote the planned roster to
    confirmed — real examples, observed kinds, tree-derived parents — and the
    rest of the batch labels against that confirmed vocabulary.

    With *cache_dir* set, each call's RAW return is written as
    ``label_<stem>_cNN_raw.json`` and the final concept roster as
    ``concept_roster.json`` — functional files the next run reloads. With
    *debug* additionally set, the per-call input listings
    (``label_<stem>_cNN_input.txt``) and section-retry artifacts are written
    too (debug-only; never read back).
    """
    total_blocks = sum(len(b) for b in docs.values())
    log(f"Pass B: labeling {total_blocks} block(s) across {len(docs)} doc(s)...")
    warnings: list[str] = []
    # The roster (a.k.a. the docset's concept "schema" — see --schema-path; not
    # renamed because it ACCUMULATES during labeling, unlike a static schema).
    # When the caller supplies a seed, it IS the roster and planning is skipped;
    # otherwise one planning call sees every document's skeleton side by side
    # and names the SHARED roles. Either way the per-document calls below apply
    # it and may extend it for roles it missed, which propagate to later docs.
    roster: dict[str, RosterEntry]
    if schema_seed is not None:
        roster = _seed_entries_from_schema(schema_seed)
        log(f"Pass B.1: seeded roster from schema ({len(roster)} concept(s)); planning skipped")
    elif roster_seed is not None:
        # Legacy flat seed: confirmed (it is a prior run's vocabulary) but not
        # frozen — it carries no examples, so observed ones still enrich it.
        roster = {
            name: RosterEntry(description=str(desc), confirmed=True)
            for name, desc in roster_seed.items()
        }
        log(f"Pass B.1: seeded roster from schema ({len(roster)} concept(s)); planning skipped")
    else:
        planned = plan_concept_roster(
            docs, config=config, cache_dir=cache_dir, debug=debug, log=log, refine=roster_refine
        )
        roster = {name: RosterEntry(description=desc) for name, desc in planned.items()}
    # The planned/seeded descriptions are real role descriptions. Snapshot them
    # before per-document labeling coins description-less concepts, so the
    # schema `role` comes from descriptions (never a leaked value).
    descriptions = {name: entry.description for name, entry in roster.items() if entry.description}

    # Pilot staging (unseeded runs only): the LARGEST documents (same sort as
    # planning) label first, then _promote_pilot folds their observations into
    # the roster so the remainder labels against confirmed concepts. Seeded
    # runs skip staging — every seed entry is already confirmed.
    order = list(docs)
    pilot: list[str] = []
    if schema_seed is None and roster_seed is None and len(docs) > _PILOT_MAX_DOCS:
        largest = sorted(docs.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:_PILOT_MAX_DOCS]
        chosen = {name for name, _ in largest}
        pilot = [name for name in docs if name in chosen]
        order = pilot + [name for name in docs if name not in chosen]
        log(f"Pass B: pilot stage — labeling the {len(pilot)} largest doc(s) first")

    for idx, doc_name in enumerate(order):
        warnings.extend(
            _label_one_document(
                doc_name,
                docs[doc_name],
                roster,
                config=config,
                cache_dir=cache_dir,
                debug=debug,
                log=log,
            )
        )
        if pilot and idx == len(pilot) - 1:
            _promote_pilot(docs, pilot, roster, descriptions, log)

    log(f"Pass B: roster holds {len(roster)} concept(s)")
    # Functional file the next run reloads — written regardless of --debug.
    # Kept in the legacy flat {concept: description-or-example} shape (the
    # full-fidelity vocabulary lives in schema.json).
    flat = {
        name: (entry.description or (entry.examples[0] if entry.examples else ""))
        for name, entry in roster.items()
    }
    cache_write(
        cache_dir,
        "concept_roster.json",
        json.dumps(flat, indent=2, ensure_ascii=False),
        debug=True,
    )
    # Final artifact: the docset vocabulary in the schema.json format
    # (role/example/kind/parent_role) — concept names unchanged. Written to the
    # docset dir (cache_dir's parent) so the viewer and eval find it; kept
    # regardless of --debug, like the v1 schema.json.
    if cache_dir is not None:
        try:
            # Concepts coined during labeling have no planned description; get a
            # real one-line role for them (their roster entry holds observed
            # examples), so the schema role is a description, not a humanized name.
            undescribed = {
                name: (entry.examples[0] if entry.examples else "")
                for name, entry in roster.items()
                if not entry.description
            }
            if undescribed:
                descriptions.update(
                    describe_concepts(
                        undescribed, config=config, cache_dir=cache_dir, debug=debug, log=log
                    )
                )
            schema = derive_schema(docs, roster, descriptions)
            schema.save(Path(cache_dir).parent / "schema.json")
            log(f"Pass B: wrote schema.json ({len(schema.tags)} tag(s))")
        except Exception as exc:  # schema export must never sink a labeled batch
            log(f"[label] schema.json export failed ({exc}); skipped")
    return warnings
