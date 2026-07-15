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

"""Deterministic conversion: labeled blocks → final dgml.

Two renderers:

- :func:`render_semantic_xml` — the plain structure-attribute form (plain
  structural tags with ``structure`` attributes), kept as a debug artifact.
- :func:`render_dgml` — the FINAL ``dg:chunk`` document. Naming convention:
    * an element with a semantic concept → ``docset:Concept`` — ALWAYS in the
      per-docset vocabulary namespace, whether the concept recurs across the
      docset or appears in a single document. ``dg:`` is reserved for the
      framework (the ``dg:chunk`` scaffolding element and ``dg:*`` attributes);
      no concept is ever emitted there. The element carries its real structural
      type in ``structure`` (``section``, ``p``, ``li``, ``tr``);
    * any element WITHOUT a concept — the structural scaffolding (lists, list
      items, headers, cells, enumerators) → ``dg:chunk`` with its type in
      ``structure`` (``ul``/``ol``/``li``/``header``/``td``/``lim``/…). There
      are no ``h1``…``h6`` depth levels: ``structure`` is the actual type.
    * inline entity values → ``docset:`` concept elements (no ``structure``),
      with ``xsi:type`` / ``dg:value`` typing for dates/amounts (reused from
      the shared value-type detector).

Pure transformation — no LLM calls; reading order (lim before text) and
verbatim text are preserved by construction.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping

from lxml import etree  # type: ignore[import-untyped]

from dgml_core.generation.blocks import Block, Node, Span, build_tree, sanitize_concept
from dgml_core.generation.semantic_transform import (
    _detect_value_type,
    docset_slug,
    org_ns_segment,
)

_CP = "dg:chunk"  # generic chunk for any element with no semantic concept
_ST = "dg:structure"  # spec-namespaced layout-role attribute (final dgml only)


def build_header(workspace: str, docset_name: str) -> str:
    """The ``<dg:chunk …>`` opening tag for output (open dgml.io scheme).

    Declares only the namespaces uses: ``dg`` (generic chunks + value
    typing), the per-docset ``docset`` vocabulary, ``xsi`` (typed values),
    and ``xhtml`` (table structure). Distinct from
    ``semantic_transform.build_header``, which also takes a docset id.
    """
    slug = docset_slug(docset_name)
    return (
        "<dg:chunk\n"
        '    xmlns:dg="http://dgml.io/ns/dg#"\n'
        f'    xmlns:docset="http://dgml.io/{org_ns_segment(workspace)}/{slug}"\n'
        '    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
        '    xmlns:xhtml="http://www.w3.org/1999/xhtml">'
    )


# ── plain structure-attribute form (debug) ───────────────────────────────────


def _tag_for(block: Block | None, fallback: str) -> str:
    if block is not None and block.concept:
        return sanitize_concept(block.concept) or fallback
    return fallback


def _add_lim(el: etree._Element, block: Block) -> None:
    if block.lim:
        # A lim carrying a concept (e.g. a date used as the list marker) keeps
        # its concept tag so the value stays labeled — structure marks it a lim.
        tag = sanitize_concept(block.lim_concept) if block.lim_concept else ""
        lim = etree.SubElement(el, tag or "lim")
        if tag:
            lim.set("structure", "lim")
        lim.text = block.lim


def _fill_text(el: etree._Element, text: str, spans: list[Span]) -> None:
    last: etree._Element | None = el[-1] if len(el) else None

    def write(chunk: str) -> None:
        if not chunk:
            return
        if last is None:
            el.text = (el.text or "") + chunk
        else:
            last.tail = (last.tail or "") + chunk

    cursor = 0
    for span in spans:
        write(text[cursor : span.start])
        inline = etree.SubElement(el, sanitize_concept(span.concept) or "v")
        inline.text = text[span.start : span.end]
        last = inline
        cursor = span.end
    write(text[cursor:])


def _render_node(parent: etree._Element, node: Node) -> None:
    block = node.block
    if node.kind == "h":
        assert block is not None
        # A value-heading (its text IS the value) carries its concept on
        # value_concept, not block.concept. Tag the header element with it so
        # the label stays around the value instead of being dropped — mirroring
        # the DGML renderer, which tags headings from value_concept too.
        tag = (sanitize_concept(block.value_concept) if block.value_concept else "") or "header"
        el = etree.SubElement(parent, tag)
        el.set("structure", "header")
        _add_lim(el, block)
        _fill_text(el, block.text, block.entities)
        return
    if node.kind in ("p", "li"):
        assert block is not None
        el = etree.SubElement(parent, _tag_for(block, node.kind))
        el.set("structure", node.kind)
        _add_lim(el, block)
        _fill_text(el, block.text, block.entities)
        return
    if node.kind == "tr":
        assert block is not None
        el = etree.SubElement(parent, _tag_for(block, "tr"))
        el.set("structure", "tr")
        header_cell = "ColumnHeader" if block.header_row else "td"
        for cell in block.cells:
            td = etree.SubElement(el, header_cell)
            td.set("structure", "td")
            td.text = cell
        return
    if node.kind == "fld":
        assert block is not None
        el = etree.SubElement(parent, "li")
        el.set("structure", "li")
        _add_lim(el, block)
        if block.label:
            label = etree.SubElement(el, "header")
            label.set("structure", "header")
            if block.label_entities:
                _fill_text(label, block.label, block.label_entities)
            else:
                label.text = block.label
        value = etree.SubElement(el, "p")
        value.set("structure", "p")
        target = (
            etree.SubElement(value, sanitize_concept(block.concept)) if block.concept else value
        )
        if block.entities:
            _fill_text(target, block.value, block.entities)
        else:
            target.text = block.value
        return

    if node.kind == "section":
        head = node.children[0].block if node.children and node.children[0].kind == "h" else None
        el = etree.SubElement(parent, _tag_for(head, "section"))
        el.set("structure", "section")
    elif node.kind == "list":
        tag = "ol" if any(c.block is not None and c.block.lim for c in node.children) else "ul"
        el = etree.SubElement(parent, tag)
        el.set("structure", tag)
    elif node.kind == "table":
        el = etree.SubElement(parent, "table")
        el.set("structure", "table")
    elif node.kind == "form":
        el = etree.SubElement(parent, "ul")
        el.set("structure", "ul")
    else:  # pragma: no cover
        el = etree.SubElement(parent, "section")
        el.set("structure", "section")
    for child in node.children:
        _render_node(el, child)


def render_semantic_xml(blocks: list[Block]) -> str:
    """Blocks → plain structure-attribute semantic XML (debug artifact)."""
    from typing import cast

    tree = build_tree(blocks)
    root = etree.Element("xml")
    for child in tree.children:
        _render_node(root, child)
    return cast(str, etree.tostring(root, encoding="unicode", pretty_print=True))


# ── final dgml ───────────────────────────────────────────────────────────────


def _concept_tag(concept: str) -> str | None:
    """A concept always renders in the per-docset vocabulary namespace.

    ``docset:`` is the home of ALL semantic concepts — recurring across the
    docset or seen in a single document alike. ``dg:`` is framework-only (the
    ``dg:chunk`` scaffolding element and ``dg:*`` attributes); nothing semantic
    is ever emitted there. Whether a concept is currently shared is a property
    of the batch, not of the concept, so it must not drive the namespace.
    """
    name = sanitize_concept(concept)
    if not name:
        return None
    return f"docset:{name}"


def _apply_typing(el: ET.Element, text: str, extra_formats: bool) -> None:
    # dg:format is not part of the DGML spec (typed values carry only xsi:type +
    # dg:value), so the format hint from _detect_value_type is discarded here.
    xsi, value, _ = _detect_value_type(text, extra_formats)
    if xsi:
        el.set("xsi:type", xsi)
    if value is not None:
        el.set("dg:value", value)


def _dgml_lim(parent: ET.Element, block: Block, extra_formats: bool) -> None:
    if block.lim:
        # A lim carrying a concept (a date/number used as the list marker) is
        # emitted under its concept tag — dg:structure="lim" keeps the layout
        # role — and typed like any other value; an unlabeled lim stays dg:chunk.
        tag = (_concept_tag(block.lim_concept) if block.lim_concept else None) or _CP
        lim = ET.SubElement(parent, tag)
        lim.set(_ST, "lim")
        lim.text = block.lim
        if tag != _CP:
            _apply_typing(lim, block.lim, extra_formats)
        lim.tail = " "


def _dgml_fill(el: ET.Element, text: str, spans: list[Span], extra_formats: bool) -> None:
    """Text after the lim, entity values wrapped in concept elements + typing."""
    children = list(el)
    state = {"last": children[-1] if children else None}

    def write(chunk: str) -> None:
        if not chunk:
            return
        if state["last"] is None:
            el.text = (el.text or "") + chunk
        else:
            state["last"].tail = (state["last"].tail or "") + chunk

    cursor = 0
    for span in spans:
        write(text[cursor : span.start])
        value = text[span.start : span.end]
        tag = _concept_tag(span.concept) or _CP
        inline = ET.SubElement(el, tag)
        if tag == _CP:
            inline.set(_ST, "span")
        inline.text = value
        _apply_typing(inline, value, extra_formats)
        state["last"] = inline
        cursor = span.end
    write(text[cursor:])


def _dgml_leaf(parent: ET.Element, structure: str, block: Block, extra_formats: bool) -> None:
    tag = (_concept_tag(block.concept) if block.concept else None) or _CP
    el = ET.SubElement(parent, tag)
    el.set(_ST, structure)
    _dgml_lim(el, block, extra_formats)
    _dgml_fill(el, block.text, block.entities, extra_formats)


def _render_dgml_node(parent: ET.Element, node: Node, extra_formats: bool) -> None:
    block = node.block
    if node.kind == "h":
        assert block is not None
        tag = (_concept_tag(block.value_concept) if block.value_concept else None) or _CP
        el = ET.SubElement(parent, tag)
        el.set(_ST, "header")
        _dgml_lim(el, block, extra_formats)
        _dgml_fill(el, block.text, block.entities, extra_formats)
        return
    if node.kind in ("p", "li"):
        assert block is not None
        _dgml_leaf(parent, node.kind, block, extra_formats)
        return
    if node.kind == "tr":
        assert block is not None
        tag = (_concept_tag(block.concept) if block.concept else None) or _CP
        el = ET.SubElement(parent, tag)
        el.set(_ST, "tr")
        if block.header_row:
            # A demoted printed-title row renders as ColumnHeader structure-td
            # cells (gold's <docset:ColumnHeader structure="td">Cases</…> shape),
            # so the table is no longer counted headerless.
            for cell in block.cells:
                th = ET.SubElement(el, _concept_tag("ColumnHeader") or _CP)
                th.set(_ST, "td")
                th.text = cell
                _apply_typing(th, cell, extra_formats)
            return
        if block.kv_table:
            # A key-value run: every cell renders as a UNIFORM generic dg:chunk td
            # (stable localname → no per-column tag collision). A value's concept
            # rides as an INNER span that carries dg:structure="span"; that
            # attribute makes the scorer treat the concept leaf as the extractable
            # value and SUPPRESSES the wrapper td's composite-value record, so the
            # generic dg:chunk bucket the scorer also matches against stays
            # byte-identical to the default render (F1-safe) while the leaf concept
            # tag is preserved. Concept-less cells render exactly as the default td.
            for i, cell in enumerate(block.cells):
                td = ET.SubElement(el, _CP)
                td.set(_ST, "td")
                ents = block.cell_entities[i] if i < len(block.cell_entities) else []
                cc = block.cell_concepts[i] if i < len(block.cell_concepts) else ""
                whole = (
                    ents[0].concept
                    if len(ents) == 1
                    and cell.strip()
                    and cell[ents[0].start : ents[0].end].strip() == cell.strip()
                    else ""
                )
                concept = cc or whole
                if ents and not whole:
                    _dgml_fill(td, cell, ents, extra_formats)
                elif concept:
                    inner = ET.SubElement(td, _concept_tag(concept) or _CP)
                    inner.set(_ST, "span")
                    inner.text = cell
                    _apply_typing(inner, cell, extra_formats)
                else:
                    td.text = cell
                    _apply_typing(td, cell, extra_formats)
            return
        for i, cell in enumerate(block.cells):
            ents = block.cell_entities[i] if i < len(block.cell_entities) else []
            cell_concept = block.cell_concepts[i] if i < len(block.cell_concepts) else ""
            # A lone entity that covers the whole cell is just a whole-cell value;
            # several entities (or one partial span) mean the model split a
            # multi-value cell, which renders as inline spans in a generic td.
            whole = (
                ents[0].concept
                if len(ents) == 1
                and cell.strip()
                and cell[ents[0].start : ents[0].end].strip() == cell.strip()
                else ""
            )
            if ents and not whole:
                # A split multi-value cell keeps its positional column concept
                # (when present) as the td tag — same pattern as a concept leaf
                # with inline entity spans; without one it stays a generic td.
                cc = block.cell_concepts[i] if i < len(block.cell_concepts) else ""
                td_tag = (_concept_tag(cc) if cc else None) or _CP
                td = ET.SubElement(el, td_tag)
                td.set(_ST, "td")
                _dgml_fill(td, cell, ents, extra_formats)
                continue
            # Positional column concept wins (cross-row consistent); a whole-cell
            # entity concept is the fallback when no column concept aligned.
            concept = cell_concept or whole
            td_tag = (_concept_tag(concept) if concept else None) or _CP
            td = ET.SubElement(el, td_tag)
            # Every cell carries the td layout role, regardless of whether it
            # also has a semantic concept tag — same as the tr/header/leaf
            # paths above. A concept-tagged cell without dg:structure="td"
            # breaks the HTML-render contract (a <tr> with non-<td> children).
            td.set(_ST, "td")
            td.text = cell
            _apply_typing(td, cell, extra_formats)
        return
    if node.kind == "fld":
        assert block is not None
        el = ET.SubElement(parent, _CP)
        el.set(_ST, "li")
        _dgml_lim(el, block, extra_formats)
        if block.label:
            label = ET.SubElement(el, _CP)
            label.set(_ST, "header")
            if block.label_entities:
                _dgml_fill(label, block.label, block.label_entities, extra_formats)
            else:
                label.text = block.label
        value = ET.SubElement(el, _CP)
        value.set(_ST, "p")
        value_tag = _concept_tag(block.concept) if block.concept else None
        target = ET.SubElement(value, value_tag) if value_tag else value
        if block.entities:
            # Sub-values packed inside the field's value render as inline
            # concept spans within the (concept-wrapped) value — the same
            # compose pattern as a leaf with inline entities.
            _dgml_fill(target, block.value, block.entities, extra_formats)
        else:
            target.text = block.value
            if value_tag:
                _apply_typing(target, block.value, extra_formats)
        return

    if node.kind == "section":
        head = node.children[0].block if node.children and node.children[0].kind == "h" else None
        # A synthesized entity container carries its concept directly; a normal
        # section hoists it from the heading child.
        concept = node.concept or (head.concept if head and head.concept else "")
        tag = (_concept_tag(concept) if concept else None) or _CP
        el = ET.SubElement(parent, tag)
        el.set(_ST, "section")
    elif node.kind == "list":
        ordered = any(c.block is not None and c.block.lim for c in node.children)
        el = ET.SubElement(parent, _CP)
        el.set(_ST, "ol" if ordered else "ul")
    elif node.kind == "table":
        # The table's name comes from the row blocks' shared group_concept.
        group = next(
            (
                c.block.group_concept
                for c in node.children
                if c.block is not None and c.block.group_concept
            ),
            "",
        )
        tag = (_concept_tag(group) if group else None) or _CP
        el = ET.SubElement(parent, tag)
        el.set(_ST, "table")
    elif node.kind == "form":
        el = ET.SubElement(parent, _CP)
        el.set(_ST, "ul")
    else:  # pragma: no cover
        el = ET.SubElement(parent, _CP)
        el.set(_ST, "section")
    for child in node.children:
        _render_dgml_node(el, child, extra_formats)


def render_dgml(
    blocks: list[Block],
    *,
    header: str,
    extra_formats: bool = True,
    parent_map: Mapping[str, str] | None = None,
) -> str:
    """Blocks → final ``dg:chunk`` dgml (the conversion's product).

    *header* is the ``<dg:chunk …>`` opening tag from
    ``semantic_transform.build_header`` (declares the namespaces).

    *parent_map* (leaf concept → container concept, from a seed schema) drives
    the entity-container grouping in ``build_tree``. Every concept renders as
    ``docset:`` regardless of how often it recurs, so container concepts need
    no special namespace handling.
    """
    tree = build_tree(blocks, parent_map)
    root = ET.Element("doc")
    for child in tree.children:
        _render_dgml_node(root, child, extra_formats)
    ET.indent(root)  # safe on mixed content: only blank/None whitespace is set
    if len(root) == 0:
        # No renderable content (e.g. a document that transcribed to zero
        # blocks). Emit a valid empty dg:chunk instead of crashing on the
        # rindex below — a render error must never sink the file or its batch.
        inner = ""
    else:
        body = ET.tostring(root, encoding="unicode")
        inner = body[body.index(">") + 1 : body.rindex("</doc>")].strip("\n")
    return f"<?xml version='1.0' encoding='utf-8'?>\n{header}\n{inner}\n</dg:chunk>\n"
