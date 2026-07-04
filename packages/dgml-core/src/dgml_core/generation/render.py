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

"""Deterministic rendering: block tree → compact concept-attribute XML.

Structural elements are a small closed vocabulary (`sec`, `h`, `p`, `list`,
`li`, `table`, `tr`, `td`, `form`, `fld`, `label`, `value`); semantics ride
an optional `concept` attribute; inline entities become `<v concept="...">`
elements around their span. The rendered text is byte-identical to the
transcript by construction — tags are only ever inserted *around* spans.
"""

from __future__ import annotations

from typing import cast

from lxml import etree  # type: ignore[import-untyped]

from dgml_core.generation.blocks import Block, Node, Span, build_tree


def _set_concept(el: etree._Element, block: Block | None) -> None:
    if block is not None and block.concept:
        el.set("concept", block.concept)


def _fill_text_with_spans(el: etree._Element, text: str, spans: list[Span]) -> None:
    """Write `text` into `el` after any existing children, wrapping spans in <v>.

    Reading order: when `el` already has children (the `<lim>` enumerator),
    the text must serialize AFTER them — i.e. into the last child's tail, not
    `el.text` (element text always precedes children in XML, which would
    render "Heading text<lim>1.1</lim>" with the printed number displaced).
    Spans are validated non-overlapping and sorted by the labeling pass;
    concatenating itertext() reproduces lim + `text` in reading order.
    """
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
        v = etree.SubElement(el, "v")
        v.set("concept", span.concept)
        v.text = text[span.start : span.end]
        last = v
        cursor = span.end
    write(text[cursor:])


def _add_lim(el: etree._Element, block: Block) -> None:
    if block.lim:
        lim = etree.SubElement(el, "lim")
        lim.text = block.lim


def _render_leaf(parent: etree._Element, kind: str, block: Block) -> None:
    el = etree.SubElement(parent, kind)
    _set_concept(el, block)
    _add_lim(el, block)
    _fill_text_with_spans(el, block.text, block.entities)


def _render_node(parent: etree._Element, node: Node) -> None:
    if node.kind in ("h", "p", "li"):
        assert node.block is not None
        _render_leaf(parent, node.kind, node.block)
        return
    if node.kind == "tr":
        assert node.block is not None
        tr = etree.SubElement(parent, "tr")
        _set_concept(tr, node.block)
        for cell in node.block.cells:
            td = etree.SubElement(tr, "td")
            td.text = cell
        return
    if node.kind == "fld":
        assert node.block is not None
        fld = etree.SubElement(parent, "fld")
        _add_lim(fld, node.block)
        if node.block.label:
            label = etree.SubElement(fld, "label")
            label.text = node.block.label
        value = etree.SubElement(fld, "value")
        # A form field's concept names the VALUE's role (lessor-name = the
        # party string), so it marks exactly the value element — the wrapper
        # holds the printed label/lim, which are not part of the value.
        _set_concept(value, node.block)
        value.text = node.block.value
        return
    # containers: section / list / table / form
    el = etree.SubElement(parent, {"section": "sec"}.get(node.kind, node.kind))
    # A section's concept is its heading's concept (the labeler labels the
    # heading block; the renderer lifts it to the container).
    if node.kind == "section" and node.children:
        head = node.children[0]
        if head.kind == "h" and head.block is not None and head.block.concept:
            el.set("concept", head.block.concept)
    for child in node.children:
        _render_node(el, child)


def render_xml(blocks: list[Block], *, doc_name: str = "") -> str:
    """Render a labeled flat block list to the compact concept-attribute XML."""
    tree = build_tree(blocks)
    root = etree.Element("doc")
    if doc_name:
        root.set("name", doc_name)
    for child in tree.children:
        _render_node(root, child)
    return cast(str, etree.tostring(root, encoding="unicode", pretty_print=True))


def flat_text(blocks: list[Block]) -> str:
    """The transcript's full text — for verbatim checks against the render."""
    return "\n".join(b.flat_text() for b in blocks)
