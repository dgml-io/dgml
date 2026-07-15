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

"""Serialize extracted values into a ``dg:extraction`` element and back.

The grounded-extraction engine produces a values tree that mirrors the docset
schema: a leaf is ``{"text": ..., "locations": [{"page_number", "bounding_box"}]}``,
a container is a dict of child fields, and a collection is a list of such dicts.

A leaf may instead be **computed** (spec §7/§13) — derived by the model
reasoning over other extracted values rather than read off the page:
``{"text", "value"?, "computed": true, "derived_from": [dotted paths]}``.
Computed leaves serialize with ``dg:origin="computed"``, a mandatory
``dg:value``, and ``dg:itemprop="computedFrom"``/``dg:href`` naming their
source elements; each referenced source element gets a stable ``xml:id``
derived from its path so the ``dg:href`` targets resolve within the file.

Per spec §13, extracted values live inside the **core DGML file** as a
``<dg:extraction>`` element — a direct child of the root ``dg:chunk`` — holding
the schema's fields as ``docset:`` elements, each with its text content, a
``dg:origin`` built from the grounded locations, and (where the text normalizes)
an ``xsi:type`` + ``dg:value``. Two cases:

* **full-extraction** — the file already has a generated document tree;
  :func:`embed_extraction_into` adds/replaces the ``dg:extraction`` sibling.
* **extraction** — no document tree yet; :func:`standalone_extraction_doc`
  writes a minimal ``dg:chunk`` holding only the ``dg:extraction`` element.

:func:`dgml_xml_to_values` re-derives the values-shape JSON the CLI returns when
a caller asks for ``--format json``.
"""

from __future__ import annotations

import re
from typing import Any

from lxml import etree  # type: ignore[import-untyped]

from .extraction_schema import Tag, Vocabulary
from .generation.semantic_transform import _detect_value_type
from .matching import (
    LeafPath,
    get_at_path,
    is_computed_leaf,
    parse_path,
    path_to_str,
    walk_computed_leaves,
)

DG_NS = "http://dgml.io/ns/dg#"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
XML_NS = "http://www.w3.org/XML/1998/namespace"

_NSMAP_BASE = {"dg": DG_NS, "xsi": XSI_NS}
EXTRACTION_TAG = f"{{{DG_NS}}}extraction"
CHUNK_TAG = f"{{{DG_NS}}}chunk"
_ORIGIN_COMPUTED = "computed"


# ── values tree → DGML XML ───────────────────────────────────────────────────


def _origin_from_locations(locations: Any) -> str | None:
    """Build a ``dg:origin`` value from a grounded_field ``locations`` array.

    ``"<page> <x1> <y1> <x2> <y2>"`` per box, ``;``-joined for multi-box spans —
    the same format :mod:`dgml_core.xml_grounding` emits. Returns ``None`` when
    no usable box is present.
    """
    if not isinstance(locations, list):
        return None
    boxes: list[str] = []
    for loc in locations:
        if not isinstance(loc, dict):
            continue
        page = loc.get("page_number")
        box = loc.get("bounding_box")
        if not isinstance(page, int) or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            coords = " ".join(str(int(c)) for c in box)
        except (TypeError, ValueError):
            continue
        boxes.append(f"{page} {coords}")
    return "; ".join(boxes) if boxes else None


def _typed_value(text: str, value_type: str) -> tuple[str | None, str | None]:
    """Normalize *text* to (xsi_type, dg_value) for a schema-declared datatype.

    The schema's declared type wins over heuristic detection, so
    ``xsd:integer`` on "181 CREDITS" yields ``dg:value="181"``.
    """
    if value_type == "integer":
        # Strip thousands separators first so "8,500" → "8500", not "8"
        # (mirrors the decimal branch below).
        m = re.search(r"-?\d+", text.replace(",", ""))
        return ("integer", m.group()) if m else (None, None)
    if value_type == "decimal":
        m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return ("decimal", m.group()) if m else (None, None)
    # Dates/booleans/etc.: reuse the detector's normalization, keep declared type.
    _, dg_value, _ = _detect_value_type(text, extra_formats=False)
    return (value_type, dg_value if dg_value is not None else text.strip())


# ── computed fields (spec §7/§13) ────────────────────────────────────────────

_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _id_for_path(path: LeafPath) -> str:
    """A stable ``xml:id`` for a values-tree path: ``("LineItems", 0,
    "Quantity")`` → ``"line-items-0-quantity"``. Paths are unique within a
    tree, so the ids are too."""
    parts = [
        str(seg) if isinstance(seg, int) else _CAMEL_SPLIT_RE.sub("-", seg).lower() for seg in path
    ]
    ncname = "-".join(parts)
    if not ncname or not (ncname[0].isalpha() or ncname[0] == "_"):
        ncname = f"v-{ncname}"
    return ncname


def _collect_ref_ids(values: dict[str, Any]) -> dict[LeafPath, str]:
    """Assign an ``xml:id`` to every values-tree path some computed leaf's
    ``derived_from`` references. Malformed or dangling paths get no id — the
    corresponding ``dg:href`` entry is dropped at emission."""
    ids: dict[LeafPath, str] = {}
    for _path, leaf in walk_computed_leaves(values):
        refs = leaf.get("derived_from")
        if not isinstance(refs, list):
            continue
        for raw in refs:
            if not isinstance(raw, str):
                continue
            path = parse_path(raw)
            if path is None or path in ids or get_at_path(values, path) is None:
                continue
            ids[path] = _id_for_path(path)
    return ids


def count_dropped_refs(values: dict[str, Any]) -> int:
    """Number of ``derived_from`` entries across all computed leaves that will
    NOT emit a ``dg:href`` target — malformed path, non-string entry, or a
    path that dangles in *values*. Mirrors the drop logic in
    :func:`_collect_ref_ids`/:func:`_set_computed_attrs`; surfaced in
    ``extraction_stats.json`` so incomplete provenance is visible instead of
    silently shrinking the href list."""
    dropped = 0
    for _path, leaf in walk_computed_leaves(values):
        refs = leaf.get("derived_from")
        if not isinstance(refs, list):
            continue
        for raw in refs:
            if not isinstance(raw, str):
                dropped += 1
                continue
            path = parse_path(raw)
            if path is None or get_at_path(values, path) is None:
                dropped += 1
    return dropped


def unattributed_computed_fields(xml: str | bytes) -> list[str]:
    """Local names of ``dg:origin="computed"`` elements that carry no
    ``dg:href`` — a computed value whose sources can't be walked (spec §13
    says computed fields always name their sources). Used by the workspace
    consistency check."""
    root = etree.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    out: list[str] = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if el.get(f"{{{DG_NS}}}origin") == _ORIGIN_COMPUTED and not el.get(f"{{{DG_NS}}}href"):
            out.append(_local_name(el))
    return out


def _maybe_set_id(el: etree._Element, path: LeafPath, ids: dict[LeafPath, str]) -> None:
    if path in ids:
        el.set(f"{{{XML_NS}}}id", ids[path])


def _set_computed_attrs(
    el: etree._Element,
    value: dict[str, Any],
    value_type: str | None,
    ids: dict[LeafPath, str],
) -> None:
    """Emit the computed-field attribute set (spec §13): ``dg:origin="computed"``,
    a mandatory ``dg:value`` (the model's canonical ``value``, normalized), and
    ``dg:itemprop="computedFrom"``/``dg:href`` listing the resolvable sources."""
    el.set(f"{{{DG_NS}}}origin", _ORIGIN_COMPUTED)

    text_str = el.text or ""
    raw_value = value.get("value")
    canonical = str(raw_value).strip() if raw_value is not None else None
    if value_type is not None:
        xsi_type: str | None = value_type
        dg_value = canonical
        if dg_value is None:
            xsi_type, dg_value = _typed_value(text_str, value_type)
    else:
        xsi_type, dg_value, _ = _detect_value_type(canonical or text_str, extra_formats=False)
        if canonical is not None and dg_value is None:
            dg_value = canonical
    if dg_value is None:
        dg_value = text_str.strip()
    if xsi_type:
        el.set(f"{{{XSI_NS}}}type", xsi_type)
    el.set(f"{{{DG_NS}}}value", dg_value)

    refs = value.get("derived_from")
    hrefs: list[str] = []
    if isinstance(refs, list):
        for raw in refs:
            if not isinstance(raw, str):
                continue
            path = parse_path(raw)
            if path is not None and path in ids:
                hrefs.append(f"#{ids[path]}")
    if hrefs:
        el.set(f"{{{DG_NS}}}itemprop", "computedFrom")
        el.set(f"{{{DG_NS}}}href", "; ".join(hrefs))


def _add_field(
    parent: etree._Element,
    docset_ns: str,
    tag_name: str,
    value: dict[str, Any],
    value_type: str | None = None,
    *,
    path: LeafPath = (),
    ids: dict[LeafPath, str] | None = None,
) -> None:
    ids = ids or {}
    el = etree.SubElement(parent, f"{{{docset_ns}}}{tag_name}")
    _maybe_set_id(el, path, ids)
    text = value.get("text")
    text_str = "" if text is None else str(text)
    if text_str:
        el.text = text_str
    if is_computed_leaf(value):
        _set_computed_attrs(el, value, value_type, ids)
        return
    if text_str:
        if value_type is not None:
            xsi_type, dg_value = _typed_value(text_str, value_type)
        else:
            xsi_type, dg_value, _ = _detect_value_type(text_str, extra_formats=False)
        if xsi_type and dg_value is not None:
            el.set(f"{{{XSI_NS}}}type", xsi_type)
            el.set(f"{{{DG_NS}}}value", dg_value)
    origin = _origin_from_locations(value.get("locations"))
    if origin:
        el.set(f"{{{DG_NS}}}origin", origin)


def _add_tag(
    parent: etree._Element,
    docset_ns: str,
    tag: Tag,
    value: Any,
    *,
    path: LeafPath,
    ids: dict[LeafPath, str],
) -> None:
    if value is None:
        return  # field not extracted — omit the element entirely
    if tag.kind == "field":
        if isinstance(value, dict):
            _add_field(parent, docset_ns, tag.name, value, tag.value_type, path=path, ids=ids)
        return
    if tag.kind == "container":
        if not isinstance(value, dict):
            return
        el = etree.SubElement(parent, f"{{{docset_ns}}}{tag.name}")
        _maybe_set_id(el, path, ids)
        _add_children(el, docset_ns, tag.children, value, path=path, ids=ids)
        return
    if tag.kind == "choice":
        # Either the typed scalar (a {text, locations} value) or the group of
        # children — decided by whether any child key is present.
        if not isinstance(value, dict):
            return
        child_names = {c.name for c in tag.children}
        if any(k in value for k in child_names):
            el = etree.SubElement(parent, f"{{{docset_ns}}}{tag.name}")
            _maybe_set_id(el, path, ids)
            _add_children(el, docset_ns, tag.children, value, path=path, ids=ids)
        else:
            _add_field(parent, docset_ns, tag.name, value, tag.value_type, path=path, ids=ids)
        return
    if tag.kind == "collection":
        if not isinstance(value, list):
            return
        el = etree.SubElement(parent, f"{{{docset_ns}}}{tag.name}")
        _maybe_set_id(el, path, ids)
        item = tag.item
        item_name = (item.name if item else None) or tag.item_name or tag.name
        item_is_leaf = item is not None and item.kind == "field"
        item_type = item.value_type if item is not None else None
        for i, entry in enumerate(value):
            if not isinstance(entry, dict):
                continue
            item_path = path + (i,)
            if item_is_leaf:
                # list of grounded text values — each entry is a leaf field.
                _add_field(el, docset_ns, item_name, entry, item_type, path=item_path, ids=ids)
            else:
                item_el = etree.SubElement(el, f"{{{docset_ns}}}{item_name}")
                _maybe_set_id(item_el, item_path, ids)
                _add_children(item_el, docset_ns, tag.children, entry, path=item_path, ids=ids)
        return


def _add_children(
    parent: etree._Element,
    docset_ns: str,
    children: list[Tag],
    value: dict[str, Any],
    *,
    path: LeafPath = (),
    ids: dict[LeafPath, str] | None = None,
) -> None:
    ids = ids if ids is not None else {}
    for child in children:
        if child.name in value:
            _add_tag(
                parent, docset_ns, child, value[child.name], path=path + (child.name,), ids=ids
            )


def _serialize(root: etree._Element) -> str:
    xml = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="utf-8")
    return xml.decode("utf-8")  # type: ignore[no-any-return]


def _build_extraction(
    parent: etree._Element, docset_ns: str, vocab: Vocabulary, values: dict[str, Any]
) -> etree._Element:
    """Append a ``<dg:extraction>`` element (with the field tree) under *parent*.

    Pass 1 (:func:`_collect_ref_ids`) assigns an ``xml:id`` to every path some
    computed leaf derives from; pass 2 emits the tree, stamping those ids on
    the source elements and ``dg:href`` on the computed ones."""
    ext = etree.SubElement(parent, EXTRACTION_TAG)
    _add_children(ext, docset_ns, vocab.roots, values, ids=_collect_ref_ids(values))
    return ext


def standalone_extraction_doc(values: dict[str, Any], *, vocab: Vocabulary) -> str:
    """A minimal core DGML file (spec mode ``extraction``): a root ``dg:chunk``
    holding only the ``dg:extraction`` element. Used when no document tree
    (``docset generate``) exists yet for the file."""
    docset_ns = vocab.namespace_uri
    nsmap = {**_NSMAP_BASE, "docset": docset_ns}
    root = etree.Element(CHUNK_TAG, nsmap=nsmap)
    _build_extraction(root, docset_ns, vocab, values)
    return _serialize(root)


def embed_extraction_into(core_xml: str, values: dict[str, Any], *, vocab: Vocabulary) -> str:
    """Add (or replace) the ``dg:extraction`` element in an existing core DGML
    file (spec mode ``full-extraction``). The document tree is left untouched;
    any prior ``dg:extraction`` child of the root is replaced."""
    root = etree.fromstring(core_xml.encode("utf-8"))
    if root.tag != CHUNK_TAG:
        raise ValueError("core DGML file root is not dg:chunk")
    for child in list(root):
        if child.tag == EXTRACTION_TAG:
            root.remove(child)
    _build_extraction(root, vocab.namespace_uri, vocab, values)
    return _serialize(root)


def carry_extraction_over(prior_xml: str, new_xml: str) -> str:
    """Copy the ``dg:extraction`` element from *prior_xml* into *new_xml*.

    Used by ``docset generate`` when it (re)writes a ``<stem>.dgml.xml`` that
    already carried extracted values — a fresh document-tree render must not
    lose them (spec mode ``full-extraction``). The element is moved verbatim
    (values, origins, hrefs, xml:ids all preserved), replacing any
    ``dg:extraction`` already present in *new_xml*. Returns *new_xml*
    unchanged when *prior_xml* has none."""
    prior_root = etree.fromstring(prior_xml.encode("utf-8"))
    ext = next((c for c in prior_root if c.tag == EXTRACTION_TAG), None)
    if ext is None:
        return new_xml
    root = etree.fromstring(new_xml.encode("utf-8"))
    if root.tag != CHUNK_TAG:
        raise ValueError("core DGML file root is not dg:chunk")
    for child in list(root):
        if child.tag == EXTRACTION_TAG:
            root.remove(child)
    root.append(ext)  # moving it re-homes namespace declarations as needed
    return _serialize(root)


# ── DGML XML → values tree ───────────────────────────────────────────────────


def _locations_from_origin(origin: str | None) -> list[dict[str, Any]]:
    if not origin:
        return []
    out: list[dict[str, Any]] = []
    for box in origin.split(";"):
        parts = box.split()
        if len(parts) != 5:
            continue
        try:
            page, x1, y1, x2, y2 = (int(p) for p in parts)
        except ValueError:
            continue
        out.append({"page_number": page, "bounding_box": [x1, y1, x2, y2]})
    return out


def _local_name(el: etree._Element) -> str:
    tag = el.tag
    if isinstance(tag, str) and "}" in tag:
        return tag.split("}", 1)[1]
    return str(tag)


class _ParseAcc:
    """Per-parse collector for computed-field href resolution.

    ``id_paths`` maps every ``xml:id`` seen to its values-tree path;
    ``computed`` holds the computed leaves whose ``derived_from`` still
    carries raw ``#id`` hrefs. :meth:`resolve` rewrites those back to dotted
    paths once the whole tree has been walked."""

    def __init__(self) -> None:
        self.id_paths: dict[str, LeafPath] = {}
        self.computed: list[dict[str, Any]] = []

    def record(self, el: etree._Element, path: LeafPath) -> None:
        xml_id = el.get(f"{{{XML_NS}}}id")
        if xml_id:
            self.id_paths[xml_id] = path

    def resolve(self) -> None:
        for leaf in self.computed:
            leaf["derived_from"] = [
                path_to_str(self.id_paths[raw[1:]])
                if raw.startswith("#") and raw[1:] in self.id_paths
                else raw
                for raw in leaf["derived_from"]
            ]


def _leaf_value(
    el: etree._Element, path: LeafPath = (), acc: _ParseAcc | None = None
) -> dict[str, Any]:
    if acc is not None:
        acc.record(el, path)
    leaf: dict[str, Any] = {"text": (el.text or "").strip()}
    dg_value = el.get(f"{{{DG_NS}}}value")
    if dg_value is not None:
        leaf["value"] = dg_value
    origin = el.get(f"{{{DG_NS}}}origin")
    if origin == _ORIGIN_COMPUTED:
        leaf["computed"] = True
        href = el.get(f"{{{DG_NS}}}href")
        leaf["derived_from"] = [r.strip() for r in href.split(";") if r.strip()] if href else []
        if acc is not None and leaf["derived_from"]:
            acc.computed.append(leaf)
        return leaf
    leaf["locations"] = _locations_from_origin(origin)
    return leaf


def _element_to_value_vocab(
    el: etree._Element, tag: Tag, path: LeafPath = (), acc: _ParseAcc | None = None
) -> Any:
    """Project *el* using the schema tag *tag* to classify it precisely.

    Unlike the inference path, this distinguishes a one-item collection from a
    container, since the kind comes from the schema rather than the child count.
    """
    if tag.kind == "field":
        return _leaf_value(el, path, acc)
    if tag.kind == "choice":
        # Structured if any child element is present; else a scalar leaf.
        child_names = {c.name for c in tag.children}
        if any(isinstance(c.tag, str) and _local_name(c) in child_names for c in el):
            return _container_to_value(el, tag.children, path, acc)
        return _leaf_value(el, path, acc)
    if tag.kind == "collection":
        if acc is not None:
            acc.record(el, path)
        item_name = (tag.item.name if tag.item else None) or tag.item_name or tag.name
        items = [c for c in el if isinstance(c.tag, str) and _local_name(c) == item_name]
        if tag.item is not None and tag.item.kind == "field":
            return [_leaf_value(c, path + (i,), acc) for i, c in enumerate(items)]
        return [
            _container_to_value(item, tag.children, path + (i,), acc)
            for i, item in enumerate(items)
        ]
    return _container_to_value(el, tag.children, path, acc)


def _container_to_value(
    el: etree._Element, children: list[Tag], path: LeafPath = (), acc: _ParseAcc | None = None
) -> dict[str, Any]:
    if acc is not None:
        acc.record(el, path)
    by_name = {_local_name(c): c for c in el if isinstance(c.tag, str)}
    out: dict[str, Any] = {}
    for child in children:
        match = by_name.get(child.name)
        if match is not None:
            out[child.name] = _element_to_value_vocab(match, child, path + (child.name,), acc)
    return out


def _element_to_value(el: etree._Element, path: LeafPath = (), acc: _ParseAcc | None = None) -> Any:
    children = [c for c in el if isinstance(c.tag, str)]
    if not children:
        return _leaf_value(el, path, acc)
    if acc is not None:
        acc.record(el, path)
    child_names = [_local_name(c) for c in children]
    # A collection emits repeated identical item tags; a container holds
    # distinct child tags. (When the same name repeats, it is a collection.)
    if len(child_names) > 1 and len(set(child_names)) == 1:
        return [_element_to_value(c, path + (i,), acc) for i, c in enumerate(children)]
    out: dict[str, Any] = {}
    for child in children:
        name = _local_name(child)
        out[name] = _element_to_value(child, path + (name,), acc)
    return out


def has_extraction(xml: str) -> bool:
    """True if the core DGML file carries a ``dg:extraction`` element — i.e. the
    file has extracted values, not merely a generated document tree."""
    root = etree.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    return any(c.tag == EXTRACTION_TAG for c in root)


def has_document_tree(xml: str) -> bool:
    """True if the core DGML file has generated document-tree content — any root
    child other than ``dg:extraction``. Distinguishes a real ``full-extraction``
    target from an extraction-only file left by a prior ``extract`` run."""
    root = etree.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    return any(isinstance(c.tag, str) and c.tag != EXTRACTION_TAG for c in root)


def dgml_xml_to_values(xml: str, *, vocab: Vocabulary | None = None) -> dict[str, Any]:
    """Project a core DGML file's ``dg:extraction`` element to values-shape JSON.

    ``{tag: {text, value?, locations}}`` for grounded leaves, ``{text, value?,
    computed, derived_from}`` for computed ones (hrefs mapped back to dotted
    paths where they resolve in-file; cross-file or unknown targets stay raw),
    nested dicts for containers, lists for collections. When *vocab* is given,
    the schema classifies each element (so a single-item collection stays a
    list); otherwise the structure is inferred from the XML alone. Looks for
    the ``dg:extraction`` element under the root; if absent, falls back to the
    root's own children.
    """
    root = etree.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    container = next((c for c in root if c.tag == EXTRACTION_TAG), root)
    out: dict[str, Any] = {}
    acc = _ParseAcc()
    if vocab is not None:
        by_name = {t.name: t for t in vocab.roots}
        for child in container:
            if not isinstance(child.tag, str):
                continue
            name = _local_name(child)
            tag = by_name.get(name)
            out[name] = (
                _element_to_value_vocab(child, tag, (name,), acc)
                if tag
                else _element_to_value(child, (name,), acc)
            )
        acc.resolve()
        return out
    for child in container:
        if not isinstance(child.tag, str):
            continue
        name = _local_name(child)
        out[name] = _element_to_value(child, (name,), acc)
    acc.resolve()
    return out
