"""DGML to JSON-LD converter (XAST-convention lossless export).

The DGML XML file is the system of record. This module produces the
XAST-convention JSON-LD export — a DOM-style tree representation following
the conventions of https://github.com/syntax-tree/xast, extended with
JSON-LD identity and linking.

Public API:
    xml_to_jsonld(source)        -> dict[str, Any]
    xml_to_jsonld_string(source) -> str
    main()                       -> CLI entry point
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from lxml import etree

__all__ = ["main", "xml_to_jsonld", "xml_to_jsonld_string"]

_XAST_NS = "http://dgml.io/ns/xast#"
_XML_NS = "http://www.w3.org/XML/1998/namespace"
_DG_NS = "http://dgml.io/ns/dg#"
_ITEMPROP = f"{{{_DG_NS}}}itemprop"
_HREF = f"{{{_DG_NS}}}href"
_INPUTS = f"{{{_DG_NS}}}inputs"
_ORIGIN = f"{{{_DG_NS}}}origin"
_XML_ID = f"{{{_XML_NS}}}id"

# DGML 0.7's reserved dg:origin value for a value the page doesn't print.
# Retired in 0.71: read as no origin, with the element's dg:href as its
# dg:inputs (spec Appendix C).
_LEGACY_COMPUTED = "computed"

# A reference list (dg:href, dg:inputs): "#a; #b" as writers join it, or the
# older space-separated "#a #b" — readers split on either (spec §8).
_REF_SPLIT = re.compile(r"[;\s]+")

# Fixed xast: terms always added to @context.
_XAST_CONTEXT: dict[str, Any] = {
    "xast": _XAST_NS,
    "children": {"@id": "xast:children", "@container": "@list"},
    "attributes": {"@id": "xast:attributes", "@container": "@index"},
    "nodeType": {"@id": "xast:nodeType"},
    "value": {"@id": "xast:value"},
}


def _build_context(nsmap: dict[str | None, str]) -> dict[str, Any]:
    ctx: dict[str, Any] = dict(_XAST_CONTEXT)
    for prefix, uri in nsmap.items():
        if prefix is None:
            ctx["@vocab"] = uri
        else:
            ctx[prefix] = uri
    return ctx


def _reverse_nsmap(nsmap: dict[str | None, str]) -> dict[str, str]:
    """uri → prefix (None prefix mapped to empty string for default ns)."""
    return {uri: (prefix or "") for prefix, uri in nsmap.items()}


def _clark_to_curie(clark: str, rev: dict[str, str], cache: dict[str, str]) -> str:
    # lxml represents qualified names in Clark notation: {uri}local
    # e.g. {http://dgml.io/ns/dg#}chunk
    # We convert to CURIE (Compact URI Expression) using declared prefixes:
    # e.g. dg:chunk
    if clark in cache:
        return cache[clark]
    if clark.startswith("{"):
        uri, local = clark[1:].split("}", 1)
        prefix = rev.get(uri, "")
        curie = f"{prefix}:{local}" if prefix else local
    else:
        curie = clark
    cache[clark] = curie
    return curie


def _split_refs(value: str) -> list[str]:
    """The references in a dg:href / dg:inputs list."""
    return [ref for ref in _REF_SPLIT.split(value) if ref]


def _normalize_id(value: str) -> str:
    """Prefix plain local ids with #; leave URNs and full URIs unchanged."""
    if value.startswith("#") or ":" in value or "/" in value:
        return value
    return f"#{value}"


def _convert_element(
    elem: etree._Element,
    rev: dict[str, str],
    cache: dict[str, str],
) -> dict[str, Any]:
    node: dict[str, Any] = {"nodeType": "xast:element"}

    # @type from element tag
    tag = str(elem.tag) if not isinstance(elem.tag, str) else elem.tag
    node["@type"] = _clark_to_curie(tag, rev, cache)

    # @id from id or xml:id attribute
    raw_id = elem.attrib.get("id") or elem.attrib.get(_XML_ID)
    if raw_id is not None:
        node["@id"] = _normalize_id(str(raw_id))

    # A 0.7 computed element reads as its 0.71 equivalent: no origin, and its
    # dg:href targets are its dg:inputs, not a semantic link.
    legacy = elem.attrib.get(_ORIGIN) == _LEGACY_COMPUTED
    inputs = elem.attrib.get(_INPUTS) or (elem.attrib.get(_HREF) if legacy else None)

    # dg:itemprop/dg:href on this element → named link property on this node.
    # dg:href may hold more than one id when the property aggregates several
    # targets; JSON-LD requires a single IRI per "@id", so multiple targets
    # become a list of link objects instead of one.
    itemprop = elem.attrib.get(_ITEMPROP)
    href = elem.attrib.get(_HREF)
    if itemprop is not None and href is not None and not legacy:
        ids = _split_refs(href)
        node[itemprop] = {"@id": ids[0]} if len(ids) == 1 else [{"@id": i} for i in ids]

    # dg:inputs → the elements the value used, always a list (spec A.2).
    if inputs:
        node[_clark_to_curie(_INPUTS, rev, cache)] = [{"@id": ref} for ref in _split_refs(inputs)]

    # attributes — all except id, xml:id, dg:itemprop, dg:href, dg:inputs
    # (and a 0.7 element's "computed" origin, which is not a location)
    skip = {"id", _XML_ID, _ITEMPROP, _HREF, _INPUTS}
    if legacy:
        skip.add(_ORIGIN)
    attrs: dict[str, str] = {}
    for k, v in elem.attrib.items():
        k = str(k)
        if k in skip:
            continue
        attrs[_clark_to_curie(k, rev, cache)] = v
    if attrs:
        node["attributes"] = attrs

    # children — ordered, always present
    children: list[dict[str, Any]] = []

    if elem.text and elem.text.strip():
        children.append({"nodeType": "xast:text", "value": elem.text.strip()})

    for child in elem:
        if not isinstance(child.tag, str):
            continue  # skip comments, PIs
        children.append(_convert_element(child, rev, cache))
        if child.tail and child.tail.strip():
            children.append({"nodeType": "xast:text", "value": child.tail.strip()})

    node["children"] = children
    return node


def xml_to_jsonld(source: str | Path) -> dict[str, Any]:
    """Parse a DGML XML file and return a JSON-LD dict (XAST export)."""
    tree = etree.parse(str(source))
    root = tree.getroot()

    context = _build_context(root.nsmap)
    rev = _reverse_nsmap(root.nsmap)
    cache: dict[str, str] = {}

    return {
        "@context": context,
        "@graph": [_convert_element(root, rev, cache)],
    }


def xml_to_jsonld_string(source: str | Path, *, indent: int = 2) -> str:
    """Parse a DGML XML file and return a JSON-LD string (XAST export)."""
    return json.dumps(xml_to_jsonld(source), ensure_ascii=False, indent=indent)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: dgml2jsonld <file.dgml.xml>", file=sys.stderr)
        sys.exit(1)
    print(xml_to_jsonld_string(sys.argv[1]))
