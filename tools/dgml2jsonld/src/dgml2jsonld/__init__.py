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
_XML_ID = f"{{{_XML_NS}}}id"

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

    # dg:itemprop/dg:href on this element → named link property on this node
    itemprop = elem.attrib.get(_ITEMPROP)
    href = elem.attrib.get(_HREF)
    if itemprop is not None and href is not None:
        node[itemprop] = {"@id": href}

    # attributes — all except id, xml:id, dg:itemprop, dg:href
    skip = {"id", _XML_ID, _ITEMPROP, _HREF}
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
