"""Convert DGML docset schema.rnc files to a JSON-LD context + graph.

Public API:
    rnc_to_jsonld(source)        -> dict[str, Any]
    rnc_to_jsonld_string(source) -> str
    main()                       -> CLI entry point
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import rnc2rng

__all__ = ["main", "rnc_to_jsonld", "rnc_to_jsonld_string"]


def _find(node: Any, *types: str) -> list[Any]:
    """Return immediate children of node matching any of the given types."""
    return [c for c in (node.value if isinstance(node.value, list) else []) if c.type in types]


def _deep(node: Any, *types: str) -> list[Any]:
    """Return all descendants matching any of the given types."""
    results: list[Any] = []
    queue = list(node.value) if isinstance(node.value, list) else []
    while queue:
        n = queue.pop(0)
        if not hasattr(n, "type"):
            continue
        if n.type in types:
            results.append(n)
        if isinstance(n.value, list):
            queue.extend(n.value)
    return results


def _parse_docs(doc_nodes: list[Any]) -> tuple[str | None, str | None]:
    """Extract description and example from DOCUMENTATION nodes."""
    description: str | None = None
    example: str | None = None
    for doc in doc_nodes:
        for line in doc.value or []:
            line = line.strip()
            if line.startswith("Example:"):
                example = line[len("Example:") :].strip()
            elif line:
                description = line
    return description, example


def _siblings_share(define_node: Any) -> bool | None:
    """Return True/False from siblingsShare attribute, or None if absent."""
    for attr in _deep(define_node, "ATTR"):
        names = _find(attr, "NAME")
        literals = _find(attr, "LITERAL")
        if names and names[0].name == "siblingsShare" and literals:
            return literals[0].name == "true"
    return None


def _elem_name(define_node: Any) -> str | None:
    """Return the element tag name (e.g. 'docset:LiabilityCap') or None."""
    for elem in _deep(define_node, "ELEM"):
        names = _find(elem, "NAME")
        if names:
            return str(names[0].name)
    return None


def _group_refs(define_node: Any) -> list[str]:
    """Return the REF names from a group definition (CHOICE of REFs, or single REF)."""
    refs: list[str] = []
    for assign in _find(define_node, "ASSIGN"):
        choices = _find(assign, "CHOICE")
        if choices:
            for choice in choices:
                for ref in _find(choice, "REF"):
                    refs.append(str(ref.name))
        else:
            for ref in _find(assign, "REF"):
                refs.append(str(ref.name))
    return refs


def rnc_to_jsonld(source: str | Path) -> dict[str, Any]:
    """Parse a DGML schema.rnc and return its JSON-LD representation."""
    text = Path(source).read_text(encoding="utf-8")
    root = rnc2rng.loads(text)

    # Collect namespace prefix and URI from the RNC namespace declaration
    prefix: str = "docset"
    ns_uri: str = ""
    for ns in _find(root, "NS"):
        if isinstance(ns.value, list) and ns.value:
            prefix = str(ns.name)
            ns_uri = str(ns.value[0])
            break

    p = prefix
    context: dict[str, Any] = {
        p: ns_uri,
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "Tag": f"{p}:Tag",
        "TagGroup": f"{p}:TagGroup",
        "siblingsShare": {"@id": f"{p}:siblingsShare", "@type": "xsd:boolean"},
        "members": {"@id": f"{p}:members", "@type": "@id", "@container": "@set"},
        "description": f"{p}:description",
        "example": f"{p}:example",
    }

    graph: list[dict[str, Any]] = []

    for define in _find(root, "DEFINE"):
        name = str(define.name)
        doc_nodes = _find(define, "DOCUMENTATION")
        elem_tag = _elem_name(define)

        if elem_tag is not None:
            # Element definition → Tag node
            share = _siblings_share(define)
            tag_type = (
                f"{p}:SharedTag"
                if share is True
                else f"{p}:NonSharedTag"
                if share is False
                else f"{p}:Tag"
            )
            local = elem_tag.split(":")[-1] if ":" in elem_tag else elem_tag
            node: dict[str, Any] = {
                "@id": f"{p}:{local}",
                "@type": tag_type,
            }
            description, example = _parse_docs(doc_nodes)
            if description:
                node["description"] = description
            if example:
                node["example"] = example
            graph.append(node)
        else:
            # Group definition → TagGroup node
            refs = _group_refs(define)
            if refs:
                graph.append(
                    {
                        "@id": f"{p}:{name}",
                        "@type": f"{p}:TagGroup",
                        "members": [f"{p}:{r}" for r in refs],
                    }
                )

    return {"@context": context, "@graph": graph}


def rnc_to_jsonld_string(source: str | Path, *, indent: int = 2) -> str:
    """Parse a DGML schema.rnc and return its JSON-LD as a string."""
    return json.dumps(rnc_to_jsonld(source), ensure_ascii=False, indent=indent)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: rnc2jsonld <schema.rnc>", file=sys.stderr)
        sys.exit(1)
    print(rnc_to_jsonld_string(sys.argv[1]))
