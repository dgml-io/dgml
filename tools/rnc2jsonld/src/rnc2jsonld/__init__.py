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


def _collect_namespaces(root: Any) -> tuple[dict[str, str], str]:
    """Collect all declared namespace prefix→URI pairs from the schema.

    Returns (prefix_to_uri, default_prefix). `default_prefix` is the RNC
    `default namespace` declaration (the docset's own vocabulary — used for
    meta-schema terms like Tag/TagGroup and for unprefixed element names),
    falling back to the first plain `namespace` declaration if the schema
    declares no default.
    """
    ns_map: dict[str, str] = {}
    default_prefix: str | None = None
    for node in _find(root, "NS", "DEFAULT_NS"):
        if isinstance(node.value, list) and node.value:
            ns_map[str(node.name)] = str(node.value[0])
            if node.type == "DEFAULT_NS" and default_prefix is None:
                default_prefix = str(node.name)
    if default_prefix is None and ns_map:
        default_prefix = next(iter(ns_map))
    return ns_map, default_prefix or "docset"


def rnc_to_jsonld(source: str | Path) -> dict[str, Any]:
    """Parse a DGML schema.rnc and return its JSON-LD representation."""
    text = Path(source).read_text(encoding="utf-8")
    root = rnc2rng.loads(text)

    ns_map, p = _collect_namespaces(root)

    context: dict[str, Any] = dict(ns_map)
    context.update(
        {
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "Tag": f"{p}:Tag",
            "TagGroup": f"{p}:TagGroup",
            "members": {"@id": f"{p}:members", "@type": "@id", "@container": "@set"},
            "description": f"{p}:description",
            "example": f"{p}:example",
        }
    )

    defines = _find(root, "DEFINE")

    # A group's REF names point at other DEFINE pattern names (e.g. "dg.chunk"),
    # which may differ from the element tag those patterns actually resolve to
    # (e.g. "dg:chunk"). Resolve every DEFINE name to its final @id up front so
    # group members reference the same @id used for the Tag/TagGroup node.
    name_to_id: dict[str, str] = {}
    elem_tags: dict[str, str | None] = {}
    for define in defines:
        name = str(define.name)
        elem_tag = _elem_name(define)
        elem_tags[name] = elem_tag
        # elem_tag already carries its own namespace prefix (e.g. "dg:chunk"
        # vs "docset:RentRoll") except when written unprefixed in the RNC,
        # which means it belongs to the default namespace.
        name_to_id[name] = (
            (elem_tag if ":" in elem_tag else f"{p}:{elem_tag}")
            if elem_tag is not None
            else f"{p}:{name}"
        )

    graph: list[dict[str, Any]] = []

    for define in defines:
        name = str(define.name)
        doc_nodes = _find(define, "DOCUMENTATION")
        elem_tag = elem_tags[name]

        if elem_tag is not None:
            # Element definition → Tag node
            node: dict[str, Any] = {
                "@id": name_to_id[name],
                "@type": f"{p}:Tag",
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
                        "@id": name_to_id[name],
                        "@type": f"{p}:TagGroup",
                        "members": [name_to_id.get(r, f"{p}:{r}") for r in refs],
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
