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

"""Convert between the at-rest extraction schema (RELAX NG Compact, ``schema.rnc``)
and the internal ``grounded_field`` JSON Schema the extraction engine drives off.

The DGML spec (§12) makes RELAX NG Compact the canonical docset schema. The
grounded-extraction engine in :mod:`dgml_core.grounded`, however, was built
around a JSON Schema whose every leaf is a reusable ``grounded_field``. Rather
than rewrite the engine, this module is the deterministic bridge:

    RNC (at rest)  ⇄  Vocabulary (intermediate)  ⇄  grounded_field JSON Schema

The RNC handled here is the constrained subset the spec defines — a namespace
declaration, a ``start`` rule rooted at ``dg:chunk``, and one named pattern per
tag of the form ``Name = element docset:Name { content }`` with
``##`` doc comments. It is **not** a general RELAX NG implementation; anything
outside the subset raises :class:`SchemaInvalid`. Full RELAX NG (RNG/Jing)
validation of instance documents is intentionally out of scope, so this module
pulls in no third-party RELAX NG dependency (keeping the Apache-2.0 license clean).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .errors import SchemaInvalid
from .generation.semantic_transform import docset_slug, org_ns_segment

# The canonical grounded_field definition. Every leaf value in the JSON Schema
# the engine consumes is a ``$ref`` to this; it mirrors the shape the schema-gen
# prompt in grounded.py documents, so the engine's _drop_bboxes_from_schema /
# _expand_refs helpers keep working unchanged.
GROUNDED_FIELD: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "locations": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "page_number": {"type": "integer", "minimum": 1},
                    "bounding_box": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "[left, top, right, bottom] in image pixels, top-left.",
                    },
                },
                "required": ["page_number", "bounding_box"],
            },
        },
    },
    "required": ["text", "locations"],
}

_GROUNDED_FIELD_REF = "#/definitions/grounded_field"

# The computed alternative (spec §7/§13): a value the model derives by
# reasoning over other extracted values (an InvoiceTotal summed from line
# items) instead of reading it off the page. No ``locations`` — grounding is
# expressed as ``derived_from`` paths into the same values tree, which the XML
# serializer turns into dg:origin="computed" + dg:itemprop/dg:href.
COMPUTED_FIELD: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "Display form of the result (e.g. '$349.85')."},
        "value": {
            "type": "string",
            "description": "Canonical machine-readable result (e.g. '349.85').",
        },
        "computed": {
            "type": "boolean",
            "description": "Always true — marks the value as derived, not read off the page.",
        },
        "derived_from": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Dotted paths of the values in this same submission the result "
                "derives from (e.g. 'LineItems[0].Quantity')."
            ),
        },
    },
    "required": ["text", "computed", "derived_from"],
}

_COMPUTED_FIELD_REF = "#/definitions/computed_field"

# Every leaf slot accepts either form; anyOf branch order is grounded-first so
# the common case reads first in prompts and provider tool schemas.
_LEAF_UNION: dict[str, Any] = {
    "anyOf": [{"$ref": _GROUNDED_FIELD_REF}, {"$ref": _COMPUTED_FIELD_REF}]
}


# ── Intermediate representation ──────────────────────────────────────────────


@dataclass
class Tag:
    """One node in a docset extraction vocabulary.

    ``kind`` is one of:
      * ``"field"``      — a grounded leaf value (RNC content model ``text`` or
        an ``xsd:`` datatype; the datatype, if any, is in ``value_type``).
      * ``"container"``  — an object grouping children (``(text | refs)*``).
      * ``"collection"`` — a repeatable list; ``item_name`` is the singular
        item tag and ``children`` are the item's fields (RNC: a plural element
        whose content is ``Item*`` plus a singular ``Item`` element def).
      * ``"choice"``     — an element that is EITHER a typed scalar
        (``value_type``) OR a group of child elements (``children``). RNC:
        ``( xsd:integer | ( Min, Max ) )``.
    """

    name: str
    kind: str
    description: str | None = None
    example: str | None = None
    prompt: str | None = None  # `## Prompt:` — where to find / how to derive the value (§13)
    value_type: str | None = None  # XSD datatype for a typed leaf / choice scalar (e.g. "integer")
    children: list[Tag] = field(default_factory=list)
    item_name: str | None = None
    # collection only: the singular item as its own container Tag, so the item
    # element's name + annotations survive an RNC round-trip (the grounded JSON
    # array shape has no slot for them). `children`/`item_name` mirror it.
    item: Tag | None = None


@dataclass
class Vocabulary:
    """A parsed extraction schema: the docset namespace plus its root tags."""

    namespace_uri: str
    roots: list[Tag]


# ── JSON Schema → Vocabulary ─────────────────────────────────────────────────


def _grounded_leaf(node: dict[str, Any]) -> bool:
    """True for a leaf slot: a bare grounded_field $ref (the form LLM-generated
    schemas use) or the grounded/computed anyOf union this module emits."""
    if node.get("$ref") == _GROUNDED_FIELD_REF:
        return True
    branches = node.get("anyOf")
    if not isinstance(branches, list) or not branches:
        return False
    refs = {b.get("$ref") for b in branches if isinstance(b, dict)}
    return len(refs) == len(branches) and refs <= {_GROUNDED_FIELD_REF, _COMPUTED_FIELD_REF}


_WORD_SPLIT_RE = re.compile(r"[^0-9A-Za-z]+")


def _pascal_case(raw: str) -> str:
    """PascalCase a field name without the suffix-stripping that the generation
    pipeline's ``sanitize_concept`` applies — ``line_items`` → ``LineItems``,
    not ``Line``. Extraction tag names are taken verbatim from the schema."""
    parts = [w for w in _WORD_SPLIT_RE.split(raw.strip()) if w]
    pascal = "".join(w[:1].upper() + w[1:] for w in parts)
    if pascal and not (pascal[0].isalpha() or pascal[0] == "_"):
        pascal = f"_{pascal}"
    return pascal


def _node_to_tag(name: str, node: dict[str, Any]) -> Tag:
    tag_name = _pascal_case(name) or "Field"
    description = node.get("description")
    example = node.get("example")
    prompt = node.get("prompt")

    if _grounded_leaf(node):
        return Tag(
            name=tag_name,
            kind="field",
            description=description,
            example=example,
            prompt=prompt,
        )

    node_type = node.get("type")
    if node_type == "array":
        items = node.get("items")
        if not isinstance(items, dict):
            raise SchemaInvalid(f"array field '{name}' must define an object 'items'")
        item_name = _singularize(tag_name)
        if _grounded_leaf(items):
            # A list of grounded text values (spec's uniform short-item list) —
            # the item is a leaf field, not a container of sub-fields.
            item_tag = Tag(name=item_name, kind="field")
            children: list[Tag] = []
        else:
            children = _properties_to_tags(items.get("properties"))
            item_tag = Tag(name=item_name, kind="container", children=children)
        return Tag(
            name=tag_name,
            kind="collection",
            description=description,
            example=example,
            prompt=prompt,
            children=children,
            item_name=item_name,
            item=item_tag,
        )

    if node_type == "object" or "properties" in node:
        return Tag(
            name=tag_name,
            kind="container",
            description=description,
            example=example,
            prompt=prompt,
            children=_properties_to_tags(node.get("properties")),
        )

    raise SchemaInvalid(
        f"field '{name}' is neither a grounded_field $ref, object, nor array — "
        'every leaf must be {"$ref": "#/definitions/grounded_field"}'
    )


def _properties_to_tags(properties: Any) -> list[Tag]:
    if properties is None:
        return []
    if not isinstance(properties, dict):
        raise SchemaInvalid("'properties' must be a JSON object")
    tags: list[Tag] = []
    seen: set[str] = set()
    for key, node in properties.items():
        if not isinstance(node, dict):
            raise SchemaInvalid(f"property '{key}' must be a JSON object")
        tag = _node_to_tag(str(key), node)
        if tag.name in seen:
            raise SchemaInvalid(f"duplicate tag name '{tag.name}' after normalization")
        seen.add(tag.name)
        tags.append(tag)
    return tags


def json_schema_to_vocabulary(schema: dict[str, Any], *, namespace_uri: str) -> Vocabulary:
    """Build a :class:`Vocabulary` from a grounded_field JSON Schema."""
    if not isinstance(schema, dict):
        raise SchemaInvalid("schema must be a JSON object")
    roots = _properties_to_tags(schema.get("properties"))
    if not roots:
        raise SchemaInvalid("schema has no 'properties' — nothing to extract")
    return Vocabulary(namespace_uri=namespace_uri, roots=roots)


# ── Vocabulary → JSON Schema ─────────────────────────────────────────────────


def _tag_to_node(tag: Tag) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if tag.description:
        extra["description"] = tag.description
    if tag.example:
        extra["example"] = tag.example
    if tag.prompt:
        # Carried into the JSON Schema so the extraction LLM (which is shown the
        # schema) sees the per-field guidance from `## Prompt:`.
        extra["prompt"] = tag.prompt

    if tag.kind == "field":
        # Typed leaves stay leaf-union refs; the type is applied at
        # serialization (xsi:type/dg:value), not carried in the tool schema.
        return {**_LEAF_UNION, **extra}
    if tag.kind == "collection":
        if tag.item is not None and tag.item.kind == "field":
            items_node: dict[str, Any] = dict(_LEAF_UNION)
        else:
            items_node = {"type": "object", "properties": _tags_to_properties(tag.children)}
        return {"type": "array", "items": items_node, **extra}
    if tag.kind == "container":
        return {"type": "object", "properties": _tags_to_properties(tag.children), **extra}
    if tag.kind == "choice":
        # Either the typed scalar (a grounded or computed value) or the group
        # of children.
        return {
            "anyOf": [
                *_LEAF_UNION["anyOf"],
                {"type": "object", "properties": _tags_to_properties(tag.children)},
            ],
            **extra,
        }
    raise SchemaInvalid(f"unknown tag kind '{tag.kind}'")


def _tags_to_properties(tags: list[Tag]) -> dict[str, Any]:
    return {tag.name: _tag_to_node(tag) for tag in tags}


def vocabulary_to_json_schema(vocab: Vocabulary) -> dict[str, Any]:
    """Render a :class:`Vocabulary` back to a grounded_field JSON Schema.

    Carries ``description``/``example``/``prompt`` onto each property so the
    RNC ⇄ JSON ⇄ RNC round-trip is lossless. The extraction engine ignores
    those extra keys (it only walks for grounded_field $refs and properties).
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "definitions": {"grounded_field": GROUNDED_FIELD, "computed_field": COMPUTED_FIELD},
        "properties": _tags_to_properties(vocab.roots),
    }


# ── Vocabulary → RNC ─────────────────────────────────────────────────────────


def _doc_comment(tag: Tag) -> str:
    lines: list[str] = []
    if tag.description:
        for line in tag.description.splitlines():
            lines.append(f"## {line}".rstrip())
    if tag.example:
        lines.append(f"## Example: {tag.example}")
    if tag.prompt:
        lines.append(f"## Prompt: {tag.prompt}")
    return "".join(f"{line}\n" for line in lines)


def _emit_tag_defs(tag: Tag, out: list[str], seen: set[str]) -> None:
    """Append the RNC pattern def(s) for *tag* (and its descendants) to *out*.

    *seen* dedups shared pattern names across the recursion — a tag emits its
    def at most once.
    """
    if tag.name in seen:
        return
    seen.add(tag.name)

    if tag.kind == "field":
        content = f"xsd:{tag.value_type}" if tag.value_type else "text"
        out.append(
            f"{_doc_comment(tag)}{tag.name} =\n"
            f"  element docset:{tag.name} {{\n"
            f"    {content}\n  }}\n"
        )
        return

    if tag.kind == "choice":
        scalar = f"xsd:{tag.value_type}" if tag.value_type else "text"
        group = ", ".join(c.name for c in tag.children)
        out.append(
            f"{_doc_comment(tag)}{tag.name} =\n"
            f"  element docset:{tag.name} {{\n"
            f"    ( {scalar} | ( {group} ) )\n  }}\n"
        )
        for child in tag.children:
            _emit_tag_defs(child, out, seen)
        return

    if tag.kind == "collection":
        item_tag = tag.item or Tag(
            name=tag.item_name or _singularize(tag.name),
            kind="container",
            children=tag.children,
        )
        out.append(
            f"{_doc_comment(tag)}{tag.name} =\n"
            f"  element docset:{tag.name} {{\n"
            f"    {item_tag.name}*\n  }}\n"
        )
        # Recurse on the singular item (a container) so it and its children are
        # emitted exactly once, carrying the item's own annotations.
        _emit_tag_defs(item_tag, out, seen)
        return

    if tag.kind == "container":
        refs = " | ".join(["text", *(c.name for c in tag.children)])
        out.append(
            f"{_doc_comment(tag)}{tag.name} =\n"
            f"  element docset:{tag.name} {{\n"
            f"    ({refs})*\n  }}\n"
        )
        for child in tag.children:
            _emit_tag_defs(child, out, seen)
        return

    raise SchemaInvalid(f"unknown tag kind '{tag.kind}'")


def vocabulary_to_rnc(vocab: Vocabulary) -> str:
    """Serialize a :class:`Vocabulary` to RELAX NG Compact (the at-rest form).

    Matches the spec's docset-schema form (§12/§13): a ``namespace docset``
    declaration followed by element definitions, roots first. There is no
    ``start``/``dg:chunk`` rule — that wrapping is an output-format concern
    (``dg:extraction``), not part of the docset vocabulary. Roots are the
    element defs not referenced as a child by any other element.
    """
    parts: list[str] = [f'namespace docset = "{vocab.namespace_uri}"\n']
    defs: list[str] = []
    seen: set[str] = set()
    for tag in vocab.roots:
        _emit_tag_defs(tag, defs, seen)
    for body in defs:
        parts.append("\n")
        parts.append(body)
    return "".join(parts)


# ── RNC → Vocabulary ─────────────────────────────────────────────────────────

_NAMESPACE_RE = re.compile(r'^\s*namespace\s+(\w+)\s*=\s*"([^"]*)"\s*$')
_DEF_HEAD_RE = re.compile(r"^(\w+)\s*=\s*$")
_ELEMENT_RE = re.compile(r"^\s*element\s+(\w+):(\w+)\s*\{\s*$")
_COLLECTION_RE = re.compile(r"^\s*(\w+)\*\s*,?\s*$")
_CONTAINER_RE = re.compile(r"^\s*\(\s*(.+?)\s*\)\*\s*,?\s*$")
_TEXT_RE = re.compile(r"^\s*(?:text|\(\s*text\s*\)\*)\s*,?\s*$")
_TYPED_RE = re.compile(r"^\s*xsd:(\w+)\s*,?\s*$")
# ( <scalar> | ( RefA, RefB, ... ) ) — a typed-scalar-or-group choice element.
_CHOICE_RE = re.compile(r"^\s*\(\s*(text|xsd:\w+)\s*\|\s*\(\s*([^()|]+)\)\s*\)\s*,?\s*$")
_START_BODY_RE = re.compile(r"^\s*element\s+dg:chunk\s*\{\s*$")


@dataclass
class _RawDef:
    name: str
    description: str | None
    example: str | None
    prompt: str | None
    body_lines: list[str]  # the element's content-model lines


def _parse_doc_comments(comments: list[str]) -> tuple[str | None, str | None, str | None]:
    desc_lines: list[str] = []
    example: str | None = None
    prompt: str | None = None
    for raw in comments:
        text = raw[2:].strip() if raw.startswith("##") else raw.strip()
        if text.startswith("Example:"):
            example = text[len("Example:") :].strip()
        elif text.startswith("Prompt:"):
            prompt = text[len("Prompt:") :].strip()
        else:
            desc_lines.append(text)
    description = "\n".join(desc_lines).strip() or None
    return description, example, prompt


def _refs_in_body(body_lines: list[str]) -> list[str]:
    """The element names a def's content model references (excluding ``text``)."""
    for line in body_lines:
        choice = _CHOICE_RE.match(line)
        if choice:
            return [r.strip() for r in choice.group(2).split(",") if r.strip()]
        coll = _COLLECTION_RE.match(line)
        if coll:
            return [coll.group(1)]
        cont = _CONTAINER_RE.match(line)
        if cont:
            return _split_refs(cont.group(1))
    return []


def _parse_rnc_defs(rnc: str) -> tuple[str, list[str], dict[str, _RawDef]]:
    """Tokenize RNC into (namespace_uri, root_names, {name: _RawDef}).

    A ``start = element dg:chunk {...}`` rule is accepted (and names the roots)
    but is optional: the spec's docset-schema form omits it, in which case the
    roots are the element defs not referenced as a child by any other element,
    in definition order.
    """
    lines = rnc.splitlines()
    namespace_uri = ""
    start_refs: list[str] = []
    defs: dict[str, _RawDef] = {}
    order: list[str] = []

    i = 0
    pending_comments: list[str] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("##"):
            pending_comments.append(stripped)
            i += 1
            continue

        ns = _NAMESPACE_RE.match(line)
        if ns:
            if ns.group(1) == "docset":
                namespace_uri = ns.group(2)
            pending_comments = []
            i += 1
            continue

        if stripped == "start =":
            block, i = _read_brace_block(lines, i + 1)
            start_refs = _parse_start_block(block)
            pending_comments = []
            continue

        head = _DEF_HEAD_RE.match(stripped)
        if head:
            name = head.group(1)
            block, i = _read_brace_block(lines, i + 1)
            body = _parse_element_block(name, block)
            description, example, prompt = _parse_doc_comments(pending_comments)
            defs[name] = _RawDef(
                name=name,
                description=description,
                example=example,
                prompt=prompt,
                body_lines=body,
            )
            order.append(name)
            pending_comments = []
            continue

        raise SchemaInvalid(f"unexpected RNC line: {stripped!r}")

    if not namespace_uri:
        raise SchemaInvalid('RNC is missing a `namespace docset = "..."` declaration')
    if not defs:
        raise SchemaInvalid("RNC defines no extraction elements")

    roots = start_refs
    if not roots:
        referenced: set[str] = set()
        for raw in defs.values():
            referenced.update(_refs_in_body(raw.body_lines))
        roots = [name for name in order if name not in referenced]
        if not roots:
            raise SchemaInvalid("RNC has no root element (every def is referenced by another)")
    return namespace_uri, roots, defs


def _read_brace_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Read the ``element ... { ... }`` block opening at/after *start*.

    Returns the inner lines (between the opening ``{`` and matching ``}``) and
    the index just past the closing brace.
    """
    depth = 0
    inner: list[str] = []
    i = start
    opened = False
    while i < len(lines):
        line = lines[i]
        opens = line.count("{")
        closes = line.count("}")
        if not opened and opens:
            opened = True
            depth += opens - closes
            # Keep the element-open line itself so callers can read the tag.
            inner.append(line)
            i += 1
            if depth == 0:
                return inner, i
            continue
        if opened:
            depth += opens - closes
            if depth <= 0:
                # Drop the bare closing-brace line.
                if line.strip() != "}":
                    inner.append(line)
                return inner, i + 1
            inner.append(line)
        i += 1
    raise SchemaInvalid("unterminated '{' in RNC definition")


def _parse_element_block(name: str, block: list[str]) -> list[str]:
    if not block:
        raise SchemaInvalid(f"definition '{name}' has an empty body")
    el = _ELEMENT_RE.match(block[0])
    if not el:
        raise SchemaInvalid(f"definition '{name}' must wrap `element docset:{name} {{ ... }}`")
    body: list[str] = []
    for line in block[1:]:
        if not line.strip():
            continue
        body.append(line)
    return body


def _parse_start_block(block: list[str]) -> list[str]:
    if not block or not _START_BODY_RE.match(block[0]):
        raise SchemaInvalid("`start` rule must be `element dg:chunk { (text | ...)* }`")
    for line in block[1:]:
        if not line.strip():
            continue
        cont = _CONTAINER_RE.match(line)
        if cont:
            return _split_refs(cont.group(1))
    raise SchemaInvalid("`start` rule has no `(text | ...)*` content model")


def _split_refs(inner: str) -> list[str]:
    return [r.strip() for r in inner.split("|") if r.strip() and r.strip() != "text"]


def _raw_to_tag(raw: _RawDef, defs: dict[str, _RawDef], stack: tuple[str, ...]) -> Tag:
    if raw.name in stack:
        raise SchemaInvalid(f"recursive RNC definition through '{raw.name}'")
    body = raw.body_lines
    single = body[0] if len(body) == 1 else ""

    if single and _TEXT_RE.match(single):
        return Tag(
            name=raw.name,
            kind="field",
            description=raw.description,
            example=raw.example,
            prompt=raw.prompt,
        )

    typed = _TYPED_RE.match(single) if single else None
    if typed:
        return Tag(
            name=raw.name,
            kind="field",
            description=raw.description,
            example=raw.example,
            prompt=raw.prompt,
            value_type=typed.group(1),
        )

    choice = _CHOICE_RE.match(single) if single else None
    if choice:
        scalar = choice.group(1)
        value_type = None if scalar == "text" else scalar.split(":", 1)[1]
        ref_names = [r.strip() for r in choice.group(2).split(",") if r.strip()]
        children = _resolve_refs(ref_names, defs, stack + (raw.name,))
        return Tag(
            name=raw.name,
            kind="choice",
            description=raw.description,
            example=raw.example,
            prompt=raw.prompt,
            value_type=value_type,
            children=children,
        )

    coll = _COLLECTION_RE.match(single) if single else None
    if coll:
        item_name = coll.group(1)
        item_def = defs.get(item_name)
        if item_def is None:
            raise SchemaInvalid(f"collection '{raw.name}' references unknown item '{item_name}'")
        item_tag = _raw_to_tag(item_def, defs, stack + (raw.name,))
        return Tag(
            name=raw.name,
            kind="collection",
            description=raw.description,
            example=raw.example,
            prompt=raw.prompt,
            children=item_tag.children,
            item_name=item_name,
            item=item_tag,
        )

    cont = _CONTAINER_RE.match(single) if single else None
    if cont:
        children = _resolve_refs(_split_refs(cont.group(1)), defs, stack + (raw.name,))
        return Tag(
            name=raw.name,
            kind="container",
            description=raw.description,
            example=raw.example,
            prompt=raw.prompt,
            children=children,
        )

    raise SchemaInvalid(f"definition '{raw.name}' has an unrecognized content model")


def _resolve_refs(names: list[str], defs: dict[str, _RawDef], stack: tuple[str, ...]) -> list[Tag]:
    tags: list[Tag] = []
    for name in names:
        raw = defs.get(name)
        if raw is None:
            raise SchemaInvalid(f"reference to undefined pattern '{name}'")
        tags.append(_raw_to_tag(raw, defs, stack))
    return tags


def parse_rnc(rnc: str) -> Vocabulary:
    """Parse the constrained RNC subset into a :class:`Vocabulary`.

    Raises :class:`SchemaInvalid` for anything outside the subset.
    """
    namespace_uri, root_names, defs = _parse_rnc_defs(rnc)
    roots = _resolve_refs(root_names, defs, ())
    return Vocabulary(namespace_uri=namespace_uri, roots=roots)


def validate_rnc(rnc: str) -> None:
    """Validate that *rnc* is well-formed within the supported subset."""
    parse_rnc(rnc)


# ── Top-level conveniences ───────────────────────────────────────────────────


def json_schema_to_rnc(schema: dict[str, Any], *, workspace: str, docset_name: str) -> str:
    """Convert a grounded_field JSON Schema to the at-rest RNC form.

    The docset namespace is built from *workspace* and *docset_name* the same
    way the generation pipeline does (``http://dgml.io/{workspace}/{slug}``),
    so an extraction docset and a generated docset share one namespace.
    """
    namespace_uri = f"http://dgml.io/{org_ns_segment(workspace)}/{docset_slug(docset_name)}"
    vocab = json_schema_to_vocabulary(schema, namespace_uri=namespace_uri)
    return vocabulary_to_rnc(vocab)


def rnc_to_json_schema(rnc: str) -> dict[str, Any]:
    """Convert at-rest RNC to the grounded_field JSON Schema the engine drives off."""
    return vocabulary_to_json_schema(parse_rnc(rnc))


# ── helpers ──────────────────────────────────────────────────────────────────


def _singularize(name: str) -> str:
    """Naive PascalCase singularization for collection item names.

    Good enough for tag naming (``LineItems`` → ``LineItem``, ``Annexures`` →
    ``Annexure``, ``Parties`` → ``Party``). Falls back to ``<Name>Item`` when a
    word does not end in a recognized plural so we never collide with the plural.
    """
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("ses") or name.endswith("xes") or name.endswith("zes"):
        return name[:-2]
    if name.endswith("s") and not name.endswith("ss"):
        return name[:-1]
    return f"{name}Item"
