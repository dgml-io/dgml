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
and the internal ``extracted_value`` JSON Schema the extraction engine drives off.

The DGML spec (§12) makes RELAX NG Compact the canonical docset schema. The
grounded-extraction engine in :mod:`dgml_core.grounded`, however, was built
around a JSON Schema whose every leaf is a reusable ``extracted_value``
(verbatim text + optional normalized value + grounded-or-computed provenance).
Rather than rewrite the engine, this module is the deterministic bridge:

    RNC (at rest)  ⇄  Vocabulary (intermediate)  ⇄  extracted_value JSON Schema

The RNC handled here is the constrained subset the spec defines — a namespace
declaration, a ``start`` rule rooted at ``dg:chunk``, and one named pattern per
tag of the form ``Name = element docset:Name { content }`` with
``##`` doc comments; a field's content model is ``text``, an ``xsd:`` datatype,
or a value enumeration (``( "a" | "b" )``). It is **not** a general RELAX NG
implementation; anything outside the subset raises :class:`SchemaInvalid`.
Full RELAX NG (RNG/Jing) validation of instance documents is intentionally out
of scope, so this module pulls in no third-party RELAX NG dependency (keeping
the Apache-2.0 license clean).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .errors import SchemaInvalid
from .generation.semantic_transform import docset_slug, org_ns_segment

# The shared ``locations`` shape: one entry per page region a value grounds to.
_LOCATIONS_SCHEMA: dict[str, Any] = {
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
}

# The legacy grounded_field definition. Older exported schemas (and LLM-authored
# ones following the old convention) use a ``$ref`` to this; it is still
# accepted on input but no longer emitted — see EXTRACTED_VALUE below.
GROUNDED_FIELD: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "locations": _LOCATIONS_SCHEMA,
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

# The merged leaf definition (spec §13): every extracted value carries the
# verbatim ``text``, an optional normalized ``value``, and its provenance —
# either grounded to the page (``locations``) or computed from other values
# (``computed``/``derived_from``), never both. One shape for both cases keeps
# the provider tool schema small: a large schema whose every leaf is a
# two-branch anyOf union doubles the constrained-decoding state count, and
# Gemini rejects oversized tool schemas with "too many states for serving".
# The single shape is also the only clean carrier for a model-returned
# normalized ``value`` on grounded leaves (enum classification like
# "Electric Service" → "electric" cannot be derived by code-side regexes).
EXTRACTED_VALUE: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "Verbatim as printed on the document.",
        },
        "value": {
            "type": "string",
            "description": (
                "Normalized machine-readable form (an enum token, ISO date, or "
                "plain number). Omit when the text cannot be normalized."
            ),
        },
        "locations": _LOCATIONS_SCHEMA,
        "computed": {
            "type": "boolean",
            "description": "True marks the value as derived, not read off the page.",
        },
        "derived_from": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Computed values only: dotted paths of the values in this same "
                "submission the result derives from (e.g. 'LineItems[0].Quantity')."
            ),
        },
    },
    "required": ["text"],
}

_EXTRACTED_VALUE_REF = "#/definitions/extracted_value"


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
    # `## Invariant:` — a machine-checkable relation this field must satisfy
    # against the rest of the extracted values, e.g. `count(LineItems)` or
    # `sum(LineItems[].Amount)`. Report-only; see
    # dgml_core.extraction_xml.check_invariants.
    invariant: str | None = None
    value_type: str | None = None  # XSD datatype for a typed leaf / choice scalar (e.g. "integer")
    # field only: closed set of normalized-value tokens (RNC value enumeration,
    # e.g. `( "electric" | "water" )`). The verbatim page text stays free-form;
    # the enum constrains the model-returned normalized `value` → dg:value.
    # Mutually exclusive with `value_type` — an enum IS the leaf's type.
    enum_values: list[str] | None = None
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
    """True for a leaf slot: an extracted_value $ref (the form this module
    emits), a bare legacy grounded_field $ref, or the legacy grounded/computed
    anyOf union — older exports and LLM-authored schemas still use those."""
    if node.get("$ref") in {_EXTRACTED_VALUE_REF, _GROUNDED_FIELD_REF}:
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
    # A `title` names the element directly (standard JSON Schema dialects put
    # the target DGML element name there); the property key is the fallback.
    title = node.get("title")
    raw_name = title if isinstance(title, str) and title.strip() else name
    tag_name = _pascal_case(raw_name) or "Field"
    description = node.get("description")
    example = node.get("example")
    prompt = node.get("prompt")
    raw_invariant = node.get("invariant")
    invariant = _validate_invariant(raw_invariant) if isinstance(raw_invariant, str) else None

    if _grounded_leaf(node):
        value_type, enum_values = _leaf_value_types(node, tag_name=tag_name)
        return Tag(
            name=tag_name,
            kind="field",
            description=description,
            example=example,
            prompt=prompt,
            invariant=invariant,
            value_type=value_type,
            enum_values=enum_values,
        )

    node_type = node.get("type")
    if node_type == "array":
        items = node.get("items")
        if not isinstance(items, dict):
            raise SchemaInvalid(f"array field '{name}' must define an object 'items'")
        raw_item_name = node.get("item_name")
        if raw_item_name is not None and (
            not isinstance(raw_item_name, str) or not raw_item_name.strip()
        ):
            raise SchemaInvalid(f"array field '{name}' has a non-string or empty 'item_name'")
        item_name = (
            _pascal_case(raw_item_name) if raw_item_name else _singularize(tag_name)
        ) or _singularize(tag_name)
        if _grounded_leaf(items):
            # A list of grounded text values (spec's uniform short-item list) —
            # the item is a leaf field, not a container of sub-fields.
            item_type, item_enum = _leaf_value_types(items, tag_name=item_name)
            item_tag = Tag(
                name=item_name, kind="field", value_type=item_type, enum_values=item_enum
            )
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


# Property names an object schema may use and still be a *leaf* (one extracted
# value) rather than a container of fields. Matches EXTRACTED_VALUE's shape.
_LEAF_SHAPE_KEYS = frozenset({"text", "value", "locations", "derived_from", "computed"})

# Annotation keys a $ref node may carry as siblings; they are merged over the
# resolved target (2020-12 semantics: siblings apply alongside the $ref).
_REF_SIBLING_KEYS = frozenset(
    {
        "title",
        "description",
        "prompt",
        "example",
        "invariant",
        "datatype",
        "value_enum",
        "item_name",
    }
)

# Common plain-decimal `pattern` spellings; a leaf `value` constrained to one
# of these is a decimal. Matched literally — guessing datatypes from arbitrary
# regexes would misfire on id/code patterns.
_DECIMAL_VALUE_PATTERNS = frozenset({r"^-?\d+(\.\d+)?$", r"^-?\d+(?:\.\d+)?$"})


def _is_leaf_shape(node: dict[str, Any]) -> bool:
    """True for an object schema that IS an extracted value (a leaf), rather
    than a container of fields: its properties are a subset of the
    extracted-value keys with ``text`` present."""
    if node.get("type") != "object":
        return False
    props = node.get("properties")
    if not isinstance(props, dict) or "text" not in props:
        return False
    return set(props) <= _LEAF_SHAPE_KEYS


def _value_subschema_types(value_schema: Any) -> dict[str, Any]:
    """Map a leaf's ``value`` subschema to the internal ``datatype``/
    ``value_enum`` sidecars."""
    if not isinstance(value_schema, dict):
        return {}
    enum = value_schema.get("enum")
    if isinstance(enum, list) and enum:
        return {"value_enum": [str(v) for v in enum]}
    fmt = value_schema.get("format")
    by_format = {"date": "date", "date-time": "dateTime", "time": "time", "uri": "anyURI"}
    if isinstance(fmt, str) and fmt in by_format:
        return {"datatype": by_format[fmt]}
    vtype = value_schema.get("type")
    if vtype == "integer":
        return {"datatype": "integer"}
    if vtype == "boolean":
        return {"datatype": "boolean"}
    if vtype == "number":
        return {"datatype": "decimal"}
    pattern = value_schema.get("pattern")
    if isinstance(pattern, str) and pattern in _DECIMAL_VALUE_PATTERNS:
        return {"datatype": "decimal"}
    return {}


def _resolve_schema_dialect(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a standard-dialect JSON Schema into the internal form the
    property walker (:func:`_node_to_tag`) understands.

    Accepted on top of the internal dialect:

    * a **root ``$ref``** naming the document object;
    * local ``$defs``/``definitions`` ``$ref``\\s anywhere, resolved inline
      with sibling annotation keys merged over the target;
    * the **merged leaf shape** recognized structurally — an object whose
      properties are a subset of ``{text, value, locations, derived_from,
      computed}`` with ``text`` present becomes an ``extracted_value`` leaf,
      its ``value`` subschema mapped to ``datatype``/``value_enum``
      (``enum`` list, ``format: date``, ``type: integer/boolean/number``,
      or a plain-decimal ``pattern``);
    * ``title`` as the element name (handled downstream in ``_node_to_tag``).

    Schemas already in the internal form pass through unchanged (internal
    leaf ``$ref``\\s are conventional and left alone).
    """
    defs: dict[str, Any] = {}
    for block_key in ("definitions", "$defs"):
        block = schema.get(block_key)
        if isinstance(block, dict):
            defs.update(block)

    internal_leaf_names = {"extracted_value", "grounded_field", "computed_field"}

    def deref(ref: str) -> tuple[str, Any] | None:
        for prefix in ("#/definitions/", "#/$defs/"):
            if ref.startswith(prefix):
                name = ref[len(prefix) :]
                return name, defs.get(name)
        return None

    def walk(node: Any, stack: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [walk(entry, stack) for entry in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str):
            found = deref(ref)
            if found is not None:
                name, target = found
                if name in internal_leaf_names:
                    return node  # conventional internal leaf ref — leave alone
                if not isinstance(target, dict):
                    raise SchemaInvalid(f"$ref to undefined schema definition '{name}'")
                if name in stack:
                    raise SchemaInvalid(f"recursive $ref through definition '{name}'")
                resolved = walk(target, stack + (name,))
                if isinstance(resolved, dict) and resolved.get("$ref") == _EXTRACTED_VALUE_REF:
                    # A leaf reached through a shared definition: keep only its
                    # structural sidecars. Its prose (title/description) states
                    # the shared value mechanics, not anything about this
                    # field — the field's own prose comes from the $ref node's
                    # siblings below.
                    resolved = {
                        k: v for k, v in resolved.items() if k in {"$ref", "datatype", "value_enum"}
                    }
                siblings = {
                    k: v for k, v in node.items() if k in _REF_SIBLING_KEYS and v is not None
                }
                if siblings and isinstance(resolved, dict):
                    resolved = {**resolved, **siblings}
                return resolved
            return node  # unknown ref form — leave for the walker to reject
        if _is_leaf_shape(node):
            leaf: dict[str, Any] = {"$ref": _EXTRACTED_VALUE_REF}
            leaf.update(_value_subschema_types(node.get("properties", {}).get("value")))
            for k in ("title", "description", "prompt", "example"):
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    leaf[k] = v
            return leaf
        return {k: walk(v, stack) for k, v in node.items() if k not in {"$defs", "definitions"}}

    root: dict[str, Any] = schema
    root_ref = schema.get("$ref")
    if isinstance(root_ref, str):
        found = deref(root_ref)
        if found is None or not isinstance(found[1], dict):
            raise SchemaInvalid(f"root $ref '{root_ref}' does not resolve to a local definition")
        root = found[1]
    resolved_root = walk(root, ())
    if not isinstance(resolved_root, dict):
        raise SchemaInvalid("schema root must resolve to a JSON object")
    return resolved_root


def json_schema_to_vocabulary(schema: dict[str, Any], *, namespace_uri: str) -> Vocabulary:
    """Build a :class:`Vocabulary` from an extraction JSON Schema.

    Accepts the internal extracted_value dialect this module emits, the
    legacy grounded_field forms, and standard-dialect schemas (root ``$ref``,
    ``$defs``, ``title``, structural leaf shapes) via
    :func:`_resolve_schema_dialect`.
    """
    if not isinstance(schema, dict):
        raise SchemaInvalid("schema must be a JSON object")
    resolved = _resolve_schema_dialect(schema)
    roots = _properties_to_tags(resolved.get("properties"))
    if not roots:
        raise SchemaInvalid("schema has no 'properties' — nothing to extract")
    return Vocabulary(namespace_uri=namespace_uri, roots=roots)


# ── Typed field tree → Vocabulary ────────────────────────────────────────────
#
# The schema-generation LLM submits a *typed field tree* — a recursive list of
# ``{name, kind, datatype?, description?, example?, prompt?, fields?, item?}``
# nodes — which maps one-to-one onto :class:`Tag`. This is the direct path from
# an LLM proposal to the at-rest RNC (:func:`field_tree_to_rnc`): the model
# picks a datatype per leaf, so the vocabulary carries ``value_type`` natively
# and :func:`vocabulary_to_rnc` emits ``xsd:`` typed leaves — no grounded_field
# JSON Schema in between.

# XSD datatypes a leaf may declare. ``text`` (or an omitted datatype) is the
# untyped default. The rest all round-trip through the extraction serializer's
# value normalization (``dgml_core.extraction_xml._typed_value``): ``integer``
# and ``decimal`` get numeric cleanup, the others keep their declared type with
# the value detector's normalized ``dg:value``.
FIELD_DATATYPES: frozenset[str] = frozenset(
    {"date", "dateTime", "decimal", "integer", "boolean", "gYear", "time", "anyURI"}
)
_TEXT_DATATYPES: frozenset[str] = frozenset({"text", "string", ""})


def _normalize_datatype(raw: Any, *, tag_name: str) -> str | None:
    """Map a node's ``datatype`` to a :class:`Tag` ``value_type`` (``None`` = text)."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise SchemaInvalid(f"field '{tag_name}' has a non-string datatype {raw!r}")
    dt = raw.strip()
    if dt.startswith("xsd:"):
        dt = dt[4:]
    if dt in _TEXT_DATATYPES:
        return None
    if dt in FIELD_DATATYPES:
        return dt
    raise SchemaInvalid(
        f"field '{tag_name}' has unsupported datatype {raw!r}; "
        f"use 'text' or one of {sorted(FIELD_DATATYPES)}"
    )


def _normalize_enum_values(raw: Any, *, tag_name: str) -> list[str] | None:
    """Validate a leaf's enum token list (``None`` = not an enum field).

    Tokens must be non-empty strings without double quotes or control
    characters, so the RNC value enumeration (`( "a" | "b" )`) stays
    parseable and round-trips byte-for-byte.
    """
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise SchemaInvalid(f"field '{tag_name}' has a non-list or empty enum: {raw!r}")
    tokens: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            raise SchemaInvalid(f"field '{tag_name}' has a non-string or empty enum value")
        token = entry.strip()
        if '"' in token or "\\" in token or any(ord(c) < 0x20 for c in token):
            raise SchemaInvalid(
                f"field '{tag_name}' enum value {token!r} contains a quote, backslash, "
                "or control character"
            )
        if token in tokens:
            raise SchemaInvalid(f"field '{tag_name}' enum repeats value {token!r}")
        tokens.append(token)
    return tokens


# The `## Invariant:` grammar — deliberately two function forms over a dotted
# path, not an expression language. These cover the relations that actually
# recur in extraction schemas: a count field agreeing with the collection it
# counts, and a total agreeing with the leaf it sums across entries.
#
#   count(Path.To.Collection)
#   sum(Path.To.Collection[].LeafName)
#
# Path segments are tag names (PascalCase); `[]` marks the collection whose
# entries are summed. Checked report-only after extraction — see
# dgml_core.extraction_xml.check_invariants.
#
# Two limits are deliberate, and both are load-bearing when deciding whether a
# given schema rule is expressible here:
#
# * **One term.** `sum(A[].x) + sum(B[].y)` has no form. A rule that spans two
#   collections (a total composed of one list's amounts *plus* a second list's,
#   say) cannot be written as an invariant and must not be approximated
#   by one of its terms — that reports a violation on correct output.
# * **Paths resolve from the submission root, through dict hops only.** A field
#   nested inside a collection entry therefore cannot reference a sibling
#   collection within that same entry; there is no relative path form.
#
# Anything outside the two forms is rejected at schema load rather than skipped,
# so an unexpressible rule fails loudly instead of silently never running.
_INVARIANT_COUNT_RE = re.compile(r"^count\(\s*([A-Za-z_][\w.]*)\s*\)$")
_INVARIANT_SUM_RE = re.compile(r"^sum\(\s*([A-Za-z_][\w.]*)\[\]\.(\w+)\s*\)$")


def parse_invariant(text: str) -> tuple[str, str, str | None] | None:
    """Parse an invariant into ``(kind, collection_path, leaf_name)``.

    ``kind`` is ``"count"`` (``leaf_name`` is ``None``) or ``"sum"``. Returns
    ``None`` when *text* is not a recognized form, so callers can decide
    between rejecting a schema and skipping a check.
    """
    count = _INVARIANT_COUNT_RE.match(text.strip())
    if count:
        return ("count", count.group(1), None)
    total = _INVARIANT_SUM_RE.match(text.strip())
    if total:
        return ("sum", total.group(1), total.group(2))
    return None


def _validate_invariant(text: str) -> str:
    """Accept a supported invariant form, else raise :class:`SchemaInvalid`."""
    if parse_invariant(text) is None:
        raise SchemaInvalid(
            f"unsupported '## Invariant:' {text!r}; expected count(Path.To.Collection) "
            "or sum(Path.To.Collection[].LeafName)"
        )
    return text.strip()


def _leaf_value_types(
    node: dict[str, Any], *, tag_name: str
) -> tuple[str | None, list[str] | None]:
    """Read a JSON Schema leaf node's ``datatype``/``value_enum`` sidecars into
    the (value_type, enum_values) pair a :class:`Tag` carries. The two are
    mutually exclusive — an enum IS the leaf's type."""
    value_type = _normalize_datatype(node.get("datatype"), tag_name=tag_name)
    enum_values = _normalize_enum_values(node.get("value_enum"), tag_name=tag_name)
    if value_type is not None and enum_values is not None:
        raise SchemaInvalid(f"field '{tag_name}' cannot carry both a datatype and an enum")
    return value_type, enum_values


def _field_node_to_tag(node: Any) -> Tag:
    if not isinstance(node, dict):
        raise SchemaInvalid(f"schema node must be an object, got {type(node).__name__}")
    raw_name = node.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise SchemaInvalid(f"schema node is missing a non-empty 'name': {node!r}")
    name = _pascal_case(raw_name) or "Field"

    kind = node.get("kind", "field")
    if not isinstance(kind, str):
        raise SchemaInvalid(f"node '{name}' has a non-string 'kind'")
    kind = kind.strip().lower()

    description = node.get("description")
    example = node.get("example")
    prompt = node.get("prompt")

    if kind == "field":
        enum_values = _normalize_enum_values(node.get("enum"), tag_name=name)
        value_type = _normalize_datatype(node.get("datatype"), tag_name=name)
        if enum_values is not None and value_type is not None:
            raise SchemaInvalid(f"field '{name}' cannot carry both a datatype and an enum")
        return Tag(
            name=name,
            kind="field",
            description=description,
            example=example,
            prompt=prompt,
            invariant=(
                _validate_invariant(node["invariant"])
                if isinstance(node.get("invariant"), str)
                else None
            ),
            value_type=value_type,
            enum_values=enum_values,
        )

    if kind == "container":
        return Tag(
            name=name,
            kind="container",
            description=description,
            example=example,
            prompt=prompt,
            children=_field_nodes_to_tags(node.get("fields"), context=name),
        )

    if kind == "collection":
        # The repeated item is either given explicitly as ``item`` (a node) or
        # implied by the collection's own ``fields`` (the item's fields).
        raw_item = node.get("item")
        if isinstance(raw_item, dict):
            item_tag = _field_node_to_tag(raw_item)
            if item_tag.kind == "field":
                # A list of bare typed values: keep the item as a leaf field.
                children: list[Tag] = []
            else:
                children = item_tag.children
        else:
            item_name = _singularize(name)
            children = _field_nodes_to_tags(node.get("fields"), context=name)
            item_tag = Tag(name=item_name, kind="container", children=children)
        return Tag(
            name=name,
            kind="collection",
            description=description,
            example=example,
            prompt=prompt,
            children=children,
            item_name=item_tag.name,
            item=item_tag,
        )

    raise SchemaInvalid(
        f"node '{name}' has unknown kind {kind!r}; use 'field', 'container', or 'collection'"
    )


def _field_nodes_to_tags(nodes: Any, *, context: str) -> list[Tag]:
    if nodes is None:
        return []
    if not isinstance(nodes, list):
        raise SchemaInvalid(f"'{context}' fields must be a list of nodes")
    tags: list[Tag] = []
    seen: set[str] = set()
    for node in nodes:
        tag = _field_node_to_tag(node)
        if tag.name in seen:
            raise SchemaInvalid(f"duplicate tag name '{tag.name}' under '{context}'")
        seen.add(tag.name)
        tags.append(tag)
    return tags


def field_tree_to_vocabulary(fields: Any, *, namespace_uri: str) -> Vocabulary:
    """Build a :class:`Vocabulary` from an LLM-submitted typed field tree.

    *fields* is the list of top-level nodes (see the module comment above).
    Raises :class:`SchemaInvalid` for a malformed tree.
    """
    roots = _field_nodes_to_tags(fields, context="<root>")
    if not roots:
        raise SchemaInvalid("field tree is empty — nothing to extract")
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
    if tag.invariant:
        extra["invariant"] = tag.invariant

    if tag.kind == "field":
        # Leaves are extracted_value refs. `datatype`/`value_enum` ride along
        # as sidecar keys so the projection is lossless (and `value_enum` is
        # specialized into the leaf's `value.enum` when the engine inlines the
        # tool schema — see grounded._expand_refs).
        node: dict[str, Any] = {"$ref": _EXTRACTED_VALUE_REF}
        if tag.value_type:
            node["datatype"] = tag.value_type
        if tag.enum_values:
            node["value_enum"] = list(tag.enum_values)
        node.update(extra)
        return node
    if tag.kind == "collection":
        if tag.item is not None and tag.item.kind == "field":
            items_node = _tag_to_node(tag.item)
        else:
            items_node = {"type": "object", "properties": _tags_to_properties(tag.children)}
        node = {"type": "array", "items": items_node}
        # The singular item's name only needs a sidecar when it can't be
        # re-derived from the collection name (keeps the round-trip lossless
        # without cluttering the common case).
        item_name = tag.item.name if tag.item is not None else tag.item_name
        if item_name and item_name != _singularize(tag.name):
            node["item_name"] = item_name
        node.update(extra)
        return node
    if tag.kind == "container":
        return {"type": "object", "properties": _tags_to_properties(tag.children), **extra}
    if tag.kind == "choice":
        # Either the typed scalar (a grounded or computed value) or the group
        # of children.
        return {
            "anyOf": [
                {"$ref": _EXTRACTED_VALUE_REF},
                {"type": "object", "properties": _tags_to_properties(tag.children)},
            ],
            **extra,
        }
    raise SchemaInvalid(f"unknown tag kind '{tag.kind}'")


def _tags_to_properties(tags: list[Tag]) -> dict[str, Any]:
    return {tag.name: _tag_to_node(tag) for tag in tags}


def vocabulary_to_json_schema(vocab: Vocabulary) -> dict[str, Any]:
    """Render a :class:`Vocabulary` to the extracted_value JSON Schema the
    extraction engine drives off.

    Carries ``description``/``example``/``prompt``/``datatype``/``value_enum``
    onto each property so the RNC ⇄ JSON ⇄ RNC round-trip is lossless. The
    extraction engine treats the sidecar keys as annotations (``value_enum``
    is additionally specialized into the tool schema's ``value.enum``).
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "definitions": {"extracted_value": EXTRACTED_VALUE},
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
    if tag.invariant:
        lines.append(f"## Invariant: {tag.invariant}")
    return "".join(f"{line}\n" for line in lines)


# Canonical wrapping for RNC value enumerations: packed on one line when the
# content line (4-space indent included) fits the 100-column limit, otherwise
# one token per line. Deterministic from the token list, so the round-trip
# stays byte-for-byte.
_ENUM_PACKED_MAX = 96  # 100 columns minus the 4-space content indent


def _enum_content(values: list[str]) -> str:
    parts = [f'"{v}"' for v in values]
    packed = f"( {' | '.join(parts)} )"
    if len(packed) <= _ENUM_PACKED_MAX:
        return packed
    lines = [f"( {parts[0]}"]
    lines.extend(f"      | {p}" for p in parts[1:])
    return "\n".join(lines) + " )"


def _collection_item_tag(tag: Tag) -> Tag:
    return tag.item or Tag(
        name=tag.item_name or _singularize(tag.name),
        kind="container",
        children=tag.children,
    )


def _render_def(tag: Tag) -> str:
    """The RNC pattern def for *tag* alone (descendants render separately)."""
    if tag.kind == "field":
        if tag.enum_values:
            content = _enum_content(tag.enum_values)
        else:
            content = f"xsd:{tag.value_type}" if tag.value_type else "text"
    elif tag.kind == "choice":
        scalar = f"xsd:{tag.value_type}" if tag.value_type else "text"
        group = ", ".join(c.name for c in tag.children)
        content = f"( {scalar} | ( {group} ) )"
    elif tag.kind == "collection":
        content = f"{_collection_item_tag(tag).name}*"
    elif tag.kind == "container":
        refs = " | ".join(["text", *(c.name for c in tag.children)])
        content = f"({refs})*"
    else:
        raise SchemaInvalid(f"unknown tag kind '{tag.kind}'")
    return f"{_doc_comment(tag)}{tag.name} =\n  element docset:{tag.name} {{\n    {content}\n  }}\n"


def _emit_tag_defs(tag: Tag, out: list[str], seen: dict[str, str]) -> None:
    """Append the RNC pattern def(s) for *tag* (and its descendants) to *out*.

    *seen* maps each emitted pattern name to its rendered def. A tag whose
    rendering is byte-identical to an already-emitted def is intentional
    reuse (e.g. one line-item structure referenced from two levels) and
    is skipped; the same name rendering *differently* is a silent-collapse
    hazard — two distinct fields would share one definition, and one side's
    annotations/content would win arbitrarily — so it is a hard error.
    """
    rendered = _render_def(tag)
    prev = seen.get(tag.name)
    if prev is not None:
        if prev != rendered:
            raise SchemaInvalid(
                f"tag '{tag.name}' is defined twice with different content or "
                "annotations; every occurrence of a tag name shares one schema "
                f"definition — rename one occurrence (e.g. a level-qualified "
                f"name like '{tag.name}' prefixed with its parent's name)"
            )
        return
    seen[tag.name] = rendered
    out.append(rendered)

    if tag.kind == "collection":
        # Recurse on the singular item (a container) so it and its children are
        # emitted exactly once, carrying the item's own annotations.
        _emit_tag_defs(_collection_item_tag(tag), out, seen)
    elif tag.kind in ("choice", "container"):
        for child in tag.children:
            _emit_tag_defs(child, out, seen)


def _collect_referenced_names(tag: Tag, acc: set[str]) -> None:
    """Names *tag*'s definition (and its descendants') content models reference."""
    if tag.kind == "collection":
        item = _collection_item_tag(tag)
        acc.add(item.name)
        _collect_referenced_names(item, acc)
    elif tag.kind in ("container", "choice"):
        for child in tag.children:
            acc.add(child.name)
            _collect_referenced_names(child, acc)


def vocabulary_to_rnc(vocab: Vocabulary) -> str:
    """Serialize a :class:`Vocabulary` to RELAX NG Compact (the at-rest form).

    Matches the spec's docset-schema form (§12/§13): a ``namespace docset``
    declaration followed by element definitions, roots first. Normally there
    is no ``start``/``dg:chunk`` rule — roots are the element defs not
    referenced as a child by any other element. When some root's tag is ALSO
    referenced by another definition (a document-level field whose identical
    definition is reused inside a nested structure), that implicit rule would
    silently demote the root on re-parse, so an explicit ``start`` rule naming
    the roots is emitted in that case.
    """
    referenced: set[str] = set()
    for tag in vocab.roots:
        _collect_referenced_names(tag, referenced)

    parts: list[str] = [f'namespace docset = "{vocab.namespace_uri}"\n']
    if any(tag.name in referenced for tag in vocab.roots):
        parts.append("\n")
        parts.append(_render_start_rule([tag.name for tag in vocab.roots]))
    defs: list[str] = []
    seen: dict[str, str] = {}
    for tag in vocab.roots:
        _emit_tag_defs(tag, defs, seen)
    for body in defs:
        parts.append("\n")
        parts.append(body)
    return "".join(parts)


def _render_start_rule(root_names: list[str]) -> str:
    """The explicit ``start`` rule naming the docset roots, wrapped like enum
    content: packed on one line when it fits the 100-column limit, else one
    ref per line (the parser joins lines before matching)."""
    refs = ["text", *root_names]
    packed = f"({' | '.join(refs)})*"
    if len(packed) <= _ENUM_PACKED_MAX:
        content = packed
    else:
        lines = [f"({refs[0]}"]
        lines.extend(f"      | {r}" for r in refs[1:])
        content = "\n".join(lines) + ")*"
    return f"start =\n  element dg:chunk {{\n    {content}\n  }}\n"


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
# ( "a" | "b" | ... ) — a value enumeration (matched against the whitespace-
# joined body so the canonical one-token-per-line wrapping parses too).
_ENUM_BODY_RE = re.compile(r'^\(\s*"[^"]*"(?:\s*\|\s*"[^"]*")*\s*\)$')
_ENUM_TOKEN_RE = re.compile(r'"([^"]*)"')
_START_BODY_RE = re.compile(r"^\s*element\s+dg:chunk\s*\{\s*$")


@dataclass
class _RawDef:
    name: str
    description: str | None
    example: str | None
    prompt: str | None
    invariant: str | None
    body_lines: list[str]  # the element's content-model lines


def _parse_doc_comments(
    comments: list[str],
) -> tuple[str | None, str | None, str | None, str | None]:
    desc_lines: list[str] = []
    example: str | None = None
    prompt: str | None = None
    invariant: str | None = None
    for raw in comments:
        text = raw[2:].strip() if raw.startswith("##") else raw.strip()
        if text.startswith("Example:"):
            example = text[len("Example:") :].strip()
        elif text.startswith("Prompt:"):
            prompt = text[len("Prompt:") :].strip()
        elif text.startswith("Invariant:"):
            invariant = _validate_invariant(text[len("Invariant:") :].strip())
        else:
            desc_lines.append(text)
    description = "\n".join(desc_lines).strip() or None
    return description, example, prompt, invariant


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
            description, example, prompt, invariant = _parse_doc_comments(pending_comments)
            defs[name] = _RawDef(
                name=name,
                description=description,
                example=example,
                prompt=prompt,
                invariant=invariant,
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
    # The content model may be wrapped across lines (one ref per line, the
    # canonical form for many roots) — join before matching.
    joined = " ".join(line.strip() for line in block[1:] if line.strip())
    cont = _CONTAINER_RE.match(joined)
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

    # Value enumeration — may span multiple lines (canonical wrapping), so it
    # is matched against the whitespace-joined body.
    joined = " ".join(line.strip() for line in body)
    if _ENUM_BODY_RE.match(joined):
        return Tag(
            name=raw.name,
            kind="field",
            description=raw.description,
            example=raw.example,
            prompt=raw.prompt,
            invariant=raw.invariant,
            enum_values=_normalize_enum_values(_ENUM_TOKEN_RE.findall(joined), tag_name=raw.name),
        )

    if single and _TEXT_RE.match(single):
        return Tag(
            name=raw.name,
            kind="field",
            description=raw.description,
            example=raw.example,
            prompt=raw.prompt,
            invariant=raw.invariant,
        )

    typed = _TYPED_RE.match(single) if single else None
    if typed:
        return Tag(
            name=raw.name,
            kind="field",
            description=raw.description,
            example=raw.example,
            prompt=raw.prompt,
            invariant=raw.invariant,
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


def field_tree_to_rnc(fields: Any, *, workspace: str, docset_name: str) -> str:
    """Render an LLM-submitted typed field tree straight to the at-rest RNC form.

    This is the schema-generation path: the model proposes a typed field tree
    (each leaf carrying a datatype), which maps directly onto the vocabulary and
    its RNC serialization — no grounded_field JSON Schema in between. The docset
    namespace is built from *workspace* and *docset_name* exactly as
    :func:`json_schema_to_rnc` does, so extraction and generated docsets share
    one namespace.
    """
    namespace_uri = f"http://dgml.io/{org_ns_segment(workspace)}/{docset_slug(docset_name)}"
    vocab = field_tree_to_vocabulary(fields, namespace_uri=namespace_uri)
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
