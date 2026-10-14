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
schema: a leaf is ``{"text": ..., "value"?: ..., "locations": [{"page_number",
"bounding_box"}]}`` (``value`` is the model's normalized form — for enum fields
the schema token, validated against the enum before it becomes ``dg:value``),
a container is a dict of child fields, and a collection is a list of such dicts.

Provenance follows DGML 0.71 (see :mod:`dgml_core.dg_refs`). A leaf may name
the other values it used — ``"inputs": [dotted paths]`` — and, when the page
doesn't print it, say how it was produced — ``"method": "calculated" |
"inferred" | "defaulted"``. That gives four cases:

* ``locations`` only — read from its printed text;
* ``locations`` + ``inputs`` — printed in part ("3/1"), completed from the
  inputs (the commencement year);
* ``inputs`` + ``method`` (``calculated`` or ``inferred``) — not printed,
  derived from the inputs;
* ``method: "defaulted"`` — a producer default, with no inputs.

``locations`` serialize as ``dg:origin`` boxes, ``inputs`` as ``dg:inputs``
(``"#id; #id"``), ``method`` as ``dg:method``, and an unprinted leaf always
carries ``dg:value``. Each referenced element gets a stable ``xml:id`` derived
from its path so the ``dg:inputs`` targets resolve within the file. The 0.7
form (``dg:origin="computed"`` + ``dg:itemprop="computedFrom"``/``dg:href``)
is never written, and is read as its 0.71 equivalent.

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
from decimal import Decimal, InvalidOperation
from typing import Any, NamedTuple

from lxml import etree  # type: ignore[import-untyped]

from .dg_refs import (
    INPUTS_ATTR,
    METHOD_ATTR,
    METHOD_CALCULATED,
    METHOD_DEFAULTED,
    METHOD_INFERRED,
    METHODS,
    ORIGIN_ATTR,
    effective_inputs,
    effective_method,
    effective_origin,
    format_refs,
    is_legacy_computed,
    parse_ref,
)
from .extraction_schema import Tag, Vocabulary, parse_invariant, resolve_invariant_path
from .generation.semantic_transform import _detect_value_type
from .matching import (
    LeafPath,
    get_at_path,
    is_unprinted_leaf,
    parse_path,
    path_to_str,
    walk_all_leaves,
    walk_input_leaves,
)

DG_NS = "http://dgml.io/ns/dg#"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
XML_NS = "http://www.w3.org/XML/1998/namespace"

_NSMAP_BASE = {"dg": DG_NS, "xsi": XSI_NS}
EXTRACTION_TAG = f"{{{DG_NS}}}extraction"
CHUNK_TAG = f"{{{DG_NS}}}chunk"


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


# ── value provenance: inputs and method (DGML 0.71) ──────────────────────────

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
    """Assign an ``xml:id`` to every values-tree path some leaf's ``inputs``
    references. Malformed or dangling paths get no id — the corresponding
    ``dg:inputs`` entry is dropped at emission."""
    ids: dict[LeafPath, str] = {}
    for _path, leaf in walk_input_leaves(values):
        refs = leaf.get("inputs")
        if not isinstance(refs, list):
            continue
        for raw in refs:
            path = _resolvable_input(values, raw)
            if path is not None and path not in ids:
                ids[path] = _id_for_path(path)
    return ids


def _resolvable_input(values: dict[str, Any], raw: Any) -> LeafPath | None:
    """The path an ``inputs`` entry names, when it will emit a ``dg:inputs``
    target: a well-formed dotted path to a field, container or collection in
    *values* — something the serializer emits as an element that can carry
    the ``xml:id``. ``None`` for a non-string, malformed or dangling entry,
    and for a path into a leaf's own keys (``Total.value``,
    ``EndDate.evidence[0]``), which no element answers to; all are dropped."""
    if not isinstance(raw, str):
        return None
    path = parse_path(raw)
    if path is None or not isinstance(get_at_path(values, path), (dict, list)):
        return None
    if any(_is_leaf(get_at_path(values, path[:i])) for i in range(1, len(path))):
        return None
    return path


def _is_leaf(node: Any) -> bool:
    """True for a values-tree leaf (a dict with a ``text`` key — tag names are
    PascalCase, so no field is called that)."""
    return isinstance(node, dict) and "text" in node


def _evidence_boxed(item: Any) -> bool:
    """True for an ``evidence`` item with a bounding box, which emits a
    ``dg:chunk``; one without is dropped."""
    return isinstance(item, dict) and _origin_from_locations(item.get("locations")) is not None


def count_dropped_inputs(values: dict[str, Any], *, evidence: bool = True) -> int:
    """Number of ``inputs`` entries across all leaves that will NOT emit a
    ``dg:inputs`` target — malformed path, non-string entry, or a path that
    dangles in *values* — plus ``evidence`` items with no bounding box, which
    emit no ``dg:chunk``. Mirrors the drop logic in
    :func:`_collect_ref_ids`/:func:`_resolved_inputs`/:func:`_add_evidence`;
    surfaced in ``extraction_stats.json`` so incomplete provenance is visible
    instead of silently shrinking the inputs list.

    ``evidence=False`` skips the evidence items: before phases 2/3 none has a
    box yet."""
    dropped = 0
    for _path, leaf in walk_input_leaves(values):
        items = leaf.get("evidence") if evidence else None
        if isinstance(items, list):
            dropped += sum(1 for item in items if not _evidence_boxed(item))
        refs = leaf.get("inputs")
        if isinstance(refs, list):
            dropped += sum(1 for raw in refs if _resolvable_input(values, raw) is None)
    return dropped


# ── input citations: verify, repair or drop positional input paths ───────────

_CITE_SEP = "="
_CITE_NORM_RE = re.compile(r"[^0-9a-z.\-]+")


class CitationResult(NamedTuple):
    """What :func:`verify_input_citations` did: inputs re-pointed (a
    miscounted index, or a path written relative to the citing value), and
    cited inputs dropped because nothing — or more than one thing — matched."""

    repaired: int
    dropped: int


def _cite_norm(text: Any) -> str:
    """Text as compared for a citation: lowercase, with only letters, digits,
    ``.`` and ``-`` kept — so "$1,234.50" and "1234.50" agree."""
    return _CITE_NORM_RE.sub("", str(text).lower())


# A plain decimal: no exponent ("1e3"), NaN or Infinity, which Decimal would
# also parse (and a signalling NaN raises when compared).
_CITE_NUMBER_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)")


def _cite_number(norm: str) -> Decimal | None:
    if not _CITE_NUMBER_RE.fullmatch(norm):
        return None
    try:
        return Decimal(norm)
    except InvalidOperation:
        return None


def _cited_text_matches(node: Any, cited: str) -> bool:
    """True when leaf *node* reads as *cited*: its verbatim ``text`` or its
    normalized ``value`` equals the citation, ignoring case, spacing and
    punctuation — or, when both are numbers, has the same numeric value
    (".45" cites "0.45", "1,234.50" cites "1234.5")."""
    if not _is_leaf(node):
        return False
    want = _cite_norm(cited)
    if not want:
        # Nothing but punctuation ("$", "%"): compare the bare text instead.
        bare = " ".join(cited.split()).lower()
        return bool(bare) and any(
            " ".join(str(node[key]).split()).lower() == bare
            for key in ("text", "value")
            if node.get(key) is not None
        )
    want_number = _cite_number(want)
    for key in ("text", "value"):
        if node.get(key) is None:
            continue
        have = _cite_norm(node[key])
        if have == want:
            return True
        if want_number is not None and (have_number := _cite_number(have)) is not None:
            if have_number == want_number:
                return True
    return False


def _repair_index(values: dict[str, Any], path: LeafPath, cited: str) -> LeafPath | None:
    """The one path that differs from *path* only in its last array index and
    whose leaf matches *cited*, or ``None`` when no sibling does or several do
    (two rows with the same amount can't be told apart)."""
    last = max((i for i, seg in enumerate(path) if isinstance(seg, int)), default=None)
    if last is None:
        return None
    siblings = get_at_path(values, path[:last])
    if not isinstance(siblings, list):
        return None
    found = [
        candidate
        for j in range(len(siblings))
        if _cited_text_matches(
            get_at_path(values, candidate := path[:last] + (j,) + path[last + 1 :]), cited
        )
    ]
    return found[0] if len(found) == 1 else None


def _anchorings(leaf_path: LeafPath, path: LeafPath) -> list[LeafPath]:
    """Where *path* may point: from the root first, then — for a path written
    relative to the citing value ("LineItemName" for a line item's own name,
    "ChargeLineItems[6].Amount" from inside its meter reading) — under each of
    the value's ancestors, nearest first."""
    out = [path]
    for k in range(len(leaf_path) - 1, 0, -1):
        candidate = leaf_path[:k] + path
        if candidate not in out:
            out.append(candidate)
    return out


def _resolve_input(
    values: dict[str, Any], leaf_path: LeafPath, path: LeafPath, cited: str | None
) -> tuple[LeafPath | None, bool]:
    """``(target path, re-pointed?)`` for one input, or ``(None, False)`` when
    nothing can be trusted.

    With a citation, the first anchoring whose leaf reads as the cited text
    wins; failing that, the first one with exactly one matching sibling at its
    last array index (a miscounted index). A target that isn't a leaf (a whole
    row) can't be compared, so the first anchoring that resolves stands.
    Without a citation, the first anchoring that resolves stands."""
    anchorings = _anchorings(leaf_path, path)
    resolving = [(c, t) for c in anchorings if (t := get_at_path(values, c)) is not None]
    if cited is None or (resolving and not _is_leaf(resolving[0][1])):
        if not resolving:
            return None, False
        return resolving[0][0], resolving[0][0] != path
    for candidate, target in resolving:
        if _cited_text_matches(target, cited):
            return candidate, candidate != path
    for candidate in anchorings:
        if (fixed := _repair_index(values, candidate, cited)) is not None:
            return fixed, True
    return None, False


def verify_input_citations(values: dict[str, Any]) -> CitationResult:
    """Check every input against the text it cites, in place.

    The model writes an input as ``"<dotted path>=<the text it read there>"``
    (``"MeterReadings[0].ChargeLineItems[6].Amount=58.20"``). Two slips are
    common and both name a real but wrong (or no) element — one ``dgml check``
    can't catch when the target exists:

    * a miscounted array index, e.g. after two identical rows;
    * a path written relative to the citing value instead of from the root.

    The cited text makes both fixable: an input is kept when its target reads
    as the citation, re-pointed (counted as repaired) when the citation
    matches under one of the value's ancestors or at exactly one sibling
    index, and dropped (counted) otherwise — never left pointing at the wrong
    element. See :func:`_resolve_input`.

    Every entry is rewritten to its bare root path, which is what the
    serializer and the values JSON carry. An uncited entry is anchored the
    same way but can't be checked; one that resolves nowhere is left for the
    serializer to drop and :func:`count_dropped_inputs` to count.
    """
    repaired = dropped = 0
    for leaf_path, leaf in walk_all_leaves(values):
        refs = leaf.get("inputs")
        if not isinstance(refs, list):
            continue
        kept: list[Any] = []
        for raw in refs:
            if not isinstance(raw, str):
                kept.append(raw)
                continue
            path_text, sep, cited_text = raw.partition(_CITE_SEP)
            path_text = path_text.strip()
            path = parse_path(path_text)
            if path is None:
                if path_text:
                    kept.append(path_text)  # malformed; counted downstream
                else:
                    # No path at all ("=58.20"): nothing downstream would
                    # see it to count, so count the drop here.
                    dropped += 1
                continue
            # An empty citation ("Total=") checks nothing: treat it as uncited
            # rather than dropping a path that may well be right.
            cited = cited_text if sep and cited_text.strip() else None
            target, moved = _resolve_input(values, leaf_path, path, cited)
            if target == leaf_path:
                # A value can't be its own input (a relative path anchored
                # back onto the citing value): that is a dg:inputs cycle.
                dropped += 1
                continue
            if target is None:
                if cited is None:
                    kept.append(path_text)  # dangles; counted downstream
                else:
                    dropped += 1
                continue
            kept.append(path_to_str(target))
            repaired += moved
        if kept:
            leaf["inputs"] = kept
        else:
            del leaf["inputs"]
    return CitationResult(repaired, dropped)


# ── printed values grounding couldn't box are dropped ────────────────────────


class UnboxedResult(NamedTuple):
    """What :func:`drop_unboxed_values` removed: printed values with no box
    (``values``), unprinted values with nothing to verify them by
    (``unsupported``), inputs that named a removed value (``inputs``), and
    each removed value as ``"path=text"`` (``paths``)."""

    values: int
    inputs: int
    paths: list[str]
    unsupported: int = 0


def _unverifiable(leaf: dict[str, Any], values: dict[str, Any]) -> str | None:
    """Why *leaf* can't be written, or ``None`` when it can: ``"unboxed"``
    for a printed value no box was found for, ``"unsupported"`` for a value
    with no box, no default, and nothing it used (no resolvable input and no
    boxed evidence) — the model returned it bare, or claimed a calculation
    or inference and named nothing."""
    if "locations" in leaf:
        return "unboxed" if _origin_from_locations(leaf["locations"]) is None else None
    method = leaf.get("method")
    if method == METHOD_DEFAULTED:
        return None
    if method in (METHOD_CALCULATED, METHOD_INFERRED):
        refs = leaf.get("inputs")
        items = leaf.get("evidence")
        if (
            isinstance(refs, list) and any(_resolvable_input(values, r) is not None for r in refs)
        ) or (isinstance(items, list) and any(_evidence_boxed(i) for i in items)):
            return None
    return "unsupported"


def drop_unboxed_values(values: dict[str, Any]) -> UnboxedResult:
    """Remove every value that can't be verified, in place.

    Every value written must be checkable: a printed value (one with
    ``locations``) by its box, an unprinted one by its ``method`` and what it
    used. So two kinds are not written (see :func:`_unverifiable`):

    * a printed value grounding found no box for — written without
      ``dg:origin`` it would read as unsupported, or, with ``inputs``, as an
      inferred judgment;
    * a value with no box, no default and nothing it used.

    A row or section emptied by the removals goes too. Removing a list entry
    shifts the ones after it, so every remaining ``inputs`` path is
    re-indexed; an input naming a removed value is dropped. That can leave
    another value with nothing it used, so this repeats until nothing more is
    removed."""
    unboxed = unsupported = dropped_inputs = 0
    dropped_paths: list[str] = []
    while True:
        marks = {
            path: reason
            for path, leaf in walk_all_leaves(values)
            if (reason := _unverifiable(leaf, values)) is not None
        }
        if not marks:
            break
        pruned = _prune_marked(values, marks)
        for path_text, reason in pruned.dropped:
            if reason == "unboxed":
                unboxed += 1
                dropped_paths.append(path_text)
            else:
                unsupported += 1
                dropped_paths.append(f"{path_text} (unsupported)")
        dropped_inputs += _remap_inputs(values, pruned.removed, pruned.removed_index)
    return UnboxedResult(unboxed, dropped_inputs, dropped_paths, unsupported)


class _Pruned(NamedTuple):
    removed: set[LeafPath]
    removed_index: dict[LeafPath, list[int]]
    dropped: list[tuple[str, str]]  # ("path=text", reason) per removed leaf


def _prune_marked(values: dict[str, Any], marks: dict[LeafPath, str]) -> _Pruned:
    """Remove the leaves at the *marks* paths, and any row or section that
    leaves empty, in place. Returns what was removed (in pre-removal paths),
    the removed indices of each list, and each removed leaf with its reason."""
    removed: set[LeafPath] = set()
    removed_index: dict[LeafPath, list[int]] = {}
    dropped: list[tuple[str, str]] = []

    def prune(node: Any, path: LeafPath) -> bool:
        if isinstance(node, dict):
            if _is_leaf(node):
                reason = marks.get(path)
                if reason is None:
                    return False
                dropped.append((f"{path_to_str(path)}={node.get('text')}", reason))
                removed.add(path)
                return True
            had = bool(node)
            for key in list(node):
                if prune(node[key], path + (key,)):
                    del node[key]
            if had and not node:
                removed.add(path)
                return True
            return False
        if isinstance(node, list):
            had = bool(node)
            kept = []
            for i, entry in enumerate(node):
                if prune(entry, path + (i,)):
                    removed_index.setdefault(path, []).append(i)
                else:
                    kept.append(entry)
            node[:] = kept
            if had and not node:
                removed.add(path)
                return True
        return False

    prune(values, ())
    return _Pruned(removed, removed_index, dropped)


def _remap_inputs(
    values: dict[str, Any], removed: set[LeafPath], removed_index: dict[LeafPath, list[int]]
) -> int:
    """Re-index every ``inputs`` path past removed list entries, in place;
    drop (and count) inputs that named a removed value."""

    def remap(path: LeafPath) -> LeafPath | None:
        if any(path[:k] in removed for k in range(1, len(path) + 1)):
            return None
        out: list[Any] = []
        for k, seg in enumerate(path):
            if isinstance(seg, int):
                seg -= sum(1 for r in removed_index.get(path[:k], ()) if r < seg)
            out.append(seg)
        return tuple(out)

    dropped = 0
    for _path, leaf in walk_all_leaves(values):
        refs = leaf.get("inputs")
        if not isinstance(refs, list):
            continue
        kept_refs: list[Any] = []
        for raw in refs:
            parsed = parse_path(raw) if isinstance(raw, str) else None
            if parsed is None:
                kept_refs.append(raw)  # malformed; counted downstream
                continue
            moved = remap(parsed)
            if moved is None:
                dropped += 1
            else:
                kept_refs.append(path_to_str(moved))
        if kept_refs:
            leaf["inputs"] = kept_refs
        else:
            del leaf["inputs"]
    return dropped


_DERIVATION_TOLERANCE = Decimal("0.05")


def _leaf_number(leaf: Any) -> Decimal | None:
    """A leaf's numeric reading — its normalized ``value`` if numeric, else its
    ``text`` with common money formatting stripped. ``None`` when not numeric."""
    if not isinstance(leaf, dict):
        return None
    for raw in (leaf.get("value"), leaf.get("text")):
        if raw is None:
            continue
        cleaned = str(raw).replace(",", "").replace("$", "").strip()
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            continue
    return None


def check_derivations(values: dict[str, Any]) -> tuple[int, int]:
    """Recompute checkable calculated leaves; return ``(checked, mismatched)``.

    Report-only quality signal (surfaced in ``extraction_stats.json``): for
    each ``method: "calculated"`` leaf whose ``inputs`` paths all resolve to
    leaves with numeric readings, the leaf is *checked*; it counts as
    *mismatched* when its own numeric value matches neither the SUM of the
    inputs, the COUNT of the inputs, nor any single input (a
    passthrough/min/max style derivation) within a $0.05 tolerance.
    ``inferred`` values are judgments, not calculations — a verifier checks
    only that their inputs exist — and non-numeric calculations are skipped,
    not counted.
    """
    checked = 0
    mismatched = 0
    for _path, leaf in walk_input_leaves(values):
        if leaf.get("method") != METHOD_CALCULATED or leaf.get("evidence"):
            # Evidence is printed text, not a number to add up.
            continue
        refs = leaf.get("inputs")
        if not isinstance(refs, list) or not refs:
            continue
        inputs: list[Decimal] = []
        resolvable = True
        for raw in refs:
            path = parse_path(raw) if isinstance(raw, str) else None
            source = get_at_path(values, path) if path is not None else None
            number = _leaf_number(source)
            if number is None:
                resolvable = False
                break
            inputs.append(number)
        if not resolvable:
            continue
        result = _leaf_number(leaf)
        if result is None:
            continue
        checked += 1
        candidates = [sum(inputs, Decimal(0)), Decimal(len(inputs)), *inputs]
        if not any(abs(result - c) <= _DERIVATION_TOLERANCE for c in candidates):
            mismatched += 1
    return checked, mismatched


def _resolve_collection(values: dict[str, Any], path: str) -> list[Any] | None:
    """Walk a dotted tag path to a collection, or ``None`` if it isn't one.

    Only dict hops are followed: an invariant naming a path that runs through a
    repeated entry is unresolvable rather than ambiguously "the first one".
    """
    node: Any = values
    for segment in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(segment)
    return node if isinstance(node, list) else None


def check_invariants(values: dict[str, Any], vocab: Vocabulary) -> tuple[int, list[str]]:
    """Evaluate the schema's ``## Invariant:`` annotations against *values*.

    Returns ``(checked, violations)`` where each violation is a human-readable
    one-liner naming the field, the expectation and what was found. Report-only
    by design (spec §13: extraction never adjusts values to force consistency)
    — the caller surfaces the counts in ``extraction_stats.json`` and
    ``dgml check``.

    An invariant whose field or collection is absent from *values* is **not**
    counted: extraction schemas are nullable throughout, so "not extracted" is
    a legal outcome and must not read as a violation. Only a field that has a
    value, with a resolvable collection to compare against, is checked.

    The two forms differ on an empty collection, deliberately. ``count`` is
    about cardinality, which is known as soon as the collection resolves, so
    an empty one is checked against zero. ``sum`` is about leaf values, so it
    is skipped unless at least one entry yielded a number — otherwise a total
    whose per-entry amounts simply weren't extracted would report as a
    violation.
    """
    checked = 0
    violations: list[str] = []

    def visit(tag: Tag, value: Any) -> None:
        nonlocal checked
        if value is None:
            return
        if tag.kind == "field":
            if tag.invariant and isinstance(value, dict):
                _check_one(tag, value)
            return
        if tag.kind == "collection":
            if not isinstance(value, list):
                return
            item = tag.item
            for entry in value:
                if item is not None and item.kind == "field":
                    visit(item, entry)
                elif isinstance(entry, dict):
                    for child in tag.children:
                        visit(child, entry.get(child.name))
            return
        if isinstance(value, dict):
            for child in tag.children:
                visit(child, value.get(child.name))

    def _check_one(tag: Tag, leaf: dict[str, Any]) -> None:
        nonlocal checked
        assert tag.invariant is not None
        parsed = parse_invariant(tag.invariant)
        if parsed is None:  # pragma: no cover - schema load rejects these
            return
        kind, path, leaf_name = parsed
        actual = _leaf_number(leaf)
        if actual is None:
            return  # field present but not numeric — nothing to compare
        # Invariants are evaluated against the whole submission, so a path is
        # resolved from the root regardless of where the annotated field sits;
        # an unqualified path is read under the schema's one root, as at load.
        collection = _resolve_collection(values, resolve_invariant_path(vocab, path) or path)
        if collection is None:
            return
        if kind == "count":
            checked += 1
            expected = Decimal(len(collection))
            if actual != expected:
                violations.append(f"{tag.name}: expected count({path}) = {expected}, got {actual}")
            return
        # sum(<collection>[].<leaf>)
        assert leaf_name is not None
        addends = [
            number
            for entry in collection
            if isinstance(entry, dict)
            for number in (_leaf_number(entry.get(leaf_name)),)
            if number is not None
        ]
        if not addends:
            return
        checked += 1
        expected = sum(addends, Decimal(0))
        if abs(actual - expected) > _DERIVATION_TOLERANCE:
            violations.append(
                f"{tag.name}: expected sum({path}[].{leaf_name}) = {expected}, got {actual}"
            )

    for root in vocab.roots:
        visit(root, values.get(root.name))
    return checked, violations


def count_unnormalized_enum_values(values: dict[str, Any], vocab: Vocabulary) -> int:
    """Number of enum-field leaves whose ``value`` is missing or not one of the
    schema's enum tokens. Those leaves serialize text-only (no ``dg:value``),
    so the misses are surfaced in ``extraction_stats.json`` instead of
    vanishing silently."""
    count = 0

    def visit(tag: Tag, value: Any) -> None:
        nonlocal count
        if value is None:
            return
        if tag.kind == "field":
            if tag.enum_values is not None and isinstance(value, dict):
                raw = value.get("value")
                token = str(raw).strip() if raw is not None else None
                if token not in tag.enum_values:
                    count += 1
            return
        if tag.kind == "collection":
            if not isinstance(value, list):
                return
            item = tag.item
            for entry in value:
                if item is not None and item.kind == "field":
                    visit(item, entry)
                elif isinstance(entry, dict):
                    for child in tag.children:
                        visit(child, entry.get(child.name))
            return
        # container / choice-with-children: descend by child name.
        if isinstance(value, dict):
            for child in tag.children:
                visit(child, value.get(child.name))

    for root in vocab.roots:
        visit(root, values.get(root.name))
    return count


def provenance_issues(xml: str | bytes) -> list[tuple[str, str, str]]:
    """Check every ``dg:value`` element's provenance against DGML 0.71.

    Returns ``(kind, local_name, detail)`` triples, used by the workspace
    consistency check:

    * ``value_unsupported`` — a 0.7 ``dg:origin="computed"`` element with no
      ``dg:href``: a value that claims to be derived and names nothing it
      used. (In 0.71 a bare value — no boxes, inputs or method — is
      indistinguishable from one grounding couldn't box, so the XML alone
      can't call it unsupported; extraction counts those itself, as
      ``unsupported_values`` in ``extraction_stats.json``.)
    * ``method_misplaced`` — ``dg:method`` on a boxed element, an unknown
      method, ``defaulted`` with inputs, or ``calculated``/``inferred``
      without them.
    * ``inputs_dangling`` — a same-file ``dg:inputs`` target with no element.
    * ``inputs_cycle`` — an element that is its own input, directly or
      through other inputs.

    A 0.7 ``dg:origin="computed"`` element is read as its 0.71 equivalent
    (its ``dg:href`` targets as inputs), so old files check cleanly.
    Cross-file (``fileid#id``) inputs can't be resolved against one file and
    are not checked.
    """
    root = etree.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    elements = [el for el in root.iter() if isinstance(el.tag, str)]
    by_id = {xid: el for el in elements if (xid := el.get(f"{{{XML_NS}}}id"))}
    out: list[tuple[str, str, str]] = []
    edges: dict[str, list[str]] = {}
    for el in elements:
        name = _local_name(el)
        tokens = effective_inputs(el)
        method = el.get(METHOD_ATTR)
        boxed = effective_origin(el) is not None
        if method is not None:
            if method not in METHODS:
                out.append(("method_misplaced", name, f"unknown dg:method {method!r}"))
            elif boxed:
                out.append(("method_misplaced", name, "dg:method on an element with dg:origin"))
            elif method == METHOD_DEFAULTED and tokens:
                out.append(("method_misplaced", name, "defaulted value with dg:inputs"))
            elif method != METHOD_DEFAULTED and not tokens:
                out.append(("method_misplaced", name, f"{method} value with no dg:inputs"))
        if is_legacy_computed(el) and not tokens:
            out.append(("value_unsupported", name, 'dg:origin="computed" with no dg:href'))
        local: list[str] = []
        for token in tokens:
            ref = parse_ref(token)
            if ref is None or not ref.local:
                continue
            if ref.xml_id not in by_id:
                out.append(("inputs_dangling", name, token))
            else:
                local.append(ref.xml_id)
        own_id = el.get(f"{{{XML_NS}}}id")
        if own_id and local:
            edges[own_id] = local
    for xid in _ids_on_cycles(edges):
        out.append(("inputs_cycle", _local_name(by_id[xid]), f"#{xid}"))
    return out


def _ids_on_cycles(edges: dict[str, list[str]]) -> list[str]:
    """The ids that reach themselves through ``dg:inputs`` edges, in document
    order of first discovery. Iterative DFS; graphs here are small."""
    on_cycle: list[str] = []
    seen: set[str] = set()
    for start in edges:
        stack = list(edges[start])
        visited: set[str] = set()
        while stack:
            node = stack.pop()
            if node == start:
                if start not in seen:
                    seen.add(start)
                    on_cycle.append(start)
                break
            if node in visited:
                continue
            visited.add(node)
            stack.extend(edges.get(node, ()))
    return on_cycle


def _maybe_set_id(el: etree._Element, path: LeafPath, ids: dict[LeafPath, str]) -> None:
    if path in ids:
        el.set(f"{{{XML_NS}}}id", ids[path])


def _resolved_inputs(value: dict[str, Any], ids: dict[LeafPath, str]) -> list[str]:
    """The ``#id`` targets of *value*'s ``inputs`` that resolve in this file."""
    refs = value.get("inputs")
    out: list[str] = []
    if isinstance(refs, list):
        for raw in refs:
            if not isinstance(raw, str):
                continue
            path = parse_path(raw)
            if path is not None and path in ids:
                out.append(f"#{ids[path]}")
    return out


def _add_evidence(parent: etree._Element, value: dict[str, Any], path: LeafPath) -> list[str]:
    """Emit each grounded ``evidence`` item of *value* as an anonymous
    ``<dg:chunk xml:id dg:origin>text</dg:chunk>`` under *parent* (so, ahead
    of the field element that uses it) and return their ``#id`` references.

    ``dg:chunk`` already means "this boundary is real, the name is not": the
    text is printed and a verifier can crop it, but the schema has no field
    for it. Ids are the field's path id plus ``--in-<n>``; a double hyphen
    never comes out of :func:`_id_for_path`, so they can't collide with a
    field's. Items without a box are dropped (and counted by
    :func:`count_dropped_inputs`)."""
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or value.get("method") == METHOD_DEFAULTED:
        return []
    refs: list[str] = []
    base = _id_for_path(path)
    for item in evidence:
        if not isinstance(item, dict):
            continue
        origin = _origin_from_locations(item.get("locations"))
        if origin is None:
            continue
        chunk_id = f"{base}--in-{len(refs)}"
        chunk = etree.SubElement(parent, CHUNK_TAG)
        chunk.set(f"{{{XML_NS}}}id", chunk_id)
        chunk.set(ORIGIN_ATTR, origin)
        chunk.text = str(item.get("text", ""))
        refs.append(f"#{chunk_id}")
    return refs


def _set_unprinted_attrs(
    el: etree._Element,
    value: dict[str, Any],
    value_type: str | None,
    ids: dict[LeafPath, str],
    enum_values: list[str] | None = None,
    evidence_refs: list[str] | None = None,
) -> None:
    """Emit the attribute set of a value the page doesn't print (DGML 0.71):
    a mandatory ``dg:value`` (the model's canonical ``value``, normalized),
    ``dg:inputs`` listing the resolvable inputs, and ``dg:method``. No
    ``dg:origin`` — there is nothing printed to box.

    ``calculated``/``inferred`` require inputs: when none resolve, the method
    is dropped rather than letting the claim stand. (Extraction drops such a
    value before it gets here — see :func:`drop_unboxed_values` — so this only
    guards direct callers.) ``defaulted`` never carries inputs."""

    text_str = el.text or ""
    raw_value = value.get("value")
    canonical = str(raw_value).strip() if raw_value is not None else None
    if enum_values is not None:
        # Enum fields carry the schema token as dg:value, no xsi:type (the
        # tokens aren't XSD types). An unnormalizable value falls back to the
        # display text so the mandatory dg:value is still present.
        xsi_type: str | None = None
        dg_value = canonical if canonical in enum_values else None
    elif value_type == "string":
        # Explicitly string-typed: keep dg:value (mandatory when unprinted) but
        # verbatim, with no xsi:type and no numeric coercion (see _add_field).
        xsi_type = None
        dg_value = canonical if canonical is not None else text_str.strip()
    elif value_type is not None:
        xsi_type = value_type
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

    method = value.get("method")
    inputs = (
        [] if method == METHOD_DEFAULTED else _resolved_inputs(value, ids) + (evidence_refs or [])
    )
    if inputs:
        el.set(INPUTS_ATTR, format_refs(inputs))
    if method == METHOD_DEFAULTED or (method in METHODS and inputs):
        el.set(METHOD_ATTR, method)


def _add_field(
    parent: etree._Element,
    docset_ns: str,
    tag_name: str,
    value: dict[str, Any],
    value_type: str | None = None,
    *,
    enum_values: list[str] | None = None,
    path: LeafPath = (),
    ids: dict[LeafPath, str] | None = None,
) -> None:
    ids = ids or {}
    evidence_refs = _add_evidence(parent, value, path)
    el = etree.SubElement(parent, f"{{{docset_ns}}}{tag_name}")
    _maybe_set_id(el, path, ids)
    text = value.get("text")
    text_str = "" if text is None else str(text)
    if text_str:
        el.text = text_str
    if is_unprinted_leaf(value):
        _set_unprinted_attrs(el, value, value_type, ids, enum_values, evidence_refs)
        return
    # Grounded leaf: the model-returned normalized `value` wins when it
    # validates (an enum token from the schema's closed set, or a value the
    # declared datatype accepts); otherwise fall back to normalizing the
    # verbatim text. An enum leaf with no valid token stays text-only —
    # counted upstream as unnormalized, never guessed.
    raw_value = value.get("value")
    canonical = str(raw_value).strip() if raw_value is not None else None
    xsi_type: str | None = None
    dg_value: str | None = None
    if enum_values is not None:
        if canonical in enum_values:
            dg_value = canonical
    elif value_type == "string":
        # An explicitly string-typed field (e.g. an account/invoice identifier):
        # keep the text verbatim with no xsi:type and no dg:value, suppressing the
        # numeric/date heuristic so all-digit ids aren't coerced to integers (which
        # would drop leading zeros and assert a numeric meaning the id doesn't have).
        pass
    elif text_str or canonical:
        if value_type is not None:
            if canonical:
                xsi_type, dg_value = _typed_value(canonical, value_type)
            if dg_value is None and text_str:
                xsi_type, dg_value = _typed_value(text_str, value_type)
        else:
            xsi_type, dg_value, _ = _detect_value_type(canonical or text_str, extra_formats=False)
            if canonical and dg_value is None:
                dg_value = canonical
    if dg_value is not None:
        if xsi_type:
            el.set(f"{{{XSI_NS}}}type", xsi_type)
        el.set(f"{{{DG_NS}}}value", dg_value)
    origin = _origin_from_locations(value.get("locations"))
    if origin:
        el.set(ORIGIN_ATTR, origin)
    # Printed in part: the boxes say where the printed text is, dg:inputs
    # what completed it. A boxed value never carries dg:method.
    inputs = _resolved_inputs(value, ids) + evidence_refs
    if inputs:
        el.set(INPUTS_ATTR, format_refs(inputs))


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
            _add_field(
                parent,
                docset_ns,
                tag.name,
                value,
                tag.value_type,
                enum_values=tag.enum_values,
                path=path,
                ids=ids,
            )
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
        item_enum = item.enum_values if item is not None else None
        for i, entry in enumerate(value):
            if not isinstance(entry, dict):
                continue
            item_path = path + (i,)
            if item_is_leaf:
                # list of grounded text values — each entry is a leaf field.
                _add_field(
                    el,
                    docset_ns,
                    item_name,
                    entry,
                    item_type,
                    enum_values=item_enum,
                    path=item_path,
                    ids=ids,
                )
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
    leaf names as an input; pass 2 emits the tree, stamping those ids on the
    input elements and ``dg:inputs`` on the leaves that use them."""
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
    (values, origins, inputs, xml:ids all preserved), replacing any
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
    """Per-parse collector for ``dg:inputs`` resolution.

    ``id_paths`` maps every ``xml:id`` seen to its values-tree path;
    ``chunks`` maps each evidence ``dg:chunk`` id to its ``{text,
    locations}``; ``with_inputs`` holds the leaves whose ``inputs`` still
    carry raw ``#id`` references. :meth:`resolve` rewrites those once the
    whole tree has been walked: a chunk reference moves to the leaf's
    ``evidence``, a field reference becomes its dotted path."""

    def __init__(self) -> None:
        self.id_paths: dict[str, LeafPath] = {}
        self.chunks: dict[str, dict[str, Any]] = {}
        self.with_inputs: list[dict[str, Any]] = []

    def record_chunks(self, container: etree._Element) -> None:
        """Collect the evidence chunks: ``dg:chunk``\\s some ``dg:inputs``
        under *container* names. Structural chunks of a document tree are
        not evidence and stay where they are."""
        referenced = {
            ref.xml_id
            for el in container.iter()
            if isinstance(el.tag, str)
            for token in effective_inputs(el)
            if (ref := parse_ref(token)) is not None and ref.local
        }
        for chunk in container.iter(CHUNK_TAG):
            xml_id = chunk.get(f"{{{XML_NS}}}id")
            if chunk is container or xml_id not in referenced:
                continue
            self.chunks[xml_id] = {
                "text": (chunk.text or "").strip(),
                "locations": _locations_from_origin(chunk.get(ORIGIN_ATTR)),
            }

    def record(self, el: etree._Element, path: LeafPath) -> None:
        xml_id = el.get(f"{{{XML_NS}}}id")
        if xml_id:
            self.id_paths[xml_id] = path

    def resolve(self) -> None:
        for leaf in self.with_inputs:
            inputs: list[str] = []
            evidence: list[dict[str, Any]] = []
            for raw in leaf["inputs"]:
                local = raw[1:] if raw.startswith("#") else None
                if local is not None and local in self.chunks:
                    evidence.append(dict(self.chunks[local]))
                elif local is not None and local in self.id_paths:
                    inputs.append(path_to_str(self.id_paths[local]))
                else:
                    inputs.append(raw)
            if inputs:
                leaf["inputs"] = inputs
            else:
                del leaf["inputs"]
            if evidence:
                leaf["evidence"] = evidence


def _leaf_value(
    el: etree._Element, path: LeafPath = (), acc: _ParseAcc | None = None
) -> dict[str, Any]:
    if acc is not None:
        acc.record(el, path)
    leaf: dict[str, Any] = {"text": (el.text or "").strip()}
    dg_value = el.get(f"{{{DG_NS}}}value")
    if dg_value is not None:
        leaf["value"] = dg_value
    # Read through dg_refs so a 0.7 dg:origin="computed" element comes back
    # in its 0.71 form: no boxes, its dg:href targets as inputs.
    origin = effective_origin(el)
    inputs = effective_inputs(el)
    method = effective_method(el)
    # A 0.7 computed element with no sources has no boxes either: it reads
    # as a bare value (no locations), not as a printed one.
    if origin is not None or (method is None and not is_legacy_computed(el)):
        leaf["locations"] = _locations_from_origin(origin)
    if inputs:
        leaf["inputs"] = inputs
        if acc is not None:
            acc.with_inputs.append(leaf)
    if method is not None:
        leaf["method"] = method
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
        items = [c for c in _value_children(el, acc) if _local_name(c) == item_name]
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
    by_name = {_local_name(c): c for c in _value_children(el, acc)}
    out: dict[str, Any] = {}
    for child in children:
        match = by_name.get(child.name)
        if match is not None:
            out[child.name] = _element_to_value_vocab(match, child, path + (child.name,), acc)
    return out


def _value_children(el: etree._Element, acc: _ParseAcc | None) -> list[etree._Element]:
    """*el*'s element children, minus the evidence ``dg:chunk``\\s, which
    belong to the field that names them in ``dg:inputs``, not the tree."""
    chunks = acc.chunks if acc is not None else {}
    return [
        c
        for c in el
        if isinstance(c.tag, str)
        and not (c.tag == CHUNK_TAG and c.get(f"{{{XML_NS}}}id") in chunks)
    ]


def _element_to_value(el: etree._Element, path: LeafPath = (), acc: _ParseAcc | None = None) -> Any:
    children = _value_children(el, acc)
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

    ``{tag: {text, value?, locations, inputs?}}`` for printed leaves (in full,
    or in part with ``inputs``), ``{text, value?, inputs?, method}`` for
    unprinted ones (``dg:inputs`` mapped back to dotted paths where they
    resolve in-file; cross-file or unknown targets stay raw; a 0.7
    ``dg:origin="computed"`` element reads as its 0.71 equivalent),
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
    acc.record_chunks(container)
    if vocab is not None:
        by_name = {t.name: t for t in vocab.roots}
        for child in _value_children(container, acc):
            name = _local_name(child)
            tag = by_name.get(name)
            out[name] = (
                _element_to_value_vocab(child, tag, (name,), acc)
                if tag
                else _element_to_value(child, (name,), acc)
            )
        acc.resolve()
        return out
    for child in _value_children(container, acc):
        name = _local_name(child)
        out[name] = _element_to_value(child, (name,), acc)
    acc.resolve()
    return out
