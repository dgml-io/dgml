"""schema.json <-> full-schema.rnc (RELAX NG Compact) — both directions.

The forward direction renders a docset's ``schema.json`` (Schema v1) as a
RELAX NG Compact grammar, enriched with what the generated ``*.dgml.xml``
files actually contain:

- observed data types (``xsi:type`` + normalized ``dg:value``) — a tag whose
  typed occurrences all agree gets its ``dg:value`` datatype pinned;
- leaf vs container shape (element children observed / schema kind);
- ``dg:structure`` roles and occurrence counts (informative comments).

Every ``schema.json`` field (``role``, ``kind``, ``parent_role``,
``example``/``examples``, top-level ``notes``) is
serialized LOSSLESSLY into ``# Field: value`` comment lines (values
JSON-encoded), so :func:`rnc_to_schema_dict` can reconstruct the exact
Schema v1 dict from the ``.rnc`` — the RNC doubles as a human-editing
surface for the schema.

Round-trip guarantee: json -> rnc -> json is semantically identical (same
tags, same field values; tag order is normalized to sorted).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

DG_NS = "http://dgml.io/ns/dg#"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

# xsi:type (as emitted by _detect_value_type) -> XSD datatype for dg:value
_XSD_FOR_XSI = {
    "date": "xsd:date",
    "time": "xsd:time",
    "gYear": "xsd:gYear",
    "boolean": "xsd:boolean",
    "integer": "xsd:integer",
    "decimal": "xsd:decimal",
    "anyURI": "xsd:anyURI",
}

# The lossless comment contract: `# <Field>: <value>` lines above each define.
_FIELD_RE = re.compile(r"^# (Description|Kind|Parent|Example|Examples|Notes): (.*)$")
_DEFINE_RE = re.compile(r"^([A-Za-z][\w.]*)\s*=\s*element\s+([A-Za-z][\w:.*-]*)\s*\{")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class _Observed:
    """Per-tag facts harvested from the generated DGML XML files."""

    def __init__(self) -> None:
        self.xsi_types: dict[str, Counter[str]] = defaultdict(Counter)
        self.structures: dict[str, Counter[str]] = defaultdict(Counter)
        self.has_children: dict[str, bool] = defaultdict(bool)
        self.occurrences: Counter[str] = Counter()
        self.docset_ns = ""


def _scan_dgml(xml_paths: Sequence[Path]) -> _Observed:
    obs = _Observed()
    for path in xml_paths:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue  # a broken file must not block the schema render
        for el in root.iter():
            ns = el.tag.rsplit("}", 1)[0].lstrip("{") if "}" in el.tag else ""
            if "dgml.io" in ns and ns != DG_NS:
                obs.docset_ns = obs.docset_ns or ns
            # Every concept lives in the docset: namespace. dg: is framework-only
            # (the dg:chunk scaffolding element + dg:* attributes), so a concept
            # is simply any element in the docset namespace.
            is_concept = ns == obs.docset_ns and bool(obs.docset_ns)
            if not is_concept:
                continue
            name = _local(el.tag)
            obs.occurrences[name] += 1
            t = el.get(f"{{{XSI_NS}}}type")
            if t:
                obs.xsi_types[name][t] += 1
            s = el.get(f"{{{DG_NS}}}structure")
            if s:
                obs.structures[name][s] += 1
            if len(el):
                obs.has_children[name] = True
    return obs


def build_rnc(
    schema_data: Mapping[str, Any],
    xml_paths: Sequence[Path] = (),
    label: str = "",
) -> str:
    """Render a Schema v1 dict as RELAX NG Compact text.

    *xml_paths* (the docset's generated ``*.dgml.xml`` files) supply observed
    data types and shapes; without them the grammar is emitted from the schema
    alone (all-text leaves, placeholder namespace).
    """
    tags: Mapping[str, Mapping[str, Any]] = schema_data.get("tags", {})
    obs = _scan_dgml(xml_paths)
    names = sorted(tags)
    children_of: dict[str, list[str]] = defaultdict(list)
    for n in names:
        parent = str(tags[n].get("parent_role", ""))
        if parent in tags and parent != n:
            children_of[parent].append(n)

    lines: list[str] = []
    w = lines.append
    w("# " + "=" * 74)
    w(f"# DGML docset schema — {label}" if label else "# DGML docset schema")
    w(f"# RELAX NG Compact.  Data types observed in {len(xml_paths)} .dgml.xml file(s):")
    w("#   element text is the RAW page text; the normalized, typed value is")
    w("#   carried by @dg:value, described by @xsi:type (XSD built-in names).")
    w("# `# Field: value` comments are the LOSSLESS schema.json serialization;")
    w("#   regenerate the JSON via dgml_core.generation.rnc.rnc_to_schema_dict.")
    w("# " + "=" * 74)
    w("")
    if schema_data.get("notes"):
        w(f"# Notes: {json.dumps(schema_data['notes'], ensure_ascii=False)}")
        w("")
    if obs.docset_ns:
        w(f'default namespace docset = "{obs.docset_ns}"')
    else:
        w('default namespace docset = "http://dgml.io/UNKNOWN"  # no XML scanned')
    w(f'namespace dg = "{DG_NS}"')
    w(f'namespace xsi = "{XSI_NS}"')
    w('datatypes xsd = "http://www.w3.org/2001/XMLSchema-datatypes"')
    w("")
    w("start = dg.chunk")
    w("")
    w("# --- attributes common to every DGML element ------------------------")
    w('# dg:origin  = grounding bbox: "page x1 y1 x2 y2"')
    w("# dg:itemprop / dg:href = semantic link (predicate + '#id' target(s))")
    w("common.atts =")
    w("  attribute dg:structure { text }?,")
    w("  attribute dg:origin { text }?,")
    w("  attribute xml:id { xsd:ID }?,")
    w("  attribute dg:itemprop { text }?,")
    w("  attribute dg:href { text }?,")
    w("  attribute dg:value { text }?,")
    w("  attribute xsi:type { text }?")
    w("")
    w("# --- generic structural chunk (unlabeled scaffolding) ----------------")
    w("dg.chunk = element dg:chunk { common.atts, mixed { any.docset* } }")
    w("")
    w("# a docset concept with no individual definition below — a rare/one-off")
    w("# tag, or one coined during labeling that isn't in schema.json yet. It")
    w("# stays in the docset vocabulary; nothing semantic is ever emitted in dg:.")
    w("docset.other = element docset:* { common.atts, mixed { any.docset* } }")
    w("")
    w("# any docset element may appear where structure allows it")
    w("any.docset = dg.chunk")
    w("  | docset.other")
    for n in names:
        w(f"  | {n}")
    w("")

    for n in names:
        t = tags[n]
        kind = str(t.get("kind", "inline"))
        role = str(t.get("role") or "")
        parent = str(t.get("parent_role", ""))
        kids = children_of.get(n, [])
        obs_n = obs.occurrences.get(n, 0)

        # Lossless schema.json fields — one `# Field: value` line each, values
        # JSON-encoded. rnc_to_schema_dict parses exactly these keys; omitted
        # lines reconstruct as the SchemaTag defaults.
        w("# " + "-" * 66)
        if role:
            w(f"# Description: {json.dumps(role, ensure_ascii=False)}")
        w(f"# Kind: {kind}")
        if parent:
            w(f"# Parent: {parent}")
        if t.get("example"):
            w(f"# Example: {json.dumps(t['example'], ensure_ascii=False)}")
        if t.get("examples"):
            w(f"# Examples: {json.dumps(t['examples'], ensure_ascii=False)}")
        if obs_n:
            st = ", ".join(f"{k}({v})" for k, v in obs.structures[n].most_common(3))
            w(f"# Observed: {obs_n} occurrence(s)" + (f"; dg:structure: {st}" if st else ""))

        ts = obs.xsi_types.get(n)
        typed: str | None = None
        if ts:
            mix = ", ".join(f"{k}({v})" for k, v in ts.most_common())
            w(f"# Data type (from DGML): {mix} of {obs_n} occurrence(s)")
            if len(ts) == 1:
                # Every typed occurrence agrees — pin the datatype, but only
                # when xsi:type is present: on a semlink (dg:itemprop) dg:value
                # instead carries an ISO-8601 offset or multiplier (e.g. "P7D").
                best = next(iter(ts))
                xsd = _XSD_FOR_XSI.get(best, "text")
                typed = (
                    f'  ((attribute xsi:type {{ "{best}" }},\n'
                    f"    attribute dg:value {{ {xsd} }})\n"
                    f"   | attribute dg:value {{ text }})?,  # bare dg:value = link payload"
                )
            else:
                w("#   (mixed types observed -> datatype left unpinned)")

        if typed:
            # typed leaves still carry the layout/grounding attributes
            atts = (
                "  attribute dg:structure { text }?,\n"
                "  attribute dg:origin { text }?,\n"
                "  attribute xml:id { xsd:ID }?,\n"
                "  attribute dg:itemprop { text }?,\n"
                "  attribute dg:href { text }?,\n" + typed
            )
        else:
            atts = "  common.atts,"

        if obs.has_children.get(n) or kind in ("section", "row") or kids:
            if kids:
                w(f"# Children (from schema hierarchy): {', '.join(kids)}")
            content = "mixed { any.docset* }"
        else:
            content = "text"
        w(f"{n} = element {n} {{")
        w(atts)
        w(f"  {content}")
        w("}")
        w("")

    unplanned = sorted(set(obs.occurrences) - set(tags))
    if unplanned:
        w("# --- tags observed in the DGML but absent from schema.json -----------")
        w("# (coined during labeling; consider promoting them into the schema)")
        for n in unplanned:
            w(f"#   {n}  ({obs.occurrences[n]} occurrence(s))")
        w("")

    return "\n".join(lines) + "\n"


def rnc_to_schema_dict(text: str) -> dict[str, Any]:
    """Reconstruct the Schema v1 dict from ``.rnc`` text written by build_rnc."""
    notes = ""
    tags: dict[str, dict[str, Any]] = {}
    pending: dict[str, str] = {}
    for line in text.splitlines():
        fm = _FIELD_RE.match(line)
        if fm:
            key, val = fm.group(1), fm.group(2)
            if key == "Notes":
                notes = str(json.loads(val))
            else:
                pending[key] = val
            continue
        dm = _DEFINE_RE.match(line)
        if not dm:
            continue
        elname = dm.group(2)
        if ":" in elname or "*" in elname:  # dg:chunk / dg:* scaffolding, not a concept
            pending = {}
            continue
        tags[elname] = {
            "name": elname,
            "role": str(json.loads(pending["Description"])) if "Description" in pending else "",
            "kind": pending.get("Kind", "inline"),
            "example": str(json.loads(pending["Example"])) if "Example" in pending else "",
            "examples": list(json.loads(pending["Examples"])) if "Examples" in pending else [],
            "parent_role": pending.get("Parent", ""),
        }
        pending = {}
    return {"tags": tags, "notes": notes}


def write_docset_rnc(docset_dir: Path) -> Path | None:
    """Write ``<docset_dir>/full-schema.rnc`` from its schema.json + generated XML.

    Returns the written path, or ``None`` when the docset has no schema.json
    yet (nothing to render). Called at the end of ``docset generate`` so the
    RNC always reflects the final grounded, linked XML. The filename matches
    :meth:`dgml_core.storage.Workspace.docset_full_schema_path` — this is the
    artifact DGMLX bundles ship and attest (superseding schema.json there).
    """
    schema_path = docset_dir / "schema.json"
    if not schema_path.exists():
        return None
    schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
    xml_paths = sorted(docset_dir.glob("files/*/*.dgml.xml"))
    docset_json = docset_dir / "docset.json"
    label = (
        str(json.loads(docset_json.read_text(encoding="utf-8")).get("name", docset_dir.name))
        if docset_json.exists()
        else docset_dir.name
    )
    out = docset_dir / "full-schema.rnc"
    rendered = build_rnc(schema_data, xml_paths, label=label)
    out.write_text(rendered, encoding="utf-8")
    return out
