"""Tests for schema.json <-> schema.rnc (dgml_core.generation.rnc).

All schema/XML content here is SYNTHETIC — invented tags and values only.
"""

from __future__ import annotations

import json
from pathlib import Path

from dgml_core.generation.rnc import build_rnc, rnc_to_schema_dict, write_docset_rnc

_SCHEMA = {
    "tags": {
        "Invoice": {
            "name": "Invoice",
            "role": "One invoice document",
            "kind": "section",
            "example": "",
            "examples": [],
            "parent_role": "",
        },
        "InvoiceNumber": {
            "name": "InvoiceNumber",
            "role": 'Unique "identifier" of the invoice',  # quotes must round-trip
            "kind": "inline",
            "example": "INV-001",
            "examples": ["INV-001", "INV-002"],
            "parent_role": "Invoice",
        },
    },
    "notes": "synthetic test schema",
}

_XML = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"'
    ' xmlns:docset="http://dgml.io/test/SyntheticNs"'
    ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    '<docset:Invoice dg:structure="section">'
    '<docset:InvoiceNumber xsi:type="integer" dg:value="1001">1001</docset:InvoiceNumber>'
    "<docset:CoinedTag>free text</docset:CoinedTag>"
    "</docset:Invoice>"
    "</dg:chunk>"
)


def test_rnc_round_trips_schema_json(tmp_path: Path) -> None:
    """json -> rnc -> json reconstructs the exact Schema v1 dict."""
    xml = tmp_path / "doc.dgml.xml"
    xml.write_text(_XML, encoding="utf-8")
    rnc = build_rnc(_SCHEMA, [xml], label="synthetic")
    assert rnc_to_schema_dict(rnc) == _SCHEMA


def test_build_rnc_pins_observed_types_and_shapes(tmp_path: Path) -> None:
    xml = tmp_path / "doc.dgml.xml"
    xml.write_text(_XML, encoding="utf-8")
    rnc = build_rnc(_SCHEMA, [xml], label="synthetic")
    # namespace picked up from the scanned XML
    assert 'default namespace docset = "http://dgml.io/test/SyntheticNs"' in rnc
    # all typed occurrences agree -> dg:value datatype pinned behind xsi:type
    assert 'attribute xsi:type { "integer" }' in rnc
    assert "attribute dg:value { xsd:integer }" in rnc
    # container renders mixed; leaf with children observed nowhere stays text
    assert "Invoice = element Invoice {" in rnc
    # a concept observed but absent from the schema (coined during labeling) is
    # reported — and it lives in docset:, never dg:
    assert "CoinedTag" in rnc
    # the catch-all for undefined concepts is docset:*, and nothing semantic is
    # ever emitted in the framework dg: namespace
    assert "element docset:* {" in rnc
    assert "element dg:* {" not in rnc


def test_rnc_reverse_defaults_without_comments() -> None:
    """A define with no comment block reconstructs as SchemaTag defaults."""
    rnc = "SomeTag = element SomeTag {\n  common.atts,\n  text\n}\n"
    data = rnc_to_schema_dict(rnc)
    assert data["tags"]["SomeTag"] == {
        "name": "SomeTag",
        "role": "",
        "kind": "inline",
        "example": "",
        "examples": [],
        "parent_role": "",
    }
    assert data["notes"] == ""


def test_write_docset_rnc(tmp_path: Path) -> None:
    """Writes <docset>/full-schema.rnc from schema.json + XML; None without one."""
    assert write_docset_rnc(tmp_path) is None  # no schema.json yet

    (tmp_path / "schema.json").write_text(json.dumps(_SCHEMA), encoding="utf-8")
    (tmp_path / "docset.json").write_text(json.dumps({"name": "synthetic"}), encoding="utf-8")
    xml_dir = tmp_path / "files" / "f1"
    xml_dir.mkdir(parents=True)
    (xml_dir / "doc.dgml.xml").write_text(_XML, encoding="utf-8")

    out = write_docset_rnc(tmp_path)
    assert out == tmp_path / "full-schema.rnc" and out.exists()
    assert rnc_to_schema_dict(out.read_text(encoding="utf-8")) == _SCHEMA
