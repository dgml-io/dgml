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

"""Tests for schema.json <-> schema.rnc (dgml_core.generation.rnc).

All schema/XML content here is SYNTHETIC — invented tags and values only.
"""

from __future__ import annotations

import json
from pathlib import Path

from dgml_core import layout
from dgml_core.generation.rnc import build_rnc, rnc_to_schema_dict, write_docset_rnc
from dgml_core.storage import Workspace

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


def test_write_docset_rnc(workspace: Workspace) -> None:
    """Writes the docset's full-schema.rnc from schema.json + XML; None without one."""
    ws = workspace
    did = "d1"
    assert write_docset_rnc(ws, did) is None  # no schema.json yet

    # The generation schema is a blob (exact Schema.save bytes), not a document.
    ws.blobs.put_blob(layout.docset_generation_schema_key(did), json.dumps(_SCHEMA).encode("utf-8"))
    ws.docs.put_doc("docsets", did, {"id": did, "name": "synthetic"})
    ws.blobs.put_blob("docsets/d1/files/f1/doc.dgml.xml", _XML.encode("utf-8"))

    key = write_docset_rnc(ws, did)
    assert key == "docsets/d1/full-schema.rnc"
    assert rnc_to_schema_dict(ws.blobs.get_blob(key).decode("utf-8")) == _SCHEMA
