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
"""read_dgml (dgml_output) / get_extracted_values (extraction): the library reads behind
``dgml extraction get-values``.

Seeds the pair's records and core DGML file directly (no PDF pipeline).
"""

from __future__ import annotations

from typing import Any

import pytest
from dgml_core import layout
from dgml_core.dgml_output import read_dgml
from dgml_core.docsets import DocSetStore
from dgml_core.errors import (
    DgmlNotFound,
    DocSetNotFound,
    FileNotFound,
    InvalidArgument,
    ValuesNotFound,
)
from dgml_core.extraction import get_extracted_values
from dgml_core.extraction_schema import json_schema_to_rnc, parse_rnc
from dgml_core.extraction_xml import standalone_extraction_doc
from dgml_core.storage import Workspace

_SCHEMA = {
    "definitions": {"grounded_field": {"type": "object"}},
    "properties": {
        "vendor_name": {"$ref": "#/definitions/grounded_field"},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"product_name": {"$ref": "#/definitions/grounded_field"}},
            },
        },
    },
}
_RNC = json_schema_to_rnc(_SCHEMA, workspace="acme", docset_name="MSA")

_VALUES: dict[str, Any] = {
    "VendorName": {
        "text": "Acme",
        "locations": [{"page_number": 1, "bounding_box": [10, 20, 30, 40]}],
    },
    "LineItems": [{"ProductName": {"text": "Widget", "locations": []}}],
}

_TREE_ONLY = (
    b'<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" xmlns:docset="http://dgml.io/ns/dgml/test">'
    b"<docset:Header>Master Services Agreement</docset:Header></dg:chunk>"
)


def _seed(ws: Workspace, *, assign: bool = True, xml: bytes | None = None) -> None:
    """File ``f001`` (``msa.pdf``) and docset ``ds01``, optionally assigned and
    with a core DGML file."""
    ws.docs.put_doc(
        layout.Collection.FILES,
        "f001",
        {
            "id": "f001",
            "original_path": "/src/msa.pdf",
            "original_filename": "msa.pdf",
            "sha256": "0" * 64,
            "added_at": "2026-06-11T00:00:00Z",
            "page_count": 1,
            "text_mode": "digital",
        },
    )
    ws.docs.put_doc(
        layout.Collection.DOCSETS,
        "ds01",
        {"id": "ds01", "name": "MSA", "description": "", "key_questions": []},
    )
    if assign:
        DocSetStore(ws).add_file("ds01", "f001")
    if xml is not None:
        ws.blobs.put_blob(layout.dgml_xml_key("ds01", "f001", "msa"), xml)


def _extraction_xml() -> bytes:
    return standalone_extraction_doc(_VALUES, vocab=parse_rnc(_RNC)).encode("utf-8")


def test_read_dgml_returns_stored_bytes_and_canonical_key(workspace: Workspace) -> None:
    xml = _extraction_xml()
    _seed(workspace, xml=xml)
    doc = read_dgml(workspace, "ds01", "f001")
    assert doc.key == "docsets/ds01/files/f001/msa.dgml.xml"
    assert doc.xml == xml


def test_read_dgml_returns_a_tree_without_extraction(workspace: Workspace) -> None:
    """The default read is the full DGML — a generated tree needs no extraction."""
    _seed(workspace, xml=_TREE_ONLY)
    assert read_dgml(workspace, "ds01", "f001").xml == _TREE_ONLY
    with pytest.raises(ValuesNotFound, match="run 'dgml extraction extract' first"):
        get_extracted_values(workspace, "ds01", "f001")


def test_get_values_uses_schema_vocabulary(workspace: Workspace) -> None:
    _seed(workspace, xml=_extraction_xml())
    DocSetStore(workspace).set_schema("ds01", _RNC)
    values = get_extracted_values(workspace, "ds01", "f001")
    assert values["VendorName"]["text"] == "Acme"
    assert values["VendorName"]["locations"][0]["bounding_box"] == [10, 20, 30, 40]
    # The schema says LineItems is a collection, so one item stays a list.
    assert values["LineItems"] == [{"ProductName": {"text": "Widget", "locations": []}}]


def test_get_values_without_schema_infers_structure(workspace: Workspace) -> None:
    _seed(workspace, xml=_extraction_xml())
    values = get_extracted_values(workspace, "ds01", "f001")
    assert values["VendorName"]["text"] == "Acme"


def test_unknown_docset(workspace: Workspace) -> None:
    _seed(workspace, xml=_extraction_xml())
    with pytest.raises(DocSetNotFound):
        read_dgml(workspace, "nope", "f001")
    with pytest.raises(DocSetNotFound):
        get_extracted_values(workspace, "nope", "f001")


def test_unknown_file(workspace: Workspace) -> None:
    _seed(workspace)
    with pytest.raises(FileNotFound, match="'nope' not found"):
        read_dgml(workspace, "ds01", "nope")


def test_unassigned_file(workspace: Workspace) -> None:
    _seed(workspace, assign=False, xml=_extraction_xml())
    with pytest.raises(FileNotFound, match="not assigned"):
        read_dgml(workspace, "ds01", "f001")
    with pytest.raises(FileNotFound, match="not assigned"):
        get_extracted_values(workspace, "ds01", "f001")


def test_assigned_without_dgml(workspace: Workspace) -> None:
    """No DGML at all is DgmlNotFound from both reads; VALUES_NOT_FOUND is only
    for a DGML that exists without a dg:extraction element."""
    _seed(workspace)
    with pytest.raises(DgmlNotFound):
        read_dgml(workspace, "ds01", "f001")
    with pytest.raises(DgmlNotFound):
        get_extracted_values(workspace, "ds01", "f001")


@pytest.mark.parametrize(("docset_id", "file_id"), [("", "f001"), ("ds01", " ")])
def test_empty_ids(workspace: Workspace, docset_id: str, file_id: str) -> None:
    _seed(workspace)  # the docset exists, so a blank file id reaches its own check
    with pytest.raises(InvalidArgument):
        read_dgml(workspace, docset_id, file_id)
