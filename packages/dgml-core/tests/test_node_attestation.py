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

"""Element-level attestation: export_node / prove_node and the xpath helpers.

Seeds the generated DGML XML directly on disk (no PDF pipeline) — node
attestation hashes the XML artifact, not document semantics.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from dgml_core.errors import DocSetNotFound, FileNotFound, InvalidArgument, NotFoundError
from dgml_core.merkle import merkle_root, proof_from_json, proof_to_json
from dgml_core.node_attestation import (
    element_xpath,
    export_node,
    load_dgml_root,
    ordered_elements,
    prove_node,
    resolve_child_path,
    resolve_xpath,
)
from dgml_core.storage import Workspace, write_json_atomic
from lxml import etree  # type: ignore[import-untyped]

DGML_XML = b"""\
<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" \
xmlns:docset="http://dgml.io/ns/dgml/test">
  <docset:Header structure="h1">Resident Ledger</docset:Header>
  <docset:Entries>
    <docset:Entry>
      <docset:Date>2026-01-01</docset:Date>
      <docset:Amount>100.00</docset:Amount>
    </docset:Entry>
    <docset:Entry>
      <docset:Date>2026-02-01</docset:Date>
      <docset:Amount>250.00</docset:Amount>
    </docset:Entry>
  </docset:Entries>
</dg:chunk>
"""


def _seed(ws: Workspace, file_id: str = "f001", docset_id: str = "ds01") -> None:
    file_dir = ws.file_dir(file_id)
    file_dir.mkdir(parents=True)
    write_json_atomic(
        ws.file_json_path(file_id),
        {
            "id": file_id,
            "original_path": "/src/ledger.pdf",
            "original_filename": "ledger.pdf",
            "sha256": "0" * 64,
            "added_at": "2026-06-11T00:00:00Z",
            "page_count": 1,
            "text_mode": "digital",
        },
    )
    ws.docset_dir(docset_id).mkdir(parents=True)
    write_json_atomic(
        ws.docset_dir(docset_id) / "docset.json",
        {"id": docset_id, "name": "Test", "description": "", "key_questions": []},
    )
    xml_path = ws.file_dgml_xml_path(docset_id, file_id, "ledger")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_bytes(DGML_XML)


# --- export -------------------------------------------------------------------


def test_export_by_xpath_and_by_leaf_agree(workspace: Workspace) -> None:
    _seed(workspace)
    by_xpath = export_node(
        workspace, "f001", "ds01", xpath="/dg:chunk/docset:Entries/docset:Entry[2]"
    )
    by_leaf = export_node(workspace, "f001", "ds01", leaf_index=by_xpath.leaf_index)
    assert by_leaf == by_xpath
    assert by_xpath.xpath == "/dg:chunk/docset:Entries/docset:Entry[2]"


def test_export_node_hash_is_sha256_of_node_xml(workspace: Workspace) -> None:
    _seed(workspace)
    att = export_node(workspace, "f001", "ds01", xpath="/dg:chunk/docset:Header")
    assert hashlib.sha256(att.node_xml.encode()).hexdigest() == att.node_hash
    assert att.proof.leaf_hash == att.node_hash
    assert "Resident Ledger" in att.node_xml


def test_export_root_matches_whole_tree_merkle_root(workspace: Workspace) -> None:
    _seed(workspace)
    att = export_node(workspace, "f001", "ds01", leaf_index=0)
    assert att.root_hash == merkle_root(etree.fromstring(DGML_XML))
    # leaf 0 is the document element itself.
    assert att.xpath == "/dg:chunk"
    assert att.leaf_count == len(ordered_elements(etree.fromstring(DGML_XML)))


def test_export_by_child_path_agrees_with_xpath_and_leaf(workspace: Workspace) -> None:
    _seed(workspace)
    by_xpath = export_node(
        workspace, "f001", "ds01", xpath="/dg:chunk/docset:Entries/docset:Entry[2]"
    )
    # root -> Entries (2nd child) -> Entry (2nd child).
    by_child_path = export_node(workspace, "f001", "ds01", child_path=[1, 1])
    assert by_child_path == by_xpath


def test_export_by_child_path_empty_selects_root(workspace: Workspace) -> None:
    _seed(workspace)
    att = export_node(workspace, "f001", "ds01", child_path=[])
    assert att.xpath == "/dg:chunk"
    assert att.leaf_index == 0


def test_export_requires_exactly_one_selector(workspace: Workspace) -> None:
    _seed(workspace)
    with pytest.raises(InvalidArgument, match="exactly one"):
        export_node(workspace, "f001", "ds01")
    with pytest.raises(InvalidArgument, match="exactly one"):
        export_node(workspace, "f001", "ds01", leaf_index=1, xpath="/dg:chunk")
    with pytest.raises(InvalidArgument, match="exactly one"):
        export_node(workspace, "f001", "ds01", leaf_index=1, child_path=[0])


def test_export_leaf_out_of_range(workspace: Workspace) -> None:
    _seed(workspace)
    with pytest.raises(InvalidArgument, match="out of range"):
        export_node(workspace, "f001", "ds01", leaf_index=99)


def test_export_ambiguous_xpath_rejected(workspace: Workspace) -> None:
    _seed(workspace)
    with pytest.raises(InvalidArgument, match="matched 2 elements"):
        export_node(workspace, "f001", "ds01", xpath="//docset:Entry")


def test_export_missing_file_docset_or_xml(workspace: Workspace) -> None:
    with pytest.raises(FileNotFound):
        export_node(workspace, "nope", "ds01", leaf_index=0)
    _seed(workspace)
    with pytest.raises(DocSetNotFound):
        export_node(workspace, "f001", "nope", leaf_index=0)
    # Docset exists but the pair has no generated XML.
    workspace.docset_dir("ds02").mkdir(parents=True)
    with pytest.raises(NotFoundError, match="no generated DGML XML"):
        export_node(workspace, "f001", "ds02", leaf_index=0)


# --- xpath helpers ------------------------------------------------------------


def test_element_xpath_round_trips_through_resolve(workspace: Workspace) -> None:
    _seed(workspace)
    root = load_dgml_root(workspace, "f001", "ds01")
    for el in ordered_elements(root):
        assert resolve_xpath(root, element_xpath(el)) is el


def test_resolve_child_path_matches_element_xpath(workspace: Workspace) -> None:
    _seed(workspace)
    root = load_dgml_root(workspace, "f001", "ds01")
    entry_2 = resolve_child_path(root, [1, 1])
    assert element_xpath(entry_2) == "/dg:chunk/docset:Entries/docset:Entry[2]"
    assert resolve_child_path(root, []) is root


def test_resolve_child_path_out_of_range(workspace: Workspace) -> None:
    _seed(workspace)
    root = load_dgml_root(workspace, "f001", "ds01")
    with pytest.raises(InvalidArgument, match="out of range"):
        resolve_child_path(root, [99])
    with pytest.raises(InvalidArgument, match="out of range"):
        resolve_child_path(root, [1, 99])


def test_element_xpath_handles_default_namespace() -> None:
    root = etree.fromstring(b'<doc xmlns="http://d"><item/><item/></doc>')
    second = root[1]
    xpath = element_xpath(second)
    assert xpath == "/*[local-name()='doc']/*[local-name()='item'][2]"
    assert resolve_xpath(root, xpath) is second


# --- prove --------------------------------------------------------------------


def test_prove_round_trip_via_json(workspace: Workspace) -> None:
    """Export → serialize the proof → revive → prove. The full off-band loop."""
    _seed(workspace)
    att = export_node(workspace, "f001", "ds01", xpath="/dg:chunk/docset:Entries")

    revived = proof_from_json(json.loads(json.dumps(proof_to_json(att.proof))))
    result = prove_node(workspace, "f001", "ds01", att.root_hash, revived)
    assert result.valid is True
    assert result.computed_node_hash == att.node_hash
    assert result.xpath == att.xpath


def test_prove_detects_node_tamper(workspace: Workspace) -> None:
    _seed(workspace)
    att = export_node(workspace, "f001", "ds01", xpath="/dg:chunk/docset:Header")

    xml_path = workspace.file_dgml_xml_path("ds01", "f001", "ledger")
    xml_path.write_bytes(DGML_XML.replace(b"Resident Ledger", b"TAMPERED"))

    result = prove_node(workspace, "f001", "ds01", att.root_hash, att.proof)
    assert result.valid is False
    assert result.computed_node_hash != result.expected_node_hash


def test_prove_detects_wrong_root(workspace: Workspace) -> None:
    _seed(workspace)
    att = export_node(workspace, "f001", "ds01", leaf_index=1)
    result = prove_node(workspace, "f001", "ds01", "f" * 64, att.proof)
    assert result.valid is False
    # The node itself is unchanged; only the root claim is wrong.
    assert result.computed_node_hash == result.expected_node_hash


def test_prove_raises_when_document_restructured(workspace: Workspace) -> None:
    _seed(workspace)
    att = export_node(workspace, "f001", "ds01", leaf_index=8)
    xml_path = workspace.file_dgml_xml_path("ds01", "f001", "ledger")
    xml_path.write_bytes(b"<dg:chunk xmlns:dg='http://dgml.io/ns/dg#'/>")
    with pytest.raises(ValueError, match="out of range"):
        prove_node(workspace, "f001", "ds01", att.root_hash, att.proof)
