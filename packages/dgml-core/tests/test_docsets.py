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

from __future__ import annotations

import logging

import pytest
from dgml_core import layout
from dgml_core.docsets import DocSetStore
from dgml_core.errors import (
    DocSetNotFound,
    FileNotFound,
    InvalidArgument,
    SchemaInvalid,
    SchemaNotFound,
)
from dgml_core.storage import Workspace
from dgml_core.workspace_ops import WorkspaceOps

# A minimal valid extraction schema in the supported RNC subset (RNC is the
# canonical at-rest form since the extraction rework).
_RNC = """\
namespace dg = "http://dgml.io/ns/dg#"
namespace docset = "http://www.dgml.io/ws/X"

start =
  element dg:chunk {
    (text | Title)*
  }

Title =
  element docset:Title {
    text
  }
"""

_RNC2 = _RNC.replace("Title", "Heading")


def test_create_and_get(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="Contracts", description="signed contracts")
    assert ds.name == "Contracts"
    assert ds.key_questions == []  # default
    assert store.get(ds.id) == ds


def test_create_with_key_questions(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    questions = [
        "What is the agreement date?",
        "Who are the parties?",
        "What is the term?",
    ]
    ds = store.create(name="Contracts", key_questions=questions)
    assert ds.key_questions == questions
    assert store.get(ds.id).key_questions == questions


def test_update_key_questions(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X", key_questions=["q1", "q2"])
    # Passing key_questions=None leaves the list alone.
    untouched = store.update(ds.id, description="new")
    assert untouched.key_questions == ["q1", "q2"]
    # Passing an explicit list replaces.
    updated = store.update(ds.id, key_questions=["a", "b", "c"])
    assert updated.key_questions == ["a", "b", "c"]
    assert store.get(ds.id).key_questions == ["a", "b", "c"]


def test_from_json_tolerates_missing_key_questions(workspace: Workspace) -> None:
    """A docset.json without `key_questions` must round-trip as an empty list."""
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    # A docset.json that omits the optional key_questions field.
    minimal = {"id": ds.id, "name": "Minimal", "description": "no key_questions"}
    workspace.docs.put_doc("docsets", ds.id, minimal)
    loaded = store.get(ds.id)
    assert loaded.key_questions == []
    assert loaded.name == "Minimal"


def test_create_empty_name_rejected(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    with pytest.raises(InvalidArgument):
        store.create(name="   ")


def test_list_all(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    a = store.create(name="A")
    b = store.create(name="B")
    assert {d.id for d in store.list_all()} == {a.id, b.id}


def test_update(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="Old", description="old desc")
    updated = store.update(ds.id, name="New", description="new desc")
    assert updated.name == "New"
    assert updated.description == "new desc"
    assert store.get(ds.id) == updated


def test_update_only_name(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X", description="keep me")
    updated = store.update(ds.id, name="Y")
    assert updated.name == "Y"
    assert updated.description == "keep me"


def test_delete(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    store.delete(ds.id)
    with pytest.raises(DocSetNotFound):
        store.get(ds.id)


def test_get_missing(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    with pytest.raises(DocSetNotFound):
        store.get("doesnotexist1")


def test_add_remove_file_reference(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    fid = "abcdefghijkl"
    workspace.docs.put_doc("files", fid, {"id": fid})
    store.add_file(ds.id, fid)
    assert store.list_files(ds.id) == [fid]
    store.remove_file(ds.id, fid)
    assert store.list_files(ds.id) == []


def test_add_file_to_missing_docset(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    fid = "abcdefghijkl"
    with pytest.raises(DocSetNotFound):
        store.add_file("nosuchdocset", fid)


def test_add_missing_file(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    with pytest.raises(FileNotFound):
        store.add_file(ds.id, "doesnotexist1")


def test_remove_file_not_assigned(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    fid = "abcdefghijkl"
    with pytest.raises(FileNotFound):
        store.remove_file(ds.id, fid)


def test_add_file_rejects_empty_file_id(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    with pytest.raises(InvalidArgument):
        store.add_file(ds.id, "")
    with pytest.raises(InvalidArgument):
        store.add_file(ds.id, "   ")
    assert store.list_files(ds.id) == []


def test_add_file_rejects_nonexistent_file_id(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    with pytest.raises(FileNotFound):
        store.add_file(ds.id, "doesnotexist1")
    assert store.list_files(ds.id) == []


def test_remove_file_rejects_empty_file_id(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    with pytest.raises(InvalidArgument):
        store.remove_file(ds.id, "")


def test_delete_rejects_empty_docset_id_preserves_other_docsets(
    workspace: Workspace,
) -> None:
    """Regression: delete('') must not wipe the entire docsets directory.

    Without the empty-id guard, shutil.rmtree(docset_dir('')) collapses to
    rmtree(docsets_dir) and silently destroys every DocSet in the workspace.
    """
    store = DocSetStore(workspace)
    keep_a = store.create(name="A")
    keep_b = store.create(name="B")
    with pytest.raises(InvalidArgument):
        store.delete("")
    with pytest.raises(InvalidArgument):
        store.delete("   ")
    assert {d.id for d in store.list_all()} == {keep_a.id, keep_b.id}
    assert workspace.docsets_dir.is_dir()


def test_get_rejects_empty_docset_id(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    with pytest.raises(InvalidArgument):
        store.get("")


def test_update_rejects_empty_docset_id(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    with pytest.raises(InvalidArgument):
        store.update("", name="new")


# ---- extraction schema ------------------------------------------------------


def test_schema_get_missing(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    assert store.has_schema(ds.id) is False
    with pytest.raises(SchemaNotFound):
        store.get_schema(ds.id)


def test_schema_set_and_roundtrip(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    store.set_schema(ds.id, _RNC)
    assert store.has_schema(ds.id) is True
    assert store.get_schema(ds.id) == _RNC
    # Persisted on disk as extraction-schema.rnc in the docset directory.
    schema_key = layout.docset_extraction_schema_key(ds.id)
    assert schema_key.endswith("extraction-schema.rnc")
    assert workspace.blobs.get_blob(schema_key).decode("utf-8") == _RNC


def test_schema_set_replaces_previous(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    store.set_schema(ds.id, _RNC)
    store.set_schema(ds.id, _RNC2)
    assert store.get_schema(ds.id) == _RNC2


def test_schema_set_rejects_non_rnc(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    # Non-string inputs and strings outside the supported RNC subset both fail.
    for bad in ([1, 2, 3], 7, None, "not an rnc schema", "{}"):
        with pytest.raises(SchemaInvalid):
            store.set_schema(ds.id, bad)  # type: ignore[arg-type]
    assert store.has_schema(ds.id) is False


def test_schema_clear(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    # Clearing when nothing is set is a no-op returning False.
    assert store.clear_schema(ds.id) is False
    store.set_schema(ds.id, _RNC)
    assert store.clear_schema(ds.id) is True
    assert store.has_schema(ds.id) is False
    # Idempotent.
    assert store.clear_schema(ds.id) is False


def test_schema_ops_require_existing_docset(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    with pytest.raises(DocSetNotFound):
        store.get_schema("nonexistent1")
    with pytest.raises(DocSetNotFound):
        store.set_schema("nonexistent1", _RNC)
    with pytest.raises(DocSetNotFound):
        store.clear_schema("nonexistent1")


def test_schema_ops_reject_empty_docset_id(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    for op in (store.get_schema, store.has_schema, store.clear_schema):
        with pytest.raises(InvalidArgument):
            op("")
    with pytest.raises(InvalidArgument):
        store.set_schema("", _RNC)


def test_schema_survives_docset_delete_other(workspace: Workspace) -> None:
    """Deleting one docset must not affect another's schema."""
    store = DocSetStore(workspace)
    keep = store.create(name="keep")
    drop = store.create(name="drop")
    store.set_schema(keep.id, _RNC)
    store.set_schema(drop.id, _RNC2)
    store.delete(drop.id)
    assert store.get_schema(keep.id) == _RNC


def test_list_files_rejects_empty_docset_id(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    with pytest.raises(InvalidArgument):
        store.list_files("")


def test_add_file_rejects_empty_docset_id(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    fid = "abcdefghijkl"
    with pytest.raises(InvalidArgument):
        store.add_file("", fid)


def test_remove_file_rejects_empty_docset_id(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    with pytest.raises(InvalidArgument):
        store.remove_file("", "abcdefghijkl")


# ------------------------------------------------- assignment cascade semantics


def _assigned_pair(workspace: Workspace) -> tuple[DocSetStore, str, str]:
    """A docset with one assigned file that has generated artifacts."""
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    fid = "abcdefghijkl"
    workspace.docs.put_doc("files", fid, {"id": fid})
    store.add_file(ds.id, fid)
    workspace.blobs.put_blob(f"docsets/{ds.id}/files/{fid}/report.dgml.xml", b"<x/>")
    workspace.docs.put_doc("extraction_stats", f"{ds.id}/{fid}", {"matched": 3})
    return store, ds.id, fid


def test_add_file_records_an_assignment_document(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    fid = "abcdefghijkl"
    workspace.docs.put_doc("files", fid, {"id": fid})
    store.add_file(ds.id, fid)

    doc = workspace.docs.get_doc("assignments", f"{ds.id}/{fid}")
    assert doc is not None
    assert doc["docset_id"] == ds.id
    assert doc["file_id"] == fid
    assert doc["assigned_at"]  # the relationship can carry metadata now


def test_unassign_removes_record_and_pair_artifacts(workspace: Workspace) -> None:
    """The cascade must delete the assignment *and* the pair's artifacts.

    Each is a separate store object, so this only passes if `unassign` deletes
    all three explicitly — the behavior a remote backend depends on. It used to
    also pass on local disk when the assignment delete alone did an rmtree."""
    store, did, fid = _assigned_pair(workspace)
    WorkspaceOps(workspace).unassign(did, fid)

    assert workspace.docs.get_doc("assignments", f"{did}/{fid}") is None
    assert not workspace.blobs.blob_exists(f"docsets/{did}/files/{fid}/report.dgml.xml")
    assert workspace.docs.get_doc("extraction_stats", f"{did}/{fid}") is None
    assert store.list_files(did) == []


def test_remove_file_removes_pair_artifacts(workspace: Workspace) -> None:
    store, did, fid = _assigned_pair(workspace)
    store.remove_file(did, fid)
    assert store.list_files(did) == []
    assert not workspace.blobs.blob_exists(f"docsets/{did}/files/{fid}/report.dgml.xml")
    assert workspace.docs.get_doc("extraction_stats", f"{did}/{fid}") is None


def test_docset_delete_removes_every_assignment_and_artifact(workspace: Workspace) -> None:
    store, did, fid = _assigned_pair(workspace)
    store.delete(did)
    assert workspace.docs.get_doc("assignments", f"{did}/{fid}") is None
    assert workspace.docs.find_docs("assignments", {"docset_id": did}) == []
    assert workspace.blobs.list_blobs(f"docsets/{did}/") == []
    # the underlying file is untouched
    assert workspace.docs.get_doc("files", fid) is not None


def test_reassign_is_idempotent(workspace: Workspace) -> None:
    store, did, fid = _assigned_pair(workspace)
    store.add_file(did, fid)  # re-adding replaces the same document
    assert store.list_files(did) == [fid]
    assert len(workspace.docs.find_docs("assignments", {"docset_id": did})) == 1
    # and it must not disturb the pair's generated artifacts
    assert workspace.blobs.blob_exists(f"docsets/{did}/files/{fid}/report.dgml.xml")


# ------------------------------------------------------------ docsets_for_file


def test_docsets_for_file_lists_each_assignment(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    fid = "abcdefghijkl"
    workspace.docs.put_doc("files", fid, {"id": fid})
    a = store.create(name="A", description="first")
    b = store.create(name="B")
    store.create(name="Unrelated")
    store.add_file(b.id, fid)
    store.add_file(a.id, fid)

    result = store.docsets_for_file(fid)

    assert [r.docset.id for r in result] == sorted([a.id, b.id])
    by_id = {r.docset.id: r for r in result}
    assert by_id[a.id].docset == a
    for did in (a.id, b.id):
        doc = workspace.docs.get_doc("assignments", f"{did}/{fid}")
        assert doc is not None
        assert by_id[did].assigned_at == doc["assigned_at"]
        assert by_id[did].assigned_at


def test_docsets_for_file_unassigned_returns_empty(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    store.create(name="A")
    workspace.docs.put_doc("files", "abcdefghijkl", {"id": "abcdefghijkl"})
    assert store.docsets_for_file("abcdefghijkl") == []


def test_docsets_for_file_reflects_remove_and_docset_delete(workspace: Workspace) -> None:
    store, did, fid = _assigned_pair(workspace)
    other = store.create(name="Y")
    store.add_file(other.id, fid)

    store.remove_file(did, fid)
    assert [r.docset.id for r in store.docsets_for_file(fid)] == [other.id]
    store.delete(other.id)
    assert store.docsets_for_file(fid) == []


def test_docsets_for_file_rejects_empty_and_missing_file(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    with pytest.raises(InvalidArgument):
        store.docsets_for_file("  ")
    with pytest.raises(FileNotFound):
        store.docsets_for_file("nonexistent0")


def test_docsets_for_file_missing_assigned_at(workspace: Workspace) -> None:
    """Assignments written by the layout migration carry no timestamp."""
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    fid = "abcdefghijkl"
    workspace.docs.put_doc("files", fid, {"id": fid})
    workspace.docs.put_doc(
        "assignments", layout.pair_id(ds.id, fid), {"docset_id": ds.id, "file_id": fid}
    )

    [assignment] = store.docsets_for_file(fid)
    assert assignment.docset == ds
    assert assignment.assigned_at is None
    assert assignment.to_json() == {"docset": ds.to_json(), "assigned_at": None}


def test_docsets_for_file_skips_dangling_assignment(workspace: Workspace) -> None:
    store, did, fid = _assigned_pair(workspace)
    workspace.docs.put_doc(
        "assignments",
        layout.pair_id("gonedocset01", fid),
        {"docset_id": "gonedocset01", "file_id": fid, "assigned_at": "2026-01-01T00:00:00Z"},
    )
    assert [r.docset.id for r in store.docsets_for_file(fid)] == [did]


def test_docsets_for_file_skips_corrupt_docset_like_list_all(workspace: Workspace) -> None:
    """A corrupt ``docset.json`` reads as absent, as it does for ``list_all`` —
    it must not raise and hide the file's other, healthy DocSets."""
    store, did, fid = _assigned_pair(workspace)
    broken = store.create(name="Broken")
    store.add_file(broken.id, fid)
    (workspace.root / "docsets" / broken.id / "docset.json").write_text(
        '{"id": "truncated', encoding="utf-8"
    )

    assert [ds.id for ds in store.list_all()] == [did]
    assert [r.docset.id for r in store.docsets_for_file(fid)] == [did]


def test_docsets_for_file_warns_about_the_skipped_docset(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    store, _did, fid = _assigned_pair(workspace)
    broken = store.create(name="Broken")
    store.add_file(broken.id, fid)
    (workspace.root / "docsets" / broken.id / "docset.json").write_text("{", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="dgml_core.docsets"):
        store.docsets_for_file(fid)

    [record] = caplog.records
    assert broken.id in record.getMessage()
    assert fid in record.getMessage()


def test_docsets_for_file_unaffected_by_malformed_manifest_elsewhere(
    workspace: Workspace,
) -> None:
    """A bad manifest only breaks lookups for files assigned to that DocSet.

    This is why the DocSets are read one assignment at a time rather than via
    ``list_all``: a manifest that parses but is malformed (valid JSON, no
    ``name``) raises from ``list_all`` and would otherwise fail this lookup for
    every file in the workspace."""
    store, did, fid = _assigned_pair(workspace)
    unrelated = store.create(name="Unrelated")
    workspace.docs.put_doc("docsets", unrelated.id, {"id": unrelated.id})

    with pytest.raises(KeyError):
        store.list_all()
    assert [r.docset.id for r in store.docsets_for_file(fid)] == [did]


# ---- extraction guidance ---------------------------------------------------


def test_guidance_get_missing(workspace: Workspace) -> None:
    from dgml_core.errors import GuidanceNotFound

    store = DocSetStore(workspace)
    ds = store.create(name="Bills")
    assert store.has_guidance(ds.id) is False
    with pytest.raises(GuidanceNotFound):
        store.get_guidance(ds.id)


def test_guidance_set_and_roundtrip(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="Bills")
    text = "# Charge classification\nClassify by behavior, not name.\n"
    store.set_guidance(ds.id, text)
    assert store.has_guidance(ds.id) is True
    assert store.get_guidance(ds.id) == text
    key = f"docsets/{ds.id}/extraction-guidance.md"
    assert workspace.blobs.blob_exists(key)
    assert workspace.blobs.get_blob(key).decode("utf-8") == text


def test_guidance_set_replaces_and_clear(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="Bills")
    store.set_guidance(ds.id, "v1")
    store.set_guidance(ds.id, "v2")
    assert store.get_guidance(ds.id) == "v2"
    assert store.clear_guidance(ds.id) is True
    assert store.has_guidance(ds.id) is False
    assert store.clear_guidance(ds.id) is False


def test_guidance_rejects_empty_text_and_missing_docset(workspace: Workspace) -> None:
    from dgml_core.errors import DocSetNotFound, InvalidArgument

    store = DocSetStore(workspace)
    ds = store.create(name="Bills")
    with pytest.raises(InvalidArgument):
        store.set_guidance(ds.id, "   ")
    with pytest.raises(DocSetNotFound):
        store.set_guidance("nope00000000", "text")
    with pytest.raises(DocSetNotFound):
        store.get_guidance("nope00000000")
