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

"""Tests for the cascading-delete operations layer."""

from __future__ import annotations

import pytest
from dgml_core import layout
from dgml_core.docsets import DocSetStore
from dgml_core.errors import DocSetNotFound, FileNotFound, InvalidArgument
from dgml_core.storage import Workspace
from dgml_core.workspace_ops import WorkspaceOps


def _pair(ws: Workspace, docset_id: str, file_id: str) -> None:
    """An assigned (docset, file) pair with both a generated blob and a
    dependent document, so a cascade has all three kinds of key to remove."""
    ws.docs.put_doc(layout.Collection.FILES, file_id, {"id": file_id})
    ws.docs.put_doc(layout.Collection.DOCSETS, docset_id, {"id": docset_id, "name": "X"})
    DocSetStore(ws).add_file(docset_id, file_id)
    ws.blobs.put_blob(layout.dgml_xml_key(docset_id, file_id, "report"), b"<x/>")
    ws.docs.put_doc(
        layout.Collection.EXTRACTION_STATS, layout.pair_id(docset_id, file_id), {"matched": 3}
    )


def _pair_keys(docset_id: str, file_id: str) -> tuple[str, str, str]:
    return (
        layout.pair_id(docset_id, file_id),
        layout.dgml_xml_key(docset_id, file_id, "report"),
        layout.docset_pair_prefix(docset_id, file_id),
    )


# ------------------------------------------------------------------ cascades


def test_unassign_removes_record_dependents_and_blobs(workspace: Workspace) -> None:
    _pair(workspace, "d1", "f1")
    pair, xml_key, _ = _pair_keys("d1", "f1")

    WorkspaceOps(workspace).unassign("d1", "f1")

    assert workspace.docs.get_doc(layout.Collection.ASSIGNMENTS, pair) is None
    assert workspace.docs.get_doc(layout.Collection.EXTRACTION_STATS, pair) is None
    assert not workspace.blobs.blob_exists(xml_key)
    # the file itself is untouched — a docset is a grouping, not an owner
    assert workspace.docs.get_doc(layout.Collection.FILES, "f1") is not None


def test_delete_file_unassigns_from_every_docset(workspace: Workspace) -> None:
    _pair(workspace, "d1", "f1")
    _pair(workspace, "d2", "f1")
    workspace.blobs.put_blob(layout.file_source_key("f1", "r.pdf"), b"%PDF-1.4\n")

    WorkspaceOps(workspace).delete_file("f1")

    assert workspace.docs.get_doc(layout.Collection.FILES, "f1") is None
    assert workspace.docs.find_docs(layout.Collection.ASSIGNMENTS, {"file_id": "f1"}) == []
    assert workspace.blobs.list_blobs(layout.file_prefix("f1")) == []
    for did in ("d1", "d2"):
        assert workspace.blobs.list_blobs(layout.docset_pair_prefix(did, "f1")) == []
        # the docsets themselves survive
        assert workspace.docs.get_doc(layout.Collection.DOCSETS, did) is not None


def test_delete_docset_leaves_the_files_alone(workspace: Workspace) -> None:
    _pair(workspace, "d1", "f1")
    _pair(workspace, "d1", "f2")
    workspace.blobs.put_blob(layout.file_source_key("f1", "r.pdf"), b"%PDF-1.4\n")

    WorkspaceOps(workspace).delete_docset("d1")

    assert workspace.docs.get_doc(layout.Collection.DOCSETS, "d1") is None
    assert workspace.docs.find_docs(layout.Collection.ASSIGNMENTS, {"docset_id": "d1"}) == []
    assert workspace.blobs.list_blobs(layout.docset_prefix("d1")) == []
    assert workspace.docs.get_doc(layout.Collection.FILES, "f1") is not None
    assert workspace.blobs.blob_exists(layout.file_source_key("f1", "r.pdf"))


# ------------------------------------------------------------------ ordering


def test_interrupted_cascade_leaves_orphaned_bytes_not_a_dangling_record(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the ordering rule.

    The store has no cross-key transaction, so a cascade *will* sometimes be
    interrupted. What matters is which half-state it leaves: orphaned bytes are
    recoverable garbage a later sweep can identify and remove, whereas a record
    whose artifacts are gone is indistinguishable from a valid entity and
    misleads everything downstream. Killing the blob delete — the last step —
    must therefore leave the record already gone."""
    _pair(workspace, "d1", "f1")
    pair, xml_key, _ = _pair_keys("d1", "f1")
    ops = WorkspaceOps(workspace)

    def boom(prefix: str) -> None:
        raise OSError("interrupted before the blobs were removed")

    monkeypatch.setattr(ops.blobs, "delete_blobs", boom)
    with pytest.raises(OSError):
        ops.unassign("d1", "f1")

    assert workspace.docs.get_doc(layout.Collection.ASSIGNMENTS, pair) is None
    assert workspace.docs.get_doc(layout.Collection.EXTRACTION_STATS, pair) is None
    assert workspace.blobs.blob_exists(xml_key)  # orphaned, and recoverable


def test_interrupted_cascade_is_resumable(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-running after an interruption completes it; the already-deleted
    documents make the second pass a no-op rather than an error."""
    _pair(workspace, "d1", "f1")
    _, xml_key, prefix = _pair_keys("d1", "f1")
    ops = WorkspaceOps(workspace)

    real = ops.blobs.delete_blobs
    monkeypatch.setattr(ops.blobs, "delete_blobs", lambda p: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        ops.unassign("d1", "f1")

    monkeypatch.setattr(ops.blobs, "delete_blobs", real)
    ops.unassign("d1", "f1")  # replay
    assert not workspace.blobs.blob_exists(xml_key)
    assert workspace.blobs.list_blobs(prefix) == []


def test_cascades_are_idempotent(workspace: Workspace) -> None:
    """Running a completed cascade again must be a clean no-op on the parts that
    are already gone (the entity check is what raises, not a missing key)."""
    _pair(workspace, "d1", "f1")
    ops = WorkspaceOps(workspace)
    ops.unassign("d1", "f1")
    ops.unassign("d1", "f1")  # no error
    ops.delete_docset("d1")
    with pytest.raises(DocSetNotFound):
        ops.delete_docset("d1")


# -------------------------------------------------------------- preconditions


def test_missing_entities_raise(workspace: Workspace) -> None:
    ops = WorkspaceOps(workspace)
    with pytest.raises(FileNotFound):
        ops.delete_file("nosuchfile")
    with pytest.raises(DocSetNotFound):
        ops.delete_docset("nosuchdocset")


def test_blank_ids_rejected(workspace: Workspace) -> None:
    ops = WorkspaceOps(workspace)
    for blank in ("", "   "):
        with pytest.raises(InvalidArgument):
            ops.delete_file(blank)
        with pytest.raises(InvalidArgument):
            ops.delete_docset(blank)


# --------------------------------------------------------- one store per op


def test_a_cascade_resolves_the_backend_once(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolving the backend means reading config, importing the provider and
    constructing it — a fresh SDK client per call on a remote store. A cascade
    over N assignments issues on the order of 3N store calls, so it must resolve
    once, not per call.

    Counted from a *cold* workspace so the assertion holds on its own terms
    rather than riding on the ``Workspace.blobs`` / ``.docs`` caches."""
    for fid in ("f1", "f2", "f3"):
        _pair(workspace, "d1", fid)

    import dgml_core.storage_resolve as storage_resolve

    built = {"blobs": 0, "docs": 0}
    real_blob = storage_resolve.make_blob_store
    real_doc = storage_resolve.make_doc_store

    def counting_blob(config: object) -> object:
        built["blobs"] += 1
        return real_blob(config)  # type: ignore[arg-type]

    def counting_doc(config: object) -> object:
        built["docs"] += 1
        return real_doc(config)  # type: ignore[arg-type]

    monkeypatch.setattr(storage_resolve, "make_blob_store", counting_blob)
    monkeypatch.setattr(storage_resolve, "make_doc_store", counting_doc)

    cold = Workspace(root=workspace.root)  # nothing cached yet
    WorkspaceOps(cold).delete_docset("d1")
    # Resolved once, not per call. ``docs`` is zero rather than one because this
    # fixture is zero-config: both roles are ``LocalStore``, so the two configs
    # compare equal and ``ws.docs`` shares the instance ``make_blob_store`` built.
    assert built == {"blobs": 1, "docs": 0}
