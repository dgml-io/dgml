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

"""A whole workspace on MongoDB — blobs *and* documents, via ``MongoGridFSStore``.

The contract tests cover each role in isolation; these cover the composition:
that the resolver's flat form wires one class into both roles, that blobs and
documents keep out of each other's way in a shared database, and — the one that
matters most — that a file's attestation root does not depend on which backend
the workspace happens to live on.

Deliberately free of ghostscript and any PDF library: artifacts are placed
directly, so the suite runs the same everywhere rather than skipping.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from dgml_core import layout
from dgml_core.file_attestation import attest_file_version, collect_file_version
from dgml_core.storage import Workspace
from dgml_core.storage_service import BlobStore, DocStore
from dgml_storage_mongo import MongoDocStore

SOURCE = b"%PDF-1.4 pretend source document\n"
PAGES = {1: b"\x89PNG\r\n\x1a\n page one", 2: b"\x89PNG\r\n\x1a\n page two"}


def _manifest(file_id: str) -> dict[str, object]:
    return {
        "id": file_id,
        "original_path": "../files/report.pdf",
        "original_filename": "report.pdf",
        "sha256": hashlib.sha256(SOURCE).hexdigest(),
        "added_at": "2026-01-01T00:00:00Z",
        "page_count": len(PAGES),
    }


def _populate(ws: Workspace, file_id: str) -> None:
    """Place one file's worth of artifacts: manifest (document) + source and
    page images (blobs)."""
    ws.docs.put_doc(layout.Collection.FILES, file_id, _manifest(file_id))
    ws.blobs.put_blob(layout.file_source_key(file_id, "report.pdf"), SOURCE)
    for page, data in PAGES.items():
        ws.blobs.put_blob(layout.file_page_image_key(file_id, page), data)


@pytest.fixture
def ws(mongo_gridfs_workspace: Workspace) -> Workspace:
    """An all-Mongo workspace."""
    return mongo_gridfs_workspace


# ------------------------------------------------------------- composition


def test_flat_form_serves_both_roles_from_one_class(ws: Workspace) -> None:
    # One *class* in both roles, and genuinely both interfaces.
    assert type(ws.blobs) is type(ws.docs)
    assert isinstance(ws.blobs, BlobStore)
    assert isinstance(ws.docs, DocStore)
    assert isinstance(ws.docs, MongoDocStore)

    # One class *and* one instance: both roles resolve to the same config, so
    # Workspace builds the provider once and serves both from it. Pinned because
    # it governs how many MongoClients a flat-form workspace holds — one, not
    # two.
    assert ws.blobs is ws.docs


def test_nothing_lands_on_local_disk(ws: Workspace) -> None:
    _populate(ws, "f1")
    assert ws.blobs.get_blob(layout.file_source_key("f1", "report.pdf")) == SOURCE
    assert ws.docs.get_doc(layout.Collection.FILES, "f1") == _manifest("f1")
    # ``init`` scaffolds files/ and docsets/ on local disk unconditionally, but
    # no artifact lands in them: config.toml is the only local content, because
    # the config names the store and so cannot live inside it.
    assert list((ws.root / layout.FILES_DIR).rglob("*")) == []
    assert list((ws.root / layout.DOCSETS_DIR).rglob("*")) == []
    assert sorted(p.name for p in ws.root.iterdir()) == [
        layout.CONFIG_FILE,
        layout.DOCSETS_DIR,
        layout.FILES_DIR,
    ]


def test_documents_and_blobs_share_a_database_without_colliding(ws: Workspace) -> None:
    """``files`` is both a document collection and a blob key prefix. The blob
    half is namespaced into the GridFS bucket's ``<bucket>.files`` /
    ``<bucket>.chunks``, so neither shadows the other."""
    _populate(ws, "f1")

    # The manifest is a document — never a blob, whatever the key prefix.
    blob_keys = ws.blobs.list_blobs(layout.file_prefix("f1"))
    assert not any(key.endswith(layout.FILE_MANIFEST) for key in blob_keys)
    assert layout.file_source_key("f1", "report.pdf") in blob_keys

    # And the document collection is untouched by blob writes.
    assert [d["id"] for d in ws.docs.find_docs(layout.Collection.FILES, {})] == ["f1"]


def test_delete_blobs_leaves_documents_alone(ws: Workspace) -> None:
    """The cascade contract: ``delete_blobs`` runs last and is blob-only, so an
    interrupted cascade orphans bytes rather than stranding a record."""
    _populate(ws, "f1")
    ws.blobs.delete_blobs(layout.file_prefix("f1"))
    assert ws.blobs.list_blobs(layout.file_prefix("f1")) == []
    assert ws.docs.get_doc(layout.Collection.FILES, "f1") is not None


def test_usage_log_still_appends(ws: Workspace) -> None:
    ws.docs.append_doc(layout.Collection.USAGE, {"op": "generate", "tokens": 10})
    ws.docs.append_doc(layout.Collection.USAGE, {"op": "extract", "tokens": 20})
    assert len(ws.docs.find_docs(layout.Collection.USAGE, {})) == 2


# --------------------------------------------------------------- attestation


def test_attestation_root_is_backend_independent(ws: Workspace, tmp_path: Path) -> None:
    """Proof of Origin must not move with the storage backend.

    This store serves ``sha256_blob`` from a digest recorded in GridFS
    ``metadata`` at write time rather than re-hashing the bytes, so this is the
    test that keeps that shortcut honest: identical artifacts must produce an
    identical Merkle root on Mongo and on local disk.
    """
    local_root = tmp_path / "local-ws"
    local_root.mkdir()
    local_ws = Workspace(root=local_root)
    local_ws.init()  # no config.toml → bundled local store for both roles

    roots = {}
    for label, workspace in {"gridfs": ws, "local": local_ws}.items():
        _populate(workspace, "f1")
        roots[label] = attest_file_version(collect_file_version(workspace, "f1")).root

    assert len(set(roots.values())) == 1, roots

    attestation = attest_file_version(collect_file_version(ws, "f1"))
    assert [leaf.slot_id for leaf in attestation.leaves] == [
        "source",
        "page_image[1]",
        "page_image[2]",
    ]
    # Each leaf is the plain SHA-256 of the stored bytes, not a derived checksum.
    source_leaf = next(leaf for leaf in attestation.leaves if leaf.slot_id == "source")
    assert source_leaf.leaf_hash == hashlib.sha256(SOURCE).hexdigest()
