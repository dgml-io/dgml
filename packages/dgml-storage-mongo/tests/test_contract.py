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

"""MongoDocStore obeys the DocStore contract (mongomock, or real Mongo in CI)."""

from __future__ import annotations

from pathlib import Path

import pytest
from dgml_core import layout
from dgml_core.errors import InvalidArgument, StorageConfigInvalid
from dgml_core.storage import Workspace
from dgml_core.storage_service import StorageConfig
from dgml_storage_mongo import MongoDocStore

from .conftest import PROVIDER

# ------------------------------------------------------------------ config


def test_requires_a_database(tmp_path: Path) -> None:
    with pytest.raises(StorageConfigInvalid):
        MongoDocStore.parse_config(StorageConfig(provider=PROVIDER, root=tmp_path))


def test_rejects_unknown_and_credential_fields(tmp_path: Path) -> None:
    for bad in ({"mongo_database": "d", "typo": 1}, {"mongo_database": "d", "mongo_password": "x"}):
        with pytest.raises(StorageConfigInvalid):
            MongoDocStore.parse_config(StorageConfig(provider=PROVIDER, root=tmp_path, options=bad))


def test_bad_port_rejected(tmp_path: Path) -> None:
    with pytest.raises(StorageConfigInvalid):
        MongoDocStore.parse_config(
            StorageConfig(
                provider=PROVIDER, root=tmp_path, options={"mongo_database": "d", "mongo_port": "x"}
            )
        )


# ------------------------------------------------------------------ documents


def test_doc_round_trip_without_id_leak(docs: MongoDocStore) -> None:
    assert docs.get_doc("files", "f1") is None
    docs.put_doc("files", "f1", {"id": "f1", "sha256": "aa"})
    got = docs.get_doc("files", "f1")
    assert got == {"id": "f1", "sha256": "aa"}  # no Mongo _id leaks into the body


def test_put_replaces_not_merges(docs: MongoDocStore) -> None:
    docs.put_doc("files", "f1", {"id": "f1", "a": 1, "b": 2})
    docs.put_doc("files", "f1", {"id": "f1", "a": 9})
    assert docs.get_doc("files", "f1") == {"id": "f1", "a": 9}  # b is gone


def test_find_docs_queries_and_empty_is_all(docs: MongoDocStore) -> None:
    docs.put_doc("files", "f1", {"id": "f1", "kind": "pdf"})
    docs.put_doc("files", "f2", {"id": "f2", "kind": "pdf"})
    docs.put_doc("files", "f3", {"id": "f3", "kind": "docx"})
    assert len(docs.find_docs("files", {})) == 3  # empty query = whole collection
    pdfs = docs.find_docs("files", {"kind": "pdf"})
    assert {d["id"] for d in pdfs} == {"f1", "f2"}


def test_composite_ids_and_delete(docs: MongoDocStore) -> None:
    docs.put_doc("assignments", "d1/f1", {"docset_id": "d1", "file_id": "f1"})
    assert docs.get_doc("assignments", "d1/f1") == {"docset_id": "d1", "file_id": "f1"}
    docs.delete_doc("assignments", "d1/f1")
    docs.delete_doc("assignments", "d1/f1")  # idempotent
    assert docs.get_doc("assignments", "d1/f1") is None


def test_delete_docs_returns_count(docs: MongoDocStore) -> None:
    for n in range(3):
        docs.put_doc("files", f"f{n}", {"id": f"f{n}", "kind": "pdf"})
    docs.put_doc("files", "keep", {"id": "keep", "kind": "docx"})
    assert docs.delete_docs("files", {"kind": "pdf"}) == 3
    assert {d["id"] for d in docs.find_docs("files", {})} == {"keep"}


def test_append_doc_is_usage_only(docs: MongoDocStore) -> None:
    docs.append_doc(layout.Collection.USAGE, {"op": "generate", "tokens": 10})
    docs.append_doc(layout.Collection.USAGE, {"op": "extract", "tokens": 20})
    events = docs.find_docs(layout.Collection.USAGE, {})
    assert {e["op"] for e in events} == {"generate", "extract"}
    with pytest.raises(InvalidArgument):
        docs.append_doc("files", {"id": "nope"})  # addressed collection → rejected


# ------------------------------------------------------------------ pipeline


def test_workspace_routes_docs_to_mongo_and_blobs_to_local(mongo_docs_workspace: Workspace) -> None:
    from dgml_core.storage_local import LocalStore

    ws = mongo_docs_workspace
    assert isinstance(ws.docs, MongoDocStore)
    assert isinstance(ws.blobs, LocalStore)

    ws.docs.put_doc(layout.Collection.FILES, "f1", {"id": "f1"})
    ws.blobs.put_blob("files/f1/report.pdf", b"pdf")
    # The document is in Mongo (not on disk); the blob is on local disk.
    assert not (ws.root / "files" / "f1" / "file.json").exists()
    assert (ws.root / "files" / "f1" / "report.pdf").is_file()
    assert ws.docs.get_doc(layout.Collection.FILES, "f1") == {"id": "f1"}
