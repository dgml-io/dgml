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


def test_ids_with_separators_round_trip(docs: MongoDocStore) -> None:
    """Caller-supplied ids may contain `-` and `_` (`file add --id`). They are
    Mongo `_id` values and one half of a composite `<docset>/<file>` id."""
    docs.put_doc("files", "invoice_2024_q1", {"id": "invoice_2024_q1"})
    assert docs.get_doc("files", "invoice_2024_q1") == {"id": "invoice_2024_q1"}
    docs.put_doc("assignments", "my_set-2/invoice_2024", {"file_id": "invoice_2024"})
    assert docs.get_doc("assignments", "my_set-2/invoice_2024") == {"file_id": "invoice_2024"}
    docs.delete_doc("files", "invoice_2024_q1")
    assert docs.get_doc("files", "invoice_2024_q1") is None


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


# ----------------------------------------------------- one namespace per workspace


def _docs_for(
    mongo_config: StorageConfig, workspace_id: str | None, **options: object
) -> MongoDocStore:
    config = StorageConfig(
        provider=PROVIDER,
        root=mongo_config.root,
        options={**mongo_config.options, **options},
        workspace_id=workspace_id,
    )
    return MongoDocStore(MongoDocStore.parse_config(config))


def test_prefix_is_validated(tmp_path: Path) -> None:
    for bad in ("", "a.b", "a$b", "_lead", "a" * 17, 3):
        with pytest.raises(StorageConfigInvalid):
            MongoDocStore.parse_config(
                StorageConfig(
                    provider=PROVIDER,
                    root=tmp_path,
                    options={"mongo_database": "d", "prefix": bad},
                    workspace_id="ws-test",
                )
            )


def test_a_workspace_without_an_id_is_refused(mongo_config: StorageConfig) -> None:
    """Collections without the id would be shared by every id-less workspace on the
    database, and would move the moment this one got an id."""
    with pytest.raises(StorageConfigInvalid, match="workspace's id"):
        _docs_for(mongo_config, None)


def test_collections_are_named_for_the_prefix_and_workspace_id(
    mongo_config: StorageConfig,
) -> None:
    """``<prefix>_<id>_<name>``, with the id added at runtime and ``prefix`` defaulting to
    ``dgml`` — the config holds only what the user wrote."""
    default = _docs_for(mongo_config, "ws_a")
    default.put_doc(layout.Collection.FILES, "f1", {"id": "f1"})
    default.append_doc(layout.Collection.USAGE, {"op": "generate"})
    assert set(default._db.list_collection_names()) == {"dgml_ws_a_files", "dgml_ws_a_usage"}

    custom = _docs_for(mongo_config, "ws_a", prefix="contracts")
    custom.put_doc(layout.Collection.FILES, "f1", {"id": "f1"})
    assert "contracts_ws_a_files" in custom._db.list_collection_names()


def test_workspaces_with_one_config_share_a_database_without_colliding(
    mongo_config: StorageConfig,
) -> None:
    """The regression this exists for: ``workspace`` is a singleton collection whose
    ``_id`` is the literal ``"workspace"``, so without the id in its name the second
    workspace on a database silently overwrote the first one's identity."""
    a, b = _docs_for(mongo_config, "ws-a"), _docs_for(mongo_config, "ws-b")
    ws = layout.Collection.WORKSPACE
    a.put_doc(ws, ws, {"name": "A"})
    b.put_doc(ws, ws, {"name": "B"})
    a.put_doc(layout.Collection.FILES, "f1", {"id": "f1"})

    assert a.get_doc(ws, ws) == {"name": "A"}
    assert b.get_doc(ws, ws) == {"name": "B"}
    assert b.find_docs(layout.Collection.FILES, {}) == []
