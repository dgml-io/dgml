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

"""The list of workspaces in MongoDB.

Three things are worth testing here and the rest follows from them:

1. **Parity.** A workspace must be described identically whichever backend lists it.
   The derived listing fields are computed in the base class precisely so that is
   structural, and the fidelity test below is what proves the config text itself
   survives a round trip byte-for-byte, which the local backend also promises.
2. **Lost updates.** A config is written whole, so an overwrite discards the other
   writer's `[storage]` table and comments — not just the field being written. That is
   why this backend has a revision and the local one does not.
3. **The representation.** `config_toml` must stay a single string. A future
   "let's just parse it into BSON" refactor has to fail loudly here rather than quietly
   start dropping comments and rejecting valid TOML dates.
"""

from __future__ import annotations

import pytest
from dgml_core.errors import (
    CorruptMetadata,
    WorkspacesConfigInvalid,
    WorkspacesUnavailable,
    WorkspacesWriteConflict,
)
from dgml_core.layout import Collection
from dgml_core.workspaces_local import LocalDirWorkspacesStore
from dgml_core.workspaces_store import WorkspacesConfig, WorkspacesStore
from dgml_storage_mongo import MongoWorkspacesStore

CONFIG = """\
# a comment the user wrote
[workspace]
name = "Acme Contracts"
organization = "acme"
storage_service = "bym"
created_at = "2026-08-26T18:04:11Z"
"""

# A real id: base32-lower, so only [a-z2-7] — 0/1/8/9 never appear in one.
WID = "ws_7qxdm2pjk3n5rwts"


# ---------------------------------------------------------------- configuration


def test_requires_a_database() -> None:
    with pytest.raises(WorkspacesConfigInvalid, match="mongo_database"):
        MongoWorkspacesStore.parse_config(
            WorkspacesConfig(provider="dgml_storage_mongo:MongoWorkspacesStore")
        )


def test_config_errors_name_the_workspaces_table() -> None:
    """The shared validator is reused, but it must report against the table the user
    actually wrote — this is configured under ``[workspaces]``, not ``[storage]``."""
    with pytest.raises(WorkspacesConfigInvalid, match=r"\[workspaces\]"):
        MongoWorkspacesStore.parse_config(
            WorkspacesConfig(provider="dgml_storage_mongo:MongoWorkspacesStore")
        )


@pytest.mark.parametrize("field", ["bucket", "prefix", "mongo_uri", "mongo_password"])
def test_unknown_or_credential_fields_are_rejected(field: str) -> None:
    """Including the credential-shaped ones: there is deliberately no way to put a
    connection string in config, even a user-level one."""
    cfg = WorkspacesConfig(
        provider="dgml_storage_mongo:MongoWorkspacesStore",
        options={"mongo_database": "db", field: "x"},
    )
    with pytest.raises(WorkspacesConfigInvalid):
        MongoWorkspacesStore.parse_config(cfg)


@pytest.mark.parametrize("collection", sorted(member.value for member in Collection))
def test_a_workspace_document_collection_is_refused(collection: str) -> None:
    """This collection may share a database with a workspace's own documents, so it must
    not be able to shadow one of them. The GridFS store makes this argument in prose;
    here it is enforced."""
    cfg = WorkspacesConfig(
        provider="dgml_storage_mongo:MongoWorkspacesStore",
        options={"mongo_database": "db", "mongo_collection": collection},
    )
    with pytest.raises(WorkspacesConfigInvalid, match="workspace's own documents"):
        MongoWorkspacesStore.parse_config(cfg)


@pytest.mark.parametrize("collection", ["blobs.files", "blobs.chunks"])
def test_a_gridfs_collection_is_refused(collection: str) -> None:
    cfg = WorkspacesConfig(
        provider="dgml_storage_mongo:MongoWorkspacesStore",
        options={"mongo_database": "db", "mongo_collection": collection},
    )
    with pytest.raises(WorkspacesConfigInvalid, match="GridFS"):
        MongoWorkspacesStore.parse_config(cfg)


def test_label_names_the_database_and_collection(
    workspaces_store: MongoWorkspacesStore,
) -> None:
    """It goes into "no workspace ws_… in <label>", so it has to be specific enough to
    act on."""
    label = workspaces_store.label()
    assert label.startswith("mongo:")
    assert "dgml_workspaces" in label


# ------------------------------------------------------------------------- CRUD


def test_write_read_list_delete(workspaces_store: MongoWorkspacesStore) -> None:
    assert workspaces_store.read_config(WID) is None
    assert workspaces_store.exists(WID) is False

    workspaces_store.write_config(WID, CONFIG)
    found = workspaces_store.read_config(WID)
    assert found is not None and found[0] == CONFIG
    assert workspaces_store.exists(WID) is True
    assert workspaces_store.list_ids() == [WID]

    assert workspaces_store.delete(WID) is True
    assert workspaces_store.delete(WID) is False
    assert workspaces_store.list_ids() == []


def test_list_entries_derives_from_the_config(
    workspaces_store: MongoWorkspacesStore,
) -> None:
    workspaces_store.write_config(WID, CONFIG)
    (entry,) = workspaces_store.list_entries()
    assert entry.workspace_id == WID
    assert entry.name == "Acme Contracts"
    assert entry.organization == "acme"
    assert entry.storage_service == "bym"
    assert entry.created_at == "2026-08-26T18:04:11Z"


def test_list_entries_transfers_no_config_text(
    workspaces_store: MongoWorkspacesStore,
) -> None:
    """The whole reason the projection exists: `workspace list` must not pull every
    workspace's config across the wire to render a table."""
    workspaces_store.write_config(WID, CONFIG)
    seen: list[dict[str, int] | None] = []
    original = workspaces_store._docs.find

    def _spy(query: dict[str, object], projection: dict[str, int] | None = None):  # type: ignore[no-untyped-def]
        seen.append(projection)
        return original(query, projection)

    workspaces_store._docs.find = _spy  # type: ignore[method-assign]
    workspaces_store.list_entries()
    assert seen and all("config_toml" not in (p or {}) for p in seen)


def test_derived_fields_are_regenerated_on_every_write(
    workspaces_store: MongoWorkspacesStore,
) -> None:
    """They are a projection, never authority — so editing the config must move them,
    with nothing re-indexing anything."""
    workspaces_store.write_config(WID, CONFIG)
    found = workspaces_store.read_config(WID)
    assert found is not None
    workspaces_store.write_config(
        WID, found[0].replace("Acme Contracts", "Renamed"), expected_revision=found[1]
    )
    (entry,) = workspaces_store.list_entries()
    assert entry.name == "Renamed"


def test_a_document_without_config_toml_is_corrupt(
    workspaces_store: MongoWorkspacesStore,
) -> None:
    """Something other than dgml wrote it. Better a clear error than treating a
    workspace as configuration-free and opening it on the local default."""
    workspaces_store._docs.insert_one({"_id": WID, "name": "hand-written"})
    with pytest.raises(CorruptMetadata, match="config_toml"):
        workspaces_store.read_config(WID)


# ------------------------------------------------------------------- fidelity


HOSTILE = (
    "# leading comment\r\n"
    "\n"
    "[workspace]\n"
    'name = "Zed"   # trailing comment\n'
    "\n"
    "# between tables — non-ASCII: café ☕\n"
    "[storage.bym.blobs]\n"
    'bucket = "dgml-dev"\n'
    "\n"
    "[storage.bym.docs]\n"
    "mongo_port = 27017\n"
    "a.b.c = 1\n"
    "\n"
    "# TOML dates BSON cannot encode — the reason this field is text, not a document\n"
    "[generation]\n"
    "cutoff = 2026-01-01\n"
    "at = 07:32:00\n"
    "stamped = 2026-01-01T07:32:00.123456-08:00\n"
    "\n"
    "[[thing]]\n"
    'label = """multi\nline"""'  # deliberately no trailing newline
)


def test_config_text_round_trips_byte_for_byte(
    workspaces_store: MongoWorkspacesStore,
) -> None:
    """Comments, key order, CRLF, a missing trailing newline, dotted keys, an array of
    tables — and crucially the bare local date and the microsecond offset datetime,
    which a parsed-BSON store would reject outright or silently truncate."""
    workspaces_store.write_config(WID, HOSTILE)
    found = workspaces_store.read_config(WID)
    assert found is not None
    assert found[0] == HOSTILE


def test_the_config_is_stored_as_one_string(
    workspaces_store: MongoWorkspacesStore,
) -> None:
    """A landmine for a future "let's parse it into BSON" refactor: that change would
    make comment preservation backend-dependent and would refuse configs the local
    backend accepts, so it has to fail here rather than in a user's workspace."""
    workspaces_store.write_config(WID, CONFIG)
    doc = workspaces_store._docs.find_one({"_id": WID})
    assert doc is not None
    assert isinstance(doc["config_toml"], str)
    assert "storage" not in doc  # not exploded into sub-documents


# ---------------------------------------------------------------- lost updates


def test_a_stale_revision_is_refused(
    workspaces_store: MongoWorkspacesStore, workspaces_store_b: MongoWorkspacesStore
) -> None:
    """Two machines, one workspace. The loser must be told, not silently discarded —
    its write carries a whole config, so it would take the winner's `[storage]` table
    and comments with it."""
    workspaces_store.write_config(WID, CONFIG)
    first = workspaces_store.read_config(WID)
    assert first is not None
    text, stale_revision = first

    # The other machine reads, then writes first.
    other = workspaces_store_b.read_config(WID)
    assert other is not None
    workspaces_store_b.write_config(
        WID,
        other[0] + "\n[models]\nadvanced = 'x'\n",
        expected_revision=other[1],
    )

    with pytest.raises(WorkspacesWriteConflict, match="another writer"):
        workspaces_store.write_config(WID, text, expected_revision=stale_revision)

    # The winner's content is intact.
    current = workspaces_store.read_config(WID)
    assert current is not None and "[models]" in current[0]


def test_a_fresh_revision_is_accepted_and_bumped(
    workspaces_store: MongoWorkspacesStore,
) -> None:
    workspaces_store.write_config(WID, CONFIG)
    found = workspaces_store.read_config(WID)
    assert found is not None
    text, revision = found
    assert isinstance(revision, int)

    returned = workspaces_store.write_config(WID, text, expected_revision=revision)
    assert returned == revision + 1

    after = workspaces_store.read_config(WID)
    assert after is not None and after[1] == revision + 1


def test_writing_a_deleted_workspace_says_so(
    workspaces_store: MongoWorkspacesStore, workspaces_store_b: MongoWorkspacesStore
) -> None:
    """Distinguished from a losing race on purpose: "it is gone" and "someone else
    changed it" call for different responses, and a caller retrying blindly would treat
    them the same."""
    workspaces_store.write_config(WID, CONFIG)
    found = workspaces_store.read_config(WID)
    assert found is not None
    workspaces_store_b.delete(WID)

    with pytest.raises(WorkspacesWriteConflict, match="no longer in the store"):
        workspaces_store.write_config(WID, CONFIG, expected_revision=found[1])


def test_two_instances_share_one_database(
    workspaces_store: MongoWorkspacesStore, workspaces_store_b: MongoWorkspacesStore
) -> None:
    """Guards the conflict tests against going vacuous: two mongomock clients built from
    the same URI are separate in-memory universes by default, so without the memoizing
    factory in conftest the tests above would pass while asserting nothing."""
    workspaces_store.write_config(WID, CONFIG)
    assert workspaces_store_b.read_config(WID) is not None


# --------------------------------------------------------------------- parity


def test_both_backends_describe_a_workspace_identically(
    workspaces_store: MongoWorkspacesStore, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    """The strongest assertion in the file. Identical text in, identical listing row
    out — which holds because the derivation lives in shared base-class code rather than
    being re-implemented per backend."""
    local: WorkspacesStore = LocalDirWorkspacesStore(
        LocalDirWorkspacesStore.parse_config(
            WorkspacesConfig(
                provider="dgml_core.workspaces_local:LocalDirWorkspacesStore",
                options={"root": str(tmp_path / "workspaces")},
            )
        )
    )
    for store in (local, workspaces_store):
        store.write_config(WID, HOSTILE)

    local_entry, mongo_entry = local.list_entries()[0], workspaces_store.list_entries()[0]
    assert local_entry == mongo_entry

    local_text = local.read_config(WID)
    mongo_text = workspaces_store.read_config(WID)
    assert local_text is not None and mongo_text is not None
    assert local_text[0] == mongo_text[0] == HOSTILE


# --------------------------------------------------------------- reachability


def test_an_unreachable_store_is_reported_actionably(
    workspaces_store: MongoWorkspacesStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The most likely operational failure of this backend — server down, wrong host,
    VPN off — and every command needs the store to find a workspace.

    pymongo connects lazily, so it surfaces on the first *operation*. Left unwrapped it
    arrives as an `INTERNAL_ERROR` carrying a paragraph of driver internals, which tells
    a caller (or an agent parsing the envelope) nothing it can act on.
    """
    from pymongo.errors import ServerSelectionTimeoutError

    def _boom(*args: object, **kwargs: object) -> None:
        raise ServerSelectionTimeoutError("localhost:27099: [Errno 61] Connection refused")

    monkeypatch.setattr(workspaces_store._docs, "find_one", _boom)
    with pytest.raises(WorkspacesUnavailable) as caught:
        workspaces_store.read_config(WID)
    message = str(caught.value)
    assert workspaces_store.label() in message
    assert "[workspaces]" in message  # names where to look


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.read_config(WID),
        lambda s: s.exists(WID),
        lambda s: s.list_ids(),
        lambda s: s.list_configs(),
        lambda s: s.list_entries(),
        lambda s: s.delete(WID),
        lambda s: s.write_config(WID, CONFIG),
    ],
    ids=["read", "exists", "list_ids", "list_configs", "list_entries", "delete", "write"],
)
def test_every_operation_reports_unreachability(
    workspaces_store: MongoWorkspacesStore,
    monkeypatch: pytest.MonkeyPatch,
    call: object,
) -> None:
    """Every entry point, not just the one that happened to be tested: which operation
    runs first depends on the command, so a single unwrapped method means the bad
    envelope comes back at random."""
    from pymongo.errors import ConnectionFailure

    def _boom(*args: object, **kwargs: object) -> None:
        raise ConnectionFailure("connection refused")

    for method in ("find_one", "find", "update_one", "delete_one"):
        monkeypatch.setattr(workspaces_store._docs, method, _boom)
    with pytest.raises(WorkspacesUnavailable):
        call(workspaces_store)  # type: ignore[operator]


def test_a_networked_store_names_no_config_file(
    workspaces_store: MongoWorkspacesStore,
) -> None:
    """There is no file here, so `config_file` is None and `config_location` falls back
    to naming the store and the id.

    The distinction is the point: the local backend answers with a real, editable path,
    and this one must not invent one — a plausible-looking path that does not exist
    invites a caller to open or restore it."""
    workspaces_store.write_config(WID, CONFIG)
    assert workspaces_store.config_file(WID) is None
    location = workspaces_store.config_location(WID)
    assert location == f"{workspaces_store.label()}/{WID}"
    assert location.startswith("mongo:")
