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

"""Tests for the pluggable BlobStore/DocStore: LocalStore, resolver, config, fingerprint."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from dgml_core import (
    DEFAULT_STORAGE_PROVIDER,
    BlobStore,
    DocStore,
    LocalStore,
    StorageConfig,
    Workspace,
    load_store_configs,
    make_blob_store,
    make_doc_store,
    storage_fingerprint,
)
from dgml_core.errors import (
    InvalidArgument,
    StorageConfigInvalid,
    StorageProviderUnresolvable,
)
from dgml_core.hashing import CHUNK_SIZE

from .conftest import DefaultBridgeStore, default_bridge_store, local_store

# --------------------------------------------------------------------------- blobs


def test_blob_maps_to_real_on_disk_path(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    key = "files/abc/page_images/page_1.png"
    assert store.blob_exists(key) is False
    store.put_blob(key, b"\x89PNG-data")
    assert store.blob_exists(key) is True
    assert store.get_blob(key) == b"\x89PNG-data"
    # the blob is at the exact legacy path — no sandbox prefix
    assert (
        tmp_path / "files" / "abc" / "page_images" / "page_1.png"
    ).read_bytes() == b"\x89PNG-data"
    # put overwrites (S3 semantics = update)
    store.put_blob(key, b"replaced")
    assert store.get_blob(key) == b"replaced"


def test_blob_missing_get_raises(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get_blob("files/x/page_images/page_1.png")


def test_blob_delete_is_idempotent(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_blob("files/a/report.pdf", b"1")
    store.delete_blob("files/a/report.pdf")
    store.delete_blob("files/a/report.pdf")  # no error on missing
    assert store.blob_exists("files/a/report.pdf") is False


def test_list_blobs_excludes_documents(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_blob("files/f1/report.pdf", b"pdf")
    store.put_blob("files/f1/page_images/page_1.png", b"png")
    store.put_blob("files/f1/page_text/page_1.json", b'{"words": []}')  # a blob, despite .json
    store.put_doc("files", "f1", {"id": "f1"})  # -> files/f1/file.json (a document)
    # blobs (incl. page_text) are listed; only the file.json manifest is excluded
    assert store.list_blobs("files/f1/") == [
        "files/f1/page_images/page_1.png",
        "files/f1/page_text/page_1.json",
        "files/f1/report.pdf",
    ]


def test_delete_blobs_is_blob_only_and_prunes(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_blob("files/f1/report.pdf", b"pdf")
    store.put_blob("files/f1/page_images/page_1.png", b"png")
    store.put_doc("files", "f1", {"id": "f1"})
    store.put_blob("files/f2/report.pdf", b"other")

    store.delete_blobs("files/f1/")
    # blobs under the prefix are gone; the document beside them is untouched
    assert store.list_blobs("files/f1/") == []
    assert store.get_doc("files", "f1") == {"id": "f1"}
    # the emptied blob subdir is pruned; the file dir stays (still holds file.json)
    assert not (tmp_path / "files" / "f1" / "page_images").exists()
    assert (tmp_path / "files" / "f1").is_dir()
    # a sibling under the same parent is untouched; a missing prefix is a no-op
    assert store.get_blob("files/f2/report.pdf") == b"other"
    store.delete_blobs("files/nope/")

    # once the document is gone too, delete_blobs prunes the now-empty file dir
    store.delete_doc("files", "f1")
    store.delete_blobs("files/f1/")
    assert not (tmp_path / "files" / "f1").exists()


def test_delete_blobs_preserves_assignments(tmp_path: Path) -> None:
    """A blob-only delete must never take the assignment with it.

    This used to need an explicit prune guard, because an *empty* pair directory
    was itself the assignment. Now the assignment is a document that lives in
    that directory, so it survives for the ordinary reason: ``delete_blobs``
    does not touch documents, and the directory is not empty."""
    store = local_store(tmp_path)
    store.put_doc("assignments", "d1/f1", {"docset_id": "d1", "file_id": "f1"})
    store.put_blob("docsets/d1/files/f1/report.dgml.xml", b"<x/>")
    store.delete_blobs("docsets/d1/files/f1/")
    assert not store.blob_exists("docsets/d1/files/f1/report.dgml.xml")
    assert store.get_doc("assignments", "d1/f1") == {"docset_id": "d1", "file_id": "f1"}
    assert (tmp_path / "docsets" / "d1" / "files" / "f1").is_dir()


def test_upload_download_blob(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    store.upload_blob("files/d/e.bin", src)
    assert store.get_blob("files/d/e.bin") == b"payload"
    dest = tmp_path / "out" / "dl.bin"
    store.download_blob("files/d/e.bin", dest)
    assert dest.read_bytes() == b"payload"


def test_download_blob_missing_key_raises(tmp_path: Path) -> None:
    """DGMLX export stages every leaf through ``download_blob``, so a blob that
    vanished between listing and export must surface as ``FileNotFoundError``
    — the same contract ``get_blob`` gave that code before."""
    store = local_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.download_blob("files/d/gone.bin", tmp_path / "out" / "dl.bin")


def test_download_blob_overwrites_existing_dest(tmp_path: Path) -> None:
    """Re-exporting an unpacked bundle over an older one must replace it, not
    append or fail — ``write_bytes`` used to guarantee this."""
    store = local_store(tmp_path)
    store.put_blob("files/d/e.bin", b"new")
    dest = tmp_path / "out" / "dl.bin"
    dest.parent.mkdir()
    dest.write_bytes(b"stale-and-longer")
    store.download_blob("files/d/e.bin", dest)
    assert dest.read_bytes() == b"new"


def test_blob_key_traversal_rejected(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    for bad in ("/abs/key", "../escape", "a/../../b"):
        with pytest.raises(ValueError):
            store.put_blob(bad, b"x")


# ----------------------------------------------------------------------- documents


def test_manifest_stored_verbatim_at_legacy_path(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    record = {"id": "f1", "sha256": "aa", "page_count": 2}
    store.put_doc("files", "f1", record)
    # byte-for-byte the FileRecord JSON — no injected ``_id``, at the real path
    on_disk = json.loads((tmp_path / "files" / "f1" / "file.json").read_text(encoding="utf-8"))
    assert on_disk == record
    assert store.get_doc("files", "f1") == record


def test_doc_put_get_update(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    assert store.get_doc("files", "f1") is None
    store.put_doc("files", "f1", {"id": "f1", "sha256": "aa"})
    assert store.get_doc("files", "f1") == {"id": "f1", "sha256": "aa"}
    # put replaces (update)
    store.put_doc("files", "f1", {"id": "f1", "sha256": "bb"})
    assert store.get_doc("files", "f1") == {"id": "f1", "sha256": "bb"}


def test_docset_and_workspace_collections(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_doc("docsets", "d1", {"id": "d1", "name": "Contracts"})
    assert (tmp_path / "docsets" / "d1" / "docset.json").is_file()
    store.put_doc("workspace", "workspace", {"name": "W", "organization": "Acme"})
    assert json.loads((tmp_path / "workspace.json").read_text())["organization"] == "Acme"


def _assign(store: LocalStore, docset_id: str, file_id: str) -> None:
    store.put_doc(
        "assignments",
        f"{docset_id}/{file_id}",
        {"docset_id": docset_id, "file_id": file_id},
    )


def test_assignments_are_documents(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    _assign(store, "d1", "f1")
    _assign(store, "d1", "f2")
    _assign(store, "d2", "f1")
    # on disk it's a real document in the pair directory, not the directory itself
    pair = tmp_path / "docsets" / "d1" / "files" / "f1"
    assert json.loads((pair / "assignment.json").read_text()) == {
        "docset_id": "d1",
        "file_id": "f1",
    }
    assert store.get_doc("assignments", "d1/f1") == {"docset_id": "d1", "file_id": "f1"}
    assert store.get_doc("assignments", "d1/nope") is None
    # both relationship directions are queryable
    assert sorted(d["file_id"] for d in store.find_docs("assignments", {"docset_id": "d1"})) == [
        "f1",
        "f2",
    ]
    assert sorted(d["docset_id"] for d in store.find_docs("assignments", {"file_id": "f1"})) == [
        "d1",
        "d2",
    ]
    assert len(store.find_docs("assignments", {})) == 3
    # deleting the record drops the document, and the emptied pair dir with it
    store.delete_doc("assignments", "d1/f1")
    assert not pair.exists()
    assert len(store.find_docs("assignments", {})) == 2


def test_assignment_body_is_stored_not_reconstructed(tmp_path: Path) -> None:
    """A real document can carry fields the path cannot encode (assigned_at,
    and later provenance) — the whole point of not being a marker directory."""
    store = local_store(tmp_path)
    body = {"docset_id": "d1", "file_id": "f1", "assigned_at": "2026-07-31T00:00:00Z"}
    store.put_doc("assignments", "d1/f1", body)
    assert store.get_doc("assignments", "d1/f1") == body
    assert store.find_docs("assignments", {"assigned_at": "2026-07-31T00:00:00Z"}) == [body]


def test_delete_doc_assignment_leaves_pair_artifacts(tmp_path: Path) -> None:
    """Deleting the assignment deletes the record and nothing else.

    The old marker-directory representation made this impossible: ``delete_doc``
    had to ``rmtree`` the pair directory, so a single-document delete silently
    took the generated dgml.xml and extraction_stats with it. Removing the
    pair's artifacts is now the caller's explicit job (``Workspace.unassign``),
    which is what makes the cascade behave the same on every backend."""
    store = local_store(tmp_path)
    _assign(store, "d1", "f1")
    store.put_blob("docsets/d1/files/f1/report.dgml.xml", b"<x/>")
    store.put_doc("extraction_stats", "d1/f1", {"matched": 3})

    store.delete_doc("assignments", "d1/f1")

    assert store.get_doc("assignments", "d1/f1") is None
    assert store.get_blob("docsets/d1/files/f1/report.dgml.xml") == b"<x/>"
    assert store.get_doc("extraction_stats", "d1/f1") == {"matched": 3}


def test_assignment_json_is_not_a_blob(tmp_path: Path) -> None:
    """``assignment.json`` must stay out of the blob namespace.

    ``collect_file_version`` hashes ``list_blobs`` output into the attestation
    Merkle tree, so a document leaking into it would change on-chain roots."""
    store = local_store(tmp_path)
    _assign(store, "d1", "f1")
    store.put_blob("docsets/d1/files/f1/report.dgml.xml", b"<x/>")
    assert store.list_blobs("docsets/d1/files/f1/") == ["docsets/d1/files/f1/report.dgml.xml"]


def test_bare_pair_directory_is_not_an_assignment(tmp_path: Path) -> None:
    """A directory alone records nothing — only the document does.

    Earlier revisions treated the bare existence of ``docsets/<did>/files/<fid>/``
    as the assignment. That representation cannot survive its own deletion: once
    ``delete_doc`` removes the record, a pair directory still holding generated
    artifacts is indistinguishable from an assignment that was never deleted, so
    the delete silently un-does itself. Recognising bare directories is therefore
    not a compatibility shim we can keep — it is a correctness hole."""
    store = local_store(tmp_path)
    (tmp_path / "docsets" / "d1" / "files" / "f1").mkdir(parents=True)
    assert store.get_doc("assignments", "d1/f1") is None
    assert store.find_docs("assignments", {}) == []


def test_append_doc_rejects_an_addressable_collection(tmp_path: Path) -> None:
    """``append_doc`` is only for the append-only log; documents addressed by id
    go through ``put_doc``, which is the only create path."""
    store = local_store(tmp_path)
    with pytest.raises(InvalidArgument, match="append-only"):
        store.append_doc("files", {"sha256": "aa"})


def test_delete_doc_and_delete_docs(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_doc("files", "f1", {"id": "f1"})
    store.put_doc("files", "f2", {"id": "f2"})
    store.delete_doc("files", "f1")
    assert store.get_doc("files", "f1") is None
    store.delete_doc("files", "missing")  # no error

    for did, fid in [("d1", "f1"), ("d1", "f2"), ("d2", "f1")]:
        _assign(store, did, fid)
    removed = store.delete_docs("assignments", {"docset_id": "d1"})
    assert removed == 2
    assert len(store.find_docs("assignments", {})) == 1


def test_usage_is_append_only(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.append_doc("usage", {"op": "transcribe", "cost_usd": 0.01})
    store.append_doc("usage", {"op": "label", "cost_usd": 0.02})
    events = store.find_docs("usage", {})
    assert [e["op"] for e in events] == ["transcribe", "label"]
    assert "_id" not in events[0]  # stored verbatim as JSONL
    assert (tmp_path / "usage.jsonl").exists()
    assert store.find_docs("usage", {"op": "label"}) == [{"op": "label", "cost_usd": 0.02}]
    assert store.delete_docs("usage", {"op": "transcribe"}) == 1
    assert [e["op"] for e in store.find_docs("usage", {})] == ["label"]


def test_usage_tolerates_corrupt_tail(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.append_doc("usage", {"op": "ok"})
    with (tmp_path / "usage.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"op": "truncated"')  # crashed mid-append, no newline / close brace
    assert [e["op"] for e in store.find_docs("usage", {})] == ["ok"]


# ------------------------------------------------------------------ resolver/config


def test_make_store_default_is_local(tmp_path: Path) -> None:
    cfg = StorageConfig(provider=DEFAULT_STORAGE_PROVIDER, root=tmp_path)
    blobs = make_blob_store(cfg)
    docs = make_doc_store(cfg)
    assert isinstance(blobs, LocalStore) and isinstance(blobs, BlobStore)
    assert isinstance(docs, LocalStore) and isinstance(docs, DocStore)


def test_make_store_bad_provider() -> None:
    for bad in ["noColon", "no.module.here.at.all:Class", "json:Nonexistent"]:
        with pytest.raises(StorageProviderUnresolvable):
            make_blob_store(StorageConfig(provider=bad, root=Path(".")))


def test_make_store_not_a_storage_subclass() -> None:
    # importable + resolvable, but not a BlobStore / DocStore
    with pytest.raises(StorageProviderUnresolvable):
        make_blob_store(StorageConfig(provider="json:JSONDecoder", root=Path(".")))
    with pytest.raises(StorageProviderUnresolvable):
        make_doc_store(StorageConfig(provider="json:JSONDecoder", root=Path(".")))


def test_local_store_rejects_unknown_options(tmp_path: Path) -> None:
    with pytest.raises(StorageConfigInvalid):
        LocalStore.parse_config(
            StorageConfig(provider=DEFAULT_STORAGE_PROVIDER, root=tmp_path, options={"bucket": "x"})
        )


def test_load_store_configs_defaults_to_local(tmp_path: Path) -> None:
    # No [storage] section → both roles on the bundled local-disk default.
    ws = Workspace.resolve(tmp_path)
    blob_cfg, doc_cfg = load_store_configs(ws)
    assert blob_cfg.provider == doc_cfg.provider == DEFAULT_STORAGE_PROVIDER
    assert blob_cfg.root == doc_cfg.root == ws.root
    assert blob_cfg.options == doc_cfg.options == {}


def test_load_store_configs_flat_serves_both_roles(tmp_path: Path) -> None:
    from .conftest import write_config

    ws = Workspace.resolve(tmp_path)
    write_config(ws, {"storage": {"provider": "my_pkg.store:MyStore", "bucket": "b1"}})
    blob_cfg, doc_cfg = load_store_configs(ws)
    for cfg in (blob_cfg, doc_cfg):
        assert cfg.provider == "my_pkg.store:MyStore"
        assert cfg.options == {"bucket": "b1"}
    # A bare [storage] table cannot also name other services.
    with pytest.raises(StorageConfigInvalid):
        load_store_configs(ws, "svcA")


def test_load_store_configs_invalid_provider(tmp_path: Path) -> None:
    from .conftest import write_config

    ws = Workspace.resolve(tmp_path)
    write_config(ws, {"storage": {"provider": ""}})
    with pytest.raises(StorageConfigInvalid):
        load_store_configs(ws)


def test_load_store_configs_per_role_subtables(tmp_path: Path) -> None:
    from .conftest import write_config

    ws = Workspace.resolve(tmp_path)
    write_config(
        ws,
        {
            "storage": {
                "mix": {
                    "blobs": {"provider": "pkg:S3", "bucket": "b"},
                    "docs": {"provider": "pkg:Mongo", "mongo_database": "d"},
                }
            }
        },
    )
    blob_cfg, doc_cfg = load_store_configs(ws, "mix")
    assert (blob_cfg.provider, blob_cfg.options) == ("pkg:S3", {"bucket": "b"})
    assert (doc_cfg.provider, doc_cfg.options) == ("pkg:Mongo", {"mongo_database": "d"})


def test_load_store_configs_omitted_role_falls_back_to_local(tmp_path: Path) -> None:
    from .conftest import write_config

    ws = Workspace.resolve(tmp_path)
    # Only the blob role is a named backend; docs default to local disk.
    write_config(ws, {"storage": {"mix": {"blobs": {"provider": "pkg:S3", "bucket": "b"}}}})
    blob_cfg, doc_cfg = load_store_configs(ws, "mix")
    assert blob_cfg.provider == "pkg:S3"
    assert doc_cfg.provider == DEFAULT_STORAGE_PROVIDER


def test_load_store_configs_rejects_mixed_form(tmp_path: Path) -> None:
    from .conftest import write_config

    ws = Workspace.resolve(tmp_path)
    write_config(
        ws,
        {"storage": {"mix": {"provider": "pkg:Both", "blobs": {"provider": "pkg:S3"}}}},
    )
    with pytest.raises(StorageConfigInvalid):
        load_store_configs(ws, "mix")


def test_load_store_configs_missing_named_service_raises(tmp_path: Path) -> None:
    from .conftest import write_config

    ws = Workspace.resolve(tmp_path)
    write_config(ws, {"storage": {"svcA": {"provider": "my_pkg:A"}}})
    with pytest.raises(StorageConfigInvalid):
        load_store_configs(ws, "nope")
    # ...but an absent "default" in named form still falls back to local disk.
    blob_cfg, doc_cfg = load_store_configs(ws, "default")
    assert blob_cfg.provider == doc_cfg.provider == DEFAULT_STORAGE_PROVIDER


# ---------------------------------------------------------- snapshot / resolution


def test_storage_snapshot_drops_secrets_and_round_trips_fingerprint() -> None:
    from dgml_core.storage_resolve import fingerprint_of_snapshot, storage_snapshot

    cfg = StorageConfig(
        provider="p:C",
        root=Path("/tmp/ws"),
        options={"bucket": "b", "region": "us-east-1", "secret_key": "SHH", "api_token": "T"},
    )
    snap = storage_snapshot(cfg)
    assert snap == {"provider": "p:C", "bucket": "b", "region": "us-east-1"}  # secrets dropped
    # The snapshot's fingerprint reproduces the config's fingerprint (integrity seal).
    assert fingerprint_of_snapshot(snap) == storage_fingerprint(cfg)


def test_fingerprint_pair_over_two_snapshots() -> None:
    from dgml_core.storage_resolve import fingerprint_pair, snapshot_pair

    blob = StorageConfig(provider="pkg:S3", root=Path("/w"), options={"bucket": "b"})
    doc = StorageConfig(provider="pkg:Mongo", root=Path("/w"), options={"mongo_database": "d"})
    pair = snapshot_pair(blob, doc)
    assert pair == {
        "blobs": {"provider": "pkg:S3", "bucket": "b"},
        "docs": {"provider": "pkg:Mongo", "mongo_database": "d"},
    }
    assert fingerprint_pair(pair).startswith("sha256:")
    # Empty pair reads as unsealed (trust-on-first-use).
    assert fingerprint_pair({}) == ""
    # Swapping the two backends changes the fingerprint (order matters).
    assert fingerprint_pair(pair) != fingerprint_pair(snapshot_pair(doc, blob))


def test_resolve_store_configs_unregistered_is_local(tmp_path: Path) -> None:
    from dgml_core.registry import resolve_store_configs

    ws = Workspace.resolve(tmp_path)  # not in the registry
    blob_cfg, doc_cfg = resolve_store_configs(ws)
    assert blob_cfg.provider == doc_cfg.provider == DEFAULT_STORAGE_PROVIDER
    assert blob_cfg.root == doc_cfg.root == ws.root


def _same_instance(blobs: object, docs: object) -> bool:
    """Whether the two roles are the same object. Takes ``object`` because
    ``BlobStore`` and ``DocStore`` are unrelated ABCs, so a direct ``is`` reads to
    mypy as a non-overlapping identity check — sharing is exactly the case where
    one object satisfies both."""
    return blobs is docs


def test_same_backend_for_both_roles_is_one_instance(tmp_path: Path) -> None:
    """A provider serving both roles is constructed once, so it holds one
    connection per workspace rather than one per role.

    Sharing keys off config *equality*, so it covers the two ways a config can
    name one backend: the zero-config default, and two per-role tables that are
    identical. Identical config means an identical backend either way."""
    from .conftest import write_config

    flat = Workspace.resolve(tmp_path / "flat")
    flat.root.mkdir()
    assert _same_instance(flat.blobs, flat.docs)  # zero-config: both roles are LocalStore

    per_role = Workspace.resolve(tmp_path / "per-role")
    per_role.root.mkdir()
    write_config(
        per_role,
        {
            "storage": {
                "default": {
                    "blobs": {"provider": DEFAULT_STORAGE_PROVIDER},
                    "docs": {"provider": DEFAULT_STORAGE_PROVIDER},
                }
            }
        },
    )
    assert _same_instance(per_role.blobs, per_role.docs)


def test_split_backends_stay_distinct_and_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sharing must not cost laziness. When the two roles resolve to *different*
    backends, touching one role never constructs the other — otherwise a
    docs-only command on a split-provider workspace would open the blob store's
    SDK client and credentials for nothing."""
    import dgml_core.storage_resolve as storage_resolve

    from .conftest import write_config

    ws = Workspace.resolve(tmp_path)
    write_config(
        ws,
        {
            "storage": {
                "default": {
                    "blobs": {"provider": f"{DefaultBridgeStore.__module__}:DefaultBridgeStore"},
                    "docs": {"provider": DEFAULT_STORAGE_PROVIDER},
                }
            }
        },
    )

    built = 0
    real_make = storage_resolve.make_blob_store

    def counting_make(config: object) -> object:
        nonlocal built
        built += 1
        return real_make(config)  # type: ignore[arg-type]

    monkeypatch.setattr(storage_resolve, "make_blob_store", counting_make)

    docs = ws.docs
    assert isinstance(docs, LocalStore) and not isinstance(docs, DefaultBridgeStore)
    assert built == 0  # the blob role was never resolved

    assert isinstance(ws.blobs, DefaultBridgeStore)
    assert built == 1
    assert ws.blobs is not ws.docs


def test_store_configs_resolve_once_for_both_roles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registry + config resolution is per workspace, not per role. It re-reads
    ``workspaces.json`` once per registry entry and rebuilds a settings class on
    every call, so doing it twice was pure duplicated parsing."""
    import dgml_core.registry as registry

    calls = 0
    real_resolve = registry.resolve_store_configs

    def counting_resolve(ws: Workspace) -> object:
        nonlocal calls
        calls += 1
        return real_resolve(ws)

    monkeypatch.setattr(registry, "resolve_store_configs", counting_resolve)

    ws = Workspace.resolve(tmp_path)
    assert (ws.blobs, ws.docs) == (ws.blobs, ws.docs)
    assert calls == 1


# --------------------------------------------------------------------- fingerprint


def test_fingerprint_stable_and_location_sensitive() -> None:
    root = Path("/tmp/ws")
    a = StorageConfig(provider="p:C", root=root, options={"bucket": "b", "prefix": "x"})
    a2 = StorageConfig(provider="p:C", root=Path("/other"), options={"prefix": "x", "bucket": "b"})
    b = StorageConfig(provider="p:C", root=root, options={"bucket": "OTHER", "prefix": "x"})
    # stable across option order and independent of the local root
    assert storage_fingerprint(a) == storage_fingerprint(a2)
    # trips when the location changes
    assert storage_fingerprint(a) != storage_fingerprint(b)


def test_fingerprint_ignores_credential_rotation() -> None:
    root = Path("/tmp/ws")
    a = StorageConfig(provider="p:C", root=root, options={"bucket": "b", "api_key": "OLD"})
    b = StorageConfig(provider="p:C", root=root, options={"bucket": "b", "api_key": "NEW"})
    assert storage_fingerprint(a) == storage_fingerprint(b)


def test_third_party_plugin_resolves_by_dotted_path() -> None:
    # dgml_core.storage_local:LocalStore is resolved exactly like a third party's
    # own dotted path — proving the plug-in mechanism end to end.
    cfg = StorageConfig(provider="dgml_core.storage_local:LocalStore", root=Path("."))
    assert isinstance(make_blob_store(cfg), LocalStore)
    assert isinstance(make_doc_store(cfg), LocalStore)


# --------------------------------------------------------------------------- path bridge


def test_materialize_local_yields_real_path_zero_copy(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_blob("files/a/report.pdf", b"%PDF-1.7")
    with store.materialize("files/a/report.pdf") as path:
        # zero copy: it's the actual on-disk blob, not a temp copy
        assert path == tmp_path / "files" / "a" / "report.pdf"
        assert path.read_bytes() == b"%PDF-1.7"


def test_materialize_missing_raises(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        with store.materialize("files/a/nope.pdf"):
            pass


def test_materialize_default_downloads_to_temp_and_cleans_up(tmp_path: Path) -> None:
    store = default_bridge_store(tmp_path)
    store.put_blob("files/a/report.pdf", b"%PDF-1.7")
    with store.materialize("files/a/report.pdf") as path:
        # default impl: a temp copy, NOT the real blob path
        assert path != tmp_path / "files" / "a" / "report.pdf"
        assert path.read_bytes() == b"%PDF-1.7"
        held = path
    assert not held.exists()  # cleaned up on exit


_PAGES = "files/a/page_images"

# staged_write's contract has to hold identically on a zero-copy local store and
# on one that stages through temp files, so each case below runs against both.
_BOTH_STORES: list[Callable[[Path], LocalStore]] = [local_store, default_bridge_store]


@pytest.mark.parametrize("make_store_", _BOTH_STORES)
def test_staged_write_yields_an_empty_dir_and_persists_on_exit(
    tmp_path: Path, make_store_: Callable[[Path], LocalStore]
) -> None:
    store = make_store_(tmp_path)
    store.put_blob(f"{_PAGES}/page_1.png", b"old")
    with store.staged_write(_PAGES) as d:
        assert list(d.iterdir()) == []  # never the live destination
        (d / "page_1.png").write_bytes(b"img1")
        (d / "page_2.png").write_bytes(b"img2")
    assert store.get_blob(f"{_PAGES}/page_1.png") == b"img1"
    assert store.get_blob(f"{_PAGES}/page_2.png") == b"img2"


@pytest.mark.parametrize("make_store_", _BOTH_STORES)
def test_staged_write_replaces_rather_than_adds(
    tmp_path: Path, make_store_: Callable[[Path], LocalStore]
) -> None:
    """Re-render a document whose page count dropped: the extra pages must go.

    A purely additive implementation leaves page_3..5 behind, and because
    ``collect_file_version`` hashes ``list_blobs(page_images)`` those phantom
    pages would enter the attestation — making the Merkle root depend on which
    backend the workspace happens to live on."""
    store = make_store_(tmp_path)
    with store.staged_write(_PAGES) as d:
        for n in range(1, 6):
            (d / f"page_{n}.png").write_bytes(f"v1-{n}".encode())
    assert len(store.list_blobs(_PAGES + "/")) == 5

    with store.staged_write(_PAGES) as d:
        for n in range(1, 3):
            (d / f"page_{n}.png").write_bytes(f"v2-{n}".encode())

    assert store.list_blobs(_PAGES + "/") == [f"{_PAGES}/page_1.png", f"{_PAGES}/page_2.png"]
    assert store.get_blob(f"{_PAGES}/page_1.png") == b"v2-1"


@pytest.mark.parametrize("make_store_", _BOTH_STORES)
def test_staged_write_does_not_persist_on_error(
    tmp_path: Path, make_store_: Callable[[Path], LocalStore]
) -> None:
    """A failed render leaves the previous content untouched — it must not
    half-clobber the prefix it was about to replace."""
    store = make_store_(tmp_path)
    store.put_blob(f"{_PAGES}/page_1.png", b"old")
    with pytest.raises(RuntimeError), store.staged_write(_PAGES) as d:
        (d / "page_1.png").write_bytes(b"new")
        (d / "page_2.png").write_bytes(b"new")
        raise RuntimeError("render failed")
    assert store.list_blobs(_PAGES + "/") == [f"{_PAGES}/page_1.png"]
    assert store.get_blob(f"{_PAGES}/page_1.png") == b"old"


@pytest.mark.parametrize("make_store_", _BOTH_STORES)
def test_staged_write_preserves_documents_under_the_prefix(
    tmp_path: Path, make_store_: Callable[[Path], LocalStore]
) -> None:
    """Replacement is scoped to blobs; a document sharing the prefix survives."""
    store = make_store_(tmp_path)
    store.put_doc("extraction_stats", "d1/f1", {"matched": 3})
    with store.staged_write("docsets/d1/files/f1") as d:
        (d / "report.dgml.xml").write_bytes(b"<x/>")
    assert store.get_doc("extraction_stats", "d1/f1") == {"matched": 3}
    assert store.get_blob("docsets/d1/files/f1/report.dgml.xml") == b"<x/>"


def test_staged_write_parity_between_backends(tmp_path: Path) -> None:
    """The same sequence of renders must leave both stores holding the same
    keys — the property W10's cross-backend parity test will rely on."""

    def run(store: LocalStore) -> list[str]:
        with store.staged_write(_PAGES) as d:
            for n in range(1, 4):
                (d / f"page_{n}.png").write_bytes(b"v1")
        with store.staged_write(_PAGES) as d:
            (d / "page_1.png").write_bytes(b"v2")
        return store.list_blobs(_PAGES + "/")

    assert run(local_store(tmp_path / "local")) == run(default_bridge_store(tmp_path / "bridge"))


def test_staged_write_scratch_is_not_visible_as_blobs(tmp_path: Path) -> None:
    """LocalStore stages inside the workspace so the hand-off is a rename; that
    scratch must stay out of the blob namespace (as must the embedding cache)."""
    store = local_store(tmp_path)
    with store.staged_write(_PAGES) as d:
        (d / "page_1.png").write_bytes(b"img1")
        assert store.list_blobs("") == []  # mid-flight staging is invisible
    (tmp_path / ".cache" / "embeddings").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".cache" / "embeddings" / "e.npy").write_bytes(b"x")
    assert store.list_blobs("") == [f"{_PAGES}/page_1.png"]


def test_coverage_report_round_trips_through_local_store(tmp_path: Path) -> None:
    """The --debug coverage report is a docset-level blob: LocalStore must accept
    the write, list it, and delete it with the prefix (regression — the allow-list
    didn't recognise the key, so put_blob rejected it)."""
    from dgml_core import layout

    store = local_store(tmp_path)
    key = layout.docset_coverage_report_key("d1")
    store.put_blob(key, b'{"coverage": 0.9}')  # would raise INVALID_ARGUMENT before the fix
    assert store.get_blob(key) == b'{"coverage": 0.9}'
    assert key in store.list_blobs("docsets/d1/")  # visible to readers
    store.delete_blobs("docsets/d1/")
    assert not store.blob_exists(key)  # and cleaned up with the docset


def test_staged_write_cleans_up_empty_scratch_dir(tmp_path: Path) -> None:
    """LocalStore stages under ``.cache/staging/``; a successful render must not
    leave that scratch parent behind as an empty directory."""
    store = local_store(tmp_path)
    with store.staged_write(_PAGES) as d:
        (d / "page_1.png").write_bytes(b"img1")
        assert (tmp_path / ".cache" / "staging").is_dir()  # holds the temp dir mid-flight
    assert not (tmp_path / ".cache" / "staging").exists()  # removed once empty on exit
    assert store.get_blob(f"{_PAGES}/page_1.png") == b"img1"  # results still persisted


def test_materialize_dir_local_yields_real_dir_zero_copy(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_blob("files/a/page_images/page_1.png", b"img1")
    store.put_blob("files/a/page_images/page_2.png", b"img2")
    with store.materialize_dir("files/a/page_images") as d:
        assert d == tmp_path / "files" / "a" / "page_images"  # zero copy
        assert sorted(p.name for p in d.glob("*.png")) == ["page_1.png", "page_2.png"]


def test_materialize_dir_default_downloads_and_cleans_up(tmp_path: Path) -> None:
    store = default_bridge_store(tmp_path)
    store.put_blob("files/a/page_images/page_1.png", b"img1")
    store.put_blob("files/a/page_images/page_2.png", b"img2")
    with store.materialize_dir("files/a/page_images") as d:
        assert d != tmp_path / "files" / "a" / "page_images"  # a temp copy
        assert (d / "page_1.png").read_bytes() == b"img1"
        assert (d / "page_2.png").read_bytes() == b"img2"
        held = d
    assert not held.exists()  # cleaned up on exit


def test_working_dir_local_is_in_place_zero_copy(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    store.put_blob("docsets/d1/cache/existing.json", b"old")
    with store.working_dir("docsets/d1/cache") as work:
        assert work == tmp_path / "docsets" / "d1" / "cache"  # real dir, in place
        assert (work / "existing.json").read_bytes() == b"old"
        (work / "new.json").write_bytes(b"new")
    # writes land in the store directly, no sync step needed
    assert store.get_blob("docsets/d1/cache/new.json") == b"new"


def test_working_dir_default_syncs_down_and_up(tmp_path: Path) -> None:
    store = default_bridge_store(tmp_path)
    store.put_blob("docsets/d1/cache/existing.json", b"old")
    with store.working_dir("docsets/d1/cache") as work:
        # a temp working copy whose parent is stable scratch (siblings can live there)
        assert work != tmp_path / "docsets" / "d1" / "cache"
        assert work.name == "cache"
        assert (work / "existing.json").read_bytes() == b"old"  # downloaded in
        (work / "new.json").write_bytes(b"new")
        (work.parent / "schema.json").write_bytes(b"{}")  # a sibling, not synced
        held = work
    assert not held.exists()  # temp cleaned up
    assert store.get_blob("docsets/d1/cache/new.json") == b"new"  # uploaded out
    assert not store.blob_exists("docsets/d1/schema.json")  # sibling not uploaded


@pytest.mark.parametrize("make_store_", _BOTH_STORES)
def test_working_dir_syncs_deletions_out(
    tmp_path: Path, make_store_: Callable[[Path], LocalStore]
) -> None:
    """A read-modify-write area syncs *removals* too — an evicted cache entry
    must not be resurrected on the next run by an upload-only sync."""
    store = make_store_(tmp_path)
    store.put_blob("docsets/d1/cache/keep.json", b"keep")
    store.put_blob("docsets/d1/cache/evict.json", b"evict")
    with store.working_dir("docsets/d1/cache") as work:
        (work / "evict.json").unlink()
        (work / "added.json").write_bytes(b"added")
    assert store.list_blobs("docsets/d1/cache/") == [
        "docsets/d1/cache/added.json",
        "docsets/d1/cache/keep.json",
    ]


# --------------------------------------------------------------------------- sha256_blob

# A payload spanning several CHUNK_SIZE reads, with a deliberate non-multiple
# tail, so a store that mishandled chunk boundaries would produce a different
# digest than the one-shot reference. Deterministic on purpose: a hash mismatch
# has to be reproducible from the same bytes to be debuggable.
_MULTI_CHUNK = bytes(range(256)) * ((CHUNK_SIZE * 2) // 256) + b"tail" * 4 + b"!"


# Both path-bridge implementations: LocalStore's zero-copy override and the base
# download-to-temp default that every third-party store inherits.
StoreFactory = Callable[[Path], LocalStore]
STORE_FACTORIES: list[StoreFactory] = [local_store, default_bridge_store]


@pytest.mark.parametrize("make_store_", STORE_FACTORIES)
def test_sha256_blob_is_plain_sha256_of_the_exact_stored_bytes(
    tmp_path: Path, make_store_: StoreFactory
) -> None:
    """The conformance test for ``sha256_blob``'s documented contract.

    Attestation leaves — and therefore the Merkle roots anchored on chain — are
    built from this digest, so it must equal the plain SHA-256 of the whole byte
    sequence and never a derived checksum (an S3 multipart ETag or composite
    ``ChecksumSHA256`` is a checksum-of-checksums and would silently produce
    divergent roots). Asserted for both the zero-copy ``LocalStore`` path and the
    base download-to-temp path that every third-party store inherits.
    """
    store = make_store_(tmp_path)
    key = "files/a/report.pdf"
    store.put_blob(key, _MULTI_CHUNK)
    expected = hashlib.sha256(_MULTI_CHUNK).hexdigest()
    assert store.sha256_blob(key) == expected
    # ...and equals hashing the get_blob result, the API it replaces.
    assert store.sha256_blob(key) == hashlib.sha256(store.get_blob(key)).hexdigest()


@pytest.mark.parametrize("make_store_", STORE_FACTORIES)
def test_sha256_blob_handles_empty_blob(tmp_path: Path, make_store_: StoreFactory) -> None:
    store = make_store_(tmp_path)
    store.put_blob("files/a/empty.bin", b"")
    assert store.sha256_blob("files/a/empty.bin") == hashlib.sha256(b"").hexdigest()


@pytest.mark.parametrize("make_store_", STORE_FACTORIES)
def test_sha256_blob_missing_raises(tmp_path: Path, make_store_: StoreFactory) -> None:
    store = make_store_(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.sha256_blob("files/a/nope.pdf")


@pytest.mark.parametrize("bridge", [LocalStore, DefaultBridgeStore])
def test_sha256_blob_does_not_load_the_blob_whole(tmp_path: Path, bridge: type[LocalStore]) -> None:
    """The point of the method: no whole-blob allocation. A store whose
    ``get_blob`` is a landmine still hashes fine — both through ``LocalStore``'s
    zero-copy ``materialize`` and through the inherited download-to-temp default,
    so neither path can quietly regress to a whole read."""

    class _NoGetBlob(bridge):  # type: ignore[valid-type,misc]
        def get_blob(self, key: str) -> bytes:
            raise AssertionError(f"get_blob called for {key!r}")

    store = _NoGetBlob(LocalStore.parse_config(StorageConfig(DEFAULT_STORAGE_PROVIDER, tmp_path)))
    store.put_blob("files/a/report.pdf", _MULTI_CHUNK)
    assert store.sha256_blob("files/a/report.pdf") == hashlib.sha256(_MULTI_CHUNK).hexdigest()


# ----------------------------------------------------------- list_blobs scope


def test_list_blobs_prefix_matches_are_boundary_correct(tmp_path: Path) -> None:
    """Scanning only the prefix's subtree must not change *which* keys match —
    in particular a sibling whose name extends the prefix stays excluded."""
    store = local_store(tmp_path)
    store.put_blob("files/f1/page_images/page_1.png", b"a")
    store.put_blob("files/f10/page_images/page_1.png", b"b")
    store.put_blob("docsets/d1/full-schema.rnc", b"c")

    assert store.list_blobs("files/f1/") == ["files/f1/page_images/page_1.png"]
    assert store.list_blobs("files/f10/") == ["files/f10/page_images/page_1.png"]
    assert len(store.list_blobs("files/")) == 2
    assert len(store.list_blobs("")) == 3


def test_list_blobs_prefix_naming_a_dir_still_matches_extending_siblings(tmp_path: Path) -> None:
    """The narrowing bug this guards against.

    ``files/f1`` (no trailing slash) is a raw string prefix, so S3 semantics
    match ``files/f1x/...`` as well as ``files/f1/...``. Scanning only the
    directory the prefix happens to name would silently drop the sibling —
    a short list, not an error. Only a trailing slash makes the prefix
    unambiguous, which is why that is the single-directory fast path.

    Not reachable through a current caller (ids are fixed-length, so no id can
    prefix another), but the contract is what the next caller will rely on."""
    store = local_store(tmp_path)
    store.put_blob("files/f1/a.pdf", b"a")
    store.put_blob("files/f1x/b.pdf", b"b")

    assert store.list_blobs("files/f1") == ["files/f1/a.pdf", "files/f1x/b.pdf"]
    assert store.list_blobs("files/f1/") == ["files/f1/a.pdf"]  # slash = that dir only


def test_list_blobs_matches_a_full_key_as_a_prefix(tmp_path: Path) -> None:
    """A prefix can name a file rather than a directory."""
    store = local_store(tmp_path)
    store.put_blob("files/f1/page_images/page_1.png", b"a")
    store.put_blob("files/f1/page_images/page_10.png", b"b")
    key = "files/f1/page_images/page_1.png"
    assert store.list_blobs(key) == [key]
    assert store.list_blobs("files/f1/page_images/page_1") == [key, key.replace("_1.", "_10.")]


def test_list_blobs_accepts_a_partial_segment_prefix(tmp_path: Path) -> None:
    """An object store matches keys by raw string prefix, including mid-segment;
    narrowing the walk must not quietly drop that."""
    store = local_store(tmp_path)
    store.put_blob("files/abc/report.pdf", b"a")
    store.put_blob("files/abd/report.pdf", b"b")
    assert store.list_blobs("files/ab") == ["files/abc/report.pdf", "files/abd/report.pdf"]
    assert store.list_blobs("files/abc") == ["files/abc/report.pdf"]
    assert store.list_blobs("files/zz") == []
    assert store.list_blobs("nosuchdir/") == []


def test_list_blobs_does_not_walk_outside_the_prefix(tmp_path: Path) -> None:
    """The point of the change: unrelated subtrees are never visited."""
    store = local_store(tmp_path)
    store.put_blob("files/f1/report.pdf", b"a")
    for n in range(50):
        store.put_blob(f"files/other{n}/page_images/page_1.png", b"x")

    visited: list[Path] = []
    real_rglob = Path.rglob

    def counting_rglob(self: Path, pattern: str):  # type: ignore[no-untyped-def]
        visited.append(self)
        return real_rglob(self, pattern)

    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setattr(Path, "rglob", counting_rglob)
    try:
        assert store.list_blobs("files/f1/") == ["files/f1/report.pdf"]
    finally:
        mp.undo()
    assert visited == [tmp_path / "files" / "f1"]
