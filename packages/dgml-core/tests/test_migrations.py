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

"""Tests for workspace schema migrations."""

from __future__ import annotations

from pathlib import Path

import pytest
from dgml_core.docsets import DocSetStore
from dgml_core.errors import WorkspaceMigrationFailed
from dgml_core.migrations import (
    WORKSPACE_SCHEMA_VERSION,
    migrate_workspace,
    pending_migrations,
    stamp_schema_version,
    workspace_schema_version,
)
from dgml_core.storage import Workspace


def _legacy_assignment(ws: Workspace, docset_id: str, file_id: str) -> Path:
    """An assignment as it was stored before assignment.json: a bare directory.

    Inherently LocalStore-specific — the migration exists to upgrade a legacy
    on-disk layout, so this reaches the real directory via the kept
    ``docsets_dir`` property (there is no store-API way to make an empty dir)."""
    pair = ws.docsets_dir / docset_id / "files" / file_id
    pair.mkdir(parents=True, exist_ok=True)
    return pair


def test_unstamped_workspace_reads_as_version_zero(workspace: Workspace) -> None:
    assert workspace_schema_version(workspace) == 0
    # A single revision while the storage layer is unmerged (see migrations module).
    assert [m.version for m in pending_migrations(workspace)] == [1]


def test_stamp_preserves_other_meta_fields(workspace: Workspace) -> None:
    workspace.write_meta(name="W", organization="Acme")
    stamp_schema_version(workspace)
    meta = workspace.read_meta()
    assert meta["schema_version"] == WORKSPACE_SCHEMA_VERSION
    assert (meta["name"], meta["organization"]) == ("W", "Acme")
    assert workspace.organization == "Acme"  # the namespace source still resolves


def test_current_workspace_has_nothing_pending(workspace: Workspace) -> None:
    stamp_schema_version(workspace)
    assert pending_migrations(workspace) == []
    assert migrate_workspace(workspace) == []


def test_migration_upgrades_legacy_assignments(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    for fid in ("aaaaaaaaaaaa", "bbbbbbbbbbbb"):
        workspace.docs.put_doc("files", fid, {"id": fid})
        _legacy_assignment(workspace, ds.id, fid)
    # a generated artifact in one pair — the migration must not disturb it
    workspace.blobs.put_blob(f"docsets/{ds.id}/files/aaaaaaaaaaaa/r.dgml.xml", b"<x/>")

    assert store.list_files(ds.id) == []  # invisible before migrating

    migrate_workspace(workspace)

    assert store.list_files(ds.id) == ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
    assert workspace.docs.get_doc("assignments", f"{ds.id}/aaaaaaaaaaaa") == {
        "docset_id": ds.id,
        "file_id": "aaaaaaaaaaaa",
    }
    assert workspace.blobs.get_blob(f"docsets/{ds.id}/files/aaaaaaaaaaaa/r.dgml.xml") == b"<x/>"
    # The same v1 migration also backfilled the workspace_id (store-agnostic part).
    assert workspace.workspace_id is not None
    assert workspace_schema_version(workspace) == WORKSPACE_SCHEMA_VERSION


def test_migration_is_idempotent(workspace: Workspace) -> None:
    """Re-running must not duplicate or overwrite: concurrent CLI invocations
    can both reach the migration, and a crash must be resumable."""
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    fid = "aaaaaaaaaaaa"
    workspace.docs.put_doc("files", fid, {"id": fid})
    _legacy_assignment(workspace, ds.id, fid)

    migrate_workspace(workspace)
    assert store.list_files(ds.id) == [fid]
    # replay from scratch: the version stamp is what normally short-circuits, so
    # force the migration to run a second time against already-migrated data — with
    # both parts idempotent (id present, assignment already a document) it changes
    # nothing.
    stamp_schema_version(workspace, 0)
    assert migrate_workspace(workspace)[0].changed == 0
    assert store.list_files(ds.id) == [fid]


def test_migration_does_not_touch_existing_assignment_documents(workspace: Workspace) -> None:
    """An assignment written by the current code keeps its body — in particular
    its assigned_at — rather than being flattened by the migration."""
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    fid = "aaaaaaaaaaaa"
    workspace.docs.put_doc("files", fid, {"id": fid})
    store.add_file(ds.id, fid)
    before = workspace.docs.get_doc("assignments", f"{ds.id}/{fid}")
    assert before is not None and before["assigned_at"]

    stamp_schema_version(workspace, 0)
    migrate_workspace(workspace)
    # The migration leaves an assignment document written by current code untouched
    # (in particular it keeps its assigned_at) rather than flattening it.
    assert workspace.docs.get_doc("assignments", f"{ds.id}/{fid}") == before


def test_migration_ignores_non_pair_directories(workspace: Workspace) -> None:
    """Only ``docsets/<did>/files/<fid>/`` is an assignment; a docset directory
    or a stray nested directory must not become one."""
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    (workspace.docsets_dir / ds.id / "files").mkdir(parents=True, exist_ok=True)
    (workspace.docsets_dir / ds.id / "scratch").mkdir(parents=True, exist_ok=True)

    migrate_workspace(workspace)
    assert workspace.docs.find_docs("assignments", {}) == []


def test_empty_workspace_migrates_cleanly(workspace: Workspace) -> None:
    migrate_workspace(workspace)
    assert workspace_schema_version(workspace) == WORKSPACE_SCHEMA_VERSION
    assert workspace.docs.find_docs("assignments", {}) == []  # nothing to upgrade


def test_backfill_workspace_id_mints_and_is_idempotent(workspace: Workspace) -> None:
    """The store-agnostic part of the v1 migration gives a pre-id workspace a stable
    id, then is a no-op once present."""
    assert workspace.workspace_id is None
    migrate_workspace(workspace)
    wid = workspace.workspace_id
    assert wid is not None and wid.startswith("ws_")

    # Re-run against already-migrated data: no new id, none minted, nothing changes.
    stamp_schema_version(workspace, 0)
    assert migrate_workspace(workspace)[0].changed == 0
    assert workspace.workspace_id == wid  # unchanged


def test_unwritable_workspace_fails_loudly(workspace: Workspace, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A read-only workspace must raise, not silently serve incomplete data:
    un-migrated assignments would list as empty, which reads as a real answer."""
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    _legacy_assignment(workspace, ds.id, "aaaaaaaaaaaa")

    import dgml_core.storage_local as storage_local

    def _boom(path: Path, text: str) -> None:
        raise PermissionError(f"read-only file system: {path}")

    monkeypatch.setattr(storage_local, "_write_text_atomic", _boom)
    with pytest.raises(WorkspaceMigrationFailed, match="writable"):
        migrate_workspace(workspace)


# ------------------------------------- moving the storage binding into config.toml


_LOCAL = "dgml_core.storage_local:LocalStore"


def _legacy_row(ws: Workspace, *, service: str, snapshot: dict[str, object]) -> str:
    """Write a pre-upgrade index row — the shape an older dgml wrote, carrying the
    binding inline."""
    import json

    from dgml_core import registry

    wid = "ws_legacyxxxxxxxxxx"
    registry.registry_path().parent.mkdir(parents=True, exist_ok=True)
    registry.registry_path().write_text(
        json.dumps(
            {
                wid: {
                    "name": "W",
                    "organization": "Acme",
                    "root": str(ws.root),
                    "storage_service": service,
                    "storage": snapshot,
                    "storage_fingerprint": "sha256:deadbeef",
                    "created_at": "2026-08-05T12:00:00Z",
                    "schema_version": 1,
                }
            }
        )
    )
    return wid


def test_seeds_storage_from_the_legacy_snapshot_not_the_template(tmp_path: Path) -> None:
    """**The highest-risk case in this change.** The old resolver pinned a workspace to
    its registry snapshot, so a later edit to the ``[storage.<name>]`` template was
    never in effect. Seeding from the template instead would silently relocate the
    workspace to a different backend and orphan every file already stored."""
    from dgml_core import workspace_config
    from dgml_core.migrations import migrate_workspace_config
    from dgml_core.storage_resolve import resolve_store_configs

    ws = Workspace(root=tmp_path / "ws")
    ws.root.mkdir()
    # The template says one thing...
    ws.config_path.write_text(f'[storage.svcA]\nprovider = "{_LOCAL}"\nprefix = "EDITED-AWAY"\n')
    # ...the snapshot (what the workspace actually ran on) says another.
    _legacy_row(
        ws, service="svcA", snapshot={"blobs": {"provider": _LOCAL}, "docs": {"provider": _LOCAL}}
    )

    assert migrate_workspace_config(ws) == 1

    table = workspace_config.read_storage_table(ws, "svcA")
    assert table is not None
    assert "prefix" not in table["blobs"], "template leaked into the migrated binding"
    blob_cfg, _ = resolve_store_configs(ws)
    assert "prefix" not in blob_cfg.options


def test_migration_seals_and_records_identity(tmp_path: Path) -> None:
    from dgml_core import workspace_config
    from dgml_core.migrations import migrate_workspace_config
    from dgml_core.storage_resolve import resolve_store_configs, storage_fingerprint_pair

    ws = Workspace(root=tmp_path / "ws")
    ws.root.mkdir()
    wid = _legacy_row(ws, service="default", snapshot={"blobs": {"provider": _LOCAL}})
    migrate_workspace_config(ws)

    identity = workspace_config.read_identity(ws)
    assert identity.workspace_id == wid
    assert identity.organization == "Acme"
    assert identity.storage_service == "default"
    assert identity.storage_fingerprint == storage_fingerprint_pair(*resolve_store_configs(ws))


def test_config_migration_is_idempotent(tmp_path: Path) -> None:
    from dgml_core.migrations import migrate_workspace_config

    ws = Workspace(root=tmp_path / "ws")
    ws.root.mkdir()
    _legacy_row(ws, service="default", snapshot={"blobs": {"provider": _LOCAL}})

    assert migrate_workspace_config(ws) == 1
    first = ws.config_path.read_text()
    assert migrate_workspace_config(ws) == 0
    assert ws.config_path.read_text() == first


def test_migration_leaves_the_legacy_index_untouched(tmp_path: Path) -> None:
    """The migration copies *out* of the legacy index and never writes to it.

    It used to rewrite the row to strip the now-powerless `storage` keys. That is no
    longer worth doing: nothing resolves through the file any more, rewriting it would
    make a dead file look maintained, and `dgml workspace import` needs to read it
    exactly as the older dgml left it."""
    import json

    from dgml_core import registry
    from dgml_core.migrations import migrate_workspace_config

    ws = Workspace(root=tmp_path / "ws")
    ws.root.mkdir()
    wid = _legacy_row(ws, service="default", snapshot={"blobs": {"provider": _LOCAL}})
    before = registry.registry_path().read_bytes()

    assert migrate_workspace_config(ws) == 1

    assert registry.registry_path().read_bytes() == before
    row = json.loads(registry.registry_path().read_text())[wid]
    assert row["storage"] == {"blobs": {"provider": _LOCAL}}
    # ...and the binding it described now lives in the workspace's own config.
    assert "[storage.default.blobs]" in (ws.config_text or "")


def test_migration_never_invents_a_config(tmp_path: Path) -> None:
    """No legacy row *and* no config file: creating one would be a guess, and the
    wrong one for a workspace whose remote config was deleted — it would be re-sealed
    onto the local default while its data sits elsewhere."""
    from dgml_core.migrations import migrate_workspace_config

    ws = Workspace(root=tmp_path / "ws")
    ws.root.mkdir()
    assert migrate_workspace_config(ws) == 0
    assert not ws.config_path.exists()


def test_migration_seals_a_pre_existing_config_with_no_legacy_row(tmp_path: Path) -> None:
    """A workspace already resolving from its own config has nothing to move, but is
    still sealed — otherwise it would stay permanently unguarded."""
    from dgml_core import workspace_config
    from dgml_core.migrations import migrate_workspace_config

    ws = Workspace(root=tmp_path / "ws")
    ws.root.mkdir()
    ws.config_path.write_text(f'[storage]\nprovider = "{_LOCAL}"\n')
    assert migrate_workspace_config(ws) == 1
    assert workspace_config.read_identity(ws).storage_fingerprint is not None


def test_migration_folds_a_flat_pre_split_snapshot(tmp_path: Path) -> None:
    """Entries written before the blob/doc split carried one flat snapshot serving
    both roles."""
    from dgml_core import workspace_config
    from dgml_core.migrations import migrate_workspace_config

    ws = Workspace(root=tmp_path / "ws")
    ws.root.mkdir()
    _legacy_row(ws, service="default", snapshot={"provider": _LOCAL, "prefix": "p"})
    migrate_workspace_config(ws)

    table = workspace_config.read_storage_table(ws, "default")
    assert table is not None
    assert table["blobs"]["provider"] == table["docs"]["provider"] == _LOCAL
    assert table["blobs"]["prefix"] == "p"
