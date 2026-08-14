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
