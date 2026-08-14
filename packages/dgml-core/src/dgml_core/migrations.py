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

"""Workspace schema migrations — bring an existing workspace up to date.

A workspace records the layout revision it was written against as
``schema_version`` in ``workspace.json``. :func:`migrate_workspace` applies
every registered migration above that number, in order, and stamps the new
version. The CLI runs it once per command, immediately after resolving the
workspace, so an older workspace upgrades itself the first time any command
touches it — there is no ``dgml migrate`` to remember and no flag day.

Design rules for anything added to :data:`_MIGRATIONS`:

- **Idempotent.** Two processes may run the same migration concurrently, and a
  crash must leave a state the next run can finish. Write only what is absent.
- **Additive where possible.** A migration that only adds files can be replayed
  and cannot corrupt a workspace it has already visited.
- **Silent when there is nothing to do.** The common case is a current
  workspace: that path must cost one document read.
- **Store-agnostic in signature.** A migration may target a single backend (the
  first one repairs a layout that only ever existed on local disk) but it must
  no-op cleanly on the others rather than raise.

While the storage layer is still an unmerged branch, **fold new data changes
into migration 1 rather than adding a migration 2**. Users upgrade from
released ``dgml`` straight to the merged result, so no workspace will ever sit
at an intermediate revision, and a chain of migrations mirroring our commit
order would be dead code the day it shipped. Add a new version only for a
change that lands *after* the storage layer is released.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .storage import Workspace

# Bump when a migration is added; new workspaces are stamped with this.
WORKSPACE_SCHEMA_VERSION = 1

# Where the stamp lives: workspace.json's ``schema_version``. A workspace with
# no workspace.json at all (they predate it) reads as version 0.
_VERSION_FIELD = "schema_version"


@dataclass(frozen=True)
class Migration:
    """One ordered, idempotent upgrade step."""

    version: int
    name: str
    description: str
    apply: Callable[[Workspace], int]
    """Perform the upgrade; return how many items were changed (0 = nothing
    to do). Must be safe to call on an already-migrated workspace."""


@dataclass(frozen=True)
class MigrationResult:
    migration: Migration
    changed: int

    def summary(self) -> str:
        return f"{self.migration.name}: {self.changed} item(s) migrated"


def _backfill_workspace_id(ws: Workspace) -> int:
    """Give a pre-id workspace a stable ``workspace_id`` in ``workspace.json``.

    Store-AGNOSTIC — reads/writes ``workspace.json`` only through the store's
    document API (`read_meta`/`write_meta`), so it runs on **every** backend (a
    remote workspace needs an id too), and does no disk globbing. Idempotent: a
    no-op once an id is present. Because it must run everywhere, it lives outside
    the LocalStore guard below (calling it unconditionally is the point).
    """
    from .registry import mint_workspace_id

    if ws.workspace_id is not None:
        return 0
    ws.write_meta(
        name=ws.display_name, organization=ws.organization, workspace_id=mint_workspace_id()
    )
    return 1


def _upgrade_assignments_to_documents(ws: Workspace) -> int:
    """Give every pre-``assignment.json`` DocSet assignment a real document.

    Before the storage layer an assignment *was* the bare directory
    ``docsets/<did>/files/<fid>/``. That representation cannot survive its own
    deletion — once the record is removed, a pair directory still holding
    generated artifacts is indistinguishable from a live assignment — so bare
    directories are no longer read as assignments and have to be upgraded.

    Directory-as-record only ever existed on local disk, so this **self-guards**
    on ``LocalStore`` and is a no-op on any other backend. Writes only the
    documents that are missing, and touches nothing else in the pair directory.
    """
    from .layout import ASSIGNMENT_MANIFEST, DOCSET_FILES_DIR, Collection, pair_id
    from .storage_local import LocalStore

    store = ws.docs  # LocalStore is both a BlobStore and a DocStore
    if not isinstance(store, LocalStore):
        return 0

    migrated = 0
    for pair_dir in sorted(ws.docsets_dir.glob(f"*/{DOCSET_FILES_DIR}/*")):
        if not pair_dir.is_dir() or (pair_dir / ASSIGNMENT_MANIFEST).is_file():
            continue
        docset_id, file_id = pair_dir.parent.parent.name, pair_dir.name
        store.put_doc(
            Collection.ASSIGNMENTS,
            pair_id(docset_id, file_id),
            {"docset_id": docset_id, "file_id": file_id},
        )
        migrated += 1
    return migrated


def _migrate_to_v1(ws: Workspace) -> int:
    """The single revision the whole (still-unmerged) storage layer ships as.

    Per this module's "fold into migration 1 while unmerged" rule, every 0→1 data
    change lives in this one step rather than a chain of intermediate versions —
    released ``dgml`` (version 0) upgrades straight to the merged result. It bundles
    two independent, individually-idempotent parts:

    - :func:`_backfill_workspace_id` — store-agnostic; runs on every backend.
    - :func:`_upgrade_assignments_to_documents` — LocalStore-only (self-guarded).

    Both are **one** migration deliberately: two separate ``Migration`` entries at
    the same version would break crash-resume (the version is stamped after each,
    so a crash between them would make the next run skip the second). Keep the
    store-agnostic backfill OUT of the LocalStore-guarded upgrade — call the two as
    siblings so the backfill still runs on a remote store. When the storage layer is
    released, further changes get their own version 2, not more code here.
    """
    return _backfill_workspace_id(ws) + _upgrade_assignments_to_documents(ws)


_MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="storage-layer-v1",
        description=(
            "Backfill workspace_id (all backends) and upgrade bare-directory DocSet "
            "assignments to documents (LocalStore)"
        ),
        apply=_migrate_to_v1,
    ),
)


def workspace_schema_version(ws: Workspace) -> int:
    """The layout revision ``ws`` was last written against (0 if unstamped)."""
    value = ws.read_meta().get(_VERSION_FIELD)
    return value if isinstance(value, int) else 0


def stamp_schema_version(ws: Workspace, version: int = WORKSPACE_SCHEMA_VERSION) -> None:
    """Record ``version`` in ``workspace.json``, preserving its other fields.

    Called for a freshly created workspace so it is never mistaken for an old
    one, and after each migration so a crash mid-sequence resumes correctly."""
    from .layout import Collection

    meta = dict(ws.read_meta())
    meta[_VERSION_FIELD] = version
    ws.docs.put_doc(Collection.WORKSPACE, Collection.WORKSPACE, meta)


def pending_migrations(ws: Workspace) -> list[Migration]:
    """Registered migrations newer than the workspace's recorded version."""
    current = workspace_schema_version(ws)
    return [m for m in _MIGRATIONS if m.version > current]


def migrate_workspace(ws: Workspace) -> list[MigrationResult]:
    """Bring ``ws`` up to :data:`WORKSPACE_SCHEMA_VERSION`.

    Returns a result per migration that ran (empty when already current, which
    is the common path and costs a single document read). The version is
    stamped after each step, so an interrupted run resumes rather than
    restarting. Raises :class:`~dgml_core.errors.WorkspaceMigrationFailed` if a
    migration cannot be written — an un-migrated workspace reads as valid but
    incomplete, so this must not be swallowed.
    """
    from .errors import WorkspaceMigrationFailed

    pending = pending_migrations(ws)
    if not pending:
        return []

    results: list[MigrationResult] = []
    for migration in pending:
        try:
            changed = migration.apply(ws)
            stamp_schema_version(ws, migration.version)
        except OSError as exc:
            raise WorkspaceMigrationFailed(
                f"could not apply workspace migration {migration.name!r} to {ws.root}: {exc}. "
                "The workspace needs to be writable to be upgraded."
            ) from exc
        results.append(MigrationResult(migration=migration, changed=changed))
    return results
