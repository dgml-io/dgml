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
from typing import Any

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


def _snapshot_to_table(storage: dict[str, Any]) -> dict[str, Any] | None:
    """A legacy registry ``storage`` snapshot as a ``[storage.<service>]`` table body.

    The snapshot is ``{"blobs": {provider, …}, "docs": {…}}``, or — for entries written
    before the blob/doc split — one flat ``{provider, …}`` serving both roles. A role
    whose snapshot names no provider is **omitted**, so it falls back to the bundled
    local store exactly as the old resolver's own missing-provider branch did.
    Returns ``None`` when neither role is usable."""
    if isinstance(storage.get("provider"), str):
        storage = {"blobs": storage, "docs": storage}
    table: dict[str, Any] = {}
    for role in ("blobs", "docs"):
        role_snapshot = storage.get(role)
        if isinstance(role_snapshot, dict) and isinstance(role_snapshot.get("provider"), str):
            table[role] = dict(role_snapshot)
    return table or None


def _legacy_str(entry: dict[str, Any], key: str) -> str | None:
    """One string field of a legacy index row, or ``None`` if absent or the wrong type."""
    value = entry.get(key)
    return value if isinstance(value, str) and value else None


def migrate_workspace_config(ws: Workspace, *, assume_local_when_unbound: bool = False) -> int:
    """Move a legacy per-machine storage binding into the workspace's own ``config.toml``.

    **Store-free, and deliberately not a** :data:`_MIGRATIONS` **entry.** Migration
    dispatch reads ``workspace.json`` *through the store*, and the store is the very
    thing this repairs — so it cannot be version-dispatched without a chicken-and-egg.
    It is guarded on content instead: a workspace whose config already carries a
    ``storage_fingerprint`` returns immediately. That also makes it correct for a
    workspace a development build already stamped at the current version, which a
    registered migration would silently skip.

    The binding is seeded from the **registry snapshot**, never from the merged config.
    A workspace created with ``--storage svcA`` whose ``[storage.svcA]`` template was
    later edited is pinned by the old resolver to the snapshot; seeding from the
    template instead would silently relocate it to a different backend and orphan its
    data.

    A workspace with no legacy row is already resolving from its ``config.toml``, so
    there is nothing to move — but it is still sealed, so the drift guard covers
    workspaces that predate it rather than leaving them permanently unguarded.

    ``assume_local_when_unbound`` covers the workspace with **no** recorded binding
    anywhere — old enough that the index never carried one, or one whose ``config.toml``
    was deleted. Those two cases are indistinguishable, and in the second the binding is
    unrecoverable regardless, so refusing preserves nothing. It still defaults to
    ``False``: on the per-command path, inventing a local binding would seal the
    workspace to local disk and write the next file there, starting a second corpus while
    reporting success. ``dgml workspace import`` passes ``True``, because a caller
    explicitly adopting a directory needs an outcome they can act on — and it reports the
    assumption rather than making it quietly.

    Idempotent, and safe to interrupt: both writes replace whole tables, so a crash
    between them leaves a workspace that is re-migrated byte-identically. Returns 1
    when it wrote, 0 otherwise."""
    from . import registry, workspace_config
    from .errors import WorkspaceMigrationFailed
    from .storage_resolve import (
        DEFAULT_STORAGE_PROVIDER,
        DEFAULT_STORAGE_SERVICE,
        resolve_store_configs,
        storage_fingerprint_pair,
    )

    if workspace_config.read_identity(ws).storage_fingerprint:
        return 0

    legacy = registry.raw_entry_by_root(ws.root) or {}
    raw_storage = legacy.get("storage")
    table = _snapshot_to_table(raw_storage) if isinstance(raw_storage, dict) else None
    service = legacy.get("storage_service")
    if not isinstance(service, str) or not service:
        service = DEFAULT_STORAGE_SERVICE

    assumed_local = False
    if table is None and not ws.config_present:
        # Nothing recorded anywhere: a workspace old enough that the index never carried
        # a binding, or one whose config.toml was deleted. The two are indistinguishable
        # from here, and in the second case the binding is unrecoverable either way —
        # nothing on this machine remembers which backend that file named.
        if not assume_local_when_unbound:
            # The per-command path takes the cautious branch. Inventing a binding on an
            # ordinary `file add` would seal the workspace to local disk and write the
            # file there, so a workspace whose config was deleted would start a second,
            # split corpus while reporting success. Leaving it absent makes the caller
            # report a missing config, which is the truth.
            return 0
        # `workspace import` takes the other branch, because the caller is explicitly
        # adopting this directory and a refusal would leave them nothing to act on. Local
        # disk is the only binding that could have existed for a workspace predating the
        # storage layer, and the assumption is reported rather than made quietly.
        table = {
            "blobs": {"provider": DEFAULT_STORAGE_PROVIDER},
            "docs": {"provider": DEFAULT_STORAGE_PROVIDER},
        }
        assumed_local = True

    try:
        if table is not None:
            assumed_banner = (
                "# Written by dgml on import. This workspace recorded no storage binding\n"
                "# anywhere — it predates the binding being recorded, or its config.toml\n"
                "# was lost — so local disk was assumed, the only backend it could have\n"
                "# used at the time. If its data is actually on a remote backend, change\n"
                "# this table and run `dgml workspace reseal`.\n"
            )
            migrated_banner = (
                "# Migrated by dgml from this machine's workspace index — it records the\n"
                "# backend this workspace's data is already on. If you had edited this\n"
                "# table before upgrading, that edit was never in effect (the old index\n"
                "# pinned the workspace to the snapshot above) and is not preserved here.\n"
                "# To move the data, change this table and run `dgml workspace reseal`.\n"
            )
            workspace_config.write_storage_table(
                ws,
                service,
                table,
                banner=assumed_banner if assumed_local else migrated_banner,
            )
        # Resolve directly rather than through ``ws.store_configs``: the table was just
        # written, and the cached property must not memoize a pre-write pair.
        fingerprint = storage_fingerprint_pair(*resolve_store_configs(ws))
        workspace_config.write_identity(
            ws,
            workspace_id=_legacy_str(legacy, "workspace_id"),
            name=_legacy_str(legacy, "name"),
            organization=_legacy_str(legacy, "organization"),
            storage_service=service,
            storage_fingerprint=fingerprint,
        )
    except OSError as exc:
        raise WorkspaceMigrationFailed(
            f"could not write {ws.config_location} while moving this workspace's storage "
            f"binding out of the machine registry: {exc}"
        ) from exc

    # The legacy index row is deliberately left exactly as it was. It is no longer
    # written or resolved through (see :mod:`dgml_core.registry`), so rewriting it would
    # only make a dead file look maintained — and ``dgml workspace import`` still needs
    # to read it as the older dgml left it.
    return 1


def _mirror_identity_into_config(ws: Workspace) -> int:
    """Copy ``workspace_id``/``organization`` from ``workspace.json`` into the
    ``[workspace]`` block, so both can be read without opening the store.

    Writes only what is absent — :func:`migrate_workspace_config` already supplies them
    for a workspace that had a registry row, and this covers the rest (a freshly
    backfilled id, or a workspace that was never indexed).

    The two copies are deliberately **never compared**. ``config.toml`` is the
    store-free bootstrap copy and ``workspace.json`` the one that travels with the
    data; treating a difference as an error would turn a restored backup or a
    hand-copied config into a hard failure with no good repair."""
    from . import workspace_config

    identity = workspace_config.read_identity(ws)
    if identity.workspace_id and identity.name and identity.organization:
        return 0
    workspace_config.write_identity(
        ws,
        workspace_id=identity.workspace_id or ws.workspace_id,
        name=identity.name or ws.display_name,
        organization=identity.organization or ws.organization,
    )
    return 1


def _backfill_workspace_id(ws: Workspace) -> int:
    """Give a pre-id workspace a stable ``workspace_id`` in ``workspace.json``.

    Store-AGNOSTIC — reads/writes ``workspace.json`` only through the store's
    document API (`read_meta`/`write_meta`), so it runs on **every** backend (a
    remote workspace needs an id too), and does no disk globbing. Idempotent: a
    no-op once an id is present. Because it must run everywhere, it lives outside
    the LocalStore guard below (calling it unconditionally is the point).
    """
    from .workspace_id import mint_workspace_id

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
    - :func:`_mirror_identity_into_config` — store-free write of that id (and the
      organization) into ``config.toml``; ordered after the backfill so a
      freshly-minted id is mirrored in the same pass.
    - :func:`_upgrade_assignments_to_documents` — LocalStore-only (self-guarded).

    These are **one** migration deliberately: separate ``Migration`` entries at the
    same version would break crash-resume (the version is stamped after each, so a
    crash between them would make the next run skip the rest). Keep the store-agnostic
    parts OUT of the LocalStore-guarded upgrade — call them as siblings so they still
    run on a remote store. When the storage layer is released, further changes get
    their own version 2, not more code here.

    Note the storage binding itself is *not* migrated here — see
    :func:`migrate_workspace_config`, which must run store-free and ahead of dispatch.
    """
    return (
        _backfill_workspace_id(ws)
        + _mirror_identity_into_config(ws)
        + _upgrade_assignments_to_documents(ws)
    )


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
