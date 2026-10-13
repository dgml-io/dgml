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

"""Creating a workspace: claim an id, bind storage, seal, stamp.

Idempotent and safe to re-run, including from a second machine against a shared
store of workspaces — which is most of why the order below is what it is.
"""

from __future__ import annotations

import contextlib
import tomllib
from dataclasses import dataclass
from typing import Any

from . import workspace_config as wsconfig
from .errors import (
    ConflictError,
    CorruptMetadata,
    InvalidArgument,
    WorkspacesWriteConflict,
    now_iso,
)
from .migrations import stamp_schema_version
from .storage import Workspace
from .storage_resolve import (
    DEFAULT_STORAGE_SERVICE,
    check_store_configs,
    load_store_configs,
    resolve_service_configs,
    storage_fingerprint_pair,
)
from .workspace_config import WorkspaceIdentity
from .workspace_id import ID_SHAPE, generate_unique_workspace_id, is_workspace_id
from .workspaces_resolve import default_workspaces_store
from .workspaces_store import WorkspacesStore

__all__ = ["CreateWorkspaceResult", "create_workspace"]


@dataclass(frozen=True)
class CreateWorkspaceResult:
    """What :func:`create_workspace` did.

    The two ``*_changed_from`` fields report a re-run that **re-identified** an
    existing workspace: they are facts, not warnings, so a caller decides whether
    and how loudly to say so. Both are destructive in a way that only shows up
    later — a changed organization splits the corpus across two namespaces, and a
    changed storage service leaves earlier artifacts on the old backend.
    """

    workspace: Workspace
    identity: WorkspaceIdentity
    organization_changed_from: str | None = None
    storage_service_changed_from: str | None = None


def _validate_seed_config(text: str) -> None:
    """Reject a seed config that cannot mean what it says.

    A ``[workspaces]`` table selects the machine's store of workspaces and is read
    only from the user config, so here it would be silently inert — worse than an
    error, because it looks like it redirects where workspaces are listed.
    """
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise InvalidArgument(f"seed config is not valid TOML: {exc}") from exc
    if "workspaces" in parsed:
        raise InvalidArgument(
            "seed config declares a [workspaces] table. That table selects the machine's "
            "store of workspaces and is read only from the user config, so it would have "
            "no effect here. Remove it."
        )


def _already_held(store: WorkspacesStore, workspace_id: str) -> ConflictError:
    """The refusal for an id the store already holds — one wording however it is hit."""
    return ConflictError(
        f"{store.label()} already holds a workspace {workspace_id}. If it is this "
        f"workspace, re-run create against it — create_workspace("
        f"Workspace.resolve({workspace_id!r})) is safe to re-run. Otherwise pick "
        f"another id.",
        kind="workspace",
        existing_id=workspace_id,
    )


def _declared_stands(declared: Any, existing: Any) -> bool:
    """Whether every key ``declared`` sets holds the declared value in ``existing``,
    recursively. Keys only ``existing`` has don't count against it — ``write_identity``
    adds machine-managed ``[workspace]`` keys beside whatever a seed declared."""
    if isinstance(declared, dict) and isinstance(existing, dict):
        return all(_declared_stands(v, existing.get(k)) for k, v in declared.items())
    return bool(declared == existing)


def _seed_already_applied(ws: Workspace, seed_toml: str) -> bool:
    """Whether everything ``seed_toml`` declares already stands in ``ws``'s config — the
    re-run of a seeded create, which must stay a no-op."""
    try:
        existing = tomllib.loads(ws.config_text or "")
    except tomllib.TOMLDecodeError as exc:
        # The same failure every other reader of the config reports.
        raise CorruptMetadata(f"invalid TOML in {ws.config_location}: {exc}") from exc
    return _declared_stands(tomllib.loads(seed_toml), existing)


def _materialize_storage_table(ws: Workspace, service: str, *, seeded: bool) -> None:
    """Give a new workspace the ``[storage.<service>]`` table it resolves from.

    A seeded workspace keeps its ``[storage]`` exactly as written; otherwise the
    named service is copied down from the user-level config so the workspace is
    self-describing from the moment it exists. Never clobbers a table the config
    already defines."""
    if seeded or wsconfig.read_storage_table(ws, service) is not None:
        return
    blob_cfg, doc_cfg = load_store_configs(ws, service)
    table: dict[str, Any] = {
        "blobs": {"provider": blob_cfg.provider, **dict(blob_cfg.options)},
        "docs": {"provider": doc_cfg.provider, **dict(doc_cfg.options)},
    }
    wsconfig.write_storage_table(ws, service, table)


def create_workspace(
    workspace: Workspace | None = None,
    *,
    workspace_id: str | None = None,
    organization: str | None = None,
    name: str | None = None,
    storage_service: str | None = None,
    seed_toml: str | None = None,
) -> CreateWorkspaceResult:
    """Create (or re-create) a workspace and return what was done.

    ``workspace`` names an existing root to build into — a *detached* workspace,
    living at a path. Passing ``None`` instead puts the workspace in the machine's
    store of workspaces, under ``workspace_id`` or a generated one.

    ``seed_toml`` is a **template**: its text becomes the new workspace's config
    and the source is then forgotten, so later edits to it have no effect.

    ``organization`` is required for a genuinely new workspace and optional once
    the config records one — it is embedded in every docset namespace URI, so
    re-typing it is how a typo re-organizes an entire corpus.

    Safe to re-run: an id, name, organization and storage service already
    recorded in the config win over anything derived locally. A ``workspace_id``
    that *disagrees* with the recorded one is refused rather than discarded —
    ``create`` never re-identifies a workspace — and one another workspace already
    holds raises :class:`~dgml_core.errors.ConflictError`, since proceeding would
    overwrite that workspace's config in the store.
    """
    if seed_toml is not None:
        _validate_seed_config(seed_toml)
    if workspace_id is not None and not is_workspace_id(workspace_id):
        raise InvalidArgument(
            f"workspace_id {workspace_id!r} is not a well-formed workspace id: it must be "
            f"{ID_SHAPE}."
        )

    # Every id check runs before anything is written: a rejected id must not leave a
    # half-built workspace behind, and for a listed one the id decides the root.
    #
    # The store of workspaces is built only on the two paths that consult it. A
    # detached create with no id never does, and must keep working when the
    # configured [workspaces] provider cannot even be imported on this machine.
    if workspace is None:
        store = default_workspaces_store()
        if workspace_id is not None and store.exists(workspace_id):
            raise ConflictError(
                f"{store.label()} already holds a workspace {workspace_id}. If it is this "
                f"workspace, re-run create against it — create_workspace("
                f"Workspace.resolve({workspace_id!r})) is safe to re-run. Otherwise pick "
                f"another id.",
                kind="workspace",
                existing_id=workspace_id,
            )
        # The id comes first: for a listed workspace the root is derived from it —
        # the reverse of the detached order.
        new_id = workspace_id or generate_unique_workspace_id(store)
        seed = seed_toml or ""
        try:
            # The claim. Create-if-absent, so of two creates racing the same id exactly
            # one wins and the loser is refused here instead of overwriting the row.
            store.create_config(new_id, seed)
        except WorkspacesWriteConflict:
            raise _already_held(store, new_id) from None
        except BaseException:
            # The claim's ack can be lost after the server applied it, which would
            # strand a row that raises ConflictError on every retry of the same id. But
            # the same error can also mean the insert never applied, and then any row
            # under this id is another writer's. The two are told apart by reading the
            # row back: this call has written nothing but the seed, so a row that still
            # reads as the seed is its own. That check is only sound right here, before
            # the build adds identity and storage to the config — after that the text
            # never equals the seed, and the build's rollback below must not use it.
            # One ambiguity remains: another writer racing the same id with
            # byte-identical seed text (which could be empty) reads as this call's row,
            # so its fresh claim can be deleted here. Bounded: their unfinished create
            # fails with a retryable conflict at its next conditional write, and a row
            # that got any further no longer reads as the seed — a built workspace is
            # never deleted.
            with contextlib.suppress(Exception):
                if store.read_config(new_id) == seed:
                    store.delete(new_id)
            raise
        ws = Workspace(root=store.workspace_root(new_id), workspaces_id=new_id)
        try:
            return _build(ws, workspace_id, organization, name, storage_service, seed_toml)
        except WorkspacesWriteConflict:
            # Another writer changed the row after this call claimed it — it is theirs
            # now, and deleting it would destroy their workspace.
            raise
        except BaseException:
            # Any other failure must remove the row this call claimed, and a cleanup
            # that fails must not mask the cause.
            with contextlib.suppress(Exception):
                store.delete(new_id)
            raise
    else:
        ws = workspace
        if workspace_id is not None:
            # What this workspace is already called: the id it is listed under, else
            # the one its own config records. An id that agrees is a re-run; one that
            # disagrees would re-identify the workspace; one nobody records yet must
            # still be free in the store, or a detached workspace would shadow a listed
            # one at `--workspace <id>`.
            known = ws.workspaces_id or wsconfig.read_identity(ws).workspace_id
            if known is not None and known != workspace_id:
                raise InvalidArgument(
                    f"workspace_id {workspace_id!r} does not match {known!r}, the id this "
                    f"workspace already records. create never re-identifies a workspace; "
                    f"to make a new one called {workspace_id!r}, create it somewhere else."
                )
            if known is None:
                store = default_workspaces_store()
                if store.exists(workspace_id):
                    raise _already_held(store, workspace_id)
    wrote_seed = False
    made_root = not ws.root.exists()
    if seed_toml is not None:
        if not (ws.config_text or "").strip():
            ws.root.mkdir(parents=True, exist_ok=True)
            wsconfig.write_config_text(ws, seed_toml)
            wrote_seed = True
        elif not _seed_already_applied(ws, seed_toml):
            # Refused, never silently dropped: ignoring it would bind through the user
            # config instead of the backend the seed names.
            raise InvalidArgument(
                f"{ws.config_location} already has a config, and it differs from the seed "
                f"config. A seed only initializes a workspace with no config; it never "
                f"replaces one. Edit that config directly, or re-run without the seed."
            )
    try:
        return _build(ws, workspace_id, organization, name, storage_service, seed_toml)
    except WorkspacesWriteConflict:
        # Another writer changed the row after this call seeded it — resetting it now
        # would discard their write, so leave it to them.
        raise
    except BaseException:
        # Same promise as the listed path: a seed this call wrote must not survive a
        # failed create, or the documented retry is refused as "differs from the seed".
        if wrote_seed:
            with contextlib.suppress(Exception):
                if ws.workspaces_id is not None:
                    # A row addressed by id: back to empty — through a fresh Workspace,
                    # because the build's last write went through one of its own, so
                    # this one's memoized text (the reset's conflict token) is stale
                    # and a conflict-detecting backend would refuse the reset.
                    fresh = Workspace(root=ws.root, workspaces_id=ws.workspaces_id)
                    wsconfig.write_config_text(fresh, "")
                else:
                    ws.config_path.unlink()
                    if made_root:
                        ws.root.rmdir()
        raise


def _build(
    ws: Workspace,
    workspace_id: str | None,
    organization: str | None,
    name: str | None,
    storage_service: str | None,
    seed_toml: str | None,
) -> CreateWorkspaceResult:
    """Everything after the seed is in place and, for a listed workspace, its row is
    claimed."""
    seeded = seed_toml is not None

    # Read once: on a re-run (or a second machine sharing a store) these are the
    # values that must survive.
    recorded = wsconfig.read_identity(ws)

    # Prefer the caller's name, then what the config records, and only then the
    # directory name — without the middle term, re-running against a shared config
    # renames the workspace after whatever the local directory happens to be called.
    display_name = name or recorded.name or ws.root.name

    resolved_org = organization or recorded.organization
    if resolved_org is None:
        raise InvalidArgument(
            "organization is required to create a workspace. It is embedded in this "
            "workspace's docset namespace URIs (http://dgml.io/<organization>/"
            "<DocSetSlug>), so pick a stable identifier for your org. It becomes optional "
            "once the workspace's config.toml records one."
        )
    org_changed_from = (
        recorded.organization
        if organization is not None
        and recorded.organization is not None
        and organization != recorded.organization
        else None
    )

    # Inherit the recorded service for a sharper reason than the name: without it,
    # re-running on a workspace bound to `acme` silently rebound it to the local-disk
    # `default` and re-sealed, so the next write went to local disk while the corpus
    # sat in S3.
    service = storage_service or recorded.storage_service or DEFAULT_STORAGE_SERVICE
    service_changed_from = (
        recorded.storage_service
        if storage_service is not None
        and recorded.storage_service is not None
        and storage_service != recorded.storage_service
        else None
    )

    if seeded:
        # A seed exists to name a backend. If it declares services but not the one
        # selected, binding would fall through to the bundled local store — building
        # the workspace somewhere the caller did not ask for, discovered only once
        # their data appears to be missing. Checked before the service resolves so a
        # mistyped service is answered with the services the seed *does* declare,
        # not a bare "no [storage.<service>] configured".
        declared = wsconfig.declared_services(ws)
        if wsconfig.read_storage_table(ws, service) is None and declared:
            raise InvalidArgument(
                f"{ws.config_location} declares no [storage.{service}]. It does declare "
                f"{', '.join(f'[storage.{d}]' for d in declared)} — pass in a storage name, "
                f"or the workspace would be created on the bundled local-disk store instead "
                f"of the backend this config names."
            )
    # Reuse the id the config already carries; generate only for a genuinely new
    # workspace. Minting unconditionally forked the id on a re-run, and on a second
    # machine sharing a config it changed the workspace's identity outright.
    resolved_id = (
        ws.workspaces_id or recorded.workspace_id or workspace_id or generate_unique_workspace_id()
    )

    # Validate the named service — shape *and* provider classes — before anything is
    # built, so a bad service or `provider =` fails without a half-built workspace.
    # Under the id about to be written: a store that namespaces by workspace id needs
    # one to accept its config, and the config does not carry it yet.
    check_store_configs(*resolve_service_configs(ws, service, workspace_id=resolved_id))

    # Write the whole binding — the [storage.<service>] table *and* the
    # `storage_service` pointer — before anything resolves a store. Resolution reads
    # that pointer to pick the table, so sealing any earlier resolves against a config
    # that does not yet name the service: the workspace would be built on the bundled
    # local store and sealed to it, then fail STORAGE_BACKEND_MISMATCH on the next
    # command once the pointer became readable.
    ws.root.mkdir(parents=True, exist_ok=True)
    _materialize_storage_table(ws, service, seeded=seeded)

    wsconfig.write_identity(
        ws,
        workspace_id=resolved_id,
        name=display_name,
        organization=resolved_org,
        storage_service=service,
        created_at=recorded.created_at or now_iso(),
    )

    # Re-open now the config is complete: `store_configs` is a cached_property, so a
    # fresh object is what guarantees the seal comes from the finished binding rather
    # than a memoized guess.
    ws = Workspace(root=ws.root, workspaces_id=ws.workspaces_id)
    wsconfig.write_identity(ws, storage_fingerprint=storage_fingerprint_pair(*ws.store_configs))

    # Build through the selected backend. Nothing is scaffolded first: stores create
    # their own containers on write, so the workspace exists by virtue of its config
    # and this first document.
    ws.write_meta(name=display_name, organization=resolved_org, workspace_id=resolved_id)
    # Stamp the layout revision so a brand-new workspace is never mistaken for an old
    # one and re-scanned by the migration on first use.
    stamp_schema_version(ws)

    return CreateWorkspaceResult(
        workspace=ws,
        identity=wsconfig.read_identity(ws),
        organization_changed_from=org_changed_from,
        storage_service_changed_from=service_changed_from,
    )
