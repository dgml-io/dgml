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

"""The per-machine workspace registry.

A workspace has a stable ``workspace_id`` (minted at ``dgml workspace create``,
carried in ``workspace.json``). This module maintains a small JSON index —
``~/.config/dgml/workspaces.json`` (sibling of the user ``config.toml``) — mapping
each ``workspace_id`` to where that workspace lives, so it can be opened by id
(``dgml --workspace <id>``) and listed (``dgml workspace list``).

The registry is **per-machine** state, deliberately separate from the
per-workspace ``workspace.json`` (which travels with the directory): the same
workspace opened on two machines has one id but two registry entries with
different roots. It is machine-managed (JSON, like every other metadata file —
``workspace.json``, ``docset.json``), not hand-edited like ``config.toml``.

Each entry is **self-describing** about the workspace's stores: ``storage_service``
names the ``config.toml`` ``[storage.<name>]`` template it was created from, and
``storage`` holds a **non-secret snapshot pair** of that template —
``{"blobs": {provider, …}, "docs": {provider, …}}``, one per backend role (a
workspace configures its blob store and its document store independently). The
snapshot is *authoritative* for opening the workspace, so it opens even if the
template is later edited or removed — the registry alone records where a workspace's
data lives. ``storage_fingerprint`` hashes the pair and is recomputed on open
(:func:`verify_storage_seal`) to detect a hand-edited entry. Secrets never enter the
registry; they are read from the template/env at open.

Today only ``LocalStore`` ships, so every entry also records a local ``root``.
Reconstructing a *remote* store from ``entry.storage`` (open-by-id with no local
root) lands with the remote store.
"""

from __future__ import annotations

import base64
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .storage import Workspace, read_json, user_config_path, write_json_atomic

if TYPE_CHECKING:
    from .storage_service import StorageConfig

REGISTRY_FILE = "workspaces.json"
_ID_PREFIX = "ws_"


def registry_path() -> Path:
    """The registry file, next to the user ``config.toml`` (honors
    ``XDG_CONFIG_HOME``/``APPDATA``)."""
    return user_config_path().parent / REGISTRY_FILE


def new_workspace_id() -> str:
    """A fresh opaque workspace id: ``ws_`` + 16 lowercase base32 chars (80 bits).

    Non-semantic (survives a directory rename) and hyphen/separator-free — the
    ``ws_`` prefix lets ``Workspace.resolve`` tell an id from a path without a
    dedicated flag. Not collision-checked — use :func:`mint_workspace_id` when
    assigning an id to a workspace."""
    slug = base64.b32encode(secrets.token_bytes(10)).decode("ascii").lower().rstrip("=")
    return f"{_ID_PREFIX}{slug}"


def mint_workspace_id() -> str:
    """A fresh workspace id guaranteed not to already be in this machine's registry.

    80 bits from :func:`secrets` won't collide in practice; the registry re-roll is
    belt-and-suspenders so two workspaces can never share an id (and shadow each
    other on open)."""
    wid = new_workspace_id()
    while get(wid) is not None:
        wid = new_workspace_id()
    return wid


@dataclass(frozen=True)
class RegistryEntry:
    """One workspace's row in the registry — its self-describing store record.

    ``root`` is the local store location. ``storage_service`` names the
    ``config.toml`` ``[storage.<name>]`` template the workspace was created from
    (where its secrets live, and the target of a re-seal). ``storage`` is the
    **non-secret snapshot pair** of that template —
    ``{"blobs": {provider, …}, "docs": {provider, …}}`` — which is *authoritative*
    for opening the workspace (so it opens even if the template is later
    edited/removed), and ``storage_fingerprint`` is that pair's hash: recomputed on
    open to detect a hand-edited entry
    (:class:`~dgml_core.errors.StorageBackendMismatch`).
    """

    workspace_id: str
    name: str
    organization: str
    root: str | None
    storage_service: str
    storage: dict[str, Any]
    storage_fingerprint: str
    created_at: str
    schema_version: int

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "organization": self.organization,
            "storage_service": self.storage_service,
            "storage": self.storage,
            "storage_fingerprint": self.storage_fingerprint,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }
        if self.root is not None:
            d["root"] = self.root
        return d

    @classmethod
    def from_dict(cls, workspace_id: str, data: dict[str, Any]) -> RegistryEntry:
        service = data.get("storage_service")
        raw_storage = data.get("storage")
        storage = raw_storage if isinstance(raw_storage, dict) else {}
        # Back-compat: a single-provider snapshot (``{"provider": …}``, pre blob/doc
        # split) becomes the same backend for both roles.
        if "provider" in storage:
            storage = {"blobs": storage, "docs": storage}
        return cls(
            workspace_id=workspace_id,
            name=str(data.get("name", "")),
            organization=str(data.get("organization", "")),
            root=data.get("root"),
            # Back-compat: entries written before named services default to "default".
            storage_service=service if isinstance(service, str) and service else "default",
            storage=storage,
            storage_fingerprint=str(data.get("storage_fingerprint", "")),
            created_at=str(data.get("created_at", "")),
            schema_version=int(data["schema_version"])
            if isinstance(data.get("schema_version"), int)
            else 0,
        )


def _read_raw() -> dict[str, Any]:
    """The registry as a raw ``{id: entry-dict}`` mapping (``{}`` when absent).

    Raises :class:`~dgml_core.errors.CorruptMetadata` on malformed JSON /
    duplicate ids (via :func:`read_json`) or a non-object top level."""
    from .errors import CorruptMetadata

    path = registry_path()
    if not path.exists():
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        raise CorruptMetadata(f"{path} must contain a JSON object of workspace_id -> entry")
    return data


def read_registry() -> dict[str, RegistryEntry]:
    """Every registered workspace, keyed by ``workspace_id``."""
    return {wid: RegistryEntry.from_dict(wid, entry) for wid, entry in _read_raw().items()}


def register(entry: RegistryEntry) -> None:
    """Insert or replace ``entry`` (idempotent upsert by id), atomically.

    Whole-file read-modify-write; each write is atomic (write-temp-rename). Writes
    happen only at create / ``workspace register`` / first-open registration, so
    the (non-cross-process-atomic) RMW is acceptable — an interleaved lost update
    self-heals on the next open, since register is idempotent."""
    data = _read_raw()
    data[entry.workspace_id] = entry.to_dict()
    write_json_atomic(registry_path(), data)


def get(workspace_id: str) -> RegistryEntry | None:
    entry = _read_raw().get(workspace_id)
    return RegistryEntry.from_dict(workspace_id, entry) if isinstance(entry, dict) else None


def get_by_root(root: Path) -> RegistryEntry | None:
    """The entry whose local ``root`` is ``root`` (path addressing / open-by-path).

    Deterministic on the off chance two ids share a root: lowest id wins."""
    target = root.resolve()
    for wid in sorted(_read_raw()):
        entry = get(wid)
        if entry is not None and entry.root is not None and Path(entry.root).resolve() == target:
            return entry
    return None


def list_entries() -> list[RegistryEntry]:
    """All entries, sorted by id (stable output for ``dgml workspace list``)."""
    reg = read_registry()
    return [reg[wid] for wid in sorted(reg)]


def remove(workspace_id: str) -> bool:
    """Drop ``workspace_id`` from the registry. Returns whether it was present."""
    data = _read_raw()
    if workspace_id not in data:
        return False
    del data[workspace_id]
    write_json_atomic(registry_path(), data)
    return True


# --------------------------------------------------- building entries from a workspace


def entry_for(
    ws: Workspace,
    *,
    name: str,
    organization: str,
    workspace_id: str,
    service: str,
    created_at: str,
    schema_version: int,
) -> RegistryEntry:
    """Build ``ws``'s self-describing registry entry from the named storage
    ``service``: its local ``root``, the service ``name``, a **non-secret snapshot**
    of that service's config (the authoritative location record), and the snapshot's
    ``storage_fingerprint`` (recomputed on open to detect a hand-edited entry).

    The ``storage`` snapshot is a **pair** — ``{"blobs": …, "docs": …}`` — one
    non-secret snapshot per role; ``storage_fingerprint`` hashes the pair.
    Store-free — reads the ``config.toml`` template, not the store. (Lazy imports
    keep ``registry`` free of a top-level ``storage_service`` / ``migrations``
    cycle.)"""
    from .storage_resolve import fingerprint_pair, load_store_configs, snapshot_pair

    blob_cfg, doc_cfg = load_store_configs(ws, service)
    snapshot = snapshot_pair(blob_cfg, doc_cfg)
    return RegistryEntry(
        workspace_id=workspace_id,
        name=name,
        organization=organization,
        root=str(ws.root),
        storage_service=service,
        storage=snapshot,
        storage_fingerprint=fingerprint_pair(snapshot),
        created_at=created_at,
        schema_version=schema_version,
    )


def seal_entry(
    ws: Workspace, *, workspace_id: str, name: str, organization: str, service: str
) -> None:
    """Assemble ``ws``'s entry for the named ``service`` (stamping ``created_at`` /
    current schema version) and upsert it — the one place an entry is written.
    Store-free (see :func:`entry_for`), so it is safe to call *before* the store is
    first used at ``workspace create`` (the entry must exist for ``Workspace.store``
    to resolve the chosen service)."""
    from .errors import now_iso
    from .migrations import WORKSPACE_SCHEMA_VERSION

    register(
        entry_for(
            ws,
            name=name,
            organization=organization,
            workspace_id=workspace_id,
            service=service,
            created_at=now_iso(),
            schema_version=WORKSPACE_SCHEMA_VERSION,
        )
    )


def register_workspace(ws: Workspace, *, name: str, organization: str, service: str) -> str:
    """Register a **brand-new** workspace on this machine and return its minted
    ``workspace_id``: mints the id and seals the entry for the named ``service``.

    Store-free (see :func:`seal_entry`), so it is the *first* step of ``dgml
    workspace create`` — it runs before ``ws.init()`` / ``ws.write_meta`` so that
    ``Workspace.store`` resolves the chosen backend when the caller then builds the
    workspace through it. The caller supplies ``name``/``organization``/``service``;
    there is no ``workspace.json`` to read yet. To re-register an *existing*
    workspace instead, use :func:`reregister_workspace`."""
    wid = mint_workspace_id()
    seal_entry(ws, workspace_id=wid, name=name, organization=organization, service=service)
    return wid


def reregister_workspace(
    ws: Workspace,
    *,
    name: str | None = None,
    organization: str | None = None,
    service: str | None = None,
) -> str:
    """Re-register an **already-initialized** ``ws`` on this machine, returning its
    ``workspace_id`` (minting one into ``workspace.json`` if absent).

    The *authoritative* re-register (``dgml workspace register``) — it re-seals the
    entry (the moved-directory / adopt-new-config / repair-a-hand-edited-entry fix),
    unlike the additive :func:`ensure_registered`. ``service`` selects the storage
    template to snapshot; when ``None`` it keeps the entry's current service (else
    ``"default"``). ``name``/``organization`` default to the workspace's own
    identity. (For a brand-new workspace use :func:`register_workspace`.)"""
    from .storage_resolve import DEFAULT_STORAGE_SERVICE

    existing = get_by_root(ws.root)
    if existing is not None:
        # Already indexed: take identity from the (store-free) entry, so a re-seal
        # works even when the entry's own storage snapshot was hand-edited into an
        # unopenable state — that is exactly what this repairs.
        wid = existing.workspace_id
        name = existing.name if name is None else name
        organization = existing.organization if organization is None else organization
        if service is None:
            service = existing.storage_service
    else:
        # Not indexed here (e.g. a moved directory): read identity from the
        # workspace itself, minting an id if it lacks one.
        name = ws.display_name if name is None else name
        organization = ws.organization if organization is None else organization
        current = ws.workspace_id
        if current is None:
            wid = mint_workspace_id()
            ws.write_meta(name=name, organization=organization, workspace_id=wid)
        else:
            wid = current
        if service is None:
            service = DEFAULT_STORAGE_SERVICE
    seal_entry(ws, workspace_id=wid, name=name, organization=organization, service=service)
    return wid


def ensure_registered(ws: Workspace) -> None:
    """Add ``ws`` to this machine's registry if it has an id and isn't indexed yet.

    Idempotent and additive: never overwrites an existing entry (that is what
    :func:`reregister_workspace` / the explicit ``dgml workspace register`` does). A
    no-op for a workspace with no ``workspace_id`` (one is minted by the backfill
    migration on first open). Snapshots the ``"default"`` service — an
    unregistered workspace opened on this machine runs on the bundled local store,
    so that is what it is being registered as (an explicit ``workspace register
    --storage`` re-seals it to a named service)."""
    from .storage_resolve import DEFAULT_STORAGE_SERVICE

    wid = ws.workspace_id
    if wid is None or get(wid) is not None:
        return
    seal_entry(
        ws,
        workspace_id=wid,
        name=ws.display_name,
        organization=ws.organization,
        service=DEFAULT_STORAGE_SERVICE,
    )


def verify_storage_seal(ws: Workspace) -> None:
    """Refuse to open ``ws`` if its registry entry's ``storage`` snapshot was
    hand-edited so it no longer matches the ``storage_fingerprint`` sealed beside it.

    An **integrity check on the registry entry itself** — the registry is
    machine-managed, so a snapshot that doesn't hash to its recorded fingerprint
    means the JSON was edited out of band, and the workspace's store config can no
    longer be trusted. Pure local read + recompute; it does **not** consult
    ``config.toml`` (editing a template never trips this — the entry's snapshot is
    authoritative). A no-op for an unregistered or unsealed workspace
    (trust-on-first-use).

    Store-free, so it can run before the stores are first built — a tampered
    entry is caught before its config is used to construct a backend. Raises
    :class:`~dgml_core.errors.StorageBackendMismatch`."""
    from .storage_resolve import fingerprint_pair

    entry = get_by_root(ws.root)
    if entry is None or not entry.storage_fingerprint:
        return
    if fingerprint_pair(entry.storage) != entry.storage_fingerprint:
        from .errors import StorageBackendMismatch

        raise StorageBackendMismatch(
            f"the storage config recorded for this workspace in the registry "
            f"({registry_path()}) has been modified and no longer matches its sealed "
            f"fingerprint. The registry is machine-managed — do not hand-edit it. Run "
            f"`dgml workspace register {ws.root} --storage {entry.storage_service}` to "
            f"re-seal it from your config, or restore the entry."
        )


def resolve_store_configs(ws: Workspace) -> tuple[StorageConfig, StorageConfig]:
    """The effective ``(blob_cfg, doc_cfg)`` pair for opening ``ws`` — the store
    selection that :attr:`dgml_core.storage.Workspace.blobs` /
    :attr:`~dgml_core.storage.Workspace.docs` build from.

    For a **registered** workspace each role's non-secret identity comes from the
    entry's snapshot (authoritative and self-contained — the stores open even if
    ``config.toml`` was edited/deleted); only *secret* options are merged in from
    that role's ``config.toml`` template (or the provider SDK's own credential chain
    when the template is gone). ``config.toml`` never overrides the non-secret
    identity. An **unregistered** workspace (a raw ``Workspace(root=…)``, or one
    being created before its entry is written) resolves the ``"default"`` service —
    both roles on the bundled local-disk store, zero config.

    Lives here rather than in :mod:`dgml_core.storage_resolve` because it consults
    the registry; everything it needs from the resolver is imported one-way."""
    from .errors import StorageConfigInvalid
    from .storage_resolve import DEFAULT_STORAGE_SERVICE, load_store_configs

    entry = get_by_root(ws.root)  # local read, store-free
    if entry is None:
        return load_store_configs(ws, DEFAULT_STORAGE_SERVICE)
    service = entry.storage_service or DEFAULT_STORAGE_SERVICE
    tmpl_blob: StorageConfig | None
    tmpl_doc: StorageConfig | None
    try:
        tmpl_blob, tmpl_doc = load_store_configs(ws, service)
    except StorageConfigInvalid:
        # Template gone/renamed — the location is still known from the snapshot;
        # creds may come from the provider SDK's own chain (env, instance role, …).
        tmpl_blob = tmpl_doc = None
    storage = entry.storage if isinstance(entry.storage, dict) else {}
    return (
        _role_from_entry(ws, storage.get("blobs"), tmpl_blob),
        _role_from_entry(ws, storage.get("docs"), tmpl_doc),
    )


def _role_from_entry(ws: Workspace, snapshot: Any, template: StorageConfig | None) -> StorageConfig:
    """One role's effective config: non-secret identity from the entry ``snapshot``
    (authoritative), secrets merged from that role's ``template`` config. Falls back
    to the template (then the bundled local store) when the snapshot has no usable
    provider."""
    from .storage_resolve import DEFAULT_STORAGE_PROVIDER, secret_options
    from .storage_service import StorageConfig

    provider = snapshot.get("provider") if isinstance(snapshot, dict) else None
    if not isinstance(provider, str) or not provider.strip():
        return template or StorageConfig(provider=DEFAULT_STORAGE_PROVIDER, root=ws.root)
    non_secret = {k: v for k, v in snapshot.items() if k != "provider"}
    secrets = secret_options(template) if template is not None else {}
    return StorageConfig(provider=provider, root=ws.root, options={**non_secret, **secrets})
