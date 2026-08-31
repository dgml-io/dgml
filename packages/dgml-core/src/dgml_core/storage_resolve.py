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

"""Resolving a workspace's storage backends from configuration.

DGML's store *resolver*, kept separate from the store *abstraction*
(:mod:`dgml_core.storage_service`, the :class:`BlobStore` / :class:`DocStore`
interfaces a third party implements). A workspace has **two** independently
configured backends — a blob store and a document store — so everything here
comes in pairs:

- **Read** — :func:`load_store_configs` resolves a named service's
  ``[storage.<name>.blobs]`` / ``.docs`` templates from the merged ``config.toml``
  into a ``(blob_cfg, doc_cfg)`` pair of :class:`StorageConfig`.
- **Build** — :func:`make_blob_store` / :func:`make_doc_store` resolve a
  ``provider`` dotted path to its interface subclass and construct it.
- **Identify** — :func:`storage_fingerprint` hashes one backend's credential-free
  identity; :func:`storage_fingerprint_pair` seals the two together.
- **Decide** — :func:`resolve_store_configs` picks *which* pair a given workspace
  opens with, and :func:`verify_storage_fingerprint` checks that pair against the
  seal recorded in the workspace's own ``config.toml``.

**Storage is the one config section that does not layer.** Every other section
deep-merges across the five layers in :mod:`dgml_core.config`; a storage service the
workspace defines in its own ``config.toml`` is taken *whole*, so the workspace is
self-describing and its seal cannot be tripped by an edit to the user-level config. A
service the workspace does *not* define falls back to the merged config, which is what
makes ``[storage.<name>]`` templates shared across workspaces still work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import load_merged_config
from .errors import StorageConfigInvalid
from .models_config import ConfigSection
from .provider import import_provider_class
from .storage import Workspace
from .storage_service import BlobStore, DocStore, StorageConfig

# The bundled default: local disk. Implements both interfaces, so it is the
# zero-config default for the blob role *and* the document role.
DEFAULT_STORAGE_PROVIDER = "dgml_core.storage_local:LocalStore"

# The storage-service name a workspace uses when none is chosen at create time,
# and the name a bare (unnamed) ``[storage]`` table resolves as.
DEFAULT_STORAGE_SERVICE = "default"

# Reserved sub-table keys of a service: the per-role backends. A named service
# ``[storage.<name>]`` either sets a single top-level ``provider`` (one class for
# both roles) or these two sub-tables (a backend each); it may not do both.
_ROLE_KEYS = ("blobs", "docs")

# Option keys never folded into the store fingerprint — rotating a credential must
# not read as "the store moved". A provider carrying an inline credential must name
# the option so one of these appears in it (see the storage-package READMEs);
# every in-tree provider takes its credentials from the environment instead.
_SECRET_HINTS = ("key", "secret", "token", "password", "credential")

# Option keys that name *where this machine keeps the data* rather than *which store
# this is*, and so are also outside the identity fingerprint. Same argument as the one
# that already excludes ``StorageConfig.root``: a workspace moved to another path is
# the same workspace on the same backend, so relocating it must not read as a storage
# change and must not require a re-seal. Matched exactly rather than by substring —
# these are specific option names, not a family like the secret hints.
_LOCATION_HINTS = frozenset({"workspace_path"})

# ------------------------------------------------------------ building stores


def _import_store_class(provider: str, base: Any) -> Any:
    """Import the dotted ``"module.path:ClassName"`` ``provider`` and check it is a
    subclass of ``base`` (:class:`BlobStore` or :class:`DocStore`).

    A thin binding of :func:`dgml_core.provider.import_provider_class` to this
    section's vocabulary and bundled default; the mechanism is shared with the
    ``[workspaces]`` resolver."""
    return import_provider_class(
        provider, base, kind="storage", default_hint=DEFAULT_STORAGE_PROVIDER
    )


def make_blob_store(config: StorageConfig) -> BlobStore:
    """Instantiate the :class:`BlobStore` named by ``config`` (resolve provider →
    ``parse_config`` → construct, where the provider's lazy SDK import happens)."""
    cls = _import_store_class(config.provider, BlobStore)
    store: BlobStore = cls(cls.parse_config(config))
    return store


def make_doc_store(config: StorageConfig) -> DocStore:
    """Instantiate the :class:`DocStore` named by ``config``."""
    cls = _import_store_class(config.provider, DocStore)
    store: DocStore = cls(cls.parse_config(config))
    return store


# ------------------------------------------------------------ reading config


def _config_from(section: Mapping[str, Any], root: Path) -> StorageConfig:
    """Build a :class:`StorageConfig` from one role table (``provider`` + the rest
    as ``options``). Raises :class:`StorageConfigInvalid` for a bad shape.

    Table-valued keys are dropped alongside the role keys: a store option is always a
    scalar or a list, so a nested table here is a sibling *service* that shared the
    merged ``[storage]`` namespace, not an option for this provider. Keeping it would
    fail :meth:`~dgml_core.storage_service._StoreBase._check_no_extra_fields` with a
    confusing "unknown field" naming another workspace's service."""
    provider = section.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise StorageConfigInvalid("'storage.provider' must be a non-empty string")
    options = {
        k: v
        for k, v in section.items()
        if k != "provider" and k not in _ROLE_KEYS and not isinstance(v, dict)
    }
    return StorageConfig(provider=provider, root=root, options=options)


def _select_service_table(section: Mapping[str, Any], service: str) -> Mapping[str, Any] | None:
    """The ``[storage.<service>]`` config table, or ``None`` when absent.

    A bare ``[storage]`` that is *itself* a service config — a top-level
    ``provider`` (flat), or top-level ``blobs``/``docs`` sub-tables — is the
    ``"default"`` service; any other name then raises. Otherwise ``[storage]`` is a
    namespace of named services and ``section[service]`` is selected."""
    default_is_inline = isinstance(section.get("provider"), str) or any(
        isinstance(section.get(r), dict) for r in _ROLE_KEYS
    )
    if default_is_inline:
        if service != DEFAULT_STORAGE_SERVICE:
            raise StorageConfigInvalid(
                f"no storage service {service!r}: config has a single [storage] table"
            )
        return section
    sub = section.get(service)
    if sub is not None and not isinstance(sub, dict):
        raise StorageConfigInvalid(f"[storage.{service}] must be a table")
    return sub


def _role_config(table: Mapping[str, Any] | None, role: str, root: Path) -> StorageConfig:
    """Resolve one role (``"blobs"``/``"docs"``) of a service table to a
    :class:`StorageConfig`. A per-role sub-table wins; else a flat top-level
    ``provider`` serves both roles; else (role omitted / no config) the bundled
    local-disk default."""
    if table is not None:
        sub = table.get(role)
        if isinstance(sub, dict):
            return _config_from(sub, root)
        if isinstance(table.get("provider"), str):
            return _config_from(table, root)  # flat: one provider for both roles
    return StorageConfig(provider=DEFAULT_STORAGE_PROVIDER, root=root)


def load_store_configs(
    workspace: Workspace, service: str = DEFAULT_STORAGE_SERVICE
) -> tuple[StorageConfig, StorageConfig]:
    """Resolve a named service into a ``(blob_cfg, doc_cfg)`` pair.

    ``config.toml`` defines services as ``[storage.<name>]`` tables. Each is either
    **flat** (a top-level ``provider`` used for both roles — the class must
    implement both), or **per-role** (``[storage.<name>.blobs]`` /
    ``[storage.<name>.docs]`` sub-tables, each its own ``provider`` + options; a
    role omitted falls back to the bundled local store). A bare ``[storage]`` is the
    ``"default"`` service; no ``[storage]`` at all → both roles on local disk (zero
    config).

    Validates only the *generic shape* — provider resolution and field validation
    happen lazily in :func:`make_blob_store` / :func:`make_doc_store`. Raises
    :class:`StorageConfigInvalid` for a malformed shape or an unknown named service.
    """
    root = workspace.root
    section = load_merged_config(workspace).get(ConfigSection.STORAGE) or {}
    if not isinstance(section, dict):
        raise StorageConfigInvalid("'storage' must be a table")
    table = _select_service_table(section, service)
    if table is None and service != DEFAULT_STORAGE_SERVICE:
        raise StorageConfigInvalid(f"no [storage.{service}] configured")
    if table is not None:
        _reject_mixed_form(table, service)
    return _role_config(table, "blobs", root), _role_config(table, "docs", root)


def _reject_mixed_form(table: Mapping[str, Any], service: str) -> None:
    """A service table sets *either* one top-level ``provider`` for both roles *or*
    per-role sub-tables — never both, which would leave which one wins ambiguous."""
    has_provider = isinstance(table.get("provider"), str)
    has_roles = any(isinstance(table.get(r), dict) for r in _ROLE_KEYS)
    if has_provider and has_roles:
        raise StorageConfigInvalid(
            f"[storage.{service}] has both a top-level 'provider' and blobs/docs "
            f"sub-tables; use one form (a single provider for both roles, or a "
            f"backend each)"
        )


# --------------------------------------------------------- resolving a workspace


def resolve_store_configs(workspace: Workspace) -> tuple[StorageConfig, StorageConfig]:
    """The effective ``(blob_cfg, doc_cfg)`` pair ``workspace`` opens with.

    The workspace's own ``config.toml`` is authoritative. Its ``[workspace]`` block
    names which service it binds to (``storage_service``, defaulting to ``"default"``),
    and if it defines that ``[storage.<service>]`` table itself, **that table is used
    whole** — the user-level config contributes nothing. Only when the workspace does
    not define the service does resolution fall back to the merged config, which is how
    a shared ``[storage.<name>]`` template still serves many workspaces.

    Reached through :attr:`dgml_core.storage.Workspace.store_configs`, which caches it.
    """
    from . import workspace_config

    service = workspace_config.read_identity(workspace).storage_service or DEFAULT_STORAGE_SERVICE
    own = workspace_config.read_storage_table(workspace, service)
    if own is not None:
        root = workspace.root
        _reject_mixed_form(own, service)
        return _role_config(own, "blobs", root), _role_config(own, "docs", root)
    return load_store_configs(workspace, service)


def verify_storage_fingerprint(workspace: Workspace) -> None:
    """Raise :class:`~dgml_core.errors.StorageBackendMismatch` if the workspace's
    resolved storage no longer matches the seal recorded in its ``config.toml``.

    A workspace with no recorded fingerprint is **unsealed** and passes untouched —
    trust-on-first-use, which is what lets a hand-built or freshly-migrated workspace
    open before it has ever been sealed.

    Store-free: resolution reads TOML only, so a config naming an unreachable backend
    is rejected here rather than after a failed connection."""
    from . import workspace_config
    from .errors import StorageBackendMismatch

    recorded = workspace_config.read_identity(workspace).storage_fingerprint
    if not recorded:
        return
    blob_cfg, doc_cfg = workspace.store_configs
    if storage_fingerprint_pair(blob_cfg, doc_cfg) == recorded:
        return
    raise StorageBackendMismatch(
        f"the [storage] configuration this workspace resolves no longer matches the "
        f"storage_fingerprint recorded in {workspace.config_location}. Its data is on the "
        f"previously sealed backend, so opening it against the new configuration could "
        f"read or write the wrong store. If the change was intended, run "
        f"'dgml workspace reseal {workspace.root}' to accept it; otherwise restore the "
        f"[storage] table."
    )


# ------------------------------------------------------------ identity / seal


def _excluded_from_identity(key: str) -> bool:
    """Whether an option key is outside the store-identity hash."""
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_HINTS) or lowered in _LOCATION_HINTS


def _identity_hash(provider: str, options: Mapping[str, Any]) -> str:
    """The canonical credential-free store-identity hash for one backend."""
    identity = {
        "provider": provider,
        "options": {k: v for k, v in sorted(options.items()) if not _excluded_from_identity(k)},
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def storage_fingerprint(config: StorageConfig) -> str:
    """Credential-free content hash of one backend's identity (provider + non-secret,
    non-location options). See :func:`storage_fingerprint_pair` for the two-backend seal
    a workspace records in its ``config.toml``."""
    return _identity_hash(config.provider, config.options)


def storage_fingerprint_pair(blob_cfg: StorageConfig, doc_cfg: StorageConfig) -> str:
    """The seal a workspace records: a credential-free hash over both roles.

    Three kinds of thing are deliberately outside the hash, all for the same reason —
    the seal answers "is this the same store", not "is this the same address on this
    machine":

    - ``root``, so a workspace copied or moved to another path keeps its seal.
    - Secret-named options (:data:`_SECRET_HINTS`), so rotating a credential never reads
      as "the store moved".
    - Location options (:data:`_LOCATION_HINTS`), for exactly the ``root`` argument: an
      option naming where *this machine* keeps the data describes an address, not an
      identity."""
    body = json.dumps(
        {"blobs": storage_fingerprint(blob_cfg), "docs": storage_fingerprint(doc_cfg)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()
