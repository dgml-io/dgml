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
- **Identify** — :func:`storage_snapshot` + :func:`snapshot_pair` produce the
  credential-free identity the registry records; :func:`fingerprint_pair` seals
  it; :func:`secret_options` is the credential complement merged back at open.

Deciding *which* configs a given (possibly registered) workspace opens with lives
one layer up, in :func:`dgml_core.registry.resolve_store_configs`, because it
consults the registry — this module has no knowledge of the registry.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import load_merged_config
from .errors import StorageConfigInvalid, StorageProviderUnresolvable
from .models_config import ConfigSection
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

# Option keys never folded into the store fingerprint / snapshot — rotating a
# credential must not read as "the store moved", and secrets never reach the
# plaintext registry.
_SECRET_HINTS = ("key", "secret", "token", "password", "credential")

# ------------------------------------------------------------ building stores


def _import_store_class(provider: str, base: Any) -> Any:
    """Import the dotted ``"module.path:ClassName"`` ``provider`` and check it is a
    subclass of ``base`` (:class:`BlobStore` or :class:`DocStore`).

    Raises :class:`StorageProviderUnresolvable` if the string is malformed, the
    module/attribute can't be imported, or the target is not a ``base`` subclass —
    the last catches "a doc provider used where a blob provider is required". Returns
    the class (``Any``: it is a concrete subclass only known at runtime)."""
    if ":" not in provider:
        raise StorageProviderUnresolvable(
            f"storage provider must be a dotted path 'module.path:ClassName' "
            f"(got {provider!r}); the bundled default is {DEFAULT_STORAGE_PROVIDER!r}"
        )
    module_path, _, class_name = provider.partition(":")
    if not module_path or not class_name:
        raise StorageProviderUnresolvable(
            f"storage provider {provider!r} must have the form 'module.path:ClassName'"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise StorageProviderUnresolvable(
            f"could not import storage module {module_path!r} for provider {provider!r}: "
            f"{exc}. Is the package installed in this environment?"
        ) from exc
    try:
        obj = getattr(module, class_name)
    except AttributeError as exc:
        raise StorageProviderUnresolvable(
            f"module {module_path!r} has no attribute {class_name!r} (provider {provider!r})"
        ) from exc
    if not (isinstance(obj, type) and issubclass(obj, base)):
        raise StorageProviderUnresolvable(
            f"provider {provider!r} resolved to {obj!r}, which is not a {base.__name__} subclass"
        )
    return obj


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
    as ``options``). Raises :class:`StorageConfigInvalid` for a bad shape."""
    provider = section.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise StorageConfigInvalid("'storage.provider' must be a non-empty string")
    options = {k: v for k, v in section.items() if k != "provider" and k not in _ROLE_KEYS}
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
        has_provider = isinstance(table.get("provider"), str)
        has_roles = any(isinstance(table.get(r), dict) for r in _ROLE_KEYS)
        if has_provider and has_roles:
            raise StorageConfigInvalid(
                f"[storage.{service}] has both a top-level 'provider' and blobs/docs "
                f"sub-tables; use one form (a single provider for both roles, or a "
                f"backend each)"
            )
    return _role_config(table, "blobs", root), _role_config(table, "docs", root)


# ------------------------------------------------------------ identity / seal


def _identity_hash(provider: str, options: Mapping[str, Any]) -> str:
    """The canonical credential-free store-identity hash for one backend."""
    identity = {
        "provider": provider,
        "options": {
            k: v
            for k, v in sorted(options.items())
            if not any(hint in k.lower() for hint in _SECRET_HINTS)
        },
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def storage_fingerprint(config: StorageConfig) -> str:
    """Credential-free content hash of one backend's identity (provider + non-secret
    options). See :func:`fingerprint_pair` for the two-backend seal a workspace uses."""
    return _identity_hash(config.provider, config.options)


def storage_snapshot(config: StorageConfig) -> dict[str, Any]:
    """One backend's **non-secret** identity as a flat dict — ``{"provider": …,
    <opt>: …}``. Secret-hinted options are dropped, so credentials never reach the
    plaintext registry. :func:`secret_options` is the complement."""
    snapshot: dict[str, Any] = {"provider": config.provider}
    snapshot.update(
        (k, v)
        for k, v in config.options.items()
        if not any(hint in k.lower() for hint in _SECRET_HINTS)
    )
    return snapshot


def secret_options(config: StorageConfig) -> dict[str, Any]:
    """The **secret-hinted** options of one backend — the complement of
    :func:`storage_snapshot`. Merged back into a registered workspace's non-secret
    snapshot at open time; never persisted to the registry."""
    return {
        k: v for k, v in config.options.items() if any(hint in k.lower() for hint in _SECRET_HINTS)
    }


def fingerprint_of_snapshot(snapshot: Mapping[str, Any]) -> str:
    """Recompute one backend's identity hash from a persisted :func:`storage_snapshot`
    — equal to ``storage_fingerprint`` of the config it was taken from. Returns ``""``
    when the snapshot has no ``provider``."""
    provider = snapshot.get("provider")
    if not isinstance(provider, str):
        return ""
    options = {k: v for k, v in snapshot.items() if k != "provider"}
    return _identity_hash(provider, options)


def snapshot_pair(blob_cfg: StorageConfig, doc_cfg: StorageConfig) -> dict[str, Any]:
    """The non-secret snapshot a workspace's registry entry records:
    ``{"blobs": storage_snapshot(blob_cfg), "docs": storage_snapshot(doc_cfg)}``."""
    return {"blobs": storage_snapshot(blob_cfg), "docs": storage_snapshot(doc_cfg)}


def fingerprint_pair(pair: Mapping[str, Any]) -> str:
    """Credential-free hash of a ``{"blobs": …, "docs": …}`` snapshot pair — the
    integrity seal recomputed on open. Returns ``""`` when the pair is empty/unset,
    treated as unsealed (trust-on-first-use)."""
    blobs = pair.get("blobs") if isinstance(pair.get("blobs"), Mapping) else None
    docs = pair.get("docs") if isinstance(pair.get("docs"), Mapping) else None
    if not blobs and not docs:
        return ""
    body = json.dumps(
        {"blobs": dict(blobs or {}), "docs": dict(docs or {})},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()
