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

"""The **legacy** per-machine workspace index — read only, for importing.

``~/.config/dgml/workspaces.json`` used to be how a machine knew which workspaces it
had: a JSON object mapping each ``workspace_id`` to where that workspace was last
seen. It is no longer written, and nothing resolves through it.

What replaced it is :mod:`dgml_core.workspaces_store` — a pluggable store that holds
the list of workspaces *and* each workspace's ``config.toml``. The difference that
matters is authority: this file's rows were a second copy of facts that lived
elsewhere, so they could disagree with the workspace they described and had to be
rewritten on every open to stay current. A listing row now comes out of the
workspace's own config, so there is nothing to keep in sync.

This module survives so ``dgml workspace import`` can read what an older dgml left
behind, and so :func:`dgml_core.migrations.migrate_workspace_config` can still lift a
pre-upgrade row's inline ``storage`` snapshot into the workspace's own config. Both
are read-only. Nothing here writes the file, and once every workspace a machine cares
about has been imported it can be deleted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .storage import read_json, user_config_path

# Re-exported (redundant-alias spelling, so ``mypy --strict`` accepts it) for the
# legacy tests that still reach for it here. Id minting itself lives in
# :mod:`dgml_core.workspace_id`.
from .workspace_id import new_workspace_id as new_workspace_id

REGISTRY_FILE = "workspaces.json"


def registry_path() -> Path:
    """The registry file, next to the user ``config.toml`` (honors
    ``XDG_CONFIG_HOME``/``APPDATA``)."""
    return user_config_path().parent / REGISTRY_FILE


@dataclass(frozen=True)
class RegistryEntry:
    """One workspace's row in the index: where it was last seen, and what to call it.

    ``root`` is the local directory the workspace was opened at. ``name`` and
    ``organization`` are copies carried so ``dgml workspace list`` can render a row
    without opening (and possibly failing to reach) every workspace's store.

    ``config_path`` is recorded **only** when the workspace's ``config.toml`` lives
    outside ``root`` (``--workspace-config``); absent means the default
    ``<root>/config.toml``, which is derivable and would be noise to store. It is a
    **hint, not an address**: resolution consults it only when the default is missing,
    and ignores it when it points at a file that is gone. Nothing here is authoritative
    — a stale hint degrades to the same "config is missing" error you would get
    without it, never to opening the wrong config.
    """

    workspace_id: str
    name: str
    organization: str
    root: str | None
    created_at: str
    schema_version: int
    config_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "organization": self.organization,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }
        if self.root is not None:
            d["root"] = self.root
        if self.config_path is not None:
            d["config_path"] = self.config_path
        return d

    @classmethod
    def from_dict(cls, workspace_id: str, data: dict[str, Any]) -> RegistryEntry:
        """Parse one row, ignoring unknown keys.

        Tolerating extras is what upgrades a pre-existing ``workspaces.json`` for
        free: the ``storage`` / ``storage_service`` / ``storage_fingerprint`` an older
        dgml wrote are read as noise here and dropped the next time the row is
        rewritten. The migration reads them from the raw JSON before that happens (see
        :func:`dgml_core.migrations.migrate_workspace_config`)."""
        return cls(
            workspace_id=workspace_id,
            name=str(data.get("name", "")),
            organization=str(data.get("organization", "")),
            root=data.get("root"),
            created_at=str(data.get("created_at", "")),
            schema_version=int(data["schema_version"])
            if isinstance(data.get("schema_version"), int)
            else 0,
            config_path=data.get("config_path")
            if isinstance(data.get("config_path"), str)
            else None,
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


def list_entries() -> list[RegistryEntry]:
    """All rows, sorted by id (stable output for ``dgml workspace import``)."""
    reg = read_registry()
    return [reg[wid] for wid in sorted(reg)]


def raw_entry_by_root(root: Path) -> dict[str, Any] | None:
    """The **unparsed** row whose ``root`` is ``root``, or ``None``.

    Exists for the callers that need fields :class:`RegistryEntry` deliberately drops:
    :func:`dgml_core.migrations.migrate_workspace_config` reads a pre-upgrade row's
    ``storage`` snapshot out of it, and ``dgml workspace import`` reads the same row to
    decide what it is importing."""
    raw = _read_raw()
    target = root.resolve()
    for wid in sorted(raw):
        data = raw.get(wid)
        if not isinstance(data, dict):
            continue
        entry_root = data.get("root")
        if isinstance(entry_root, str) and Path(entry_root).resolve() == target:
            return {**data, "workspace_id": wid}
    return None
