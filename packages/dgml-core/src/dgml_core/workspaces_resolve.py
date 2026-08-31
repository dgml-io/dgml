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

"""Resolving which :class:`~dgml_core.workspaces_store.WorkspacesStore` a machine uses.

The ``[workspaces]`` table of the **user** config selects it::

    # ~/.config/dgml/config.toml — omit the table entirely for local disk
    [workspaces]
    provider = "dgml_storage_mongo:MongoWorkspacesStore"
    mongo_host = "localhost"
    mongo_database = "dgml_workspaces"

Read with :mod:`tomllib` straight from :func:`dgml_core.storage.user_config_path`,
deliberately bypassing :func:`dgml_core.config.load_merged_config`. That function takes
a ``Workspace``, and this store is what *produces* a workspace's config — resolving it
through the merged config is the chicken-and-egg.

Which is also why **``[workspaces]`` in a workspace's own ``config.toml`` is ignored**,
and cannot be honoured even in principle: the store was already used to fetch that
file. Two further reasons it must not layer: it would make a machine-global answer
depend on the current directory, so ``dgml workspace list`` would change with whichever
workspace resolved first; and it is the same hazard
:mod:`dgml_core.workspace_config` already refuses for the ``[workspace]`` identity
block. Enforcement is mechanical rather than by convention — ``workspaces`` is not a
:class:`~dgml_core.models_config.ConfigSection` and is not declared on the settings
class, so ``extra="ignore"`` drops it before any loader can see it.
"""

from __future__ import annotations

import functools
import tomllib
from typing import Any

from .errors import CorruptMetadata, WorkspacesConfigInvalid
from .provider import import_provider_class
from .storage import user_config_path
from .workspaces_store import WORKSPACES_SECTION, WorkspacesConfig, WorkspacesStore

#: The bundled default: a folder per workspace on local disk. Implements the whole
#: interface with no dependencies, so a machine that has never configured anything
#: still has a working list of workspaces.
DEFAULT_WORKSPACES_PROVIDER = "dgml_core.workspaces_local:LocalDirWorkspacesStore"


def load_workspaces_config() -> WorkspacesConfig:
    """The machine's ``[workspaces]`` section, from the user config alone.

    A missing file, a missing table, or a table with no ``provider`` all resolve to the
    bundled local-disk store — the last so ``[workspaces] root = "/data/ws"`` means
    what it obviously means rather than erroring on a missing provider."""
    path = user_config_path()
    try:
        with path.open("rb") as fh:
            parsed: dict[str, Any] = tomllib.load(fh)
    except FileNotFoundError:
        return WorkspacesConfig(provider=DEFAULT_WORKSPACES_PROVIDER)
    except tomllib.TOMLDecodeError as exc:
        raise CorruptMetadata(f"invalid TOML in {path}: {exc}") from exc
    except OSError as exc:
        raise CorruptMetadata(f"could not read {path}: {exc}") from exc

    table = parsed.get(WORKSPACES_SECTION)
    if table is None:
        return WorkspacesConfig(provider=DEFAULT_WORKSPACES_PROVIDER)
    if not isinstance(table, dict):
        raise WorkspacesConfigInvalid(f"[{WORKSPACES_SECTION}] in {path} must be a table")

    provider = table.get("provider", DEFAULT_WORKSPACES_PROVIDER)
    if not isinstance(provider, str) or not provider.strip():
        raise WorkspacesConfigInvalid(f"'{WORKSPACES_SECTION}.provider' must be a non-empty string")
    options = {k: v for k, v in table.items() if k != "provider"}
    return WorkspacesConfig(provider=provider, options=options)


def make_workspaces_store(config: WorkspacesConfig) -> WorkspacesStore:
    """Instantiate the store named by ``config`` (resolve provider → ``parse_config``
    → construct, where the provider's lazy SDK import happens)."""
    cls = import_provider_class(
        config.provider,
        WorkspacesStore,
        kind=WORKSPACES_SECTION,
        default_hint=DEFAULT_WORKSPACES_PROVIDER,
    )
    store: WorkspacesStore = cls(cls.parse_config(config))
    return store


@functools.lru_cache(maxsize=1)
def default_workspaces_store() -> WorkspacesStore:
    """The machine's store of workspaces.

    Memoized, for two reasons: there is exactly one ``[workspaces]`` table per user
    config, so re-reading it per call is waste; and a networked backend holds a client
    that should be one per process rather than one per ``Workspace`` object.

    **Tests must call** ``default_workspaces_store.cache_clear()`` — a fixture that
    repoints ``$DGML_WORKSPACES`` or the user config after this has been called once
    would otherwise get the previous store."""
    return make_workspaces_store(load_workspaces_config())
