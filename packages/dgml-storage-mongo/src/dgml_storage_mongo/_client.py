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

"""Connection and option handling shared by the blob and document stores.

Both stores take the same three identity options and the same environment
variable, so the validation and the URI construction live here once rather than
being spelled twice (and drifting).
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

from dgml_core.errors import DgmlError, StorageConfigInvalid
from dgml_core.storage_service import StorageConfig

#: Environment variable holding the full MongoDB connection string, including
#: any credentials. Deliberately not a config key — see :mod:`.store`.
MONGO_URI_ENV = "DGML_MONGO_URI"

#: Checked *before* ``DGML_MONGO_URI`` by the store of workspaces. Two variables rather
#: than one because a URI is used verbatim, before a database is selected, so a single
#: one cannot express both a workspace's data credentials and the workspaces store's —
#: which is the point of keeping the two in separate databases (see :mod:`.workspaces`).
WORKSPACES_URI_ENV = "DGML_WORKSPACES_MONGO_URI"

#: The identity options every store in this package accepts. Host, port, and
#: database — never a credential.
IDENTITY_FIELDS = frozenset({"mongo_host", "mongo_port", "mongo_database"})

#: Outer part of every workspace-data collection name when the config sets no
#: ``prefix`` — see :func:`workspace_namespace`.
DEFAULT_PREFIX = "dgml"

#: What a workspace-data store's ``prefix`` may look like. Excludes ``$`` and ``.``: ``$``
#: is not allowed in a collection name, and a ``.`` could make a name that ends like
#: GridFS's own ``<bucket>.files`` / ``<bucket>.chunks``.
_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")

#: Longest ``prefix`` accepted, so the full collection name stays well inside Mongo's
#: 255-byte namespace limit.
MAX_PREFIX_LEN = 16


def validate_prefix(
    options: Mapping[str, Any], *, error: type[DgmlError] = StorageConfigInvalid
) -> None:
    """Check the optional ``prefix`` option, or raise ``error``."""
    prefix = options.get("prefix")
    if prefix is None:
        return
    if not isinstance(prefix, str) or not _PREFIX_RE.match(prefix):
        raise error(
            f"'prefix' must start with a letter or digit and contain only letters, "
            f"digits, '_' and '-' (got {prefix!r})"
        )
    if len(prefix) > MAX_PREFIX_LEN:
        raise error(f"'prefix' must be at most {MAX_PREFIX_LEN} characters (got {prefix!r})")


def require_workspace_id(provider_name: str, config: StorageConfig) -> None:
    """Refuse a config with no workspace id, rather than default one.

    Names built without the id would be shared by every id-less workspace on the
    database, and would move the moment this workspace got an id."""
    if not config.workspace_id:
        raise StorageConfigInvalid(
            f"provider {provider_name!r} needs the workspace's id to name its collections, "
            f"and this workspace records none"
        )


def workspace_namespace(config: StorageConfig) -> str:
    """``<prefix>_<workspace id>``, with ``prefix`` defaulting to ``dgml``.

    The id is always part of it, so any number of workspaces — and other applications —
    can share one database. Documents and GridFS's bucket both live under it, so one
    workspace's data stays together."""
    prefix = config.options.get("prefix") or DEFAULT_PREFIX
    return f"{prefix}_{config.workspace_id}"


def prefixed(namespace: str, name: str) -> str:
    """``name`` in ``namespace``: ``<namespace>_<name>``.

    Every collection a workspace-data store touches goes through here, so one
    workspace's names can never meet another's in a shared database."""
    return f"{namespace}_{name}"


def validate_identity(
    provider_name: str,
    options: Mapping[str, Any],
    *,
    section: str = "storage",
    error: type[DgmlError] = StorageConfigInvalid,
) -> None:
    """Check the shared ``mongo_*`` identity options, or raise ``error``.

    ``section`` and ``error`` let the store of workspaces reuse this while reporting
    against the ``[workspaces]`` table it was actually configured from."""
    database = options.get("mongo_database")
    if not isinstance(database, str) or not database.strip():
        raise error(f"[{section}] provider {provider_name!r} requires a 'mongo_database'")
    host = options.get("mongo_host")
    if host is not None and not isinstance(host, str):
        raise error("'mongo_host' must be a string")
    port = options.get("mongo_port")
    # bool is an int subclass, and `mongo_port = true` is a typo, not a port.
    if port is not None and (isinstance(port, bool) or not isinstance(port, int)):
        raise error("'mongo_port' must be an integer")


def connect(options: Mapping[str, Any], *, uri_env: str | None = None) -> Any:
    """The configured database handle.

    Authentication is all-or-nothing via the environment: ``DGML_MONGO_URI`` is
    used verbatim when set (credentials, TLS, replica set and all), otherwise
    ``mongo_host``:``mongo_port`` is contacted with no auth. There is
    deliberately no username/password config key — see :mod:`.store`.

    ``uri_env`` names a variable checked first, so one process can hold separate
    credentials for a workspace's data and for the store of workspaces.

    Untyped return: ``pymongo``'s ``Database`` is generic over the document type
    and the stores hold it as ``Any`` rather than thread that parameter through
    a sample.
    """
    # Lazy SDK import with an actionable message, per the ABC's contract: a
    # workspace that never opens one of these stores must not need pymongo.
    try:
        from pymongo import MongoClient
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise DgmlError("the mongo backend needs pymongo: pip install dgml-storage-mongo") from exc

    uri = os.environ.get(uri_env or "") or os.environ.get(MONGO_URI_ENV)
    if not uri:
        host = str(options.get("mongo_host") or "localhost")
        port = int(options.get("mongo_port") or 27017)
        uri = f"mongodb://{host}:{port}"
    return MongoClient(uri)[str(options["mongo_database"])]
