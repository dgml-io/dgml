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
from collections.abc import Mapping
from typing import Any

from dgml_core.errors import DgmlError, StorageConfigInvalid

#: Environment variable holding the full MongoDB connection string, including
#: any credentials. Deliberately not a config key — see :mod:`.store`.
MONGO_URI_ENV = "DGML_MONGO_URI"

#: The identity options every store in this package accepts. Host, port, and
#: database — never a credential.
IDENTITY_FIELDS = frozenset({"mongo_host", "mongo_port", "mongo_database"})


def validate_identity(provider_name: str, options: Mapping[str, Any]) -> None:
    """Check the shared ``mongo_*`` identity options, or raise
    :class:`~dgml_core.errors.StorageConfigInvalid`."""
    database = options.get("mongo_database")
    if not isinstance(database, str) or not database.strip():
        raise StorageConfigInvalid(
            f"[storage] provider {provider_name!r} requires a 'mongo_database'"
        )
    host = options.get("mongo_host")
    if host is not None and not isinstance(host, str):
        raise StorageConfigInvalid("'mongo_host' must be a string")
    port = options.get("mongo_port")
    # bool is an int subclass, and `mongo_port = true` is a typo, not a port.
    if port is not None and (isinstance(port, bool) or not isinstance(port, int)):
        raise StorageConfigInvalid("'mongo_port' must be an integer")


def connect(options: Mapping[str, Any]) -> Any:
    """The configured database handle.

    Authentication is all-or-nothing via the environment: ``DGML_MONGO_URI`` is
    used verbatim when set (credentials, TLS, replica set and all), otherwise
    ``mongo_host``:``mongo_port`` is contacted with no auth. There is
    deliberately no username/password config key — see :mod:`.store`.

    Untyped return: ``pymongo``'s ``Database`` is generic over the document type
    and the stores hold it as ``Any`` rather than thread that parameter through
    a sample.
    """
    # Lazy SDK import with an actionable message, per the ABC's contract: a
    # workspace that never opens one of these stores must not need pymongo.
    try:
        from pymongo import MongoClient
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise DgmlError(
            "the mongo sample store needs pymongo: pip install dgml-storage-mongo"
        ) from exc

    uri = os.environ.get(MONGO_URI_ENV)
    if not uri:
        host = str(options.get("mongo_host") or "localhost")
        port = int(options.get("mongo_port") or 27017)
        uri = f"mongodb://{host}:{port}"
    return MongoClient(uri)[str(options["mongo_database"])]
