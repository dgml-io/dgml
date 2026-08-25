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

"""Both storage roles on one MongoDB database.

The resolver's **flat form** is one ``[storage.<name>]`` table whose single
top-level ``provider`` serves blobs *and* documents::

    [storage.acme]
    provider       = "dgml_storage_mongo:MongoGridFSStore"
    mongo_database = "dgml_dev"

One database holds both: blobs in GridFS's ``<bucket>.files`` /
``<bucket>.chunks``, documents in collections named for
:class:`~dgml_core.layout.Collection` members. The names cannot collide, which
is what makes sharing a database safe.

The flat form gives you one *instance* as well as one class: both roles resolve
to the same config, so ``Workspace`` builds the provider once and serves
``blobs`` and ``docs`` from it — one ``MongoClient`` per workspace, not one per
role.

What this does **not** buy you: atomic cascades. Both halves living in one
database makes a transaction spanning a document delete and its blob deletes
*theoretically* available, but ``BlobStore`` and ``DocStore`` expose no
transaction handle, and ``delete_blobs`` is documented to run last precisely
because cross-store atomicity is assumed impossible. Capturing it would take an
interface change in ``dgml-core``, not a provider.
"""

from __future__ import annotations

from dgml_core.storage_service import StorageConfig

from ._client import IDENTITY_FIELDS, connect
from .gridfs_store import MongoGridFSBlobStore
from .store import MongoDocStore

__all__ = ["MongoGridFSStore"]


class MongoGridFSStore(MongoGridFSBlobStore, MongoDocStore):
    """Blobs *and* documents in one MongoDB database.

    Inherits each role's methods unchanged from
    :class:`~dgml_storage_mongo.gridfs_store.MongoGridFSBlobStore` and
    :class:`~dgml_storage_mongo.store.MongoDocStore`; only construction is
    overridden, to bind both roles to a single client.
    """

    name = "mongo-gridfs"
    config_fields = IDENTITY_FIELDS | {"mongo_bucket"}

    @classmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        # The blob half owns the extra ``mongo_bucket`` option, so its validation
        # is the superset — delegate rather than restate it.
        return MongoGridFSBlobStore.parse_config(config)

    def __init__(self, config: StorageConfig) -> None:
        db = connect(config.options)
        self._bind_gridfs(db, config.options.get("mongo_bucket"))
        self._bind_docs(db)
