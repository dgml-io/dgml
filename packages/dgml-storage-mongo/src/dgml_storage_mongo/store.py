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

"""A sample :class:`~dgml_core.storage_service.DocStore`: documents in MongoDB.

DGML documents (file and docset manifests, assignments, extraction stats, the
usage log) live in MongoDB collections — the collection API ``DocStore`` was
modelled on, so nearly every method is a one-line delegation. A reference to
copy, not a tuned production store. Pair it with a
:class:`~dgml_core.storage_service.BlobStore`: ``dgml-storage-s3``, the bundled
local store, or this package's own
:class:`~dgml_storage_mongo.gridfs_store.MongoGridFSBlobStore` (see
:class:`~dgml_storage_mongo.MongoGridFSStore` for both roles at once).

Note that per-page text is a **blob**, not a document — it lives under
``files/<id>/page_text/`` and is read through the blob store.

**Sample, not supported.** Resolved by dotted path like any third party's own::

    [storage.acme.docs]
    provider = "dgml_storage_mongo:MongoDocStore"
    mongo_host = "localhost"
    mongo_database = "dgml_dev"

Credentials
-----------

**Credentials are read from the environment, never from DGML config.** Mongo
reads ``DGML_MONGO_URI`` if set (the full connection string, including any
credentials); otherwise it connects to ``mongo_host``:``mongo_port`` with no
auth. Config carries *identity only* — host, port, database.

There is deliberately no username/password config key: a half-credential in
config (a username with no way to supply the password) builds a URI pymongo
rejects, and an inline password would sit in the workspace's plaintext
``config.toml`` — which now lives beside the workspace and is likely to be committed
or synced. ``dgml_core.storage_resolve`` only treats option names matching
``("key", "secret", "token", "password", "credential")`` as secrets, and ``mongo_uri``
matches none of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dgml_core.errors import InvalidArgument
from dgml_core.layout import Collection
from dgml_core.storage_service import DocStore, StorageConfig

from ._client import (
    IDENTITY_FIELDS,
    MONGO_URI_ENV,
    connect,
    validate_identity,
)

__all__ = ["MONGO_URI_ENV", "MongoDocStore"]


class MongoDocStore(DocStore):
    """DGML documents in MongoDB collections."""

    name = "mongo"
    config_fields = IDENTITY_FIELDS

    # ---- configuration ----

    @classmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        cls._check_no_extra_fields(config.options)
        validate_identity(cls.name, config.options)
        return config

    def __init__(self, config: StorageConfig) -> None:
        # Authentication is all-or-nothing via the environment (see
        # ``_client.connect``). There is deliberately no username/password
        # config key: a half-credential in config builds a URI pymongo rejects,
        # and adding the password key is exactly the plaintext-registry leak
        # this design avoids.
        self._bind_docs(connect(config.options))

    def _bind_docs(self, db: Any) -> None:
        """Attach the document collections to an open database.

        Split out of ``__init__`` so the combined stores can bind both roles to
        one client instead of calling each parent's ``__init__`` and opening a
        second. That is a *class*-level concern: a workspace whose roles resolve
        to the same config also shares one instance between them, so the flat
        form holds a single client either way."""
        self._db: Any = db

    # ---- Documents (MongoDB) ----
    #
    # Mongo needs an ``_id``; the DGML document body must not carry one. Every
    # write sets it and every read strips it, so ``get_doc`` returns exactly the
    # body ``put_doc`` was given — ``FileRecord.from_json`` and friends do not
    # expect an extra field.

    def put_doc(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None:
        self._db[collection].replace_one({"_id": doc_id}, {**doc, "_id": doc_id}, upsert=True)

    def get_doc(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        found = self._db[collection].find_one({"_id": doc_id})
        return _strip_id(found) if found is not None else None

    def find_docs(self, collection: str, query: Mapping[str, Any]) -> list[dict[str, Any]]:
        # An empty query means the whole collection, not "no results".
        return [_strip_id(doc) for doc in self._db[collection].find(dict(query))]

    def delete_doc(self, collection: str, doc_id: str) -> None:
        self._db[collection].delete_one({"_id": doc_id})

    def delete_docs(self, collection: str, query: Mapping[str, Any]) -> int:
        return int(self._db[collection].delete_many(dict(query)).deleted_count)

    def append_doc(self, collection: str, doc: dict[str, Any]) -> None:
        # Append-only (the usage log): no id, never fetched or replaced
        # individually. Mongo mints its own ``_id``, which reads strip.
        #
        # Rejected for every other collection, matching LocalStore. Mongo would
        # happily insert an id-less document anywhere, but appending to an
        # addressed collection is a caller bug, and a caller bug that raises on
        # one backend and passes on another is the whole class of defect this
        # package exists to surface.
        if collection != Collection.USAGE:
            raise InvalidArgument(
                f"{collection!r} is not an append-only collection; use put_doc "
                f"(append-only: {Collection.USAGE.value!r})"
            )
        self._db[collection].insert_one(dict(doc))


def _strip_id(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The document body without Mongo's ``_id`` routing key."""
    return {k: v for k, v in doc.items() if k != "_id"}
