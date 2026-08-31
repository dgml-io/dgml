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

"""A sample :class:`~dgml_core.workspaces_store.WorkspacesStore`: the list of
workspaces in MongoDB.

This is the payoff of making the list pluggable: point two machines at one database
and they see one set of workspaces. ``dgml --workspace ws_…`` opens the same workspace
on a laptop and in CI, with no config file passed between them::

    # ~/.config/dgml/config.toml
    [workspaces]
    provider = "dgml_storage_mongo:MongoWorkspacesStore"
    mongo_host = "localhost"
    mongo_database = "dgml_workspaces"

One document per workspace, ``_id`` = its ``workspace_id``:

.. code-block:: javascript

    {
      _id:         "ws_7qxdm2pjk3n5rwts",
      config_toml: "…verbatim UTF-8 text…",   // AUTHORITATIVE
      revision:    7,                          // CAS token, see below
      name:         "Acme Contracts",          // ↓ derived, regenerated on every write
      organization: "acme",
      storage_service: "bym",
      created_at: "2026-08-26T18:04:11Z",
      updated_at: ISODate(…),
      schema_version: 1
    }

Why the config is one **verbatim text** field
---------------------------------------------

Not because parsing it would be inconvenient — because it does not work. Checked
against ``bson``: TOML's bare local date (``d = 2026-01-01``) parses to a
``datetime.date``, which BSON refuses to encode at all, and an offset datetime comes
back with microseconds truncated and the zone normalized to UTC. Both are legal in a
``[generation]`` or ``[ocr]`` override, so a parsed-BSON store would **reject a valid
config.toml**, failing at write time far from the line that caused it.

Beyond that: this repo owns no TOML *writer* to round-trip back through —
``dgml_core.workspace_config._toml_value`` deliberately raises rather than guess —
nothing anywhere queries inside a config (every reader takes a whole table), and text
is what keeps ``workspace_config``'s promise that a user's comments and key order
survive a write *identical on both backends*. A comment that survives on local disk
and vanishes here is exactly the class of defect this package exists to surface.

Derived fields and why they are safe
------------------------------------

``name``, ``organization``, ``storage_service`` and ``created_at`` are duplicated out
of the config so ``dgml workspace list`` is one query instead of N parses. They are
regenerated from the text on **every** write, in the base class, so the two backends
cannot render a workspace differently. Nothing reads them as authority.

Deliberately *not* denormalized: ``storage_fingerprint``. It is the seal, and a
queryable copy of a seal is an invitation to compare against the copy instead of the
thing. Also no path of any kind — where a workspace's files sit is per-machine, and a
shared column recording it is precisely the mistake the per-machine JSON index made.

Lost updates, and why this one needs a CAS
------------------------------------------

``write_identity`` and ``reseal`` are read-modify-write over the *whole* config text.
So a lost update here does not drop the field being written — it discards the other
machine's ``[storage]`` table, ``[models]`` edits and comments, and the result still
parses. The old per-machine index tolerated interleaved writes because its rows were a
cache; that argument does not transfer to authority.

Hence ``revision``: every write is conditional on the revision that was read, and a
mismatch raises :class:`~dgml_core.errors.WorkspacesWriteConflict` rather than
overwriting. ``updated_at`` is for humans and ordering only and must **never** be the
predicate — this package already contains the case study for why, in
``gridfs_store``'s notes on millisecond ``uploadDate`` ties.

Credentials
-----------

``$DGML_WORKSPACES_MONGO_URI``, else ``$DGML_MONGO_URI``, else
``mongo_host``:``mongo_port`` unauthenticated. Never a config key, even though this
config lives in the user's own ``~/.config/dgml/config.toml`` rather than beside a
workspace: ``dgml_core.storage_resolve``'s secret-name filter is *fingerprint*
machinery and nothing hashes these options, so an inline URI would buy no protection
whatsoever — it would simply be a password in a file. And ``~/.config`` is arguably
*more* likely to be synced between machines than a workspace directory, which is the
very scenario a shared list of workspaces exists for.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from dgml_core.errors import (
    CorruptMetadata,
    WorkspacesConfigInvalid,
    WorkspacesUnavailable,
    WorkspacesWriteConflict,
)
from dgml_core.layout import Collection
from dgml_core.workspaces_store import WorkspacesConfig, WorkspacesStore

from ._client import IDENTITY_FIELDS, WORKSPACES_URI_ENV, connect, validate_identity

#: Default collection. Prefixed rather than a bare ``workspaces`` because that is one
#: character from ``Collection.WORKSPACE`` — safe today, but not a thing to rely on in a
#: database that may also hold a workspace's own documents.
DEFAULT_COLLECTION = "dgml_workspaces"

#: This document shape's own version, independent of a workspace's schema_version.
CATALOG_SCHEMA_VERSION = 1


class MongoWorkspacesStore(WorkspacesStore):
    """The list of workspaces, and each one's ``config.toml``, in one collection."""

    name = "mongo-workspaces"
    config_fields = IDENTITY_FIELDS | {"mongo_collection"}

    # ---- configuration ----

    @classmethod
    def parse_config(cls, config: WorkspacesConfig) -> WorkspacesConfig:
        cls._check_no_extra_fields(config.options)
        validate_identity(
            cls.name,
            config.options,
            section=cls.config_section,
            error=WorkspacesConfigInvalid,
        )
        collection = config.options.get("mongo_collection")
        if collection is None:
            return config
        if not isinstance(collection, str) or not collection.strip():
            raise WorkspacesConfigInvalid(
                "'workspaces.mongo_collection' must be a non-empty string"
            )
        # Executable form of the collision argument the GridFS store only makes in prose:
        # this collection may share a database with a workspace's own documents, so it
        # must not be able to shadow one of them or a GridFS bucket.
        if collection in {member.value for member in Collection}:
            raise WorkspacesConfigInvalid(
                f"'workspaces.mongo_collection' cannot be {collection!r}: that is a "
                f"collection a workspace's own documents use, and the two may share a "
                f"database"
            )
        if collection.endswith((".files", ".chunks")):
            raise WorkspacesConfigInvalid(
                f"'workspaces.mongo_collection' cannot be {collection!r}: '.files' and "
                f"'.chunks' are GridFS's own collections"
            )
        return config

    def __init__(self, config: WorkspacesConfig) -> None:
        db = connect(config.options, uri_env=WORKSPACES_URI_ENV)
        self._db = db
        self._name = str(config.options.get("mongo_collection") or DEFAULT_COLLECTION)
        self._docs = db[self._name]

    def label(self) -> str:
        return f"mongo:{self._db.name}/{self._name}"

    @contextmanager
    def _reachable(self) -> Iterator[None]:
        """Turn a connection failure into :class:`WorkspacesUnavailable`.

        pymongo connects lazily, so an unreachable server surfaces on the first
        *operation* rather than at construction — which means every method needs this,
        and without it the failure arrives as an ``INTERNAL_ERROR`` carrying a paragraph
        of driver internals. This is the ordinary operational failure of a networked
        store (server down, wrong host, VPN off) and every dgml command needs the store
        to find a workspace, so it deserves a code a caller can act on."""
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        try:
            yield
        except (ServerSelectionTimeoutError, ConnectionFailure) as exc:
            raise WorkspacesUnavailable(
                f"cannot reach the store of workspaces at {self.label()}: "
                f"{type(exc).__name__}. Every command needs it to find a workspace — "
                f"check that the server is running and reachable, and that the "
                f"[workspaces] table in your user config (and $DGML_WORKSPACES_MONGO_URI "
                f"if set) name the right host."
            ) from exc

    # ---- the list of workspaces ----

    def read_config(self, workspace_id: str) -> tuple[str, int | None] | None:
        with self._reachable():
            doc = self._docs.find_one({"_id": workspace_id})
        if doc is None:
            return None
        text = doc.get("config_toml")
        if not isinstance(text, str):
            raise CorruptMetadata(
                f"{self.label()} document {workspace_id!r} has no 'config_toml' string; "
                f"something other than dgml wrote it"
            )
        revision = doc.get("revision")
        return text, revision if isinstance(revision, int) else None

    def write_config(
        self, workspace_id: str, text: str, *, expected_revision: int | None = None
    ) -> int | None:
        derived = self._derive(workspace_id, text)
        with self._reachable():
            return self._store(workspace_id, text, derived, expected_revision)

    def _store(
        self,
        workspace_id: str,
        text: str,
        derived: dict[str, Any],
        expected_revision: int | None,
    ) -> int | None:
        if expected_revision is None:
            # A first write, or a caller that read nothing. Upsert unconditionally: there
            # is no revision to be stale against.
            self._docs.update_one(
                {"_id": workspace_id},
                {
                    "$set": {"config_toml": text, **derived},
                    "$currentDate": {"updated_at": True},
                    "$setOnInsert": {
                        "revision": 1,
                        "schema_version": CATALOG_SCHEMA_VERSION,
                    },
                },
                upsert=True,
            )
            found = self._docs.find_one({"_id": workspace_id}, {"revision": 1})
            revision = (found or {}).get("revision")
            return revision if isinstance(revision, int) else None

        result = self._docs.update_one(
            {"_id": workspace_id, "revision": expected_revision},
            {
                "$set": {"config_toml": text, **derived},
                "$inc": {"revision": 1},
                "$currentDate": {"updated_at": True},
            },
        )
        if result.matched_count == 0:
            # One extra read purely so the message is accurate about which of the two
            # things happened; a caller retrying blindly would treat them identically.
            missing = self._docs.find_one({"_id": workspace_id}, {"_id": 1}) is None
            detail = (
                "it is no longer in the store"
                if missing
                else "another writer changed it since it was read"
            )
            raise WorkspacesWriteConflict(
                f"cannot write the config for {workspace_id} in {self.label()}: {detail}. "
                f"Its config is written whole, so overwriting would discard whatever that "
                f"writer changed. Re-run the command to work from the current config."
            )
        return expected_revision + 1

    def list_configs(self) -> dict[str, str]:
        with self._reachable():
            return self._list_configs()

    def _list_configs(self) -> dict[str, str]:
        return {
            str(doc["_id"]): doc["config_toml"]
            for doc in self._docs.find({}, {"config_toml": 1}).sort("_id", 1)
            if isinstance(doc.get("config_toml"), str)
        }

    def delete(self, workspace_id: str) -> bool:
        with self._reachable():
            return bool(self._docs.delete_one({"_id": workspace_id}).deleted_count)

    # ---- overrides that avoid work the defaults would waste ----

    def exists(self, workspace_id: str) -> bool:
        """Projected so a boolean does not fetch a whole config — minting an id calls
        this per candidate."""
        with self._reachable():
            return self._docs.find_one({"_id": workspace_id}, {"_id": 1}) is not None

    def list_ids(self) -> list[str]:
        with self._reachable():
            return [str(doc["_id"]) for doc in self._docs.find({}, {"_id": 1}).sort("_id", 1)]

    def list_entries(self) -> list[Any]:
        """One query against the projection, no config text transferred.

        Derived from the same ``_derive`` the writes use, so a row here and a row from
        the local backend describe a workspace identically."""
        from dgml_core.workspace_config import WorkspaceIdentity

        with self._reachable():
            return [
                WorkspaceIdentity(
                    workspace_id=str(doc["_id"]),
                    name=doc.get("name"),
                    organization=doc.get("organization"),
                    storage_service=doc.get("storage_service"),
                    created_at=doc.get("created_at"),
                )
                for doc in self._docs.find(
                    {}, {"name": 1, "organization": 1, "storage_service": 1, "created_at": 1}
                ).sort("_id", 1)
            ]

    # ---- internals ----

    @staticmethod
    def _derive(workspace_id: str, text: str) -> dict[str, Any]:
        """The queryable projection of ``text``. Regenerated on every write, never read
        as authority — see the module docstring."""
        from dgml_core.workspace_config import identity_from_text

        identity = identity_from_text(text, workspace_id=workspace_id)
        return {
            "name": identity.name,
            "organization": identity.organization,
            "storage_service": identity.storage_service,
            "created_at": identity.created_at,
        }
