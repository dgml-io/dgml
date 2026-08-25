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

"""A sample :class:`~dgml_core.storage_service.BlobStore`: blobs via **GridFS**.

Blobs live in a GridFS bucket, via the ``gridfs`` module that ships inside
``pymongo`` — there is nothing extra to install. GridFS exists to get past the
16 MB BSON document cap, which a source PDF has every right to exceed.

Delegating to the spec rather than hand-rolling a chunk layout buys two things:

- **Cross-driver interoperability.** GridFS is a spec every official MongoDB
  driver implements identically, so a Node or Go service can read blobs this
  store wrote.
- **Ranged reads.** GridFS download streams seek, so a large artifact can be
  served by byte range without materializing it. Nothing in the DGML pipeline
  asks for that today, but a service in front of a workspace might.

It costs the revision bookkeeping in :ref:`the note below <gridfs-revisions>`,
which is the substance of this class.

Layout
------

Standard GridFS, so the collections are the spec's: ``<bucket>.files`` (one
document per revision, holding ``filename``, ``length``, ``chunkSize``,
``uploadDate``, ``metadata``) and ``<bucket>.chunks`` (``files_id``, ``n``,
``data``). Default bucket name ``blobs``, so ``blobs.files`` /
``blobs.chunks`` — neither collides with a
:class:`~dgml_core.layout.Collection` member, so blobs and documents can share
one database.

The blob key is the GridFS ``filename``. ``sha256`` is written into
``metadata`` at upload time, which keeps :meth:`sha256_blob` a single indexed
read here too.

.. _gridfs-revisions:

Revisions, and why reads do not use ``open_download_stream_by_name``
-------------------------------------------------------------------

GridFS addresses a blob by ``filename`` **plus** ``uploadDate``; a
``BlobStore`` addresses it by key alone. Reconciling the two is the whole
substance of this class:

``upload_from_stream`` *versions* — it mints a new file id and leaves the prior
revision in place. So :meth:`put_blob` captures the prior revision ids before
uploading and deletes them after. That imposes replace semantics, and orders the
write so the new bytes land complete before the old are collected: a crash
leaves an orphaned revision (recoverable) rather than a key with no bytes
behind it.

``open_download_stream_by_name`` is deliberately **not** used. Its ``revision``
parameter is defined against ``uploadDate``, which is millisecond resolution:
two writes to one key inside the same millisecond carry an identical
``uploadDate``, leaving ``revision=-1`` ("the most recent") genuinely undefined
— and the failure mode is a silent stale read. Reads here resolve the revision
explicitly and break a tie on ``_id`` instead, then stream by file id.

That narrows the window rather than closing it: ``_id`` (an ``ObjectId``) is
monotonic within a process but not across processes writing in the same second.
DGML never has two writers on one key — ``staged_write`` regenerates a whole
prefix from a single process — so this is sound for the pipeline. It is,
however, the one place this backend is weaker than a store keyed by the blob key
alone, where no ordering question exists at all. Worth remembering before
pointing two writers at one workspace.

Torn reads are retried, not raised: GridFS reports a collected revision as
``CorruptGridFile``, and this store re-resolves and converges on the current
bytes rather than handing that exception to a caller mid-pipeline.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import IO, Any

from dgml_core.errors import DgmlError
from dgml_core.hashing import sha256_file
from dgml_core.storage_service import BlobStore, StorageConfig

from ._client import IDENTITY_FIELDS, connect, validate_identity

#: Default GridFS bucket name, overridable with the ``mongo_bucket`` option so
#: several workspaces can share a database.
DEFAULT_BUCKET = "blobs"

#: Bytes per GridFS chunk, overriding the 255 KiB GridFS default: at 255 KiB a
#: 300 DPI page image is a dozen documents, and page images are the bulk of a
#: workspace.
CHUNK_BYTES = 1024 * 1024

#: Bounded retries when a concurrent overwrite collects the revision mid-read.
_READ_ATTEMPTS = 3


class MongoGridFSBlobStore(BlobStore):
    """Blobs in a GridFS bucket. Inherits the path bridge
    (:meth:`~dgml_core.storage_service.BlobStore.materialize` and friends) from
    :class:`~dgml_core.storage_service.BlobStore`."""

    name = "mongo-gridfs"
    config_fields = IDENTITY_FIELDS | {"mongo_bucket"}

    # ---- configuration ----

    @classmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        cls._check_no_extra_fields(config.options)
        validate_identity(cls.name, config.options)
        bucket = config.options.get("mongo_bucket")
        if bucket is not None and (not isinstance(bucket, str) or not bucket.strip()):
            raise _invalid("'mongo_bucket' must be a non-empty string")
        return config

    def __init__(self, config: StorageConfig) -> None:
        self._bind_gridfs(connect(config.options), config.options.get("mongo_bucket"))

    def _bind_gridfs(self, db: Any, bucket: str | None = None) -> None:
        """Attach a GridFS bucket to an open database.

        Split out of ``__init__`` so :class:`~dgml_storage_mongo.MongoGridFSStore`
        can bind both roles to one client instead of opening a second. That is a
        *class*-level concern: a workspace whose roles resolve to the same config
        also shares one instance between them, so the flat form holds a single
        client either way."""
        # Lazy, like every SDK import here: ``gridfs`` ships inside pymongo, so
        # it is present whenever pymongo is, but importing at module scope would
        # make this module unimportable without it.
        import gridfs

        name = str(bucket or DEFAULT_BUCKET)
        self._bucket: Any = gridfs.GridFSBucket(db, bucket_name=name, chunk_size_bytes=CHUNK_BYTES)
        self._files: Any = db[f"{name}.files"]

    # ---- internals ----

    @staticmethod
    def _prefix_query(prefix: str) -> dict[str, Any]:
        """Match every key under ``prefix``. ``re.escape`` is load-bearing —
        keys hold filenames, and ``page_1.png`` read as a regex would also
        match ``page_1Xpng``."""
        return {"filename": {"$regex": f"^{re.escape(prefix)}"}}

    def _revisions(self, key: str) -> list[Any]:
        """Every live revision id for ``key``, oldest-looking first."""
        return [
            doc["_id"]
            for doc in self._files.find({"filename": key}, {"_id": 1}).sort(
                [("uploadDate", 1), ("_id", 1)]
            )
        ]

    def _current(self, key: str) -> dict[str, Any]:
        """The revision a read should use.

        Explicitly resolved rather than left to ``revision=-1``: the tie on
        ``uploadDate`` is broken by ``_id`` so a same-millisecond rewrite cannot
        silently serve stale bytes."""
        doc = self._files.find_one(
            {"filename": key},
            sort=[("uploadDate", -1), ("_id", -1)],
        )
        if doc is None:
            raise FileNotFoundError(f"no blob at key {key!r}")
        return dict(doc)

    def _write(self, key: str, source: IO[bytes], sha256: str) -> None:
        """Upload a new revision, then collect the ones it replaces.

        ``sha256`` is passed in rather than teed off the upload: it goes into
        ``metadata`` with the revision, in one write, so a reader can never see
        a revision whose recorded digest has not been filled in yet."""
        stale = self._revisions(key)
        new_id = self._bucket.upload_from_stream(key, source, metadata={"sha256": sha256})
        # Only now — a reader that already resolved an old revision keeps reading
        # consistent old bytes until this point.
        for old in stale:
            if old != new_id:
                self._bucket.delete(old)

    def _read(self, key: str) -> bytes:
        """The current revision's bytes, retrying if a concurrent overwrite
        collected it mid-read."""
        import gridfs

        for _ in range(_READ_ATTEMPTS):
            doc = self._current(key)
            try:
                with self._bucket.open_download_stream(doc["_id"]) as stream:
                    return bytes(stream.read())
            except (gridfs.NoFile, gridfs.errors.CorruptGridFile):
                continue  # the revision was collected under us; re-resolve
        raise DgmlError(
            f"blob {key!r} kept changing while being read: its revision was replaced "
            f"{_READ_ATTEMPTS} times. Something else is writing this key concurrently."
        )

    # ---- Blobs ----

    def put_blob(self, key: str, data: bytes) -> None:
        import io

        self._write(key, io.BytesIO(data), hashlib.sha256(data).hexdigest())

    def get_blob(self, key: str) -> bytes:
        return self._read(key)

    def delete_blob(self, key: str) -> None:
        for revision in self._revisions(key):
            self._bucket.delete(revision)

    def blob_exists(self, key: str) -> bool:
        return self._files.find_one({"filename": key}, {"_id": 1}) is not None

    def list_blobs(self, prefix: str) -> list[str]:
        # De-duplicated: GridFS holds one document per *revision*, so a key
        # mid-overwrite would otherwise be reported twice.
        return sorted(
            {str(doc["filename"]) for doc in self._files.find(self._prefix_query(prefix))}
        )

    def upload_blob(self, key: str, src: Path) -> None:
        # Hashed in a separate chunked pass rather than teed off the upload, so
        # the digest is known before the revision exists. Costs a second read of
        # a local file; buys a revision that is never briefly digest-less.
        digest = sha256_file(src)
        with src.open("rb") as handle:
            self._write(key, handle, digest)

    def download_blob(self, key: str, dest: Path) -> None:
        data = self._read(key)  # raises FileNotFoundError before touching dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def delete_blobs(self, prefix: str) -> None:
        # Cascades call this last (see WorkspaceOps): the authoritative record
        # dies first, so an interrupted cascade leaves recoverable orphaned bytes.
        for doc in list(self._files.find(self._prefix_query(prefix), {"_id": 1})):
            self._bucket.delete(doc["_id"])

    # ---- Derived reads ----

    def sha256_blob(self, key: str) -> str:
        """The digest recorded at upload time, from GridFS ``metadata`` — one
        indexed read, no download.

        Still the plain SHA-256 of the exact stored bytes, as the ABC requires:
        it is computed over the same stream GridFS persisted. This is the one
        place this backend beats an object store, where the only trustworthy
        digest is a full re-read — S3's multipart ETag and composite
        ``ChecksumSHA256`` are checksums-of-checksums and explicitly not this
        value."""
        recorded = (self._current(key).get("metadata") or {}).get("sha256")
        if not recorded:
            # A blob written by something other than this store (or by an older
            # version of it) has no recorded digest. Fall back to hashing rather
            # than reporting a wrong value into an attestation leaf.
            return hashlib.sha256(self._read(key)).hexdigest()
        return str(recorded)


def _invalid(message: str) -> Exception:
    from dgml_core.errors import StorageConfigInvalid

    return StorageConfigInvalid(message)
