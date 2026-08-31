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

"""Pluggable workspace storage — two independent stores, blobs and documents.

A workspace's data is two unrelated kinds, each with its own small, familiar API
and its own pluggable backend:

- :class:`BlobStore` — opaque bytes (page images, PDFs, XML, schema files),
  modeled on the S3 object API (``put_blob`` / ``get_blob`` / ``list_blobs`` / …),
  plus a concrete path bridge for tools that demand a real filesystem path.
- :class:`DocStore` — JSON documents (manifests, page text, assignments, usage),
  modeled on the MongoDB collection API (``put_doc`` / ``get_doc`` / ``find_docs``
  / …).

The two are configured and resolved **independently**: a workspace can put its
blobs on one backend and its documents on another (e.g. S3 blobs + Mongo docs),
or run both on the bundled :class:`dgml_core.storage_local.LocalStore` — which
implements *both* interfaces over one directory (the zero-config default).

This module is **only the abstraction** — the two interfaces a third party
implements, plus the :class:`StorageConfig` each receives. Turning configuration
into live stores (reading ``[storage.<name>.blobs]`` / ``.docs`` templates,
importing the ``provider`` dotted paths, constructing the stores, and hashing
their identity) is the resolver's job and lives in
:mod:`dgml_core.storage_resolve`.

Writing your own store
----------------------

1. ``pip install dgml`` (the wheel — no repo clone).
2. Subclass :class:`BlobStore` **or** :class:`DocStore` (or both, like
   ``LocalStore``), implementing :meth:`~_StoreBase.parse_config` (call
   :meth:`~_StoreBase._check_no_extra_fields` first), ``__init__`` (lazy SDK
   import — raise an actionable error if a dependency is missing), and that
   interface's methods.
3. Make the class importable by the interpreter running dgml.
4. Point ``config.toml`` at it — ``[storage.<name>.blobs] provider =
   "your_pkg.mod:YourBlobStore"`` and/or ``[storage.<name>.docs] provider = …`` —
   see :func:`dgml_core.storage_resolve.load_store_configs`.

For a :class:`BlobStore`, the path bridge (:meth:`~BlobStore.materialize` and
friends) and :meth:`~BlobStore.sha256_blob` are concrete — you get working
versions from the abstract blob methods and only override them if your backend
can do better.
"""

from __future__ import annotations

import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import layout
from .hashing import sha256_file
from .provider import ProviderConfigFields


@dataclass(frozen=True)
class StorageConfig:
    """A resolved single-role ``storage`` config section (blobs *or* docs).

    ``provider`` is the dotted path identifying the store class. ``options`` holds
    the section's remaining (non-``provider``) fields verbatim — a provider's own
    settings (``bucket``, ``endpoint_url``, ``mongo_database``, …). ``root`` is the
    local workspace root, always available as bootstrap (the config names the store,
    so it cannot live inside it); a ``LocalStore`` writes under it, and a remote
    store may use it for temp staging.
    """

    provider: str
    root: Path
    options: Mapping[str, Any] = field(default_factory=dict)


class _StoreBase(ProviderConfigFields, ABC):
    """Config machinery shared by :class:`BlobStore` and :class:`DocStore`.

    Subclasses declare ``config_fields`` — the JSON keys they accept under a
    ``storage`` sub-table besides the universal ``provider`` — and are rejected for
    any other key by :meth:`~dgml_core.provider.ProviderConfigFields._check_no_extra_fields`
    (catches typos and stale fields). A concrete class may implement one interface (an
    S3 blob store) or both (``LocalStore``); it provides one ``parse_config`` /
    ``__init__`` either way.

    The field machinery itself lives in :class:`~dgml_core.provider.ProviderConfigFields`,
    shared with the ``[workspaces]`` providers; the defaults there already name this
    section, so nothing needs restating.
    """

    @classmethod
    @abstractmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        """Validate the provider's option fields and return the (possibly
        normalized) config. Call :meth:`_check_no_extra_fields` first; raise
        :class:`StorageConfigInvalid` for missing or malformed fields."""

    @abstractmethod
    def __init__(self, config: StorageConfig) -> None:
        """Set the store up from ``config``. Lazy-import any SDK here and raise an
        actionable :class:`dgml_core.errors.DgmlError` if it is missing."""


class BlobStore(_StoreBase):
    """A pluggable **blob** backend — opaque bytes addressed by key.

    Modeled on the S3 object API. The path bridge (:meth:`materialize` and friends)
    and :meth:`sha256_blob` are concrete, built purely on the abstract blob
    primitives, so every blob store gets working versions for free.
    """

    # ---- Blobs — modeled on the S3 object API (key -> bytes) ----

    @abstractmethod
    def put_blob(self, key: str, data: bytes) -> None:
        """Create or overwrite the blob at ``key`` (S3 ``put_object``)."""

    @abstractmethod
    def get_blob(self, key: str) -> bytes:
        """Return the blob at ``key``. Raise :class:`FileNotFoundError` if absent.

        Returns the whole blob in memory — fine for DGML's artifact sizes (PDFs,
        page images, schemas, one dgml.xml), which is the working assumption
        throughout. Use this only when the bytes themselves are needed (parsing
        XML, base64-encoding an image, ``json.loads``). A caller that only needs
        a digest should use :meth:`sha256_blob`; one that needs a real path
        should use :meth:`download_blob` / :meth:`materialize`. Both avoid
        holding the blob whole."""

    @abstractmethod
    def delete_blob(self, key: str) -> None:
        """Delete the blob at ``key``. A missing key is a no-op (idempotent)."""

    @abstractmethod
    def blob_exists(self, key: str) -> bool:
        """Whether a blob exists at ``key`` (S3 ``head_object``)."""

    @abstractmethod
    def list_blobs(self, prefix: str) -> list[str]:
        """All blob keys under ``prefix`` (S3 ``list_objects_v2``), sorted."""

    @abstractmethod
    def upload_blob(self, key: str, src: Path) -> None:
        """Store the file at ``src`` as the blob ``key`` (S3 ``upload_file``)."""

    @abstractmethod
    def download_blob(self, key: str, dest: Path) -> None:
        """Write the blob ``key`` to the local path ``dest`` (S3 ``download_file``)."""

    @abstractmethod
    def delete_blobs(self, prefix: str) -> None:
        """Delete every blob whose key is under ``prefix`` (an object store: list +
        batch-delete; ``LocalStore``: remove the blob files and prune now-empty
        directories). Documents are left untouched — a cascade delete composes this
        with ``delete_doc`` / ``delete_docs`` in the caller, so each store only ever
        does operations native to it (no store needs the blob/document layout). A
        prefix that matches nothing is a no-op.

        Callers must run this **last** in a cascade. That is the contract
        :class:`dgml_core.workspace_ops.WorkspaceOps` implements — *the
        authoritative record dies first*, so an interrupted cascade leaves
        orphaned bytes (recoverable) rather than a record pointing at bytes that
        are gone (indistinguishable from a valid entity). It also happens to be
        what lets ``LocalStore`` prune the emptied container, which it can only
        do once the documents beside those blobs are gone."""

    # ---- Path bridge — for tools that demand a real filesystem path ----
    #
    # Some pipeline steps speak *paths*, not bytes: ghostscript renders page
    # images to ``-sOutputFile=<dir>/page_%d.png``, pdfminer / the PDF converter
    # read a PDF path, ``lxml.etree.parse`` wants a path. These concrete helpers
    # bridge that gap on top of the blob primitives, so every store gets them for
    # free; ``LocalStore`` overrides each one for a zero-copy passthrough (the key
    # already *is* an on-disk path), keeping local I/O byte-for-byte identical to
    # the pre-store code.
    #
    # A remote store overriding these should stage under ``StorageConfig.root``
    # rather than the default ``tempfile`` location: ``TMPDIR`` is a RAM-backed
    # tmpfs on many container images, which would silently turn a bounded-memory
    # read back into a whole-blob allocation plus a copy.

    @contextmanager
    def materialize(self, key: str) -> Iterator[Path]:
        """Yield a real local path holding the blob at ``key`` for a
        path-only reader (ghostscript, pdfminer, ``lxml.etree.parse``).

        Default: download to a temp file, cleaned up on exit. ``LocalStore``
        yields the real file with no copy. Raises :class:`FileNotFoundError`
        if the blob is absent."""
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / Path(key).name
            self.download_blob(key, dest)
            yield dest

    @contextmanager
    def staged_write(self, key_prefix: str) -> Iterator[Path]:
        """Yield an empty local directory for a tool that emits a *batch* of
        files by path (ghostscript rendering a file's page images).

        **The prefix is replaced, not added to.** On clean exit the blobs under
        ``key_prefix`` are *exactly* the files written into the yielded
        directory: everything written is stored (preserving relative paths) and
        any pre-existing blob under the prefix that was not rewritten is
        deleted. If the body raises, nothing is persisted and the prefix is left
        as it was.

        Replacement is part of the contract rather than an implementation
        detail because the callers regenerate a whole set at once — re-render a
        document whose page count dropped from 10 to 5 and the stale
        ``page_6..10`` must not survive. They would otherwise be hashed into the
        file's attestation, so a purely additive implementation makes the Merkle
        root depend on which backend the workspace happens to live on.

        A store overriding this must keep both halves of the contract: an
        **empty** directory on entry, and an exact replacement on exit."""
        prefix = key_prefix.rstrip("/")
        stale = set(self.list_blobs(prefix + "/"))
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            yield staging
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(staging).as_posix()
                    key = layout.pair_id(prefix, rel)
                    self.upload_blob(key, path)
                    stale.discard(key)
            for key in sorted(stale):
                self.delete_blob(key)

    @contextmanager
    def materialize_dir(self, prefix: str) -> Iterator[Path]:
        """Yield a local directory holding every blob under ``prefix`` (each at
        its path relative to ``prefix``), for a tool that *scans a directory* of
        files (OCR reading a file's rendered page images).

        Default: download the matching blobs into a temp dir, cleaned up on
        exit. ``LocalStore`` yields the real directory with no copy. The
        directory may be empty/absent if nothing matches — the caller handles
        that (OCR raises its own \"no page images\" error)."""
        base = prefix.rstrip("/") + "/"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for key in self.list_blobs(base):
                self.download_blob(key, out / key[len(base) :])
            yield out

    @contextmanager
    def working_dir(self, prefix: str) -> Iterator[Path]:
        """Yield a local, read-write working directory synced with ``prefix``:
        download its blobs in on entry, upload the directory's files back out on
        exit. For a read-modify-write working area the pipeline reloads across
        runs (the generation ``cache/``).

        As with :meth:`staged_write`, the sync back is a **replacement**: a blob
        the body deleted locally is deleted from the store, not silently
        resurrected on the next run.

        The yielded directory is named after the last segment of ``prefix`` and
        lives inside a fresh temp dir, so its *parent* is a stable per-call
        scratch location — a sibling artifact written next to it (generation's
        ``schema.json``) has somewhere to go, and is deliberately *not* synced.
        Default: temp dir, downloaded in and uploaded out. ``LocalStore`` yields
        the real directory (no copy, no sync — writes and deletes already land
        in the store).

        Unlike :meth:`staged_write`, a crash does not roll back identically
        across stores: the default persists nothing (the upload runs after the
        ``yield``, not in a ``finally``), while ``LocalStore`` has already
        written in place. That is tolerated because the only caller is a
        regenerable cache — do not use this for artifacts that must not be
        half-written."""
        base = prefix.rstrip("/") + "/"
        segment = prefix.rstrip("/").rsplit("/", 1)[-1] or "data"
        stale = set(self.list_blobs(base))
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / segment
            work.mkdir(parents=True, exist_ok=True)
            for key in stale:
                self.download_blob(key, work / key[len(base) :])
            yield work
            for path in sorted(work.rglob("*")):
                if path.is_file():
                    key = base + path.relative_to(work).as_posix()
                    self.upload_blob(key, path)
                    stale.discard(key)
            for key in sorted(stale):
                self.delete_blob(key)

    # ---- Derived reads — composed from the primitives above ----

    def sha256_blob(self, key: str) -> str:
        """Return the lowercase hex SHA-256 digest of the blob at ``key``.

        The digest of the blob's **exact stored bytes** — the same value as
        ``hashlib.sha256(self.get_blob(key)).hexdigest()``, computed without ever
        holding the whole blob in memory. This is what attestation leaves are
        built from, so it is part of DGML's on-chain contract: an override MUST
        return the plain SHA-256 of the full byte sequence and never a derived
        checksum (S3's multipart ETag and composite ``ChecksumSHA256`` are
        checksums-of-checksums and are **not** this value).

        Default: :meth:`materialize` plus the chunked
        :func:`dgml_core.hashing.sha256_file`. On ``LocalStore`` that is
        zero-copy — the key already *is* an on-disk path, so the real file is
        read in fixed-size chunks with no temp copy and no whole-blob
        allocation. On a remote store it is a managed (ranged, retryable)
        download to a temp file rather than one long-lived response body, which
        is what makes hashing a large artifact reliable there. Raises
        :class:`FileNotFoundError` if the blob is absent."""
        with self.materialize(key) as path:
            return sha256_file(path)


class DocStore(_StoreBase):
    """A pluggable **document** backend — JSON documents in named collections.

    Modeled on the MongoDB collection API. Documents carry no store-managed id in
    their body: a store keys them by ``(collection, doc_id)`` and never leaks its
    own ``_id`` into the returned dict.
    """

    # ---- JSON documents — modeled on the MongoDB collection API ----

    @abstractmethod
    def append_doc(self, collection: str, doc: dict[str, Any]) -> None:
        """Append ``doc`` to an **append-only** ``collection`` (the usage log).

        Such a document has no id: it is never fetched or replaced individually,
        only enumerated with :meth:`find_docs`. Which collections are append-only
        is the store's own business — ``LocalStore`` backs ``usage`` with
        ``usage.jsonl`` and rejects anything else.

        Deliberately *not* a Mongo-style ``insert_one``: an insert that fails on
        a duplicate id would be a create-if-absent primitive, and nothing in DGML
        needs one (creates go through :meth:`put_doc`, which is idempotent by
        design). Adding it later is easy; shipping a method whose documented
        semantics no implementation honours is not."""

    @abstractmethod
    def get_doc(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        """Return the document with ``_id == doc_id`` (Mongo ``find_one``), or None."""

    @abstractmethod
    def find_docs(self, collection: str, query: Mapping[str, Any]) -> list[dict[str, Any]]:
        """All documents in ``collection`` matching every field in ``query``
        (Mongo ``find``). An empty ``query`` returns the whole collection.

        **Ordering is unspecified.** A path-shaped store returns path order, a
        document database returns insertion order, and neither is wrong. A
        caller whose output is user-visible or order-sensitive must sort — see
        ``FileStore.list_all`` / ``DocSetStore.list_all``."""

    @abstractmethod
    def put_doc(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None:
        """Insert or replace the document with ``_id == doc_id`` (Mongo
        ``replace_one(upsert=True)``) — this is the update path."""

    @abstractmethod
    def delete_doc(self, collection: str, doc_id: str) -> None:
        """Delete the document with ``_id == doc_id``. Missing is a no-op."""

    @abstractmethod
    def delete_docs(self, collection: str, query: Mapping[str, Any]) -> int:
        """Delete every document in ``collection`` matching ``query`` (Mongo
        ``delete_many``). Returns the number deleted."""
