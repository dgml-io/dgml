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

"""The bundled local-disk store — a :class:`BlobStore` **and** a :class:`DocStore`.

Implements both interfaces over one directory, so the zero-config default keeps
blobs and documents on local disk. Maps both APIs onto the **existing** workspace
directory layout (see
``docs/storage-layout.md``), so a local workspace on disk is byte-for-byte what
it is today — no migration, and everything that reads the tree directly
(``dgml check``, attestation, DGMLX bundles, external tooling) keeps working.

- **Blob keys are the on-disk relative paths** themselves: a blob is stored at
  ``<root>/<key>`` — ``files/<id>/page_images/page_1.png``, ``files/<id>/<name>``,
  ``files/<id>/page_text/page_1.json`` (bulky per-page word boxes, a blob like the
  page images despite the ``.json`` name), ``docsets/<did>/files/<fid>/<stem>.dgml.xml``,
  ``docsets/<did>/full-schema.rnc``.
- **JSON documents map by ``(collection, id)`` to their real manifest paths**:
  ``files`` → ``files/<id>/file.json``, ``docsets`` → ``docsets/<id>/docset.json``,
  ``assignments`` → ``docsets/<did>/files/<fid>/assignment.json``, and so on. The
  document is stored **verbatim** — no ``_id`` is injected, so ``file.json`` is
  exactly the ``FileRecord`` JSON it is today. One collection is special-cased:
  ``usage`` is the append-only ``usage.jsonl`` (one JSON object per line).

Every record is a *file*; no directory is load-bearing. Earlier revisions
recorded an assignment as the bare existence of ``docsets/<did>/files/<fid>/``,
which made a document's lifetime inseparable from its container's — deleting
the record meant deleting the directory (and the generated artifacts inside
it), and pruning an emptied directory could silently unassign a file.

Blobs and documents interleave in the same directories, so :meth:`list_blobs`
excludes the recognized document/reserved filenames. Every write is temp-file +
atomic rename.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .errors import CorruptMetadata, InvalidArgument
from .layout import (
    CACHE_DIR,
    DOC_LAYOUTS,
    DOCSET_FILES_DIR,
    DOCSETS_DIR,
    STAGING_DIR,
    USAGE_FILE,
    Collection,
    is_blob_key,
)
from .storage import read_json
from .storage_service import BlobStore, DocStore, StorageConfig

# The layout — collections, key shapes, document placement and blob
# classification — lives in ``layout.py``, shared with ``Workspace`` and the
# domain layer so a store key and a real filesystem path cannot drift apart.
# This module only maps those keys onto ``root/<key>``.


def _safe_segment(seg: str) -> str:
    """A single path segment, rejecting traversal / separators."""
    if not seg or "/" in seg or seg in (".", "..") or "\\" in seg:
        raise ValueError(f"invalid id/key segment {seg!r}")
    return seg


def _safe_rel(rel: str) -> Path:
    """A workspace-relative POSIX path, rejecting absolute paths and ``..``."""
    if not rel or rel.startswith("/"):
        raise ValueError(f"invalid storage key {rel!r}: must be a non-empty relative path")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"invalid storage key {rel!r}: '..' is not allowed")
    return Path(*parts)


def _split_id(doc_id: str, n: int) -> list[str]:
    """Split a composite document id (e.g. ``"<did>/<fid>"``) into ``n`` safe
    segments."""
    parts = doc_id.split("/")
    if len(parts) != n:
        raise ValueError(f"document id {doc_id!r} must have {n} '/'-separated parts")
    return [_safe_segment(p) for p in parts]


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _matches(doc: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    return all(doc.get(k) == v for k, v in query.items())


class LocalStore(BlobStore, DocStore):
    """Local-disk store over today's workspace layout. Takes no options; its
    location is the workspace root."""

    name = "local"
    config_fields = frozenset()

    @classmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        cls._check_no_extra_fields(config.options)
        return config

    def __init__(self, config: StorageConfig) -> None:
        self._root = Path(config.root)

    # ---- Blobs (S3-shaped): the key *is* the on-disk relative path ----

    def _blob_path(self, key: str) -> Path:
        return self._root / _safe_rel(key)

    @staticmethod
    def _check_writable_blob(key: str) -> None:
        """Reject a blob key that a document or bootstrap file already owns.

        On this store blob keys and document paths share one namespace, so
        ``put_blob("files/<id>/file.json", …)`` would overwrite that file's
        manifest — and because ``list_blobs`` excludes document names, the blob
        would then be invisible to every reader afterwards. A remote store keeps
        the two namespaces apart and needs no such check.

        No caller can currently reach this: every reserved basename ends in
        ``.json``/``.jsonl`` and no ingestable source suffix does
        (``.pdf``/``.doc``/``.docx``/``.xls``/``.xlsx``), so this is
        defence-in-depth rather than a fix for a live bug. It earns its place by
        making the *next* document filename or accepted extension fail loudly
        here instead of silently clobbering a manifest."""
        _safe_rel(key)  # shape first: traversal is a lower-level violation
        if not is_blob_key(key):
            raise InvalidArgument(
                f"{key!r} is not a writable blob key for the local store — it collides with "
                "a workspace document or reserved file. Rename the source file."
            )

    def put_blob(self, key: str, data: bytes) -> None:
        self._check_writable_blob(key)
        _write_bytes_atomic(self._blob_path(key), data)

    def get_blob(self, key: str) -> bytes:
        path = self._blob_path(key)
        if not path.is_file():
            raise FileNotFoundError(f"no blob at key {key!r}")
        return path.read_bytes()

    def delete_blob(self, key: str) -> None:
        self._blob_path(key).unlink(missing_ok=True)

    def blob_exists(self, key: str) -> bool:
        return self._blob_path(key).is_file()

    def list_blobs(self, prefix: str) -> list[str]:
        # Walk only what the prefix can reach, not the whole workspace. Not
        # cosmetic: `dgml check` calls this once per file, so scanning from the
        # root made it O(files x total blobs).
        root = self._root
        keys: list[str] = []
        for base in self._scan_bases(prefix):
            if base.is_file():
                if is_blob_key(rel := base.relative_to(root).as_posix()):
                    keys.append(rel)
                continue
            keys.extend(
                rel
                for path in base.rglob("*")
                if path.is_file() and is_blob_key(rel := path.relative_to(root).as_posix())
            )
        # Still string-filtered: a base may hold non-matching entries when the
        # prefix ends mid-segment.
        return sorted(k for k in keys if k.startswith(prefix))

    def _scan_bases(self, prefix: str) -> list[Path]:
        """Every place a key matching ``prefix`` can live.

        Keys match by **raw string prefix**, S3-style, which is not the same as
        "everything inside a directory". A prefix ending mid-segment also selects
        *siblings*: ``files/ab`` matches ``files/abc/…`` and ``files/abd/…``
        alike. Only a prefix ending in ``/`` is unambiguous — nothing outside
        that one directory can match it — which is the fast, single-base case
        every hot caller uses.

        Getting this wrong is silent: narrowing to the directory when the prefix
        merely *happens* to name one would quietly drop every sibling that
        extends it, returning a short list rather than an error."""
        rel = prefix.strip("/")
        if not rel:
            return [self._root]
        candidate = self._root / _safe_rel(rel)
        if prefix.endswith("/"):
            return [candidate] if candidate.is_dir() else []
        parent = candidate.parent
        if not parent.is_dir():
            return []
        return sorted(p for p in parent.iterdir() if p.name.startswith(candidate.name))

    def upload_blob(self, key: str, src: Path) -> None:
        self._check_writable_blob(key)
        dest = self._blob_path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dest)

    def download_blob(self, key: str, dest: Path) -> None:
        src = self._blob_path(key)
        if not src.is_file():
            raise FileNotFoundError(f"no blob at key {key!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    # ---- Path bridge — zero-copy: the key already IS an on-disk path ----

    @contextlib.contextmanager
    def materialize(self, key: str) -> Iterator[Path]:
        path = self._blob_path(key)
        if not path.is_file():
            raise FileNotFoundError(f"no blob at key {key!r}")
        yield path

    @contextlib.contextmanager
    def staged_write(self, key_prefix: str) -> Iterator[Path]:
        # Stage into the workspace's own scratch, then move the results into
        # place. Two reasons this isn't the older "hand back the destination and
        # let the tool write in place":
        #
        #   * the contract yields an *empty* directory and replaces the prefix,
        #     which writing in place cannot honour — a stale page_7.png from a
        #     longer previous render would survive unless the tool happened to
        #     delete it, and the store would disagree with a remote backend
        #     about what the file's page images are;
        #   * a crash mid-render must leave the prefix as it was, matching the
        #     base implementation, rather than half-clobbering it.
        #
        # Staging under the workspace root keeps it on one filesystem, so the
        # hand-off below is a rename — still no second copy of the bytes.
        prefix = key_prefix.rstrip("/")
        scratch = self._root / CACHE_DIR / STAGING_DIR
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            staging = Path(tmp)
            yield staging
            self.delete_blobs(prefix)
            dest = self._blob_path(prefix)
            dest.mkdir(parents=True, exist_ok=True)
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    target = dest / path.relative_to(staging)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(target)
        # The per-render temp dir is gone now; drop the shared scratch parent too so
        # a workspace isn't left with an empty ``.cache/staging/``. ``rmdir`` removes
        # only an *empty* directory (atomic on every OS) and raises ``OSError``
        # otherwise — ``ENOTEMPTY`` on POSIX, the equivalent on Windows — which we
        # ignore, so a concurrent ``staged_write`` still holding a temp dir here is
        # safe.
        with contextlib.suppress(OSError):
            scratch.rmdir()

    @contextlib.contextmanager
    def materialize_dir(self, prefix: str) -> Iterator[Path]:
        # The blobs already live under this directory — hand it back directly.
        yield self._blob_path(prefix.rstrip("/"))

    @contextlib.contextmanager
    def working_dir(self, prefix: str) -> Iterator[Path]:
        # The real directory IS under the store root — writes and deletes land
        # in place, so there is nothing to download in or upload back out, and
        # the replace-on-exit contract holds for free. (Unlike staged_write this
        # stays in-place: the caller is a regenerable cache, so persisting a
        # partial one on a crash is harmless, and staging it would mean copying
        # the whole cache in and out on every run.)
        work = self._blob_path(prefix.rstrip("/"))
        work.mkdir(parents=True, exist_ok=True)
        yield work

    def delete_blobs(self, prefix: str) -> None:
        # Remove only blob files under the prefix (documents that live beside them —
        # file.json, extraction_stats.json, … — are left for delete_doc), then prune
        # any directories emptied as a result so the tree matches a recursive remove.
        base = self._root / _safe_rel(prefix)
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file() and is_blob_key(path.relative_to(self._root).as_posix()):
                    path.unlink()
        elif base.is_file() and is_blob_key(base.relative_to(self._root).as_posix()):
            base.unlink()
        self._prune_empty_dirs(base)

    def _prune_empty_dirs(self, base: Path) -> None:
        """Remove empty directories in and above ``base`` (bottom-up), stopping at
        the workspace root's top-level directories (``files/``, ``docsets/``) and the
        root itself — so composed blob+document deletes leave no lingering empty dirs
        (matching the historical recursive remove).

        No directory is load-bearing: every record is a document, so pruning an
        empty directory can never destroy one. (It could when an assignment *was*
        an empty ``docsets/<did>/files/<fid>/``, which is why this used to need a
        guard against removing them.)"""
        if base.is_dir():
            subdirs = sorted(
                (p for p in base.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts),
                reverse=True,
            )
            for sub in subdirs:
                with contextlib.suppress(OSError):
                    sub.rmdir()
        directory = base
        while directory != self._root and directory.parent != self._root:
            try:
                parent = directory.parent
                directory.rmdir()
            except OSError:
                break  # not empty (or already gone) → stop
            directory = parent

    # ---- JSON documents (Mongo-shaped): mapped to today's manifest paths ----

    def _assignment_dir(self, doc_id: str) -> Path:
        """The per-(docset, file) directory for an assignment id ``did/fid``."""
        did, fid = _split_id(doc_id, 2)
        return self._root / DOCSETS_DIR / did / DOCSET_FILES_DIR / fid

    def _doc_path(self, collection: str, doc_id: str) -> Path:
        layout = DOC_LAYOUTS.get(collection)
        if layout is None:
            # Unknown collection: a generic per-id file, kept out of the blob
            # namespace by its ``.json`` extension under a same-named directory.
            return self._root / _safe_segment(collection) / f"{_safe_segment(doc_id)}.json"
        if not layout.id_parts:
            rel = layout.template
        else:
            segments = _split_id(doc_id, len(layout.id_parts))
            rel = layout.template.format(**dict(zip(layout.id_parts, segments, strict=True)))
        return self._root / _safe_rel(rel)

    def append_doc(self, collection: str, doc: dict[str, Any]) -> None:
        # ``usage`` is the workspace's only append-only collection; everything
        # else is addressed by id and belongs in put_doc.
        if collection != Collection.USAGE:
            raise InvalidArgument(
                f"{collection!r} is not an append-only collection; use put_doc "
                f"(append-only: {Collection.USAGE.value!r})"
            )
        line = json.dumps(doc, separators=(",", ":"), ensure_ascii=False)
        path = self._root / USAGE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def get_doc(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        if collection == Collection.USAGE:
            return None  # append-only; read via find_docs, not by id
        path = self._doc_path(collection, doc_id)
        if not path.is_file():
            return None
        return self._read_doc(path)

    def find_docs(self, collection: str, query: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [doc for doc in self._iter_docs(collection) if _matches(doc, query)]

    def put_doc(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None:
        # Stored verbatim — the manifest keeps its own fields (e.g. ``id``); no
        # ``_id`` is injected, so ``file.json`` is byte-identical to today.
        _write_text_atomic(
            self._doc_path(collection, doc_id),
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        )

    def delete_doc(self, collection: str, doc_id: str) -> None:
        if collection == Collection.USAGE:
            return
        self._doc_path(collection, doc_id).unlink(missing_ok=True)
        if collection == Collection.ASSIGNMENTS:
            # Deleting the assignment removes the record and nothing else — the
            # pair's generated dgml.xml / extraction_stats are separate objects
            # that a cascade deletes explicitly (see ``Workspace.unassign``).
            # Drop the pair directory if that left it empty, so the tree matches
            # what a recursive remove would leave.
            with contextlib.suppress(OSError):
                self._assignment_dir(doc_id).rmdir()

    def delete_docs(self, collection: str, query: Mapping[str, Any]) -> int:
        if collection == Collection.USAGE:
            path = self._root / USAGE_FILE
            if not path.is_file():
                return 0
            docs = list(self._iter_docs(collection))
            kept = [doc for doc in docs if not _matches(doc, query)]
            _write_text_atomic(
                path, "".join(json.dumps(d, separators=(",", ":")) + "\n" for d in kept)
            )
            return len(docs) - len(kept)
        if collection == Collection.ASSIGNMENTS:
            matched = self.find_docs(collection, query)
            for doc in matched:
                self.delete_doc(collection, f"{doc['docset_id']}/{doc['file_id']}")
            return len(matched)
        removed = 0
        for path in self._doc_paths(collection):
            doc = self._read_doc(path)
            if _matches(doc, query):
                path.unlink(missing_ok=True)
                removed += 1
        return removed

    # ---- document enumeration ----

    def _doc_paths(self, collection: str) -> list[Path]:
        layout = DOC_LAYOUTS.get(collection)
        pattern = layout.glob if layout is not None else f"{collection}/*.json"
        return sorted(p for p in self._root.glob(pattern) if p.is_file())

    def _iter_docs(self, collection: str) -> Iterator[dict[str, Any]]:
        if collection == Collection.USAGE:
            path = self._root / USAGE_FILE
            if not path.is_file():
                return
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a corrupt tail line from a crashed append
                if isinstance(obj, dict):
                    yield obj
            return
        for path in self._doc_paths(collection):
            try:
                yield self._read_doc(path)
            except CorruptMetadata:
                # A corrupt manifest reads as absent for enumeration (matches the
                # historical list_all behavior of skipping unparseable docsets).
                continue

    @staticmethod
    def _read_doc(path: Path) -> dict[str, Any]:
        # ``read_json`` gives duplicate-key rejection and raises CorruptMetadata on
        # bad JSON — the same contract the manifest readers relied on.
        obj = read_json(path)
        if not isinstance(obj, dict):
            raise ValueError(f"document {path} is not a JSON object")
        return obj
