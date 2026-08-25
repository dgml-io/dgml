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

"""MongoGridFSBlobStore obeys the BlobStore contract (mongomock, or real Mongo in CI)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from dgml_core.errors import DgmlError, StorageConfigInvalid
from dgml_core.storage_service import StorageConfig
from dgml_storage_mongo import CHUNK_BYTES, MongoGridFSBlobStore

from .conftest import GRIDFS_PROVIDER, chunk_collection

# ------------------------------------------------------------------ config


def test_requires_a_database(tmp_path: Path) -> None:
    with pytest.raises(StorageConfigInvalid):
        MongoGridFSBlobStore.parse_config(StorageConfig(provider=GRIDFS_PROVIDER, root=tmp_path))


def test_rejects_unknown_and_credential_fields(tmp_path: Path) -> None:
    for bad in ({"mongo_database": "d", "typo": 1}, {"mongo_database": "d", "mongo_password": "x"}):
        with pytest.raises(StorageConfigInvalid):
            MongoGridFSBlobStore.parse_config(
                StorageConfig(provider=GRIDFS_PROVIDER, root=tmp_path, options=bad)
            )


def test_bad_port_rejected(tmp_path: Path) -> None:
    # `True` is an int subclass — a bool port is a typo, not a port.
    for port in ("x", True):
        with pytest.raises(StorageConfigInvalid):
            MongoGridFSBlobStore.parse_config(
                StorageConfig(
                    provider=GRIDFS_PROVIDER,
                    root=tmp_path,
                    options={"mongo_database": "d", "mongo_port": port},
                )
            )


def test_bucket_option_accepted_and_validated(tmp_path: Path) -> None:
    """``mongo_bucket`` selects the GridFS bucket so several workspaces can share
    a database. Blank is rejected rather than silently falling back."""
    config = StorageConfig(
        provider=GRIDFS_PROVIDER,
        root=tmp_path,
        options={"mongo_database": "d", "mongo_bucket": "custom"},
    )
    assert MongoGridFSBlobStore.parse_config(config) is config
    with pytest.raises(StorageConfigInvalid):
        MongoGridFSBlobStore.parse_config(
            StorageConfig(
                provider=GRIDFS_PROVIDER,
                root=tmp_path,
                options={"mongo_database": "d", "mongo_bucket": "  "},
            )
        )


# ------------------------------------------------------------------- blobs


def test_round_trip_and_missing_key(blobs: MongoGridFSBlobStore) -> None:
    key = "files/f1/report.pdf"
    assert blobs.blob_exists(key) is False
    with pytest.raises(FileNotFoundError):
        blobs.get_blob(key)
    blobs.put_blob(key, b"pdf-bytes")
    assert blobs.get_blob(key) == b"pdf-bytes"
    assert blobs.blob_exists(key) is True


def test_put_overwrites_and_leaves_no_orphan_chunks(blobs: MongoGridFSBlobStore) -> None:
    """GridFS versions by filename; this store replaces. A rewritten blob must
    not accumulate revisions — page images are rewritten on every re-render."""
    key = "files/f1/page_images/page_1.png"
    for payload in (b"first", b"second-and-longer", b"third"):
        blobs.put_blob(key, payload)
    assert blobs.get_blob(key) == b"third"
    assert blobs.list_blobs("files/f1/") == [key]  # one key, not three revisions
    # And one live chunk, not three: GridFS versions by filename, so this is the
    # assertion that catches a store which uploads without collecting the old
    # revision — unbounded growth, one revision per re-render.
    assert chunk_collection(blobs).count_documents({}) == 1


def test_delete_is_idempotent_and_collects_chunks(blobs: MongoGridFSBlobStore) -> None:
    key = "files/f1/report.pdf"
    blobs.put_blob(key, b"x" * (CHUNK_BYTES + 1))  # two chunks
    assert chunk_collection(blobs).count_documents({}) == 2
    blobs.delete_blob(key)
    blobs.delete_blob(key)  # missing key is a no-op
    assert blobs.blob_exists(key) is False
    assert chunk_collection(blobs).count_documents({}) == 0


def test_empty_blob_is_stored_not_missing(blobs: MongoGridFSBlobStore) -> None:
    blobs.put_blob("files/f1/empty.txt", b"")
    assert blobs.blob_exists("files/f1/empty.txt") is True
    assert blobs.get_blob("files/f1/empty.txt") == b""
    assert blobs.sha256_blob("files/f1/empty.txt") == hashlib.sha256(b"").hexdigest()


def test_list_and_delete_respect_the_trailing_slash(blobs: MongoGridFSBlobStore) -> None:
    """The prefix contract: ``files/ab/`` must not select ``files/abc/``."""
    blobs.put_blob("files/ab/x.png", b"a")
    blobs.put_blob("files/abc/y.png", b"b")
    assert blobs.list_blobs("files/ab/") == ["files/ab/x.png"]
    blobs.delete_blobs("files/ab/")
    assert blobs.list_blobs("files/") == ["files/abc/y.png"]
    blobs.delete_blobs("files/nothing/")  # matches nothing → no-op


def test_list_blobs_is_sorted(blobs: MongoGridFSBlobStore) -> None:
    for name in ("c.png", "a.png", "b.png"):
        blobs.put_blob(f"files/f1/page_images/{name}", b"x")
    assert blobs.list_blobs("files/f1/page_images/") == [
        "files/f1/page_images/a.png",
        "files/f1/page_images/b.png",
        "files/f1/page_images/c.png",
    ]


def test_prefix_metacharacters_are_not_regex(blobs: MongoGridFSBlobStore) -> None:
    """Keys hold filenames. An unescaped ``.`` would match any character."""
    blobs.put_blob("files/f1/page_1.png", b"real")
    blobs.put_blob("files/f1/page_1Xpng", b"decoy")
    assert blobs.list_blobs("files/f1/page_1.png") == ["files/f1/page_1.png"]


def test_multi_chunk_round_trip_and_digest(blobs: MongoGridFSBlobStore) -> None:
    data = bytes(range(256)) * 20_000  # ~5 MB, several chunks
    blobs.put_blob("files/f1/big.bin", data)
    assert blobs.get_blob("files/f1/big.bin") == data
    assert blobs.sha256_blob("files/f1/big.bin") == hashlib.sha256(data).hexdigest()


def test_blob_larger_than_the_bson_document_cap(blobs: MongoGridFSBlobStore) -> None:
    """The reason for chunking at all: MongoDB caps a BSON document at 16 MB,
    and a source PDF has no size bound."""
    data = b"z" * (20 * 1024 * 1024)
    blobs.put_blob("files/f1/huge.pdf", data)
    assert blobs.sha256_blob("files/f1/huge.pdf") == hashlib.sha256(data).hexdigest()
    assert len(blobs.get_blob("files/f1/huge.pdf")) == len(data)


def test_sha256_is_the_plain_digest_of_stored_bytes(blobs: MongoGridFSBlobStore) -> None:
    """It is served from the manifest rather than a re-read, so it has to be
    proven equal to the digest of what ``get_blob`` returns — this value is
    attestation's leaf input."""
    data = b"\x00\xff" * 700_000
    blobs.put_blob("files/f1/a.bin", data)
    assert (
        blobs.sha256_blob("files/f1/a.bin")
        == hashlib.sha256(blobs.get_blob("files/f1/a.bin")).hexdigest()
    )
    with pytest.raises(FileNotFoundError):
        blobs.sha256_blob("files/f1/gone.bin")


def test_a_torn_read_is_detected_not_returned_short(blobs: MongoGridFSBlobStore) -> None:
    """The manifest records ``chunks`` and ``length`` so an incomplete
    generation is distinguishable from a short blob. Simulated by collecting the
    chunks a concurrent overwrite would have collected — a real reader hits this
    only if it resolved the manifest just before another writer flipped it, and
    then retries; a *persistent* inconsistency raises instead of silently
    handing back truncated bytes.
    """
    key = "files/f1/report.pdf"
    blobs.put_blob(key, b"x" * (CHUNK_BYTES * 2))
    chunk_collection(blobs).delete_one({"n": 1})  # lose the tail

    with pytest.raises(DgmlError, match="kept changing while being read"):
        blobs.get_blob(key)


def test_upload_and_download_blob(blobs: MongoGridFSBlobStore, tmp_path: Path) -> None:
    src = tmp_path / "in.bin"
    src.write_bytes(b"filebytes" * 200_000)
    blobs.upload_blob("files/f1/a.bin", src)
    assert blobs.sha256_blob("files/f1/a.bin") == hashlib.sha256(src.read_bytes()).hexdigest()

    dest = tmp_path / "nested" / "out" / "a.bin"  # parents created for us
    blobs.download_blob("files/f1/a.bin", dest)
    assert dest.read_bytes() == src.read_bytes()


def test_download_missing_key_raises_and_leaves_no_file(
    blobs: MongoGridFSBlobStore, tmp_path: Path
) -> None:
    dest = tmp_path / "out.bin"
    with pytest.raises(FileNotFoundError):
        blobs.download_blob("files/f1/missing.bin", dest)
    assert not dest.exists()


def test_upload_overwrites_an_existing_blob(blobs: MongoGridFSBlobStore, tmp_path: Path) -> None:
    blobs.put_blob("files/f1/a.bin", b"old")
    src = tmp_path / "in.bin"
    src.write_bytes(b"new")
    blobs.upload_blob("files/f1/a.bin", src)
    assert blobs.get_blob("files/f1/a.bin") == b"new"


# ------------------------------------------------------------- path bridge
#
# These are inherited from BlobStore, not overridden. Tested anyway: they are
# what the pipeline actually calls (ghostscript, pdfminer, the generation cache),
# and they are only correct if this store's primitives are.


def test_materialize_yields_a_real_path(blobs: MongoGridFSBlobStore) -> None:
    blobs.put_blob("files/f1/src.pdf", b"pdf-bytes")
    with blobs.materialize("files/f1/src.pdf") as path:
        assert Path(path).read_bytes() == b"pdf-bytes"


def test_staged_write_replaces_the_whole_prefix(blobs: MongoGridFSBlobStore) -> None:
    """A re-render whose page count dropped must not leave stale pages behind —
    they would otherwise be hashed into the file's attestation."""
    with blobs.staged_write("files/f1/page_images") as staging:
        for n in (1, 2, 3):
            (Path(staging) / f"page_{n}.png").write_bytes(f"img{n}".encode())
    assert len(blobs.list_blobs("files/f1/page_images/")) == 3

    with blobs.staged_write("files/f1/page_images") as staging:
        for n in (1, 2):
            (Path(staging) / f"page_{n}.png").write_bytes(f"new{n}".encode())
    assert blobs.list_blobs("files/f1/page_images/") == [
        "files/f1/page_images/page_1.png",
        "files/f1/page_images/page_2.png",
    ]
    assert blobs.get_blob("files/f1/page_images/page_1.png") == b"new1"


def test_staged_write_persists_nothing_on_error(blobs: MongoGridFSBlobStore) -> None:
    blobs.put_blob("files/f1/page_images/page_1.png", b"orig")
    with pytest.raises(RuntimeError):
        with blobs.staged_write("files/f1/page_images") as staging:
            (Path(staging) / "page_1.png").write_bytes(b"half-written")
            raise RuntimeError("renderer died")
    assert blobs.get_blob("files/f1/page_images/page_1.png") == b"orig"


def test_materialize_dir_and_working_dir(blobs: MongoGridFSBlobStore) -> None:
    for n in (1, 2):
        blobs.put_blob(f"files/f1/page_images/page_{n}.png", f"img{n}".encode())
    with blobs.materialize_dir("files/f1/page_images/") as directory:
        assert sorted(p.name for p in Path(directory).iterdir()) == ["page_1.png", "page_2.png"]

    # working_dir syncs both ways: writes land, local deletes propagate.
    with blobs.working_dir("docsets/d1/cache") as work:
        (Path(work) / "state.json").write_text("{}")
    assert blobs.list_blobs("docsets/d1/cache/") == ["docsets/d1/cache/state.json"]
    with blobs.working_dir("docsets/d1/cache") as work:
        (Path(work) / "state.json").unlink()
    assert blobs.list_blobs("docsets/d1/cache/") == []
