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

"""S3BlobStore obeys the BlobStore contract (against moto, or real S3 in CI)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from dgml_core.storage_service import StorageConfig
from dgml_storage_s3 import S3BlobStore

from .conftest import PROVIDER, make_bucket

# ------------------------------------------------------------------ config


def test_requires_a_bucket(tmp_path: Path) -> None:
    from dgml_core.errors import StorageConfigInvalid

    with pytest.raises(StorageConfigInvalid):
        S3BlobStore.parse_config(StorageConfig(provider=PROVIDER, root=tmp_path))


def test_rejects_unknown_and_credential_fields(tmp_path: Path) -> None:
    from dgml_core.errors import StorageConfigInvalid

    for bad in ({"bucket": "b", "typo": 1}, {"bucket": "b", "secret_key": "x"}):
        with pytest.raises(StorageConfigInvalid):
            S3BlobStore.parse_config(StorageConfig(provider=PROVIDER, root=tmp_path, options=bad))


# ------------------------------------------------------------------ blobs


def test_blob_round_trip_and_missing(blobs: S3BlobStore) -> None:
    assert blobs.blob_exists("files/f/a.pdf") is False
    with pytest.raises(FileNotFoundError):
        blobs.get_blob("files/f/a.pdf")
    blobs.put_blob("files/f/a.pdf", b"pdf-bytes")
    assert blobs.blob_exists("files/f/a.pdf") is True
    assert blobs.get_blob("files/f/a.pdf") == b"pdf-bytes"
    blobs.put_blob("files/f/a.pdf", b"replaced")  # overwrite = update
    assert blobs.get_blob("files/f/a.pdf") == b"replaced"


def test_delete_is_idempotent(blobs: S3BlobStore) -> None:
    blobs.put_blob("files/f/a.pdf", b"1")
    blobs.delete_blob("files/f/a.pdf")
    blobs.delete_blob("files/f/a.pdf")  # missing key is a no-op
    assert not blobs.blob_exists("files/f/a.pdf")


def test_upload_download(blobs: S3BlobStore, tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    blobs.upload_blob("files/d/e.bin", src)
    assert blobs.get_blob("files/d/e.bin") == b"payload"
    dest = tmp_path / "out" / "dl.bin"  # parents created
    blobs.download_blob("files/d/e.bin", dest)
    assert dest.read_bytes() == b"payload"
    with pytest.raises(FileNotFoundError):
        blobs.download_blob("files/d/gone.bin", tmp_path / "x.bin")


def test_list_is_sorted_and_prefix_scoped(blobs: S3BlobStore) -> None:
    blobs.put_blob("files/f1/report.pdf", b"a")
    blobs.put_blob("files/f1/page_images/page_1.png", b"b")
    blobs.put_blob("files/f2/report.pdf", b"c")
    assert blobs.list_blobs("files/f1/") == [
        "files/f1/page_images/page_1.png",
        "files/f1/report.pdf",
    ]


def test_list_and_delete_paginate_past_1000(blobs: S3BlobStore) -> None:
    # The single most important wire behaviour the in-process fake still models:
    # list_objects_v2 caps at 1000 keys, so a naive one-page read silently drops.
    for n in range(1050):
        blobs.put_blob(f"files/big/{n:04d}.bin", b"x")
    assert len(blobs.list_blobs("files/big/")) == 1050
    blobs.delete_blobs("files/big/")  # batches past the 1000-key delete cap
    assert blobs.list_blobs("files/big/") == []


def test_delete_blobs_is_prefix_scoped(blobs: S3BlobStore) -> None:
    blobs.put_blob("files/f1/a.png", b"a")
    blobs.put_blob("files/f2/a.png", b"b")
    blobs.delete_blobs("files/f1/")
    assert blobs.list_blobs("files/f1/") == []
    assert blobs.get_blob("files/f2/a.png") == b"b"  # sibling untouched


def test_sha256_blob_is_plain_digest_not_etag(blobs: S3BlobStore) -> None:
    # Attestation leaves are built from this, so it must be the plain SHA-256 of
    # the exact stored bytes — never S3's multipart ETag (a checksum-of-checksums).
    data = b"attest-me" * 100
    blobs.put_blob("files/f/x.bin", data)
    assert blobs.sha256_blob("files/f/x.bin") == hashlib.sha256(data).hexdigest()


def test_prefix_isolates_tenants_sharing_a_bucket(tmp_path: Path) -> None:
    _bucket, options = make_bucket()
    a = S3BlobStore(
        S3BlobStore.parse_config(
            StorageConfig(provider=PROVIDER, root=tmp_path, options={**options, "prefix": "wsA"})
        )
    )
    b = S3BlobStore(
        S3BlobStore.parse_config(
            StorageConfig(provider=PROVIDER, root=tmp_path, options={**options, "prefix": "wsB"})
        )
    )
    a.put_blob("files/f/a.pdf", b"from-A")
    assert b.blob_exists("files/f/a.pdf") is False  # separate namespaces
    assert a.list_blobs("files/") == ["files/f/a.pdf"]  # prefix stripped on return
