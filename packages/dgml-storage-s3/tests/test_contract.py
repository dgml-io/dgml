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

from .conftest import PROVIDER, make_store_options

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


def test_keys_with_separator_ids_round_trip(blobs: S3BlobStore) -> None:
    """Caller-supplied ids may contain `-` and `_` (`file add --id`), and an id
    is a key segment — prefix scoping must not confuse two similar ones."""
    blobs.put_blob("files/invoice_2024_q1/report.pdf", b"a")
    blobs.put_blob("files/invoice_2024_q1/page_images/page_1.png", b"b")
    blobs.put_blob("files/invoice_2024/report.pdf", b"c")
    assert blobs.get_blob("files/invoice_2024_q1/report.pdf") == b"a"
    assert blobs.list_blobs("files/invoice_2024_q1/") == [
        "files/invoice_2024_q1/page_images/page_1.png",
        "files/invoice_2024_q1/report.pdf",
    ]
    blobs.delete_blobs("files/invoice_2024_q1/")
    assert blobs.list_blobs("files/invoice_2024_q1/") == []
    assert blobs.get_blob("files/invoice_2024/report.pdf") == b"c"


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


def _store(options: dict[str, object], workspace_id: str | None, root: Path) -> S3BlobStore:
    config = StorageConfig(provider=PROVIDER, root=root, options=options, workspace_id=workspace_id)
    return S3BlobStore(S3BlobStore.parse_config(config))


def test_prefix_isolates_tenants_sharing_a_bucket(tmp_path: Path) -> None:
    base, options = make_store_options()
    a = _store({**options, "prefix": f"{base}/tenantA"}, "ws-test", tmp_path)
    b = _store({**options, "prefix": f"{base}/tenantB"}, "ws-test", tmp_path)
    a.put_blob("files/f/a.pdf", b"from-A")
    assert b.blob_exists("files/f/a.pdf") is False  # separate namespaces
    assert a.list_blobs("files/") == ["files/f/a.pdf"]  # prefix stripped on return


# ----------------------------------------------------- one namespace per workspace


def test_keys_go_under_the_prefix_then_the_workspace_id(tmp_path: Path) -> None:
    """The id is appended at runtime, not stamped into config: the config holds only
    what the user wrote, and the store works out the rest."""
    opts = {"bucket": "b"}
    assert _store(opts, "ws_a", tmp_path)._obj("files/x") == "dgml/ws_a/files/x"
    assert (
        _store({**opts, "prefix": "/contracts/"}, "ws_a", tmp_path)._obj("files/x")
        == "contracts/ws_a/files/x"
    )
    # An explicit empty prefix is the bucket root — but the id is still there.
    assert _store({**opts, "prefix": ""}, "ws_a", tmp_path)._obj("files/x") == "ws_a/files/x"


def test_a_workspace_without_an_id_is_refused(tmp_path: Path) -> None:
    """Keys without the id would be shared by every id-less workspace on the bucket,
    and would move the moment this one got an id."""
    from dgml_core.errors import StorageConfigInvalid

    with pytest.raises(StorageConfigInvalid, match="workspace's id"):
        _store({"bucket": "b"}, None, tmp_path)


def test_workspaces_with_one_config_share_a_bucket_without_colliding(tmp_path: Path) -> None:
    """The case this exists for: two workspaces bound to the same config write the same
    store keys, and must still see only their own objects."""
    _base, options = make_store_options()
    a, b = _store(options, "ws-a", tmp_path), _store(options, "ws-b", tmp_path)
    a.put_blob("files/f/x.pdf", b"from-a")
    b.put_blob("files/f/x.pdf", b"from-b")
    assert a.get_blob("files/f/x.pdf") == b"from-a"
    assert b.get_blob("files/f/x.pdf") == b"from-b"

    a.delete_blobs("files/")
    assert b.list_blobs("files/") == ["files/f/x.pdf"]
