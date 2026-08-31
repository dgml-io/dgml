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

"""A sample :class:`~dgml_core.storage_service.BlobStore`: blobs in S3.

Blobs live in an S3-compatible bucket — the object API ``BlobStore`` was modelled
on, so nearly every method is a one-line delegation. That is the point: a
reference to copy, not a tuned production store. Pair it with a
:class:`~dgml_core.storage_service.DocStore` (e.g. ``dgml-storage-mongo``, or the
bundled local store) for the document half.

**Sample, not supported.** Resolved by dotted path like any third party's own::

    [storage.acme.blobs]
    provider = "dgml_storage_s3:S3BlobStore"
    bucket = "dgml-dev"
    endpoint_url = "http://localhost:9000"   # MinIO; omit for real AWS S3

    [storage.acme.docs]
    provider = "dgml_storage_mongo:MongoDocStore"
    mongo_database = "dgml_dev"

MinIO is not a separate backend — it speaks the S3 API, so the same class runs
against it locally and against AWS in production by changing ``endpoint_url``.

Credentials
-----------

**Credentials are read from the environment, never from DGML config.** S3 uses
boto3's default chain (``AWS_ACCESS_KEY_ID`` / ``~/.aws/credentials`` / an IAM
role). Config carries *identity only* — bucket, endpoint, prefix — which is also
exactly what should define the store fingerprint.

This is not merely stylistic. ``dgml_core.storage_resolve`` excludes secret-hinted
options from a store's identity fingerprint by a **substring match on the option
name** against ``("key", "secret", "token", "password", "credential")``. If you ever
add an inline-credential option, its name must contain one of those substrings — both
so rotating it does not read as "the store moved", and because this config now lives
in the workspace's own plaintext ``config.toml``, which sits beside the workspace and
is likely to be committed or synced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dgml_core.errors import DgmlError, StorageConfigInvalid
from dgml_core.storage_service import BlobStore, StorageConfig

#: S3 caps a single ``delete_objects`` call at 1000 keys.
_DELETE_BATCH = 1000


class S3BlobStore(BlobStore):
    """Blobs in an S3-compatible bucket. Inherits the path bridge and
    ``sha256_blob`` from :class:`~dgml_core.storage_service.BlobStore`."""

    name = "s3"
    config_fields = frozenset({"bucket", "region", "endpoint_url", "prefix"})

    # ---- configuration ----

    @classmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        cls._check_no_extra_fields(config.options)
        bucket = config.options.get("bucket")
        if not isinstance(bucket, str) or not bucket.strip():
            raise StorageConfigInvalid(f"[storage] provider {cls.name!r} requires a 'bucket'")
        prefix = config.options.get("prefix")
        if prefix is not None and not isinstance(prefix, str):
            raise StorageConfigInvalid("'prefix' must be a string")
        return config

    def __init__(self, config: StorageConfig) -> None:
        # Lazy SDK import with an actionable message, per the ABC's contract:
        # a workspace that never opens this store must not need boto3.
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise DgmlError("the s3 sample store needs boto3: pip install dgml-storage-s3") from exc

        opts = config.options
        self._bucket = str(opts["bucket"])
        # An optional key prefix lets several workspaces share one bucket. Kept
        # normalized to "" or "…/" so key joins never double or drop a slash.
        raw_prefix = str(opts.get("prefix") or "").strip("/")
        self._prefix = f"{raw_prefix}/" if raw_prefix else ""

        client_kwargs: dict[str, Any] = {}
        if opts.get("region"):
            client_kwargs["region_name"] = str(opts["region"])
        if opts.get("endpoint_url"):
            client_kwargs["endpoint_url"] = str(opts["endpoint_url"])
        # Credentials come from boto3's default chain — env, shared config, or
        # an instance role. Never from DGML config.
        # Untyped: boto3 ships no stubs, and adding boto3-stubs for a sample
        # would put a large dev dependency in the tree for little gain.
        self._s3: Any = boto3.client("s3", **client_kwargs)

    # ---- key mapping ----

    def _obj(self, key: str) -> str:
        """The S3 object key for a store key (adds the configured prefix)."""
        return f"{self._prefix}{key}"

    def _key(self, obj: str) -> str:
        """The store key for an S3 object key (strips the configured prefix)."""
        return obj[len(self._prefix) :] if self._prefix else obj

    # ---- Blobs (S3) ----

    def put_blob(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self._bucket, Key=self._obj(key), Body=data)

    def get_blob(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self._s3.get_object(Bucket=self._bucket, Key=self._obj(key))
        except ClientError as exc:
            if _is_missing(exc):
                raise FileNotFoundError(f"no blob at key {key!r}") from exc
            raise
        body: bytes = response["Body"].read()
        return body

    def delete_blob(self, key: str) -> None:
        # S3 delete is already idempotent — a missing key is not an error.
        self._s3.delete_object(Bucket=self._bucket, Key=self._obj(key))

    def blob_exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=self._bucket, Key=self._obj(key))
        except ClientError as exc:
            if _is_missing(exc):
                return False
            raise
        return True

    def list_blobs(self, prefix: str) -> list[str]:
        # MUST paginate. list_objects_v2 returns at most 1000 keys per response,
        # and the contract here is *every* key under the prefix — `_entity_ids`
        # lists a whole workspace, which passes 1000 blobs at ~60 files. Ignoring
        # NextContinuationToken would silently return a short list rather than
        # fail, which is the worst failure mode there is.
        paginator = self._s3.get_paginator("list_objects_v2")
        keys = [
            self._key(obj["Key"])
            for page in paginator.paginate(Bucket=self._bucket, Prefix=self._obj(prefix))
            for obj in page.get("Contents", [])
        ]
        return sorted(keys)

    def upload_blob(self, key: str, src: Path) -> None:
        self._s3.upload_file(str(src), self._bucket, self._obj(key))

    def download_blob(self, key: str, dest: Path) -> None:
        from botocore.exceptions import ClientError

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._s3.download_file(self._bucket, self._obj(key), str(dest))
        except ClientError as exc:
            if _is_missing(exc):
                raise FileNotFoundError(f"no blob at key {key!r}") from exc
            raise

    def delete_blobs(self, prefix: str) -> None:
        # Cascades call this last (see WorkspaceOps): the authoritative record
        # dies first, so an interrupted cascade leaves recoverable orphaned bytes.
        keys = self.list_blobs(prefix)
        for start in range(0, len(keys), _DELETE_BATCH):
            batch = keys[start : start + _DELETE_BATCH]
            self._s3.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": [{"Key": self._obj(k)} for k in batch], "Quiet": True},
            )


def _is_missing(exc: Any) -> bool:
    """Whether a botocore ``ClientError`` means "no such key/bucket".

    ``head_object`` reports a missing key as ``404``/``NoSuchKey`` depending on
    the operation and the server, and MinIO and AWS do not agree on every code,
    so match on both.
    """
    error = getattr(exc, "response", {}).get("Error", {})
    return str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}
