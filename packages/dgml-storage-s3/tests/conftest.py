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

"""Fixtures for the sample S3 **blob** store.

Two modes, same tests:

- **Real S3** when ``DGML_TEST_S3_ENDPOINT`` is set (the ``docker compose`` stack,
  or anything S3-compatible) — this is what exercises wire behaviour (genuine
  pagination, real error codes).
- **In-process ``moto``** otherwise.

The fallback matters: a suite that skips when Docker is not running looks like
coverage without being any. The default ``uv run pytest`` always runs the whole
thing; CI additionally runs it against SeaweedFS.

Every test gets its own **key prefix** inside one shared bucket, so runs never
share state. The document half of a workspace uses the bundled local store here,
so these tests need no Mongo.

Prefix rather than bucket, deliberately: each S3 bucket becomes a SeaweedFS
*collection*, and each new collection triggers a burst of volume growth (7
volumes by default). A bucket per test multiplied that by the test count and
exhausted the volume budget on CI's smaller disk, where `PutObject` then fails
with `InternalError`. Isolating by prefix is also what ``prefix`` is *for* — it
is how several workspaces share one bucket in a real deployment — so the fixture
now matches how the store is actually used.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from dgml_core.storage import Workspace
from dgml_core.storage_service import StorageConfig
from dgml_storage_s3 import S3BlobStore

S3_ENDPOINT_ENV = "DGML_TEST_S3_ENDPOINT"

#: Whether the suite is pointed at a real S3 endpoint rather than moto.
USING_REAL_S3 = bool(os.environ.get(S3_ENDPOINT_ENV))

PROVIDER = "dgml_storage_s3:S3BlobStore"


@pytest.fixture(autouse=True)
def _isolate_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Sandbox the user-level config + per-machine registry into a temp dir, so a
    test never reads or writes the developer's real ``~/.config/dgml``."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    monkeypatch.delenv("DGML_HOME", raising=False)


@pytest.fixture(autouse=True)
def _fake_s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stand up in-process S3 (``moto``) unless a real endpoint is configured."""
    if USING_REAL_S3:
        yield
        return

    from moto import mock_aws

    # boto3 refuses to sign requests without credentials; moto ignores the values.
    for var, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(var, value)
    with mock_aws():
        yield


#: One bucket for the whole pytest session. Unique per run so concurrent CI jobs
#: against a shared endpoint never collide.
SESSION_BUCKET = f"dgml-test-{uuid.uuid4().hex[:12]}"


def make_store_options() -> tuple[str, dict[str, object]]:
    """Return ``(prefix, S3 options)`` for a test, in the shared session bucket.

    Creation is idempotent rather than once-per-session on purpose: under
    ``moto`` the mock is torn down between tests, so the bucket has to be
    re-created each time; against a real endpoint only the first call creates it
    and the rest get ``BucketAlreadyOwnedByYou``, which is not an error here.
    """
    import boto3
    from botocore.exceptions import ClientError

    prefix = f"t{uuid.uuid4().hex[:12]}"
    options: dict[str, object] = {
        "bucket": SESSION_BUCKET,
        "region": "us-east-1",
        "prefix": prefix,
    }
    endpoint = os.environ.get(S3_ENDPOINT_ENV)
    client_kwargs = {"region_name": "us-east-1"}
    if endpoint:
        options["endpoint_url"] = endpoint
        client_kwargs["endpoint_url"] = endpoint
    try:
        boto3.client("s3", **client_kwargs).create_bucket(Bucket=SESSION_BUCKET)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {
            "BucketAlreadyOwnedByYou",
            "BucketAlreadyExists",
        }:
            raise
    return prefix, options


@pytest.fixture
def s3_config(tmp_path: Path) -> StorageConfig:
    """A per-test prefix in the shared bucket, as a resolved blob-store config."""
    _prefix, options = make_store_options()
    return StorageConfig(provider=PROVIDER, root=tmp_path / "ws", options=options)


@pytest.fixture
def blobs(s3_config: StorageConfig) -> S3BlobStore:
    return S3BlobStore(S3BlobStore.parse_config(s3_config))


@pytest.fixture
def s3_blobs_workspace(tmp_path: Path) -> Workspace:
    """A workspace whose **blobs** live on S3 and **docs** on local disk, so the
    whole pipeline runs with S3 blobs without any of it knowing.

    The S3 backend is the ``default`` service's blob role, so an unregistered
    ``Workspace`` resolves it with no registry entry needed."""
    _prefix, options = make_store_options()
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    lines = [f'{k} = "{v}"' for k, v in options.items()]
    (root / "config.toml").write_text(
        f'[storage.default.blobs]\nprovider = "{PROVIDER}"\n' + "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    ws = Workspace(root=root)
    return ws
