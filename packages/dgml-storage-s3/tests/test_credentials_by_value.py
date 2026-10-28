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

"""`aws_*` credential options by value reach the boto3 client; absent, the chain applies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dgml_core.errors import StorageConfigInvalid
from dgml_core.storage_service import StorageConfig
from dgml_storage_s3.store import S3BlobStore


def _store(tmp_path: Path, **extra: Any) -> tuple[S3BlobStore, dict[str, Any]]:
    import boto3

    seen: dict[str, Any] = {}
    real = boto3.client

    def recording(service: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(service, **kwargs)

    cfg = StorageConfig(
        provider="dgml_storage_s3:S3BlobStore",
        root=tmp_path,
        options={"bucket": "b", **extra},
        workspace_id="ws-test",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(boto3, "client", recording)
        store = S3BlobStore(S3BlobStore.parse_config(cfg))
    return store, seen


def test_credentials_by_value_reach_the_client(tmp_path: Path) -> None:
    _, kwargs = _store(
        tmp_path,
        aws_access_key_id="AKIA...",
        aws_secret_access_key="s3cr3t",
        aws_session_token="tok",
    )
    assert kwargs["aws_access_key_id"] == "AKIA..."
    assert kwargs["aws_secret_access_key"] == "s3cr3t"
    assert kwargs["aws_session_token"] == "tok"


def test_without_them_the_chain_is_left_alone(tmp_path: Path) -> None:
    _, kwargs = _store(tmp_path)
    assert not any(k.startswith("aws_") for k in kwargs)


def test_credential_options_must_be_non_empty_strings(tmp_path: Path) -> None:
    cfg = StorageConfig(
        provider="dgml_storage_s3:S3BlobStore",
        root=tmp_path,
        options={"bucket": "b", "aws_secret_access_key": ""},
        workspace_id="ws-test",
    )
    with pytest.raises(StorageConfigInvalid, match="aws_secret_access_key"):
        S3BlobStore.parse_config(cfg)
