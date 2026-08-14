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

"""The point of the split: **S3 blobs + Mongo docs** composed into one workspace.

Both sample packages are workspace members, so both import in the shared venv.
Uses moto (from the package's autouse fixture) + mongomock, or the real MinIO +
MongoDB when DGML_TEST_S3_ENDPOINT / DGML_TEST_MONGO_URI are set.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from dgml_core import layout
from dgml_core.storage import Workspace
from dgml_storage_mongo import MongoDocStore
from dgml_storage_s3 import S3BlobStore

from .conftest import PROVIDER, make_bucket

MONGO_PROVIDER = "dgml_storage_mongo:MongoDocStore"


@pytest.fixture
def _fake_mongo(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if os.environ.get("DGML_TEST_MONGO_URI"):
        monkeypatch.setenv("DGML_MONGO_URI", os.environ["DGML_TEST_MONGO_URI"])
        yield
        return
    import mongomock
    import pymongo

    monkeypatch.delenv("DGML_MONGO_URI", raising=False)
    monkeypatch.setattr(pymongo, "MongoClient", mongomock.MongoClient)
    yield


@pytest.fixture
def mixed_workspace(_fake_mongo: None, tmp_path: Path) -> Workspace:
    """A workspace with S3 blobs and Mongo docs, as the ``default`` service."""
    _bucket, s3_opts = make_bucket()
    db = f"dgml_test_{_bucket[-12:]}"
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    s3_lines = "\n".join(f'{k} = "{v}"' for k, v in s3_opts.items())
    (root / "config.toml").write_text(
        f'[storage.default.blobs]\nprovider = "{PROVIDER}"\n{s3_lines}\n\n'
        f'[storage.default.docs]\nprovider = "{MONGO_PROVIDER}"\nmongo_database = "{db}"\n',
        encoding="utf-8",
    )
    ws = Workspace(root=root)
    ws.init()
    return ws


def test_blobs_on_s3_docs_on_mongo(mixed_workspace: Workspace) -> None:
    ws = mixed_workspace
    assert isinstance(ws.blobs, S3BlobStore)
    assert isinstance(ws.docs, MongoDocStore)

    ws.blobs.put_blob("files/f1/report.pdf", b"pdf")
    ws.docs.put_doc(layout.Collection.FILES, "f1", {"id": "f1", "sha256": "aa"})

    # Nothing landed on local disk: blobs are in S3, documents in Mongo.
    assert not (ws.root / "files" / "f1" / "report.pdf").exists()
    assert not (ws.root / "files" / "f1" / "file.json").exists()
    assert ws.blobs.get_blob("files/f1/report.pdf") == b"pdf"
    assert ws.docs.get_doc(layout.Collection.FILES, "f1") == {"id": "f1", "sha256": "aa"}


def test_workspace_meta_round_trips_through_mongo(mixed_workspace: Workspace) -> None:
    ws = mixed_workspace
    ws.write_meta(name="Acme", organization="acme", workspace_id="ws_composedxxxxxxx")
    assert ws.read_meta()["name"] == "Acme"
    assert ws.workspace_id == "ws_composedxxxxxxx"
