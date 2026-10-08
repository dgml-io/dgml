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

"""`mongo_uri` by value: wins over the environment, validated, and an in-memory
workspace on GridFS leaves the local root empty when a file is added."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from dgml_core import FileStore, Workspace, layout
from dgml_core.configuration import Configuration, Identity, ProviderSpec, Storage
from dgml_core.errors import StorageConfigInvalid
from dgml_storage_mongo._client import connect, validate_identity

from .conftest import BOTH_GRIDFS_PROVIDER


def test_mongo_uri_option_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    import pymongo

    seen: list[str] = []

    def recording(uri: str, *args: Any, **kwargs: Any) -> Any:
        # A stub, not a client: under DGML_TEST_MONGO_URI the real pymongo would try
        # to resolve these made-up hosts, which the offline guard refuses.
        seen.append(uri)
        return MagicMock()

    monkeypatch.setattr(pymongo, "MongoClient", recording)
    monkeypatch.setenv("DGML_MONGO_URI", "mongodb://env-host:27017")

    connect({"mongo_database": "d", "mongo_uri": "mongodb://opt-host:27017"})
    connect({"mongo_database": "d"})
    assert seen == ["mongodb://opt-host:27017", "mongodb://env-host:27017"]


def test_mongo_uri_must_be_a_non_empty_string() -> None:
    with pytest.raises(StorageConfigInvalid, match="mongo_uri"):
        validate_identity("mongo", {"mongo_database": "d", "mongo_uri": ""})
    with pytest.raises(StorageConfigInvalid, match="mongo_uri"):
        validate_identity("mongo", {"mongo_database": "d", "mongo_uri": 3})


def _minimal_pdf() -> bytes:
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<</Size {len(objs) + 1}/Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def test_in_memory_gridfs_workspace_keeps_the_root_empty_on_file_add(tmp_path: Path) -> None:
    src = tmp_path / "doc.pdf"
    src.write_bytes(_minimal_pdf())
    root = tmp_path / "root"
    root.mkdir()
    cfg = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="T", organization="Org"),
        storage=Storage.combined(
            ProviderSpec(
                BOTH_GRIDFS_PROVIDER, {"mongo_database": f"dgml_test_{uuid.uuid4().hex[:12]}"}
            )
        ),
    )
    ws = Workspace.open(configuration=cfg, root=root)
    result = FileStore(ws).add(src)  # page rendering may soft-fail without ghostscript
    assert result.created
    assert ws.blobs.blob_exists(layout.file_source_key(result.record.id, "doc.pdf"))
    assert result.record.original_path == str(src)  # absolute: no portable root to be relative to
    assert list(root.iterdir()) == []  # every artifact went to GridFS; staging used tempfile
