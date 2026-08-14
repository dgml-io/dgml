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

"""Fixtures for the sample Mongo **document** store.

Two modes, same tests: real MongoDB when ``DGML_TEST_MONGO_URI`` is set (the
``docker compose`` stack), else in-process ``mongomock``. The default
``uv run pytest`` always runs the whole thing; CI additionally runs it against a
real Mongo. Every test gets its own database. The blob half of a workspace uses
the bundled local store here, so these tests need no S3.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from dgml_core.storage import Workspace
from dgml_core.storage_service import StorageConfig
from dgml_storage_mongo import MongoDocStore

MONGO_URI_ENV = "DGML_TEST_MONGO_URI"
USING_REAL_MONGO = bool(os.environ.get(MONGO_URI_ENV))
PROVIDER = "dgml_storage_mongo:MongoDocStore"


@pytest.fixture(autouse=True)
def _isolate_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Sandbox the user-level config + per-machine registry into a temp dir."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    monkeypatch.delenv("DGML_HOME", raising=False)


@pytest.fixture(autouse=True)
def _fake_mongo(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Use in-process mongomock unless a real Mongo URI is configured."""
    if USING_REAL_MONGO:
        monkeypatch.setenv("DGML_MONGO_URI", os.environ[MONGO_URI_ENV])
        yield
        return
    import mongomock
    import pymongo

    monkeypatch.delenv("DGML_MONGO_URI", raising=False)
    # store.py imports MongoClient inside __init__, so patching the attribute
    # before construction is enough.
    monkeypatch.setattr(pymongo, "MongoClient", mongomock.MongoClient)
    yield


@pytest.fixture
def mongo_config(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        provider=PROVIDER,
        root=tmp_path / "ws",
        options={"mongo_database": f"dgml_test_{uuid.uuid4().hex[:12]}"},
    )


@pytest.fixture
def docs(mongo_config: StorageConfig) -> MongoDocStore:
    return MongoDocStore(MongoDocStore.parse_config(mongo_config))


@pytest.fixture
def mongo_docs_workspace(tmp_path: Path) -> Workspace:
    """A workspace whose **docs** live in Mongo and **blobs** on local disk. The
    Mongo backend is the ``default`` service's doc role, so an unregistered
    ``Workspace`` resolves it with no registry entry needed."""
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    db = f"dgml_test_{uuid.uuid4().hex[:12]}"
    (root / "config.toml").write_text(
        f'[storage.default.docs]\nprovider = "{PROVIDER}"\nmongo_database = "{db}"\n',
        encoding="utf-8",
    )
    ws = Workspace(root=root)
    ws.init()  # docs → Mongo, blobs → local disk
    return ws
