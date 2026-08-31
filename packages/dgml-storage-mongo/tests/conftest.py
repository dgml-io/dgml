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

"""Fixtures for the sample Mongo stores — documents, blobs, and both at once.

Two modes, same tests: real MongoDB when ``DGML_TEST_MONGO_URI`` is set (the
``docker compose`` stack), else in-process ``mongomock``. The default
``uv run pytest`` always runs the whole thing; CI additionally runs it against a
real Mongo. Every test gets its own database.

``mongo_docs_workspace`` keeps blobs on local disk (so those tests need no S3);
``mongo_gridfs_workspace`` puts *both* roles in Mongo via ``MongoGridFSStore``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dgml_core.storage import Workspace
from dgml_core.storage_service import StorageConfig
from dgml_core.workspaces_resolve import default_workspaces_store
from dgml_core.workspaces_store import WORKSPACES_ENV_VAR, WorkspacesConfig
from dgml_storage_mongo import MongoDocStore, MongoGridFSBlobStore, MongoWorkspacesStore

MONGO_URI_ENV = "DGML_TEST_MONGO_URI"
USING_REAL_MONGO = bool(os.environ.get(MONGO_URI_ENV))
PROVIDER = "dgml_storage_mongo:MongoDocStore"
GRIDFS_PROVIDER = "dgml_storage_mongo:MongoGridFSBlobStore"
BOTH_GRIDFS_PROVIDER = "dgml_storage_mongo:MongoGridFSStore"


@pytest.fixture(autouse=True)
def _isolate_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Sandbox the user-level config and the machine's store of workspaces into temp
    dirs, so neither the developer's real config nor their ``~/dgml-workspaces`` is
    touched. The memoized store is cleared for the same reason as in the other suites."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    monkeypatch.setenv(WORKSPACES_ENV_VAR, str(tmp_path_factory.mktemp("dgml-workspaces")))
    monkeypatch.delenv("DGML_HOME", raising=False)
    default_workspaces_store.cache_clear()


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
    monkeypatch.delenv("DGML_WORKSPACES_MONGO_URI", raising=False)

    # A URI-memoizing factory, NOT `mongomock.MongoClient` directly. Two mongomock
    # clients built from the same URI are separate in-memory universes, so two store
    # instances that should be looking at one database would silently see different
    # data — and a concurrency test written against them would pass while asserting
    # nothing. Same shape of silent green the README warns about for GridFS.
    universes: dict[str, Any] = {}

    def _client(uri: str = "mongodb://localhost:27017", *args: Any, **kwargs: Any) -> Any:
        if uri not in universes:
            universes[uri] = mongomock.MongoClient(uri, *args, **kwargs)
        return universes[uri]

    # store.py imports MongoClient inside __init__, so patching the attribute
    # before construction is enough.
    monkeypatch.setattr(pymongo, "MongoClient", _client)
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


@pytest.fixture(autouse=True)
def _mongomock_gridfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``mongomock`` drive ``GridFSBucket``.

    Two gaps to close. ``gridfs`` type-checks its arguments with ``isinstance``
    against pymongo classes, which ``enable_gridfs_integration`` widens to accept
    mongomock's. And ``mongomock.MongoClient`` has no ``.options`` property, so
    GridFSBucket's ``db.client.options.timeout`` falls through mongomock's
    attribute-style database accessor and ``_timeout`` becomes a ``Collection``,
    tripping pymongo's timeout wrapper on first upload.

    Note the shape of that failure if you ever debug it: ``_TimeoutContext``
    sets its context variable *before* raising, so the first GridFS call fails
    and every later one silently passes. A batch run can look green off one
    poisoned test — check GridFS tests individually.
    """
    if USING_REAL_MONGO:
        return
    import mongomock
    from mongomock.gridfs import enable_gridfs_integration

    enable_gridfs_integration()
    monkeypatch.setattr(
        mongomock.MongoClient,
        "options",
        property(lambda self: SimpleNamespace(timeout=None)),
        raising=False,
    )


@pytest.fixture
def blobs(mongo_config: StorageConfig) -> MongoGridFSBlobStore:
    return MongoGridFSBlobStore(MongoGridFSBlobStore.parse_config(mongo_config))


def chunk_collection(store: MongoGridFSBlobStore) -> Any:
    """The GridFS bucket's chunk collection.

    Reaches inside the store because the properties it backs — an overwrite
    leaving no orphaned chunks, a torn read being detected — are only observable
    from there."""
    return store._files.database[f"{store._files.name.rsplit('.', 1)[0]}.chunks"]


@pytest.fixture
def mongo_gridfs_workspace(tmp_path: Path) -> Workspace:
    """Both roles in Mongo, with blobs in GridFS."""
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    db = f"dgml_test_{uuid.uuid4().hex[:12]}"
    (root / "config.toml").write_text(
        f'[storage.default]\nprovider = "{BOTH_GRIDFS_PROVIDER}"\nmongo_database = "{db}"\n',
        encoding="utf-8",
    )
    ws = Workspace(root=root)
    ws.init()
    return ws


@pytest.fixture
def workspaces_config(tmp_path: Path) -> WorkspacesConfig:
    """A per-test database for the store of workspaces."""
    return WorkspacesConfig(
        provider="dgml_storage_mongo:MongoWorkspacesStore",
        options={"mongo_database": f"dgml_ws_test_{uuid.uuid4().hex[:12]}"},
    )


@pytest.fixture
def workspaces_store(workspaces_config: WorkspacesConfig) -> MongoWorkspacesStore:
    return MongoWorkspacesStore(MongoWorkspacesStore.parse_config(workspaces_config))


@pytest.fixture
def workspaces_store_b(workspaces_config: WorkspacesConfig) -> MongoWorkspacesStore:
    """A *second* instance over the same database, for the lost-update tests.

    Two instances rather than two calls on one, because the conflict being tested is
    between two processes that each read, then each write.
    """
    return MongoWorkspacesStore(MongoWorkspacesStore.parse_config(workspaces_config))
