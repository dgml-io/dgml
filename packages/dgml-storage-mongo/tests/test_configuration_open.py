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

"""An in-memory `Configuration` opening a Mongo-backed workspace: meta lands in Mongo,
nothing lands on disk, and a second workspace id on the same database is its own
namespace."""

from __future__ import annotations

import uuid
from pathlib import Path

from dgml_core import DocSetStore, Workspace
from dgml_core.configuration import Configuration, Identity, ProviderSpec, Storage

from .conftest import BOTH_GRIDFS_PROVIDER


def _cfg(db: str, workspace_id: str = "tenant-1") -> Configuration:
    return Configuration.build(
        identity=Identity(workspace_id=workspace_id, name="T", organization="Org"),
        storage=Storage.combined(ProviderSpec(BOTH_GRIDFS_PROVIDER, {"mongo_database": db})),
    )


def test_open_writes_meta_to_mongo_and_nothing_to_disk(tmp_path: Path) -> None:
    db = f"dgml_test_{uuid.uuid4().hex[:12]}"
    ws = Workspace.open(configuration=_cfg(db), root=tmp_path)
    assert ws.read_meta()["workspace_id"] == "tenant-1"
    ds = DocSetStore(ws).create(name="Bills")
    assert DocSetStore(ws).get(ds.id).name == "Bills"
    assert list(tmp_path.iterdir()) == []  # remote stores: the root stays empty

    # A second process with the same Configuration finds the same workspace.
    again = Workspace.open(configuration=_cfg(db), root=tmp_path)
    assert again.read_meta()["workspace_id"] == "tenant-1"
    assert [d.name for d in DocSetStore(again).list_all()] == ["Bills"]

    # Mongo namespaces every document by workspace id, so another id on the same
    # database is a *separate* workspace, not a conflict (that is LocalStore's case).
    other = Workspace.open(configuration=_cfg(db, "tenant-2"), root=tmp_path)
    assert other.read_meta()["workspace_id"] == "tenant-2"
    assert DocSetStore(other).list_all() == []
    assert [d.name for d in DocSetStore(again).list_all()] == ["Bills"]
