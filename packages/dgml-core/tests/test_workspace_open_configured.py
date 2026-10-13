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

"""`Workspace.open(configuration=…)`: no create step, meta written once, never re-identified."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import pytest
from dgml_core import DocSetStore, Workspace
from dgml_core.configuration import (
    Configuration,
    Identity,
    Model,
    Models,
    ProviderSpec,
    Storage,
)
from dgml_core.errors import ConflictError, InvalidArgument
from dgml_core.migrations import WORKSPACE_SCHEMA_VERSION, workspace_schema_version
from dgml_core.storage import user_config_path

_LOCAL = Storage.combined(ProviderSpec("dgml_core.storage_local:LocalStore"))


def _cfg(workspace_id: str = "tenant-1") -> Configuration:
    return Configuration.build(
        identity=Identity(workspace_id=workspace_id, name="Tenant", organization="TenantOrg"),
        storage=_LOCAL,
        models=Models(advanced=Model("cfg/advanced")),
    )


def test_first_open_writes_meta_and_stamps_version(tmp_path: Path) -> None:
    ws = Workspace.open(configuration=_cfg(), root=tmp_path)
    assert ws.read_meta() == {
        "name": "Tenant",
        "organization": "TenantOrg",
        "workspace_id": "tenant-1",
        "schema_version": WORKSPACE_SCHEMA_VERSION,
    }
    assert workspace_schema_version(ws) == WORKSPACE_SCHEMA_VERSION
    assert ws.is_initialized()
    assert ws.organization == "TenantOrg" and ws.display_name == "Tenant"
    assert ws.workspace_id == "tenant-1"


def test_reopen_is_idempotent(tmp_path: Path) -> None:
    Workspace.open(configuration=_cfg(), root=tmp_path)
    again = Workspace.open(configuration=_cfg(), root=tmp_path)
    assert again.read_meta()["workspace_id"] == "tenant-1"


def test_another_workspace_id_on_the_same_backend_is_a_conflict(tmp_path: Path) -> None:
    Workspace.open(configuration=_cfg("tenant-1"), root=tmp_path)
    with pytest.raises(ConflictError, match="already holds workspace 'tenant-1'") as info:
        Workspace.open(configuration=_cfg("tenant-2"), root=tmp_path)
    assert info.value.existing_id == "tenant-1"
    assert Workspace(root=tmp_path, configuration=_cfg("tenant-1")).read_meta()["name"] == "Tenant"


def test_changed_organization_reorganizes_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Same rule as `workspace create --organization`: not refused, but loud."""
    Workspace.open(configuration=_cfg(), root=tmp_path)
    renamed = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant", organization="NewOrg"),
        storage=_LOCAL,
    )
    with caplog.at_level(logging.WARNING, logger="dgml_core.storage"):
        ws = Workspace.open(configuration=renamed, root=tmp_path)
    assert "'NewOrg' differs from the 'TenantOrg'" in caplog.text
    meta = ws.read_meta()
    assert meta["organization"] == "NewOrg" and ws.organization == "NewOrg"
    assert meta["schema_version"] == WORKSPACE_SCHEMA_VERSION  # merge-preserving rewrite


def test_changed_name_is_updated_quietly(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    Workspace.open(configuration=_cfg(), root=tmp_path)
    renamed = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant v2", organization="TenantOrg"),
        storage=_LOCAL,
    )
    with caplog.at_level(logging.WARNING, logger="dgml_core.storage"):
        ws = Workspace.open(configuration=renamed, root=tmp_path)
    assert caplog.text == "" and ws.read_meta()["name"] == "Tenant v2"


def _pinned(tmp_path: Path) -> Configuration:
    """A local store told where to live, so the Workspace itself needs no root."""
    return Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant", organization="TenantOrg"),
        storage=Storage.combined(
            ProviderSpec("dgml_core.storage_local:LocalStore", {"workspace_path": str(tmp_path)})
        ),
    )


def test_no_root_means_a_scratch_dir(tmp_path: Path) -> None:
    ws = Workspace.open(configuration=_pinned(tmp_path))
    assert ws.root.is_dir() and ws.root.name.startswith("dgml-ws-")
    assert (tmp_path / "workspace.json").exists()  # the data went where the store was told


def test_local_store_without_a_root_is_refused(tmp_path: Path) -> None:
    """The local store's data *is* the root; a scratch dir would vanish with the object."""
    with pytest.raises(InvalidArgument, match=r"pass root=.*or set 'workspace_path'"):
        Workspace.open(configuration=_cfg())
    Workspace.open(configuration=_cfg(), root=tmp_path)  # with a root: fine


def test_scratch_dir_goes_with_its_workspace(tmp_path: Path) -> None:
    ws = Workspace.open(configuration=_pinned(tmp_path))
    root = ws.root
    del ws
    gc.collect()
    assert not root.exists()


def test_a_callers_root_is_left_alone(tmp_path: Path) -> None:
    ws = Workspace.open(configuration=_cfg(), root=tmp_path)
    del ws
    gc.collect()
    assert tmp_path.is_dir()


def test_path_or_id_is_refused_with_a_configuration(tmp_path: Path) -> None:
    with pytest.raises(InvalidArgument, match="whole workspace"):
        Workspace.open(tmp_path, configuration=_cfg())
    with pytest.raises(InvalidArgument, match="root applies only"):
        Workspace.open(tmp_path, root=tmp_path)


def test_nothing_on_disk_is_read_or_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the feature: user config, env and store of workspaces play no part."""
    user_config_path().parent.mkdir(parents=True, exist_ok=True)
    user_config_path().write_text('[models]\nadvanced = "user/advanced"\n', encoding="utf-8")
    monkeypatch.setenv("DGML_MODELS__ADVANCED", "env/advanced")
    from dgml_core.grounded import load_grounded_config

    ws = Workspace.open(configuration=_cfg(), root=tmp_path / "ws")
    assert load_grounded_config(ws).values_model == "cfg/advanced"
    assert ws.config_text is None and not (tmp_path / "ws" / "config.toml").exists()
    assert ws.config_location == "in-memory configuration"
    # And the usual store calls work against it.
    ds = DocSetStore(ws).create(name="Bills")
    assert DocSetStore(ws).get(ds.id).name == "Bills"


def test_writers_refuse(tmp_path: Path) -> None:
    from dgml_core import workspace_config

    ws = Workspace.open(configuration=_cfg(), root=tmp_path)
    with pytest.raises(InvalidArgument, match="never written"):
        workspace_config.write_identity(ws, name="x")
    with pytest.raises(InvalidArgument, match="never written"):
        workspace_config.reseal(ws)
