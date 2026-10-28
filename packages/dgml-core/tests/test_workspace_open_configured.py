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
from dgml_core.storage_local import LocalStore

from .conftest import register_provider, scratch_configuration

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


def test_workspace_path_becomes_the_root(tmp_path: Path) -> None:
    """A local store told where to live makes that the workspace root — not a scratch
    dir beside it — so `files_dir` and every other direct reader of `ws.root` look at
    the tree the store actually writes."""
    ws = Workspace.open(configuration=_pinned(tmp_path))
    assert ws.root == tmp_path.resolve()
    assert ws.files_dir == tmp_path.resolve() / "files"
    assert (tmp_path / "workspace.json").exists()


def test_local_store_without_a_root_is_refused(tmp_path: Path) -> None:
    """The local store's data *is* the root; a scratch dir would vanish with the object."""
    with pytest.raises(InvalidArgument, match=r"pass root=.*or set 'workspace_path'"):
        Workspace.open(configuration=_cfg())
    Workspace.open(configuration=_cfg(), root=tmp_path)  # with a root: fine


def test_a_workspace_path_root_is_not_removed_with_the_workspace(tmp_path: Path) -> None:
    """The caller's directory is the root, and it is not ours to remove."""
    ws = Workspace.open(configuration=_pinned(tmp_path))
    root = ws.root
    del ws
    gc.collect()
    assert root.exists() and (root / "workspace.json").exists()


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


def test_local_store_alias_without_a_root_is_refused() -> None:
    """The guard decides on the resolved class, not the provider string: another
    spelling of the same class is still the local store."""
    alias = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant", organization="TenantOrg"),
        storage=Storage.combined(ProviderSpec("dgml_core:LocalStore")),
    )
    with pytest.raises(InvalidArgument, match=r"pass root=.*or set 'workspace_path'"):
        Workspace.open(configuration=alias)


def test_a_relative_root_is_resolved_at_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ws.root` must not move with the working directory: the store wrote where the
    caller's path pointed at open time, and every direct reader must keep seeing it."""
    monkeypatch.chdir(tmp_path)
    ws = Workspace.open(configuration=_cfg(), root=Path("ws"))
    assert ws.root == (tmp_path / "ws").resolve()
    monkeypatch.chdir(tmp_path.parent)
    assert ws.files_dir == (tmp_path / "ws").resolve() / "files"


def test_root_must_agree_with_workspace_path(tmp_path: Path) -> None:
    """An explicit `root=` beside a `workspace_path` is fine when they are the same
    directory (however spelled) and refused when they are not: `ws.root` must be the
    tree the store writes, or every direct reader of it looks at the wrong place."""
    cfg = _pinned(tmp_path / "ws")
    ws = Workspace.open(configuration=cfg, root=tmp_path / "ws")
    assert ws.root == (tmp_path / "ws").resolve()
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "ws")
    Workspace.open(configuration=cfg, root=link)  # another spelling of the same place
    with pytest.raises(InvalidArgument, match="disagree"):
        Workspace.open(configuration=cfg, root=tmp_path / "elsewhere")


def test_blobs_and_docs_with_different_workspace_paths_are_refused(tmp_path: Path) -> None:
    split = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant", organization="TenantOrg"),
        storage=Storage(
            blobs=ProviderSpec(
                "dgml_core.storage_local:LocalStore", {"workspace_path": str(tmp_path / "a")}
            ),
            docs=ProviderSpec(
                "dgml_core.storage_local:LocalStore", {"workspace_path": str(tmp_path / "b")}
            ),
        ),
    )
    with pytest.raises(InvalidArgument, match="different 'workspace_path'"):
        Workspace.open(configuration=split)


# ---- the scratch-root path: every store remote, no local data at all ------------


def test_local_store_subclass_without_a_root_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subclass is still the local store: its data is still the root."""

    class _Sub(LocalStore):
        pass

    sub = Configuration.build(
        identity=Identity(workspace_id="tenant-1", name="Tenant", organization="TenantOrg"),
        storage=Storage.combined(ProviderSpec(register_provider(monkeypatch, _Sub, "Sub"))),
    )
    with pytest.raises(InvalidArgument, match=r"pass root=.*or set 'workspace_path'"):
        Workspace.open(configuration=sub)


def test_scratch_dir_goes_with_its_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every store remote there is no local data, so the root is a scratch dir
    that holds nothing and is removed with the object."""
    ws = Workspace.open(configuration=scratch_configuration(monkeypatch))
    root = ws.root
    assert ws.scratch
    assert root.is_dir() and root.name.startswith("dgml-ws-")
    assert ws.read_meta()["workspace_id"] == "tenant-1"  # went to the store, not to disk
    assert list(root.iterdir()) == []
    del ws
    gc.collect()
    assert not root.exists()


def test_only_a_generated_root_is_scratch(tmp_path: Path) -> None:
    assert not Workspace.open(configuration=_cfg(), root=tmp_path).scratch
    assert not Workspace.open(configuration=_pinned(tmp_path)).scratch
    with pytest.raises(InvalidArgument, match="scratch root"):
        Workspace(root=tmp_path, scratch=True)


def test_a_scratch_workspace_caches_embeddings_for_its_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache is not lasting output: it lives under the scratch root, so repeated
    runs on a held `Workspace` reuse it, and it goes with the root at GC."""
    ws = Workspace.open(configuration=scratch_configuration(monkeypatch))
    cache = ws.embedding_cache_dir
    assert cache.is_relative_to(ws.root)
    cache.mkdir(parents=True)
    (cache / "x.npy").write_bytes(b"")
    assert ws.embedding_cache_dir == cache  # stable while the object lives
    del ws
    gc.collect()
    assert not cache.exists()
