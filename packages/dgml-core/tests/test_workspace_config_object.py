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

"""`Workspace.config`: one `Configuration` per workspace, supplied or derived.

For a workspace addressed by path the object is derived from the TOML merge and
must equal what the merge itself returns, so a loader and a `ws.config` reader
never disagree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dgml_core.config import load_merged_config
from dgml_core.configuration import Configuration, Identity, Model, Models, ProviderSpec, Storage
from dgml_core.errors import CorruptMetadata, InvalidArgument, StorageConfigInvalid
from dgml_core.models_config import ConfigSection
from dgml_core.storage import Workspace, user_config_path
from dgml_core.storage_resolve import load_store_configs, resolve_store_configs

_LOCAL_TABLE = '[storage]\nprovider = "dgml_core.storage_local:LocalStore"\n'
_CFG = Configuration.build(
    identity=Identity(workspace_id="tenant-1", name="T", organization="Org"),
    storage=Storage.combined(ProviderSpec("dgml_core.storage_local:LocalStore")),
)


def _write_toml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_supplied_configuration_is_config(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path, configuration=_CFG)
    assert ws.config is _CFG


def test_in_memory_cli_overrides_layer_like_toml_ones(tmp_path: Path) -> None:
    """Same rules as the settings model on the TOML path: deep-merge, drop an undeclared
    section, reject a non-table."""
    cfg = Configuration.build(
        identity=_CFG.identity,
        storage=_CFG.storage,
        models=Models(light=Model("cfg/light"), standard=Model("cfg/std")),
    )
    ws = Workspace(root=tmp_path, configuration=cfg)
    merged = load_merged_config(ws, cli_overrides={"models": {"light": "cli/light"}, "other": {}})
    assert merged[ConfigSection.MODELS] == {"light": "cli/light", "standard": "cfg/std"}
    assert set(merged) == {ConfigSection.MODELS}
    with pytest.raises(CorruptMetadata):
        load_merged_config(ws, cli_overrides={"generation": "haiku"})


def test_load_store_configs_serves_the_in_memory_binding(tmp_path: Path) -> None:
    """The public resolver must not fall through to the local default for a workspace
    whose binding lives on its Configuration rather than in a [storage] table."""
    cfg = Configuration.build(
        identity=_CFG.identity,
        storage=Storage(
            blobs=ProviderSpec("s3:Blobs", {"bucket": "b"}),
            docs=ProviderSpec("mongo:Docs", {"mongo_database": "d"}),
        ),
    )
    ws = Workspace(root=tmp_path, configuration=cfg)
    blobs, docs = load_store_configs(ws)
    assert (blobs.provider, docs.provider) == ("s3:Blobs", "mongo:Docs")
    assert blobs.workspace_id == "tenant-1" and blobs.root == tmp_path
    assert load_store_configs(ws) == resolve_store_configs(ws)
    with pytest.raises(StorageConfigInvalid, match="configured in memory"):
        load_store_configs(ws, "other")


def test_in_memory_configuration_is_exclusive_with_other_sources(tmp_path: Path) -> None:
    with pytest.raises(InvalidArgument, match="only config source"):
        Workspace(root=tmp_path, configuration=_CFG, workspaces_id="tenant-1")
    with pytest.raises(InvalidArgument, match="only config source"):
        Workspace(root=tmp_path, configuration=_CFG, config_override=tmp_path / "c.toml")


def test_configuration_is_not_part_of_equality(tmp_path: Path) -> None:
    assert Workspace(root=tmp_path, configuration=_CFG) == Workspace(root=tmp_path)


def test_scratch_root_is_a_fresh_empty_dir() -> None:
    a, b = Workspace.scratch_root(), Workspace.scratch_root()
    assert a.is_dir() and b.is_dir() and a != b and not any(a.iterdir())


def test_derived_config_matches_the_toml_merge(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_toml(
        user_config_path(),
        '[models]\nadvanced = "user/adv"\nlight = "user/light"\n[grounded]\nmax_tool_iters = 7\n',
    )
    _write_toml(
        workspace.config_path,
        '[workspace]\nworkspace_id = "wsid00000001"\nname = "N"\norganization = "O"\n'
        '[models]\nadvanced = "ws/adv"\n' + _LOCAL_TABLE,
    )
    monkeypatch.setenv("DGML_MODELS__LIGHT", "env/light")

    cfg = workspace.config
    assert cfg.sections == load_merged_config(workspace)
    assert cfg.sections[ConfigSection.MODELS]["advanced"] == "ws/adv"  # workspace over user
    assert cfg.sections[ConfigSection.MODELS]["light"] == "env/light"  # env over both
    assert cfg.sections[ConfigSection.GROUNDED] == {"max_tool_iters": 7}
    assert cfg.identity == Identity(workspace_id="wsid00000001", name="N", organization="O")

    expected = resolve_store_configs(workspace)
    assert cfg.storage.store_configs(root=workspace.root, workspace_id="wsid00000001") == expected


def test_derived_config_is_always_fresh(workspace: Workspace) -> None:
    _write_toml(workspace.config_path, _LOCAL_TABLE + '[models]\nadvanced = "one"\n')
    assert workspace.config.sections[ConfigSection.MODELS]["advanced"] == "one"
    _write_toml(workspace.config_path, _LOCAL_TABLE + '[models]\nadvanced = "two"\n')
    assert workspace.config.sections[ConfigSection.MODELS]["advanced"] == "two"


def test_derived_identity_may_be_partial(workspace: Workspace) -> None:
    _write_toml(
        workspace.config_path, '[storage]\nprovider = "dgml_core.storage_local:LocalStore"\n'
    )
    assert workspace.config.identity == Identity()


def test_build_requires_a_complete_identity() -> None:
    with pytest.raises(InvalidArgument, match="missing organization"):
        Configuration.build(
            identity=Identity(workspace_id="t", name="n"),
            storage=Storage.combined(ProviderSpec("x:Y")),
        )
