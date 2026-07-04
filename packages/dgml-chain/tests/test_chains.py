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

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dgml_chain.chains import ANCHOR_PRECOMPILE, ChainConfig, ChainStore


def test_builtins_present_without_config_file(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "chains.json")
    names = set(store.all())
    assert {"nvnm-testnet", "nvnm-mainnet"} <= names
    testnet = store.get("nvnm-testnet")
    assert testnet.chain_id == 787111
    assert testnet.anchor_address == ANCHOR_PRECOMPILE


def test_add_persists_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "chains.json"
    store = ChainStore(path)
    store.add(ChainConfig(name="local", rpc_url="http://localhost:8545", chain_id=1337))
    # New store instance reads the same file back.
    reloaded = ChainStore(path)
    assert reloaded.get("local").chain_id == 1337
    assert json.loads(path.read_text())["local"]["rpc_url"] == "http://localhost:8545"


def test_cannot_redefine_or_remove_builtin(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "chains.json")
    with pytest.raises(ValueError):
        store.add(ChainConfig(name="nvnm-testnet", rpc_url="http://x", chain_id=1))
    with pytest.raises(ValueError):
        store.remove("nvnm-testnet")


def test_remove_custom(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "chains.json")
    store.add(ChainConfig(name="local", rpc_url="http://x", chain_id=1))
    store.remove("local")
    assert "local" not in store.all()
    with pytest.raises(KeyError):
        store.remove("local")


def test_get_unknown_raises(tmp_path: Path) -> None:
    store = ChainStore(tmp_path / "chains.json")
    with pytest.raises(KeyError):
        store.get("nope")
