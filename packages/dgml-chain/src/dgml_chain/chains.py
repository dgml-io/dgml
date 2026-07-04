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

"""Chain configuration: built-in NVNM defaults plus user-added chains.

A chain is a config entry — RPC URL, EVM chain id, and the anchor precompile
address. NVNM testnet and mainnet ship as protected built-ins; any other EVM
chain that exposes a compatible anchor precompile can be added by the user and
persisted to a JSON file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The anchor precompile lives at a fixed address on NVNM (identical across
# testnet and mainnet). Other chains may deploy the anchor interface elsewhere,
# so it is configurable per chain but defaults to the NVNM precompile.
ANCHOR_PRECOMPILE = "0x0000000000000000000000000000000000000A00"


@dataclass(frozen=True)
class ChainConfig:
    """An EVM chain DGML can anchor to."""

    name: str
    rpc_url: str
    chain_id: int
    anchor_address: str = ANCHOR_PRECOMPILE
    explorer: str | None = None
    native_token: str | None = None

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "rpc_url": self.rpc_url,
            "chain_id": self.chain_id,
            "anchor_address": self.anchor_address,
        }
        if self.explorer:
            d["explorer"] = self.explorer
        if self.native_token:
            d["native_token"] = self.native_token
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> ChainConfig:
        try:
            return cls(
                name=str(d["name"]),
                rpc_url=str(d["rpc_url"]),
                chain_id=int(d["chain_id"]),
                anchor_address=str(d.get("anchor_address", ANCHOR_PRECOMPILE)),
                explorer=d.get("explorer"),
                native_token=d.get("native_token"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid chain config: {exc}") from exc


# Seeded from the NVNM chain reference. Native token is the EVM gas token
# (wrapped); shown for humans, not used in any calculation.
BUILTIN_CHAINS: dict[str, ChainConfig] = {
    "nvnm-testnet": ChainConfig(
        name="nvnm-testnet",
        rpc_url="https://evm.testnet.nvnmchain.io",
        chain_id=787111,
        explorer="https://explorer.evm.testnet.nvnmchain.io",
        native_token="wmantraUSD",
    ),
    "nvnm-mainnet": ChainConfig(
        name="nvnm-mainnet",
        rpc_url="https://evm.nvnmchain.io",
        chain_id=1611,
        explorer="https://evm.explorer.nvnmchain.io",
        native_token="wmmUSD",
    ),
}


class ChainStore:
    """Built-in chains overlaid with user-added chains from a JSON file.

    The config file is a JSON object ``{"<name>": {chain config}, ...}``. It is
    created lazily on the first ``add``. Built-in chains cannot be removed or
    shadowed (adding a chain whose name collides with a built-in is rejected).
    """

    def __init__(self, config_path: Path | None) -> None:
        self.config_path = config_path
        self._custom: dict[str, ChainConfig] = self._load()

    def _load(self) -> dict[str, ChainConfig]:
        if self.config_path is None or not self.config_path.exists():
            return {}
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"chain config {self.config_path} is not a JSON object")
        return {name: ChainConfig.from_json(cfg) for name, cfg in raw.items()}

    def _save(self) -> None:
        if self.config_path is None:
            raise ValueError("no chain-config path set; cannot persist custom chains")
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {name: cfg.to_json() for name, cfg in sorted(self._custom.items())}
        self.config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def all(self) -> dict[str, ChainConfig]:
        """All chains: built-ins overlaid with (never shadowed by) custom ones."""
        return {**BUILTIN_CHAINS, **self._custom}

    def get(self, name: str) -> ChainConfig:
        chains = self.all()
        if name not in chains:
            known = ", ".join(sorted(chains)) or "(none)"
            raise KeyError(f"unknown chain {name!r}; configured chains: {known}")
        return chains[name]

    def add(self, cfg: ChainConfig) -> None:
        if cfg.name in BUILTIN_CHAINS:
            raise ValueError(f"{cfg.name!r} is a built-in chain and cannot be redefined")
        self._custom[cfg.name] = cfg
        self._save()

    def remove(self, name: str) -> None:
        if name in BUILTIN_CHAINS:
            raise ValueError(f"{name!r} is a built-in chain and cannot be removed")
        if name not in self._custom:
            raise KeyError(f"no custom chain named {name!r}")
        del self._custom[name]
        self._save()

    def is_builtin(self, name: str) -> bool:
        return name in BUILTIN_CHAINS
