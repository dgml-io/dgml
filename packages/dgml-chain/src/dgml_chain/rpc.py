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

"""A minimal EVM JSON-RPC client over the standard library.

Only the handful of ``eth_*`` methods DGML needs to prepare, broadcast, and
confirm anchor transactions and to read anchor records. No web3.py, no extra
HTTP dependency — just ``urllib``. The NVNM MCP server's Go client wrapped the
same JSON-RPC surface; this is the Python equivalent.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from typing import Any

DEFAULT_TIMEOUT = 15


def _user_agent() -> str:
    """A descriptive User-Agent.

    urllib's default (``Python-urllib/X.Y``) is rejected with HTTP 403 by the
    WAF in front of the public NVNM RPC endpoints, so we always send our own.
    """
    try:
        return f"dgml-chain/{version('dgml-chain')}"
    except PackageNotFoundError:
        return "dgml-chain"


_USER_AGENT = _user_agent()


class RpcError(RuntimeError):
    """A JSON-RPC ``error`` envelope or a transport failure."""


def from_hex(value: str) -> int:
    """Decode a 0x-prefixed hex quantity to int."""
    return int(value, 16)


class EvmRpc:
    """Thin JSON-RPC 2.0 client for a single EVM endpoint."""

    def __init__(self, rpc_url: str, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.rpc_url = rpc_url
        self.timeout = timeout

    def _call(self, method: str, params: list[Any]) -> Any:
        body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        req = urllib.request.Request(
            self.rpc_url,
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/json", "User-Agent": _USER_AGENT},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RpcError(f"RPC transport error calling {method}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RpcError(f"RPC returned non-JSON for {method}: {exc}") from exc
        if isinstance(payload, dict) and payload.get("error"):
            err = payload["error"]
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise RpcError(f"RPC error from {method}: {msg}")
        return payload.get("result") if isinstance(payload, dict) else None

    # --- reads ---------------------------------------------------------------

    def chain_id(self) -> int:
        return from_hex(self._call("eth_chainId", []))

    def block_number(self) -> int:
        return from_hex(self._call("eth_blockNumber", []))

    def get_balance(self, address: str, block: str = "latest") -> int:
        return from_hex(self._call("eth_getBalance", [address, block]))

    def get_transaction_count(self, address: str, block: str = "pending") -> int:
        return from_hex(self._call("eth_getTransactionCount", [address, block]))

    def gas_price(self) -> int:
        return from_hex(self._call("eth_gasPrice", []))

    def max_priority_fee(self) -> int:
        return from_hex(self._call("eth_maxPriorityFeePerGas", []))

    def estimate_gas(self, tx: dict[str, Any]) -> int:
        return from_hex(self._call("eth_estimateGas", [tx]))

    def call(self, to: str, data: str, block: str = "latest") -> str:
        """``eth_call`` against a contract; returns 0x-prefixed return data."""
        result: str = self._call("eth_call", [{"to": to, "data": data}, block])
        return result

    def get_transaction_receipt(self, tx_hash: str) -> dict[str, Any] | None:
        result: dict[str, Any] | None = self._call("eth_getTransactionReceipt", [tx_hash])
        return result

    # --- writes --------------------------------------------------------------

    def send_raw_transaction(self, signed_tx_hex: str) -> str:
        """Broadcast a signed transaction; returns its hash."""
        tx_hash: str = self._call("eth_sendRawTransaction", [signed_tx_hex])
        return tx_hash
