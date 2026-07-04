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

"""Build an unsigned EVM transaction for an anchor write.

Ports the nonce/gas/fee logic from the NVNM MCP server's ``prepare.go``:

- nonce from the pending count (so back-to-back writes don't collide),
- gas = estimate * 1.2 (20% headroom over the precompile's estimate),
- EIP-1559 (type-2) by default: ``maxPriorityFeePerGas`` from the node (1 gwei
  floor) and ``maxFeePerGas = 2 * gasPrice`` for base-fee headroom,
- legacy (type-0) on request, using a single ``gasPrice``.

The returned dict uses snake_case keys; ``signer.sign_tx`` translates them to
the camelCase eth_account expects. Keeping the unsigned tx as plain data lets
callers inspect or persist it before signing (``--dry-run``).
"""

from __future__ import annotations

from typing import Any

from .rpc import EvmRpc, RpcError

GWEI = 10**9


def build_unsigned_tx(
    rpc: EvmRpc,
    *,
    from_address: str,
    to: str,
    data: str,
    chain_id: int,
    value: int = 0,
    legacy: bool = False,
) -> dict[str, Any]:
    """Assemble an unsigned transaction, fetching nonce/gas/fees from ``rpc``."""
    nonce = rpc.get_transaction_count(from_address, "pending")
    gas_estimate = rpc.estimate_gas(
        {"from": from_address, "to": to, "data": data, "value": hex(value)}
    )
    gas = gas_estimate * 12 // 10  # 20% buffer, matching the Go server

    tx: dict[str, Any] = {
        "chain_id": chain_id,
        "nonce": nonce,
        "gas": gas,
        "to": to,
        "value": value,
        "data": data,
    }

    if legacy:
        tx["type"] = 0
        tx["gas_price"] = rpc.gas_price()
        return tx

    gas_price = rpc.gas_price()
    try:
        tip = rpc.max_priority_fee()
    except RpcError:
        tip = GWEI
    if tip <= 0:
        tip = GWEI
    tx["type"] = 2
    tx["max_priority_fee_per_gas"] = tip
    # 2x gasPrice for base-fee headroom, plus the tip so the invariant
    # maxFeePerGas >= maxPriorityFeePerGas always holds (a node-reported tip
    # larger than 2x gasPrice would otherwise produce a tx the node rejects).
    tx["max_fee_per_gas"] = 2 * gas_price + tip
    return tx
