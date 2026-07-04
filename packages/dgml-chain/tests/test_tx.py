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

from typing import Any

from dgml_chain.rpc import RpcError
from dgml_chain.tx import GWEI, build_unsigned_tx

_FROM = "0x000000000000000000000000000000000000dEaD"
_TO = "0x0000000000000000000000000000000000000A00"


class FakeRpc:
    """Stand-in for EvmRpc; records nothing, returns fixed quotes."""

    def __init__(self, *, tip: int | None = 5 * GWEI, gas_price: int = 10 * GWEI) -> None:
        self._tip = tip
        self._gas_price = gas_price

    def get_transaction_count(self, address: str, block: str = "pending") -> int:
        return 7

    def estimate_gas(self, tx: dict[str, Any]) -> int:
        return 100_000

    def gas_price(self) -> int:
        return self._gas_price

    def max_priority_fee(self) -> int:
        if self._tip is None:
            raise RpcError("eth_maxPriorityFeePerGas not supported")
        return self._tip


def test_eip1559_tx_shape() -> None:
    tx = build_unsigned_tx(
        FakeRpc(),  # type: ignore[arg-type]
        from_address=_FROM,
        to=_TO,
        data="0xabcd",
        chain_id=787111,
    )
    assert tx["type"] == 2
    assert tx["nonce"] == 7
    assert tx["gas"] == 120_000  # 100_000 * 1.2
    assert tx["max_priority_fee_per_gas"] == 5 * GWEI
    assert tx["max_fee_per_gas"] == 25 * GWEI  # 2 * gas_price + tip
    assert "gas_price" not in tx
    assert tx["chain_id"] == 787111


def test_max_fee_never_below_tip() -> None:
    # A node-reported tip larger than 2*gasPrice must not produce an invalid
    # tx (maxFeePerGas < maxPriorityFeePerGas) the node would reject.
    tx = build_unsigned_tx(
        FakeRpc(tip=50 * GWEI, gas_price=1 * GWEI),  # type: ignore[arg-type]
        from_address=_FROM,
        to=_TO,
        data="0x",
        chain_id=1,
    )
    assert tx["max_fee_per_gas"] >= tx["max_priority_fee_per_gas"]


def test_legacy_tx_shape() -> None:
    tx = build_unsigned_tx(
        FakeRpc(),  # type: ignore[arg-type]
        from_address=_FROM,
        to=_TO,
        data="0xabcd",
        chain_id=1337,
        legacy=True,
    )
    assert tx["type"] == 0
    assert tx["gas_price"] == 10 * GWEI
    assert "max_fee_per_gas" not in tx


def test_tip_falls_back_to_one_gwei_when_unsupported() -> None:
    tx = build_unsigned_tx(
        FakeRpc(tip=None),  # type: ignore[arg-type]
        from_address=_FROM,
        to=_TO,
        data="0x",
        chain_id=1,
    )
    assert tx["max_priority_fee_per_gas"] == GWEI


def test_zero_tip_floored_to_one_gwei() -> None:
    tx = build_unsigned_tx(
        FakeRpc(tip=0),  # type: ignore[arg-type]
        from_address=_FROM,
        to=_TO,
        data="0x",
        chain_id=1,
    )
    assert tx["max_priority_fee_per_gas"] == GWEI
