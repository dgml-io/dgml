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

"""Unit tests for dgml_core.staking helpers that need no chain or workspace."""

from __future__ import annotations

from typing import Any

import pytest
from dgml_chain import ChainConfig
from dgml_core.errors import ChainTxReverted, RecordNotFound
from dgml_core.staking import (
    _fetch_record,
    _find_record,
    _finish_stake,
    _wait_for_receipt,
    _wei_to_eth,
)


@pytest.mark.parametrize(
    ("wei", "expected"),
    [
        (0, "0"),
        (10**18, "1"),
        (10 * 10**18, "10"),
        (5 * 10**18 + 1, "5.000000000000000001"),
        (1234567890123456789, "1.234567890123456789"),
    ],
)
def test_wei_to_eth_is_exact(wei: int, expected: str) -> None:
    assert _wei_to_eth(wei) == expected


def _rec(checksum: str, uri: str, *, is_latest: bool) -> dict[str, Any]:
    return {"checksum": checksum, "uri": uri, "is_latest": is_latest}


def test_find_record_prefers_is_latest() -> None:
    records = [
        _rec("ab", "dgmlx://f/d", is_latest=False),
        _rec("ab", "dgmlx://f/d", is_latest=True),
        _rec("ab", "dgmlx://other", is_latest=True),  # wrong uri, excluded
    ]
    picked = _find_record(records, "ab", "dgmlx://f/d")
    assert picked is not None and picked["is_latest"] is True and picked["uri"] == "dgmlx://f/d"


def test_find_record_none_when_no_match() -> None:
    assert _find_record([_rec("zz", "dgmlx://f", is_latest=True)], "ab", "dgmlx://f") is None


class _FakeAnchor:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records

    def get_records(self, registry: str, *, checksum: str = "", **_: Any) -> list[dict[str, Any]]:
        return self._records


def test_fetch_record_prefers_is_latest() -> None:
    anchor = _FakeAnchor(
        [
            _rec("ab", "dgmlx://f", is_latest=False),
            _rec("ab", "dgmlx://f", is_latest=True),
        ]
    )
    assert _fetch_record(anchor, "reg", "ab")["is_latest"] is True  # type: ignore[arg-type]


def test_fetch_record_raises_when_empty() -> None:
    with pytest.raises(RecordNotFound):
        _fetch_record(_FakeAnchor([]), "reg", "ab")  # type: ignore[arg-type]


class _FakeReceiptRpc:
    def __init__(self, receipt: dict[str, Any]) -> None:
        self._receipt = receipt

    def get_transaction_receipt(self, tx_hash: str) -> dict[str, Any]:
        return self._receipt


@pytest.mark.parametrize("status", ["0x1", "0x01", 1])
def test_wait_for_receipt_success(status: Any) -> None:
    rpc = _FakeReceiptRpc({"status": status, "blockNumber": "0x5"})
    assert _wait_for_receipt(rpc, "0xabc")["blockNumber"] == "0x5"  # type: ignore[arg-type]


def test_wait_for_receipt_accepts_missing_status() -> None:
    # A mined receipt with no status field is accepted, not treated as a revert.
    rpc = _FakeReceiptRpc({"blockNumber": "0x5"})
    assert _wait_for_receipt(rpc, "0xabc")["blockNumber"] == "0x5"  # type: ignore[arg-type]


@pytest.mark.parametrize("status", ["0x0", "0x00", 0])
def test_wait_for_receipt_reverts_on_zero(status: Any) -> None:
    rpc = _FakeReceiptRpc({"status": status})
    with pytest.raises(ChainTxReverted):
        _wait_for_receipt(rpc, "0xabc")  # type: ignore[arg-type]


def test_finish_stake_writes_named_record(tmp_path: Any) -> None:
    chain = ChainConfig(name="t", rpc_url="http://x", chain_id=1)
    uri, checksum = "dgmlx://f/d#1", "abc"
    record = {"checksum": checksum, "uri": uri, "is_latest": True}

    class _Rpc:
        def send_raw_transaction(self, signed_tx_hex: str) -> str:
            return "0xtx"

        def get_transaction_receipt(self, tx_hash: str) -> dict[str, Any]:
            return {"status": "0x1", "blockNumber": "0x1"}

    class _Anchor:
        def get_records(
            self, registry: str, *, checksum: str = "", **_: Any
        ) -> list[dict[str, Any]]:
            return [record]

    signed = {"from": "0x1", "signed_tx": "0x02", "tx_hash": "0xtx"}
    save_dir = tmp_path / "bundle"
    out = _finish_stake(
        _Rpc(),  # type: ignore[arg-type]
        _Anchor(),  # type: ignore[arg-type]
        chain,
        signed,
        {"kind": "dgml-node"},
        "reg",
        checksum,
        uri,
        save_dir,
        "record-node-1.json",
    )
    assert (save_dir / "record-node-1.json").exists()
    assert not (save_dir / "record.json").exists()
    assert out["record_path"].endswith("record-node-1.json")
    assert out["receipt_status"] == "success"
