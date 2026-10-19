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

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dgml_chain import ChainConfig
from dgml_core import staking
from dgml_core.errors import ChainConfigError, ChainTxReverted, InvalidArgument, RecordNotFound
from dgml_core.staking import (
    _default_bundle_dir,
    _fetch_record,
    _find_record,
    _finish_stake,
    _wait_for_receipt,
    _wei_to_eth,
    chain_add,
    chain_list,
    resolve_chain_config_path,
    stake_node,
)
from dgml_core.storage import Workspace

from .conftest import scratch_configuration


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

    def get_records(
        self, registry_id: int, *, checksum: str = "", **_: Any
    ) -> list[dict[str, Any]]:
        return self._records


def test_fetch_record_prefers_is_latest() -> None:
    anchor = _FakeAnchor(
        [
            _rec("ab", "dgmlx://f", is_latest=False),
            _rec("ab", "dgmlx://f", is_latest=True),
        ]
    )
    picked = _fetch_record(anchor, 1, "ab", registry_name="reg")  # type: ignore[arg-type]
    assert picked["is_latest"] is True


def test_fetch_record_raises_when_empty() -> None:
    with pytest.raises(RecordNotFound):
        _fetch_record(_FakeAnchor([]), 1, "ab", registry_name="reg")  # type: ignore[arg-type]


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
            self, registry_id: int, *, checksum: str = "", **_: Any
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
        1,
        checksum,
        uri,
        save_dir,
        "record-node-1.json",
    )
    assert (save_dir / "record-node-1.json").exists()
    assert not (save_dir / "record.json").exists()
    assert out["record_path"].endswith("record-node-1.json")
    assert out["receipt_status"] == "success"


def test_default_bundle_dir_does_not_collide(tmp_path: Path) -> None:
    """Ids may contain hyphens (`file add --id`), so the default bundle dir
    cannot be a `<file>-<docset>` stem: that makes ("report-v2", "alpha")
    and ("report", "v2-alpha") the same directory."""
    ws = Workspace(root=tmp_path / "ws")
    a = _default_bundle_dir(ws, "report-v2", "alpha")
    b = _default_bundle_dir(ws, "report", "v2-alpha")
    assert a != b
    # Without a docset the path stays a single segment under the bundles dir.
    assert _default_bundle_dir(ws, "report-v2", None) == ws.root / "dgmlx-bundles" / "report-v2"


def test_scratch_workspace_writes_nothing_lasting_under_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scratch root is removed with the workspace: the bundle (reported output)
    needs an explicit dir, and custom chains have nowhere to live but a named path."""
    ws = Workspace.open(configuration=scratch_configuration(monkeypatch))
    with pytest.raises(InvalidArgument, match="pass output_dir"):
        _default_bundle_dir(ws, "report", None)
    monkeypatch.delenv("DGML_CHAINS", raising=False)
    assert resolve_chain_config_path(ws, None) is None
    # Built-ins still resolve; adding a custom chain is refused, not written somewhere.
    assert chain_list(ws, None)["chains"]
    with pytest.raises(ChainConfigError, match="pass config_path"):
        chain_add(
            ws,
            name="mine",
            rpc_url="http://x",
            chain_id=1,
            anchor_address="0x" + "1" * 40,
            explorer=None,
            native_token=None,
            config_path=None,
        )
    # An explicit location still wins; the environment is not read on this path.
    assert resolve_chain_config_path(ws, tmp_path / "c.json") == tmp_path / "c.json"
    monkeypatch.setenv("DGML_CHAINS", str(tmp_path / "env.json"))
    assert resolve_chain_config_path(ws, None) is None
    assert resolve_chain_config_path(Workspace(root=tmp_path), None) == tmp_path / "env.json"


def test_stake_node_dry_run_needs_no_bundle_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run stops after signing and writes nothing, so a scratch root — which
    has no bundle dir to offer — is no obstacle. A broadcast still needs one up front."""
    ws = Workspace.open(configuration=scratch_configuration(monkeypatch))
    chain = ChainConfig(name="t", rpc_url="http://x", chain_id=1)
    att = SimpleNamespace(
        leaf_index=1, leaf_count=2, xpath="/a", root_hash="r", node_hash="n", proof=[]
    )
    signed = {"from": "0x1", "signed_tx": "0x02", "tx_hash": "0xtx"}
    monkeypatch.setattr(staking, "get_chain", lambda *_: chain)
    monkeypatch.setattr(staking, "_resolve_from", lambda *_: "0x1")
    monkeypatch.setattr(staking, "_clients", lambda _: (object(), object()))
    monkeypatch.setattr(staking, "export_node", lambda *a, **k: att)
    monkeypatch.setattr(staking, "_resolve_registry_id", lambda *_: 7)
    monkeypatch.setattr(staking, "_prepare_and_sign", lambda *a, **k: ({"nonce": 0}, signed))
    monkeypatch.setattr(staking, "proof_to_json", lambda _: {})
    monkeypatch.setattr(staking, "node_metadata", lambda *_: "{}")
    common: dict[str, Any] = dict(
        file_id="f",
        docset_id="d",
        leaf_index=1,
        xpath=None,
        chain_name="t",
        registry="reg",
        from_address=None,
        output_dir=None,
        config_path=None,
        legacy=False,
    )
    out = stake_node(ws, dry_run=True, **common)
    assert out["broadcast"] is False and out["signed_tx"] == "0x02"
    assert list(ws.root.iterdir()) == []
    with pytest.raises(InvalidArgument, match="pass output_dir"):
        stake_node(ws, dry_run=False, **common)
