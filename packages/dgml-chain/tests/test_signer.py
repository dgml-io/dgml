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

import pytest
from dgml_chain import signer
from eth_account import Account

# A well-known Ganache test key — never used on any real chain. The address is
# derived so the pairing can never drift.
_KEY = "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d"
_ADDR = Account.from_key(_KEY).address

_UNSIGNED: dict[str, Any] = {
    "chain_id": 787111,
    "nonce": 0,
    "gas": 120000,
    "to": "0x0000000000000000000000000000000000000A00",
    "value": 0,
    "data": "0xabcd",
    "type": 2,
    "max_fee_per_gas": 20_000_000_000,
    "max_priority_fee_per_gas": 5_000_000_000,
}


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signer, "load_key", lambda service="", account="": _KEY)


def test_sign_tx_returns_address_and_raw() -> None:
    out = signer.sign_tx(_UNSIGNED, expected_from=_ADDR)
    assert out["from"].lower() == _ADDR.lower()
    assert out["signed_tx"].startswith("0x")
    assert out["tx_hash"].startswith("0x")
    # Type-2 signed payloads start with the 0x02 envelope byte.
    assert out["signed_tx"].startswith("0x02")


def test_sign_tx_refuses_mismatched_from() -> None:
    with pytest.raises(ValueError, match="refusing to sign"):
        signer.sign_tx(_UNSIGNED, expected_from="0x" + "11" * 20)


def test_legacy_tx_signs() -> None:
    legacy = {
        "chain_id": 1337,
        "nonce": 1,
        "gas": 21000,
        "to": _ADDR,
        "value": 0,
        "data": "0x",
        "type": 0,
        "gas_price": 10_000_000_000,
    }
    out = signer.sign_tx(legacy)
    assert out["from"].lower() == _ADDR.lower()


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signer, "load_key", lambda service="", account="": None)
    with pytest.raises(ValueError, match="no key in keyring"):
        signer.sign_tx(_UNSIGNED)
