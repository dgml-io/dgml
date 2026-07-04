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

"""Transaction signing with a key held in the OS keyring.

The private key never leaves this process and is never printed. It lives in the
OS keyring (default service ``nvnm-wallet``, account ``default``) and is read
only at signing time. ``sign_tx`` refuses to sign if the keyring key does not
control the transaction's intended ``from`` address.
"""

from __future__ import annotations

import sys
from typing import Any

DEFAULT_SERVICE = "nvnm-wallet"
DEFAULT_ACCOUNT = "default"


def _get_keyring() -> Any:
    """Return the keyring backend.

    On macOS, use the Keychain backend directly: a ``keyringrc.cfg`` can force
    the 'fail' backend globally (one has been found doing exactly that), and
    auto-detection honors it. Everywhere else, defer to detection.
    """
    import keyring

    if sys.platform == "darwin":
        from keyring.backends import macOS

        return macOS.Keyring()  # type: ignore[no-untyped-call]
    return keyring.get_keyring()


def load_key(service: str = DEFAULT_SERVICE, account: str = DEFAULT_ACCOUNT) -> str | None:
    key: str | None = _get_keyring().get_password(service, account)
    return key


def keyring_address(service: str = DEFAULT_SERVICE, account: str = DEFAULT_ACCOUNT) -> str | None:
    """The EVM address controlled by the keyring key, or ``None`` if unset.

    Used to default ``--from`` so callers need not repeat their own address.
    """
    from eth_account import Account

    key = load_key(service, account)
    if not key:
        return None
    return str(Account.from_key(key).address)


def _eth_account_tx(unsigned: dict[str, Any]) -> dict[str, Any]:
    """Translate an unsigned-tx dict (from ``tx.build_unsigned_tx``) to the
    camelCase shape ``eth_account`` expects."""
    tx: dict[str, Any] = {
        "chainId": int(unsigned["chain_id"]),
        "nonce": int(unsigned["nonce"]),
        "gas": int(unsigned["gas"]),
        "to": unsigned["to"],
        "value": int(unsigned.get("value", 0)),
        "data": unsigned["data"],
    }
    if int(unsigned.get("type", 2)) == 2:
        tx["type"] = 2
        tx["maxFeePerGas"] = int(unsigned["max_fee_per_gas"])
        tx["maxPriorityFeePerGas"] = int(unsigned["max_priority_fee_per_gas"])
    else:
        tx["gasPrice"] = int(unsigned["gas_price"])
    return tx


def sign_tx(
    unsigned: dict[str, Any],
    *,
    service: str = DEFAULT_SERVICE,
    account: str = DEFAULT_ACCOUNT,
    expected_from: str | None = None,
) -> dict[str, Any]:
    """Sign an unsigned transaction with the keyring key.

    Returns ``{from, signed_tx, tx_hash}``. Raises ``ValueError`` if no key is
    found or if the key does not control ``expected_from``.
    """
    from eth_account import Account

    key = load_key(service, account)
    if not key:
        raise ValueError(f"no key in keyring under service={service!r} account={account!r}")
    acct = Account.from_key(key)

    if expected_from and expected_from.lower() != acct.address.lower():
        raise ValueError(
            f"keyring key controls {acct.address} but the transaction was "
            f"prepared for {expected_from} — refusing to sign"
        )

    signed = acct.sign_transaction(_eth_account_tx(unsigned))
    raw = getattr(signed, "raw_transaction", None)
    if raw is None:  # older eth-account
        raw = signed.rawTransaction
    return {
        "from": acct.address,
        "signed_tx": "0x" + bytes(raw).hex(),
        "tx_hash": "0x" + bytes(signed.hash).hex(),
    }
