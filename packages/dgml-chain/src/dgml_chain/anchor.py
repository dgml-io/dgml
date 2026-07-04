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

"""Anchor-precompile ABI: encode write calldata, decode view results.

The contract surface (vendored ``abi/anchoring.json``) is:

- ``addRegistry(name, description, metadata) -> registryId``
- ``addRecord(record) -> recordId`` — ``record`` is a 10-field tuple
- ``grantRole(registryId, checksum, account, role)``
- ``records(registry, checksum, recordId, index, pagination)`` (view)
- ``registries(registryId, name, pagination)`` (view)

The canonical type strings used by ``eth_abi`` are derived from the ABI JSON so
they cannot drift from the vendored contract. ``records()`` is keyed by
registry **name** (the precompile does not accept a registry id there).
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from eth_abi import decode, encode  # type: ignore[attr-defined]
from eth_utils import function_signature_to_4byte_selector  # type: ignore[attr-defined]

from .rpc import EvmRpc

# Pagination input the precompile expects: (key, offset, limit, countTotal,
# reverse). Defaults select the natural page.
_DEFAULT_PAGINATION = (b"", 0, 0, False, False)
# A record's mutable-on-chain fields are server-assigned; the writer sends
# empty/zero placeholders for them.
_DEFAULT_STATUS = "Active"


def _load_abi() -> list[dict[str, Any]]:
    raw = files("dgml_chain").joinpath("abi", "anchoring.json").read_text(encoding="utf-8")
    abi: list[dict[str, Any]] = json.loads(raw)
    return abi


_ABI = _load_abi()
_FUNCS: dict[str, dict[str, Any]] = {f["name"]: f for f in _ABI if f.get("type") == "function"}


def _canonical_type(item: dict[str, Any]) -> str:
    """Build an eth_abi type string from an ABI input/output item.

    Recurses into tuples and preserves array suffixes (e.g. ``tuple[]`` →
    ``(...)[]``).
    """
    t = str(item["type"])
    if t.startswith("tuple"):
        inner = ",".join(_canonical_type(c) for c in item["components"])
        suffix = t[len("tuple") :]  # "", "[]", "[N]"
        return f"({inner}){suffix}"
    return t


def _signature(name: str) -> str:
    fn = _FUNCS[name]
    return f"{name}({','.join(_canonical_type(i) for i in fn['inputs'])})"


def _selector(name: str) -> bytes:
    return function_signature_to_4byte_selector(_signature(name))


def _input_types(name: str) -> list[str]:
    return [_canonical_type(i) for i in _FUNCS[name]["inputs"]]


def _output_types(name: str) -> list[str]:
    return [_canonical_type(o) for o in _FUNCS[name]["outputs"]]


def _encode_call(name: str, args: list[Any]) -> str:
    data = _selector(name) + encode(_input_types(name), args)
    return "0x" + data.hex()


_RECORD_FIELDS = (
    "registry",
    "uri",
    "checksum",
    "checksum_algo",
    "metadata",
    "timestamp",
    "status",
    "record_id",
    "index",
    "is_latest",
)


def _record_to_dict(values: tuple[Any, ...]) -> dict[str, Any]:
    """Map a decoded 10-field record tuple to snake_case keys.

    The ``checksum_algo`` key matches the proving side (the on-chain field is
    ``checksumAlgo``); ``record_id``/``is_latest`` likewise.
    """
    return dict(zip(_RECORD_FIELDS, values, strict=True))


_REGISTRY_FIELDS = ("id", "name", "description", "creator", "created_at", "metadata")


def _registry_to_dict(values: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(_REGISTRY_FIELDS, values, strict=True))


# --- write calldata (no network) --------------------------------------------


def encode_add_registry(name: str, description: str, metadata: str) -> str:
    return _encode_call("addRegistry", [name, description, metadata])


def encode_add_record(
    *,
    registry: str,
    uri: str,
    checksum: str,
    checksum_algo: str,
    metadata: str,
    status: str = _DEFAULT_STATUS,
) -> str:
    record = (
        registry,
        uri,
        checksum,
        checksum_algo,
        metadata,
        "",  # timestamp — on-chain fills
        status,
        0,  # recordId — on-chain assigns
        0,  # index
        False,  # isLatest
    )
    return _encode_call("addRecord", [record])


def encode_grant_role(registry_id: int, checksum: str, account: str, role: str) -> str:
    return _encode_call("grantRole", [registry_id, checksum, account, role])


# --- views (read via eth_call) ----------------------------------------------


class AnchorContract:
    """Read the anchor precompile at ``address`` over an ``EvmRpc``."""

    def __init__(self, rpc: EvmRpc, address: str) -> None:
        self.rpc = rpc
        self.address = address

    def _call_view(self, name: str, args: list[Any]) -> tuple[Any, ...]:
        data = _encode_call(name, args)
        result = self.rpc.call(self.address, data)
        raw = bytes.fromhex(result[2:] if result.startswith("0x") else result)
        return tuple(decode(_output_types(name), raw))

    def get_records(
        self,
        registry: str,
        *,
        checksum: str = "",
        record_id: int = 0,
        index: int = 0,
    ) -> list[dict[str, Any]]:
        records, _pagination = self._call_view(
            "records", [registry, checksum, record_id, index, _DEFAULT_PAGINATION]
        )
        return [_record_to_dict(r) for r in records]

    def get_registries(self, *, registry_id: int = 0, name: str = "") -> list[dict[str, Any]]:
        registries, _pagination = self._call_view(
            "registries", [registry_id, name, _DEFAULT_PAGINATION]
        )
        return [_registry_to_dict(r) for r in registries]
