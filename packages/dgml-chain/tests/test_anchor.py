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

from dgml_chain.anchor import (
    AnchorContract,
    _output_types,
    encode_add_record,
    encode_add_registry,
    encode_grant_role,
)
from eth_abi import encode  # type: ignore[attr-defined]
from eth_utils import function_signature_to_4byte_selector  # type: ignore[attr-defined]

_ANCHOR = "0x0000000000000000000000000000000000000A00"


def test_selectors_match_canonical_signatures() -> None:
    # The 4-byte selector is the prefix of the encoded calldata.
    rec = encode_add_record(
        registry="r", uri="dgmlx://f", checksum="ab", checksum_algo="sha256", metadata="{}"
    )
    record_sig = "addRecord((string,string,string,string,string,string,string,uint64,uint64,bool))"
    assert rec.startswith("0x" + function_signature_to_4byte_selector(record_sig).hex())

    reg = encode_add_registry("name", "desc", "{}")
    assert reg.startswith(
        "0x" + function_signature_to_4byte_selector("addRegistry(string,string,string)").hex()
    )

    gr = encode_grant_role(1, "ab", "0x" + "11" * 20, "editor")
    assert gr.startswith(
        "0x" + function_signature_to_4byte_selector("grantRole(uint64,string,address,string)").hex()
    )


class FakeRpc:
    """Returns a pre-encoded ``records``/``registries`` view payload."""

    def __init__(self, payload_hex: str) -> None:
        self._payload = payload_hex

    def call(self, to: str, data: str, block: str = "latest") -> str:
        return self._payload


def _encode_records(records: list[tuple[object, ...]]) -> str:
    raw = encode(_output_types("records"), [records, (b"", len(records))])
    return "0x" + raw.hex()


def test_get_records_decodes_to_snake_case() -> None:
    rec = (
        "myreg",
        "dgmlx://f00000/ds00000",
        "deadbeef",
        "sha256",
        '{"kind":"dgmlx"}',
        "2026-06-24T00:00:00Z",
        "Active",
        3,
        0,
        True,
    )
    anchor = AnchorContract(FakeRpc(_encode_records([rec])), _ANCHOR)  # type: ignore[arg-type]
    out = anchor.get_records("myreg", checksum="deadbeef")
    assert len(out) == 1
    r = out[0]
    assert r["checksum"] == "deadbeef"
    assert r["checksum_algo"] == "sha256"
    assert r["uri"] == "dgmlx://f00000/ds00000"
    assert r["record_id"] == 3
    assert r["is_latest"] is True


def test_get_records_empty() -> None:
    anchor = AnchorContract(FakeRpc(_encode_records([])), _ANCHOR)  # type: ignore[arg-type]
    assert anchor.get_records("myreg") == []
