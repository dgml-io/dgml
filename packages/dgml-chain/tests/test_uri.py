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

import json

import pytest
from dgml_chain.uri import (
    CHECKSUM_ALGO,
    build_node_uri,
    build_uri,
    bundle_metadata,
    node_metadata,
    parse_uri,
)


def test_build_and_parse_file_uri() -> None:
    assert build_uri("f00000", None) == "dgmlx://f00000"
    assert build_uri("f00000", "ds00000") == "dgmlx://f00000/ds00000"
    assert parse_uri("dgmlx://f00000") == {
        "file_id": "f00000",
        "docset_id": None,
        "leaf_index": None,
    }
    assert parse_uri("dgmlx://f00000/ds00000") == {
        "file_id": "f00000",
        "docset_id": "ds00000",
        "leaf_index": None,
    }


def test_build_and_parse_node_uri() -> None:
    assert build_node_uri("f00000", "ds00000", 7) == "dgmlx://f00000/ds00000#7"
    assert parse_uri("dgmlx://f00000/ds00000#7") == {
        "file_id": "f00000",
        "docset_id": "ds00000",
        "leaf_index": 7,
    }


def test_parse_uri_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_uri("http://example.com")


def test_parse_uri_rejects_node_fragment_without_docset() -> None:
    with pytest.raises(ValueError):
        parse_uri("dgmlx://f00000#3")


def test_metadata_is_compact_json() -> None:
    bm = bundle_metadata(5)
    assert json.loads(bm) == {"kind": "dgmlx", "slots": 5}
    assert " " not in bm  # compact separators

    proof = {"leaf_index": 2, "leaf_hash": "ab", "path": []}
    nm = node_metadata("rootbeef", proof)
    parsed = json.loads(nm)
    assert parsed == {"kind": "dgml-node", "root_hash": "rootbeef", "proof": proof}


def test_checksum_algo_pinned() -> None:
    assert CHECKSUM_ALGO == "sha256"
