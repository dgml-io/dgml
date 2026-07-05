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

"""Direct EVM-chain anchoring for DGML artifacts.

Replaces the NVNM MCP server with direct, in-process chain communication: a
stdlib JSON-RPC client, anchor-precompile ABI encode/decode, EIP-1559
transaction building, and keyring-backed signing. Works against NVNM Chain and
any EVM chain exposing a compatible anchor precompile (see ``ChainStore``).

The Merkle/hashing/canonicalization that produces the anchored checksums lives
in the core ``dgml`` package; this package owns only the chain transport.
"""

from __future__ import annotations

from .anchor import (
    AnchorContract,
    encode_add_record,
    encode_add_registry,
    encode_grant_role,
)
from .chains import ANCHOR_PRECOMPILE, BUILTIN_CHAINS, ChainConfig, ChainStore
from .rpc import EvmRpc, RpcError
from .signer import keyring_address, sign_tx
from .tx import build_unsigned_tx
from .uri import (
    CHECKSUM_ALGO,
    build_node_uri,
    build_uri,
    bundle_metadata,
    node_metadata,
    parse_uri,
)

__all__ = [
    "ANCHOR_PRECOMPILE",
    "BUILTIN_CHAINS",
    "CHECKSUM_ALGO",
    "AnchorContract",
    "ChainConfig",
    "ChainStore",
    "EvmRpc",
    "RpcError",
    "build_node_uri",
    "build_unsigned_tx",
    "build_uri",
    "bundle_metadata",
    "encode_add_record",
    "encode_add_registry",
    "encode_grant_role",
    "keyring_address",
    "node_metadata",
    "parse_uri",
    "sign_tx",
]
