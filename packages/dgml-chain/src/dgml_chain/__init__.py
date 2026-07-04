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
