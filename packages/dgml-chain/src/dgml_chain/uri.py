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

"""The ``dgmlx://`` URI scheme and on-chain record metadata helpers.

The workspace is local, so an anchored record's URI does not point at a
fetchable location — it encodes *which* workspace objects were attested so a
later proving step can parse them back out::

    dgmlx://<file_id>                     file-side artifacts only
    dgmlx://<file_id>/<docset_id>         docset-scoped (schema + DGML XML)
    dgmlx://<file_id>/<docset_id>#<leaf>  ONE element of the DGML XML
                                          (node-level; <leaf> is the 0-based
                                          Merkle leaf index)

IDs are the dgml CLI's lowercase alphanumeric identifiers. This module is the
single source of truth for the scheme and the record-metadata shape; both the
staking and proving halves import it so they cannot drift.
"""

from __future__ import annotations

import json
import re
from typing import Any

# The checksum algorithm pinned across the whole attestation stack: the
# manifest's `algorithm` attribute, the on-chain `checksumAlgo`, and the
# RFC-6962 Merkle leaf hashes are all SHA-256.
CHECKSUM_ALGO = "sha256"

_URI_RE = re.compile(r"^dgmlx://([a-z0-9]+)(?:/([a-z0-9]+))?(?:#(\d+))?$")


def build_uri(file_id: str, docset_id: str | None) -> str:
    """Build a bundle URI for a file (optionally docset-scoped)."""
    return f"dgmlx://{file_id}/{docset_id}" if docset_id else f"dgmlx://{file_id}"


def build_node_uri(file_id: str, docset_id: str, leaf_index: int) -> str:
    """Build a node URI naming one DGML XML element by its Merkle leaf index."""
    return f"dgmlx://{file_id}/{docset_id}#{leaf_index}"


def parse_uri(uri: str) -> dict[str, Any]:
    """Invert the scheme into ``{file_id, docset_id, leaf_index}``.

    ``docset_id`` and ``leaf_index`` are ``None`` when absent. Raises
    ``ValueError`` on a malformed URI or a ``#leaf`` fragment without a docset.
    """
    m = _URI_RE.match(uri)
    if m is None:
        raise ValueError(
            f"not a dgmlx URI: {uri!r} (expected dgmlx://<file_id>[/<docset_id>][#<leaf>])"
        )
    leaf = int(m.group(3)) if m.group(3) is not None else None
    if leaf is not None and m.group(2) is None:
        raise ValueError(f"node URI {uri!r} has a #leaf fragment but no docset id")
    return {"file_id": m.group(1), "docset_id": m.group(2), "leaf_index": leaf}


def bundle_metadata(slot_count: int) -> str:
    """Compact JSON metadata for a bundle (DGMLX) record."""
    return json.dumps({"kind": "dgmlx", "slots": slot_count}, separators=(",", ":"))


def node_metadata(root_hash: str, proof: dict[str, Any]) -> str:
    """Compact JSON metadata for a node record.

    Carries the document tree's Merkle root and the RFC 6962 inclusion path so
    a verifier can re-prove the node against the tree without the other nodes.
    """
    return json.dumps(
        {"kind": "dgml-node", "root_hash": root_hash, "proof": proof},
        separators=(",", ":"),
    )
