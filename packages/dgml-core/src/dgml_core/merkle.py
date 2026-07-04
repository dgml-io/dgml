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

"""Merkle attestation for DGML XML trees.

Stake a DGML tree (or any subtree, down to a single element) to a blockchain
by publishing its Merkle root. Later, prove a specific element was in the
staked tree using ``(element, proof, root_hash)``.

Algorithm
---------
- ``leaf_hash(elem) = sha256(c14n_exclusive(elem))`` — lowercase, unprefixed hex.
- Leaves enumerated in DFS pre-order over every XML element. Text nodes are
  *not* separate leaves; their content is absorbed into the parent element's
  canonicalization.
- Binary Merkle tree built bottom-up: pair-wise SHA-256 over the *raw bytes*
  of the two child hashes (``sha256(bytes.fromhex(L) + bytes.fromhex(R))``).
- Odd-out convention: **RFC 6962** — a lone unpaired node at any level
  promotes to the next level unchanged (no self-duplication). Avoids the
  CVE-2012-2459 second-preimage weakness present in the Bitcoin variant.
- No domain separation between leaf and internal hashes — the user-facing
  identity ``canonical_hash(elem) == leaf_hash(elem)`` is preserved. Callers
  should publish ``leaf_count`` alongside the root so verifiers can detect
  same-root / different-leaf-count constructions.

Sub-tree staking is supported directly: because exclusive c14n is independent
of inherited xmlns context, passing any inner element to :func:`merkle_root`
produces the same hash as parsing that subtree as a standalone document.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from lxml import etree  # type: ignore[import-untyped]
from lxml.etree import _Element  # type: ignore[import-untyped]

HASH_HEX_LEN: int = 64
Side = Literal["L", "R"]


@dataclass(frozen=True)
class ProofStep:
    """One step on the path from a leaf to the Merkle root.

    ``sibling`` is the hex SHA-256 of the node on the opposite side of the
    current node at that level. ``side`` tells the verifier which side the
    *sibling* sits on:

    - ``"L"`` — sibling is to the left; next hash is ``H(sibling || current)``.
    - ``"R"`` — sibling is to the right; next hash is ``H(current || sibling)``.
    """

    sibling: str
    side: Side


@dataclass(frozen=True)
class MerkleProof:
    """An inclusion proof for a single element in a staked tree.

    ``leaf_hash`` matches :func:`canonical_hash` of the element being proven.
    ``leaf_index`` is the 0-based DFS pre-order index; carried for
    transparency and debugging — verification itself only needs ``path``.
    ``leaf_count`` is the total number of elements in the staked tree;
    publishing it alongside the root defends against attacks that exploit
    the lack of leaf-vs-internal-node domain separation.
    """

    leaf_hash: str
    leaf_index: int
    leaf_count: int
    path: list[ProofStep]


@dataclass(frozen=True)
class MerkleTree:
    """Full diagnostic view of the Merkle tree.

    ``levels[0]`` is the leaf list (same as :func:`merkle_leaves`); each
    subsequent level is the bottom-up pair-wise reduction; ``levels[-1]``
    is always a single-element list containing the root hash.
    """

    leaves: list[str]
    levels: list[list[str]]
    root: str


def canonical_hash(element: _Element) -> str:
    """SHA-256 of the Exclusive C14N canonicalization of ``element``.

    Uses XML C14N 1.0 with ``exclusive=True`` (NOT C14N 2.0, which has no
    exclusive mode). Comments are stripped so the hash is deterministic
    regardless of whether comments are present.
    """
    blob: bytes = etree.tostring(
        element,
        method="c14n",
        exclusive=True,
        with_comments=False,
    )
    return hashlib.sha256(blob).hexdigest()


def merkle_leaves(root: _Element) -> list[str]:
    """Ordered leaf hash list: ``canonical_hash`` of every element in DFS pre-order.

    ``root.iter()`` walks the tree in document (pre-order) order — the same
    traversal the rest of the package uses for XML iteration. The
    ``isinstance(el.tag, str)`` guard defensively skips lxml's special
    comment / processing-instruction node objects (their ``.tag`` is a
    callable). DGML doesn't emit comments or PIs in final output, but
    callers feeding hand-authored XML get correct behaviour without
    surprises.
    """
    return [canonical_hash(el) for el in root.iter() if isinstance(el.tag, str)]


def merkle_root(root: _Element) -> str:
    """The Merkle root hash for the tree rooted at ``root``.

    For a single-element tree, this equals ``canonical_hash(root)`` — no
    pairing is performed. Raises :class:`ValueError` if ``root`` yields no
    element leaves (which is impossible for a well-formed XML element, but
    guarded for clarity).
    """
    leaves = merkle_leaves(root)
    if not leaves:
        raise ValueError("Cannot compute Merkle root of an empty element list")
    return merkle_root_from_hashes(leaves)


def merkle_root_from_hashes(leaves: list[str]) -> str:
    """RFC 6962 Merkle root over an arbitrary list of hex leaf hashes.

    The generic counterpart to :func:`merkle_root` — same algorithm
    (pair-wise SHA-256 over raw bytes, lone odd-out promotes unchanged),
    but takes the leaves as already-computed hex strings instead of
    deriving them from XML elements. Used by ``dgml.file_attestation``
    to roll a multi-artifact file version up to one root.

    Raises :class:`ValueError` if ``leaves`` is empty.
    """
    if not leaves:
        raise ValueError("Cannot compute Merkle root of an empty leaf list")
    return _root_from_leaves(leaves)


def merkle_proof(root: _Element, target: _Element) -> MerkleProof:
    """Build an inclusion proof for ``target`` within the tree rooted at ``root``.

    ``target`` must be the *same Python object* as one of the elements
    reachable from ``root.iter()`` — identity is compared with ``is``, not
    equality. Re-parsing the same XML produces new element objects with the
    same hashes but different Python identity; the realistic workflow is to
    parse once, then build all proofs you need from that single tree.
    """
    elements = [el for el in root.iter() if isinstance(el.tag, str)]
    try:
        idx = next(i for i, el in enumerate(elements) if el is target)
    except StopIteration as exc:
        raise ValueError(
            "target element is not a node of root (object-identity check failed)"
        ) from exc
    leaves = [canonical_hash(el) for el in elements]
    return MerkleProof(
        leaf_hash=leaves[idx],
        leaf_index=idx,
        leaf_count=len(leaves),
        path=_build_proof_path(leaves, idx),
    )


def verify_proof(root_hash: str, element: _Element, proof: MerkleProof) -> bool:
    """Verify that ``element`` was a member of the tree whose root is ``root_hash``.

    Returns ``True`` iff (a) the canonical hash of ``element`` matches the
    proof's claimed ``leaf_hash``, and (b) walking the sibling chain
    reproduces ``root_hash``. Returns ``False`` for any tamper — text edit,
    attribute change, structural mutation, wrong root, or doctored sibling
    hash. A *structurally* malformed proof (a ``ProofStep.side`` that is
    neither ``"L"`` nor ``"R"``) is not a verification outcome but bad
    input, and raises :class:`ValueError` rather than silently reporting
    non-membership.
    """
    candidate = canonical_hash(element)
    if candidate != proof.leaf_hash:
        return False
    current = candidate
    for step in proof.path:
        if step.side == "L":
            current = _hash_pair(step.sibling, current)
        elif step.side == "R":
            current = _hash_pair(current, step.sibling)
        else:
            # Structurally invalid proof (shouldn't happen via the typed API,
            # but reachable via the JSON-revival path users may add). This is
            # malformed input, not a failed verification — surface it loudly
            # rather than masquerading as a clean "not a member" result.
            raise ValueError(f"invalid proof step side: {step.side!r} (expected 'L' or 'R')")
    return current == root_hash


def proof_to_json(proof: MerkleProof) -> dict[str, object]:
    """Plain-data view of ``proof``, suitable for ``json.dumps``.

    Inverse of :func:`proof_from_json`. The shape is part of the public
    contract (CLI payloads and on-chain anchoring metadata embed it):
    ``{"leaf_hash", "leaf_index", "leaf_count", "path": [{"sibling", "side"}]}``.
    """
    return {
        "leaf_hash": proof.leaf_hash,
        "leaf_index": proof.leaf_index,
        "leaf_count": proof.leaf_count,
        "path": [{"sibling": s.sibling, "side": s.side} for s in proof.path],
    }


def proof_from_json(data: object) -> MerkleProof:
    """Revive a :class:`MerkleProof` from the :func:`proof_to_json` shape.

    Raises :class:`ValueError` on any structural problem — wrong types,
    non-hex or wrong-length hashes, a ``side`` that isn't ``"L"``/``"R"``,
    or an inconsistent ``leaf_index``/``leaf_count`` pair. Validating here
    keeps :func:`verify_proof` free to treat its input as well-formed.
    """
    if not isinstance(data, dict):
        raise ValueError(f"proof must be a JSON object, got {type(data).__name__}")
    leaf_hash = _require_hash(data.get("leaf_hash"), "leaf_hash")
    leaf_index = data.get("leaf_index")
    leaf_count = data.get("leaf_count")
    if not isinstance(leaf_index, int) or isinstance(leaf_index, bool) or leaf_index < 0:
        raise ValueError(f"leaf_index must be a non-negative integer, got {leaf_index!r}")
    if not isinstance(leaf_count, int) or isinstance(leaf_count, bool) or leaf_count < 1:
        raise ValueError(f"leaf_count must be a positive integer, got {leaf_count!r}")
    if leaf_index >= leaf_count:
        raise ValueError(f"leaf_index {leaf_index} out of range for leaf_count {leaf_count}")
    raw_path = data.get("path")
    if not isinstance(raw_path, list):
        raise ValueError("path must be a list of proof steps")
    steps: list[ProofStep] = []
    for i, raw in enumerate(raw_path):
        if not isinstance(raw, dict):
            raise ValueError(f"path[{i}] must be a JSON object")
        side = raw.get("side")
        if side not in ("L", "R"):
            raise ValueError(f"path[{i}].side must be 'L' or 'R', got {side!r}")
        steps.append(ProofStep(_require_hash(raw.get("sibling"), f"path[{i}].sibling"), side))
    return MerkleProof(
        leaf_hash=leaf_hash, leaf_index=leaf_index, leaf_count=leaf_count, path=steps
    )


def merkle_tree(root: _Element) -> MerkleTree:
    """Compute the full Merkle tree — leaves, every intermediate level, and the root.

    Useful for diagnostics, batch-proof generation, and documenting an
    attestation. Costs the same as :func:`merkle_root` plus the memory of
    storing intermediate levels.
    """
    leaves = merkle_leaves(root)
    if not leaves:
        raise ValueError("Cannot build a Merkle tree from an empty element list")
    levels = _build_levels(leaves)
    return MerkleTree(leaves=leaves, levels=levels, root=levels[-1][0])


# --- private helpers ---------------------------------------------------------


def _hash_pair(left_hex: str, right_hex: str) -> str:
    """SHA-256 over the *raw bytes* of two hex hashes — never their hex text."""
    return hashlib.sha256(bytes.fromhex(left_hex) + bytes.fromhex(right_hex)).hexdigest()


_HASH_RE = re.compile(rf"^[0-9a-f]{{{HASH_HEX_LEN}}}$")


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.match(value) is None:
        raise ValueError(f"{label} must be a {HASH_HEX_LEN}-char lowercase hex hash")
    return value


def _pair_level(level: list[str]) -> list[str]:
    """One bottom-up reduction with RFC 6962 promote-lone-node semantics."""
    out: list[str] = []
    i = 0
    while i + 1 < len(level):
        out.append(_hash_pair(level[i], level[i + 1]))
        i += 2
    if i < len(level):
        # Lone odd-out promotes unchanged.
        out.append(level[i])
    return out


def _root_from_leaves(leaves: list[str]) -> str:
    level = leaves
    while len(level) > 1:
        level = _pair_level(level)
    return level[0]


def _build_levels(leaves: list[str]) -> list[list[str]]:
    levels: list[list[str]] = [leaves]
    while len(levels[-1]) > 1:
        levels.append(_pair_level(levels[-1]))
    return levels


def _build_proof_path(leaves: list[str], idx: int) -> list[ProofStep]:
    """Bottom-up authentication path for the leaf at ``idx``.

    At each level: if our node has a sibling, emit a :class:`ProofStep`
    naming it; if our node is the lone odd-out, skip emitting (it promotes
    unchanged, so the verifier needs nothing for this level).
    """
    path: list[ProofStep] = []
    level = leaves
    pos = idx
    while len(level) > 1:
        if pos % 2 == 1:
            # Right-side node — its partner is the left sibling.
            path.append(ProofStep(sibling=level[pos - 1], side="L"))
        elif pos + 1 < len(level):
            # Left-side node with a partner on the right.
            path.append(ProofStep(sibling=level[pos + 1], side="R"))
        # else: lone odd-out at this level; no step emitted.
        level = _pair_level(level)
        pos //= 2
    return path
