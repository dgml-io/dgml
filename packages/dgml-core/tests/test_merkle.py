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

"""Tests for ``dgml_core.merkle`` — canonical hashing + RFC 6962 Merkle attestation.

XML fixtures are inline strings parsed with ``lxml.etree.fromstring``, matching
the existing pattern in ``test_semantic_layout.py``. lxml is a base dependency,
so no ``importorskip`` guard is needed.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from dgml_core.merkle import (
    HASH_HEX_LEN,
    MerkleProof,
    ProofStep,
    canonical_hash,
    merkle_leaves,
    merkle_proof,
    merkle_root,
    merkle_tree,
    proof_from_json,
    proof_to_json,
    verify_proof,
)
from lxml import etree  # type: ignore[import-untyped]


def _h(left_hex: str, right_hex: str) -> str:
    """Mirror of the module-private ``_hash_pair`` for hand-computed expectations."""
    return hashlib.sha256(bytes.fromhex(left_hex) + bytes.fromhex(right_hex)).hexdigest()


# --- canonical_hash ----------------------------------------------------------


def test_canonical_hash_is_sha256_of_exclusive_c14n_bytes() -> None:
    """Pins the spec identity: leaf_hash == sha256(exclusive_c14n(element))."""
    el = etree.fromstring(b'<x a="1"><y>hello</y></x>')
    expected_bytes = etree.tostring(el, method="c14n", exclusive=True, with_comments=False)
    expected = hashlib.sha256(expected_bytes).hexdigest()
    assert canonical_hash(el) == expected
    assert len(canonical_hash(el)) == HASH_HEX_LEN


def test_canonical_hash_attribute_order_independent() -> None:
    """C14N normalizes attribute order — serialization swaps shouldn't change the hash."""
    a = etree.fromstring(b'<x a="1" b="2" c="3"/>')
    b = etree.fromstring(b'<x c="3" a="1" b="2"/>')
    assert canonical_hash(a) == canonical_hash(b)


def test_canonical_hash_prefix_names_are_significant() -> None:
    """Exclusive C14N preserves prefix names — renaming them changes the hash.

    Exclusive C14N's job is to scope namespace declarations to the subtree
    where they're used; it does *not* rewrite prefix labels. Two elements
    with the same content under different prefix labels are different
    canonical XML and therefore different hashes. Document authors should
    keep prefix labels stable across attestations.
    """
    a = etree.fromstring(b'<dg:x xmlns:dg="http://x"><dg:y>1</dg:y></dg:x>')
    b = etree.fromstring(b'<q:x xmlns:q="http://x"><q:y>1</q:y></q:x>')
    assert canonical_hash(a) != canonical_hash(b)


def test_canonical_hash_inherited_namespace_declaration_independent() -> None:
    """Exclusive C14N's real promise: an inner subtree's hash is independent of
    *where* its xmlns is declared (root vs subtree) as long as the prefix label
    and URI are unchanged."""
    full = etree.fromstring(b'<r xmlns:dg="http://x"><dg:inner><dg:leaf>v</dg:leaf></dg:inner></r>')
    inner = full[0]
    standalone = etree.fromstring(b'<dg:inner xmlns:dg="http://x"><dg:leaf>v</dg:leaf></dg:inner>')
    assert canonical_hash(inner) == canonical_hash(standalone)


def test_canonical_hash_text_change_changes_hash() -> None:
    a = etree.fromstring(b"<x>hello</x>")
    b = etree.fromstring(b"<x>hello </x>")
    assert canonical_hash(a) != canonical_hash(b)


def test_canonical_hash_attribute_value_change_changes_hash() -> None:
    a = etree.fromstring(b'<x a="1"/>')
    b = etree.fromstring(b'<x a="2"/>')
    assert canonical_hash(a) != canonical_hash(b)


# --- merkle_leaves -----------------------------------------------------------


def test_merkle_leaves_dfs_preorder() -> None:
    """Leaf list matches a hand-written pre-order enumeration."""
    root = etree.fromstring(b"<r><a><b/><c/></a><d/></r>")
    # Pre-order: r, a, b, c, d
    elements = [root, root[0], root[0][0], root[0][1], root[1]]
    assert merkle_leaves(root) == [canonical_hash(e) for e in elements]


def test_merkle_leaves_skips_comments_and_pis() -> None:
    """The ``isinstance(el.tag, str)`` guard excludes lxml comment / PI nodes."""
    root = etree.fromstring(b"<r><!-- skip me --><?pi data?><a/></r>")
    # Only <r> and <a> are real elements — 2 leaves.
    assert len(merkle_leaves(root)) == 2


def test_comments_are_fully_invisible_to_attestation() -> None:
    """``with_comments=False`` strips them from c14n bytes AND they're not leaves —
    adding or removing a comment does not change any hash."""
    a = etree.fromstring(b"<r><a/></r>")
    b = etree.fromstring(b"<r><!-- comment --><a/></r>")
    assert canonical_hash(a) == canonical_hash(b)
    assert merkle_root(a) == merkle_root(b)


def test_processing_instructions_affect_parent_hash_but_arent_leaves() -> None:
    """PIs are skipped as separate Merkle leaves, but lxml's c14n keeps PIs in
    the canonical bytes — so a PI contributes to its parent element's hash.

    This asymmetry with comments is documented in docs/merkle-attestation.md.
    """
    a = etree.fromstring(b"<r><a/></r>")
    b = etree.fromstring(b"<r><?pi data?><a/></r>")
    # Same leaf count: PI is not a Merkle leaf.
    assert len(merkle_leaves(a)) == len(merkle_leaves(b)) == 2
    # But the parent's hash differs because c14n keeps the PI inline.
    assert canonical_hash(a) != canonical_hash(b)
    assert merkle_root(a) != merkle_root(b)


# --- merkle_root -------------------------------------------------------------


def test_merkle_root_single_element() -> None:
    """Trivial case: a leaf-only tree has root == canonical_hash(root)."""
    el = etree.fromstring(b'<x a="1"/>')
    assert merkle_root(el) == canonical_hash(el)


def test_merkle_root_two_leaves_hand_computed() -> None:
    """Two-leaf tree: root == H(leaf0 || leaf1) with raw-byte concatenation."""
    root = etree.fromstring(b"<r><a/></r>")
    leaves = merkle_leaves(root)
    assert len(leaves) == 2
    expected = _h(leaves[0], leaves[1])
    assert merkle_root(root) == expected


def test_merkle_root_odd_count_promotes_lone_node_rfc6962() -> None:
    """3-leaf tree: L2 promotes unchanged to level 1, then pairs with H(L0,L1).

    Locks RFC 6962 over the Bitcoin convention; in Bitcoin's variant the
    expected root would be H(H(L0,L1), H(L2,L2)) — different value.
    """
    root = etree.fromstring(b"<r><a/><b/></r>")
    leaves = merkle_leaves(root)
    assert len(leaves) == 3
    # Level 1: [H(L0,L1), L2]  (L2 promoted)
    # Root:    H(H(L0,L1), L2)
    expected = _h(_h(leaves[0], leaves[1]), leaves[2])
    assert merkle_root(root) == expected


def test_merkle_root_deterministic_across_runs() -> None:
    root = etree.fromstring(b"<r><a>1</a><b>2</b><c><d/></c></r>")
    assert merkle_root(root) == merkle_root(root)


def test_merkle_root_raises_on_no_element_leaves() -> None:
    """Defensive guard — well-formed XML always has a root element, but lock the contract."""

    class _Fake:
        def iter(self) -> list[object]:  # pragma: no cover - simple stub
            return []

    with pytest.raises(ValueError):
        merkle_root(_Fake())


# --- merkle_tree -------------------------------------------------------------


def test_merkle_tree_consistent_with_root_and_leaves() -> None:
    root = etree.fromstring(b"<r><a/><b/><c/><d/><e/></r>")  # 6 leaves (r + 5 children)
    tree = merkle_tree(root)
    assert tree.leaves == merkle_leaves(root)
    assert tree.levels[0] == tree.leaves
    assert tree.levels[-1] == [tree.root]
    assert tree.root == merkle_root(root)


# --- merkle_proof / verify_proof roundtrip ----------------------------------


def _make_tree() -> etree._Element:
    return etree.fromstring(
        b"""
        <root>
          <header><title>Doc</title><date>2026-06-03</date></header>
          <body>
            <p>Paragraph one.</p>
            <p>Paragraph two.</p>
            <section>
              <h>Sub</h>
              <p>Inner.</p>
            </section>
          </body>
          <footer/>
        </root>
    """.strip()
    )


def test_proof_roundtrip_every_element() -> None:
    """Every element in a non-trivial tree can be proved and verified."""
    root = _make_tree()
    root_hash = merkle_root(root)
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        proof = merkle_proof(root, el)
        assert verify_proof(root_hash, el, proof), f"verify failed for <{el.tag}>"


def test_proof_for_root_element_itself() -> None:
    """The root element is leaf 0 — its proof should also verify."""
    root = _make_tree()
    proof = merkle_proof(root, root)
    assert proof.leaf_index == 0
    assert verify_proof(merkle_root(root), root, proof)


def test_proof_leaf_count_matches_total_leaves() -> None:
    root = _make_tree()
    leaves = merkle_leaves(root)
    proof = merkle_proof(root, root)
    assert proof.leaf_count == len(leaves)


# --- tamper detection --------------------------------------------------------


def test_verify_detects_text_tamper() -> None:
    root = _make_tree()
    root_hash = merkle_root(root)
    target = root.find("body/p")
    assert target is not None
    proof = merkle_proof(root, target)
    target.text = "Paragraph ONE."  # mutate after proof generation
    assert not verify_proof(root_hash, target, proof)


def test_verify_detects_attribute_tamper() -> None:
    root = etree.fromstring(b'<r><a k="v"/></r>')
    target = root[0]
    proof = merkle_proof(root, target)
    root_hash = merkle_root(root)
    target.set("k", "v2")
    assert not verify_proof(root_hash, target, proof)


def test_verify_detects_structural_tamper() -> None:
    """Adding a child to the verified element changes its c14n bytes."""
    root = _make_tree()
    root_hash = merkle_root(root)
    target = root.find("body/section")
    assert target is not None
    proof = merkle_proof(root, target)
    etree.SubElement(target, "extra").text = "injected"
    assert not verify_proof(root_hash, target, proof)


def test_verify_detects_wrong_root_hash() -> None:
    root = _make_tree()
    target = root.find("body/p")
    assert target is not None
    proof = merkle_proof(root, target)
    bad_root = "0" * HASH_HEX_LEN
    assert not verify_proof(bad_root, target, proof)


def test_verify_detects_tampered_sibling_in_path() -> None:
    root = _make_tree()
    root_hash = merkle_root(root)
    target = root.find("body/section/p")
    assert target is not None
    proof = merkle_proof(root, target)
    # Build a tampered proof: flip one byte of the first sibling.
    original = proof.path[0]
    tampered_step = ProofStep(sibling="ff" + original.sibling[2:], side=original.side)
    tampered = MerkleProof(
        leaf_hash=proof.leaf_hash,
        leaf_index=proof.leaf_index,
        leaf_count=proof.leaf_count,
        path=[tampered_step, *proof.path[1:]],
    )
    assert not verify_proof(root_hash, target, tampered)


def test_verify_rejects_invalid_side_in_proof() -> None:
    """A malformed ``side`` (via the JSON-revival path) is bad input, not a
    failed verification: it raises ``ValueError`` rather than returning False."""
    root = etree.fromstring(b"<r><a/><b/></r>")
    target = root[0]
    proof = merkle_proof(root, target)
    bad = MerkleProof(
        leaf_hash=proof.leaf_hash,
        leaf_index=proof.leaf_index,
        leaf_count=proof.leaf_count,
        path=[ProofStep(sibling=proof.path[0].sibling, side="X")],  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="invalid proof step side"):
        verify_proof(merkle_root(root), target, bad)


# --- error paths -------------------------------------------------------------


def test_merkle_proof_unknown_target_raises_value_error() -> None:
    root = etree.fromstring(b"<r><a/></r>")
    stranger = etree.fromstring(b"<a/>")  # different Python object, same hash
    with pytest.raises(ValueError, match="object-identity"):
        merkle_proof(root, stranger)


# --- subtree staking under multi-namespace documents ------------------------


def test_subtree_staking_is_namespace_independent() -> None:
    """Exclusive C14N guarantees inner subtree hashes are independent of outer xmlns context."""
    full = etree.fromstring(
        b"""
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
                  xmlns:docset="http://dgml.io/ns/dgml/test"
                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <docset:Invoice>
            <docset:Number>INV-001</docset:Number>
            <docset:Total xsi:type="decimal">42.00</docset:Total>
          </docset:Invoice>
        </dg:chunk>
    """.strip()
    )
    inner = full[0]  # the docset:Invoice element

    # Re-parse the same subtree as a standalone document — different object identity,
    # different parent context, but exclusive C14N hashes the same bytes.
    standalone_xml = etree.tostring(inner)
    standalone = etree.fromstring(standalone_xml)

    assert merkle_root(inner) == merkle_root(standalone)
    assert canonical_hash(inner) == canonical_hash(standalone)


def test_subtree_root_can_be_verified_against_subtree_proof() -> None:
    """Stake an inner subtree, prove an element inside it, verify against the subtree root.

    This is the headline 'stake a subtree (down to a leaf element) individually' workflow.
    """
    full = etree.fromstring(
        b"""
        <dg:chunk xmlns:dg="http://x">
          <section>
            <para>One.</para>
            <para>Two.</para>
            <para>Three.</para>
          </section>
        </dg:chunk>
    """.strip()
    )
    subtree = full[0]
    subtree_root_hash = merkle_root(subtree)
    target = subtree[1]  # the second <para>
    proof = merkle_proof(subtree, target)
    assert verify_proof(subtree_root_hash, target, proof)


# --- proof JSON serialization -------------------------------------------------


def test_proof_json_round_trip_verifies() -> None:
    tree = etree.fromstring(b"<r><a>x</a><b>y</b><c>z</c></r>")
    target = tree[1]
    proof = merkle_proof(tree, target)

    revived = proof_from_json(json.loads(json.dumps(proof_to_json(proof))))
    assert revived == proof
    assert verify_proof(merkle_root(tree), target, revived)


def test_proof_from_json_rejects_malformed_input() -> None:
    tree = etree.fromstring(b"<r><a>x</a><b>y</b></r>")
    good = proof_to_json(merkle_proof(tree, tree[0]))

    with pytest.raises(ValueError, match="JSON object"):
        proof_from_json([good])
    with pytest.raises(ValueError, match="leaf_hash"):
        proof_from_json({**good, "leaf_hash": "zz"})
    with pytest.raises(ValueError, match="leaf_index"):
        proof_from_json({**good, "leaf_index": -1})
    with pytest.raises(ValueError, match="leaf_index"):
        proof_from_json({**good, "leaf_index": True})
    with pytest.raises(ValueError, match="out of range"):
        proof_from_json({**good, "leaf_index": 99})
    with pytest.raises(ValueError, match="side"):
        path = [{"sibling": "0" * 64, "side": "left"}]
        proof_from_json({**good, "path": path})
    with pytest.raises(ValueError, match=r"path\[0\].sibling"):
        path = [{"sibling": "not-hex", "side": "L"}]
        proof_from_json({**good, "path": path})
