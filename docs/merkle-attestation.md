# DGML Merkle Attestation

This document describes how to **stake** a DGML XML tree (or any subtree,
down to a single leaf element) to an external system — typically a
blockchain — and later prove that a specific element was in the staked tree
at its original position.

The mechanism is a SHA-256 Merkle tree over the elements of a DGML document.
Only the **root hash** needs to be published on-chain; everything else
(element XML + inclusion proof) lives off-chain.

## Overview

The lifecycle is three steps:

1. **Stake.** Compute `merkle_root(tree)` and publish that 64-char hex hash
   to whatever immutable medium you trust (blockchain, transparency log,
   notarized timestamp). Optionally publish `leaf_count` alongside it —
   see [Security](#security).
2. **Prove.** When you need to attest that a particular element was in the
   tree, call `merkle_proof(tree, element)` to produce a `MerkleProof`. The
   proof is small (one hash per relevant tree level) and self-contained.
3. **Verify.** Anyone with the published root hash, the element's XML, and
   the proof can call `verify_proof(root_hash, element, proof)` — no access
   to the original tree or any other elements is required.

Sub-tree staking works the same way: pass any inner element to `merkle_root`
as if it were the whole document. Exclusive C14N (see below) guarantees the
subtree's hash is independent of where it sat in the parent document.

## When to use this

- **Anchoring a docset snapshot.** Publish one Merkle root on-chain to
  commit to the exact state of every DGML in the set at that moment.
- **Proving a clause was in a contract.** A counterparty receives just the
  clause + its proof + the published root. They never see the other clauses.
- **Third-party verification without access to the original PDF.** The PDF
  produced the DGML once; the DGML's Merkle root is the durable attestation.

## Algorithm specification

The algorithm is deterministic. Two callers with the same XML tree produce
the same root hash byte-for-byte.

- **Leaf hash.** For every XML element `e` in the tree:
  ```
  leaf_hash(e) = sha256(exclusive_c14n(e))     # lowercase, unprefixed hex
  ```
  Exclusive C14N is [XML C14N 1.0](https://www.w3.org/TR/xml-exc-c14n/) with
  the `exclusive=True` flag — *not* C14N 2.0, which has no exclusive mode.
  Comments are excluded (`with_comments=False`).

- **Enumeration.** Leaves are listed in **DFS pre-order** over every XML
  element in the tree. Text nodes, comments, and processing instructions
  are **not** separate Merkle leaves. Their treatment inside the parent
  element's canonicalization differs by node type — see the table below.

### What contributes to a parent element's hash

`canonical_hash(elem)` is `sha256` over `etree.tostring(elem, method="c14n",
exclusive=True, with_comments=False)`. The c14n flags pin which kinds of
child nodes survive into the canonical bytes:

| Child node type   | Separate Merkle leaf? | Part of parent's c14n bytes? | Net effect on attestation |
|---|---|---|---|
| Child element     | ✅ yes                | ✅ yes (recursively)          | Tamper detection at both levels |
| Text / tail text  | ❌ no                 | ✅ yes                        | Mutating text changes parent's hash |
| Attribute         | ❌ no                 | ✅ yes                        | Mutating an attribute changes parent's hash |
| Comment           | ❌ no                 | ❌ no (`with_comments=False`) | **Fully invisible** — adding or removing a comment changes no hash |
| Processing instr. | ❌ no                 | ✅ yes (lxml has no equivalent strip flag) | Adding, removing, or mutating a PI **does** change the parent's hash |

The PI / comment asymmetry is a direct consequence of lxml's c14n API:
`with_comments` is the only strip flag exposed. DGML itself doesn't emit
comments or PIs in normal output, so the asymmetry rarely matters in
practice — but if you feed hand-authored XML with PIs into the attestation,
be aware that those PIs are load-bearing.

- **Tree construction.** Pair-wise SHA-256 over the *raw bytes* of two
  child hashes:
  ```
  internal_hash(L, R) = sha256(bytes.fromhex(L) + bytes.fromhex(R))
  ```
  Build level-by-level bottom-up. When a level has an odd number of nodes,
  the unpaired node **promotes unchanged** to the next level — this is the
  [RFC 6962 / Certificate Transparency](https://datatracker.ietf.org/doc/html/rfc6962)
  convention, chosen over Bitcoin's "duplicate the last hash" rule because
  it avoids the [CVE-2012-2459](https://bitcointalk.org/?topic=102395)
  second-preimage weakness.

- **Domain separation.** None. The hash of a leaf element and the hash of
  an internal node use the same SHA-256 invocation with no prefix byte. This
  preserves the user-facing identity `canonical_hash(elem) == leaf_hash(elem)`.
  See [Security](#security) for the trade-off and mitigation.

### Trivial cases

- **Single-element tree.** `merkle_root(elem)` equals `canonical_hash(elem)`;
  the proof is an empty list; verification reduces to a direct hash
  comparison.
- **3-element tree.** Leaf 2 is the lone odd-out at level 0; it promotes
  unchanged to level 1 where it pairs with `H(L0,L1)`. The proof for
  leaf 2 contains exactly one step.

## API surface

All symbols live in `dgml.merkle`. The package ships type information
(`py.typed`).

| Symbol | Purpose |
|---|---|
| `canonical_hash(element)` | SHA-256 over the exclusive C14N of one element. |
| `merkle_leaves(root)` | Ordered DFS pre-order leaf-hash list. |
| `merkle_root(root)` | Compute the Merkle root of the tree. |
| `merkle_proof(root, target)` | Build an inclusion proof for `target`. |
| `verify_proof(root_hash, element, proof)` | Verify an inclusion proof. |
| `merkle_tree(root)` | Diagnostic: full leaf list + every intermediate level + root. |
| `ProofStep(sibling, side)` | One sibling hash + side ("L" or "R"). |
| `MerkleProof(leaf_hash, leaf_index, leaf_count, path)` | Inclusion proof. |
| `MerkleTree(leaves, levels, root)` | Diagnostic tree view. |
| `proof_to_json(proof)` / `proof_from_json(data)` | JSON round-trip for a `MerkleProof` — `{"leaf_hash", "leaf_index", "leaf_count", "path": [{"sibling", "side"}]}`. Revival validates structure and raises `ValueError` on malformed input. |

`verify_proof` returns `bool` rather than raising on tamper — it's designed
to be safe to call inside untrusted verification pipelines.

`merkle_proof` raises `ValueError` when `target` isn't reachable from `root`
by Python object identity. See [Object identity](#object-identity).

## Examples

### Stake a whole document

```python
from lxml import etree
from dgml_core.merkle import merkle_root

root = etree.parse("invoice.dgml.xml").getroot()
root_hash = merkle_root(root)
# Publish root_hash (and len(merkle_leaves(root)) — see Security) on-chain.
```

### Stake a subtree

Any inner element works directly — no detachment or reparsing needed.

```python
from dgml_core.merkle import merkle_root

invoice = root.find(".//docset:Invoice", namespaces={"docset": "..."})
invoice_root_hash = merkle_root(invoice)
```

### Generate a proof for a specific element

```python
from dgml_core.merkle import merkle_proof

clause = root.find(".//docset:TerminationClause", namespaces={"docset": "..."})
proof = merkle_proof(root, clause)
# proof.leaf_hash, proof.leaf_index, proof.leaf_count, proof.path
```

### Verify a proof

```python
from lxml import etree
from dgml_core.merkle import verify_proof

# The verifier has only: root_hash (from chain), the element XML, the proof.
element = etree.fromstring(received_xml_bytes)
ok = verify_proof(root_hash, element, proof)
assert ok, "element does not belong to the staked tree"
```

## Hash format

- **Lowercase hex**, **unprefixed**, 64 characters
  (e.g. `0eb001dcba24b1586c88011d84680842933a157323b145d538158cea00d907f5`).
- Note that Ethereum tooling typically expects an `0x` prefix on hex
  hashes; the caller is responsible for that conversion at the chain
  boundary. Inside DGML, hashes are always raw lowercase hex.

## Interop and versioning notes

- **Pin your `dgml` and `lxml` versions** when generating attestations that
  must be re-verifiable years later. Exclusive C14N is RFC-stable, but lxml
  has shipped serialization bug fixes over the years; the bytes a 2026 lxml
  produces today may differ at the margin from a 2018 lxml on edge-case
  inputs.
- Attestations are **not Bitcoin-SPV-compatible** — Bitcoin uses
  double-SHA-256 with a different leaf hash function.
- Attestations are **not interchangeable** with stdlib
  `xml.etree.ElementTree.canonicalize()` — even with `exclusive=True`,
  stdlib's defaults differ subtly from lxml's. lxml's
  `tostring(method="c14n", exclusive=True, with_comments=False)` is the
  reference implementation for this module.

## Security

### Exclusive C14N for subtree staking

The "exclusive" in Exclusive C14N means the canonicalized output only carries
namespace declarations actually *used* by the subtree — none inherited from
ancestors are smuggled in. This is exactly what we need for subtree staking:
the bytes (and therefore the hash) of an inner element are the same whether
it lives inside a multi-namespace document or is parsed as a standalone
fragment.

### CVE-2012-2459 mitigation

Without domain separation, a Merkle root can in principle be reached by
multiple different leaf lists: an attacker who controls some elements
*could* construct an internal-node hash that looks like a leaf hash and
forge a different tree with the same root. The standard mitigation is to
also publish the `leaf_count` alongside the root — a verifier that knows
the expected leaf count can detect any tree-shape forgery.

`MerkleProof.leaf_count` is included on every proof for exactly this
reason. **Stakers should commit `(root_hash, leaf_count)` together** (e.g.
in a single on-chain transaction) rather than the root alone. Verifiers
can sanity-check that the proof's `leaf_count` matches what was published.

### Object identity

`merkle_proof(root, target)` identifies `target` by Python object identity
(`is`), not by content equality. This means:

- **Parse once, prove many.** Build all the proofs you need from one
  `etree.fromstring(...)` call.
- **Re-parsing the same XML produces new element objects.** The hashes
  match, but `target is element_from_re_parse` is False, so passing a
  re-parsed element to `merkle_proof` raises `ValueError`. If you only
  have a re-parsed handle, re-derive the index by hash: build
  `merkle_leaves(root)` and find the index whose hash equals
  `canonical_hash(your_element)`.

`verify_proof` does **not** require object identity — it operates purely on
hashes and the proof's `path`. Verifiers can re-parse freely.

## Limitations

- **lxml only.** The module operates on `lxml.etree._Element`. If you have
  a stdlib `xml.etree.ElementTree.Element`, re-parse via
  `lxml.etree.fromstring(ET.tostring(elem))` first.
- **Whole tree in memory.** No streaming API; large documents are bounded
  by RAM rather than by the algorithm.
- **Mixed content, comments, and processing instructions.** DGML doesn't
  emit these in normal output. None of them become separate Merkle leaves;
  see the table in [Algorithm specification](#algorithm-specification) for
  precisely how each one contributes (or doesn't) to the parent element's
  hash.
