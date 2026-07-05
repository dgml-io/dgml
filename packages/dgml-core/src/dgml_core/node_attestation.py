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

"""Element-level attestation over a file's generated DGML XML.

Where :mod:`dgml.file_attestation` rolls a file's whole artifact set into
one Merkle root (the DGMLX bundle), this module attests a *single element*
of the DGML XML: its canonical hash, the XML tree's Merkle root, and the
RFC 6962 inclusion proof connecting the two. The trio is what an external
notary needs to later prove "this exact node was part of that exact
document" without ever seeing the other nodes (see
``docs/merkle-attestation.md``).

Node addressing
---------------

Three interchangeable coordinates name an element:

- **leaf index** — the 0-based DFS pre-order position among the
  document's elements; identical to :func:`dgml.merkle.merkle_leaves`
  ordering and to ``MerkleProof.leaf_index``.
- **XPath** — a canonical, fully positional expression using the
  document's own namespace prefixes (``/dg:chunk/docset:LedgerEntry[3]``).
  :func:`element_xpath` emits it; :func:`resolve_xpath` evaluates any
  XPath (canonical or hand-written) against the document's root nsmap
  and requires exactly one element match.
- **child path** — a list of 0-based child-element indices walked from
  the document root (``[1, 1]`` = "the root's 2nd child element's 2nd
  child element"). This is the coordinate a DOM tree view naturally has
  (e.g. a browser's ``Element.children``) when the user has clicked a
  node but the caller has no ready-made XPath or leaf index for it.
  :func:`resolve_child_path` walks it directly against the same
  element-only child ordering :func:`ordered_elements` uses (comments
  and processing instructions are skipped at every level, matching how
  ``Element.children`` already excludes them).

Export resolves either coordinate to the element, and always reports
both, plus the element's exclusive-C14N serialization (the exact bytes
the node hash covers — a holder can ``sha256`` it directly).

Proving
-------

:func:`prove_node` re-reads the *current* workspace XML, takes the
element at the proof's recorded leaf index, and verifies the inclusion
proof against the expected root. Any change to the node's subtree, the
proof, or the root yields ``valid=False``; a document so reshaped that
the leaf index no longer exists raises :class:`ValueError` (structural
mismatch, not a tamper verdict — same contract as
:func:`dgml.file_attestation.verify_file_version`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lxml import etree  # type: ignore[import-untyped]
from lxml.etree import _Element  # type: ignore[import-untyped]

from .errors import DocSetNotFound, FileNotFound, InvalidArgument, NotFoundError
from .merkle import MerkleProof, canonical_hash, merkle_proof, merkle_root, verify_proof
from .models import FileRecord
from .storage import Workspace, read_json


@dataclass(frozen=True)
class NodeAttestation:
    """Everything needed to attest one DGML XML element externally.

    ``node_hash`` equals ``proof.leaf_hash`` (carried flat because it is
    the value a caller anchors as the record checksum). ``node_xml`` is
    the element's exclusive-C14N serialization — hashing its UTF-8 bytes
    with SHA-256 reproduces ``node_hash`` exactly.
    """

    file_id: str
    docset_id: str
    leaf_index: int
    leaf_count: int
    xpath: str
    node_hash: str
    root_hash: str
    proof: MerkleProof
    node_xml: str


@dataclass(frozen=True)
class NodeVerifyResult:
    """Outcome of re-verifying a node proof against the current workspace.

    ``computed_node_hash`` is the canonical hash of whatever element sits
    at the proof's leaf index *now*; on a failed verify, comparing it to
    the proof's ``leaf_hash`` distinguishes "this node changed" from
    "the tree around it changed".
    """

    file_id: str
    docset_id: str
    leaf_index: int
    xpath: str
    expected_root: str
    expected_node_hash: str
    computed_node_hash: str
    valid: bool


# --- document loading ---------------------------------------------------------


def load_dgml_root(ws: Workspace, file_id: str, docset_id: str) -> _Element:
    """Parse the generated DGML XML for ``(file, docset)`` and return its root.

    Reads the canonical per-(docset, file) location the generator writes
    to — the same artifact the DGMLX ``dgml_xml`` slot hashes, so node
    proofs chain consistently with bundle attestations.

    Raises:
        :class:`InvalidArgument` — empty ``file_id`` or ``docset_id``.
        :class:`FileNotFound` / :class:`DocSetNotFound` — unknown ids.
        :class:`NotFoundError` — the file/docset exist but no DGML XML
            has been generated for the pair.
        :class:`ValueError` — the XML on disk is not well-formed.
    """
    if not file_id.strip():
        raise InvalidArgument("file id must not be empty")
    if not docset_id.strip():
        raise InvalidArgument("docset id must not be empty")
    if not ws.file_dir(file_id).exists():
        raise FileNotFound(f"file '{file_id}' not found in workspace")
    if not ws.docset_dir(docset_id).exists():
        raise DocSetNotFound(f"docset '{docset_id}' not found in workspace")

    record = FileRecord.from_json(read_json(ws.file_json_path(file_id)))
    xml_path = ws.file_dgml_xml_path(docset_id, file_id, Path(record.original_filename).stem)
    if not xml_path.exists():
        raise NotFoundError(
            f"no generated DGML XML for file '{file_id}' in docset '{docset_id}' "
            f"(expected {xml_path})"
        )
    try:
        tree = etree.parse(str(xml_path))
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"{xml_path} is not well-formed XML: {exc}") from exc
    root: _Element = tree.getroot()
    return root


def ordered_elements(root: _Element) -> list[_Element]:
    """The document's elements in DFS pre-order — the Merkle leaf order."""
    return [el for el in root.iter() if isinstance(el.tag, str)]


# --- xpath <-> element --------------------------------------------------------


def element_xpath(element: _Element) -> str:
    """Canonical positional XPath for ``element``, using the document's prefixes.

    Each step is the element's qualified name as written (``docset:Item``),
    with a 1-based positional predicate whenever the element has same-tag
    siblings. Elements in a *default* (unprefixed) namespace use a
    ``*[local-name()='…']`` step, since XPath 1.0 has no way to reference
    a default namespace by name.
    """
    steps: list[str] = []
    el = element
    while True:
        parent = el.getparent()
        step = _step_name(el)
        if parent is None:
            steps.append(step)  # document element is unique; no predicate
            break
        same_tag = [sib for sib in parent if isinstance(sib.tag, str) and sib.tag == el.tag]
        if len(same_tag) > 1:
            step += f"[{same_tag.index(el) + 1}]"
        steps.append(step)
        el = parent
    return "/" + "/".join(reversed(steps))


def resolve_xpath(root: _Element, xpath: str) -> _Element:
    """Evaluate ``xpath`` against ``root``'s tree; require exactly one element.

    Namespace prefixes resolve through the root element's nsmap (DGML
    output declares every prefix on the document element). Raises
    :class:`InvalidArgument` for an unparseable expression, a non-element
    result, or a match count other than one — anchoring an ambiguous
    selector would attest a different node than the user inspected.
    """
    nsmap = {prefix: uri for prefix, uri in (root.nsmap or {}).items() if prefix}
    try:
        found = root.getroottree().xpath(xpath, namespaces=nsmap)
    except etree.XPathError as exc:
        raise InvalidArgument(f"invalid xpath {xpath!r}: {exc}") from exc
    if not isinstance(found, list):
        raise InvalidArgument(f"xpath {xpath!r} evaluates to a {type(found).__name__}, not nodes")
    elements = [el for el in found if isinstance(el, _Element) and isinstance(el.tag, str)]
    if len(elements) != 1:
        raise InvalidArgument(f"xpath {xpath!r} matched {len(elements)} elements (need exactly 1)")
    return elements[0]


def resolve_child_path(root: _Element, child_path: list[int]) -> _Element:
    """Walk ``child_path`` from ``root`` to the element it addresses.

    Each index selects among the current element's *element* children only
    (comments/PIs are skipped), so the coordinate matches a DOM tree view's
    ``Element.children`` addressing exactly. An empty path means ``root``
    itself. Raises :class:`InvalidArgument` if any index is out of range.
    """
    el = root
    for depth, idx in enumerate(child_path):
        children = [c for c in el if isinstance(c.tag, str)]
        if not 0 <= idx < len(children):
            raise InvalidArgument(
                f"child_path {child_path!r} index {idx} at depth {depth} out of range "
                f"({len(children)} child elements)"
            )
        el = children[idx]
    return el


def _step_name(el: _Element) -> str:
    qname = etree.QName(el)
    if el.prefix:
        return f"{el.prefix}:{qname.localname}"
    if qname.namespace:
        return f"*[local-name()='{qname.localname}']"
    return str(qname.localname)


# --- export / prove -----------------------------------------------------------


def export_node(
    ws: Workspace,
    file_id: str,
    docset_id: str,
    *,
    leaf_index: int | None = None,
    xpath: str | None = None,
    child_path: list[int] | None = None,
) -> NodeAttestation:
    """Attest one element of the file's DGML XML.

    Exactly one of ``leaf_index`` / ``xpath`` / ``child_path`` selects the
    element; the other two coordinates are derived and reported. Raises
    :class:`InvalidArgument` for a missing/multiple selector or an
    out-of-range index, plus everything :func:`load_dgml_root` raises.
    """
    provided = sum(v is not None for v in (leaf_index, xpath, child_path))
    if provided != 1:
        raise InvalidArgument("provide exactly one of leaf_index, xpath, or child_path")

    root = load_dgml_root(ws, file_id, docset_id)
    elements = ordered_elements(root)

    if leaf_index is not None:
        if not 0 <= leaf_index < len(elements):
            raise InvalidArgument(
                f"leaf index {leaf_index} out of range (document has {len(elements)} elements)"
            )
        target = elements[leaf_index]
    elif xpath is not None:
        target = resolve_xpath(root, xpath)
    else:
        assert child_path is not None
        target = resolve_child_path(root, child_path)

    proof = merkle_proof(root, target)
    canonical: bytes = etree.tostring(target, method="c14n", exclusive=True, with_comments=False)
    return NodeAttestation(
        file_id=file_id,
        docset_id=docset_id,
        leaf_index=proof.leaf_index,
        leaf_count=proof.leaf_count,
        xpath=element_xpath(target),
        node_hash=proof.leaf_hash,
        root_hash=merkle_root(root),
        proof=proof,
        node_xml=canonical.decode("utf-8"),
    )


def prove_node(
    ws: Workspace,
    file_id: str,
    docset_id: str,
    root_hash: str,
    proof: MerkleProof,
) -> NodeVerifyResult:
    """Verify ``proof`` against the workspace's current DGML XML.

    The candidate element is the one at ``proof.leaf_index`` in today's
    document. ``valid`` is ``True`` iff its canonical hash matches the
    proof's leaf hash *and* the sibling path reproduces ``root_hash``.
    Raises :class:`ValueError` when the index no longer exists (the
    document was restructured — an inventory mismatch, not a tamper
    verdict), plus everything :func:`load_dgml_root` raises.
    """
    root = load_dgml_root(ws, file_id, docset_id)
    elements = ordered_elements(root)
    if proof.leaf_index >= len(elements):
        raise ValueError(
            f"proof leaf index {proof.leaf_index} out of range — the document now has "
            f"{len(elements)} elements (was {proof.leaf_count} when attested)"
        )
    target = elements[proof.leaf_index]
    return NodeVerifyResult(
        file_id=file_id,
        docset_id=docset_id,
        leaf_index=proof.leaf_index,
        xpath=element_xpath(target),
        expected_root=root_hash,
        expected_node_hash=proof.leaf_hash,
        computed_node_hash=canonical_hash(target),
        valid=verify_proof(root_hash, target, proof),
    )
