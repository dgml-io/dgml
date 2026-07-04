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

"""File-level Merkle attestation across a file's DGML artifacts.

A file's *DGML version* is the set of on-disk artifacts that, taken
together, constitute everything DGML knows about that file at a moment
in time. Five artifact categories participate:

1. Source document (binary) — the original ``.pdf``/``.docx``/… (slot ``source``)
2. Page images, one PNG per page (binary)
3. The DocSet's ``full-schema.rnc`` generation tag schema (binary — raw RNC
   bytes; docset-scoped). The RNC render is lossless over ``schema.json``
   (every field survives as ``# Field: value`` comments), so attesting it
   covers the JSON exchange form; ``schema.json`` itself is not a leaf.
4. The DocSet's ``extraction-schema.rnc`` grounded extraction schema
   (binary — raw RNC bytes; docset-scoped)
5. The file's generated ``<stem>.dgml.xml`` (XML; docset-scoped)

The first two are *file-side* and present whenever the file has been
added. The last three are *docset-side* — they only exist for a given
``(file, docset)`` pair. Passing ``docset_id=None`` to
:func:`collect_file_version` therefore attests over the file-side
artifacts only.

The per-page text JSONs under ``page_text/`` (the token files produced by
text extraction) are intentionally **excluded** — they are an intermediate
artifact, not part of the portable bundle, and never participate in a
version or its root.

Any subset of slots can be missing (generation hasn't run yet, an image
failed to render, etc.); the version attests over whatever currently
exists, and a verifier compares slot-by-slot against the recorded
inventory.

Hashing by artifact kind
------------------------

- ``BINARY`` — SHA-256 of the raw bytes.
- ``XML`` — :func:`dgml.merkle.merkle_root` of the parsed element tree.
  This nests the existing exclusive-C14N + RFC 6962 element-Merkle
  attestation as one leaf of the file attestation, so a verifier holding
  the DGML XML and an inclusion proof can later prove individual
  elements were part of the attested file version.

Leaf ordering
-------------

Leaves are emitted in a fixed slot order (source → page images by page
number → full schema → extraction schema → DGML XML). Two workspaces with the same artifact set
produce the same root regardless of filesystem walk order.

For a *portable bundle* (see :func:`export_attestation`) the page
ordering is driven by an explicit ``number`` attribute recorded in the
``META-INF/dgml-attestation.xml`` attestation file rather than by parsing
the on-disk filename. That file's ``<artifacts>`` inventory maps each
artifact's role to a relative path inside the bundle, so the bundle's
files can be named anything — verification re-derives the canonical leaf
order from the inventory's ``number`` attributes alone. The attestation
file is *not itself* a leaf; it only carries the recorded Merkle root,
the rendering provenance, and the ordering metadata. A verifier reads it,
re-hashes the referenced artifacts in canonical order, and compares the
recomputed root against the one the file carries.

Outer Merkle reduction is RFC 6962 (pair-wise SHA-256 over raw bytes;
lone odd-out promotes unchanged), via
:func:`dgml.merkle.merkle_root_from_hashes` — the same algorithm the
XML attestation already uses.

Errors vs. failures
-------------------

Following the same contract as :mod:`dgml.merkle`, malformed inputs
raise rather than silently returning a false verification. Missing
artifacts are *not* an error — they're a smaller version. Tampered
content on a present slot returns ``False`` from
:func:`verify_file_version`. A *structural* mismatch (different slot
inventory, missing file, missing docset) raises.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from lxml import etree  # type: ignore[import-untyped]

from .errors import (
    AttestationInvalid,
    DocSetNotFound,
    FileNotFound,
    InvalidArgument,
)
from .hashing import sha256_file
from .merkle import merkle_root, merkle_root_from_hashes
from .models import FileRecord
from .opc import (
    CONTENT_TYPES_FILENAME,
    DGMLX_EXTENSION,
    PACKAGE_RELS_PATH,
    REL_TYPE_ATTESTATION,
    REL_TYPE_DGML_XML,
    REL_TYPE_MAIN_DOCUMENT,
    PackageRelationship,
    write_content_types,
    write_package_rels,
    zip_package,
)
from .pages import PAGE_GLOB
from .storage import Workspace, read_json

_ATTESTATION_VERSION = "1"

# The bundle's single namespaced attestation file, ``META-INF/dgml-attestation.xml``.
# It is both the manifest and the provenance record: the Merkle root, the
# workspace identity, the rendering provenance from ``file.json`` (page-image
# DPI / renderer, plus the PDF converter for a non-PDF source), and the
# ``<artifacts>`` inventory that drives verification. The file is not itself a
# leaf of the root.
ATTESTATION_NS = "http://dgml.io/ns/attestation"
METADATA_DIRNAME = "META-INF"
METADATA_FILENAME = "dgml-attestation.xml"

# All hashing in this module — leaf hashes and Merkle inner nodes alike — is
# SHA-256. The algorithm is not user-selectable, so the attestation file records
# it explicitly (as the 'algorithm' attribute on <merkle-root>) rather than
# leaving bundle holders to infer it from tooling versions. An attestation file
# with no 'algorithm' attribute is read as sha256; any other recorded value is
# rejected rather than risking a false tamper verdict from re-hashing with the
# wrong algorithm.
_HASH_ALGORITHM = "sha256"


class ArtifactKind(StrEnum):
    """How a single artifact's leaf hash is computed."""

    BINARY = "binary"
    XML = "xml"


@dataclass(frozen=True)
class ArtifactRef:
    """One leaf in a file attestation.

    ``slot_id`` is a stable, human-readable identifier for the artifact's
    role in the version (``"source"``, ``"page_image[3]"``, ``"full_schema"``,
    ``"dgml_xml"``). It names the role, not the file format — ``"source"`` is
    the original document whatever its extension (``.pdf``, ``.docx``, …).
    Survives renames of the underlying file and is what a verifier compares
    slot-by-slot.

    ``leaf_hash`` is lowercase hex SHA-256: for ``BINARY`` it's the file
    bytes; for ``XML`` it's :func:`dgml.merkle.merkle_root` of the parsed tree.

    ``number`` is the 1-based page index for ``page_image`` slots and
    ``None`` for the single-instance slots (``source``, ``full_schema``,
    ``extraction_schema``, ``dgml_xml``). It's what drives ordering in a portable bundle's
    ``META-INF/dgml-attestation.xml`` — the on-disk filename is never parsed.
    """

    slot_id: str
    path: Path
    kind: ArtifactKind
    leaf_hash: str
    number: int | None = None


@dataclass(frozen=True)
class FileVersion:
    """The artifacts that currently exist for a ``(file, docset?)`` pair.

    ``artifacts`` is in canonical slot order, ready to be Merkle-rolled
    by :func:`attest_file_version`. ``docset_id`` is ``None`` for
    file-only versions.
    """

    file_id: str
    docset_id: str | None
    artifacts: tuple[ArtifactRef, ...]


@dataclass(frozen=True)
class FileAttestation:
    """A Merkle attestation over one :class:`FileVersion`.

    ``leaves`` is the full slot inventory carried alongside ``root`` so a
    verifier can locate the precise mismatch on a failed verify
    (different slot set vs. one tampered slot). For canonical-only
    publication, ship ``root`` plus the list of ``slot_id`` strings;
    keep the leaf hashes private if proof generation isn't needed.
    """

    file_id: str
    docset_id: str | None
    leaves: tuple[ArtifactRef, ...]
    root: str


# --- discovery --------------------------------------------------------------


# Page images use the renderer's canonical extension (PNG; see
# dgml.pages.PAGE_GLOB) — deriving it here keeps attestation in lockstep
# with whatever the render pipeline writes rather than hardcoding a guess.
_PAGE_IMAGE_EXT = PAGE_GLOB.rsplit(".", 1)[-1]
_PAGE_FILE_RE = re.compile(rf"^page_(\d+)\.{_PAGE_IMAGE_EXT}$")


def _page_num(path: Path) -> int:
    """Pull the page index out of a ``page_N.<ext>`` page-image filename.

    Raises :class:`ValueError` if the name doesn't match — a stray file in
    ``page_images/`` is bad input, not a missing artifact, and surfacing it
    loudly beats silently sorting alphabetically.
    """
    m = _PAGE_FILE_RE.match(path.name)
    if m is None:
        raise ValueError(f"unexpected file name '{path.name}' (expected 'page_N.<ext>')")
    return int(m.group(1))


def collect_file_version(
    ws: Workspace,
    file_id: str,
    docset_id: str | None = None,
) -> FileVersion:
    """Discover and hash every artifact that exists for ``file_id``.

    When ``docset_id`` is ``None``, only file-side artifacts (source, page
    images) are considered. When
    ``docset_id`` is provided, the docset's ``full-schema.rnc`` (generation
    tag schema), its ``extraction-schema.rnc`` (grounded extraction schema),
    and the file's ``<stem>.dgml.xml`` are also included if they exist. The
    per-page text JSONs under ``page_text/`` are never attested, and neither
    is ``schema.json`` — the lossless RNC render supersedes it.

    Returns a :class:`FileVersion` with artifacts in canonical slot
    order. Hashing happens during collection so the returned refs are
    self-contained.

    Raises:
        :class:`InvalidArgument` — empty ``file_id`` or ``docset_id``.
        :class:`FileNotFound` — file's directory doesn't exist.
        :class:`DocSetNotFound` — ``docset_id`` given but missing.
        :class:`CorruptMetadata` — ``file.json`` (needed for the PDF
            filename) is missing or unparseable, or the schema JSON
            isn't valid JSON.
        :class:`ValueError` — a stray file in ``page_images/`` doesn't
            match the expected name pattern, or the DGML XML isn't
            well-formed, or no artifacts at all were discovered (an empty
            version is meaningless to attest).
    """
    if not file_id.strip():
        raise InvalidArgument("file id must not be empty")
    if docset_id is not None and not docset_id.strip():
        raise InvalidArgument("docset id must not be empty when provided")
    if not ws.file_dir(file_id).exists():
        raise FileNotFound(f"file '{file_id}' not found in workspace")
    if docset_id is not None and not ws.docset_dir(docset_id).exists():
        raise DocSetNotFound(f"docset '{docset_id}' not found in workspace")

    # file.json is consulted even though it's not itself a leaf — the
    # PDF's on-disk name (and therefore the DGML XML's expected stem) live
    # there. A workspace without a parseable file.json is structurally
    # broken; let the read_json call's CorruptMetadata propagate.
    record = FileRecord.from_json(read_json(ws.file_json_path(file_id)))

    refs: list[ArtifactRef] = []

    # Slot 1: the original source document (a .pdf, or the .docx/.xls/… that
    # was converted). Named "source" — the role, not the file format.
    source_path = ws.file_dir(file_id) / record.original_filename
    if source_path.exists():
        refs.append(_binary_ref("source", source_path))

    # Slot 2: page images, ordered by page number (not lexicographic —
    # 'page_10.png' sorts before 'page_2.png' alphabetically).
    pages_dir = ws.file_pages_dir(file_id)
    if pages_dir.exists():
        for img_path in sorted(pages_dir.glob(PAGE_GLOB), key=_page_num):
            n = _page_num(img_path)
            refs.append(_binary_ref(f"page_image[{n}]", img_path, number=n))

    # Per-page text JSONs (`page_text/`) are deliberately *not* attested: the
    # token files are an intermediate text-extraction artifact, not part of the
    # portable bundle. They never participate in a file version or its root.

    if docset_id is not None:
        # Slot 3: the generation tag schema that governs this file's DGML XML
        # (`full-schema.rnc`, the lossless RNC render written at the end of
        # `docset generate`). Hashed as raw bytes, like the extraction schema.
        # schema.json is deliberately not a leaf — the RNC carries every one
        # of its fields as `# Field: value` comments.
        full_schema_path = ws.docset_full_schema_path(docset_id)
        if full_schema_path.exists():
            refs.append(_binary_ref("full_schema", full_schema_path))

        # Slot 4: the grounded extraction schema (`extraction-schema.rnc`,
        # RELAX NG Compact) that governs this file's `dg:extraction`. Hashed as
        # raw bytes — RNC is plain text, neither JSON nor XML. Present only once
        # `extraction set-schema` / `generate-schema` has run for the docset.
        extraction_schema_path = ws.docset_schema_path(docset_id)
        if extraction_schema_path.exists():
            refs.append(_binary_ref("extraction_schema", extraction_schema_path))

        # Slot 5: DGML XML output for this file.
        dgml_xml_path = ws.file_dgml_xml_path(
            docset_id, file_id, Path(record.original_filename).stem
        )
        if dgml_xml_path.exists():
            refs.append(_xml_ref("dgml_xml", dgml_xml_path))

    if not refs:
        scope = f" in docset '{docset_id}'" if docset_id else ""
        raise ValueError(f"no artifacts found for file '{file_id}'{scope}")

    return FileVersion(file_id=file_id, docset_id=docset_id, artifacts=tuple(refs))


def attest_file_version(version: FileVersion) -> FileAttestation:
    """Roll the artifacts of ``version`` up to a Merkle root.

    The artifacts are taken as-is and in the order they appear — the
    caller is :func:`collect_file_version` in the normal flow, which
    already emits them in canonical slot order. Raises
    :class:`ValueError` on an empty version.
    """
    if not version.artifacts:
        raise ValueError("cannot attest an empty file version")
    root = merkle_root_from_hashes([a.leaf_hash for a in version.artifacts])
    return FileAttestation(
        file_id=version.file_id,
        docset_id=version.docset_id,
        leaves=version.artifacts,
        root=root,
    )


def attest_file(
    ws: Workspace,
    file_id: str,
    docset_id: str | None = None,
) -> FileAttestation:
    """Convenience: :func:`collect_file_version` then :func:`attest_file_version`."""
    return attest_file_version(collect_file_version(ws, file_id, docset_id))


def verify_file_version(ws: Workspace, attestation: FileAttestation) -> bool:
    """Re-collect and re-attest the file version; compare to ``attestation``.

    Returns ``True`` iff the recomputed root matches. Returns ``False``
    when a present slot's content has been tampered with (its leaf hash
    no longer matches the original).

    Raises :class:`ValueError` when the *structure* of the version has
    changed — different slot count or different slot identifiers — since
    that is not a tampering signal but an inventory mismatch the
    caller has to reconcile. (Re-running generation, deleting a page
    image, etc. all change the slot set legitimately; the right
    response is to take a fresh attestation, not to call it a fail.)
    """
    current = collect_file_version(ws, attestation.file_id, attestation.docset_id)
    original_slots = [a.slot_id for a in attestation.leaves]
    current_slots = [a.slot_id for a in current.artifacts]
    if current_slots != original_slots:
        raise ValueError(
            "slot inventory differs from attestation: "
            f"original={original_slots} current={current_slots}"
        )
    return attest_file_version(current).root == attestation.root


# --- portable bundles: attestation file + export/verify ---------------------


@dataclass(frozen=True)
class AttestationEntry:
    """One artifact declared by a bundle's attestation-file ``<artifacts>`` inventory.

    ``rel_path`` is POSIX-style and relative to the bundle root. ``number``
    carries the page index for ``page_image`` slots (``None`` otherwise) and
    is the *only* thing the verifier uses to order page artifacts — the
    filename is never parsed.
    """

    slot_id: str
    kind: ArtifactKind
    number: int | None
    rel_path: str


@dataclass(frozen=True)
class AttestationInventory:
    """Parsed inventory from a bundle's ``META-INF/dgml-attestation.xml``.

    ``root`` is the Merkle root recorded at export time. ``entries`` is in
    canonical slot order (source → page images by ``number`` →
    full schema → extraction schema → DGML XML), ready to drive re-hashing. The
    attestation file is *not* a leaf of the attestation it describes — it
    only holds the recorded root, provenance, and the ordering metadata.
    """

    file_id: str
    docset_id: str | None
    root: str
    entries: tuple[AttestationEntry, ...]


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of verifying a bundle directory against its attestation file.

    ``valid`` is ``True`` iff ``computed_root == expected_root``.
    ``expected_root`` is the root recorded in the attestation file;
    ``computed_root`` is what re-hashing the on-disk artifacts in inventory
    order produces.
    """

    file_id: str
    docset_id: str | None
    expected_root: str
    computed_root: str
    valid: bool
    slot_ids: tuple[str, ...]


def export_attestation(
    ws: Workspace,
    file_id: str,
    out_dir: Path,
    docset_id: str | None = None,
    *,
    unpacked: bool = False,
) -> tuple[FileAttestation, Path | None, Path | None]:
    """Attest ``file_id`` and write a portable DGMLX bundle to ``out_dir``.

    Discovers and attests the file version from the workspace and assembles the
    bundle — every artifact under ``source/`` / ``page_images/`` /
    the top level, the single ``META-INF/dgml-attestation.xml``
    attestation file (Merkle root + rendering provenance + ``<artifacts>``
    inventory), and the OPC parts (``[Content_Types].xml`` + ``_rels/.rels``).

    The two output modes are mutually exclusive:

    - **default** — the bundle is staged in a temporary directory, zipped into a
      portable ``<stem>.dgmlx`` archive in ``out_dir``, and the staging removed,
      so ``out_dir`` holds only the archive.
    - **``unpacked=True``** — the loose bundle tree is written directly into
      ``out_dir`` and **no archive is produced**.

    The attestation file is deliberately *not* part of the attestation: it
    records the root so a verifier holding only the bundle can re-check it, and
    its inventory decouples leaf ordering from filenames.

    Returns the :class:`FileAttestation`, the loose attestation-file path
    (set only when ``unpacked``), and the ``.dgmlx`` archive path (set only when
    not ``unpacked``) — exactly one of the latter two is non-``None``.
    """
    attestation = attest_file(ws, file_id, docset_id)
    record = FileRecord.from_json(read_json(ws.file_json_path(file_id)))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(record.original_filename).stem

    # --unpacked assembles the loose tree directly in out_dir (no archive); the
    # default stages it in a throwaway dir, zips it, and discards the tree.
    staging = out_dir if unpacked else Path(tempfile.mkdtemp(prefix="dgmlx-"))
    archive_path: Path | None = None
    try:
        rel_paths: dict[str, str] = {}
        for ref in attestation.leaves:
            rel = _export_rel_path(ref)
            dest = staging / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ref.path, dest)
            rel_paths[ref.slot_id] = rel

        attestation_path = write_attestation(staging, attestation, record, rel_paths)
        parts = _write_opc_parts(staging, rel_paths)
        if not unpacked:
            archive_path = out_dir / f"{stem}{DGMLX_EXTENSION}"
            zip_package(staging, archive_path, [*parts, CONTENT_TYPES_FILENAME])
    finally:
        if not unpacked:
            shutil.rmtree(staging, ignore_errors=True)

    return attestation, (attestation_path if unpacked else None), archive_path


def _write_opc_parts(staging: Path, rel_paths: dict[str, str]) -> list[str]:
    """Write the OPC ``[Content_Types].xml`` and ``_rels/.rels`` into ``staging``.

    The relationships name the ``source/`` original as the main document, the
    generated DGML XML as dgml-xml (when present), and the attestation file as
    the attestation. Returns the bundle's part list (every part except the
    special ``[Content_Types].xml``), ready to be handed to :func:`zip_package`.
    """
    attestation_rel = f"{METADATA_DIRNAME}/{METADATA_FILENAME}"
    parts = [*rel_paths.values(), attestation_rel, PACKAGE_RELS_PATH]

    # Package relationships, in declaration order; rIds are assigned
    # sequentially over whatever is present (consumers match on Type, not Id).
    rel_specs: list[tuple[str, str]] = []
    source_rel = rel_paths.get("source")  # the original document, any format
    if source_rel is not None:
        rel_specs.append((REL_TYPE_MAIN_DOCUMENT, source_rel))
    dgml_xml_rel = rel_paths.get("dgml_xml")  # only on a docset-scoped export
    if dgml_xml_rel is not None:
        rel_specs.append((REL_TYPE_DGML_XML, dgml_xml_rel))
    rel_specs.append((REL_TYPE_ATTESTATION, attestation_rel))
    relationships = [
        PackageRelationship(f"rId{i}", rel_type, target)
        for i, (rel_type, target) in enumerate(rel_specs, start=1)
    ]

    write_content_types(staging, parts)
    write_package_rels(staging, relationships)
    return parts


def write_attestation(
    out_dir: Path,
    attestation: FileAttestation,
    record: FileRecord,
    rel_paths: dict[str, str],
) -> Path:
    """Write the bundle's ``META-INF/dgml-attestation.xml`` attestation file.

    This single namespaced file is both the provenance record and the manifest:
    it carries the Merkle root, the workspace identity (``file-id``, optional
    ``docset-id``), the rendering provenance from ``file.json``
    (``page-image-dpi`` / ``page-image-renderer`` and, for a converted non-PDF
    source, ``pdf-converter`` — each omitted when absent), and the
    ``<artifacts>`` inventory mapping every leaf's role to its relative path,
    with per-page ``number`` attributes.

    ``rel_paths`` maps each leaf's ``slot_id`` to its POSIX-style path relative
    to ``out_dir``; leaves are taken in canonical order so the inventory's
    grouping elements come out ordered. The file is *not* a leaf of the
    attestation it carries. Returns the written path.
    """
    q = _attestation_qname
    root_el = etree.Element(q("dgml-attestation"), nsmap={None: ATTESTATION_NS})
    root_el.set("version", _ATTESTATION_VERSION)
    if record.page_image_dpi is not None:
        root_el.set("page-image-dpi", str(record.page_image_dpi))
    if record.page_image_renderer is not None:
        root_el.set("page-image-renderer", record.page_image_renderer)
    if record.pdf_converter is not None:
        root_el.set("pdf-converter", record.pdf_converter)
    root_el.set("file-id", attestation.file_id)
    if attestation.docset_id is not None:
        root_el.set("docset-id", attestation.docset_id)

    mr = etree.SubElement(root_el, q("merkle-root"), algorithm=_HASH_ALGORITHM)
    mr.text = attestation.root

    arts = etree.SubElement(root_el, q("artifacts"))
    page_images_el: etree._Element | None = None

    for ref in attestation.leaves:
        rel = rel_paths[ref.slot_id]
        if ref.slot_id == "source":
            etree.SubElement(arts, q("source")).text = rel
        elif ref.slot_id.startswith("page_image["):
            if page_images_el is None:
                page_images_el = etree.SubElement(arts, q("page-images"))
            el = etree.SubElement(page_images_el, q("page-image"), number=str(ref.number))
            el.text = rel
        elif ref.slot_id == "full_schema":
            etree.SubElement(arts, q("full-schema")).text = rel
        elif ref.slot_id == "extraction_schema":
            etree.SubElement(arts, q("extraction-schema")).text = rel
        elif ref.slot_id == "dgml_xml":
            etree.SubElement(arts, q("dgml-xml")).text = rel
        else:  # pragma: no cover - guards against a future slot kind
            raise ValueError(f"cannot serialize unknown slot to attestation: {ref.slot_id!r}")

    meta_dir = out_dir / METADATA_DIRNAME
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / METADATA_FILENAME
    etree.ElementTree(root_el).write(
        str(path), encoding="utf-8", xml_declaration=True, pretty_print=True
    )
    return path


def read_attestation(directory: Path) -> AttestationInventory:
    """Parse ``META-INF/dgml-attestation.xml`` into an :class:`AttestationInventory`.

    Entries are returned in canonical slot order, with page artifacts
    sorted by their ``number`` attribute (not document order, not
    filename). Raises :class:`AttestationInvalid` for a missing, malformed,
    or internally inconsistent attestation file (bad/duplicate page numbers,
    unknown elements, missing root).
    """
    q = _attestation_qname
    path = directory / METADATA_DIRNAME / METADATA_FILENAME
    rel = f"{METADATA_DIRNAME}/{METADATA_FILENAME}"
    if not path.exists():
        raise AttestationInvalid(f"no {rel} found in {directory}")
    try:
        tree = etree.parse(str(path))
    except etree.XMLSyntaxError as exc:
        raise AttestationInvalid(f"{path} is not well-formed XML: {exc}") from exc

    root_el = tree.getroot()
    if root_el.tag != q("dgml-attestation"):
        raise AttestationInvalid(
            f"unexpected attestation root <{etree.QName(root_el).localname}> "
            "(expected <dgml-attestation> in the dgml attestation namespace)"
        )

    file_id = root_el.get("file-id")
    if not file_id:
        raise AttestationInvalid("attestation file is missing the 'file-id' attribute")
    docset_id = root_el.get("docset-id")  # absent → file-only version

    mr = root_el.find(q("merkle-root"))
    if mr is None or not (mr.text or "").strip():
        raise AttestationInvalid("attestation file is missing a non-empty <merkle-root>")
    algorithm = mr.get("algorithm", _HASH_ALGORITHM)  # absent → assume sha256
    if algorithm != _HASH_ALGORITHM:
        raise AttestationInvalid(
            f"unsupported hash algorithm {algorithm!r} (only {_HASH_ALGORITHM!r} is supported)"
        )
    root_hash = mr.text.strip()

    arts = root_el.find(q("artifacts"))
    if arts is None:
        raise AttestationInvalid("attestation file is missing the <artifacts> element")

    entries: list[AttestationEntry] = []

    source_el = arts.find(q("source"))
    if source_el is not None:
        entries.append(AttestationEntry("source", ArtifactKind.BINARY, None, _rel_text(source_el)))

    entries.extend(
        _read_page_group(arts, q("page-images"), q("page-image"), "page_image", ArtifactKind.BINARY)
    )

    full_schema_el = arts.find(q("full-schema"))
    if full_schema_el is not None:
        entries.append(
            AttestationEntry("full_schema", ArtifactKind.BINARY, None, _rel_text(full_schema_el))
        )

    extraction_schema_el = arts.find(q("extraction-schema"))
    if extraction_schema_el is not None:
        entries.append(
            AttestationEntry(
                "extraction_schema", ArtifactKind.BINARY, None, _rel_text(extraction_schema_el)
            )
        )

    dgml_el = arts.find(q("dgml-xml"))
    if dgml_el is not None:
        entries.append(AttestationEntry("dgml_xml", ArtifactKind.XML, None, _rel_text(dgml_el)))

    if not entries:
        raise AttestationInvalid("attestation file declares no artifacts")

    return AttestationInventory(
        file_id=file_id,
        docset_id=docset_id,
        root=root_hash,
        entries=tuple(entries),
    )


def collect_from_attestation(directory: Path) -> FileVersion:
    """Re-hash the artifacts the attestation file references, in canonical order.

    The returned :class:`FileVersion` is what :func:`attest_file_version`
    rolls into a root for verification. Page ordering follows the inventory's
    ``number`` attributes; filenames are opaque. Raises
    :class:`AttestationInvalid` if the inventory references an artifact that
    isn't present on disk.
    """
    return _collect_from_attestation(directory, read_attestation(directory))


def verify_attestation_dir(directory: Path) -> VerifyResult:
    """Verify a bundle directory against the root recorded in its attestation file.

    Reads ``META-INF/dgml-attestation.xml``, re-hashes the referenced artifacts
    in canonical order, recomputes the Merkle root, and compares it to the
    recorded root. Returns a :class:`VerifyResult` (``valid`` is the boolean
    outcome). Raises :class:`AttestationInvalid` for a structurally broken
    bundle (missing/malformed attestation file, missing artifact) and the same
    content errors as collection (e.g. :class:`CorruptMetadata` for a schema
    JSON that no longer parses).
    """
    inventory = read_attestation(directory)
    version = _collect_from_attestation(directory, inventory)
    computed = attest_file_version(version).root
    return VerifyResult(
        file_id=inventory.file_id,
        docset_id=inventory.docset_id,
        expected_root=inventory.root,
        computed_root=computed,
        valid=computed == inventory.root,
        slot_ids=tuple(a.slot_id for a in version.artifacts),
    )


def verify_bundle(path: Path) -> VerifyResult:
    """Verify a DGMLX bundle given either its loose directory or its ``.dgmlx`` archive.

    A directory is verified in place (see :func:`verify_attestation_dir`). A
    ``.dgmlx`` archive is extracted to a temporary directory, verified, and the
    temporary copy removed. Raises :class:`AttestationInvalid` if ``path`` is
    neither, does not carry the ``.dgmlx`` extension, or is not a readable zip
    archive.
    """
    if path.is_dir():
        return verify_attestation_dir(path)
    if path.is_file():
        if path.suffix.lower() != DGMLX_EXTENSION:
            raise AttestationInvalid(
                f"{path} is not a {DGMLX_EXTENSION} archive (expected a {DGMLX_EXTENSION} "
                "file or an unpacked bundle directory)"
            )
        try:
            with tempfile.TemporaryDirectory(prefix="dgmlx-verify-") as tmp:
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(tmp)  # zipfile sanitizes member paths (no zip-slip)
                return verify_attestation_dir(Path(tmp))
        except zipfile.BadZipFile as exc:
            raise AttestationInvalid(f"{path} is not a valid .dgmlx (zip) archive: {exc}") from exc
    raise AttestationInvalid(f"no DGMLX bundle directory or .dgmlx archive at {path}")


# --- private helpers --------------------------------------------------------


def _attestation_qname(local: str) -> str:
    """Clark-notation tag for ``local`` in the dgml attestation namespace."""
    return f"{{{ATTESTATION_NS}}}{local}"


def _export_rel_path(ref: ArtifactRef) -> str:
    """POSIX-style bundle path for ``ref``. Ordering never depends on it."""
    if ref.slot_id == "source":
        return f"source/{ref.path.name}"
    if ref.slot_id.startswith("page_image["):
        return f"page_images/{ref.path.name}"
    # full-schema.rnc, extraction-schema.rnc, and <stem>.dgml.xml sit at the bundle root.
    return ref.path.name


def _rel_text(el: etree._Element) -> str:
    rel = (el.text or "").strip()
    if not rel:
        raise AttestationInvalid(f"attestation <{etree.QName(el).localname}> has no path text")
    return rel


def _read_page_group(
    arts: etree._Element,
    group_tag: str,
    item_tag: str,
    slot_prefix: str,
    kind: ArtifactKind,
) -> list[AttestationEntry]:
    """Parse a ``<page-images>`` group into ordered entries.

    ``group_tag`` / ``item_tag`` are Clark-notation namespaced tags. Items are
    sorted by their ``number`` attribute — *not* by document order or filename —
    which is what makes the bundle filename-independent. Missing, non-integer,
    or duplicate ``number`` values are errors.
    """
    group = arts.find(group_tag)
    if group is None:
        return []
    label = etree.QName(item_tag).localname
    seen: set[int] = set()
    items: list[tuple[int, str]] = []
    for el in group.findall(item_tag):
        raw = el.get("number")
        if raw is None:
            raise AttestationInvalid(f"<{label}> is missing the required 'number' attribute")
        try:
            num = int(raw)
        except ValueError as exc:
            raise AttestationInvalid(f"<{label}> has non-integer number {raw!r}") from exc
        if num in seen:
            raise AttestationInvalid(f"duplicate <{label}> number {num}")
        seen.add(num)
        items.append((num, _rel_text(el)))
    items.sort(key=lambda pair: pair[0])
    return [AttestationEntry(f"{slot_prefix}[{num}]", kind, num, rel) for num, rel in items]


def _collect_from_attestation(directory: Path, inventory: AttestationInventory) -> FileVersion:
    refs: list[ArtifactRef] = []
    for entry in inventory.entries:
        abs_path = directory / entry.rel_path
        if not abs_path.exists():
            raise AttestationInvalid(
                f"attestation file references missing artifact: {entry.rel_path}"
            )
        if entry.kind is ArtifactKind.BINARY:
            refs.append(_binary_ref(entry.slot_id, abs_path, number=entry.number))
        else:
            refs.append(_xml_ref(entry.slot_id, abs_path, number=entry.number))
    return FileVersion(
        file_id=inventory.file_id,
        docset_id=inventory.docset_id,
        artifacts=tuple(refs),
    )


def _binary_ref(slot_id: str, path: Path, *, number: int | None = None) -> ArtifactRef:
    return ArtifactRef(slot_id, path, ArtifactKind.BINARY, sha256_file(path), number)


def _xml_ref(slot_id: str, path: Path, *, number: int | None = None) -> ArtifactRef:
    return ArtifactRef(slot_id, path, ArtifactKind.XML, _hash_xml_file(path), number)


def _hash_xml_file(path: Path) -> str:
    """Parse ``path`` as XML and return :func:`dgml.merkle.merkle_root`."""
    try:
        tree = etree.parse(str(path))
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"{path} is not well-formed XML: {exc}") from exc
    return merkle_root(tree.getroot())
