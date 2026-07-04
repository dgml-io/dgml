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

"""Tests for ``dgml_core.file_attestation`` — file-version Merkle attestation.

Builds artifacts on disk under the ``workspace`` fixture (no real PDF
pipeline needed — the attestation cares about bytes, not semantics) and
exercises:

- canonical slot ordering and discovery
- per-kind hashing (binary / XML)
- root recomputation matches the same RFC 6962 Merkle the XML
  attestation uses
- "missing artifact" is a smaller version (no error)
- structurally invalid input raises; tampered leaf returns ``False``
- XML leaves use ``merkle_root`` — XML reformatting that preserves
  exclusive C14N preserves the leaf hash
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from dgml_core.errors import (
    AttestationInvalid,
    CorruptMetadata,
    DocSetNotFound,
    FileNotFound,
    InvalidArgument,
)
from dgml_core.file_attestation import (
    ATTESTATION_NS,
    METADATA_DIRNAME,
    METADATA_FILENAME,
    ArtifactKind,
    ArtifactRef,
    FileAttestation,
    attest_file,
    attest_file_version,
    collect_file_version,
    collect_from_attestation,
    export_attestation,
    read_attestation,
    verify_attestation_dir,
    verify_bundle,
    verify_file_version,
)
from dgml_core.merkle import merkle_root, merkle_root_from_hashes
from dgml_core.storage import Workspace, write_json_atomic
from lxml import etree  # type: ignore[import-untyped]

# --- minimal on-disk fixture helpers ----------------------------------------


def _seed_file(
    ws: Workspace,
    file_id: str,
    *,
    pages: int,
    pdf_name: str = "doc.pdf",
    pdf_converter: str | None = None,
) -> None:
    """Drop a fake file directory: file.json, a source placeholder, page
    images and page text. Bytes are arbitrary — attestation hashes the
    bytes, it doesn't validate they're a real PDF/JPEG."""
    file_dir = ws.file_dir(file_id)
    file_dir.mkdir(parents=True)
    (file_dir / pdf_name).write_bytes(b"%PDF-1.4\n%fake-pdf-bytes\n")
    write_json_atomic(
        ws.file_json_path(file_id),
        {
            "id": file_id,
            "original_path": f"/src/{pdf_name}",
            "original_filename": pdf_name,
            "sha256": "0" * 64,
            "added_at": "2026-06-05T00:00:00Z",
            "page_count": pages,
            "text_mode": "digital",
            "page_image_dpi": 300,
            "page_image_renderer": "ghostscript",
            "pdf_converter": pdf_converter,
        },
    )
    pages_dir = ws.file_pages_dir(file_id)
    pages_dir.mkdir(parents=True)
    text_dir = ws.file_text_dir(file_id)
    text_dir.mkdir(parents=True)
    for n in range(1, pages + 1):
        (pages_dir / f"page_{n}.png").write_bytes(f"fake-png-page-{n}".encode())
        page_text = {
            "file_id": file_id,
            "page": n,
            "words": [{"t": f"w{n}", "l": [0, 0, 1, 1]}],
        }
        (text_dir / f"page_{n}.json").write_text(json.dumps(page_text), encoding="utf-8")


_FULL_SCHEMA_RNC = "# Role: root\nstart = element dg:chunk { text }\n"


def _seed_docset(
    ws: Workspace,
    docset_id: str,
    *,
    full_schema: str | None = None,
    extraction_schema: str | None = None,
) -> None:
    ws.docset_dir(docset_id).mkdir(parents=True)
    write_json_atomic(
        ws.docset_dir(docset_id) / "docset.json",
        {"id": docset_id, "name": "Test", "description": "", "key_questions": []},
    )
    if full_schema is not None:
        # The attestation "full_schema" slot is the generation tag schema
        # (full-schema.rnc, RELAX NG Compact), hashed as raw bytes.
        ws.docset_full_schema_path(docset_id).write_text(full_schema, encoding="utf-8")
    if extraction_schema is not None:
        # The attestation "extraction_schema" slot is the grounded extraction
        # schema (RELAX NG Compact), hashed as raw bytes.
        ws.docset_schema_path(docset_id).write_text(extraction_schema, encoding="utf-8")


def _seed_dgml_xml(
    ws: Workspace, docset_id: str, pdf_stem: str, xml: bytes, file_id: str = "f001"
) -> Path:
    path = ws.file_dgml_xml_path(docset_id, file_id, pdf_stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(xml)
    return path


_ATTESTATION_REL = f"{METADATA_DIRNAME}/{METADATA_FILENAME}"


def _attestation_path(directory: Path) -> Path:
    """Path to a bundle's ``META-INF/dgml-attestation.xml``."""
    return directory / METADATA_DIRNAME / METADATA_FILENAME


def _write_raw_attestation(directory: Path, xml: str) -> Path:
    """Write a hand-authored attestation file (for error-contract tests).

    ``xml`` is the full literal document; the caller supplies the
    ``xmlns`` so a test can exercise namespace/structure defects.
    """
    path = _attestation_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    return path


def _export(
    ws: Workspace, file_id: str, out_dir: Path, docset_id: str | None = None
) -> tuple[FileAttestation, Path | None, Path | None]:
    """Export keeping the unpacked bundle, so the loose files land in ``out_dir``.

    Most tests below inspect or tamper with the loose tree; ``--unpacked`` is the
    mode that produces it (and no archive). Default (archive-only) export has
    its own tests.
    """
    return export_attestation(ws, file_id, out_dir, docset_id, unpacked=True)


# --- slot discovery + ordering ----------------------------------------------


def test_file_only_collects_source_and_pages_in_slot_order(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=2)
    version = collect_file_version(workspace, "f001")
    assert version.docset_id is None
    slot_ids = [a.slot_id for a in version.artifacts]
    # page_text/ is deliberately not attested — only source + page images.
    assert slot_ids == ["source", "page_image[1]", "page_image[2]"]
    kinds = [a.kind for a in version.artifacts]
    assert kinds == [ArtifactKind.BINARY, ArtifactKind.BINARY, ArtifactKind.BINARY]


def test_page_text_files_are_never_attested(workspace: Workspace) -> None:
    """The token files under `page_text/` exist on disk but are intentionally
    excluded from the file version — no `page_text[...]` slot ever appears."""
    _seed_file(workspace, "f001", pages=2)
    assert workspace.file_text_dir("f001").exists()  # token files are on disk
    version = collect_file_version(workspace, "f001")
    assert not any(a.slot_id.startswith("page_text[") for a in version.artifacts)


def test_page_files_ordered_numerically_not_lexicographically(workspace: Workspace) -> None:
    """page_10 must sort *after* page_2, not before."""
    _seed_file(workspace, "f001", pages=12)
    version = collect_file_version(workspace, "f001")
    image_slots = [a.slot_id for a in version.artifacts if a.slot_id.startswith("page_image[")]
    assert image_slots == [f"page_image[{n}]" for n in range(1, 13)]


def test_docset_adds_schema_and_dgml_xml_in_slot_order(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1, pdf_name="contract.pdf")
    _seed_docset(
        workspace,
        "ds01",
        full_schema=_FULL_SCHEMA_RNC,
        extraction_schema='namespace docset = "http://www.dgml.io/acme/x#"\n',
    )
    _seed_dgml_xml(workspace, "ds01", "contract", b"<dg:chunk xmlns:dg='http://x'><a/></dg:chunk>")

    version = collect_file_version(workspace, "f001", "ds01")
    assert version.docset_id == "ds01"
    slot_ids = [a.slot_id for a in version.artifacts]
    # extraction_schema sits between the generation schema and the DGML XML.
    assert slot_ids == ["source", "page_image[1]", "full_schema", "extraction_schema", "dgml_xml"]
    schema_slot, extraction_slot, xml_slot = version.artifacts[-3:]
    assert (schema_slot.kind, extraction_slot.kind, xml_slot.kind) == (
        ArtifactKind.BINARY,  # RNC is raw text, hashed by bytes
        ArtifactKind.BINARY,
        ArtifactKind.XML,
    )


def test_extraction_schema_slot_present_without_generation_schema(workspace: Workspace) -> None:
    """The extraction schema is an independent slot — it appears even when the
    generation full-schema.rnc is absent, still ordered ahead of the DGML XML."""
    _seed_file(workspace, "f001", pages=1, pdf_name="contract.pdf")
    _seed_docset(
        workspace,
        "ds01",
        extraction_schema='namespace docset = "http://www.dgml.io/acme/x#"\n',
    )
    _seed_dgml_xml(workspace, "ds01", "contract", b"<dg:chunk xmlns:dg='http://x'><a/></dg:chunk>")

    slot_ids = [a.slot_id for a in collect_file_version(workspace, "f001", "ds01").artifacts]
    assert slot_ids == ["source", "page_image[1]", "extraction_schema", "dgml_xml"]


def test_missing_artifacts_silently_excluded(workspace: Workspace) -> None:
    """Page images dir absent but source present → version has the present
    slots and nothing else; not an error."""
    _seed_file(workspace, "f001", pages=2)
    # Remove the page-images dir to simulate rendering not having run yet.
    import shutil

    shutil.rmtree(workspace.file_pages_dir("f001"))
    version = collect_file_version(workspace, "f001")
    slot_ids = [a.slot_id for a in version.artifacts]
    assert slot_ids == ["source"]


def test_docset_with_no_schema_or_xml_yields_file_only_slots(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1)
    _seed_docset(workspace, "ds01")  # no schema, no dgml_xml
    version = collect_file_version(workspace, "f001", "ds01")
    assert [a.slot_id for a in version.artifacts] == ["source", "page_image[1]"]


# --- error contracts --------------------------------------------------------


def test_empty_file_id_raises_invalid_argument(workspace: Workspace) -> None:
    with pytest.raises(InvalidArgument):
        collect_file_version(workspace, "  ")


def test_empty_docset_id_raises_invalid_argument(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1)
    with pytest.raises(InvalidArgument):
        collect_file_version(workspace, "f001", "  ")


def test_missing_file_raises_file_not_found(workspace: Workspace) -> None:
    with pytest.raises(FileNotFound):
        collect_file_version(workspace, "nope")


def test_missing_docset_raises_docset_not_found(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1)
    with pytest.raises(DocSetNotFound):
        collect_file_version(workspace, "f001", "nope")


def test_empty_version_raises_value_error(workspace: Workspace) -> None:
    """File dir exists but completely empty (no PDF, no pages, no text)."""
    workspace.file_dir("f001").mkdir(parents=True)
    write_json_atomic(
        workspace.file_json_path("f001"),
        {
            "id": "f001",
            "original_path": "/src/doc.pdf",
            "original_filename": "doc.pdf",
            "sha256": "0" * 64,
            "added_at": "2026-06-05T00:00:00Z",
            "page_count": 0,
            "text_mode": "digital",
        },
    )
    # file.json exists but no PDF, no page_images dir, no page_text dir.
    with pytest.raises(ValueError, match="no artifacts found"):
        collect_file_version(workspace, "f001")


def test_corrupt_file_json_raises_corrupt_metadata(workspace: Workspace) -> None:
    workspace.file_dir("f001").mkdir(parents=True)
    workspace.file_json_path("f001").write_text("{not json", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        collect_file_version(workspace, "f001")


def test_schema_json_is_not_attested(workspace: Workspace) -> None:
    """schema.json is superseded by full-schema.rnc: its presence creates no
    slot, and even a corrupt one is never read during collection."""
    _seed_file(workspace, "f001", pages=1)
    _seed_docset(workspace, "ds01", full_schema=_FULL_SCHEMA_RNC)
    workspace.docset_generation_schema_path("ds01").write_text("not json", encoding="utf-8")
    slot_ids = [a.slot_id for a in collect_file_version(workspace, "f001", "ds01").artifacts]
    assert "schema" not in slot_ids
    assert "full_schema" in slot_ids


def test_unexpected_page_filename_raises(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1)
    # Stray file matching the ``page_*.png`` glob but not the strict
    # ``page_<digits>.png`` shape the attestation requires for ordering.
    (workspace.file_pages_dir("f001") / "page_thumb.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected file name"):
        collect_file_version(workspace, "f001")


def test_malformed_dgml_xml_raises_value_error(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1, pdf_name="doc.pdf")
    _seed_docset(workspace, "ds01")
    _seed_dgml_xml(workspace, "ds01", "doc", b"<not-closed>")
    with pytest.raises(ValueError, match="not well-formed XML"):
        collect_file_version(workspace, "f001", "ds01")


def test_attest_empty_version_raises() -> None:
    """attest_file_version's own guard, independent of discovery."""
    from dgml_core.file_attestation import FileVersion  # local import — keep module imports minimal

    with pytest.raises(ValueError, match="empty file version"):
        attest_file_version(FileVersion(file_id="f001", docset_id=None, artifacts=()))


# --- per-kind hashing -------------------------------------------------------


def test_binary_leaf_hash_is_sha256_of_file_bytes(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1)
    version = collect_file_version(workspace, "f001")
    source_ref = next(a for a in version.artifacts if a.slot_id == "source")
    expected = hashlib.sha256(source_ref.path.read_bytes()).hexdigest()
    assert source_ref.leaf_hash == expected


def test_xml_leaf_hash_is_merkle_root_of_parsed_tree(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1, pdf_name="doc.pdf")
    _seed_docset(workspace, "ds01")
    xml_bytes = b"<dg:chunk xmlns:dg='http://x'><a>hi</a><b/></dg:chunk>"
    _seed_dgml_xml(workspace, "ds01", "doc", xml_bytes)
    version = collect_file_version(workspace, "f001", "ds01")
    xml_ref = next(a for a in version.artifacts if a.slot_id == "dgml_xml")
    expected = merkle_root(etree.fromstring(xml_bytes))
    assert xml_ref.leaf_hash == expected


def test_xml_leaf_is_invariant_to_exclusive_c14n_equivalent_serializations(
    workspace: Workspace,
) -> None:
    """Two XML serializations that exclusive C14N normalizes to the same
    bytes hash to the same dgml_xml leaf — and the hash is independent of
    moving xmlns declarations (exclusive C14N's defining property)."""
    _seed_file(workspace, "f001", pages=1, pdf_name="doc.pdf")
    _seed_docset(workspace, "ds01")
    xml_a = b'<dg:chunk xmlns:dg="http://x"><dg:a b="1" c="2"/></dg:chunk>'
    # Attribute order differs; comments added (stripped by c14n); namespace
    # decl moved inward (the canonical form is identical post-c14n).
    xml_b = b'<!--hi--><dg:chunk xmlns:dg="http://x"><dg:a c="2" b="1"/></dg:chunk>'
    _seed_dgml_xml(workspace, "ds01", "doc", xml_a)
    hash_a = collect_file_version(workspace, "f001", "ds01").artifacts[-1].leaf_hash
    _seed_dgml_xml(workspace, "ds01", "doc", xml_b)
    hash_b = collect_file_version(workspace, "f001", "ds01").artifacts[-1].leaf_hash
    assert hash_a == hash_b


# --- Merkle root + verify roundtrip -----------------------------------------


def test_root_matches_manual_merkle_over_leaf_hashes(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=2)
    att = attest_file(workspace, "f001")
    expected = merkle_root_from_hashes([a.leaf_hash for a in att.leaves])
    assert att.root == expected


def test_single_artifact_root_equals_leaf_hash(workspace: Workspace) -> None:
    """RFC 6962: a 1-leaf tree has root == leaf (no pairing). Matches the
    merkle.py contract."""
    workspace.file_dir("f001").mkdir(parents=True)
    (workspace.file_dir("f001") / "doc.pdf").write_bytes(b"only-pdf")
    write_json_atomic(
        workspace.file_json_path("f001"),
        {
            "id": "f001",
            "original_path": "/src/doc.pdf",
            "original_filename": "doc.pdf",
            "sha256": "0" * 64,
            "added_at": "2026-06-05T00:00:00Z",
            "page_count": 1,
            "text_mode": "digital",
        },
    )
    att = attest_file(workspace, "f001")
    assert len(att.leaves) == 1
    assert att.root == att.leaves[0].leaf_hash


def test_verify_roundtrip_returns_true_when_unchanged(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=2)
    _seed_docset(workspace, "ds01", full_schema=_FULL_SCHEMA_RNC)
    _seed_dgml_xml(workspace, "ds01", "doc", b"<dg:chunk xmlns:dg='http://x'><a/></dg:chunk>")
    att = attest_file(workspace, "f001", "ds01")
    assert verify_file_version(workspace, att) is True


def test_verify_returns_false_on_pdf_tamper(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1)
    att = attest_file(workspace, "f001")
    (workspace.file_dir("f001") / "doc.pdf").write_bytes(b"%PDF-1.4\n%TAMPERED\n")
    assert verify_file_version(workspace, att) is False


def test_page_text_tamper_does_not_affect_verification(workspace: Workspace) -> None:
    """`page_text/` is not attested, so editing a token file leaves the root
    unchanged — verification still passes."""
    _seed_file(workspace, "f001", pages=1)
    att = attest_file(workspace, "f001")
    text_path = workspace.file_text_dir("f001") / "page_1.json"
    parsed = json.loads(text_path.read_text())
    parsed["words"][0]["t"] = "TAMPERED"
    text_path.write_text(json.dumps(parsed), encoding="utf-8")
    assert verify_file_version(workspace, att) is True


def test_verify_returns_false_on_schema_tamper(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1)
    _seed_docset(workspace, "ds01", full_schema=_FULL_SCHEMA_RNC)
    att = attest_file(workspace, "f001", "ds01")
    workspace.docset_full_schema_path("ds01").write_text(
        _FULL_SCHEMA_RNC + "# tampered\n", encoding="utf-8"
    )
    assert verify_file_version(workspace, att) is False


def test_verify_returns_false_on_extraction_schema_tamper(workspace: Workspace) -> None:
    """The extraction schema is an attested slot: editing extraction-schema.rnc
    after attestation flips verification to False."""
    _seed_file(workspace, "f001", pages=1)
    _seed_docset(
        workspace,
        "ds01",
        extraction_schema='namespace docset = "http://www.dgml.io/acme/x#"\n',
    )
    att = attest_file(workspace, "f001", "ds01")
    workspace.docset_schema_path("ds01").write_text(
        'namespace docset = "http://www.dgml.io/acme/y#"\n', encoding="utf-8"
    )
    assert verify_file_version(workspace, att) is False


def test_verify_raises_on_slot_inventory_mismatch(workspace: Workspace) -> None:
    """Adding or removing an artifact changes the *structure* — that's
    not tampering, it's a different version. Surface it loudly."""
    _seed_file(workspace, "f001", pages=2)
    att = attest_file(workspace, "f001")
    # Delete one page image — slot count drops by 1.
    (workspace.file_pages_dir("f001") / "page_2.png").unlink()
    with pytest.raises(ValueError, match="slot inventory differs"):
        verify_file_version(workspace, att)


def test_verify_raises_when_file_was_deleted(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1)
    att = attest_file(workspace, "f001")
    import shutil

    shutil.rmtree(workspace.file_dir("f001"))
    with pytest.raises(FileNotFound):
        verify_file_version(workspace, att)


def test_two_independent_file_versions_have_distinct_roots(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1, pdf_name="a.pdf")
    _seed_file(workspace, "f002", pages=1, pdf_name="b.pdf")
    # The fixture writes identical placeholder bytes; give the two sources
    # distinct content so their (source-only + page-image) roots must differ.
    (workspace.file_dir("f001") / "a.pdf").write_bytes(b"%PDF-1.4\n%aaa\n")
    (workspace.file_dir("f002") / "b.pdf").write_bytes(b"%PDF-1.4\n%bbb\n")
    assert attest_file(workspace, "f001").root != attest_file(workspace, "f002").root


def test_same_file_with_and_without_docset_differs(workspace: Workspace) -> None:
    """Same file, but the docset-scoped version pulls in schema +
    dgml_xml — two more leaves → different root."""
    _seed_file(workspace, "f001", pages=1, pdf_name="doc.pdf")
    _seed_docset(workspace, "ds01", full_schema=_FULL_SCHEMA_RNC)
    _seed_dgml_xml(workspace, "ds01", "doc", b"<dg:chunk xmlns:dg='http://x'><a/></dg:chunk>")
    file_only = attest_file(workspace, "f001")
    with_docset = attest_file(workspace, "f001", "ds01")
    assert file_only.root != with_docset.root
    # File-only slots are the prefix of the docset-scoped version.
    file_only_slots = [a.slot_id for a in file_only.leaves]
    with_docset_slots = [a.slot_id for a in with_docset.leaves]
    assert with_docset_slots[: len(file_only_slots)] == file_only_slots


def test_attestation_is_deterministic_across_calls(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=3)
    a = attest_file(workspace, "f001")
    b = attest_file(workspace, "f001")
    assert a.root == b.root
    assert tuple(x.leaf_hash for x in a.leaves) == tuple(x.leaf_hash for x in b.leaves)


# --- ArtifactRef is a value type --------------------------------------------


def test_artifact_ref_is_frozen_and_hashable() -> None:
    """Frozen dataclasses can be set keys / dict keys — useful for
    callers building 'changed-slot' diffs across two attestations."""
    from dataclasses import FrozenInstanceError

    ref = ArtifactRef("source", Path("/tmp/x.pdf"), ArtifactKind.BINARY, "0" * 64)
    with pytest.raises(FrozenInstanceError):
        ref.leaf_hash = "ff" * 32  # type: ignore[misc]
    assert hash(ref) == hash(ref)


def test_file_attestation_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    att = FileAttestation(file_id="f001", docset_id=None, leaves=(), root="x")
    with pytest.raises(FrozenInstanceError):
        att.root = "y"  # type: ignore[misc]


# --- page numbers are recorded on refs --------------------------------------


def test_page_refs_carry_their_number(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=2)
    version = collect_file_version(workspace, "f001")
    by_slot = {a.slot_id: a for a in version.artifacts}
    assert by_slot["source"].number is None
    assert by_slot["page_image[1]"].number == 1
    assert by_slot["page_image[2]"].number == 2


# --- portable bundle: export + manifest -------------------------------------


def test_export_writes_artifacts_and_attestation(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=2, pdf_name="contract.pdf")
    out_dir = workspace.root.parent / "bundle"
    _, attestation_path, _ = _export(workspace, "f001", out_dir)

    assert attestation_path == _attestation_path(out_dir)
    assert attestation_path.exists()
    # Artifacts copied into the bundle under their slot subdirs.
    assert (out_dir / "source" / "contract.pdf").exists()
    assert (out_dir / "page_images" / "page_1.png").exists()
    assert (out_dir / "page_images" / "page_2.png").exists()
    # The token files under page_text/ are never bundled.
    assert not (out_dir / "page_text").exists()


def test_export_unpacked_writes_loose_opc_parts_no_archive(workspace: Workspace) -> None:
    """--unpacked writes the OPC package as loose files — [Content_Types].xml +
    _rels/.rels naming the source as main document and the attestation — and
    produces no .dgmlx archive. The loose tree verifies as a directory."""
    _seed_file(workspace, "f001", pages=2, pdf_name="My Contract.docx")
    out_dir = workspace.root.parent / "bundle"
    _, attestation_path, archive_path = _export(workspace, "f001", out_dir)

    # --unpacked: loose files, no archive.
    assert archive_path is None
    assert attestation_path == _attestation_path(out_dir)
    assert not any(p.suffix == ".dgmlx" for p in out_dir.iterdir())

    # [Content_Types].xml covers every present extension.
    ct_root = etree.parse(str(out_dir / "[Content_Types].xml")).getroot()
    ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    exts = {el.get("Extension") for el in ct_root.findall(f"{{{ns}}}Default")}
    # docx source + png page images + the .xml attestation file + .rels. No
    # 'json' — page_text is not bundled and this file-only export has no schema.
    assert {"docx", "png", "xml", "rels"} <= exts
    assert "json" not in exts

    # _rels/.rels: main-document → the (percent-encoded) source, attestation → metadata.
    rels_root = etree.parse(str(out_dir / "_rels" / ".rels")).getroot()
    rns = "http://schemas.openxmlformats.org/package/2006/relationships"
    by_type = {
        el.get("Type"): el.get("Target") for el in rels_root.findall(f"{{{rns}}}Relationship")
    }
    assert by_type["http://dgml.io/ns/relationships/main-document"] == "source/My%20Contract.docx"
    assert by_type["http://dgml.io/ns/relationships/attestation"] == "META-INF/dgml-attestation.xml"
    # No DGML XML in a file-only export → no dgml-xml relationship.
    assert "http://dgml.io/ns/relationships/dgml-xml" not in by_type

    # The loose tree verifies as a directory.
    assert verify_attestation_dir(out_dir).valid is True


def test_export_converted_source_bundles_only_the_original(workspace: Workspace) -> None:
    """A converted (non-PDF) source export carries the original under `source/`
    and does NOT include the converted working PDF — only the original source
    is attested, named with the main-document relationship."""
    _seed_file(workspace, "f001", pages=1, pdf_name="My Contract.docx")
    out_dir = workspace.root.parent / "bundle"
    attestation, _, _ = _export(workspace, "f001", out_dir)

    # The original lands under source/; there is no pdf/ part or pdf slot.
    assert (out_dir / "source" / "My Contract.docx").exists()
    assert not (out_dir / "pdf").exists()
    assert "pdf" not in [a.slot_id for a in attestation.leaves]

    rels_root = etree.parse(str(out_dir / "_rels" / ".rels")).getroot()
    rns = "http://schemas.openxmlformats.org/package/2006/relationships"
    by_type = {
        el.get("Type"): el.get("Target") for el in rels_root.findall(f"{{{rns}}}Relationship")
    }
    assert "http://dgml.io/ns/relationships/pdf" not in by_type
    assert by_type["http://dgml.io/ns/relationships/main-document"] == "source/My%20Contract.docx"

    assert verify_attestation_dir(out_dir).valid is True


def test_export_default_writes_only_the_dgmlx_archive(workspace: Workspace) -> None:
    """By default the loose bundle is staged out of sight: ``out_dir`` ends up
    holding only the ``<stem>.dgmlx`` archive, the returned attestation path is
    ``None``, and the archive is a valid OPC zip that verifies."""
    _seed_file(workspace, "f001", pages=2, pdf_name="My Contract.docx")
    out_dir = workspace.root.parent / "bundle"
    attestation, attestation_path, archive_path = export_attestation(workspace, "f001", out_dir)

    assert attestation_path is None
    assert archive_path == out_dir / "My Contract.dgmlx"
    # Nothing loose was left behind — only the archive.
    assert list(out_dir.iterdir()) == [archive_path]
    assert not (out_dir / "source").exists()
    assert not (out_dir / "META-INF").exists()

    # The archive is a valid zip: [Content_Types].xml first, OPC parts at the
    # root, and it never packs itself in.
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
    assert names[0] == "[Content_Types].xml"
    assert "_rels/.rels" in names
    assert "source/My Contract.docx" in names
    assert "My Contract.dgmlx" not in names

    # The archive is self-sufficient for verification.
    result = verify_bundle(archive_path)
    assert result.valid is True
    assert result.expected_root == result.computed_root == attestation.root


def test_verify_bundle_on_archive_equals_directory(workspace: Workspace) -> None:
    """verify_bundle accepts a .dgmlx archive (default export) and a loose dir
    (--unpacked export) interchangeably, yielding the same verdict and root."""
    _seed_file(workspace, "f001", pages=2)
    archive_dir = workspace.root.parent / "arch"
    loose_dir = workspace.root.parent / "loose"
    _, _, archive_path = export_attestation(workspace, "f001", archive_dir)  # default → archive
    export_attestation(workspace, "f001", loose_dir, unpacked=True)  # --unpacked → dir
    assert archive_path is not None

    from_archive = verify_bundle(archive_path)
    from_dir = verify_bundle(loose_dir)
    assert from_archive.valid is from_dir.valid is True
    assert from_archive.computed_root == from_dir.computed_root


def test_verify_bundle_rejects_non_bundle(workspace: Workspace, tmp_path: Path) -> None:
    """A missing path, a wrong-extension file, and a non-zip .dgmlx all raise
    rather than mis-verifying."""
    with pytest.raises(AttestationInvalid, match="no DGMLX bundle"):
        verify_bundle(tmp_path / "does-not-exist")

    # A real zip but not named .dgmlx is rejected on the extension, before unzip.
    wrong_ext = tmp_path / "bundle.zip"
    with zipfile.ZipFile(wrong_ext, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
    with pytest.raises(AttestationInvalid, match=r"is not a \.dgmlx archive"):
        verify_bundle(wrong_ext)

    # Right extension but not actually a zip → caught at unzip time.
    junk = tmp_path / "not-a-zip.dgmlx"
    junk.write_bytes(b"not a zip archive")
    with pytest.raises(AttestationInvalid, match=r"not a valid \.dgmlx"):
        verify_bundle(junk)


def test_rels_includes_dgml_xml_relationship_when_present(workspace: Workspace) -> None:
    """A docset-scoped export that has a <stem>.dgml.xml names it in _rels/.rels
    via the dgml-xml relationship, alongside main-document and attestation."""
    _seed_file(workspace, "f001", pages=1, pdf_name="doc.pdf")
    _seed_docset(workspace, "ds01", full_schema=_FULL_SCHEMA_RNC)
    _seed_dgml_xml(workspace, "ds01", "doc", b"<dg:chunk xmlns:dg='http://x'><a/></dg:chunk>")
    out_dir = workspace.root.parent / "bundle"
    _export(workspace, "f001", out_dir, "ds01")

    rels_root = etree.parse(str(out_dir / "_rels" / ".rels")).getroot()
    rns = "http://schemas.openxmlformats.org/package/2006/relationships"
    by_type = {
        el.get("Type"): el.get("Target") for el in rels_root.findall(f"{{{rns}}}Relationship")
    }
    assert by_type["http://dgml.io/ns/relationships/main-document"] == "source/doc.pdf"
    assert by_type["http://dgml.io/ns/relationships/dgml-xml"] == "doc.dgml.xml"
    assert by_type["http://dgml.io/ns/relationships/attestation"] == "META-INF/dgml-attestation.xml"
    # Ids are unique across all three relationships.
    ids = [el.get("Id") for el in rels_root.findall(f"{{{rns}}}Relationship")]
    assert sorted(ids) == ["rId1", "rId2", "rId3"]


def test_export_writes_attestation_metadata(workspace: Workspace) -> None:
    """A non-PDF source records its converter, so the metadata file carries the
    full rendering provenance plus the Merkle root and workspace identity."""
    _seed_file(workspace, "f001", pages=2, pdf_name="contract.docx", pdf_converter="LibreOffice")
    _seed_docset(workspace, "ds01", full_schema=_FULL_SCHEMA_RNC)
    out_dir = workspace.root.parent / "bundle"
    attestation, _, _ = _export(workspace, "f001", out_dir, "ds01")

    meta_path = out_dir / METADATA_DIRNAME / METADATA_FILENAME
    assert meta_path.exists()
    root_el = etree.parse(str(meta_path)).getroot()
    assert root_el.tag == f"{{{ATTESTATION_NS}}}dgml-attestation"
    assert root_el.get("version") == "1"
    assert root_el.get("page-image-dpi") == "300"
    assert root_el.get("page-image-renderer") == "ghostscript"
    assert root_el.get("pdf-converter") == "LibreOffice"
    assert root_el.get("file-id") == "f001"
    assert root_el.get("docset-id") == "ds01"
    mr = root_el.find(f"{{{ATTESTATION_NS}}}merkle-root")
    assert mr is not None
    assert mr.text == attestation.root
    # The metadata file is not an attested artifact and doesn't disturb verification.
    assert verify_attestation_dir(out_dir).valid is True


def test_export_bundles_extraction_schema_at_root(workspace: Workspace) -> None:
    """A docset-scoped export copies extraction-schema.rnc to the bundle root,
    declares it in the attestation inventory as its own slot, covers the .rnc
    extension in [Content_Types].xml, and still verifies."""
    _seed_file(workspace, "f001", pages=1, pdf_name="doc.pdf")
    _seed_docset(
        workspace,
        "ds01",
        full_schema=_FULL_SCHEMA_RNC,
        extraction_schema='namespace docset = "http://www.dgml.io/acme/x#"\n',
    )
    _seed_dgml_xml(workspace, "ds01", "doc", b"<dg:chunk xmlns:dg='http://x'><a/></dg:chunk>")
    out_dir = workspace.root.parent / "bundle"
    _export(workspace, "f001", out_dir, "ds01")

    # The .rnc lands at the bundle root, alongside full-schema.rnc and the DGML XML.
    assert (out_dir / "extraction-schema.rnc").exists()

    # Declared in the attestation inventory as the extraction_schema slot.
    manifest = read_attestation(out_dir)
    entry = next(e for e in manifest.entries if e.slot_id == "extraction_schema")
    assert entry.rel_path == "extraction-schema.rnc"
    assert entry.kind is ArtifactKind.BINARY

    # [Content_Types].xml covers the .rnc extension.
    ct_root = etree.parse(str(out_dir / "[Content_Types].xml")).getroot()
    ns = "http://schemas.openxmlformats.org/package/2006/content-types"
    exts = {el.get("Extension") for el in ct_root.findall(f"{{{ns}}}Default")}
    assert "rnc" in exts

    assert verify_attestation_dir(out_dir).valid is True


def test_attestation_metadata_omits_converter_for_pdf_source(workspace: Workspace) -> None:
    """A PDF source was never converted, so 'pdf-converter' is absent (not empty);
    'docset-id' is likewise absent for a file-only export."""
    _seed_file(workspace, "f001", pages=1, pdf_name="doc.pdf")
    out_dir = workspace.root.parent / "bundle"
    _export(workspace, "f001", out_dir)

    root_el = etree.parse(str(out_dir / METADATA_DIRNAME / METADATA_FILENAME)).getroot()
    assert root_el.get("pdf-converter") is None
    assert root_el.get("docset-id") is None
    assert root_el.get("page-image-dpi") == "300"


def test_manifest_records_the_merkle_root_but_is_not_a_leaf(workspace: Workspace) -> None:
    """The manifest carries the root for verifiers, yet is excluded from the
    artifact set, so the recorded root equals the attestation over the
    file-side artifacts alone."""
    _seed_file(workspace, "f001", pages=2)
    out_dir = workspace.root.parent / "bundle"
    attestation, _, _ = _export(workspace, "f001", out_dir)

    manifest = read_attestation(out_dir)
    assert manifest.root == attestation.root
    assert manifest.file_id == "f001"
    assert manifest.docset_id is None
    # The algorithm isn't user-selectable, so the attestation file states it outright.
    text = _attestation_path(out_dir).read_text(encoding="utf-8")
    assert '<merkle-root algorithm="sha256">' in text
    # The attestation file is not among the referenced artifacts.
    assert all(_ATTESTATION_REL not in e.rel_path for e in manifest.entries)


def test_manifest_without_algorithm_attribute_reads_as_sha256(workspace: Workspace) -> None:
    """An attestation file that omits the 'algorithm' attribute is read as sha256."""
    _seed_file(workspace, "f001", pages=1)
    out_dir = workspace.root.parent / "bundle"
    _export(workspace, "f001", out_dir)

    att_path = _attestation_path(out_dir)
    text = att_path.read_text(encoding="utf-8")
    assert '<merkle-root algorithm="sha256">' in text
    att_path.write_text(
        text.replace('<merkle-root algorithm="sha256">', "<merkle-root>"), encoding="utf-8"
    )

    assert verify_attestation_dir(out_dir).valid is True


def test_manifest_with_unsupported_algorithm_raises(workspace: Workspace) -> None:
    """Re-hashing with the wrong algorithm would read as tampering; fail
    structurally instead."""
    _seed_file(workspace, "f001", pages=1)
    out_dir = workspace.root.parent / "bundle"
    _export(workspace, "f001", out_dir)

    att_path = _attestation_path(out_dir)
    text = att_path.read_text(encoding="utf-8")
    att_path.write_text(text.replace('algorithm="sha256"', 'algorithm="sha3-256"'))

    with pytest.raises(AttestationInvalid, match="unsupported hash algorithm 'sha3-256'"):
        read_attestation(out_dir)


def test_export_docset_scoped_records_docset_id_schema_and_xml(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1, pdf_name="doc.pdf")
    _seed_docset(workspace, "ds01", full_schema=_FULL_SCHEMA_RNC)
    _seed_dgml_xml(workspace, "ds01", "doc", b"<dg:chunk xmlns:dg='http://x'><a/></dg:chunk>")
    out_dir = workspace.root.parent / "bundle"
    _export(workspace, "f001", out_dir, "ds01")

    manifest = read_attestation(out_dir)
    assert manifest.docset_id == "ds01"
    slots = [e.slot_id for e in manifest.entries]
    assert slots == ["source", "page_image[1]", "full_schema", "dgml_xml"]
    assert (out_dir / "full-schema.rnc").exists()
    assert not (out_dir / "schema.json").exists()  # schema.json is never bundled
    assert (out_dir / "doc.dgml.xml").exists()


def test_collect_from_attestation_reproduces_export_order_and_root(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=3)
    out_dir = workspace.root.parent / "bundle"
    attestation, _, _ = _export(workspace, "f001", out_dir)

    version = collect_from_attestation(out_dir)
    assert [a.slot_id for a in version.artifacts] == [a.slot_id for a in attestation.leaves]
    assert attest_file_version(version).root == attestation.root


# --- verify roundtrip on a bundle -------------------------------------------


def test_verify_dir_true_on_untouched_bundle(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=2)
    _seed_docset(workspace, "ds01", full_schema=_FULL_SCHEMA_RNC)
    _seed_dgml_xml(workspace, "ds01", "doc", b"<dg:chunk xmlns:dg='http://x'><a/></dg:chunk>")
    out_dir = workspace.root.parent / "bundle"
    attestation, _, _ = _export(workspace, "f001", out_dir, "ds01")

    result = verify_attestation_dir(out_dir)
    assert result.valid is True
    assert result.expected_root == result.computed_root == attestation.root
    assert result.file_id == "f001"
    assert result.docset_id == "ds01"


def test_verify_dir_false_on_tampered_artifact(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=2)
    out_dir = workspace.root.parent / "bundle"
    _export(workspace, "f001", out_dir)

    (out_dir / "page_images" / "page_1.png").write_bytes(b"TAMPERED")
    result = verify_attestation_dir(out_dir)
    assert result.valid is False
    assert result.computed_root != result.expected_root


def test_editing_recorded_root_is_detected(workspace: Workspace) -> None:
    """The attestation file isn't itself attested, so the artifacts still hash
    to the real root — flipping the recorded root just makes verify report a
    mismatch."""
    _seed_file(workspace, "f001", pages=1)
    out_dir = workspace.root.parent / "bundle"
    _export(workspace, "f001", out_dir)

    att_path = _attestation_path(out_dir)
    text = att_path.read_text()
    real_root = read_attestation(out_dir).root
    att_path.write_text(text.replace(real_root, "f" * 64), encoding="utf-8")

    result = verify_attestation_dir(out_dir)
    assert result.valid is False
    assert result.expected_root == "f" * 64
    assert result.computed_root == real_root


# --- ordering is driven by `number`, not filenames --------------------------


def test_ordering_uses_number_attribute_not_lexicographic_filename(workspace: Workspace) -> None:
    """A bundle with ten pages: lexicographic filename order would put
    page_10 before page_2. The manifest's number attribute keeps the
    canonical numeric order, matching the workspace attestation."""
    _seed_file(workspace, "f001", pages=10)
    out_dir = workspace.root.parent / "bundle"
    attestation, _, _ = _export(workspace, "f001", out_dir)

    version = collect_from_attestation(out_dir)
    image_slots = [a.slot_id for a in version.artifacts if a.slot_id.startswith("page_image[")]
    assert image_slots == [f"page_image[{n}]" for n in range(1, 11)]
    assert attest_file_version(version).root == attestation.root


def test_attestation_document_order_is_irrelevant_only_number_matters(
    workspace: Workspace,
) -> None:
    """Shuffling the <page-image> elements in the inventory (while keeping
    each element's number + path) doesn't change the verification — the
    verifier sorts by number, not document order."""
    _seed_file(workspace, "f001", pages=3)
    out_dir = workspace.root.parent / "bundle"
    attestation, attestation_path, _ = _export(workspace, "f001", out_dir)

    ns = {"a": ATTESTATION_NS}
    tree = etree.parse(str(attestation_path))
    group = tree.getroot().find("a:artifacts/a:page-images", ns)
    children = list(group)
    for child in children:
        group.remove(child)
    for child in reversed(children):  # reverse document order, same numbers
        group.append(child)
    tree.write(str(attestation_path), encoding="utf-8", xml_declaration=True)

    result = verify_attestation_dir(out_dir)
    assert result.valid is True
    assert result.computed_root == attestation.root


def test_renaming_bundle_files_keeps_verification_when_inventory_tracks_them(
    workspace: Workspace,
) -> None:
    """Filename independence: rename the physical page-image files to
    opaque names and update the inventory paths to match (numbers
    unchanged). The recomputed root is identical."""
    _seed_file(workspace, "f001", pages=2)
    out_dir = workspace.root.parent / "bundle"
    attestation, attestation_path, _ = _export(workspace, "f001", out_dir)

    # Rename page_1.png -> img-a.bin, page_2.png -> img-b.bin.
    renames = {"page_images/page_1.png": "img-a.bin", "page_images/page_2.png": "img-b.bin"}
    for old_rel, new_name in renames.items():
        (out_dir / old_rel).rename(out_dir / new_name)

    ns = {"a": ATTESTATION_NS}
    tree = etree.parse(str(attestation_path))
    for el in tree.getroot().findall("a:artifacts/a:page-images/a:page-image", ns):
        n = el.get("number")
        el.text = "img-a.bin" if n == "1" else "img-b.bin"
    tree.write(str(attestation_path), encoding="utf-8", xml_declaration=True)

    result = verify_attestation_dir(out_dir)
    assert result.valid is True
    assert result.computed_root == attestation.root


# --- attestation-file error contracts ---------------------------------------


def test_read_attestation_missing_file_raises(workspace: Workspace) -> None:
    with pytest.raises(AttestationInvalid, match=r"no META-INF/dgml-attestation\.xml"):
        read_attestation(workspace.root.parent / "nope")


def test_read_attestation_malformed_xml_raises(workspace: Workspace) -> None:
    d = workspace.root.parent / "bundle"
    _write_raw_attestation(d, "<not-closed")
    with pytest.raises(AttestationInvalid, match="not well-formed"):
        read_attestation(d)


def test_read_attestation_wrong_root_element_raises(workspace: Workspace) -> None:
    d = workspace.root.parent / "bundle"
    # Right local name but no/foreign namespace is still a wrong-root error.
    _write_raw_attestation(d, "<other/>")
    with pytest.raises(AttestationInvalid, match="unexpected attestation root"):
        read_attestation(d)


def test_read_attestation_missing_root_hash_raises(workspace: Workspace) -> None:
    d = workspace.root.parent / "bundle"
    _write_raw_attestation(
        d,
        f'<dgml-attestation xmlns="{ATTESTATION_NS}" file-id="f001">'
        "<artifacts><source>x.pdf</source></artifacts></dgml-attestation>",
    )
    with pytest.raises(AttestationInvalid, match="merkle-root"):
        read_attestation(d)


def test_read_attestation_page_missing_number_raises(workspace: Workspace) -> None:
    d = workspace.root.parent / "bundle"
    _write_raw_attestation(
        d,
        f'<dgml-attestation xmlns="{ATTESTATION_NS}" file-id="f001"><merkle-root>'
        + "0" * 64
        + "</merkle-root>"
        "<artifacts><page-images><page-image>p.jpg</page-image></page-images></artifacts>"
        "</dgml-attestation>",
    )
    with pytest.raises(AttestationInvalid, match="missing the required 'number'"):
        read_attestation(d)


def test_read_attestation_duplicate_page_number_raises(workspace: Workspace) -> None:
    d = workspace.root.parent / "bundle"
    _write_raw_attestation(
        d,
        f'<dgml-attestation xmlns="{ATTESTATION_NS}" file-id="f001"><merkle-root>'
        + "0" * 64
        + "</merkle-root>"
        '<artifacts><page-images><page-image number="1">a.jpg</page-image>'
        '<page-image number="1">b.jpg</page-image></page-images></artifacts>'
        "</dgml-attestation>",
    )
    with pytest.raises(AttestationInvalid, match="duplicate"):
        read_attestation(d)


def test_collect_from_attestation_missing_artifact_raises(workspace: Workspace) -> None:
    _seed_file(workspace, "f001", pages=1)
    out_dir = workspace.root.parent / "bundle"
    _export(workspace, "f001", out_dir)
    (out_dir / "page_images" / "page_1.png").unlink()
    with pytest.raises(AttestationInvalid, match="missing artifact"):
        verify_attestation_dir(out_dir)
