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

"""Unit tests for the OPC packaging helpers (content types, rels, zip)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from dgml_core.opc import (
    CONTENT_TYPES_FILENAME,
    PackageRelationship,
    content_type_for_ext,
    write_content_types,
    write_package_rels,
    zip_package,
)
from lxml import etree  # type: ignore[import-untyped]

_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _defaults(path: Path) -> dict[str, str]:
    root = etree.parse(str(path)).getroot()
    return {
        el.get("Extension"): el.get("ContentType") for el in root.findall(f"{{{_CT_NS}}}Default")
    }


def test_content_type_for_ext_known_and_fallback() -> None:
    assert content_type_for_ext("png") == "image/png"
    assert content_type_for_ext("PNG") == "image/png"  # case-insensitive
    assert content_type_for_ext("wat") == "application/octet-stream"  # unknown → fallback


def test_write_content_types_one_default_per_extension(tmp_path: Path) -> None:
    parts = [
        "source/contract.docx",
        "page_images/page_1.png",
        "schema.json",
        "META-INF/dgml-attestation.xml",
        "_rels/.rels",
    ]
    path = write_content_types(tmp_path, parts)
    assert path.name == CONTENT_TYPES_FILENAME

    defaults = _defaults(path)
    # One Default per distinct extension, deduped; 'rels' always present even
    # though the dotfile name '_rels/.rels' has no pathlib suffix.
    assert defaults == {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "png": "image/png",
        "json": "application/json",
        "xml": "application/xml",
        "rels": "application/vnd.openxmlformats-package.relationships+xml",
    }
    # No per-part Override is emitted — Defaults cover every part.
    assert etree.parse(str(path)).getroot().find(f"{{{_CT_NS}}}Override") is None


def test_write_package_rels_percent_encodes_targets(tmp_path: Path) -> None:
    main_type = "http://dgml.io/ns/relationships/main-document"
    att_type = "http://dgml.io/ns/relationships/attestation"
    rels = [
        PackageRelationship("rId1", main_type, "source/My Doc.docx"),
        PackageRelationship("rId2", att_type, "META-INF/dgml-attestation.xml"),
    ]
    path = write_package_rels(tmp_path, rels)
    assert path == tmp_path / "_rels" / ".rels"

    root = etree.parse(str(path)).getroot()
    found = {
        el.get("Id"): (el.get("Type"), el.get("Target"))
        for el in root.findall(f"{{{_REL_NS}}}Relationship")
    }
    # Spaces in the source name are percent-encoded; '/' separators preserved.
    assert found["rId1"] == (main_type, "source/My%20Doc.docx")
    assert found["rId2"] == (att_type, "META-INF/dgml-attestation.xml")


def test_zip_package_root_layout_and_no_self_inclusion(tmp_path: Path) -> None:
    out_dir = tmp_path / "bundle"
    (out_dir / "source").mkdir(parents=True)
    (out_dir / "source" / "a.docx").write_bytes(b"docx")
    (out_dir / "manifest.xml").write_bytes(b"<x/>")
    (out_dir / CONTENT_TYPES_FILENAME).write_bytes(b"<Types/>")

    archive = out_dir / "a.dgmlx"
    parts = ["source/a.docx", "manifest.xml", CONTENT_TYPES_FILENAME]
    zip_package(out_dir, archive, parts)

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    # [Content_Types].xml is stored first; arcnames are package-root-relative;
    # the archive never packs itself in even though it lives inside out_dir.
    assert names[0] == CONTENT_TYPES_FILENAME
    assert set(names) == {CONTENT_TYPES_FILENAME, "source/a.docx", "manifest.xml"}
    assert "a.dgmlx" not in names
