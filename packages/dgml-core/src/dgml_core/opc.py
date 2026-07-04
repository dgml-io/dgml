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

"""OPC (Open Packaging Conventions, ECMA-376 Part 2) packaging for DGMLX bundles.

A DGMLX bundle is shaped as an **OPC package**: a set of *parts* (the source
document, page images, page text, schema, and the
``META-INF/dgml-attestation.xml`` attestation file) described by two well-known
parts —

- ``[Content_Types].xml`` (§10.1) — maps each part to a MIME content type, by
  file extension. It is *not* itself a part and is never listed inside itself.
- ``_rels/.rels`` (§9) — the package-level relationships part, naming the
  package's main document and its metadata by relationship-type URI.

The whole package is then zipped into a single portable ``<stem>.dgmlx``
archive (the OPC physical package is a ZIP). This module owns just the OPC
surface — content-type registry, relationships, and the zip — and is
deliberately separate from the Merkle attestation in :mod:`dgml.file_attestation`.

Part names inside the package are POSIX paths relative to the package root.
Where they appear as URI references (relationship ``Target``s) their segments
are percent-encoded per RFC 3986, so a source file named ``Master Services
Agreement.docx`` is referenced as ``source/Master%20Services%20Agreement.docx``.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from lxml import etree  # type: ignore[import-untyped]

CONTENT_TYPES_FILENAME = "[Content_Types].xml"
PACKAGE_RELS_PATH = "_rels/.rels"
DGMLX_EXTENSION = ".dgmlx"

_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_RELS_CONTENT_TYPE = "application/vnd.openxmlformats-package.relationships+xml"

# DGML relationship-type URIs (an interop contract: consumers match on these).
# The main-document relationship always points at the source original; the
# dgml-xml relationship at the generated DGML XML (when present); the
# attestation relationship at the attestation/verification part.
REL_TYPE_MAIN_DOCUMENT = "http://dgml.io/ns/relationships/main-document"
REL_TYPE_DGML_XML = "http://dgml.io/ns/relationships/dgml-xml"
REL_TYPE_ATTESTATION = "http://dgml.io/ns/relationships/attestation"

# Extension (no dot, lowercase) → MIME content type. Covers every extension a
# bundle currently holds; an unknown extension falls back to octet-stream
# rather than failing the export. ``rels`` is always declared because every OPC
# package carries ``_rels/.rels``.
_CONTENT_TYPE_BY_EXT: dict[str, str] = {
    "rels": _RELS_CONTENT_TYPE,
    "xml": "application/xml",
    "png": "image/png",
    "json": "application/json",
    "rnc": "text/plain",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "xls": "application/vnd.ms-excel",
}
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


@dataclass(frozen=True)
class PackageRelationship:
    """One ``<Relationship>`` in ``_rels/.rels``.

    ``target`` is a POSIX part path relative to the package root (it is
    percent-encoded on serialization). ``rel_id`` must be unique within the
    relationships part.
    """

    rel_id: str
    rel_type: str
    target: str


def _ext(rel_path: str) -> str:
    """Lowercase extension of a POSIX part path, without the dot.

    Handles dotfile-style part names like ``_rels/.rels`` (whose OPC extension
    is ``rels``) that :attr:`pathlib.PurePath.suffix` treats as suffix-less.
    """
    name = rel_path.rsplit("/", 1)[-1]
    _, dot, ext = name.rpartition(".")
    return ext.lower() if dot else ""


def content_type_for_ext(ext: str) -> str:
    """MIME content type for a (dotless, any-case) extension."""
    return _CONTENT_TYPE_BY_EXT.get(ext.lower(), _DEFAULT_CONTENT_TYPE)


def write_content_types(out_dir: Path, part_rel_paths: Iterable[str]) -> Path:
    """Write ``[Content_Types].xml`` covering every part's extension.

    Emits one ``<Default>`` per distinct extension present (plus ``rels``, which
    every OPC package needs for its relationships part). No per-part
    ``<Override>`` is written — every DGMLX part's content type follows from its
    extension. ``[Content_Types].xml`` itself is not a part and is not listed.
    Returns the written path.
    """
    exts = {_ext(p) for p in part_rel_paths if _ext(p)} | {"rels"}
    root = etree.Element(f"{{{_CT_NS}}}Types", nsmap={None: _CT_NS})
    for ext in sorted(exts):
        default = etree.SubElement(root, f"{{{_CT_NS}}}Default")
        default.set("Extension", ext)
        default.set("ContentType", content_type_for_ext(ext))

    path = out_dir / CONTENT_TYPES_FILENAME
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, pretty_print=True
    )
    return path


def write_package_rels(out_dir: Path, relationships: Sequence[PackageRelationship]) -> Path:
    """Write the package relationships part ``_rels/.rels``.

    Each relationship's ``Target`` is percent-encoded (segments only; the path
    separators are preserved). Returns the written path.
    """
    root = etree.Element(f"{{{_REL_NS}}}Relationships", nsmap={None: _REL_NS})
    for rel in relationships:
        el = etree.SubElement(root, f"{{{_REL_NS}}}Relationship")
        el.set("Id", rel.rel_id)
        el.set("Type", rel.rel_type)
        el.set("Target", quote(rel.target, safe="/"))

    rels_path = out_dir / PACKAGE_RELS_PATH
    rels_path.parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(root).write(
        str(rels_path), encoding="UTF-8", xml_declaration=True, pretty_print=True
    )
    return rels_path


def zip_package(out_dir: Path, archive_path: Path, part_rel_paths: Sequence[str]) -> Path:
    """Zip the named package parts under ``out_dir`` into ``archive_path``.

    ``[Content_Types].xml`` is stored first (the OPC streaming convention).
    Each part is written with its POSIX path as the archive name so the package
    root is the archive root. Only the explicitly named parts are added, so an
    ``archive_path`` that lives inside ``out_dir`` is never packed into itself.
    Returns ``archive_path``.
    """
    ordered = [CONTENT_TYPES_FILENAME] + [p for p in part_rel_paths if p != CONTENT_TYPES_FILENAME]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in ordered:
            zf.write(out_dir / rel, arcname=rel)
    return archive_path
