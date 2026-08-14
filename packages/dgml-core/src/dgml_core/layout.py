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

"""The workspace layout — the single source of truth for keys and collections.

Everything that names a piece of workspace data lives here: the document
collections, the blob key builders, and the mapping from a document
``(collection, id)`` to the file that holds it. Three consumers share it:

- the **domain layer** (``files``, ``docsets``, ``consistency``, attestation, …)
  builds keys with the ``*_key`` / ``*_prefix`` functions below;
- :class:`dgml_core.storage.Workspace` derives its ``Path``-returning helpers
  from the same builders, so a real filesystem path and a store key can never
  disagree;
- :class:`dgml_core.storage_local.LocalStore` maps keys onto those paths and
  uses :data:`DOC_LAYOUTS` to place documents and :func:`is_blob_key` to tell
  blobs from documents.

Keys are **workspace-root-relative POSIX strings** and deliberately do not
depend on a workspace root: ``files/<id>/page_images/page_1.png`` names the same
thing whether it lives on disk or in a bucket. That is why these are module
functions rather than :class:`~dgml_core.storage.Workspace` methods.

Blob keys and document paths share one namespace, an inheritance of the on-disk
layout (``files/<id>/`` holds both the source PDF and ``file.json``). So blob
membership is defined by an **allow-list** of key shapes — :data:`_BLOB_RULES` —
rather than by excluding known document names. A pattern that matches nothing is
not a blob, which makes a new document filename safe by default instead of
silently becoming attestable content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# ---------------------------------------------------------------- directories

FILES_DIR = "files"
DOCSETS_DIR = "docsets"
DOCSET_FILES_DIR = "files"  # the per-docset pair dir: docsets/<did>/files/<fid>/
PAGE_IMAGES_DIR = "page_images"
PAGE_TEXT_DIR = "page_text"

# Workspace-internal scratch, never part of the blob namespace: the clustering
# embedding cache and ``staged_write``'s staging area both live here.
CACHE_DIR = ".cache"
STAGING_DIR = "staging"
EMBEDDINGS_DIR = "embeddings"

# ------------------------------------------------------------------ filenames

# Documents (JSON managed through the document API).
FILE_MANIFEST = "file.json"
DOCSET_MANIFEST = "docset.json"
ASSIGNMENT_MANIFEST = "assignment.json"
ERRORS_FILE = "errors.json"
EXTRACTION_STATS_FILE = "extraction_stats.json"
WORKSPACE_FILE = "workspace.json"

# Bootstrap / append-only, outside both APIs.
CONFIG_FILE = "config.toml"
USAGE_FILE = "usage.jsonl"

# Blobs. The generation schema is a *blob*, not a document: it is round-tripped
# as the exact bytes ``Schema.save`` produced (which differ from this layer's
# JSON serialization — it drops ``example`` and omits the trailing newline), so
# re-serializing it through the document API would drift.
GENERATION_SCHEMA_FILE = "schema.json"
EXTRACTION_SCHEMA_FILE = "extraction-schema.rnc"
EXTRACTION_GUIDANCE_FILE = "extraction-guidance.md"
FULL_SCHEMA_FILE = "full-schema.rnc"
# The optional word-coverage report, written directly under the docset dir only
# under ``dgml --debug docset generate``.
COVERAGE_REPORT_FILE = "coverage_report.json"
PAGE_IMAGE_TEMPLATE = "page_%d.png"
PAGE_TEXT_TEMPLATE = "page_%d.json"
DGML_XML_SUFFIX = ".dgml.xml"


class Collection(StrEnum):
    """The document collections the workspace layout recognizes.

    A ``StrEnum`` so it stays interchangeable with the generic
    ``collection: str`` interface — callers may pass ``Collection.FILES`` or
    ``"files"``, and a third-party store can use any collection name it likes.
    """

    FILES = "files"
    DOCSETS = "docsets"
    WORKSPACE = "workspace"
    ERRORS = "errors"
    ASSIGNMENTS = "assignments"
    EXTRACTION_STATS = "extraction_stats"
    USAGE = "usage"  # append-only; a store may special-case it


# ------------------------------------------------------- document id encoding


def pair_id(docset_id: str, file_id: str) -> str:
    """The composite document id for a (docset, file) pair.

    ``assignments`` and ``extraction_stats`` are keyed by the pair, encoded as
    ``"<docset_id>/<file_id>"``. Kept here so callers and stores agree on the
    encoding instead of each spelling the f-string themselves."""
    return f"{docset_id}/{file_id}"


# ------------------------------------------------------------- blob key paths
#
# Every ``*_prefix`` ends with ``/`` and every ``*_key`` does not. The trailing
# slash is load-bearing, not cosmetic: ``list_blobs`` and ``delete_blobs`` match
# by string prefix, so ``docsets/d1`` would also select ``docsets/d10``'s
# contents. Bridge helpers (``staged_write`` and friends) strip it themselves.

GENERATION_CACHE_DIR = "cache"


def file_prefix(file_id: str) -> str:
    """Everything belonging to one file: ``files/<id>/``."""
    return f"{FILES_DIR}/{file_id}/"


def file_source_key(file_id: str, filename: str) -> str:
    """The stored original (or the PDF converted from it), under its own name."""
    return f"{file_prefix(file_id)}{filename}"


def file_pages_prefix(file_id: str) -> str:
    return f"{file_prefix(file_id)}{PAGE_IMAGES_DIR}/"


def file_page_image_key(file_id: str, page: int) -> str:
    return f"{file_pages_prefix(file_id)}{PAGE_IMAGE_TEMPLATE % page}"


def file_text_prefix(file_id: str) -> str:
    return f"{file_prefix(file_id)}{PAGE_TEXT_DIR}/"


def file_page_text_key(file_id: str, page: int) -> str:
    return f"{file_text_prefix(file_id)}{PAGE_TEXT_TEMPLATE % page}"


def docset_prefix(docset_id: str) -> str:
    """Everything belonging to one docset: ``docsets/<id>/``."""
    return f"{DOCSETS_DIR}/{docset_id}/"


def docset_files_prefix(docset_id: str) -> str:
    """The docset's pair directories: ``docsets/<id>/files/``."""
    return f"{docset_prefix(docset_id)}{DOCSET_FILES_DIR}/"


def docset_pair_prefix(docset_id: str, file_id: str) -> str:
    """One pair's artifacts: ``docsets/<did>/files/<fid>/``."""
    return f"{docset_files_prefix(docset_id)}{file_id}/"


def docset_extraction_schema_key(docset_id: str) -> str:
    """The grounded *extraction* schema (RELAX NG Compact)."""
    return f"{docset_prefix(docset_id)}{EXTRACTION_SCHEMA_FILE}"


def docset_guidance_key(docset_id: str) -> str:
    """The docset-level extraction guidance (free-form markdown), injected into
    the phase-1 extraction prompt for every file in the docset."""
    return f"{docset_prefix(docset_id)}{EXTRACTION_GUIDANCE_FILE}"


def docset_full_schema_key(docset_id: str) -> str:
    """``schema.json`` rendered as RNC — the whole-document schema that ships in
    DGMLX bundles and is hashed into the file attestation."""
    return f"{docset_prefix(docset_id)}{FULL_SCHEMA_FILE}"


def dgml_xml_key(docset_id: str, file_id: str, file_stem: str) -> str:
    """The DGML XML output for one file in a docset.

    Lives in the pair directory so placement never depends on the original
    filename being unique within the docset. Pass ``Path(name).stem``."""
    return f"{docset_pair_prefix(docset_id, file_id)}{file_stem}{DGML_XML_SUFFIX}"


def docset_generation_schema_key(docset_id: str) -> str:
    """The generation *tag* schema written by ``docset generate`` — the machine
    exchange format that seeds later runs via ``--schema-path``. Distinct from
    the extraction schema so the two never clobber."""
    return f"{docset_prefix(docset_id)}{GENERATION_SCHEMA_FILE}"


def dgml_grounded_xml_key(docset_id: str, file_id: str, file_stem: str) -> str:
    """The optional pre-grounded sibling of :func:`dgml_xml_key`, if a run left one."""
    return f"{docset_pair_prefix(docset_id, file_id)}{file_stem}.dgml.grounded.xml"


def pair_artifact_key(docset_id: str, file_id: str, filename: str) -> str:
    """An arbitrary named artifact in a pair directory (e.g. a grounding-stats
    sidecar written next to the DGML XML)."""
    return f"{docset_pair_prefix(docset_id, file_id)}{filename}"


def docset_coverage_report_key(docset_id: str) -> str:
    """The optional ``--debug`` word-coverage report: ``docsets/<id>/coverage_report.json``."""
    return f"{docset_prefix(docset_id)}{COVERAGE_REPORT_FILE}"


def generation_cache_prefix(docset_id: str) -> str:
    """The generation run's reloadable working cache."""
    return f"{docset_prefix(docset_id)}{GENERATION_CACHE_DIR}/"


# -------------------------------------------------------- document placement


@dataclass(frozen=True)
class DocLayout:
    """Where a document collection lands on a path-shaped store.

    ``template`` is a format string over the id parts (e.g.
    ``"files/{id}/file.json"``); ``id_parts`` names the ``/``-separated segments
    of the document id it consumes. ``glob`` (derived by replacing each
    placeholder with ``*``) enumerates the collection under the root.
    """

    template: str
    id_parts: tuple[str, ...]

    @property
    def glob(self) -> str:
        return re.sub(r"\{[^}]+\}", "*", self.template)


_FILE_DIR_T = f"{FILES_DIR}/{{id}}"
_DOCSET_DIR_T = f"{DOCSETS_DIR}/{{id}}"
_PAIR_DIR_T = f"{DOCSETS_DIR}/{{did}}/{DOCSET_FILES_DIR}/{{fid}}"

#: Per-collection placement. ``USAGE`` is absent — it is append-only and a
#: store handles it however suits it (``LocalStore`` uses ``usage.jsonl``).
DOC_LAYOUTS: dict[str, DocLayout] = {
    Collection.FILES: DocLayout(f"{_FILE_DIR_T}/{FILE_MANIFEST}", ("id",)),
    Collection.ERRORS: DocLayout(f"{_FILE_DIR_T}/{ERRORS_FILE}", ("id",)),
    Collection.DOCSETS: DocLayout(f"{_DOCSET_DIR_T}/{DOCSET_MANIFEST}", ("id",)),
    Collection.WORKSPACE: DocLayout(WORKSPACE_FILE, ()),
    Collection.ASSIGNMENTS: DocLayout(f"{_PAIR_DIR_T}/{ASSIGNMENT_MANIFEST}", ("did", "fid")),
    Collection.EXTRACTION_STATS: DocLayout(
        f"{_PAIR_DIR_T}/{EXTRACTION_STATS_FILE}", ("did", "fid")
    ),
}

#: Every fixed filename a document occupies, plus the two bootstrap files.
#: Used to reject a blob key that would collide with a document.
RESERVED_BASENAMES: frozenset[str] = frozenset(
    basename
    for layout in DOC_LAYOUTS.values()
    if "{" not in (basename := layout.template.rsplit("/", 1)[-1])
) | {CONFIG_FILE, USAGE_FILE}


# ------------------------------------------------------- blob classification

# Allow-listed blob key shapes, as regexes over the root-relative POSIX key. A
# key that matches none of these is not blob content. Ordered widest-last: the
# file-source rule is the loose one, so it explicitly excludes the reserved
# document basenames that share ``files/<id>/``.
_SEG = r"[^/]+"
_BLOB_RULES: tuple[re.Pattern[str], ...] = (
    re.compile(rf"^{FILES_DIR}/{_SEG}/{PAGE_IMAGES_DIR}/{_SEG}$"),
    re.compile(rf"^{FILES_DIR}/{_SEG}/{PAGE_TEXT_DIR}/{_SEG}$"),
    re.compile(
        rf"^{DOCSETS_DIR}/{_SEG}/"
        rf"(?:{EXTRACTION_SCHEMA_FILE}|{EXTRACTION_GUIDANCE_FILE}"
        rf"|{FULL_SCHEMA_FILE}|{GENERATION_SCHEMA_FILE})$"
    ),
    # The optional --debug word-coverage report, directly under the docset dir.
    re.compile(rf"^{DOCSETS_DIR}/{_SEG}/{re.escape(COVERAGE_REPORT_FILE)}$"),
    re.compile(rf"^{DOCSETS_DIR}/{_SEG}/{GENERATION_CACHE_DIR}/.+$"),
    re.compile(rf"^{DOCSETS_DIR}/{_SEG}/{DOCSET_FILES_DIR}/{_SEG}/{_SEG}$"),
    # The stored original / converted PDF sits directly in the file directory,
    # alongside its manifest — hence the reserved-name exclusion.
    re.compile(rf"^{FILES_DIR}/{_SEG}/{_SEG}$"),
)


def is_blob_key(key: str) -> bool:
    """Whether ``key`` names blob content (as opposed to a document, a
    bootstrap file, or workspace-internal scratch).

    An allow-list: a key shape nobody declared is *not* a blob. That way adding
    a document filename cannot accidentally make it attestable content — the
    failure mode of the previous deny-list, where anything unrecognized was
    treated as a blob."""
    if key.endswith(".tmp"):
        return False  # an in-flight atomic write
    if key.split("/", 1)[0] == CACHE_DIR:
        return False  # workspace-internal scratch
    if key.rsplit("/", 1)[-1] in RESERVED_BASENAMES:
        return False
    return any(rule.match(key) for rule in _BLOB_RULES)
