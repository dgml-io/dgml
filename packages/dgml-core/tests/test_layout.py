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

"""Tests for the shared workspace layout: keys, collections, blob classification."""

from __future__ import annotations

from pathlib import Path

import pytest
from dgml_core import layout
from dgml_core.errors import InvalidArgument
from dgml_core.storage import Workspace

from .conftest import local_store

# --------------------------------------------------------------------- keys


def test_keys_are_root_relative_posix() -> None:
    assert layout.file_source_key("f1", "a.pdf") == "files/f1/a.pdf"
    assert layout.file_page_image_key("f1", 3) == "files/f1/page_images/page_3.png"
    assert layout.file_page_text_key("f1", 3) == "files/f1/page_text/page_3.json"
    assert layout.docset_extraction_schema_key("d1") == "docsets/d1/extraction-schema.rnc"
    assert layout.docset_full_schema_key("d1") == "docsets/d1/full-schema.rnc"
    assert layout.docset_generation_schema_key("d1") == "docsets/d1/schema.json"
    assert layout.dgml_xml_key("d1", "f1", "report") == "docsets/d1/files/f1/report.dgml.xml"
    assert layout.pair_artifact_key("d1", "f1", "s.json") == "docsets/d1/files/f1/s.json"
    assert layout.pair_id("d1", "f1") == "d1/f1"


def test_prefixes_end_with_a_separator() -> None:
    """The trailing slash is load-bearing: ``list_blobs`` / ``delete_blobs`` match
    by string prefix, so a bare ``docsets/d1`` would also select ``docsets/d10``."""
    prefixes = [
        layout.file_prefix("f1"),
        layout.file_pages_prefix("f1"),
        layout.file_text_prefix("f1"),
        layout.docset_prefix("d1"),
        layout.docset_files_prefix("d1"),
        layout.docset_pair_prefix("d1", "f1"),
        layout.generation_cache_prefix("d1"),
    ]
    assert all(p.endswith("/") for p in prefixes), prefixes
    assert not layout.docset_prefix("d1").startswith(layout.docset_prefix("d10"))


def test_sibling_ids_do_not_share_a_prefix() -> None:
    assert not layout.docset_pair_prefix("d1", "f1").startswith(
        layout.docset_pair_prefix("d1", "f10")
    )
    assert not layout.file_prefix("f1").startswith(layout.file_prefix("f10"))


# ------------------------------------------------------- blob classification


@pytest.mark.parametrize(
    "key",
    [
        "files/f1/report.pdf",
        "files/f1/report.docx",
        "files/f1/page_images/page_1.png",
        "files/f1/page_text/page_1.json",
        "docsets/d1/extraction-schema.rnc",
        "docsets/d1/full-schema.rnc",
        "docsets/d1/schema.json",
        "docsets/d1/coverage_report.json",  # --debug word-coverage report
        "docsets/d1/files/f1/report.dgml.xml",
        "docsets/d1/cache/labeled/doc.json",
    ],
)
def test_blob_keys_accepted(key: str) -> None:
    assert layout.is_blob_key(key) is True


def test_coverage_report_key_is_a_writable_blob() -> None:
    # The builder and the allow-list must agree, so --debug docset generate can
    # persist the report through the store (regression: it collided with the
    # allow-list and put_blob rejected it).
    key = layout.docset_coverage_report_key("d1")
    assert key == "docsets/d1/coverage_report.json"
    assert layout.is_blob_key(key) is True


@pytest.mark.parametrize(
    "key",
    [
        "files/f1/file.json",  # document
        "files/f1/errors.json",  # document
        "docsets/d1/docset.json",  # document
        "docsets/d1/files/f1/assignment.json",  # document
        "docsets/d1/files/f1/extraction_stats.json",  # document
        "workspace.json",  # document
        "config.toml",  # bootstrap
        "usage.jsonl",  # append-only log
        ".cache/embeddings/e.npy",  # workspace-internal scratch
        ".cache/staging/tmp1/page_1.png",  # in-flight staged write
        "files/f1/page_images/page_1.png.tmp",  # in-flight atomic write
        "files/f1/page_images/nested/deeper.png",  # not a declared shape
        "stray.txt",  # not a declared shape
        "docsets/d1/files/f1/nested/deep.xml",  # not a declared shape
    ],
)
def test_non_blob_keys_rejected(key: str) -> None:
    assert layout.is_blob_key(key) is False


def test_every_document_path_is_excluded_from_the_blob_namespace() -> None:
    """Derived rather than hand-listed: any collection added to DOC_LAYOUTS must
    stay out of the blob namespace, or ``list_blobs`` would return it and
    attestation could hash it."""
    for collection, doc_layout in layout.DOC_LAYOUTS.items():
        key = doc_layout.template.format(**dict.fromkeys(doc_layout.id_parts, "x"))
        assert layout.is_blob_key(key) is False, f"{collection} document is blob-visible: {key}"


# ---------------------------------------------- the local store's write guard


def test_put_blob_rejects_a_document_key(tmp_path: Path) -> None:
    """Writing a blob into a document's slot would clobber the manifest *and* be
    invisible to ``list_blobs`` afterwards.

    Not reachable through any current caller (no ingestable source suffix ends in
    ``.json``), so this pins the guard for the next document filename or accepted
    extension rather than covering a live bug."""
    store = local_store(tmp_path)
    store.put_doc(layout.Collection.FILES, "f1", {"id": "f1"})
    with pytest.raises(InvalidArgument, match="collides with"):
        store.put_blob(layout.file_source_key("f1", layout.FILE_MANIFEST), b"not a manifest")
    assert store.get_doc(layout.Collection.FILES, "f1") == {"id": "f1"}


def test_upload_blob_rejects_a_document_key(tmp_path: Path) -> None:
    store = local_store(tmp_path)
    src = tmp_path / "src.bin"
    src.write_bytes(b"x")
    with pytest.raises(InvalidArgument, match="collides with"):
        store.upload_blob("docsets/d1/docset.json", src)


def test_traversal_still_raises_value_error(tmp_path: Path) -> None:
    """Key *shape* is checked before namespace ownership, so a traversal attempt
    keeps its lower-level ValueError rather than being reported as a collision."""
    store = local_store(tmp_path)
    for bad in ("/abs/key", "../escape", "a/../../b"):
        with pytest.raises(ValueError):
            store.put_blob(bad, b"x")


# --------------------------------------------- paths agree with keys


def test_local_path_agrees_with_keys(tmp_path: Path) -> None:
    """``local_path`` must resolve a key to its location under the root, and
    ``blob_key`` must invert it — the drift the shared layout exists to prevent.
    ``Workspace`` names nothing itself: every key comes from a ``layout``
    builder, and the filesystem escape is ``local_path`` composed with one."""
    ws = Workspace(root=tmp_path)
    keys = [
        layout.file_prefix("f1"),
        layout.file_pages_prefix("f1"),
        layout.file_text_prefix("f1"),
        layout.file_source_key("f1", "doc.pdf"),
        layout.docset_prefix("d1"),
        layout.docset_files_prefix("d1"),
        layout.docset_pair_prefix("d1", "f1"),
        layout.docset_extraction_schema_key("d1"),
        layout.docset_full_schema_key("d1"),
        layout.docset_generation_schema_key("d1"),
        layout.dgml_xml_key("d1", "f1", "r"),
        layout.file_page_image_key("f1", 1),
        layout.file_page_text_key("f1", 1),
    ]
    for key in keys:
        assert ws.local_path(key) == tmp_path / key.rstrip("/"), key
        assert ws.blob_key(ws.local_path(key)) == key.rstrip("/"), key


# ------------------------------------------------------- the store is cached


def test_workspace_blobs_is_resolved_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolving means reading config, importing the provider and constructing
    it — a fresh SDK client per call on a remote backend, across ~100 call
    sites. A workspace's blob store is one static choice, so it is cached."""
    import dgml_core.storage_resolve as storage_resolve

    built = 0
    real_make = storage_resolve.make_blob_store

    def counting_make(config: object) -> object:
        nonlocal built
        built += 1
        return real_make(config)  # type: ignore[arg-type]

    monkeypatch.setattr(storage_resolve, "make_blob_store", counting_make)

    ws = Workspace(root=tmp_path)
    first = ws.blobs
    for _ in range(5):
        assert ws.blobs is first
    assert built == 1

    # Caching is per-workspace, not global: a separate instance resolves its own.
    assert Workspace(root=tmp_path).blobs is not first
    assert built == 2
