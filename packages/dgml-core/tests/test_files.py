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

from __future__ import annotations

import shutil
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest
from dgml_core import layout
from dgml_core.conversion import ConverterConfig, DocConverter
from dgml_core.docsets import DocSetStore
from dgml_core.errors import (
    ConflictError,
    FileNotFound,
    InvalidArgument,
    InvalidPDF,
    UnsupportedFileType,
)
from dgml_core.files import ConflictPolicy, FileStore
from dgml_core.storage import Workspace

from .conftest import needs_gs


@pytest.fixture
def store(workspace: Workspace) -> FileStore:
    return FileStore(workspace)


class _StubDocxConverter(DocConverter):
    """Returns deterministic bytes so persistence can be asserted without a
    real converter binary."""

    name: ClassVar[str] = "stub-docx"
    input_formats: ClassVar[frozenset[str]] = frozenset({".docx"})
    config_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def parse_config(cls, section: Mapping[str, Any]) -> ConverterConfig:
        cls._check_no_extra_fields(section)
        return ConverterConfig(provider=str(section["provider"]))

    def __init__(self, config: ConverterConfig) -> None:
        pass

    def to_pdf(self, path: Path) -> bytes:
        return b"%PDF-stub:" + Path(path).name.encode()


_stub_mod = types.ModuleType("files_stub_conv")
_stub_mod._StubDocxConverter = _StubDocxConverter  # type: ignore[attr-defined]
sys.modules["files_stub_conv"] = _stub_mod
_STUB_DOCX = "files_stub_conv:_StubDocxConverter"


def test_convertible_source_persists_converted_pdf(
    store: FileStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A convertible source is stored as-is and its converted PDF is persisted
    alongside it at ``<stem>.pdf`` (the artifact generation later reuses)."""
    from .conftest import write_config

    write_config(store.ws, {"conversion": {"docx": {"provider": _STUB_DOCX}}})
    # The PDF-only post-steps need ghostscript / pdfminer; stub them out so the
    # test isolates the conversion-persistence behavior.
    monkeypatch.setattr(FileStore, "_safe_page_count", lambda self, *a, **k: (None, None))
    monkeypatch.setattr(FileStore, "_render_pages", lambda self, *a, **k: None)
    monkeypatch.setattr(FileStore, "_extract_text", lambda self, *a, **k: (None, None))

    src = tmp_path / "foo.docx"
    src.write_bytes(b"original docx bytes")
    result = store.add(src)

    assert result.conversion_error is None
    assert result.record.original_filename == "foo.docx"
    # original preserved + converted PDF persisted, both as blobs under the file
    assert (
        store.ws.blobs.get_blob(layout.file_source_key(result.record.id, "foo.docx"))
        == b"original docx bytes"
    )
    assert (
        store.ws.blobs.get_blob(layout.file_source_key(result.record.id, "foo.pdf"))
        == b"%PDF-stub:foo.docx"
    )
    assert result.record.pdf_converter == "stub-docx"  # converter named on the record


@needs_gs
def test_add_pdf(store: FileStore, sample_pdf: Path) -> None:
    result = store.add(sample_pdf)
    assert result.created
    assert result.record.sha256
    assert result.record.original_filename == "sample.pdf"
    assert result.record.page_count == 2
    assert result.page_render_error is None
    # A PDF source records renderer provenance but no converter.
    assert result.record.page_image_dpi == 300
    assert result.record.page_image_renderer == "ghostscript"
    assert result.record.pdf_converter is None
    pages = _page_pngs(store.ws, result.record.id)
    assert len(pages) == 2


@needs_gs
def test_add_pdf_custom_dpi_is_rendered_and_recorded(store: FileStore, sample_pdf: Path) -> None:
    result = store.add(sample_pdf, dpi=150)
    assert result.page_render_error is None
    assert result.record.page_image_dpi == 150
    pages = _page_pngs(store.ws, result.record.id)
    assert len(pages) == 2
    # The record has to describe the pixels actually on disk, since `dgml check`
    # reproduces this geometry when it repairs the file later.
    width, height = _png_size(store.ws.blobs.get_blob(pages[0]))
    at_300 = store.add(sample_pdf, on_conflict=ConflictPolicy.DUPLICATE)
    w300, h300 = _png_size(store.ws.blobs.get_blob(_page_pngs(store.ws, at_300.record.id)[0]))
    assert width < w300 and height < h300


def _png_size(data: bytes) -> tuple[int, int]:
    """Width/height from a PNG's IHDR — avoids depending on an image library."""
    header = data[16:24]
    return int.from_bytes(header[:4], "big"), int.from_bytes(header[4:], "big")


def _page_pngs(ws: Workspace, file_id: str) -> list[str]:
    """Sorted page-image blob keys for ``file_id`` (store analogue of globbing
    page_*.png in the page-images dir)."""
    return sorted(
        k for k in ws.blobs.list_blobs(layout.file_pages_prefix(file_id)) if k.endswith(".png")
    )


def test_add_rejects_nonpositive_dpi(store: FileStore, sample_pdf: Path) -> None:
    # Rejected before anything is written, so a bad flag can't leave a
    # half-built File behind for `dgml check` to puzzle over.
    for bad in (0, -300):
        with pytest.raises(ValueError, match="dpi"):
            store.add(sample_pdf, dpi=bad)
    assert not any(store.ws.files_dir.iterdir())


@needs_gs
def test_original_path_stored_relative_to_workspace(store: FileStore, sample_pdf: Path) -> None:
    """original_path is recorded relative to the workspace root and still
    resolves back to the source from there — keeping the workspace portable."""
    result = store.add(sample_pdf)
    # Fixtures put the source at tmp_path/sample.pdf and the workspace at
    # tmp_path/ws, so the source is one level up from the workspace root.
    assert result.record.original_path == "../sample.pdf"
    assert not Path(result.record.original_path).is_absolute()
    resolved = (store.ws.root / result.record.original_path).resolve()
    assert resolved == sample_pdf.resolve()


def test_reject_non_pdf(store: FileStore, tmp_path: Path) -> None:
    bad = tmp_path / "x.txt"
    bad.write_text("not a pdf")
    with pytest.raises(UnsupportedFileType):
        store.add(bad)


def test_reject_invalid_magic(store: FileStore, tmp_path: Path) -> None:
    bad = tmp_path / "fake.pdf"
    bad.write_bytes(b"NOT A PDF")
    with pytest.raises(InvalidPDF):
        store.add(bad)


def test_reject_missing_path(store: FileStore, tmp_path: Path) -> None:
    with pytest.raises(FileNotFound):
        store.add(tmp_path / "nope.pdf")


@needs_gs
def test_conflict_hash_default_errors(store: FileStore, sample_pdf: Path) -> None:
    store.add(sample_pdf)
    with pytest.raises(ConflictError) as excinfo:
        store.add(sample_pdf)
    assert excinfo.value.kind == "hash"


@needs_gs
def test_conflict_hash_skip_returns_existing(store: FileStore, sample_pdf: Path) -> None:
    first = store.add(sample_pdf)
    second = store.add(sample_pdf, on_conflict=ConflictPolicy.SKIP)
    assert second.record.id == first.record.id
    assert not second.created
    assert second.conflict_kind == "hash"


@needs_gs
def test_conflict_hash_duplicate_creates_new(store: FileStore, sample_pdf: Path) -> None:
    first = store.add(sample_pdf)
    second = store.add(sample_pdf, on_conflict=ConflictPolicy.DUPLICATE)
    assert second.record.id != first.record.id
    assert second.created


@needs_gs
def test_conflict_path_default_errors(
    store: FileStore, sample_pdf: Path, sample_pdf_alt: Path
) -> None:
    store.add(sample_pdf)
    shutil.copy2(sample_pdf_alt, sample_pdf)
    with pytest.raises(ConflictError) as excinfo:
        store.add(sample_pdf)
    assert excinfo.value.kind == "path"


@needs_gs
def test_conflict_path_replace_swaps(
    store: FileStore, sample_pdf: Path, sample_pdf_alt: Path
) -> None:
    first = store.add(sample_pdf)
    shutil.copy2(sample_pdf_alt, sample_pdf)
    second = store.add(sample_pdf, on_conflict=ConflictPolicy.REPLACE)
    assert second.record.id != first.record.id
    assert {r.id for r in store.list_all()} == {second.record.id}


@needs_gs
def test_conflict_path_duplicate_keeps_both(
    store: FileStore, sample_pdf: Path, sample_pdf_alt: Path
) -> None:
    first = store.add(sample_pdf)
    shutil.copy2(sample_pdf_alt, sample_pdf)
    second = store.add(sample_pdf, on_conflict=ConflictPolicy.DUPLICATE)
    assert {first.record.id, second.record.id} <= {r.id for r in store.list_all()}


@needs_gs
def test_delete_removes_docset_references(
    store: FileStore, workspace: Workspace, sample_pdf: Path
) -> None:
    f = store.add(sample_pdf)
    docsets = DocSetStore(workspace)
    ds = docsets.create(name="X")
    docsets.add_file(ds.id, f.record.id)
    assert docsets.list_files(ds.id) == [f.record.id]
    store.delete(f.record.id)
    assert docsets.list_files(ds.id) == []


def test_delete_missing(store: FileStore) -> None:
    with pytest.raises(FileNotFound):
        store.delete("doesnotexist1")


def test_delete_rejects_empty_file_id_preserves_other_files(
    store: FileStore, workspace: Workspace
) -> None:
    """Regression: delete('') must not wipe the entire files directory or
    every docset's file-reference subdir. Both shutil.rmtree calls in
    delete() collapse to parent paths if the file_id is empty.
    """
    keep_a = "aaaaaaaaaaaa"
    keep_b = "bbbbbbbbbbbb"
    workspace.docs.put_doc("files", keep_a, {"id": keep_a})
    workspace.docs.put_doc("files", keep_b, {"id": keep_b})
    docsets = DocSetStore(workspace)
    ds = docsets.create(name="X")
    docsets.add_file(ds.id, keep_a)

    with pytest.raises(InvalidArgument):
        store.delete("")
    with pytest.raises(InvalidArgument):
        store.delete("   ")

    assert workspace.docs.get_doc("files", keep_a) is not None
    assert workspace.docs.get_doc("files", keep_b) is not None
    assert workspace.files_dir.is_dir()
    assert docsets.list_files(ds.id) == [keep_a]


def test_get_rejects_empty_file_id(store: FileStore) -> None:
    with pytest.raises(InvalidArgument):
        store.get("")


@needs_gs
def test_replace_on_hash_conflict_emits_note(store: FileStore, sample_pdf: Path) -> None:
    first = store.add(sample_pdf)
    second = store.add(sample_pdf, on_conflict=ConflictPolicy.REPLACE)
    assert second.record.id == first.record.id
    assert second.created is False
    assert second.conflict_kind == "hash"
    assert second.note is not None
    assert "no-op" in second.note


@needs_gs
def test_skip_on_hash_conflict_emits_note(store: FileStore, sample_pdf: Path) -> None:
    store.add(sample_pdf)
    second = store.add(sample_pdf, on_conflict=ConflictPolicy.SKIP)
    assert second.note is not None


def test_page_count_failure_soft_fails(
    store: FileStore, workspace: Workspace, tmp_path: Path
) -> None:
    """A file that has the PDF magic header but is otherwise malformed
    should still get a record (with page_count=None and a recorded error),
    not abort the add operation mid-way."""
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4\n<<not-actually-valid-pdf-content>>")
    result = store.add(bad)
    assert result.created is True
    assert result.record.page_count is None
    assert result.page_count_error is not None
    # file.json must exist — the partial-failure recovery is the whole point.
    assert workspace.docs.get_doc("files", result.record.id) is not None
    # The recorded error is permanent so consistency check won't loop.
    from dgml_core.errors import load_recorded_errors

    recorded = load_recorded_errors(workspace, result.record.id)
    assert any(e.operation == "pdf_page_count" and e.permanent for e in recorded)
