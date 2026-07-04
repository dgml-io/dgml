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
from dgml_core.storage import Workspace, write_json_atomic

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
    write_json_atomic(store.ws.config_path, {"conversion": {"docx": {"provider": _STUB_DOCX}}})
    # The PDF-only post-steps need ghostscript / pdfminer; stub them out so the
    # test isolates the conversion-persistence behavior.
    monkeypatch.setattr(FileStore, "_safe_page_count", lambda self, *a, **k: (None, None))
    monkeypatch.setattr(FileStore, "_render_pages", lambda self, *a, **k: None)
    monkeypatch.setattr(FileStore, "_extract_text", lambda self, *a, **k: (None, None))

    src = tmp_path / "foo.docx"
    src.write_bytes(b"original docx bytes")
    result = store.add(src)

    fdir = store.ws.file_dir(result.record.id)
    assert result.conversion_error is None
    assert result.record.original_filename == "foo.docx"
    assert (fdir / "foo.docx").read_bytes() == b"original docx bytes"  # original preserved
    assert (fdir / "foo.pdf").read_bytes() == b"%PDF-stub:foo.docx"  # converted persisted
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
    pages = list(store.ws.file_pages_dir(result.record.id).glob("page_*.png"))
    assert len(pages) == 2


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
    workspace.file_dir(keep_a).mkdir(parents=True)
    workspace.file_dir(keep_b).mkdir(parents=True)
    docsets = DocSetStore(workspace)
    ds = docsets.create(name="X")
    docsets.add_file(ds.id, keep_a)

    with pytest.raises(InvalidArgument):
        store.delete("")
    with pytest.raises(InvalidArgument):
        store.delete("   ")

    assert workspace.file_dir(keep_a).is_dir()
    assert workspace.file_dir(keep_b).is_dir()
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
    assert workspace.file_json_path(result.record.id).exists()
    # The recorded error is permanent so consistency check won't loop.
    from dgml_core.errors import load_recorded_errors

    recorded = load_recorded_errors(workspace.file_errors_path(result.record.id))
    assert any(e.operation == "pdf_page_count" and e.permanent for e in recorded)
