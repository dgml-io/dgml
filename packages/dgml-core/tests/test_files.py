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


def test_add_pdf_with_pypdfium2_renderer(store: FileStore, sample_pdf: Path) -> None:
    """A workspace configured for pypdfium2 renders through PDFium (no
    ghostscript needed — hence no ``needs_gs``) and records the provider."""
    from .conftest import write_config

    write_config(store.ws, {"pdf": {"provider": "pypdfium2"}})
    result = store.add(sample_pdf)
    assert result.created
    assert result.page_render_error is None
    assert result.record.page_image_renderer == "pypdfium2"
    assert len(_page_pngs(store.ws, result.record.id)) == 2


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
    # Nothing written: the directory is created by the first write, so on a rejected
    # add it should not exist at all (and must certainly be empty if it does).
    files_dir = store.ws.files_dir
    assert not files_dir.exists() or not any(files_dir.iterdir())


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


# --------------------------------------------------------------------------
# Caller-supplied file ids (`file add --id`).
#
# The hazard these guard: `put_doc` is an upsert on every backend, so a record
# written under an id another record holds silently replaces it. Every rule
# below exists to make that unreachable.
# --------------------------------------------------------------------------


@needs_gs
def test_add_with_requested_id_uses_it(store: FileStore, sample_pdf: Path) -> None:
    result = store.add(sample_pdf, file_id="my-report-1")
    assert result.created
    assert result.record.id == "my-report-1"
    assert store.get("my-report-1").sha256 == result.record.sha256


@needs_gs
@pytest.mark.parametrize("file_id", ["abc", "a" * 40, "a_b-c", "0start"])
def test_add_requested_id_accepts_grammar_bounds(
    store: FileStore, sample_pdf: Path, file_id: str
) -> None:
    """The grammar's edges have to survive being used as a real path segment."""
    assert store.add(sample_pdf, file_id=file_id).record.id == file_id


@pytest.mark.parametrize("file_id", ["ab", "A" * 12, "-lead", "a.b", "a/b", ""])
def test_add_rejects_malformed_requested_id(
    store: FileStore, workspace: Workspace, sample_pdf: Path, file_id: str
) -> None:
    with pytest.raises(InvalidArgument):
        store.add(sample_pdf, file_id=file_id)
    # Nothing half-built — the shape check runs before any write.
    assert store.list_all() == []
    assert not (workspace.root / "files").exists()


def test_uppercase_uuid_is_rejected(
    store: FileStore, workspace: Workspace, sample_pdf: Path
) -> None:
    """A lowercase UUID is a valid id, so the uppercase form is a realistic trap —
    several systems emit them that way. It must be refused before anything is
    written, not case-folded: the id is how the caller's system and this workspace
    name the same document."""
    import uuid

    with pytest.raises(InvalidArgument):
        store.add(sample_pdf, file_id=str(uuid.uuid4()).upper())
    assert store.list_all() == []
    assert not (workspace.root / "files").exists()


@needs_gs
@pytest.mark.parametrize("policy", list(ConflictPolicy))
def test_requested_id_taken_by_different_content_conflicts(
    store: FileStore, sample_pdf: Path, sample_pdf_alt: Path, policy: ConflictPolicy
) -> None:
    """No --on-conflict policy may reuse an id held by different content.

    The final assertion is the one that matters: an unguarded add would upsert
    over the held record and destroy it."""
    first = store.add(sample_pdf, file_id="keep-me")
    with pytest.raises(ConflictError) as excinfo:
        store.add(sample_pdf_alt, file_id="keep-me", on_conflict=policy)
    assert excinfo.value.kind == "id"
    assert excinfo.value.existing_id == "keep-me"
    assert store.get("keep-me").sha256 == first.record.sha256


@needs_gs
def test_requested_id_same_content_skip_is_idempotent(store: FileStore, sample_pdf: Path) -> None:
    store.add(sample_pdf, file_id="doc-1")
    again = store.add(sample_pdf, file_id="doc-1", on_conflict=ConflictPolicy.SKIP)
    assert not again.created
    assert again.record.id == "doc-1"
    assert again.conflict_kind == "hash"
    assert len(store.list_all()) == 1


@needs_gs
def test_requested_id_same_content_error_raises_hash_not_id(
    store: FileStore, sample_pdf: Path
) -> None:
    """Re-adding identical content still diagnoses as a hash conflict."""
    store.add(sample_pdf, file_id="doc-1")
    with pytest.raises(ConflictError) as excinfo:
        store.add(sample_pdf, file_id="doc-1")
    assert excinfo.value.kind == "hash"


@needs_gs
def test_requested_id_same_content_replace_is_noop(store: FileStore, sample_pdf: Path) -> None:
    store.add(sample_pdf, file_id="doc-1")
    again = store.add(sample_pdf, file_id="doc-1", on_conflict=ConflictPolicy.REPLACE)
    assert not again.created
    assert again.record.id == "doc-1"


@needs_gs
def test_requested_id_same_content_duplicate_rejected(store: FileStore, sample_pdf: Path) -> None:
    """`duplicate` means "make a second record", but the named id already has
    one — so there is nowhere to put it."""
    store.add(sample_pdf, file_id="doc-1")
    with pytest.raises(ConflictError) as excinfo:
        store.add(sample_pdf, file_id="doc-1", on_conflict=ConflictPolicy.DUPLICATE)
    assert excinfo.value.kind == "id"
    assert len(store.list_all()) == 1


@needs_gs
def test_requested_id_pins_which_same_hash_record_returns(
    store: FileStore, sample_pdf: Path
) -> None:
    """_find_conflicts returns the id-sorted *first* same-content record, which
    need not be the one the caller named."""
    store.add(sample_pdf, file_id="aaa-1")
    store.add(sample_pdf, file_id="zzz-9", on_conflict=ConflictPolicy.DUPLICATE)
    got = store.add(sample_pdf, file_id="zzz-9", on_conflict=ConflictPolicy.SKIP)
    assert got.record.id == "zzz-9"


@needs_gs
def test_free_id_rejected_when_skip_returns_hash_match(store: FileStore, sample_pdf: Path) -> None:
    """`skip` would hand back a record with a different id than the one asked
    for; returning it silently is how a caller mis-addresses a document."""
    store.add(sample_pdf)
    with pytest.raises(InvalidArgument):
        store.add(sample_pdf, file_id="wanted", on_conflict=ConflictPolicy.SKIP)
    with pytest.raises(FileNotFound):
        store.get("wanted")


@needs_gs
def test_free_id_rejected_when_skip_returns_path_match(
    store: FileStore, sample_pdf: Path, sample_pdf_alt: Path
) -> None:
    store.add(sample_pdf)
    shutil.copy2(sample_pdf_alt, sample_pdf)
    with pytest.raises(InvalidArgument):
        store.add(sample_pdf, file_id="wanted", on_conflict=ConflictPolicy.SKIP)


@needs_gs
def test_free_id_rejected_when_replace_is_hash_noop(store: FileStore, sample_pdf: Path) -> None:
    store.add(sample_pdf)
    with pytest.raises(InvalidArgument):
        store.add(sample_pdf, file_id="wanted", on_conflict=ConflictPolicy.REPLACE)


@needs_gs
def test_free_id_honoured_when_replace_swaps_path(
    store: FileStore, sample_pdf: Path, sample_pdf_alt: Path
) -> None:
    """`replace` on a *path* conflict creates a new record, so it can and must
    carry the requested id — the case a policy-based rule would wrongly reject."""
    store.add(sample_pdf)
    shutil.copy2(sample_pdf_alt, sample_pdf)
    second = store.add(sample_pdf, file_id="rev-2", on_conflict=ConflictPolicy.REPLACE)
    assert second.created
    assert second.record.id == "rev-2"
    assert {r.id for r in store.list_all()} == {"rev-2"}


@needs_gs
def test_replace_reingest_keeps_requested_id(
    store: FileStore, sample_pdf: Path, sample_pdf_alt: Path
) -> None:
    """Re-ingesting a revised document under its own id: the held record is the
    one `replace` is about to delete, so the id is about to be free."""
    first = store.add(sample_pdf, file_id="invoice-2024")
    shutil.copy2(sample_pdf_alt, sample_pdf)
    second = store.add(sample_pdf, file_id="invoice-2024", on_conflict=ConflictPolicy.REPLACE)
    assert second.created
    assert second.record.id == "invoice-2024"
    assert second.record.sha256 != first.record.sha256
    assert {r.id for r in store.list_all()} == {"invoice-2024"}


@needs_gs
def test_free_id_honoured_with_duplicate(store: FileStore, sample_pdf: Path) -> None:
    store.add(sample_pdf)
    second = store.add(sample_pdf, file_id="copy-2", on_conflict=ConflictPolicy.DUPLICATE)
    assert second.created
    assert second.record.id == "copy-2"
    assert len(store.list_all()) == 2


@needs_gs
def test_add_without_id_still_mints(store: FileStore, sample_pdf: Path) -> None:
    from dgml_core.ids import ID_LENGTH, is_record_id

    record_id = store.add(sample_pdf).record.id
    assert len(record_id) == ID_LENGTH
    assert is_record_id(record_id)
