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

from pathlib import Path

from dgml_core.consistency import check_workspace
from dgml_core.docsets import DocSetStore
from dgml_core.errors import (
    RecordedError,
    append_recorded_error,
    load_recorded_errors,
    now_iso,
)
from dgml_core.files import FileStore
from dgml_core.pages import PAGE_GLOB
from dgml_core.storage import Workspace, read_json, write_json_atomic

from .conftest import needs_gs


@needs_gs
def test_clean_workspace_passes(workspace: Workspace, text_pdf: Path) -> None:
    # Use ``text_pdf`` (real digital text) so the text-extraction check is
    # satisfied; the blank ``sample_pdf`` would (correctly) flag itself as
    # text_extraction_failed_permanent.
    FileStore(workspace).add(text_pdf)
    report = check_workspace(workspace)
    assert report.ok, report.to_json()
    assert report.files_checked == 1


@needs_gs
def test_missing_pdf_detected(workspace: Workspace, sample_pdf: Path) -> None:
    f = FileStore(workspace).add(sample_pdf)
    pdf = workspace.file_dir(f.record.id) / f.record.original_filename
    pdf.unlink()
    report = check_workspace(workspace)
    assert any(i.kind == "missing_pdf" for i in report.issues)


@needs_gs
def test_hash_mismatch_detected(workspace: Workspace, sample_pdf: Path) -> None:
    f = FileStore(workspace).add(sample_pdf)
    pdf = workspace.file_dir(f.record.id) / f.record.original_filename
    pdf.write_bytes(b"%PDF-1.4\nbroken-but-still-pdf-magic")
    report = check_workspace(workspace)
    assert any(i.kind == "hash_mismatch" for i in report.issues)


@needs_gs
def test_missing_pages_re_rendered(workspace: Workspace, sample_pdf: Path) -> None:
    f = FileStore(workspace).add(sample_pdf)
    pages = workspace.file_pages_dir(f.record.id)
    for p in pages.glob(PAGE_GLOB):
        p.unlink()
    report = check_workspace(workspace)
    repaired = [i for i in report.issues if i.kind == "page_count_mismatch" and i.repaired]
    assert repaired, report.to_json()
    assert len(list(pages.glob(PAGE_GLOB))) == 2


@needs_gs
def test_bogus_zero_page_count_with_pages_on_disk_is_consistent(
    workspace: Workspace, text_pdf: Path
) -> None:
    """A stored page_count of 0 (pdfminer can emit it for renderable PDFs) must
    be treated as unknown, not as authoritative — so a file with its pages
    intact on disk is NOT flagged as a spurious ``expected 0`` mismatch."""
    f = FileStore(workspace).add(text_pdf)
    json_path = workspace.file_json_path(f.record.id)
    data = read_json(json_path)
    data["page_count"] = 0
    write_json_atomic(json_path, data)

    report = check_workspace(workspace)
    assert not [i for i in report.issues if i.kind == "page_count_mismatch"], report.to_json()


@needs_gs
def test_bogus_zero_page_count_re_renders_missing_pages(
    workspace: Workspace, sample_pdf: Path
) -> None:
    """With a bogus stored page_count of 0 AND no pages on disk, check must
    still recover by re-rendering (ghostscript is authoritative) rather than
    silently treating 0 rendered == 0 expected as consistent."""
    f = FileStore(workspace).add(sample_pdf)
    json_path = workspace.file_json_path(f.record.id)
    data = read_json(json_path)
    data["page_count"] = 0
    write_json_atomic(json_path, data)
    pages = workspace.file_pages_dir(f.record.id)
    for p in pages.glob(PAGE_GLOB):
        p.unlink()

    report = check_workspace(workspace)
    assert any(i.kind == "page_count_mismatch" and i.repaired for i in report.issues), (
        report.to_json()
    )
    assert len(list(pages.glob(PAGE_GLOB))) == 2


@needs_gs
def test_permanent_error_blocks_retry(workspace: Workspace, sample_pdf: Path) -> None:
    f = FileStore(workspace).add(sample_pdf)
    for p in workspace.file_pages_dir(f.record.id).glob(PAGE_GLOB):
        p.unlink()
    append_recorded_error(
        workspace.file_errors_path(f.record.id),
        RecordedError(
            operation="render_pages",
            message="simulated permanent failure",
            occurred_at=now_iso(),
            permanent=True,
        ),
    )
    report = check_workspace(workspace)
    assert any(i.kind == "page_render_failed_permanent" for i in report.issues)
    assert not list(workspace.file_pages_dir(f.record.id).glob(PAGE_GLOB))


@needs_gs
def test_retry_errors_clears_and_retries(workspace: Workspace, sample_pdf: Path) -> None:
    f = FileStore(workspace).add(sample_pdf)
    for p in workspace.file_pages_dir(f.record.id).glob(PAGE_GLOB):
        p.unlink()
    append_recorded_error(
        workspace.file_errors_path(f.record.id),
        RecordedError(
            operation="render_pages",
            message="simulated permanent failure",
            occurred_at=now_iso(),
            permanent=True,
        ),
    )
    report = check_workspace(workspace, retry_errors=True)
    assert len(list(workspace.file_pages_dir(f.record.id).glob(PAGE_GLOB))) == 2
    assert load_recorded_errors(workspace.file_errors_path(f.record.id)) == []
    assert any(i.kind == "page_count_mismatch" and i.repaired for i in report.issues)


def test_dangling_docset_reference(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    (workspace.docset_files_dir(ds.id) / "missingfileid").mkdir()
    report = check_workspace(workspace)
    assert any(i.kind == "dangling_file_reference" for i in report.issues)


def test_orphan_file_dir_missing_metadata(workspace: Workspace) -> None:
    workspace.file_dir("orphanedfile").mkdir(parents=True)
    report = check_workspace(workspace)
    assert any(i.target_type == "file" and i.kind == "missing_metadata" for i in report.issues)


def test_corrupt_file_metadata_does_not_crash(workspace: Workspace) -> None:
    """A corrupt file.json must be reported, not crash the whole walk."""
    fid = "corruptfileid"
    workspace.file_dir(fid).mkdir(parents=True)
    workspace.file_json_path(fid).write_text("{not valid json")
    report = check_workspace(workspace)
    assert any(
        i.target_type == "file" and i.target_id == fid and i.kind == "corrupt_metadata"
        for i in report.issues
    )


def test_corrupt_docset_metadata_does_not_crash(workspace: Workspace) -> None:
    did = "corruptdocsetid"
    workspace.docset_dir(did).mkdir(parents=True)
    workspace.docset_json_path(did).write_text("{not valid json")
    report = check_workspace(workspace)
    assert any(
        i.target_type == "docset" and i.target_id == did and i.kind == "corrupt_metadata"
        for i in report.issues
    )


def test_corrupt_metadata_alongside_clean_continues_walk(
    workspace: Workspace,
) -> None:
    """A corrupt metadata file early in the walk must not stop later
    files/docsets from being checked."""
    bad = "aaaaaaaaaaaa"
    good = "zzzzzzzzzzzz"
    workspace.file_dir(bad).mkdir(parents=True)
    workspace.file_json_path(bad).write_text("{not json")
    workspace.file_dir(good).mkdir(parents=True)  # missing metadata, but cleanly missing
    report = check_workspace(workspace)
    issues_by_id = {i.target_id: i.kind for i in report.issues if i.target_type == "file"}
    assert issues_by_id.get(bad) == "corrupt_metadata"
    assert issues_by_id.get(good) == "missing_metadata"


def test_check_no_longer_falls_back_to_any_pdf(workspace: Workspace) -> None:
    """If the named PDF is missing, surface missing_pdf rather than silently
    using a different PDF that happens to be in the directory."""
    fid = "fabfileabcde"
    workspace.file_dir(fid).mkdir(parents=True)
    workspace.file_json_path(fid).write_text(
        '{"id": "fabfileabcde", "original_path": "/tmp/x.pdf",'
        ' "original_filename": "x.pdf", "sha256": "deadbeef",'
        ' "added_at": "2026-05-08T00:00:00Z", "page_count": 1}'
    )
    (workspace.file_dir(fid) / "something_else.pdf").write_bytes(b"%PDF-1.4\n")
    report = check_workspace(workspace)
    assert any(i.target_id == fid and i.kind == "missing_pdf" for i in report.issues)


@needs_gs
def test_unattributed_computed_field_flagged(workspace: Workspace, text_pdf: Path) -> None:
    """A dg:origin="computed" element with no dg:href in a docset file's DGML
    XML is an unauditable derivation — check reports it. An attributed
    computed element (dg:href present) passes clean."""
    f = FileStore(workspace).add(text_pdf)
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    store.add_file(ds.id, f.record.id)

    xml_path = workspace.file_dgml_xml_path(ds.id, f.record.id, "doc")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" xmlns:docset="http://x/ns">'
        "<dg:extraction>"
        '<docset:Total dg:origin="computed" dg:value="10">10</docset:Total>'
        "</dg:extraction></dg:chunk>",
        encoding="utf-8",
    )
    report = check_workspace(workspace)
    flagged = [i for i in report.issues if i.kind == "computed_field_unattributed"]
    assert len(flagged) == 1
    assert "Total" in flagged[0].message
    assert flagged[0].target_id == ds.id

    xml_path.write_text(
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" xmlns:docset="http://x/ns">'
        "<dg:extraction>"
        '<docset:Part xml:id="p1" dg:origin="1 1 2 3 4">4</docset:Part>'
        '<docset:Total dg:origin="computed" dg:value="10" '
        'dg:itemprop="computedFrom" dg:href="#p1">10</docset:Total>'
        "</dg:extraction></dg:chunk>",
        encoding="utf-8",
    )
    report = check_workspace(workspace)
    assert not [i for i in report.issues if i.kind == "computed_field_unattributed"]
