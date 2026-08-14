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
from typing import Any

import pytest
from dgml_core import layout
from dgml_core.consistency import _reextract, check_workspace
from dgml_core.docsets import DocSetStore
from dgml_core.errors import (
    RecordedError,
    append_recorded_error,
    load_recorded_errors,
    now_iso,
)
from dgml_core.files import FileStore
from dgml_core.storage import Workspace

from .conftest import needs_gs


def _page_pngs(ws: Workspace, file_id: str) -> list[str]:
    """Rendered page-image blob keys for ``file_id`` — the store analogue of
    globbing ``page_*.png`` in the workspace's page-images dir."""
    return [k for k in ws.blobs.list_blobs(layout.file_pages_prefix(file_id)) if k.endswith(".png")]


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
    workspace.blobs.delete_blob(layout.file_source_key(f.record.id, f.record.original_filename))
    report = check_workspace(workspace)
    assert any(i.kind == "missing_pdf" for i in report.issues)


@needs_gs
def test_hash_mismatch_detected(workspace: Workspace, sample_pdf: Path) -> None:
    f = FileStore(workspace).add(sample_pdf)
    workspace.blobs.put_blob(
        layout.file_source_key(f.record.id, f.record.original_filename),
        b"%PDF-1.4\nbroken-but-still-pdf-magic",
    )
    report = check_workspace(workspace)
    assert any(i.kind == "hash_mismatch" for i in report.issues)


@needs_gs
def test_missing_pages_re_rendered(workspace: Workspace, sample_pdf: Path) -> None:
    f = FileStore(workspace).add(sample_pdf)
    workspace.blobs.delete_blobs(layout.file_pages_prefix(f.record.id))
    report = check_workspace(workspace)
    repaired = [i for i in report.issues if i.kind == "page_count_mismatch" and i.repaired]
    assert repaired, report.to_json()
    assert len(_page_pngs(workspace, f.record.id)) == 2


@needs_gs
def test_bogus_zero_page_count_with_pages_on_disk_is_consistent(
    workspace: Workspace, text_pdf: Path
) -> None:
    """A stored page_count of 0 (pdfminer can emit it for renderable PDFs) must
    be treated as unknown, not as authoritative — so a file with its pages
    intact on disk is NOT flagged as a spurious ``expected 0`` mismatch."""
    f = FileStore(workspace).add(text_pdf)
    data = workspace.docs.get_doc("files", f.record.id)
    assert data is not None
    data["page_count"] = 0
    workspace.docs.put_doc("files", f.record.id, data)

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
    data = workspace.docs.get_doc("files", f.record.id)
    assert data is not None
    data["page_count"] = 0
    workspace.docs.put_doc("files", f.record.id, data)
    workspace.blobs.delete_blobs(layout.file_pages_prefix(f.record.id))

    report = check_workspace(workspace)
    assert any(i.kind == "page_count_mismatch" and i.repaired for i in report.issues), (
        report.to_json()
    )
    assert len(_page_pngs(workspace, f.record.id)) == 2


@needs_gs
def test_permanent_error_blocks_retry(workspace: Workspace, sample_pdf: Path) -> None:
    f = FileStore(workspace).add(sample_pdf)
    workspace.blobs.delete_blobs(layout.file_pages_prefix(f.record.id))
    append_recorded_error(
        workspace,
        f.record.id,
        RecordedError(
            operation="render_pages",
            message="simulated permanent failure",
            occurred_at=now_iso(),
            permanent=True,
        ),
    )
    report = check_workspace(workspace)
    assert any(i.kind == "page_render_failed_permanent" for i in report.issues)
    assert not _page_pngs(workspace, f.record.id)


@needs_gs
def test_retry_errors_clears_and_retries(workspace: Workspace, sample_pdf: Path) -> None:
    f = FileStore(workspace).add(sample_pdf)
    workspace.blobs.delete_blobs(layout.file_pages_prefix(f.record.id))
    append_recorded_error(
        workspace,
        f.record.id,
        RecordedError(
            operation="render_pages",
            message="simulated permanent failure",
            occurred_at=now_iso(),
            permanent=True,
        ),
    )
    report = check_workspace(workspace, retry_errors=True)
    assert len(_page_pngs(workspace, f.record.id)) == 2
    assert load_recorded_errors(workspace, f.record.id) == []
    assert any(i.kind == "page_count_mismatch" and i.repaired for i in report.issues)


def test_dangling_docset_reference(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    ds = store.create(name="X")
    # An assignment to a file that has no manifest — the dangling reference.
    workspace.docs.put_doc(
        layout.Collection.ASSIGNMENTS,
        layout.pair_id(ds.id, "missingfileid"),
        {"docset_id": ds.id, "file_id": "missingfileid"},
    )
    report = check_workspace(workspace)
    assert any(i.kind == "dangling_file_reference" for i in report.issues)


def test_orphan_file_dir_missing_metadata(workspace: Workspace) -> None:
    # A blob-orphan: artifacts present, no manifest.
    workspace.blobs.put_blob("files/orphanedfile/report.pdf", b"%PDF-1.4\n")
    report = check_workspace(workspace)
    assert any(i.target_type == "file" and i.kind == "missing_metadata" for i in report.issues)


def test_corrupt_file_metadata_does_not_crash(workspace: Workspace) -> None:
    """A corrupt file.json must be reported, not crash the whole walk."""
    fid = "corruptfileid"
    workspace.blobs.put_blob(f"files/{fid}/report.pdf", b"%PDF-1.4\n")  # makes the id visible
    # Inject a corrupt manifest directly on disk — put_doc only accepts a valid
    # dict (can't write invalid JSON) and put_blob refuses a document key. This is
    # a LocalStore-specific failure mode (the manifest is a JSON file that get_doc
    # parses); local_path is the sanctioned filesystem escape for such fixtures.
    workspace.local_path(f"{layout.file_prefix(fid)}file.json").write_text(
        "{not valid json", encoding="utf-8"
    )
    report = check_workspace(workspace)
    assert any(
        i.target_type == "file" and i.target_id == fid and i.kind == "corrupt_metadata"
        for i in report.issues
    )


def test_corrupt_docset_metadata_does_not_crash(workspace: Workspace) -> None:
    did = "corruptdocsetid"
    workspace.blobs.put_blob(f"docsets/{did}/extraction-schema.rnc", b"start = text\n")
    # Corrupt manifest on disk — LocalStore-specific; see
    # test_corrupt_file_metadata_does_not_crash for the rationale.
    workspace.local_path(f"{layout.docset_prefix(did)}docset.json").write_text(
        "{not valid json", encoding="utf-8"
    )
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
    workspace.blobs.put_blob(f"files/{bad}/report.pdf", b"%PDF-1.4\n")
    # Corrupt manifest on disk — LocalStore-specific; see
    # test_corrupt_file_metadata_does_not_crash for the rationale.
    workspace.local_path(f"{layout.file_prefix(bad)}file.json").write_text(
        "{not json", encoding="utf-8"
    )
    workspace.blobs.put_blob(f"files/{good}/report.pdf", b"%PDF-1.4\n")  # blob-orphan: no manifest
    report = check_workspace(workspace)
    issues_by_id = {i.target_id: i.kind for i in report.issues if i.target_type == "file"}
    assert issues_by_id.get(bad) == "corrupt_metadata"
    assert issues_by_id.get(good) == "missing_metadata"


def test_check_no_longer_falls_back_to_any_pdf(workspace: Workspace) -> None:
    """If the named PDF is missing, surface missing_pdf rather than silently
    using a different PDF that happens to be in the directory."""
    fid = "fabfileabcde"
    workspace.docs.put_doc(
        "files",
        fid,
        {
            "id": "fabfileabcde",
            "original_path": "/tmp/x.pdf",
            "original_filename": "x.pdf",
            "sha256": "deadbeef",
            "added_at": "2026-05-08T00:00:00Z",
            "page_count": 1,
        },
    )
    workspace.blobs.put_blob(layout.file_source_key(fid, "something_else.pdf"), b"%PDF-1.4\n")
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

    xml_key = layout.dgml_xml_key(ds.id, f.record.id, "doc")
    workspace.blobs.put_blob(
        xml_key,
        (
            b'<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" xmlns:docset="http://x/ns">'
            b"<dg:extraction>"
            b'<docset:Total dg:origin="computed" dg:value="10">10</docset:Total>'
            b"</dg:extraction></dg:chunk>"
        ),
    )
    report = check_workspace(workspace)
    flagged = [i for i in report.issues if i.kind == "computed_field_unattributed"]
    assert len(flagged) == 1
    assert "Total" in flagged[0].message
    assert flagged[0].target_id == ds.id

    workspace.blobs.put_blob(
        xml_key,
        (
            b'<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" xmlns:docset="http://x/ns">'
            b"<dg:extraction>"
            b'<docset:Part xml:id="p1" dg:origin="1 1 2 3 4">4</docset:Part>'
            b'<docset:Total dg:origin="computed" dg:value="10" '
            b'dg:itemprop="computedFrom" dg:href="#p1">10</docset:Total>'
            b"</dg:extraction></dg:chunk>"
        ),
    )
    report = check_workspace(workspace)
    assert not [i for i in report.issues if i.kind == "computed_field_unattributed"]


@pytest.mark.parametrize("debug", [True, False])
def test_reextract_hybrid_threads_debug(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, debug: bool
) -> None:
    """The hybrid re-extraction branch forwards ``debug`` to
    ``extract_text_hybrid`` so ``dgml check --debug`` records hybrid_merge
    usage (and plain ``check`` records nothing)."""
    import dgml_core.consistency as consistency
    from dgml_core.text_extraction import ExtractDigitalResult

    captured: dict[str, Any] = {}

    monkeypatch.setattr(consistency, "load_ocr_config", lambda ws: object())
    monkeypatch.setattr(consistency, "load_text_extraction_config", lambda ws: object())

    def fake_hybrid(*args: Any, **kwargs: Any) -> ExtractDigitalResult:
        captured["debug"] = kwargs.get("debug")
        return ExtractDigitalResult(pages_written=1, pages_with_words=1, total_words=1)

    monkeypatch.setattr(consistency, "extract_text_hybrid", fake_hybrid)

    fid = "fid"
    workspace.blobs.put_blob(f"files/{fid}/doc.pdf", b"%PDF-1.4\n")
    source_key = layout.file_source_key(fid, "doc.pdf")
    _reextract(workspace, source_key, fid, "hybrid", verbose=False, debug=debug)

    assert captured["debug"] is debug
