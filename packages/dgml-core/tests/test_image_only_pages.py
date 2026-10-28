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

"""Digital-mode pages that are a full-page image with little or no text.

A scanned page added with the default ``--text-mode digital`` has no text
layer, or one of a few stray words (a Bates stamp, a page number). The
fixtures are small synthetic PDFs: scans like that, next to good digital
pages that must not change.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from dgml_core import layout
from dgml_core.consistency import check_workspace
from dgml_core.errors import OcrFailed, load_recorded_errors
from dgml_core.files import FileStore
from dgml_core.ocr import (
    BUILTIN_OCR_PROVIDERS,
    OcrConfig,
    OcrProvider,
    OcrProviderName,
    extract_text_ocr,
)
from dgml_core.storage import Workspace
from dgml_core.text_extraction import (
    MIN_SCAN_TEXT_WORDS,
    PageTextDefect,
    extract_text_digital,
)

from .conftest import make_fake_png, write_ocr_config

pytest.importorskip("pdfminer")

GOOD_LINES = [
    "Commercial policy change request",
    "Agency Acrisure Insurance Services LLC",
    "Policy number 0731-AB effective 11/06/2023",
    "Add vehicle 2019 Freightliner Cascadia tractor",
]
STRAY = ["Bates", "ABC000123"]  # fewer than MIN_SCAN_TEXT_WORDS


# --- synthetic PDF builder -----------------------------------------------------


def _cid_hex(word: str) -> str:
    """Glyph ids in an ``Identity-H`` font with no ToUnicode CMap."""
    return "".join(f"{ord(ch) - 29:04X}" for ch in word)


def _write_pdf(path: Path, pages: list[dict[str, Any]]) -> None:
    """Write a PDF whose pages are described by dicts of:

    - ``image``: paint one image over 95% of the page (a scan) or, with
      ``image_scale``, over that fraction of it;
    - ``text``: lines in Helvetica (a font pdfminer resolves to Unicode);
    - ``cid_words``: words in a Type0 ``Identity-H`` font with no ToUnicode,
      one per line, which pdfminer can only report as ``(cid:N)``.
    """
    out = bytearray()
    offsets: list[int] = []

    def add_object(body: bytes) -> int:
        offsets.append(len(out))
        obj_num = len(offsets)
        out.extend(f"{obj_num} 0 obj\n".encode())
        out.extend(body)
        out.extend(b"\nendobj\n")
        return obj_num

    out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    width, height = 612, 792
    n = len(pages)
    catalog_id, pages_id, helv_id, cid_id, desc_id, fd_id, image_id = 1, 2, 3, 4, 5, 6, 7
    page_ids = list(range(8, 8 + n))
    content_ids = list(range(8 + n, 8 + 2 * n))

    add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    add_object(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    add_object(
        f"<< /Type /Font /Subtype /Type0 /BaseFont /TDZHWY+Arial-Identity-H "
        f"/Encoding /Identity-H /DescendantFonts [{desc_id} 0 R] >>".encode()
    )
    add_object(
        f"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /TDZHWY+Arial "
        f"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        f"/FontDescriptor {fd_id} 0 R /DW 600 >>".encode()
    )
    add_object(
        b"<< /Type /FontDescriptor /FontName /TDZHWY+Arial /Flags 32 "
        b"/FontBBox [0 -200 1000 900] /ItalicAngle 0 /Ascent 900 /Descent -200 "
        b"/CapHeight 700 /StemV 80 >>"
    )
    pixels = bytes([200, 200, 200, 200])
    assert (
        add_object(
            b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 "
            b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length 4 >>\nstream\n"
            + pixels
            + b"\nendstream"
        )
        == image_id
    )

    for pid, cid in zip(page_ids, content_ids, strict=True):
        body = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {width} {height}] "
            f"/Contents {cid} 0 R /Resources << /Font << /F1 {helv_id} 0 R "
            f"/F2 {cid_id} 0 R >> /XObject << /Im0 {image_id} 0 R >> >> >>"
        ).encode()
        assert add_object(body) == pid

    for spec, cid in zip(pages, content_ids, strict=True):
        ops: list[str] = []
        if spec.get("image") or spec.get("image_scale"):
            scale = float(spec.get("image_scale", 0.95))
            ops.append(f"q {width * scale:.2f} 0 0 {height * scale:.2f} 15 20 cm /Im0 Do Q")
        y = 740
        for line in spec.get("text", []):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"BT /F1 11 Tf 72 {y} Td ({escaped}) Tj ET")
            y -= 14
        for word in spec.get("cid_words", []):
            ops.append(f"BT /F2 11 Tf 72 {y} Td <{_cid_hex(word)}> Tj ET")
            y -= 14
        stream = ("\n".join(ops) + "\n").encode()
        body = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream"
        assert add_object(body) == cid

    xref = len(out)
    out.extend(f"xref\n0 {len(offsets) + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        (
            f"trailer\n<< /Size {len(offsets) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(bytes(out))


def _bad_pages() -> list[dict[str, Any]]:
    """Page 1 good digital, 2 a scan with no text, 3 a scan carrying a few
    stray words, 4 good digital again."""
    return [
        {"text": GOOD_LINES},
        {"image": True},
        {"image": True, "text": [" ".join(STRAY)]},
        {"text": GOOD_LINES},
    ]


def _good_pages() -> list[dict[str, Any]]:
    """Pages the rule must leave alone: plain text; text over a full-page
    image (letterhead, or a scan with a real text layer); a small logo over
    text; a page of unresolved glyphs (not this rule's business); and a blank
    page with no image."""
    return [
        {"text": GOOD_LINES},
        {"image": True, "text": GOOD_LINES * 2},
        {"image_scale": 0.1, "text": GOOD_LINES},
        {"cid_words": [f"AGENCY{i}" for i in range(20)]},
        {},
    ]


@pytest.fixture
def bad_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "scanned.pdf"
    _write_pdf(path, _bad_pages())
    return path


@pytest.fixture
def good_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "good.pdf"
    _write_pdf(path, _good_pages())
    return path


def _words(dir_: Path, page: int) -> list[str]:
    data = json.loads((dir_ / f"page_{page}.json").read_text())
    return [w["t"] for w in data["words"]]


def _ws_words(ws: Workspace, file_id: str, page: int) -> list[str]:
    data = ws.read_page_text(file_id, page)
    assert data is not None
    return [w["t"] for w in data["words"]]


# --- fake OCR provider + page images --------------------------------------------


OCR_CALLS: list[int] = []


class _FakeOcr(OcrProvider):
    name = OcrProviderName.AZURE.value
    config_fields = frozenset({"endpoint"})
    fail = False

    @classmethod
    def parse_config(cls, config: OcrConfig) -> OcrConfig:
        return config

    def __init__(self, config: OcrConfig) -> None:
        self.config = config

    def analyze_image(
        self, image_bytes: bytes, image_dims_px: tuple[int, int], page_num: int
    ) -> list[dict[str, Any]]:
        if _FakeOcr.fail:
            raise OcrFailed(f"simulated provider failure on page {page_num}")
        OCR_CALLS.append(page_num)
        return [{"t": f"OCR{page_num}", "l": [10, 10, 60, 30]}]


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    OCR_CALLS.clear()
    _FakeOcr.fail = False


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch, workspace: Workspace) -> Workspace:
    """Workspace whose ``[ocr]`` names "azure", repointed at the offline fake."""
    monkeypatch.setitem(BUILTIN_OCR_PROVIDERS, OcrProviderName.AZURE.value, f"{__name__}:_FakeOcr")
    write_ocr_config(workspace, {"provider": "azure", "endpoint": "https://x/"})
    return workspace


@pytest.fixture
def no_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``[ocr]`` and no on-device default: the non-macOS contract."""
    monkeypatch.setattr("dgml_core.ocr.sys.platform", "linux")


@pytest.fixture(autouse=True)
def _fake_render(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed one fake PNG per page instead of running a real renderer, so the
    suite needs no ghostscript and OCR reads deterministic bytes."""
    from pdfminer.high_level import extract_pages

    def render(self: FileStore, pdf_path: Path, file_id: str, **_: Any) -> None:
        n = sum(1 for _ in extract_pages(str(pdf_path), laparams=None))
        for page in range(1, n + 1):
            self.ws.blobs.put_blob(
                layout.file_page_image_key(file_id, page),
                make_fake_png(2550, 3300, f"p{page}".encode()),
            )

    monkeypatch.setattr(FileStore, "_render_pages", render)


# --- detection ----------------------------------------------------------------------


def test_detects_scanned_pages_without_text(bad_pdf: Path, tmp_path: Path) -> None:
    out = tmp_path / "pt"
    result = extract_text_digital(bad_pdf, out, file_id="f", dpi=72)
    assert result.defects == {2: PageTextDefect.IMAGE_ONLY, 3: PageTextDefect.STRAY_TEXT}
    # The words are still written as pdfminer produced them; acting on the
    # verdict is the caller's job.
    assert _words(out, 3) == STRAY
    assert len(STRAY) < MIN_SCAN_TEXT_WORDS


def test_good_digital_pages_are_not_flagged(good_pdf: Path, tmp_path: Path) -> None:
    result = extract_text_digital(good_pdf, tmp_path / "pt", file_id="f", dpi=72)
    assert result.defects == {}
    # A healthy summary is exactly what it always was.
    assert set(result.to_summary()) == {"mode", "pages_written", "pages_with_words", "total_words"}


# --- file add: OCR fallback ---------------------------------------------------------


def test_file_add_digital_takes_scanned_pages_from_ocr(
    fake_ocr: Workspace, bad_pdf: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="dgml_core")
    result = FileStore(fake_ocr).add(bad_pdf)
    fid = result.record.id

    assert result.record.text_mode == "digital"
    assert sorted(OCR_CALLS) == [2, 3]  # only the scanned pages
    assert _ws_words(fake_ocr, fid, 2) == ["OCR2"]
    assert _ws_words(fake_ocr, fid, 3) == ["OCR3"]
    assert _ws_words(fake_ocr, fid, 1)[:2] == ["Commercial", "policy"]
    assert _ws_words(fake_ocr, fid, 4)[:2] == ["Commercial", "policy"]

    assert result.text_extraction_error is None
    assert result.text_extraction is not None
    assert result.text_extraction["ocr_fallback_pages"] == [2, 3]
    assert result.text_extraction["pages_with_words"] == 4
    assert "unusable_pages" not in result.text_extraction
    assert not [e for e in load_recorded_errors(fake_ocr, fid) if e.operation == "text_extraction"]
    assert "pages 2-3" in caplog.text and "OCR" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_file_add_good_digital_pdf_is_unchanged_and_never_touches_ocr(
    workspace: Workspace,
    good_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def boom(*_: Any, **__: Any) -> Any:
        raise AssertionError("OCR config must not be read for a healthy digital file")

    monkeypatch.setattr("dgml_core.ocr.load_ocr_config", boom)
    caplog.set_level(logging.INFO, logger="dgml_core")
    result = FileStore(workspace).add(good_pdf)

    reference = tmp_path / "ref"
    extract_text_digital(good_pdf, reference, file_id=result.record.id)
    for page in range(1, 6):
        stored = workspace.blobs.get_blob(layout.file_page_text_key(result.record.id, page))
        assert stored == (reference / f"page_{page}.json").read_bytes()
    assert result.text_extraction is not None
    assert set(result.text_extraction) == {
        "mode",
        "pages_written",
        "pages_with_words",
        "total_words",
    }
    # The blank page is reported exactly as before.
    assert result.text_extraction_error == "1/5 pages had no extractable digital text"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- file add: no OCR available -------------------------------------------------------


def test_file_add_digital_without_ocr_warns_and_records_error(
    workspace: Workspace, bad_pdf: Path, no_ocr: None, caplog: pytest.LogCaptureFixture
) -> None:
    result = FileStore(workspace).add(bad_pdf)
    fid = result.record.id

    # Nothing is invented and nothing real is dropped.
    assert _ws_words(workspace, fid, 2) == []
    assert _ws_words(workspace, fid, 3) == STRAY
    assert _ws_words(workspace, fid, 1)[:2] == ["Commercial", "policy"]

    err = result.text_extraction_error
    assert err is not None
    assert "2/4 pages" in err
    assert "pages 2-3" in err
    assert "no OCR provider configured" in err
    assert "--text-mode ocr" in err
    assert result.text_extraction is not None
    assert result.text_extraction["unusable_pages"] == {"2": "image_only", "3": "stray_text"}
    errs = [e for e in load_recorded_errors(workspace, fid) if e.operation == "text_extraction"]
    assert errs and all(e.permanent for e in errs)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert fid in message and "pages 2-3" in message
    assert "dgml check --retry-errors" in message


def test_file_add_digital_ocr_failure_degrades_like_no_provider(
    fake_ocr: Workspace, bad_pdf: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _FakeOcr.fail = True
    result = FileStore(fake_ocr).add(bad_pdf)
    fid = result.record.id
    assert _ws_words(fake_ocr, fid, 2) == []
    assert _ws_words(fake_ocr, fid, 3) == STRAY
    assert result.text_extraction_error is not None
    assert "simulated provider failure" in result.text_extraction_error
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


# --- dgml check -----------------------------------------------------------------------


def test_check_retry_errors_recovers_once_ocr_is_configured(
    workspace: Workspace,
    bad_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The remedy the warning names is one that works."""
    monkeypatch.setattr("dgml_core.ocr.sys.platform", "linux")
    fid = FileStore(workspace).add(bad_pdf).record.id
    assert _ws_words(workspace, fid, 2) == []

    monkeypatch.setitem(BUILTIN_OCR_PROVIDERS, OcrProviderName.AZURE.value, f"{__name__}:_FakeOcr")
    write_ocr_config(workspace, {"provider": "azure", "endpoint": "https://x/"})
    check_workspace(workspace, retry_errors=True)

    assert _ws_words(workspace, fid, 2) == ["OCR2"]
    assert _ws_words(workspace, fid, 3) == ["OCR3"]
    assert _ws_words(workspace, fid, 1)[:2] == ["Commercial", "policy"]
    errs = [e for e in load_recorded_errors(workspace, fid) if e.operation == "text_extraction"]
    assert not errs


# --- extract_text_ocr page subset -----------------------------------------------------


def test_extract_text_ocr_page_subset_leaves_other_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(BUILTIN_OCR_PROVIDERS, OcrProviderName.AZURE.value, f"{__name__}:_FakeOcr")
    images = tmp_path / "img"
    images.mkdir()
    for p in (1, 2, 3):
        (images / f"page_{p}.png").write_bytes(make_fake_png(100, 100, f"p{p}".encode()))
    out = tmp_path / "pt"
    out.mkdir()
    (out / "page_1.json").write_text('{"page":1,"words":[{"t":"keep","l":[0,0,1,1]}]}\n')
    cfg = OcrConfig(provider=OcrProviderName.AZURE, options={"endpoint": "https://x/"})
    result = extract_text_ocr(
        tmp_path / "unused.pdf", out, file_id="f", page_images_dir=images, config=cfg, pages={2}
    )
    assert result.pages_written == 1
    assert OCR_CALLS == [2]
    assert _words(out, 1) == ["keep"]
    assert _words(out, 2) == ["OCR2"]
    assert not (out / "page_3.json").exists()


def test_extract_text_ocr_page_subset_requires_each_page_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(BUILTIN_OCR_PROVIDERS, OcrProviderName.AZURE.value, f"{__name__}:_FakeOcr")
    images = tmp_path / "img"
    images.mkdir()
    (images / "page_1.png").write_bytes(make_fake_png(100, 100))
    cfg = OcrConfig(provider=OcrProviderName.AZURE, options={"endpoint": "https://x/"})
    with pytest.raises(OcrFailed, match="page 2"):
        extract_text_ocr(
            tmp_path / "unused.pdf",
            tmp_path / "pt",
            file_id="f",
            page_images_dir=images,
            config=cfg,
            pages=[2],
        )
