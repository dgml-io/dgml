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

"""Tests for hybrid text-mode (digital + OCR merged by bounding-box overlap)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
from dgml_core import layout
from dgml_core.errors import OcrFailed
from dgml_core.files import FileStore
from dgml_core.hybrid import (
    LEVENSHTEIN_THRESHOLD,
    MERGE_BATCH_SIZE,
    _iou,
    _levenshtein_distance,
    _merge_words,
    _raster_page_numbers,
    extract_text_hybrid,
)
from dgml_core.ocr import BUILTIN_OCR_PROVIDERS, OcrConfig, OcrProvider, OcrProviderName
from dgml_core.storage import Workspace
from dgml_core.text_extraction import MAX_CID_WORDS_PER_PAGE, TextMode

from .conftest import make_fake_png, write_ocr_config

# ---------------------------------------------------------------------------
# IoU sanity
# ---------------------------------------------------------------------------


def _write_page(dir_: Path, page: int, payload: dict[str, Any]) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"page_{page}.json").write_text(json.dumps(payload))


def test_merge_rotates_digital_into_deskewed_frame(tmp_path: Path) -> None:
    """On a deskewed OCR page, the digital boxes (still in the original render's
    frame) are rotated by the same transform and then merged normally — so
    high-fidelity digital text survives on rotated pages instead of being
    dropped. Here a 180° rotation maps the digital box onto the matching OCR
    box, so digital wins on agreement and its (rotated) box is what's emitted."""
    from dgml_core.hybrid import _merge_into

    digital_dir = tmp_path / "digital"
    ocr_dir = tmp_path / "ocr"
    out_dir = tmp_path / "out"

    # Digital saw the word in the ORIGINAL (pre-rotation) frame.
    _write_page(
        digital_dir,
        1,
        {
            "file_id": "fid",
            "page": 1,
            "width": 100,
            "height": 60,
            "words": [{"t": "hello", "l": [10, 10, 30, 20]}],
        },
    )
    # OCR deskewed the page 180°; its word sits where the digital word lands
    # after the same 180° rotation about the page centre: [70, 40, 90, 50].
    _write_page(
        ocr_dir,
        1,
        {
            "file_id": "fid",
            "page": 1,
            "width": 100,
            "height": 60,
            "words": [{"t": "hello", "l": [70, 40, 90, 50]}],
            "rotation": 180.0,
        },
    )

    _merge_into(digital_dir, ocr_dir, out_dir, file_id="fid")

    payload = json.loads((out_dir / "page_1.json").read_text())
    # Digital was rotated to [70, 40, 90, 50], overlaps the identical-text OCR
    # word, so digital wins on agreement — one word, at the rotated position.
    assert payload["words"] == [{"t": "hello", "l": [70, 40, 90, 50]}]
    assert payload["rotation"] == 180.0
    assert (payload["width"], payload["height"]) == (100, 60)


def test_iou_identical_boxes_is_one() -> None:
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    assert _iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_touching_boxes_is_zero() -> None:
    """Edge-touching boxes have zero overlap area."""
    assert _iou([0, 0, 10, 10], [10, 0, 20, 10]) == 0.0


def test_iou_partial_overlap() -> None:
    # 10x10 boxes overlapping by 5x5 → intersection 25, union 175 → 1/7.
    iou = _iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert iou == pytest.approx(25 / 175)


@pytest.fixture(autouse=True)
def _hybrid_logs_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """Capture ``dgml_core.hybrid``'s INFO lines — what ``dgml --verbose`` shows."""
    caplog.set_level(logging.INFO, logger="dgml_core")


def _logged(caplog: pytest.LogCaptureFixture) -> str:
    """Everything logged so far, one message per line, without level prefixes."""
    return "\n".join(r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Merge rules
# ---------------------------------------------------------------------------


def test_merge_ocr_only_word_kept_silently(caplog: pytest.LogCaptureFixture) -> None:
    """OCR-only words are always kept — the default outcome, so nothing is
    logged for them beyond the per-page summary."""
    digital: list[dict[str, Any]] = []
    ocr = [{"t": "scanned", "l": [10, 10, 60, 30]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert merged == ocr
    err = _logged(caplog)
    assert "warning" not in err  # OCR-only is not a warning condition
    assert "no matching digital text" not in err
    assert "digital_words=0 ocr_words=1 merged=1" in err


def test_merge_same_text_overlap_takes_digital(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Overlap + identical text → take digital (exact character codes)."""
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "Hello", "l": [11, 11, 61, 31]}]  # nearly identical bbox
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert merged == digital  # digital wins on overlap+similar
    err = _logged(caplog)
    assert "warning" not in err
    assert "digital_words=1 ocr_words=1 merged=1" in err


def test_merge_normalizes_dashes_before_levenshtein(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dash-family code points are folded to ASCII hyphen-minus before
    measuring distance, so digital ``out<MINUS>of<MINUS>pocket`` and OCR
    ``out-of-pocket`` count as identical — no warning, digital wins, and
    the original digital text is preserved in the output.
    """
    minus = chr(0x2212)  # MINUS SIGN — what pdfminer emits in some PDFs
    digital_text = f"out{minus}of{minus}pocket"
    ocr_text = "out-of-pocket"

    # Raw helper still sees the substitutions; normalization is merge-internal.
    assert _levenshtein_distance(digital_text, ocr_text) == 2

    digital = [{"t": digital_text, "l": [10, 10, 100, 30]}]
    ocr = [{"t": ocr_text, "l": [11, 11, 101, 31]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    # Digital wins (normalized distance == 0). The original digital text
    # is preserved so downstream consumers see what the PDF actually held.
    assert merged == digital
    err = _logged(caplog)
    assert "warning" not in err


def test_merge_normalizes_all_dash_variants(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every dash code point in :data:`_DASH_TABLE` folds to ASCII hyphen for
    comparison purposes. Each variant overlapping an ASCII-hyphen OCR word
    should resolve silently to digital."""
    from dgml_core.hybrid import _DASH_CODEPOINTS, _normalize_for_compare

    for cp in _DASH_CODEPOINTS:
        ch = chr(cp)
        assert _normalize_for_compare(f"a{ch}b") == "a-b", (
            f"expected U+{cp:04X} to fold to ASCII hyphen"
        )

        digital = [{"t": f"a{ch}b", "l": [10, 10, 60, 30]}]
        ocr = [{"t": "a-b", "l": [10, 10, 60, 30]}]
        merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
        assert merged == digital, f"digital should win for U+{cp:04X}"

    # All cases should have been silent on the warning channel.
    assert "warning" not in _logged(caplog)


def test_merge_conflicting_text_above_threshold_takes_ocr_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Overlap + Levenshtein > threshold → take OCR with a warning."""
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "Greetings", "l": [10, 10, 60, 30]}]  # distance 7
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert merged == ocr  # OCR wins on overlap+different
    err = _logged(caplog)
    assert "'Hello'" in err
    assert "'Greetings'" in err
    assert "using OCR" in err
    assert "levenshtein=" in err


def test_merge_digital_only_word_is_dropped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A digital word with no overlapping OCR word is assumed invisible-to-
    human-eye and **dropped** (not kept) — but the drop is logged."""
    digital = [{"t": "stamp", "l": [200, 200, 260, 220]}]
    ocr = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    # The OCR word is kept; the digital-only "stamp" is dropped.
    assert merged == [{"t": "Hello", "l": [10, 10, 60, 30]}]
    err = _logged(caplog)
    assert "'stamp'" in err
    assert "not detected by OCR" in err
    assert "dropping" in err


def test_merge_digital_only_region_drop_is_one_aggregated_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A dropped digital-only region logs ONE line naming a capped sample of
    its words — a watermark or hidden text layer must not flood the log with
    a line per word."""
    from dgml_core.hybrid import _DROP_LOG_SAMPLE

    n = _DROP_LOG_SAMPLE + 3
    # One tight cluster of overlapping boxes → a single digital-only region.
    digital = [{"t": f"w{i}", "l": [10, 10, 60, 30]} for i in range(n)]
    ocr = [{"t": "visible", "l": [500, 500, 560, 520]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert merged == ocr
    err = _logged(caplog)
    assert err.count("dropping") == 1
    assert f"{n} digital word(s) not detected by OCR" in err
    assert "'w0'" in err  # sampled tokens are named…
    assert "(+3 more)" in err  # …and the overflow is a count


def test_merge_combination_warns_for_each_problem(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All four cases mixed on one page.

    - OCR ``Hello`` ↔ digital ``Hello`` at same position → digital wins (silent).
    - OCR ``Greetings`` ↔ digital ``world`` at same position, very different →
      OCR wins, one warning.
    - OCR ``page`` with no digital overlap → OCR kept, silently (the default).
    - Digital ``stamp`` with no OCR overlap → dropped with a warning.
    """
    digital = [
        {"t": "Hello", "l": [10, 10, 60, 30]},
        {"t": "world", "l": [70, 10, 120, 30]},
        {"t": "stamp", "l": [400, 400, 460, 420]},
    ]
    ocr = [
        {"t": "Hello", "l": [10, 10, 60, 30]},
        {"t": "Greetings", "l": [70, 10, 120, 30]},  # distance 9 — clearly different
        {"t": "page", "l": [200, 200, 240, 220]},  # OCR-only
    ]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert merged == [
        {"t": "Hello", "l": [10, 10, 60, 30]},  # digital, taken on similarity
        {"t": "Greetings", "l": [70, 10, 120, 30]},  # OCR, conflict winner
        {"t": "page", "l": [200, 200, 240, 220]},  # OCR-only
    ]
    err = _logged(caplog)
    assert err.count("using OCR") == 1  # only the world/Greetings pair
    assert err.count("dropping") == 1  # only "stamp"
    assert "no matching digital text" not in err  # OCR-only "page" is the default


# ---------------------------------------------------------------------------
# Split / merge regions (tokenization mismatch)
# ---------------------------------------------------------------------------


def test_boxes_overlap_coverage_catches_contained_box() -> None:
    """A small box mostly inside a big one overlaps by coverage even when IoU
    is below the threshold — that's what fixes split/merge matching."""
    from dgml_core.hybrid import _boxes_overlap

    big = [0, 0, 100, 20]
    small = [0, 0, 30, 20]  # fully inside big horizontally
    assert _iou(big, small) < 0.5  # IoU alone would miss it
    assert _boxes_overlap(big, small)  # coverage rescues it
    assert not _boxes_overlap([0, 0, 10, 10], [50, 50, 60, 60])  # disjoint


def test_merge_ocr_split_agreeing_keeps_digital_tokens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """OCR splits a span digital keeps whole; texts agree → keep digital
    (PDF font is more reliable than OCR's tokenization)."""
    digital = [{"t": "Analysis)Inadequate", "l": [814, 2268, 1229, 2314]}]
    ocr = [
        {"t": "Analysis)", "l": [813, 2264, 985, 2315]},
        {"t": "Inadequate", "l": [994, 2263, 1230, 2316]},
    ]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=16)
    assert merged == digital  # texts agree → digital wins regardless of token count
    # Agreement is the designed outcome, so the region logs nothing.
    assert "tokenization mismatch" not in _logged(caplog)


def test_merge_digital_split_agreeing_keeps_digital_tokens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Digital splits a span OCR keeps whole; texts agree → keep digital."""
    digital = [
        {"t": "Analysis)", "l": [813, 2264, 985, 2315]},
        {"t": "Inadequate", "l": [994, 2263, 1230, 2316]},
    ]
    ocr = [{"t": "Analysis)Inadequate", "l": [814, 2268, 1229, 2314]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=16)
    assert merged == digital  # texts agree → digital wins, silently
    assert "tokenization mismatch" not in _logged(caplog)


def test_merge_split_tie_goes_to_digital(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Equal token counts in a mixed region with agreeing text → digital."""
    # Identical boxes force a single 2-vs-2 region.
    box = [0, 0, 80, 10]
    digital = [{"t": "AB", "l": box}, {"t": "CD", "l": box}]
    ocr = [{"t": "AC", "l": box}, {"t": "BD", "l": box}]
    # Concats "ABCD" vs "ACBD" — distance 2, within threshold → "agree".
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert merged == digital  # 2 == 2 tie → digital, silently (text agrees)
    assert "tokenization mismatch" not in _logged(caplog)


def test_merge_split_disagreeing_takes_ocr(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mixed region whose concatenations differ beyond threshold → OCR, even
    though digital here has fewer tokens."""
    digital = [{"t": "ABCDEFGH", "l": [0, 0, 80, 10]}]
    ocr = [
        {"t": "XYZ", "l": [0, 0, 40, 10]},
        {"t": "WVUT", "l": [40, 0, 80, 10]},
    ]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert merged == ocr
    err = _logged(caplog)
    assert "text differs" in err
    assert "keeping OCR's 2 tokens" in err


def test_merge_dollar_blanks_vs_ocr_dollar_takes_ocr(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Real 1:1-via-coverage case: digital captured fill-in underscores OCR
    didn't see. Boxes overlap by containment; texts differ → OCR's '$'."""
    digital = [{"t": '$_____."', "l": [1454, 123, 1635, 169]}]
    ocr = [{"t": "$", "l": [1455, 120, 1481, 170]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert merged == ocr
    err = _logged(caplog)
    assert "using OCR" in err


# ---------------------------------------------------------------------------
# Levenshtein helper
# ---------------------------------------------------------------------------


def test_levenshtein_identical_is_zero() -> None:
    assert _levenshtein_distance("Hello", "Hello") == 0


def test_levenshtein_empty_strings() -> None:
    assert _levenshtein_distance("", "abc") == 3
    assert _levenshtein_distance("abc", "") == 3
    assert _levenshtein_distance("", "") == 0


def test_levenshtein_substitution() -> None:
    # "Hello" → "He11o" is two substitutions.
    assert _levenshtein_distance("Hello", "He11o") == 2


def test_levenshtein_insertion_deletion() -> None:
    assert _levenshtein_distance("cat", "cats") == 1
    assert _levenshtein_distance("cats", "cat") == 1


def test_merge_levenshtein_boundary_at_threshold_takes_digital(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Distance == threshold counts as similar — digital wins."""
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "He11o", "l": [10, 10, 60, 30]}]  # distance 2
    assert _levenshtein_distance("Hello", "He11o") == LEVENSHTEIN_THRESHOLD

    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert merged == digital
    assert "warning" not in _logged(caplog)


def test_merge_levenshtein_just_over_threshold_takes_ocr(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Distance == threshold + 1 → OCR wins with a warning."""
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "He11x", "l": [10, 10, 60, 30]}]  # distance 3
    assert _levenshtein_distance("Hello", "He11x") == LEVENSHTEIN_THRESHOLD + 1

    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert merged == ocr
    err = _logged(caplog)
    assert "using OCR" in err
    assert f"levenshtein={LEVENSHTEIN_THRESHOLD + 1}" in err


# ---------------------------------------------------------------------------
# CID guard: too many "(cid:" words in digital text → OCR-only for page
# ---------------------------------------------------------------------------


def test_merge_cid_guard_falls_back_to_ocr_only(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When pdfminer couldn't resolve glyphs to Unicode (lots of '(cid:N)'
    tokens), the digital output is unusable. The merge logs a 'unicode
    error' and returns just the OCR words."""
    digital = [
        {"t": f"(cid:{i})", "l": [10 * i, 10, 10 * i + 8, 30]}
        for i in range(MAX_CID_WORDS_PER_PAGE + 1)
    ]
    # Add one normal-looking digital word too; it should still be ignored
    # because the page as a whole is junk.
    digital.append({"t": "real", "l": [500, 500, 540, 520]})
    ocr = [
        {"t": "scanned", "l": [10, 10, 60, 30]},
        {"t": "page", "l": [70, 10, 110, 30]},
    ]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=7)
    assert merged == ocr
    err = _logged(caplog)
    assert "unicode error" in err
    assert "page=7" in err
    assert f"{MAX_CID_WORDS_PER_PAGE + 1} words containing '(cid:'" in err


def test_merge_cid_guard_below_threshold_does_not_trigger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At-or-below ``MAX_CID_WORDS_PER_PAGE`` is tolerated — the normal merge
    runs and individual cid words are subject to the regular rules."""
    digital = [
        {"t": f"(cid:{i})", "l": [10 * i, 10, 10 * i + 8, 30]}
        for i in range(MAX_CID_WORDS_PER_PAGE)
    ]
    ocr = [{"t": "scanned", "l": [500, 500, 560, 520]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1)
    # The cid tokens are digital-only (no OCR overlap) so they all drop.
    # The OCR word stays.
    assert merged == ocr
    err = _logged(caplog)
    assert "unicode error" not in err


# ---------------------------------------------------------------------------
# Scan guard: a full-page raster image means the digital text is a baked-in
# OCR layer, not the document's own character codes
# ---------------------------------------------------------------------------


def _write_scanned_pdf(path: Path, *, pages: int, image_scale: float, text: str) -> None:
    """Write a PDF whose every page paints one image scaled to ``image_scale``
    of the page in each dimension, then draws ``text`` over it.

    ``image_scale=0.9`` models a scan (the picture IS the page); a small value
    models a born-digital page carrying a logo. The image is a 2x2 grey JPEG-
    free raw sample, big enough for pdfminer to report a bbox and nothing more
    — the merge never looks at pixels, only at how much of the page they cover.
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
    draw_w, draw_h = width * image_scale, height * image_scale

    catalog_id, pages_id, font_id = 1, 2, 3
    image_id = 4
    page_obj_ids = list(range(image_id + 1, image_id + 1 + pages))
    content_obj_ids = list(range(page_obj_ids[-1] + 1, page_obj_ids[-1] + 1 + pages))

    add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    add_object(f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode())
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pixels = bytes([200, 200, 200, 200])
    img = (
        b"<< /Type /XObject /Subtype /Image /Width 2 /Height 2 "
        b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Length 4 >>\nstream\n"
        + pixels
        + b"\nendstream"
    )
    assert add_object(img) == image_id

    for page_id, content_id in zip(page_obj_ids, content_obj_ids, strict=True):
        body = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {width} {height}] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> "
            f"/XObject << /Im0 {image_id} 0 R >> >> >>"
        ).encode()
        assert add_object(body) == page_id

    for content_id in content_obj_ids:
        stream_body = (
            f"q {draw_w:.2f} 0 0 {draw_h:.2f} 0 0 cm /Im0 Do Q\n"
            f"BT /F1 24 Tf 100 700 Td ({text}) Tj ET\n"
        ).encode()
        body = f"<< /Length {len(stream_body)} >>\nstream\n".encode() + stream_body + b"endstream"
        assert add_object(body) == content_id

    xref_offset = len(out)
    n_objects = len(offsets)
    out.extend(f"xref\n0 {n_objects + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        (
            f"trailer\n<< /Size {n_objects + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(bytes(out))


def test_raster_page_numbers_finds_full_page_image(tmp_path: Path) -> None:
    pdf = tmp_path / "scan.pdf"
    _write_scanned_pdf(pdf, pages=2, image_scale=0.9, text="baked in")
    assert _raster_page_numbers(pdf) == {1, 2}


def test_raster_page_numbers_ignores_a_small_image(tmp_path: Path) -> None:
    """A born-digital page's own artwork covers a few percent of the page, not
    most of it, so its high-fidelity text layer keeps winning regions."""
    pdf = tmp_path / "logo.pdf"
    _write_scanned_pdf(pdf, pages=1, image_scale=0.1, text="native text")
    assert _raster_page_numbers(pdf) == set()


def test_raster_page_numbers_on_unparseable_pdf_is_empty(tmp_path: Path) -> None:
    """Never fail ingest over the probe: an unreadable PDF just merges the old
    way."""
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"not a pdf at all")
    assert _raster_page_numbers(pdf) == set()


def test_merge_scan_guard_drops_the_baked_layer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The baked layer's `402` would otherwise overrule OCR's `.02` — the two
    are one edit apart, so the mixed-region rule calls them the same word."""
    digital = [{"t": "402", "l": [100, 10, 140, 30]}]
    ocr = [{"t": ".02", "l": [102, 10, 142, 30]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=3, raster_page=True)
    assert merged == ocr
    err = _logged(caplog)
    assert "scan guard" in err
    assert "page=3" in err
    assert "using OCR for the entire page" in err


def test_merge_without_scan_guard_keeps_the_baked_layer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same region on a page that is NOT a scan: digital still wins, which
    is what makes the guard the thing that has to notice."""
    digital = [{"t": "402", "l": [100, 10, 140, 30]}]
    ocr = [{"t": ".02", "l": [102, 10, 142, 30]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=3)
    assert merged == digital
    assert "scan guard" not in _logged(caplog)


def test_merge_scan_guard_silent_when_page_has_no_digital_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A scan with no baked layer needs no guard — the merge already yields
    OCR — so it says nothing."""
    ocr = [{"t": "scanned", "l": [10, 10, 60, 30]}]
    merged = _merge_words([], ocr, file_id="fid", page_num=1, raster_page=True)
    assert merged == ocr
    assert "scan guard" not in _logged(caplog)


def test_extract_text_hybrid_scan_guard_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A scanned PDF's baked text never reaches page_text: the output is the
    OCR words alone, even where a baked word sits right on top of one."""
    pdf = tmp_path / "scan.pdf"
    _write_scanned_pdf(pdf, pages=1, image_scale=0.9, text="baked")
    pages_dir = tmp_path / "page_images"
    _seed_page_images(pages_dir, n=1)
    _install_fake_provider(
        monkeypatch,
        words_by_page={1: [{"t": "baked!", "l": [100, 70, 190, 100]}]},
    )
    out_dir = tmp_path / "page_text"
    cfg = OcrConfig(provider=OcrProviderName.AZURE, options={"endpoint": "https://x/"})
    extract_text_hybrid(pdf, out_dir, file_id="fid", page_images_dir=pages_dir, config=cfg)
    words = json.loads((out_dir / "page_1.json").read_text())["words"]
    assert [w["t"] for w in words] == ["baked!"]
    assert "scan guard" in _logged(caplog)


# ---------------------------------------------------------------------------
# End-to-end extract_text_hybrid (fake OCR provider, real digital path)
# ---------------------------------------------------------------------------


def _install_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    words_by_page: dict[int, list[dict[str, Any]]] | None = None,
    fail_on_page: int | None = None,
) -> None:
    class FakeProvider(OcrProvider):
        name = OcrProviderName.AZURE.value
        # Stands in for Azure, so it declares Azure's option: the framework rejects
        # anything a provider doesn't name, whether or not parse_config checks.
        config_fields = frozenset({"endpoint"})

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            return config

        def __init__(self, config: OcrConfig) -> None:
            self.config = config

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            if fail_on_page is not None and page_num == fail_on_page:
                raise OcrFailed(f"simulated provider failure on page {page_num}")
            if words_by_page is None:
                return []
            return list(words_by_page.get(page_num, []))

    # Repoint the "azure" alias at the fake rather than reaching into a registry:
    # these tests drive the workspace config, which says provider = "azure", so the
    # real resolver still runs end to end.
    monkeypatch.setattr(sys.modules[__name__], "FakeProvider", FakeProvider, raising=False)
    monkeypatch.setitem(
        BUILTIN_OCR_PROVIDERS, OcrProviderName.AZURE.value, f"{__name__}:FakeProvider"
    )


def _seed_page_images(pages_dir: Path, n: int, w: int = 612, h: int = 792) -> None:
    pages_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (pages_dir / f"page_{i}.png").write_bytes(make_fake_png(w, h, f"p{i}".encode()))


def test_extract_text_hybrid_merges_per_page_and_writes_output(
    text_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end: real digital extraction from a fixture PDF, fake OCR words.
    OCR words land in the output; digital words with no OCR overlap are
    treated as invisible-to-human-eye and dropped with warnings."""
    pages_dir = tmp_path / "page_images"
    _seed_page_images(pages_dir, n=2)

    # Digital extraction will find "Hello", "World" on page 1 and
    # "Second", "Page", "Text" on page 2 (see conftest text_pdf fixture). We
    # seed OCR with words placed far from those positions so every digital
    # word is dropped (digital-only branch) and every OCR word lands.
    _install_fake_provider(
        monkeypatch,
        words_by_page={
            1: [{"t": "OCR_P1", "l": [10, 10, 60, 30]}],
            2: [{"t": "OCR_P2", "l": [10, 10, 60, 30]}],
        },
    )

    out_dir = tmp_path / "page_text"
    cfg = OcrConfig(provider=OcrProviderName.AZURE, options={"endpoint": "https://x/"})
    result = extract_text_hybrid(
        text_pdf, out_dir, file_id="fid", page_images_dir=pages_dir, config=cfg
    )

    assert result.pages_written == 2
    assert result.pages_with_words == 2
    page1 = json.loads((out_dir / "page_1.json").read_text())
    page2 = json.loads((out_dir / "page_2.json").read_text())

    p1_texts = [w["t"] for w in page1["words"]]
    assert p1_texts == ["OCR_P1"]

    p2_texts = [w["t"] for w in page2["words"]]
    assert p2_texts == ["OCR_P2"]

    # Each digital word with no OCR overlap should have produced a "dropping"
    # warning on stderr.
    err = _logged(caplog)
    for token in ("Hello", "World", "Second", "Page", "Text"):
        assert f"'{token}'" in err
    assert "dropping" in err


def test_extract_text_hybrid_continues_when_digital_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If pdfminer cannot parse the PDF, hybrid logs a stderr warning and
    proceeds with OCR-only output rather than aborting."""
    # A "PDF" with the right magic but no valid structure — pdfminer raises.
    broken_pdf = tmp_path / "broken.pdf"
    broken_pdf.write_bytes(b"%PDF-1.4\nnot really a pdf\n%%EOF\n")

    pages_dir = tmp_path / "page_images"
    _seed_page_images(pages_dir, n=1)
    _install_fake_provider(
        monkeypatch,
        words_by_page={1: [{"t": "scanned", "l": [10, 10, 60, 30]}]},
    )

    out_dir = tmp_path / "page_text"
    cfg = OcrConfig(provider=OcrProviderName.AZURE, options={"endpoint": "https://x/"})
    result = extract_text_hybrid(
        broken_pdf,
        out_dir,
        file_id="fid",
        page_images_dir=pages_dir,
        config=cfg,
    )

    assert result.pages_written == 1
    page1 = json.loads((out_dir / "page_1.json").read_text())
    assert [w["t"] for w in page1["words"]] == ["scanned"]
    assert "digital extraction failed" in _logged(caplog)


def test_extract_text_hybrid_propagates_ocr_failure(
    text_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages_dir = tmp_path / "page_images"
    _seed_page_images(pages_dir, n=2)
    _install_fake_provider(monkeypatch, fail_on_page=1)

    cfg = OcrConfig(provider=OcrProviderName.AZURE, options={"endpoint": "https://x/"})
    out_dir = tmp_path / "page_text"
    with pytest.raises(OcrFailed):
        extract_text_hybrid(text_pdf, out_dir, file_id="fid", page_images_dir=pages_dir, config=cfg)


# ---------------------------------------------------------------------------
# Wiring: FileStore.add(text_mode=hybrid) and CLI surface
# ---------------------------------------------------------------------------


def test_file_add_hybrid_records_hybrid_mode_and_summary(
    workspace: Workspace,
    text_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`FileStore.add(..., text_mode=HYBRID)` runs digital + OCR + merge and
    records ``text_mode: "hybrid"`` plus a summary tagged ``mode: "hybrid"``."""
    write_ocr_config(
        workspace,
        {
            "provider": "azure",
            "endpoint": "https://example.cognitiveservices.azure.com/",
        },
    )
    _install_fake_provider(
        monkeypatch,
        words_by_page={
            1: [{"t": "OCR_P1", "l": [10, 10, 60, 30]}],
            2: [{"t": "OCR_P2", "l": [10, 10, 60, 30]}],
        },
    )

    # Skip the actual ghostscript render — seed page_images directly via a
    # patch of render_pages so the OCR path finds JPEGs without `gs`.
    import dgml_core.files as files_mod

    def fake_render(
        pdf_path: Path, output_dir: Path, *, dpi: int = 300, config: object = None
    ) -> int:
        _seed_page_images(output_dir, n=2)
        return 2

    monkeypatch.setattr(files_mod, "render_pages", fake_render)

    result = FileStore(workspace).add(text_pdf, text_mode=TextMode.HYBRID)

    assert result.created is True
    assert result.record.text_mode == "hybrid"
    assert result.text_extraction_error is None
    assert result.text_extraction is not None
    assert result.text_extraction["mode"] == "hybrid"
    assert result.text_extraction["pages_written"] == 2

    texts = [
        k
        for k in workspace.blobs.list_blobs(layout.file_text_prefix(result.record.id))
        if k.endswith(".json")
    ]
    assert len(texts) == 2


# ---------------------------------------------------------------------------
# LLM-driven merge (text_extraction config present)
# ---------------------------------------------------------------------------

from collections.abc import Callable  # noqa: E402

from dgml_core.text_extraction_config import TextExtractionConfig  # noqa: E402

_LLM_CONFIG = TextExtractionConfig(
    model="ollama_chat/gemma4:latest", api_base="http://localhost:11434"
)


def _install_fake_call(
    monkeypatch: pytest.MonkeyPatch,
    decider: Callable[[dict[str, Any]], list[dict[str, Any]]],
    *,
    raw: str | None = None,
) -> list[dict[str, Any]]:
    """Patch ``dgml_core.hybrid.call`` with a fake that replies per region.

    Returns a list that records each call's parsed payload, so tests can
    assert whether (and with what) the LLM was invoked. ``decider`` maps a
    region payload dict → its decision list. ``raw`` overrides the reply
    with a literal string (to exercise malformed-response handling).
    """
    calls: list[dict[str, Any]] = []

    def _fake_call(
        config: Any,
        *,
        system_prompt: Any,
        user_content: list[dict[str, Any]],
        cache: bool = False,
    ) -> str:
        payload = json.loads(user_content[0]["text"])
        calls.append(payload)
        if raw is not None:
            return raw
        return json.dumps({c["id"]: decider(c) for c in payload["regions"]})

    monkeypatch.setattr("dgml_core.hybrid.call", _fake_call)
    return calls


def _merge_llm(
    digital: list[dict[str, Any]],
    ocr: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _merge_words(
        digital,
        ocr,
        file_id="fid",
        page_num=1,
        text_extraction_config=_LLM_CONFIG,
    )


def test_llm_not_called_for_identical_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mixed region whose tokens already match keeps digital with no call."""
    calls = _install_fake_call(monkeypatch, lambda c: [])
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "Hello", "l": [11, 11, 61, 31]}]
    merged = _merge_llm(digital, ocr)
    assert merged == digital
    assert calls == []  # never reached the LLM


def test_llm_verbose_logs_accepted_text(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The per-region INFO line includes the final accepted token text."""
    _install_fake_call(monkeypatch, lambda c: [{"ref": c["digital"][0]["id"], "t": "file"}])
    digital = [{"t": "ﬁle", "l": [10, 10, 40, 30]}]
    ocr = [{"t": "file", "l": [11, 11, 41, 31]}]
    _merge_words(
        digital,
        ocr,
        file_id="fid",
        page_num=1,
        text_extraction_config=_LLM_CONFIG,
    )
    err = _logged(caplog)
    assert "LLM resolved" in err
    assert "'file'" in err  # the accepted token text is surfaced
    assert "digital=['ﬁle']" in err  # original digital text
    assert "ocr=['file']" in err  # original OCR text


def test_llm_take_digital(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disagreeing region, model picks digital → digital words emitted."""
    _install_fake_call(monkeypatch, lambda c: [{"ref": t["id"]} for t in c["digital"]])
    digital = [{"t": "outof", "l": [10, 10, 90, 30]}]
    ocr = [{"t": "out", "l": [10, 10, 45, 30]}, {"t": "of", "l": [50, 10, 80, 30]}]
    merged = _merge_llm(digital, ocr)
    assert merged == digital


def test_llm_take_ocr_splits_runtogether(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model picks OCR's finer tokenization → both OCR words, their boxes."""
    _install_fake_call(monkeypatch, lambda c: [{"ref": t["id"]} for t in c["ocr"]])
    digital = [{"t": "outof", "l": [10, 10, 90, 30]}]
    ocr = [{"t": "out", "l": [10, 10, 45, 30]}, {"t": "of", "l": [50, 10, 80, 30]}]
    merged = _merge_llm(digital, ocr)
    assert merged == ocr


def test_llm_deligature_combination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Override text (de-ligature) while inheriting the digital token's box."""
    _install_fake_call(monkeypatch, lambda c: [{"ref": c["digital"][0]["id"], "t": "file"}])
    digital = [{"t": "ﬁle", "l": [10, 10, 40, 30]}]  # 'ﬁle'
    ocr = [{"t": "file", "l": [11, 11, 41, 31]}]
    merged = _merge_llm(digital, ocr)
    assert merged == [{"t": "file", "l": [10, 10, 40, 30]}]


def test_llm_ocr_only_kept_without_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """OCR-only regions keep OCR and never reach the LLM."""
    calls = _install_fake_call(monkeypatch, lambda c: [])
    ocr = [{"t": "scanned", "l": [10, 10, 60, 30]}]
    merged = _merge_llm([], ocr)
    assert merged == ocr
    assert calls == []  # OCR-only is resolved without the model


def test_llm_ocr_only_noise_kept_not_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even noise-looking OCR-only text is kept now — the LLM is not consulted."""
    calls = _install_fake_call(monkeypatch, lambda c: [])
    ocr = [{"t": "~^", "l": [10, 10, 60, 30]}]
    merged = _merge_llm([], ocr)
    assert merged == ocr  # kept, not dropped
    assert calls == []


def test_llm_digital_only_and_ocr_only_resolved_without_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-sided regions need no LLM call: digital-only dropped, OCR-only kept."""
    calls = _install_fake_call(monkeypatch, lambda c: [{"ref": t["id"]} for t in c["ocr"]])
    digital = [{"t": "ghost", "l": [200, 200, 260, 220]}]  # no OCR overlap
    ocr = [{"t": "real", "l": [10, 10, 60, 30]}]
    merged = _merge_llm(digital, ocr)
    assert merged == ocr  # ghost dropped, real kept
    assert calls == []  # neither single-sided region reaches the LLM


def test_llm_unknown_ref_falls_back_to_nearest_box(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid ref keeps the override text but borrows the region's box."""
    _install_fake_call(monkeypatch, lambda c: [{"ref": "does-not-exist", "t": "file"}])
    digital = [{"t": "ﬁle", "l": [10, 10, 40, 30]}]
    ocr = [{"t": "file", "l": [11, 11, 41, 31]}]
    merged = _merge_llm(digital, ocr)
    # Box borrowed from the first reading-order token in the region (digital).
    assert merged == [{"t": "file", "l": [10, 10, 40, 30]}]


def test_llm_malformed_response_falls_back_to_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unparseable reply → the page is resolved by the heuristic instead."""
    _install_fake_call(monkeypatch, lambda c: [], raw="not json {{{")
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "Greetings", "l": [11, 11, 61, 31]}]  # far → heuristic takes OCR
    merged = _merge_llm(digital, ocr)
    assert merged == ocr


def test_llm_missing_region_id_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply that omits a sent region is structurally invalid → heuristic."""
    _install_fake_call(monkeypatch, lambda c: [], raw='{"nonexistent": []}')
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "Greetings", "l": [11, 11, 61, 31]}]
    merged = _merge_llm(digital, ocr)
    assert merged == ocr  # heuristic: disagreement → OCR


def test_llm_failed_merge_logs_raw_output(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed merge logs the raw model output so truncation is diagnosable."""
    _install_fake_call(monkeypatch, lambda c: [], raw="not json {{{")
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "Greetings", "l": [11, 11, 61, 31]}]
    _merge_words(
        digital,
        ocr,
        file_id="fid",
        page_num=1,
        text_extraction_config=_LLM_CONFIG,
    )
    err = _logged(caplog)
    assert "LLM merge failed" in err
    assert "raw output was:" in err
    assert "not json {{{" in err  # the verbatim reply is surfaced


# ---------------------------------------------------------------------------
# Region batching (one page → several bounded LLM calls)
# ---------------------------------------------------------------------------


def _row_regions(n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build ``n`` non-overlapping mixed regions, one per row, that disagree.

    Each row's digital and OCR text differ enough that the heuristic takes OCR
    while the LLM is free to pick either — so a row's output reveals which
    resolver handled it. Rows are 40px apart (height 30) so they never region
    together, and increasing-y means reading order matches row order.
    """
    digital: list[dict[str, Any]] = []
    ocr: list[dict[str, Any]] = []
    for i in range(n):
        y = i * 40
        digital.append({"t": f"dig{i}", "l": [10, y, 60, y + 30]})
        ocr.append({"t": f"ocr{i}", "l": [11, y + 1, 61, y + 31]})
    return digital, ocr


def test_llm_batches_large_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """More than MERGE_BATCH_SIZE regions go out in several calls, all resolved."""
    calls = _install_fake_call(monkeypatch, lambda c: [{"ref": t["id"]} for t in c["ocr"]])
    n = MERGE_BATCH_SIZE + 5
    digital, ocr = _row_regions(n)
    merged = _merge_llm(digital, ocr)
    assert merged == ocr  # model kept OCR for every region
    assert len(calls) == 2  # ceil(n / MERGE_BATCH_SIZE)
    # Every region was sent exactly once, none dropped or duplicated.
    assert sum(len(c["regions"]) for c in calls) == n


def test_llm_failed_batch_falls_back_without_double_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed batch falls back to the heuristic for *its* regions only.

    Successful batches keep their LLM result and no region is emitted twice —
    the regression this guards against is the heuristic re-resolving regions
    the LLM already handled.
    """
    calls: list[dict[str, Any]] = []

    def _fake_call(
        config: Any,
        *,
        system_prompt: Any,
        user_content: list[dict[str, Any]],
        cache: bool = False,
    ) -> str:
        payload = json.loads(user_content[0]["text"])
        calls.append(payload)
        if len(calls) == 2:  # second batch's reply is truncated garbage
            return "not json {{{"
        # First batch: model picks digital (heuristic would pick OCR).
        return json.dumps(
            {c["id"]: [{"ref": t["id"]} for t in c["digital"]] for c in payload["regions"]}
        )

    monkeypatch.setattr("dgml_core.hybrid.call", _fake_call)

    n = MERGE_BATCH_SIZE + 3
    digital, ocr = _row_regions(n)
    merged = _merge_llm(digital, ocr)

    assert len(calls) == 2
    # Batch 1 (rows 0..MERGE_BATCH_SIZE-1): LLM → digital.
    # Batch 2 (the rest): failed → heuristic → OCR.
    expected = digital[:MERGE_BATCH_SIZE] + ocr[MERGE_BATCH_SIZE:]
    assert merged == expected
    assert len(merged) == n  # one token per region — nothing written twice
