"""Tests for hybrid text-mode (digital + OCR merged by bounding-box overlap)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from dgml_core.errors import OcrFailed
from dgml_core.files import FileStore
from dgml_core.hybrid import (
    LEVENSHTEIN_THRESHOLD,
    MAX_CID_WORDS_PER_PAGE,
    MERGE_BATCH_SIZE,
    _iou,
    _levenshtein_distance,
    _merge_words,
    extract_text_hybrid,
)
from dgml_core.ocr import OcrConfig, OcrProvider, OcrProviderName
from dgml_core.storage import Workspace
from dgml_core.text_extraction import PAGE_TEXT_GLOB, TextMode

from .conftest import make_fake_png, write_ocr_config

# ---------------------------------------------------------------------------
# IoU sanity
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Merge rules
# ---------------------------------------------------------------------------


def test_merge_default_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    """Without ``verbose=True`` (the default), the merge emits nothing on
    stderr — not even the per-page summary."""
    digital = [
        {"t": "stamp", "l": [200, 200, 260, 220]},  # digital-only, would warn
        {"t": "Hello", "l": [10, 10, 60, 30]},  # very different from OCR, would warn
    ]
    ocr = [{"t": "Greetings", "l": [10, 10, 60, 30]}]
    _merge_words(digital, ocr, file_id="fid", page_num=1)
    assert capsys.readouterr().err == ""


def test_merge_ocr_only_word_kept_and_logged(capsys: pytest.CaptureFixture[str]) -> None:
    """OCR-only words are always kept, and (under verbose) an info line tells
    the operator which tokens OCR contributed that digital missed."""
    digital: list[dict[str, Any]] = []
    ocr = [{"t": "scanned", "l": [10, 10, 60, 30]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    assert merged == ocr
    err = capsys.readouterr().err
    assert "warning" not in err  # OCR-only is not a warning condition
    assert "info" in err
    assert "'scanned'" in err
    assert "no matching digital text" in err
    assert "digital_words=0 ocr_words=1 merged=1" in err


def test_merge_same_text_overlap_takes_digital(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Overlap + identical text → take digital (exact character codes)."""
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "Hello", "l": [11, 11, 61, 31]}]  # nearly identical bbox
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    assert merged == digital  # digital wins on overlap+similar
    err = capsys.readouterr().err
    assert "warning" not in err
    assert "digital_words=1 ocr_words=1 merged=1" in err


def test_merge_normalizes_dashes_before_levenshtein(
    capsys: pytest.CaptureFixture[str],
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
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    # Digital wins (normalized distance == 0). The original digital text
    # is preserved so downstream consumers see what the PDF actually held.
    assert merged == digital
    err = capsys.readouterr().err
    assert "warning" not in err


def test_merge_normalizes_all_dash_variants(
    capsys: pytest.CaptureFixture[str],
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
        merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
        assert merged == digital, f"digital should win for U+{cp:04X}"

    # All cases should have been silent on the warning channel.
    assert "warning" not in capsys.readouterr().err


def test_merge_conflicting_text_above_threshold_takes_ocr_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Overlap + Levenshtein > threshold → take OCR with a warning."""
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "Greetings", "l": [10, 10, 60, 30]}]  # distance 7
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    assert merged == ocr  # OCR wins on overlap+different
    err = capsys.readouterr().err
    assert "'Hello'" in err
    assert "'Greetings'" in err
    assert "using OCR" in err
    assert "levenshtein=" in err


def test_merge_digital_only_word_is_dropped_with_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A digital word with no overlapping OCR word is assumed invisible-to-
    human-eye and **dropped** (not kept) — but the drop is logged."""
    digital = [{"t": "stamp", "l": [200, 200, 260, 220]}]
    ocr = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    # The OCR word is kept; the digital-only "stamp" is dropped.
    assert merged == [{"t": "Hello", "l": [10, 10, 60, 30]}]
    err = capsys.readouterr().err
    assert "'stamp'" in err
    assert "not detected by OCR" in err
    assert "dropping" in err


def test_merge_combination_warns_for_each_problem(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """All four cases mixed on one page.

    - OCR ``Hello`` ↔ digital ``Hello`` at same position → digital wins (silent).
    - OCR ``Greetings`` ↔ digital ``world`` at same position, very different →
      OCR wins, one warning.
    - OCR ``page`` with no digital overlap → OCR kept silently.
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
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    assert merged == [
        {"t": "Hello", "l": [10, 10, 60, 30]},  # digital, taken on similarity
        {"t": "Greetings", "l": [70, 10, 120, 30]},  # OCR, conflict winner
        {"t": "page", "l": [200, 200, 240, 220]},  # OCR-only
    ]
    err = capsys.readouterr().err
    assert err.count("using OCR") == 1  # only the world/Greetings pair
    assert err.count("dropping") == 1  # only "stamp"
    assert err.count("no matching digital text") == 1  # only "page" (OCR-only)


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
    capsys: pytest.CaptureFixture[str],
) -> None:
    """OCR splits a span digital keeps whole; texts agree → keep digital
    (PDF font is more reliable than OCR's tokenization)."""
    digital = [{"t": "Analysis)Inadequate", "l": [814, 2268, 1229, 2314]}]
    ocr = [
        {"t": "Analysis)", "l": [813, 2264, 985, 2315]},
        {"t": "Inadequate", "l": [994, 2263, 1230, 2316]},
    ]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=16, verbose=True)
    assert merged == digital  # texts agree → digital wins regardless of token count
    err = capsys.readouterr().err
    assert "tokenization mismatch" in err
    assert "text agrees" in err
    assert "keeping digital's 1 tokens" in err


def test_merge_digital_split_agreeing_keeps_digital_tokens(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Digital splits a span OCR keeps whole; texts agree → keep digital."""
    digital = [
        {"t": "Analysis)", "l": [813, 2264, 985, 2315]},
        {"t": "Inadequate", "l": [994, 2263, 1230, 2316]},
    ]
    ocr = [{"t": "Analysis)Inadequate", "l": [814, 2268, 1229, 2314]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=16, verbose=True)
    assert merged == digital  # texts agree → digital wins
    err = capsys.readouterr().err
    assert "keeping digital's 2 tokens" in err


def test_merge_split_tie_goes_to_digital(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Equal token counts in a mixed region with agreeing text → digital."""
    # Identical boxes force a single 2-vs-2 region.
    box = [0, 0, 80, 10]
    digital = [{"t": "AB", "l": box}, {"t": "CD", "l": box}]
    ocr = [{"t": "AC", "l": box}, {"t": "BD", "l": box}]
    # Concats "ABCD" vs "ACBD" — distance 2, within threshold → "agree".
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    assert merged == digital  # 2 == 2 tie → digital
    err = capsys.readouterr().err
    assert "keeping digital's 2 tokens" in err


def test_merge_split_disagreeing_takes_ocr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mixed region whose concatenations differ beyond threshold → OCR, even
    though digital here has fewer tokens."""
    digital = [{"t": "ABCDEFGH", "l": [0, 0, 80, 10]}]
    ocr = [
        {"t": "XYZ", "l": [0, 0, 40, 10]},
        {"t": "WVUT", "l": [40, 0, 80, 10]},
    ]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    assert merged == ocr
    err = capsys.readouterr().err
    assert "text differs" in err
    assert "keeping OCR's 2 tokens" in err


def test_merge_dollar_blanks_vs_ocr_dollar_takes_ocr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Real 1:1-via-coverage case: digital captured fill-in underscores OCR
    didn't see. Boxes overlap by containment; texts differ → OCR's '$'."""
    digital = [{"t": '$_____."', "l": [1454, 123, 1635, 169]}]
    ocr = [{"t": "$", "l": [1455, 120, 1481, 170]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    assert merged == ocr
    err = capsys.readouterr().err
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
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Distance == threshold counts as similar — digital wins."""
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "He11o", "l": [10, 10, 60, 30]}]  # distance 2
    assert _levenshtein_distance("Hello", "He11o") == LEVENSHTEIN_THRESHOLD

    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    assert merged == digital
    assert "warning" not in capsys.readouterr().err


def test_merge_levenshtein_just_over_threshold_takes_ocr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Distance == threshold + 1 → OCR wins with a warning."""
    digital = [{"t": "Hello", "l": [10, 10, 60, 30]}]
    ocr = [{"t": "He11x", "l": [10, 10, 60, 30]}]  # distance 3
    assert _levenshtein_distance("Hello", "He11x") == LEVENSHTEIN_THRESHOLD + 1

    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    assert merged == ocr
    err = capsys.readouterr().err
    assert "using OCR" in err
    assert f"levenshtein={LEVENSHTEIN_THRESHOLD + 1}" in err


# ---------------------------------------------------------------------------
# CID guard: too many "(cid:" words in digital text → OCR-only for page
# ---------------------------------------------------------------------------


def test_merge_cid_guard_falls_back_to_ocr_only(
    capsys: pytest.CaptureFixture[str],
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
    merged = _merge_words(digital, ocr, file_id="fid", page_num=7, verbose=True)
    assert merged == ocr
    err = capsys.readouterr().err
    assert "unicode error" in err
    assert "page=7" in err
    assert f"{MAX_CID_WORDS_PER_PAGE + 1} words containing '(cid:'" in err
    assert "cid_guard=true" in err


def test_merge_cid_guard_below_threshold_does_not_trigger(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """At-or-below ``MAX_CID_WORDS_PER_PAGE`` is tolerated — the normal merge
    runs and individual cid words are subject to the regular rules."""
    digital = [
        {"t": f"(cid:{i})", "l": [10 * i, 10, 10 * i + 8, 30]}
        for i in range(MAX_CID_WORDS_PER_PAGE)
    ]
    ocr = [{"t": "scanned", "l": [500, 500, 560, 520]}]
    merged = _merge_words(digital, ocr, file_id="fid", page_num=1, verbose=True)
    # The cid tokens are digital-only (no OCR overlap) so they all drop.
    # The OCR word stays.
    assert merged == ocr
    err = capsys.readouterr().err
    assert "unicode error" not in err


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
        name = OcrProviderName.AZURE
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
            return OcrConfig(provider=cls.name)

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

    from dgml_core.ocr import _PROVIDERS

    monkeypatch.setitem(_PROVIDERS, OcrProviderName.AZURE, FakeProvider)


def _seed_page_images(pages_dir: Path, n: int, w: int = 612, h: int = 792) -> None:
    pages_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (pages_dir / f"page_{i}.png").write_bytes(make_fake_png(w, h, f"p{i}".encode()))


def test_extract_text_hybrid_merges_per_page_and_writes_output(
    text_pdf: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    cfg = OcrConfig(provider=OcrProviderName.AZURE, endpoint="https://x/")
    result = extract_text_hybrid(
        text_pdf, out_dir, file_id="fid", page_images_dir=pages_dir, config=cfg, verbose=True
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
    err = capsys.readouterr().err
    for token in ("Hello", "World", "Second", "Page", "Text"):
        assert f"'{token}'" in err
    assert "dropping" in err


def test_extract_text_hybrid_continues_when_digital_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    cfg = OcrConfig(provider=OcrProviderName.AZURE, endpoint="https://x/")
    result = extract_text_hybrid(
        broken_pdf,
        out_dir,
        file_id="fid",
        page_images_dir=pages_dir,
        config=cfg,
        verbose=True,
    )

    assert result.pages_written == 1
    page1 = json.loads((out_dir / "page_1.json").read_text())
    assert [w["t"] for w in page1["words"]] == ["scanned"]
    assert "digital extraction failed" in capsys.readouterr().err


def test_extract_text_hybrid_propagates_ocr_failure(
    text_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages_dir = tmp_path / "page_images"
    _seed_page_images(pages_dir, n=2)
    _install_fake_provider(monkeypatch, fail_on_page=1)

    cfg = OcrConfig(provider=OcrProviderName.AZURE, endpoint="https://x/")
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

    def fake_render(pdf_path: Path, output_dir: Path) -> int:
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

    text_dir = workspace.file_text_dir(result.record.id)
    assert len(list(text_dir.glob(PAGE_TEXT_GLOB))) == 2


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
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The per-region verbose line includes the final accepted token text."""
    _install_fake_call(monkeypatch, lambda c: [{"ref": c["digital"][0]["id"], "t": "file"}])
    digital = [{"t": "ﬁle", "l": [10, 10, 40, 30]}]
    ocr = [{"t": "file", "l": [11, 11, 41, 31]}]
    _merge_words(
        digital,
        ocr,
        file_id="fid",
        page_num=1,
        text_extraction_config=_LLM_CONFIG,
        verbose=True,
    )
    err = capsys.readouterr().err
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
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
        verbose=True,
    )
    err = capsys.readouterr().err
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
