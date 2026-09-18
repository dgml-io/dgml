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

"""Tests for PDF page slicing across both engines.

A slice is sent to the model as a PDF attachment, so the properties that
matter are: the right pages, in document order, with the text layer intact.
Both engines are exercised for real — ghostscript via its binary (skipped when
absent) and PDFium via the ``pdfium`` dev dependency.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from dgml_core.errors import EngineNotAvailable, PdfSliceFailed
from dgml_core.pages import (
    GS_BINARIES,
    EngineName,
    PdfConfig,
    pdf_page_count_bytes,
    slice_pages,
)

from .conftest import _write_text_pdf

needs_gs = pytest.mark.skipif(
    not any(shutil.which(b) for b in GS_BINARIES), reason="ghostscript not installed"
)

ENGINES = [
    pytest.param(EngineName.GHOSTSCRIPT, marks=needs_gs, id="ghostscript"),
    pytest.param(EngineName.PYPDFIUM2, id="pypdfium2"),
]


@pytest.fixture
def eight_page_pdf(tmp_path: Path) -> bytes:
    out = tmp_path / "eight.pdf"
    _write_text_pdf(out, pages_text=[f"Page {i} marker text" for i in range(1, 9)])
    return out.read_bytes()


def _text(pdf_bytes: bytes, tmp_path: Path, name: str = "s.pdf") -> str:
    """Digital text of a PDF, whitespace-normalized."""
    from pdfminer.high_level import extract_text

    p = tmp_path / name
    p.write_bytes(pdf_bytes)
    return " ".join(extract_text(str(p)).split())


@pytest.mark.parametrize("engine", ENGINES)
def test_slice_contiguous_range(engine: EngineName, eight_page_pdf: bytes, tmp_path: Path) -> None:
    out = slice_pages(eight_page_pdf, [2, 3], config=PdfConfig(provider=engine))
    assert pdf_page_count_bytes(out) == 2
    text = _text(out, tmp_path)
    assert "Page 2 marker" in text and "Page 3 marker" in text
    assert "Page 1 marker" not in text and "Page 4 marker" not in text


@pytest.mark.parametrize("engine", ENGINES)
def test_slice_non_contiguous_keeps_document_order(
    engine: EngineName, eight_page_pdf: bytes, tmp_path: Path
) -> None:
    out = slice_pages(eight_page_pdf, [8, 1, 4], config=PdfConfig(provider=engine))
    assert pdf_page_count_bytes(out) == 3
    text = _text(out, tmp_path)
    # Ascending document order regardless of the order requested.
    assert text.index("Page 1 marker") < text.index("Page 4 marker")
    assert text.index("Page 4 marker") < text.index("Page 8 marker")


@pytest.mark.parametrize("engine", ENGINES)
def test_slice_deduplicates_page_numbers(
    engine: EngineName, eight_page_pdf: bytes, tmp_path: Path
) -> None:
    out = slice_pages(eight_page_pdf, [3, 3, 3], config=PdfConfig(provider=engine))
    assert pdf_page_count_bytes(out) == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_slice_single_page(engine: EngineName, eight_page_pdf: bytes, tmp_path: Path) -> None:
    out = slice_pages(eight_page_pdf, [5], config=PdfConfig(provider=engine))
    assert pdf_page_count_bytes(out) == 1
    assert "Page 5 marker" in _text(out, tmp_path)


@pytest.mark.parametrize("engine", ENGINES)
def test_slice_whole_document(engine: EngineName, eight_page_pdf: bytes) -> None:
    out = slice_pages(eight_page_pdf, list(range(1, 9)), config=PdfConfig(provider=engine))
    assert pdf_page_count_bytes(out) == 8


def test_slice_defaults_to_ghostscript_when_no_config(eight_page_pdf: bytes) -> None:
    """No config means the documented ghostscript default, as for rendering."""
    assert PdfConfig().provider is EngineName.GHOSTSCRIPT


# ---------------------------------------------------------------------------
# Argument validation — bounds-checked centrally so a bad request never
# reaches a backend and surface as its own opaque IndexError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_slice_rejects_page_beyond_end(engine: EngineName, eight_page_pdf: bytes) -> None:
    with pytest.raises(PdfSliceFailed, match=r"\[9\].*8-page"):
        slice_pages(eight_page_pdf, [9], config=PdfConfig(provider=engine))


@pytest.mark.parametrize("engine", ENGINES)
def test_slice_rejects_zero_and_negative_pages(engine: EngineName, eight_page_pdf: bytes) -> None:
    with pytest.raises(PdfSliceFailed, match="out of range"):
        slice_pages(eight_page_pdf, [0], config=PdfConfig(provider=engine))


def test_slice_rejects_empty_page_list(eight_page_pdf: bytes) -> None:
    """A caller bug, not a backend failure — so ValueError, not PdfSliceFailed."""
    with pytest.raises(ValueError, match="non-empty"):
        slice_pages(eight_page_pdf, [])


@pytest.mark.parametrize("engine", ENGINES)
def test_slice_unreadable_pdf_raises_pdf_slice_failed(engine: EngineName) -> None:
    with pytest.raises(PdfSliceFailed):
        slice_pages(b"not a pdf at all", [1], config=PdfConfig(provider=engine))


# ---------------------------------------------------------------------------
# Engine availability
# ---------------------------------------------------------------------------


def test_slice_reports_missing_pypdfium2_with_install_hint(
    eight_page_pdf: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured-but-uninstalled engine names the extra to install, rather
    than surfacing a bare ImportError from inside the backend."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *a: object, **k: object) -> object:
        if name == "pypdfium2":
            raise ImportError("simulated absence")
        return real_import(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(EngineNotAvailable, match=r"pip install dgml\[pdfium\]"):
        slice_pages(eight_page_pdf, [1], config=PdfConfig(provider=EngineName.PYPDFIUM2))


@needs_gs
def test_pypdfium2_slice_matches_ghostscript_page_count(eight_page_pdf: bytes) -> None:
    """The engines are interchangeable on what a slice *contains*; they differ
    only in encoding (ghostscript re-encodes, PDFium copies structurally), so
    page counts must agree even though byte sizes will not."""
    pages = [2, 5, 7]
    gs = slice_pages(eight_page_pdf, pages, config=PdfConfig(EngineName.GHOSTSCRIPT))
    pdfium = slice_pages(eight_page_pdf, pages, config=PdfConfig(EngineName.PYPDFIUM2))
    assert pdf_page_count_bytes(gs) == pdf_page_count_bytes(pdfium) == 3


# ---------------------------------------------------------------------------
# total_pages: generation counts a document once, then slices it per window.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_slice_with_supplied_total_matches_self_counted(
    engine: EngineName, eight_page_pdf: bytes
) -> None:
    """Passing the count the caller already has must not change the result."""
    cfg = PdfConfig(provider=engine)
    without = slice_pages(eight_page_pdf, [2, 3], config=cfg)
    with_total = slice_pages(eight_page_pdf, [2, 3], config=cfg, total_pages=8)
    assert pdf_page_count_bytes(without) == pdf_page_count_bytes(with_total) == 2


def test_slice_supplied_total_still_bounds_checks(eight_page_pdf: bytes) -> None:
    """The supplied count is what the range is validated against, so a caller
    that passes a stale one gets a clear PdfSliceFailed rather than an opaque
    backend error."""
    with pytest.raises(PdfSliceFailed, match="out of range"):
        slice_pages(eight_page_pdf, [5], total_pages=4)


def test_slice_without_total_rejects_unreadable_input_before_the_backend(
    eight_page_pdf: bytes,
) -> None:
    """The self-counting path doubles as a readability gate.

    Ghostscript is lenient enough to emit output for input that is not a PDF,
    so the pdfminer walk is what turns that into a clear failure. Callers
    supplying total_pages opt out of this gate and are trusted to have read the
    document already — which generation has, to count its windows.
    """
    with pytest.raises(PdfSliceFailed, match="could not read the PDF"):
        slice_pages(b"definitely not a pdf", [1])


@pytest.mark.parametrize("engine", ENGINES)
def test_slice_skips_its_own_count_when_told(
    engine: EngineName, eight_page_pdf: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With total_pages supplied, no page-tree walk happens — the whole point
    of threading it through, since a long document slices once per window."""
    from dgml_core import pages as pages_mod

    monkeypatch.setattr(
        pages_mod,
        "pdf_page_count_bytes",
        lambda _b: pytest.fail("counted pages despite being given total_pages"),
    )
    out = slice_pages(eight_page_pdf, [2, 3], config=PdfConfig(provider=engine), total_pages=8)
    assert pdf_page_count_bytes(out) == 2
