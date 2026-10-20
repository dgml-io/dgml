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

"""Tests for the page-render abstraction: the ghostscript default, the
pypdfium2 provider, the ``[pdf]`` config loader, and the render cache.

Ghostscript is faked by monkeypatching ``subprocess.run`` (and the binary
probe in ``pages_ghostscript``) so these tests run without the system binary
and can assert exactly how many times the renderer is invoked — the whole
point of the cache. The pypdfium2 tests run against the real PDFium (a dev
dependency, no system binary needed).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from dgml_core import pages, pages_ghostscript
from dgml_core.errors import PageRenderFailed, PdfConfigInvalid
from dgml_core.pages import (
    DEFAULT_PAGE_IMAGE_SPEC,
    PAGE_CACHE_ENV,
    PAGE_MAX_EDGE,
    PAGE_MIN_EDGE,
    EngineName,
    PageImageSpec,
    PdfConfig,
    list_page_images,
    load_pdf_config,
    make_renderer,
    page_number_from_image_name,
    render_pages,
)
from dgml_core.storage import Workspace

from .conftest import (
    _write_pdf_with_bad_count,
    _write_pdf_with_cropbox,
    _write_pdf_with_mediabox,
    _write_text_pdf,
    write_config,
)


def _png_size(path: Path) -> tuple[int, int]:
    """Pixel dimensions straight from the PNG IHDR — no image library needed."""
    import struct

    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n"), f"not a PNG: {path}"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _image(path: Path) -> tuple[str, tuple[int, int]]:
    """``(format, size)`` of a stored page image, as Pillow reads it."""
    from PIL import Image

    with Image.open(path) as im:
        return str(im.format), im.size


def _stored(out: Path, page: int = 1) -> tuple[str, tuple[int, int]]:
    """``(format, size)`` of page ``page`` in ``out``, whichever encoding it
    was stored with; the extension must agree with the format."""
    (path,) = [p for p in list_page_images(out) if page_number_from_image_name(p.name) == page]
    fmt, size = _image(path)
    assert {"JPEG": ".jpg", "PNG": ".png"}[fmt] == path.suffix, path
    return fmt, size


def _stems(out: Path) -> list[str]:
    return [p.stem for p in list_page_images(out)]


def _engine_png(width: int = 1275, height: int = 1650) -> bytes:
    """A real (white) PNG, as an engine would write — the store step decodes it."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def _fake_gs_factory(
    n_pages: int, counter: list[int], size: tuple[int, int] = (1275, 1650)
) -> object:
    """Return a fake ``subprocess.run`` that "renders" ``n_pages`` PNGs.

    Parses the ``-sOutputFile=.../page_%d.png`` template out of the command
    and writes one ``size`` PNG per page, mirroring ghostscript's own
    numbering, then records the invocation in ``counter``.
    """
    png = _engine_png(*size)

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        counter.append(1)
        template = next(a.split("=", 1)[1] for a in cmd if a.startswith("-sOutputFile="))
        for i in range(1, n_pages + 1):
            Path(template.replace("%d", str(i))).write_bytes(png)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_run


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake bytes for hashing")
    return p


def test_no_cache_env_always_renders(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(2, calls))

    assert render_pages(pdf, tmp_path / "out1") == 2
    assert render_pages(pdf, tmp_path / "out2") == 2
    # No cache configured: ghostscript runs for every call.
    assert len(calls) == 2


def test_cache_miss_populates_then_hit_skips_ghostscript(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(3, calls))

    out1 = tmp_path / "out1"
    assert render_pages(pdf, out1) == 3
    assert len(calls) == 1  # miss -> one render
    assert _stems(out1) == ["page_1", "page_2", "page_3"]

    # A later render of identical bytes into a fresh dir is served from cache;
    # ghostscript must not be invoked again — assert by making it fail loudly.
    monkeypatch.setattr(
        pages_ghostscript,
        "ghostscript_path",
        lambda: (_ for _ in ()).throw(AssertionError("gs called")),
    )
    out2 = tmp_path / "out2"
    assert render_pages(pdf, out2) == 3
    assert len(calls) == 1  # still one — the hit copied from cache
    assert _stems(out2) == ["page_1", "page_2", "page_3"]


def test_cache_key_differs_by_content(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(1, calls))

    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-1.4 completely different bytes")

    render_pages(pdf, tmp_path / "a")
    render_pages(other, tmp_path / "b")
    # Distinct content -> distinct cache keys -> two renders, no false hit.
    assert len(calls) == 2


def test_partial_cache_entry_is_treated_as_miss(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(2, calls))

    # A cache entry with images but no `.complete` marker (e.g. an interrupted
    # writer) must not be trusted — render_pages should re-render.
    entry = cache / pages._pdf_cache_key(pdf, pages.DEFAULT_DPI)
    entry.mkdir(parents=True)
    (entry / "page_1.jpg").write_bytes(b"stale")

    assert render_pages(pdf, tmp_path / "out") == 2
    assert len(calls) == 1


def test_dpi_reaches_ghostscript(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    seen: list[list[str]] = []

    def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(cmd)
        template = next(a.split("=", 1)[1] for a in cmd if a.startswith("-sOutputFile="))
        Path(template.replace("%d", "1")).write_bytes(_engine_png())
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", capture)

    render_pages(pdf, tmp_path / "out", dpi=150)
    assert "-r150" in seen[0]


def test_cache_entries_do_not_collide_across_dpi(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same bytes at two resolutions are different renders. Serving the 300-dpi
    # entry for a 150-dpi request would hand back page images that disagree
    # with the page_text/ boxes written alongside them.
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(2, calls))

    assert render_pages(pdf, tmp_path / "a", dpi=300) == 2
    assert render_pages(pdf, tmp_path / "b", dpi=150) == 2
    assert len(calls) == 2, "the second dpi must miss, not reuse the first render"
    assert pages._pdf_cache_key(pdf, 300) != pages._pdf_cache_key(pdf, 150)

    # ...and each is still cached in its own right.
    assert render_pages(pdf, tmp_path / "c", dpi=150) == 2
    assert len(calls) == 2


def test_cache_entries_do_not_collide_across_renderers(pdf: Path) -> None:
    # Same bytes at the same dpi through different backends are different
    # renders — the pixels differ subtly (anti-aliasing, ±1px rounding).
    key_gs = pages._pdf_cache_key(pdf, 300, EngineName.GHOSTSCRIPT)
    key_pdfium = pages._pdf_cache_key(pdf, 300, EngineName.PYPDFIUM2)
    assert key_gs != key_pdfium
    # The default parameter is the ghostscript default.
    assert pages._pdf_cache_key(pdf, 300) == key_gs


def test_cache_entries_do_not_collide_across_image_specs(pdf: Path) -> None:
    # Different edge bounds are different stored pixels for the same bytes,
    # dpi and renderer.
    keys = {
        pages._pdf_cache_key(pdf, 300, EngineName.GHOSTSCRIPT, spec)
        for spec in (
            DEFAULT_PAGE_IMAGE_SPEC,
            PageImageSpec(max_edge=2000),
            PageImageSpec(min_edge=None),
        )
    }
    assert len(keys) == 3


# ---------------------------------------------------------------------------
# The pypdfium2 renderer (real PDFium — a dev dependency, no system binary)
# ---------------------------------------------------------------------------


def test_pypdfium2_renders_all_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    pdf_path = tmp_path / "doc.pdf"
    _write_text_pdf(pdf_path, pages_text=["Hello World", "Second Page"])
    # Ghostscript must play no part — fail loudly if its probe runs.
    monkeypatch.setattr(
        pages_ghostscript,
        "ghostscript_path",
        lambda: (_ for _ in ()).throw(AssertionError("gs called")),
    )

    out = tmp_path / "out"
    config = PdfConfig(provider=EngineName.PYPDFIUM2)
    assert render_pages(pdf_path, out, dpi=72, config=config) == 2
    assert _stems(out) == ["page_1", "page_2"]
    for page in (1, 2):
        _stored(out, page)


def test_pypdfium2_dpi_scales_pixel_dimensions(tmp_path: Path) -> None:
    # US Letter is 612x792 pt; at 72 dpi that's 612x792 px, at 144 dpi doubled.

    pdf_path = tmp_path / "doc.pdf"
    _write_text_pdf(pdf_path, pages_text=["Hello"])
    config = PdfConfig(provider=EngineName.PYPDFIUM2)

    out72 = tmp_path / "out72"
    unbounded = PageImageSpec(min_edge=None, max_edge=None)
    render_pages(pdf_path, out72, dpi=72, config=config, image_spec=unbounded)
    assert _stored(out72)[1] == (612, 792)

    out144 = tmp_path / "out144"
    render_pages(pdf_path, out144, dpi=144, config=config)
    assert _stored(out144)[1] == (1224, 1584)


def test_pypdfium2_unreadable_pdf_raises_page_render_failed(tmp_path: Path) -> None:
    from dgml_core.errors import PageRenderFailed

    bogus = tmp_path / "bogus.pdf"
    bogus.write_bytes(b"not a pdf at all")
    config = PdfConfig(provider=EngineName.PYPDFIUM2)
    with pytest.raises(PageRenderFailed, match="pypdfium2"):
        render_pages(bogus, tmp_path / "out", config=config)


def test_pypdfium2_render_is_served_from_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    pdf_path = tmp_path / "doc.pdf"
    _write_text_pdf(pdf_path, pages_text=["Hello"])
    config = PdfConfig(provider=EngineName.PYPDFIUM2)

    assert render_pages(pdf_path, tmp_path / "a", config=config) == 1
    # A hit constructs no renderer at all — prove it by making construction fail.
    monkeypatch.setattr(
        pages,
        "make_renderer",
        lambda cfg: (_ for _ in ()).throw(AssertionError("renderer constructed on a cache hit")),
    )
    assert render_pages(pdf_path, tmp_path / "b", config=config) == 1


# ---------------------------------------------------------------------------
# load_pdf_config — reading the [pdf] section of config.toml
# ---------------------------------------------------------------------------


def test_pdf_config_defaults_to_ghostscript(workspace: Workspace) -> None:
    assert load_pdf_config(workspace) == PdfConfig(provider=EngineName.GHOSTSCRIPT)


def test_pdf_config_empty_section_is_default(workspace: Workspace) -> None:
    write_config(workspace, {"pdf": {}})
    assert load_pdf_config(workspace).provider is EngineName.GHOSTSCRIPT


def test_pdf_config_selects_pypdfium2(workspace: Workspace) -> None:
    write_config(workspace, {"pdf": {"provider": "pypdfium2"}})
    assert load_pdf_config(workspace).provider is EngineName.PYPDFIUM2


def test_pdf_config_rejects_unknown_provider(workspace: Workspace) -> None:
    write_config(workspace, {"pdf": {"provider": "poppler"}})
    with pytest.raises(PdfConfigInvalid, match="poppler"):
        load_pdf_config(workspace)


def test_pdf_config_rejects_unknown_fields(workspace: Workspace) -> None:
    write_config(workspace, {"pdf": {"provider": "ghostscript", "dpi": 300}})
    with pytest.raises(PdfConfigInvalid, match="dpi"):
        load_pdf_config(workspace)


# ---------------------------------------------------------------------------
# Geometry: an engine's PNG must be exactly MediaBox x dpi, and the stored
# image exactly PageImageSpec.fit of that — the size page_text/ declares.
#
# dg:origin boxes are documented as image pixels "matching page_images/
# page_N.jpg", and page_text/ derives its width/height from the MediaBox
# scaled by dpi and refit by the same spec. A renderer whose output disagrees
# mis-grounds every box on the page, silently — dgml check does not compare
# the two.
#
# These assert 300 dpi (the default) deliberately: 72 and 144 are the two dpi
# values whose scale factors (1.0, 2.0) are exact in binary floating point, so
# a ceil-vs-round error cannot manifest at either.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dpi", [72, 150, 300, 600])
@pytest.mark.parametrize("provider", [EngineName.GHOSTSCRIPT, EngineName.PYPDFIUM2])
def test_render_dimensions_are_exactly_mediabox_times_dpi(
    tmp_path: Path, dpi: int, provider: EngineName, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    from .conftest import PAGE_HEIGHT_PTS, PAGE_WIDTH_PTS

    pdf_path = tmp_path / "letter.pdf"
    _write_text_pdf(pdf_path, pages_text=["Hello"])
    expected = (round(PAGE_WIDTH_PTS * dpi / 72), round(PAGE_HEIGHT_PTS * dpi / 72))

    raw = tmp_path / f"raw{dpi}"
    raw.mkdir()
    unbounded = PageImageSpec(min_edge=None, max_edge=None)
    make_renderer(PdfConfig(provider=provider)).render(pdf_path, raw, dpi=dpi, image_spec=unbounded)
    assert _png_size(raw / "page_1.png") == expected

    out = tmp_path / f"out{dpi}"
    render_pages(pdf_path, out, dpi=dpi, config=PdfConfig(provider=provider))
    assert _stored(out)[1] == DEFAULT_PAGE_IMAGE_SPEC.fit(*expected)


@pytest.mark.parametrize("provider", [EngineName.GHOSTSCRIPT, EngineName.PYPDFIUM2])
def test_render_uses_mediabox_not_cropbox(
    tmp_path: Path, provider: EngineName, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CropBox trimming the MediaBox must not change the raster size.

    PDFium sizes its canvas from the CropBox by default; ghostscript uses the
    MediaBox, and so does page_text/. Left uncorrected, a trimmed PDF rendered
    at a different scale puts every word box off its own image.
    """
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    pdf_path = tmp_path / "cropped.pdf"
    _write_pdf_with_cropbox(pdf_path, crop=(36, 36, 576, 756))
    out = tmp_path / "out"
    render_pages(pdf_path, out, dpi=300, config=PdfConfig(provider=provider))
    # MediaBox is US Letter, so the raster is the full 2550x3300 regardless.
    assert _stored(out)[1] == (2550, 3300)


@pytest.mark.parametrize("provider", [EngineName.GHOSTSCRIPT, EngineName.PYPDFIUM2])
def test_render_ignores_overstated_page_count(
    tmp_path: Path, provider: EngineName, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A catalog /Count larger than the real page tree must not lose pages.

    pdf_page_count deliberately walks the page tree rather than trusting
    /Count. A renderer that iterates the catalog count instead runs off the end
    and, because the staged write is discarded on error, loses every page.
    """
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    pdf_path = tmp_path / "badcount.pdf"
    _write_pdf_with_bad_count(pdf_path, real_pages=2, declared=5)
    out = tmp_path / "out"
    assert render_pages(pdf_path, out, dpi=72, config=PdfConfig(provider=provider)) == 2
    assert _stems(out) == ["page_1", "page_2"]


@pytest.mark.parametrize("provider", [EngineName.GHOSTSCRIPT, EngineName.PYPDFIUM2])
def test_render_survives_degenerate_mediabox(
    tmp_path: Path, provider: EngineName, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-area MediaBox must not be installed as the CropBox.

    The MediaBox override that fixes CropBox handling would otherwise hand the
    backend a zero-area box and make an already-odd page unrenderable. Either
    outcome is acceptable here — a rendered page or a clean PageRenderFailed —
    but not a crash or a zero-byte PNG.
    """
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    pdf_path = tmp_path / "degenerate.pdf"
    _write_pdf_with_mediabox(pdf_path, media=(0, 0, 0, 0))
    out = tmp_path / "out"
    try:
        count = render_pages(pdf_path, out, dpi=72, config=PdfConfig(provider=provider))
    except PageRenderFailed:
        return
    for image in list_page_images(out):
        assert image.stat().st_size > 0, "wrote an empty page image"
    assert count >= 0


# ---------------------------------------------------------------------------
# PageImageSpec — the stored page image's edge bounds and encoding
# ---------------------------------------------------------------------------


def test_fit_keeps_a_page_inside_the_bounds_exactly() -> None:
    # Letter and A4 at 300 dpi are both stored as rendered.
    assert DEFAULT_PAGE_IMAGE_SPEC.fit(2550, 3300) == (2550, 3300)
    assert DEFAULT_PAGE_IMAGE_SPEC.fit(2480, 3508) == (2480, 3508)
    assert DEFAULT_PAGE_IMAGE_SPEC.fit(3508, 2480) == (3508, 2480)


def test_fit_scales_a_large_sheet_down_to_the_max_edge() -> None:
    # The large-format reproduction from #157: a 4959 x 7017 page image.
    assert DEFAULT_PAGE_IMAGE_SPEC.fit(4959, 7017) == (2479, PAGE_MAX_EDGE)
    assert DEFAULT_PAGE_IMAGE_SPEC.fit(7017, 4959) == (PAGE_MAX_EDGE, 2479)


def test_fit_scales_a_small_page_up_to_the_min_edge() -> None:
    # Letter at 72 dpi; a 2 x 3.5 in card at 150 dpi.
    assert DEFAULT_PAGE_IMAGE_SPEC.fit(612, 792) == (773, PAGE_MIN_EDGE)
    assert DEFAULT_PAGE_IMAGE_SPEC.fit(300, 525) == (571, PAGE_MIN_EDGE)


def test_fit_never_collapses_an_edge() -> None:
    assert DEFAULT_PAGE_IMAGE_SPEC.fit(1, 40000) == (1, PAGE_MAX_EDGE)


def test_unbounded_spec_never_resizes() -> None:
    unbounded = PageImageSpec(min_edge=None, max_edge=None)
    assert unbounded.fit(9000, 600) == (9000, 600)
    assert unbounded.fit(10, 10) == (10, 10)


@pytest.mark.parametrize(
    ("min_edge", "max_edge"), [(0, 100), (10, -1), (200, 100)], ids=["zero", "negative", "inverted"]
)
def test_spec_rejects_bad_bounds(min_edge: int, max_edge: int) -> None:
    with pytest.raises(ValueError):
        PageImageSpec(min_edge=min_edge, max_edge=max_edge)


def test_page_number_from_image_name() -> None:
    assert page_number_from_image_name("page_12.jpg") == 12
    assert page_number_from_image_name("page_3.png") == 3
    assert page_number_from_image_name("page_3.jpeg") is None
    assert page_number_from_image_name("page_x.jpg") is None
    assert page_number_from_image_name("thumb_1.jpg") is None


def test_list_page_images_orders_by_page_across_extensions(tmp_path: Path) -> None:
    for name in ("page_10.jpg", "page_2.png", "page_1.jpg", "notes.txt", "page_3.tif"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_page_images(tmp_path)] == [
        "page_1.jpg",
        "page_2.png",
        "page_10.jpg",
    ]


# ---------------------------------------------------------------------------
# render_pages — fitting and encoding the stored images
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", [EngineName.GHOSTSCRIPT, EngineName.PYPDFIUM2])
def test_render_scales_a_large_sheet_to_the_max_edge(
    tmp_path: Path, provider: EngineName, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large-format page is stored with its longest edge at the cap, aspect
    kept — however big the engine's own render was."""
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    pdf_path = tmp_path / "sheet.pdf"
    # A2 portrait: 1190.55 x 1683.78 pt, 4961 x 7016 px at 300 dpi.
    _write_text_pdf(pdf_path, pages_text=["Hello"], page_size=(1190.55, 1683.78))
    out = tmp_path / "out"
    render_pages(pdf_path, out, dpi=300, config=PdfConfig(provider=provider))
    rendered = (round(1190.55 * 300 / 72), round(1683.78 * 300 / 72))
    _, size = _stored(out)
    assert size == DEFAULT_PAGE_IMAGE_SPEC.fit(*rendered)
    assert max(size) == PAGE_MAX_EDGE


def test_render_skips_the_decompression_bomb_guard_for_its_own_output(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine's PNG is trusted input: a sheet past Pillow's bomb limit is
    exactly the page the fit exists to shrink, so it must not be refused."""
    from PIL import Image

    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(1, [], size=(4000, 5000)))
    # 20M pixels is over twice this limit, where Image.open raises outright.
    with monkeypatch.context() as m:
        m.setattr(Image, "MAX_IMAGE_PIXELS", 1_000_000)
        render_pages(pdf, tmp_path / "out")
    assert _stored(tmp_path / "out")[1] == (2806, PAGE_MAX_EDGE)


def test_render_clears_stale_images_of_either_extension(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(1, []))
    out = tmp_path / "out"
    out.mkdir()
    (out / "page_1.png").write_bytes(b"old legacy render")
    (out / "page_7.jpg").write_bytes(b"old longer render")
    assert render_pages(pdf, out) == 1
    assert _stems(out) == ["page_1"]


def test_render_reports_an_unreadable_engine_image(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")

    def broken(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        template = next(a.split("=", 1)[1] for a in cmd if a.startswith("-sOutputFile="))
        Path(template.replace("%d", "1")).write_bytes(b"\x89PNG\r\n\x1a\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", broken)
    with pytest.raises(PageRenderFailed, match="page 1"):
        render_pages(pdf, tmp_path / "out")


# ---------------------------------------------------------------------------
# Format choice — each page is stored as the smaller of PNG and JPEG
# ---------------------------------------------------------------------------


def _noise_png(width: int, height: int) -> bytes:
    """A photo-like (incompressible) PNG, which a JPEG beats by far."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.effect_noise((width, height), 64).convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _fake_gs_writing(monkeypatch: pytest.MonkeyPatch, png: bytes) -> None:
    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        template = next(a.split("=", 1)[1] for a in cmd if a.startswith("-sOutputFile="))
        Path(template.replace("%d", "1")).write_bytes(png)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    monkeypatch.setattr(subprocess, "run", fake_run)


def test_a_flat_page_inside_the_bounds_keeps_the_engine_png(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A text-like page compresses far better losslessly; unresized, the
    engine's own PNG bytes are stored as they are, with no re-encode."""
    engine = _engine_png(1275, 1650)
    _fake_gs_writing(monkeypatch, engine)
    render_pages(pdf, tmp_path / "out")
    assert (tmp_path / "out" / "page_1.png").read_bytes() == engine
    assert _stems(tmp_path / "out") == ["page_1"]


def test_a_photo_like_page_is_stored_as_jpeg(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_gs_writing(monkeypatch, _noise_png(1275, 1650))
    render_pages(pdf, tmp_path / "out")
    assert _stored(tmp_path / "out") == ("JPEG", (1275, 1650))


@pytest.mark.parametrize(("make", "fmt"), [(_engine_png, "PNG"), (_noise_png, "JPEG")])
def test_a_resized_page_is_stored_as_the_smaller_encoding(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch, make: Any, fmt: str
) -> None:
    """A refit page has no engine bytes to reuse, so both encodings are made
    from the resized image and the smaller kept."""
    _fake_gs_writing(monkeypatch, make(600, 400))
    render_pages(pdf, tmp_path / "out")
    assert _stored(tmp_path / "out") == (fmt, (PAGE_MIN_EDGE, 667))


def test_one_file_can_mix_formats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The choice is per page: a text page and a photo page of one PDF land as
    a PNG and a JPEG, and both are listed in page order."""
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    monkeypatch.setattr(pages_ghostscript, "ghostscript_path", lambda: "gs")
    pngs = {1: _engine_png(1275, 1650), 2: _noise_png(1275, 1650)}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        template = next(a.split("=", 1)[1] for a in cmd if a.startswith("-sOutputFile="))
        for n, png in pngs.items():
            Path(template.replace("%d", str(n))).write_bytes(png)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    doc = tmp_path / "doc.pdf"
    doc.write_bytes(b"%PDF-1.4 two pages")
    assert render_pages(doc, tmp_path / "out") == 2
    assert [p.name for p in list_page_images(tmp_path / "out")] == ["page_1.png", "page_2.jpg"]


# ---------------------------------------------------------------------------
# PDFium renders an out-of-bounds page straight at its fitted size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "page_size",
    [(1190.55, 1683.78), (144.0, 216.0)],  # A2 (over the max), a 2 x 3 in label (under the min)
    ids=["large-sheet", "small-label"],
)
def test_pypdfium2_renders_out_of_bounds_pages_at_the_fitted_size(
    tmp_path: Path, page_size: tuple[float, float]
) -> None:
    """The engine's own PNG is already the stored size — the fit is applied by
    choosing the render scale, not by resizing a 300-dpi render afterwards."""
    pdf_path = tmp_path / "doc.pdf"
    _write_text_pdf(pdf_path, pages_text=["Hello"], page_size=page_size)
    raw = tmp_path / "raw"
    raw.mkdir()
    make_renderer(PdfConfig(provider=EngineName.PYPDFIUM2)).render(
        pdf_path, raw, dpi=300, image_spec=DEFAULT_PAGE_IMAGE_SPEC
    )
    nominal = (round(page_size[0] * 300 / 72), round(page_size[1] * 300 / 72))
    assert _png_size(raw / "page_1.png") == DEFAULT_PAGE_IMAGE_SPEC.fit(*nominal)
    assert DEFAULT_PAGE_IMAGE_SPEC.fit(*nominal) != nominal


def test_pypdfium2_fitted_render_keeps_the_far_edge_of_a_long_strip(tmp_path: Path) -> None:
    """A 226 x 20000 pt receipt strip fits to 40 x 3508, where the narrow axis's
    rounding (39.65 -> 40) makes the uniform render scale ~1% too large for the
    long axis. That overshoot is page content, not a float pixel: it must be
    squeezed to the target (as ghostscript's resize does), not cropped off the
    bottom — a 100 pt black band at the foot of the page must still be there."""
    from PIL import Image

    width, height = 226, 20000
    stream = b"0 0 226 100 re f\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
        f"/Contents 4 0 R >>".encode(),
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    pdf_path = tmp_path / "strip.pdf"
    pdf_path.write_bytes(bytes(out))

    raw = tmp_path / "raw"
    raw.mkdir()
    make_renderer(PdfConfig(provider=EngineName.PYPDFIUM2)).render(
        pdf_path, raw, dpi=300, image_spec=DEFAULT_PAGE_IMAGE_SPEC
    )
    with Image.open(raw / "page_1.png") as im:
        gray = im.convert("L")
        assert gray.size == (40, PAGE_MAX_EDGE)
        # The band spans the last 100 * 3508 / 20000 ~= 17.5 rows.
        assert gray.getpixel((20, PAGE_MAX_EDGE - 5)) == 0
        assert gray.getpixel((20, PAGE_MAX_EDGE - 30)) == 255


def test_pypdfium2_pages_need_no_resize_in_the_store_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the render already at the fitted size, render_pages never resamples."""
    from PIL import Image

    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    pdf_path = tmp_path / "sheet.pdf"
    _write_text_pdf(pdf_path, pages_text=["Hello"], page_size=(1190.55, 1683.78))

    def no_resize(self: Image.Image, *args: object, **kwargs: object) -> Image.Image:
        raise AssertionError("store step resized a PDFium render")

    monkeypatch.setattr(Image.Image, "resize", no_resize)
    render_pages(pdf_path, tmp_path / "out", config=PdfConfig(provider=EngineName.PYPDFIUM2))
    assert max(_stored(tmp_path / "out")[1]) == PAGE_MAX_EDGE


def test_ghostscript_still_renders_at_the_nominal_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ghostscript renders every page at one resolution, so its engine output
    stays nominal and render_pages fits it."""
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    pdf_path = tmp_path / "label.pdf"
    _write_text_pdf(pdf_path, pages_text=["Hello"], page_size=(144.0, 216.0))
    raw = tmp_path / "raw"
    raw.mkdir()
    make_renderer(PdfConfig(provider=EngineName.GHOSTSCRIPT)).render(
        pdf_path, raw, dpi=300, image_spec=DEFAULT_PAGE_IMAGE_SPEC
    )
    assert _png_size(raw / "page_1.png") == (600, 900)
    out = tmp_path / "out"
    render_pages(pdf_path, out, config=PdfConfig(provider=EngineName.GHOSTSCRIPT))
    assert _stored(out)[1] == (667, PAGE_MIN_EDGE)
