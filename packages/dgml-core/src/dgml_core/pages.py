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

"""PDF engines: abstract interfaces, config loader, and dispatchers.

DGML needs two things from a PDF library — rasterizing pages to images, and
slicing a page range into a new PDF — and takes both from one **engine**.
Loads the ``[pdf]`` section of ``<workspace>/config.toml`` (via
:func:`load_pdf_config`), dispatches :func:`render_pages` and
:func:`slice_pages` to the configured engine, and owns everything
engine-independent: fitting and encoding the stored page images
(:class:`PageImageSpec`), their filenames, the optional content-addressed render
cache, page counting, and slice bounds-checking.

Engine implementations live in sibling modules so this file stays focused on
the abstraction (mirroring :mod:`dgml_core.ocr`):

- :mod:`dgml_core.pages_ghostscript` — the system ``ghostscript`` binary,
  invoked as a subprocess (the zero-config default; see CLAUDE.md for the
  licensing rationale)
- :mod:`dgml_core.pages_pypdfium2` — PDFium via the ``pypdfium2`` package,
  in-process (``pip install dgml[pdfium]``)

One ``provider`` key selects the engine for *both* capabilities. That is
deliberate rather than a simplification: the reason to switch is usually "do
not require a system binary", which is only satisfied when neither operation
shells out. :class:`PdfConfig` is a dataclass so a future per-capability
override (a ``slicer`` key) would not change any caller's signature.

Adding a new engine
-------------------

1. Add a value to :class:`EngineName`.
2. Create ``pages_<name>.py`` with a :class:`PageRenderer` subclass and a
   :class:`PdfSlicer` subclass. Put the availability check (lazy package
   import, or binary probe) in a shared module-level helper both call, and
   raise :class:`EngineNotAvailable` with an install hint when missing.
3. Add an :class:`EngineSpec` for it in ``_build_registry`` below.
4. If it takes engine-specific config fields, declare them in the spec's
   ``config_fields`` — :func:`_check_no_extra_fields` rejects anything else.

Renderers carry a hard geometry contract; read :meth:`PageRenderer.render`
before implementing one.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from . import layout
from .errors import GhostscriptNotFound, PdfConfigInvalid
from .models_config import ConfigSection

if TYPE_CHECKING:
    from .storage import Workspace

DEFAULT_DPI = 300
# What an engine writes: one lossless PNG per page, at exactly the requested dpi
# or already at its fitted size (see :meth:`PageRenderer.render`), into a scratch
# directory. :func:`render_pages` then fits each to the :class:`PageImageSpec`
# and stores it under the layout's page-image name.
PAGE_FILENAME_TEMPLATE = "page_%d.png"
_ENGINE_PAGE_GLOB = "page_*.png"

# Bounds on the longest edge of a stored page image, in pixels. The max is A4's
# long edge at 300 dpi (Letter's is 3300), so both common page sizes are stored
# as rendered while a large-format sheet (a 4959 x 7017 drawing, say) is scaled
# down to fit: such a page otherwise produces an image too large for vision
# models (Anthropic rejects one over 8000 px, or over 10 MiB of base64) and too
# heavy for every consumer that loads it whole. The min scales a small page (a
# receipt, a label) up so OCR and vision models get legible glyphs. Ghostscript
# reaches either bound by resizing its render; PDFium rasterizes the page
# straight at the fitted size.
PAGE_MAX_EDGE = 3508
PAGE_MIN_EDGE = 1000
# JPEG quality for stored page images. 90 keeps small glyphs clean for OCR.
#
# Each page is stored as whichever of PNG and JPEG is smaller. Neither wins
# everywhere: a ghostscript text page (crisp, unantialiased glyphs on white) is
# ~0.1 MB as a PNG and ~1 MB as a JPEG, while a photo-heavy or scanned page is
# ~15 MB as a PNG and ~2 MB as a JPEG — and it is the heavy pages that break
# consumers. A page's extension says which it is.
PAGE_JPEG_QUALITY = 90

# Optional content-addressed render cache. When ``$DGML_PAGE_CACHE`` names a
# directory, :func:`render_pages` copies its output there keyed by the PDF's
# content hash (plus renderer + dpi) and, on a later call for identical bytes,
# copies back instead of re-rendering. Off by default — rendering is
# unchanged unless the env var is set. Intended for workflows that re-ingest the
# same PDFs into many workspaces (e.g. the clustering sweep's per-cell
# workspaces), where the render is otherwise repeated once per workspace.
PAGE_CACHE_ENV = "DGML_PAGE_CACHE"
_CACHE_COMPLETE_MARKER = ".complete"

GS_TIMEOUT_SECONDS = 600

# On Windows the console executable is named ``gswin64c`` / ``gswin32c``, not
# ``gs`` (the Artifex installer ships no ``gs.exe``). Probe those first, then
# fall back to ``gs`` for MSYS/Cygwin shells that expose the Unix name.
GS_BINARIES: tuple[str, ...] = (
    ("gswin64c", "gswin32c", "gs") if sys.platform == "win32" else ("gs",)
)


class EngineName(StrEnum):
    """Identifier of a PDF engine, as written in workspace config.

    One engine supplies both capabilities DGML needs from a PDF library:
    rasterizing pages to images, and slicing a page range into a new PDF.
    """

    GHOSTSCRIPT = "ghostscript"
    PYPDFIUM2 = "pypdfium2"


# The engine used when a workspace declares no [pdf] config: the system
# ghostscript binary, DGML's original engine. Unlike OCR there is no platform
# split and no warning — ghostscript is the documented default.
DEFAULT_ENGINE = EngineName.GHOSTSCRIPT


@dataclass(frozen=True)
class PdfConfig:
    """Parsed ``pdf`` section of the workspace config.

    By construction (via :func:`load_pdf_config`) this object is well-formed
    for the engine it names. No engine takes extra config fields today; the
    dataclass exists so adding one later — a pixel format, or a ``slicer``
    override selecting a different engine for slicing than for rendering — is
    not a signature change for every caller.
    """

    provider: EngineName = DEFAULT_ENGINE


def load_pdf_config(workspace: Workspace) -> PdfConfig:
    """Read and validate the ``pdf`` section of ``<workspace>/config.toml``.

    When the merged config has no ``pdf`` section — or an empty one — defaults
    to ghostscript (:data:`DEFAULT_ENGINE`), silently: unlike OCR there is a
    built-in default on every platform. Raises :class:`PdfConfigInvalid` when a
    section exists but is malformed.
    """
    # Imported lazily: config.py imports storage.py which must not need us first.
    from .config import load_merged_config

    section = load_merged_config(workspace).get(ConfigSection.PDF)
    if not section:
        # Absent or empty — `provider` is what selects an engine, so a bare
        # `[pdf]` is the same as none at all.
        return PdfConfig()
    if not isinstance(section, dict):
        raise PdfConfigInvalid("'pdf' must be a table")

    provider_str = section.get("provider")
    valid_providers = [e.value for e in EngineName]
    if provider_str not in valid_providers:
        raise PdfConfigInvalid(
            f"'pdf.provider' must be one of {valid_providers} (got {provider_str!r})"
        )
    engine = EngineName(provider_str)
    _check_no_extra_fields(engine, section)
    return PdfConfig(provider=engine)


def _check_no_extra_fields(engine: EngineName, section: dict[str, Any]) -> None:
    """Raise :class:`PdfConfigInvalid` for keys the engine does not accept.

    Catches typos and fields left behind after switching provider. Config
    fields are declared on the engine rather than on either capability class,
    because one ``[pdf]`` section configures both.
    """
    allowed = _ENGINES[engine].config_fields | {"provider"}
    unknown = set(section.keys()) - allowed
    if unknown:
        raise PdfConfigInvalid(
            f"unknown fields in 'pdf' for provider {engine.value!r}: "
            f"{sorted(unknown)}. Allowed: {sorted(allowed)}"
        )


@dataclass(frozen=True)
class PageImageSpec:
    """The longest-edge bounds each rendered page is fit into before it is
    stored.

    This decides the page images' *pixel space*, which ``page_text/`` word
    boxes and every ``dg:origin`` attribute are expressed in, so digital text
    extraction takes the same spec as the render (see :meth:`fit`). ``None``
    for an edge means unbounded on that side.
    """

    min_edge: int | None = PAGE_MIN_EDGE
    max_edge: int | None = PAGE_MAX_EDGE

    def __post_init__(self) -> None:
        for name in ("min_edge", "max_edge"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be a positive integer; got {value!r}")
        if self.min_edge is not None and self.max_edge is not None:
            if self.min_edge > self.max_edge:
                raise ValueError(
                    f"min_edge ({self.min_edge}) must not exceed max_edge ({self.max_edge})"
                )

    def fit(self, width: int, height: int) -> tuple[int, int]:
        """The stored size of a page rendered at ``width`` x ``height`` pixels.

        Scales uniformly so the longest edge lands in ``[min_edge,
        max_edge]``; a page already inside the range keeps its exact size.
        Both the render and digital text extraction call this on the same
        ``round(pts * dpi / 72)`` size, which is what keeps their pixel spaces
        identical.
        """
        longest = max(width, height)
        if longest <= 0:
            return width, height
        if self.max_edge is not None and longest > self.max_edge:
            target = self.max_edge
        elif self.min_edge is not None and longest < self.min_edge:
            target = self.min_edge
        else:
            return width, height
        return max(1, round(width * target / longest)), max(1, round(height * target / longest))


DEFAULT_PAGE_IMAGE_SPEC = PageImageSpec()

_PAGE_IMAGE_NAME_RE = re.compile(
    rf"^page_(\d+)\.(?:{'|'.join(re.escape(e) for e in layout.PAGE_IMAGE_EXTS)})$"
)


def page_number_from_image_name(name: str) -> int | None:
    """Parse ``page_<N>.jpg`` or ``page_<N>.png`` → ``N``; ``None`` otherwise."""
    match = _PAGE_IMAGE_NAME_RE.match(name)
    return int(match.group(1)) if match else None


def list_page_images(directory: Path) -> list[Path]:
    """The page images in ``directory``, in page order (``page_2`` before
    ``page_10``), whichever extension each was stored with."""
    found = [
        (number, path)
        for path in directory.glob("page_*.*")
        if (number := page_number_from_image_name(path.name)) is not None
    ]
    return [path for _, path in sorted(found)]


# ---------------------------------------------------------------------------
# Renderer interface
# ---------------------------------------------------------------------------


class PageRenderer(ABC):
    """Common interface for PDF page-image backends.

    Implementations are constructed from a :class:`PdfConfig` (which is where
    lazy package imports / binary probes live) and implement :meth:`render` for
    a whole PDF. The shared wrapper :func:`render_pages` handles the render
    cache, clearing stale page images, fitting and encoding the stored images
    (:class:`PageImageSpec`), and counting the output — renderers only need to
    write ``page_N.png`` files.
    """

    @abstractmethod
    def __init__(self, config: PdfConfig) -> None:
        """Prepare the backend: lazy-import its package or probe its binary.
        Raise :class:`EngineNotAvailable` (or its ghostscript-specific subclass
        :class:`GhostscriptNotFound`) with an actionable install hint when the
        backend is missing."""

    @abstractmethod
    def render(
        self, pdf_path: Path, output_dir: Path, *, dpi: int, image_spec: PageImageSpec
    ) -> None:
        """Rasterize every page of ``pdf_path`` into ``output_dir``.

        Write one PNG per page named per :data:`PAGE_FILENAME_TEMPLATE`
        (``page_1.png`` …, 1-based, ghostscript's own numbering).
        ``output_dir`` is an empty scratch directory; :func:`render_pages`
        fits and encodes what is written there into the stored images.

        Let ``nominal`` be ``round(pts * dpi / 72)`` pixels on each axis,
        measured from the **MediaBox** and after applying ``/Rotate``;
        ``image_spec.fit(nominal)`` is the coordinate space ``page_text/`` word
        boxes and every ``dg:origin`` attribute are expressed in. The image
        written for a page MUST be exactly one of those two sizes: ``nominal``
        (and :func:`render_pages` resizes it to the fit), or the fit itself,
        rendered directly at that resolution — sharper for a small page, and far
        cheaper for a large sheet, than resizing afterwards. :meth:`PageImageSpec.fit`
        is idempotent, so the store step needs no telling which it got; a backend
        that targets the fit must therefore hit it exactly, since a page one pixel
        short of a bound reads as already fitted. A backend whose natural output
        differs (a different page box, or its own rounding) must correct for it.

        Raise :class:`PageRenderFailed` for backend errors. Renderers may be
        called for many PDFs from one process but are not called concurrently
        for the same output directory.
        """


class PdfSlicer(ABC):
    """Common interface for extracting a page range into a new PDF.

    Slicing feeds the generation pipeline's per-window transcription, where the
    result is sent to the model as a PDF attachment — so a slice must stay a
    valid PDF with its text layer intact, not a rasterization.

    Bytes in, bytes out: both in-process backends can slice without touching
    the filesystem, and the caller already holds the document in memory. A
    subprocess backend spills to a tempdir inside its own implementation.
    """

    @abstractmethod
    def __init__(self, config: PdfConfig) -> None:
        """Prepare the backend, as :meth:`PageRenderer.__init__`."""

    @abstractmethod
    def slice(self, pdf_bytes: bytes, page_numbers: Sequence[int]) -> bytes:
        """Return a PDF containing only ``page_numbers`` from ``pdf_bytes``.

        ``page_numbers`` are 1-based, may be non-contiguous, and are emitted in
        ascending document order. They are validated against the document's
        real page count by :func:`slice_pages` before this is called, so an
        implementation may assume they are in range.

        Raise :class:`PdfSliceFailed` for backend errors.
        """


@dataclass(frozen=True)
class EngineSpec:
    """One engine's capabilities and the config fields it accepts.

    Pairs an engine's renderer and slicer so they share a single availability
    probe and a single config section: ``[pdf] provider`` selects the engine,
    and both capabilities follow from it. That is deliberate — the motivating
    use case is "no system binary installed", which is only satisfied when
    *both* operations avoid it.
    """

    name: EngineName
    renderer: type[PageRenderer]
    slicer: type[PdfSlicer]
    config_fields: frozenset[str] = frozenset()


def make_renderer(config: PdfConfig) -> PageRenderer:
    """Instantiate the renderer class for ``config.provider``."""
    return _ENGINES[config.provider].renderer(config)


def make_slicer(config: PdfConfig) -> PdfSlicer:
    """Instantiate the slicer class for ``config.provider``."""
    return _ENGINES[config.provider].slicer(config)


# ---------------------------------------------------------------------------
# Renderer-independent helpers
# ---------------------------------------------------------------------------


def ghostscript_path() -> str:
    """Return the absolute path to the ghostscript binary or raise :class:`GhostscriptNotFound`."""
    for name in GS_BINARIES:
        found = shutil.which(name)
        if found is not None:
            return found
    raise GhostscriptNotFound(
        f"ghostscript ({'/'.join(GS_BINARIES)}) is not installed or not on PATH"
    )


def pdf_page_count(path: Path) -> int:
    """Return the page count of ``path`` by walking pdfminer's page tree.

    Uses ``PDFPage.create_pages`` (a page-tree traversal, no layout analysis),
    so it's cheap and avoids trusting the possibly-wrong ``/Count`` field.
    """
    with path.open("rb") as fh:
        return _count_pages_in(fh)


def _count_pages_in(stream: BinaryIO) -> int:
    """Walk a PDF's page tree and count the pages it actually contains."""
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser

    document = PDFDocument(PDFParser(stream))
    return sum(1 for _ in PDFPage.create_pages(document))


def pdf_page_count_bytes(pdf_bytes: bytes) -> int:
    """Page count for an in-memory PDF, by the same page-tree walk as
    :func:`pdf_page_count`."""
    import io

    return _count_pages_in(io.BytesIO(pdf_bytes))


def slice_pages(
    pdf_bytes: bytes,
    page_numbers: Sequence[int],
    *,
    config: PdfConfig | None = None,
    total_pages: int | None = None,
) -> bytes:
    """Return a PDF holding only ``page_numbers`` from ``pdf_bytes``.

    ``page_numbers`` are 1-based and may be non-contiguous; the slice contains
    them in ascending document order. ``config`` selects the engine; ``None``
    means the ghostscript default.

    Page numbers are bounds-checked here, against the same page-tree walk that
    produced them upstream, rather than in each backend: an out-of-range
    request is a caller bug and should read as :class:`PdfSliceFailed` with the
    offending numbers named, not as whatever ``IndexError`` a backend happens
    to raise from inside a C extension.

    Pass ``total_pages`` when the caller already knows it — generation slices
    one window at a time from a document whose length it counted up front, and
    re-walking the page tree per window is pure waste on a long document. Doing
    so also opts out of the readability gate the self-counting path provides
    (ghostscript will emit output for input that is not a PDF, where the
    pdfminer walk raises), so supply it only for a document already read.
    """
    from .errors import PdfSliceFailed

    if not page_numbers:
        raise ValueError("page_numbers must be non-empty")
    if config is None:
        config = PdfConfig()

    ordered = sorted(set(page_numbers))
    if total_pages is not None:
        total = total_pages
    else:
        try:
            total = pdf_page_count_bytes(pdf_bytes)
        except Exception as exc:
            raise PdfSliceFailed(f"could not read the PDF to slice it: {exc}") from exc
    out_of_range = [n for n in ordered if n < 1 or n > total]
    if out_of_range:
        raise PdfSliceFailed(f"page(s) {out_of_range} out of range for a {total}-page PDF")

    return make_slicer(config).slice(pdf_bytes, ordered)


def pinned_slice(
    pdf_bytes: bytes,
    page_numbers: Sequence[int],
    *,
    config: PdfConfig | None,
    make: Callable[[], bytes],
) -> bytes:
    """The slice *make* produces of ``page_numbers`` (1-based), pinned to the
    running batch job, if any.

    Slicing is not byte-reproducible across processes: ghostscript's
    ``pdfwrite`` stamps a random trailer ``/ID`` and an XMP creation date, and
    PDFium's output differs run to run too. A batch job replays a resumed run's
    requests by their digest, so a fresh slice on resume would look like a
    changed input and be paid for again. Under a batch job the first slice of
    each (document content, engine, page list) is therefore recorded in the
    job's input store and every later run of the job reuses those exact bytes.
    Outside a job this is exactly ``make()``. Every slice that goes into a
    model request must come through here."""
    # No session can exist unless dgml_core.batch.jobs is loaded, so this
    # never imports the batch package.
    jobs = sys.modules.get("dgml_core.batch.jobs")
    job = None if jobs is None else jobs.active_session()
    if job is None:
        return make()
    engine = config.provider if config is not None else "default"
    pages = list(page_numbers)
    source = hashlib.sha256(pdf_bytes).hexdigest()[:32]
    span = hashlib.sha256(",".join(map(str, pages)).encode()).hexdigest()[:16]
    name = f"slices/{source}/{engine}/{pages[0]}-{pages[-1]}-{span}.pdf"
    data = job.rewind_input(name, make)
    assert data is not None  # slicing always produces bytes
    return bytes(data)


def _page_cache_root() -> Path | None:
    """Cache directory from ``$DGML_PAGE_CACHE``, or ``None`` when unset/empty."""
    root = os.environ.get(PAGE_CACHE_ENV)
    return Path(root) if root else None


def _pdf_cache_key(
    pdf_path: Path,
    dpi: int,
    renderer: EngineName = DEFAULT_ENGINE,
    image_spec: PageImageSpec = DEFAULT_PAGE_IMAGE_SPEC,
) -> str:
    """Content hash keying the render cache: renderer + dpi + edge bounds +
    JPEG quality + the PDF bytes.

    Everything that shapes the stored pixels is folded in so a change to any
    of it invalidates entries rather than serving mismatched renders for the
    same bytes — a 150-dpi render and a 300-dpi render of the same PDF get
    distinct entries, and so do a ghostscript render and a pypdfium2 render
    (their pixels differ), or two edge bounds.
    """
    spec = image_spec
    digest = hashlib.sha256(
        f"{renderer.value}:{dpi}:{spec.min_edge}:{spec.max_edge}:{PAGE_JPEG_QUALITY}:"
        f"smaller-of-png-jpeg\n".encode()
    )
    with pdf_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clear_page_images(output_dir: Path) -> None:
    """Create ``output_dir`` if needed and remove every stored page image in it,
    of either extension, so a retry or a format change leaves no orphans."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in list_page_images(output_dir):
        existing.unlink()


def _replace_pages_from(src_dir: Path, output_dir: Path) -> int:
    """Clear ``output_dir``'s page images and copy ``src_dir``'s in; return count."""
    _clear_page_images(output_dir)
    pages = list_page_images(src_dir)
    for image in pages:
        shutil.copy2(image, output_dir / image.name)
    return len(pages)


def _store_pages(raw_dir: Path, output_dir: Path, image_spec: PageImageSpec) -> int:
    """Fit each engine PNG in ``raw_dir`` to ``image_spec`` and store it in
    ``output_dir`` as the smaller of PNG and JPEG, resizing with Lanczos a page
    the engine wrote at its nominal size outside the bounds (one already
    written at its fitted size is left as is). Returns the page count.

    Raises :class:`PageRenderFailed` for an engine image Pillow cannot read.
    """
    from .errors import PageRenderFailed

    stored = 0
    for raw in sorted(raw_dir.glob(_ENGINE_PAGE_GLOB)):
        page = page_number_from_image_name(raw.name)
        if page is None:
            continue
        try:
            _store_page(raw, output_dir, page, image_spec)
        except (OSError, SyntaxError, ValueError) as exc:
            # SyntaxError is how Pillow's PNG plugin reports a malformed file.
            raise PageRenderFailed(f"could not store the render of page {page}: {exc}") from exc
        stored += 1
    return stored


def _store_page(raw: Path, output_dir: Path, page: int, image_spec: PageImageSpec) -> None:
    """Fit one engine PNG and store page ``page`` in ``output_dir`` as
    ``page_<page>.png`` or ``page_<page>.jpg``, whichever encoding is smaller
    (see :func:`_store_pages`). A tie keeps the lossless PNG.

    A page the fit leaves at its rendered size keeps the engine's own PNG
    bytes as the PNG candidate, so only the JPEG is encoded; a resized page
    encodes both.
    """
    from PIL import Image, PngImagePlugin

    jpeg = io.BytesIO()
    png: bytes | None = None
    # The engine's own output, so Image.open's decompression-bomb guard
    # (aimed at untrusted input) would only refuse a legitimately huge
    # sheet — the very page this step exists to shrink. Opening through
    # the PNG plugin directly skips that guard.
    with PngImagePlugin.PngImageFile(raw) as rendered:
        size = image_spec.fit(rendered.width, rendered.height)
        resized = size != rendered.size
        stored: Image.Image = rendered
        if resized:
            stored = rendered.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)
            buf = io.BytesIO()
            stored.save(buf, format="PNG")
            png = buf.getvalue()
        # JPEG holds only RGB/L; a mode change alone never re-encodes the PNG,
        # whose engine bytes already hold the page at its stored size.
        if stored.mode not in ("RGB", "L"):
            stored = stored.convert("RGB")
        stored.save(jpeg, format="JPEG", quality=PAGE_JPEG_QUALITY, optimize=True)
    png_size = len(png) if png is not None else raw.stat().st_size

    def target(ext: str) -> Path:
        return output_dir / (layout.PAGE_IMAGE_TEMPLATE % (page, ext))

    if png_size <= jpeg.getbuffer().nbytes:
        if png is None:
            shutil.move(raw, target(layout.PNG_PAGE_IMAGE_EXT))
            return
        target(layout.PNG_PAGE_IMAGE_EXT).write_bytes(png)
    else:
        target(layout.JPEG_PAGE_IMAGE_EXT).write_bytes(jpeg.getvalue())
    # Freed now rather than when the whole render is done, so the scratch
    # directory never holds every page's full-size render at once.
    raw.unlink()


def render_pages(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = DEFAULT_DPI,
    config: PdfConfig | None = None,
    image_spec: PageImageSpec = DEFAULT_PAGE_IMAGE_SPEC,
) -> int:
    """Render each PDF page at ``dpi`` and store it per ``image_spec``. Returns
    the number of pages written.

    ``config`` selects the backend; ``None`` means the ghostscript default
    (callers with a workspace at hand should pass
    :func:`load_pdf_config`'s result instead). Stale page images in
    ``output_dir`` are removed first so retries do not leave orphans behind.

    ``dpi`` trades resolution for speed and disk: 300 (the default) is archival
    quality; ~150 roughly halves rasterization time and file size and is usually
    ample for OCR and the downscaled clustering vision encoder. Whatever value
    is used here also has to reach digital text extraction, since ``page_text/``
    word boxes are expressed in *this* render's pixel space.

    ``image_spec`` bounds each page's longest edge (:class:`PageImageSpec`;
    ``[PAGE_MIN_EDGE, PAGE_MAX_EDGE]`` by default), and each page is stored as
    whichever of PNG and JPEG is smaller (see :data:`PAGE_JPEG_QUALITY`). A
    page outside the bounds is stored at :meth:`PageImageSpec.fit` of its
    rendered size, so its effective resolution differs from ``dpi`` — digital
    text extraction must be given the same spec to land in the same pixel
    space.

    When ``$DGML_PAGE_CACHE`` is set, an identical PDF (same bytes, renderer,
    dpi, and edge bounds) rendered before is served from that cache without invoking
    the backend (which then need not even be installed); otherwise the render
    is populated into the cache on success.
    """
    if config is None:
        config = PdfConfig()

    cache_entry: Path | None = None
    cache_root = _page_cache_root()
    if cache_root is not None:
        cache_entry = cache_root / _pdf_cache_key(pdf_path, dpi, config.provider, image_spec)
        if (cache_entry / _CACHE_COMPLETE_MARKER).exists():
            return _replace_pages_from(cache_entry, output_dir)

    renderer = make_renderer(config)
    _clear_page_images(output_dir)

    # Beside ``output_dir`` rather than in ``$TMPDIR``: the full-size engine
    # renders stay on the same filesystem as the stored images (LocalStore's
    # workspace staging, say), never on a RAM-backed /tmp.
    with tempfile.TemporaryDirectory(prefix=".dgml-render-", dir=output_dir.parent) as raw_dir:
        renderer.render(pdf_path, Path(raw_dir), dpi=dpi, image_spec=image_spec)
        count = _store_pages(Path(raw_dir), output_dir, image_spec)

    if cache_entry is not None:
        _populate_cache(cache_entry, output_dir)
    return count


def _populate_cache(cache_entry: Path, output_dir: Path) -> None:
    """Best-effort copy of the fresh render into the cache, marked complete last.

    Writing the ``.complete`` marker only after every image is copied means a
    reader either sees a fully populated entry or treats it as a miss — never a
    partial one. Cache I/O failures are swallowed: the render already succeeded.
    """
    try:
        cache_entry.mkdir(parents=True, exist_ok=True)
        for image in list_page_images(output_dir):
            shutil.copy2(image, cache_entry / image.name)
        (cache_entry / _CACHE_COMPLETE_MARKER).write_text("", encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Engine registry
#
# Built at module load by a function call so the engine modules' imports of
# this module see fully-defined PageRenderer / PdfSlicer ABCs and the PdfConfig
# dataclass. Doing the import here (rather than at the top of the file) avoids a
# circular dependency: pages_ghostscript / pages_pypdfium2 import from us.
# (Same pattern as dgml_core.ocr's _PROVIDERS.)
# ---------------------------------------------------------------------------


def _register_engines(specs: list[EngineSpec]) -> dict[EngineName, EngineSpec]:
    """Build a name-keyed registry from a list of engine specs.

    Iterating a list (rather than a dict literal) lets us detect collisions:
    two specs claiming the same :class:`EngineName` is a copy-paste bug that a
    dict literal would silently resolve by overwriting. Raising here keeps the
    failure at import time, before any render or slice call.
    """
    registry: dict[EngineName, EngineSpec] = {}
    for spec in specs:
        if spec.name in registry:
            raise RuntimeError(
                f"duplicate PDF engine registration for {spec.name.value!r}: "
                f"{registry[spec.name].renderer.__name__} and {spec.renderer.__name__}"
            )
        registry[spec.name] = spec
    return registry


def _build_registry() -> dict[EngineName, EngineSpec]:
    from .pages_ghostscript import GhostscriptRenderer, GhostscriptSlicer
    from .pages_pypdfium2 import Pypdfium2Renderer, Pypdfium2Slicer

    return _register_engines(
        [
            EngineSpec(
                name=EngineName.GHOSTSCRIPT,
                renderer=GhostscriptRenderer,
                slicer=GhostscriptSlicer,
            ),
            EngineSpec(
                name=EngineName.PYPDFIUM2,
                renderer=Pypdfium2Renderer,
                slicer=Pypdfium2Slicer,
            ),
        ]
    )


_ENGINES: dict[EngineName, EngineSpec] = _build_registry()
