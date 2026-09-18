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
engine-independent: the page filename contract (``page_N.png``), the optional
content-addressed render cache, page counting, and slice bounds-checking.

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
import os
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

from .errors import GhostscriptNotFound, PdfConfigInvalid
from .models_config import ConfigSection

if TYPE_CHECKING:
    from .storage import Workspace

DEFAULT_DPI = 300
PAGE_FILENAME_TEMPLATE = "page_%d.png"
PAGE_GLOB = "page_*.png"

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


# ---------------------------------------------------------------------------
# Renderer interface
# ---------------------------------------------------------------------------


class PageRenderer(ABC):
    """Common interface for PDF page-image backends.

    Implementations are constructed from a :class:`PdfConfig` (which is where
    lazy package imports / binary probes live) and implement :meth:`render` for
    a whole PDF. The shared wrapper :func:`render_pages` handles the render
    cache, clearing stale page images, and counting the output — renderers only
    need to write ``page_N.png`` files.
    """

    @abstractmethod
    def __init__(self, config: PdfConfig) -> None:
        """Prepare the backend: lazy-import its package or probe its binary.
        Raise :class:`EngineNotAvailable` (or its ghostscript-specific subclass
        :class:`GhostscriptNotFound`) with an actionable install hint when the
        backend is missing."""

    @abstractmethod
    def render(self, pdf_path: Path, output_dir: Path, *, dpi: int) -> None:
        """Rasterize every page of ``pdf_path`` into ``output_dir``.

        Write one PNG per page named per :data:`PAGE_FILENAME_TEMPLATE`
        (``page_1.png`` …, 1-based, ghostscript's own numbering).
        ``output_dir`` exists and holds no stale page images when called.

        The image written for a page MUST be exactly
        ``round(pts * dpi / 72)`` pixels on each axis, measured from the
        **MediaBox** and after applying ``/Rotate`` — that is the coordinate
        space ``page_text/`` word boxes and every ``dg:origin`` attribute are
        expressed in. A backend whose natural output differs (a different page
        box, or its own rounding) must correct for it.

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


def _page_cache_root() -> Path | None:
    """Cache directory from ``$DGML_PAGE_CACHE``, or ``None`` when unset/empty."""
    root = os.environ.get(PAGE_CACHE_ENV)
    return Path(root) if root else None


def _pdf_cache_key(pdf_path: Path, dpi: int, renderer: EngineName = DEFAULT_ENGINE) -> str:
    """Content hash keying the render cache: renderer + dpi + the PDF bytes.

    Renderer and dpi are folded in so a change to either invalidates entries
    rather than serving mismatched renders for the same bytes — a 150-dpi
    render and a 300-dpi render of the same PDF get distinct entries, and so
    do a ghostscript render and a pypdfium2 render (their pixels differ).
    """
    digest = hashlib.sha256(f"{renderer.value}:{dpi}\n".encode())
    with pdf_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_pages_from(src_dir: Path, output_dir: Path) -> int:
    """Clear ``output_dir``'s page PNGs and copy ``src_dir``'s in; return count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob(PAGE_GLOB):
        existing.unlink()
    pages = sorted(src_dir.glob(PAGE_GLOB))
    for png in pages:
        shutil.copy2(png, output_dir / png.name)
    return len(pages)


def render_pages(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = DEFAULT_DPI,
    config: PdfConfig | None = None,
) -> int:
    """Render each PDF page to a PNG at ``dpi``. Returns the number of pages written.

    ``config`` selects the backend; ``None`` means the ghostscript default
    (callers with a workspace at hand should pass
    :func:`load_pdf_config`'s result instead). Stale page images in
    ``output_dir`` are removed first so retries do not leave orphans behind.

    ``dpi`` trades resolution for speed and disk: 300 (the default) is archival
    quality; ~150 roughly halves rasterization time and file size and is usually
    ample for OCR and the downscaled clustering vision encoder. Whatever value
    is used here also has to reach digital text extraction, since ``page_text/``
    word boxes are expressed in *this* render's pixel space.

    When ``$DGML_PAGE_CACHE`` is set, an identical PDF (same bytes, renderer,
    and dpi) rendered before is served from that cache without invoking
    the backend (which then need not even be installed); otherwise the render
    is populated into the cache on success.

    PNG (not JPEG) is the canonical format: pixel-perfect for text-on-white
    document scans, ~5-10x smaller on disk than JPEG q92 for these
    workloads, and the same format the generation pipeline already
    consumes for LLM input — so workspace renders can be reused directly
    rather than re-rasterized through a second renderer.
    """
    if config is None:
        config = PdfConfig()

    cache_entry: Path | None = None
    cache_root = _page_cache_root()
    if cache_root is not None:
        cache_entry = cache_root / _pdf_cache_key(pdf_path, dpi, config.provider)
        if (cache_entry / _CACHE_COMPLETE_MARKER).exists():
            return _replace_pages_from(cache_entry, output_dir)

    renderer = make_renderer(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob(PAGE_GLOB):
        existing.unlink()

    renderer.render(pdf_path, output_dir, dpi=dpi)

    count = len(list(output_dir.glob(PAGE_GLOB)))
    if cache_entry is not None:
        _populate_cache(cache_entry, output_dir)
    return count


def _populate_cache(cache_entry: Path, output_dir: Path) -> None:
    """Best-effort copy of the fresh render into the cache, marked complete last.

    Writing the ``.complete`` marker only after every PNG is copied means a
    reader either sees a fully populated entry or treats it as a miss — never a
    partial one. Cache I/O failures are swallowed: the render already succeeded.
    """
    try:
        cache_entry.mkdir(parents=True, exist_ok=True)
        for png in output_dir.glob(PAGE_GLOB):
            shutil.copy2(png, cache_entry / png.name)
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
