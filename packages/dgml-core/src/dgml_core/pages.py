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

"""PDF page rendering via the system ``ghostscript`` binary.

Ghostscript is invoked as a subprocess. It is a system-level dependency,
not a Python package — see CLAUDE.md for the licensing rationale.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import GhostscriptNotFound, PageRenderFailed, PdfSliceFailed

DEFAULT_DPI = 300
RENDERER_NAME = "ghostscript"
PAGE_FILENAME_TEMPLATE = "page_%d.png"
PAGE_GLOB = "page_*.png"
GS_TIMEOUT_SECONDS = 600

# Optional content-addressed render cache. When ``$DGML_PAGE_CACHE`` names a
# directory, :func:`render_pages` copies its output there keyed by the PDF's
# content hash (plus renderer + dpi) and, on a later call for identical bytes,
# copies back instead of re-invoking ghostscript. Off by default — rendering is
# unchanged unless the env var is set. Intended for workflows that re-ingest the
# same PDFs into many workspaces (e.g. the clustering sweep's per-cell
# workspaces), where the render is otherwise repeated once per workspace.
PAGE_CACHE_ENV = "DGML_PAGE_CACHE"
_CACHE_COMPLETE_MARKER = ".complete"

# On Windows the console executable is named ``gswin64c`` / ``gswin32c``, not
# ``gs`` (the Artifex installer ships no ``gs.exe``). Probe those first, then
# fall back to ``gs`` for MSYS/Cygwin shells that expose the Unix name.
GS_BINARIES: tuple[str, ...] = (
    ("gswin64c", "gswin32c", "gs") if sys.platform == "win32" else ("gs",)
)


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
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfparser import PDFParser

    with path.open("rb") as fh:
        document = PDFDocument(PDFParser(fh))
        return sum(1 for _ in PDFPage.create_pages(document))


def extract_pdf_pages(pdf_path: Path, output_path: Path, page_numbers: Sequence[int]) -> None:
    """Write a new PDF at ``output_path`` containing only ``page_numbers``.

    ``page_numbers`` are 1-based and may be non-contiguous; ghostscript's
    ``-sPageList`` emits the selected pages in ascending document order. Uses
    the ``pdfwrite`` device, so no Python PDF library is involved.
    """
    if not page_numbers:
        raise ValueError("page_numbers must be non-empty")

    gs = ghostscript_path()
    page_list = ",".join(str(n) for n in page_numbers)
    cmd = [
        gs,
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        "-dSAFER",
        "-sDEVICE=pdfwrite",
        f"-sPageList={page_list}",
        f"-sOutputFile={output_path}",
        str(pdf_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=GS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise PdfSliceFailed(f"ghostscript timed out after {GS_TIMEOUT_SECONDS}s") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise PdfSliceFailed(f"ghostscript exited {result.returncode}: {stderr}")
    if not output_path.exists():
        raise PdfSliceFailed(f"ghostscript wrote no output for pages {page_list}")


def _page_cache_root() -> Path | None:
    """Cache directory from ``$DGML_PAGE_CACHE``, or ``None`` when unset/empty."""
    root = os.environ.get(PAGE_CACHE_ENV)
    return Path(root) if root else None


def _pdf_cache_key(pdf_path: Path) -> str:
    """Content hash keying the render cache: renderer + dpi + the PDF bytes.

    Renderer and dpi are folded in so a change to either invalidates entries
    rather than serving mismatched renders for the same bytes.
    """
    digest = hashlib.sha256(f"{RENDERER_NAME}:{DEFAULT_DPI}\n".encode())
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
) -> int:
    """Render each PDF page to a PNG. Returns the number of pages written.

    Stale page images in ``output_dir`` are removed first so retries do not
    leave orphans behind.

    When ``$DGML_PAGE_CACHE`` is set, an identical PDF (same bytes, renderer,
    and dpi) rendered before is served from that cache without invoking
    ghostscript; otherwise the render is populated into the cache on success.

    PNG (not JPEG) is the canonical format: pixel-perfect for text-on-white
    document scans, ~5-10x smaller on disk than JPEG q92 for these
    workloads, and the same format the generation pipeline already
    consumes for LLM input — so workspace renders can be reused directly
    rather than re-rasterized through a second renderer.
    """
    cache_entry: Path | None = None
    cache_root = _page_cache_root()
    if cache_root is not None:
        cache_entry = cache_root / _pdf_cache_key(pdf_path)
        if (cache_entry / _CACHE_COMPLETE_MARKER).exists():
            # Cache hit — no ghostscript needed (so it need not even be installed).
            return _replace_pages_from(cache_entry, output_dir)

    gs = ghostscript_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob(PAGE_GLOB):
        existing.unlink()

    output_template = str(output_dir / PAGE_FILENAME_TEMPLATE)
    cmd = [
        gs,
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        "-dSAFER",
        "-sDEVICE=png16m",
        f"-r{DEFAULT_DPI}",
        f"-sOutputFile={output_template}",
        str(pdf_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=GS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise PageRenderFailed(f"ghostscript timed out after {GS_TIMEOUT_SECONDS}s") from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise PageRenderFailed(f"ghostscript exited {result.returncode}: {stderr}")

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
