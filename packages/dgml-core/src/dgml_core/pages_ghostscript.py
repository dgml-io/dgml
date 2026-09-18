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

"""The ghostscript engine: page rendering and page slicing (the default).

Ghostscript is invoked as a subprocess. It is a system-level dependency,
not a Python package — see CLAUDE.md for the licensing rationale.

Rendering uses the ``png16m`` device; slicing uses ``pdfwrite`` with
``-sPageList``. Both go through :func:`_run_gs`, so a missing binary and a
non-zero exit report identically whichever capability was invoked.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .errors import PageRenderFailed, PdfSliceFailed
from .pages import (
    GS_TIMEOUT_SECONDS,
    PAGE_FILENAME_TEMPLATE,
    PageRenderer,
    PdfConfig,
    PdfSlicer,
    ghostscript_path,
)


def _run_gs(gs: str, args: list[str], *, what: str) -> subprocess.CompletedProcess[str]:
    """Run the ghostscript binary at ``gs`` with the shared flag set.

    ``gs`` is the path each capability probed at construction time, so the
    binary is located once per engine instance rather than once per call.
    ``what`` is ``"render"`` or ``"slice"`` and selects the exception type, so
    the two capabilities surface as the soft-fail codes their callers already
    handle (``PAGE_RENDER_FAILED`` / ``PDF_SLICE_FAILED``).
    """
    failure: type[PageRenderFailed] | type[PdfSliceFailed] = (
        PageRenderFailed if what == "render" else PdfSliceFailed
    )
    cmd = [gs, "-dNOPAUSE", "-dBATCH", "-dQUIET", "-dSAFER", *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=GS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise failure(f"ghostscript timed out after {GS_TIMEOUT_SECONDS}s") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise failure(f"ghostscript exited {result.returncode}: {stderr}")
    return result


class GhostscriptRenderer(PageRenderer):
    """Rasterize pages with ghostscript's ``png16m`` device."""

    def __init__(self, config: PdfConfig) -> None:
        # Probing here (not in render) keeps the "engine missing" failure at
        # construction time, symmetric with the SDK imports in OCR providers —
        # and a cache hit never constructs a renderer, so gs need not be
        # installed to serve one.
        self._gs = ghostscript_path()

    def render(self, pdf_path: Path, output_dir: Path, *, dpi: int) -> None:
        _run_gs(
            self._gs,
            [
                "-sDEVICE=png16m",
                f"-r{dpi}",
                f"-sOutputFile={output_dir / PAGE_FILENAME_TEMPLATE}",
                str(pdf_path),
            ],
            what="render",
        )


class GhostscriptSlicer(PdfSlicer):
    """Extract pages with ghostscript's ``pdfwrite`` device.

    ``-sPageList`` emits the selected pages in ascending document order, so no
    Python PDF library is involved. Ghostscript is a subprocess and needs real
    paths, so the in-memory document is spilled to a tempdir here — the cost
    stays with the backend that requires it rather than in the shared wrapper.
    """

    def __init__(self, config: PdfConfig) -> None:
        self._gs = ghostscript_path()

    def slice(self, pdf_bytes: bytes, page_numbers: Sequence[int]) -> bytes:
        page_list = ",".join(str(n) for n in page_numbers)
        with tempfile.TemporaryDirectory(prefix="dgml-gs-slice-") as td:
            tmp = Path(td)
            src, out = tmp / "in.pdf", tmp / "out.pdf"
            src.write_bytes(pdf_bytes)
            _run_gs(
                self._gs,
                [
                    "-sDEVICE=pdfwrite",
                    f"-sPageList={page_list}",
                    f"-sOutputFile={out}",
                    str(src),
                ],
                what="slice",
            )
            if not out.exists() or out.stat().st_size == 0:
                raise PdfSliceFailed(f"ghostscript wrote no output for pages {page_list}")
            return out.read_bytes()
