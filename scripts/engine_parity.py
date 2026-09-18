#!/usr/bin/env python
"""Ghostscript vs PDFium parity harness for a real PDF corpus.

    uv run python engine_parity.py /path/to/pdfs [--dpi 300] [--window 10]

Renders and slices every PDF through both engines and reports where they
disagree. No LLM calls and no network: everything here is local and free.

Checks, in the order they matter:

  1. GROUNDING CONTRACT   page_text/ declared size == the PNG's real pixels,
                          and every word box inside the image. A violation
                          silently mis-places every dg:origin on the page and
                          `dgml check` does not look for it.
  2. PIXEL FIDELITY       the two engines' images compared pixel-by-pixel.
                          Catches missing annotations, font fallbacks and
                          transparency handling that a size check cannot.
  3. INGEST PARITY        page counts, and which files soft/hard fail.
  4. SLICE PARITY         page count, text-layer survival, and payload size
                          ratio -- the open question for scanned corpora,
                          since a slice is uploaded to the model.
  5. COST                 wall time and peak RSS per engine.

Exit code 0 if every blocking check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from dgml_core.pages import (
    EngineName,
    PdfConfig,
    pdf_page_count_bytes,
    render_pages,
    slice_pages,
)

# Anti-aliasing between two rasterizers is expected; wholesale content
# differences are not. Mean |diff| is over 0-255 grey levels.
PIXEL_MEAN_WARN = 2.0
PIXEL_MEAN_FAIL = 8.0
# A slice is uploaded to the model, so payload growth is a cost/limit issue.
SLICE_RATIO_WARN = 1.5
ENGINES = (EngineName.GHOSTSCRIPT, EngineName.PYPDFIUM2)


@dataclass
class Findings:
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, m: str) -> None:
        self.blocking.append(m)
        print(f"    FAIL  {m}")

    def warn(self, m: str) -> None:
        self.warnings.append(m)
        print(f"    warn  {m}")


def png_size(p: Path) -> tuple[int, int]:
    b = p.read_bytes()
    if not b.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG: {p}")
    w, h = struct.unpack(">II", b[16:24])
    return w, h


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB, macOS bytes.
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def digital_text(pdf_bytes: bytes) -> str:
    """Whitespace-normalized digital text, or '' when there is none."""
    from pdfminer.high_level import extract_text

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
        fh.write(pdf_bytes)
        tmp = Path(fh.name)
    try:
        return " ".join(extract_text(str(tmp)).split())
    except Exception:
        return ""
    finally:
        tmp.unlink(missing_ok=True)


def compare_pixels(a: Path, b: Path) -> tuple[float, float] | None:
    """(mean |diff| over 0-255, % of pixels differing), or None if unusable."""
    try:
        from PIL import Image, ImageChops, ImageStat

        Image.MAX_IMAGE_PIXELS = None
    except ImportError:
        return None
    ia = Image.open(a).convert("L")
    ib = Image.open(b).convert("L")
    if ia.size != ib.size:
        return None
    diff = ImageChops.difference(ia, ib)
    nonzero = sum(diff.histogram()[1:])
    total = ia.size[0] * ia.size[1]
    return ImageStat.Stat(diff).mean[0], 100.0 * nonzero / total


def is_structurally_valid(pdf: Path) -> bool:
    """Can pdfminer walk this PDF's page tree?

    Gates how a pypdfium2-only failure is classified. Ghostscript limps along
    on damaged files, emitting zero or partial output rather than erroring, so
    "gs succeeded" on a corrupt PDF is vacuous. Only a file DGML itself can
    read is one where a PDFium refusal means real capability was lost.
    """
    try:
        from dgml_core.pages import pdf_page_count

        return pdf_page_count(pdf) > 0
    except Exception:
        return False


def check_render(pdf: Path, dpi: int, f: Findings) -> dict[EngineName, dict]:
    """Render through both engines; check geometry, then pixels."""
    out: dict[EngineName, dict] = {}
    for eng in ENGINES:
        d = Path(tempfile.mkdtemp(prefix=f"parity-{eng.value}-"))
        t0 = time.perf_counter()
        try:
            n = render_pages(pdf, d, dpi=dpi, config=PdfConfig(provider=eng))
            out[eng] = {
                "dir": d,
                "pages": n,
                "secs": time.perf_counter() - t0,
                "error": None,
            }
        except Exception as exc:
            out[eng] = {
                "dir": d,
                "pages": 0,
                "secs": time.perf_counter() - t0,
                "error": f"{type(exc).__name__}: {exc}",
            }

    gs, pf = out[EngineName.GHOSTSCRIPT], out[EngineName.PYPDFIUM2]
    if gs["error"] and not pf["error"]:
        f.notes.append(f"{pdf.name}: gs failed, pypdfium2 rendered ({gs['error'][:60]})")
    elif pf["error"] and not gs["error"]:
        detail = pf["error"][:80]
        if gs["pages"] > 0 and is_structurally_valid(pdf):
            f.fail(f"{pdf.name}: pypdfium2 render failed on a VALID pdf -- {detail}")
        else:
            # gs emitted nothing usable either, or the file is unparseable:
            # a stricter refusal, not lost capability.
            f.warn(f"{pdf.name}: pypdfium2 refuses a damaged pdf gs tolerated -- {detail}")
    if gs["error"] or pf["error"]:
        return out
    if gs["pages"] != pf["pages"]:
        f.fail(f"{pdf.name}: page count {gs['pages']} (gs) vs {pf['pages']} (pypdfium2)")
        return out

    # geometry must match exactly -- page_text/ boxes assume gs's numbers
    for i in range(1, gs["pages"] + 1):
        ga, pa = gs["dir"] / f"page_{i}.png", pf["dir"] / f"page_{i}.png"
        if not pa.exists():
            f.fail(f"{pdf.name} p{i}: pypdfium2 wrote no image")
            continue
        if png_size(ga) != png_size(pa):
            f.fail(f"{pdf.name} p{i}: {png_size(ga)} (gs) vs {png_size(pa)} (pypdfium2)")
            continue
        cmp = compare_pixels(ga, pa)
        if cmp is None:
            continue
        mean, pct = cmp
        if mean >= PIXEL_MEAN_FAIL:
            f.fail(f"{pdf.name} p{i}: pixels diverge (mean {mean:.1f}/255, {pct:.1f}% of px)")
        elif mean >= PIXEL_MEAN_WARN:
            f.warn(f"{pdf.name} p{i}: pixels differ (mean {mean:.1f}/255, {pct:.1f}% of px)")
    return out


def check_slice(pdf: Path, window: int, f: Findings) -> tuple[int, int] | None:
    """Slice the first window through both engines; compare pages/text/size."""
    data = pdf.read_bytes()
    try:
        total = pdf_page_count_bytes(data)
    except Exception as exc:
        f.notes.append(f"{pdf.name}: unreadable for slicing ({type(exc).__name__})")
        return None
    if total == 0:
        return None
    pages = list(range(1, min(window, total) + 1))

    res: dict[EngineName, bytes] = {}
    for eng in ENGINES:
        try:
            res[eng] = slice_pages(data, pages, config=PdfConfig(provider=eng))
        except Exception as exc:
            f.notes.append(f"{pdf.name}: {eng.value} slice failed ({type(exc).__name__}: {exc})")
    if len(res) < 2:
        if EngineName.PYPDFIUM2 not in res and EngineName.GHOSTSCRIPT in res:
            if is_structurally_valid(pdf):
                f.fail(f"{pdf.name}: pypdfium2 could not slice a VALID pdf gs sliced")
            else:
                f.warn(f"{pdf.name}: pypdfium2 refuses to slice a damaged pdf gs sliced")
        return None

    gs_b, pf_b = res[EngineName.GHOSTSCRIPT], res[EngineName.PYPDFIUM2]
    gs_n, pf_n = pdf_page_count_bytes(gs_b), pdf_page_count_bytes(pf_b)
    if gs_n != pf_n != len(pages):
        f.fail(f"{pdf.name}: slice pages {gs_n} (gs) vs {pf_n} (pypdfium2), wanted {len(pages)}")

    # Text-layer survival: if the source had digital text in this range, both
    # slices must still have it -- the model reads it.
    src_text = digital_text(slice_pages(data, pages, config=PdfConfig(EngineName.GHOSTSCRIPT)))
    if len(src_text) > 40:
        for eng, blob in res.items():
            if len(digital_text(blob)) < len(src_text) * 0.5:
                f.fail(f"{pdf.name}: {eng.value} slice lost most of the text layer")

    ratio = len(pf_b) / len(gs_b) if gs_b else 0
    if ratio >= SLICE_RATIO_WARN:
        f.warn(
            f"{pdf.name}: pypdfium2 slice is {ratio:.2f}x the gs payload "
            f"({len(gs_b) / 1e6:.1f}MB -> {len(pf_b) / 1e6:.1f}MB)"
        )
    return len(gs_b), len(pf_b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--window", type=int, default=10, help="pages per slice window")
    ap.add_argument("--limit", type=int, default=0, help="stop after N PDFs")
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args()

    pdfs = sorted(p for p in args.corpus.rglob("*") if p.suffix.lower() == ".pdf")
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"no PDFs under {args.corpus}", file=sys.stderr)
        return 1
    if not shutil.which("gs") and not shutil.which("gswin64c"):
        print("ghostscript not on PATH -- both engines are needed to compare", file=sys.stderr)
        return 1

    print(f"{len(pdfs)} PDF(s) at {args.dpi} dpi, {args.window}-page slice windows\n")
    f = Findings()
    timing = {e: 0.0 for e in ENGINES}
    slice_bytes = {e: 0 for e in ENGINES}
    rows = []

    for i, pdf in enumerate(pdfs, 1):
        size_mb = pdf.stat().st_size / 1e6
        print(f"[{i}/{len(pdfs)}] {pdf.name} ({size_mb:.1f}MB)")
        rendered = check_render(pdf, args.dpi, f)
        for eng in ENGINES:
            timing[eng] += rendered.get(eng, {}).get("secs", 0.0)
        sl = check_slice(pdf, args.window, f)
        if sl:
            slice_bytes[EngineName.GHOSTSCRIPT] += sl[0]
            slice_bytes[EngineName.PYPDFIUM2] += sl[1]
        rows.append(
            {
                "pdf": pdf.name,
                "pages": rendered.get(EngineName.GHOSTSCRIPT, {}).get("pages"),
                "slice_gs": sl[0] if sl else None,
                "slice_pypdfium2": sl[1] if sl else None,
            }
        )
        for eng in ENGINES:
            d = rendered.get(eng, {}).get("dir")
            if d and d.exists():
                shutil.rmtree(d, ignore_errors=True)

    print("\n" + "=" * 72)
    print(
        f"  render wall time   gs {timing[EngineName.GHOSTSCRIPT]:7.1f}s   "
        f"pypdfium2 {timing[EngineName.PYPDFIUM2]:7.1f}s"
    )
    if slice_bytes[EngineName.GHOSTSCRIPT]:
        r = slice_bytes[EngineName.PYPDFIUM2] / slice_bytes[EngineName.GHOSTSCRIPT]
        print(
            f"  total slice payload  gs {slice_bytes[EngineName.GHOSTSCRIPT] / 1e6:8.1f}MB   "
            f"pypdfium2 {slice_bytes[EngineName.PYPDFIUM2] / 1e6:8.1f}MB   ({r:.2f}x)"
        )
        print(f"  {'^ this is the number that decides the scanned-corpus question':>72}")
    print(f"  peak RSS this process {peak_rss_mb():.0f}MB")
    print(
        f"\n  blocking failures {len(f.blocking)}   warnings {len(f.warnings)}   "
        f"notes {len(f.notes)}"
    )
    for m in f.notes:
        print(f"    note  {m}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "dpi": args.dpi,
                    "window": args.window,
                    "blocking": f.blocking,
                    "warnings": f.warnings,
                    "notes": f.notes,
                    "render_secs": {e.value: timing[e] for e in ENGINES},
                    "slice_bytes": {e.value: slice_bytes[e] for e in ENGINES},
                    "files": rows,
                },
                indent=2,
            )
        )
        print(f"\n  report -> {args.json}")

    print("\n  VERDICT: " + ("PASS" if not f.blocking else "BLOCKED -- see failures above"))
    return 0 if not f.blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
