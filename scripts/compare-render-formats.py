"""Compare 300dpi JPEG vs 150dpi PNG renderings on OCR + digital extraction.

One-off validation harness for the canonical-format flip (see issue #24
follow-up plan). For each input PDF, renders the PDF twice via ``gs``
— once as 300dpi JPEG, once as 150dpi PNG — then runs Azure Document
Intelligence OCR and pdfminer digital extraction over both renderings,
and emits a markdown report comparing text content and word-bbox
alignment in normalized 0-1000 space.

The harness shells out to ghostscript directly so it stays independent
of any in-progress refactor of ``dgml.pages.render_pages`` — we want
to compare today's production behavior against the *proposed* behavior
honestly.

Usage::

    export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=...
    export AZURE_DOCUMENT_INTELLIGENCE_KEY=...
    uv run python scripts/compare-render-formats.py \\
        examples/course-descriptions/documents/.../*.pdf \\
        --out /tmp/comparison_report.md

Skip OCR for a quick local sanity check on the wiring::

    uv run python scripts/compare-render-formats.py path/to.pdf --skip-ocr
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from dgml_core.ocr import OcrConfig, OcrProviderName, load_ocr_config
from dgml_core.pages import ghostscript_path
from dgml_core.storage import Workspace
from dgml_core.text_extraction import extract_text_digital

# ---------------------------------------------------------------------------
# Render specs — two configurations we want to compare side-by-side.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderSpec:
    """A single ghostscript rendering configuration."""

    name: str  # short id used in report tables and filenames
    device: str  # ghostscript ``-sDEVICE`` value
    suffix: str  # output file extension (with leading dot)
    dpi: int
    extra_flags: tuple[str, ...] = ()


JPEG_300 = RenderSpec(
    name="jpeg-300",
    device="jpeg",
    suffix=".jpg",
    dpi=300,
    extra_flags=("-dJPEGQ=92",),
)
PNG_150 = RenderSpec(
    name="png-150",
    device="png16m",
    suffix=".png",
    dpi=150,
)


def parse_spec(s: str) -> RenderSpec:
    """Parse a ``<format>-<dpi>`` shorthand into a RenderSpec.

    Supported formats: ``jpeg`` (writes ``.jpg`` at quality 92),
    ``png`` (writes ``.png`` via ``png16m``).
    """
    try:
        fmt, dpi_str = s.rsplit("-", 1)
        dpi = int(dpi_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid render spec {s!r}; expected '<format>-<dpi>'") from exc
    fmt = fmt.lower()
    if fmt == "jpeg":
        return RenderSpec(
            name=s, device="jpeg", suffix=".jpg", dpi=dpi, extra_flags=("-dJPEGQ=92",)
        )
    if fmt == "png":
        return RenderSpec(name=s, device="png16m", suffix=".png", dpi=dpi)
    raise ValueError(f"unsupported format {fmt!r} in spec {s!r}; expected 'jpeg' or 'png'")


# ---------------------------------------------------------------------------
# Image dimension parsing — format-specific minimal parsers, no PIL dep.
# ---------------------------------------------------------------------------


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_dimensions(data: bytes) -> tuple[int, int]:
    """Return ``(width, height)`` from a PNG's IHDR chunk.

    PNG layout: 8-byte signature, then a chunk laid out as 4-byte
    big-endian length, 4-byte type ('IHDR'), then payload starting with
    width (uint32 BE) at byte offset 16 and height (uint32 BE) at offset 20.
    """
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("not a PNG: missing signature")
    if len(data) < 24:
        raise ValueError("truncated PNG: header less than 24 bytes")
    if data[12:16] != b"IHDR":
        raise ValueError("PNG IHDR chunk missing")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Return ``(width, height)`` from a JPEG's SOF marker.

    Inlined here (rather than imported from dgml_core.ocr) so the harness can
    still compare JPEG baselines even after dgml.ocr drops JPEG support
    once the canonical format flips to PNG.
    """
    n = len(data)
    if n < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise ValueError("not a JPEG: missing SOI marker")
    i = 2
    while i + 1 < n:
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 7 > n:
                raise ValueError("truncated SOF segment")
            height, width = struct.unpack(">HH", data[i + 3 : i + 7])
            return width, height
        if i + 2 > n:
            raise ValueError(f"truncated marker 0x{marker:02x}")
        seg_len = struct.unpack(">H", data[i : i + 2])[0]
        i += seg_len
    raise ValueError("no SOF marker found in JPEG")


def image_dimensions(data: bytes, suffix: str) -> tuple[int, int]:
    """Dispatch to the right parser based on file suffix."""
    if suffix == ".jpg":
        return jpeg_dimensions(data)
    if suffix == ".png":
        return png_dimensions(data)
    raise ValueError(f"unsupported image suffix {suffix!r}")


# ---------------------------------------------------------------------------
# Ghostscript wrapper.
# ---------------------------------------------------------------------------


def render_with_gs(pdf_path: Path, output_dir: Path, spec: RenderSpec) -> list[Path]:
    """Render every page of ``pdf_path`` into ``output_dir`` per ``spec``.

    Returns the list of output paths sorted by page number (1-based).
    Mirrors ``dgml.pages.render_pages`` argument shape but the format
    and DPI are caller-controlled.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / f"page_%d{spec.suffix}")
    cmd = [
        ghostscript_path(),
        "-dNOPAUSE",
        "-dBATCH",
        "-dQUIET",
        "-dSAFER",
        f"-sDEVICE={spec.device}",
        f"-r{spec.dpi}",
        *spec.extra_flags,
        f"-sOutputFile={template}",
        str(pdf_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"ghostscript ({spec.name}) exited {result.returncode}: {(result.stderr or '').strip()}"
        )

    def _page_index(path: Path) -> int:
        stem = path.stem  # "page_3"
        return int(stem.split("_", 1)[1])

    return sorted(output_dir.glob(f"page_*{spec.suffix}"), key=_page_index)


# ---------------------------------------------------------------------------
# Word data model — shared between OCR and digital extraction outputs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Word:
    """A single extracted word with its pixel-space bbox.

    ``bbox`` is ``(left, top, right, bottom)`` in image pixels relative
    to ``page_dims`` (width, height). Normalize to a DPI-independent
    0-1000 space via :meth:`normalized_bbox`.
    """

    text: str
    bbox: tuple[int, int, int, int]
    page_dims: tuple[int, int]  # (width, height) in pixels

    def normalized_bbox(self) -> tuple[float, float, float, float]:
        w, h = self.page_dims
        left, top, right, bot = self.bbox
        return (left * 1000 / w, top * 1000 / h, right * 1000 / w, bot * 1000 / h)


@dataclass
class PageResult:
    """OCR + digital extraction for a single page under one RenderSpec."""

    page_num: int
    dims: tuple[int, int]
    ocr_words: list[Word] = field(default_factory=list)
    ocr_error: str | None = None
    digital_words: list[Word] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OCR runner — calls AzureProvider in a thread pool.
# ---------------------------------------------------------------------------


def _resolve_ocr_config(workspace_arg: Path | str | None) -> OcrConfig:
    """Resolve an OcrConfig from either an explicit workspace or env vars.

    Resolution order:
    1. ``workspace_arg`` → ``<workspace>/config.json``'s ``ocr`` section
       (handles all auth modes including token-based ``DefaultAzureCredential``)
    2. Auto-resolved workspace (``$DGML_HOME`` or ``./dgml-workspace``) if
       its config has an ``ocr`` section
    3. ``AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT`` + ``…_KEY`` env vars
    """
    # Explicit workspace wins.
    if workspace_arg is not None:
        ws = Workspace.resolve(workspace_arg)
        return load_ocr_config(ws)

    # Auto-resolved workspace: only use it if its config.json actually has
    # an `ocr` section — otherwise fall through to env vars.
    auto_ws = Workspace.resolve()
    if auto_ws.config_path.exists():
        try:
            return load_ocr_config(auto_ws)
        except Exception:  # OcrConfigMissing, OcrConfigInvalid
            pass

    # Env-var fallback.
    endpoint = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if not endpoint:
        raise RuntimeError(
            "no OCR config found: pass --workspace, or set "
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT (+ optionally …_KEY), "
            "or pass --skip-ocr."
        )
    key = os.environ.get("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    return OcrConfig(provider=OcrProviderName.AZURE, endpoint=endpoint, api_key=key)


def run_ocr_on_pages(
    image_paths: list[Path],
    suffix: str,
    config: OcrConfig,
    *,
    max_concurrency: int = 8,
) -> dict[int, tuple[list[Word], str | None]]:
    """Run Azure OCR on each image. Returns ``{page_num: (words, error)}``.

    Errors are caught per-page so one bad page doesn't tank the comparison.
    """
    from dgml_core.ocr_azure import AzureProvider

    provider = AzureProvider(config)

    def _one(path: Path) -> tuple[int, list[Word], str | None]:
        page_num = int(path.stem.split("_", 1)[1])
        try:
            data = path.read_bytes()
            dims = image_dimensions(data, suffix)
            raw_words = provider.analyze_image(data, dims, page_num)
        except Exception as exc:  # OcrFailed, ValueError, etc.
            return page_num, [], f"{type(exc).__name__}: {exc}"
        words = [
            Word(text=w["t"], bbox=tuple(w["l"]), page_dims=dims)  # type: ignore[arg-type]
            for w in raw_words
        ]
        return page_num, words, None

    out: dict[int, tuple[list[Word], str | None]] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
        futures = [executor.submit(_one, p) for p in image_paths]
        for future in as_completed(futures):
            page_num, words, err = future.result()
            out[page_num] = (words, err)
    return out


# ---------------------------------------------------------------------------
# Digital extraction runner.
# ---------------------------------------------------------------------------


def run_digital(
    pdf_path: Path, dpi: int, page_dims: dict[int, tuple[int, int]]
) -> dict[int, list[Word]]:
    """Run pdfminer digital extraction at ``dpi`` and bucket words by page.

    ``page_dims`` provides the rendered image dimensions per page so that
    Word.normalized_bbox() can produce the same 0-1000 space whether
    coordinates came from OCR or digital extraction.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        extract_text_digital(pdf_path, out_dir, file_id="harness", dpi=dpi)

        import json

        result: dict[int, list[Word]] = {}
        for page_json in sorted(out_dir.glob("page_*.json")):
            payload = json.loads(page_json.read_text(encoding="utf-8"))
            page_num = int(payload["page"])
            # Prefer the dims pdfminer computed at this DPI — exactly
            # matches what the per-page JSON would record at runtime.
            dims = (int(payload["width"]), int(payload["height"]))
            page_dims.setdefault(page_num, dims)
            result[page_num] = [
                Word(text=w["t"], bbox=tuple(w["l"]), page_dims=dims)  # type: ignore[arg-type]
                for w in payload["words"]
            ]
        return result


# ---------------------------------------------------------------------------
# Comparison metrics.
# ---------------------------------------------------------------------------


def _word_text_blob(words: list[Word]) -> str:
    """Concatenate word text in reading-order for similarity scoring.

    Reading order = top-then-left, computed in **normalized 0-1000 space**.
    Sorting in raw pixel space is unstable across DPIs: at 300dpi a 0.5pt
    vertical gap rounds to ~2px which can swap neighboring words, but at
    150dpi the same gap rounds to ~1px and may not. Normalizing first
    makes the sort DPI-invariant so the resulting blob captures content
    order rather than pixel-rounding accidents.
    """

    def _key(w: Word) -> tuple[float, float]:
        nl, nt = w.normalized_bbox()[:2]
        return (nt, nl)

    ordered = sorted(words, key=_key)
    return " ".join(w.text for w in ordered)


def _jaccard(a: list[Word], b: list[Word]) -> float:
    """Multiset Jaccard on word text."""
    from collections import Counter

    ca, cb = Counter(w.text for w in a), Counter(w.text for w in b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 1.0


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    al, at, ar, ab = a
    bl, bt, br, bb = b
    il = max(al, bl)
    it = max(at, bt)
    ir = min(ar, br)
    ib = min(ab, bb)
    if ir <= il or ib <= it:
        return 0.0
    inter = (ir - il) * (ib - it)
    area_a = (ar - al) * (ab - at)
    area_b = (br - bl) * (bb - bt)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_words_by_text(a: list[Word], b: list[Word]) -> list[tuple[Word, Word]]:
    """Pair words across two lists by exact text content.

    For each text-content key, pair in order of vertical position. If
    counts differ for a given text, the surplus on either side is dropped
    (uncounted in mean-IoU).
    """
    from collections import defaultdict

    def _by_text(words: list[Word]) -> dict[str, list[Word]]:
        out: dict[str, list[Word]] = defaultdict(list)
        for w in words:
            out[w.text].append(w)
        for ws in out.values():
            ws.sort(key=lambda w: (w.bbox[1], w.bbox[0]))
        return out

    a_by, b_by = _by_text(a), _by_text(b)
    pairs: list[tuple[Word, Word]] = []
    for text in a_by.keys() & b_by.keys():
        for aw, bw in zip(a_by[text], b_by[text], strict=False):
            pairs.append((aw, bw))
    return pairs


def _mean_iou(pairs: list[tuple[Word, Word]]) -> float:
    if not pairs:
        return 0.0
    return sum(_iou(a.normalized_bbox(), b.normalized_bbox()) for a, b in pairs) / len(pairs)


@dataclass
class CompareStats:
    n_words_a: int
    n_words_b: int
    text_similarity: float  # SequenceMatcher ratio of word-blob text
    jaccard: float  # word multiset Jaccard
    matched_word_pairs: int
    mean_iou: float
    # Multiset symmetric difference: words present in A but not B (and vice versa).
    # Top-by-count, capped — meant for at-a-glance "what actually differs."
    only_a: list[tuple[str, int]] = field(default_factory=list)
    only_b: list[tuple[str, int]] = field(default_factory=list)


def _multiset_diff(
    a: list[Word], b: list[Word], cap: int = 8
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Top-``cap`` entries of the multiset symmetric difference, by count."""
    from collections import Counter

    ca, cb = Counter(w.text for w in a), Counter(w.text for w in b)
    only_a = sorted((ca - cb).items(), key=lambda kv: (-kv[1], kv[0]))[:cap]
    only_b = sorted((cb - ca).items(), key=lambda kv: (-kv[1], kv[0]))[:cap]
    return only_a, only_b


def compare(a: list[Word], b: list[Word]) -> CompareStats:
    """Compare two word lists representing the same page under different renderings."""
    text_a, text_b = _word_text_blob(a), _word_text_blob(b)
    similarity = difflib.SequenceMatcher(None, text_a, text_b).ratio() if text_a or text_b else 1.0
    jaccard = _jaccard(a, b)
    pairs = _match_words_by_text(a, b)
    only_a, only_b = _multiset_diff(a, b)
    return CompareStats(
        n_words_a=len(a),
        n_words_b=len(b),
        text_similarity=similarity,
        jaccard=jaccard,
        matched_word_pairs=len(pairs),
        mean_iou=_mean_iou(pairs),
        only_a=only_a,
        only_b=only_b,
    )


# ---------------------------------------------------------------------------
# Per-PDF orchestration.
# ---------------------------------------------------------------------------


@dataclass
class PdfReport:
    pdf_path: Path
    n_pages: int
    # Per-page OCR comparisons (jpeg-300 vs png-150)
    ocr_per_page: dict[int, CompareStats] = field(default_factory=dict)
    ocr_skipped_reason: str | None = None
    # Per-page digital comparisons (extracted at dpi=300 vs dpi=150)
    digital_per_page: dict[int, CompareStats] = field(default_factory=dict)
    # Aggregate
    ocr_summary: CompareStats | None = None
    digital_summary: CompareStats | None = None
    ocr_errors: list[str] = field(default_factory=list)


def process_pdf(
    pdf_path: Path,
    work_dir: Path,
    *,
    baseline: RenderSpec,
    candidate: RenderSpec,
    skip_ocr: bool,
    ocr_config: OcrConfig | None,
    max_concurrency: int,
) -> PdfReport:
    work_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = work_dir / baseline.name
    candidate_dir = work_dir / candidate.name

    print(f"  rendering {baseline.name} → {baseline_dir}")
    baseline_paths = render_with_gs(pdf_path, baseline_dir, baseline)
    print(f"  rendering {candidate.name} → {candidate_dir}")
    candidate_paths = render_with_gs(pdf_path, candidate_dir, candidate)

    if len(baseline_paths) != len(candidate_paths):
        raise RuntimeError(
            f"page count differs between renderings: "
            f"{baseline.name}={len(baseline_paths)} {candidate.name}={len(candidate_paths)}"
        )

    report = PdfReport(pdf_path=pdf_path, n_pages=len(baseline_paths))

    # ---- OCR ----
    baseline_words: dict[int, list[Word]] = {}
    candidate_words: dict[int, list[Word]] = {}
    if skip_ocr:
        report.ocr_skipped_reason = "--skip-ocr"
    elif ocr_config is None:
        report.ocr_skipped_reason = "no OCR config"
    else:
        print(f"  OCR ({baseline.name})…")
        for pn, (words, err) in run_ocr_on_pages(
            baseline_paths, baseline.suffix, ocr_config, max_concurrency=max_concurrency
        ).items():
            baseline_words[pn] = words
            if err:
                report.ocr_errors.append(f"page {pn} {baseline.name}: {err}")
        print(f"  OCR ({candidate.name})…")
        for pn, (words, err) in run_ocr_on_pages(
            candidate_paths, candidate.suffix, ocr_config, max_concurrency=max_concurrency
        ).items():
            candidate_words[pn] = words
            if err:
                report.ocr_errors.append(f"page {pn} {candidate.name}: {err}")

    if baseline_words and candidate_words:
        for pn in sorted(baseline_words.keys() & candidate_words.keys()):
            report.ocr_per_page[pn] = compare(baseline_words[pn], candidate_words[pn])
        all_a = [w for pn in baseline_words for w in baseline_words[pn]]
        all_b = [w for pn in candidate_words for w in candidate_words[pn]]
        report.ocr_summary = compare(all_a, all_b)

    # ---- Digital extraction ----
    print(f"  digital extraction (dpi={baseline.dpi} + dpi={candidate.dpi})…")
    digital_a = run_digital(pdf_path, dpi=baseline.dpi, page_dims={})
    digital_b = run_digital(pdf_path, dpi=candidate.dpi, page_dims={})
    for pn in sorted(digital_a.keys() & digital_b.keys()):
        report.digital_per_page[pn] = compare(digital_a[pn], digital_b[pn])
    all_a = [w for pn in digital_a for w in digital_a[pn]]
    all_b = [w for pn in digital_b for w in digital_b[pn]]
    report.digital_summary = compare(all_a, all_b)
    return report


# ---------------------------------------------------------------------------
# Markdown report.
# ---------------------------------------------------------------------------


def _fmt(stats: CompareStats | None) -> str:
    if stats is None:
        return "—"
    return (
        f"words {stats.n_words_a}/{stats.n_words_b} · "
        f"sim={stats.text_similarity:.3f} · "
        f"jaccard={stats.jaccard:.3f} · "
        f"matched={stats.matched_word_pairs} · "
        f"meanIoU={stats.mean_iou:.3f}"
    )


def _fmt_tokens(items: list[tuple[str, int]]) -> str:
    """Markdown-cell rendering of a top-N token-count list."""
    if not items:
        return "—"
    parts: list[str] = []
    for text, count in items:
        # Escape pipes so they don't break the table cell.
        safe = text.replace("|", "\\|").replace("`", "\\`")
        parts.append(f"`{safe}` x{count}" if count > 1 else f"`{safe}`")
    return ", ".join(parts)


def render_report(
    reports: list[PdfReport],
    *,
    baseline: RenderSpec,
    candidate: RenderSpec,
    skip_ocr: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# {baseline.name} vs {candidate.name} — render comparison")
    lines.append("")
    lines.append(
        f"Each comparison treats `{baseline.name}` as the **baseline** and "
        f"`{candidate.name}` as the **candidate**. Stat shorthand:"
    )
    lines.append("")
    lines.append("- `words A/B` — word counts under baseline / candidate")
    lines.append(
        "- `sim` — `difflib.SequenceMatcher` ratio of reading-order word "
        "blobs (1.0 = identical text content)"
    )
    lines.append("- `jaccard` — multiset Jaccard over word strings")
    lines.append("- `matched` — words paired across renderings by exact text content")
    lines.append(
        "- `meanIoU` — mean bbox IoU over matched pairs, with bboxes "
        "normalized to a DPI-independent 0-1000 space"
    )
    lines.append("")

    # Top-level summary table
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"| PDF | Pages | OCR ({baseline.name} vs {candidate.name}) | "
        f"Digital (dpi={baseline.dpi} vs dpi={candidate.dpi}) |"
    )
    lines.append("|---|---:|---|---|")
    for r in reports:
        ocr_cell = (
            _fmt(r.ocr_summary)
            if r.ocr_summary
            else (f"_skipped: {r.ocr_skipped_reason}_" if r.ocr_skipped_reason else "—")
        )
        lines.append(
            f"| {r.pdf_path.name} | {r.n_pages} | {ocr_cell} | {_fmt(r.digital_summary)} |"
        )
    lines.append("")

    if skip_ocr:
        lines.append(
            "_OCR comparison was skipped via `--skip-ocr`; only digital extraction was compared._"
        )
        lines.append("")

    # Per-PDF detail
    for r in reports:
        lines.append(f"## {r.pdf_path.name}")
        lines.append("")
        lines.append(f"- Path: `{r.pdf_path}`")
        lines.append(f"- Pages: {r.n_pages}")
        if r.ocr_errors:
            lines.append(f"- OCR errors: {len(r.ocr_errors)}")
            for err in r.ocr_errors[:10]:
                lines.append(f"  - {err}")
            if len(r.ocr_errors) > 10:
                lines.append(f"  - …and {len(r.ocr_errors) - 10} more")
        lines.append("")

        if r.ocr_per_page:
            lines.append("### OCR per page")
            lines.append("")
            lines.append(
                f"| Page | {baseline.name} → {candidate.name} | "
                f"only in {baseline.name} | only in {candidate.name} |"
            )
            lines.append("|---:|---|---|---|")
            for pn in sorted(r.ocr_per_page):
                s = r.ocr_per_page[pn]
                lines.append(
                    f"| {pn} | {_fmt(s)} | {_fmt_tokens(s.only_a)} | {_fmt_tokens(s.only_b)} |"
                )
            lines.append("")

        if r.digital_per_page:
            lines.append("### Digital extraction per page")
            lines.append("")
            lines.append(f"| Page | dpi={baseline.dpi} → dpi={candidate.dpi} |")
            lines.append("|---:|---|")
            for pn in sorted(r.digital_per_page):
                lines.append(f"| {pn} | {_fmt(r.digital_per_page[pn])} |")
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two PDF rendering configurations on OCR + extraction."
    )
    parser.add_argument("pdfs", nargs="+", type=Path, help="Input PDFs to compare.")
    parser.add_argument(
        "--baseline",
        type=parse_spec,
        default=JPEG_300,
        help="Baseline render spec as <format>-<dpi> (default: jpeg-300).",
    )
    parser.add_argument(
        "--candidate",
        type=parse_spec,
        default=PNG_150,
        help="Candidate render spec as <format>-<dpi> (default: png-150).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("comparison_report.md"),
        help="Path to write the markdown report (default: ./comparison_report.md).",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Directory to hold per-PDF rendered images. Defaults to a "
            "tempdir that's removed on exit; pass an explicit path to "
            "keep the renderings for inspection."
        ),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help=(
            "Workspace whose config.json holds the OCR section. Defaults "
            "to the same resolution the dgml CLI uses ($DGML_HOME, then "
            "./dgml-workspace). Env vars are used only as a last resort."
        ),
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR comparison; compare digital extraction only.",
    )
    parser.add_argument(
        "--ocr-concurrency",
        type=int,
        default=8,
        help="Max concurrent Azure OCR calls per rendering (default 8).",
    )
    args = parser.parse_args(argv)

    pdfs: list[Path] = [p.resolve() for p in args.pdfs]
    for p in pdfs:
        if not p.exists():
            print(f"error: PDF not found: {p}", file=sys.stderr)
            return 2
        if p.suffix.lower() != ".pdf":
            print(f"error: not a .pdf: {p}", file=sys.stderr)
            return 2

    ocr_config: OcrConfig | None = None
    if not args.skip_ocr:
        try:
            ocr_config = _resolve_ocr_config(args.workspace)
            print(
                f"OCR config: provider={ocr_config.provider.value} endpoint={ocr_config.endpoint}"
            )
        except Exception as exc:
            print(f"warning: could not resolve OCR config: {exc}", file=sys.stderr)
            print("warning: continuing with --skip-ocr semantics.", file=sys.stderr)

    using_tempdir = args.work_dir is None
    work_root = (
        Path(tempfile.mkdtemp(prefix="dgml-render-compare-")) if using_tempdir else args.work_dir
    )
    work_root.mkdir(parents=True, exist_ok=True)

    print(f"work dir: {work_root}")
    print(f"report:   {args.out.resolve()}")
    print()

    reports: list[PdfReport] = []
    try:
        for idx, pdf in enumerate(pdfs, 1):
            print(f"[{idx}/{len(pdfs)}] {pdf.name}")
            subdir = work_root / pdf.stem
            try:
                report = process_pdf(
                    pdf,
                    subdir,
                    baseline=args.baseline,
                    candidate=args.candidate,
                    skip_ocr=args.skip_ocr or ocr_config is None,
                    ocr_config=ocr_config,
                    max_concurrency=args.ocr_concurrency,
                )
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            reports.append(report)
            print()
    finally:
        if using_tempdir:
            shutil.rmtree(work_root, ignore_errors=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        render_report(
            reports,
            baseline=args.baseline,
            candidate=args.candidate,
            skip_ocr=args.skip_ocr or ocr_config is None,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
