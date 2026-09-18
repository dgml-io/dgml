# `dgml-core` package

The **library** behind the `dgml` CLI. Distribution name `dgml-core`, import
name `dgml_core`. Everything that turns a PDF into DGML lives here: the
PDF→DGML generation pipeline, OCR, page-image rendering, digital/OCR/hybrid
text extraction, grounding, classification, attestation, the LLM client, and
workspace/storage CRUD.

The `dgml` package (the CLI) depends on this one and is the only first-party
caller; `translators-pdf` also depends on it (for the `DocConverter` ABC in
[src/dgml_core/conversion.py](src/dgml_core/conversion.py)). Nothing here may
import `dgml` (the CLI) — the dependency is strictly one-way.

## Public API

The supported library surface is what
[src/dgml_core/__init__.py](src/dgml_core/__init__.py) exports (e.g.
`Workspace`, `FileStore`, `DocSetStore`, the error hierarchy, attestation
helpers). Anything not exported is internal and may change without notice
pre-1.0. Consumers `import dgml_core`; `from dgml import …` is intentionally
unsupported (the CLI package re-exports nothing).

## Optional extras

`aws`, `azure`, `macos`, `pdfium`, `clustering`, and `chain` are declared
here. The `dgml` CLI mirrors them as pass-throughs, so `pip install dgml[aws]`
resolves to `dgml-core[aws]`. Keep the two extra lists in sync when you add or
rename one.

## OCR providers

`--text-mode ocr` dispatches through an `OcrProvider` ABC defined in
[src/dgml_core/ocr.py](src/dgml_core/ocr.py). Concrete providers live in sibling
modules — `src/dgml_core/ocr_aws.py`, `src/dgml_core/ocr_azure.py`,
`src/dgml_core/ocr_macos.py` — and register themselves via the `_PROVIDERS`
dict at the bottom of `ocr.py`.

Each provider owns three things: its SDK lazy-import (in `__init__`),
its config-section validation (`parse_config` classmethod), and its
per-page API call (`analyze_image`). The shared loop in
`extract_text_ocr` handles filesystem I/O and result aggregation —
providers never touch the disk.

To add a new provider: see the "Adding a new provider" section in the
[src/dgml_core/ocr.py](src/dgml_core/ocr.py) module docstring.

## PDF engines

PDF work follows the same shape as OCR. One **engine** supplies both
capabilities DGML needs — rasterizing pages (`PageRenderer`) and slicing a
page range (`PdfSlicer`) — and one `[pdf] provider` key selects it, because
the motivating use case ("no system binary") is only satisfied when both
avoid one.

The two ABCs, the config loader (`load_pdf_config`) and the `EngineSpec`
registry live in [src/dgml_core/pages.py](src/dgml_core/pages.py). Each
engine's two implementations live together in one sibling module so they share
a single availability probe: `src/dgml_core/pages_ghostscript.py` (the
default, a subprocess over the system `gs` binary) and
`src/dgml_core/pages_pypdfium2.py` (PDFium in-process, the `pdfium` extra).

The shared wrappers own everything engine-independent: `render_pages` handles
the `$DGML_PAGE_CACHE` cache, stale-image cleanup and page counting;
`slice_pages` bounds-checks page numbers against the real page count so a bad
request never reaches a backend.

**Renderer geometry is a contract, not a preference.** A page's PNG must be
exactly `round(pts * dpi / 72)` pixels per axis, measured from the
**MediaBox** and after `/Rotate` — that is the space `page_text/` boxes and
every `dg:origin` live in, and nothing at runtime checks that the two agree.
A backend whose natural output differs must correct for it (PDFium needs both
corrections; see `Pypdfium2Renderer._render_page`).

To add a new engine: see the "Adding a new engine" section in the
[src/dgml_core/pages.py](src/dgml_core/pages.py) module docstring.
