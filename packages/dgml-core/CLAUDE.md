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

Two entry points look alike and are not: `Workspace.open(...)` resolves a
workspace **and** brings it up to date (config migration, seal check,
initialized check, layout migration) and is what anything reading or writing
a workspace calls; `Workspace.resolve(...)` only answers "which workspace" and
is for the handful of operations that run before one exists. Creating one is
`create_workspace(...)`, not a `Workspace` constructor.

## Logging

Library code logs through `logger = logging.getLogger(__name__)`, so every
record lands under `dgml_core.*`. Never `print`, never write to `sys.stderr`,
never use `warnings.warn` for runtime events, and never configure handlers,
levels or formats. The caller routes. `__init__.py` attaches a `NullHandler`
to `dgml_core`, so a caller that configures nothing sees nothing. The `dgml`
CLI is one such caller (`_configure_logging` in `cli.py`). Levels:

- **WARNING**: the user must act, or the output is degraded (a tier fallback,
  an unreachable model, a missing OCR provider). The CLI shows these by default.
- **INFO**: what `dgml --verbose` shows (hybrid merge decisions, per-page
  failures, workspace-upgrade notices).
- **DEBUG**: detail beyond `--verbose`. No CLI switch maps to it (`DGML_DEBUG=1`
  is just an env-var alias for `--verbose`); library callers opt in with
  `logging.getLogger("dgml_core").setLevel(DEBUG)`.

A warning that a per-file or per-page loop would repeat is deduped through
module-level `_WARNED_*` state (`models_config.py`, `ocr.py`, `rotation.py`),
keyed by whatever makes a recurrence informative — a `(tier, fallback)` pair, a
workspace root, or nothing (a bool) when the condition is process-global. Tests
reset that state in an autouse fixture.

Structured events a caller may act on belong in return values or a typed
callback (`on_migration`), not in log text. `debug=` controls telemetry and
intermediate files, never log output. Tests assert with pytest's `caplog`, not
`capsys`.

## Configuration

A `Workspace` always has exactly one `Configuration` (`ws.config`,
[src/dgml_core/configuration.py](src/dgml_core/configuration.py)), and every
loader (`load_grounded_config`, `load_ocr_config`, …) reads its sections through
`load_merged_config`. It is built one of two ways:

- **In memory**, by an application: `Configuration.build(identity=…, storage=…,
  models=Models(…), …)` with typed sections and credentials **by value**, then
  `Workspace.open(configuration=cfg)`. No `config.toml`, no user config, no
  `DGML_*` environment, no store of workspaces, nothing written back; the first
  open writes the workspace's meta document to the docstore and later opens check
  it names the same workspace. `root` is optional when every store is remote (an empty temp dir stands in); the
  local store, whose data it is, cannot be opened without one.
- **From TOML**, for a workspace addressed by path or id: derived on access from
  the usual merge (user config → workspace config → environment) plus the
  `[workspace]` and `[storage.<svc>]` tables — the only path that writes.

The typed section classes render to exactly the keys their loader reads, so
validation and the `*_CONFIG_INVALID` codes stay in the loaders; add a key to a
loader and to its section class together. Readers that the derivation itself
uses (`read_identity`, `resolve_store_configs`, `load_merged_config`) branch on
the *supplied* `ws.configuration`; everything downstream reads `ws.config` or
goes through those. Library code never calls `user_config_path()` outside that
TOML derivation. The derived form is not memoized (a file-backed config is
re-read on every access, as `config_text` has always been), and deriving it
costs several reads of `config.toml`, which is why the loaders take the TOML
merge directly rather than through `ws.config`. See
[docs/library-configuration.md](../../docs/library-configuration.md).

## Optional extras

`aws`, `azure`, `macos`, `pdfium`, `clustering`, and `chain` are declared
here. The `dgml` CLI mirrors them as pass-throughs, so `pip install dgml[aws]`
resolves to `dgml-core[aws]`. Keep the two extra lists in sync when you add or
rename one.

## OCR providers

`--text-mode ocr` dispatches through an `OcrProvider` ABC defined in
[src/dgml_core/ocr.py](src/dgml_core/ocr.py). Like `[conversion]` and
`[storage]`, `ocr.provider` is a dotted `"module.path:ClassName"` resolved at
use time via `dgml_core.provider.import_provider_class` — **there is no registry
of privileged classes.** The providers DGML bundles
(`src/dgml_core/ocr_aws.py`, `src/dgml_core/ocr_azure.py`,
`src/dgml_core/ocr_macos.py`) are named by exactly the same kind of path a third
party's would be; `BUILTIN_OCR_PROVIDERS` maps the short names `aws` / `azure` /
`macos` onto them so existing configs keep working.

Each provider owns three things: its SDK lazy-import (in `__init__`),
its config-option validation (`parse_config` classmethod, which receives the
`[ocr]` table minus `provider` as `config.options`), and its per-page API call
(`analyze_image`). The shared loop in `extract_text_ocr` handles filesystem I/O
and result aggregation — providers never touch the disk.

Unlike `load_conversion_config`, `load_ocr_config` resolves the provider class
and runs its validation **eagerly**: `file add` validates OCR config before it
touches the filesystem, so a bad `[ocr]` table is rejected with no record
created. A workspace names exactly one OCR provider, so there is no fan-out cost.

To write a new provider: see the "Writing your own provider" section in the
[src/dgml_core/ocr.py](src/dgml_core/ocr.py) module docstring, and
[docs/ocr-providers.md](../../docs/ocr-providers.md).

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
