# Pluggable document-to-PDF converters

## Context

Converting a source document to PDF is pluggable: users choose the converter **per format family** (`docx`, `xlsx`) by pointing at a converter class via config. Conversion quality becomes the user's choice and budget.

Two structural decisions shape the whole design:

- **Core ships no converters and assumes no defaults.** The `dgml` package provides only the `DocConverter` abstraction + the resolution/dispatch machinery. With no converter configured for a format, that format is simply unsupported; `.pdf` always works because it needs no converter.
- **All concrete converters live outside the `dgml` wheel** — including the ones we provide (LibreOffice, Aspose.Words, the xlsx island-renderer). They ship in a separate, separately-installable workspace package so the core wheel stays dependency-light and license-pure.

The abstraction follows the same shape as the `OcrProvider` ABC + `config.json` section design; the packaging follows the `dgml-clustering` workspace-member pattern.

**Decisions locked in:**

- BYO mechanism: **dotted-path string in config** (`"my_pkg.mod:ClassName"`), placed directly in the family's `provider` field. The provided converters are referenced the *same* way — there is no separate "built-in" code path.
- **No default behavior.** A converter must be explicitly configured for a format family before that format is accepted.
- Provided converters live in a separate workspace package, **`translators-pdf`** (import `translators_pdf`), installed on demand (`pip install translators-pdf[xlsx]` for the renderer's deps).
- Selection: `conversion` section in `<workspace>/config.json`, keyed by format family, mirroring `ocr`.
- Friendly short names via entry points: **deferred** (see Future Extensions) — dotted paths are the v1 mechanism.

## Licensing guardrail (non-negotiable)

Keeping converters out of the core wheel is partly a license-hygiene decision:

- The **`dgml` wheel has zero converter dependencies** → the `pip-licenses` audit is trivially clean, with nothing for a PDF-space copyleft dep to sneak in through.
- **`translators-pdf`** is still part of the Apache-2.0 repo, so the same allow-list applies to *its* `pyproject.toml`. `reportlab` (BSD) + `openpyxl` (MIT) are acceptable direct deps there, gated behind the `xlsx` extra.
- **`aspose-words` is proprietary/commercial and must not appear as a declared dependency anywhere** — not in `dgml`, not in `translators-pdf`, not as an extra. The Aspose.Words converter lazy-imports `aspose.words` inside `__init__` and raises an actionable error (`"pip install aspose-words and provide a license"`) if absent — exactly how LibreOffice/ghostscript are treated as external, user-installed tools.

## Design

### Core: `packages/dgml/src/dgml/conversion.py`

Ships the abstraction and the resolver only — no concrete converters.

- `@dataclass(frozen=True) ConverterConfig` — `provider: str` plus optional provider-specific fields (e.g. xlsx `row_gap`, `col_gap`, `orientation`; command `command`, `timeout`).
- `class DocConverter(ABC)`:

  ```python
  name: ClassVar[str]                       # informational, e.g. "libreoffice"
  input_formats: ClassVar[frozenset[str]]   # {".docx", ".doc"} / {".xlsx", ".xls"}
  config_fields: ClassVar[frozenset[str]]
  @classmethod
  @abstractmethod
  def parse_config(cls, section: dict[str, Any]) -> ConverterConfig: ...
  @abstractmethod
  def __init__(self, config: ConverterConfig) -> None: ...   # lazy SDK/binary import here
  @abstractmethod
  def to_pdf(self, path: Path) -> bytes: ...
  ```

- Provide a `_check_no_extra_fields` classmethod that rejects unknown keys in a provider's config section (typo / stale-field guard).
- Resolution is dotted-path only: a `provider` (`"module.path:ClassName"`) is imported via `importlib.import_module` + `getattr` and checked to be a `DocConverter` subclass, else a clear `ConversionConfigInvalid`. A string without a `:` is rejected with a hint pointing at `translators-pdf`. There is **no `_BUILTIN` registry of concrete classes** in core — provided and third-party converters resolve identically. (The hook for unioning in entry-point-registered short names later goes here; see Future Extensions.)
- `make_converter(config: ConverterConfig) -> DocConverter`: resolves `config.provider` to its class and instantiates it (where the lazy SDK/binary import fires). Called at convert time, not config-load time.
- `load_conversion_config(workspace) -> dict[str, ConverterConfig]`: reads the `conversion` section of `config.json` (via `read_json` / `Workspace.config_path`), keyed by format family (`"docx"`, `"xlsx"`). Validates only the **generic shape** (each family is an object with a non-empty string `provider`) and keeps the section verbatim; it does **not** import the provider class or run `parse_config`, so loading the config never imports the converter package. Provider resolution and field validation are deferred to `make_converter` (convert time) — many callers load the config without ever converting (e.g. `docset generate` over files that already have their PDFs, the add-time suffix gate), and shouldn't pay for an import they don't use. Returns **only the families explicitly configured** — absent families are absent from the dict (a missing file/section yields `{}`). Raises `ConversionConfigInvalid` only for a bad *shape* (non-object section/family, or missing/blank `provider`); a bad provider path or unknown provider-specific field surfaces later, from `make_converter`.
- **No defaults, no defaulting warning.** There is no implicit fallback: an unconfigured family means the format is unsupported, surfaced at dispatch (below), rather than silently converted by some default tool.
- The "tool actually missing" case is **not** handled here — it surfaces lazily at converter construction (`__init__`): LibreOffice-not-on-PATH, `pip install translators-pdf[xlsx]`, or aspose-not-licensed each raise an actionable error.

### Public-API commitment (required for BYO and for `translators-pdf`)

`CLAUDE.md` says anything not exported from the top-level `__init__.py` is internal and "may change without notice pre-1.0." Both `translators-pdf` and third-party BYO converters subclass `DocConverter`, so `from dgml_core.conversion import DocConverter, ConverterConfig` must be a **stable** surface. So: **export `DocConverter` and `ConverterConfig` from `dgml_core/__init__.py` and treat them (and the `_check_no_extra_fields` contract) as supported API**, with the versioning discipline that implies. Our own provided converters consume the exact same public surface as a third party would — the sign the abstraction is right.

### Provided converters: new workspace package `packages/translators-pdf/`

Distribution `translators-pdf`, import package `translators_pdf` (a distinct top-level package, mirroring how `dgml-clustering` imports as `clustering`). It is **not** a `dgml_core.` submodule: `dgml-core` ships an `__init__.py` (a regular package), so a separate distribution can't cleanly extend the `dgml_core` namespace via PEP 420 — hence the separate top-level name. Depends on `dgml-core` (for the ABC) as a normal workspace dependency.

Modules (one converter each):

- **`translators_pdf/libreoffice.py`** — `LibreOfficeConverter`, handling docx **and** xlsx (a headless `soffice` subprocess converts both). Resolves the `soffice` binary cross-platform and uses a per-process `-env:UserInstallation=…` lock dir for concurrency safety. No installable dependency (subprocess only), but `soffice` must be on PATH (or at a standard install location). No config fields. `name = "libreoffice"`. Example — one provider for both families:

  ```jsonc
  { "conversion": {
      "docx": { "provider": "translators_pdf.libreoffice:LibreOfficeConverter" },
      "xlsx": { "provider": "translators_pdf.libreoffice:LibreOfficeConverter" }
  } }
  ```

- **`translators_pdf/aspose.py`** — two Aspose converters sharing a private `_AsposeConverter` base (config parsing, license handling, and temp-dir save scaffolding live there; each concrete class only names its SDK and how to open/save): `AsposeWordsConverter` for docx (`input_formats = {".docx", ".doc"}`, lazy-imports `aspose.words`, `name = "aspose-words"`) and `AsposeCellsConverter` for xlsx (`input_formats = {".xlsx", ".xls"}`, lazy-imports `aspose.cells`, `name = "aspose-cells"`). Each raises the actionable error if its SDK is missing. **No dependency declared anywhere** (`pip install aspose-words` / `pip install aspose-cells-python` yourself). Optional `license` field points at an Aspose license file (without it, Aspose runs in watermarked evaluation mode; one Aspose.Total license typically covers both products). Example:

  ```jsonc
  { "conversion": {
      "docx": {
        "provider": "translators_pdf.aspose:AsposeWordsConverter",
        "license": "/path/to/Aspose.Total.lic"   // optional
      },
      "xlsx": {
        "provider": "translators_pdf.aspose:AsposeCellsConverter",
        "license": "/path/to/Aspose.Total.lic"   // optional
      }
  } }
  ```

- **`translators_pdf/xlsx.py`** — `XlsxIslandsConverter` for xlsx: detects table "islands" in each sheet and renders them to a PDF, returning bytes. Lazy-imports `openpyxl`/`reportlab` in `__init__`; raises `"pip install translators-pdf[xlsx]"` if missing. `name = "xlsx-islands"`. Config fields (all optional): `row_gap` / `col_gap` (max empty rows/cols tolerated within one island, default `2`), `orientation` (`"landscape"` default or `"portrait"`). Tall islands split across pages, but it's tuned for tidy tables — a very wide or dense sheet can produce an island too large to fit a page, which fails with a clear error rather than a valid PDF. Example:

  ```jsonc
  { "conversion": {
      "xlsx": {
        "provider": "translators_pdf.xlsx:XlsxIslandsConverter",
        "row_gap": 4,
        "col_gap": 2,
        "orientation": "portrait"
      }
  } }
  ```

- **`translators_pdf/command.py`** — `CommandConverter`, the generic CLI escape hatch for users who have a converter binary and want zero Python. `name = "command"`. Config carries an **argv list** (not a shell string → no injection/quoting), plus an optional `timeout` (seconds, default `180`). Example — Gnumeric's `ssconvert` for xlsx, plus LibreOffice for docx via the output-dir form:

  ```jsonc
  { "conversion": {
      "xlsx": {
        "provider": "translators_pdf.command:CommandConverter",
        "command": ["ssconvert", "{input}", "{output}"]
      },
      "docx": {
        "provider": "translators_pdf.command:CommandConverter",
        "command": ["soffice","--headless","--convert-to","pdf","--outdir","{output_dir}","{input}"],
        "timeout": 300
      }
  } }
  ```

  `to_pdf(path)` substitutes the tokens, runs the argv with the shared timeout, then reads the produced PDF back as bytes. Two contracts it owns so users don't have to:
  - **Output-control split:** `{output}` → converter writes that exact file (ssconvert, unoconv); `{output_dir}` → converter names it itself (LibreOffice) and `to_pdf` discovers the single produced PDF.
  - **Verify output exists + is non-empty — don't trust exit codes** (LibreOffice exits 0 on failure).
  - Format-agnostic: accepts whatever family key it's registered under rather than declaring a fixed `input_formats`.

`packages/translators-pdf/pyproject.toml`:

```toml
[project]
dependencies = ["dgml"]                       # workspace source ref

[project.optional-dependencies]
xlsx = ["reportlab>=4.0", "openpyxl>=3.1"]    # BSD + MIT, only for the xlsx renderer
```

### Using a converter (provided or third-party — same mechanism)

The provided converters are referenced by dotted path, identical to a user's own:

```jsonc
{ "conversion": {
    "docx": { "provider": "translators_pdf.libreoffice:LibreOfficeConverter" },
    "xlsx": { "provider": "translators_pdf.xlsx:XlsxIslandsConverter", "row_gap": 4 }
}}
```

### BYO converter — what the end-user does

No repo clone, no venv from source; the dgml wheel resolves the class at runtime:

1. `pip install dgml-core` into their own venv (the library that defines the ABC; `pip install dgml` pulls it in transitively too). (To use a *provided* converter instead of writing one, `pip install translators-pdf` — or `translators-pdf[xlsx]` for the renderer.)
2. Write a class subclassing the public `DocConverter`:

   ```python
   from dgml_core.conversion import DocConverter, ConverterConfig
   from pathlib import Path
   from typing import Any, ClassVar

   class MyConverter(DocConverter):
       name: ClassVar[str] = "my-converter"
       input_formats: ClassVar[frozenset[str]] = frozenset({".xlsx"})
       config_fields: ClassVar[frozenset[str]] = frozenset({"some_option"})

       @classmethod
       def parse_config(cls, section: dict[str, Any]) -> ConverterConfig:
           cls._check_no_extra_fields(section)
           return ConverterConfig(provider=section["provider"], ...)

       def __init__(self, config: ConverterConfig) -> None:
           ...   # lazy-import their SDK; actionable error if missing

       def to_pdf(self, path: Path) -> bytes:
           ...
   ```

3. Make it importable by the same interpreter running dgml — cleanest is their own small `pip install`-ed package in that venv; quick path is a loose `.py` on `PYTHONPATH`/CWD.
4. Point the format family's `provider` at the dotted path:

   ```jsonc
   { "conversion": { "xlsx": { "provider": "my_pkg.converters:MyConverter", "some_option": 4 } } }
   ```

5. `make_converter` sees the `":"`, does `importlib.import_module` + `getattr`, asserts it's a `DocConverter` subclass, calls `parse_config`, instantiates, and runs `to_pdf`.

Resolution error envelopes (mirror the OCR error types): module not importable (`ImportError` → "installed in this venv / on PYTHONPATH?"), attribute missing, not a `DocConverter` subclass, `parse_config` raises. `make_converter` should sanity-check `input_formats` against the family key rather than require a match.

**Trust note:** a dotted path (and a `command` argv) is arbitrary code execution by config — acceptable because it's the user's own config running as themselves (same trust model as user-installed LibreOffice/Aspose), but a deliberate, documented choice.

### Wire into the single dispatch point

`load_document_as_pdf` in `generation/document.py` gains a required resolved-converters argument:

```python
def load_document_as_pdf(
    path: Path, *, converters: dict[str, ConverterConfig]
) -> bytes:
```

There is always a workspace, so `converters` is always resolved from `load_conversion_config(workspace)` and passed in.

- `.pdf` → unchanged (`load_pdf`), always supported, needs no converter.
- `.docx`/`.doc` → look up the `docx` family in `converters`; if present, `make_converter(...).to_pdf(path)`.
- `.xlsx`/`.xls` → look up the `xlsx` family.
- **Family not configured → raise a clear, actionable error**, e.g. `"no converter configured for .xlsx; set conversion.xlsx.provider in config.json (see translators-pdf for ready-made converters)"`. No default, no fallback.

### Thread config from the CLI

The CLI commands that own the workspace (`docset generate`, and the planner path) resolve `load_conversion_config(workspace)` once and pass the resulting dict down to the call sites: `pipeline.py:260`, `pipeline.py:421`, and `planner.py:87` *(re-verify these line numbers against the current tree before implementing — they drift)*. Plumb it via `ConvertOptions` (preferred — already flows through pipeline) so the signature churn stays in one struct.

### Ingestion gates (so docx/xlsx actually flow through a workspace)

To let Excel/Word into a workspace, add-time ingestion accepts source formats alongside `.pdf`:

- `files.py:_validate_pdf` — accept the configured converters' input formats (`.docx`/`.doc`/`.xlsx`/`.xls`) alongside `.pdf`. The accepted set is config-driven: whatever the workspace's `conversion` config covers. An unconfigured format is rejected at add-time with the "no converter configured" message.
- `cli.py:_gather_pdfs` — also collect those extensions.
- At **add-time**, convert source → PDF with the resolved converter, **persist** the PDF alongside the stored original at `files/<file_id>/<stem>.pdf`, and render page images from it via ghostscript. The document is converted exactly once: generation reuses the persisted PDF (`load_document_as_pdf` checks for the sibling `<stem>.pdf` before converting), so the bytes the page images were rendered from are byte-identical to those generation slices — no add-vs-generate drift. Files added before conversions were persisted (no sibling PDF) fall back to on-demand conversion. The rest of the pipeline stays PDF-only.

  The persisted PDF is a derived artifact (like `page_images/`); `dgml check` and file-attestation key off the stored original (`original_filename`) and ignore it. Teaching them to treat a missing converted PDF as regenerable, and to attest it, is a possible follow-up.

This widens the CLI's accepted inputs, so per `packages/dgml/CLAUDE.md` the four files move together: `cli.py`, `tests/test_cli.py`, `docs/cli-reference.md`, and the dgml `SKILL.md`.

## Future extensions (not now)

**Entry-point short names.** Add a `dgml.converters` entry-point group so an installed converter package can register friendly names, letting config say `"provider": "libreoffice"` instead of a full dotted path. `make_converter` would union `importlib.metadata.entry_points(group="dgml.converters")` into a name→class map and try it before falling back to dotted-path resolution. This is now *more* compelling than when first deferred (provided converters are external, so a registry of names is exactly what entry points give for free), but it's pure sugar over the dotted-path mechanism — ship dotted paths first, layer names on without reshaping the ABC.

## Files to create / modify

**Create:**
- Core abstraction: `packages/dgml/src/dgml/conversion.py`; export from `packages/dgml/src/dgml/__init__.py`.
- New package `packages/translators-pdf/` (`pyproject.toml`, `py.typed`, `src/translators_pdf/{libreoffice,aspose,xlsx,command}.py + _xlsx_detector.py/_xlsx_renderer.py` + ported xlsx detector/renderer), per the "Adding a new package" steps in root `CLAUDE.md`.
- Tests: `packages/dgml/tests/test_conversion.py` (core resolver) and `packages/translators-pdf/tests/` (each provided converter).
- This doc: `docs/conversion.md`.

**Modify:** `generation/document.py` (dispatch), `generation/pipeline.py` + `generation/planner.py` (thread config), root `pyproject.toml` + `uv.lock` (register the new workspace member), and for add-time ingestion: `files.py`, `cli.py`, `tests/test_cli.py`, `docs/cli-reference.md`, `.claude/skills/dgml/SKILL.md`.

## Verification

- `scripts/verify.sh` — ruff, format, mypy `--strict`, pytest, license audit. The audit covers the whole workspace including `translators-pdf`; it must stay green, proving `aspose-words` is in no `pyproject.toml`.
- Core unit tests in `test_conversion.py`:
  - dotted-path resolves a test stub converter; unknown/garbage provider raises a clear error; non-`DocConverter` class raises.
  - `load_conversion_config` returns only configured families and validates shape only (missing/blank `provider` raises); it does **not** import the provider — a config naming an absent module still loads, and resolution/field errors surface from `make_converter` at convert time. An unconfigured format raises the "no converter configured" error at dispatch.
- `translators-pdf` tests: each provided converter raises the actionable error when its tool/extra is absent; argv substitution for the `command` provider covers both `{output}` and `{output_dir}` discovery and surfaces non-zero/empty-output failures.
- End-to-end (manual, needs LibreOffice): `uv sync`, `pip install -e packages/translators-pdf[xlsx]` into the venv, configure `conversion.docx`/`conversion.xlsx` to the provided converters, and convert a sample `.docx` and `.xlsx` through `load_document_as_pdf` (and via `dgml docset generate` if ingestion is wired), confirming PDF bytes / page count.
- Confirm that with **no `conversion` config**, adding/generating a `.docx` fails with the actionable "no converter configured" error and a `.pdf` still works.
