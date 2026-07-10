# Converting Word and Excel documents to PDF

DGML's pipeline works on PDFs. To ingest a Word (`.docx`, `.doc`) or Excel
(`.xlsx`, `.xls`) document, DGML first converts it to PDF. `.pdf` inputs
need no conversion and always work.

Conversion is **pluggable**: you choose which converter handles each format
family (`docx`, `xlsx`) by naming it in your workspace config. There is **no
default converter** — until you configure one for a format, that format is
unsupported, and adding or generating such a file fails with an actionable
error telling you what to set.

The `dgml` command ships with **no converters bundled**. Ready-made
converters live in a separate package, **`translators-pdf`**, that you
install on demand. You can also point at your own converter class.

## Installing a converter

The `dgml` / `dgml-core` install by itself only handles `.pdf`. To convert
Office documents you need a converter available in the same environment.

### Provided converters (`translators-pdf`)

If you have the repo checked out locally, `translators-pdf` is a workspace
member; `uv sync` makes it importable, or `uv pip install -e "packages/translators-pdf[xlsx]"`
into the workspace venv.

To install it from PyPI instead:

```bash
pip install translators-pdf          # LibreOffice, Aspose, and command converters
pip install "translators-pdf[xlsx]"  # also the built-in xlsx island renderer
```

The `[xlsx]` extra pulls in `reportlab` and `openpyxl`, needed only by the
island renderer. The other converters have no Python dependencies of their
own — they drive an external tool (see each converter below).

## Configuring a converter

Converters are selected in the `conversion` section of your workspace's
`config.json` (`<workspace>/config.json`), keyed by format family. Each
family names a `provider` — a dotted path `"module.path:ClassName"` — plus
any options that provider accepts.

```jsonc
{ "conversion": {
    "docx": { "provider": "translators_pdf.libreoffice:LibreOfficeConverter" },
    "xlsx": { "provider": "translators_pdf.xlsx:XlsxIslandsConverter", "row_gap": 4 }
}}
```

Only families you configure are supported. A family you omit stays
unsupported — there is no implicit fallback to some default tool.

## Provided converters

All of the converters below are referenced by their dotted path in the
`provider` field, exactly like a converter you write yourself.

### LibreOffice — `translators_pdf.libreoffice:LibreOfficeConverter`

Handles **both** `docx` and `xlsx` by driving a headless `soffice`
subprocess. No Python dependencies and no config fields — but LibreOffice
must be installed and `soffice` must be on your `PATH` (or at a standard
install location). It resolves `soffice` cross-platform and uses a
per-process lock directory so concurrent conversions are safe.

```jsonc
{ "conversion": {
    "docx": { "provider": "translators_pdf.libreoffice:LibreOfficeConverter" },
    "xlsx": { "provider": "translators_pdf.libreoffice:LibreOfficeConverter" }
} }
```

### Aspose — `AsposeWordsConverter` / `AsposeCellsConverter`

High-fidelity commercial converters. `AsposeWordsConverter` handles `docx`
(`.docx`, `.doc`); `AsposeCellsConverter` handles `xlsx` (`.xlsx`, `.xls`).

The Aspose SDKs are **not** installed by `translators-pdf` — install them
yourself:

```bash
pip install aspose-words          # for AsposeWordsConverter
pip install aspose-cells-python   # for AsposeCellsConverter
```

Without a license, Aspose runs in watermarked evaluation mode. Point the
optional `license` field at an Aspose license file to remove the watermark
(one Aspose.Total license typically covers both products). If the SDK is
missing, the converter raises an actionable error telling you what to
install.

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

### Xlsx island renderer — `translators_pdf.xlsx:XlsxIslandsConverter`

Handles `xlsx` with no external tool. It detects table "islands" in each
sheet and renders them to PDF. Requires the `[xlsx]` extra — from a local
repo checkout, `uv pip install -e "packages/translators-pdf[xlsx]"`; from PyPI,
`pip install "translators-pdf[xlsx]"`. If `reportlab`/`openpyxl` are missing
it raises `"pip install translators-pdf[xlsx]"`.

Config fields (all optional):

- `row_gap` / `col_gap` — max empty rows/columns tolerated within one island
  (default `2`).
- `orientation` — `"landscape"` (default) or `"portrait"`.

Tall islands split across pages. It is tuned for tidy tables: a very wide or
dense sheet can produce an island too large to fit a page, which fails with a
clear error rather than a bad PDF.

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

### Command — `translators_pdf.command:CommandConverter`

A generic escape hatch for driving any converter binary you already have,
with no Python code. It works for any family you register it under. Config
carries a `command` **argv list** (not a shell string, so there is no quoting
or injection to worry about) plus an optional `timeout` in seconds
(default `180`).

Two substitution tokens control where the output lands:

- `{input}` — the source file path.
- `{output}` — the exact PDF path the tool should write (e.g. `ssconvert`,
  `unoconv`).
- `{output_dir}` — a directory the tool writes into, naming the file itself
  (e.g. LibreOffice); the converter then discovers the produced PDF.

It verifies the output PDF exists and is non-empty rather than trusting exit
codes (LibreOffice, for one, exits 0 even on failure).

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

## Using your own converter

You don't need the repo source to supply a converter — the `dgml` wheel
resolves your class at runtime from its dotted path.

1. Install the library that defines the base class into your venv:
   `pip install dgml-core` (or `pip install dgml`, which pulls it in).
2. Write a class subclassing `DocConverter`:

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
           ...   # lazy-import your SDK; raise an actionable error if missing

       def to_pdf(self, path: Path) -> bytes:
           ...
   ```

3. Make it importable by the same interpreter running `dgml`. Cleanest is
   your own small `pip install`-ed package in that venv; the quick path is a
   loose `.py` file on `PYTHONPATH` or the current working directory. For the
   quick path, save the class as e.g. `my_converters.py` and make Python find
   it — either run `dgml` from the directory containing that file, or point
   `PYTHONPATH` at its directory:

   ```bash
   export PYTHONPATH="/path/to/dir/with/my_converters:$PYTHONPATH"
   ```

   Then reference it by its module name (the filename without `.py`), e.g.
   `"provider": "my_converters:MyConverter"`.
4. Point the format family's `provider` at the dotted path:

   ```jsonc
   { "conversion": { "xlsx": { "provider": "my_pkg.converters:MyConverter", "some_option": 4 } } }
   ```

**Note on trust:** a dotted `provider` path (and a `command` argv) runs
arbitrary code from your config, as you. This is the same trust model as a
user-installed LibreOffice or Aspose — keep your `config.json` under your own
control.

## How conversion fits the workflow

Once a converter is configured for a family, that format flows through the
workspace like a PDF:

- **Adding files.** `dgml` accepts the configured source formats alongside
  `.pdf`. At add time the source is converted once, the resulting PDF is
  stored next to the original, and page images are rendered from it. Adding a
  file whose format has no configured converter is rejected with the "no
  converter configured" message.
- **Generating.** Generation reuses the PDF produced at add time, so there is
  no second conversion and no drift. (Files added before conversion was
  stored fall back to converting on demand.)

The converted PDF is a derived artifact, like page images. `dgml check` and
file attestation key off the stored original document, not the converted PDF.

## Troubleshooting

- **"no converter configured for .xlsx …"** — add a `conversion.xlsx`
  (or `.docx`) entry to `config.json` pointing at a converter. This appears
  both when adding a file and when generating.
- **"pip install translators-pdf[xlsx]"** — the island renderer's
  dependencies are missing; install the extra.
- **Aspose SDK error** — `pip install aspose-words` or
  `aspose-cells-python` into the venv running `dgml`.
- **LibreOffice not found** — install LibreOffice and ensure `soffice` is on
  `PATH`.
- **A `provider` string with no `:`** — providers must be a full dotted path
  `"module:ClassName"`; a bare name is rejected with a hint toward
  `translators-pdf`.
- **Module/attribute/subclass errors at convert time** — the provider path
  is resolved lazily when a conversion actually runs. Confirm the module is
  installed in (or importable by) the interpreter running `dgml`, the class
  name is correct, and it subclasses `DocConverter`.
