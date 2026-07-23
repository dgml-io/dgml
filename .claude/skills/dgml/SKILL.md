---
name: dgml
description: Use when the user wants to manage a DGML workspace from the shell — initialize a workspace, add PDFs or Office documents (docx/xlsx, when a converter is configured) single or bulk from a directory, create or manage DocSets, assign files to DocSets, convert PDFs to DGML XML (grounded in place with dg:origin bounding boxes as part of generation), extract structured values against a schema (RNC schema generation, grounded value extraction to DGML), or run a consistency check. Triggers on phrases like "add these PDFs", "ingest these Word/Excel docs", "create a docset", "import documents into DGML", "ingest a folder of PDFs", "ground the DGML", "add bounding boxes", "extract values", "extraction schema", "generate a schema", "pull fields from these docs", "dgml workspace", "check the workspace".
---

# Using the `dgml` CLI

The `dgml` CLI manages a workspace of **Files** (PDFs copied in, hashed, page-rendered) and **DocSets** (named groupings of Files). It is JSON-default and non-interactive — built to be scripted.

Authoritative references in this repo:
- [docs/cli-reference.md](../../../docs/cli-reference.md) — full command surface
- [docs/storage-layout.md](../../../docs/storage-layout.md) — on-disk format
- [packages/dgml/src/dgml/cli.py](../../../packages/dgml/src/dgml/cli.py) — implementation

When in doubt about a flag or response shape, read those first.

## Invocation

From inside this repo, use `uv run dgml …` (the venv is managed by `uv sync`). In an environment where the `dgml` distribution is installed normally, plain `dgml …` works. Examples below use `uv run dgml`; substitute as appropriate.

## Workspace resolution

The workspace root is picked in this order:

1. `--workspace <path>` flag
2. `$DGML_HOME`
3. `./dgml-workspace` (default, relative to cwd)

Setup — the minimum is a **single** command:

1. `dgml workspace create [path] --organization <org>` — creates the workspace (`docsets/` + `files/`), records its identity in `workspace.json`, and (if no shared `local_config.json` exists yet) seeds one from the bundled template and copies it to `config.json`. The optional positional `path` is where to create it (`dgml workspace create ./ws …`); omit it to use the resolved root (global `--workspace` → `$DGML_HOME` → `./dgml-workspace`). `--organization` is **required** and is embedded in this workspace's docset namespace URIs (`http://dgml.io/<org>/<DocSetSlug>`); pass an optional `--name` for a human-readable label (defaults to the workspace directory name). When it seeds the config, the response carries `local_config_created: true` and a `next_action` telling you to edit the models/OCR endpoint (there are **no** default models — an unconfigured model is a hard error, never a silent paid call).
2. `dgml init` (optional, run first) — only when you want to review/edit the shared `local_config.json` **before** creating any workspace. It creates the peer `local_config.json` from the template and nothing else.

Configure once, create many: sibling workspaces reuse the same `local_config.json`. After editing it, re-run `dgml workspace create --organization <org> --force` to re-sync it into an existing workspace. Any command other than `init` / `workspace create` on an uninitialized workspace fails with `WORKSPACE_NOT_INITIALIZED`.

## Output contract — parse it with `jq`

- stdout: JSON success payload
- stderr: JSON error envelope `{ "error": { "code": "...", "message": "..." } }`
- exit codes: `0` success, `1` error, `2` `check` ran but found issues

Capture IDs from JSON with `jq`, never by string-munging text output:

```bash
ds=$(uv run dgml docset create --name "Q2 contracts" | jq -r .id)
fid=$(uv run dgml file add /tmp/contract.pdf | jq -r .file.id)
```

Use `--format text` only when a human asked for readable output directly; do not parse it.

## Common workflows

### Add a single PDF and assign it to a DocSet

```bash
uv run dgml workspace create --organization Acme              # once: create workspace (seeds config too)
ds=$(uv run dgml docset create --name "Q2 contracts" | jq -r .id)
fid=$(uv run dgml file add /path/to/contract.pdf --on-conflict skip | jq -r .file.id)
uv run dgml docset add-file "$fid" --docset "$ds"
```

### One-shot: add a PDF and let an LLM pick (or create) its DocSet

When the user has the `classification` section configured in
`<workspace>/config.json`, `--auto-classify` replaces the three-step
add → choose-docset → assign dance with a single call. The CLI runs a vision LLM against the
file's rendered page images, then either calls `docset add-file` against
the best-fitting existing DocSet or creates a new DocSet and assigns
the file to it. Prefer this when bulk-importing heterogeneous documents
where the right DocSet isn't known upfront.

DocSets are **document-type-specific**, not topical buckets. The
classifier's rubric is schema-shareability: a new file is grouped with
an existing DocSet only when the same extraction schema would fit
both. So a property-tax bill and a tax-abatement (PILOT) agreement
both concern property taxes, but yield different DocSets — and that's
correct. Each new DocSet carries a list of `key_questions` (proposed
by the LLM at creation time, persisted in `docset.json`) describing
what the first pages of that document type can answer; future
classifications match against those questions, not against
topical similarity.

```bash
uv run dgml workspace create --organization Acme   # once: create workspace (seeds config too)
payload=$(uv run dgml file add /path/to/doc.pdf --auto-classify)
ds=$(jq -r .classification.docset_id <<<"$payload")
qs=$(jq -r '.classification.docset_key_questions | join(" | ")' <<<"$payload")
err=$(jq -r '.classification.error // empty' <<<"$payload")
if [ -n "$err" ]; then
  echo "auto-classify failed: $err" >&2   # file is still added; just not assigned
else
  echo "assigned to $ds — questions: $qs"
fi
```

When seeding a workspace with hand-curated DocSets that will then
absorb files via `--auto-classify`, pass `--key-question` to anchor
the type:

```bash
uv run dgml docset create --name "Lease Abstract" \
  --description "One-page summary of commercial lease terms" \
  --key-question "Who is the tenant?" \
  --key-question "What is the lease term?" \
  --key-question "What is the monthly base rent?"
```

Key contract points:
- A missing or invalid `classification` config is a **hard** error
  (exit 1, `CLASSIFICATION_CONFIG_MISSING` / `_INVALID`): config is a
  precondition, so the command aborts rather than recording the same
  error on every file. For a bulk directory add it's checked once up
  front, before any file is added.
- Once a valid config is in hand, failures of the classification *call*
  itself (LLM/network/auth error) are **soft**: the file is still added
  and the failure lands in `classification.error` with exit 0.
- On `--on-conflict skip` of a duplicate, classification is **skipped**
  (`classification.performed: false`), making bulk loops idempotent
  without burning extra LLM calls.
- Config schema and the full payload shape are in
  [docs/cli-reference.md](../../../docs/cli-reference.md) under
  "Auto-classification".

### Bulk: add every PDF in a directory to one DocSet

The canonical pattern. Pass the **directory** to `file add` and it ingests
every `*.pdf` (case-insensitive) under it in a single run — one subprocess,
one config load — and returns a single envelope with a `summary` count block
and a per-item `results` array (each entry carries a `status`: `added` /
`skipped` / `soft_failed` / `hard_failed`). `--on-conflict skip` makes re-runs
safe (returns the existing record on a hash- or path-collision); `docset
add-file` is also idempotent. `--recursive` walks subdirectories (default: top
level only).

```bash
DIR=/path/to/pdfs
uv run dgml workspace create --organization Acme

ds=$(uv run dgml docset create --name "Imported PDFs" | jq -r .id)

# One call adds every PDF under $DIR. --on-conflict skip → safe to re-run.
payload=$(uv run dgml file add "$DIR" --on-conflict skip)

# The command exits 0 even when individual files soft- or hard-fail. Surface
# those from the payload so they aren't buried (don't just trust the exit code).
jq -r '.results[]
  | select(.status == "soft_failed" or .status == "hard_failed")
  | "  [\(.status)] \(.path) " +
      (.error.message // .text_extraction_error // .page_render_error // .page_count_error)' \
  <<<"$payload"

# Assign every successfully-added file (entries with a `.file` record) to the DocSet.
for fid in $(jq -r '.results[] | select(.file) | .file.id' <<<"$payload"); do
  uv run dgml docset add-file "$fid" --docset "$ds"
done

uv run dgml check    # authoritative health signal afterward
```

The `summary` block (`{total, added, skipped, soft_failed, hard_failed}`,
summing to `total`) is the quick health read; report it to the user.

Variants:
- **Add to an existing DocSet:** skip the `docset create` step; pass its known ID as `$ds`. Find it with `uv run dgml docset list | jq -r '.docsets[] | select(.name=="…") | .id'`.
- **Auto-route heterogeneous PDFs into DocSets**: drop the `docset create` step and the `docset add-file` loop; pass `--auto-classify` to `file add` instead. Each file lands in the best-fitting existing DocSet, or in a new one the LLM proposes — and DocSets created mid-run are visible to later files in the same batch, so similar PDFs cluster. Requires `classification` config in `<workspace>/config.json`; see the one-shot example above. Read each file's `.results[].classification` block for the outcome.
- **Recurse into subdirectories:** add `--recursive`.
- **Hidden errors:** a PDF that fails to parse, render, or extract digital text still produces an entry — `soft_failed` (the `page_*`/`text_extraction_error` fields are set on its `file` entry) or `hard_failed` (the entry has an `error` object and no `file`). `dgml check` afterward is the authoritative whole-workspace health signal.

**Fallback — one `file add` per PDF.** Only if you're on a build of `dgml`
without the directory form of `file add` (it predates this feature), drop
back to the shell loop. `-print0` / `read -d ''` handles odd filenames;
`-maxdepth 1` makes it non-recursive.

```bash
find "$DIR" -type f -iname '*.pdf' -print0 |
  while IFS= read -r -d '' pdf; do
    payload=$(uv run dgml file add "$pdf" --on-conflict skip)
    fid=$(jq -r .file.id <<<"$payload")
    for field in page_render_error page_count_error text_extraction_error; do
      msg=$(jq -r ".$field // empty" <<<"$payload")
      [ -n "$msg" ] && echo "  [$(basename "$pdf")] $field: $msg"
    done
    uv run dgml docset add-file "$fid" --docset "$ds"
  done
```

### Office documents (docx/xlsx)

`file add` accepts not just `.pdf` but convertible sources — `.docx`/`.doc`/`.xlsx`/`.xls` — **when a converter is configured** for that format family in `<workspace>/config.json`. The source is converted to PDF at add time, then ingested exactly like a PDF (same `page_images/`, `page_text/`, generation). With no converter configured for a non-PDF format, the add fails with `UNSUPPORTED_FILE_TYPE` — there is no default. Ready-made converters ship in the `translators-pdf` package (LibreOffice, Aspose.Words, an xlsx island-renderer, a generic command runner); or point a family at your own class. Minimal config:

```jsonc
// <workspace>/config.json
{ "conversion": {
    "docx": { "provider": "translators_pdf.libreoffice:LibreOfficeConverter" },
    "xlsx": { "provider": "translators_pdf.xlsx:XlsxIslandsConverter" }
} }
```

Install the providers you reference: `pip install translators-pdf` (LibreOffice needs `soffice` on PATH; `pip install translators-pdf[xlsx]` for the xlsx renderer). A convertible source that can't be converted (missing binary/SDK) is a *soft* fail — the File record is created with `conversion_error` set, like `page_render_error`. See [conversion.md](../../../docs/conversion.md).

### Text extraction

`file add` extracts digital text at ingest time (default `--text-mode digital`, via `pdfminer.six`) and writes one compact JSON per page to `<workspace>/files/<file_id>/page_text/page_N.json`. Word locations are integer `[left, top, right, bottom]` pixel boxes that align with the matching `page_images/page_N.png`.

A PDF with no extractable digital text (e.g. a scan) still gets a File record — the failure is *soft*. The response payload has `text_extraction_error` set, a permanent error is recorded, and `dgml check` reports `text_extraction_failed_permanent` until `--retry-errors` clears it. Always look at `text_extraction_error` after `file add`, the same way you look at `page_render_error`.

`--text-mode ocr` runs the cloud provider configured in `<workspace>/config.json` (Azure Document Intelligence or AWS Textract). Requires the corresponding extra installed: `pip install dgml[azure]` or `pip install dgml[aws]`. Without a config the command fails with `OCR_CONFIG_MISSING` before any record is created. See [storage-layout.md](../../../docs/storage-layout.md) for the config schema. Secrets live in env vars, never in `config.json`.

`--text-mode hybrid` runs digital extraction and OCR per page, then merges them by grouping words that cover the same region into overlap clusters (boxes overlap on IoU > 0.5 *or* one box mostly contained in the other, so split/merge tokenization resolves as a unit). Each cluster is resolved as a whole: OCR-only clusters are kept; digital-only clusters are assumed invisible to the human eye (hidden form layers, white-on-white, off-page text) and **dropped**; mixed clusters compare the two sides' concatenated text by dash-normalized Levenshtein distance — if they agree (distance ≤ 2) **digital wins** (its characters come straight from the PDF font, which is more reliable than OCR even when OCR's tokenization is finer); if they disagree OCR wins as the authority on what's visible. A page whose digital text is mostly unresolved glyphs (pdfminer `(cid:N)` sentinels) falls back to OCR entirely. Default is silent — add the global `--verbose` flag (`uv run dgml --verbose file add … --text-mode hybrid`) to surface per-page warnings and the merge summary on stderr. Needs the same `ocr` workspace config as `--text-mode ocr` (validated up front — missing config returns `OCR_CONFIG_MISSING` before any record is created). Use it when neither pure digital nor pure OCR is good enough on its own (e.g. PDFs with embedded text plus scanned form fields). Optionally, an LLM can make the per-cluster merge decision instead of the Levenshtein heuristic — add a `text_extraction` section to `config.json` (e.g. a local Ollama model); see [storage-layout.md](../../../docs/storage-layout.md). It's opt-in, and any LLM failure falls back to the heuristic per page.

### OCR setup recipe

```bash
# One-time per workspace: write the provider config.
cat > "$DGML_HOME/config.json" <<'EOF'
{
  "ocr": {
    "provider": "azure",
    "endpoint": "https://my-resource.cognitiveservices.azure.com/",
    "api_key_env": "AZURE_DOCINTEL_KEY"
  }
}
EOF

# Once per shell session: export the secret.
export AZURE_DOCINTEL_KEY="..."

# Add files using OCR.
uv run dgml file add /path/to/scan.pdf --text-mode ocr
```

### Inspect state

```bash
uv run dgml status                       # workspace + counts
uv run dgml docset list                  # all docsets
uv run dgml docset show <docset_id>      # one docset
uv run dgml docset list-files <docset_id>
uv run dgml file list
uv run dgml file show <file_id>
```

### Convert a DocSet's PDFs into DGML XML

`dgml docset generate <docset_id>` runs the typed-block PDF→DGML
pipeline over every file in the docset: each window is transcribed into a
flat list of typed JSON blocks (`generation.model`), then ONE batch-wide
semantic-labeling call assigns concept tags across all of the docset's
documents at once (`generation.label_model`), and the result is rendered
deterministically into namespaced `dg:chunk` XML. The labeling vocabulary
(the "roster") is planned automatically from the documents, or pinned up
front with `--schema-path` (see below). Unseeded runs are staged: the largest
documents label first (a pilot) and their observed evidence — verbatim
examples, kinds, hierarchy — confirms the vocabulary the rest of the batch
labels against. There is no separate transform pass. The pipeline is part of
the base `dgml` install and reuses the workspace's pre-rendered `page_images/`.

**Choose the models — config only, no flags.** The models are not CLI flags:
`generate` reads them solely from the `generation` section of
`<workspace>/config.json`, so each is one explicit, visible choice per
workspace (matching every other model-consuming command). Both are **required**:
`model` (per-page transcription) and `label_model` (the
single batch-wide labeling call — a stronger model here is cheap). Without a
`generation` section, `generate` fails with `GENERATION_CONFIG_MISSING`. See
the `generation` config in [storage-layout.md](../../../docs/storage-layout.md).

Grounding runs in place as part of `generate`, adding `dg:origin` boxes and —
when observable in the source — `dg:style` (inline CSS for bold/italic/size/
color/uppercase). For `--text-mode digital`/`hybrid` the style facts come
deterministically from the PDF glyphs (free, no LLM). For `--text-mode ocr`
there are no font facts, so `dg:style` is empty unless the workspace opts
into the image-based path by adding a `style` section to
`config.json` (`{"style": {"model": "<vision model>"}}` — its presence is the
switch), which has that model read the page images to infer the same styles
**plus `text-align`** (off by default; honored only for OCR files). See
[storage-layout.md](../../../docs/storage-layout.md).

**Bulk-import a folder of PDFs then convert them.** The typical workflow
when the user has a directory of PDFs they want as DGML — set the model once
in config, add all files to one docset, then generate (no per-run flags):

```bash
DIR=/path/to/pdfs
uv run dgml workspace create --organization Acme

# `workspace create` already seeds config.json (including generation.model /
# label_model) from the shared local_config.json template. Adjust the models to
# taste — a cheaper model for the bulk transcription, a stronger one for the
# single batch-wide labeling call:
cat > "${DGML_HOME:-./dgml-workspace}/config.json" <<'EOF'
{
  "generation": {
    "model": "anthropic/claude-haiku-4-5",
    "label_model": "anthropic/claude-sonnet-4-6",
    "api_key_env": "ANTHROPIC_API_KEY"
  }
}
EOF

ds=$(uv run dgml docset create --name "Imported PDFs" | jq -r .id)

payload=$(uv run dgml file add "$DIR" --on-conflict skip)
for fid in $(jq -r '.results[] | select(.file) | .file.id' <<<"$payload"); do
  uv run dgml docset add-file "$fid" --docset "$ds"
done

# Models come from config.json — there are no --model/--label-model flags.
uv run dgml docset generate "$ds"
```

**Pin the vocabulary for consistent labels (`--schema-path`).** Labeling is
non-deterministic run-to-run; to lock the concept vocabulary, pass a schema a
prior run exported — `schema.json` (Schema v1: a `tags` map of concept name →
`{role, kind, parent_role, …}`) or its RELAX NG Compact render `full-schema.rnc`
(both land at the docset root; the `.rnc` is the human-friendly editing
surface and reverses losslessly). The planning pass is skipped and that
vocabulary is used as-is with full fidelity — role descriptions, curated
examples, and kind all feed the labeling prompt, and the tag hierarchy
(`parent_role`) also seeds entity-container grouping — and per-document
labeling still extends it for roles it doesn't cover. Only these exported
formats are accepted (not a flat `{concept: description}` mapping). The
natural loop is "generate once, review/curate the schema, then reuse it":

```bash
# 1) first run plans the vocabulary and exports it to docsets/<id>/schema.json
#    (+ full-schema.rnc, the same schema as commented RELAX NG Compact)
uv run dgml docset generate "$ds"
# 2) reuse (optionally hand-curate) either export on later runs
uv run dgml docset generate "$ds" \
  --schema-path "$DGML_HOME/docsets/$ds/full-schema.rnc"
```

Output always goes to the docset directory in the workspace — there is
no output-directory flag. Each file's DGML lands at
`<workspace>/docsets/<docset-id>/files/<file-id>/<stem>.dgml.xml` (the
per-(docset, file) directory). This deterministic placement is what file
attestation keys on. Read the per-file output path from each `results[].output`
in the JSON payload rather than reconstructing it.

**`--debug` keeps debug artifacts and logs LLM usage.** A generate run always
writes the small functional `cache/` files the next run reloads
(`*_blocks.json`, `label_*_cNN_raw.json`, `concept_roster.json`). By default it
skips the debug-only cache artifacts (raw LLM responses,
`*.concept.xml`/`*.semantic.xml` renders, prompt listings) and
`coverage_report.json`, so the workspace stays clean — and it writes **no**
`usage.jsonl` cost/token rows. Pass the global `--debug` flag
(`uv run dgml --debug docset generate "$ds"`) to retain those artifacts and to
record per-operation cost telemetry to `<workspace>/usage.jsonl` (this gating
applies to every LLM command — classify, cluster, generate). Coverage summaries
still print on stderr under `--verbose` regardless.

**Parallelism.** `--max-parallel-calls` transcribes that many documents
concurrently (windows *within* a document stay serial, since each window
sees the previous window's tail). The calls are network-bound, so threads
overlap the latency. Raise it on high-RPM paid tiers; set `1` to serialize
if you hit 429s.

**Grounding is built in.** As the last step, generation grounds each
`<stem>.dgml.xml` *in place* against the file's `page_text/` OCR — adding
a `dg:origin` bounding-box attribute (`<page> <x1> <y1> <x2> <y2>`,
space-separated, in integer image pixels, matching `page_images/page_N.png`) to
every element whose subtree grounded: elements with text content get one
box per visual line, and pure containers (sections, lists, tables, rows,
the document root) get one union box per page. It's deterministic (no LLM, no config),
so there's no separate ground step or `.grounded.xml` file — the
canonical `<stem>.dgml.xml` already carries page positions. A file with
no `page_text/` is written but left ungrounded (the `outputs[]` entry has
`grounded: false` with a `grounding_error`); files that ground report
`grounded: true` with `matched_token_pct` and `elements_annotated`
(a count that includes annotated containers). Pass
`--debug` to also write a per-file `<stem>.dgml.grounding_stats.json`
sidecar (match rates, largest ungrounded snippets) — `matched_token_pct`
below the high 90s usually means generation dropped or paraphrased text.

**Labeling failures are visible without `--verbose`.** A misconfigured
`generation.label_model` is caught two ways. Before any transcription spend, a
pre-flight check fails the whole run fast — `AUTH_ERROR` if the model's provider
key is absent (unless `api_base` is set), `GENERATION_CONFIG_INVALID` for a
malformed model string. A failure that slips past it at runtime (transient
network error, or a well-formed but nonexistent model id) doesn't discard the
transcription: the file still converts (`status: converted`, exit 0, unlabeled)
and its `results[]` entry carries a `label_error: {code:
"LABEL_MODEL_UNREACHABLE", message}`. So after a `generate`, check for it rather
than assuming labeled output:

```bash
uv run dgml docset generate "$ds" \
  | jq '.results[] | select(.label_error) | {source, label_error}'
```

**Resume on rerun.** If a file's per-(docset, file) `<stem>.dgml.xml`
already holds a generated document tree, that file is skipped — re-invoking
the same command after a crash only re-processes unfinished documents. When
*every* file is already converted the command exits 0 with no LLM call made.
An extraction-only file (extract ran before generate) is NOT skipped:
generate builds its tree and carries the existing `dg:extraction` over
(`full-extraction`); re-renders preserve it too.

**Growing a docset (add docs later, stay consistent).** Because existing files
are skipped, adding a document and re-running generates only the new one — and
by default it's labeled seeded with the docset's own `schema.json` (full
fidelity: descriptions, observed examples, kind, hierarchy; falls back to the
flat `cache/concept_roster.json`), so its tags stay consistent with the rest
(no `--schema-path` needed). Every concept is emitted in the `docset:` vocabulary
namespace (`dg:` is framework-only), so growing the docset never flips a tag's
prefix; an already-generated file is still re-rendered deterministically when
its output otherwise changes as the docset's schema/roster grows (reported under
`rerendered` / `summary.rerendered`; no re-LLM). Pass `--no-roster` to label the
new docs in isolation instead.

```bash
uv run dgml docset add-file "$ds" "$new_fid"
uv run dgml docset generate "$ds"   # only the new file; reuses the docset schema
```

**Output.** Like every other `dgml` command, `docset generate` emits a
single JSON object on stdout — pipe it straight to `jq`. Pass 1/2/4
progress lines go to stderr and only under `--verbose`. The payload is the
shared batch envelope: a `summary` count block (`{total, converted, skipped,
failed}`) plus a per-item `results` array, each entry carrying a
`status` (`converted` / `skipped` / `failed`). A top-level `rerendered` lists
already-generated files re-rendered because the docset namespacing shifted. A file whose source has gone
missing is a `failed` entry (with an `error` object) rather than a run-level
abort — the batch finishes and exits 0, so check `summary.failed` and surface
any `failed` results. Each `failed` entry's `error.message` carries a short,
single-line cause — including a transcription failure from an LLM/provider
error that survived retries — so you can report *why* without re-running under
`--verbose` (the full error still goes to stderr there). The all-skipped resume case emits the same shape with
`summary.converted == 0`. See the
[cli-reference.md](../../../docs/cli-reference.md) entry under
`dgml docset generate` for the full payload shape and flag table.

### Grounding (bounding boxes) — built into `generate`

There is **no** separate `dgml docset ground` command. Generation grounds
each `<stem>.dgml.xml` in place with `dg:origin` boxes (see "Grounding is
built in" above), so a single `dgml docset generate "$ds"` produces
grounded DGML. Inspect grounding quality straight from its payload:

```bash
uv run dgml docset generate "$ds" \
  | jq '.results[] | {source, status, grounded, matched_token_pct}'
```

To re-run *just* the grounding pass on already-generated XML without
paying to regenerate (e.g. after a grounding change), use the
maintenance script — it grounds in place and is not part of the public
CLI (pass `--debug` to also write the per-file
`<stem>.dgml.grounding_stats.json` sidecar):

```bash
uv run python scripts/ground.py --docset "$ds" [--file "$fid"] [--debug]
```

### Extract structured values (schema-driven)

Where `docset generate` transcribes a *whole* document, **extraction** pulls a
defined set of fields and grounds each value to the page. It needs a `grounded`
section in `<workspace>/config.json` (`schema_model`, `values_model`, optional
API-key env vars) — the LLM is configurable like every other model-using
command, with per-call `--schema-model` / `--values-model` overrides.

The schema is **RELAX NG Compact** (`extraction-schema.rnc`) at rest, matching the DGML
spec; extracted values are written as a **`dg:extraction` element inside the
file's core `<stem>.dgml.xml`** (spec §13) — there is no separate values file.
The workflow is generate-schema → extract → get-values:

```bash
# 1) Propose a schema from sample PDFs (stored as extraction-schema.rnc). --from-file is
#    repeatable; omit it to sample every file in the docset. The model picks an
#    XSD datatype per leaf, so the generated RNC has typed leaves natively
#    (xsd:date, xsd:decimal, xsd:integer, …) — dates/amounts/counts come back as
#    typed dg:value at extraction, not bare text.
uv run dgml extraction generate-schema "$ds" --from-file "$fid"

#    …or set one yourself. set-schema accepts RNC *or* a grounded-field JSON
#    Schema and converts JSON to RNC on the way in (RNC is the only on-disk form).
#    A hand-written .rnc may carry `## Prompt:` lines telling the LLM where to
#    find each field.
uv run dgml extraction set-schema "$ds" --schema-file schema.rnc

# 2) Extract values for a file → writes a dg:extraction element into the file's
#    <stem>.dgml.xml. `mode` is full-extraction if a generated tree already
#    exists, else extraction. Order doesn't matter: a later `docset generate`
#    builds the tree and carries the dg:extraction over (full-extraction).
#    NOTE: once the schema is set, assigning a file (docset add-file /
#    --auto-classify / cluster into an existing docset) auto-extracts it —
#    check the payload's `extraction` block; run `extract` manually only for
#    files assigned before the schema existed or to re-extract.
uv run dgml extraction extract "$ds" "$fid" | jq '{mode, tool_calls, field_count, xml_path}'

# 3) Read them back. Default is values-shape JSON (projected from dg:extraction);
#    --as xml returns the whole core DGML document.
uv run dgml extraction get-values "$ds" "$fid" \
  | jq '.values | to_entries[] | {tag: .key, text: .value.text, value: .value.value}'
uv run dgml extraction get-values "$ds" "$fid" --as xml | jq -r .xml
```

Inspect or convert the stored schema with `get-schema` (`--schema-format rnc`
default, or `json` for the engine's grounded-field projection). Typed values
carry a normalized `dg:value`/`xsi:type` (dates, amounts, integers) and every
value carries a `dg:origin` box back to `page_images/page_N.png`.

A schema field's `## Prompt:` may describe a **derivation rule** instead of a
place on the page ("Compute as sum of Quantity × UnitPrice for each LineItem").
Such fields come back *computed* (spec §13): in values JSON as
`{text, value, computed: true, derived_from: [dotted paths]}` with no
`locations`; in the XML as `dg:origin="computed"` + `dg:value` +
`dg:itemprop="computedFrom"`/`dg:href` pointing at the source elements (which
get `xml:id`s). When writing a derivation `## Prompt:`, make every input it
mentions an extracted field of its own — `dg:href` can only point at extracted
elements, so an un-extracted input leaves the computed value unverifiable
(unresolvable refs are counted as `matching.dropped_refs` in
`extraction_stats.json`, and `dgml check` flags source-less computed fields
as `computed_field_unattributed`). Errors are the
usual envelope: `SCHEMA_NOT_FOUND` (no schema set), `NO_FILES` (empty docset,
no `--from-file`), `VALUES_NOT_FOUND` (extract not run yet),
`GROUNDED_CONFIG_MISSING` (no `grounded` config). See
[cli-reference.md](../../../docs/cli-reference.md) for full payloads.

### Semantic links — built into `generate`

The final step of `generate` adds semantic **links** — relationships the tree's
nesting can't capture — to each grounded `<stem>.dgml.xml`, in place. It writes
`dg:itemprop` (predicate) + `dg:href` (`#id`, or space-separated `#id`s) on the
subject and assigns `xml:id`s to both ends. Covers references (`references`,
`incorporates`, `signatoryOf`, …), relative dates (`relativeTo`/`effectiveOn`
with an ISO-8601 offset in `dg:value`), and derived values (`greaterOf`/
`lesserOf` formulas, `escalates`, `valueFrom`). The model proposes links on the
labeling model (`generation.label_model`), then a skeptical pass verifies them.
Each converted file's `results` entry carries a `links` count.

```bash
uv run dgml docset generate "$ds" | jq '.results[] | {source, links}'
uv run dgml docset generate "$ds" --no-semlinks   # skip the link step
```

### Cluster the unassigned files

`dgml cluster` is the bulk counterpart to `dgml file add --auto-classify`:
it groups every currently-unassigned file into a cluster and assigns each
file to a DocSet — either an existing one whose name matches the cluster
label, or a freshly-created DocSet whose name and description come from
the vision LLM. **It has side effects on the workspace** (creates DocSets,
adds file assignments).

Requires `pip install dgml[clustering]` (pulls in the `dgml-clustering`
ML stack). Without it the command exits 1 with `MISSING_EXTRA`. The same
setup as `--auto-classify` is needed whenever any cluster requires a new
DocSet: a `classification` section in `<workspace>/config.json` and
`pip install dgml[classification]`. **Partial success is the contract** —
once the command runs, exit code is always 0; per-cluster failures
(missing config, LLM error) land in `failed_file_ids`. Always check that
field before reporting success to the user.

Pass `--skip-existing` to make a resume/re-run cheap: if every file is
already assigned to a DocSet, the clusterer never runs and the payload
comes back with `skipped: true`. On any real run (or when at least one file
is still unassigned) `skipped` is `false`. The field is always present.

```bash
payload=$(uv run dgml cluster --skip-existing)
if [ "$(jq -r .skipped <<<"$payload")" = "true" ]; then
  echo "nothing to cluster — every file is already assigned"
fi
failed=$(jq -r '.failed_file_ids | length' <<<"$payload")
if [ "$failed" -gt 0 ]; then
  echo "$failed file(s) couldn't be auto-clustered — see .failed_file_ids" >&2
fi
```

Algorithm settings (encoders, fusion, manifold, training, scenario) default
to the bundled config. To tune them, either add a `clustering` section to
`<workspace>/config.json` (persistent, applies to every run) or pass a
one-off file with `--config PATH`. The `--config` file uses the same field
schema and **replaces** the workspace section for that run — reach for it to
A/B configs or drive a GPU encoder on a single run without editing
`config.json`. A bad path exits 1 with `CLUSTERING_CONFIG_INVALID`.

```bash
# Try a GPU-encoder config on one run without touching config.json
uv run dgml cluster --config ./clustering_gpu.json
```

`--method` selects *how* documents are grouped, orthogonal to `--mode`
(default `embedding`, the statistical pipeline). For a **very small corpus** —
a handful of documents, where tf-idf/neighbor statistics have too little signal
and clusters collapse into one bucket — use `--method llm`: it sends every
document's first pages to the vision LLM in one call and lets it partition *and*
name the groups (no embedding step, `--config` ignored). It needs the same
`classification` config as `--auto-classify` (missing config ⇒ every file in
`failed_file_ids`) and caps one call at 24 files. Prefer `--method auto` when
the corpus size is unknown: it routes ≤ `--small-corpus-threshold` files
(default 8) to the LLM and larger corpora to the embedding pipeline.

```bash
# Let DGML pick the engine by corpus size (LLM for tiny folders, embedding otherwise)
uv run dgml cluster --method auto
```

Sample payload:

```json
{
  "clusters": {"k7q3xb91pmrf": "Contracts"},
  "failed_file_ids": [],
  "skipped": false
}
```

`clusters` is a `{file_id: docset_name}` map. The value is the actual
DocSet name the file ended up in — either an existing DocSet's name
(when the algorithm matched it to one) or the LLM-proposed name for a
newly-created DocSet (when the algorithm couldn't match, the LLM was
asked to name a new one, bundling multiple files per cluster into a
single call). The only time a placeholder like `"unknown_0"` shows up
here is for a file that also appears in `failed_file_ids`.

`failed_file_ids` covers two distinct cases: files whose `page_1.png` is
missing (page render failed at ingest — can't be embedded) and files
whose cluster needed LLM naming but that call failed. Both surface in
the same channel.

### Recover from issues

```bash
uv run dgml check                  # report inconsistencies (exit 2 if any)
uv run dgml check --retry-errors   # clear permanent-error markers and retry
```

`check` covers: missing/corrupt metadata, missing PDFs, hash mismatches, unreadable PDFs, page-count mismatches, render failures, and dangling DocSet → File references. Permanent errors (e.g. corrupt PDF, ghostscript failure) are recorded so they don't get retried on every run; `--retry-errors` clears that.

### Export a file as a DGMLX bundle

A **DGMLX bundle** is the Merkle-attested, portable export of everything
DGML knows about a file: the artifacts (source → page images → page text →
optionally a DocSet's `full-schema.rnc` / `extraction-schema.rnc` and the
file's `<stem>.dgml.xml`), a
single `META-INF/dgml-attestation.xml`, and the OPC parts
(`[Content_Types].xml` + `_rels/.rels`), all zipped into a portable
`<stem>.dgmlx` archive. The two output modes are mutually exclusive: **by
default `dgml dgmlx export` writes only that `.dgmlx` archive** to
`--output-dir` (payload field `dgmlx`); **`--unpacked` writes only the loose
bundle tree there and no archive** (payload field `attestation`).

The attestation file is both the manifest and the provenance record: it
carries the Merkle root, the `<artifacts>` inventory (relative paths +
per-page `number` attributes), the workspace identity (`file-id`, plus
`docset-id` when `--docset` was given), and the rendering provenance from
`file.json` (`page-image-dpi`, `page-image-renderer`, and `pdf-converter`
for a converted non-PDF source). The bundle is **filename-independent**:
ordering comes from the `number` attributes, not the on-disk names. The
attestation file is *not* itself a leaf of the root; its loose path is in
the payload's `attestation` field **only with `--unpacked`**. `_rels/.rels`
names the `source/` original as the main document, the `<stem>.dgml.xml` as
dgml-xml when present, and the attestation file as the attestation.

`dgml dgmlx verify` takes either the `.dgmlx` archive or an unpacked
directory, re-hashes the artifacts in inventory order, and compares against
the recorded root.

```bash
# Export the docset-scoped DGMLX bundle (include schema + dgml.xml).
# Drop --docset to export only the file-side artifacts (PDF/images/text).
payload=$(uv run dgml dgmlx export "$fid" --output-dir ./bundle --docset "$ds")
root=$(jq -r .root <<<"$payload")
pkg=$(jq -r .dgmlx <<<"$payload")   # ./bundle/<stem>.dgmlx (the only output by default)
echo "exported $fid → $root ($pkg)"

# Re-verify anywhere (no workspace needed) — pass the .dgmlx directly.
uv run dgml dgmlx verify "$pkg" | jq '{valid, expected_root, computed_root}'

# Want the loose tree instead of the archive? --unpacked writes the loose
# files into ./loose (and no .dgmlx); verify reads the directory.
uv run dgml dgmlx export "$fid" --output-dir ./loose --docset "$ds" --unpacked
uv run dgml dgmlx verify ./loose | jq '{valid, expected_root, computed_root}'
```

Exit codes on `verify` mirror `check`: `0` when the bundle verifies,
`2` when it verifies-but-fails (a tampered artifact → `valid: false`,
roots differ — surface this, don't just trust exit 0), and `1`
(`ATTESTATION_INVALID`) when the bundle is structurally broken (missing or
malformed `META-INF/dgml-attestation.xml`, a referenced artifact missing
from disk, a bad/duplicate page `number`). Full payload shapes and the
attestation-file format are in
[cli-reference.md](../../../docs/cli-reference.md) under "DGMLX commands".

### Attest a single DGML XML element (node-level)

Where DGMLX attests a file's whole artifact set, `dgml node export`
attests one element of the generated DGML XML: it emits the node's
hash, the document tree's Merkle root, the RFC 6962 inclusion proof
connecting them, the element's canonical XPath, and the node's
canonical XML. `--docset` is required (the XML is docset-scoped);
select the element with exactly one of `--leaf <n>` (0-based pre-order
Merkle leaf index), `--xpath <expr>` (must match exactly one element —
the UX tree view's "Copy XPath" gives a canonical one), or
`--child-path <path>` (slash-separated 0-based child-element indices
from the root, e.g. `1/1` — useful when a caller only has a DOM
position, such as a browser tree view's `Element.children` path, and
no ready-made XPath or leaf index).

```bash
# Export the attestation payload for one element.
uv run dgml node export "$fid" --docset "$ds" \
  --xpath '/dg:chunk/docset:Entry[2]/docset:Amount' > node-proof.json
jq '{node_hash, root_hash, leaf_index}' node-proof.json

# Equivalent, addressed by DOM child-index path instead of XPath.
uv run dgml node export "$fid" --docset "$ds" --child-path '1/1' | jq .xpath

# Later: does the current workspace XML still contain exactly this node?
uv run dgml node prove "$fid" --docset "$ds" --proof node-proof.json | jq .valid
```

`prove` exit codes mirror `dgmlx verify`: `0` proven, `2` the proof no
longer holds (`valid: false` — compare `computed_node_hash` vs
`expected_node_hash` to tell "node changed" from "tree changed"), `1`
structural errors (no generated XML, malformed proof payload). Payload
shapes: cli-reference.md under "Node commands".

### Remove things

```bash
uv run dgml docset remove-file <file_id> --docset <docset_id>   # unassign only
uv run dgml docset delete <docset_id>                  # delete docset; Files untouched
uv run dgml file delete <file_id>                      # delete File; clears all docset assignments
```

## Conflict policies on `file add`

| `--on-conflict` | When useful |
|---|---|
| `error` (default) | Strict imports where any duplicate should halt the run. |
| `skip` | Idempotent re-runs of a bulk import (recommended for loops). |
| `replace` | The source path is authoritative and changed content should overwrite. Drops the old File and any DocSet assignments it had. |
| `duplicate` | You explicitly want two records for the same bytes (rare). |

Check the response payload's `conflict_kind` (`"hash"` or `"path"`), `created`, and `note` fields to understand what actually happened.

## Things to remember

- Ghostscript must be installed system-wide (`brew install ghostscript` / `apt-get install ghostscript`). Without it, page rendering fails and `dgml check` will flag it.
- A `docset delete` does **not** delete the underlying Files — they may belong to other DocSets. Use `file delete` to remove a File entirely.
- IDs are 12-char base-36 strings. Don't try to derive them; always pull them from JSON output.
- The JSON output schema is part of the public API. If a command's output shape looks wrong, the implementation is probably the source of truth — read [packages/dgml/src/dgml/cli.py](../../../packages/dgml/src/dgml/cli.py).

## Discover XML element subtrees and stake them on chain

`dgml discover` inspects a file's generated DGML XML, groups element types by
structural role, and returns representative samples.  The `depth_first` field
in each sample is the exact leaf index `dgml node export --leaf <n>` accepts —
making the discover → pick → stake workflow a straight pipeline.

**Full workflow: discover valuable fields, export attestation, stake on chain**

```bash
# 1. Generate DGML for a file in a docset (if not done yet).
dgml docset generate "$ds" --files "$fid"

# 2. Discover which element types look like extractable values.
payload=$(uv run dgml discover "$fid" --docset "$ds" --filter Values)

# 3. Inspect what was found.
jq '.tags[] | {tag: .tag, role: .role, count: .count}' <<<"$payload"

# 4. Pick a tag you want to attest; grab the depth_first index of its first sample.
df=$(jq -r '.tags[] | select(.tag=="LiabilityCap") | .samples[0].depth_first' <<<"$payload")

# 5. Export the node attestation payload (hash + Merkle proof).
node_payload=$(uv run dgml node export "$fid" --docset "$ds" --leaf "$df")
echo "node hash: $(jq -r .node_hash <<<"$node_payload")"

# 6. Stake the node on chain (requires dgml[chain] and a configured registry).
uv run dgml stake node "$fid" --docset "$ds" --leaf "$df" \
  --registry "$registry_addr" --chain nvnm-testnet
```

**Pagination pattern** — if you want samples for every tag, iterate:

```bash
uv run dgml discover "$fid" --docset "$ds" --filter All --samples 3 |
  jq -r '.tags[] | "\(.tag)\t\(.samples[0].depth_first)\t\(.role)"'
```

**Key contract points**

- `--filter All` (default) returns every non-root tag type.  Use a more
  specific filter (`Values`, `Sections`, `Density`, `Patterns`) to narrow
  down before staking.
- Semantic filters (`Who`, `When`, `Amounts`, `Definitions`, `Rules`) call
  the generation LLM configured in `<workspace>/config.json`.  If config is
  absent or the call fails, the command warns on stderr and falls back to
  `All` — it does **not** hard-fail.
- `depth_first` in each sample maps to `dgml node export --leaf <n>` and
  `dgml stake node --leaf <n>`.  The XPath in `samples[].xpath` is also
  accepted by `node export --xpath`.
- `samples[].page` is the page number from the element's `dg:origin`
  attribute; `null` for ungrounded XML.
- Framework (`dg:`) elements are excluded by default; pass
  `--include-structural` to see them.
- Default output includes `tag`, `count`, and samples with `xpath` + `xml`
  (attributes stripped).
  Pass `--full` to also get `role`, `filters`, `depth_first`, and `page`.
- `--search <term>` filters results to tags whose name contains the term
  (case-insensitive). Useful for browsing by keyword, e.g. `--search date`.
- `--search-content <term>` filters to tags whose sample XML contains the
  term — finds tags by what they hold, not what they're called.
