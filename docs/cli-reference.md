# `dgml` CLI Reference

The `dgml` CLI is designed for both humans and LLM-agent consumption.
Output is JSON by default; errors are structured envelopes; commands are
flag-driven (no interactive prompts) and idempotent where reasonable.

## Conventions

- **stdout** carries the success payload as a JSON object.
- **stderr** carries error envelopes:
  ```json
  { "error": { "code": "FILE_NOT_FOUND", "message": "..." } }
  ```
- Exit codes:
  - `0` — success
  - `1` — error (anything in the `error` envelope)
  - `2` — `dgml check` ran but found issues

`--format text` switches to a basic key/value text format if a human is
driving the CLI directly.

**Boolean flags** follow one convention: a positive `--flag` (e.g.
`--auto-classify`, `--recursive`, `--force`, `--skip-existing`) turns on a
behavior that is **off by default**; a `--no-*` flag (e.g. `docset
generate`'s `--no-coverage`) opts **out** of a step that is **on by
default**. So `--no-*` appears only where the default is "do it"; everything
else is opt-in.

A complete list of error `code` values is in the [Error code
reference](#error-code-reference) at the end of this document.

## Global flags

These parse in **any position** — before the subcommand
(`dgml --format text file list`), after it (`dgml file list --format text`),
or after a command group (`dgml docset --format text list`).

| Flag             | Description |
|------------------|-------------|
| `--workspace`    | Override the workspace root. Default: `$DGML_HOME` then `./dgml-workspace`. |
| `--format`       | `json` (default) or `text`. |
| `--verbose`      | Emit informational diagnostics to stderr. Controls hybrid text-mode warnings (digital/OCR conflicts, OCR misses) and the per-page merge summary, plus the `docset generate` pipeline's progress lines. Off by default — stderr stays reserved for error envelopes. |
| `--debug`        | Keep intermediate debug files in the workspace **and** record LLM cost/token telemetry to `<workspace>/usage.jsonl`. Off by default, so only final files (and the small functional cache the next run reloads) are kept. With `--debug` off: no `usage.jsonl` rows are written for **any** operation (classify, cluster, transcribe, label, links, schema/value extraction, hybrid merge); `docset generate` skips the debug-only `cache/` artifacts (raw LLM dumps, `*.concept.xml`/`*.semantic.xml`, prompt listings) and `coverage_report.json`; and the in-place grounding pass skips the `<stem>.dgml.grounding_stats.json` sidecar. The functional `cache/` files (`*_blocks.json`, `label_*_cNN_raw.json`, `concept_roster.json`) are **always** written — incremental generation reloads them. Pass `--debug` to retain the debug artifacts and log usage. (Coverage summaries still print on stderr under `--verbose` either way.) |

## Workspace commands

### `dgml init [--refresh]`
Establish the **shared config** — nothing else. `init` creates the peer
`local_config.json` (a copy of the bundled default template) in the directory
that contains the workspace (`<workspace-parent>/local_config.json`; with the
default `./dgml-workspace` this is `./local_config.json`). It does **not**
create `docsets/`, `files/`, or any workspace `config.json` — that is
`dgml workspace create`.

Configure once, create many: every workspace that is a sibling of that
`local_config.json` inherits it. Edit the file (review the models and the OCR
endpoint) before running `dgml workspace create`.

- **No `local_config.json` yet:** copies the bundled template there and returns
  `local_config_created: true` with a `next_action` hint.
- **Already present:** no-op; returns `local_config_created: false`.
- **`--refresh`:** overwrite `local_config.json` from the bundled template
  (pull the latest baseline / new knobs). The previous file is copied to
  `local_config.json.bak` first. Invoking the flag is the consent — no prompt.

Output (JSON):

```json
{
  "local_config_path": "…/local_config.json",
  "local_config_created": true,
  "refreshed": false,
  "next_action": "edit …/local_config.json then run dgml workspace create"
}
```

`next_action` is present only when the file was just created. Advisory text
goes to stderr; stdout stays the JSON contract.

`local_config.json` lands in a working directory, so add it to your
`.gitignore` (this repo already does).

### `dgml workspace create [PATH] --organization ORG [--name NAME] [--force]`
`PATH` is the optional directory to create the workspace in — pass it to avoid
the redundant global `--workspace` (`dgml workspace create ./ws …`). When
omitted, the root resolves in the usual order (global `--workspace` → `$DGML_HOME`
→ `./dgml-workspace`); a `PATH` given here overrides that.

Create a workspace **from the shared config**. Steps:
1. Seeds the peer `local_config.json` from the bundled template if it does not
   already exist — so `create` works **without a prior `dgml init`**. (When it
   is created here, the response carries `local_config_created: true` and a
   `next_action` telling you how to edit the config; see below.)
2. Creates `docsets/` and `files/`.
3. Copies `local_config.json` → `<workspace>/config.json` **verbatim**
   (comments intact), only if `config.json` does not already exist.
4. Writes the workspace identity (`name` + `organization`) to
   `<workspace>/workspace.json`.

`--organization` is **required**. It is embedded in this workspace's docset
namespace URIs (`http://dgml.io/<organization>/<DocSetSlug>`), replacing the
workspace directory name that older releases used there — so pick a stable
identifier for your org (changing it later shifts the namespaces of newly
generated XML). `--name` is optional human-readable identity metadata and
defaults to the workspace directory name.

`--force` overwrites an existing `config.json` with the current
`local_config.json` (doubles as "re-sync my edited shared config into this
workspace"); the overwrite is noted on stderr.

Output (JSON):

```json
{
  "workspace": "…/dgml-workspace",
  "name": "dgml-workspace",
  "organization": "Acme",
  "initialized": true,
  "config_path": "…/dgml-workspace/config.json",
  "config_written": true,
  "local_config_path": "…/local_config.json",
  "local_config_created": false
}
```

`config_written` is `false` when an existing `config.json` was left untouched
(no `--force`). `local_config_created` is `true` only when this call seeded the
shared `local_config.json` (no prior `dgml init`); in that case an extra
`next_action` field is present telling you to edit the models and OCR endpoint
in `<workspace>/config.json` (or edit the shared `local_config.json` and re-run
`dgml workspace create --force` to re-sync).

### `dgml status`
Summary: workspace path, count of docsets, count of files.

### `dgml check [--retry-errors]`
Walk the workspace and report inconsistencies. Issue kinds emitted today:

| Kind | Target | Meaning |
|---|---|---|
| `missing_metadata` | file/docset | `file.json` or `docset.json` missing (or missing required field like `original_filename`) |
| `corrupt_metadata` | file/docset | `file.json`/`docset.json` exists but is not valid JSON |
| `missing_pdf` | file | The PDF named in `file.json` is missing from the file directory |
| `hash_mismatch` | file | Stored sha256 doesn't match current PDF bytes |
| `pdf_unreadable` | file | pypdf can't parse the PDF (records a permanent error so the next check skips re-parsing) |
| `pdf_unreadable_permanent` | file | A previous parse recorded a permanent failure; not retried without `--retry-errors` |
| `page_count_mismatch` | file | `page_images/` has the wrong number of PNGs (`repaired: true` if rerendered successfully) |
| `page_render_failed` | file | ghostscript failed during rerender |
| `page_render_failed_permanent` | file | A previous render recorded a permanent failure; not retried without `--retry-errors` |
| `page_text_count_mismatch` | file | `page_text/` has the wrong number of per-page JSONs (`repaired: true` if re-extracted successfully) |
| `page_text_corrupt` | file | A `page_text/page_N.json` exists but is not valid JSON / missing required fields |
| `text_extraction_failed` | file | pdfminer.six failed to extract any words (records a permanent error so future checks skip retry) |
| `text_extraction_failed_permanent` | file | A previous extraction recorded a permanent failure; not retried without `--retry-errors` |
| `dangling_file_reference` | docset | DocSet references a File ID that doesn't exist |
| `computed_field_unattributed` | docset | A `dg:origin="computed"` element in a file's DGML XML has no `dg:href` sources — the derivation can't be audited (spec §13 requires computed fields to name their sources) |

`--retry-errors` clears recorded permanent errors and re-attempts the
failed operations.

> **Note:** `check` validates the stored **original** for each file (the
> `original_filename` named in `file.json` — its presence and sha256). For a
> file added from a convertible source (docx/xlsx/…), the converted
> `<stem>.pdf` persisted alongside it (see [Document conversion](conversion.md))
> is a derived artifact that `check` does **not** verify — a missing or
> corrupted converted PDF is not reported. `page_images/` and `page_text/`
> (derived from it at add time) are checked as usual; if the converted PDF is
> gone, generation falls back to re-converting from the original.

### `dgml cluster [--skip-existing] [--config PRESET|PATH] [--mode auto|fresh|incremental] [--method auto|embedding|llm] [--small-corpus-threshold N]`

Requires `pip install dgml[clustering]`. The extra pulls in the
`dgml-clustering` workspace package and its ML stack (embedding models,
`leidenalg`, `scipy`, `sklearn`); without it the
command exits 1 with `MISSING_EXTRA`.

`--skip-existing` makes the command a no-op when **every** file is already
assigned to a DocSet — the clusterer is not run and the payload comes back
with `skipped: true` (and empty `clusters`/`failed_file_ids`).
Use it to make resume/re-run loops cheap. Without the flag (or when at least
one file is still unassigned) the command clusters as normal and reports
`skipped: false`.

`--config PRESET|PATH` selects the clustering configuration for this run.
It is either a **bundled preset name** — `small` (CPU-only tf-idf + Leiden, no
UMAP; for tiny corpora), `light` (CPU-only tf-idf + Leiden/UMAP; the default),
`medium` (tf-idf text fused with a 2B vision encoder, large CPU / Apple MPS),
or `heavy` (8B vision encoder alone + Leiden/UMAP, GPU) — or a **path** to a
standalone clustering config JSON. The JSON holds the same fields as the
`clustering` section of `<workspace>/config.json` (`encoder_text`,
`encoder_image`, `fusion`, `manifold`, `training`, `scenario`); it is
deep-merged over the bundled defaults exactly like the workspace section, and
**replaces** that section for this run (the two are not combined). An unknown
preset name, or a path that doesn't exist / isn't valid JSON / isn't a JSON
object, exits 1 with `CLUSTERING_CONFIG_INVALID`. Use it to A/B configs without
editing `config.json`, or to move up the CPU → MPS → GPU tiers on a specific
run.

`--mode auto|fresh|incremental` selects fresh vs incremental clustering
(default `auto`):

- `fresh` — cluster all unassigned files from scratch into emergent clusters
  (scenario S1), ignoring any existing DocSets as prototypes.
- `incremental` — the "S3" workflow: grow an **existing** clustering. Each
  existing DocSet becomes a category whose prototype is reconstructed from
  *all* of its already-assigned members' embeddings (few-shot S3). New files
  are assigned to the nearest existing DocSet when they fit; the rest form
  emergent `unknown_N` clusters that are LLM-named into new DocSets. Forcing
  this mode on a workspace with no DocSets exits 1 with
  `INCREMENTAL_WITHOUT_CLUSTERS`.
- `auto` — resolves to `incremental` when the workspace already has DocSets,
  else `fresh`.

`--method auto|embedding|llm` selects *how* documents are grouped, orthogonal
to `--mode` (default `embedding`):

- `embedding` — the statistical pipeline (encode → project → cluster) described
  below. The right choice once a corpus is large enough for tf-idf / neighbor
  statistics to be meaningful.
- `llm` — send **every** document's rendered first pages to the vision LLM in a
  single call and let it partition them by document type. Built for **very small
  corpora**, where the embedding pipeline has too little signal to cluster
  reliably (tf-idf has almost nothing to weight, k-NN graphs are dominated by
  noise). The model partitions *and* names emergent groups in the one call, so
  no second per-cluster naming round-trip is needed. `--config` is ignored on
  this path (there is no embedding pipeline to configure).
- `auto` — route to `llm` when at most `--small-corpus-threshold` files are
  clusterable, else `embedding`.

`--small-corpus-threshold N` (default `8`) is the cutoff `--method auto` uses:
corpora of at most `N` clusterable files go to the LLM partitioner, larger ones
to the embedding pipeline. Ignored for `--method embedding` / `--method llm`.

Both `--method llm` and `--method auto` (when it routes to the LLM) require the
same `classification` config as `--auto-classify` (see "Auto-classification"
above) — the LLM partitioner *is* the classifier's vision machinery. A missing
`classification` section makes the LLM path soft-fail: every clusterable file
lands in `failed_file_ids`. The LLM path caps a single call at 24 files; any
beyond that are reported in `failed_file_ids` so you can fall back to the
embedding pipeline for larger corpora.

Cluster files not currently assigned to any DocSet, and **assign each
clustered file to a DocSet**. Runs in two passes:

1. Files whose cluster name matches an existing DocSet are assigned to
   that DocSet immediately.
2. Files whose cluster doesn't match are grouped by cluster name; for
   each unmatched cluster the vision LLM is sent up to
   `MAX_FILES_PER_CLUSTER_NAMING` files and asked to propose
   a `(name, description)` for a fresh DocSet. The DocSet is created and
   every file in that cluster is assigned to it.

Partial success is the contract: if classification config is missing/
invalid, or the LLM call for a given cluster fails, the files in that
cluster fall into `failed_file_ids`. Every other cluster (matched or
successfully named) is still assigned. The command always exits `0`.

```json
{
  "clusters": {
    "k7q3xb91pmrf": "Contracts",
    "abc123def456": "Receipts",
    "xyz789": "unknown_1"
  },
  "failed_file_ids": ["xyz789"],
  "skipped": false,
  "mode": "incremental",
  "n_assigned_existing": 2,
  "n_new_clusters": 1,
  "assignments": {
    "k7q3xb91pmrf": {"docset": "Contracts", "confidence": 0.83, "is_new": false},
    "abc123def456": {"docset": "Receipts", "confidence": 0.71, "is_new": false}
  }
}
```

The first three fields are the core contract. The remaining
fields are **additive** (optional — consumers can ignore them) and describe
the incremental workflow:

| Field | Meaning |
|---|---|
| `clusters` | Map from file id to the DocSet name the file ended up in. Either an existing DocSet's name (the algorithm matched the file to it) or the LLM-proposed name for a newly-created DocSet. Files that failed to assign keep their algorithmic placeholder label (e.g. `"unknown_1"`) here *and* appear in `failed_file_ids`. |
| `failed_file_ids` | Files whose cluster needed LLM naming and that naming failed (missing config, no page images, provider error, …). Re-run after fixing the underlying cause; assignments are idempotent. |
| `skipped` | `true` only when `--skip-existing` was passed and there were no unassigned files (the clusterer never ran); `false` on every actual clustering run. Always present. |
| `mode` | The effective run mode after resolving `auto` — `"fresh"` or `"incremental"`. |
| `n_assigned_existing` | Number of files assigned to a DocSet that already existed before this run (the incremental "fit an existing cluster" case). |
| `n_new_clusters` | Number of new DocSets created this run (emergent clusters that were LLM-named). |
| `assignments` | Per-file detail: `docset` (final DocSet name), `confidence` (nearest-prototype confidence in `[0, 1]`, or `null` for emergent clusters), and `is_new` (whether the DocSet was created this run). |

LLM naming requires the same workspace setup as `--auto-classify`:

- A `classification` section in `<workspace>/config.json` (see
  "Auto-classification" above).
- `pip install dgml[classification]` (provides `litellm`).

If neither is in place, every unmatched cluster's files end up in
`failed_file_ids`; matched files still get assigned. The clustering
algorithm runs via the `dgml-clustering` workspace package
([packages/dgml-core/src/dgml_core/run_clustering.py](../packages/dgml-core/src/dgml_core/run_clustering.py))
— in incremental mode, S3 (few-shot) when existing DocSets have usable
members, S2 (partial-labels, name-only) when they don't; in fresh mode, S1
(unsupervised). See [docs/incremental-clustering.md](incremental-clustering.md)
for the full incremental ("S3") workflow and the evaluation harness.
Files whose first-page image is missing (page render failed at ingest)
are routed into `failed_file_ids` along with LLM-naming failures.

Algorithm settings (encoder, fusion, manifold, training, …) come from
the bundled
[clustering_config.json](../packages/dgml-core/src/dgml_core/clustering_config.json).
Operators can override any subset of them in one of two ways: an optional
`clustering` section in `<workspace>/config.json` (peer to `classification`),
or a standalone file passed with `--config PATH` (which replaces that section
for the run). Both use the same field schema — see the
[`clustering`](storage-layout.md#clustering-optional) entry in the
storage-layout doc for the field rules. Missing section and no `--config` ⇒
bundled defaults stand.

Errors:

| Code | Cause |
|---|---|
| `MISSING_EXTRA` | The `clustering` extra is not installed. |
| `CLUSTERING_CONFIG_INVALID` | `<workspace>/config.json` has a `clustering` section that isn't a JSON object, or a field inside it failed schema validation (typo, out-of-enum value, etc.). |

## DocSet commands

```
dgml docset create --name NAME [--description DESC] [--key-question Q ...]
dgml docset list
dgml docset show <docset_id>
dgml docset update <docset_id> [--name NAME] [--description DESC]
dgml docset delete <docset_id>
dgml docset add-file <file_id> --docset <docset_id>   # auto-extracts when the
                                                     # DocSet has an extraction schema
dgml docset remove-file <file_id> --docset <docset_id>
dgml docset list-files <docset_id>
dgml docset generate <docset_id> [--window-size <n>] [--max-tokens <n>] [...]
```

`docset delete` removes the DocSet and its file-assignment markers, but
**does not delete the underlying Files**. Files remain in
`<workspace>/files/` and may still belong to other DocSets.

**Auto-extract on assignment.** When the target DocSet has an extraction
schema set (`extraction-schema.rnc`), every assignment path fires value
extraction on the newly-assigned file: `docset add-file`, `file add
--auto-classify` (existing-DocSet decisions), and `cluster` (existing-DocSet
matches — a DocSet created mid-run can't have a schema yet). The payload
gains an `extraction` block; extraction failures are **soft** (the error
lands in `extraction.error`, the assignment stands, exit stays 0). No schema
→ plain assignment, no block.

```json
{
  "docset_id": "o8vr8rs488vg",
  "file_id": "5kqt9r5fowno",
  "assigned": true,
  "extraction": {
    "performed": true,
    "model": "gemini/gemini-2.5-pro",
    "tool_calls": 0,
    "error": null
  }
}
```

### `dgml docset generate <docset_id> [flags]`

Run the typed-block PDF→DGML pipeline over every file in a DocSet: each
PDF is transcribed window-by-window into a flat list of typed JSON blocks
(`generation.model`), then ONE batch-wide semantic-labeling call assigns
concept tags across all of the docset's documents at once
(`generation.label_model`), and the result is rendered deterministically
into a namespaced `dg:chunk`
document. The labeling vocabulary (the "roster") is planned automatically
from the documents, or supplied up front with `--schema-path` to make labels
deterministic. Unseeded runs are staged: the largest documents label first (a
pilot), their observed evidence — verbatim example values, kinds, hierarchy —
confirms the planned vocabulary, and the rest of the batch labels against it.
There is no separate transform pass.

**Incremental, consistent growth.** Already-generated files are skipped (see
resume below), so adding a document and re-running generates only the new one.
To keep its tags consistent with the existing docset, the new document is
labeled seeded with the docset's own `schema.json` — full fidelity: role
descriptions, observed examples, kind, hierarchy — falling back to the flat
`cache/concept_roster.json` when no schema exists (default; disable both with
`--no-roster`). Every concept is emitted in the per-docset
`docset:` vocabulary namespace (`dg:` is framework-only), so growing the docset
never flips a tag's prefix. An already-generated file is still re-rendered
deterministically when its output changes as the docset's schema/roster grows
(e.g. entity-container grouping) — no re-transcription or re-labeling. These
show up in the top-level `rerendered` list.

Output always goes to the docset directory in the workspace
(`<workspace>/docsets/<docset-id>/`) — there is no output-directory flag,
so artifact placement is deterministic. Each file's generated DGML lands in
its per-(docset, file) directory at
`<workspace>/docsets/<docset-id>/files/<file-id>/<stem>.dgml.xml`. The
`cache/` at the docset root holds the small functional files the next run
reloads (`*_blocks.json`, `label_*_cNN_raw.json`, `concept_roster.json`) and
is always written. Its debug-only artifacts (raw LLM dumps,
`*.concept.xml`/`*.semantic.xml`, prompt listings) and `coverage_report.json`
are written only under the global `--debug` flag — a default run leaves the
workspace with final files plus that functional cache. The docset root also
gets `schema.json` (the generation tag schema, written during labeling) and —
at the very end of the run, after grounding and the semlink pass —
`full-schema.rnc`: the same schema rendered as RELAX NG Compact with the data
types observed in the final XML, losslessly reversible back to `schema.json`
(see [storage-layout.md](storage-layout.md)). The RNC render is the form that
ships in DGMLX bundles and is hashed into the file attestation.

**Grounding is part of generation.** After each `<stem>.dgml.xml` is
rendered it is grounded *in place* against the file's page OCR — a
`dg:origin` bounding-box attribute is added to every element whose
subtree grounded (leaf elements, mixed-content parents, *and* pure
containers), so the canonical `<stem>.dgml.xml` already carries page
positions. This pass is fully
deterministic (no LLM, no config): the DGML tree and the OCR word stream
are both "the document in reading order", so it is solved as a sequence
alignment (rare-n-gram anchoring + windowed diff), with a
weighted-similarity recovery pass for OCR noise, a span-search rescue for
repeated content, and a row-context pass for punctuation-only cells,
interleaved multi-line table cells, and digit-discrepant text. A file
with no `page_text/` (added without `--text-mode digital`/`ocr`/`hybrid`)
is written but left ungrounded, with a warning — it does not fail the run.

**The `dg:origin` attribute.** Qualified to whatever URI the document binds
the `dg` prefix to (the open `dgml.io` scheme on generated DGML; plain
`origin` on namespace-free XML). Its value is a `"; "`-separated list of
boxes, each `<page> <x1> <y1> <x2> <y2>` (space-separated) in integer image
pixels — top-left origin, 300 dpi, relative to the page's
`page_images/page_N.png`:

```xml
<docset:Body structure="h3"
    dg:origin="3 307 367 1098 428; 4 307 254 1093 376">...
```

Elements with text-node children carry one box per visual line on each
page — the CSS `getClientRects()` analogue, uniform for leaves and
mixed-content parents. Pure containers (all-element children — sections,
lists, tables, rows, the document root) carry one union box per page — the
`getBoundingClientRect()` analogue — covering their grounded subtree. An
element is annotated only when at least half of its subtree's tokens
grounded.

**The `dg:style` attribute.** Alongside `dg:origin`, the grounding pass also
emits `dg:style` — observed visual formatting as inline CSS that can be copied
verbatim into an HTML `style` attribute. It is **sparse**: emitted only when a
property is evident in the source and applied at the most specific element
where observable, chosen by char-weighted majority over the element's own text
(if more than half a chunk's characters are bold, the whole chunk is bold).
Like `dg:origin`, it is qualified to the document's `dg` URI.

An inheriting property (`color`, `font-*`, `text-align`, `text-transform`,
`white-space`) is emitted only on the element that *introduces* it — a
descendant that would merely inherit the same value from an ancestor does not
repeat it (`dg:style` is copied verbatim into HTML `style`, where these
properties inherit). So a paragraph rendered entirely red carries
`color: red` once, on the paragraph — not again on every inner span.
Conversely, a descendant whose own formatting *differs* from what it would
inherit carries the overriding value **even when that value is the CSS
default** — e.g. a plain run inside a bold heading emits `font-weight: normal`,
so it doesn't render bold under HTML inheritance. Non-inheriting properties
(`text-decoration`, `background-color`) are never suppressed this way.

```xml
<docset:CompanyName dg:structure="span" dg:style="font-weight: bold; color: gray">Acme Corp</docset:CompanyName>
```

For `--text-mode digital` and `hybrid`, the facts are read deterministically
from the PDF's glyphs via pdfminer (font weight/slant from the font name, size,
and text fill color), carried through `page_text/page_N.json`. The deterministic
path derives `font-weight`, `font-style`, `font-size` (an `em` bucket relative to
the page's modal body size: `0.75em | 1em | 1.25em | 1.5em | 2em`), `color` (any
CSS named color), and `text-transform: uppercase` (all-caps text).
`--text-mode ocr` carries no font facts, so `dg:style` is empty there unless the
workspace opts into the image-based path by adding a `style`
section to `config.json` (its presence is the switch; it names a vision
`model` — see [storage-layout.md](storage-layout.md)). That path assesses the
same properties from the page image **plus `text-align`** (which needs the
rendered layout to judge). It is honored only for OCR files and never competes
with the deterministic digital/hybrid path. A malformed `style` section fails
`generate` up front with `STYLE_CONFIG_INVALID`.

Requires only the base `dgml` install — the generation pipeline reuses
the workspace's pre-rendered `page_images/page_N.png` files at LLM
input time, so no extra rasterizer is needed at run time (and no
GPL/poppler escape hatch). For non-workspace inputs (library callers
passing arbitrary paths), the pipeline renders to a tempdir via the
same canonical `pages.render_pages` (ghostscript).

The models are **not** CLI flags — like every other model-consuming command
(`docset schema generate`, `file extract`, `discover`), `generate` reads them
solely from the `generation` section of `<workspace>/config.json`, so each is
one visible, deliberate choice per workspace. There is no code default and both
`model` and `label_model` are required: a missing
section or model fails with `GENERATION_CONFIG_MISSING`, a malformed one with
`GENERATION_CONFIG_INVALID`. See the [`generation` config
section](storage-layout.md#generation-required-for-dgml-docset-generate) for the
fields (`model`, `label_model`, `api_key`/`api_key_env`, `api_base`).
| `--window-size <n>` | `10` | Pages per transcription window. |
| `--temperature <f>` | `0.0` | LLM temperature. |
| `--max-tokens <n>` | `32000` | LLM max output tokens per call. |
| `--no-coverage` | off | Skip word-coverage metrics (unique-lexicon recall, ROUGE-1/2) computed against the workspace `page_text/`. |
| `--cache-dir <dir>` | `<docset-dir>/cache` | Directory for the generation cache (functional `*_blocks.json` / `label_*_cNN_raw.json` / `concept_roster.json`, always written; plus per-window debug snapshots when `--debug` is set). |

The global `--debug` flag also writes the per-file
`<stem>.dgml.grounding_stats.json` sidecar (grounding match rates,
ungrounded snippets); the `dg:origin` boxes themselves are always written
into `<stem>.dgml.xml` regardless.
| `--max-parallel-calls <n>` | `4` | Max documents transcribed concurrently (windows *within* a document stay serial). The LLM call is network-bound, so threads overlap the latency. Set to `1` to disable. Tune to your provider's RPM tier — e.g. Gemini free ~10-15 RPM, Gemini paid Flash 500 RPM, OpenAI free 500 RPM, Anthropic tier-1 ~50 RPM. |
| `--schema-path <f>` | none | Exported schema to seed labeling with — either `docsets/<id>/schema.json` (Schema v1: a `tags` map of `name -> {role, kind, parent_role, …}`) or its lossless RELAX NG Compact render `docsets/<id>/full-schema.rnc` (both written by `docset generate`). When given, this vocabulary is used as-is with **full fidelity** — role descriptions, curated examples, and kind all feed the labeling prompt — and the planning pass is **skipped**, making labels deterministic across runs; the tag hierarchy (`parent_role`) also seeds entity-container grouping. Per-document labeling still extends it for roles the schema doesn't cover. A flat `{concept: description}` mapping is **not** accepted. |
| `--no-roster` | off | Disable automatic vocabulary reuse. By default an incremental generate seeds labeling from the docset's own `schema.json` (full fidelity: descriptions, observed examples, kind, hierarchy), falling back to the flat `cache/concept_roster.json` from a prior run, so newly-added documents stay tag-consistent with the existing docset; this flag labels them in isolation instead. Unlike `--schema-path`, automatic reuse does **not** seed entity-container grouping. Ignored when `--schema-path` is given. |
| `--no-semlinks` | off | Skip the final semantic-link pass. By default each grounded `<stem>.dgml.xml` gets semantic links added in place — relationships the tree's nesting can't capture, written as `dg:itemprop` (predicate) + `dg:href` (`#id`, or space-separated `#id`s) on the subject, with `xml:id`s assigned to both ends. Covers references (`references`, `incorporates`, `signatoryOf`, …), relative dates (`relativeTo`/`effectiveOn`, ISO-8601 offset in `dg:value`), and derived values (`greaterOf`/`lesserOf` formulas, `escalates`, `valueFrom`). The model proposes links on the labeling model (`generation.label_model`), then a skeptical pass verifies them. Each converted file's `results` entry carries a `links` count. |

**Document-level resume.** If a file's per-(docset, file)
`<stem>.dgml.xml` already holds a generated document tree, that file is
skipped — a crashed run can be re-invoked with the same arguments and only
unfinished documents are re-processed. Within a re-processed document,
transcription itself also resumes: a cached `cache/<stem>_blocks.json`
(written right after Pass A) is reloaded verbatim instead of re-transcribing,
so re-running a document whose output was removed — or relabeling a docset
after deleting its `.dgml.xml` outputs and label caches — pays only for
labeling and rendering. Delete the `_blocks.json` file to force a fresh
transcription. When *every* assigned file is already
converted the command exits 0 with the same envelope
(`summary.converted == 0`, each file a `skipped` entry in `results`) and no
LLM call is made. An **extraction-only** file (a `dg:extraction` with no
tree, from running `extraction extract` first) does *not* count as
converted: generate builds its tree and carries the existing
`dg:extraction` over into the fresh render (`full-extraction`). The same
carry-over protects re-rendered files — a namespacing-driven re-render
never drops extracted values.

Payload on a normal run — the shared batch envelope (`summary` count block +
per-item `results`, each carrying a `status`), matching the bulk `file add`:

```json
{
  "docset_id": "p9pjusnwg50l",
  "docset_name": "Q2 contracts",
  "summary": { "total": 3, "converted": 2, "skipped": 1, "failed": 0 },
  "rerendered": [],
  "output_dir": "/ws/docsets/p9pjusnwg50l",
  "coverage_report": "/ws/docsets/p9pjusnwg50l/coverage_report.json",
  "results": [
    {"status": "skipped", "file_id": "ab55kdjs93kk", "source": "already-done.pdf", "output": "/ws/docsets/p9pjusnwg50l/files/ab55kdjs93kk/already-done.dgml.xml"},
    {"status": "converted", "file_id": "k7q3xb91pmrf", "source": "contract-a.pdf", "output": "/ws/docsets/p9pjusnwg50l/files/k7q3xb91pmrf/contract-a.dgml.xml", "grounded": true, "matched_token_pct": 99.6, "elements_annotated": 445}
  ]
}
```

`summary` counts always sum to `total`: every assigned file lands in exactly
one of `converted` / `skipped` / `failed`. A per-file problem does not abort
the run — it becomes a `failed` entry in `results` and the batch continues,
exiting 0 (partial success, matching `dgml cluster`).
Three things produce a `failed` entry:

- **`FILE_NOT_FOUND`** — the file's source is missing from
  `<workspace>/files/<file_id>/`.
- **`GENERATION_FAILED`** — the pipeline produced no output for the file
  (transcription failed and was skipped internally), **or** two assigned files
  share a filename (the pipeline keys documents by filename and can't tell them
  apart, so both are failed; rename to convert them). When transcription failed
  with a captured cause (e.g. an LLM/provider error that survived retries), the
  `error.message` carries a short, single-line summary of it — so the reason is
  available without `--verbose`. The full, untruncated error still goes to
  stderr under `--verbose`.

`output_dir` is the docset directory; per-file DGML lands under its
`files/<file-id>/` subdirectory (see each `results` entry's `output`).
`coverage_report` is the report path only when a report was actually written;
it is `null` when `--no-coverage` is set, no file had `page_text/`, or
`--debug` was not passed (coverage is still computed and its per-file summary
printed under `--verbose`, but the `coverage_report.json` file is written only
under `--debug`).

A `failed` entry looks like:

```json
{ "status": "failed", "file_id": "ab55kdjs93kk", "source": "gone.pdf",
  "error": { "code": "FILE_NOT_FOUND", "message": "source not found at ..." } }
```

Errors (run-level, error envelope + exit 1):

| Code | Cause |
|---|---|
| `DOCSET_NOT_FOUND` | `<docset_id>` does not exist. |
| `EMPTY_DOCSET` | DocSet exists but has no files assigned. |
| `GENERATION_CONFIG_MISSING` | No `generation` section (or a missing `model` / `label_model`) in `<workspace>/config.json`. |
| `GENERATION_CONFIG_INVALID` | The `generation` section is malformed (bad model string, both `api_key` and `api_key_env` set, etc.). |

> stdout is a single JSON object. The transcription / labeling / render
> progress lines go to stderr, and only when `--verbose` is passed.

> **Grounding is part of `generate`.** There is no separate `dgml docset
> ground` command — `generate` writes `dg:origin` boxes into each
> `<stem>.dgml.xml` itself (see above). To re-run *just* the grounding pass
> on already-generated XML without regenerating (a maintenance/debug
> operation outside the public CLI), use `scripts/ground.py`:
> `uv run python scripts/ground.py --docset <id> [--file <id>] [--debug]`.

## Extraction commands

Schema-driven value extraction pulls a defined set of fields out of a document
and grounds each value back to the source page. It is distinct from `docset
generate`, which transcribes the *whole* document.

Two formats are involved:

- **Schema** — the canonical at-rest form is **RELAX NG Compact**
  (`extraction-schema.rnc`, per the DGML spec §12/§13). The CLI also *accepts* a
  grounded-field JSON Schema
  on input and converts it to RNC before storing. Schemas may carry
  `## Prompt:` annotations guiding the LLM where to find each field.
- **Values** — extraction writes a `dg:extraction` element **inside the file's
  core `<stem>.dgml.xml`** (spec §13), holding the schema's fields as `docset:`
  elements with `dg:value`/`xsi:type` and `dg:origin`. There is no separate
  values file. Two modes (reported as `mode` on the `extract` payload):
  `full-extraction` when the file already has a generated document tree (the
  `dg:extraction` is added as a sibling), or `extraction` when the core file is
  created with only the `dg:extraction` element. The CLI can return the values
  as values-shape JSON on request.

The LLM is configurable like every other model-using command — via the
`grounded` section of the workspace `config.json` (`schema_model`,
`values_model`, API keys, `max_tool_iters`), with per-call overrides on the
commands below.

### `dgml extraction generate-schema <docset_id> [--from-file ID ...] [--schema-model M]`

Ask the configured `schema_model` to propose an extraction schema from one or
more sample PDFs, then store it as `extraction-schema.rnc`. `--from-file` is repeatable and
defaults to every file in the DocSet. Errors `NO_FILES` if the DocSet is empty
and no `--from-file` is given.

```json
{
  "docset_id": "o8vr8rs488vg",
  "schema_format": "rnc",
  "schema": "namespace dg = \"http://dgml.io/ns/dg#\"\n...",
  "from_file_ids": ["5kqt9r5fowno"],
  "model": "anthropic/claude-opus-4-7"
}
```

### `dgml extraction set-schema <docset_id> --schema-file PATH`

Set the DocSet's extraction schema from a file. Accepts a `.rnc` (RELAX NG
Compact) or `.json` (grounded-field JSON Schema) document — JSON is converted to
RNC. Anything outside the supported RNC subset is rejected with `SCHEMA_INVALID`.
RNC is the only on-disk form. Returns `{docset_id, schema_format: "rnc", schema}`.

### `dgml extraction get-schema <docset_id> [--schema-format rnc|json]`

Return the DocSet's schema as canonical RNC (default) or as the engine's
grounded-field JSON Schema projection (`--schema-format json`). Errors
`SCHEMA_NOT_FOUND` if none is set.

### `dgml extraction extract <docset_id> <file_id> [--values-model M]`

Extract values from a file against the DocSet schema and write a `dg:extraction`
element into the file's core `<stem>.dgml.xml`. Runs a three-phase pipeline
(LLM text+pages → code OCR matching → per-page LLM bbox). If the file already has
a generated document tree the extraction is added alongside it
(`mode: full-extraction`); otherwise a minimal core file is created
(`mode: extraction`). `extraction_stats.json` is written only under the global
`--debug` flag. Errors `SCHEMA_NOT_FOUND` if the DocSet has no schema.

```json
{
  "docset_id": "o8vr8rs488vg",
  "file_id": "5kqt9r5fowno",
  "model": "gemini/gemini-2.5-pro",
  "mode": "full-extraction",
  "tool_calls": 2,
  "field_count": 7,
  "xml_path": ".../docsets/o8vr8rs488vg/files/5kqt9r5fowno/Invoice 2025.dgml.xml"
}
```

> `generate` and `extract` compose in either order: extract-then-generate
> builds the tree and carries the `dg:extraction` over; generate-then-extract
> embeds the extraction alongside the existing tree. Both end at
> `full-extraction`.

### `dgml extraction get-values <docset_id> <file_id> [--as values|xml]`

Return previously extracted values from the file's core `<stem>.dgml.xml`.
`--as values` (default) projects the `dg:extraction` element to values-shape
JSON; `--as xml` returns the core DGML document. Errors `VALUES_NOT_FOUND` if the
file has no `dg:extraction` element yet (extraction not run).

```json
{
  "docset_id": "o8vr8rs488vg",
  "file_id": "5kqt9r5fowno",
  "format": "values",
  "values": {
    "LiabilityCap": {
      "text": "$500,000",
      "value": "500000",
      "locations": [{"page_number": 2, "bounding_box": [460, 310, 1800, 355]}]
    }
  }
}
```

A leaf whose schema `## Prompt:` describes a derivation rule (spec §13) comes
back **computed** instead of grounded: `computed: true`, no `locations`, and
`derived_from` listing the dotted paths of the values it was derived from
(cross-file or unresolvable `dg:href` targets stay as raw references):

```json
{
  "InvoiceTotal": {
    "text": "$349.85",
    "value": "349.85",
    "computed": true,
    "derived_from": ["LineItems[0].Quantity", "LineItems[0].UnitPrice"]
  }
}
```

In the XML form (`--as xml`), the same field carries `dg:origin="computed"`,
`dg:value`, and `dg:itemprop="computedFrom"`/`dg:href` pointing at the source
elements, which are stamped with matching `xml:id` attributes.

**Schema-authoring rule for derivations:** every input a `## Prompt:`
derivation rule mentions must itself be an extracted field in the schema.
The model can fold any page content into its arithmetic, but `dg:href` can
only point at extracted elements — an un-extracted input leaves the computed
value unverifiable. A `derived_from` entry that doesn't resolve is dropped
from `dg:href` and counted in `extraction_stats.json` under
`matching.dropped_refs` (written under `--debug`); a computed field that ends
up with no `dg:href` at all is flagged by `dgml check` as
`computed_field_unattributed`.

Errors across the group: `DOCSET_NOT_FOUND`, `FILE_NOT_FOUND`,
`SCHEMA_NOT_FOUND`, `SCHEMA_INVALID`, `NO_FILES`, `VALUES_NOT_FOUND`,
`GROUNDED_CONFIG_MISSING`, `GROUNDED_CONFIG_INVALID`.

## File commands

### `dgml file add <path> [--recursive] [--on-conflict POLICY] [--text-mode MODE] [--auto-classify]`

Add a File. The source is copied into the workspace, hashed, its pages
are rendered to 300 dpi PNGs via `gs`, and per-page word boxes are
written to `page_text/` according to `--text-mode`.

`<path>` is a `.pdf`, or a convertible source (`.docx`/`.doc`/`.xlsx`/`.xls`)
when a converter is configured for its format family in the workspace
`conversion` config. A convertible source is converted to PDF at add time and
the result is **persisted** alongside the stored original at
`files/<file_id>/<stem>.pdf`; pages are rendered from it and generation reuses
it (the document is converted exactly once). With no converter configured for a
non-PDF format, the add fails with `UNSUPPORTED_FILE_TYPE`; there is no default
converter. See [Document conversion](conversion.md).

`<path>` may also be a **directory**, in which case every ingestible file
(`.pdf` plus the convertible source extensions, case-insensitive) in it is
added in a single run — see [Bulk add (a directory)](#bulk-add-a-directory)
below. `--recursive` controls whether subdirectories are walked; it is
ignored when `<path>` is a single file.

| `--on-conflict` | Behavior |
|---|---|
| `error` (default) | Fail loudly on any conflict. |
| `skip` | Return the existing record; no new record. |
| `replace` | On path-conflict: delete the old record (and its DocSet assignments) and add a new one. On hash-conflict: equivalent to `skip` (content already matches). |
| `duplicate` | Always create a new record, even when an exact duplicate exists. |

| `--text-mode` | Behavior |
|---|---|
| `digital` (default) | Extract digital text from the PDF with `pdfminer.six`. A permanent text-extraction error is recorded for files with no digital text — the File record is still created (soft fail). |
| `ocr` | Send each rendered page image to the cloud provider configured in `<workspace>/config.json`. Requires `pip install dgml[azure]` or `pip install dgml[aws]`. See "OCR configuration" below. |
| `hybrid` | Run `digital` then `ocr` and merge the two per-page results by grouping words covering the same area into overlap regions (boxes overlap on IoU > 0.5 *or* one mostly contained in the other, so split/merge tokenization is resolved as a unit). Each region is resolved as a whole: OCR-only regions are kept; digital-only regions (no overlapping OCR) are assumed invisible to the human eye and dropped; mixed regions compare both sides' concatenated text by dash-normalized Levenshtein distance — if they agree (distance ≤ 2) digital wins (its characters come straight from the PDF font, more reliable than OCR even when OCR's tokenization is finer), and if they disagree OCR wins. A page whose digital text is mostly unresolved glyphs (pdfminer `(cid:N)` sentinels) falls back to OCR entirely. Default is silent — pass the global `--verbose` flag to surface per-page warnings and the merge summary on stderr. Requires the same `ocr` workspace config as `--text-mode ocr`. Optionally, an LLM can make the per-region decision instead of this heuristic — declare a `text_extraction` section in `config.json` (e.g. a local Ollama model); see [storage-layout.md](storage-layout.md#text_extraction-optional). Any LLM failure falls back to the heuristic for that page. |

Conflict types recorded in the success payload as `conflict_kind`:

- **`hash`** — exact byte-for-byte duplicate of an existing File.
- **`path`** — different content but the same source path
  (`original_path`) as an existing File.

The `dgml file add` response also includes:

- `created` — `false` if an existing record was returned instead of creating a new one.
- `note` — human-readable explanation when the policy did something
  surprising (e.g. `replace` on a hash-conflict is a no-op since content is
  already identical).
- `page_render_error` — set if ghostscript failed or rendered a wrong page count.
- `page_count_error` — set if pypdf could not parse the PDF to extract a
  page count. The File record is still created (with `page_count: null`)
  and a permanent error is recorded; consistency check will skip retrying
  unless invoked with `--retry-errors`.
- `text_extraction_error` — set if pdfminer.six failed to parse the PDF or
  the PDF had no extractable digital text on any page. The File record is
  still created and a permanent error is recorded.
- `text_extraction` — summary object on success: `{ mode, pages_written,
  pages_with_words, total_words }`. `null` when extraction itself failed.
- `conversion_error` — set if a convertible source (docx/xlsx/…) could not be
  converted to PDF (missing converter binary/SDK, conversion failure). The
  File record is still created (with `page_count: null`) and a permanent error
  is recorded. `null` for PDFs and successful conversions.
- `classification` — present **only** when `--auto-classify` is passed.
  See "Auto-classification" below.

Error codes that can come back on `file add`:

| Code | Cause |
|---|---|
| `OCR_CONFIG_MISSING` | `--text-mode ocr` or `--text-mode hybrid` but `<workspace>/config.json` is missing or has no `ocr` section. No record is created. |
| `OCR_CONFIG_INVALID` | `<workspace>/config.json` has an `ocr` section with invalid fields. No record is created. |
| `TEXT_EXTRACTION_CONFIG_INVALID` | `--text-mode hybrid` but the optional `text_extraction` section in `<workspace>/config.json` is malformed. No record is created. |
| `UNSUPPORTED_FILE_TYPE` | Path is not a `.pdf` and is not a convertible source with a converter configured for its format family. |
| `INVALID_PDF` | File does not start with the `%PDF-` magic. |
| `CONVERSION_CONFIG_INVALID` | The `conversion` section of `<workspace>/config.json` is malformed or names an unresolvable/invalid provider. |
| `CONFLICT` | Hash- or path-conflict and `--on-conflict error`. |
| `CLASSIFICATION_CONFIG_MISSING` | `--auto-classify` was passed but `<workspace>/config.json` is missing or has no `classification` section. |
| `CLASSIFICATION_CONFIG_INVALID` | The `classification` section exists but a required field is missing or malformed. |

The classification config is a precondition for `--auto-classify`, so a
missing/invalid one is a **hard** error (exit 1) rather than a per-file
soft error — every file would otherwise report the same thing. For a bulk
directory add the config is checked once up front, so the run aborts before
any file is added.

Soft-fail codes recorded on the File rather than returned as an envelope (OCR/hybrid-specific):

- `OCR_FAILED` / `AUTH_ERROR` — provider API or credential failure during OCR (also applies to `hybrid` since it runs OCR per page). File record is created with `text_extraction_error` set and a permanent error recorded.

Soft-fail codes surfaced in the `classification.error` field (auto-classify-specific). These cover failures of the classification *call*, after a valid config is in hand — the File record is still created and exit code stays `0`:

- `CLASSIFICATION_FAILED` — the LLM call itself failed (network, malformed response, missing `litellm`, unknown DocSet id, …).
- `AUTH_ERROR` — `classification.api_key_env` names an env var that isn't set.

### Bulk add (a directory)

When `<path>` is a directory, `dgml file add` walks it for ingestible
files, lex-sorts them, and runs the same per-file pipeline on each — one
subprocess, one config load, one DocSet-store read for the whole batch
instead of per file. "Ingestible" means `.pdf` always, plus the convertible
source extensions (`.docx`/`.doc`/`.xlsx`/`.xls`, case-insensitive) **whose
format family has a converter configured** in the workspace `conversion`
config. Convertible sources with no configured converter are skipped, not
gathered — so a folder of PDFs with stray Office docs doesn't produce a pile
of per-file failures. (A malformed `conversion` config aborts the run with
`CONVERSION_CONFIG_INVALID`.) `--recursive` descends into subdirectories; the
default scans the top level only (the `find -maxdepth 1` equivalent).

`--on-conflict`, `--text-mode`, and `--auto-classify` apply per file
exactly as for a single add. `--on-conflict skip` is the recommended
bulk flag — it makes re-runs idempotent. With `--auto-classify`, a
DocSet created for one file becomes visible to the files processed
after it, so similar PDFs in the batch cluster into the same DocSet.

Each file commits independently: a single bad PDF (or a conflict under
`--on-conflict error`) is recorded in its entry and the run continues.
The command exits `0` as long as the run completes — per-file failures
are reported, not raised. Only a run-level abort (workspace not
initialized, directory unreadable) returns a non-zero error envelope.

The payload is a single envelope — the shared batch shape (`summary` count
block + per-item `results`, each carrying a `status`), matching `docset
generate`.

```json
{
  "directory": "/path/to/pdfs",
  "recursive": false,
  "summary": {
    "total": 3,
    "added": 1,
    "skipped": 1,
    "soft_failed": 1,
    "hard_failed": 1
  },
  "results": [
    {
      "status": "added",
      "path": "/path/to/pdfs/clean.pdf",
      "file": { "id": "kxlv1o15powg", "...": "..." },
      "created": true,
      "conflict_kind": null,
      "page_render_error": null,
      "page_count_error": null,
      "text_extraction_error": null,
      "conversion_error": null,
      "text_extraction": { "mode": "digital", "...": "..." },
      "note": null
    },
    {
      "status": "hard_failed",
      "path": "/path/to/pdfs/broken.pdf",
      "error": { "code": "INVALID_PDF", "message": "..." }
    }
  ]
}
```

Each entry's `status` is one of `added` / `skipped` / `soft_failed` /
`hard_failed`, matching the summary buckets. A successful entry otherwise
carries the same fields as a single `file add` response (plus `path`, and
`classification` when `--auto-classify` is set). A hard-failed entry
has `status`, `path`, and an `error` object instead of a `file` record.

`summary` counts (they sum to `total`):

| Field | Meaning |
|---|---|
| `total` | Ingestible files found (`.pdf` + convertible sources; other extensions are ignored, not counted). |
| `added` | Newly created File records with no recorded soft error. |
| `skipped` | Existing records returned via `--on-conflict skip`/`replace` (`created: false`). |
| `soft_failed` | Added, but with a `page_render_error`, `page_count_error`, `text_extraction_error`, or `conversion_error` recorded. |
| `hard_failed` | The add raised (bad PDF, conflict under `--on-conflict error`, …); the entry carries an `error` object. |

Run `dgml check` afterward as the authoritative health signal for the
whole workspace.

## Auto-classification

`--auto-classify` on `dgml file add` uses a configured vision LLM to look at
the new file's rendered page images and either assign it to an existing
DocSet or create a new one. Configure the model via the `classification`
section in `<workspace>/config.json`:

```json
{
  "classification": {
    "model": "gemini/gemini-3.1-flash-lite",
    "max_pages": 3,
    "api_key_env": "GEMINI_API_KEY"
  }
}
```

| Field | Required | Meaning |
|---|---|---|
| `model` | yes | `<provider>/<model>` in [litellm](https://docs.litellm.ai/docs/providers) form — e.g. `gemini/gemini-3.1-flash-lite`, `anthropic/claude-opus-4-7`, `openai/gpt-4o`. |
| `max_pages` | no (default `3`) | How many rendered page images (`page_images/page_1.png` …) to send to the LLM. Cap is per-classification cost: 1 is the cheap setting, 4+ is the thorough one. |
| `api_key` | no | Optional literal API key. Use only on per-developer workspaces (config.json isn't checked in). Mutually exclusive with `api_key_env`. |
| `api_key_env` | no | Optional name of the env var to read the API key from. Mutually exclusive with `api_key`. When neither is set, litellm uses its built-in per-provider lookup (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …). When `api_key_env` references an unset env var, `AUTH_ERROR` is raised. |

The `litellm` SDK ships in the base install — no extra needed.

The LLM is forced to pick exactly one of two tools:

- `assign_to_existing_docset(docset_id)` — the new file would answer
  the same `key_questions` as one of the existing DocSets above (i.e.
  a single extraction schema would work for both). Topical overlap
  alone is not enough.
- `create_new_docset(name, description, key_questions)` — no existing
  DocSet fits. The LLM proposes a document-type-specific name (e.g.
  "Property Tax Bill", not "Property Tax Records"), a one-sentence
  description, and 3-7 concrete questions the first pages of this
  document type can answer. The `key_questions` are persisted on the
  new DocSet and shown to future classifications.

Classification runs **after** the file is added, and only when `created`
is `true`. Re-runs on a duplicate (`--on-conflict skip`) skip the LLM
call entirely: `classification.performed` is `false` and the existing
record is returned untouched.

The `classification` payload block:

```json
"classification": {
  "performed": true,
  "model": "gemini/gemini-3.1-flash-lite",
  "decision": "existing",
  "docset_id": "k7q3xb91pmrf",
  "docset_created": false,
  "docset_name": "Vendor Invoices",
  "docset_key_questions": [
    "What is the vendor name?",
    "What is the invoice total?",
    "What is the invoice date?"
  ],
  "error": null
}
```

`docset_key_questions` echoes the assigned DocSet's `key_questions`
(empty list for DocSets created without them). When `decision`
is `"new"`, this is the list the LLM just proposed and that has been
persisted on the freshly-created DocSet.

When the file already existed (`created: false`):

```json
"classification": {
  "performed": false,
  "reason": "file already existed; classification skipped"
}
```

Classification works even when digital text extraction failed (scanned
PDFs), because it operates on the rendered page images — the same PNGs
the OCR mode uses. It does **not** work when page rendering itself failed
(`page_render_error` set, `page_images/` empty); in that case
`classification.error` is `CLASSIFICATION_FAILED: no page images found …`
and the file is left unassigned.

## OCR configuration

When `--text-mode ocr` is used, the provider and its settings come from
`<workspace>/config.json`. A workspace is per-developer / not checked
into source control, so secrets *may* live directly in `config.json`
(`api_key`) — but the safer default is to use `api_key_env` and keep
the key in an environment variable.

### Azure Document Intelligence

```json
{
  "ocr": {
    "provider": "azure",
    "endpoint": "https://<resource>.cognitiveservices.azure.com/",
    "api_key_env": "AZURE_DOCINTEL_KEY"
  }
}
```

Auth resolution, in order of precedence:

- `api_key` (optional, literal string) — used verbatim if set.
- `api_key_env` (optional, env var **name**) — env var is read and used.
- Neither set — falls back to `DefaultAzureCredential` (env vars,
  managed identity, `az login`, …).

`api_key` and `api_key_env` are mutually exclusive; setting both yields
`OCR_CONFIG_INVALID`. A referenced-but-unset env var produces
`AUTH_ERROR`.

### AWS Textract

```json
{
  "ocr": {
    "provider": "aws",
    "region": "us-east-1",
    "profile": "default"
  }
}
```

- `profile` is **optional**. When omitted, boto3's default credential
  chain is used (env vars, `~/.aws/credentials`, IAM role, SSO).
- Textract is invoked once per rendered page image (5 MB sync limit).

## Managing secrets locally

The CLI reads secrets from environment variables. The lookup path
depends on the feature and is **not** shared across OCR and
classification:

- **OCR — Azure**: reads the env var named by `ocr.api_key_env`. If that
  field is omitted, falls back to the Azure SDK's `DefaultAzureCredential`
  chain (env vars like `AZURE_CLIENT_ID`/`AZURE_TENANT_ID`, managed
  identity, `az login`, …). The classification env vars below are not
  consulted.
- **OCR — AWS**: reads from the boto3 credential chain — `ocr.profile` if
  set, otherwise standard AWS env vars (`AWS_ACCESS_KEY_ID`, …),
  `~/.aws/credentials`, IAM role, SSO. The classification env vars below
  are not consulted.
- **Classification only**: reads the env var named by
  `classification.api_key_env` if set. If that field is omitted, litellm
  looks up its own per-provider env var based on the `model` prefix —
  `GEMINI_API_KEY` for `gemini/…`, `OPENAI_API_KEY` for `openai/…`,
  `ANTHROPIC_API_KEY` for `anthropic/…`, etc. These provider env vars are
  classification-specific; setting `GEMINI_API_KEY` does nothing for OCR.

The CLI does **not** auto-load a `.env` file. Pick whichever of these
patterns your team already uses to populate the environment:

- **Manual export** (one shell session):
  ```bash
  export AZURE_DOCINTEL_KEY="..."
  uv run dgml file add scan.pdf --text-mode ocr
  ```
- **`direnv`** — drop an `.envrc` in the workspace directory; it loads
  automatically on `cd` and unloads on leave.
- **Sourced env file** — keep a gitignored file of `KEY=value` lines
  (`.env`, `config.env`, `secrets.env` — `source` doesn't care which),
  then:
  ```bash
  set -a; source config.env; set +a
  uv run dgml file add scan.pdf --text-mode ocr
  ```
- **Secret-manager wrappers**:
  ```bash
  op run --env-file=.env -- uv run dgml file add scan.pdf --text-mode ocr   # 1Password
  aws-vault exec my-profile -- uv run dgml file add ...                     # aws-vault
  ```
- **Azure token auth** — omit `api_key_env` from `config.json` entirely
  and run `az login`. `DefaultAzureCredential` will pick up the session
  with no env vars needed.

The CLI doesn't auto-load `.env` deliberately: doing so would surprise
users about file location and override precedence and would pull in a
dependency. Keeping the contract at "we read `os.environ`" lets teams
plug in whichever workflow they already trust.

### `dgml file list`

### `dgml file show <file_id>`

### `dgml file delete <file_id>`
Removes the File and any DocSet assignments to it. Does not affect other
Files or DocSets.

## DGMLX commands

A **DGMLX bundle** is the Merkle-attested, portable export of a file's
*DGML version* — the set of on-disk artifacts that together constitute
everything DGML knows about it: the source document (a `.pdf`, or the
`.docx`/`.xls`/… it was converted from), one image per page, and — when a
DocSet is named — that DocSet's `full-schema.rnc` (and
`extraction-schema.rnc`, when set) plus the file's
`<stem>.dgml.xml`. `schema.json` itself is not bundled: the RNC render is
lossless over it, so attesting the `.rnc` covers the JSON exchange form. The `dgmlx` commands roll those artifacts up to a single
RFC-6962 Merkle root and package them into a portable,
**filename-independent** bundle.

The per-page text JSONs under `page_text/` (the token files from text
extraction) are an intermediate artifact and are **not** included in the
bundle or its Merkle root.

The bundle's ordering is driven by the `META-INF/dgml-attestation.xml`
attestation file, not by filenames. Each page artifact carries an explicit
`number` attribute; the verifier orders leaves by that number, so the
artifact files inside the bundle can be named anything. The attestation
file is **not** part of the attestation — it only records the relative
paths, the per-page numbers, the rendering provenance, and the Merkle root
so a holder of just the bundle can re-verify it.

Canonical leaf order: source → page images (by `number`) → full schema →
extraction schema → DGML XML. Missing slots are simply absent (a smaller
version), not an error.

#### `META-INF/dgml-attestation.xml` (the attestation file)

This single namespaced file is both the **manifest** (artifact inventory +
ordering) and the **provenance record**. It carries:

- the **Merkle root** (`<merkle-root>`, with the `algorithm` attribute);
- the **workspace identity** — `file-id` (always present) and `docset-id`
  (present only when exported with `--docset`);
- the **rendering provenance** from `file.json` — `page-image-dpi`,
  `page-image-renderer`, and (only for a non-PDF source converted to PDF)
  `pdf-converter`. A field absent from `file.json` is omitted, not emitted
  empty;
- the `<artifacts>` **inventory** — each leaf's role mapped to its relative
  path, with per-page `number` attributes.

These are metadata for attribution and verification — the file itself is
not a leaf of the Merkle root.

```xml
<?xml version='1.0' encoding='utf-8'?>
<dgml-attestation xmlns="http://dgml.io/ns/attestation" version="1"
                  page-image-dpi="300" page-image-renderer="ghostscript"
                  pdf-converter="LibreOffice"
                  file-id="f00000000abc" docset-id="ds0000000xyz">
  <merkle-root algorithm="sha256">9f1c…64-hex…</merkle-root>
  <artifacts>
    <source>source/contract.docx</source>
    <page-images>
      <page-image number="1">page_images/page_1.png</page-image>
      <page-image number="2">page_images/page_2.png</page-image>
    </page-images>
    <full-schema>full-schema.rnc</full-schema>
    <dgml-xml>contract.dgml.xml</dgml-xml>
  </artifacts>
</dgml-attestation>
```

The hash algorithm is not user-selectable; `<merkle-root>`'s `algorithm`
attribute records what was used (always `sha256` — both leaf hashes and
Merkle inner nodes). Verification rejects an attestation file recording any
other algorithm (`ATTESTATION_INVALID`), and treats one without the attribute
(written by older versions) as `sha256`.

#### OPC packaging + the `.dgmlx` archive

The bundle is also shaped as an **OPC package** (Open Packaging
Conventions, ECMA-376 Part 2 — the same container family as `.docx`):

- `[Content_Types].xml` (§10.1) — a content-type registry with one
  `<Default>` per extension present (`pdf`/`docx`/…, `png`, `json`,
  `xml`, `rels`). No per-part `<Override>` is needed; an unknown
  extension maps to `application/octet-stream`. This file is not itself a
  part and is not listed inside itself.
- `_rels/.rels` (§9) — the package relationships part. Up to three
  relationships: the **main document**
  (`http://dgml.io/ns/relationships/main-document`) always points at the
  `source/` original; **dgml-xml** (`http://dgml.io/ns/relationships/dgml-xml`)
  points at the generated `<stem>.dgml.xml` when the export is docset-scoped
  and the XML exists; and **attestation**
  (`http://dgml.io/ns/relationships/attestation`) points at
  `META-INF/dgml-attestation.xml` — the verification entry point. Targets
  are percent-encoded, so a source named `My Doc.docx` is referenced as
  `source/My%20Doc.docx`.

The package is zipped into a portable `<stem>.dgmlx` archive (stem = the
source filename's stem) in `<dir>`, with `[Content_Types].xml` as the first
entry. The archive is built from the explicit part list, so it never packs
itself (or a prior run's archive) back in. The OPC parts and the archive
don't participate in the Merkle root.

### `dgml dgmlx export <file_id> --output-dir <dir> [--docset <docset_id>] [--unpacked]`

Attests the file's current artifacts and writes the DGMLX bundle (artifacts
+ `META-INF/dgml-attestation.xml` + `[Content_Types].xml` + `_rels/.rels`).
The two output modes are mutually exclusive:

- **default** — only the `<stem>.dgmlx` archive is written to `<dir>`; the
  bundle is staged in a temp directory and removed after zipping.
- **`--unpacked`** — the loose bundle tree is written into `<dir>` and **no
  archive is produced**.

With `--docset`, the docset-scoped artifacts (`full-schema.rnc`,
`<stem>.dgml.xml`) are included if present; without it, only the file-side
artifacts (source, page images) are attested.

Success payload (exit `0`):

```json
{
  "file_id": "f00000000abc",
  "docset_id": null,
  "output_dir": "/path/to/bundle",
  "dgmlx": "/path/to/bundle/contract.dgmlx",
  "root": "9f1c…64-hex…",
  "slots": ["source", "page_image[1]", "page_image[2]"]
}
```

The payload carries exactly one output path: `dgmlx` (the archive) by
default, or `attestation` (the loose `META-INF/dgml-attestation.xml` path)
with `--unpacked`.

### `dgml dgmlx verify <path>`

`<path>` is either a `.dgmlx` archive or an unpacked bundle directory. An
archive is extracted to a temporary directory first; a directory is read in
place. Either way verify reads `META-INF/dgml-attestation.xml`, re-hashes the
referenced artifacts in canonical order (page ordering from the `number`
attributes, never the filenames), recomputes the Merkle root, and compares it
to the recorded root.

Success payload:

```json
{
  "path": "/path/to/contract.dgmlx",
  "file_id": "f00000000abc",
  "docset_id": null,
  "valid": true,
  "expected_root": "9f1c…",
  "computed_root": "9f1c…",
  "slots": ["source", "page_image[1]", "page_image[2]"]
}
```

Exit codes mirror `dgml check`: `0` when the bundle verifies, `2` when
it verifies-but-fails (a tampered or altered artifact → roots differ,
`valid: false`), and `1` (error envelope, `ATTESTATION_INVALID`) when the
bundle is structurally broken — `<path>` is neither a directory nor a
readable `.dgmlx` archive, a missing/malformed
`META-INF/dgml-attestation.xml`, a referenced artifact absent from disk,
or a bad/duplicate page `number`.

## Node commands

Element-level attestation over a file's generated DGML XML. Every
element of the document is a Merkle leaf (see
[merkle-attestation.md](merkle-attestation.md)); the `node` commands
export the attestation payload for one element — its hash, the
document tree's Merkle root, and the inclusion proof connecting the
two — and later re-verify it. Node attestation is docset-scoped
(`--docset` is required) and reads the same canonical
`docsets/<id>/files/<id>/<stem>.dgml.xml` artifact the DGMLX bundle's
`dgml_xml` slot hashes.

An element is addressed by exactly one coordinate:

- `--leaf <n>` — 0-based DFS pre-order index (the Merkle leaf index).
- `--xpath <expr>` — an XPath matching exactly one element, resolved
  against the document's own namespace prefixes. The UX tree view's
  "Copy XPath" emits a canonical positional form
  (`/dg:chunk/docset:Entry[2]/docset:Amount`).
- `--child-path <path>` — slash-separated 0-based child-element indices
  walked from the document root (e.g. `1/1` = "the root's 2nd child
  element's 2nd child element"), skipping comments/PIs at every level.
  This is the coordinate a DOM tree view naturally has (a browser's
  `Element.children`) when a caller has a node reference but no
  ready-made XPath or leaf index for it. An empty string selects the
  document root.

### `dgml node export <file_id> --docset <docset_id> (--leaf <n> | --xpath <expr> | --child-path <path>)`

Success payload (exit `0`):

```json
{
  "file_id": "f00000000abc",
  "docset_id": "ds0000000xyz",
  "leaf_index": 3,
  "leaf_count": 4,
  "xpath": "/dg:chunk/docset:Entry/docset:Amount",
  "node_hash": "ab12…64-hex…",
  "root_hash": "9f1c…64-hex…",
  "proof": {
    "leaf_hash": "ab12…",
    "leaf_index": 3,
    "leaf_count": 4,
    "path": [{"sibling": "77aa…", "side": "L"}]
  },
  "node_xml": "<docset:Amount xmlns:docset=\"…\">100</docset:Amount>"
}
```

`node_hash` equals `proof.leaf_hash` and is the SHA-256 of `node_xml`'s
UTF-8 bytes (`node_xml` is the element's exclusive-C14N serialization,
so a holder can re-hash it directly). `root_hash` is the Merkle root of
the whole DGML XML tree — the same value the DGMLX bundle records as
the `dgml_xml` slot's leaf hash.

### `dgml node prove <file_id> --docset <docset_id> --proof <path|->`

`--proof` takes a `node export` payload (any JSON object carrying
`root_hash` and `proof`); `-` reads stdin. The element at the proof's
`leaf_index` in the workspace's *current* DGML XML is re-hashed and the
inclusion proof re-walked.

Success payload:

```json
{
  "file_id": "f00000000abc",
  "docset_id": "ds0000000xyz",
  "leaf_index": 3,
  "xpath": "/dg:chunk/docset:Entry/docset:Amount",
  "expected_root": "9f1c…",
  "expected_node_hash": "ab12…",
  "computed_node_hash": "ab12…",
  "valid": true
}
```

On a failed proof, comparing `computed_node_hash` against
`expected_node_hash` distinguishes "this node changed" from "the tree
around it changed". Exit codes mirror `dgml dgmlx verify`: `0` proven,
`2` computed-but-mismatched (`valid: false`), `1` (error envelope) for
structural problems — unknown ids, no generated XML, a malformed proof
payload (`INVALID_ARGUMENT`), or a document so restructured the leaf
index no longer exists.

## Discovery commands

### `dgml discover <file_id> --docset <docset_id> [options]`

Analyse a File's generated DGML XML, group element types by structural role,
and filter them so agents and humans can quickly identify which tag types are
worth staking (via `dgml node export` / `dgml stake node`).

The command works on the grounded DGML XML when available
(`<stem>.dgml.grounded.xml`), falling back to the plain XML.  Each result
entry carries representative element **samples**: the `depth_first` field
is the 0-based DFS pre-order leaf index that `dgml node export --leaf <n>`
accepts directly.

**Flags**

| Flag | Default | Description |
|---|---|---|
| `--docset <id>` | required | DocSet the DGML XML was generated in. |
| `--filter <name>` | `all` | Filter to apply; see the filter table below. |
| `--samples N` | `2` | Maximum element samples per tag type. |
| `--include-structural` | off | Include `dg:`-namespace framework elements in results. |
| `--full` | off | Full output: includes `role`, `filters`, `depth_first`, `page`, and XML attributes in each sample. Default strips attributes and omits those fields. |
| `--search <term>` | — | Case-insensitive substring filter on tag names (e.g. `date`, `price`). |
| `--search-content <term>` | — | Case-insensitive substring filter on sample XML text content. |

**Algorithmic filters** (no LLM call, identical results to the HTML app)

| Filter | What it selects |
|---|---|
| `all` | Every non-root tag type in the document (default). |
| `values` | Tags whose instances are mostly leaf text nodes (`role = leaf-value` or `textRatio ≥ 0.5`). Best for staking individual field values. |
| `sections` | Tags with high betweenness or high ancestor coverage — structural section headers. |
| `density` | Information-dense branches: high token-per-depth, high child type variety, or many leaf descendants. |
| `patterns` | Tags with the highest structural entropy — most variable child composition. |

**Semantic filters** (require a generation LLM config in `<workspace>/config.json`)

| Filter | Selects tags the LLM categorises as … |
|---|---|
| `Who` | Parties, entities, persons. |
| `When` | Dates, durations, periods. |
| `Amounts` | Monetary values, quantities, rates. |
| `Definitions` | Defined terms, glossary entries. |
| `Rules` | Conditions, obligations, prohibitions. |

If the generation LLM config is absent or the call fails and a semantic
filter was requested, `dgml discover` warns on stderr and falls back to
`All` — it does **not** hard-fail.

**Success output (exit 0)**

```json
{
  "file_id": "f00000000abc",
  "docset_id": "ds0000000xyz",
  "filter": "values",
  "tag_count": 2,
  "tags": [
    {
      "tag": "LiabilityCap",
      "count": 1,
      "role": "leaf-value",
      "filters": ["values"],
      "samples": [
        {
          "depth_first": 5,
          "xpath": "/dg:chunk/docset:IndemnificationClause/docset:LiabilityCap",
          "page": 2,
          "xml": "<docset:LiabilityCap xsi:type=\"decimal\" dg:value=\"500000\" dg:origin=\"2 460 410 1800 455\">$500,000</docset:LiabilityCap>"
        }
      ]
    }
  ]
}
```

**Field notes**

- `role` — `"leaf-value"` | `"container"` | `"hybrid"` | `"mixed"` (same
  roles as the HTML app).
- `filters` — all algorithmic filters the tag would pass (useful when
  `--filter all` is used and you want to know a tag's category).
- `samples[].depth_first` — pass directly to `dgml node export --leaf <n>`
  to get the attestation payload for that specific element.
- `samples[].page` — first token of the element's `dg:origin` attribute
  (the page number); `null` when no origin is present (ungrounded XML).
- `samples[].xml` — the element serialized as XML with namespace
  declarations stripped (for display; not the canonical C14N form).

**Error codes**

`FILE_NOT_FOUND`, `DOCSET_NOT_FOUND`, `NOT_FOUND` (no generated DGML XML),
`INVALID_ARGUMENT` (bad filter name or empty ids).

## Chain attestation commands

Anchor a DGMLX bundle's Merkle root, or a single node's hash, directly
on an EVM chain — no MCP server. These commands require the `chain`
extra (`pip install dgml[chain]`); without it they return a
`MISSING_EXTRA` error envelope. The local Merkle/hashing is identical to
the `dgmlx`/`node` commands; these add the chain transport (a stdlib
JSON-RPC client, anchor-precompile ABI encoding, EIP-1559 signing).

The anchored checksum, URI, and metadata are **public on-chain** —
never put document content in them. Node records expose only hashes.

**Configuration & key handling**

- `--chain <name>` selects a configured chain (env `NVNM_CHAIN`,
  default `nvnm-testnet`). `nvnm-testnet` and `nvnm-mainnet` are
  built-in; add others with `dgml chain add`.
- `--registry <name>` is the registry **name** on the chain (env
  `NVNM_REGISTRY`). Create one with `dgml registry create`.
- `--from <addr>` is the sender EVM address (env `NVNM_FROM_ADDRESS`);
  it defaults to the address controlled by the keyring key.
- The signing key lives in the OS keyring (service `nvnm-wallet`,
  account `default`; override with `NVNM_KEY_SERVICE` /
  `NVNM_KEY_ACCOUNT`). It is never read except at signing time and never
  printed. Signing refuses if the key does not control `--from`.
- Write commands (`stake`, `registry create`) build, sign, **and
  broadcast** by default; `--dry-run` stops after signing and emits the
  unsigned + signed transaction for review without spending gas.
  `--legacy` uses a type-0 transaction instead of EIP-1559.

### `dgml chain {list,show,add,remove}`

Manage chain configs. Custom chains persist to a JSON file resolved
`--chain-config` → `$DGML_CHAINS` → `<workspace>/chains.json`. Built-in
chains cannot be removed or redefined.

```bash
dgml chain list
dgml chain show nvnm-testnet
dgml chain add --name local --rpc-url http://localhost:8545 --chain-id 1337 \
  [--anchor-address 0x…] [--explorer https://…] [--native-token TOKEN]
dgml chain remove local
```

A chain entry: `{name, rpc_url, chain_id, anchor_address, explorer?,
native_token?, builtin}`.

### `dgml wallet status --chain <name> [--address <addr>]`

Read-only balance + pending nonce. `--address` defaults to the keyring
key's address. Payload: `{chain, address, balance_wei, balance_eth,
native_token, nonce, funded}`.

### `dgml registry {create,list} --chain <name>`

```bash
dgml registry create --chain nvnm-testnet --name my-registry \
  --description "…" [--metadata '{}'] [--from 0x…] [--dry-run] [--legacy]
dgml registry list --chain nvnm-testnet [--name my-registry]
```

`create` anchors a new registry (the creator becomes its admin) and on
success returns `{chain, registry, from, tx_hash, broadcast,
receipt_status, block_number, explorer_url}`. `list` decodes the
on-chain `registries` view.

### `dgml stake file <file_id> [--docset <id>] [--unpacked] --chain <name> --registry <name>`

Export the file's DGMLX bundle, anchor its Merkle root as the record
checksum (URI `dgmlx://<file_id>[/<docset_id>]`), broadcast, await the
receipt, then fetch and save the anchored record to `record.json` in the
output dir. Success payload includes `checksum` (the Merkle root),
`uri`, `tx_hash`, `receipt_status`, `record`, `record_path`,
`explorer_url`, and `bundle_dir` (the output directory). By default the
bundle is written as a single portable `<stem>.dgmlx` archive whose path
is reported in `dgmlx`; pass `--unpacked` to write the loose bundle tree
instead, in which case the payload reports the loose attestation-file path
in `attestation` (and no `dgmlx`). `--output-dir` overrides the output
location (default `<workspace>/dgmlx-bundles/<ids>`); the archive (or loose
tree) and `record.json` are written there.

The saved record path is reported in `record_path`; keep it for offline
proving. Bundle records save as `record.json`; node records save as
`record-node-<leaf>.json` so a file's bundle and its nodes never clobber
each other in the shared output dir.

### `dgml stake node <file_id> --docset <id> (--leaf <n> | --xpath <expr>) --chain <name> --registry <name>`

Anchor one DGML XML element: the record checksum is the node hash and
the metadata carries `{kind: "dgml-node", root_hash, proof}` (the URI
gains a `#<leaf>` fragment). Same broadcast/confirm/save flow as `stake
file`.

### `dgml prove {file,node} --chain <name> (--registry <name> --checksum <hex> | --record-json <path|->)`

Re-verify an anchored record against the current workspace. Supply the
record either by looking it up on-chain (`--registry` + `--checksum`) or
from a saved `--record-json` (`-` for stdin). `prove file` re-exports
the bundle and compares the recomputed Merkle root to the anchored
checksum; `prove node` re-hashes the element and re-walks its proof
against the recorded root. Exit codes mirror `dgmlx verify`: `0` proven,
`2` mismatch (`valid: false`), `1` for structural errors.

## Error code reference

Every value the CLI can put in an `error.code` field, plus the soft-fail
codes it records on a File (surfaced in `dgml check` and in `file add`
payload fields like `text_extraction_error`, never in a top-level `error`
envelope). **Hard** = emitted as the stderr `error` envelope with exit `1`;
**soft** = recorded/returned in a payload field, exit unaffected.

| Code | Kind | Meaning |
|---|---|---|
| `WORKSPACE_NOT_INITIALIZED` | hard | A command that needs a workspace ran against a directory with no workspace layout (run `dgml workspace create`). |
| `LOCAL_CONFIG_MISSING` | hard | The peer `local_config.json` is absent when a library caller invokes `Workspace.write_config_from_local` directly. `dgml workspace create` does not raise this — it seeds `local_config.json` from the bundled template instead. |
| `MISSING_EXTRA` | hard | A command needs an optional extra that isn't installed (e.g. `dgml[clustering]`). |
| `INVALID_ARGUMENT` | hard | An argument is malformed or empty (e.g. blank `file_id`, unreadable `--proof`). |
| `INTERNAL_ERROR` | hard | Unexpected exception; the message is a short, single-line `<ExcType>: <msg>` (capped, whitespace collapsed). Pass `--verbose` (or set `DGML_DEBUG=1`) for the full stderr traceback. |
| `NOT_FOUND` | hard | Generic not-found (base for the specific codes below). |
| `DOCSET_NOT_FOUND` | hard | No DocSet with the given id. |
| `FILE_NOT_FOUND` | hard / soft | A File id, assignment, or source is missing. Soft as a per-item `results` entry in `docset generate`/`ground`. |
| `UNSUPPORTED_FILE_TYPE` | hard | `file add` path is neither a PDF nor a convertible source. |
| `INVALID_PDF` | hard | File does not start with the `%PDF-` magic. |
| `CONFLICT` | hard | Hash- or path-conflict under `--on-conflict error`. |
| `CONVERSION_CONFIG_INVALID` | hard | The `conversion` config section is malformed. |
| `CONVERSION_FAILED` | hard / soft | A docx/xlsx→PDF conversion failed (soft as `conversion_error` on a bulk add entry). |
| `OCR_CONFIG_MISSING` | hard | `--text-mode ocr`/`hybrid` with no `ocr` config section. |
| `OCR_CONFIG_INVALID` | hard | The `ocr` config section has invalid fields. |
| `OCR_FAILED` | soft | Provider API failure during `--text-mode ocr`/`hybrid`; recorded on the File (`text_extraction_error`). |
| `TEXT_EXTRACTION_CONFIG_INVALID` | hard | The optional `text_extraction` (hybrid-merge) config is malformed. |
| `STYLE_CONFIG_INVALID` | hard | The optional `style` (image-based `dg:style` for OCR files) config section is malformed; fails `generate` up front. |
| `AUTH_ERROR` | hard / soft | A referenced API-key env var is unset (soft in `classification.error`). |
| `CLASSIFICATION_CONFIG_MISSING` | hard | `--auto-classify` with no `classification` config. |
| `CLASSIFICATION_CONFIG_INVALID` | hard | The `classification` config has a missing/invalid field. |
| `CLASSIFICATION_FAILED` | soft | The classification LLM call failed; lands in `classification.error`. |
| `CLUSTERING_CONFIG_INVALID` | hard | The optional `clustering` config section failed validation. |
| `GROUNDING_FAILED` | soft | Grounding a file failed; surfaces as `grounded: false` with a `grounding_error` on that file's `docset generate` result entry. |
| `GENERATION_FAILED` | soft | `docset generate` produced no output for a file (transcription failed) or two files shared a filename; per-item `failed` entry in `results`. |
| `SCHEMA_NOT_FOUND` | hard | An `extraction` command needs an `extraction-schema.rnc` the DocSet doesn't have. |
| `SCHEMA_INVALID` | hard | A schema passed to `extraction set-schema` is not valid RNC (within the supported subset) or not a JSON object. |
| `NO_FILES` | hard | `extraction generate-schema` has no sample files (empty DocSet and no `--from-file`). |
| `VALUES_NOT_FOUND` | hard | `extraction get-values` ran before `extraction extract` for that file. |
| `GROUNDED_CONFIG_MISSING` | hard | An `extraction` command needs a `grounded` config section that is absent. |
| `GROUNDED_CONFIG_INVALID` | hard | The `grounded` config section has a missing/invalid field. |
| `SCHEMA_GENERATION_FAILED` | hard | The schema-generation LLM call failed or returned a non-object. |
| `VALUES_EXTRACTION_FAILED` | hard | The value-extraction pipeline failed. |
| `CHAIN_CONFIG` | hard | A chain config is missing/invalid, or `dgml chain add/remove` hit a bad/built-in chain. |
| `CHAIN_RPC` | hard | A JSON-RPC call to the chain failed (network, bad RPC URL, node error). |
| `CHAIN_TX_REVERTED` | hard | A broadcast `stake`/`registry create` transaction reverted on-chain. |
| `WALLET_KEY_MISSING` | hard | No signing key in the OS keyring, or it doesn't control `--from`. |
| `RECORD_NOT_FOUND` | hard | `prove` could not find the anchored record (bad checksum/registry). |
| `MANIFEST_INVALID` | hard | A `dgmlx verify` bundle is structurally broken (missing/duplicate page number, absent artifact). |
| `GHOSTSCRIPT_NOT_FOUND` | soft | The ghostscript binary (`gs`, or `gswin64c`/`gswin32c` on Windows) is not on `PATH`; recorded as a page-render failure. |
| `PAGE_RENDER_FAILED` | soft | ghostscript failed to render a page; recorded on the File (`page_render_error`). |
| `PDF_SLICE_FAILED` | soft | A PDF page-slice operation failed during generation. |
| `TEXT_EXTRACTION_FAILED` | soft | pdfminer.six extracted no digital text; recorded (`text_extraction_error`). |
| `CORRUPT_METADATA` | hard / soft | A `file.json`/`docset.json` is not valid JSON (also reported by `dgml check`). |
| `NOT_IMPLEMENTED` | hard | A requested mode/path is not implemented. |
| `DGML_ERROR` | hard | Generic base code; specific codes above are preferred. |

Codes that read as soft above are the same identifiers, just delivered in a
payload field instead of the `error` envelope — see the `dgml check` section
for the workspace-health view and the `dgml file add` section for the
per-file soft-fail fields.

## System requirements

- Python 3.11+
- Ghostscript (`gs`) — installed system-wide for page-image rendering.
  See [CLAUDE.md](../CLAUDE.md) for the licensing rationale (ghostscript
  is AGPL but invoked as a subprocess; it is not bundled with `dgml`).

## Examples for an LLM agent

Add a file, capture its ID, assign it to a docset:

```bash
dgml init
ds=$(dgml docset create --name "Q2 contracts" | jq -r .id)
fid=$(dgml file add /tmp/example.pdf | jq -r .file.id)
dgml docset add-file "$fid" --docset "$ds"
```

Add a file and let the LLM pick (or create) the right DocSet:

```bash
dgml init
# Assumes <workspace>/config.json has a `classification` section and
# GEMINI_API_KEY (or the env var named in `classification.api_key_env`)
# is set in this shell.
payload=$(dgml file add /tmp/example.pdf --auto-classify)
ds=$(jq -r .classification.docset_id <<<"$payload")
created=$(jq -r .classification.docset_created <<<"$payload")
echo "assigned to docset $ds (newly created: $created)"
```

Bulk-add a directory of PDFs and assign each to one DocSet:

```bash
dgml init
ds=$(dgml docset create --name "Imported PDFs" | jq -r .id)
payload=$(dgml file add /path/to/pdfs --on-conflict skip)   # one call, one envelope
for fid in $(jq -r '.results[] | select(.file) | .file.id' <<<"$payload"); do
  dgml docset add-file "$fid" --docset "$ds"
done
echo "summary:"; jq .summary <<<"$payload"
```

Recover from a failed batch import:

```bash
dgml check                 # see what's broken
dgml check --retry-errors  # try the failed render(s) again
```
