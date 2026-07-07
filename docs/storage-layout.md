# DGML Workspace Storage Layout

A DGML workspace is a directory tree on disk. Everything DGML reads and
writes lives under one root directory.

## Bounding-box convention

Every bounding box DGML stores — in `page_text/page_N.json` and in the
`dg:origin` attributes of `<stem>.dgml.xml` (both the generated document tree
and the `dg:extraction` element) — uses **one**
convention: integer **image pixels** `[left, top, right, bottom]`,
top-left origin, at 300 dpi relative to the page's
`page_images/page_N.png`. Page is carried in a sibling `page_number`
field for structured forms, or as a leading integer inside each
`dg:origin` box (`<page> <x1> <y1> <x2> <y2>`, space-separated) since one
element can span pages.

## Resolving the workspace root

The root is determined in this order:

1. `--workspace <path>` CLI flag (or `Workspace.resolve(<path>)` in code).
2. The `DGML_HOME` environment variable.
3. Default: `./dgml-workspace` (relative to the current working directory).

`dgml workspace create --organization <org>` (or `Workspace.init()` in code)
creates the directory layout for a fresh workspace and records its identity in
`workspace.json`. It seeds the shared `local_config.json` from the bundled
template when absent, so `dgml init` first is **optional** — run `init`
separately only when you want to review/edit the shared config before creating
any workspace (the "configure once, create many" flow). The CLI refuses to
operate on an uninitialized workspace except for `init` and `workspace create`.
See
[the three-layer config model](#where-config-comes-from--the-three-layer-model)
for how config flows from the bundled template into each workspace.

## Directory structure

```
<workspace_root>/
├── workspace.json                    # { name, organization } — written by `workspace create`
├── config.json                       # OCR / LLM / clustering settings (optional)
├── usage.jsonl                       # LLM call event log (optional)
├── docsets/
│   └── <docset_id>/                  # 12-char base-36 ID
│       ├── docset.json               # { id, name, description, key_questions }
│       ├── extraction-schema.rnc      # grounded extraction schema, RELAX NG Compact (optional)
│       ├── schema.json               # generation tag schema, written by `generate` (present after generation)
│       ├── full-schema.rnc           # schema.json as RELAX NG Compact, written by `generate` (see below)
│       └── files/
│           └── <file_id>/            # marker dir; <stem>.dgml.xml lands here
│                                     # (generated tree and/or dg:extraction),
│                                     # plus its grounded/stats siblings (below)
└── files/
    └── <file_id>/                    # 12-char base-36 ID
        ├── <original_filename>       # source copied in (a .pdf, or a
        │                             #   convertible source like .docx/.xlsx)
        ├── <stem>.pdf                # converted PDF — only when the source was
        │                             #   not already a PDF; what pages/text and
        │                             #   generation use (see docs/conversion.md)
        ├── file.json                 # metadata (see schema below)
        ├── page_images/              # 300 dpi PNG page renders (cacheable; see below)
        │   ├── page_1.png
        │   └── page_2.png
        ├── page_text/                # one JSON of word boxes per page
        │   ├── page_1.json
        │   └── page_2.json
        └── errors.json               # recorded fatal errors (optional)
```

IDs are 12 lowercase alphanumerics — `~62` bits of entropy each, generated
with `secrets.choice` ([packages/dgml/src/dgml/ids.py](../packages/dgml/src/dgml/ids.py)).

## Page-image render cache (`$DGML_PAGE_CACHE`, optional)

Rendering `page_images/` shells out to ghostscript, which dominates the cost
of `dgml file add`. The render is a pure function of the PDF bytes (plus the
fixed 300 dpi and renderer), so when the **`DGML_PAGE_CACHE`** environment
variable names a directory, the renderer keys each render by a content hash
and reuses it:

- **Hit** — an identical PDF rendered before is copied from the cache and
  ghostscript is not invoked (it need not even be installed).
- **Miss** — the PDF is rendered normally, then copied into the cache. A
  `.complete` marker is written last, so an interrupted write reads as a miss
  rather than a partial hit.

The cache is **off by default**; unset, rendering is unchanged. It is keyed by
content, not by workspace — so it pays off when the same PDFs are ingested into
many workspaces (e.g. the clustering sweep's per-cell workspaces in
[evaluation/clustering/](../evaluation/clustering/), which sets it automatically;
`--no-page-cache` opts out). Entries are plain `<hash>/page_*.png` directories
and are safe to delete at any time.

## `workspace.json`

The workspace identity, written by `dgml workspace create`:

```json
{
  "name": "Acme Contracts",
  "organization": "Acme"
}
```

- `organization` — embedded in every docset namespace URI this workspace
  generates (`http://dgml.io/<organization>/<DocSetSlug>`), across both the
  generated document tree (`dgml docset generate`) and the extraction schema
  (`dgml extraction generate-schema` / `set-schema`). Set once at
  `workspace create` (`--organization`, required). It is sanitized into a legal
  URI path segment before use — whitespace runs collapse to a hyphen and
  URI-illegal characters are dropped (`"Andrew Corp"` → `Andrew-Corp`), so the
  stored display value and the URI segment can differ. Already-valid segments
  are unchanged, including the workspace **directory name** that
  `Workspace.organization` falls back to for workspaces created before
  `workspace.json` existed (e.g. `dgml-workspace`), preserving their namespaces.
- `name` — human-readable label (`--name`, optional; defaults to the workspace
  directory name). Surfaced by `dgml status`; not used in URIs.

## `config.json` (optional)

Workspace-level settings. Required when `--text-mode ocr` is used or
when LLM-backed schema generation / value extraction is enabled (see
the `grounded` section below).

### Where config comes from — the three-layer model

You configure **once** and reuse across **every** workspace. Config flows
through three layers:

```text
source default_config.json   (shipped in the dgml-core wheel; the baseline template)
        │  seeded by `dgml init` (or `dgml init --refresh` to re-pull)
        ▼
<workspace-parent>/local_config.json   (your ONE shared config, a peer of the
        │  copied by `dgml workspace create`   workspace; reused by every sibling)
        ▼
<workspace>/config.json      (per-workspace; what the loaders actually read)
```

- `dgml init` copies the bundled template to `local_config.json` — a **peer of
  the workspace root** (`workspace.root.parent / "local_config.json"`; with the
  default `./dgml-workspace` this is `./local_config.json`). Edit it once.
- `dgml workspace create` copies `local_config.json` verbatim to
  `<workspace>/config.json`. Every sibling workspace inherits the same shared
  config.
- Only `<workspace>/config.json` is read at runtime. Editing the shared
  `local_config.json` afterwards does not touch existing workspaces until you
  re-run `dgml workspace create --force` (which re-syncs it).

The template prefills the **model** fields (`classification.model`,
`generation.model`/`label_model`, `grounded.schema_model`/`values_model`) and
the **OCR provider + endpoint** — the decisions that cost money or need an
account. The free knobs (`max_pages`, `max_tool_iters`, `temperature`,
`max_tokens`) are **not** in the file; they keep their code defaults (documented
per section below). There are **no in-code model defaults**: a loader raises its
`*_CONFIG_MISSING` code when a model is unconfigured, so DGML never makes a paid
LLM call you didn't set up.

The optional `text_extraction` section is **not** in the template, so
`--text-mode hybrid` uses its built-in (free) heuristic by default. Add the
section yourself to route the hybrid merge through an LLM instead (see below).

`local_config.json` lands in a working directory next to the workspace, so it
is **git-ignored** (this repo already ignores it) — treat it like a local
secret/config file, not source.

**Comments.** `config.json` and `local_config.json` may carry **full-line**
`//` comments (a line whose first non-whitespace characters are `//`). They are
stripped before parsing; `//` inside a string value (e.g. an `https://`
endpoint) is never touched. Machine-written manifests (`file.json`,
`docset.json`, …) are strict JSON — comments are a config-only affordance.

**Secrets policy.** A workspace is per-developer / per-deployment — not
checked into source control and not necessarily shared. By default the
config file references API keys via `*_api_key_env` env-var-name fields
(which never store the secret itself, just the env var to look it up
in). But every section that accepts `*_api_key_env` also accepts a
literal `*_api_key` field; a developer who keeps their workspace local
can drop the key value directly into config.json if they prefer. The
two are mutually exclusive per side, and the literal wins when both are
present. When neither is set, downstream tooling falls back to its
default credential chain (Entra ID for Azure, the standard
`ANTHROPIC_API_KEY` / `GEMINI_API_KEY` env vars for litellm, etc.).

### `classification` (optional, required for `dgml file add --auto-classify`)

```json
{
  "classification": {
    "model": "gemini/gemini-3.1-flash-lite"
  }
}
```

Field rules:

- `model` — required. Vision-capable, provider-prefixed litellm model id used to
  route a file to a DocSet. A small, low-latency model is sufficient.
- `max_pages` — optional positive int, default `3`. First-N pages shown to the
  classifier.
- `api_key` / `api_key_env` — optional literal key / env-var name, mutually
  exclusive. When neither is set, litellm falls back to its provider-default env
  var (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …).

### `ocr` (optional, required for `--text-mode ocr`)

```json
{
  "ocr": {
    "provider": "azure",
    "endpoint": "https://example.cognitiveservices.azure.com/",
    "api_key_env": "AZURE_DOCINTEL_KEY"
  }
}
```

For AWS:

```json
{
  "ocr": {
    "provider": "aws",
    "region": "us-east-1",
    "profile": "default"
  }
}
```

Field rules:

- `provider` — required. `"azure"` or `"aws"`.
- `endpoint` — required for Azure.
- `api_key` — Azure-only, optional. A literal API key. Mutually
  exclusive with `api_key_env`.
- `api_key_env` — Azure-only, optional. The **name** of an env var
  holding the API key. When neither `api_key` nor `api_key_env` is set,
  authentication falls through to `DefaultAzureCredential` (Entra ID).
- `region` — required for AWS.
- `profile` — AWS-only, optional. The boto3 profile name from
  `~/.aws/credentials`. When unset, the default credential chain runs.

### `grounded` (optional, required for `dgml docset schema generate` / `dgml file extract`)

```json
{
  "grounded": {
    "schema_model": "anthropic/claude-opus-4-7",
    "values_model": "gemini/gemini-2.5-pro",
    "schema_api_key_env": "ANTHROPIC_API_KEY",
    "values_api_key_env": "GEMINI_API_KEY"
  }
}
```

Field rules:

- `schema_model` — required. Provider-prefixed litellm model id used by
  `dgml docset schema generate`.
- `values_model` — required. Provider-prefixed litellm model id used by
  `dgml file extract` and the auto-extract hook on `docset add-file`.
- `schema_api_key` / `values_api_key` — optional literal keys for the
  respective sides. Mutually exclusive with the matching `*_env` field.
- `schema_api_key_env` / `values_api_key_env` — optional env var names
  for the respective sides. When neither the literal nor the env-name
  field is set on a side, litellm falls back to its provider-default
  env var (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc.).
- `max_tool_iters` — optional positive int, default 20. Cap on
  `get_page_words` tool calls per extraction.

### `generation` (required for `dgml docset generate`)

Names the two LLMs the PDF→DGML pipeline runs. There is **no code default** and
no CLI flag — like the other model-consuming commands, `generate` reads its
models solely from this section, so which model runs is one visible choice per
workspace. Both `model` and `label_model` are **required**.
Without this section, generation fails with `GENERATION_CONFIG_MISSING`.

```json
{
  "generation": {
    "model": "anthropic/claude-haiku-4-5",
    "label_model": "anthropic/claude-sonnet-4-6",
    "api_key_env": "ANTHROPIC_API_KEY"
  }
}
```

Field rules:

- `model` — required. Provider-prefixed litellm model id for
  the per-page **transcription** (the bulk of the calls).
- `label_model` — required. Model for the single batch-wide **semantic
  labeling** call, named explicitly so a stronger labeling model is a
  deliberate choice. (This is also the model used by the final semantic-link
  pass and by `dgml discover`'s semantic filters.)
- `api_key` / `api_key_env` — optional literal key / env-var name, mutually
  exclusive. When neither is set, litellm falls back to its provider-default
  env var (`ANTHROPIC_API_KEY`, etc.).
- `api_base` — optional endpoint URL (e.g. for a self-hosted/proxy
  provider); omit for hosted providers.

A malformed section fails the next `docset generate` with
`GENERATION_CONFIG_INVALID`.

### `text_extraction` (optional)

Switches the per-page merge used by `--text-mode hybrid` from its
built-in heuristic to an LLM. Hybrid mode reconciles the digital and OCR
word streams cluster by cluster; when this section is present, each
to-decide cluster is handed to the configured model, which chooses
digital text, OCR text, or a combination (e.g. de-ligaturing a word, or
splitting a run-together token). When the section is **absent**, hybrid
mode uses its deterministic Levenshtein heuristic — so this is purely
opt-in and existing workspaces are unaffected.

This section *tunes the merge within hybrid mode*; it does **not** select
the text mode. The `--text-mode` flag still chooses which extractor runs.

```json
{
  "text_extraction": {
    "model": "ollama_chat/gemma4:latest",
    "api_base": "http://localhost:11434",
    "temperature": 0.0
  }
}
```

Field rules:

- `model` — required. Provider-prefixed litellm model id. A local
  [Ollama](https://ollama.com/) model (`ollama/<name>`) keeps the merge
  on-device; any litellm-supported model works.
- `api_base` — optional. The endpoint URL. Required for Ollama
  (`http://localhost:11434`); omit for hosted providers.
- `api_key` / `api_key_env` — optional literal key / env-var name,
  mutually exclusive. Local providers need neither; when both are unset,
  litellm falls back to its provider-default env var.
- `temperature` — optional number, default `0.0` (deterministic merges).
- `max_tokens` — optional positive int, default 4000. Cap on the merge
  response size; raise it if very dense pages truncate.

All of a page's to-decide clusters go out in one call. Any failure
(model unreachable, timeout, unparseable response) falls back to the
heuristic for that page, so a flaky local model never aborts a file.
Under `--debug`, each call is logged to `usage.jsonl` under operation
`hybrid_merge`.
A malformed section fails the next hybrid extraction with error code
`TEXT_EXTRACTION_CONFIG_INVALID`.

### `style` (optional)

Enables image-based `dg:style` for `--text-mode ocr`
files. Digital and hybrid files derive `dg:style` deterministically from
the PDF glyphs during grounding, but OCR carries no font information — so
by default OCR files get no `dg:style`. **The section's presence is the
switch:** when it is present, the grounding pass has the configured vision
`model` read each page image and report the observed formatting per
grounded snippet (filtered to the allow-list). When the section is
**absent**, OCR files stay unstyled — purely opt-in, existing workspaces
unaffected.

The setting is honored **only for files whose recorded `text_mode` is
`ocr`**; it never overrides or competes with the deterministic
digital/hybrid path.

```json
{
  "style": {
    "model": "anthropic/claude-haiku-4-5"
  }
}
```

Field rules:

- `model` — **required** (the section exists only to enable the path).
  Provider-prefixed litellm model id; must be vision-capable (it is shown
  page images).
- `api_base` — optional endpoint URL (e.g. for a local Ollama vision model).
- `api_key` / `api_key_env` — optional literal key / env-var name,
  mutually exclusive; when both unset, litellm falls back to its
  provider-default env var.
- `max_tokens` — optional positive int, default 4000.

A malformed section (including a missing `model`) is validated up front by
`docset generate` and fails fast with error code `STYLE_CONFIG_INVALID`.

### `clustering` (optional)

Overrides for the bundled clustering defaults used by `dgml cluster`
(and the auto-cluster step of `dgml file add --auto-classify`). The
shipped defaults live in
[packages/dgml-core/src/dgml_core/clustering_config.json](../packages/dgml-core/src/dgml_core/clustering_config.json)
and stand on their own — this section only needs to spell out the
fields you want to change.

The same overlay can also be supplied as a standalone file for a single
run via `dgml cluster --config PATH` (the file's top-level keys are what
this section's `clustering` value holds — i.e. drop the `clustering`
wrapper). When `--config` is given it replaces this section for that run.

```json
{
  "clustering": {
    "encoder_text": {"name": "e5"},
    "training": {"epochs": 50}
  }
}
```

Field rules:

- The section is a partial overlay: every top-level key is optional,
  and within each section any subset of fields can be set. Missing
  keys fall through to the bundled default.
- Overrides are deep-merged: `{"training": {"epochs": 50}}` keeps
  `training.loss` and `training.trainable_projector` at their bundled
  defaults rather than wiping them out.
- The `scenario` section is partly dynamic: its *regime* — `name`,
  `known_categories`, `n_shots` — is picked from the workspace state at
  call time, so overriding those keys is ignored. Its clustering-algorithm
  knobs (`cluster_algorithm`, `leiden_*`, `reduce_method`, `reduce_dim`, …)
  *are* honored, so you can switch algorithm or retune k / resolution /
  reduction here.
- Field names and value enums come from the `Config` pydantic schema
  in the `dgml-clustering` package
  ([packages/clustering/src/clustering/config/schema.py](../packages/clustering/src/clustering/config/schema.py)).
  A typo or out-of-enum value fails the next `dgml cluster` call with
  error code `CLUSTERING_CONFIG_INVALID`.

## `docset.json`

```json
{
  "id": "fdadsf99asdfz",
  "name": "Contracts 2026",
  "description": "Signed customer contracts for FY26",
  "key_questions": [
    "What is the effective date?",
    "Who are the contracting parties?",
    "What is the contract term?"
  ]
}
```

- `key_questions` — list of concrete questions that documents in this
  DocSet can answer from their first pages. Drives the
  schema-shareability rubric used by `dgml file add --auto-classify`:
  a new file is assigned here only if it would answer the same
  questions. Optional; older `docset.json` files written without this
  field read back as an empty list.

## `docsets/<id>/extraction-schema.rnc` (optional)

The grounded **extraction schema** for the docset, in **RELAX NG Compact**
(the DGML spec's canonical schema form). When present, files assigned to
this docset can have their values extracted against it; the result is a
`dg:extraction` element in the file's `<stem>.dgml.xml` (see below).

A docset has **at most one extraction schema**. `dgml extraction set-schema`
accepts either a `.rnc` document or a grounded-field JSON Schema (`.json`,
converted to RNC on the way in); `dgml extraction generate-schema` produces
one from sample PDFs. RNC is the only on-disk form. Replacing it overwrites
the file atomically; clearing it removes the file.

The schema describes the fields to extract as a docset vocabulary — element
definitions of the form `Name = element docset:Name { content }`
with `##` doc comments (`## description`, `## Example:`, `## Prompt:`) — within
the constrained subset the toolkit understands (`dgml_core.extraction_schema`).
It follows the spec §12/§13 form (a `namespace docset` declaration plus element
defs; roots are the unreferenced elements — no `start`/`dg:chunk` rule), and a
`start` rule is also accepted if present. Internally it is converted to the
engine's `grounded_field` JSON Schema, whose
leaf values carry `{ "text", "locations": [{ "page_number", "bounding_box":
[left, top, right, bottom] }] }` in integer image pixels (top-left origin, 300
dpi, relative to `page_images/page_N.png`), so every extracted value traces
back to one or more regions of the source PDF.

When present, `extraction-schema.rnc` is one of the artifacts captured in a
file's attestation (its own `extraction_schema` slot, hashed as raw RNC bytes),
alongside `schema.json` and the file's `<stem>.dgml.xml` — see
[merkle-attestation.md](merkle-attestation.md).

## `docsets/<id>/schema.json` (optional)

The **generation tag schema** for the docset — the canonical set of DGML
XML tag names that locks element structure across the docset's documents.
Written by `dgml docset generate` (the labeling pass derives it from the
labeled documents and saves it here). A prior run's `schema.json` can be fed
back into a later run via `--schema-path` to pin the vocabulary — then it is
injected as a locked contract on every generation call, so similar documents
converge on the same tags. It is the schema captured in a file's attestation alongside that
file's `<stem>.dgml.xml` (see [merkle-attestation.md](merkle-attestation.md)).

Distinct from `extraction-schema.rnc` above, and the two never collide: this one
governs the generated full-document tree; the extraction schema governs the
`dg:extraction` element. Both can coexist in one `<stem>.dgml.xml`
(`full-extraction`). The body is the planner's `Schema` document
(canonical tag names plus per-tag metadata). Generation also writes a
`cache/` at the docset root. It holds **functional** files the next
`generate` run reloads — `*_blocks.json`, `label_*_cNN_raw.json`, and
`concept_roster.json` (used for incremental generation and roster reuse) —
which are always written. Its **debug-only** artifacts (raw LLM dumps,
`*.concept.xml`/`*.semantic.xml`, prompt listings) and the separate
`coverage_report.json` are written only when `dgml --debug docset generate`
is used; a default run leaves just the functional cache.

## `docsets/<id>/full-schema.rnc` (optional)

The same generation tag schema rendered as **RELAX NG Compact**, written at
the very end of `dgml docset generate` (after grounding and the semlink
pass, so it reflects the final XML). It adds what the generated documents
*show*: observed `xsi:type` data types (pinned onto `@dg:value` when every
typed occurrence agrees), leaf-vs-container shape, and `dg:structure` roles.
Every `schema.json` field is serialized losslessly into `# Field: value`
comment lines, so the JSON can be reconstructed from the `.rnc` (and the
`.rnc` can be hand-edited as the schema's editing surface) via
`dgml_core.generation.rnc.rnc_to_schema_dict` — or fed straight back to a
run with `--schema-path full-schema.rnc`. Because the render is lossless, it —
not `schema.json` — is what ships in DGMLX bundles and is hashed into the
file attestation (slot `full_schema`). Validate documents against it without
a JDK: `uvx rnc2rng full-schema.rnc full-schema.rng && xmllint --noout --relaxng
full-schema.rng files/*/*.dgml.xml`.

## `usage.jsonl` (optional)

Append-only event log of LLM-backed operations that ran against the
workspace — classification, clustering's DocSet-naming, transcription
(`transcribe`), labeling (`label`), semantic links (`links`), schema
generation, value extraction, and hybrid text-merge (`hybrid_merge`).

**Recording is gated on `--debug`.** Without `--debug`, no rows are
written for any operation; pass `--debug` to log cost/token telemetry
alongside the other debug artifacts. One JSON object per line; readers
tolerate corrupt tail lines from a crashed mid-write append. The CLI
never reads this file; it exists for introspection and cost accounting
by external tooling that aggregates and renders it.

One record:

```jsonc
{
  "at": "2026-05-15T17:42:00Z",
  "operation": "extract_values",     // classify | schema_generate | extract_values | transcribe | label | links | hybrid_merge
  "model": "gemini/gemini-3-flash-preview",
  "cost_usd": 0.0123,                // null when litellm doesn't price the model
  "prompt_tokens": 12345,
  "completion_tokens": 234,
  "total_tokens": 12579,
  "duration_s": 15.2,
  "outcome": "ok",                   // "ok" | "error"
  "context": {                       // operation-specific identifiers
    "file_id": "kxlv1o15powg",
    "docset_id": "syfpfggdvqty",
    "tool_calls": 5
  },
  "error": null                      // string when outcome="error"
}
```

`extract_values` records ONE event per extraction even when the model
required multiple internal turns; the per-call costs and token counts
are summed before recording. Partial cost (LLM calls made before a
later failure) is preserved on `outcome=error` rows.

## `docsets/<id>/files/<file_id>/<stem>.dgml.xml` (optional)

The DGML for this (docset, file) pair — the single per-file DGML artifact.
`<stem>` is the source PDF's filename stem. It holds, per spec §13, up to two
things under its root `dg:chunk`:

- the **generated document tree** (`dgml docset generate`), and/or
- a **`dg:extraction`** element (`dgml extraction extract`) — a direct child of
  the root holding the docset schema's extracted fields as `docset:` elements,
  each with its text content, a normalized `dg:value`/`xsi:type` where the text
  is a recognizable typed value, and a `dg:origin` grounding it to the page:

```xml
<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
          xmlns:docset="http://www.dgml.io/<organization>/<slug>"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <!-- generated document tree, if `generate` ran (full-extraction mode) -->
  <dg:extraction>
    <docset:Title dg:origin="1 220 475 919 539">Health and Wellness, BAS</docset:Title>
  </dg:extraction>
</dg:chunk>
```

When `generate` ran first, `extract` adds the `dg:extraction` element alongside
the tree (`full-extraction`); otherwise it writes a minimal `dg:chunk` holding
only the `dg:extraction` element (`extraction`). `dgml extraction get-values`
projects the `dg:extraction` element back to values-shape JSON
(`{tag: {text, value?, locations}}`). Placing this file in the marker dir
(rather than at the docset root) makes the artifact path deterministic
and unique per file, which is what file attestation
([packages/dgml/src/dgml/file_attestation.py](../packages/dgml/src/dgml/file_attestation.py))
treats as the DGML slot of the file version. The shared `schema.json` and the
functional `cache/` files stay at the docset root; the debug-only cache
artifacts, `semantic/`, and `coverage_report.json` are written there only
under `--debug`.

This file is **grounded in place**: as the last step of generation, the
rendered tree is aligned against the file's `page_text/` OCR and a
`dg:origin` attribute (plain `origin` on namespace-free XML) is written
onto every element whose subtree grounded — so `<stem>.dgml.xml`
carries page positions directly, with no separate grounded artifact. Each
attribute is a `"; "`-separated box list, each box `<page> <x1> <y1> <x2>
<y2>` (space-separated) in integer image pixels (top-left origin, 300
dpi, relative to `page_images/page_N.png`). Elements with text-node
children (leaves and mixed-content parents) carry one box per visual
line on each page (a parent's lines cover its whole subtree); pure
containers (all-element children — sections, lists, tables, rows, the
document root) carry one union box per page covering their subtree. A
file with no `page_text/` is left ungrounded. The grounded boxes share the one project-wide coordinate
convention with `values.json` and `page_text`; the only shape difference
is that a `dg:origin` box carries its page as a leading integer because
one element can span pages, whereas `values.json` keeps the page in a
sibling `page_number` field.

## `<stem>.dgml.grounding_stats.json` (optional, `--debug`)

Written next to `<stem>.dgml.xml` only when `dgml docset generate` is run
with `--debug` (or via `scripts/ground.py --debug`). Match-rate
telemetry for the grounding pass: token counts per pass (aligned /
recovered / rescued), per-text-node buckets, and the largest ungrounded
snippets with element paths — the visibility into where generation
dropped or paraphrased document text.

## `file.json`

```json
{
  "id": "ab55kdjs93kk",
  "original_path": "../../inbox/dental-select.pdf",
  "original_filename": "dental-select.pdf",
  "sha256": "<hex digest of the PDF bytes>",
  "added_at": "2026-05-08T17:42:00Z",
  "page_count": 2,
  "text_mode": "digital",
  "page_image_dpi": 300,
  "page_image_renderer": "ghostscript",
  "pdf_converter": null
}
```

`original_path` records where the source was added from, stored relative to
the workspace root so a workspace stays portable — it can be moved or checked
into a repo on another machine and still point at a source committed
alongside it. It falls back to an absolute path only when no relative path
exists (a source on a different drive on Windows). `original_filename` is the
source's basename.

`page_count` is the number of pages reported by pypdf at add time. The
consistency check uses it to validate that `page_images/` and `page_text/`
each contain one file per page.

`text_mode` records how text was extracted at add time. One of
`"digital"`, `"ocr"`, or `"hybrid"` (digital + OCR merged by bounding-box
overlap, OCR wins on conflict).

`page_image_dpi` and `page_image_renderer` record how `page_images/` were
rendered — currently always `300` and `"ghostscript"`, but stored per file
so a later renderer or DPI change is detectable. They are `null` if a
non-PDF source failed to convert (no page images were produced).

`pdf_converter` names the converter that turned a non-PDF source into the
PDF the pipeline ran on (the converter's name with any trailing
`"converter"` suffix removed, e.g. `"libreoffice"`). It is `null` when the
source was already a PDF.

## `page_text/page_N.json`

One per page, written regardless of `text_mode` (`"digital"`, `"ocr"`,
or `"hybrid"` all share this shape). Word locations are
in **image-pixel space** matching the corresponding `page_images/page_N.png`
render — i.e. ints with the top-left origin, computed as
`round(pdf_pts * dpi / 72)` where `dpi` is the same 300 dpi used by
`render_pages`. Files are compact (one line, no pretty-printing) so a
workspace with many pages doesn't bloat on disk:

```json
{"file_id":"ab55kdjs93kk","page":1,"width":2550,"height":3300,"words":[{"t":"Hello","l":[100,210,182,242],"s":{"b":1,"sz":24.0,"c":"red"}},{"t":"world","l":[190,210,290,242],"s":{"sz":12.0}}]}
```

- `width` / `height` — dimensions of the matching `page_images/page_N.png`.
- `words[*].t` — word text (whitespace-separated run of non-whitespace chars).
- `words[*].l` — `[left, top, right, bottom]` ints (top-left origin, pixels).
- `words[*].s` — observed style facts, present only on the digital path (and
  digital-derived `hybrid` words); absent on OCR words. `sz` is
  recorded for every word with sized glyphs — which is essentially every digital
  word — so `s` is present on nearly all of them; `b`/`i`/`c` appear only when
  that non-default formatting was seen. Keys: `b` (bold, `1`), `i` (italic, `1`),
  `sz` (glyph size in PDF points, float), `c` (dominant CSS named color).
  Grounding aggregates these per element into the `dg:style` attribute — `sz`
  feeds the page's modal body-size baseline that `font-size` em-buckets against.

## `errors.json`

Persistent record of fatal failures for an item. Optional — only written
when something goes wrong.

```json
{
  "errors": [
    {
      "operation": "render_pages",
      "message": "ghostscript exited 1: ...",
      "occurred_at": "2026-05-08T17:42:01Z",
      "permanent": true
    }
  ]
}
```

`permanent: true` errors are NOT retried by `dgml check` unless
`--retry-errors` is passed. Use this for failures re-running cannot fix
(corrupt PDF, missing system dep, etc.). Errors with
`permanent: false` are retried on every consistency check.

## DocSet ↔ File assignments

When a File is assigned to a DocSet, an empty directory named after the
file's ID is created under `<workspace>/docsets/<docset_id>/files/`.
Future revisions may put per-(DocSet, File) data inside, but for now those
directories exist only as cross-reference markers.

- Removing a **File** deletes its directory under `files/` AND every
  marker directory under `docsets/*/files/<file_id>/`.
- Removing a **DocSet** leaves the underlying Files untouched.
- The `replace` conflict policy on `dgml file add` deletes the existing
  File entirely, which means its DocSet assignments are also dropped. Use
  `duplicate` if you need both records to coexist.

## Atomicity

JSON files are written via write-to-temp + rename so partial writes can't
corrupt existing state. Multi-step operations (e.g. add file = mkdir +
copy PDF + render pages + write metadata) are NOT transactional; if a
fatal error happens midway, the consistency check is the recovery
mechanism.
