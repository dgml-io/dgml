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

1. `--workspace <path-or-id>` CLI flag (or `Workspace.resolve(<path-or-id>)` in
   code). The argument is a filesystem path **or** a `ws_…` workspace id: when it
   exactly matches an id in the [per-machine registry](#per-machine-workspace-registry),
   the workspace opens at that entry's recorded root; otherwise it is treated as a
   path (the `ws_` prefix + base32 charset means an id can't be mistaken for one).
2. The `DGML_HOME` environment variable.
3. Default: `./dgml-workspace` (relative to the current working directory).

`dgml workspace create --organization <org>` (or `Workspace.init()` in code)
creates the directory layout for a fresh workspace and records its identity in
`workspace.json`. Config is owned by `dgml init` (the "configure once per
machine" flow) — `workspace create` does not create or touch it. If the
user-level config is absent, the workspace is still created and a warning is
printed telling you to run `dgml init`. The CLI refuses to operate on an
uninitialized workspace except for `init` and `workspace create`. See
[the resolution order](#where-config-comes-from--the-resolution-order) for how
config merges across layers.

## Directory structure

```
<workspace_root>/
├── workspace.json                    # { name, organization, workspace_id, schema_version } — written by `workspace create`
├── config.toml                       # OCR / LLM / clustering settings (optional)
├── usage.jsonl                       # LLM call event log (optional)
├── docsets/
│   └── <docset_id>/                  # 12-char base-36 ID
│       ├── docset.json               # { id, name, description, key_questions }
│       ├── extraction-schema.rnc      # grounded extraction schema, RELAX NG Compact (optional)
│       ├── extraction-guidance.md     # docset-level extraction guidance shown to the LLM (optional)
│       ├── schema.json               # generation tag schema, written by `generate` (present after generation)
│       ├── full-schema.rnc           # schema.json as RELAX NG Compact, written by `generate` (see below)
│       └── files/
│           └── <file_id>/            # one assigned (DocSet, File) pair
│               ├── assignment.json   # { docset_id, file_id, assigned_at } — the assignment record
│               └── <stem>.dgml.xml   # generated tree and/or dg:extraction,
│                                     #   plus its grounded/stats siblings (below)
├── .cache/                           # workspace-internal scratch; never workspace data,
│   ├── embeddings/                   #   excluded from the blob namespace and safe to delete
│   └── staging/                      #   in-flight batch writes (page renders, text extraction)
└── files/
    └── <file_id>/                    # 12-char base-36 ID
        ├── <original_filename>       # source copied in (a .pdf, or a
        │                             #   convertible source like .docx/.xlsx)
        ├── <stem>.pdf                # converted PDF — only when the source was
        │                             #   not already a PDF; what pages/text and
        │                             #   generation use (see docs/conversion.md)
        ├── file.json                 # metadata (see schema below)
        ├── page_images/              # PNG page renders at page_image_dpi
        │                             #   (300 by default; cacheable, see below)
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
of `dgml file add`. The render is a pure function of the PDF bytes, the
renderer, and the dpi, so when the **`DGML_PAGE_CACHE`** environment variable
names a directory, the renderer keys each render by a hash of all three and
reuses it:

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
  "organization": "Acme",
  "workspace_id": "ws_7f3k9q2m4b8xr5wa",
  "schema_version": 1
}
```

- `workspace_id` — the workspace's **stable handle** (`ws_` + 16 lowercase
  base32 chars, 80 bits from `secrets`). Opaque and non-semantic, so it survives a
  directory rename. Minted at `workspace create` and carried here so the directory
  self-describes; it also keys the [per-machine registry](#per-machine-workspace-registry).
  A workspace created before this field existed is given one automatically the
  first time any command opens it (a schema migration). `dgml --workspace <workspace_id>`
  opens the workspace by this id.
- `schema_version` — the on-disk layout revision this workspace was last written
  against. `dgml` migrates an older workspace up to the current revision in place
  the first time a command touches it (see
  [migrations](../packages/dgml-core/src/dgml_core/migrations.py)); a workspace
  with no `workspace.json` at all reads as version 0.
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

## Per-machine workspace registry

A single **per-machine** index maps each `workspace_id` to where that workspace
lives, so workspaces can be listed (`dgml workspace list`) and opened by id
(`dgml --workspace <workspace_id>`). It sits next to the user config —
`$XDG_CONFIG_HOME/dgml/workspaces.json` if set, else `%APPDATA%\dgml\workspaces.json`
on Windows, else `~/.config/dgml/workspaces.json` — and is machine-managed **JSON**
(not hand-edited, unlike `config.toml`), an object keyed by `workspace_id`:

```json
{
  "ws_7f3k9q2m4b8xr5wa": {
    "name": "Acme Contracts",
    "organization": "Acme",
    "root": "/Users/me/acme-ws",
    "storage_service": "default",
    "storage": {
      "blobs": { "provider": "dgml_core.storage_local:LocalStore" },
      "docs": { "provider": "dgml_core.storage_local:LocalStore" }
    },
    "storage_fingerprint": "sha256:…",
    "created_at": "2026-08-05T12:00:00Z",
    "schema_version": 1
  }
}
```

- The registry is **per-machine**, deliberately separate from `workspace.json`
  (which travels with the directory): the same workspace opened on two machines has
  one `workspace_id` but two registry entries, each with that machine's `root`.
- `root` is the local store location (used for open-by-id and the reverse lookup).
  Only `LocalStore` ships today, so every entry has a `root`.
- **The entry is self-describing about the workspace's stores.** `storage_service`
  names the [`config.toml` storage template](#storage-services-storage) the
  workspace was created from (where its secrets live, and the target of a re-seal).
  `storage` is a **non-secret snapshot pair** of that template — one snapshot each
  for the `blobs` and `docs` roles (provider + non-secret options — never
  credentials), since a workspace configures its blob store and document store
  independently. It is *authoritative* for opening the workspace: it opens from this
  snapshot even if the template is later edited or removed, so the registry alone
  records where a workspace's data lives. Editing the `config.toml` template does
  **not** change an existing workspace's stores — `dgml workspace register --storage
  <name>` is the explicit "adopt new config" (re-seal).
- `storage_fingerprint` is a credential-free hash of the snapshot. On open it is
  recomputed from the entry and compared: a mismatch means the machine-managed JSON
  was **hand-edited**, and the command hard-fails with `STORAGE_BACKEND_MISMATCH`
  (repair with `dgml workspace register … --storage <name>`). It is *not* compared
  against `config.toml`.
- Entries are added automatically: `workspace create` records a new workspace
  (including its storage snapshot), and the first time any command opens a workspace
  on a machine it is auto-registered there (additive — it never overwrites an
  existing entry). `dgml workspace register` is the explicit override that re-seals
  a moved directory's `root` or switches its storage service.
- Each write is atomic (write-temp-rename); registration is an idempotent upsert by
  id, so a lost update from a concurrent write self-heals on the next open.

## Configuration (`config.toml`)

LLM / OCR / clustering settings, in **TOML**. Required when `--text-mode ocr`
is used or when LLM-backed generation / schema / value extraction runs.

### Where config comes from — the resolution order

Configuration is a **deep merge** across five layers, each overriding the keys
of those above it (a layer overrides only what it sets and inherits the rest):

| # | Layer | Location |
|---|---|---|
| 1 | Built-in defaults | shipped in the wheel (dataclass defaults: `max_pages`, `temperature`, …) |
| 2 | **User config** | `$XDG_CONFIG_HOME/dgml/config.toml` if set, else `%APPDATA%\dgml\config.toml` on Windows, else `~/.config/dgml/config.toml` — written by `dgml init` |
| 3 | Workspace config | `<workspace>/config.toml` (optional per-workspace overrides) |
| 4 | Environment variables | `DGML_`-prefixed, `__` for nesting |
| 5 | CLI flags | per invocation (e.g. `--schema-model`) |

`dgml init` writes the **user config** (layer 2) — configure once per machine;
every workspace inherits it. A per-workspace `config.toml` (layer 3) is optional
and created by hand only for workspace-specific overrides; `dgml workspace
create` does **not** write one.

**Env-var overrides (layer 4).** Prefix `DGML_`, split path segments on `__`,
lowercased — e.g. `DGML_MODELS__ADVANCED=gemini/gemini-2.5-pro`,
`DGML_GENERATION__LABEL_MODEL=…`, `DGML_OCR__ENDPOINT=…`. This overrides config
**settings**; it is distinct from provider **secret** vars (`ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, …), which litellm and the `*_api_key_env` indirection use to
supply the actual key. `DGML_HOME` (workspace root) and `DGML_DEBUG` are reserved
and never treated as config.

There are **no in-code model defaults**: a loader raises its `*_CONFIG_MISSING`
code when a model can't be resolved from any layer, so DGML never makes a paid
LLM call you didn't set up.

### Storage services (`[storage]`)

Where a workspace's data physically lives. A workspace has **two independently
configured backends** — a **blob** store (page images, PDFs, XML, schemas) and a
**document** store (manifests, page text, assignments, the usage log) — so it can
mix them (e.g. S3 blobs + Mongo docs, or S3 blobs + local docs). By default there
is nothing to configure — both run on the bundled local-disk store. To use a
pluggable backend, define one or more **named storage services**; each is selected
at `dgml workspace create --storage <name>` and snapshotted into that workspace's
[registry entry](#per-machine-workspace-registry).

```toml
# A named service with a backend per role. Each provider is a dotted "module:Class"
# path; the remaining keys are that provider's own options.
[storage.acme.blobs]
provider     = "dgml_storage_s3:S3BlobStore"
bucket       = "acme-contracts"
region       = "us-east-1"

[storage.acme.docs]
provider       = "dgml_storage_mongo:MongoDocStore"
mongo_database = "dgml"
```

- **Per-role form** — `[storage.<name>.blobs]` / `[storage.<name>.docs]` subtables,
  each with its own `provider` + options. A role you omit falls back to the bundled
  local store, so `[storage.<name>.blobs]` alone puts blobs on the backend and keeps
  documents on local disk.
- **Flat form** — a `[storage.<name>]` with a single top-level `provider` (and no
  `blobs`/`docs` subtables) uses that one class for **both** roles; it must implement
  both `BlobStore` and `DocStore` (the bundled `LocalStore` does). A table may not
  set both a top-level `provider` and role subtables.
- **`default` and back-compat** — the reserved name **`default`** is what a workspace
  uses when `--storage` is omitted; a bare `[storage]` (flat or with `blobs`/`docs`)
  *is* the `default` service, and no `[storage]` at all is the zero-config local
  store for both roles.
- **Secrets vs. identity.** Each backend's non-secret identity (provider + options
  like `bucket`/`region`) is snapshotted into the registry entry and is authoritative
  for opening the workspace. Secret-hinted options (keys containing `key`, `secret`,
  `token`, `password`, `credential`) are **never** written to the registry; they are
  read from this template (or the provider SDK's own credential chain) at open and
  are excluded from the seal fingerprint, so rotating a credential never trips it.
- **Pinned semantics.** Editing a `[storage.<name>]` template does not change an
  existing workspace's stores — the workspace stays on its recorded snapshot. Use
  `dgml workspace register --storage <name>` to re-seal it to the current template.

### The `[models]` tiers

The simplest way to configure models is the `[models]` block — four tiers that
back the per-task models:

```toml
[models]
light    = "gemini/gemini-flash-lite-latest"  # classification, style
standard = "anthropic/claude-haiku-4-5"    # transcription, text extraction
advanced = "anthropic/claude-sonnet-5"     # labeling, value extraction
expert   = "anthropic/claude-opus-5"       # schema generation
```

(The tier→task mapping lives in code and may change; it is not written into the
file.) Each per-task field below is an **override** that wins over its tier; when
a task names no model of its own it falls back to its tier. A tier that is unset
falls back to the nearest set tier (nearest lower first, then higher) with a
warning — so a minimal config that sets only, say, `standard` still resolves
every task.

Tiers name only models — they carry no credentials. Credentials are configured
per task on the task's own section (e.g. `generation.api_key_env`,
`grounded.schema_api_key`); a model sourced from a tier uses its task section's
credentials, or falls back to litellm's per-provider env var when the section
sets none.

`dgml init --provider {anthropic,google,mixed}` writes a ready-made
`[models]` table; omit `--provider` to auto-detect from the API-key env vars
that are set (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).

**Secrets policy.** By default config references API keys via `*_api_key_env`
env-var-name fields (which store the env var name, not the secret). Every
section that accepts `*_api_key_env` also accepts a literal `*_api_key`; the two
are mutually exclusive per side and the literal wins. When neither is set,
downstream tooling falls back to its default credential chain (Entra ID for
Azure, the conventional `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` env vars for
litellm, etc.).

**Migration.** The config format was JSON (`config.json`) before this release.
A workspace whose only config is a legacy `config.json` raises
`LEGACY_CONFIG_PRESENT`; run `dgml init` to write the TOML user config and copy
any settings across.

### `classification` (optional, required for `dgml file add --auto-classify`)

The model defaults to the `[models].light` tier; add this section only to
override it or set classification-specific credentials.

```toml
[classification]
model = "gemini/gemini-flash-lite-latest"
```

Field rules:

- `model` — optional; falls back to the `light` tier. Vision-capable,
  provider-prefixed litellm model id used to route a file to a DocSet.
- `max_pages` — optional positive int, default `3`. First-N pages shown to the
  classifier.
- `api_key` / `api_key_env` / `api_base` — optional; mutually-exclusive key /
  env-var name, plus an optional endpoint. Apply whether the model is set here or
  comes from the `light` tier; when unset, litellm uses its per-provider env var.

### `ocr` (optional, required for `--text-mode ocr`)

Not a model tier — an OCR backend. On macOS an absent `[ocr]` section defaults
to the on-device Apple Vision engine.

```toml
[ocr]
provider = "azure"
endpoint = "https://example.cognitiveservices.azure.com/"
api_key_env = "AZURE_DOCINTEL_KEY"
```

For AWS:

```toml
[ocr]
provider = "aws"
region = "us-east-1"
profile = "default"
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

The two models default to tiers — `schema_model` ← `expert`, `values_model` ←
`advanced`. Add this section only to override a model or set per-side
credentials.

```toml
[grounded]
schema_model = "anthropic/claude-opus-5"
values_model = "gemini/gemini-2.5-pro"
schema_api_key_env = "ANTHROPIC_API_KEY"
values_api_key_env = "GEMINI_API_KEY"
```

Field rules:

- `schema_model` — optional; falls back to the `expert` tier. Used by
  `dgml docset schema generate`.
- `values_model` — optional; falls back to the `advanced` tier. Used by
  `dgml file extract` and the auto-extract hook on `docset add-file`.
- `schema_api_key` / `values_api_key` — optional literal keys per side,
  mutually exclusive with the matching `*_env` field.
- `schema_api_key_env` / `values_api_key_env` — optional env var names per side.
- `schema_api_base` / `values_api_base` — optional endpoint per side.
  These per-side credentials apply whether the model is set here or comes from
  its tier; when unset, litellm uses its per-provider env var.
- `max_tool_iters` — optional positive int, default 20. Cap on
  `get_page_words` tool calls per extraction.

### `generation` (required for `dgml docset generate`)

The two LLMs the PDF→DGML pipeline runs. Each defaults to a tier —
`model` (per-page **transcription**) ← `standard`, `label_model` (the batch-wide
**semantic labeling** call) ← `advanced` — so this section is optional. There is
no CLI flag; the models are a visible config choice. If neither a field nor its
tier resolves a model, generation fails with `GENERATION_CONFIG_MISSING`.

```toml
[generation]
# Overrides (optional — the tiers cover both by default):
label_model = "anthropic/claude-opus-5"
```

Field rules:

- `model` — optional; falls back to the `standard` tier. Per-page transcription.
- `label_model` — optional; falls back to the `advanced` tier. The single
  batch-wide semantic-labeling call (also used by the final semantic-link pass
  and `dgml discover`'s semantic filters).
- Transcription credentials: `api_key` / `api_key_env` / `api_base`.
- Labeling credentials: `label_api_key` / `label_api_key_env` /
  `label_api_base`. The two models carry **independent** credentials because
  they may name different providers (e.g. the default `mixed` config transcribes
  on Anthropic and labels on Gemini). These apply whether the models are set here
  or come from their tiers; when unset, litellm uses its per-provider env var.

A malformed section fails the next `docset generate` with
`GENERATION_CONFIG_INVALID`.

### `text_extraction` (optional)

Switches the per-page merge used by `--text-mode hybrid` from its
built-in heuristic to an LLM. Hybrid mode reconciles the digital and OCR
word streams cluster by cluster; with `enabled = true`, each to-decide
cluster is handed to the configured model, which chooses digital text, OCR
text, or a combination (e.g. de-ligaturing a word, or splitting a
run-together token). Without it — whether the section is absent, empty, or
sets `enabled = false` — hybrid mode uses its deterministic Levenshtein
heuristic. `dgml init` writes the section with `enabled = false`.

A section that is configured but not enabled logs a one-line warning to
stderr rather than being ignored in silence.

This section *tunes the merge within hybrid mode*; it does **not** select
the text mode. The `--text-mode` flag still chooses which extractor runs.

```toml
[text_extraction]
enabled = true
model = "ollama_chat/gemma4:latest"
api_base = "http://localhost:11434"
temperature = 0.0
```

Field rules:

- `enabled` — optional bool, default `false`. The on switch; everything else in
  the section is ignored while it is false.
- `model` — optional; falls back to the `standard` tier. Provider-prefixed
  litellm model id. A local [Ollama](https://ollama.com/) model
  (`ollama/<name>`) keeps the merge on-device; any litellm-supported model works.
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
by default OCR files get no `dg:style`. **`enabled = true` is the switch:**
with it, the grounding pass has the configured vision `model` read each page
image and report the observed formatting per grounded snippet (filtered to the
allow-list). Without it — whether the section is absent, empty, or sets
`enabled = false` — OCR files stay unstyled. `dgml init` writes the section
with `enabled = false`, so the feature is advertised but never on by default.

A section that is configured (a model, credentials) but not enabled logs a
one-line warning to stderr rather than being ignored in silence.

The setting is honored **only for files whose recorded `text_mode` is
`ocr`**; it never overrides or competes with the deterministic
digital/hybrid path.

```toml
[style]
enabled = true
model = "anthropic/claude-haiku-4-5"
```

Field rules:

- `enabled` — optional bool, default `false`. The on switch; everything else in
  the section is ignored while it is false.
- `model` — optional; falls back to the `light` tier. Provider-prefixed litellm
  model id; must be vision-capable (it is shown page images). A model alone does
  **not** enable the feature.
- `api_base` — optional endpoint URL (e.g. for a local Ollama vision model).
- `api_key` / `api_key_env` — optional literal key / env-var name,
  mutually exclusive; when both unset, litellm falls back to its
  provider-default env var.
- `max_tokens` — optional positive int, default 4000.

A malformed **enabled** section (including one whose `model` resolves to
nothing) is validated up front by `docset generate` and fails fast with error
code `STYLE_CONFIG_INVALID`. A disabled section is never validated, so shipping
`enabled = false` alone is always safe.

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
with `##` doc comments (`## description`, `## Example:`, `## Prompt:`,
`## Invariant:`) — within
the constrained subset the toolkit understands (`dgml_core.extraction_schema`).
It follows the spec §12/§13 form (a `namespace docset` declaration plus element
defs; roots are the unreferenced elements — no `start`/`dg:chunk` rule), and a
`start` rule is also accepted if present. A field's content model is `text`, an
`xsd:` datatype, or a **value enumeration** (`( "electric" | "water" | … )`)
constraining the normalized value to a closed token set. Internally it is
converted to the engine's `extracted_value` JSON Schema, whose leaf values
carry `{ "text", "value"?, "locations": [{ "page_number", "bounding_box":
[left, top, right, bottom] }] }` — verbatim text, optional normalized value
(enum token, ISO date, plain number), and locations in integer image pixels
(top-left origin, 300 dpi, relative to `page_images/page_N.png`) — so every
extracted value traces back to one or more regions of the source PDF.

When present, `extraction-schema.rnc` is one of the artifacts captured in a
file's attestation (its own `extraction_schema` slot, hashed as raw RNC bytes),
alongside `schema.json` and the file's `<stem>.dgml.xml` — see
[merkle-attestation.md](merkle-attestation.md).

## `docsets/<id>/extraction-guidance.md` (optional)

Docset-level **extraction guidance** — free-form markdown/plain text holding
domain rules that apply to the whole document kind rather than any single
field: classification decision rules, disambiguation conventions, cross-field
consistency rules the extractor should honor. Written verbatim by
`dgml extraction set-guidance` (replaced atomically; removed by clearing);
read by every `dgml extraction extract` against the docset and injected into
the phase-1 extraction prompt after the schema. Complements the per-field
`## Prompt:` annotations in `extraction-schema.rnc`.

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
`concept_roster.json` (the flat legacy vocabulary; incremental reuse prefers
the docset's `schema.json` and falls back to this file) —
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
(`{tag: {text, value?, locations}}`). Placing this file in the pair directory
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
rendered — the renderer is currently always `"ghostscript"`; the dpi is `300`
unless `dgml file add --dpi N` set otherwise. They are stored per file both so
a later renderer change is detectable and because they are load-bearing: the
dpi is the scale of every `page_text/` word box (see below), and `dgml check
--retry-errors` re-renders and re-extracts at the *recorded* value so a repair
reproduces the file's existing geometry instead of today's default. They are
`null` if a non-PDF source failed to convert (no page images were produced).

`pdf_converter` names the converter that turned a non-PDF source into the
PDF the pipeline ran on (the converter's name with any trailing
`"converter"` suffix removed, e.g. `"libreoffice"`). It is `null` when the
source was already a PDF.

## `page_text/page_N.json`

One per page, written regardless of `text_mode` (`"digital"`, `"ocr"`,
or `"hybrid"` all share this shape). Word locations are
in **image-pixel space** matching the corresponding `page_images/page_N.png`
render — i.e. ints with the top-left origin, computed as
`round(pdf_pts * dpi / 72)` where `dpi` is the file's `page_image_dpi` — the
same value `render_pages` used, 300 unless `--dpi` said otherwise. Consumers
of these coordinates should read `page_image_dpi` off `file.json` rather than
assuming 300: the boxes in a File added with `--dpi 150` are half the size of
the same File's at 300. Files are compact (one line, no pretty-printing) so a
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

When a File is assigned to a DocSet, an `assignment.json` is written to
`<workspace>/docsets/<docset_id>/files/<file_id>/`, holding
`{ docset_id, file_id, assigned_at }`. The pair directory also holds that
pair's generated artifacts (`<stem>.dgml.xml`, `extraction_stats.json`).

Earlier revisions recorded the assignment as the *bare existence* of that
directory, with no file inside. That could not survive its own deletion —
removing the record meant removing the directory, and therefore the generated
artifacts with it — so the record is now a document like any other. A workspace
written before this change is upgraded automatically on first use — see
`schema_version` under [`workspace.json`](#workspacejson).

- Removing a **File** deletes its directory under `files/` AND every
  pair directory under `docsets/*/files/<file_id>/`.
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
