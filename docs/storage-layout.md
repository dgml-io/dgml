# DGML Workspace Storage Layout

A DGML workspace is a directory tree on disk. Everything DGML reads and
writes lives under one root directory.

## Bounding-box convention

Every bounding box DGML stores — in `page_text/page_N.json` and in the
`dg:origin` attributes of `<stem>.dgml.xml` (both the generated document tree
and the `dg:extraction` element) — uses **one**
convention: integer **image pixels** `[left, top, right, bottom]`,
top-left origin, relative to the page's `page_images/page_N.{png,jpg}` (rendered
at 300 dpi by default, then fit to the page-edge bounds; see
[Page images](#page-images-jpeg-edge-bounds)). Page is carried in a sibling `page_number`
field for structured forms, or as a leading integer inside each
`dg:origin` box (`<page> <x1> <y1> <x2> <y2>`, space-separated) since one
element can span pages.

## Resolving the workspace root

The root is determined in this order:

1. `--workspace <path-or-id>` CLI flag (or, in code, `Workspace.open(<path-or-id>)` —
   which resolves *and* migrates, seal-checks and requires an initialized workspace;
   `Workspace.resolve` does only the first step and is for the commands that run before a
   workspace exists). The argument is a filesystem path **or** a workspace id. Since an id needs no
   distinguishing prefix (`my-workspace` is as valid as a generated `ws_qf7imkc7f6oqzfwt`)
   it is also a legal directory name, so the two are told apart in four steps:

   1. **Not a well-formed id** — it carries a separator, a dot, an uppercase letter, or
      is outside 3–40 characters — so it is a path. No store is built to decide this,
      and it is what keeps `./my-workspace` addressing the directory.
   2. **[The store of workspaces](#the-store-of-workspaces) holds it** → that workspace.
   3. **A directory of that name exists** → a path. This is the same cwd-relative
      reading a path argument has always had, so nothing about `--workspace notes`
      moves.
   4. **Neither** → `WORKSPACE_NOT_FOUND`, naming both places looked in.

   Step 4 is the important one: falling through to path resolution would turn a typo'd
   id into a new directory in the working directory. And because step 2 precedes step 3,
   a listed id always wins over a same-named local directory — no `mkdir` can redirect a
   working command at a different workspace — with `./name` as the escape for addressing
   the directory.
2. The `DGML_HOME` environment variable — also either a path or an id.
3. Default: `./dgml-workspace` (relative to the current working directory).

   Note this is a *lookup* of last resort, not a place anything creates: `dgml
   workspace create` with no path builds the workspace in
   [the store of workspaces](#the-store-of-workspaces) instead. Item 3 is what keeps a
   workspace made by an older dgml — or by `create <path>` — opening with no arguments.

`dgml workspace create --organization <org>` (or `create_workspace(organization=…)` in code)
creates the directory layout for a fresh workspace and records its identity in
`workspace.json`. Where it creates it depends on whether you name a place: a path (or
`--workspace` / `$DGML_HOME` pointing at one) makes a workspace in that directory,
addressed by path; naming none puts it in the store of workspaces, addressed by its
`ws_…` id. Config is owned by `dgml init` (the "configure once per
machine" flow) — `workspace create` does not create or touch it. If the
user-level config is absent, the workspace is still created and a warning is
printed telling you to run `dgml init`. The CLI refuses to operate on an
uninitialized workspace except for `init` and `workspace create`. See
[the resolution order](#where-config-comes-from--the-resolution-order) for how
config merges across layers.

## Directory structure

Only for a workspace whose data is on local disk, and **only what has been written**:
nothing is pre-created. `workspace create` writes `config.toml` and `workspace.json`;
`docsets/` and `files/` appear when the first docset or file lands in them. A workspace
whose blobs and documents both live on a remote backend has no `docsets/` or `files/`
at all — see [storage services](#storage-services-storage).

```
<workspace_root>/
├── workspace.json                    # { name, organization, workspace_id, schema_version } — written by `workspace create`
├── config.toml                       # storage binding + settings — REQUIRED
├── usage.jsonl                       # LLM call event log (optional)
├── batches/                          # batch job state (optional; any --batch run; removed once empty)
│   └── <job_id>/                     #   bj_<12 hex>
│       ├── job.json                  #   manifest (see "batches/<job_id>/" below)
│       ├── lease.json                #   which process holds the job right now
│       ├── responses/<key>.json      #   one received response per request key
│       └── inputs/<name>             #   inputs a resumed run restores
├── docsets/
│   └── <docset_id>/                  # generated, or set by `docset create --id`
│       ├── docset.json               # { id, name, description, key_questions }
│       ├── extraction-schema.rnc      # grounded extraction schema, RELAX NG Compact (optional)
│       ├── extraction-guidance.md     # docset-level extraction guidance shown to the LLM (optional)
│       ├── authored-schema.json     # the tag schema a user SUPPLIED via --schema-path (optional)
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
    └── <file_id>/                    # generated, or set by `file add --id`
        ├── <original_filename>       # source copied in (a .pdf, or a
        │                             #   convertible source like .docx/.xlsx)
        ├── <stem>.pdf                # converted PDF — only when the source was
        │                             #   not already a PDF; what pages/text and
        │                             #   generation use (see docs/conversion.md)
        ├── file.json                 # metadata (see schema below)
        ├── page_images/              # JPEG/PNG page renders at page_image_dpi
        │                             #   (300 by default), fit to the page-edge
        │                             #   bounds (see below; cacheable)
        │   ├── page_1.png
        │   └── page_2.png
        ├── page_text/                # one JSON of word boxes per page
        │   ├── page_1.json
        │   └── page_2.json
        └── errors.json               # recorded fatal errors (optional)
```

A generated ID is 12 lowercase alphanumerics — `~62` bits of entropy each, from
`secrets.choice`
([packages/dgml-core/src/dgml_core/ids.py](../packages/dgml-core/src/dgml_core/ids.py)).

A File or DocSet ID can also be **set by the caller** with `dgml file add --id` or
`dgml docset create --id`: 3 to 40 characters using only lowercase letters, digits, hyphens
and underscores, starting with a letter or digit. The generated form is a strict subset
of that, so both shapes are valid everywhere an ID appears.

## Page images (JPEG, edge bounds)

Each page is rendered at the file's dpi (`page_image_dpi`, 300 by default) and
then stored with its **longest edge between 1000 and 3508 pixels**, as a PNG
(`page_N.png`) or a JPEG at quality 90 (`page_N.jpg`), whichever is smaller for
that page — a crisp text page is usually far smaller as a PNG, a photo or scan
far smaller as a JPEG, so one file's pages can mix the two. A page whose render already falls in that range — Letter (2550 ×
3300) and A4 (2480 × 3508) at 300 dpi — is stored at exactly its render size; a
large-format sheet is scaled down to a 3508-pixel long edge, and a small page
(or a low `--dpi`) is scaled up to a 1000-pixel one. The scale is uniform, so
the aspect ratio is kept.

The bounds decide the pixel space every box is in: digital text extraction
applies the same fit, so `page_text/` and `dg:origin` boxes always match the
stored image, refit or not. The bounds are not recorded per file; `dgml check`
repairs with the current ones.

Files added before the edge bounds have **full-size PNG** renders
(`page_N.png`). They stay valid — their boxes are in those PNGs' pixels — and
every reader accepts either extension. If `dgml check` has to re-render such a
file's pages, it writes bounded images: identical geometry for a page already
inside the bounds, but a page outside them (a large-format sheet, or a
`--dpi` low enough to fall under the min edge) no longer matches its stored
`page_text/` boxes. Delete and re-add such a file instead, which re-extracts
its text against the new images.

## Page-image render cache (`$DGML_PAGE_CACHE`, optional)

Rendering `page_images/` runs the configured renderer (the system
ghostscript binary by default, or PDFium via `[pdf] provider =
"pypdfium2"`), which dominates the cost of `dgml file add`. The render is a
pure function of the PDF bytes, the renderer, the dpi, and the stored-image
format and edge bounds, so when the **`DGML_PAGE_CACHE`** environment
variable names a directory, the renderer keys each render by a hash of them
all and reuses it:

- **Hit** — an identical PDF rendered before is copied from the cache and
  the render backend is not invoked (it need not even be installed).
- **Miss** — the PDF is rendered normally, then copied into the cache. A
  `.complete` marker is written last, so an interrupted write reads as a miss
  rather than a partial hit.

The cache is **off by default**; unset, rendering is unchanged. It is keyed by
content, not by workspace — so it pays off when the same PDFs are ingested into
many workspaces (e.g. the clustering sweep's per-cell workspaces in
[evaluation/clustering/](../evaluation/clustering/), which sets it automatically;
`--no-page-cache` opts out). Entries are plain `<hash>/page_*.{jpg,png}` directories
and are safe to delete at any time.

## Staging scratch space (`$TMPDIR`)

A workspace whose blobs live on a **remote** backend has no local directory to work in,
but parts of the pipeline need real paths — ghostscript renders page images to a
directory, pdfminer reads a PDF path. Those are staged through Python's `tempfile`,
which picks a directory in this order:

1. **`$TMPDIR`** — the one to set
2. `$TEMP`, then `$TMP`
3. `/tmp`, `/var/tmp`, `/usr/tmp` (on Windows: `%LOCALAPPDATA%\Temp`, `%SYSTEMROOT%\Temp`, …)
4. the current working directory

**Point `TMPDIR` at real disk on a container platform.** `$TMPDIR` — or `/tmp` when it
is unset — is RAM-backed on Cloud Run, on `emptyDir: {medium: Memory}` volumes, and by
default on Fedora/RHEL/Arch, so staging there counts against the memory limit. A whole
batch is staged at once: roughly 1.5 GB of page images for a 500-page document.

```bash
TMPDIR=/mnt/scratch dgml file add big.pdf
```

Set it **before the process starts** — Python memoizes `tempfile.gettempdir()` on first
call, so exporting it mid-run has no effect.

Workspaces on the bundled local-disk store are unaffected: `LocalStore` stages in the
workspace's own `.cache/staging/`.

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

- `workspace_id` — the workspace's **stable handle**: 3 to 40 characters using only
  lowercase letters, digits, hyphens and underscores
  starting with a letter or digit, so it is always a safe single path segment. Generated at
  `workspace create` as `ws_` + 16 lowercase base32 chars (80 bits from `secrets`) —
  opaque and non-semantic, so it survives a directory rename — or set outright with
  `workspace create --id my-workspace`. Carried here so the directory self-describes;
  it is also how [the store of workspaces](#the-store-of-workspaces) keys it.
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

## The store of workspaces

Which workspaces this machine can open is held in a **store of workspaces**, and so is
each one's `config.toml`. It is selected by the `[workspaces]` table of the **user**
config, and two backends ship.

### Local disk (the default)

One folder per workspace under `~/dgml-workspaces/`, each holding that workspace's
`config.toml` — and, for the bundled `LocalStore`, its data too:

```text
~/dgml-workspaces/
└── ws_7qxdm2pjk3n5rwts/
    ├── config.toml
    ├── files/
    └── docsets/
```

The parent is `$DGML_WORKSPACES` when set, else `[workspaces] root`, else
`~/dgml-workspaces`. Not hidden and not under an XDG base directory, because it holds
source documents and page images rather than settings.

The folder name *being* the `workspace_id` is what makes this work with no index: there
is nothing to keep in sync. A folder is listed only if it **holds a `config.toml`** — the
config being the record — and only if its name could be a `workspace_id` at all, so
neither a stray `notes.bak/` nor a loose file in the parent is half-listed. Note the name
test alone is weak now that an id needs no prefix (a plain lowercase folder name is a
well-formed id, which is exactly what `workspace create --id my-workspace` produces);
it is the `config.toml` that decides.

### MongoDB

```toml
# ~/.config/dgml/config.toml
[workspaces]
provider = "dgml_storage_mongo:MongoWorkspacesStore"
mongo_host = "localhost"
mongo_database = "dgml_workspaces"
```

One document per workspace, `_id` = its `workspace_id`, holding its `config.toml` as
verbatim text plus a small derived projection for listing. Point two machines at one
database and `dgml --workspace ws_…` opens the same workspace on both, with no config
file passed between them. See
[the package README](../packages/dgml-storage-mongo/README.md#the-list-of-workspaces)
for the document shape, the compare-and-swap on writes, and what is deliberately *not*
stored there.

### What crosses the interface

Exactly one thing per workspace: the text of its `config.toml`. A listing row is
**derived** from that text, never stored beside it — so `dgml workspace list` can report
`name`, `organization` and `storage_service` precisely *because* they are not a second
copy that could disagree.

`root` in a listing row is computed on the machine doing the listing (the
`workspace_path` its config declares, else the standard folder). It is never stored: where
a workspace's files sit is per-machine, and a shared column recording it is the mistake
described below.

`[workspaces]` is read **only** from the user config, with `tomllib`, never through the
merged loader. The same table in a workspace's own `config.toml` is ignored, and cannot
be honoured even in principle: the store was already used to fetch that file.

### Not listed is not broken

A workspace addressed by path — `dgml --workspace ./ws`, `$DGML_HOME`,
`./dgml-workspace` — is **detached**: it works exactly as before and is simply not in
the store, so it does not appear in `workspace list`. `dgml workspace import <path>`
adds one when you want it there.

Resolving by path never consults the store.

### The legacy index (`workspaces.json`)

Older versions kept a per-machine JSON index at `~/.config/dgml/workspaces.json`
mapping each `workspace_id` to where that workspace was last seen. It is **no longer
written and nothing resolves through it**.

Its rows were a second copy of facts that lived elsewhere, so they could disagree with
the workspace they described and had to be rewritten on every open to stay current —
including correcting the recorded `root` of a workspace that had moved. None of that
exists now: a listing row comes out of the workspace's own config.

`dgml workspace import` with no arguments sweeps every workspace the old index lists
into the store. Nothing happens automatically, the file is left in place so a
half-finished sweep can be repeated, and once the workspaces you care about are imported
it can be deleted.

## The workspace config (`config.toml`)

Every workspace has a `config.toml`. It is **required**, and it is what *makes* a
directory a workspace: it names the storage backend, so a directory without one fails
with `WORKSPACE_NOT_INITIALIZED` rather than silently falling back to local disk — an
absent config is indistinguishable from a remote-backed workspace whose config was
deleted, and both want the same answer from the caller.

Where it lives depends on how the workspace is addressed: `<workspace>/config.toml` for
one addressed by path, and inside [the store of workspaces](#the-store-of-workspaces)
for one addressed by id — which, on a networked backend, is not a file at all.

(`--workspace-config` and `$DGML_CONFIG` used to point at a config kept elsewhere. They
have been removed: that only ever worked because the per-machine index recorded the
location and handed it back on the next open, so with nothing recording it the flag would
have to be repeated forever. To start a workspace from a config you authored, use
`dgml workspace create --from-config <path>`, which copies it in.)

`dgml workspace create` writes it, adding a machine-managed `[workspace]` block:

```toml
# Written by dgml — do not edit by hand.
# `dgml workspace reseal` regenerates storage_fingerprint after a [storage] change.
[workspace]
workspace_id        = "ws_7f3k9q2m4b8xr5wa"
name                = "Acme Contracts"
organization        = "Acme"
storage_service     = "acme"
storage_fingerprint = "sha256:…"

[storage.acme.blobs]
provider = "dgml_storage_s3:S3BlobStore"
bucket   = "acme-contracts"
```

- `storage_service` names which `[storage.<name>]` table this workspace binds to
  (default: `"default"`).
- `workspace_id`, `name`, and `organization` duplicate `workspace.json`. That is
  deliberate and the two are never compared: this file is the **store-free bootstrap
  copy** — readable before the backend is reachable, which is what lets
  `workspace list` describe a Mongo-backed workspace while Mongo is down — and
  `workspace.json` is the copy that travels with the data.
- The `[workspace]` block is read directly from this file, **never merged** across
  config layers. A `workspace_id` in the user-level config would otherwise apply to
  every workspace on the machine, and `DGML_WORKSPACE__*` would let an environment
  variable silence the seal for one invocation.

### Storage does not layer

Every other config section deep-merges across the [five layers](#where-config-comes-from--the-resolution-order).
**Storage is the exception.** A service the workspace defines in its own `config.toml`
is taken **whole**; only a service it does *not* define falls back to the merged
config, which is what keeps a shared `[storage.<name>]` template useful across many
workspaces.

Replacement rather than merging is what makes a workspace self-describing: a
workspace that inherited `bucket` or `mongo_database` from the user config would
silently move its data when that file was edited.

### The storage seal (`storage_fingerprint`)

`storage_fingerprint` is a credential-free hash of the workspace's **resolved**
`blobs`/`docs` pair, recorded when the workspace is created and re-checked on every
command — store-free, before any backend is contacted. A mismatch hard-fails with
`STORAGE_BACKEND_MISMATCH`: the workspace's data is on the previously sealed backend,
so opening it against a new configuration could read or write the wrong store.

- Editing `[storage]` **does** trip it. `dgml workspace reseal <path>` accepts the
  change; `--check` reports drift without writing. (This inverts the pre-1.0
  behaviour, where a config edit could never change an existing workspace's stores.)
- Rotating a credential does **not** trip it — secret-hinted option names are outside
  the hash.
- Copying or moving a workspace does **not** trip it — `root` is outside the hash too.
- A workspace with no recorded fingerprint is *unsealed* and opens untouched
  (trust-on-first-use), which is how a hand-built or just-migrated workspace works.

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
| 3 | Workspace config | The workspace's own `config.toml` — a file in its directory, or held in [the store of workspaces](#the-store-of-workspaces). **Required**; carries the storage binding plus any per-workspace overrides |
| 4 | Environment variables | `DGML_`-prefixed, `__` for nesting |
| 5 | CLI flags | per invocation (e.g. `--schema-model`) |

> **Two exceptions to the deep merge.** The `storage` section does **not** layer — a
> service the workspace defines is taken whole (see
> [Storage does not layer](#storage-does-not-layer)). Nor does the `[workspace]`
> identity block, which is read directly from the workspace's own file so a stray key
> in the user config cannot apply to every workspace on the machine.

`dgml init` writes the **user config** (layer 2) — configure once per machine; every
workspace inherits it. A workspace's own `config.toml` (layer 3) is **required** and is
written by `dgml workspace create`: it carries the storage binding, so it is not
optional and not merely a place for overrides. It may also carry any other section as a
per-workspace override.

**Env-var overrides (layer 4).** Prefix `DGML_`, split path segments on `__`,
lowercased — e.g. `DGML_MODELS__ADVANCED=gemini/gemini-3.1-pro-preview`,
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
at `dgml workspace create --storage <name>` and materialized into that workspace's own
[`config.toml`](#the-workspace-config-configtoml), which is authoritative from then on.

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
  both `BlobStore` and `DocStore` (the bundled `LocalStore` does, as does the sample
  `dgml_storage_mongo:MongoGridFSStore`). A table may not set both a top-level `provider`
  and role subtables.
- **One backend means one instance.** When both roles resolve to the same backend, a
  workspace constructs it **once** and serves both roles from that single instance —
  so `MongoGridFSStore` in the flat form holds one `MongoClient`, not one per role.
  This keys off the resolved config, not the syntax, so it covers the flat form *and*
  two per-role subtables that happen to be identical. Roles that resolve differently
  are still built independently, and lazily: a command that only reads documents never
  constructs the blob store.
- **`default` and back-compat** — the reserved name **`default`** is what a workspace
  uses when `--storage` is omitted; a bare `[storage]` (flat or with `blobs`/`docs`)
  *is* the `default` service, and no `[storage]` at all is the zero-config local
  store for both roles.
- **Secrets vs. identity.** Secret-hinted options (keys containing `key`, `secret`,
  `token`, `password`, `credential`, or the word `uri` as in `mongo_uri`) are excluded from the
  [seal fingerprint](#the-storage-seal-storage_fingerprint), so rotating a credential
  never reads as "the store moved". Every in-tree provider takes its credentials from
  the **environment** instead of config — S3 via the boto3 chain, Mongo via
  `$DGML_MONGO_URI` — and a third-party provider that accepts an inline credential
  must name the option so one of those substrings appears in it.

  ⚠️ A workspace's `config.toml` now sits **beside the workspace** and is likely to be
  committed or synced. Prefer env-var indirection over literal secrets in it.
- **Editing a template changes the workspaces that use it.** A workspace that does
  not define the service itself resolves it from here, so an edit takes effect after
  `dgml workspace reseal <path>` accepts it. A workspace that *does* define
  `[storage.<name>]` in its own config is unaffected — storage does not layer.
- **Each workspace gets its own namespace in a shared backend.** A bucket or database
  named in a template is usually shared — by every workspace created from it, and
  often by other applications. So the S3 and MongoDB stores always put a workspace's
  data under its **workspace id**, after an optional `prefix` you choose:

  | Backend | `prefix` | Data lands in |
  |---|---|---|
  | S3 | not set | `s3://<bucket>/dgml/<id>/files/…` |
  | S3 | `"contracts"` | `s3://<bucket>/contracts/<id>/files/…` |
  | MongoDB | not set | collections `dgml_<id>_files`, …, GridFS bucket `dgml_<id>_blobs` |
  | MongoDB | `"contracts"` | collections `contracts_<id>_files`, …, GridFS bucket `contracts_<id>_blobs` |

  The id is added when the store is opened, not written into config, so `config.toml`
  holds only the `prefix` you wrote — and a template can be shared by any number of
  workspaces as it is. It comes from the `workspace_id` in `config.toml`'s
  `[workspace]` block, which `workspace create` writes and which never changes, so a
  workspace's data never moves. Local-disk storage is already per-workspace and has no
  prefix.

### The `[models]` tiers

The simplest way to configure models is the `[models]` block. One `family` key
picks a whole provider family's curated defaults for the four tiers that back
the per-task models:

```toml
[models]
family = "anthropic_google"   # or anthropic / google / openai
```

A family-based config *tracks* dgml's shipped defaults — an upgrade may move a
tier to a newer model. To pin models, set explicit tiers, with or without a
family (an explicit tier always overrides its family default):

```toml
[models]
family   = "anthropic_google"
light    = "gemini/gemini-3.5-flash-lite"  # classification, style
standard = "anthropic/claude-haiku-5-5"    # transcription, text extraction
advanced = "anthropic/claude-sonnet-5-5"   # labeling, value extraction
expert   = "anthropic/claude-opus-5-5"     # schema generation
```

(The tier→task mapping lives in code and may change; it is not written into the
file.) Each per-task field below is an **override** that wins over its tier; when
a task names no model of its own it falls back to its tier. A tier that is unset
falls back to the nearest set tier (nearest lower first, then higher) with a
warning — so a minimal config that sets only, say, `standard` still resolves
every task.

`family` is shorthand for its four tiers *within its config layer*: tiers that
layer leaves unset are filled from the family before the layers merge. So a
workspace config (or `DGML_MODELS__FAMILY`) setting `family = "openai"` replaces
explicit tiers in the user config, while tiers set in the same layer as the
family, or a higher one, still override it. To drop a user-level pin in one
workspace, restate the family there.

A model and its credentials are **one group, written together in one config
layer**: a tier's are the `<tier>_*` keys beside it, a task's are the keys on its
own section, and a credential in a layer that does not set its model is refused
(`*_CONFIG_INVALID`) by the task that would have used it — the merge itself never
fails, so commands that call no model are unaffected. A group never borrows from
another tier, another section or another layer, and a key can never be separated
from the model it was written beside.

```toml
[models]
family = "anthropic_google"
anthropic_api_key_env = "ANTHROPIC_API_KEY"   # onto every keyless anthropic/… model
google_api_key_env    = "GEMINI_API_KEY"      # … and every keyless gemini/… one
expert          = "anthropic/claude-opus-5-5"
expert_api_key_env = "OPUS_KEY"               # this tier's own key …
expert_api_base = "https://proxy.example"     # … and endpoint
```

- **Per task**, on the task's own section (`generation.api_key_env`,
  `grounded.schema_api_key`, `classification.api_base`, …) — for the model that
  section names in the same layer. Credentials in a layer whose section names no
  model are refused when that task runs: a key beside no model has no model it was
  meant for.
- **Per tier**: `<tier>_api_key` / `<tier>_api_key_env` / `<tier>_api_base` next
  to the tier, taken whole by every task that gets its model from that tier. They
  require the tier to be set **in the same layer** (a `family` there counts), else
  every task that reads the tiers is refused, and they move with it: a layer that
  sets `expert` replaces `expert`'s whole group, so a workspace `config.toml` that
  overrides `expert` does **not** inherit a user-level `expert_api_key_env`.

  Because a `family` sets every tier in its layer, a tier credential may sit
  beside the family alone, with no explicit tier:

  ```toml
  [models]
  family = "anthropic"
  expert_api_key_env = "OPUS_KEY"   # sent to whatever model the family picks for expert
  ```

  The key then goes to the family's default model for that tier, which **may
  change across dgml upgrades** — and in a mixed family may belong to another
  provider (in `anthropic_google`, `light` is a Gemini model, so `light_api_key`
  there must be a Google key). To know which model a key is sent to, pin the tier
  beside it (`expert = "anthropic/claude-opus-5-5"`).
- **Per provider**: `anthropic_api_key`, `google_api_key`, `openai_api_key` (or
  the `_env` form) merges like any other setting — the highest layer that sets
  either form wins, so a key in `DGML_MODELS__ANTHROPIC_API_KEY_ENV` overrides one
  in a config file, and one form in the user config never clashes with the other in
  the workspace config. After the merge it is written onto every model of that
  provider (by model-id prefix: `anthropic/`, `gemini/`, `openai/`) that has no key
  of its own and no `api_base`: tiers, task sections, and per-call overrides
  (`--model` / `--label-model`, a `--generation-config` profile or file,
  `--values-model` / `--schema-model`) alike, whichever layer set the model. One
  key per provider covers a mixed family such as `anthropic_google`. A model
  pointed at a custom `api_base` never receives it; dgml warns there, because
  litellm would send its own env-var key to that host instead.

**Secrets injected through the environment** follow the same rules. The
environment is a layer like any other, so `DGML_GENERATION__API_KEY_ENV` alone makes
generation fail with `GENERATION_CONFIG_INVALID` — it names no model — and
`DGML_GENERATION__MODEL` alone replaces the file's `generation` model group whole,
`api_base` and key included.
For an `anthropic/`, `gemini/` or `openai/` model, inject the provider key
(`DGML_MODELS__ANTHROPIC_API_KEY_ENV=…`): it reaches every keyless model of that
provider, whichever layer set the model. For any other model, or one at a custom
`api_base`, set the whole group in the environment (`DGML_GENERATION__MODEL`,
`DGML_GENERATION__API_BASE`, `DGML_GENERATION__API_KEY_ENV`).

A model with no credentials falls back to litellm's per-provider env var — the one
fallback there is. A tier that is unset borrows its fallback tier's model and
credentials together.

For an `openai/`, `anthropic/` or `gemini/` model at a custom `api_base`, set
`api_key` (or `api_key_env`) beside it — the server's real key, or any placeholder
such as `"EMPTY"` if it checks none. litellm requires a key for these providers, and
without one it reads the provider's own environment variable and sends that to the
custom host. `ollama/` models need no key and have no such fallback.

`dgml init --provider {anthropic,anthropic_google,google,openai}` writes a
family-only `[models]` block; omit `--provider` to auto-detect from the API-key
env vars that are set (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`
— checked in that order, so an OpenAI key never overrides a provider the other
two already resolve).

**Secrets policy.** By default config references API keys via `*_api_key_env`
env-var-name fields (which store the env var name, not the secret). Every
section that accepts `*_api_key_env` also accepts a literal `*_api_key`; the two
are mutually exclusive per side and the literal wins. When neither is set,
downstream tooling falls back to its default credential chain (Entra ID for
Azure, the conventional `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` /
`OPENAI_API_KEY` env vars for litellm, etc.).

**Migration.** The config format was JSON (`config.json`) before this release.
A workspace whose only config is a legacy `config.json` raises
`LEGACY_CONFIG_PRESENT`; run `dgml init` to write the TOML user config and copy
any settings across.

### `classification` (optional, required for `dgml file add --auto-classify`)

The model defaults to the `[models].light` tier; add this section only to
override it (with that model's own credentials, if any).

```toml
[classification]
model = "gemini/gemini-3.5-flash-lite"
```

Field rules:

- `model` — optional; falls back to the `light` tier. Vision-capable,
  provider-prefixed litellm model id used to route a file to a DocSet.
- `max_pages` — optional positive int, default `3`. First-N pages shown to the
  classifier.
- `api_key` / `api_key_env` / `api_base` — optional; mutually-exclusive key /
  env-var name, plus an optional endpoint, for the `model` set here. They require
  `model` in the same config layer (else `CLASSIFICATION_CONFIG_INVALID` when the
  classifier runs); a model from the `light` tier takes the tier's own
  `light_api_key` and friends. When unset, litellm uses its per-provider env var.

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

- `provider` — required. A bundled short name (`"azure"`, `"aws"`, `"macos"`)
  or a dotted `"module.path:ClassName"` naming your own `OcrProvider`
  subclass. The short names are aliases for the bundled classes' own dotted
  paths; see [ocr-providers.md](ocr-providers.md).
- `endpoint` — required for Azure.
- `api_key` — Azure-only, optional. A literal API key. Mutually
  exclusive with `api_key_env`.
- `api_key_env` — Azure-only, optional. The **name** of an env var
  holding the API key. When neither `api_key` nor `api_key_env` is set,
  authentication falls through to `DefaultAzureCredential` (Entra ID).
- `region` — required for AWS.
- `profile` — AWS-only, optional. The boto3 profile name from
  `~/.aws/credentials`. When unset, the default credential chain runs.

Every other field is passed to the named provider, which declares the keys it
accepts and rejects the rest — so a custom provider's options go here too.

### `grounded` (optional, required for `dgml docset schema generate` / `dgml file extract`)

The two models default to tiers — `schema_model` ← `expert`, `values_model` ←
`advanced`. Add this section only to override a model or set per-side
credentials.

```toml
[grounded]
schema_model = "anthropic/claude-opus-5-5"
values_model = "gemini/gemini-3.1-pro-preview"
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
  These per-side credentials are for the model set here and require it in the
  same config layer (else `GROUNDED_CONFIG_INVALID` when that side runs); a model
  from its tier takes the tier's credentials. When unset, litellm uses its
  per-provider env var.
- `max_tool_iters` — optional positive int, default 20. Cap on
  `get_page_words` tool calls per extraction.
- `values_reasoning_effort` — optional, default `"medium"`. The reasoning budget
  of the value-extraction call: one of `"none"`, `"minimal"`, `"low"`,
  `"medium"`, `"high"`, `"xhigh"` (passed to the provider through litellm), or
  `"default"` to send no reasoning effort and take the provider's own default.
  It is the largest cost and latency dial on extraction, and the right setting
  differs by model. Location grounding is not affected.
- `locations_reasoning_effort` — optional, default `"high"`. The reasoning
  budget of the location-grounding call on pages with OCR words, same values as
  above. Anthropic models never receive it (that call forces its tool, and the
  wrapper drops the budget); every other provider does, and at the default it is
  the larger half of a Gemini values model's extraction latency. A page that
  goes through the no-words grid path (no OCR words and a known image size)
  keeps its own `"medium"`, which this field does not change. The value is
  checked by name, not against what the model supports: one the provider
  refuses fails in phase 3, after the values call has been paid for, on the
  first page that needs grounding.

### `generation` (required for `dgml docset generate`)

The two LLMs the PDF→DGML pipeline runs. Both default to the `standard` tier —
`model` (per-page **transcription**) and `label_model` (the batch-wide
**semantic labeling** call) — so this section is optional. There is no CLI
flag; the models are a visible config choice. If neither a field nor the tier
resolves a model, generation fails with `GENERATION_CONFIG_MISSING`.

Labeling previously fell back to the `advanced` tier. It no longer does:
measured over 13 docsets in two independent corpora, 3 draws each, the
`standard` tier matched or beat `advanced` on every metric while costing about
half as much, so the stronger tier was not earning its price on this task. Set
`label_model` explicitly to override.

```toml
[generation]
# Overrides (optional — the tiers cover both by default):
label_model = "anthropic/claude-opus-5-5"
# Anthropic extended thinking for both passes. Default "disabled".
thinking = "adaptive"
```

Field rules:

- `model` — optional; falls back to the `standard` tier. Per-page transcription.
- `label_model` — optional; falls back to the `standard` tier. The single
  batch-wide semantic-labeling call (also used by the final semantic-link pass
  and `dgml discover`'s semantic filters).
- Transcription credentials: `api_key` / `api_key_env` / `api_base`.
- Labeling credentials: `label_api_key` / `label_api_key_env` /
  `label_api_base`. The two models carry **independent** credentials because
  they may name different providers (e.g. the `anthropic_google` family blends
  Gemini and Anthropic models). Each set is for the model set here and requires
  it in the same config layer (else `GENERATION_CONFIG_INVALID` when generation
  runs); a model from its tier takes the tier's credentials. When unset, litellm
  uses its per-provider env var.
- `thinking` — optional; `"disabled"` (default), `"adaptive"`, or
  `"between_tools"` (Claude Sonnet 5.5 rejects `"disabled"`; this is its off
  switch, thinking only between tool calls). Anthropic
  extended thinking, applied to **both** passes; ignored for non-Anthropic
  models. Omitting the field on the wire is not the same as turning thinking
  off: Claude 4.6+/5 models think adaptively unless told not to, so generation
  states the mode rather than inheriting it. The default is `"disabled"`
  because on an internal 5-docset benchmark (three draws per arm, transcription
  frozen so only labeling varied) it scored higher than adaptive on
  exact-match and token-overlap F1, individually and pooled, at roughly 2.7x
  less cost and 4.5x less wall time. Set `"adaptive"` to restore the model
  default. Any other value fails with `GENERATION_CONFIG_INVALID`.

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
  (`http://localhost:11434`); omit for hosted providers. Like `api_key`, it
  belongs to a `model` set in the same config layer (else
  `TEXT_EXTRACTION_CONFIG_INVALID` when the feature runs); a model from the tier
  takes the tier's own credentials.
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
model = "anthropic/claude-haiku-5-5"
```

Field rules:

- `enabled` — optional bool, default `false`. The on switch; everything else in
  the section is ignored while it is false.
- `model` — optional; falls back to the `light` tier. Provider-prefixed litellm
  model id; must be vision-capable (it is shown page images). A model alone does
  **not** enable the feature.
- `api_base` — optional endpoint URL (e.g. for a local Ollama vision model). Like
  `api_key`, it belongs to a `model` set in the same config layer (else
  `STYLE_CONFIG_INVALID` when the feature runs); a model from the tier takes the
  tier's own credentials.
- `api_key` / `api_key_env` — optional literal key / env-var name,
  mutually exclusive; when both unset, litellm falls back to its
  provider-default env var.
- `max_tokens` — optional positive int, default 4000.

A malformed **enabled** section (including one whose `model` resolves to
nothing) is validated up front by `docset generate` and fails fast with error
code `STYLE_CONFIG_INVALID`. A disabled section is never validated, so shipping
`enabled = false` alone is always safe.

### `clustering` (optional)

Overrides for the bundled clustering defaults used by `dgml cluster`.
(`dgml file add --auto-classify` does *not* read this section — it
classifies one file at a time via the `classification` section below,
and never runs the clustering pipeline.) The
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
(top-left origin, relative to `page_images/page_N.{png,jpg}`) — so every
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
labeled documents and saves it here). This is the **observed** vocabulary —
`seed ∪ everything coined during the run` — and it is rewritten at the end of
every run. A prior run's `schema.json` can be fed back into a later run via
`--schema-path` to pin the vocabulary — then it is injected as a locked
contract on every generation call, so similar documents converge on the same
tags. It is the schema captured in a file's attestation alongside that
file's `<stem>.dgml.xml` (see [merkle-attestation.md](merkle-attestation.md)).

A user-supplied vocabulary is **not** kept here — see
[`authored-schema.json`](#docsetsidauthored-schemajson-optional) below. Keeping
the two apart is what makes a seeded run reproducible: otherwise the run's own
output becomes the next run's input.

Distinct from `extraction-schema.rnc` above, and the two never collide: this one
governs the generated full-document tree; the extraction schema governs the
`dg:extraction` element. Both can coexist in one `<stem>.dgml.xml`
(`full-extraction`). The body is the planner's `Schema` document
(canonical tag names plus per-tag metadata). Generation also writes a
`cache/` at the docset root. It holds **functional** files the next
`generate` run reloads — `*_blocks.json`, `label_*_cNN_raw.json` (one per
chunk whose reply parsed; a bisected chunk's halves are `cNNa`/`cNNb`),
`concept_roster.json` (the flat legacy vocabulary; incremental reuse prefers
the docset's `authored-schema.json`, then its `schema.json`, and falls back to
this file), and
`semlinks/<hash>.json` (one document's semantic links, keyed on what the link
model reads — tag names and text — so re-rendering or grounding a document
replays them instead of paying for the pass again) —
which are always written. Its **debug-only** artifacts (raw LLM dumps,
`*.concept.xml`/`*.semantic.xml`, prompt listings) and the separate
`coverage_report.json` are written only when `dgml --debug docset generate`
is used; a default run leaves just the functional cache.

## `docsets/<id>/authored-schema.json` (optional)

The generation tag schema a **user supplied**, written by `dgml docset generate
--schema-path <file>`. Same Schema v1 body as `schema.json`, and deliberately a
separate file: `derive_schema` rewrites `schema.json` at the end of every run
with `seed ∪ everything the labeling pass coined`, and the next incremental run
auto-seeds from it — so ground truth goes in and a polluted vocabulary comes
back out. This slot is never written by derivation, which is what lets a later
`generate` with no flags re-seed from what the author actually wrote.

Whether the remembered schema is applied strictly or as a foundation is a
per-run choice (`--extend-schema`), not a property of the file: the vocabulary
persists, the mode does not.

Seed precedence for a `generate` run:
`--schema-path` → `authored-schema.json` → `schema.json` → `cache/concept_roster.json`
(`--no-roster` uses none of them).

Stored in canonical Schema v1 form regardless of which input form it was
authored in — a plain newline-delimited tag list, a JSON `{name: description}`
object, a `schema.json`, or a `full-schema.rnc` — so there is exactly one shape
to read back. See
[Supplying your own tag schema](cli-reference.md#supplying-your-own-tag-schema).

Not attested. `full_schema` still hashes `full-schema.rnc`, unchanged.

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

**Recording is opt-in.** The CLI turns it on under `--debug`: without
`--debug`, no rows are written for any operation; pass `--debug` to log
cost/token telemetry alongside the other debug artifacts. A library caller
sets `record_usage=True` on the operation (or on its `LLMConfig`) instead,
which records rows without writing any debug artifact; `extract_values`
also returns the same totals on `ExtractionResult.usage` for callers that
keep telemetry out of the workspace. One JSON object per line; readers
tolerate corrupt tail lines from a crashed mid-write append. The CLI
never reads this file; it exists for introspection and cost accounting
by external tooling that aggregates and renders it.

One record:

```jsonc
{
  "at": "2026-05-15T17:42:00Z",
  "operation": "extract_values",     // classify | cluster | schema_generate | extract_values | transcribe | label | links | hybrid_merge | style_annotate | batch_unused | llm_call (no operation named); a library caller's LLMConfig may name its own
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

Under a [batch job](#batchesjob_id-optional) the rows of a run that ends
paused (`--no-wait`) are not written; the run that completes the job writes
the rows a blocking run would. A response a later run replays after an
interrupted run already counted it contributes zero cost and tokens, so every
response is billed exactly once across the job's rows. A response the job paid
for that no run of it ever used (a superseded batch's results, a request the
command stopped asking for, a batch collected only as the run ended) is still
shown: when the job completes it writes one `batch_unused` row per such
response, at the tier that served it, with `context`
`{"unused": true, "batch_job": "<job id>"}`. No command's own row counts it.
A job no run will finish writes them too: `dgml batch cancel` for every
response its runs collected and no row counted yet (a paused run's included),
and `dgml batch delete` / `prune` for any left before the job is removed. These
rows follow the command's own setting: a job whose command records no usage
(run without `--debug`) writes none.

## `batches/<job_id>/` (optional)

State of one batch job — see
[Batch jobs](cli-reference.md#batch-jobs---no-wait-and-dgml-batch). Written by
any `--batch` run (a plain blocking run keeps one silently, for crash
recovery, deleted when that run finishes — unless it ends leaving the job
`failed`, when the payload names it in `batch.job`). A completed job keeps only
its summarized `job.json`; `dgml batch prune` and `dgml batch delete` remove
jobs. Deleting the last job also removes the then-empty `batches/` directory
itself, so a workspace whose batch runs all finished has no `batches/`.

- **`job.json`** — the manifest:

  ```jsonc
  {
    "job_id": "bj_3f9a1c07d2e4",
    "command": "docset generate",          // docset generate | extraction extract |
                                           // extraction generate-schema | file add
    "argv": ["docset", "generate", "…", "--batch", "--no-wait"],  // replayed by `batch resume`
    "cwd": "/home/me/project",              // replayed from here (relative paths)
    "created_at": "…", "updated_at": "…",
    "status": "pending",                   // pending | completed | failed (`batch status`
                                           // derives `ready` when asked; never stored)
    "error": null,
    "runs": 3,                             // runs of the command under this job so far
    "provider_batches": [                  // every provider batch the job submitted
      {"job": {"provider": "anthropic", "job_id": "msgbatch_…", "custom_ids": ["…"],
               "submitted_at": "…", "extra": {}},
       "model": "anthropic/claude-haiku-5-5", "api_base": null,
       "stage": "transcribe",               // the pipeline stage the batch served
       "credential": {"section": "generation", "field": "transcribe",
                      "env": "MY_ANTHROPIC_KEY"},   // where the key comes from; never the key
       "key_fingerprint": "3f9a1c07d2e4",   // first 12 hex digits of sha256(key) the batch
                                           // was created with (null if the backend cannot
                                           // tell); a 404 under a different key is not
                                           // "gone": the run stops, the record stays open
       "keys": {"<custom_id>": "<request key>"},
       "state": "collected",               // open | collected | dropped (canceled) |
                                           // uncertain (create outcome unknown; has "error"
                                           // and a placeholder "uncertain_…" job_id) |
                                           // settling (canceled at --batch-deadline, cancel
                                           // not settled; has "settling_reason" and
                                           // "possibly_double_billed", and "collected" (the
                                           // custom_ids already stored, when a collection
                                           // failed part-way: never counted late); once it
                                           // ends it gets "late_billed": {"requests",
                                           // "cost_usd"}, "settled_at", and becomes dropped)
       "last_status": {"state": "ended", "succeeded": 12, "errored": 0, "expired": 0,
                       "canceled": 0, "processing": 0},
       "cleanup": "done"}                   // dropped batches only: provider-side delete
                                           // "pending" (cancel not settled) | "done"
                                           // A record may also carry "gone": true — the
                                           // provider answered 404 (batch deleted or never
                                           // existed); closed out as collected (every
                                           // response stored) or dropped, no cleanup owed
    ],
    "billed": ["<request key>", "…"],      // responses already counted on a usage row
    "inputs": {"generate/schema.json": true, "presence:generate/blocks/a.pdf": false},
    "state": {},                           // command-specific (file add: its ingest results)
    "run_stats": {"e1:anthropic/claude-haiku-5-5": {"1": {"batches": 1, "…": 0}}},
                                           // per executor, per run: summed for the payload
    "record_usage": true,                  // the command writes usage rows (`--debug`);
                                           // the job's own `batch_unused` rows follow it
    "deadline": {"at": "2026-09-30T18:00:00Z", "seconds": 21600.0,
                 "runs": {"1": {"canceled_batches": 0, "…": 0}}}
                                           // only with --batch-deadline: the absolute UTC
                                           // instant every run honors, the duration it was
                                           // set from, and each run's deadline counters
  }
  ```

- **`lease.json`** — `{"owner", "expires"}` (epoch seconds): the process running
  the job. Renewed every minute, expires 10 minutes after the last renewal;
  `dgml batch unlock` removes it. A runner never re-creates a lease it finds
  gone or held by another owner: it stops (`BATCH_JOB_BUSY`).
- **`responses/<key>.json`** — one received response. The key is
  `<sha256 of the request>-<occurrence>`: the digest is taken over the litellm
  request as canonical JSON, leaving out the API key, the timeout and any
  credential header, and the occurrence number keeps two identical requests in
  one run apart. The file holds the litellm response and only the hidden fields
  accounting needs (`response_cost`, `dgml_tier`) — no provider headers and never
  a credential.
- **`inputs/slices/<source sha>/<engine>/<first>-<last>-<pages sha>.pdf`** — each
  PDF slice sent to a model (a transcription window, a schema sample trimmed to
  fit, an extraction page range; page numbers 1-based), recorded the first time it is cut and
  reused by every resume, because slicing is not byte-reproducible; and
  `inputs/generate/converted/<name>.pdf` for a document converted on demand.
  Deleted with the responses when the job completes.
- **`inputs/<name>`** — the content an input had when the job started, restored
  by every resumed run (`docset generate`'s `schema.json` and
  `concept_roster.json`). Inputs recorded only by presence (a document's
  `_blocks.json`) have no file here; their flag is in `job.json` `inputs`.

## `docsets/<id>/files/<file_id>/<stem>.dgml.xml` (optional)

The DGML for this (docset, file) pair — the single per-file DGML artifact.
`<stem>` is the source PDF's filename stem. It holds, per spec §13, up to two
things under its root `dg:chunk`:

- the **generated document tree** (`dgml docset generate`), and/or
- a **`dg:extraction`** element (`dgml extraction extract`) — a direct child of
  the root holding the docset schema's extracted fields as `docset:` elements,
  each with its text content, a normalized `dg:value`/`xsi:type` where the text
  is a recognizable typed value, and its provenance (DGML 0.71): `dg:origin`
  boxes for its printed text, `dg:inputs` naming the elements the value used,
  and `dg:method` (`calculated` | `inferred` | `defaulted`) on a value the
  page doesn't print. Printed text a value used that no schema field extracts
  is an anonymous `<dg:chunk xml:id dg:origin>` next to the field. A 0.7
  `dg:origin="computed"` element still reads, with its `dg:href` targets as
  its inputs:

```xml
<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
          xmlns:docset="http://www.dgml.io/<organization>/<slug>"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <!-- generated document tree, if `generate` ran (full-extraction mode) -->
  <dg:extraction>
    <docset:Title xml:id="title" dg:origin="1 220 475 919 539">Health and Wellness, BAS</docset:Title>
    <docset:Credits xsi:type="integer" dg:value="120" dg:inputs="#title"
        dg:method="inferred">120</docset:Credits>
  </dg:extraction>
</dg:chunk>
```

When `generate` ran first, `extract` adds the `dg:extraction` element alongside
the tree (`full-extraction`); otherwise it writes a minimal `dg:chunk` holding
only the `dg:extraction` element (`extraction`). `dgml extraction get-values`
projects the `dg:extraction` element back to values-shape JSON
(`{tag: {text, value?, locations?, inputs?, evidence?, method?}}`). Placing this file in the pair directory
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
<y2>` (space-separated) in integer image pixels (top-left origin,
relative to `page_images/page_N.{png,jpg}`). Elements with text-node
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
rendered — the renderer is `"ghostscript"` (the default) or `"pypdfium2"`,
per the workspace's `[pdf] provider` config at add time; the dpi is `300`
unless `dgml file add --dpi N` set otherwise. They are stored per file both so
a later renderer change is detectable and because they are load-bearing: the
dpi is the scale of every `page_text/` word box (see below), and `dgml check
--retry-errors` re-renders and re-extracts at the *recorded* dpi **with the
recorded renderer**, so a repair reproduces the file's existing pixels
instead of today's config (backends differ subtly in anti-aliasing and ±1 px
dimension rounding). They are `null` if a non-PDF source failed to convert
(no page images were produced).

`pdf_converter` names the converter that turned a non-PDF source into the
PDF the pipeline ran on (the converter's name with any trailing
`"converter"` suffix removed, e.g. `"libreoffice"`). It is `null` only when
DGML converted nothing: the source was already a PDF, or a library caller
supplied the PDF (`FileStore.add(…, converted_pdf=)`; DGML did not check
that it matches the source). So a convertible source (`.docx`, `.xlsx`, …)
with a `null` `pdf_converter` is a supplied PDF — DGML names every converter
it runs, and a failed conversion still names the one that was tried. That
reading is an invariant: a future feature that produces no PDF at all must
say so in a field of its own rather than by leaving this one `null`.

## `page_text/page_N.json`

One per page, written regardless of `text_mode` (`"digital"`, `"ocr"`,
or `"hybrid"` all share this shape). Word locations are
in **image-pixel space** matching the corresponding `page_images/page_N.{png,jpg}`
render — i.e. ints with the top-left origin, computed as
`round(pdf_pts * dpi / 72)` where `dpi` is the file's `page_image_dpi` — the
same value `render_pages` used, 300 unless `--dpi` said otherwise — and then
scaled by the same fit to the edge bounds as the stored image, when the page
falls outside them. Consumers of these coordinates should take the page's
`width` / `height` here (or the image's own size) rather than assuming 300
dpi: the boxes in a File added with `--dpi 150` are half the size of the same
File's at 300, and a large-format sheet's are scaled to its 3508-pixel image. Files are compact (one line, no pretty-printing) so a
workspace with many pages doesn't bloat on disk:

```json
{"file_id":"ab55kdjs93kk","page":1,"width":2550,"height":3300,"words":[{"t":"Hello","l":[100,210,182,242],"s":{"b":1,"sz":24.0,"c":"red"}},{"t":"world","l":[190,210,290,242],"s":{"sz":12.0}}]}
```

- `width` / `height` — dimensions of the matching `page_images/page_N.{png,jpg}`.
- `words[*].t` — word text (whitespace-separated run of non-whitespace chars).
  On the digital path a page also carries the words of filled PDF form fields
  (printable, non-hidden text fields and combo boxes, not password fields,
  read from the field's appearance stream with true glyph boxes). The words of
  a field come right after the page content word they follow in reading order:
  the nearest label on the same line, else the nearest line above in the same
  column, else the end of the page. The page content words keep their order; a
  value the page content already prints in the field is not repeated.
- `words[*].l` — `[left, top, right, bottom]` ints (top-left origin, pixels).
- `words[*].s` — observed style facts, present only on the digital path (and
  digital-derived `hybrid` words); absent on OCR words and on the words of
  form-field values, so a filled form has the same body-size baseline as the
  blank one. `sz` is
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

Library callers look assignments up in either direction through `DocSetStore`
rather than the store directly: `list_files(docset_id)` gives a DocSet's files,
and `docsets_for_file(file_id)` gives a File's DocSets, each with its
`assigned_at` (`null` for assignments carried over by a layout migration).
`docsets_for_file` tolerates a bad manifest the way `list_all()` does: a DocSet
whose `docset.json` is missing or cannot be parsed is left out (with a warning)
rather than raised, and only files assigned to that DocSet are affected;
`dgml check` reports it.

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
