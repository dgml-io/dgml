# dgml-sample-service

**A sample, not a supported product.** A multi-tenant web service that embeds DGML
as a **library** (`dgml_core`) — it never shells out to the `dgml` CLI — with:

- a **FastAPI** backend;
- DGML's documents in **Postgres**, in typed tables (`PostgresDocStore`);
- DGML's blobs (source PDFs, page images, page text, schemas, `.dgml.xml`) in **S3** —
  a local **SeaweedFS** in development;
- page rendering with **PDFium** (`pypdfium2`), so no Ghostscript is needed;
- per-organisation **settings in Postgres**, turned into a DGML `Configuration` in memory
  on each request — there is no `config.toml` anywhere;
- a **React + TypeScript** frontend: docset CRUD, file upload, classification and
  extraction, a document + DGML viewer, a **Data** page that browses what the
  backend stored (the S3 objects and the Postgres rows), and an **APIs** page that tries
  any endpoint with the organisation's real ids offered as choices.

It follows the storage, settings and calling-pattern design for hosting DGML inside a
larger product: one DGML workspace per organisation, the workspace id *is* the
organisation's UUID, the state collections live in the host's Postgres, and blobs live in
the organisation's own bucket.

## Run it

Prerequisites: [uv](https://docs.astral.sh/uv/), Node 18+, and Docker with Compose
(Docker Desktop, OrbStack, or `brew install colima docker docker-compose`).

```bash
samples/dgml-sample-service/scripts/start.sh     # Postgres + SeaweedFS + API + web app
open http://localhost:5180
```

| Script | What it does |
|---|---|
| `scripts/start.sh` | Starts the containers (and colima, if that is your Docker), creates the default bucket, syncs the uv workspace and `npm install`s if needed, then runs the API (`:8700`) and the Vite dev server (`:5180`). Idempotent. |
| `scripts/start.sh --build` | Builds the frontend and serves it from the API at `:8700` instead of Vite. |
| `scripts/start.sh --reload` | Restarts the API on Python changes. |
| `scripts/start.sh --infra-only` | Only Postgres + SeaweedFS (run the apps yourself). |
| `scripts/start.sh --no-web` | Containers + API, no frontend. |
| `scripts/stop.sh` | Stops the apps and the containers; data is kept. |
| `scripts/stop.sh --apps-only` | Stops the API and web app, leaves the containers up. |
| `scripts/stop.sh --purge` | Stops everything and **deletes** the Postgres and SeaweedFS volumes. |
| `scripts/stop.sh --colima` | Also stops the colima VM. |
| `scripts/status.sh` | What is running, and where. |
| `scripts/logs.sh [backend\|frontend\|postgres\|seaweedfs]` | Follows a log. |

Logs and PID files live in `.run/` (git-ignored). Interactive API docs:
<http://127.0.0.1:8700/docs>. Browse the stored blobs in the SeaweedFS filer UI:
<http://localhost:18888/buckets/dgml-sample/>.

Ports and URLs can be overridden with `DGML_SAMPLE_API_PORT`, `DGML_SAMPLE_WEB_PORT`,
`DGML_SAMPLE_PG_PORT` (55432), `DGML_SAMPLE_S3_PORT` (18333), `DGML_SAMPLE_FILER_PORT`
(18888), `DGML_SAMPLE_DATABASE_URL`, `DGML_SAMPLE_S3_ENDPOINT` and
`DGML_SAMPLE_S3_BUCKET`. They are deliberately off the defaults so the stack runs beside a
local Postgres and beside the `dgml-storage-s3` test stack.

### First steps in the UI

1. Create an organisation. It starts with storage pointed at the local SeaweedFS.
2. **Settings** → choose a model family and paste its API key. Classification, schema
   generation and extraction need one; uploading and viewing do not.
3. **Docsets** → create one (a description and a few key questions help classification).
4. **Files** → drop in PDFs, optionally classifying each into a docset as it lands.
5. On a docset, **Generate** an extraction schema from its files (or paste RNC), then
   **Extract**.
6. Open a file: the pages on the left, the extracted values and the DGML XML on the right.
   Click a value or an XML element with a `dg:origin` to see where it came from on the page.

### The Data page

A read-only view of what the workspace has stored, scoped to the current organisation:

- **S3 bucket**: browse `<folder>/<organisation_id>/` folder by folder — the source PDF,
  `page_images/`, `page_text/` (DGML's word boxes), docset schemas and each pair's
  `.dgml.xml` — and preview images, PDFs, JSON and XML inline. Keys are always joined under
  the workspace prefix, so another tenant's objects can't be reached.
- **Postgres tables**: DGML's DocStore tables (`dgml_workspaces`, `dgml_docsets`,
  `dgml_files`, `dgml_assignments`) and the service's own (`organisations`,
  `dgml_settings`, `service_credentials`, `jobs`), showing column types, the organisation's
  rows only, and paging. Tables come from a fixed allow-list. Secret values are always
  replaced by `•••• redacted`.

Previews are served with `Content-Security-Policy: sandbox` and `nosniff`, so a stored
object never runs as a page of the app.

## How it uses DGML

Everything goes through public `dgml_core` calls:

| Operation | Library call |
|---|---|
| Open an organisation's workspace | `Workspace.open(configuration=Configuration.build(...))` ([tenancy.py](src/dgml_sample_service/tenancy.py)) |
| Docset CRUD, schema, guidance | `DocSetStore(ws).create / get / update / delete / set_schema / set_guidance …` |
| Upload a file | `FileStore(ws).add(path, text_mode=…)` — soft failures come back on `AddFileResult` |
| Classify | `classify_file(ws, file_id, config=load_classification_config(ws), mode=ClassifyMode.EXISTING \| EXISTING_OR_NEW)` |
| Assign (+ auto-extract) | `add_file_and_extract(ws, docset_id, file_id)` |
| Generate a schema | `grounded.generate_schema(ws, file_ids, config=load_grounded_config(ws), docset_name=…)` |
| Extract / re-extract | `extract_file(ws, docset_id, file_id)` |
| A file's docsets | `DocSetStore(ws).docsets_for_file(file_id)` |
| Read the pair's DGML and values | the `*.dgml.xml` blob under `layout.docset_pair_prefix(…)`, projected with `extraction_xml.dgml_xml_to_values` (as `dgml extraction get-values` does) |

The workspace `Configuration` built from an organisation's settings:

- **identity**: `workspace_id` = organisation UUID, `name` = its name, `organization` = its
  slug (the `http://dgml.io/<slug>/…` namespace segment, fixed at creation);
- **storage**: docs → `dgml_sample_service.docstore:PostgresDocStore` with
  `organisation_id`; blobs → `dgml_storage_s3:S3BlobStore` with the bucket, endpoint,
  region, folder as `prefix` (objects land under `<folder>/<organisation_id>/`), and
  credentials by value;
- **models**: `Models(family=…, <provider>_api_key=…)` — the key goes in by value, never
  through the process environment;
- **pdf**: `Pdf(provider="pypdfium2")`; **ocr**: `Ocr(provider="macos")` when chosen.

Opened workspaces are cached per organisation and rebuilt whenever the settings — and so
the `Configuration` — change.

### The Postgres DocStore

[docstore.py](src/dgml_sample_service/docstore.py) keeps DGML's four state collections in
typed tables, keyed by `organisation_id`:

| Collection | Table | Notes |
|---|---|---|
| `workspace` | `dgml_workspaces` | One row per organisation; `workspace_id` is not stored — it *is* the organisation id. |
| `docsets` | `dgml_docsets` | `key_questions text[]`. |
| `files` | `dgml_files` | `added_at timestamptz`, formatted back exactly as DGML wrote it. |
| `assignments` | `dgml_assignments` | Keyed `docset_id`/`file_id`, indexed by file. |
| `errors`, `extraction_stats`, `usage` | — | **Write-only outlets** to the log: reads return nothing. |

It is strict: an unknown collection, an unknown or missing field, a timestamp it cannot
round-trip, or a query outside the allow-list (an empty query everywhere, plus
`docset_id` / `file_id` on assignments) raises `InvalidArgument` before anything is
written. A dgml-core upgrade that changes a record shape fails the round-trip tests instead
of silently dropping a column. There are no foreign keys between the `dgml_` tables, because
DGML orders its own multi-step writes. The host binds its engine once at startup
(`PostgresDocStore.bind_engine(engine)`), so configuration carries identity only.

The Postgres driver is **pg8000** (BSD-3): `psycopg`/`psycopg2` are LGPL, which this
repository bans as direct dependencies.

### Settings

Stored in `dgml_settings` (model family, text mode, OCR provider, S3 connection and folder)
plus `service_credentials` for secrets (LLM key, S3 keys). Rules:

- The model family, keys and text settings can change at any time.
- The storage location **locks once the workspace holds a file** (`409 STORAGE_LOCKED`). An
  in-memory `Configuration` has no storage seal, so moving it would silently open an
  empty workspace elsewhere.
- Any storage change is **probed** first: an object is written and deleted under the
  workspace prefix. Optionally the bucket is created.
- Secrets are **write-only** in the API: reads say only whether one is set, and when.
  `service_credentials` is a sample stand-in for a vault; a real deployment would keep only
  a reference there.

### Jobs

Adding a file, classification, schema generation and extraction are slow, so they run as
background jobs on a thread pool; their state is in the `jobs` table, and the UI polls it.
A per-(organisation, file) lock keeps two jobs from writing one file's `.dgml.xml` at
once. Jobs left unfinished by a restart are marked `INTERRUPTED`.

### Errors

DGML errors map to HTTP **by class**, never by message: `NotFoundError` → 404,
`ConflictError` → 409, `InvalidArgument` / `UnsupportedFileType` / `InvalidPDF` /
`SchemaInvalid` → 422, other `DgmlError` → 500. Every error body is
`{"error": {"code", "message"}}`, and the code is DGML's stable one.

## Limits (it is a sample)

- **No auth.** Anyone who can reach the API can act for any organisation.
- **One process.** The in-process thread pool and file locks don't span replicas; use a
  durable queue and Postgres advisory locks for that.
- **Secrets live in a plain Postgres table.**
- **No full document generation.** DGML's `docset generate` (the complete semantic tree with
  `dg:origin` on every element) is orchestrated inside the CLI with no library entry point
  yet, so the viewer shows what extraction writes. The XML viewer already renders a
  generated tree when a `.dgml.xml` holds one.
- **Retry needs a successful add.** Retrying a file whose rendering or text extraction
  failed needs a per-file reprocess call that dgml-core doesn't expose yet. Extraction can
  be re-run any time.
- **LLM usage isn't recorded**: dgml-core records usage only in debug mode.

## Tests

```bash
uv run pytest samples/dgml-sample-service
```

The suite runs offline: SQLite stands in for Postgres (the store's SQL is portable
SQLAlchemy Core), `moto` for S3, and the model calls are monkeypatched. The frontend
type-checks and builds with `npm run build` in `frontend/`.
