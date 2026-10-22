# Configuring DGML as a library

An application that embeds `dgml_core` does not need a `config.toml`. It builds a
`Configuration` in code — identity, storage binding, models and any per-task settings,
with credentials passed by value — and opens the workspace with it:

```python
from dgml_core import Configuration, FileStore, Identity, ProviderSpec, Storage, Workspace
from dgml_core.configuration import Model, Models, Ocr

cfg = Configuration.build(
    identity=Identity(workspace_id=tenant.id, name=tenant.name, organization=tenant.org_slug),
    storage=Storage(
        blobs=ProviderSpec("dgml_storage_s3:S3BlobStore", {"bucket": "dgml-prod", "prefix": tenant.id,
                           "aws_access_key_id": secrets["aws_id"], "aws_secret_access_key": secrets["aws_key"]}),
        docs=ProviderSpec("dgml_storage_mongo:MongoDocStore", {"mongo_database": f"dgml_{tenant.id}",
                          "mongo_uri": secrets["mongo_uri"]}),
    ),
    models=Models(
        family="anthropic_google",
        anthropic_api_key=secrets["anthropic"],
        google_api_key=secrets["gemini"],
        expert=Model("anthropic/claude-opus-5", api_key=secrets["opus"]),  # this tier only
    ),
    ocr=Ocr(provider="azure", options={"endpoint": tenant.di_endpoint, "api_key": secrets["azure_di"]}),
)

ws = Workspace.open(configuration=cfg)   # no file, no store row; meta written on first use
FileStore(ws).add(path, file_id=document.id)
```

What this does and does not do:

- **Nothing is read from disk or the environment.** The user-level `~/.config/dgml/config.toml`,
  `DGML_*` variables and the store of workspaces play no part; `cfg` is the whole
  configuration. (One exception by design: a model side with *no* key configured still falls
  back to litellm's own per-provider environment lookup — so a multi-tenant host should keep
  provider keys out of its process environment and pass them in `cfg`.)
- **Nothing leaks through a `repr`.** Every `api_key` field — on `Model`, `Models`, and the
  resolved task configs such as `GenerationConfig` — is left out of its repr, and `ProviderSpec`,
  `StorageConfig` and `Configuration` print every option or section value whose name carries a
  credential word (`key`, `secret`, `token`, `password`, `credential`, `dsn`, `apikey`,
  `passwd`, `auth`) as `***`, nested tables and lists included, so `cfg` can be logged. A
  `*_url` / `*_uri` value is hidden only when it carries `user:password@`, so an S3
  `endpoint_url` shows. Counts such as `max_tokens`, env-var names and model ids stay visible.
- **Nothing is written back.** There is no `config.toml`, no storage seal, no store row. The
  first `open` against a fresh backend writes the workspace's meta document (name,
  organization, id, schema version) to the docstore; later opens check it names the same
  workspace and raise `ConflictError` otherwise. There is no separate create step. A changed
  `organization` is not refused — the workspace is re-organized with a WARNING log record,
  as `dgml workspace create --organization` does — so treat the identity as stable.
- **The `Configuration` is the record of where a workspace lives.** With no seal there is no
  drift detection: `open` opens whatever the object points at, and the same id pointed at a
  different bucket or database is simply a fresh, empty workspace there — not an error, since
  that backend holds nothing to disagree with. Keep each tenant's storage binding stable in
  your own records and treat a change to it as a migration, as `dgml workspace reseal` is on
  the CLI path.
- **`root` is optional when every store is remote.** The workspace gets an empty temp dir for
  its local-only paths, removed when the `Workspace` is garbage-collected. The bundled local
  store's data *is* the root, so `open` refuses to go without one for it (`INVALID_ARGUMENT`);
  pass `root=`, or set `workspace_path` on the store — which then *is* the root, not a temp
  dir beside it (a `root=` that names a different directory is refused). The check is on
  the resolved class, so an alias or subclass of the local store is held to the same rule.
  The temp dir is `ws.scratch`, and nothing lasting goes into it: staking a file, or
  broadcasting a node stake, needs an explicit `output_dir` (`INVALID_ARGUMENT`
  otherwise; a node dry run writes nothing and needs none) and custom chains need an explicit
  `config_path` (built-ins work without one). The embedding cache is not lasting output: it
  lives under the temp dir for as long as the `Workspace` does, so hold the object across
  runs to reuse it.
- **`Identity.workspace_id` must be a valid workspace id** (3–40 chars of `[a-z0-9_-]`, starting
  with a letter or digit), the same rule `create_workspace` applies: it becomes a directory
  name, an S3 key prefix and a Mongo collection prefix. Normalize your own ids at the edge.
- **Tenant = workspace.** Build one `Configuration` per tenant and cache the `Workspace` you
  open with it (`Configuration` is hashable by identity and storage) — the store clients are
  built per `Workspace` object.

## Sections

Each typed section renders to the keys the corresponding `load_*_config` reads; leave a field
`None` and the loader's default applies.

Every model is a `Model(model, api_key=None, api_key_env=None, api_base=None)`: the id plus its
own credentials, by value. It renders flat — `Generation(label_model=Model("x", api_key="k"))`
is `label_model = "x"` / `label_api_key = "k"` in TOML terms. A `Model` is one group: a task's
`Model` uses exactly the credentials it carries, a tier's likewise, and nothing is borrowed from
another tier or section. The provider keys on `Models` (`anthropic_api_key` / `google_api_key` /
`openai_api_key`) stay in the `Configuration`'s `[models]` section as written; each time a
loader reads it, they are written onto every model of that provider — the tiers `Models` sets
or expands from `family`, the section models, and a per-call override's model
(`load_grounded_config(ws, values_model=…)`, a generation profile) — unless the model has its
own key or an `api_base`. So an override runs on this configuration's key, never the server's
environment key. A model with no credentials
leaves litellm its own env var — so at a custom `api_base`, an `openai/`, `anthropic/` or
`gemini/` model needs `api_key` set beside it (a real key, or a placeholder such as `"EMPTY"`),
or litellm sends the provider's env-var key to that host; `ollama/` needs none. See
[the `[models]` tiers](storage-layout.md#the-models-tiers).

| Section | TOML table | Notes |
|---|---|---|
| `Models(family, light, standard, advanced, expert, anthropic_api_key, google_api_key, openai_api_key)` | `[models]` | `family` expands into the four tiers; an explicit tier (a `Model`) wins. A provider key is written onto the tiers and section models of that provider that carry neither their own key nor an `api_base`; one with no such model is simply unused. |
| `Grounded(schema_model, values_model, max_tool_iters, values_reasoning_effort, locations_reasoning_effort)` | `[grounded]` | Extraction (`values_model`, advanced tier) and schema generation (`schema_model`, expert tier). |
| `Classification(model, max_pages, naming_attempts)` | `[classification]` | Light tier. |
| `Ocr(provider, max_concurrency, options)` | `[ocr]` | `options` are the provider's own fields, validated by its `parse_config`. |
| `Pdf(provider)` | `[pdf]` | `ghostscript` (default) or `pypdfium2`. |
| `Style(model, max_tokens, enabled=True)` | `[style]` | Building one enables the feature. |
| `TextExtraction(model, temperature, max_tokens, enabled=True)` | `[text_extraction]` | Same. |
| `Conversion(families={FormatFamily.DOCX: ProviderSpec(...)})` | `[conversion]` | One provider per format family. |
| `Generation(model, label_model, thinking)` | `[generation]` | Both passes default to the standard tier. |
| `Clustering(overrides)` | `[clustering]` | Free-form; validated by the clusterer. |

`Storage` takes a `ProviderSpec` per role, or `Storage.combined(spec)` for one provider serving
both. Provider options — including credential ones such as `mongo_uri` or
`aws_secret_access_key` — pass through to the store's `parse_config` untouched.

## The CLI path is unchanged

A workspace addressed by path or id still has its `config.toml` (or store row), layered under
the user config and the environment. Internally it is the same object: `Workspace.config` is
derived from that merge, so every loader has one read path. That TOML path is the only one
that writes (identity, storage seal).
