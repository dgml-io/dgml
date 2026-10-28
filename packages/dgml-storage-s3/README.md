# dgml-storage-s3 — sample blob store

**A sample, not a supported product.** It exists to prove the DGML storage
abstraction actually holds on a backend that is not local disk, and to serve as a
reference for anyone writing their own. Pair it with a document store (e.g.
[`dgml-storage-mongo`](../dgml-storage-mongo), or the bundled local store) for the
document half of a workspace.

Blobs live in an S3-compatible bucket — the object API `BlobStore` was modelled
on, so nearly every method is a one-line delegation.

## Use it

```toml
# <workspace>/config.toml — S3 blobs + Mongo docs (local dev against SeaweedFS)
[storage.default.blobs]
provider = "dgml_storage_s3:S3BlobStore"
bucket = "dgml-dev"
endpoint_url = "http://localhost:9000"

[storage.default.docs]
provider = "dgml_storage_mongo:MongoDocStore"
mongo_database = "dgml_dev"
```

Drop `endpoint_url` and set `region` to run against real AWS S3. A local server is
not a separate backend — it speaks the S3 API, so it is only a different address.
Omit the `[storage.default.docs]` table to keep documents on local disk and put
only blobs in S3.

| option | required | default |
|---|---|---|
| `bucket` | yes | — |
| `region` | no | boto3's default |
| `endpoint_url` | no | real AWS S3 |
| `prefix` | no | `dgml` |

**Every object goes under `<prefix>/<workspace id>/`**, so any number of workspaces — and
other applications — can share one bucket, and one config can serve them all. The id is
added when the store is opened, from the `workspace_id` in the workspace's `config.toml`;
it is not written into `prefix`. So with no `prefix` a workspace's objects are under
`dgml/<id>/`, and with `prefix = "contracts"` under `contracts/<id>/`. An empty
`prefix = ""` puts them directly under `<id>/`.

## Credentials

**By value, for an in-memory configuration.** A workspace configured in code
(`dgml_core.configuration`, nothing written to disk) may pass `aws_access_key_id`,
`aws_secret_access_key` and `aws_session_token` as options; they go straight to the
boto3 client and take precedence over its chain. The storage seal ignores them by
name. Do not put them in a `config.toml`.

**Never put credentials in DGML config.** S3 uses boto3's default chain
(`AWS_ACCESS_KEY_ID`, `~/.aws/credentials`, IAM role).

This is not stylistic. `dgml_core.storage_resolve` decides what is a secret by
**substring match on the option name** (`key`, `secret`, `token`, `password`,
`credential`); anything else is persisted to the workspace's plaintext `config.toml`. If
you add an inline-credential option, its name must contain one of those
substrings.

## Run the tests

```bash
docker compose -f packages/dgml-storage-s3/docker-compose.yml up -d
DGML_TEST_S3_ENDPOINT=http://localhost:9000 \
DGML_TEST_MONGO_URI=mongodb://localhost:27017 \
  uv run pytest packages/dgml-storage-s3
```

With those variables unset the suite still runs, against in-process fakes (`moto`
for S3, `mongomock` for the composed S3+Mongo test) — so it never silently skips.
The fakes verify *our* logic; the containers verify wire behaviour (pagination,
real error codes, multipart). CI runs both.

The S3 service is [SeaweedFS](https://github.com/seaweedfs/seaweedfs), published on
port 9000 so the endpoint matches any other S3 address. It runs with no identity
config, so it accepts any signature — boto3 will not sign anonymously, so the
commands above still pass dummy credentials and SeaweedFS ignores their values.
Any S3-compatible server works here; this one is a dev dependency, not a
backend DGML knows about.

## If you adapt this for production

It inherits the default path bridges (`materialize`, `staged_write`,
`working_dir`) from `BlobStore` rather than overriding them. That is deliberate —
those defaults are what every third-party store gets for free, so inheriting them
means they are exercised against a real backend.

They stage through `tempfile`, which honours **`$TMPDIR`** — often a RAM-backed tmpfs on
container images (Cloud Run, `emptyDir: {medium: Memory}`, the Fedora/RHEL/Arch
defaults), where a large `staged_write` (~1.5 GB for a 500-page document) allocates into
memory and counts against the cgroup limit. Point `TMPDIR` at real disk there, and set it
before the process starts — Python memoizes `gettempdir()` on first call.

Do **not** override the bridges to stage under `StorageConfig.root` instead: scratch
space is a property of the deployment, not of a workspace, and on those platforms the
workspace root is tmpfs too. One environment variable also covers every other `tempfile`
user in DGML, including the ghostscript subprocess, rather than this store alone.

Either way it is *survivable, not small* — `staged_write` holds the whole batch before
any upload, since the caller writes its files whenever it likes.

## Licences

| component | licence | how used |
|---|---|---|
| `boto3` | Apache-2.0 | dependency of this package ✅ |
| `moto`, `mongomock` | Apache-2.0 / ISC | dev/test only ✅ |
| SeaweedFS server | Apache-2.0 | dev/CI container, never redistributed ✅ |
| MongoDB Community | SSPL | dev/CI container (composed test), never redistributed |

The root `CLAUDE.md` bans SSPL and AGPL for **Python packages shipped inside the
wheel**. MongoDB is a network service we invoke, not code we ship — the same
reasoning that already permits ghostscript. `pip-licenses` will not flag it,
which is why it is written down here.
