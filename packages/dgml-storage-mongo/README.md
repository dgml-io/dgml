# dgml-storage-mongo — sample document store

**A sample, not a supported product.** It exists to prove the DGML document
storage abstraction holds on a backend that is not local disk, and to serve as a
reference for anyone writing their own. Pair it with a blob store (e.g.
[`dgml-storage-s3`](../dgml-storage-s3), or the bundled local store) for the blob
half of a workspace.

DGML documents (manifests, page text, assignments, the usage log) live in MongoDB
collections — the collection API `DocStore` was modelled on, so nearly every
method is a one-line delegation.

## Use it

```toml
# <workspace>/config.toml — S3 blobs + Mongo docs
[storage.default.blobs]
provider = "dgml_storage_s3:S3BlobStore"
bucket = "dgml-dev"
endpoint_url = "http://localhost:9000"

[storage.default.docs]
provider = "dgml_storage_mongo:MongoDocStore"
mongo_host = "localhost"
mongo_database = "dgml_dev"
```

Omit the `[storage.default.blobs]` table to keep blobs on local disk and put only
documents in Mongo.

## Credentials

**Never put credentials in DGML config.** Mongo reads `DGML_MONGO_URI` if set (the
full connection string, including any credentials); otherwise it connects to
`mongo_host:mongo_port` with no auth.

This is not stylistic. `dgml_core.storage_resolve` decides what is a secret by
**substring match on the option name** (`key`, `secret`, `token`, `password`,
`credential`); anything else is persisted to the plaintext workspace registry. An
option named `mongo_uri` holding `mongodb://admin:hunter2@host` matches none of
them, so the password would be written out in the clear.

## Run the tests

```bash
docker compose -f packages/dgml-storage-mongo/docker-compose.yml up -d
DGML_TEST_MONGO_URI=mongodb://localhost:27017 uv run pytest packages/dgml-storage-mongo
```

With that variable unset the suite still runs, against in-process `mongomock` — so
it never silently skips.

## Licences

| component | licence | how used |
|---|---|---|
| `pymongo` | Apache-2.0 | dependency of this package ✅ |
| `mongomock` | ISC | dev/test only ✅ |
| MongoDB Community | SSPL | dev/CI container, never redistributed |

The root `CLAUDE.md` bans SSPL for **Python packages shipped inside the wheel**.
The MongoDB server is a network service we invoke, not code we ship — the same
reasoning that already permits ghostscript. `pip-licenses` will not flag it, which
is why it is written down here. [FerretDB](https://ferretdb.com) (Apache-2.0)
speaks the MongoDB wire protocol and is a drop-in for the compose file if the SSPL
question ever becomes live; `pymongo` talks to both unchanged.
