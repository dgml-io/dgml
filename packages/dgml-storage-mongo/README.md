# dgml-storage-mongo — sample blob + document stores

**A sample, not a supported product.** It exists to prove the DGML storage
abstraction holds on a backend that is not local disk, and to serve as a
reference for anyone writing their own.

| provider | role | notes |
|---|---|---|
| `MongoDocStore` | documents | manifests, assignments, extraction stats, the usage log — one collection each |
| `MongoGridFSBlobStore` | blobs | a GridFS bucket |
| `MongoGridFSStore` | both | the two above, on one database |

`MongoDocStore` is a near-direct delegation: `DocStore` was modelled on the
MongoDB collection API, so nearly every method is one line. The blob store has
actual work to do — see [Blobs in a document database](#blobs-in-a-document-database).

Note that per-page text is a **blob**, not a document: it lives under
`files/<id>/page_text/` and is read through the blob store.

## Use it

```toml
# <workspace>/config.toml — everything on Mongo (the flat form: one provider for
# both roles, which is only possible because MongoGridFSStore implements both).
[storage.default]
provider = "dgml_storage_mongo:MongoGridFSStore"
mongo_host = "localhost"
mongo_database = "dgml_dev"
mongo_bucket = "acme_blobs"   # optional, default "blobs"
```

Or mix, per role:

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

Omit a role's subtable to leave it on local disk.

| option | required | default |
|---|---|---|
| `mongo_database` | yes | — |
| `mongo_host` | no | `localhost` |
| `mongo_port` | no | `27017` |
| `mongo_bucket` | no | `blobs` — blob stores only |

## Credentials

**Never put credentials in DGML config.** Mongo reads `DGML_MONGO_URI` if set (the
full connection string, including any credentials); otherwise it connects to
`mongo_host:mongo_port` with no auth. All three providers share this, and it
accepts any URI pymongo does — `mongodb+srv://…`, auth, TLS — so a managed
cluster is only a different address.

This is not stylistic. `dgml_core.storage_resolve` decides what is a secret by
**substring match on the option name** (`key`, `secret`, `token`, `password`,
`credential`); anything else is persisted to the plaintext workspace registry. An
option named `mongo_uri` holding `mongodb://admin:hunter2@host` matches none of
them, so the password would be written out in the clear.

## Blobs in a document database

MongoDB caps a BSON document at 16 MB and a source PDF has no size bound, so a
blob has to span documents. That is exactly what GridFS specifies, so this store
delegates to it rather than hand-rolling a chunk layout — which also means the
bytes are readable by any official MongoDB driver, and that download streams
seek, so a large artifact can be served by byte range.

The collections are the spec's: **`<bucket>.files`** (one document per revision —
`filename`, `length`, `chunkSize`, `uploadDate`, `metadata`) and
**`<bucket>.chunks`** (`files_id`, `n`, `data`). Neither collides with a
`dgml_core.layout.Collection` member, which is what lets blobs and documents
share one database.

The blob key is the GridFS `filename`. Chunks are 1 MiB rather than the 255 KiB
GridFS default: at 255 KiB a 300 DPI page image is a dozen documents, and page
images are the bulk of a workspace.

### Revisions — the one real friction

GridFS addresses a blob by `filename` **plus** `uploadDate`; a `BlobStore`
addresses it by key alone. Reconciling those is the substance of this class, and
both halves are measured rather than assumed:

1. **GridFS versions instead of replacing.** `upload_from_stream(name, …)` mints a
   new file id and leaves the prior revision in place. So `put_blob` captures the
   prior revision ids before uploading and deletes them after, and `list_blobs`
   de-duplicates across revisions. A port that forgets either grows one revision
   per re-render, forever.
2. **Revision order is ambiguous at millisecond resolution.** `revision=-1` means
   "most recent by `uploadDate`", and `uploadDate` has millisecond precision. Two
   writes to one key inside the same millisecond leave the ordering genuinely
   undefined — and the failure mode is a *silent stale read*. This is not
   theoretical: writing a key twice in a row and reading it back returns the
   **first** revision, both carrying an identical `uploadDate`. (Observed under
   `mongomock`; the tie is in the data rather than the fake, so a real server is
   equally entitled to pick either — "undefined" is the problem, not which side
   you land on.)

So `open_download_stream_by_name` is deliberately **never called**. Reads resolve
the revision explicitly and break the `uploadDate` tie on `_id`. That narrows the
ambiguity to two processes writing one key inside the same second rather than
closing it — `ObjectId` is monotonic within a process, not across them. DGML
never has two writers on one key (`staged_write` regenerates a whole prefix from
one process), so it is sound for the pipeline; keep it in mind before pointing
two writers at one workspace.

Ordering the write as *upload, then collect the old revision* gives the same
"authoritative record dies last" property as the cascade contract on
`delete_blobs`: a crash leaves an orphaned revision (recoverable) rather than a
key with no bytes behind it.

**Torn reads are retried, not raised.** GridFS reports a collected revision as
`CorruptGridFile`; this store re-resolves and converges on the current bytes
rather than handing that exception to a caller mid-pipeline.

### `sha256_blob` is a lookup, not a download

The digest is computed at write time and stored in GridFS `metadata`, so hashing
a blob is one indexed document read. This is the one place this backend beats an
object store: attestation over a 200-page file becomes 200 index hits instead of
200 downloads, and on S3 there is no shortcut — the multipart ETag and composite
`ChecksumSHA256` are checksums-of-checksums and explicitly not the value the
`BlobStore` contract requires.

`test_composed.py` pins that shortcut to the contract: the same artifacts must
produce the same attestation Merkle root on Mongo and on local disk.

## If you adapt this for production

Blobs generally belong in an object store. Honest trade-off:

| | Mongo blobs | S3 blobs |
|---|---|---|
| `sha256_blob` | one indexed read ✅ | full re-read of the object |
| Storage cost | roughly 10× S3 per GB | the cheap option ✅ |
| Replication | every page image flows through the oplog to every secondary | none of your business ✅ |
| Overwrite | revision flip + retry-on-torn-read | atomic, read-after-write ✅ |
| Operational surface | one backend, one credential ✅ | a second service to run |
| Presigned URLs, lifecycle tiering | none | ✅ |
| Ranged reads | ✅ | ✅ |

Page images are the bulk of a workspace, the most regenerable bytes in it, and
the least query-worthy — paying database prices and oplog bandwidth for them is
the main thing to think twice about. The `sha256_blob` win is real but separable:
you can cache digests in the *document* store at write time and keep blobs on S3.

Also unbuilt here: no compression, and orphaned revisions from an interrupted
write are never swept (a periodic job reconciling `<bucket>.files` against what
`list_blobs` reports would do it).

### What sharing a database does *not* buy you

Atomic cascades. With both halves in one database a transaction spanning a
document delete and its blob deletes is theoretically available — but `BlobStore`
and `DocStore` expose no transaction handle, and `delete_blobs` is documented to
run last precisely because cross-store atomicity is assumed impossible. Capturing
it would take an interface change in `dgml-core`, not a provider.

### What the flat form *does* buy you

One instance. Both roles resolve to the same config, so `Workspace` constructs the
provider once and serves `blobs` and `docs` from it — a flat-form workspace holds a
single `MongoClient`, not one per role. This keys off the resolved config rather than
the syntax, so two per-role subtables naming the same database and options collapse
the same way.

## Run the tests

```bash
docker compose -f packages/dgml-storage-mongo/docker-compose.yml up -d
DGML_TEST_MONGO_URI=mongodb://localhost:27017 uv run pytest packages/dgml-storage-mongo
```

With that variable unset the suite still runs, against in-process `mongomock` — so
it never silently skips. No ghostscript or PDF library is needed either; the
composed tests place artifacts directly.

**One trap if you touch this package.** `mongomock` does not drive `GridFSBucket`
under current `pymongo` out of the box: `mongomock.MongoClient` has no `.options`
property, so GridFSBucket's `db.client.options.timeout` falls through mongomock's
attribute-style database accessor and `_timeout` becomes a `Collection`, tripping
pymongo's timeout wrapper on first upload. The `_mongomock_gridfs` fixture in
`tests/conftest.py` closes that in three lines. The reason it matters: the
original failure *leaks a context variable* before raising, so subsequent GridFS
calls start passing and a suite can look green on the strength of one poisoned
test. **Check GridFS tests individually as well as in a batch.**

## Licences

| component | licence | how used |
|---|---|---|
| `pymongo` | Apache-2.0 | dependency of this package ✅ (`gridfs` ships inside it — nothing extra) |
| `mongomock` | ISC | dev/test only ✅ |
| MongoDB Community | SSPL | dev/CI container, never redistributed |

The root `CLAUDE.md` bans SSPL for **Python packages shipped inside the wheel**.
The MongoDB server is a network service we invoke, not code we ship — the same
reasoning that already permits ghostscript. `pip-licenses` will not flag it, which
is why it is written down here. [FerretDB](https://ferretdb.com) (Apache-2.0)
speaks the MongoDB wire protocol and is a drop-in for the compose file if the SSPL
question ever becomes live; `pymongo` talks to both unchanged. That holds for the
blob store too — GridFS is a client-side convention over ordinary collections,
with no server-side feature behind it.
