# Batch mode

`--batch` sends DGML's model calls through a provider's **batch API** instead of
one call at a time. Batch APIs bill every token at **half the standard price**,
including prompt-cache reads and writes, in exchange for asynchronous delivery:
results usually arrive within an hour and are guaranteed within 24 hours (48
on Gemini).

Batch mode is an **offline mode**. It trades wall-clock time for cost, so it
suits bulk ingestion, backfills and evaluation runs, not a person waiting at a
terminal. It is opt-in and off by default. Without `--batch`, every command
behaves and prints exactly as it always has.

`dgml docset generate`, `dgml extraction extract`, `dgml extraction
generate-schema` and `dgml file add <dir> --auto-classify` take `--batch`. This
page explains what batches, what does not, and why. For flags and payload
fields, see the [CLI reference](cli-reference.md).

## Providers

| Provider | Batch backend | Model prefix |
|---|---|---|
| Anthropic (first-party API) | Message Batches | `anthropic/…` |
| Google Gemini (Developer API) | Batch API (`batchGenerateContent`) | `gemini/…` |

**Gemini.** The key comes from the stage's credentials, else `GEMINI_API_KEY`
or `GOOGLE_API_KEY`. A batch travels inline when its body is at most 20 MB,
otherwise as a JSONL file uploaded through the File API. The uploaded file and
the finished batch are deleted once their results are read. A Gemini batch may
wait up to 48 hours before it expires, so its polling deadline is 49 hours.
Cost is litellm's standard price for the reply times 0.5, cached and thinking
tokens included.

On an `anthropic_google` workspace (`[models] family`) the light tier is
Gemini. `file add --batch` classification and the OCR style pass now batch
there too; the first batch release (RFC #222) rejected them with
`BATCH_UNAVAILABLE`.

Any other route has no batch backend and is **rejected before any work starts**
with `BATCH_UNAVAILABLE`, naming the stage and model. That includes Amazon
Bedrock, Vertex AI (`vertex_ai/…`, including Gemini models served there), Azure AI and aggregators such as OpenRouter, even when they
host a Claude model. The provider is decided by how litellm resolves the model
string, so `openrouter/anthropic/claude-…` is an OpenRouter model, not an
Anthropic one.

Batch mode **never falls back to a full-price synchronous run** because a
provider is unsupported. You get an error naming the stage and model instead,
and can rerun without `--batch`.

The batch backend reuses litellm to build and decode requests, so a batch
request is byte-for-byte the request the synchronous path would send. That
relies on litellm internals, so batch mode checks the installed litellm version
against the range the backend is verified on
(`packages/dgml-core/src/dgml_core/batch/compat.py`). An installed litellm
outside that range is also `BATCH_UNAVAILABLE`, with a message naming both
versions. Runs without `--batch` are unaffected by this check.

## `dgml docset generate <docset_id> --batch`

| Call | Batches? | Stage | Round trips | Why / why not |
|---|---|---|---|---|
| Transcription (Pass A) | Yes | `transcribe` | one per page window of the longest document | Documents are independent; windows of one document are chained. |
| Roster planning (draft, then refine) | Yes | `plan` | 2 | One request chain per docset, as a one-unit stage. Unseeded runs only. |
| Gap planning (`--extend-schema`) | Yes | `plan_gaps` | 1 | One request per docset, as a one-unit stage. |
| Labeling (Pass B), closed vocabulary | Yes (unless `--no-batch-label`) | `label` | one per chunk of the longest document (+1 for a section retry) | No document can change the roster the others see. |
| Labeling (Pass B), open or extending vocabulary | Yes (unless `--no-batch-label`), one document at a time | `label` | about one per document (+1 for a section retry or a split chunk) | Each document is labeled against the roster the previous ones grew; see below. |
| Concept descriptions | Yes | `describe` | 1 | One request per docset, after labeling. Skipped when no concept was coined (always under a closed vocabulary). |
| Image style for OCR files (`[style]` enabled) | Yes | `style` | 1 | Pages are independent: every document's pages in one wave, after grounding and before links. |
| Semantic links (propose, verify) | Yes | `links` | 2 | Each document's link plan is independent. |

**Planning and descriptions.** Roster planning, `--extend-schema`'s gap
planning and the descriptions of concepts coined during labeling are each one
request chain per docset, so nothing can share their waves. They still bill at
half price as one-unit stages over the labeling model. The requests are the
synchronous ones, and so are their results and cache files. When labeling
stays synchronous (`--no-batch-label`), the labeling pass's usage row splits into a `batch` part
(planning, descriptions) and a `standard` part (labeling), marked
`context.tier_split: true`.

**Image style.** With the workspace's `style` section enabled, grounding an
OCR file sends one vision request per page. Under `--batch` grounding prepares
those requests instead, and every document's pages go out as one `style`
wave once all documents are grounded, before the link pass (which reads the
styled tree, as it does without `--batch`). The styled DGML is the same as the
synchronous run's.

**Labeling batches one document at a time under an open vocabulary.** Without
a supplied schema, the docset's tag list, the *roster*, grows while documents
are labeled. Each document is labeled against the roster as the previous
documents left it: it reuses the names they coined, instead of inventing a
synonym for the same field. That document-to-document hand-off is how DGML
keeps tags consistent across a docset. A single batch would send every
document's request at once, so no document would see what the others coined;
that was measured to lower cross-document tag consistency, so batch mode never
does it. Instead each document is its own batch stage, in the synchronous
order (pilot stage included): its chunks go out together in one wave, a
section retry or a split chunk in the next, and the roster is updated once the
document is done. Every request, label and roster is byte-identical to the
synchronous run, at half price. `--extend-schema` keeps the vocabulary open,
so it labels this way too.

The price is latency: about one queue round trip per document (two when a
section retry or a split fires). For more than a handful of documents, run
with `--no-wait` and resume from cron or an agent (see [Job mode](#job-mode)).
`--no-batch-label` (or `[generation] batch_label = false`) labels with
ordinary synchronous calls instead, exactly as before batch labeling, when
latency matters more than the labeling half of the bill.

Labeling batches every document at once when the vocabulary is **closed**: a
schema you supplied with `--schema-path` (or one a previous `--schema-path`
run left on the docset), without `--extend-schema`. Then no document can
change the roster, so every document's labeling requests go out together, and
the result is the one labeling them one by one gives.

The transcription, labeling, link and (when enabled) style models are checked
for a batch backend before any work starts. The labeling model is checked
under every vocabulary, because planning, descriptions and links batch over it
even when labeling itself stays synchronous.

## `dgml extraction extract <docset_id> <file_id>... | --all --batch`

| Call | Batches? | Stage | Round trips | Why |
|---|---|---|---|---|
| Value extraction (phase 1), location (phase 3) | Yes | `extraction phase 1` / `3` | 2 | Files and pages are independent; phase 3 needs phase 1's values. |

Both LLM phases of grounded extraction batch across every file: phase 1 (value
extraction) for all files in one wave, the deterministic phase-2 matching
locally, then phase 3 (locating unresolved values on page images) for every
file and page in one wave. A file whose phase 1 takes extra turns (the chunked
protocol, a truncation retry, the permissive-schema fallback) rides extra
waves. One file id keeps the single-file payload (plus the `batch` block).

## `dgml extraction generate-schema <docset_id> --batch`

Schema generation is one request per docset (the configured `schema_model`,
Opus on the default profile), so batch mode is a single round trip. The stored
schema, the payload and the usage row are the same as a synchronous run's,
apart from the `batch` block and `tier: "batch"`. On a small docset this call
can be the largest single line of a run's cost, which is why it batches.

## `dgml file add <dir> --auto-classify [existing] --batch`

| Call | Batches? | Stage | Round trips | Why |
|---|---|---|---|---|
| Classification, `--auto-classify existing` | Yes | `classification` | 1 | No file can add a DocSet, so every request is built from the same list. |
| Classification, default mode (`existing-or-new`) | Yes, in order | `classification` | one per file | Any reply may create a DocSet that the next file's request lists (by id); see below. |
| Auto-extraction of assigned files | Yes | `extraction` | 2 | As `extraction extract`, across every DocSet the files landed in. |
| Hybrid text merge (`--text-mode hybrid` with a `[text_extraction]` model) | No | — | — | Runs inside ingest, page by page; see below. |

Ingests every file, classifies them, then runs the auto-extraction of the
assigned files as one batched extraction across all their docsets.

- **`--auto-classify existing`** classifies every file in one wave: no file can
  create a docset, so each request depends only on the docsets the run started
  with.
- **The default mode** classifies the files in order, **one wave per file**.
  A file's request lists every docset that exists when it is built, by id, in
  its prompt and in its tool schema, and any reply may create a new docset.
  So the request for file *k*+1 cannot be built until file *k*'s reply is in
  and applied, exactly as in the synchronous file-by-file loop. Grouping
  consecutive files would be exact only if none of them could create a
  docset, which nothing guarantees. A directory of *N* files therefore takes
  *N* classification round trips (a file that makes no request, such as one
  with no rendered page, takes none). Prefer `existing` mode for large curated
  ingests where that latency matters.

`--batch` needs `--auto-classify`; without it, a directory add is rejected
before any file is added. A single-file add rejects `--batch` too.

**Why the hybrid text merge stays synchronous.** `--text-mode hybrid` with a
`[text_extraction]` model asks that model to merge each page's digital and
OCR words while the file is ingested. Those calls run inside ingest, page by
page, with a heuristic fallback per request, and the merge model is typically
a local one with no batch API. It runs at standard price.

## How ordering works

Providers process a batch's requests concurrently and return results in any
order. DGML does not rely on that order. It enforces order itself: a request is
only created once every response it depends on has arrived.

1. The pipeline collects the next pending request from every document.
2. It submits them together as one **wave** and waits until the whole wave has
   ended. Requests in one wave never depend on each other.
3. It matches each result to its document by request id, never by position.
4. Each document then decides its next request, if any.

Stage boundaries are the same as without `--batch`: labeling starts only after
every document is transcribed, and results are applied in input order, so the
output is deterministic whichever result arrived first.

## Latency

A run takes as many batch round trips as its longest chain of dependent
requests: the transcription windows of its longest document, then two for
roster planning (unseeded runs), the labeling waves (about one per document
under an open vocabulary; one per chunk of the longest document under a closed
one; none under `--no-batch-label`), one for descriptions when labeling coined a concept, one for image style, and up to
two for links (none when every link plan is already cached). The docset's size
changes how large the batches are, not how many there are.

Each round trip waits in the provider's queue, usually minutes, sometimes
hours, at most 24 hours (48 on Gemini). A 100-page document with 10-page windows needs at
least ten transcription round trips. That is why batch mode is for work nobody
is waiting on. Leave it off for interactive runs and for a handful of
documents, where the savings are cents.

## Job mode

For runs too long to keep a process waiting, add `--no-wait`: the command
submits its wave and exits with a `batch_job` payload naming a job.
`dgml batch resume <job_id>` continues it; responses already received are
replayed at no cost and batches still open are collected, never resubmitted.

```bash
dgml extraction extract <docset_id> --all --batch --no-wait   # prints a job id
# then, from cron every 15–30 minutes:
dgml batch status <job_id>    # read-only: ready / pending / completed / failed
dgml batch resume <job_id>    # when status is ready
```

A plain blocking `--batch` run keeps a job too, silently, so a run that
crashes mid-wave can be resumed instead of paying again. See
[Batch jobs](cli-reference.md#batch-jobs---no-wait-and-dgml-batch) for every
`dgml batch` subcommand and payload.

## What the run reports

Without `--batch`, output and JSON payloads are byte-identical to a synchronous
run. With `--batch`, the generated DGML and cache files are the same as a
synchronous run would produce from the same model replies. Batch mode adds two
things.

**A `batch` block in the JSON payload**, one entry per stage, or the reason it
stayed synchronous:

```jsonc
"batch": {
  "enabled": true,
  "stages": {
    "transcribe": {"waves": 3, "batches": 3, "requests": 5, "batch_ok": 5,
                   "sync_fallbacks": 0, "resubmitted": 0, "failed": 0,
                   "batch_ids": ["msgbatch_…", "…"],
                   "cost_usd": 0.021, "standard_cost_usd": 0.042, "saved_usd": 0.021},
    "plan": {"waves": 2, "batches": 2, "requests": 2, "batch_ok": 2, …},
    "label": {"mode": "per-document", "documents": 3, "waves": 3, "requests": 5,
              "batch_ok": 5, …},
    "links": {"waves": 2, "batches": 2, "requests": 4, "batch_ok": 4, …}
  }
}
```

`waves` is the number of round trips, which is what drives wall time.
`cost_usd` is what was billed, at the tier actually used; `standard_cost_usd`
is what the same responses would have cost synchronously; `saved_usd` is the
difference. For a stage served entirely by the batch they differ by exactly
2x; a request that fell back to a synchronous call counts in `sync_fallbacks`
and narrows the gap. The three are `null` when a response had no price.

The `label` entry carries `"mode"`:

- `"all-at-once"`: closed vocabulary, the usual counters.
- `"per-document"`: open or extending vocabulary, the usual counters (summed
  over every document's stage) plus `documents`, how many were labeled.
- `"sync"`: `--no-batch-label`; the entry is
  `{"skipped": "batch labeling is off (batch_label = false)", "mode": "sync"}`.
 A stage can also
report `{"skipped": "--no-semlinks"}` or
`{"skipped": "every link plan was cached"}`. `plan`, `plan_gaps`, `describe`
and `style` appear only when that call ran (`style` reports
`{"skipped": "no OCR document with a page to style"}` when it is enabled but
nothing needed it).

`extraction extract` and `extraction generate-schema` report one block for the
run, the same counters and cost fields plus `provider` (no `stages`):

```jsonc
"batch": {"provider": "anthropic", "waves": 2, "batches": 2, "requests": 5,
          "batch_ok": 5, "sync_fallbacks": 0, "resubmitted": 0, "failed": 0,
          "batch_ids": ["msgbatch_…", "msgbatch_…"],
          "cost_usd": 0.041, "standard_cost_usd": 0.082, "saved_usd": 0.041}
```

`file add <dir>` reports one such block per stage that ran, keyed
`classification` and `extraction` (the latter with the `docset_ids` it
covered).

**`"tier": "batch"` on usage rows.** Rows in `usage.jsonl` (written under
`--debug`) record which pricing tier was billed, and `cost_usd` on a batch row
is already the batch price. A batch row's `duration_s` includes the time the
request spent waiting in the provider's queue. A scope whose responses span
tiers (some served by the batch, some by synchronous calls) writes one row per
tier, marked `context.tier_split: true`.

## When things fail

- **One request fails inside a batch.** A retryable failure, such as an expired
  request, is resubmitted once in a follow-up wave. If it still fails, or the
  error isn't retryable, that request alone runs synchronously at the standard
  price, and counts in the stage's `sync_fallbacks`.
- **One document or file fails.** It fails the way it would without
  `--batch`, with the same error, and the others carry on: a document whose
  transcription fails is dropped and reported failed; a file whose extraction
  or classification fails gets its error in its result entry.
- **The provider rejects a batch as a whole** (too large, over a queue limit).
  The batch is split in half and each half resubmitted, down to single
  requests; only a single request still rejected runs synchronously. The
  stage's `bisections` counts the splits.
- **A batch create's outcome is unknown** (a timeout or server error after the
  request was sent), or the create is refused at the account's rate limit or
  quota: nothing is resubmitted or run at full price. The wave's other open
  batches are cancelled and the stage fails with `BATCH_EXECUTION_FAILED`.
- **A stage fails as a whole** (every batch refused, polling failed, or a
  batch still unfinished at the polling deadline, the provider's expiry (24
  hours, 48 on Gemini) plus an hour). Open batches are cancelled. Documents that had already finished
  keep their results; one still transcribing is dropped, one still labeling is
  written with a `label_error` (under per-document labeling, so is every later
  document; the resume relabels from the failed one onward), and a failed link stage gives each document a
  `link_error` and still writes it. In `extraction extract` and `file add`, a
  file still in flight gets `BATCH_EXECUTION_FAILED` as its entry. Batch mode
  does not silently rerun the whole wave at full price. The run's job ends
  `failed` with every response it received kept, and the payload's `batch`
  block names it (`"job": {"job_id", "status": "failed", "resume"}`):
  `dgml batch resume <job_id>` picks the work up once the provider is back.
- **The process dies mid-wave** (a crash, `kill -9`, a reboot). Its batches
  stay open at the provider. `dgml batch status <job_id>` shows them and the
  dead process's lease (`stale` once it expires, 10 minutes after its last
  renewal; `dgml batch unlock` clears it at once), and `dgml batch resume`
  collects them without submitting anything twice.

## Configuration

| Setting | Effect |
|---|---|
| `--batch` | Turn batch mode on for this run. |
| `--no-batch` (`docset generate`) | Force synchronous, overriding the config. |
| `[generation] batch = true` | Make batch mode the default for `docset generate` in this workspace. Anything but a boolean is `GENERATION_CONFIG_INVALID`. |
| `DGML_GENERATION__BATCH=true` | The same, from the environment (`true`/`false`, `1`/`0`, `yes`/`no`). |
| `--no-batch-label` (`docset generate`) | Label with ordinary synchronous calls under `--batch` (everything else still batches). `--batch-label` forces batch labeling over a `false` config. Needs `--batch`. |
| `[generation] batch_label = false` | Make synchronous labeling the default under batch mode (default `true`; `DGML_GENERATION__BATCH_LABEL`). Anything but a boolean is `GENERATION_CONFIG_INVALID`. |
| `--batch-poll-interval SECONDS` | Seconds between batch status checks (default 30). |
| `--no-wait` | Submit the wave and exit; continue with `dgml batch resume <job_id>`. |
| `--job JOB_ID` | Continue an existing job (what `dgml batch resume` passes). |

## Measuring before and after

To compare a synchronous and a batch run on the same documents, run each with
`--debug`, then total cost by `tier` and `operation`:

```bash
jq -s 'group_by([.tier // "standard", .operation])
       | map({tier: (.[0].tier // "standard"), operation: .[0].operation,
              rows: length, cost_usd: (map(.cost_usd // 0) | add)})' \
   <workspace>/usage.jsonl
```

The payload's `batch` block reports the batched stages' cost without
`--debug`; synchronous stages are in `usage.jsonl`.
