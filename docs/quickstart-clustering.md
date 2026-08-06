# Quickstart — ingest a folder of PDFs and cluster them

End-to-end walkthrough: install DGML, ingest every PDF in a folder
(optionally including sub-folders), and group them into DocSets with
`dgml cluster`.

Throughout, replace `/path/to/pdfs` with your input directory and
`./dgml-workspace` with wherever you want the workspace to live.

## 1. Install

System dep first — ghostscript is required for page-image rendering:

```bash
brew install ghostscript            # macOS
sudo apt-get install ghostscript    # Debian/Ubuntu
```

Install DGML with the `clustering` extra (pulls in `dgml-clustering`
and its ML stack — embedding models, `leidenalg`, `scipy`, `sklearn`).
DGML is not published to PyPI yet, so install from a clone of this
repository:

```bash
git clone https://github.com/dgml-io/dgml.git
cd dgml
uv sync --extra clustering
```

(Once DGML is on PyPI this becomes `pip install "dgml[clustering]"`.)

Sanity-check the CLI:

```bash
uv run dgml --help
```

The commands below assume the repo venv is active (`source .venv/bin/activate`)
or that you prefix each `dgml` invocation with `uv run`.

## 2. Create a workspace

The workspace is a directory holding `docsets/` and `files/`. Anything
the CLI writes goes there.

```bash
export DGML_HOME=./dgml-workspace
dgml init --provider anthropic             # write ~/.config/dgml/config.toml
dgml workspace create --organization Acme  # create the workspace
```

`DGML_HOME` is optional — without it, pass `--workspace ./dgml-workspace`
to every command, or `dgml` will fall back to a `./dgml-workspace`
folder relative to the current directory.

## 3. Ingest a folder of PDFs

Point `dgml file add` at a directory. `--recursive` walks sub-folders;
`--on-conflict skip` makes re-runs idempotent (existing files are
returned untouched instead of erroring):

```bash
dgml file add /path/to/pdfs --recursive --on-conflict skip --text-mode hybrid
```

What this does, per PDF:

- copies it into `<workspace>/files/<file_id>/`,
- hashes the bytes (sha256),
- renders each page to a 300 dpi PNG via `gs`,
- extracts per-page word boxes with `pdfminer.six` (default
  `--text-mode digital`).

The command returns a single JSON envelope with a `summary` block and a
per-file `files` array — inspect it with `jq`:

```bash
dgml file add /path/to/pdfs --recursive --on-conflict skip | jq .summary
```

```jsonc
{
  "total": 42,        // PDFs found
  "added": 40,        // new File records
  "skipped": 2,       // already in the workspace
  "soft_failed": 0,   // record created, but a step (render/text) failed
  "hard_failed": 0    // PDF rejected outright (bad bytes, etc.)
}
```

If anything looks off, `dgml check` walks the workspace and reports
inconsistencies; `dgml check --retry-errors` re-attempts permanent
failures (failed renders, failed text extraction).

### Scanned PDFs?

If your folder is image-only scans with no embedded text, swap the
text-mode. Each provider needs an extra; the cloud ones (Azure, AWS)
are also configured in `<workspace>/config.toml` (see
[`docs/cli-reference.md`](cli-reference.md#ocr-configuration) for the
schema), while macOS Apple Vision runs on-device with no config:

```bash
# `uv sync` makes the venv match exactly what you list, so keep the
# clustering extra from step 1 and add the OCR provider you need:
uv sync --extra clustering --extra macos   # Apple Vision — on-device, macOS only, zero-config
# or, for cloud OCR (add an `ocr` section to config.toml first):
uv sync --extra clustering --extra azure   # Azure Document Intelligence
uv sync --extra clustering --extra aws     # AWS Textract

dgml file add /path/to/pdfs --recursive --on-conflict skip --text-mode hybrid
```

(Once DGML is on PyPI these become `pip install "dgml[macos]"` etc.)

On macOS, Apple Vision is the default OCR engine even with no `ocr`
section in `config.toml` — just install the extra. `hybrid` runs
digital extraction first, then OCR, and merges the two — the right
default when a folder mixes born-digital and scanned PDFs.

## 4. Cluster the unassigned files into DocSets

By default `dgml cluster` only touches files that aren't already in a
DocSet — exactly the state you're in after a fresh ingest. With no
existing DocSets it runs a fresh **S1 (unsupervised)** clustering; with
existing DocSets it switches to **incremental** mode and grows them
(**S3** few-shot when the DocSets have members, **S2** name-only
otherwise). That's the `--mode auto` default; force either side with
`--mode fresh` / `--mode incremental` (see
[`docs/incremental-clustering.md`](incremental-clustering.md)).

Not sure which regime fits your case — or need the closed-set (S4/S5)
scenarios where every document is forced into a known category? See
[Choosing a clustering scenario](choosing-a-clustering-scenario.md).

```bash
dgml cluster
```

The command:

1. embeds each file from its `page_text` (the bundled default is a
   corpus-fitted TF-IDF text encoder over the first page; a file still
   needs a rendered first-page image to be eligible),
2. clusters them in the configured manifold,
3. for clusters that match an existing DocSet's name, assigns the files
   to that DocSet,
4. for unmatched clusters, calls the configured vision LLM to propose
   `(name, description)`, creates the DocSet, and assigns the files.

Step 4 needs the `classification` section in `<workspace>/config.toml`
(LLM model id + API key env var) — same config used by
`dgml file add --auto-classify`. Without it, matched clusters still get
assigned and unmatched ones land in `failed_file_ids`; re-run after
filling the config in. See
[`docs/cli-reference.md`](cli-reference.md#auto-classification) for the
exact shape.

Response (truncated):

```jsonc
{
  "clusters": {
    "k7q3xb91pmrf": "Contracts",
    "abc123def456": "Receipts",
    "xyz789":      "Property Tax Bill"   // newly-proposed DocSet name
  },
  "failed_file_ids": []
}
```

### Only a handful of documents?

The embedding pipeline above needs a corpus large enough for its statistics to
mean something — tf-idf has almost nothing to weight on a few documents, k-NN
graphs are dominated by noise, and clusters collapse into one bucket (or all
noise). For very small corpora, skip embeddings entirely and let the vision LLM
partition the documents directly:

```bash
dgml cluster --method llm
```

`--method llm` sends every document's rendered first pages to the LLM in a
single call and asks it to group them by document type, then names each
emergent group — the same vision machinery `dgml file add --auto-classify`
uses, so it needs the same `classification` section in `<workspace>/config.toml`
(without it, every file lands in `failed_file_ids`). It partitions *and* names
in one round-trip, and a single call covers up to 24 files.

Prefer `--method auto` to let DGML choose: it routes corpora of at most
`--small-corpus-threshold` files (default 8) to the LLM and larger ones to the
embedding pipeline — the right default for a folder whose size you don't know
up front.

```bash
dgml cluster --method auto                          # LLM for ≤8 files, else embedding
dgml cluster --method auto --small-corpus-threshold 12   # raise the cutoff
```

The `--method embedding` default (used by the plain `dgml cluster` above) is
unchanged, so existing large-corpus runs behave exactly as before.

## 5. Tune the clustering (optional)

The defaults cluster a folder sensibly out of the box; everything here is
optional. There are two ways to override them, both using the same field
schema:

- **Per workspace** — add a `clustering` section to
  `<workspace>/config.toml`. It's a *partial overlay*, deep-merged over
  the bundled defaults, so you only spell out what you change.
- **Per run** — `dgml cluster --config PATH` points at a standalone JSON
  with the same fields (drop the `clustering` wrapper); it *replaces* the
  section for that run. `--config` also accepts a bundled preset **name**
  (`small` / `light` / `medium` / `heavy`).

```toml
# <workspace>/config.toml — change only what you need
[clustering.encoder_text]
name = "bge"
model_id = "BAAI/bge-small-en-v1.5"
embedding_dim = 384
[clustering.manifold]
name = "euclidean"
dim = 384
[clustering.scenario]
leiden_resolution = 0.7
leiden_k_neighbors = 20
```

Field names and value enums come from the `Config` schema
([`packages/clustering/src/clustering/config/schema.py`](../packages/clustering/src/clustering/config/schema.py));
a typo or out-of-enum value fails the next run with
`CLUSTERING_CONFIG_INVALID`. The scenario *regime* (`name`,
`known_categories`, `n_shots`) is chosen automatically from the workspace
state, so overriding those is ignored — but every algorithm knob
(`cluster_algorithm`, `leiden_*`, `hdbscan_*`, `reduce_*`, `threshold*`)
*is* honored.

### Compute presets

Each preset is a complete, self-contained config tuned for a hardware
budget. Higher tiers add **image/vision embeddings** for better separation
at the cost of more compute (and a model download / GPU).

| Preset | Target hardware | Representation | Clustering |
|---|---|---|---|
| `small` | CPU-only, tiny corpora | `tfidf` text, 256-d | Leiden, no UMAP |
| `light` (default) | CPU-only | `tfidf` text, 256-d | Leiden + UMAP |
| `medium` | large CPU / Apple MPS | `tfidf` text + 2B vision, fused 1280-d | Leiden + UMAP |
| `heavy` | GPU | 8B vision only, 1024-d | Leiden + UMAP |

`small` drops UMAP (`reduce_method: none`) and uses a small k-NN graph
(`k=5`) — meant for corpora too small for UMAP to help. `medium` fuses the
tf-idf text vector with a `Qwen3-VL-Embedding-2B` image embedding
(`fusion: concat_norm`); `heavy` clusters on the larger
`Qwen3-VL-Embedding-8B` image embedding alone.

```bash
dgml cluster --config medium
```

Copy one and pass its file path to `--config` as a starting point for a
custom config.

### Parameters and when to change them

The default pipeline is: **TF-IDF text encoder → UMAP reduction → Leiden
community detection**. The knobs below are grouped by stage; set each
under its config section (e.g. `scenario.leiden_resolution`,
`encoder_text.name`). Defaults are the shipped values in
[`clustering_config.json`](../packages/dgml-core/src/dgml_core/clustering_config.json).

**Representation — how each document is turned into a vector**

| Parameter (section) | What it controls | Default | Raise / switch up when… | Lower / switch down when… |
|---|---|---|---|---|
| `encoder_text.name` | Text embedding model. `tfidf` (bag-of-words, fast, CPU) vs dense sentence encoders `bge` / `e5` / `gte` (semantic, need a model download). | `tfidf` | Categories differ by *meaning*, not vocabulary; short docs; TF-IDF under-separates. Switch to a dense encoder (`bge` / `e5`), or add a vision encoder as the `medium` / `heavy` presets do. | You want zero downloads / CPU-only speed and the vocabularies are already distinctive. |
| `encoder_text.embedding_dim` + `manifold.dim` | Vector width. Must match the encoder (`tfidf` 256, `bge` 384, `e5` 1024). Keep these two equal. A multi-view `text_view` (below) shares this width across its views, so give it 256 per view. | 256 | Switching to a wider encoder, or combining text views. | Switching to a narrower encoder. |
| `encoder_text.extra.text_view` | Which text `tfidf` embeds: `page1` (first page only), `full` (every page), `headers` (just the title/header words, picked out by font size and page position) or `salient_boost` (those headers repeated ahead of the full body). `headers` is not recommended on its own: across four internal corpora it scored below *both* `page1` and `full` on all four, by 0.145 mean NMI against the default. | `page1` | The first page doesn't characterize the doc (cover pages, boilerplate); use `full`. | First pages are highly distinctive (forms, letterheads) — cheaper and less noisy. |
| `encoder_text.extra.text_view` — **several views at once** | Views can be combined with `+` (`page1+full+salient_boost`). Rather than one bag of words, `tfidf` then fits an independent block per view — its own vocabulary, document frequencies and SVD basis — and stacks them, so the reducer can use whichever view separates a given pair of documents. | one view | You don't know which view suits the corpus, which is the usual case: no single view wins everywhere. Across four internal corpora this raised mean NMI from 0.63 to 0.69 and beat the single-view default on **all four**. | You want the cheapest possible encoder — it costs one TF-IDF fit and one SVD per view — or the corpus is small enough that one view already separates it. |

> **A combined `text_view` needs the width to go with it.** The views *share*
> `encoder_text.embedding_dim`, so three views at the default 256 leaves each with
> only 85 components — measured, that is *worse* than a plain `page1` run on a
> corpus with enough documents for 256 components to be real. Set `embedding_dim`
> (and `manifold.dim`) to 256 × the number of views:
>
> ```json
> {"clustering": {
>   "encoder_text": {"name": "tfidf", "embedding_dim": 768,
>                    "extra": {"text_view": "page1+full+salient_boost"}},
>   "manifold": {"name": "euclidean", "dim": 768}
> }}
> ```

**Reduction — compress before clustering** (`scenario.*`)

| Parameter | What it controls | Default | Raise it when… | Lower it when… |
|---|---|---|---|---|
| `reduce_method` | Dimensionality reducer applied before clustering (`umap`, `pca`, …, or `none`). High-dim distances concentrate and hurt clustering, so reducing first is standard. | `umap` | — | Set to `none` only for very low-dim encoders or debugging. |
| `reduce_dim` | Target dimensionality (`0` = off). | `10` | Clusters are collapsing/merging distinct categories — keep more structure (try 15–30). | Results are noisy/fragmented — squeeze to 5–10 to denoise. |

**Leiden — the default community detection** (`scenario.*`). *The first
knob to reach for is `leiden_resolution`.*

| Parameter | What it controls | Default | Raise it when… | Lower it when… |
|---|---|---|---|---|
| `leiden_resolution` | Community granularity — the main over/under-clustering dial. | `1.0` | **Under-clustering** (distinct categories merged into one cluster) — raise toward 1.5–2. | **Over-clustering** (one true category split across clusters; high homogeneity, low completeness) — lower toward 0.5–0.8. |
| `leiden_k_neighbors` | `k` for the k-NN graph the communities are found on. More neighbors → denser graph → fewer, larger clusters. | `25` | Graph is fragmenting into too many clusters; or a large corpus. | Small corpus (**must** be `< n_docs`; on tiny sets drop to ~5–10) or you want finer clusters. |
| `leiden_graph_method` | Graph construction: `knn`, `mutual_knn` (stricter, drops one-way edges), `radius`. | `knn` | Use `mutual_knn` to break weak bridges when unrelated docs get glued together. | Stay on `knn` for well-connected small corpora. |
| `leiden_min_cluster_size` | Communities smaller than this are dropped to the noise bucket (`-1`). | `2` | Raise to suppress tiny splinter clusters. | Set to `1` to keep every singleton community. |

**HDBSCAN — density-based, an alternative to Leiden**
(`scenario.*`, active when `cluster_algorithm: hdbscan`). Non-parametric
in cluster count; routes low-density docs to a noise bucket. All bundled
presets use Leiden, but HDBSCAN pairs well with dense (vision) encoders.

| Parameter | What it controls | Default | Raise it when… | Lower it when… |
|---|---|---|---|---|
| `cluster_algorithm` | Clustering algorithm: `leiden` (default here), `hdbscan`, `kmeans` (needs `k_clusters`), `dbscan`, `optics`, … | `leiden` | Switch to `hdbscan` for dense encoders / when you want automatic noise rejection. | — |
| `hdbscan_min_cluster_size` | Smallest admissible cluster; the main HDBSCAN dial. | `2` | Fewer, larger clusters and more aggressive noise flagging. | More, smaller clusters (min is 2). |

**Confidence — how sure the run is about each document** (`scenario.*`).
A fresh clustering run reports a per-document confidence: the softmax peak
over that document's distances to the discovered cluster centroids (docs the
algorithm routed to the noise bucket get `0.0`). It's an **ordinal** signal —
useful for ranking the least-certain documents for a spot check, *not* a
calibrated probability, and not comparable across runs or configs.

| Parameter | What it controls | Default | Raise it when… | Lower it when… |
|---|---|---|---|---|
| `confidence_temperature` | Softmax temperature for that confidence. `auto` scales it to the distance between clusters, so the scores spread out instead of all pinning at `1.00` when the clusters are far apart (which is the norm after a UMAP reduction). | `auto` | Pin a float `> 1` for uniformly more conservative scores. | Pin `1.0` to read the raw softmax peak with no rescaling. |

**Incremental novelty gate** (`scenario.*`, `--mode incremental`). These
decide whether a *new* document fits an existing DocSet or is "novel" and
opens a new cluster. The `dgml cluster --mode incremental` path ships a
**conservative default**: `threshold_quantile: 0.9`, which keeps the closest
90 % of each incoming batch as "known" and lets the farthest 10 % open new
categories. The quantile gate is used because it auto-calibrates to the
corpus's own distance scale (no manifold-unit tuning). The three gates
compose — a doc is novel if **any** active gate flags it. Set any gate
explicitly to override the default; set `threshold_quantile: null` to turn
gating off and force every doc into its nearest DocSet.

At the framework level (the `clustering` package, outside the CLI) all three
gates still default to `None`; the conservative default is applied only by the
incremental CLI path.

| Parameter | What it controls | Default (incremental CLI) | Raise it when… | Lower it when… |
|---|---|---|---|---|
| `threshold_quantile` | Auto-calibrates a distance cutoff to keep the closest `q` fraction as "known"; the rest become novel. Manifold-independent — adapts to the batch's distance scale. | `0.9` | Too many new clusters opening — raise toward 0.95 to keep more docs as known. | Genuinely new categories are being absorbed — lower toward 0.8 to flag more as novel. |
| `threshold_confidence` | Softmax-confidence floor in `[0,1]`; docs whose nearest-prototype confidence is below it become novel (new cluster). Manifold-independent — the easiest to reason about. | `None` | Genuinely new categories are being absorbed into existing DocSets — raise it (e.g. 0.4–0.5) to reject more as novel. | New clusters are opening for docs that really belong to an existing DocSet — lower it. |
| `threshold` | Absolute manifold-distance cutoff (unit depends on `manifold`; needs re-tuning if you change it). | `None` | You want a hard distance gate and know the scale. | — |

### Calibrated confidence and a review queue

The novelty gates above answer "*is this a new category?*" A separate question
is "*am I confident enough in this assignment to apply it unattended?*" That
one is what `scenario.calibration` answers, and the two are independent: a
novelty gate rewrites the assignment (the doc opens a new cluster), while the
review decision **never** changes where a document landed. It only adds a flag.

Flagged files come back on each assignment as `"review": true`, and collected
into a top-level `review_queue` list, so you can confirm just those:

```bash
dgml cluster | jq -r '.review_queue[]'
```

Everything here is **off by default** — an unconfigured run flags nothing and
`review_queue` is `[]`.

| Parameter | What it controls | Default | Notes |
|---|---|---|---|
| `calibration.abstain_threshold` | Confidence floor in `[0,1]`. Any assignment below it is flagged for review. | `None` | The simplest knob, and the only one that works in every scenario. Start around 0.5 and adjust to the queue size you can actually work through. |
| `calibration.method` | `temperature` (one-parameter rescaling, fit by maximum likelihood) or `platt` (logistic map fit against whether the top-1 was right). `none` keeps the raw ordinal score. | `none` | Needs labeled examples, so it only applies when the run has a support set (an incremental run over existing DocSets, or S3/S5). Name-only runs ignore it. |
| `calibration.coverage` | Target coverage in `(0,1)` for a split-conformal gate: flag the tail so that roughly this fraction of assignments are kept unflagged. | `None` | Prefer this over a hand-picked floor when you want to size the queue rather than guess a number — it adapts to the corpus instead of assuming a scale. Read it as a **review budget**, not a probability that the assignment is right: at `coverage: 0.9` about a tenth of the batch is flagged. It runs slightly under budget in practice (measured ≈0.92 achieved against 0.90 requested, over four corpora at 8 support documents per class), because the leave-one-out calibration set is a little easier than a fresh batch. |

A word on what "calibrated" buys you: with `method: none` the confidence is
ordinal, so a 0.83 means "more certain than the 0.6 next to it in this run" and
nothing more. Once a method is fit, the number is on a stable scale and *is*
comparable across runs — which is what makes a fixed `abstain_threshold`
meaningful over time rather than something you re-tune every corpus.

The fit is **leave-one-out**: each labeled example is scored against a prototype
rebuilt without it. Fitting on the ordinary prototypes would be self-flattering
(every document helps build the prototype it's measured against) and the
resulting thresholds would be optimistic on documents the run has not seen.

### Asking an LLM about the hard cases

A review queue tells you which assignments are shaky. `scenario.consolidation`
is the option to do something about them automatically: it takes the
least-confident tail, offers each document its nearest few clusters, and asks
the vision model the one question it is good at — *does this belong to one of
these, or is it genuinely something new?*

The cost model is the point. The embedding pipeline handles the whole corpus
cheaply, and the LLM only ever sees the documents the statistics were unsure
about — so spend scales with uncertainty, not with corpus size. On a clean run
it can be zero documents.

It is **off by default**, and needs a `classification` section in your workspace
`config.toml` (the same model and API key auto-classification uses). Enable it
in the `scenario` section of your clustering config:

```json
{
  "scenario": {
    "consolidation": {
      "enabled": true,
      "selector": { "strategy": "quantile", "quantile": 0.1 },
      "apply": "suggest"
    }
  }
}
```

| Parameter | What it controls | Default |
|---|---|---|
| `consolidation.enabled` | Master switch. | `false` |
| `consolidation.apply` | `suggest` records each verdict and flags the document for review, leaving labels alone. `auto` writes the reassignments into the result. | `suggest` |
| `consolidation.selector.strategy` | How the tail is chosen: `quantile` (bottom fraction by confidence), `confidence` (absolute floor), `margin` (narrow top-1/top-2 gap), `noise` (only the noise bucket). | `quantile` |
| `consolidation.selector.max_docs` | Hard ceiling on documents adjudicated — your cost cap. | `200` |
| `consolidation.candidates_k` | How many nearby clusters each document is offered. | `3` |
| `consolidation.mode` | `reassign` asks per document; `repartition` re-groups the whole contested subset in one call. | `reassign` |
| `consolidation.model` | Override the adjudication model. `null` reuses the workspace classification model. | `null` |

`suggest` is the default deliberately: an LLM overruling the embedding
partition is a change worth seeing before it lands. Either way every verdict —
old label, new label, confidence, and the model's one-line rationale — is
recorded in the run metadata, so `auto` is auditable rather than opaque.

Two behaviours worth knowing about, because both look like "nothing happened":

- **A flat confidence column suppresses the tail.** If every document scored
  about the same (an uncalibrated score can saturate near 1.0 for all of them),
  a bottom-quantile cut would select an essentially arbitrary set and let the
  model perturb assignments that were fine. The pass skips instead and says so
  in its metadata. Fixing the *confidence* signal — see the section above — is
  what makes selection meaningful; an absolute `confidence_threshold` also
  works, since it ranks nothing.
- **Failures are soft.** A missing API key, a provider outage, or a malformed
  reply degrades to the plain embedding result with the reason in metadata. A
  consolidation problem never fails your clustering run.

To find out whether it actually helped, capture a run with the pass off and one
with it on, then diff them:

```bash
dgml cluster --workspace ./ws --json > before.json
# ... enable clustering.scenario.consolidation in the workspace config ...
dgml cluster --workspace ./ws --json > after.json
dgml file list --workspace ./ws --json > files.json

uv run python scripts/clustering_metrics.py \
    --before before.json --after after.json --files files.json
```

That prints both runs side by side with deltas — cluster shape, confidence, and
(when ground truth is available) ARI, NMI, purity, and mapped accuracy — plus
the list of documents that changed DocSet, which is the thing to actually read
before switching `apply` to `auto`. Ground truth comes from a `--labels` map or
from a one-folder-per-class corpus layout; with neither, the external scores are
reported as unavailable rather than as zeros.

### Symptom → knob

- **One true category split across several clusters** (high homogeneity,
  low completeness) → lower `leiden_resolution`; or raise
  `leiden_k_neighbors`; or raise `reduce_dim`.
- **Distinct categories merged into one cluster** → raise
  `leiden_resolution`; try `leiden_graph_method: mutual_knn`; or move to a
  dense encoder (`bge`/`e5`).
- **Lots of tiny/noise clusters** → raise `leiden_min_cluster_size` (or
  `hdbscan_min_cluster_size`); lower `reduce_dim`.
- **Incremental run assigns every new doc to old DocSets, never opens new
  ones** → the default `threshold_quantile: 0.9` should already let outliers
  through; lower it toward 0.8, or add `scenario.threshold_confidence`
  (start ~0.4). Check you didn't set `threshold_quantile: null`.
- **Incremental run opens too many new DocSets** → raise
  `scenario.threshold_quantile` toward 0.95, or disable gating with
  `threshold_quantile: null`.
- **Assignments are mostly right but you can't tell which ones to trust** →
  set `scenario.calibration.abstain_threshold` and work the `review_queue`;
  add `calibration.method: temperature` if you want the threshold to keep
  meaning the same thing on the next corpus.
- **A handful of documents are wrong and no config change fixes them without
  breaking the rest** → enable `scenario.consolidation` and let the LLM
  adjudicate just that tail. Set `selector.max_docs` to whatever you're
  willing to spend.
- **Consolidation is enabled but reports `n_selected: 0`** → read the `notes`
  in its metadata. Usually the confidence column is flat, so the tail was
  suppressed on purpose; give the score some resolution (`calibration.method`)
  or select on an absolute `selector.confidence_threshold` instead.

## 6. Inspect what came out

```bash
dgml docset list                       # all DocSets, with file counts
dgml docset list-files <docset_id>     # which files are in one
dgml status                            # workspace-wide summary
```

Spot-check by file:

```bash
dgml file list
dgml file show <file_id>
```

## 7. Re-run safely

The pipeline is designed to be idempotent. To bring a folder up to
date after adding more PDFs, run the same two commands again:

```bash
dgml file add /path/to/pdfs --recursive --on-conflict skip
dgml cluster
```

`--on-conflict skip` returns the existing record for any PDF already
ingested; `dgml cluster` only touches files not yet in a DocSet, so
clustering picks up exactly the new arrivals.

## Where to go next

- [`docs/cli-reference.md`](cli-reference.md) — full command reference,
  including `--auto-classify`, schema generation, and the
  `dgml docset generate` PDF → DGML pass.
- [`docs/storage-layout.md`](storage-layout.md) — on-disk format of the
  workspace.
- [`packages/clustering/README.md`](../packages/clustering/README.md) —
  the clustering framework itself: scenarios, encoders, fusion,
  manifolds, and the Python API for driving it directly.
