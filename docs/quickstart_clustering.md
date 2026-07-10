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
and its ML stack — embedding models, `leidenalg`, `scipy`, `sklearn`):

```bash
pip install "dgml[clustering]"
```

Sanity-check the CLI:

```bash
dgml --help
```

## 2. Create a workspace

The workspace is a directory holding `docsets/` and `files/`. Anything
the CLI writes goes there.

```bash
export DGML_HOME=./dgml-workspace
dgml init              # seed the shared local_config.json
dgml workspace create  # create the workspace from it
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
are also configured in `<workspace>/config.json` (see
[`docs/cli-reference.md`](cli-reference.md#ocr-configuration) for the
schema), while macOS Apple Vision runs on-device with no config:

```bash
pip install "dgml[macos]"     # Apple Vision — on-device, macOS only, zero-config
# or, for cloud OCR (add an `ocr` section to config.json first):
pip install "dgml[azure]"     # Azure Document Intelligence
pip install "dgml[aws]"       # AWS Textract

dgml file add /path/to/pdfs --recursive --on-conflict skip --text-mode hybrid
```

On macOS, Apple Vision is the default OCR engine even with no `ocr`
section in `config.json` — just install the extra. `hybrid` runs
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

Step 4 needs the `classification` section in `<workspace>/config.json`
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

## 5. Tune the clustering (optional)

The defaults cluster a folder sensibly out of the box; everything here is
optional. There are two ways to override them, both using the same field
schema:

- **Per workspace** — add a `clustering` section to
  `<workspace>/config.json`. It's a *partial overlay*, deep-merged over
  the bundled defaults, so you only spell out what you change.
- **Per run** — `dgml cluster --config PATH` points at a standalone JSON
  with the same fields (drop the `clustering` wrapper); it *replaces* the
  section for that run. `--config` also accepts a bundled preset **name**
  (`light` / `medium` / `heavy`).

```jsonc
// <workspace>/config.json — change only what you need
{
  "clustering": {
    "encoder_text": {"name": "bge", "model_id": "BAAI/bge-small-en-v1.5", "embedding_dim": 384},
    "manifold": {"name": "euclidean", "dim": 384},
    "scenario": {"leiden_resolution": 0.7, "leiden_k_neighbors": 20}
  }
}
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
budget. Higher tiers use a stronger (denser) text encoder — better
separation, more compute.

| Preset | Target hardware | Text encoder | Clustering |
|---|---|---|---|
| `light` (default) | CPU-only | `tfidf`, 256-d | Leiden + UMAP |
| `medium` | large CPU / Apple MPS | `bge-small`, 384-d | Leiden + UMAP |
| `heavy` | GPU | `e5-large`, 1024-d | HDBSCAN + UMAP |

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
| `encoder_text.name` | Text embedding model. `tfidf` (bag-of-words, fast, CPU) vs dense sentence encoders `bge` / `e5` / `gte` (semantic, need a model download). | `tfidf` | Categories differ by *meaning*, not vocabulary; short docs; TF-IDF under-separates. Move to `bge` (→ `medium`) or `e5` (→ `heavy`). | You want zero downloads / CPU-only speed and the vocabularies are already distinctive. |
| `encoder_text.embedding_dim` + `manifold.dim` | Vector width. Must match the encoder (`tfidf` 256, `bge` 384, `e5` 1024). Keep these two equal. | 256 | Switching to a wider encoder. | Switching to a narrower encoder. |
| `encoder_text.extra.text_view` | Which text is embedded: `page1` (first page only) or the full document. | `page1` | The first page doesn't characterize the doc (cover pages, boilerplate); use full text. | First pages are highly distinctive (forms, letterheads) — cheaper and less noisy. |

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

**HDBSCAN — density-based, the `heavy` preset's algorithm**
(`scenario.*`, active when `cluster_algorithm: hdbscan`). Non-parametric
in cluster count; routes low-density docs to a noise bucket.

| Parameter | What it controls | Default | Raise it when… | Lower it when… |
|---|---|---|---|---|
| `cluster_algorithm` | Clustering algorithm: `leiden` (default here), `hdbscan`, `kmeans` (needs `k_clusters`), `dbscan`, `optics`, … | `leiden` | Switch to `hdbscan` for dense encoders / when you want automatic noise rejection. | — |
| `hdbscan_min_cluster_size` | Smallest admissible cluster; the main HDBSCAN dial. | `2` | Fewer, larger clusters and more aggressive noise flagging. | More, smaller clusters (min is 2). |

**Incremental novelty gate** (`scenario.*`, `--mode incremental`). These
decide whether a *new* document fits an existing DocSet or is "novel" and
opens a new cluster. All default to `None` — meaning **every** incoming
doc is forced into its nearest existing DocSet (nothing is ever treated as
novel). Set one to let new categories emerge.

| Parameter | What it controls | Default | Raise it when… | Lower it when… |
|---|---|---|---|---|
| `threshold_confidence` | Softmax-confidence floor in `[0,1]`; docs whose nearest-prototype confidence is below it become novel (new cluster). Manifold-independent — the easiest to reason about. | `None` | Genuinely new categories are being absorbed into existing DocSets — raise it (e.g. 0.4–0.5) to reject more as novel. | New clusters are opening for docs that really belong to an existing DocSet — lower it. |
| `threshold_quantile` | Auto-calibrates a distance cutoff to keep the closest `q` fraction as "known". | `None` | Prefer auto-tuning over hand-picking a confidence (e.g. `0.8`). | — |
| `threshold` | Absolute manifold-distance cutoff (unit depends on `manifold`; needs re-tuning if you change it). | `None` | You want a hard distance gate and know the scale. | — |

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
  ones** → set `scenario.threshold_confidence` (start ~0.4).

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
