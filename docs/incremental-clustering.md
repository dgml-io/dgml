# Incremental clustering (the "S3" workflow)

Most workspaces are not clustered once and frozen — documents arrive in
batches. **Incremental clustering** grows an existing clustering: given a
workspace that already has DocSets (from a prior `dgml cluster` run or manual
curation) and a fresh batch of unassigned files, it assigns each new file to
the DocSet it belongs in, and opens new DocSets for the files that don't fit
any existing cluster.

This is a **run mode** of `dgml cluster`, not a separate command or a new
clustering scenario. It reuses the existing scenario machinery
(`clustering.scenarios`): existing DocSets become nearest-prototype categories
(few-shot S3), and leftover documents are clustered into emergent buckets
exactly as a fresh S1 run would be.

## Three cases

Incremental clustering handles all three ways a new batch can land:

1. **All fit** — every new file is assigned to an existing DocSet.
   `n_new_clusters == 0`.
2. **Some fit, some don't** — the fits are assigned to existing DocSets; the
   rest form one or more new `unknown_N` clusters, each LLM-named into a new
   DocSet. `n_assigned_existing > 0` and `n_new_clusters > 0`.
3. **None fit** — every new file forms a new cluster. `n_assigned_existing == 0`
   and `n_new_clusters > 0`. (Behaviourally similar to a fresh run, but the
   existing DocSets still act as prototypes to *reject* against.)

Whether a document "fits" is decided by the nearest-prototype gates on the
scenario (`threshold`, `threshold_confidence`, `threshold_quantile`) — a
document too far from every existing prototype is routed to the emergent
bucket. The incremental CLI ships a conservative default,
`threshold_quantile: 0.9` (the closest 90 % of a batch count as fitting, the
farthest 10 % become novel), so "some fit" and "none fit" happen out of the
box; without it every document would always be forced into its nearest DocSet.
Override it — or disable gating with `threshold_quantile: null` — in the
`scenario` section of your clustering config.

## How prototypes are built

In incremental mode each existing DocSet becomes a category whose prototype is
the on-manifold mean of **all** of its already-assigned members that have a
rendered page image (bounded by `MAX_SUPPORT_SAMPLES_PER_DOCSET`). Embeddings
are content-hashed and cached by the encoder layer, so re-embedding members
seen on a previous run is cheap. A DocSet with no usable members contributes a
name-only (S2) prototype instead.

## CLI

```bash
# Auto — incremental when the workspace already has DocSets, else fresh.
dgml cluster --workspace ./ws

# Force incremental (errors with INCREMENTAL_WITHOUT_CLUSTERS if no DocSets).
dgml cluster --workspace ./ws --mode incremental

# Force a fresh, from-scratch clustering, ignoring existing DocSets.
dgml cluster --workspace ./ws --mode fresh

# Pick a compute tier by preset name (default is light).
dgml cluster --workspace ./ws --config medium

# Small incoming batch? Partition it with the vision LLM instead of embeddings.
# --method is orthogonal to --mode: existing DocSets are still offered to the
# model as categories to assign into, so incremental growth works the same way.
dgml cluster --workspace ./ws --method auto
```

See [`dgml cluster`](cli-reference.md) for the full flag reference and the
JSON output shape (including the additive `mode`, `n_assigned_existing`,
`n_new_clusters`, and `assignments` fields).

## Config presets (compute tiers)

Four bundled presets tune the encoder/manifold/algorithm stack for a compute
budget. Higher tiers add **image/vision embeddings** rather than a denser text
encoder. Each is a complete, self-contained clustering config, resolvable by
name via `--config`:

| Preset | Target | Representation | Clustering |
|---|---|---|---|
| `small` | CPU-only, tiny corpora | `tfidf` text (256-d) | Leiden, no UMAP |
| `light` | CPU-only (default) | `tfidf` text (256-d) | Leiden + UMAP |
| `medium` | large CPU / Apple MPS | `tfidf` text + 2B vision, fused (1280-d) | Leiden + UMAP |
| `heavy` | GPU | 8B vision only (1024-d) | Leiden + UMAP |

They live alongside the library at
`packages/dgml-core/src/dgml_core/clustering_preset_{small,light,medium,heavy}.json`.
Copy one and pass it as `--config <path>` as a starting point for a custom
config. The presets are also the sweep targets in the evaluation harness — the
sweep writes the best config it finds for each tier back into these files.

## Evaluation harness

The two-phase evaluation lives under [`evaluation/clustering/`](../evaluation/clustering).
It splits a labeled workspace into a **phase-1 corpus** (clustered first) and
one or more **incoming batches** (fed to incremental clustering), then scores
the incremental assignments. See
[`evaluation/clustering/INCREMENTAL.md`](../evaluation/clustering/INCREMENTAL.md)
for the operator's guide: prerequisites (the classification vision-LLM config
this needs), the single-run and sweep entry points, the splitter strategies,
the metric set (the standard clustering metrics — purity, NMI, ARI,
homogeneity/completeness/V-measure, mapped accuracy — plus incremental-specific
metrics: correct-assignment rate to existing clusters, outlier detection
precision/recall, and new-cluster quality), the HTML report, the analysis of
how phase-1 clustering quality influences incremental results, the
light/medium/heavy config sweeps, and troubleshooting.
