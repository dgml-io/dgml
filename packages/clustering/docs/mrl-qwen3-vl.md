# MRL clustering with Qwen3-VL-Embedding

This note documents the option to cluster DGML files with
**Qwen3-VL-Embedding** (`Qwen/Qwen3-VL-Embedding-2B` / `-8B`) as a
Matryoshka (MRL) encoder — usable as the text side, the image side, or both,
and runnable either in-process or against a hosted server.

## Why this model

Qwen3-VL-Embedding is a purpose-built **multimodal** embedding model (base:
Qwen3-VL-\*-Instruct, Apache-2.0 — compatible with this project's license
policy). It maps text, page images, and mixed inputs into **one shared
vector space**, and it is **Matryoshka-trained**: the native embedding (2048-d
for 2B, 4096-d for 8B) can be truncated to any width in `[64, native]` and
re-normalized with only a small quality cost. That last property is the whole
point for clustering — see below.

| Checkpoint                     | Params | Native dim | MRL range   |
| ------------------------------ | ------ | ---------- | ----------- |
| `Qwen/Qwen3-VL-Embedding-2B`   | 2B     | 2048       | `[64, 2048]`|
| `Qwen/Qwen3-VL-Embedding-8B`   | 8B     | 4096       | `[64, 4096]`|

## Why MRL helps clustering

Clustering cost (k-means, HDBSCAN, Leiden on a k-NN graph) scales with
dimensionality, and in very high dimensions pairwise distances *concentrate*,
so density-based methods drift toward all-noise. Because MRL front-loads the
important information into the leading coordinates, truncating to a small
prefix (say 128–256) and re-normalizing keeps most of the cluster structure
while making distances cheaper to compute and better-behaved. You **embed once
at full width** and then **cluster at the width the data actually needs**.

The one non-negotiable step: **re-normalize after truncating**. Slicing off
dimensions changes a vector's norm, so cosine / dot-product similarity is only
meaningful again after an L2 renormalize. `mrl_truncate` does this by default.

## What was added

Everything lives in the existing `dgml-clustering` package; no new runtime
dependencies (the server backend uses only the standard library).

### 1. `clustering.encoders.mrl`

- `mrl_truncate(embeddings, dim, *, normalize=True)` — the canonical
  "take the first `dim` dims, then L2-renormalize" op on a `[B, D]` torch
  tensor. Rejects widening (MRL can only shrink).
- `mrl_dimension_sweep(embeddings, dims, cluster_fn, *, score_fn=None,
  normalize=True)` — cluster a `[N, D]` numpy matrix at several prefix widths
  and return a `SweepResult` (`dims`, `scores`, `best_dim`, `best_score`,
  `best_labels`). Picks the **smallest** width achieving the top score; the
  default score is a cosine silhouette over non-noise points. `cluster_fn`
  wraps whatever algorithm you use, so the sweep stays algorithm-agnostic.

### 2. `Qwen3VLEmbeddingEncoder` (generalized)

- **Modality.** Now accepts `str`, `PIL.Image`, or a `{"text", "image"}` dict,
  and registers under `qwen3_vl_embedding` / `qwen3_vl_embedding_2b` for use as
  **either** `encoder_text` **or** `encoder_image`. Pairing the same family on
  both sides fuses two views from one shared space.
- **Backend** (`extra['backend']`):
  - `"local"` (default) — in-process `sentence_transformers`; MRL via the
    model's `truncate_dim` + `normalize_embeddings=True`.
  - `"server"` — POST to an OpenAI-compatible `/v1/embeddings` endpoint (vLLM
    `--task embed` or SGLang). No torch model in this process. MRL truncation
    is applied **client-side** via `mrl_truncate`, so the configured
    `embedding_dim` is honored whether or not the server truncated.

The single-vector guard is unchanged: `multi_vector=True` is rejected.

### 3. Configs

```
configs/encoder_image/qwen3_vl_embedding.yaml          # local, 8B  (existing)
configs/encoder_image/qwen3_vl_embedding_2b.yaml       # local, 2B  (existing)
configs/encoder_image/qwen3_vl_embedding_server.yaml   # server, 8B (new)
configs/encoder_text/qwen3_vl_embedding.yaml           # local, 8B  (new)
configs/encoder_text/qwen3_vl_embedding_2b.yaml        # local, 2B  (new)
configs/encoder_text/qwen3_vl_embedding_server.yaml    # server, 8B (new)
```

## How to use it

### Local, multimodal (text + page images), one shared space

```
encoder_text=qwen3_vl_embedding_2b \
encoder_image=qwen3_vl_embedding_2b \
fusion=concat_norm
```

Keep `embedding_dim` equal on both sides so the two shared-space views line up.

### Server backend

Start a server (example, vLLM):

```bash
vllm serve Qwen/Qwen3-VL-Embedding-8B --task embed --trust-remote-code
```

Then point the config at it:

```
encoder_image=qwen3_vl_embedding_server   # extra.base_url=http://localhost:8000/v1
```

> The multimodal `input` schema for hosted embeddings still varies between
> server versions; the client sends text as a bare string and images as an
> OpenAI chat-style `image_url` content list, and always re-truncates
> client-side. Adjust `_server_input_element` if your server expects a
> different shape.

### Choosing a width with a sweep

```python
import numpy as np
from sklearn.cluster import KMeans
from clustering.encoders.mrl import mrl_dimension_sweep

full = np.load("embeddings.npy")            # [N, native], embedded once
sweep = mrl_dimension_sweep(
    full, [64, 128, 256, 512],
    cluster_fn=lambda X: KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(X),
)
print(sweep.best_dim, sweep.best_score)     # smallest width that clusters well
```

See `examples/mrl_qwen3_vl_clustering.py` for an end-to-end run.

## Testing

`tests/test_mrl.py` and `tests/test_qwen3_vl_backends.py` run offline (no
weights, no network): the truncation/renorm invariants, the dimension sweep on
synthetic clusters, the pure request/response builders, and the server
backend's construction + encode path (with a monkeypatched embed).
