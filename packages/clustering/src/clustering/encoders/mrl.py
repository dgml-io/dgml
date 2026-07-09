# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Matryoshka Representation Learning (MRL) helpers.

Qwen3-VL-Embedding — and the Matryoshka-trained text frontier (Stella,
Jina v3, …) — pack their most important information into the *leading*
coordinates of the embedding. Truncating to the first ``d`` dimensions and
re-normalizing therefore yields a smaller vector that preserves most of the
semantic structure, unlike a naive truncation of a non-MRL model. Smaller
vectors are cheaper to store, faster to cluster, and often *better behaved*
for density-based clustering (HDBSCAN and friends), whose distances
concentrate — and whose clusters collapse to all-noise — in very high
dimensions.

Two entry points:

* :func:`mrl_truncate` — the canonical "take the first ``d`` dims and
  L2-renormalize" operation on a batch of embeddings. The Qwen3-VL *server*
  backend routes through this to honor the configured width (the *local*
  SentenceTransformer path gets the same effect from ``truncate_dim``);
  it's also what you call to slice a cached full-width embedding after the
  fact.
* :func:`mrl_dimension_sweep` — choose a clustering dimensionality
  empirically: cluster at several candidate prefix widths, score each, and
  keep the smallest ``d`` that doesn't hurt cluster quality. This is the MRL
  payoff for clustering — embed once at full width, then cluster cheaply at
  the width the data actually needs.

``mrl_truncate`` works on ``torch`` tensors (the encoder contract);
``mrl_dimension_sweep`` works on ``numpy`` arrays (the clustering / sklearn
contract).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.functional import normalize as l2_normalize

__all__ = ["SweepResult", "mrl_dimension_sweep", "mrl_truncate"]


def mrl_truncate(
    embeddings: torch.Tensor, dim: int | None, *, normalize: bool = True
) -> torch.Tensor:
    """Truncate a batch of MRL embeddings to the first ``dim`` dims, then renormalize.

    Args:
        embeddings: ``[B, D]`` batch. Must be 2-D.
        dim: Target width. ``None`` (or ``dim == D``) keeps the full width.
            Must be in ``[1, D]`` otherwise — MRL can only *shrink* a vector,
            never invent dimensions.
        normalize: L2-normalize each row after truncation (the default, and
            what you want for cosine / dot-product similarity). Truncation
            changes a vector's norm, so re-normalizing is not optional for
            most downstream metrics — hence the default.

    Returns:
        A contiguous ``[B, dim]`` (or ``[B, D]``) float tensor.

    Raises:
        ValueError: If ``embeddings`` is not 2-D, or ``dim`` is out of range.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D [B, D] tensor, got shape {tuple(embeddings.shape)}.")
    native = int(embeddings.shape[-1])
    if dim is not None and dim != native:
        if dim <= 0:
            raise ValueError(f"MRL dim must be positive, got {dim}.")
        if dim > native:
            raise ValueError(
                f"MRL dim {dim} exceeds the native embedding width {native}; "
                "MRL can only shrink an embedding, not widen it."
            )
        embeddings = embeddings[:, :dim]
    if normalize:
        embeddings = l2_normalize(embeddings, p=2.0, dim=-1)
    return embeddings.contiguous()


@dataclass(frozen=True)
class SweepResult:
    """Outcome of an MRL dimension sweep.

    Attributes:
        dims: The candidate widths evaluated, ascending.
        scores: Cluster-quality score per ``dims`` entry (higher = better).
            ``-inf`` marks a width whose clustering degenerated (fewer than
            two non-noise clusters), so it can never win the sweep.
        best_dim: The smallest width achieving the maximum score.
        best_score: The score at ``best_dim``.
        best_labels: The cluster labels produced at ``best_dim``.
    """

    dims: tuple[int, ...]
    scores: tuple[float, ...]
    best_dim: int
    best_score: float
    best_labels: np.ndarray


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    normalized: np.ndarray = x / np.clip(norms, a_min=1e-12, a_max=None)
    return normalized


def _default_score(features: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette over non-noise points; ``-inf`` when it isn't defined.

    Noise points (label ``-1``, produced by HDBSCAN / DBSCAN / graph_cc) are
    excluded. Silhouette needs at least two clusters and at least one more
    sample than clusters, so degenerate assignments score ``-inf`` and are
    never picked as the best width.
    """
    from sklearn.metrics import silhouette_score

    mask = labels != -1
    kept = labels[mask]
    n_clusters = int(np.unique(kept).size)
    if n_clusters < 2 or int(mask.sum()) <= n_clusters:
        return float("-inf")
    return float(silhouette_score(features[mask], kept, metric="cosine"))


def mrl_dimension_sweep(
    embeddings: np.ndarray,
    dims: Sequence[int],
    cluster_fn: Callable[[np.ndarray], np.ndarray],
    *,
    score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    normalize: bool = True,
) -> SweepResult:
    """Cluster at several MRL widths and return the cheapest good one.

    For each candidate ``d`` the leading ``d`` dims are sliced off (and
    L2-renormalized when ``normalize``), clustered with ``cluster_fn``, and
    scored with ``score_fn``. The smallest ``d`` attaining the top score wins
    — on a tie you get the cheaper representation.

    Args:
        embeddings: ``[N, D]`` full-width embeddings (one row per document).
        dims: Candidate widths to try. Deduplicated and sorted ascending;
            values above ``D`` are dropped (with ``D`` itself kept so the
            full-width baseline is always in the sweep).
        cluster_fn: Maps an ``[N, d]`` feature matrix to an ``[N]`` integer
            label array (``-1`` = noise). Wrap your KMeans / HDBSCAN / Leiden
            call here so the sweep stays agnostic to the algorithm.
        score_fn: ``(features, labels) -> float`` (higher = better). Defaults
            to a cosine silhouette over non-noise points.
        normalize: L2-renormalize each truncated prefix before clustering.

    Returns:
        A :class:`SweepResult`.

    Raises:
        ValueError: If ``embeddings`` is not 2-D or ``dims`` is empty (after
            filtering to ``<= D``).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D [N, D] array, got shape {embeddings.shape}.")
    native = int(embeddings.shape[-1])
    score = score_fn if score_fn is not None else _default_score

    candidates = sorted({int(d) for d in dims if 0 < int(d) <= native} | {native})
    if not candidates:
        raise ValueError(f"No valid dims to sweep for native width {native}: {list(dims)}.")

    tried: list[int] = []
    scores: list[float] = []
    labels_by_dim: dict[int, np.ndarray] = {}
    for d in candidates:
        feats = embeddings[:, :d]
        if normalize:
            feats = _l2_normalize(feats)
        labels = np.asarray(cluster_fn(feats))
        labels_by_dim[d] = labels
        tried.append(d)
        scores.append(score(feats, labels))

    # Ascending order + strict ">" means the first (smallest) dim wins ties.
    best_idx = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best_idx]:
            best_idx = i
    best_dim = tried[best_idx]
    return SweepResult(
        dims=tuple(tried),
        scores=tuple(scores),
        best_dim=best_dim,
        best_score=scores[best_idx],
        best_labels=labels_by_dim[best_dim],
    )
