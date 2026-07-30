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

"""Manifold-aware clustering helpers used by S1-style scenarios.

Four clustering algorithms are exposed, all fully manifold-aware:

- :func:`manifold_kmeans` — Riemannian Lloyd's algorithm. Requires a
  user-supplied ``k``. For Euclidean geometry the result is identical to
  standard k-means; for Spherical / Hyperbolic / Product we use the
  manifold's pairwise distance for assignment and the manifold's
  :meth:`expmap0` / Fréchet-mean approximation for the centroid update.
- :func:`manifold_hdbscan` — density-based, *parameter-free* with respect
  to ``k``. Builds the full manifold pairwise-distance matrix and runs
  scikit-learn's HDBSCAN with ``metric='precomputed'``, so the density
  estimate respects manifold geometry rather than ambient Euclidean
  distance. Noise points get label ``-1``; cluster representatives are
  on-manifold medoids (actual data points, so trivially valid on any
  manifold).
- :func:`manifold_graph_cc` — similarity-graph → radius-graph →
  connected-components. Equivalent to single-linkage agglomerative
  clustering cut at distance ``r``, but expressed in graph terms so it
  composes cleanly with sparse-graph backends if we ever need to scale.
  Two auto-radius heuristics are built in: the canonical DBSCAN k-NN
  knee (Ester et al. 1996, with Kneedle-style knee detection) and an
  MST max-gap cut. CCs smaller than ``min_cluster_size`` get label
  ``-1`` to match HDBSCAN's noise convention.
- :func:`manifold_leiden` — Louvain community detection (Blondel et al.
  2008) via networkx (BSD). Builds a weighted graph from the manifold
  distances — k-NN by default, with mutual-k-NN and radius-graph variants
  — then partitions it by modularity. Cluster count emerges from the
  partition. Communities smaller than ``min_cluster_size`` are folded into
  the noise bucket, matching the HDBSCAN convention. (Named 'leiden' for the
  algorithm family, but implemented with networkx's Louvain — the GPL
  leidenalg/igraph backend can't ship in an Apache-2.0-licensed package.)

The high-level :func:`cluster_embeddings` dispatcher picks between the
four based on the ``algorithm`` string (the same enum used in
``ScenarioConfig.cluster_algorithm``), keeping S1 / S2 callers algorithm-
agnostic.

This module also owns :func:`assign_to_prototypes`, the nearest-prototype
classifier used by S2 / S3 / S4 / S5 — including the three composable
gating modes (absolute distance, softmax confidence, and quantile
auto-calibration) for routing out-of-distribution documents to the
"unknown" bucket.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import torch

from clustering.calibration import Calibrator
from clustering.config.schema import ScenarioConfig
from clustering.manifolds.base import ManifoldHead

_log = logging.getLogger(__name__)

ClusterAlgorithm = Literal[
    "kmeans",
    "hdbscan",
    "graph_cc",
    "leiden",
    "dbscan",
    "optics",
    "affinity_propagation",
    "meanshift",
]
GraphCCRadiusMethod = Literal["knee", "mst_gap"]
LeidenGraphMethod = Literal["knn", "mutual_knn", "radius"]
LeidenQuality = Literal["modularity", "cpm"]
ReduceMethod = Literal[
    "none",
    "pca",
    "truncated_svd",
    "random_projection",
    "kernel_pca",
    "isomap",
    "lle",
    "spectral",
    "umap",
]


def reduce_embeddings(
    embeddings: torch.Tensor,
    *,
    method: ReduceMethod = "none",
    n_components: int = 0,
    seed: int = 0,
) -> torch.Tensor:
    """Reduce ``embeddings`` to ``n_components`` dims before clustering.

    Density-based clustering (HDBSCAN especially) degrades badly on raw
    transformer embeddings: in hundreds of dimensions pairwise distances
    concentrate into a narrow band, the density estimate flattens, and
    HDBSCAN routes nearly everything to the noise bucket. The standard
    remedy — used by BERTopic and most embedding-clustering pipelines — is
    to project to a low-dimensional space first.

    Methods (all unsupervised; everything but ``umap`` ships with
    scikit-learn, already a dependency):

    - ``"none"`` — passthrough.
    - ``"pca"`` — linear, fast, deterministic. The dependency-free baseline.
    - ``"truncated_svd"`` — PCA without mean-centering; near-identical for
      dense normalized embeddings, mainly a win on sparse input.
    - ``"random_projection"`` — Gaussian Johnson-Lindenstrauss projection.
      No fitting, approximately distance-preserving, extremely fast.
    - ``"kernel_pca"`` — nonlinear PCA with a cosine kernel (suited to
      L2-normalized embeddings); deterministic.
    - ``"isomap"`` / ``"lle"`` / ``"spectral"`` — classic nonlinear manifold
      learners (Isomap, Locally Linear Embedding, Laplacian Eigenmaps).
      Neighbour-graph based; slower but capture local manifold structure.
    - ``"umap"`` — the gold standard partner for HDBSCAN (cosine metric).
      Requires ``umap-learn``.

    Args:
        embeddings: ``[N, D]`` input vectors (typically on-manifold /
            L2-normalized).
        method: One of the reducers above.
        n_components: Target dimensionality. Clamped to ``[1, min(N-1, D)]``.
            ``<= 0`` or ``>= D`` is a passthrough.
        seed: Reproducibility seed (ignored by the few estimators that are
            already deterministic, e.g. Isomap).

    Returns:
        ``[N, n_components]`` reduced tensor, or ``embeddings`` unchanged
        when no reduction applies.
    """
    n = int(embeddings.shape[0])
    d = int(embeddings.shape[-1])
    if method == "none" or n_components <= 0 or n_components >= d or n < 3:
        return embeddings
    k = max(1, min(int(n_components), n - 1, d))

    x = (
        # .float() so bf16 embeddings (e.g. native-dtype VL embedders) survive
        # the numpy boundary — numpy has no bfloat16.
        embeddings.detach().cpu().float().numpy()
        if hasattr(embeddings, "detach")
        else np.asarray(embeddings)
    )
    reduced = _fit_reduce(x, method=method, k=k, n=n, seed=seed)
    return torch.as_tensor(np.asarray(reduced), dtype=torch.float32)


# Nonlinear reducers that build a k-nearest-neighbour graph. They clamp
# ``n_neighbors`` for small corpora, but can still degenerate or error on
# pathologically small / collinear inputs — where we fall back to PCA.
_NEIGHBOUR_BASED: frozenset[ReduceMethod] = frozenset({"isomap", "lle", "spectral", "umap"})


def _fit_reduce(
    x: np.ndarray[Any, Any], *, method: ReduceMethod, k: int, n: int, seed: int
) -> np.ndarray[Any, Any]:
    """Dispatch to the requested reducer and return the ``[N, k]`` array.

    Neighbour-based learners clamp ``n_neighbors`` into ``[2, N-1]`` (and,
    for LLE, above ``k``) so they don't blow up on small corpora. As a final
    guard, if such a reducer still raises on a degenerate corpus we fall back
    to PCA (well-defined for any ``n > k``) rather than aborting the whole
    clustering run. A missing optional dependency (``umap-learn``) is *not*
    swallowed — that stays a hard, actionable error.
    """
    try:
        return _dispatch_reduce(x, method=method, k=k, n=n, seed=seed)
    except ImportError:
        raise
    except Exception as exc:
        if method not in _NEIGHBOUR_BASED:
            raise
        warnings.warn(
            f"reduce method {method!r} failed on this corpus (n={n}, k={k}): "
            f"{exc}. Falling back to PCA.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _dispatch_reduce(x, method="pca", k=k, n=n, seed=seed)


def _dispatch_reduce(
    x: np.ndarray[Any, Any], *, method: ReduceMethod, k: int, n: int, seed: int
) -> np.ndarray[Any, Any]:
    """Fit the requested reducer and return the ``[N, k]`` array."""
    if method == "pca":
        from sklearn.decomposition import PCA

        return cast("np.ndarray[Any, Any]", PCA(n_components=k, random_state=seed).fit_transform(x))
    if method == "truncated_svd":
        from sklearn.decomposition import TruncatedSVD

        return cast(
            "np.ndarray[Any, Any]",
            TruncatedSVD(n_components=k, random_state=seed).fit_transform(x),
        )
    if method == "random_projection":
        from sklearn.random_projection import GaussianRandomProjection

        return cast(
            "np.ndarray[Any, Any]",
            GaussianRandomProjection(n_components=k, random_state=seed).fit_transform(x),
        )
    if method == "kernel_pca":
        from sklearn.decomposition import KernelPCA

        return cast(
            "np.ndarray[Any, Any]",
            KernelPCA(n_components=k, kernel="cosine", random_state=seed).fit_transform(x),
        )
    if method == "isomap":
        from sklearn.manifold import Isomap

        n_neighbors = max(2, min(10, n - 1))
        return cast(
            "np.ndarray[Any, Any]",
            Isomap(n_components=k, n_neighbors=n_neighbors).fit_transform(x),
        )
    if method == "lle":
        from sklearn.manifold import LocallyLinearEmbedding

        # Standard LLE needs n_neighbors > n_components.
        n_neighbors = min(max(k + 1, min(10, n - 1)), n - 1)
        return cast(
            "np.ndarray[Any, Any]",
            LocallyLinearEmbedding(
                n_components=k, n_neighbors=n_neighbors, random_state=seed
            ).fit_transform(x),
        )
    if method == "spectral":
        from sklearn.manifold import SpectralEmbedding

        n_neighbors = max(2, min(10, n - 1))
        return cast(
            "np.ndarray[Any, Any]",
            SpectralEmbedding(
                n_components=k, n_neighbors=n_neighbors, random_state=seed
            ).fit_transform(x),
        )
    if method == "umap":
        try:
            import umap
        except ImportError as exc:  # pragma: no cover — optional dep
            raise ImportError(
                "reduce method 'umap' requires umap-learn. Install it "
                "(`pip install umap-learn`) or use a scikit-learn reducer (e.g. 'pca')."
            ) from exc

        # Clamp n_neighbors like the sibling neighbour-based reducers, and pin
        # random init. UMAP's default spectral init routes through scipy's
        # ``eigsh``, which raises ``TypeError`` for k >= N on small corpora
        # (fewer points than requested components); ``init="random"`` sidesteps
        # that path entirely.
        n_neighbors = max(2, min(15, n - 1))
        return cast(
            "np.ndarray[Any, Any]",
            umap.UMAP(
                n_components=k,
                n_neighbors=n_neighbors,
                init="random",
                metric="cosine",
                random_state=seed,
            ).fit_transform(x),
        )
    raise ValueError(f"Unknown reduce method {method!r}.")  # pragma: no cover — Literal-guarded


def manifold_kmeans(
    embeddings: torch.Tensor,
    k: int,
    manifold: ManifoldHead,
    *,
    n_iter: int = 25,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Riemannian k-means on ``embeddings`` (already on-manifold).

    Args:
        embeddings: ``[N, D]`` on-manifold points.
        k: Number of clusters.
        manifold: Active manifold head — supplies ``pairwise_dist``,
            ``expmap0``, and (for the centroid update) acts as the
            ambient → manifold map.
        n_iter: Maximum Lloyd iterations.
        seed: Reproducibility seed for centroid initialisation.

    Returns:
        ``(labels, centroids)`` — ``labels`` shape ``[N]`` (int),
        ``centroids`` shape ``[k, D]`` (on-manifold).
    """
    n = embeddings.shape[0]
    if n < k:
        raise ValueError(f"Need at least k={k} points; got n={n}.")

    g = torch.Generator()
    g.manual_seed(seed)
    init = torch.randperm(n, generator=g)[:k]
    centroids = embeddings[init].clone()

    labels = torch.zeros(n, dtype=torch.long)
    for _ in range(n_iter):
        d = manifold.pairwise_dist(embeddings, centroids)  # [N, k]
        new_labels = d.argmin(dim=-1)
        # Convergence: assignments stable.
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        # Centroid update: cluster-mean in ambient, then expmap0 onto manifold.
        # This is an approximation of the Fréchet mean and is correct for
        # Euclidean exactly; for Spherical / Hyperbolic it's a fast, good-
        # enough proxy in practice.
        new_centroids = torch.zeros_like(centroids)
        for ki in range(k):
            mask = labels == ki
            count = int(mask.sum().item() if hasattr(mask.sum(), "item") else mask.sum())
            if count == 0:
                # Reinitialise empty cluster to a random point.
                new_centroids[ki] = embeddings[int(torch.randint(0, n, (1,), generator=g).item())]
            else:
                cluster_pts = embeddings[mask]
                ambient_mean = cluster_pts.mean(dim=0)
                new_centroids[ki] = manifold.expmap0(ambient_mean.unsqueeze(0)).squeeze(0)
        centroids = new_centroids

    return labels, centroids


def manifold_hdbscan(
    embeddings: torch.Tensor,
    manifold: ManifoldHead,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: Literal["eom", "leaf"] = "eom",
    allow_single_cluster: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manifold-aware HDBSCAN clustering on ``embeddings``.

    We compute the full pairwise manifold-distance matrix and hand it to
    :class:`sklearn.cluster.HDBSCAN` with ``metric='precomputed'``. This
    keeps the mutual-reachability / core-distance estimates faithful to
    the active geometry — radians on the sphere, Poincaré units in
    hyperbolic space, the L2-style metric on the Euclidean tangent — so
    density structure is measured in the same units the rest of the
    pipeline uses.

    Density-based clustering is naturally *non-parametric in k*: the
    algorithm decides how many clusters survive, and points in
    low-density regions are flagged as noise (label ``-1``). Cluster
    indices in the returned tensor are remapped to a contiguous
    ``0..C-1`` range and the centroids are returned as on-manifold
    medoids (the in-cluster point that minimises summed manifold
    distance to its peers), so the output is a drop-in replacement for
    :func:`manifold_kmeans` for downstream code that just needs
    ``(labels, centroids)``.

    Args:
        embeddings: ``[N, D]`` on-manifold points.
        manifold: Active manifold head — supplies ``pairwise_dist``.
        min_cluster_size: Smallest group size that survives as a
            cluster; everything smaller is folded into noise. Defaults
            to sklearn's 5.
        min_samples: Core-distance neighbourhood size. ``None`` →
            ``min_cluster_size`` (sklearn default).
        cluster_selection_epsilon: Merge clusters whose mutual-
            reachability distance is below this cutoff. Manifold-unit-
            dependent (radians on sphere, etc.); leave at ``0.0`` to
            disable.
        cluster_selection_method: ``"eom"`` (Excess of Mass — robust
            default) or ``"leaf"`` (more fine-grained).
        allow_single_cluster: Permit HDBSCAN to return a single cluster.
            Off by default because for the canonical use case the whole
            corpus is one cluster of nothing useful.

    Returns:
        ``(labels, centroids)`` — ``labels`` shape ``[N]`` (int, with
        ``-1`` for noise), ``centroids`` shape ``[C, D]`` (on-manifold
        medoids; ``C`` is the number of clusters HDBSCAN found, may
        be ``0`` if everything was noise).
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError as exc:  # pragma: no cover — scikit-learn is a core dep
        raise ImportError(
            "manifold_hdbscan requires scikit-learn>=1.3 (provides "
            "sklearn.cluster.HDBSCAN). Install with "
            "`pip install 'scikit-learn>=1.3'`."
        ) from exc

    n = embeddings.shape[0]
    d_dim = int(embeddings.shape[-1])
    if n == 0:
        empty_labels = torch.zeros((0,), dtype=torch.long)
        empty_centroids = torch.zeros((0, d_dim), dtype=embeddings.dtype)
        return empty_labels, empty_centroids

    # ── Build a clean, symmetric, non-negative distance matrix ──────────
    d = manifold.pairwise_dist(embeddings, embeddings)  # [N, N]
    d_np = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
    # Numerical hygiene: sklearn's precomputed path checks for symmetry
    # and non-negativity. Round-off in the manifold formulas can produce
    # tiny asymmetries / negative epsilons; clamp them.
    d_np = 0.5 * (d_np + d_np.T)
    np.fill_diagonal(d_np, 0.0)
    # Replace any inf values (e.g. acosh overflow on Windows longdouble) with
    # a large finite distance so downstream int conversion never sees inf.
    d_np = np.clip(d_np, 0.0, np.finfo(np.float64).max).astype(np.float64, copy=False)

    # ── Cluster ─────────────────────────────────────────────────────────
    model = HDBSCAN(
        min_cluster_size=max(2, int(min_cluster_size)),
        min_samples=min_samples,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        metric="precomputed",
    )
    raw_labels = np.asarray(model.fit_predict(d_np), dtype=np.int64)

    # ── Remap surviving cluster ids to contiguous 0..C-1, keep -1 as noise ─
    unique = sorted({int(c) for c in raw_labels.tolist() if int(c) >= 0})
    remap = {c: i for i, c in enumerate(unique)}
    remapped = np.array(
        [remap[int(c)] if int(c) >= 0 else -1 for c in raw_labels.tolist()],
        dtype=np.int64,
    )
    labels = torch.as_tensor(remapped, dtype=torch.long)

    # ── On-manifold medoid centroids ────────────────────────────────────
    if not unique:
        centroids = torch.zeros((0, d_dim), dtype=embeddings.dtype)
        return labels, centroids

    centroids = torch.zeros((len(unique), d_dim), dtype=embeddings.dtype)
    for new_idx, _ in enumerate(unique):
        members = np.where(remapped == new_idx)[0]
        # Medoid: the in-cluster point with smallest summed distance to peers.
        # For singleton clusters the sum is trivially zero and we pick that point.
        sub = d_np[np.ix_(members, members)]
        medoid_local = int(sub.sum(axis=1).argmin())
        medoid_idx = int(members[medoid_local])
        centroids[new_idx] = embeddings[medoid_idx]

    return labels, centroids


# ─────────────────────────────────────────────────────────────────────────
# Graph-connected-components clustering
# ─────────────────────────────────────────────────────────────────────────
def _knee_index(y: np.ndarray[Any, Any]) -> int:
    """Kneedle-style knee detection on an arbitrarily-shaped 1-D curve.

    Returns the index whose ``(i, y[i])`` lies furthest (perpendicular
    distance) from the chord connecting ``(0, y[0])`` and ``(N-1, y[-1])``.
    For an L-shaped curve — sorted-ascending k-NN distances are the
    canonical example — this is the elbow / knee.

    Robust to direction (works on ascending, descending, or generally
    monotone curves) and degenerate cases (constant curve → returns
    the last index; ``N < 3`` → returns ``N-1``).
    """
    n = len(y)
    if n < 3:
        return n - 1
    y_f = np.asarray(y, dtype=np.float64)
    x0, y0 = 0.0, float(y_f[0])
    x1, y1 = float(n - 1), float(y_f[-1])
    dx = x1 - x0
    dy = y1 - y0
    denom = float(np.hypot(dx, dy)) or 1.0
    xs = np.arange(n, dtype=np.float64)
    # Perpendicular distance from (xs[i], y_f[i]) to the chord.
    # |dy * x - dx * y + (x1 * y0 - x0 * y1)| / sqrt(dx² + dy²)
    perp = np.abs(dy * xs - dx * y_f + (x1 * y0 - x0 * y1)) / denom
    return int(np.argmax(perp))


def _radius_knee(d_np: np.ndarray[Any, Any], *, k: int) -> float:
    """k-NN distance knee — the canonical DBSCAN radius heuristic
    (Ester, Kriegel, Sander & Xu 1996).

    For each point compute its ``k``-th nearest-neighbour distance
    (excluding self). The resulting ``N``-vector, sorted ascending, has
    a characteristic L-shape: dense cluster cores hug the bottom; sparse
    / noise points spike at the top. The knee of that curve is the
    distance at which a point transitions from "neighbour-rich" to
    "neighbour-poor", and is the natural cutoff for a radius graph.
    ``k`` is clamped to ``[1, N-1]``.
    """
    n = d_np.shape[0]
    if n <= 1:
        return 0.0
    kk = max(1, min(int(k), n - 1))
    # Row-sort ascending: column 0 is self (distance 0); column kk is
    # the k-th nearest neighbour.
    sorted_rows = np.sort(d_np, axis=1)
    knn_dists = sorted_rows[:, kk]
    y = np.sort(knn_dists)  # ascending across points
    return float(y[_knee_index(y)])


def _radius_mst_gap(d_np: np.ndarray[Any, Any]) -> float:
    """Largest-gap cut in the MST edge-weight distribution.

    Single-linkage agglomerative clustering corresponds to traversing
    the minimum spanning tree in order of edge weight. The "biggest
    gap" between consecutive sorted MST edges marks the most natural
    cluster boundary: cutting all MST edges above that gap yields the
    cluster forest. We return the *lower* endpoint of that gap so
    callers can use ``d <= r`` as the adjacency predicate.

    Note: requires scipy (already a core dep).
    """
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
    except ImportError as exc:  # pragma: no cover — scipy is a core dep
        raise ImportError(
            "manifold_graph_cc with r_method='mst_gap' requires scipy. "
            "Install with `pip install 'scipy>=1.12'`."
        ) from exc

    mst = minimum_spanning_tree(d_np).toarray()
    weights = mst[mst > 0]
    if weights.size == 0:
        return 0.0
    if weights.size == 1:
        # Only one edge: keep it, so r >= that weight → everything
        # connects into a single CC.
        return float(weights[0])
    sorted_w = np.sort(weights)
    gaps = np.diff(sorted_w)
    gap_idx = int(np.argmax(gaps))
    return float(sorted_w[gap_idx])


def _connected_components_from_radius(d_np: np.ndarray[Any, Any], r: float) -> np.ndarray[Any, Any]:
    """Adjacency = ``d <= r`` (excluding self-loops); return CC labels.

    scipy's csgraph is the obvious choice — it handles the sparse
    representation and union-find internally. We pass a dense boolean
    matrix wrapped in a CSR view; for the small N our scenarios cluster
    (≤ corpus size), the constant factor is fine. Sparse construction
    becomes worthwhile once ``N`` gets into the hundreds of thousands,
    at which point this whole code path needs a chunked-distance
    rewrite anyway.
    """
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError as exc:  # pragma: no cover — scipy is a core dep
        raise ImportError(
            "manifold_graph_cc requires scipy for connected_components. "
            "Install with `pip install 'scipy>=1.12'`."
        ) from exc

    n = d_np.shape[0]
    adj = (d_np <= float(r)) & ~np.eye(n, dtype=bool)
    _, raw = connected_components(csr_matrix(adj), directed=False)
    return cast("np.ndarray[Any, Any]", raw.astype(np.int64))


def manifold_graph_cc(
    embeddings: torch.Tensor,
    manifold: ManifoldHead,
    *,
    radius: float | None = None,
    r_method: GraphCCRadiusMethod = "knee",
    k_neighbors: int = 4,
    min_cluster_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manifold-aware similarity-graph → radius-graph → connected-components.

    Pipeline:

    1. **Similarity graph.** Compute the full pairwise manifold-distance
       matrix ``D[N, N]`` via :meth:`ManifoldHead.pairwise_dist`. (Strictly
       speaking we work with distances rather than similarities; "similarity
       graph" is the colloquial framing — the geometry is identical and
       inverting to similarities adds no information.)
    2. **Radius graph.** Threshold ``D <= r`` to keep only short edges.
       The radius ``r`` is either supplied explicitly via ``radius=`` or
       auto-determined (see below).
    3. **Connected components.** Run union-find / BFS over the radius
       graph; every connected subgraph becomes a cluster. Components of
       size below ``min_cluster_size`` are folded into the noise bucket
       (label ``-1``), mirroring HDBSCAN's behaviour and the convention
       the rest of the scenarios already expect.

    Auto-radius determination (when ``radius`` is ``None``):

    - ``r_method="knee"`` (default): The canonical DBSCAN heuristic
      (Ester et al. 1996). For each point compute its ``k_neighbors``-th
      nearest-neighbour distance; sort the resulting N-vector ascending;
      pick the knee via Kneedle-style perpendicular distance from the
      chord. ``k_neighbors=4`` is the DBSCAN paper's recommendation for
      2-D data and is a reasonable default in higher dimensions; raise
      it for noisier corpora.
    - ``r_method="mst_gap"``: Build the minimum spanning tree of the
      distance matrix, sort its edge weights ascending, find the largest
      consecutive gap, and use the gap's lower endpoint as ``r``. Cleaner
      theoretical interpretation (single-linkage cut at maximum
      separation) but more sensitive to outliers that introduce large
      MST edges.

    Args:
        embeddings: ``[N, D]`` on-manifold points.
        manifold: Active manifold head — supplies ``pairwise_dist``.
        radius: Explicit radius. Overrides ``r_method`` when given.
            Manifold-unit-dependent (radians on sphere, Poincaré units
            in hyperbolic space, …).
        r_method: Auto-radius heuristic when ``radius`` is ``None``.
        k_neighbors: ``k`` for the ``knee`` heuristic. Ignored for
            ``mst_gap``. Clamped to ``[1, N-1]`` internally.
        min_cluster_size: Smallest CC that survives as a real cluster.
            Default ``2`` folds isolated points (size-1 CCs) into noise.
            Set to ``1`` to keep every CC, including singletons, as its
            own cluster.

    Returns:
        ``(labels, centroids)`` — ``labels`` shape ``[N]`` (int; ``-1``
        marks noise / sub-``min_cluster_size`` CCs), ``centroids`` shape
        ``[C, D]`` (on-manifold medoids, one per surviving cluster).
    """
    n = embeddings.shape[0]
    d_dim = int(embeddings.shape[-1])
    if n == 0:
        return (
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0, d_dim), dtype=embeddings.dtype),
        )

    # ── Build a clean, symmetric, non-negative distance matrix ──────────
    d = manifold.pairwise_dist(embeddings, embeddings)  # [N, N]
    d_np = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
    d_np = 0.5 * (d_np + d_np.T)
    np.fill_diagonal(d_np, 0.0)
    # Replace any inf values (e.g. acosh overflow on Windows longdouble) with
    # a large finite distance so downstream int conversion never sees inf.
    d_np = np.clip(d_np, 0.0, np.finfo(np.float64).max).astype(np.float64, copy=False)

    # ── Determine radius ────────────────────────────────────────────────
    if radius is None:
        if r_method == "knee":
            r = _radius_knee(d_np, k=k_neighbors)
        elif r_method == "mst_gap":
            r = _radius_mst_gap(d_np)
        else:
            raise ValueError(f"Unknown r_method {r_method!r}; expected 'knee' or 'mst_gap'.")
        _log.info(
            "manifold_graph_cc: auto-radius via %r → r=%.6f (N=%d, k=%d)",
            r_method,
            r,
            n,
            k_neighbors,
        )
    else:
        r = float(radius)
        _log.info("manifold_graph_cc: using explicit radius r=%.6f (N=%d)", r, n)

    # ── Connected components on the radius graph ────────────────────────
    raw_labels = _connected_components_from_radius(d_np, r)

    # ── Fold sub-``min_cluster_size`` CCs into noise ────────────────────
    unique, counts = np.unique(raw_labels, return_counts=True)
    noise_ids = {
        int(c)
        for c, sz in zip(unique.tolist(), counts.tolist(), strict=True)
        if int(sz) < int(min_cluster_size)
    }
    surviving = sorted(int(c) for c in unique.tolist() if int(c) not in noise_ids)
    remap = {c: i for i, c in enumerate(surviving)}
    remapped = np.array(
        [remap[int(c)] if int(c) not in noise_ids else -1 for c in raw_labels.tolist()],
        dtype=np.int64,
    )
    labels = torch.as_tensor(remapped, dtype=torch.long)

    # ── On-manifold medoid centroids ────────────────────────────────────
    if not surviving:
        centroids = torch.zeros((0, d_dim), dtype=embeddings.dtype)
        return labels, centroids

    centroids = torch.zeros((len(surviving), d_dim), dtype=embeddings.dtype)
    for new_idx in range(len(surviving)):
        members = np.where(remapped == new_idx)[0]
        sub = d_np[np.ix_(members, members)]
        medoid_local = int(sub.sum(axis=1).argmin())
        centroids[new_idx] = embeddings[int(members[medoid_local])]

    return labels, centroids


# ─────────────────────────────────────────────────────────────────────────
# Leiden community detection
# ─────────────────────────────────────────────────────────────────────────
def _knn_edges(
    d_np: np.ndarray[Any, Any], *, k: int, mutual: bool
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build the (mutual) k-NN edge list from a precomputed distance matrix.

    Returns ``(src, dst, dist)`` arrays of equal length (one entry per
    undirected edge, ``src < dst`` to deduplicate). Self-loops are
    excluded; for the non-mutual variant we union the directed edges
    from both endpoints' k-NN lists so the resulting graph is
    symmetric (k-NN is inherently asymmetric: ``i`` may be in ``j``'s
    k-NN without the reverse holding).
    """
    n = d_np.shape[0]
    if n <= 1:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, empty.astype(np.float64)
    kk = max(1, min(int(k), n - 1))
    # For each row, indices of the k nearest neighbours (excluding self).
    # argpartition is O(N) per row vs argsort's O(N log N) — small win on
    # corpora large enough to want k-NN clustering in the first place.
    # We exclude self by setting the diagonal to +inf temporarily.
    d_masked = d_np.copy()
    np.fill_diagonal(d_masked, np.inf)
    nn_idx = np.argpartition(d_masked, kth=kk - 1, axis=1)[:, :kk]  # [N, k]

    # Build the membership set per node for the mutual check
    nn_set: list[set[int]] = [{int(j) for j in row} for row in nn_idx.tolist()]

    src_list: list[int] = []
    dst_list: list[int] = []
    for i in range(n):
        for j_raw in nn_idx[i].tolist():
            j = int(j_raw)
            if j == i:
                continue
            if mutual and i not in nn_set[j]:
                continue
            # Canonicalise to undirected edge with src < dst.
            a, b = (i, j) if i < j else (j, i)
            src_list.append(a)
            dst_list.append(b)

    if not src_list:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, empty.astype(np.float64)

    # Dedup — for non-mutual k-NN the same (a, b) shows up twice if
    # they're mutual; for mutual_knn it always shows up twice. Either
    # way, np.unique on a structured view is cleanest.
    edges = np.stack([np.asarray(src_list), np.asarray(dst_list)], axis=1)
    edges = np.unique(edges, axis=0)
    src = edges[:, 0].astype(np.int64)
    dst = edges[:, 1].astype(np.int64)
    dist = d_np[src, dst].astype(np.float64)
    return src, dst, dist


def _radius_edges(
    d_np: np.ndarray[Any, Any], *, radius: float
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build the radius-graph edge list (``d <= radius``, no self-loops)."""
    n = d_np.shape[0]
    mask = (d_np <= float(radius)) & ~np.eye(n, dtype=bool)
    # Take only the upper triangle so each edge appears once.
    iu, ju = np.triu_indices(n, k=1)
    sel = mask[iu, ju]
    src = iu[sel].astype(np.int64)
    dst = ju[sel].astype(np.int64)
    dist = d_np[src, dst].astype(np.float64)
    return src, dst, dist


def _distance_to_similarity(dist: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Gaussian RBF similarity: ``w = exp(-d^2 / (2*sigma^2))`` with ``sigma`` = median
    edge distance. Scale-free in the manifold's distance units — works
    identically on radians (spherical), Poincaré units (hyperbolic) and
    raw L2 (Euclidean).

    Empty input → empty output. Constant-distance input (e.g. all edges
    at the same length) → all weights equal to ``exp(-1/2) ≈ 0.606``,
    which still yields a well-defined modularity objective.
    """
    if dist.size == 0:
        return np.zeros((0,), dtype=np.float64)
    sigma = float(np.median(dist))
    if sigma <= 0.0:
        # Degenerate: zero-distance edges. Treat as uniform weight 1.
        return np.ones_like(dist, dtype=np.float64)
    return cast(
        "np.ndarray[Any, Any]",
        np.exp(-(dist**2) / (2.0 * sigma**2)).astype(np.float64),
    )


def manifold_leiden(
    embeddings: torch.Tensor,
    manifold: ManifoldHead,
    *,
    graph_method: LeidenGraphMethod = "knn",
    k_neighbors: int = 15,
    radius: float | None = None,
    r_method: GraphCCRadiusMethod = "knee",
    quality: LeidenQuality = "modularity",
    resolution: float = 1.0,
    min_cluster_size: int = 2,
    seed: int = 0,
    n_iterations: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manifold-aware Louvain community detection (Blondel et al. 2008).

    Backed by networkx's Louvain partitioner (BSD-3) — the
    modularity-optimization predecessor of Leiden, with very similar
    partitions on the graphs built here. (The GPL ``leidenalg`` +
    ``python-igraph`` Leiden backend can't ship in this package's Apache-2.0
    license.) The function name and the ``"leiden"`` algorithm selector name
    the algorithm family. Three graph-construction modes feed into the same
    partitioner:

    - ``graph_method="knn"`` (default): for each point connect to its
      ``k_neighbors`` nearest neighbours in manifold distance. The
      adjacency is symmetrized (edges from both endpoints' k-NN lists
      are unioned) so the partitioning sees an undirected graph.
    - ``graph_method="mutual_knn"``: keep an edge only when both
      endpoints have each other in their k-NN list. Strictly fewer
      edges than ``knn`` — less prone to bridging weak clusters, more
      likely to leave true outliers as isolated singletons.
    - ``graph_method="radius"``: same radius-graph construction used
      by :func:`manifold_graph_cc`, with the same auto-radius
      heuristics (``knee`` / ``mst_gap``). Use when you want Leiden's
      partitioning quality on top of an interpretable distance cutoff.

    Edge weights are Gaussian RBF on manifold distance:
    ``w = exp(-d^2 / (2*sigma^2))`` with ``sigma`` the median edge distance —
    scale-free in the manifold's distance units, so the same ``resolution``
    has roughly the same effect across Euclidean / Spherical / Hyperbolic
    setups.

    Quality functions:

    - ``quality="modularity"`` (default): Reichardt-Bornholdt modularity,
      which is the classical Newman-Girvan modularity at ``resolution=1.0``
      but lets higher values produce smaller, finer-grained clusters. This
      is exactly what networkx's Louvain optimizes via its ``resolution``
      (gamma) parameter. Has a well-known resolution limit on very large
      graphs.
    - ``quality="cpm"``: the Constant Potts Model has no networkx Louvain
      equivalent, so it is approximated by the same resolution-scaled
      modularity optimization as ``"modularity"``. Retained for API
      compatibility; prefer ``"modularity"`` unless you have a specific
      reason and have retuned ``resolution`` for your graph density.

    Communities smaller than ``min_cluster_size`` are folded into the
    noise bucket (label ``-1``), matching HDBSCAN's convention. Cluster
    representatives are on-manifold medoids per surviving community.

    Args:
        embeddings: ``[N, D]`` on-manifold points.
        manifold: Active manifold head — supplies ``pairwise_dist``.
        graph_method: How to build the graph (see above).
        k_neighbors: ``k`` for the (mutual-)k-NN graphs, clamped to
            ``[1, N-1]``. **Not** ignored under ``graph_method="radius"``:
            with ``radius=None`` and ``r_method="knee"`` it is the ``k``
            of the k-NN-distance knee, so it still sets the radius — the
            same role ``graph_cc_k_neighbors`` and ``dbscan_k_neighbors``
            play for their algorithms, but spelled with the k-NN knob
            rather than a dedicated one. It *is* ignored when an explicit
            ``radius`` is given, or under ``r_method="mst_gap"``.
        radius: Explicit radius for the ``radius`` graph mode. If
            ``None``, auto-pick via ``r_method``. Ignored for k-NN
            modes.
        r_method: Auto-radius heuristic for the ``radius`` mode.
        quality: Optimisation objective (see above).
        resolution: Resolution parameter — higher ⇒ more, smaller
            communities. ``1.0`` matches classical modularity.
        min_cluster_size: Smallest community kept as a real cluster;
            anything smaller goes to noise.
        seed: Reproducibility seed for Louvain's randomized moves.
        n_iterations: Cap on Louvain aggregation passes (maps to
            networkx's ``max_level``). ``-1`` = run until convergence
            (networkx default — no cap).

    Returns:
        ``(labels, centroids)`` — ``labels`` shape ``[N]`` (int, with
        ``-1`` for noise), ``centroids`` shape ``[C, D]`` (on-manifold
        medoids; ``C`` is the number of surviving communities).
    """
    import networkx as nx

    n = embeddings.shape[0]
    d_dim = int(embeddings.shape[-1])
    if n == 0:
        return (
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0, d_dim), dtype=embeddings.dtype),
        )

    # ── Pairwise manifold distance, hygiene-cleaned ─────────────────────
    d = manifold.pairwise_dist(embeddings, embeddings)
    d_np = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
    d_np = 0.5 * (d_np + d_np.T)
    np.fill_diagonal(d_np, 0.0)
    # Replace any inf values (e.g. acosh overflow on Windows longdouble) with
    # a large finite distance so downstream int conversion never sees inf.
    d_np = np.clip(d_np, 0.0, np.finfo(np.float64).max).astype(np.float64, copy=False)

    # ── Build edge list ─────────────────────────────────────────────────
    if graph_method == "knn":
        src, dst, dist = _knn_edges(d_np, k=k_neighbors, mutual=False)
    elif graph_method == "mutual_knn":
        src, dst, dist = _knn_edges(d_np, k=k_neighbors, mutual=True)
    elif graph_method == "radius":
        if radius is None:
            r = _radius_knee(d_np, k=k_neighbors) if r_method == "knee" else _radius_mst_gap(d_np)
            _log.info(
                "manifold_leiden: auto-radius via %r → r=%.6f (N=%d, k=%d)",
                r_method,
                r,
                n,
                k_neighbors,
            )
        else:
            r = float(radius)
        src, dst, dist = _radius_edges(d_np, radius=r)
    else:
        raise ValueError(
            f"Unknown graph_method {graph_method!r}; expected 'knn', 'mutual_knn', or 'radius'."
        )

    weights = _distance_to_similarity(dist)

    # ── Run Louvain community detection (networkx, BSD-3) ───────────────
    # leidenalg + python-igraph is GPL and can't ship in this Apache-2.0 package, so
    # this uses networkx's Louvain. Its RB-modularity objective takes the same
    # ``resolution`` (gamma) parameter, so ``quality="modularity"`` maps
    # directly. The CPM objective has no networkx equivalent, so
    # ``quality="cpm"`` is served by the same resolution-scaled modularity
    # optimizer (an approximation — see docstring).
    if quality not in ("modularity", "cpm"):
        raise ValueError(f"Unknown quality {quality!r}; expected 'modularity' or 'cpm'.")

    g: nx.Graph[int] = nx.Graph()
    g.add_nodes_from(range(int(n)))
    if weights.size:
        g.add_weighted_edges_from(zip(src.tolist(), dst.tolist(), weights.tolist(), strict=True))
    else:
        g.add_edges_from(zip(src.tolist(), dst.tolist(), strict=True))

    # ``max_level`` caps the aggregation passes; ``n_iterations < 0`` (run to
    # convergence) maps to networkx's default (no cap).
    louvain_kwargs: dict[str, Any] = {}
    if n_iterations >= 0:
        louvain_kwargs["max_level"] = int(n_iterations)
    communities = nx.community.louvain_communities(
        g,
        weight="weight",
        resolution=float(resolution),
        seed=int(seed),
        **louvain_kwargs,
    )
    raw_labels = np.empty((int(n),), dtype=np.int64)
    for comm_id, community in enumerate(communities):
        for node in community:
            raw_labels[int(node)] = comm_id
    _log.info(
        "manifold_leiden: louvain(%s) @ resolution=%.3f → %d raw communities (N=%d)",
        quality,
        resolution,
        len(communities),
        n,
    )

    # ── Fold sub-``min_cluster_size`` communities into noise ────────────
    unique, counts = np.unique(raw_labels, return_counts=True)
    noise_ids = {
        int(c)
        for c, sz in zip(unique.tolist(), counts.tolist(), strict=True)
        if int(sz) < int(min_cluster_size)
    }
    surviving = sorted(int(c) for c in unique.tolist() if int(c) not in noise_ids)
    remap = {c: i for i, c in enumerate(surviving)}
    remapped = np.array(
        [remap[int(c)] if int(c) not in noise_ids else -1 for c in raw_labels.tolist()],
        dtype=np.int64,
    )
    labels = torch.as_tensor(remapped, dtype=torch.long)

    # ── On-manifold medoid centroids ────────────────────────────────────
    if not surviving:
        centroids = torch.zeros((0, d_dim), dtype=embeddings.dtype)
        return labels, centroids

    centroids = torch.zeros((len(surviving), d_dim), dtype=embeddings.dtype)
    for new_idx in range(len(surviving)):
        members = np.where(remapped == new_idx)[0]
        sub = d_np[np.ix_(members, members)]
        medoid_local = int(sub.sum(axis=1).argmin())
        centroids[new_idx] = embeddings[int(members[medoid_local])]

    return labels, centroids


# ─────────────────────────────────────────────────────────────────────────
# Additional scikit-learn clusterers that don't take ``k``
#
# DBSCAN, OPTICS and AffinityPropagation all accept a precomputed pairwise
# matrix, so — like HDBSCAN and graph_cc above — they cluster in the active
# manifold geometry rather than ambient Euclidean space. MeanShift is the
# odd one out: scikit-learn implements it only over Euclidean feature space
# (no ``metric='precomputed'`` path), so it clusters the embedding
# coordinates directly. All four share the same ``(labels, centroids)``
# contract — labels in ``0..C-1`` with ``-1`` for noise, centroids as
# on-manifold medoids — so they drop straight into :func:`cluster_embeddings`.
# ─────────────────────────────────────────────────────────────────────────
def _manifold_distance_matrix(
    embeddings: torch.Tensor, manifold: ManifoldHead
) -> np.ndarray[Any, Any]:
    """Precomputed pairwise manifold-distance matrix, cleaned for sklearn.

    Symmetrised, zero-diagonal, non-negative and finite — the same numerical
    hygiene :func:`manifold_hdbscan` / :func:`manifold_graph_cc` apply before
    handing a ``metric='precomputed'`` matrix to scikit-learn.
    """
    d = manifold.pairwise_dist(embeddings, embeddings)  # [N, N]
    d_np = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
    d_np = 0.5 * (d_np + d_np.T)
    np.fill_diagonal(d_np, 0.0)
    d_np = np.clip(d_np, 0.0, np.finfo(np.float64).max).astype(np.float64, copy=False)
    return cast("np.ndarray[Any, Any]", d_np)


def _labels_and_medoids(
    raw_labels: np.ndarray[Any, Any],
    d_np: np.ndarray[Any, Any],
    embeddings: torch.Tensor,
    *,
    min_cluster_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remap raw sklearn labels to contiguous ``0..C-1`` and build medoids.

    Negative labels are noise; groups smaller than ``min_cluster_size`` are
    folded into noise too. Centroids are on-manifold medoids (the in-cluster
    point minimising summed distance to its peers), computed from ``d_np`` —
    the same convention used by every other clusterer in this module, so the
    output is geometry-faithful even when the clustering itself ran in
    Euclidean space (MeanShift).
    """
    d_dim = int(embeddings.shape[-1])
    unique, counts = np.unique(raw_labels, return_counts=True)
    noise_ids = {
        int(c)
        for c, sz in zip(unique.tolist(), counts.tolist(), strict=True)
        if int(c) < 0 or int(sz) < int(min_cluster_size)
    }
    surviving = sorted(int(c) for c in unique.tolist() if int(c) not in noise_ids)
    remap = {c: i for i, c in enumerate(surviving)}
    remapped = np.array(
        [remap[int(c)] if int(c) not in noise_ids else -1 for c in raw_labels.tolist()],
        dtype=np.int64,
    )
    labels = torch.as_tensor(remapped, dtype=torch.long)

    if not surviving:
        return labels, torch.zeros((0, d_dim), dtype=embeddings.dtype)

    centroids = torch.zeros((len(surviving), d_dim), dtype=embeddings.dtype)
    for new_idx in range(len(surviving)):
        members = np.where(remapped == new_idx)[0]
        sub = d_np[np.ix_(members, members)]
        medoid_local = int(sub.sum(axis=1).argmin())
        centroids[new_idx] = embeddings[int(members[medoid_local])]
    return labels, centroids


def manifold_dbscan(
    embeddings: torch.Tensor,
    manifold: ManifoldHead,
    *,
    eps: float | None = None,
    r_method: GraphCCRadiusMethod = "knee",
    k_neighbors: int = 4,
    min_samples: int = 5,
    min_cluster_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manifold-aware DBSCAN (Ester, Kriegel, Sander & Xu 1996).

    Density clustering on the precomputed manifold-distance matrix via
    :class:`sklearn.cluster.DBSCAN` with ``metric='precomputed'``. Two core
    knobs and no ``k``:

    - ``eps``: neighbourhood radius (manifold units). When ``None`` it is
      auto-picked with the same heuristics :func:`manifold_graph_cc` uses —
      the canonical k-NN-distance knee (``r_method='knee'``, Ester et al.'s
      recommendation) or the MST max-gap (``r_method='mst_gap'``).
    - ``min_samples``: core-point neighbourhood size. Larger ⇒ more points
      flagged as noise.

    ``min_cluster_size`` folds sub-threshold clusters into noise after the
    fact (DBSCAN itself has no such knob); the default ``1`` keeps every
    cluster DBSCAN returns.

    Returns ``(labels, centroids)`` — ``labels`` ``[N]`` (``-1`` = noise),
    ``centroids`` ``[C, D]`` on-manifold medoids.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as exc:  # pragma: no cover — scikit-learn is a core dep
        raise ImportError(
            "manifold_dbscan requires scikit-learn. Install with `pip install scikit-learn`."
        ) from exc

    n = embeddings.shape[0]
    d_dim = int(embeddings.shape[-1])
    if n == 0:
        return torch.zeros((0,), dtype=torch.long), torch.zeros((0, d_dim), dtype=embeddings.dtype)

    d_np = _manifold_distance_matrix(embeddings, manifold)

    if eps is None:
        if r_method == "knee":
            r = _radius_knee(d_np, k=k_neighbors)
        elif r_method == "mst_gap":
            r = _radius_mst_gap(d_np)
        else:
            raise ValueError(f"Unknown r_method {r_method!r}; expected 'knee' or 'mst_gap'.")
        # A degenerate knee at 0 would make every point its own noise cluster;
        # nudge to the smallest positive off-diagonal distance so DBSCAN can
        # connect at least the nearest pairs.
        if r <= 0.0:
            off_diag = d_np[~np.eye(n, dtype=bool)]
            positive = off_diag[off_diag > 0.0]
            r = float(positive.min()) if positive.size else 1.0
        _log.info(
            "manifold_dbscan: auto-eps via %r → eps=%.6f (N=%d, k=%d)", r_method, r, n, k_neighbors
        )
    else:
        r = float(eps)
        _log.info("manifold_dbscan: using explicit eps=%.6f (N=%d)", r, n)

    model = DBSCAN(eps=r, min_samples=max(1, int(min_samples)), metric="precomputed")
    raw_labels = np.asarray(model.fit_predict(d_np), dtype=np.int64)
    return _labels_and_medoids(raw_labels, d_np, embeddings, min_cluster_size=min_cluster_size)


def manifold_optics(
    embeddings: torch.Tensor,
    manifold: ManifoldHead,
    *,
    min_samples: int = 5,
    xi: float = 0.05,
    min_cluster_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manifold-aware OPTICS (Ankerst, Breunig, Kriegel & Sander 1999).

    OPTICS generalises DBSCAN to clusters of *varying* density by ordering
    points along a reachability plot and extracting clusters from its
    valleys, so there is no single ``eps`` to tune. We run
    :class:`sklearn.cluster.OPTICS` with ``metric='precomputed'`` and the
    default ``cluster_method='xi'`` extraction.

    - ``min_samples``: core-distance neighbourhood size (the main density
      knob). Clamped to ``[2, N-1]``.
    - ``xi``: minimum relative steepness of a reachability valley wall that
      marks a cluster boundary; lower ⇒ more, finer clusters.
    - ``min_cluster_size``: smallest extractable cluster. ``None`` →
      scikit-learn's default (ties it to ``min_samples``); an ``int`` sets an
      absolute floor.

    Returns ``(labels, centroids)`` — ``labels`` ``[N]`` (``-1`` = noise),
    ``centroids`` ``[C, D]`` on-manifold medoids.
    """
    try:
        from sklearn.cluster import OPTICS
    except ImportError as exc:  # pragma: no cover — scikit-learn is a core dep
        raise ImportError(
            "manifold_optics requires scikit-learn. Install with `pip install scikit-learn`."
        ) from exc

    n = embeddings.shape[0]
    d_dim = int(embeddings.shape[-1])
    if n == 0:
        return torch.zeros((0,), dtype=torch.long), torch.zeros((0, d_dim), dtype=embeddings.dtype)

    d_np = _manifold_distance_matrix(embeddings, manifold)

    # OPTICS needs ``2 <= min_samples <= N`` and at least 2 points to run.
    if n < 2:
        return torch.zeros((n,), dtype=torch.long), embeddings.clone()
    ms = max(2, min(int(min_samples), n - 1))
    model = OPTICS(
        min_samples=ms,
        xi=float(xi),
        min_cluster_size=min_cluster_size,
        metric="precomputed",
        cluster_method="xi",
    )
    raw_labels = np.asarray(model.fit_predict(d_np), dtype=np.int64)
    return _labels_and_medoids(raw_labels, d_np, embeddings)


def manifold_affinity_propagation(
    embeddings: torch.Tensor,
    manifold: ManifoldHead,
    *,
    damping: float = 0.5,
    preference: float | None = None,
    max_iter: int = 200,
    convergence_iter: int = 15,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manifold-aware Affinity Propagation (Frey & Dueck 2007).

    Exemplar-based clustering by message passing — the algorithm discovers
    both the exemplars and their count, so no ``k`` is required. We feed
    :class:`sklearn.cluster.AffinityPropagation` a precomputed *similarity*
    matrix (``affinity='precomputed'``) built as the negated manifold-distance
    matrix, so "more similar" means "closer on the manifold".

    - ``damping`` (``[0.5, 1.0)``): smooths message updates to avoid
      oscillation; raise it toward ``1.0`` if the run fails to converge.
    - ``preference``: the self-similarity each point gets, which controls how
      many exemplars emerge (higher ⇒ more clusters). ``None`` follows
      scikit-learn and uses the median input similarity — a reasonable
      data-driven default.

    Non-convergence yields an all-noise labelling (every label ``-1``),
    surfaced through the usual noise bucket rather than raising.

    Returns ``(labels, centroids)`` — ``labels`` ``[N]`` (``-1`` = noise),
    ``centroids`` ``[C, D]`` on-manifold medoids (which coincide with the
    exemplars Affinity Propagation selects).
    """
    try:
        from sklearn.cluster import AffinityPropagation
    except ImportError as exc:  # pragma: no cover — scikit-learn is a core dep
        raise ImportError(
            "manifold_affinity_propagation requires scikit-learn. "
            "Install with `pip install scikit-learn`."
        ) from exc

    n = embeddings.shape[0]
    d_dim = int(embeddings.shape[-1])
    if n == 0:
        return torch.zeros((0,), dtype=torch.long), torch.zeros((0, d_dim), dtype=embeddings.dtype)
    if n == 1:
        return torch.zeros((1,), dtype=torch.long), embeddings.clone()

    d_np = _manifold_distance_matrix(embeddings, manifold)
    similarity = -d_np  # closer ⇒ more similar

    model = AffinityPropagation(
        damping=float(damping),
        preference=preference,
        max_iter=int(max_iter),
        convergence_iter=int(convergence_iter),
        affinity="precomputed",
        random_state=seed,
    )
    raw_labels = np.asarray(model.fit_predict(similarity), dtype=np.int64)
    return _labels_and_medoids(raw_labels, d_np, embeddings)


def manifold_meanshift(
    embeddings: torch.Tensor,
    manifold: ManifoldHead,
    *,
    bandwidth: float | None = None,
    quantile: float = 0.3,
    bin_seeding: bool = False,
    cluster_all: bool = True,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mode-seeking MeanShift (Comaniciu & Meer 2002).

    MeanShift finds the modes of the sample density and assigns each point to
    the mode it climbs to, discovering the cluster count on its own.

    Unlike the other clusterers here it is **not manifold-aware**:
    scikit-learn's :class:`~sklearn.cluster.MeanShift` has no
    ``metric='precomputed'`` path, so it operates on the raw embedding
    coordinates in ambient Euclidean space. On a non-Euclidean representation
    manifold the assignments are therefore only approximate; the returned
    centroids are still on-manifold medoids (computed from the manifold
    distance matrix) so downstream code stays consistent.

    - ``bandwidth``: kernel radius in embedding-coordinate units. ``None`` →
      estimated from the data via :func:`sklearn.cluster.estimate_bandwidth`
      using ``quantile``.
    - ``quantile`` (``(0, 1]``): fraction of pairwise distances used by the
      bandwidth estimator; larger ⇒ wider kernel ⇒ fewer clusters.
    - ``cluster_all``: when ``True`` (default) every point is assigned to its
      nearest mode; when ``False`` orphans become noise (``-1``).

    Returns ``(labels, centroids)`` — ``labels`` ``[N]`` (``-1`` = noise iff
    ``cluster_all=False``), ``centroids`` ``[C, D]`` on-manifold medoids.
    """
    try:
        from sklearn.cluster import MeanShift, estimate_bandwidth
    except ImportError as exc:  # pragma: no cover — scikit-learn is a core dep
        raise ImportError(
            "manifold_meanshift requires scikit-learn. Install with `pip install scikit-learn`."
        ) from exc

    n = embeddings.shape[0]
    d_dim = int(embeddings.shape[-1])
    if n == 0:
        return torch.zeros((0,), dtype=torch.long), torch.zeros((0, d_dim), dtype=embeddings.dtype)
    if n == 1:
        return torch.zeros((1,), dtype=torch.long), embeddings.clone()

    if hasattr(embeddings, "detach"):
        x_raw = embeddings.detach().cpu().numpy()
    else:
        x_raw = np.asarray(embeddings)
    x_np = x_raw.astype(np.float64, copy=False)

    bw = bandwidth
    if bw is None:
        bw = float(estimate_bandwidth(x_np, quantile=float(quantile), random_state=seed))
        # A zero/non-positive estimate (e.g. many coincident points) makes
        # MeanShift raise; fall back to letting it self-estimate per seed.
        if bw <= 0.0:
            bw = None
        _log.info("manifold_meanshift: estimated bandwidth=%s (N=%d, q=%.3f)", bw, n, quantile)

    model = MeanShift(bandwidth=bw, bin_seeding=bin_seeding, cluster_all=cluster_all)
    raw_labels = np.asarray(model.fit_predict(x_np), dtype=np.int64)

    # Medoids are computed in the manifold geometry even though clustering ran
    # in Euclidean space — keeps centroids comparable to the other algorithms.
    d_np = _manifold_distance_matrix(embeddings, manifold)
    return _labels_and_medoids(raw_labels, d_np, embeddings)


def cluster_embeddings(
    embeddings: torch.Tensor,
    manifold: ManifoldHead,
    *,
    algorithm: ClusterAlgorithm = "kmeans",
    k: int | None = None,
    seed: int = 0,
    # HDBSCAN-only knobs (ignored for other algorithms).
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: Literal["eom", "leaf"] = "eom",
    allow_single_cluster: bool = False,
    # graph_cc-only knobs.  Prefixed so they don't collide with HDBSCAN's
    # ``min_cluster_size``.
    graph_cc_radius: float | None = None,
    graph_cc_r_method: GraphCCRadiusMethod = "knee",
    graph_cc_k_neighbors: int = 4,
    graph_cc_min_cluster_size: int = 2,
    # Leiden-only knobs.
    leiden_graph_method: LeidenGraphMethod = "knn",
    leiden_k_neighbors: int = 15,
    leiden_radius: float | None = None,
    leiden_r_method: GraphCCRadiusMethod = "knee",
    leiden_quality: LeidenQuality = "modularity",
    leiden_resolution: float = 1.0,
    leiden_min_cluster_size: int = 2,
    leiden_n_iterations: int = -1,
    # DBSCAN-only knobs.
    dbscan_eps: float | None = None,
    dbscan_r_method: GraphCCRadiusMethod = "knee",
    dbscan_k_neighbors: int = 4,
    dbscan_min_samples: int = 5,
    dbscan_min_cluster_size: int = 1,
    # OPTICS-only knobs.
    optics_min_samples: int = 5,
    optics_xi: float = 0.05,
    optics_min_cluster_size: int | None = None,
    # Affinity-propagation-only knobs.
    affinity_damping: float = 0.5,
    affinity_preference: float | None = None,
    affinity_max_iter: int = 200,
    affinity_convergence_iter: int = 15,
    # MeanShift-only knobs.
    meanshift_bandwidth: float | None = None,
    meanshift_quantile: float = 0.3,
    meanshift_bin_seeding: bool = False,
    meanshift_cluster_all: bool = True,
    # K-means-only knobs.
    n_iter: int = 25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the requested manifold-aware clustering algorithm.

    - ``kmeans`` requires ``k`` (Lloyd's algorithm).
    - ``hdbscan`` ignores ``k`` and discovers the cluster count from
      density structure.
    - ``graph_cc`` ignores ``k`` and either uses an explicit
      ``graph_cc_radius`` or auto-determines it via ``graph_cc_r_method``
      (Kneedle k-NN knee, or MST max-gap).
    - ``leiden`` ignores ``k`` and partitions a manifold-distance graph
      (k-NN / mutual k-NN / radius) by modularity or CPM.
    - ``dbscan`` / ``optics`` ignore ``k`` (density-based on the precomputed
      manifold-distance matrix; ``dbscan`` auto-picks ``eps`` like graph_cc).
    - ``affinity_propagation`` ignores ``k`` (exemplar-based on a precomputed
      manifold-similarity matrix).
    - ``meanshift`` ignores ``k`` (mode-seeking; Euclidean on the raw
      embedding coordinates — *not* manifold-aware, see :func:`manifold_meanshift`).

    Keeps S1 / S2 callers parameterised on ``algorithm`` without
    branching at every call site.
    """
    if algorithm == "kmeans":
        if k is None:
            raise ValueError("cluster_embeddings(algorithm='kmeans') requires k; got None.")
        return manifold_kmeans(embeddings, k=k, manifold=manifold, n_iter=n_iter, seed=seed)
    if algorithm == "hdbscan":
        return manifold_hdbscan(
            embeddings,
            manifold=manifold,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
        )
    if algorithm == "graph_cc":
        return manifold_graph_cc(
            embeddings,
            manifold=manifold,
            radius=graph_cc_radius,
            r_method=graph_cc_r_method,
            k_neighbors=graph_cc_k_neighbors,
            min_cluster_size=graph_cc_min_cluster_size,
        )
    if algorithm == "leiden":
        return manifold_leiden(
            embeddings,
            manifold=manifold,
            graph_method=leiden_graph_method,
            k_neighbors=leiden_k_neighbors,
            radius=leiden_radius,
            r_method=leiden_r_method,
            quality=leiden_quality,
            resolution=leiden_resolution,
            min_cluster_size=leiden_min_cluster_size,
            seed=seed,
            n_iterations=leiden_n_iterations,
        )
    if algorithm == "dbscan":
        return manifold_dbscan(
            embeddings,
            manifold=manifold,
            eps=dbscan_eps,
            r_method=dbscan_r_method,
            k_neighbors=dbscan_k_neighbors,
            min_samples=dbscan_min_samples,
            min_cluster_size=dbscan_min_cluster_size,
        )
    if algorithm == "optics":
        return manifold_optics(
            embeddings,
            manifold=manifold,
            min_samples=optics_min_samples,
            xi=optics_xi,
            min_cluster_size=optics_min_cluster_size,
        )
    if algorithm == "affinity_propagation":
        return manifold_affinity_propagation(
            embeddings,
            manifold=manifold,
            damping=affinity_damping,
            preference=affinity_preference,
            max_iter=affinity_max_iter,
            convergence_iter=affinity_convergence_iter,
            seed=seed,
        )
    if algorithm == "meanshift":
        return manifold_meanshift(
            embeddings,
            manifold=manifold,
            bandwidth=meanshift_bandwidth,
            quantile=meanshift_quantile,
            bin_seeding=meanshift_bin_seeding,
            cluster_all=meanshift_cluster_all,
            seed=seed,
        )
    raise ValueError(
        f"Unknown clustering algorithm {algorithm!r}; expected one of 'kmeans', 'hdbscan', "
        "'graph_cc', 'leiden', 'dbscan', 'optics', 'affinity_propagation', 'meanshift'."
    )


def _auto_temperature(centroids: torch.Tensor, manifold: ManifoldHead) -> float:
    """Softmax temperature set to the characteristic inter-centroid distance.

    With ``T = 1`` the softmax over negative manifold distances saturates to
    ~1.0 for *every* document whenever the clusters are well separated — e.g.
    after a UMAP / t-SNE reduction spreads them far apart — leaving the
    confidence column with no usable spread to rank on (every document ties at
    1.0, so a bottom-quantile cut is arbitrary). Scaling ``T`` to the median
    pairwise centroid distance rescales the logits so the peak softmax lands in
    ``[1/C, 1]`` with real gradient: boundary documents score distinctly lower
    than core ones. The *ordering* is unchanged (peak softmax is monotone in the
    nearest-vs-next distance gap for a fixed ``T``), so this is a rescaling of
    the ordinal signal, not a change of what "more confident" means.

    Returns ``1.0`` (a no-op) when there are fewer than two centroids, or when
    the centroids are coincident (degenerate median distance).
    """
    c = int(centroids.shape[0])
    if c < 2:
        return 1.0
    dmat = manifold.pairwise_dist(centroids, centroids)  # [C, C]
    iu = torch.triu_indices(c, c, offset=1)
    pdist = dmat[iu[0], iu[1]]
    med = float(pdist.median().item()) if int(pdist.numel()) else 0.0
    return med if med > 1e-9 else 1.0


def _resolve_temperature(
    temperature: float | Literal["auto"],
    centroids: torch.Tensor,
    manifold: ManifoldHead,
) -> float:
    """Coerce the ``temperature`` argument to a concrete positive scale.

    ``"auto"`` defers to :func:`_auto_temperature`; a float is validated ``> 0``.
    """
    if isinstance(temperature, str):
        if temperature != "auto":
            raise ValueError(
                f"temperature must be a positive float or 'auto'; got {temperature!r}."
            )
        return _auto_temperature(centroids, manifold)
    t = float(temperature)
    if t <= 0.0:
        raise ValueError(f"temperature must be > 0; got {temperature!r}.")
    return t


def cluster_scores(
    embeddings: torch.Tensor,
    centroids: torch.Tensor,
    manifold: ManifoldHead,
    *,
    temperature: float | Literal["auto"] = 1.0,
) -> torch.Tensor:
    """Soft-assignment matrix ``[N, C]``: softmax over negative manifold
    distances from each document to each centroid.

    This is the row-normalized geometry the ordinal confidence reads (as its
    per-row peak), and what any margin-style reading (top1 - top2) needs in
    order to rank documents by how contested their assignment is. Column ``j``
    corresponds to the cluster whose centroid is ``centroids[j]``.
    ``temperature`` matches :func:`cluster_confidence` — pass ``"auto"`` to
    scale it to the inter-centroid separation (see :func:`_auto_temperature`).
    """
    temp = _resolve_temperature(temperature, centroids, manifold)
    d = manifold.pairwise_dist(embeddings, centroids)  # [N, C]
    return torch.nn.functional.softmax(-d / temp, dim=-1)


def cluster_confidence(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    manifold: ManifoldHead,
    *,
    temperature: float | Literal["auto"] = 1.0,
) -> list[float | None]:
    """Per-point ordinal confidence for an unsupervised clustering.

    Every algorithm in :func:`cluster_embeddings` returns
    ``(labels, centroids)`` where the centroids are cluster representatives
    on the manifold — true Fréchet-mean centroids for ``kmeans`` /
    ``meanshift`` and on-manifold *medoids* for the density / graph
    algorithms (``hdbscan``, ``graph_cc``, ``leiden``, ``dbscan``,
    ``optics``, ``affinity_propagation``). This turns that geometry into a
    single per-document confidence signal so the unsupervised (S1) path
    stops handing downstream consumers a column of ``None``.

    The signal is the **peak of the softmax over negative manifold distances
    to the centroids** — the *same form* as the nearest-prototype confidence
    :func:`assign_to_prototypes` produces for S2-S5, so a consumer reads
    "confidence" the same way (higher = more certain) in every scenario.
    Concretely, for a document with distances ``d_1..d_C`` to the ``C``
    centroids, ``confidence = max_j softmax(-d / temperature)_j``. Note the
    *scale* is not identical across scenarios: this path divides by
    ``confidence_temperature`` (default ``"auto"`` — the median inter-centroid
    distance), whereas ``assign_to_prototypes`` fixes the temperature at 1.0,
    so the numbers are comparable *within* a run, not calibrated across
    scenarios. A document that sits
    much closer to one centroid than the rest scores near ``1``; one that is
    roughly equidistant from several scores near ``1/C``. That is a
    medoid-margin reading on the centroid methods and a boundary ("border")
    reading on the density / graph methods — a point on a cluster's edge is
    closer to being reassigned, so its peak softmax is lower.

    **Noise points** (``labels[i] == -1``, emitted by every density / graph
    algorithm for low-density regions) are the algorithm's own statement that
    the document does not belong to any cluster, so they are pinned to
    ``0.0`` regardless of geometry — the honest floor of the scale.

    The score is deliberately *ordinal*, not a calibrated probability:
    softmax temperature is a free scale and the medoid geometry is only a
    proxy for HDBSCAN's native membership / GLOSH scores. Temperature/Platt
    scaling and conformal coverage guarantees are a separate, heavier layer.
    What this buys today is a real, monotone, per-document signal the novelty
    gate and any review step can threshold on instead of a null.

    Args:
        embeddings: ``[N, D]`` points, in the *same* space the clustering ran
            in (i.e. the reduced Euclidean space when a reducer is active, so
            distances line up with the returned ``centroids``).
        labels: ``[N]`` integer cluster ids as returned by
            :func:`cluster_embeddings` (``-1`` = noise).
        centroids: ``[C, D]`` cluster representatives (may be empty when the
            algorithm routed everything to noise).
        manifold: Active manifold head for the clustering space — supplies
            ``pairwise_dist``.
        temperature: Softmax temperature. A positive float flattens the
            distribution as it grows (lower, more conservative confidences);
            ``1.0`` matches :func:`assign_to_prototypes`. ``"auto"`` scales the
            temperature to the inter-centroid separation so the signal has
            usable spread even when clusters are far apart (see
            :func:`_auto_temperature`) — the right default for the S1
            unsupervised path, whose distances live in a UMAP-reduced space.

    Returns:
        Length-``N`` list of confidences in ``[0, 1]``. Noise points are
        ``0.0``; when there are no centroids at all every point is ``0.0``.
        Typed ``float | None`` to slot directly into
        :attr:`~clustering.scenarios.base.ScenarioResult.confidence`, but in
        practice every entry is a concrete float.
    """
    n = int(embeddings.shape[0])
    if n == 0:
        return []

    label_ints = [int(x) for x in labels.tolist()]

    # No surviving clusters — the algorithm called the whole corpus noise.
    if int(centroids.shape[0]) == 0:
        return [0.0] * n

    probs = cluster_scores(embeddings, centroids, manifold, temperature=temperature)
    peak = probs.amax(dim=-1)  # [N]
    peak_vals = [float(p) for p in peak.tolist()]

    return [0.0 if label_ints[i] < 0 else peak_vals[i] for i in range(n)]


def emergent_bucket_k_neighbors(
    n: int,
    configured: int,
    *,
    graph_method: LeidenGraphMethod = "knn",
) -> int:
    """Graph degree to use when clustering an *emergent bucket* of ``n`` docs.

    ``configured`` is the corpus-level ``leiden_k_neighbors``. It is chosen for
    the size of a whole corpus, but the unknown bucket that S2/S3 hand to the
    clusterer is a far smaller sub-problem, and at that scale the corpus-level
    value meets or exceeds ``n``. The k-NN graph is then *complete*, so the
    neighbourhood structure — the only thing the ``k`` in k-NN contributes —
    is gone and the partition rests entirely on the RBF edge weights. Those
    are normalised by the median edge distance, so a bucket whose groups are
    only weakly separated ends up with near-uniform weights, and modularity's
    optimum over a near-uniform complete graph is a single community.

    This is an empirical effect, not an identity: a complete graph over two
    *strongly* separated groups still splits, because the weight contrast
    survives the normalisation. Weak separation is the case that matters here
    — the bucket holds what the assignment gate could not confidently place,
    which selects for exactly that. Measured leave-one-class-out on four
    internal corpora at the shipped config's degree of 25, the unknown bucket
    came back as one cluster on 16 of 24 splits.

    Note the collapse is not avoided by clamping to ``n - 1``: that is what
    :func:`manifold_leiden` already does internally, so an ``n - 1`` ceiling
    reproduces the uncapped behaviour exactly (measured: zero differences over
    the same 24 splits). The degree has to come *down*, and sub-linearly — a
    linear rule pins to its own floor at this scale.

    The result never exceeds ``configured``, so a config that already asks for
    a small degree passes through untouched — including values below the floor
    of 2 that the ``sqrt`` term carries for tiny buckets.

    Only ``graph_method="knn"`` is scaled, deliberately:

    - ``"mutual_knn"`` requires *reciprocal* membership, so the edge count
      falls off far faster than the degree does; scaling it over-fragments
      buckets that the unscaled degree partitions correctly.
    - ``"radius"`` does not build a k-NN graph at all, but ``k_neighbors`` is
      not inert there either — with ``leiden_radius=None`` it feeds the
      auto-radius knee heuristic, so scaling it silently retunes the radius.

    Neither mode is covered by the measurement above and both have a plausible
    mechanism for harm, so both pass through unchanged.
    """
    if graph_method != "knn":
        return configured
    return min(configured, max(2, math.isqrt(max(n, 0))))


def cluster_emergent_bucket(
    embeddings: torch.Tensor,
    *,
    scenario: ScenarioConfig,
    manifold: ManifoldHead,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster the unknown/emergent bucket shared by S2 and S3.

    Both scenarios split their input into documents that matched a known
    prototype and documents that did not, then partition the leftovers into
    ``unknown_<i>`` clusters. This is that second step, in one place: it honours
    ``scenario.cluster_algorithm`` and forwards the algorithm knobs, with the
    graph degree scaled to the bucket by :func:`emergent_bucket_k_neighbors`.

    ``k`` is still derived as ``max(2, min(8, n))``. Only ``kmeans`` consumes
    it — every other algorithm derives its own cluster count — and in the 2..8
    range that formula degenerates to ``k == n``, i.e. one cluster per document.
    That is a real defect on a ``kmeans`` config and it is deliberately *not*
    addressed here: no alternative ``k`` rule has yet measured as an improvement
    across the full corpus set, and guessing one would trade a known bad value
    for an unmeasured one.
    """
    sc = scenario
    n = int(embeddings.shape[0])
    return cluster_embeddings(
        embeddings,
        manifold=manifold,
        algorithm=sc.cluster_algorithm,
        k=max(2, min(8, n)),
        seed=seed,
        min_cluster_size=sc.hdbscan_min_cluster_size,
        min_samples=sc.hdbscan_min_samples,
        cluster_selection_epsilon=sc.hdbscan_cluster_selection_epsilon,
        cluster_selection_method=sc.hdbscan_cluster_selection_method,
        allow_single_cluster=sc.hdbscan_allow_single_cluster,
        graph_cc_radius=sc.graph_cc_radius,
        graph_cc_r_method=sc.graph_cc_r_method,
        graph_cc_k_neighbors=sc.graph_cc_k_neighbors,
        graph_cc_min_cluster_size=sc.graph_cc_min_cluster_size,
        leiden_graph_method=sc.leiden_graph_method,
        leiden_k_neighbors=emergent_bucket_k_neighbors(
            n, sc.leiden_k_neighbors, graph_method=sc.leiden_graph_method
        ),
        leiden_radius=sc.leiden_radius,
        leiden_r_method=sc.leiden_r_method,
        leiden_quality=sc.leiden_quality,
        leiden_resolution=sc.leiden_resolution,
        leiden_min_cluster_size=sc.leiden_min_cluster_size,
        leiden_n_iterations=sc.leiden_n_iterations,
        dbscan_eps=sc.dbscan_eps,
        dbscan_r_method=sc.dbscan_r_method,
        dbscan_k_neighbors=sc.dbscan_k_neighbors,
        dbscan_min_samples=sc.dbscan_min_samples,
        dbscan_min_cluster_size=sc.dbscan_min_cluster_size,
        optics_min_samples=sc.optics_min_samples,
        optics_xi=sc.optics_xi,
        optics_min_cluster_size=sc.optics_min_cluster_size,
        affinity_damping=sc.affinity_damping,
        affinity_preference=sc.affinity_preference,
        affinity_max_iter=sc.affinity_max_iter,
        affinity_convergence_iter=sc.affinity_convergence_iter,
        meanshift_bandwidth=sc.meanshift_bandwidth,
        meanshift_quantile=sc.meanshift_quantile,
        meanshift_bin_seeding=sc.meanshift_bin_seeding,
        meanshift_cluster_all=sc.meanshift_cluster_all,
    )


@dataclass
class AssignmentResult:
    """Output of :func:`assign_to_prototypes`.

    Carries the per-document tensors (``labels`` / ``distances`` /
    ``confidence`` / ``probs``), the *effective* thresholds that were actually
    applied (distinct from the user-supplied config values whenever quantile
    auto-calibration kicks in), and — when a
    :class:`~clustering.calibration.Calibrator` is supplied — the
    ``calibrated_confidence`` and the per-document ``abstain`` flag that feeds
    a human review queue.
    """

    labels: torch.Tensor  # [N] int (-1 = unassigned)
    distances: torch.Tensor  # [N] nearest-prototype manifold distance
    confidence: torch.Tensor  # [N] softmax peak in [0, 1]
    probs: torch.Tensor  # [N, K] softmax over -distance
    effective_threshold: float | None = None
    effective_confidence_threshold: float | None = None
    # ── Calibration + abstain ────────────────────────────────────────────
    # ``calibrated_confidence`` equals ``confidence`` when no calibrator was
    # supplied (i.e. the raw ordinal signal), so a consumer can read this field
    # unconditionally. ``abstain`` flags documents the gate is not confident
    # enough to auto-accept — route them to review. ``calibration`` is the
    # fitted calibrator's provenance dict, or ``None``.
    calibrated_confidence: torch.Tensor | None = None
    abstain: torch.Tensor | None = None
    calibration: dict[str, Any] | None = None


def assign_to_prototypes(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    manifold: ManifoldHead,
    *,
    threshold: float | None = None,
    threshold_confidence: float | None = None,
    threshold_quantile: float | None = None,
    calibrator: Calibrator | None = None,
    abstain_threshold: float | None = None,
) -> AssignmentResult:
    """Nearest-prototype classification with three composable unknown-bucket gates.

    Gating modes (all optional, all compose via OR — a doc is unassigned
    if **any** active gate flags it):

    - ``threshold`` — raw manifold distance cutoff. Manifold-unit-dependent.
    - ``threshold_confidence`` — softmax-over-prototypes confidence floor in
      ``[0, 1]``. Manifold-independent (operates on the softmax of
      ``-distance``, scale-invariant per row).
    - ``threshold_quantile`` — if set in ``(0, 1)``, the distance threshold is
      auto-calibrated to the ``q``-quantile of the *empirical* nearest-
      prototype distance distribution over the supplied ``embeddings``.
      ``q=0.8`` keeps the closest 80 % of docs as known. Composes with
      ``threshold_confidence``: both gates are OR-ed.
      (If ``threshold`` is *also* passed, ``threshold_quantile`` overrides
      it — the quantile is treated as a higher-priority calibration.)

    Orthogonal to all three is the **abstain** decision. The gates above decide
    *novelty* — whether a document belongs to a known class at all — and change
    its label to ``-1``. Abstention decides *review*: whether a human should
    look at the assignment before it is trusted. It never changes the predicted
    label, so a document can be confidently assigned and still be flagged, or
    routed to the unknown bucket without being flagged.

    Args:
        embeddings: ``[N, D]`` on-manifold query points.
        prototypes: ``[K, D]`` on-manifold class prototypes.
        manifold: Active manifold head.
        threshold: Absolute distance cutoff (see above).
        threshold_confidence: Confidence floor in ``[0, 1]``.
        threshold_quantile: Distance quantile in ``(0, 1)``.
        calibrator: Fitted :class:`~clustering.calibration.Calibrator` used to
            rescale the reported confidence and (if it carries a conformal
            threshold) decide abstention. ``None`` reports the raw ordinal
            confidence.
        abstain_threshold: Absolute floor on the (calibrated) confidence, OR-ed
            with any conformal gate the calibrator carries. Usable without a
            calibrator, which is how the unlabeled scenarios abstain.

    Returns:
        :class:`AssignmentResult`. ``effective_threshold`` is the distance
        cutoff that was actually applied (after any quantile calibration);
        ``effective_confidence_threshold`` echoes the user-supplied value or
        ``None``. Both are persisted into scenario metadata so the
        comparison view can recover the operating point.
    """
    d = manifold.pairwise_dist(embeddings, prototypes)  # [N, K]
    logits = -d
    probs = torch.nn.functional.softmax(logits, dim=-1)
    distances = d.amin(dim=-1)
    labels = d.argmin(dim=-1)
    confidence = probs.amax(dim=-1)

    # ── Auto-calibrate distance threshold via quantile, if requested ─────
    eff_dist = threshold
    if threshold_quantile is not None:
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError(f"threshold_quantile must be in (0, 1); got {threshold_quantile!r}.")
        d_np = distances.detach().numpy() if hasattr(distances, "numpy") else np.asarray(distances)
        eff_dist = float(np.quantile(d_np, threshold_quantile))

    # Validate confidence threshold range up-front so callers get a clear error.
    eff_conf = threshold_confidence
    if eff_conf is not None and not 0.0 <= eff_conf <= 1.0:
        raise ValueError(f"threshold_confidence must be in [0, 1]; got {eff_conf!r}.")

    # ── Compose gating masks (OR over active modes) ──────────────────────
    unassigned: torch.Tensor | None = None
    if eff_dist is not None:
        mask_d = distances > eff_dist
        unassigned = mask_d if unassigned is None else _bool_or(unassigned, mask_d)
    if eff_conf is not None:
        mask_c = confidence < eff_conf
        unassigned = mask_c if unassigned is None else _bool_or(unassigned, mask_c)

    if unassigned is not None:
        labels = torch.where(unassigned, torch.full_like(labels, -1), labels)

    # ── Calibrate the reported confidence, then decide abstention ─────────
    # Deliberately after the novelty gates and deliberately not feeding back
    # into ``labels``: this is the review decision, not the routing decision.
    # With no calibrator the calibrated confidence is just the ordinal softmax
    # peak, so the field is always readable, and an ``abstain_threshold`` still
    # applies as a plain floor on it.
    if calibrator is not None:
        calibrated_confidence, abstain = calibrator.apply(logits)
        calibration = calibrator.as_dict()
    else:
        calibrated_confidence = confidence.clone() if hasattr(confidence, "clone") else confidence
        abstain = torch.zeros_like(labels, dtype=torch.bool)
        calibration = None
    if abstain_threshold is not None:
        abstain = _bool_or(abstain, calibrated_confidence < abstain_threshold)

    return AssignmentResult(
        labels=labels,
        distances=distances,
        confidence=confidence,
        probs=probs,
        effective_threshold=eff_dist,
        effective_confidence_threshold=eff_conf,
        calibrated_confidence=calibrated_confidence,
        abstain=abstain,
        calibration=calibration,
    )


def _bool_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """OR over two boolean tensors. ``a | b`` works on real torch; for the
    sandbox stub we fall back to ``(a + b) > 0`` which has identical
    semantics on boolean / 0-1 integer arrays."""
    try:
        return a | b
    except TypeError:
        return (a + b) > 0
