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

"""Tests for the no-``k`` scikit-learn clusterers added to
:mod:`clustering.scenarios.clustering`.

These algorithms (DBSCAN, OPTICS, Affinity Propagation, MeanShift) discover
the cluster count themselves, so the tests assert the shared
``(labels, centroids)`` contract and that three well-separated Gaussian blobs
are recovered as more than one cluster — rather than pinning an exact count,
which is sensitive to each algorithm's defaults.
"""

from __future__ import annotations

import pytest
import torch
from clustering.config.schema import ManifoldConfig
from clustering.manifolds import build_manifold
from clustering.scenarios.clustering import (
    ClusterAlgorithm,
    _radius_knee,
    cluster_embeddings,
    manifold_affinity_propagation,
    manifold_dbscan,
    manifold_leiden,
    manifold_meanshift,
    manifold_optics,
)

_ALGORITHMS: list[ClusterAlgorithm] = ["dbscan", "optics", "affinity_propagation", "meanshift"]


def _three_blobs(dim: int = 8, per: int = 12) -> torch.Tensor:
    """Three tight, well-separated Gaussian blobs in Euclidean space."""
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([0.0, 10.0, 20.0])
    blobs = []
    for c in centers:
        pt = c + 0.1 * torch.randn(per, dim, generator=g)
        blobs.append(pt)
    return torch.cat(blobs, dim=0)


def _assert_contract(labels: torch.Tensor, centroids: torch.Tensor, n: int, dim: int) -> None:
    assert labels.shape == (n,)
    assert labels.dtype == torch.long
    n_clusters = int(labels.max().item()) + 1 if labels.numel() and labels.max() >= 0 else 0
    # Labels are contiguous 0..C-1 (plus -1 noise); centroid count matches.
    non_noise = sorted({int(x) for x in labels.tolist() if int(x) >= 0})
    assert non_noise == list(range(len(non_noise)))
    assert centroids.shape == (len(non_noise), dim)
    assert n_clusters == len(non_noise)


@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_discovers_multiple_clusters(algorithm: ClusterAlgorithm) -> None:
    dim = 8
    emb = _three_blobs(dim=dim)
    n = emb.shape[0]
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))

    labels, centroids = cluster_embeddings(emb, manifold=manifold, algorithm=algorithm, seed=0)

    _assert_contract(labels, centroids, n, dim)
    non_noise = {int(x) for x in labels.tolist() if int(x) >= 0}
    assert len(non_noise) >= 2, f"{algorithm} collapsed separable blobs into {non_noise}"


@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_empty_input(algorithm: ClusterAlgorithm) -> None:
    dim = 4
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    labels, centroids = cluster_embeddings(
        torch.zeros((0, dim)), manifold=manifold, algorithm=algorithm
    )
    assert labels.shape == (0,)
    assert centroids.shape == (0, dim)


@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_single_point(algorithm: ClusterAlgorithm) -> None:
    dim = 4
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    labels, _ = cluster_embeddings(torch.ones((1, dim)), manifold=manifold, algorithm=algorithm)
    assert labels.shape == (1,)
    # A lone point is either its own cluster or noise — never crashes.
    assert int(labels[0].item()) in (-1, 0)


def test_dbscan_explicit_eps_matches_auto_contract() -> None:
    dim = 8
    emb = _three_blobs(dim=dim)
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    labels, centroids = manifold_dbscan(emb, manifold=manifold, eps=1.0, min_samples=3)
    _assert_contract(labels, centroids, emb.shape[0], dim)
    assert len({int(x) for x in labels.tolist() if int(x) >= 0}) >= 2


def test_meanshift_cluster_all_false_allows_noise() -> None:
    # With a tiny bandwidth, lone points can't reach a mode → noise (-1).
    dim = 8
    emb = _three_blobs(dim=dim)
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    labels, _ = manifold_meanshift(
        emb, manifold=manifold, bandwidth=0.05, cluster_all=False, seed=0
    )
    assert labels.shape == (emb.shape[0],)


def test_optics_and_affinity_importable_and_run() -> None:
    # Smoke test the two functions not otherwise exercised individually.
    dim = 8
    emb = _three_blobs(dim=dim)
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    for fn in (manifold_optics, manifold_affinity_propagation):
        labels, centroids = fn(emb, manifold=manifold)
        _assert_contract(labels, centroids, emb.shape[0], dim)


def test_leiden_radius_mode_still_consumes_k_neighbors() -> None:
    """``k_neighbors`` is not inert under ``graph_method="radius"``.

    With no explicit ``radius`` and the default knee heuristic it is the ``k``
    of the k-NN-distance knee, so it still sets the radius — the same role
    ``graph_cc_k_neighbors`` and ``dbscan_k_neighbors`` play for their
    algorithms, just spelled with the k-NN knob. Pinned because the docstring
    used to claim the opposite, which sends a reader to tune the wrong knob.
    """
    dim = 8
    emb = _three_blobs(dim=dim)
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    d_np = manifold.pairwise_dist(emb, emb).detach().numpy()

    r_small = _radius_knee(d_np, k=2)
    r_big = _radius_knee(d_np, k=20)
    assert r_small != r_big, "the knee must move with k, else this test proves nothing"

    # Auto-radius at k=20 must reproduce the run that is *told* that radius.
    auto, _ = manifold_leiden(emb, manifold, graph_method="radius", k_neighbors=20, seed=0)
    explicit, _ = manifold_leiden(emb, manifold, graph_method="radius", radius=r_big, seed=0)
    assert torch.equal(auto, explicit)


def test_leiden_radius_mode_ignores_k_neighbors_once_radius_is_explicit() -> None:
    """The other half of the contract: an explicit ``radius`` does make
    ``k_neighbors`` inert, so the docstring's "ignored" is right for that case
    and wrong only for the auto-radius one.
    """
    dim = 8
    emb = _three_blobs(dim=dim)
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    a, _ = manifold_leiden(emb, manifold, graph_method="radius", radius=1.0, k_neighbors=2, seed=0)
    b, _ = manifold_leiden(emb, manifold, graph_method="radius", radius=1.0, k_neighbors=20, seed=0)
    assert torch.equal(a, b)


# ── leiden k selection (opt-in silhouette-chosen graph degree) ────────────────


def test_leiden_k_selection_off_is_byte_identical() -> None:
    """The shipped default (``leiden_k_selection="none"``) must route through the
    plain single leiden run — byte-identical to calling it directly."""
    dim = 8
    emb = _three_blobs(dim=dim, per=12)
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    off, _ = cluster_embeddings(emb, manifold=manifold, algorithm="leiden", leiden_k_neighbors=25)
    ref, _ = manifold_leiden(emb, manifold, graph_method="knn", k_neighbors=25, seed=0)
    assert torch.equal(off, ref)


def test_leiden_k_selection_large_margin_never_switches() -> None:
    """A margin the silhouette gap can never clear keeps the configured k, so
    the result equals the selection-off run — the gate's conservative extreme."""
    dim = 8
    emb = _three_blobs(dim=dim, per=12)
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    off, _ = cluster_embeddings(emb, manifold=manifold, algorithm="leiden", leiden_k_neighbors=25)
    held, _ = cluster_embeddings(
        emb,
        manifold=manifold,
        algorithm="leiden",
        leiden_k_neighbors=25,
        leiden_k_selection="silhouette",
        leiden_k_selection_margin=9.9,
    )
    assert torch.equal(held, off)


def test_leiden_k_selection_only_fires_on_the_knn_graph() -> None:
    """Selection is scoped to ``graph_method="knn"`` (what it was measured on);
    a non-knn graph ignores it and runs the configured k unchanged."""
    dim = 8
    emb = _three_blobs(dim=dim, per=12)
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    ref, _ = manifold_leiden(emb, manifold, graph_method="mutual_knn", k_neighbors=25, seed=0)
    got, _ = cluster_embeddings(
        emb,
        manifold=manifold,
        algorithm="leiden",
        leiden_graph_method="mutual_knn",
        leiden_k_neighbors=25,
        leiden_k_selection="silhouette",
        leiden_k_selection_margin=0.0,
    )
    assert torch.equal(got, ref)


def test_leiden_k_selection_is_a_noop_when_the_candidate_would_not_lower_k() -> None:
    """The sparser candidate is ``max(2, (n-1)//8)``. When the configured k is
    already at or below it, there is no sparser alternative to weigh, so the
    configured run is returned unchanged (no wasted second clustering effect)."""
    dim = 8
    emb = _three_blobs(dim=dim, per=12)  # n = 36 → candidate k = 4
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    ref, _ = manifold_leiden(emb, manifold, graph_method="knn", k_neighbors=3, seed=0)
    got, _ = cluster_embeddings(
        emb,
        manifold=manifold,
        algorithm="leiden",
        leiden_k_neighbors=3,  # <= candidate 4, so selection can't lower it
        leiden_k_selection="silhouette",
        leiden_k_selection_margin=0.0,
    )
    assert torch.equal(got, ref)


def _closish_blobs(dim: int = 8, per: int = 15) -> torch.Tensor:
    """Three blobs close enough that a dense k-NN graph blurs their boundaries,
    so the sparser candidate earns a higher silhouette (n = 3*per)."""
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([0.0, 1.2, 2.4])
    return torch.cat([c + 0.35 * torch.randn(per, dim, generator=g) for c in centers], dim=0)


def test_leiden_k_selection_switches_to_the_higher_silhouette_partition() -> None:
    """The point of the feature: when the sparser candidate degree yields a
    higher-silhouette partition, the selector returns *that* partition (the
    candidate k = ``(n-1)//8``), not the configured-k one — and it is never a
    lower-silhouette choice than leaving k alone."""
    from clustering.scenarios.clustering import _partition_silhouette

    dim = 8
    emb = _closish_blobs(dim=dim, per=15)  # n = 45 → candidate k = 5
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    off, _ = cluster_embeddings(emb, manifold=manifold, algorithm="leiden", leiden_k_neighbors=35)
    candidate, _ = manifold_leiden(emb, manifold, graph_method="knn", k_neighbors=5, seed=0)
    sel, _ = cluster_embeddings(
        emb,
        manifold=manifold,
        algorithm="leiden",
        leiden_k_neighbors=35,
        leiden_k_selection="silhouette",
        leiden_k_selection_margin=0.0,
    )
    # The candidate must genuinely win on silhouette here, else the test proves nothing.
    assert _partition_silhouette(emb, candidate) > _partition_silhouette(emb, off)
    assert torch.equal(sel, candidate), "selector returns the higher-silhouette candidate partition"
    assert _partition_silhouette(emb, sel) >= _partition_silhouette(emb, off)


def test_leiden_k_selection_margin_rejects_a_negative_value() -> None:
    from clustering.config.schema import ScenarioConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="leiden_k_selection_margin must be >= 0"):
        ScenarioConfig(name="s1", leiden_k_selection_margin=-0.1)


def test_leiden_k_selection_does_not_leak_into_the_emergent_bucket() -> None:
    """The feature is scoped to the S1/fresh path. The S2/S3 emergent bucket
    already scales its own graph degree (#83's sqrt cap), so enabling
    ``leiden_k_selection`` must NOT also fire there — the bucket must cluster
    identically on or off. Guards against a future accidental threading of the
    flag through ``cluster_emergent_bucket``."""
    from clustering.config.schema import ScenarioConfig
    from clustering.scenarios.clustering import cluster_emergent_bucket

    dim = 8
    emb = _three_blobs(dim=dim, per=12)
    manifold = build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))
    base = ScenarioConfig(name="s2", cluster_algorithm="leiden", leiden_graph_method="knn")
    on = base.model_copy(
        update={"leiden_k_selection": "silhouette", "leiden_k_selection_margin": 0.0}
    )
    off_labels, _ = cluster_emergent_bucket(emb, scenario=base, manifold=manifold, seed=0)
    on_labels, _ = cluster_emergent_bucket(emb, scenario=on, manifold=manifold, seed=0)
    assert torch.equal(off_labels, on_labels)
