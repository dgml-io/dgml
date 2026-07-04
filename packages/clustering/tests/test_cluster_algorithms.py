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
    cluster_embeddings,
    manifold_affinity_propagation,
    manifold_dbscan,
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
