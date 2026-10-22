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

"""Tests for :func:`clustering.scenarios.clustering.cluster_confidence` and
the S1 confidence signal it feeds.

The signal is ordinal — peak softmax over negative manifold distances to the
returned centroids/medoids, with noise (-1) pinned to 0.0 — so the tests
assert range, monotonicity, and the noise/empty edge cases rather than exact
values, plus that S1's ``ScenarioResult.confidence`` is now populated with
real floats instead of ``None``.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from clustering.config.schema import (
    Auto,
    ClusterAlgorithm,
    Config,
    ManifoldConfig,
    ManifoldName,
    ScenarioConfig,
    ScenarioName,
)
from clustering.data.datasets import DocumentDataset, DocumentRecord
from clustering.manifolds import build_manifold
from clustering.scenarios import build_scenario
from clustering.scenarios.clustering import (
    cluster_confidence,
    cluster_embeddings,
    cluster_scores,
)
from PIL import Image
from pydantic import ValidationError

_DIM = 8


def _euclidean(dim: int = _DIM) -> Any:
    return build_manifold(ManifoldConfig(name=ManifoldName.EUCLIDEAN, dim=dim, curvature=0.0))


def _three_blobs(dim: int = _DIM, per: int = 10) -> torch.Tensor:
    """Three tight, well-separated Gaussian blobs in Euclidean space."""
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([0.0, 10.0, 20.0])
    return torch.cat([c + 0.1 * torch.randn(per, dim, generator=g) for c in centers], dim=0)


# ── cluster_confidence: contract & range ───────────────────────────────────
def test_confidence_matches_length_and_range() -> None:
    emb = _three_blobs()
    manifold = _euclidean()
    labels, centroids = cluster_embeddings(
        emb, manifold=manifold, algorithm=ClusterAlgorithm.KMEANS, k=3
    )

    conf = cluster_confidence(emb, labels, centroids, manifold)

    assert len(conf) == emb.shape[0]
    assert all(c is not None for c in conf)  # no more null column
    assert all(0.0 <= float(c) <= 1.0 for c in conf)  # type: ignore[arg-type]


def test_well_separated_blobs_are_high_confidence() -> None:
    emb = _three_blobs()
    manifold = _euclidean()
    labels, centroids = cluster_embeddings(
        emb, manifold=manifold, algorithm=ClusterAlgorithm.KMEANS, k=3
    )

    conf = cluster_confidence(emb, labels, centroids, manifold)

    # Cleanly separable blobs → every point sits far nearer its own centroid.
    assert min(float(c) for c in conf) > 0.9  # type: ignore[arg-type]


def test_noise_points_are_zero() -> None:
    # Two clusters plus one explicit noise point (label -1). The noise entry
    # must be pinned to 0.0 regardless of geometry; the rest stay positive.
    manifold = _euclidean(dim=2)
    emb = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0], [5.0, 9.0]], dtype=torch.float32
    )
    labels = torch.tensor([0, 0, 1, 1, -1], dtype=torch.long)
    centroids = torch.tensor([[0.05, 0.0], [10.05, 0.0]], dtype=torch.float32)

    conf = cluster_confidence(emb, labels, centroids, manifold)

    assert conf[4] == 0.0
    assert all(float(c) > 0.0 for c in conf[:4])  # type: ignore[arg-type]


def test_all_noise_empty_centroids_all_zero() -> None:
    manifold = _euclidean(dim=2)
    emb = torch.zeros((4, 2))
    labels = torch.full((4,), -1, dtype=torch.long)
    centroids = torch.zeros((0, 2))  # algorithm found no surviving cluster

    conf = cluster_confidence(emb, labels, centroids, manifold)

    assert conf == [0.0, 0.0, 0.0, 0.0]


def test_empty_input_returns_empty() -> None:
    manifold = _euclidean(dim=2)
    conf = cluster_confidence(
        torch.zeros((0, 2)), torch.zeros((0,), dtype=torch.long), torch.zeros((0, 2)), manifold
    )
    assert conf == []


def test_single_cluster_is_certain() -> None:
    # One centroid ⇒ softmax over a single logit ⇒ 1.0 for every assigned point.
    manifold = _euclidean(dim=2)
    emb = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=torch.float32)
    labels = torch.zeros((3,), dtype=torch.long)
    centroids = torch.tensor([[1.0, 0.3]], dtype=torch.float32)

    conf = cluster_confidence(emb, labels, centroids, manifold)

    assert conf == [1.0, 1.0, 1.0]


def test_higher_temperature_is_more_conservative() -> None:
    # A borderline point (equidistant-ish from two centroids) should score
    # lower as temperature rises (the softmax flattens toward 1/C).
    manifold = _euclidean(dim=2)
    emb = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    labels = torch.zeros((1,), dtype=torch.long)
    centroids = torch.tensor([[1.0, 0.0], [-1.2, 0.0]], dtype=torch.float32)

    cold = float(cluster_confidence(emb, labels, centroids, manifold, temperature=0.5)[0])  # type: ignore[arg-type]
    warm = float(cluster_confidence(emb, labels, centroids, manifold, temperature=5.0)[0])  # type: ignore[arg-type]

    assert cold > warm


def test_non_positive_temperature_raises() -> None:
    manifold = _euclidean(dim=2)
    emb = torch.zeros((1, 2))
    labels = torch.zeros((1,), dtype=torch.long)
    centroids = torch.zeros((1, 2))
    with pytest.raises(ValueError, match="temperature"):
        cluster_confidence(emb, labels, centroids, manifold, temperature=0.0)


def test_invalid_temperature_string_raises() -> None:
    manifold = _euclidean(dim=2)
    emb = torch.zeros((1, 2))
    labels = torch.zeros((1,), dtype=torch.long)
    centroids = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="temperature"):
        cluster_confidence(emb, labels, centroids, manifold, temperature="nope")  # type: ignore[arg-type]


# ── auto temperature: breaks the saturation that flattens the signal ────────
def test_auto_temperature_spreads_well_separated_blobs() -> None:
    # With T=1.0 every well-separated point pins at ~1.0 (no spread), so a
    # bottom-quantile cut over the column is arbitrary. "auto" rescales to the
    # inter-centroid distance → a real spread appears.
    emb = _three_blobs()
    manifold = _euclidean()
    labels, centroids = cluster_embeddings(
        emb, manifold=manifold, algorithm=ClusterAlgorithm.KMEANS, k=3
    )

    raw = [float(c) for c in cluster_confidence(emb, labels, centroids, manifold)]  # type: ignore[arg-type]
    auto = [
        float(c)  # type: ignore[arg-type]
        for c in cluster_confidence(emb, labels, centroids, manifold, temperature=Auto.AUTO)
    ]

    # T=1.0 saturates: no usable spread.
    assert max(raw) - min(raw) < 1e-3
    # auto gives a consumer something to rank on.
    assert max(auto) - min(auto) > 0.05
    assert all(0.0 <= c <= 1.0 for c in auto)


def test_auto_temperature_preserves_ordering() -> None:
    # Rescaling the temperature must not reorder documents by confidence: a
    # core point stays more confident than a boundary point under "auto".
    manifold = _euclidean(dim=2)
    emb = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)  # core, boundary
    labels = torch.zeros((2,), dtype=torch.long)
    centroids = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float32)

    auto = [
        float(c)  # type: ignore[arg-type]
        for c in cluster_confidence(emb, labels, centroids, manifold, temperature=Auto.AUTO)
    ]
    assert auto[0] > auto[1]  # the point on top of its centroid is more confident


def test_auto_temperature_no_ops_on_coincident_centroids() -> None:
    # Degenerate geometry: identical centroids ⇒ median pairwise distance 0,
    # which would be a divide-by-zero scale. "auto" must fall back to 1.0 and
    # still return a valid distribution (a tie, since the logits are equal).
    manifold = _euclidean(dim=2)
    emb = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    centroids = torch.zeros((2, 2), dtype=torch.float32)

    scores = cluster_scores(emb, centroids, manifold, temperature=Auto.AUTO)

    assert torch.allclose(scores, torch.full((1, 2), 0.5), atol=1e-6)


def test_cluster_scores_shape_and_normalization() -> None:
    emb = _three_blobs(per=5)
    manifold = _euclidean()
    _, centroids = cluster_embeddings(
        emb, manifold=manifold, algorithm=ClusterAlgorithm.KMEANS, k=3
    )

    scores = cluster_scores(emb, centroids, manifold, temperature=Auto.AUTO)
    assert tuple(scores.shape) == (emb.shape[0], centroids.shape[0])
    # Rows are softmax distributions.
    assert torch.allclose(scores.sum(dim=-1), torch.ones(emb.shape[0]), atol=1e-5)


# ── config validation ──────────────────────────────────────────────────────
def test_config_rejects_non_positive_confidence_temperature() -> None:
    # Caught at config-validation time, not partway through a run.
    with pytest.raises(ValidationError, match="confidence_temperature"):
        ScenarioConfig(name=ScenarioName.S1, k_clusters=2, confidence_temperature=0.0)


def test_config_accepts_auto_and_positive_float() -> None:
    assert ScenarioConfig(name=ScenarioName.S1, k_clusters=2).confidence_temperature == "auto"
    pinned = ScenarioConfig(name=ScenarioName.S1, k_clusters=2, confidence_temperature=2.5)
    assert pinned.confidence_temperature == 2.5


# ── S1 integration: ScenarioResult.confidence is populated ──────────────────
class _InMemoryDataset(DocumentDataset):
    """Tiny corpus of two well-separated text "classes"."""

    def __init__(self, labels: list[str | None]) -> None:
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> DocumentRecord:
        label = self._labels[index]
        return DocumentRecord(
            doc_id=f"doc_{index}",
            label=label,
            image=Image.new("RGB", (8, 8), color=(index * 9 % 255, 0, 0)),
            text=f"{label or 'unlabeled'} document number {index}",
            thumbnail_path=None,
        )


def _s1_config(algorithm: str = "kmeans") -> Config:
    raw: dict[str, Any] = {
        "scenario": {"name": "s1", "k_clusters": 2, "cluster_algorithm": algorithm},
        "encoder_text": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "encoder_image": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "fusion": {"name": "late_concat", "output_dim": 2 * _DIM},
        "manifold": {"name": "euclidean", "dim": 2 * _DIM},
        "training": {"epochs": 0},
        "logger": {"name": "none"},
        "corpus": {"root": "."},
        "device": "cpu",
        "seed": 0,
    }
    return Config.model_validate(raw)


def test_s1_result_confidence_is_populated() -> None:
    labels: list[str | None] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    result = build_scenario(_s1_config()).fit_predict(_InMemoryDataset(labels))

    # One confidence per document, all real floats in range — the old S1 path
    # returned [None] * N here.
    assert len(result.confidence) == len(labels)
    assert all(isinstance(c, float) for c in result.confidence)
    assert all(0.0 <= float(c) <= 1.0 for c in result.confidence)  # type: ignore[arg-type]


def test_s1_populates_scores_and_class_names() -> None:
    labels: list[str | None] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    result = build_scenario(_s1_config()).fit_predict(_InMemoryDataset(labels))

    # S1 used to emit no scores; it now exposes the soft-assignment matrix, so a
    # consumer can read the top1 - top2 margin and not just the peak.
    assert result.scores is not None
    assert result.scores.shape[0] == len(labels)
    assert result.class_names is not None
    assert result.scores.shape[1] == len(result.class_names)
    # Column names line up with the ``cluster_N`` predictions.
    assert all(name.startswith("cluster_") for name in result.class_names)


def test_s1_confidence_aligns_with_predictions() -> None:
    labels: list[str | None] = [None] * 8
    result = build_scenario(_s1_config(algorithm="hdbscan")).fit_predict(_InMemoryDataset(labels))

    # Any document HDBSCAN routed to the noise bucket must carry 0.0 confidence.
    for pred, conf in zip(result.predictions, result.confidence, strict=True):
        if pred == "cluster_noise":
            assert conf == 0.0
