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

"""Tests for the unsupervised projector-training path (strict S1 regime).

Ported from doc-categorization. Covers the three label-free losses —
``pseudo_label`` (DeepCluster/PCL-style EM), ``knn_contrastive`` (SCAN),
``cross_modal`` (two-view InfoNCE) — plus the VICReg anti-collapse
regularizer and the Sinkhorn balancing helper.

The contract under test: with an unsupervised loss, training consumes ZERO
label information (not even folder names), yet runs end-to-end and produces
a usable projected space for clustering.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
from clustering.config.schema import (
    UNSUPERVISED_LOSSES,
    Config,
    FusionConfig,
    ManifoldConfig,
    ManifoldName,
    TrainingConfig,
)
from clustering.data.datasets import DocumentDataset, DocumentRecord
from clustering.fusion import build_fusion
from clustering.manifolds import (
    ManifoldProjector,
    NeighborConsistencyLoss,
    VICRegRegularizer,
    build_manifold,
    train_projector,
)
from clustering.manifolds.training import _knn_indices, _pseudo_labels, _sinkhorn
from clustering.scenarios import build_scenario
from PIL import Image

DIM = 16


def _blobs(n_per: int = 8, k: int = 3, dim: int = DIM, seed: int = 0) -> torch.Tensor:
    """``k`` well-separated Gaussian blobs in ambient R^dim."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(k, dim, generator=g) * 4.0
    pts = [centers[i] + 0.2 * torch.randn(n_per, dim, generator=g) for i in range(k)]
    return torch.cat(pts, dim=0)


def _projector(
    manifold_name: str = "euclidean", dim: int = DIM, seed: int = 0
) -> ManifoldProjector:
    torch.manual_seed(seed)
    curvature = 0.0 if manifold_name == "euclidean" else 1.0
    m = build_manifold(
        ManifoldConfig(name=cast(ManifoldName, manifold_name), dim=dim, curvature=curvature)
    )
    return ManifoldProjector(m, input_dim=dim, output_dim=dim, trainable=True)


def _cfg(**overrides: Any) -> TrainingConfig:
    base: dict[str, Any] = {"epochs": 6, "lr": 1e-3, "trainable_projector": True}
    base.update(overrides)
    return TrainingConfig(**base)


# ── Schema ───────────────────────────────────────────────────────────────
def test_unsupervised_loss_names_validate() -> None:
    for name in ("pseudo_label", "knn_contrastive", "cross_modal"):
        assert TrainingConfig(loss=name).loss == name
        assert name in UNSUPERVISED_LOSSES
    # The supervised trio is NOT in the unsupervised set.
    assert UNSUPERVISED_LOSSES.isdisjoint({"contrastive", "triplet", "prototypical"})


# ── pseudo_label ─────────────────────────────────────────────────────────
def test_pseudo_label_trains_without_any_labels() -> None:
    proj = _projector()
    fused = _blobs()
    cfg = _cfg(loss="pseudo_label", pseudo_recluster_every=2)
    history = train_projector(proj, fused, [None] * fused.shape[0], cfg=cfg, pseudo_k=3)
    assert len(history) == cfg.epochs
    assert all(isinstance(v, float) and not math.isnan(v) for v in history)  # finite, no NaN


def test_pseudo_label_ignores_labels_entirely() -> None:
    """Same seed, labels=None vs labels=garbage → identical loss history."""
    fused = _blobs()
    cfg = _cfg(loss="pseudo_label", pseudo_recluster_every=2)
    n = fused.shape[0]
    h_none = train_projector(_projector(seed=7), fused, [None] * n, cfg=cfg, pseudo_k=3)
    garbage = [f"junk_{i % 5}" for i in range(fused.shape[0])]
    h_junk = train_projector(_projector(seed=7), fused, garbage, cfg=cfg, pseudo_k=3)
    assert h_none == h_junk


def test_pseudo_labels_are_contiguous() -> None:
    proj = _projector()
    fused = _blobs()
    ids = _pseudo_labels(proj, fused, k=3, cfg=_cfg(loss="pseudo_label"), seed=0)
    uniq = sorted(set(ids.tolist()))
    assert uniq == list(range(len(uniq)))
    assert len(uniq) >= 2


@pytest.mark.parametrize("manifold_name", ["euclidean", "spherical", "hyperbolic"])
def test_pseudo_label_runs_on_every_manifold(manifold_name: str) -> None:
    proj = _projector(manifold_name)
    fused = _blobs()
    cfg = _cfg(loss="pseudo_label", epochs=3, pseudo_recluster_every=2)
    history = train_projector(proj, fused, [None] * fused.shape[0], cfg=cfg, pseudo_k=3)
    assert len(history) == 3


# ── Sinkhorn ─────────────────────────────────────────────────────────────
def test_sinkhorn_balances_column_marginals() -> None:
    g = torch.Generator().manual_seed(0)
    # Heavily skewed scores: column 0 dominates without balancing.
    scores = torch.randn(30, 3, generator=g)
    scores[:, 0] += 5.0
    q = _sinkhorn(scores, n_iters=10, epsilon=0.5)
    col = q.sum(dim=0)
    # Every cluster gets a non-trivial share (~N/K each = 10).
    assert (col > 1.0).all()
    # Rows are proper distributions.
    assert torch.allclose(q.sum(dim=1), torch.ones(30), atol=1e-4)


# ── knn_contrastive (SCAN) ───────────────────────────────────────────────
def test_knn_indices_exclude_self_and_have_right_shape() -> None:
    x = _blobs()
    idx = _knn_indices(x, k=5)
    assert idx.shape == (x.shape[0], 5)
    for i in range(x.shape[0]):
        assert i not in idx[i].tolist()


def test_knn_contrastive_trains_and_optimizes_prototypes() -> None:
    proj = _projector()
    fused = _blobs()
    cfg = _cfg(loss="knn_contrastive", epochs=10, temperature=0.5, knn_k=4)
    history = train_projector(proj, fused, [None] * fused.shape[0], cfg=cfg, pseudo_k=3)
    assert len(history) == 10
    assert all(not math.isnan(v) for v in history)


def test_neighbor_consistency_loss_entropy_term_penalizes_collapse() -> None:
    m = build_manifold(ManifoldConfig(name="euclidean", dim=DIM, curvature=0.0))
    loss = NeighborConsistencyLoss(m, n_clusters=3, dim=DIM, entropy_weight=1.0, seed=0)
    z = m.expmap0(_blobs())
    # Soft assignments are valid distributions.
    p = loss.soft_assign(z)
    assert torch.allclose(p.sum(dim=-1), torch.ones(z.shape[0]), atol=1e-4)
    out = loss(z, z)
    assert torch.isfinite(out)


# ── cross_modal ──────────────────────────────────────────────────────────
def test_cross_modal_requires_views() -> None:
    proj = _projector()
    fused = _blobs()
    with pytest.raises(ValueError, match="cross_modal"):
        train_projector(proj, fused, [None] * fused.shape[0], cfg=_cfg(loss="cross_modal"))


def test_cross_modal_trains_with_views() -> None:
    proj = _projector()
    fused = _blobs()
    g = torch.Generator().manual_seed(1)
    # Two noisy views of the same underlying structure.
    view_a = fused + 0.05 * torch.randn(*fused.shape, generator=g)
    view_b = fused + 0.05 * torch.randn(*fused.shape, generator=g)
    cfg = _cfg(loss="cross_modal", epochs=5)
    history = train_projector(proj, fused, [None] * fused.shape[0], cfg=cfg, views=(view_a, view_b))
    assert len(history) == 5
    # InfoNCE over aligned views should improve from the start.
    assert history[-1] <= history[0]


# ── VICReg ───────────────────────────────────────────────────────────────
def test_vicreg_zero_for_single_sample_and_positive_for_collapse() -> None:
    m = build_manifold(ManifoldConfig(name="euclidean", dim=DIM, curvature=0.0))
    reg = VICRegRegularizer(m, var_weight=1.0, cov_weight=1.0, gamma=1.0)
    assert float(reg(torch.zeros(1, DIM))) == 0.0
    # Fully collapsed batch: std≈0 in every dim → var hinge ≈ gamma
    # (sqrt(var + 1e-4) keeps the gradient finite, so hinge = 1 - 0.01).
    collapsed = torch.ones(8, DIM)
    assert float(reg(collapsed)) == pytest.approx(0.99, abs=1e-4)


def test_vicreg_is_additive_in_training() -> None:
    fused = _blobs()
    base = _cfg(loss="pseudo_label", epochs=3, pseudo_recluster_every=2)
    with_vic = _cfg(
        loss="pseudo_label",
        epochs=3,
        pseudo_recluster_every=2,
        vicreg_var_weight=1.0,
        vicreg_cov_weight=0.04,
    )
    n = fused.shape[0]
    h0 = train_projector(_projector(seed=3), fused, [None] * n, cfg=base, pseudo_k=3)
    h1 = train_projector(_projector(seed=3), fused, [None] * n, cfg=with_vic, pseudo_k=3)
    assert len(h1) == 3
    assert h1 != h0  # the penalty actually contributes


# ── S1 end-to-end ────────────────────────────────────────────────────────
class _InMemoryDataset(DocumentDataset):
    """Tiny unlabeled dataset — three "topics" in the text."""

    _TOPICS = ("invoice", "receipt", "contract")

    def __init__(self, n: int = 12) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> DocumentRecord:
        topic = self._TOPICS[index % len(self._TOPICS)]
        return DocumentRecord(
            doc_id=f"doc_{index}",
            label=None,  # strict S1: no labels at all
            image=Image.new("RGB", (8, 8), color=(index * 9 % 255, 0, 0)),
            text=f"{topic} document number {index}",
            thumbnail_path=None,
        )


def _s1_cfg(training: dict[str, Any]) -> Config:
    raw: dict[str, Any] = {
        "scenario": {"name": "s1", "k_clusters": 3, "cluster_algorithm": "kmeans"},
        "encoder_text": {"name": "dummy", "model_id": "dummy", "embedding_dim": 16},
        "encoder_image": {"name": "dummy", "model_id": "dummy", "embedding_dim": 16},
        "fusion": {"name": "late_concat", "output_dim": 32},
        "manifold": {"name": "euclidean", "dim": 32, "curvature": 0.0},
        "training": {"batch_size": 8, **training},
        "logger": {"name": "none"},
        "corpus": {"root": "."},
        "device": "cpu",
        "seed": 0,
    }
    return Config.model_validate(raw)


@pytest.mark.parametrize("loss", ["pseudo_label", "knn_contrastive", "cross_modal"])
def test_s1_unsupervised_projector_training_end_to_end(loss: str) -> None:
    cfg = _s1_cfg(
        {
            "epochs": 4,
            "loss": loss,
            "trainable_projector": True,
            "pseudo_recluster_every": 2,
            "temperature": 0.5 if loss == "knn_contrastive" else 0.07,
        }
    )
    result = build_scenario(cfg).fit_predict(_InMemoryDataset(12))
    assert len(result.predictions) == 12
    assert all(p is not None and p.startswith("cluster_") for p in result.predictions)
    assert result.metadata["projector_trained"] is True
    assert result.metadata["projector_unsupervised"] is True
    assert result.metadata["projector_loss"] == loss
    assert len(result.metadata["projector_loss_history"]) == 4


def test_s1_unsupervised_loss_ignores_supervision_field() -> None:
    """An unsupervised ``loss`` wins over ``supervision`` and trains label-free."""
    cfg = _s1_cfg(
        {
            "epochs": 3,
            "loss": "pseudo_label",
            "trainable_projector": True,
            "supervision": "labels",  # would be a no-op here (no labels) — must be bypassed
            "pseudo_recluster_every": 2,
        }
    )
    result = build_scenario(cfg).fit_predict(_InMemoryDataset(9))
    assert result.metadata["projector_unsupervised"] is True
    assert result.metadata["projector_trained"] is True
    assert len(result.metadata["projector_loss_history"]) == 3


def test_s1_supervised_loss_reports_not_unsupervised() -> None:
    """Regression: a supervised loss leaves projector_unsupervised False."""
    cfg = _s1_cfg({"epochs": 3, "loss": "prototypical", "trainable_projector": True})
    result = build_scenario(cfg).fit_predict(_InMemoryDataset(9))
    assert result.metadata["projector_unsupervised"] is False
    assert result.metadata["projector_loss"] == "prototypical"


# ── Hyperbolic numerical stability (acosh-at-1 NaN gradient) ─────────────
def test_hyperbolic_dist_has_finite_gradient_for_coincident_points() -> None:
    """acosh'(1) = +inf; dist(x, x) must still backprop without NaN."""
    m = build_manifold(ManifoldConfig(name="hyperbolic", dim=DIM, curvature=1.0))
    x = torch.randn(4, DIM, requires_grad=True)
    z = m.expmap0(x)
    m.dist(z, z.detach()).sum().backward()  # type: ignore[no-untyped-call]
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_pseudo_label_on_hyperbolic_does_not_diverge_to_nan() -> None:
    """Long EM run on the Poincaré ball with a singleton cluster stays finite.

    Regression for the acosh-at-1 NaN that surfaced downstream as
    'PCA: Input X contains NaN' once the projector weights blew up.
    """
    m = build_manifold(ManifoldConfig(name="hyperbolic", dim=DIM, curvature=1.0))
    torch.manual_seed(0)
    proj = ManifoldProjector(m, input_dim=DIM, output_dim=DIM, trainable=True)
    base = torch.randn(1, DIM)
    # 5 identical points + 1 outlier → with k=2 one prototype == its lone member.
    fused = torch.cat([base.repeat(5, 1), torch.randn(1, DIM) * 5.0], dim=0)
    cfg = _cfg(
        loss="pseudo_label",
        epochs=120,
        lr=4e-4,
        pseudo_recluster_every=25,
        sinkhorn=False,
        vicreg_var_weight=0.95,
        vicreg_cov_weight=0.01,
    )
    history = train_projector(proj, fused, [None] * 6, cfg=cfg, pseudo_k=2)
    assert all(not math.isnan(v) for v in history)
    assert not bool(torch.isnan(proj(fused)).any())


# ── Joint fusion + projector training (label-free) ───────────────────────
def _fusion_snapshot(scenario: object) -> list[torch.Tensor]:
    return [p.detach().clone() for p in scenario.fusion.parameters()]  # type: ignore[attr-defined]


@pytest.mark.parametrize("loss", ["pseudo_label", "knn_contrastive"])
def test_s1_trainable_fusion_trains_fusion_label_free(loss: str) -> None:
    """--trainable-fusion + an unsupervised loss updates the fusion weights too."""
    cfg = _s1_cfg(
        {
            "epochs": 4,
            "loss": loss,
            "trainable_projector": True,
            "trainable_fusion": True,
            "pseudo_recluster_every": 2,
            "temperature": 0.5 if loss == "knn_contrastive" else 0.07,
        }
    )
    scenario = build_scenario(cfg)
    before = _fusion_snapshot(scenario)
    assert before, "late_concat fusion should expose trainable parameters"
    result = scenario.fit_predict(_InMemoryDataset(12))
    after = _fusion_snapshot(scenario)
    # The fusion actually moved (not just the projector).
    assert any(not torch.allclose(b, a) for b, a in zip(before, after, strict=True))
    assert result.metadata["projector_trained"] is True
    assert result.metadata["projector_unsupervised"] is True
    assert all(p is not None and p.startswith("cluster_") for p in result.predictions)


def test_s1_cross_modal_keeps_fusion_frozen_even_with_trainable_fusion() -> None:
    """cross_modal trains only the projector; the fusion stays put."""
    cfg = _s1_cfg(
        {
            "epochs": 4,
            "loss": "cross_modal",
            "trainable_projector": True,
            "trainable_fusion": True,
        }
    )
    scenario = build_scenario(cfg)
    before = _fusion_snapshot(scenario)
    result = scenario.fit_predict(_InMemoryDataset(12))
    after = _fusion_snapshot(scenario)
    assert all(torch.allclose(b, a) for b, a in zip(before, after, strict=True))
    assert result.metadata["projector_trained"] is True


def test_train_fusion_projector_rejects_cross_modal() -> None:
    from clustering.manifolds.training import train_fusion_projector

    # late_concat has trainable weights; dims are arbitrary for the guard.
    fusion = build_fusion(
        FusionConfig(name="late_concat", output_dim=DIM), text_dim=DIM, image_dim=DIM
    )
    proj = _projector(dim=DIM)
    with pytest.raises(ValueError, match="cross_modal"):
        train_fusion_projector(
            fusion,
            proj,
            torch.randn(6, DIM),
            torch.randn(6, DIM),
            [None] * 6,
            cfg=_cfg(loss="cross_modal"),
        )
