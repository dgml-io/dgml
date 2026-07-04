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

"""Tests for Riemannian gradient updates via geoopt.

Three contracts:

1. Each :class:`ManifoldHead` exposes a matching ``geoopt`` manifold
   (``Euclidean`` / ``Sphere`` / ``PoincareBall`` / ``ProductManifold``)
   and a canonical on-manifold origin.
2. With ``manifold_bias=True`` the projector registers a
   ``geoopt.ManifoldParameter`` initialized on-manifold, and remains
   on-manifold after a few training steps.
3. The default path (``riemannian=False``, ``manifold_bias=False``) is
   unchanged — no on-manifold anchor, forward stays ``expmap0(linear(x))``.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from clustering.config.schema import ManifoldComponent, ManifoldConfig, TrainingConfig
from clustering.manifolds import (
    ManifoldProjector,
    build_manifold,
    train_projector,
)

# Skip the whole module cleanly if geoopt isn't installed — sandbox CI
# may lack it even though it's a declared core dep.
_GEOOPT_AVAILABLE = importlib.util.find_spec("geoopt") is not None
pytestmark = pytest.mark.skipif(
    not _GEOOPT_AVAILABLE, reason="geoopt not installed in this environment"
)


def _anchor_tensor(p: ManifoldProjector) -> torch.Tensor:
    """Fetch the projector's anchor as a plain tensor (mypy-friendly)."""
    anchor = p._anchor
    assert isinstance(anchor, torch.Tensor)
    return anchor.detach()


# ── Heads expose the right geoopt manifold ──────────────────────────────
def test_euclidean_head_to_geoopt_is_euclidean() -> None:
    import geoopt

    m = build_manifold(ManifoldConfig(name="euclidean", dim=8, curvature=0.0))
    assert isinstance(m.to_geoopt(), geoopt.Euclidean)


def test_spherical_head_to_geoopt_is_sphere() -> None:
    import geoopt

    m = build_manifold(ManifoldConfig(name="spherical", dim=8, curvature=1.0))
    assert isinstance(m.to_geoopt(), geoopt.Sphere)


def test_hyperbolic_head_to_geoopt_is_poincare_ball() -> None:
    import geoopt

    m = build_manifold(ManifoldConfig(name="hyperbolic", dim=8, curvature=1.0))
    assert isinstance(m.to_geoopt(), geoopt.PoincareBall)


def test_product_head_to_geoopt_is_product() -> None:
    import geoopt

    m = build_manifold(
        ManifoldConfig(
            name="product",
            dim=8,
            components=[
                ManifoldComponent(name="euclidean", dim=4, curvature=0.0),
                ManifoldComponent(name="hyperbolic", dim=4, curvature=1.0),
            ],
        )
    )
    assert isinstance(m.to_geoopt(), geoopt.ProductManifold)


# ── Origin shapes ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cfg",
    [
        ManifoldConfig(name="euclidean", dim=8, curvature=0.0),
        ManifoldConfig(name="spherical", dim=8, curvature=1.0),
        ManifoldConfig(name="hyperbolic", dim=8, curvature=1.0),
        ManifoldConfig(
            name="product",
            dim=8,
            components=[
                ManifoldComponent(name="euclidean", dim=4, curvature=0.0),
                ManifoldComponent(name="hyperbolic", dim=4, curvature=1.0),
            ],
        ),
    ],
    ids=["euclidean", "spherical", "hyperbolic", "product"],
)
def test_manifold_origin_shape_matches_dim(cfg: ManifoldConfig) -> None:
    m = build_manifold(cfg)
    origin = m.manifold_origin()
    assert origin.shape == (m.dim,)


def test_spherical_origin_is_unit_norm() -> None:
    """Sphere origin must be on the unit sphere — zeros wouldn't be."""
    m = build_manifold(ManifoldConfig(name="spherical", dim=8, curvature=1.0))
    origin = m.manifold_origin()
    assert torch.isclose(origin.norm(), torch.tensor(1.0), atol=1e-6)


# ── Projector with manifold_bias=True ───────────────────────────────────
def test_projector_manifold_bias_registers_manifold_parameter() -> None:
    import geoopt

    m = build_manifold(ManifoldConfig(name="hyperbolic", dim=8, curvature=1.0))
    p = ManifoldProjector(m, input_dim=8, output_dim=8, trainable=True, manifold_bias=True)
    # Exactly one ManifoldParameter (the anchor) plus the linear's weight + bias.
    manifold_params = [
        param for param in p.parameters() if isinstance(param, geoopt.ManifoldParameter)
    ]
    assert len(manifold_params) == 1
    assert manifold_params[0].shape == (8,)


def test_projector_manifold_bias_init_on_manifold_for_sphere() -> None:
    """The anchor's init must be on the manifold — for the sphere that means unit norm."""
    m = build_manifold(ManifoldConfig(name="spherical", dim=8, curvature=1.0))
    p = ManifoldProjector(m, input_dim=8, output_dim=8, trainable=True, manifold_bias=True)
    assert torch.isclose(_anchor_tensor(p).norm(), torch.tensor(1.0), atol=1e-5)


def test_projector_without_manifold_bias_has_no_anchor() -> None:
    m = build_manifold(ManifoldConfig(name="euclidean", dim=8, curvature=0.0))
    p = ManifoldProjector(m, input_dim=8, output_dim=8, trainable=True, manifold_bias=False)
    assert p._anchor is None


def test_projector_default_forward_unchanged_without_anchor() -> None:
    """Backwards-compat: no anchor → forward equals manifold.expmap0(linear(x))."""
    m = build_manifold(ManifoldConfig(name="euclidean", dim=8, curvature=0.0))
    p = ManifoldProjector(m, input_dim=8, output_dim=8, trainable=True, manifold_bias=False)
    x = torch.randn(4, 8) * 0.1
    assert torch.allclose(p(x), m.expmap0(p.linear(x)), atol=1e-5)


# ── Riemannian training preserves on-manifold invariant ─────────────────
def _toy_two_class(n: int = 4, d: int = 8) -> tuple[torch.Tensor, list[str]]:
    torch.manual_seed(0)
    a = torch.randn(n, d) * 0.1 + torch.tensor([0.5] + [0.0] * (d - 1))
    b = torch.randn(n, d) * 0.1 + torch.tensor([-0.5] + [0.0] * (d - 1))
    return torch.cat([a, b], dim=0), ["A"] * n + ["B"] * n


@pytest.mark.parametrize("manifold_name", ["euclidean", "spherical", "hyperbolic"])
def test_riemannian_training_keeps_anchor_on_manifold(manifold_name: str) -> None:
    """After RiemannianAdam steps, the anchor must still satisfy the manifold's constraints."""
    curvature = 1.0 if manifold_name != "euclidean" else 0.0
    cfg_m = ManifoldConfig.model_validate({"name": manifold_name, "dim": 8, "curvature": curvature})
    m = build_manifold(cfg_m)
    p = ManifoldProjector(m, input_dim=8, output_dim=8, trainable=True, manifold_bias=True)
    fused, labels = _toy_two_class()
    cfg = TrainingConfig(epochs=3, loss="prototypical", lr=1e-2, riemannian=True)
    history = train_projector(p, fused, labels, cfg=cfg)
    assert len(history) == 3

    anchor = _anchor_tensor(p)
    if manifold_name == "spherical":
        assert torch.isclose(anchor.norm(), torch.tensor(1.0), atol=1e-4)
    elif manifold_name == "hyperbolic":
        # Strictly inside the Poincaré ball.
        assert float(anchor.norm()) < 1.0
    else:
        # Euclidean — any finite vector is on-manifold.
        assert torch.isfinite(anchor).all()


def test_riemannian_training_actually_moves_anchor() -> None:
    """A non-zero gradient on the anchor must produce a non-zero update."""
    m = build_manifold(ManifoldConfig(name="hyperbolic", dim=8, curvature=1.0))
    p = ManifoldProjector(m, input_dim=8, output_dim=8, trainable=True, manifold_bias=True)
    fused, labels = _toy_two_class()
    before = _anchor_tensor(p).clone()
    cfg = TrainingConfig(epochs=5, loss="prototypical", lr=1e-1, riemannian=True)
    train_projector(p, fused, labels, cfg=cfg)
    after = _anchor_tensor(p)
    assert not torch.allclose(before, after, atol=1e-6), "anchor did not move"


def test_riemannian_false_path_unchanged() -> None:
    """riemannian=False must produce no on-manifold anchor and behave as before."""
    m = build_manifold(ManifoldConfig(name="euclidean", dim=8, curvature=0.0))
    p = ManifoldProjector(m, input_dim=8, output_dim=8, trainable=True, manifold_bias=False)
    fused, labels = _toy_two_class()
    cfg = TrainingConfig(epochs=2, loss="prototypical", lr=1e-2, riemannian=False)
    history = train_projector(p, fused, labels, cfg=cfg)
    assert len(history) == 2
    # No anchor; only the linear's parameters.
    assert p._anchor is None
