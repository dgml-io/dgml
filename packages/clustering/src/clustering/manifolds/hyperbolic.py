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

"""Hyperbolic manifold — Poincaré ball model.

Formulas follow Ganea, Bécigneul & Hofmann (NeurIPS 2018). With curvature
``c > 0`` the ball has radius ``1/sqrt(c)``; we default to ``c = 1`` (unit
ball). Inputs to :meth:`project` are mapped *into* the ball with a
``tanh``-style contraction so users can feed arbitrary ambient vectors.

The math here is intentionally explicit and stateless. For Riemannian
optimizers, wrap the parameters via geoopt.PoincareBall in your training
code; the forward operations do not need geoopt to be installed.
"""

from __future__ import annotations

from typing import cast

import torch

from clustering.config.schema import ManifoldConfig
from clustering.manifolds.base import ManifoldHead, register_manifold

_EPS = 1e-6
_MAX_NORM = 1.0 - 1e-3  # clip ‖x‖ strictly inside the ball for numerics
# acosh'(t) = 1/sqrt(t^2 - 1) blows up to +inf at t = 1, so clamping the
# distance argument at exactly 1.0 yields NaN gradients whenever two points
# coincide (dist = 0) — common during training: a singleton cluster's
# prototype equals its member. Clamp strictly above 1 so the gradient stays
# finite. The distance error is ~sqrt(2·margin) ≈ 1.4e-3, negligible for
# clustering, and points at/below the bound get a (correct) zero gradient.
_MIN_ACOSH_ARG = 1.0 + _EPS


def _conformal_factor(x: torch.Tensor, c: float) -> torch.Tensor:
    """λ_x = 2 / (1 - c ||x||^2). Shape: trailing-dim is kept for broadcasting."""
    sq = (x * x).sum(dim=-1, keepdim=True)
    return cast(torch.Tensor, 2.0 / (1.0 - c * sq).clamp(min=_EPS))


def _mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
    """Möbius addition x ⊕_c y in the Poincaré ball of curvature c."""
    xy = (x * y).sum(dim=-1, keepdim=True)
    xx = (x * x).sum(dim=-1, keepdim=True)
    yy = (y * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c * xy + c * yy) * x + (1.0 - c * xx) * y
    den = (1.0 + 2.0 * c * xy + (c**2) * xx * yy).clamp(min=_EPS)
    return cast(torch.Tensor, num / den)


def _project_into_ball(x: torch.Tensor, c: float) -> torch.Tensor:
    """Ensure ‖√c · x‖ < 1 strictly (numerical safety).

    Uses ``clamp(max=1.0)`` on the contraction factor so points already
    inside the ball are unchanged and only out-of-ball points are scaled in.
    """
    norm = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    max_norm = _MAX_NORM / (c**0.5)
    scale = (max_norm / norm).clamp(max=1.0)
    return cast(torch.Tensor, x * scale)


class HyperbolicHead(ManifoldHead):
    """Poincaré-ball head with curvature ``c > 0`` (unit ball when ``c = 1``)."""

    def __init__(self, cfg: ManifoldConfig) -> None:
        super().__init__()
        self.dim = cfg.dim
        self.c: float = float(cfg.curvature) if cfg.curvature > 0 else 1.0

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Idempotent projection back into the ball.

        Points already strictly inside the ball pass through unchanged.
        Points outside are scaled radially to ``||sqrt(c) * y|| = 1 - delta``.

        To map an *unconstrained* ambient vector into the ball (as you would
        at the manifold-head boundary), use :meth:`expmap0` instead.
        """
        return _project_into_ball(x, self.c)

    def expmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Exponential map from the origin: ``R^d -> B^d``.

        ``expmap0(x) = (x / ‖x‖) · tanh(√c · ‖x‖) / √c``.
        Used to push unconstrained fusion outputs onto the manifold.

        We pipe the result through :func:`_project_into_ball` because
        ``tanh`` can saturate to ``1.0`` in float32 for large inputs, which
        would place the result *on* (not strictly inside) the ball boundary.
        """
        norm = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        direction = x / norm
        sqrt_c = self.c**0.5
        y = direction * torch.tanh(sqrt_c * norm) / sqrt_c
        return _project_into_ball(y, self.c)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = _project_into_ball(x, self.c)
        y = _project_into_ball(y, self.c)
        diff = x - y
        diff_sq = (diff * diff).sum(dim=-1)
        x_sq = (x * x).sum(dim=-1)
        y_sq = (y * y).sum(dim=-1)
        # acosh(1 + 2c||x-y||^2 / ((1-c||x||^2)(1-c||y||^2))) / sqrt(c)
        num = 2.0 * self.c * diff_sq
        den = ((1.0 - self.c * x_sq) * (1.0 - self.c * y_sq)).clamp(min=_EPS)
        # Clamp strictly above 1.0 (avoids NaN gradients at coincident points)
        # and below 1e15 (prevents OverflowError on Windows where acosh(inf)
        # can't be converted downstream). See ``_MIN_ACOSH_ARG``.
        arg = (1.0 + num / den).clamp(min=_MIN_ACOSH_ARG, max=1e15)
        return cast(torch.Tensor, torch.acosh(arg) / (self.c**0.5))

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x = _project_into_ball(x, self.c)
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        lam = _conformal_factor(x, self.c)
        # second_term = tanh(√c · λ_x · ‖v‖ / 2) · v / (√c · ‖v‖)
        sqrt_c = self.c**0.5
        coef = torch.tanh(sqrt_c * lam * v_norm / 2.0) / (sqrt_c * v_norm)
        return _project_into_ball(_mobius_add(x, coef * v, self.c), self.c)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = _project_into_ball(x, self.c)
        y = _project_into_ball(y, self.c)
        sub = _mobius_add(-x, y, self.c)
        sub_norm = sub.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        lam = _conformal_factor(x, self.c)
        sqrt_c = self.c**0.5
        coef = (2.0 / (sqrt_c * lam)) * torch.atanh((sqrt_c * sub_norm).clamp(max=1.0 - _EPS))
        return cast(torch.Tensor, coef * (sub / sub_norm))

    def to_geoopt(self) -> object:
        # geoopt's PoincareBall takes the curvature directly. Matches our
        # convention: c > 0, unit ball at c = 1.
        import geoopt

        return geoopt.PoincareBall(c=self.c)


@register_manifold("hyperbolic")
def _factory(cfg: ManifoldConfig) -> ManifoldHead:
    return HyperbolicHead(cfg)
