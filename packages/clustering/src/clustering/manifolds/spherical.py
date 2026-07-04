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

"""Spherical manifold (unit hypersphere ``S^{d-1}``).

We implement the unit-radius sphere (curvature +1). The ``curvature`` field
is read from config but only enters via the prefactor ``1/sqrt(c)`` in the
distance formula; for ``c=1`` it simplifies to the standard great-circle
distance ``arccos(<x, y>)``.
"""

from __future__ import annotations

from typing import cast

import torch

from clustering.config.schema import ManifoldConfig
from clustering.manifolds.base import ManifoldHead, register_manifold

_EPS = 1e-6


def _safe_normalize(x: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    return cast(torch.Tensor, x / norm)


class SphericalHead(ManifoldHead):
    def __init__(self, cfg: ManifoldConfig) -> None:
        super().__init__()
        self.dim = cfg.dim
        self.c: float = float(cfg.curvature) if cfg.curvature > 0 else 1.0

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return _safe_normalize(x)

    def expmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained ambient vector onto the sphere.

        Equivalent to :meth:`project` (L2 normalize) — the sphere has no
        privileged origin, so the "from-origin exponential map" idiom maps
        to the closest on-sphere unit vector.
        """
        return _safe_normalize(x)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Use the chord-length / asin formulation: more stable than acos near
        # x ≈ y (where acos has unbounded derivative). For unit-norm x, y on
        # the sphere, the angle is 2*asin(||x - y|| / 2).
        #
        # We defensively L2-normalize both inputs first — the chord→angle
        # identity only holds on the unit sphere, and the caller may pass
        # slightly-off-sphere embeddings (e.g. raw fused vectors, or
        # outputs of the trainable projector's linear layer before
        # expmap0). Hyperbolic.dist does the equivalent via
        # _project_into_ball; this is the spherical-side parity fix.
        # Inputs that are already on the sphere are unchanged (modulo
        # float epsilon) so this is a no-op on the hot path.
        x = _safe_normalize(x)
        y = _safe_normalize(y)
        chord = (x - y).norm(dim=-1) * 0.5
        return cast(torch.Tensor, 2.0 * torch.asin(chord.clamp(max=1.0)) / (self.c**0.5))

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x on S, v tangent at x (v · x = 0).
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        return cast(torch.Tensor, torch.cos(v_norm) * x + torch.sin(v_norm) * (v / v_norm))

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ip = (x * y).sum(dim=-1, keepdim=True).clamp(min=-1.0 + _EPS, max=1.0 - _EPS)
        d = torch.acos(ip)  # angle between x and y
        v = y - ip * x
        v_norm = v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        return cast(torch.Tensor, v * (d / v_norm))

    def to_geoopt(self) -> object:
        # Curvature only enters via the 1/sqrt(c) scaling of the distance;
        # geoopt.Sphere() is the unit sphere (c = 1). For c != 1 we still
        # return Sphere() — the Riemannian gradient direction is unchanged,
        # only the metric scale differs. Wire a custom geoopt.Sphere subclass
        # if you need exact curvature semantics under the optimizer.
        import geoopt

        return geoopt.Sphere()

    def manifold_origin(self) -> torch.Tensor:
        """First basis vector ``e_0``. The zero vector is not on the sphere."""
        origin = torch.zeros(self.dim)
        origin[0] = 1.0
        return origin


@register_manifold("spherical")
def _factory(cfg: ManifoldConfig) -> ManifoldHead:
    return SphericalHead(cfg)
