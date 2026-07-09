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

"""Product manifold — composes mixed geometries along disjoint dim ranges.

E.g. ``product = euclidean(128) x hyperbolic(128)`` runs a 256-d embedding
where the first 128 dims live in flat ``R^128`` and the last 128 dims live
in a Poincaré ball. The Riemannian product distance is

    d_prod(x, y) = sqrt( Σ_i  d_i(x_i, y_i)² )

and project / expmap / logmap apply per-component.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch

from clustering.config.schema import ManifoldComponent, ManifoldConfig
from clustering.manifolds.base import ManifoldHead, register_manifold
from clustering.manifolds.euclidean import EuclideanHead
from clustering.manifolds.hyperbolic import HyperbolicHead
from clustering.manifolds.spherical import SphericalHead


def _component_head(c: ManifoldComponent) -> ManifoldHead:
    sub_cfg = ManifoldConfig(name=c.name, dim=c.dim, curvature=c.curvature)
    if c.name == "euclidean":
        return EuclideanHead(sub_cfg)
    if c.name == "spherical":
        return SphericalHead(sub_cfg)
    if c.name == "hyperbolic":
        return HyperbolicHead(sub_cfg)
    raise ValueError(f"Unknown component manifold: {c.name!r}")


class ProductHead(ManifoldHead):
    """Per-dim Cartesian product of submanifolds."""

    def __init__(self, cfg: ManifoldConfig) -> None:
        super().__init__()
        if cfg.components is None:
            raise ValueError("Product manifold requires components.")
        self.dim = cfg.dim
        self.components = tuple(_component_head(c) for c in cfg.components)
        # Per-component slice indices into the trailing dim.
        offsets: list[int] = [0]
        for c in cfg.components:
            offsets.append(offsets[-1] + c.dim)
        self._offsets: tuple[int, ...] = tuple(offsets)

    def _slices(self) -> Iterator[slice]:
        for i in range(len(self.components)):
            yield slice(self._offsets[i], self._offsets[i + 1])

    def project(self, x: torch.Tensor) -> torch.Tensor:
        parts = [
            head.project(x[..., sl])
            for head, sl in zip(self.components, self._slices(), strict=True)
        ]
        return torch.cat(parts, dim=-1)

    def expmap0(self, x: torch.Tensor) -> torch.Tensor:
        parts = [
            head.expmap0(x[..., sl])
            for head, sl in zip(self.components, self._slices(), strict=True)
        ]
        return torch.cat(parts, dim=-1)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sqs: list[torch.Tensor] = []
        for head, sl in zip(self.components, self._slices(), strict=True):
            d = head.dist(x[..., sl], y[..., sl])
            sqs.append(d * d)
        total = sqs[0]
        for s in sqs[1:]:
            total = total + s
        return cast(torch.Tensor, total.clamp(min=0.0) ** 0.5)

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        parts = [
            head.expmap(x[..., sl], v[..., sl])
            for head, sl in zip(self.components, self._slices(), strict=True)
        ]
        return torch.cat(parts, dim=-1)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        parts = [
            head.logmap(x[..., sl], y[..., sl])
            for head, sl in zip(self.components, self._slices(), strict=True)
        ]
        return torch.cat(parts, dim=-1)

    def to_geoopt(self) -> object:
        """``geoopt.ProductManifold`` of per-component geoopt manifolds.

        Geoopt expects each factor to be paired with the *trailing-dim
        size* it occupies, in the order of concatenation — which matches
        our :attr:`_offsets` slicing exactly.
        """
        import geoopt

        factors = tuple((head.to_geoopt(), head.dim) for head in self.components)
        return geoopt.ProductManifold(*factors)

    def manifold_origin(self) -> torch.Tensor:
        """Concatenated per-component origins (Euclidean: 0, Sphere: e_0, Poincaré: 0)."""
        return torch.cat([head.manifold_origin() for head in self.components], dim=-1)


@register_manifold("product")
def _factory(cfg: ManifoldConfig) -> ManifoldHead:
    return ProductHead(cfg)
