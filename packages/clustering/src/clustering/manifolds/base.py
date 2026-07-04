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

"""ManifoldHead ABC + registry.

The forward math (``project`` / ``dist`` / ``expmap`` / ``logmap``) is
implemented in pure torch in each concrete subclass. For Riemannian
gradient updates (RSGD, RAdam), wire the head's parameters through
``geoopt`` — that's a future task; the framework's forward path does not
require geoopt to be installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
from torch import nn

from clustering.config.schema import ManifoldConfig


class ManifoldHead(nn.Module, ABC):
    """Projection head for a Riemannian manifold.

    All concrete subclasses expose:

    - :meth:`project` — push an ambient ``R^d`` vector onto the manifold.
    - :meth:`dist`    — row-wise distance between matched-shape inputs.
    - :meth:`pairwise_dist` — distance between every pair of rows.
    - :meth:`expmap`  — move from a base point along a tangent vector.
    - :meth:`logmap`  — recover the tangent vector that points from base to target.

    The convention is that ``dist(x, y).shape == x.shape[:-1]``.
    """

    dim: int

    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Push the ambient vector ``x`` onto the manifold."""

    @abstractmethod
    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Row-wise manifold distance for matched shapes."""

    def pairwise_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cross-batch manifold distance: ``[B1, D] x [B2, D] -> [B1, B2]``.

        Default impl broadcasts ``dist`` over a Cartesian grid. Subclasses
        may override for efficiency.
        """
        b1, b2 = x.shape[0], y.shape[0]
        xx = x.unsqueeze(1).expand(b1, b2, x.shape[-1])  # [B1, B2, D]
        yy = y.unsqueeze(0).expand(b1, b2, y.shape[-1])  # [B1, B2, D]
        return self.dist(xx, yy)

    @abstractmethod
    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map: move from base ``x`` along tangent ``v``."""

    @abstractmethod
    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithm map: tangent at ``x`` pointing to ``y``."""

    def expmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Map an unconstrained ambient vector onto the manifold (exp from origin).

        Default impl delegates to :meth:`expmap` from a zero base, which is
        correct for Euclidean. Spherical and Hyperbolic override this.
        """
        return self.expmap(torch.zeros_like(x), x)

    def to_geoopt(self) -> object:
        """Return the corresponding :mod:`geoopt` manifold object.

        Used by Riemannian optimizers (``geoopt.optim.RiemannianAdam`` /
        ``RiemannianSGD``) and by ``geoopt.ManifoldParameter`` to constrain
        a learnable tensor to live on this manifold.

        Subclasses should override; the default raises
        :class:`NotImplementedError` so unsupported manifolds fail loudly
        rather than silently degrading to a Euclidean update.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.to_geoopt() is not implemented. "
            "Override it to enable Riemannian optimization on this manifold."
        )

    def manifold_origin(self) -> torch.Tensor:
        """Canonical on-manifold initialization point of shape ``[dim]``.

        Used by :class:`~clustering.manifolds.projector.ManifoldProjector`
        to initialize a learnable on-manifold bias / anchor. Default is the
        zero vector — correct for Euclidean and the Poincaré ball. The
        sphere overrides this because the zero vector is not on the unit
        sphere; the product manifold concatenates per-component origins.
        """
        return torch.zeros(self.dim)


# ── Registry ─────────────────────────────────────────────────────────────
ManifoldFactory = Callable[..., ManifoldHead]
_REGISTRY: dict[str, ManifoldFactory] = {}


def register_manifold(name: str) -> Callable[[ManifoldFactory], ManifoldFactory]:
    def deco(fn: ManifoldFactory) -> ManifoldFactory:
        if name in _REGISTRY:
            raise ValueError(f"Manifold {name!r} is already registered.")
        _REGISTRY[name] = fn
        return fn

    return deco


def build_manifold(cfg: ManifoldConfig) -> ManifoldHead:
    if cfg.name not in _REGISTRY:
        raise KeyError(f"Unknown manifold {cfg.name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.name](cfg)


def registered_manifolds() -> list[str]:
    return sorted(_REGISTRY)
