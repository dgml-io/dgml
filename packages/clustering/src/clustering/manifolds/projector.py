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

"""Trainable projector that maps ambient ``R^d`` onto the active manifold.

``ManifoldProjector`` wraps a :class:`~clustering.manifolds.base.ManifoldHead`
with an optional ``nn.Linear`` pre-projection. Three roles:

1. **Dim adaptation.** When the fusion's ``output_dim`` differs from
   ``manifold.dim``, the linear is automatically inserted. (Without the
   projector, those two would have to match — the framework would
   otherwise raise a shape error at the manifold boundary.)
2. **Trainable head.** With ``trainable=True``, the linear's weights are
   learned by :func:`clustering.manifolds.training.train_projector` under
   one of the manifold-aware losses (contrastive / triplet / prototypical).
3. **On-manifold anchor (Riemannian).** With ``manifold_bias=True``, the
   projector also exposes a :class:`geoopt.ManifoldParameter` that lives
   *on* the manifold and serves as the base point of the forward
   ``expmap(anchor, tangent)``. This is the parameter that
   :class:`geoopt.optim.RiemannianAdam` updates along the manifold's
   geodesics — i.e. (together with the prototypical loss's learnable
   prototypes) a parameter for which "Riemannian gradient updates" is
   more than a re-spelling of plain Adam.

If ``trainable=False``, ``manifold_bias=False`` and the two dims match,
the projector is exactly equivalent to calling ``manifold.expmap0(x)``
directly — no parameters, no behavior change. So the default scenario
path is unaffected.

Passing ``force_identity=True`` makes that parameter-free passthrough
*explicit* and guaranteed: the linear is always :class:`nn.Identity`,
never a dim-adaptation / trainable layer. It requires ``input_dim ==
output_dim`` (a true identity cannot bridge differing dimensions) and is
incompatible with ``trainable=True``. Use it for a clean, fully
reproducible untrained baseline where you want to be sure no
randomly-initialized linear sneaks in.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from clustering.manifolds.base import ManifoldHead


class ManifoldProjector(nn.Module):
    """Optional-linear → manifold ``expmap`` head with optional on-manifold anchor.

    The projector exposes the manifold's distance / expmap / logmap
    operations via thin delegations.
    """

    def __init__(
        self,
        manifold: ManifoldHead,
        *,
        input_dim: int,
        output_dim: int | None = None,
        trainable: bool = False,
        manifold_bias: bool = False,
        force_identity: bool = False,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else manifold.dim
        self.trainable = trainable
        self.manifold_bias = manifold_bias
        self.force_identity = force_identity

        if force_identity:
            # Explicit parameter-free passthrough. A true identity can't
            # bridge dims and has nothing to train, so reject the
            # contradictory combinations loudly rather than silently
            # falling back to a linear.
            if trainable:
                raise ValueError(
                    "ManifoldProjector(force_identity=True) is incompatible with "
                    "trainable=True: an identity projector has no parameters to train."
                )
            if input_dim != self.output_dim:
                raise ValueError(
                    "ManifoldProjector(force_identity=True) requires "
                    f"input_dim == output_dim; got input_dim={input_dim}, "
                    f"output_dim={self.output_dim}. A parameter-free identity cannot "
                    "bridge differing dimensions — match fusion.output_dim to "
                    "manifold.dim, or drop force_identity to let the dim-adaptation "
                    "linear handle it."
                )
            self.linear: nn.Module = nn.Identity()
        elif trainable or input_dim != self.output_dim:
            self.linear = nn.Linear(input_dim, self.output_dim, bias=True)
            # Identity-init when shapes match: makes "before training"
            # behave like a no-op so untrained metrics are interpretable.
            if input_dim == self.output_dim:
                self._identity_init()
        else:
            self.linear = nn.Identity()

        # Optional on-manifold anchor — initialized at the manifold's
        # origin so the very-first forward equals ``expmap0(linear(x))``
        # on origin-symmetric manifolds (Euclidean, Poincaré ball).
        # Held as a geoopt.ManifoldParameter so RiemannianAdam updates it
        # along the manifold's geodesics. Register under ``_anchor`` so
        # ``self.parameters()`` picks it up (or stays empty when
        # manifold_bias=False).
        self.register_parameter("_anchor", None)
        if manifold_bias:
            self._anchor = self._build_anchor()

    def _identity_init(self) -> None:
        if not isinstance(self.linear, nn.Linear):
            return
        with torch.no_grad():
            weight = self.linear.weight
            # Eye matrix in-place: w[i, j] = 1 if i == j else 0.
            for i in range(self.output_dim):
                for j in range(self.input_dim):
                    weight[i, j] = 1.0 if i == j else 0.0
            if self.linear.bias is not None:
                for i in range(self.output_dim):
                    self.linear.bias[i] = 0.0

    def _build_anchor(self) -> Any:
        """Create a ``geoopt.ManifoldParameter`` initialized at the manifold origin.

        ``RiemannianAdam`` inspects the parameter's ``manifold``
        attribute to decide whether to Riemannian-update it (on-manifold)
        or fall back to plain Adam, so storing the geoopt manifold on
        the parameter is what wires Riemannian gradients in.
        """
        import geoopt

        gmf: Any = self.manifold.to_geoopt()
        origin = self.manifold.manifold_origin().detach().clone()
        # ``projx`` snaps the init to the manifold (idempotent if already
        # on-manifold) — protects against numerical drift in custom
        # ``manifold_origin`` implementations.
        if hasattr(gmf, "projx"):
            origin = gmf.projx(origin)
        return geoopt.ManifoldParameter(origin, manifold=gmf, requires_grad=True)

    # ── Forward: ambient → on-manifold ───────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ambient = self.linear(x)
        if self._anchor is None:
            # No anchor: map the tangent vector through the origin.
            return self.manifold.expmap0(ambient)
        # Riemannian path: anchor is the base point on M; ambient is
        # interpreted as a tangent vector at the anchor (projected onto
        # T_anchor M via geoopt's ``proju`` when the manifold needs it,
        # e.g. the sphere; on Euclidean / PoincaréBall ``proju`` is a
        # no-op or identity).
        gmf: Any = self.manifold.to_geoopt()
        base = self._anchor.expand_as(ambient)
        tangent = gmf.proju(base, ambient) if hasattr(gmf, "proju") else ambient
        return cast(torch.Tensor, gmf.expmap(base, tangent))

    # ── Manifold-op delegations (so callers can treat us like a Head) ────
    @property
    def dim(self) -> int:
        return self.manifold.dim

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.manifold.project(x)

    def expmap0(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.manifold.expmap(x, v)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.manifold.logmap(x, y)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.manifold.dist(x, y)

    def pairwise_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.manifold.pairwise_dist(x, y)
