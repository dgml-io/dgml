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

"""Manifold-aware losses.

All losses consume :class:`~clustering.manifolds.base.ManifoldHead`-routed
distances rather than raw Euclidean — so the same loss code does the right
thing whether the embedding lives in flat ``R^d``, on a sphere, or in a
Poincaré ball.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from clustering.manifolds.base import ManifoldHead


# ── Contrastive (InfoNCE) ────────────────────────────────────────────────
class ContrastiveLoss(nn.Module):
    """Symmetric InfoNCE-style contrastive loss with manifold distance.

    Negative distance is treated as the logit; in Euclidean space this is
    the usual NT-Xent up to a sign convention.
    """

    def __init__(self, manifold: ManifoldHead, *, temperature: float = 0.07) -> None:
        super().__init__()
        self.manifold = manifold
        self.temperature = temperature

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        # anchors, positives: [B, D] on-manifold.
        b = anchors.shape[0]
        d = self.manifold.pairwise_dist(anchors, positives)  # [B, B]
        # Logits: similarity-like, so we negate distance.
        logits = -d / self.temperature
        labels = torch.arange(b)
        # Symmetric loss.
        loss_a = nn.functional.cross_entropy(logits, labels)
        loss_b = nn.functional.cross_entropy(logits.t(), labels)
        return (loss_a + loss_b) * 0.5


# ── Triplet ─────────────────────────────────────────────────────────────
class TripletLoss(nn.Module):
    """Margin triplet loss: ``relu(d(a, p) - d(a, n) + margin)``."""

    def __init__(self, manifold: ManifoldHead, *, margin: float = 0.2) -> None:
        super().__init__()
        self.manifold = manifold
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_pos = self.manifold.dist(anchor, positive)
        d_neg = self.manifold.dist(anchor, negative)
        return (d_pos - d_neg + self.margin).clamp(min=0.0).mean()


# ── Neighbor consistency (SCAN) ──────────────────────────────────────────
class NeighborConsistencyLoss(nn.Module):
    """SCAN-style clustering loss (Van Gansbeke et al. 2020), manifold-aware.

    Holds ``n_clusters`` learnable prototypes as *tangent* vectors that are
    pushed onto the manifold via ``expmap0`` on every forward, so plain Adam
    keeps them valid on any geometry. Soft assignments are
    ``softmax(-dist(z, prototypes) / T)``.

    Loss = consistency + ``entropy_weight`` · negative-entropy:

    - **Consistency**: ``-log Σ_c p(anchor)_c · p(neighbor)_c`` — an anchor
      and its mined nearest neighbor should land in the same cluster.
    - **Entropy regularizer**: negative entropy of the *mean* assignment —
      penalizes degenerate solutions where every document maps to one
      cluster. This is the anti-collapse term.
    """

    def __init__(
        self,
        manifold: ManifoldHead,
        *,
        n_clusters: int,
        dim: int,
        temperature: float = 0.5,
        entropy_weight: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        g = torch.Generator().manual_seed(seed)
        # Small tangent init → prototypes start near the manifold origin
        # and spread out as training progresses.
        self.prototypes = nn.Parameter(torch.randn(n_clusters, dim, generator=g) * 0.05)

    def soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """``[B, D]`` on-manifold → ``[B, C]`` soft cluster assignment."""
        protos = self.manifold.expmap0(self.prototypes)
        d = self.manifold.pairwise_dist(z, protos)  # [B, C]
        return nn.functional.softmax(-d / self.temperature, dim=-1)

    def forward(self, anchors: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        pa = self.soft_assign(anchors)  # [B, C]
        pn = self.soft_assign(neighbors)  # [B, C]
        consistency = -torch.log((pa * pn).sum(dim=-1) + eps).mean()
        p_mean = pa.mean(dim=0)  # [C]
        neg_entropy = (p_mean * torch.log(p_mean + eps)).sum()
        return consistency + self.entropy_weight * neg_entropy


# ── VICReg regularizer ───────────────────────────────────────────────────
class VICRegRegularizer(nn.Module):
    """Variance + covariance anti-collapse penalty (Bardes et al. 2022).

    Operates on *tangent-space* coordinates (``logmap`` at the manifold
    origin) so the same code is correct on Euclidean, spherical, and
    hyperbolic geometry. Additive on top of any primary loss:

    - **Variance**: hinge ``relu(gamma - std(z_d))`` per dimension — keeps
      every embedding dimension alive.
    - **Covariance**: squared off-diagonal entries of the feature
      covariance — decorrelates dimensions so variance can't hide in one
      direction.
    """

    def __init__(
        self,
        manifold: ManifoldHead,
        *,
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # On-manifold → tangent at origin. Euclidean: identity (z - 0).
        origin = self.manifold.manifold_origin().to(z.dtype).unsqueeze(0).expand_as(z)
        t = self.manifold.logmap(origin, z)  # [B, D]
        b, d = t.shape
        if b < 2:
            return torch.zeros(())
        std = torch.sqrt(t.var(dim=0) + 1e-4)  # [D]
        var_loss = torch.relu(self.gamma - std).mean()
        centered = t - t.mean(dim=0, keepdim=True)
        cov = (centered.t() @ centered) / (b - 1)  # [D, D]
        off_diag = cov - torch.diag(torch.diagonal(cov))
        cov_loss = (off_diag**2).sum() / d
        return cast(torch.Tensor, self.var_weight * var_loss + self.cov_weight * cov_loss)


# ── Prototypical ─────────────────────────────────────────────────────────
class PrototypicalLoss(nn.Module):
    """Prototypical-network style classification with manifold distance.

    Logits = ``-distance`` to each class prototype; loss = cross-entropy.
    """

    def __init__(self, manifold: ManifoldHead) -> None:
        super().__init__()
        self.manifold = manifold

    def forward(
        self,
        embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        d = self.manifold.pairwise_dist(embeddings, prototypes)  # [B, C]
        return nn.functional.cross_entropy(-d, labels)
