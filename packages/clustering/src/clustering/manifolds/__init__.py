"""Manifolds — Research Axis #2.

Importing this module registers Euclidean, Spherical, Hyperbolic (Poincaré
ball), and Product manifolds. Forward math is implemented in pure torch in
each subclass; for Riemannian optimization wrap parameters via geoopt in
your training code.
"""

from __future__ import annotations

from clustering.manifolds.base import (
    ManifoldHead,
    build_manifold,
    register_manifold,
    registered_manifolds,
)
from clustering.manifolds.euclidean import EuclideanHead
from clustering.manifolds.hyperbolic import HyperbolicHead
from clustering.manifolds.losses import (
    ContrastiveLoss,
    NeighborConsistencyLoss,
    PrototypicalLoss,
    TripletLoss,
    VICRegRegularizer,
)
from clustering.manifolds.product import ProductHead
from clustering.manifolds.projector import ManifoldProjector
from clustering.manifolds.spherical import SphericalHead
from clustering.manifolds.training import (
    train_fusion_projector,
    train_projector,
    train_projector_cross_modal,
)

# Side-effect imports already happened above (each module registers itself).

__all__ = [
    "ContrastiveLoss",
    "EuclideanHead",
    "HyperbolicHead",
    "ManifoldHead",
    "ManifoldProjector",
    "NeighborConsistencyLoss",
    "ProductHead",
    "PrototypicalLoss",
    "SphericalHead",
    "TripletLoss",
    "VICRegRegularizer",
    "build_manifold",
    "register_manifold",
    "registered_manifolds",
    "train_fusion_projector",
    "train_projector",
    "train_projector_cross_modal",
]
