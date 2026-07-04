"""Fusion — Research Axis #1.

Importing this module registers all fusion variants: ``none``,
``concat_norm``, ``late_concat``, ``cross_attention``, ``gated``,
``late_interaction``.
"""

from __future__ import annotations

# Side-effect imports so the registry is populated.
from clustering.fusion import (
    concat_norm,  # noqa: F401
    cross_attention,  # noqa: F401
    gated,  # noqa: F401
    late_concat,  # noqa: F401
    late_interaction,  # noqa: F401
    none_,  # noqa: F401
)
from clustering.fusion.base import (
    Fusion,
    FusionOutput,
    build_fusion,
    maxsim,
    register_fusion,
    registered_fusions,
)

__all__ = [
    "Fusion",
    "FusionOutput",
    "build_fusion",
    "maxsim",
    "register_fusion",
    "registered_fusions",
]
