"""Text + image + multi-vector (ColPali) encoders.

All encoders implement :class:`Encoder` returning :class:`EncoderOutput`
``(pooled, tokens?)``. Single-vector encoders leave ``tokens=None``; ColPali
fills both.

Importing this module registers every built-in encoder with the registry.
"""

from __future__ import annotations

# Side-effect imports — each module decorates its factory with `register_encoder`.
# `text` registers every SentenceTransformer-compatible name:
# st_minilm, e5, bge, gte, stella, jina.
from clustering.encoders import (
    colpali,  # noqa: F401  (registers "colpali")
    dummy,  # noqa: F401  (registers "dummy")
    image,  # noqa: F401  (registers "dit", "vit", "donut")
    lexical,  # noqa: F401  (registers "tfidf")
    qwen3_vl_embedding,  # noqa: F401  (registers "qwen3_vl_embedding")
    qwen_vl,  # noqa: F401  (registers "qwen_vl")
    siglip,  # noqa: F401  (registers "siglip")
    text,  # noqa: F401
)
from clustering.encoders.base import (
    Encoder,
    EncoderOutput,
    build_encoder,
    register_encoder,
    registered_encoders,
)
from clustering.encoders.caching import CachingEncoder, encoder_fingerprint
from clustering.encoders.mrl import SweepResult, mrl_dimension_sweep, mrl_truncate

__all__ = [
    "CachingEncoder",
    "Encoder",
    "EncoderOutput",
    "SweepResult",
    "build_encoder",
    "encoder_fingerprint",
    "mrl_dimension_sweep",
    "mrl_truncate",
    "register_encoder",
    "registered_encoders",
]
