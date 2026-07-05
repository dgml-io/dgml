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

"""Fusion ABC, output type, and a name-keyed registry.

All fusions take ``(text: EncoderOutput, image: EncoderOutput)`` and return
a :class:`FusionOutput` whose ``pooled`` field is the downstream embedding.
``late_interaction`` is the only variant that *also* fills ``tokens`` — its
multi-vector representation is consumed by the scorer at categorization time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from clustering.config.schema import FusionConfig
from clustering.encoders.base import EncoderOutput


@dataclass(frozen=True)
class FusionOutput:
    """Output of a :class:`Fusion` forward pass."""

    pooled: torch.Tensor  # [B, D_out]
    tokens: torch.Tensor | None = None  # [B, N, D] for late_interaction

    @property
    def is_multi_vector(self) -> bool:
        return self.tokens is not None


class Fusion(nn.Module, ABC):
    """Abstract fusion module."""

    output_dim: int

    @abstractmethod
    def forward(self, text: EncoderOutput, image: EncoderOutput) -> FusionOutput:
        """Combine the two encoder outputs into a single fused representation."""


FusionFactory = Callable[..., Fusion]
_REGISTRY: dict[str, FusionFactory] = {}


def register_fusion(name: str) -> Callable[[FusionFactory], FusionFactory]:
    """Decorator: register a fusion factory under ``name``."""

    def deco(fn: FusionFactory) -> FusionFactory:
        if name in _REGISTRY:
            raise ValueError(f"Fusion {name!r} is already registered.")
        _REGISTRY[name] = fn
        return fn

    return deco


def build_fusion(
    cfg: FusionConfig,
    *,
    text_dim: int,
    image_dim: int,
) -> Fusion:
    """Look up the factory for ``cfg.name`` and instantiate it.

    Args:
        cfg: Validated :class:`FusionConfig`.
        text_dim: Output dim of the text encoder (``encoder_text.embedding_dim``).
        image_dim: Output dim of the image encoder.
    """
    if cfg.name not in _REGISTRY:
        raise KeyError(f"Unknown fusion {cfg.name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[cfg.name](cfg, text_dim=text_dim, image_dim=image_dim)


def registered_fusions() -> list[str]:
    return sorted(_REGISTRY)


def _maxsim(query: torch.Tensor, doc: torch.Tensor) -> torch.Tensor:
    """ColBERT-style MaxSim between query and document multi-vector reps.

    Args:
        query: ``[..., Nq, D]`` query token embeddings.
        doc:   ``[..., Nd, D]`` document token embeddings.

    Returns:
        Score of shape ``[...]`` — for each query token, take the max
        similarity over document tokens, then sum across query tokens.
    """
    sim = query @ doc.transpose(-1, -2)  # [..., Nq, Nd]
    per_query = sim.amax(dim=-1)  # [..., Nq]   max over doc tokens
    return per_query.sum(dim=-1)  # [...]       sum over query tokens


# Re-exported for callers that consume tokens directly (e.g. categorization).
maxsim = _maxsim
