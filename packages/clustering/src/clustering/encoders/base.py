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

"""Encoder ABC, :class:`EncoderOutput` dataclass, and a name-keyed registry.

The pooled/tokens split is the single place in the codebase where the
multi-vector / single-vector distinction lives. Multi-vector consumers read
``tokens``; single-vector fusion variants consume ``pooled``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch

from clustering.config.schema import EncoderConfig

T = TypeVar("T")


@dataclass(frozen=True)
class EncoderOutput:
    """Result of an :meth:`Encoder.encode` call.

    Attributes:
        pooled: ``[B, D]`` per-sample embedding. Always present.
        tokens: ``[B, N, D]`` per-token (or per-patch) embeddings.
            ``None`` for single-vector encoders; populated by
            multi-vector encoders.
    """

    pooled: torch.Tensor
    tokens: torch.Tensor | None = None

    @property
    def is_multi_vector(self) -> bool:
        return self.tokens is not None

    @property
    def batch_size(self) -> int:
        return int(self.pooled.shape[0])

    @property
    def dim(self) -> int:
        return int(self.pooled.shape[-1])


class Encoder(ABC, Generic[T]):
    """Abstract encoder.

    Subclasses fill ``embedding_dim`` / ``multi_vector`` and implement
    :meth:`encode` for a specific input type ``T`` (``str`` for text,
    ``PIL.Image.Image`` for image).
    """

    embedding_dim: int
    multi_vector: bool

    @abstractmethod
    def encode(self, batch: Sequence[T]) -> EncoderOutput:
        """Encode ``batch`` → :class:`EncoderOutput`."""

    def __call__(self, batch: Sequence[T]) -> EncoderOutput:
        return self.encode(batch)


# ── Registry ─────────────────────────────────────────────────────────────
EncoderFactory = Callable[..., Encoder[Any]]
_REGISTRY: dict[str, EncoderFactory] = {}


def register_encoder(name: str) -> Callable[[EncoderFactory], EncoderFactory]:
    """Decorator to register an encoder factory under ``name``."""

    def deco(fn: EncoderFactory) -> EncoderFactory:
        if name in _REGISTRY:
            raise ValueError(f"Encoder {name!r} is already registered.")
        _REGISTRY[name] = fn
        return fn

    return deco


def build_encoder(
    cfg: EncoderConfig, *, device: str = "auto", cache_dir: Path | None = None
) -> Encoder[Any]:
    """Look up the factory for ``cfg.name`` and instantiate it.

    Args:
        cfg: Validated :class:`EncoderConfig`.
        device: Device spec (``auto``, ``cuda``, ``cuda:N``, ``mps``, ``cpu``).
            Forwarded to factories that care; ignored by ``dummy``.
        cache_dir: If set, wrap the encoder in a :class:`CachingEncoder` that
            persists embeddings under this directory keyed by a config
            fingerprint + per-input content hash. ``None`` (the default)
            disables caching entirely.

    Raises:
        KeyError: If ``cfg.name`` has no registered factory.
    """
    if cfg.name not in _REGISTRY:
        raise KeyError(f"Unknown encoder {cfg.name!r}. Registered: {sorted(_REGISTRY)}")
    encoder = _REGISTRY[cfg.name](cfg, device=device)
    if cache_dir is not None:
        # Imported lazily to avoid a circular import (caching imports this module).
        from clustering.encoders.caching import CachingEncoder, encoder_fingerprint

        return CachingEncoder(encoder, cache_dir, encoder_fingerprint(cfg))
    return encoder


def registered_encoders() -> list[str]:
    return sorted(_REGISTRY)
