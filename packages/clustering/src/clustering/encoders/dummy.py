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

"""Deterministic hash-based dummy encoder.

Lets the rest of the framework be tested without downloading any HF weights.
Same input bytes → same output vector, modulo embedding_dim. Distinct
inputs almost-certainly produce distinct outputs (sha256 collisions are
not a real concern at our scale).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder

_TOKENS_PER_DOC = 8


def _hash_to_vector(payload: bytes, dim: int) -> np.ndarray[Any, Any]:
    """Map ``payload`` to a deterministic ``dim``-d float32 vector."""
    out = bytearray()
    chunk = payload
    while len(out) < dim * 4:
        chunk = hashlib.sha256(chunk).digest()
        out.extend(chunk)
    arr = np.frombuffer(bytes(out[: dim * 4]), dtype=np.uint32).astype(np.float32)
    # Center to roughly [-1, 1) so cosine similarity isn't degenerate.
    arr = arr / np.float32(2**31) - np.float32(1.0)
    return arr.copy()  # detach from the read-only buffer view


def _payload(x: Any) -> bytes:
    if isinstance(x, str):
        return x.encode("utf-8")
    if isinstance(x, Image.Image):
        return x.tobytes() + f"|size={x.size}".encode()
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    return repr(x).encode("utf-8")


class DummyEncoder(Encoder[Any]):
    """Hash-based encoder. Accepts text *or* images interchangeably."""

    def __init__(self, cfg: EncoderConfig) -> None:
        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim
        self.multi_vector = cfg.multi_vector

    def encode(self, batch: Sequence[Any]) -> EncoderOutput:
        items = list(batch)
        pooled = np.stack([_hash_to_vector(_payload(x), self.embedding_dim) for x in items])
        tokens: torch.Tensor | None = None
        if self.multi_vector:
            tokens_np = np.stack(
                [
                    np.stack(
                        [
                            _hash_to_vector(_payload(x) + f"@{i}".encode(), self.embedding_dim)
                            for i in range(_TOKENS_PER_DOC)
                        ]
                    )
                    for x in items
                ]
            )
            tokens = torch.from_numpy(tokens_np)
        return EncoderOutput(pooled=torch.from_numpy(pooled), tokens=tokens)


@register_encoder("dummy")
def _factory(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    # device is irrelevant for the dummy encoder; accept it for API uniformity.
    del device
    return DummyEncoder(cfg)
