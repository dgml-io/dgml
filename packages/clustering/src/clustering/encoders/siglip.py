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

"""SigLIP image encoder (sigmoid-loss CLIP; Zhai et al. 2023).

SigLIP has no ``CLS`` token — its vision tower ends in an attention-pooling
head — so the generic CLS/mean pooling in :mod:`clustering.encoders.image`
would silently take the wrong vector. We therefore load the vision tower
directly and use its attention-pooled ``pooler_output`` as ``pooled``.

When ``multi_vector=True`` the per-patch ``last_hidden_state`` is also
returned for ``fusion=late_interaction``. SigLIP uses a fixed input
resolution, so every image yields the same patch grid and the tokens stack
cleanly into ``[B, N, D]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder
from clustering.utils import resolve_device


class SiglipEncoder(Encoder[Image.Image]):
    """SigLIP vision-tower image encoder.

    Single-vector by default; set ``multi_vector=true`` to additionally emit
    per-patch tokens for late-interaction fusion.
    """

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
        try:
            from transformers import AutoImageProcessor, SiglipVisionModel
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Add the 'encoders' extra with: "
                "`uv sync --extra encoders`."
            ) from exc
        if cfg.model_id is None:
            raise ValueError(f"Encoder {cfg.name!r} requires a model_id.")

        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim
        self.multi_vector = cfg.multi_vector
        info = resolve_device(device)
        self.device = info.torch_device

        self.processor = AutoImageProcessor.from_pretrained(cfg.model_id)  # type: ignore[no-untyped-call]
        # `from_pretrained` loses its concrete return type under transformers 5.x's
        # typing; route through Any so `.to(...).eval()` type-checks (same as qwen_vl).
        loaded: Any = SiglipVisionModel.from_pretrained(cfg.model_id)
        self.model: Any = loaded.to(self.device).eval()

    @torch.no_grad()
    def encode(self, batch: Sequence[Image.Image]) -> EncoderOutput:
        inputs = self.processor(images=list(batch), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        pooled = outputs.pooler_output  # [B, D] — attention-pooled, no CLS token
        tokens = outputs.last_hidden_state if self.multi_vector else None  # [B, N, D]
        return EncoderOutput(
            pooled=pooled.detach().cpu(),
            tokens=None if tokens is None else tokens.detach().cpu(),
        )


@register_encoder("siglip")
def _factory(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return SiglipEncoder(cfg, device=device)
