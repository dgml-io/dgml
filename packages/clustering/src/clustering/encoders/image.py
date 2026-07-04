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

"""HF-Vision image encoders: DiT, ViT, Donut.

All three share an ``AutoImageProcessor + AutoModel`` backbone; we pool the
CLS token if present, else mean-pool ``last_hidden_state``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder
from clustering.utils import resolve_device


class HFVisionEncoder(Encoder[Image.Image]):
    """Generic AutoModel-based image encoder (DiT / ViT / Donut)."""

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
        try:
            from transformers import AutoConfig, AutoImageProcessor, AutoModel
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
        # Donut is an encoder-decoder; transformers 5.x dropped it from the
        # AutoModel mapping (VisionEncoderDecoderConfig is "Unrecognized"), so
        # load its vision encoder directly. ViT / DiT stay on AutoModel.
        loaded: Any
        if getattr(AutoConfig.from_pretrained(cfg.model_id), "model_type", "") == (
            "vision-encoder-decoder"
        ):
            from transformers import VisionEncoderDecoderModel

            loaded = VisionEncoderDecoderModel.from_pretrained(cfg.model_id).encoder
        else:
            loaded = AutoModel.from_pretrained(cfg.model_id)
        self.model: Any = loaded.to(self.device).eval()

    @torch.no_grad()
    def encode(self, batch: Sequence[Image.Image]) -> EncoderOutput:
        inputs = self.processor(images=list(batch), return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        last = outputs.last_hidden_state  # [B, N, D]
        pooled = last[:, 0, :] if last.shape[1] > 1 else last.mean(dim=1)  # CLS-style
        return EncoderOutput(pooled=pooled.detach().cpu())


@register_encoder("dit")
def _factory_dit(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return HFVisionEncoder(cfg, device=device)


@register_encoder("vit")
def _factory_vit(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return HFVisionEncoder(cfg, device=device)


@register_encoder("donut")
def _factory_donut(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return HFVisionEncoder(cfg, device=device)
