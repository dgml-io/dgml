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

"""ColPali multi-vector encoder (PaliGemma-based).

Emits one embedding per image patch. ``pooled`` is the mean of the patch
vectors so pooled-fusion variants still work; ``tokens`` carries the full
multi-vector grid for ``fusion=late_interaction``.

Notes:
    The colpali-engine library has shifted its public API a few times.
    This adapter targets the ``ColPali`` + ``ColPaliProcessor`` surface that
    has been stable since the 0.3 series. If you pin a different version,
    adjust the imports here — the rest of the framework consumes only the
    common :class:`EncoderOutput` and is unaffected.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder
from clustering.utils import resolve_device


class ColPaliEncoder(Encoder[Image.Image]):
    """ColPali image encoder. Always multi-vector."""

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
        except ImportError as exc:
            raise ImportError(
                "colpali-engine is not installed. Add the 'colpali' extra with: "
                "`uv sync --extra colpali`."
            ) from exc
        if cfg.model_id is None:
            raise ValueError("Encoder 'colpali' requires a model_id.")
        if not cfg.multi_vector:
            raise ValueError(
                "ColPali is a multi-vector encoder. Set multi_vector=true in "
                "configs/encoder_image/colpali.yaml (it is by default)."
            )

        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim
        self.multi_vector = True
        info = resolve_device(device)
        self.device = info.torch_device

        self.processor = ColPaliProcessor.from_pretrained(cfg.model_id)
        self.model = ColPali.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float32,
            device_map=self.device,
        ).eval()

    @torch.no_grad()
    def encode(self, batch: Sequence[Image.Image]) -> EncoderOutput:
        items = list(batch)
        inputs = self.processor.process_images(items).to(self.device)
        tokens = self.model(**inputs)  # [B, N_patches, D]
        # Mean-pool patches → a usable pooled vector for non-late-interaction fusion.
        pooled = tokens.mean(dim=1)
        return EncoderOutput(
            pooled=pooled.detach().cpu(),
            tokens=tokens.detach().cpu(),
        )


@register_encoder("colpali")
def _factory(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return ColPaliEncoder(cfg, device=device)
