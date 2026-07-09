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

"""``concat_norm`` fusion — parameter-free L2-normalized weighted concat.

Unlike ``late_concat`` / ``gated`` / ``cross_attention`` (which interpose
randomly-initialized ``nn.Linear`` layers and therefore only make sense once
trained), this fusion has **no parameters**: it L2-normalizes each modality's
pooled vector, scales text by ``1 - image_weight`` and image by ``image_weight``,
and concatenates. That makes it the multimodal analogue of ``none`` for the
untrained S1 baseline — a clean way to blend text semantics with page layout
without inserting a random projection that scrambles the embeddings.

The single ``image_weight`` knob (see :class:`FusionConfig`) trades the two
modalities off against each other:

* ``0.0`` ⇒ text only (image block is zeroed) — equivalent to ``none`` text.
* ``1.0`` ⇒ image only — equivalent to ``none`` image.
* ``0.5`` ⇒ equal-norm blend.

Output dim is ``text_dim + image_dim``: both blocks are kept, so downstream
distance metrics see each modality in its own subspace rather than summed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from clustering.config.schema import FusionConfig
from clustering.encoders.base import EncoderOutput
from clustering.fusion.base import Fusion, FusionOutput, register_fusion


class ConcatNormFusion(Fusion):
    """L2-normalize each modality, weight, and concatenate. No parameters."""

    def __init__(self, cfg: FusionConfig, *, text_dim: int, image_dim: int) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.output_dim = text_dim + image_dim
        # Clamp into [0, 1] so the weights stay a convex blend.
        self.image_weight = float(min(max(cfg.image_weight, 0.0), 1.0))
        self.text_weight = 1.0 - self.image_weight

    def forward(self, text: EncoderOutput, image: EncoderOutput) -> FusionOutput:
        # Normalize per modality so neither dominates the concatenation purely
        # by having a larger native norm, then apply the blend weights.
        t = nn.functional.normalize(text.pooled, dim=-1) * self.text_weight
        i = nn.functional.normalize(image.pooled, dim=-1) * self.image_weight
        fused = torch.cat([t, i], dim=-1)
        return FusionOutput(pooled=fused)


@register_fusion("concat_norm")
def _factory(cfg: FusionConfig, *, text_dim: int, image_dim: int, **_: Any) -> Fusion:
    return ConcatNormFusion(cfg, text_dim=text_dim, image_dim=image_dim)
