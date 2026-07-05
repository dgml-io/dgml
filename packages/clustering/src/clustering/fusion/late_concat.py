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

"""``late_concat`` fusion — concatenate pooled embeddings, then a 2-layer MLP."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from clustering.config.schema import FusionConfig
from clustering.encoders.base import EncoderOutput
from clustering.fusion.base import Fusion, FusionOutput, register_fusion


class LateConcatFusion(Fusion):
    """Concat + 2-layer MLP head over pooled embeddings."""

    def __init__(self, cfg: FusionConfig, *, text_dim: int, image_dim: int) -> None:
        super().__init__()
        in_dim = text_dim + image_dim
        self.output_dim = cfg.output_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, text: EncoderOutput, image: EncoderOutput) -> FusionOutput:
        x = torch.cat([text.pooled, image.pooled], dim=-1)
        return FusionOutput(pooled=self.mlp(x))


@register_fusion("late_concat")
def _factory(cfg: FusionConfig, *, text_dim: int, image_dim: int, **_: Any) -> Fusion:
    return LateConcatFusion(cfg, text_dim=text_dim, image_dim=image_dim)
