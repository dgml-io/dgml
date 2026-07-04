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

"""``cross_attention`` fusion — symmetric cross-attention between modalities.

Both modalities are projected to a shared dim, then each attends to the
other using torch's ``MultiheadAttention``. The two attended outputs are
concatenated and projected to the final ``output_dim``.

For pooled inputs (shape ``[B, D]``) we use a sequence length of 1 — the
attention math degenerates to a learned weighted mix, which is what we
want for fixed-size single-vector embeddings.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from clustering.config.schema import FusionConfig
from clustering.encoders.base import EncoderOutput
from clustering.fusion.base import Fusion, FusionOutput, register_fusion


class CrossAttentionFusion(Fusion):
    def __init__(self, cfg: FusionConfig, *, text_dim: int, image_dim: int) -> None:
        super().__init__()
        d = cfg.hidden_dim
        self.t_proj = nn.Linear(text_dim, d)
        self.i_proj = nn.Linear(image_dim, d)
        self.t2i = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.i2t = nn.MultiheadAttention(
            embed_dim=d, num_heads=cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d, cfg.output_dim),
            nn.GELU(),
        )
        self.output_dim = cfg.output_dim

    def forward(self, text: EncoderOutput, image: EncoderOutput) -> FusionOutput:
        t = self.t_proj(text.pooled).unsqueeze(1)  # [B, 1, D]
        i = self.i_proj(image.pooled).unsqueeze(1)  # [B, 1, D]
        t_att, _ = self.t2i(t, i, i)
        i_att, _ = self.i2t(i, t, t)
        fused = torch.cat([t_att.squeeze(1), i_att.squeeze(1)], dim=-1)
        return FusionOutput(pooled=self.head(fused))


@register_fusion("cross_attention")
def _factory(cfg: FusionConfig, *, text_dim: int, image_dim: int, **_: Any) -> Fusion:
    return CrossAttentionFusion(cfg, text_dim=text_dim, image_dim=image_dim)
