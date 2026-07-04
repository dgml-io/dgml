"""``gated`` fusion — sigmoid gates weight each modality per-sample.

    s = sigmoid(W · [t; i])
    fused = s · t + (1 - s) · i

where ``t`` and ``i`` are linear projections of the pooled text/image
embeddings into a shared ``output_dim``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from clustering.config.schema import FusionConfig
from clustering.encoders.base import EncoderOutput
from clustering.fusion.base import Fusion, FusionOutput, register_fusion


class GatedFusion(Fusion):
    def __init__(self, cfg: FusionConfig, *, text_dim: int, image_dim: int) -> None:
        super().__init__()
        d = cfg.output_dim
        self.t_proj = nn.Linear(text_dim, d)
        self.i_proj = nn.Linear(image_dim, d)
        self.gate = nn.Linear(2 * d, 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.output_dim = d

    def forward(self, text: EncoderOutput, image: EncoderOutput) -> FusionOutput:
        t = self.dropout(self.t_proj(text.pooled))
        i = self.dropout(self.i_proj(image.pooled))
        sigma = torch.sigmoid(self.gate(torch.cat([t, i], dim=-1)))
        fused = sigma * t + (1.0 - sigma) * i
        return FusionOutput(pooled=fused)


@register_fusion("gated")
def _factory(cfg: FusionConfig, *, text_dim: int, image_dim: int, **_: Any) -> Fusion:
    return GatedFusion(cfg, text_dim=text_dim, image_dim=image_dim)
