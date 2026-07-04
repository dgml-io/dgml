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

"""``late_interaction`` fusion — ColBERT-style MaxSim, no learned parameters.

The multi-vector representation flows through unchanged; the actual MaxSim
scoring happens at categorization time (the scenario calls
:func:`clustering.fusion.base.maxsim` between document tokens and category
prototype tokens).

``pooled`` is filled with the mean over tokens so the manifold head still
has a sensible single-vector input for projection and visualisation.
"""

from __future__ import annotations

from typing import Any

from clustering.config.schema import FusionConfig
from clustering.encoders.base import EncoderOutput
from clustering.fusion.base import Fusion, FusionOutput, register_fusion


class LateInteractionFusion(Fusion):
    def __init__(self, cfg: FusionConfig, *, text_dim: int, image_dim: int) -> None:
        super().__init__()
        del cfg
        # Output dim follows whichever side carries the multi-vector tokens.
        # Image side is the typical ColPali path; text-side dim is the fallback.
        self.output_dim = image_dim or text_dim

    def forward(self, text: EncoderOutput, image: EncoderOutput) -> FusionOutput:
        if image.tokens is not None:
            tokens = image.tokens
        elif text.tokens is not None:
            tokens = text.tokens
        else:
            raise NotImplementedError(
                "fusion=late_interaction requires at least one encoder with "
                "multi_vector=True (e.g. encoder_image=colpali). "
                "Got text.tokens=None and image.tokens=None."
            )
        pooled = tokens.mean(dim=1)
        return FusionOutput(pooled=pooled, tokens=tokens)


@register_fusion("late_interaction")
def _factory(cfg: FusionConfig, *, text_dim: int, image_dim: int, **_: Any) -> Fusion:
    return LateInteractionFusion(cfg, text_dim=text_dim, image_dim=image_dim)
