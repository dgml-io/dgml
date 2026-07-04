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

"""``none`` fusion — best-single-modality baseline.

Picks one modality and passes its ``pooled`` (and ``tokens``, if present)
through unchanged. The chosen modality is configured via
``fusion.prefer_modality`` (default: ``image``, since image-based document
categorization is the central use case). To evaluate the *other* single
modality, run the same scenario again with ``fusion.prefer_modality=text``.
"""

from __future__ import annotations

from typing import Any

from clustering.config.schema import FusionConfig
from clustering.encoders.base import EncoderOutput
from clustering.fusion.base import Fusion, FusionOutput, register_fusion


class NoneFusion(Fusion):
    """Identity fusion — no parameters."""

    def __init__(self, cfg: FusionConfig, *, text_dim: int, image_dim: int) -> None:
        super().__init__()
        self.prefer: str = cfg.prefer_modality
        self.output_dim = image_dim if self.prefer == "image" else text_dim

    def forward(self, text: EncoderOutput, image: EncoderOutput) -> FusionOutput:
        chosen = image if self.prefer == "image" else text
        return FusionOutput(pooled=chosen.pooled, tokens=chosen.tokens)


@register_fusion("none")
def _factory(cfg: FusionConfig, *, text_dim: int, image_dim: int, **_: Any) -> Fusion:
    return NoneFusion(cfg, text_dim=text_dim, image_dim=image_dim)
