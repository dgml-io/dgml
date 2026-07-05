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

"""Tests for the parameter-free `none` fusion modality switch and for
training the fusion module jointly with the projector.

Covered contracts:

1. ``none`` fusion passes the configured modality through unchanged and
   reports that modality's dimensionality.
2. The :class:`_FusionProjector` pipeline maps stacked per-modality
   features onto the manifold with the expected shape.
3. :func:`train_fusion_projector` actually updates the fusion's weights
   (the bug we're guarding against: the fusion staying at its random
   init because only the projector was ever optimized).
"""

from __future__ import annotations

import torch
from clustering.config.schema import FusionConfig, ManifoldConfig, TrainingConfig
from clustering.encoders.base import EncoderOutput
from clustering.fusion import build_fusion
from clustering.manifolds import ManifoldProjector, build_manifold, train_fusion_projector
from clustering.manifolds.training import _FusionProjector


def test_none_fusion_prefers_configured_modality() -> None:
    text = EncoderOutput(pooled=torch.zeros(2, 5))
    image = EncoderOutput(pooled=torch.ones(2, 7))

    img_fusion = build_fusion(
        FusionConfig(name="none", prefer_modality="image"), text_dim=5, image_dim=7
    )
    assert img_fusion.output_dim == 7
    assert torch.allclose(img_fusion(text, image).pooled, image.pooled)

    txt_fusion = build_fusion(
        FusionConfig(name="none", prefer_modality="text"), text_dim=5, image_dim=7
    )
    assert txt_fusion.output_dim == 5
    assert torch.allclose(txt_fusion(text, image).pooled, text.pooled)


def test_fusion_projector_pipeline_forward_shape() -> None:
    m = build_manifold(ManifoldConfig(name="euclidean", dim=8, curvature=0.0))
    fusion = build_fusion(FusionConfig(name="late_concat", output_dim=8), text_dim=6, image_dim=4)
    projector = ManifoldProjector(m, input_dim=8, output_dim=8, force_identity=True)
    pipeline = _FusionProjector(fusion, projector, text_dim=6)

    stacked = torch.randn(3, 10)  # 6 (text) + 4 (image)
    out = pipeline(stacked)
    assert out.shape == (3, 8)


def _toy_two_class(
    n: int = 4, dt: int = 6, di: int = 4
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    torch.manual_seed(0)
    text = torch.randn(2 * n, dt) * 0.1
    image = torch.randn(2 * n, di) * 0.1
    text[:n, 0] += 1.0
    image[n:, 0] += 1.0
    return text, image, ["A"] * n + ["B"] * n


def test_train_fusion_projector_updates_fusion_weights() -> None:
    m = build_manifold(ManifoldConfig(name="euclidean", dim=8, curvature=0.0))
    fusion = build_fusion(FusionConfig(name="late_concat", output_dim=8), text_dim=6, image_dim=4)
    projector = ManifoldProjector(m, input_dim=8, output_dim=8, trainable=True)
    text, image, labels = _toy_two_class()

    before = fusion.mlp[0].weight.detach().clone()  # type: ignore[index, union-attr]
    cfg = TrainingConfig(epochs=3, loss="prototypical", lr=1e-1, trainable_fusion=True)
    history = train_fusion_projector(fusion, projector, text, image, labels, cfg=cfg)

    assert len(history) == 3
    after = fusion.mlp[0].weight.detach()  # type: ignore[index, union-attr]
    assert not torch.allclose(before, after, atol=1e-7), "fusion weights did not update"


def test_train_fusion_projector_noop_without_epochs() -> None:
    m = build_manifold(ManifoldConfig(name="euclidean", dim=8, curvature=0.0))
    fusion = build_fusion(FusionConfig(name="late_concat", output_dim=8), text_dim=6, image_dim=4)
    projector = ManifoldProjector(m, input_dim=8, output_dim=8, trainable=True)
    text, image, labels = _toy_two_class()
    cfg = TrainingConfig(epochs=0, trainable_fusion=True)
    assert train_fusion_projector(fusion, projector, text, image, labels, cfg=cfg) == []
