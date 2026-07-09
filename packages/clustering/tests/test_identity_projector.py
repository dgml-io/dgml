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

"""Tests for the forced-identity projector path.

``ManifoldProjector(force_identity=True)`` must:

1. Use a parameter-free :class:`torch.nn.Identity` linear (no parameters
   at all), so its forward equals ``manifold.expmap0(x)`` directly.
2. Reject ``trainable=True`` — an identity has nothing to train.
3. Reject mismatched dims — a true identity cannot bridge them.

A companion test pins the schema-level mutual-exclusion guard between
``identity_projector`` and ``trainable_projector`` on
:class:`TrainingConfig`.
"""

from __future__ import annotations

import pytest
import torch
from clustering.config.schema import ManifoldConfig, TrainingConfig
from clustering.manifolds import ManifoldProjector, build_manifold
from torch import nn


def test_force_identity_uses_parameter_free_identity() -> None:
    m = build_manifold(ManifoldConfig(name="euclidean", dim=8, curvature=0.0))
    p = ManifoldProjector(m, input_dim=8, output_dim=8, force_identity=True)
    assert isinstance(p.linear, nn.Identity)
    # No trainable parameters anywhere on the projector.
    assert not any(True for _ in p.parameters())
    assert p._anchor is None


def test_force_identity_forward_equals_expmap0() -> None:
    m = build_manifold(ManifoldConfig(name="spherical", dim=8, curvature=1.0))
    p = ManifoldProjector(m, input_dim=8, output_dim=8, force_identity=True)
    x = torch.randn(4, 8) * 0.1
    assert torch.allclose(p(x), m.expmap0(x), atol=1e-6)


def test_force_identity_rejects_trainable() -> None:
    m = build_manifold(ManifoldConfig(name="euclidean", dim=8, curvature=0.0))
    with pytest.raises(ValueError, match="incompatible with"):
        ManifoldProjector(m, input_dim=8, output_dim=8, force_identity=True, trainable=True)


def test_force_identity_rejects_dim_mismatch() -> None:
    m = build_manifold(ManifoldConfig(name="euclidean", dim=8, curvature=0.0))
    with pytest.raises(ValueError, match="input_dim == output_dim"):
        ManifoldProjector(m, input_dim=16, output_dim=8, force_identity=True)


def test_training_config_identity_and_trainable_mutually_exclusive() -> None:
    # Each alone is fine.
    assert TrainingConfig(identity_projector=True).identity_projector is True
    assert TrainingConfig(trainable_projector=True).trainable_projector is True
    # Both together is a loud error.
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingConfig(identity_projector=True, trainable_projector=True)
