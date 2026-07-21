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

"""S5 name+support prototype blend: opt-in, behaviour-preserving at alpha=0/None.

Blending the encoded category-name prototype with the support mean is a strong
few-shot prior (measured +0.02 to +0.09 accuracy at K=1-2 across three corpora).
The knob defaults off; these tests pin the blend math, the validator, and that
the flag actually reaches S5's prototypes end-to-end.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from clustering.config.schema import Config
from clustering.data.datasets import DocumentDataset, DocumentRecord
from clustering.scenarios import build_scenario
from clustering.scenarios.s5_full_supervised import S5FullSupervised
from PIL import Image

_DIM = 16


def _unit(x: torch.Tensor) -> torch.Tensor:
    unit: torch.Tensor = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return unit


# ── blend math ────────────────────────────────────────────────────────────
def test_blend_alpha_endpoints_and_middle() -> None:
    support = torch.tensor([[3.0, 0.0], [0.0, 5.0]])  # deliberately non-unit norms
    name = torch.tensor([[0.0, 2.0], [4.0, 0.0]])
    a0 = S5FullSupervised._blend_prototypes(support, name, 0.0)
    a1 = S5FullSupervised._blend_prototypes(support, name, 1.0)
    amid = S5FullSupervised._blend_prototypes(support, name, 0.5)
    assert torch.allclose(a0, _unit(support), atol=1e-6), "alpha=0 must be the support direction"
    assert torch.allclose(a1, _unit(name), atol=1e-6), "alpha=1 must be the name direction"
    assert torch.allclose(amid.norm(dim=-1), torch.ones(2), atol=1e-6), "blend stays unit-norm"
    # midpoint sits strictly between the two endpoints
    assert not torch.allclose(amid, a0, atol=1e-3) and not torch.allclose(amid, a1, atol=1e-3)


# ── config validation ─────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
def test_blend_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="name_prototype_blend"):
        _config(blend=bad)


# ── end-to-end: the flag reaches S5's prototypes ──────────────────────────
class _Labeled(DocumentDataset):
    def __init__(self, labels: list[str | None]) -> None:
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> DocumentRecord:
        label = self._labels[index]
        return DocumentRecord(
            doc_id=f"doc_{index}",
            label=label,
            image=Image.new("RGB", (8, 8), color=(index * 20 % 255, 0, 0)),
            text=f"{label or 'x'} sample text {index}",
            thumbnail_path=None,
        )


def _config(blend: float | None, manifold: str = "euclidean") -> Config:
    scenario: dict[str, Any] = {"name": "s5", "known_categories": ["A", "B"], "n_shots": 2}
    if blend is not None:
        scenario["name_prototype_blend"] = blend
    raw: dict[str, Any] = {
        "scenario": scenario,
        "encoder_text": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "encoder_image": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "fusion": {"name": "late_concat", "output_dim": 2 * _DIM},
        "manifold": {"name": manifold, "dim": 2 * _DIM},
        "training": {"epochs": 0},
        "logger": {"name": "none"},
        "corpus": {"root": "."},
        "device": "cpu",
        "seed": 0,
    }
    return Config.model_validate(raw)


def test_blend_off_by_default_preserves_support_only() -> None:
    support = _Labeled(["A", "A", "B", "B"])
    unknown = _Labeled([None] * 4)
    res = build_scenario(_config(blend=None)).fit_predict(unknown, support)
    assert res.metadata["prototype_source"] == "support_mean"


def test_blend_on_changes_prototype_source_and_runs() -> None:
    support = _Labeled(["A", "A", "B", "B"])
    unknown = _Labeled([None] * 4)
    res = build_scenario(_config(blend=0.5)).fit_predict(unknown, support)
    assert res.metadata["prototype_source"] == "name_support_blend(alpha=0.5)"
    assert all(p in ("A", "B") for p in res.predictions)


def test_blend_rejects_non_euclidean_manifold() -> None:
    """The direction-space blend is only sound on a euclidean head — reject others."""
    support = _Labeled(["A", "A", "B", "B"])
    unknown = _Labeled([None] * 4)
    scenario = build_scenario(_config(blend=0.5, manifold="spherical"))
    with pytest.raises(ValueError, match="euclidean"):
        scenario.fit_predict(unknown, support)
