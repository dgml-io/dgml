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

"""Multi-page image pooling: opt-in, and correct when on.

``scenario.pooling_pages`` mean-pools the first N page renders on the image
side. Default 1 = page-1 only (unchanged). Measured to help ambiguous-cover
multi-page corpora (+0.10 on discovery2) and hurt header-discriminative forms,
hence opt-in. These tests pin: (a) the validator, (b) pool=1 is exactly the
single-image path, (c) pool=2 is the mean of the two page embeddings.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from clustering.config.schema import Config
from clustering.data.datasets import DocumentRecord
from clustering.scenarios import build_scenario
from PIL import Image

_DIM = 16


def _config(pooling_pages: int) -> Config:
    raw: dict[str, Any] = {
        "scenario": {"name": "s1", "k_clusters": 2, "pooling_pages": pooling_pages},
        "encoder_text": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "encoder_image": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "fusion": {"name": "late_concat", "output_dim": 2 * _DIM},
        "manifold": {"name": "euclidean", "dim": 2 * _DIM},
        "training": {"epochs": 0},
        "logger": {"name": "none"},
        "corpus": {"root": "."},
        "device": "cpu",
        "seed": 0,
    }
    return Config.model_validate(raw)


def _rgb(seed: int) -> Image.Image:
    return Image.new("RGB", (8, 8), color=(seed * 37 % 255, seed * 91 % 255, seed * 13 % 255))


def _records() -> list[DocumentRecord]:
    out = []
    for i in range(3):
        p1, p2 = _rgb(i), _rgb(i + 100)  # two visibly different pages per doc
        out.append(
            DocumentRecord(
                doc_id=f"d{i}",
                label=None,
                image=p1,
                text="t",
                thumbnail_path=None,
                page_images=(p1, p2),
            )
        )
    return out


def test_pooling_pages_must_be_positive() -> None:
    with pytest.raises(ValueError, match="pooling_pages"):
        _config(0)


def test_pool_one_is_the_single_image_path() -> None:
    scenario = build_scenario(_config(1))
    recs = _records()
    got = scenario._encode_images(recs).pooled
    want = scenario.image_encoder.encode([r.image for r in recs]).pooled
    assert torch.allclose(got, want, atol=1e-6), "pool=1 must equal encoding page 1 only"


def test_pool_two_is_mean_of_both_pages() -> None:
    scenario = build_scenario(_config(2))
    recs = _records()
    got = scenario._encode_images(recs).pooled
    # Independently: encode each doc's two pages and average.
    want_rows = []
    for r in recs:
        pages = scenario.image_encoder.encode(list(r.page_images)).pooled
        want_rows.append(pages.mean(dim=0))
    want = torch.stack(want_rows)
    assert torch.allclose(got, want, atol=1e-6), "pool=2 must be the mean of both page embeddings"
    # And it must differ from the page-1-only result.
    page1 = build_scenario(_config(1))._encode_images(recs).pooled
    assert not torch.allclose(got, page1, atol=1e-4), "pooling had no effect vs page 1"
