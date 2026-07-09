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

"""Tests for the Matryoshka (MRL) helpers.

``mrl_truncate`` (torch) and ``mrl_dimension_sweep`` (numpy + sklearn) run
without any model weights or network — pure array ops.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from clustering.encoders.mrl import SweepResult, mrl_dimension_sweep, mrl_truncate


def test_truncate_shrinks_and_renormalizes() -> None:
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1.0
    out = mrl_truncate(x, 2)
    assert out.shape == (3, 2)
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-6)


def test_truncate_keeps_leading_dims() -> None:
    x = torch.tensor([[1.0, 2.0, 99.0, 99.0]])
    out = mrl_truncate(x, 2, normalize=False)
    assert torch.equal(out, torch.tensor([[1.0, 2.0]]))


def test_truncate_none_keeps_width_but_normalizes() -> None:
    x = torch.tensor([[3.0, 4.0]])
    out = mrl_truncate(x, None)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[0.6, 0.8]]), atol=1e-6)


def test_truncate_equal_dim_is_noop_shape() -> None:
    x = torch.randn(5, 8)
    out = mrl_truncate(x, 8, normalize=False)
    assert out.shape == (5, 8)
    assert torch.equal(out, x)


def test_truncate_rejects_widening() -> None:
    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match="can only shrink"):
        mrl_truncate(x, 8)


def test_truncate_rejects_nonpositive() -> None:
    x = torch.randn(2, 4)
    with pytest.raises(ValueError, match="must be positive"):
        mrl_truncate(x, 0)


def test_truncate_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        mrl_truncate(torch.randn(2, 3, 4), 2)


def _clustered_embeddings(seed: int = 0) -> np.ndarray:
    """Three tight clusters carried entirely by the first two dims; the
    remaining dims are pure noise, so a small MRL prefix should cluster at
    least as well as the full width."""
    rng = np.random.default_rng(seed)
    centers = np.array([[5.0, 0.0], [-5.0, 5.0], [-5.0, -5.0]])
    pts = np.repeat(centers, 20, axis=0) + rng.normal(scale=0.2, size=(60, 2))
    noise = rng.normal(scale=3.0, size=(60, 14))
    return np.hstack([pts, noise]).astype(np.float32)


def test_dimension_sweep_prefers_informative_prefix() -> None:
    from sklearn.cluster import KMeans

    embeddings = _clustered_embeddings()

    def cluster_fn(feats: np.ndarray) -> np.ndarray:
        labels: np.ndarray = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(feats)
        return labels

    result = mrl_dimension_sweep(embeddings, [2, 4, 8, 16], cluster_fn)
    assert isinstance(result, SweepResult)
    # Native width (16) is always appended to the sweep.
    assert result.dims == (2, 4, 8, 16)
    assert result.best_labels.shape == (60,)
    # The signal lives in the first 2 dims; the 2-d prefix should win (or at
    # least match) the noisy full-width representation.
    assert result.best_dim <= 4
    assert result.scores[0] >= result.scores[-1]


def test_dimension_sweep_ties_pick_smaller_dim() -> None:
    embeddings = np.random.default_rng(1).normal(size=(30, 8)).astype(np.float32)

    def constant_labels(feats: np.ndarray) -> np.ndarray:
        return np.array([0, 1] * (feats.shape[0] // 2))

    # A feature-independent score makes every width score identically ⇒ a
    # genuine tie, which must be broken toward the cheaper (smaller) width.
    def constant_score(feats: np.ndarray, labels: np.ndarray) -> float:
        return 1.0

    result = mrl_dimension_sweep(embeddings, [4, 8], constant_labels, score_fn=constant_score)
    assert result.best_dim == 4


def test_dimension_sweep_requires_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        mrl_dimension_sweep(np.zeros((2, 3, 4)), [2], lambda f: np.zeros(len(f), dtype=int))
