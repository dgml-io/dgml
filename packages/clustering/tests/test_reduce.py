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

"""Tests for the pre-clustering dimensionality reduction helper."""

from __future__ import annotations

import importlib.util

import pytest
import torch
from clustering.scenarios.clustering import reduce_embeddings

_SKLEARN = importlib.util.find_spec("sklearn") is not None
_UMAP = importlib.util.find_spec("umap") is not None


def test_reduce_none_is_passthrough() -> None:
    x = torch.randn(10, 16)
    assert reduce_embeddings(x, method="none", n_components=4) is x


def test_reduce_zero_components_is_passthrough() -> None:
    x = torch.randn(10, 16)
    assert reduce_embeddings(x, method="pca", n_components=0) is x


def test_reduce_components_geq_dim_is_passthrough() -> None:
    x = torch.randn(10, 16)
    assert reduce_embeddings(x, method="pca", n_components=16) is x
    assert reduce_embeddings(x, method="pca", n_components=99) is x


def test_reduce_tiny_corpus_is_passthrough() -> None:
    # n < 3 — nothing meaningful to reduce.
    x = torch.randn(2, 16)
    assert reduce_embeddings(x, method="pca", n_components=4) is x


_SKLEARN_METHODS = [
    "pca",
    "truncated_svd",
    "random_projection",
    "kernel_pca",
    "isomap",
    "lle",
    "spectral",
]


@pytest.mark.skipif(not _SKLEARN, reason="scikit-learn not installed")
@pytest.mark.parametrize("method", _SKLEARN_METHODS)
def test_reduce_sklearn_methods_shape(method: str) -> None:
    # Enough samples and well-separated structure so neighbour-graph
    # methods (isomap/lle/spectral) stay numerically well-behaved.
    torch.manual_seed(0)
    a = torch.randn(15, 16) + 5.0
    b = torch.randn(15, 16) - 5.0
    x = torch.cat([a, b], dim=0)
    out = reduce_embeddings(x, method=method, n_components=4, seed=0)  # type: ignore[arg-type]
    assert out.shape == (30, 4)


@pytest.mark.skipif(not _SKLEARN, reason="scikit-learn not installed")
def test_reduce_pca_clamps_components_to_n_minus_one() -> None:
    # 4 samples → at most 3 PCA components, even if more are requested.
    x = torch.randn(4, 16)
    out = reduce_embeddings(x, method="pca", n_components=10, seed=0)
    assert out.shape[0] == 4
    assert out.shape[1] <= 3


@pytest.mark.skipif(not _UMAP, reason="umap-learn not installed")
def test_reduce_umap_small_corpus_does_not_crash() -> None:
    # 10 samples with the default umap reducer can trip UMAP's spectral init
    # ("Cannot use scipy.linalg.eigh for sparse A with k >= N"); clamped
    # n_neighbors + random init must keep it working.
    torch.manual_seed(0)
    x = torch.randn(10, 256)
    out = reduce_embeddings(x, method="umap", n_components=5, seed=0)
    assert out.shape == (10, 5)


@pytest.mark.skipif(not _SKLEARN, reason="scikit-learn not installed")
def test_reduce_neighbour_based_falls_back_to_pca_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # If a neighbour-based reducer explodes on a degenerate corpus, the run
    # falls back to PCA (with a warning) instead of aborting.
    import clustering.scenarios.clustering as mod

    real_dispatch = mod._dispatch_reduce

    def boom(x, *, method, k, n, seed):  # type: ignore[no-untyped-def]
        if method == "spectral":
            raise RuntimeError("synthetic reducer failure")
        return real_dispatch(x, method=method, k=k, n=n, seed=seed)

    monkeypatch.setattr(mod, "_dispatch_reduce", boom)

    x = torch.randn(12, 16)
    with pytest.warns(RuntimeWarning, match="Falling back to PCA"):
        out = reduce_embeddings(x, method="spectral", n_components=4, seed=0)
    assert out.shape == (12, 4)
