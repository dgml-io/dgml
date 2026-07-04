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

"""Euclidean manifold — flat ``R^d`` with the usual L2 metric."""

from __future__ import annotations

from typing import cast

import torch

from clustering.config.schema import ManifoldConfig
from clustering.manifolds.base import ManifoldHead, register_manifold


class EuclideanHead(ManifoldHead):
    """Identity projection; L2 distance; trivial expmap/logmap."""

    def __init__(self, cfg: ManifoldConfig) -> None:
        super().__init__()
        self.dim = cfg.dim

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, (x - y).norm(dim=-1))

    def pairwise_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cdist(x, y, p=2)

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return x + v

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - x

    def to_geoopt(self) -> object:
        import geoopt

        return geoopt.Euclidean()


@register_manifold("euclidean")
def _factory(cfg: ManifoldConfig) -> ManifoldHead:
    return EuclideanHead(cfg)
