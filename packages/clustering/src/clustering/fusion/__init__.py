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

"""Fusion — Research Axis #1.

Importing this module registers all fusion variants: ``none``,
``concat_norm``, ``late_concat``, ``cross_attention``, ``gated``,
``late_interaction``.
"""

from __future__ import annotations

# Side-effect imports so the registry is populated.
from clustering.fusion import (
    concat_norm,  # noqa: F401
    cross_attention,  # noqa: F401
    gated,  # noqa: F401
    late_concat,  # noqa: F401
    late_interaction,  # noqa: F401
    none_,  # noqa: F401
)
from clustering.fusion.base import (
    Fusion,
    FusionOutput,
    build_fusion,
    maxsim,
    register_fusion,
    registered_fusions,
)

__all__ = [
    "Fusion",
    "FusionOutput",
    "build_fusion",
    "maxsim",
    "register_fusion",
    "registered_fusions",
]
