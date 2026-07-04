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

"""Spherical ``dist`` must be robust to non-unit-norm inputs.

The chord→angle identity ``2·asin(‖x-y‖/2)`` only holds for points on the
unit sphere. The defensive ``_safe_normalize`` inside ``SphericalHead.dist``
makes it robust to slightly-off-sphere inputs (e.g. raw projector linear
outputs before ``expmap0``). ``HyperbolicHead.dist`` already has the
equivalent defense via ``_project_into_ball``; this is the spherical-side
parity fix ported from doc-categorization.
"""

from __future__ import annotations

import torch
from clustering.config.schema import ManifoldConfig
from clustering.manifolds import build_manifold


def test_spherical_dist_invariant_under_radial_scaling() -> None:
    """``spherical.dist(a*x, b*y) == spherical.dist(x, y)`` for a, b > 0.

    Both inputs lie on the same rays through the origin, so their
    geodesic distance on the sphere is determined by the angle between
    them — independent of magnitude. The defensive normalize makes this
    hold for any positive scaling.
    """
    m = build_manifold(ManifoldConfig(name="spherical", dim=8, curvature=1.0))
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    y = torch.randn(5, 8)
    # Reference distance (unit-norm inputs).
    x_unit = m.project(x)
    y_unit = m.project(y)
    d_ref = m.dist(x_unit, y_unit)
    # Scaled inputs — should produce the same distances after the fix.
    d_scaled = m.dist(3.7 * x_unit, 0.42 * y_unit)
    assert torch.allclose(d_ref, d_scaled, atol=1e-5), (
        f"spherical.dist not invariant under radial scaling: "
        f"max diff = {(d_ref - d_scaled).abs().max().item():.2e}"
    )


def test_spherical_dist_handles_unnormalized_raw_inputs() -> None:
    """Raw (off-sphere) inputs must produce the same dist as their normalized form.

    If someone hands raw ambient vectors to ``spherical.dist`` (instead
    of going through ``expmap0`` first), the answer must still equal
    the geodesic distance between the corresponding sphere points.
    """
    m = build_manifold(ManifoldConfig(name="spherical", dim=8, curvature=1.0))
    torch.manual_seed(1)
    x_raw = torch.randn(4, 8) * 5.0  # large ambient magnitudes — definitely off-sphere
    y_raw = torch.randn(4, 8) * 0.01  # very small magnitudes
    d_raw = m.dist(x_raw, y_raw)
    d_proj = m.dist(m.project(x_raw), m.project(y_raw))
    assert torch.allclose(d_raw, d_proj, atol=1e-5)
