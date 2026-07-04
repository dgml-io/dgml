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

"""Utilities — leaf module with no internal dependencies.

Exports the small set of cross-cutting helpers used everywhere else:
deterministic seeding, device auto-selection, and run-id hashing.
"""

from __future__ import annotations

from clustering.utils.device import DeviceInfo, DeviceKind, auto_select_device, resolve_device
from clustering.utils.runid import run_id_for
from clustering.utils.seed import seed_everything

__all__ = [
    "DeviceInfo",
    "DeviceKind",
    "auto_select_device",
    "resolve_device",
    "run_id_for",
    "seed_everything",
]
