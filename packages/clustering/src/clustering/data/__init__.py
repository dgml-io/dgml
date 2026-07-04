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

"""Data loading: dataset wrapper.

For the low-thousand-document scale we target, everything fits in memory.
Heavy on-disk structures (ANN indexes, tile servers) are intentionally absent.
"""

from __future__ import annotations

from clustering.data.datasets import DocumentDataset, DocumentRecord

__all__ = [
    "DocumentDataset",
    "DocumentRecord",
]
