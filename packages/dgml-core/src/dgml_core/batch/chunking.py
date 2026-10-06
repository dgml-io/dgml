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

"""Split a wave of requests into batches a backend will accept."""

from __future__ import annotations

import json
from collections.abc import Sequence

from dgml_core.batch.backend import BatchBackend
from dgml_core.batch.types import BatchRequest


def request_size(request: BatchRequest, backend: BatchBackend) -> int:
    """Bytes the request occupies on the wire, as UTF-8 JSON of ``encode``."""
    return len(json.dumps(backend.encode(request), ensure_ascii=False).encode("utf-8"))


DEFAULT_OVERHEAD_BYTES = 256


def plan_batches(
    requests: Sequence[BatchRequest],
    backend: BatchBackend,
    *,
    overhead_bytes: int = DEFAULT_OVERHEAD_BYTES,
) -> list[list[BatchRequest]]:
    """Greedy, order-preserving split under ``max_requests`` and ``max_bytes``.

    Each request counts as its encoded size plus ``overhead_bytes`` — the
    per-line envelope a provider wraps it in (JSONL framing, ``custom_id``
    keys, separators) that ``encode`` does not see. A new batch starts when
    adding the next request would exceed either limit. A single request
    larger than ``max_bytes`` still gets a batch of its own: this function
    never drops work, and the provider's own error at ``submit`` is the right
    place for "too big" to surface.
    """
    batches: list[list[BatchRequest]] = []
    current: list[BatchRequest] = []
    current_bytes = 0
    for request in requests:
        size = request_size(request, backend) + overhead_bytes
        if current and (
            len(current) >= backend.max_requests or current_bytes + size > backend.max_bytes
        ):
            batches.append(current)
            current, current_bytes = [], 0
        current.append(request)
        current_bytes += size
    if current:
        batches.append(current)
    return batches
