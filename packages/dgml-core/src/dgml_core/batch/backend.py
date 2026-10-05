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

"""The one interface every provider batch client implements."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from dgml_core.batch.types import BatchItemError, BatchJob, BatchRequest, BatchStatus


class BatchBackend(Protocol):
    """A provider's asynchronous batch endpoint, behind one shape.

    ``encode`` turns one :class:`BatchRequest` (litellm kwargs) into the
    provider's wire body for that request. It is also what size planning
    measures, so it must be a pure function of the request. ``submit`` sends
    a batch and returns a :class:`BatchJob`; ``poll`` reports progress;
    ``results`` yields ``(custom_id, ModelResponse | BatchItemError)`` pairs
    in whatever order the provider returns them — never rely on position.

    A result's ``ModelResponse`` is a litellm response object shaped exactly
    like the synchronous ``litellm.completion`` return, so the callers'
    parsing (``choices[0].message``, ``finish_reason``, ``usage``,
    ``_hidden_params["response_cost"]``) is unchanged. Backends price
    ``response_cost`` at the batch rate.

    ``submit`` never retries a create call whose outcome it cannot prove: a
    timeout or server error after the request was sent — or any failure once
    the create has answered 2xx (an unparseable answer, one naming no batch)
    — raises :class:`~dgml_core.batch.types.BatchSubmitUncertain` (the batch
    may exist and be billing); a batch-level size/limit refusal raises
    :class:`~dgml_core.batch.types.BatchRejected`, and a rate-limit/quota one
    :class:`~dgml_core.batch.types.BatchThrottled`. **Any other exception
    from ``submit`` is a promise that the provider did not accept the batch**:
    the executor runs its requests synchronously (or fails the wave when
    nothing was accepted), so a backend must never let a post-send failure
    escape as anything but ``BatchSubmitUncertain``.

    ``poll``, ``results`` and ``cancel`` raise
    :class:`~dgml_core.batch.types.BatchNotFound` when the provider answers
    404 — the batch was deleted or never existed (callers treat it as ended).

    ``max_wait_s`` is the provider's documented maximum batch lifetime — past
    it a batch that has not ended is expired by the provider.
    ``cancel_settle_s`` is how long a canceled batch may take to reach a
    terminal state (providers settle cancels on sweep cycles of minutes); a
    run that cancels on purpose (a run-level deadline) waits that long before
    giving up on collecting it. ``cleanup``
    best-effort deletes the provider-side artifacts (uploaded input, result
    files, the batch record where the API allows) of an ended batch whose
    results have been fully collected; it never raises for a failed delete.
    ``release`` drops any encodings the backend cached for those ids (encode
    caches so planning and submit share one encoding).
    """

    provider: str
    max_requests: int
    max_bytes: int
    max_wait_s: float
    cancel_settle_s: float

    def encode(self, request: BatchRequest) -> dict[str, Any]: ...

    def submit(self, requests: Sequence[BatchRequest]) -> BatchJob: ...

    def poll(self, job: BatchJob) -> BatchStatus: ...

    def results(self, job: BatchJob) -> Iterator[tuple[str, Any | BatchItemError]]: ...

    def cancel(self, job: BatchJob) -> None: ...

    def cleanup(self, job: BatchJob) -> None: ...

    def release(self, custom_ids: Sequence[str]) -> None: ...
