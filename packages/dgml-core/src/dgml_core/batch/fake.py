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

"""An in-memory :class:`BatchBackend` for tests and offline driver runs.

Scripted rather than simulated: the caller decides, per ``custom_id``, what
comes back — a litellm ``ModelResponse`` or a :class:`BatchItemError` — and
in what order, how many polls a batch takes to end, and whether ``submit``
fails. Everything submitted is recorded so a test can assert on the exact
wire shape the driver produced.

Each request's scripted outcome is resolved exactly once per job — at the
first poll that ends the batch (or at ``results`` if never polled) — and
cached, so a callable script with side effects runs once per request and
``poll`` counts agree with what ``results`` later yields. Cancelling a job
freezes every not-yet-resolved item as a ``canceled`` error.

Batch-level failures are scripted per batch: ``fail_submit`` may be a callable
of the batch that returns the exception ``submit`` raises (or ``None`` to
accept), and ``reject_batch`` a predicate over an accepted batch's requests
that ends it with every item ``batch_rejected``. ``release`` and ``cleanup``
record their calls; ``max_wait_s`` and ``cancel_settle_s`` are plain
configuration.

``processed_early`` scripts partial progress: a request it holds for is
resolved at its batch's first poll while the batch is still running, so a
cancel after that keeps its result and freezes only the rest as ``canceled``
— the partial results providers return for a batch canceled mid-way.
``cancel_settle_polls`` makes a canceled batch report ``running`` (the
provider's ``canceling``) for that many polls before it ends.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from litellm import ModelResponse

from dgml_core.batch.types import (
    BatchItemError,
    BatchJob,
    BatchRequest,
    BatchState,
    BatchStatus,
)

ResponseScript = Mapping[str, Any] | Callable[[BatchRequest], Any]
SubmitScript = Exception | Callable[[list[BatchRequest]], Exception | None]


def fake_model_response(
    text: str,
    *,
    finish_reason: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
    cost: float | None = None,
    model: str = "fake/model",
) -> ModelResponse:
    """A real ``litellm.ModelResponse`` shaped like a completion return.

    Built through litellm's own constructor so attribute access
    (``response.choices[0].message.content``), item access
    (``response["choices"][0]["message"]["content"]``), ``usage`` and
    ``_hidden_params["response_cost"]`` all behave as they do on a live
    response — which is what every caller in this package reads.
    """
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    response = ModelResponse(
        choices=[{"message": message, "finish_reason": finish_reason, "index": 0}],
        usage=usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        model=model,
    )
    if cost is not None:
        response._hidden_params["response_cost"] = cost
    return response


@dataclass
class _JobState:
    requests: list[BatchRequest]
    polls: int = 0
    canceled: bool = False
    polls_since_cancel: int = 0
    rejected: bool = False
    # custom_id → outcome, filled once (see module docstring).
    outcomes: dict[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return len(self.outcomes) == len(self.requests)


class FakeBackend:
    """Scripted backend. See the module docstring."""

    def __init__(
        self,
        responses: ResponseScript,
        *,
        provider: str = "fake",
        shuffle: bool = False,
        seed: int = 0,
        polls_until_ended: int = 0,
        fail_submit: SubmitScript | None = None,
        reject_batch: Callable[[list[BatchRequest]], bool] | None = None,
        max_requests: int = 100_000,
        max_bytes: int = 256 * 1024**2,
        max_wait_s: float = 24 * 3600.0,
        cancel_settle_s: float = 180.0,
        processed_early: Callable[[BatchRequest], bool] | None = None,
        cancel_settle_polls: int = 0,
    ) -> None:
        self._processed_early = processed_early
        #: Polls a canceled batch reports `running` for; public so a test can
        #: let a long-settling cancel end later.
        self.cancel_settle_polls = cancel_settle_polls
        self.provider = provider
        self.max_requests = max_requests
        self.max_bytes = max_bytes
        self.max_wait_s = max_wait_s
        self.cancel_settle_s = cancel_settle_s
        self._reject_batch = reject_batch
        self._responses = responses
        self._shuffle = shuffle
        self._rng = random.Random(seed)
        self._polls_until_ended = polls_until_ended
        self._fail_submit = fail_submit
        # Recorders.
        self.submitted: list[list[BatchRequest]] = []
        self.encoded: list[dict[str, Any]] = []
        self.polls = 0
        self.canceled: list[str] = []
        #: Every submit attempt's batch, accepted or not (``submitted`` has
        #: only the accepted ones).
        self.attempted: list[list[BatchRequest]] = []
        self.released: list[tuple[str, ...]] = []
        self.cleaned: list[str] = []
        self._jobs: dict[str, _JobState] = {}

    # ---- BatchBackend --------------------------------------------------

    def encode(self, request: BatchRequest) -> dict[str, Any]:
        return {"custom_id": request.custom_id, "params": request.kwargs}

    def submit(self, requests: Sequence[BatchRequest]) -> BatchJob:
        batch = list(requests)
        self.attempted.append(batch)
        failure = self._fail_submit
        if callable(failure) and not isinstance(failure, BaseException):
            failure = failure(batch)
        if failure is not None:
            raise failure
        self.submitted.append(batch)
        self.encoded.extend(self.encode(r) for r in batch)
        job_id = f"fake_batch_{len(self.submitted):04d}"
        rejected = self._reject_batch is not None and self._reject_batch(batch)
        self._jobs[job_id] = _JobState(requests=batch, rejected=rejected)
        return BatchJob(
            provider=self.provider, job_id=job_id, custom_ids=tuple(r.custom_id for r in batch)
        )

    def poll(self, job: BatchJob) -> BatchStatus:
        self.polls += 1
        state = self._state(job)
        state.polls += 1
        if state.canceled:
            state.polls_since_cancel += 1
            if state.polls_since_cancel <= self.cancel_settle_polls:
                return BatchStatus(state=BatchState.RUNNING)
            self._resolve(state)
            return self._counts(state, BatchState.CANCELED)
        if state.polls <= self._polls_until_ended:
            if self._processed_early is not None:
                for request in state.requests:
                    if request.custom_id not in state.outcomes and self._processed_early(request):
                        state.outcomes[request.custom_id] = self._scripted(request)
            return BatchStatus(state=BatchState.RUNNING, processing=len(state.requests))
        self._resolve(state)
        return self._counts(state, BatchState.FAILED if state.rejected else BatchState.ENDED)

    def results(self, job: BatchJob) -> Iterator[tuple[str, Any | BatchItemError]]:
        state = self._state(job)
        self._resolve(state)
        order = list(state.requests)
        if self._shuffle:
            self._rng.shuffle(order)
        for request in order:
            yield request.custom_id, state.outcomes[request.custom_id]

    def cancel(self, job: BatchJob) -> None:
        self.canceled.append(job.job_id)
        state = self._state(job)
        state.canceled = True
        # Everything the script has not yet answered is frozen as canceled;
        # outcomes already resolved stay as they were (they were delivered).
        self._resolve(state)

    def release(self, custom_ids: Sequence[str]) -> None:
        self.released.append(tuple(custom_ids))

    def cleanup(self, job: BatchJob) -> None:
        self._state(job)  # an unknown job is a caller bug
        self.cleaned.append(job.job_id)

    # ---- internals -----------------------------------------------------

    def _state(self, job: BatchJob) -> _JobState:
        try:
            return self._jobs[job.job_id]
        except KeyError:
            raise KeyError(f"unknown fake batch job {job.job_id!r}") from None

    def _resolve(self, state: _JobState) -> None:
        """Fill every unresolved outcome exactly once."""
        if state.resolved:
            return
        for request in state.requests:
            if request.custom_id in state.outcomes:
                continue
            if state.canceled:
                outcome: Any = BatchItemError(
                    request.custom_id, "canceled", "batch canceled before this request ran"
                )
            elif state.rejected:
                outcome = BatchItemError(
                    request.custom_id, "batch_rejected", "batch failed validation (fake)"
                )
            else:
                outcome = self._scripted(request)
            state.outcomes[request.custom_id] = outcome

    def _scripted(self, request: BatchRequest) -> Any | BatchItemError:
        if callable(self._responses):
            return self._responses(request)
        try:
            return self._responses[request.custom_id]
        except KeyError:
            return BatchItemError(
                request.custom_id, "invalid", f"no scripted response for {request.custom_id!r}"
            )

    @staticmethod
    def _counts(state: _JobState, final: BatchState) -> BatchStatus:
        succeeded = errored = expired = canceled = 0
        for outcome in state.outcomes.values():
            if not isinstance(outcome, BatchItemError):
                succeeded += 1
            elif outcome.kind == "expired":
                expired += 1
            elif outcome.kind == "canceled":
                canceled += 1
            else:
                errored += 1
        return BatchStatus(
            state=final, succeeded=succeeded, errored=errored, expired=expired, canceled=canceled
        )
