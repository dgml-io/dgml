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

"""Value types shared by every batch backend and the wave driver.

A :class:`BatchRequest` carries the exact litellm completion kwargs the
synchronous path would have sent (see ``llm._build_completion_kwargs``), so a
backend's ``encode`` is a pure translation and the request the model sees is
byte-for-byte the sync request. A :class:`BatchJob` is what a backend hands
back after ``submit``: enough to poll and collect later — including from a
different process, which is why it round-trips through JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from dgml_core.errors import now_iso


@dataclass(frozen=True, eq=False)
class BatchRequest:
    """One completion to run inside a batch.

    ``kwargs`` are litellm ``completion`` kwargs exactly as
    ``llm._build_completion_kwargs`` produces them (model, messages, tools,
    max_tokens, …). ``custom_id`` must be unique within a submitted batch;
    results come back keyed by it, in any order. Identity follows that rule:
    two requests are equal, and hash alike, exactly when their ``custom_id``
    matches, so a set of requests dedups by id.
    """

    custom_id: str
    kwargs: dict[str, Any]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BatchRequest):
            return NotImplemented
        return self.custom_id == other.custom_id

    def __hash__(self) -> int:
        return hash(self.custom_id)


@dataclass
class BatchJob:
    """A submitted batch, as the provider knows it.

    ``extra`` is backend-private state a backend needs to poll or fetch
    results (input/output file ids, result URLs). It must stay JSON-safe:
    jobs are persisted so a later process can resume them.
    """

    provider: str
    job_id: str
    custom_ids: tuple[str, ...]
    submitted_at: str = field(default_factory=now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "job_id": self.job_id,
            "custom_ids": list(self.custom_ids),
            "submitted_at": self.submitted_at,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> BatchJob:
        return cls(
            provider=str(data["provider"]),
            job_id=str(data["job_id"]),
            custom_ids=tuple(str(c) for c in data.get("custom_ids", [])),
            submitted_at=str(data.get("submitted_at") or now_iso()),
            extra=dict(data.get("extra") or {}),
        )


class BatchState(StrEnum):
    """Lifecycle of a submitted batch, normalized across providers."""

    PENDING = "pending"
    RUNNING = "running"
    ENDED = "ended"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(frozen=True)
class BatchStatus:
    """One ``poll`` observation. Counts are per request within the batch;
    ``raw`` keeps the provider's own status payload for diagnostics."""

    state: BatchState
    succeeded: int = 0
    errored: int = 0
    expired: int = 0
    canceled: int = 0
    processing: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.state in (BatchState.ENDED, BatchState.FAILED, BatchState.CANCELED)


ItemErrorKind = Literal["errored", "expired", "canceled", "invalid", "batch_rejected"]

_RETRYABLE_KINDS: frozenset[str] = frozenset({"errored", "expired"})


@dataclass(frozen=True, init=False)
class BatchItemError:
    """A per-request failure inside an otherwise delivered batch.

    ``retryable`` defaults from ``kind``: a provider-side error or a 24-hour
    expiry says "not now" and is resubmitted by the driver; an invalid request
    or a cancellation would fail identically next time. Backends may override
    the default when the provider's error object says otherwise.

    ``batch_rejected`` is set on EVERY item of a batch the provider accepted
    and then failed as a whole for a batch-level reason (over an enqueued-token
    or queue limit, too large) — nothing about the request itself. It is not
    ``retryable``: resubmitting the same batch as-is meets the same limit, so
    the executor splits such a batch in half and resubmits the halves instead
    of the generic resubmit loop.
    """

    custom_id: str
    kind: ItemErrorKind
    message: str
    retryable: bool

    def __init__(
        self,
        custom_id: str,
        kind: ItemErrorKind,
        message: str,
        retryable: bool | None = None,
    ) -> None:
        object.__setattr__(self, "custom_id", custom_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "message", message)
        object.__setattr__(
            self, "retryable", kind in _RETRYABLE_KINDS if retryable is None else retryable
        )


class BatchSubmitUncertain(Exception):
    """A batch-create call failed in a way that does not prove the provider
    rejected it (a timeout or server error after the request was sent): the
    batch may exist and be billing. Never retried or resubmitted blindly."""


class BatchThrottled(Exception):
    """A batch-create call was refused at the account's rate limit or quota
    (still HTTP 429 after every retry, or a quota / billing refusal such as
    OpenAI's ``insufficient_quota``). Nothing was accepted, but neither a
    smaller batch nor a full-price synchronous run is the answer: the wave
    fails loudly (retry later, or check billing and quota)."""


class BatchRejected(Exception):
    """The provider refused a whole batch for a batch-level reason (too large,
    over a queue or enqueued-token limit), not for any one request in it."""


class BatchNotFound(Exception):
    """The provider no longer knows this batch (HTTP 404 on poll, results or
    cancel): it was deleted or never existed. Callers treat it as ended."""
