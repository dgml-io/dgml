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

"""The one failure-classification policy every batch backend applies.

Backends differ in transport (raw ``httpx``, the OpenAI SDK under litellm) and
in how a provider spells an error, but the questions asked of a failure are
the same everywhere, so they are answered here once:

* **Did a create call reach the provider?** A batch create is not idempotent
  and none of the providers takes an idempotency key for it, so a create is
  retried only when the failure *proves* nothing was accepted — the
  connection was never established, or the provider refused with a 429. A
  timeout after the request was sent, or a 5xx, leaves the batch possibly
  created and billing: :attr:`CreateOutcome.UNCERTAIN`, never retried.
* **Is it the account's rate limit or quota?** A create still refused with
  429 after every retry, or refused for quota or billing (OpenAI's
  ``insufficient_quota``, Gemini's ``RESOURCE_EXHAUSTED``, a low credit
  balance), is :attr:`CreateOutcome.THROTTLED`
  (:class:`~dgml_core.batch.types.BatchThrottled`): a smaller batch would be
  refused the same way and a synchronous run bills the same account, so the
  wave fails loudly instead of bisecting or falling back. Only a 429 that
  names the enqueued-token limit stays bisectable.
* **Is a refusal about the batch or about us?** A 4xx that names a size,
  queue or token limit is a batch-level refusal
  (:attr:`CreateOutcome.REJECTED`, :class:`~dgml_core.batch.types.BatchRejected`);
  an authentication, permission or credential problem, or any other client
  error, is an ordinary configuration error surfaced as the backend's own.
* **Did an accepted batch fail as a whole for a limit?**
  :func:`is_batch_limit_error` — the same limit vocabulary, applied to the
  batch-level error a provider reports after accepting a batch; every item of
  such a batch comes back as ``batch_rejected``.

Idempotent calls (poll, results, cancel, file uploads — a duplicate upload is
an orphan file, never a charge) keep the plain "transient → retry" rule of
:func:`idempotent_retryable`.
"""

from __future__ import annotations

from enum import StrEnum

import httpx

#: Statuses an idempotent call retries (timeouts, lock conflicts, rate limits,
#: server errors, Anthropic's 529 "overloaded").
RETRY_STATUSES: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

# Transport failures raised before any byte of the request left the process:
# the connection (or a pooled one) was never obtained.
_NOT_SENT: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)

# A refusal naming one of these is about the batch's size or the account's
# batch capacity, not about any one request.
_LIMIT_MARKERS = (
    "limit",
    "exceed",
    "too large",
    "too_large",
    "too big",
    "too many",
    "too_many",
    "quota",
    "resource_exhausted",
    "request_too_large",
)
# …unless it is really an authentication/permission problem (Gemini reports a
# bad key as 400 INVALID_ARGUMENT, so the status alone does not say).
_CREDENTIAL_MARKERS = (
    "api key",
    "api_key",
    "apikey",
    "credential",
    "authenticat",
    "unauthori",
    "permission",
    "forbidden",
)
_CREDENTIAL_STATUSES = frozenset({401, 403, 407})
# A refusal naming one of these is the account's quota or billing, which no
# retry within seconds, smaller batch or synchronous call gets around.
_QUOTA_MARKERS = (
    "insufficient_quota",
    "quota",
    "billing",
    "credit balance",
    "resource_exhausted",
)
# A rate-limit refusal naming this is about how many tokens the account has
# queued in batches: a smaller batch can fit, so it stays bisectable.
_ENQUEUED_MARKERS = ("enqueued",)


class CreateOutcome(StrEnum):
    """What a failed, non-idempotent create call means."""

    RETRY = "retry"  # provably not accepted and transient: try again
    UNCERTAIN = "uncertain"  # may have been accepted: never retry
    REJECTED = "rejected"  # refused for a batch-level size/limit reason
    THROTTLED = "throttled"  # refused at the account's rate limit or quota: fail loudly
    ERROR = "error"  # refused for any other reason (auth, config, malformed)


def is_credential_problem(status_code: int | None, detail: str) -> bool:
    """An authentication / permission / malformed-credential refusal."""
    if status_code in _CREDENTIAL_STATUSES:
        return True
    text = detail.lower()
    return any(marker in text for marker in _CREDENTIAL_MARKERS)


def is_batch_limit_error(detail: str, *, status_code: int | None = None) -> bool:
    """A refusal of a whole batch for its size or the account's batch capacity.

    ``detail`` is the provider's error code and/or message (any casing).
    413 is a limit by definition; a credential problem never is.
    """
    if is_credential_problem(status_code, detail):
        return False
    if status_code == 413:
        return True
    text = detail.lower()
    return any(marker in text for marker in _LIMIT_MARKERS)


def is_quota_error(detail: str, *, status_code: int | None = None) -> bool:
    """A refusal for the account's quota or billing (never a credential one)."""
    if status_code in _CREDENTIAL_STATUSES:
        return False
    text = detail.lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def is_enqueued_limit_error(detail: str) -> bool:
    """A refusal at the account's enqueued-batch-token limit (bisectable)."""
    text = detail.lower()
    return any(marker in text for marker in _ENQUEUED_MARKERS)


def classify_create_failure(
    *,
    status_code: int | None = None,
    transport_error: BaseException | None = None,
    detail: str = "",
) -> CreateOutcome:
    """Classify one failed create attempt (see the module docstring).

    Exactly one of ``status_code`` (the provider answered) or
    ``transport_error`` (it did not) describes the failure.
    """
    if status_code is None:
        if isinstance(transport_error, _NOT_SENT):
            return CreateOutcome.RETRY
        return CreateOutcome.UNCERTAIN
    if status_code < 400:
        # The provider answered success; only reading its answer failed (the
        # SDK's APIResponseValidationError carries the 2xx): it was accepted.
        return CreateOutcome.UNCERTAIN
    throttled = is_quota_error(detail, status_code=status_code) and not (
        is_enqueued_limit_error(detail)
    )
    if status_code == 429:
        # A plain rate limit clears in seconds: retry it (still refused after
        # the last attempt, it is THROTTLED — see the retry loop). Quota does not.
        return CreateOutcome.THROTTLED if throttled else CreateOutcome.RETRY
    if status_code in (408, 409) or status_code >= 500:
        # 408/409: the server gave up on, or conflicted with, a request it had
        # started to read — acceptance is not ruled out. 5xx/529: the same.
        return CreateOutcome.UNCERTAIN
    if is_credential_problem(status_code, detail):
        return CreateOutcome.ERROR
    if throttled:
        return CreateOutcome.THROTTLED
    if is_batch_limit_error(detail, status_code=status_code):
        return CreateOutcome.REJECTED
    return CreateOutcome.ERROR


def idempotent_retryable(
    *, status_code: int | None = None, transport_error: BaseException | None = None
) -> bool:
    """Whether an idempotent call (GET, cancel, upload) should be retried."""
    if status_code is None:
        return isinstance(transport_error, httpx.TransportError)
    return status_code in RETRY_STATUSES


def failure_of(exc: BaseException) -> tuple[int | None, BaseException | None]:
    """``(status_code, transport_error)`` for an exception raised by an HTTP
    client or an SDK built on ``httpx`` (the OpenAI SDK, litellm).

    The status comes from the exception's ``status_code`` (SDK status errors);
    the transport error is the first :class:`httpx.TransportError` in its
    cause chain (the SDK wraps it in ``APIConnectionError``/``APITimeoutError``).
    ``(None, None)`` means the exception says nothing about the wire.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        return status, None
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, httpx.TransportError):
            return None, current
        current = current.__cause__ or current.__context__
    return None, None
