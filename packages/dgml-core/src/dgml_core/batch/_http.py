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

"""The one retry loop every batch backend's provider calls go through.

:func:`call_with_retries` runs one provider call with bounded, backed-off
retries under the :mod:`dgml_core.batch._policy` rules. The call is any
zero-argument callable: a raw ``httpx`` request (Anthropic) returns a
:class:`httpx.Response` whose status is checked here; an SDK call (OpenAI via
litellm) returns its object or raises, and the status / transport error is
read off the exception. So the create-safety rule — never resend a create
whose acceptance is not disproved — is one piece of code for all providers.

Outcomes:

* success → the call's return value;
* a non-idempotent (create) call that may have been accepted — including
  one that failed in a way the policy cannot read (no status, no transport
  error), or with a success status →
  :class:`~dgml_core.batch.types.BatchSubmitUncertain`;
* a create refused for a batch-level size/limit reason (or still refused at
  the enqueued-token limit after the last attempt) →
  :class:`~dgml_core.batch.types.BatchRejected`;
* a create refused for quota/billing, or still rate-limited after the last
  attempt → :class:`~dgml_core.batch.types.BatchThrottled`;
* anything else that is not retried, or retries exhausted →
  :class:`HttpFailure`, which each backend turns into its own error type with
  its own (unchanged) message. An exception the policy cannot read (no
  status, no transport error in its chain) from an idempotent call
  propagates untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TypeVar

import httpx

from dgml_core.batch._policy import (
    CreateOutcome,
    classify_create_failure,
    failure_of,
    idempotent_retryable,
    is_enqueued_limit_error,
)
from dgml_core.batch.types import BatchRejected, BatchSubmitUncertain, BatchThrottled

T = TypeVar("T")


class HttpFailure(Exception):
    """A provider call failed and will not be retried (see the module docstring).

    ``detail`` is the last failure as one line (``"HTTP 400: …"`` or
    ``"ConnectError: …"``); ``exhausted`` says the retry budget ran out (vs a
    failure that is never retried); ``cause`` is the original exception for an
    SDK call, so an adapter can re-raise it unchanged.
    """

    def __init__(
        self,
        detail: str,
        *,
        status_code: int | None,
        exhausted: bool,
        attempts: int,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.exhausted = exhausted
        self.attempts = attempts
        self.cause = cause


def _response_detail(response: httpx.Response, limit: int) -> str:
    try:
        text = response.text
    except Exception:  # a streamed or undecodable body: the status says enough
        text = ""
    return f"HTTP {response.status_code}: {text[:limit]}"


def call_with_retries(
    call: Callable[[], T],
    *,
    what: str,
    idempotent: bool,
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None],
    detail_chars: int = 500,
) -> T:
    """Run ``call`` under the shared retry policy; ``what`` names it in errors
    (e.g. ``"POST https://…/batches"``). Delays double from ``base_delay``."""
    last = ""
    last_status: int | None = None
    last_cause: BaseException | None = None
    for attempt in range(attempts):
        status: int | None = None
        transport: BaseException | None = None
        cause: BaseException | None = None
        try:
            result = call()
        except Exception as exc:
            status, transport = failure_of(exc)
            if status is None and transport is None:
                if not idempotent:
                    # Nothing proves a create that failed this way was not
                    # accepted (an SDK failing to parse its 2xx answer, say).
                    raise uncertain_create(what, exc) from exc
                raise
            cause = exc
            last = f"{type(exc).__name__}: {exc}"
            if status is not None:
                last = f"HTTP {status}: {str(exc)[:detail_chars]}"
        else:
            if not isinstance(result, httpx.Response) or result.status_code < 400:
                return result
            status = result.status_code
            last = _response_detail(result, detail_chars)
        last_status, last_cause = status, cause

        if idempotent:
            retry = idempotent_retryable(status_code=status, transport_error=transport)
        else:
            outcome = classify_create_failure(
                status_code=status, transport_error=transport, detail=last
            )
            if outcome is CreateOutcome.UNCERTAIN:
                raise BatchSubmitUncertain(
                    f"{what}: {last} — the batch may have been created (and be billing); "
                    "it is not resubmitted automatically"
                ) from cause
            if outcome is CreateOutcome.REJECTED:
                raise BatchRejected(f"{what} refused the batch: {last}") from cause
            if outcome is CreateOutcome.THROTTLED:
                raise BatchThrottled(
                    f"{what} refused the batch at the account's quota: {last}"
                ) from cause
            retry = outcome is CreateOutcome.RETRY
        if not retry:
            raise HttpFailure(
                last, status_code=status, exhausted=False, attempts=attempt + 1, cause=cause
            ) from cause
        if attempt + 1 < attempts:
            sleep(base_delay * (2**attempt))

    if not idempotent and last_status == 429:
        # Still refused after every attempt: the provider did not take the
        # batch, and will not right now. Only the enqueued-token limit is
        # about the batch's size (a smaller one may fit); any other rate
        # limit fails the wave loudly rather than bisecting or running sync.
        if is_enqueued_limit_error(last):
            raise BatchRejected(
                f"{what} refused the batch after {attempts} attempts: {last}"
            ) from last_cause
        raise BatchThrottled(
            f"{what} still rate-limited after {attempts} attempts: {last}"
        ) from last_cause
    raise HttpFailure(
        last, status_code=last_status, exhausted=True, attempts=attempts, cause=last_cause
    ) from last_cause


def uncertain_create(what: str, cause: BaseException) -> BatchSubmitUncertain:
    """The error for a create that may have been accepted although its
    outcome could not be read — raised by the retry loop, and by a backend
    whose create answered 2xx with a body it cannot parse or that names no
    batch: the batch may exist, so it must never be treated as refused."""
    return BatchSubmitUncertain(
        f"{what}: {type(cause).__name__}: {cause} — the batch may have been created "
        "(and be billing); it is not resubmitted automatically"
    )


def request(
    client: httpx.Client, method: str, url: str, **kwargs: Any
) -> Callable[[], httpx.Response]:
    """A zero-argument call sending one ``httpx`` request (for
    :func:`call_with_retries`)."""
    return lambda: client.request(method, url, **kwargs)


def stream_request(
    client: httpx.Client, method: str, url: str, **kwargs: Any
) -> Callable[[], httpx.Response]:
    """Like :func:`request`, but a successful response comes back with its
    body unread (``stream=True``), for results files too large to buffer:
    the caller iterates it (``iter_lines``) and must close it. An error
    response is read and closed at once, so its detail is still reported."""

    def call() -> httpx.Response:
        response = client.send(client.build_request(method, url, **kwargs), stream=True)
        if response.status_code >= 400:
            try:
                response.read()
            finally:
                response.close()
        return response

    return call


def iter_response_lines(response: httpx.Response) -> Iterator[str]:
    """The non-blank lines of a streamed *response*, stripped, read one chunk
    at a time; the response is closed when iteration ends or is abandoned."""
    try:
        for line in response.iter_lines():
            line = line.strip()
            if line:
                yield line
    finally:
        response.close()
