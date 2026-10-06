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

"""The shared batch failure policy, retry loop and encode cache (offline)."""

from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest
from dgml_core.batch import BatchRequest
from dgml_core.batch._capture import EncodeCache
from dgml_core.batch._http import HttpFailure, call_with_retries
from dgml_core.batch._policy import (
    CreateOutcome,
    classify_create_failure,
    failure_of,
    is_batch_limit_error,
)
from dgml_core.batch.types import BatchItemError, BatchRejected, BatchSubmitUncertain

_REQ = httpx.Request("POST", "https://provider.example/v1/batches")


@pytest.mark.parametrize(
    ("kwargs", "outcome"),
    [
        ({"transport_error": httpx.ConnectError("refused", request=_REQ)}, CreateOutcome.RETRY),
        ({"transport_error": httpx.ConnectTimeout("slow", request=_REQ)}, CreateOutcome.RETRY),
        ({"transport_error": httpx.PoolTimeout("pool", request=_REQ)}, CreateOutcome.RETRY),
        ({"transport_error": httpx.ReadTimeout("slow", request=_REQ)}, CreateOutcome.UNCERTAIN),
        ({"transport_error": httpx.WriteTimeout("slow", request=_REQ)}, CreateOutcome.UNCERTAIN),
        ({"transport_error": httpx.RemoteProtocolError("eof")}, CreateOutcome.UNCERTAIN),
        ({"status_code": 429}, CreateOutcome.RETRY),
        ({"status_code": 408}, CreateOutcome.UNCERTAIN),
        ({"status_code": 409}, CreateOutcome.UNCERTAIN),
        ({"status_code": 500}, CreateOutcome.UNCERTAIN),
        ({"status_code": 529}, CreateOutcome.UNCERTAIN),
        ({"status_code": 413}, CreateOutcome.REJECTED),
        ({"status_code": 400, "detail": "HTTP 400: too many requests in batch"}, "rejected"),
        ({"status_code": 400, "detail": "enqueued token limit exceeded"}, "rejected"),
        ({"status_code": 400, "detail": "API key not valid (limit 1)"}, CreateOutcome.ERROR),
        ({"status_code": 401, "detail": "rate limit"}, CreateOutcome.ERROR),
        ({"status_code": 403}, CreateOutcome.ERROR),
        ({"status_code": 400, "detail": "HTTP 400: malformed body"}, CreateOutcome.ERROR),
        ({"status_code": 404}, CreateOutcome.ERROR),
    ],
)
def test_create_failure_classification(kwargs: dict[str, Any], outcome: str) -> None:
    assert classify_create_failure(**kwargs) == outcome


def test_limit_errors_exclude_credential_problems() -> None:
    assert is_batch_limit_error("token_limit_exceeded: Enqueued token limit reached")
    assert is_batch_limit_error("RESOURCE_EXHAUSTED: quota")
    assert not is_batch_limit_error("invalid_request: bad body")
    assert not is_batch_limit_error("PERMISSION_DENIED: over limit", status_code=None)


def test_failure_of_reads_sdk_status_and_wrapped_transport_errors() -> None:
    status_error = openai.InternalServerError(
        "boom", response=httpx.Response(502, request=_REQ), body=None
    )
    assert failure_of(status_error) == (502, None)
    cause = httpx.ReadTimeout("slow", request=_REQ)
    try:
        raise openai.APITimeoutError(request=_REQ) from cause
    except openai.APITimeoutError as exc:
        assert failure_of(exc) == (None, cause)
    assert failure_of(ValueError("not a wire failure")) == (None, None)


class _Script:
    """Answers each call with the next scripted response or exception."""

    def __init__(self, *steps: int | Exception) -> None:
        self.steps = list(steps)
        self.calls = 0

    def __call__(self) -> httpx.Response:
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return httpx.Response(step, request=_REQ, text=f"status {step}")


def _run(script: _Script, *, idempotent: bool, sleeps: list[float] | None = None) -> Any:
    return call_with_retries(
        script,
        what="POST /v1/batches",
        idempotent=idempotent,
        attempts=3,
        base_delay=1.0,
        sleep=(sleeps if sleeps is not None else []).append,
    )


def test_idempotent_calls_retry_transient_failures_with_backoff() -> None:
    sleeps: list[float] = []
    script = _Script(503, httpx.ReadTimeout("slow", request=_REQ), 200)
    assert _run(script, idempotent=True, sleeps=sleeps).status_code == 200
    assert sleeps == [1.0, 2.0] and script.calls == 3


def test_idempotent_retries_exhausted_is_an_http_failure() -> None:
    with pytest.raises(HttpFailure) as info:
        _run(_Script(503, 503, 503), idempotent=True)
    assert info.value.exhausted and info.value.attempts == 3 and info.value.status_code == 503


def test_client_errors_are_never_retried() -> None:
    script = _Script(400, 200)
    with pytest.raises(HttpFailure) as info:
        _run(script, idempotent=True)
    assert not info.value.exhausted and script.calls == 1


def test_create_is_sent_once_when_it_may_have_landed() -> None:
    for step in (500, 529, httpx.ReadTimeout("slow", request=_REQ)):
        script = _Script(step, 200)
        with pytest.raises(BatchSubmitUncertain, match="may have been created"):
            _run(script, idempotent=False)
        assert script.calls == 1


def test_create_retries_only_provable_non_acceptance() -> None:
    script = _Script(httpx.ConnectError("refused", request=_REQ), 429, 200)
    assert _run(script, idempotent=False).status_code == 200
    assert script.calls == 3


def test_create_rate_limited_throughout_is_throttled_not_rejected() -> None:
    """A rate/quota refusal is not a batch-size problem: never bisected (F2)."""
    from dgml_core.batch.types import BatchThrottled

    script = _Script(429, 429, 429)
    with pytest.raises(BatchThrottled, match="after 3 attempts"):
        _run(script, idempotent=False)
    assert script.calls == 3


def test_create_over_quota_is_throttled_at_once() -> None:
    """``insufficient_quota`` will not clear in seconds: no retry, no bisect (F2)."""
    from dgml_core.batch.types import BatchThrottled

    class _Quota(_Script):
        def __call__(self) -> httpx.Response:
            self.calls += 1
            return httpx.Response(429, request=_REQ, json={"error": {"code": "insufficient_quota"}})

    script = _Quota()
    with pytest.raises(BatchThrottled, match="insufficient_quota"):
        _run(script, idempotent=False)
    assert script.calls == 1


@pytest.mark.parametrize(
    ("status", "detail"),
    [
        (429, "HTTP 429: insufficient_quota: You exceeded your current quota"),
        (400, "HTTP 400: Your credit balance is too low to access the API"),
        (400, "HTTP 400: quota exceeded for this project"),
        (429, "HTTP 429: RESOURCE_EXHAUSTED: Resource has been exhausted (e.g. check quota)"),
    ],
)
def test_quota_refusals_are_classified_throttled(status: int, detail: str) -> None:
    assert classify_create_failure(status_code=status, detail=detail) == "throttled"


def test_create_rate_limited_at_the_enqueued_token_limit_is_still_bisectable() -> None:
    class _Enqueued(_Script):
        def __call__(self) -> httpx.Response:
            self.calls += 1
            return httpx.Response(429, request=_REQ, text="Enqueued token limit reached")

    with pytest.raises(BatchRejected, match="after 3 attempts"):
        _run(_Enqueued(), idempotent=False)


def test_create_never_connecting_is_an_ordinary_failure() -> None:
    errors = [httpx.ConnectError("refused", request=_REQ) for _ in range(3)]
    with pytest.raises(HttpFailure) as info:
        _run(_Script(*errors), idempotent=False)
    assert info.value.exhausted


def test_unreadable_exceptions_of_an_idempotent_call_propagate_untouched() -> None:
    with pytest.raises(KeyError):
        _run(_Script(KeyError("bug")), idempotent=True)


def test_an_unreadable_create_failure_is_uncertain_and_never_resent() -> None:
    """Nothing proves such a create was not accepted (an SDK can fail parsing a
    2xx): it may exist, so it is neither resent nor treated as refused (F5)."""
    script = _Script(KeyError("id"), 200)
    with pytest.raises(BatchSubmitUncertain, match="KeyError") as info:
        _run(script, idempotent=False)
    assert isinstance(info.value.__cause__, KeyError)
    assert script.calls == 1


def test_a_create_failure_carrying_a_success_status_is_uncertain() -> None:
    """e.g. the SDK's ``APIResponseValidationError`` (status 200): the provider
    accepted the create; only reading its answer failed (F5)."""
    assert classify_create_failure(status_code=200) == CreateOutcome.UNCERTAIN
    assert classify_create_failure(status_code=201, detail="quota") == CreateOutcome.UNCERTAIN


def test_non_http_results_pass_through() -> None:
    assert call_with_retries(
        lambda: {"id": "b"}, what="x", idempotent=False, sleep=lambda _s: None
    ) == {"id": "b"}


def test_batch_rejected_items_are_not_retryable_by_default() -> None:
    item = BatchItemError("a", "batch_rejected", "over the enqueued-token limit")
    assert not item.retryable


def test_encode_cache_is_keyed_on_id_and_content() -> None:
    cache: EncodeCache[str] = EncodeCache()
    a = BatchRequest("a", {"model": "m", "messages": [{"role": "user", "content": "x"}]})
    cache.put(a, "body-x")
    assert cache.get(BatchRequest("a", dict(a.kwargs))) == "body-x"
    changed = BatchRequest("a", {"model": "m", "messages": [{"role": "user", "content": "y"}]})
    assert cache.get(changed) is None
    cache.put(changed, "body-y")
    assert len(cache) == 1  # one entry per id: a stale encoding never accumulates
    cache.release(["a", "missing"])
    assert len(cache) == 0
