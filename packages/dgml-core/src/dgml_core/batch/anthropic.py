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

"""Anthropic Message Batches backend (``POST /v1/messages/batches``).

Half-price, asynchronous completions on the first-party Anthropic API. Two
design rules keep this a pure transport rather than a second request encoder:

* **Encoding IS the synchronous path.** litellm has no batch *creation* for
  Anthropic, but its synchronous Anthropic route already turns our
  OpenAI-format kwargs into the exact ``/v1/messages`` body. ``encode`` runs
  ``litellm.completion`` against the shared capturing HTTP handler
  (:mod:`dgml_core.batch._capture`) that records the body litellm was about to
  POST and never sends it, so the ``params`` placed in the batch are
  byte-for-byte what the sync call would have sent (the batch API rejects only
  ``stream``; nothing else differs).
* **Decoding IS litellm's.** A succeeded result carries an Anthropic message
  object; ``AnthropicConfig.transform_parsed_response`` — the same function
  the sync route ends in — turns it into a ``ModelResponse`` shaped exactly
  like a ``litellm.completion`` return, so every caller's parsing and
  :func:`dgml_core.usage.extract_cost_and_tokens` are unchanged. The one
  difference is price: ``_hidden_params["response_cost"]`` is litellm's
  standard-rate cost times :data:`BATCH_PRICE_MULTIPLIER`.

Provider calls go through the shared retry policy
(:mod:`dgml_core.batch._http`): the create POST is never resent once it may
have been accepted (the endpoint takes no idempotency key).

litellm internals this module depends on (pin the litellm range when they
change): ``litellm.llms.custom_httpx.http_handler.HTTPHandler`` (the sync
route's HTTP seam, and its ``post`` signature),
``litellm.llms.anthropic.chat.transformation.AnthropicConfig
.transform_parsed_response``, and ``litellm.completion_cost``.

No optional dependency: ``httpx`` ships with litellm.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dgml_core.batch._capture import (
    HTTPX_HANDLER,
    CaptureFailed,
    EncodeCache,
    capture_sync_request,
)
from dgml_core.batch._http import (
    HttpFailure,
    call_with_retries,
    iter_response_lines,
    request,
    stream_request,
    uncertain_create,
)
from dgml_core.batch.registry import BackendConfig, register_backend
from dgml_core.batch.types import (
    BatchItemError,
    BatchJob,
    BatchNotFound,
    BatchRequest,
    BatchState,
    BatchStatus,
    ItemErrorKind,
)

PROVIDER = "anthropic"
DEFAULT_API_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
# Message Batches bill every token at half the standard price.
BATCH_PRICE_MULTIPLIER = 0.5
MAX_REQUESTS = 100_000
MAX_BYTES = 256 * 1024 * 1024
# A batch not ended within 24 hours of creation expires (Message Batches docs).
MAX_WAIT_S = 24 * 3600.0
# How long a canceled batch may sit `canceling` (measured 2026-09-30, n=17:
# bimodal — most settle in 3-50 s, about a third in 280-385 s, max 383 s; no
# work was done while canceling). 450 s covers the slow mode with margin.
CANCEL_SETTLE_S = 450.0
_RETRY_ATTEMPTS = 3
# What the capturing handler answers with so litellm's sync route completes
# its (discarded) response transform without ever reaching the network.
_DRY_RUN_MESSAGE: dict[str, Any] = {
    "id": "msg_dry_run",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": ""}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 0, "output_tokens": 0},
}


class AnthropicBatchError(Exception):
    """A transport or protocol failure talking to the batches endpoint.

    Per-request outcomes never raise; they come back as
    :class:`BatchItemError` from ``results``. This is for the batch as a
    whole: a rejected create, a poll that cannot be read, a missing
    ``results_url``.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class _Encoded:
    params: dict[str, Any]
    # The sync route's ``anthropic-beta`` header for this request, if any.
    # Batch requests carry no per-request headers, so ``submit`` unions the
    # betas of a batch onto the create call.
    betas: tuple[str, ...]


def _split_betas(header: str | None) -> tuple[str, ...]:
    if not header:
        return ()
    return tuple(sorted({b.strip() for b in header.split(",") if b.strip()}))


def _api_root(api_base: str | None) -> str:
    """The origin of the Anthropic API, from either form ``api_base`` takes.

    litellm's convention lets ``api_base`` name the full ``/v1/messages``
    endpoint; the batches endpoint hangs off the same ``/v1``.
    """
    root = (api_base or DEFAULT_API_BASE).rstrip("/")
    for suffix in ("/v1/messages/batches", "/v1/messages", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    return root


def _bare_model(model: str) -> str:
    return model.split("/", 1)[1] if model.startswith("anthropic/") else model


class AnthropicBatchBackend:
    """See the module docstring.

    ``http_client`` and ``retry_delay`` exist for tests (a mock transport and
    no backoff); production callers take the defaults.
    """

    provider = PROVIDER
    max_requests = MAX_REQUESTS
    max_bytes = MAX_BYTES
    max_wait_s = MAX_WAIT_S
    cancel_settle_s = CANCEL_SETTLE_S

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        http_client: Any | None = None,
        retry_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        import httpx

        self._model = model
        self._api_key = api_key
        self._root = _api_root(api_base)
        self._client: httpx.Client = http_client or httpx.Client(
            timeout=httpx.Timeout(120.0, connect=15.0)
        )
        self._retry_delay = retry_delay
        self._sleep = sleep
        # Filled by ``encode`` and consumed by ``submit`` so a request is
        # encoded once even though planning encodes it first. ``submit`` drops
        # its entries whether or not it succeeds; ``release`` drops the rest.
        self._cache: EncodeCache[_Encoded] = EncodeCache()

    # ---- encoding ---------------------------------------------------------

    def _encode(self, request: BatchRequest) -> _Encoded:
        cached = self._cache.get(request)
        if cached is not None:
            return cached
        try:
            # The dry run never authenticates, so planning works before
            # credentials resolve (litellm only checks a key is present).
            captured = capture_sync_request(
                request.kwargs,
                seam=HTTPX_HANDLER,
                reply=_DRY_RUN_MESSAGE,
                api_key=self._key(),
                expect_path="/v1/messages",
            )
        except CaptureFailed as exc:
            raise AnthropicBatchError(
                f"could not capture the request body for {request.custom_id!r}: {exc}"
            ) from exc
        params = dict(captured.body)
        params.pop("stream", None)  # the one Messages parameter batches reject
        betas = _split_betas(captured.headers.get("anthropic-beta"))
        return self._cache.put(request, _Encoded(params=params, betas=betas))

    def encode(self, request: BatchRequest) -> dict[str, Any]:
        """The ``{"custom_id", "params"}`` entry the create call carries."""
        return {"custom_id": request.custom_id, "params": self._encode(request).params}

    # ---- transport --------------------------------------------------------

    def _key(self) -> str | None:
        return self._api_key or os.environ.get("ANTHROPIC_API_KEY") or None

    def _headers(self) -> dict[str, str]:
        key = self._key()
        if not key:
            raise AnthropicBatchError(
                "no Anthropic API key: set ANTHROPIC_API_KEY or the model's api_key"
            )
        return {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "accept": "application/json",
        }

    def _request(
        self,
        method: str,
        url: str,
        *,
        idempotent: bool = True,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        """One HTTP call under the shared retry policy (:mod:`._http`).

        ``idempotent=False`` marks the batch create: it is retried only when
        the failure proves nothing was accepted, and raises
        :class:`BatchSubmitUncertain` / :class:`BatchRejected` otherwise.
        ``stream=True`` returns the response with its body unread (the caller
        iterates and closes it).
        """
        call = (stream_request if stream else request)(self._client, method, url, **kwargs)
        try:
            return call_with_retries(
                call,
                what=f"{method} {url}",
                idempotent=idempotent,
                attempts=_RETRY_ATTEMPTS,
                base_delay=self._retry_delay,
                sleep=self._sleep,
                detail_chars=500,
            )
        except HttpFailure as failure:
            if failure.exhausted:
                raise AnthropicBatchError(
                    f"{method} {url} failed after {failure.attempts} attempts: {failure.detail}"
                ) from failure
            raise AnthropicBatchError(
                f"{method} {url} failed: {failure.detail}", status_code=failure.status_code
            ) from failure

    def _batch_url(self, job_id: str = "") -> str:
        base = f"{self._root}/v1/messages/batches"
        return f"{base}/{job_id}" if job_id else base

    # ---- BatchBackend -----------------------------------------------------

    def submit(self, requests: Sequence[BatchRequest]) -> BatchJob:
        if not requests:
            raise AnthropicBatchError("cannot submit an empty batch")
        try:
            encoded = [(r.custom_id, self._encode(r)) for r in requests]
            headers = self._headers()
            betas = sorted({b for _, e in encoded for b in e.betas})
            if betas:
                headers["anthropic-beta"] = ",".join(betas)
            body = {"requests": [{"custom_id": cid, "params": e.params} for cid, e in encoded]}
            response = self._request(
                "POST", self._batch_url(), idempotent=False, headers=headers, json=body
            )
            try:  # the create answered 2xx: from here on, the batch exists
                obj = response.json()
                job_id = obj["id"] if isinstance(obj, dict) else None
                if not isinstance(job_id, str) or not job_id:
                    raise ValueError(f"no batch id in the create answer: {response.text[:200]!r}")
            except Exception as exc:
                raise uncertain_create(f"POST {self._batch_url()}", exc) from exc
        finally:
            # Submitted or failed, these encodings are spent: a retry re-encodes.
            self._cache.release(r.custom_id for r in requests)
        return BatchJob(
            provider=PROVIDER,
            job_id=job_id,
            custom_ids=tuple(r.custom_id for r in requests),
            extra={
                "results_url": obj.get("results_url"),
                "betas": betas,
                "batch": obj,
            },
        )

    @staticmethod
    def _status_from(obj: Mapping[str, Any]) -> BatchStatus:
        raw_state = str(obj.get("processing_status", ""))
        state = {
            "in_progress": BatchState.RUNNING,
            "canceling": BatchState.RUNNING,
            "ended": BatchState.ENDED,
        }.get(raw_state, BatchState.PENDING)
        counts = dict(obj.get("request_counts") or {})

        def n(name: str) -> int:
            value = counts.get(name, 0)
            return int(value) if isinstance(value, int) else 0

        return BatchStatus(
            state=state,
            succeeded=n("succeeded"),
            errored=n("errored"),
            expired=n("expired"),
            canceled=n("canceled"),
            processing=n("processing"),
            raw=dict(obj),
        )

    @staticmethod
    def _not_found(job: BatchJob, what: str, exc: AnthropicBatchError) -> BatchNotFound | None:
        """:class:`BatchNotFound` for a 404 (the batch was deleted or never
        existed), else ``None``."""
        if exc.status_code != 404:
            return None
        return BatchNotFound(f"anthropic batch {job.job_id} not found ({what}): {exc}")

    def poll(self, job: BatchJob) -> BatchStatus:
        try:
            response = self._request("GET", self._batch_url(job.job_id), headers=self._headers())
        except AnthropicBatchError as exc:
            if (missing := self._not_found(job, "poll", exc)) is not None:
                raise missing from exc
            raise
        obj = dict(response.json())
        job.extra["results_url"] = obj.get("results_url")
        job.extra["batch"] = obj
        return self._status_from(obj)

    def results(self, job: BatchJob) -> Iterator[tuple[str, Any | BatchItemError]]:
        url = job.extra.get("results_url")
        if not url:
            status = self.poll(job)
            url = job.extra.get("results_url")
            if not url:
                raise AnthropicBatchError(
                    f"batch {job.job_id} has no results yet (state {status.state.value})"
                )
        # Streamed: a results file can be as large as the batch (256 MB).
        try:
            response = self._request("GET", str(url), headers=self._headers(), stream=True)
        except AnthropicBatchError as exc:
            if (missing := self._not_found(job, "results", exc)) is not None:
                raise missing from exc
            raise
        for line in iter_response_lines(response):
            item = json.loads(line)
            custom_id = str(item.get("custom_id", ""))
            yield custom_id, self._decode_result(custom_id, dict(item.get("result") or {}))

    def cancel(self, job: BatchJob) -> None:
        try:
            self._request("POST", f"{self._batch_url(job.job_id)}/cancel", headers=self._headers())
        except AnthropicBatchError as exc:
            if (missing := self._not_found(job, "cancel", exc)) is not None:
                raise missing from exc
            raise

    def cleanup(self, job: BatchJob) -> None:
        """Delete the ended batch (``DELETE /v1/messages/batches/{id}``; its
        results go with it). A batch already gone (404) counts as cleaned. Any
        other failure — the endpoint refuses a batch still processing or
        canceling — raises :class:`AnthropicBatchError`: callers treat cleanup
        as best-effort and log the failure
        (:func:`~dgml_core.batch.executor.cleanup_batch`), never fail on it."""
        try:
            self._request("DELETE", self._batch_url(job.job_id), headers=self._headers())
        except AnthropicBatchError as exc:
            if exc.status_code == 404:
                return
            raise

    def release(self, custom_ids: Sequence[str]) -> None:
        self._cache.release(custom_ids)

    # ---- decoding ---------------------------------------------------------

    def _decode_result(self, custom_id: str, result: Mapping[str, Any]) -> Any | BatchItemError:
        kind = str(result.get("type", ""))
        if kind == "succeeded":
            message = result.get("message")
            if not isinstance(message, dict):
                return BatchItemError(custom_id, "invalid", "succeeded result carries no message")
            return self.decode_message(message)
        if kind == "errored":
            # The error object nests: {"type": "error", "error": {"type": ..., "message": ...}}.
            error = dict(result.get("error") or {})
            inner = error.get("error") if isinstance(error.get("error"), dict) else error
            etype = str(inner.get("type", "")) if isinstance(inner, dict) else ""
            emsg = str(inner.get("message", "")) if isinstance(inner, dict) else str(error)
            err_kind: ItemErrorKind = "invalid" if "invalid_request" in etype else "errored"
            return BatchItemError(custom_id, err_kind, f"{etype}: {emsg}".strip(": "))
        if kind == "expired":
            return BatchItemError(custom_id, "expired", "batch expired before this request ran")
        if kind == "canceled":
            return BatchItemError(
                custom_id, "canceled", "batch was canceled before this request ran"
            )
        return BatchItemError(custom_id, "invalid", f"unknown result type {kind!r}")

    def decode_message(self, message: Mapping[str, Any]) -> Any:
        """An Anthropic message object → the ``ModelResponse`` the sync path returns,
        priced at the batch rate."""
        import httpx
        import litellm
        from litellm import ModelResponse
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        model = _bare_model(str(message.get("model") or self._model or ""))
        raw = httpx.Response(200, request=httpx.Request("GET", self._batch_url()))
        response = AnthropicConfig().transform_parsed_response(
            completion_response=dict(message),
            raw_response=raw,
            model_response=ModelResponse(),
        )
        response.model = model
        response._hidden_params["custom_llm_provider"] = PROVIDER
        cost: float | None
        try:
            cost = float(
                litellm.completion_cost(
                    completion_response=response, model=model, custom_llm_provider=PROVIDER
                )
            )
            cost *= BATCH_PRICE_MULTIPLIER
        except Exception:  # unknown model price → unknown cost, as on the sync path
            cost = None
        response._hidden_params["response_cost"] = cost
        response._hidden_params["tier"] = "batch"
        return response


def _available() -> None:
    import httpx  # noqa: F401
    import litellm  # noqa: F401

    from dgml_core.batch.compat import require_supported_litellm

    require_supported_litellm()


def _factory(config: BackendConfig) -> AnthropicBatchBackend:
    return AnthropicBatchBackend(
        model=config.model, api_key=config.api_key, api_base=config.api_base
    )


register_backend(PROVIDER, _factory, available=_available)
