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

"""OpenAI Batch API backend, built on litellm's native file-based batches.

Flow: the requests are serialized as JSONL (one ``{"custom_id", "method",
"url", "body"}`` line each), uploaded with ``litellm.create_file(purpose=
"batch")``, and submitted with ``litellm.create_batch``. ``poll`` reads
``litellm.retrieve_batch``; ``results`` downloads the output and error files
with ``litellm.file_content`` and turns each line back into a litellm
``ModelResponse`` (or a :class:`BatchItemError`).

Wire parity is the design constraint. ``encode`` runs the request's own
kwargs through the synchronous ``litellm.completion`` path against an OpenAI
SDK client whose ``httpx`` transport records the serialized request and never
sends it (the shared :mod:`dgml_core.batch._capture` seam, as for Anthropic),
so each line's ``body`` IS the sync wire body — param mapping,
``drop_params``, ``max_tokens`` → ``max_completion_tokens``, cache-marker
stripping, and every litellm-only kwarg (``num_retries``, ``extra_headers``,
``metadata``, …) handled by litellm itself. Per-request HTTP headers (e.g.
``extra_headers``) have no place in a batch line and are not carried.

Create safety: the OpenAI SDK retries a POST twice by default (on timeouts,
connection errors, 408/409/429/5xx) and sends no idempotency key with it, so
a create whose response was lost could be billed twice. ``create_batch`` is
therefore called with ``max_retries=0`` under the shared retry policy
(:mod:`dgml_core.batch._http`), which resends only when nothing was accepted.

Results are decoded with litellm's own chat-completion converter, so usage,
``finish_reason`` and ``tool_calls`` parse exactly as on a sync response, and
``_hidden_params["response_cost"]`` is litellm's price at the batch rate
(half the synchronous price).

Limits (OpenAI batch guide, 2026-09): 50,000 requests and 200 MB per input
file, 24-hour completion window.
"""

from __future__ import annotations

import copy
import io
import json
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from dgml_core.batch._capture import OPENAI_SDK, CaptureFailed, EncodeCache, capture_sync_request
from dgml_core.batch._http import HttpFailure, call_with_retries, uncertain_create
from dgml_core.batch._policy import is_batch_limit_error
from dgml_core.batch.registry import BackendConfig, register_backend
from dgml_core.batch.types import (
    BatchItemError,
    BatchJob,
    BatchNotFound,
    BatchRejected,
    BatchRequest,
    BatchState,
    BatchStatus,
    BatchThrottled,
    ItemErrorKind,
)
from dgml_core.errors import BatchUnavailable

PROVIDER = "openai"
ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
# The completion window: a batch not finished 24 hours after creation expires.
MAX_WAIT_S = 24 * 3600.0
# How long a canceled batch may stay `cancelling` (measured 2026-09-30:
# quantized on ~300 s sweeps — ~306 s, or ~606 s when canceled before
# `in_progress`). It KEEPS PROCESSING while cancelling (one batch went 2/20 →
# 19/20), so giving up early re-runs billed work synchronously. 660 s covers
# two sweeps.
CANCEL_SETTLE_S = 660.0
MAX_REQUESTS = 50_000
MAX_BYTES = 200 * 1024 * 1024
BATCH_PRICE_FACTOR = 0.5
_CREATE_ATTEMPTS = 3
# What the capturing transport answers with so litellm's sync route completes
# its (discarded) response transform without ever reaching the network.
_DRY_RUN_COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-dry-run",
    "object": "chat.completion",
    "created": 0,
    "model": "dry-run",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
}

_STATE = {
    "validating": BatchState.RUNNING,
    "in_progress": BatchState.RUNNING,
    "finalizing": BatchState.RUNNING,
    "cancelling": BatchState.RUNNING,
    "completed": BatchState.ENDED,
    "failed": BatchState.FAILED,
    "expired": BatchState.FAILED,
    "cancelled": BatchState.CANCELED,
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field off a pydantic object or a dict alike."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dict(dump())
    return dict(vars(obj))


_RESPONSES_REASON = (
    "litellm serves it through the OpenAI Responses API (/v1/responses), not Chat "
    "Completions, and batch mode encodes only /v1/chat/completions requests"
)


def _responses_unavailable(what: str) -> BatchUnavailable:
    return BatchUnavailable(
        f"batch mode is not available for {what}: {_RESPONSES_REASON}. Use a chat-mode "
        "model, or run without --batch"
    )


def responses_routed(kwargs: Mapping[str, Any]) -> bool:
    """Whether litellm serves a chat call with these ``kwargs`` through the
    Responses API instead of Chat Completions — the one rule behind every
    Responses-API refusal (pre-flight, backend resolve, encode).

    It is litellm's own bridge check, fed the fields that decide it: the model
    (cost-map mode ``responses`` — gpt-5-pro, o3-pro, the codex models — a
    ``responses/`` model, or ``litellm.route_all_chat_openai_to_responses``),
    and ``tools`` + ``reasoning_effort`` (gpt-5.4+ bridges that combination).
    An unknown model answers ``False``; the encode capture still refuses any
    request that reaches ``/v1/responses`` anyway."""
    model = kwargs.get("model")
    if not isinstance(model, str):
        return False
    try:
        from litellm.main import responses_api_bridge_check

        info, _ = responses_api_bridge_check(
            model=model,
            custom_llm_provider=PROVIDER,
            tools=kwargs.get("tools") or None,
            reasoning_effort=kwargs.get("reasoning_effort"),
        )
    except Exception:
        return False
    return isinstance(info, dict) and info.get("mode") == "responses"


def refuse_responses_routed(kwargs: Mapping[str, Any]) -> None:
    """Raise :class:`BatchUnavailable` when :func:`responses_routed` says a
    request shaped like ``kwargs`` would go to the Responses API. The registry
    runs it as this backend's pre-flight (``assert_batchable``), so a run is
    refused before any step pays, not when its first request is encoded."""
    if not responses_routed(kwargs):
        return
    model = kwargs.get("model")
    if responses_routed({"model": model}):
        raise _responses_unavailable(f"model {model!r}")
    raise _responses_unavailable(
        f"requests to {model!r} with tools and reasoning_effort (this stage sends both)"
    )


def refuse_responses_only_model(model: str) -> None:
    """Raise :class:`BatchUnavailable` when litellm routes every chat call of
    *model* to the Responses API, before any request."""
    refuse_responses_routed({"model": model})


def encode_body(kwargs: dict[str, Any], *, api_key: str | None = None) -> dict[str, Any]:
    """The chat-completions body the sync litellm path would send for ``kwargs``
    (captured from that path, never re-assembled; see the module docstring).

    A request litellm routes to the Responses API instead (a responses-only
    model, or gpt-5.4+ with tools and ``reasoning_effort``) raises
    :class:`BatchUnavailable`: nothing is sent, and it must not silently run
    at full price either. The rule is :func:`responses_routed` (the same one
    the pre-flight applies); the capture refusing any ``/v1/responses`` URL
    is the backstop for a shape that rule does not know."""
    refuse_responses_routed(kwargs)
    try:
        return capture_sync_request(
            kwargs,
            seam=OPENAI_SDK,
            reply=_DRY_RUN_COMPLETION,
            api_key=api_key,
            expect_path=ENDPOINT.removeprefix("/v1"),
        ).body
    except CaptureFailed as exc:
        if any(urlsplit(u).path.rstrip("/").endswith("/responses") for u in exc.refused_urls):
            raise _responses_unavailable(
                f"this request to {kwargs.get('model')!r} (its parameters route it there)"
            ) from exc
        raise


def decode_response(body: dict[str, Any]) -> Any:
    """A litellm ``ModelResponse`` from one chat-completion body, batch-priced."""
    import litellm
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )

    response = convert_to_model_response_object(
        response_object=copy.deepcopy(body), model_response_object=litellm.ModelResponse()
    )
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        response._hidden_params["response_cost"] = float(cost) * BATCH_PRICE_FACTOR
    return response


def _not_found(job: BatchJob, what: str, exc: BaseException) -> BatchNotFound | None:
    """:class:`BatchNotFound` for an SDK 404 (the batch — or its output file —
    was deleted or never existed), else ``None``."""
    if getattr(exc, "status_code", None) != 404:
        return None
    return BatchNotFound(f"openai batch {job.job_id} not found ({what}): {exc}")


def _item_error(custom_id: str, status_code: int | None, message: str) -> BatchItemError:
    if status_code is not None and 400 <= status_code < 500 and status_code != 429:
        return BatchItemError(custom_id, "invalid", message)
    return BatchItemError(custom_id, "errored", message)


class OpenAIBatchBackend:
    """:class:`~dgml_core.batch.BatchBackend` for ``openai/`` models."""

    provider = PROVIDER
    max_requests = MAX_REQUESTS
    max_bytes = MAX_BYTES
    max_wait_s = MAX_WAIT_S
    cancel_settle_s = CANCEL_SETTLE_S

    def __init__(
        self,
        config: BackendConfig,
        *,
        retry_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = config.model
        self._api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self._api_base = config.api_base
        self._retry_delay = retry_delay
        self._sleep = sleep
        # Planning encodes every request and ``submit`` encodes it again; one
        # litellm dry run serves both. ``submit`` drops what it spent.
        self._bodies: EncodeCache[dict[str, Any]] = EncodeCache()

    # ---- BatchBackend --------------------------------------------------

    def encode(self, request: BatchRequest) -> dict[str, Any]:
        body = self._bodies.get(request)
        if body is None:
            body = self._bodies.put(request, encode_body(request.kwargs, api_key=self._api_key))
        return {
            "custom_id": request.custom_id,
            "method": "POST",
            "url": ENDPOINT,
            "body": copy.deepcopy(body),
        }

    def release(self, custom_ids: Sequence[str]) -> None:
        self._bodies.release(custom_ids)

    def submit(self, requests: Sequence[BatchRequest]) -> BatchJob:
        try:
            return self._submit(requests)
        finally:
            # Submitted or failed, these encodings are spent: a retry re-encodes.
            self._bodies.release(r.custom_id for r in requests)

    def _submit(self, requests: Sequence[BatchRequest]) -> BatchJob:
        import litellm

        lines = [json.dumps(self.encode(r), ensure_ascii=False) for r in requests]
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        # An upload is safe to retry (a duplicate is an unbilled orphan file),
        # so it keeps the SDK's own retries.
        file_obj = litellm.create_file(
            file=("dgml-batch.jsonl", payload, "application/jsonl"),
            purpose="batch",
            custom_llm_provider=PROVIDER,
            **self._credentials(),
        )
        input_file_id = str(_get(file_obj, "id"))

        def create() -> Any:
            return litellm.create_batch(
                completion_window=COMPLETION_WINDOW,
                endpoint=ENDPOINT,
                input_file_id=input_file_id,
                custom_llm_provider=PROVIDER,
                max_retries=0,  # the shared policy decides; the SDK's retries would not
                **self._credentials(),
            )

        try:
            batch = call_with_retries(
                create,
                what="POST /v1/batches",
                idempotent=False,
                attempts=_CREATE_ATTEMPTS,
                base_delay=self._retry_delay,
                sleep=self._sleep,
            )
        except HttpFailure as failure:
            # Refused outright, so no batch references the upload: drop it.
            self._discard_file(input_file_id)
            if failure.cause is not None:
                raise failure.cause from None
            raise
        except (BatchRejected, BatchThrottled):
            self._discard_file(input_file_id)
            raise
        try:  # create_batch returned: from here on, the batch exists
            job_id = _get(batch, "id")
            if not isinstance(job_id, str) or not job_id:
                raise ValueError(f"no batch id in the create answer: {batch!r:.200}")
            raw = _jsonable(batch)
        except Exception as exc:
            raise uncertain_create("POST /v1/batches", exc) from exc
        return BatchJob(
            provider=PROVIDER,
            job_id=job_id,
            custom_ids=tuple(r.custom_id for r in requests),
            extra={"input_file_id": input_file_id, "batch": raw},
        )

    def poll(self, job: BatchJob) -> BatchStatus:
        import litellm

        try:
            batch = litellm.retrieve_batch(
                batch_id=job.job_id, custom_llm_provider=PROVIDER, **self._credentials()
            )
        except Exception as exc:
            if (missing := _not_found(job, "poll", exc)) is not None:
                raise missing from exc
            raise
        raw = _jsonable(batch)
        job.extra["batch"] = raw
        status = str(_get(batch, "status", ""))
        state = _STATE.get(status, BatchState.PENDING)
        counts = _get(batch, "request_counts")
        total = int(_get(counts, "total", 0) or 0)
        completed = int(_get(counts, "completed", 0) or 0)
        failed = int(_get(counts, "failed", 0) or 0)
        processing = max(0, total - completed - failed) if state is BatchState.RUNNING else 0
        expired = 0
        if status == "expired":
            expired = max(0, total - completed - failed)
        return BatchStatus(
            state=state,
            succeeded=completed,
            errored=failed,
            expired=expired,
            processing=processing,
            raw=raw,
        )

    def results(self, job: BatchJob) -> Iterator[tuple[str, Any | BatchItemError]]:
        try:
            yield from self._results(job)
        except Exception as exc:
            if (missing := _not_found(job, "results", exc)) is not None:
                raise missing from exc
            raise

    def _results(self, job: BatchJob) -> Iterator[tuple[str, Any | BatchItemError]]:
        batch = job.extra.get("batch") or {}
        seen: set[str] = set()
        for key in ("output_file_id", "error_file_id"):
            file_id = batch.get(key)
            if not file_id:
                continue
            for line in self._read_lines(str(file_id)):
                custom_id, outcome = self._decode_line(line)
                seen.add(custom_id)
                yield custom_id, outcome
        # Requests the provider never answered. Why decides whether a retry can help.
        status = str(batch.get("status", ""))
        kind, message = _unanswered(status, batch)
        for custom_id in job.custom_ids:
            if custom_id not in seen:
                yield custom_id, BatchItemError(custom_id, kind, message)

    def cancel(self, job: BatchJob) -> None:
        import litellm

        try:
            batch = litellm.cancel_batch(
                batch_id=job.job_id, custom_llm_provider=PROVIDER, **self._credentials()
            )
        except Exception as exc:
            if (missing := _not_found(job, "cancel", exc)) is not None:
                raise missing from exc
            raise
        job.extra["batch"] = _jsonable(batch)

    def cleanup(self, job: BatchJob) -> None:
        """Delete the batch's input, output and error files. OpenAI has no
        batch-delete endpoint; the batch record itself stays. Every file is
        tried; one already gone (404) counts as deleted, and any other
        failure raises :class:`RuntimeError` naming each file that failed —
        callers treat cleanup as best-effort and log the failure
        (:func:`~dgml_core.batch.executor.cleanup_batch`)."""
        batch = job.extra.get("batch") or {}
        ids = [job.extra.get("input_file_id"), batch.get("input_file_id")]
        ids += [batch.get("output_file_id"), batch.get("error_file_id")]
        failures: list[str] = []
        for file_id in dict.fromkeys(i for i in ids if isinstance(i, str) and i):
            try:
                self._delete_file(file_id)
            except Exception as exc:
                if getattr(exc, "status_code", None) == 404:
                    continue
                failures.append(f"{file_id}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError(f"cleanup of {job.job_id} failed: {'; '.join(failures)}")

    # ---- internals -----------------------------------------------------

    def _discard_file(self, file_id: str) -> None:
        """:meth:`_delete_file` on a create's error path: a failure is
        ignored so the create's own error is what the caller sees (the File
        API keeps the orphaned upload until it is deleted by hand)."""
        try:
            self._delete_file(file_id)
        except Exception:
            return

    def _delete_file(self, file_id: str) -> None:
        import litellm

        litellm.file_delete(file_id=file_id, custom_llm_provider=PROVIDER, **self._credentials())

    def _credentials(self) -> dict[str, Any]:
        creds: dict[str, Any] = {}
        if self._api_key:
            creds["api_key"] = self._api_key
        if self._api_base:
            creds["api_base"] = self._api_base
        return creds

    def _read_lines(self, file_id: str) -> Iterator[dict[str, Any]]:
        import litellm

        content = litellm.file_content(
            file_id=file_id, custom_llm_provider=PROVIDER, **self._credentials()
        )
        for raw in _content_lines(content):
            raw = raw.strip()
            if raw:
                yield json.loads(raw)

    @staticmethod
    def _decode_line(line: dict[str, Any]) -> tuple[str, Any | BatchItemError]:
        custom_id = str(line.get("custom_id", ""))
        error = line.get("error")
        response = line.get("response") or {}
        status_code = response.get("status_code")
        body = response.get("body")
        if error:
            code = str(error.get("code") or "error")
            message = f"{code}: {error.get('message', '')}".strip()
            # A request the batch never ran is written to the error file with
            # no response and one of these codes: it expired (retryable) or
            # was canceled, like an unanswered request of such a batch.
            if code == "batch_expired":
                return custom_id, BatchItemError(custom_id, "expired", message)
            if code in ("batch_cancelled", "batch_canceled"):
                return custom_id, BatchItemError(custom_id, "canceled", message)
            return custom_id, _item_error(custom_id, status_code, message)
        if status_code != 200 or not isinstance(body, dict):
            detail = _get(_get(body, "error"), "message", "") if isinstance(body, dict) else ""
            return custom_id, _item_error(
                custom_id, status_code, f"HTTP {status_code}: {detail}".strip()
            )
        return custom_id, decode_response(body)


def _batch_errors(batch: dict[str, Any]) -> str:
    """The batch-level ``errors`` payload as one readable line ('' when absent)."""
    errors = batch.get("errors")
    data = _get(errors, "data", errors)
    if not data:
        return ""
    items = data if isinstance(data, list) else [data]
    parts = []
    for item in items:
        code = _get(item, "code", "") or ""
        text = _get(item, "message", "") or ""
        line = _get(item, "line")
        where = f" (line {line})" if line is not None else ""
        parts.append(f"{code}: {text}{where}".strip(": ").strip())
    return "; ".join(p for p in parts if p)


def _failed_at_batch_limit(batch: dict[str, Any]) -> bool:
    """A ``failed`` batch whose errors are all batch-level and name a limit.

    Any line-level error (it carries a ``line``) is a validation failure of
    the input, which resubmission would repeat.
    """
    errors = batch.get("errors")
    data = _get(errors, "data", errors)
    if not data:
        return False
    items = data if isinstance(data, list) else [data]
    if any(_get(item, "line") is not None for item in items):
        return False
    return any(
        is_batch_limit_error(f"{_get(item, 'code', '') or ''}: {_get(item, 'message', '') or ''}")
        for item in items
    )


def _unanswered(status: str, batch: dict[str, Any]) -> tuple[ItemErrorKind, str]:
    """Kind and message for a request the provider returned no line for.

    - ``failed`` at a batch-level limit (e.g. ``token_limit_exceeded`` — the
      organization's enqueued-token limit — with no line-level error):
      ``batch_rejected``; nothing is wrong with the requests themselves.
    - ``failed`` otherwise: the batch was rejected before running (input
      validation). The same input fails the same way, so the kind is
      non-retryable ``invalid`` and the message carries the batch's own
      ``errors``.
    - ``expired``: the 24-hour window closed first. Retryable ``expired``.
    - ``cancelled``/``cancelling``: non-retryable ``canceled``.
    - anything else (e.g. ``completed`` with a line missing): a provider-side
      gap, retryable ``errored``.
    """
    if status == "failed":
        detail = _batch_errors(batch)
        if _failed_at_batch_limit(batch):
            return "batch_rejected", "batch refused at a batch-level limit: " + detail
        return "invalid", "batch failed validation" + (f": {detail}" if detail else "")
    if status == "expired":
        return "expired", "batch expired before this request ran (24h window)"
    if status in ("cancelled", "cancelling"):
        return "canceled", "batch canceled before this request ran"
    return "errored", f"no result for this request (batch status {status!r})"


def _content_lines(content: Any) -> Iterator[str | bytes]:
    """The lines of a downloaded file, one at a time. A file can be as large
    as its batch (200 MB): it is iterated where the object allows (the SDK's
    binary content's ``iter_lines``), never decoded into one more whole-file
    string and split into a list of every line."""
    if isinstance(content, bytes | bytearray):
        yield from io.BytesIO(content)
        return
    if isinstance(content, str):
        yield from io.StringIO(content)
        return
    iter_lines = getattr(content, "iter_lines", None)
    if callable(iter_lines):
        yield from iter_lines()
        return
    data = getattr(content, "content", None)
    if isinstance(data, bytes):
        yield from io.BytesIO(data)
        return
    raise TypeError(f"unreadable file content object: {type(content).__name__}")


def _jsonable(obj: Any) -> dict[str, Any]:
    """A JSON-safe dict of a batch object (``BatchJob.extra`` is persisted)."""
    return dict(json.loads(json.dumps(_to_dict(obj), default=str)))


def _available() -> None:
    """Import-only probe: litellm's batch/file entry points exist, at a version
    whose internals this backend was verified against."""
    from litellm import (
        cancel_batch,
        create_batch,
        create_file,
        file_content,
        file_delete,
        retrieve_batch,
    )

    from dgml_core.batch.compat import require_supported_litellm

    del cancel_batch, create_batch, create_file, file_content, file_delete, retrieve_batch
    require_supported_litellm()


def make_backend(config: BackendConfig) -> OpenAIBatchBackend:
    refuse_responses_only_model(config.model)
    return OpenAIBatchBackend(config)


register_backend(PROVIDER, make_backend, available=_available, refuse=refuse_responses_routed)
