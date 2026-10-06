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

"""Gemini Developer API batch backend (``gemini/…`` models).

The Gemini Batch API (``models/{model}:batchGenerateContent``) runs a set of
``GenerateContentRequest`` bodies asynchronously at half price. litellm 1.85.1
has no client for it, so this module talks to it over ``httpx`` — already a
litellm dependency, so the backend needs no optional package — while reusing
litellm for the two translations that matter:

- **Encoding.** The per-request body is exactly what litellm's synchronous
  ``gemini/`` path posts. litellm builds that body deep inside
  ``VertexLLM.completion`` (``transform_request`` on the Gemini config raises
  ``NotImplementedError``), so rather than re-assemble its inputs — and drift
  from them — :meth:`GeminiBatchBackend.encode` runs ``litellm.completion`` on
  the request's own kwargs against the shared capturing HTTP handler
  (:mod:`dgml_core.batch._capture`) that records the body and refuses to
  send. Parity with the sync path holds by construction.
- **Decoding.** Each ``GenerateContentResponse`` goes through litellm's own
  Gemini → OpenAI response transform, so callers read ``choices``,
  ``finish_reason``, ``tool_calls`` and ``usage`` exactly as on a live
  ``litellm.completion`` return. ``_hidden_params["response_cost"]`` is
  litellm's standard-rate price times :data:`BATCH_RATE`.

Requests travel inline when the batch fits the inline cap, and as a JSONL
file uploaded through the File API otherwise; results are read back from the
finished operation inline or from its ``responsesFile``. The two modes are
invisible to callers.

Provider calls go through the shared retry policy (:mod:`dgml_core.batch._http`):
the ``batchGenerateContent`` create is never resent once it may have been
accepted (the endpoint takes no idempotency key).
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import httpx

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
from dgml_core.batch._policy import is_batch_limit_error
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

PROVIDER = "gemini"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com"
API_VERSION = "v1beta"
# The documented cap on an inline ``batchGenerateContent`` body; bigger
# batches go through the File API (2 GB per file).
INLINE_MAX_BYTES = 20 * 1024 * 1024
FILE_MAX_BYTES = 2 * 1024**3
# No documented per-batch request cap; the byte limits bind first.
MAX_REQUESTS = 100_000
# Batch usage is billed at 50% of the standard rate.
BATCH_RATE = 0.5
# A job still pending or running 48 hours after creation expires (Batch API docs).
MAX_WAIT_S = 48 * 3600.0
# How long a canceled job may take to end (measured 2026-09-30, n=6: 3-7 s;
# a canceled running job ends SUCCEEDED with every item failed). 60 s is ample.
CANCEL_SETTLE_S = 60.0

_RETRY_ATTEMPTS = 3
# What the capturing handler answers with so litellm's sync route completes
# its (discarded) response transform without ever reaching the network.
_DRY_RUN_RESPONSE: dict[str, Any] = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": ""}]}, "finishReason": "STOP", "index": 0}
    ],
    "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
}
_RETRY_BASE_DELAY_S = 2.0

# Operation states, with either the ``JOB_STATE_`` (docs) or ``BATCH_STATE_``
# (REST reference) prefix stripped.
_STATE_MAP: dict[str, BatchState] = {
    "PENDING": BatchState.PENDING,
    "RUNNING": BatchState.RUNNING,
    "SUCCEEDED": BatchState.ENDED,
    "FAILED": BatchState.FAILED,
    "CANCELLED": BatchState.CANCELED,
    "CANCELED": BatchState.CANCELED,
    # Expired: the job ended without serving every request; the unserved ones
    # surface as ``expired`` item errors so the driver resubmits them.
    "EXPIRED": BatchState.ENDED,
}

# google.rpc.Code numbers → status names, for errors that carry only ``code``.
_GRPC_CODES: dict[int, str] = {
    1: "CANCELLED",
    2: "UNKNOWN",
    3: "INVALID_ARGUMENT",
    4: "DEADLINE_EXCEEDED",
    5: "NOT_FOUND",
    7: "PERMISSION_DENIED",
    8: "RESOURCE_EXHAUSTED",
    9: "FAILED_PRECONDITION",
    10: "ABORTED",
    11: "OUT_OF_RANGE",
    12: "UNIMPLEMENTED",
    13: "INTERNAL",
    14: "UNAVAILABLE",
    16: "UNAUTHENTICATED",
}
# Statuses that would fail identically on resubmission.
_INVALID_STATUSES = frozenset(
    {
        "INVALID_ARGUMENT",
        "FAILED_PRECONDITION",
        "PERMISSION_DENIED",
        "NOT_FOUND",
        "UNAUTHENTICATED",
        "OUT_OF_RANGE",
        "UNIMPLEMENTED",
    }
)


class GeminiBatchError(Exception):
    """The Gemini batch endpoint refused or failed a request after retries."""


@dataclass
class _LoggingStub:
    """The only attribute litellm's parsed-response transform reads."""

    optional_params: dict[str, Any]


def _available() -> None:
    """Import-only probe: everything this backend needs ships with litellm, at a
    version whose internals it was verified against."""
    import httpx  # noqa: F401
    import litellm  # noqa: F401

    from dgml_core.batch.compat import require_supported_litellm

    require_supported_litellm()


def _bare_model(model: str) -> str:
    """``gemini/gemini-2.5-pro`` → ``gemini-2.5-pro`` (the URL form)."""
    return model.split("/", 1)[1] if model.startswith(f"{PROVIDER}/") else model


def _error_status(error: dict[str, Any]) -> str:
    status = error.get("status")
    if isinstance(status, str) and status:
        return status
    code = error.get("code")
    if isinstance(code, int) and not isinstance(code, bool):
        return _GRPC_CODES.get(code, "UNKNOWN")
    return "UNKNOWN"


def _item_error(custom_id: str, error: dict[str, Any]) -> BatchItemError:
    status = _error_status(error)
    message = str(error.get("message") or status)
    kind: ItemErrorKind
    if status == "CANCELLED":
        kind = "canceled"
    elif status in _INVALID_STATUSES:
        kind = "invalid"
    else:
        kind = "errored"
    return BatchItemError(custom_id, kind, f"{status}: {message}")


def _inlined_responses(container: dict[str, Any]) -> list[dict[str, Any]]:
    """``inlinedResponses`` as the REST reference nests it (an object holding
    a list) or as the docs' jq examples flatten it (a bare list)."""
    inlined = container.get("inlinedResponses")
    if isinstance(inlined, dict):
        inlined = inlined.get("inlinedResponses")
    if not isinstance(inlined, list):
        return []
    return [item for item in inlined if isinstance(item, dict)]


def _inline_size(inline: dict[str, Any], items: Sequence[dict[str, Any]]) -> int:
    """``len(json.dumps(inline, ensure_ascii=False).encode())`` without
    building that string: *inline* is the create body with *items* as its
    request list, and each item is serialized on its own (so the probe never
    holds more than one request's JSON at a time)."""
    requests: dict[str, Any] = inline["batch"]["input_config"]["requests"]
    held = requests["requests"]
    requests["requests"] = []
    try:
        shell = len(json.dumps(inline, ensure_ascii=False).encode("utf-8"))
    finally:
        requests["requests"] = held
    body = sum(len(json.dumps(item, ensure_ascii=False).encode("utf-8")) for item in items)
    return shell + body + 2 * max(0, len(items) - 1)  # ", " between list items


def _obj(value: Any) -> dict[str, Any]:
    """``value`` when it is a JSON object, else an empty one."""
    return value if isinstance(value, dict) else {}


def _to_int(value: Any) -> int:
    """batchStats counts arrive as int64 strings."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class GeminiBatchBackend:
    """:class:`~dgml_core.batch.BatchBackend` for the Gemini Developer API.

    One instance serves one model (``config.model``); every request submitted
    through it must name that model, since the batch endpoint is per model.
    ``transport``/``client`` exist for tests (an ``httpx.MockTransport``) and
    ``sleep`` so retry backoff can be observed without waiting.
    """

    provider = PROVIDER
    max_requests = MAX_REQUESTS
    max_bytes = FILE_MAX_BYTES
    max_wait_s = MAX_WAIT_S
    cancel_settle_s = CANCEL_SETTLE_S

    def __init__(
        self,
        config: BackendConfig,
        *,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = config.model
        self._bare = _bare_model(config.model)
        key = config.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                f"batch mode for {config.model!r} needs a Gemini API key: set generation "
                "credentials or the GEMINI_API_KEY / GOOGLE_API_KEY environment variable"
            )
        self._api_key = key
        base = (config.api_base or DEFAULT_API_BASE).rstrip("/")
        if base.endswith(f"/{API_VERSION}"):
            base = base[: -len(API_VERSION) - 1]
        self._base = base
        self._client = client or httpx.Client(transport=transport, timeout=timeout)
        self._sleep = sleep
        # encode runs for size planning and again at submit; the litellm round
        # trip runs once per distinct request. ``submit`` drops the entries it
        # spent, ``release`` the ones never submitted.
        self._bodies: EncodeCache[dict[str, Any]] = EncodeCache()
        self._submissions = 0
        # The finished operation a ``poll`` last saw, per job, for ``results``
        # to reuse: an ended inline batch's operation carries every response
        # twice (``metadata.output`` and ``response``), so fetching it again
        # right after the poll that saw it end doubles the largest download.
        self._ended: dict[str, dict[str, Any]] = {}

    # ── encode ───────────────────────────────────────────────────────────

    def encode(self, request: BatchRequest) -> dict[str, Any]:
        model = str(request.kwargs.get("model", ""))
        if _bare_model(model) != self._bare:
            raise ValueError(
                f"request {request.custom_id!r} names model {model!r}; this backend "
                f"serves {self.model!r}"
            )
        body = self._bodies.get(request)
        if body is None:
            body = self._bodies.put(request, self._sync_request_body(request.kwargs))
        return {"request": copy.deepcopy(body), "metadata": {"key": request.custom_id}}

    def release(self, custom_ids: Sequence[str]) -> None:
        self._bodies.release(custom_ids)

    def _sync_request_body(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """The body litellm's synchronous path would post for ``kwargs``.

        The shared capture (:func:`~dgml_core.batch._capture.capture_sync_request`)
        records the JSON body and raises before any bytes leave the process.
        The API key never enters the body (it travels as a header), so the
        backend's own key is supplied only to satisfy litellm's pre-flight check.
        """
        try:
            return capture_sync_request(
                kwargs,
                seam=HTTPX_HANDLER,
                reply=_DRY_RUN_RESPONSE,
                api_key=self._api_key,
                expect_path=":generateContent",
            ).body
        except CaptureFailed as exc:
            raise GeminiBatchError(str(exc)) from exc

    # ── submit ───────────────────────────────────────────────────────────

    def submit(self, requests: Sequence[BatchRequest]) -> BatchJob:
        batch = list(requests)
        if not batch:
            raise ValueError("cannot submit an empty batch")
        try:
            return self._submit(batch)
        finally:
            # Submitted or failed, these encodings are spent: a retry re-encodes.
            self._bodies.release(r.custom_id for r in batch)

    def _submit(self, batch: list[BatchRequest]) -> BatchJob:
        encoded = [self.encode(r) for r in batch]
        self._submissions += 1
        display_name = f"dgml-{self._bare}-{self._submissions}-{int(time.time())}"
        inline = {
            "batch": {
                "display_name": display_name,
                "input_config": {"requests": {"requests": encoded}},
            }
        }
        extra: dict[str, Any] = {"model": self.model, "display_name": display_name}
        # The payload is serialized once, one request at a time: a file-mode
        # batch may be 2 GB, so neither the inline-size probe nor the upload
        # may hold it as a second (or third) whole-batch copy.
        body: bytes
        if _inline_size(inline, encoded) <= INLINE_MAX_BYTES:
            body = json.dumps(inline, ensure_ascii=False).encode("utf-8")
            extra["mode"] = "inline"
        else:
            # UTF-8 JSON, exactly as planning (``request_size``) measured it.
            lines = [
                json.dumps(
                    {"key": item["metadata"]["key"], "request": item["request"]},
                    ensure_ascii=False,
                ).encode("utf-8")
                + b"\n"
                for item in encoded
            ]
            del encoded, inline
            file_name = self._upload_jsonl(lines, display_name=display_name)
            del lines
            payload = {
                "batch": {"display_name": display_name, "input_config": {"file_name": file_name}}
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            extra["mode"] = "file"
            extra["input_file"] = file_name
        create_url = f"{self._base}/{API_VERSION}/models/{self._bare}:batchGenerateContent"
        response = self._request(
            "POST",
            create_url,
            content=body,
            headers={"Content-Type": "application/json"},
            idempotent=False,
        )
        try:  # the create answered 2xx: from here on, the batch exists
            created = response.json()
            name = created.get("name") if isinstance(created, dict) else None
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"batch creation returned no operation name: {response.text[:200]!r}"
                )
        except Exception as exc:
            raise uncertain_create(f"POST {create_url}", exc) from exc
        return BatchJob(
            provider=self.provider,
            job_id=name,
            custom_ids=tuple(r.custom_id for r in batch),
            extra=extra,
        )

    def _upload_jsonl(self, lines: list[bytes], *, display_name: str) -> str:
        """Resumable File API upload of a JSONL payload (its *lines*, each
        ending in a newline, sent as they are — never joined into one more
        copy); returns ``files/…``."""
        size = sum(len(line) for line in lines)
        start = self._request(
            "POST",
            f"{self._base}/upload/{API_VERSION}/files",
            json={"file": {"display_name": display_name}},
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            },
        )
        upload_url = start.headers.get("x-goog-upload-url")
        if not upload_url:
            raise GeminiBatchError("File API upload start returned no X-Goog-Upload-URL header")
        finish = self._request(
            "POST",
            upload_url,
            content=lines,
            headers={
                "Content-Length": str(size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
        )
        file_name = (finish.json().get("file") or {}).get("name")
        if not isinstance(file_name, str) or not file_name:
            raise GeminiBatchError(f"File API upload returned no file name: {finish.text[:300]}")
        return file_name

    # ── poll / results / cancel ──────────────────────────────────────────

    @staticmethod
    def _not_found(job: BatchJob, what: str, exc: GeminiBatchError) -> BatchNotFound | None:
        """:class:`BatchNotFound` for a 404 (the batch — or its responses
        file — was deleted or never existed), else ``None``."""
        if getattr(exc.__cause__, "status_code", None) != 404:
            return None
        return BatchNotFound(f"gemini batch {job.job_id} not found ({what}): {exc}")

    def poll(self, job: BatchJob) -> BatchStatus:
        try:
            operation = self._operation(job)
        except GeminiBatchError as exc:
            if (missing := self._not_found(job, "poll", exc)) is not None:
                raise missing from exc
            raise
        status = self._status(operation)
        if status.done:
            # An ended operation never changes again: safe to hand to results.
            self._ended[job.job_id] = operation
        return status

    def results(self, job: BatchJob) -> Iterator[tuple[str, Any | BatchItemError]]:
        try:
            yield from self._results(job)
        except GeminiBatchError as exc:
            if (missing := self._not_found(job, "results", exc)) is not None:
                raise missing from exc
            raise

    def _results(self, job: BatchJob) -> Iterator[tuple[str, Any | BatchItemError]]:
        operation = self._ended.pop(job.job_id, None) or self._operation(job)
        status = self._status(operation)
        if not status.done:
            raise GeminiBatchError(
                f"batch {job.job_id!r} is still {status.state.value}; poll until it ends"
            )
        pending = set(job.custom_ids)
        seen: set[str] = set()
        for key, item in self._delivered(operation):
            if key in seen:
                continue  # a provider duplicate never overrides the first delivery
            seen.add(key)
            pending.discard(key)
            error = item.get("error")
            if isinstance(error, dict):
                yield key, _item_error(key, error)
                continue
            body = item.get("response")
            if not isinstance(body, dict):
                yield (
                    key,
                    BatchItemError(key, "errored", "result carries neither response nor error"),
                )
                continue
            yield key, self._decode(body)
        # Whatever the provider never returned: classify by how the job ended.
        raw_state = str(status.raw.get("state") or "")
        op_error = operation.get("error")
        for key in [c for c in job.custom_ids if c in pending]:
            if raw_state.endswith("EXPIRED"):
                yield key, BatchItemError(key, "expired", "batch expired before this request ran")
            elif status.state is BatchState.CANCELED:
                yield key, BatchItemError(key, "canceled", "batch was canceled")
            elif isinstance(op_error, dict):
                op_status = _error_status(op_error)
                op_message = str(op_error.get("message", ""))
                # The accepted batch failed as a whole at a size/quota limit:
                # nothing about this request, so the caller decides (see types).
                kind: ItemErrorKind = (
                    "batch_rejected"
                    if is_batch_limit_error(f"{op_status}: {op_message}")
                    else "errored"
                )
                yield key, BatchItemError(key, kind, f"batch failed: {op_status}: {op_message}")
            else:
                yield key, BatchItemError(key, "errored", "no result returned for this request")

    def cancel(self, job: BatchJob) -> None:
        try:
            self._request("POST", f"{self._base}/{API_VERSION}/{job.job_id}:cancel", json={})
        except GeminiBatchError as exc:
            if (missing := self._not_found(job, "cancel", exc)) is not None:
                raise missing from exc
            raise

    def cleanup(self, job: BatchJob) -> None:
        """Best-effort delete of an ended batch's provider-side artifacts: the
        uploaded input file (file mode, ``files.delete``) and the batch
        operation itself (``batches.delete``). The operation's
        ``responsesFile`` (``files/batch-<id>``) goes with the batch: verified
        live, it answers 404 once the batch is deleted. Every target is
        tried; one already gone (404 — or, for the input file, 403
        ``PERMISSION_DENIED``, which is how the File API answers for a file
        that no longer exists, while the same key could delete the batch)
        counts as deleted, and any other failure
        raises :class:`GeminiBatchError` naming each target that failed —
        callers treat cleanup as best-effort and log the failure
        (:func:`~dgml_core.batch.executor.cleanup_batch`). The File API also
        expires uploads on its own after 48 hours (the upload reply's
        ``expirationTime``)."""
        targets = []
        input_file = job.extra.get("input_file")
        if isinstance(input_file, str) and input_file:
            targets.append(input_file)
        targets.append(job.job_id)
        self._ended.pop(job.job_id, None)
        failures: list[str] = []
        # A File API file that no longer exists — deleted by an earlier
        # attempt of this cleanup, or expired — answers 403 PERMISSION_DENIED,
        # not 404. It is our own upload, so that means gone, provided the key
        # itself works (the batch delete below succeeds); otherwise the 403 is
        # a credential problem and stays a failure.
        gone_files: list[str] = []
        key_works = False
        for name in targets:
            try:
                self._request("DELETE", f"{self._base}/{API_VERSION}/{name}")
            except GeminiBatchError as exc:
                status = getattr(exc.__cause__, "status_code", None)
                if status == 404:
                    key_works = key_works or name == job.job_id
                    continue
                if status == 403 and name.startswith("files/") and "PERMISSION_DENIED" in str(exc):
                    gone_files.append(f"{name}: {exc}")
                    continue
                failures.append(f"{name}: {exc}")
            else:
                key_works = True
        if not key_works:
            failures = gone_files + failures
        if failures:
            raise GeminiBatchError(f"cleanup of {job.job_id} failed: {'; '.join(failures)}")

    # ── internals ────────────────────────────────────────────────────────

    def _operation(self, job: BatchJob) -> dict[str, Any]:
        payload = self._request("GET", f"{self._base}/{API_VERSION}/{job.job_id}").json()
        if not isinstance(payload, dict):
            raise GeminiBatchError(f"operation {job.job_id!r} returned a non-object payload")
        return payload

    def _status(self, operation: dict[str, Any]) -> BatchStatus:
        metadata = _obj(operation.get("metadata"))
        response = _obj(operation.get("response"))
        raw_state = str(metadata.get("state") or response.get("state") or "")
        suffix = raw_state.rsplit("STATE_", 1)[-1] if raw_state else ""
        state = _STATE_MAP.get(suffix)
        if state is None:
            if operation.get("done"):
                state = BatchState.FAILED if operation.get("error") else BatchState.ENDED
            else:
                state = BatchState.RUNNING if raw_state else BatchState.PENDING
        stats = _obj(metadata.get("batchStats"))
        succeeded = _to_int(stats.get("successfulRequestCount"))
        errored = _to_int(stats.get("failedRequestCount"))
        processing = _to_int(stats.get("pendingRequestCount"))
        expired = canceled = 0
        if suffix == "EXPIRED":
            expired, processing = processing, 0
        elif state is BatchState.CANCELED:
            canceled, processing = processing, 0
        return BatchStatus(
            state=state,
            succeeded=succeeded,
            errored=errored,
            expired=expired,
            canceled=canceled,
            processing=processing,
            raw={"state": raw_state, "done": bool(operation.get("done")), "batchStats": stats},
        )

    def _delivered(self, operation: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
        """Every per-request result the finished operation carries, keyed."""
        response = _obj(operation.get("response"))
        metadata = _obj(operation.get("metadata"))
        output = _obj(metadata.get("output"))
        items = _inlined_responses(response) or _inlined_responses(output)
        for item in items:
            meta = item.get("metadata")
            key = meta.get("key") if isinstance(meta, dict) else item.get("key")
            if isinstance(key, str):
                yield key, item
        file_name = response.get("responsesFile") or output.get("responsesFile")
        if isinstance(file_name, str) and file_name:
            # Streamed: a responses file can be as large as the batch (2 GB).
            download = self._request(
                "GET",
                f"{self._base}/download/{API_VERSION}/{file_name}:download?alt=media",
                stream=True,
            )
            for line in iter_response_lines(download):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and isinstance(row.get("key"), str):
                    yield row["key"], row

    def _decode(self, body: dict[str, Any]) -> Any:
        """``GenerateContentResponse`` → litellm ``ModelResponse`` at batch price."""
        import litellm
        from litellm import ModelResponse
        from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig

        raw = httpx.Response(
            200, json=body, request=httpx.Request("POST", f"{self._base}/batch-result")
        )
        config = GoogleAIStudioGeminiConfig()
        response = config._transform_google_generate_content_to_openai_model_response(
            completion_response=body,
            model_response=ModelResponse(),
            model=self._bare,
            logging_obj=cast(Any, _LoggingStub(optional_params={})),
            raw_response=raw,
        )
        try:
            cost = litellm.completion_cost(completion_response=response, model=self.model)
        except Exception:
            cost = None
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            response._hidden_params["response_cost"] = float(cost) * BATCH_RATE
        return response

    def _request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        content: bytes | Iterable[bytes] | None = None,
        headers: dict[str, str] | None = None,
        idempotent: bool = True,
        stream: bool = False,
    ) -> httpx.Response:
        """One HTTP call with the API key header under the shared retry policy
        (:mod:`._http`). ``idempotent=False`` marks the batch create: retried
        only when the failure proves nothing was accepted, raising
        :class:`BatchSubmitUncertain` / :class:`BatchRejected` otherwise.
        ``content`` may be a re-iterable of chunks (sent as is, so a retry
        resends it whole); ``stream=True`` returns the response with its body
        unread (the caller iterates and closes it)."""
        request_headers = {"x-goog-api-key": self._api_key, **(headers or {})}
        send = stream_request if stream else request
        try:
            return call_with_retries(
                send(
                    self._client, method, url, json=json, content=content, headers=request_headers
                ),
                what=f"{method} {url}",
                idempotent=idempotent,
                attempts=_RETRY_ATTEMPTS,
                base_delay=_RETRY_BASE_DELAY_S,
                sleep=self._sleep,
                detail_chars=300,
            )
        except HttpFailure as failure:
            raise GeminiBatchError(f"{method} {url} failed: {failure.detail}") from failure


def _factory(config: BackendConfig) -> GeminiBatchBackend:
    return GeminiBatchBackend(config)


def register() -> None:
    """Register this backend for the ``gemini`` provider."""
    register_backend(PROVIDER, _factory, available=_available)


__all__ = [
    "BATCH_RATE",
    "FILE_MAX_BYTES",
    "INLINE_MAX_BYTES",
    "GeminiBatchBackend",
    "GeminiBatchError",
    "register",
]
