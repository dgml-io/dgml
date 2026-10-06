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

"""Gemini batch backend: wire parity with the sync path, decode parity,
transport behavior, and registry resolution. All offline.

Wire parity is measured independently of the backend's own capture: the sync
path runs for real (``llm.call*`` → ``_completion_with_retry`` →
``litellm.completion``) with litellm's ``HTTPHandler.post`` patched at the
class to record the posted body and answer with a canned Gemini response.
``encode`` must then produce that exact body from the kwargs the sync call
handed litellm.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import litellm
import pytest
from dgml_core import llm
from dgml_core.batch import (
    BackendConfig,
    BatchItemError,
    BatchJob,
    BatchRequest,
    BatchState,
    plan_batches,
    provider_of,
    registered_providers,
    resolve_backend,
)
from dgml_core.batch import gemini as gemini_mod
from dgml_core.batch.gemini import BATCH_RATE, GeminiBatchBackend, GeminiBatchError
from dgml_core.batch.types import BatchRejected, BatchSubmitUncertain
from dgml_core.usage import extract_cost_and_tokens
from litellm.llms.custom_httpx.http_handler import HTTPHandler

MODEL = "gemini/gemini-2.5-pro"
KEY = "test-gemini-key"
BASE = "https://generativelanguage.googleapis.com"

_PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit the result.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        },
    }
]
_FORCED = {"type": "function", "function": {"name": "submit"}}


def _text_response(text: str = "hello", finish: str = "STOP") -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": finish,
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 30,
            "totalTokenCount": 150,
        },
        "modelVersion": "gemini-2.5-pro",
    }


def _tool_response() -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"functionCall": {"name": "submit", "args": {"value": "42"}}}],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 200,
            "candidatesTokenCount": 12,
            "totalTokenCount": 212,
        },
        "modelVersion": "gemini-2.5-pro",
    }


class _SyncCapture:
    """Records what the real sync path sent: litellm kwargs and HTTP bodies."""

    def __init__(self) -> None:
        self.kwargs: list[dict[str, Any]] = []
        self.bodies: list[dict[str, Any]] = []
        self.responses: list[Any] = []


@pytest.fixture(autouse=True)
def _gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture
def sync_capture(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[dict[str, Any]]], _SyncCapture]:
    """Patch the sync transport; answer each POST with the next canned body."""

    def install(replies: list[dict[str, Any]]) -> _SyncCapture:
        cap = _SyncCapture()
        queue = list(replies)
        real_completion = litellm.completion

        def recording_completion(**kwargs: Any) -> Any:
            cap.kwargs.append(json.loads(json.dumps(kwargs, default=repr)))
            response = real_completion(**kwargs)
            cap.responses.append(response)
            return response

        def fake_post(self: Any, url: str, data: Any = None, json: Any = None, **_: Any) -> Any:
            body = json if json is not None else data
            cap.bodies.append(body if isinstance(body, dict) else __import__("json").loads(body))
            reply = queue.pop(0)
            return httpx.Response(200, json=reply, request=httpx.Request("POST", url))

        monkeypatch.setattr(litellm, "completion", recording_completion)
        monkeypatch.setattr(HTTPHandler, "post", fake_post)
        return cap

    return install


def _ok(result: Any) -> Any:
    """A delivered ModelResponse (fails the test on an item error)."""
    assert not isinstance(result, BatchItemError), result
    return result


def _backend(**kw: Any) -> GeminiBatchBackend:
    return GeminiBatchBackend(BackendConfig(model=MODEL, api_key=KEY), **kw)


# ── (a) wire parity ──────────────────────────────────────────────────────


def _run_text(cfg: llm.LLMConfig, content: list[dict[str, Any]]) -> None:
    llm.call(cfg, system_prompt="You transcribe documents.", user_content=content, cache=True)


SCENARIOS: dict[str, tuple[list[dict[str, Any]], Callable[[], object]]] = {
    "text_system_user": (
        [_text_response()],
        lambda: _run_text(
            llm.LLMConfig(model=MODEL, temperature=0.0), [{"type": "text", "text": "hi"}]
        ),
    ),
    "pdf_block": (
        [_text_response()],
        lambda: _run_text(
            llm.LLMConfig(model=MODEL, temperature=0.0, max_tokens=32000),
            llm.build_user_content(instruction_text="transcribe", pdf_bytes=_PDF),
        ),
    ),
    "image_block": (
        [_text_response()],
        lambda: _run_text(
            llm.LLMConfig(model=MODEL, temperature=0.0),
            llm.build_user_content(instruction_text="look", images=[_PNG]),
        ),
    ),
    "reasoning_effort": (
        [_text_response()],
        lambda: _run_text(
            llm.LLMConfig(model=MODEL, temperature=0.0, reasoning_effort="low"),
            [{"type": "text", "text": "think"}],
        ),
    ),
    "extra_safety_settings": (
        [_text_response()],
        lambda: _run_text(
            llm.LLMConfig(
                model=MODEL,
                temperature=0.0,
                extra={
                    "safety_settings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
                    ]
                },
            ),
            [{"type": "text", "text": "hi"}],
        ),
    ),
    "tools_forced": (
        [_tool_response()],
        lambda: llm.call_with_tools(
            llm.LLMConfig(
                model=MODEL,
                max_tokens=None,
                max_completion_tokens=8000,
                temperature=0.0,
                reasoning_effort="high",
            ),
            messages=[
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": [{"type": "text", "text": "SCHEMA"}]},
            ],
            tools=_TOOLS,
            tool_choice=_FORCED,
            cache=True,
        ),
    ),
    "continuation_user_turn": (
        [_text_response("part-1", finish="MAX_TOKENS"), _text_response("part-2")],
        lambda: llm.call_continued(
            llm.LLMConfig(model=MODEL, temperature=0.0),
            system_prompt="SYS",
            user_content=[{"type": "text", "text": "go"}],
            cache=True,
        ),
    ),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_encode_matches_the_sync_request_body(
    name: str, sync_capture: Callable[[list[dict[str, Any]]], _SyncCapture]
) -> None:
    replies, run = SCENARIOS[name]
    cap = sync_capture(list(replies))
    run()
    # Snapshot before encoding: encode itself goes through litellm.completion,
    # which the recorder would otherwise log as another sync call.
    sent_kwargs, sent_bodies = list(cap.kwargs), list(cap.bodies)
    assert len(sent_bodies) == len(replies) == len(sent_kwargs)
    backend = _backend()
    for i, (kwargs, body) in enumerate(zip(sent_kwargs, sent_bodies, strict=True)):
        encoded = backend.encode(BatchRequest(f"{name}-{i}", kwargs))
        assert encoded["metadata"] == {"key": f"{name}-{i}"}
        assert encoded["request"] == body, f"{name} step {i} diverges from the sync body"
        assert KEY not in json.dumps(encoded)


def test_continuation_second_step_carries_the_partial_and_a_user_turn(
    sync_capture: Callable[[list[dict[str, Any]]], _SyncCapture],
) -> None:
    replies, run = SCENARIOS["continuation_user_turn"]
    cap = sync_capture(list(replies))
    run()
    sent = list(cap.kwargs)
    second = _backend().encode(BatchRequest("c1", sent[1]))["request"]
    roles = [c["role"] for c in second["contents"]]
    assert roles[-2:] == ["model", "user"]
    assert second["contents"][-2]["parts"][0]["text"] == "part-1"


def test_encode_rejects_a_request_for_another_model() -> None:
    kwargs = {"model": "gemini/gemini-2.5-flash", "messages": [{"role": "user", "content": "x"}]}
    with pytest.raises(ValueError, match=re.escape("serves 'gemini/gemini-2.5-pro'")):
        _backend().encode(BatchRequest("x", kwargs))


def test_encode_runs_litellm_once_per_distinct_request(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    calls: list[int] = []
    real = backend._sync_request_body

    def counting(kwargs: dict[str, Any]) -> dict[str, Any]:
        calls.append(1)
        return real(kwargs)

    monkeypatch.setattr(backend, "_sync_request_body", counting)
    kwargs = {"model": MODEL, "messages": [{"role": "user", "content": "x"}]}
    req = BatchRequest("a", kwargs)
    first = backend.encode(req)
    backend.encode(req)
    backend.encode(BatchRequest("a", dict(kwargs)))  # equal content, new dict
    assert len(calls) == 1
    changed = backend.encode(
        BatchRequest("a", {**kwargs, "messages": [{"role": "user", "content": "y"}]})
    )
    assert len(calls) == 2
    assert changed != first
    # Callers get their own copy; mutating it never poisons the cache.
    first["request"]["contents"].clear()
    assert backend.encode(req)["request"]["contents"]


# ── (b) decode parity ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("reply", "run_name"),
    [(_text_response(), "text_system_user"), (_tool_response(), "tools_forced")],
)
def test_decode_matches_the_sync_response(
    reply: dict[str, Any],
    run_name: str,
    sync_capture: Callable[[list[dict[str, Any]]], _SyncCapture],
) -> None:
    cap = sync_capture([reply])
    SCENARIOS[run_name][1]()
    sync = cap.responses[0]
    batch = _backend()._decode(reply)

    s_choice, b_choice = sync.choices[0], batch.choices[0]
    assert b_choice.finish_reason == s_choice.finish_reason
    assert b_choice.message.content == s_choice.message.content
    s_calls = [
        (c.function.name, json.loads(c.function.arguments))
        for c in s_choice.message.tool_calls or []
    ]
    b_calls = [
        (c.function.name, json.loads(c.function.arguments))
        for c in b_choice.message.tool_calls or []
    ]
    assert b_calls == s_calls

    s_usage, b_usage = extract_cost_and_tokens(sync), extract_cost_and_tokens(batch)
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "cache_read_tokens"):
        assert b_usage[field] == s_usage[field], field
    assert s_usage["cost_usd"] is not None
    assert b_usage["cost_usd"] == pytest.approx(s_usage["cost_usd"] * BATCH_RATE)


def test_decode_reads_both_access_styles() -> None:
    batch = _backend()._decode(_text_response("hi"))
    assert batch["choices"][0]["message"]["content"] == "hi"
    assert batch.choices[0].message.content == "hi"


# ── (c) transport: submit / poll / results / cancel ──────────────────────


class _Server:
    """A scripted Gemini batch endpoint behind ``httpx.MockTransport``."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.routes: dict[tuple[str, str], list[httpx.Response]] = {}

    def add(self, method: str, path: str, *responses: httpx.Response) -> None:
        self.routes.setdefault((method, path), []).extend(responses)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        key = (request.method, request.url.path)
        queue = self.routes.get(key)
        if not queue:
            return httpx.Response(404, json={"error": {"message": f"no route {key}"}})
        return queue.pop(0) if len(queue) > 1 else queue[0]

    def backend(self, sleeps: list[float] | None = None) -> GeminiBatchBackend:
        record = sleeps if sleeps is not None else []
        return _backend(transport=httpx.MockTransport(self.handler), sleep=record.append)


def _req(cid: str, text: str = "hi") -> BatchRequest:
    return BatchRequest(cid, {"model": MODEL, "messages": [{"role": "user", "content": text}]})


CREATE = "/v1beta/models/gemini-2.5-pro:batchGenerateContent"
OP = "batches/abc123"


def _op(state: str, *, response: dict[str, Any] | None = None, **meta: Any) -> httpx.Response:
    # Shaped like the live service: ``done`` is absent until the batch ends,
    # and an ended batch carries its results both as ``response`` and under
    # ``metadata.output``.
    body: dict[str, Any] = {"name": OP, "metadata": {"state": state, **meta}}
    if response is not None:
        body["done"] = True
        body["response"] = response
        body["metadata"]["output"] = response
    return httpx.Response(200, json=body)


def test_submit_inline_posts_the_encoded_requests_with_the_key_header() -> None:
    server = _Server()
    server.add("POST", CREATE, _op("BATCH_STATE_PENDING"))
    backend = server.backend()
    job = backend.submit([_req("a"), _req("b", "yo")])

    assert job.provider == "gemini" and job.job_id == OP
    assert job.custom_ids == ("a", "b")
    assert job.extra["mode"] == "inline"
    create = server.calls[-1]
    assert create.headers["x-goog-api-key"] == KEY
    payload = json.loads(create.content)
    requests = payload["batch"]["input_config"]["requests"]["requests"]
    assert [r["metadata"]["key"] for r in requests] == ["a", "b"]
    assert requests[0]["request"] == backend.encode(_req("a"))["request"]
    # A persisted job resumes against a fresh backend instance.
    assert BatchJob.from_json(job.to_json()) == job


def test_submit_switches_to_file_mode_above_the_inline_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemini_mod, "INLINE_MAX_BYTES", 10)
    server = _Server()
    upload_url = "https://generativelanguage.googleapis.com/upload/session/xyz"
    server.add(
        "POST",
        "/upload/v1beta/files",
        httpx.Response(200, headers={"x-goog-upload-url": upload_url}, json={}),
    )
    server.add(
        "POST", "/upload/session/xyz", httpx.Response(200, json={"file": {"name": "files/in1"}})
    )
    server.add("POST", CREATE, _op("BATCH_STATE_PENDING"))
    job = server.backend().submit([_req("a"), _req("b")])

    assert job.extra == {**job.extra, "mode": "file", "input_file": "files/in1"}
    start, upload, create = server.calls
    assert start.headers["x-goog-upload-protocol"] == "resumable"
    assert start.headers["x-goog-upload-command"] == "start"
    assert start.headers["x-goog-upload-header-content-type"] == "application/jsonl"
    assert upload.headers["x-goog-upload-command"] == "upload, finalize"
    lines = [json.loads(line) for line in upload.content.decode().splitlines()]
    assert [line["key"] for line in lines] == ["a", "b"]
    assert "request" in lines[0]
    assert json.loads(create.content)["batch"]["input_config"] == {"file_name": "files/in1"}


@pytest.mark.parametrize(
    ("raw", "state"),
    [
        ("BATCH_STATE_PENDING", BatchState.PENDING),
        ("JOB_STATE_RUNNING", BatchState.RUNNING),
        ("BATCH_STATE_SUCCEEDED", BatchState.ENDED),
        ("JOB_STATE_FAILED", BatchState.FAILED),
        ("BATCH_STATE_CANCELLED", BatchState.CANCELED),
        ("BATCH_STATE_EXPIRED", BatchState.ENDED),
    ],
)
def test_poll_maps_states(raw: str, state: BatchState) -> None:
    server = _Server()
    server.add(
        "GET",
        f"/v1beta/{OP}",
        _op(raw, batchStats={"requestCount": "3", "pendingRequestCount": "1"}),
    )
    job = BatchJob(provider="gemini", job_id=OP, custom_ids=("a", "b", "c"))
    status = server.backend().poll(job)
    assert status.state is state
    if raw.endswith("EXPIRED"):
        assert status.expired == 1 and status.processing == 0


def test_poll_reports_batch_stats() -> None:
    server = _Server()
    stats = {
        "requestCount": "3",
        "successfulRequestCount": "2",
        "failedRequestCount": "1",
        "pendingRequestCount": "0",
    }
    server.add("GET", f"/v1beta/{OP}", _op("BATCH_STATE_SUCCEEDED", batchStats=stats, response={}))
    status = server.backend().poll(
        BatchJob(provider="gemini", job_id=OP, custom_ids=("a", "b", "c"))
    )
    assert (status.succeeded, status.errored, status.processing) == (2, 1, 0)
    assert status.done


def _inline(*items: dict[str, Any], nested: bool = True) -> dict[str, Any]:
    return {"inlinedResponses": {"inlinedResponses": list(items)} if nested else list(items)}


@pytest.mark.parametrize("nested", [True, False])
def test_results_decode_success_and_classify_item_errors(nested: bool) -> None:
    server = _Server()
    response = _inline(
        {"metadata": {"key": "b"}, "error": {"code": 3, "message": "bad request"}},
        {"metadata": {"key": "a"}, "response": _text_response("ok-a")},
        {"metadata": {"key": "c"}, "error": {"status": "UNAVAILABLE", "message": "busy"}},
        {"metadata": {"key": "a"}, "response": _text_response("duplicate")},
        nested=nested,
    )
    server.add("GET", f"/v1beta/{OP}", _op("BATCH_STATE_SUCCEEDED", response=response))
    job = BatchJob(provider="gemini", job_id=OP, custom_ids=("a", "b", "c", "d"))
    out = dict(server.backend().results(job))

    assert _ok(out["a"]).choices[0].message.content == "ok-a"
    assert _ok(out["a"])._hidden_params.get("response_cost") is not None
    assert isinstance(out["b"], BatchItemError) and out["b"].kind == "invalid"
    assert not out["b"].retryable
    assert (
        isinstance(out["c"], BatchItemError) and out["c"].kind == "errored" and out["c"].retryable
    )
    # Returned by nobody on a SUCCEEDED job: an error to retry, never silence.
    assert isinstance(out["d"], BatchItemError) and out["d"].kind == "errored"


def test_results_mark_unserved_items_expired_on_an_expired_job() -> None:
    server = _Server()
    response = _inline({"metadata": {"key": "a"}, "response": _text_response()})
    server.add("GET", f"/v1beta/{OP}", _op("BATCH_STATE_EXPIRED", response=response))
    out = dict(
        server.backend().results(BatchJob(provider="gemini", job_id=OP, custom_ids=("a", "b")))
    )
    assert not isinstance(out["a"], BatchItemError)
    assert (
        isinstance(out["b"], BatchItemError) and out["b"].kind == "expired" and out["b"].retryable
    )


def test_results_mark_unserved_items_canceled_on_a_canceled_job() -> None:
    server = _Server()
    server.add("GET", f"/v1beta/{OP}", _op("BATCH_STATE_CANCELLED", response={}))
    out = dict(server.backend().results(BatchJob(provider="gemini", job_id=OP, custom_ids=("a",))))
    assert isinstance(out["a"], BatchItemError) and out["a"].kind == "canceled"
    assert not out["a"].retryable


def test_results_read_a_responses_file() -> None:
    server = _Server()
    server.add(
        "GET",
        f"/v1beta/{OP}",
        _op("BATCH_STATE_SUCCEEDED", response={"responsesFile": "files/out1"}),
    )
    jsonl = "\n".join(
        [
            json.dumps({"key": "a", "response": _text_response("from-file")}),
            json.dumps({"key": "b", "error": {"status": "INTERNAL", "message": "x"}}),
        ]
    )
    server.add(
        "GET", "/download/v1beta/files/out1:download", httpx.Response(200, text=jsonl + "\n")
    )
    out = dict(
        server.backend().results(BatchJob(provider="gemini", job_id=OP, custom_ids=("a", "b")))
    )
    assert _ok(out["a"]).choices[0].message.content == "from-file"
    assert isinstance(out["b"], BatchItemError) and out["b"].kind == "errored"
    download = server.calls[-1]
    assert download.url.params["alt"] == "media"


class _Chunks(httpx.SyncByteStream):
    """A response body delivered in chunks, counting how many were read."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.read = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self.chunks:
            self.read += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


def test_a_responses_file_is_streamed_not_read_whole() -> None:
    """F6: the responses file (up to 2 GB) is iterated line by line."""
    server = _Server()
    server.add(
        "GET",
        f"/v1beta/{OP}",
        _op("BATCH_STATE_SUCCEEDED", response={"responsesFile": "files/out1"}),
    )
    error = {"status": "INTERNAL", "message": "x"}
    body = [(json.dumps({"key": f"k{i}", "error": error}) + "\n").encode() for i in range(40)]
    stream = _Chunks(body)
    server.add("GET", "/download/v1beta/files/out1:download", httpx.Response(200, stream=stream))
    ids = tuple(f"k{i}" for i in range(40))
    results = server.backend().results(BatchJob(provider="gemini", job_id=OP, custom_ids=ids))
    assert next(results)[0] == "k0"
    assert stream.read < len(body)
    assert len(list(results)) == 39 and stream.closed


def test_file_mode_submit_never_serializes_the_whole_batch_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F6: a file-mode payload (up to 2 GB) is built once, one line at a time —
    no whole-batch JSON string for the inline-size probe, no joined copy."""
    monkeypatch.setattr(gemini_mod, "INLINE_MAX_BYTES", 10)
    sizes: list[int] = []
    real_dumps = json.dumps

    def dumps(obj: Any, *args: Any, **kwargs: Any) -> str:
        out = real_dumps(obj, *args, **kwargs)
        sizes.append(len(out))
        return out

    server = _Server()
    upload_url = "https://generativelanguage.googleapis.com/upload/session/xyz"
    server.add(
        "POST",
        "/upload/v1beta/files",
        httpx.Response(200, headers={"x-goog-upload-url": upload_url}, json={}),
    )
    server.add(
        "POST", "/upload/session/xyz", httpx.Response(200, json={"file": {"name": "files/in1"}})
    )
    server.add("POST", CREATE, _op("BATCH_STATE_PENDING"))
    backend = server.backend()
    requests = [_req(f"r{i}", f"text number {i}") for i in range(40)]
    for request in requests:
        backend.encode(request)  # planning's encodes: not part of the submit
    monkeypatch.setattr(json, "dumps", dumps)  # the module gemini serializes with
    backend.submit(requests)
    monkeypatch.setattr(json, "dumps", real_dumps)

    upload = server.calls[1]
    lines = upload.content.decode().splitlines()
    assert [json.loads(line)["key"] for line in lines] == [f"r{i}" for i in range(40)]
    assert int(upload.headers["content-length"]) == len(upload.content)
    assert max(sizes) < len(upload.content) / 10  # one request at a time, at most


@pytest.mark.parametrize("call", ["poll", "results", "cancel", "responses_file"])
def test_a_batch_the_provider_no_longer_knows_is_batch_not_found(call: str) -> None:
    from dgml_core.batch.types import BatchNotFound

    server = _Server()  # every unscripted route answers 404
    if call == "responses_file":
        server.add(
            "GET",
            f"/v1beta/{OP}",
            _op("BATCH_STATE_SUCCEEDED", response={"responsesFile": "files/out1"}),
        )
    job = BatchJob(provider="gemini", job_id=OP, custom_ids=("a",))
    backend = server.backend()
    with pytest.raises(BatchNotFound):
        if call in ("results", "responses_file"):
            list(backend.results(job))
        else:
            getattr(backend, call)(job)


def test_the_inline_size_probe_equals_the_serialized_inline_body() -> None:
    backend = _Server().backend()
    for n in (1, 2, 7):
        items = [backend.encode(_req(f"r{i}", f"Grüße {i} — 東京")) for i in range(n)]
        inline: dict[str, Any] = {
            "batch": {"display_name": "d", "input_config": {"requests": {"requests": items}}}
        }
        expected = len(json.dumps(inline, ensure_ascii=False).encode("utf-8"))
        assert gemini_mod._inline_size(inline, items) == expected
        assert inline["batch"]["input_config"]["requests"]["requests"] is items


def test_results_refuse_a_running_job() -> None:
    server = _Server()
    server.add("GET", f"/v1beta/{OP}", _op("BATCH_STATE_RUNNING"))
    with pytest.raises(GeminiBatchError, match="still running"):
        list(server.backend().results(BatchJob(provider="gemini", job_id=OP, custom_ids=("a",))))


def test_cancel_posts_to_the_operation() -> None:
    server = _Server()
    server.add("POST", f"/v1beta/{OP}:cancel", httpx.Response(200, json={}))
    server.backend().cancel(BatchJob(provider="gemini", job_id=OP, custom_ids=("a",)))
    assert server.calls[-1].method == "POST"
    assert server.calls[-1].url.path == f"/v1beta/{OP}:cancel"


def test_transient_errors_retry_with_backoff_then_succeed() -> None:
    """Idempotent calls (here a poll) retry transient failures with backoff."""
    server = _Server()
    server.add(
        "GET",
        f"/v1beta/{OP}",
        httpx.Response(503, text="overloaded"),
        httpx.Response(429, text="slow down"),
        _op("BATCH_STATE_RUNNING"),
    )
    sleeps: list[float] = []
    status = server.backend(sleeps).poll(BatchJob(provider="gemini", job_id=OP, custom_ids=("a",)))
    assert status.state is BatchState.RUNNING
    assert sleeps == [2.0, 4.0]


def test_create_retries_a_rate_limit_refusal_then_succeeds() -> None:
    server = _Server()
    server.add("POST", CREATE, httpx.Response(429, text="slow down"), _op("BATCH_STATE_PENDING"))
    sleeps: list[float] = []
    assert server.backend(sleeps).submit([_req("a")]).job_id == OP
    assert sleeps == [2.0] and len(server.calls) == 2


def test_create_still_quota_refused_is_throttled_not_rejected() -> None:
    from dgml_core.batch.types import BatchThrottled

    server = _Server()
    exhausted = httpx.Response(
        429,
        json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}},
    )
    server.add("POST", CREATE, exhausted, exhausted, exhausted)
    with pytest.raises(BatchThrottled, match="RESOURCE_EXHAUSTED"):
        server.backend([]).submit([_req("a")])


@pytest.mark.parametrize("status", [500, 503, 504])
def test_create_server_error_is_uncertain_and_never_resent(status: int) -> None:
    """The create may have landed: resending it could bill the batch twice."""
    server = _Server()
    server.add("POST", CREATE, httpx.Response(status, text="boom"), _op("BATCH_STATE_PENDING"))
    sleeps: list[float] = []
    with pytest.raises(BatchSubmitUncertain, match=f"HTTP {status}"):
        server.backend(sleeps).submit([_req("a")])
    assert len(server.calls) == 1 and sleeps == []


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, text="<html>gateway</html>"),
        httpx.Response(200, json={"metadata": {}}),  # no operation name
        httpx.Response(200, json=["not", "an", "object"]),
    ],
)
def test_an_accepted_create_whose_answer_cannot_be_read_is_uncertain(
    reply: httpx.Response,
) -> None:
    """A 2xx means the batch exists: never a refusal that reruns it sync (F5)."""
    server = _Server()
    server.add("POST", CREATE, reply)
    with pytest.raises(BatchSubmitUncertain, match="may have been created"):
        server.backend([]).submit([_req("a")])
    assert len(server.calls) == 1


def test_create_read_timeout_is_uncertain_and_never_resent() -> None:
    calls: list[httpx.Request] = []

    def slow(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ReadTimeout("no answer", request=request)

    backend = _backend(transport=httpx.MockTransport(slow), sleep=lambda _s: None)
    with pytest.raises(BatchSubmitUncertain, match="ReadTimeout"):
        backend.submit([_req("a")])
    assert len(calls) == 1


def test_create_refused_at_a_size_limit_is_batch_rejected() -> None:
    server = _Server()
    server.add(
        "POST",
        CREATE,
        httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": "Request payload size exceeds the limit: 20971520 bytes.",
                }
            },
        ),
    )
    with pytest.raises(BatchRejected, match="exceeds the limit"):
        server.backend().submit([_req("a")])
    assert len(server.calls) == 1


def test_create_refused_for_a_bad_key_is_a_backend_error() -> None:
    """Gemini reports a bad key as 400 INVALID_ARGUMENT: not a batch-level limit."""
    server = _Server()
    server.add(
        "POST",
        CREATE,
        httpx.Response(
            400,
            json={"error": {"status": "INVALID_ARGUMENT", "message": "API key not valid."}},
        ),
    )
    with pytest.raises(GeminiBatchError, match="API key not valid"):
        server.backend().submit([_req("a")])


def test_file_mode_lines_are_utf8_exactly_as_planning_measures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dgml_core.batch.chunking import request_size

    monkeypatch.setattr(gemini_mod, "INLINE_MAX_BYTES", 10)
    server = _Server()
    upload_url = "https://generativelanguage.googleapis.com/upload/session/xyz"
    server.add(
        "POST",
        "/upload/v1beta/files",
        httpx.Response(200, headers={"x-goog-upload-url": upload_url}, json={}),
    )
    server.add(
        "POST", "/upload/session/xyz", httpx.Response(200, json={"file": {"name": "files/in1"}})
    )
    server.add("POST", CREATE, _op("BATCH_STATE_PENDING"))
    backend = server.backend()
    request = _req("a", "Grüße — 東京 ✓")
    measured = request_size(request, backend)
    backend.submit([request])
    upload = server.calls[1]
    assert "Grüße — 東京 ✓".encode() in upload.content
    assert b"\\u" not in upload.content
    # The line carries the same request bytes planning measured, re-keyed.
    line = upload.content.rstrip(b"\n")
    item = backend.encode(request)
    rekeyed = json.dumps({"key": "a", "request": item["request"]}, ensure_ascii=False).encode(
        "utf-8"
    )
    assert line == rekeyed
    assert abs(len(line) - measured) < 64


def test_submit_drops_its_cached_encodings_and_release_drops_the_rest() -> None:
    server = _Server()
    server.add("POST", CREATE, _op("BATCH_STATE_PENDING"))
    backend = server.backend()
    backend.encode(_req("planned-only"))
    backend.encode(_req("a"))
    backend.submit([_req("a")])
    assert len(backend._bodies) == 1  # only the never-submitted request
    backend.release(["planned-only", "unknown"])
    assert len(backend._bodies) == 0


def test_cleanup_deletes_the_input_file_and_the_batch() -> None:
    server = _Server()
    server.add("DELETE", "/v1beta/files/in1", httpx.Response(200, json={}))
    server.add("DELETE", f"/v1beta/{OP}", httpx.Response(200, json={}))
    job = BatchJob(
        provider="gemini",
        job_id=OP,
        custom_ids=("a",),
        extra={"mode": "file", "input_file": "files/in1"},
    )
    server.backend().cleanup(job)
    assert [(c.method, c.url.path) for c in server.calls] == [
        ("DELETE", "/v1beta/files/in1"),
        ("DELETE", f"/v1beta/{OP}"),
    ]
    assert all(c.headers["x-goog-api-key"] == KEY for c in server.calls)


def test_cleanup_is_best_effort() -> None:
    server = _Server()  # every route 404s
    job = BatchJob(provider="gemini", job_id=OP, custom_ids=("a",), extra={"mode": "inline"})
    server.backend().cleanup(job)  # already gone (404) counts as deleted
    assert [(c.method, c.url.path) for c in server.calls] == [("DELETE", f"/v1beta/{OP}")]


_GONE_FILE = httpx.Response(
    403,
    json={
        "error": {
            "code": 403,
            "status": "PERMISSION_DENIED",
            "message": "You do not have permission to access the File in1 or it may not exist.",
        }
    },
)


@pytest.mark.parametrize("batch_reply", [httpx.Response(200, json={}), httpx.Response(404)])
def test_cleanup_counts_a_403_on_an_already_deleted_input_file_as_done(
    batch_reply: httpx.Response,
) -> None:
    """F7: the File API answers 403 PERMISSION_DENIED for a file that no longer
    exists (deleted by an earlier cleanup attempt, or expired after 48 h). With
    the same key otherwise working, that is 'already gone', not a failure owed
    forever."""
    server = _Server()
    server.add("DELETE", "/v1beta/files/in1", _GONE_FILE)
    server.add("DELETE", f"/v1beta/{OP}", batch_reply)
    job = BatchJob(
        provider="gemini",
        job_id=OP,
        custom_ids=("a",),
        extra={"mode": "file", "input_file": "files/in1"},
    )
    server.backend().cleanup(job)  # no raise: nothing is owed any more


def test_a_file_403_is_still_a_failure_when_the_key_itself_is_refused() -> None:
    server = _Server()
    server.add("DELETE", "/v1beta/files/in1", _GONE_FILE)
    server.add("DELETE", f"/v1beta/{OP}", _GONE_FILE)
    job = BatchJob(
        provider="gemini",
        job_id=OP,
        custom_ids=("a",),
        extra={"mode": "file", "input_file": "files/in1"},
    )
    with pytest.raises(GeminiBatchError, match="files/in1"):
        server.backend().cleanup(job)


def test_a_cleanup_failure_raises_after_trying_every_target_and_is_logged() -> None:
    from dgml_core.batch.executor import cleanup_batch

    server = _Server()
    server.add("DELETE", "/v1beta/files/in1", httpx.Response(400, json={"error": {}}))
    server.add("DELETE", f"/v1beta/{OP}", httpx.Response(200, json={}))
    job = BatchJob(
        provider="gemini",
        job_id=OP,
        custom_ids=("a",),
        extra={"mode": "file", "input_file": "files/in1"},
    )
    backend = server.backend()
    with pytest.raises(GeminiBatchError, match="files/in1"):
        backend.cleanup(job)
    assert [(c.method, c.url.path) for c in server.calls] == [
        ("DELETE", "/v1beta/files/in1"),
        ("DELETE", f"/v1beta/{OP}"),  # still tried after the first failed
    ]
    lines: list[str] = []
    cleanup_batch(backend, job, lines.append)  # never raises
    assert len(lines) == 1 and f"cleanup of {OP} failed" in lines[0]


def test_max_wait_is_the_48_hour_expiry() -> None:
    assert _backend().max_wait_s == 48 * 3600


def test_a_batch_failed_at_a_quota_limit_rejects_every_unserved_item() -> None:
    server = _Server()
    body = {
        "name": OP,
        "metadata": {"state": "BATCH_STATE_FAILED"},
        "done": True,
        "error": {"code": 8, "message": "Enqueued token quota exceeded for this model."},
    }
    server.add("GET", f"/v1beta/{OP}", httpx.Response(200, json=body))
    job = BatchJob(provider="gemini", job_id=OP, custom_ids=("a", "b"))
    out = dict(server.backend().results(job))
    for key in ("a", "b"):
        item = out[key]
        assert isinstance(item, BatchItemError) and item.kind == "batch_rejected"
        assert not item.retryable and "RESOURCE_EXHAUSTED" in item.message


def test_a_batch_failed_otherwise_keeps_retryable_errored_items() -> None:
    server = _Server()
    body = {
        "name": OP,
        "metadata": {"state": "BATCH_STATE_FAILED"},
        "done": True,
        "error": {"code": 13, "message": "internal error"},
    }
    server.add("GET", f"/v1beta/{OP}", httpx.Response(200, json=body))
    out = dict(server.backend().results(BatchJob(provider="gemini", job_id=OP, custom_ids=("a",))))
    assert isinstance(out["a"], BatchItemError) and out["a"].kind == "errored"
    assert out["a"].retryable


def test_transport_errors_retry_and_then_raise() -> None:
    attempts: list[int] = []

    def boom(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.ConnectError("refused", request=request)

    backend = _backend(transport=httpx.MockTransport(boom), sleep=lambda _s: None)
    with pytest.raises(GeminiBatchError, match="ConnectError"):
        backend.poll(BatchJob(provider="gemini", job_id=OP, custom_ids=("a",)))
    assert len(attempts) == 3


def test_client_errors_are_not_retried() -> None:
    server = _Server()
    server.add("POST", CREATE, httpx.Response(400, json={"error": {"message": "bad"}}))
    sleeps: list[float] = []
    with pytest.raises(GeminiBatchError, match="HTTP 400"):
        server.backend(sleeps).submit([_req("a")])
    assert sleeps == [] and len(server.calls) == 1


def test_api_base_override_and_env_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _op("BATCH_STATE_RUNNING")

    backend = GeminiBatchBackend(
        BackendConfig(model=MODEL, api_base="https://proxy.example/v1beta"),
        transport=httpx.MockTransport(handler),
    )
    backend.poll(BatchJob(provider="gemini", job_id=OP, custom_ids=("a",)))
    assert str(seen[0].url) == f"https://proxy.example/v1beta/{OP}"
    assert seen[0].headers["x-goog-api-key"] == "google-key"


def test_missing_key_is_a_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiBatchBackend(BackendConfig(model=MODEL))


def test_plan_batches_uses_the_backend_limits() -> None:
    backend = _backend()
    batches = plan_batches([_req(f"r{i}") for i in range(5)], backend)
    assert [len(b) for b in batches] == [5]


def test_file_mode_starts_just_past_the_inline_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A batch whose inline body is exactly the cap goes inline; one byte
    smaller a cap and the same batch goes through the File API."""
    sizes: list[int] = []
    real_size = gemini_mod._inline_size

    def recording(inline: dict[str, Any], items: Any) -> int:
        sizes.append(real_size(inline, items))
        return sizes[-1]

    monkeypatch.setattr(gemini_mod, "_inline_size", recording)
    probe = _Server()
    probe.add("POST", CREATE, _op("BATCH_STATE_PENDING"))
    probe.backend().submit([_req("a"), _req("b")])
    (size,) = sizes

    for cap, mode in ((size, "inline"), (size - 1, "file")):
        monkeypatch.setattr(gemini_mod, "INLINE_MAX_BYTES", cap)
        server = _Server()
        upload_url = "https://generativelanguage.googleapis.com/upload/session/xyz"
        server.add(
            "POST",
            "/upload/v1beta/files",
            httpx.Response(200, headers={"x-goog-upload-url": upload_url}, json={}),
        )
        server.add(
            "POST", "/upload/session/xyz", httpx.Response(200, json={"file": {"name": "files/in1"}})
        )
        server.add("POST", CREATE, _op("BATCH_STATE_PENDING"))
        job = server.backend().submit([_req("a"), _req("b")])
        assert job.extra["mode"] == mode
        assert len(server.calls) == (1 if mode == "inline" else 3)


# ── (d) live-shaped operations ───────────────────────────────────────────
#
# Small inline payloads holding only the fields the decoder reads, in the
# shapes the live service sends: ``done`` absent until the batch ends, ended
# results under both ``response`` and ``metadata.output``, error items with
# only a numeric google.rpc ``code``, thought-summary parts, implicit cache
# hits, and the two ways a canceled batch ends.

LIVE_MODEL = "gemini/gemini-3.1-flash-lite"


def _live_backend(routes: dict[tuple[str, str], httpx.Response]) -> tuple[Any, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        reply = routes.get((request.method, request.url.path))
        if reply is None:
            return httpx.Response(404, json={"error": {"code": 404, "status": "NOT_FOUND"}})
        return reply

    backend = GeminiBatchBackend(
        BackendConfig(model=LIVE_MODEL, api_key=KEY),
        transport=httpx.MockTransport(handler),
        sleep=lambda _s: None,
    )
    return backend, calls


def _usage(prompt: int, out: int, **extra: Any) -> dict[str, Any]:
    total = prompt + out + int(extra.get("thoughtsTokenCount", 0))
    return {
        "promptTokenCount": prompt,
        "candidatesTokenCount": out,
        "totalTokenCount": total,
        **extra,
    }


def _candidate(*parts: dict[str, Any]) -> dict[str, Any]:
    return {"content": {"role": "model", "parts": list(parts)}, "finishReason": "STOP"}


_PLAIN = {"candidates": [_candidate({"text": "pineapple"})], "usageMetadata": _usage(8, 1)}
_TOOL = {
    "candidates": [
        _candidate({"functionCall": {"name": "record", "args": {"city": "Springfield"}}})
    ],
    "usageMetadata": _usage(60, 9),
}
_PDF_REPLY = {
    "candidates": [_candidate({"text": "Example Co. invoice"})],
    "usageMetadata": _usage(
        2094,
        4,
        promptTokensDetails=[
            {"modality": "TEXT", "tokenCount": 14},
            {"modality": "IMAGE", "tokenCount": 2080},
        ],
    ),
}
_THINK = {
    "candidates": [_candidate({"text": "Adding the numbers.", "thought": True}, {"text": "391"})],
    "usageMetadata": _usage(20, 3, thoughtsTokenCount=133),
}
_INLINE_IDS = ("t-plain", "t-tool", "t-pdf", "t-invalid")
_LIVE_OP = "batches/syntheticop"


def _live_items() -> list[dict[str, Any]]:
    return [
        {"metadata": {"key": "t-plain"}, "response": _PLAIN},
        {"metadata": {"key": "t-tool"}, "response": _TOOL},
        {"metadata": {"key": "t-pdf"}, "response": _PDF_REPLY},
        {"metadata": {"key": "t-invalid"}, "error": {"code": 3, "message": "bad part"}},
    ]


def _ended_op(where: str, results: dict[str, Any], **meta: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": _LIVE_OP,
        "done": True,
        "metadata": {"state": "BATCH_STATE_SUCCEEDED", **meta},
    }
    if where in ("both", "response"):
        body["response"] = results
    if where in ("both", "metadata.output"):
        body["metadata"]["output"] = results
    return body


def _assert_half_price(response: Any) -> None:
    standard = litellm.completion_cost(completion_response=response, model=LIVE_MODEL)
    assert isinstance(standard, float) and standard > 0
    assert response._hidden_params["response_cost"] == pytest.approx(BATCH_RATE * standard)


def test_a_running_operation_has_no_done_field_and_is_not_done() -> None:
    running = {
        "name": _LIVE_OP,
        "metadata": {
            "state": "BATCH_STATE_RUNNING",
            "batchStats": {"requestCount": "4", "pendingRequestCount": "4"},
        },
    }
    backend, _ = _live_backend({("GET", f"/v1beta/{_LIVE_OP}"): httpx.Response(200, json=running)})
    status = backend.poll(BatchJob(provider="gemini", job_id=_LIVE_OP, custom_ids=_INLINE_IDS))
    assert status.state is BatchState.RUNNING
    assert status.processing == 4 and not status.done


@pytest.mark.parametrize("where", ["both", "response", "metadata.output"])
def test_ended_inline_results_decode_from_either_nesting(where: str) -> None:
    results = {"inlinedResponses": {"inlinedResponses": _live_items()}}
    stats = {
        "requestCount": "4",
        "successfulRequestCount": "3",
        "failedRequestCount": "1",
        "pendingRequestCount": "0",
    }
    op = _ended_op(where, results, batchStats=stats)
    backend, _ = _live_backend({("GET", f"/v1beta/{_LIVE_OP}"): httpx.Response(200, json=op)})
    job = BatchJob(provider="gemini", job_id=_LIVE_OP, custom_ids=_INLINE_IDS)
    status = backend.poll(job)
    assert status.state is BatchState.ENDED
    assert (status.succeeded, status.errored, status.processing) == (3, 1, 0)

    out = dict(backend.results(job))
    assert set(out) == set(_INLINE_IDS)  # read once, though "both" carries two copies
    plain = _ok(out["t-plain"])
    assert plain.choices[0].message.content == "pineapple"
    assert plain.choices[0].finish_reason == "stop"
    assert (plain.usage.prompt_tokens, plain.usage.completion_tokens) == (8, 1)
    tool = _ok(out["t-tool"])
    assert tool.choices[0].finish_reason == "tool_calls"
    (call,) = tool.choices[0].message.tool_calls
    assert call.function.name == "record"
    assert json.loads(call.function.arguments) == {"city": "Springfield"}
    pdf = _ok(out["t-pdf"])
    assert pdf.usage.prompt_tokens == 2094  # a PDF bills as IMAGE-modality prompt tokens
    assert pdf.usage.prompt_tokens_details.image_tokens == 2080
    for cid in ("t-plain", "t-tool", "t-pdf"):
        _assert_half_price(out[cid])
    # An error item with only a numeric google.rpc code (3 = INVALID_ARGUMENT).
    bad = out["t-invalid"]
    assert isinstance(bad, BatchItemError) and bad.kind == "invalid" and not bad.retryable
    assert bad.message.startswith("INVALID_ARGUMENT")


def test_results_reuse_the_operation_the_final_poll_fetched() -> None:
    op = _ended_op("both", {"inlinedResponses": {"inlinedResponses": _live_items()}})
    backend, calls = _live_backend({("GET", f"/v1beta/{_LIVE_OP}"): httpx.Response(200, json=op)})
    job = BatchJob(provider="gemini", job_id=_LIVE_OP, custom_ids=_INLINE_IDS)
    assert backend.poll(job).done
    list(backend.results(job))
    assert len(calls) == 1  # one download, not two
    list(backend.results(job))  # no poll in between (a resumed process): fetch it
    assert len(calls) == 2


def test_file_mode_results_stream_and_price_thinking_tokens() -> None:
    op = _ended_op("both", {"responsesFile": "files/batch-syntheticop"})
    rows = [
        {"key": "t-think", "response": _THINK},
        {"key": "t-plain", "response": _PLAIN},
        {"key": "t-invalid", "error": {"code": 3, "message": "bad part"}},
    ]
    jsonl = "".join(json.dumps(row) + "\n" for row in rows)
    backend, calls = _live_backend(
        {
            ("GET", f"/v1beta/{_LIVE_OP}"): httpx.Response(200, json=op),
            (
                "GET",
                "/download/v1beta/files/batch-syntheticop:download",
            ): httpx.Response(200, text=jsonl),
        }
    )
    ids = ("t-plain", "t-think", "t-invalid")
    out = dict(backend.results(BatchJob(provider="gemini", job_id=_LIVE_OP, custom_ids=ids)))
    assert set(out) == set(ids)
    think = _ok(out["t-think"])
    # Thought-summary parts are not content; thoughtsTokenCount bills as output.
    assert think.choices[0].message.content == "391"
    assert think.usage.completion_tokens_details.reasoning_tokens == 133
    assert think.usage.completion_tokens == 136
    _assert_half_price(think)
    _assert_half_price(out["t-plain"])
    assert isinstance(out["t-invalid"], BatchItemError) and out["t-invalid"].kind == "invalid"
    assert calls[-1].url.params["alt"] == "media"


def test_an_implicit_cache_hit_is_priced_at_the_batch_cache_rate() -> None:
    miss = {"candidates": [_candidate({"text": "a"})], "usageMetadata": _usage(4100, 1)}
    hit = {
        "candidates": [_candidate({"text": "b"})],
        "usageMetadata": _usage(4100, 1, cachedContentTokenCount=4080),
    }
    items = [
        {"metadata": {"key": "k-0"}, "response": miss},
        {"metadata": {"key": "k-1"}, "response": hit},
    ]
    op = _ended_op("both", {"inlinedResponses": {"inlinedResponses": items}})
    backend, _ = _live_backend({("GET", f"/v1beta/{_LIVE_OP}"): httpx.Response(200, json=op)})
    job = BatchJob(provider="gemini", job_id=_LIVE_OP, custom_ids=("k-0", "k-1"))
    out = dict(backend.results(job))
    cached, uncached = _ok(out["k-1"]), _ok(out["k-0"])
    assert cached.usage.prompt_tokens_details.cached_tokens == 4080
    assert not uncached.usage.prompt_tokens_details.cached_tokens
    _assert_half_price(cached)
    _assert_half_price(uncached)
    # The cached prefix is discounted on top of the batch rate.
    assert cached._hidden_params["response_cost"] < uncached._hidden_params["response_cost"]


def test_a_batch_canceled_before_it_started_marks_items_canceled() -> None:
    # It ends CANCELLED, done, with an INTERNAL (13) error attached.
    op = {
        "name": _LIVE_OP,
        "done": True,
        "metadata": {
            "state": "BATCH_STATE_CANCELLED",
            "batchStats": {"requestCount": "1", "pendingRequestCount": "1"},
        },
        "error": {"code": 13, "message": "Batch was cancelled."},
    }
    backend, _ = _live_backend({("GET", f"/v1beta/{_LIVE_OP}"): httpx.Response(200, json=op)})
    job = BatchJob(provider="gemini", job_id=_LIVE_OP, custom_ids=("c-1",))
    status = backend.poll(job)
    assert status.state is BatchState.CANCELED
    assert (status.canceled, status.processing) == (1, 0)
    (item,) = [outcome for _cid, outcome in backend.results(job)]
    assert isinstance(item, BatchItemError) and item.kind == "canceled" and not item.retryable


def test_a_batch_canceled_after_it_started_ends_succeeded_with_canceled_items() -> None:
    # Each unserved item comes back as an error with only code 1 (CANCELLED).
    items = [{"metadata": {"key": "smoke"}, "error": {"code": 1}}]
    stats = {"requestCount": "1", "failedRequestCount": "1", "pendingRequestCount": "0"}
    op = _ended_op("both", {"inlinedResponses": {"inlinedResponses": items}}, batchStats=stats)
    backend, _ = _live_backend({("GET", f"/v1beta/{_LIVE_OP}"): httpx.Response(200, json=op)})
    job = BatchJob(provider="gemini", job_id=_LIVE_OP, custom_ids=("smoke",))
    status = backend.poll(job)
    assert status.state is BatchState.ENDED and status.errored == 1
    (item,) = [outcome for _cid, outcome in backend.results(job)]
    assert isinstance(item, BatchItemError) and item.kind == "canceled" and not item.retryable


# ── (e) registry ─────────────────────────────────────────────────────────


def test_gemini_is_registered_and_resolves() -> None:
    assert "gemini" in registered_providers()
    assert provider_of(MODEL) == "gemini"
    backend = resolve_backend(MODEL, api_key=KEY)
    assert isinstance(backend, GeminiBatchBackend)
    assert backend.provider == "gemini"


# ── live smoke (opt-in) ──────────────────────────────────────────────────


@pytest.fixture
def live_backend() -> Iterator[GeminiBatchBackend]:
    if not (os.environ.get("DGML_LIVE_BATCH_TESTS") and os.environ.get("GEMINI_LIVE_API_KEY")):
        pytest.skip("live Gemini batch smoke needs DGML_LIVE_BATCH_TESTS=1 and GEMINI_LIVE_API_KEY")
    yield GeminiBatchBackend(BackendConfig(model=MODEL, api_key=os.environ["GEMINI_LIVE_API_KEY"]))


@pytest.mark.allow_network
def test_live_submit_poll_cancel(live_backend: GeminiBatchBackend) -> None:
    job = live_backend.submit([_req("smoke", "Reply with the single word OK.")])
    assert live_backend.poll(job).state in {
        BatchState.PENDING,
        BatchState.RUNNING,
        BatchState.ENDED,
    }
    live_backend.cancel(job)
    # Leave nothing behind: wait for the cancel to land, then delete the batch.
    deadline = time.monotonic() + 120
    while not live_backend.poll(job).done and time.monotonic() < deadline:
        time.sleep(3)
    live_backend.cleanup(job)
