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

"""Anthropic Message Batches backend: wire parity, decode parity, transport.

Wire parity is the contract that matters: the ``params`` a batch carries must
be the body the synchronous ``litellm.completion`` route would have POSTed.
The two are captured through *different* seams — the sync body by patching
``HTTPHandler.post`` at class level, the batch body through the backend's own
capturing handler subclass — so a litellm upgrade that changes the sync
encoding without the batch encoding following it fails here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import httpx
import litellm
import pytest
from dgml_core import llm
from dgml_core.batch import (
    AnthropicBatchBackend,
    AnthropicBatchError,
    BatchItemError,
    BatchJob,
    BatchRequest,
    BatchState,
    plan_batches,
    provider_of,
    resolve_backend,
)
from dgml_core.batch.anthropic import BATCH_PRICE_MULTIPLIER, _api_root
from dgml_core.batch.types import BatchRejected, BatchSubmitUncertain, BatchThrottled
from dgml_core.usage import extract_cost_and_tokens
from litellm.llms.custom_httpx.http_handler import HTTPHandler

from .conftest import FakeLLMResponse

MODEL = "anthropic/claude-haiku-4-5"
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

TEXT_MESSAGE: dict[str, Any] = {
    "id": "msg_text",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "hello from the batch"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 120,
        "output_tokens": 30,
        "cache_read_input_tokens": 40,
        "cache_creation_input_tokens": 8,
    },
}
TOOL_MESSAGE: dict[str, Any] = {
    "id": "msg_tool",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [
        {"type": "text", "text": "submitting"},
        {"type": "tool_use", "id": "toolu_1", "name": "submit", "input": {"value": "x"}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 50, "output_tokens": 12},
}
LENGTH_MESSAGE: dict[str, Any] = {
    **TEXT_MESSAGE,
    "id": "msg_len",
    "content": [{"type": "text", "text": "cut off"}],
    "stop_reason": "max_tokens",
}


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def sync_capture(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Run ``litellm.completion`` on the sync route and return the body it
    was about to POST (class-level patch on litellm's HTTP handler)."""
    captured: dict[str, Any] = {}

    def fake_post(
        self: Any,
        url: str,
        data: Any = None,
        json_body: Any = None,
        params: Any = None,
        headers: Any = None,
        stream: bool = False,
        timeout: Any = None,
        files: Any = None,
        content: Any = None,
        logging_obj: Any = None,
        **kwargs: Any,
    ) -> httpx.Response:
        body = data if data is not None else kwargs.get("json", json_body)
        captured["body"] = json.loads(body) if isinstance(body, str | bytes) else dict(body)
        captured["headers"] = dict(headers or {})
        return httpx.Response(200, json=TEXT_MESSAGE, request=httpx.Request("POST", url))

    monkeypatch.setattr(HTTPHandler, "post", fake_post)

    def run(kwargs: dict[str, Any]) -> dict[str, Any]:
        captured.clear()
        with llm._quiet_stdout():
            litellm.completion(**kwargs)
        body = dict(captured["body"])
        body.pop("stream", None)
        return body

    return run


def _first_step(steps: Any) -> dict[str, Any]:
    """The kwargs of a ``steps_*`` generator's first request."""
    step = next(steps)
    steps.close()
    return dict(step)


def _wrapper_shapes() -> list[tuple[str, dict[str, Any]]]:
    """The requests DGML's wrappers send to Anthropic, built by the wrappers
    themselves (the same ``steps_*`` generators the sync path drives)."""
    cfg = llm.LLMConfig(model=MODEL, temperature=0.0, max_tokens=4000)
    text = [{"type": "text", "text": "Read the page and answer."}]
    plain = _first_step(llm.steps_call(cfg, system_prompt="SYS", user_content=text))
    cached = _first_step(llm.steps_call(cfg, system_prompt="SYS", user_content=text, cache=True))
    split_system = _first_step(
        llm.steps_call(cfg, system_prompt=("STABLE PREFIX", "VARIABLE"), user_content=text)
    )
    # A continuation round: the second request carries the partial answer.
    continued = llm.steps_continued(cfg, system_prompt="SYS", user_content=text)
    next(continued)
    prefill = dict(continued.send(FakeLLMResponse("partial answer", finish_reason="length")))
    continued.close()
    forced_tool = _first_step(
        llm.steps_with_tools(
            cfg,
            messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": text}],
            tools=_TOOLS,
            tool_choice={"type": "function", "function": {"name": "submit"}},
        )
    )
    return [
        ("call_plain", plain),
        ("call_cached", cached),
        ("call_split_system", split_system),
        ("continued_prefill", prefill),
        ("tools_forced", forced_tool),
    ]


def _extra_shapes() -> list[tuple[str, dict[str, Any]]]:
    """Request shapes the wrappers build with documents, images or effort."""
    cfg = llm.LLMConfig(model=MODEL, temperature=0.0, max_tokens=4000)
    pdf = llm._build_completion_kwargs(
        cfg,
        messages=[
            llm._build_system_message("SYS", cache=True, is_anthropic=True),
            {
                "role": "user",
                "content": llm._mark_document_cacheable(
                    llm.build_user_content(instruction_text="read", pdf_bytes=_PDF)
                ),
            },
        ],
    )
    image = llm._build_completion_kwargs(
        cfg,
        messages=[
            {"role": "system", "content": "SYS"},
            {
                "role": "user",
                "content": llm.build_user_content(instruction_text="look", images=[_PNG]),
            },
        ],
    )
    auto_tools_effort = llm._build_completion_kwargs(
        llm.LLMConfig(
            model=MODEL, max_tokens=None, max_completion_tokens=8000, reasoning_effort="high"
        ),
        messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "go"}],
        tools=_TOOLS,
    )
    return [("pdf_cached", pdf), ("image", image), ("tools_auto_effort", auto_tools_effort)]


SHAPES = _wrapper_shapes() + _extra_shapes()


@pytest.mark.parametrize(("name", "kwargs"), SHAPES, ids=[n for n, _ in SHAPES])
def test_wire_parity_with_sync_route(
    name: str, kwargs: dict[str, Any], sync_capture: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    backend = AnthropicBatchBackend(api_key="test-key")
    encoded = backend.encode(BatchRequest(name, kwargs))
    assert encoded["custom_id"] == name
    assert encoded["params"] == sync_capture(kwargs)
    assert "stream" not in encoded["params"]
    assert encoded["params"]["model"] == "claude-haiku-4-5" or encoded["params"][
        "model"
    ].startswith("claude")


def test_encode_is_a_pure_function_of_the_request() -> None:
    backend = AnthropicBatchBackend(api_key="test-key")
    kwargs = _wrapper_shapes()[0][1]
    a = backend.encode(BatchRequest("a", kwargs))
    b = AnthropicBatchBackend(api_key="test-key").encode(BatchRequest("a", kwargs))
    assert a == b
    # And planning sizes it without credentials or a network.
    batches = plan_batches([BatchRequest("a", kwargs)], AnthropicBatchBackend())
    assert [[r.custom_id for r in b] for b in batches] == [["a"]]


# --- decode parity -----------------------------------------------------------


def _sync_response(monkeypatch: pytest.MonkeyPatch, message: dict[str, Any]) -> Any:
    def fake_post(self: Any, url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(200, json=message, request=httpx.Request("POST", url))

    monkeypatch.setattr(HTTPHandler, "post", fake_post)
    with llm._quiet_stdout():
        return litellm.completion(
            model=MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=64
        )


@pytest.mark.parametrize(
    "message", [TEXT_MESSAGE, TOOL_MESSAGE, LENGTH_MESSAGE], ids=["text", "tool_use", "length"]
)
def test_decode_parity_with_sync_route(
    monkeypatch: pytest.MonkeyPatch, message: dict[str, Any]
) -> None:
    sync = _sync_response(monkeypatch, message)
    batch = AnthropicBatchBackend(api_key="test-key").decode_message(message)

    assert batch.choices[0].message.content == sync.choices[0].message.content
    assert batch.choices[0].finish_reason == sync.choices[0].finish_reason
    sync_calls = sync.choices[0].message.tool_calls or []
    batch_calls = batch.choices[0].message.tool_calls or []
    assert [(c.function.name, c.function.arguments) for c in batch_calls] == [
        (c.function.name, c.function.arguments) for c in sync_calls
    ]
    # Item access, as ``call``/``call_with_refinement`` read it.
    assert batch["choices"][0]["message"]["content"] == sync["choices"][0]["message"]["content"]

    got, want = extract_cost_and_tokens(batch), extract_cost_and_tokens(sync)
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
    ):
        assert got[key] == want[key], key
    standard = litellm.completion_cost(
        completion_response=sync, model="claude-haiku-4-5", custom_llm_provider="anthropic"
    )
    assert standard > 0
    assert got["cost_usd"] == pytest.approx(standard * BATCH_PRICE_MULTIPLIER)
    assert batch._hidden_params["tier"] == "batch"


def test_decode_unknown_model_leaves_cost_unknown() -> None:
    backend = AnthropicBatchBackend(api_key="test-key")
    response = backend.decode_message({**TEXT_MESSAGE, "model": "claude-not-a-real-model"})
    assert extract_cost_and_tokens(response)["cost_usd"] is None
    assert response.choices[0].message.content == "hello from the batch"


# --- transport ---------------------------------------------------------------


def _results_jsonl() -> str:
    lines = [
        {"custom_id": "text", "result": {"type": "succeeded", "message": TEXT_MESSAGE}},
        {"custom_id": "tool", "result": {"type": "succeeded", "message": TOOL_MESSAGE}},
        {
            "custom_id": "bad",
            "result": {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "max_tokens too big"},
                },
            },
        },
        {
            "custom_id": "flaky",
            "result": {
                "type": "errored",
                "error": {"type": "error", "error": {"type": "api_error", "message": "boom"}},
            },
        },
        {"custom_id": "late", "result": {"type": "expired"}},
        {"custom_id": "gone", "result": {"type": "canceled"}},
    ]
    return "\n".join(json.dumps(line) for line in lines) + "\n"


class _Server:
    """A scripted Anthropic batches endpoint behind ``httpx.MockTransport``."""

    def __init__(
        self,
        *,
        fail_first_post_with: int | None = None,
        fail_message: str = "overloaded",
        fail_get_with: list[int] | None = None,
    ) -> None:
        self.calls: list[httpx.Request] = []
        self.fail_first_post_with = fail_first_post_with
        self.fail_message = fail_message
        self.fail_get_with = list(fail_get_with or [])
        self.polls = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if request.method == "GET" and self.fail_get_with:
            return httpx.Response(self.fail_get_with.pop(0), json={"error": {"message": "busy"}})
        if request.method == "DELETE" and path == "/v1/messages/batches/msgbatch_1":
            return httpx.Response(200, json={"id": "msgbatch_1", "type": "message_batch_deleted"})
        if request.method == "POST" and path == "/v1/messages/batches":
            if self.fail_first_post_with is not None:
                status, self.fail_first_post_with = self.fail_first_post_with, None
                return httpx.Response(status, json={"error": {"message": self.fail_message}})
            return httpx.Response(
                200,
                json={
                    "id": "msgbatch_1",
                    "type": "message_batch",
                    "processing_status": "in_progress",
                    "request_counts": {
                        "processing": 6,
                        "succeeded": 0,
                        "errored": 0,
                        "canceled": 0,
                        "expired": 0,
                    },
                    "results_url": None,
                },
            )
        if request.method == "GET" and path == "/v1/messages/batches/msgbatch_1":
            self.polls += 1
            ended = self.polls >= 2
            return httpx.Response(
                200,
                json={
                    "id": "msgbatch_1",
                    "processing_status": "ended" if ended else "in_progress",
                    "request_counts": {
                        "processing": 0 if ended else 6,
                        "succeeded": 2 if ended else 0,
                        "errored": 2 if ended else 0,
                        "canceled": 1 if ended else 0,
                        "expired": 1 if ended else 0,
                    },
                    "results_url": (
                        "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"
                        if ended
                        else None
                    ),
                },
            )
        if request.method == "GET" and path == "/v1/messages/batches/msgbatch_1/results":
            return httpx.Response(200, text=_results_jsonl())
        if request.method == "POST" and path == "/v1/messages/batches/msgbatch_1/cancel":
            return httpx.Response(200, json={"id": "msgbatch_1", "processing_status": "canceling"})
        return httpx.Response(404, json={"error": "no route"})


def _backend(server: _Server, **kwargs: Any) -> AnthropicBatchBackend:
    return AnthropicBatchBackend(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(server.handle)),
        retry_delay=0.0,
        sleep=lambda _s: None,
        **kwargs,
    )


def _requests() -> list[BatchRequest]:
    kwargs = _wrapper_shapes()[0][1]
    return [BatchRequest(cid, kwargs) for cid in ("text", "tool", "bad", "flaky", "late", "gone")]


def test_submit_poll_results_cancel_round_trip() -> None:
    server = _Server()
    backend = _backend(server)
    job = backend.submit(_requests())

    create = server.calls[0]
    assert create.headers["x-api-key"] == "test-key"
    assert create.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(create.content)
    assert [r["custom_id"] for r in body["requests"]] == list(job.custom_ids)
    assert body["requests"][0]["params"] == backend.encode(_requests()[0])["params"]
    assert job.provider == "anthropic" and job.job_id == "msgbatch_1"
    assert BatchJob.from_json(job.to_json()).job_id == job.job_id

    first = backend.poll(job)
    assert first.state is BatchState.RUNNING and first.processing == 6 and not first.done
    second = backend.poll(job)
    assert second.state is BatchState.ENDED and second.done
    assert (second.succeeded, second.errored, second.canceled, second.expired) == (2, 2, 1, 1)
    assert job.extra["results_url"].endswith("/msgbatch_1/results")

    outcomes: dict[str, Any] = dict(backend.results(job))
    assert set(outcomes) == set(job.custom_ids)
    assert outcomes["text"].choices[0].message.content == "hello from the batch"
    assert outcomes["tool"].choices[0].message.tool_calls[0].function.name == "submit"
    assert extract_cost_and_tokens(outcomes["text"])["cost_usd"] == pytest.approx(
        litellm.completion_cost(
            completion_response=outcomes["text"],
            model="claude-haiku-4-5",
            custom_llm_provider="anthropic",
        )
        * BATCH_PRICE_MULTIPLIER
    )
    bad = outcomes["bad"]
    assert isinstance(bad, BatchItemError) and bad.kind == "invalid" and not bad.retryable
    assert "max_tokens too big" in bad.message
    flaky = outcomes["flaky"]
    assert isinstance(flaky, BatchItemError) and flaky.kind == "errored" and flaky.retryable
    late = outcomes["late"]
    assert isinstance(late, BatchItemError) and late.kind == "expired" and late.retryable
    gone = outcomes["gone"]
    assert isinstance(gone, BatchItemError) and gone.kind == "canceled" and not gone.retryable

    backend.cancel(job)
    assert server.calls[-1].url.path.endswith("/msgbatch_1/cancel")


def test_results_polls_first_when_no_results_url_yet() -> None:
    server = _Server()
    backend = _backend(server)
    job = backend.submit(_requests())
    # First poll (inside results) still says in_progress → no results yet.
    with pytest.raises(AnthropicBatchError, match="no results yet"):
        list(backend.results(job))
    # A later call finds the second poll ended and streams the file.
    assert len(dict(backend.results(job))) == 6


def test_submit_retries_a_rate_limit_refusal_then_succeeds() -> None:
    server = _Server(fail_first_post_with=429)
    backend = _backend(server)
    job = backend.submit(_requests()[:1])
    assert job.job_id == "msgbatch_1"
    posts = [c for c in server.calls if c.method == "POST"]
    assert len(posts) == 2


@pytest.mark.parametrize("status", [500, 503, 529, 408])
def test_submit_never_resends_a_create_that_may_have_landed(status: int) -> None:
    """No idempotency key on the create: a resend after a lost response bills twice."""
    server = _Server(fail_first_post_with=status)
    backend = _backend(server)
    with pytest.raises(BatchSubmitUncertain, match=f"HTTP {status}"):
        backend.submit(_requests()[:1])
    assert len([c for c in server.calls if c.method == "POST"]) == 1


def test_submit_after_a_read_timeout_is_uncertain() -> None:
    posts: list[httpx.Request] = []

    def slow(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        raise httpx.ReadTimeout("no answer", request=request)

    backend = AnthropicBatchBackend(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(slow)),
        retry_delay=0.0,
        sleep=lambda _s: None,
    )
    with pytest.raises(BatchSubmitUncertain, match="ReadTimeout"):
        backend.submit(_requests()[:1])
    assert len(posts) == 1


def test_submit_retries_when_the_connection_was_never_made() -> None:
    attempts: list[int] = []

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"id": "msgbatch_1", "processing_status": "in_progress"})

    backend = AnthropicBatchBackend(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(flaky)),
        retry_delay=0.0,
        sleep=lambda _s: None,
    )
    assert backend.submit(_requests()[:1]).job_id == "msgbatch_1"
    assert len(attempts) == 2


def test_submit_still_rate_limited_after_every_attempt_is_throttled() -> None:
    def always_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"type": "rate_limit_error"}})

    backend = AnthropicBatchBackend(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(always_429)),
        retry_delay=0.0,
        sleep=lambda _s: None,
    )
    with pytest.raises(BatchThrottled, match="after 3 attempts"):
        backend.submit(_requests()[:1])


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, text="<html>gateway</html>"),
        httpx.Response(200, json={"type": "message_batch"}),  # no id
        httpx.Response(200, json=["not", "an", "object"]),
    ],
)
def test_an_accepted_create_whose_answer_cannot_be_read_is_uncertain(
    reply: httpx.Response,
) -> None:
    """A 2xx means the batch exists: failing to read it must never look like a
    refusal (which would rerun every request at full price) (F5)."""
    posts: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return reply

    backend = AnthropicBatchBackend(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        retry_delay=0.0,
        sleep=lambda _s: None,
    )
    with pytest.raises(BatchSubmitUncertain, match="may have been created"):
        backend.submit(_requests()[:1])
    assert len(posts) == 1


def test_an_unreadable_accepted_create_fails_the_wave_without_a_sync_rerun() -> None:
    from dgml_core.batch.executor import BatchExecutor
    from dgml_core.errors import BatchExecutionFailed

    backend = AnthropicBatchBackend(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, text="<html/>"))
        ),
        retry_delay=0.0,
        sleep=lambda _s: None,
    )
    sync: list[Any] = []
    executor = BatchExecutor(backend, sync_execute=sync.append, sleep=lambda _s: None)
    kwargs = _requests()[0].kwargs
    with pytest.raises(BatchExecutionFailed, match="may exist"):
        executor.run_wave({"a": kwargs})
    assert sync == []


def test_submit_refused_at_a_size_limit_is_batch_rejected() -> None:
    server = _Server(fail_first_post_with=413, fail_message="request_too_large")
    with pytest.raises(BatchRejected, match="HTTP 413"):
        _backend(server).submit(_requests()[:1])
    assert len([c for c in server.calls if c.method == "POST"]) == 1


def test_submit_auth_failure_is_a_backend_error() -> None:
    server = _Server(fail_first_post_with=401, fail_message="invalid x-api-key")
    with pytest.raises(AnthropicBatchError, match="HTTP 401") as exc_info:
        _backend(server).submit(_requests()[:1])
    assert exc_info.value.status_code == 401


def test_poll_keeps_retrying_transient_failures() -> None:
    server = _Server(fail_get_with=[529, 503])
    backend = _backend(server)
    job = backend.submit(_requests()[:1])
    assert backend.poll(job).state is BatchState.RUNNING
    assert len([c for c in server.calls if c.method == "GET"]) == 3


def test_cleanup_deletes_the_batch_and_is_best_effort() -> None:
    server = _Server()
    backend = _backend(server)
    job = backend.submit(_requests()[:1])
    backend.cleanup(job)
    assert (server.calls[-1].method, server.calls[-1].url.path) == (
        "DELETE",
        "/v1/messages/batches/msgbatch_1",
    )
    assert server.calls[-1].headers["x-api-key"] == "test-key"
    # An unknown batch 404s: already gone counts as cleaned.
    backend.cleanup(BatchJob(provider="anthropic", job_id="msgbatch_gone", custom_ids=()))


def test_a_cleanup_failure_raises_and_the_executor_logs_it() -> None:
    """The endpoint refuses to delete a batch still processing (or canceling):
    the backend reports it rather than swallowing it, and the best-effort
    wrapper every caller goes through logs it without failing anything."""
    from dgml_core.batch.executor import cleanup_batch

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": {"message": "batch is still canceling"}})

    backend = AnthropicBatchBackend(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        retry_delay=0.0,
        sleep=lambda _s: None,
    )
    job = BatchJob(provider="anthropic", job_id="msgbatch_busy", custom_ids=())
    with pytest.raises(AnthropicBatchError, match="409"):
        backend.cleanup(job)
    lines: list[str] = []
    cleanup_batch(backend, job, lines.append)  # never raises
    assert len(lines) == 1
    assert "cleanup of msgbatch_busy failed" in lines[0] and "still canceling" in lines[0]


def test_release_drops_cached_encodings_and_max_wait_is_24h() -> None:
    backend = _backend(_Server())
    requests = _requests()[:2]
    for r in requests:
        backend.encode(r)
    assert len(backend._cache) == 2
    backend.release([requests[0].custom_id, "never-encoded"])
    assert len(backend._cache) == 1
    backend.release([requests[1].custom_id])
    assert len(backend._cache) == 0
    assert backend.max_wait_s == 24 * 3600


def test_submit_does_not_retry_client_errors() -> None:
    server = _Server(fail_first_post_with=400)
    backend = _backend(server)
    with pytest.raises(AnthropicBatchError, match="HTTP 400") as exc_info:
        backend.submit(_requests()[:1])
    assert exc_info.value.status_code == 400
    assert len([c for c in server.calls if c.method == "POST"]) == 1


def test_submit_unions_request_betas_onto_the_create_call(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _Server()
    backend = _backend(server)
    requests = _requests()[:2]
    # Simulate litellm having sent per-request beta headers on the sync route.
    from dgml_core.batch.anthropic import _Encoded

    for req, beta in zip(requests, ("beta-b", "beta-a,beta-b"), strict=True):
        backend._cache.put(req, _Encoded(params={"model": "m"}, betas=tuple(beta.split(","))))
    job = backend.submit(requests)
    assert server.calls[0].headers["anthropic-beta"] == "beta-a,beta-b"
    assert job.extra["betas"] == ["beta-a", "beta-b"]
    assert len(backend._cache) == 0


def test_submit_without_credentials_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    server = _Server()
    backend = AnthropicBatchBackend(
        http_client=httpx.Client(transport=httpx.MockTransport(server.handle)),
        retry_delay=0.0,
        sleep=lambda _s: None,
    )
    # Encoding (planning) still works without a key…
    backend.encode(_requests()[0])
    # …but submitting does not.
    with pytest.raises(AnthropicBatchError, match="no Anthropic API key"):
        backend.submit(_requests()[:1])


def test_empty_submit_is_an_error() -> None:
    with pytest.raises(AnthropicBatchError, match="empty"):
        _backend(_Server()).submit([])


@pytest.mark.parametrize(
    ("api_base", "root"),
    [
        (None, "https://api.anthropic.com"),
        ("https://proxy.example/v1/messages", "https://proxy.example"),
        ("https://proxy.example/v1/", "https://proxy.example"),
        ("https://proxy.example", "https://proxy.example"),
    ],
)
def test_api_root_accepts_every_api_base_form(api_base: str | None, root: str) -> None:
    assert _api_root(api_base) == root


# --- registry ----------------------------------------------------------------


def test_registered_for_first_party_anthropic_models() -> None:
    backend = resolve_backend(MODEL, api_key="k")
    assert isinstance(backend, AnthropicBatchBackend)
    assert backend.provider == "anthropic"
    assert provider_of(MODEL) == "anthropic"
    assert backend.max_requests == 100_000 and backend.max_bytes == 256 * 1024 * 1024


# --- live smoke (opt-in) -----------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or os.environ["ANTHROPIC_API_KEY"] == "test-key",
    reason="live Anthropic smoke test needs a real ANTHROPIC_API_KEY",
)
@pytest.mark.allow_network
def test_live_smoke_round_trip() -> None:  # pragma: no cover - network
    import time

    backend = AnthropicBatchBackend()
    kwargs = llm._build_completion_kwargs(
        llm.LLMConfig(model=MODEL, max_tokens=16),
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
    )
    job = backend.submit([BatchRequest("ping", kwargs)])
    deadline = time.monotonic() + 15 * 60
    while not backend.poll(job).done:
        assert time.monotonic() < deadline, "batch did not end within 15 minutes"
        time.sleep(10)
    outcomes = dict(backend.results(job))
    response = outcomes["ping"]
    assert not isinstance(response, BatchItemError), response
    assert "pong" in response.choices[0].message.content.lower()
    assert extract_cost_and_tokens(response)["cost_usd"] is not None


def test_failed_submit_does_not_leak_a_stale_body_to_a_reused_id() -> None:
    """Ids repeat across stages; a cached encoding must never outlive its kwargs."""
    golden = _wrapper_shapes()
    kwargs_a = golden[0][1]
    kwargs_b = next(k for _, k in golden if k["messages"] != kwargs_a["messages"])
    server = _Server(fail_first_post_with=400)
    backend = _backend(server)
    body_a = backend.encode(BatchRequest("d0001_x", kwargs_a))["params"]
    with pytest.raises(AnthropicBatchError):
        backend.submit([BatchRequest("d0001_x", kwargs_a)])
    body_b = AnthropicBatchBackend(api_key="test-key").encode(BatchRequest("d0001_x", kwargs_b))
    assert body_b["params"] != body_a
    backend.encode(BatchRequest("d0001_x", kwargs_b))
    backend.submit([BatchRequest("d0001_x", kwargs_b)])
    sent = json.loads(server.calls[-1].content)["requests"][0]
    assert sent["custom_id"] == "d0001_x"
    assert sent["params"] == body_b["params"]
    assert len(backend._cache) == 0


# ---- BatchNotFound: a 404 from poll / results / cancel ----------------------------


@pytest.mark.parametrize("call", ["poll", "results", "cancel"])
def test_a_batch_the_provider_no_longer_knows_is_batch_not_found(call: str) -> None:
    from dgml_core.batch.types import BatchNotFound

    backend = AnthropicBatchBackend(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(404, json={"error": {"type": "not_found_error"}})
            )
        ),
        retry_delay=0.0,
        sleep=lambda _s: None,
    )
    url = "https://api.anthropic.com/v1/messages/batches/msgbatch_gone/results"
    job = BatchJob(
        provider="anthropic",
        job_id="msgbatch_gone",
        custom_ids=("a",),
        extra={"results_url": url},
    )
    with pytest.raises(BatchNotFound, match="msgbatch_gone"):
        if call == "results":
            list(backend.results(job))
        else:
            getattr(backend, call)(job)


# ---- F6: results are streamed line by line, never buffered whole ---------------


class _Chunks(httpx.SyncByteStream):
    """A response body delivered in chunks, counting how many were read."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.read = 0
        self.closed = False

    def __iter__(self) -> Any:
        for chunk in self.chunks:
            self.read += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


def test_results_are_streamed_not_read_whole() -> None:
    error = {"type": "errored", "error": {"type": "error", "error": {"type": "api_error"}}}
    body = [
        (json.dumps({"custom_id": f"r{i}", "result": error}) + "\n").encode() for i in range(50)
    ]
    stream = _Chunks(body)
    url = "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"

    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == url
        return httpx.Response(200, stream=stream)

    backend = AnthropicBatchBackend(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        retry_delay=0.0,
        sleep=lambda _s: None,
    )
    job = BatchJob(
        provider="anthropic",
        job_id="msgbatch_1",
        custom_ids=tuple(f"r{i}" for i in range(50)),
        extra={"results_url": url},
    )
    results = backend.results(job)
    first_id, _outcome = next(results)
    assert first_id == "r0"
    assert stream.read < len(body)  # the first result came before the body was read
    rest = list(results)
    assert len(rest) == 49 and stream.read == len(body) and stream.closed
