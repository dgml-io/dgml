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

"""OpenAI batch backend: wire parity, decode parity, lifecycle, registry.

Wire parity is the load-bearing test. Each scenario builds kwargs with the
real ``llm._build_completion_kwargs``, sends them through the SYNC
``litellm.completion`` path with an OpenAI client whose HTTP transport is a
``MockTransport`` (so the exact request body the sync path puts on the wire is
captured), and asserts the batch line's ``body`` is identical.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import litellm
import openai
import pytest
from dgml_core import llm, prompts
from dgml_core.batch import (
    BatchItemError,
    BatchJob,
    BatchRequest,
    BatchState,
    registered_providers,
    resolve_backend,
)
from dgml_core.batch import openai as ob
from dgml_core.batch.executor import BatchExecutor
from dgml_core.batch.registry import BackendConfig
from dgml_core.batch.types import BatchRejected, BatchSubmitUncertain, BatchThrottled
from dgml_core.usage import extract_cost_and_tokens
from litellm.types.llms.openai import HttpxBinaryResponseContent, OpenAIFileObject
from litellm.types.utils import LiteLLMBatch

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
_PDF = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
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

_CHAT_BODY: dict[str, Any] = {
    "id": "chatcmpl-x",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-2024-08-06",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}


def _sync_wire_body(kwargs: dict[str, Any], response: dict[str, Any] | None = None) -> Any:
    """Run ``kwargs`` through sync ``litellm.completion``; return (wire body, response)."""
    captured: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json=response or _CHAT_BODY)

    client = openai.OpenAI(
        api_key="sk-test", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    result = litellm.completion(**kwargs, client=client)
    assert len(captured) == 1
    return captured[0], result


def _kw(config: llm.LLMConfig, messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return llm._build_completion_kwargs(config, messages=messages, **extra)


def _sys_user(user: Any) -> list[dict[str, Any]]:
    return [{"role": "system", "content": "SYS"}, {"role": "user", "content": user}]


SCENARIOS: dict[str, dict[str, Any]] = {
    "text_system_user": _kw(
        llm.LLMConfig(model="openai/gpt-5.4", temperature=0.0, timeout=30.0, api_key="sk-test"),
        _sys_user([{"type": "text", "text": "U"}]),
    ),
    "image_url_content": _kw(
        llm.LLMConfig(model="openai/gpt-5.4", temperature=0.0, api_key="sk-test"),
        _sys_user(llm.build_user_content(instruction_text="look", images=[_PNG, _PNG])),
    ),
    "pdf_file_block": _kw(
        llm.LLMConfig(model="gpt-4o", temperature=0.0, api_key="sk-test"),
        _sys_user(llm.build_user_content(instruction_text="read", pdf_bytes=_PDF)),
    ),
    "tools_forced_choice": _kw(
        llm.LLMConfig(
            model="gpt-4o",
            api_key="sk-test",
            temperature=0.0,
            max_tokens=None,
            max_completion_tokens=8000,
            reasoning_effort="high",
        ),
        _sys_user("go"),
        tools=_TOOLS,
        tool_choice=_FORCED,
    ),
    "reasoning_effort_o_series": _kw(
        llm.LLMConfig(
            model="openai/o4-mini",
            temperature=0.0,
            max_tokens=None,
            max_completion_tokens=4000,
            reasoning_effort="low",
            api_key="sk-test",
        ),
        _sys_user("go"),
    ),
    "max_completion_tokens_only": _kw(
        llm.LLMConfig(
            model="openai/gpt-5.4", api_key="sk-test", max_tokens=None, max_completion_tokens=12000
        ),
        _sys_user("go"),
    ),
    "assistant_continuation_user_turn": _kw(
        llm.LLMConfig(model="openai/gpt-5.4", temperature=0.0, api_key="sk-test"),
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": [{"type": "text", "text": "U"}]},
            {"role": "assistant", "content": "part-1"},
            {"role": "user", "content": prompts.get(prompts.PromptKey.CONTINUE_TRUNCATED)},
        ],
    ),
    # litellm-only kwargs arriving through LLMConfig.extra must be handled by
    # litellm (headers, retries, logging metadata), never leak into the body;
    # a real OpenAI parameter in ``extra`` (``user``) must reach it.
    "llm_config_extra_litellm_only_keys": _kw(
        llm.LLMConfig(
            model="openai/gpt-5.4",
            temperature=0.0,
            api_key="sk-test",
            extra={
                "num_retries": 2,
                "extra_headers": {"X-Trace": "abc"},
                "metadata": {"run": "r1"},
                "user": "dgml",
            },
        ),
        _sys_user("go"),
    ),
    "cache_markers_are_stripped_like_sync": _kw(
        llm.LLMConfig(
            model="gpt-4o",
            api_key="sk-test",
            temperature=0.0,
            max_tokens=None,
            max_completion_tokens=8000,
            reasoning_effort="high",
        ),
        llm._mark_system_message_cacheable(
            [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": [{"type": "text", "text": "SCHEMA"}]},
            ]
        ),
        tools=_TOOLS,
    ),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_encoded_body_matches_sync_wire_body(name: str) -> None:
    kwargs = SCENARIOS[name]
    wire, _ = _sync_wire_body(kwargs)
    line = ob.OpenAIBatchBackend(BackendConfig(model=kwargs["model"])).encode(
        BatchRequest(custom_id="r1", kwargs=kwargs)
    )
    assert line["custom_id"] == "r1"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/chat/completions"
    assert line["body"] == wire


def test_litellm_only_extra_keys_never_reach_the_body() -> None:
    kwargs = SCENARIOS["llm_config_extra_litellm_only_keys"]
    body = ob.OpenAIBatchBackend(BackendConfig(model=kwargs["model"])).encode(
        BatchRequest("r1", kwargs)
    )["body"]
    for key in ("num_retries", "extra_headers", "metadata", "api_key", "timeout", "caching"):
        assert key not in body, key
    assert body["user"] == "dgml"


def test_encode_runs_litellm_once_per_distinct_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = ob.encode_body

    def counting(kwargs: dict[str, Any], **kw: Any) -> dict[str, Any]:
        calls.append(1)
        return real(kwargs, **kw)

    monkeypatch.setattr(ob, "encode_body", counting)
    backend = ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o"))
    kwargs = SCENARIOS["text_system_user"]
    first = backend.encode(BatchRequest("a", kwargs))
    backend.encode(BatchRequest("a", dict(kwargs)))
    assert len(calls) == 1
    first["body"]["messages"].clear()  # a caller's copy never poisons the cache
    assert backend.encode(BatchRequest("a", kwargs))["body"]["messages"]
    backend.release(["a"])
    backend.encode(BatchRequest("a", kwargs))
    assert len(calls) == 2


def test_encode_is_pure_and_leaves_kwargs_untouched() -> None:
    kwargs = SCENARIOS["tools_forced_choice"]
    before = json.dumps(kwargs, sort_keys=True)
    backend = ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o"))
    first = backend.encode(BatchRequest("a", kwargs))
    second = backend.encode(BatchRequest("a", kwargs))
    assert first == second
    assert json.dumps(kwargs, sort_keys=True) == before
    assert "api_key" not in first["body"] and "timeout" not in first["body"]


# ---- decode parity -------------------------------------------------------------

_TOOL_BODY: dict[str, Any] = {
    **_CHAT_BODY,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "submit", "arguments": '{"value":"v"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 50},
    },
}


@pytest.mark.parametrize("body", [_CHAT_BODY, _TOOL_BODY], ids=["text", "tool_calls"])
def test_decoded_response_matches_sync_response_at_half_cost(body: dict[str, Any]) -> None:
    kwargs = SCENARIOS["tools_forced_choice"]
    _, sync = _sync_wire_body(kwargs, response=body)
    batch = ob.decode_response(body)
    sync_usage = extract_cost_and_tokens(sync)
    batch_usage = extract_cost_and_tokens(batch)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cache_read_tokens"):
        assert batch_usage[key] == sync_usage[key]
    assert batch_usage["cost_usd"] == pytest.approx(sync_usage["cost_usd"] * 0.5)
    assert batch.choices[0].finish_reason == sync.choices[0].finish_reason
    assert batch.choices[0].message.content == sync.choices[0].message.content
    assert batch["choices"][0]["message"]["content"] == sync.choices[0].message.content
    sync_calls = sync.choices[0].message.tool_calls or []
    batch_calls = batch.choices[0].message.tool_calls or []
    assert [(c.id, c.function.name, c.function.arguments) for c in batch_calls] == [
        (c.id, c.function.name, c.function.arguments) for c in sync_calls
    ]
    llm._require_choices(batch)


def test_decode_unpriceable_model_leaves_cost_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_: Any) -> float:
        raise ValueError("no price")

    monkeypatch.setattr(litellm, "completion_cost", boom)
    response = ob.decode_response(_CHAT_BODY)
    assert "response_cost" not in response._hidden_params
    assert extract_cost_and_tokens(response)["cost_usd"] is None


# ---- lifecycle against a monkeypatched litellm ---------------------------------


def _batch_obj(data: dict[str, Any]) -> LiteLLMBatch:
    """What ``litellm.create_batch``/``retrieve_batch``/``cancel_batch`` really
    return: a ``LiteLLMBatch`` (pydantic), not a dict (verified live, 2026-09)."""
    return LiteLLMBatch(
        **{
            "object": "batch",
            "endpoint": ob.ENDPOINT,
            "completion_window": "24h",
            "created_at": 0,
            "input_file_id": "file_in",
            **data,
        }
    )


def _file_content(text: str) -> HttpxBinaryResponseContent:
    """What ``litellm.file_content`` really returns (verified live, 2026-09)."""
    return HttpxBinaryResponseContent(httpx.Response(200, content=text.encode("utf-8")))


class _FakeLiteLLM:
    """Records litellm batch/file calls and serves canned batch state, as the
    same types real litellm returns (``LiteLLMBatch``, ``OpenAIFileObject``,
    ``HttpxBinaryResponseContent``), so the backend's field access is
    exercised against the real shapes rather than plain dicts."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.uploaded: bytes = b""
        self.batch: dict[str, Any] = {
            "id": "batch_1",
            "status": "validating",
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
            "output_file_id": None,
            "error_file_id": None,
        }
        self.files: dict[str, str] = {}
        # Exceptions ``create_batch`` raises, one per call, before succeeding.
        self.create_errors: list[Exception] = []
        self.deleted: list[str] = []
        self.delete_error: Exception | None = None
        for name in ("create_file", "create_batch", "retrieve_batch", "file_content"):
            monkeypatch.setattr(litellm, name, getattr(self, name))
        monkeypatch.setattr(litellm, "cancel_batch", self.cancel_batch)
        monkeypatch.setattr(litellm, "file_delete", self.file_delete)

    def create_file(self, **kw: Any) -> OpenAIFileObject:
        self.calls.append(("create_file", kw))
        _, payload, _ = kw["file"]
        self.uploaded = payload
        return OpenAIFileObject(
            id="file_in",
            bytes=len(payload),
            created_at=0,
            filename="dgml-batch.jsonl",
            object="file",
            purpose="batch",
            status="processed",
        )

    def create_batch(self, **kw: Any) -> LiteLLMBatch:
        self.calls.append(("create_batch", kw))
        if self.create_errors:
            raise self.create_errors.pop(0)
        return _batch_obj(self.batch)

    def file_delete(self, **kw: Any) -> dict[str, Any]:
        self.calls.append(("file_delete", kw))
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted.append(kw["file_id"])
        return {"id": kw["file_id"], "deleted": True}

    def retrieve_batch(self, **kw: Any) -> LiteLLMBatch:
        self.calls.append(("retrieve_batch", kw))
        return _batch_obj(self.batch)

    def file_content(self, **kw: Any) -> HttpxBinaryResponseContent:
        self.calls.append(("file_content", kw))
        return _file_content(self.files[kw["file_id"]])

    def cancel_batch(self, **kw: Any) -> LiteLLMBatch:
        self.calls.append(("cancel_batch", kw))
        self.batch["status"] = "cancelling"
        return _batch_obj(self.batch)


def _line(custom_id: str, body: dict[str, Any]) -> str:
    return json.dumps(
        {
            "id": f"req_{custom_id}",
            "custom_id": custom_id,
            "response": {"status_code": 200, "request_id": "x", "body": body},
            "error": None,
        }
    )


def _requests(n: int) -> list[BatchRequest]:
    kwargs = SCENARIOS["text_system_user"]
    return [BatchRequest(custom_id=f"r{i}", kwargs=kwargs) for i in range(n)]


def test_submit_uploads_jsonl_and_creates_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    backend = ob.OpenAIBatchBackend(
        BackendConfig(model="openai/gpt-5.4", api_key="sk-cfg", api_base="https://proxy/v1")
    )
    job = backend.submit(_requests(2))
    assert job.provider == "openai" and job.job_id == "batch_1"
    assert job.custom_ids == ("r0", "r1")
    name, upload = fake.calls[0]
    assert name == "create_file"
    assert upload["purpose"] == "batch" and upload["custom_llm_provider"] == "openai"
    assert upload["api_key"] == "sk-cfg" and upload["api_base"] == "https://proxy/v1"
    lines = [json.loads(x) for x in fake.uploaded.decode().splitlines()]
    assert [x["custom_id"] for x in lines] == ["r0", "r1"]
    assert lines[0] == backend.encode(_requests(1)[0])
    name, create = fake.calls[1]
    assert name == "create_batch"
    assert create["input_file_id"] == "file_in"
    assert create["endpoint"] == "/v1/chat/completions"
    assert create["completion_window"] == "24h"
    assert create["custom_llm_provider"] == "openai"
    # The job must survive a JSON round-trip (chunk 15 persists it).
    assert BatchJob.from_json(json.loads(json.dumps(job.to_json()))).extra == job.extra


def test_api_key_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o")).submit(_requests(1))
    assert fake.calls[0][1]["api_key"] == "sk-env"


@pytest.mark.parametrize(
    ("status", "state"),
    [
        ("validating", BatchState.RUNNING),
        ("in_progress", BatchState.RUNNING),
        ("finalizing", BatchState.RUNNING),
        ("cancelling", BatchState.RUNNING),
        ("completed", BatchState.ENDED),
        ("failed", BatchState.FAILED),
        ("expired", BatchState.FAILED),
        ("cancelled", BatchState.CANCELED),
    ],
)
def test_poll_maps_status_and_counts(
    monkeypatch: pytest.MonkeyPatch, status: str, state: BatchState
) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    backend = ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o"))
    job = backend.submit(_requests(3))
    fake.batch.update(status=status, request_counts={"total": 3, "completed": 1, "failed": 1})
    got = backend.poll(job)
    assert got.state is state
    assert (got.succeeded, got.errored) == (1, 1)
    assert got.processing == (1 if state is BatchState.RUNNING else 0)
    assert got.expired == (1 if status == "expired" else 0)
    assert fake.calls[-1] == (
        "retrieve_batch",
        {"batch_id": "batch_1", "custom_llm_provider": "openai"},
    )
    assert job.extra["batch"]["status"] == status


def test_results_decode_outputs_errors_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    backend = ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o"))
    job = backend.submit(_requests(5))
    fake.files["file_out"] = "\n".join(
        [
            _line("r2", _TOOL_BODY),
            _line("r0", _CHAT_BODY),
            json.dumps(
                {
                    "custom_id": "r4",
                    "response": {
                        "status_code": 400,
                        "body": {"error": {"message": "bad param"}},
                    },
                    "error": None,
                }
            ),
        ]
    )
    fake.files["file_err"] = json.dumps(
        {
            "custom_id": "r1",
            "response": None,
            "error": {"code": "server_error", "message": "try again"},
        }
    )
    fake.batch.update(status="completed", output_file_id="file_out", error_file_id="file_err")
    backend.poll(job)
    got = dict(backend.results(job))
    assert set(got) == {"r0", "r1", "r2", "r3", "r4"}
    r0, r2 = got["r0"], got["r2"]
    assert not isinstance(r0, BatchItemError) and not isinstance(r2, BatchItemError)
    assert r0.choices[0].message.content == "ok"
    assert r2.choices[0].message.tool_calls[0].function.name == "submit"
    r1, r3, r4 = got["r1"], got["r3"], got["r4"]
    assert isinstance(r1, BatchItemError) and r1.kind == "errored" and r1.retryable
    assert "server_error" in r1.message
    assert isinstance(r4, BatchItemError) and r4.kind == "invalid" and not r4.retryable
    assert "HTTP 400" in r4.message and "bad param" in r4.message
    # Missing from a COMPLETED batch: a provider-side gap, surfaced (never dropped)
    # as a retryable error; ``expired`` is reserved for a batch that expired.
    assert isinstance(r3, BatchItemError) and r3.kind == "errored" and r3.retryable
    assert "'completed'" in r3.message
    assert ("file_content", {"file_id": "file_out", "custom_llm_provider": "openai"}) in (
        fake.calls
    )


@pytest.mark.parametrize(("status_code", "kind"), [(429, "errored"), (500, "errored")])
def test_retryable_http_statuses(
    monkeypatch: pytest.MonkeyPatch, status_code: int, kind: str
) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    backend = ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o"))
    job = backend.submit(_requests(1))
    fake.files["file_out"] = json.dumps(
        {"custom_id": "r0", "response": {"status_code": status_code, "body": {}}, "error": None}
    )
    fake.batch.update(status="completed", output_file_id="file_out")
    backend.poll(job)
    (cid, outcome), *_ = list(backend.results(job))
    assert cid == "r0" and isinstance(outcome, BatchItemError)
    assert outcome.kind == kind and outcome.retryable


def test_cancel_then_results_reports_canceled(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    backend = ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o"))
    job = backend.submit(_requests(2))
    backend.cancel(job)
    assert fake.calls[-1] == (
        "cancel_batch",
        {"batch_id": "batch_1", "custom_llm_provider": "openai"},
    )
    fake.batch.update(status="cancelled")
    backend.poll(job)
    got = dict(backend.results(job))
    assert all(isinstance(v, BatchItemError) and v.kind == "canceled" for v in got.values())
    assert not any(v.retryable for v in got.values() if isinstance(v, BatchItemError))


def test_failed_batch_maps_every_request_to_non_retryable_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch rejected at validation fails identically on resubmit: no retry."""
    fake = _FakeLiteLLM(monkeypatch)
    backend = ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o"))
    job = backend.submit(_requests(3))
    fake.batch.update(
        status="failed",
        errors={
            "object": "list",
            "data": [
                {"code": "invalid_request", "message": "bad body", "line": 2, "param": None},
                {"code": "too_large", "message": "file exceeds limit", "line": None},
            ],
        },
    )
    assert backend.poll(job).state is BatchState.FAILED
    got = dict(backend.results(job))
    assert set(got) == {"r0", "r1", "r2"}
    for outcome in got.values():
        assert isinstance(outcome, BatchItemError)
        assert outcome.kind == "invalid" and not outcome.retryable
        assert "batch failed validation" in outcome.message
        assert "invalid_request: bad body (line 2)" in outcome.message
        assert "too_large: file exceeds limit" in outcome.message
    # No output/error files exist for a batch that never ran.
    assert not any(name == "file_content" for name, _ in fake.calls)


def test_failed_batch_without_errors_payload_is_still_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    backend = ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o"))
    job = backend.submit(_requests(1))
    fake.batch.update(status="failed")
    backend.poll(job)
    ((_, outcome),) = list(backend.results(job))
    assert isinstance(outcome, BatchItemError)
    assert outcome.kind == "invalid" and not outcome.retryable
    assert outcome.message == "batch failed validation"


def test_expired_batch_keeps_answered_results_and_expires_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    backend = ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o"))
    job = backend.submit(_requests(2))
    fake.files["file_out"] = _line("r0", _CHAT_BODY)
    fake.batch.update(
        status="expired",
        output_file_id="file_out",
        request_counts={"total": 2, "completed": 1, "failed": 0},
    )
    status = backend.poll(job)
    assert status.state is BatchState.FAILED and status.expired == 1
    got = dict(backend.results(job))
    assert not isinstance(got["r0"], BatchItemError)
    r1 = got["r1"]
    assert isinstance(r1, BatchItemError)
    assert r1.kind == "expired" and r1.retryable
    assert "expired" in r1.message


# ---- create safety -------------------------------------------------------------

_REQ = httpx.Request("POST", "https://api.openai.com/v1/batches")


def _status_error(cls: type[openai.APIStatusError], status: int, message: str) -> Exception:
    return cls(
        message,
        response=httpx.Response(status, request=_REQ, json={"error": {"message": message}}),
        body={"error": {"message": message}},
    )


def _connect_error() -> Exception:
    try:
        raise openai.APIConnectionError(request=_REQ) from httpx.ConnectError(
            "refused", request=_REQ
        )
    except openai.APIConnectionError as exc:
        return exc


def _timeout_error() -> Exception:
    try:
        raise openai.APITimeoutError(request=_REQ) from httpx.ReadTimeout("slow", request=_REQ)
    except openai.APITimeoutError as exc:
        return exc


def _creates(fake: _FakeLiteLLM) -> list[dict[str, Any]]:
    return [kw for name, kw in fake.calls if name == "create_batch"]


def _quiet_backend() -> ob.OpenAIBatchBackend:
    return ob.OpenAIBatchBackend(BackendConfig(model="gpt-4o"), sleep=lambda _s: None)


def test_create_disables_the_sdk_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK would resend a POST twice with no idempotency key; it must not."""
    fake = _FakeLiteLLM(monkeypatch)
    _quiet_backend().submit(_requests(1))
    assert _creates(fake)[0]["max_retries"] == 0


@pytest.mark.parametrize(
    "error",
    [
        _status_error(openai.InternalServerError, 500, "boom"),
        _status_error(openai.InternalServerError, 503, "unavailable"),
        _timeout_error(),
    ],
    ids=["500", "503", "read-timeout"],
)
def test_create_that_may_have_landed_is_uncertain_and_never_resent(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    fake.create_errors = [error]
    with pytest.raises(BatchSubmitUncertain):
        _quiet_backend().submit(_requests(1))
    assert len(_creates(fake)) == 1
    # The batch may exist and reference the upload: it is left alone.
    assert fake.deleted == []


@pytest.mark.parametrize(
    "error",
    [_connect_error(), _status_error(openai.RateLimitError, 429, "slow down")],
    ids=["connect-refused", "429"],
)
def test_create_refused_before_acceptance_is_retried(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    fake.create_errors = [error]
    job = _quiet_backend().submit(_requests(1))
    assert job.job_id == "batch_1"
    assert len(_creates(fake)) == 2


def test_create_rate_limited_on_every_attempt_is_throttled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    fake.create_errors = [_status_error(openai.RateLimitError, 429, "queue full")] * 3
    with pytest.raises(BatchThrottled):
        _quiet_backend().submit(_requests(1))
    assert len(_creates(fake)) == 3
    assert fake.deleted == ["file_in"]


def _validation_error() -> Exception:
    """The SDK's failure to parse a 2xx create answer (status 200)."""
    return openai.APIResponseValidationError(
        response=httpx.Response(200, request=_REQ, json={"object": "batch"}),
        body={"object": "batch"},
    )


@pytest.mark.parametrize(
    "error",
    [_validation_error(), ValueError("litellm could not transform the response")],
    ids=["sdk-validation-200", "unreadable"],
)
def test_a_create_that_fails_after_it_may_have_been_accepted_is_uncertain(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    fake.create_errors = [error]
    with pytest.raises(BatchSubmitUncertain):
        _quiet_backend().submit(_requests(1))
    assert len(_creates(fake)) == 1
    assert fake.deleted == []  # a batch may reference the upload


def test_an_accepted_create_without_a_batch_id_is_uncertain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    monkeypatch.setattr(litellm, "create_batch", lambda **_kw: {"status": "validating"})
    with pytest.raises(BatchSubmitUncertain, match="no batch id"):
        _quiet_backend().submit(_requests(1))
    assert fake.deleted == []


class _StreamedContent:
    """A file-content object that can only be iterated (``.text`` would hold
    the whole — up to 200 MB — file in memory, again): F6."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.read = 0

    @property
    def text(self) -> str:
        raise AssertionError("results must not read the whole file into one string")

    @property
    def content(self) -> bytes:
        raise AssertionError("results must not read the whole file into one buffer")

    def iter_lines(self) -> Any:
        for line in self.lines:
            self.read += 1
            yield line


def test_result_files_are_iterated_line_by_line(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    lines = [_line(f"r{i}", _CHAT_BODY) for i in range(30)]
    content = _StreamedContent(lines)
    monkeypatch.setattr(litellm, "file_content", lambda **_kw: content)
    fake.batch.update(status="completed", output_file_id="file_out")
    job = BatchJob(
        provider="openai",
        job_id="batch_1",
        custom_ids=tuple(f"r{i}" for i in range(30)),
        extra={"batch": dict(fake.batch)},
    )
    results = _quiet_backend().results(job)
    assert next(results)[0] == "r0"
    assert content.read < len(lines)
    assert len(list(results)) == 29


def test_result_files_as_bytes_still_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    blob = ("\n".join(_line(f"r{i}", _CHAT_BODY) for i in range(3)) + "\n").encode()
    monkeypatch.setattr(litellm, "file_content", lambda **_kw: blob)
    fake.batch.update(status="completed", output_file_id="file_out")
    job = BatchJob(
        provider="openai",
        job_id="batch_1",
        custom_ids=("r0", "r1", "r2"),
        extra={"batch": dict(fake.batch)},
    )
    out = dict(_quiet_backend().results(job))
    assert sorted(out) == ["r0", "r1", "r2"]
    assert not any(isinstance(v, BatchItemError) for v in out.values())


@pytest.mark.parametrize(
    ("code", "kind", "retryable"),
    [
        ("batch_expired", "expired", True),
        ("batch_cancelled", "canceled", False),
        ("batch_canceled", "canceled", False),
    ],
)
def test_error_file_lines_for_expired_or_cancelled_requests_keep_their_kind(
    monkeypatch: pytest.MonkeyPatch, code: str, kind: str, retryable: bool
) -> None:
    """F8: OpenAI writes a request the batch never ran to the error file with
    ``response: null`` and ``error.code`` batch_expired / batch_cancelled."""
    fake = _FakeLiteLLM(monkeypatch)
    line = json.dumps(
        {
            "id": "batch_req_1",
            "custom_id": "r0",
            "response": None,
            "error": {"code": code, "message": "This request could not be executed"},
        }
    )
    fake.files["file_err"] = line + "\n"
    fake.batch.update(status="expired", error_file_id="file_err")
    job = BatchJob(
        provider="openai", job_id="batch_1", custom_ids=("r0",), extra={"batch": dict(fake.batch)}
    )
    ((cid, outcome),) = list(_quiet_backend().results(job))
    assert cid == "r0" and isinstance(outcome, BatchItemError)
    assert outcome.kind == kind and outcome.retryable is retryable
    assert code in outcome.message


@pytest.mark.parametrize("call", ["poll", "results", "cancel"])
def test_a_batch_the_provider_no_longer_knows_is_batch_not_found(
    monkeypatch: pytest.MonkeyPatch, call: str
) -> None:
    from dgml_core.batch.types import BatchNotFound

    fake = _FakeLiteLLM(monkeypatch)
    gone = _status_error(openai.NotFoundError, 404, "No batch found with id 'batch_gone'.")

    def not_found(**_kw: Any) -> Any:
        raise gone

    for name in ("retrieve_batch", "cancel_batch", "file_content"):
        monkeypatch.setattr(litellm, name, not_found)
    fake.batch.update(status="completed", output_file_id="file_out")
    job = BatchJob(
        provider="openai", job_id="batch_gone", custom_ids=("a",), extra={"batch": fake.batch}
    )
    backend = _quiet_backend()
    with pytest.raises(BatchNotFound, match="batch_gone") as info:
        if call == "results":
            list(backend.results(job))
        else:
            getattr(backend, call)(job)
    assert info.value.__cause__ is gone


def test_create_refused_at_a_limit_is_batch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    fake.create_errors = [
        _status_error(openai.BadRequestError, 400, "Enqueued token limit exceeded for gpt-4o")
    ]
    with pytest.raises(BatchRejected, match="limit"):
        _quiet_backend().submit(_requests(1))
    assert len(_creates(fake)) == 1
    assert fake.deleted == ["file_in"]


def test_create_refused_otherwise_reraises_the_sdk_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    fake.create_errors = [_status_error(openai.AuthenticationError, 401, "bad key")]
    with pytest.raises(openai.AuthenticationError):
        _quiet_backend().submit(_requests(1))
    assert len(_creates(fake)) == 1
    assert fake.deleted == ["file_in"]


def test_failed_batch_at_the_enqueued_token_limit_is_batch_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    backend = _quiet_backend()
    job = backend.submit(_requests(3))
    fake.batch.update(
        status="failed",
        errors={
            "object": "list",
            "data": [
                {
                    "code": "token_limit_exceeded",
                    "message": "Enqueued token limit reached for gpt-4o in organization org-x.",
                    "line": None,
                    "param": None,
                }
            ],
        },
    )
    backend.poll(job)
    got = dict(backend.results(job))
    assert set(got) == {"r0", "r1", "r2"}
    for outcome in got.values():
        assert isinstance(outcome, BatchItemError)
        assert outcome.kind == "batch_rejected" and not outcome.retryable
        assert "token_limit_exceeded" in outcome.message


class _EnqueueLimitedLiteLLM:
    """A stateful litellm stand-in for the executor: every batch of more than
    ``limit`` requests is accepted and then ``failed`` at the enqueued-token
    limit; smaller batches complete with a line per request."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
        self.limit = limit
        self.uploads: dict[str, list[str]] = {}
        self.batches: dict[str, dict[str, Any]] = {}
        self.files: dict[str, str] = {}
        for name in ("create_file", "create_batch", "retrieve_batch", "file_content"):
            monkeypatch.setattr(litellm, name, getattr(self, name))
        monkeypatch.setattr(litellm, "cancel_batch", self.cancel_batch)
        monkeypatch.setattr(litellm, "file_delete", lambda **_kw: {"deleted": True})

    def create_file(self, **kw: Any) -> dict[str, Any]:
        _, payload, _ = kw["file"]
        file_id = f"file_in_{len(self.uploads) + 1}"
        self.uploads[file_id] = [
            json.loads(x)["custom_id"] for x in payload.decode().splitlines() if x
        ]
        return {"id": file_id}

    def create_batch(self, **kw: Any) -> dict[str, Any]:
        ids = self.uploads[kw["input_file_id"]]
        batch_id = f"batch_{len(self.batches) + 1}"
        if len(ids) > self.limit:
            error = {"code": "token_limit_exceeded", "message": "Enqueued token limit reached."}
            batch: dict[str, Any] = {
                "id": batch_id,
                "status": "failed",
                "errors": {"object": "list", "data": [{**error, "line": None}]},
                "request_counts": {"total": 0, "completed": 0, "failed": 0},
            }
        else:
            out = f"file_out_{batch_id}"
            self.files[out] = "\n".join(_line(cid, _CHAT_BODY) for cid in ids)
            batch = {
                "id": batch_id,
                "status": "completed",
                "output_file_id": out,
                "request_counts": {"total": len(ids), "completed": len(ids), "failed": 0},
            }
        self.batches[batch_id] = batch
        return {**batch, "status": "validating"}

    def retrieve_batch(self, **kw: Any) -> dict[str, Any]:
        return dict(self.batches[kw["batch_id"]])

    def file_content(self, **kw: Any) -> bytes:
        return self.files[kw["file_id"]].encode("utf-8")

    def cancel_batch(self, **kw: Any) -> dict[str, Any]:
        raise AssertionError(f"nothing should be canceled: {kw['batch_id']}")


def test_failed_batch_at_the_enqueued_token_limit_is_bisected_by_the_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration: an OpenAI batch accepted and then failed at the enqueued-
    token limit reaches the executor as all-``batch_rejected`` and is split in
    half and resubmitted — never run as a whole wave at full sync price."""
    from dgml_core.batch import BatchExecutor

    fake = _EnqueueLimitedLiteLLM(monkeypatch, limit=2)
    sync_calls: list[dict[str, Any]] = []

    def sync(kwargs: dict[str, Any]) -> Any:
        sync_calls.append(kwargs)
        raise AssertionError("no request may fall back to the sync path")

    ex = BatchExecutor(_quiet_backend(), sync_execute=sync, sleep=lambda _s: None, min_wave_size=1)
    kwargs = SCENARIOS["text_system_user"]
    out = ex.run_wave({f"r{i}": kwargs for i in range(4)})

    assert set(out) == {"r0", "r1", "r2", "r3"}
    assert all(r.choices[0].message.content == "ok" for r in out.values())
    assert sync_calls == []
    assert list(fake.uploads.values()) == [["r0", "r1", "r2", "r3"], ["r0", "r1"], ["r2", "r3"]]
    assert ex.stats.bisections == 1 and ex.stats.batch_ok == 4


# ---- the shapes a real run returns, as small synthetic dicts ---------------------
#
# Synthetic data holding only the fields the backend reads, in the shapes the
# provider actually sends: a request the provider rejects lands in the ERROR file
# as an HTTP 400 ``response`` with ``"error": null`` (not as an ``error``
# object), and a ``validating`` batch reports ``request_counts`` of all zeros,
# ``total`` included.

_SHAPE_IDS = ("t-text", "t-tool", "t-pdf-a", "t-pdf-b", "t-image", "t-invalid")


def _shape_body(message: dict[str, Any], usage: dict[str, Any], finish: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-synthetic",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5-mini-2025-08-07",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    }


def _shape_text(text: str, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return _shape_body({"role": "assistant", "content": text}, usage or _SHAPE_USAGE, "stop")


_SHAPE_USAGE = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
_SHAPE_PDF_USAGE = {
    "prompt_tokens": 2000,
    "completion_tokens": 40,
    "total_tokens": 2040,
    "prompt_tokens_details": {"cached_tokens": 1536},
}
_SHAPE_TOOL_MESSAGE = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "submit", "arguments": '{"value": "forty-two"}'},
        }
    ],
}
_SHAPE_OUTPUT = "\n".join(
    [
        _line("t-text", _shape_text("pong")),
        _line("t-tool", _shape_body(_SHAPE_TOOL_MESSAGE, _SHAPE_USAGE, "tool_calls")),
        _line("t-pdf-a", _shape_text("a", _SHAPE_PDF_USAGE)),
        _line("t-pdf-b", _shape_text("b")),
        _line("t-image", _shape_text("c")),
    ]
)
_SHAPE_ERROR = json.dumps(
    {
        "id": "req_t-invalid",
        "custom_id": "t-invalid",
        "response": {
            "status_code": 400,
            "body": {"error": {"message": "Invalid value for 'tool_choice': no such tool."}},
        },
        "error": None,
    }
)
_SHAPE_BATCH: dict[str, dict[str, Any]] = {
    "validating": {
        "id": "batch_synthetic",
        "status": "validating",
        "request_counts": {"total": 0, "completed": 0, "failed": 0},
    },
    "in_progress": {
        "id": "batch_synthetic",
        "status": "in_progress",
        "request_counts": {"total": 6, "completed": 0, "failed": 0},
    },
    "completed": {
        "id": "batch_synthetic",
        "status": "completed",
        "request_counts": {"total": 6, "completed": 5, "failed": 1},
        "output_file_id": "file_out",
        "error_file_id": "file_err",
    },
}


def _shape_fake(monkeypatch: pytest.MonkeyPatch) -> _FakeLiteLLM:
    fake = _FakeLiteLLM(monkeypatch)
    fake.batch = dict(_SHAPE_BATCH["validating"])
    fake.files["file_out"] = _SHAPE_OUTPUT
    fake.files["file_err"] = _SHAPE_ERROR
    return fake


def _shape_requests() -> list[BatchRequest]:
    kwargs = _kw(llm.LLMConfig(model="openai/gpt-5-mini"), [{"role": "user", "content": "x"}])
    return [BatchRequest(custom_id=c, kwargs=kwargs) for c in _SHAPE_IDS]


def test_poll_counts_through_the_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _shape_fake(monkeypatch)
    backend = _quiet_backend()
    job = backend.submit(_shape_requests())
    assert job.job_id == "batch_synthetic"
    got = backend.poll(job)
    assert got.state is BatchState.RUNNING and got.processing == 0
    fake.batch = dict(_SHAPE_BATCH["in_progress"])
    got = backend.poll(job)
    assert got.state is BatchState.RUNNING and got.processing == 6
    fake.batch = dict(_SHAPE_BATCH["completed"])
    got = backend.poll(job)
    assert got.state is BatchState.ENDED and (got.succeeded, got.errored) == (5, 1)
    # The persisted job state is plain JSON (the LiteLLMBatch was dumped).
    assert json.loads(json.dumps(job.to_json()))["extra"]["batch"]["status"] == "completed"


def test_results_decode_text_tool_and_rejected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _shape_fake(monkeypatch)
    backend = _quiet_backend()
    job = backend.submit(_shape_requests())
    fake.batch = dict(_SHAPE_BATCH["completed"])
    backend.poll(job)
    got: dict[str, Any] = dict(backend.results(job))
    assert set(got) == set(_SHAPE_IDS)

    assert got["t-text"].choices[0].message.content == "pong"
    assert got["t-text"].choices[0].finish_reason == "stop"
    tool = got["t-tool"].choices[0]
    assert tool.finish_reason == "tool_calls"
    assert tool.message.tool_calls[0].function.name == "submit"
    assert json.loads(tool.message.tool_calls[0].function.arguments) == {"value": "forty-two"}

    invalid = got["t-invalid"]
    assert isinstance(invalid, BatchItemError)
    assert invalid.kind == "invalid" and not invalid.retryable
    assert invalid.message.startswith("HTTP 400: Invalid value for 'tool_choice'")

    # Priced at the batch rate, cache reads included (``prompt_tokens_details``):
    # the logged cost is exactly half the standard one.
    pdf = got["t-pdf-a"]
    usage = extract_cost_and_tokens(pdf)
    assert usage["prompt_tokens"] == 2000 and usage["cache_read_tokens"] == 1536
    standard = litellm.completion_cost(completion_response=pdf)
    assert usage["cost_usd"] == pytest.approx(standard * 0.5)
    prices = litellm.model_cost["gpt-5-mini"]
    by_hand = (
        (2000 - 1536) * prices["input_cost_per_token_batches"]
        + 1536 * prices["cache_read_input_token_cost"] * 0.5
        + 40 * prices["output_cost_per_token_batches"]
    )
    assert usage["cost_usd"] == pytest.approx(by_hand)


def test_executor_falls_back_only_for_the_rejected_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _shape_fake(monkeypatch)
    polls = iter(["in_progress", "completed"])

    def retrieve(**kw: Any) -> LiteLLMBatch:
        fake.calls.append(("retrieve_batch", kw))
        fake.batch = dict(_SHAPE_BATCH[next(polls)])
        return _batch_obj(fake.batch)

    monkeypatch.setattr(litellm, "retrieve_batch", retrieve)
    synced: list[dict[str, Any]] = []

    def sync(kwargs: dict[str, Any]) -> Any:
        synced.append(kwargs)
        raise openai.BadRequestError(
            "no such tool", response=httpx.Response(400, request=_REQ), body=None
        )

    ex = BatchExecutor(_quiet_backend(), sync_execute=sync, sleep=lambda _s: None)
    out = ex.run_wave({r.custom_id: r.kwargs for r in _shape_requests()})
    assert len(synced) == 1
    assert isinstance(out["t-invalid"], openai.BadRequestError)
    assert all(out[c].choices for c in _SHAPE_IDS if c != "t-invalid")
    assert (ex.stats.batch_ok, ex.stats.sync_fallbacks, ex.stats.resubmitted) == (5, 1, 0)
    # Every uploaded and produced file is deleted once the batch is collected.
    assert fake.deleted == ["file_in", "file_out", "file_err"]


# ---- cleanup / lifetime ----------------------------------------------------------


def test_cleanup_deletes_input_output_and_error_files(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    backend = _quiet_backend()
    job = backend.submit(_requests(1))
    fake.batch.update(status="completed", output_file_id="file_out", error_file_id="file_err")
    backend.poll(job)
    backend.cleanup(job)
    assert fake.deleted == ["file_in", "file_out", "file_err"]
    assert all(kw["custom_llm_provider"] == "openai" for n, kw in fake.calls if n == "file_delete")


def test_cleanup_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeLiteLLM(monkeypatch)
    backend = _quiet_backend()
    job = backend.submit(_requests(1))
    fake.delete_error = openai.NotFoundError(
        "gone", response=httpx.Response(404, request=_REQ), body=None
    )
    backend.cleanup(job)  # already gone (404) counts as deleted
    assert [n for n, _ in fake.calls].count("file_delete") == 1


def test_a_cleanup_failure_raises_after_trying_every_file_and_is_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dgml_core.batch.executor import cleanup_batch

    fake = _FakeLiteLLM(monkeypatch)
    backend = _quiet_backend()
    job = backend.submit(_requests(1))
    fake.batch.update(status="completed", output_file_id="file_out", error_file_id="file_err")
    backend.poll(job)
    fake.delete_error = openai.InternalServerError(
        "boom", response=httpx.Response(500, request=_REQ), body=None
    )
    with pytest.raises(RuntimeError, match="file_in") as failed:
        backend.cleanup(job)
    assert "file_out" in str(failed.value) and "file_err" in str(failed.value)
    assert [n for n, _ in fake.calls].count("file_delete") == 3
    lines: list[str] = []
    cleanup_batch(backend, job, lines.append)  # never raises
    assert len(lines) == 1 and f"cleanup of {job.job_id} failed" in lines[0]


def test_max_wait_is_the_completion_window() -> None:
    assert _quiet_backend().max_wait_s == 24 * 3600


def test_a_deadline_cancel_waits_two_openai_sweeps_to_settle() -> None:
    """OpenAI settles a cancel on ~5-minute sweeps and keeps processing while
    `cancelling`, so the --batch-deadline settle wait covers two sweeps."""
    from dgml_core.batch.deadline import cancel_settle_s
    from dgml_core.batch.executor import POLL_MARGIN_S

    assert cancel_settle_s(ob.OpenAIBatchBackend) == 660.0
    ex = BatchExecutor(_quiet_backend())
    assert ex.cancel_settle_s == 660.0
    assert ex.max_poll_s == 24 * 3600 + POLL_MARGIN_S


# ---- registry ------------------------------------------------------------------


@pytest.mark.parametrize("model", ["openai/gpt-5.4", "gpt-4o", "openai/o4-mini"])
def test_registry_resolves_openai_models(model: str) -> None:
    assert "openai" in registered_providers()
    backend = resolve_backend(model, api_key="sk-x")
    assert isinstance(backend, ob.OpenAIBatchBackend)
    assert backend.provider == "openai"
    assert backend.max_requests == 50_000
    assert backend.max_bytes == 200 * 1024 * 1024


# ---- live smoke ----------------------------------------------------------------


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
@pytest.mark.allow_network
def test_live_submit_poll_cancel() -> None:  # pragma: no cover - network
    backend = resolve_backend("openai/gpt-4o-mini")
    kwargs = _kw(
        llm.LLMConfig(model="openai/gpt-4o-mini", max_tokens=16),
        [{"role": "user", "content": "Say ok."}],
    )
    job = backend.submit([BatchRequest("live-1", kwargs)])
    try:
        status = backend.poll(job)
        assert status.state in (BatchState.RUNNING, BatchState.ENDED)
        backend.cancel(job)
    finally:
        backend.cleanup(job)  # the uploaded input file would otherwise outlive the test


# ---- Responses-API bridge: never a network call, never batched (F1) --------------
#
# litellm routes some chat calls to ``/v1/responses`` (models whose cost-map
# mode is ``responses``; gpt-5.4+ with tools AND reasoning_effort), through its
# own HTTP stack rather than the injected client. The encode dry run must then
# make no network attempt at all, and batch mode must refuse the request loudly
# (BATCH_UNAVAILABLE) instead of running it at full price.

_BRIDGED: dict[str, dict[str, Any]] = {
    "gpt-5-pro": {"model": "openai/gpt-5-pro"},
    "o3-pro": {"model": "openai/o3-pro"},
    "gpt-5-codex": {"model": "openai/gpt-5-codex"},
    "gpt-5.4_tools_reasoning": {
        "model": "openai/gpt-5.4",
        "tools": _TOOLS,
        "reasoning_effort": "low",
    },
}


class _NetworkAttempt(AssertionError):
    pass


@pytest.fixture
def network_attempts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every real transport/socket attempt, each refused (never reaches the net)."""
    import socket

    attempts: list[str] = []

    def http(self: Any, request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        raise _NetworkAttempt(str(request.url))

    async def ahttp(self: Any, request: httpx.Request) -> httpx.Response:
        attempts.append(str(request.url))
        raise _NetworkAttempt(str(request.url))

    def connect(self: Any, address: Any) -> None:
        attempts.append(f"socket {address}")
        raise _NetworkAttempt(str(address))

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", http)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", ahttp)
    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


def _bridged_kwargs(name: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": "hi"}], **_BRIDGED[name]}


@pytest.mark.parametrize("name", sorted(_BRIDGED))
def test_capture_of_responses_bridged_request_makes_no_network_attempt(
    name: str, network_attempts: list[str]
) -> None:
    from dgml_core.batch._capture import OPENAI_SDK, capture_sync_request

    with pytest.raises(Exception):  # noqa: B017 - any refusal; the point is no I/O
        capture_sync_request(
            _bridged_kwargs(name),
            seam=OPENAI_SDK,
            reply=ob._DRY_RUN_COMPLETION,
            api_key="sk-real-key",
        )
    assert network_attempts == []


@pytest.mark.parametrize("name", sorted(_BRIDGED))
def test_encode_of_responses_bridged_request_is_batch_unavailable(
    name: str, network_attempts: list[str]
) -> None:
    from dgml_core.errors import BatchUnavailable

    backend = ob.OpenAIBatchBackend(BackendConfig(model="openai/gpt-5.4", api_key="sk-x"))
    with pytest.raises(BatchUnavailable, match="Responses API"):
        backend.encode(BatchRequest("r1", _bridged_kwargs(name)))
    assert network_attempts == []


@pytest.mark.parametrize("model", ["openai/gpt-5-pro", "openai/o3-pro", "openai/gpt-5-codex"])
def test_responses_only_model_is_refused_at_resolve(
    model: str, network_attempts: list[str]
) -> None:
    from dgml_core.errors import BatchUnavailable

    with pytest.raises(BatchUnavailable, match="Responses API"):
        resolve_backend(model, api_key="sk-x")
    assert network_attempts == []


def test_bridged_request_fails_the_wave_instead_of_running_at_full_price(
    network_attempts: list[str],
) -> None:
    from dgml_core.errors import BatchUnavailable

    sync_calls: list[Any] = []
    backend = ob.OpenAIBatchBackend(BackendConfig(model="openai/gpt-5.4", api_key="sk-x"))
    executor = BatchExecutor(backend, sync_execute=sync_calls.append, sleep=lambda _s: None)
    with pytest.raises(BatchUnavailable):
        executor.run_wave({"r1": _bridged_kwargs("gpt-5.4_tools_reasoning")})
    assert sync_calls == [] and network_attempts == []


def test_the_network_block_is_lifted_after_a_capture(network_attempts: list[str]) -> None:
    import socket

    from dgml_core.batch._capture import OPENAI_SDK, capture_sync_request

    before = (httpx.HTTPTransport.handle_request, socket.socket.connect, socket.getaddrinfo)
    kwargs = {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    capture_sync_request(kwargs, seam=OPENAI_SDK, reply=ob._DRY_RUN_COMPLETION, api_key="sk-x")
    with pytest.raises(Exception):  # noqa: B017
        capture_sync_request(
            _bridged_kwargs("gpt-5-pro"), seam=OPENAI_SDK, reply=ob._DRY_RUN_COMPLETION
        )
    assert (httpx.HTTPTransport.handle_request, socket.socket.connect, socket.getaddrinfo) == (
        before
    )
    assert network_attempts == []


def test_capture_rejects_a_request_to_an_unexpected_endpoint(
    network_attempts: list[str],
) -> None:
    from dgml_core.batch._capture import OPENAI_SDK, CaptureFailed, capture_sync_request

    kwargs = {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(CaptureFailed, match="/v1/embeddings"):
        capture_sync_request(
            kwargs,
            seam=OPENAI_SDK,
            reply=ob._DRY_RUN_COMPLETION,
            api_key="sk-x",
            expect_path="/v1/embeddings",
        )
    assert network_attempts == []


# ---- X2: the Responses-API refusal is a pre-flight too ------------------------------
#
# A run that would reach an encode-time refusal must be refused before any step
# pays: assert_batchable applies the same rule to each stage's model and to the
# request shape the stage sends (tools + reasoning_effort for schema generation
# and extraction).


@pytest.mark.parametrize("model", ["openai/gpt-5-pro", "openai/o3-pro", "openai/gpt-5-codex"])
def test_preflight_refuses_a_responses_only_model(model: str, network_attempts: list[str]) -> None:
    from dgml_core.batch import assert_batchable
    from dgml_core.errors import BatchUnavailable

    with pytest.raises(BatchUnavailable, match=r"stage 'transcribe'.*Responses API"):
        assert_batchable({"transcribe": model})
    assert network_attempts == []


def test_preflight_refuses_extraction_that_litellm_bridges(network_attempts: list[str]) -> None:
    """gpt-5.4 chat is batchable, but extraction sends tools + reasoning_effort,
    which litellm bridges to /v1/responses: the stage is refused up front."""
    from dgml_core.batch import assert_batchable
    from dgml_core.errors import BatchUnavailable
    from dgml_core.grounded import schema_batch_request, values_batch_request

    assert_batchable({"transcribe": "openai/gpt-5.4"})  # no tools, no reasoning: chat
    with pytest.raises(BatchUnavailable, match=r"stage 'extraction'.*Responses API"):
        assert_batchable({"extraction": values_batch_request("openai/gpt-5.4")})
    with pytest.raises(BatchUnavailable, match=r"stage 'schema'.*Responses API"):
        assert_batchable({"schema": schema_batch_request("openai/gpt-5.4")})
    # A chat-mode model with the same shape stays batchable, as do other providers.
    assert_batchable(
        {
            "extraction": values_batch_request("openai/gpt-4o"),
            "schema": schema_batch_request("anthropic/claude-opus-4-7"),
        }
    )
    assert network_attempts == []


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh"])
def test_preflight_judges_extraction_by_the_configured_effort(
    network_attempts: list[str], effort: str
) -> None:
    """The extraction pre-flight is fed the CONFIGURED values effort: any named
    effort (litellm's ``"none"`` included — it is sent) plus tools bridges
    gpt-5.4 to /v1/responses, so the phase-1 stage is refused before any step
    pays; a chat-mode model with the same configured effort is allowed."""
    from dgml_core.batch import assert_batchable
    from dgml_core.batch.openai import responses_routed
    from dgml_core.errors import BatchUnavailable
    from dgml_core.grounded import values_batch_request, values_batch_stages

    request = values_batch_request("openai/gpt-5.4", effort)
    assert request["reasoning_effort"] == effort
    assert responses_routed(request)
    with pytest.raises(BatchUnavailable, match=r"stage 'extraction' .*Responses API"):
        assert_batchable(values_batch_stages("openai/gpt-5.4", effort))
    assert_batchable(values_batch_stages("openai/gpt-4o", effort))
    assert network_attempts == []


def test_preflight_default_effort_unroutes_phase1_but_not_phase3(
    network_attempts: list[str],
) -> None:
    """``"default"`` (``None``) sends no reasoning effort, so gpt-5.4's phase-1
    request stays on Chat Completions and the pre-flight lets it through — but
    phase 3 always sends ``submit_locations`` with its own fixed effort, which
    still bridges, so the stage is refused there (by name) instead of failing
    at encode after phase 1 had paid. A chat-mode model is allowed."""
    from dgml_core.batch import assert_batchable
    from dgml_core.batch.openai import responses_routed
    from dgml_core.errors import BatchUnavailable
    from dgml_core.grounded import values_batch_request, values_batch_stages

    request = values_batch_request("openai/gpt-5.4", None)
    assert "reasoning_effort" not in request
    assert not responses_routed(request)
    assert_batchable({"extraction": request})  # phase 1 alone is allowed
    with pytest.raises(BatchUnavailable, match=r"stage 'extraction.locations' .*Responses API"):
        assert_batchable(values_batch_stages("openai/gpt-5.4", None))
    assert_batchable(values_batch_stages("openai/gpt-4o", None))
    assert network_attempts == []


def test_preflight_and_encode_share_one_rule(network_attempts: list[str]) -> None:
    """Every bridged shape the encode path refuses, the pre-flight refuses."""
    from dgml_core.batch import assert_batchable
    from dgml_core.errors import BatchUnavailable

    for name in sorted(_BRIDGED):
        with pytest.raises(BatchUnavailable, match="Responses API"):
            assert_batchable({"stage": _bridged_kwargs(name)})
    assert network_attempts == []
