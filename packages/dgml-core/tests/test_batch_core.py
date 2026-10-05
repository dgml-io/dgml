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

"""Tests for the provider-agnostic batch core (`dgml_core.batch`)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from dgml_core.batch import (
    DEFAULT_OVERHEAD_BYTES,
    BackendConfig,
    BatchItemError,
    BatchJob,
    BatchRequest,
    BatchState,
    FakeBackend,
    assert_batchable,
    fake_model_response,
    plan_batches,
    provider_of,
    register_backend,
    registered_providers,
    request_size,
    resolve_backend,
    unregister_backend,
)
from dgml_core.batch import registry as batch_registry
from dgml_core.errors import BatchUnavailable

BEDROCK_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    # Start every test from an empty registry and restore it afterwards. Built-in
    # backends register themselves on import, so without this a test's expected
    # provider list would change every time a new built-in lands, and a test that
    # overrides one would leak its fake into later modules.
    saved = dict(batch_registry._REGISTRY)
    batch_registry._REGISTRY.clear()
    yield
    batch_registry._REGISTRY.clear()
    batch_registry._REGISTRY.update(saved)


def _req(cid: str, text: str = "x") -> BatchRequest:
    return BatchRequest(
        custom_id=cid,
        kwargs={"model": "fake/model", "messages": [{"role": "user", "content": text}]},
    )


def _fake_factory(cfg: BackendConfig) -> FakeBackend:
    return FakeBackend({}, provider=provider_of(cfg.model))


# ---- provider resolution (litellm alone) -------------------------------------


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("anthropic/claude-haiku-4-5", "anthropic"),
        ("claude-sonnet-4-6", "anthropic"),
        ("openai/gpt-5.4", "openai"),
        ("gpt-4o", "openai"),
        ("gemini/gemini-2.5-pro", "gemini"),
        ("bedrock/" + BEDROCK_ID, "bedrock"),
        ("vertex_ai/claude-sonnet-4-6", "vertex_ai"),
        # Claude served by another host is that host's provider, not Anthropic's
        # first-party API — its batch endpoint is a different (or absent) thing.
        ("openrouter/anthropic/claude-sonnet-4", "openrouter"),
        ("azure_ai/claude-3-5-sonnet", "azure_ai"),
        (BEDROCK_ID, "bedrock"),
    ],
)
def test_provider_of_is_litellm_resolution(model: str, provider: str) -> None:
    assert provider_of(model) == provider


def test_provider_of_unparseable_model_raises() -> None:
    with pytest.raises(BatchUnavailable) as info:
        provider_of("totally-unknown-model-xyz")
    assert "totally-unknown-model-xyz" in str(info.value)


@pytest.mark.parametrize(
    "model",
    ["openrouter/anthropic/claude-sonnet-4", "azure_ai/claude-3-5-sonnet", BEDROCK_ID],
)
def test_hosted_claude_is_not_routed_to_anthropic_backend(model: str) -> None:
    register_backend("anthropic", _fake_factory)
    with pytest.raises(BatchUnavailable) as info:
        resolve_backend(model)
    assert f"no batch backend for provider '{provider_of(model)}'" in str(info.value)
    assert "providers with one: anthropic" in str(info.value)


# ---- registry ----------------------------------------------------------------


def test_resolve_backend_returns_registered_fake_per_provider() -> None:
    seen: list[BackendConfig] = []

    def factory(cfg: BackendConfig) -> FakeBackend:
        seen.append(cfg)
        return FakeBackend({}, provider=provider_of(cfg.model))

    for provider in ("anthropic", "openai", "gemini"):
        register_backend(provider, factory)
    for model in ("anthropic/claude-haiku-4-5", "openai/gpt-5.4", "gemini/gemini-2.5-pro"):
        backend = resolve_backend(model, api_key="k", api_base=None)
        assert backend.provider == provider_of(model)
    assert [c.model for c in seen] == [
        "anthropic/claude-haiku-4-5",
        "openai/gpt-5.4",
        "gemini/gemini-2.5-pro",
    ]
    assert all(c.api_key == "k" for c in seen)


def test_resolve_backend_rejects_unregistered_provider_and_lists_registered() -> None:
    register_backend("openai", _fake_factory)
    with pytest.raises(BatchUnavailable) as info:
        resolve_backend("bedrock/" + BEDROCK_ID)
    msg = str(info.value)
    assert "bedrock/" + BEDROCK_ID in msg
    assert "no batch backend for provider 'bedrock'" in msg
    assert "providers with one: openai" in msg
    assert info.value.code == "BATCH_UNAVAILABLE"


def test_missing_dependency_quotes_hint_only_when_registered() -> None:
    def probe() -> None:
        raise ImportError("No module named 'anthropic'")

    register_backend("anthropic", _fake_factory, available=probe)
    with pytest.raises(BatchUnavailable) as info:
        resolve_backend("anthropic/claude-haiku-4-5")
    msg = str(info.value)
    assert "No module named 'anthropic'" in msg
    assert "anthropic/claude-haiku-4-5" in msg
    assert "pip install" not in msg
    assert isinstance(info.value.__cause__, ImportError)

    register_backend(
        "anthropic", _fake_factory, available=probe, install_hint="pip install dgml[batch-x]"
    )
    with pytest.raises(BatchUnavailable) as info:
        resolve_backend("anthropic/claude-haiku-4-5")
    assert "`pip install dgml[batch-x]`" in str(info.value)


def test_factory_import_error_is_missing_dependency_other_errors_propagate() -> None:
    def importless(cfg: BackendConfig) -> FakeBackend:
        raise ImportError("no sdk")

    def misconfigured(cfg: BackendConfig) -> FakeBackend:
        raise ValueError("bad api_base")

    register_backend("anthropic", importless)
    with pytest.raises(BatchUnavailable):
        resolve_backend("anthropic/claude-haiku-4-5")

    register_backend("anthropic", misconfigured)
    with pytest.raises(ValueError, match="bad api_base"):
        resolve_backend("anthropic/claude-haiku-4-5")


def test_assert_batchable_probes_each_provider_once_without_instantiating() -> None:
    probes: list[str] = []
    built: list[str] = []

    def make(provider: str) -> None:
        def probe() -> None:
            probes.append(provider)

        def factory(cfg: BackendConfig) -> FakeBackend:
            built.append(cfg.model)
            return FakeBackend({}, provider=provider)

        register_backend(provider, factory, available=probe)

    make("anthropic")
    make("openai")
    assert_batchable(
        {
            "transcribe": "anthropic/claude-haiku-4-5",
            "label": "anthropic/claude-sonnet-4-6",
            "links": "openai/gpt-5.4",
        }
    )
    assert sorted(probes) == ["anthropic", "openai"]
    assert built == []


def test_assert_batchable_names_first_offending_stage() -> None:
    register_backend("anthropic", _fake_factory)
    with pytest.raises(BatchUnavailable) as info:
        assert_batchable(
            {
                "transcribe": "anthropic/claude-haiku-4-5",
                "label": "bedrock/" + BEDROCK_ID,
                "links": "bedrock/" + BEDROCK_ID,
            }
        )
    assert str(info.value).startswith("stage 'label' (model 'bedrock/" + BEDROCK_ID + "'):")


def test_assert_batchable_missing_dependency_names_stage() -> None:
    def probe() -> None:
        raise ImportError("no sdk")

    register_backend("gemini", _fake_factory, available=probe, install_hint="pip install g")
    with pytest.raises(BatchUnavailable) as info:
        assert_batchable({"transcribe": "gemini/gemini-2.5-pro"})
    msg = str(info.value)
    assert msg.startswith("stage 'transcribe'")
    assert "`pip install g`" in msg


def test_registered_providers_lists_and_unregister_is_idempotent() -> None:
    register_backend("zzz-test", lambda cfg: FakeBackend({}))
    assert "zzz-test" in registered_providers()
    unregister_backend("zzz-test")
    unregister_backend("zzz-test")
    assert "zzz-test" not in registered_providers()


# ---- chunking ----------------------------------------------------------------


def test_plan_batches_splits_on_count_and_keeps_order() -> None:
    backend = FakeBackend({}, max_requests=2)
    reqs = [_req(f"r{i}") for i in range(5)]
    batches = plan_batches(reqs, backend)
    assert [[r.custom_id for r in b] for b in batches] == [["r0", "r1"], ["r2", "r3"], ["r4"]]


def test_plan_batches_splits_on_bytes_including_overhead() -> None:
    backend = FakeBackend({})
    one = request_size(_req("r0", "a" * 100), backend)
    reqs = [_req(f"r{i}", "a" * 100) for i in range(5)]

    # Exactly two fit when overhead is counted.
    backend.max_bytes = 2 * (one + DEFAULT_OVERHEAD_BYTES)
    assert [len(b) for b in plan_batches(reqs, backend)] == [2, 2, 1]
    # One byte short of two: the overhead tips the second request over.
    backend.max_bytes = 2 * (one + DEFAULT_OVERHEAD_BYTES) - 1
    assert [len(b) for b in plan_batches(reqs, backend)] == [1, 1, 1, 1, 1]
    # With no overhead a budget of exactly two encoded sizes fits two.
    backend.max_bytes = 2 * one
    assert [len(b) for b in plan_batches(reqs, backend, overhead_bytes=0)] == [2, 2, 1]
    assert [len(b) for b in plan_batches(reqs, backend)] == [1, 1, 1, 1, 1]


def test_plan_batches_isolates_oversize_request() -> None:
    backend = FakeBackend({})
    small = _req("s", "a")
    big = _req("big", "b" * 10_000)
    backend.max_bytes = (request_size(small, backend) + DEFAULT_OVERHEAD_BYTES) * 3
    batches = plan_batches([small, big, _req("t", "a")], backend)
    assert [[r.custom_id for r in b] for b in batches] == [["s"], ["big"], ["t"]]


def test_plan_batches_empty() -> None:
    assert plan_batches([], FakeBackend({})) == []


# ---- fake backend ------------------------------------------------------------


def test_fake_backend_records_and_returns_scripted_results() -> None:
    ok = fake_model_response("hello", usage={"prompt_tokens": 3, "completion_tokens": 1}, cost=0.5)
    err = BatchItemError("r1", "errored", "boom")
    backend = FakeBackend({"r0": ok, "r1": err}, polls_until_ended=2)
    job = backend.submit([_req("r0"), _req("r1")])
    assert job.custom_ids == ("r0", "r1")
    assert backend.submitted == [[_req("r0"), _req("r1")]]
    assert backend.encoded[0] == {"custom_id": "r0", "params": _req("r0").kwargs}

    assert backend.poll(job).state is BatchState.RUNNING
    assert backend.poll(job).state is BatchState.RUNNING
    status = backend.poll(job)
    assert status.state is BatchState.ENDED and status.done
    assert (status.succeeded, status.errored) == (1, 1)
    assert backend.polls == 3

    results = dict(backend.results(job))
    assert results["r0"] is ok
    assert results["r0"].choices[0].message.content == "hello"
    assert results["r0"]["choices"][0]["message"]["content"] == "hello"
    assert results["r0"]._hidden_params["response_cost"] == 0.5
    assert results["r0"].usage.prompt_tokens == 3
    assert results["r1"] is err


def test_fake_backend_shuffles_results_deterministically() -> None:
    reqs = [_req(f"r{i}") for i in range(20)]
    script = {r.custom_id: fake_model_response(r.custom_id) for r in reqs}
    a = FakeBackend(script, shuffle=True, seed=7)
    b = FakeBackend(script, shuffle=True, seed=7)
    order_a = [cid for cid, _ in a.results(a.submit(reqs))]
    order_b = [cid for cid, _ in b.results(b.submit(reqs))]
    assert order_a == order_b
    assert order_a != [r.custom_id for r in reqs]
    assert sorted(order_a) == sorted(r.custom_id for r in reqs)


def test_fake_backend_callable_script_runs_once_per_request() -> None:
    calls: list[str] = []

    def script(req: BatchRequest) -> object:
        calls.append(req.custom_id)
        return fake_model_response(req.custom_id.upper())

    backend = FakeBackend(script, polls_until_ended=1)
    job = backend.submit([_req("a"), _req("b")])
    assert calls == []  # nothing resolved while RUNNING
    backend.poll(job)
    assert calls == []
    backend.poll(job)  # ENDED: resolves exactly once
    backend.poll(job)
    list(backend.results(job))
    list(backend.results(job))
    assert sorted(calls) == ["a", "b"]
    outcome = dict(backend.results(job))["a"]
    assert not isinstance(outcome, BatchItemError)
    assert outcome.choices[0].message.content == "A"


def test_fake_backend_results_without_poll_resolve_once() -> None:
    calls: list[str] = []

    def script(req: BatchRequest) -> object:
        calls.append(req.custom_id)
        return fake_model_response("x")

    backend = FakeBackend(script)
    job = backend.submit([_req("a")])
    list(backend.results(job))
    backend.poll(job)
    assert calls == ["a"]


def test_fake_backend_missing_scripted_id_is_invalid() -> None:
    backend = FakeBackend({})
    job = backend.submit([_req("nope")])
    (_cid, outcome), *_ = list(backend.results(job))
    assert isinstance(outcome, BatchItemError) and outcome.kind == "invalid"
    assert not outcome.retryable


def test_fake_backend_fail_submit() -> None:
    backend = FakeBackend({}, fail_submit=RuntimeError("down"))
    with pytest.raises(RuntimeError, match="down"):
        backend.submit([_req("a")])


def test_fake_backend_cancel_freezes_unresolved_items_as_canceled() -> None:
    calls: list[str] = []

    def script(req: BatchRequest) -> object:
        calls.append(req.custom_id)
        return fake_model_response("x")

    backend = FakeBackend(script, polls_until_ended=5)
    job = backend.submit([_req("a"), _req("b")])
    assert backend.poll(job).state is BatchState.RUNNING
    backend.cancel(job)
    assert backend.canceled == [job.job_id]
    assert calls == []  # the script never ran for canceled items
    status = backend.poll(job)
    assert status.state is BatchState.CANCELED
    assert (status.canceled, status.succeeded) == (2, 0)
    results = dict(backend.results(job))
    assert all(isinstance(o, BatchItemError) and o.kind == "canceled" for o in results.values())
    assert all(not o.retryable for o in results.values())


def test_fake_backend_cancel_after_end_keeps_delivered_outcomes() -> None:
    backend = FakeBackend({"a": fake_model_response("x")})
    job = backend.submit([_req("a")])
    assert backend.poll(job).state is BatchState.ENDED
    backend.cancel(job)
    status = backend.poll(job)
    assert status.state is BatchState.CANCELED
    assert (status.succeeded, status.canceled) == (1, 0)
    outcome = dict(backend.results(job))["a"]
    assert not isinstance(outcome, BatchItemError)


def test_fake_backend_counts_expired() -> None:
    backend = FakeBackend({"a": BatchItemError("a", "expired", "24h")})
    job = backend.submit([_req("a")])
    status = backend.poll(job)
    assert status.expired == 1 and status.errored == 0


def test_fake_backend_unknown_job() -> None:
    backend = FakeBackend({})
    with pytest.raises(KeyError):
        backend.poll(BatchJob(provider="fake", job_id="nope", custom_ids=()))


# ---- types -------------------------------------------------------------------


def test_batch_job_json_round_trip() -> None:
    job = BatchJob(
        provider="anthropic",
        job_id="msgbatch_1",
        custom_ids=("a", "b"),
        submitted_at="2026-09-24T00:00:00Z",
        extra={"results_url": "https://x"},
    )
    assert BatchJob.from_json(job.to_json()) == job
    assert job.to_json()["custom_ids"] == ["a", "b"]


def test_batch_item_error_retryable_defaults() -> None:
    assert BatchItemError("a", "errored", "m").retryable is True
    assert BatchItemError("a", "expired", "m").retryable is True
    assert BatchItemError("a", "canceled", "m").retryable is False
    assert BatchItemError("a", "invalid", "m").retryable is False
    assert BatchItemError("a", "errored", "m", retryable=False).retryable is False


def test_batch_request_identity_is_custom_id() -> None:
    a1 = _req("a", "first")
    a2 = _req("a", "second")
    b = _req("b")
    assert a1 == a2 and hash(a1) == hash(a2)
    assert a1 != b
    assert len({a1, a2, b}) == 2
    assert a1 != "a"
    with pytest.raises(AttributeError):
        a1.custom_id = "z"  # type: ignore[misc]
