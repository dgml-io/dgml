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

"""Unit tests for the OpenAI text-embedding encoder (litellm mocked — no key)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from clustering.config.schema import EncoderConfig
from clustering.encoders import build_encoder
from clustering.encoders.openai_embeddings import OpenAIEncoder

MODEL = "openai/text-embedding-3-large"


def _cfg(*, max_length: int | None = None, **extra: Any) -> EncoderConfig:
    return EncoderConfig(
        name="openai",
        model_id=MODEL,
        embedding_dim=3,
        max_length=max_length,
        extra={"api_key": "test-key", **extra},
    )


def _fake_embedding(recorder: list[dict[str, Any]]) -> Any:
    """A litellm.embedding stub whose vectors identify their own input.

    Row order is the only thing tying a document to its vector, so the stub
    encodes the input's length into the vector: a reordered or shortened
    response shows up as a wrong row, not as a shape that happens to match.
    """

    def fake(*, model: str, input: list[str], **kwargs: Any) -> Any:
        recorder.append({"model": model, "input": list(input), **kwargs})
        return SimpleNamespace(data=[{"embedding": [float(len(t)), 0.0, 1.0]} for t in input])

    return fake


def test_openai_is_registered() -> None:
    assert isinstance(build_encoder(_cfg()), OpenAIEncoder)


def test_openai_encodes_in_batches_preserving_row_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("litellm.embedding", _fake_embedding(calls))

    enc = OpenAIEncoder(_cfg(batch_size=2))
    out = enc.encode(["a", "bb", "ccc", "dddd", "eeeee"])

    assert out.pooled.shape == (5, 3)
    assert enc.multi_vector is False
    assert [c["input"] for c in calls] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert out.pooled[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_openai_truncates_to_max_length(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("litellm.embedding", _fake_embedding(calls))

    # 10 tokens -> 40 characters.
    out = OpenAIEncoder(_cfg(max_length=10)).encode(["x" * 500, "short"])

    assert [len(t) for t in calls[0]["input"]] == [40, 5]
    assert out.pooled[:, 0].tolist() == [40.0, 5.0]


def test_openai_without_max_length_sends_text_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("litellm.embedding", _fake_embedding(calls))

    OpenAIEncoder(_cfg()).encode(["y" * 500])

    assert [len(t) for t in calls[0]["input"]] == [500]


def test_openai_forwards_dimensions_only_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("litellm.embedding", _fake_embedding(calls))

    OpenAIEncoder(_cfg()).encode(["a"])
    # Provider default left alone unless asked for.
    assert "dimensions" not in calls[0]
    # `num_retries` is ours, not litellm's — forwarding it would read as a retry
    # policy and do nothing on an embedding call.
    assert "num_retries" not in calls[0]

    calls.clear()
    OpenAIEncoder(_cfg(dimensions=3)).encode(["a"])
    assert calls[0]["dimensions"] == 3


def test_openai_rejects_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    def wrong_width(*, model: str, input: list[str], **kwargs: Any) -> Any:
        return SimpleNamespace(data=[{"embedding": [0.0, 1.0]} for _ in input])  # 2-d, cfg wants 3

    monkeypatch.setattr("litellm.embedding", wrong_width)
    with pytest.raises(ValueError, match="embedding_dim"):
        OpenAIEncoder(_cfg()).encode(["a"])


def test_openai_retries_transient_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 mid-corpus must not throw away the embeddings already paid for."""
    from litellm.exceptions import RateLimitError

    calls: list[dict[str, Any]] = []
    ok = _fake_embedding(calls)
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)

    attempts = {"n": 0}

    def flaky(**kwargs: Any) -> Any:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RateLimitError("rate limited", llm_provider="openai", model=MODEL)
        return ok(**kwargs)

    monkeypatch.setattr("litellm.embedding", flaky)
    out = OpenAIEncoder(_cfg(num_retries=3)).encode(["hello"])

    assert attempts["n"] == 2  # first failed, retry succeeded
    assert len(slept) == 1  # backed off once
    assert out.pooled[:, 0].tolist() == [5.0]


def test_openai_empty_batch_returns_shaped_empty() -> None:
    out = OpenAIEncoder(_cfg()).encode([])
    assert out.pooled.shape == (0, 3)


def test_openai_config_fingerprint_excludes_api_key() -> None:
    # Rotating the key must not orphan the cache (credential is not a model setting).
    from clustering.encoders.caching import encoder_fingerprint

    a = encoder_fingerprint(_cfg(api_key="key-1"))
    b = encoder_fingerprint(_cfg(api_key="key-2"))
    assert a == b


def test_openai_rejects_a_short_response(monkeypatch: pytest.MonkeyPatch) -> None:
    # A response with fewer rows than inputs would misattribute every embedding
    # after the gap, so it must raise rather than return a misaligned tensor.
    def short(*, model: str, input: list[str], **kwargs: Any) -> Any:
        return SimpleNamespace(data=[{"embedding": [1.0, 0.0, 1.0]}])  # 1 row for 2 inputs

    monkeypatch.setattr("litellm.embedding", short)
    with pytest.raises(ValueError, match="cannot be aligned"):
        OpenAIEncoder(_cfg()).encode(["a", "b"])


def test_openai_does_not_retry_a_deterministic_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # An auth failure is deterministic: it must propagate on the first attempt,
    # not get swept into the transient tuple and slept on.
    from litellm.exceptions import AuthenticationError

    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    attempts = {"n": 0}

    def auth_fail(**kwargs: Any) -> Any:
        attempts["n"] += 1
        raise AuthenticationError("bad key", llm_provider="openai", model=MODEL)

    monkeypatch.setattr("litellm.embedding", auth_fail)
    with pytest.raises(AuthenticationError):
        OpenAIEncoder(_cfg(num_retries=3)).encode(["a"])
    assert attempts["n"] == 1  # not retried
    assert slept == []  # never backed off


def test_openai_gives_up_after_num_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.exceptions import RateLimitError

    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    attempts = {"n": 0}

    def always_429(**kwargs: Any) -> Any:
        attempts["n"] += 1
        raise RateLimitError("rate limited", llm_provider="openai", model=MODEL)

    monkeypatch.setattr("litellm.embedding", always_429)
    with pytest.raises(RateLimitError):
        OpenAIEncoder(_cfg(num_retries=2)).encode(["a"])
    assert attempts["n"] == 3  # first + 2 retries, bounded
    assert len(slept) == 2  # backed off between attempts, not after the last


def test_openai_resolves_key_from_env_then_defers_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No literal api_key: read the named env var, else leave None for litellm.
    cfg_no_key = EncoderConfig(name="openai", model_id=MODEL, embedding_dim=3, extra={})
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert OpenAIEncoder(cfg_no_key).api_key == "from-env"

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAIEncoder(cfg_no_key).api_key is None


def test_openai_accepts_object_style_response(monkeypatch: pytest.MonkeyPatch) -> None:
    # litellm may return items with an `.embedding` attribute rather than a dict.
    def object_style(*, model: str, input: list[str], **kwargs: Any) -> Any:
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(len(t)), 0.0, 1.0]) for t in input]
        )

    monkeypatch.setattr("litellm.embedding", object_style)
    out = OpenAIEncoder(_cfg()).encode(["a", "bb"])
    assert out.pooled[:, 0].tolist() == [1.0, 2.0]
