# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Which providers can continue a prefilled assistant turn.

These are regression tests for a failure that was expensive precisely because
it did not look like a failure. Gemini rejects any request whose last message
is a model turn, so every truncated transcription errored; the resulting DGML
carried structural chunks and no semantic tags, which read as "Gemini is a poor
model" rather than "the call was malformed".
"""

from __future__ import annotations

import pytest
from dgml_core.llm import (
    LLMConfig,
    is_gemini_model,
    is_openai_reasoning_model,
    supports_assistant_prefill,
)


@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemini-3.6-flash",
        "gemini/gemini-3.1-pro-preview",
        "gemini/gemini-3-flash-preview",
        "vertex_ai/gemini-2.5-pro",
    ],
)
def test_gemini_never_prefills(model: str) -> None:
    # "Requests ending with a model turn are not supported." (400)
    assert is_gemini_model(model)
    assert not supports_assistant_prefill(model)


@pytest.mark.parametrize("model", ["gpt-5.4", "openai/gpt-4o", "o3-mini"])
def test_openai_never_prefills(model: str) -> None:
    assert not supports_assistant_prefill(model)


@pytest.mark.parametrize("model", ["anthropic/claude-haiku-4-5", "claude-opus-5"])
def test_anthropic_prefills(model: str) -> None:
    assert supports_assistant_prefill(model)
    assert not is_gemini_model(model)


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-5", "claude-sonnet-5"])
def test_sonnet5_refuses_prefill_despite_being_anthropic(model: str) -> None:
    """Measured, not assumed.

    claude-sonnet-5 returns 400 "This model does not support assistant message
    prefill" on the transcription path, which sets no reasoning_effort — so it
    is the model, not the documented thinking/prefill incompatibility. It cost
    17-25% of documents per draw, and since F1 is scored over the survivors the
    arm then topped the sweep.
    """
    assert not supports_assistant_prefill(model)


def test_haiku_is_not_caught_by_the_sonnet5_rule() -> None:
    # The exclusion must stay narrow: haiku-4-5 prefills fine over thousands of
    # calls, so a blanket "newer Claude" rule would lose a working fast path.
    assert supports_assistant_prefill("anthropic/claude-haiku-4-5")
    assert supports_assistant_prefill("anthropic/claude-sonnet-4-6")


def test_unknown_provider_still_defaults_to_prefill() -> None:
    # A self-hosted or proxied OpenAI-compatible server passes a trailing
    # assistant turn through, where prefill is the natural behaviour.
    assert supports_assistant_prefill("my-local/llama-3-70b")


def test_anthropic_model_names_are_not_matched_as_gemini() -> None:
    # 'google' appears in the Gemini patterns; make sure that cannot leak.
    assert not is_gemini_model("anthropic/claude-haiku-4-5")
    assert not is_gemini_model("gpt-5.4")


def test_reasoning_disables_prefill_even_on_anthropic() -> None:
    """Extended thinking and prefill are mutually exclusive on Anthropic.

    Mirrors the gate in call_continued; kept as a test so the interaction is
    documented somewhere executable. The model has to be one that prefills in
    the first place, or the assertion passes for the wrong reason — hence
    haiku-4-5 rather than sonnet-5, which refuses prefill outright (see
    test_sonnet5_refuses_prefill_despite_being_anthropic).
    """
    model = "anthropic/claude-haiku-4-5"
    with_reasoning = LLMConfig(model=model, reasoning_effort="medium")
    without = LLMConfig(model=model)
    assert supports_assistant_prefill(model) and without.reasoning_effort is None

    def gate(c: LLMConfig) -> bool:
        return supports_assistant_prefill(c.model) and c.reasoning_effort is None

    assert gate(without)
    assert not gate(with_reasoning)


# ---------------------------------------------------------------------------
# Temperature: OpenAI reasoning families accept only the default (1)
# ---------------------------------------------------------------------------
#
# `litellm.drop_params` is supposed to absorb this and does for models it has
# metadata for, but it passed `temperature` straight through for gpt-5.5, which
# answered 400 and cost 42 of 114 transcription calls — a cell that produced no
# scoreable documents and looked like a model too weak to transcribe.


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5.5",
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "gpt-5",
        "openai/gpt-5.9-turbo",  # a release that does not exist yet
        "o3-mini",
        "openai/o4-mini",
        "o1",
    ],
)
def test_reasoning_models_reject_explicit_temperature(model: str) -> None:
    assert is_openai_reasoning_model(model)


@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-4o",
        "gpt-4-turbo",
        "anthropic/claude-haiku-4-5",
        "gemini/gemini-3.6-flash",
    ],
)
def test_non_reasoning_models_keep_temperature(model: str) -> None:
    assert not is_openai_reasoning_model(model)


def test_future_gpt5_point_releases_are_covered_by_prefix() -> None:
    """The bug was a version the code had never heard of.

    Matching by family prefix rather than an enumerated list is the point: an
    allow-list stops covering the line the day a new version ships.
    """
    for v in ("gpt-5.6", "gpt-5.7-mini", "openai/gpt-5.12"):
        assert is_openai_reasoning_model(v), v
