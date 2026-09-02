# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prompt variant selection.

The point of these is the *strictness*: a variant that is misspelled, or that
overrides a prompt that doesn't exist, must fail loudly. A run that silently
ignored the variant it was asked for produces results that look valid and
aren't — which is precisely the failure this mechanism exists to prevent.
"""

from __future__ import annotations

import pytest
from dgml_core.generation import prompts


def test_baseline_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DGML_PROMPT_VARIANT", raising=False)
    assert prompts.active_variant() == prompts.BASELINE
    assert prompts.describe() == "prompts: baseline"


def test_shipped_fewshot_variant_extends_the_baseline() -> None:
    base = prompts.get("label_system")
    few = prompts.get("label_system", variant="fewshot")
    assert few != base
    # A variant is a superset/edit of the baseline, not a rewrite from scratch.
    assert few.startswith(base[:200])
    assert len(few) > len(base)


def test_variant_only_overrides_what_it_names() -> None:
    # `fewshot` touches label_system; everything else must fall through.
    assert prompts.get("transcribe_system_compact", variant="fewshot") == prompts.get(
        "transcribe_system_compact"
    )


def test_env_selects_the_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DGML_PROMPT_VARIANT", "fewshot")
    assert prompts.active_variant() == "fewshot"
    assert prompts.get("label_system") == prompts.get("label_system", variant="fewshot")


def test_unknown_variant_raises_rather_than_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DGML_PROMPT_VARIANT", "fewshpt")  # typo
    with pytest.raises(ValueError, match="unknown prompt variant"):
        prompts.active_variant()


def test_available_variants_lists_baseline_first() -> None:
    variants = prompts.available_variants()
    assert variants[0] == prompts.BASELINE
    assert "fewshot" in variants


def test_unknown_prompt_still_raises() -> None:
    with pytest.raises(KeyError):
        prompts.get("no_such_prompt")


def test_describe_names_what_changed() -> None:
    assert prompts.describe("fewshot") == "prompts: variant=fewshot overrides=['label_system']"
