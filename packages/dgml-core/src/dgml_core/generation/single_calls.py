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

"""The labeling pass's docset-wide calls, routable through a batch stage.

Pass B makes three calls that are not per document: roster planning (a draft,
then a grounded refine turn), gap planning for an extendable authored
vocabulary (one call), and concept descriptions for the roles coined during
labeling (one call). Each is one request chain per docset, so it cannot share
a wave with anything else, but it still bills at half price as a one-unit batch
stage — one round trip per request in the chain.

The call sites (:func:`dgml_core.generation.label.plan_concept_roster`,
:func:`~dgml_core.generation.label.describe_concepts`) go through
:func:`call` / :func:`call_with_refinement` here. With no runner installed
these are exactly ``llm.call`` / ``llm.call_with_refinement`` (a test that
patches either still intercepts every request). Inside
:func:`single_calls_through` they build the very same request with the step
form (``llm.steps_call`` / ``llm.steps_with_refinement``) and hand it to the
runner, which drives it — in batch mode, as a one-unit batch stage (see
:func:`dgml_core.generation.pipeline.batch_single_call_runner`). Either way the
request bytes, the result and the usage accounting are the synchronous call's;
only the tier differs.

A runner returns the generator's result or raises what the synchronous call
would have raised, so each call site's own failure handling (planning and
description are best-effort) is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

from dgml_core import llm

#: Drives one request chain: ``runner(stage, config, steps) -> result``.
#: *stage* names it ("plan", "plan_gaps", "describe"); *config* is the config
#: *steps* was built from, whose usage row the run records.
SingleCallRunner = Callable[[str, llm.LLMConfig, llm.LLMSteps[Any]], Any]

#: Stage names of the three calls.
STAGE_PLAN = "plan"
STAGE_PLAN_GAPS = "plan_gaps"
STAGE_DESCRIBE = "describe"

_RUNNER: ContextVar[SingleCallRunner | None] = ContextVar("dgml_single_call_runner", default=None)


@contextmanager
def single_calls_through(runner: SingleCallRunner | None) -> Iterator[None]:
    """Route this context's planning and description calls through *runner*
    (``None``: synchronously, as without it)."""
    token = _RUNNER.set(runner)
    try:
        yield
    finally:
        _RUNNER.reset(token)


def call(
    stage: str,
    config: llm.LLMConfig,
    *,
    system_prompt: str | tuple[str, str],
    user_content: list[dict[str, Any]],
    cache: bool = False,
) -> str:
    """``llm.call``, or its step form through the installed runner."""
    runner = _RUNNER.get()
    if runner is None:
        return llm.call(config, system_prompt=system_prompt, user_content=user_content, cache=cache)
    steps = llm.steps_call(
        config, system_prompt=system_prompt, user_content=user_content, cache=cache
    )
    return cast(str, runner(stage, config, steps))


def call_with_refinement(
    stage: str,
    config: llm.LLMConfig,
    *,
    system_prompt: str | tuple[str, str],
    user_content: list[dict[str, Any]],
    refine_instruction: list[dict[str, Any]],
    cache: bool = False,
) -> tuple[str, str]:
    """``llm.call_with_refinement``, or its step form through the installed
    runner (the draft and the refine request, one after the other)."""
    runner = _RUNNER.get()
    if runner is None:
        return llm.call_with_refinement(
            config,
            system_prompt=system_prompt,
            user_content=user_content,
            refine_instruction=refine_instruction,
            cache=cache,
        )
    steps = llm.steps_with_refinement(
        config,
        system_prompt=system_prompt,
        user_content=user_content,
        refine_instruction=refine_instruction,
        cache=cache,
    )
    return cast(tuple[str, str], runner(stage, config, steps))
