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

"""Shared LiteLLM dispatch.

Every LLM call in DGML — DGML generation, classification, schema
generation, and grounded value extraction — flows through this module's
:class:`LLMConfig` and the :func:`call` / :func:`call_with_tools`
wrappers. Routing every site through one wrapper keeps two things
consistent:

- **Provider-aware kwarg shaping.** Anthropic's API rejects extended
  thinking (``reasoning_effort``) together with a forced ``tool_choice``.
  The wrapper drops ``reasoning_effort`` for Anthropic-routed models
  when ``tool_choice`` is forced; callers state what they want and the
  wrapper omits fields the provider would reject.
- **Usage telemetry.** The call functions record usage themselves: when a
  config carries a ``workspace`` and ``debug`` is set, each call appends one
  :class:`UsageEvent` to ``usage.jsonl`` (labelled by ``config.operation`` /
  ``config.context``), on both success and failure. Callers don't wire this up
  per call. :func:`record_usage_for` is an optional scope that aggregates the
  calls inside it into a single row for multi-call operations. All recording is
  gated on ``--debug``.

Lives at the package root (:mod:`dgml_core.llm`) so generation and the
non-generation call sites share one implementation.
"""

from __future__ import annotations

import base64
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from typing import Any, cast

import litellm

from .errors import now_iso
from .storage import Workspace
from .usage import (
    OUTCOME_ERROR,
    OUTCOME_OK,
    UsageEvent,
    add_partial,
    extract_cost_and_tokens,
    record_usage,
)

# The CLI contract is "stdout = a single JSON object" (see :mod:`dgml.cli`).
# LiteLLM, by default, prints a "Give Feedback / Get Help" banner to *stdout*
# whenever it maps an exception — including transient errors we catch and
# retry — which prepends non-JSON lines to the payload and breaks ``| jq``
# consumers. Silence that banner globally; :func:`_quiet_stdout` is the
# belt-and-suspenders guard for anything else a dependency writes to stdout.
litellm.suppress_debug_info = True

PDF_NATIVE_MODEL_PATTERNS = [
    r"claude",
    r"gemini",
    r"gpt-4",
    r"gpt-5",
    r"o1",
    r"o3",
    r"o4",
]

ANTHROPIC_MODEL_PATTERNS = [r"claude", r"anthropic"]


def is_anthropic_model(model: str) -> bool:
    """True when the model is routed to Anthropic.

    Anthropic models need ``cache_control`` markers for prompt caching
    (other providers do this implicitly) and reject ``reasoning_effort``
    when ``tool_choice`` forces a function call. Both rules key off this
    check.
    """
    m = model.lower()
    return any(re.search(p, m) for p in ANTHROPIC_MODEL_PATTERNS)


def supports_native_pdf(model: str) -> bool:
    """Heuristic for which models accept base64 PDF documents through LiteLLM.

    LiteLLM exposes `supports_pdf_input` in recent releases; fall back to a
    regex check on the model name if it is unavailable.
    """
    try:
        return bool(litellm.supports_pdf_input(model=model))
    except Exception:
        pass
    model_lower = model.lower()
    return any(re.search(p, model_lower) for p in PDF_NATIVE_MODEL_PATTERNS)


def supports_vision(model: str) -> bool:
    try:
        return bool(litellm.supports_vision(model=model))
    except Exception:
        return False


@dataclass
class LLMConfig:
    """Configuration for a single LiteLLM dispatch.

    Most fields map directly to a ``litellm.completion`` kwarg; ``None``
    means "don't pass this field" so the provider's default applies.
    The Anthropic ``reasoning_effort`` rule is enforced in
    :func:`_build_completion_kwargs`, not here, so callers state intent
    and the wrapper drops conflicting kwargs.

    ``max_tokens`` vs ``max_completion_tokens``: ``max_tokens`` is the
    older OpenAI alias and the generation pipeline still uses it;
    ``max_completion_tokens`` is the newer field grounded extraction
    paths prefer (it's what GPT-5 / o-series accept). Set whichever the
    target provider expects.
    """

    model: str
    api_key: str | None = None
    api_base: str | None = None
    # ``None`` means "don't send temperature" so the provider's own default
    # applies. Schema generation deliberately relies on that — see the note
    # in :mod:`dgml.grounded` about wanting some creativity in field-name
    # choice. Callers that want deterministic decoding pass ``0.0``.
    temperature: float | None = None
    max_tokens: int | None = 16000
    max_completion_tokens: int | None = None
    timeout: float | None = None
    reasoning_effort: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- Usage telemetry -------------------------------------------------
    # Recording context carried on the config so the call functions log
    # ``usage.jsonl`` rows themselves rather than each caller wrapping the call.
    # ``workspace`` is where the row is written; recording is GATED on
    # ``debug`` (no ``--debug`` → no rows, for every operation). ``operation``
    # and ``context`` label the row. Leave ``workspace`` None (the default)
    # for library callers that don't want telemetry.
    workspace: Workspace | None = None
    debug: bool = False
    operation: str | None = None
    context: dict[str, Any] | None = None
    # Internal: set by an active :func:`record_usage_for` scope. While set,
    # the call functions fold their usage into it (one aggregated row for the
    # whole scope) instead of each writing its own row. Never set by callers.
    _usage_sink: dict[str, Any] | None = field(default=None, repr=False, compare=False)


@dataclass
class CallResult:
    """Outcome of a single :func:`litellm.completion` call.

    Wraps the raw response (so callers can read whatever they need off
    it), the extracted message + tool calls, and cost/token metrics
    parsed via :func:`extract_cost_and_tokens`. Multi-call sites fold
    ``usage`` into a running totals dict via :func:`add_partial`;
    single-call sites pass ``usage`` straight through to a
    :class:`UsageEvent`.
    """

    response: Any
    message: Any
    content: str | None
    tool_calls: list[Any]
    finish_reason: str | None
    usage: dict[str, Any]
    duration_s: float


def _pdf_content_block(pdf_bytes: bytes) -> dict[str, Any]:
    """Unified LiteLLM content block for an inline base64 PDF."""
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    return {
        "type": "file",
        "file": {
            "file_data": f"data:application/pdf;base64,{b64}",
        },
    }


def _image_content_block(png_bytes: bytes) -> dict[str, Any]:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


def build_user_content(
    *,
    instruction_text: str,
    pdf_bytes: bytes | None = None,
    images: list[bytes] | None = None,
) -> list[dict[str, Any]]:
    """Build the content array for the user message: text + document attachments."""
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction_text}]
    if pdf_bytes is not None:
        content.append(_pdf_content_block(pdf_bytes))
    if images:
        for img in images:
            content.append(_image_content_block(img))
    return content


def _build_system_message(
    system_prompt: str | tuple[str, str],
    *,
    cache: bool,
    is_anthropic: bool,
) -> dict[str, Any]:
    """Build the system message, applying `cache_control` for Anthropic models.

    `system_prompt` may be a plain string (current behaviour) or a
    `(static_prefix, dynamic_suffix)` tuple. When `cache=True` and the model
    is Anthropic, the static prefix is marked with `cache_control: ephemeral`
    so subsequent calls within the cache TTL (default 5 min) replay it at
    ~10% token cost. For non-Anthropic providers caching happens implicitly;
    we just concatenate.
    """
    if isinstance(system_prompt, tuple):
        static_prefix, dynamic_suffix = system_prompt
    else:
        static_prefix, dynamic_suffix = system_prompt, ""

    if not cache or not is_anthropic:
        joined = static_prefix if not dynamic_suffix else f"{static_prefix}\n{dynamic_suffix}"
        return {"role": "system", "content": joined}

    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": static_prefix,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    if dynamic_suffix:
        blocks.append({"type": "text", "text": dynamic_suffix})
    return {"role": "system", "content": blocks}


def _mark_document_cacheable(
    user_content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tag PDF/image blocks with `cache_control: ephemeral` for Anthropic.

    Returns a shallow-copied list with shallow-copied content blocks so we
    don't mutate the caller's input. Only the last document-like block is
    tagged — Anthropic allows at most 4 cache breakpoints per request, and
    a marker at the end of the doc covers everything before it.
    """
    out: list[dict[str, Any]] = [dict(b) for b in user_content]
    last_doc_idx: int | None = None
    for i, blk in enumerate(out):
        if blk.get("type") in {"file", "image_url"}:
            last_doc_idx = i
    if last_doc_idx is not None:
        out[last_doc_idx] = {**out[last_doc_idx], "cache_control": {"type": "ephemeral"}}
    return out


def _mark_last_block_cacheable(
    user_content: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tag the LAST content block with `cache_control: ephemeral` for Anthropic.

    Used for multi-turn refinement, where the shared prefix is the first user
    turn's text (the document listing), not an attached PDF. Returns a shallow
    copy so the caller's list is untouched.
    """
    if not user_content:
        return user_content
    out: list[dict[str, Any]] = [dict(b) for b in user_content]
    out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
    return out


def call_with_refinement(
    config: LLMConfig,
    *,
    system_prompt: str | tuple[str, str],
    user_content: list[dict[str, Any]],
    refine_instruction: list[dict[str, Any]],
    cache: bool = False,
) -> tuple[str, str]:
    """Grounded two-request refinement; returns ``(draft, refined)``.

    Request 1 is ``(system, user_content)`` → a draft. Request 2 replays
    ``(system, user_content, assistant=draft, refine_instruction)`` → the
    refined answer. Because the model revises its OWN draft while the original
    ``user_content`` is still in view, the second turn is grounded self-critique
    rather than an independent re-draw — it raises recall (fills gaps the draft
    missed) and converges run-to-run variance.

    With ``cache=True`` on Anthropic, the system prefix and the last block of
    ``user_content`` are marked cacheable, so request 2 replays the shared
    prefix at ~10% token cost (within the 5-min TTL).
    """
    is_anthropic = is_anthropic_model(config.model)
    sys_msg = _build_system_message(system_prompt, cache=cache, is_anthropic=is_anthropic)
    user_blocks = (
        _mark_last_block_cacheable(user_content) if (cache and is_anthropic) else user_content
    )
    user_msg = {"role": "user", "content": user_blocks}

    # Both requests fold into one aggregated row.
    with _record_call(config) as totals:
        with _quiet_stdout():
            draft_resp = litellm.completion(
                **_build_completion_kwargs(config, messages=[sys_msg, user_msg])
            )
        add_partial(totals, extract_cost_and_tokens(draft_resp))
        draft = cast(str, draft_resp["choices"][0]["message"]["content"])

        refine_msgs: list[dict[str, Any]] = [
            sys_msg,
            user_msg,
            {"role": "assistant", "content": draft},
            {"role": "user", "content": refine_instruction},
        ]
        with _quiet_stdout():
            refined_resp = litellm.completion(
                **_build_completion_kwargs(config, messages=refine_msgs)
            )
        add_partial(totals, extract_cost_and_tokens(refined_resp))
        refined = cast(str, refined_resp["choices"][0]["message"]["content"])
    return draft, refined


def _is_tool_choice_forced(tool_choice: Any) -> bool:
    """Does ``tool_choice`` *require* the model to call a function?

    Anthropic's "thinking + forced tool" incompatibility triggers on
    forced choice only. ``"required"`` and a ``{"type": "function", ...}``
    object both force a call; ``"auto"`` / ``"none"`` / ``None`` do not.
    """
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice not in ("auto", "none")
    if isinstance(tool_choice, dict):
        return tool_choice.get("type") == "function"
    return False


def _build_completion_kwargs(
    config: LLMConfig,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the kwargs dict for ``litellm.completion``.

    Provider-aware shaping rule: when the model is Anthropic-routed AND
    ``tool_choice`` forces a function call, drop ``reasoning_effort`` —
    Anthropic's API rejects the combination with
    ``invalid_request_error: Thinking may not be enabled when tool_choice
    forces tool use``. All other providers and non-forced choices keep
    ``reasoning_effort`` if the config set one. ``temperature`` is never
    sent to Anthropic-routed models — newer Claude models reject it as
    deprecated, and older ones only accept 1 with thinking enabled.
    """
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
    }
    # Anthropic: never send temperature. Newer Claude models reject it
    # outright ("`temperature` is deprecated for this model") and older ones
    # reject anything but 1 when thinking is enabled — together the provider
    # default is the only always-safe value.
    if config.temperature is not None and not is_anthropic_model(config.model):
        kwargs["temperature"] = config.temperature
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    if config.max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = config.max_completion_tokens
    if config.timeout is not None:
        kwargs["timeout"] = config.timeout
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    if config.reasoning_effort is not None:
        forced = _is_tool_choice_forced(tool_choice)
        if not (forced and is_anthropic_model(config.model)):
            kwargs["reasoning_effort"] = config.reasoning_effort

    kwargs.update(config.extra)
    return kwargs


@contextmanager
def _quiet_stdout() -> Iterator[None]:
    """Redirect anything written to stdout onto stderr for the duration.

    Guards the JSON-on-stdout CLI contract against dependencies (LiteLLM in
    particular) that ``print`` directly to stdout. ``suppress_debug_info``
    silences the known LiteLLM banner; this catches the rest. dgml's own
    output is unaffected — ``_emit`` writes the JSON payload after the
    completion call returns, outside this block.
    """
    with redirect_stdout(sys.stderr):
        yield


def _completion_with_retry(kwargs: dict[str, Any], *, max_retries: int = 3) -> Any:
    """Call litellm.completion with exponential-backoff retries for transient errors."""
    import litellm

    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with _quiet_stdout():
                return litellm.completion(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            # Retry on transient network/server errors only.
            transient = any(
                t in msg
                for t in (
                    "10054",
                    "connection",
                    "reset",
                    "timeout",
                    "internalservererror",
                    "overloaded",
                    "529",
                    "503",
                )
            )
            if not transient or attempt == max_retries - 1:
                raise
            last_exc = exc
            time.sleep(delay)
            delay *= 2
    raise last_exc  # type: ignore[misc]


def _usage_enabled(config: LLMConfig) -> bool:
    """Usage recording happens only under ``--debug`` and only when a
    workspace to write to is set on the config."""
    return config.debug and config.workspace is not None


@contextmanager
def _record_call(config: LLMConfig) -> Iterator[dict[str, Any]]:
    """Per-call auto-recording used *inside* the entry functions.

    Yields a totals dict the entry function accumulates each completion's
    usage into (via :func:`add_partial`). On exit:

    - If an aggregation scope is active (``config._usage_sink`` set by
      :func:`record_usage_for`), fold the totals into it and write nothing —
      the scope emits one combined row.
    - Otherwise, append one :class:`UsageEvent` (gated on ``--debug`` +
      workspace). A single call therefore yields a single row.

    Exceptions propagate after the totals are recorded, so a failed call
    still leaves a row (or contributes its partial usage to the scope).
    The write itself can never break the caller (see :func:`record_usage`).
    """
    totals = empty_usage_totals()
    started = time.monotonic()
    error_msg: str | None = None
    outcome = OUTCOME_OK
    try:
        yield totals
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        outcome = OUTCOME_ERROR
        raise
    finally:
        if config._usage_sink is not None:
            add_partial(config._usage_sink, totals)
        elif config.debug and config.workspace is not None:
            record_usage(
                config.workspace,
                UsageEvent(
                    at=now_iso(),
                    operation=config.operation or "llm_call",
                    model=config.model,
                    cost_usd=totals["cost_usd"],
                    prompt_tokens=totals["prompt_tokens"],
                    completion_tokens=totals["completion_tokens"],
                    total_tokens=totals["total_tokens"],
                    duration_s=round(time.monotonic() - started, 3),
                    outcome=outcome,
                    context=config.context or {},
                    error=error_msg,
                ),
            )


def call(
    config: LLMConfig,
    *,
    system_prompt: str | tuple[str, str],
    user_content: list[dict[str, Any]],
    cache: bool = False,
) -> str:
    """Invoke the configured model and return the assistant text.

    `cache=True` enables provider prompt caching. For Anthropic models it
    adds `cache_control: {"type": "ephemeral"}` markers on the static system
    prefix and the last attached document. For other providers caching is
    implicit (Gemini/OpenAI cache stable prefixes automatically) so the flag
    is a no-op there.
    """
    is_anthropic = is_anthropic_model(config.model)
    sys_msg = _build_system_message(system_prompt, cache=cache, is_anthropic=is_anthropic)
    user_blocks = (
        _mark_document_cacheable(user_content) if (cache and is_anthropic) else user_content
    )
    messages: list[dict[str, Any]] = [
        sys_msg,
        {"role": "user", "content": user_blocks},
    ]
    kwargs = _build_completion_kwargs(config, messages=messages)
    with _record_call(config) as totals:
        response = _completion_with_retry(kwargs)
        add_partial(totals, extract_cost_and_tokens(response))
        return cast(str, response["choices"][0]["message"]["content"])


def call_continued(
    config: LLMConfig,
    *,
    system_prompt: str | tuple[str, str],
    user_content: list[dict[str, Any]],
    cache: bool = False,
    max_rounds: int = 4,
) -> str:
    """Like :func:`call`, but transparently continue a length-truncated reply.

    When the model stops with ``finish_reason == "length"`` (Anthropic
    ``max_tokens``), the partial reply is fed back as an assistant turn so the
    provider resumes generation from its exact end (prefill continuation), and
    the chunks are concatenated into one coherent output. Loops until the reply
    finishes for another reason or ``max_rounds`` is reached. An untruncated
    reply costs exactly one call, identical to :func:`call`.
    """
    is_anthropic = is_anthropic_model(config.model)
    sys_msg = _build_system_message(system_prompt, cache=cache, is_anthropic=is_anthropic)
    user_blocks = (
        _mark_document_cacheable(user_content) if (cache and is_anthropic) else user_content
    )
    base: list[dict[str, Any]] = [sys_msg, {"role": "user", "content": user_blocks}]
    acc = ""
    # One aggregated row for the whole continuation (all rounds summed).
    with _record_call(config) as totals:
        for _ in range(max_rounds):
            # On continuation rounds the accumulated text becomes an assistant
            # prefill; the provider resumes from its exact end (a length cut lands
            # mid-token, so there is no trailing whitespace to trip Anthropic).
            messages = base + ([{"role": "assistant", "content": acc}] if acc else [])
            response = _completion_with_retry(_build_completion_kwargs(config, messages=messages))
            add_partial(totals, extract_cost_and_tokens(response))
            choice = response.choices[0]
            acc += cast(str, choice.message.content or "")
            if getattr(choice, "finish_reason", None) != "length":
                break
    return acc


def call_with_tools(
    config: LLMConfig,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None = None,
) -> CallResult:
    """Invoke the configured model with tool definitions; return the full
    message plus parsed usage.

    Unlike :func:`call`, this exposes the assistant message object so
    callers can inspect ``tool_calls`` and ``finish_reason``. Provider-
    aware kwarg shaping is applied here, so callers don't have to know
    that Anthropic rejects ``reasoning_effort`` with a forced
    ``tool_choice``.

    ``tool_choice`` defaults to ``None``, meaning the kwarg is omitted
    on the wire — every major provider treats absent ``tool_choice`` as
    ``"auto"`` (model decides whether to call a tool), so callers that
    want auto behaviour can leave the argument unset. Pass
    ``"required"`` or a ``{"type": "function", ...}`` dict to force a
    tool call; the Anthropic ``reasoning_effort`` drop is keyed off
    that forced choice.
    """
    kwargs = _build_completion_kwargs(
        config,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
    )
    started = time.monotonic()
    with _record_call(config) as totals:
        response = _completion_with_retry(kwargs)
        usage = extract_cost_and_tokens(response)
        add_partial(totals, usage)
    duration_s = round(time.monotonic() - started, 3)

    message = response.choices[0].message
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    content = getattr(message, "content", None)
    finish_reason = getattr(response.choices[0], "finish_reason", None)

    return CallResult(
        response=response,
        message=message,
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        duration_s=duration_s,
    )


def empty_usage_totals() -> dict[str, Any]:
    """A fresh totals dict shaped like :func:`extract_cost_and_tokens`'s
    output. Callers running multi-call loops accumulate per-call usage
    into one of these via :func:`dgml.usage.add_partial`.
    """
    return {
        "cost_usd": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


@contextmanager
def record_usage_for(config: LLMConfig) -> Iterator[None]:
    """Aggregate every LLM call made with *config* inside this block into ONE
    ``usage.jsonl`` row, instead of one row per call.

    Recording context (``workspace``, ``operation``, ``context``) is read off
    the config, and the whole scope is gated on ``config.debug`` — with
    ``--debug`` off (or no workspace) this is a transparent no-op. Use it only
    for a genuinely multi-call operation (e.g. a per-page extraction loop);
    single-call sites need no wrapper — the call records its own row.

    While the scope is open, the entry functions fold their per-call usage into
    a shared accumulator rather than each writing a row; on exit — success or
    exception — one combined :class:`UsageEvent` is appended. Nesting is safe:
    an inner scope defers to the outer one. The write can never break the
    caller (see :func:`record_usage`); exceptions propagate after the row.
    """
    # Disabled, or already inside an outer scope → pass through untouched.
    if not _usage_enabled(config) or config._usage_sink is not None:
        yield
        return

    totals = empty_usage_totals()
    started = time.monotonic()
    error_msg: str | None = None
    outcome = OUTCOME_OK
    config._usage_sink = totals
    try:
        yield
    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {exc}"
        outcome = OUTCOME_ERROR
        raise
    finally:
        config._usage_sink = None
        workspace = config.workspace
        if workspace is not None:
            record_usage(
                workspace,
                UsageEvent(
                    at=now_iso(),
                    operation=config.operation or "llm_call",
                    model=config.model,
                    cost_usd=totals["cost_usd"],
                    prompt_tokens=totals["prompt_tokens"],
                    completion_tokens=totals["completion_tokens"],
                    total_tokens=totals["total_tokens"],
                    duration_s=round(time.monotonic() - started, 3),
                    outcome=outcome,
                    context=config.context or {},
                    error=error_msg,
                ),
            )


__all__ = [
    "ANTHROPIC_MODEL_PATTERNS",
    "PDF_NATIVE_MODEL_PATTERNS",
    "CallResult",
    "LLMConfig",
    "add_partial",
    "build_user_content",
    "call",
    "call_with_tools",
    "empty_usage_totals",
    "is_anthropic_model",
    "record_usage_for",
    "supports_native_pdf",
    "supports_vision",
]
