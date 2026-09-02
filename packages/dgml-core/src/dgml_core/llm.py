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
- **Provider-aware message shaping.** Prompt-cache markers
  (``cache_control``) are Anthropic-only, and continuing a length-truncated
  reply by prefilling an assistant turn works on Anthropic ONLY. OpenAI treats
  a trailing assistant message as a finished turn and starts a fresh reply;
  Gemini refuses the request outright (``400``, "Requests ending with a model
  turn are not supported"). Anthropic itself disallows prefill when extended
  thinking is enabled. :func:`call_continued` asks explicitly for a
  continuation in those cases (see :func:`supports_assistant_prefill`), so a
  caller gets one coherent output on every provider.
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

from .errors import EmptyModelResponse, ModelNotSupported, now_iso, short_error_message
from .prompts import PromptKey
from .prompts import get as prompt
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

# Providers disagree about which sampling knobs they accept: OpenAI's reasoning
# families (gpt-5, o-series) reject any explicit ``temperature`` but 1, and
# Anthropic rejects it as deprecated on newer models. Callers here state the
# decoding they want (extraction asks for 0.0 so the source-text → schema-slot
# mapping is deterministic) rather than tracking per-provider support.
#
# `drop_params` handles most mismatches, but NOT reliably: it only drops a
# parameter for models it has metadata for, and passes it through for one it does
# not recognise. A new model release therefore fails closed at the provider —
# gpt-5.5 lost 42 of 114 calls that way. So the two known-categorical rules are
# enforced explicitly in `_build_completion_kwargs` and drop_params is the
# backstop, not the mechanism.
litellm.drop_params = True

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
GEMINI_MODEL_PATTERNS = [r"gemini", r"vertex_ai", r"google"]

# Anthropic models that refuse a prefilled assistant turn, despite prefill being
# an Anthropic feature. Measured, not assumed: claude-sonnet-5 answers
#
#   400 "This model does not support assistant message prefill.
#        The conversation must end with a user message."
#
# on the transcription path, which sets no `reasoning_effort` — so this is not
# the documented thinking/prefill incompatibility, it is the model. It cost
# 17-25% of documents in every claude-sonnet-5 transcriber draw, and because F1
# is computed over the survivors the arm then scored *highest* of the sweep.
# claude-haiku-4-5 prefills fine across thousands of calls, so this stays a
# narrow list rather than a blanket rule.
ANTHROPIC_NO_PREFILL_PATTERNS = [r"claude-sonnet-5"]

# OpenAI families that accept ONLY the default temperature (1). Prefix-matched
# so future point releases are covered automatically; see
# is_openai_reasoning_model().
OPENAI_REASONING_MODEL_PATTERNS = [
    r"gpt-5",
    r"^o[1-9]\b",
    r"^o[1-9]-",
    r"/o[1-9]\b",
    r"/o[1-9]-",
]

# Matches the model ids litellm routes to OpenAI's own API: the ``openai/``
# prefix, plus the bare families litellm accepts without one (``gpt-5.4``,
# ``o4-mini``, …). Deliberately NOT matched: ``azure/``-prefixed deployments of
# the same models — Azure OpenAI is a different endpoint with its own quirks,
# and nothing here has been validated against it.
OPENAI_MODEL_PATTERNS = [
    r"^openai/",
    r"^gpt-",
    r"^o[1-9]\b",
    r"^o[1-9]-",
]


def is_model_reachability_error(exc: BaseException) -> bool:
    """True when *exc* means the model could not be reached or used at all.

    Auth failure, bad model id, connection error — a config problem worth
    surfacing — as opposed to a call that *succeeded* but returned unusable or
    empty content (a soft outcome we don't flag). LiteLLM's documented contract
    is that every provider-side failure it raises (authentication, bad-request,
    not-found, connection, rate limit, timeout) is an OpenAI-SDK exception, all
    of which derive from ``openai.APIError``. (Note ``litellm.exceptions.APIError``
    is litellm's *own* subclass and is NOT a base of the concrete errors, so it
    can't be used here.) A ``ValueError`` / ``JSONDecodeError`` from parsing a
    *successful* response is not an ``openai.APIError``, so a garbage-but-returned
    payload stays soft.
    """
    import openai  # transitive via litellm; its exception classes ARE litellm's

    return isinstance(exc, openai.APIError)


def is_anthropic_model(model: str) -> bool:
    """True when the model is routed to Anthropic.

    Anthropic models need ``cache_control`` markers for prompt caching
    (other providers do this implicitly) and reject ``reasoning_effort``
    when ``tool_choice`` forces a function call. Both rules key off this
    check.
    """
    m = model.lower()
    return any(re.search(p, m) for p in ANTHROPIC_MODEL_PATTERNS)


def is_openai_model(model: str) -> bool:
    """True when the model is routed to OpenAI's own API.

    OpenAI differs from the other two providers in ways the wrapper has to
    absorb rather than push onto callers: it will not extend a prefilled
    assistant turn (see :func:`supports_assistant_prefill`), and it rejects
    ``cache_control`` markers, which is why every marker in this module is
    gated on :func:`is_anthropic_model` instead of "not OpenAI".

    ``max_tokens`` → ``max_completion_tokens`` is left to litellm's own OpenAI
    transformations. The ``temperature`` restriction is NOT — see
    :func:`is_openai_reasoning_model`, which the kwarg builder consults directly
    because ``drop_params`` was observed passing the parameter through for an
    unrecognised model and losing the call.
    """
    m = model.lower()
    return any(re.search(p, m) for p in OPENAI_MODEL_PATTERNS)


def is_gemini_model(model: str) -> bool:
    """True for Google Gemini / Vertex AI models."""
    m = model.lower()
    return any(re.search(p, m) for p in GEMINI_MODEL_PATTERNS)


def is_openai_reasoning_model(model: str) -> bool:
    """True for OpenAI families that accept only the default ``temperature``.

    The gpt-5 line and the o-series reject an explicit temperature with a 400
    rather than ignoring it. Matched by family prefix rather than by an
    enumerated list of versions: a new ``gpt-5.x`` inherits the constraint, and
    an allow-list would silently stop covering it the day it ships — which is
    exactly how gpt-5.5 slipped through and lost 42 calls.
    """
    m = model.lower()
    return any(re.search(p, m) for p in OPENAI_REASONING_MODEL_PATTERNS)


def supports_assistant_prefill(model: str) -> bool:
    """Can this provider *continue* a trailing assistant message?

    Anthropic resumes generation from the exact end of a prefilled assistant
    turn, which is what makes the cheap continuation in :func:`call_continued`
    work: replay the partial reply as an assistant turn and concatenate what
    comes back.

    OpenAI does not. A trailing assistant message is read as a *completed*
    turn, so the model answers afresh — it re-emits the reply from the top, and
    concatenating that onto the partial yields duplicated, unparseable output.

    **Gemini does not either, and refuses outright.** The API rejects any
    request whose final message is a model turn::

        400 INVALID_ARGUMENT
        "Requests ending with a model turn are not supported."

    This was previously assumed to work, and the cost of the assumption was
    total rather than partial: every Gemini transcription that hit the
    truncation path failed, and the affected runs produced DGML containing
    structural chunks and *no semantic tags at all* — 3 distinct tag names
    across an entire workspace, against ~1500 for a working model. Tag-blind
    recall stayed near 74%, so the text was being read and simply never
    labelled, which made the failure look like poor model quality rather than a
    broken call.

    For both providers :func:`call_continued` falls back to showing the partial
    as an assistant turn followed by an explicit continuation instruction in a
    final *user* turn, which satisfies Gemini's constraint and stops OpenAI
    restarting the reply.

    **Some Anthropic models refuse it too**, which is why this is not simply
    "is it Anthropic". ``claude-sonnet-5`` returns the same shape of 400 as
    Gemini on a path that sets no ``reasoning_effort``, so it is the model and
    not the thinking/prefill incompatibility — see
    ``ANTHROPIC_NO_PREFILL_PATTERNS``.

    Defaults to ``True`` for anything unrecognized (a custom ``api_base``,
    a self-hosted or proxied model): most OpenAI-compatible servers pass a
    trailing assistant turn straight into the prompt, where prefill is the
    natural behaviour.
    """
    m = model.lower()
    if any(re.search(p, m) for p in ANTHROPIC_NO_PREFILL_PATTERNS):
        return False
    return not (is_openai_model(model) or is_gemini_model(model))


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


def _mark_system_message_cacheable(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a shallow copy of *messages* with the system message tagged
    ``cache_control: {"type": "ephemeral"}`` for Anthropic (tool-call path).

    Mirrors :func:`_build_system_message` for :func:`call_with_tools`, whose
    callers pass a fully-built ``messages`` list rather than the
    ``system_prompt`` + ``user_content`` split the text path uses. Only the
    first ``role == "system"`` message is tagged (the extraction call sites have
    exactly one, always first): a string ``content`` is wrapped into a single
    cacheable text block; a list ``content`` gets the marker on its last block.

    The system prompt is the block callers are most likely to share across
    requests, but a marker is not a guarantee: the provider caches nothing if
    the prefix ahead of the breakpoint — tools, then system — is shorter than
    its minimum cacheable length, and nothing is read back unless a later
    request repeats that prefix exactly.

    Per-request-volatile user content (per-file PDF, per-page image, per-page
    OCR words) is deliberately left untouched — a caller that also has a
    *stable* user block (e.g. the docset schema text in phase-1 extraction) tags
    it itself before calling. A shallow copy is returned so the caller's list,
    reused across a tool loop, is never mutated.
    """
    out: list[dict[str, Any]] = [dict(m) for m in messages]
    for i, msg in enumerate(out):
        if msg.get("role") != "system":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            out[i] = {
                **msg,
                "content": [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        elif isinstance(content, list) and content:
            blocks = [dict(b) for b in content]
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            out[i] = {**msg, "content": blocks}
        break
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

    # Both requests fold into one aggregated row. Route through
    # _completion_with_retry so an empty/transient response is retried rather
    # than crashing on choices[0] (same guard as every other call site).
    with _record_call(config) as totals:
        draft_resp = _completion_with_retry(
            _build_completion_kwargs(config, messages=[sys_msg, user_msg])
        )
        add_partial(totals, extract_cost_and_tokens(draft_resp))
        draft = cast(str, draft_resp["choices"][0]["message"]["content"])

        refine_msgs: list[dict[str, Any]] = [
            sys_msg,
            user_msg,
            {"role": "assistant", "content": draft},
            {"role": "user", "content": refine_instruction},
        ]
        refined_resp = _completion_with_retry(
            _build_completion_kwargs(config, messages=refine_msgs)
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


def _require_supported_model(model: str, api_base: str | None) -> None:
    """Fail fast when litellm doesn't recognize *model*.

    An unrecognized id (a misspelling, a missing/wrong provider prefix, or a
    model this litellm version doesn't know) otherwise surfaces indirectly at
    call time — e.g. litellm blames a parameter it can't validate for the
    unknown model. Checking up front raises a clear :class:`ModelNotSupported`
    naming the model instead.

    Skipped when ``api_base`` is set: a custom endpoint (proxy / self-hosted,
    e.g. Ollama or vLLM) legitimately serves models litellm has no metadata for,
    so we trust the caller — mirroring the generation pre-flight, which likewise
    skips its key check when ``api_base`` is set.
    """
    if api_base:
        return
    try:
        litellm.get_model_info(model)
    except Exception as exc:
        raise ModelNotSupported(
            f"model '{model}' is not a recognized model id — check the spelling and "
            "provider prefix (e.g. 'gemini/gemini-2.5-pro'); it may be unavailable in "
            f"this litellm version ({short_error_message(exc)})"
        ) from exc


def model_max_output_tokens(model: str, api_base: str | None = None) -> int | None:
    """The model's documented output-token ceiling, or ``None`` when unknown.

    Read from litellm's model metadata so callers can ask for a model's real
    ceiling instead of hardcoding one number across a fleet whose limits
    differ (frontier Claude/Gemini allow 128K; Haiku 4.5 caps at 64K).
    ``None`` for a custom ``api_base`` (self-hosted/proxy — no metadata) or an
    id litellm doesn't know, in which case callers keep their own default.
    """
    if api_base:
        return None
    try:
        info = litellm.get_model_info(model)
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    raw = info.get("max_output_tokens") or info.get("max_tokens")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
        return None
    return int(raw)


def _clamp_output_tokens(requested: int, model: str, api_base: str | None) -> int:
    """Lower *requested* to the model's ceiling when we know it.

    Only ever lowers: asking a model for more output tokens than it supports
    is a provider 400, and the caller's intent ("give me as much room as this
    model allows") is served by the ceiling.
    """
    ceiling = model_max_output_tokens(model, api_base)
    return min(requested, ceiling) if ceiling is not None else requested


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
    _require_supported_model(config.model, config.api_base)
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
    }
    # Anthropic: never send temperature. Newer Claude models reject it
    # outright ("`temperature` is deprecated for this model") and older ones
    # reject anything but 1 when thinking is enabled — together the provider
    # default is the only always-safe value.
    # OpenAI's reasoning families accept only the default temperature (1) and
    # reject an explicit value outright:
    #
    #   400 "Unsupported value: 'temperature' does not support 0.0 with this
    #        model. Only the default (1) value is supported."
    #
    # `litellm.drop_params` is supposed to absorb this, and does for the models
    # it has metadata for — but it silently passes the parameter through for one
    # it does not recognise, and the request fails. That is what happened with
    # gpt-5.5: 42 of 114 transcription calls died on it and the cell produced
    # zero scoreable documents, which read as "the model cannot transcribe".
    #
    # Extraction asks for 0.0 to make the source-text → schema-slot mapping
    # deterministic. Where the provider forbids that, sampling at the default is
    # strictly better than not calling at all, so drop the parameter rather than
    # the request. Anthropic is excluded for a different reason: newer Claude
    # models reject temperature as deprecated.
    if (
        config.temperature is not None
        and not is_anthropic_model(config.model)
        and not is_openai_reasoning_model(config.model)
    ):
        kwargs["temperature"] = config.temperature
    # Both caps are clamped to the model's documented ceiling — a request for
    # more output than the model allows is a provider 400, and callers ask for
    # the largest useful value rather than tracking per-model limits.
    if config.max_tokens is not None:
        kwargs["max_tokens"] = _clamp_output_tokens(
            config.max_tokens, config.model, config.api_base
        )
    if config.max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = _clamp_output_tokens(
            config.max_completion_tokens, config.model, config.api_base
        )
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
    """Call litellm.completion with exponential-backoff retries for transient
    failures — both raised errors and *empty* responses.

    A successful call can still come back with an empty ``choices`` list (zero
    candidates, zero completion tokens): observed intermittently with Gemini via
    litellm, and transient — an immediate retry clears it. Left unhandled it
    surfaces downstream as ``response.choices[0]`` → ``IndexError`` and aborts
    the whole request (e.g. a document's extraction). So an empty response is
    retried like any other transient failure, and only raises
    :class:`EmptyModelResponse` once it persists across every attempt.
    """
    import litellm

    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with _quiet_stdout():
                response = litellm.completion(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            # Retry on transient network/server errors only. Rate limits count:
            # they say "not now", not "never", and the pipeline runs documents
            # concurrently, so hitting one is expected rather than exceptional.
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
                    "429",
                    "rate limit",
                    "rate_limit",
                    "ratelimit",
                    "too many requests",
                )
            )
            if not transient or attempt == max_retries - 1:
                raise
            last_exc = exc
            time.sleep(delay)
            delay *= 2
            continue
        # A successful call can still carry an empty choices list (transient
        # provider glitch); retry it rather than let choices[0] IndexError.
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if choices:
            return response
        if attempt == max_retries - 1:
            raise EmptyModelResponse(
                f"model returned no choices after {max_retries} attempts "
                f"(model={kwargs.get('model')!r})"
            )
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
                    cache_read_tokens=totals["cache_read_tokens"],
                    cache_creation_tokens=totals["cache_creation_tokens"],
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
    # Extended thinking and assistant prefill are mutually exclusive on
    # Anthropic: with reasoning enabled the API rejects a prefilled turn
    # ("This model does not support assistant message prefill"). That is the
    # other half of the truncation-path breakage — it cost ~13% of calls on one
    # model — so gate on the request shape, not just the provider.
    prefill = supports_assistant_prefill(config.model) and config.reasoning_effort is None
    acc = ""
    # One aggregated row for the whole continuation (all rounds summed).
    with _record_call(config) as totals:
        for _ in range(max_rounds):
            # On continuation rounds the accumulated text becomes an assistant
            # prefill; the provider resumes from its exact end (a length cut lands
            # mid-token, so there is no trailing whitespace to trip Anthropic).
            # Where prefill isn't honoured (OpenAI) the partial is still shown as
            # the assistant turn, but a final user turn has to ask for the
            # continuation explicitly — otherwise the model restarts the reply.
            messages = list(base)
            if acc:
                messages.append({"role": "assistant", "content": acc})
                if not prefill:
                    messages.append(
                        {
                            "role": "user",
                            "content": prompt(PromptKey.CONTINUE_TRUNCATED),
                        }
                    )
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
    cache: bool = False,
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

    ``cache=True`` enables provider prompt caching. For Anthropic-routed
    models it tags the system message with ``cache_control: {"type":
    "ephemeral"}`` (via :func:`_mark_system_message_cacheable`), so the
    tools + system prefix — identical across every call in a docset —
    replays at ~10% token cost within the 5-minute TTL. Callers that also
    carry a *stable* user-content block (e.g. the docset schema text that is
    byte-identical across every file) tag that block themselves before
    calling; per-request-volatile blocks (per-file PDF, per-page image/OCR
    words) are never tagged. litellm forwards ``cache_control`` on system and
    message content blocks for Anthropic even on tool-carrying requests, so
    caching composes with ``tools``. Caching changes only billing and
    latency — the request the model sees, and hence its output, are
    unaffected. For non-Anthropic providers caching is implicit and the flag
    is a no-op.
    """
    if cache and is_anthropic_model(config.model):
        messages = _mark_system_message_cacheable(messages)
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
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
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
                    cache_read_tokens=totals["cache_read_tokens"],
                    cache_creation_tokens=totals["cache_creation_tokens"],
                    duration_s=round(time.monotonic() - started, 3),
                    outcome=outcome,
                    context=config.context or {},
                    error=error_msg,
                ),
            )


__all__ = [
    "ANTHROPIC_MODEL_PATTERNS",
    "OPENAI_MODEL_PATTERNS",
    "PDF_NATIVE_MODEL_PATTERNS",
    "CallResult",
    "LLMConfig",
    "add_partial",
    "build_user_content",
    "call",
    "call_with_tools",
    "empty_usage_totals",
    "is_anthropic_model",
    "is_gemini_model",
    "is_openai_model",
    "is_openai_reasoning_model",
    "record_usage_for",
    "supports_assistant_prefill",
    "supports_native_pdf",
    "supports_vision",
]
