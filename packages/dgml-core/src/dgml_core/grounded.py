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

"""LLM-backed schema generation and grounded value extraction.

This module powers two CLI surfaces:

- ``dgml extraction generate-schema`` — the configured ``schema_model`` is
  given sample PDFs from the docset and asked to propose a *typed field tree*
  describing the structured information to extract, choosing an ``xsd``
  datatype (or enum token set) for each leaf. That tree is rendered straight
  to the at-rest RELAX NG Compact schema
  (:func:`dgml_core.extraction_schema.field_tree_to_rnc`) — no JSON Schema
  intermediate. Downstream extraction attributes every value back to a region
  of the source via the ``extracted_value`` form
  (``{text, value?, locations: [{page_number, bounding_box}]}``).

- ``dgml extraction extract`` (and the auto-extract hook on
  ``dgml docset add-file``) — the configured ``values_model`` is given a PDF
  plus the docset's schema, and asked to produce values matching that schema.
  To keep output attributable to the source, the model is granted a
  ``get_page_words`` tool that returns OCR-extracted words and their bounding
  boxes from the workspace. The model produces final results via a
  ``submit_values`` tool call.

Coordinate space contract:
- Bounding boxes are integer image pixels ``[left, top, right, bottom]``
  (top-left origin) at 300 dpi, relative to ``page_images/page_N.png`` —
  one convention end-to-end. This is what ``page_text/page_N.json``
  stores, what the ``extracted_value`` form uses in schemas and values,
  and what the ``get_page_words`` tool hands the model. The model reads
  pixel word boxes and returns pixel bboxes, so every hop speaks one
  language with no conversion.
- The one exception is phase 3 on a page with no OCR words (a scan added
  in digital mode): with no pixel words to measure against, the model is
  asked for boxes on a 0-1000 grid over the page image, and
  :func:`_to_page_pixels` scales them to pixels before they are stored.
  What is stored is always pixels.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import layout, llm
from .config import load_merged_config
from .dg_refs import METHOD_DEFAULTED, METHOD_INFERRED, METHODS
from .docsets import DocSetStore
from .errors import (
    CONVERT_TO_PDF_OPERATION,
    AuthError,
    ConversionFailed,
    FileNotFound,
    GroundedConfigInvalid,
    GroundedConfigMissing,
    SchemaGenerationFailed,
    SchemaInvalid,
    ValuesExtractionFailed,
    load_recorded_errors,
    now_iso,
)
from .extraction_schema import (
    FIELD_DATATYPES,
    Tag,
    Vocabulary,
    check_invariant_paths,
    field_tree_to_rnc,
    parse_rnc,
    rnc_to_json_schema,
)
from .extraction_xml import (
    check_derivations,
    check_invariants,
    count_dropped_inputs,
    count_unnormalized_enum_values,
    drop_unboxed_values,
    embed_extraction_into,
    has_document_tree,
    standalone_extraction_doc,
    verify_input_citations,
)
from .files import FileStore
from .llm import (
    CallResult,
    LLMConfig,
    LLMSteps,
    _mark_document_cacheable,
    _no_text_detail,
    is_anthropic_model,
    is_request_rejection,
    is_request_too_large,
    model_max_output_tokens,
    steps_with_tools,
)
from .matching import (
    UnmatchedItem,
    get_at_path,
    locate_pageless_values,
    parse_path,
    path_to_str,
    run_phase2_matching,
    verify_boxes,
    walk_all_leaves,
    walk_input_leaves,
    walk_unprinted_leaves,
)
from .models_config import ConfigSection, Tier, resolve_tiered_model
from .ocr import _image_dimensions as _png_dimensions
from .pages import PdfConfig, load_pdf_config, pdf_page_count_bytes, slice_pages
from .prompts import PromptKey
from .prompts import get as prompt
from .storage import Workspace
from .textmatch import Word as TextWord
from .textmatch import fuzzy_norm, line_boxes
from .toon import encode_phase3_words
from .usage import (
    OPERATION_EXTRACT_VALUES,
    OPERATION_SCHEMA_GENERATE,
    OUTCOME_ERROR,
    OUTCOME_OK,
    TIERS_KEY,
    UsageEvent,
    add_partial,
    public_totals,
    record_usage,
    scope_events,
    with_tier,
)

logger = logging.getLogger(__name__)

# ---- Constants ------------------------------------------------------------

DEFAULT_MAX_TOOL_ITERS = 20

# Per-call output token cap — deliberately set to the largest ceiling any
# current frontier model offers (128K on Claude Opus/Sonnet 5, Sonnet 4.6,
# Gemini 2.5 Pro). `llm._build_completion_kwargs` clamps this down to each
# model's documented limit, so a model with a lower ceiling (Haiku 4.5 at
# 64K) gets its own maximum rather than a 400. Asking for the ceiling matters
# for value extraction: a long itemized document can genuinely need
# >65K output tokens, and clipping surfaces as truncated tool-call arguments.
_DEFAULT_MAX_COMPLETION_TOKENS = 128000

# Temperature for value extraction specifically. Schema generation
# benefits from a little creativity (it's choosing field names);
# value extraction is a faithful mapping of source text → schema
# slots, so we want deterministic, systematic behavior — go through
# the table row by row, not "make a stylistic call about how many
# entries to include". 0 forces greedy decoding.
_DEFAULT_VALUES_TEMPERATURE = 0.0

# Reasoning / "thinking" budget for the models. litellm normalizes
# this across providers — for Gemini it maps to
# thinkingConfig.thinkingLevel, for Anthropic to extended-thinking
# budget, for OpenAI to reasoning_effort. We default to "high"
# because the multi-page array completeness task in particular
# benefits from the model actively reasoning over which pages it
# has and hasn't covered, instead of pattern-matching its way to a
# small answer. Without an explicit value, providers default low
# (Gemini's REST API defaults are notably lower than AI Studio's
# UI defaults, which is the trap we just hit).
_DEFAULT_REASONING_EFFORT = "high"

# Value extraction is the exception: it runs at "medium". Thinking tokens bill as
# output tokens and output is the bulk of extraction spend, which makes the
# reasoning budget the largest cost and latency dial on this path. Measured
# against reference labels, "medium" cost about a quarter less and finished about
# twice as fast, with extraction quality equivalent to "high" and no document
# losing entries from a repeating field — including the long, table-heavy
# documents the "high" default was originally chosen to protect.
#
# This is deliberately a separate constant rather than a change to
# _DEFAULT_REASONING_EFFORT. Schema generation and location grounding force
# ``tool_choice``, so the LLM wrapper drops ``reasoning_effort`` for
# Anthropic-routed models and the value would be inert there on Claude — but it
# is NOT dropped for other providers. Location grounding on pages with OCR words
# takes ``grounded.locations_reasoning_effort`` (_LOCATIONS_REASONING_EFFORT
# unless the workspace sets it); see _PHASE3_GRID_REASONING_EFFORT for pages
# without them.
_VALUES_REASONING_EFFORT = "medium"

# What ``grounded.values_reasoning_effort`` (``--values-effort``) and
# ``grounded.locations_reasoning_effort`` (``--locations-effort``) accept: the
# effort names litellm takes, sent as given, plus ``"default"``, which sends no
# ``reasoning_effort`` at all and leaves the budget to the provider. ``"none"``
# is litellm's own value and is sent (it asks the provider to turn thinking
# off), which is why omitting the field needs a word of its own. The names
# are checked here, not against what the model supports: a budget the
# provider refuses fails at the call that sends it.
REASONING_EFFORT_CHOICES = ("none", "minimal", "low", "medium", "high", "xhigh")
REASONING_EFFORT_PROVIDER_DEFAULT = "default"
# The names these had when only the values effort was configurable.
VALUES_REASONING_EFFORT_CHOICES = REASONING_EFFORT_CHOICES
VALUES_REASONING_EFFORT_PROVIDER_DEFAULT = REASONING_EFFORT_PROVIDER_DEFAULT

# Location grounding on a page with OCR words (phase 3) runs at the general
# default unless ``grounded.locations_reasoning_effort`` says otherwise. This
# is the one owner of that default: GroundedConfig carries it and the phase-3
# functions take the configured value as a required argument.
_LOCATIONS_REASONING_EFFORT = _DEFAULT_REASONING_EFFORT

# Location grounding on a page with no OCR words (the 0-1000 grid path) runs at
# "medium". There the model only reads a position off the image; at "high" it
# spent most of its output reasoning without placing boxes any better. Measured
# on Gemini over 30 records-retrieval documents, most of them scans, 3 repeats:
# "medium" placed as many boxes on scanned pages as "high", for about half the
# grounding cost; "low" was cheaper again but started to miss boxes. Pages with
# words take ``grounded.locations_reasoning_effort`` (default
# _LOCATIONS_REASONING_EFFORT), which that measurement did not cover.
_PHASE3_GRID_REASONING_EFFORT = "medium"


# HTTP timeout for a single litellm call. The default in litellm is
# in the few-minutes range and isn't long enough for high-reasoning
# extractions on a multi-page document (observed real-world timeouts
# at ~13min on a 7-page ledger with reasoning=high). 30 min gives
# the model room to think through a long table without us cutting
# the connection.
_DEFAULT_TIMEOUT_SECONDS = 1800

# Default number of files the schema-generation step samples from a
# docset when the caller doesn't specify any. Three is a deliberate
# middle ground — enough variation for the model to spot what generalizes
# across instances of the document kind, few enough to keep cost bounded.
DEFAULT_SCHEMA_SAMPLE_SIZE = 3

logger = logging.getLogger(__name__)

_TOOL_GET_PAGE_WORDS = "get_page_words"
_TOOL_SUBMIT_SCHEMA = "submit_schema"
_TOOL_SUBMIT_VALUES = "submit_values"
_TOOL_APPEND_ENTRIES = "append_entries"
_TOOL_SUBMIT_LOCATIONS = "submit_locations"

# Phase-1 turns per attempt that may end with plain text (no tool call)
# before the attempt fails. Reachable only where phase 1 runs on auto tool
# choice (see _phase1_attempt_steps): on Anthropic, on an endpoint that rejected
# "required", and where litellm's drop_params removed it. There a model
# sometimes answers in prose, asks a question, or writes the submission as
# JSON text instead of calling the tool. Each such turn is answered with a
# reminder and retried, still on auto. Retries do not spend max_tool_iters.
_MAX_NO_TOOL_CALL_RETRIES = 2

# How much of a text-only reply the error message quotes.
_NO_TOOL_CALL_REPLY_PREVIEW_CHARS = 500

# (model, api_base, reasoning_effort) endpoints shown to refuse
# tool_choice="required": a request sent with it was rejected and the same
# request on auto then succeeded. The effort is part of the key because a
# Claude model behind an unrecognised alias rejects "required" only together
# with thinking. Later
# attempts against them start on auto instead of re-sending a doomed request,
# and the warning is emitted once, when an endpoint is added.
_REQUIRED_TOOL_CHOICE_REJECTED: set[tuple[str, str | None, str | None]] = set()


# ---- Config ---------------------------------------------------------------


@dataclass(frozen=True)
class GroundedConfig:
    """Parsed ``grounded`` section of the workspace config.

    Two models are configured separately because schema generation
    (creative, schema-shaping) and value extraction (faithful,
    grounding-heavy) have different strengths across providers.

    API key resolution per side, in order of precedence:
    1. ``*_api_key``       — literal key in the config file. Allowed
                             but only safe in workspaces that aren't
                             shared or checked in.
    2. ``*_api_key_env``   — name of an env var holding the key.
    3. Neither             — litellm falls back to its own per-provider
                             env-var conventions (``ANTHROPIC_API_KEY``,
                             ``GEMINI_API_KEY``, etc.).

    Setting both ``*_api_key`` and ``*_api_key_env`` for the same side
    is a config error.
    """

    schema_model: str
    values_model: str
    schema_api_key: str | None = None
    values_api_key: str | None = None
    schema_api_key_env: str | None = None
    values_api_key_env: str | None = None
    schema_api_base: str | None = None
    values_api_base: str | None = None
    max_tool_iters: int = DEFAULT_MAX_TOOL_ITERS
    # Reasoning budget of the value-extraction call (phase 1). ``None`` sends no
    # ``reasoning_effort``.
    values_reasoning_effort: str | None = _VALUES_REASONING_EFFORT
    # Reasoning budget of location grounding (phase 3) on a page with OCR
    # words. ``None`` sends no ``reasoning_effort``. Anthropic models drop it
    # either way (the call forces its tool); every other provider sends it,
    # and at the default it is the larger half of a Gemini values model's
    # latency. A page on the no-words grid path keeps _PHASE3_GRID_REASONING_EFFORT.
    # Checked by name only: a budget the model refuses fails in phase 3, after
    # the values call has been paid for, on the first page that needs grounding.
    locations_reasoning_effort: str | None = _LOCATIONS_REASONING_EFFORT


def parse_reasoning_effort(raw: Any, *, source: str) -> str | None:
    """Validate a reasoning effort (values or locations) from the config file
    or the CLI.

    Returns the effort to send, or ``None`` for ``"default"`` (send nothing).
    ``source`` names where the value came from, for the error message. The
    name is checked, not the model's support for it.
    """
    if raw == REASONING_EFFORT_PROVIDER_DEFAULT:
        return None
    if isinstance(raw, str) and raw in REASONING_EFFORT_CHOICES:
        return raw
    accepted = ", ".join(
        repr(v) for v in (*REASONING_EFFORT_CHOICES, REASONING_EFFORT_PROVIDER_DEFAULT)
    )
    raise GroundedConfigInvalid(f"{source} must be one of {accepted} if set (got {raw!r})")


# The parser's name when only the values effort was configurable.
parse_values_reasoning_effort = parse_reasoning_effort


def load_grounded_config(
    workspace: Workspace,
    *,
    schema_model: str | None = None,
    values_model: str | None = None,
) -> GroundedConfig:
    """Resolve the two grounded models (schema generation → ``expert`` tier,
    value extraction → ``advanced`` tier) and their credentials from the merged
    config's ``[grounded]`` section and ``[models]`` tiers. A model set on the
    section overrides its tier.

    ``schema_model`` / ``values_model`` are per-call overrides and resolve
    exactly as if written in ``[grounded]``: the section's own ``schema_*`` /
    ``values_*`` credentials still apply, and nothing a tier carries does.
    """
    merged = load_merged_config(workspace)
    overrides = {
        key: model
        for key, model in (("schema_model", schema_model), ("values_model", values_model))
        if model is not None
    }
    if overrides:
        section = merged.get(ConfigSection.GROUNDED)
        base: dict[str, Any] = section if isinstance(section, dict) else {}
        merged = {**merged, ConfigSection.GROUNDED: {**base, **overrides}}
    schema = resolve_tiered_model(
        merged,
        section_name=ConfigSection.GROUNDED,
        tier=Tier.EXPERT,
        invalid=GroundedConfigInvalid,
        missing=GroundedConfigMissing,
        prefix="schema_",
    )
    values = resolve_tiered_model(
        merged,
        section_name=ConfigSection.GROUNDED,
        tier=Tier.ADVANCED,
        invalid=GroundedConfigInvalid,
        missing=GroundedConfigMissing,
        prefix="values_",
    )

    section = merged.get(ConfigSection.GROUNDED)
    sec: dict[str, Any] = section if isinstance(section, dict) else {}
    max_tool_iters_raw = sec.get("max_tool_iters", DEFAULT_MAX_TOOL_ITERS)
    if (
        not isinstance(max_tool_iters_raw, int)
        or isinstance(max_tool_iters_raw, bool)
        or max_tool_iters_raw < 1
    ):
        raise GroundedConfigInvalid("'grounded.max_tool_iters' must be a positive integer if set")
    values_reasoning_effort = (
        parse_reasoning_effort(
            sec["values_reasoning_effort"], source="'grounded.values_reasoning_effort'"
        )
        if "values_reasoning_effort" in sec
        else _VALUES_REASONING_EFFORT
    )
    locations_reasoning_effort = (
        parse_reasoning_effort(
            sec["locations_reasoning_effort"], source="'grounded.locations_reasoning_effort'"
        )
        if "locations_reasoning_effort" in sec
        else _LOCATIONS_REASONING_EFFORT
    )

    return GroundedConfig(
        schema_model=schema.model,
        values_model=values.model,
        schema_api_key=schema.api_key,
        values_api_key=values.api_key,
        schema_api_key_env=schema.api_key_env,
        values_api_key_env=values.api_key_env,
        schema_api_base=schema.api_base,
        values_api_base=values.api_base,
        max_tool_iters=max_tool_iters_raw,
        values_reasoning_effort=values_reasoning_effort,
        locations_reasoning_effort=locations_reasoning_effort,
    )


# ---- Tool: get_page_words --------------------------------------------------


def get_page_words(
    workspace: Workspace,
    file_id: str,
    page: int,
    start_idx: int | None = None,
    end_idx: int | None = None,
) -> dict[str, Any]:
    """Return OCR words and bounding boxes for a page in image-pixel space.

    This is the body of the ``get_page_words`` tool the extraction LLM
    is allowed to call. It is also useful directly from Python.

    Words are returned with their original OCR index so callers (the
    model) can address them stably even when subsetting. Boxes are the
    integer ``[left, top, right, bottom]`` image pixels ``page_text``
    already stores — the single coordinate vocabulary the model reads
    and writes.
    """
    if page < 1:
        raise ValueError("page must be 1-indexed (≥ 1)")
    payload = workspace.read_page_text(file_id, page)
    if payload is None:
        raise FileNotFound(
            f"no page_text for file '{file_id}' page {page}; "
            "was the file added with --text-mode digital or ocr?"
        )
    words: list[dict[str, Any]] = payload.get("words", [])

    s = 0 if start_idx is None else max(0, start_idx)
    e = len(words) if end_idx is None else min(len(words), end_idx)
    out_words = []
    for i in range(s, e):
        w = words[i]
        left, top, right, bottom = w["l"]
        # Bboxes are integer image pixels [left, top, right, bottom] — the
        # same shape page_text stores and the model returns. No conversion.
        bbox = [round(left), round(top), round(right), round(bottom)]
        out_words.append(
            {
                "idx": i,
                "text": w["t"],
                "location": {"page_number": page, "bounding_box": bbox},
            }
        )
    return {
        "page": page,
        "total_words": len(words),
        "words": out_words,
    }


# ---- PDF input helpers -----------------------------------------------------


def _pdf_bytes(workspace: Workspace, file_id: str) -> bytes:
    """Return the bytes of the single ``*.pdf`` stored for ``file_id``.

    The PDF is found by the name ``file add`` stored it under
    (:func:`layout.file_pdf_key`, from the record's ``original_filename``),
    not by matching a suffix: a source named ``INVOICE.PDF`` is stored under
    that name, so a case-sensitive ``.pdf`` match missed it. There is no
    fallback scan: a filename is not unique in the workspace, so a PDF found
    anywhere else under the prefix could belong to another source.

    A convertible source whose conversion failed at ``file add`` has a
    record but no PDF, and the converter's error is among the file's
    recorded errors: such a file raises :class:`ConversionFailed` repeating
    that error. A file with no PDF and no recorded conversion error raises
    :class:`FileNotFound` as before.
    """
    pdf_key = layout.file_pdf_key(file_id, FileStore(workspace).get(file_id).original_filename)
    if workspace.blobs.blob_exists(pdf_key):
        return workspace.blobs.get_blob(pdf_key)
    try:
        failed = [
            err.message.strip()
            for err in load_recorded_errors(workspace, file_id)
            if err.operation == CONVERT_TO_PDF_OPERATION
            and isinstance(err.message, str)
            and err.message.strip()
        ]
    except (TypeError, KeyError, ValueError, AttributeError):
        # The lookup is a diagnostic: a malformed errors document (a
        # wrong shape, a missing field) must not turn the missing PDF
        # into an internal error; a record without a usable message is
        # skipped the same way.
        failed = []
    if failed:
        raise ConversionFailed(
            f"file '{file_id}' has no source PDF: converting it failed: {failed[-1]}"
        )
    raise FileNotFound(f"file '{file_id}' has no source PDF")


def _pdf_content_block(pdf_bytes: bytes) -> dict[str, Any]:
    """An OpenAI-style ``file`` content block carrying the PDF inline.

    litellm normalizes this across providers: Anthropic gets a
    ``document`` block, Gemini gets ``inline_data``. The base64 string is
    the same in both cases.
    """
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    return {
        "type": "file",
        "file": {"file_data": f"data:application/pdf;base64,{b64}"},
    }


# ---- Schema generation -----------------------------------------------------


def generate_schema(
    workspace: Workspace,
    file_ids: list[str],
    *,
    config: GroundedConfig,
    docset_name: str,
    debug: bool = False,
) -> str:
    """Ask the configured LLM to propose an extraction schema from one or
    more PDFs of the same kind, returning the at-rest RELAX NG Compact form.

    The model submits a *typed field tree* — each leaf carrying an ``xsd``
    datatype it chose (``date``, ``decimal``, ``integer``, …) — which we render
    straight to RNC via :func:`field_tree_to_rnc`. There is no grounded_field
    JSON Schema intermediate: types are native and the returned RNC is the
    canonical on-disk schema. *docset_name* (with ``workspace.organization``)
    fixes the docset namespace so it matches the generated docset's.

    Sending multiple examples lets the model see what's stable across
    instances vs. what's per-document — the schema it returns is meant
    to fit *all* of them, not just one. Callers (the CLI) decide how
    to pick the sample.

    If the provider refuses the request as too large, every sample stays in
    it but the largest are cut to their leading pages (a fair share of half
    what was just sent), the model is told which samples are partial, and the
    call is retried. A sample whose first page alone exceeds its share is
    left out with a warning. A request the provider accepts is unchanged.

    Raises :class:`SchemaGenerationFailed` on any non-config failure
    (no files, missing PDF, malformed LLM response, network error).
    """
    llm_config, steps = generate_schema_steps(workspace, file_ids, config=config, debug=debug)
    try:
        result = llm.drive(steps, llm_config)
    except SchemaGenerationFailed:
        raise
    except Exception as exc:
        raise _schema_call_failed(exc) from exc
    return schema_rnc_from_result(result, workspace=workspace, docset_name=docset_name)


def generate_schema_steps(
    workspace: Workspace,
    file_ids: list[str],
    *,
    config: GroundedConfig,
    debug: bool = False,
) -> tuple[LLMConfig, LLMSteps[CallResult]]:
    """The schema-generation request as ``(config, steps)``.

    Everything that can fail before a request exists — no files, an unreadable
    PDF, an unset key env var — fails HERE, eagerly, so no driver ever sends
    (or pays for) a request that could not succeed. *steps* yields the forced
    ``submit_schema`` call (again, trimmed, after a too-large refusal) and
    returns its
    :class:`~dgml_core.llm.CallResult`; turn that into RNC with
    :func:`schema_rnc_from_result`. Drive it with :func:`dgml_core.llm.drive`
    (which writes the one usage row from *config*)."""
    if not file_ids:
        raise SchemaGenerationFailed(
            "schema generation requires at least one example file_id; got an empty list"
        )

    # Read every PDF up front so a permission/missing-file error fails
    # the call before we burn an LLM API request.
    originals = [(fid, _pdf_bytes(workspace, fid)) for fid in file_ids]

    api_key = _resolve_api_key(config.schema_api_key, config.schema_api_key_env)
    tools = [_submit_schema_tool()]
    # max_tokens=None so the wrapper doesn't add the max_tokens alias alongside
    # max_completion_tokens. reasoning_effort is set unconditionally; the
    # wrapper drops it for Anthropic + forced tool_choice.
    # Single call → the driver records its usage row (gated on --debug) from the
    # context carried on the config.
    llm_config = LLMConfig(
        model=config.schema_model,
        api_key=api_key,
        api_base=config.schema_api_base,
        max_tokens=None,
        max_completion_tokens=_DEFAULT_MAX_COMPLETION_TOKENS,
        timeout=_DEFAULT_TIMEOUT_SECONDS,
        reasoning_effort=_DEFAULT_REASONING_EFFORT,
        workspace=workspace,
        debug=debug,
        operation=OPERATION_SCHEMA_GENERATE,
        context={"from_file_ids": list(file_ids)},
    )
    steps = _schema_request_steps(
        workspace, llm_config, originals=originals, tools=tools, model=config.schema_model
    )
    return llm_config, steps


def _schema_request_steps(
    workspace: Workspace,
    llm_config: LLMConfig,
    *,
    originals: list[tuple[str, bytes]],
    tools: list[dict[str, Any]],
    model: str,
) -> LLMSteps[CallResult]:
    """The forced ``submit_schema`` request, resent with the samples trimmed
    while the provider refuses it as too large (see :func:`generate_schema`).

    A failed request other than a too-large refusal raises its own exception,
    which the usage row records and :func:`generate_schema` wraps as
    ``"schema generation call failed: <Type>: <message>"``; a failure to trim
    raises :class:`SchemaGenerationFailed` directly."""
    samples = [_SchemaSample(fid, pdf) for fid, pdf in originals]
    pdf_config: PdfConfig | None = None
    while True:
        try:
            result = yield from steps_with_tools(
                llm_config,
                messages=_schema_messages(samples),
                tools=tools,
                tool_choice={"type": "function", "function": {"name": _TOOL_SUBMIT_SCHEMA}},
                # Deliberately NOT cached: the cacheable prefix is tools + system,
                # and this system prompt is well short of the provider's minimum
                # cacheable prefix, so a breakpoint would create no entry. Schema
                # generation also runs once per docset, so there is no reuse to
                # capture even if it did.
            )
            break
        except Exception as exc:
            if not is_request_too_large(exc):
                # Raised as is, so the usage row records the provider's error;
                # generate_schema wraps it.
                raise
            if pdf_config is None:
                try:
                    pdf_config = load_pdf_config(workspace)
                except Exception as config_exc:
                    raise SchemaGenerationFailed(
                        f"schema generation call failed: {type(exc).__name__}: {exc} "
                        f"The samples could not be trimmed: {config_exc}"
                    ) from exc
            # Halve what the refused request carried; samples are cut from the
            # originals each round, so the budget only ever shrinks.
            budget = sum(_b64_len(len(sample.pdf)) for sample in samples) // 2
            samples = _fit_samples(workspace, originals, budget, pdf_config)
            if not samples:
                raise SchemaGenerationFailed(
                    f"the planning documents are too large for one schema request to "
                    f"{model!r}, even trimmed to their first page: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
    _warn_trimmed_samples([fid for fid, _pdf in originals], samples)
    return result


def _schema_call_failed(exc: BaseException) -> SchemaGenerationFailed:
    """The error a failed schema request surfaces as."""
    return SchemaGenerationFailed(f"schema generation call failed: {type(exc).__name__}: {exc}")


def schema_rnc_from_result(result: CallResult, *, workspace: Workspace, docset_name: str) -> str:
    """Render the ``submit_schema`` reply in *result* to at-rest RNC.

    Parsing happens after the request's usage row is written (a reply that
    parses badly was still a successful, billed call), exactly as the inline
    call site did. Raises :class:`SchemaGenerationFailed` on a missing or wrong
    tool call, a non-list field tree, or an invalid one."""
    fields = _parse_submit_call(result.response, expected_tool=_TOOL_SUBMIT_SCHEMA, field="fields")

    if not isinstance(fields, list):
        raise SchemaGenerationFailed("LLM returned a non-list 'fields' — expected a field tree")
    try:
        return field_tree_to_rnc(fields, workspace=workspace.organization, docset_name=docset_name)
    except SchemaInvalid as exc:
        raise SchemaGenerationFailed(f"LLM returned an invalid field tree: {exc}") from exc


@dataclass(frozen=True)
class _SchemaSample:
    """A planning document as sent to schema generation: the whole PDF, or
    (``pages_sent`` set) its first ``pages_sent`` of ``total_pages`` pages."""

    file_id: str
    pdf: bytes
    pages_sent: int | None = None
    total_pages: int | None = None


def _schema_messages(samples: list[_SchemaSample]) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": _schema_user_prompt(len(samples))},
    ]
    excerpts = [
        f"- Sample {i} (pages 1-{s.pages_sent} of {s.total_pages})"
        for i, s in enumerate(samples, start=1)
        if s.pages_sent is not None
    ]
    if excerpts:
        # Only on a retry after the provider refused the request as too large.
        user_content.append(
            {
                "type": "text",
                "text": prompt(PromptKey.SCHEMA_USER_EXCERPTS).format(excerpts="\n".join(excerpts)),
            }
        )
    user_content.extend(_pdf_content_block(s.pdf) for s in samples)
    return [
        {"role": "system", "content": prompt(PromptKey.SCHEMA_SYSTEM)},
        {"role": "user", "content": user_content},
    ]


def _b64_len(n: int) -> int:
    """Length of the base64 encoding of *n* bytes."""
    return 4 * ((n + 2) // 3)


def _fit_samples(
    workspace: Workspace,
    originals: list[tuple[str, bytes]],
    budget: int,
    pdf_config: PdfConfig,
) -> list[_SchemaSample]:
    """The planning documents cut to fit *budget* base64 bytes between them.

    The budget is shared max-min fairly: a document within an equal share is
    sent whole, and the larger ones split what is left and are sent as the
    leading pages that fit their share (sliced with the workspace's PDF engine
    and measured, since pages differ in size). A document whose first page
    alone exceeds its share is left out. Order is kept.
    """
    sizes = {fid: _b64_len(len(pdf)) for fid, pdf in originals}
    shares: dict[str, int] = {}
    remaining, left = budget, len(originals)
    for fid, _pdf in sorted(originals, key=lambda item: sizes[item[0]]):
        share = remaining // left
        shares[fid] = min(sizes[fid], share)
        remaining -= shares[fid]
        left -= 1

    fitted: list[_SchemaSample] = []
    for fid, pdf in originals:
        share = shares[fid]
        if sizes[fid] <= share:
            fitted.append(_SchemaSample(fid, pdf))
            continue
        sample = _leading_pages(workspace, fid, pdf, share, pdf_config)
        if sample is not None:
            fitted.append(sample)
    return fitted


def _leading_pages(
    workspace: Workspace, file_id: str, pdf: bytes, share: int, pdf_config: PdfConfig
) -> _SchemaSample | None:
    """The longest leading page range of *pdf* found to fit *share* base64
    bytes, or ``None`` when even page 1 does not."""
    count = FileStore(workspace).get(file_id).page_count
    if not isinstance(count, int) or count <= 0:
        try:
            count = pdf_page_count_bytes(pdf)
        except Exception:
            return None  # unreadable, so it cannot be cut; it is already too big whole
    if count <= 1:
        return None
    whole = _b64_len(len(pdf))
    pages = max(1, min(count - 1, count * share // whole))
    while True:
        try:
            sliced = slice_pages(pdf, range(1, pages + 1), config=pdf_config, total_pages=count)
        except Exception as exc:
            raise SchemaGenerationFailed(
                f"could not cut file '{file_id}' to its first {pages} pages to fit the "
                f"schema request: {type(exc).__name__}: {exc}"
            ) from exc
        size = _b64_len(len(sliced))
        if size <= share:
            return _SchemaSample(file_id, sliced, pages_sent=pages, total_pages=count)
        if pages == 1:
            return None
        pages = max(1, min(pages - 1, pages * share // size))


def _warn_trimmed_samples(file_ids: list[str], samples: list[_SchemaSample]) -> None:
    """Warn about the samples a too-large request cost: those sent partial and
    those left out. Silent when every sample went whole."""
    sent = {s.file_id for s in samples}
    for fid in file_ids:
        if fid not in sent:
            logger.warning(
                "schema generation: left out file %s; even its first page did not fit "
                "in the request",
                fid,
            )
    partial = [
        f"{s.file_id} (pages 1-{s.pages_sent} of {s.total_pages})" for s in samples if s.pages_sent
    ]
    if partial:
        logger.warning(
            "schema generation: the samples were too large for one request; sent the "
            "leading pages of %s",
            ", ".join(partial),
        )


def _schema_user_prompt(n_files: int) -> str:
    if n_files == 1:
        intro = prompt(PromptKey.SCHEMA_USER_INTRO_SINGLE)
    else:
        intro = prompt(PromptKey.SCHEMA_USER_INTRO_MULTI).format(n_files=n_files)
    return intro + "\n\n" + prompt(PromptKey.SCHEMA_USER_BODY)


#: Max nesting depth of the *declared* field-tree schema. Only constrains
#: constrain-decoders (Gemini); the parser handles arbitrary depth, so other
#: providers may return deeper. Bounded because Gemini rejects the deeper
#: (~27 KB) form with "too many states for serving"; depth 3 (~13 KB) is
#: accepted and covers the docset schemas seen so far.
_SCHEMA_TREE_MAX_DEPTH = 3

_NODE_KINDS = ("field", "container", "collection")


def _field_node_schema(depth: int, *, allow_item: bool = True) -> dict[str, Any]:
    """JSON Schema for one field-tree node, with keys ``_field_node_to_tag`` reads.

    Declared explicitly rather than as a bare ``{"type": "object"}``: Gemini
    constrain-decodes tool arguments against the schema, so an empty object makes
    it return empty ``{}`` nodes and generation fails with "missing 'name'"
    (issue #73). Nesting is inlined, not ``$ref``'d (adapters resolve $ref
    inconsistently). The deepest level narrows ``kind`` to ``field`` (no child
    slot left); ``item`` does not consume a level (the parser flattens it); and
    ``datatype`` is enum-narrowed to spellings ``_normalize_datatype`` already
    canonicalizes.
    """
    kinds = list(_NODE_KINDS) if depth > 0 else ["field"]
    props: dict[str, Any] = {
        "name": {"type": "string", "description": "Field name, in the document's own wording."},
        "kind": {"type": "string", "enum": kinds},
        "datatype": {"type": "string", "enum": ["text", *sorted(FIELD_DATATYPES)]},
        "description": {"type": "string"},
        "example": {"type": "string"},
        "prompt": {"type": "string"},
    }
    if depth > 0:
        props["fields"] = {
            "type": "array",
            "description": "Children of a 'container' (or a 'collection' item's fields).",
            "items": _field_node_schema(depth - 1),
        }
        if allow_item:
            props["item"] = {
                **_field_node_schema(depth, allow_item=False),
                "description": "A 'collection''s repeated item, described explicitly.",
            }
    return {
        "type": "object",
        "properties": props,
        "required": ["name", "kind"],
        "additionalProperties": False,
    }


def _submit_schema_tool() -> dict[str, Any]:
    # The tree structure is also documented in the prompt and enforced by the
    # deterministic Python parser (``field_tree_to_vocabulary``); the tool schema
    # declares it too so providers that constrain-decode tool arguments emit a
    # populated tree rather than empty objects.
    return {
        "type": "function",
        "function": {
            "name": _TOOL_SUBMIT_SCHEMA,
            "description": (
                "Submit the docset's extraction schema as a typed field tree "
                "(the list of top-level fields)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "description": (
                            "Top-level fields. Each node is an object "
                            "{name, kind, datatype?, description?, example?, prompt?, "
                            "fields?, item?}. kind is 'field' (a grounded leaf; give "
                            "its datatype), 'container' (groups child 'fields'), or "
                            "'collection' (a repeated entity — describe the item via "
                            "'item' or its 'fields'). datatype is one of "
                            "'text', 'date', 'dateTime', 'decimal', 'integer', "
                            "'boolean', 'gYear', 'time', 'anyURI'."
                        ),
                        "items": _field_node_schema(_SCHEMA_TREE_MAX_DEPTH),
                    }
                },
                "required": ["fields"],
                # Same rationale as _expand_refs: closes the hole where Gemini
                # accepts a fabricated sibling property next to the required one.
                "additionalProperties": False,
            },
        },
    }


# ---- Value extraction ------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of an extraction call.

    ``values`` is the structured values tree (the in-process form). It is
    persisted as a ``dg:extraction`` element inside the file's core
    ``<stem>.dgml.xml`` at ``xml_path`` (spec §13) — no separate file.
    ``mode`` is ``"full-extraction"`` when the file already had a generated
    document tree (extraction added as a sibling) or ``"extraction"`` when the
    core file was created with only the ``dg:extraction`` element.
    ``tool_calls`` is the count of ``get_page_words`` grounding lookups the
    model performed before finalizing — a rough proxy for how much grounding it
    leaned on. ``model`` is the values model that produced it.
    """

    values: dict[str, Any]
    tool_calls: int
    xml_key: str
    mode: str
    model: str


def extract_values(
    workspace: Workspace,
    docset_id: str,
    file_id: str,
    *,
    config: GroundedConfig,
    write_stats: bool = True,
    debug: bool = False,
) -> ExtractionResult:
    """Ask the configured LLM to extract values from a file against a docset's schema.

    Runs in three phases:

    1. **Values + pages (LLM).** The schema is shown with
       ``bounding_box`` stripped from ``grounded_field.locations``. The
       model returns each value as ``{text, locations: [{page_number}]}``.
       Giving the model only "what's the value and which page is it on?"
       lets it cover multi-page arrays reliably — earlier single-pass
       extractions under-counted rows when the same call also had to
       compute bboxes.

    2. **Match in code.** :func:`dgml.matching.run_phase2_matching` walks
       the phase-1 values, looks up each text on its page in
       ``page_text/page_N.json``, and commits unambiguous OCR-word spans
       as bboxes. Free, fast, deterministic. Anything the code can't
       resolve (no match, or ambiguous) falls through.

    3. **Locate the leftovers (LLM, page-by-page).** For each page with
       unresolved items, send the page image + OCR words + the list of
       unresolved ids to the model and ask for an ``id → bbox`` mapping.
       Code patches those bboxes back into the values tree — the model
       never has to echo the whole structure, so phase-3 prompts and
       outputs stay small.

    Cost telemetry (``--debug`` only, via ``debug``): a single
    ``extract_values`` row in ``usage.jsonl`` sums phase 1 + 3 (phase 2 is
    code only). Per-phase timings, costs, and match percentages are also
    written to ``extraction_stats.json`` (unless ``write_stats=False``, which
    the CLI sets unless ``--debug``) so the UX can render them without
    re-deriving anything from the usage log.
    """
    run = _FileExtraction(
        workspace, docset_id, file_id, config=config, write_stats=write_stats, debug=debug
    )
    try:
        steps = run.phase1_steps()
        try:
            llm.drive(steps, run.phase1_config())
        finally:
            # Mirror the latches into the stats on every exit, so a failure
            # midway still reports how far phase 1 got.
            run.mirror_phase1()
        pages = run.after_phase1()
        per_page: list[_PageOutcome] = []
        if pages:
            per_page = _run_phase3_pages(run, pages)
        return run.finish(per_page)
    except BaseException as exc:
        run.fail(exc)
        raise
    finally:
        run.record()


@dataclass(frozen=True)
class _Phase3PageResult:
    """One page's phase-3 answer: ``{id -> locations}`` in page pixels, whether
    the page went through the 0-1000 grid, and how many boxes were dropped."""

    locations: dict[str, list[dict[str, Any]]]
    grid: bool
    boxes_dropped: int


#: One phase-3 page's result, as :meth:`_FileExtraction.finish` folds it:
#: ``(page_number, items, the page's answer, that page's usage totals)``.
_PageOutcome = tuple[int, list[UnmatchedItem], _Phase3PageResult, dict[str, Any]]


class _FileExtraction:
    """Every non-LLM step of extracting one file, around the phase-1 and
    phase-3 generators :func:`extract_values` drives.

    Drivers differ only in how the phase-1 and phase-3 generators are
    executed; everything else — schema load, the phase-1 request, post-phase-1
    normalization and checks, phase-2 matching, patching the phase-3 boxes in,
    the XML write, the usage row and ``extraction_stats`` — lives here once, so
    no driver can drift from it. Construction does the setup that
    :func:`extract_values` always did *before* it started recording (schema,
    guidance, PDF, credentials), so a setup failure raises from the constructor
    and leaves no usage row or stats, exactly as before.

    Call order: :meth:`phase1_steps` → drive → :meth:`mirror_phase1` (always) →
    :meth:`after_phase1` → drive each page from :meth:`phase3_steps` →
    :meth:`finish`; :meth:`fail` on an exception; :meth:`record` last, always.
    """

    def __init__(
        self,
        workspace: Workspace,
        docset_id: str,
        file_id: str,
        *,
        config: GroundedConfig,
        write_stats: bool,
        debug: bool,
    ) -> None:
        self.workspace = workspace
        self.docset_id = docset_id
        self.file_id = file_id
        self.config = config
        self.write_stats = write_stats
        self.debug = debug

        store = DocSetStore(workspace)
        self.store = store
        rnc_schema = store.get_schema(docset_id)  # RNC text; raises SchemaNotFound
        # The .dgml.xml is written under the (docset, file) pair's prefix, which only the
        # docset's file list reaches — so an unassigned file would get an orphan artifact.
        if not store.is_assigned(docset_id, file_id):
            raise FileNotFound(
                f"file '{file_id}' is not assigned to docset '{docset_id}'; assign it first"
            )
        # A stored schema is not re-checked on read; here it is about to be used.
        self.vocab = check_invariant_paths(parse_rnc(rnc_schema))
        self.schema = rnc_to_json_schema(rnc_schema)
        self.guidance = store.get_guidance(docset_id) if store.has_guidance(docset_id) else None
        self.pdf_bytes = _pdf_bytes(workspace, file_id)
        self.api_key = _resolve_api_key(config.values_api_key, config.values_api_key_env)
        self.api_base = config.values_api_base

        self.phase1_totals: dict[str, Any] = _empty_totals()
        self.phase3_totals: dict[str, Any] = _empty_totals()
        self.tool_calls_total = 0
        self.outcome = OUTCOME_ERROR
        self.error_msg: str | None = None
        self.started = time.monotonic()

        # Per-phase timings are filled as each phase finishes so a failure
        # midway still surfaces what we got done.
        self.phase1_started = 0.0
        self.phase3_started = 0.0
        self.phase1_duration = 0.0
        self.phase2_duration = 0.0
        self.phase3_duration = 0.0
        self.phase3_page_calls = 0
        self.phase3_grid_pages = 0
        self.phase3_boxes_dropped = 0
        self.phase3_pages_out_of_range = 0
        self.phase2_matched = 0
        self.phase3_matched = 0
        self.unmatched_count = 0
        self.total_locations = 0
        self.unprinted_fields = 0
        self.partly_printed_fields = 0
        self.dropped_inputs = 0
        self.citations_dropped = 0
        self.unsupported_values = 0
        self.repaired_inputs = 0
        self.located_pageless = 0
        self.dropped_unboxed_values = 0
        self.dropped_unboxed_paths: list[str] = []
        self.rejected_boxes = 0
        self.snapped_boxes = 0
        self.unnormalized_enums = 0
        self.derivations_checked = 0
        self.derivations_mismatched = 0
        self.invariants_checked = 0
        self.invariant_violations: list[str] = []
        self.phase1_layout: dict[str, Any] | None = None
        self.phase1_tool_schema_mode = "inlined"
        self.phase1_chunk_calls = 0
        self.phase1_truncated_retries = 0
        self.phase1_envelope_repairs = 0
        self.phase1_no_tool_call_retries = 0
        self.phase1_pdf_parts: list[tuple[int, int]] = []
        self.phase1 = Phase1Result(totals=self.phase1_totals)
        self._phase2_values: dict[str, Any] = {}
        self._phase2_unmatched_locations = 0

    # ---- phase 1 ------------------------------------------------------

    def phase1_config(self) -> LLMConfig:
        return _phase1_llm_config(
            self.config.values_model,
            self.api_key,
            self.api_base,
            self.config.values_reasoning_effort,
        )

    def phase1_steps(self) -> LLMSteps[Phase1Result]:
        """Build the phase-1 request and return its generator (text + page
        numbers, no bboxes)."""
        self.phase1_started = time.monotonic()
        phase1_schema = _drop_bboxes_from_schema(self.schema)
        # The tool parameter gets the expanded schema with prose annotations
        # stripped — the identical keys remain in the user-prompt copy
        # (phase1_schema), so no guidance is lost, and the smaller schema
        # keeps Gemini's constrained decoder within its state budget.
        phase1_tool_schema = _strip_annotations(_expand_refs(phase1_schema))
        # Schema + docset guidance are byte-identical for every file in a
        # docset, so a breakpoint here covers tools + system + schema + guidance
        # across files. Keeping the chunking directive OUT of this block is what
        # lets the retry still read it: the directive goes in its own block
        # after, so the cached prefix is unchanged between the two attempts.
        schema_text = _values_phase1_user_prompt(phase1_schema, self.guidance)
        values_model = self.config.values_model

        def _phase1_messages(
            *, chunked: bool, pdf: bytes, part: _PdfPart | None
        ) -> list[dict[str, Any]]:
            # Rebuilt per attempt — each phase-1 attempt appends to its message list.
            anthropic = is_anthropic_model(values_model)
            schema_block: dict[str, Any] = {"type": "text", "text": schema_text}
            # ``cache_control`` is Anthropic-specific, so gate on the provider
            # the same way ``call_with_tools`` gates its own marker — otherwise
            # a ``gemini/*`` values model gets an Anthropic-only key in user
            # content.
            if anthropic:
                schema_block["cache_control"] = {"type": "ephemeral"}
            user_content: list[dict[str, Any]] = [schema_block]
            if chunked:
                user_content.append(
                    {"type": "text", "text": prompt(PromptKey.VALUES_PHASE1_RETRY_CHUNKED)}
                )
            if part is not None:
                # Only when the whole document was refused as too large. After
                # the schema block, so its cache entry still serves every part.
                user_content.append({"type": "text", "text": part.prompt_text()})
            user_content.append(_pdf_content_block(pdf))
            if anthropic and chunked:
                # The per-file PDF earns a breakpoint only here. Chunked mode
                # is several turns over one unchanged prefix, so turns after
                # the first read it back; on the ordinary single-turn path a
                # breakpoint could only ever pay the write premium, which is
                # why the document is otherwise left untagged.
                user_content = _mark_document_cacheable(user_content)
            return [
                {"role": "system", "content": prompt(PromptKey.VALUES_PHASE1_SYSTEM)},
                {"role": "user", "content": user_content},
            ]

        # Every phase-1 fallback (chunked retry on truncation, permissive
        # schema on a "too many states" refusal, page-range parts when the
        # whole document is refused as too large) lives inside phase1_steps,
        # so every driver just runs it.
        return phase1_steps(
            workspace=self.workspace,
            file_id=self.file_id,
            run=self.phase1,
            pdf_bytes=self.pdf_bytes,
            messages_factory=_phase1_messages,
            tool_schema=phase1_tool_schema,
            vocab=self.vocab,
            model=values_model,
            api_key=self.api_key,
            api_base=self.api_base,
            max_tool_iters=self.config.max_tool_iters,
            reasoning_effort=self.config.values_reasoning_effort,
        )

    def mirror_phase1(self) -> None:
        """Copy phase 1's latches into the stats (call on every exit)."""
        self.phase1_tool_schema_mode = self.phase1.tool_schema_mode
        self.phase1_truncated_retries = self.phase1.truncated_retries
        self.phase1_envelope_repairs = self.phase1.counters["envelope_repairs"]
        self.phase1_no_tool_call_retries = self.phase1.counters["no_tool_call_retries"]
        self.phase1_pdf_parts = list(self.phase1.pdf_parts)
        self.phase1_chunk_calls = self.phase1.chunk_calls

    def after_phase1(self) -> list[tuple[int, list[UnmatchedItem]]]:
        """Post-process phase 1, run phase 2, and return the phase-3 work as
        ``[(page_number, items)]`` in page order (empty when phase 2 matched
        everything)."""
        phase1_values = self.phase1.values
        self.phase1_layout = self.phase1.layout
        self.tool_calls_total += self.phase1.tool_calls
        self.phase1_duration = round(time.monotonic() - self.phase1_started, 3)
        # Recorded above, before this refusal raises, so a refused run's
        # --debug usage row and stats sidecar still carry what phase 1 did.
        if self.phase1.refusal is not None:
            raise ValuesExtractionFailed(self.phase1.refusal)
        # The merged extracted_value leaf shape lets a sloppy model blur the
        # printed/unprinted boundary; normalize before phases 2/3 (and the
        # serializer) so their invariants hold regardless.
        demoted_claims = _normalize_leaf_provenance(phase1_values)
        # Inputs are positional paths the model counts out in its own output;
        # each cites the text it read there, so a miscounted index is
        # re-pointed at the one sibling that matches, or dropped — never left
        # naming a real but wrong element. Then normalize again: a value that
        # lost every input can no longer claim to be calculated or inferred.
        citations = verify_input_citations(phase1_values)
        self.repaired_inputs = citations.repaired
        self.citations_dropped = citations.dropped
        demoted_claims += _normalize_leaf_provenance(phase1_values)
        # A value read off the page but returned without locations would be
        # dropped as unverifiable; place it on the first page that prints its
        # exact text so phase 2 can ground it.
        self.located_pageless = locate_pageless_values(
            self.workspace,
            self.file_id,
            phase1_values,
            FileStore(self.workspace).get(self.file_id).page_count,
        )
        # Provisional: if a later phase fails, the stats still show the
        # claims phase 1 made with nothing behind them. Replaced in finish()
        # by the count of values dropped as unsupported.
        self.unsupported_values = demoted_claims
        # Unprinted leaves (calculated, inferred, defaulted) carry no
        # locations — phases 2 and 3 never see them; counted here so the
        # stats file can attest they were deliberate, not dropped. Partly
        # printed leaves are grounded as usual and also name their inputs.
        # dropped_inputs is inputs that won't resolve to a dg:inputs target —
        # incomplete provenance that would otherwise vanish silently at
        # serialization. Counted for paths now, so a later phase failing
        # still leaves it in the stats, and again after grounding to add the
        # evidence that never got a box.
        self.dropped_inputs = self.citations_dropped + count_dropped_inputs(
            phase1_values, evidence=False
        )
        self.unprinted_fields = sum(1 for _ in walk_unprinted_leaves(phase1_values))
        self.partly_printed_fields = sum(
            1 for _, leaf in walk_input_leaves(phase1_values) if "locations" in leaf
        )
        # Enum leaves whose normalized value isn't one of the schema's tokens
        # serialize text-only; count them so misses are visible in stats.
        self.unnormalized_enums = count_unnormalized_enum_values(phase1_values, self.vocab)
        # Recompute checkable derivations (report-only): a mismatch means a
        # calculated leaf's value agrees with neither the sum, the count, nor
        # any single one of its inputs.
        self.derivations_checked, self.derivations_mismatched = check_derivations(phase1_values)
        # Schema-declared `## Invariant:` relations (count/sum across the
        # submission). Report-only, like the derivation recompute above.
        self.invariants_checked, self.invariant_violations = check_invariants(
            phase1_values, self.vocab
        )

        # --- Phase 2: code-side OCR matching ------------------------
        phase2_result = run_phase2_matching(
            self.workspace, self.file_id, phase1_values, layout=self.phase1_layout
        )
        self.phase2_duration = phase2_result.stats.duration_s
        self.phase2_matched = phase2_result.stats.matched_locations
        self.total_locations = phase2_result.stats.total_locations
        self._phase2_unmatched_locations = phase2_result.stats.unmatched_locations

        # --- Phase 3 starts here: per-page LLM for remaining unmatched ---
        self.phase3_started = time.monotonic()
        self._phase2_values = phase2_result.values
        # Items on pages the file does not have are dropped before any call and
        # stay unmatched (#155); the count goes to the stats sidecar.
        unmatched, self.phase3_pages_out_of_range = _drop_out_of_range_pages(
            self.workspace,
            self.file_id,
            phase2_result.unmatched,
            page_count=FileStore(self.workspace).get(self.file_id).page_count,
        )
        return sorted(_group_by_page(unmatched).items())

    # ---- phase 3 ------------------------------------------------------

    def phase3_config(self) -> LLMConfig:
        return _phase3_llm_config(
            self.config.values_model,
            self.api_key,
            self.api_base,
            reasoning_effort=self.config.locations_reasoning_effort,
        )

    def phase3_steps(
        self, page: int, items: list[UnmatchedItem], totals: dict[str, Any]
    ) -> LLMSteps[_Phase3PageResult]:
        """The phase-3 generator for one page, folding its usage into *totals*."""
        return phase3_page_steps(
            workspace=self.workspace,
            file_id=self.file_id,
            page_number=page,
            items=items,
            values=self._phase2_values,
            model=self.config.values_model,
            api_key=self.api_key,
            api_base=self.api_base,
            max_tool_iters=self.config.max_tool_iters,
            totals=totals,
            reasoning_effort=self.config.locations_reasoning_effort,
        )

    # ---- finish -------------------------------------------------------

    def finish(self, per_page: list[_PageOutcome]) -> ExtractionResult:
        """Patch the phase-3 boxes in, write the XML, and mark the run ok."""
        final_values = self._phase2_values
        if per_page:
            (
                self.phase3_matched,
                self.phase3_page_calls,
                self.phase3_grid_pages,
                self.phase3_boxes_dropped,
            ) = _fold_phase3(final_values, per_page, self.phase3_totals)
        self.unmatched_count = self._phase2_unmatched_locations - self.phase3_matched
        self.phase3_duration = round(time.monotonic() - self.phase3_started, 3)
        # Phase 3's boxes are drawn by a model; where the page has a text
        # layer, a box whose words don't read as its value is wrong: it is
        # moved to the nearest exact match, or stripped before the drop below.
        self.snapped_boxes, self.rejected_boxes = verify_boxes(
            self.workspace, self.file_id, final_values
        )
        # A printed value grounding couldn't box can't be tied to the page,
        # and a value with no box, no default and nothing it used can't be
        # verified: drop them rather than write them with no provenance.
        unboxed = drop_unboxed_values(final_values)
        self.dropped_unboxed_values = unboxed.values
        self.dropped_unboxed_paths = unboxed.paths
        # After grounding, so evidence that never got a box (and so emits no
        # dg:chunk) counts as dropped too.
        self.dropped_inputs = (
            self.citations_dropped + unboxed.inputs + count_dropped_inputs(final_values)
        )
        self.unsupported_values = unboxed.unsupported
        # Count what is written, not what phase 1 returned: the drop above can
        # remove unprinted values (unsupported) and partly printed ones (unboxed).
        self.unprinted_fields = sum(1 for _ in walk_unprinted_leaves(final_values))
        self.partly_printed_fields = sum(
            1 for _, leaf in walk_input_leaves(final_values) if "locations" in leaf
        )

        # Extracted values live as a dg:extraction element in the file's core
        # <stem>.dgml.xml (spec §13): added as a sibling of an existing document
        # tree (full-extraction), or written as a standalone dg:chunk when no
        # tree exists yet (extraction).
        # Read again after phase 3, as before: a file deleted meanwhile
        # raises here rather than getting an orphan XML written for it. Same
        # for the assignment: `unassign` during the LLM phases cleared the
        # pair's prefix, and writing now would recreate an orphan under it.
        workspace = self.workspace
        stem = Path(FileStore(workspace).get(self.file_id).original_filename).stem
        if not self.store.is_assigned(self.docset_id, self.file_id):
            # `unassign` also deleted the pair's stats doc; record() must not
            # put it back.
            self.write_stats = False
            raise FileNotFound(
                f"file '{self.file_id}' is no longer assigned to docset '{self.docset_id}'"
            )
        xml_key = layout.dgml_xml_key(self.docset_id, self.file_id, stem)
        existing = (
            workspace.blobs.get_blob(xml_key).decode("utf-8")
            if workspace.blobs.blob_exists(xml_key)
            else None
        )
        if existing is not None and has_document_tree(existing):
            # A generated document tree is present — add extraction alongside it.
            mode = "full-extraction"
            doc = embed_extraction_into(existing, final_values, vocab=self.vocab)
        else:
            # No tree (fresh, or a prior extraction-only file) — (re)write standalone.
            mode = "extraction"
            doc = standalone_extraction_doc(final_values, vocab=self.vocab)
        workspace.blobs.put_blob(xml_key, doc.encode("utf-8"))
        self.outcome = OUTCOME_OK
        return ExtractionResult(
            values=final_values,
            tool_calls=self.tool_calls_total,
            xml_key=xml_key,
            mode=mode,
            model=self.config.values_model,
        )

    def fail(self, exc: BaseException) -> None:
        """Note why the run failed, for the usage row and the stats."""
        if isinstance(exc, ValuesExtractionFailed):
            self.error_msg = str(exc)
        elif isinstance(exc, Exception):
            # Non-ValuesExtractionFailed errors (programmer bug) — still
            # record what we can before letting them propagate.
            self.error_msg = f"{type(exc).__name__}: {exc}"
        # An interrupt records the error outcome with no message, as before.

    def record(self) -> None:
        """Write the usage row (``--debug``) and the stats (when opted in)."""
        # Usage recording is gated on --debug (like every other LLM path).
        # This aggregates phase 1 + 3 into one row; the internal per-call
        # configs carry no workspace, so they don't each auto-record.
        if self.debug:
            merged_totals = _merge_totals(self.phase1_totals, self.phase3_totals)
            event = UsageEvent(
                at=now_iso(),
                operation=OPERATION_EXTRACT_VALUES,
                model=self.config.values_model,
                cost_usd=merged_totals["cost_usd"],
                prompt_tokens=merged_totals["prompt_tokens"],
                completion_tokens=merged_totals["completion_tokens"],
                total_tokens=merged_totals["total_tokens"],
                cache_read_tokens=merged_totals["cache_read_tokens"],
                cache_creation_tokens=merged_totals["cache_creation_tokens"],
                duration_s=round(time.monotonic() - self.started, 3),
                outcome=self.outcome,
                context={
                    "file_id": self.file_id,
                    "docset_id": self.docset_id,
                    "tool_calls": self.tool_calls_total,
                },
                error=self.error_msg,
            )
            # One row, or one per tier when phase 1 and phase 3 responses
            # were served by different tiers.
            for part in scope_events(event, merged_totals):
                record_usage(self.workspace, part)
        # Even on failure, partial numbers help diagnose where we
        # stalled. Wrapped in try/except so telemetry can never break
        # the caller. Suppressed entirely unless the caller opted in
        # (the CLI does so only under --debug).
        try:
            if self.write_stats:
                _write_extraction_stats(
                    workspace=self.workspace,
                    docset_id=self.docset_id,
                    file_id=self.file_id,
                    model=self.config.values_model,
                    outcome=self.outcome,
                    error_msg=self.error_msg,
                    phase1_totals=self.phase1_totals,
                    phase3_totals=self.phase3_totals,
                    phase1_duration=self.phase1_duration,
                    phase2_duration=self.phase2_duration,
                    phase3_duration=self.phase3_duration,
                    phase3_page_calls=self.phase3_page_calls,
                    phase3_grid_pages=self.phase3_grid_pages,
                    phase3_boxes_dropped=self.phase3_boxes_dropped,
                    phase3_pages_out_of_range=self.phase3_pages_out_of_range,
                    phase2_matched=self.phase2_matched,
                    phase3_matched=self.phase3_matched,
                    unmatched=self.unmatched_count,
                    total_locations=self.total_locations,
                    unprinted_fields=self.unprinted_fields,
                    partly_printed_fields=self.partly_printed_fields,
                    dropped_inputs=self.dropped_inputs,
                    unsupported_values=self.unsupported_values,
                    repaired_inputs=self.repaired_inputs,
                    located_pageless=self.located_pageless,
                    dropped_unboxed_values=self.dropped_unboxed_values,
                    dropped_unboxed_paths=self.dropped_unboxed_paths,
                    rejected_boxes=self.rejected_boxes,
                    snapped_boxes=self.snapped_boxes,
                    unnormalized_enums=self.unnormalized_enums,
                    derivations_checked=self.derivations_checked,
                    derivations_mismatched=self.derivations_mismatched,
                    invariants_checked=self.invariants_checked,
                    invariant_violations=self.invariant_violations,
                    phase1_layout=self.phase1_layout,
                    phase1_tool_schema=self.phase1_tool_schema_mode,
                    phase1_chunk_calls=self.phase1_chunk_calls,
                    phase1_truncated_retries=self.phase1_truncated_retries,
                    phase1_pdf_parts=self.phase1_pdf_parts,
                    phase1_envelope_repairs=self.phase1_envelope_repairs,
                    phase1_no_tool_call_retries=self.phase1_no_tool_call_retries,
                )
        except Exception:
            pass


def _run_phase3_pages(
    run: _FileExtraction, pages: list[tuple[int, list[UnmatchedItem]]]
) -> list[_PageOutcome]:
    """Drive one file's phase-3 pages synchronously, concurrently across pages.

    Page calls are independent — each looks only at its own page's OCR, image,
    and items — so they run on a :class:`ThreadPoolExecutor` (litellm's HTTP
    call releases the GIL). Each page accumulates into its own totals, merged
    by :meth:`_FileExtraction.finish` after every call returns. A failed page
    raises out of the pool's ordered ``map``, so the first failing page (in
    page order) is the one reported, as before.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _do_page(page: int, items: list[UnmatchedItem]) -> _PageOutcome:
        local_totals = _empty_totals()
        page_result = llm.drive(run.phase3_steps(page, items, local_totals), run.phase3_config())
        return page, items, page_result, local_totals

    workers = min(_PHASE3_MAX_PARALLEL, max(1, len(pages)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda kv: _do_page(*kv), pages))


def _empty_totals() -> dict[str, Any]:
    return {
        "cost_usd": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        # Anthropic prompt-cache counters (default 0, summable). These flow
        # into ``extraction_stats.json`` via the ``**phaseN_totals`` spread
        # and into the ``usage.jsonl`` row via ``_merge_totals``.
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }


def _merge_totals(*partials: dict[str, Any]) -> dict[str, Any]:
    """Sum cost/token fields across totals dicts.

    Each partial uses the same ``None`` semantics as ``add_partial``:
    ``None`` means "the provider didn't tell us". The merge stays
    ``None`` only when every partial is ``None`` for that key; if any
    one reports a number, the others' ``None``s are treated as zero so
    we don't drop information."""
    merged: dict[str, Any] = _empty_totals()
    for key in merged:
        observations = [p[key] for p in partials if p[key] is not None]
        if observations:
            merged[key] = sum(observations)
    # The per-tier breakdowns merge alongside (``add_partial`` carries them),
    # without touching the aggregate sums above.
    tiers: dict[str, Any] = {}
    for p in partials:
        add_partial(tiers, {TIERS_KEY: p.get(TIERS_KEY) or {}})
    if TIERS_KEY in tiers:
        merged[TIERS_KEY] = tiers[TIERS_KEY]
    return merged


def _write_extraction_stats(
    *,
    workspace: Workspace,
    docset_id: str,
    file_id: str,
    model: str,
    outcome: str,
    error_msg: str | None,
    phase1_totals: dict[str, Any],
    phase3_totals: dict[str, Any],
    phase1_duration: float,
    phase2_duration: float,
    phase3_duration: float,
    phase3_page_calls: int,
    phase3_grid_pages: int,
    phase3_boxes_dropped: int,
    phase3_pages_out_of_range: int,
    phase2_matched: int,
    phase3_matched: int,
    unmatched: int,
    total_locations: int,
    unprinted_fields: int,
    partly_printed_fields: int,
    dropped_inputs: int,
    unsupported_values: int,
    repaired_inputs: int,
    located_pageless: int,
    dropped_unboxed_values: int,
    dropped_unboxed_paths: list[str],
    rejected_boxes: int,
    snapped_boxes: int,
    unnormalized_enums: int,
    derivations_checked: int,
    derivations_mismatched: int,
    invariants_checked: int,
    invariant_violations: list[str],
    phase1_layout: dict[str, Any] | None,
    phase1_tool_schema: str,
    phase1_chunk_calls: int,
    phase1_truncated_retries: int,
    phase1_envelope_repairs: int,
    phase1_no_tool_call_retries: int,
    phase1_pdf_parts: list[tuple[int, int]] | None = None,
) -> None:
    """Write ``extraction_stats.json`` into the file's marker directory.

    Phase 2's row carries no cost/token fields because it never makes an
    LLM call. The three match counts together cover every phase-1
    location: ``phase2_matched + phase3_matched + unmatched == total``.
    Computed (reasoned) fields carry no locations, so they sit outside
    that invariant and are counted separately."""
    stats = {
        "completed_at": now_iso(),
        "model": model,
        "outcome": outcome,
        "error": error_msg,
        "phases": {
            "phase1": {
                "duration_s": phase1_duration,
                # 1 = ordinary single submit_values; >1 = the chunked
                # protocol (append_entries / done=false continuations).
                "chunk_calls": phase1_chunk_calls,
                # times phase 1 was restarted with the explicit chunking
                # directive after a finish_reason='length' truncation.
                "truncated_retries": phase1_truncated_retries,
                # [first_page, last_page] of each page-range part, present only
                # when the provider refused the whole document as too large and
                # phase 1 ran once per part.
                **({"pdf_parts": [list(p) for p in phase1_pdf_parts]} if phase1_pdf_parts else {}),
                # submit_values calls whose envelope had to be repaired before
                # the tree could be read (see _repair_submit_values_args).
                "envelope_repairs": phase1_envelope_repairs,
                # turns that ended with no tool call and were retried with a
                # reminder (never forced; reachable only where phase 1 runs on
                # auto tool choice: Anthropic, or a fallback from "required").
                "no_tool_call_retries": phase1_no_tool_call_retries,
                **public_totals(phase1_totals),
            },
            "phase2": {"duration_s": phase2_duration},
            "phase3": {
                "duration_s": phase3_duration,
                "page_calls": phase3_page_calls,
                # pages with no OCR words, located on the 0-1000 grid, and
                # returned boxes dropped as empty or off the page (their
                # items stay unmatched).
                "grid_pages": phase3_grid_pages,
                "boxes_dropped": phase3_boxes_dropped,
                # pages phase 1 cited that the file does not have (outside
                # 1..page_count, no page image); their items stay unmatched
                # and no call is made for them.
                "pages_out_of_range": phase3_pages_out_of_range,
                **public_totals(phase3_totals),
            },
        },
        "matching": {
            "total_locations": total_locations,
            "matched_phase2": phase2_matched,
            "matched_phase3": phase3_matched,
            "unmatched": unmatched,
            "unprinted_fields": unprinted_fields,
            "partly_printed_fields": partly_printed_fields,
            "dropped_inputs": dropped_inputs,
            # Values claimed as calculated/inferred that named nothing they
            # used (or whose inputs all dropped): written with no dg:method,
            # so the XML can't tell them from a value grounding missed.
            "unsupported_values": unsupported_values,
            # Inputs whose cited text matched a different entry of the same
            # array than the index the model gave: re-pointed there.
            "repaired_inputs": repaired_inputs,
            # Values phase 1 returned without locations, placed on the first
            # page printing their exact text.
            "located_pageless": located_pageless,
            # Printed values phases 2/3 found no box for: not written.
            "dropped_unboxed_values": dropped_unboxed_values,
            # Which ones ("path=text"), so a drop is diagnosable without
            # re-running the extraction.
            "dropped_unboxed": dropped_unboxed_paths,
            # Boxes whose words on the text layer don't read as their value:
            # moved to the one exact match on their line (snapped), or, with
            # no single match there, stripped (rejected).
            "rejected_boxes": rejected_boxes,
            "snapped_boxes": snapped_boxes,
            "unnormalized_enum_values": unnormalized_enums,
            "derivations_checked": derivations_checked,
            "derivations_mismatched": derivations_mismatched,
            # Schema-declared invariants: how many were evaluable, and the
            # human-readable text of each violation so a failure is
            # actionable without re-deriving it from the values tree.
            "invariants_checked": invariants_checked,
            "invariants_violated": len(invariant_violations),
            "invariant_violations": invariant_violations,
        },
        # Phase-1's emitted layout hint, persisted so we can audit
        # whether the model produced a useful descriptor for each
        # array (some models drop optional tool-call parameters).
        "phase1_layout": phase1_layout,
        # "inlined" when the docset schema rode inside the submit_values
        # tool parameter (provider-enforced shape); "permissive" when the
        # provider rejected the inlined schema (Gemini "too many states")
        # and the retry ran with a bare object parameter + code-side
        # vocabulary pruning.
        "phase1_tool_schema": phase1_tool_schema,
    }
    workspace.docs.put_doc(
        layout.Collection.EXTRACTION_STATS, layout.pair_id(docset_id, file_id), stats
    )


# ---- Step plumbing shared by phase 1 and phase 3 --------------------------

#: Message prefix for a phase-1 request that could not be completed.
_PHASE1_CALL_FAILED = "extraction call failed"


def _phase3_call_failed(page_number: int) -> str:
    """Message prefix for a phase-3 request on *page_number* that failed."""
    return f"phase 3 page {page_number} call failed"


def _tool_call_step(
    config: LLMConfig,
    *,
    failure: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] | None = None,
    cache: bool = False,
) -> LLMSteps[CallResult]:
    """:func:`~dgml_core.llm.steps_with_tools`, with any error it raises while
    building the request or reading the response reported as
    :class:`ValuesExtractionFailed` ``"<failure>: <Type>: <message>"`` — the
    same wrapping the inline call site applied to the whole ``call_with_tools``
    call.

    Drivers deliver a failed request by throwing its exception in at the pending
    ``yield`` (see :data:`~dgml_core.llm.LLMSteps`), so executor failures land
    here too and get the same single prefix, whichever driver runs it. Only
    ``Exception`` is wrapped;
    ``KeyboardInterrupt`` and ``GeneratorExit`` pass through untouched.
    """
    try:
        return (
            yield from steps_with_tools(
                config, messages=messages, tools=tools, tool_choice=tool_choice, cache=cache
            )
        )
    except Exception as exc:
        raise ValuesExtractionFailed(f"{failure}: {type(exc).__name__}: {exc}") from exc


def _phase1_llm_config(
    model: str,
    api_key: str | None,
    api_base: str | None,
    reasoning_effort: str | None = _VALUES_REASONING_EFFORT,
) -> LLMConfig:
    """The request config for every phase-1 turn.

    Phase 1 leaves Anthropic on auto tool choice (see
    :func:`_phase1_attempt_steps`), so the wrapper keeps reasoning_effort there
    — only a forced tool_choice triggers the Anthropic drop. That also makes
    this the only call site where the reasoning budget has any effect on
    Claude, which is why the values reasoning effort applies here and nowhere
    else: *reasoning_effort* is :data:`_VALUES_REASONING_EFFORT` unless the
    workspace set ``grounded.values_reasoning_effort`` (``None`` sends no
    ``reasoning_effort``). It carries no workspace, so driving with it writes no usage row:
    :func:`extract_values` records one row for the whole extraction.
    """
    return LLMConfig(
        model=model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=None,
        max_completion_tokens=_DEFAULT_MAX_COMPLETION_TOKENS,
        temperature=_DEFAULT_VALUES_TEMPERATURE,
        timeout=_DEFAULT_TIMEOUT_SECONDS,
        reasoning_effort=reasoning_effort,
    )


def _phase3_llm_config(
    model: str,
    api_key: str | None,
    api_base: str | None,
    *,
    reasoning_effort: str | None,
) -> LLMConfig:
    """The request config for a phase-3 page call.

    reasoning_effort is set unconditionally — the wrapper drops it for
    Anthropic-routed models because tool_choice forces a function call, and
    Anthropic rejects extended thinking with forced tools. *reasoning_effort*
    is ``grounded.locations_reasoning_effort`` on a page with OCR words
    (``None`` sends none) and :data:`_PHASE3_GRID_REASONING_EFFORT` on a page
    on the 0-1000 grid. No workspace, so driving with it writes no usage row.
    """
    return LLMConfig(
        model=model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=None,
        max_completion_tokens=_DEFAULT_MAX_COMPLETION_TOKENS,
        temperature=_DEFAULT_VALUES_TEMPERATURE,
        timeout=_DEFAULT_TIMEOUT_SECONDS,
        reasoning_effort=reasoning_effort,
    )


# ---- Phase 3: per-page LLM for unmatched items ----------------------------


_PHASE3_MAX_PARALLEL = 8


def _drop_out_of_range_pages(
    workspace: Workspace,
    file_id: str,
    unmatched: list[UnmatchedItem],
    *,
    page_count: int | None,
) -> tuple[list[UnmatchedItem], int]:
    """Drop the phase-3 items on pages the file does not have, before any call.

    Phase 1 occasionally cites page 2 of a one-page invoice. Such a page has
    no image to send, and raising over it threw away the whole tree, phase-2
    matches included (#155). The items simply stay unmatched; the number of
    pages dropped goes to the stats sidecar. A page is dropped only when the
    record's ``page_count`` is known and positive, the page is outside
    ``1..page_count``, and the page has no image: a stale or zero count
    cannot hide a page that was rendered, and a record without a count keeps
    the loud failure for a missing page image. Returns
    ``(items_to_locate, pages_out_of_range)``.
    """
    if page_count is None or page_count <= 0:
        return unmatched, 0
    outside = {item.page_number for item in unmatched if not 1 <= item.page_number <= page_count}
    missing = {
        page
        for page in outside
        if not workspace.blobs.blob_exists(layout.file_page_image_key(file_id, page))
    }
    kept = [item for item in unmatched if item.page_number not in missing]
    return kept, len(missing)


def _group_by_page(unmatched: list[UnmatchedItem]) -> dict[int, list[UnmatchedItem]]:
    """Phase-3 work items grouped by page, in encounter order within a page."""
    by_page: dict[int, list[UnmatchedItem]] = {}
    for item in unmatched:
        by_page.setdefault(item.page_number, []).append(item)
    return by_page


def _fold_phase3(
    values: dict[str, Any], per_page: list[_PageOutcome], totals: dict[str, Any]
) -> tuple[int, int, int, int]:
    """Patch each page's located boxes into *values* and fold its usage into
    *totals*. Returns ``(matched_count, page_calls, grid_pages, boxes_dropped)``:
    ``grid_pages`` counts pages located on the 0-1000 grid (no OCR words), and
    ``boxes_dropped`` the returned boxes discarded as off the page or empty."""
    matched_count = 0
    page_calls = 0
    grid_pages = 0
    boxes_dropped = 0
    for _page, items, page_result, local_totals in per_page:
        page_calls += 1
        grid_pages += page_result.grid
        boxes_dropped += page_result.boxes_dropped
        add_partial(totals, local_totals)
        for item in items:
            model_locs = page_result.locations.get(item.id)
            if not model_locs:
                continue
            if _patch_value_with_locations(values, item, model_locs):
                matched_count += 1
    return matched_count, page_calls, grid_pages, boxes_dropped


def phase3_page_steps(
    *,
    workspace: Workspace,
    file_id: str,
    page_number: int,
    items: list[UnmatchedItem],
    values: dict[str, Any],
    model: str,
    api_key: str | None,
    api_base: str | None,
    max_tool_iters: int,
    totals: dict[str, Any] | None = None,
    reasoning_effort: str | None,
) -> LLMSteps[_Phase3PageResult]:
    """Phase 3 for ONE page, as an :data:`~dgml_core.llm.LLMSteps` generator.

    Yields the forced ``submit_locations`` request, receives the response, and
    returns a :class:`_Phase3PageResult`: ``{id: [{page_number, bounding_box},
    ...]}`` parsed from the model's tool call, with the page's grid/dropped
    counts. A page with OCR words runs at *reasoning_effort* (the configured
    ``grounded.locations_reasoning_effort``, ``None`` = send none); a page with
    none is asked for boxes on a 0-1000 grid (at
    :data:`_PHASE3_GRID_REASONING_EFFORT`), scaled to pixels when parsed. One
    generator per page, so a driver can fan a file's pages out concurrently
    (the sync :func:`_run_phase3_pages` does so on a thread pool). Drive it with the config from
    :func:`_phase3_llm_config`; a failed request thrown in at the pending
    ``yield`` surfaces as :class:`ValuesExtractionFailed` ``"phase 3 page <n>
    call failed: <Type>: <message>"``. Each response's usage is folded into
    *totals* when given, so phase-3 cost survives in the caller's accounting.
    The page image and OCR words are read from *workspace* before the request
    is yielded.

    Deliberately uncached, for three independent reasons: the cacheable prefix
    is tools + system and ``_submit_locations_tool`` is built from *this page's*
    unmatched ids, so the prefix differs on every call and no read can match;
    page calls run in parallel, and an entry is only readable once the first
    response has begun streaming; and the phase-3 system prompt alone is under
    the minimum cacheable prefix. Marking it only risks paying the cache-write
    premium for a read that cannot arrive — the same reasoning that keeps the
    PDF and image blocks untagged."""
    image_key = layout.file_page_image_key(file_id, page_number)
    if not workspace.blobs.blob_exists(image_key):
        raise ValuesExtractionFailed(
            f"phase 3: no page image for file '{file_id}' page {page_number}"
        )

    try:
        page_words = get_page_words(workspace, file_id, page_number)
    except FileNotFound:
        page_words = {"page": page_number, "total_words": 0, "words": []}

    image_bytes = workspace.blobs.get_blob(image_key)
    image_size = _page_pixel_size(workspace, file_id, page_number, image_bytes)
    # A page with no words (a scan added in digital mode, say) gives the model
    # nothing to measure pixels against. Asked for pixels anyway it spends its
    # reasoning guessing the image resolution, and still answers in its own
    # 0-1000 frame, so the boxes land in the page's top-left corner. Ask for
    # that frame outright and scale it to pixels here.
    normalized = image_size is not None and not page_words.get("words")
    by_words = bool(page_words.get("words"))
    user_text = _phase3_user_prompt(
        page_number=page_number,
        items=items,
        page_words=page_words,
        page_anchors=_collect_page_anchors(values, page_number),
        image_size=image_size,
        grid=normalized,
    )
    if normalized:
        system_key = PromptKey.VALUES_PHASE3_SYSTEM_NO_WORDS
    elif by_words:
        system_key = PromptKey.VALUES_PHASE3_SYSTEM
    else:
        # No words and no known image size: only the pixel contract is left.
        system_key = PromptKey.VALUES_PHASE3_SYSTEM_PIXELS
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt(system_key)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                _image_content_block(image_bytes),
            ],
        },
    ]
    tools = [_submit_locations_tool([it.id for it in items], by_words=by_words)]

    # A page with OCR words takes the configured locations effort
    # (``grounded.locations_reasoning_effort``, ``None`` = send none); the
    # no-words grid path keeps its own constant. The effort was checked by name
    # at load time, not against the model: one the provider refuses fails here,
    # after phase 1 has been paid for.
    llm_config = _phase3_llm_config(
        model,
        api_key,
        api_base,
        reasoning_effort=_PHASE3_GRID_REASONING_EFFORT if normalized else reasoning_effort,
    )
    forced_tool_choice = {
        "type": "function",
        "function": {"name": _TOOL_SUBMIT_LOCATIONS},
    }

    for _ in range(max_tool_iters):
        result = yield from _tool_call_step(
            llm_config,
            failure=_phase3_call_failed(page_number),
            messages=messages,
            tools=tools,
            tool_choice=forced_tool_choice,
        )
        if totals is not None:
            add_partial(totals, with_tier(result.usage, result.response, None))

        if not result.tool_calls:
            raise ValuesExtractionFailed(f"phase 3 page {page_number}: model returned no tool call")
        call = result.tool_calls[0]
        if call.function.name != _TOOL_SUBMIT_LOCATIONS:
            raise ValuesExtractionFailed(
                f"phase 3 page {page_number}: unexpected tool {call.function.name!r}"
            )
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValuesExtractionFailed(
                f"phase 3 page {page_number}: malformed JSON args: {exc}"
            ) from exc
        locations, dropped = _parse_submit_locations(
            args,
            page_number,
            image_size=image_size,
            normalized=normalized,
            words_by_idx=(
                {w["idx"]: w for w in page_words["words"] if isinstance(w.get("idx"), int)}
                if by_words
                else None
            ),
        )
        return _Phase3PageResult(locations=locations, grid=normalized, boxes_dropped=dropped)

    raise ValuesExtractionFailed(
        f"phase 3 page {page_number} exceeded max_tool_iters={max_tool_iters}"
    )


def _submit_word_ranges_tool(ids: list[str]) -> dict[str, Any]:
    """The ``by_words`` form of :func:`_submit_locations_tool`. Copying four
    pixel coordinates per box out of a page-long word table is where models
    slip — mixing one row's x with another's y, miscopying an edge — so the
    model names words by their ``idx`` and :func:`_boxes_from_word_ranges`
    does the geometry."""
    return {
        "type": "function",
        "function": {
            "name": _TOOL_SUBMIT_LOCATIONS,
            "description": (
                "Submit the OCR words of each requested id's value, as word index "
                "ranges. Each id corresponds to a value already extracted from the "
                "document; you are only locating it, not changing text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "locations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string", "enum": ids},
                                "words": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        # [first_idx, last_idx], inclusive: the
                                        # value's words on one visual line.
                                        "type": "array",
                                        "items": {"type": "integer", "minimum": 0},
                                        "minItems": 2,
                                        "maxItems": 2,
                                    },
                                },
                            },
                            "required": ["id", "words"],
                        },
                    }
                },
                "required": ["locations"],
                "additionalProperties": False,
            },
        },
    }


def _boxes_from_word_ranges(
    ranges: Any, words_by_idx: dict[int, dict[str, Any]]
) -> list[list[int]] | None:
    """One ``[left, top, right, bottom]`` box per visual line covering the
    words named by *ranges* (``[first_idx, last_idx]`` pairs, inclusive), or
    ``None`` when a range is malformed or names a word the page doesn't have
    — the value then stays unresolved rather than half-located."""
    if not isinstance(ranges, list) or not ranges:
        return None
    picked: dict[int, dict[str, Any]] = {}
    for pair in ranges:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(i, int) for i in pair)
            or pair[0] > pair[1]
        ):
            return None
        for idx in range(pair[0], pair[1] + 1):
            word = words_by_idx.get(idx)
            if word is None:
                return None
            picked[idx] = word
    return line_boxes(
        [
            TextWord(
                idx=idx,
                text=str(w.get("text", "")),
                text_norm=fuzzy_norm(str(w.get("text", ""))),
                left=int(w["location"]["bounding_box"][0]),
                top=int(w["location"]["bounding_box"][1]),
                right=int(w["location"]["bounding_box"][2]),
                bottom=int(w["location"]["bounding_box"][3]),
            )
            for idx, w in picked.items()
        ]
    )


def _parse_submit_locations(
    args: dict[str, Any],
    page_number: int,
    *,
    image_size: tuple[int, int] | None = None,
    normalized: bool = False,
    words_by_idx: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Parse a ``submit_locations`` tool-args payload into
    ``({id → locations}, boxes_dropped)``. Malformed entries are dropped
    silently — they'll show up as unresolved in the stats.

    With ``normalized`` the model answered on a 0-1000 grid and each box is
    scaled to ``image_size`` pixels. A box that is empty or inverted, or (with
    ``image_size`` known) leaves the page, is dropped rather than stored and
    counted in ``boxes_dropped``: the value then stays unresolved instead of
    pointing at the wrong spot.

    With ``words_by_idx`` (a page with OCR words) an entry's ``words`` ranges
    are turned into boxes by :func:`_boxes_from_word_ranges`; a malformed or
    unknown range counts as one dropped box. An entry that answers with
    ``bounding_boxes`` anyway is still read as pixels."""
    raw = args.get("locations")
    out: dict[str, list[dict[str, Any]]] = {}
    dropped = 0
    if not isinstance(raw, list):
        return out, dropped
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        item_id = entry.get("id")
        if not isinstance(item_id, str):
            continue
        if words_by_idx is not None and "words" in entry:
            boxes = _boxes_from_word_ranges(entry.get("words"), words_by_idx)
            if boxes is None:
                dropped += 1
            else:
                out[item_id] = [{"page_number": page_number, "bounding_box": b} for b in boxes]
            continue
        bboxes = entry.get("bounding_boxes")
        if not isinstance(bboxes, list):
            continue
        locs: list[dict[str, Any]] = []
        for bbox in bboxes:
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if not all(isinstance(c, (int, float)) for c in bbox):
                continue
            box = _to_page_pixels(bbox, image_size=image_size, normalized=normalized)
            if box is None:
                dropped += 1
                continue
            # Boxes are integer image pixels [left, top, right, bottom].
            locs.append({"page_number": page_number, "bounding_box": box})
        if locs:
            out[item_id] = locs
    return out, dropped


# Tolerance, as a share of the page side, for a box edge past the image
# border (about 50-65 px on a 300 dpi Letter page): over-covering by a few
# pixels is what the prompt asks for, so an edge just outside is clamped, and
# only a box well off the page is dropped.
_BOX_EDGE_SLACK = 0.02


def _to_page_pixels(
    bbox: Sequence[float],
    *,
    image_size: tuple[int, int] | None,
    normalized: bool,
) -> list[int] | None:
    """Turn one returned box into integer page pixels, or ``None`` to drop it.

    Scales a 0-1000 box when ``normalized``, then (with ``image_size`` known)
    clamps edges within the slack of the border and rejects a box that is
    inverted, empty, or lies off the page."""
    left, top, right, bottom = (float(c) for c in bbox)
    if normalized and image_size is not None:
        width, height = image_size
        left, right = left * width / 1000, right * width / 1000
        top, bottom = top * height / 1000, bottom * height / 1000
    if image_size is not None:
        width, height = image_size
        slack_x, slack_y = width * _BOX_EDGE_SLACK, height * _BOX_EDGE_SLACK
        if (
            left < -slack_x
            or top < -slack_y
            or right > width + slack_x
            or bottom > height + slack_y
        ):
            return None
        left, right = max(0.0, left), min(float(width), right)
        top, bottom = max(0.0, top), min(float(height), bottom)
    box = [round(left), round(top), round(right), round(bottom)]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _page_pixel_size(
    workspace: Workspace, file_id: str, page_number: int, image_bytes: bytes
) -> tuple[int, int] | None:
    """``(width, height)`` of the page image the model is shown: read from the
    PNG header, else from ``page_text`` (which records the same size), else
    ``None``, in which case phase 3 keeps the plain pixel prompt."""
    size: tuple[Any, Any]
    try:
        size = _png_dimensions(image_bytes)
    except ValueError:
        payload = workspace.read_page_text(file_id, page_number) or {}
        size = (payload.get("width"), payload.get("height"))
    width, height = size
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    return None


def _patch_value_with_locations(
    values: dict[str, Any],
    item: UnmatchedItem,
    locations: list[dict[str, Any]],
) -> bool:
    """Replace the page-only entry for ``item`` with ``locations``.

    Walks ``item.path``, finds the first location whose ``page_number``
    matches and has no ``bounding_box`` yet, and swaps it for
    ``locations`` (which may be multiple entries when the text wraps).
    Returns True on successful patch."""
    cur: Any = values
    for seg in item.path:
        if isinstance(cur, dict):
            cur = cur.get(seg)
        elif isinstance(cur, list) and isinstance(seg, int) and 0 <= seg < len(cur):
            cur = cur[seg]
        else:
            return False
        if cur is None:
            return False
    if not isinstance(cur, dict):
        return False
    locs = cur.get("locations")
    if not isinstance(locs, list):
        return False
    for i, loc in enumerate(locs):
        if not isinstance(loc, dict):
            continue
        if loc.get("page_number") != item.page_number:
            continue
        if "bounding_box" in loc:
            continue
        cur["locations"] = locs[:i] + locations + locs[i + 1 :]
        return True
    return False


def _collect_page_anchors(values: dict[str, Any], page_number: int) -> list[dict[str, Any]]:
    """Already-matched locations on this page — sent as phase-3 context
    so the model can anchor unmatched items to the same row as their
    siblings. Capped at 80 entries (sampled) to keep busy-page prompts
    bounded; the anchors are a spatial reference, not a complete map."""
    from .matching import _walk_leaves

    out: list[dict[str, Any]] = []
    for path, leaf in _walk_leaves(values):
        for loc in leaf.get("locations", []):
            if not isinstance(loc, dict):
                continue
            if loc.get("page_number") != page_number:
                continue
            bbox = loc.get("bounding_box")
            if not isinstance(bbox, list):
                continue
            out.append(
                {
                    "path": path_to_str(path),
                    "text": leaf.get("text", ""),
                    "bounding_box": bbox,
                }
            )
    if len(out) > 80:
        step = max(1, len(out) // 80)
        out = out[::step][:80]
    return out


def _phase3_user_prompt(
    *,
    page_number: int,
    items: list[UnmatchedItem],
    page_words: dict[str, Any],
    page_anchors: list[dict[str, Any]],
    image_size: tuple[int, int] | None = None,
    grid: bool = False,
) -> str:
    items_lines = [
        f"- id: {it.id}; path: {path_to_str(it.path)}; text: {json.dumps(it.text)}" for it in items
    ]
    anchors_lines = [
        f"- {a['path']}: text={json.dumps(a['text'])} bbox={a['bounding_box']}"
        for a in page_anchors
    ] or ["(none — these are the first values located on this page)"]
    # The OCR word listing is always rendered as a compact TOON table (see
    # :mod:`dgml_core.toon`, measured at -72.2% input tokens versus the former
    # ``json.dumps(..., indent=2)`` array). Lossless; only the words given TO
    # the model change — the ``submit_locations`` response contract is untouched.
    if grid:
        # A page reaches the grid only without words, so phase 2 placed nothing
        # on it and there are no anchors to show.
        return prompt(PromptKey.VALUES_PHASE3_USER_NO_WORDS).format(
            page_number=page_number,
            needs_locating="\n".join(items_lines),
        )
    size_line = (
        f" The page image is {image_size[0]} x {image_size[1]} pixels (width x height)."
        if image_size is not None
        else ""
    )
    user_key = (
        PromptKey.VALUES_PHASE3_USER
        if page_words.get("words")
        else PromptKey.VALUES_PHASE3_USER_PIXELS
    )
    return prompt(user_key).format(
        page_number=page_number,
        image_size=size_line,
        ocr_words=encode_phase3_words(page_words.get("words", [])),
        known_locations="\n".join(anchors_lines),
        needs_locating="\n".join(items_lines),
    )


def _submit_locations_tool(ids: list[str], *, by_words: bool = False) -> dict[str, Any]:
    """Tool spec for phase 3's submit. ``id`` is constrained to the set
    of ids we're asking about so the model can't invent extras and we
    don't need to filter on the way back.

    With ``by_words`` (a page with OCR words) each id is answered with word
    index ranges, one per visual line, and code builds the boxes from those
    words; otherwise with boxes (the 0-1000 grid on a page without words)."""
    if by_words:
        return _submit_word_ranges_tool(ids)
    return {
        "type": "function",
        "function": {
            "name": _TOOL_SUBMIT_LOCATIONS,
            "description": (
                "Submit one or more bounding boxes for each requested id. "
                "Each id corresponds to a value already extracted from the "
                "document; you are only attaching bboxes, not changing text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "locations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string", "enum": ids},
                                "bounding_boxes": {
                                    "type": "array",
                                    "minItems": 1,
                                    "items": {
                                        # [left, top, right, bottom]: image pixels,
                                        # or the 0-1000 grid on a page without
                                        # OCR words (see phase3_page_steps).
                                        "type": "array",
                                        "items": {
                                            "type": "integer",
                                            "minimum": 0,
                                        },
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                },
                            },
                            "required": ["id", "bounding_boxes"],
                        },
                    }
                },
                "required": ["locations"],
                "additionalProperties": False,
            },
        },
    }


def _image_content_block(image_bytes: bytes) -> dict[str, Any]:
    """An OpenAI-style image content block. litellm normalizes this
    across providers; Gemini gets ``inline_data`` with the same base64
    payload. MIME is detected from magic bytes so the helper works for
    any rendered format without baked-in assumptions."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif image_bytes.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        mime = "image/gif"
    elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError(
            "unrecognized image format for inline content block; "
            "expected JPEG/PNG/GIF/WEBP magic bytes"
        )
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


class _OutputTruncated(ValuesExtractionFailed):
    """A phase-1 turn stopped with ``finish_reason == "length"`` — the tool
    call was clipped mid-output. Caught by :func:`extract_values`, which
    retries once with an explicit chunked-submission directive; surfaces as a
    plain :class:`ValuesExtractionFailed` if the retry truncates too."""


def _merge_values(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge a follow-up ``submit_values`` chunk into the accumulated tree.

    In place on *base*: non-leaf dicts merge recursively, lists concatenate
    (chunks carry only NEW entries, per the prompt contract), and leaves
    (dicts carrying ``text``) or scalars replace. Returns *base*."""
    for key, value in extra.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict) and "text" not in value:
            _merge_values(current, value)
        elif isinstance(current, list) and isinstance(value, list):
            current.extend(value)
        else:
            base[key] = value
    return base


# ---- Phase 1 over page-range parts (a document refused as too large) ------


@dataclass(frozen=True)
class _PdfPart:
    """One page range of a document, sent to phase 1 as its own PDF."""

    first_page: int
    last_page: int
    total_pages: int

    def prompt_text(self) -> str:
        return prompt(PromptKey.VALUES_PHASE1_PART).format(
            first_page=self.first_page,
            last_page=self.last_page,
            total_pages=self.total_pages,
        )


_Phase1Run = tuple[dict[str, Any], int, int]


def _phase1_in_parts(
    workspace: Workspace,
    file_id: str,
    pdf_bytes: bytes,
    *,
    model: str,
    run: Callable[[bytes, _PdfPart | None], LLMSteps[_Phase1Run]],
    refused: ValuesExtractionFailed,
) -> LLMSteps[list[tuple[_PdfPart, _Phase1Run]]]:
    """Run phase 1 over page ranges of a document whose whole-PDF request the
    provider refused as too large (*refused*).

    The document is cut in half, and any half the provider refuses too is cut
    again, until each part is accepted. Parts are sliced with the workspace's
    PDF engine, the same one generation's windows use. Returns each part with
    its phase-1 result, in document order. A single page that is still too
    large fails the file.
    """
    record_count = FileStore(workspace).get(file_id).page_count
    try:
        page_count = (
            record_count
            if isinstance(record_count, int) and record_count > 0
            else pdf_page_count_bytes(pdf_bytes)
        )
        pdf_config = load_pdf_config(workspace)
    except Exception as exc:
        raise ValuesExtractionFailed(
            f"{refused} The document could not be split into page ranges: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if page_count <= 1:
        raise ValuesExtractionFailed(
            f"page 1 alone is too large for one extraction request to {model!r}: {refused}"
        ) from refused
    logger.info(
        "extraction: %s refused file %s (%d pages, %d bytes) as too large for one "
        "request; sending it in page ranges",
        model,
        file_id,
        page_count,
        len(pdf_bytes),
    )

    def attempt(first: int, last: int) -> LLMSteps[list[tuple[_PdfPart, _Phase1Run]]]:
        try:
            sliced = slice_pages(
                pdf_bytes, range(first, last + 1), config=pdf_config, total_pages=page_count
            )
        except Exception as exc:
            raise ValuesExtractionFailed(
                f"{refused} Slicing pages {first}-{last} out of it failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        part = _PdfPart(first_page=first, last_page=last, total_pages=page_count)
        try:
            return [(part, (yield from run(sliced, part)))]
        except ValuesExtractionFailed as exc:
            if not is_request_too_large(exc):
                raise
            if first == last:
                raise ValuesExtractionFailed(
                    f"page {first} alone is too large for one extraction request "
                    f"to {model!r}: {exc}"
                ) from exc
            mid = (first + last) // 2
            return (yield from attempt(first, mid)) + (yield from attempt(mid + 1, last))

    mid = (1 + page_count) // 2
    return (yield from attempt(1, mid)) + (yield from attempt(mid + 1, page_count))


def _merge_parts(parts: list[tuple[_PdfPart, _Phase1Run]]) -> _Phase1Run:
    """Combine the phase-1 results of a document's parts into one, as if the
    whole document had been read at once: page numbers go back to the
    document's, values merge in document order (see :func:`_merge_part_values`),
    the first part to describe an array's layout wins, and the counts add up."""
    values: dict[str, Any] = {}
    merged_layout: dict[str, Any] = {}
    tool_calls = chunk_calls = 0
    for part, (args, part_tool_calls, part_chunk_calls) in parts:
        part_values = args["values"]
        _offset_page_numbers(part_values, part.first_page - 1)
        _merge_part_values(values, part_values)
        part_layout = args.get("layout")
        if isinstance(part_layout, dict):
            for key, value in part_layout.items():
                merged_layout.setdefault(key, value)
        tool_calls += part_tool_calls
        chunk_calls += part_chunk_calls
    merged: dict[str, Any] = {"values": values}
    if merged_layout:
        merged["layout"] = merged_layout
    return merged, tool_calls, chunk_calls


def _offset_page_numbers(values: Any, offset: int) -> None:
    """Shift every location's ``page_number`` by *offset*, in place: a part's
    values number its own PDF's pages from 1. A leaf's ``evidence`` items
    (a flat ``page_number``, or ``locations``) shift with it."""
    if not offset:
        return

    def shift(holder: Any) -> None:
        if not isinstance(holder, dict):
            return
        page = holder.get("page_number")
        if isinstance(page, int) and not isinstance(page, bool):
            holder["page_number"] = page + offset

    if isinstance(values, dict):
        if "text" in values:
            locations = values.get("locations")
            for loc in locations if isinstance(locations, list) else []:
                shift(loc)
            evidence = values.get("evidence")
            for item in evidence if isinstance(evidence, list) else []:
                shift(item)
                item_locations = item.get("locations") if isinstance(item, dict) else None
                for loc in item_locations if isinstance(item_locations, list) else []:
                    shift(loc)
            return
        for value in values.values():
            _offset_page_numbers(value, offset)
    elif isinstance(values, list):
        for value in values:
            _offset_page_numbers(value, offset)


def _renumber_page_words(result: dict[str, Any], page: int) -> None:
    """Show a ``get_page_words`` result under the page number the model asked
    for (its part's numbering), in place."""
    result["page"] = page
    for word in result.get("words", []):
        location = word.get("location")
        if isinstance(location, dict):
            location["page_number"] = page


def _is_empty_leaf(value: Any) -> bool:
    """A leaf or scalar that carries no value."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict) and "text" in value:
        text = value.get("text")
        return text is None or (isinstance(text, str) and not text.strip())
    return False


def _rebase_input(raw: str, base: dict[str, Any]) -> str:
    """:func:`_rebase_ref` for an ``inputs`` entry, which may carry the text
    it cites after ``=`` (``"LineItems[0].Amount=$12.75"``)."""
    path_text, sep, cited = raw.partition("=")
    rebased = _rebase_ref(path_text.strip(), base)
    return f"{rebased}{sep}{cited}" if sep else rebased


def _rebase_ref(ref: str, base: dict[str, Any]) -> str:
    """Rewrite a later part's input path for the merged tree: an
    index into an array the merge appends to *base*'s copy of it moves past
    *base*'s entries."""
    path = parse_path(ref)
    if path is None:
        return ref
    cur: Any = base
    for i, seg in enumerate(path):
        if isinstance(seg, int):
            if isinstance(cur, list):
                return path_to_str((*path[:i], seg + len(cur), *path[i + 1 :]))
            return ref
        if not isinstance(cur, dict):
            return ref
        cur = cur.get(seg)
    return ref


def _merge_tree(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        current = base.get(key)
        if isinstance(current, list) and isinstance(value, list):
            current.extend(value)
        elif (
            isinstance(current, dict)
            and isinstance(value, dict)
            and "text" not in current
            and "text" not in value
        ):
            _merge_tree(current, value)
        elif key not in base or _is_empty_leaf(current):
            base[key] = value


def _merge_part_values(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Merge one part's values into those of the parts before it, in place on
    *base*: arrays concatenate (each part submits only its own pages' entries,
    in document order), objects merge recursively, and a leaf keeps the
    earliest part's value unless that one is empty. *extra*'s ``inputs`` paths
    (and 0.7 ``derived_from``) are first rebased onto the concatenated arrays,
    keeping any ``=cited text``. Returns *base*."""
    for _path, leaf in walk_all_leaves(extra):
        for key in ("inputs", "derived_from"):
            refs = leaf.get(key)
            if isinstance(refs, list):
                leaf[key] = [_rebase_input(r, base) if isinstance(r, str) else r for r in refs]
    _merge_tree(base, extra)
    return base


def _apply_append_entries(acc_args: dict[str, Any] | None, args: dict[str, Any]) -> dict[str, Any]:
    """Apply one ``append_entries`` call to the accumulated values; the
    returned dict is the tool result (an ``error`` key means the call was
    rejected — the model sees it and can correct course; the run only fails
    on protocol-independent problems)."""
    if acc_args is None:
        return {
            "error": (
                f"no values submitted yet — call {_TOOL_SUBMIT_VALUES} with done=false "
                f"before {_TOOL_APPEND_ENTRIES}"
            )
        }
    raw_path = args.get("path")
    entries = args.get("entries")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return {"error": "append_entries requires a dotted 'path' string"}
    if not isinstance(entries, list):
        return {"error": "append_entries requires an 'entries' array"}
    path = parse_path(raw_path.strip())
    if path is None:
        return {"error": f"could not parse path {raw_path!r}"}
    target = get_at_path(acc_args["values"], path)
    if target is None:
        return {
            "error": (
                f"path {raw_path!r} does not exist in the submitted values; include the "
                "array (with its first entries) in submit_values before appending to it"
            )
        }
    if not isinstance(target, list):
        return {"error": f"path {raw_path!r} is not an array"}
    target.extend(entries)
    return {"recorded": len(entries), "total_entries": len(target)}


def _append_entries_tool() -> dict[str, Any]:
    """The chunked-submission continuation tool: extend an array submitted by
    an earlier ``submit_values(done=false)`` call. Entries are validated
    code-side against the docset vocabulary after the run (the per-path item
    schema can't be expressed in one static tool spec)."""
    return {
        "type": "function",
        "function": {
            "name": _TOOL_APPEND_ENTRIES,
            "description": (
                "Append the next batch of array entries to the values tree "
                "submitted by an earlier submit_values call with done=false. "
                "Entries must follow the same item structure as the array in "
                "the schema, in document order, without repeating entries "
                "already submitted. Set done=true on the final call of the "
                "whole extraction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Dotted path of the array to extend, e.g. "
                            "'Transactions' or 'Sections[0].LineItems'."
                        ),
                    },
                    "entries": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "The next batch of array entries.",
                    },
                    "done": {
                        "type": "boolean",
                        "description": (
                            "True when this call completes the extraction; "
                            "false/omitted when more calls follow."
                        ),
                    },
                },
                "required": ["path", "entries"],
            },
        },
    }


@dataclass
class Phase1Result:
    """Phase-1 progress and outcome for one file.

    Passed INTO :func:`phase1_steps` and returned by it, because it is also the
    state that must survive a re-run: the permissive-schema fallback restarts
    the generator after a refused request, and the truncation latch
    (``chunked`` / ``truncated_retries``) has to carry across that restart.

    ``chunk_calls`` and ``tool_calls`` describe the attempt that SUCCEEDED
    (0 until one does); ``values`` / ``layout`` are that attempt's merged,
    vocabulary-pruned payload. ``totals`` accumulates cost/token usage across
    every attempt, including failed ones, ``counters["envelope_repairs"]``
    the ``submit_values`` envelopes repaired in any attempt (see
    :func:`_repair_submit_values_args`), and ``counters["no_tool_call_retries"]``
    the text-only turns retried with a reminder. ``pdf_parts`` lists the
    ``(first_page, last_page)`` of each part when the document was sent in
    page ranges (empty when it went whole).

    ``refusal`` is set when the successful attempt submitted keys of which
    nothing fits the schema. :func:`phase1_steps` does not raise it: the caller
    first records what phase 1 did (tool calls, layout, duration), then raises
    it as :class:`ValuesExtractionFailed`.
    """

    tool_schema_mode: str = "inlined"
    chunked: bool = False
    truncated_retries: int = 0
    chunk_calls: int = 0
    tool_calls: int = 0
    values: dict[str, Any] = field(default_factory=dict)
    layout: dict[str, Any] | None = None
    totals: dict[str, Any] = field(default_factory=lambda: _empty_totals())
    counters: dict[str, int] = field(
        default_factory=lambda: {"envelope_repairs": 0, "no_tool_call_retries": 0}
    )
    pdf_parts: list[tuple[int, int]] = field(default_factory=list)
    refusal: str | None = None


def phase1_steps(
    *,
    workspace: Workspace,
    file_id: str,
    run: Phase1Result,
    pdf_bytes: bytes,
    messages_factory: Callable[..., list[dict[str, Any]]],
    tool_schema: dict[str, Any],
    vocab: Vocabulary,
    model: str,
    api_key: str | None,
    api_base: str | None,
    max_tool_iters: int,
    reasoning_effort: str | None = _VALUES_REASONING_EFFORT,
) -> LLMSteps[Phase1Result]:
    """Phase 1 for ONE file, as an :data:`~dgml_core.llm.LLMSteps` generator.

    Yields every phase-1 request in order and returns *run* filled in. The
    chunked protocol (``submit_values`` with ``done: false`` then
    ``append_entries``) and the truncation retry (a ``finish_reason='length'``
    turn restarts the attempt in chunked mode with the mandatory directive) are
    simply more steps of the same generator.

    Two one-way latches, each enabled by the failure it answers, restart the
    attempt; since a latch only ever flips off → on and its handler re-raises
    once set, at most three attempts run per PDF sent:

    * a ``finish_reason='length'`` turn → chunked submission with the mandatory
      directive (``run.chunked``); a second truncation fails the file;
    * the provider refusing the inlined tool schema ("too many states",
      Gemini's constrained decoder) → the permissive object parameter plus
      code-side vocabulary pruning (``run.tool_schema_mode``). That refusal is
      an executor failure the driver throws in at the pending ``yield``, so
      every driver gets the fallback without doing anything.

    When the provider refuses the whole *pdf_bytes* as too large, the document
    is sent in page ranges instead (see :func:`_phase1_in_parts`), the parts'
    results merged as one, and ``run.pdf_parts`` set to their page ranges.

    *messages_factory* is called as ``messages_factory(chunked=..., pdf=...,
    part=...)`` for each attempt and must return a fresh list (the attempt
    appends to it); *part* is ``None`` for the whole document.
    *tool_schema* is the inlined ``submit_values`` parameter; the permissive
    one replaces it once that latch is set. *reasoning_effort* is the
    configured ``grounded.values_reasoning_effort`` (``None`` sends none).
    Drive it with the config from :func:`_phase1_llm_config` (same effort);
    any failure it does not recover from surfaces as
    :class:`ValuesExtractionFailed` (``"extraction call failed: <Type>:
    <message>"`` for a failed request). *run* may already carry latches from an
    earlier run and they are honored.
    """

    def _run(pdf: bytes, part: _PdfPart | None) -> LLMSteps[_Phase1Run]:
        while True:
            values_param = (
                _PERMISSIVE_VALUES_PARAM if run.tool_schema_mode == "permissive" else tool_schema
            )
            try:
                return (
                    yield from _phase1_attempt_steps(
                        workspace=workspace,
                        file_id=file_id,
                        messages=messages_factory(chunked=run.chunked, pdf=pdf, part=part),
                        tools=_phase1_tools(values_param, chunked=run.chunked),
                        chunked=run.chunked,
                        model=model,
                        api_key=api_key,
                        api_base=api_base,
                        max_tool_iters=max_tool_iters,
                        totals=run.totals,
                        vocab=vocab,
                        counters=run.counters,
                        reasoning_effort=reasoning_effort,
                        page_offset=part.first_page - 1 if part is not None else 0,
                    )
                )
            except _OutputTruncated as exc:
                if run.chunked:
                    raise ValuesExtractionFailed(
                        f"{exc} The chunked-submission retry was truncated too; "
                        "consider a model with a higher output ceiling or "
                        "splitting the document."
                    ) from exc
                run.chunked = True
                run.truncated_retries += 1
            except ValuesExtractionFailed as exc:
                if run.tool_schema_mode == "permissive" or not _is_tool_schema_too_large(exc):
                    raise
                # Gemini's constrained decoder rejected the inlined schema; the
                # model still sees the full schema in the user prompt.
                run.tool_schema_mode = "permissive"

    # The whole PDF goes in one request, as it always has. Only when the
    # provider refuses that request as too large is the document split into
    # page ranges, halving until each part is accepted.
    try:
        args, tool_calls, chunk_calls = yield from _run(pdf_bytes, None)
    except ValuesExtractionFailed as exc:
        if not is_request_too_large(exc):
            raise
        parts = yield from _phase1_in_parts(
            workspace, file_id, pdf_bytes, model=model, run=_run, refused=exc
        )
        args, tool_calls, chunk_calls = _merge_parts(parts)
        run.pdf_parts = [(p.first_page, p.last_page) for p, _ in parts]
    run.chunk_calls = chunk_calls
    run.tool_calls = tool_calls
    layout_ = args.get("layout") or None
    run.layout = layout_ if isinstance(layout_, dict) else None
    # Keys that name no schema root, and roots whose value has the wrong
    # shape, are dropped at serialization (the writer walks vocab.roots).
    # A tree that had keys but keeps nothing would land as an empty
    # <dg:extraction/> reported as success: flag it so the caller refuses it
    # and the failure is visible. An empty tree, or one whose roots are all
    # null, stays a legal "nothing found" outcome (the serializer reads null as
    # "not extracted"), and leaf internals (a leaf without text, a collection
    # of malformed entries) stay the serializer's business, as before.
    submitted_keys = sorted(k for k, v in args["values"].items() if v is not None)
    kept = _prune_to_vocabulary(args["values"], vocab)
    if submitted_keys and not kept:
        shown = ", ".join(repr(k) for k in submitted_keys[:6])
        if len(submitted_keys) > 6:
            shown += f" and {len(submitted_keys) - 6} more"
        run.refusal = (
            f"{_TOOL_SUBMIT_VALUES!r} returned nothing that fits the schema: got keys "
            f"{shown}, expected roots {sorted(tag.name for tag in vocab.roots)} "
            "shaped as the schema defines; refusing to write an empty extraction"
        )
    # Enforce the vocabulary code-side on exactly the paths whose payload
    # never met a provider-side shape check: permissive mode (the values
    # parameter was a bare object) and chunked mode (append_entries
    # batches are validated only as generic objects). The inlined
    # single-shot path already got additionalProperties:false at the API
    # layer, and pruning it too would discard off-schema values *before*
    # phases 2/3 rather than at serialization — a behavior change beyond
    # what these fallbacks need.
    if run.tool_schema_mode == "permissive" or run.chunked:
        args["values"] = kept
    run.values = args["values"]
    return run


_ENVELOPE_KEYS = frozenset({"values", "layout", "done"})


def _repair_submit_values_args(
    args: dict[str, Any], vocab: Vocabulary, *, chunked: bool = False
) -> tuple[dict[str, Any], str | None]:
    """Undo the ways a model mis-shapes a ``submit_values`` call, or leave it alone.

    Three shapes have been captured in the wild, all from Anthropic models on
    ordinary one-page invoices (dgml-io/dgml#154, #150). The tree inside each
    is complete and correct; only the envelope around it is wrong:

    * ``{"values": {"values": {...}, "layout": {...}}}`` -- the tool's own
      parameter envelope repeated one level down, with ``values`` the only
      top-level key.
    * ``{"values": {"values": {...}}, "layout": {...}}`` -- the same, with
      ``layout`` left at the top.
    * ``{"values": "{\"values\": {...}}, \"layout\": {...} }"}`` -- the
      envelope serialized as a JSON *string*, usually with the outer object
      closed one brace early so the rest of the envelope trails it as text
      (see :func:`_decode_string_envelope`).

    Left alone, the first two pass every later phase (the nested leaves ground
    fine) and then serialize as an empty ``<dg:extraction/>`` reported as a
    success; the third raises.

    The decision is by shape, not by key name: a candidate tree *fits* when
    :func:`_prune_to_vocabulary` keeps something of it, the same check that
    decides what serializes. A ``values`` object that fits is never touched,
    whatever else it carries, so a well-formed tree always wins; a repair
    happens only when the object as sent fits nothing and exactly one
    candidate (one level down, or inside the string) does. When a schema
    itself names a root ``values`` or ``layout`` the readings can collide: a
    nested envelope then looks well-formed and is left alone, and a string
    both readings of which fit is refused rather than guessed; either call
    behaves as before this repair existed. Fitting is the vocabulary's
    root-and-shape check, not a leaf check, so a leaf missing its ``text``
    is the serializer's business as before. One envelope needs no fit: an
    inner tree that is empty (a legal "nothing found") is recognized when
    the wrapper around it carries only the envelope's own keys. Returns the
    repaired ``args`` and a short label of the repair, or ``(args, None)``.
    """

    def fits(candidate: Any) -> bool:
        return isinstance(candidate, dict) and bool(_prune_to_vocabulary(candidate, vocab))

    def wraps_an_empty_tree(candidate: dict[str, Any]) -> bool:
        inner = candidate.get("values")
        return isinstance(inner, dict) and not inner and set(candidate) <= _ENVELOPE_KEYS

    values = args.get("values")
    if fits(values):
        return args, None
    repaired: dict[str, Any]
    if isinstance(values, str) and values.lstrip().startswith("{"):
        envelope = _decode_string_envelope(values)
        if envelope is None:
            return args, None
        inner_fits, outer_fits = fits(envelope.get("values")), fits(envelope)
        if (inner_fits or wraps_an_empty_tree(envelope)) and not outer_fits:
            repaired = {"values": envelope["values"]}
            _carry_envelope_fields(repaired, args, envelope, chunked=chunked)
        elif outer_fits and not inner_fits:
            # The string held the tree itself. A layout or done beside its
            # roots is carried out as an envelope field; in the tree it is
            # not a root and the serializer ignores it, as before.
            repaired = {"values": envelope}
            _carry_envelope_fields(repaired, args, envelope, chunked=chunked)
        else:
            return args, None
        return repaired, "string"
    if isinstance(values, dict) and (fits(values.get("values")) or wraps_an_empty_tree(values)):
        repaired = {"values": values["values"]}
        _carry_envelope_fields(repaired, args, values, chunked=chunked)
        return repaired, "nested"
    return args, None


def _is_layout(candidate: Any) -> bool:
    """Whether ``candidate`` is a ``layout`` the loop can use: a dict of
    per-array descriptors of the shape :func:`_layout_param_schema`
    declares (a known ``kind``, ``columns`` a list of names when present,
    nothing else). An empty dict describes no array, so it is not one: a
    ``layout: {}`` on the call must not shadow a populated layout left
    inside the mis-shaped ``values``."""
    if not isinstance(candidate, dict) or not candidate:
        return False
    for descriptor in candidate.values():
        if not isinstance(descriptor, dict) or not set(descriptor) <= {"kind", "columns"}:
            return False
        if descriptor.get("kind") not in ("table", "free_form"):
            return False
        columns = descriptor.get("columns")
        if columns is not None and not (
            isinstance(columns, list) and all(isinstance(c, str) for c in columns)
        ):
            return False
    return True


def _carry_envelope_fields(
    repaired: dict[str, Any],
    outer: dict[str, Any],
    inner: dict[str, Any] | None,
    *,
    chunked: bool,
) -> None:
    """Carry ``layout`` and ``done`` into a repaired call, by validity.

    The call's own field wins when it is valid (a layout of known
    descriptors, a ``bool`` done); otherwise the copy the model left inside
    the mis-shaped ``values`` fills in, when it is valid. Anything else is
    dropped, and the loop's defaults apply. ``done`` is a chunked-protocol
    field: in single-shot mode it is not carried, so a stray ``done: false``
    cannot turn one submission into a continuation the model was never
    offered.
    """
    checks: list[tuple[str, Callable[[Any], bool]]] = [("layout", _is_layout)]
    if chunked:
        checks.append(("done", lambda v: isinstance(v, bool)))
    for key, valid in checks:
        outer_value = outer.get(key)
        inner_value = inner.get(key) if inner is not None else None
        if valid(outer_value):
            repaired[key] = outer_value
        elif valid(inner_value):
            repaired[key] = inner_value


def _decode_string_envelope(text: str) -> dict[str, Any] | None:
    """Decode a ``submit_values`` envelope the model serialized as a string.

    The damage seen is the outer object closed one brace early, so the rest
    of the envelope (``"layout": {...}``, ``"done": false``) trails the first
    complete object as extra text. Moving that brace to the end turns the
    whole string back into the object the model meant to send, every field
    included. Trailing text that does not read that way is not understood,
    and ``None`` leaves the call to fail loudly as before.
    """
    text = text.strip()
    try:
        lead, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return None
    if text[end:].strip():
        try:
            lead = json.loads(text[: end - 1] + text[end:])
        except ValueError:
            return None
    return lead if isinstance(lead, dict) else None


def _phase1_attempt_steps(
    *,
    workspace: Workspace,
    file_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    api_key: str | None,
    api_base: str | None,
    max_tool_iters: int,
    totals: dict[str, Any],
    vocab: Vocabulary,
    counters: dict[str, int],
    chunked: bool = False,
    reasoning_effort: str | None = _VALUES_REASONING_EFFORT,
    page_offset: int = 0,
) -> LLMSteps[tuple[dict[str, Any], int, int]]:
    """One phase-1 attempt: a multi-turn loop until the model finishes submitting.

    Returns ``(submit_args, tool_calls_run, chunk_calls)``: ``submit_args``
    carries the merged ``values`` tree plus optional sibling fields like
    ``layout``; ``chunk_calls`` counts the submission calls (1 for the
    ordinary single ``submit_values``, more when the model used the chunked
    protocol).
    Mutates ``totals`` by adding cost/token deltas from every litellm call
    so the surrounding ``extract_values`` records a single usage row across
    both phases, and ``counters["envelope_repairs"]`` for every submission
    whose argument envelope had to be unwrapped first (see
    :func:`_repair_submit_values_args`, which decides against ``vocab``),
    and ``counters["no_tool_call_retries"]`` for every turn that ended with
    no tool call and was retried; all survive a loop that fails afterwards.

    Two submission protocols:

    * **Single-shot** (ordinary documents): one ``submit_values`` call, its
      ``done`` flag omitted or true.
    * **Chunked** (very large outputs): ``submit_values`` with
      ``done: false`` (scalars + first array batches), then repeated
      ``append_entries`` calls extending an array at a dotted path, the
      last one carrying ``done: true``. Each non-terminal call gets an
      acknowledging tool result so the model continues the same turn loop.

    ``chunked`` must match the tool set the caller built (see
    :func:`_phase1_tools`): the chunked protocol is honored only when it is
    set, so an ``append_entries`` call that arrives without it is treated as
    an unknown tool. That keeps the caller's invariant true by construction —
    if appends ran, chunked mode was on, so the payload gets pruned.

    A turn that stops with ``finish_reason == "length"`` raises
    :class:`_OutputTruncated` so :func:`phase1_steps` can retry with an explicit
    chunking directive.

    Every provider but Anthropic gets ``tool_choice="required"`` on every
    call, so a text-only turn cannot happen there, unless litellm's global
    ``drop_params`` silently removes the parameter for a provider it believes
    lacks it (the call then runs on auto, as it did before "required"). A call that an endpoint
    rejects as an invalid request is resent once on auto; when that succeeds,
    the endpoint is remembered as refusing ``"required"`` (warned about once)
    and this and later attempts against it run on auto. On auto (Anthropic,
    see below, or such an endpoint) a text-only turn that stopped normally
    (``finish_reason`` ``"stop"``, or none reported) is retried up to
    :data:`_MAX_NO_TOOL_CALL_RETRIES` times per attempt, outside the
    ``max_tool_iters`` budget: the reply is kept in the history and a
    reminder to call ``submit_values`` follows it (or, for a chunked run that
    already recorded part of its submission, to continue with
    ``append_entries``, since a resent full tree would duplicate its entries).
    A retry is never forced, so the turn that produces the values keeps its
    thinking budget. A text-only turn that did not stop normally (a content
    filter, a refusal stop) fails at once. Every failure quotes the reply's
    ``finish_reason``, whether it was reasoning only, and its opening text.

    *page_offset* is non-zero when the PDF sent is a page range of the file
    that starts after page *page_offset*. The model numbers that PDF's pages
    from 1, so a ``get_page_words`` lookup reads page ``page + page_offset``
    and the result is shown under the model's own number. The values it
    submits keep the PDF's numbering; the caller shifts them.
    """
    # Phase 1 leaves Anthropic on tool_choice="auto" (omitted on the wire)
    # because any forced choice, "required" included, makes the wrapper drop
    # reasoning_effort for Anthropic, which rejects thinking with forced tool
    # use. Auto keeps extended thinking on for Claude. That also makes this
    # the only call site where the reasoning budget has any effect on Claude,
    # which is why :data:`_VALUES_REASONING_EFFORT` applies here and nowhere
    # else. Other providers keep their reasoning under a forced choice, so
    # they get "required" and can never end a turn without a tool call.
    # ``reasoning_effort`` is that constant unless the workspace set
    # ``grounded.values_reasoning_effort`` (``None`` = send none).
    llm_config = _phase1_llm_config(model, api_key, api_base, reasoning_effort)

    tool_calls_run = 0
    chunk_calls = 0
    acc_args: dict[str, Any] | None = None  # accumulated submit_values args
    no_tool_call_retries = 0
    endpoint = (model, api_base, reasoning_effort)
    tool_choice = (
        None
        if is_anthropic_model(model) or endpoint in _REQUIRED_TOOL_CHOICE_REJECTED
        else "required"
    )

    def _ack(call_id: str, name: str, payload: dict[str, Any]) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(payload),
            }
        )

    turns = 0
    while turns < max_tool_iters:
        try:
            # Always cached: the system prompt and the schema block ahead of the
            # per-file PDF are byte-identical for every file in the docset, so
            # each file after the first reads that prefix instead of re-sending
            # it. ``steps_with_tools`` no-ops the marker for non-Anthropic models.
            try:
                result = yield from steps_with_tools(
                    llm_config, messages=messages, tools=tools, tool_choice=tool_choice, cache=True
                )
            except Exception as exc:
                # Some endpoints refuse "required" (an OpenAI-compatible server
                # without forced tool choice, or Claude behind an alias that
                # is_anthropic_model cannot recognise, which then rejects
                # thinking under a forced choice). Resend once on auto, which
                # is what phase 1 sent before "required". Only a request
                # rejection qualifies: a timeout or rate limit would only be
                # repeated, a schema overflow belongs to the caller's
                # permissive-schema fallback, and a body too large for the
                # provider belongs to the page-range split. A failing resend raises its own
                # error, the one the auto request alone would have produced.
                if not (
                    tool_choice == "required"
                    and is_request_rejection(exc)
                    and not _is_tool_schema_too_large(exc)
                    and not is_request_too_large(exc)
                ):
                    raise
                result = yield from steps_with_tools(
                    llm_config, messages=messages, tools=tools, cache=True
                )
                # The same request succeeded once tool choice was left to the
                # model, so "required" was the cause: stay on auto for this
                # attempt and start later ones there.
                _remember_required_tool_choice_rejected(endpoint, exc)
                tool_choice = None
        except Exception as exc:
            raise ValuesExtractionFailed(
                f"{_PHASE1_CALL_FAILED}: {type(exc).__name__}: {exc}"
            ) from exc

        add_partial(totals, with_tier(result.usage, result.response, None))

        if result.finish_reason == "length":
            # Without this check a clipped submit_values surfaces as an
            # opaque malformed-JSON error; name the real cause instead. The
            # cap reported is the effective one — the request is clamped to
            # the model's own ceiling when that is lower than what we ask for.
            effective_cap = min(
                _DEFAULT_MAX_COMPLETION_TOKENS,
                model_max_output_tokens(model, api_base) or _DEFAULT_MAX_COMPLETION_TOKENS,
            )
            raise _OutputTruncated(
                "model output hit the completion-token cap "
                f"(finish_reason='length', max_completion_tokens={effective_cap}, "
                f"the ceiling for {model!r}) before finishing its tool call — "
                "the extraction output was truncated."
            )

        if not result.tool_calls:
            # Only a normal stop is worth a reminder: a content filter or a
            # refusal stop would end the same way again, and each retry
            # resends the document.
            # No finish_reason at all is common on OpenAI-compatible servers
            # and says nothing abnormal happened, so it is retried too.
            if (
                result.finish_reason not in ("stop", None)
                or no_tool_call_retries >= _MAX_NO_TOOL_CALL_RETRIES
            ):
                raise ValuesExtractionFailed(
                    "model returned no tool call; the run is required to end "
                    f"with a {_TOOL_SUBMIT_VALUES!r} call carrying the final values "
                    f"(after {no_tool_call_retries} retries on this attempt; last "
                    f"reply: {_describe_text_reply(result)})"
                )
            no_tool_call_retries += 1
            counters["no_tool_call_retries"] += 1
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "file %s: phase-1 reply had no tool call (%s); retry %d of %d",
                    file_id,
                    _describe_text_reply(result),
                    no_tool_call_retries,
                    _MAX_NO_TOOL_CALL_RETRIES,
                )
            # Keep the reply so the model sees what it said; an empty assistant
            # turn is rejected by some providers, so it is dropped instead.
            if isinstance(result.content, str) and result.content.strip():
                messages.append({"role": "assistant", "content": result.content})
            # A chunked run that already recorded a done=false submission is
            # mid-sequence: lists merge by concatenation, so a resent full tree
            # would duplicate every entry already recorded.
            nudge = (
                PromptKey.VALUES_PHASE1_NUDGE_CONTINUE
                if chunked and acc_args is not None
                else PromptKey.VALUES_PHASE1_NUDGE_SUBMIT
            )
            messages.append({"role": "user", "content": prompt(nudge)})
            # A retry does not spend the turn budget, so running out of turns
            # never cuts one short; _MAX_NO_TOOL_CALL_RETRIES bounds them.
            continue

        turns += 1
        messages.append(_serialize_assistant_message(result.message))

        for call in result.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValuesExtractionFailed(
                    f"model produced malformed JSON arguments for {name!r}: {exc}"
                ) from exc

            if name == _TOOL_SUBMIT_VALUES:
                args, repaired = _repair_submit_values_args(args, vocab, chunked=chunked)
                if repaired is not None:
                    counters["envelope_repairs"] += 1
                values = args.get("values")
                if not isinstance(values, dict):
                    raise ValuesExtractionFailed(
                        f"{_TOOL_SUBMIT_VALUES!r} was called without a 'values' object"
                    )
                chunk_calls += 1
                if acc_args is None:
                    acc_args = args
                else:
                    acc_args["values"] = _merge_values(acc_args["values"], values)
                    if isinstance(args.get("layout"), dict):
                        merged_layout = acc_args.get("layout")
                        if isinstance(merged_layout, dict):
                            merged_layout.update(args["layout"])
                        else:
                            acc_args["layout"] = args["layout"]
                assert acc_args is not None
                if args.get("done", True):
                    return acc_args, tool_calls_run, chunk_calls
                _ack(
                    call.id,
                    name,
                    {
                        "recorded": True,
                        "next": (
                            f"continue with {_TOOL_APPEND_ENTRIES} (or another "
                            f"{_TOOL_SUBMIT_VALUES}); set done=true on the last call"
                        ),
                    },
                )
                continue

            if name == _TOOL_APPEND_ENTRIES and chunked:
                chunk_calls += 1
                outcome = _apply_append_entries(acc_args, args)
                if "error" not in outcome and args.get("done", False):
                    assert acc_args is not None  # _apply_append_entries verified
                    return acc_args, tool_calls_run, chunk_calls
                _ack(call.id, name, outcome)
                continue

            if name == _TOOL_GET_PAGE_WORDS:
                tool_calls_run += 1
                page = args.get("page")
                if not isinstance(page, int):
                    raise ValuesExtractionFailed(f"{name!r} requires integer 'page' argument")
                start_idx = args.get("start_idx")
                end_idx = args.get("end_idx")
                try:
                    tool_result: dict[str, Any] = get_page_words(
                        workspace,
                        file_id,
                        page + page_offset if page >= 1 else page,
                        start_idx if isinstance(start_idx, int) else None,
                        end_idx if isinstance(end_idx, int) else None,
                    )
                    if page_offset:
                        _renumber_page_words(tool_result, page)
                except Exception as exc:
                    # Bubble tool errors back to the model as a tool result;
                    # don't fail the whole extraction on a bad lookup.
                    tool_result = {"error": f"{type(exc).__name__}: {exc}"}
                _ack(call.id, name, tool_result)
                continue

            raise ValuesExtractionFailed(f"model called unknown tool: {name!r}")

    raise ValuesExtractionFailed(
        f"extraction exceeded max_tool_iters={max_tool_iters} "
        f"without producing a {_TOOL_SUBMIT_VALUES!r} call"
    )


def _remember_required_tool_choice_rejected(
    endpoint: tuple[str, str | None, str | None], exc: BaseException
) -> None:
    """Record that *endpoint* refuses ``tool_choice="required"`` and warn the
    first time.

    On auto a model may end a turn without a tool call, which the reminder
    retries cover but cannot rule out, so the fallback is worth surfacing."""
    if endpoint in _REQUIRED_TOOL_CHOICE_REJECTED:
        return
    _REQUIRED_TOOL_CHOICE_REJECTED.add(endpoint)
    model = endpoint[0]
    logger.warning(
        "values model %r rejected tool_choice='required' (%s: %s); phase-1 "
        "extraction falls back to auto tool choice for it",
        model,
        type(exc).__name__,
        str(exc)[:300],
    )


def _describe_text_reply(result: CallResult) -> str:
    """Summarize a text-only model reply for an error or log message: the
    opening of its text (whitespace collapsed), then :func:`_no_text_detail`'s
    ``finish_reason`` and whether the reply was reasoning only, which is what
    tells a model that spent its budget thinking apart from an empty reply."""
    content = result.content
    text = " ".join(content.split()) if isinstance(content, str) else ""
    if len(text) > _NO_TOOL_CALL_REPLY_PREVIEW_CHARS:
        text = text[:_NO_TOOL_CALL_REPLY_PREVIEW_CHARS] + "..."
    shown = json.dumps(text) if text else "no text"
    return shown + _no_text_detail(result.response)


def _normalized_evidence(raw: Any) -> list[dict[str, Any]]:
    """The usable ``evidence`` items of a phase-1 leaf, as ``{text,
    locations}``: non-empty verbatim ``text`` and a page to ground it on —
    the tool's flat ``page_number``, or a ``locations`` list. Anything else
    can't be grounded, so it can't become a ``dg:chunk`` a verifier could
    crop."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        locations = item.get("locations")
        pages = (
            [
                loc
                for loc in locations
                if isinstance(loc, dict) and isinstance(loc.get("page_number"), int)
            ]
            if isinstance(locations, list)
            else []
        )
        if not pages and isinstance(item.get("page_number"), int):
            # An empty or unusable ``locations`` must not hide the flat page.
            pages = [{"page_number": item["page_number"]}]
        if pages:
            out.append({"text": text, "locations": pages})
    return out


def _normalize_leaf_provenance(values: Any) -> int:
    """Enforce the DGML 0.71 provenance invariants on phase-1 leaves, in place.

    Returns the number of unsupported claims it demoted: values the model
    marked as calculated or inferred (or 0.7 ``computed``) without naming
    anything they used.

    The merged ``extracted_value`` leaf shape lets a sloppy model return
    combinations the format can't express. Phases 2/3 and the XML serializer
    rely on these invariants, so this pass restores them:

    * the 0.7 keys are read as their 0.71 form — ``derived_from`` becomes
      ``inputs`` and ``computed`` is dropped;
    * ``inputs`` is a non-empty list of strings, or absent;
    * ``evidence`` (printed text the value used that the schema doesn't
      extract) is a non-empty list of ``{text, locations}`` items, each with
      a page to ground on, or absent;
    * ``method`` is one of ``calculated``/``inferred``/``defaulted``, or absent;
    * a leaf with at least one usable location is printed (in full, or in part
      when it also has ``inputs``) and never carries a ``method``;
    * a leaf with no usable location loses its ``locations`` key. A
      ``defaulted`` one loses its ``inputs`` and ``evidence``; one with
      either and no method is ``inferred`` (the safe reading);
      ``calculated``/``inferred`` with neither lose the method, leaving an
      unsupported value (counted in the return value), which is not written;
    * an empty-string ``value`` is removed.
    """
    if isinstance(values, list):
        return sum(_normalize_leaf_provenance(entry) for entry in values)
    if not isinstance(values, dict):
        return 0
    if isinstance(values.get("text"), str):
        # A leaf (only leaves carry a lowercase "text" key — tag names are
        # PascalCase). Normalize in place; never descend further.
        if values.get("value") == "":
            del values["value"]
        legacy_refs = values.pop("derived_from", None)
        claimed = bool(values.pop("computed", None))
        # An empty/null ``inputs`` must not hide 0.7 ``derived_from`` paths.
        refs = values.get("inputs") or legacy_refs
        if isinstance(refs, str):
            # A lone path the model didn't wrap in a list.
            refs = [refs]
        inputs = (
            [r for r in refs if isinstance(r, str) and r.strip()] if isinstance(refs, list) else []
        )
        if inputs:
            values["inputs"] = inputs
        else:
            values.pop("inputs", None)
        evidence = _normalized_evidence(values.get("evidence"))
        if evidence:
            values["evidence"] = evidence
        else:
            values.pop("evidence", None)
        if values.get("method") not in METHODS:
            values.pop("method", None)
        locations = values.get("locations")
        grounded = isinstance(locations, list) and any(
            isinstance(loc, dict) and isinstance(loc.get("page_number"), int) for loc in locations
        )
        if grounded:
            values.pop("method", None)
            return 0
        values.pop("locations", None)
        method = values.get("method")
        used = bool(inputs or evidence)
        if method == METHOD_DEFAULTED:
            values.pop("inputs", None)
            values.pop("evidence", None)
            return 0
        if used and method is None:
            values["method"] = METHOD_INFERRED
        elif not used and method is not None:
            del values["method"]
            return 1
        return 1 if claimed and not used else 0
    return sum(_normalize_leaf_provenance(child) for child in values.values())


def _drop_bboxes_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the schema with ``bounding_box`` stripped from the
    leaf definition's ``locations[]`` (``extracted_value``, plus the legacy
    ``grounded_field`` for older exported schemas). ``evidence`` items carry
    a bare ``page_number`` already.

    Phase-1 extraction shows the model this slimmer shape so it focuses
    on getting the text and page numbers right without the bbox burden.
    Schemas without a conventional leaf definition are returned unchanged.
    """
    out = copy.deepcopy(schema)
    defs = out.get("definitions")
    if not isinstance(defs, dict):
        return out
    for def_name in ("extracted_value", "grounded_field"):
        leaf_def = defs.get(def_name)
        if not isinstance(leaf_def, dict):
            continue
        _drop_bbox_from_locations(leaf_def.get("properties", {}).get("locations"))
    return out


def _drop_bbox_from_locations(locs: Any) -> None:
    """Strip ``bounding_box`` from a ``locations`` array schema, in place."""
    if not isinstance(locs, dict):
        return
    items = locs.get("items", {})
    props = items.get("properties")
    if isinstance(props, dict) and "bounding_box" in props:
        del props["bounding_box"]
    required = items.get("required")
    if isinstance(required, list):
        items["required"] = [r for r in required if r != "bounding_box"]


def _values_phase1_user_prompt(schema: dict[str, Any], guidance: str | None = None) -> str:
    """The phase-1 user prompt: instructions + schema, plus the docset's
    extraction guidance (``extraction-guidance.md``) when one is set.

    Both parts are byte-identical for every file in a docset, which is what
    makes this block worth a cache breakpoint. The chunked-submission
    directive is deliberately NOT included: it is appended as its own block by
    the caller, so that adding it on a retry leaves this prefix — and the
    cache entry written for it — untouched."""
    text = prompt(PromptKey.VALUES_PHASE1_USER).format(schema=json.dumps(schema, indent=2))
    if guidance:
        text += "\n\n" + prompt(PromptKey.VALUES_PHASE1_GUIDANCE).format(guidance=guidance.strip())
    return text


def _phase1_tools(values_schema: dict[str, Any], *, chunked: bool) -> list[dict[str, Any]]:
    """The phase-1 tool set.

    ``chunked`` gates the whole chunked-submission protocol — the ``done``
    flag on ``submit_values`` and the ``append_entries`` continuation tool.
    Single-shot is the default contract: offering a continuation tool the run
    has no reason to use invites the model to split output that fits in one
    call, and costs tool-schema tokens on every request. Chunking is an
    escalation, enabled only after an attempt was truncated (see
    :func:`extract_values`), where the prompt carries the matching protocol.
    """
    tools = [_submit_values_tool(values_schema, with_layout=True, chunked=chunked)]
    if chunked:
        tools.append(_append_entries_tool())
    return tools


def _submit_values_tool(
    values_schema: dict[str, Any], *, with_layout: bool = False, chunked: bool = False
) -> dict[str, Any]:
    """Build the ``submit_values`` tool spec.

    ``values_schema`` is the docset's own schema, already $ref-expanded
    by :func:`_expand_refs`. Inlining it as the tool's ``values``
    parameter type lets the provider's tool-call validator enforce the
    grounded_field shape (e.g. require ``page_number`` exactly — no
    ``_page_number`` typo) at the API layer, instead of relying on the
    model to follow the prompt perfectly.

    ``with_layout`` adds a sibling ``layout`` parameter for phase 1's
    use only — phase 2 reads it as a hint, phase 3 doesn't need it.
    """
    properties: dict[str, Any] = {"values": values_schema}
    if with_layout:
        properties["layout"] = _layout_param_schema()
    if chunked:
        # Only offered in chunked mode — a `done` flag on a single-shot call
        # is an invitation to split output we have room for, and every unused
        # parameter is schema the constrained decoder still has to carry.
        properties["done"] = {
            "type": "boolean",
            "description": (
                "Omit (or true) when this call carries the complete "
                "extraction; false when more array entries will follow via "
                "append_entries calls."
            ),
        }
        description = (
            "Submit the extracted values. Pass done=false when the output is "
            "too large for one call and continue with append_entries; a call "
            "with done omitted or true ends the run."
        )
    else:
        description = (
            "Submit the final extracted values. Call exactly once when "
            "extraction is complete; this ends the run."
        )
    return {
        "type": "function",
        "function": {
            "name": _TOOL_SUBMIT_VALUES,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["values"],
            },
        },
    }


def _layout_param_schema() -> dict[str, Any]:
    """JSON-Schema for the optional ``layout`` parameter on phase 1's
    submit_values call. Keys are dotted array paths (e.g.
    ``"transactions"``); values describe whether the array is laid out
    as a table (with ordered column field names) or as a free-form
    list."""
    return {
        "type": "object",
        "description": (
            "Optional per-array layout descriptors. Keys are dotted "
            "paths of arrays in the schema (e.g. 'transactions' or "
            "'company.contacts'). Values describe how the array's "
            "items are laid out on the page; phase 2 uses 'table' "
            "layouts to assign same-row cells to columns in visual "
            "left-to-right order."
        ),
        "additionalProperties": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "enum": ["table", "free_form"]},
                "columns": {
                    "type": "array",
                    "description": (
                        "For 'table' kind only: the array's leaf "
                        "fields in visual left-to-right order."
                    ),
                    "items": {"type": "string"},
                },
            },
            "required": ["kind"],
        },
    }


# Prose annotation keys the tool schema doesn't need: they live on in the
# user-prompt copy of the schema, and for a large docset schema they are most
# of the bytes — stripping them keeps the provider's constrained decoder
# (notably Gemini's, which rejects big schemas with "too many states for
# serving") within budget. `enum` is deliberately NOT here: it is load-bearing
# for constrained decoding of normalized enum values.
_ANNOTATION_KEYS = frozenset({"description", "prompt", "example", "datatype", "value_enum"})


def _strip_annotations(node: Any) -> Any:
    """Deep-copy *node* (a tool schema) without prose annotation keys."""
    if isinstance(node, dict):
        return {k: _strip_annotations(v) for k, v in node.items() if k not in _ANNOTATION_KEYS}
    if isinstance(node, list):
        return [_strip_annotations(x) for x in node]
    return node


# The fallback `values` tool parameter when the provider rejects the inlined
# docset schema: shape enforcement moves from the API layer to the prompt
# (which carries the full schema) plus code-side pruning against the
# vocabulary (_prune_to_vocabulary).
_PERMISSIVE_VALUES_PARAM: dict[str, Any] = {
    "type": "object",
    "description": (
        "The extracted values tree. MUST follow the JSON Schema given in the "
        "prompt exactly — same nesting, same field names, and the leaf value "
        "shape it defines."
    ),
}


def _is_tool_schema_too_large(exc: Exception) -> bool:
    """True when the provider rejected the request because the inlined tool
    schema overflows its constrained-decoding budget. Matched on the error
    text because litellm surfaces it as a provider BadRequest string —
    Gemini's wording is "too many states for serving"."""
    return "too many states" in str(exc).lower()


def _prune_to_vocabulary(values: dict[str, Any], vocab: Vocabulary) -> dict[str, Any]:
    """Drop keys and wrong-shape nodes the docset vocabulary doesn't define.

    The code-side stand-in for the ``additionalProperties: false`` the
    inlined tool schema enforces at the API layer — used when the provider
    rejected the inlined schema and phase 1 ran with the permissive object
    parameter. Leaf internals are left untouched (they are normalized by
    :func:`_normalize_leaf_provenance`)."""

    def prune_children(children: list[Tag], value: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for child in children:
            pruned = prune_tag(child, value.get(child.name))
            if pruned is not None:
                out[child.name] = pruned
        return out

    def prune_tag(tag: Tag, value: Any) -> Any:
        if value is None:
            return None
        if tag.kind == "field":
            return value if isinstance(value, dict) else None
        if tag.kind == "choice":
            if not isinstance(value, dict):
                return None
            if any(c.name in value for c in tag.children):
                return prune_children(tag.children, value)
            return value  # the scalar-leaf alternative
        if tag.kind == "collection":
            if not isinstance(value, list):
                return None
            item = tag.item
            entries = []
            for entry in value:
                if item is not None and item.kind == "field":
                    pruned = prune_tag(item, entry)
                elif isinstance(entry, dict):
                    pruned = prune_children(tag.children, entry)
                else:
                    pruned = None
                if pruned is not None:
                    entries.append(pruned)
            return entries
        # container
        return prune_children(tag.children, value) if isinstance(value, dict) else None

    return {
        root.name: pruned
        for root in vocab.roots
        if (pruned := prune_tag(root, values.get(root.name))) is not None
    }


def _specialize_leaf(expanded: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    """Apply a leaf $ref node's sidecar keys to its expanded definition.

    ``value_enum`` narrows the leaf's ``value`` property to the schema's
    closed token set (an ``enum``) so the provider's constrained decoder
    enforces it; every other sidecar key (``prompt``, ``example``,
    ``description``, ``datatype``) is carried as an annotation. *expanded*
    is a fresh tree built by ``_expand_refs``' walk, so mutating it here
    never touches the shared definition."""
    extras = dict(extras)
    enum_values = extras.pop("value_enum", None)
    if (
        isinstance(enum_values, list)
        and enum_values
        and isinstance(expanded.get("properties"), dict)
        and isinstance(expanded["properties"].get("value"), dict)
    ):
        expanded["properties"]["value"] = {
            **expanded["properties"]["value"],
            "enum": list(enum_values),
        }
    expanded.update(extras)
    return expanded


def _expand_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline-expand local ``#/definitions/...`` $refs in a JSON Schema,
    and tighten every ``type: object`` node by setting
    ``additionalProperties: false``.

    The schema we hand to litellm's tool-call parameter spec must be
    self-contained — provider adapters resolve $ref inconsistently and
    relative-path resolution against a sub-schema doesn't always look
    where the operator's schema expects. This walker rewrites every
    ``{"$ref": "#/definitions/X"}`` to the body of ``definitions.X``,
    recursively (so chains of defs collapse), and strips the
    ``definitions`` / ``$schema`` blocks from the resulting schema
    since they're no longer referenced.

    Why also force ``additionalProperties: false``: observed in the
    wild that Gemini's tool-call argument validation accepts extra
    properties even when ``required`` declares specific names (e.g.
    a ``locations[]`` item appearing with ``page_number`` + a fabricated
    ``bounding_2_box`` instead of the required ``bounding_box``).
    Setting ``additionalProperties: false`` on every constrained object
    node closes that hole: any property name not listed in
    ``properties`` is rejected at the API layer. We only set the flag
    when ``properties`` is present and the schema author hasn't already
    specified ``additionalProperties`` — so a deliberately
    open-ended map in the docset schema (rare, but possible) still
    works.

    A ``$ref`` node may carry sidecar keys next to the ref (``prompt``,
    ``example``, ``description``, ``datatype``, ``value_enum`` — the
    per-field annotations the RNC bridge emits). Those are merged onto
    the expanded definition so per-field guidance survives into the tool
    schema; ``value_enum`` is *specialized* into the expanded leaf's
    ``value.enum`` so the provider constrain-decodes the normalized
    value to the schema's closed token set.

    Unknown $ref forms (external URLs, non-``definitions`` JSON Pointers)
    are left untouched; the downstream provider will fail loudly on
    them rather than silently produce a wrong-shape result.
    """
    defs = schema.get("definitions") or {}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/definitions/"):
                key = ref[len("#/definitions/") :]
                target = defs.get(key)
                if isinstance(target, dict):
                    expanded = walk(target)
                    assert isinstance(expanded, dict)  # walk(dict) is a dict
                    extras = {k: v for k, v in node.items() if k != "$ref"}
                    return _specialize_leaf(expanded, extras) if extras else expanded
                return node  # unresolved — keep as-is so the model error is visible
            out: dict[str, Any] = {
                k: walk(v) for k, v in node.items() if k not in {"$schema", "definitions"}
            }
            # Tighten constrained objects after recursing so the rule
            # also applies to any object the recursion just expanded.
            if (
                out.get("type") == "object"
                and isinstance(out.get("properties"), dict)
                and "additionalProperties" not in out
            ):
                out["additionalProperties"] = False
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    result = walk(schema)
    # `walk` is typed as Any so help mypy by re-narrowing — the top
    # level of a schema is always an object, so this is correct.
    assert isinstance(result, dict)
    return result


# ---- Shared response-parsing helpers --------------------------------------


def _parse_submit_call(response: Any, *, expected_tool: str, field: str) -> dict[str, Any]:
    """Pull the single forced tool call off a litellm completion."""
    try:
        choices = response.choices
        msg = choices[0].message
        calls = list(msg.tool_calls or [])
    except (AttributeError, IndexError, TypeError) as exc:
        raise SchemaGenerationFailed(f"could not read tool call from LLM response: {exc}") from exc
    if not calls:
        raise SchemaGenerationFailed(
            f"model returned no tool call; expected one call to {expected_tool!r}"
        )
    if len(calls) > 1:
        raise SchemaGenerationFailed(
            f"model returned {len(calls)} tool calls; expected exactly one to {expected_tool!r}"
        )
    call = calls[0]
    if call.function.name != expected_tool:
        raise SchemaGenerationFailed(
            f"model called unexpected tool: {call.function.name!r} (expected {expected_tool!r})"
        )
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        raise SchemaGenerationFailed(f"model produced malformed JSON arguments: {exc}") from exc
    value = args.get(field)
    if value is None:
        raise SchemaGenerationFailed(f"model omitted required {field!r} field in tool arguments")
    return value  # type: ignore[no-any-return]


def _serialize_assistant_message(msg: Any) -> dict[str, Any]:
    """Convert a litellm assistant message (with tool_calls) back into a
    plain dict that's safe to append to the message history for the next
    turn. litellm returns OpenAI-shaped objects regardless of provider."""
    calls_out: list[dict[str, Any]] = []
    for c in msg.tool_calls or []:
        calls_out.append(
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments,
                },
            }
        )
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": calls_out,
    }


def _resolve_api_key(literal: str | None, env_name: str | None) -> str | None:
    """Resolve an API key.

    Precedence: literal value > env var lookup > ``None`` (let litellm
    fall back to its per-provider env-var conventions:
    ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, ...).

    Mutual exclusion of ``literal`` and ``env_name`` is enforced
    upstream in :func:`load_grounded_config`.
    """
    if literal is not None:
        return literal
    if env_name is None:
        return None
    import os

    key = os.environ.get(env_name)
    if not key:
        raise AuthError(
            f"environment variable ${env_name} is not set "
            "(referenced by a *_api_key_env field in config.toml[grounded])"
        )
    return key


# ---- Public re-exports (for cli.py and the FileStore hook) ---------------

__all__ = [
    "DEFAULT_MAX_TOOL_ITERS",
    "REASONING_EFFORT_CHOICES",
    "REASONING_EFFORT_PROVIDER_DEFAULT",
    "VALUES_REASONING_EFFORT_CHOICES",
    "VALUES_REASONING_EFFORT_PROVIDER_DEFAULT",
    "ExtractionResult",
    "GroundedConfig",
    "extract_values",
    "generate_schema",
    "get_page_words",
    "load_grounded_config",
]
