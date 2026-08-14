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
"""

from __future__ import annotations

import base64
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import layout
from .config import load_merged_config
from .docsets import DocSetStore
from .errors import (
    AuthError,
    FileNotFound,
    GroundedConfigInvalid,
    GroundedConfigMissing,
    SchemaGenerationFailed,
    SchemaInvalid,
    ValuesExtractionFailed,
    now_iso,
)
from .extraction_schema import (
    FIELD_DATATYPES,
    Tag,
    Vocabulary,
    field_tree_to_rnc,
    parse_rnc,
    rnc_to_json_schema,
)
from .extraction_xml import (
    check_derivations,
    check_invariants,
    count_dropped_refs,
    count_unnormalized_enum_values,
    embed_extraction_into,
    has_document_tree,
    standalone_extraction_doc,
)
from .files import FileStore
from .llm import (
    LLMConfig,
    _mark_document_cacheable,
    call_with_tools,
    is_anthropic_model,
    model_max_output_tokens,
)
from .matching import (
    UnmatchedItem,
    get_at_path,
    parse_path,
    path_to_str,
    run_phase2_matching,
    walk_computed_leaves,
)
from .models_config import ConfigSection, Tier, resolve_tiered_model
from .prompts import PromptKey
from .prompts import get as prompt
from .storage import Workspace
from .toon import encode_phase3_words
from .usage import (
    OPERATION_EXTRACT_VALUES,
    OPERATION_SCHEMA_GENERATE,
    OUTCOME_ERROR,
    OUTCOME_OK,
    UsageEvent,
    add_partial,
    record_usage,
)

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

_TOOL_GET_PAGE_WORDS = "get_page_words"
_TOOL_SUBMIT_SCHEMA = "submit_schema"
_TOOL_SUBMIT_VALUES = "submit_values"
_TOOL_APPEND_ENTRIES = "append_entries"
_TOOL_SUBMIT_LOCATIONS = "submit_locations"


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


def load_grounded_config(workspace: Workspace) -> GroundedConfig:
    """Resolve the two grounded models (schema generation → ``expert`` tier,
    value extraction → ``advanced`` tier) and their credentials from the merged
    config's ``[grounded]`` section and ``[models]`` tiers. A model set on the
    section overrides its tier."""
    merged = load_merged_config(workspace)
    schema = resolve_tiered_model(
        merged,
        section_name=ConfigSection.GROUNDED,
        tier=Tier.EXPERT,
        invalid=GroundedConfigInvalid,
        missing=GroundedConfigMissing,
        model_field="schema_model",
        key_field="schema_api_key",
        env_field="schema_api_key_env",
        base_field="schema_api_base",
    )
    values = resolve_tiered_model(
        merged,
        section_name=ConfigSection.GROUNDED,
        tier=Tier.ADVANCED,
        invalid=GroundedConfigInvalid,
        missing=GroundedConfigMissing,
        model_field="values_model",
        key_field="values_api_key",
        env_field="values_api_key_env",
        base_field="values_api_base",
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
    """Return the bytes of the single ``*.pdf`` stored for ``file_id``."""
    keys = workspace.blobs.list_blobs(layout.file_prefix(file_id))
    pdfs = [k for k in keys if k.endswith(".pdf")]
    if not pdfs:
        raise FileNotFound(f"file '{file_id}' has no source PDF")
    return workspace.blobs.get_blob(pdfs[0])


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

    Raises :class:`SchemaGenerationFailed` on any non-config failure
    (no files, missing PDF, malformed LLM response, network error).
    """
    if not file_ids:
        raise SchemaGenerationFailed(
            "schema generation requires at least one example file_id; got an empty list"
        )

    # Read every PDF up front so a permission/missing-file error fails
    # the call before we burn an LLM API request.
    pdf_blocks: list[dict[str, Any]] = []
    for fid in file_ids:
        pdf_bytes = _pdf_bytes(workspace, fid)
        pdf_blocks.append(_pdf_content_block(pdf_bytes))

    api_key = _resolve_api_key(config.schema_api_key, config.schema_api_key_env)
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": _schema_user_prompt(len(file_ids))},
    ]
    user_content.extend(pdf_blocks)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt(PromptKey.SCHEMA_SYSTEM)},
        {"role": "user", "content": user_content},
    ]
    tools = [_submit_schema_tool()]
    # max_tokens=None so the wrapper doesn't add the max_tokens alias alongside
    # max_completion_tokens. reasoning_effort is set unconditionally; the
    # wrapper drops it for Anthropic + forced tool_choice.
    # Single call → records its own usage row (gated on --debug) from the
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

    try:
        result = call_with_tools(
            llm_config,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": _TOOL_SUBMIT_SCHEMA}},
            # Deliberately NOT cached: the cacheable prefix is tools + system,
            # and this system prompt is well short of the provider's minimum
            # cacheable prefix, so a breakpoint would create no entry. Schema
            # generation also runs once per docset, so there is no reuse to
            # capture even if it did.
        )
    except Exception as exc:
        raise SchemaGenerationFailed(
            f"schema generation call failed: {type(exc).__name__}: {exc}"
        ) from exc
    fields = _parse_submit_call(result.response, expected_tool=_TOOL_SUBMIT_SCHEMA, field="fields")

    if not isinstance(fields, list):
        raise SchemaGenerationFailed("LLM returned a non-list 'fields' — expected a field tree")
    try:
        return field_tree_to_rnc(fields, workspace=workspace.organization, docset_name=docset_name)
    except SchemaInvalid as exc:
        raise SchemaGenerationFailed(f"LLM returned an invalid field tree: {exc}") from exc


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
    leaned on.
    """

    values: dict[str, Any]
    tool_calls: int
    xml_key: str
    mode: str


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
    store = DocSetStore(workspace)
    rnc_schema = store.get_schema(docset_id)  # RNC text; raises SchemaNotFound
    vocab = parse_rnc(rnc_schema)
    schema = rnc_to_json_schema(rnc_schema)
    guidance = store.get_guidance(docset_id) if store.has_guidance(docset_id) else None
    pdf_bytes = _pdf_bytes(workspace, file_id)
    api_key = _resolve_api_key(config.values_api_key, config.values_api_key_env)
    api_base = config.values_api_base

    phase1_totals: dict[str, Any] = _empty_totals()
    phase3_totals: dict[str, Any] = _empty_totals()
    tool_calls_total = 0
    outcome = OUTCOME_ERROR
    error_msg: str | None = None
    started = time.monotonic()

    # Per-phase timings are filled as each phase finishes so a failure
    # midway still surfaces what we got done.
    phase1_duration = 0.0
    phase2_duration = 0.0
    phase3_duration = 0.0
    phase3_page_calls = 0
    phase2_matched = 0
    phase3_matched = 0
    unmatched_count = 0
    total_locations = 0
    computed_fields = 0
    dropped_refs = 0
    unnormalized_enums = 0
    derivations_checked = 0
    derivations_mismatched = 0
    invariants_checked = 0
    invariant_violations: list[str] = []
    phase1_layout: dict[str, Any] | None = None
    phase1_tool_schema_mode = "inlined"
    phase1_chunk_calls = 0
    phase1_truncated_retries = 0

    try:
        # --- Phase 1: text + page numbers, no bboxes (LLM) ----------
        phase1_started = time.monotonic()
        phase1_schema = _drop_bboxes_from_schema(schema)
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
        schema_text = _values_phase1_user_prompt(phase1_schema, guidance)

        def _phase1_messages(*, chunked: bool) -> list[dict[str, Any]]:
            # Rebuilt per attempt — _run_extract_loop mutates its message list.
            anthropic = is_anthropic_model(config.values_model)
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
            user_content.append(_pdf_content_block(pdf_bytes))
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

        # Two independent fallbacks compose around phase 1, each a one-way
        # latch enabled by the failure it answers:
        # * provider rejects the inlined tool schema ("too many states")
        #   → permissive object parameter + code-side vocabulary pruning;
        # * output truncated (finish_reason='length')
        #   → chunked submission (see _phase1_tools) with a mandatory directive.
        # Because a latch is only ever flipped from off to on, and each handler
        # re-raises once its own latch is set, the loop runs at most three
        # attempts (initial + one per latch) before terminating.
        chunked = False
        while True:
            tool_schema = (
                _PERMISSIVE_VALUES_PARAM
                if phase1_tool_schema_mode == "permissive"
                else phase1_tool_schema
            )
            try:
                phase1_args, phase1_tool_calls, phase1_chunk_calls = _run_extract_loop(
                    workspace=workspace,
                    file_id=file_id,
                    messages=_phase1_messages(chunked=chunked),
                    tools=_phase1_tools(tool_schema, chunked=chunked),
                    chunked=chunked,
                    model=config.values_model,
                    api_key=api_key,
                    api_base=api_base,
                    max_tool_iters=config.max_tool_iters,
                    totals=phase1_totals,
                )
                break
            except _OutputTruncated as exc:
                if chunked:
                    raise ValuesExtractionFailed(
                        f"{exc} The chunked-submission retry was truncated too; "
                        "consider a model with a higher output ceiling or "
                        "splitting the document."
                    ) from exc
                chunked = True
                phase1_truncated_retries += 1
            except ValuesExtractionFailed as exc:
                if phase1_tool_schema_mode == "permissive" or not _is_tool_schema_too_large(exc):
                    raise
                # Gemini's constrained decoder rejected the inlined schema;
                # the model still sees the full schema in the user prompt.
                phase1_tool_schema_mode = "permissive"
        # Enforce the vocabulary code-side on exactly the paths whose payload
        # never met a provider-side shape check: permissive mode (the values
        # parameter was a bare object) and chunked mode (append_entries
        # batches are validated only as generic objects). The inlined
        # single-shot path already got additionalProperties:false at the API
        # layer, and pruning it too would discard off-schema values *before*
        # phases 2/3 rather than at serialization — a behavior change beyond
        # what these fallbacks need.
        if phase1_tool_schema_mode == "permissive" or chunked:
            phase1_args["values"] = _prune_to_vocabulary(phase1_args["values"], vocab)
        phase1_values = phase1_args["values"]
        phase1_layout = phase1_args.get("layout") or None
        if not isinstance(phase1_layout, dict):
            phase1_layout = None
        tool_calls_total += phase1_tool_calls
        phase1_duration = round(time.monotonic() - phase1_started, 3)
        # The merged extracted_value leaf shape lets a sloppy model blur the
        # grounded/computed boundary; normalize before phases 2/3 (and the
        # serializer) so their invariants hold regardless.
        _normalize_leaf_provenance(phase1_values)
        # Computed (reasoned) leaves carry no locations — phases 2 and 3
        # never see them; counted here so the stats file can attest they
        # were deliberate, not dropped. dropped_refs counts derived_from
        # entries that won't resolve to a dg:href target — incomplete
        # provenance that would otherwise vanish silently at serialization.
        computed_fields = sum(1 for _ in walk_computed_leaves(phase1_values))
        dropped_refs = count_dropped_refs(phase1_values)
        # Enum leaves whose normalized value isn't one of the schema's tokens
        # serialize text-only; count them so misses are visible in stats.
        unnormalized_enums = count_unnormalized_enum_values(phase1_values, vocab)
        # Recompute checkable derivations (report-only): a mismatch means a
        # computed leaf's value agrees with neither the sum, the count, nor
        # any single one of its derived_from inputs.
        derivations_checked, derivations_mismatched = check_derivations(phase1_values)
        # Schema-declared `## Invariant:` relations (count/sum across the
        # submission). Report-only, like the derivation recompute above.
        invariants_checked, invariant_violations = check_invariants(phase1_values, vocab)

        # --- Phase 2: code-side OCR matching ------------------------
        phase2_result = run_phase2_matching(workspace, file_id, phase1_values, layout=phase1_layout)
        phase2_duration = phase2_result.stats.duration_s
        phase2_matched = phase2_result.stats.matched_locations
        total_locations = phase2_result.stats.total_locations

        # --- Phase 3: per-page LLM for remaining unmatched ----------
        phase3_started = time.monotonic()
        final_values = phase2_result.values
        if phase2_result.unmatched:
            final_values, phase3_matched, phase3_page_calls = _run_phase3(
                workspace=workspace,
                file_id=file_id,
                values=final_values,
                unmatched=phase2_result.unmatched,
                model=config.values_model,
                api_key=api_key,
                api_base=api_base,
                max_tool_iters=config.max_tool_iters,
                totals=phase3_totals,
            )
        unmatched_count = phase2_result.stats.unmatched_locations - phase3_matched
        phase3_duration = round(time.monotonic() - phase3_started, 3)

        # Extracted values live as a dg:extraction element in the file's core
        # <stem>.dgml.xml (spec §13): added as a sibling of an existing document
        # tree (full-extraction), or written as a standalone dg:chunk when no
        # tree exists yet (extraction).
        stem = Path(FileStore(workspace).get(file_id).original_filename).stem
        xml_key = layout.dgml_xml_key(docset_id, file_id, stem)
        existing = (
            workspace.blobs.get_blob(xml_key).decode("utf-8")
            if workspace.blobs.blob_exists(xml_key)
            else None
        )
        if existing is not None and has_document_tree(existing):
            # A generated document tree is present — add extraction alongside it.
            mode = "full-extraction"
            doc = embed_extraction_into(existing, final_values, vocab=vocab)
        else:
            # No tree (fresh, or a prior extraction-only file) — (re)write standalone.
            mode = "extraction"
            doc = standalone_extraction_doc(final_values, vocab=vocab)
        workspace.blobs.put_blob(xml_key, doc.encode("utf-8"))
        outcome = OUTCOME_OK
        return ExtractionResult(
            values=final_values, tool_calls=tool_calls_total, xml_key=xml_key, mode=mode
        )
    except ValuesExtractionFailed as exc:
        error_msg = str(exc)
        raise
    except Exception as exc:
        # Non-ValuesExtractionFailed errors (programmer bug) — still
        # record what we can before letting them propagate.
        error_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Usage recording is gated on --debug (like every other LLM path).
        # This aggregates phase 1 + 3 into one row; the internal per-call
        # configs carry no workspace, so they don't each auto-record.
        if debug:
            merged_totals = _merge_totals(phase1_totals, phase3_totals)
            record_usage(
                workspace,
                UsageEvent(
                    at=now_iso(),
                    operation=OPERATION_EXTRACT_VALUES,
                    model=config.values_model,
                    cost_usd=merged_totals["cost_usd"],
                    prompt_tokens=merged_totals["prompt_tokens"],
                    completion_tokens=merged_totals["completion_tokens"],
                    total_tokens=merged_totals["total_tokens"],
                    cache_read_tokens=merged_totals["cache_read_tokens"],
                    cache_creation_tokens=merged_totals["cache_creation_tokens"],
                    duration_s=round(time.monotonic() - started, 3),
                    outcome=outcome,
                    context={
                        "file_id": file_id,
                        "docset_id": docset_id,
                        "tool_calls": tool_calls_total,
                    },
                    error=error_msg,
                ),
            )
        # Even on failure, partial numbers help diagnose where we
        # stalled. Wrapped in try/except so telemetry can never break
        # the caller. Suppressed entirely unless the caller opted in
        # (the CLI does so only under --debug).
        try:
            if write_stats:
                _write_extraction_stats(
                    workspace=workspace,
                    docset_id=docset_id,
                    file_id=file_id,
                    model=config.values_model,
                    outcome=outcome,
                    error_msg=error_msg,
                    phase1_totals=phase1_totals,
                    phase3_totals=phase3_totals,
                    phase1_duration=phase1_duration,
                    phase2_duration=phase2_duration,
                    phase3_duration=phase3_duration,
                    phase3_page_calls=phase3_page_calls,
                    phase2_matched=phase2_matched,
                    phase3_matched=phase3_matched,
                    unmatched=unmatched_count,
                    total_locations=total_locations,
                    computed_fields=computed_fields,
                    dropped_refs=dropped_refs,
                    unnormalized_enums=unnormalized_enums,
                    derivations_checked=derivations_checked,
                    derivations_mismatched=derivations_mismatched,
                    invariants_checked=invariants_checked,
                    invariant_violations=invariant_violations,
                    phase1_layout=phase1_layout,
                    phase1_tool_schema=phase1_tool_schema_mode,
                    phase1_chunk_calls=phase1_chunk_calls,
                    phase1_truncated_retries=phase1_truncated_retries,
                )
        except Exception:
            pass


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
    phase2_matched: int,
    phase3_matched: int,
    unmatched: int,
    total_locations: int,
    computed_fields: int,
    dropped_refs: int,
    unnormalized_enums: int,
    derivations_checked: int,
    derivations_mismatched: int,
    invariants_checked: int,
    invariant_violations: list[str],
    phase1_layout: dict[str, Any] | None,
    phase1_tool_schema: str,
    phase1_chunk_calls: int,
    phase1_truncated_retries: int,
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
                **phase1_totals,
            },
            "phase2": {"duration_s": phase2_duration},
            "phase3": {
                "duration_s": phase3_duration,
                "page_calls": phase3_page_calls,
                **phase3_totals,
            },
        },
        "matching": {
            "total_locations": total_locations,
            "matched_phase2": phase2_matched,
            "matched_phase3": phase3_matched,
            "unmatched": unmatched,
            "computed_fields": computed_fields,
            "dropped_refs": dropped_refs,
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


# ---- Phase 3: per-page LLM for unmatched items ----------------------------


_PHASE3_MAX_PARALLEL = 8


def _run_phase3(
    *,
    workspace: Workspace,
    file_id: str,
    values: dict[str, Any],
    unmatched: list[UnmatchedItem],
    model: str,
    api_key: str | None,
    api_base: str | None,
    max_tool_iters: int,
    totals: dict[str, Any],
) -> tuple[dict[str, Any], int, int]:
    """Resolve ``unmatched`` items via one LLM call per page, run in
    parallel across pages.

    Each page-call sends the page image, OCR words, already-matched
    context for that page (anchors), and the list of ``(id, path, text)``
    items to locate. The model returns ``{id → [bbox, ...]}`` which we
    patch back into ``values`` in code. Phase-3 ids are short and unique
    per page (assigned in :mod:`dgml.matching`), so the model only echoes
    them — never the path or text.

    Page calls are independent — each looks only at its own page's
    OCR, image, and items — so they run concurrently via a
    :class:`ThreadPoolExecutor` (litellm's HTTP call releases the
    GIL). Each thread accumulates into its own ``totals`` dict and we
    merge them after all calls return, keeping the cost telemetry
    accurate without needing a lock on the hot path.

    Returns ``(values, matched_count, page_calls)``.
    """
    from concurrent.futures import ThreadPoolExecutor

    by_page: dict[int, list[UnmatchedItem]] = {}
    for item in unmatched:
        by_page.setdefault(item.page_number, []).append(item)

    def _do_page(
        page: int, items: list[UnmatchedItem]
    ) -> tuple[int, list[UnmatchedItem], dict[str, list[dict[str, Any]]], dict[str, Any]]:
        local_totals = _empty_totals()
        page_results = _phase3_call_for_page(
            workspace=workspace,
            file_id=file_id,
            page_number=page,
            items=items,
            values=values,
            model=model,
            api_key=api_key,
            api_base=api_base,
            max_tool_iters=max_tool_iters,
            totals=local_totals,
        )
        return page, items, page_results, local_totals

    workers = min(_PHASE3_MAX_PARALLEL, max(1, len(by_page)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        per_page = list(ex.map(lambda kv: _do_page(*kv), sorted(by_page.items())))

    matched_count = 0
    page_calls = 0
    for _page, items, page_results, local_totals in per_page:
        page_calls += 1
        add_partial(totals, local_totals)
        for item in items:
            model_locs = page_results.get(item.id)
            if not model_locs:
                continue
            if _patch_value_with_locations(values, item, model_locs):
                matched_count += 1
    return values, matched_count, page_calls


def _phase3_call_for_page(
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
    totals: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """One litellm call: send the page + ids that need locating, return
    ``{id: [{page_number, bounding_box}, ...]}`` parsed from the model's
    ``submit_locations`` tool call.

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

    user_text = _phase3_user_prompt(
        page_number=page_number,
        items=items,
        page_words=page_words,
        page_anchors=_collect_page_anchors(values, page_number),
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt(PromptKey.VALUES_PHASE3_SYSTEM)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                _image_content_block(workspace.blobs.get_blob(image_key)),
            ],
        },
    ]
    tools = [_submit_locations_tool([it.id for it in items])]

    # reasoning_effort is set unconditionally — the wrapper drops it for
    # Anthropic-routed models because tool_choice forces a function call
    # below, and Anthropic rejects extended thinking with forced tools.
    llm_config = LLMConfig(
        model=model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=None,
        max_completion_tokens=_DEFAULT_MAX_COMPLETION_TOKENS,
        temperature=_DEFAULT_VALUES_TEMPERATURE,
        timeout=_DEFAULT_TIMEOUT_SECONDS,
        reasoning_effort=_DEFAULT_REASONING_EFFORT,
    )
    forced_tool_choice = {
        "type": "function",
        "function": {"name": _TOOL_SUBMIT_LOCATIONS},
    }

    for _ in range(max_tool_iters):
        try:
            result = call_with_tools(
                llm_config,
                messages=messages,
                tools=tools,
                tool_choice=forced_tool_choice,
            )
        except Exception as exc:
            raise ValuesExtractionFailed(
                f"phase 3 page {page_number} call failed: {type(exc).__name__}: {exc}"
            ) from exc
        add_partial(totals, result.usage)

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
        return _parse_submit_locations(args, page_number)

    raise ValuesExtractionFailed(
        f"phase 3 page {page_number} exceeded max_tool_iters={max_tool_iters}"
    )


def _parse_submit_locations(
    args: dict[str, Any], page_number: int
) -> dict[str, list[dict[str, Any]]]:
    """Parse a ``submit_locations`` tool-args payload into
    ``{id → locations}``. Malformed entries are dropped silently —
    they'll show up as unresolved in the stats."""
    raw = args.get("locations")
    out: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        item_id = entry.get("id")
        bboxes = entry.get("bounding_boxes")
        if not isinstance(item_id, str) or not isinstance(bboxes, list):
            continue
        locs: list[dict[str, Any]] = []
        for bbox in bboxes:
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if not all(isinstance(c, (int, float)) for c in bbox):
                continue
            # Boxes are integer image pixels [left, top, right, bottom].
            locs.append({"page_number": page_number, "bounding_box": [round(c) for c in bbox]})
        if locs:
            out[item_id] = locs
    return out


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
    return prompt(PromptKey.VALUES_PHASE3_USER).format(
        page_number=page_number,
        ocr_words=encode_phase3_words(page_words.get("words", [])),
        known_locations="\n".join(anchors_lines),
        needs_locating="\n".join(items_lines),
    )


def _submit_locations_tool(ids: list[str]) -> dict[str, Any]:
    """Tool spec for phase 3's submit. ``id`` is constrained to the set
    of ids we're asking about so the model can't invent extras and we
    don't need to filter on the way back."""
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
                                        # [left, top, right, bottom] in image pixels.
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


def _run_extract_loop(
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
    chunked: bool = False,
) -> tuple[dict[str, Any], int, int]:
    """Run a multi-turn extraction loop until the model finishes submitting.

    Returns ``(submit_args, tool_calls_run, chunk_calls)`` — ``submit_args``
    carries the merged ``values`` tree plus optional sibling fields like
    ``layout``; ``chunk_calls`` counts the submission calls (1 for the
    ordinary single ``submit_values``, more when the model used the chunked
    protocol). Mutates ``totals`` by adding cost/token deltas from every
    litellm call so the surrounding ``extract_values`` records a single
    usage row across both phases.

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
    :class:`_OutputTruncated` so the caller can retry with an explicit
    chunking directive.
    """
    # Phase 1 uses tool_choice="auto" (the default in call_with_tools) so
    # the model can call get_page_words between turns. With auto choice
    # the wrapper keeps reasoning_effort for every provider, including
    # Anthropic — only forced tool_choice triggers the Anthropic drop.
    llm_config = LLMConfig(
        model=model,
        api_key=api_key,
        api_base=api_base,
        max_tokens=None,
        max_completion_tokens=_DEFAULT_MAX_COMPLETION_TOKENS,
        temperature=_DEFAULT_VALUES_TEMPERATURE,
        timeout=_DEFAULT_TIMEOUT_SECONDS,
        reasoning_effort=_DEFAULT_REASONING_EFFORT,
    )

    tool_calls_run = 0
    chunk_calls = 0
    acc_args: dict[str, Any] | None = None  # accumulated submit_values args

    def _ack(call_id: str, name: str, payload: dict[str, Any]) -> None:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(payload),
            }
        )

    for _ in range(max_tool_iters):
        try:
            # Always cached: the system prompt and the schema block ahead of the
            # per-file PDF are byte-identical for every file in the docset, so
            # each file after the first reads that prefix instead of re-sending
            # it. ``call_with_tools`` no-ops the marker for non-Anthropic models.
            result = call_with_tools(llm_config, messages=messages, tools=tools, cache=True)
        except Exception as exc:
            raise ValuesExtractionFailed(
                f"extraction call failed: {type(exc).__name__}: {exc}"
            ) from exc

        add_partial(totals, result.usage)

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
            raise ValuesExtractionFailed(
                "model returned no tool call; the run is required to end "
                f"with a {_TOOL_SUBMIT_VALUES!r} call carrying the final values"
            )

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
                        page,
                        start_idx if isinstance(start_idx, int) else None,
                        end_idx if isinstance(end_idx, int) else None,
                    )
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


def _normalize_leaf_provenance(values: Any) -> None:
    """Enforce the grounded-XOR-computed invariant on phase-1 leaves, in place.

    The merged ``extracted_value`` leaf shape lets a sloppy model return both
    ``locations`` and ``computed``/``derived_from`` on one leaf, an empty
    ``locations`` array on a computed value, or an empty-string ``value``.
    Phases 2/3 and the XML serializer rely on the invariants the old split
    shapes guaranteed — a grounded leaf has ``locations`` and no computed
    keys; a computed leaf has ``computed``/``derived_from`` (a list) and no
    ``locations`` key — so this pass restores them:

    * a leaf with at least one usable location is grounded → computed keys
      are dropped;
    * a leaf with computed markers and no usable location is computed →
      ``computed`` is forced true, ``derived_from`` to a list of strings,
      and ``locations`` removed;
    * an unusable ``locations`` array (empty, or no valid page_number) on a
      plain leaf is removed;
    * an empty-string ``value`` is removed.
    """
    if isinstance(values, list):
        for entry in values:
            _normalize_leaf_provenance(entry)
        return
    if not isinstance(values, dict):
        return
    if isinstance(values.get("text"), str):
        # A leaf (only leaves carry a lowercase "text" key — tag names are
        # PascalCase). Normalize in place; never descend further.
        if values.get("value") == "":
            del values["value"]
        locations = values.get("locations")
        grounded = isinstance(locations, list) and any(
            isinstance(loc, dict) and isinstance(loc.get("page_number"), int) for loc in locations
        )
        if grounded:
            values.pop("computed", None)
            values.pop("derived_from", None)
        elif "computed" in values or "derived_from" in values:
            values["computed"] = True
            refs = values.get("derived_from")
            values["derived_from"] = (
                [r for r in refs if isinstance(r, str)] if isinstance(refs, list) else []
            )
            values.pop("locations", None)
        else:
            values.pop("locations", None)
        return
    for child in values.values():
        _normalize_leaf_provenance(child)


def _drop_bboxes_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the schema with ``bounding_box`` stripped from the
    leaf definition's ``locations[]`` (``extracted_value``, plus the legacy
    ``grounded_field`` for older exported schemas).

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
        locs = leaf_def.get("properties", {}).get("locations", {})
        items = locs.get("items", {})
        props = items.get("properties")
        if isinstance(props, dict) and "bounding_box" in props:
            del props["bounding_box"]
        required = items.get("required")
        if isinstance(required, list):
            items["required"] = [r for r in required if r != "bounding_box"]
    return out


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
    "ExtractionResult",
    "GroundedConfig",
    "extract_values",
    "generate_schema",
    "get_page_words",
    "load_grounded_config",
]
