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

- ``dgml extraction generate-schema`` — Claude is given a PDF from the docset
  and asked to propose a *typed field tree* describing the structured
  information to extract, choosing an ``xsd`` datatype for each leaf. That tree
  is rendered straight to the at-rest RELAX NG Compact schema
  (:func:`dgml_core.extraction_schema.field_tree_to_rnc`) — no grounded_field
  JSON Schema intermediate. Downstream extraction attributes every value back
  to a region of the source via the ``grounded_field`` form
  (``{text, locations: [{page_number, bounding_box}]}``).

- ``dgml file extract`` (and the auto-extract hook on
  ``dgml docset add-file``) — Gemini is given a PDF plus the docset's
  schema, and asked to produce values matching that schema. To keep
  output attributable to the source, the model is granted a ``get_page_words``
  tool that returns OCR-extracted words and their bounding boxes from
  the workspace. The model produces final results via a ``submit_values``
  tool call.

Coordinate space contract:
- Bounding boxes are integer image pixels ``[left, top, right, bottom]``
  (top-left origin) at 300 dpi, relative to ``page_images/page_N.png`` —
  one convention end-to-end. This is what ``page_text/page_N.json``
  stores, what the ``grounded_field`` form uses in schemas and values,
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

from .docsets import DocSetStore
from .errors import (
    AuthError,
    CorruptMetadata,
    FileNotFound,
    GroundedConfigInvalid,
    GroundedConfigMissing,
    SchemaGenerationFailed,
    SchemaInvalid,
    ValuesExtractionFailed,
    now_iso,
)
from .extraction_schema import field_tree_to_rnc, parse_rnc, rnc_to_json_schema
from .extraction_xml import (
    count_dropped_refs,
    embed_extraction_into,
    has_document_tree,
    standalone_extraction_doc,
)
from .files import FileStore
from .llm import LLMConfig, call_with_tools
from .matching import (
    UnmatchedItem,
    path_to_str,
    run_phase2_matching,
    walk_computed_leaves,
)
from .prompts import get as prompt
from .storage import Workspace, read_config, read_json, write_json_atomic, write_text_atomic
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

# Per-call output token cap. Set high to avoid silent clipping that
# would surface as malformed-JSON tool-call arguments. Frontier
# Gemini/Claude models cap at 64K-65K output tokens; 65000 hits that
# ceiling while leaving the provider's bookkeeping headroom. Tuned
# for value extraction which can be large (fully-grounded entries
# carry text plus several bbox locations apiece).
_DEFAULT_MAX_COMPLETION_TOKENS = 65000

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
    max_tool_iters: int = DEFAULT_MAX_TOOL_ITERS


def load_grounded_config(workspace: Workspace) -> GroundedConfig:
    """Read and validate the ``grounded`` section of ``<workspace>/config.json``."""
    if not workspace.config_path.exists():
        raise GroundedConfigMissing(
            f"no config.json at {workspace.config_path}; "
            "schema-gen and value extraction require a workspace config with a 'grounded' section"
        )
    try:
        data = read_config(workspace.config_path)
    except CorruptMetadata as exc:
        raise GroundedConfigInvalid(f"{workspace.config_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise GroundedConfigInvalid(f"{workspace.config_path} must contain a JSON object")
    section = data.get("grounded")
    if section is None:
        raise GroundedConfigMissing(f"{workspace.config_path} has no 'grounded' section")
    if not isinstance(section, dict):
        raise GroundedConfigInvalid("'grounded' must be a JSON object")

    schema_model = section.get("schema_model")
    if not isinstance(schema_model, str) or not schema_model.strip():
        raise GroundedConfigInvalid(
            "'grounded.schema_model' must be a non-empty string (e.g. 'anthropic/claude-opus-4-7')"
        )
    values_model = section.get("values_model")
    if not isinstance(values_model, str) or not values_model.strip():
        raise GroundedConfigInvalid(
            "'grounded.values_model' must be a non-empty string (e.g. 'gemini/gemini-2.5-pro')"
        )

    schema_api_key = _validate_optional_str(
        section.get("schema_api_key"), "grounded.schema_api_key"
    )
    values_api_key = _validate_optional_str(
        section.get("values_api_key"), "grounded.values_api_key"
    )
    schema_api_key_env = _validate_optional_str(
        section.get("schema_api_key_env"), "grounded.schema_api_key_env"
    )
    values_api_key_env = _validate_optional_str(
        section.get("values_api_key_env"), "grounded.values_api_key_env"
    )
    if schema_api_key is not None and schema_api_key_env is not None:
        raise GroundedConfigInvalid(
            "set at most one of 'grounded.schema_api_key' / 'grounded.schema_api_key_env', not both"
        )
    if values_api_key is not None and values_api_key_env is not None:
        raise GroundedConfigInvalid(
            "set at most one of 'grounded.values_api_key' / 'grounded.values_api_key_env', not both"
        )

    max_tool_iters_raw = section.get("max_tool_iters", DEFAULT_MAX_TOOL_ITERS)
    if (
        not isinstance(max_tool_iters_raw, int)
        or isinstance(max_tool_iters_raw, bool)
        or max_tool_iters_raw < 1
    ):
        raise GroundedConfigInvalid("'grounded.max_tool_iters' must be a positive integer if set")

    return GroundedConfig(
        schema_model=schema_model,
        values_model=values_model,
        schema_api_key=schema_api_key,
        values_api_key=values_api_key,
        schema_api_key_env=schema_api_key_env,
        values_api_key_env=values_api_key_env,
        max_tool_iters=max_tool_iters_raw,
    )


def _validate_optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise GroundedConfigInvalid(f"'{field_name}' must be a non-empty string if set")
    return value


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
    text_path = workspace.file_text_dir(file_id) / f"page_{page}.json"
    if not text_path.exists():
        raise FileNotFound(
            f"no page_text for file '{file_id}' page {page} "
            f"(expected at {text_path}); was the file added with --text-mode digital or ocr?"
        )
    payload = read_json(text_path)
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


def _pdf_path(workspace: Workspace, file_id: str) -> Path:
    """Find the single ``*.pdf`` under ``files/<file_id>/``."""
    file_dir = workspace.file_dir(file_id)
    if not file_dir.exists():
        raise FileNotFound(f"file '{file_id}' not found at {file_dir}")
    pdfs = list(file_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFound(f"file '{file_id}' has no source PDF in {file_dir}")
    return pdfs[0]


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
        pdf_bytes = _pdf_path(workspace, fid).read_bytes()
        pdf_blocks.append(_pdf_content_block(pdf_bytes))

    api_key = _resolve_api_key(config.schema_api_key, config.schema_api_key_env)
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": _schema_user_prompt(len(file_ids))},
    ]
    user_content.extend(pdf_blocks)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt("extraction_schema_system")},
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
        intro = prompt("extraction_schema_user_intro_single")
    else:
        intro = prompt("extraction_schema_user_intro_multi").format(n_files=n_files)
    return intro + "\n\n" + prompt("extraction_schema_user_body")


def _submit_schema_tool() -> dict[str, Any]:
    # The tree structure is documented in the prompt and enforced by the
    # deterministic Python parser (``field_tree_to_vocabulary``); the tool
    # schema keeps ``fields`` a free-form array of objects so providers don't
    # have to support a recursive JSON Schema.
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
                        "items": {"type": "object"},
                    }
                },
                "required": ["fields"],
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
    xml_path: Path
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
    pdf_bytes = _pdf_path(workspace, file_id).read_bytes()
    api_key = _resolve_api_key(config.values_api_key, config.values_api_key_env)

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
    phase1_layout: dict[str, Any] | None = None

    try:
        # --- Phase 1: text + page numbers, no bboxes (LLM) ----------
        phase1_started = time.monotonic()
        phase1_schema = _drop_bboxes_from_schema(schema)
        phase1_values_schema = _expand_refs(phase1_schema)
        phase1_messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt("extraction_values_phase1_system")},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _values_phase1_user_prompt(phase1_schema)},
                    _pdf_content_block(pdf_bytes),
                ],
            },
        ]
        phase1_args, phase1_tool_calls = _run_extract_loop(
            workspace=workspace,
            file_id=file_id,
            messages=phase1_messages,
            tools=[_submit_values_tool(phase1_values_schema, with_layout=True)],
            model=config.values_model,
            api_key=api_key,
            max_tool_iters=config.max_tool_iters,
            totals=phase1_totals,
        )
        phase1_values = phase1_args["values"]
        phase1_layout = phase1_args.get("layout") or None
        if not isinstance(phase1_layout, dict):
            phase1_layout = None
        tool_calls_total += phase1_tool_calls
        phase1_duration = round(time.monotonic() - phase1_started, 3)
        # Computed (reasoned) leaves carry no locations — phases 2 and 3
        # never see them; counted here so the stats file can attest they
        # were deliberate, not dropped. dropped_refs counts derived_from
        # entries that won't resolve to a dg:href target — incomplete
        # provenance that would otherwise vanish silently at serialization.
        computed_fields = sum(1 for _ in walk_computed_leaves(phase1_values))
        dropped_refs = count_dropped_refs(phase1_values)

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
        xml_path = workspace.file_dgml_xml_path(docset_id, file_id, stem)
        existing = xml_path.read_text(encoding="utf-8") if xml_path.exists() else None
        if existing is not None and has_document_tree(existing):
            # A generated document tree is present — add extraction alongside it.
            mode = "full-extraction"
            doc = embed_extraction_into(existing, final_values, vocab=vocab)
        else:
            # No tree (fresh, or a prior extraction-only file) — (re)write standalone.
            mode = "extraction"
            doc = standalone_extraction_doc(final_values, vocab=vocab)
        write_text_atomic(xml_path, doc)
        outcome = OUTCOME_OK
        return ExtractionResult(
            values=final_values, tool_calls=tool_calls_total, xml_path=xml_path, mode=mode
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
                    phase1_layout=phase1_layout,
                )
        except Exception:
            pass


def _empty_totals() -> dict[str, Any]:
    return {
        "cost_usd": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
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
    phase1_layout: dict[str, Any] | None,
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
            "phase1": {"duration_s": phase1_duration, **phase1_totals},
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
        },
        # Phase-1's emitted layout hint, persisted so we can audit
        # whether the model produced a useful descriptor for each
        # array (some models drop optional tool-call parameters).
        "phase1_layout": phase1_layout,
    }
    write_json_atomic(workspace.docset_file_extraction_stats_path(docset_id, file_id), stats)


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
    max_tool_iters: int,
    totals: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """One litellm call: send the page + ids that need locating, return
    ``{id: [{page_number, bounding_box}, ...]}`` parsed from the model's
    ``submit_locations`` tool call."""
    image_path = workspace.file_dir(file_id) / "page_images" / f"page_{page_number}.png"
    if not image_path.exists():
        raise ValuesExtractionFailed(
            f"phase 3: no page image at {image_path} for page {page_number}"
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
        {"role": "system", "content": prompt("extraction_values_phase3_system")},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                _image_content_block(image_path.read_bytes()),
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
    return prompt("extraction_values_phase3_user").format(
        page_number=page_number,
        ocr_words=json.dumps(page_words.get("words", []), indent=2),
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


def _run_extract_loop(
    *,
    workspace: Workspace,
    file_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    api_key: str | None,
    max_tool_iters: int,
    totals: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Run a multi-turn extraction loop until the model calls ``submit_values``.

    Returns ``(submit_args, tool_calls_run)`` — ``submit_args`` is the
    full args dict the model passed to ``submit_values``, so callers can
    pluck out optional sibling fields like ``layout`` in addition to
    ``values``. Mutates ``totals`` by adding cost/token deltas from
    every litellm call so the surrounding ``extract_values`` records a
    single usage row across both phases.
    """
    # Phase 1 uses tool_choice="auto" (the default in call_with_tools) so
    # the model can call get_page_words between turns. With auto choice
    # the wrapper keeps reasoning_effort for every provider, including
    # Anthropic — only forced tool_choice triggers the Anthropic drop.
    llm_config = LLMConfig(
        model=model,
        api_key=api_key,
        max_tokens=None,
        max_completion_tokens=_DEFAULT_MAX_COMPLETION_TOKENS,
        temperature=_DEFAULT_VALUES_TEMPERATURE,
        timeout=_DEFAULT_TIMEOUT_SECONDS,
        reasoning_effort=_DEFAULT_REASONING_EFFORT,
    )

    tool_calls_run = 0
    for _ in range(max_tool_iters):
        try:
            result = call_with_tools(llm_config, messages=messages, tools=tools)
        except Exception as exc:
            raise ValuesExtractionFailed(
                f"extraction call failed: {type(exc).__name__}: {exc}"
            ) from exc

        add_partial(totals, result.usage)

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
                return args, tool_calls_run

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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": json.dumps(tool_result),
                    }
                )
                continue

            raise ValuesExtractionFailed(f"model called unknown tool: {name!r}")

    raise ValuesExtractionFailed(
        f"extraction exceeded max_tool_iters={max_tool_iters} "
        f"without producing a {_TOOL_SUBMIT_VALUES!r} call"
    )


def _drop_bboxes_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the schema with ``bounding_box`` stripped from
    ``definitions.grounded_field.locations[]``.

    Phase-1 extraction shows the model this slimmer shape so it focuses
    on getting the text and page numbers right without the bbox burden.
    Schemas without the conventional ``$ref: '#/definitions/grounded_field'``
    pattern are returned unchanged — the convention is what the
    schema-gen step always produces.
    """
    out = copy.deepcopy(schema)
    defs = out.get("definitions")
    if not isinstance(defs, dict):
        return out
    gf = defs.get("grounded_field")
    if not isinstance(gf, dict):
        return out
    locs = gf.get("properties", {}).get("locations", {})
    items = locs.get("items", {})
    props = items.get("properties")
    if isinstance(props, dict) and "bounding_box" in props:
        del props["bounding_box"]
    required = items.get("required")
    if isinstance(required, list):
        items["required"] = [r for r in required if r != "bounding_box"]
    return out


def _values_phase1_user_prompt(schema: dict[str, Any]) -> str:
    return prompt("extraction_values_phase1_user").format(schema=json.dumps(schema, indent=2))


def _submit_values_tool(
    values_schema: dict[str, Any], *, with_layout: bool = False
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
    return {
        "type": "function",
        "function": {
            "name": _TOOL_SUBMIT_VALUES,
            "description": (
                "Submit the final extracted values. Call exactly once when "
                "extraction is complete; this ends the run."
            ),
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
                    return walk(target)
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
            "(referenced by a *_api_key_env field in config.json[grounded])"
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
