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

"""Orchestration: transcribe each document, label the batch, render.

The block contract (flat, typed, verbatim) carries a document end to end:
transcription emits typed blocks per page window, one batch-wide labeling
call assigns concepts across every document at once, and the tree and final
XML are assembled deterministically in plain code. Coverage is measured on
the rendered XML with the ``dgml_core.generation.coverage`` tools.
"""

from __future__ import annotations

import glob
import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from dgml_core import llm
from dgml_core.conversion import ConverterConfig
from dgml_core.errors import short_error_message
from dgml_core.generation import document
from dgml_core.generation.blocks import Block, Span
from dgml_core.generation.label import (
    _parse_labels_json,
    apply_labels,
    label_documents,
    propagate_list_consistency,
    propagate_table_consistency,
    wrap_detected_values,
)
from dgml_core.generation.render import render_xml
from dgml_core.generation.schema import Schema
from dgml_core.generation.to_semantic import (
    render_dgml,
    render_semantic_xml,
)
from dgml_core.generation.transcribe import transcribe_document
from dgml_core.storage import Workspace
from dgml_core.usage import OPERATION_LABEL, OPERATION_TRANSCRIBE


def load_labeled_docs_from_cache(cache_dir: Path | str, stems: list[str]) -> dict[str, list[Block]]:
    """Rebuild fully-labeled blocks for already-generated docs from cache."""
    cache = Path(cache_dir)
    docs: dict[str, list[Block]] = {}
    for stem in stems:
        blocks_file = cache / f"{stem}_blocks.json"
        if not blocks_file.exists():
            continue
        raw_blocks = json.loads(blocks_file.read_text(encoding="utf-8"))
        blocks = [
            Block(**{**b, "entities": [Span(**sp) for sp in b.get("entities", [])]})
            for b in raw_blocks
        ]
        for label_file in sorted(cache.glob(f"label_{glob.escape(stem)}_c*_raw.json")):
            payload = _parse_labels_json(label_file.read_text(encoding="utf-8"))
            apply_labels(blocks, payload.get("labels", {}) or {}, doc_name=stem)
        propagate_table_consistency(blocks)
        propagate_list_consistency(blocks)
        wrap_detected_values(blocks)
        docs[stem] = blocks
    return docs


@dataclass
class ConvertOptions:
    """Knobs for the pipeline — deliberately few."""

    # Transcription model — REQUIRED, no default. Which model runs is a
    # user-visible choice; the CLI sources it from the workspace's
    # `generation.model` (config.json), never a silent code
    # default. See dgml_core.generation.config.load_generation_config.
    model: str
    # The labeling pass benefits most from a strong model; it is a handful of
    # small-output calls per batch, so escalating it is cheap. None = reuse
    # `model` (no second, separately-billed model introduced implicitly).
    label_model: str | None = None
    # API credentials for the transcription/labeling calls. None lets litellm
    # fall back to its per-provider env-var conventions (ANTHROPIC_API_KEY, …).
    api_key: str | None = None
    api_base: str | None = None
    window_size: int = 10
    temperature: float = 0.0
    max_tokens: int = 32000
    cache_dir: Path | str | None = None
    # document name → its page_text/ dir (per-page word JSONs written before
    # generation). When a document has one, each transcription window is
    # completeness-checked against its pages' words and retried once if the
    # model stopped early (see transcribe._GATE_RECALL). None disables the gate.
    page_text_dirs: Mapping[str, Path] | None = None
    # When True, also write the debug-only cache artifacts (raw LLM dumps,
    # intermediate .concept.xml/.semantic.xml renders, prompt listings). The
    # functional cache files the next run reloads (_blocks.json,
    # label_*_cNN_raw.json, concept_roster.json) are always written when
    # cache_dir is set, regardless of this flag.
    debug: bool = False
    # When set, convert_batch returns the FINAL dgml (dg:chunk, concept tags +
    # dg:chunk scaffolding, value typing) instead of the windowed-shape
    # intermediate. This is the standard dg:chunk opening tag from
    # semantic_transform.build_header. Empty → return the intermediate
    # (library/test shape).
    dgml_header: str = ""
    # Documents transcribed concurrently. Windows WITHIN a document stay
    # serial (window N+1 receives window N's tail for the `continues`
    # contract); across documents there is no dependency. Set 1 to serialize
    # on 429s.
    max_parallel_docs: int = 4
    # Per-format-family converters (docx/xlsx → PDF), from the workspace
    # `conversion` config. Passed to load_document_as_pdf so non-PDF inputs
    # convert; None/empty means PDF-only (every input must already be a PDF).
    converters: dict[str, ConverterConfig] | None = None
    # Optional full-fidelity schema seed (from --schema-path or the docset's
    # own schema.json on an incremental run). Seeds the roster with role
    # descriptions, curated examples, kind, and hierarchy; Pass B.1 planning
    # is skipped. Takes precedence over roster_seed.
    schema_seed: Schema | None = None
    # Legacy flat {concept: description} roster seed (cache/concept_roster.json
    # fallback). When set (and schema_seed is not), it seeds the roster and
    # Pass B.1 planning is skipped.
    roster_seed: dict[str, str] | None = None
    # Optional leaf-concept → container-concept map (from a seed schema's
    # parent/children). Drives the entity-container grouping in render_dgml
    # (e.g. BuyerAddress/BuyerPhone → BuyerInformation). None = no grouping.
    parent_map: dict[str, str] | None = None
    progress: Callable[[str], None] | None = field(default=None)
    # Workspace to record LLM usage into. When set (and ``debug`` is True), the
    # transcription/labeling calls append rows to ``usage.jsonl``; None disables
    # recording (library/test callers that don't want telemetry).
    workspace: Workspace | None = None


def _config(
    opts: ConvertOptions, model: str | None = None, *, operation: str | None = None
) -> llm.LLMConfig:
    return llm.LLMConfig(
        model=model or opts.model,
        api_key=opts.api_key,
        api_base=opts.api_base,
        temperature=opts.temperature,
        max_tokens=opts.max_tokens,
        workspace=opts.workspace,
        debug=opts.debug,
        operation=operation,
    )


def convert_batch(
    inputs: list[Path | str],
    *,
    options: ConvertOptions,
    on_output: Callable[[str, str], None] | None = None,
    on_error: Callable[[str, str], None] | None = None,
    prior_docs: Mapping[str, list[Block]] | None = None,
    prior_outputs: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """PDFs → labeled semantic XML via typed blocks.

    Returns ``{name: xml}`` by default. Pass *on_output* — called ``(name, xml)``
    as each document is rendered — to consume each result and have it freed
    immediately (write to disk, score, …) instead of accumulating every
    rendered DGML string in memory; in that case the returned dict is empty.
    (The whole batch's parsed blocks still live in memory for the shared
    labeling pass — that is a separate, inherent floor.)

    Pass *on_error* — called ``(name, message)`` with a short one-line reason —
    to learn *why* a document was dropped during transcription. The full error
    still goes to *options.progress* (the verbose log); this hands the caller a
    compact cause it can put in a machine-readable payload. Called once per
    failed document, serially, after the (possibly concurrent) transcription
    pass — so the callback need not be thread-safe.

    *prior_docs* (already-generated docs from cache) are re-rendered so the
    whole docset stays consistent as its schema/roster grows; any whose render
    changes is re-emitted via *on_output* (skipped if unchanged per
    *prior_outputs*).
    """
    opts = options
    log = opts.progress or (lambda _m: None)

    paths = [Path(p) for p in inputs]
    # name → short failure reason, populated in-thread (unique keys per doc),
    # surfaced to *on_error* serially below.
    transcribe_errors: dict[str, str] = {}

    def _transcribe(path: Path) -> list[Block] | None:
        # One bad document (unreadable/unconvertible PDF, LLM/network error)
        # must not sink the whole batch — log it and skip, so the other
        # documents' transcription and the shared labeling pass still run.
        try:
            pdf_bytes = document.load_document_as_pdf(path, converters=opts.converters or {})
            return transcribe_document(
                pdf_bytes,
                doc_name=path.name,
                config=_config(opts, operation=OPERATION_TRANSCRIBE),
                window_size=opts.window_size,
                cache_dir=opts.cache_dir,
                debug=opts.debug,
                log=log,
                page_text_dir=(opts.page_text_dirs or {}).get(path.name),
            )
        except Exception as exc:
            log(f"[transcribe] {path.name} FAILED: {exc}; skipping")
            transcribe_errors[path.name] = short_error_message(exc)
            return None

    # Windows within a document are serial (the `continues` contract);
    # documents are independent, so transcribe them concurrently.
    workers = max(1, min(opts.max_parallel_docs, len(paths)))
    if workers == 1 or len(paths) == 1:
        block_lists = [_transcribe(p) for p in paths]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            block_lists = list(pool.map(_transcribe, paths))
    # Hand each transcription failure's short reason to the caller (serially,
    # in input order) so it can name the cause in machine-readable output.
    if on_error is not None:
        for path in paths:
            reason = transcribe_errors.get(path.name)
            if reason is not None:
                on_error(path.name, reason)
    # Drop documents that failed transcription (block_lists entry is None).
    docs: dict[str, list[Block]] = {
        path.name: blocks
        for path, blocks in zip(paths, block_lists, strict=True)
        if blocks is not None
    }

    # One aggregated usage row for the whole labeling pass: label_documents
    # threads this single config through every call, so a scope on it folds
    # them into one row (gated on --debug via the config).
    label_config = _config(opts, opts.label_model, operation=OPERATION_LABEL)
    label_config.context = {"doc_count": len(docs)}
    with llm.record_usage_for(label_config):
        label_documents(
            docs,
            config=label_config,
            cache_dir=opts.cache_dir,
            debug=opts.debug,
            log=log,
            roster_seed=opts.roster_seed,
            schema_seed=opts.schema_seed,
        )

    outputs: dict[str, str] = {}
    for name, blocks in docs.items():
        # With dgml_header set, the product output is the final dg:chunk dgml
        # (concept tags where labeled, dg:chunk scaffolding otherwise, value
        # typing). Without it, the plain structure-attribute form is returned
        # (library/test shape). The compact concept render and the
        # structure-attribute XML are kept as debug artifacts in the cache.
        if opts.dgml_header:
            xml = render_dgml(blocks, header=opts.dgml_header, parent_map=opts.parent_map)
        else:
            xml = render_semantic_xml(blocks)
        # Stream to the sink (freed immediately) or accumulate for the return.
        if on_output is not None:
            on_output(name, xml)
        else:
            outputs[name] = xml
        # Debug-only intermediate renders — never read back, so gated on
        # --debug (the functional blocks/roster caches are written elsewhere).
        if opts.debug and opts.cache_dir is not None:
            cache = Path(opts.cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            (cache / f"{Path(name).stem}.concept.xml").write_text(
                render_xml(blocks, doc_name=name), encoding="utf-8"
            )
            (cache / f"{Path(name).stem}.semantic.xml").write_text(
                render_semantic_xml(blocks), encoding="utf-8"
            )

    # Re-render prior docs whose rendered XML changed and emit only those. All
    # concepts are docset:-namespaced, so sharing no longer shifts prefixes; a
    # prior render still changes when entity-container grouping moves as the
    # docset's schema/roster grows (or, once, migrating legacy dg:-namespaced
    # concepts to docset:).
    if prior_docs and opts.dgml_header and on_output is not None:
        for name, blocks in prior_docs.items():
            xml = render_dgml(blocks, header=opts.dgml_header, parent_map=opts.parent_map)
            if prior_outputs is not None and prior_outputs.get(name) == xml:
                continue
            log(f"re-rendering {name} (docset render changed)")
            on_output(name, xml)
    return outputs
