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
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from dgml_core import llm
from dgml_core.concurrency import map_concurrent
from dgml_core.conversion import ConverterConfig
from dgml_core.errors import GenerationConfigInvalid, short_error_message
from dgml_core.generation import document
from dgml_core.generation.blocks import Block, Span
from dgml_core.generation.config import (
    LABEL_MODE_ALL_AT_ONCE,
    LABEL_MODE_PER_DOCUMENT,
    LABEL_MODE_SYNC,
)
from dgml_core.generation.label import (
    RosterEntry,
    _parse_labels_json,
    apply_labels,
    label_documents,
    plan_concept_roster,
    propagate_list_consistency,
    propagate_table_consistency,
    wrap_detected_values,
)
from dgml_core.generation.render import render_xml
from dgml_core.generation.schema import Schema
from dgml_core.generation.single_calls import SingleCallRunner, single_calls_through
from dgml_core.generation.to_semantic import (
    render_dgml,
    render_semantic_xml,
)
from dgml_core.generation.transcribe import transcribe_document, transcribe_steps
from dgml_core.generation.vocab import OPEN_VOCAB, TagVocab
from dgml_core.pages import PdfConfig
from dgml_core.storage import Workspace
from dgml_core.usage import OPERATION_LABEL, OPERATION_TRANSCRIBE

if TYPE_CHECKING:
    # Type-only: ``dgml_core.batch`` loads every provider backend at import, so
    # it is imported inside the batch code path, never at module import (the
    # non-batch path stays as light as it was; see upstream #160).
    from dgml_core.batch import BatchExecutor


def load_labeled_docs_from_cache(
    cache_dir: Path | str, stems: list[str], vocab: TagVocab | None = None
) -> dict[str, list[Block]]:
    """Rebuild fully-labeled blocks for already-generated docs from cache.

    *vocab* MUST be the same one a fresh run would use. This path replays the
    cached label JSON through ``apply_labels``, so a different vocabulary here
    would resolve the same model output differently and a replayed document
    would diverge from a freshly-labeled one — silently breaking the
    reproducibility a pinned schema exists to provide.
    """
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
        for label_file in sorted(cache.glob(f"label_{glob.escape(stem)}_*_raw.json")):
            try:
                payload = _parse_labels_json(label_file.read_text(encoding="utf-8"))
            except ValueError:
                # A bisected chunk's unparseable reply, which a fresh run applied
                # nothing from. Caches written by older versions can hold one.
                continue
            apply_labels(blocks, payload.get("labels", {}) or {}, doc_name=stem, vocab=vocab)
        propagate_table_consistency(blocks)
        propagate_list_consistency(blocks)
        wrap_detected_values(blocks)
        docs[stem] = blocks
    return docs


@dataclass
class BatchOptions:
    """Run the batchable stages through the provider's batch endpoint.

    Set :attr:`ConvertOptions.batch` to one of these to enable batch mode;
    ``None`` (the default) is the synchronous pipeline, untouched. Pass A
    always batches. Pass B labeling batches too unless ``label`` is ``False``
    (then it runs with ordinary synchronous calls, under every vocabulary),
    and the vocabulary picks how: every document at once under a vocabulary
    the run cannot change (see
    :func:`dgml_core.generation.label.is_batchable_vocab`), otherwise one
    document at a time, in the serial order, with output byte-identical to
    the synchronous run (see :func:`_per_document_labeler`). The ``label``
    stage's stats carry ``mode``: ``"all-at-once"``, ``"per-document"`` or
    ``"sync"`` (``LABEL_MODE_*`` in :mod:`dgml_core.generation.config`).
    ``poll_interval_s``,
    ``max_poll_s`` and ``min_wave_size`` configure the
    :class:`dgml_core.batch.BatchExecutor`; ``log`` receives its progress
    lines (defaults to the run's ``progress`` log).

    ``stats`` is an OUTPUT: :func:`convert_batch` fills it with one entry per
    stage it ran — ``WaveStats.to_json()`` for a batched stage, or
    ``{"skipped": "<reason>"}`` for one that stayed synchronous — so the
    caller can report what the batch path did without a signature change.
    """

    poll_interval_s: float = 30.0
    #: ``None`` = the backend's own expiry window plus a margin (see
    #: :func:`dgml_core.batch.executor.default_max_poll_s`).
    max_poll_s: float | None = None
    min_wave_size: int = 1
    log: Callable[[str], None] | None = None
    stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: role ("transcribe" / "label") → the non-secret credential reference a
    #: batch job records with each provider batch (see
    #: :func:`dgml_core.batch.jobs.credential_ref`). Empty = none recorded.
    credentials: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Batch Pass B labeling (``[generation] batch_label``; the CLI's
    #: ``--no-batch-label`` turns it off). ``False`` labels synchronously.
    label: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.label, bool):
            raise GenerationConfigInvalid(
                f"BatchOptions.label must be true or false, got {self.label!r} "
                f"({type(self.label).__name__})"
            )


def make_batch_executor(
    batch: BatchOptions,
    *,
    model: str,
    api_key: str | None,
    api_base: str | None,
    log: Callable[[str], None] = lambda _m: None,
    role: str = "transcribe",
) -> BatchExecutor:
    """A :class:`~dgml_core.batch.BatchExecutor` over *model*'s batch backend.

    Imports :mod:`dgml_core.batch` here, only when batch mode actually runs.
    Raises :class:`dgml_core.errors.BatchUnavailable` when the model's
    provider has no batch backend — the rejection policy: never a silent
    fallback to full-price synchronous calls.
    """
    from dgml_core.batch import make_executor

    return make_executor(
        model,
        api_key=api_key,
        api_base=api_base,
        poll_interval_s=batch.poll_interval_s,
        max_poll_s=batch.max_poll_s,
        min_wave_size=batch.min_wave_size,
        log=batch.log or log,
        credential=batch.credentials.get(role),
    )


def batch_single_call_runner(
    batch: BatchOptions,
    *,
    api_key: str | None,
    api_base: str | None,
    log: Callable[[str], None] = lambda _m: None,
) -> SingleCallRunner:
    """The batch driver for Pass B's docset-wide calls (roster planning, gap
    planning, concept descriptions; see :mod:`dgml_core.generation.single_calls`).

    Each call runs as a one-unit batch stage over the labeling model — one wave
    per request in its chain (planning: the draft, then the refine turn) — with
    the stage's stats in ``batch.stats[<stage>]``. The unit's usage folds into
    the labeling pass's aggregated row exactly as the synchronous call's does,
    tagged with the tier that served it. A request the stage could not serve
    raises what the synchronous call would have (the call sites are
    best-effort and catch it); a stage-wide failure raises its error the same
    way; a paused job propagates.

    *api_key* / *api_base* are the labeling model's credentials (the config the
    call carries names the model)."""

    def run(stage: str, config: llm.LLMConfig, steps: llm.LLMSteps[Any]) -> Any:
        from dgml_core.batch import Unit, run_stage

        try:
            executor = make_batch_executor(
                batch, model=config.model, api_key=api_key, api_base=api_base, log=log, role="label"
            )
        except BaseException:
            steps.close()
            raise
        try:
            outcome = run_stage(
                [Unit(name=stage, config=config, steps=steps)], executor, log=log, stage=stage
            )[stage]
        finally:
            batch.stats[stage] = executor.stats.to_json()
        if outcome.error is not None:
            raise outcome.error
        return outcome.result

    return run


#: Reported when the vocabulary would batch but the roster built from it can
#: still change between documents (an entry not frozen, or a legal tag with
#: no entry yet).
ROSTER_NOT_FIXED = "the roster can still change between documents"

#: The ``label`` stage's ``skipped`` reason when batch labeling is off.
BATCH_LABEL_OFF = "batch labeling is off (batch_label = false)"


def unbatchable_label_reason(
    vocab: TagVocab, roster: Mapping[str, RosterEntry] | None = None
) -> str | None:
    """Why Pass B stays serial under batch mode, or ``None`` when it batches.

    With no *roster* (the CLI's pre-flight, before any roster exists) only
    the vocabulary is judged; with one, this is exactly
    :func:`~dgml_core.generation.label.is_batchable_vocab`, phrased as a
    reason. Both halves come from
    :func:`~dgml_core.generation.label.vocab_batch_blocker`, so the
    pre-flight and the decision cannot diverge.
    """
    from dgml_core.generation.label import is_batchable_vocab, vocab_batch_blocker

    blocker = vocab_batch_blocker(vocab)
    if blocker is not None:
        return blocker
    if roster is not None and not is_batchable_vocab(vocab, roster):
        return ROSTER_NOT_FIXED
    return None


@dataclass
class ConvertOptions:
    """Knobs for the pipeline — deliberately few."""

    # Transcription model — REQUIRED, no default. Which model runs is a
    # user-visible choice; the CLI sources it from the workspace's
    # `generation.model` (config.toml), never a silent code
    # default. See dgml_core.generation.config.load_generation_config.
    model: str
    # Labeling model for the single batch-wide semantic-labeling call —
    # REQUIRED, no default. Named explicitly and never implicitly reused from
    # `model`, so a second (separately-billed) model is always a deliberate
    # choice; pass the same string as `model` if you want one model for both.
    label_model: str
    # Transcription-model credentials. None lets litellm fall back to its
    # per-provider env-var conventions (ANTHROPIC_API_KEY, …).
    api_key: str | None = None
    api_base: str | None = None
    # Labeling-model credentials, resolved independently of the transcription
    # ones (the labeling model may name a different provider). They are NOT
    # inherited from `api_key`/`api_base`; None lets litellm fall back to the
    # provider's conventional env var.
    label_api_key: str | None = None
    label_api_base: str | None = None
    window_size: int = 10
    temperature: float = 0.0
    # Output ceiling per call, clamped per model downstream. Headroom, not a
    # target: a long document produces MORE calls, not bigger ones, because
    # transcription is windowed (`window_size`) and labeling is chunked
    # (label._MAX_CHUNK_CHARS). Measured over 1,884 cached replies the largest
    # window is ~16.7K tokens and the largest label chunk ~19.4K. The one call
    # that scales with the CORPUS rather than a chunk is describe_concepts,
    # whose output grows with the concept roster and has been observed at
    # ~29.9K — 93% of the previous 32000 default. 64000 is ~2x the largest
    # reply seen, matches claude-haiku-4-5's own ceiling (so transcription is
    # unaffected either way), and still bounds a runaway reply.
    max_tokens: int = 64000
    # Anthropic extended thinking for both passes, one of
    # :data:`~dgml_core.llm.ANTHROPIC_THINKING_MODES`; ``None`` leaves the
    # model's own default in force (adaptive, on Claude 4.6+/5). The CLI passes
    # ``[generation] thinking``, which defaults to ``"disabled"``.
    thinking: str | None = None
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
    # PDF engine for page slicing (the per-window transcription payload), from
    # the workspace `pdf` config. None means the ghostscript default — which is
    # also what library callers with no workspace get.
    pdf_config: PdfConfig | None = None
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
    # The tag vocabulary every concept — model-emitted or replayed from cache —
    # resolves against, in BOTH labeling and rendering. A closed one makes the
    # seed a contract: the emitted docset: tags are a subset of its names, and
    # unmatched content renders as dg:chunk with its text intact. None = open,
    # i.e. coin freely (today's behavior, and the right default for a library
    # caller). Whether a seed closes the vocabulary is CLI policy, not a
    # property of the seed: the CLI closes on a vocabulary a PERSON authored
    # (`--schema-path`, or one a previous run remembered) and not on one it
    # derived from its own labels — and `--extend-schema` keeps an authored
    # vocabulary open so labeling may add to it.
    vocab: TagVocab | None = None
    progress: Callable[[str], None] | None = field(default=None)
    # Workspace to record LLM usage into. When set (and ``debug`` is True), the
    # transcription/labeling calls append rows to ``usage.jsonl``; None disables
    # recording (library/test callers that don't want telemetry).
    workspace: Workspace | None = None
    # Batch mode (see BatchOptions). None = the synchronous pipeline, exactly
    # as without this field.
    batch: BatchOptions | None = None


def _config(
    opts: ConvertOptions, model: str | None = None, *, operation: str | None = None
) -> llm.LLMConfig:
    return llm.LLMConfig(
        model=model or opts.model,
        api_key=opts.api_key,
        api_base=opts.api_base,
        temperature=opts.temperature,
        max_tokens=opts.max_tokens,
        thinking=opts.thinking,
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
    on_label_error: Callable[[str, dict[str, str]], None] | None = None,
    on_off_schema: Callable[[str, Counter[str]], None] | None = None,
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

    Pass *on_label_error* — called ``(name, {code, message})`` — to learn that a
    document's *labeling* could not reach the model at all (auth, bad model id,
    connection), as distinct from a soft "produced few/no labels" outcome. The
    document still renders (unlabeled), so this lets the caller surface a
    misconfigured ``label_model`` without discarding the transcription. Labeling
    completes before any ``on_output`` fires, so a per-file result built in
    ``on_output`` can read whatever this reported.

    Pass *on_off_schema* — called ``(name, Counter[concept])`` — to learn which
    concepts fell outside an AUTHORED ``options.vocab`` for a document: refused
    under a closed vocabulary, coined under one that extends. Fired only for
    documents that had any, before their output is emitted.

    *prior_docs* (already-generated docs from cache) are re-rendered so the
    whole docset stays consistent as its schema/roster grows; any whose render
    changes is re-emitted via *on_output* (skipped if unchanged per
    *prior_outputs*).
    """
    opts = options
    vocab = opts.vocab or OPEN_VOCAB
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
                pdf_config=opts.pdf_config,
            )
        except Exception as exc:
            log(f"[transcribe] {path.name} FAILED: {exc}; skipping")
            transcribe_errors[path.name] = short_error_message(exc)
            return None

    # Windows within a document are serial (the `continues` contract);
    # documents are independent, so transcribe them concurrently.
    workers = max(1, min(opts.max_parallel_docs, len(paths)))
    if opts.batch is not None:
        block_lists = _transcribe_batched(paths, opts, opts.batch, transcribe_errors, log)
    elif workers == 1 or len(paths) == 1:
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
    # them into one row (gated on --debug via the config). The labeling model is
    # always configured independently of transcription (it may name a different
    # provider), so it gets its own LLMConfig with its own credentials.
    label_config = llm.LLMConfig(
        model=opts.label_model,
        api_key=opts.label_api_key,
        api_base=opts.label_api_base,
        temperature=opts.temperature,
        max_tokens=opts.max_tokens,
        thinking=opts.thinking,
        workspace=opts.workspace,
        debug=opts.debug,
        operation=OPERATION_LABEL,
    )
    label_config.context = {"doc_count": len(docs)}

    # An extendable authored vocabulary gets its additions PLANNED, once, here
    # — before labeling, because the vocabulary they form has to govern the
    # render too, and `render_dgml` runs after `label_documents` returns.
    #
    # Coining freely during labeling produced an output vocabulary larger than
    # an unseeded run's, most of it not the user's, because supplying a schema
    # skips the planning pass and leaves labeling inventing per document. One
    # gap-planning call over every skeleton names the shared roles the schema
    # misses; closing over the union keeps the additions a bounded, reviewable
    # set instead of an open tail.
    # Batch mode: the pass's docset-wide calls (roster planning, gap planning,
    # concept descriptions) run as one-unit batch stages.
    single_runner = (
        None
        if opts.batch is None
        else batch_single_call_runner(
            opts.batch, api_key=opts.label_api_key, api_base=opts.label_api_base, log=log
        )
    )
    gap_seed: dict[str, str] = {}
    if vocab.extends and opts.schema_seed is not None and docs:
        with llm.record_usage_for(label_config), single_calls_through(single_runner):
            gap_seed = plan_concept_roster(
                docs,
                config=label_config,
                cache_dir=opts.cache_dir,
                debug=opts.debug,
                log=log,
                refine=False,  # over-proposing is the failure mode here
                existing={tag.name: tag.role for tag in opts.schema_seed.tags.values()},
            )
        if gap_seed:
            vocab = vocab.with_additions(gap_seed)
            log(
                f"Pass B.1: vocabulary bounded at {len(vocab.supplied)} authored "
                f"+ {len(vocab.added)} planned tag(s)"
            )
        else:
            # Planning is best-effort — it returns {} both when the schema
            # genuinely covers the documents and when the call failed. Closing
            # on an empty result would silently turn an extend run into a
            # strict one, which is a stricter contract than the user asked
            # for, so degrade to unbounded coining instead.
            log("Pass B.1: no gap concepts planned; labeling may coin unbounded")

    batch_label = None
    label_document = None
    if opts.batch is not None and opts.batch.label:
        batch_label = _batch_labeler(docs, opts, opts.batch, label_config, log)
        label_document = _per_document_labeler(docs, opts, opts.batch, label_config, log)
    elif opts.batch is not None and docs:
        log(f"Pass B: labeling stays synchronous under batch mode ({BATCH_LABEL_OFF})")
        opts.batch.stats["label"] = {"skipped": BATCH_LABEL_OFF, "mode": LABEL_MODE_SYNC}
    with llm.record_usage_for(label_config), single_calls_through(single_runner):
        label_documents(
            docs,
            config=label_config,
            cache_dir=opts.cache_dir,
            debug=opts.debug,
            log=log,
            roster_seed=opts.roster_seed,
            schema_seed=opts.schema_seed,
            gap_seed=gap_seed or None,
            vocab=vocab,
            on_label_error=on_label_error,
            on_off_schema=on_off_schema,
            batch_label=batch_label,
            label_document=label_document,
        )

    def _emit(item: tuple[str, list[Block]]) -> tuple[str, str]:
        # With dgml_header set, the product output is the final dg:chunk dgml
        # (concept tags where labeled, dg:chunk scaffolding otherwise, value
        # typing). Without it, the plain structure-attribute form is returned
        # (library/test shape). The compact concept render and the
        # structure-attribute XML are kept as debug artifacts in the cache.
        name, blocks = item
        if opts.dgml_header:
            xml = render_dgml(
                blocks, header=opts.dgml_header, parent_map=opts.parent_map, vocab=vocab
            )
        else:
            xml = render_semantic_xml(blocks)
        # Stream to the sink (freed immediately) or accumulate for the return.
        if on_output is not None:
            on_output(name, xml)
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
        return name, xml

    # The per-document sink is the expensive half of a run — grounding plus the
    # semantic-link model calls — and documents are independent, so it runs on
    # the same bounded pool transcription uses. Only when there is no sink is
    # this pure rendering, which is cheap enough to stay inline.
    # `map_concurrent` returns results in input order, so a caller folding them
    # into shared state still sees a deterministic sequence; sinks that mutate
    # caller state must key by document and order the fold themselves.
    emit_workers = opts.max_parallel_docs if on_output is not None else 1
    emitted = map_concurrent(_emit, list(docs.items()), max_workers=emit_workers)
    outputs: dict[str, str] = {} if on_output is not None else dict(emitted)

    # Re-render prior docs whose rendered XML changed and emit only those. All
    # concepts are docset:-namespaced, so sharing does not shift prefixes; a
    # prior render still changes when entity-container grouping moves as the
    # docset's schema/roster grows (or when migrating legacy dg:-namespaced
    # concepts to docset:).
    if prior_docs and opts.dgml_header and on_output is not None:
        # Rendering is cheap and decides *which* priors changed, so it stays
        # inline and in order; only the sink — grounding plus the link model
        # calls — goes on the pool.
        changed: list[tuple[str, str]] = []
        for name, blocks in prior_docs.items():
            xml = render_dgml(
                blocks, header=opts.dgml_header, parent_map=opts.parent_map, vocab=vocab
            )
            if prior_outputs is not None and prior_outputs.get(name) == xml:
                continue
            log(f"re-rendering {name} (docset render changed)")
            changed.append((name, xml))
        map_concurrent(lambda item: on_output(*item), changed, max_workers=opts.max_parallel_docs)
    return outputs


def _transcribe_batched(
    paths: list[Path],
    opts: ConvertOptions,
    batch: BatchOptions,
    transcribe_errors: dict[str, str],
    log: Callable[[str], None],
) -> list[list[Block] | None]:
    """Pass A for every document through the batch endpoint, one wave per
    window position (window k of every document together).

    Same per-document contract as the sync ``_transcribe``: one fresh config
    per document; a document that fails (unreadable input, or a failed
    request or parse) is logged, recorded in *transcribe_errors* with the same
    short reason, and dropped; a cached document makes no request and writes
    no usage row. Each document's usage row is the one sync would write,
    tagged ``tier="batch"`` (the driver records it from the document's own
    config, which the generator stamps with ``{"doc": name}``).
    """
    from dgml_core.batch import Unit, run_stage

    def failed(path: Path, exc: BaseException) -> None:
        log(f"[transcribe] {path.name} FAILED: {exc}; skipping")
        transcribe_errors[path.name] = short_error_message(exc)

    from dgml_core.batch.jobs import active_session

    job = active_session()

    def load(path: Path) -> bytes:
        def convert() -> bytes:
            return document.load_document_as_pdf(path, converters=opts.converters or {})

        on_demand = path.suffix.lower() != ".pdf" and not path.with_suffix(".pdf").exists()
        if job is None or not on_demand:
            return convert()
        # A document converted on demand (no PDF persisted at ingest) is not
        # byte-reproducible either; pin it to the job so its slices are.
        data = job.rewind_input(f"generate/converted/{path.name}.pdf", convert)
        assert data is not None
        return bytes(data)

    units: list[Unit] = []
    for path in paths:
        try:
            pdf_bytes = load(path)
        except Exception as exc:
            failed(path, exc)
            continue
        config = _config(opts, operation=OPERATION_TRANSCRIBE)
        steps = transcribe_steps(
            pdf_bytes,
            doc_name=path.name,
            config=config,
            window_size=opts.window_size,
            cache_dir=opts.cache_dir,
            debug=opts.debug,
            log=log,
            page_text_dir=(opts.page_text_dirs or {}).get(path.name),
            pdf_config=opts.pdf_config,
        )
        units.append(Unit(name=path.name, config=config, steps=steps))

    results: dict[str, list[Block]] = {}
    if units:
        executor = make_batch_executor(
            batch, model=opts.model, api_key=opts.api_key, api_base=opts.api_base, log=log
        )
        outcomes = run_stage(units, executor, log=log, stage="transcribe")
        batch.stats["transcribe"] = executor.stats.to_json()
        by_name = {path.name: path for path in paths}
        for name, outcome in outcomes.items():
            if outcome.error is not None:
                failed(by_name[name], outcome.error)
            else:
                results[name] = outcome.result
    return [results.get(path.name) for path in paths]


#: What labeling one document returns: (warnings, label_error, off_schema).
LabelResult = tuple[list[str], dict[str, str] | None, list[str]]


def _batch_labeler(
    docs: Mapping[str, list[Block]],
    opts: ConvertOptions,
    batch: BatchOptions,
    label_config: llm.LLMConfig,
    log: Callable[[str], None],
) -> Callable[[list[str], dict[str, RosterEntry], TagVocab], dict[str, LabelResult] | None]:
    """The ``batch_label`` hook :func:`label_documents` calls once its roster is
    built: label every document as one batch stage when the roster cannot
    change between documents (``mode: "all-at-once"``), else return ``None``
    — the serial loop then labels one document per batch stage through
    :func:`_per_document_labeler`.

    Every document shares *label_config*, as the serial loop does, so the
    documents' usage folds into the one labeling-pass row that
    ``convert_batch``'s ``record_usage_for(label_config)`` scope writes — the
    driver marks that row ``tier="batch"``. A document whose generator raises
    re-raises here, first in input order, as the serial loop would have. A
    failure of the batch stage itself (``outcome.stage_error``) does not: the
    documents labeled before it keep their labels, and each unfinished one is
    reported with a ``label_error`` carrying that failure — its transcription
    is kept, as for an unreachable label model.
    """

    def label(
        order: list[str], roster: dict[str, RosterEntry], vocab: TagVocab
    ) -> dict[str, LabelResult] | None:
        from dgml_core.generation.label import label_document_steps

        reason = unbatchable_label_reason(vocab, roster)
        if reason is not None:
            # _per_document_labeler takes over, one document per stage.
            log(f"Pass B: labeling one document per batch stage, in order ({reason})")
            return None
        if not order:
            return {}

        from dgml_core.batch import Unit, run_stage

        units = [
            Unit(
                name=name,
                config=label_config,
                steps=label_document_steps(
                    name,
                    docs[name],
                    roster,
                    config=label_config,
                    cache_dir=opts.cache_dir,
                    debug=opts.debug,
                    log=log,
                    vocab=vocab,
                ),
            )
            for name in order
        ]
        executor = make_batch_executor(
            batch,
            model=opts.label_model,
            api_key=opts.label_api_key,
            api_base=opts.label_api_base,
            log=log,
            role="label",
        )
        log(f"Pass B: labeling {len(units)} doc(s) as one batch stage (the roster is fixed)")
        outcomes = run_stage(units, executor, log=log, stage="label")
        batch.stats["label"] = {**executor.stats.to_json(), "mode": LABEL_MODE_ALL_AT_ONCE}
        for name in order:
            outcome = outcomes[name]
            if outcome.error is not None and not outcome.stage_error:
                raise outcome.error
        results: dict[str, LabelResult] = {}
        for name in order:
            outcome = outcomes[name]
            if outcome.error is None:
                results[name] = outcome.result
                continue
            results[name] = _stage_failed_label(name, outcome.error)
        return results

    return label


def _stage_failed_label(name: str, error: BaseException) -> LabelResult:
    """The label result of a document a failed batch stage left unlabeled:
    a warning and a ``label_error`` carrying that failure (its transcription
    is kept, as for an unreachable label model)."""
    code = str(getattr(error, "code", type(error).__name__))
    message = short_error_message(error)
    return [f"labeling {name} failed: {message}"], {"code": code, "message": message}, []


def _per_document_labeler(
    docs: Mapping[str, list[Block]],
    opts: ConvertOptions,
    batch: BatchOptions,
    label_config: llm.LLMConfig,
    log: Callable[[str], None],
) -> Callable[[str, list[Block], dict[str, RosterEntry], TagVocab], LabelResult]:
    """The ``label_document`` hook of :func:`label_documents` under batch
    labeling: label ONE document as one batch stage (``mode: "per-document"``).
    Only reached when the vocabulary can change between documents (or the run
    is pilot-staged); otherwise :func:`_batch_labeler` labels them all at once.

    Open-vocabulary labeling is serial because each document is prompted with
    the roster as the documents before it left it; labeling them all at once
    against a frozen roster measurably hurts cross-document consistency, so
    it is never done. Instead :func:`label_documents` keeps its serial loop —
    same order, pilot stage and ``_promote_pilot`` included — and calls this
    hook for each document in turn: one :func:`~dgml_core.batch.run_stage`
    over that document's :func:`label_document_steps`. Its chunks fan out
    into one wave (they share the document's roster snapshot), the section
    retry follows in the next, and the roster is updated in chunk order once
    the document has finished — so every request, every label and the roster
    handed to the next document are byte-identical to the synchronous run.
    The price is latency: about one batch round trip per document (two when
    a section retry or a bisect fires).

    Usage is the synchronous run's: every document shares *label_config*, so
    it folds, in document order, into ``convert_batch``'s one labeling-pass
    row, marked ``tier="batch"`` by the driver. One executor serves every
    document, so ``batch.stats["label"]`` is its cumulative
    ``WaveStats.to_json()`` plus ``mode`` and ``documents``. A document whose
    generator raises re-raises here, as the serial loop would. A failure of
    the batch stage itself (``outcome.stage_error``) leaves that document —
    and every later one, without another attempt — with a ``label_error``
    carrying the failure, as the all-at-once labeler does. In job mode
    (``--no-wait``) a pending wave pauses the whole run; the resume replays
    every earlier document's stage from the job's store and continues.
    """
    executor: BatchExecutor | None = None
    stage_failure: BaseException | None = None
    labeled = 0

    def label_one(
        name: str, blocks: list[Block], roster: dict[str, RosterEntry], vocab: TagVocab
    ) -> LabelResult:
        nonlocal executor, stage_failure, labeled
        from dgml_core.batch import Unit, run_stage
        from dgml_core.generation.label import label_document_steps

        if stage_failure is not None:
            return _stage_failed_label(name, stage_failure)
        if executor is None:
            executor = make_batch_executor(
                batch,
                model=opts.label_model,
                api_key=opts.label_api_key,
                api_base=opts.label_api_base,
                log=log,
                role="label",
            )
        labeled += 1
        log(f"Pass B: {name}: labeling as batch stage {labeled}/{len(docs)} (per document)")
        unit = Unit(
            name=name,
            config=label_config,
            steps=label_document_steps(
                name,
                blocks,
                roster,
                config=label_config,
                cache_dir=opts.cache_dir,
                debug=opts.debug,
                log=log,
                vocab=vocab,
            ),
        )
        outcome = run_stage([unit], executor, log=log, stage="label")[name]
        batch.stats["label"] = {
            **executor.stats.to_json(),
            "mode": LABEL_MODE_PER_DOCUMENT,
            "documents": labeled,
        }
        if outcome.error is None:
            return cast(LabelResult, outcome.result)
        if not outcome.stage_error:
            raise outcome.error
        stage_failure = outcome.error
        log(
            f"Pass B: batch labeling failed at {name}; it and every later document "
            f"stay unlabeled: {short_error_message(outcome.error)}"
        )
        return _stage_failed_label(name, outcome.error)

    return label_one
