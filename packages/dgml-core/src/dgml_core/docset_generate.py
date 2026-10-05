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

"""Generating a DocSet: the whole ``docset generate`` orchestration as one call.

:func:`generate_docset` converts every file in a DocSet to DGML XML via the
typed-block pipeline, grounds each rendered document in place, adds semantic
links, measures coverage, and refreshes the docset's schema artifacts — the
layer *above* :func:`~dgml_core.generation.convert_batch` that was previously
reachable only through the ``dgml`` CLI. Progress is diagnostic and goes to
this module's logger at INFO (see the Logging section of the package
CLAUDE.md), or to the ``on_progress`` callback when one is passed; every
per-file fact an embedding application needs comes back structured on the
returned :class:`GenerateReport`, never as text to parse.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import layout
from .conversion import load_conversion_config
from .docsets import DocSetStore
from .errors import (
    DgmlError,
    EmptyDocSet,
    GenerationFailed,
    InvalidArgument,
    short_error_message,
)
from .files import FileStore
from .models import DocSet
from .pages import load_pdf_config
from .storage import Workspace

if TYPE_CHECKING:
    from .generation.schema import Schema

__all__ = ["GenerateFileResult", "GenerateReport", "generate_docset"]

logger = logging.getLogger(__name__)


#: Distinct rejected concept names reported per file in `unmatched_concepts`.
#: Enough to recognize the pattern (aliases? new roles? junk?) without turning
#: a results payload into a log.
_UNMATCHED_EXAMPLES = 10


@dataclass(frozen=True)
class GenerateFileResult:
    """One file's outcome in a :class:`GenerateReport`.

    ``status`` is ``"converted"``, ``"skipped"`` (already generated) or
    ``"failed"``. Every other field is populated only where it applies —
    ``output`` on success, ``error`` on failure, and the per-pass fields
    (``grounded`` and friends, ``links``, ``label_error``, ``link_error``,
    ``off_schema_concepts``) only on a converted file, mirroring the CLI's
    conditional JSON keys. ``to_json`` renders exactly the row the CLI emits.
    """

    status: str
    file_id: str
    source: str
    #: Store key of the rendered ``<stem>.dgml.xml`` (converted and skipped files).
    output: str | None = None
    #: ``{code, message}`` for a failed file.
    error: dict[str, str] | None = None
    #: Semantic links applied to the final XML (converted files).
    links: int | None = None
    #: Whether grounding annotated the rendered XML (converted files).
    grounded: bool | None = None
    matched_token_pct: float | None = None
    elements_annotated: int | None = None
    #: ``{code, message}`` when grounding failed (``grounded`` is False).
    grounding_error: dict[str, str] | None = None
    #: ``{code, message}`` when the labeling model was unreachable.
    label_error: dict[str, str] | None = None
    #: Short reason when the semantic-link pass could not complete.
    link_error: str | None = None
    #: ``{count, distinct, examples}`` for concepts outside an authored
    #: vocabulary — rejections under a strict schema, additions under
    #: extend-schema (see :class:`GenerateReport.schema_extended`).
    off_schema_concepts: dict[str, Any] | None = None

    def to_json(self, *, extend_schema: bool = False) -> dict[str, Any]:
        """This file's row in the ``results`` array, keys in the CLI's order
        and present only where the matching pass ran or failed."""
        row: dict[str, Any] = {
            "status": self.status,
            "file_id": self.file_id,
            "source": self.source,
        }
        if self.output is not None:
            row["output"] = self.output
        if self.error is not None:
            row["error"] = dict(self.error)
        if self.links is not None:
            row["links"] = self.links
        if self.grounded is not None:
            row["grounded"] = self.grounded
            if self.grounded:
                row["matched_token_pct"] = self.matched_token_pct
                row["elements_annotated"] = self.elements_annotated
            else:
                row["grounding_error"] = dict(self.grounding_error or {})
        if self.label_error is not None:
            row["label_error"] = dict(self.label_error)
        if self.link_error is not None:
            row["link_error"] = self.link_error
        if self.off_schema_concepts is not None:
            key = "added_concepts" if extend_schema else "unmatched_concepts"
            row[key] = dict(self.off_schema_concepts)
        return row


@dataclass(frozen=True)
class GenerateReport:
    """What :func:`generate_docset` did, per file and in aggregate.

    ``skipped`` + ``failed`` + ``converted`` always cover every file assigned
    to the docset (``total``), so the three lists are the whole story — a
    partial failure is a ``failed`` entry here, never an exception.
    ``to_json`` renders the exact envelope ``dgml docset generate`` prints.
    """

    docset: DocSet
    total: int
    skipped: list[GenerateFileResult]
    failed: list[GenerateFileResult]
    converted: list[GenerateFileResult]
    #: Already-generated documents re-rendered (no re-LLM) because the growing
    #: docset changed their namespacing.
    rerendered: list[str]
    #: The docset's store key (``docsets/<id>``) — the prefix the per-file DGML
    #: lives under. A store-native key, not a local path, so it is meaningful
    #: on any backend.
    output_key: str
    #: Store key of the coverage report, when one was written (coverage on +
    #: debug), else None.
    coverage_report_key: str | None
    #: The effective transcription/labeling models and where they came from
    #: (workspace config, a profile/file, and/or explicit overrides), so every
    #: run's model choice is recorded in its output, not just in config.toml.
    model: str
    label_model: str
    model_source: str
    #: Whether the vocabulary was an authored schema kept open (extend-schema):
    #: decides if off-schema concepts render as ``added_concepts`` (coined and
    #: used) or ``unmatched_concepts`` (refused).
    schema_extended: bool

    @property
    def results(self) -> list[GenerateFileResult]:
        """Every file's outcome, in the CLI's order: skipped, failed, converted."""
        return [*self.skipped, *self.failed, *self.converted]

    def to_json(self) -> dict[str, Any]:
        """The single ``docset generate`` envelope, built the same way whether
        or not any file actually needed converting (so the paths can't drift)."""
        return {
            "docset_id": self.docset.id,
            "docset_name": self.docset.name,
            "summary": {
                "total": self.total,
                "converted": len(self.converted),
                "skipped": len(self.skipped),
                "failed": len(self.failed),
            },
            "models": {
                "model": self.model,
                "label_model": self.label_model,
                "source": self.model_source,
            },
            "output_key": self.output_key,
            "coverage_report": self.coverage_report_key,
            "results": [r.to_json(extend_schema=self.schema_extended) for r in self.results],
            "rerendered": list(self.rerendered),
        }


def _load_schema_roster(path: Path) -> dict[str, str]:
    """Load a flat ``{concept: description}`` JSON roster (the shape emitted at
    ``cache/concept_roster.json``) into a roster seed.

    Used for automatic roster reuse on an incremental generate, so newly-added
    documents stay tag-consistent with the docset's existing vocabulary. Concept
    keys are sanitized to PascalCase; descriptions are truncated to the roster
    hint length. Raises ``InvalidArgument`` on a missing / malformed file or one
    with no usable concepts.
    """
    from .generation.blocks import sanitize_concept

    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InvalidArgument(f"roster file not found: {path}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidArgument(f"roster is not valid JSON ({path}): {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidArgument(
            f"roster must be a JSON {{concept: description}} object, got "
            f"{type(raw).__name__} ({path})"
        )

    roster: dict[str, str] = {}
    for name, description in raw.items():
        concept = sanitize_concept(str(name))
        if concept:
            roster[concept] = str(description)[:60]
    if not roster:
        raise InvalidArgument(f"roster produced no usable concepts ({path})")
    return roster


def _schema_parent_map(schema: Schema) -> dict[str, str]:
    """The leaf → container map ``render_dgml`` groups entity containers with.

    Names pass through VERBATIM (only ``sanitize_tag_name`` for XML validity):
    the map's keys and values must be the same strings the labeling roster and
    the emitted tags use, and ``sanitize_concept`` — written for model output —
    would fold names like ``Notes`` to nothing and silently break the pairing.
    """
    from .generation.schema import sanitize_tag_name

    parent_map: dict[str, str] = {}
    for tag in schema.tags.values():
        if tag.name and tag.parent_role:
            parent_map[sanitize_tag_name(tag.name)] = sanitize_tag_name(tag.parent_role)
    return parent_map


def _load_schema_seed(
    path: Path, label: str = "schema"
) -> tuple[Schema, dict[str, str], list[str]]:
    """Load a user-supplied tag schema into ``(schema, parent_map, notes)``.

    ``schema_path`` takes any of four forms, detected by CONTENT rather than
    by file extension so the parameter stays one parameter:

    - a plain newline-delimited tag list (``#`` comments and blanks ignored);
    - a JSON ``{name: one-line description}`` object — the recommended form;
    - an exported ``schema.json`` (Schema v1: a ``tags`` map of
      ``name -> {role, kind, examples, parent_role}``);
    - its lossless RELAX NG Compact render ``full-schema.rnc`` (``.rnc``
      suffix; the ``# Field: value`` comment contract carries the same fields).

    The schema seeds the labeling vocabulary with full fidelity — role
    descriptions, curated examples, kind, hierarchy (via
    ``ConvertOptions.schema_seed``); each tag's ``parent_role`` also becomes
    the leaf → container ``parent_map`` that drives entity-container grouping
    in ``render_dgml``. *notes* are the loader's remarks about anything it had
    to change, for the progress log.

    Deliberately NOT built on ``_load_schema_roster``: that reader exists for
    the legacy ``concept_roster.json`` reuse path, truncates descriptions to 60
    characters, and pushes names through ``sanitize_concept`` — the exact
    mangling an authored schema must not suffer.

    Raises ``InvalidArgument`` on a missing file, or on anything the loader
    cannot read unambiguously. A bad schema must fail HERE, at load, and never
    as a tag that quietly failed to appear hours later. *label* names whatever
    asked for the file, since the automatic-reuse path reads a stored schema
    that the user did not name on the command line.
    """
    from .generation.schema import parse_authored_schema, schema_from_dict

    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InvalidArgument(f"{label} file not found: {path}") from exc
    except OSError as exc:
        raise InvalidArgument(f"{label} could not be read ({path}): {exc}") from exc

    try:
        if Path(path).suffix.lower() == ".rnc":
            from .generation.rnc import rnc_to_schema_dict

            schema, notes = schema_from_dict(rnc_to_schema_dict(text))
        else:
            schema, notes = parse_authored_schema(text)
    except InvalidArgument as exc:
        raise InvalidArgument(f"{label} {path}: {exc}") from exc
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        raise InvalidArgument(f"{label} {path} could not be parsed: {exc}") from exc

    return schema, _schema_parent_map(schema), notes


def _has_generated_tree(xml_text: str) -> bool:
    """True when a ``<stem>.dgml.xml`` holds a generated document tree — the
    generate skip test. An extraction-only file (whose root has just a
    ``dg:extraction`` child) or an unparseable one returns False so generation
    proceeds and (re)builds the tree."""
    from .extraction_xml import has_document_tree

    try:
        return has_document_tree(xml_text)
    except Exception:
        return False


def generate_docset(
    ws: Workspace,
    docset_id: str,
    *,
    generation_config: str | None = None,
    model: str | None = None,
    label_model: str | None = None,
    window_size: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 64000,
    thinking: str | None = None,
    max_parallel_docs: int = 4,
    cache_dir: Path | None = None,
    schema_path: Path | None = None,
    extend_schema: bool = False,
    reuse_roster: bool = True,
    coverage: bool = True,
    semlinks: bool = True,
    semlink_cache: bool = True,
    semlink_verify: bool = True,
    debug: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> GenerateReport:
    """Convert every file in a DocSet to DGML XML via the typed-block pipeline.

    Per window: flat JSON block transcription (``generation.model``); then ONE
    batch-wide semantic-labeling call across all documents
    (``generation.label_model``); then deterministic ``dg:chunk`` rendering. Word
    coverage is measured on the rendered XML unless ``coverage=False``.

    Each rendered ``<stem>.dgml.xml`` is then grounded in place against the
    file's page OCR — ``dg:origin`` boxes are written onto every element with
    text content (deterministic, no LLM). A file with no ``page_text/`` is
    left ungrounded with a warning rather than failing the run. ``debug``
    additionally writes the per-file ``<stem>.dgml.grounding_stats.json``.

    Per-file problems (a missing source, a duplicate filename, a transcription
    that produced nothing) are ``failed`` entries on the returned
    :class:`GenerateReport`, and the run continues — partial success, matching
    the pipeline's soft-failure design. What raises instead is what makes the
    whole run meaningless: :class:`~dgml_core.errors.DocSetNotFound`,
    :class:`~dgml_core.errors.EmptyDocSet`, a malformed style config, an
    unresolvable model, or a missing API key (the pre-flight failures, caught
    before any transcription spend).

    ``model`` / ``label_model`` / ``generation_config`` (a bundled profile name
    or a config-file path) override the workspace's merged ``generation``
    config, exactly like the CLI flags of the same names; the effective models
    and their source come back on the report. Progress lines log at INFO on
    this module's logger unless ``on_progress`` is passed, which then receives
    them instead — every fact a caller should branch on is on the report,
    never only in these strings.
    """
    from . import llm
    from .extraction_xml import carry_extraction_over, has_extraction
    from .generation import (
        ConvertOptions,
        convert_batch,
        resolve_generation_api_key,
        resolve_generation_config,
        resolve_generation_label_api_key,
        validate_generation_models,
    )
    from .generation import coverage as cov_mod
    from .generation import links as links_mod
    from .generation.blocks import Block, block_concept_labels
    from .generation.links import apply_plan, plan_links
    from .generation.pipeline import load_labeled_docs_from_cache
    from .generation.rnc import write_docset_rnc
    from .generation.to_semantic import build_header
    from .generation.vocab import TagVocab
    from .usage import OPERATION_LINKS
    from .xml_grounding import ground_dgml_xml

    log: Callable[[str], None] = on_progress if on_progress is not None else logger.info

    ds_store = DocSetStore(ws)
    file_store = FileStore(ws)

    ds = ds_store.get(docset_id)
    file_ids = ds_store.list_files(docset_id)
    if not file_ids:
        raise EmptyDocSet(f"DocSet '{docset_id}' has no files assigned.")

    # Validate the optional `style` config up front — before any LLM
    # transcription — rather than surfacing per-file during grounding. A
    # malformed section fails fast with STYLE_CONFIG_INVALID; a referenced-but-
    # unset `api_key_env` fails fast with AUTH_ERROR (the grounding-time style
    # pass is best-effort and would otherwise swallow this silently, after the
    # transcription spend).
    from .style_config import load_style_config, resolve_api_key

    style_config = load_style_config(ws)
    if style_config is not None:
        resolve_api_key(style_config)

    # Resolve the effective LLM models: the merged 'generation' config by default
    # (per-task field or [models] tier, each model with its own credentials since
    # the two may name different providers), optionally overlaid by
    # `generation_config` (a bundled profile or a config file) and/or `model` /
    # `label_model`. With no overrides this is load_generation_config and still
    # raises GENERATION_CONFIG_MISSING when nothing resolves a model.
    # `gen_model_source` records where the models came from and is echoed into
    # the report so the choice stays visible/recorded.
    gen_cfg, gen_model_source = resolve_generation_config(
        ws,
        config=generation_config,
        model=model,
        label_model=label_model,
    )
    gen_model = gen_cfg.model
    resolved_label_model = gen_cfg.label_model
    gen_api_key = resolve_generation_api_key(gen_cfg)
    gen_api_base = gen_cfg.api_base
    label_api_key = resolve_generation_label_api_key(gen_cfg)
    label_api_base = gen_cfg.label_api_base
    log(
        f"[models] transcription={gen_model} labeling={resolved_label_model} "
        f"(source: {gen_model_source})"
    )

    # Pre-flight — fail fast BEFORE any transcription spend on the two model
    # misconfigurations detectable offline: a malformed model string, or a
    # missing API key for either model's provider. A present-but-wrong key or a
    # well-formed-but-nonexistent model id can't be caught here; those surface
    # per file as label_error (see _on_label_error below). Mirrors the style-
    # config pre-flight above.
    validate_generation_models(gen_cfg, gen_api_key, label_api_key)

    # The semantic-link pass runs on the labeling model (and its credentials).
    # One config per DOCUMENT, never one shared by all of them. Documents are
    # linked concurrently on the emit pool, and `llm.record_usage_for` marks the
    # open aggregation scope on the config object itself — so a shared config
    # means the second document to start folds its tokens into whichever scope
    # opened first, and the row that lands names one document while covering
    # several. Per-document configs also give each row a `doc` context, so the
    # pass can be read per file rather than only in aggregate.
    def _link_config(doc_name: str) -> llm.LLMConfig:
        config = llm.LLMConfig(
            model=resolved_label_model,
            api_key=label_api_key,
            api_base=label_api_base,
            workspace=ws,
            debug=debug,
            operation=OPERATION_LINKS,
        )
        config.context = {"doc": doc_name}
        return config

    # The docset prefix is always the output base — schema.json,
    # coverage_report.json, cache/, and semantic/ live under it. Each file's
    # final .dgml.xml lands at its per-(docset, file) key (see
    # layout.dgml_xml_key) so placement is deterministic and stable.
    # The docset's store key (``docsets/<id>``) — the prefix the cache, coverage
    # report, and per-file DGML live under. Reported to the user and used to
    # build child keys; no directory is created here (the store owns that).
    # Slash-stripped because it is echoed as ``output_key`` in the JSON result,
    # where the trailing-slash form would be a breaking change; nothing
    # prefix-matches on it.
    output_key = layout.docset_prefix(docset_id).rstrip("/")

    # Resolve each assigned file into exactly one bucket so the summary counts
    # always sum to `total`: skipped (already converted), failed (source
    # missing, or a duplicate filename the pipeline can't disambiguate), or a
    # to-convert candidate. Partial success — a per-file problem is recorded
    # and the run continues, matching `dgml cluster`.
    skipped_results: list[GenerateFileResult] = []
    failed_results: list[GenerateFileResult] = []
    # original_filename → list of (file_id, out_xml_key, page_text_prefix).
    # Grouped by filename to detect collisions: convert_batch keys documents by
    # filename, so two files sharing a basename can't both convert in one run.
    candidates: dict[str, list[tuple[str, str, str | None]]] = {}
    # Already-generated docs, for whole-docset roster reuse + namespacing recompute.
    prior_stems: dict[str, str] = {}  # cache stem → original_filename
    prior_out_paths: dict[str, str] = {}  # original_filename → existing .dgml.xml key
    # original_filename → file id for grounding (resolves the file's page OCR).
    # Spans candidates *and* re-rendered prior docs; kept separate from
    # filename_to_fid so the failure-reconciliation loop stays candidate-only.
    name_to_fid: dict[str, str] = {}
    for fid in file_ids:
        record = file_store.get(fid)
        name = record.original_filename
        stem = Path(name).stem
        # Generation slices the persisted <stem>.pdf, or — for a file added before
        # conversions were persisted — the original source, converted on demand.
        # Both are store blobs under the file's prefix; materialized to a real
        # path for transcription just before convert_batch (below).
        if not (
            ws.blobs.blob_exists(layout.file_source_key(fid, f"{stem}.pdf"))
            or ws.blobs.blob_exists(layout.file_source_key(fid, name))
        ):
            failed_results.append(
                GenerateFileResult(
                    status="failed",
                    file_id=fid,
                    source=name,
                    error={
                        "code": "FILE_NOT_FOUND",
                        "message": f"no source PDF for file '{fid}'",
                    },
                )
            )
            log(f"Source missing for {name} (file '{fid}') — reported as failed")
            continue
        out_xml_key = layout.dgml_xml_key(docset_id, fid, stem)
        if ws.blobs.blob_exists(out_xml_key) and _has_generated_tree(
            ws.blobs.get_blob(out_xml_key).decode("utf-8")
        ):
            # Skip only when a generated document tree is present. An
            # extraction-only file (`extraction extract` ran before
            # `generate`) falls through and gets its tree built; _on_output
            # carries the existing dg:extraction over into the fresh render.
            skipped_results.append(
                GenerateFileResult(status="skipped", file_id=fid, source=name, output=out_xml_key)
            )
            prior_stems[stem] = name
            prior_out_paths[name] = out_xml_key
            name_to_fid[name] = fid  # in case it re-renders below and needs re-grounding
            log(f"Skipping {name} (already converted)")
            continue
        pt_prefix = layout.file_text_prefix(fid)
        candidates.setdefault(name, []).append(
            (fid, out_xml_key, pt_prefix if ws.blobs.list_blobs(pt_prefix) else None)
        )

    # Same-basename collision: the typed-block pipeline keys documents by
    # filename, so it can't tell two same-named files apart in one batch.
    # Fail them explicitly instead of silently dropping/misattributing output.
    convert_names: list[str] = []
    dgml_xml_keys: dict[str, str] = {}
    filename_to_fid: dict[str, str] = {}
    page_text_dirs: dict[str, Path] = {}
    page_text_prefixes: dict[str, str] = {}
    for name, group in candidates.items():
        if len(group) > 1:
            for fid, _out, _pref in group:
                failed_results.append(
                    GenerateFileResult(
                        status="failed",
                        file_id=fid,
                        source=name,
                        error={
                            "code": GenerationFailed.code,
                            "message": (
                                f"duplicate filename '{name}' within the docset; the "
                                "generation pipeline keys documents by filename, so give "
                                "each file a unique name before converting"
                            ),
                        },
                    )
                )
            log(f"Duplicate filename '{name}' across {len(group)} files — reported as failed")
            continue
        fid, out_xml_key, pt_pfx = group[0]
        convert_names.append(name)
        dgml_xml_keys[name] = out_xml_key
        filename_to_fid[name] = fid
        name_to_fid[name] = fid
        if pt_pfx is not None:
            page_text_prefixes[name] = pt_pfx

    # Coverage is computed (and its per-file summary logged) whenever the
    # caller didn't disable it, but the coverage_report.json *file* is an
    # intermediate artifact persisted only under `debug`.
    cov_report_key = layout.docset_coverage_report_key(docset_id) if (coverage and debug) else None
    written: list[GenerateFileResult] = []
    rerendered: list[str] = []
    cov_results: list[dict[str, Any]] = []
    # `_on_output` runs on convert_batch's document pool, so it records its
    # per-document results here — each document owns one key, so no two threads
    # write the same entry — and the three lists above are extended in a fixed
    # order once the batch has drained. Appending straight from the workers
    # would make the report depend on completion order.
    converted_by_name: dict[str, GenerateFileResult] = {}
    cov_by_name: dict[str, dict[str, Any]] = {}
    rerendered_by_name: dict[str, None] = {}
    # Already-generated docs reloaded from cache (populated below when there is
    # new work) so namespacing spans the whole docset and flipped originals
    # re-render. _on_output reads prior_outputs to route/flag them.
    prior_docs: dict[str, list[Block]] = {}
    prior_outputs: dict[str, str] = {}

    # name → short reason for a per-document transcription failure, so the
    # reconciliation loop below can name the cause in the report instead of
    # the generic "produced no output" message. The full error still goes to
    # the log at INFO via convert_batch's progress hook.
    gen_errors: dict[str, str] = {}
    # name → {code, message} when a file's labeling couldn't reach the model at
    # all (bad model id, wrong/absent key, network). Surfaced as label_error on
    # the converted entry so a misconfigured label_model is visible without
    # reading the progress log; the document still renders (unlabeled).
    # Labeling completes before any _on_output fires, so the entry below can
    # read this.
    label_errors: dict[str, dict[str, str]] = {}
    # name -> short reason when the semantic-link pass could not complete. The
    # document keeps its (unlinked) DGML, so without this a rate limit or a bad
    # model id looked exactly like "this document has no links".
    link_errors: dict[str, str] = {}
    # name -> {count, distinct, examples} for the concepts that fell outside an
    # AUTHORED vocabulary. Reported as `unmatched_concepts` under a strict
    # schema (refused, so the list is what the schema is missing) and as
    # `added_concepts` under extend_schema (coined and used, so the list is
    # the candidate set for the schema's next revision). Absent from a file's
    # entry when nothing went outside, like every other conditional field here.
    off_schema_concepts: dict[str, dict[str, Any]] = {}

    def _on_error(name: str, message: str) -> None:
        gen_errors[name] = message

    def _on_label_error(name: str, err: dict[str, str]) -> None:
        label_errors[name] = err
        log(f"[label] {name}: model unreachable ({err.get('message', '')})")

    def _on_off_schema(name: str, tally: Counter[str]) -> None:
        # Which names the model reached for outside the supplied schema. The
        # most actionable output of either mode — under strict these are gaps
        # to consider adding, under extend they are additions to review.
        # Reported per file, not just logged, so it is readable without the
        # progress log.
        off_schema_concepts[name] = {
            "count": sum(tally.values()),
            "distinct": len(tally),
            "examples": [concept for concept, _n in tally.most_common(_UNMATCHED_EXAMPLES)],
        }

    def _semlink_cache_key(xml_text: str) -> str:
        """Cache address for one document's semantic links.

        Keyed on what the plan depends on — the document's text and shape, via
        links.listing_digest — plus the labeling model, both link prompts, and
        whether the review pass runs. Attributes and tag names are deliberately
        not part of it (see links.listing_digest), so grounding a document or
        renaming its concepts hits rather than paying for the pass again. Parts
        are length-prefixed so two different inputs cannot concatenate to the
        same key.
        """
        digest = hashlib.sha256()
        for part in (
            links_mod.listing_digest(xml_text).encode("utf-8"),
            resolved_label_model.encode("utf-8"),
            links_mod.SYSTEM_PROMPT.encode("utf-8"),
            links_mod.VERIFY_SYSTEM_PROMPT.encode("utf-8") if semlink_verify else b"",
        ):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        prefix = layout.generation_cache_prefix(docset_id)
        return f"{prefix}semlinks/{digest.hexdigest()}"

    def _on_output(name: str, xml: str) -> None:
        xml_key = dgml_xml_keys[name]
        # A blob already at this key may carry extracted values — an
        # extraction-only file getting its tree now, or a full-extraction
        # file being re-rendered. Capture its dg:extraction before the fresh
        # render overwrites it; re-embedded below after grounding + semlinks.
        prior_with_extraction: str | None = None
        if ws.blobs.blob_exists(xml_key):
            try:
                prior_text = ws.blobs.get_blob(xml_key).decode("utf-8")
                if has_extraction(prior_text):
                    prior_with_extraction = prior_text
            except Exception:
                prior_with_extraction = None  # unparseable prior — nothing to carry
        ws.blobs.put_blob(xml_key, xml.encode("utf-8"))
        # Ground in place: re-parse the just-written tree, align it against the
        # file's page OCR, and rewrite <stem>.dgml.xml with dg:origin boxes.
        # Deterministic and free; a file with no page_text is left ungrounded.
        # Runs for re-rendered prior docs too — their fresh XML would otherwise
        # lose the boxes a previous run grounded in. Grounding needs a real path
        # (lxml); materialize the blob to a working copy, ground in place, then
        # persist the result (and its stats sidecar) back through the store.
        grounded: bool
        matched_token_pct: float | None = None
        elements_annotated: int | None = None
        grounding_error: dict[str, str] | None = None
        try:
            with ws.blobs.materialize(xml_key) as gpath:
                res = ground_dgml_xml(
                    ws,
                    name_to_fid[name],
                    gpath,
                    output_path=gpath,
                    force=True,
                    write_stats=debug,
                    debug=debug,
                )
                ws.blobs.put_blob(xml_key, gpath.read_bytes())
                if res.stats_path is not None and res.stats_path.exists():
                    ws.blobs.put_blob(
                        layout.pair_artifact_key(docset_id, name_to_fid[name], res.stats_path.name),
                        res.stats_path.read_bytes(),
                    )
        except DgmlError as exc:
            grounded = False
            grounding_error = {"code": exc.code, "message": str(exc)}
            log(f"[ground] {name}: not grounded ({exc})")
        else:
            grounded = True
            matched_token_pct = res.stats["matched_token_pct"]
            elements_annotated = res.stats["elements_annotated"]
            log(
                f"[ground] {name}: {res.stats['elements_annotated']} element(s), "
                f"{res.stats['matched_token_pct']}% tokens matched"
            )
        # Final step: add semantic links in place (dg:itemprop/dg:href). Runs on
        # re-rendered priors too — their fresh XML would otherwise lose the links.
        # The pass is a pure function of (grounded XML, labeling model, link
        # prompts), so it is content-addressed: a hit replays the exact bytes the
        # model call would have written, making a repeat run free rather than
        # merely cheaper. The applied-link count is cached alongside so the
        # reported `links` is identical on both paths.
        links_added = 0
        if semlinks:
            source = ws.blobs.get_blob(xml_key).decode("utf-8")
            plan_key = f"{_semlink_cache_key(source)}.json"
            try:
                cached = (
                    ws.blobs.get_blob(plan_key)
                    if semlink_cache and ws.blobs.blob_exists(plan_key)
                    else None
                )
                if cached is not None:
                    plan = json.loads(cached)
                    hit = " (cached)"
                else:
                    plan = plan_links(source, _link_config(name), verify=semlink_verify)
                    ws.blobs.put_blob(plan_key, json.dumps(plan).encode("utf-8"))
                    hit = ""
                # The plan is applied to the CURRENT tree either way, so a cache
                # hit and a fresh call write the same links onto whatever the
                # render and grounding just produced.
                linked, applied = apply_plan(source, plan)
                ws.blobs.put_blob(xml_key, linked.encode("utf-8"))
                # `applied` is what the XML actually carries, so the reported
                # count matches the document. What the plan asked for and did
                # not get is diagnosed separately — chiefly links discarded
                # because dg:itemprop/dg:href are attributes on the subject, so
                # a second link on one subject overwrites the first.
                links_added = len(applied)
                losses = links_mod.plan_losses(source, plan)
                folded = f", {losses.merged} merged" if losses.merged else ""
                lost = f", {losses.displaced} displaced" if losses.displaced else ""
                nested = f", {losses.nested} nested dropped" if losses.nested else ""
                log(f"[semlinks] {name}: {links_added} link(s){folded}{lost}{nested}{hit}")
            except Exception as exc:  # a link-pass failure must not lose the DGML
                link_errors[name] = short_error_message(exc)
                log(f"[semlinks] {name}: skipped ({exc})")
        # Re-embed the prior dg:extraction last, after grounding + semlinks
        # have finished rewriting the tree, so the extraction subtree is
        # spliced in verbatim and never run through those passes.
        if prior_with_extraction is not None:
            try:
                merged = carry_extraction_over(
                    prior_with_extraction, ws.blobs.get_blob(xml_key).decode("utf-8")
                )
                ws.blobs.put_blob(xml_key, merged.encode("utf-8"))
                log(f"[extraction] {name}: carried dg:extraction over into the fresh render")
            except Exception as exc:  # never lose the fresh DGML over the merge
                log(f"[extraction] {name}: dg:extraction NOT carried over ({exc})")
        if name in prior_outputs:
            # an already-generated doc whose namespacing flipped
            rerendered_by_name[name] = None
            return
        pt_dir = page_text_dirs.get(name)
        if coverage and pt_dir is not None:
            result = cov_mod.compute_coverage(xml, name, page_text_dir=pt_dir)
            log(cov_mod.coverage_summary_line(result))
            # debug: how many assigned labels reached the DGML (e.g. "180
            # labels exported over 200 total"). Reloads labeled blocks from
            # cache; best-effort, recorded under `label_propagation`.
            if debug:
                try:
                    stem = Path(name).stem
                    labeled = load_labeled_docs_from_cache(cache, [stem]).get(stem)
                    if labeled is not None:
                        prop = cov_mod.compute_label_propagation(
                            block_concept_labels(labeled), xml, source_name=name
                        )
                        result["label_propagation"] = {
                            k: v for k, v in prop.items() if k != "source"
                        }
                        log(cov_mod.label_propagation_summary_line(prop))
                except Exception as exc:  # debug-only diagnostic — never fatal
                    log(f"[labels] {name}: propagation check skipped ({exc})")
            cov_by_name[name] = result
        # Each field set only when that step failed, like grounding_error,
        # which is present only when grounded is False.
        converted_by_name[name] = GenerateFileResult(
            status="converted",
            file_id=filename_to_fid[name],
            source=name,
            output=xml_key,
            links=links_added,
            grounded=grounded,
            matched_token_pct=matched_token_pct,
            elements_annotated=elements_annotated,
            grounding_error=grounding_error,
            label_error=label_errors.get(name),
            link_error=link_errors.get(name),
            off_schema_concepts=off_schema_concepts.get(name),
        )

    if convert_names:
        # The cache always exists — it holds functional files the next run
        # reloads (blocks, per-chunk labels, concept_roster.json). `debug` only
        # controls whether the extra debug-only artifacts are also written
        # (threaded via ConvertOptions.debug below).
        # The cache is a store-backed working directory: its blobs are pulled in
        # before the run and pushed back after (LocalStore works in place, no
        # copy). A `cache_dir` override stays a plain local directory (explicit
        # scratch, not store-backed). schema.json — written by labeling next to
        # the cache — rides along, persisted as the docset's generation-schema
        # blob (exact bytes, so no reserialization drift).
        with contextlib.ExitStack() as _cache_stack:
            if cache_dir is not None:
                cache = Path(cache_dir)
            else:
                cache = _cache_stack.enter_context(
                    ws.blobs.working_dir(layout.generation_cache_prefix(docset_id))
                )
            schema_key = layout.docset_generation_schema_key(docset_id)
            authored_key = layout.docset_authored_schema_key(docset_id)
            schema_json_local = cache.parent / "schema.json"
            authored_local = cache.parent / layout.AUTHORED_SCHEMA_FILE
            if cache_dir is None:
                for key, dest in ((schema_key, schema_json_local), (authored_key, authored_local)):
                    if not dest.exists() and ws.blobs.blob_exists(key):
                        ws.blobs.download_blob(key, dest)
            roster_path = Path(cache) / "concept_roster.json"
            schema_seed = None
            roster_seed: dict[str, str] | None = None
            parent_map_seed: dict[str, str] = {}
            # Set on a schema_path run: the authored vocabulary, persisted
            # below to a slot derive_schema never writes, so the next run seeds
            # from what the user wrote rather than from this run's own output.
            authored_seed: Schema | None = None
            # Whether the seed in hand is one a PERSON wrote (this run's
            # schema_path, or one a previous run remembered) as opposed to one
            # the pipeline derived from its own labels. Only the former closes.
            authored = False
            if schema_path is not None:
                schema_seed, parent_map_seed, schema_notes = _load_schema_seed(Path(schema_path))
                authored_seed = schema_seed
                authored = True
                log(
                    f"Loaded schema: {len(schema_seed.tags)} concept(s), "
                    f"{len(parent_map_seed)} container link(s) from {schema_path}"
                )
                for note in schema_notes:
                    log(f"[schema] {note}")
            elif reuse_roster:
                # Incremental reuse in precedence order: the vocabulary the USER
                # authored first (never overwritten by derive_schema), then the
                # derived schema.json — full fidelity (role descriptions,
                # observed examples, kind, hierarchy) — then the flat
                # cache/concept_roster.json fallback. Only the authored slot
                # carries hierarchy through: entity-container grouping stays
                # something the user opted into, never inferred from a run's
                # own observations.
                from .generation.schema import Schema

                if authored_local.exists():
                    try:
                        schema_seed, parent_map_seed, _notes = _load_schema_seed(
                            authored_local, layout.AUTHORED_SCHEMA_FILE
                        )
                        authored = True
                        log(f"Reusing the docset's authored schema: {len(schema_seed.tags)} tag(s)")
                    except InvalidArgument as exc:
                        log(f"[schema] authored-schema.json unusable ({exc}); ignoring")
                        schema_seed, parent_map_seed = None, {}
                if schema_seed is None and schema_json_local.exists():
                    try:
                        schema_seed = Schema.load(schema_json_local)
                        log(f"Reusing docset schema: {len(schema_seed.tags)} tag(s)")
                    except (json.JSONDecodeError, TypeError, ValueError, OSError):
                        schema_seed = None
                if schema_seed is None and roster_path.exists():
                    try:
                        roster_seed = _load_schema_roster(roster_path)
                        log(f"Reusing docset roster: {len(roster_seed)} concept(s)")
                    except InvalidArgument:
                        roster_seed = None

            # Closure keys on AUTHORSHIP, not on the mere presence of a seed.
            # A vocabulary a PERSON wrote is a specification: supplying one
            # means the output carries those tag names and no others, with no
            # flag to half-apply it. A vocabulary the PIPELINE derived from its
            # own previous output is not a specification — it is a hint for
            # consistency — so automatic reuse of schema.json /
            # concept_roster.json seeds exactly as it always has and keeps
            # coining. That distinction is what lets this feature be all-or-
            # nothing without changing what an ordinary incremental generate
            # does.
            seed_names = (
                list(schema_seed.tags) if schema_seed is not None else list(roster_seed or {})
            )
            # extend_schema keeps an AUTHORED vocabulary open: the user's names
            # are still authoritative and reused first, but labeling may coin for
            # a role they did not cover, and every coinage is reported back as a
            # candidate for the next revision. It is meaningless without an
            # authored schema, so say so rather than silently doing nothing.
            if extend_schema and not authored:
                raise InvalidArgument(
                    "Extending the schema needs a supplied schema to extend. Supply a "
                    "schema file, or run on a docset where an earlier run supplied one "
                    "and left an authored schema. (Without a supplied schema, labeling "
                    "already coins its own vocabulary.)"
                )
            vocab = TagVocab.build(
                seed_names,
                closed=authored and bool(seed_names) and not extend_schema,
                authored=authored,
            )
            if vocab.closed:
                log(
                    f"Vocabulary CLOSED at {len(vocab.names)} tag(s): the generated DGML uses "
                    "these tag names and no others. Unmatched content still renders "
                    "(as dg:chunk, text intact)."
                )
            elif vocab.extends:
                log(
                    f"Vocabulary EXTENDS {len(vocab.names)} authored tag(s): these are reused "
                    "wherever one fits; a role they do not cover may be coined, and every "
                    "coinage is reported under added_concepts."
                )
            elif seed_names:
                log(f"Seeded with {len(seed_names)} derived tag(s); labeling may coin more")

            # Reload already-generated docs from cache so the whole docset stays
            # consistent as its schema/roster grows; changed originals re-render
            # (no re-LLM). Replay resolves through the SAME vocabulary a fresh
            # run uses, or a re-rendered prior would diverge from its neighbours.
            labeled_priors = load_labeled_docs_from_cache(cache, list(prior_stems), vocab)
            for stem, blocks in labeled_priors.items():
                nm = prior_stems[stem]
                prior_docs[nm] = blocks
                prior_outputs[nm] = ws.blobs.get_blob(prior_out_paths[nm]).decode("utf-8")
                dgml_xml_keys[nm] = prior_out_paths[nm]

            # Materialize each file's page_text/ into a local dir the pipeline
            # (transcribe gate + coverage) reads. LocalStore yields the real dir
            # zero-copy; a remote store downloads it to a temp dir held open for
            # the whole batch. Populate the same dict `_on_output` closes over.
            with contextlib.ExitStack() as pt_stack:
                for nm, pref in page_text_prefixes.items():
                    page_text_dirs[nm] = pt_stack.enter_context(ws.blobs.materialize_dir(pref))
                # Materialize each file's source dir so transcription's path tools
                # (load_document_as_pdf → ghostscript) get the original + its
                # persisted sibling <stem>.pdf on disk. LocalStore yields the real
                # dir (zero-copy); a remote store downloads it for the batch.
                pdf_paths: list[Path | str] = [
                    pt_stack.enter_context(
                        ws.blobs.materialize_dir(layout.file_prefix(filename_to_fid[nm]))
                    )
                    / nm
                    for nm in convert_names
                ]
                options = ConvertOptions(
                    model=gen_model,
                    label_model=resolved_label_model,
                    api_key=gen_api_key,
                    api_base=gen_api_base,
                    window_size=window_size,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking=thinking or gen_cfg.thinking,
                    max_parallel_docs=max_parallel_docs,
                    cache_dir=cache,
                    debug=debug,
                    page_text_dirs=page_text_dirs,
                    workspace=ws,
                    dgml_header=build_header(ws.organization, ds.name),
                    converters=load_conversion_config(ws),
                    pdf_config=load_pdf_config(ws),
                    roster_seed=roster_seed,
                    schema_seed=schema_seed,
                    parent_map=parent_map_seed or None,
                    vocab=vocab,
                    progress=log,
                )
                convert_batch(
                    pdf_paths,
                    options=options,
                    on_output=_on_output,
                    on_error=_on_error,
                    on_label_error=_on_label_error,
                    on_off_schema=_on_off_schema,
                    prior_docs=prior_docs,
                    prior_outputs=prior_outputs,
                )
            # Documents were converted on a pool, so fold the per-document
            # results back in a fixed order — queued order for converted files,
            # docset order for re-rendered priors — and the report stays
            # identical to the serial run it replaced.
            written.extend(
                converted_by_name[name] for name in convert_names if name in converted_by_name
            )
            cov_results.extend(cov_by_name[name] for name in convert_names if name in cov_by_name)
            rerendered.extend(name for name in prior_docs if name in rerendered_by_name)
            # convert_batch silently drops documents whose transcription failed, so
            # `_on_output` never fires for them. Reconcile: any queued file with no
            # output is a per-file failure, not a vanished row (keeps counts summing
            # to `total`).
            produced = {entry.source for entry in written}
            for name, fid in filename_to_fid.items():
                if name not in produced:
                    message = gen_errors.get(
                        name, "the generation pipeline produced no output for this file"
                    )
                    failed_results.append(
                        GenerateFileResult(
                            status="failed",
                            file_id=fid,
                            source=name,
                            error={"code": GenerationFailed.code, "message": message},
                        )
                    )
            if cov_report_key is not None and cov_results:
                # Merge into any existing report so an incremental run keeps the
                # already-generated docs' coverage instead of overwriting it.
                existing_docs: list[dict[str, Any]] = []
                if ws.blobs.blob_exists(cov_report_key):
                    try:
                        existing_docs = json.loads(ws.blobs.get_blob(cov_report_key)).get(
                            "documents", []
                        )
                    except json.JSONDecodeError:
                        existing_docs = []
                merged = cov_mod.merge_coverage_documents(existing_docs, cov_results)
                ws.blobs.put_blob(
                    cov_report_key, cov_mod.dump_coverage_report(merged).encode("utf-8")
                )
            # Persist schema.json (labeling wrote it next to the cache) as the
            # docset's generation-schema blob — exact bytes, before write_docset_rnc
            # reads it back — then flush the cache working dir to the store.
            if cache_dir is None and schema_json_local.exists():
                ws.blobs.put_blob(schema_key, schema_json_local.read_bytes())
            # The authored vocabulary lands in a slot derive_schema never
            # touches. Without this, ground truth goes in and `seed union
            # everything coined` comes back out, and the NEXT run auto-seeds
            # from that polluted version — which is precisely why a seeded run
            # is not reproducible today. Stored in canonical Schema v1 form
            # whatever form it was authored in (tag list, {name: description},
            # RNC), so there is one shape to read back.
            if cache_dir is None and authored_seed is not None:
                authored_seed.save(authored_local)
                ws.blobs.put_blob(authored_key, authored_local.read_bytes())
                log(f"[schema] wrote {layout.AUTHORED_SCHEMA_FILE} (authored vocabulary)")
    else:
        log("Nothing to convert — every file is already converted, missing, or a duplicate name.")

    # Final step, after every file is converted, grounded and semlinked:
    # refresh the docset's full-schema.rnc (schema.json rendered as RELAX NG
    # Compact, with data types observed in the final XML). Best-effort — a
    # schema render failure must not fail the generate.
    try:
        rnc_key = write_docset_rnc(ws, docset_id)
        if rnc_key is not None:
            log(f"[schema] wrote {rnc_key.rsplit('/', 1)[-1]}")
    except Exception as exc:
        log(f"[schema] full-schema.rnc skipped ({exc})")

    return GenerateReport(
        docset=ds,
        total=len(file_ids),
        skipped=skipped_results,
        failed=failed_results,
        converted=written,
        rerendered=rerendered,
        output_key=output_key,
        # Report the coverage report key only if a report was actually written.
        coverage_report_key=cov_report_key if cov_results else None,
        model=gen_model,
        label_model=resolved_label_model,
        model_source=gen_model_source,
        schema_extended=extend_schema,
    )
