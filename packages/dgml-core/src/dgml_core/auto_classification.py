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

"""Auto-classification of freshly added files: classify, assign, auto-extract.

What ``dgml file add --auto-classify`` does after each file is ingested, as
library calls. Each produces the ``classification`` block the CLI embeds per
file (``performed``, ``model``, ``decision``, ``docset_id``, …, and an
``extraction`` block when the file lands in a DocSet with a schema):

- :func:`auto_classify` — one file, synchronously (the LLM picks an existing
  DocSet or proposes a new one; assignment then auto-extracts).
- :func:`prepare_bulk_classify` + :func:`classify_bulk_batch` — a whole
  directory through the provider's batch API: every file's classification in
  one wave (assign-only mode) or one wave per file, in order (the default
  mode, where a file may create a DocSet the next one must see), then the
  auto-extraction of every assigned file — across all DocSets — as one batch
  run. Each block is exactly what :func:`auto_classify` would have written for
  that file in the same mode.

Failures after the classification config is in hand are soft: the file is
already added, so they land in the block's ``error`` and the caller carries
on. Only a missing/invalid config (loaded up front) and
:class:`~dgml_core.errors.NoExistingDocSets` are hard.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dgml_core.classification import (
    ClassificationConfig,
    ClassificationDecision,
    classify_file,
    load_classification_config,
)
from dgml_core.docsets import DocSetStore
from dgml_core.errors import DgmlError, NoExistingDocSets
from dgml_core.files import AddFileResult
from dgml_core.models import DocSet
from dgml_core.storage import Workspace

if TYPE_CHECKING:
    from dgml_core.batch import BatchExecutor
    from dgml_core.grounded import GroundedConfig


def _no_log(_message: str) -> None:
    return None


def soft_error(exc: BaseException | None) -> str:
    """A per-file soft-fail message, formatted as the synchronous path formats
    it: ``"<CODE>: <msg>"`` for a DgmlError, ``"<Type>: <msg>"`` otherwise."""
    if isinstance(exc, DgmlError):
        return f"{exc.code}: {exc}"
    return f"{type(exc).__name__}: {exc}"


def classification_block(config: ClassificationConfig) -> dict[str, Any]:
    """A fresh ``classification`` block for a file whose classification ran."""
    return {
        "performed": True,
        "model": config.model,
        "decision": None,
        "docset_id": None,
        "docset_created": False,
        "docset_name": None,
        "docset_key_questions": [],
        "error": None,
    }


def apply_classification(
    ws: Workspace,
    file_id: str,
    block: dict[str, Any],
    decision: ClassificationDecision,
    docsets: list[DocSet] | None,
    assign: Callable[[str], dict[str, Any] | None],
    create: Callable[[ClassificationDecision], DocSet] | None = None,
) -> None:
    """Act on a file's classification *decision*, filling *block* in place.

    *assign* assigns the file to an existing DocSet and returns its extraction
    block (``None`` when the DocSet has no schema). The synchronous path
    extracts inside it; the batch path only assigns, then fills
    ``block["extraction"]`` once the directory's extraction wave is back — the
    key lands last either way, so the block has the same shape. A DocSet
    created for a ``new`` decision is appended to *docsets* when given.

    *create* makes the DocSet a ``new`` decision names (default: create it in
    the store). A resumed batch job passes one that returns the DocSet its
    first run already created for this file instead of creating it twice.
    """
    docset_store = DocSetStore(ws)
    try:
        if decision.decision == "existing":
            assert decision.existing_docset_id is not None
            extraction_block = assign(decision.existing_docset_id)
            existing = docset_store.get(decision.existing_docset_id)
            block.update(
                decision="existing",
                docset_id=existing.id,
                docset_name=existing.name,
                docset_key_questions=list(existing.key_questions),
            )
            if extraction_block is not None:
                block["extraction"] = extraction_block
        elif decision.decision == "new":
            assert decision.new_name is not None and decision.new_description is not None
            if create is not None:
                created = create(decision)
            else:
                created = docset_store.create(
                    name=decision.new_name,
                    description=decision.new_description,
                    key_questions=list(decision.new_key_questions),
                )
            docset_store.add_file(created.id, file_id)
            if docsets is not None:
                docsets.append(created)
            block.update(
                decision="new",
                docset_id=created.id,
                docset_created=True,
                docset_name=created.name,
                docset_key_questions=list(created.key_questions),
            )
        else:  # unreachable — classify_file returns only these three
            raise AssertionError(f"unhandled classification decision: {decision.decision}")
    except DgmlError as exc:
        block["error"] = f"{exc.code}: {exc}"


def auto_classify(
    ws: Workspace,
    result: AddFileResult,
    *,
    config: ClassificationConfig | None = None,
    docsets: list[DocSet] | None = None,
    allow_new: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    """Run LLM auto-classification on a freshly added File and assign it.

    Returns the ``classification`` block embedded in ``dgml file add`` output.

    ``allow_new=False`` (``--auto-classify existing``) forbids creating a
    DocSet and always assigns: the LLM is given only the assign tool and must
    return the best-fitting DocSet even when the fit is poor. With no DocSets
    to choose from the mode has no possible outcome, so it is a **hard** error
    (``NO_EXISTING_DOCSETS``) — a precondition on the request rather than a
    failure of the classification call. Callers check it before adding any
    file; the re-raise below keeps it hard if one ever doesn't.

    A missing or invalid classification config is a **hard** failure: when
    ``config`` is not supplied it is loaded here via
    :func:`load_classification_config`, whose error propagates. Bulk callers
    load the config once up front and pass it in, so the run aborts before any
    file is processed when it's missing. Failures *after* config is in hand —
    the LLM/classify call, auth — stay soft: the File record is already
    stored, so they land in the block's ``error``.

    ``docsets``, when supplied, is a mutable list the caller maintains across
    a bulk run: it is forwarded to :func:`classify_file` so the LLM sees
    DocSets created earlier in the same run, and any freshly-created DocSet
    is appended to it here so later files can be assigned to it.

    Skipped (and reported as ``performed: false``) when the add returned an
    existing record rather than creating a new one — re-runs stay idempotent,
    and neither config nor an LLM call is spent on a duplicate.
    """
    if not result.created:
        return {
            "performed": False,
            "reason": "file already existed; classification skipped",
        }

    if config is None:
        config = load_classification_config(ws)

    file_id = result.record.id
    block = classification_block(config)

    try:
        decision = classify_file(
            ws, file_id, config=config, docsets=docsets, allow_new=allow_new, debug=debug
        )
    except NoExistingDocSets:
        # A precondition on the request, not a failure of the call — callers
        # check it before ingesting anything. Kept hard even if one didn't:
        # soft-failing would leave the unassigned file this mode prevents.
        raise
    except DgmlError as exc:
        block["error"] = f"{exc.code}: {exc}"
        return block

    def _assign_and_extract(docset_id: str) -> dict[str, Any] | None:
        # Assign, and auto-extract when the target DocSet has an extraction
        # schema set (soft-fail — the extraction block carries any error; the
        # assignment itself stands).
        from dgml_core.extraction import add_file_and_extract

        return add_file_and_extract(ws, docset_id, file_id, write_stats=debug, debug=debug)

    apply_classification(ws, file_id, block, decision, docsets, _assign_and_extract)
    return block


# ---- a directory through the batch API ---------------------------------------


@dataclass
class BulkClassifyBatch:
    """State for classifying a directory's files through the batch API,
    settled before any file is added: the executors (credentials already
    resolved) or why one could not be built, and the files awaiting
    classification (append ``(classification entry, add result)`` pairs to
    ``pending`` as files are added; :func:`classify_bulk_batch` fills each
    entry's ``classification`` key)."""

    pending: list[tuple[dict[str, Any], AddFileResult]] = field(default_factory=list)
    classifier: BatchExecutor | None = None
    classifier_error: DgmlError | None = None
    grounded: GroundedConfig | None = None
    grounded_error: str | None = None
    extractor: BatchExecutor | None = None
    extractor_error: DgmlError | None = None


def prepare_bulk_classify(
    ws: Workspace,
    *,
    config: ClassificationConfig,
    docsets: list[DocSet],
    poll_interval_s: float = 30.0,
    log: Callable[[str], None] = _no_log,
) -> BulkClassifyBatch:
    """Check the batch backends and build the executors for a directory
    classified through the batch API — all before any file is added.

    Either mode batches (see :func:`classify_bulk_batch` for how). A DocSet
    the default mode creates during the run has no extraction schema, so the
    DocSets that exist now decide whether auto-extraction can run at all.
    Every model that will run is checked for a batch backend up front — the
    classification model always, the values model when a DocSet that files
    can be assigned to already has an extraction schema — and a model with
    none is a hard :class:`~dgml_core.errors.BatchUnavailable`.

    Credentials are resolved here too, by building each executor. A failure
    (an unset ``api_key_env``) is *not* fatal: the synchronous run soft-fails
    each file on it, so it is kept and each file later gets exactly the
    classification/extraction error the synchronous run would record.

    A batch job session, when used, must already be active: the executors
    are bound to it.
    """
    from dgml_core.batch import StageRequest, assert_batchable
    from dgml_core.classification import classification_batch_executor
    from dgml_core.errors import BatchUnavailable

    state = BulkClassifyBatch()
    models: dict[str, StageRequest] = {"classification": config.model}
    doc_store = DocSetStore(ws)
    if any(doc_store.has_schema(ds.id) for ds in docsets):
        from dgml_core.grounded import load_grounded_config, values_batch_stages

        try:
            state.grounded = load_grounded_config(ws)
            models.update(
                values_batch_stages(
                    state.grounded.values_model, state.grounded.values_reasoning_effort
                )
            )
        except DgmlError as exc:
            # Soft, exactly like the synchronous auto-extract: every extraction
            # block will carry this error; classification still runs.
            state.grounded_error = f"{exc.code}: {exc}"
    assert_batchable(models)  # BATCH_UNAVAILABLE, before anything is added

    knobs: dict[str, Any] = {"poll_interval_s": poll_interval_s, "log": log}
    try:
        state.classifier = classification_batch_executor(config, **knobs)
    except BatchUnavailable:
        raise
    except DgmlError as exc:
        state.classifier_error = exc
    if state.grounded is not None:
        from dgml_core.grounded import values_batch_executor

        try:
            state.extractor = values_batch_executor(state.grounded, **knobs)
        except BatchUnavailable:
            raise
        except DgmlError as exc:
            state.extractor_error = exc
    return state


#: Job state key: file id → the DocSet a ``new`` decision created for it, so a
#: resumed job re-uses that DocSet (and its id, which later files' requests
#: name) instead of creating a second one when it replays the decision.
CREATED_DOCSETS_STATE = "classify_created_docsets"


def classify_bulk_batch(
    ws: Workspace,
    batch: BulkClassifyBatch,
    *,
    config: ClassificationConfig,
    docsets: list[DocSet],
    debug: bool = False,
    allow_new: bool = False,
) -> dict[str, Any]:
    """Classify every pending file through the batch API, assign each, then
    run the auto-extraction of every assigned file — across all DocSets — as
    one batch run (two waves).

    Assign-only mode (``allow_new=False``) classifies every file in ONE wave:
    no file can add a DocSet, so each request depends only on *docsets*. The
    default mode (``allow_new=True``) classifies the files IN ORDER, one wave
    per file that makes a request: any reply may create a DocSet, and every
    later file's request lists it (by id, in its prompt and its tool schema),
    exactly as the synchronous run's file-by-file loop does. Each file is
    classified against the DocSet list the files before it left, and its
    DocSet is created before the next file's request is built.

    Each entry's ``classification`` block is the one the synchronous run would
    have written: same keys, same order, same soft-fail messages. A failure
    building or running a wave never aborts: it lands in each affected file's
    block, and every file is still reported. Returns the ``batch`` block (one
    entry per stage that ran: ``classification``, ``extraction``).
    """
    from dgml_core.classification import classify_files_batch
    from dgml_core.errors import AuthError

    created = [(entry, res) for entry, res in batch.pending if res.created]
    for entry, res in batch.pending:
        if not res.created:
            entry["classification"] = auto_classify(ws, res, config=config, docsets=docsets)
    out: dict[str, Any] = {}
    if not created:
        return out

    # Assign-only mode with a single DocSet makes no request at all.
    if batch.classifier is None and (allow_new or len(docsets) > 1):
        err = batch.classifier_error
        if isinstance(err, AuthError):
            # The synchronous run fails every file on this before any request
            # (after its own page check), so running it per file is exact and
            # makes no LLM call.
            for entry, res in created:
                entry["classification"] = auto_classify(
                    ws, res, config=config, docsets=docsets, allow_new=allow_new, debug=debug
                )
            return out
        for entry, _res in created:
            block = classification_block(config)
            block["error"] = soft_error(err)
            entry["classification"] = block
        return out

    # (docset id, file id) → that file's classification block, to extract.
    to_extract: dict[tuple[str, str], dict[str, Any]] = {}

    def assigner(file_id: str, block: dict[str, Any]) -> Callable[[str], None]:
        def assign_only(docset_id: str) -> None:
            store = DocSetStore(ws)
            store.add_file(docset_id, file_id)
            if store.has_schema(docset_id):
                to_extract[(docset_id, file_id)] = block

        return assign_only

    if allow_new:
        assert batch.classifier is not None
        _classify_in_order(ws, batch.classifier, created, config, docsets, assigner, debug)
        stats = batch.classifier.stats
        out["classification"] = {"provider": batch.classifier.backend.provider, **stats.to_json()}
    else:
        fids = [res.record.id for _, res in created]
        decisions: dict[str, Any]
        try:
            # With a single DocSet no request is made, so no executor is needed.
            decisions = dict(
                classify_files_batch(
                    ws,
                    fids,
                    config=config,
                    docsets=docsets,
                    executor=batch.classifier,  # type: ignore[arg-type]
                    debug=debug,
                )
            )
        except NoExistingDocSets:
            raise
        except Exception as exc:
            decisions = dict.fromkeys(fids, exc)
        if batch.classifier is not None:
            stats = batch.classifier.stats
            out["classification"] = {
                "provider": batch.classifier.backend.provider,
                **stats.to_json(),
            }
        for entry, res in created:
            file_id = res.record.id
            block = classification_block(config)
            entry["classification"] = block
            decision = decisions[file_id]
            if isinstance(decision, NoExistingDocSets):
                raise decision
            if isinstance(decision, Exception):
                block["error"] = soft_error(decision)
                continue
            apply_classification(
                ws, file_id, block, decision, docsets, _returning_none(assigner(file_id, block))
            )

    if to_extract:
        stats_block = _extract_assigned_batch(ws, to_extract, batch=batch, debug=debug)
        if stats_block is not None:
            out["extraction"] = stats_block
    return out


def _returning_none(assign: Callable[[str], None]) -> Callable[[str], dict[str, Any] | None]:
    """*assign* as :func:`apply_classification`'s ``assign``: it reports no
    extraction block, because the batch path extracts after every file is in."""

    def run(docset_id: str) -> dict[str, Any] | None:
        assign(docset_id)
        return None

    return run


def _classify_in_order(
    ws: Workspace,
    executor: BatchExecutor,
    created: list[tuple[dict[str, Any], AddFileResult]],
    config: ClassificationConfig,
    docsets: list[DocSet],
    assigner: Callable[[str, dict[str, Any]], Callable[[str], None]],
    debug: bool,
) -> None:
    """The default mode's classification: file by file, one wave each, every
    decision applied (its DocSet created and appended to *docsets*) before the
    next file's request is built — the synchronous loop's order exactly.

    Under a batch job, each DocSet a ``new`` decision creates is recorded in
    the job's state (:data:`CREATED_DOCSETS_STATE`) the moment it exists, so a
    resumed run — which replays every earlier file's reply from the job —
    re-uses it rather than creating it again. Later files' requests name it by
    id, so a second DocSet would change them and the job would pay for them
    twice."""
    from dgml_core.batch.jobs import active_session
    from dgml_core.classification import classify_file_batch
    from dgml_core.errors import DocSetNotFound

    session = active_session()
    made: dict[str, str] = (
        session.state.setdefault(CREATED_DOCSETS_STATE, {}) if session is not None else {}
    )

    def creator(file_id: str) -> Callable[[ClassificationDecision], DocSet]:
        def create(decision: ClassificationDecision) -> DocSet:
            store = DocSetStore(ws)
            prior = made.get(file_id)
            if prior is not None:
                try:
                    return store.get(prior)
                except DocSetNotFound:
                    pass  # deleted since that run: create it afresh
            assert decision.new_name is not None and decision.new_description is not None
            docset = store.create(
                name=decision.new_name,
                description=decision.new_description,
                key_questions=list(decision.new_key_questions),
            )
            if session is not None:
                made[file_id] = docset.id
                session.persist()
            return docset

        return create

    for entry, res in created:
        file_id = res.record.id
        block = classification_block(config)
        entry["classification"] = block
        try:
            decision = classify_file_batch(
                ws,
                file_id,
                config=config,
                docsets=docsets,
                executor=executor,
                allow_new=True,
                debug=debug,
            )
        except Exception as exc:
            block["error"] = soft_error(exc)
            continue
        apply_classification(
            ws,
            file_id,
            block,
            decision,
            docsets,
            _returning_none(assigner(file_id, block)),
            creator(file_id),
        )


def _extract_assigned_batch(
    ws: Workspace,
    items: dict[tuple[str, str], dict[str, Any]],
    *,
    batch: BulkClassifyBatch,
    debug: bool,
) -> dict[str, Any] | None:
    """Auto-extract every ``(docset id, file id)`` in *items* as ONE batch run —
    one phase-1 wave and one phase-3 wave however many DocSets they span —
    writing each file's ``extraction`` block exactly as the synchronous
    auto-extract would. Returns the run's stats, or ``None`` when no batch ran
    (the grounded config or its credentials failed; each block says why)."""
    from dgml_core.errors import AuthError
    from dgml_core.extraction import auto_extract
    from dgml_core.grounded import extract_values_batch_pairs

    if batch.grounded is None:
        for block in items.values():
            block["extraction"] = {
                "performed": True,
                "model": None,
                "tool_calls": None,
                "error": batch.grounded_error,
            }
        return None
    grounded = batch.grounded
    if batch.extractor is None:
        for (docset_id, fid), block in items.items():
            if isinstance(batch.extractor_error, AuthError):
                # Every file fails on this at setup, before any request, so the
                # synchronous auto-extract is exact and makes no LLM call.
                block["extraction"] = auto_extract(
                    ws, docset_id, fid, config=grounded, write_stats=debug, debug=debug
                )
            else:
                block["extraction"] = {
                    "performed": True,
                    "model": grounded.values_model,
                    "tool_calls": None,
                    "error": soft_error(batch.extractor_error),
                }
        return None

    executor = batch.extractor
    pairs = list(items)
    results: dict[tuple[str, str], Any]
    try:
        results = dict(
            extract_values_batch_pairs(
                ws, pairs, config=grounded, executor=executor, write_stats=debug, debug=debug
            )
        )
    except Exception as exc:
        results = dict.fromkeys(pairs, exc)
    for pair, block in items.items():
        extraction: dict[str, Any] = {
            "performed": True,
            "model": grounded.values_model,
            "tool_calls": None,
            "error": None,
        }
        outcome = results[pair]
        if isinstance(outcome, Exception):
            extraction["error"] = soft_error(outcome)
        else:
            extraction["tool_calls"] = outcome.tool_calls
        block["extraction"] = extraction
    return {
        "provider": executor.backend.provider,
        "docset_ids": sorted({docset_id for docset_id, _ in pairs}),
        **executor.stats.to_json(),
    }
