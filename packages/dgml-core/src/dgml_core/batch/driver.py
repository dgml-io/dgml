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

"""Drive many ``steps_*`` generators through a :class:`BatchExecutor` in waves.

Each :class:`Unit` is one independent piece of work (a document, a file, a
page) expressed as an :data:`dgml_core.llm.LLMSteps` generator — the same
generator the synchronous path drives one request at a time. The driver runs
them side by side: advance every live generator to its next request, hand the
whole set to the executor as one wave, deliver the responses, repeat until
every generator has returned. Requests inside one unit stay strictly ordered
(window N+1 is only built once window N's response is in); across units
there is no ordering at all.

A request the executor could not serve at all arrives as an ``Exception`` and
is delivered to its unit's generator with ``throw()`` at the pending ``yield``
(the :data:`dgml_core.llm.LLMSteps` contract), so a stage generator that
catches around ``yield from llm.steps_*(...)`` — retry once, short-circuit on
reachability, bisect — behaves exactly as its sync twin does around
``llm.call(...)``. Only an exception escaping ``throw()`` fails the unit.

Accounting mirrors :func:`dgml_core.llm.drive` exactly — one ``_record_call``
scope per unit for its whole lifetime, each response's usage folded in before
the generator sees it (summed per wrapper call, as ``drive`` does), the
generator closed on every exit — so a unit run here writes the same usage row
a sync run would, tagged with the batch tier. Inside an enclosing
``record_usage_for`` sink the per-call subtotals are folded in at the end of
the stage, in *units* order rather than finish order, so the sink's float sum
associates exactly as the sync path's document-by-document run does.

**A stage-wide failure keeps finished work.** When the executor cannot bring
a wave to an end (:class:`~dgml_core.errors.BatchExecutionFailed`, or any other
``Exception`` escaping ``run_wave``), the units that had already finished keep
their results; every unit still running gets that error as its outcome,
flagged ``stage_error``, and :func:`run_stage` returns normally. Only an
interrupt (a non-``Exception`` ``BaseException``, which includes
:class:`~dgml_core.errors.BatchPending`) closes every unit and propagates.

**Fan-out inside a unit.** A unit's generator may yield an
:class:`dgml_core.llm.FanOut` of child generators (see its contract). The
children's requests are extra in-flight requests of the same unit, joining the
same waves as every other unit's; the parent resumes, with the children's
results in child order, once every child has returned. Each child is a
*strand* with its own pending request and its own usage fold; a finished
child's subtotals are folded into its parent's, after the parent's own
preceding ones and in child order, so the unit's sums associate exactly as
the sync driver's one-child-after-another run does. Request ids stay
``u<index>_<stem>-<n>``, *n* counting the unit's requests in the
deterministic order its strands become ready. Every unit is admitted up
front.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from dgml_core import llm
from dgml_core.batch.executor import TIER_MARKER, BatchExecutor
from dgml_core.llm import SINK_TIER_KEY
from dgml_core.usage import TIER_BATCH, add_partial, extract_cost_and_tokens, with_tier

#: Every request id the driver emits matches this — the strictest provider
#: rule (Anthropic: ``^[a-zA-Z0-9_-]{1,64}$``), applied to every backend.
CUSTOM_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_NON_ID_CHARS = re.compile(r"[^A-Za-z0-9]+")
# ``u<index>_<stem>`` is capped here; ``-<n>`` (the step counter) follows, so
# the full id stays within 64 characters for any realistic step count.
_MAX_PREFIX_CHARS = 48


@dataclass(frozen=True)
class Unit:
    """One independent generator to drive, with the config its usage row uses.

    ``name`` keys the stage's outcomes, so it must be unique within a stage
    (:func:`run_stage` rejects duplicates). ``config`` is the object the
    generator itself holds: the driver records the unit's usage row from it
    at the end, so context the generator sets on it (``config.context``)
    reaches the row, as it does under :func:`dgml_core.llm.drive`.
    """

    name: str
    config: llm.LLMConfig
    steps: llm.LLMFlow[Any]


@dataclass
class UnitOutcome:
    """How one unit ended: its generator's return value, or the exception that
    stopped it, plus how many of its requests each path served.

    ``stage_error`` marks an ``error`` that is the whole stage's (the executor
    failed a wave while this unit was still running), not the unit's own: a
    caller that re-raises a unit's own error may prefer to soft-fail these."""

    result: Any = None
    error: BaseException | None = None
    stage_error: bool = False
    steps: int = 0
    batch_steps: int = 0
    sync_steps: int = 0
    failed_steps: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class _Strand:
    """One generator being driven inside a unit: the unit's own (the root) or a
    :class:`~dgml_core.llm.FanOut` child.

    A strand is, at any moment, waiting on one request (``pending_step``),
    suspended at a ``FanOut`` (``children``), or ``done``. Its usage fold sums
    its own responses per wrapper call and hands each finished subtotal on —
    to the unit's accounting for the root, to the strand's own ``subs`` for a
    child (folded into the parent's at the join, :func:`_join_usage`).
    """

    gen: llm.LLMFlow[Any]
    parent: _Strand | None
    index: int = 0
    fold: llm._UsageFold | None = None
    subs: list[dict[str, Any]] = field(default_factory=list)
    pending_id: str = ""
    pending_step: Any = None
    children: list[_Strand] | None = None
    done: bool = False
    result: Any = None
    error: Exception | None = None


@dataclass
class _Live:
    """Per-unit driver state while its generator is running."""

    unit: Unit
    id_prefix: str
    outcome: UnitOutcome
    totals: dict[str, Any] | None = None
    scope: Any = None  # the open ``_record_call`` context manager, once a request exists
    fold: llm._UsageFold | None = None
    # With an enclosing sink: the unit's per-call subtotals, held until the
    # stage ends and then folded into ``sink`` in units order.
    sink: dict[str, Any] | None = None
    deferred: list[dict[str, Any]] = field(default_factory=list)
    n: int = 0
    closed: bool = False
    root: _Strand | None = None
    #: Strands whose request is in the current wave, in wave order.
    in_flight: list[_Strand] = field(default_factory=list)
    #: Strands whose next request is ready, in the order they became ready
    #: (that order assigns their ids and their place in the wave).
    ready: list[_Strand] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.fold = llm._UsageFold(self._fold_sub)
        self.root = _Strand(gen=self.unit.steps, parent=None, fold=self.fold)

    def next_id(self) -> str:
        self.n += 1
        pending_id = f"{self.id_prefix}-{self.n}"
        if not CUSTOM_ID_PATTERN.match(pending_id):  # pragma: no cover - by construction
            raise AssertionError(f"driver produced an invalid request id {pending_id!r}")
        return pending_id

    def open_scope(self) -> None:
        """Open the unit's accounting scope — only once it has a request, so a
        unit that returns without asking for anything leaves no usage row."""
        if self.scope is None:
            self.scope = llm._record_call(self.unit.config)
            self.totals = self.scope.__enter__()
            self.sink = self.unit.config._usage_sink

    def _fold_sub(self, sub: dict[str, Any]) -> None:
        if self.sink is not None:
            self.deferred.append(sub)
        else:
            assert self.totals is not None
            add_partial(self.totals, sub)

    def fold_deferred(self) -> None:
        """Fold the held subtotals into the enclosing sink (once)."""
        if self.sink is not None:
            for sub in self.deferred:
                add_partial(self.sink, sub)
        self.deferred = []


def _child_strand(parent: _Strand, index: int, gen: llm.LLMFlow[Any]) -> _Strand:
    strand = _Strand(gen=gen, parent=parent, index=index)
    strand.fold = llm._UsageFold(strand.subs.append)
    return strand


def _join_usage(parent: _Strand) -> None:
    """Fold every child's subtotals into *parent*, in child order, after the
    parent's own pending one — the sync driver's sequence."""
    assert parent.fold is not None and parent.children is not None
    parent.fold.flush()
    for child in parent.children:
        assert child.fold is not None
        child.fold.flush()
        for sub in child.subs:
            parent.fold.emit(sub)
        child.subs = []


def _close_strand(strand: _Strand) -> None:
    """Close *strand* and everything below it, children first, folding what
    each had spent into its parent (in child order) on the way up."""
    if strand.children is not None:
        for child in strand.children:
            _close_strand(child)
        _join_usage(strand)
        strand.children = None
    try:
        strand.gen.close()
    finally:
        strand.done = True
        assert strand.fold is not None
        strand.fold.flush()


def _id_prefix(index: int, name: str) -> str:
    """``u<index>_<readable stem>``: the index guarantees uniqueness, the stem
    (ASCII alphanumerics of the unit name, ``_``-joined) keeps ids legible."""
    head = f"u{index}"
    stem = _NON_ID_CHARS.sub("_", name).strip("_")
    room = _MAX_PREFIX_CHARS - len(head) - 1
    stem = stem[:room].rstrip("_") if room > 0 else ""
    return f"{head}_{stem}" if stem else head


def _finish(state: _Live, exc: BaseException | None, *, own: bool = False) -> None:
    """Close one unit's generator and its accounting scope (once).

    *own* marks *exc* as the unit's own failure (raised by its generator), as
    opposed to a stage-wide one closing every unit."""
    if state.closed:
        return
    state.closed = True
    state.ready = []
    try:
        assert state.root is not None
        _close_strand(state.root)
    finally:
        state.outcome.error = exc
        if own and exc is not None and state.scope is None:
            # Failed before its first request: sync ``drive`` would already
            # have opened the scope, so an error row is still owed.
            state.open_scope()
        if state.scope is not None:
            if exc is None:
                state.scope.__exit__(None, None, None)
            else:
                # ``_record_call`` re-raises the same exception object inside
                # ``__exit__``; contextlib swallows that re-raise and returns
                # False, so nothing escapes here.
                state.scope.__exit__(type(exc), exc, exc.__traceback__)


def _served_by_batch(response: Any) -> bool:
    hidden = getattr(response, "_hidden_params", None)
    return isinstance(hidden, dict) and hidden.get(TIER_MARKER) == TIER_BATCH


def run_stage(
    units: Sequence[Unit],
    executor: BatchExecutor,
    *,
    log: Callable[[str], None] = lambda _m: None,
    stage: str | None = None,
) -> dict[str, UnitOutcome]:
    """Drive every unit to completion through *executor*, wave by wave.

    *stage* names the pipeline stage these waves serve ("transcribe",
    "extraction phase 3", ...). It only labels the executor's log lines, as
    ``[batch <stage> wave <n>]``, so a run's diagnostics say which stage each
    submitted batch belongs to. See :func:`_run_stage` for the semantics.
    """
    with executor.in_stage(stage):
        return _run_stage(units, executor, log=log)


def _run_stage(
    units: Sequence[Unit],
    executor: BatchExecutor,
    *,
    log: Callable[[str], None] = lambda _m: None,
) -> dict[str, UnitOutcome]:
    """Drive every unit to completion through *executor*, wave by wave.

    Returns ``{unit.name: outcome}`` in *units* order. A unit whose generator
    raises (a parse error, an empty response, …) is closed and reported in its
    outcome; the other units continue, and the failed unit's usage row still
    carries what it spent. An executor-level failure
    (:class:`dgml_core.errors.BatchExecutionFailed`, or any ``Exception`` out of
    ``run_wave``) becomes the outcome of every unit not yet finished (marked
    ``stage_error``; each writes its partial row) while finished units keep
    their results, and the outcomes are returned. An interrupt closes every
    open unit the same way and propagates.

    Usage rows: each unit's row is recorded from ``unit.config`` itself (so
    context its generator sets reaches the row) with ``tier="batch"`` — the
    driver sets the tier on each distinct config for the stage's duration and
    restores it afterwards. The tier is read per response from the executor's
    marker, so a unit served wholly by synchronous fallbacks records
    ``"standard"``, and one served by both tiers records one row per tier
    (:func:`dgml_core.usage.scope_events`). A unit that returns without making a request
    writes no row. Inside an enclosing :func:`dgml_core.llm.record_usage_for`
    scope the rows fold into that scope as usual, and the scope's own row is
    marked batch-tier too.
    """
    names = [unit.name for unit in units]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"unit names must be unique within a stage; duplicated: {dupes}")

    outcomes: dict[str, UnitOutcome] = {}
    live: list[_Live] = []
    for index, unit in enumerate(units):
        outcome = UnitOutcome()
        outcomes[unit.name] = outcome
        live.append(_Live(unit=unit, id_prefix=_id_prefix(index, unit.name), outcome=outcome))

    # The tier goes on the configs the generators hold (not a copy), for the
    # stage's lifetime; restored in the finally. An enclosing usage scope is
    # marked batch through its sink (record_usage_for reads it on exit).
    saved_tiers: dict[int, tuple[llm.LLMConfig, str]] = {}
    for unit in units:
        cfg = unit.config
        if id(cfg) not in saved_tiers:
            saved_tiers[id(cfg)] = (cfg, cfg.tier)
            cfg.tier = TIER_BATCH
            if cfg._usage_sink is not None:
                cfg._usage_sink[SINK_TIER_KEY] = TIER_BATCH

    def resume(state: _Live, strand: _Strand, response: Any | None, *, fresh: bool = False) -> None:
        """Move one strand on: to its next request (queued on ``state.ready``),
        into a ``FanOut`` (its children primed in child order), or to its end
        — which, for the last child of a ``FanOut`` to finish, resumes the
        parent with the joined results, and for the root finishes the unit.

        *fresh* primes the generator; otherwise *response* is an
        ``Exception`` to throw in at the pending yield, or the value to send
        (a response, or a joined ``FanOut``'s result list)."""
        while True:
            gen = strand.gen
            try:
                if fresh:
                    step = next(gen)
                elif isinstance(response, Exception):
                    step = gen.throw(response)
                else:
                    step = gen.send(response)
            except StopIteration as done:
                strand.result = done.value
                error: Exception | None = None
            except Exception as exc:  # one unit's failure never sinks the stage
                where = "before its first request" if state.n == 0 else f"after step {state.n}"
                who = (
                    state.unit.name
                    if strand.parent is None
                    else (f"{state.unit.name} (fan-out child {strand.index})")
                )
                log(f"[batch] {who}: failed {where}: {type(exc).__name__}: {exc}")
                error = exc
            else:
                fresh = False
                if isinstance(step, llm.FanOut):
                    assert strand.fold is not None
                    strand.fold.flush()
                    strand.children = [
                        _child_strand(strand, i, child) for i, child in enumerate(step.children)
                    ]
                    if not strand.children:
                        strand.children = None
                        response = []  # an empty FanOut joins at once
                        continue
                    for child in list(strand.children):
                        resume(state, child, None, fresh=True)
                        if state.closed:
                            return
                    return
                state.open_scope()
                strand.pending_step = step
                state.ready.append(strand)
                return
            # The strand ended (returned, or raised ``error``).
            strand.done = True
            strand.error = error
            parent = strand.parent
            if parent is None:
                if error is None:
                    state.outcome.result = strand.result
                    _finish(state, None)
                else:
                    _finish(state, error, own=True)
                return
            try:
                strand.gen.close()
            finally:
                assert strand.fold is not None
                strand.fold.flush()
            assert parent.children is not None
            if not all(child.done for child in parent.children):
                return
            # The join: every child has returned. Fold their usage in child
            # order, then resume the parent with the results (or the first
            # child failure, in child order).
            _join_usage(parent)
            children, parent.children = parent.children, None
            failure = next((c.error for c in children if c.error is not None), None)
            strand, fresh = parent, False
            response = failure if failure is not None else [c.result for c in children]

    def queue(wave: dict[str, dict[str, Any]], state: _Live) -> None:
        """Give each of *state*'s ready strands its id and its place in *wave*."""
        ready, state.ready = state.ready, []
        for strand in ready:
            strand.pending_id = state.next_id()
            wave[strand.pending_id] = strand.pending_step
            state.in_flight.append(strand)

    try:
        wave: dict[str, dict[str, Any]] = {}
        active: list[_Live] = []
        for state in live:  # primed in units order; each joins the first wave
            assert state.root is not None
            resume(state, state.root, None, fresh=True)
            if not state.closed:
                queue(wave, state)
                active.append(state)

        while active:
            responses = executor.run_wave(wave)
            wave = {}
            still_active: list[_Live] = []
            for state in active:
                in_flight, state.in_flight = state.in_flight, []
                for strand in in_flight:
                    if state.closed:
                        break
                    response = responses[strand.pending_id]
                    state.outcome.steps += 1
                    if isinstance(response, Exception):
                        # The synchronous fallback itself failed: no usage to
                        # fold in; the generator decides what happens.
                        state.outcome.sync_steps += 1
                        state.outcome.failed_steps += 1
                    else:
                        assert strand.fold is not None
                        strand.fold.add(
                            strand.pending_step,
                            with_tier(
                                extract_cost_and_tokens(response),
                                response,
                                state.unit.config.tier,
                            ),
                        )
                        if _served_by_batch(response):
                            state.outcome.batch_steps += 1
                        else:
                            state.outcome.sync_steps += 1
                    resume(state, strand, response)
                if not state.closed:
                    queue(wave, state)
                    still_active.append(state)
            active = still_active
    except Exception as exc:
        # The stage failed, not any one unit: finished units keep their
        # results; the rest get the error (and write their partial rows).
        unfinished = [state for state in live if not state.closed]
        log(
            f"[batch] stage failed with {len(unfinished)} unit(s) unfinished "
            f"({len(live) - len(unfinished)} finished): {type(exc).__name__}: {exc}"
        )
        for state in unfinished:
            state.outcome.stage_error = True
            _finish(state, exc)
    except BaseException as exc:
        for state in live:
            _finish(state, exc)
        raise
    finally:
        # Every unit is closed by now (finished, failed, or closed by the
        # handlers above), so each one's subtotals are final.
        for state in live:
            state.fold_deferred()
        for cfg, tier in saved_tiers.values():
            cfg.tier = tier
    return outcomes


def run_stage_sync(
    units: Sequence[Unit],
    execute: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, UnitOutcome]:
    """The non-batch twin of :func:`run_stage`: drive each unit in order
    through :func:`dgml_core.llm.drive`, returning the same outcome shape.

    Callers that build :class:`Unit` lists once can route them to either
    driver and read results identically; the usage rows are the ones the
    inline wrappers would have written (standard tier). Only ``Exception``
    is captured per unit — an interrupt propagates, as it does from
    ``llm.drive``.
    """
    names = [unit.name for unit in units]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"unit names must be unique within a stage; duplicated: {dupes}")

    outcomes: dict[str, UnitOutcome] = {}
    for unit in units:
        outcome = UnitOutcome()
        outcomes[unit.name] = outcome
        counted = _CountingExecute(execute)
        try:
            outcome.result = llm.drive(unit.steps, unit.config, counted)
        except Exception as exc:
            outcome.error = exc
        outcome.steps = counted.calls
        outcome.sync_steps = counted.calls
        outcome.failed_steps = counted.failures
    return outcomes


class _CountingExecute:
    """Wrap an executor callable to count the requests a unit made."""

    def __init__(self, execute: Callable[[dict[str, Any]], Any] | None) -> None:
        self._execute = execute
        self.calls = 0
        self.failures = 0

    def __call__(self, step: dict[str, Any]) -> Any:
        self.calls += 1
        run = self._execute if self._execute is not None else llm._completion_with_retry
        try:
            return run(step)
        except Exception:
            self.failures += 1
            raise
