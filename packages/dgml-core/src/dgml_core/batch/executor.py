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

"""Run one wave of completion requests through a provider's batch endpoint.

A *wave* is a set of independent requests that are all ready at the same
moment — window N of every document, say. :class:`BatchExecutor.run_wave`
takes the wave as ``{custom_id: kwargs}``, splits it into batches the backend
accepts, submits, polls until every batch has ended, and returns
``{custom_id: ModelResponse | Exception}`` — exactly one outcome per request.

Serving every request is the executor's job, not the caller's. A request the
provider could not serve is retried where the provider said "not now"
(``expired``, and other ``retryable`` errors, resubmitted up to
``max_item_retries`` times in a follow-up mini-wave) and otherwise executed
synchronously through the ordinary completion path, at full price; only when
that synchronous call itself fails is the request's outcome the exception it
raised, which the driver delivers to the unit's generator with ``throw()``. A caller
that would rather pay full price than wait a batch round-trip for a tiny wave can
set ``min_wave_size``: waves smaller than it run synchronously. The default (1)
batches every wave, including a lone request, because batch mode is an offline
cost mode — the tail windows of a long document and a one-document link stage
are exactly the waves a latency-first default would have billed at full price.
Each response records which path served it in ``_hidden_params["dgml_tier"]``;
pricing is left to the backend.

**Batch-level rejection bisects.** A batch the provider refuses as a whole —
``submit`` raising :class:`~dgml_core.batch.types.BatchRejected`, or every item
of a collected batch reported ``batch_rejected`` — says nothing about any one
request (too large, over a queue or enqueued-token limit). Its requests are
split in half and each half resubmitted, recursively, down to single
requests; only a single request still rejected at batch level runs
synchronously. Each split counts in ``WaveStats.bisections``.

**An uncertain create is never repeated.** ``submit`` raising
:class:`~dgml_core.batch.types.BatchSubmitUncertain` (a timeout or server
error after the request was sent) means the batch may exist and be billing:
it is neither resubmitted nor run synchronously. The wave fails with
:class:`~dgml_core.errors.BatchExecutionFailed` saying so, after canceling the
wave's other open batches.

**A rate-limited or over-quota create fails the wave.** ``submit`` raising
:class:`~dgml_core.batch.types.BatchThrottled` (still 429 after every retry,
or a quota / billing refusal such as ``insufficient_quota``) is neither
bisected — a smaller batch meets the same limit — nor run synchronously — the
same account pays full price, if it is served at all. The wave fails with
:class:`~dgml_core.errors.BatchExecutionFailed` saying so (retry later, or
check billing and quota), after canceling the wave's other open batches.

**Backend hygiene hooks** (optional on a backend, looked up with ``getattr``):
``release(custom_ids)`` is called after every submit attempt, accepted or not,
so a backend can drop cached encodings; ``cleanup(job)`` after a batch has
been fully collected, best-effort (a failure is logged, never raised) — never
for a batch left open. ``max_wait_s`` (how long the provider may legitimately
keep a batch running) sets the default polling deadline.

**A run-level deadline** (:class:`~dgml_core.batch.deadline.BatchDeadline`,
optional) bounds the wait further: once it has passed, every open batch of the
wave is canceled, given the backend's ``cancel_settle_s`` (a per-provider,
measured bound) to settle, and collected — a request the provider already
answered keeps its batch result — and everything else, in this wave and every
later one, runs synchronously. Nothing new is submitted after it. A cancel
still unsettled after the wait may keep processing, so its requests count as
``possibly_double_billed`` (a WARNING names the batch); without a job nothing
can reconcile it later. When both apply, the earlier of the deadline and
``max_poll_s`` wins.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from dgml_core.batch.backend import BatchBackend
from dgml_core.batch.chunking import plan_batches, request_size
from dgml_core.batch.deadline import CANCEL_SETTLE_POLL_S, BatchDeadline, cancel_settle_s
from dgml_core.batch.types import (
    BatchItemError,
    BatchJob,
    BatchRejected,
    BatchRequest,
    BatchSubmitUncertain,
    BatchThrottled,
    ItemErrorKind,
)
from dgml_core.errors import BatchExecutionFailed, BatchUnavailable
from dgml_core.usage import TIER_BATCH, TIER_STANDARD

# Marker key on ``response._hidden_params`` naming the path that produced the
# response (``TIER_BATCH`` or ``TIER_STANDARD``): stamped here, defined in
# :mod:`dgml_core.usage`, which splits usage rows by it.
from dgml_core.usage import TIER_MARKER as TIER_MARKER

logger = logging.getLogger(__name__)

#: How many synchronous fallbacks of one wave run at once by default — the
#: synchronous pipeline's ``--max-parallel-calls`` default.
DEFAULT_SYNC_WORKERS = 4

#: How :meth:`BatchExecutor._deadline_error` messages begin (what marks an
#: item failure as the deadline's doing).
_DEADLINE_MESSAGE = "batch deadline passed: "

#: Polling deadline when neither the caller nor the backend (``max_wait_s``)
#: says otherwise: the classic 24-hour batch completion window.
DEFAULT_MAX_POLL_S = 24 * 3600.0
#: Added to a backend's ``max_wait_s`` for the default polling deadline, so a
#: batch the provider is still legitimately running is never canceled early —
#: the provider expires it first, and its items come back ``expired``.
POLL_MARGIN_S = 3600.0


def default_max_poll_s(backend: object) -> float:
    """The polling deadline for *backend* when the caller set none: its
    ``max_wait_s`` plus :data:`POLL_MARGIN_S`, else :data:`DEFAULT_MAX_POLL_S`."""
    wait = getattr(backend, "max_wait_s", None)
    if isinstance(wait, int | float) and not isinstance(wait, bool) and wait > 0:
        return float(wait) + POLL_MARGIN_S
    return DEFAULT_MAX_POLL_S


@dataclass
class WaveStats:
    """Running totals over every :meth:`BatchExecutor.run_wave` call."""

    waves: int = 0
    batches: int = 0
    requests: int = 0
    batch_ok: int = 0
    sync_fallbacks: int = 0
    resubmitted: int = 0
    failed: int = 0
    batch_ids: list[str] = field(default_factory=list)
    #: Requests served from a batch job's store instead of the provider (a
    #: resumed ``--job`` run replaying what an earlier run received).
    replayed: int = 0
    #: Batches split in half after a batch-level rejection (see the module
    #: docstring); reported only when non-zero, so other payloads keep shape.
    bisections: int = 0
    #: What the responses this executor received cost (see :meth:`add_cost`).
    cost_usd: float = 0.0
    standard_cost_usd: float = 0.0
    unpriced: int = 0

    def add_cost(self, response: Any, tier: str) -> None:
        """Count one response received from the provider — a batch result
        (*tier* batch) or a synchronous fallback (standard). ``cost_usd`` is
        what it was billed at; ``standard_cost_usd`` what it would have cost
        synchronously (a batch price is half the standard one). A response
        without a price makes the totals unknown (``null``)."""
        hidden = getattr(response, "_hidden_params", None)
        cost = hidden.get("response_cost") if isinstance(hidden, dict) else None
        if not isinstance(cost, int | float) or isinstance(cost, bool):
            self.unpriced += 1
            return
        self.cost_usd += float(cost)
        self.standard_cost_usd += float(cost) * (2 if tier == TIER_BATCH else 1)

    def cost_json(self) -> dict[str, float | None]:
        return cost_fields(self.cost_usd, self.standard_cost_usd, self.unpriced)

    def run_json(self) -> dict[str, Any]:
        """This object's own counters (what a job records per run)."""
        out = self.to_json()
        out["unpriced"] = self.unpriced
        return out

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "waves": self.waves,
            "batches": self.batches,
            "requests": self.requests,
            "batch_ok": self.batch_ok,
            "sync_fallbacks": self.sync_fallbacks,
            "resubmitted": self.resubmitted,
            "failed": self.failed,
            "batch_ids": list(self.batch_ids),
        }
        # Only a resumed job run replays; every other payload keeps its shape.
        if self.replayed:
            out["replayed"] = self.replayed
        if self.bisections:
            out["bisections"] = self.bisections
        out.update(self.cost_json())
        return out


def cost_fields(cost: float, standard: float, unpriced: int) -> dict[str, float | None]:
    """The three cost fields of a ``batch`` block, ``null`` when any response
    they cover had no price from the provider (a partial sum would understate)."""
    if unpriced:
        return {"cost_usd": None, "standard_cost_usd": None, "saved_usd": None}
    return {
        "cost_usd": round(cost, 6),
        "standard_cost_usd": round(standard, 6),
        "saved_usd": round(standard - cost, 6),
    }


def _has_choices(response: Any) -> bool:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    return bool(choices)


def _mark_tier(response: Any, tier: str) -> None:
    """Stamp the serving path on the response without touching its cost."""
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        hidden[TIER_MARKER] = tier
        return
    try:
        response._hidden_params = {TIER_MARKER: tier}
    except (AttributeError, TypeError):
        pass


def cleanup_batch(
    backend: object, job: BatchJob, log: Callable[[str], None], *, tag: str = "[batch]"
) -> None:
    """Best-effort ``backend.cleanup(job)`` (provider-side artifacts of an
    ended, fully collected batch), when the backend has one. A failure is
    logged, never raised: cleanup is hygiene, not part of the result."""
    cleanup = getattr(backend, "cleanup", None)
    if cleanup is None:
        return
    try:
        cleanup(job)
    except Exception as exc:
        log(f"{tag} cleanup of {job.job_id} failed (ignored): {type(exc).__name__}: {exc}")


def uncertain_message(provider: str, requests: int, cause: BaseException) -> str:
    """What the user is told when a batch create's outcome is unknown."""
    return (
        f"creating a batch of {requests} request(s) on {provider!r} failed without proof "
        f"the provider rejected it ({type(cause).__name__}: {cause}); the batch may exist "
        "and be billing. It was not resubmitted and nothing ran at full price: check the "
        "provider's batch console and cancel it if present before retrying"
    )


class BatchExecutor:
    """Submit waves of completion kwargs to a :class:`BatchBackend`.

    ``sync_execute`` is the synchronous fallback for requests the batch path
    could not serve; it defaults to ``llm._completion_with_retry`` (resolved at
    call time so tests that patch the module attribute intercept it).
    ``sleep`` and ``log`` are injectable for tests. ``max_poll_s=None`` (the
    default) derives the polling deadline from the backend
    (:func:`default_max_poll_s`); an explicit value is used as given.
    ``deadline`` is the run's :class:`BatchDeadline` (shared by every executor
    of the run), or ``None`` for no deadline — exactly the behavior without one.
    ``sync_workers`` bounds how many of a wave's synchronous fallbacks run at
    once (the synchronous pipeline's own default, ``--max-parallel-calls``
    4): a wave that runs synchronously — after the deadline, below
    ``min_wave_size``, or what its batches could not serve — takes about as
    long as the synchronous pipeline would, not one request after another.
    """

    def __init__(
        self,
        backend: BatchBackend,
        *,
        sync_execute: Callable[[dict[str, Any]], Any] | None = None,
        poll_interval_s: float = 30.0,
        max_poll_s: float | None = None,
        min_wave_size: int = 1,
        max_item_retries: int = 1,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda _m: None,
        deadline: BatchDeadline | None = None,
        sync_workers: int = DEFAULT_SYNC_WORKERS,
    ) -> None:
        self.backend = backend
        self.sync_workers = max(1, sync_workers)
        # Fallbacks run on a pool: their counters are updated under this lock.
        self._count_lock = threading.Lock()
        self.deadline = deadline
        #: After the deadline fired: how long a canceled batch may take to end.
        self.cancel_settle_s = cancel_settle_s(backend)
        self._sync_execute = sync_execute
        self.poll_interval_s = poll_interval_s
        self.max_poll_s = default_max_poll_s(backend) if max_poll_s is None else max_poll_s
        self.min_wave_size = min_wave_size
        self.max_item_retries = max_item_retries
        self._sleep = sleep
        self._log = log
        self.stats = WaveStats()
        # Which pipeline stage the current waves serve ("transcribe", "links",
        # "extraction phase 3", ...), set by run_stage. Only log lines use it.
        self.stage: str | None = None
        self._stage_wave_base = 0

    @property
    def _tag(self) -> str:
        """Log prefix naming the stage and the wave within it, e.g.
        ``[batch links wave 2]``; plain ``[batch]`` outside a named stage."""
        if not self.stage:
            return "[batch]"
        return f"[batch {self.stage} wave {self.stats.waves - self._stage_wave_base}]"

    @contextlib.contextmanager
    def in_stage(self, stage: str | None) -> Iterator[None]:
        """Label every log line written while the block runs with *stage*, and
        number its waves from 1. Restores the previous label on exit."""
        prev = (self.stage, self._stage_wave_base)
        self.stage, self._stage_wave_base = stage, self.stats.waves
        try:
            yield
        finally:
            self.stage, self._stage_wave_base = prev

    # ---- public --------------------------------------------------------

    def run_wave(self, steps: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        """Serve every step; return ``{custom_id: ModelResponse | Exception}``.

        What the batch could not serve after the retry budget is executed
        synchronously; a request's outcome is an ``Exception`` only when that
        synchronous call raised (the same exception the sync path would have
        seen), so one request's failure never aborts the wave. Raises
        :class:`BatchExecutionFailed` only when the wave cannot be brought to
        an end: the provider accepted none of its batches, a poll or results
        call failed, or the shared polling deadline passed (every open job is
        cancelled first). A request the backend cannot encode, or whose batch
        alone was refused while others were accepted, runs synchronously.
        """
        self.stats.waves += 1
        self.stats.requests += len(steps)
        responses: dict[str, Any] = {}
        if not steps:
            return responses

        pending: dict[str, dict[str, Any]] = dict(steps)
        if len(pending) < self.min_wave_size:
            self._log(
                f"{self._tag} wave of {len(pending)} request(s) is below min_wave_size="
                f"{self.min_wave_size}; running synchronously"
            )
            responses.update(self._run_sync([(cid, kw, False) for cid, kw in pending.items()]))
            return responses
        if self._deadline_passed() and not self._has_open_batches(pending):
            self._log(
                f"{self._tag} batch deadline passed; running {len(pending)} request(s) "
                "synchronously"
            )
            responses.update(self._run_sync([(cid, kw, True) for cid, kw in pending.items()]))
            return responses

        attempts = 0
        while pending:
            served, failed = self._submit_and_collect(pending)
            responses.update(served)
            retry: dict[str, dict[str, Any]] = {}
            past_deadline = self._deadline_passed()
            to_sync: list[tuple[str, dict[str, Any], bool]] = []
            for custom_id, error in failed.items():
                retryable = error.retryable and attempts < self.max_item_retries
                if retryable and not past_deadline:
                    retry[custom_id] = pending[custom_id]
                    continue
                self._log(
                    f"{self._tag} {custom_id}: {error.kind} ({error.message}); "
                    "executing synchronously"
                )
                # The deadline's doing: its batch was canceled or never sent,
                # or it would have been resubmitted but for the deadline.
                deadline_caused = retryable or self._is_deadline_error(error)
                to_sync.append((custom_id, pending[custom_id], deadline_caused))
            responses.update(self._run_sync(to_sync))
            if retry:
                attempts += 1
                self.stats.resubmitted += len(retry)
                self._log(f"{self._tag} resubmitting {len(retry)} request(s) (attempt {attempts})")
            pending = retry
        return responses

    # ---- deadline --------------------------------------------------------

    def _deadline_passed(self) -> bool:
        """Whether the run's deadline has passed (the first time fires it)."""
        return self.deadline is not None and self.deadline.check()

    def _has_open_batches(self, pending: Mapping[str, dict[str, Any]]) -> bool:
        """Whether a provider batch submitted earlier still carries any of
        *pending* (a job's resume: see ``ReplayExecutor``). Such requests are
        canceled and collected after the deadline, never simply rerun."""
        return False

    def _poll_pause(self) -> float:
        """The sleep between two polling rounds: the poll interval, cut short
        so the round after it runs right at the deadline."""
        if self.deadline is None:
            return self.poll_interval_s
        return max(0.0, min(self.poll_interval_s, self.deadline.remaining()))

    @staticmethod
    def _deadline_error(custom_id: str, why: str) -> BatchItemError:
        return BatchItemError(custom_id, "canceled", f"{_DEADLINE_MESSAGE}{why}", retryable=False)

    @staticmethod
    def _own_verdict(failed: Mapping[str, BatchItemError], custom_id: str) -> bool:
        """Whether a canceled batch already answered *custom_id* ``invalid`` —
        a verdict on the request itself, kept as its outcome (it would run
        synchronously with or without the deadline), not replaced by a
        deadline error."""
        error = failed.get(custom_id)
        return error is not None and error.kind == "invalid"

    @staticmethod
    def _is_deadline_error(error: BatchItemError) -> bool:
        """Whether *error* is one :meth:`_deadline_error` made."""
        return error.kind == "canceled" and error.message.startswith(_DEADLINE_MESSAGE)

    def _count_deadline_fallback(self) -> None:
        """Count a request the deadline took off the batch path in the
        deadline's ``sync_after_deadline`` (a request that falls back for its
        own reason — it cannot be encoded, the provider called it invalid, the
        wave is below ``min_wave_size`` — is not)."""
        if self.deadline is not None:
            with self._count_lock:
                self.deadline.sync_after_deadline += 1

    def _run_sync(self, items: list[tuple[str, dict[str, Any], bool]]) -> dict[str, Any]:
        """Run ``(custom_id, kwargs, deadline_caused)`` requests synchronously,
        up to :attr:`sync_workers` at once; ``{custom_id: response or
        Exception}`` in *items* order. A deadline-caused one is counted in the
        deadline's ``sync_after_deadline`` (:meth:`_count_deadline_fallback`).
        Neither raises, so one request never stops the rest.

        Only the calls run on the pool: their costs are added on this thread,
        in *items* order, once all have returned — ``cost_usd`` is a float
        sum, and adding in completion order would make the job's totals depend
        on which request happened to finish first."""
        from dgml_core.concurrency import map_concurrent

        def run(item: tuple[str, dict[str, Any], bool]) -> Any:
            _cid, kwargs, deadline_caused = item
            if deadline_caused:
                self._count_deadline_fallback()
            return self._sync_call(kwargs)

        outcomes = map_concurrent(run, items, max_workers=self.sync_workers)
        for outcome in outcomes:
            self._account_sync(outcome)
        return {cid: outcome for (cid, _kw, _d), outcome in zip(items, outcomes, strict=True)}

    def _await_cancels(self, jobs: list[BatchJob]) -> set[str]:
        """Poll canceled *jobs* until each has ended or ``cancel_settle_s`` has
        elapsed on the deadline's clock, whichever is first; the ids of those
        that ended, whose results can now be collected.

        The wait is a time budget, not a number of rounds: it lasts the full
        ``cancel_settle_s`` whatever the poll interval (a pause is the poll
        interval capped at :data:`CANCEL_SETTLE_POLL_S`, and the last one is
        cut short so the final poll lands on the budget's end)."""
        assert self.deadline is not None
        pause = min(self.poll_interval_s, CANCEL_SETTLE_POLL_S)
        until = self.deadline.now() + self.cancel_settle_s
        ended: set[str] = set()
        waiting = list(jobs)
        # A backstop only: the clock ends the wait first, unless it never moves
        # (an injected sleep that does not advance it).
        rounds = int(self.cancel_settle_s / pause) + 2 if pause > 0 else 2
        for _ in range(rounds):
            still: list[BatchJob] = []
            for job in waiting:
                try:
                    done = self.backend.poll(job).done
                except Exception as exc:
                    self._log(f"{self._tag} polling canceled batch {job.job_id} failed: {exc}")
                    done = False
                if done:
                    ended.add(job.job_id)
                else:
                    still.append(job)
            waiting = still
            remaining = until - self.deadline.now()
            if not waiting or remaining <= 0:
                break
            self._sleep(min(pause, remaining) if pause > 0 else remaining)
        return ended

    def _cancel_and_collect(
        self,
        open_jobs: list[tuple[BatchJob, list[BatchRequest]]],
        pending: Mapping[str, dict[str, Any]],
        served: dict[str, Any],
        failed: dict[str, BatchItemError],
    ) -> None:
        """The deadline passed with *open_jobs* still running: cancel each,
        let the cancels settle, and collect what the provider already produced
        through the ordinary ``results`` contract. Every request left without
        a result lands in *failed*, non-retryable, so it runs synchronously."""
        assert self.deadline is not None
        canceled: list[tuple[BatchJob, list[BatchRequest]]] = []
        for job, batch in open_jobs:
            try:
                self.backend.cancel(job)
            except Exception as exc:
                self._log(
                    f"{self._tag} deadline: canceling {job.job_id} failed "
                    f"({type(exc).__name__}: {exc}); its requests run synchronously"
                )
                # Not canceled, so the provider may well run (and bill) it.
                self._unsettled(
                    job,
                    len(batch),
                    "no batch job records it, so its late cost is not reconciled",
                    why=f"could not be canceled ({type(exc).__name__}: {exc})",
                )
                for request in batch:
                    failed[request.custom_id] = self._deadline_error(
                        request.custom_id, "its batch could not be canceled"
                    )
                continue
            self.deadline.canceled_batches += 1
            canceled.append((job, batch))
        ended = self._await_cancels([job for job, _b in canceled])
        for job, batch in canceled:
            before = len(served)
            if job.job_id in ended:
                try:
                    self._collect(job, batch, pending, served, failed)
                except Exception as exc:
                    self._log(
                        f"{self._tag} collecting canceled batch {job.job_id} failed "
                        f"({type(exc).__name__}: {exc})"
                    )
                    # Whatever it processed is billed but unread: running those
                    # requests again synchronously may pay for them twice.
                    self._unsettled(
                        job,
                        sum(1 for r in batch if r.custom_id not in served),
                        "no batch job records it, so its late cost is not reconciled",
                        why=(
                            f"ended but its results could not be read ({type(exc).__name__}: {exc})"
                        ),
                    )
                else:
                    self._cleanup(job)
            else:
                unsettled = sum(1 for r in batch if r.custom_id not in served)
                self._unsettled(
                    job, unsettled, "no batch job records it, so its late cost is not reconciled"
                )
            got = len(served) - before
            self.deadline.collected_after_cancel += got
            for request in batch:
                if request.custom_id not in served and not self._own_verdict(
                    failed, request.custom_id
                ):
                    failed[request.custom_id] = self._deadline_error(
                        request.custom_id, "its batch was canceled before it ran"
                    )
            self._log(
                f"{self._tag} {job.job_id} canceled at the batch deadline: {got} of "
                f"{len(batch)} result(s) collected; the rest run synchronously"
            )

    def _unsettled(
        self, job: BatchJob, requests: int, what_next: str, *, why: str | None = None
    ) -> None:
        """A batch at the deadline whose outcome is unknown: its cancel did not
        settle within ``cancel_settle_s`` (the default *why*), the cancel
        itself failed, or the canceled batch's results could not be read. Its
        *requests* now run synchronously while the provider may still be
        processing (and billing) them — counted in the deadline's
        ``possibly_double_billed`` and reported as a WARNING (the output's
        cost is degraded). *what_next* says what becomes of the batch."""
        assert self.deadline is not None
        self.deadline.possibly_double_billed += requests
        why = why or f"was canceled but had not settled after {self.cancel_settle_s:.0f}s"
        self._log(f"{self._tag} batch {job.job_id} at the deadline {why}")
        logger.warning(
            "batch deadline: %s batch %s %s; its %d request(s) run synchronously and may be "
            "billed twice if the provider is still processing them (%s)",
            job.provider,
            job.job_id,
            why,
            requests,
            what_next,
        )

    # ---- internals -----------------------------------------------------

    def _submit_and_collect(
        self, pending: Mapping[str, dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, BatchItemError]]:
        """One submit → poll → collect round over *pending*.

        Every batch of the round is submitted first, then all of them are
        polled together under one ``max_poll_s`` budget, and each is collected
        as soon as it ends — so a wave split across batches takes as long as
        its slowest batch, not the sum of them.

        Returns ``(served, failed)``: served responses carry choices and the
        batch tier marker; every id in *pending* appears in exactly one of the
        two dicts. A request whose ``encode`` raised, or whose batch the
        provider refused at submit (while other batches were accepted), is in
        *failed* as non-retryable, so it runs synchronously. A batch rejected
        at batch level is bisected first (see the module docstring). An id the
        provider never reported is treated as errored.
        """
        served: dict[str, Any] = {}
        failed: dict[str, BatchItemError] = {}
        if self._deadline_passed():  # nothing new is submitted after it
            return served, {c: self._deadline_error(c, "not submitted") for c in pending}

        encodable = self._encodable(pending, failed)
        if not encodable:
            return served, failed

        jobs: list[tuple[BatchJob, list[BatchRequest]]] = []
        refused: list[tuple[list[BatchRequest], Exception]] = []

        def accept(job: BatchJob, batch: list[BatchRequest]) -> None:
            jobs.append((job, batch))

        plan = plan_batches(encodable, self.backend)
        for index, batch in enumerate(plan):
            try:
                self._submit_bisecting(batch, accept, refused)
            except BatchSubmitUncertain as exc:
                self._release_unsubmitted(plan[index + 1 :])
                raise self._uncertain_failure(batch, exc, [job for job, _b in jobs]) from exc
            except BatchExecutionFailed:  # throttled: stop what this wave started
                self._release_unsubmitted(plan[index + 1 :])
                self._cancel_quietly([job for job, _b in jobs])
                raise

        self._raise_if_nothing_accepted(bool(jobs), refused)
        self._fail_refused(refused, failed)

        self._wait_all(jobs, pending, served, failed)
        self.stats.batch_ok += len(served)
        return served, failed

    def _encodable(
        self, pending: Mapping[str, dict[str, Any]], failed: dict[str, BatchItemError]
    ) -> list[BatchRequest]:
        """*pending* as requests, minus any the backend cannot encode (those
        fail alone, as non-retryable items in *failed*, not the wave).

        :class:`BatchUnavailable` from ``encode`` is not a per-request failure:
        the backend says batch mode cannot serve the request at all (e.g. an
        OpenAI request litellm routes to the Responses API), so it propagates
        and fails the run loudly instead of running the request at full price."""
        encodable: list[BatchRequest] = []
        for custom_id, kwargs in pending.items():
            request = BatchRequest(custom_id, kwargs)
            try:
                request_size(request, self.backend)
            except BatchUnavailable:
                self._release(encodable)
                raise
            except Exception as exc:
                self._log(f"{self._tag} {custom_id}: cannot encode ({type(exc).__name__}: {exc})")
                failed[custom_id] = BatchItemError(
                    custom_id, "invalid", f"encode failed: {type(exc).__name__}: {exc}"
                )
                continue
            encodable.append(request)
        return encodable

    def _submit_bisecting(
        self,
        batch: list[BatchRequest],
        accept: Callable[[BatchJob, list[BatchRequest]], None],
        refused: list[tuple[list[BatchRequest], Exception]],
    ) -> None:
        """Submit *batch*; *accept* each batch the provider takes.

        A :class:`BatchRejected` batch of more than one request is split in
        half and each half submitted the same way (depth-first, in request
        order, so the same batch always yields the same sub-batches); a single
        rejected request, or any other submit error, lands in *refused*.
        :class:`BatchSubmitUncertain` propagates untouched — the caller must
        fail the wave, never resubmit. ``backend.release`` follows every
        attempt, accepted or not."""
        try:
            try:
                job = self.backend.submit(batch)
            finally:
                self._release(batch)
        except BatchSubmitUncertain:
            raise
        except BatchThrottled as exc:
            # Neither a smaller batch nor a full-price synchronous run is the
            # answer to the account's rate limit or quota: fail the wave. The
            # caller cancels the wave's other open batches.
            raise self._throttled_failure(batch, exc) from exc
        except BatchRejected as exc:
            if len(batch) > 1:
                self._bisect(batch, exc, accept, refused)
                return
            self._log(
                f"{self._tag} {batch[0].custom_id}: rejected at batch level even alone "
                f"({exc}); executing synchronously"
            )
            refused.append((batch, exc))
            return
        except Exception as exc:
            # The backend contract: anything but BatchSubmitUncertain means the
            # provider did not accept the batch (a post-send or post-2xx
            # failure is uncertain, never this), so running it sync is safe.
            self._log(
                f"{self._tag} submission of {len(batch)} request(s) refused: "
                f"{type(exc).__name__}: {exc}"
            )
            refused.append((batch, exc))
            return
        self.stats.batches += 1
        self.stats.batch_ids.append(job.job_id)
        self._log(f"{self._tag} submitted {job.job_id} ({len(batch)} request(s))")
        accept(job, batch)

    def _bisect(
        self,
        batch: list[BatchRequest],
        why: object,
        accept: Callable[[BatchJob, list[BatchRequest]], None],
        refused: list[tuple[list[BatchRequest], Exception]],
    ) -> None:
        """Resubmit *batch* (rejected at batch level: *why*) as two halves."""
        self.stats.bisections += 1
        mid = len(batch) // 2
        self._log(
            f"{self._tag} batch of {len(batch)} request(s) rejected at batch level ({why}); "
            f"resubmitting as {mid} + {len(batch) - mid}"
        )
        self._submit_bisecting(batch[:mid], accept, refused)
        self._submit_bisecting(batch[mid:], accept, refused)

    def _raise_if_nothing_accepted(
        self, accepted: bool, refused: list[tuple[list[BatchRequest], Exception]]
    ) -> None:
        """The provider accepted nothing this round: a batch-level failure,
        not a per-request one (running every request at full price would hide
        an outage or a misconfiguration). Single requests rejected at batch
        level after bisection are per-request by construction, so they alone
        do not trigger it."""
        if accepted:
            return
        hard = [(b, e) for b, e in refused if not isinstance(e, BatchRejected)]
        if not hard:
            return
        first_batch, cause = hard[0]
        raise BatchExecutionFailed(
            f"batch submission of {len(first_batch)} request(s) to "
            f"{self.backend.provider!r} failed: {type(cause).__name__}: {cause}"
        ) from cause

    @staticmethod
    def _fail_refused(
        refused: list[tuple[list[BatchRequest], Exception]], failed: dict[str, BatchItemError]
    ) -> None:
        for refused_batch, why in refused:
            kind: ItemErrorKind = "batch_rejected" if isinstance(why, BatchRejected) else "invalid"
            for request in refused_batch:
                failed[request.custom_id] = BatchItemError(
                    request.custom_id,
                    kind,
                    f"batch submission refused: {type(why).__name__}: {why}",
                    retryable=False,
                )

    def _uncertain_failure(
        self, batch: list[BatchRequest], cause: BaseException, open_jobs: list[BatchJob]
    ) -> BatchExecutionFailed:
        """The wave's failure after an uncertain batch create: every other
        open job of the wave is canceled (best-effort), nothing is resubmitted
        or run synchronously, and the message tells the user to check."""
        self._cancel_quietly(open_jobs)
        self._log(f"{self._tag} batch create outcome unknown: {type(cause).__name__}: {cause}")
        error = BatchExecutionFailed(uncertain_message(self.backend.provider, len(batch), cause))
        error.__cause__ = cause
        return error

    def _release_unsubmitted(self, batches: Iterable[list[BatchRequest]]) -> None:
        """The wave failed before *batches* were submitted: drop the encodings
        planning cached for them (``release`` otherwise follows a submit)."""
        remaining = [request for batch in batches for request in batch]
        if remaining:
            self._release(remaining)

    def _cancel_quietly(self, jobs: Iterable[BatchJob]) -> None:
        """Best-effort cancel of *jobs* on the way out of a failed wave."""
        for job in jobs:
            with contextlib.suppress(Exception):
                self.backend.cancel(job)

    def _throttled_failure(
        self, batch: list[BatchRequest], cause: BatchThrottled
    ) -> BatchExecutionFailed:
        """The wave's failure after a create refused at the rate limit or
        quota: nothing is bisected or run synchronously, and the message says
        what to do."""
        self._log(f"{self._tag} batch create throttled: {cause}")
        return BatchExecutionFailed(
            f"creating a batch of {len(batch)} request(s) on {self.backend.provider!r} was "
            f"refused at the provider's rate limit or quota ({cause}). Nothing was split or "
            "run at full price, and the wave's other open batches are canceled. Retry "
            "later, or check the account's billing and quota"
        )

    def _release(self, batch: list[BatchRequest]) -> None:
        release = getattr(self.backend, "release", None)
        if release is None:
            return
        try:
            release([r.custom_id for r in batch])
        except Exception as exc:  # hygiene only: never fails the run
            self._log(f"{self._tag} backend release failed: {type(exc).__name__}: {exc}")

    def _cleanup(self, job: BatchJob) -> None:
        cleanup_batch(self.backend, job, self._log, tag=self._tag)

    @staticmethod
    def _rejected_whole(ids: Iterable[str], failed: Mapping[str, BatchItemError]) -> bool:
        """Every id in *ids* came back ``batch_rejected``."""
        listed = list(ids)
        return bool(listed) and all(
            (err := failed.get(cid)) is not None and err.kind == "batch_rejected" for cid in listed
        )

    def _wait_all(
        self,
        jobs: list[tuple[BatchJob, list[BatchRequest]]],
        pending: Mapping[str, dict[str, Any]],
        served: dict[str, Any],
        failed: dict[str, BatchItemError],
    ) -> None:
        """Poll every job together; collect each as it ends.

        A collected batch whose every item was rejected at batch level is
        bisected and its halves join the jobs being polled. A poll or results
        failure, an uncertain create, or the shared polling deadline passing,
        cancels every job still open (best-effort, so nothing keeps billing) and
        raises :class:`BatchExecutionFailed` chained to the cause.
        """
        deadline = time.monotonic() + self.max_poll_s
        open_jobs = list(jobs)
        # Halves of a batch rejected at batch level, submitted this round.
        bisected: list[tuple[BatchJob, list[BatchRequest]]] = []

        def cancel_open() -> None:
            for job, _batch in open_jobs + bisected:
                with contextlib.suppress(Exception):
                    self.backend.cancel(job)

        def abort(message: str, cause: BaseException | None) -> BatchExecutionFailed:
            cancel_open()
            error = BatchExecutionFailed(message)
            error.__cause__ = cause
            return error

        while open_jobs:
            still_open: list[tuple[BatchJob, list[BatchRequest]]] = []
            for job, batch in open_jobs:
                try:
                    status = self.backend.poll(job)
                except Exception as exc:
                    raise abort(
                        f"polling batch {job.job_id} failed: {type(exc).__name__}: {exc}", exc
                    ) from exc
                if not status.done:
                    still_open.append((job, batch))
                    continue
                self._log(
                    f"{self._tag} {job.job_id} {status.state.value}: {status.succeeded} ok, "
                    f"{status.errored} errored, {status.expired} expired, "
                    f"{status.canceled} canceled"
                )
                try:
                    self._collect(job, batch, pending, served, failed)
                except Exception as exc:
                    open_jobs = [j for j in open_jobs if j[0] is not job]
                    raise abort(
                        f"collecting results of batch {job.job_id} failed: "
                        f"{type(exc).__name__}: {exc}",
                        exc,
                    ) from exc
                self._cleanup(job)
                ids = [r.custom_id for r in batch]
                if (
                    len(batch) > 1
                    and self._rejected_whole(ids, failed)
                    and not self._deadline_passed()
                ):
                    why = failed[ids[0]].message
                    for cid in ids:
                        del failed[cid]
                    refused: list[tuple[list[BatchRequest], Exception]] = []
                    try:
                        self._bisect(batch, why, lambda j, b: bisected.append((j, b)), refused)
                    except BatchSubmitUncertain as exc:
                        open_jobs = [j for j in open_jobs if j[0] is not job]
                        cancel_open()
                        raise self._uncertain_failure(batch, exc, []) from exc
                    except BatchExecutionFailed:  # a half was throttled
                        open_jobs = [j for j in open_jobs if j[0] is not job]
                        cancel_open()
                        raise
                    self._fail_refused(refused, failed)
            open_jobs = still_open + bisected
            bisected.clear()
            if not open_jobs:
                return
            if self._deadline_passed():
                self._cancel_and_collect(open_jobs, pending, served, failed)
                return
            if time.monotonic() >= deadline:
                ids_text = ", ".join(job.job_id for job, _b in open_jobs)
                raise abort(
                    f"batch {ids_text} not finished after {self.max_poll_s:.0f}s; canceled", None
                )
            self._sleep(self._poll_pause())

    def _collect(
        self,
        job: BatchJob,
        batch: list[BatchRequest],
        pending: Mapping[str, dict[str, Any]],
        served: dict[str, Any],
        failed: dict[str, BatchItemError],
    ) -> None:
        seen: set[str] = set()
        for custom_id, outcome in self.backend.results(job):
            if custom_id not in pending or custom_id in seen:
                continue  # unknown or duplicate id: never trust position
            seen.add(custom_id)
            if isinstance(outcome, BatchItemError):
                failed[custom_id] = outcome
            elif not _has_choices(outcome):
                failed[custom_id] = BatchItemError(
                    custom_id, "errored", "batch result carries no choices"
                )
            else:
                _mark_tier(outcome, TIER_BATCH)
                self.stats.add_cost(outcome, TIER_BATCH)
                served[custom_id] = outcome
        for request in batch:
            if request.custom_id not in seen:
                failed[request.custom_id] = BatchItemError(
                    request.custom_id, "errored", "no result returned for this request"
                )

    def _fallback(self, kwargs: dict[str, Any]) -> Any:
        """Run one request synchronously; its response, or the ``Exception`` it
        raised (returned, not raised)."""
        outcome = self._sync_call(kwargs)
        self._account_sync(outcome)
        return outcome

    def _account_sync(self, outcome: Any) -> None:
        """Add a synchronous fallback's cost (standard tier) — on the thread
        that folds the wave, never on a pool worker (see :meth:`_run_sync`)."""
        if not isinstance(outcome, BaseException):
            self.stats.add_cost(outcome, TIER_STANDARD)

    def _sync_call(self, kwargs: dict[str, Any]) -> Any:
        """One synchronous fallback call — safe on a pool worker: it
        touches only lock-guarded counters and leaves cost to
        :meth:`_account_sync`."""
        run = self._sync_execute
        if run is None:
            from dgml_core import llm  # resolved at call time (see class docstring)

            run = llm._completion_with_retry
        with self._count_lock:
            self.stats.sync_fallbacks += 1
        try:
            response = run(kwargs)
        except Exception as exc:
            with self._count_lock:
                self.stats.failed += 1
            self._log(f"{self._tag} synchronous fallback failed: {type(exc).__name__}: {exc}")
            return exc
        _mark_tier(response, TIER_STANDARD)
        return response


def make_executor(
    model: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    poll_interval_s: float = 30.0,
    max_poll_s: float | None = None,
    min_wave_size: int = 1,
    log: Callable[[str], None] = lambda _m: None,
    credential: Mapping[str, Any] | None = None,
    deadline: BatchDeadline | None = None,
) -> BatchExecutor:
    """The one way to build a :class:`BatchExecutor` for *model*.

    Resolves *model*'s batch backend with the given credentials and wraps it.
    Raises :class:`~dgml_core.errors.BatchUnavailable` when the provider has no
    batch backend (or its dependency is missing) — never a silent fallback to
    full-price synchronous calls. Every batch caller builds its executor here
    so the knobs mean the same thing everywhere. ``max_poll_s=None`` derives
    the polling deadline from the backend's ``max_wait_s`` (see
    :func:`default_max_poll_s`).

    While a batch job session is active (:mod:`dgml_core.batch.jobs`) the
    executor is a :class:`~dgml_core.batch.jobs.ReplayExecutor` bound to it,
    so every caller records and replays through the job without knowing.
    *credential* is the non-secret pointer to where *api_key* came from
    (:func:`dgml_core.batch.jobs.credential_ref`); a job records it with each
    provider batch so ``dgml batch status``/``cancel`` can re-resolve the key.

    *deadline* is the run's :class:`~dgml_core.batch.deadline.BatchDeadline`
    for a caller without a job session. Under a session the job's own deadline
    (:attr:`~dgml_core.batch.jobs.JobSession.deadline`, stored in its
    manifest) applies instead, so every executor of a run — every step of
    ``docset run`` — shares the one deadline without any caller passing it.
    """
    from dgml_core.batch.jobs import ReplayExecutor, active_session
    from dgml_core.batch.registry import resolve_backend

    backend = resolve_backend(model, api_key=api_key, api_base=api_base)
    session = active_session()
    if session is not None:
        return ReplayExecutor(
            backend,
            session=session,
            model=model,
            api_base=api_base,
            credential=credential,
            poll_interval_s=poll_interval_s,
            max_poll_s=max_poll_s,
            min_wave_size=min_wave_size,
            log=log,
            deadline=session.deadline if session.deadline is not None else deadline,
        )
    return BatchExecutor(
        backend,
        poll_interval_s=poll_interval_s,
        max_poll_s=max_poll_s,
        min_wave_size=min_wave_size,
        log=log,
        deadline=deadline,
    )
