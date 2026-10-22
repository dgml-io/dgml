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

"""A run-level batch deadline: past it, stop waiting on the provider.

A provider batch usually ends within an hour but may legitimately run for its
whole lifetime (24 hours), and a single slow wave then holds the whole run.
A :class:`BatchDeadline` bounds that. It is an absolute **wall-clock** time
(UTC epoch seconds — a job's deadline must mean the same instant in the
process that resumes it, so it cannot be monotonic), shared by every executor
of one run (``docset run``: every step), and it counts what firing cost.

Once the deadline has passed, an executor stops waiting: every open provider
batch of its wave is canceled, the cancel is given a bounded, per-provider
time to settle (the backend's ``cancel_settle_s``, else :data:`CANCEL_SETTLE_S`),
and whatever results the provider already produced are collected through the
backend's ordinary ``results`` contract (a request that got a batch result
keeps it, at batch price). Every other request of that wave, and every
request of every later wave, runs synchronously through the normal fallback
path at standard price — nothing new is submitted.

**A cancel that does not settle in time** may still be processing: OpenAI
keeps working on a ``cancelling`` batch, so its requests, run synchronously
now, can be billed twice. Each such request counts in
``possibly_double_billed``. A job keeps that batch's record (``settling``),
and a later run of the job, ``dgml batch cancel`` or ``prune`` records what
it billed once it has ended (``dgml batch status`` only polls it; see
:mod:`dgml_core.batch.jobs`). A ``--no-wait`` run never waits a cancel out:
what has not settled at its one check is kept settling at once. A run
without a job can only warn and count.

A job stores its deadline in the manifest on the run that set it (see
:mod:`dgml_core.batch.jobs`), so every resume honors the same instant.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

#: How long an executor waits, after the deadline fired, for a canceled
#: batch to settle (end) so its already-processed results can be collected —
#: the fallback for a backend that declares no ``cancel_settle_s`` of its own.
CANCEL_SETTLE_S = 180.0
#: Longest pause between two polls of a settling cancel. Providers settle
#: cancels on multi-minute sweep cycles, so polling faster gains nothing.
CANCEL_SETTLE_POLL_S = 10.0
#: Longest ``--batch-deadline`` accepted. A job may pause (``--no-wait``) and
#: resume for days, but a deadline beyond a month bounds nothing a provider
#: batch could do, and an unbounded one overflows the UTC time it is stored as.
MAX_DEADLINE_S = 30 * 86400.0


def cancel_settle_s(backend: object) -> float:
    """How long to wait for a canceled batch of *backend* to settle: its
    ``cancel_settle_s`` when it declares a positive number, else
    :data:`CANCEL_SETTLE_S`."""
    value = getattr(backend, "cancel_settle_s", CANCEL_SETTLE_S)
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return float(value)
    return CANCEL_SETTLE_S


_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.IGNORECASE)


def parse_duration(text: str) -> float:
    """Seconds in *text*: ``90m``, ``6h``, ``1d``, ``45s`` or plain seconds
    (``5400``). Raises :class:`ValueError` for anything else, or a duration
    that is not strictly positive or exceeds :data:`MAX_DEADLINE_S`."""
    match = _DURATION_RE.match(str(text))
    if match is None:
        raise ValueError(
            f"{text!r} is not a duration: use a number of seconds or a number with "
            "a unit s, m, h or d (e.g. 90m, 6h, 1d)"
        )
    seconds = float(match.group(1)) * _UNITS[(match.group(2) or "s").lower()]
    if seconds <= 0:
        raise ValueError(f"a batch deadline must be positive, got {text!r}")
    if seconds > MAX_DEADLINE_S:
        raise ValueError(f"a batch deadline may be at most 30d, got {text!r}")
    return seconds


def wall_clock() -> float:
    """The default clock: UTC epoch seconds (resolved at call time, so tests
    can patch this function)."""
    return time.time()


def format_utc(epoch: float) -> str:
    """*epoch* as ISO 8601 UTC with a ``Z`` suffix, second precision."""
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> float:
    """The epoch seconds of an ISO 8601 time :func:`format_utc` wrote."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()


class DeadlineCounter(StrEnum):
    """The counters a deadline reports (``batch.deadline`` keys, and the
    :class:`BatchDeadline` attributes), in payload order."""

    CANCELED_BATCHES = "canceled_batches"
    COLLECTED_AFTER_CANCEL = "collected_after_cancel"
    SYNC_AFTER_DEADLINE = "sync_after_deadline"
    POSSIBLY_DOUBLE_BILLED = "possibly_double_billed"


@dataclass
class BatchDeadline:
    """An absolute deadline (*at*, UTC epoch seconds) for one run's batches.

    ``clock`` is injectable for tests (default :func:`wall_clock`). The
    counters are this run's: batches canceled because the deadline passed,
    requests collected from them anyway, requests run synchronously after
    it, and — among those — requests whose canceled batch had not settled
    when the wait ended (``possibly_double_billed``: the provider may still
    process and bill them). ``fired`` is set the first time an executor finds the deadline passed
    (one WARNING is logged then; the output's cost is degraded)."""

    at: float
    clock: Callable[[], float] | None = None
    fired: bool = False
    canceled_batches: int = 0
    collected_after_cancel: int = 0
    sync_after_deadline: int = 0
    possibly_double_billed: int = 0
    #: Counters of earlier runs of the same job (summed into :meth:`to_json`).
    prior: list[dict[str, Any]] = field(default_factory=list)
    #: Guards the counters: every executor of the run (each stage, and each
    #: one's fallback pool) shares this deadline, so :meth:`count` is the only
    #: way they change.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @classmethod
    def after(cls, seconds: float, *, clock: Callable[[], float] | None = None) -> BatchDeadline:
        """A deadline *seconds* from now, in whole seconds (the precision a
        job's manifest stores it with, so every run sees the same instant),
        rounded up: never shorter than asked, and never already past."""
        now = (clock or wall_clock)()
        return cls(at=float(math.ceil(now + float(seconds))), clock=clock)

    def now(self) -> float:
        return (self.clock or wall_clock)()

    def remaining(self) -> float:
        """Seconds left (negative once passed)."""
        return self.at - self.now()

    @property
    def expired(self) -> bool:
        """Whether the deadline has passed (fired or not)."""
        return self.fired or self.now() >= self.at

    def check(self) -> bool:
        """Whether the deadline has passed; the first ``True`` fires it."""
        if self.fired:
            return True
        if self.now() < self.at:
            return False
        self.fired = True
        logger.warning(
            "batch deadline %s passed: open provider batches are canceled (results "
            "already processed are kept, at batch price) and the rest of this run runs "
            "synchronously at standard price",
            format_utc(self.at),
        )
        return True

    def count(self, counter: DeadlineCounter, n: int = 1) -> None:
        """Add *n* to *counter*, thread-safely."""
        counter = DeadlineCounter(counter)  # ValueError for an unknown name
        with self._lock:
            setattr(self, counter, int(getattr(self, counter)) + n)

    def counters(self) -> dict[str, int]:
        """This run's counters (what a job records per run)."""
        with self._lock:
            return {str(c): int(getattr(self, c)) for c in DeadlineCounter}

    def to_json(self) -> dict[str, Any]:
        """The payload's ``batch.deadline`` block: the deadline, whether it
        has passed, and the counters summed over every run of the job."""
        out: dict[str, Any] = {"at": format_utc(self.at), "expired": self.expired}
        for name, value in self.counters().items():
            out[name] = value + sum(int(p.get(name, 0)) for p in self.prior)
        return out
