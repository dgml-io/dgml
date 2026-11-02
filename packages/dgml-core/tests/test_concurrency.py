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

"""Contract tests for the shared bounded thread-pool fan-out.

The properties locked here are the ones both callers depend on:
input-order results (so folds stay deterministic), real parallel dispatch,
the inline serial fast path, and the fail-fast/drain semantics inherited
from the hand-rolled loop that used to live in ``ocr.py``.

There is deliberately no test for cancel-pending-on-failure: ``Future.cancel``
racing the pool picking up the next item is inherently timing-dependent, so any
"fewer than N items started" assertion would flake.
"""

from __future__ import annotations

import contextvars
import threading
import time

import pytest
from dgml_core.concurrency import map_concurrent


class _Boom(Exception):
    """Distinct type so a test can assert on identity, not just `Exception`."""


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_map_concurrent_preserves_input_order(workers: int) -> None:
    """Results come back in `items` order even when later items finish first.

    Every caller folds results into shared state (a counter, an XML tree) on
    its own thread, so this ordering is what makes those folds deterministic.
    """

    def slow_in_reverse(i: int) -> int:
        # Item 0 sleeps longest, so completion order is the reverse of input.
        time.sleep((4 - i) * 0.01)
        return i * 2

    assert map_concurrent(slow_in_reverse, list(range(5)), max_workers=workers) == [
        0,
        2,
        4,
        6,
        8,
    ]


def test_map_concurrent_dispatches_in_parallel() -> None:
    """Items really are dispatched concurrently.

    Each item holds a barrier until all four workers arrive, then they release
    together. A serial implementation never completes the barrier, so this
    fails at the 5s timeout instead of hanging CI.
    """
    barrier = threading.Barrier(4, timeout=5.0)

    def wait_for_everyone(i: int) -> int:
        barrier.wait()
        return i

    assert map_concurrent(wait_for_everyone, [0, 1, 2, 3], max_workers=4) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    ("items", "workers"),
    [
        ([0, 1, 2, 3], 1),  # explicitly serial
        ([0], 8),  # single item: nothing to parallelize
    ],
)
def test_map_concurrent_runs_inline_when_serial(items: list[int], workers: int) -> None:
    """A one-worker fan-out spawns no thread — it runs on the calling thread.

    This is what lets single-page documents keep their pre-parallelism
    behavior exactly, and keeps the common case debuggable.
    """
    seen: list[threading.Thread] = []

    def record_thread(i: int) -> int:
        seen.append(threading.current_thread())
        return i

    map_concurrent(record_thread, items, max_workers=workers)

    assert seen and all(t is threading.main_thread() for t in seen)


def test_map_concurrent_reraises_first_exception() -> None:
    """The original exception object propagates, not a wrapper."""
    boom = _Boom("page 2 exploded")

    def raise_on_two(i: int) -> int:
        if i == 2:
            raise boom
        return i

    with pytest.raises(_Boom) as exc_info:
        map_concurrent(raise_on_two, list(range(6)), max_workers=4)
    assert exc_info.value is boom


def test_map_concurrent_serial_stops_at_first_failure() -> None:
    """The deterministic half of fail-fast: inline, nothing runs after the
    failure (the pooled path can only cancel *unstarted* work)."""
    calls: list[int] = []

    def raise_on_first(i: int) -> int:
        calls.append(i)
        raise _Boom("immediately")

    with pytest.raises(_Boom):
        map_concurrent(raise_on_first, [0, 1, 2, 3], max_workers=1)
    assert calls == [0]


def test_map_concurrent_drains_in_flight_work() -> None:
    """In-flight items finish before the exception surfaces — we drain the
    pool rather than abandoning work that already started."""
    barrier = threading.Barrier(3, timeout=5.0)
    finished: list[int] = []
    lock = threading.Lock()

    def fail_first_after_barrier(i: int) -> int:
        barrier.wait()  # all three are in flight before any of them returns
        if i == 0:
            raise _Boom("after everyone started")
        with lock:
            finished.append(i)
        return i

    with pytest.raises(_Boom):
        map_concurrent(fail_first_after_barrier, [0, 1, 2], max_workers=3)
    assert sorted(finished) == [1, 2]


def test_map_concurrent_total_fn_runs_every_item() -> None:
    """A `fn` that catches its own exceptions makes items fully independent:
    nothing is cancelled, everything runs, and results still align by index.

    This is the contract `style_llm` relies on for per-page independence.
    """
    calls: list[int] = []
    lock = threading.Lock()

    def never_raises(i: int) -> tuple[str, object]:
        with lock:
            calls.append(i)
        try:
            if i % 2:
                raise _Boom(f"item {i}")
        except _Boom as exc:
            return ("failed", str(exc))
        return ("ok", i)

    results = map_concurrent(never_raises, list(range(6)), max_workers=3)

    assert sorted(calls) == list(range(6))
    assert results == [
        ("ok", 0),
        ("failed", "item 1"),
        ("ok", 2),
        ("failed", "item 3"),
        ("ok", 4),
        ("failed", "item 5"),
    ]


def test_map_concurrent_empty() -> None:
    """No items means no work and no pool."""
    calls: list[int] = []

    assert map_concurrent(calls.append, [], max_workers=8) == []
    assert calls == []


_JOB_ID: contextvars.ContextVar[str] = contextvars.ContextVar("test_job_id", default="-")


@pytest.mark.parametrize("workers", [1, 4])
def test_map_concurrent_runs_items_in_the_callers_context(workers: int) -> None:
    """What a caller binds (a job id its log handler stamps on every record) is
    visible to ``fn`` on a worker thread, as on the inline path. A bare
    ``ThreadPoolExecutor`` worker would see only the default."""
    token = _JOB_ID.set("job-42")
    try:
        seen = map_concurrent(lambda _i: _JOB_ID.get(), list(range(6)), max_workers=workers)
    finally:
        _JOB_ID.reset(token)

    assert seen == ["job-42"] * 6


def test_map_concurrent_gives_each_item_its_own_context() -> None:
    """Each item runs in its own copy: a value one item sets is neither seen by
    the others nor left behind in the caller."""
    barrier = threading.Barrier(3)

    def bind_then_read(i: int) -> str:
        _JOB_ID.set(f"item-{i}")
        barrier.wait(timeout=5)  # all three have set theirs before any reads
        return _JOB_ID.get()

    assert map_concurrent(bind_then_read, [0, 1, 2], max_workers=3) == [
        "item-0",
        "item-1",
        "item-2",
    ]
    assert _JOB_ID.get() == "-"


def test_map_concurrent_inline_items_get_their_own_context_too() -> None:
    """The serial path (no pool) isolates items the same way: each starts from
    the caller's value, not the previous item's, and the caller keeps its own."""

    def read_then_bind(i: int) -> str:
        before = _JOB_ID.get()
        _JOB_ID.set(f"item-{i}")
        return before

    assert map_concurrent(read_then_bind, [0, 1, 2], max_workers=1) == ["-", "-", "-"]
    assert _JOB_ID.get() == "-"
