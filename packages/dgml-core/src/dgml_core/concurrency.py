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

"""Bounded thread-pool fan-out for independent per-page work.

Several places in this package dispatch N independent, network-bound units of
work — per-page OCR provider calls, per-page vision-style calls — and then fold
the results back together on the calling thread. :func:`map_concurrent` is the
one shape they need, so they don't each hand-roll a ``ThreadPoolExecutor``.

Deliberately stdlib-only and dependency-free: :mod:`dgml_core.utils` imports
``Workspace``/``FileStore``/``DocSetStore`` and ``files`` imports ``ocr``, so
putting this there would close an ``ocr -> utils -> files -> ocr`` import cycle.
Nothing here is exported from ``dgml_core/__init__.py``; it is internal.

**Every task runs in a copy of the caller's context** (:mod:`contextvars`). A
``ThreadPoolExecutor`` worker otherwise starts with an empty one, so anything a
caller binds there — a job id its log handler stamps onto every record, a
structlog ``bind_contextvars`` — would vanish from lines logged on the worker.
Each task gets its own copy, made on the submitting thread: a context cannot be
entered by two threads at once, and a copy taken on the worker would be the
worker's (empty) one. The other pools in this package do the same.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TypeVar, cast

T = TypeVar("T")
R = TypeVar("R")


def map_concurrent(fn: Callable[[T], R], items: Sequence[T], *, max_workers: int) -> list[R]:
    """Apply ``fn`` to every element of ``items`` on a bounded thread pool.

    Returns one result per item **in ``items`` order**, regardless of the order
    in which they complete — so a caller folding results into shared state (an
    XML tree, a counter, a dict) does so deterministically, on its own thread.
    Only ``fn`` ever runs on a worker; nothing here touches caller state.

    Worker count is clamped to ``max(1, min(max_workers, len(items)))``. When
    that resolves to one — a single item, or an explicitly serial caller — the
    items run inline on the calling thread, with no pool and no thread spawned,
    so the common single-page case pays nothing and stays easy to debug.
    Either way each item runs in its own copy of the caller's context (see the
    module docstring): ``fn`` sees what the caller bound, and what ``fn`` binds
    reaches neither the next item nor the caller.

    **Failure is fail-fast**, for callers whose ``fn`` can raise. On the first
    exception, queued-but-unstarted items are cancelled (so a dead API isn't
    hammered N more times), already in-flight items are allowed to finish and
    their results discarded, and that first exception is re-raised once the pool
    has drained. Cancellation is best-effort by construction: ``Future.cancel``
    cannot interrupt work that has already started, so up to ``max_workers``
    items may still complete. ``BaseException`` is caught rather than
    ``Exception`` so a ``KeyboardInterrupt``, or the ``CancelledError`` of a
    cancelled future, takes the same path instead of escaping the drain loop.

    **Callers wanting per-item independence make ``fn`` total** — catch inside
    ``fn`` and return a sentinel or a result object, as
    :mod:`dgml_core.style_llm` does. Nothing then ever raises out of a worker,
    no item can affect another, and every item runs. Failure isolation is a
    property of the unit of work, not of the executor; fail-fast is what you
    get only when you let exceptions escape ``fn``.
    """
    workers = max(1, min(max_workers, len(items)))
    if workers == 1:
        return [contextvars.copy_context().run(fn, item) for item in items]

    results: list[R | None] = [None] * len(items)
    first_exc: BaseException | None = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[R], int] = {
            executor.submit(contextvars.copy_context().run, fn, item): index
            for index, item in enumerate(items)
        }
        for future in as_completed(futures):
            try:
                results[futures[future]] = future.result()
            except BaseException as exc:
                if first_exc is None:
                    first_exc = exc
                    # Cancel pending items so we don't keep hammering a known
                    # failure. In-flight ones finish; their results are dropped.
                    for pending in futures:
                        pending.cancel()
                continue

    if first_exc is not None:
        raise first_exc
    # Every index was assigned: we only reach here when no future raised, and
    # `as_completed` yields all of them.
    return cast("list[R]", results)
