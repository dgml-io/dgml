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

"""The :class:`dgml_core.llm.FanOut` step: fork child step generators, join on
their results — run one after another by the sync driver, side by side (in the
same waves) by the batch driver, with identical requests, results, failure
delivery and usage folding."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dgml_core import llm
from dgml_core.batch import (
    BatchExecutor,
    BatchItemError,
    BatchRequest,
    FakeBackend,
    Unit,
    run_stage,
)
from dgml_core.storage import Workspace
from dgml_core.usage import read_events

from .conftest import FakeLLMResponse

MODEL = "anthropic/claude-haiku-4-5"

#: Per-request costs chosen so the float sum depends on the fold order: child
#: order (a1, a2, b1) and batch arrival order (a1, b1, a2) disagree in the last
#: digit, so a driver that folded in arrival order would fail the bit check.
COSTS = {"a1": 0.1, "a2": 0.2, "b1": 0.17, "p1": 0.0, "p2": 0.0}


def _cfg(ws: Workspace | None = None) -> llm.LLMConfig:
    return llm.LLMConfig(model=MODEL, workspace=ws, debug=ws is not None, operation="label")


def _tag(kwargs: dict[str, Any]) -> str:
    return str(kwargs["messages"][-1]["content"][0]["text"])


def _answer(kwargs: dict[str, Any]) -> FakeLLMResponse:
    tag = _tag(kwargs)
    return FakeLLMResponse(f"r:{tag}", cost=COSTS.get(tag, 0.01), prompt_tokens=10)


def _call(cfg: llm.LLMConfig, tag: str) -> llm.LLMSteps[str]:
    return llm.steps_call(cfg, system_prompt="S", user_content=[{"type": "text", "text": tag}])


def _child(cfg: llm.LLMConfig, *tags: str, fail: bool = False) -> llm.LLMSteps[list[str]]:
    out = []
    for tag in tags:
        out.append((yield from _call(cfg, tag)))
    if fail:
        raise ValueError(f"child {tags[0]} failed")
    return out


def _parent(cfg: llm.LLMConfig, *, fail_b: bool = False, catch: bool = False) -> llm.LLMFlow[Any]:
    first = yield from _call(cfg, "p1")
    try:
        joined = yield llm.FanOut([_child(cfg, "a1", "a2"), _child(cfg, "b1", fail=fail_b)])
    except ValueError as exc:
        if not catch:
            raise
        joined = f"caught {exc}"
    last = yield from _call(cfg, "p2")
    return [first, joined, last]


def _sync(gen: Any, cfg: llm.LLMConfig) -> tuple[Any, list[str]]:
    seen: list[str] = []

    def execute(kwargs: dict[str, Any]) -> Any:
        seen.append(_tag(kwargs))
        return _answer(kwargs)

    return llm.drive(gen, cfg, execute), seen


def _batch(units: list[Unit]) -> tuple[dict[str, Any], FakeBackend, BatchExecutor]:
    def script(request: BatchRequest) -> Any:
        if _tag(request.kwargs) == "boom":
            return BatchItemError(request.custom_id, "invalid", "refused")
        return _answer(request.kwargs)

    backend = FakeBackend(script)

    def sync(kwargs: dict[str, Any]) -> Any:
        raise RuntimeError(f"sync fallback refused {_tag(kwargs)}")

    ex = BatchExecutor(backend, sleep=lambda _s: None, min_wave_size=1, sync_execute=sync)
    return run_stage(units, ex), backend, ex


def _waves(backend: FakeBackend) -> list[list[str]]:
    return [[_tag(r.kwargs) for r in batch] for batch in backend.submitted]


def test_sync_driver_runs_children_one_after_another_in_child_order() -> None:
    cfg = _cfg()
    result, seen = _sync(_parent(cfg), cfg)
    assert seen == ["p1", "a1", "a2", "b1", "p2"]
    assert result == ["r:p1", [["r:a1", "r:a2"], ["r:b1"]], "r:p2"]


def test_sync_fanout_requests_equal_the_yield_from_sequence() -> None:
    """Byte for byte: the FanOut run makes exactly the requests of the same
    children run with ``yield from`` one after another."""
    cfg = _cfg()

    def sequential() -> llm.LLMSteps[Any]:
        yield from _call(cfg, "p1")
        yield from _child(cfg, "a1", "a2")
        yield from _child(cfg, "b1")
        yield from _call(cfg, "p2")

    def record(gen: Any) -> list[dict[str, Any]]:
        sent: list[dict[str, Any]] = []

        def execute(kwargs: dict[str, Any]) -> Any:
            sent.append(dict(kwargs))
            return _answer(kwargs)

        llm.drive(gen, cfg, execute)
        return sent

    assert record(_parent(cfg)) == record(sequential())


def test_batch_driver_puts_children_in_the_same_wave() -> None:
    cfg = _cfg()
    out, backend, ex = _batch([Unit("u", cfg, _parent(cfg)), Unit("v", cfg, _call(cfg, "v1"))])
    assert _waves(backend) == [["p1", "v1"], ["a1", "b1"], ["a2"], ["p2"]]
    assert out["u"].result == ["r:p1", [["r:a1", "r:a2"], ["r:b1"]], "r:p2"]
    assert out["u"].steps == 5 and out["u"].batch_steps == 5
    assert out["v"].result == "r:v1"
    assert ex.stats.waves == 4
    ids = [r.custom_id for batch in backend.submitted for r in batch if r.custom_id[:2] == "u0"]
    assert ids == [f"u0_u-{n}" for n in range(1, 6)]  # numbered in ready order


def test_batch_and_sync_return_the_same_result_and_requests() -> None:
    cfg = _cfg()
    sync_result, sync_seen = _sync(_parent(cfg), cfg)
    out, backend, _ = _batch([Unit("u", cfg, _parent(cfg))])
    assert out["u"].result == sync_result
    assert sorted(t for wave in _waves(backend) for t in wave) == sorted(sync_seen)


def test_child_failure_is_thrown_into_the_parent_after_every_child_ran() -> None:
    cfg = _cfg()
    result, seen = _sync(_parent(cfg, fail_b=True, catch=True), cfg)
    assert seen == ["p1", "a1", "a2", "b1", "p2"]  # a ran to the end despite b
    assert result == ["r:p1", "caught child b1 failed", "r:p2"]
    out, backend, _ = _batch([Unit("u", cfg, _parent(cfg, fail_b=True, catch=True))])
    assert out["u"].result == result
    assert _waves(backend) == [["p1"], ["a1", "b1"], ["a2"], ["p2"]]


def test_uncaught_child_failure_fails_the_unit_only() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError, match="child b1 failed"):
        _sync(_parent(cfg, fail_b=True), cfg)
    out, _, _ = _batch(
        [Unit("u", cfg, _parent(cfg, fail_b=True)), Unit("v", cfg, _call(cfg, "v1"))]
    )
    assert isinstance(out["u"].error, ValueError)
    assert out["v"].result == "r:v1"


def test_first_failure_in_child_order_wins() -> None:
    cfg = _cfg()

    def parent() -> llm.LLMFlow[str]:
        try:
            yield llm.FanOut([_child(cfg, "a1", "a2", fail=True), _child(cfg, "b1", fail=True)])
        except ValueError as exc:
            return str(exc)
        return "no failure"

    # In batch, b fails first (one request) — the parent still sees a's.
    assert _sync(parent(), cfg)[0] == "child a1 failed"
    out, _, _ = _batch([Unit("u", cfg, parent())])
    assert out["u"].result == "child a1 failed"


def test_request_failure_is_delivered_to_the_child_that_made_it() -> None:
    cfg = _cfg()

    def tolerant() -> llm.LLMSteps[str]:
        try:
            return (yield from _call(cfg, "boom"))
        except RuntimeError as exc:
            return f"child saw {exc}"

    def parent() -> llm.LLMFlow[Any]:
        return (yield llm.FanOut([tolerant(), _child(cfg, "a1")]))

    out, _, _ = _batch([Unit("u", cfg, parent())])
    assert out["u"].result == ["child saw sync fallback refused boom", ["r:a1"]]


def test_nested_fanout() -> None:
    cfg = _cfg()

    def inner() -> llm.LLMFlow[list[Any]]:
        joined = yield llm.FanOut([_child(cfg, "x1", "x2"), _child(cfg, "y1")])
        tail = yield from _call(cfg, "z1")
        return [joined, tail]

    def outer() -> llm.LLMFlow[Any]:
        return (yield llm.FanOut([inner(), _child(cfg, "a1")]))

    expected = [[[["r:x1", "r:x2"], ["r:y1"]], "r:z1"], ["r:a1"]]
    result, seen = _sync(outer(), cfg)
    assert result == expected
    assert seen == ["x1", "x2", "y1", "z1", "a1"]
    out, backend, _ = _batch([Unit("u", cfg, outer())])
    assert out["u"].result == expected
    assert _waves(backend) == [["x1", "y1", "a1"], ["x2"], ["z1"]]


def test_empty_fanout_joins_at_once() -> None:
    cfg = _cfg()

    def parent() -> llm.LLMFlow[Any]:
        joined = yield llm.FanOut([])
        return joined, (yield from _call(cfg, "p1"))

    assert _sync(parent(), cfg)[0] == ([], "r:p1")
    out, backend, _ = _batch([Unit("u", cfg, parent())])
    assert out["u"].result == ([], "r:p1") and _waves(backend) == [["p1"]]


def test_children_that_make_no_request_join_during_priming() -> None:
    cfg = _cfg()

    def nothing(value: str) -> llm.LLMSteps[str]:
        return value
        yield  # pragma: no cover - makes it a generator

    def parent() -> llm.LLMFlow[Any]:
        return (yield llm.FanOut([nothing("x"), nothing("y")]))

    assert _sync(parent(), cfg)[0] == ["x", "y"]
    out, backend, _ = _batch([Unit("u", cfg, parent())])
    assert out["u"].result == ["x", "y"] and backend.submitted == []


def test_usage_folds_in_child_order_bit_for_bit(tmp_path: Path) -> None:
    """The batch driver receives a1 and b1 in one wave and a2 in the next, yet
    the row's float sums equal the sync driver's child-order fold exactly."""
    assert (0.1 + 0.2) + 0.17 != (0.1 + 0.17) + 0.2  # the order is observable

    def rows(run: str) -> list[dict[str, Any]]:
        ws = Workspace(root=tmp_path / run)
        cfg = _cfg(ws)
        with llm.record_usage_for(cfg):
            if run == "sync":
                _sync(_parent(cfg), cfg)
            else:
                _batch([Unit("u", cfg, _parent(cfg))])
        return [
            {k: v for k, v in r.items() if k not in {"at", "duration_s", "tier"}}
            for r in read_events(ws)
        ]

    sync_rows, batch_rows = rows("sync"), rows("batch")
    assert batch_rows == sync_rows
    assert sync_rows[0]["cost_usd"] == (((0.0 + 0.1) + 0.2) + 0.17) + 0.0


def test_usage_row_without_a_sink_matches_sync(tmp_path: Path) -> None:
    def rows(run: str) -> list[dict[str, Any]]:
        ws = Workspace(root=tmp_path / run)
        cfg = _cfg(ws)
        if run == "sync":
            _sync(_parent(cfg), cfg)
        else:
            _batch([Unit("u", cfg, _parent(cfg))])
        return [
            {k: v for k, v in r.items() if k not in {"at", "duration_s", "tier"}}
            for r in read_events(ws)
        ]

    assert rows("batch") == rows("sync")


def test_interrupt_in_a_child_closes_every_strand_and_propagates() -> None:
    cfg = _cfg()
    closed: list[str] = []

    def child(tag: str, interrupt: bool) -> llm.LLMSteps[str]:
        try:
            reply = yield from _call(cfg, tag)
            if interrupt:
                raise KeyboardInterrupt
            return reply
        finally:
            closed.append(tag)

    def parent() -> llm.LLMFlow[Any]:
        try:
            return (yield llm.FanOut([child("a1", False), child("b1", True)]))
        finally:
            closed.append("parent")

    with pytest.raises(KeyboardInterrupt):
        _batch([Unit("u", cfg, parent())])
    assert sorted(closed) == ["a1", "b1", "parent"]
    closed.clear()
    with pytest.raises(KeyboardInterrupt):
        _sync(parent(), cfg)
    assert sorted(closed) == ["a1", "b1", "parent"]
