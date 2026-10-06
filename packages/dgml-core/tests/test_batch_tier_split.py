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

"""A usage scope whose responses span pricing tiers writes one row per tier.

A scope (one ``_record_call``, one ``record_usage_for`` block, one file's
extraction) used to write a single row labeled with one ``tier`` even when its
responses were served partly by a batch and partly by the synchronous tier
(a synchronous fallback for an item the batch could not serve). Such a scope
writes one row per tier, each with that tier's own tokens and cost, the same
``operation``, ``model``, ``outcome`` and ``error``, and ``context`` plus
``"tier_split": true``. The scope's wall time goes on the first part only, so
summing any numeric field over the parts gives the scope's total. A scope
served by one tier writes exactly the row it always did.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from dgml_core import llm, usage
from dgml_core.batch import (
    TIER_MARKER,
    BatchExecutor,
    BatchItemError,
    BatchRequest,
    FakeBackend,
    Unit,
    run_stage,
)
from dgml_core.storage import Workspace
from dgml_core.usage import BILLED_MARKER, TIER_BATCH, TIER_STANDARD, read_events

from .conftest import FakeLLMResponse

MODEL = "anthropic/claude-haiku-4-5"


def _cfg(ws: Workspace, **kw: Any) -> llm.LLMConfig:
    kw.setdefault("operation", "transcribe")
    return llm.LLMConfig(model=MODEL, workspace=ws, debug=True, **kw)


def _text(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": content}]


def _key(custom_id: str) -> str:
    prefix, n = custom_id.rsplit("-", 1)
    return f"{prefix.split('_', 1)[1]}#{n}"


def _executor(table: dict[str, Any], sync: dict[str, Any] | None = None) -> BatchExecutor:
    """A batch executor scripted per ``<unit>#<step>``; a step scripted as
    ``"fallback"`` fails at batch level and is served synchronously from *sync*
    (keyed by the request's first user text)."""

    def script(req: BatchRequest) -> Any:
        answer = table[_key(req.custom_id)]
        if answer == "fallback":
            return BatchItemError(req.custom_id, "invalid", "rejected")
        return answer

    def sync_execute(kwargs: dict[str, Any]) -> Any:
        user = next(m for m in kwargs["messages"] if m["role"] == "user")
        return (sync or {})[user["content"][0]["text"]]

    return BatchExecutor(
        FakeBackend(script),
        min_wave_size=1,
        max_item_retries=0,
        sleep=lambda _s: None,
        sync_execute=sync_execute,
    )


def _two_calls(cfg: llm.LLMConfig, name: str, *, fail: bool = False) -> Unit:
    """One unit making two ordinary calls (texts ``<name>1`` / ``<name>2``),
    optionally failing after the second response."""

    def gen() -> Generator[dict[str, Any], Any, str]:
        first = yield from llm.steps_call(cfg, system_prompt="S", user_content=_text(f"{name}1"))
        second = yield from llm.steps_call(cfg, system_prompt="S", user_content=_text(f"{name}2"))
        if fail:
            raise ValueError("unparseable")
        return first + second

    return Unit(name, cfg, gen())


_FIELDS = (
    "cost_usd",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
)


def _sums(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[k] for k in _FIELDS)


# ---- batch driver: one unit ------------------------------------------------------


def test_unit_spanning_tiers_writes_one_row_per_tier(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws, context={"doc": "a.pdf"})
    ex = _executor(
        {
            "a#1": FakeLLMResponse(
                "x", cost=0.01, prompt_tokens=100, completion_tokens=10, cache_read_tokens=3
            ),
            "a#2": "fallback",
        },
        sync={"a2": FakeLLMResponse("y", cost=0.04, prompt_tokens=200, completion_tokens=20)},
    )
    out = run_stage([_two_calls(cfg, "a")], ex)
    assert out["a"].result == "xy"
    assert out["a"].batch_steps == 1 and out["a"].sync_steps == 1

    rows = read_events(ws)
    assert [r["tier"] for r in rows] == [TIER_BATCH, TIER_STANDARD]
    batch_row, std_row = rows
    assert _sums(batch_row) == (0.01, 100, 10, 110, 3, 0)
    assert _sums(std_row) == (0.04, 200, 20, 220, 0, 0)
    for row in rows:
        assert row["operation"] == "transcribe" and row["model"] == MODEL
        assert row["outcome"] == "ok" and row["error"] is None
        assert row["context"] == {"doc": "a.pdf", "tier_split": True}
    # Wall time is the scope's, carried once so it sums correctly.
    assert std_row["duration_s"] == 0.0
    assert batch_row["at"] == std_row["at"]


def test_single_tier_unit_writes_the_row_it_always_did(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws, context={"doc": "a.pdf"})
    ex = _executor(
        {
            "a#1": FakeLLMResponse("x", cost=0.01, prompt_tokens=100, completion_tokens=10),
            "a#2": FakeLLMResponse("y", cost=0.04, prompt_tokens=200, completion_tokens=20),
        }
    )
    run_stage([_two_calls(cfg, "a")], ex)
    rows = read_events(ws)
    assert len(rows) == 1
    assert rows[0]["tier"] == TIER_BATCH
    assert rows[0]["context"] == {"doc": "a.pdf"}  # no tier_split marker
    assert rows[0]["cost_usd"] == 0.01 + 0.04
    assert rows[0]["prompt_tokens"] == 300


def test_unit_served_wholly_by_the_fallback_is_a_standard_row(tmp_path: Path) -> None:
    """One tier, but not the stage's: the row says what served (and priced) it."""
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)
    ex = _executor(
        {"a#1": "fallback", "a#2": "fallback"},
        sync={
            "a1": FakeLLMResponse("x", cost=0.02, prompt_tokens=1),
            "a2": FakeLLMResponse("y", cost=0.03, prompt_tokens=2),
        },
    )
    run_stage([_two_calls(cfg, "a")], ex)
    rows = read_events(ws)
    assert len(rows) == 1
    assert rows[0]["tier"] == TIER_STANDARD
    assert rows[0]["context"] == {}
    assert rows[0]["cost_usd"] == 0.02 + 0.03


def test_error_mid_scope_marks_every_part_with_the_error(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)
    ex = _executor(
        {"a#1": FakeLLMResponse("x", cost=0.01, prompt_tokens=1), "a#2": "fallback"},
        sync={"a2": FakeLLMResponse("y", cost=0.04, prompt_tokens=4)},
    )
    out = run_stage([_two_calls(cfg, "a", fail=True)], ex)
    assert isinstance(out["a"].error, ValueError)

    rows = read_events(ws)
    assert [r["tier"] for r in rows] == [TIER_BATCH, TIER_STANDARD]
    assert [r["cost_usd"] for r in rows] == [0.01, 0.04]
    assert all(r["outcome"] == "error" for r in rows)
    assert all(r["error"] == "ValueError: unparseable" for r in rows)
    assert all(r["context"] == {"tier_split": True} for r in rows)


# ---- record_usage_for scope ------------------------------------------------------


def test_enclosing_scope_spanning_tiers_splits(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws, operation="label", context={"docset_id": "ds1"})
    ex = _executor(
        {
            "a#1": FakeLLMResponse("x", cost=0.01, prompt_tokens=10, completion_tokens=1),
            "a#2": FakeLLMResponse("y", cost=0.02, prompt_tokens=20, completion_tokens=2),
            "b#1": "fallback",
            "b#2": FakeLLMResponse("w", cost=0.03, prompt_tokens=30, completion_tokens=3),
        },
        sync={"b1": FakeLLMResponse("z", cost=0.5, prompt_tokens=500, completion_tokens=50)},
    )
    with llm.record_usage_for(cfg):
        run_stage([_two_calls(cfg, "a"), _two_calls(cfg, "b")], ex)

    rows = read_events(ws)
    assert [r["tier"] for r in rows] == [TIER_BATCH, TIER_STANDARD]
    batch_row, std_row = rows
    assert batch_row["cost_usd"] == pytest.approx(0.06)
    assert batch_row["prompt_tokens"] == 60 and batch_row["completion_tokens"] == 6
    assert _sums(std_row) == (0.5, 500, 50, 550, 0, 0)
    assert all(r["operation"] == "label" for r in rows)
    assert all(r["context"] == {"docset_id": "ds1", "tier_split": True} for r in rows)


def _pooled_run(tmp_path: Path, *, sync_workers: int) -> tuple[list[dict[str, Any]], Any]:
    """Four units whose second call each falls back to the synchronous tier, in
    one wave. With a pool, the fake synchronous calls finish in *reverse* wave
    order (each waits for the next unit's to finish first)."""
    ws = Workspace(root=tmp_path)
    names = ["a", "b", "c", "d"]
    sync_costs = {"a": 0.1, "b": 0.2, "c": 0.7, "d": 0.11}
    done = {n: threading.Event() for n in names}
    finished: list[str] = []

    def script(req: BatchRequest) -> Any:
        unit, step = _key(req.custom_id).split("#")
        if step == "2":
            return BatchItemError(req.custom_id, "invalid", "rejected")
        return FakeLLMResponse("x", cost=0.1 if unit == "a" else 0.0, prompt_tokens=1)

    def sync_execute(kwargs: dict[str, Any]) -> Any:
        user = next(m for m in kwargs["messages"] if m["role"] == "user")
        unit = user["content"][0]["text"][:-1]
        later = names.index(unit) + 1
        if sync_workers > 1 and later < len(names):
            assert done[names[later]].wait(timeout=10)
        finished.append(unit)
        done[unit].set()
        return FakeLLMResponse("y", cost=sync_costs[unit], prompt_tokens=10, completion_tokens=2)

    ex = BatchExecutor(
        FakeBackend(script),
        min_wave_size=1,
        max_item_retries=0,
        sleep=lambda _s: None,
        sync_execute=sync_execute,
        sync_workers=sync_workers,
    )
    cfg = _cfg(ws)
    run_stage([_two_calls(cfg, n) for n in names], ex)
    assert finished == (names[::-1] if sync_workers > 1 else names)
    rows = [{k: v for k, v in r.items() if k not in ("at", "duration_s")} for r in read_events(ws)]
    return rows, ex.stats


def test_pooled_fallbacks_fold_identically_whatever_finishes_first(tmp_path: Path) -> None:
    """A wave's synchronous fallbacks run on a pool, but what they cost is
    folded in wave order: the executor's float totals and every per-tier usage
    row are bit-identical to a serial run's, though the pool finished them in
    reverse order. (The costs are chosen so that adding them in completion
    order would change the last bit.)"""
    forward = 0.1 + 0.1 + 0.2 + 0.7 + 0.11
    reverse = 0.1 + 0.11 + 0.7 + 0.2 + 0.1
    assert forward != reverse  # the test would prove nothing otherwise

    pooled_rows, pooled = _pooled_run(tmp_path / "pool", sync_workers=4)
    serial_rows, serial = _pooled_run(tmp_path / "serial", sync_workers=1)

    assert pooled.cost_usd.hex() == serial.cost_usd.hex() == forward.hex()
    assert pooled.standard_cost_usd.hex() == serial.standard_cost_usd.hex()
    assert pooled.sync_fallbacks == serial.sync_fallbacks == 4
    assert pooled_rows == serial_rows
    # Unit a spans both tiers (a batch part, a standard part); b-d were served
    # by the batch at no cost and the fallback, still one part per tier.
    assert [(r["tier"], r["cost_usd"]) for r in pooled_rows][:2] == [
        (TIER_BATCH, 0.1),
        (TIER_STANDARD, 0.1),
    ]


# ---- job mode: billed replays and buffered rows ----------------------------------


def test_billed_replays_do_not_add_a_tier(tmp_path: Path) -> None:
    """A replayed response an earlier run already billed contributes zero and
    no tier: a scope of billed batch replies plus fresh standard ones is one
    standard row, not a zero batch part."""
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)
    billed = FakeLLMResponse("x", cost=0.01, prompt_tokens=1)
    billed._hidden_params.update({TIER_MARKER: TIER_BATCH, BILLED_MARKER: True})
    fresh = FakeLLMResponse("y", cost=0.02, prompt_tokens=2)
    fresh._hidden_params[TIER_MARKER] = TIER_STANDARD
    replies = iter([billed, fresh])
    with llm.record_usage_for(cfg):
        for text in ("a", "b"):
            llm.drive(
                llm.steps_call(cfg, system_prompt="S", user_content=_text(text)),
                cfg,
                lambda _s: next(replies),
            )
    rows = read_events(ws)
    assert len(rows) == 1
    assert rows[0]["tier"] == TIER_STANDARD and rows[0]["context"] == {}
    assert rows[0]["cost_usd"] == 0.02


def test_buffered_rows_hold_every_part_until_flushed(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)
    ex = _executor(
        {"a#1": FakeLLMResponse("x", cost=0.01), "a#2": "fallback"},
        sync={"a2": FakeLLMResponse("y", cost=0.04)},
    )
    with usage.buffered_usage() as held:
        run_stage([_two_calls(cfg, "a")], ex)
        assert read_events(ws) == []
        assert [event.tier for _ws, event in held] == [TIER_BATCH, TIER_STANDARD]
        usage.flush_usage(held)
    rows = read_events(ws)
    assert [(r["tier"], r["cost_usd"]) for r in rows] == [
        (TIER_BATCH, 0.01),
        (TIER_STANDARD, 0.04),
    ]
