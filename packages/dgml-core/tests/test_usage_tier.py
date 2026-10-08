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

"""The ``tier`` field on usage rows.

Every row says which pricing tier served it. The synchronous path always
writes ``"standard"``; that is the only change it sees. A scope whose
responses came from more than one tier writes one row per tier
(``context.tier_split``), so the money is attributed to where it was spent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dgml_core import layout, llm
from dgml_core.grounded import _merge_totals
from dgml_core.storage import Workspace
from dgml_core.usage import (
    TIER_BATCH,
    TIER_MARKER,
    TIER_STANDARD,
    TIERS_KEY,
    UsageEvent,
    add_partial,
    public_totals,
    read_events,
    record_usage,
    scope_events,
    with_tier,
)

from .conftest import FakeLLMResponse, local_tree_path

MODEL = "anthropic/claude-haiku-4-5"

ROW_KEYS = [
    "at",
    "operation",
    "model",
    "cost_usd",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "duration_s",
    "outcome",
    "context",
    "error",
    "cache_read_tokens",
    "cache_creation_tokens",
    "tier",
]


def _event(**overrides: Any) -> UsageEvent:
    base: dict[str, Any] = dict(
        at="2026-01-01T00:00:00Z",
        operation="transcribe",
        model=MODEL,
        cost_usd=0.03,
        prompt_tokens=30,
        completion_tokens=3,
        total_tokens=33,
        duration_s=2.0,
        outcome="ok",
    )
    base.update(overrides)
    return UsageEvent(**base)


def _cfg(ws: Workspace, **kw: Any) -> llm.LLMConfig:
    kw.setdefault("operation", "transcribe")
    return llm.LLMConfig(model=MODEL, workspace=ws, debug=True, **kw)


def _marked(text: str, tier: str, cost: float) -> FakeLLMResponse:
    response = FakeLLMResponse(text, cost=cost, prompt_tokens=10, completion_tokens=1)
    response._hidden_params[TIER_MARKER] = tier
    return response


def _drive_one(cfg: llm.LLMConfig, reply: Any) -> None:
    llm.drive(
        llm.steps_call(cfg, system_prompt="S", user_content=[{"type": "text", "text": "hi"}]),
        cfg,
        lambda _step: reply,
    )


# ---- the field ------------------------------------------------------------------


def test_tier_is_additive_and_defaults_to_standard() -> None:
    row = _event().to_json()
    assert list(row) == ROW_KEYS  # every old key, same order, tier last
    assert row["tier"] == TIER_STANDARD == "standard"


def test_rows_without_tier_still_read(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    path = local_tree_path(ws, layout.USAGE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = _event(at="t1").to_json()
    del old["tier"]
    path.write_text(json.dumps(old) + "\n", encoding="utf-8")
    record_usage(ws, _event(at="t2"))
    rows = read_events(ws)
    assert "tier" not in rows[0]  # not backfilled
    assert rows[1]["tier"] == "standard"


# ---- the synchronous path -----------------------------------------------------------


def test_sync_rows_are_standard_per_call_and_per_scope(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws, context={"k": 1})
    _drive_one(cfg, FakeLLMResponse("x", cost=0.01))
    with llm.record_usage_for(cfg):
        _drive_one(cfg, FakeLLMResponse("y", cost=0.02))
        _drive_one(cfg, FakeLLMResponse("z", cost=0.03))
    rows = read_events(ws)
    assert [(r["tier"], r["context"]) for r in rows] == [
        ("standard", {"k": 1}),
        ("standard", {"k": 1}),
    ]
    assert rows[1]["cost_usd"] == 0.05
    assert all(list(r) == ROW_KEYS for r in rows)


def test_config_tier_reaches_the_row(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws, tier=TIER_BATCH)
    _drive_one(cfg, FakeLLMResponse("x", cost=0.01))
    assert read_events(ws)[0]["tier"] == "batch"


# ---- a scope that spans tiers -------------------------------------------------------


def test_scope_spanning_tiers_writes_one_row_per_tier(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws, operation="label", context={"docset_id": "ds1"})
    with llm.record_usage_for(cfg):
        _drive_one(cfg, _marked("a", TIER_BATCH, 0.01))
        _drive_one(cfg, _marked("b", TIER_STANDARD, 0.25))
        _drive_one(cfg, _marked("c", TIER_BATCH, 0.02))
    rows = read_events(ws)
    assert [(r["tier"], r["cost_usd"], r["prompt_tokens"]) for r in rows] == [
        ("batch", 0.03, 20),
        ("standard", 0.25, 10),
    ]
    assert all(r["context"] == {"docset_id": "ds1", "tier_split": True} for r in rows)
    assert rows[0]["at"] == rows[1]["at"]


def test_split_puts_duration_on_the_first_part_and_error_on_all() -> None:
    totals: dict[str, Any] = {}
    add_partial(totals, with_tier({"cost_usd": 0.5, "prompt_tokens": 5}, None, TIER_STANDARD))
    add_partial(totals, with_tier({"cost_usd": 0.25, "prompt_tokens": 5}, None, TIER_BATCH))
    event = _event(cost_usd=0.75, prompt_tokens=10, outcome="error", error="E: boom")
    parts = scope_events(event, totals)
    assert [(p.tier, p.cost_usd, p.duration_s) for p in parts] == [
        ("batch", 0.25, 2.0),
        ("standard", 0.5, 0.0),
    ]
    assert all(p.outcome == "error" and p.error == "E: boom" for p in parts)


def test_untiered_responses_take_the_rows_tier() -> None:
    totals: dict[str, Any] = {}
    add_partial(totals, with_tier({"cost_usd": 0.5}, None, None))
    event = _event(tier=TIER_BATCH)
    assert scope_events(event, totals) == [event]
    assert event.tier == "batch"


# ---- extraction keeps its stats file clean ----------------------------------------------


def test_extraction_totals_merge_tiers_and_strip_them_for_stats() -> None:
    phase1: dict[str, Any] = {}
    phase3: dict[str, Any] = {}
    add_partial(phase1, with_tier({"cost_usd": 0.5, "prompt_tokens": 5}, None, TIER_BATCH))
    add_partial(phase3, with_tier({"cost_usd": 0.25, "prompt_tokens": 1}, None, TIER_STANDARD))
    merged = _merge_totals(
        {
            **dict.fromkeys(("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens")),
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            **phase1,
        },
        {
            **dict.fromkeys(("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens")),
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            **phase3,
        },
    )
    assert merged["cost_usd"] == 0.75
    assert sorted(merged[TIERS_KEY]) == ["batch", "standard"]
    assert TIERS_KEY not in public_totals(merged)
