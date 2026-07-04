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

from __future__ import annotations

import json
from types import SimpleNamespace

from dgml_core.storage import Workspace
from dgml_core.usage import (
    UsageEvent,
    add_partial,
    extract_cost_and_tokens,
    read_events,
    record_usage,
)

# ---------------------------------------------------------------------------
# record_usage + read_events
# ---------------------------------------------------------------------------


def test_record_usage_appends_and_read_back(workspace: Workspace) -> None:
    record_usage(
        workspace,
        UsageEvent(
            at="2026-05-15T17:00:00Z",
            operation="classify",
            model="gemini/flash-lite",
            cost_usd=0.0001,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            duration_s=0.5,
            outcome="ok",
            context={"file_id": "abc"},
        ),
    )
    record_usage(
        workspace,
        UsageEvent(
            at="2026-05-15T17:01:00Z",
            operation="schema_generate",
            model="anthropic/opus",
            cost_usd=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            duration_s=2.1,
            outcome="error",
            context={"from_file_ids": ["a", "b"]},
            error="ValueError: nope",
        ),
    )
    events = read_events(workspace)
    assert len(events) == 2
    assert events[0]["operation"] == "classify"
    assert events[0]["cost_usd"] == 0.0001
    assert events[1]["outcome"] == "error"
    assert events[1]["error"] == "ValueError: nope"
    assert events[1]["context"] == {"from_file_ids": ["a", "b"]}


def test_read_events_returns_empty_when_missing(workspace: Workspace) -> None:
    assert read_events(workspace) == []


def test_read_events_tolerates_corrupt_lines(workspace: Workspace) -> None:
    """A truncated tail from a crash-mid-append must not break readers
    for the entire file."""
    path = workspace.usage_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                json.dumps({"at": "t1", "operation": "classify", "model": "m"}),
                "{ this is not json",  # corrupt mid-write line
                "",  # blank
                json.dumps({"at": "t2", "operation": "extract_values", "model": "m"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    events = read_events(workspace)
    assert [e["at"] for e in events] == ["t1", "t2"]


def test_record_usage_swallows_write_errors(workspace: Workspace) -> None:
    """Cost telemetry must never break the operation it's reporting on.

    Simulate a write failure by pointing the workspace root at a path
    that isn't writable (a regular file blocking the directory).
    """
    # Replace usage_log_path's parent with a file so mkdir/open fails.
    blocker = workspace.root / "blocker"
    blocker.write_text("not a dir", encoding="utf-8")
    bad_ws = Workspace(root=blocker / "usage.jsonl")
    record_usage(  # Must not raise.
        bad_ws,
        UsageEvent(
            at="t",
            operation="classify",
            model="m",
            cost_usd=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            duration_s=0.0,
            outcome="ok",
        ),
    )


# ---------------------------------------------------------------------------
# extract_cost_and_tokens
# ---------------------------------------------------------------------------


def test_extract_cost_from_hidden_params() -> None:
    response = SimpleNamespace(
        _hidden_params={"response_cost": 0.00345},
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15),
    )
    out = extract_cost_and_tokens(response)
    assert out == {
        "cost_usd": 0.00345,
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }


def test_extract_cost_handles_missing_cost() -> None:
    response = SimpleNamespace(
        _hidden_params={},
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    out = extract_cost_and_tokens(response)
    assert out["cost_usd"] is None
    assert out["prompt_tokens"] == 1


def test_extract_cost_handles_missing_usage() -> None:
    response = SimpleNamespace(_hidden_params={"response_cost": 0.1})
    out = extract_cost_and_tokens(response)
    assert out["cost_usd"] == 0.1
    assert out["prompt_tokens"] is None
    assert out["completion_tokens"] is None
    assert out["total_tokens"] is None


def test_extract_cost_rejects_bool_as_int() -> None:
    """Python's bool is a subclass of int — make sure we don't accept
    True as a token count or cost."""
    response = SimpleNamespace(
        _hidden_params={"response_cost": True},
        usage=SimpleNamespace(prompt_tokens=False, completion_tokens=0, total_tokens=0),
    )
    out = extract_cost_and_tokens(response)
    assert out["cost_usd"] is None
    assert out["prompt_tokens"] is None
    assert out["completion_tokens"] == 0  # the literal 0, not a bool


# ---------------------------------------------------------------------------
# add_partial
# ---------------------------------------------------------------------------


def test_add_partial_treats_none_as_zero_when_other_is_set() -> None:
    acc = {
        "cost_usd": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    add_partial(
        acc, {"cost_usd": 0.5, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    )
    add_partial(
        acc, {"cost_usd": 0.5, "prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
    )
    assert acc == {
        "cost_usd": 1.0,
        "prompt_tokens": 30,
        "completion_tokens": 10,
        "total_tokens": 40,
    }


def test_add_partial_leaves_none_when_all_contributions_none() -> None:
    acc = {
        "cost_usd": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    add_partial(
        acc,
        {"cost_usd": None, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    )
    assert acc["cost_usd"] is None
    assert acc["prompt_tokens"] is None
