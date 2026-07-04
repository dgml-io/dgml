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

"""LLM usage / cost event log.

Every LLM-backed operation (classification, schema generation, value
extraction) appends one JSON line to ``<workspace>/usage.jsonl`` so the
workspace carries a permanent record of what was spent. Records include
the model, token counts, cost in USD (when known), wall time, the
operation outcome, and a small per-operation ``context`` blob
(``file_id``, ``docset_id``, ``tool_calls``, etc.).

The event log is append-only and deliberately permissive: write failures
here MUST NOT break the LLM-using operation that called us, and read
failures (corrupt lines, truncated tail) MUST be tolerated by readers.

Why JSONL instead of a single growing JSON array: appends are O(1) and
crash-safe; a partial line at end-of-file from a crash mid-write is
trivial to skip on read instead of breaking JSON parse for the whole
file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .storage import Workspace

# Operation identifiers — keep these stable; the UX filters by them.
OPERATION_CLASSIFY = "classify"
OPERATION_SCHEMA_GENERATE = "schema_generate"
OPERATION_EXTRACT_VALUES = "extract_values"
OPERATION_HYBRID_MERGE = "hybrid_merge"
OPERATION_STYLE_ANNOTATE = "style_annotate"
OPERATION_TRANSCRIBE = "transcribe"
OPERATION_LABEL = "label"
OPERATION_LINKS = "links"

OUTCOME_OK = "ok"
OUTCOME_ERROR = "error"


@dataclass
class UsageEvent:
    """One LLM call worth of accounting.

    ``cost_usd`` and the token counts can be ``None`` when litellm
    doesn't know the price for the model in use; the UX surfaces those
    as "unknown" rather than treating them as zero.
    """

    at: str
    operation: str
    model: str
    cost_usd: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    duration_s: float
    outcome: str
    context: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def record_usage(workspace: Workspace, event: UsageEvent) -> None:
    """Append a single usage event to ``<workspace>/usage.jsonl``.

    A write failure here is swallowed: cost telemetry must never break
    the operation it's reporting on. Worst case the row is missing from
    the log; the user's PDF is still ingested / classified / extracted.
    """
    try:
        path = workspace.usage_log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_json(), separators=(",", ":"))
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        # Intentional broad catch: never let logging take down the
        # caller. The usage log is best-effort telemetry.
        pass


def extract_cost_and_tokens(response: Any) -> dict[str, Any]:
    """Pull cost + token usage off a litellm completion response.

    litellm's normalized field is ``response._hidden_params['response_cost']``
    — populated for every model where it knows the price. Token counts
    come off the standard OpenAI-shaped ``response.usage``. Any field we
    can't read returns ``None`` (the JSONL row carries the null forward
    rather than fabricating a zero).
    """
    out: dict[str, Any] = {
        "cost_usd": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        cost = hidden.get("response_cost")
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            out["cost_usd"] = float(cost)
    usage = getattr(response, "usage", None)
    if usage is not None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = getattr(usage, name, None)
            if isinstance(val, int) and not isinstance(val, bool):
                out[name] = val
    return out


def add_partial(acc: dict[str, Any], inc: dict[str, Any]) -> None:
    """Sum cost + token counters across multiple litellm calls.

    ``None`` is treated as "unknown, contributing zero" so a partial
    set of priced calls still produces a meaningful total. The
    accumulator's value stays ``None`` only if every contribution is
    ``None`` for that field.
    """
    for k in ("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens"):
        a = acc.get(k)
        b = inc.get(k)
        if a is None and b is None:
            continue
        acc[k] = (a or 0) + (b or 0)


def read_events(workspace: Workspace) -> list[dict[str, Any]]:
    """Read all events from ``usage.jsonl``. Tolerates corrupt lines
    (skips them silently) and a missing file (returns ``[]``)."""
    path = workspace.usage_log_path
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Tail line from a crashed-mid-write append — skip it.
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out
