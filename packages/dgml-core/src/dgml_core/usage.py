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

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from . import layout
from .storage import Workspace

# Operation identifiers — keep these stable; the UX filters by them.
OPERATION_CLASSIFY = "classify"
OPERATION_CLUSTER = "cluster"
OPERATION_SCHEMA_GENERATE = "schema_generate"
OPERATION_EXTRACT_VALUES = "extract_values"
OPERATION_HYBRID_MERGE = "hybrid_merge"
OPERATION_STYLE_ANNOTATE = "style_annotate"
OPERATION_TRANSCRIBE = "transcribe"
OPERATION_LABEL = "label"
OPERATION_LINKS = "links"

OUTCOME_OK = "ok"
OUTCOME_ERROR = "error"

# Pricing tier the call was billed at. ``standard`` is the synchronous
# Messages/Chat API; ``batch`` is a provider's asynchronous batch tier
# (typically 50% of standard). Rows carry it so a run with and without
# batching can be compared from ``usage.jsonl`` alone.
TIER_STANDARD = "standard"
TIER_BATCH = "batch"


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
    # Anthropic prompt-cache accounting. Default 0 rather than None so rows
    # written before these fields existed — and any UsageEvent built without
    # them — stay valid, and so the values sum across calls. Populated by
    # ``extract_cost_and_tokens``.
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # Pricing tier (``TIER_STANDARD`` / ``TIER_BATCH``). Defaults to standard so
    # rows written before the field existed — and events built without it —
    # stay valid; readers do not backfill a missing key.
    tier: str = TIER_STANDARD

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def record_usage(workspace: Workspace, event: UsageEvent) -> None:
    """Append a single usage event to the workspace's ``usage`` log.

    Routed through the storage backend's append-only ``usage`` collection (the
    local store keeps it as ``<workspace>/usage.jsonl``, one JSON object per
    line). A write failure here is swallowed: cost telemetry must never break
    the operation it's reporting on. Worst case the row is missing from the log;
    the user's PDF is still ingested / classified / extracted.
    """
    try:
        workspace.docs.append_doc(layout.Collection.USAGE, event.to_json())
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

    Anthropic prompt-cache counters are surfaced by litellm on that same
    normalized ``response.usage`` object as ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens``; we mirror them as ``cache_read_tokens``
    and ``cache_creation_tokens``. Unlike the cost/token fields these default
    to ``0`` rather than ``None``: a provider reporting no cache activity
    genuinely used zero cache, and the values are summed across calls. A
    fallback also checks ``_hidden_params`` in case a litellm version
    relocates them there. OpenAI and Gemini responses carry their cache hits
    only as ``usage.prompt_tokens_details.cached_tokens``, which fills
    ``cache_read_tokens`` when no explicit ``cache_read_input_tokens`` exists.
    """
    out: dict[str, Any] = {
        "cost_usd": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    # litellm normalizes the Anthropic counters onto ``usage`` under these
    # names; map each source name to the field we persist.
    cache_fields = (
        ("cache_read_input_tokens", "cache_read_tokens"),
        ("cache_creation_input_tokens", "cache_creation_tokens"),
    )
    explicit_read = False
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        cost = hidden.get("response_cost")
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            out["cost_usd"] = float(cost)
        # Fallback: some litellm paths stash the cache counters in
        # ``_hidden_params``. ``usage`` (read below) takes precedence.
        for src, dst in cache_fields:
            val = hidden.get(src)
            if isinstance(val, int) and not isinstance(val, bool):
                out[dst] = val
                explicit_read = explicit_read or dst == "cache_read_tokens"
    usage = getattr(response, "usage", None)
    if usage is not None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            val = getattr(usage, name, None)
            if isinstance(val, int) and not isinstance(val, bool):
                out[name] = val
        # Canonical location for the Anthropic prompt-cache counters.
        for src, dst in cache_fields:
            val = getattr(usage, src, None)
            if isinstance(val, int) and not isinstance(val, bool):
                out[dst] = val
                explicit_read = explicit_read or dst == "cache_read_tokens"
        # OpenAI and Gemini report cache hits only here; litellm does not copy
        # them to ``cache_read_input_tokens``. An explicit counter wins.
        if not explicit_read:
            details = getattr(usage, "prompt_tokens_details", None)
            val = getattr(details, "cached_tokens", None)
            if isinstance(val, int) and not isinstance(val, bool):
                out["cache_read_tokens"] = val
    return out


_USAGE_FIELDS = (
    "cost_usd",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
)


def add_partial(acc: dict[str, Any], inc: dict[str, Any]) -> None:
    """Sum cost + token counters across multiple litellm calls.

    ``None`` is treated as "unknown, contributing zero" so a partial
    set of priced calls still produces a meaningful total. The
    accumulator's value stays ``None`` only if every contribution is
    ``None`` for that field.

    A per-tier breakdown on *inc* (:data:`TIERS_KEY`, see :func:`with_tier`)
    is merged into *acc*'s, tier by tier, alongside — never instead of — the
    aggregate sums, so the aggregate's float association is unchanged.
    """
    for k in _USAGE_FIELDS:
        a = acc.get(k)
        b = inc.get(k)
        if a is None and b is None:
            continue
        acc[k] = (a or 0) + (b or 0)
    inc_tiers = inc.get(TIERS_KEY)
    if inc_tiers:
        acc_tiers = acc.setdefault(TIERS_KEY, {})
        for tier, sub in inc_tiers.items():
            # A fresh part starts as an empty totals dict (None costs, 0 cache
            # counters), so its None semantics match the aggregate's.
            part = acc_tiers.setdefault(
                tier, dict.fromkeys(_USAGE_FIELDS[:4]) | dict.fromkeys(_USAGE_FIELDS[4:], 0)
            )
            add_partial(part, sub)


# ---- per-tier split ------------------------------------------------------------
#
# A usage scope (one call, one ``record_usage_for`` block, one file's
# extraction) folds many responses into one totals dict. When those responses
# were served by different pricing tiers — a batch plus synchronous fallbacks
# for the items it could not serve — one row labeled with one tier would
# misstate where the money went. So each folded response also records its
# usage under its own tier in a private breakdown on the totals dict, and the
# scope's row is written by :func:`scope_events`: one row when the scope saw
# one tier (exactly the row it always wrote), one row per tier otherwise.

#: Private key on a totals dict: ``{tier: subtotal}`` for every response folded
#: in that named its tier (:data:`TIER_MARKER` on its ``_hidden_params``) or was
#: given one by its driver. The key never reaches a usage row or a stats file.
TIERS_KEY = "_tiers"

#: Per-response marker (``_hidden_params["dgml_tier"]``) naming the tier that
#: served it. A batch executor stamps it; a plain synchronous response has none.
TIER_MARKER = "dgml_tier"

#: ``context`` key set to ``True`` on every row of a scope split by tier.
TIER_SPLIT_CONTEXT_KEY = "tier_split"


def with_tier(usage: dict[str, Any], response: Any, default: str | None) -> dict[str, Any]:
    """*usage* (from :func:`extract_cost_and_tokens` on *response*) with a
    one-tier breakdown attached, for :func:`add_partial` to carry upward.

    The tier is the response's :data:`TIER_MARKER`, else *default* (the
    driving config's tier); ``None`` leaves the response untiered (resolved to
    the row's own tier when the row is written). *usage* itself is not
    modified.
    """
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        marker = hidden.get(TIER_MARKER)
        if isinstance(marker, str) and marker:
            default = marker
    tier = default or ""
    out = dict(usage)
    out[TIERS_KEY] = {tier: {k: usage.get(k) for k in _USAGE_FIELDS}}
    return out


def public_totals(totals: dict[str, Any]) -> dict[str, Any]:
    """*totals* without its private keys (the per-tier breakdown): what may be
    spread into a persisted record."""
    return {k: v for k, v in totals.items() if not k.startswith("_")}


def scope_events(event: UsageEvent, totals: dict[str, Any]) -> list[UsageEvent]:
    """The row(s) a scope writes: *event* (built from the scope's aggregate
    *totals*, ``tier`` set to the scope's own label) split by tier.

    - The breakdown names at most one tier (untiered responses count as
      *event*'s tier): one row, *event* itself — its sums are the aggregate,
      exactly as before — with ``tier`` the one that served the scope.
    - Several tiers: one row per tier, in tier-name order, each with that
      tier's sums and every other field of *event*, ``context`` plus
      ``"tier_split": true``. The scope's ``duration_s`` goes on the first
      part only (the rest carry ``0.0``), so summing any numeric field over
      the parts gives the scope's total. ``outcome`` and ``error`` are the
      scope's, on every part.
    """
    breakdown: dict[str, dict[str, Any]] = {}
    for tier, sub in (totals.get(TIERS_KEY) or {}).items():
        resolved = tier or event.tier
        if resolved in breakdown:
            add_partial(breakdown[resolved], sub)
        else:
            breakdown[resolved] = dict(sub)
    if len(breakdown) <= 1:
        if breakdown:
            event.tier = next(iter(breakdown))
        return [event]
    events: list[UsageEvent] = []
    for i, tier in enumerate(sorted(breakdown)):
        sub = breakdown[tier]
        events.append(
            replace(
                event,
                tier=tier,
                duration_s=event.duration_s if i == 0 else 0.0,
                context={**event.context, TIER_SPLIT_CONTEXT_KEY: True},
                cost_usd=sub.get("cost_usd"),
                prompt_tokens=sub.get("prompt_tokens"),
                completion_tokens=sub.get("completion_tokens"),
                total_tokens=sub.get("total_tokens"),
                cache_read_tokens=sub.get("cache_read_tokens") or 0,
                cache_creation_tokens=sub.get("cache_creation_tokens") or 0,
            )
        )
    return events


def read_events(workspace: Workspace) -> list[dict[str, Any]]:
    """Read all events from the workspace's ``usage`` log. Tolerates corrupt
    lines (skips them silently) and a missing log (returns ``[]``)."""
    return workspace.docs.find_docs(layout.Collection.USAGE, {})
