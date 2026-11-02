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

"""Vision-LLM adjudicator for clustering consolidation.

:meth:`clustering.scenarios.base.Scenario.consolidate` selects the
low-confidence tail and applies verdicts, but stays LLM-free. This module
supplies the missing half: a concrete
:class:`~clustering.consolidation.Adjudicator` that asks the configured vision
model to reconsider each borderline document against its nearest candidate
clusters — the same assign-vs-create question :mod:`dgml_core.classification`
already poses, but pre-seeded with the embedding-derived candidates so the model
adjudicates rather than searches blind.

Two modes:

- **reassign** (default): one constrained tool call per document — *"which of
  these candidate types does this belong to, or is it novel?"* Run twice with
  the candidate order rotated; the reported confidence is the model's
  self-report scaled by the two-attempt agreement, an ordinal signal.
- **repartition**: one batch grouping call over the whole selected subset (a
  contested region), mapping each document to the existing type its group
  matched or a novel bucket.

Everything is soft-fail: a per-document LLM error drops just that document's
verdict (its assignment is left untouched), and a total failure is caught by
the framework's :func:`clustering.consolidation.consolidate` guard.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from typing import TYPE_CHECKING, Any

from clustering.config.schema import ConsolidationMode
from clustering.consolidation import AdjudicationRequest, AdjudicationVerdict

from .classification import ClassificationConfig, _resolve_api_key
from .llm import CallResult, LLMConfig, call_with_tools
from .usage import UsageOperation
from .utils import MANY_IMAGE_MAX_EDGE, image_to_data_url

if TYPE_CHECKING:
    from clustering.data.datasets import DocumentDataset

_TOOL_ADJUDICATE = "adjudicate"
_TOOL_REGROUP = "regroup_documents"
# Sentinel choice meaning "none of the candidates — a genuinely new type".
_NOVEL = "__novel__"
_TEXT_SNIPPET_CHARS = 800


class LLMAdjudicator:
    """A litellm-backed :class:`~clustering.consolidation.Adjudicator`.

    Construct with the workspace :class:`ClassificationConfig` (model +
    api-key precedence reused verbatim) and call it with the dataset plus the
    framework's :class:`~clustering.consolidation.AdjudicationRequest` list.
    """

    def __init__(
        self,
        config: ClassificationConfig,
        *,
        attempts: int = 2,
        record_usage: bool = False,
    ) -> None:
        self.config = config
        self.attempts = max(1, attempts)
        self.record_usage = record_usage

    # ── Adjudicator protocol ─────────────────────────────────────────────
    def __call__(
        self,
        dataset: DocumentDataset,
        requests: list[AdjudicationRequest],
        *,
        mode: ConsolidationMode,
        batch_size: int,
    ) -> dict[str, AdjudicationVerdict]:
        if not requests:
            return {}
        if mode == ConsolidationMode.REPARTITION:
            return self._repartition(dataset, requests, batch_size)
        # "reassign" and "auto" both use the per-document path; "auto" would
        # route genuinely contested *regions* to repartition, but a per-doc
        # candidate pick is the safe, bounded default.
        return self._reassign(dataset, requests)

    # ── reassign: one decision per document ──────────────────────────────
    def _reassign(
        self, dataset: DocumentDataset, requests: list[AdjudicationRequest]
    ) -> dict[str, AdjudicationVerdict]:
        verdicts: dict[str, AdjudicationVerdict] = {}
        for req in requests:
            verdict = self._adjudicate_one(dataset, req)
            if verdict is not None:
                verdicts[req.doc_id] = verdict
        return verdicts

    def _adjudicate_one(
        self, dataset: DocumentDataset, req: AdjudicationRequest
    ) -> AdjudicationVerdict | None:
        try:
            record = dataset[req.doc_index]
        except Exception:
            return None

        picks: list[str] = []
        self_confs: list[float] = []
        rationale: str | None = None
        for attempt in range(self.attempts):
            # Rotate the candidate order between attempts so agreement reflects
            # a real robustness check rather than fixed position bias.
            choices = _rotate(req.candidate_labels, attempt) + [_NOVEL]
            content = _reassign_content(record, req)
            try:
                result = call_with_tools(
                    self._llm_config([req.doc_id]),
                    messages=[{"role": "user", "content": content}],
                    tools=[_adjudicate_tool(choices)],
                    tool_choice="required",
                )
            except Exception:
                continue
            choice, self_conf, rat = _parse_adjudicate(result)
            if choice is None:
                continue
            picks.append(choice)
            if self_conf is not None:
                self_confs.append(self_conf)
            rationale = rationale or rat

        if not picks:
            return None
        modal, count = Counter(picks).most_common(1)[0]
        # Agreement is computed over the attempts that *succeeded*, so a
        # provider hiccup on one attempt degrades to a single-attempt answer
        # rather than halving its confidence.
        agreement = count / len(picks)
        base = sum(self_confs) / len(self_confs) if self_confs else agreement
        confidence = max(0.0, min(1.0, base * agreement))
        assignment = None if modal == _NOVEL else modal
        return AdjudicationVerdict(
            assignment=assignment, confidence=confidence, rationale=rationale
        )

    # ── repartition: one batch grouping call ─────────────────────────────
    def _repartition(
        self,
        dataset: DocumentDataset,
        requests: list[AdjudicationRequest],
        batch_size: int,
    ) -> dict[str, AdjudicationVerdict]:
        batch = requests[: max(1, batch_size)]
        existing = sorted({c for req in batch for c in req.candidate_labels})
        tags: dict[str, AdjudicationRequest] = {}
        content: list[dict[str, Any]] = [{"type": "text", "text": _repartition_prompt(existing)}]
        for n, req in enumerate(batch, start=1):
            tag = f"doc_{n}"
            tags[tag] = req
            content.append({"type": "text", "text": f"=== Document {tag} ==="})
            try:
                record = dataset[req.doc_index]
            except Exception:
                continue
            # Many images in one request, so downscale each — providers cap the
            # per-image budget and full-resolution pages blow through it.
            url = _image_url(record, max_edge=MANY_IMAGE_MAX_EDGE)
            if url is not None:
                content.append({"type": "image_url", "image_url": {"url": url}})
            elif record.text:
                content.append({"type": "text", "text": record.text[:_TEXT_SNIPPET_CHARS]})

        try:
            result = call_with_tools(
                self._llm_config([req.doc_id for req in batch]),
                messages=[{"role": "user", "content": content}],
                tools=[_regroup_tool(existing)],
                tool_choice="required",
            )
            groups = _extract_groups(result)
        except Exception:
            return {}

        return _map_groups_to_verdicts(groups, tags)

    def _llm_config(self, doc_ids: list[str]) -> LLMConfig:
        return LLMConfig(
            model=self.config.model,
            api_key=_resolve_api_key(self.config),
            # Forward the custom endpoint if one is configured: it is what lets a
            # self-hosted / proxied model reach the call, and it is also what
            # skips litellm's up-front model-id check for ids it has no metadata
            # for. Dropping it here would silently ignore the workspace's
            # `classification.api_base` for consolidation alone.
            api_base=self.config.api_base,
            max_tokens=None,
            record_usage=self.record_usage,
            operation=UsageOperation.CLUSTER,
            context={"consolidation": doc_ids},
        )


# ── helpers ─────────────────────────────────────────────────────────────────
def _rotate(items: list[str], by: int) -> list[str]:
    if not items:
        return []
    k = by % len(items)
    return items[k:] + items[:k]


def _tool_args(result: CallResult) -> dict[str, Any]:
    """Parsed arguments of the first tool call, or ``{}`` if unusable."""
    if not result.tool_calls:
        return {}
    try:
        raw = result.tool_calls[0].function.arguments
    except AttributeError:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def _clamped_unit(value: Any) -> float | None:
    """``value`` as a float in ``[0, 1]``, or ``None`` if it isn't a number.

    ``bool`` is excluded explicitly — it is a subclass of ``int``, and a model
    that answers ``true`` has not given us a confidence.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _image_url(record: Any, *, max_edge: int | None = None) -> str | None:
    image = getattr(record, "image", None)
    if image is None:
        return None
    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return image_to_data_url(buf.getvalue(), max_edge=max_edge)
    except Exception:
        return None


def _reassign_content(record: Any, req: AdjudicationRequest) -> list[dict[str, Any]]:
    lines = [
        "You are re-checking a borderline document classification made by an "
        "automated clustering pipeline.",
        "",
        "Candidate document types this document might belong to:",
        *(f"  - {c}" for c in req.candidate_labels),
        "",
        f'The pipeline tentatively labeled it "{req.current_label}".',
        "",
        "Two documents share a type only if the same structured questions could "
        "be answered from each (same extraction schema) — use document type, not "
        "topic. Decide which candidate it truly belongs to. If it fits none of "
        f"them, choose `{_NOVEL}` (a genuinely new type).",
        "",
        f"Call `{_TOOL_ADJUDICATE}` with your choice, a confidence from 0.0 to "
        "1.0, and a one-line rationale.",
    ]
    content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
    if getattr(record, "text", ""):
        content.append({"type": "text", "text": record.text[:_TEXT_SNIPPET_CHARS]})
    url = _image_url(record)
    if url is not None:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def _adjudicate_tool(choices: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_ADJUDICATE,
            "description": (
                "Decide which candidate document type the document belongs to, "
                "or that it is a genuinely new type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": choices,
                        "description": (
                            f"The candidate type it belongs to, or '{_NOVEL}' if none fit."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "How sure you are of this choice, 0.0 to 1.0.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence justifying the choice.",
                    },
                },
                "required": ["choice"],
                "additionalProperties": False,
            },
        },
    }


def _parse_adjudicate(result: CallResult) -> tuple[str | None, float | None, str | None]:
    """Pull ``(choice, confidence, rationale)`` from an ``adjudicate`` call."""
    args = _tool_args(result)
    choice = args.get("choice")
    if not isinstance(choice, str) or not choice:
        return None, None, None
    rationale = args.get("rationale")
    return (
        choice,
        _clamped_unit(args.get("confidence")),
        rationale if isinstance(rationale, str) else None,
    )


def _repartition_prompt(existing: list[str]) -> str:
    lines = [
        "You are re-partitioning a small set of documents that an automated "
        "clustering pipeline was unsure about.",
        "",
        "Group them by document type (same type ⇒ the same structured questions "
        "could be answered from each). For each group, either set "
        "`existing_label` to one of the known types below if it matches, or give "
        "the group a short new `name` if it is a genuinely new type.",
    ]
    if existing:
        lines.append("")
        lines.append("Known types:")
        lines.extend(f"  - {c}" for c in existing)
    lines.append("")
    lines.append(f"Call `{_TOOL_REGROUP}` exactly once with all groups.")
    return "\n".join(lines)


def _regroup_tool(existing: list[str]) -> dict[str, Any]:
    group_props: dict[str, Any] = {
        "members": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": "Document labels (e.g. 'doc_1') in this group.",
        },
        "name": {
            "type": "string",
            "description": "For a NEW type: a short 2-5 word name. Omit if using existing_label.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "How sure you are of this grouping, 0.0 to 1.0.",
        },
    }
    if existing:
        group_props["existing_label"] = {
            "type": "string",
            "enum": existing,
            "description": "An existing type this group matches (instead of name).",
        }
    return {
        "type": "function",
        "function": {
            "name": _TOOL_REGROUP,
            "description": "Partition the attached documents into same-type groups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": group_props,
                            "required": ["members"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["groups"],
                "additionalProperties": False,
            },
        },
    }


def _extract_groups(result: CallResult) -> list[Any]:
    groups = _tool_args(result).get("groups")
    return groups if isinstance(groups, list) else []


def _map_groups_to_verdicts(
    groups: list[Any], tags: dict[str, AdjudicationRequest]
) -> dict[str, AdjudicationVerdict]:
    """Map the model's ``doc_N`` group members back onto the requests.

    A group naming an ``existing_label`` reassigns its members to it; a group
    that only proposes a new ``name`` yields a *novel* verdict, since minting
    the bucket is the framework's job (it owns the numbering).
    """
    verdicts: dict[str, AdjudicationVerdict] = {}
    for group_index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        members = group.get("members")
        if not isinstance(members, list):
            continue
        existing_label = group.get("existing_label")
        assignment = existing_label if isinstance(existing_label, str) and existing_label else None
        confidence = _clamped_unit(group.get("confidence"))
        # A novel group (no existing_label) carries a per-group token so its
        # members share one minted bucket downstream; the group index is stable
        # and unique within this response. Reassign verdicts leave it None.
        novel_group = f"g{group_index}" if assignment is None else None
        for member in members:
            req = tags.get(str(member))
            if req is None:
                continue
            verdicts[req.doc_id] = AdjudicationVerdict(
                assignment=assignment,
                confidence=confidence,
                rationale=None,
                novel_group=novel_group,
            )
    return verdicts
