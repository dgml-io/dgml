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

"""LLM-based clustering for very small corpora.

The embedding + statistical-clustering pipeline in
:mod:`dgml_core.run_clustering` is the right tool once a corpus is large
enough for document-frequency statistics and neighborhood graphs to mean
something. On a handful of documents it is not: tf-idf has almost nothing
to weight, k-NN graphs are dominated by noise, and density estimators
collapse everything into one cluster (or all noise). This module is the
small-corpus escape hatch.

Instead of embedding, it reuses the very same vision LLM machinery that
powers ``dgml file add --auto-classify`` (:mod:`dgml_core.classification`)
and asks the model to *partition* the whole corpus in a single call: every
document's rendered first pages are sent at once, each tagged with a stable
``doc_N`` label, and the model returns a set of groups — each either
matching an existing DocSet or proposing a brand-new document type (name,
description, key questions), exactly like the classifier's
``create_new_docset`` tool.

The output contract is deliberately identical to
:func:`dgml_core.run_clustering.run_clustering_detailed` so the two methods
are interchangeable behind :func:`dgml_core.clustering.clustering_internal`:
each file maps to either an *existing DocSet name* (the model judged it
belongs there) or an emergent ``"unknown_N"`` bucket. For the emergent
buckets the model's proposed name/description/key-questions ride along in
:attr:`LLMClusteringResult.proposals`, so the outer clustering flow can
create those DocSets without a second LLM round-trip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .classification import (
    ClassificationConfig,
    ClassificationDecision,
    _resolve_api_key,
)
from .errors import ClassificationFailed
from .llm import LLMConfig, call_with_tools
from .models import DocSet
from .prompts import get as prompt
from .storage import Workspace
from .usage import OPERATION_CLUSTER
from .utils import MANY_IMAGE_MAX_EDGE, gather_file_pages, image_to_data_url

_TOOL_GROUP = "group_documents"

# Default ceiling on how many files a single LLM clustering call covers.
# The whole corpus goes into one prompt (every file contributes up to
# ``config.max_pages`` page images), so cost and context grow linearly with
# the file count — this bound keeps a single call from blowing past a
# model's context window. Files beyond the cap are reported as failed so
# the caller can fall back to the embedding pipeline for large corpora
# (which is where LLM clustering stops being the right tool anyway).
DEFAULT_MAX_FILES = 24


@dataclass(frozen=True)
class LLMClusteringResult:
    """Outcome of :func:`llm_cluster_files`.

    ``clusters`` maps each successfully-grouped ``file_id`` to a cluster
    name — either the name of an existing DocSet the model assigned it to,
    or an emergent ``"unknown_N"`` bucket. This mirrors the
    ``{doc_id: cluster_name}`` contract of
    :func:`dgml_core.run_clustering.run_clustering`.

    ``proposals`` carries the model's proposed DocSet metadata for each
    emergent bucket, keyed by the same ``"unknown_N"`` name used in
    ``clusters``. Each value is a ``decision="new"``
    :class:`~dgml_core.classification.ClassificationDecision`, so the outer
    :func:`dgml_core.clustering.clustering` flow can create the DocSet
    directly instead of issuing a second naming call.

    ``failed_file_ids`` lists files that were not placed in any cluster:
    those with no rendered page image, those over the ``max_files`` cap, and
    any the model omitted from every group.
    """

    clusters: dict[str, str]
    proposals: dict[str, ClassificationDecision] = field(default_factory=dict)
    failed_file_ids: list[str] = field(default_factory=list)


def llm_cluster_files(
    workspace: Workspace,
    file_ids: list[str],
    *,
    config: ClassificationConfig,
    docsets: list[DocSet] | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    debug: bool = False,
) -> LLMClusteringResult:
    """Cluster ``file_ids`` in one vision-LLM call and return the partition.

    Every file's rendered first pages (up to ``config.max_pages`` each) are
    sent in a single request, tagged ``doc_1``…``doc_N``. The model is
    forced to call :data:`_TOOL_GROUP` exactly once, returning a list of
    groups that partition the documents by *document type* (same-schema
    grouping, the same criterion :mod:`dgml_core.classification` uses).

    ``docsets`` — when non-empty — are offered as existing categories the
    model may assign a group to (its ``id`` becomes a valid
    ``existing_docset_id``); files in such a group come back keyed to that
    DocSet's *name*. Pass ``[]`` / ``None`` for a fresh partition where every
    group is a new proposal. This is how the fresh vs incremental
    distinction in :func:`dgml_core.clustering.clustering_internal` is
    threaded through.

    Raises :class:`~dgml_core.errors.ClassificationFailed` when no file has
    a usable page image or the provider call fails, and
    :class:`~dgml_core.errors.AuthError` when ``config.api_key_env`` names an
    env var that isn't set — the same failure contract as
    :func:`dgml_core.classification.classify_file`.
    """
    docsets = docsets or []
    if max_files < 1:
        raise ValueError(f"max_files must be >= 1; got {max_files}")

    # Cap the corpus size for a single call. Overflow files are reported as
    # failed rather than silently dropped so the caller can route them.
    capped = file_ids[:max_files]
    overflow = file_ids[max_files:]

    labels: dict[str, str] = {}  # "doc_N" -> file_id
    doc_blocks: list[dict[str, Any]] = []
    usable: list[str] = []
    no_pages: list[str] = []
    for fid in capped:
        pages = gather_file_pages(workspace, fid, config.max_pages)
        if not pages:
            no_pages.append(fid)
            continue
        label = f"doc_{len(usable) + 1}"
        labels[label] = fid
        usable.append(fid)
        doc_blocks.append({"type": "text", "text": f"=== Document {label} ==="})
        for img in pages:
            # Whole corpus goes in one many-image request → downscale each page
            # under the provider's per-image cap (Anthropic rejects >2000 px in a
            # many-image request; full page renders are ~2500-3500 px).
            url = image_to_data_url(img, max_edge=MANY_IMAGE_MAX_EDGE)
            doc_blocks.append({"type": "image_url", "image_url": {"url": url}})

    if not usable:
        raise ClassificationFailed(
            f"no page images found for any of files {capped!r}; "
            "LLM clustering requires successfully rendered pages"
        )

    grouping_prompt = _build_grouping_prompt(list(labels), docsets)
    tools = [_group_documents_tool(docsets)]
    api_key = _resolve_api_key(config)
    content: list[dict[str, Any]] = [{"type": "text", "text": grouping_prompt}, *doc_blocks]

    llm_config = LLMConfig(
        model=config.model,
        api_key=api_key,
        max_tokens=None,
        # Greedy decoding: clustering should be as reproducible as possible, so
        # the same corpus yields the same partition run-to-run. Without this the
        # provider default (e.g. Gemini ~1.0) makes the model flip between, say,
        # merging vs. splitting a borderline document. Ignored for Anthropic
        # models (temperature is never sent there — see _build_completion_kwargs).
        temperature=0.0,
        workspace=workspace,
        debug=debug,
        operation=OPERATION_CLUSTER,
        context={"file_ids": usable},
    )
    try:
        result = call_with_tools(
            llm_config,
            messages=[{"role": "user", "content": content}],
            tools=tools,
            tool_choice="required",
        )
    except Exception as exc:
        raise ClassificationFailed(f"LLM call failed: {type(exc).__name__}: {exc}") from exc

    groups = _extract_groups(result.response)
    return _assemble_result(groups, labels, usable, no_pages + overflow, docsets)


def _assemble_result(
    groups: list[Any],
    labels: dict[str, str],
    usable: list[str],
    pre_failed: list[str],
    docsets: list[DocSet],
) -> LLMClusteringResult:
    """Turn the model's raw ``groups`` list into an :class:`LLMClusteringResult`.

    Tolerant by design: malformed groups, unknown ``doc_N`` labels, and
    files placed in more than one group are handled without raising — the
    first valid placement wins and anything left unplaced lands in
    ``failed_file_ids`` so the caller can route it. This mirrors the
    soft-fail spirit of :func:`dgml_core.clustering.clustering`.
    """
    name_by_id = {ds.id: ds.name for ds in docsets}
    # Canonical labels are "doc_1".."doc_N"; also index by the bare number so a
    # model that answers with "1" / "doc1" / "Document 1" still resolves.
    by_number = {label.split("_", 1)[1]: fid for label, fid in labels.items()}
    clusters: dict[str, str] = {}
    proposals: dict[str, ClassificationDecision] = {}
    assigned: set[str] = set()
    new_index = 0
    n_member_refs = 0  # total member tokens the model emitted (mapped or not)
    sample_members: list[str] = []  # a few raw member tokens, for error detail

    for group in groups:
        if not isinstance(group, dict):
            continue
        members = group.get("members")
        if not isinstance(members, list):
            continue
        n_member_refs += len(members)
        for m in members:
            if len(sample_members) < 8:
                sample_members.append(repr(m))

        docset_id = group.get("existing_docset_id")
        decision: ClassificationDecision | None = None
        if isinstance(docset_id, str) and docset_id in name_by_id:
            cluster_name = name_by_id[docset_id]
            is_existing = True
        else:
            is_existing = False
            # A new group is a valid cluster even when the model omitted the
            # name/description/key_questions (a lite model often returns only
            # the required `members`). Keep the partition regardless; the
            # naming proposal is a bonus — when absent, clustering()'s pass-2
            # names the "unknown_N" bucket with its normal fallback LLM call.
            decision = _new_group_decision(group)
            cluster_name = f"unknown_{new_index}"

        committed = 0
        for member in members:
            fid = _resolve_member(member, labels, by_number)
            if fid is None or fid in assigned:
                continue
            clusters[fid] = cluster_name
            assigned.add(fid)
            committed += 1

        if committed and not is_existing:
            if decision is not None:
                proposals[cluster_name] = decision
            new_index += 1

    # The model returned groups but not one document could be placed — usually a
    # label-format mismatch (members don't resolve to doc_1..doc_N) or groups
    # with neither a name nor a valid existing_docset_id. Surface it loudly
    # instead of silently reporting every file as "failed" (which reads like the
    # model refused to answer).
    if not clusters and n_member_refs:
        raise ClassificationFailed(
            f"{_TOOL_GROUP} returned {n_member_refs} member reference(s) across "
            f"{len(groups)} group(s), but none could be placed (members did not "
            f"resolve to doc_1..doc_{len(labels)}, or groups lacked a name / valid "
            f"existing_docset_id). Sample members: {', '.join(sample_members)}"
        )

    failed: list[str] = [fid for fid in usable if fid not in assigned]
    for fid in pre_failed:
        if fid not in failed:
            failed.append(fid)
    return LLMClusteringResult(clusters=clusters, proposals=proposals, failed_file_ids=failed)


def _resolve_member(member: Any, labels: dict[str, str], by_number: dict[str, str]) -> str | None:
    """Map one raw ``members`` entry to a file id, tolerating label-format drift.

    Accepts the canonical ``"doc_3"``, a bare/int ``3`` / ``"3"``, and noisy
    forms like ``"doc3"`` or ``"Document doc_3"`` by extracting the first run of
    digits. Returns ``None`` when nothing plausibly matches.
    """
    if isinstance(member, int):
        return by_number.get(str(member))
    if not isinstance(member, str):
        return None
    if member in labels:  # exact "doc_N"
        return labels[member]
    match = re.search(r"\d+", member)
    return by_number.get(match.group()) if match else None


def _new_group_decision(group: dict[str, Any]) -> ClassificationDecision | None:
    """Build a ``decision="new"`` :class:`ClassificationDecision` from a group.

    Returns ``None`` when the group has no usable name (nothing to create a
    DocSet from). ``description`` and ``key_questions`` are best-effort:
    missing/blank values degrade to ``""`` / ``()`` rather than failing the
    whole run, since the group's *membership* is the valuable signal.
    """
    name = group.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    description = group.get("description")
    description = description.strip() if isinstance(description, str) else ""
    questions: list[str] = []
    raw_questions = group.get("key_questions")
    if isinstance(raw_questions, list):
        for q in raw_questions:
            if isinstance(q, str) and q.strip():
                questions.append(q.strip())
    return ClassificationDecision(
        decision="new",
        new_name=name.strip(),
        new_description=description,
        new_key_questions=tuple(questions),
    )


def _extract_groups(response: Any) -> list[Any]:
    """Pull the ``groups`` array out of the single ``group_documents`` tool call.

    litellm returns OpenAI-compatible objects for every provider, so the
    ``response.choices[0].message.tool_calls[0]`` path is stable (the same
    assumption :func:`dgml_core.classification._extract_single_tool_call`
    relies on). Raises :class:`ClassificationFailed` on a malformed response.
    """
    import json

    try:
        tool_calls = response.choices[0].message.tool_calls
    except (AttributeError, IndexError, TypeError) as exc:
        raise ClassificationFailed(
            f"LLM response missing tool_calls: {type(exc).__name__}: {exc}"
        ) from exc

    if not tool_calls:
        raise ClassificationFailed("LLM response contained no tool calls")

    call = tool_calls[0]
    name = getattr(getattr(call, "function", None), "name", None)
    if name != _TOOL_GROUP:
        raise ClassificationFailed(f"LLM returned unexpected tool name: {name!r}")

    raw_args = getattr(getattr(call, "function", None), "arguments", None)
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
    except (json.JSONDecodeError, TypeError) as exc:
        raise ClassificationFailed(f"LLM tool-call arguments not valid JSON: {exc}") from exc

    groups = args.get("groups")
    if not isinstance(groups, list):
        raise ClassificationFailed(f"{_TOOL_GROUP} call missing a 'groups' array")
    return groups


def _build_grouping_prompt(doc_labels: list[str], docsets: list[DocSet]) -> str:
    """Assemble the partition prompt from the fixed pieces in ``prompts.yaml``.

    The wording lives in ``dgml_core/resources/prompts.yaml`` (keys
    ``cluster_grouping_*``) so it can be tuned without touching code; this
    function only interpolates the runtime data — the document count/labels
    and, in incremental mode, the existing DocSets offered as categories —
    and joins the pieces with blank lines.
    """
    parts = [
        prompt("cluster_grouping_intro"),
        prompt("cluster_grouping_doc_manifest").format(
            count=len(doc_labels), labels=", ".join(doc_labels)
        ),
    ]
    if docsets:
        listing = [prompt("cluster_grouping_existing_intro")]
        for ds in docsets:
            listing.append(f"- id={ds.id}")
            listing.append(f"  name: {ds.name}")
            if ds.description:
                listing.append(f"  description: {ds.description}")
            if ds.key_questions:
                listing.append("  key questions this DocSet's documents answer:")
                for q in ds.key_questions:
                    listing.append(f"    - {q}")
        parts.append("\n".join(listing))
    parts.append(prompt("cluster_grouping_instructions").format(tool=_TOOL_GROUP))
    return "\n\n".join(parts)


def _group_documents_tool(docsets: list[DocSet]) -> dict[str, Any]:
    """litellm/OpenAI tool schema for the single partition call.

    Each group carries a required ``members`` list of ``doc_N`` labels, plus
    *either* an ``existing_docset_id`` (constrained to the real DocSet ids
    when any exist) *or* a new-type triple (``name`` / ``description`` /
    ``key_questions``). The either/or isn't expressible in a portable JSON
    schema, so it's enforced in the prompt and validated leniently in
    :func:`_assemble_result`.
    """
    group_properties: dict[str, Any] = {
        "members": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
            "description": (
                "The document labels (e.g. 'doc_1', 'doc_2') that belong to "
                "this group. Every document must appear in exactly one group."
            ),
        },
        "name": {
            "type": "string",
            "description": (
                "For a NEW document type: a short document-type name (2-5 "
                "words). Omit when assigning to an existing DocSet."
            ),
        },
        "description": {
            "type": "string",
            "description": "For a NEW document type: one sentence describing the type.",
        },
        "key_questions": {
            "type": "array",
            "minItems": 3,
            "maxItems": 7,
            "items": {"type": "string"},
            "description": (
                "For a NEW document type: 3-7 concrete, type-discriminating "
                "questions answerable from the first pages."
            ),
        },
    }
    if docsets:
        group_properties["existing_docset_id"] = {
            "type": "string",
            "enum": [ds.id for ds in docsets],
            "description": (
                "The id of an existing DocSet this group of documents belongs "
                "to. Set this INSTEAD of name/description/key_questions when a "
                "group matches an existing DocSet."
            ),
        }
        # With existing DocSets available a group may use existing_docset_id
        # instead of naming, so only `members` can be unconditionally required.
        required = ["members"]
    else:
        # Fresh clustering: every group is necessarily a new type, so force the
        # model to name it in the same call. Without this a lite model returns
        # only the required `members` and skips the (optional) name, leaving
        # unnamed buckets that need a second naming round-trip.
        required = ["members", "name", "description", "key_questions"]

    return {
        "type": "function",
        "function": {
            "name": _TOOL_GROUP,
            "description": (
                "Partition all of the attached documents into groups of the "
                "same document type (documents that could share one extraction "
                "schema)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": group_properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["groups"],
                "additionalProperties": False,
            },
        },
    }
