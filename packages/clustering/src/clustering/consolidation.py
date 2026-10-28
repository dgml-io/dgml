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

"""LLM adjudication of the low-confidence tail.

The embedding pipeline does the bulk of the work cheaply and confidently; a
handful of documents sit on a cluster boundary or in a noise bucket where the
statistics are weak. This module reconsiders *only* those — selecting the
least-confident assignments, offering each its nearest candidate clusters, and
asking an LLM the one question it is good at: *does this document belong to one
of these, or is it genuinely new?* Cost scales with uncertainty, not corpus
size.

The heavy lifting stays here in the framework and is pure Python / torch:

- :func:`select_low_confidence_tail` — budgeted selection over the per-document
  confidence already on the :class:`~clustering.scenarios.base.ScenarioResult`.
- :func:`cluster_centroids` / :func:`candidate_clusters` — the ``candidates_k``
  nearest existing clusters to a document, by manifold distance to each
  cluster's centroid.
- :func:`consolidate` — assemble requests, call the injected adjudicator, and
  merge verdicts back through :meth:`Scenario.refine`.

The LLM call itself is **not** here: an :class:`Adjudicator` is injected by the
caller (dgml-core supplies a litellm-backed one), so this package keeps its
no-LLM-dependency contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

import torch

from clustering.config.schema import (
    ConsolidationApply,
    ConsolidationConfig,
    ConsolidationMode,
    ConsolidationSelectorConfig,
    SelectorStrategy,
)
from clustering.scenarios.base import ScenarioResult

if TYPE_CHECKING:
    from clustering.data.datasets import DocumentDataset
    from clustering.scenarios.base import Scenario


@dataclass(frozen=True)
class AdjudicationRequest:
    """One document handed to the adjudicator for a second opinion.

    ``candidate_labels`` are the nearest existing cluster/category names (most
    likely first); the adjudicator picks one of them, or returns ``None`` to
    flag the document as genuinely novel.
    """

    doc_id: str
    doc_index: int
    current_label: str | None
    candidate_labels: list[str]


@dataclass(frozen=True)
class AdjudicationVerdict:
    """The adjudicator's decision for one document.

    ``assignment`` is a cluster/category name to (re)assign to, or ``None`` for
    a *novel* verdict (open a new bucket). ``confidence`` is the adjudicator's
    ordinal confidence in ``[0, 1]`` (or ``None``); ``rationale`` is a short
    free-text justification for audit.

    ``novel_group`` ties together the members of one *novel* group so they land
    in the same minted bucket. In repartition mode the model can place several
    documents into one new group ("these three are the same new type"); without
    a shared token each would mint its own ``unknown_N`` and the group would be
    split back into singletons — the opposite of what repartition is for. It is
    ``None`` for reassign verdicts, where each document is judged independently
    and distinct novel buckets are correct.
    """

    assignment: str | None
    confidence: float | None = None
    rationale: str | None = None
    novel_group: str | None = None


class Adjudicator(Protocol):
    """Callback that reconsiders a batch of low-confidence documents.

    Implemented in the caller's layer (dgml-core wraps the vision LLM) so the
    framework stays LLM-free. Must be soft-failing in spirit — but callers of
    :func:`consolidate` are also guarded, so a raised exception degrades to
    "no change" rather than aborting the run.
    """

    def __call__(
        self,
        dataset: DocumentDataset,
        requests: list[AdjudicationRequest],
        *,
        mode: ConsolidationMode,
        batch_size: int,
    ) -> dict[str, AdjudicationVerdict]: ...


# Predicted-label conventions that mean "not in a real cluster" — HDBSCAN-style
# noise buckets and the unassigned sentinel.
_NOISE_SUFFIX = "_noise"

# Emergent-bucket prefixes. S1 names its clusters ``cluster_<n>`` and dgml-core
# rewrites those to ``unknown_<n>`` on the way out; the partial-label scenarios
# mint ``unknown_<n>`` directly. A novel bucket has to dodge *both* numbering
# spaces or the rewrite can silently merge it into an unrelated cluster.
_BUCKET_PREFIXES = ("unknown_", "cluster_")


def _is_noise(label: str | None) -> bool:
    return label is None or label.endswith(_NOISE_SUFFIX)


def _conf_or_low(confidence: list[float | None], i: int) -> float:
    """Confidence of doc ``i``, treating missing / ``None`` as maximally uncertain."""
    if i >= len(confidence):
        return -1.0
    c = confidence[i]
    return -1.0 if c is None else float(c)


# Below this max-min confidence gap the signal is treated as flat: a
# bottom-quantile cut over near-tied values selects an essentially arbitrary
# set of documents, so the quantile strategy is suppressed (see
# :func:`select_low_confidence_tail`). Set at 1e-3 deliberately: an
# *uncalibrated* softmax peak over well-separated clusters saturates to ~1.0
# with only float-noise variation (measured ~1e-4 across a real 96-doc corpus)
# — technically non-zero but not a rankable signal. A healthy signal (the
# auto-temperature S1 confidence, or nearest-prototype confidence in S2-S5)
# spreads one to two orders of magnitude wider, so this floor never suppresses
# a genuine ranking.
_MIN_CONFIDENCE_SPREAD = 1e-3

_FLAT_NOTE = (
    "no confidence spread — every document scored ~equally, so a "
    "bottom-quantile selection would be arbitrary; skipped (raise the "
    "confidence signal's resolution, or use an absolute confidence_threshold "
    "to adjudicate anyway)"
)

_MARGIN_DEGRADED_NOTE = (
    "strategy='margin' needs per-class scores and this scenario emits none; "
    "fell back to the bottom-quantile cut"
)


@dataclass(frozen=True)
class TailSelection:
    """The documents to adjudicate, and how they were chosen.

    ``strategy`` is the rule that actually ran, which is not always the one
    that was configured — ``margin`` degrades to ``quantile`` on a scenario
    with no per-class scores. ``notes`` explains any such degradation, or why
    an empty selection is empty; it is surfaced in the run metadata so a no-op
    consolidation pass is self-explaining rather than just silent.
    """

    indices: list[int]
    strategy: SelectorStrategy
    notes: list[str] = field(default_factory=list)


def _confidence_spread(confidence: list[float | None], n: int) -> float:
    """max-min of the per-document confidence (``None`` ⇒ ``-1``) over ``n`` docs.

    ``0.0`` when every document scored identically — the degenerate case a
    quantile cut cannot rank."""
    if n <= 0:
        return 0.0
    vals = [_conf_or_low(confidence, i) for i in range(n)]
    return max(vals) - min(vals)


def select_low_confidence_tail(
    result: ScenarioResult, cfg: ConsolidationSelectorConfig
) -> TailSelection:
    """The documents to adjudicate, per the selector config.

    The active ``strategy`` picks the tail; ``include_noise`` unions in every
    noise / unassigned document; the union is then sorted least-confident-first
    and capped at ``max_docs`` so LLM cost is bounded regardless of how
    uncertain the run is.
    """
    n = len(result.doc_ids)
    conf = result.confidence
    selected: set[int] = set()
    notes: list[str] = []
    strategy = cfg.strategy

    if strategy == SelectorStrategy.NOISE:
        # Nothing to add: the include_noise union below *is* this strategy.
        pass
    elif strategy == SelectorStrategy.CONFIDENCE and cfg.confidence_threshold is not None:
        thr = float(cfg.confidence_threshold)
        selected.update(i for i in range(n) if _conf_or_low(conf, i) < thr)
    elif (
        strategy == SelectorStrategy.MARGIN
        and cfg.margin_threshold is not None
        and result.scores is not None
    ):
        selected.update(_margin_selected(result.scores, float(cfg.margin_threshold)))
    else:
        # Bottom-quantile by confidence — the default, and where a 'margin'
        # request lands when the scenario produced no per-class scores.
        #
        # Guard the degenerate case: when the confidence signal has no spread
        # (every document ~tied — e.g. an uncalibrated softmax that saturated
        # at 1.0), a *partial* quantile cut selects an arbitrary set and lets
        # the adjudicator perturb confidently-correct assignments for no
        # reason. Suppress it — only genuine noise (via ``include_noise``)
        # still enters the tail. A full cut (``quantile >= 1``) selects every
        # document deterministically, so it is exempt: nothing arbitrary about
        # "adjudicate all".
        if strategy == SelectorStrategy.MARGIN:
            notes.append(_MARGIN_DEGRADED_NOTE)
        strategy = SelectorStrategy.QUANTILE
        q = float(cfg.quantile)
        if q >= 1.0 or _confidence_spread(conf, n) >= _MIN_CONFIDENCE_SPREAD:
            selected.update(_quantile_selected(result, q))
        else:
            notes.append(_FLAT_NOTE)

    if cfg.include_noise:
        selected.update(i for i in range(n) if _is_noise(result.predictions[i]))

    ordered = sorted(selected, key=lambda i: _conf_or_low(conf, i))
    capped = ordered[: int(cfg.max_docs)]
    if len(capped) < len(ordered):
        notes.append(
            f"capped at max_docs={cfg.max_docs}: {len(ordered) - len(capped)} further "
            "low-confidence documents were left un-adjudicated"
        )
    return TailSelection(indices=capped, strategy=strategy, notes=notes)


def _quantile_selected(result: ScenarioResult, quantile: float) -> list[int]:
    n = len(result.doc_ids)
    if n == 0 or quantile <= 0.0:
        return []
    q = min(1.0, quantile)
    k = max(1, round(n * q))
    ordered = sorted(range(n), key=lambda i: _conf_or_low(result.confidence, i))
    return ordered[:k]


def _margin_selected(scores: torch.Tensor, margin_threshold: float) -> list[int]:
    """Documents whose top1-top2 score gap falls inside the uncertainty band."""
    if int(scores.shape[0]) == 0 or int(scores.shape[-1]) < 2:
        return []
    top2 = torch.topk(scores, k=2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).tolist()
    return [i for i, m in enumerate(margin) if float(m) < margin_threshold]


def cluster_centroids(
    scenario: Scenario, result: ScenarioResult
) -> tuple[list[str], torch.Tensor | None]:
    """``(names, centroids)`` for every real (non-noise) cluster in ``result``.

    Each centroid is the manifold-mean of its members' embeddings. Computed
    once per consolidation pass and reused for every adjudicated document —
    the tail is small but the corpus need not be, and rebuilding every centroid
    per document makes selection quadratic in corpus size for no benefit.

    Returns ``([], None)`` when the run has no real clusters at all.
    """
    members: dict[str, list[int]] = {}
    for i, lbl in enumerate(result.predictions):
        if lbl is not None and not _is_noise(lbl):
            members.setdefault(lbl, []).append(i)
    if not members:
        return [], None

    names = list(members)
    centroids = torch.stack(
        [
            scenario.manifold.expmap0(
                result.embeddings[torch.tensor(members[name])].mean(dim=0).unsqueeze(0)
            ).squeeze(0)
            for name in names
        ],
        dim=0,
    )
    return names, centroids


def candidate_clusters(
    scenario: Scenario,
    result: ScenarioResult,
    index: int,
    k: int,
    *,
    centroids: tuple[list[str], torch.Tensor | None] | None = None,
) -> list[str]:
    """The ``k`` nearest existing cluster labels to document ``index``.

    Candidates are ranked by manifold distance from the document to each
    cluster centroid. Returns fewer than ``k`` labels when the run has fewer
    clusters, and an empty list when there are none. Pass ``centroids`` from
    :func:`cluster_centroids` to reuse one computation across a whole tail.
    """
    names, stack = cluster_centroids(scenario, result) if centroids is None else centroids
    if stack is None or not names:
        return []

    query = result.embeddings[index].unsqueeze(0)
    dist = scenario.manifold.pairwise_dist(query, stack).squeeze(0)
    order = sorted(range(len(names)), key=lambda j: float(dist[j].item()))
    return [names[j] for j in order[: max(0, k)]]


def _next_novel_index(labels: list[str | None]) -> int:
    """One past the highest ``unknown_<n>`` *or* ``cluster_<n>`` index in use.

    Both spaces are scanned because dgml-core rewrites S1's ``cluster_<n>`` to
    ``unknown_<n>`` downstream: minting ``unknown_0`` next to an existing
    ``cluster_0`` would collide *after* that rewrite and silently merge two
    unrelated groups.
    """
    highest = -1
    for lbl in labels:
        for prefix in _BUCKET_PREFIXES:
            if lbl and lbl.startswith(prefix):
                suffix = lbl[len(prefix) :]
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
    return highest + 1


def consolidate(
    scenario: Scenario,
    result: ScenarioResult,
    unknown_dataset: DocumentDataset,
    adjudicator: Adjudicator,
) -> ScenarioResult:
    """Run the consolidation pass (see :meth:`Scenario.consolidate`).

    No-op (returns ``result`` unchanged) when consolidation is disabled.
    Soft-fails: any adjudicator error is swallowed and the original result
    returned with the error in metadata, matching dgml-core's never-raise
    clustering philosophy.
    """
    cfg = scenario.config.scenario.consolidation
    if not cfg.enabled:
        return result

    tail = select_low_confidence_tail(result, cfg.selector)
    if not tail.indices:
        return _derived(
            result,
            {
                "enabled": True,
                "strategy": str(tail.strategy),
                "n_selected": 0,
                "notes": tail.notes or ["empty tail"],
            },
        )

    centroids = cluster_centroids(scenario, result)
    requests = [
        AdjudicationRequest(
            doc_id=result.doc_ids[i],
            doc_index=i,
            current_label=result.predictions[i],
            candidate_labels=candidate_clusters(
                scenario, result, i, cfg.candidates_k, centroids=centroids
            ),
        )
        for i in tail.indices
    ]

    try:
        verdicts = adjudicator(unknown_dataset, requests, mode=cfg.mode, batch_size=cfg.batch_size)
    except Exception as exc:
        return _derived(
            result,
            {
                "enabled": True,
                "strategy": str(tail.strategy),
                "n_selected": len(requests),
                "notes": tail.notes,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    return _apply_verdicts(scenario, result, unknown_dataset, requests, verdicts, cfg, tail)


def _apply_verdicts(
    scenario: Scenario,
    result: ScenarioResult,
    unknown_dataset: DocumentDataset,
    requests: list[AdjudicationRequest],
    verdicts: dict[str, AdjudicationVerdict],
    cfg: ConsolidationConfig,
    tail: TailSelection,
) -> ScenarioResult:
    """Merge adjudicator verdicts into the result (or record them for review)."""
    n = len(result.doc_ids)
    novel_counter = _next_novel_index(result.predictions)
    abstain_floor = scenario.config.scenario.calibration.abstain_threshold

    corrections: dict[str, str] = {}
    new_conf = list(result.confidence)
    review = list(result.review) if result.review else [False] * n
    review.extend([False] * (n - len(review)))
    records: list[dict[str, Any]] = []
    n_reassigned = 0
    n_novel = 0
    # Members of one novel group share a minted bucket: the first mints
    # ``unknown_N``, the rest reuse it, so a group the adjudicator kept together
    # is not split back into singletons.
    novel_labels: dict[str, str] = {}

    # Iterate the *requests*, not the verdict dict, so novel buckets are
    # numbered by document order — the adjudicator's dict ordering is an
    # implementation detail and would make the minted names depend on it.
    for req in requests:
        verdict = verdicts.get(req.doc_id)
        if verdict is None:
            continue
        i = req.doc_index
        if verdict.assignment is None:
            token = verdict.novel_group
            if token is not None and token in novel_labels:
                new_label = novel_labels[token]
            else:
                new_label = f"unknown_{novel_counter}"
                novel_counter += 1
                n_novel += 1
                if token is not None:
                    novel_labels[token] = new_label
        else:
            new_label = verdict.assignment
            n_reassigned += 1
        records.append(
            {
                "doc_id": req.doc_id,
                "from": result.predictions[i],
                "to": new_label,
                "confidence": verdict.confidence,
                "rationale": verdict.rationale,
            }
        )
        if cfg.apply == ConsolidationApply.AUTO:
            corrections[req.doc_id] = new_label
            if verdict.confidence is not None:
                new_conf[i] = verdict.confidence
            # The adjudicator is a second opinion, not a human sign-off, so the
            # review flag is re-derived from its confidence against the same
            # abstain floor rather than blanket-cleared: a verdict the LLM was
            # itself unsure of still belongs in the queue.
            review[i] = (
                abstain_floor is not None
                and verdict.confidence is not None
                and verdict.confidence < abstain_floor
            )
        else:  # suggest — leave labels, flag for human review
            review[i] = True

    meta: dict[str, Any] = {
        "enabled": True,
        "mode": str(cfg.mode),
        "apply": str(cfg.apply),
        "strategy": str(tail.strategy),
        "consolidated_by": "llm",
        "model": cfg.model,
        "n_selected": len(requests),
        "n_reassigned": n_reassigned,
        "n_novel": n_novel,
        "notes": tail.notes,
        "verdicts": records,
    }

    if cfg.apply == ConsolidationApply.AUTO and corrections:
        # ``refine`` applies the {doc_id: label} corrections — scenarios may
        # override it to do more (e.g. recompute prototypes) — and copies
        # confidence/review verbatim, so the updated columns are layered on top
        # of whatever it returns.
        refined = scenario.refine(result, corrections, unknown_dataset)
        return _derived(refined, meta, confidence=new_conf, review=review)

    # suggest mode: labels unchanged, only the review queue + provenance move.
    return _derived(result, meta, review=review)


def _derived(
    result: ScenarioResult,
    meta: dict[str, Any],
    *,
    confidence: list[float | None] | None = None,
    review: list[bool] | None = None,
) -> ScenarioResult:
    """``result`` plus a ``consolidation`` metadata block and optional columns.

    Every mutable field is copied, so consolidation stays a pure function of
    its input — the caller's original result is never aliased or mutated.
    """
    return replace(
        result,
        predictions=list(result.predictions),
        confidence=list(result.confidence if confidence is None else confidence),
        true_labels=list(result.true_labels),
        class_names=list(result.class_names) if result.class_names else None,
        review=list(result.review if review is None else review),
        metadata={**result.metadata, "consolidation": meta},
    )
