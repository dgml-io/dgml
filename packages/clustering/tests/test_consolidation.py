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

"""Tests for the consolidation pass: selector, candidates, apply.

The LLM adjudicator is faked (a plain callable matching the ``Adjudicator``
protocol), so these exercise the framework's pure selection / candidate /
merge logic with no provider dependency.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from clustering.config.schema import (
    Config,
    ConsolidationConfig,
    ConsolidationSelectorConfig,
    ManifoldConfig,
    ManifoldName,
    SelectorStrategy,
)
from clustering.consolidation import (
    AdjudicationRequest,
    AdjudicationVerdict,
    candidate_clusters,
    cluster_centroids,
    select_low_confidence_tail,
)
from clustering.data.datasets import DocumentDataset, DocumentRecord
from clustering.manifolds import build_manifold
from clustering.scenarios import build_scenario
from clustering.scenarios.base import ScenarioResult
from PIL import Image
from pydantic import ValidationError

_DIM = 8


def _result(
    preds: list[str | None],
    conf: list[float | None],
    emb: torch.Tensor,
    *,
    scores: torch.Tensor | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        run_id="r",
        scenario_name="s1",
        doc_ids=[f"d{i}" for i in range(len(preds))],
        embeddings=emb,
        predictions=preds,
        confidence=conf,
        true_labels=[None] * len(preds),
        scores=scores,
    )


def _selected(result: ScenarioResult, cfg: ConsolidationSelectorConfig) -> list[int]:
    return select_low_confidence_tail(result, cfg).indices


# ── selector ────────────────────────────────────────────────────────────────
def test_quantile_selects_least_confident_plus_noise() -> None:
    res = _result(
        ["cluster_0", "cluster_0", "cluster_1", "cluster_noise"],
        [0.9, 0.2, 0.8, 0.0],
        torch.randn(4, _DIM),
    )
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.QUANTILE, quantile=0.5, include_noise=True
    )
    selected = set(_selected(res, cfg))
    assert 1 in selected  # 0.2 — lowest real cluster
    assert 3 in selected  # noise bucket
    assert 0 not in selected  # 0.9 — confident, kept


def test_confidence_threshold_strategy() -> None:
    res = _result(["a", "b", "c"], [0.95, 0.4, 0.1], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.CONFIDENCE, confidence_threshold=0.5, include_noise=False
    )
    assert set(_selected(res, cfg)) == {1, 2}


def test_max_docs_caps_the_tail_and_says_so() -> None:
    res = _result(["a"] * 10, [i / 10 for i in range(10)], torch.randn(10, _DIM))
    cfg = ConsolidationSelectorConfig(strategy=SelectorStrategy.QUANTILE, quantile=1.0, max_docs=3)
    selection = select_low_confidence_tail(res, cfg)
    # The cap keeps the *least* confident first.
    assert selection.indices == [0, 1, 2]
    # A silent cap reads like "we looked at everything"; it must be reported.
    assert any("max_docs=3" in note and "7 further" in note for note in selection.notes)


def test_none_confidence_is_maximally_uncertain() -> None:
    res = _result(["a", "b", "c"], [None, 0.9, 0.8], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.QUANTILE, quantile=0.34, include_noise=False
    )
    assert _selected(res, cfg) == [0]


def test_noise_strategy_selects_only_noise() -> None:
    # 'noise' is a strategy in its own right, not quantile-plus-noise: the
    # confident real clusters must stay out of the tail even though one of them
    # is the least confident document in the run.
    res = _result(["a", "b", "cluster_noise"], [0.1, 0.9, 0.5], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(strategy=SelectorStrategy.NOISE, include_noise=True)
    selection = select_low_confidence_tail(res, cfg)
    assert selection.indices == [2]
    assert selection.strategy == "noise"


# ── margin strategy ──────────────────────────────────────────────────────────
def test_margin_selects_narrow_top_two_gaps() -> None:
    # doc 0 is contested (gap 0.05), doc 1 is decisive (gap 0.8).
    scores = torch.tensor([[0.50, 0.45, 0.05], [0.90, 0.10, 0.00]], dtype=torch.float32)
    res = _result(["a", "a"], [0.5, 0.9], torch.randn(2, _DIM), scores=scores)
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.MARGIN, margin_threshold=0.2, include_noise=False
    )
    selection = select_low_confidence_tail(res, cfg)
    assert selection.indices == [0]
    assert selection.strategy == "margin"
    assert selection.notes == []


def test_margin_degrades_to_quantile_and_reports_it() -> None:
    # No per-class scores (S1 emits none), so 'margin' cannot run. It falls
    # back to the quantile cut — but silently swapping the operator's selection
    # rule is exactly the kind of thing that only shows up on the invoice.
    res = _result(["a", "b", "c", "d"], [0.1, 0.4, 0.7, 0.9], torch.randn(4, _DIM))
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.MARGIN, margin_threshold=0.2, quantile=0.25, include_noise=False
    )
    selection = select_low_confidence_tail(res, cfg)
    assert selection.indices == [0]  # the quantile cut ran instead
    assert selection.strategy == "quantile"
    assert any("per-class scores" in note for note in selection.notes)


def test_margin_needs_at_least_two_classes() -> None:
    scores = torch.tensor([[0.9], [0.8]], dtype=torch.float32)
    res = _result(["a", "a"], [0.9, 0.8], torch.randn(2, _DIM), scores=scores)
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.MARGIN, margin_threshold=1.0, include_noise=False
    )
    # A one-column score matrix has no top1-top2 gap to measure.
    assert _selected(res, cfg) == []


# ── degenerate-tail guard: a flat confidence signal must not be sliced ────────
def test_flat_confidence_suppresses_partial_quantile() -> None:
    # Every document tied at 1.0 (the saturated-softmax case). A partial
    # bottom-quantile cut would pick an arbitrary subset, so it is suppressed.
    res = _result(["a", "b", "c", "d"], [1.0, 1.0, 1.0, 1.0], torch.randn(4, _DIM))
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.QUANTILE, quantile=0.5, include_noise=False
    )
    selection = select_low_confidence_tail(res, cfg)
    assert selection.indices == []
    assert any("no confidence spread" in note for note in selection.notes)


def test_flat_real_confidence_still_adjudicates_noise() -> None:
    # All real clusters tie at 1.0 but a noise bucket sits at 0.0. The noise
    # doc gives the column genuine spread, so the guard does not fire here —
    # and the noise document must always be adjudicated via include_noise.
    res = _result(["a", "b", "cluster_noise"], [1.0, 1.0, 0.0], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.QUANTILE, quantile=0.5, include_noise=True
    )
    assert 2 in _selected(res, cfg)


def test_fully_flat_with_noise_flag_selects_only_noise() -> None:
    # When *every* document (noise included) ties, the quantile cut is
    # suppressed and only the noise-flagged documents are adjudicated.
    res = _result(["a", "b", "cluster_noise"], [1.0, 1.0, 1.0], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.QUANTILE, quantile=0.5, include_noise=True
    )
    assert set(_selected(res, cfg)) == {2}


def test_near_tied_saturated_confidence_is_suppressed() -> None:
    # The real failure mode: an uncalibrated softmax that saturated to ~1.0
    # with only float-noise variation (~1e-4). That is not a rankable signal,
    # so the partial quantile cut must still be suppressed.
    res = _result(
        ["a", "b", "c", "d"],
        [1.0000, 1.0001, 0.9999, 1.0002],
        torch.randn(4, _DIM),
    )
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.QUANTILE, quantile=0.25, include_noise=False
    )
    assert _selected(res, cfg) == []


def test_flat_confidence_full_quantile_is_exempt() -> None:
    # quantile >= 1.0 means "adjudicate everything" — deterministic, not
    # arbitrary — so the guard does not apply even with a flat signal.
    res = _result(["a", "b", "c"], [1.0, 1.0, 1.0], torch.randn(3, _DIM))
    cfg = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.QUANTILE, quantile=1.0, include_noise=False
    )
    assert set(_selected(res, cfg)) == {0, 1, 2}


def test_flat_confidence_absolute_threshold_still_works() -> None:
    # The 'confidence' strategy is an absolute cutoff, not a ranking, so it is
    # unaffected by the spread guard: a threshold above the tied value selects
    # everything; below it selects nothing.
    res = _result(["a", "b", "c"], [1.0, 1.0, 1.0], torch.randn(3, _DIM))
    hit = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.CONFIDENCE, confidence_threshold=1.0, include_noise=False
    )
    miss = ConsolidationSelectorConfig(
        strategy=SelectorStrategy.CONFIDENCE, confidence_threshold=0.5, include_noise=False
    )
    assert set(_selected(res, hit)) == set()  # strict <, so a tie at the cutoff is kept
    assert _selected(res, miss) == []


# ── config validation ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantile": 1.5},
        {"quantile": -0.1},
        {"max_docs": -1},
        {"confidence_threshold": 2.0},
        {"margin_threshold": -0.5},
        # A strategy whose own knob is unset used to fall through to quantile.
        {"strategy": "confidence"},
        {"strategy": "margin"},
        # ...and this combination selects literally nothing.
        {"strategy": "noise", "include_noise": False},
    ],
)
def test_selector_config_rejects_incoherent_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ConsolidationSelectorConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"candidates_k": 0}, {"batch_size": 0}])
def test_consolidation_config_rejects_empty_bounds(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ConsolidationConfig(**kwargs)


def test_consolidation_is_off_by_default() -> None:
    cfg = ConsolidationConfig()
    assert cfg.enabled is False
    # 'suggest' by default: an LLM overruling the embedding partition is a
    # change a human should see before it lands.
    assert cfg.apply == "suggest"


# ── candidate assembly ────────────────────────────────────────────────────────
def _euclidean_scenario() -> SimpleNamespace:
    manifold = build_manifold(ManifoldConfig(name=ManifoldName.EUCLIDEAN, dim=2, curvature=0.0))
    return SimpleNamespace(manifold=manifold)


def test_candidate_clusters_orders_by_distance() -> None:
    scenario = _euclidean_scenario()
    emb = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0], [0.2, 0.0]], dtype=torch.float32
    )
    # Two clusters: 'near' at the origin, 'far' around x=10; doc 4 is near the origin.
    res = _result(["near", "near", "far", "far", "near"], [0.9] * 5, emb)
    cands = candidate_clusters(scenario, res, index=4, k=2)  # type: ignore[arg-type]
    assert cands == ["near", "far"]


def test_candidate_clusters_ignores_noise() -> None:
    scenario = _euclidean_scenario()
    emb = torch.tensor([[0.0, 0.0], [5.0, 0.0], [2.5, 0.0]], dtype=torch.float32)
    res = _result(["cluster_noise", "real", None], [0.0, 0.9, None], emb)
    cands = candidate_clusters(scenario, res, index=2, k=3)  # type: ignore[arg-type]
    assert cands == ["real"]  # noise + None excluded


def test_candidate_clusters_with_no_real_clusters() -> None:
    scenario = _euclidean_scenario()
    emb = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    res = _result(["cluster_noise", None], [0.0, None], emb)
    assert cluster_centroids(scenario, res) == ([], None)  # type: ignore[arg-type]
    assert candidate_clusters(scenario, res, index=0, k=3) == []  # type: ignore[arg-type]


def test_precomputed_centroids_match_recomputing_them() -> None:
    # The whole point of hoisting the centroids out of the per-document loop is
    # that it changes cost, not answers.
    scenario = _euclidean_scenario()
    emb = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0], [5.0, 0.0]], dtype=torch.float32
    )
    res = _result(["near", "near", "far", "far", "near"], [0.9] * 5, emb)
    centroids = cluster_centroids(scenario, res)  # type: ignore[arg-type]
    for i in range(5):
        assert candidate_clusters(scenario, res, index=i, k=2, centroids=centroids) == (  # type: ignore[arg-type]
            candidate_clusters(scenario, res, index=i, k=2)  # type: ignore[arg-type]
        )


# ── end-to-end consolidate via a fake adjudicator ─────────────────────────────
class _MemDataset(DocumentDataset):
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> DocumentRecord:
        return DocumentRecord(
            doc_id=f"doc_{index}",
            label=None,
            image=Image.new("RGB", (8, 8), color=(index * 9 % 255, 0, 0)),
            text=f"document {index}",
            thumbnail_path=None,
        )


def _s1_config(
    apply: str,
    *,
    abstain_threshold: float | None = None,
    **selector: Any,
) -> Config:
    raw: dict[str, Any] = {
        "scenario": {
            "name": "s1",
            "k_clusters": 2,
            "cluster_algorithm": "kmeans",
            "calibration": {"abstain_threshold": abstain_threshold},
            "consolidation": {
                "enabled": True,
                "apply": apply,
                "candidates_k": 2,
                "selector": {"strategy": "quantile", "quantile": 1.0, **selector},
            },
        },
        "encoder_text": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "encoder_image": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "fusion": {"name": "late_concat", "output_dim": 2 * _DIM},
        "manifold": {"name": "euclidean", "dim": 2 * _DIM},
        "training": {"epochs": 0},
        "logger": {"name": "none"},
        "corpus": {"root": "."},
        "device": "cpu",
        "seed": 0,
    }
    return Config.model_validate(raw)


def _first_candidate_adjudicator(
    dataset: DocumentDataset,
    requests: list[AdjudicationRequest],
    *,
    mode: str,
    batch_size: int,
) -> dict[str, AdjudicationVerdict]:
    """Verdict: reassign to the first candidate if any, else declare novel."""
    out: dict[str, AdjudicationVerdict] = {}
    for req in requests:
        if req.candidate_labels:
            out[req.doc_id] = AdjudicationVerdict(
                assignment=req.candidate_labels[0], confidence=0.77, rationale="closest"
            )
        else:
            out[req.doc_id] = AdjudicationVerdict(assignment=None, confidence=0.4, rationale="new")
    return out


def _all_novel_adjudicator(
    dataset: DocumentDataset,
    requests: list[AdjudicationRequest],
    *,
    mode: str,
    batch_size: int,
) -> dict[str, AdjudicationVerdict]:
    # Reversed on purpose: the framework must number novel buckets by document
    # order, not by whatever order the adjudicator happened to return.
    return {
        req.doc_id: AdjudicationVerdict(assignment=None, confidence=0.9)
        for req in reversed(requests)
    }


def test_disabled_consolidation_is_noop() -> None:
    raw = _s1_config("auto").model_dump()
    raw["scenario"]["consolidation"]["enabled"] = False
    scenario = build_scenario(Config.model_validate(raw))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)
    same = scenario.consolidate(result, ds, _first_candidate_adjudicator)
    assert same.predictions == result.predictions
    assert "consolidation" not in same.metadata


def test_consolidate_auto_applies_reassignments() -> None:
    scenario = build_scenario(_s1_config("auto"))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)
    consolidated = scenario.consolidate(result, ds, _first_candidate_adjudicator)

    meta = consolidated.metadata["consolidation"]
    assert meta["enabled"] is True
    assert meta["consolidated_by"] == "llm"
    assert meta["n_selected"] >= 1
    assert len(consolidated.predictions) == len(result.predictions)
    # auto mode writes the verdict confidence onto reassigned docs.
    assert 0.77 in [c for c in consolidated.confidence if c is not None]
    # Every verdict is recorded for audit, with where it came from.
    assert len(meta["verdicts"]) == meta["n_selected"]
    assert all(
        {"doc_id", "from", "to", "confidence", "rationale"} <= set(v) for v in meta["verdicts"]
    )


def test_consolidate_suggest_leaves_labels_but_flags_review() -> None:
    scenario = build_scenario(_s1_config("suggest"))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)
    consolidated = scenario.consolidate(result, ds, _first_candidate_adjudicator)

    # suggest mode: labels unchanged, but selected docs are flagged for review.
    assert consolidated.predictions == result.predictions
    assert any(consolidated.review)
    assert consolidated.metadata["consolidation"]["apply"] == "suggest"


def test_novel_buckets_dodge_the_downstream_cluster_rewrite() -> None:
    # S1 labels its clusters ``cluster_<n>`` and dgml-core rewrites those to
    # ``unknown_<n>`` on the way out. Minting ``unknown_0`` beside an existing
    # ``cluster_0`` would collide *after* that rewrite and silently merge two
    # unrelated groups, so novel numbering has to clear both spaces.
    scenario = build_scenario(_s1_config("auto"))
    ds = _MemDataset(4)
    result = scenario.fit_predict(ds)
    base = _result(
        ["cluster_0", "cluster_0", "cluster_1", "cluster_1"],
        [0.2, 0.9, 0.9, 0.9],
        result.embeddings[:4],
    )
    consolidated = scenario.consolidate(base, ds, _all_novel_adjudicator)

    minted = {p for p in consolidated.predictions if p and p.startswith("unknown_")}
    rewritten = {f"unknown_{lbl[len('cluster_') :]}" for lbl in ("cluster_0", "cluster_1")}
    assert minted.isdisjoint(rewritten)
    # Numbered by document order, so the mapping is reproducible run to run.
    assert consolidated.predictions[:2] == ["unknown_2", "unknown_3"]


def _one_novel_group_adjudicator(
    dataset: DocumentDataset,
    requests: list[AdjudicationRequest],
    *,
    mode: str,
    batch_size: int,
) -> dict[str, AdjudicationVerdict]:
    # Every selected doc is placed in ONE novel group (shared token) — the
    # repartition case "these documents are all the same new type".
    return {
        req.doc_id: AdjudicationVerdict(assignment=None, confidence=0.9, novel_group="g0")
        for req in requests
    }


def test_a_multi_member_novel_group_lands_in_one_bucket() -> None:
    # Repartition can keep several documents together as one new type. Members
    # sharing a ``novel_group`` token must mint a single ``unknown_N`` between
    # them, not one bucket each — otherwise the group is split back into
    # singletons, the exact opposite of what repartition is for.
    scenario = build_scenario(_s1_config("auto"))
    ds = _MemDataset(4)
    result = scenario.fit_predict(ds)
    base = _result(
        ["cluster_0", "cluster_0", "cluster_1", "cluster_1"],
        [0.2, 0.2, 0.2, 0.2],
        result.embeddings[:4],
    )
    consolidated = scenario.consolidate(base, ds, _one_novel_group_adjudicator)

    minted = [p for p in consolidated.predictions if p and p.startswith("unknown_")]
    assert len(minted) == 4  # all four were adjudicated novel
    assert len(set(minted)) == 1  # ...into a single shared bucket


def test_auto_mode_rederives_review_from_the_abstain_floor() -> None:
    # An LLM second opinion is not a human sign-off. A verdict the adjudicator
    # was itself unsure of must stay in the review queue.
    ds = _MemDataset(4)
    confident = build_scenario(_s1_config("auto", abstain_threshold=0.5))
    unsure = build_scenario(_s1_config("auto", abstain_threshold=0.9))
    result = confident.fit_predict(ds)

    # Verdict confidence is 0.77: above a 0.5 floor, below a 0.9 one.
    assert not any(confident.consolidate(result, ds, _first_candidate_adjudicator).review)
    assert all(unsure.consolidate(result, ds, _first_candidate_adjudicator).review)


def test_consolidate_noops_on_flat_confidence_with_note() -> None:
    # End-to-end: a flat confidence column + a partial quantile selector must
    # leave the partition untouched and explain why in metadata — this is the
    # guard that stops consolidation from *degrading* an already-good run.
    scenario = build_scenario(_s1_config("auto", quantile=0.2, include_noise=False))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)
    flat = _result(
        list(result.predictions),
        [1.0] * len(result.doc_ids),
        result.embeddings,
    )
    consolidated = scenario.consolidate(flat, ds, _first_candidate_adjudicator)

    assert consolidated.predictions == flat.predictions
    meta = consolidated.metadata["consolidation"]
    assert meta["n_selected"] == 0
    assert any("no confidence spread" in note for note in meta["notes"])


def test_consolidate_soft_fails_on_adjudicator_error() -> None:
    scenario = build_scenario(_s1_config("auto"))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)

    def _boom(dataset: Any, requests: Any, *, mode: str, batch_size: int) -> Any:
        raise RuntimeError("provider down")

    consolidated = scenario.consolidate(result, ds, _boom)
    # No raise; labels intact; the error is recorded in metadata.
    assert consolidated.predictions == result.predictions
    assert "provider down" in consolidated.metadata["consolidation"]["error"]


def test_consolidate_ignores_verdicts_for_documents_it_never_asked_about() -> None:
    scenario = build_scenario(_s1_config("auto"))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)

    def _extra(dataset: Any, requests: Any, *, mode: str, batch_size: int) -> Any:
        return {"not-a-real-doc-id": AdjudicationVerdict(assignment="whatever", confidence=1.0)}

    consolidated = scenario.consolidate(result, ds, _extra)
    assert consolidated.predictions == result.predictions
    assert consolidated.metadata["consolidation"]["n_reassigned"] == 0


def test_consolidate_does_not_mutate_the_input_result() -> None:
    # ``consolidate`` is a pure function of its input: the caller still holds
    # the pre-consolidation result and may want to compare or fall back to it.
    scenario = build_scenario(_s1_config("auto"))
    ds = _MemDataset(6)
    result = scenario.fit_predict(ds)
    before_preds = list(result.predictions)
    before_conf = list(result.confidence)
    before_review = list(result.review)
    before_meta = dict(result.metadata)

    scenario.consolidate(result, ds, _first_candidate_adjudicator)

    assert result.predictions == before_preds
    assert result.confidence == before_conf
    assert result.review == before_review
    assert result.metadata == before_meta
