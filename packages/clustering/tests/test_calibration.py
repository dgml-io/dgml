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

"""Tests for :mod:`clustering.calibration` and the abstain/review wiring.

Three properties carry most of the weight here:

1. **Calibration never changes an assignment.** Whatever the method, the
   predicted label is the nearest prototype; only the reported number and the
   review flag move. Several tests assert predictions are byte-identical with
   and without a calibrator.
2. **Leave-one-out means leave-one-out.** The calibration prototypes must be
   built the way inference builds them — same ``n_shots`` cap, same
   post-processing transform — and must exclude the document being scored.
3. **Degrading is not failing.** A support set too small to calibrate, an
   all-correct calibration set, an empty category: all ordinary on a small
   corpus, so all must return an identity/``None`` rather than raise.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from clustering.calibration import (
    Calibrator,
    fit_calibrator,
    fit_support_calibrator,
    support_loo_logits,
)
from clustering.config.schema import CalibrationConfig, ManifoldConfig, ScenarioConfig
from clustering.manifolds import build_manifold
from clustering.scenarios.clustering import assign_to_prototypes
from pydantic import ValidationError

_DIM = 4


def _euclidean(dim: int = _DIM) -> Any:
    return build_manifold(ManifoldConfig(name="euclidean", dim=dim, curvature=0.0))


def _labeled_support(per: int = 4, dim: int = _DIM) -> tuple[torch.Tensor, list[str | None]]:
    """Two clean classes, ``per`` samples each, separated along axis 0."""
    g = torch.Generator().manual_seed(7)
    a = torch.zeros(per, dim) + 0.05 * torch.randn(per, dim, generator=g)
    b = torch.zeros(per, dim) + 0.05 * torch.randn(per, dim, generator=g)
    b[:, 0] += 6.0
    labels: list[str | None] = [*["A"] * per, *["B"] * per]
    return torch.cat([a, b], dim=0), labels


def _mixed_logits(m: int = 40, k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits where the argmax is right most of the time but not always.

    An all-correct calibration set is degenerate for Platt (nothing to
    separate) and pushes the fitted temperature to the bound, so the
    parametric tests need a set with real errors in it.
    """
    g = torch.Generator().manual_seed(3)
    labels = torch.randint(0, k, (m,), generator=g)
    logits = torch.randn(m, k, generator=g)
    # Nudge the true class up by a margin small enough that ~a quarter of the
    # rows still land on the wrong argmax.
    logits[torch.arange(m), labels] += 1.0
    return logits, labels


# ── Calibrator: the identity default ───────────────────────────────────────
def test_default_calibrator_is_the_identity() -> None:
    logits = torch.tensor([[3.0, 1.0], [0.5, 0.4]], dtype=torch.float32)
    cal, abstain = Calibrator().apply(logits)

    expected = torch.softmax(logits, dim=-1).amax(dim=-1)
    assert torch.allclose(cal, expected, atol=1e-6)
    assert not bool(abstain.any())  # an unconfigured calibrator abstains on nothing


def test_apply_on_empty_batch_returns_empty() -> None:
    cal, abstain = Calibrator(method="temperature", temperature=2.0).apply(torch.zeros((0, 3)))
    assert cal.shape == (0,)
    assert abstain.shape == (0,)


def test_apply_output_stays_in_unit_range() -> None:
    logits, _ = _mixed_logits()
    for c in (
        Calibrator(method="temperature", temperature=0.1),
        Calibrator(method="temperature", temperature=50.0),
        Calibrator(method="platt", platt_a=12.0, platt_b=-9.0),
    ):
        cal, _ = c.apply(logits)
        assert float(cal.min()) >= 0.0
        assert float(cal.max()) <= 1.0


def test_absolute_floor_flags_only_documents_below_it() -> None:
    logits = torch.tensor([[9.0, 0.0], [0.1, 0.0]], dtype=torch.float32)
    _, abstain = Calibrator(abstain_threshold=0.9).apply(logits)
    # First row is a near-certain top1; second is nearly a coin flip.
    assert abstain.tolist() == [False, True]


def test_as_dict_round_trips_the_operating_point() -> None:
    c = Calibrator(method="platt", platt_a=2.0, platt_b=-1.0, coverage=0.9, conformal_threshold=0.3)
    d = c.as_dict()
    assert d["method"] == "platt"
    assert d["coverage"] == 0.9
    assert d["conformal_threshold"] == 0.3
    assert Calibrator(**d) == c  # provenance is enough to rebuild the calibrator


# ── fit_calibrator ─────────────────────────────────────────────────────────
def test_temperature_fit_improves_log_likelihood() -> None:
    logits, labels = _mixed_logits()
    fitted = fit_calibrator(logits, labels, method="temperature")

    def nll(t: float) -> float:
        logp = torch.log_softmax(logits / t, dim=-1)
        return float(-logp[torch.arange(logits.shape[0]), labels].mean())

    # A fit that does not beat T=1 is not a fit.
    assert nll(fitted.temperature) <= nll(1.0) + 1e-9


def test_platt_fit_is_monotone_in_the_raw_score() -> None:
    logits, labels = _mixed_logits()
    fitted = fit_calibrator(logits, labels, method="platt")
    assert fitted.method == "platt"

    # Platt is a 1-D logistic map on top1, so it must preserve the ranking:
    # a calibrated confidence is a rescaling, never a reordering.
    raw = torch.softmax(logits, dim=-1).amax(dim=-1)
    cal, _ = fitted.apply(logits)
    order_raw = torch.argsort(raw)
    assert torch.all(torch.diff(cal[order_raw]) >= -1e-6)


def test_conformal_threshold_achieves_requested_coverage() -> None:
    logits, labels = _mixed_logits(m=200)
    coverage = 0.9
    fitted = fit_calibrator(logits, labels, method="temperature", coverage=coverage)
    assert fitted.conformal_threshold is not None

    # The kept set is everything whose nonconformity is within q̂; by
    # construction of the split-conformal quantile, that is at least the
    # requested fraction of the calibration set.
    probs = torch.softmax(logits / max(fitted.temperature, 1e-2), dim=-1)
    kept = (1.0 - probs.amax(dim=-1)) <= fitted.conformal_threshold
    assert float(kept.float().mean()) >= coverage


def test_conformal_calibrates_the_statistic_it_thresholds() -> None:
    """q̂ must be fitted on ``1 - p_top1``, the score ``apply`` compares against.

    Fitting on the textbook ``1 - p(true)`` instead is silently self-defeating:
    ``p(true) <= p_top1`` with equality only on correct predictions, so the
    calibration set's *errors* push q̂ above anything the applied statistic can
    reach and the gate stops firing. Measured over four corpora, that put
    achieved coverage at 0.98-0.99 against a requested 0.90. This test fails if
    the two statistics ever drift apart again: on a calibration set with real
    errors the two q̂ values differ, and only the top1 one reproduces here.
    """
    logits, labels = _mixed_logits(m=200)
    fitted = fit_calibrator(logits, labels, method="temperature", coverage=0.9)
    assert fitted.conformal_threshold is not None

    probs = torch.softmax(logits / max(fitted.temperature, 1e-2), dim=-1)
    top1 = probs.amax(dim=-1)
    true_prob = probs[torch.arange(logits.shape[0]), labels]
    # The fixture has misassigned documents, so the two candidate scores are not
    # the same distribution — otherwise this test would pass either way.
    assert float((true_prob < top1 - 1e-9).float().mean()) > 0.05

    _, abstain = fitted.apply(logits)
    assert torch.equal(abstain, (1.0 - top1) > fitted.conformal_threshold)
    # And the gate is live: a q̂ fitted on the wrong statistic sits above the
    # applied score's range, which shows up as flagging nothing at all.
    assert int(abstain.sum()) > 0


def test_conformal_rejects_out_of_range_coverage() -> None:
    logits, labels = _mixed_logits(m=8)
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="coverage"):
            fit_calibrator(logits, labels, method="none", coverage=bad)


def test_too_few_points_degrades_to_identity() -> None:
    # One calibration point cannot support a temperature fit — degrade rather
    # than raise, so a two-document DocSet does not take the run down.
    fitted = fit_calibrator(
        torch.tensor([[1.0, 0.0]]), torch.tensor([0]), method="temperature", coverage=0.9
    )
    assert fitted.method == "none"
    assert fitted.temperature == 1.0
    assert fitted.conformal_threshold is None


def test_unknown_method_degrades_to_identity() -> None:
    logits, labels = _mixed_logits(m=10)
    fitted = fit_calibrator(logits, labels, method="bogus")  # type: ignore[arg-type]
    assert fitted.method == "none"


def test_platt_on_an_all_correct_set_is_the_identity_map() -> None:
    # No incorrect examples ⇒ the logistic fit is unidentifiable. It must fall
    # back to a pass-through rather than diverge; an easy corpus is the norm.
    logits = torch.tensor([[9.0, 0.0], [0.0, 9.0], [8.0, 0.0], [0.0, 8.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 0, 1])
    fitted = fit_calibrator(logits, labels, method="platt")
    assert fitted.platt_identity is True

    # The bug this guards: (a, b) = (1, 0) is NOT a pass-through — `apply` would
    # compute sigmoid(top1), squashing a 0.999 peak to ~0.73. A true identity
    # must report the raw top-1 softmax confidence unchanged.
    cal, _ = fitted.apply(logits)
    raw_top1 = torch.softmax(logits, dim=-1).amax(dim=-1)
    assert torch.allclose(cal, raw_top1, atol=1e-6)
    assert cal.max().item() > 0.9  # would be ~0.73 under the sigmoid bug


# ── support_loo_logits ─────────────────────────────────────────────────────
def test_loo_shape_and_labels() -> None:
    emb, labels = _labeled_support(per=4)
    out = support_loo_logits(emb, labels, ["A", "B"], _euclidean())
    assert out is not None
    logits, row_labels = out

    assert logits.shape == (8, 2)  # every sample yields one held-out row
    assert row_labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_loo_prototype_excludes_the_held_out_document() -> None:
    # Class A is one far outlier plus two tight points. Under leave-one-out the
    # outlier is scored against a prototype built from the tight pair only, so
    # it looks far from its own class — which is the whole point: fitting on
    # ordinary prototypes would let the outlier drag its own target toward it.
    manifold = _euclidean(dim=2)
    emb = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [4.0, 0.0], [9.0, 0.0], [9.1, 0.0]], dtype=torch.float32
    )
    labels: list[str | None] = ["A", "A", "A", "B", "B"]
    out = support_loo_logits(emb, labels, ["A", "B"], manifold)
    assert out is not None
    logits, _ = out

    # Row 2 is the outlier. Its own-class logit is -distance to mean(0.0, 0.1),
    # i.e. about -3.95 — it would be ~ -1.3 if it helped build the prototype.
    assert float(logits[2, 0]) == pytest.approx(-3.95, abs=1e-3)


def test_loo_honours_the_n_shots_cap() -> None:
    # Inference builds prototypes from the first ``n_shots`` support samples, so
    # calibration must too — otherwise the calibrator is fit against a model the
    # run never uses.
    manifold = _euclidean(dim=2)
    emb = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [8.0, 0.0], [20.0, 0.0], [21.0, 0.0]], dtype=torch.float32
    )
    labels: list[str | None] = ["A", "A", "A", "B", "B"]

    uncapped = support_loo_logits(emb, labels, ["A", "B"], manifold)
    capped = support_loo_logits(emb, labels, ["A", "B"], manifold, n_shots=2)
    assert uncapped is not None and capped is not None
    assert not torch.allclose(uncapped[0], capped[0])

    # The cap slides over the *reduced* class — exclude, then take the first
    # ``n_shots``. That is what leave-one-out means: rebuild the prototype the
    # way ``_build_prototypes`` would have if the document were never collected.
    # Holding out sample 0 leaves [1, 2] → prototype at mean(1.0, 8.0) = 4.5.
    assert float(capped[0][0, 0]) == pytest.approx(-4.5, abs=1e-4)
    # Holding out sample 2 — already outside the n_shots window — leaves [0, 1],
    # exactly the prototype inference uses.
    assert float(capped[0][2, 0]) == pytest.approx(-7.5, abs=1e-4)


def test_loo_honours_central_selection() -> None:
    # Same invariant as the n_shots-cap test but for the *selection* knob: the
    # calibrator must build its leave-one-out prototypes the way inference does.
    # Class A is an OUTLIER first, then a tight typical cluster. With n_shots=2,
    # "order" keeps the first two remaining (dragging in the outlier), while
    # "central" keeps the two nearest the post-exclusion mean (dropping it).
    manifold = _euclidean(dim=2)
    emb = torch.tensor(
        [
            [10.0, 0.0],  # A outlier, first in order
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [50.0, 0.0],  # B
            [50.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels: list[str | None] = ["A", "A", "A", "A", "B", "B"]

    order = support_loo_logits(emb, labels, ["A", "B"], manifold, n_shots=2, selection="order")
    central = support_loo_logits(emb, labels, ["A", "B"], manifold, n_shots=2, selection="central")
    assert order is not None and central is not None
    assert not torch.allclose(order[0], central[0])

    # Row 1 = holding out sample (0,0). "order" builds A's prototype from
    # {outlier, (0,1)} = (5, 0.5) → own-class logit ≈ -5.02; "central" builds it
    # from the two typical rows {(1,0),(0,1)} = (0.5, 0.5) → logit ≈ -0.707.
    assert float(order[0][1, 0]) == pytest.approx(-5.0249, abs=1e-3)
    assert float(central[0][1, 0]) == pytest.approx(-0.7071, abs=1e-3)


def test_loo_applies_the_prototype_transform() -> None:
    # S5 blends a name prototype into the support mean; calibration has to
    # replay that or it measures a different geometry than inference.
    emb, labels = _labeled_support(per=3)
    manifold = _euclidean()

    plain = support_loo_logits(emb, labels, ["A", "B"], manifold)
    shifted = support_loo_logits(
        emb, labels, ["A", "B"], manifold, prototype_transform=lambda p: p + 3.0
    )
    assert plain is not None and shifted is not None
    assert not torch.allclose(plain[0], shifted[0])


def test_loo_skips_singleton_categories_but_keeps_their_column() -> None:
    # A singleton has no leave-one-out prototype (nothing remains), so it
    # contributes no row — but it still supplies a column, so the class count
    # the calibrator sees matches the class count at inference.
    manifold = _euclidean(dim=2)
    emb = torch.tensor([[0.0, 0.0], [0.1, 0.0], [9.0, 0.0]], dtype=torch.float32)
    labels: list[str | None] = ["A", "A", "B"]
    out = support_loo_logits(emb, labels, ["A", "B"], manifold)
    assert out is not None
    logits, row_labels = out

    assert logits.shape == (2, 2)  # two A rows, no B row
    assert row_labels.tolist() == [0, 0]


def test_loo_drops_categories_with_no_samples() -> None:
    # An empty category must not be given a fabricated prototype: inventing one
    # at the held-out document's own position would make an unrelated class the
    # nearest prototype and poison the row.
    manifold = _euclidean(dim=2)
    emb = torch.tensor([[0.0, 0.0], [0.1, 0.0], [9.0, 0.0], [9.1, 0.0]], dtype=torch.float32)
    labels: list[str | None] = ["A", "A", "B", "B"]
    out = support_loo_logits(emb, labels, ["A", "B", "Ghost"], manifold)
    assert out is not None
    logits, _ = out

    assert logits.shape == (4, 2)  # "Ghost" gets no column at all
    # Every document still scores nearest to its own class.
    assert logits.argmax(dim=-1).tolist() == [0, 0, 1, 1]


def test_loo_ignores_unlabeled_and_out_of_vocabulary_samples() -> None:
    manifold = _euclidean(dim=2)
    emb = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [9.0, 0.0], [9.1, 0.0], [50.0, 0.0], [51.0, 0.0]],
        dtype=torch.float32,
    )
    labels: list[str | None] = ["A", "A", "B", "B", None, "NotACategory"]
    out = support_loo_logits(emb, labels, ["A", "B"], manifold)
    assert out is not None
    logits, row_labels = out

    assert logits.shape == (4, 2)
    assert row_labels.tolist() == [0, 0, 1, 1]


def test_loo_returns_none_when_nothing_usable() -> None:
    manifold = _euclidean(dim=2)
    # All singletons ⇒ no leave-one-out row can be built at all.
    emb = torch.tensor([[0.0, 0.0], [9.0, 0.0]], dtype=torch.float32)
    assert support_loo_logits(emb, ["A", "B"], ["A", "B"], manifold) is None
    assert support_loo_logits(torch.zeros((0, 2)), [], ["A"], manifold) is None


# ── fit_support_calibrator ─────────────────────────────────────────────────
def test_support_calibrator_is_none_when_nothing_asked_for() -> None:
    emb, labels = _labeled_support()
    # An abstain floor alone is not a reason to fit: it applies to the ordinal
    # confidence downstream just as well.
    assert (
        fit_support_calibrator(
            emb,
            labels,
            ["A", "B"],
            _euclidean(),
            method="none",
            coverage=None,
            abstain_threshold=0.5,
        )
        is None
    )


def test_support_calibrator_is_none_when_support_too_small() -> None:
    manifold = _euclidean(dim=2)
    emb = torch.tensor([[0.0, 0.0], [9.0, 0.0]], dtype=torch.float32)
    assert (
        fit_support_calibrator(
            emb,
            ["A", "B"],
            ["A", "B"],
            manifold,
            method="temperature",
            coverage=0.9,
            abstain_threshold=None,
        )
        is None
    )


def test_support_calibrator_carries_the_abstain_floor_through() -> None:
    emb, labels = _labeled_support(per=5)
    fitted = fit_support_calibrator(
        emb,
        labels,
        ["A", "B"],
        _euclidean(),
        method="temperature",
        coverage=0.8,
        abstain_threshold=0.6,
    )
    assert fitted is not None
    assert fitted.abstain_threshold == 0.6
    assert fitted.coverage == 0.8
    assert fitted.n_calibration == 10


# ── assign_to_prototypes: abstention is orthogonal to assignment ───────────
def _two_prototypes() -> tuple[torch.Tensor, torch.Tensor, Any]:
    manifold = _euclidean(dim=2)
    prototypes = torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32)
    embeddings = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [9.0, 0.0]], dtype=torch.float32)
    return embeddings, prototypes, manifold


def test_no_calibrator_reproduces_the_ordinal_confidence() -> None:
    embeddings, prototypes, manifold = _two_prototypes()
    result = assign_to_prototypes(embeddings, prototypes, manifold)

    assert result.calibrated_confidence is not None
    assert torch.allclose(result.calibrated_confidence, result.confidence, atol=1e-6)
    assert result.calibration is None
    assert result.abstain is not None
    assert not bool(result.abstain.any())


def test_abstain_threshold_without_a_calibrator_flags_the_boundary_doc() -> None:
    embeddings, prototypes, manifold = _two_prototypes()
    result = assign_to_prototypes(embeddings, prototypes, manifold, abstain_threshold=0.9)

    assert result.abstain is not None
    # Index 2 sits midway between the two prototypes — a genuine coin flip.
    assert bool(result.abstain[2])
    assert not bool(result.abstain[0])


def test_abstention_never_changes_the_predicted_label() -> None:
    embeddings, prototypes, manifold = _two_prototypes()
    plain = assign_to_prototypes(embeddings, prototypes, manifold)
    gated = assign_to_prototypes(
        embeddings,
        prototypes,
        manifold,
        calibrator=Calibrator(method="temperature", temperature=5.0, abstain_threshold=0.99),
        abstain_threshold=None,
    )

    # Every document is flagged, and not one label moved: review and novelty
    # are separate decisions and only novelty may write ``-1``.
    assert gated.abstain is not None
    assert bool(gated.abstain.all())
    assert plain.labels.tolist() == gated.labels.tolist()


def test_calibrator_and_explicit_floor_compose_with_or() -> None:
    embeddings, prototypes, manifold = _two_prototypes()
    # The conformal gate alone flags nothing here (huge threshold); the floor
    # alone flags the boundary document. Passing both must flag the union.
    result = assign_to_prototypes(
        embeddings,
        prototypes,
        manifold,
        calibrator=Calibrator(method="none", conformal_threshold=1.0, abstain_threshold=0.9),
    )
    assert result.abstain is not None
    assert result.abstain.tolist() == [False, False, True, False]
    assert result.calibration is not None
    assert result.calibration["method"] == "none"


# ── config validation ──────────────────────────────────────────────────────
def test_calibration_defaults_are_off() -> None:
    cal = ScenarioConfig(name="s1", k_clusters=2).calibration
    assert cal.method == "none"
    assert cal.coverage is None
    assert cal.abstain_threshold is None


def test_calibration_config_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError, match="coverage"):
        CalibrationConfig(method="temperature", coverage=1.0)
    with pytest.raises(ValidationError, match="abstain_threshold"):
        CalibrationConfig(abstain_threshold=1.5)


def test_calibration_config_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        CalibrationConfig(method="isotonic")  # type: ignore[arg-type]
