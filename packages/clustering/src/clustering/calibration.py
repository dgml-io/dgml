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

"""Confidence calibration and abstention for nearest-prototype assignment.

The raw nearest-prototype signal (peak of ``softmax(-distance)``) is a fine
*ordinal* score — bigger means more confident — but it is not a probability.
Softmax over manifold distances is arbitrarily peaked depending on the distance
scale, so ``0.9`` from one run does not mean the same thing as ``0.9`` from
another, and neither means "90% of such assignments are correct". Anywhere a
misfiled document has real consequences, the number a reviewer sees has to mean
something, and there has to be a principled way to say "don't auto-file this
one".

Three standard, dependency-light pieces (all already-declared deps — numpy /
scipy / scikit-learn / torch; no LLM dependency lives here):

- **Temperature scaling** — a single scalar ``T`` fit by minimizing the
  negative log-likelihood of ``softmax(logits / T)`` against held-out labels.
  The cheapest and most robust post-hoc multiclass calibrator (Guo et al.,
  2017), and the one to reach for first: it cannot reorder predictions, so it
  changes the reported number without changing which category wins.
- **Platt scaling** — a 1-D logistic map ``sigmoid(a·s + b)`` fit on the top-1
  score ``s`` against assignment correctness. Recalibrates the reported
  confidence directly, and unlike temperature can correct a systematic
  over-confidence that is not a pure scale effect.
- **Conformal abstention** — a distribution-free split-conformal threshold on
  the nonconformity score ``1 - p_top1``. Given a target ``coverage`` it yields
  ``q̂`` such that routing every document with ``1 - p_top1 > q̂`` to a review
  queue keeps at most ``1 - coverage`` of a comparable batch, with no
  distributional assumptions. This is the piece that turns "confidence" into an
  operational decision: a review *budget* you can staff.

  The statistic is ``1 - p_top1`` and not the textbook ``1 - p(true)`` because
  the true label is exactly what is unavailable at inference. Calibrating on
  ``1 - p(true)`` while thresholding ``1 - p_top1`` — which an earlier revision
  of this module did — silently voids the guarantee: ``p(true) <= p_top1``
  always, with equality only on correct predictions, so the calibration set's
  errors inflate ``q̂`` past a value the applied statistic can reach and the gate
  stops firing. Measured over four corpora, that mismatch put achieved coverage
  at 0.98-0.99 against a requested 0.90 and cost 2-5x the review recall. The
  price of the fix is the weaker promise stated above: a budget on how much gets
  flagged, not a containment probability for the true label.

Everything here is fit on **labeled** data, and only the few-shot / supervised
scenarios (S3 / S5) have any — their per-DocSet support members. Fitting on
those naively would be self-flattering, since each document helps build the
prototype it is then scored against, so :func:`support_loo_logits` builds a
*leave-one-out* calibration set instead: each support document is scored
against prototypes recomputed without it. S1 / S2 have no labels at all; they
keep the ordinal signal and can still abstain via a plain confidence floor.

Nothing here changes which category a document is assigned to. Calibration
rescales the reported confidence; abstention flags a document for review. The
novelty gates in :func:`~clustering.scenarios.clustering.assign_to_prototypes`
are the only thing that moves a document to a different bucket.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from clustering.config.schema import SupportSelection
    from clustering.manifolds.base import ManifoldHead

CalibrationMethod = Literal["none", "temperature", "platt"]

# Bounds for the temperature search. T -> 0 sharpens toward argmax; large T
# flattens toward uniform. The optimum for well-behaved logits is near 1.
_T_MIN = 1e-2
_T_MAX = 1e2


def _softmax_np(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Row-wise numerically-stable softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return cast("np.ndarray[Any, Any]", exp / exp.sum(axis=-1, keepdims=True))


def _nll(temperature: float, logits: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    """Mean negative log-likelihood of ``softmax(logits / T)`` under ``labels``."""
    probs = _softmax_np(logits / max(temperature, _T_MIN))
    n = logits.shape[0]
    true = probs[np.arange(n), labels]
    return float(-np.log(np.clip(true, 1e-12, 1.0)).mean())


def _fit_temperature(logits: np.ndarray[Any, Any], labels: np.ndarray[Any, Any]) -> float:
    """Fit the temperature that minimizes NLL via bounded scalar search.

    One parameter over a bounded interval, so a derivative-free bounded search
    is both sufficient and immune to the initialization sensitivity an LBFGS
    fit would have here.
    """
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(_nll, args=(logits, labels), bounds=(_T_MIN, _T_MAX), method="bounded")
    t = float(getattr(res, "x", 1.0))
    return float(np.clip(t, _T_MIN, _T_MAX))


def _fit_platt(
    scores: np.ndarray[Any, Any], correct: np.ndarray[Any, Any]
) -> tuple[float, float, bool]:
    """Fit ``sigmoid(a·score + b) ≈ P(correct)`` via 1-D logistic regression.

    Returns ``(a, b, is_identity)``. Degenerate calibration sets — all-correct
    or all-wrong — carry no gradient for a logistic fit, so there is no map to
    learn: return ``is_identity=True`` and let :meth:`Calibrator.apply` report
    the raw top-1 confidence unchanged (the conformal gate still carries the
    review signal). Returning ``(1, 0)`` as if it were a pass-through would be a
    bug: ``sigmoid(1·s + 0)`` squashes ``[0, 1]`` into ``[0.5, 0.73]``. An
    all-correct support set is the common case on small, clean corpora, so this
    is a normal path, not an edge case.
    """
    if len({int(c) for c in correct.tolist()}) < 2:
        return 1.0, 0.0, True
    from sklearn.linear_model import LogisticRegression

    # C=1e6 ⇒ effectively unregularized: with a single feature and few dozen
    # points, shrinking the slope toward zero would just flatten the map.
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(scores.reshape(-1, 1), correct.astype(np.int64))
    a = float(model.coef_[0][0])
    b = float(model.intercept_[0])
    return a, b, False


def _conformal_threshold(nonconformity: np.ndarray[Any, Any], coverage: float) -> float:
    """Split-conformal quantile ``q̂`` for a target ``coverage`` in ``(0, 1)``.

    Keeping every point whose nonconformity is ``<= q̂`` gives at least
    ``coverage`` marginal coverage in finite samples (Vovk et al.), using the
    standard conservative rank ``ceil((n + 1)·coverage) / n``. The ``"higher"``
    interpolation keeps the guarantee one-sided rather than averaging across
    the two neighbouring order statistics.
    """
    n = int(nonconformity.shape[0])
    if n == 0:
        return 1.0
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(nonconformity, level, method="higher"))


def _to_numpy(t: torch.Tensor, dtype: Any) -> np.ndarray[Any, Any]:
    """Detach a tensor to numpy, tolerating the torch-less sandbox stub."""
    if hasattr(t, "detach"):
        return cast("np.ndarray[Any, Any]", t.detach().cpu().numpy().astype(dtype))
    return np.asarray(t, dtype=dtype)


@dataclass(frozen=True)
class Calibrator:
    """A fitted confidence calibrator plus an optional abstain gate.

    Immutable and cheap to carry on
    :attr:`~clustering.scenarios.base.ScenarioResult.metadata` as provenance
    (see :meth:`as_dict`) — which matters, because a calibrated confidence is
    only interpretable if you know what produced it. Apply it to a batch of
    assignment logits (``-distance`` to each prototype) with :meth:`apply`.

    The default is a deliberate identity: ``method="none"`` with no thresholds
    reproduces the uncalibrated ordinal confidence and abstains on nothing, so
    an unconfigured :class:`Calibrator` never changes behavior.
    """

    method: CalibrationMethod = "none"
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0
    # A degenerate Platt fit (all-correct / all-wrong support) has no logistic
    # map to learn. It must then report the raw top-1 confidence unchanged — NOT
    # ``sigmoid(a·top1 + b)`` with ``(a, b) = (1, 0)``, which is a real function
    # that squashes ``[0, 1]`` into ``[0.5, 0.73]`` and is the opposite of a
    # pass-through. This flag makes ``apply`` skip the sigmoid on that path.
    platt_identity: bool = False
    conformal_threshold: float | None = None
    coverage: float | None = None
    abstain_threshold: float | None = None
    n_calibration: int = 0

    def _tempered_softmax(self, logits_np: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        t = self.temperature if self.method == "temperature" else 1.0
        return _softmax_np(logits_np / max(t, _T_MIN))

    def apply(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(calibrated_confidence, abstain)`` for an ``[N, K]`` logit batch.

        ``calibrated_confidence`` is a float tensor in ``[0, 1]``; ``abstain``
        is a boolean ``[N]`` tensor flagging documents that fall below the
        conformal threshold or the absolute floor and should be routed to human
        review. The two gates compose with OR — a document abstains if *either*
        fires. An empty batch returns empty tensors.

        The conformal test is on the *uncalibrated* nonconformity
        ``1 - p_top1`` under the fitted temperature, because that is the scale
        ``q̂`` was estimated on; the Platt map (when active) rescales only the
        number reported to the caller.
        """
        n = int(logits.shape[0])
        if n == 0:
            return torch.zeros((0,), dtype=torch.float32), torch.zeros((0,), dtype=torch.bool)

        logits_np = _to_numpy(logits, np.float64)
        probs = self._tempered_softmax(logits_np)
        top1 = probs.max(axis=-1)

        if self.method == "platt" and not self.platt_identity:
            cal = 1.0 / (1.0 + np.exp(-(self.platt_a * top1 + self.platt_b)))
        else:
            cal = top1
        cal = np.clip(cal, 0.0, 1.0)

        abstain = np.zeros(n, dtype=bool)
        if self.conformal_threshold is not None:
            abstain |= (1.0 - top1) > self.conformal_threshold
        if self.abstain_threshold is not None:
            abstain |= cal < self.abstain_threshold

        return (
            torch.as_tensor(cal, dtype=torch.float32),
            torch.as_tensor(abstain, dtype=torch.bool),
        )

    def as_dict(self) -> dict[str, Any]:
        """Provenance dict for :class:`ScenarioResult` metadata / attestation."""
        return {
            "method": self.method,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "platt_identity": self.platt_identity,
            "conformal_threshold": self.conformal_threshold,
            "coverage": self.coverage,
            "abstain_threshold": self.abstain_threshold,
            "n_calibration": self.n_calibration,
        }


def fit_calibrator(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    method: CalibrationMethod = "temperature",
    coverage: float | None = None,
    abstain_threshold: float | None = None,
) -> Calibrator:
    """Fit a :class:`Calibrator` from labeled assignment logits.

    Args:
        logits: ``[M, K]`` assignment logits (``-distance`` to each of the
            ``K`` prototypes) for ``M`` labeled calibration documents.
        labels: ``[M]`` integer class indices in ``[0, K)``.
        method: ``"temperature"`` (default), ``"platt"``, or ``"none"``.
        coverage: If set in ``(0, 1)``, also fit a split-conformal abstain
            threshold targeting that coverage — i.e. a review budget: at most
            ``1 - coverage`` of a batch drawn like the calibration set is
            flagged. ``None`` disables conformal abstention.
        abstain_threshold: Optional absolute calibrated-confidence floor;
            documents below it abstain regardless of conformal coverage.

    Returns:
        A fitted :class:`Calibrator`. With fewer than two calibration points,
        or an unrecognized method, returns an identity (``method="none"``)
        calibrator so callers degrade to the ordinal signal rather than fail —
        a two-document DocSet should not take a run down.

    Raises:
        ValueError: If ``coverage`` is given and is not strictly inside
            ``(0, 1)``. A silent clamp here would hand back a guarantee the
            caller did not ask for.
    """
    if coverage is not None and not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1); got {coverage!r}.")

    logits_np = _to_numpy(logits, np.float64)
    labels_np = _to_numpy(labels, np.int64)
    m = int(logits_np.shape[0])
    if m < 2 or method not in ("none", "temperature", "platt"):
        return Calibrator(method="none", n_calibration=m)

    temperature = _fit_temperature(logits_np, labels_np) if method == "temperature" else 1.0

    platt_a, platt_b, platt_identity = 1.0, 0.0, False
    if method == "platt":
        probs = _softmax_np(logits_np)
        top1 = probs.max(axis=-1)
        pred = probs.argmax(axis=-1)
        correct = (pred == labels_np).astype(np.int64)
        platt_a, platt_b, platt_identity = _fit_platt(top1, correct)

    conformal_threshold: float | None = None
    if coverage is not None:
        cal_probs = _softmax_np(logits_np / max(temperature, _T_MIN))
        # Calibrate on the statistic `Calibrator.apply` actually thresholds. Using
        # `1 - p(true)` here instead would be the textbook nonconformity, but the
        # true label does not exist at inference, so `apply` has no choice but to
        # test `1 - p_top1` — and `p(true) <= p_top1`, so a q-hat fitted on the
        # former is systematically too large for the latter to reach. See the
        # module docstring for what that measured.
        conformal_threshold = _conformal_threshold(1.0 - cal_probs.max(axis=-1), coverage)

    return Calibrator(
        method=method,
        temperature=temperature,
        platt_a=platt_a,
        platt_b=platt_b,
        platt_identity=platt_identity,
        conformal_threshold=conformal_threshold,
        coverage=coverage,
        abstain_threshold=abstain_threshold,
        n_calibration=m,
    )


def support_loo_logits(
    support_embeddings: torch.Tensor,
    support_labels: list[str | None],
    categories: list[str],
    manifold: ManifoldHead,
    *,
    n_shots: int | None = None,
    selection: SupportSelection = "order",
    prototype_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Build leave-one-out calibration logits from a labeled support set.

    For each support document, its own category's prototype is recomputed with
    that document *excluded* (the ambient mean of the remaining same-class
    samples pushed onto the manifold), and the row is ``-distance`` to every
    category prototype. Fitting on the ordinary prototypes instead would be
    self-flattering: each document contributes to the prototype it is scored
    against, so it looks closer to its own class than a genuinely unseen
    document would, and the fitted temperature / conformal threshold would
    inherit that optimism.

    The prototype construction mirrors
    :meth:`~clustering.scenarios.base.Scenario._build_prototypes` exactly,
    ``n_shots`` capping included — a calibration set fit against a differently
    built prototype would be calibrating the wrong model.

    Categories with fewer than two support samples contribute no leave-one-out
    row (there is nothing left to form the prototype from) and are skipped as
    *held-out* documents, though they still supply a prototype *column* so the
    number of classes matches inference. Categories with no samples at all are
    dropped from the column space entirely rather than given a fabricated
    prototype; ``_build_prototypes`` raises on those, so they cannot occur at
    inference either.

    Args:
        support_embeddings: ``[M, D]`` on-manifold support embeddings.
        support_labels: Length-``M`` labels aligned with the embeddings.
        categories: Ordered category names. Column ``j`` of the returned logits
            corresponds to the ``j``-th category that has at least one sample.
        manifold: Active manifold head (``expmap0`` + ``pairwise_dist``).
        n_shots: Same per-category cap ``_build_prototypes`` applies, or
            ``None`` for no cap.
        selection: Same support-sample selection ``_support_prototypes`` applies
            (``"order"`` / ``"central"``). Must match inference or the LOO
            prototypes would be built differently from the ones the run scores
            against. Under leave-one-out, centrality is recomputed on the
            post-exclusion member set, mirroring how inference would build the
            prototype from those same rows.
        prototype_transform: Applied to the ``[K, D]`` leave-one-out prototype
            stack before scoring. Scenarios that post-process their prototypes
            — S5's name/support blend, for one — must pass the same transform
            they use at inference, or the calibration would be fit against
            prototypes the run never actually uses.

    Returns:
        ``(logits[M', K], labels[M'])``, or ``None`` when fewer than two usable
        rows remain — too little to calibrate anything, so the caller should
        keep the ordinal signal.
    """
    by_cat: dict[str, list[int]] = {c: [] for c in categories}
    for i, lbl in enumerate(support_labels):
        if lbl is not None and lbl in by_cat:
            by_cat[lbl].append(i)

    # Only categories with samples get a prototype column — matching what
    # `_build_prototypes` would produce (it raises on an empty category).
    columns = [c for c in categories if by_cat[c]]
    if not columns:
        return None
    cat_index = {c: j for j, c in enumerate(columns)}

    def _prototype(cat: str, exclude: int | None) -> torch.Tensor:
        members = [k for k in by_cat[cat] if k != exclude]
        if n_shots is not None and len(members) > n_shots:
            if selection == "central":
                # Recompute centrality on the post-exclusion set, mirroring how
                # _support_prototypes would build the prototype from these rows.
                m_emb = support_embeddings[torch.tensor(members)]
                dist = torch.linalg.norm(m_emb - m_emb.mean(dim=0, keepdim=True), dim=1)
                keep = torch.argsort(dist, stable=True)[:n_shots].tolist()
                members = [members[int(j)] for j in keep]
            else:  # "order"
                members = members[:n_shots]
        ambient_mean = support_embeddings[torch.tensor(members)].mean(dim=0)
        proto: torch.Tensor = manifold.expmap0(ambient_mean.unsqueeze(0)).squeeze(0)
        return proto

    rows: list[torch.Tensor] = []
    row_labels: list[int] = []
    for cat in columns:
        if len(by_cat[cat]) < 2:
            continue  # no leave-one-out prototype possible for a singleton
        for held_out in by_cat[cat]:
            protos = [
                _prototype(other, exclude=held_out if other == cat else None) for other in columns
            ]
            proto_stack = torch.stack(protos, dim=0)  # [K, D]
            if prototype_transform is not None:
                proto_stack = prototype_transform(proto_stack)
            query = support_embeddings[held_out].unsqueeze(0)  # [1, D]
            dist = manifold.pairwise_dist(query, proto_stack).squeeze(0)  # [K]
            rows.append(-dist)
            row_labels.append(cat_index[cat])

    if len(rows) < 2:
        return None
    return torch.stack(rows, dim=0), torch.tensor(row_labels, dtype=torch.long)


def fit_support_calibrator(
    support_embeddings: torch.Tensor,
    support_labels: list[str | None],
    categories: list[str],
    manifold: ManifoldHead,
    *,
    method: CalibrationMethod,
    coverage: float | None,
    abstain_threshold: float | None,
    n_shots: int | None = None,
    selection: SupportSelection = "order",
    prototype_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Calibrator | None:
    """Fit a calibrator from a labeled support set, or ``None`` if not possible.

    The wrapper the labeled scenarios (S3 / S5) call. Returns ``None`` — meaning
    "keep the ordinal confidence" — in two cases: nothing was asked for (no
    parametric method *and* no conformal coverage), or the support set is too
    small to build a leave-one-out calibration set. Both are ordinary
    situations on a small corpus, so neither raises. A plain
    ``abstain_threshold`` floor still applies downstream in either case, which
    is why it is not enough on its own to trigger a fit.
    """
    if method == "none" and coverage is None:
        return None
    loo = support_loo_logits(
        support_embeddings,
        support_labels,
        categories,
        manifold,
        n_shots=n_shots,
        selection=selection,
        prototype_transform=prototype_transform,
    )
    if loo is None:
        return None
    logits, labels = loo
    return fit_calibrator(
        logits,
        labels,
        method=method,
        coverage=coverage,
        abstain_threshold=abstain_threshold,
    )
