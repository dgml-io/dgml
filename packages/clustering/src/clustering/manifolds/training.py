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

"""Train a :class:`ManifoldProjector` with a manifold-aware loss.

The encoders and the fusion module are assumed frozen — only the
projector's linear (and, for the prototypical loss, the on-manifold
prototypes) are updated. This is the cheap, reproducible training regime:

- Run the corpus through encoders + fusion once.
- Train the projector for ``training.epochs`` steps against
  ``training.loss``:

  - **Supervised** (S3 / S5): ``contrastive``, ``triplet``,
    ``prototypical`` — consume corpus ground-truth labels.
  - **Unsupervised** (strict S1): ``pseudo_label`` (DeepCluster/PCL-style
    EM against Riemannian k-means pseudo-labels, optionally Sinkhorn-
    balanced), ``knn_contrastive`` (SCAN: kNN positives mined in the
    frozen fused space + learnable prototypes + entropy regularization),
    and ``cross_modal`` (InfoNCE between the text-only and image-only
    fused views of each document, passed via ``views=``). These never
    observe labels — not even folder names.

  Any loss can additionally carry a VICReg variance+covariance
  anti-collapse penalty (``vicreg_var_weight`` / ``vicreg_cov_weight``).
- Re-project all docs through the trained projector for the final
  scenario output.

Riemannian optimization
-----------------------

When ``geoopt`` is installed (it is in the workspace lockfile), the loop
uses :class:`geoopt.optim.RiemannianAdam`. The projector's linear weights
live in Euclidean ``R^{d x d}``, so RAdam updates them with the same step
as plain Adam — but for the prototypical loss the **prototypes** are
wrapped as :class:`geoopt.ManifoldParameter` on the active manifold
(:class:`geoopt.Euclidean` / :class:`geoopt.Sphere` /
:class:`geoopt.PoincareBall`) and updated along the manifold's
geodesics. That's where the Riemannian step actually matters.

When the projector was built with ``manifold_bias=True`` (gated on
``training.riemannian`` in the scenario pipeline), its on-manifold
**anchor** is also a :class:`geoopt.ManifoldParameter`; it arrives here
via ``projector.parameters()`` and gets the same geodesic update.

If ``geoopt`` is not importable we fall back to ``torch.optim.Adam`` plus
non-learnable per-epoch class-mean prototypes (the original behaviour).

Sampling per epoch is full-batch (the corpus fits in memory, by design),
which keeps training deterministic and easy to reason about. Per-batch
sampling can be slotted in later via the same loss interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn

from clustering.config.schema import UNSUPERVISED_LOSSES, TrainingConfig
from clustering.encoders.base import EncoderOutput
from clustering.fusion.base import Fusion
from clustering.manifolds.base import ManifoldHead
from clustering.manifolds.losses import (
    ContrastiveLoss,
    NeighborConsistencyLoss,
    PrototypicalLoss,
    TripletLoss,
    VICRegRegularizer,
)
from clustering.manifolds.projector import ManifoldProjector


class _FusionProjector(nn.Module):
    """Fusion → projector pipeline presented as a single trainable head.

    The supervised loop in :func:`train_projector` is agnostic to what it's
    optimizing: it only needs something callable that maps a feature matrix
    onto the manifold, exposes ``.manifold``, and yields ``.parameters()``.
    This wrapper re-runs the (trainable) fusion on cached per-modality
    pooled embeddings and then the projector, so gradients flow through
    *both* fusion and projector while the frozen encoders stay out of the
    graph entirely.

    Forward consumes a stacked ``[N, text_dim + image_dim]`` tensor (the
    concatenation of the two encoders' pooled outputs) and splits it back
    into the two modality views the fusion expects.
    """

    def __init__(self, fusion: Fusion, projector: ManifoldProjector, *, text_dim: int) -> None:
        super().__init__()
        self.fusion = fusion
        self.projector = projector
        self._text_dim = int(text_dim)

    @property
    def manifold(self) -> ManifoldHead:
        return self.projector.manifold

    def forward(self, stacked: torch.Tensor) -> torch.Tensor:
        text_pooled = stacked[..., : self._text_dim]
        image_pooled = stacked[..., self._text_dim :]
        fused = self.fusion(EncoderOutput(pooled=text_pooled), EncoderOutput(pooled=image_pooled))
        return torch.as_tensor(self.projector(fused.pooled))


# Either head can be driven by the supervised loop below.
_TrainableHead = ManifoldProjector | _FusionProjector


def train_projector(
    projector: _TrainableHead,
    fused: torch.Tensor,
    labels: Sequence[str | None],
    *,
    cfg: TrainingConfig,
    seed: int = 0,
    views: tuple[torch.Tensor, torch.Tensor] | None = None,
    pseudo_k: int | None = None,
) -> list[float]:
    """Train ``projector`` against ``fused`` / ``labels`` for ``cfg.epochs`` epochs.

    ``projector`` is usually a :class:`ManifoldProjector`, but any head that
    is callable, exposes ``.manifold`` / ``.parameters()`` and supports
    ``.train()`` / ``.eval()`` works — e.g. the :class:`_FusionProjector`
    pipeline used by :func:`train_fusion_projector` to update fusion and
    projector jointly.

    Args:
        projector: The :class:`ManifoldProjector` whose parameters to train.
            Must have at least one trainable parameter (otherwise this is
            a no-op and we raise).
        fused: ``[N, D_fused]`` pre-projection embeddings (encoder + fusion
            outputs). We never re-run the encoder during training.
        labels: ``[N]`` ground-truth labels. Entries that are ``None`` (or
            empty string) are excluded from the training loop. Ignored
            entirely when ``cfg.loss`` is unsupervised — labels never leak
            into the strict-S1 path.
        cfg: Resolved :class:`TrainingConfig`. ``cfg.loss`` selects the
            objective; ``cfg.epochs`` / ``cfg.lr`` / ``cfg.weight_decay``
            drive the optimiser.
        seed: Reproducibility seed for Adam state.
        views: ``(view_a, view_b)`` pair of ``[N, D_fused]`` tensors —
            required for ``loss='cross_modal'`` (text-only / image-only
            fused views of the same documents, row-aligned with ``fused``).
        pseudo_k: Override for the pseudo-cluster count used by
            ``pseudo_label`` / ``knn_contrastive``. Falls back to
            ``cfg.pseudo_k``, then 8.

    Returns:
        Per-epoch loss history (length ``cfg.epochs``). Empty if there
        are too few (labeled) samples to train on.

    Raises:
        ValueError: If ``projector`` has no trainable parameters, the loss
            name is not recognised, or ``cross_modal`` is requested without
            ``views``.
    """
    if cfg.epochs <= 0:
        return []

    params = list(projector.parameters())
    if not any(getattr(p, "requires_grad", True) for p in params):
        raise ValueError(
            "train_projector: projector has no trainable parameters. "
            "Construct it with trainable=True (or different input/output dims)."
        )

    unsupervised = cfg.loss in UNSUPERVISED_LOSSES
    label_ids = torch.zeros(0, dtype=torch.long)  # supervised losses only
    label_set: list[str] = []
    knn_idx: torch.Tensor | None = None
    view_a: torch.Tensor | None = None
    view_b: torch.Tensor | None = None

    if unsupervised:
        # Labels are deliberately ignored — every document trains. The
        # supervision signal comes from the data itself (cluster structure,
        # mined neighbors, or the two modality views).
        x_train = fused.detach()
        if x_train.shape[0] < 2:
            return []
    else:
        # Filter to labeled samples.
        keep: list[int] = []
        keep_labels: list[str] = []
        for i, lbl in enumerate(labels):
            if lbl is None or lbl == "":
                continue
            keep.append(i)
            keep_labels.append(lbl)
        if len(keep) < 2:
            return []

        label_set = sorted(set(keep_labels))
        if len(label_set) < 2:
            # Single-class training is degenerate for every supervised loss.
            return []

        label_to_idx = {lbl: i for i, lbl in enumerate(label_set)}
        label_ids = torch.tensor([label_to_idx[lbl] for lbl in keep_labels], dtype=torch.long)
        x_train = fused[torch.tensor(keep, dtype=torch.long)].detach()

    # Build the loss against the projector's underlying manifold.
    if cfg.loss == "contrastive":
        loss_fn: torch.nn.Module = ContrastiveLoss(projector.manifold, temperature=cfg.temperature)
    elif cfg.loss == "triplet":
        loss_fn = TripletLoss(projector.manifold, margin=cfg.margin)
    elif cfg.loss in ("prototypical", "pseudo_label"):
        loss_fn = PrototypicalLoss(projector.manifold)
    elif cfg.loss == "knn_contrastive":
        k = _resolve_pseudo_k(pseudo_k, cfg, n=int(x_train.shape[0]))
        with torch.no_grad():
            out_dim = int(projector(x_train[:1]).shape[-1])
        loss_fn = NeighborConsistencyLoss(
            projector.manifold,
            n_clusters=k,
            dim=out_dim,
            temperature=cfg.temperature,
            entropy_weight=cfg.entropy_weight,
            seed=seed,
        )
        # Neighbors are mined ONCE in the frozen fused (input) space — the
        # supervision signal must not drift with the projector.
        knn_idx = _knn_indices(x_train, k=cfg.knn_k)
    elif cfg.loss == "cross_modal":
        if views is None:
            raise ValueError(
                "train_projector: loss='cross_modal' requires views=(text_view, "
                "image_view) — see Scenario.unimodal_views()."
            )
        loss_fn = ContrastiveLoss(projector.manifold, temperature=cfg.temperature)
        view_a = views[0].detach()
        view_b = views[1].detach()
    else:
        raise ValueError(f"Unknown loss: {cfg.loss!r}")

    # ── On-manifold learnable prototypes (supervised prototypical only) ─────
    # When geoopt is available and the manifold supports `to_geoopt()`, the
    # prototypes are ManifoldParameters constrained to the active manifold;
    # RiemannianAdam updates them along geodesics. Otherwise we fall back to
    # recomputing them as the projected class-mean each epoch. Returns
    # ``(None, None)`` for every loss other than ``prototypical``.
    prototypes_param, geoopt_manifold = _maybe_build_prototypes(
        projector=projector,
        x_train=x_train,
        label_ids=label_ids,
        n_classes=len(label_set),
        loss_name=cfg.loss,
    )

    # ── Optimizer (Riemannian when available) ───────────────────────────────
    # Losses may carry their own trainable parameters (e.g. the SCAN
    # prototypes) — fold them in so the optimizer actually updates them.
    torch.manual_seed(seed)
    opt_params: list[torch.Tensor] = [*params, *loss_fn.parameters()]
    if prototypes_param is not None:
        opt_params.append(prototypes_param)
    optimizer = _build_optimizer(opt_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Optional VICReg anti-collapse penalty, additive on top of any loss.
    vicreg: VICRegRegularizer | None = None
    if cfg.vicreg_var_weight > 0.0 or cfg.vicreg_cov_weight > 0.0:
        vicreg = VICRegRegularizer(
            projector.manifold,
            var_weight=cfg.vicreg_var_weight,
            cov_weight=cfg.vicreg_cov_weight,
            gamma=cfg.vicreg_gamma,
        )

    # ── Training loop ──────────────────────────────────────────────────────
    history: list[float] = []
    pseudo_ids: torch.Tensor | None = None
    for epoch in range(cfg.epochs):
        projector.train()

        if cfg.loss == "pseudo_label":
            # E-step: re-derive pseudo-labels from the CURRENT projected
            # space at epoch 0 and every ``pseudo_recluster_every`` epochs.
            if epoch % max(1, cfg.pseudo_recluster_every) == 0:
                pseudo_ids = _pseudo_labels(
                    projector,
                    x_train,
                    k=_resolve_pseudo_k(pseudo_k, cfg, n=int(x_train.shape[0])),
                    cfg=cfg,
                    seed=seed + epoch,
                )
            assert pseudo_ids is not None
            # M-step: prototypical loss against pseudo-cluster prototypes.
            projected = projector(x_train)
            prototypes = _class_prototypes(projector, x_train, pseudo_ids)
            loss = loss_fn(projected, prototypes, pseudo_ids)
        elif cfg.loss == "knn_contrastive":
            projected = projector(x_train)
            assert knn_idx is not None
            # Rotate deterministically through the k mined neighbors.
            neighbors = projected[knn_idx[:, epoch % knn_idx.shape[1]]]
            loss = loss_fn(projected, neighbors)
        elif cfg.loss == "cross_modal":
            assert view_a is not None
            assert view_b is not None
            za = projector(view_a)
            zb = projector(view_b)
            projected = za  # for the (optional) VICReg term below
            loss = loss_fn(za, zb)
        elif cfg.loss == "prototypical":
            projected = projector(x_train)
            if prototypes_param is not None:
                # Learnable on-manifold prototypes.
                prototypes = prototypes_param
            else:
                # Fallback: recompute on-manifold prototypes each epoch.
                prototypes = _class_prototypes(projector, x_train, label_ids)
            loss = loss_fn(projected, prototypes, label_ids)
        elif cfg.loss == "contrastive":
            projected = projector(x_train)
            positives = _within_class_positives(projected, label_ids)
            loss = loss_fn(projected, positives)
        elif cfg.loss == "triplet":
            projected = projector(x_train)
            anchors, positives, negatives = _triplet_samples(projected, label_ids)
            loss = loss_fn(anchors, positives, negatives)
        else:  # pragma: no cover — guarded above
            raise ValueError(f"Unknown loss: {cfg.loss!r}")

        if vicreg is not None:
            loss = loss + vicreg(projected)

        loss_val = float(loss.item() if hasattr(loss, "item") else loss)
        history.append(loss_val)

        optimizer.zero_grad()
        if hasattr(loss, "backward"):
            loss.backward()
        optimizer.step()

    # Touch geoopt_manifold so type-checkers don't complain about the unused
    # binding — it lives on prototypes_param for the optimizer to use.
    del geoopt_manifold

    projector.eval()
    return history


def train_fusion_projector(
    fusion: Fusion,
    projector: ManifoldProjector,
    text_pooled: torch.Tensor,
    image_pooled: torch.Tensor,
    labels: Sequence[str | None],
    *,
    cfg: TrainingConfig,
    seed: int = 0,
    pseudo_k: int | None = None,
) -> list[float]:
    """Train the fusion module *and* the projector jointly.

    Unlike :func:`train_projector` — which consumes pre-fused embeddings and
    leaves the fusion frozen — this wraps ``fusion`` and ``projector`` into a
    single :class:`_FusionProjector` head and optimizes both under the same
    loss. The (frozen) encoders are not involved: their pooled outputs arrive
    here as ``text_pooled`` / ``image_pooled`` and are concatenated into the
    stacked feature matrix the wrapper splits back apart.

    Works for the supervised losses and for the label-free
    ``pseudo_label`` / ``knn_contrastive`` losses (their pseudo-labels,
    prototypes and mined neighbors are all computed *through* the wrapped
    pipeline, so gradients reach the fusion too). ``cross_modal`` is not
    supported here: its two modality views must themselves flow through the
    trainable fusion, which the stacked single-fusion wrapper can't express
    — train it with the fusion frozen via :func:`train_projector` instead.

    Args:
        fusion: The fusion module to train (must have parameters; a no-op
            otherwise).
        projector: The projector to train alongside it.
        text_pooled: ``[N, D_text]`` pooled text-encoder embeddings.
        image_pooled: ``[N, D_image]`` pooled image-encoder embeddings,
            row-aligned with ``text_pooled``.
        labels: ``[N]`` ground-truth labels (``None`` entries are skipped).
            Ignored when ``cfg.loss`` is unsupervised.
        cfg: Resolved :class:`TrainingConfig`.
        seed: Reproducibility seed.
        pseudo_k: Override for the pseudo-cluster count used by
            ``pseudo_label`` / ``knn_contrastive`` (forwarded to
            :func:`train_projector`).

    Returns:
        Per-epoch loss history (empty when there's nothing trainable or too
        few (labeled) samples).

    Raises:
        ValueError: If ``cfg.loss == 'cross_modal'`` (unsupported here).
    """
    if cfg.epochs <= 0:
        return []
    if cfg.loss == "cross_modal":
        raise ValueError(
            "train_fusion_projector: loss='cross_modal' cannot train the fusion jointly "
            "(its two modality views must each flow through the trainable fusion). Train "
            "the projector with the fusion frozen via train_projector instead."
        )
    pipeline = _FusionProjector(fusion, projector, text_dim=int(text_pooled.shape[-1]))
    if not any(getattr(p, "requires_grad", True) for p in pipeline.parameters()):
        # Neither fusion nor projector has trainable parameters (e.g. `none`
        # fusion + identity projector) — nothing to do.
        return []
    stacked = torch.cat([text_pooled, image_pooled], dim=-1)
    return train_projector(pipeline, stacked, labels, cfg=cfg, seed=seed, pseudo_k=pseudo_k)


def train_projector_cross_modal(
    projector: ManifoldProjector,
    text_view: torch.Tensor,
    image_view: torch.Tensor,
    *,
    cfg: TrainingConfig,
    seed: int = 0,
) -> list[float]:
    """Label-free CLIP-style training: align the two views of each document.

    ``text_view[i]`` and ``image_view[i]`` are the fused embeddings of
    document ``i`` with the *other* modality replaced by a constant
    placeholder. The symmetric InfoNCE loss
    (:class:`~clustering.manifolds.losses.ContrastiveLoss`, manifold
    distance as the logit) pulls a document's two views together and
    pushes other documents away — no labels required.

    Args:
        projector: The :class:`ManifoldProjector` whose parameters to train.
        text_view: ``[N, D_fused]`` text-only fused embeddings.
        image_view: ``[N, D_fused]`` image-only fused embeddings, row-aligned
            with ``text_view``.
        cfg: Resolved :class:`TrainingConfig`; ``epochs`` / ``lr`` /
            ``weight_decay`` / ``temperature`` apply, ``loss`` is ignored
            (the objective is fixed to symmetric InfoNCE).
        seed: Reproducibility seed for the optimizer state.

    Returns:
        Per-epoch loss history (length ``cfg.epochs``). Empty if there are
        fewer than two documents (no negatives to contrast against).

    Raises:
        ValueError: If ``projector`` has no trainable parameters or the two
            views disagree on shape.
    """
    if cfg.epochs <= 0:
        return []
    if text_view.shape != image_view.shape:
        raise ValueError(
            "train_projector_cross_modal: views must be row-aligned and equal-shaped; "
            f"got {tuple(text_view.shape)} vs {tuple(image_view.shape)}."
        )
    if text_view.shape[0] < 2:
        return []

    params = list(projector.parameters())
    if not any(getattr(p, "requires_grad", True) for p in params):
        raise ValueError(
            "train_projector_cross_modal: projector has no trainable parameters. "
            "Construct it with trainable=True (or different input/output dims)."
        )

    loss_fn = ContrastiveLoss(projector.manifold, temperature=cfg.temperature)
    torch.manual_seed(seed)
    optimizer = _build_optimizer(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_text = text_view.detach()
    x_image = image_view.detach()
    history: list[float] = []
    for _ in range(cfg.epochs):
        projector.train()
        loss = loss_fn(projector(x_text), projector(x_image))
        history.append(float(loss.item() if hasattr(loss, "item") else loss))
        optimizer.zero_grad()
        if hasattr(loss, "backward"):
            loss.backward()
        optimizer.step()

    projector.eval()
    return history


# ── Helpers ─────────────────────────────────────────────────────────────


def _resolve_pseudo_k(pseudo_k: int | None, cfg: TrainingConfig, *, n: int) -> int:
    """Pseudo-cluster count: explicit arg → ``cfg.pseudo_k`` → 8, clamped to [2, n]."""
    k = pseudo_k if pseudo_k is not None else (cfg.pseudo_k if cfg.pseudo_k is not None else 8)
    return max(2, min(k, n))


def _class_prototypes(
    projector: _TrainableHead,
    x_train: torch.Tensor,
    class_ids: torch.Tensor,
) -> torch.Tensor:
    """On-manifold prototypes: per-class ambient mean pushed through the projector.

    ``class_ids`` must be contiguous in ``0..C-1``. Defensively falls back
    to the overall mean for any empty class so prototype row ``c`` always
    aligns with class id ``c``.
    """
    n_classes = int(class_ids.max().item()) + 1
    protos: list[torch.Tensor] = []
    for cid in range(n_classes):
        mask = class_ids == cid
        if bool(mask.any() if hasattr(mask, "any") else mask.sum() > 0):
            class_mean = x_train[mask].mean(dim=0).unsqueeze(0)
        else:  # pragma: no cover — contiguous remapping should prevent this
            class_mean = x_train.mean(dim=0).unsqueeze(0)
        protos.append(projector(class_mean).squeeze(0))
    return torch.stack(protos, dim=0)


def _knn_indices(x: torch.Tensor, *, k: int) -> torch.Tensor:
    """``[N, k]`` cosine nearest-neighbor indices in ambient space, self excluded.

    Mined once on the frozen fused embeddings — these are SCAN's "pretext"
    neighbors and must stay fixed while the projector trains.
    """
    n = int(x.shape[0])
    k_eff = max(1, min(k, n - 1))
    xn = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    sim = xn @ xn.t()  # [N, N]
    sim = sim - torch.eye(n) * 1e9  # exclude self
    return cast(torch.Tensor, sim.topk(k_eff, dim=-1).indices)


def _sinkhorn(scores: torch.Tensor, *, n_iters: int, epsilon: float) -> torch.Tensor:
    """Sinkhorn-Knopp balanced soft assignment (SwAV, Caron et al. 2020).

    Args:
        scores: ``[N, K]`` similarity scores (higher = closer).
        n_iters: Row/column normalization rounds.
        epsilon: Entropic regularization temperature.

    Returns:
        ``[N, K]`` soft assignment whose column marginals are (approximately)
        uniform — i.e. every pseudo-cluster receives ~N/K documents.
    """
    q = torch.exp(scores / max(epsilon, 1e-8))
    q = q / q.sum()
    n, k = q.shape
    for _ in range(n_iters):
        # Columns sum to 1/K (balanced clusters)...
        q = q / q.sum(dim=0, keepdim=True).clamp(min=1e-12) / k
        # ...rows sum to 1/N (each doc fully assigned).
        q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-12) / n
    return q * n  # rows now sum to 1


def _pseudo_labels(
    projector: _TrainableHead,
    x_train: torch.Tensor,
    *,
    k: int,
    cfg: TrainingConfig,
    seed: int,
) -> torch.Tensor:
    """E-step of the pseudo-label EM loop: cluster the current projected space.

    Runs Riemannian k-means on ``projector(x_train)`` (no gradients), then —
    when ``cfg.sinkhorn`` — rebalances assignments with Sinkhorn-Knopp so no
    cluster starves or swallows the corpus. Returned ids are remapped to a
    contiguous ``0..C-1`` range so they can directly index prototype rows.
    """
    # Lazy import: scenarios.clustering imports manifolds.base, and the
    # scenarios package imports this module — resolving at call time keeps
    # the module graph acyclic.
    from clustering.scenarios.clustering import manifold_kmeans

    with torch.no_grad():
        z = projector(x_train)
        labels, centroids = manifold_kmeans(z, k, projector.manifold, seed=seed)
        if cfg.sinkhorn:
            scores = -projector.manifold.pairwise_dist(z, centroids)
            q = _sinkhorn(scores, n_iters=cfg.sinkhorn_iters, epsilon=cfg.sinkhorn_epsilon)
            labels = q.argmax(dim=-1)

    # Remap to contiguous ids (k-means / Sinkhorn may leave empty clusters).
    ids = labels.tolist() if hasattr(labels, "tolist") else list(labels)
    uniq = sorted({int(c) for c in ids})
    remap = {c: i for i, c in enumerate(uniq)}
    return torch.tensor([remap[int(c)] for c in ids], dtype=torch.long)


def _maybe_build_prototypes(
    *,
    projector: _TrainableHead,
    x_train: torch.Tensor,
    label_ids: torch.Tensor,
    n_classes: int,
    loss_name: str,
) -> tuple[torch.Tensor | None, object | None]:
    """Build learnable on-manifold prototypes if conditions are met.

    Returns ``(prototypes_param, geoopt_manifold)``. Both are ``None`` when
    we should fall back to per-epoch class-mean prototypes (e.g. geoopt is
    not installed, the manifold doesn't expose ``to_geoopt``, or the loss
    isn't prototypical).
    """
    if loss_name != "prototypical":
        return None, None
    try:
        import geoopt
    except ImportError:
        return None, None
    try:
        geoopt_manifold = projector.manifold.to_geoopt()
    except NotImplementedError:
        return None, None

    # Initial prototypes: class-mean → projected. Detached so the optimizer
    # owns the gradient flow into the prototype tensor, not into x_train.
    init = []
    with torch.no_grad():
        for cid in range(n_classes):
            mask = label_ids == cid
            class_mean = x_train[mask].mean(dim=0).unsqueeze(0)
            init.append(projector(class_mean).squeeze(0))
    init_tensor = torch.stack(init, dim=0).detach().clone()

    # Wrap as a ManifoldParameter on the active geoopt manifold. RAdam
    # detects the manifold attribute and routes the update through it.
    prototypes_param = geoopt.ManifoldParameter(init_tensor, manifold=geoopt_manifold)
    return prototypes_param, geoopt_manifold


def _build_optimizer(
    params: Sequence[torch.Tensor],
    *,
    lr: float,
    weight_decay: float,
) -> Any:
    """Pick the best optimizer available.

    Priority:
      1. ``geoopt.optim.RiemannianAdam`` — Riemannian update for any
         ``ManifoldParameter`` in ``params``; plain Adam step for the rest.
      2. ``torch.optim.Adam`` — the original behaviour.
      3. A sandbox-only no-op optimizer (when torch.optim isn't present).
    """
    try:
        import geoopt

        return geoopt.optim.RiemannianAdam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            stabilize=1,  # re-project params back to the manifold every step
        )
    except ImportError:
        pass
    try:
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    except AttributeError:
        return _NoopOptimizer(params)


class _NoopOptimizer:
    """No-op stand-in for sandbox runs that don't have torch.optim available."""

    def __init__(self, params: Sequence[torch.Tensor]) -> None:
        self.params = list(params)

    def zero_grad(self) -> None:
        return None

    def step(self) -> None:
        return None


def _within_class_positives(projected: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    """For each anchor, return a same-class positive (cyclic shift within class)."""
    ids = label_ids.tolist() if hasattr(label_ids, "tolist") else list(label_ids)
    positives: list[int] = list(range(len(ids)))
    by_class: dict[int, list[int]] = {}
    for i, c in enumerate(ids):
        by_class.setdefault(int(c), []).append(i)
    for indices in by_class.values():
        for k, src in enumerate(indices):
            positives[src] = indices[(k + 1) % len(indices)]
    return projected[torch.tensor(positives, dtype=torch.long)]


def _triplet_samples(
    projected: torch.Tensor, label_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (anchor, positive, negative) triplets deterministically."""
    ids = label_ids.tolist() if hasattr(label_ids, "tolist") else list(label_ids)
    by_class: dict[int, list[int]] = {}
    for i, c in enumerate(ids):
        by_class.setdefault(int(c), []).append(i)
    classes = sorted(by_class)
    if len(classes) < 2:
        # Caller must guarantee ≥2 classes; if not, degrade gracefully.
        return projected, projected, projected
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    for ci, c in enumerate(classes):
        same = by_class[c]
        other = by_class[classes[(ci + 1) % len(classes)]]
        for k, src in enumerate(same):
            anchors.append(src)
            positives.append(same[(k + 1) % len(same)])
            negatives.append(other[k % len(other)])
    return (
        projected[torch.tensor(anchors, dtype=torch.long)],
        projected[torch.tensor(positives, dtype=torch.long)],
        projected[torch.tensor(negatives, dtype=torch.long)],
    )
