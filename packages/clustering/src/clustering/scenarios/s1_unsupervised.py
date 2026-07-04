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

"""S1 — fully unsupervised clustering.

No labels, no category names. We embed the whole corpus, then run a
manifold-aware clustering algorithm to discover groups. Two algorithms
are supported via :class:`ScenarioConfig.cluster_algorithm`:

- ``kmeans`` (default): Riemannian Lloyd's algorithm; requires
  ``k_clusters``. Predictions are ``cluster_<i>`` for ``i = 0..k-1``.
- ``hdbscan``: density-based, parameter-free in ``k``. The cluster
  count emerges from data density; noise points are routed to a
  ``cluster_noise`` bucket. ``k_clusters`` is ignored when set.

Trainable projector
-------------------
When ``training.epochs > 0`` and the projector has trainable parameters,
S1 trains the manifold projector *before* the unsupervised clustering
step. There are two ways to drive that training:

**Loss-driven (strict S1, label-free).** When ``training.loss`` is one of
the unsupervised losses (``pseudo_label`` / ``knn_contrastive`` /
``cross_modal``), the projector trains with NO access to labels — not even
folder names — and this takes priority over ``training.supervision``.
``pseudo_label`` alternates Riemannian k-means pseudo-labels (optionally
Sinkhorn-balanced) against a prototypical loss (DeepCluster/PCL-style EM);
``knn_contrastive`` is SCAN-style neighbor consistency with an entropy
anti-collapse regularizer; ``cross_modal`` aligns the text-only and
image-only fused views of each document with InfoNCE. Any of them can
carry an additive VICReg variance/covariance penalty.

**Supervision-driven.** Otherwise ``training.supervision`` picks
the signal:

- ``"labels"`` (default): ground-truth labels carried on the records
  (e.g. subfolder names — the ones NMI / ARI are computed against).
  Semi-supervised cluster discovery: labels shape the geometry, but the
  cluster count and assignments are still discovered, not dictated.
  Skipped when no labels are available.
- ``"pseudo_labels"``: fully label-free, DeepCluster-style. Cluster the
  current projection with manifold k-means, treat cluster ids as labels,
  train, optionally re-cluster and repeat (``training.pseudo_rounds``).
- ``"cross_modal"``: fully label-free, CLIP-style. A document's
  text-only and image-only fused views are positives under symmetric
  InfoNCE; all other documents are negatives.

In every mode, inference below remains label-free; the training signal
only shapes the embedding space.

The cascade can later rename clusters via :meth:`Scenario.refine` once
a human assigns category names.
"""

from __future__ import annotations

import torch

from clustering.config.schema import UNSUPERVISED_LOSSES
from clustering.data.datasets import DocumentDataset
from clustering.scenarios.base import Scenario, ScenarioResult
from clustering.scenarios.clustering import cluster_embeddings, reduce_embeddings


class S1Unsupervised(Scenario):
    name = "s1"

    # Fallback pseudo-label cluster count when neither
    # ``training.pseudo_k`` nor ``scenario.k_clusters`` is set.
    DEFAULT_PSEUDO_K = 8

    # Unsupervised losses whose training signal is computed *through* the
    # fusion+projector pipeline, so the fusion can be trained jointly with
    # ``--trainable-fusion``. ``cross_modal`` is excluded: its two modality
    # views must each flow through the trainable fusion, which the stacked
    # single-fusion wrapper can't express, so it stays fusion-frozen.
    _JOINT_FUSION_UNSUP_LOSSES = frozenset({"pseudo_label", "knn_contrastive"})

    def _trains_fusion_jointly(self) -> bool:
        """True when the unsupervised run should train fusion + projector jointly."""
        tr = self.config.training
        return (
            tr.epochs > 0
            and tr.trainable_fusion
            and tr.loss in self._JOINT_FUSION_UNSUP_LOSSES
            and any(True for _ in self.fusion.parameters())
        )

    def _train_unsupervised(
        self, unknown_dataset: DocumentDataset, fused: torch.Tensor
    ) -> list[float]:
        """Strict-S1 label-free projector training selected by ``training.loss``.

        Routes to :func:`train_projector` with the labels stripped to
        ``[None] * N`` so no future code path can consume them. For
        ``cross_modal`` the two unimodal fused views are mined first and
        passed via ``views=``. When ``--trainable-fusion`` is set and the
        loss is one of :attr:`_JOINT_FUSION_UNSUP_LOSSES`, the fusion is
        trained jointly with the projector instead (see
        :meth:`_train_unsupervised_with_fusion`). A no-op (empty history)
        when training is disabled, the projector has no trainable params, or
        there are too few documents.
        """
        from clustering.manifolds.training import train_projector

        tr = self.config.training
        if tr.epochs <= 0 or int(fused.shape[0]) < 2:
            return []
        if not any(True for _ in self.projector.parameters()):
            return []

        if self._trains_fusion_jointly():
            return self._train_unsupervised_with_fusion(unknown_dataset)

        views: tuple[torch.Tensor, torch.Tensor] | None = None
        if tr.loss == "cross_modal":
            _, text_view, image_view = self.unimodal_views(unknown_dataset)
            views = (text_view, image_view)

        n = int(fused.shape[0])
        return train_projector(
            self.projector,
            fused,
            [None] * n,  # strict S1: labels are never observed during training
            cfg=tr,
            seed=self.config.seed,
            views=views,
            pseudo_k=self.config.scenario.k_clusters,
        )

    def _train_unsupervised_with_fusion(self, dataset: DocumentDataset) -> list[float]:
        """Label-free joint fusion+projector training (pseudo_label / knn_contrastive).

        Re-encodes the dataset into per-modality pooled embeddings (frozen
        encoders, ``no_grad``) and trains the fusion module together with the
        projector under the configured unsupervised loss — the pseudo-labels,
        prototypes and mined neighbors are all computed through the wrapped
        pipeline, so gradients reach the fusion. The caller must re-embed
        afterwards, since the fusion weights have changed.
        """
        from clustering.manifolds.training import train_fusion_projector

        _, text_pooled, image_pooled, _ = self.modality_pooled(dataset)
        n = int(text_pooled.shape[0])
        if n < 2:
            return []
        return train_fusion_projector(
            self.fusion,
            self.projector,
            text_pooled,
            image_pooled,
            [None] * n,  # strict S1: labels are never observed during training
            cfg=self.config.training,
            seed=self.config.seed,
            pseudo_k=self.config.scenario.k_clusters,
        )

    def _train_with_pseudo_labels(self, fused: torch.Tensor) -> list[float]:
        """DeepCluster-style label-free training rounds.

        Each round: project ``fused`` with the current projector, run
        manifold k-means, use the cluster ids as pseudo-labels, and train
        the projector for ``training.epochs`` epochs against them. The
        next round re-clusters the *improved* projection, so labels and
        geometry co-evolve. Returns the concatenated loss history across
        rounds (empty when training is impossible or degenerate).
        """
        tr = self.config.training
        if tr.epochs <= 0 or int(fused.shape[0]) < 2:
            return []
        if not any(True for _ in self.projector.parameters()):
            return []

        k = tr.pseudo_k or self.config.scenario.k_clusters or self.DEFAULT_PSEUDO_K
        k = min(k, int(fused.shape[0]))

        history: list[float] = []
        for round_i in range(max(1, tr.pseudo_rounds)):
            with torch.no_grad():
                embeddings = self.projector(fused)
            labels_t, _ = cluster_embeddings(
                embeddings,
                manifold=self.manifold,
                algorithm="kmeans",
                k=k,
                seed=self.config.seed + round_i,
            )
            labels_arr = labels_t.detach().numpy() if hasattr(labels_t, "numpy") else labels_t
            pseudo: list[str | None] = [f"pseudo_{int(li)}" for li in labels_arr.tolist()]
            if len({p for p in pseudo if p is not None}) < 2:
                # Degenerate single cluster — no contrastive signal left.
                break
            history.extend(self.maybe_train_projector(fused, pseudo))
        return history

    def fit_predict(
        self,
        unknown_dataset: DocumentDataset,
        support_dataset: DocumentDataset | None = None,
    ) -> ScenarioResult:
        # S1 ignores ``support_dataset`` — cluster assignment is fully
        # unsupervised. (Labels on the *unknown* dataset's own records may
        # still train the projector below; they never leak into inference.)
        del support_dataset

        # Pre-projection embeddings — feed these to the projector trainer
        # so we don't re-run the encoders after training.
        doc_ids, fused, true_labels = self.fused_embeddings(unknown_dataset)

        # Optionally train the projector before clustering. The
        # supervision source is configurable; every branch degrades
        # safely to a no-op (empty history) when ``epochs == 0``, when
        # the projector has no trainable params, or when its signal is
        # unavailable (no labels / too few documents).
        supervision = self.config.training.supervision
        unsupervised_loss = self.config.training.loss in UNSUPERVISED_LOSSES
        if unsupervised_loss:
            # Loss-driven strict-S1 training takes priority over the
            # ``supervision`` field — the projector trains label-free.
            fusion_trained = self._trains_fusion_jointly()
            train_history = self._train_unsupervised(unknown_dataset, fused)
            if fusion_trained:
                # The fusion weights changed, so the pre-training ``fused``
                # above is stale — re-embed before the label-free clustering.
                _, fused, _ = self.fused_embeddings(unknown_dataset)
        elif supervision == "pseudo_labels":
            train_history = self._train_with_pseudo_labels(fused)
        elif supervision == "cross_modal":
            train_history = self.maybe_train_projector_cross_modal(unknown_dataset)
        elif self.config.training.trainable_fusion:
            # "labels" supervision, fusion trained jointly with the projector.
            # The fusion weights change, so the pre-training ``fused`` above
            # is stale — re-embed before the label-free clustering step.
            train_history = self.maybe_train_fusion_and_projector(unknown_dataset)
            _, fused, _ = self.fused_embeddings(unknown_dataset)
        else:  # "labels" — ground truth carried on the records, if any.
            train_history = self.maybe_train_projector(fused, true_labels)

        # Final on-manifold embeddings come from the (possibly trained)
        # projector. Inference below uses only ``embeddings`` — labels
        # never leak into the cluster assignment step.
        embeddings = self.projector(fused)
        sc = self.config.scenario
        algorithm = sc.cluster_algorithm

        # K is required for k-means and ignored for hdbscan; only fall
        # back to the heuristic when we'll actually consume it.
        k = sc.k_clusters
        if algorithm == "kmeans" and k is None:
            # Heuristic fallback: if the corpus carries labels, use that
            # count; otherwise default to 8.
            unique_labels = {label for label in true_labels if label is not None}
            k = len(unique_labels) if unique_labels else 8

        # Optional dimensionality reduction before clustering. On high-dim
        # embeddings, density-based clustering collapses to all-noise; when a
        # reducer is configured we cluster in the reduced Euclidean space
        # (distances there are density-friendly), independent of the
        # representation manifold. The stored ``embeddings`` stay original.
        cluster_input = embeddings
        cluster_manifold = self.manifold
        if sc.reduce_method != "none" and sc.reduce_dim > 0:
            cluster_input = reduce_embeddings(
                embeddings,
                method=sc.reduce_method,
                n_components=sc.reduce_dim,
                seed=self.config.seed,
            )
            if cluster_input is not embeddings:
                from clustering.config.schema import ManifoldConfig
                from clustering.manifolds import build_manifold

                cluster_manifold = build_manifold(
                    ManifoldConfig(name="euclidean", dim=int(cluster_input.shape[-1]))
                )

        labels_t, centroids = cluster_embeddings(
            cluster_input,
            manifold=cluster_manifold,
            algorithm=algorithm,
            k=k,
            seed=self.config.seed,
            min_cluster_size=sc.hdbscan_min_cluster_size,
            min_samples=sc.hdbscan_min_samples,
            cluster_selection_epsilon=sc.hdbscan_cluster_selection_epsilon,
            cluster_selection_method=sc.hdbscan_cluster_selection_method,
            allow_single_cluster=sc.hdbscan_allow_single_cluster,
            graph_cc_radius=sc.graph_cc_radius,
            graph_cc_r_method=sc.graph_cc_r_method,
            graph_cc_k_neighbors=sc.graph_cc_k_neighbors,
            graph_cc_min_cluster_size=sc.graph_cc_min_cluster_size,
            leiden_graph_method=sc.leiden_graph_method,
            leiden_k_neighbors=sc.leiden_k_neighbors,
            leiden_radius=sc.leiden_radius,
            leiden_r_method=sc.leiden_r_method,
            leiden_quality=sc.leiden_quality,
            leiden_resolution=sc.leiden_resolution,
            leiden_min_cluster_size=sc.leiden_min_cluster_size,
            leiden_n_iterations=sc.leiden_n_iterations,
            dbscan_eps=sc.dbscan_eps,
            dbscan_r_method=sc.dbscan_r_method,
            dbscan_k_neighbors=sc.dbscan_k_neighbors,
            dbscan_min_samples=sc.dbscan_min_samples,
            dbscan_min_cluster_size=sc.dbscan_min_cluster_size,
            optics_min_samples=sc.optics_min_samples,
            optics_xi=sc.optics_xi,
            optics_min_cluster_size=sc.optics_min_cluster_size,
            affinity_damping=sc.affinity_damping,
            affinity_preference=sc.affinity_preference,
            affinity_max_iter=sc.affinity_max_iter,
            affinity_convergence_iter=sc.affinity_convergence_iter,
            meanshift_bandwidth=sc.meanshift_bandwidth,
            meanshift_quantile=sc.meanshift_quantile,
            meanshift_bin_seeding=sc.meanshift_bin_seeding,
            meanshift_cluster_all=sc.meanshift_cluster_all,
        )
        # Pull labels out as Python ints. (Stub-friendly conversion.)
        labels_arr = labels_t.detach().numpy() if hasattr(labels_t, "numpy") else labels_t
        labels_list = [int(li) for li in labels_arr.tolist()]
        # HDBSCAN emits ``-1`` for noise; render that as a distinct bucket
        # so consumers don't have to special-case negative integers in
        # category strings.
        predictions: list[str | None] = [
            "cluster_noise" if li == -1 else f"cluster_{li}" for li in labels_list
        ]
        confidence: list[float | None] = [None] * len(doc_ids)

        n_noise = sum(1 for li in labels_list if li == -1)
        n_clusters_found = int(centroids.shape[0])

        return ScenarioResult(
            run_id=self.run_id,
            scenario_name=self.name,
            doc_ids=doc_ids,
            embeddings=embeddings,
            predictions=predictions,
            confidence=confidence,
            true_labels=true_labels,
            metadata={
                "k_clusters": k,
                "n_clusters_found": n_clusters_found,
                "n_noise": n_noise,
                "centroids_shape": tuple(centroids.shape),
                "algorithm": algorithm,
                "reduce_method": sc.reduce_method,
                "reduce_dim": (
                    int(cluster_input.shape[-1]) if cluster_input is not embeddings else 0
                ),
                # Mirror S3/S5: surface training info so downstream
                # tools (run comparison, model selection) can plot
                # convergence and rank trained vs untrained runs.
                "projector_trained": bool(train_history),
                "projector_loss_history": train_history,
                "projector_unsupervised": unsupervised_loss,
                "projector_loss": self.config.training.loss,
                "supervision": supervision,
            },
        )
