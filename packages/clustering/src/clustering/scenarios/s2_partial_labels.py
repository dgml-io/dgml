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

"""S2 — partial labels (some categories known, no samples).

For each document we score against the known-category prototypes (built
from category names à la S4). Documents whose nearest-prototype distance
exceeds the configured threshold are pushed into an "unknown" bucket and
clustered separately, with labels ``unknown_<i>``.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from clustering.data.datasets import DocumentDataset
from clustering.scenarios.base import Scenario, ScenarioResult
from clustering.scenarios.clustering import assign_to_prototypes, cluster_embeddings


class S2PartialLabels(Scenario):
    name = "s2"

    PROMPT_TEMPLATE: ClassVar[str] = "a scanned document of category: {category}"

    def fit_predict(
        self,
        unknown_dataset: DocumentDataset,
        support_dataset: DocumentDataset | None = None,
    ) -> ScenarioResult:
        # S2 builds prototypes from category names alone; no labeled
        # samples are consumed.
        del support_dataset
        cats = self.config.scenario.known_categories
        if not cats:
            raise ValueError("S2 requires scenario.known_categories to be non-empty.")

        # ── Build known-category prototypes from names ───────────────────
        prompts = [self.PROMPT_TEMPLATE.format(category=c) for c in cats]
        known_protos = self.encode_texts(prompts)

        # ── Embed corpus + initial assignment with composable gates ──────
        doc_ids, embeddings, true_labels = self.embed(unknown_dataset)
        sc = self.config.scenario
        result = assign_to_prototypes(
            embeddings,
            known_protos,
            self.manifold,
            threshold=sc.threshold,
            threshold_confidence=sc.threshold_confidence,
            threshold_quantile=sc.threshold_quantile,
        )
        labels_t, conf_t = result.labels, result.confidence

        labels_arr = labels_t.detach().numpy() if hasattr(labels_t, "numpy") else labels_t
        conf_arr = conf_t.detach().numpy() if hasattr(conf_t, "numpy") else conf_t

        # ── Cluster the unassigned bucket into emergent categories ───────
        predictions: list[str | None] = [None] * len(doc_ids)
        confidence: list[float | None] = [None] * len(doc_ids)
        unknown_idx = [i for i, li in enumerate(labels_arr.tolist()) if int(li) == -1]
        n_unknown = len(unknown_idx)

        if n_unknown >= 2:
            unknown_emb = embeddings[torch.tensor(unknown_idx)]
            # For k-means we still need an explicit k; for HDBSCAN it's
            # ignored. The same dispatcher used by S1 keeps the unknown
            # bucket honouring ``scenario.cluster_algorithm`` end-to-end.
            k_unknown = max(2, min(8, n_unknown))
            ulabels_t, _ = cluster_embeddings(
                unknown_emb,
                manifold=self.manifold,
                algorithm=sc.cluster_algorithm,
                k=k_unknown,
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
            ulabels_arr = ulabels_t.detach().numpy() if hasattr(ulabels_t, "numpy") else ulabels_t
            for src, dst in zip(ulabels_arr.tolist(), unknown_idx, strict=True):
                src_i = int(src)
                predictions[dst] = "unknown_noise" if src_i == -1 else f"unknown_{src_i}"
                confidence[dst] = None
        elif n_unknown == 1:
            predictions[unknown_idx[0]] = "unknown_0"

        # Fill in the known-assignment predictions.
        for i, li in enumerate(labels_arr.tolist()):
            if int(li) != -1:
                predictions[i] = cats[int(li)]
                confidence[i] = float(conf_arr[i])

        return ScenarioResult(
            run_id=self.run_id,
            scenario_name=self.name,
            doc_ids=doc_ids,
            embeddings=embeddings,
            predictions=predictions,
            confidence=confidence,
            true_labels=true_labels,
            metadata={
                "categories": list(cats),
                # Echo the user-supplied gate config + the effective
                # post-calibration values, so `compare_runs` and the UI
                # can reconstruct the operating point.
                "threshold": sc.threshold,
                "threshold_confidence": sc.threshold_confidence,
                "threshold_quantile": sc.threshold_quantile,
                "effective_threshold": result.effective_threshold,
                "effective_confidence_threshold": result.effective_confidence_threshold,
                "n_known_assigned": int(sum(1 for p in predictions if p in cats)),
                "n_unknown": n_unknown,
            },
        )
