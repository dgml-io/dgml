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

"""S3 — partial labels + few-shot per known category.

The known-side path is the same as S5 (prototypes are manifold-means of
the labeled support samples). The unknown-side path is the same as S2
(cluster the unassigned bucket into ``unknown_<i>``). The configurable
threshold controls when a document gets routed to the unknown bucket.

This is the canonical "warm-start an emerging taxonomy" scenario and the
typical landing pad of the cascade after S1 → human-label step.
"""

from __future__ import annotations

import torch

from clustering.data.datasets import DocumentDataset
from clustering.scenarios.base import Scenario, ScenarioResult
from clustering.scenarios.clustering import assign_to_prototypes, manifold_kmeans


class S3PartialFewShot(Scenario):
    name = "s3"
    # If ``config.scenario.n_shots`` is unset, take at most this many
    # support samples per category when building prototypes.
    DEFAULT_N_SHOTS = 4

    def fit_predict(
        self,
        unknown_dataset: DocumentDataset,
        support_dataset: DocumentDataset | None = None,
    ) -> ScenarioResult:
        cats = self.config.scenario.known_categories
        if not cats:
            raise ValueError("S3 requires scenario.known_categories to be non-empty.")
        if support_dataset is None or len(support_dataset) == 0:
            raise ValueError(
                "S3 requires a non-empty support_dataset of labeled examples. "
                "Use S2 if you only have category names (no samples)."
            )
        n_shots = self.config.scenario.n_shots or self.DEFAULT_N_SHOTS

        # ── Embed support set (drives projector training + prototypes) ──
        _, support_fused, support_labels = self.fused_embeddings(support_dataset)
        train_history = self.maybe_train_projector(support_fused, support_labels)
        support_embeddings = self.projector(support_fused)
        known_protos = self._support_prototypes(
            support_embeddings,
            support_labels,
            categories=cats,
            n_shots=n_shots,
        )

        # ── Embed unknown set + initial assignment with composable gates ─
        doc_ids, fused, true_labels = self.fused_embeddings(unknown_dataset)
        embeddings = self.projector(fused)

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

        # Same unknown-bucket clustering as S2.
        predictions: list[str | None] = [None] * len(doc_ids)
        confidence: list[float | None] = [None] * len(doc_ids)
        unknown_idx = [i for i, li in enumerate(labels_arr.tolist()) if int(li) == -1]
        n_unknown = len(unknown_idx)

        if n_unknown >= 2:
            unknown_emb = embeddings[torch.tensor(unknown_idx)]
            k_unknown = max(2, min(8, n_unknown))
            ulabels_t, _ = manifold_kmeans(
                unknown_emb, k=k_unknown, manifold=self.manifold, seed=self.config.seed
            )
            ulabels_arr = ulabels_t.detach().numpy() if hasattr(ulabels_t, "numpy") else ulabels_t
            for src, dst in zip(ulabels_arr.tolist(), unknown_idx, strict=True):
                predictions[dst] = f"unknown_{int(src)}"
        elif n_unknown == 1:
            predictions[unknown_idx[0]] = "unknown_0"

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
                "n_shots": n_shots,
                "n_support": len(support_dataset),
                # Echo user config + the effective post-calibration values
                # so `compare_runs` and the UI can recover the operating point.
                "threshold": sc.threshold,
                "threshold_confidence": sc.threshold_confidence,
                "threshold_quantile": sc.threshold_quantile,
                "effective_threshold": result.effective_threshold,
                "effective_confidence_threshold": result.effective_confidence_threshold,
                "n_unknown": n_unknown,
                "projector_trained": bool(train_history),
                "projector_loss_history": train_history,
            },
        )
