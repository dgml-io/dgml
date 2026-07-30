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
from clustering.scenarios.base import UNKNOWN_NOISE_LABEL, Scenario, ScenarioResult
from clustering.scenarios.clustering import assign_to_prototypes, cluster_emergent_bucket


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
        # No calibrator here: S2's prototypes come from category *names*, so
        # there is no labeled support set to fit temperature/Platt/conformal
        # against. The confidence stays ordinal and the review decision is a
        # plain floor on it — which is honest about what the number is.
        result = assign_to_prototypes(
            embeddings,
            known_protos,
            self.manifold,
            threshold=sc.threshold,
            threshold_confidence=sc.threshold_confidence,
            threshold_quantile=sc.threshold_quantile,
            abstain_threshold=sc.calibration.abstain_threshold,
        )
        labels_t, conf_t = result.labels, result.confidence

        labels_arr = labels_t.detach().numpy() if hasattr(labels_t, "numpy") else labels_t
        conf_arr = conf_t.detach().numpy() if hasattr(conf_t, "numpy") else conf_t
        abstain_list = (
            [bool(x) for x in result.abstain.tolist()] if result.abstain is not None else None
        )

        # ── Cluster the unassigned bucket into emergent categories ───────
        predictions: list[str | None] = [None] * len(doc_ids)
        confidence: list[float | None] = [None] * len(doc_ids)
        # Only known-category assignments can abstain: a document routed to the
        # unknown bucket has no assignment to review yet — it is waiting on a
        # new category, which is a different decision.
        review: list[bool] = [False] * len(doc_ids)
        unknown_idx = [i for i, li in enumerate(labels_arr.tolist()) if int(li) == -1]
        n_unknown = len(unknown_idx)

        if n_unknown >= 2:
            unknown_emb = embeddings[torch.tensor(unknown_idx)]
            ulabels_t, _ = cluster_emergent_bucket(
                unknown_emb,
                scenario=sc,
                manifold=self.manifold,
                seed=self.config.seed,
            )
            ulabels_arr = ulabels_t.detach().numpy() if hasattr(ulabels_t, "numpy") else ulabels_t
            for src, dst in zip(ulabels_arr.tolist(), unknown_idx, strict=True):
                src_i = int(src)
                predictions[dst] = UNKNOWN_NOISE_LABEL if src_i == -1 else f"unknown_{src_i}"
                confidence[dst] = None
        elif n_unknown == 1:
            predictions[unknown_idx[0]] = "unknown_0"

        # Fill in the known-assignment predictions.
        for i, li in enumerate(labels_arr.tolist()):
            if int(li) != -1:
                predictions[i] = cats[int(li)]
                confidence[i] = float(conf_arr[i])
                if abstain_list is not None:
                    review[i] = abstain_list[i]

        return ScenarioResult(
            run_id=self.run_id,
            scenario_name=self.name,
            doc_ids=doc_ids,
            embeddings=embeddings,
            predictions=predictions,
            confidence=confidence,
            true_labels=true_labels,
            review=review,
            metadata={
                "categories": list(cats),
                "n_review": int(sum(review)),
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
