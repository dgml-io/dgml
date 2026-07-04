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

"""S5 — all categories known + labeled samples per category.

Prototypes are the manifold-mean of the labeled samples per category,
read from ``support_dataset``. Each document in ``unknown_dataset`` is
then assigned to its nearest prototype.
"""

from __future__ import annotations

from clustering.data.datasets import DocumentDataset
from clustering.scenarios.base import Scenario, ScenarioResult
from clustering.scenarios.clustering import assign_to_prototypes


class S5FullSupervised(Scenario):
    name = "s5"
    # If ``config.scenario.n_shots`` is unset, take at most this many
    # support samples per category when building prototypes.
    DEFAULT_N_SHOTS = 8

    def fit_predict(
        self,
        unknown_dataset: DocumentDataset,
        support_dataset: DocumentDataset | None = None,
    ) -> ScenarioResult:
        cats = self.config.scenario.known_categories
        if not cats:
            raise ValueError("S5 requires scenario.known_categories to be non-empty.")
        if support_dataset is None or len(support_dataset) == 0:
            raise ValueError(
                "S5 requires a non-empty support_dataset of labeled examples. "
                "Use S4 if you only have category names (no samples)."
            )
        n_shots = self.config.scenario.n_shots or self.DEFAULT_N_SHOTS

        # ── Embed support set (drives projector training + prototypes) ──
        _, support_fused, support_labels = self.fused_embeddings(support_dataset)
        train_history = self.maybe_train_projector(support_fused, support_labels)
        support_embeddings = self.projector(support_fused)
        prototypes = self._support_prototypes(
            support_embeddings,
            support_labels,
            categories=cats,
            n_shots=n_shots,
        )

        # ── Embed unknown set + assign to prototypes ─────────────────────
        doc_ids, fused, true_labels = self.fused_embeddings(unknown_dataset)
        embeddings = self.projector(fused)

        result = assign_to_prototypes(embeddings, prototypes, self.manifold)
        labels_t, conf_t, probs_t = result.labels, result.confidence, result.probs
        labels_arr = labels_t.detach().numpy() if hasattr(labels_t, "numpy") else labels_t
        conf_arr = conf_t.detach().numpy() if hasattr(conf_t, "numpy") else conf_t
        predictions: list[str | None] = [cats[int(li)] for li in labels_arr.tolist()]
        confidence: list[float | None] = [float(c) for c in conf_arr.tolist()]

        return ScenarioResult(
            run_id=self.run_id,
            scenario_name=self.name,
            doc_ids=doc_ids,
            embeddings=embeddings,
            predictions=predictions,
            confidence=confidence,
            true_labels=true_labels,
            scores=probs_t,
            class_names=list(cats),
            metadata={
                "prototype_source": "support_mean",
                "n_shots": n_shots,
                "n_support": len(support_dataset),
                "categories": list(cats),
                "projector_trained": bool(train_history),
                "projector_loss_history": train_history,
            },
        )
