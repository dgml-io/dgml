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

"""S4 — all categories known, zero-shot (names only).

Prototypes come from the text encoder applied to the bare category names
(plus an optional prompt template). Each document is assigned to its
nearest prototype under the manifold distance.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from clustering.data.datasets import DocumentDataset
from clustering.scenarios.base import Scenario, ScenarioResult
from clustering.scenarios.clustering import assign_to_prototypes


class S4ZeroShot(Scenario):
    name = "s4"

    # Prompt-engineering hook — overridable on a per-corpus basis.
    PROMPT_TEMPLATE: ClassVar[str] = "a scanned document of category: {category}"

    def _build_prototypes(self) -> tuple[list[str], torch.Tensor]:
        cats = self.config.scenario.known_categories
        if not cats:
            raise ValueError("S4 requires scenario.known_categories to be non-empty.")
        prompts = [self.PROMPT_TEMPLATE.format(category=c) for c in cats]
        prototypes = self.encode_texts(prompts)
        return cats, prototypes

    def fit_predict(
        self,
        unknown_dataset: DocumentDataset,
        support_dataset: DocumentDataset | None = None,
    ) -> ScenarioResult:
        # S4 is zero-shot: prototypes come from category names, not samples.
        del support_dataset
        cats, prototypes = self._build_prototypes()
        doc_ids, embeddings, true_labels = self.embed(unknown_dataset)

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
            metadata={"prototype_source": "name", "categories": list(cats)},
        )
