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

import torch

from clustering.data.datasets import DocumentDataset
from clustering.scenarios.base import Scenario, ScenarioResult
from clustering.scenarios.clustering import assign_to_prototypes


class S5FullSupervised(Scenario):
    name = "s5"
    # If ``config.scenario.n_shots`` is unset, take at most this many
    # support samples per category when building prototypes.
    DEFAULT_N_SHOTS = 8

    @staticmethod
    def _blend_prototypes(
        support_protos: torch.Tensor, name_protos: torch.Tensor, alpha: float
    ) -> torch.Tensor:
        """Convex-combine name and support prototypes in unit-direction space.

        Both inputs are L2-normalized, mixed ``alpha*name + (1-alpha)*support``,
        then renormalized -- a direction-space blend so a name-only prototype and
        a support mean, which have very different norms, contribute by direction
        rather than magnitude. The result is a unit vector; the caller restricts
        this to a euclidean manifold (``fit_predict``). ``alpha`` is assumed
        validated to ``[0, 1]`` by :class:`ScenarioConfig`.
        """

        def _unit(x: torch.Tensor) -> torch.Tensor:
            unit: torch.Tensor = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            return unit

        return _unit(alpha * _unit(name_protos) + (1.0 - alpha) * _unit(support_protos))

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

        # ── Optionally blend in a name prototype (few-shot prior) ────────
        blend = self.config.scenario.name_prototype_blend
        prototype_source = "support_mean"
        if blend:
            # The blend mixes a name prototype with the support mean in the
            # unit-direction space. That is only geometrically sound on a flat
            # (euclidean) manifold, where the projector output equals the
            # on-manifold representation; on spherical/hyperbolic heads the two
            # operands live in different spaces (one on-manifold, one ambient)
            # and averaging them is meaningless. It is also the only manifold the
            # gain was measured on — so require it rather than misbehave silently.
            if self.config.manifold.name != "euclidean":
                raise ValueError(
                    "scenario.name_prototype_blend is only supported with a euclidean "
                    f"manifold; got manifold.name={self.config.manifold.name!r}."
                )
            name_protos = self.encode_texts(list(cats))
            prototypes = self._blend_prototypes(prototypes, name_protos, blend)
            prototype_source = f"name_support_blend(alpha={blend:g})"

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
                "prototype_source": prototype_source,
                "n_shots": n_shots,
                "n_support": len(support_dataset),
                "categories": list(cats),
                "projector_trained": bool(train_history),
                "projector_loss_history": train_history,
            },
        )
