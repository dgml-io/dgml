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

"""Scenario ABC + :class:`ScenarioResult` + shared pipeline helpers.

Every scenario glues the same five pieces together: text encoder, image
encoder, fusion, manifold head, and a scenario-specific predictor. The
base class encapsulates that wiring; subclasses override :meth:`fit_predict`
to implement the scenario-specific predictor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, cast

import torch

from clustering.config.schema import Config
from clustering.data.datasets import DocumentDataset
from clustering.encoders import build_encoder
from clustering.fusion import build_fusion
from clustering.manifolds import build_manifold
from clustering.utils.runid import run_id_for


@dataclass
class ScenarioResult:
    """The full output of a scenario run."""

    run_id: str
    scenario_name: str
    doc_ids: list[str]
    embeddings: torch.Tensor  # [N, D] — on-manifold
    predictions: list[str | None]  # category label per doc; None = unassigned
    confidence: list[float | None]  # in [0, 1] when meaningful, else None
    true_labels: list[str | None]  # ground truth from corpus, if present
    # Optional score matrix for top-k accuracy (classification scenarios).
    # ``scores[i, j]`` is the score of document ``i`` against class ``class_names[j]``.
    scores: torch.Tensor | None = None
    class_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Scenario(ABC):
    """Base class for all S1-S5 pipelines.

    Subclasses must:
      - declare ``name: ScenarioName`` at class scope
      - implement :meth:`fit_predict`

    Subclasses may:
      - override :meth:`refine` to support feedback loops
      - override :meth:`embed` to inject custom encoding logic
    """

    name: str

    def __init__(self, config: Config) -> None:
        from clustering.manifolds.projector import ManifoldProjector

        self.config = config
        self.run_id = run_id_for(config.model_dump(mode="json"))
        self.text_encoder = build_encoder(
            config.encoder_text, device=config.device, cache_dir=config.cache_dir
        )
        self.image_encoder = build_encoder(
            config.encoder_image, device=config.device, cache_dir=config.cache_dir
        )
        self.fusion = build_fusion(
            config.fusion,
            text_dim=config.encoder_text.embedding_dim,
            image_dim=config.encoder_image.embedding_dim,
        )
        self.manifold = build_manifold(config.manifold)
        # The projector wraps the manifold; it has trainable parameters
        # iff (config.training.trainable_projector) or the fusion output
        # dim differs from the manifold dim. The default config keeps it
        # parameter-free, so this layer is a no-op for callers on the default config.
        # ``manifold_bias`` is gated on ``training.riemannian`` — the
        # only place an on-manifold anchor is useful is when we plan
        # to update it with a Riemannian optimizer.
        self.projector = ManifoldProjector(
            self.manifold,
            input_dim=self.fusion.output_dim,
            output_dim=config.manifold.dim,
            trainable=config.training.trainable_projector,
            manifold_bias=config.training.riemannian,
            force_identity=config.training.identity_projector,
        )
        # Default to eval mode: the fusion MLP carries Dropout, and an
        # untrained projector should be deterministic. The supervised
        # trainer flips the modules it updates to train() for its loop and
        # back to eval() when done, so this only governs the inference /
        # embedding paths below. Without it, Dropout stays active during
        # embedding and injects noise into every vector.
        self.fusion.eval()
        self.projector.eval()

    # ── Shared pipeline pieces ────────────────────────────────────────────
    def fused_embeddings(
        self,
        dataset: DocumentDataset,
        *,
        batch_size: int | None = None,
    ) -> tuple[list[str], torch.Tensor, list[str | None]]:
        """Run encoder + fusion only — *no* manifold projection.

        Useful when the projector needs to be trained before final
        embedding (the trainer reuses these vectors instead of re-running
        the encoders).
        """
        batch_size = batch_size or self.config.training.batch_size
        all_fused: list[torch.Tensor] = []
        all_ids: list[str] = []
        all_labels: list[str | None] = []

        n = len(dataset)
        # Encoders + fusion are frozen here — no autograd graph needed, and
        # eval mode (set in __init__) keeps Dropout inactive so embeddings
        # are deterministic.
        with torch.no_grad():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                records = [dataset[i] for i in range(start, stop)]
                texts = [r.text or "" for r in records]
                images = [r.image for r in records]

                text_out = self.text_encoder.encode(texts)
                image_out = self.image_encoder.encode(images)
                fused = self.fusion(text_out, image_out)

                all_fused.append(fused.pooled)
                all_ids.extend(r.doc_id for r in records)
                all_labels.extend(r.label for r in records)

        if not all_fused:
            empty = torch.zeros((0, self.config.fusion.output_dim))
            return all_ids, empty, all_labels
        return all_ids, torch.cat(all_fused, dim=0), all_labels

    def modality_pooled(
        self,
        dataset: DocumentDataset,
        *,
        batch_size: int | None = None,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, list[str | None]]:
        """Per-modality pooled encoder embeddings — *before* fusion.

        Returns ``(doc_ids, text_pooled, image_pooled, labels)`` with the
        two tensors row-aligned. Used to train the fusion module
        end-to-end: the (frozen) encoders run once under ``no_grad`` here,
        and the trainer re-runs the trainable fusion on these cached
        vectors so gradients flow through fusion (and the projector).
        """
        batch_size = batch_size or self.config.training.batch_size
        all_ids: list[str] = []
        all_labels: list[str | None] = []
        text_parts: list[torch.Tensor] = []
        image_parts: list[torch.Tensor] = []

        n = len(dataset)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                records = [dataset[i] for i in range(start, stop)]
                texts = [r.text or r.doc_id for r in records]
                images = [r.image for r in records]

                text_parts.append(self.text_encoder.encode(texts).pooled)
                image_parts.append(self.image_encoder.encode(images).pooled)
                all_ids.extend(r.doc_id for r in records)
                all_labels.extend(r.label for r in records)

        if not text_parts:
            empty_t = torch.zeros((0, self.config.encoder_text.embedding_dim))
            empty_i = torch.zeros((0, self.config.encoder_image.embedding_dim))
            return all_ids, empty_t, empty_i, all_labels
        return all_ids, torch.cat(text_parts, dim=0), torch.cat(image_parts, dim=0), all_labels

    def embed(
        self,
        dataset: DocumentDataset,
        *,
        batch_size: int | None = None,
    ) -> tuple[list[str], torch.Tensor, list[str | None]]:
        """Encode + fuse + project through the manifold projector.

        Returns ``(doc_ids, on_manifold_embeddings, true_labels)``. If the
        projector has been trained beforehand (via :meth:`maybe_train_projector`),
        the projection reflects that training.
        """
        doc_ids, fused, labels = self.fused_embeddings(dataset, batch_size=batch_size)
        with torch.no_grad():
            return doc_ids, self.projector(fused), labels

    def unimodal_views(
        self,
        dataset: DocumentDataset,
        *,
        batch_size: int | None = None,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Text-only and image-only fused views of every document.

        Each view runs the *same* fusion with the other modality replaced
        by a constant placeholder (a neutral grey image / an empty string),
        so both views land in the fused space the projector consumes.
        Used by the label-free cross-modal training objective.

        Returns ``(doc_ids, text_view, image_view)`` with the two tensors
        row-aligned: row ``i`` of each is the same document.
        """
        from PIL import Image

        batch_size = batch_size or self.config.training.batch_size
        all_ids: list[str] = []
        text_parts: list[torch.Tensor] = []
        image_parts: list[torch.Tensor] = []
        placeholder_image = Image.new("RGB", (32, 32), color=(128, 128, 128))

        n = len(dataset)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                records = [dataset[i] for i in range(start, stop)]
                texts = [r.text or r.doc_id for r in records]
                images = [r.image for r in records]

                text_out = self.text_encoder.encode(texts)
                image_out = self.image_encoder.encode(images)
                placeholder_image_out = self.image_encoder.encode(
                    [placeholder_image] * len(records)
                )
                placeholder_text_out = self.text_encoder.encode([""] * len(records))

                text_parts.append(self.fusion(text_out, placeholder_image_out).pooled)
                image_parts.append(self.fusion(placeholder_text_out, image_out).pooled)
                all_ids.extend(r.doc_id for r in records)

        if not text_parts:
            empty = torch.zeros((0, self.config.fusion.output_dim))
            return all_ids, empty, empty
        return all_ids, torch.cat(text_parts, dim=0), torch.cat(image_parts, dim=0)

    def maybe_train_projector(
        self,
        fused: torch.Tensor,
        labels: list[str | None],
    ) -> list[float]:
        """If ``training.epochs > 0`` and the projector has parameters, train it.

        Returns the per-epoch loss history (empty if no training ran).
        Idempotent — subsequent calls re-train from the current state.
        """
        from clustering.manifolds.training import train_projector

        if self.config.training.epochs <= 0:
            return []
        if not any(True for _ in self.projector.parameters()):
            return []
        return train_projector(
            self.projector,
            fused,
            labels,
            cfg=self.config.training,
            seed=self.config.seed,
        )

    def maybe_train_fusion_and_projector(self, dataset: DocumentDataset) -> list[float]:
        """If ``training.trainable_fusion`` and ``epochs > 0``, train fusion + projector jointly.

        Re-encodes the dataset into per-modality pooled embeddings (frozen
        encoders, ``no_grad``) and trains the fusion module together with
        the projector under the supervised loss. Returns the per-epoch loss
        history (empty when training is disabled or there's nothing
        trainable). The caller must re-embed afterwards, since the fusion
        weights have changed.
        """
        from clustering.manifolds.training import train_fusion_projector

        if self.config.training.epochs <= 0 or not self.config.training.trainable_fusion:
            return []
        _, text_pooled, image_pooled, labels = self.modality_pooled(dataset)
        return train_fusion_projector(
            self.fusion,
            self.projector,
            text_pooled,
            image_pooled,
            labels,
            cfg=self.config.training,
            seed=self.config.seed,
        )

    def maybe_train_projector_cross_modal(self, dataset: DocumentDataset) -> list[float]:
        """Label-free projector training via cross-modal InfoNCE.

        No-op (returns ``[]``) when ``training.epochs <= 0``, when the
        projector has no parameters, or when the dataset holds fewer than
        two documents.
        """
        from clustering.manifolds.training import train_projector_cross_modal

        if self.config.training.epochs <= 0:
            return []
        if not any(True for _ in self.projector.parameters()):
            return []
        _, text_view, image_view = self.unimodal_views(dataset)
        return train_projector_cross_modal(
            self.projector,
            text_view,
            image_view,
            cfg=self.config.training,
            seed=self.config.seed,
        )

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Build category prototypes by running each text through the full pipeline.

        We pair each text with a neutral placeholder image so the prototypes
        live in the same fused space as document embeddings — otherwise
        prototype-vs-document distance would compare vectors of different
        dimensionality.  Prototypes go through the *same* projector as
        documents so they share the trained ambient → manifold map.
        """
        from PIL import Image

        with torch.no_grad():
            text_out = self.text_encoder.encode(texts)
            placeholder = Image.new("RGB", (32, 32), color=(128, 128, 128))
            image_out = self.image_encoder.encode([placeholder] * len(texts))
            fused = self.fusion(text_out, image_out)
            return cast(torch.Tensor, self.projector(fused.pooled))

    def _support_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: list[str | None],
        *,
        categories: list[str],
        n_shots: int | None = None,
    ) -> torch.Tensor:
        """Build one on-manifold prototype per category from labeled samples.

        For each category in ``categories`` we collect the ``support_embeddings``
        rows whose corresponding ``support_labels`` entry matches, optionally
        cap to the first ``n_shots`` of them (in dataset order), take the
        ambient mean, then push it onto the manifold via :meth:`expmap0`.
        Raises a clear ``ValueError`` when a category has zero samples — the
        caller (S3 / S5) typically wants to know rather than silently dropping
        the class.

        Args:
            support_embeddings: ``[M, D]`` on-manifold embeddings of the
                support documents.
            support_labels: Length-``M`` label list aligned with
                ``support_embeddings``. ``None`` entries are ignored (they
                should not occur in a properly-prepared support set).
            categories: Ordered list of category names. The returned tensor
                has one row per category in this order.
            n_shots: Maximum number of samples per category to average.
                ``None`` (default) means use every matching sample.

        Returns:
            ``[len(categories), D]`` on-manifold prototype tensor.
        """
        protos: list[torch.Tensor] = []
        for cat in categories:
            support_idx = [i for i, lbl in enumerate(support_labels) if lbl == cat]
            if n_shots is not None:
                support_idx = support_idx[:n_shots]
            if not support_idx:
                raise ValueError(
                    f"No support samples for category {cat!r}. "
                    "Provide at least one labeled example per category in the support dataset."
                )
            support_emb = support_embeddings[torch.tensor(support_idx)]
            ambient_mean = support_emb.mean(dim=0)
            protos.append(self.manifold.expmap0(ambient_mean.unsqueeze(0)).squeeze(0))
        return torch.stack(protos, dim=0)

    # ── Subclass API ──────────────────────────────────────────────────────
    @abstractmethod
    def fit_predict(
        self,
        unknown_dataset: DocumentDataset,
        support_dataset: DocumentDataset | None = None,
    ) -> ScenarioResult:
        """Run the full scenario pipeline against ``unknown_dataset``.

        ``support_dataset`` holds labeled examples per category. Only S3
        and S5 consume it; the unsupervised / zero-shot scenarios
        (S1, S2, S4) ignore it. When required and missing, scenarios raise
        ``ValueError`` rather than silently degrading.
        """

    def refine(
        self,
        result: ScenarioResult,
        user_feedback: dict[str, str],
        unknown_dataset: DocumentDataset,
    ) -> ScenarioResult:
        """Update predictions given ``{doc_id: corrected_label}`` feedback.

        Default impl applies the corrections in-place; scenarios that can do
        more (e.g. recompute prototypes from new labels) override.
        """
        new_preds = list(result.predictions)
        for i, doc_id in enumerate(result.doc_ids):
            if doc_id in user_feedback:
                new_preds[i] = user_feedback[doc_id]
        return ScenarioResult(
            run_id=result.run_id,
            scenario_name=result.scenario_name,
            doc_ids=result.doc_ids,
            embeddings=result.embeddings,
            predictions=new_preds,
            confidence=list(result.confidence),
            true_labels=list(result.true_labels),
            scores=result.scores,
            class_names=list(result.class_names) if result.class_names else None,
            metadata={**result.metadata, "refined": True},
        )
