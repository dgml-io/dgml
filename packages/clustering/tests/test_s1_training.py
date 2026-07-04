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

"""S1 semi-supervised projector training (ported from doc-categorization).

S1's cluster assignment is label-free, but when the dataset's records
carry ground-truth labels and ``training.epochs > 0``, the manifold
projector is trained on those labels *before* clustering. With no labels
(or ``epochs == 0``) S1 degrades to the pure unsupervised baseline.
"""

from __future__ import annotations

from typing import Any

import torch
from clustering.config.schema import Config
from clustering.data.datasets import DocumentDataset, DocumentRecord
from clustering.scenarios import build_scenario
from PIL import Image

_DIM = 16


class _InMemoryDataset(DocumentDataset):
    """Tiny labeled dataset — two well-separated text "classes"."""

    def __init__(self, labels: list[str | None]) -> None:
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> DocumentRecord:
        label = self._labels[index]
        text = f"{label or 'unlabeled'} document number {index}"
        return DocumentRecord(
            doc_id=f"doc_{index}",
            label=label,
            image=Image.new("RGB", (8, 8), color=(index * 9 % 255, 0, 0)),
            text=text,
            thumbnail_path=None,
        )


def _config(*, epochs: int, supervision: str = "labels", pseudo_rounds: int = 1) -> Config:
    raw: dict[str, Any] = {
        "scenario": {"name": "s1", "k_clusters": 2, "cluster_algorithm": "kmeans"},
        "encoder_text": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "encoder_image": {"name": "dummy", "model_id": "dummy", "embedding_dim": _DIM},
        "fusion": {"name": "late_concat", "output_dim": 2 * _DIM},
        "manifold": {"name": "euclidean", "dim": 2 * _DIM},
        "training": {
            "epochs": epochs,
            "loss": "prototypical",
            "trainable_projector": True,
            "supervision": supervision,
            "pseudo_rounds": pseudo_rounds,
        },
        "logger": {"name": "none"},
        "corpus": {"root": "."},
        "device": "cpu",
        "seed": 0,
    }
    return Config.model_validate(raw)


def test_s1_trains_projector_on_record_labels() -> None:
    labels: list[str | None] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    dataset = _InMemoryDataset(labels)
    scenario = build_scenario(_config(epochs=3))
    result = scenario.fit_predict(dataset)
    assert result.metadata["projector_trained"] is True
    history = result.metadata["projector_loss_history"]
    assert len(history) == 3
    # Every doc still gets a label-free cluster assignment.
    assert all(p is not None and p.startswith("cluster_") for p in result.predictions)


def test_s1_training_changes_embeddings() -> None:
    labels: list[str | None] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    untrained = build_scenario(_config(epochs=0)).fit_predict(_InMemoryDataset(labels))
    trained = build_scenario(_config(epochs=5)).fit_predict(_InMemoryDataset(labels))
    assert not torch.allclose(untrained.embeddings, trained.embeddings, atol=1e-6), (
        "training had no effect on the projected embeddings"
    )


def test_s1_without_labels_degrades_to_unsupervised() -> None:
    dataset = _InMemoryDataset([None] * 8)
    scenario = build_scenario(_config(epochs=3))
    result = scenario.fit_predict(dataset)
    assert result.metadata["projector_trained"] is False
    assert result.metadata["projector_loss_history"] == []
    assert all(p is not None for p in result.predictions)


def test_s1_epochs_zero_reports_untrained() -> None:
    labels: list[str | None] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    dataset = _InMemoryDataset(labels)
    scenario = build_scenario(_config(epochs=0))
    result = scenario.fit_predict(dataset)
    assert result.metadata["projector_trained"] is False


# ── Label-free supervision: pseudo-labels ────────────────────────────────
def test_s1_pseudo_labels_trains_without_any_labels() -> None:
    dataset = _InMemoryDataset([None] * 8)
    scenario = build_scenario(_config(epochs=3, supervision="pseudo_labels", pseudo_rounds=2))
    result = scenario.fit_predict(dataset)
    assert result.metadata["supervision"] == "pseudo_labels"
    assert result.metadata["projector_trained"] is True
    # 2 rounds x 3 epochs.
    assert len(result.metadata["projector_loss_history"]) == 6
    assert all(p is not None for p in result.predictions)


def test_s1_pseudo_labels_changes_embeddings() -> None:
    labels: list[str | None] = [None] * 8
    untrained = build_scenario(_config(epochs=0)).fit_predict(_InMemoryDataset(labels))
    trained = build_scenario(_config(epochs=5, supervision="pseudo_labels")).fit_predict(
        _InMemoryDataset(labels)
    )
    assert not torch.allclose(untrained.embeddings, trained.embeddings, atol=1e-6)


# ── Label-free supervision: cross-modal InfoNCE ──────────────────────────
def test_s1_cross_modal_trains_without_any_labels() -> None:
    dataset = _InMemoryDataset([None] * 8)
    scenario = build_scenario(_config(epochs=3, supervision="cross_modal"))
    result = scenario.fit_predict(dataset)
    assert result.metadata["supervision"] == "cross_modal"
    assert result.metadata["projector_trained"] is True
    assert len(result.metadata["projector_loss_history"]) == 3
    assert all(p is not None for p in result.predictions)


def test_s1_cross_modal_changes_embeddings() -> None:
    labels: list[str | None] = [None] * 8
    untrained = build_scenario(_config(epochs=0)).fit_predict(_InMemoryDataset(labels))
    trained = build_scenario(_config(epochs=5, supervision="cross_modal")).fit_predict(
        _InMemoryDataset(labels)
    )
    assert not torch.allclose(untrained.embeddings, trained.embeddings, atol=1e-6)


def test_s1_label_supervision_ignores_pseudo_settings() -> None:
    """Default supervision still trains on record labels (regression guard)."""
    labels: list[str | None] = ["A", "A", "A", "A", "B", "B", "B", "B"]
    scenario = build_scenario(_config(epochs=2, supervision="labels", pseudo_rounds=3))
    result = scenario.fit_predict(_InMemoryDataset(labels))
    assert result.metadata["supervision"] == "labels"
    # Plain label training: epochs only, no rounds multiplier.
    assert len(result.metadata["projector_loss_history"]) == 2
