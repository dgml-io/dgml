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

"""End-to-end coverage for S5 closed-set prototype assignment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from clustering.config.schema import Config
from clustering.data.datasets import DocumentDataset, DocumentRecord
from clustering.encoders.base import Encoder, EncoderOutput
from clustering.scenarios.s5_full_supervised import S5FullSupervised
from PIL import Image


@dataclass(frozen=True)
class _Record:
    doc_id: str
    text: str
    label: str | None = None


class _InMemoryDataset(DocumentDataset):
    def __init__(self, records: list[_Record]) -> None:
        self._records = records
        self._image = Image.new("RGB", (8, 8))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> DocumentRecord:
        record = self._records[index]
        return DocumentRecord(
            doc_id=record.doc_id,
            label=record.label,
            image=self._image,
            text=record.text,
            thumbnail_path=None,
        )


class _LookupEncoder(Encoder[Any]):
    """Return explicit 2-D vectors so prototype geometry is transparent."""

    embedding_dim = 2
    multi_vector = False

    def __init__(self, vectors: dict[str, tuple[float, float]]) -> None:
        self._vectors = vectors

    def encode(self, batch: Sequence[Any]) -> EncoderOutput:
        rows = [self._vectors[item] for item in batch]
        return EncoderOutput(pooled=torch.tensor(rows, dtype=torch.float32))


def _config(*, categories: list[str] | None = None, n_shots: int | None = 2) -> Config:
    scenario: dict[str, Any] = {
        "name": "s5",
        "known_categories": categories if categories is not None else ["Invoice", "Contract"],
    }
    if n_shots is not None:
        scenario["n_shots"] = n_shots
    return Config.model_validate(
        {
            "scenario": scenario,
            "encoder_text": {"name": "dummy", "embedding_dim": 2},
            "encoder_image": {"name": "dummy", "embedding_dim": 2},
            "fusion": {"name": "none", "prefer_modality": "text", "output_dim": 2},
            "manifold": {"name": "euclidean", "dim": 2, "curvature": 0.0},
            "training": {"epochs": 0, "identity_projector": True, "batch_size": 8},
            "logger": {"name": "none"},
            "corpus": {"root": "."},
            "device": "cpu",
            "seed": 0,
        }
    )


def _scenario(
    vectors: dict[str, tuple[float, float]],
    *,
    categories: list[str] | None = None,
    n_shots: int | None = 2,
) -> S5FullSupervised:
    scenario = S5FullSupervised(_config(categories=categories, n_shots=n_shots))
    scenario.text_encoder = _LookupEncoder(vectors)
    return scenario


def test_s5_assigns_to_support_prototypes_with_scores_and_confidence() -> None:
    vectors = {
        "invoice_support_1": (0.0, 0.0),
        "invoice_support_2": (0.0, 2.0),
        "contract_support_1": (10.0, 0.0),
        "contract_support_2": (10.0, 2.0),
        "invoice_query": (0.1, 1.0),
        "contract_query": (9.9, 1.0),
    }
    support = _InMemoryDataset(
        [
            _Record("si1", "invoice_support_1", "Invoice"),
            _Record("si2", "invoice_support_2", "Invoice"),
            _Record("sc1", "contract_support_1", "Contract"),
            _Record("sc2", "contract_support_2", "Contract"),
        ]
    )
    unknown = _InMemoryDataset(
        [
            _Record("qi", "invoice_query", "Invoice"),
            _Record("qc", "contract_query", "Contract"),
        ]
    )

    result = _scenario(vectors).fit_predict(unknown, support)

    assert result.scenario_name == "s5"
    assert result.doc_ids == ["qi", "qc"]
    assert result.predictions == ["Invoice", "Contract"]
    assert all(pred is not None and not pred.startswith("unknown_") for pred in result.predictions)
    assert result.true_labels == ["Invoice", "Contract"]
    assert result.class_names == ["Invoice", "Contract"]
    assert result.embeddings.shape == (2, 2)
    assert result.scores is not None
    assert result.scores.shape == (2, 2)
    assert torch.allclose(result.scores.sum(dim=1), torch.ones(2))
    assert all(conf is not None and 0.99 < conf <= 1.0 for conf in result.confidence)
    assert result.metadata == {
        "prototype_source": "support_mean",
        "n_shots": 2,
        "n_support": 4,
        "categories": ["Invoice", "Contract"],
        "projector_trained": False,
        "projector_loss_history": [],
    }


def test_s5_n_shots_uses_first_samples_per_category() -> None:
    vectors = {
        "invoice_first": (0.0, 0.0),
        "invoice_outlier": (100.0, 0.0),
        "contract_first": (10.0, 0.0),
        "query": (1.0, 0.0),
    }
    support = _InMemoryDataset(
        [
            _Record("si1", "invoice_first", "Invoice"),
            _Record("si2", "invoice_outlier", "Invoice"),
            _Record("sc1", "contract_first", "Contract"),
        ]
    )
    unknown = _InMemoryDataset([_Record("q", "query")])

    result = _scenario(vectors, n_shots=1).fit_predict(unknown, support)

    assert result.predictions == ["Invoice"]
    assert result.metadata["n_shots"] == 1
    assert result.metadata["n_support"] == 3


def test_s5_requires_non_empty_known_categories() -> None:
    scenario = _scenario({}, categories=[])

    with pytest.raises(ValueError, match="known_categories"):
        scenario.fit_predict(_InMemoryDataset([]), _InMemoryDataset([]))


def test_s5_requires_non_empty_support_dataset() -> None:
    scenario = _scenario({})

    with pytest.raises(ValueError, match="non-empty support_dataset"):
        scenario.fit_predict(_InMemoryDataset([]), _InMemoryDataset([]))


def test_s5_requires_support_for_every_category() -> None:
    vectors = {"invoice_support": (0.0, 0.0), "query": (0.1, 0.0)}
    support = _InMemoryDataset([_Record("si", "invoice_support", "Invoice")])
    unknown = _InMemoryDataset([_Record("q", "query")])

    with pytest.raises(ValueError, match="No support samples for category 'Contract'"):
        _scenario(vectors).fit_predict(unknown, support)


@pytest.mark.parametrize("bad_label", [None, "Unexpected"])
def test_s5_rejects_unlabeled_or_unknown_support_rows(bad_label: str | None) -> None:
    vectors = {
        "invoice_support": (0.0, 0.0),
        "contract_support": (10.0, 0.0),
        "bad_support": (5.0, 0.0),
        "query": (0.1, 0.0),
    }
    support = _InMemoryDataset(
        [
            _Record("si", "invoice_support", "Invoice"),
            _Record("sc", "contract_support", "Contract"),
            _Record("sx", "bad_support", bad_label),
        ]
    )
    unknown = _InMemoryDataset([_Record("q", "query")])

    with pytest.raises(ValueError, match="support_dataset labels"):
        _scenario(vectors).fit_predict(unknown, support)
