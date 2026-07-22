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

"""End-to-end coverage for the assignment scenarios S2 / S3 / S4.

Complements ``test_s5_full_supervised.py``. Uses deterministic 2-D lookup
embeddings (no model download, no mocked ``fit_predict``) so prototype
geometry and the unknown-bucket novelty gate are transparent:

- **S4** (zero-shot): prototypes from category-name prompts; every document is
  forced into a known category — no ``unknown_*`` bucket.
- **S2** (partial names): same name prototypes plus a novelty gate; an
  out-of-scope document (roughly equidistant from every prototype, so low
  confidence) is rejected into an emergent ``unknown_*`` cluster.
- **S3** (few-shot): prototypes from the *support* set (not names) plus the
  same novelty gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from clustering.config.schema import Config
from clustering.data.datasets import DocumentDataset, DocumentRecord
from clustering.encoders.base import Encoder, EncoderOutput
from clustering.scenarios.base import Scenario
from clustering.scenarios.s2_partial_labels import S2PartialLabels
from clustering.scenarios.s3_partial_few_shot import S3PartialFewShot
from clustering.scenarios.s4_zero_shot import S4ZeroShot
from PIL import Image

# S2 and S4 build name prototypes from this template (see the scenarios).
PROMPT = "a scanned document of category: {}"


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
    """Return explicit 2-D vectors (keyed by text) so geometry is transparent."""

    embedding_dim = 2
    multi_vector = False

    def __init__(self, vectors: dict[str, tuple[float, float]]) -> None:
        self._vectors = vectors

    def encode(self, batch: Sequence[Any]) -> EncoderOutput:
        rows = [self._vectors[item] for item in batch]
        return EncoderOutput(pooled=torch.tensor(rows, dtype=torch.float32))


def _config(
    name: str,
    categories: list[str],
    *,
    n_shots: int | None = None,
    threshold_confidence: float | None = None,
) -> Config:
    scenario: dict[str, Any] = {"name": name, "known_categories": categories}
    if n_shots is not None:
        scenario["n_shots"] = n_shots
    if threshold_confidence is not None:
        scenario["threshold_confidence"] = threshold_confidence
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


def _with_encoder(scenario: Scenario, vectors: dict[str, tuple[float, float]]) -> Scenario:
    scenario.text_encoder = _LookupEncoder(vectors)
    return scenario


# ── S4: zero-shot, closed-set (name prototypes, no unknown bucket) ───────────
def test_s4_assigns_every_doc_to_a_known_category() -> None:
    vectors = {
        PROMPT.format("Invoice"): (0.0, 0.0),
        PROMPT.format("Contract"): (10.0, 0.0),
        "invoice_query": (0.5, 0.0),
        "contract_query": (9.5, 0.0),
    }
    unknown = _InMemoryDataset(
        [_Record("qi", "invoice_query", "Invoice"), _Record("qc", "contract_query", "Contract")]
    )
    scenario = _with_encoder(S4ZeroShot(_config("s4", ["Invoice", "Contract"])), vectors)

    result = scenario.fit_predict(unknown)

    assert result.scenario_name == "s4"
    assert result.predictions == ["Invoice", "Contract"]
    # Closed set: S4 never emits an emergent unknown bucket.
    assert all(p in {"Invoice", "Contract"} for p in result.predictions)
    assert result.scores is not None and result.scores.shape == (2, 2)
    assert torch.allclose(result.scores.sum(dim=1), torch.ones(2))
    assert all(c is not None and 0.0 < c <= 1.0 for c in result.confidence)


def test_s4_requires_known_categories() -> None:
    scenario = _with_encoder(S4ZeroShot(_config("s4", [])), {})
    with pytest.raises(ValueError, match="known_categories"):
        scenario.fit_predict(_InMemoryDataset([]))


# ── S2: partial names + novelty gate (out-of-scope → unknown_*) ──────────────
def test_s2_assigns_known_and_rejects_out_of_scope() -> None:
    vectors = {
        PROMPT.format("Invoice"): (0.0, 0.0),
        PROMPT.format("Contract"): (10.0, 0.0),
        "invoice_query": (0.2, 0.0),
        "contract_query": (9.8, 0.0),
        "out_of_scope": (5.0, 40.0),  # ~equidistant from both prototypes → low confidence
    }
    unknown = _InMemoryDataset(
        [
            _Record("qi", "invoice_query", "Invoice"),
            _Record("qc", "contract_query", "Contract"),
            _Record("qx", "out_of_scope", None),
        ]
    )
    scenario = _with_encoder(
        S2PartialLabels(_config("s2", ["Invoice", "Contract"], threshold_confidence=0.9)),
        vectors,
    )

    result = scenario.fit_predict(unknown)

    assert result.scenario_name == "s2"
    assert result.predictions[0] == "Invoice"
    assert result.predictions[1] == "Contract"
    # The out-of-scope doc is rejected by the confidence gate into an emergent bucket.
    assert result.predictions[2] is not None and result.predictions[2].startswith("unknown_")
    assert result.metadata["n_unknown"] == 1
    # Known assignments carry a confidence; the rejected one does not.
    assert result.confidence[0] is not None and result.confidence[1] is not None
    assert result.confidence[2] is None


def test_s2_requires_known_categories() -> None:
    scenario = _with_encoder(S2PartialLabels(_config("s2", [])), {})
    with pytest.raises(ValueError, match="known_categories"):
        scenario.fit_predict(_InMemoryDataset([]))


# ── S3: few-shot support prototypes + novelty gate ──────────────────────────
def test_s3_assigns_from_support_and_rejects_out_of_scope() -> None:
    vectors = {
        "invoice_support_1": (0.0, 0.0),
        "invoice_support_2": (0.0, 2.0),
        "contract_support_1": (10.0, 0.0),
        "contract_support_2": (10.0, 2.0),
        "invoice_query": (0.1, 1.0),
        "contract_query": (9.9, 1.0),
        "out_of_scope": (5.0, 40.0),
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
            _Record("qx", "out_of_scope", None),
        ]
    )
    cfg = _config("s3", ["Invoice", "Contract"], n_shots=2, threshold_confidence=0.9)
    scenario = _with_encoder(S3PartialFewShot(cfg), vectors)

    result = scenario.fit_predict(unknown, support)

    assert result.scenario_name == "s3"
    assert result.predictions[0] == "Invoice"
    assert result.predictions[1] == "Contract"
    assert result.predictions[2] is not None and result.predictions[2].startswith("unknown_")


def test_s3_requires_support_dataset() -> None:
    scenario = _with_encoder(S3PartialFewShot(_config("s3", ["Invoice", "Contract"])), {})
    with pytest.raises(ValueError, match="non-empty support_dataset"):
        scenario.fit_predict(_InMemoryDataset([]), _InMemoryDataset([]))
