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
from clustering.scenarios.base import UNKNOWN_NOISE_LABEL, Scenario
from clustering.scenarios.clustering import LeidenGraphMethod, emergent_bucket_k_neighbors
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
    **scenario_overrides: Any,
) -> Config:
    scenario: dict[str, Any] = {"name": name, "known_categories": categories}
    if n_shots is not None:
        scenario["n_shots"] = n_shots
    if threshold_confidence is not None:
        scenario["threshold_confidence"] = threshold_confidence
    # Anything else (``cluster_algorithm``, the ``leiden_*`` knobs, …) goes
    # straight through to ``ScenarioConfig``, which validates it strictly.
    scenario.update(scenario_overrides)
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


def test_s3_renders_noise_in_the_unknown_bucket_like_s2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S3 now clusters its unknown bucket through the same dispatcher as S2, so
    ``-1`` is genuinely reachable — a density algorithm such as ``leiden`` or
    ``hdbscan`` emits it for points it declines to place. Pin the rendering: an
    unguarded f-string would produce ``"unknown_-1"``, a string that reads as an
    ordinary cluster to every consumer downstream.

    (This guard predates the switch, where it was defensive. The switch is what
    made it load-bearing.)
    """
    vectors = {
        "invoice_support_1": (0.0, 0.0),
        "invoice_support_2": (0.0, 2.0),
        "contract_support_1": (10.0, 0.0),
        "contract_support_2": (10.0, 2.0),
        "far_1": (5.0, 40.0),
        "far_2": (5.0, 41.0),
    }
    support = _InMemoryDataset(
        [
            _Record("si1", "invoice_support_1", "Invoice"),
            _Record("si2", "invoice_support_2", "Invoice"),
            _Record("sc1", "contract_support_1", "Contract"),
            _Record("sc2", "contract_support_2", "Contract"),
        ]
    )
    unknown = _InMemoryDataset([_Record("qa", "far_1"), _Record("qb", "far_2")])
    cfg = _config("s3", ["Invoice", "Contract"], n_shots=2, threshold_confidence=0.9)
    scenario = _with_encoder(S3PartialFewShot(cfg), vectors)

    def _all_noise(emb: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full((emb.shape[0],), -1, dtype=torch.long), torch.empty(0, emb.shape[1])

    monkeypatch.setattr(
        "clustering.scenarios.s3_partial_few_shot.cluster_emergent_bucket", _all_noise
    )
    result = scenario.fit_predict(unknown, support)

    assert result.predictions == [UNKNOWN_NOISE_LABEL, UNKNOWN_NOISE_LABEL]


# ── The emergent (unknown) bucket: degree scaling shared by S2 and S3 ────────
@pytest.mark.parametrize(
    ("n", "configured", "expected"),
    [
        # Never raises a configured value that is already small.
        (100, 2, 2),
        (100, 1, 1),
        # The sqrt term is floored at 2 — a graph of degree < 2 cannot express
        # a community — but the floor sits *under* the ceiling, so the two
        # rows above still pass a smaller configured value straight through.
        (2, 25, 2),
        (3, 25, 2),
        (4, 25, 2),
        # Sub-linear growth over the sizes an unknown bucket actually takes.
        (9, 25, 3),
        (16, 25, 4),
        (25, 25, 5),
        # Degenerate inputs must not raise or go negative.
        (0, 25, 2),
        (1, 25, 2),
    ],
)
def test_emergent_bucket_k_neighbors_is_a_ceiling_not_a_setting(
    n: int, configured: int, expected: int
) -> None:
    assert emergent_bucket_k_neighbors(n, configured) == expected


def test_emergent_bucket_degree_never_exceeds_the_configured_value() -> None:
    """The rule may only lower the degree. If it could raise it, a user who
    deliberately configured a sparse graph would silently get a denser one.
    """
    for n in range(0, 200):
        for configured in (1, 2, 4, 15, 25, 60):
            assert emergent_bucket_k_neighbors(n, configured) <= configured


@pytest.mark.parametrize("graph_method", ["mutual_knn", "radius"])
def test_emergent_bucket_degree_is_untouched_off_the_knn_graph(
    graph_method: LeidenGraphMethod,
) -> None:
    """Only the plain k-NN graph is scaled.

    ``mutual_knn`` needs *reciprocal* membership, so its edge count falls away
    much faster than its degree does, and ``radius`` feeds ``k_neighbors`` into
    the auto-radius knee heuristic rather than into a k-NN graph. Neither is
    covered by the measurement that motivated the scaling and both have a
    mechanism for harm, so both must pass through untouched.
    """
    for n in (0, 2, 6, 9, 30, 200):
        assert emergent_bucket_k_neighbors(n, 25, graph_method=graph_method) == 25


def _two_far_groups() -> tuple[dict[str, tuple[float, float]], _InMemoryDataset]:
    """Six out-of-scope docs in two groups of three.

    The groups are only *modestly* apart — between-group distance about 1.5
    times the within-group spread. That is deliberate, and it is the whole reason the
    fixture is worth anything: the edge weights are a Gaussian RBF on distance,
    so two groups flung to opposite ends of the space separate under *any*
    degree, complete graph included. Such a fixture would make the tests below
    pass without the fix. At this separation the weight contrast is small
    enough that a complete graph really does have one community, which is what
    the corpora show and what the negative control pins.
    """
    vectors = {
        # S3 builds prototypes from the support docs; S2 builds them from the
        # category-name prompts. Place the prompts at the manifold mean of the
        # matching support pair so both scenarios see the same known geometry —
        # that is what makes their unknown buckets directly comparable.
        "invoice_support_1": (0.0, 0.0),
        "invoice_support_2": (0.0, 2.0),
        "contract_support_1": (10.0, 0.0),
        "contract_support_2": (10.0, 2.0),
        PROMPT.format("Invoice"): (0.0, 1.0),
        PROMPT.format("Contract"): (10.0, 1.0),
        # The whole bucket sits near x = 5, the perpendicular bisector of the
        # two prototypes, and far up the y axis — so each doc is ~equidistant
        # from both prototypes and the confidence gate rejects it. Being merely
        # far away is not enough: a doc far from *both* prototypes but nearer
        # one of them still gets a confident assignment.
        #
        # Within the bucket the split is along y: group a spans y 100-102,
        # group b spans y 103-105.
        "a1": (5.0, 100.0),
        "a2": (7.0, 101.0),
        "a3": (5.0, 102.0),
        "b1": (5.0, 103.0),
        "b2": (7.0, 104.0),
        "b3": (5.0, 105.0),
    }
    unknown = _InMemoryDataset([_Record(f"q{t}", t) for t in ("a1", "a2", "a3", "b1", "b2", "b3")])
    return vectors, unknown


def _support() -> _InMemoryDataset:
    return _InMemoryDataset(
        [
            _Record("si1", "invoice_support_1", "Invoice"),
            _Record("si2", "invoice_support_2", "Invoice"),
            _Record("sc1", "contract_support_1", "Contract"),
            _Record("sc2", "contract_support_2", "Contract"),
        ]
    )


def _unknown_cluster_names(result: Any) -> set[str]:
    return {p for p in result.predictions if p is not None and p.startswith("unknown_")}


def _unknown_grouping(result: Any) -> set[frozenset[str]]:
    """The unknown bucket's partition, as sets of doc ids.

    Compared against cluster *names* this is invariant to label numbering, and
    it pins which docs ended up together rather than only how many groups came
    back — a count alone is satisfied by any 3/3 split, including a wrong one.
    """
    groups: dict[str, set[str]] = {}
    for doc_id, pred in zip(result.doc_ids, result.predictions, strict=True):
        if pred is not None and pred.startswith("unknown_"):
            groups.setdefault(pred, set()).add(doc_id)
    return {frozenset(g) for g in groups.values()}


# The partition the geometry asks for: the a-docs together, the b-docs together.
EXPECTED_GROUPING = {frozenset({"qa1", "qa2", "qa3"}), frozenset({"qb1", "qb2", "qb3"})}


def test_s3_honours_cluster_algorithm_on_the_unknown_bucket() -> None:
    """S3 used to call ``manifold_kmeans`` directly, silently ignoring
    ``cluster_algorithm`` and every ``leiden_*`` knob — despite its own
    docstring claiming the unknown-side path matches S2's.
    """
    vectors, unknown = _two_far_groups()
    cfg = _config(
        "s3",
        ["Invoice", "Contract"],
        n_shots=2,
        threshold_confidence=0.9,
        cluster_algorithm="leiden",
        leiden_k_neighbors=25,
    )
    scenario = _with_encoder(S3PartialFewShot(cfg), vectors)

    result = scenario.fit_predict(unknown, _support())

    # Two groups in, two clusters out. Before the switch this bucket of 6 came
    # back as 6 singleton clusters (k = max(2, min(8, 6)) = 6, one per doc).
    assert _unknown_grouping(result) == EXPECTED_GROUPING


def test_emergent_bucket_does_not_collapse_under_a_corpus_scale_degree() -> None:
    """The configured degree is chosen for a whole corpus. Applied unscaled to a
    6-document bucket it builds a *complete* graph, which has exactly one
    community — so every novel document lands in a single undifferentiated
    cluster regardless of content. This is the S2 side of the same bug.
    """
    vectors, unknown = _two_far_groups()
    cfg = _config(
        "s2",
        ["Invoice", "Contract"],
        threshold_confidence=0.9,
        cluster_algorithm="leiden",
        leiden_k_neighbors=25,
    )
    scenario = _with_encoder(S2PartialLabels(cfg), vectors)

    result = scenario.fit_predict(unknown)

    assert _unknown_grouping(result) == EXPECTED_GROUPING


def test_unscaled_degree_would_collapse_the_same_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative control for the two tests above.

    Without this branch's scaling the bucket collapses to one cluster. Pinning
    that here is what stops the assertions above from passing for an unrelated
    reason — a fixture that separates under any degree would prove nothing.
    """
    vectors, unknown = _two_far_groups()
    cfg = _config(
        "s2",
        ["Invoice", "Contract"],
        threshold_confidence=0.9,
        cluster_algorithm="leiden",
        leiden_k_neighbors=25,
    )
    scenario = _with_encoder(S2PartialLabels(cfg), vectors)

    # Restore the pre-fix behaviour: pass the configured degree through as-is.
    monkeypatch.setattr(
        "clustering.scenarios.clustering.emergent_bucket_k_neighbors",
        lambda n, configured, **_: configured,
    )
    result = scenario.fit_predict(unknown)

    assert len(_unknown_cluster_names(result)) == 1


def test_s2_and_s3_partition_the_same_bucket_identically() -> None:
    """S3's docstring says the unknown-side path *is* S2's. Now that both route
    through one helper, pin it — this is the property that stops the two from
    drifting apart again.
    """
    vectors, unknown = _two_far_groups()
    common: dict[str, Any] = {
        "threshold_confidence": 0.9,
        "cluster_algorithm": "leiden",
        "leiden_k_neighbors": 25,
    }
    s2 = _with_encoder(S2PartialLabels(_config("s2", ["Invoice", "Contract"], **common)), vectors)
    s3 = _with_encoder(
        S3PartialFewShot(_config("s3", ["Invoice", "Contract"], n_shots=2, **common)), vectors
    )

    r2 = s2.fit_predict(unknown)
    r3 = s3.fit_predict(unknown, _support())

    assert r2.predictions == r3.predictions
    # Equality alone is satisfied vacuously when both sides collapse the bucket
    # to one cluster — which is exactly what they did before this scaling. Pin
    # the partition too, so the test cannot pass by agreeing on the wrong answer.
    assert _unknown_grouping(r2) == EXPECTED_GROUPING
