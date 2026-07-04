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

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from clustering.data.datasets import DocumentDataset, DocumentRecord
from clustering.scenarios.base import ScenarioResult
from dgml_core.errors import ClusteringConfigInvalid
from dgml_core.run_clustering import run_clustering
from PIL import Image


@pytest.fixture(autouse=True)
def _stub_encoders() -> Iterator[None]:
    """``Scenario.__init__`` eagerly builds the text + image encoders from the
    bundled config (the corpus-fitted ``tfidf`` text encoder, which would
    otherwise need a workspace ``corpus_dir`` to fit). These tests mock
    ``fit_predict``, so the encoders are constructed but never used — stub
    construction to a no-op MagicMock to keep the tests hermetic (no corpus
    read / model download / network)."""
    with patch("clustering.scenarios.base.build_encoder", return_value=MagicMock()):
        yield


class _FakeDataset(DocumentDataset):
    """Minimal in-memory DocumentDataset for boundary tests.

    Tests that exercise run_clustering's scenario-selection / label-rewrite
    logic don't need real images — they just need __len__ and __getitem__
    to satisfy the type. Scenarios themselves are mocked at the fit_predict
    boundary so no actual encoding runs.
    """

    def __init__(self, doc_ids: list[str]) -> None:
        self._doc_ids = list(doc_ids)
        self._image = Image.new("RGB", (8, 8))

    def __len__(self) -> int:
        return len(self._doc_ids)

    def __getitem__(self, index: int) -> DocumentRecord:
        return DocumentRecord(
            doc_id=self._doc_ids[index],
            label=None,
            image=self._image,
            text="",
            thumbnail_path=None,
        )


def _result(doc_ids: list[str], predictions: list[str | None]) -> ScenarioResult:
    return ScenarioResult(
        run_id="r",
        scenario_name="test",
        doc_ids=doc_ids,
        embeddings=torch.zeros((len(doc_ids), 2)),
        predictions=predictions,
        confidence=[None] * len(doc_ids),
        true_labels=[None] * len(doc_ids),
    )


def test_run_clustering_with_known_categories_picks_s2() -> None:
    dataset = _FakeDataset(["a", "b"])
    fake = _result(["a", "b"], ["Foo", "unknown_0"])

    with patch(
        "dgml_core.run_clustering.S2PartialLabels.fit_predict", return_value=fake
    ) as mock_s2:
        out = run_clustering(dataset, known_categories=["Foo", "Bar"])

    assert out == {"a": "Foo", "b": "unknown_0"}
    # The dataset is forwarded to S2.fit_predict; no support_dataset by default.
    mock_s2.assert_called_once_with(dataset, None)


def test_run_clustering_threads_known_categories_into_config() -> None:
    dataset = _FakeDataset(["a"])
    fake = _result(["a"], ["Receipts"])

    captured: dict[str, Any] = {}

    def _capture(
        self: Any,
        dataset: DocumentDataset,
        support_dataset: DocumentDataset | None = None,
    ) -> ScenarioResult:
        captured["scenario_known"] = list(self.config.scenario.known_categories or [])
        captured["scenario_name"] = self.config.scenario.name
        return fake

    with patch("dgml_core.run_clustering.S2PartialLabels.fit_predict", _capture):
        run_clustering(dataset, known_categories=["Receipts", "POs"])

    assert captured == {
        "scenario_known": ["Receipts", "POs"],
        "scenario_name": "s2",
    }


def test_run_clustering_no_known_categories_picks_s1_and_rewrites_labels() -> None:
    """S1 emits ``cluster_N``; run_clustering rewrites to ``unknown_N`` so
    callers see one convention regardless of scenario."""
    dataset = _FakeDataset(["a", "b", "c"])
    fake = _result(["a", "b", "c"], ["cluster_0", "cluster_1", "cluster_0"])

    with patch("dgml_core.run_clustering.S1Unsupervised.fit_predict", return_value=fake) as mock_s1:
        out = run_clustering(dataset, known_categories=[])

    assert out == {"a": "unknown_0", "b": "unknown_1", "c": "unknown_0"}
    mock_s1.assert_called_once()


def test_all_known_zero_samples_picks_s4() -> None:
    dataset = _FakeDataset(["a", "b"])
    fake = _result(["a", "b"], ["Foo", "Bar"])

    with patch("dgml_core.run_clustering.S4ZeroShot.fit_predict", return_value=fake) as mock_s4:
        out = run_clustering(
            dataset,
            known_categories=["Foo", "Bar"],
            all_categories_known=True,
        )

    assert out == {"a": "Foo", "b": "Bar"}
    mock_s4.assert_called_once_with(dataset, None)


def test_all_known_with_samples_picks_s5_and_forwards_support() -> None:
    dataset = _FakeDataset(["a", "b"])
    support = _FakeDataset(["s1", "s2"])
    fake = _result(["a", "b"], ["Foo", "Bar"])

    captured: dict[str, Any] = {}

    def _capture(
        self: Any, unknown: DocumentDataset, support_ds: DocumentDataset | None = None
    ) -> ScenarioResult:
        captured["scenario_name"] = self.config.scenario.name
        captured["n_shots"] = self.config.scenario.n_shots
        captured["support_is"] = support_ds
        return fake

    with patch("dgml_core.run_clustering.S5FullSupervised.fit_predict", _capture):
        out = run_clustering(
            dataset,
            known_categories=["Foo", "Bar"],
            all_categories_known=True,
            n_samples_per_category=3,
            support_dataset=support,
        )

    assert out == {"a": "Foo", "b": "Bar"}
    assert captured == {"scenario_name": "s5", "n_shots": 3, "support_is": support}


def test_partial_with_samples_picks_s3() -> None:
    dataset = _FakeDataset(["a", "b"])
    support = _FakeDataset(["s1"])
    fake = _result(["a", "b"], ["Foo", "unknown_0"])

    with patch(
        "dgml_core.run_clustering.S3PartialFewShot.fit_predict", return_value=fake
    ) as mock_s3:
        out = run_clustering(
            dataset,
            known_categories=["Foo"],
            n_samples_per_category=2,
            support_dataset=support,
        )

    assert out == {"a": "Foo", "b": "unknown_0"}
    mock_s3.assert_called_once_with(dataset, support)


def test_samples_without_support_dataset_raises() -> None:
    dataset = _FakeDataset(["a"])
    with pytest.raises(ValueError, match="non-empty support_dataset"):
        run_clustering(
            dataset,
            known_categories=["Foo"],
            n_samples_per_category=2,
        )


def test_negative_n_samples_raises() -> None:
    dataset = _FakeDataset(["a"])
    with pytest.raises(ValueError, match="n_samples_per_category must be >= 0"):
        run_clustering(
            dataset,
            known_categories=["Foo"],
            n_samples_per_category=-1,
        )


def test_run_clustering_drops_none_predictions() -> None:
    """A ``None`` prediction (e.g. a single-element unknown bucket in S2 that
    can't form its own cluster) is omitted from the result — the file
    surfaces as un-clustered to the caller."""
    dataset = _FakeDataset(["a", "b"])
    fake = _result(["a", "b"], ["Foo", None])

    patch_target = "dgml_core.run_clustering.S2PartialLabels.fit_predict"
    with patch(patch_target, return_value=fake):
        out = run_clustering(dataset, known_categories=["Foo"])

    assert out == {"a": "Foo"}


# ---------------------------------------------------------------------------
# overrides — partial config overlay on top of the bundled defaults
# ---------------------------------------------------------------------------


def test_overrides_replace_top_level_section() -> None:
    """A user override for one top-level key (``training``) replaces just
    that section; the others (encoder, fusion, manifold, …) keep their
    bundled defaults."""
    dataset = _FakeDataset(["a"])
    fake = _result(["a"], ["Foo"])

    captured: dict[str, Any] = {}

    def _capture(
        self: Any,
        dataset: DocumentDataset,
        support_dataset: DocumentDataset | None = None,
    ) -> ScenarioResult:
        captured["epochs"] = self.config.training.epochs
        captured["identity_projector"] = self.config.training.identity_projector
        captured["encoder_text_name"] = self.config.encoder_text.name
        return fake

    overrides = {"training": {"epochs": 5}}
    with patch("dgml_core.run_clustering.S2PartialLabels.fit_predict", _capture):
        run_clustering(dataset, known_categories=["Foo"], overrides=overrides)

    # Overridden field wins.
    assert captured["epochs"] == 5
    # Sibling field within the same section: the deep-merge keeps the bundled
    # default (identity_projector=true from clustering_config.json) rather than
    # replacing the whole `training` block with just the overridden key.
    assert captured["identity_projector"] is True
    # An unrelated top-level section is untouched.
    assert captured["encoder_text_name"] == "tfidf"


def test_overrides_scenario_regime_cannot_be_overridden() -> None:
    """The scenario *regime* (``name``) is driven by the arguments to
    ``run_clustering`` — a user override for it is silently discarded so
    callers can't accidentally pin the wrong scenario for their inputs."""
    dataset = _FakeDataset(["a"])
    fake = _result(["a"], ["Foo"])

    captured: dict[str, Any] = {}

    def _capture(
        self: Any,
        dataset: DocumentDataset,
        support_dataset: DocumentDataset | None = None,
    ) -> ScenarioResult:
        captured["scenario_name"] = self.config.scenario.name
        return fake

    # User tries to force S4 despite passing known_categories without
    # all_categories_known=True — should be ignored, real scenario stays S2.
    with patch("dgml_core.run_clustering.S2PartialLabels.fit_predict", _capture):
        run_clustering(
            dataset,
            known_categories=["Foo"],
            overrides={"scenario": {"name": "s4"}},
        )

    assert captured["scenario_name"] == "s2"


def test_overrides_scenario_algorithm_knobs_apply() -> None:
    """Scenario clustering-algorithm knobs (``cluster_algorithm``,
    ``leiden_*``, ``reduce_*``) ARE overridable — they layer under the
    dynamic regime, so the leiden/UMAP bundled defaults take effect and
    operators can retune them via config.json / --config."""
    dataset = _FakeDataset(["a"])
    fake = _result(["a"], ["Foo"])

    captured: dict[str, Any] = {}

    def _capture(
        self: Any,
        dataset: DocumentDataset,
        support_dataset: DocumentDataset | None = None,
    ) -> ScenarioResult:
        captured["scenario_name"] = self.config.scenario.name
        captured["cluster_algorithm"] = self.config.scenario.cluster_algorithm
        captured["leiden_k_neighbors"] = self.config.scenario.leiden_k_neighbors
        captured["reduce_dim"] = self.config.scenario.reduce_dim
        return fake

    with patch("dgml_core.run_clustering.S2PartialLabels.fit_predict", _capture):
        run_clustering(
            dataset,
            known_categories=["Foo"],
            overrides={"scenario": {"cluster_algorithm": "leiden", "leiden_k_neighbors": 15}},
        )

    # The regime (name) is still dynamic, but the algorithm knobs survive —
    # the override wins for k, and the bundled default (reduce_dim 10, UMAP)
    # is preserved rather than reset to the schema default.
    assert captured["scenario_name"] == "s2"
    assert captured["cluster_algorithm"] == "leiden"
    assert captured["leiden_k_neighbors"] == 15
    assert captured["reduce_dim"] == 10


def test_overrides_invalid_field_raises_clustering_config_invalid() -> None:
    """An override that fails pydantic validation surfaces as
    :class:`ClusteringConfigInvalid` (a :class:`DgmlError`), not a raw
    pydantic exception — so the CLI can render a clean error envelope."""
    dataset = _FakeDataset(["a"])
    with pytest.raises(ClusteringConfigInvalid, match="validation"):
        run_clustering(
            dataset,
            known_categories=["Foo"],
            overrides={"encoder_text": {"name": "definitely_not_a_real_encoder"}},
        )
