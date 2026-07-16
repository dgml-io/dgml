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

"""Tests for the dgml-side pieces of the clustering pipeline.

The outer ``clustering()`` and ``dgml cluster`` CLI command are covered
in ``test_cli.py``; this file focuses on the workspace-aware dataset and
``clustering_internal`` boundary (skipping files with no rendered page,
threading known categories).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core.clustering import (
    DEFAULT_INCREMENTAL_NOVELTY_QUANTILE,
    _resolve_mode,
    _with_incremental_novelty_default,
    clustering_internal,
    load_clustering_overrides,
    load_clustering_preset,
    resolve_clustering_overrides,
)
from dgml_core.dataset import WorkspaceFileDataset
from dgml_core.docsets import DocSetStore
from dgml_core.errors import ClusteringConfigInvalid, IncrementalWithoutClusters
from dgml_core.run_clustering import DocPrediction
from dgml_core.storage import Workspace


def _dp(cluster_name: str, confidence: float | None = None) -> DocPrediction:
    """Shorthand for a mocked ``run_clustering_detailed`` outcome."""
    return DocPrediction(cluster_name=cluster_name, confidence=confidence)


def _seed_file(workspace: Workspace, file_id: str) -> None:
    """Materialize a minimal File record on disk so list_all() finds it."""
    from dgml_core.models import FileRecord
    from dgml_core.storage import write_json_atomic

    record = FileRecord(
        id=file_id,
        original_path=f"/fake/{file_id}.pdf",
        original_filename=f"{file_id}.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=1,
        text_mode="digital",
    )
    workspace.file_dir(file_id).mkdir(parents=True, exist_ok=True)
    write_json_atomic(workspace.file_json_path(file_id), record.to_json())


def _seed_page_image(workspace: Workspace, file_id: str) -> Path:
    """Write a tiny but valid PNG to ``page_1.png`` for ``file_id``."""
    from PIL import Image

    page_dir = workspace.file_pages_dir(file_id)
    page_dir.mkdir(parents=True, exist_ok=True)
    path = page_dir / "page_1.png"
    Image.new("RGB", (8, 8), color=(123, 200, 50)).save(path, "PNG")
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    return ws


# ---------------------------------------------------------------------------
# WorkspaceFileDataset
# ---------------------------------------------------------------------------


def test_workspace_file_dataset_returns_record_with_page_image(workspace: Workspace) -> None:
    _seed_file(workspace, "f1")
    _seed_page_image(workspace, "f1")

    ds = WorkspaceFileDataset(workspace, ["f1"])
    assert len(ds) == 1

    record = ds[0]
    assert record.doc_id == "f1"
    assert record.label is None
    assert record.text == ""
    assert record.thumbnail_path is None
    # Image loaded from page_1.png — confirm by checking size matches what we wrote.
    assert record.image.size == (8, 8)


def test_workspace_file_dataset_lazy_loads(workspace: Workspace) -> None:
    """Constructing the dataset doesn't read any images — only __getitem__ does."""
    _seed_file(workspace, "a")
    _seed_file(workspace, "b")
    # Note: neither "a" nor "b" has a page_1.png. Construction must still succeed.
    ds = WorkspaceFileDataset(workspace, ["a", "b"])
    assert len(ds) == 2
    # Accessing an item without a page image raises — confirms lazy loading.
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_workspace_file_dataset_threads_labels(workspace: Workspace) -> None:
    """When ``labels`` is provided, ``__getitem__`` returns the matching
    label; files missing from the map come back with ``label=None``."""
    _seed_file(workspace, "a")
    _seed_file(workspace, "b")
    _seed_page_image(workspace, "a")
    _seed_page_image(workspace, "b")

    ds = WorkspaceFileDataset(workspace, ["a", "b"], labels={"a": "Contracts"})
    assert ds[0].label == "Contracts"
    assert ds[1].label is None


def test_workspace_file_dataset_iterates(workspace: Workspace) -> None:
    _seed_file(workspace, "a")
    _seed_file(workspace, "b")
    _seed_page_image(workspace, "a")
    _seed_page_image(workspace, "b")

    ds = WorkspaceFileDataset(workspace, ["a", "b"])
    records = list(ds)
    assert [r.doc_id for r in records] == ["a", "b"]


# ---------------------------------------------------------------------------
# clustering_internal
# ---------------------------------------------------------------------------


def test_clustering_internal_empty_workspace(workspace: Workspace) -> None:
    result = clustering_internal(workspace)
    assert result.clusters == {}
    assert result.render_skipped == []
    # No DocSets ⇒ auto resolves to fresh.
    assert result.mode == "fresh"


def test_clustering_internal_skips_files_without_page_image(workspace: Workspace) -> None:
    """Files whose page_1.png is missing land in the skipped list and are
    never sent to the clusterer."""
    _seed_file(workspace, "with_image")
    _seed_page_image(workspace, "with_image")
    _seed_file(workspace, "no_image")

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"with_image": _dp("unknown_0")},
    ) as mock_run:
        result = clustering_internal(workspace)

    assert result.clusters == {"with_image": "unknown_0"}
    assert result.render_skipped == ["no_image"]
    # Only the usable file was passed to the clusterer.
    dataset_arg = mock_run.call_args[0][0]
    assert dataset_arg.file_ids == ["with_image"]


def test_clustering_internal_threads_existing_docset_names(workspace: Workspace) -> None:
    """Existing DocSet names are passed to the clusterer as ``known_categories``,
    so the underlying scenario can match files against them."""
    DocSetStore(workspace).create(name="Contracts")
    DocSetStore(workspace).create(name="Receipts")
    _seed_file(workspace, "f1")
    _seed_page_image(workspace, "f1")

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"f1": _dp("Contracts", 0.8)},
    ) as mock_run:
        result = clustering_internal(workspace)

    assert sorted(mock_run.call_args.kwargs["known_categories"]) == ["Contracts", "Receipts"]
    # DocSets exist ⇒ auto resolves to incremental, and confidence is threaded.
    assert result.mode == "incremental"
    assert result.confidences == {"f1": 0.8}


def test_clustering_internal_builds_support_set_from_docset_members(workspace: Workspace) -> None:
    """When DocSets have members with rendered pages, those files are
    sampled (capped per-docset) into a labeled support_dataset and
    n_samples_per_category is set so run_clustering escalates to S3."""
    store = DocSetStore(workspace)
    contracts = store.create(name="Contracts")
    receipts = store.create(name="Receipts")

    # Three Contracts members; first two have page images, third doesn't.
    for fid in ("c1", "c2", "c3"):
        _seed_file(workspace, fid)
        store.add_file(contracts.id, fid)
    _seed_page_image(workspace, "c1")
    _seed_page_image(workspace, "c2")

    # One Receipts member with a page image.
    _seed_file(workspace, "r1")
    _seed_page_image(workspace, "r1")
    store.add_file(receipts.id, "r1")

    # Unassigned file to drive the unknown dataset.
    _seed_file(workspace, "u1")
    _seed_page_image(workspace, "u1")

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"u1": _dp("Contracts", 0.7)},
    ) as mock_run:
        clustering_internal(workspace)

    kwargs = mock_run.call_args.kwargs
    # Incremental reconstructs prototypes from all usable members; here the
    # busiest category (Contracts) has 2 with page images (c1, c2 — not c3),
    # so n_samples_per_category (the S3 per-category shot cap) is 2.
    assert kwargs["n_samples_per_category"] == 2
    support_ds = kwargs["support_dataset"]
    assert support_ds is not None
    assert sorted(support_ds.file_ids) == ["c1", "c2", "r1"]
    assert support_ds.labels == {"c1": "Contracts", "c2": "Contracts", "r1": "Receipts"}


def test_clustering_internal_skips_support_when_docsets_have_no_usable_files(
    workspace: Workspace,
) -> None:
    """A DocSet with no rendered members contributes no samples; with
    zero usable samples overall, run_clustering falls back to the
    name-only S2 path (no n_samples_per_category, no support_dataset)."""
    DocSetStore(workspace).create(name="Contracts")
    _seed_file(workspace, "u1")
    _seed_page_image(workspace, "u1")

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"u1": _dp("unknown_0")},
    ) as mock_run:
        clustering_internal(workspace)

    kwargs = mock_run.call_args.kwargs
    assert "n_samples_per_category" not in kwargs
    assert "support_dataset" not in kwargs


def test_clustering_internal_all_unusable_skips_clusterer(workspace: Workspace) -> None:
    _seed_file(workspace, "no_image")

    with patch("dgml_core.clustering.run_clustering_detailed") as mock_run:
        result = clustering_internal(workspace)

    assert result.clusters == {}
    assert result.render_skipped == ["no_image"]
    mock_run.assert_not_called()


def test_clustering_internal_forwards_workspace_overrides(workspace: Workspace) -> None:
    """The ``clustering`` section of ``<workspace>/config.json`` is loaded
    and forwarded to ``run_clustering`` as ``overrides=`` so users can
    override individual settings (encoder, training, …) without copying
    the whole bundled default. ``corpus_dir`` is additionally injected into
    ``encoder_text.extra`` so corpus-fitted text encoders can fit."""
    _seed_file(workspace, "f1")
    _seed_page_image(workspace, "f1")
    workspace.config_path.write_text(
        json.dumps({"clustering": {"training": {"epochs": 7}}}),
        encoding="utf-8",
    )

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"f1": _dp("unknown_0")},
    ) as mock_run:
        clustering_internal(workspace)

    forwarded = mock_run.call_args.kwargs["overrides"]
    assert forwarded["training"] == {"epochs": 7}
    assert forwarded["encoder_text"]["extra"]["corpus_dir"] == str(workspace.files_dir)


def test_clustering_internal_passes_empty_overrides_when_no_config(workspace: Workspace) -> None:
    """No config.json ⇒ only the injected ``corpus_dir`` is forwarded
    (bundled defaults otherwise stand), not a different keyword shape that
    would skip the path."""
    _seed_file(workspace, "f1")
    _seed_page_image(workspace, "f1")

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"f1": _dp("unknown_0")},
    ) as mock_run:
        clustering_internal(workspace)

    assert mock_run.call_args.kwargs["overrides"] == {
        "encoder_text": {"extra": {"corpus_dir": str(workspace.files_dir)}}
    }


# ---------------------------------------------------------------------------
# incremental novelty-gate default — _with_incremental_novelty_default
# ---------------------------------------------------------------------------


def test_novelty_default_injected_when_no_gate() -> None:
    """With no gate set, a conservative quantile gate is injected."""
    out = _with_incremental_novelty_default({})
    assert out == {"scenario": {"threshold_quantile": DEFAULT_INCREMENTAL_NOVELTY_QUANTILE}}


def test_novelty_default_merges_into_existing_scenario() -> None:
    """The gate is added alongside unrelated scenario knobs, not replacing them."""
    out = _with_incremental_novelty_default({"scenario": {"leiden_resolution": 1.5}})
    assert out["scenario"] == {
        "leiden_resolution": 1.5,
        "threshold_quantile": DEFAULT_INCREMENTAL_NOVELTY_QUANTILE,
    }


@pytest.mark.parametrize("gate", ["threshold", "threshold_confidence", "threshold_quantile"])
def test_novelty_default_suppressed_by_any_explicit_gate(gate: str) -> None:
    """Any explicit gate the user set wins; no default is layered on top."""
    overrides = {"scenario": {gate: 0.5}}
    assert _with_incremental_novelty_default(overrides) == overrides


def test_novelty_default_respects_explicit_null_gate() -> None:
    """Setting a gate to ``null`` deliberately disables gating — the default
    must not override that choice."""
    overrides = {"scenario": {"threshold_quantile": None}}
    assert _with_incremental_novelty_default(overrides) == overrides


def test_novelty_default_does_not_mutate_input() -> None:
    original = {"scenario": {"leiden_resolution": 1.0}}
    _with_incremental_novelty_default(original)
    assert original == {"scenario": {"leiden_resolution": 1.0}}


def test_clustering_internal_incremental_injects_novelty_default(workspace: Workspace) -> None:
    """The incremental embedding path forwards the conservative quantile gate
    so new categories can emerge instead of every doc being absorbed."""
    DocSetStore(workspace).create(name="Contracts")
    _seed_file(workspace, "u1")
    _seed_page_image(workspace, "u1")

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"u1": _dp("Contracts", 0.7)},
    ) as mock_run:
        result = clustering_internal(workspace)

    assert result.mode == "incremental"
    scenario = mock_run.call_args.kwargs["overrides"]["scenario"]
    assert scenario["threshold_quantile"] == DEFAULT_INCREMENTAL_NOVELTY_QUANTILE


def test_clustering_internal_fresh_does_not_inject_novelty_default(workspace: Workspace) -> None:
    """Fresh mode clusters from scratch (S1, no prototypes) — no gate injected."""
    DocSetStore(workspace).create(name="Contracts")
    _seed_file(workspace, "u1")
    _seed_page_image(workspace, "u1")

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"u1": _dp("unknown_0")},
    ) as mock_run:
        clustering_internal(workspace, mode="fresh")

    scenario = mock_run.call_args.kwargs["overrides"].get("scenario", {})
    assert "threshold_quantile" not in scenario


def test_clustering_internal_incremental_respects_user_gate(workspace: Workspace) -> None:
    """A user-set gate in config.json wins over the injected default."""
    DocSetStore(workspace).create(name="Contracts")
    _seed_file(workspace, "u1")
    _seed_page_image(workspace, "u1")
    workspace.config_path.write_text(
        json.dumps({"clustering": {"scenario": {"threshold_confidence": 0.5}}}),
        encoding="utf-8",
    )

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"u1": _dp("Contracts", 0.7)},
    ) as mock_run:
        clustering_internal(workspace)

    scenario = mock_run.call_args.kwargs["overrides"]["scenario"]
    assert scenario["threshold_confidence"] == 0.5
    assert "threshold_quantile" not in scenario


# ---------------------------------------------------------------------------
# load_clustering_overrides — reading the workspace config.json
# ---------------------------------------------------------------------------


def _write_config(workspace: Workspace, payload: dict[str, Any]) -> None:
    workspace.config_path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_clustering_overrides_returns_empty_when_no_config(workspace: Workspace) -> None:
    """No config.json at all ⇒ the bundled defaults stand."""
    assert load_clustering_overrides(workspace) == {}


def test_load_clustering_overrides_returns_empty_when_no_section(workspace: Workspace) -> None:
    """A config.json without a ``clustering`` section is treated the same
    as a missing file — bundled defaults stand."""
    _write_config(workspace, {"classification": {"model": "gemini/gemini-3.1-flash-lite"}})
    assert load_clustering_overrides(workspace) == {}


def test_load_clustering_overrides_reads_section(workspace: Workspace) -> None:
    _write_config(
        workspace,
        {
            "classification": {"model": "gemini/gemini-3.1-flash-lite"},
            "clustering": {"training": {"epochs": 42}},
        },
    )
    assert load_clustering_overrides(workspace) == {"training": {"epochs": 42}}


def test_load_clustering_overrides_section_not_object_raises(workspace: Workspace) -> None:
    _write_config(workspace, {"clustering": "oops"})
    with pytest.raises(ClusteringConfigInvalid, match="must be a JSON object"):
        load_clustering_overrides(workspace)


def test_load_clustering_overrides_corrupt_json_raises(workspace: Workspace) -> None:
    workspace.config_path.write_text("{this is not valid json", encoding="utf-8")
    with pytest.raises(ClusteringConfigInvalid, match="is not valid JSON"):
        load_clustering_overrides(workspace)


def test_load_clustering_overrides_top_level_not_object_raises(workspace: Workspace) -> None:
    workspace.config_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ClusteringConfigInvalid, match="must contain a JSON object"):
        load_clustering_overrides(workspace)


# ---------------------------------------------------------------------------
# mode resolution — auto / fresh / incremental
# ---------------------------------------------------------------------------


def test_resolve_mode_auto_picks_by_docsets() -> None:
    assert _resolve_mode("auto", has_docsets=False) == "fresh"
    assert _resolve_mode("auto", has_docsets=True) == "incremental"


def test_resolve_mode_forced_values_pass_through() -> None:
    assert _resolve_mode("fresh", has_docsets=True) == "fresh"
    assert _resolve_mode("incremental", has_docsets=True) == "incremental"


def test_resolve_mode_incremental_without_docsets_raises() -> None:
    with pytest.raises(IncrementalWithoutClusters, match="requires at least one existing DocSet"):
        _resolve_mode("incremental", has_docsets=False)


def test_clustering_internal_fresh_mode_ignores_existing_docsets(workspace: Workspace) -> None:
    """`mode='fresh'` clusters from scratch (S1) even when DocSets exist —
    no known_categories, no support set."""
    DocSetStore(workspace).create(name="Contracts")
    _seed_file(workspace, "u1")
    _seed_page_image(workspace, "u1")

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"u1": _dp("unknown_0")},
    ) as mock_run:
        result = clustering_internal(workspace, mode="fresh")

    assert result.mode == "fresh"
    kwargs = mock_run.call_args.kwargs
    assert kwargs["known_categories"] == []
    assert "support_dataset" not in kwargs


def test_clustering_internal_incremental_without_docsets_raises(workspace: Workspace) -> None:
    _seed_file(workspace, "u1")
    _seed_page_image(workspace, "u1")
    with pytest.raises(IncrementalWithoutClusters):
        clustering_internal(workspace, mode="incremental")


# ---------------------------------------------------------------------------
# config presets — small / light / medium / heavy + override resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["small", "light", "medium", "heavy"])
def test_load_clustering_preset_known(name: str) -> None:
    preset = load_clustering_preset(name)
    assert isinstance(preset, dict)
    # Presets are lean override files deep-merged over the bundled defaults;
    # they only spell out the keys that differ, so these are the ones common
    # to every tier (each may also override the encoders on top).
    assert {"fusion", "manifold", "scenario"} <= set(preset)


def test_load_clustering_preset_unknown_raises() -> None:
    with pytest.raises(ClusteringConfigInvalid, match="unknown clustering preset"):
        load_clustering_preset("gigantic")


def test_resolve_overrides_none_reads_workspace_section(workspace: Workspace) -> None:
    _write_config(workspace, {"clustering": {"training": {"epochs": 3}}})
    assert resolve_clustering_overrides(workspace, config=None) == {"training": {"epochs": 3}}


def test_resolve_overrides_preset_name(workspace: Workspace) -> None:
    assert resolve_clustering_overrides(workspace, config="medium") == load_clustering_preset(
        "medium"
    )


def test_resolve_overrides_path(workspace: Workspace, tmp_path: Path) -> None:
    cfg = tmp_path / "custom.json"
    cfg.write_text(json.dumps({"scenario": {"leiden_k_neighbors": 9}}), encoding="utf-8")
    assert resolve_clustering_overrides(workspace, config=str(cfg)) == {
        "scenario": {"leiden_k_neighbors": 9}
    }
