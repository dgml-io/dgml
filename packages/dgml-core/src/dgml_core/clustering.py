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

"""Clustering over unassigned files in a workspace.

Two modes, both driven through the same :func:`clustering` entry point:

- **fresh** — no clusters exist yet (or the caller forces it): every
  unassigned file is clustered from scratch into emergent ``"unknown_N"``
  buckets (scenario S1), which are then LLM-named into new DocSets.
- **incremental** — clusters already exist and a new batch of files has
  arrived (the "S3" workflow). Each existing DocSet becomes a category
  prototype reconstructed from *all* of its already-assigned members'
  embeddings (few-shot / S3). New files are assigned to the nearest
  existing DocSet when they fit, and the leftovers form fresh
  ``"unknown_N"`` clusters that are LLM-named into new DocSets. This
  covers the three incremental cases: everything fits an existing
  cluster, some fit and some form new clusters, or nothing fits.

``mode="auto"`` (the default) picks incremental when the workspace
already has DocSets and fresh otherwise. ``mode="fresh"`` /
``mode="incremental"`` force the respective path; forcing incremental on
a workspace with no DocSets raises :class:`IncrementalWithoutClusters`.

:func:`clustering_internal` invokes ``dgml.run_clustering`` over the
workspace's unassigned files; :func:`clustering` walks the resulting
``file_id → cluster_name`` map and assigns each file to a DocSet, asking
the configured vision LLM (via :func:`propose_new_docset_for_files`) to
name DocSets for unmatched clusters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal, get_args

from .classification import (
    load_classification_config,
    propose_new_docset_for_files,
)
from .dataset import WorkspaceFileDataset
from .docsets import DocSetStore
from .errors import (
    ClusteringConfigInvalid,
    CorruptMetadata,
    DgmlError,
    IncrementalWithoutClusters,
)
from .pages import PAGE_GLOB
from .run_clustering import resolve_text_settings, run_clustering_detailed
from .storage import Workspace, read_config, read_json
from .utils import unassigned_file_ids

# Cap on how many files from a single cluster get sent to the LLM when
# naming a new DocSet. Each contributes up to ``config.max_pages`` page
# images; this bound keeps the LLM call cost/context predictable on
# large clusters. The first N files in the cluster are used as the
# naming sample.
MAX_FILES_PER_CLUSTER_NAMING = 2

# The incremental path reconstructs each existing DocSet's prototype from
# *all* of its members that have a rendered page image (bounded only by
# this ceiling, which keeps a pathologically large DocSet from dominating
# embedding cost). Embeddings are content-hashed and cached by the encoder
# layer, so re-embedding already-seen members is cheap on repeat runs.
MAX_SUPPORT_SAMPLES_PER_DOCSET = 64

# Clustering run modes. ``auto`` resolves to ``incremental`` when the
# workspace already has DocSets, else ``fresh``.
ClusterMode = Literal["auto", "fresh", "incremental"]
CLUSTER_MODES: tuple[str, ...] = get_args(ClusterMode)

# Named, bundled config presets. ``light`` mirrors the bundled default
# (CPU-only tf-idf + Leiden/UMAP); ``medium`` uses a dense sentence
# encoder (large CPU / Apple MPS); ``heavy`` uses a large dense encoder +
# HDBSCAN (GPU). Each maps to a ``clustering_preset_<name>.json`` resource
# shipped alongside this module.
CONFIG_PRESETS: tuple[str, ...] = ("light", "medium", "heavy")


def load_clustering_preset(name: str) -> dict[str, Any]:
    """Load a bundled config preset (``light`` / ``medium`` / ``heavy``).

    Returns the preset's overrides dict (same shape as the ``clustering``
    section of ``<workspace>/config.json``). Raises
    :class:`ClusteringConfigInvalid` for an unknown preset name.
    """
    if name not in CONFIG_PRESETS:
        raise ClusteringConfigInvalid(
            f"unknown clustering preset {name!r}; choose one of {', '.join(CONFIG_PRESETS)} "
            "or pass a path to a config JSON."
        )
    text = (resources.files("dgml_core") / f"clustering_preset_{name}.json").read_text(
        encoding="utf-8"
    )
    data: dict[str, Any] = json.loads(text)
    return data


def resolve_clustering_overrides(
    workspace: Workspace,
    *,
    config: str | None,
) -> dict[str, Any]:
    """Resolve the effective clustering overrides for a run.

    ``config`` is the raw value of the CLI ``--config`` flag:

    - ``None`` → the ``clustering`` section of ``<workspace>/config.json``
      (:func:`load_clustering_overrides`), or ``{}`` when absent.
    - a preset name (``light`` / ``medium`` / ``heavy``) →
      :func:`load_clustering_preset`.
    - anything else → treated as a path to a standalone config JSON
      (:func:`load_clustering_config_file`).
    """
    if config is None:
        return load_clustering_overrides(workspace)
    if config in CONFIG_PRESETS:
        return load_clustering_preset(config)
    return load_clustering_config_file(Path(config))


@dataclass
class _InternalResult:
    """Outcome of clustering the unassigned files (pre-DocSet-assignment)."""

    clusters: dict[str, str]
    render_skipped: list[str]
    confidences: dict[str, float | None] = field(default_factory=dict)
    mode: str = "fresh"
    known_categories: list[str] = field(default_factory=list)


def load_clustering_overrides(workspace: Workspace) -> dict[str, Any]:
    """Read the ``clustering`` section of ``<workspace>/config.json``.

    Returns ``{}`` when the file doesn't exist or has no ``clustering``
    section — the bundled defaults in
    :data:`dgml_core.run_clustering._CONFIG_RESOURCE` stand on their own.
    Raises :class:`ClusteringConfigInvalid` when the file exists but is
    malformed (not valid JSON, not a JSON object) or the ``clustering``
    section itself isn't a JSON object. Field-level validation happens
    later in :func:`dgml.run_clustering._build_config` after the merge.
    """
    if not workspace.config_path.exists():
        return {}
    try:
        data = read_config(workspace.config_path)
    except CorruptMetadata as exc:
        raise ClusteringConfigInvalid(f"{workspace.config_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ClusteringConfigInvalid(f"{workspace.config_path} must contain a JSON object")
    section = data.get("clustering")
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ClusteringConfigInvalid("'clustering' section in config.json must be a JSON object")
    return section


def load_clustering_config_file(path: Path) -> dict[str, Any]:
    """Read a standalone clustering config JSON file (the CLI ``--config`` flag).

    The file is a JSON object holding the same fields the ``clustering``
    section of ``<workspace>/config.json`` would (e.g. ``encoder_text``,
    ``scenario``); its contents are used directly as the overrides
    deep-merged over the bundled defaults in
    :data:`dgml_core.run_clustering._CONFIG_RESOURCE`. When supplied it
    *replaces* the workspace's ``clustering`` section for that run.

    Raises :class:`ClusteringConfigInvalid` when the file is missing, not
    valid JSON, or not a JSON object. Field-level validation happens later
    in :func:`dgml.run_clustering._build_config` after the merge.
    """
    if not path.exists():
        raise ClusteringConfigInvalid(f"clustering config file not found: {path}")
    try:
        data = read_json(path)
    except CorruptMetadata as exc:
        raise ClusteringConfigInvalid(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ClusteringConfigInvalid(f"{path} must contain a JSON object")
    return data


def clustering(
    workspace: Workspace,
    *,
    skip_existing: bool = False,
    config: str | None = None,
    mode: str = "auto",
    debug: bool = False,
) -> dict[str, Any]:
    """Cluster the unassigned files in ``workspace`` and assign each to a DocSet.

    Calls :func:`clustering_internal` to map each unassigned file to a
    cluster name, then in two passes:

    1. Files whose cluster name matches an existing DocSet are assigned
       to that DocSet immediately.
    2. Files whose cluster doesn't match are grouped by cluster name;
       for each unmatched cluster :func:`propose_new_docset_for_files`
       sends up to :data:`MAX_FILES_PER_CLUSTER_NAMING` files to the
       vision LLM, the proposed DocSet is created, and every file in
       that cluster is assigned to it.

    Partial success: if classification config is missing/invalid, or the
    LLM call for a given cluster fails, the files in *that* cluster
    land in ``failed_file_ids`` while every other cluster (matched or
    successfully named) is still assigned. Files whose page rendering
    failed at ingest time (no ``page_1.png``) also land in
    ``failed_file_ids``. The function does not raise.

    Returns a JSON-serializable dict. The first three keys are the core
    contract; the rest are **additive** incremental-workflow fields
    (optional — consumers can ignore them):

    - ``clusters``: the ``file_id → docset_name`` map for every file
      whose cluster was successfully assigned. Placeholder labels from
      the algorithm (e.g. ``"unknown_0"``) are rewritten to the actual
      DocSet name the file landed in — either an existing DocSet's
      name or the new name the LLM proposed. Files that failed to
      assign keep their placeholder label here (and also appear in
      ``failed_file_ids``).
    - ``failed_file_ids``: file IDs that were not assigned to any
      DocSet — either because their first-page image was missing or
      because their cluster needed LLM naming and that naming failed
      (missing config, no page images, provider error, …).
    - ``skipped``: ``True`` only when ``skip_existing`` was passed and there
      were no unassigned files (the clusterer never ran); ``False`` on every
      actual clustering run. Always present so callers can read it directly.
    - ``mode``: the effective run mode — ``"fresh"`` or ``"incremental"``
      (``"auto"`` is resolved to one of these before running).
    - ``n_assigned_existing``: number of files assigned to a DocSet that
      already existed before this run (only meaningful for incremental).
    - ``n_new_clusters``: number of *new* DocSets created this run.
    - ``assignments``: ``{file_id: {"docset", "confidence", "is_new"}}``
      for every successfully-assigned file — ``confidence`` is the
      nearest-prototype confidence in ``[0, 1]`` (``null`` for emergent
      clusters), ``is_new`` flags files that landed in a DocSet created
      this run.

    ``skip_existing`` makes the whole call a no-op (returns ``skipped: True``,
    empty maps) when every file is already assigned — cheap to use on resume.

    ``config`` selects the clustering configuration: ``None`` uses the
    workspace ``config.json`` ``clustering`` section (bundled defaults when
    absent); a preset name (``light`` / ``medium`` / ``heavy``) loads a
    bundled preset; anything else is treated as a path to a standalone
    config JSON. See :func:`resolve_clustering_overrides`.

    ``mode`` selects fresh vs incremental clustering; see the module
    docstring. ``"incremental"`` on a workspace with no DocSets raises
    :class:`IncrementalWithoutClusters`.
    """
    if mode not in CLUSTER_MODES:
        raise ClusteringConfigInvalid(
            f"unknown clustering mode {mode!r}; choose one of {', '.join(CLUSTER_MODES)}."
        )
    if skip_existing and not unassigned_file_ids(workspace):
        return {
            "clusters": {},
            "failed_file_ids": [],
            "skipped": True,
            "mode": "incremental" if DocSetStore(workspace).list_all() else "fresh",
            "n_assigned_existing": 0,
            "n_new_clusters": 0,
            "assignments": {},
        }

    internal = clustering_internal(workspace, config=config, mode=mode)
    clusters = internal.clusters

    docset_store = DocSetStore(workspace)
    name_to_id = {d.name: d.id for d in docset_store.list_all()}
    existing_names = set(name_to_id)

    # Per-file assignment detail (additive output). Seeded here for files
    # matched to existing DocSets; extended below for LLM-named clusters.
    assignments: dict[str, dict[str, Any]] = {}

    # Pass 1: assign files whose cluster matches an existing DocSet;
    # collect the rest, grouped by cluster name, for the LLM pass. An
    # existing DocSet may carry an extraction schema — assignment then
    # auto-extracts (soft-fail into the assignment's `extraction` block).
    # Newly-created DocSets (pass 2) can't have a schema yet, so they
    # assign plainly.
    from .extraction import add_file_and_extract

    unmatched: dict[str, list[str]] = {}
    for file_id, cluster_name in clusters.items():
        if cluster_name in name_to_id:
            extraction_block = add_file_and_extract(
                workspace, name_to_id[cluster_name], file_id, write_stats=debug, debug=debug
            )
            assignments[file_id] = {
                "docset": cluster_name,
                "confidence": internal.confidences.get(file_id),
                "is_new": False,
            }
            if extraction_block is not None:
                assignments[file_id]["extraction"] = extraction_block
        else:
            unmatched.setdefault(cluster_name, []).append(file_id)

    failed_file_ids: list[str] = list(internal.render_skipped)
    n_new_clusters = 0

    # Pass 2: name each unmatched cluster via LLM, create its DocSet,
    # assign all of its files. Per-cluster failures are recorded into
    # `failed_file_ids` rather than raised.
    if unmatched:
        try:
            classification_config = load_classification_config(workspace)
        except DgmlError:
            # Without config, no unmatched cluster can be named. Every file
            # in those clusters fails to assign.
            for cluster_files in unmatched.values():
                failed_file_ids.extend(cluster_files)
            unmatched = {}

        for cluster_files in unmatched.values():
            sample = cluster_files[:MAX_FILES_PER_CLUSTER_NAMING]
            try:
                decision = propose_new_docset_for_files(
                    workspace, sample, config=classification_config, debug=debug
                )
            except DgmlError:
                failed_file_ids.extend(cluster_files)
                continue
            new_name = decision.new_name or ""
            new_ds = docset_store.create(
                name=new_name,
                description=decision.new_description or "",
                key_questions=list(decision.new_key_questions),
            )
            n_new_clusters += 1
            for file_id in cluster_files:
                docset_store.add_file(new_ds.id, file_id)
                # Replace the algorithm's placeholder cluster name (e.g.
                # "unknown_0") with the actual DocSet name the file landed
                # in, so the returned `clusters` map reflects where each
                # file is really assigned.
                clusters[file_id] = new_name
                assignments[file_id] = {
                    "docset": new_name,
                    "confidence": None,
                    "is_new": True,
                }

    n_assigned_existing = sum(
        1 for detail in assignments.values() if detail["docset"] in existing_names
    )
    return {
        "clusters": clusters,
        "failed_file_ids": failed_file_ids,
        "skipped": False,
        "mode": internal.mode,
        "n_assigned_existing": n_assigned_existing,
        "n_new_clusters": n_new_clusters,
        "assignments": assignments,
    }


def clustering_internal(
    workspace: Workspace,
    config: str | None = None,
    mode: str = "auto",
) -> _InternalResult:
    """Cluster every unassigned file into a cluster name.

    A cluster name is either the name of an existing DocSet (the file is
    judged to belong with that DocSet's existing members) or
    ``"unknown_<n>"`` (a fresh cluster proposed for files that don't fit
    any existing DocSet). The actual clustering is delegated to
    :func:`dgml.run_clustering.run_clustering_detailed`.

    ``mode`` selects fresh vs incremental (see the module docstring):

    - **fresh** ⇒ pure unsupervised S1 over the unassigned files. Existing
      DocSets, if any, are ignored as prototypes.
    - **incremental** ⇒ each existing DocSet is a category whose prototype
      is reconstructed from *all* its members that have a rendered page
      (few-shot S3, or name-only S2 if a DocSet has no usable members).
      Requires at least one DocSet.
    - **auto** ⇒ incremental when DocSets exist, else fresh.

    Files whose first-page image is missing (page rendering failed at
    ingest time) can't be embedded; they're returned in
    :attr:`_InternalResult.render_skipped` so the outer :func:`clustering`
    can route them into ``failed_file_ids``.

    Returns an :class:`_InternalResult`. Empty workspace ⇒ an empty
    result carrying the resolved ``mode``.
    """
    docset_store = DocSetStore(workspace)
    docsets = docset_store.list_all()
    effective_mode = _resolve_mode(mode, has_docsets=bool(docsets))

    file_ids = unassigned_file_ids(workspace)
    if not file_ids:
        return _InternalResult(clusters={}, render_skipped=[], mode=effective_mode)

    usable: list[str] = []
    skipped: list[str] = []
    for fid in file_ids:
        if any(workspace.file_pages_dir(fid).glob(PAGE_GLOB)):
            usable.append(fid)
        else:
            skipped.append(fid)

    if not usable:
        return _InternalResult(clusters={}, render_skipped=skipped, mode=effective_mode)

    # In fresh mode, ignore existing DocSets as prototypes: cluster from
    # scratch (S1). In incremental mode, they drive known-category
    # prototypes.
    known_categories = [d.name for d in docsets]
    proto_docsets = docsets
    if effective_mode == "fresh":
        known_categories = []
        proto_docsets = []

    # Clustering overrides. ``config`` may be a preset name, a path, or None
    # (workspace config.json section). Missing config/section → empty dict
    # and the bundled defaults stand. A malformed file/section/preset raises
    # ClusteringConfigInvalid, which the CLI surfaces as an error envelope.
    overrides = resolve_clustering_overrides(workspace, config=config)
    # Point corpus-fitted text encoders at the workspace files/ dir and
    # learn the text view the configured encoder expects, so the dataset
    # assembles record.text under that same view.
    text_view, overrides = resolve_text_settings(workspace.files_dir, overrides)

    dataset = WorkspaceFileDataset(workspace, usable, text_view=text_view)

    # Build a labeled support set from existing DocSet members: for the
    # incremental path, reconstruct each category's prototype from *all*
    # of its members with a rendered page image (bounded by
    # MAX_SUPPORT_SAMPLES_PER_DOCSET). Embeddings are cached by content
    # hash, so re-embedding already-seen members is cheap. With samples in
    # hand, run_clustering escalates from name-only (S2) to few-shot (S3)
    # prototypes; ``max_shots`` tells S3 how many per category to average.
    support_file_ids: list[str] = []
    support_labels: dict[str, str] = {}
    max_shots = 0
    for docset in proto_docsets:
        picked = 0
        for fid in docset_store.list_files(docset.id):
            if picked >= MAX_SUPPORT_SAMPLES_PER_DOCSET:
                break
            if any(workspace.file_pages_dir(fid).glob(PAGE_GLOB)):
                support_file_ids.append(fid)
                support_labels[fid] = docset.name
                picked += 1
        max_shots = max(max_shots, picked)

    if support_file_ids:
        support_dataset = WorkspaceFileDataset(
            workspace, support_file_ids, labels=support_labels, text_view=text_view
        )
        detailed = run_clustering_detailed(
            dataset,
            known_categories=known_categories,
            n_samples_per_category=max_shots,
            support_dataset=support_dataset,
            overrides=overrides,
        )
    else:
        detailed = run_clustering_detailed(
            dataset, known_categories=known_categories, overrides=overrides
        )

    clusters = {doc_id: pred.cluster_name for doc_id, pred in detailed.items()}
    confidences = {doc_id: pred.confidence for doc_id, pred in detailed.items()}
    return _InternalResult(
        clusters=clusters,
        render_skipped=skipped,
        confidences=confidences,
        mode=effective_mode,
        known_categories=known_categories,
    )


def _resolve_mode(mode: str, *, has_docsets: bool) -> str:
    """Resolve ``auto`` to ``fresh`` / ``incremental`` and validate forcing.

    ``incremental`` requires at least one existing DocSet — forcing it on
    an empty workspace raises :class:`IncrementalWithoutClusters` so the
    caller gets a clear error rather than a silent fresh run.
    """
    if mode == "auto":
        return "incremental" if has_docsets else "fresh"
    if mode == "incremental" and not has_docsets:
        raise IncrementalWithoutClusters(
            "mode='incremental' requires at least one existing DocSet to assign into, "
            "but this workspace has none. Run a fresh clustering first "
            "(mode='fresh' or 'auto'), or create DocSets before incremental clustering."
        )
    return mode
