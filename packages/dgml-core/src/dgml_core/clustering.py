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

import copy
import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal, get_args

from .classification import (
    ClassificationDecision,
    load_classification_config,
    propose_new_docset_for_files,
)
from .dataset import WorkspaceFileDataset
from .docsets import DocSetStore
from .errors import (
    ClassificationFailed,
    ClusteringConfigInvalid,
    CorruptMetadata,
    DgmlError,
    IncrementalWithoutClusters,
)
from .llm_clustering import llm_cluster_files
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

# Clustering *method* — how documents are grouped, orthogonal to the
# fresh/incremental mode above.
#
# - ``embedding`` (default): the encode → project → statistical-cluster
#   pipeline in :mod:`dgml_core.run_clustering`. The right choice once a
#   corpus is large enough for tf-idf / neighbor-graph statistics to be
#   meaningful.
# - ``llm``: send every document's page images to the vision LLM in one
#   call and let it partition them (:func:`dgml_core.llm_clustering.llm_cluster_files`).
#   Built for *very small* corpora, where the embedding pipeline has too
#   little signal to cluster reliably.
# - ``auto``: pick ``llm`` when the number of clusterable files is at or
#   below :data:`SMALL_CORPUS_MAX_FILES`, else ``embedding``.
ClusterMethod = Literal["auto", "embedding", "llm"]
CLUSTER_METHODS: tuple[str, ...] = get_args(ClusterMethod)

# At or below this many clusterable files, ``method="auto"`` routes to the
# LLM partitioner instead of the embedding pipeline. Small enough that the
# whole corpus fits comfortably in one multimodal prompt; large enough that
# bigger corpora, where embeddings start to pay off, take the statistical
# path.
SMALL_CORPUS_MAX_FILES = 8

# The three composable novelty gates on ``ScenarioConfig`` (see
# ``clustering.config.schema``). A document is routed to the "unknown" bucket
# — i.e. treated as *novel* and allowed to open a new DocSet — iff any active
# gate flags it. In the framework all three default to ``None`` (no gating),
# which is the right library default but the wrong *product* default for
# incremental runs: with every gate off, S2/S3 force every incoming document
# into its nearest existing DocSet and nothing is ever novel.
_NOVELTY_GATE_KEYS: tuple[str, ...] = ("threshold", "threshold_confidence", "threshold_quantile")

# Conservative novelty gate shipped by default on the incremental CLI path so
# "none fit" and "some fit" can actually happen out of the box. We use the
# *quantile* gate because it is the most corpus-robust of the three: it
# auto-calibrates a distance cutoff to the incoming batch's own nearest-
# prototype distance distribution, so it needs no hand-tuned, manifold-unit-
# dependent number and travels across encoders/manifolds unchanged. ``0.9``
# keeps the closest 90 % of a batch as "known" and flags only the farthest
# 10 % as novel — deliberately cautious, so a genuinely homogeneous batch is
# barely disturbed while clear out-of-distribution documents can still open a
# new category. Users override it (or turn gating off with an explicit
# ``threshold_quantile: null``) via the ``scenario`` config section.
DEFAULT_INCREMENTAL_NOVELTY_QUANTILE = 0.9


def _with_incremental_novelty_default(overrides: dict[str, Any]) -> dict[str, Any]:
    """Return ``overrides`` with a conservative novelty gate for incremental runs.

    The framework's three novelty gates all default to ``None`` (see
    :data:`_NOVELTY_GATE_KEYS`), which in incremental mode absorbs every
    incoming document into its nearest existing DocSet. When the caller has not
    set *any* gate, inject :data:`DEFAULT_INCREMENTAL_NOVELTY_QUANTILE` as
    ``scenario.threshold_quantile`` so new categories can emerge out of the box.

    Any explicit gate the user provides — including ``threshold_quantile: null``
    to deliberately disable gating — wins and suppresses the default. The input
    is never mutated; a fresh dict is returned.
    """
    scenario = overrides.get("scenario")
    if isinstance(scenario, dict) and any(k in scenario for k in _NOVELTY_GATE_KEYS):
        # The user spoke about novelty gating (even setting a gate to null to
        # turn it off) — respect their choice and don't layer a default on top.
        return overrides
    merged = copy.deepcopy(overrides)
    scenario_out = merged.get("scenario")
    if not isinstance(scenario_out, dict):
        scenario_out = {}
    scenario_out["threshold_quantile"] = DEFAULT_INCREMENTAL_NOVELTY_QUANTILE
    merged["scenario"] = scenario_out
    return merged


# Named, bundled config presets, ordered by compute budget. The tiers scale
# by adding image/vision embeddings: ``small`` and ``light`` are CPU-only,
# text-only (tf-idf + Leiden), ``small`` skipping UMAP for tiny corpora and
# ``light`` (the default) reducing with UMAP; ``medium`` fuses a 2B vision
# encoder into the text signal; ``heavy`` clusters on an 8B vision encoder
# alone (GPU). Each maps to a ``clustering_preset_<name>.json`` resource
# shipped alongside this module.
CONFIG_PRESETS: tuple[str, ...] = ("small", "light", "medium", "heavy")


def load_clustering_preset(name: str) -> dict[str, Any]:
    """Load a bundled config preset (``small`` / ``light`` / ``medium`` / ``heavy``).

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
    - a preset name (``small`` / ``light`` / ``medium`` / ``heavy``) →
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
    method: str = "embedding"
    known_categories: list[str] = field(default_factory=list)
    # LLM-method only: proposed DocSet metadata for each emergent
    # ``"unknown_N"`` cluster, keyed by cluster name. When present,
    # :func:`clustering` creates those DocSets from the proposal instead of
    # issuing a second LLM naming call. Empty for the embedding method.
    proposals: dict[str, ClassificationDecision] = field(default_factory=dict)


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
    method: str = "embedding",
    small_corpus_threshold: int = SMALL_CORPUS_MAX_FILES,
    debug: bool = False,
) -> dict[str, Any]:
    """Cluster the unassigned files in ``workspace`` and assign each to a DocSet.

    Calls :func:`clustering_internal` to map each unassigned file to a
    cluster name, then in two passes:

    1. Files whose cluster name matches an existing DocSet are assigned
       to that DocSet immediately.
    2. Files whose cluster doesn't match are grouped by cluster name;
       for each unmatched cluster the DocSet to create is named either
       from a proposal the clusterer already produced (the ``llm`` method
       partitions and names in one call) or, failing that, by sending up to
       :data:`MAX_FILES_PER_CLUSTER_NAMING` files to
       :func:`propose_new_docset_for_files`. The proposed DocSet is created
       and every file in that cluster is assigned to it.

    ``method`` selects how documents are grouped (orthogonal to ``mode``):
    ``"embedding"`` (default) uses the statistical pipeline; ``"llm"`` uses
    the vision-LLM partitioner built for very small corpora; ``"auto"``
    picks ``"llm"`` when at most ``small_corpus_threshold`` files are
    clusterable, else ``"embedding"``. See :data:`ClusterMethod`.

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
    absent); a preset name (``small`` / ``light`` / ``medium`` / ``heavy``)
    loads a bundled preset; anything else is treated as a path to a standalone
    config JSON. See :func:`resolve_clustering_overrides`.

    ``mode`` selects fresh vs incremental clustering; see the module
    docstring. ``"incremental"`` on a workspace with no DocSets raises
    :class:`IncrementalWithoutClusters`.
    """
    if mode not in CLUSTER_MODES:
        raise ClusteringConfigInvalid(
            f"unknown clustering mode {mode!r}; choose one of {', '.join(CLUSTER_MODES)}."
        )
    if method not in CLUSTER_METHODS:
        raise ClusteringConfigInvalid(
            f"unknown clustering method {method!r}; choose one of {', '.join(CLUSTER_METHODS)}."
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

    internal = clustering_internal(
        workspace,
        config=config,
        mode=mode,
        method=method,
        small_corpus_threshold=small_corpus_threshold,
        debug=debug,
    )
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

    # Pass 2: for each unmatched cluster, obtain a DocSet proposal, create
    # the DocSet, and assign all of its files. The proposal comes either
    # from the clusterer itself (the `llm` method partitions *and* names in
    # a single call — see `internal.proposals`) or, for the embedding
    # method, from a per-cluster `propose_new_docset_for_files` naming call.
    # Per-cluster failures are recorded into `failed_file_ids`, not raised.
    if unmatched:
        # Only the naming fallback needs a classification config; clusters
        # that already carry a proposal don't. Load it lazily and tolerate
        # its absence so a fully pre-named (llm-method) run works even with
        # no `classification` config section.
        classification_config = None
        if any(internal.proposals.get(name) is None for name in unmatched):
            try:
                classification_config = load_classification_config(workspace)
            except DgmlError:
                classification_config = None

        for cluster_name, cluster_files in unmatched.items():
            decision = internal.proposals.get(cluster_name)
            if decision is None:
                if classification_config is None:
                    # No proposal and no config to make one → these files
                    # can't be named into a DocSet.
                    failed_file_ids.extend(cluster_files)
                    continue
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
    method: str = "embedding",
    small_corpus_threshold: int = SMALL_CORPUS_MAX_FILES,
    debug: bool = False,
) -> _InternalResult:
    """Cluster every unassigned file into a cluster name.

    A cluster name is either the name of an existing DocSet (the file is
    judged to belong with that DocSet's existing members) or
    ``"unknown_<n>"`` (a fresh cluster proposed for files that don't fit
    any existing DocSet).

    ``method`` selects the clustering engine (orthogonal to ``mode``):

    - **embedding** (default) ⇒ the statistical pipeline via
      :func:`dgml.run_clustering.run_clustering_detailed`.
    - **llm** ⇒ the vision-LLM partitioner
      (:func:`dgml_core.llm_clustering.llm_cluster_files`), for very small
      corpora. Emergent clusters carry a naming proposal on
      :attr:`_InternalResult.proposals`.
    - **auto** ⇒ ``llm`` when at most ``small_corpus_threshold`` files are
      clusterable, else ``embedding``.

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

    # Route to the LLM partitioner for small corpora before doing any of the
    # embedding-pipeline setup below (which the LLM path doesn't need).
    effective_method = _resolve_method(
        method, n_usable=len(usable), threshold=small_corpus_threshold
    )
    if effective_method == "llm":
        return _llm_cluster_internal(
            workspace,
            usable,
            skipped,
            proto_docsets=proto_docsets,
            known_categories=known_categories,
            effective_mode=effective_mode,
            debug=debug,
        )

    # Clustering overrides. ``config`` may be a preset name, a path, or None
    # (workspace config.json section). Missing config/section → empty dict
    # and the bundled defaults stand. A malformed file/section/preset raises
    # ClusteringConfigInvalid, which the CLI surfaces as an error envelope.
    overrides = resolve_clustering_overrides(workspace, config=config)
    # Incremental runs assign into existing DocSets via S2/S3's nearest-
    # prototype gate. With the framework's all-``None`` gate defaults that gate
    # never fires, so every new document is forced into its closest DocSet and
    # nothing is ever novel. Ship a conservative quantile gate by default (only
    # when the user hasn't set one) so new categories can emerge. Fresh mode
    # clusters from scratch (S1, no prototypes) and needs no gate.
    if effective_mode == "incremental":
        overrides = _with_incremental_novelty_default(overrides)
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
        method="embedding",
        known_categories=known_categories,
    )


def _resolve_method(method: str, *, n_usable: int, threshold: int) -> str:
    """Resolve ``auto`` to ``llm`` / ``embedding`` from the corpus size.

    A corpus at or below ``threshold`` clusterable files is "small" — too
    small for the embedding statistics to be reliable — so it goes to the
    LLM partitioner. ``embedding`` / ``llm`` are returned unchanged.
    """
    if method == "auto":
        return "llm" if n_usable <= threshold else "embedding"
    return method


def _llm_cluster_internal(
    workspace: Workspace,
    usable: list[str],
    skipped: list[str],
    *,
    proto_docsets: list[Any],
    known_categories: list[str],
    effective_mode: str,
    debug: bool,
) -> _InternalResult:
    """Cluster ``usable`` files with the vision-LLM partitioner.

    ``proto_docsets`` (empty in fresh mode) are offered to the model as
    existing categories a group may be assigned to; matched files come back
    keyed to the DocSet's name, emergent ones as ``"unknown_N"`` with a
    naming proposal in :attr:`_InternalResult.proposals`.

    A missing/invalid ``classification`` config propagates (it's a setup
    error the caller must fix). A *runtime* LLM failure soft-fails the whole
    batch — every usable file is routed into ``render_skipped`` so the outer
    :func:`clustering` reports them in ``failed_file_ids`` rather than
    crashing, mirroring the embedding path's per-cluster soft-fail.
    """
    ccfg = load_classification_config(workspace)
    try:
        result = llm_cluster_files(
            workspace, usable, config=ccfg, docsets=proto_docsets, debug=debug
        )
    except ClassificationFailed:
        return _InternalResult(
            clusters={},
            render_skipped=[*skipped, *usable],
            mode=effective_mode,
            method="llm",
            known_categories=known_categories,
        )

    render_skipped = list(skipped)
    for fid in result.failed_file_ids:
        if fid not in render_skipped:
            render_skipped.append(fid)
    return _InternalResult(
        clusters=result.clusters,
        render_skipped=render_skipped,
        confidences=dict.fromkeys(result.clusters),
        mode=effective_mode,
        method="llm",
        known_categories=known_categories,
        proposals=result.proposals,
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
