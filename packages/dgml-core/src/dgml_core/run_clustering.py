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

"""Cluster a :class:`DocumentDataset` into category names.

Bridges from "I have a bag of documents and maybe a list of known
category names (and maybe labeled samples per category)" to a concrete
``{doc_id: cluster_name}`` map. Scenario choice is automatic from the
arguments:

============================  ===========================  ====================  =========
``known_categories``          ``all_categories_known``     ``n_samples``         scenario
============================  ===========================  ====================  =========
empty                         (ignored)                    (ignored)             S1
non-empty                     ``False`` (default)          ``0`` (default)       S2
non-empty                     ``False``                    ``> 0``               S3
non-empty                     ``True``                     ``0``                 S4
non-empty                     ``True``                     ``> 0``               S5
============================  ===========================  ====================  =========

- ``all_categories_known=False`` lets the scenario add ``"unknown_N"``
  buckets for documents that don't fit any known prototype.
- ``all_categories_known=True`` constrains every document to one of the
  known categories — no emergent clusters.
- ``n_samples_per_category > 0`` switches to few-shot / fully-supervised
  prototype construction, which requires ``support_dataset``.

S1 emits raw ``"cluster_N"`` labels; we rewrite those to ``"unknown_N"``
so callers see one consistent emergent-cluster naming convention
regardless of scenario.

Static defaults (encoder, fusion, manifold, training, …) live in
``clustering_config.json`` shipped alongside this module. Callers can
override any subset of them via the ``overrides=`` parameter on
:func:`run_clustering` (typically read from the ``clustering`` section of
a workspace ``config.json``; see
:func:`dgml.clustering.load_clustering_overrides`). The scenario *regime*
(``name`` / ``known_categories`` / ``n_shots``) depends on the arguments
above and is fixed at call time, but the scenario's clustering-algorithm
knobs (``cluster_algorithm``, ``leiden_*``, ``reduce_*``) are overridable
like any other field — so the leiden + UMAP defaults in the bundled config
actually take effect, and operators can retune them.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from clustering.config.schema import Config
from clustering.data.datasets import DocumentDataset
from clustering.scenarios.base import Scenario
from clustering.scenarios.s1_unsupervised import S1Unsupervised
from clustering.scenarios.s2_partial_labels import S2PartialLabels
from clustering.scenarios.s3_partial_few_shot import S3PartialFewShot
from clustering.scenarios.s4_zero_shot import S4ZeroShot
from clustering.scenarios.s5_full_supervised import S5FullSupervised

from .errors import ClusteringConfigInvalid

_CONFIG_RESOURCE = "clustering_config.json"


def resolve_text_settings(
    files_dir: Path, overrides: dict[str, Any] | None
) -> tuple[str, dict[str, Any]]:
    """Resolve the effective text view and inject the corpus directory.

    Corpus-fitted text encoders (``tfidf``) need
    ``encoder_text.extra.corpus_dir`` pointed at the workspace ``files/``
    dir to fit document frequencies over the whole corpus, and the dataset
    must assemble ``record.text`` under the *same* ``text_view`` the encoder
    fits on. Both are derived here from the merged bundled-default +
    ``overrides`` config so the dataset and the encoder always agree.

    Returns ``(text_view, overrides')`` where ``overrides'`` is ``overrides``
    with ``corpus_dir`` merged into ``encoder_text.extra``. Injecting
    ``corpus_dir`` is harmless for encoders that don't read it (dense
    encoders ignore unknown ``extra`` keys).
    """
    base: dict[str, Any] = json.loads(
        (resources.files("dgml_core") / _CONFIG_RESOURCE).read_text(encoding="utf-8")
    )
    merged = _deep_merge(base, overrides or {})
    encoder_text = merged.get("encoder_text")
    extra = encoder_text.get("extra") if isinstance(encoder_text, dict) else None
    text_view = str(extra.get("text_view", "full")) if isinstance(extra, dict) else "full"
    injected = _deep_merge(
        overrides or {},
        {"encoder_text": {"extra": {"corpus_dir": str(files_dir)}}},
    )
    return text_view, injected


@dataclass(frozen=True)
class DocPrediction:
    """A single document's clustering outcome.

    ``cluster_name`` is the assigned cluster/category name (an existing
    category name, or an emergent ``"unknown_N"`` bucket). ``confidence``
    is the assignment confidence in ``[0, 1]`` when the scenario produced
    one (nearest-prototype scenarios S2-S5), or ``None`` for emergent
    clusters and for scenarios that don't expose a confidence (S1).
    """

    cluster_name: str
    confidence: float | None


def run_clustering(
    dataset: DocumentDataset,
    *,
    known_categories: list[str],
    all_categories_known: bool = False,
    n_samples_per_category: int = 0,
    support_dataset: DocumentDataset | None = None,
    overrides: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, str]:
    """Cluster ``dataset`` and return ``{doc_id: cluster_name}``.

    Thin wrapper over :func:`run_clustering_detailed` that drops the
    per-document confidence. See that function and the module docstring for
    the scenario-selection matrix and the argument semantics.
    """
    return {
        doc_id: pred.cluster_name
        for doc_id, pred in run_clustering_detailed(
            dataset,
            known_categories=known_categories,
            all_categories_known=all_categories_known,
            n_samples_per_category=n_samples_per_category,
            support_dataset=support_dataset,
            overrides=overrides,
            cache_dir=cache_dir,
        ).items()
    }


def run_clustering_detailed(
    dataset: DocumentDataset,
    *,
    known_categories: list[str],
    all_categories_known: bool = False,
    n_samples_per_category: int = 0,
    support_dataset: DocumentDataset | None = None,
    overrides: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> dict[str, DocPrediction]:
    """Cluster ``dataset`` and return ``{doc_id: DocPrediction}``.

    Same behaviour as :func:`run_clustering`, but also surfaces the
    per-document assignment confidence carried on
    :class:`~clustering.scenarios.base.ScenarioResult`. The incremental
    (S3-workflow) path in :mod:`dgml_core.clustering` uses this to report
    how confident each newly-ingested document's assignment to an existing
    cluster is.

    See the module docstring for the scenario-selection matrix.

    ``support_dataset`` is consumed only when
    ``n_samples_per_category > 0`` (S3 / S5); it must contain labeled
    examples whose ``label`` matches one of ``known_categories``.

    ``overrides`` is a partial config dict deep-merged on top of the
    bundled :data:`_CONFIG_RESOURCE` defaults — typically the
    ``clustering`` section of a workspace ``config.json`` (loaded by
    :func:`dgml.clustering.load_clustering_overrides`). Per-key overlay:
    a partial ``{"encoder_text": {"name": "e5"}}`` leaves every other
    section (fusion, manifold, training, …) at its bundled default.
    Unrecognized fields surface as a :class:`ClusteringConfigInvalid`.
    """
    if n_samples_per_category < 0:
        raise ValueError(f"n_samples_per_category must be >= 0; got {n_samples_per_category}.")
    if n_samples_per_category > 0 and (support_dataset is None or len(support_dataset) == 0):
        raise ValueError(
            "n_samples_per_category > 0 requires a non-empty support_dataset of "
            "labeled examples per known category."
        )

    config = _build_config(
        known_categories=known_categories,
        all_categories_known=all_categories_known,
        n_samples_per_category=n_samples_per_category,
        overrides=overrides,
        cache_dir=cache_dir,
    )
    scenario = _pick_scenario(
        config,
        known_categories=known_categories,
        all_categories_known=all_categories_known,
        n_samples_per_category=n_samples_per_category,
    )
    result = scenario.fit_predict(dataset, support_dataset)

    # S1's raw labels are "cluster_N"; rewrite to "unknown_N" so the
    # caller's contract ("emergent cluster ⇒ 'unknown_N'") is the same
    # across scenarios. Every other scenario already either produces
    # "unknown_N" directly (S2 / S3) or only known category names
    # (S4 / S5), so this is a no-op for them.
    rewrite = _cluster_to_unknown if not known_categories else _identity

    return {
        doc_id: DocPrediction(cluster_name=rewrite(pred), confidence=conf)
        for doc_id, pred, conf in zip(
            result.doc_ids, result.predictions, result.confidence, strict=True
        )
        if pred is not None
    }


def _pick_scenario(
    config: Config,
    *,
    known_categories: list[str],
    all_categories_known: bool,
    n_samples_per_category: int,
) -> Scenario:
    if not known_categories:
        return S1Unsupervised(config)
    if all_categories_known:
        return S5FullSupervised(config) if n_samples_per_category > 0 else S4ZeroShot(config)
    return S3PartialFewShot(config) if n_samples_per_category > 0 else S2PartialLabels(config)


def _identity(label: str) -> str:
    return label


def _cluster_to_unknown(label: str) -> str:
    """Rewrite S1's ``"cluster_N"`` → ``"unknown_N"``. Leaves anything else
    untouched (defensive — should never fire in practice)."""
    if label.startswith("cluster_"):
        return "unknown_" + label[len("cluster_") :]
    return label


def _build_config(
    *,
    known_categories: list[str],
    all_categories_known: bool,
    n_samples_per_category: int,
    overrides: dict[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> Config:
    """Build a complete :class:`Config` from the bundled config plus a
    dynamic scenario section, with optional user overrides merged in.

    Static fields (encoder, fusion, manifold, corpus, logger, training)
    come from :data:`_CONFIG_RESOURCE` shipped alongside this module.
    Anything missing from the JSON falls through to the field's schema
    default — the file only needs to spell out overrides.

    ``overrides`` is then deep-merged on top: per-key overlay so users
    can replace just one setting (e.g. ``encoder_text``) and leave the
    rest at the bundled defaults. Pydantic validation runs after the
    merge and raises :class:`ClusteringConfigInvalid` on bad fields.

    The scenario section is special: its *regime* (``name``,
    ``known_categories``, ``n_shots``) is computed here per the matrix in
    the module docstring and always wins, but its clustering-algorithm
    knobs (``cluster_algorithm``, ``leiden_*``, ``reduce_*``, …) come from
    the merged config so operators can tune them. The dynamic regime is
    layered on top of the merged scenario, overriding only the keys it owns.

    ``corpus.root`` is a placeholder — ``run_clustering`` takes the
    dataset directly, so the corpus config isn't actually consulted by
    the scenario pipeline. The schema requires the field, so the
    bundled config pins it to a valid path.
    """
    config_text = (resources.files("dgml_core") / _CONFIG_RESOURCE).read_text(encoding="utf-8")
    fields: dict[str, Any] = json.loads(config_text)
    if overrides:
        fields = _deep_merge(fields, overrides)
    if cache_dir is not None:
        # Infrastructure, not a user knob: wire the workspace embedding cache so
        # re-embedding unchanged files across runs is cheap. Set after the merge
        # so a workspace config.json can't accidentally clobber it.
        fields["cache_dir"] = str(cache_dir)
    # The scenario *regime* — ``name`` plus ``known_categories`` / ``n_shots``
    # — is dynamic per call (driven by the arguments above) and always wins, so
    # callers can't pin the wrong scenario for their inputs. The clustering-
    # algorithm knobs (``cluster_algorithm``, ``leiden_*``, ``reduce_*``, …)
    # come from the merged bundled-default + overrides scenario so operators can
    # tune them via config.json / --config. Layer the dynamic regime on top so
    # it overrides only the keys it owns and leaves the algorithm knobs intact.
    base_scenario = fields.get("scenario")
    if not isinstance(base_scenario, dict):
        base_scenario = {}
    fields["scenario"] = _deep_merge(
        base_scenario,
        _scenario_section(
            known_categories=known_categories,
            all_categories_known=all_categories_known,
            n_samples_per_category=n_samples_per_category,
        ),
    )
    try:
        return Config.model_validate(fields)
    except Exception as exc:
        raise ClusteringConfigInvalid(
            f"clustering config failed validation: {type(exc).__name__}: {exc}"
        ) from exc


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``.

    Dict-valued keys recurse; everything else replaces. Returns a fresh
    dict — neither input is mutated. Used to overlay a user's partial
    ``clustering`` config on top of the bundled defaults.
    """
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        existing = out.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            out[key] = _deep_merge(existing, value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _scenario_section(
    *,
    known_categories: list[str],
    all_categories_known: bool,
    n_samples_per_category: int,
) -> dict[str, Any]:
    if not known_categories:
        return {"name": "s1"}
    if all_categories_known:
        name = "s5" if n_samples_per_category > 0 else "s4"
    else:
        name = "s3" if n_samples_per_category > 0 else "s2"
    section: dict[str, Any] = {"name": name, "known_categories": list(known_categories)}
    if n_samples_per_category > 0:
        section["n_shots"] = n_samples_per_category
    return section
