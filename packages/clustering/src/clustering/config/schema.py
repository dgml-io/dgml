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

"""Pydantic schemas for the resolved framework configuration.

Every Hydra config group in ``configs/`` maps to one of these models. Models
are frozen (immutable) and forbid extra fields so typos become loud errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── Enum-like literals (also used as ClearML / W&B tags) ─────────────────
FusionName = Literal["none", "concat_norm", "late_concat", "cross_attention", "gated"]
ManifoldName = Literal["euclidean", "spherical", "hyperbolic", "product"]
ScenarioName = Literal["s1", "s2", "s3", "s4", "s5"]
# Text-encoder names that share the SentenceTransformer adapter — the
# instruction-tuned / Matryoshka / MTEB-frontier crew (E5, BGE, GTE,
# Stella, Jina v3). Distinct registry names rather than a single
# "sentence_transformers" entry so each gets its own Hydra config file
# with the model's canonical prefix templates, default model_id, etc.
EncoderName = Literal[
    "dummy",
    "st_minilm",
    "e5",
    "bge",
    "gte",
    "stella",
    "jina",
    "tfidf",
    "dit",
    "vit",
    "donut",
    "qwen_vl",
    "qwen3_vl_embedding",
    "qwen3_vl_embedding_2b",
]
LoggerName = Literal["none", "clearml", "wandb", "multi"]
LossName = Literal[
    "contrastive",
    "triplet",
    "prototypical",
    "pseudo_label",
    "knn_contrastive",
    "cross_modal",
]
# Losses that train the projector WITHOUT ground-truth labels (S1 regime).
# The supervised trio above consumes corpus labels; these three derive their
# training signal from the data itself:
#   pseudo_label    — DeepCluster/PCL-style EM: cluster → pseudo-labels →
#                     prototypical loss → re-cluster (optionally Sinkhorn-
#                     balanced so no cluster swallows the corpus).
#   knn_contrastive — SCAN-style: kNN positives in the frozen fused space +
#                     learnable prototypes + entropy regularization.
#   cross_modal     — CLIP-style InfoNCE between the text-only and
#                     image-only fused views of each document.
UNSUPERVISED_LOSSES: frozenset[str] = frozenset({"pseudo_label", "knn_contrastive", "cross_modal"})
DeviceSpec = str  # "auto" | "cuda" | "cuda:N" | "mps" | "cpu" — validated by utils.resolve_device


class _StrictModel(BaseModel):
    """Base for all config sections — strict, immutable, no extras."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# ── Encoder ───────────────────────────────────────────────────────────────
class EncoderConfig(_StrictModel):
    """One encoder (text or image side)."""

    name: EncoderName
    model_id: str | None = None
    embedding_dim: int = 384
    max_length: int | None = None
    # Whether this encoder emits multi-vector ``tokens`` in addition to
    # ``pooled``. Set explicitly so downstream consumers can be validated at
    # construction time without instantiating the model.
    multi_vector: bool = False
    # ── Instruction-prefix templates (E5 / BGE / GTE / Stella / Jina) ─────
    # Instruction-tuned text embedders are trained with task prefixes
    # (e.g. E5's ``"query: "`` / ``"passage: "``); using the wrong side
    # or omitting them costs several MTEB points. We surface both, and
    # the SentenceTransformer adapter prepends ``doc_prefix`` to every
    # input it encodes — that matches our corpus-encoding use case.
    # ``query_prefix`` is kept on the config so a future asymmetric
    # ``encode_query`` path can pick it up without a schema change.
    # Leave both ``None`` for models without instruction tuning
    # (e.g. ``st_minilm``).
    query_prefix: str | None = None
    doc_prefix: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# ── Fusion ────────────────────────────────────────────────────────────────
class FusionConfig(_StrictModel):
    name: FusionName
    hidden_dim: int = 256
    output_dim: int = 256
    n_heads: int = 4
    dropout: float = 0.1
    # Which single modality ``none`` fusion passes through unchanged.
    # Ignored by every other fusion (which consume both modalities).
    # ``text`` is often the stronger zero-shot signal for document
    # categorization; ``image`` is the default for layout-driven corpora.
    prefer_modality: Literal["image", "text"] = "image"
    # ── concat_norm blend weight ──────────────────────────────────────────
    # Only consulted by the parameter-free ``concat_norm`` fusion. Each
    # modality's pooled vector is L2-normalized, then text is scaled by
    # ``1 - image_weight`` and image by ``image_weight`` before
    # concatenation, so a single knob trades off text vs layout signal.
    # ``0.0`` ⇒ text only (image contributes nothing), ``1.0`` ⇒ image only,
    # ``0.5`` ⇒ equal-norm blend. Ignored by every other fusion.
    image_weight: float = 0.5


# ── Manifold ──────────────────────────────────────────────────────────────
class ManifoldComponent(_StrictModel):
    name: Literal["euclidean", "spherical", "hyperbolic"]
    dim: int
    curvature: float = 1.0


class ManifoldConfig(_StrictModel):
    name: ManifoldName
    dim: int = 256
    curvature: float = 1.0
    components: list[ManifoldComponent] | None = None

    @model_validator(mode="after")
    def _check_product(self) -> ManifoldConfig:
        if self.name == "product":
            if not self.components or len(self.components) < 2:
                raise ValueError(
                    f"Product manifold requires at least 2 components; got {self.components!r}."
                )
            total = sum(c.dim for c in self.components)
            if total != self.dim:
                raise ValueError(
                    f"Product manifold dim ({self.dim}) must equal sum of component dims ({total})."
                )
        return self


# ── Scenario ──────────────────────────────────────────────────────────────
class ScenarioConfig(_StrictModel):
    name: ScenarioName
    k_clusters: int | None = None
    n_shots: int | None = None
    known_categories: list[str] | None = None
    # ── Unknown-bucket gating (S2 / S3) ───────────────────────────────────
    # All three thresholds compose: a document is routed to the "unknown"
    # bucket iff ANY active threshold says so. Leave them all ``None`` to
    # assign every document to its nearest prototype.
    threshold: float | None = None
    """Absolute manifold-distance cutoff. Docs with nearest-prototype
    distance > threshold → unknown bucket. Manifold-unit-dependent
    (radians for spherical, Poincaré units for hyperbolic, …) — needs
    re-tuning when ``manifold=`` changes."""
    threshold_confidence: float | None = None
    """Softmax-over-prototype-distances floor in ``[0, 1]``. Docs with
    nearest-prototype confidence < this value → unknown bucket. Manifold-
    independent because it operates on the softmax distribution, not raw
    distance."""
    threshold_quantile: float | None = None
    """If set, auto-calibrate ``threshold`` to the ``q``-quantile of the
    *empirical* nearest-prototype distance distribution. ``q=0.8`` keeps
    the closest 80 % of documents as known; the rest go to the unknown
    bucket. Manifold-independent because the calibration adapts to the
    distance scale at hand. Composes with the other two: the auto-set
    threshold is then OR-combined with any explicit ``threshold_confidence``."""
    # Clustering algorithm for S1 / unknown-bucket in S2.
    # - ``kmeans``: Riemannian Lloyd's algorithm, requires ``k_clusters``.
    # - ``hdbscan``: density-based, non-parametric in k (``k_clusters`` is
    #   ignored when set). Noise points are routed to a ``*_noise`` bucket
    #   instead of an integer-indexed cluster.
    # - ``graph_cc``: pairwise manifold-distance → radius graph →
    #   connected components. Radius is either explicit (``graph_cc_radius``)
    #   or auto-determined (``graph_cc_r_method``).
    # - ``leiden``: Leiden community detection (Traag, Waltman & van Eck
    #   2019) on a manifold-distance graph (k-NN / mutual-k-NN / radius)
    #   with modularity or CPM quality. Requires the ``[graph]`` extra.
    # - ``dbscan``: density-based on the precomputed manifold-distance matrix.
    #   ``eps`` is explicit (``dbscan_eps``) or auto (``dbscan_r_method``,
    #   reusing graph_cc's k-NN-knee / MST-gap heuristics).
    # - ``optics``: density-based with varying density; extracts clusters from
    #   the reachability plot (no single ``eps``). Tune ``optics_min_samples``
    #   / ``optics_xi``.
    # - ``affinity_propagation``: exemplar-based message passing on a
    #   precomputed manifold-*similarity* matrix; discovers the cluster count.
    # - ``meanshift``: mode-seeking. NOT manifold-aware — clusters the raw
    #   embedding coordinates in Euclidean space (sklearn has no precomputed
    #   path). Bandwidth is explicit or auto-estimated.
    cluster_algorithm: Literal[
        "kmeans",
        "hdbscan",
        "graph_cc",
        "leiden",
        "dbscan",
        "optics",
        "affinity_propagation",
        "meanshift",
    ] = "hdbscan"
    # HDBSCAN hyper-parameters (ignored when ``cluster_algorithm='kmeans'``).
    # The defaults follow scikit-learn; tune ``min_cluster_size`` first.
    hdbscan_min_cluster_size: int = 2
    hdbscan_min_samples: int | None = None
    hdbscan_cluster_selection_epsilon: float = 0.000001
    hdbscan_cluster_selection_method: Literal["eom", "leaf"] = "eom"
    hdbscan_allow_single_cluster: bool = False
    # Graph-CC hyper-parameters (ignored unless
    # ``cluster_algorithm='graph_cc'``).
    # - ``graph_cc_radius``: explicit radius. If ``None``, auto-pick
    #   using ``graph_cc_r_method``.
    # - ``graph_cc_r_method``: ``"knee"`` (canonical DBSCAN k-NN knee
    #   via Kneedle) or ``"mst_gap"`` (largest gap in MST edge weights).
    # - ``graph_cc_k_neighbors``: ``k`` for the knee heuristic.
    # - ``graph_cc_min_cluster_size``: components smaller than this go
    #   to the noise bucket (label = -1). Default ``2`` folds isolated
    #   singletons; set to ``1`` to keep every CC.
    graph_cc_radius: float | None = None
    graph_cc_r_method: Literal["knee", "mst_gap"] = "knee"
    graph_cc_k_neighbors: int = 4
    graph_cc_min_cluster_size: int = 2
    # Leiden hyper-parameters (ignored unless
    # ``cluster_algorithm='leiden'``). All knobs default to values that
    # are sensible for typical embedding corpora; tune
    # ``leiden_resolution`` first.
    # - ``leiden_graph_method``: ``"knn"`` (default) / ``"mutual_knn"``
    #   (stricter) / ``"radius"`` (reuse graph_cc-style radius graph).
    # - ``leiden_k_neighbors``: ``k`` for the (mutual-)k-NN graph. Also
    #   used as ``k`` for the knee-based auto-radius when
    #   ``leiden_graph_method='radius'`` and ``leiden_radius=None``.
    # - ``leiden_radius`` / ``leiden_r_method``: only consulted when
    #   ``leiden_graph_method='radius'``. Same semantics as graph_cc.
    # - ``leiden_quality``: ``"modularity"`` (default, classical
    #   Newman-Girvan via RBConfiguration) or ``"cpm"`` (no resolution
    #   limit).
    # - ``leiden_resolution``: higher ⇒ more, smaller communities. 1.0
    #   is classical modularity.
    # - ``leiden_min_cluster_size``: communities smaller than this go to
    #   the noise bucket (label = -1).
    # - ``leiden_n_iterations``: ``-1`` = run to convergence.
    leiden_graph_method: Literal["knn", "mutual_knn", "radius"] = "knn"
    leiden_k_neighbors: int = 15
    leiden_radius: float | None = None
    leiden_r_method: Literal["knee", "mst_gap"] = "knee"
    leiden_quality: Literal["modularity", "cpm"] = "modularity"
    leiden_resolution: float = 1.0
    leiden_min_cluster_size: int = 2
    leiden_n_iterations: int = -1
    # DBSCAN hyper-parameters (ignored unless ``cluster_algorithm='dbscan'``).
    # - ``dbscan_eps``: neighbourhood radius. ``None`` → auto via
    #   ``dbscan_r_method`` (``"knee"`` k-NN-distance knee / ``"mst_gap"``).
    # - ``dbscan_k_neighbors``: ``k`` for the knee heuristic.
    # - ``dbscan_min_samples``: core-point neighbourhood size.
    # - ``dbscan_min_cluster_size``: clusters smaller than this go to noise
    #   (label = -1); default ``1`` keeps every DBSCAN cluster.
    dbscan_eps: float | None = None
    dbscan_r_method: Literal["knee", "mst_gap"] = "knee"
    dbscan_k_neighbors: int = 4
    dbscan_min_samples: int = 5
    dbscan_min_cluster_size: int = 1
    # OPTICS hyper-parameters (ignored unless ``cluster_algorithm='optics'``).
    # - ``optics_min_samples``: core-distance neighbourhood size (main knob).
    # - ``optics_xi``: min reachability-valley steepness for a cluster
    #   boundary; lower ⇒ more, finer clusters.
    # - ``optics_min_cluster_size``: smallest extractable cluster. ``None`` →
    #   sklearn default (tied to ``min_samples``).
    optics_min_samples: int = 5
    optics_xi: float = 0.05
    optics_min_cluster_size: int | None = None
    # Affinity-propagation hyper-parameters (ignored unless
    # ``cluster_algorithm='affinity_propagation'``).
    # - ``affinity_damping`` (``[0.5, 1.0)``): raise toward 1.0 if it fails to
    #   converge.
    # - ``affinity_preference``: self-similarity controlling cluster count
    #   (higher ⇒ more). ``None`` → median input similarity (sklearn default).
    affinity_damping: float = 0.5
    affinity_preference: float | None = None
    affinity_max_iter: int = 200
    affinity_convergence_iter: int = 15
    # MeanShift hyper-parameters (ignored unless
    # ``cluster_algorithm='meanshift'``). Clusters raw embedding coordinates in
    # Euclidean space — not manifold-aware.
    # - ``meanshift_bandwidth``: kernel radius. ``None`` → estimated from the
    #   data using ``meanshift_quantile``.
    # - ``meanshift_quantile`` (``(0, 1]``): larger ⇒ wider kernel ⇒ fewer
    #   clusters.
    # - ``meanshift_cluster_all``: ``False`` lets orphan points become noise.
    meanshift_bandwidth: float | None = None
    meanshift_quantile: float = 0.3
    meanshift_bin_seeding: bool = False
    meanshift_cluster_all: bool = True
    # ── Dimensionality reduction (applied before clustering) ──────────────
    # Raw transformer embeddings are high-dimensional, where pairwise
    # distances concentrate and density-based clustering (HDBSCAN) collapses
    # to all-noise. Reducing first is the standard fix. ``reduce_method``
    # picks the reducer; ``reduce_dim`` is the target dimensionality (0 =
    # off). When active, clustering runs in the reduced Euclidean space
    # regardless of the representation manifold. Strongly recommended with
    # ``cluster_algorithm='hdbscan'`` (try reduce_dim 5-15).
    reduce_method: Literal[
        "none",
        "pca",
        "truncated_svd",
        "random_projection",
        "kernel_pca",
        "isomap",
        "lle",
        "spectral",
        "umap",
    ] = "none"
    reduce_dim: int = 0


# ── Corpus ────────────────────────────────────────────────────────────────
class CorpusConfig(_StrictModel):
    root: Path
    use_subfolder_labels: bool = True
    image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    pdf_extensions: tuple[str, ...] = (".pdf",)
    thumbnail_dir: Path | None = None
    thumbnail_size: int = 256


# ── Logger ────────────────────────────────────────────────────────────────
class LoggerConfig(_StrictModel):
    name: LoggerName = "none"
    project: str = "doc-cat"
    entity: str | None = None
    tags: list[str] = Field(default_factory=list)


# ── Training ──────────────────────────────────────────────────────────────
class TrainingConfig(_StrictModel):
    batch_size: int = 4
    epochs: int = 0  # 0 = no training (S1/S4 baselines)
    lr: float = 1e-4
    weight_decay: float = 0.0
    loss: LossName = "prototypical"
    margin: float = 0.2
    temperature: float = 0.07
    # ── Unsupervised losses (S1, ``loss`` ∈ UNSUPERVISED_LOSSES) ───────────
    # These train the projector with NO label access (not even folder
    # names); they are selected via ``loss`` and run regardless of the
    # ``supervision`` field below (which governs the semi-supervised
    # paths). Inference stays label-free either way.
    #
    # pseudo_label: re-cluster the projected space (Riemannian k-means) at
    # epoch 0 and every ``pseudo_recluster_every`` epochs to refresh the
    # pseudo-labels the prototypical M-step trains against. ``pseudo_k``
    # (shared with the supervision path) sets the EM cluster count; ``None`` →
    # scenario ``k_clusters`` → 8.
    pseudo_recluster_every: int = 25
    # Sinkhorn-Knopp balanced assignment (SwAV trick) for pseudo-labels —
    # prevents the EM loop from collapsing into one giant cluster.
    sinkhorn: bool = True
    sinkhorn_iters: int = 3
    sinkhorn_epsilon: float = 0.05
    # knn_contrastive: number of neighbors mined in the frozen fused space,
    # and weight of the SCAN entropy regularizer (pushes the mean soft
    # cluster assignment toward uniform — the anti-collapse term).
    knn_k: int = 5
    entropy_weight: float = 1.0
    # VICReg anti-collapse penalty, additive on top of ANY loss when the
    # weights are > 0. Applied in the tangent space at the manifold origin.
    # ``vicreg_gamma`` is the target per-dimension std for the hinge.
    vicreg_var_weight: float = 0.0
    vicreg_cov_weight: float = 0.0
    vicreg_gamma: float = 1.0
    # ── Projector ─────────────────────────────────────────────────────────
    # Force a trainable linear projector even when fusion.output_dim ==
    # manifold.dim. With the default ``False``, the projector is identity
    # (or whatever dim-adaptation linear is needed) and has no parameters;
    # set ``True`` to learn the ambient → manifold map under the configured
    # loss. Training only runs when ``epochs > 0`` and labeled samples
    # are present (S3 / S5).
    trainable_projector: bool = False
    # ── Identity projector ────────────────────────────────────────────────
    # Force a parameter-free identity projector: fused embeddings pass
    # straight through to ``manifold.expmap0`` with no linear map at all.
    # Requires ``fusion.output_dim == manifold.dim`` (a true identity
    # cannot bridge differing dims) and is mutually exclusive with
    # ``trainable_projector``. Use it for a clean, fully-reproducible
    # untrained baseline that is guaranteed not to insert a
    # (randomly-initialized) dim-adaptation linear. With the default
    # ``False`` the projector is chosen automatically: identity when dims
    # match and the projector isn't trainable, otherwise a linear.
    identity_projector: bool = True
    # ── Trainable fusion ──────────────────────────────────────────────────
    # Train the fusion module jointly with the projector (under the same
    # loss / supervision). Without this the fusion's weights stay at their
    # random initialization — for ``late_concat`` a random MLP, which
    # scrambles embeddings — because the supervised trainer otherwise only
    # touches the projector. Only meaningful when ``epochs > 0`` and the
    # fusion actually has parameters (a no-op for the parameter-free
    # ``none`` fusion). Currently honoured by S1's ``labels`` supervision
    # path; pseudo-label / cross-modal paths keep the fusion frozen.
    trainable_fusion: bool = False
    # ── Riemannian optimization ───────────────────────────────────────────
    # Set ``True`` to attach a learnable on-manifold anchor to the
    # projector (a ``geoopt.ManifoldParameter`` initialized at the
    # manifold origin) which ``geoopt.optim.RiemannianAdam`` updates
    # along the manifold's geodesics. Independent of (and composable
    # with) the prototypical loss's learnable on-manifold prototypes,
    # which are always active when geoopt is available. With the default
    # ``riemannian=False``, the projector behaves exactly as before — no
    # anchor, forward stays ``expmap0(linear(x))``.
    riemannian: bool = False
    # ── Supervision source ────────────────────────────────────────────────
    # What signal trains the projector:
    # - ``"labels"`` (default): ground-truth labels carried on the dataset
    #   records (e.g. folder names). Used by S3/S5 support sets and by
    #   S1's semi-supervised mode.
    # - ``"pseudo_labels"``: DeepCluster-style, fully label-free — cluster
    #   the current projection with manifold k-means, treat cluster ids as
    #   labels, train, and optionally re-cluster (``pseudo_rounds``).
    # - ``"cross_modal"``: CLIP-style, fully label-free — symmetric
    #   InfoNCE where a document's text-only view and image-only view are
    #   positives and all other documents are negatives.
    # S1 honours all three; S3/S5 always train on their support labels.
    supervision: Literal["labels", "pseudo_labels", "cross_modal"] = "labels"
    # Cluster → train rounds for ``supervision="pseudo_labels"``. Each
    # round re-clusters the current projection and runs ``epochs`` epochs
    # against the fresh pseudo-labels.
    pseudo_rounds: int = 1
    # Number of clusters for the pseudo-labeling k-means. ``None`` →
    # fall back to ``scenario.k_clusters``, then 8.
    pseudo_k: int | None = None

    @model_validator(mode="after")
    def _check_identity_projector(self) -> TrainingConfig:
        """``identity_projector`` and ``trainable_projector`` are mutually exclusive.

        ``identity_projector`` defaults to ``True`` (a parameter-free passthrough
        baseline). Because it is on by default, an *explicit* request to train the
        projector takes precedence over a merely *defaulted* identity passthrough:
        we silently turn identity off rather than rejecting the config. This keeps
        the True default from breaking every training config (S1/S3/S5,
        ``epochs > 0``). Only when **both** flags are set explicitly is it a real
        contradiction and a loud error — you can't train a parameter-free map.
        """
        if self.identity_projector and self.trainable_projector:
            identity_explicit = "identity_projector" in self.model_fields_set
            trainable_explicit = "trainable_projector" in self.model_fields_set
            if trainable_explicit and not identity_explicit:
                # Training was explicitly requested; identity is only on by default.
                # Frozen model → bypass the setattr guard to resolve in favor of training.
                object.__setattr__(self, "identity_projector", False)
                return self
            raise ValueError(
                "training.identity_projector and training.trainable_projector are "
                "mutually exclusive: an identity projector has no parameters to train. "
                "Drop one (identity_projector=True forces a parameter-free passthrough; "
                "trainable_projector=True learns the ambient → manifold map)."
            )
        return self


# ── Top-level ─────────────────────────────────────────────────────────────
def _dummy_encoder() -> EncoderConfig:
    """Default for an unspecified modality: the parameter-free hash-based
    ``dummy`` encoder.

    Lets a *monomodal* config name only the side it cares about and leave the
    other implicit — e.g. ``fusion=none`` + ``prefer_modality=image`` with just
    ``encoder_image`` set; the dummy text side is built (cheaply) and discarded.
    NOTE: a fusion that consumes *both* modalities (``concat_norm``, ``gated``,
    ``cross_attention``, …) will silently blend in this dummy noise — specify
    both encoders explicitly for multimodal runs.
    """
    return EncoderConfig(name="dummy", model_id="dummy")


class Config(_StrictModel):
    """Top-level resolved configuration. Frozen + extra-forbidden."""

    scenario: ScenarioConfig
    encoder_text: EncoderConfig = Field(default_factory=_dummy_encoder)
    encoder_image: EncoderConfig = Field(default_factory=_dummy_encoder)
    fusion: FusionConfig
    manifold: ManifoldConfig
    corpus: CorpusConfig
    logger: LoggerConfig
    training: TrainingConfig
    seed: int = 0
    device: DeviceSpec = "auto"
    output_dir: Path = Path("runs")
    # When set, encoders persist embeddings under this directory keyed by a
    # config fingerprint + per-input content hash, so re-runs that only change
    # fusion / manifold / clustering reuse the (expensive, deterministic)
    # encoder forward pass. ``None`` disables caching. See
    # ``clustering.encoders.caching``.
    cache_dir: Path | None = None

    def tags(self) -> list[str]:
        """Tags written to ClearML / W&B for run-comparison filters."""
        base = [
            f"scenario:{self.scenario.name}",
            f"fusion:{self.fusion.name}",
            f"manifold:{self.manifold.name}",
            f"encoder_text:{self.encoder_text.name}",
            f"encoder_image:{self.encoder_image.name}",
        ]
        return base + list(self.logger.tags)
