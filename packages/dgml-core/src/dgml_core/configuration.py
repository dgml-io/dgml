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

"""A workspace's configuration as one in-memory object.

Every workspace has exactly one :class:`Configuration` (``Workspace.config``). It is
built one of two ways:

- **In code**, by an application that keeps its settings elsewhere —
  :meth:`Configuration.build` takes typed sections and the storage binding, and the
  workspace is opened with ``Workspace.open(configuration=…)``. No ``config.toml``,
  no user config, no ``DGML_*`` environment variables, nothing written back.
- **From TOML**, for a workspace addressed by path or held in the store of workspaces —
  :meth:`Configuration.from_merged`, called by ``Workspace.config`` with the result of
  the usual merge (user config → workspace config → environment).

Either way the loaders (``load_grounded_config`` and friends) read the same thing: the
``sections`` mapping, in the exact shape ``load_merged_config`` has always returned, so
validation and the ``*_CONFIG_INVALID`` error codes stay where they are. The typed
section classes here are *builders*: each renders itself to the keys its loader reads
and nothing else, so a field name that does not exist cannot be written.

A model is a :class:`Model` wherever one is configured — a ``[models]`` tier or a
task's own — carrying its credentials with it, normally **by value** (``api_key=…``);
``api_key_env`` names an env var instead, as in ``config.toml``. The TOML stays flat:
``Model`` renders to the ``model`` / ``api_key`` / ``api_key_env`` / ``api_base`` key
group its position implies (``label_model`` / ``label_api_key`` …).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from .models_config import ConfigSection, Model, ModelsConfig
from .storage_service import StorageConfig
from .workspace_id import ID_SHAPE, is_workspace_id

__all__ = [
    "Classification",
    "Clustering",
    "Configuration",
    "Conversion",
    "Generation",
    "Grounded",
    "Identity",
    "Model",
    "Models",
    "Ocr",
    "Pdf",
    "ProviderSpec",
    "Storage",
    "Style",
    "TextExtraction",
]


# ---------------------------------------------------------------------------
# Identity and storage — the parts of a config.toml that are not merge sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Identity:
    """Who the workspace is: the ``[workspace]`` block, minus what only a stored
    config needs (``storage_service``, ``storage_fingerprint``, ``created_at``).

    ``organization`` is embedded in every docset namespace URI
    (``http://dgml.io/<organization>/<DocSetSlug>``); changing it later splits the
    corpus across two namespaces."""

    workspace_id: str | None = None
    name: str | None = None
    organization: str | None = None

    def require_complete(self) -> None:
        """Raise :class:`~dgml_core.errors.InvalidArgument` unless every field is set
        and the id is well-formed — the rule for an identity supplied in memory, as
        ``create_workspace`` applies it. One derived from TOML may be partial (a
        workspace created before ids existed, say).

        The shape matters here as much as on the CLI: the id becomes a directory
        name, an S3 key prefix and a Mongo collection prefix."""
        from .errors import InvalidArgument

        missing = [f.name for f in fields(self) if not getattr(self, f.name)]
        if missing:
            raise InvalidArgument(f"Identity is missing {', '.join(missing)}")
        assert self.workspace_id is not None
        if not is_workspace_id(self.workspace_id):
            raise InvalidArgument(
                f"workspace_id {self.workspace_id!r} is not a valid workspace id: {ID_SHAPE}"
            )


@dataclass(frozen=True)
class ProviderSpec:
    """A provider class plus its options — one ``[storage.<svc>.blobs]`` /
    ``[conversion.<family>]`` table. ``provider`` is the dotted
    ``module.path:ClassName`` the resolver imports; ``options`` is validated by that
    class's ``parse_config`` (unknown keys are rejected there), so a provider's own
    settings — including credential-named ones — pass through untouched."""

    provider: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def table(self) -> dict[str, Any]:
        return {"provider": self.provider, **dict(self.options)}

    def _frozen(self) -> tuple[Any, ...]:
        return (self.provider, tuple(sorted(self.options.items())))


@dataclass(frozen=True)
class Storage:
    """The storage binding: where blobs and documents go. One provider may serve
    both roles (:meth:`combined`), in which case the workspace builds it once."""

    blobs: ProviderSpec
    docs: ProviderSpec

    @classmethod
    def combined(cls, spec: ProviderSpec) -> Storage:
        return cls(blobs=spec, docs=spec)

    def store_configs(
        self, *, root: Any, workspace_id: str | None
    ) -> tuple[StorageConfig, StorageConfig]:
        """The ``(blob_cfg, doc_cfg)`` pair ``Workspace.store_configs`` hands to the
        store factories. ``root`` only matters to the bundled local store."""
        return (
            StorageConfig(self.blobs.provider, root, self.blobs.options, workspace_id),
            StorageConfig(self.docs.provider, root, self.docs.options, workspace_id),
        )


# ---------------------------------------------------------------------------
# Typed sections — each renders to the keys its loader reads
# ---------------------------------------------------------------------------


def _credential_prefix(name: str) -> str:
    """The key prefix a :class:`Model` field's credentials take: ``model`` → ``""``,
    ``label_model`` → ``label_``, a tier such as ``light`` → ``light_``."""
    return name[: -len("model")] if name.endswith("model") else f"{name}_"


def _table(obj: Any, *, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
    """The non-``None`` fields of a flat section dataclass, as a TOML-shaped table.
    ``None`` means "not set": the key is omitted so the loader applies its default.
    A :class:`Model` field flattens into its key group."""
    out: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if f.name in exclude or value is None:
            continue
        if isinstance(value, Model):
            out.update(value.table(f.name, _credential_prefix(f.name)))
        else:
            out[f.name] = value
    return out


@dataclass(frozen=True)
class Models:
    """``[models]``: the four tiers, or a provider ``family`` that expands into them
    (an explicit tier wins over its family default). A tier is a :class:`Model`, so it
    may carry its own credentials; ``<provider>_api_key`` is the key for every model
    of that provider — a tier's or a task section's — that sets none itself."""

    family: str | None = None
    light: Model | None = None
    standard: Model | None = None
    advanced: Model | None = None
    expert: Model | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    openai_api_key: str | None = None

    def table(self) -> dict[str, Any]:
        return _table(self)


@dataclass(frozen=True)
class Grounded:
    """``[grounded]``: schema generation (``schema_model``, expert tier) and value
    extraction (``values_model``, advanced tier). Reads
    ``dgml_core.grounded.load_grounded_config``."""

    schema_model: Model | None = None
    values_model: Model | None = None
    max_tool_iters: int | None = None
    values_reasoning_effort: str | None = None

    def table(self) -> dict[str, Any]:
        return _table(self)


@dataclass(frozen=True)
class Classification:
    """``[classification]`` (light tier). Reads
    ``dgml_core.classification.load_classification_config``."""

    model: Model | None = None
    max_pages: int | None = None
    naming_attempts: int | None = None

    def table(self) -> dict[str, Any]:
        return _table(self)


@dataclass(frozen=True)
class Ocr:
    """``[ocr]``: ``provider`` is a built-in name (``aws`` / ``azure`` / ``macos``) or a
    dotted path; ``options`` are that provider's own fields (``endpoint``, ``api_key``,
    ``region``, …), validated by its ``parse_config``. Reads
    ``dgml_core.ocr.load_ocr_config``."""

    provider: str
    max_concurrency: int | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def table(self) -> dict[str, Any]:
        return {**_table(self, exclude=frozenset({"options"})), **dict(self.options)}


@dataclass(frozen=True)
class Pdf:
    """``[pdf]``: the page engine, ``ghostscript`` (default) or ``pypdfium2``. Reads
    ``dgml_core.pages.load_pdf_config``."""

    provider: str

    def table(self) -> dict[str, Any]:
        return _table(self)


@dataclass(frozen=True)
class Style:
    """``[style]``: image-based ``dg:style`` for OCR'd pages (light tier). Building one
    enables the feature — the TOML section's ``enabled`` switch defaults to ``True``
    here because a caller who constructs it means to use it. Reads
    ``dgml_core.style_config.load_style_config``."""

    model: Model | None = None
    max_tokens: int | None = None
    enabled: bool = True

    def table(self) -> dict[str, Any]:
        return _table(self)


@dataclass(frozen=True)
class TextExtraction:
    """``[text_extraction]``: the LLM merge for hybrid text mode (standard tier).
    Building one enables it, as for :class:`Style`. Reads
    ``dgml_core.text_extraction_config.load_text_extraction_config``."""

    model: Model | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    enabled: bool = True

    def table(self) -> dict[str, Any]:
        return _table(self)


@dataclass(frozen=True)
class Conversion:
    """``[conversion]``: one :class:`ProviderSpec` per format family (``docx``,
    ``xlsx``). Reads ``dgml_core.conversion.load_conversion_config``."""

    families: Mapping[str, ProviderSpec] = field(default_factory=dict)

    def table(self) -> dict[str, Any]:
        return {family: spec.table() for family, spec in self.families.items()}


@dataclass(frozen=True)
class Generation:
    """``[generation]``: transcription (``model``) and labeling (``label_model``), both
    standard tier by default, each with its own credentials. Reads
    ``dgml_core.generation.config.load_generation_config``."""

    model: Model | None = None
    label_model: Model | None = None
    thinking: str | None = None

    def table(self) -> dict[str, Any]:
        return _table(self)


@dataclass(frozen=True)
class Clustering:
    """``[clustering]``: free-form overrides deep-merged over the bundled defaults;
    field validation happens in ``dgml_core.run_clustering``. Reads
    ``dgml_core.clustering.load_clustering_overrides``."""

    overrides: Mapping[str, Any] = field(default_factory=dict)

    def table(self) -> dict[str, Any]:
        return dict(self.overrides)


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------

Sections = Mapping[ConfigSection, Mapping[str, Any]]


@dataclass(frozen=True)
class Configuration:
    """One workspace's configuration: identity, storage binding, and the merge
    sections every loader reads. See the module docstring for the two ways to get one.

    Equality covers everything; the hash covers identity and storage only (sections
    hold dicts), which is what a per-tenant cache of ``Workspace`` objects keys on.
    """

    identity: Identity
    storage: Storage
    sections: Sections = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.identity, self.storage.blobs._frozen(), self.storage.docs._frozen()))

    @classmethod
    def build(
        cls,
        *,
        identity: Identity,
        storage: Storage,
        models: Models | None = None,
        grounded: Grounded | None = None,
        classification: Classification | None = None,
        ocr: Ocr | None = None,
        pdf: Pdf | None = None,
        style: Style | None = None,
        text_extraction: TextExtraction | None = None,
        conversion: Conversion | None = None,
        generation: Generation | None = None,
        clustering: Clustering | None = None,
    ) -> Configuration:
        """Build from typed sections. A section left ``None`` is absent — the loaders
        already treat ``{}`` and absent alike. ``models.family`` is expanded into
        tiers here, exactly as it is for a TOML layer."""
        # Imported here: config.py imports storage.py, which names this module.
        from .config import _expand_layer

        identity.require_complete()
        typed: dict[ConfigSection, Any] = {
            ConfigSection.MODELS: models,
            ConfigSection.GROUNDED: grounded,
            ConfigSection.CLASSIFICATION: classification,
            ConfigSection.OCR: ocr,
            ConfigSection.PDF: pdf,
            ConfigSection.STYLE: style,
            ConfigSection.TEXT_EXTRACTION: text_extraction,
            ConfigSection.CONVERSION: conversion,
            ConfigSection.GENERATION: generation,
            ConfigSection.CLUSTERING: clustering,
        }
        layer = {s.value: t.table() for s, t in typed.items() if t is not None}
        return cls.from_merged(identity=identity, storage=storage, sections=_expand_layer(layer))

    @classmethod
    def from_merged(
        cls, *, identity: Identity, storage: Storage, sections: Mapping[Any, Any]
    ) -> Configuration:
        """Wrap an already-merged section mapping (string or :class:`ConfigSection`
        keys). This is how ``Workspace.config`` is derived on the TOML path."""
        return cls(
            identity=identity,
            storage=storage,
            sections={ConfigSection(str(k)): dict(v) for k, v in sections.items()},
        )

    @property
    def models(self) -> ModelsConfig:
        """The validated ``[models]`` tiers — a convenience over the loaders."""
        from .models_config import load_models_config

        return load_models_config(dict(self.sections))
