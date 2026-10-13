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

"""The ``[models]`` tier block — the simplified model entry point.

Four tiers, cheapest to strongest, each mapped to a set of tasks:

* ``light`` ...... classification, style
* ``standard`` ... transcription, text extraction
* ``advanced`` ... labeling, value extraction
* ``expert`` ..... schema generation

Each tier is a :class:`Model`: a model id plus optional credentials
(``<tier>_api_key`` / ``<tier>_api_key_env`` / ``<tier>_api_base``). A tier that
sets none takes its provider's ``<provider>_api_key`` / ``_env`` (``anthropic_``,
``google_``, ``openai_``), matched on the model id's prefix — so one key per
provider covers a mixed family such as ``anthropic_google``. A task section's own
credentials (``generation.api_key_env``, ``grounded.schema_api_key``) win over
both, and a model with none of those falls back to litellm's per-provider env
var. See :func:`resolve_tiered_model`.

``family`` picks a whole provider family's defaults: one of the
:data:`~dgml_core.default_config.PROVIDER_MODELS` keys. It is shorthand for that
family's four tiers *within its config layer* (see :func:`expand_family`): tiers
the same layer sets win, and the expanded tiers override any a lower layer set.
A family-based config tracks dgml's shipped defaults across upgrades; explicit
tiers are the pinning mechanism.

A tier that is unset falls back to the nearest set tier (nearest *lower* first,
then higher), emitting a warning — so a minimal config that sets only, say,
``standard`` still resolves every task.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .default_config import PROVIDER_MODELS, PROVIDERS
from .errors import DgmlError, ModelsConfigInvalid

logger = logging.getLogger(__name__)


class Tier(StrEnum):
    """A ``[models]`` capability tier. Definition order is cheapest → strongest,
    which :meth:`ModelsConfig._nearest_set` relies on for fallback ordering.

    A ``StrEnum``, so a member is usable directly as a config key, in f-strings,
    and anywhere a plain tier string was expected."""

    LIGHT = "light"
    STANDARD = "standard"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ConfigSection(StrEnum):
    """A top-level config section (a ``[section]`` table in ``config.toml``).

    A ``StrEnum`` whose value is the literal TOML section name, so it doubles as
    the lookup key into the merged config mapping and formats to the bare name in
    error messages. The tier-backed sections (``classification``, ``style``,
    ``text_extraction``, ``generation``, ``grounded``) are the ones passed to
    :func:`resolve_tiered_model`; the rest (``models``, ``ocr``, ``pdf``,
    ``conversion``, ``clustering``) configure non-LLM or tier-source settings."""

    MODELS = "models"
    GENERATION = "generation"
    GROUNDED = "grounded"
    CLASSIFICATION = "classification"
    OCR = "ocr"
    PDF = "pdf"
    STYLE = "style"
    TEXT_EXTRACTION = "text_extraction"
    CONVERSION = "conversion"
    CLUSTERING = "clustering"
    STORAGE = "storage"


# Cheapest → strongest. Fallback searches lower (cheaper) neighbours first.
TIERS: tuple[Tier, ...] = tuple(Tier)

# Tier fallbacks already reported this process, so a per-file loop (e.g. bulk
# extract) doesn't repeat the same warning. Keyed by (requested, used).
_WARNED_TIER_FALLBACKS: set[tuple[Tier, Tier]] = set()

# Same idea for "configured but not enabled" advisories (see `section_enabled`):
# every loader re-reads the merged config, so without this the same line would be
# written several times per command.
_WARNED_DISABLED: set[ConfigSection] = set()


@dataclass(frozen=True)
class Model:
    """A model id with its credentials — the typed form of one ``model`` / ``api_key``
    / ``api_key_env`` / ``api_base`` key group, wherever it appears: a ``[models]``
    tier, a task section's model, or a task's resolved model."""

    model: str
    api_key: str | None = None
    api_key_env: str | None = None
    api_base: str | None = None

    def table(self, model_key: str, prefix: str) -> dict[str, Any]:
        """The flat keys: ``model_key`` for the id, ``{prefix}api_key`` and so on for
        the rest (``light`` / ``light_``, ``label_model`` / ``label_``)."""
        out = {model_key: self.model}
        for name in ("api_key", "api_key_env", "api_base"):
            value = getattr(self, name)
            if value is not None:
                out[f"{prefix}{name}"] = value
        return out


Credentials = tuple[str | None, str | None]
"""An ``(api_key, api_key_env)`` pair; at most one is set."""

_NO_CREDENTIALS: Credentials = (None, None)


def provider_of(model: str) -> str | None:
    """The :data:`~dgml_core.default_config.PROVIDERS` key serving ``model``'s
    litellm prefix, or ``None`` for an unprefixed or unlisted one."""
    prefix, sep, _ = model.partition("/")
    if not sep:
        return None
    return next((p for p, prefixes in PROVIDERS.items() if prefix in prefixes), None)


@dataclass(frozen=True)
class ModelsConfig:
    """Parsed ``[models]`` block: one :class:`Model` per tier, all optional, each with
    its credentials already settled (its own ``<tier>_*`` keys, else its provider's).
    ``providers`` keeps the ``<provider>_api_key`` / ``_env`` pairs for
    :meth:`credentials_for`."""

    light: Model | None = None
    standard: Model | None = None
    advanced: Model | None = None
    expert: Model | None = None
    providers: Mapping[str, Credentials] = field(default_factory=dict)

    def credentials_for(self, model: str) -> Credentials:
        """The ``<provider>_api_key`` / ``_env`` pair for ``model``'s provider."""
        provider = provider_of(model)
        return self.providers.get(provider, _NO_CREDENTIALS) if provider else _NO_CREDENTIALS

    def resolve(self, tier: Tier) -> Model | None:
        """Resolve ``tier`` to its :class:`Model`.

        If ``tier`` has no model, fall back to the nearest set tier — lower
        (cheaper) neighbours first, then higher — and log a WARNING (once per
        process per fallback). The fallback tier's credentials come with its
        model. Returns ``None`` when no tier is set at all (the caller then
        surfaces the appropriate config error)."""
        if tier not in TIERS:
            raise ValueError(f"unknown model tier {tier!r}")
        actual = self._nearest_set(tier)
        if actual is None:
            return None
        model: Model = getattr(self, actual)
        if actual != tier and (tier, actual) not in _WARNED_TIER_FALLBACKS:
            _WARNED_TIER_FALLBACKS.add((tier, actual))
            logger.warning(
                f"[dgml] model tier '{tier}' is not set; falling back to '{actual}' "
                f"('{model.model}'). Set [models].{tier} to silence this."
            )
        return model

    def _nearest_set(self, tier: Tier) -> Tier | None:
        idx = TIERS.index(tier)
        # Lower (cheaper) neighbours nearest-first, then higher neighbours.
        order = list(range(idx - 1, -1, -1)) + list(range(idx + 1, len(TIERS)))
        if getattr(self, tier) is not None:
            return tier
        for i in order:
            if getattr(self, TIERS[i]) is not None:
                return TIERS[i]
        return None


def _opt_str(value: Any, label: str, invalid: type[DgmlError]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise invalid(f"'{label}' must be a non-empty string if set")
    return value


def _credentials(
    table: dict[str, Any], prefix: str, section: str, invalid: type[DgmlError]
) -> Credentials:
    """``{prefix}api_key`` / ``{prefix}api_key_env`` from ``table``; setting both is an
    error."""
    key_label, env_label = f"{section}.{prefix}api_key", f"{section}.{prefix}api_key_env"
    api_key = _opt_str(table.get(f"{prefix}api_key"), key_label, invalid)
    api_key_env = _opt_str(table.get(f"{prefix}api_key_env"), env_label, invalid)
    if api_key is not None and api_key_env is not None:
        raise invalid(f"set at most one of '{key_label}' / '{env_label}', not both")
    return api_key, api_key_env


def expand_family(models: dict[str, Any]) -> dict[str, Any]:
    """Return one config layer's ``[models]`` table with its ``family`` expanded
    into the tiers the table leaves unset. Called per layer *before* the merge, so
    a family overrides lower-layer tiers. A malformed or unknown family is left
    as-is for :func:`load_models_config` to reject."""
    family = models.get("family")
    defaults = PROVIDER_MODELS.get(family) if isinstance(family, str) else None
    return models if defaults is None else {**defaults, **models}


def load_models_config(merged: dict[ConfigSection, Any]) -> ModelsConfig:
    """Build a :class:`ModelsConfig` from the merged config mapping's
    ``[models]`` section (an empty section yields an all-``None`` config).

    ``family`` is validated here but not expanded — :func:`expand_family` has
    already done that per layer during the merge. Provider keys are matched to
    tiers here, on the merged table, so a key and the tier it serves may come from
    different layers."""
    section = merged.get(ConfigSection.MODELS)
    if section is None:
        return ModelsConfig()
    if not isinstance(section, dict):
        raise ModelsConfigInvalid("'models' must be a table")
    invalid = ModelsConfigInvalid
    family = _opt_str(section.get("family"), "models.family", invalid)
    if family is not None and family not in PROVIDER_MODELS:
        raise invalid(
            f"'models.family' must be one of {', '.join(sorted(PROVIDER_MODELS))}; got {family!r}"
        )
    providers = {
        p: creds
        for p in PROVIDERS
        if (creds := _credentials(section, f"{p}_", "models", invalid)) != _NO_CREDENTIALS
    }
    tiers: dict[str, Model] = {}
    for t in TIERS:
        model = _opt_str(section.get(t.value), f"models.{t}", invalid)
        api_key, api_key_env = _credentials(section, f"{t}_", "models", invalid)
        api_base = _opt_str(section.get(f"{t}_api_base"), f"models.{t}_api_base", invalid)
        if model is None:
            if (api_key, api_key_env, api_base) != (None, None, None):
                raise invalid(f"'models.{t}_api_key' and friends need 'models.{t}' to be set")
            continue
        if api_key is None and api_key_env is None:
            provider = provider_of(model)
            api_key, api_key_env = providers.get(provider or "", _NO_CREDENTIALS)
        tiers[t.value] = Model(model, api_key, api_key_env, api_base)
    return ModelsConfig(**tiers, providers=providers)


def section_enabled(
    section: dict[str, Any],
    *,
    section_name: ConfigSection,
    invalid: type[DgmlError],
) -> bool:
    """Whether an opt-in feature section carries ``enabled = true``.

    The ``style`` and ``text_extraction`` sections are switches: they configure a
    feature that is off unless explicitly turned on. ``enabled`` is that switch —
    the section's mere *presence* means nothing, so the shipped ``config.toml``
    can name both features (with comments explaining them) without enabling
    either.

    Warns — once per section per process — when a section is disabled but carries
    real configuration beyond ``enabled`` itself. A section holding only
    ``enabled = false`` is the shipped default and says nothing about intent; one
    that also names a model or credentials was written by someone who expects it
    to run. Configs predating the ``enabled`` switch look exactly like that, and
    turning them off silently is the failure mode this switch exists to prevent.

    Raises ``invalid`` when ``enabled`` is present but not a boolean.
    """
    enabled: object = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise invalid(f"'{section_name}.enabled' must be true or false")
    if not enabled and set(section) - {"enabled"} and section_name not in _WARNED_DISABLED:
        _WARNED_DISABLED.add(section_name)
        logger.warning(
            f"[dgml] the [{section_name}] config section is configured but not enabled; "
            f"it will be ignored. Set {section_name}.enabled = true to use it."
        )
    return enabled


def resolve_tiered_model(
    merged: dict[ConfigSection, Any],
    *,
    section_name: ConfigSection,
    tier: Tier,
    invalid: type[DgmlError],
    missing: type[DgmlError],
    prefix: str = "",
) -> Model:
    """Resolve one task's :class:`Model` from the ``[{section_name}]`` section of
    *merged*, or — when the section names no model — from its ``[models]`` *tier*.

    ``prefix`` names the section's key group: ``{prefix}model``, ``{prefix}api_key``,
    ``{prefix}api_key_env``, ``{prefix}api_base`` (``schema_`` / ``values_`` for
    grounded, ``label_`` for generation labeling, empty for the rest).

    Credentials, most specific first: the section's own; then the model's — a
    tier's ``<tier>_api_key`` or its provider's ``<provider>_api_key`` for a
    tier-sourced model, the provider's key alone for one the section names; else
    none, and litellm reads its per-provider env var. ``{prefix}api_base`` likewise
    overrides a tier's ``<tier>_api_base``.

    Raises ``invalid`` for a malformed value or a literal+env-name clash, and
    ``missing`` when neither the section nor the tier resolves a model. Callers
    that treat a section's mere presence as a feature switch (``style`` /
    ``text_extraction``) check that themselves before calling this.
    """
    section = merged.get(section_name)
    sec: dict[str, Any] = section if isinstance(section, dict) else {}
    model = _opt_str(sec.get(f"{prefix}model"), f"{section_name}.{prefix}model", invalid)
    api_key, api_key_env = _credentials(sec, prefix, section_name, invalid)
    api_base = _opt_str(sec.get(f"{prefix}api_base"), f"{section_name}.{prefix}api_base", invalid)
    own_credentials = api_key is not None or api_key_env is not None

    if model is not None:
        if not own_credentials:
            api_key, api_key_env = load_models_config(merged).credentials_for(model)
        return Model(model, api_key, api_key_env, api_base)

    tiered = load_models_config(merged).resolve(tier)
    if tiered is None:
        raise missing(
            f"no {prefix}model for {section_name}: set [models].family, [models].{tier}, "
            f"or '{section_name}.{prefix}model' in the config"
        )
    if not own_credentials:
        api_key, api_key_env = tiered.api_key, tiered.api_key_env
    return Model(
        tiered.model, api_key, api_key_env, api_base if api_base is not None else tiered.api_base
    )
