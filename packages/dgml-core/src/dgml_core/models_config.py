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

Each tier names only a **model**. Credentials are configured per task, on the
task's own section (e.g. ``generation.api_key_env``, ``grounded.schema_api_key``);
a model sourced from a tier uses its task section's credentials, or falls back to
litellm's per-provider env-var conventions when the section sets none.

A tier that is unset falls back to the nearest set tier (nearest *lower* first,
then higher), emitting a warning — so a minimal config that sets only, say,
``standard`` still resolves every task.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .errors import DgmlError, ModelsConfigInvalid


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
    :func:`resolve_tiered_model`; the rest (``models``, ``ocr``, ``conversion``,
    ``clustering``) configure non-LLM or tier-source settings."""

    MODELS = "models"
    GENERATION = "generation"
    GROUNDED = "grounded"
    CLASSIFICATION = "classification"
    OCR = "ocr"
    STYLE = "style"
    TEXT_EXTRACTION = "text_extraction"
    CONVERSION = "conversion"
    CLUSTERING = "clustering"
    STORAGE = "storage"


# Cheapest → strongest. Fallback searches lower (cheaper) neighbours first.
TIERS: tuple[Tier, ...] = tuple(Tier)

# Tier fallbacks already reported this process, so a per-file loop (e.g. bulk
# extract) doesn't flood stderr with the same line. Keyed by (requested, used).
_WARNED_TIER_FALLBACKS: set[tuple[Tier, Tier]] = set()

# Same idea for "configured but not enabled" advisories (see `section_enabled`):
# every loader re-reads the merged config, so without this the same line would be
# written several times per command.
_WARNED_DISABLED: set[ConfigSection] = set()


@dataclass(frozen=True)
class ModelsConfig:
    """Parsed ``[models]`` block: one model string per tier, all optional."""

    light: str | None = None
    standard: str | None = None
    advanced: str | None = None
    expert: str | None = None

    def resolve(self, tier: Tier) -> str | None:
        """Resolve ``tier`` to its model string.

        If ``tier`` has no model, fall back to the nearest set tier — lower
        (cheaper) neighbours first, then higher — and write a warning to stderr
        (always, independent of ``--verbose``). Returns ``None`` when no tier is
        set at all (the caller then surfaces the appropriate config error)."""
        if tier not in TIERS:
            raise ValueError(f"unknown model tier {tier!r}")
        actual = self._nearest_set(tier)
        if actual is None:
            return None
        if actual != tier and (tier, actual) not in _WARNED_TIER_FALLBACKS:
            _WARNED_TIER_FALLBACKS.add((tier, actual))
            sys.stderr.write(
                f"[dgml] model tier '{tier}' is not set; falling back to '{actual}' "
                f"('{getattr(self, actual)}'). Set [models].{tier} to silence this.\n"
            )
        model: str | None = getattr(self, actual)
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


def _validate_optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ModelsConfigInvalid(f"'models.{field}' must be a non-empty string if set")
    return value


def load_models_config(merged: dict[ConfigSection, Any]) -> ModelsConfig:
    """Build a :class:`ModelsConfig` from the merged config mapping's
    ``[models]`` section (an empty section yields an all-``None`` config)."""
    section = merged.get(ConfigSection.MODELS)
    if section is None:
        return ModelsConfig()
    if not isinstance(section, dict):
        raise ModelsConfigInvalid("'models' must be a table")
    return ModelsConfig(
        **{t.value: _validate_optional_str(section.get(t.value), t.value) for t in TIERS}
    )


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
        sys.stderr.write(
            f"[dgml] the [{section_name}] config section is configured but not enabled; "
            f"it will be ignored. Set {section_name}.enabled = true to use it.\n"
        )
    return enabled


@dataclass(frozen=True)
class ResolvedModel:
    """A task's resolved model id plus its (name-only) credentials."""

    model: str
    api_key: str | None
    api_key_env: str | None
    api_base: str | None


def resolve_tiered_model(
    merged: dict[ConfigSection, Any],
    *,
    section_name: ConfigSection,
    tier: Tier,
    invalid: type[DgmlError],
    missing: type[DgmlError],
    model_field: str = "model",
    key_field: str = "api_key",
    env_field: str = "api_key_env",
    base_field: str = "api_base",
) -> ResolvedModel:
    """Resolve one task's model + credentials from the ``[{section_name}]``
    section of *merged*, or — when the section names no model — from its
    ``[models]`` *tier*.

    The section's ``model_field`` overrides the tier. Credentials come solely
    from the section's ``key_field`` / ``env_field`` / ``base_field`` (they vary
    per task: e.g. ``label_api_key`` for generation labeling, ``schema_api_key``
    for grounded schema-gen); tiers carry no credentials, so a tier-sourced model
    with no section credentials falls back to litellm's per-provider env vars.

    Raises ``invalid`` for a malformed value or a literal+env-name clash, and
    ``missing`` when neither the field nor the tier resolves a model. Callers
    that treat a section's mere presence as a feature switch (``style`` /
    ``text_extraction``) check that themselves before calling this.
    """
    section = merged.get(section_name)
    sec: dict[str, Any] = section if isinstance(section, dict) else {}

    def _opt_str(value: Any, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise invalid(f"'{section_name}.{field}' must be a non-empty string if set")
        return value

    model = _opt_str(sec.get(model_field), model_field)
    api_key = _opt_str(sec.get(key_field), key_field)
    api_key_env = _opt_str(sec.get(env_field), env_field)
    api_base = _opt_str(sec.get(base_field), base_field)
    if api_key is not None and api_key_env is not None:
        raise invalid(
            f"set at most one of '{section_name}.{key_field}' / "
            f"'{section_name}.{env_field}', not both"
        )

    if model is None:
        model = load_models_config(merged).resolve(tier)
    if not isinstance(model, str) or not model.strip():
        raise missing(
            f"no {model_field} for {section_name}: set [models].{tier} or "
            f"'{section_name}.{model_field}' in the config"
        )

    return ResolvedModel(model=model, api_key=api_key, api_key_env=api_key_env, api_base=api_base)
