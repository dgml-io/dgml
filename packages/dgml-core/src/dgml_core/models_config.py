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
(``<tier>_api_key`` / ``<tier>_api_key_env`` / ``<tier>_api_base``). **A model and
its credentials are one group, written together in one config layer**: a tier's are
the ``<tier>_*`` keys beside it, a task section's model (``generation.label_model``,
``grounded.schema_model``) takes only that section's ``{prefix}api_key`` and friends,
and a credential in a layer that does not set its model is refused by the task that
resolves that model — never by the merge, which every command shares. A model with
no credentials leaves litellm to read its per-provider env var.

``<provider>_api_key`` / ``_env`` (``anthropic_``, ``google_``, ``openai_``) merges
like any other setting, its pair as one group. :func:`apply_provider_keys` then writes
it onto every model of that provider — tier, task section or per-call override — that
carries neither its own key nor an ``api_base``: a model at a custom endpoint never
receives a key meant for the provider's own service, and is warned about when it
resolves, since litellm sends its own env var there instead.

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

# Provider keys withheld from a keyless model at a custom api_base (see
# `_check_provider_key_for`), keyed by the message.
_WARNED_PROVIDER_KEY_NOT_APPLIED: set[str] = set()


@dataclass(frozen=True)
class Model:
    """A model id with its credentials — the typed form of one ``model`` / ``api_key``
    / ``api_key_env`` / ``api_base`` key group, wherever it appears: a ``[models]``
    tier, a task section's model, or a task's resolved model. The key itself is left
    out of the repr; the variable name and endpoint are not secrets."""

    model: str
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    api_base: str | None = None

    def table(self, model_key: str, prefix: str) -> dict[str, Any]:
        """The flat keys: ``model_key`` for the id, ``{prefix}api_key`` and so on for
        the rest (``light`` / ``light_``, ``label_model`` / ``label_``)."""
        out = {model_key: self.model}
        for name in GROUP_SUFFIXES:
            value = getattr(self, name)
            if value is not None:
                out[f"{prefix}{name}"] = value
        return out


Credentials = tuple[str | None, str | None]
"""An ``(api_key, api_key_env)`` pair; at most one is set."""


def provider_of(model: str) -> str | None:
    """The :data:`~dgml_core.default_config.PROVIDERS` key serving ``model``'s
    litellm prefix, or ``None`` for an unprefixed or unlisted one."""
    prefix, sep, _ = model.partition("/")
    if not sep:
        return None
    return next((p for p, prefixes in PROVIDERS.items() if prefix in prefixes), None)


@dataclass(frozen=True)
class ModelsConfig:
    """Parsed ``[models]`` block: one :class:`Model` per tier, all optional, each
    carrying exactly the credentials written beside it."""

    light: Model | None = None
    standard: Model | None = None
    advanced: Model | None = None
    expert: Model | None = None

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
    table: Mapping[str, Any], prefix: str, section: str, invalid: type[DgmlError]
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


#: The credential keys that travel with a model key as one group.
GROUP_SUFFIXES = ("api_key", "api_key_env", "api_base")

#: Sections whose ``model`` / ``<prefix>_model`` keys head a credential group. Named
#: rather than inferred: ``ocr`` also carries an ``api_key``, but no model it belongs to.
LLM_SECTIONS = frozenset(
    {
        ConfigSection.GENERATION,
        ConfigSection.GROUNDED,
        ConfigSection.CLASSIFICATION,
        ConfigSection.STYLE,
        ConfigSection.TEXT_EXTRACTION,
    }
)


def _model_keys(table: Mapping[str, Any]) -> list[str]:
    """The keys of ``table`` that head a model group: ``model`` and every ``*_model``."""
    return [k for k in table if k == "model" or k.endswith("_model")]


def _stray_credential(table: Mapping[str, Any], prefix: str) -> str | None:
    """The first of ``{prefix}api_key`` / ``_env`` / ``api_base`` set in ``table``."""
    return next((s for s in GROUP_SUFFIXES if table.get(f"{prefix}{s}") is not None), None)


def _complete_group(table: dict[str, Any], prefix: str) -> None:
    """Write ``None`` for the credentials of one model group the layer leaves unset.

    A ``None`` survives the layer merge where an absent key would not, so a higher
    layer that sets a model replaces its whole group instead of pairing the new model
    with a lower layer's key. The merge drops those ``None``s afterwards."""
    for suffix in GROUP_SUFFIXES:
        table.setdefault(f"{prefix}{suffix}", None)


def _disown_stray_group(table: dict[str, Any], prefix: str, model_key: str) -> None:
    """Set the group's model to ``None`` when the layer holds a credential for it but
    no model: the layer then owns the group whole, so no lower layer's model pairs
    with the key, and the task that resolves this model refuses it — only that task."""
    if _stray_credential(table, prefix) is not None:
        table[model_key] = None


def _stray_error(
    invalid: type[DgmlError], credential: str, model_key: str, remedy: str
) -> DgmlError:
    """*credential* and *model_key* are full dotted keys (``models.light_api_key`` /
    ``models.light``, ``generation.label_api_key`` / ``generation.label_model``)."""
    return invalid(
        f"'{credential}' is set, but not in the same config layer as '{model_key}': a "
        f"model's credentials are written beside it. Set {remedy} in that layer, or use "
        f"'models.<provider>_api_key' / '_env', which reaches every keyless model of its "
        f"provider."
    )


def _unknown_family(family: str) -> ModelsConfigInvalid:
    return ModelsConfigInvalid(
        f"'models.family' must be one of {', '.join(sorted(PROVIDER_MODELS))}; got {family!r}"
    )


def provider_keys(models: Mapping[str, Any], *, strict: bool = True) -> dict[str, Credentials]:
    """A ``[models]`` table's ``<provider>_api_key`` / ``_env`` pairs; a provider with
    neither is left out. A malformed pair — empty, or both forms set — raises
    ``ModelsConfigInvalid`` when *strict*, else is left out for the loader that would
    have used it to report."""
    out: dict[str, Credentials] = {}
    for p in PROVIDERS:
        try:
            pair = _credentials(models, f"{p}_", "models", ModelsConfigInvalid)
        except ModelsConfigInvalid:
            if strict:
                raise
            continue
        if pair != (None, None):
            out[p] = pair
    return out


def _check_provider_key_for(merged: Mapping[ConfigSection, Any], model: Model, label: str) -> None:
    """For a resolved model with no key, look at its provider's ``[models]`` pair — the
    key it would have taken. A malformed pair is reported here (the merge skipped it);
    a well-formed one the model did not take because of its ``api_base`` is warned
    about, once per process, since litellm sends its own provider env var there."""
    provider = provider_of(model.model)
    models = merged.get(ConfigSection.MODELS)
    if provider is None or not isinstance(models, dict):
        return
    try:
        api_key, api_key_env = _credentials(models, f"{provider}_", "models", ModelsConfigInvalid)
    except ModelsConfigInvalid:
        if model.api_base is None:
            raise
        return  # not the key this model would have taken
    if model.api_base is None or (api_key, api_key_env) == (None, None):
        return
    form = "api_key" if api_key is not None else "api_key_env"
    message = (
        f"[dgml] 'models.{provider}_{form}' is not sent to {label}: a provider key never "
        f"goes to a custom api_base, so litellm will send its own {provider} env var there "
        f'instead. Set a key beside the api_base (or "EMPTY" if that server checks none).'
    )
    if message not in _WARNED_PROVIDER_KEY_NOT_APPLIED:
        _WARNED_PROVIDER_KEY_NOT_APPLIED.add(message)
        logger.warning(message)


def _take_provider_key(
    table: dict[str, Any], model_key: str, prefix: str, providers: Mapping[str, Credentials]
) -> None:
    """Write the provider key onto one model group that carries no credentials of its
    own and no ``api_base`` — a model at a custom endpoint is not its provider's service."""
    model = table.get(model_key)
    if not isinstance(model, str):
        return
    if any(table.get(f"{prefix}{suffix}") is not None for suffix in GROUP_SUFFIXES):
        return
    api_key, api_key_env = providers.get(provider_of(model) or "", (None, None))
    if api_key is not None:
        table[f"{prefix}api_key"] = api_key
    if api_key_env is not None:
        table[f"{prefix}api_key_env"] = api_key_env


def normalize_models_layer(models: dict[str, Any]) -> dict[str, Any]:
    """One config layer's ``[models]`` table with its groups made atomic, before the
    merge: ``family`` expanded (:func:`expand_family`), each provider pair and each set
    tier completed with explicit ``None``s, and a tier credential whose tier this layer
    does not set — or a tier set to ``null`` — owning that tier as ``None``
    (:func:`_disown_stray_group`). Raises nothing; the loaders report what is
    malformed. An unknown ``family`` is left for :func:`load_models_config`, and a
    stray tier credential beside it is left alone rather than blamed for the typo."""
    family = models.get("family")
    unknown_family = isinstance(family, str) and family not in PROVIDER_MODELS
    out = dict(expand_family(models))
    for p in PROVIDERS:
        if _stray_credential(out, f"{p}_") in ("api_key", "api_key_env"):  # either form set
            out.setdefault(f"{p}_api_key", None)
            out.setdefault(f"{p}_api_key_env", None)
    for t in TIERS:
        prefix = f"{t}_"
        model = out.get(t.value)
        if model is None:
            if not unknown_family:
                _disown_stray_group(out, prefix, t.value)
            if t.value in out:
                _complete_group(out, prefix)
            continue
        if not isinstance(model, str):
            continue  # malformed; load_models_config reports it
        _complete_group(out, prefix)
    return out


def normalize_task_layer(section: dict[str, Any], section_name: ConfigSection) -> dict[str, Any]:
    """One layer of an LLM task section, the same way: every model key it sets
    (``model``, ``label_model``, …) has its group completed, and a credential whose
    model this layer does not set, or a model set to ``null``, owns that model as
    ``None``."""
    out = dict(section)
    prefixes = {k[: -len("model")] for k in _model_keys(section) if section.get(k) is not None}
    strays = {
        key[: -len(suffix)]
        for key, value in section.items()
        for suffix in GROUP_SUFFIXES
        if key.endswith(suffix) and value is not None
    } - prefixes
    for prefix in strays:
        _disown_stray_group(out, prefix, f"{prefix}model")
    for prefix in prefixes:
        if isinstance(section.get(f"{prefix}model"), str):
            _complete_group(out, prefix)
    for key in _model_keys(out):
        if out[key] is None:
            _complete_group(out, key[: -len("model")])
    return out


def apply_provider_keys(merged: Mapping[ConfigSection, Any]) -> None:
    """Write the merged ``[models]`` provider keys onto every tier, section and
    per-call override model with no key of its own and no ``api_base``, in place.
    Called once per read by :func:`~dgml_core.config.load_merged_config`, after the
    overrides are in: a model keyed earlier would keep that key over a higher layer's.
    A malformed pair is skipped; the loader that would have used it reports it."""
    models = merged.get(ConfigSection.MODELS)
    if not isinstance(models, dict):
        return
    keys = provider_keys(models, strict=False)
    if not keys:
        return
    for t in TIERS:
        _take_provider_key(models, t.value, f"{t}_", keys)
    for section in LLM_SECTIONS:
        table = merged.get(section)
        if isinstance(table, dict):
            for key in _model_keys(table):
                _take_provider_key(table, key, key[: -len("model")], keys)


def load_models_config(merged: dict[ConfigSection, Any]) -> ModelsConfig:
    """Build a :class:`ModelsConfig` from the merged mapping's ``[models]`` section (an
    empty section yields an all-``None`` config). ``family`` and the provider pairs
    are validated, not applied — the merge did that per layer and after it. A tier
    credential with no tier is refused here: that is where the merge leaves one a
    layer wrote beside no tier."""
    section = merged.get(ConfigSection.MODELS)
    if section is None:
        return ModelsConfig()
    if not isinstance(section, dict):
        raise ModelsConfigInvalid("'models' must be a table")
    invalid = ModelsConfigInvalid
    family = _opt_str(section.get("family"), "models.family", invalid)
    if family is not None and family not in PROVIDER_MODELS:
        raise _unknown_family(family)
    provider_keys(section)  # for the raise: a malformed pair is this block's error
    tiers: dict[str, Model] = {}
    for t in TIERS:
        model = _opt_str(section.get(t.value), f"models.{t}", invalid)
        api_key, api_key_env = _credentials(section, f"{t}_", "models", invalid)
        api_base = _opt_str(section.get(f"{t}_api_base"), f"models.{t}_api_base", invalid)
        if model is None:
            stray = _stray_credential(section, f"{t}_")
            if stray is not None:
                remedy = f"'models.{t}' (or 'models.family')"
                raise _stray_error(invalid, f"models.{t}_{stray}", f"models.{t}", remedy)
            continue
        tiers[t.value] = Model(model, api_key, api_key_env, api_base)
    return ModelsConfig(**tiers)


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

    A section that names a model takes exactly that section's credentials, never a
    tier's, and the tiers are not read — a malformed tier or family cannot break it.
    A section that names none takes the tier's :class:`Model` whole. Section
    credentials with no section model are refused, not paired with the tier's model.
    A resolved model with no key has its provider's ``[models]`` pair checked, the key
    it would have taken (:func:`_check_provider_key_for`).

    Raises ``invalid`` for a malformed value, a literal+env-name clash, or
    credentials without a model; ``missing`` when neither the section nor the tier
    resolves a model. Opt-in sections (``style`` / ``text_extraction``) check their
    switch before calling this.
    """
    section = merged.get(section_name)
    sec: dict[str, Any] = section if isinstance(section, dict) else {}
    model = _opt_str(sec.get(f"{prefix}model"), f"{section_name}.{prefix}model", invalid)
    api_key, api_key_env = _credentials(sec, prefix, section_name, invalid)
    api_base = _opt_str(sec.get(f"{prefix}api_base"), f"{section_name}.{prefix}api_base", invalid)

    if model is not None:
        resolved = Model(model, api_key, api_key_env, api_base)
        label = f"{section_name}.{prefix}model"
    else:
        stray = _stray_credential(sec, prefix)
        if stray is not None:
            model_key = f"{section_name}.{prefix}model"
            raise _stray_error(
                invalid, f"{section_name}.{prefix}{stray}", model_key, f"'{model_key}'"
            )
        tiers = load_models_config(merged)
        tiered = tiers.resolve(tier)
        if tiered is None:
            raise missing(
                f"no {prefix}model for {section_name}: set [models].family, [models].{tier}, "
                f"or '{section_name}.{prefix}model' in the config"
            )
        resolved = tiered
        label = f"models.{tiers._nearest_set(tier)}"
    if resolved.api_key is None and resolved.api_key_env is None:
        _check_provider_key_for(merged, resolved, label)
    return resolved
