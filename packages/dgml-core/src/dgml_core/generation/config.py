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

"""The ``generation`` section of the workspace config.

The PDF→DGML pipeline uses two models — ``model`` (per-page transcription) and
``label_model`` (the batch-wide semantic-labeling call). Each is optional here:
when unset it falls back to a tier from the ``[models]`` block (transcription →
``standard``, labeling → ``advanced``). Setting the per-task field overrides the
tier. By default ``docset generate`` reads its models solely from the merged
config (:func:`load_generation_config`). Mirrors
:func:`dgml_core.grounded.load_grounded_config`.

Two supported ways run ``docset generate`` against an explicit model config
without hand-editing ``config.toml`` (see :func:`resolve_generation_config`),
mirroring the ``dgml cluster --config PRESET|PATH`` precedent:

* ``--generation-config PROFILE|PATH`` — a bundled named profile
  (:data:`GENERATION_PROFILES`) or a path to a standalone JSON config file.
* ``--model`` / ``--label-model`` — per-run string overrides layered on top.

Both are applied as the merged config's highest-precedence layer
(``load_merged_config(cli_overrides=...)``), so they override the keys they name
while the rest of the resolution is unchanged: unnamed keys (e.g. a workspace
``api_key_env``) survive, and a model neither the overlay nor the config names
still falls back to its ``[models]`` tier.

Both keep the model choice visible/recorded: a profile/file is a checked-in,
named artifact, and the effective models (plus a ``source`` label recording
where they came from) are echoed into the ``docset generate`` JSON output.

The two models can name different providers (e.g. the default ``mixed`` config
uses Anthropic for transcription and Gemini for labeling), so each carries its
own credentials: ``api_key`` / ``api_key_env`` / ``api_base`` for transcription
and ``label_api_key`` / ``label_api_key_env`` / ``label_api_base`` for labeling.
These apply whether the models are set here or come from their tiers; the tiers
themselves carry no credentials.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from dgml_core.config import load_merged_config
from dgml_core.errors import (
    AuthError,
    GenerationConfigInvalid,
    GenerationConfigMissing,
    short_error_message,
)
from dgml_core.models_config import ConfigSection, Tier, resolve_tiered_model
from dgml_core.storage import Workspace


@dataclass(frozen=True)
class GenerationConfig:
    """Parsed ``generation`` models, with each model's resolved credentials.

    ``model`` (per-page transcription) and ``label_model`` (the single
    batch-wide semantic-labeling call) each resolve from the per-task field or
    its ``[models]`` tier (``standard`` / ``advanced``). Transcription is the
    bulk of the calls and runs well on a cheaper tier; labeling is a handful of
    small-output calls per batch that benefit from a stronger model.

    Each model has independent credentials so the two may name different
    providers. For either, API-key resolution precedence is: literal
    ``*_api_key`` > ``*_api_key_env`` var lookup > ``None`` (litellm's
    per-provider env-var conventions). Setting both the literal and the env-name
    for one model is a config error.
    """

    model: str
    label_model: str
    # transcription (``model``) credentials
    api_key: str | None = None
    api_key_env: str | None = None
    api_base: str | None = None
    # labeling (``label_model``) credentials
    label_api_key: str | None = None
    label_api_key_env: str | None = None
    label_api_base: str | None = None


def _resolve_from_merged(merged: dict[ConfigSection, Any]) -> GenerationConfig:
    """Resolve both generation models out of an already-merged config mapping.

    Shared by :func:`load_generation_config` and :func:`resolve_generation_config`
    so the workspace path and the ``--generation-config`` / ``--model`` /
    ``--label-model`` override path validate and tier-resolve identically.
    """
    transcribe = resolve_tiered_model(
        merged,
        section_name=ConfigSection.GENERATION,
        tier=Tier.STANDARD,
        invalid=GenerationConfigInvalid,
        missing=GenerationConfigMissing,
        model_field="model",
        key_field="api_key",
        env_field="api_key_env",
        base_field="api_base",
    )
    label = resolve_tiered_model(
        merged,
        section_name=ConfigSection.GENERATION,
        tier=Tier.ADVANCED,
        invalid=GenerationConfigInvalid,
        missing=GenerationConfigMissing,
        model_field="label_model",
        key_field="label_api_key",
        env_field="label_api_key_env",
        base_field="label_api_base",
    )
    return GenerationConfig(
        model=transcribe.model,
        label_model=label.model,
        api_key=transcribe.api_key,
        api_key_env=transcribe.api_key_env,
        api_base=transcribe.api_base,
        label_api_key=label.api_key,
        label_api_key_env=label.api_key_env,
        label_api_base=label.api_base,
    )


def load_generation_config(workspace: Workspace) -> GenerationConfig:
    """Resolve the two generation models (transcription, labeling) and their
    credentials from the merged config's ``[generation]`` section and ``[models]``
    tiers (``standard`` for transcription, ``advanced`` for labeling)."""
    return _resolve_from_merged(load_merged_config(workspace))


# Bundled, named generation profiles — each a checked-in ``generation`` overlay
# using model ids already referenced by the project. ``fast`` runs the cheap tier
# for both passes; ``balanced`` matches the shipped default (Haiku transcription +
# Sonnet labeling); ``quality`` escalates both. Selected by name via
# ``--generation-config <name>``; a name that is NOT one of these is treated as a
# file path instead. Kept deliberately small — model ids drift, so the config file
# remains the primary source of truth.
GENERATION_PROFILES: tuple[str, ...] = ("fast", "balanced", "quality")


def load_generation_profile(name: str) -> dict[str, Any]:
    """Load a bundled generation profile (``fast`` / ``balanced`` / ``quality``).

    Returns the profile's ``generation``-section dict (same shape as the
    ``[generation]`` section of ``config.toml``). Raises
    :class:`GenerationConfigInvalid` for an unknown profile name.
    """
    if name not in GENERATION_PROFILES:
        raise GenerationConfigInvalid(
            f"unknown generation profile {name!r}; choose one of "
            f"{', '.join(GENERATION_PROFILES)}, or pass a path to a config JSON"
        )
    text = (resources.files("dgml_core") / f"generation_profile_{name}.json").read_text(
        encoding="utf-8"
    )
    data = json.loads(text)
    if not isinstance(data, dict):
        raise GenerationConfigInvalid(f"generation profile {name!r} is not a JSON object")
    return data


def load_generation_config_file(path: Path) -> dict[str, Any]:
    """Read a standalone generation config JSON file (the ``--generation-config`` flag).

    The file is a JSON object holding the same fields the ``[generation]`` section
    of ``config.toml`` would (``model``, ``label_model``, optional
    ``api_key`` / ``api_key_env`` / ``api_base`` and their ``label_*`` twins). For
    convenience a full-config file wrapping the section under a top-level
    ``"generation"`` key is also accepted (the section is unwrapped).

    Raises :class:`GenerationConfigInvalid` when the file is missing, not valid
    JSON, or not a JSON object. Field-level validation happens in
    :func:`resolve_generation_config` via the shared tier resolution.
    """
    if not path.exists():
        raise GenerationConfigInvalid(f"generation config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GenerationConfigInvalid(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise GenerationConfigInvalid(f"{path} must contain a JSON object")
    nested = data.get("generation")
    return nested if isinstance(nested, dict) else data


def resolve_generation_config(
    workspace: Workspace,
    *,
    config: str | None = None,
    model: str | None = None,
    label_model: str | None = None,
) -> tuple[GenerationConfig, str]:
    """Resolve the effective generation config for a ``docset generate`` run.

    Returns the validated :class:`GenerationConfig` and a human-readable
    ``source`` label recording where it came from (echoed into the CLI JSON
    output so the model choice stays visible/recorded).

    Precedence, highest first:

    1. ``model`` / ``label_model`` — the ``--model`` / ``--label-model`` flags.
    2. ``config`` — the ``--generation-config`` flag: a bundled profile name in
       :data:`GENERATION_PROFILES` (:func:`load_generation_profile`), otherwise a
       path to a standalone config file (:func:`load_generation_config_file`).
    3. The merged config — workspace ``config.toml`` > user config > ``[models]``
       tiers, exactly as :func:`load_generation_config` resolves it.

    1 and 2 are collapsed into one ``[generation]`` overlay handed to
    :func:`~dgml_core.config.load_merged_config` as ``cli_overrides``, its
    highest-precedence layer. So an overlay overrides the keys it names and
    leaves the rest of the resolution intact — a workspace ``api_key_env`` still
    applies, and a model the overlay doesn't name still falls back to its tier.
    Passing both ``--model`` and ``--label-model`` therefore fully specifies the
    models and works with no ``[generation]`` section at all.

    With no ``config`` and no overrides this is exactly
    :func:`load_generation_config` (so an unresolvable model still raises
    :class:`GenerationConfigMissing`).

    ``source`` is one of ``"config"``, ``"profile:<name>"``, ``"file"``, or
    ``"override"``; a profile/file combined with a flag override reads
    ``"profile:<name>+override"`` / ``"file+override"``.
    """
    if config is None and model is None and label_model is None:
        return load_generation_config(workspace), "config"

    if config is None:
        overlay: dict[str, Any] = {}
        base_source = "config"
    elif config in GENERATION_PROFILES:
        overlay = dict(load_generation_profile(config))
        base_source = f"profile:{config}"
    else:
        overlay = dict(load_generation_config_file(Path(config)))
        base_source = "file"

    if model is not None:
        overlay["model"] = model
    if label_model is not None:
        overlay["label_model"] = label_model

    overridden = model is not None or label_model is not None
    if base_source == "config":
        source = "override" if overridden else "config"
    else:
        source = f"{base_source}+override" if overridden else base_source

    merged = load_merged_config(workspace, cli_overrides={ConfigSection.GENERATION.value: overlay})
    return _resolve_from_merged(merged), source


def _resolve_key(literal: str | None, env_name: str | None, ref: str) -> str | None:
    """Precedence: literal key > env-var lookup > ``None`` (litellm's per-provider
    env-var conventions). ``ref`` names the config field for the error message."""
    if literal is not None:
        return literal
    if env_name is None:
        return None
    key = os.environ.get(env_name)
    if not key:
        raise AuthError(f"environment variable ${env_name} is not set (referenced by '{ref}')")
    return key


def resolve_generation_api_key(config: GenerationConfig) -> str | None:
    """Resolve the transcription (``model``) API key."""
    return _resolve_key(config.api_key, config.api_key_env, "generation.api_key_env")


def resolve_generation_label_api_key(config: GenerationConfig) -> str | None:
    """Resolve the labeling (``label_model``) API key."""
    return _resolve_key(
        config.label_api_key, config.label_api_key_env, "generation.label_api_key_env"
    )


def validate_generation_models(
    config: GenerationConfig,
    transcribe_key: str | None,
    label_key: str | None,
) -> None:
    """Pre-flight the generation models, before any transcription spend.

    Fails fast on the two misconfigurations detectable offline, so a run whose
    labeling is doomed doesn't first burn the transcription budget:

    * a wholly-malformed model string (no resolvable provider) — raises
      :class:`GenerationConfigInvalid`;
    * a missing API key for a model's provider — raises :class:`AuthError`.

    It CANNOT catch a *present-but-wrong* key or a *well-formed-but-nonexistent*
    model id (e.g. ``anthropic/claude-typo``): both resolve a provider and are
    rejected only when the model is actually called. Those surface per file as a
    ``label_error`` during ``docset generate`` rather than here.

    ``transcribe_key`` / ``label_key`` come from :func:`resolve_generation_api_key`
    / :func:`resolve_generation_label_api_key` — each model is checked against its
    own key and its own ``api_base``, since the two may name different providers.
    The key check is SKIPPED for a model whose ``api_base`` is set — a custom
    endpoint (proxy / gateway / self-hosted) may authenticate differently, and
    litellm would otherwise report the provider's conventional env var as missing
    (a false abort).
    """
    import litellm

    checks = (
        (config.model, transcribe_key, config.api_base),
        (config.label_model, label_key, config.label_api_base),
    )
    # Dedupe identical (model, key, base) triples — the two models are often the
    # same provider and frequently the same string.
    for model, key, api_base in dict.fromkeys(checks):
        try:
            litellm.get_llm_provider(model)
        except Exception as exc:
            # get_llm_provider only parses the provider prefix, so this fires
            # only for a string with no resolvable provider (''/'::::'/bare
            # typo) — NOT for a valid-provider/bad-model id, which passes here.
            raise GenerationConfigInvalid(
                f"'{model}' is not a recognized model string "
                "(expected e.g. 'anthropic/claude-sonnet-5'): "
                f"{short_error_message(exc)}"
            ) from exc
        if api_base is not None:
            continue
        env = litellm.validate_environment(model=model, api_key=key)
        if not env.get("keys_in_environment", False):
            missing = ", ".join(env.get("missing_keys") or []) or "the provider API key"
            raise AuthError(
                f"no API key for model '{model}': {missing} not set. Set it in the "
                "environment, or configure the matching 'generation.*api_key' / "
                "'generation.*api_key_env' in the config."
            )
