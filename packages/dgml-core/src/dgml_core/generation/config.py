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
tier. There is no CLI flag: ``docset generate`` reads its models solely from the
merged config. Mirrors :func:`dgml_core.grounded.load_grounded_config`.

The two models can name different providers (e.g. the default ``mixed`` config
uses Anthropic for transcription and Gemini for labeling), so each carries its
own credentials: ``api_key`` / ``api_key_env`` / ``api_base`` for transcription
and ``label_api_key`` / ``label_api_key_env`` / ``label_api_base`` for labeling.
These apply whether the models are set here or come from their tiers; the tiers
themselves carry no credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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


def load_generation_config(workspace: Workspace) -> GenerationConfig:
    """Resolve the two generation models (transcription, labeling) and their
    credentials from the merged config's ``[generation]`` section and ``[models]``
    tiers (``standard`` for transcription, ``advanced`` for labeling)."""
    merged = load_merged_config(workspace)
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
