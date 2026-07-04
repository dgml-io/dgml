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

The PDF→DGML pipeline's models are configured here, not defaulted in code, so
*which* model runs is a visible choice in ``<workspace>/config.json`` — never a
silent default. Like the other model-consuming stages (grounded, classification)
there is no CLI flag: ``docset generate`` reads its model solely from this
section. Mirrors :func:`dgml_core.grounded.load_grounded_config`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dgml_core.errors import (
    AuthError,
    CorruptMetadata,
    GenerationConfigInvalid,
    GenerationConfigMissing,
)
from dgml_core.storage import Workspace, read_config


@dataclass(frozen=True)
class GenerationConfig:
    """Parsed ``generation`` section of the workspace config.

    ``model`` (per-page transcription) and ``label_model`` (the
    single batch-wide semantic-labeling call) are **both required** and
    configured separately: transcription is the bulk of the calls and runs well
    on a cheap tier, while labeling is a handful of small-output calls per batch
    that can benefit from a stronger model. Each is an explicit, visible choice,
    so no model runs that the config didn't name.

    API key resolution, in order of precedence:
    1. ``api_key``      — literal key in the config file. Allowed but only
                          safe in workspaces that aren't shared or checked in.
    2. ``api_key_env``  — name of an env var holding the key.
    3. Neither          — litellm falls back to its per-provider env-var
                          conventions (``ANTHROPIC_API_KEY``, etc.).

    Setting both ``api_key`` and ``api_key_env`` is a config error.
    """

    model: str
    label_model: str
    api_key: str | None = None
    api_key_env: str | None = None
    api_base: str | None = None


def _validate_optional_str(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise GenerationConfigInvalid(f"'{field_name}' must be a non-empty string if set")
    return value


def load_generation_config(workspace: Workspace) -> GenerationConfig:
    """Read and validate the ``generation`` section of ``<workspace>/config.json``."""
    if not workspace.config_path.exists():
        raise GenerationConfigMissing(
            f"no config.json at {workspace.config_path}; generation requires a workspace "
            "config with a 'generation' section naming both models — e.g. "
            '{"generation": {"model": "anthropic/claude-haiku-4-5", '
            '"label_model": "anthropic/claude-sonnet-4-6"}}'
        )
    try:
        data = read_config(workspace.config_path)
    except CorruptMetadata as exc:
        raise GenerationConfigInvalid(f"{workspace.config_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise GenerationConfigInvalid(f"{workspace.config_path} must contain a JSON object")
    section = data.get("generation")
    if section is None:
        raise GenerationConfigMissing(
            f"{workspace.config_path} has no 'generation' section "
            "(add one naming both 'model' and 'label_model')"
        )
    if not isinstance(section, dict):
        raise GenerationConfigInvalid("'generation' must be a JSON object")

    model = section.get("model")
    if not isinstance(model, str) or not model.strip():
        raise GenerationConfigInvalid(
            "'generation.model' must be a non-empty string (e.g. 'anthropic/claude-haiku-4-5')"
        )
    label_model = section.get("label_model")
    if not isinstance(label_model, str) or not label_model.strip():
        raise GenerationConfigInvalid(
            "'generation.label_model' must be a non-empty string "
            "(e.g. 'anthropic/claude-sonnet-4-6'); it is required"
        )
    api_key = _validate_optional_str(section.get("api_key"), "generation.api_key")
    api_key_env = _validate_optional_str(section.get("api_key_env"), "generation.api_key_env")
    api_base = _validate_optional_str(section.get("api_base"), "generation.api_base")
    if api_key is not None and api_key_env is not None:
        raise GenerationConfigInvalid(
            "set at most one of 'generation.api_key' / 'generation.api_key_env', not both"
        )

    return GenerationConfig(
        model=model,
        label_model=label_model,
        api_key=api_key,
        api_key_env=api_key_env,
        api_base=api_base,
    )


def resolve_generation_api_key(config: GenerationConfig) -> str | None:
    """Resolve the generation API key.

    Precedence: literal ``api_key`` > ``api_key_env`` var lookup > ``None``
    (let litellm fall back to its per-provider env-var conventions:
    ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, ...).

    Mutual exclusion of ``api_key`` and ``api_key_env`` is enforced in
    :func:`load_generation_config`.
    """
    if config.api_key is not None:
        return config.api_key
    if config.api_key_env is None:
        return None
    key = os.environ.get(config.api_key_env)
    if not key:
        raise AuthError(
            f"environment variable ${config.api_key_env} is not set "
            "(referenced by 'generation.api_key_env' in config.json)"
        )
    return key
