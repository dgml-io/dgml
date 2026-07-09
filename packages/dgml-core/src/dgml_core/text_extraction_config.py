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

"""Optional LLM configuration for hybrid text-extraction merging.

Hybrid mode (``--text-mode hybrid``) reconciles digital and OCR word
streams per page. By default it uses a deterministic Levenshtein/region
heuristic (see :mod:`dgml.hybrid`). When a workspace declares a
``text_extraction`` section in ``config.json``, the per-region merge
decision is delegated to the configured LLM instead — letting it choose
digital text, OCR text, or a combination (e.g. de-ligaturing, fixing a
run-together word).

This section *tunes the merge within hybrid mode*; it does **not** select
the text mode. The ``--text-mode`` flag still chooses which extractor
runs. When the section is absent, :func:`load_text_extraction_config`
returns ``None`` and hybrid falls back to the heuristic — so existing
workspaces are unchanged.

Config shape (all but ``model`` optional)::

    {
      "text_extraction": {
        "model": "ollama_chat/gemma4:latest",
        "api_base": "http://localhost:11434",
        "temperature": 0.0,
        "max_tokens": 4000
      }
    }

API key resolution mirrors :mod:`dgml.classification`: literal
``api_key`` > env-name lookup via ``api_key_env`` > litellm's per-provider
default env var. Setting both ``api_key`` and ``api_key_env`` is an error.
Local providers like Ollama need no key at all — set only ``api_base``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import AuthError, CorruptMetadata, TextExtractionConfigInvalid
from .storage import Workspace, read_config

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4000


@dataclass(frozen=True)
class TextExtractionConfig:
    """Parsed ``text_extraction`` section of the workspace config.

    By construction this object is well-formed:
    :func:`load_text_extraction_config` validates each field before
    returning. ``temperature`` defaults to ``0.0`` so the merge is
    deterministic; ``api_base`` carries the endpoint local providers
    (Ollama) require.
    """

    model: str
    api_base: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    temperature: float | None = DEFAULT_TEMPERATURE
    max_tokens: int | None = DEFAULT_MAX_TOKENS


def load_text_extraction_config(workspace: Workspace) -> TextExtractionConfig | None:
    """Read and validate the ``text_extraction`` section of ``config.json``.

    Returns ``None`` when no config file exists or no ``text_extraction``
    section is present — hybrid mode then uses its heuristic merge. Raises
    :class:`TextExtractionConfigInvalid` when the section exists but is
    malformed.
    """
    if not workspace.config_path.exists():
        return None

    try:
        data = read_config(workspace.config_path)
    except CorruptMetadata as exc:
        raise TextExtractionConfigInvalid(
            f"{workspace.config_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise TextExtractionConfigInvalid(f"{workspace.config_path} must contain a JSON object")

    section = data.get("text_extraction")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise TextExtractionConfigInvalid("'text_extraction' must be a JSON object")

    model = section.get("model")
    if not isinstance(model, str) or not model.strip():
        raise TextExtractionConfigInvalid(
            "'text_extraction.model' must be a non-empty string (e.g. 'ollama_chat/gemma4:latest')"
        )

    api_base = section.get("api_base")
    if api_base is not None and (not isinstance(api_base, str) or not api_base):
        raise TextExtractionConfigInvalid(
            "'text_extraction.api_base' must be a non-empty string if set"
        )

    api_key = section.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or not api_key):
        raise TextExtractionConfigInvalid(
            "'text_extraction.api_key' must be a non-empty string if set"
        )

    api_key_env = section.get("api_key_env")
    if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
        raise TextExtractionConfigInvalid(
            "'text_extraction.api_key_env' must be a non-empty env var name if set"
        )

    if api_key is not None and api_key_env is not None:
        raise TextExtractionConfigInvalid(
            "set at most one of 'text_extraction.api_key' / 'text_extraction.api_key_env', not both"
        )

    temperature = section.get("temperature", DEFAULT_TEMPERATURE)
    if temperature is not None and (
        not isinstance(temperature, int | float) or isinstance(temperature, bool)
    ):
        raise TextExtractionConfigInvalid("'text_extraction.temperature' must be a number if set")

    max_tokens = section.get("max_tokens", DEFAULT_MAX_TOKENS)
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1
    ):
        raise TextExtractionConfigInvalid(
            "'text_extraction.max_tokens' must be a positive integer if set"
        )

    return TextExtractionConfig(
        model=model,
        api_base=api_base,
        api_key=api_key,
        api_key_env=api_key_env,
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=max_tokens,
    )


def resolve_api_key(config: TextExtractionConfig) -> str | None:
    """Resolve the API key for the merge LLM.

    Precedence: literal ``config.api_key`` > env-name lookup via
    ``config.api_key_env`` > ``None`` (litellm falls back to its own
    per-provider env var; local providers like Ollama need none). Mutual
    exclusion of the two config fields is enforced in
    :func:`load_text_extraction_config`.
    """
    if config.api_key:
        return config.api_key
    if not config.api_key_env:
        return None
    key = os.environ.get(config.api_key_env)
    if not key:
        raise AuthError(
            f"environment variable ${config.api_key_env} is not set "
            "(referenced by text_extraction.api_key_env in config.json)"
        )
    return key


__all__ = [
    "TextExtractionConfig",
    "load_text_extraction_config",
    "resolve_api_key",
]
