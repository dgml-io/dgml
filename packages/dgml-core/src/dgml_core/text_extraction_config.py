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
heuristic (see :mod:`dgml.hybrid`). When the ``text_extraction`` section of
``config.toml`` sets ``enabled = true``, the per-region merge decision is
delegated to the configured LLM instead — letting it choose digital text, OCR
text, or a combination (e.g. de-ligaturing, fixing a run-together word).

This section *tunes the merge within hybrid mode*; it does **not** select
the text mode. The ``--text-mode`` flag still chooses which extractor
runs.

**``enabled = true`` is the switch** — the section's presence configures the
feature but does not turn it on. Without it :func:`load_text_extraction_config`
returns ``None`` and hybrid falls back to the heuristic. See
:mod:`dgml_core.style_config` for the rationale (shipped-template safety, and
the warning for a configured-but-disabled section); the two behave identically.

Config shape (``model`` falls back to the ``[models].standard`` tier)::

    [text_extraction]
    enabled = true
    model = "ollama_chat/gemma4:latest"
    api_base = "http://localhost:11434"
    temperature = 0.0
    max_tokens = 4000

API key resolution mirrors :mod:`dgml.classification`: literal
``api_key`` > env-name lookup via ``api_key_env`` > litellm's per-provider
default env var. Setting both ``api_key`` and ``api_key_env`` is an error.
Local providers like Ollama need no key at all — set only ``api_base``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .config import load_merged_config
from .errors import AuthError, TextExtractionConfigInvalid
from .models_config import ConfigSection, Tier, resolve_tiered_model, section_enabled
from .storage import Workspace

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
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    temperature: float | None = DEFAULT_TEMPERATURE
    max_tokens: int | None = DEFAULT_MAX_TOKENS


def load_text_extraction_config(workspace: Workspace) -> TextExtractionConfig | None:
    """Read and validate the ``text_extraction`` section of the merged config.

    Returns ``None`` unless the section sets ``enabled = true`` — hybrid mode then
    uses its heuristic merge. The section's presence configures the feature, it
    does not switch it on. When enabled, ``model`` may be omitted to fall back to
    the ``[models].standard`` tier. Raises :class:`TextExtractionConfigInvalid`
    when malformed.
    """
    merged = load_merged_config(workspace)
    section = merged.get(ConfigSection.TEXT_EXTRACTION)
    if section is None:
        return None
    if not isinstance(section, dict):
        raise TextExtractionConfigInvalid("'text_extraction' must be a table")
    sec: dict[str, Any] = section

    # Nothing below this line may run for a disabled section — see the matching
    # comment in `style_config.load_style_config` for why.
    if not section_enabled(
        sec, section_name=ConfigSection.TEXT_EXTRACTION, invalid=TextExtractionConfigInvalid
    ):
        return None

    # Enabled but no model → invalid (the tier only supplies a model for an
    # enabled feature; it does not turn the feature on).
    rm = resolve_tiered_model(
        merged,
        section_name=ConfigSection.TEXT_EXTRACTION,
        tier=Tier.STANDARD,
        invalid=TextExtractionConfigInvalid,
        missing=TextExtractionConfigInvalid,
    )

    temperature = sec.get("temperature", DEFAULT_TEMPERATURE)
    if temperature is not None and (
        not isinstance(temperature, int | float) or isinstance(temperature, bool)
    ):
        raise TextExtractionConfigInvalid("'text_extraction.temperature' must be a number if set")

    max_tokens = sec.get("max_tokens", DEFAULT_MAX_TOKENS)
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1
    ):
        raise TextExtractionConfigInvalid(
            "'text_extraction.max_tokens' must be a positive integer if set"
        )

    return TextExtractionConfig(
        model=rm.model,
        api_base=rm.api_base,
        api_key=rm.api_key,
        api_key_env=rm.api_key_env,
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
            "(referenced by text_extraction.api_key_env in the config)"
        )
    return key


__all__ = [
    "TextExtractionConfig",
    "load_text_extraction_config",
    "resolve_api_key",
]
