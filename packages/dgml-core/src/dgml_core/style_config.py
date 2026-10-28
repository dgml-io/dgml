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

"""Optional LLM configuration for image-based ``dg:style`` on OCR files.

For ``--text-mode digital``/``hybrid`` files, ``dg:style`` is derived
deterministically from the PDF glyphs during grounding (see
:mod:`dgml.style` / :mod:`dgml.xml_grounding`). OCR files carry no font
facts, so their ``dg:style`` is empty — unless a workspace opts in via the
``style`` section of ``config.toml``, which lets a vision model read each
page image and report the observed formatting (see :mod:`dgml.style_llm`).

This is off by default: **``enabled = true`` is the switch.** Without it
:func:`load_style_config` returns ``None`` and grounding leaves OCR files
unstyled. The section's *presence* configures the feature but does not turn it
on — ``dgml init`` ships ``[style] enabled = false`` so the feature is
discoverable without being enabled, and a disabled section is never validated
(a template carrying no ``model`` must not raise). A section that is configured
but not enabled warns, so a config predating this flag is not silently ignored.

Honored only for files whose recorded ``text_mode`` is ``ocr``; it never
competes with the deterministic digital/hybrid path.

Config shape (``model`` is optional — it falls back to the ``[models].light``
tier — but the feature only runs when ``enabled``)::

    [style]
    enabled = true
    model = "anthropic/claude-haiku-4-5"
    api_base = "http://localhost:11434"
    max_tokens = 4000

API key resolution mirrors :mod:`dgml.text_extraction_config`: literal
``api_key`` > env-name lookup via ``api_key_env`` > litellm's per-provider
default env var. Setting both ``api_key`` and ``api_key_env`` is an error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .config import load_merged_config
from .errors import AuthError, StyleConfigInvalid
from .models_config import ConfigSection, Tier, resolve_tiered_model, section_enabled
from .storage import Workspace

DEFAULT_MAX_TOKENS = 4000


@dataclass(frozen=True)
class StyleConfig:
    """Parsed ``style`` section of the workspace config.

    Existence of this object means the OCR image-based path is
    enabled; :func:`load_style_config` returns ``None`` when the section is
    absent. ``model`` is the vision model it uses (always populated — the
    loader requires it whenever the section is present).
    """

    model: str
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)
    api_key_env: str | None = None
    max_tokens: int | None = DEFAULT_MAX_TOKENS


def load_style_config(workspace: Workspace) -> StyleConfig | None:
    """Read and validate the ``style`` section of the merged config.

    Returns ``None`` unless the section sets ``enabled = true`` — the section's
    presence configures the feature, it does not switch it on. When enabled,
    ``model`` may be omitted to fall back to the ``[models].light`` tier. Raises
    :class:`StyleConfigInvalid` when malformed.
    """
    merged = load_merged_config(workspace)
    section = merged.get(ConfigSection.STYLE)
    if section is None:
        return None
    if not isinstance(section, dict):
        raise StyleConfigInvalid("'style' must be a table")
    sec: dict[str, Any] = section

    # Nothing below this line may run for a disabled section: the shipped
    # config.toml carries `[style] enabled = false` with no model, and the
    # no-API-key template comments `[models]` out entirely — so validating a
    # disabled section would make the default config raise on every load.
    if not section_enabled(sec, section_name=ConfigSection.STYLE, invalid=StyleConfigInvalid):
        return None

    # Enabled but no model → StyleConfigInvalid (the tier only supplies a model
    # for an enabled feature; it does not turn the feature on).
    rm = resolve_tiered_model(
        merged,
        section_name=ConfigSection.STYLE,
        tier=Tier.LIGHT,
        invalid=StyleConfigInvalid,
        missing=StyleConfigInvalid,
    )

    max_tokens = sec.get("max_tokens", DEFAULT_MAX_TOKENS)
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1
    ):
        raise StyleConfigInvalid("'style.max_tokens' must be a positive integer if set")

    return StyleConfig(
        model=rm.model,
        api_base=rm.api_base,
        api_key=rm.api_key,
        api_key_env=rm.api_key_env,
        max_tokens=max_tokens,
    )


def resolve_api_key(config: StyleConfig) -> str | None:
    """Resolve the API key for the style LLM.

    Precedence: literal ``config.api_key`` > env-name lookup via
    ``config.api_key_env`` > ``None`` (litellm falls back to its own
    per-provider env var; local providers like Ollama need none). Mutual
    exclusion of the two config fields is enforced in
    :func:`load_style_config`.
    """
    if config.api_key:
        return config.api_key
    if not config.api_key_env:
        return None
    key = os.environ.get(config.api_key_env)
    if not key:
        raise AuthError(
            f"environment variable ${config.api_key_env} is not set "
            "(referenced by style.api_key_env in the config)"
        )
    return key


__all__ = [
    "StyleConfig",
    "load_style_config",
    "resolve_api_key",
]
