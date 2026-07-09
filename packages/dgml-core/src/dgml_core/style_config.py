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
facts, so their ``dg:style`` is empty — unless a workspace opts in via a
``style`` section in ``config.json``, which lets a vision model read each
page image and report the observed formatting (see :mod:`dgml.style_llm`).

This is off by default: **the section's presence is the
switch.** When it is absent, :func:`load_style_config` returns ``None`` and
grounding leaves OCR files unstyled — so existing workspaces are unchanged.
When present it must name a vision ``model``. The setting is honored only
for files whose recorded ``text_mode`` is ``ocr``; it never competes with
the deterministic digital/hybrid path.

Config shape (``model`` is required when the section is present)::

    {
      "style": {
        "model": "anthropic/claude-haiku-4-5",
        "api_base": "http://localhost:11434",
        "max_tokens": 4000
      }
    }

API key resolution mirrors :mod:`dgml.text_extraction_config`: literal
``api_key`` > env-name lookup via ``api_key_env`` > litellm's per-provider
default env var. Setting both ``api_key`` and ``api_key_env`` is an error.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import AuthError, CorruptMetadata, StyleConfigInvalid
from .storage import Workspace, read_config

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
    api_key: str | None = None
    api_key_env: str | None = None
    max_tokens: int | None = DEFAULT_MAX_TOKENS


def load_style_config(workspace: Workspace) -> StyleConfig | None:
    """Read and validate the ``style`` section of ``config.json``.

    Returns ``None`` when no config file exists or no ``style`` section is
    present. Raises :class:`StyleConfigInvalid` when the section exists but
    is malformed.
    """
    if not workspace.config_path.exists():
        return None

    try:
        data = read_config(workspace.config_path)
    except CorruptMetadata as exc:
        raise StyleConfigInvalid(f"{workspace.config_path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise StyleConfigInvalid(f"{workspace.config_path} must contain a JSON object")

    section = data.get("style")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise StyleConfigInvalid("'style' must be a JSON object")

    model = section.get("model")
    if not isinstance(model, str) or not model.strip():
        raise StyleConfigInvalid(
            "'style.model' must be a non-empty string (the vision model that reads "
            "page images, e.g. 'anthropic/claude-haiku-4-5')"
        )

    api_base = section.get("api_base")
    if api_base is not None and (not isinstance(api_base, str) or not api_base):
        raise StyleConfigInvalid("'style.api_base' must be a non-empty string if set")

    api_key = section.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or not api_key):
        raise StyleConfigInvalid("'style.api_key' must be a non-empty string if set")

    api_key_env = section.get("api_key_env")
    if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
        raise StyleConfigInvalid("'style.api_key_env' must be a non-empty env var name if set")

    if api_key is not None and api_key_env is not None:
        raise StyleConfigInvalid(
            "set at most one of 'style.api_key' / 'style.api_key_env', not both"
        )

    max_tokens = section.get("max_tokens", DEFAULT_MAX_TOKENS)
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1
    ):
        raise StyleConfigInvalid("'style.max_tokens' must be a positive integer if set")

    return StyleConfig(
        model=model,
        api_base=api_base,
        api_key=api_key,
        api_key_env=api_key_env,
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
            "(referenced by style.api_key_env in config.json)"
        )
    return key


__all__ = [
    "StyleConfig",
    "load_style_config",
    "resolve_api_key",
]
