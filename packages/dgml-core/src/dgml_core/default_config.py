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

"""Default `[models]` tables, one per provider family.

:data:`PROVIDER_MODELS` is the single source for every provider surface: the
`--provider` choices and the config placeholder text, `canonical_provider`,
`load_models_config` (which expands `[models].family = "<key>"` into these
four tiers at runtime; an explicitly set tier overrides its default), and the
expansion `dgml init` reports on stderr. The generated config names only the
family, so a family-based config *tracks* these defaults across dgml upgrades;
users who need a reproducible pin set explicit tiers instead.

Kept apart from :mod:`dgml_core.storage` (the path-resolution and TOML-writing
machinery) so the shipped model defaults live in one obvious place and can be
updated without touching the config generator.
"""

from __future__ import annotations

# The four tiers, cheapest → strongest, per provider family. The tier→task
# mapping and per-task overrides are documented in the CLI reference, not
# baked into the config file (the mapping may change without a config rewrite).
#
# The cheap Gemini tier uses the `gemini-flash-lite-latest` *alias*, not a
# pinned version. A pinned flash-lite (`gemini-2.5-flash-lite`) was the default
# here until Google closed it to new API users — it began returning HTTP 404
# ("no longer available to new users") on a fresh key, so a new user running
# `dgml init --provider google` got a config that failed on first use. The
# alias tracks the current flash-lite and cannot go stale that way. Users who
# need a reproducible pin can still set an explicit model in their own config.
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "anthropic_google": {
        # Gemini Flash-Lite for the cheap high-volume vision work
        # (classification/style); Anthropic for the document-reasoning pipeline
        # (transcription → labeling/value-extraction → schema generation).
        "light": "gemini/gemini-flash-lite-latest",
        "standard": "anthropic/claude-haiku-4-5",
        "advanced": "anthropic/claude-sonnet-5",
        "expert": "anthropic/claude-opus-5",
    },
    "anthropic": {
        # `advanced` (labeling, value extraction) and `expert` (schema
        # generation) carry the reasoning-heavy work, so they track the current
        # Sonnet/Opus generation. Both allow 128K output tokens — twice the
        # previous generation's ceiling, which matters for value extraction on
        # charge-heavy documents (see grounded._DEFAULT_MAX_COMPLETION_TOKENS).
        # `light`/`standard` stay on Haiku 4.5: there is no Haiku 5, and the
        # cheap tiers do classification and transcription where it holds up.
        "light": "anthropic/claude-haiku-4-5",
        "standard": "anthropic/claude-haiku-4-5",
        "advanced": "anthropic/claude-sonnet-5",
        "expert": "anthropic/claude-opus-5",
    },
    "google": {
        "light": "gemini/gemini-flash-lite-latest",
        "standard": "gemini/gemini-2.5-flash",
        "advanced": "gemini/gemini-2.5-pro",
        "expert": "gemini/gemini-2.5-pro",
    },
    "openai": {
        # The gpt-5.4 family, sized to the tiers: nano/mini for the cheap
        # high-volume vision work, the full model for the reasoning-heavy
        # labeling / value-extraction / schema-generation passes. Every entry
        # accepts native PDF input and images (which the generation pipeline
        # requires) and allows 128K output tokens — the same ceiling the
        # frontier Claude/Gemini tiers give, so value extraction on
        # charge-heavy documents is not capped lower here than elsewhere.
        # `advanced` and `expert` are deliberately the same id: OpenAI's
        # reasoning depth is a per-call `reasoning_effort` knob rather than a
        # separate stronger model, and the call sites already ask for the
        # effort each task needs (see `grounded._DEFAULT_REASONING_EFFORT`).
        "light": "openai/gpt-5.4-nano",
        "standard": "openai/gpt-5.4-mini",
        "advanced": "openai/gpt-5.4",
        "expert": "openai/gpt-5.4",
    },
}

# Per family, the API-key env var(s) its tiers need at runtime (the standard
# names litellm reads). Same keys as PROVIDER_MODELS — a test pins that.
PROVIDER_API_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic_google": ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}
