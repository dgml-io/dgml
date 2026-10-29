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
# The Gemini tiers are pinned versions, so a dgml upgrade — not Google moving
# an alias — is what changes the model a workspace runs. The pins must be ones
# Google still serves to *new* API users: the previous generation was closed to
# them before its shutdown date (HTTP 404, "no longer available to new users"),
# so a fresh `dgml init --provider google` got a config that failed on first
# use. Move a pin when Google closes it, using the replacement its deprecations
# page names. `advanced`/`expert` are a preview id because Google has no GA Pro
# newer than the closed one.
PROVIDER_MODELS: dict[str, dict[str, str]] = {
    "anthropic_google": {
        # Gemini Flash-Lite for the cheap high-volume vision work
        # (classification/style); Anthropic for the document-reasoning pipeline
        # (transcription → labeling/value-extraction → schema generation).
        "light": "gemini/gemini-3.5-flash-lite",
        "standard": "anthropic/claude-haiku-5-5",
        "advanced": "anthropic/claude-sonnet-5-5",
        "expert": "anthropic/claude-opus-5-5",
    },
    "anthropic": {
        # `advanced` (labeling, value extraction) and `expert` (schema
        # generation) carry the reasoning-heavy work, so they track the current
        # Sonnet/Opus generation. Both allow 128K output tokens, which matters
        # for value extraction on charge-heavy documents (see
        # grounded._DEFAULT_MAX_COMPLETION_TOKENS). Neither accepts a forced
        # tool_choice or `thinking: disabled`; `llm._build_completion_kwargs`
        # sends the nearest request each accepts.
        # `light`/`standard` run on Haiku 5.5, which also allows 128K output
        # and does the cheap classification and transcription work.
        "light": "anthropic/claude-haiku-5-5",
        "standard": "anthropic/claude-haiku-5-5",
        "advanced": "anthropic/claude-sonnet-5-5",
        "expert": "anthropic/claude-opus-5-5",
    },
    "google": {
        "light": "gemini/gemini-3.5-flash-lite",
        "standard": "gemini/gemini-3.8-flash",
        "advanced": "gemini/gemini-3.1-pro-preview",
        "expert": "gemini/gemini-3.1-pro-preview",
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

# Provider → the litellm model-id prefixes it serves. `[models] <provider>_api_key`
# is written onto the keyless models whose id carries a prefix listed here (see
# `apply_provider_keys`). `anthropic_google` is a family that mixes two
# of these, not a provider, so it has no key of its own.
PROVIDERS: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "google": ("gemini",),
    "openai": ("openai",),
}

# Per family, the API-key env var(s) its tiers need at runtime (the standard
# names litellm reads). Same keys as PROVIDER_MODELS — a test pins that.
PROVIDER_API_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic_google": ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GEMINI_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}
