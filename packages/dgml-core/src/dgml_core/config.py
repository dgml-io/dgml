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

"""Layered configuration resolution, built on ``pydantic-settings``.

The workspace config is assembled by merging, in increasing precedence:

1. built-in defaults ............. the dataclass field defaults in each loader
   (each section defaults to an empty table here — nothing to overlay)
2. user config ................... ``$XDG_CONFIG_HOME/dgml/config.toml`` when
   ``$XDG_CONFIG_HOME`` is set, else ``%APPDATA%\\dgml\\config.toml`` on Windows,
   else ``~/.config/dgml/config.toml`` (see
   :func:`dgml_core.storage.user_config_path`)
3. workspace config .............. ``<workspace>/config.toml`` (optional)
4. environment variables ......... ``DGML_``-prefixed, ``__`` nesting
5. CLI flags ..................... an overlay dict passed by the caller

The merge is pydantic-settings' source combination (``deep_update`` across
sources): each config section is modelled as a permissive ``dict`` field, so a
higher-priority layer overrides only the keys it sets and inherits the rest.
The env-var convention is the pydantic-settings standard — ``env_prefix="DGML_"``
with ``env_nested_delimiter="__"`` — so ``DGML_MODELS__ADVANCED=…`` sets
``models.advanced``. (``DGML_HOME`` / ``DGML_DEBUG`` have no ``__`` and match no
declared field, so ``extra="ignore"`` drops them — they never reach the config.)

This layer produces a plain ``dict``; validation, defaults, and the stable
``*_CONFIG_INVALID`` error codes remain in the per-section dataclass loaders.
The returned mapping holds exactly the sections some layer actually *mentioned*
(``exclude_unset``) — including one written with no keys, which arrives as an
empty table. Only a section no layer names at all is omitted. Loaders must
therefore treat ``{}`` and "absent" alike unless they mean to distinguish them
(see :func:`dgml_core.ocr.load_ocr_config`, which does not).
"""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from pydantic_settings import (
    BaseSettings,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from .errors import CorruptMetadata, LegacyConfigPresent
from .models_config import ConfigSection
from .storage import user_config_path

if TYPE_CHECKING:
    from pathlib import Path

    from .storage import Workspace

ENV_PREFIX = "DGML_"
ENV_NESTED_DELIMITER = "__"


def _build_settings_class(user_path: Path, ws_config: dict[str, Any] | None) -> type[BaseSettings]:
    """A ``BaseSettings`` subclass sourcing layer 2 from the TOML file at *user_path*
    and layer 3 from *ws_config*, the workspace's already-parsed config (``None`` when
    it has none). Defined per call so the classmethod source hook closes over the
    runtime values — no module-level state, so it stays thread-safe.

    Layer 3 arrives **parsed** rather than as a path because a workspace's config is not
    necessarily a file: one held in the machine's store of workspaces may be a row in a
    shared database. ``TomlConfigSettingsSource`` only reads paths, but it is itself an
    ``InitSettingsSource`` subclass, so handing the parsed mapping to that base gives an
    identical result — sources are combined by ``deep_update`` over what each returns,
    so precedence and the deep merge are unchanged."""

    class _DgmlSettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix=ENV_PREFIX,
            env_nested_delimiter=ENV_NESTED_DELIMITER,
            extra="ignore",
        )

        # Each config section is a permissive table; validation lives in the
        # per-section dataclass loaders that consume the merged mapping.
        models: dict[str, Any] = {}
        generation: dict[str, Any] = {}
        grounded: dict[str, Any] = {}
        classification: dict[str, Any] = {}
        ocr: dict[str, Any] = {}
        style: dict[str, Any] = {}
        text_extraction: dict[str, Any] = {}
        conversion: dict[str, Any] = {}
        clustering: dict[str, Any] = {}
        storage: dict[str, Any] = {}

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            # Highest precedence first: CLI-flag overlay (init) > env vars >
            # workspace config > user file. (TomlConfigSettingsSource returns an
            # empty mapping for a missing file; the workspace layer gets an empty
            # mapping when the workspace has no config, which is the same thing.)
            return (
                init_settings,
                env_settings,
                InitSettingsSource(settings_cls, ws_config or {}),
                TomlConfigSettingsSource(settings_cls, toml_file=user_path),
            )

    return _DgmlSettings


def load_merged_config(
    workspace: Workspace, *, cli_overrides: dict[str, Any] | None = None
) -> dict[ConfigSection, Any]:
    """Return the fully merged config mapping every section loader consumes.

    Merges, in increasing precedence: built-in defaults → user
    ``~/.config/dgml/config.toml`` → workspace ``<workspace>/config.toml`` →
    ``DGML_`` env vars → ``cli_overrides`` (highest). A section no layer mentions
    is omitted; a section written with no keys is kept as an empty table. Keys are
    :class:`~dgml_core.models_config.ConfigSection` members (every declared field
    name is a section), so loaders index the result with the enum rather than a
    bare string.
    """
    user_path = user_config_path()
    if not user_path.exists() and workspace.has_legacy_json_config():
        raise LegacyConfigPresent(
            f"{workspace.root / 'config.json'} uses the old JSON config format, which is "
            "no longer supported. Config is now TOML — run 'dgml init' to write "
            f"{user_path}, then migrate any settings from the old file."
        )

    try:
        # Parsed here rather than inside the settings class so an unparseable workspace
        # config raises the same CorruptMetadata as an unparseable user config, whichever
        # backing it came from.
        ws_text = workspace.config_text
        ws_config = tomllib.loads(ws_text) if ws_text else None
        settings_cls = _build_settings_class(user_path, ws_config)
        settings = settings_cls(**(cli_overrides or {}))
    except tomllib.TOMLDecodeError as exc:
        raise CorruptMetadata(f"invalid TOML in a config file: {exc}") from exc
    except ValidationError as exc:
        # A section set to a non-table (e.g. `generation = "haiku"`) — a
        # malformed config, surfaced uniformly like a TOML parse error.
        raise CorruptMetadata(f"malformed config: {exc}") from exc
    dump: dict[str, Any] = settings.model_dump(exclude_unset=True)
    # Field names are exactly the section names, so every key is a ConfigSection.
    return {ConfigSection(section): value for section, value in dump.items()}
