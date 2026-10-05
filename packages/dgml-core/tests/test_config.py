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

"""Tests for the layered config merge itself (:func:`load_merged_config`).

Every section loader is built on this function, but until now it was only ever
exercised *through* those loaders — which is how a section written with no keys
came to be silently dropped from the merged mapping without any test noticing.
These tests pin the merge's own contract: which sections survive, in what
precedence order, and what a malformed file does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dgml_core.config import load_merged_config
from dgml_core.errors import CorruptMetadata, LegacyConfigPresent
from dgml_core.models_config import ConfigSection
from dgml_core.storage import Workspace, user_config_path

from .conftest import dump_toml, write_config


def _write_user_config(data: dict[str, object]) -> None:
    """Write the user-level config (resolution layer 2). Safe because the autouse
    `_isolate_user_config` fixture points XDG_CONFIG_HOME at a tmp dir."""
    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_toml(data) + "\n", encoding="utf-8")


# ---- What survives the merge ------------------------------------------------


def test_no_config_anywhere_yields_empty_mapping(workspace: Workspace) -> None:
    """A section no layer mentions stays absent. Guards the other direction of
    the `exclude_unset` switch: unset fields must not leak in as empty tables."""
    assert load_merged_config(workspace) == {}


def test_bare_workspace_section_is_preserved(workspace: Workspace) -> None:
    """A section written with no keys must survive the merge.

    Regression test: under `exclude_defaults` an all-default (i.e. keyless)
    section was dropped, so a loader could not tell "the user wrote [style]"
    from "the user never mentioned style".
    """
    workspace.config_path.write_text("[style]\n", encoding="utf-8")
    assert load_merged_config(workspace) == {ConfigSection.STYLE: {}}


def test_bare_user_section_is_preserved(workspace: Workspace) -> None:
    """Same, one layer up — the user config, with no workspace config at all."""
    _write_user_config({"style": {}})
    assert not workspace.config_path.exists()
    assert load_merged_config(workspace) == {ConfigSection.STYLE: {}}


def test_unknown_section_is_dropped(workspace: Workspace) -> None:
    """An undeclared section is not a field, so `extra="ignore"` drops it before
    it can reach `model_fields_set`. Distinct from a bare *known* section, and
    what the `[other]` fixtures in test_ocr.py rely on."""
    workspace.config_path.write_text("[other]\nkey = 1\n", encoding="utf-8")
    assert load_merged_config(workspace) == {}


def test_keys_are_config_section_members(workspace: Workspace) -> None:
    write_config(workspace, {"models": {"light": "gemini/gemini-2.5-flash-lite"}})
    merged = load_merged_config(workspace)
    assert set(merged) == {ConfigSection.MODELS}
    assert merged[ConfigSection.MODELS] == {"light": "gemini/gemini-2.5-flash-lite"}


# ---- Layering ---------------------------------------------------------------


def test_user_and_workspace_deep_merge(workspace: Workspace) -> None:
    """The workspace layer overrides only the keys it sets."""
    _write_user_config({"style": {"model": "user/model", "max_tokens": 9}})
    write_config(workspace, {"style": {"model": "ws/model"}})
    assert load_merged_config(workspace)[ConfigSection.STYLE] == {
        "model": "ws/model",
        "max_tokens": 9,
    }


def test_bare_workspace_section_does_not_clobber_user_keys(workspace: Workspace) -> None:
    """A bare section is presence, not an erasure — it must not blank out keys
    the lower layer set."""
    _write_user_config({"style": {"model": "user/model"}})
    workspace.config_path.write_text("[style]\n", encoding="utf-8")
    assert load_merged_config(workspace)[ConfigSection.STYLE] == {"model": "user/model"}


def test_env_var_creates_section(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    """An env var alone is enough to make a section present."""
    monkeypatch.setenv("DGML_STYLE__MODEL", "env/model")
    assert load_merged_config(workspace)[ConfigSection.STYLE] == {"model": "env/model"}


def test_env_var_overrides_workspace(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    write_config(workspace, {"models": {"advanced": "ws/model"}})
    monkeypatch.setenv("DGML_MODELS__ADVANCED", "env/model")
    assert load_merged_config(workspace)[ConfigSection.MODELS]["advanced"] == "env/model"


def test_cli_overrides_take_precedence(workspace: Workspace) -> None:
    """`cli_overrides` is the highest layer (and otherwise uncovered)."""
    write_config(workspace, {"models": {"light": "ws/model"}})
    merged = load_merged_config(workspace, cli_overrides={"models": {"light": "cli/model"}})
    assert merged[ConfigSection.MODELS]["light"] == "cli/model"


_ALL_TIERS = {"light": "u/l", "standard": "u/s", "advanced": "u/a", "expert": "u/e"}


def _family(name: str, **tiers: str) -> dict[str, str]:
    """A family's expanded [models] table, with *tiers* overriding."""
    from dgml_core.default_config import PROVIDER_MODELS

    return {**PROVIDER_MODELS[name], "family": name, **tiers}


def test_workspace_family_replaces_user_tiers(workspace: Workspace) -> None:
    """A user config with all four tiers (what the old `dgml init` wrote) must not
    silently defeat a higher layer's family."""
    _write_user_config({"models": _ALL_TIERS})
    write_config(workspace, {"models": {"family": "openai"}})
    assert load_merged_config(workspace)[ConfigSection.MODELS] == _family("openai")


def test_env_family_replaces_lower_tiers(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_user_config({"models": _ALL_TIERS})
    write_config(workspace, {"models": {"expert": "ws/e"}})
    monkeypatch.setenv("DGML_MODELS__FAMILY", "openai")
    assert load_merged_config(workspace)[ConfigSection.MODELS] == _family("openai")


def test_higher_family_drops_lower_pins_too(workspace: Workspace) -> None:
    """Switching family in a higher layer replaces the whole set, including tiers
    the lower layer pinned next to its own family."""
    _write_user_config({"models": {"family": "anthropic", "advanced": "u/a"}})
    write_config(workspace, {"models": {"family": "openai"}})
    assert load_merged_config(workspace)[ConfigSection.MODELS] == _family("openai")


def test_tiers_beside_or_above_family_still_win(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_user_config({"models": {"light": "u/l"}})
    write_config(workspace, {"models": {"family": "openai", "advanced": "ws/a"}})
    monkeypatch.setenv("DGML_MODELS__EXPERT", "env/e")
    assert load_merged_config(workspace)[ConfigSection.MODELS] == _family(
        "openai", advanced="ws/a", expert="env/e"
    )


def test_lower_family_fills_tiers_a_higher_layer_leaves_unset(workspace: Workspace) -> None:
    _write_user_config({"models": {"family": "google"}})
    write_config(workspace, {"models": {"advanced": "ws/a"}})
    assert load_merged_config(workspace)[ConfigSection.MODELS] == _family("google", advanced="ws/a")


def test_unknown_family_is_not_expanded(workspace: Workspace) -> None:
    """Left for `load_models_config` to reject rather than failing the merge."""
    write_config(workspace, {"models": {"family": "nope"}})
    assert load_merged_config(workspace)[ConfigSection.MODELS] == {"family": "nope"}


def test_env_var_sets_family(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DGML_MODELS__FAMILY", "openai")
    assert load_merged_config(workspace)[ConfigSection.MODELS]["family"] == "openai"


# ---- Malformed input --------------------------------------------------------


def test_malformed_toml_raises_corrupt_metadata(workspace: Workspace) -> None:
    workspace.config_path.write_text("{ not valid toml", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        load_merged_config(workspace)


def test_non_table_section_raises_corrupt_metadata(workspace: Workspace) -> None:
    """A section set to a scalar fails validation during construction, before
    either dump mode is reached."""
    workspace.config_path.write_text('generation = "haiku"\n', encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        load_merged_config(workspace)


def test_legacy_json_config_raises(workspace: Workspace) -> None:
    """A pre-migration config.json with no config.toml and no user config is a
    hard error naming the upgrade path."""
    (workspace.root / "config.json").write_text(json.dumps({"style": {}}), encoding="utf-8")
    assert not workspace.config_path.exists()
    assert not user_config_path().exists()
    with pytest.raises(LegacyConfigPresent, match="no longer supported"):
        load_merged_config(workspace)


def test_legacy_json_ignored_once_user_config_exists(workspace: Workspace) -> None:
    """The upgrade error is keyed on having no user config at all; once one
    exists the stale config.json is simply not read."""
    (workspace.root / "config.json").write_text(json.dumps({"style": {}}), encoding="utf-8")
    _write_user_config({"models": {"light": "user/model"}})
    assert load_merged_config(workspace) == {ConfigSection.MODELS: {"light": "user/model"}}


# ---- Sections that are indifferent between "absent" and "empty" -------------


@pytest.mark.parametrize(
    "section", ["models", "clustering", "conversion", "generation", "grounded"]
)
def test_bare_non_switch_section_reaches_loaders_as_empty(
    workspace: Workspace, section: str
) -> None:
    """Bare non-switch sections now arrive as `{}` rather than being dropped.

    Their loaders normalize `{}` and `None` to the same thing, so this is inert —
    pinned here so a future loader that starts distinguishing them has to do so
    deliberately. (`ocr` is the exception and has its own tests.)
    """
    Path(workspace.config_path).write_text(f"[{section}]\n", encoding="utf-8")
    assert load_merged_config(workspace) == {ConfigSection(section): {}}
