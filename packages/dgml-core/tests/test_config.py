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
import logging
from collections.abc import Callable
from pathlib import Path

import pytest
from dgml_core.config import load_merged_config
from dgml_core.errors import (
    CorruptMetadata,
    GenerationConfigInvalid,
    LegacyConfigPresent,
    ModelsConfigInvalid,
    StyleConfigInvalid,
    TextExtractionConfigInvalid,
)
from dgml_core.models_config import ConfigSection, load_models_config
from dgml_core.storage import Workspace, user_config_path
from dgml_core.style_config import load_style_config
from dgml_core.text_extraction_config import load_text_extraction_config

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


def test_cli_overrides_go_through_the_settings_model(workspace: Workspace) -> None:
    """An override is a layer like the others: an undeclared section is dropped and a
    non-table is malformed config, not a bare `ValueError` or a `str` in the mapping."""
    assert load_merged_config(workspace, cli_overrides={"other": {"k": 1}}) == {}
    with pytest.raises(CorruptMetadata):
        load_merged_config(workspace, cli_overrides={"generation": "haiku"})


# ---- Cost ---------------------------------------------------------------------


def test_merge_reads_the_workspace_config_once(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A loader call is one merge, not a `Workspace.config` derivation (which also
    reads identity and resolves storage, several reads of the same file)."""
    reads = 0
    original: property = vars(Workspace)["config_text"]

    def counting(self: Workspace) -> str | None:
        nonlocal reads
        reads += 1
        text: str | None = original.__get__(self, Workspace)
        return text

    monkeypatch.setattr(Workspace, "config_text", property(counting))
    write_config(workspace, {"models": {"light": "ws/model"}})
    load_merged_config(workspace)
    assert reads == 1


def test_merge_does_not_resolve_storage(workspace: Workspace) -> None:
    """A storage binding only `Workspace.open` / `store_configs` can reject must not
    fail a loader that never touches storage."""
    workspace.config_path.write_text(
        '[workspace]\nstorage_service = "other"\n[storage]\nprovider = "x:Y"\n'
        '[models]\nlight = "ws/model"\n',
        encoding="utf-8",
    )
    assert load_merged_config(workspace)[ConfigSection.MODELS] == {"light": "ws/model"}


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


def test_unknown_family_leaves_its_stray_tier_key_to_the_family_error(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    """A tier key beside a misspelled family is the family's mistake, not the key's: it
    is neither dropped nor warned about, and the family error says so when the tiers
    are read."""
    _write_user_config({"models": {"family": "antropic", "light_api_key": "k"}})
    merged = load_merged_config(workspace)
    assert merged[ConfigSection.MODELS] == {"family": "antropic", "light_api_key": "k"}
    with pytest.raises(ModelsConfigInvalid, match=r"got 'antropic'"):
        load_models_config(merged)
    assert "light_api_key" not in caplog.text


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


# ---- Model groups are atomic across layers ----------------------------------


def test_a_layer_that_sets_a_tier_replaces_its_credentials_too(workspace: Workspace) -> None:
    """The user config's key was written beside `anthropic/x`; a workspace that sets
    `advanced` to another model must not inherit it — that key would go to OpenAI."""
    _write_user_config({"models": {"advanced": "anthropic/x", "advanced_api_key": "sk-ant"}})
    write_config(workspace, {"models": {"advanced": "openai/y"}})
    assert load_merged_config(workspace)[ConfigSection.MODELS] == {"advanced": "openai/y"}


def test_a_layer_that_sets_a_task_model_replaces_its_credentials_too(workspace: Workspace) -> None:
    _write_user_config({"generation": {"model": "anthropic/x", "api_key_env": "ANT"}})
    write_config(workspace, {"generation": {"model": "openai/y"}})
    assert load_merged_config(workspace)[ConfigSection.GENERATION] == {"model": "openai/y"}


def test_a_layer_that_leaves_the_model_alone_keeps_its_credentials(workspace: Workspace) -> None:
    """The counterpart: only a layer that sets the *model* replaces the group."""
    _write_user_config({"models": {"advanced": "anthropic/x", "advanced_api_key": "sk-ant"}})
    write_config(workspace, {"models": {"expert": "anthropic/z"}})
    merged = load_merged_config(workspace)[ConfigSection.MODELS]
    assert merged["advanced"] == "anthropic/x"
    assert merged["advanced_api_key"] == "sk-ant"


def test_provider_key_lands_on_its_own_layers_family_tiers(workspace: Workspace) -> None:
    _write_user_config({"models": {"family": "anthropic", "anthropic_api_key_env": "ANT"}})
    merged = load_merged_config(workspace)[ConfigSection.MODELS]
    assert all(
        merged[f"{tier}_api_key_env"] == "ANT"
        for tier in ("light", "standard", "advanced", "expert")
    )


_SAME_LAYER = "is set, but not in the same config layer as"
_STRAY_TIER = rf"'models\.advanced_api_key' {_SAME_LAYER} 'models\.advanced'"
_STRAY_SECTION = rf"'generation\.api_key_env' {_SAME_LAYER} 'generation\.model'"


def test_a_tier_credential_without_its_tier_in_the_layer_fails_when_the_tiers_are_read(
    workspace: Workspace,
) -> None:
    """A key in a layer that does not set its model is refused by the tasks that read
    the tiers, not by the merge. The user layer's `family` sets `advanced`, but the
    workspace key would attach to whatever that tier resolves to — and follow it if
    the user layer changed — so the workspace layer owns that tier as unset."""
    _write_user_config({"models": {"family": "anthropic"}})
    write_config(workspace, {"models": {"advanced_api_key": "k-ws"}})
    merged = load_merged_config(workspace)
    assert "advanced" not in merged[ConfigSection.MODELS]
    with pytest.raises(ModelsConfigInvalid, match=_STRAY_TIER) as info:
        load_models_config(merged)
    assert "in the same config layer" in str(info.value)


def test_a_key_cannot_outlive_a_model_changed_underneath_it(workspace: Workspace) -> None:
    """The leak the rule exists for: the user layer's model moved to another provider
    under a workspace layer holding only a key. Refused, not sent."""
    _write_user_config({"models": {"advanced": "openai/y"}})
    write_config(workspace, {"models": {"advanced_api_key": "sk-ant"}})
    merged = load_merged_config(workspace)
    assert merged[ConfigSection.MODELS] == {"advanced_api_key": "sk-ant"}
    with pytest.raises(ModelsConfigInvalid, match=_STRAY_TIER):
        load_models_config(merged)


def test_a_section_credential_without_its_model_in_the_layer_fails_that_task(
    workspace: Workspace,
) -> None:
    """The stray key takes the model with it, so generation — and only generation —
    is refused, with the key named."""
    from dgml_core.generation import load_generation_config

    _write_user_config({"generation": {"model": "anthropic/x", "label_model": "anthropic/l"}})
    write_config(workspace, {"generation": {"api_key_env": "ANT"}})
    merged = load_merged_config(workspace)[ConfigSection.GENERATION]
    assert merged == {"api_key_env": "ANT", "label_model": "anthropic/l"}
    with pytest.raises(GenerationConfigInvalid, match=_STRAY_SECTION):
        load_generation_config(workspace)


@pytest.mark.parametrize("where", ["user", "env"])
def test_a_stray_credential_fails_at_the_loader_not_the_merge(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """Every command shares the merge, so a stray key in the user config or the
    environment must not take `file add` or `file list` down with it; the task it
    was for fails."""
    from dgml_core.generation import load_generation_config
    from dgml_core.pages import load_pdf_config

    if where == "user":
        _write_user_config({"models": {"family": "anthropic"}, "generation": {"api_key_env": "K"}})
    else:
        write_config(workspace, {"models": {"family": "anthropic"}})
        monkeypatch.setenv("DGML_GENERATION__API_KEY_ENV", "K")
    assert load_merged_config(workspace)[ConfigSection.GENERATION] == {"api_key_env": "K"}
    assert load_pdf_config(workspace) is not None
    with pytest.raises(GenerationConfigInvalid, match=_STRAY_SECTION):
        load_generation_config(workspace)


def test_a_null_model_beside_a_key_is_a_stray_key(workspace: Workspace, tmp_path: Path) -> None:
    """`"model": null` in a `--generation-config` file is no model, so the key beside
    it is refused like any other stray — not paired with the tier's model."""
    from dgml_core.generation import resolve_generation_config

    write_config(workspace, {"models": {"family": "anthropic"}})
    path = tmp_path / "gen.json"
    path.write_text(json.dumps({"model": None, "api_key_env": "Z"}), encoding="utf-8")
    with pytest.raises(GenerationConfigInvalid, match=_STRAY_SECTION):
        resolve_generation_config(workspace, config=str(path))


def test_a_null_model_in_a_higher_layer_unsets_the_lower_layers_group_too(
    workspace: Workspace, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A layer that sets a model key to null owns that group like one that sets a
    model: the lower layer's credentials go with the model, so the task falls back to
    its tier instead of tripping over a key with no model."""
    from dgml_core.generation import resolve_generation_config

    _write_user_config(
        {"models": {"family": "anthropic", "light": "anthropic/l", "light_api_key_env": "LK"}}
    )
    write_config(workspace, {"generation": {"model": "anthropic/x", "api_key_env": "K"}})
    merged = load_merged_config(
        workspace, cli_overrides={"generation": {"model": None}, "models": {"light": None}}
    )
    assert merged[ConfigSection.GENERATION] == {}
    models = merged[ConfigSection.MODELS]
    assert "light" not in models and "light_api_key_env" not in models and "standard" in models
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    # The real override path: a `--generation-config` file with `"model": null`.
    path = tmp_path / "gen.json"
    path.write_text(json.dumps({"model": None}), encoding="utf-8")
    cfg, _ = resolve_generation_config(workspace, config=str(path))
    assert cfg.model != "anthropic/x" and cfg.model.startswith("anthropic/")
    assert cfg.api_key_env is None


_OPT_IN = [
    (ConfigSection.STYLE, load_style_config, StyleConfigInvalid),
    (ConfigSection.TEXT_EXTRACTION, load_text_extraction_config, TextExtractionConfigInvalid),
]


@pytest.mark.parametrize(("section", "load", "invalid"), _OPT_IN)
@pytest.mark.parametrize("enabled", [False, None, True])
def test_a_stray_credential_in_an_opt_in_section_fails_only_when_the_section_is_on(
    workspace: Workspace,
    section: ConfigSection,
    load: Callable[[Workspace], object],
    invalid: type[Exception],
    enabled: bool | None,
) -> None:
    """Same rule as every section, and the same laziness: a disabled section never
    resolves its model, so it loads as nothing; enabled, the stray key is refused."""
    table: dict[str, object] = {"api_key_env": "K"}
    if enabled is not None:
        table["enabled"] = enabled
    write_config(workspace, {"models": {"family": "anthropic"}, section: table})
    assert load_merged_config(workspace)[section].get("api_key_env") == "K"
    if enabled is not True:
        assert load(workspace) is None
    else:
        with pytest.raises(
            invalid, match=rf"'{section}\.api_key_env' {_SAME_LAYER} '{section}\.model'"
        ):
            load(workspace)


@pytest.mark.parametrize(("section", "load", "invalid"), _OPT_IN)
def test_a_stray_credential_is_judged_by_its_own_layer_not_the_merged_section(
    workspace: Workspace,
    section: ConfigSection,
    load: Callable[[Workspace], object],
    invalid: type[Exception],
) -> None:
    """Another layer enabling the section does not rescue the key: it still has no
    model in its own layer, so it can never pair with the tier's model."""
    _write_user_config({section: {"enabled": True}})
    write_config(workspace, {"models": {"family": "anthropic"}, section: {"api_key_env": "K"}})
    assert load_merged_config(workspace)[section] == {"enabled": True, "api_key_env": "K"}
    with pytest.raises(invalid, match=rf"'{section}\.api_key_env' {_SAME_LAYER}"):
        load(workspace)


def test_a_provider_key_in_the_environment_reaches_models_set_in_files(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A secret injected through the environment (CI) serves the models config.toml
    sets, without restating the family there — which would replace pinned tiers."""
    monkeypatch.setenv("DGML_MODELS__ANTHROPIC_API_KEY_ENV", "ANT")
    write_config(workspace, {"models": {"family": "anthropic", "advanced": "openai/gpt-5"}})
    merged = load_merged_config(workspace)[ConfigSection.MODELS]
    assert merged["standard_api_key_env"] == "ANT"
    assert merged["advanced"] == "openai/gpt-5" and "advanced_api_key_env" not in merged


def test_one_provider_key_form_per_layer_is_not_a_clash(workspace: Workspace) -> None:
    """A provider's key pair merges as one group: the workspace's env-var form
    replaces the user config's literal whole, so the two never collide."""
    _write_user_config({"models": {"family": "anthropic", "anthropic_api_key": "ant"}})
    write_config(workspace, {"models": {"anthropic_api_key_env": "ANT"}})
    merged = load_merged_config(workspace)[ConfigSection.MODELS]
    assert merged["anthropic_api_key_env"] == "ANT" and "anthropic_api_key" not in merged
    assert merged["advanced_api_key_env"] == "ANT" and "advanced_api_key" not in merged


def test_provider_key_lands_on_its_own_layers_section_model(workspace: Workspace) -> None:
    write_config(
        workspace,
        {"models": {"anthropic_api_key_env": "ANT"}, "generation": {"model": "anthropic/w"}},
    )
    generation = load_merged_config(workspace)[ConfigSection.GENERATION]
    assert generation == {"model": "anthropic/w", "api_key_env": "ANT"}


def test_the_workspaces_provider_key_wins_over_the_user_configs(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    """A personal key in the user config and a team key in the workspace: the team
    key serves every keyless model — tiers and per-call overrides alike — so one run
    never splits across accounts."""
    from dgml_core.generation import resolve_generation_config

    _write_user_config({"models": {"family": "anthropic", "anthropic_api_key_env": "PERSONAL"}})
    write_config(workspace, {"models": {"anthropic_api_key_env": "TEAM"}})
    assert load_merged_config(workspace)[ConfigSection.MODELS]["standard_api_key_env"] == "TEAM"
    cfg, _ = resolve_generation_config(workspace, model="anthropic/claude-haiku-4-5")
    assert (cfg.api_key_env, cfg.label_api_key_env) == ("TEAM", "TEAM")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize(
    "user",
    [
        # Keys for a provider the config does not use.
        {"family": "anthropic", "anthropic_api_key": "ant", "openai_api_key": "oai"},
        # A lower layer's key, replaced by the workspace's.
        {"anthropic_api_key_env": "ANT"},
    ],
)
def test_provider_key_with_no_keyless_model_of_its_provider_is_quiet(
    workspace: Workspace, caplog: pytest.LogCaptureFixture, user: dict[str, str]
) -> None:
    _write_user_config({"models": user})
    write_config(workspace, {"models": {"family": "anthropic", "anthropic_api_key": "ws"}})
    load_merged_config(workspace)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_provider_key_skips_a_model_behind_a_custom_endpoint_and_says_so(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    """The configured key is never sent to an `api_base`, but a keyless model there
    makes litellm send its own env-var key instead — so that is said, by the task
    that resolves the model, not by the merge every command runs."""
    from dgml_core.classification import load_classification_config

    _write_user_config({"models": {"openai_api_key": "sk-real"}})
    write_config(
        workspace,
        {"classification": {"model": "openai/llama3", "api_base": "http://vllm:8000"}},
    )
    assert load_merged_config(workspace)[ConfigSection.CLASSIFICATION] == {
        "model": "openai/llama3",
        "api_base": "http://vllm:8000",
    }
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    load_classification_config(workspace)
    assert "is not sent to classification.model: a provider key never goes" in caplog.text


def test_same_layer_tier_at_an_api_base_is_warned_too(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    _write_user_config(
        {
            "models": {
                "family": "anthropic",
                "anthropic_api_key_env": "K",
                "standard_api_base": "http://proxy",
            }
        }
    )
    from dgml_core.generation import load_generation_config

    merged = load_merged_config(workspace)[ConfigSection.MODELS]
    assert "standard_api_key_env" not in merged and merged["advanced_api_key_env"] == "K"
    load_generation_config(workspace)  # transcription runs on the standard tier
    assert "is not sent to models.standard" in caplog.text


def test_a_malformed_provider_key_fails_the_tasks_that_would_take_it(
    workspace: Workspace,
) -> None:
    """An empty `anthropic_api_key` is this block's error, reported where the key would
    have been used — by the tiers, and by a keyless `anthropic/` section model — and
    nowhere else: the merge, `file add`, and an `openai/` section model are unaffected."""
    from dgml_core.generation import load_generation_config
    from dgml_core.pages import load_pdf_config

    _write_user_config({"models": {"family": "anthropic", "anthropic_api_key": ""}})
    merged = load_merged_config(workspace)
    assert "standard_api_key" not in merged[ConfigSection.MODELS]  # not applied, not dropped
    assert load_pdf_config(workspace) is not None
    with pytest.raises(
        ModelsConfigInvalid, match=r"'models\.anthropic_api_key' must be a non-empty"
    ):
        load_generation_config(workspace)
    write_config(workspace, {"generation": {"model": "anthropic/x", "label_model": "anthropic/l"}})
    with pytest.raises(
        ModelsConfigInvalid, match=r"'models\.anthropic_api_key' must be a non-empty"
    ):
        load_generation_config(workspace)
    write_config(workspace, {"generation": {"model": "openai/x", "label_model": "openai/l"}})
    assert load_generation_config(workspace).model == "openai/x"
