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

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dgml_core.errors import (
    AuthError,
    CorruptMetadata,
    GenerationConfigInvalid,
    GenerationConfigMissing,
)
from dgml_core.generation import (
    GENERATION_PROFILES,
    GenerationConfig,
    load_generation_config,
    load_generation_config_file,
    load_generation_profile,
    resolve_generation_api_key,
    resolve_generation_config,
    resolve_generation_label_api_key,
)
from dgml_core.storage import Workspace

from .conftest import dump_toml, write_config

MODEL = "anthropic/claude-haiku-4-5"
LABEL_MODEL = "anthropic/claude-sonnet-4-6"


def _write(workspace: Workspace, section: dict[str, object]) -> None:
    write_config(workspace, {"generation": section})


# ---------------------------------------------------------------------------
# load_generation_config
# ---------------------------------------------------------------------------


def test_missing_when_no_config_and_no_models(workspace: Workspace) -> None:
    with pytest.raises(GenerationConfigMissing):
        load_generation_config(workspace)


def test_missing_when_no_generation_section_and_no_models(workspace: Workspace) -> None:
    workspace.config_path.write_text("[ocr]\n", encoding="utf-8")
    with pytest.raises(GenerationConfigMissing):
        load_generation_config(workspace)


def test_corrupt_when_malformed_toml(workspace: Workspace) -> None:
    workspace.config_path.write_text("not = valid = toml", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        load_generation_config(workspace)


def test_invalid_when_section_not_object(workspace: Workspace) -> None:
    # A section set to a scalar is a malformed config → CorruptMetadata.
    workspace.config_path.write_text('generation = "haiku"\n', encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        load_generation_config(workspace)


def test_missing_model_without_override_or_tier(workspace: Workspace) -> None:
    # label_model set on the section, transcription model neither set nor tiered.
    _write(workspace, {"label_model": LABEL_MODEL})
    with pytest.raises(GenerationConfigMissing):
        load_generation_config(workspace)


def test_invalid_when_model_empty(workspace: Workspace) -> None:
    _write(workspace, {"model": "   ", "label_model": LABEL_MODEL})
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config(workspace)


def test_missing_label_without_override_or_tier(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL})
    with pytest.raises(GenerationConfigMissing):
        load_generation_config(workspace)


def test_invalid_when_label_empty(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL, "label_model": "   "})
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config(workspace)


def test_minimal_explicit_config(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL})
    cfg = load_generation_config(workspace)
    assert cfg.model == MODEL
    assert cfg.label_model == LABEL_MODEL
    assert cfg.api_key is None
    assert cfg.api_base is None
    assert cfg.label_api_key is None


def test_models_resolve_from_tiers(workspace: Workspace) -> None:
    # No [generation] section: transcription ← standard, labeling ← advanced.
    write_config(
        workspace,
        {"models": {"standard": MODEL, "advanced": LABEL_MODEL, "light": MODEL, "expert": MODEL}},
    )
    cfg = load_generation_config(workspace)
    assert cfg.model == MODEL  # standard
    assert cfg.label_model == LABEL_MODEL  # advanced


def test_section_model_overrides_tier(workspace: Workspace) -> None:
    write_config(
        workspace,
        {
            "models": {"standard": MODEL, "advanced": LABEL_MODEL},
            "generation": {"label_model": "openai/gpt-5"},
        },
    )
    cfg = load_generation_config(workspace)
    assert cfg.model == MODEL  # standard tier
    assert cfg.label_model == "openai/gpt-5"  # override wins


def test_tier_models_carry_no_credentials(workspace: Workspace) -> None:
    # [models] tiers name only models — no credentials — so a tier-sourced model
    # has no api_key/api_key_env/api_base (litellm uses its per-provider env var).
    write_config(workspace, {"models": {"standard": MODEL, "advanced": "gemini/gemini-2.5-pro"}})
    cfg = load_generation_config(workspace)
    assert (cfg.model, cfg.label_model) == (MODEL, "gemini/gemini-2.5-pro")
    assert (cfg.api_key, cfg.api_key_env, cfg.api_base) == (None, None, None)
    assert (cfg.label_api_key, cfg.label_api_key_env, cfg.label_api_base) == (None, None, None)


def test_per_task_credentials_apply_to_tier_models(workspace: Workspace) -> None:
    # Credentials live on the task section, independent per model, and apply even
    # when the models themselves come from the tiers.
    write_config(
        workspace,
        {
            "models": {"standard": MODEL, "advanced": "gemini/gemini-2.5-pro"},
            "generation": {
                "api_key_env": "MY_ANTH",
                "label_api_key_env": "MY_GEM",
            },
        },
    )
    cfg = load_generation_config(workspace)
    assert cfg.api_key_env == "MY_ANTH"  # transcription
    assert cfg.label_api_key_env == "MY_GEM"  # labeling


def test_env_var_overrides_config(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    write_config(workspace, {"models": {"standard": MODEL, "advanced": LABEL_MODEL}})
    monkeypatch.setenv("DGML_MODELS__ADVANCED", "openai/gpt-5")
    cfg = load_generation_config(workspace)
    assert cfg.model == MODEL  # standard, unchanged
    assert cfg.label_model == "openai/gpt-5"  # env overrides the advanced tier


def test_user_and_workspace_configs_deep_merge(workspace: Workspace) -> None:
    # The XDG-isolated user config sets all tiers; the workspace overrides one.
    from dgml_core.storage import user_config_path

    up = user_config_path()
    up.parent.mkdir(parents=True, exist_ok=True)
    up.write_text(
        dump_toml(
            {
                "models": {
                    "standard": MODEL,
                    "advanced": LABEL_MODEL,
                    "light": MODEL,
                    "expert": MODEL,
                }
            }
        ),
        encoding="utf-8",
    )
    write_config(workspace, {"models": {"advanced": "openai/gpt-5"}})
    cfg = load_generation_config(workspace)
    assert cfg.model == MODEL  # standard inherited from the user config
    assert cfg.label_model == "openai/gpt-5"  # advanced overridden by the workspace config


def test_full_config_round_trips(workspace: Workspace) -> None:
    _write(
        workspace,
        {
            "model": MODEL,
            "label_model": LABEL_MODEL,
            "api_key_env": "MY_KEY",
            "api_base": "http://localhost:11434",
            "label_api_key_env": "MY_LABEL_KEY",
        },
    )
    cfg = load_generation_config(workspace)
    assert cfg.model == MODEL
    assert cfg.label_model == LABEL_MODEL
    assert cfg.api_key_env == "MY_KEY"
    assert cfg.api_base == "http://localhost:11434"
    assert cfg.label_api_key_env == "MY_LABEL_KEY"


def test_invalid_when_api_key_and_env_both_set(workspace: Workspace) -> None:
    _write(
        workspace,
        {
            "model": MODEL,
            "label_model": LABEL_MODEL,
            "api_key": "sk-x",
            "api_key_env": "MY_KEY",
        },
    )
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config(workspace)


# ---------------------------------------------------------------------------
# resolve_generation_api_key / resolve_generation_label_api_key
# ---------------------------------------------------------------------------


def test_resolve_prefers_literal_key() -> None:
    cfg = GenerationConfig(model=MODEL, label_model=LABEL_MODEL, api_key="sk-literal")
    assert resolve_generation_api_key(cfg) == "sk-literal"


def test_resolve_none_when_unset() -> None:
    cfg = GenerationConfig(model=MODEL, label_model=LABEL_MODEL)
    assert resolve_generation_api_key(cfg) is None
    assert resolve_generation_label_api_key(cfg) is None


def test_resolve_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_GEN_KEY", "sk-from-env")
    monkeypatch.setenv("MY_LABEL_KEY", "sk-label-env")
    cfg = GenerationConfig(
        model=MODEL,
        label_model=LABEL_MODEL,
        api_key_env="MY_GEN_KEY",
        label_api_key_env="MY_LABEL_KEY",
    )
    assert resolve_generation_api_key(cfg) == "sk-from-env"
    assert resolve_generation_label_api_key(cfg) == "sk-label-env"


def test_resolve_raises_when_env_var_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_GEN_KEY", raising=False)
    cfg = GenerationConfig(model=MODEL, label_model=LABEL_MODEL, api_key_env="MISSING_GEN_KEY")
    with pytest.raises(AuthError):
        resolve_generation_api_key(cfg)


# ---------------------------------------------------------------------------
# bundled profiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GENERATION_PROFILES)
def test_bundled_profiles_are_valid(name: str) -> None:
    # Every bundled profile is a well-formed 'generation' overlay.
    section = load_generation_profile(name)
    assert isinstance(section.get("model"), str)
    assert isinstance(section.get("label_model"), str)


@pytest.mark.parametrize("name", GENERATION_PROFILES)
def test_bundled_profiles_fully_specify_models(workspace: Workspace, name: str) -> None:
    # Each profile names both models, so it resolves on a workspace with neither a
    # [generation] section nor [models] tiers — the point of --generation-config.
    cfg, source = resolve_generation_config(workspace, config=name)
    assert cfg.model
    assert cfg.label_model
    assert source == f"profile:{name}"


def test_unknown_profile_rejected() -> None:
    with pytest.raises(GenerationConfigInvalid):
        load_generation_profile("turbo")


# ---------------------------------------------------------------------------
# load_generation_config_file
# ---------------------------------------------------------------------------


def test_config_file_bare_section(tmp_path: Path) -> None:
    path = tmp_path / "gen.json"
    path.write_text(json.dumps({"model": MODEL, "label_model": LABEL_MODEL}), encoding="utf-8")
    assert load_generation_config_file(path) == {"model": MODEL, "label_model": LABEL_MODEL}


def test_config_file_unwraps_generation_key(tmp_path: Path) -> None:
    # A full-config file wrapping the section under "generation" is accepted.
    path = tmp_path / "gen.json"
    path.write_text(
        json.dumps({"generation": {"model": MODEL, "label_model": LABEL_MODEL}, "ocr": {}}),
        encoding="utf-8",
    )
    assert load_generation_config_file(path) == {"model": MODEL, "label_model": LABEL_MODEL}


def test_config_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config_file(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# resolve_generation_config
# ---------------------------------------------------------------------------


def test_resolve_no_flags_matches_load(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL})
    cfg, source = resolve_generation_config(workspace)
    assert cfg == GenerationConfig(model=MODEL, label_model=LABEL_MODEL)
    assert source == "config"


def test_resolve_no_flags_missing_config_raises(workspace: Workspace) -> None:
    # Preserves the load_generation_config missing-config behavior when nothing
    # is overridden.
    with pytest.raises(GenerationConfigMissing):
        resolve_generation_config(workspace)


def test_resolve_profile_replaces_config(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL})
    cfg, source = resolve_generation_config(workspace, config="fast")
    assert cfg.model == cfg.label_model  # fast uses the cheap tier for both
    assert source == "profile:fast"


def test_resolve_file_replaces_config(workspace: Workspace, tmp_path: Path) -> None:
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL})
    path = tmp_path / "gen.json"
    other = "anthropic/claude-opus-4-8"
    path.write_text(json.dumps({"model": other, "label_model": other}), encoding="utf-8")
    cfg, source = resolve_generation_config(workspace, config=str(path))
    assert cfg.model == other
    assert cfg.label_model == other
    assert source == "file"


def test_resolve_flags_override_config(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL})
    override = "anthropic/claude-opus-4-8"
    cfg, source = resolve_generation_config(workspace, model=override)
    assert cfg.model == override
    assert cfg.label_model == LABEL_MODEL  # untouched
    assert source == "override"


def test_resolve_flags_without_config_file(workspace: Workspace) -> None:
    # Both models supplied via flags — works with no config.json at all, which
    # is the point: run generate against an explicit model config without one.
    cfg, source = resolve_generation_config(workspace, model=MODEL, label_model=LABEL_MODEL)
    assert cfg == GenerationConfig(model=MODEL, label_model=LABEL_MODEL)
    assert source == "override"


def test_resolve_flags_over_profile(workspace: Workspace) -> None:
    override = "anthropic/claude-opus-4-8"
    cfg, source = resolve_generation_config(workspace, config="fast", label_model=override)
    assert cfg.label_model == override
    assert source == "profile:fast+override"


def test_resolve_partial_flags_without_config_incomplete(workspace: Workspace) -> None:
    # Only one model via flag, and nothing in the config or the [models] tiers to
    # resolve the other → still incomplete, rejected.
    with pytest.raises(GenerationConfigMissing):
        resolve_generation_config(workspace, model=MODEL)


def test_resolve_partial_flag_falls_back_to_tier(workspace: Workspace) -> None:
    # The overlay is the merged config's top layer, not a replacement: --model
    # alone leaves labeling to resolve from its [models] tier as usual.
    write_config(workspace, {"models": {"advanced": LABEL_MODEL}})
    cfg, source = resolve_generation_config(workspace, model=MODEL)
    assert cfg.model == MODEL
    assert cfg.label_model == LABEL_MODEL
    assert source == "override"


def test_resolve_overlay_preserves_unnamed_keys(workspace: Workspace) -> None:
    # A profile/file overrides the keys it names; a workspace credential the
    # overlay is silent about survives the run.
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL, "api_key_env": "MY_KEY"})
    cfg, source = resolve_generation_config(workspace, config="fast")
    assert cfg.label_model != LABEL_MODEL  # profile replaced the models...
    assert cfg.api_key_env == "MY_KEY"  # ...but not the credentials
    assert source == "profile:fast"
