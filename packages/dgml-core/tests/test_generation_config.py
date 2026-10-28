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

MODEL = "anthropic/claude-haiku-5-5"
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
    # No [generation] section: BOTH tasks fall back to the standard tier.
    # Labeling used to take `advanced`; measurement did not support paying for
    # it, so the fallback was moved down (see GenerationConfig's docstring).
    write_config(
        workspace,
        {"models": {"standard": MODEL, "advanced": LABEL_MODEL, "light": MODEL, "expert": MODEL}},
    )
    cfg = load_generation_config(workspace)
    assert cfg.model == MODEL  # standard
    assert cfg.label_model == MODEL  # standard too — NOT advanced


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
    assert (cfg.model, cfg.label_model) == (MODEL, MODEL)  # both from standard
    assert (cfg.api_key, cfg.api_key_env, cfg.api_base) == (None, None, None)
    assert (cfg.label_api_key, cfg.label_api_key_env, cfg.label_api_base) == (None, None, None)


def test_section_credentials_without_a_section_model_are_rejected(workspace: Workspace) -> None:
    # A key is written beside the model it is for. Credentials on [generation] with
    # no [generation] model would pair with a tier's model — a group split across
    # two tables — so they are refused when generation runs. Name the model here, or
    # move the key to [models] as `standard_api_key_env` / `advanced_api_key_env`.
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
    with pytest.raises(GenerationConfigInvalid, match=r"same config layer as 'generation\.model'"):
        load_generation_config(workspace)


def test_per_task_credentials_sit_beside_their_models(workspace: Workspace) -> None:
    write_config(
        workspace,
        {
            "generation": {
                "model": MODEL,
                "api_key_env": "MY_ANTH",
                "label_model": "gemini/gemini-2.5-pro",
                "label_api_key_env": "MY_GEM",
            },
        },
    )
    cfg = load_generation_config(workspace)
    assert cfg.api_key_env == "MY_ANTH"  # transcription
    assert cfg.label_api_key_env == "MY_GEM"  # labeling


def test_env_var_overrides_config(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    # Both generation tasks read the standard tier, so that is the one an env
    # var has to move to change them.
    write_config(workspace, {"models": {"standard": MODEL, "advanced": LABEL_MODEL}})
    monkeypatch.setenv("DGML_MODELS__STANDARD", "openai/gpt-5")
    cfg = load_generation_config(workspace)
    assert cfg.model == "openai/gpt-5"
    assert cfg.label_model == "openai/gpt-5"


def test_advanced_tier_no_longer_reaches_generation(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving `advanced` must not move labeling any more — it is reserved for
    the tasks that still ask for it (grounded extraction). A config that only
    sets `advanced` leaves generation on `standard`."""
    write_config(workspace, {"models": {"standard": MODEL, "advanced": LABEL_MODEL}})
    monkeypatch.setenv("DGML_MODELS__ADVANCED", "openai/gpt-5")
    cfg = load_generation_config(workspace)
    assert cfg.label_model == MODEL


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
    write_config(workspace, {"models": {"standard": "openai/gpt-5"}})
    cfg = load_generation_config(workspace)
    # The workspace overrides `standard`; the other tiers still come from the
    # user config, which is what the deep merge is here to prove.
    assert cfg.model == "openai/gpt-5"
    assert cfg.label_model == "openai/gpt-5"


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


def test_resolve_overlay_replaces_the_models_credentials_with_them(workspace: Workspace) -> None:
    # A profile overrides the models it names, and a model's credentials move with
    # it: the workspace's key was written beside the old models and does not follow
    # the profile's. A key outside any model group survives the overlay.
    _write(
        workspace,
        {
            "model": MODEL,
            "label_model": LABEL_MODEL,
            "api_key_env": "MY_KEY",
            "thinking": "adaptive",
        },
    )
    cfg, source = resolve_generation_config(workspace, config="fast")
    assert cfg.label_model != LABEL_MODEL  # profile replaced the models...
    assert cfg.api_key_env is None  # ...and the credentials written beside them
    assert cfg.thinking == "adaptive"  # ...but not a key outside the group
    assert source == "profile:fast"


def test_resolve_flag_drops_the_files_credentials_for_the_model_it_replaces(
    workspace: Workspace, tmp_path: Path
) -> None:
    # The file's key and proxy were written beside the file's models; a flag's model
    # must not inherit them. A model no flag replaces keeps its own.
    path = tmp_path / "gen.json"
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "api_key_env": "ANT_KEY",
                "api_base": "https://anthropic-proxy",
                "label_model": LABEL_MODEL,
                "label_api_key_env": "ANT_KEY2",
            }
        ),
        encoding="utf-8",
    )
    cfg, _ = resolve_generation_config(workspace, config=str(path), model="openai/gpt-5")
    assert (cfg.model, cfg.api_key_env, cfg.api_base) == ("openai/gpt-5", None, None)
    assert (cfg.label_model, cfg.label_api_key_env) == (LABEL_MODEL, "ANT_KEY2")

    cfg, _ = resolve_generation_config(workspace, config=str(path), label_model="openai/gpt-5")
    assert (cfg.model, cfg.api_key_env) == (MODEL, "ANT_KEY")
    assert (cfg.label_model, cfg.label_api_key_env) == ("openai/gpt-5", None)


def test_override_models_take_the_configs_provider_key(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    """A profile or flag names only models; each takes the configured key of its own
    provider, as a tier would, so the run does not fall back to the process env."""
    write_config(workspace, {"models": {"family": "anthropic", "anthropic_api_key_env": "ORG"}})
    cfg, _ = resolve_generation_config(workspace, config="fast")
    assert (cfg.api_key_env, cfg.label_api_key_env) == ("ORG", "ORG")
    cfg, _ = resolve_generation_config(workspace, label_model=LABEL_MODEL)
    assert (cfg.api_key_env, cfg.label_api_key_env) == ("ORG", "ORG")
    # Another provider's model takes none.
    cfg, _ = resolve_generation_config(workspace, model="openai/gpt-5")
    assert (cfg.api_key, cfg.api_key_env) == (None, None)
    assert "does not reach" not in caplog.text


def test_override_model_with_its_own_credentials_takes_no_provider_key(
    workspace: Workspace, tmp_path: Path
) -> None:
    """A file model's own key wins, and one at a custom endpoint gets no provider key:
    `openai/llama3` behind vLLM is not OpenAI."""
    write_config(workspace, {"models": {"anthropic_api_key": "org", "openai_api_key": "sk-real"}})
    path = tmp_path / "gen.json"
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "api_key_env": "FILE_KEY",
                "label_model": "openai/llama3",
                "label_api_base": "http://vllm:8000",
            }
        ),
        encoding="utf-8",
    )
    cfg, _ = resolve_generation_config(workspace, config=str(path))
    assert (cfg.api_key, cfg.api_key_env) == (None, "FILE_KEY")
    assert (cfg.label_api_key, cfg.label_api_base) == (None, "http://vllm:8000")
    # A flag replacing the file's model drops the file's key but takes the provider's.
    cfg, _ = resolve_generation_config(workspace, config=str(path), model=MODEL)
    assert (cfg.api_key, cfg.api_key_env) == ("org", None)


def test_override_takes_the_highest_layers_provider_key(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dgml_core.storage import user_config_path

    up = user_config_path()
    up.parent.mkdir(parents=True, exist_ok=True)
    up.write_text(
        dump_toml({"models": {"family": "anthropic", "anthropic_api_key_env": "USER"}}),
        encoding="utf-8",
    )
    write_config(workspace, {"models": {"anthropic_api_key_env": "WS"}})
    assert resolve_generation_config(workspace, model=MODEL)[0].api_key_env == "WS"
    monkeypatch.setenv("DGML_MODELS__ANTHROPIC_API_KEY_ENV", "ENV")
    assert resolve_generation_config(workspace, model=MODEL)[0].api_key_env == "ENV"


# ── [generation] thinking ───────────────────────────────────────────────────


def test_thinking_defaults_to_disabled(workspace: Workspace) -> None:
    """The shipped labeling model is a Claude 5 model, which thinks adaptively
    whenever the request omits the field. Generation states the mode instead of
    inheriting it, so a config that says nothing gets thinking OFF."""
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL})
    assert load_generation_config(workspace).thinking == "disabled"


def test_thinking_can_be_set_back_to_adaptive(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL, "thinking": "adaptive"})
    assert load_generation_config(workspace).thinking == "adaptive"


def test_thinking_between_tools_is_accepted(workspace: Workspace) -> None:
    """The off switch for Claude Sonnet 5.5, which rejects "disabled"."""
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL, "thinking": "between_tools"})
    assert load_generation_config(workspace).thinking == "between_tools"


def test_invalid_thinking_mode_is_a_config_error(workspace: Workspace) -> None:
    """Caught at load time, where the message can name the file, rather than as
    a provider 400 after transcription has already been paid for."""
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL, "thinking": "sometimes"})
    with pytest.raises(GenerationConfigInvalid, match=r"generation\.thinking"):
        load_generation_config(workspace)


def test_thinking_survives_a_model_override(workspace: Workspace) -> None:
    """--model / --label-model overlay the [generation] section; the thinking
    key is not one of the overlaid fields and must not be lost with them."""
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL, "thinking": "adaptive"})
    cfg, _source = resolve_generation_config(workspace, model=MODEL, label_model=LABEL_MODEL)
    assert cfg.thinking == "adaptive"


# ── [generation] transcription ──────────────────────────────────────────────


def test_transcription_defaults_to_text(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL})
    assert load_generation_config(workspace).transcription == "text"


def test_transcription_can_be_lines(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL, "transcription": "lines"})
    assert load_generation_config(workspace).transcription == "lines"


def test_invalid_transcription_mode_is_a_config_error(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL, "transcription": "ids"})
    with pytest.raises(GenerationConfigInvalid, match=r"generation\.transcription"):
        load_generation_config(workspace)
