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

"""The ``[models]`` block: family expansion, loading, resolution, fallback,
and the stderr warning."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from dgml_core.default_config import PROVIDER_MODELS
from dgml_core.errors import ModelsConfigInvalid
from dgml_core.models_config import (
    TIERS,
    ConfigSection,
    Model,
    ModelsConfig,
    Tier,
    apply_provider_keys,
    expand_family,
    load_models_config,
    normalize_models_layer,
    normalize_task_layer,
    provider_of,
    resolve_tiered_model,
)


def _merged(models: Any) -> dict[ConfigSection, Any]:
    return {ConfigSection.MODELS: models}


def _drop_none(table: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in table.items() if v is not None}


def _normalized(*layers: dict[str, Any]) -> ModelsConfig:
    """Layers (lowest first) as the merge sees them: each normalized, merged, its
    `None`s dropped, then the provider keys applied."""
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(normalize_models_layer(layer))
    sections = _merged(_drop_none(merged))
    apply_provider_keys(sections)
    return load_models_config(sections)


def _warned(caplog: pytest.LogCaptureFixture) -> str:
    """WARNING-and-above messages logged so far, then forget them (like
    ``capsys.readouterr``)."""
    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    caplog.clear()
    return text


def test_resolve_exact_tier_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    m = ModelsConfig(light=Model("a"), standard=Model("b"), advanced=Model("c"), expert=Model("d"))
    assert m.resolve(Tier.ADVANCED) == Model("c")
    assert _warned(caplog) == ""


def test_resolve_prefers_nearest_lower_tier(caplog: pytest.LogCaptureFixture) -> None:
    # expert unset; both standard and light set → nearest lower is standard.
    m = ModelsConfig(light=Model("l"), standard=Model("s"))
    assert m.resolve(Tier.EXPERT) == Model("s")
    assert "falling back to 'standard'" in _warned(caplog)


def test_resolve_falls_back_upward_when_nothing_below(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Only standard set; light has no lower neighbour → nearest higher is standard.
    m = ModelsConfig(standard=Model("only-standard"))
    assert m.resolve(Tier.LIGHT) == Model("only-standard")
    err = _warned(caplog)
    assert "tier 'light' is not set" in err
    assert "falling back to 'standard'" in err


def test_fallback_warning_is_deduped(caplog: pytest.LogCaptureFixture) -> None:
    m = ModelsConfig(standard=Model("only-standard"))
    m.resolve(Tier.LIGHT)
    first = _warned(caplog)
    m.resolve(Tier.LIGHT)
    second = _warned(caplog)
    assert first.count("falling back") == 1
    assert second == ""  # same (tier, used) pair — not repeated


def test_resolve_none_when_no_tier_set(caplog: pytest.LogCaptureFixture) -> None:
    assert ModelsConfig().resolve(Tier.STANDARD) is None
    assert _warned(caplog) == ""  # nothing to fall back to → no warning


def test_load_absent_or_empty_section_yields_all_none() -> None:
    assert load_models_config({}) == ModelsConfig()
    assert load_models_config(_merged({})) == ModelsConfig()


def test_load_rejects_non_table_section() -> None:
    with pytest.raises(ModelsConfigInvalid, match="must be a table"):
        load_models_config(_merged("standard"))


def _expanded(models: dict[str, Any]) -> ModelsConfig:
    """Load a single config layer the way the merge would: family expanded first."""
    return load_models_config(_merged(expand_family(models)))


@pytest.mark.parametrize("family", sorted(PROVIDER_MODELS))
def test_family_fills_every_tier(family: str, caplog: pytest.LogCaptureFixture) -> None:
    cfg = _expanded({"family": family})
    for tier in TIERS:
        assert cfg.resolve(tier) == Model(PROVIDER_MODELS[family][tier])
    assert _warned(caplog) == ""  # all tiers set → never a fallback warning


def test_explicit_tier_overrides_its_family_default() -> None:
    cfg = _expanded({"family": "google", "expert": "my/model"})
    assert cfg.expert == Model("my/model")
    assert cfg.light == Model(PROVIDER_MODELS["google"]["light"])


@pytest.mark.parametrize("models", [{}, {"light": "l"}, {"family": "nope"}, {"family": 1}])
def test_expand_family_leaves_tables_without_a_known_family_alone(models: dict[str, Any]) -> None:
    assert expand_family(models) is models


def test_load_does_not_expand_family() -> None:
    """Expansion is the merge's job (per layer); the loader only validates."""
    assert load_models_config(_merged({"family": "google"})) == ModelsConfig()


def test_unknown_family_is_rejected_naming_the_choices() -> None:
    with pytest.raises(ModelsConfigInvalid, match="anthropic_google") as exc:
        load_models_config(_merged({"family": "mixed"}))
    assert "'models.family'" in str(exc.value)


@pytest.mark.parametrize("field", ["family", "advanced"])
@pytest.mark.parametrize("value", [123, "", "   "])
def test_non_string_or_blank_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ModelsConfigInvalid, match=f"'models.{field}' must be a non-empty string"):
        load_models_config(_merged({field: value}))


def test_missing_model_error_names_the_family_key() -> None:
    class _Invalid(ModelsConfigInvalid):
        pass

    class _Missing(ModelsConfigInvalid):
        pass

    with pytest.raises(_Missing, match=r"\[models\].family"):
        resolve_tiered_model(
            {},
            section_name=ConfigSection.GENERATION,
            tier=Tier.STANDARD,
            invalid=_Invalid,
            missing=_Missing,
        )


def _resolve(
    merged: dict[ConfigSection, Any], tier: Tier = Tier.ADVANCED, prefix: str = ""
) -> Model:
    return resolve_tiered_model(
        merged,
        section_name=ConfigSection.GENERATION,
        tier=tier,
        invalid=ModelsConfigInvalid,
        missing=ModelsConfigInvalid,
        prefix=prefix,
    )


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("anthropic/claude-x", "anthropic"),
        ("gemini/flash", "google"),
        ("openai/gpt", "openai"),
        ("ollama/llama", None),
        ("unprefixed", None),
    ],
)
def test_provider_of_matches_the_litellm_prefix(model: str, provider: str | None) -> None:
    assert provider_of(model) == provider


def test_tier_credentials_are_read_with_the_tier() -> None:
    cfg = load_models_config(
        _merged({"light": "l", "light_api_key_env": "L_KEY", "light_api_base": "https://l"})
    )
    assert cfg.light == Model("l", api_key_env="L_KEY", api_base="https://l")


def test_tier_credentials_without_the_tier_are_rejected() -> None:
    with pytest.raises(ModelsConfigInvalid, match=r"same config layer as 'models\.light'"):
        load_models_config(_merged({"standard": "s", "light_api_key": "k"}))


@pytest.mark.parametrize("prefix", ["light_", "openai_"])
def test_key_and_env_name_clash_is_rejected(prefix: str) -> None:
    """Both forms in one layer: a tier's pair is caught at load, a provider's where it
    is consumed — the normalizer, since provider keys never reach the merged table."""
    table = {"light": "openai/x", f"{prefix}api_key": "k", f"{prefix}api_key_env": "K"}
    with pytest.raises(
        ModelsConfigInvalid, match=f"'models.{prefix}api_key' / 'models.{prefix}api_key_env'"
    ):
        _normalized(table)


def test_provider_key_fills_its_layers_tiers_by_model_prefix_unless_the_tier_has_its_own() -> None:
    cfg = _normalized(
        {
            "family": "anthropic_google",
            "expert": "anthropic/pinned",
            "expert_api_key": "pinned-key",
            "anthropic_api_key_env": "ANT",
            "google_api_key": "goo",
        }
    )
    assert cfg.light == Model(PROVIDER_MODELS["anthropic_google"]["light"], api_key="goo")
    assert cfg.standard == Model(PROVIDER_MODELS["anthropic_google"]["standard"], api_key_env="ANT")
    assert cfg.expert == Model("anthropic/pinned", api_key="pinned-key")


def test_provider_key_never_reaches_a_tier_with_an_api_base() -> None:
    """`openai/llama3` behind a vLLM endpoint is not OpenAI: the OpenAI key must not go there."""
    cfg = _normalized(
        {
            "openai_api_key": "sk-real",
            "light": "openai/llama3",
            "light_api_base": "http://vllm:8000",
        }
    )
    assert cfg.light == Model("openai/llama3", api_base="http://vllm:8000")


def test_provider_key_reaches_a_tier_another_layer_sets() -> None:
    """A provider key is a setting like any other: applied after the merge, it
    reaches every keyless model of its provider, whichever layer set the model."""
    cfg = _normalized(
        {"family": "anthropic", "anthropic_api_key": "ant"}, {"expert": "anthropic/other"}
    )
    assert cfg.expert == Model("anthropic/other", api_key="ant")
    assert cfg.standard == Model(PROVIDER_MODELS["anthropic"]["standard"], api_key="ant")


def test_the_higher_layers_provider_key_pair_wins_whole() -> None:
    """One form in the user config, the other in the workspace: the workspace's pair
    replaces the user's, so the two never clash."""
    cfg = _normalized(
        {"family": "anthropic", "anthropic_api_key": "user-key"},
        {"anthropic_api_key_env": "WS_KEY"},
    )
    assert cfg.standard == Model(PROVIDER_MODELS["anthropic"]["standard"], api_key_env="WS_KEY")


def test_fallback_tier_brings_its_own_credentials() -> None:
    merged = _merged({"standard": "anthropic/s", "standard_api_key": "s-key"})
    assert _resolve(merged, Tier.EXPERT) == Model("anthropic/s", api_key="s-key")


def test_section_credentials_without_a_section_model_are_rejected() -> None:
    """A key is written beside the model it is for. Section credentials with no
    section model would have to pair with a tier's model — a group split across two
    tables — so they are refused rather than paired up."""
    merged = {
        ConfigSection.MODELS: {"standard": "s", "standard_api_key": "t", "standard_api_base": "tb"},
        ConfigSection.GENERATION: {"label_api_base": "sb"},
    }
    with pytest.raises(
        ModelsConfigInvalid, match=r"same config layer as 'generation\.label_model'"
    ):
        _resolve(merged, Tier.STANDARD, "label_")
    merged[ConfigSection.GENERATION] = {"label_api_key_env": "SEC"}
    with pytest.raises(
        ModelsConfigInvalid, match=r"same config layer as 'generation\.label_model'"
    ):
        _resolve(merged, Tier.STANDARD, "label_")


def test_section_takes_the_tier_model_whole() -> None:
    merged = {
        ConfigSection.MODELS: {"standard": "s", "standard_api_key": "t", "standard_api_base": "tb"},
        ConfigSection.GENERATION: {},
    }
    assert _resolve(merged, Tier.STANDARD, "label_") == Model("s", api_key="t", api_base="tb")


def _section_layer(section: ConfigSection, table: dict[str, Any], **providers: Any) -> Model:
    """A task section beside the given provider keys, merged, keys applied, resolved."""
    models = {}
    for p, (api_key, api_key_env) in providers.items():
        models.update({f"{p}_api_key": api_key, f"{p}_api_key_env": api_key_env})
    merged = {
        section: _drop_none(normalize_task_layer(table, section)),
        ConfigSection.MODELS: _drop_none(models),
    }
    apply_provider_keys(merged)
    return resolve_tiered_model(
        merged,
        section_name=section,
        tier=Tier.LIGHT,
        invalid=ModelsConfigInvalid,
        missing=ModelsConfigInvalid,
    )


def test_section_model_takes_its_providers_key() -> None:
    model = _section_layer(ConfigSection.GENERATION, {"model": "openai/gpt"}, openai=("oai", None))
    assert model == Model("openai/gpt", api_key="oai")
    # A model of another provider takes nothing.
    model = _section_layer(
        ConfigSection.GENERATION, {"model": "ollama/llama"}, openai=("oai", None)
    )
    assert model == Model("ollama/llama")


def test_section_model_with_its_own_key_takes_no_provider_key() -> None:
    model = _section_layer(
        ConfigSection.GENERATION,
        {"model": "anthropic/x", "api_key_env": "OWN"},
        anthropic=("prov", None),
    )
    assert model == Model("anthropic/x", api_key_env="OWN")


def test_section_model_behind_a_custom_endpoint_gets_no_provider_key() -> None:
    """The leak this rule exists for: `openai/llama3` at a vLLM `api_base` is a
    third-party server, and `openai_api_key` must not be sent to it."""
    model = _section_layer(
        ConfigSection.CLASSIFICATION,
        {"model": "openai/llama3", "api_base": "http://vllm.internal:8000"},
        openai=("sk-real-openai", None),
    )
    assert model == Model("openai/llama3", api_base="http://vllm.internal:8000")


def test_malformed_models_table_does_not_break_a_section_that_names_its_model() -> None:
    """A section that names its model never consults [models], so a bad tier table
    is not that task's problem."""
    merged = {
        ConfigSection.MODELS: {"family": "nope"},
        ConfigSection.GENERATION: {"model": "openai/gpt"},
    }
    assert _resolve(merged) == Model("openai/gpt")


def test_malformed_section_credentials_are_the_sections_error() -> None:
    merged = {ConfigSection.GENERATION: {"model": "m", "api_key": "", "api_key_env": None}}
    with pytest.raises(ModelsConfigInvalid, match=r"'generation\.api_key' must be a non-empty"):
        _resolve(merged)


def test_normalize_disowns_a_tier_credential_without_its_tier_in_the_layer() -> None:
    """The layer takes the tier as unset, so the key never pairs with a lower layer's
    model; load then refuses the key with no tier."""
    out = normalize_models_layer({"light_api_key": "k", "light_api_base": "b"})
    assert out == {
        "light": None,
        "light_api_key": "k",
        "light_api_key_env": None,
        "light_api_base": "b",
    }
    with pytest.raises(
        ModelsConfigInvalid,
        match=r"'models\.light_api_key' is set, but not .* 'models\.light'",
    ):
        load_models_config(_merged(_drop_none(out)))
    # `family` sets the tier in this layer, so the key has its model.
    out = normalize_models_layer({"family": "anthropic", "light_api_key": "k"})
    assert out["light_api_key"] == "k" and out["light"] == PROVIDER_MODELS["anthropic"]["light"]


def test_normalize_leaves_a_stray_tier_key_beside_an_unknown_family_to_the_family_error() -> None:
    """A misspelled family with a tier key is the family's error, not the key's: the
    tier is not taken as unset, and load reports the family first."""
    out = normalize_models_layer({"family": "antropic", "light_api_key": "k"})
    assert out == {"family": "antropic", "light_api_key": "k"}
    with pytest.raises(ModelsConfigInvalid, match=r"'models\.family' must be one of"):
        load_models_config(_merged(out))


def test_normalize_runs_the_layer_checks_despite_an_unknown_family() -> None:
    """Only the expansion is skipped: the provider key pair and explicit tiers still
    have their groups completed. The family is left for load to report."""
    out = normalize_models_layer(
        {"family": "antropic", "standard": "anthropic/x", "anthropic_api_key": "k"}
    )
    assert out == {
        "family": "antropic",
        "standard": "anthropic/x",
        "anthropic_api_key": "k",
        "anthropic_api_key_env": None,
        "standard_api_key": None,
        "standard_api_key_env": None,
        "standard_api_base": None,
    }
    with pytest.raises(ModelsConfigInvalid, match=r"'models\.family' must be one of"):
        load_models_config(_merged(out))


def test_normalize_disowns_a_section_credential_without_its_model_in_the_layer() -> None:
    layer = {"values_api_key_env": "K", "schema_model": "s", "max_tool_iters": 3}
    out = normalize_task_layer(layer, ConfigSection.GROUNDED)
    assert _drop_none(out) == {"values_api_key_env": "K", "schema_model": "s", "max_tool_iters": 3}
    assert out["values_model"] is None and out["schema_api_key"] is None
    merged = {
        ConfigSection.GROUNDED: _drop_none(out),
        ConfigSection.MODELS: {"family": "anthropic"},
    }
    with pytest.raises(ModelsConfigInvalid, match=r"'grounded\.values_api_key_env' is set, but"):
        resolve_tiered_model(
            merged,
            section_name=ConfigSection.GROUNDED,
            tier=Tier.ADVANCED,
            invalid=ModelsConfigInvalid,
            missing=ModelsConfigInvalid,
            prefix="values_",
        )
    out = normalize_task_layer(
        {"values_model": "m", "values_api_key_env": "K"}, ConfigSection.GROUNDED
    )
    assert out["values_api_key_env"] == "K" and out["values_api_base"] is None


def test_normalize_disowns_a_stray_credential_whatever_the_sections_switch_says() -> None:
    """The switch is the loader's business: a disabled section never resolves its
    model, so the stray key is refused only once the section is on."""
    nulls = {"api_key": None, "api_base": None}
    for enabled in (False, True):
        out = normalize_task_layer({"enabled": enabled, "api_key_env": "K"}, ConfigSection.STYLE)
        assert out == {"enabled": enabled, "model": None, "api_key_env": "K", **nulls}
    out = normalize_task_layer({"enabled": False, "label_api_key": "k"}, ConfigSection.GENERATION)
    assert out == {
        "enabled": False,
        "label_model": None,
        "label_api_key": "k",
        "label_api_key_env": None,
        "label_api_base": None,
    }


def test_a_malformed_provider_key_is_left_for_the_loaders() -> None:
    """The merge raises nothing: the pair is completed but not applied, the tiers
    report it, and so does a keyless section model of that provider — the key it
    would have taken. A model of another provider never consults it."""
    out = normalize_models_layer({"family": "anthropic", "anthropic_api_key": ""})
    assert out["anthropic_api_key"] == "" and out["anthropic_api_key_env"] is None
    merged = _merged(_drop_none(out))
    apply_provider_keys(merged)
    assert "standard_api_key" not in merged[ConfigSection.MODELS]
    with pytest.raises(
        ModelsConfigInvalid, match=r"'models\.anthropic_api_key' must be a non-empty"
    ):
        load_models_config(merged)
    merged[ConfigSection.GENERATION] = {"model": "anthropic/x"}
    with pytest.raises(
        ModelsConfigInvalid, match=r"'models\.anthropic_api_key' must be a non-empty"
    ):
        _resolve(merged)
    merged[ConfigSection.GENERATION] = {"model": "openai/x"}
    assert _resolve(merged) == Model("openai/x")


def test_provider_key_pair_is_one_group_in_its_layer() -> None:
    """The layer keeps its key and blanks the other form, so a higher layer's pair
    replaces it whole; tiers take it only after the merge."""
    out = normalize_models_layer({"family": "anthropic", "anthropic_api_key": "ant"})
    assert (out["anthropic_api_key"], out["anthropic_api_key_env"]) == ("ant", None)
    assert out["advanced_api_key"] is None


def test_a_null_model_key_unsets_its_whole_group() -> None:
    """Null is a value the layer set, so the group goes with it: the merge then drops
    a lower layer's credentials along with the model, instead of leaving them stray."""
    nulls = {"api_key": None, "api_key_env": None, "api_base": None}
    assert normalize_task_layer({"model": None}, ConfigSection.GENERATION) == {
        "model": None,
        **nulls,
    }
    out = normalize_models_layer({"family": "anthropic", "light": None})
    assert out["light"] is None and all(out[f"light_{k}"] is None for k in nulls)
    assert out["standard"] == PROVIDER_MODELS["anthropic"]["standard"]
