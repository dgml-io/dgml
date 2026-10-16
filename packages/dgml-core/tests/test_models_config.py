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
    expand_family,
    load_models_config,
    normalize_models_layer,
    provider_of,
    resolve_tiered_model,
)


def _merged(models: Any) -> dict[ConfigSection, Any]:
    return {ConfigSection.MODELS: models}


def _normalized(models: dict[str, Any]) -> ModelsConfig:
    """One layer, as the merge sees it: family expanded, provider keys applied."""
    return load_models_config(_merged(normalize_models_layer(models)))


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
    with pytest.raises(ModelsConfigInvalid, match=r"'models\.light' is not"):
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


def test_provider_key_does_not_cross_layers() -> None:
    """A provider key is shorthand within its own layer. A tier another layer sets
    arrives with its group replaced, so the key does not follow it — even to a model
    of the same provider. That layer writes its own `expert_api_key` or relies on
    litellm's env var."""
    user = normalize_models_layer({"family": "anthropic", "anthropic_api_key": "ant"})
    workspace = normalize_models_layer({"expert": "anthropic/other"})
    cfg = load_models_config(_merged({**user, **workspace}))
    assert cfg.expert == Model("anthropic/other")
    assert cfg.standard == Model(PROVIDER_MODELS["anthropic"]["standard"], api_key="ant")


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
    with pytest.raises(ModelsConfigInvalid, match=r"'generation\.label_model' is not"):
        _resolve(merged, Tier.STANDARD, "label_")
    merged[ConfigSection.GENERATION] = {"label_api_key_env": "SEC"}
    with pytest.raises(ModelsConfigInvalid, match=r"'generation\.label_model' is not"):
        _resolve(merged, Tier.STANDARD, "label_")


def test_section_takes_the_tier_model_whole() -> None:
    merged = {
        ConfigSection.MODELS: {"standard": "s", "standard_api_key": "t", "standard_api_base": "tb"},
        ConfigSection.GENERATION: {},
    }
    assert _resolve(merged, Tier.STANDARD, "label_") == Model("s", api_key="t", api_base="tb")


def test_section_model_takes_no_provider_key() -> None:
    merged = {
        ConfigSection.MODELS: {"openai_api_key": "oai", "anthropic_api_key": "ant"},
        ConfigSection.GENERATION: {"model": "openai/gpt", "label_model": "ollama/llama"},
    }
    assert _resolve(merged) == Model("openai/gpt")
    assert _resolve(merged, prefix="label_") == Model("ollama/llama")


def test_section_model_behind_a_custom_endpoint_gets_no_provider_key() -> None:
    """The leak this rule exists for: `openai/llama3` at a vLLM `api_base` is a
    third-party server, and `openai_api_key` must not be sent to it."""
    merged = {
        ConfigSection.MODELS: {"openai_api_key": "sk-real-openai"},
        ConfigSection.CLASSIFICATION: {
            "model": "openai/llama3",
            "api_base": "http://vllm.internal:8000",
        },
    }
    rm = resolve_tiered_model(
        merged,
        section_name=ConfigSection.CLASSIFICATION,
        tier=Tier.LIGHT,
        invalid=ModelsConfigInvalid,
        missing=ModelsConfigInvalid,
    )
    assert rm == Model("openai/llama3", api_base="http://vllm.internal:8000")


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


def test_normalize_rejects_a_tier_credential_without_its_tier_in_the_layer() -> None:
    with pytest.raises(ModelsConfigInvalid, match=r"'models\.light' is not in this config"):
        normalize_models_layer({"light_api_key": "k"})
    # `family` sets the tier in this layer, so the key has its model.
    out = normalize_models_layer({"family": "anthropic", "light_api_key": "k"})
    assert out["light_api_key"] == "k"


def test_normalize_leaves_an_unknown_family_for_load_to_report() -> None:
    """A misspelled family with a tier key is the family's error, not the key's."""
    layer = {"family": "antropic", "light_api_key": "k"}
    assert normalize_models_layer(layer) == layer
    with pytest.raises(ModelsConfigInvalid, match=r"'models\.family' must be one of"):
        load_models_config(_merged(layer))


def test_normalize_rejects_a_section_credential_without_its_model_in_the_layer() -> None:
    from dgml_core.errors import GroundedConfigInvalid
    from dgml_core.models_config import normalize_task_layer

    with pytest.raises(GroundedConfigInvalid, match=r"'grounded\.values_model' is not in this"):
        normalize_task_layer({"values_api_key_env": "K"}, ConfigSection.GROUNDED)
    out = normalize_task_layer(
        {"values_model": "m", "values_api_key_env": "K"}, ConfigSection.GROUNDED
    )
    assert out["values_api_key_env"] == "K" and out["values_api_base"] is None


def test_provider_keys_are_consumed_and_dropped_by_the_layer() -> None:
    out = normalize_models_layer({"family": "anthropic", "anthropic_api_key": "ant"})
    assert "anthropic_api_key" not in out
    assert out["advanced_api_key"] == "ant"
