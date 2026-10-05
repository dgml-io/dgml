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
    ModelsConfig,
    Tier,
    expand_family,
    load_models_config,
    resolve_tiered_model,
)


def _merged(models: Any) -> dict[ConfigSection, Any]:
    return {ConfigSection.MODELS: models}


def _warned(caplog: pytest.LogCaptureFixture) -> str:
    """WARNING-and-above messages logged so far, then forget them (like
    ``capsys.readouterr``)."""
    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    caplog.clear()
    return text


def test_resolve_exact_tier_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    m = ModelsConfig(light="a", standard="b", advanced="c", expert="d")
    assert m.resolve(Tier.ADVANCED) == "c"
    assert _warned(caplog) == ""


def test_resolve_prefers_nearest_lower_tier(caplog: pytest.LogCaptureFixture) -> None:
    # expert unset; both standard and light set → nearest lower is standard.
    m = ModelsConfig(light="l", standard="s")
    assert m.resolve(Tier.EXPERT) == "s"
    assert "falling back to 'standard'" in _warned(caplog)


def test_resolve_falls_back_upward_when_nothing_below(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Only standard set; light has no lower neighbour → nearest higher is standard.
    m = ModelsConfig(standard="only-standard")
    assert m.resolve(Tier.LIGHT) == "only-standard"
    err = _warned(caplog)
    assert "tier 'light' is not set" in err
    assert "falling back to 'standard'" in err


def test_fallback_warning_is_deduped(caplog: pytest.LogCaptureFixture) -> None:
    m = ModelsConfig(standard="only-standard")
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
        assert cfg.resolve(tier) == PROVIDER_MODELS[family][tier]
    assert _warned(caplog) == ""  # all tiers set → never a fallback warning


def test_explicit_tier_overrides_its_family_default() -> None:
    cfg = _expanded({"family": "google", "expert": "my/model"})
    assert cfg.expert == "my/model"
    assert cfg.light == PROVIDER_MODELS["google"]["light"]


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
