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

"""`Configuration`: the typed builders render exactly the keys the loaders read.

The contract pinned here is shape, not validation — validation stays in the
loaders. Each typed section must produce the same table the equivalent TOML
would, so a loader fed `Configuration.build(...).sections` resolves the same
object it resolves from a file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dgml_core.configuration import (
    Classification,
    Clustering,
    Configuration,
    Conversion,
    Generation,
    Grounded,
    Identity,
    Model,
    Models,
    Ocr,
    Pdf,
    ProviderSpec,
    Storage,
    Style,
    TextExtraction,
)
from dgml_core.default_config import PROVIDER_MODELS
from dgml_core.errors import GroundedConfigMissing, InvalidArgument
from dgml_core.models_config import ConfigSection, Tier, resolve_tiered_model

_ID = Identity(workspace_id="tenant-1", name="Tenant One", organization="TenantOne")
_LOCAL = ProviderSpec("dgml_core.storage_local:LocalStore")
_STORAGE = Storage.combined(_LOCAL)


def _build(**sections: object) -> Configuration:
    return Configuration.build(identity=_ID, storage=_STORAGE, **sections)  # type: ignore[arg-type]


def test_no_sections_means_no_sections() -> None:
    assert _build().sections == {}


def test_none_fields_are_omitted_so_loader_defaults_apply() -> None:
    cfg = _build(grounded=Grounded(values_model=Model("x/y")))
    assert cfg.sections == {ConfigSection.GROUNDED: {"values_model": "x/y"}}


def test_every_section_renders_its_loader_keys() -> None:
    cfg = _build(
        models=Models(
            advanced=Model("a/m", api_key="ak", api_base="https://a"), anthropic_api_key="k"
        ),
        grounded=Grounded(
            schema_model=Model("s"),
            values_model=Model("v", api_key="vk", api_key_env="VK"),
            max_tool_iters=3,
        ),
        classification=Classification(model=Model("c", api_base="https://c"), max_pages=2),
        ocr=Ocr(
            provider="azure", max_concurrency=4, options={"endpoint": "https://e", "api_key": "o"}
        ),
        pdf=Pdf(provider="pypdfium2"),
        style=Style(model=Model("st")),
        text_extraction=TextExtraction(model=Model("te"), temperature=0.5, enabled=False),
        conversion=Conversion({"docx": ProviderSpec("mod:Conv", {"binary": "/usr/bin/x"})}),
        generation=Generation(
            model=Model("g"), label_model=Model("l", api_key="lk"), thinking="disabled"
        ),
        clustering=Clustering({"scenario": {"pooling_pages": 2}}),
    )
    assert cfg.sections == {
        ConfigSection.MODELS: {
            "advanced": "a/m",
            "advanced_api_key": "ak",
            "advanced_api_base": "https://a",
            "anthropic_api_key": "k",
        },
        ConfigSection.GROUNDED: {
            "schema_model": "s",
            "values_model": "v",
            "values_api_key": "vk",
            "values_api_key_env": "VK",
            "max_tool_iters": 3,
        },
        ConfigSection.CLASSIFICATION: {"model": "c", "api_base": "https://c", "max_pages": 2},
        ConfigSection.OCR: {
            "provider": "azure",
            "max_concurrency": 4,
            "endpoint": "https://e",
            "api_key": "o",
        },
        ConfigSection.PDF: {"provider": "pypdfium2"},
        ConfigSection.STYLE: {"model": "st", "enabled": True},
        ConfigSection.TEXT_EXTRACTION: {"model": "te", "temperature": 0.5, "enabled": False},
        ConfigSection.CONVERSION: {"docx": {"provider": "mod:Conv", "binary": "/usr/bin/x"}},
        ConfigSection.GENERATION: {
            "model": "g",
            "label_model": "l",
            "label_api_key": "lk",
            "thinking": "disabled",
        },
        ConfigSection.CLUSTERING: {"scenario": {"pooling_pages": 2}},
    }


def test_family_expands_into_tiers_like_a_toml_layer() -> None:
    cfg = _build(models=Models(family="anthropic", expert=Model("override/x")))
    models = cfg.sections[ConfigSection.MODELS]
    assert models["family"] == "anthropic"
    assert models["expert"] == "override/x"  # explicit tier wins
    for tier in ("light", "standard", "advanced"):
        assert models[tier] == PROVIDER_MODELS["anthropic"][tier]


def _resolve(cfg: Configuration, tier: Tier = Tier.ADVANCED, prefix: str = "values_") -> Model:
    return resolve_tiered_model(
        dict(cfg.sections),
        section_name=ConfigSection.GROUNDED,
        tier=tier,
        invalid=GroundedConfigMissing,
        missing=GroundedConfigMissing,
        prefix=prefix,
    )


def test_provider_key_serves_every_tier_of_its_provider() -> None:
    cfg = _build(models=Models(family="anthropic", anthropic_api_key="ant"))
    for tier in Tier:
        rm = _resolve(cfg, tier)
        assert rm == Model(PROVIDER_MODELS["anthropic"][tier], api_key="ant")


def test_mixed_family_takes_one_key_per_provider() -> None:
    cfg = _build(
        models=Models(family="anthropic_google", anthropic_api_key="ant", google_api_key="goo")
    )
    assert _resolve(cfg, Tier.LIGHT).api_key == "goo"  # gemini/…
    assert _resolve(cfg, Tier.ADVANCED).api_key == "ant"


def test_tier_key_wins_over_provider_key() -> None:
    cfg = _build(
        models=Models(
            family="anthropic",
            advanced=Model("anthropic/x", api_key="tier", api_base="https://t"),
            anthropic_api_key="ant",
        )
    )
    assert _resolve(cfg, Tier.ADVANCED) == Model(
        "anthropic/x", api_key="tier", api_base="https://t"
    )
    assert _resolve(cfg, Tier.EXPERT).api_key == "ant"


def test_section_key_wins_over_tier_and_provider_keys() -> None:
    cfg = _build(
        models=Models(family="anthropic", anthropic_api_key="ant"),
        grounded=Grounded(values_model=Model("anthropic/y", api_key="task")),
    )
    assert _resolve(cfg).api_key == "task"


def test_provider_key_reaches_a_section_model_of_that_provider_only() -> None:
    cfg = _build(
        models=Models(family="anthropic", anthropic_api_key="ant"),
        grounded=Grounded(schema_model=Model("anthropic/y"), values_model=Model("openai/gpt-4o")),
    )
    assert _resolve(cfg, Tier.EXPERT, "schema_") == Model("anthropic/y", api_key="ant")
    # Another provider's model never gets the Anthropic key: litellm reads OPENAI_API_KEY.
    assert _resolve(cfg) == Model("openai/gpt-4o")


def test_no_key_anywhere_leaves_litellm_its_env_var() -> None:
    """A family with no key set resolves to api_key=None, so litellm reads its own
    per-provider env var (the TOML path pins the same in test_generation_config)."""
    rm = _resolve(_build(models=Models(family="anthropic")))
    assert rm == Model(PROVIDER_MODELS["anthropic"]["advanced"])


def test_models_property_validates_tiers() -> None:
    assert _build(models=Models(advanced=Model("a/m"))).models.advanced == Model("a/m")


@pytest.mark.parametrize("bad", ["Acme.Corp/US", "ab", "UPPER", "-leading", "x" * 41])
def test_identity_rejects_a_malformed_workspace_id(bad: str) -> None:
    """Same rule as `create_workspace --id`: the id becomes a path segment, an S3 prefix
    and a Mongo collection prefix."""
    with pytest.raises(InvalidArgument, match="not a valid workspace id"):
        Configuration.build(
            identity=Identity(workspace_id=bad, name="n", organization="o"), storage=_STORAGE
        )


def test_storage_builds_the_store_config_pair() -> None:
    blob, doc = _STORAGE.store_configs(root=Path("/r"), workspace_id="tenant-1")
    assert blob == doc
    assert blob.provider == _LOCAL.provider and blob.workspace_id == "tenant-1"
    two = Storage(
        blobs=ProviderSpec("s3:Blob", {"bucket": "b"}), docs=ProviderSpec("mongo:Doc", {"db": "d"})
    ).store_configs(root=Path("/r"), workspace_id="t")
    assert two[0].options == {"bucket": "b"} and two[1].options == {"db": "d"}


def test_from_merged_accepts_string_keys() -> None:
    cfg = Configuration.from_merged(
        identity=_ID, storage=_STORAGE, sections={"models": {"advanced": "a/m"}}
    )
    assert cfg.sections == {ConfigSection.MODELS: {"advanced": "a/m"}}
    assert cfg == _build(models=Models(advanced=Model("a/m")))


def test_hashable_by_identity_and_storage() -> None:
    a = _build(models=Models(advanced=Model("a/m")))
    b = _build(models=Models(advanced=Model("other")))
    assert hash(a) == hash(b) and a != b
    assert len({a, b}) == 2
    other_storage = Configuration.build(
        identity=_ID, storage=Storage.combined(ProviderSpec("x:Y", {"k": 1}))
    )
    assert hash(other_storage) != hash(a)


def test_is_frozen() -> None:
    cfg = _build()
    with pytest.raises(AttributeError):
        cfg.identity = _ID  # type: ignore[misc]
