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
from dgml_core.config import load_merged_config
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
from dgml_core.conversion import FormatFamily
from dgml_core.default_config import PROVIDER_MODELS
from dgml_core.errors import GroundedConfigMissing, InvalidArgument
from dgml_core.models_config import ConfigSection, Tier, resolve_tiered_model
from dgml_core.storage import Workspace

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


def test_grounded_renders_both_reasoning_efforts() -> None:
    cfg = _build(
        grounded=Grounded(values_reasoning_effort="low", locations_reasoning_effort="default")
    )
    assert cfg.sections == {
        ConfigSection.GROUNDED: {
            "values_reasoning_effort": "low",
            "locations_reasoning_effort": "default",
        }
    }


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
        conversion=Conversion(
            {FormatFamily.DOCX: ProviderSpec("mod:Conv", {"binary": "/usr/bin/x"})}
        ),
        generation=Generation(
            model=Model("g"), label_model=Model("l", api_key="lk"), thinking="disabled"
        ),
        clustering=Clustering({"scenario": {"pooling_pages": 2}}),
    )
    assert cfg.sections == {
        # `anthropic_api_key` stays in [models]: the loaders' read writes it onto the
        # keyless models of its provider, per-call overrides included.
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
    """Through the loaders' read, which is where the provider keys are applied."""
    return resolve_tiered_model(
        load_merged_config(Workspace(root=Path("/unused"), configuration=cfg)),
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


def test_section_models_take_the_provider_key_of_their_provider() -> None:
    """One `Configuration` is one layer: the provider key on `Models` serves its section
    models as it serves its tiers — not a model of another provider, and not one at a
    custom `api_base`. Otherwise a keyless section model would read the server's own
    `ANTHROPIC_API_KEY`, not the one this configuration was given."""
    cfg = _build(
        models=Models(anthropic_api_key="ant"),
        grounded=Grounded(schema_model=Model("anthropic/y"), values_model=Model("openai/gpt-4o")),
        classification=Classification(model=Model("anthropic/z", api_base="https://proxy")),
    )
    assert _resolve(cfg, Tier.EXPERT, "schema_") == Model("anthropic/y", api_key="ant")
    assert _resolve(cfg) == Model("openai/gpt-4o")
    assert cfg.sections[ConfigSection.CLASSIFICATION] == {
        "model": "anthropic/z",
        "api_base": "https://proxy",
    }


def test_a_per_call_override_takes_the_configurations_provider_key(tmp_path: Path) -> None:
    """A multi-tenant caller passes each tenant's key by value: a `values_model=`
    override must run on that key, not on the server's own `ANTHROPIC_API_KEY`."""
    from dgml_core.grounded import load_grounded_config

    cfg = _build(models=Models(family="anthropic", anthropic_api_key="tenant-key"))
    ws = Workspace(root=tmp_path, configuration=cfg)
    config = load_grounded_config(
        ws, values_model="anthropic/claude-sonnet-4-6", schema_model="openai/gpt-5"
    )
    assert config.values_api_key == "tenant-key"
    assert config.schema_api_key is None  # no OpenAI key on this configuration


def test_a_per_call_provider_key_reaches_every_tier(tmp_path: Path) -> None:
    """The keys are applied after the overrides are in, so an override's provider key
    is the one the tiers get — not a copy of the configuration's baked in earlier."""
    cfg = _build(models=Models(family="anthropic", anthropic_api_key="K1"))
    ws = Workspace(root=tmp_path, configuration=cfg)
    merged = load_merged_config(ws, cli_overrides={"models": {"anthropic_api_key": "K2"}})
    assert all(merged[ConfigSection.MODELS][f"{t}_api_key"] == "K2" for t in Tier)
    assert cfg.sections[ConfigSection.MODELS]["anthropic_api_key"] == "K1"  # untouched


def test_a_stray_override_credential_is_refused_on_the_in_memory_path_too(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path, configuration=_build(models=Models(family="anthropic")))
    merged = load_merged_config(ws, cli_overrides={"grounded": {"values_api_key_env": "K"}})
    assert merged[ConfigSection.GROUNDED] == {"values_api_key_env": "K"}
    with pytest.raises(GroundedConfigMissing, match=r"'grounded\.values_api_key_env' is set, but"):
        resolve_tiered_model(
            merged,
            section_name=ConfigSection.GROUNDED,
            tier=Tier.ADVANCED,
            invalid=GroundedConfigMissing,
            missing=GroundedConfigMissing,
            prefix="values_",
        )


def test_a_tier_at_an_api_base_without_a_key_is_warned(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _build(
        models=Models(
            family="anthropic",
            anthropic_api_key="tenant-key",
            standard=Model("anthropic/x", api_base="https://proxy"),
        )
    )
    assert _resolve(cfg, Tier.STANDARD) == Model("anthropic/x", api_base="https://proxy")
    assert "'models.anthropic_api_key' is not sent to models.standard" in caplog.text


def test_from_merged_keeps_its_provider_keys_for_the_loaders() -> None:
    sections = {"models": {"standard": "anthropic/x", "anthropic_api_key": "sk"}}
    cfg = Configuration._from_merged(identity=_ID, storage=_STORAGE, sections=sections)
    assert cfg.sections[ConfigSection.MODELS] == sections["models"]  # as merged
    assert _resolve(cfg, Tier.STANDARD) == Model("anthropic/x", api_key="sk")
    assert cfg.models.standard == Model("anthropic/x", api_key="sk")


def test_a_toml_workspaces_config_keeps_its_provider_keys(tmp_path: Path) -> None:
    """Reopening with `configuration=ws.config` keeps the keys for per-call overrides."""
    from dgml_core.grounded import load_grounded_config

    from .conftest import write_config

    src = Workspace(root=tmp_path / "src")
    write_config(src, {"models": {"family": "anthropic", "anthropic_api_key_env": "ORG"}})
    reopened = Workspace(root=tmp_path / "copy", configuration=src.config)
    config = load_grounded_config(reopened, values_model="anthropic/claude-sonnet-4-6")
    assert config.values_api_key_env == "ORG"


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
    cfg = Configuration._from_merged(
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


def test_redaction_matches_whole_words_and_spares_env_names() -> None:
    from dgml_core.redaction import redacted

    table = {
        "max_tokens": 64000,
        "max_completion_tokens": 1,
        "auth_token": "t",
        "api_key": "k",
        "api_key_env": "ENV",
        "mongo_uri": "mongodb://u:p@h",
        "nested": {"aws_secret_access_key": "s", "bucket": "b"},
        "unset_key": None,
        "database_url": "postgresql://u:pw@h/d",
        "endpoint_url": "http://minio:9000",
        "plain_uri": "mongodb://h/d",
        "dsn": "d",
        "apikey": "a",
        "hosts": [{"password": "p", "host": "h"}, "h2"],
    }
    assert redacted(table) == {
        "max_tokens": 64000,
        "max_completion_tokens": 1,
        "auth_token": "***",
        "api_key": "***",
        "api_key_env": "ENV",
        "mongo_uri": "***",
        "nested": {"aws_secret_access_key": "***", "bucket": "b"},
        "unset_key": None,
        "database_url": "***",
        "endpoint_url": "http://minio:9000",
        "plain_uri": "mongodb://h/d",
        "dsn": "***",
        "apikey": "***",
        "hosts": [{"password": "***", "host": "h"}, "h2"],
    }


def test_resolved_task_configs_hide_their_keys() -> None:
    """The loaders' results carry the literal key once the merge has applied it, and
    they are what a host is likeliest to log."""
    from dgml_core.classification import ClassificationConfig
    from dgml_core.generation.config import GenerationConfig
    from dgml_core.grounded import GroundedConfig
    from dgml_core.ocr import OcrConfig
    from dgml_core.style_config import StyleConfig
    from dgml_core.text_extraction_config import TextExtractionConfig

    configs = [
        GenerationConfig(model="m", label_model="l", api_key="SECRET", label_api_key="SECRET"),
        GroundedConfig(
            schema_model="s", values_model="v", schema_api_key="SECRET", values_api_key="SECRET"
        ),
        ClassificationConfig(model="m", api_key="SECRET"),
        StyleConfig(model="m", api_key="SECRET"),
        TextExtractionConfig(model="m", api_key="SECRET"),
        OcrConfig(provider="azure", options={"endpoint": "https://e", "api_key": "SECRET"}),
    ]
    for cfg in configs:
        assert "SECRET" not in repr(cfg) + str(cfg), cfg
    assert "'endpoint': 'https://e'" in repr(configs[-1])


def test_reprs_never_show_a_credential() -> None:
    """A `Configuration` is built from secrets passed by value, and a host logs or
    debug-prints it like any object: no repr or str may carry one. Model ids, env-var
    names, endpoints and non-secret options stay visible."""
    spec = ProviderSpec(
        "s3:Blob",
        {
            "bucket": "b",
            "aws_secret_access_key": "SECRET-aws",
            "mongo_uri": "mongodb://u:SECRET-pw@h/d",
        },
    )
    expert = Model("anthropic/x", api_key="SECRET-tier", api_base="https://p")
    advanced = Model("anthropic/y", api_key_env="OPUS_KEY")
    cfg = Configuration.build(
        identity=_ID,
        storage=Storage.combined(spec),
        models=Models(
            family="anthropic", anthropic_api_key="SECRET-ant", expert=expert, advanced=advanced
        ),
        ocr=Ocr(provider="azure", options={"endpoint": "https://e", "api_key": "SECRET-ocr"}),
        conversion=Conversion(
            {FormatFamily.DOCX: ProviderSpec("mod:Conv", {"auth_token": "SECRET-tok"})}
        ),
    )
    objects = [cfg, cfg.storage, spec, cfg.models, cfg.models.expert, expert]
    objects += list(cfg.storage.store_configs(root=Path("/r"), workspace_id="t"))
    objects.append(Models(anthropic_api_key="SECRET-ant"))
    for obj in objects:
        text = repr(obj) + str(obj)
        assert "SECRET" not in text, text
    text = repr(cfg)
    for visible in ("anthropic/x", "OPUS_KEY", "https://p", "'bucket': 'b'", "https://e", "'***'"):
        assert visible in text
