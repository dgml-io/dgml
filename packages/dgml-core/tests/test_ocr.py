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

"""Config loading, ABC contract, and dispatch tests for OCR.

Provider-specific tests (Azure, AWS) live in ``test_ocr_azure.py`` and
``test_ocr_aws.py``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
from dgml_core.errors import OcrConfigInvalid, OcrConfigMissing
from dgml_core.ocr import (
    BUILTIN_OCR_PROVIDERS,
    DEFAULT_OCR_CONCURRENCY,
    OcrConfig,
    OcrProvider,
    OcrProviderName,
    extract_text_ocr,
    load_ocr_config,
    make_ocr_provider,
    resolve_provider_class,
)
from dgml_core.ocr_aws import AwsProvider
from dgml_core.ocr_azure import AzureProvider
from dgml_core.storage import Workspace

from .conftest import make_fake_png, write_ocr_config


def install_provider(monkeypatch: pytest.MonkeyPatch, cls: type[OcrProvider]) -> str:
    """Make ``cls`` resolvable as a dotted path, and return that path.

    Providers are looked up by importing ``"module:ClassName"``, so a test's
    locally-defined fake is published as an attribute of this test module for the
    duration of the test. These tests then drive the real resolver rather than
    reaching past it into a registry — which is the thing a third party's provider
    will actually exercise.
    """
    monkeypatch.setattr(sys.modules[__name__], cls.__name__, cls, raising=False)
    return f"{__name__}:{cls.__name__}"


# ---------------------------------------------------------------------------
# load_ocr_config
# ---------------------------------------------------------------------------


def test_load_ocr_config_default_warning_once_per_workspace(
    workspace: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fallback line is deduped per workspace: a bulk add validates and then
    extracts (two ``load_ocr_config`` calls per file), which must not repeat it —
    while a *different* unconfigured workspace in the same process still gets
    its own line."""
    monkeypatch.setattr(sys, "platform", "darwin")
    with caplog.at_level(logging.WARNING, logger="dgml_core.ocr"):
        load_ocr_config(workspace)
        load_ocr_config(workspace)  # a file add: validate, then extract
        assert caplog.text.count("defaulting to the on-device macOS") == 1

        other = Workspace(root=tmp_path / "ws2")
        other.root.mkdir(parents=True, exist_ok=True)
        load_ocr_config(other)
    assert caplog.text.count("defaulting to the on-device macOS") == 2
    # Each line names its workspace — that is what makes two lines useful.
    assert str(workspace.root) in caplog.text
    assert str(other.root) in caplog.text


def test_load_ocr_config_defaults_to_macos_on_darwin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """On macOS, a missing config defaults to the on-device provider and
    warns that it's doing so."""
    monkeypatch.setattr(sys, "platform", "darwin")
    with caplog.at_level(logging.WARNING, logger="dgml_core.ocr"):
        cfg = load_ocr_config(workspace)
    assert "defaulting to the on-device macOS" in caplog.text
    assert cfg.provider == OcrProviderName.MACOS


def test_load_ocr_config_no_config_raises_off_darwin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off macOS there is no built-in OCR engine, so a missing config is an
    error the user must fix."""
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(OcrConfigMissing):
        load_ocr_config(workspace)


def test_load_ocr_config_no_ocr_section_defaults_to_macos_on_darwin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # `[other]` is an *unknown* section, dropped by `extra="ignore"` before it
    # reaches the merged mapping — distinct from a bare `[ocr]` (covered below),
    # which now arrives as an empty table.
    monkeypatch.setattr(sys, "platform", "darwin")
    workspace.config_path.write_text("[other]\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="dgml_core.ocr"):
        cfg = load_ocr_config(workspace)
    assert "defaulting to the on-device macOS" in caplog.text
    assert cfg.provider == OcrProviderName.MACOS


def test_load_ocr_config_no_ocr_section_raises_off_darwin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    workspace.config_path.write_text("[other]\n", encoding="utf-8")
    with pytest.raises(OcrConfigMissing):
        load_ocr_config(workspace)


def test_load_ocr_config_bare_section_defaults_to_macos_on_darwin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A bare `[ocr]` is the same as no section at all, not a misconfiguration.

    Unlike `style` / `text_extraction`, this section's presence carries no
    meaning — `provider` selects the backend. `dgml init` ships a commented-out
    `# [ocr]` block, so uncommenting only the header must not hard-fail.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    workspace.config_path.write_text("[ocr]\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="dgml_core.ocr"):
        cfg = load_ocr_config(workspace)
    assert "defaulting to the on-device macOS" in caplog.text
    assert cfg.provider == OcrProviderName.MACOS


def test_load_ocr_config_bare_section_raises_off_darwin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    workspace.config_path.write_text("[ocr]\n", encoding="utf-8")
    with pytest.raises(OcrConfigMissing):
        load_ocr_config(workspace)


def test_load_ocr_config_invalid_toml(workspace: Workspace) -> None:
    from dgml_core.errors import CorruptMetadata

    workspace.config_path.write_text("{ not valid toml", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        load_ocr_config(workspace)


def test_load_ocr_config_azure_happy(workspace: Workspace) -> None:
    write_ocr_config(
        workspace,
        {
            "provider": "azure",
            "endpoint": "https://foo.cognitiveservices.azure.com/",
            "api_key_env": "FOO_KEY",
        },
    )
    cfg = load_ocr_config(workspace)
    assert cfg.provider == OcrProviderName.AZURE
    assert cfg.options["endpoint"] == "https://foo.cognitiveservices.azure.com/"
    assert cfg.options["api_key_env"] == "FOO_KEY"


def test_load_ocr_config_max_concurrency_defaults(workspace: Workspace) -> None:
    write_ocr_config(
        workspace,
        {"provider": "azure", "endpoint": "https://foo.cognitiveservices.azure.com/"},
    )
    cfg = load_ocr_config(workspace)
    assert cfg.max_concurrency == DEFAULT_OCR_CONCURRENCY == 5


def test_load_ocr_config_max_concurrency_override(workspace: Workspace) -> None:
    write_ocr_config(
        workspace,
        {
            "provider": "azure",
            "endpoint": "https://foo.cognitiveservices.azure.com/",
            "max_concurrency": 12,
        },
    )
    assert load_ocr_config(workspace).max_concurrency == 12


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "5"])
def test_load_ocr_config_max_concurrency_rejects_non_positive_int(
    workspace: Workspace, bad: object
) -> None:
    write_ocr_config(
        workspace,
        {
            "provider": "azure",
            "endpoint": "https://foo.cognitiveservices.azure.com/",
            "max_concurrency": bad,
        },
    )
    with pytest.raises(OcrConfigInvalid, match="max_concurrency"):
        load_ocr_config(workspace)


def test_load_ocr_config_azure_no_key_env(workspace: Workspace) -> None:
    """Token auth (no api_key_env / api_key) is valid — falls through to
    DefaultAzureCredential."""
    write_ocr_config(
        workspace, {"provider": "azure", "endpoint": "https://foo.cognitiveservices.azure.com/"}
    )
    cfg = load_ocr_config(workspace)
    assert "api_key" not in cfg.options
    assert "api_key_env" not in cfg.options


def test_load_ocr_config_azure_literal_api_key(workspace: Workspace) -> None:
    """A literal api_key in config is accepted (developers may put keys
    directly in workspace config.toml — it isn't checked in)."""
    write_ocr_config(
        workspace,
        {
            "provider": "azure",
            "endpoint": "https://foo.cognitiveservices.azure.com/",
            "api_key": "literal-test-key",
        },
    )
    cfg = load_ocr_config(workspace)
    assert cfg.options["api_key"] == "literal-test-key"
    assert "api_key_env" not in cfg.options


def test_load_ocr_config_azure_rejects_both_api_key_and_env(workspace: Workspace) -> None:
    write_ocr_config(
        workspace,
        {
            "provider": "azure",
            "endpoint": "https://foo.cognitiveservices.azure.com/",
            "api_key": "literal",
            "api_key_env": "ENV_NAME",
        },
    )
    with pytest.raises(OcrConfigInvalid, match=r"api_key.*api_key_env"):
        load_ocr_config(workspace)


def test_load_ocr_config_azure_missing_endpoint(workspace: Workspace) -> None:
    write_ocr_config(workspace, {"provider": "azure"})
    with pytest.raises(OcrConfigInvalid, match="endpoint"):
        load_ocr_config(workspace)


def test_load_ocr_config_unknown_provider(workspace: Workspace) -> None:
    write_ocr_config(workspace, {"provider": "magic"})
    with pytest.raises(OcrConfigInvalid, match="provider"):
        load_ocr_config(workspace)


def test_load_ocr_config_aws_happy(workspace: Workspace) -> None:
    write_ocr_config(
        workspace,
        {"provider": "aws", "region": "us-west-2", "profile": "prod"},
    )
    cfg = load_ocr_config(workspace)
    assert cfg.provider == OcrProviderName.AWS
    assert cfg.options["region"] == "us-west-2"
    assert cfg.options["profile"] == "prod"


def test_load_ocr_config_aws_missing_region(workspace: Workspace) -> None:
    write_ocr_config(workspace, {"provider": "aws"})
    with pytest.raises(OcrConfigInvalid, match="region"):
        load_ocr_config(workspace)


def test_load_ocr_config_azure_rejects_aws_fields(workspace: Workspace) -> None:
    """A user who switched provider but left AWS-shaped fields behind
    gets a clear error rather than silent ignore."""
    write_ocr_config(
        workspace,
        {
            "provider": "azure",
            "endpoint": "https://foo.cognitiveservices.azure.com/",
            "region": "us-east-1",  # leftover AWS field
            "profile": "default",  # leftover AWS field
        },
    )
    with pytest.raises(OcrConfigInvalid, match="unknown fields"):
        load_ocr_config(workspace)


def test_load_ocr_config_aws_rejects_azure_fields(workspace: Workspace) -> None:
    write_ocr_config(
        workspace,
        {
            "provider": "aws",
            "region": "us-east-1",
            "endpoint": "https://foo.cognitiveservices.azure.com/",  # leftover Azure field
        },
    )
    with pytest.raises(OcrConfigInvalid, match="unknown fields"):
        load_ocr_config(workspace)


def test_load_ocr_config_rejects_duplicate_provider_key(workspace: Workspace) -> None:
    """A hand-edited config with two `provider` keys is a TOML parse error
    (TOML rejects duplicate keys natively), surfaced as CorruptMetadata."""
    from dgml_core.errors import CorruptMetadata

    workspace.config_path.write_text(
        '[ocr]\nprovider = "azure"\nprovider = "aws"\nregion = "us-east-1"\n',
        encoding="utf-8",
    )
    with pytest.raises(CorruptMetadata):
        load_ocr_config(workspace)


def test_load_ocr_config_rejects_misspelled_field(workspace: Workspace) -> None:
    """A typo like `api_key_envs` (extra s) surfaces clearly rather than
    being silently ignored as 'token auth, no key configured'."""
    write_ocr_config(
        workspace,
        {
            "provider": "azure",
            "endpoint": "https://foo.cognitiveservices.azure.com/",
            "api_key_envs": "FOO_KEY",  # typo
        },
    )
    with pytest.raises(OcrConfigInvalid, match="api_key_envs"):
        load_ocr_config(workspace)


# ---------------------------------------------------------------------------
# Provider ABC contract — factory + extensibility
# ---------------------------------------------------------------------------


def test_make_provider_returns_azure_for_azure_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_AZURE_KEY", "fake-key")
    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,
        options={
            "endpoint": "https://example.cognitiveservices.azure.com/",
            "api_key_env": "TEST_AZURE_KEY",
        },
    )
    provider = make_ocr_provider(cfg)
    assert isinstance(provider, AzureProvider)
    assert isinstance(provider, OcrProvider)
    assert provider.name == OcrProviderName.AZURE


def test_make_provider_returns_aws_for_aws_config(offline_aws: None) -> None:
    cfg = OcrConfig(provider=OcrProviderName.AWS, options={"region": "us-east-1"})
    provider = make_ocr_provider(cfg)
    assert isinstance(provider, AwsProvider)
    assert isinstance(provider, OcrProvider)
    assert provider.name == OcrProviderName.AWS


def test_make_provider_accepts_a_dotted_path(offline_aws: None) -> None:
    """The built-in short names are aliases, not the namespace: naming the same
    class by its dotted path is equivalent."""
    cfg = OcrConfig(provider="dgml_core.ocr_aws:AwsProvider", options={"region": "us-east-1"})
    assert isinstance(make_ocr_provider(cfg), AwsProvider)


def test_make_provider_validates_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hand-built config (a library consumer, not load_ocr_config) is still run
    through the provider's own parse_config."""
    with pytest.raises(OcrConfigInvalid, match="region"):
        make_ocr_provider(OcrConfig(provider=OcrProviderName.AWS))


def test_custom_provider_can_drive_extract_text_ocr(
    workspace: Workspace, text_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Demonstrate the extension path: a brand-new OcrProvider subclass named by
    dotted path is invoked by ``extract_text_ocr`` with no change to dgml at all —
    no enum value to add, no registry to edit."""

    class FakeProvider(OcrProvider):
        name = "fake"
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            cls._check_no_extra_fields(config.options)
            return config

        def __init__(self, config: OcrConfig) -> None:
            self.config = config
            self.calls: list[tuple[int, tuple[int, int]]] = []

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            self.calls.append((page_num, image_dims_px))
            return [{"t": f"fake-page-{page_num}", "l": [0, 0, 1, 1]}]

    provider_path = install_provider(monkeypatch, FakeProvider)

    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    (pages_dir / "page_1.png").write_bytes(make_fake_png(100, 100, b"page-1"))
    (pages_dir / "page_2.png").write_bytes(make_fake_png(100, 100, b"page-2"))

    out_dir = tmp_path / "page_text"
    cfg = OcrConfig(provider=provider_path)
    result = extract_text_ocr(
        text_pdf, out_dir, file_id="fid", page_images_dir=pages_dir, config=cfg
    )

    assert result.pages_written == 2
    assert result.pages_with_words == 2
    assert result.total_words == 2
    p1 = json.loads((out_dir / "page_1.json").read_text())
    assert p1["words"] == [{"t": "fake-page-1", "l": [0, 0, 1, 1]}]
    p2 = json.loads((out_dir / "page_2.json").read_text())
    assert p2["words"] == [{"t": "fake-page-2", "l": [0, 0, 1, 1]}]


def test_ocr_provider_is_abstract() -> None:
    """The ABC itself can't be instantiated — forces subclasses to implement."""
    with pytest.raises(TypeError):
        OcrProvider(OcrConfig(provider=OcrProviderName.AZURE))  # type: ignore[abstract]


def test_image_dimensions_reads_width_height() -> None:
    """Happy path: known-good IHDR chunk → correct (width, height)."""
    from dgml_core.ocr import _image_dimensions

    blob = make_fake_png(1234, 5678)
    assert _image_dimensions(blob) == (1234, 5678)


def test_image_dimensions_rejects_non_png() -> None:
    """Bytes that aren't a PNG raise ValueError."""
    from dgml_core.ocr import _image_dimensions

    with pytest.raises(ValueError, match="PNG"):
        _image_dimensions(b"not a png")


def test_image_dimensions_rejects_truncated_png() -> None:
    """A PNG truncated before the IHDR raises ValueError."""
    from dgml_core.ocr import _image_dimensions

    # Signature only, no IHDR chunk.
    with pytest.raises(ValueError, match="truncated"):
        _image_dimensions(b"\x89PNG\r\n\x1a\n")


def test_image_dimensions_rejects_png_missing_ihdr() -> None:
    """A PNG-signed blob whose first chunk isn't IHDR raises ValueError."""
    from dgml_core.ocr import _image_dimensions

    # 8-byte signature, then a chunk header that isn't IHDR.
    bogus = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00" + b"NOPE" + b"\x00" * 8
    with pytest.raises(ValueError, match="IHDR"):
        _image_dimensions(bogus)


def test_extract_text_ocr_dispatches_in_parallel(
    workspace: Workspace, text_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that pages are actually dispatched concurrently.

    Each provider call holds a barrier until ``max_concurrency`` workers
    arrive, then releases together. If the loop is sequential, the
    barrier never completes and we'd hang — the test would time out.
    A barrier with timeout asserts the parallel path is wired up.
    """
    import threading

    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    n_pages = 4
    for i in range(1, n_pages + 1):
        (pages_dir / f"page_{i}.png").write_bytes(make_fake_png(100, 100, f"p{i}".encode()))

    barrier = threading.Barrier(n_pages, timeout=5.0)

    class BarrierProvider(OcrProvider):
        name = "fake"
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            return config

        def __init__(self, config: OcrConfig) -> None:
            pass

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            # Will deadlock with timeout if loop is sequential.
            barrier.wait()
            return [{"t": f"p{page_num}", "l": [0, 0, 1, 1]}]

    cfg = OcrConfig(provider=install_provider(monkeypatch, BarrierProvider))
    out_dir = tmp_path / "page_text"
    # Default max_concurrency=5 is enough for 4 pages.
    result = extract_text_ocr(
        text_pdf, out_dir, file_id="fid", page_images_dir=pages_dir, config=cfg
    )
    assert result.pages_written == n_pages
    assert result.total_words == n_pages


def test_extract_text_ocr_propagates_first_exception(
    workspace: Workspace, text_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure in any one page propagates as the original exception
    (re-raised after the executor drains). Pages that succeeded before
    the failure may have written their page_text JSON — that's
    documented partial-state behavior."""
    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    (pages_dir / "page_1.png").write_bytes(make_fake_png(100, 100, b"p1"))
    (pages_dir / "page_2.png").write_bytes(make_fake_png(100, 100, b"p2"))
    (pages_dir / "page_3.png").write_bytes(make_fake_png(100, 100, b"p3"))

    class FlakeyProvider(OcrProvider):
        name = "fake"
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            return config

        def __init__(self, config: OcrConfig) -> None:
            pass

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            if page_num == 2:
                from dgml_core.errors import OcrFailed

                raise OcrFailed("simulated provider failure on page 2")
            return []

    from dgml_core.errors import OcrFailed

    cfg = OcrConfig(provider=install_provider(monkeypatch, FlakeyProvider))
    with pytest.raises(OcrFailed, match="page 2"):
        extract_text_ocr(
            text_pdf,
            tmp_path / "page_text",
            file_id="fid",
            page_images_dir=pages_dir,
            config=cfg,
        )


def test_extract_text_ocr_max_concurrency_one_still_works(
    workspace: Workspace, text_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_concurrency=1`` forces effectively-sequential execution
    (one worker thread) but produces identical output."""
    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    (pages_dir / "page_1.png").write_bytes(make_fake_png(100, 100, b"p1"))
    (pages_dir / "page_2.png").write_bytes(make_fake_png(100, 100, b"p2"))

    class Counter(OcrProvider):
        name = "fake"
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            return config

        def __init__(self, config: OcrConfig) -> None:
            self.calls: list[int] = []

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            self.calls.append(page_num)
            return [{"t": str(page_num), "l": [0, 0, 1, 1]}]

    cfg = OcrConfig(provider=install_provider(monkeypatch, Counter))
    out_dir = tmp_path / "page_text"
    result = extract_text_ocr(
        text_pdf,
        out_dir,
        file_id="fid",
        page_images_dir=pages_dir,
        config=cfg,
        max_concurrency=1,
    )
    assert result.pages_written == 2
    assert result.total_words == 2


def test_builtin_names_resolve_to_the_bundled_classes() -> None:
    """The short names are aliases for dotted paths, and each really resolves."""
    assert resolve_provider_class(OcrProviderName.AZURE) is AzureProvider
    assert resolve_provider_class(OcrProviderName.AWS) is AwsProvider
    assert set(BUILTIN_OCR_PROVIDERS) == {p.value for p in OcrProviderName}


def test_resolve_rejects_bare_unknown_name() -> None:
    """A typo'd short name names the alternatives, both the aliases and the
    dotted-path form — it is no longer a closed set the user must pick from."""
    with pytest.raises(OcrConfigInvalid, match="dotted path") as exc:
        resolve_provider_class("magic")
    assert "'aws'" in str(exc.value)


def test_resolve_rejects_unimportable_module() -> None:
    with pytest.raises(OcrConfigInvalid, match="could not import"):
        resolve_provider_class("no_such_module_xyz:Provider")


def test_resolve_rejects_missing_attribute() -> None:
    with pytest.raises(OcrConfigInvalid, match="has no attribute"):
        resolve_provider_class("dgml_core.ocr_aws:NoSuchProvider")


def test_resolve_rejects_non_provider_class() -> None:
    """The base-class check is what stops one provider namespace bleeding into
    another — a storage backend named here fails rather than half-working."""
    with pytest.raises(OcrConfigInvalid, match="not a OcrProvider subclass"):
        resolve_provider_class("dgml_core.storage_local:LocalStore")


def test_load_ocr_config_accepts_a_custom_dotted_path(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end config path: a third-party provider with its own option field,
    validated by its own parse_config."""

    class TesseractProvider(OcrProvider):
        name = "tesseract"
        config_fields = frozenset({"lang"})

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            cls._check_no_extra_fields(config.options)
            if not isinstance(config.options.get("lang"), str):
                raise OcrConfigInvalid("tesseract OCR requires 'ocr.lang'")
            return config

        def __init__(self, config: OcrConfig) -> None:
            pass

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            return []

    path = install_provider(monkeypatch, TesseractProvider)
    write_ocr_config(workspace, {"provider": path, "lang": "eng"})
    cfg = load_ocr_config(workspace)
    assert cfg.provider == path
    assert cfg.options["lang"] == "eng"
    assert isinstance(make_ocr_provider(cfg), TesseractProvider)


def test_universal_fields_are_not_offered_to_a_custom_provider(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_concurrency`` is DGML's own dispatch setting, not a provider option,
    so it is stripped before ``parse_config`` sees the section. A provider that
    declares only its own fields must not have to know the universal ones exist —
    otherwise every third-party provider would break the day DGML adds another."""
    seen: dict[str, Any] = {}

    class NarrowProvider(OcrProvider):
        name = "narrow"
        config_fields = frozenset({"lang"})

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            cls._check_no_extra_fields(config.options)
            seen.update(config.options)
            return config

        def __init__(self, config: OcrConfig) -> None:
            pass

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            return []

    path = install_provider(monkeypatch, NarrowProvider)
    write_ocr_config(workspace, {"provider": path, "lang": "eng", "max_concurrency": 3})

    cfg = load_ocr_config(workspace)
    assert seen == {"lang": "eng"}, "universal keys leaked into the provider's options"
    assert cfg.options == {"lang": "eng"}
    assert cfg.max_concurrency == 3


def test_unknown_field_reports_config_error_even_without_a_declared_name(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``name`` is a plain ClassVar, so an ABC cannot force a third party to declare
    it. A provider that omits it must still produce OCR_CONFIG_INVALID for a user's
    typo — not an AttributeError surfacing as INTERNAL_ERROR and blaming DGML for
    what is the provider author's omission."""

    class NamelessProvider(OcrProvider):
        config_fields = frozenset({"lang"})

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            cls._check_no_extra_fields(config.options)
            return config

        def __init__(self, config: OcrConfig) -> None:
            pass

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            return []

    path = install_provider(monkeypatch, NamelessProvider)
    write_ocr_config(workspace, {"provider": path, "languag": "eng"})

    with pytest.raises(OcrConfigInvalid, match="unknown fields") as exc:
        load_ocr_config(workspace)
    # Falls back to the class name so the message still identifies the provider.
    assert "NamelessProvider" in str(exc.value)


def test_resolve_rejects_the_abstract_base_class() -> None:
    """`issubclass` is satisfied by the ABC itself, which cannot be instantiated.
    Caught at resolve time as a config error rather than surfacing later as a
    TypeError from `replace(None, …)` or from construction — i.e. INTERNAL_ERROR."""
    with pytest.raises(OcrConfigInvalid, match="abstract class") as exc:
        resolve_provider_class("dgml_core.ocr:OcrProvider")
    assert "analyze_image" in str(exc.value)


def test_resolve_rejects_a_half_implemented_subclass(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same guard covers a third party's subclass that left a method out — the
    message names which, so the author knows what to finish."""

    class HalfDone(OcrProvider):
        name = "halfdone"
        config_fields = frozenset()

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            return config

        def __init__(self, config: OcrConfig) -> None:
            pass

        # analyze_image deliberately not implemented

    # Abstract on purpose — that is the thing under test.
    path = install_provider(monkeypatch, HalfDone)  # type: ignore[type-abstract]
    with pytest.raises(OcrConfigInvalid, match="abstract class") as exc:
        resolve_provider_class(path)
    assert "analyze_image" in str(exc.value)


def test_parse_config_that_forgets_to_return_is_a_config_error(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `parse_config` that validates but falls off the end returns None. Without
    this guard the provider is constructed with None as its config — no exception,
    just a provider holding nothing — or `replace(None, …)` raises TypeError."""

    class Forgetful(OcrProvider):
        name = "forgetful"
        config_fields = frozenset()

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            return None  # type: ignore[return-value]

        def __init__(self, config: OcrConfig) -> None:
            pass

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            return []

    path = install_provider(monkeypatch, Forgetful)
    write_ocr_config(workspace, {"provider": path})
    with pytest.raises(OcrConfigInvalid, match="must return an OcrConfig"):
        load_ocr_config(workspace)
    # Same guard on the construction path, which runs parse_config independently.
    with pytest.raises(OcrConfigInvalid, match="must return an OcrConfig"):
        make_ocr_provider(OcrConfig(provider=path))


def test_unknown_fields_rejected_even_if_the_provider_never_checks(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Typo rejection is the framework's job, not opt-in on whether a third party
    remembered to call a private helper — a provider that never checks still gets
    its user's misspelled option rejected."""

    class Trusting(OcrProvider):
        name = "trusting"
        config_fields = frozenset({"lang"})

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            return config  # never calls _check_no_extra_fields

        def __init__(self, config: OcrConfig) -> None:
            pass

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            return []

    path = install_provider(monkeypatch, Trusting)
    write_ocr_config(workspace, {"provider": path, "languag": "eng"})
    with pytest.raises(OcrConfigInvalid, match="unknown fields"):
        load_ocr_config(workspace)


def test_load_ocr_config_runs_custom_provider_validation(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A custom provider's own field rules are enforced at load time — the
    up-front gate `file add` relies on, not deferred to first use."""

    class StrictProvider(OcrProvider):
        name = "strict"
        config_fields = frozenset({"lang"})

        @classmethod
        def parse_config(cls, config: OcrConfig) -> OcrConfig:
            cls._check_no_extra_fields(config.options)
            raise OcrConfigInvalid("strict provider says no")

        def __init__(self, config: OcrConfig) -> None:
            pass

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            return []

    path = install_provider(monkeypatch, StrictProvider)
    write_ocr_config(workspace, {"provider": path, "lang": "eng"})
    with pytest.raises(OcrConfigInvalid, match="strict provider says no"):
        load_ocr_config(workspace)
