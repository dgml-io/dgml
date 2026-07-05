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
import sys
from pathlib import Path
from typing import Any

import pytest
from dgml_core.errors import OcrConfigInvalid, OcrConfigMissing
from dgml_core.ocr import (
    OcrConfig,
    OcrProvider,
    OcrProviderName,
    extract_text_ocr,
    load_ocr_config,
    make_provider,
)
from dgml_core.ocr_aws import AwsProvider
from dgml_core.ocr_azure import AzureProvider
from dgml_core.storage import Workspace

from .conftest import make_fake_png, write_ocr_config

# ---------------------------------------------------------------------------
# load_ocr_config
# ---------------------------------------------------------------------------


def test_load_ocr_config_defaults_to_macos_on_darwin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On macOS, a missing config defaults to the on-device provider and
    warns that it's doing so."""
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.warns(UserWarning, match="defaulting to the on-device macOS"):
        cfg = load_ocr_config(workspace)
    assert cfg.provider is OcrProviderName.MACOS


def test_load_ocr_config_no_config_raises_off_darwin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off macOS there is no built-in OCR engine, so a missing config is an
    error the user must fix."""
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(OcrConfigMissing):
        load_ocr_config(workspace)


def test_load_ocr_config_no_ocr_section_defaults_to_macos_on_darwin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    workspace.config_path.write_text(json.dumps({"other": {}}), encoding="utf-8")
    with pytest.warns(UserWarning, match="defaulting to the on-device macOS"):
        cfg = load_ocr_config(workspace)
    assert cfg.provider is OcrProviderName.MACOS


def test_load_ocr_config_no_ocr_section_raises_off_darwin(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    workspace.config_path.write_text(json.dumps({"other": {}}), encoding="utf-8")
    with pytest.raises(OcrConfigMissing):
        load_ocr_config(workspace)


def test_load_ocr_config_invalid_json(workspace: Workspace) -> None:
    workspace.config_path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(OcrConfigInvalid):
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
    assert cfg.provider is OcrProviderName.AZURE
    assert cfg.endpoint == "https://foo.cognitiveservices.azure.com/"
    assert cfg.api_key_env == "FOO_KEY"


def test_load_ocr_config_azure_no_key_env(workspace: Workspace) -> None:
    """Token auth (no api_key_env / api_key) is valid — falls through to
    DefaultAzureCredential."""
    write_ocr_config(
        workspace, {"provider": "azure", "endpoint": "https://foo.cognitiveservices.azure.com/"}
    )
    cfg = load_ocr_config(workspace)
    assert cfg.api_key is None
    assert cfg.api_key_env is None


def test_load_ocr_config_azure_literal_api_key(workspace: Workspace) -> None:
    """A literal api_key in config is accepted (developers may put keys
    directly in workspace config.json — it isn't checked in)."""
    write_ocr_config(
        workspace,
        {
            "provider": "azure",
            "endpoint": "https://foo.cognitiveservices.azure.com/",
            "api_key": "literal-test-key",
        },
    )
    cfg = load_ocr_config(workspace)
    assert cfg.api_key == "literal-test-key"
    assert cfg.api_key_env is None


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
    assert cfg.provider is OcrProviderName.AWS
    assert cfg.region == "us-west-2"
    assert cfg.profile == "prod"


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
    """A hand-edited config with two `provider` keys would silently
    resolve to the last one under plain json.loads. Our duplicate-key
    rejection at the read_json layer surfaces it as OcrConfigInvalid
    (via CorruptMetadata)."""
    workspace.config_path.write_text(
        '{"ocr": {"provider": "azure", "provider": "aws", "region": "us-east-1"}}',
        encoding="utf-8",
    )
    with pytest.raises(OcrConfigInvalid, match="duplicate key"):
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
        endpoint="https://example.cognitiveservices.azure.com/",
        api_key_env="TEST_AZURE_KEY",
    )
    provider = make_provider(cfg)
    assert isinstance(provider, AzureProvider)
    assert isinstance(provider, OcrProvider)
    assert provider.name is OcrProviderName.AZURE


def test_make_provider_returns_aws_for_aws_config() -> None:
    cfg = OcrConfig(provider=OcrProviderName.AWS, region="us-east-1")
    provider = make_provider(cfg)
    assert isinstance(provider, AwsProvider)
    assert isinstance(provider, OcrProvider)
    assert provider.name is OcrProviderName.AWS


def test_custom_provider_can_drive_extract_text_ocr(
    workspace: Workspace, text_pdf: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Demonstrate the extension path: a brand-new OcrProvider subclass
    registered in the dispatch table is invoked by ``extract_text_ocr``
    without any other code change."""

    class FakeProvider(OcrProvider):
        name = OcrProviderName.AZURE  # reuse an existing enum value for the test
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
            return OcrConfig(provider=cls.name)

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

    from dgml_core.ocr import _PROVIDERS

    monkeypatch.setitem(_PROVIDERS, OcrProviderName.AZURE, FakeProvider)

    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    (pages_dir / "page_1.png").write_bytes(make_fake_png(100, 100, b"page-1"))
    (pages_dir / "page_2.png").write_bytes(make_fake_png(100, 100, b"page-2"))

    out_dir = tmp_path / "page_text"
    cfg = OcrConfig(
        provider=OcrProviderName.AZURE,  # routes through FakeProvider via the registry
        endpoint="https://does-not-matter/",
    )
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
        name = OcrProviderName.AZURE
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
            return OcrConfig(provider=cls.name)

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

    from dgml_core.ocr import _PROVIDERS

    monkeypatch.setitem(_PROVIDERS, OcrProviderName.AZURE, BarrierProvider)

    cfg = OcrConfig(provider=OcrProviderName.AZURE, endpoint="https://x/")
    out_dir = tmp_path / "page_text"
    # Default max_concurrency=8 is enough for 4 pages.
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
        name = OcrProviderName.AZURE
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
            return OcrConfig(provider=cls.name)

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
    from dgml_core.ocr import _PROVIDERS

    monkeypatch.setitem(_PROVIDERS, OcrProviderName.AZURE, FlakeyProvider)

    cfg = OcrConfig(provider=OcrProviderName.AZURE, endpoint="https://x/")
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
        name = OcrProviderName.AZURE
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
            return OcrConfig(provider=cls.name)

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

    from dgml_core.ocr import _PROVIDERS

    monkeypatch.setitem(_PROVIDERS, OcrProviderName.AZURE, Counter)
    cfg = OcrConfig(provider=OcrProviderName.AZURE, endpoint="https://x/")
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


def test_registry_rejects_duplicate_provider_names() -> None:
    """Catches the copy-paste-the-class footgun: two provider classes
    claiming the same OcrProviderName would silently overwrite each
    other in a dict literal. Detect at registry-build time instead."""
    from dgml_core.ocr import _register_providers
    from dgml_core.ocr_azure import AzureProvider

    class DupeAzure(OcrProvider):
        """A misconfigured provider that claims an already-taken name."""

        name = OcrProviderName.AZURE
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
            return OcrConfig(provider=cls.name)

        def __init__(self, config: OcrConfig) -> None:
            pass

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            return []

    with pytest.raises(RuntimeError, match="duplicate OcrProvider registration"):
        _register_providers([AzureProvider, DupeAzure])
