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

"""OCR text extraction — abstract provider interface, config loader, dispatcher.

Loads OCR config from ``<workspace>/config.json``, dispatches to the
configured provider, and writes the same per-page JSON shape as
:func:`dgml.text_extraction.extract_text_digital` so downstream code
(``dgml check``, consumers) doesn't care which mode produced the text.

Provider implementations live in sibling modules so this file stays
focused on the abstraction:

- :class:`dgml.ocr_macos.MacosProvider` — Apple Vision (on-device, the
  zero-config default on macOS)
- :class:`dgml.ocr_azure.AzureProvider` — Azure Document Intelligence
- :class:`dgml.ocr_aws.AwsProvider` — AWS Textract

Adding a new provider
---------------------

1. Add a value to :class:`OcrProviderName`.
2. Create a new module ``ocr_<name>.py`` with a subclass of
   :class:`OcrProvider`. Implement ``__init__`` (lazy-import the SDK;
   raise :class:`OcrFailed` if missing) and ``analyze_image``.
3. Wire the subclass into ``_build_registry`` below.
4. Extend :func:`load_ocr_config` to validate any provider-specific
   config fields.

Cloud SDKs are **optional** runtime dependencies — install with
``pip install dgml[aws]`` or ``pip install dgml[azure]``. Calling an OCR
path without the matching extra installed raises :class:`OcrFailed` with
an actionable message.
"""

from __future__ import annotations

import json
import struct
import sys
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from .errors import CorruptMetadata, OcrConfigInvalid, OcrConfigMissing, OcrFailed
from .pages import PAGE_GLOB
from .storage import Workspace, read_config
from .text_extraction import (
    PAGE_TEXT_FILENAME,
    PAGE_TEXT_GLOB,
    ExtractDigitalResult,
)

DEFAULT_OCR_CONCURRENCY = 8


class OcrProviderName(StrEnum):
    """Identifier of an OCR backend, as written in workspace config."""

    AZURE = "azure"
    AWS = "aws"
    MACOS = "macos"


# The provider used on macOS when a workspace declares no OCR config: the
# on-device Apple Vision engine. Off macOS there is no built-in engine, so
# _default_ocr_config raises OcrConfigMissing instead of using this.
DEFAULT_OCR_PROVIDER = OcrProviderName.MACOS


@dataclass(frozen=True)
class OcrConfig:
    """Parsed ``ocr`` section of the workspace config.

    Provider-specific fields are validated by :func:`load_ocr_config`; by
    construction this object is well-formed for the provider it names.
    """

    provider: OcrProviderName
    # Azure
    endpoint: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    # AWS
    region: str | None = None
    profile: str | None = None


def load_ocr_config(workspace: Workspace) -> OcrConfig:
    """Read and validate the ``ocr`` section of ``<workspace>/config.json``.

    Validation of provider-specific fields is delegated to each provider
    class (:meth:`OcrProvider.parse_config`) so this loader stays generic.

    When no config file or no ``ocr`` section is present: on macOS, defaults
    to the on-device provider (:data:`DEFAULT_OCR_PROVIDER`) and emits a
    warning; on other platforms (no built-in OCR engine) raises
    :class:`OcrConfigMissing`. Raises :class:`OcrConfigInvalid` when a
    config exists but is malformed.
    """
    if not workspace.config_path.exists():
        return _default_ocr_config()

    try:
        data = read_config(workspace.config_path)
    except CorruptMetadata as exc:
        raise OcrConfigInvalid(f"{workspace.config_path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise OcrConfigInvalid(f"{workspace.config_path} must contain a JSON object")

    ocr = data.get("ocr")
    if ocr is None:
        return _default_ocr_config()
    if not isinstance(ocr, dict):
        raise OcrConfigInvalid("'ocr' must be a JSON object")

    provider_str = ocr.get("provider")
    valid_providers = [p.value for p in OcrProviderName]
    if provider_str not in valid_providers:
        raise OcrConfigInvalid(
            f"'ocr.provider' must be one of {valid_providers} (got {provider_str!r})"
        )
    provider_name = OcrProviderName(provider_str)

    return _PROVIDERS[provider_name].parse_config(ocr)


def _default_ocr_config() -> OcrConfig:
    """Config used when the workspace declares no OCR provider.

    macOS ships a built-in on-device engine (Apple Vision), so we default
    to it — emitting a warning that we're doing so. Other platforms have no
    built-in OCR, so a missing config is an error the user must fix by
    declaring a cloud provider ('aws' or 'azure').

    Built through the default provider's own parser so it stays the single
    source of truth for that provider's required fields.
    """
    if sys.platform != "darwin":
        raise OcrConfigMissing(
            "no OCR provider configured: add an 'ocr' section to config.json "
            "with provider 'aws' or 'azure' (on-device OCR is only available "
            "on macOS)"
        )
    warnings.warn(
        "no OCR provider configured; defaulting to the on-device macOS provider "
        "(Apple Vision). Set ocr.provider in config.json to silence this warning.",
        stacklevel=2,
    )
    return _PROVIDERS[DEFAULT_OCR_PROVIDER].parse_config({"provider": DEFAULT_OCR_PROVIDER.value})


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


class OcrProvider(ABC):
    """Common interface for cloud OCR backends.

    Implementations are constructed from an :class:`OcrConfig` (which is
    where lazy SDK imports and auth setup live) and implement
    :meth:`analyze_image` for a single rendered page image. The shared
    loop in :func:`extract_text_ocr` handles filesystem I/O, per-page
    JSON output, and result aggregation — providers only need to turn
    image bytes into a list of words.

    Subclasses must declare ``config_fields`` listing the JSON keys they
    accept under ``ocr.*`` (besides the universal ``provider`` key);
    anything else is rejected by :meth:`_check_no_extra_fields` to catch
    typos and stale-after-switching-provider fields.
    """

    name: ClassVar[OcrProviderName]
    config_fields: ClassVar[frozenset[str]]

    @classmethod
    def _check_no_extra_fields(cls, section: dict[str, Any]) -> None:
        """Raise :class:`OcrConfigInvalid` for any keys in ``section`` not
        in ``cls.config_fields`` (or the universal ``provider``)."""
        allowed = cls.config_fields | {"provider"}
        unknown = set(section.keys()) - allowed
        if unknown:
            raise OcrConfigInvalid(
                f"unknown fields in 'ocr' for provider {cls.name.value!r}: "
                f"{sorted(unknown)}. Allowed: {sorted(allowed)}"
            )

    @classmethod
    @abstractmethod
    def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
        """Build an :class:`OcrConfig` from the ``ocr`` section of the
        workspace config (a plain JSON dict).

        Implementations should call :meth:`_check_no_extra_fields` first
        to reject foreign or misspelled keys, then validate the provider's
        own fields. Raise :class:`OcrConfigInvalid` for missing or
        malformed fields. The returned config must have ``provider`` set
        to ``cls.name``.
        """

    @abstractmethod
    def __init__(self, config: OcrConfig) -> None:
        """Build the SDK client. Lazy-import the SDK; raise
        :class:`OcrFailed` with a ``pip install dgml[...]`` hint if it's
        not installed. Raise :class:`AuthError` for credential setup
        failures that happen at construction time."""

    @abstractmethod
    def analyze_image(
        self,
        image_bytes: bytes,
        image_dims_px: tuple[int, int],
        page_num: int,
    ) -> list[dict[str, Any]]:
        """Return ``[{t: text, l: [left, top, right, bottom]}]`` for the image.

        Coordinates are in pixels relative to ``image_dims_px`` (top-left
        origin). Implementations may use or ignore ``image_dims_px``
        depending on whether their API returns normalized or absolute
        coordinates. ``page_num`` is for error-message context only.

        Raise :class:`OcrFailed` for provider/API errors. The shared loop
        does not retry — partial-failure semantics are the caller's
        concern (see :func:`dgml.files.FileStore._extract_text_ocr`).
        """


def make_provider(config: OcrConfig) -> OcrProvider:
    """Instantiate the provider class for ``config.provider``."""
    cls = _PROVIDERS.get(config.provider)
    if cls is None:  # defensive — load_ocr_config validates already
        raise OcrConfigInvalid(f"no provider implementation for {config.provider!r}")
    return cls(config)


def extract_text_ocr(
    pdf_path: Path,
    output_dir: Path,
    *,
    file_id: str,
    page_images_dir: Path,
    config: OcrConfig,
    max_concurrency: int = DEFAULT_OCR_CONCURRENCY,
) -> ExtractDigitalResult:
    """Run OCR using the configured provider and write per-page JSONs.

    All providers operate per rendered page image (``page_images/page_N.png``):
    one provider call per page, no whole-PDF dispatch. This keeps the
    code path symmetric, sidesteps Azure's per-file page ceilings, and
    lets a single bad page surface with a clear page number. Output
    shape matches :func:`extract_text_digital` so consumers and the
    consistency check don't special-case OCR-derived text.

    Page dimensions are read directly from each PNG's IHDR chunk —
    these are the dimensions Textract's normalized bboxes are
    referenced against by definition, so there's no chance of drift
    from a hypothetical mismatch between the PDF's mediabox and what
    ghostscript actually rendered (e.g. CropBox vs MediaBox, rotation
    metadata). ``pdf_path`` is kept in the signature for symmetry with
    :func:`extract_text_digital` but is not opened here.

    Pages are processed concurrently with up to ``max_concurrency``
    threads. The provider's ``analyze_image`` is therefore called from
    multiple threads; both shipped providers wrap stateless API calls
    that are safe to invoke concurrently against the same underlying
    SDK client. On the first per-page failure, pending pages are
    cancelled and the exception is re-raised — partial state on disk is
    possible (some pages may have written page_text JSON before the
    failure) and the caller is responsible for cleanup; today
    :meth:`dgml.files.FileStore._extract_text_ocr` records the failure
    and ``dgml check`` handles re-extraction.

    Raises :class:`OcrFailed` for provider/API errors, :class:`AuthError`
    for credential resolution failures.
    """
    provider = make_provider(config)

    page_image_paths = sorted(page_images_dir.glob(PAGE_GLOB))
    if not page_image_paths:
        raise OcrFailed(
            f"no page images found under {page_images_dir}; OCR requires rendered page images"
        )

    _clear_page_text(output_dir)

    def _process_one_page(path: Path) -> list[dict[str, Any]] | None:
        """Read one page image, derive its pixel dims, call the provider,
        write its page JSON."""
        page_num = _page_num_from_image_name(path.name)
        if page_num is None:
            return None
        image_bytes = path.read_bytes()
        try:
            dims = _image_dimensions(image_bytes)
        except ValueError as exc:
            raise OcrFailed(f"page {page_num}: invalid PNG at {path}: {exc}") from exc
        words = provider.analyze_image(image_bytes, dims, page_num)
        _write_page_json(output_dir, page_num, file_id, dims[0], dims[1], words)
        return words

    pages_written = 0
    pages_with_words = 0
    total_words = 0
    first_exc: BaseException | None = None

    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
        futures = {executor.submit(_process_one_page, p): p for p in page_image_paths}
        for future in as_completed(futures):
            try:
                words = future.result()
            except BaseException as exc:
                if first_exc is None:
                    first_exc = exc
                    # Cancel pending pages so we don't keep hammering the
                    # API after a known failure. In-flight pages finish.
                    for pending in futures:
                        pending.cancel()
                continue
            if words is None:
                continue
            pages_written += 1
            if words:
                pages_with_words += 1
                total_words += len(words)

    if first_exc is not None:
        raise first_exc

    return ExtractDigitalResult(
        pages_written=pages_written,
        pages_with_words=pages_with_words,
        total_words=total_words,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _image_dimensions(data: bytes) -> tuple[int, int]:
    """Parse ``(width, height)`` in pixels from a PNG's IHDR chunk.

    Reading dims from the image bytes (rather than computing them from
    PDF mediabox times DPI) ensures we match exactly what's on disk —
    immune to mediabox/CropBox/rotation mismatches between the PDF
    parser and the renderer.

    PNG layout: 8-byte signature, then a chunk with 4-byte big-endian
    length, 4-byte type ("IHDR"), then payload starting with width
    (uint32 BE) at byte offset 16 and height (uint32 BE) at offset 20.

    Raises ``ValueError`` if the bytes don't start with the PNG
    signature or the IHDR chunk is missing/truncated.
    """
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("not a PNG: missing signature")
    if len(data) < 24:
        raise ValueError("truncated PNG: header less than 24 bytes")
    if data[12:16] != b"IHDR":
        raise ValueError("PNG IHDR chunk missing or not first")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _page_num_from_image_name(name: str) -> int | None:
    """Parse ``page_<N>.png`` → ``N``; return None if the name doesn't match."""
    if not name.startswith("page_") or not name.endswith(".png"):
        return None
    try:
        return int(name[len("page_") : -len(".png")])
    except ValueError:
        return None


def _clear_page_text(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob(PAGE_TEXT_GLOB):
        existing.unlink()


def _write_page_json(
    output_dir: Path,
    page_num: int,
    file_id: str,
    width_px: int,
    height_px: int,
    words: list[dict[str, Any]],
) -> None:
    payload: dict[str, Any] = {
        "file_id": file_id,
        "page": page_num,
        "width": width_px,
        "height": height_px,
        "words": words,
    }
    out_path = output_dir / PAGE_TEXT_FILENAME.format(page=page_num)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Provider registry
#
# Built at module load by a function call so the provider modules' imports
# of this module see a fully-defined OcrProvider ABC and OcrConfig dataclass.
# Doing the import here (rather than at the top of the file) avoids a
# circular dependency: ocr_aws / ocr_azure import OcrProvider from us.
# ---------------------------------------------------------------------------


def _register_providers(
    classes: list[type[OcrProvider]],
) -> dict[OcrProviderName, type[OcrProvider]]:
    """Build a name-keyed registry from a list of provider classes.

    Iterating a list (rather than constructing a dict literal) lets us
    detect collisions: two providers claiming the same
    :class:`OcrProviderName` is a copy-paste bug that would otherwise
    silently overwrite. Raising here keeps the failure at import time,
    before any OCR call.
    """
    registry: dict[OcrProviderName, type[OcrProvider]] = {}
    for cls in classes:
        if cls.name in registry:
            existing = registry[cls.name].__name__
            raise RuntimeError(
                f"duplicate OcrProvider registration for {cls.name.value!r}: "
                f"{existing} and {cls.__name__}"
            )
        registry[cls.name] = cls
    return registry


def _build_registry() -> dict[OcrProviderName, type[OcrProvider]]:
    from .ocr_aws import AwsProvider
    from .ocr_azure import AzureProvider
    from .ocr_macos import MacosProvider

    return _register_providers([AzureProvider, AwsProvider, MacosProvider])


_PROVIDERS: dict[OcrProviderName, type[OcrProvider]] = _build_registry()
