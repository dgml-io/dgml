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

Loads OCR config from ``<workspace>/config.toml``, dispatches to the
configured provider, and writes the same per-page JSON shape as
:func:`dgml.text_extraction.extract_text_digital` so downstream code
(``dgml check``, consumers) doesn't care which mode produced the text.

The providers DGML bundles live in sibling modules so this file stays
focused on the abstraction:

- :class:`dgml_core.ocr_macos.MacosProvider` — Apple Vision (on-device,
  the zero-config default on macOS)
- :class:`dgml_core.ocr_azure.AzureProvider` — Azure Document Intelligence
- :class:`dgml_core.ocr_aws.AwsProvider` — AWS Textract

They hold no privileged position: each is named by the same dotted path a
third party's own provider would use, and :data:`BUILTIN_OCR_PROVIDERS`
maps the short names (``"aws"``, ``"azure"``, ``"macos"``) onto them as a
convenience so existing configs keep working.

Writing your own provider
-------------------------

1. ``pip install dgml-core`` (the wheel — no repo clone).
2. Subclass :class:`OcrProvider`, implementing :meth:`~OcrProvider.parse_config`
   (call :meth:`~dgml_core.provider.ProviderConfigFields._check_no_extra_fields`
   first), ``__init__`` (lazy SDK import — raise
   :class:`~dgml_core.errors.MissingExtra` if missing), and
   :meth:`~OcrProvider.analyze_image`.
3. Make the class importable by the interpreter running dgml.
4. Point ``ocr.provider`` at it::

       [ocr]
       provider = "your_pkg.mod:YourProvider"
       your_option = "…"

   Fields other than ``provider`` are handed to your ``parse_config`` as
   ``config.options``; declare the ones you accept in ``config_fields``.

Cloud SDKs are **optional** runtime dependencies — install with
``pip install dgml[aws]`` or ``pip install dgml[azure]``. Calling an OCR
path without the matching extra installed raises :class:`OcrFailed` with
an actionable message.
"""

from __future__ import annotations

import json
import logging
import struct
import sys
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from .concurrency import map_concurrent
from .config import load_merged_config
from .errors import DgmlError, OcrConfigInvalid, OcrConfigMissing, OcrFailed
from .models_config import ConfigSection
from .pages import PAGE_GLOB
from .provider import ProviderConfigFields, import_provider_class
from .rotation import deskew_page, should_rotate
from .storage import Workspace
from .text_extraction import (
    PAGE_TEXT_FILENAME,
    PAGE_TEXT_GLOB,
    ExtractDigitalResult,
    format_pages,
    is_cid_word,
)

logger = logging.getLogger(__name__)

# Workspace roots whose missing-OCR-provider fallback was already announced this
# process. A bulk add validates and then extracts — two `load_ocr_config` calls
# per file — so without dedup the same line would repeat ~2N times. Keyed by
# workspace root because the missing section is per-workspace state: a process
# that opens a second, equally unconfigured workspace still hears about it.
_WARNED_NO_OCR_PROVIDER: set[Path] = set()

# Pages within a file are OCR'd concurrently (one provider call per page). This
# is the default number of in-flight OCR calls; override per workspace with
# ``ocr.max_concurrency`` in config.json. Kept modest so bulk ingestion stays
# under cloud-provider rate limits.
DEFAULT_OCR_CONCURRENCY = 5


class OcrProviderName(StrEnum):
    """Short name of an OCR backend DGML bundles.

    A convenience spelling, not the namespace: ``ocr.provider`` accepts any
    dotted ``"module.path:ClassName"``, and these three names resolve through
    :data:`BUILTIN_OCR_PROVIDERS` to exactly such a path. Kept so the configs
    people already have — and the docs that taught them — go on working.
    """

    AZURE = "azure"
    AWS = "aws"
    MACOS = "macos"


#: Short name → the dotted path it stands for. The bundled providers are
#: resolved by the same importer as a third party's, so there is no registry of
#: privileged classes here — only aliases.
BUILTIN_OCR_PROVIDERS: dict[str, str] = {
    OcrProviderName.AZURE.value: "dgml_core.ocr_azure:AzureProvider",
    OcrProviderName.AWS.value: "dgml_core.ocr_aws:AwsProvider",
    OcrProviderName.MACOS.value: "dgml_core.ocr_macos:MacosProvider",
}

# The provider used on macOS when a workspace declares no OCR config: the
# on-device Apple Vision engine. Off macOS there is no built-in engine, so
# _default_ocr_config raises OcrConfigMissing instead of using this.
DEFAULT_OCR_PROVIDER = OcrProviderName.MACOS


@dataclass(frozen=True)
class OcrConfig:
    """Parsed ``ocr`` section of the workspace config.

    ``provider`` is the string the config actually wrote — a short built-in name
    or a dotted path — kept verbatim so error messages quote what the user typed.
    ``options`` holds the section's provider-specific fields: ``endpoint`` /
    ``api_key_env`` for Azure, ``region`` / ``profile`` for AWS, whatever a third
    party declares. The universal keys (``provider``, ``max_concurrency``) are
    parsed here and kept out of ``options``, so a provider's ``config_fields``
    only ever has to name its own settings.

    Provider-specific fields are validated by the provider's
    :meth:`OcrProvider.parse_config`, which :func:`load_ocr_config` runs; by
    construction an object that came from there is well-formed for the provider
    it names.
    """

    provider: str
    options: Mapping[str, Any] = field(default_factory=dict)
    # Universal: number of pages OCR'd concurrently (in-flight provider calls).
    max_concurrency: int = DEFAULT_OCR_CONCURRENCY


def resolve_provider_class(provider: str) -> type[OcrProvider]:
    """Import and return the :class:`OcrProvider` subclass named by ``provider``.

    Accepts a built-in short name (resolved through :data:`BUILTIN_OCR_PROVIDERS`)
    or a dotted ``"module.path:ClassName"``. Raises :class:`OcrConfigInvalid` —
    i.e. the documented ``OCR_CONFIG_INVALID`` code — when the string is neither a
    known short name nor a resolvable dotted path, or resolves to something that is
    not an :class:`OcrProvider`.
    """
    dotted = BUILTIN_OCR_PROVIDERS.get(provider)
    if dotted is None:
        if ":" not in provider:
            raise OcrConfigInvalid(
                f"'ocr.provider' must be one of {sorted(BUILTIN_OCR_PROVIDERS)} or a "
                f"dotted path 'module.path:ClassName' (got {provider!r})"
            )
        dotted = provider
    cls: type[OcrProvider] = import_provider_class(
        dotted, OcrProvider, kind="ocr", error=OcrConfigInvalid
    )
    return cls


def load_ocr_config(workspace: Workspace) -> OcrConfig:
    """Read and validate the ``ocr`` section of ``<workspace>/config.toml``.

    Validation of provider-specific fields is delegated to each provider
    class (:meth:`OcrProvider.parse_config`) so this loader stays generic.

    Unlike :func:`dgml_core.conversion.load_conversion_config`, this **does**
    resolve the provider class and run its validation eagerly, importing a
    third-party module here. That is deliberate: ``file add`` validates OCR config
    before it touches the filesystem, so a bad ``[ocr]`` table is rejected with no
    record created (see :meth:`dgml_core.files.FileStore.add`). A workspace names
    exactly one OCR provider, so there is no fan-out cost to importing it.

    When the merged config has no ``ocr`` section — or an empty one: on macOS,
    defaults to the on-device provider (:data:`DEFAULT_OCR_PROVIDER`) and emits a
    warning; on other platforms (no built-in OCR engine) raises
    :class:`OcrConfigMissing`. Raises :class:`OcrConfigInvalid` when a
    config exists but is malformed.
    """
    ocr = load_merged_config(workspace).get(ConfigSection.OCR)
    if not ocr:
        # Absent or empty. Unlike `style` / `text_extraction`, this section's mere
        # presence carries no meaning — `provider` is what selects a backend — so
        # a bare `[ocr]` is the same as none at all rather than a misconfiguration.
        return _default_ocr_config(workspace)
    if not isinstance(ocr, dict):
        raise OcrConfigInvalid("'ocr' must be a table")

    provider = ocr.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise OcrConfigInvalid(
            f"'ocr.provider' must be a non-empty string — one of "
            f"{sorted(BUILTIN_OCR_PROVIDERS)} or a dotted path 'module.path:ClassName' "
            f"(got {provider!r})"
        )
    return _parse_section(provider, ocr)


#: Section-level keys that belong to DGML's own dispatch loop rather than to any
#: provider. Stripped from ``options`` so a provider's ``config_fields`` never has
#: to name them — and so adding one later cannot break a third party's validation.
UNIVERSAL_OCR_FIELDS = frozenset({"provider", "max_concurrency"})


def _parse_section(provider: str, section: Mapping[str, Any]) -> OcrConfig:
    """Resolve ``provider`` and let it validate the section's provider-specific
    fields, then attach the universal ones DGML parses itself."""
    cls = resolve_provider_class(provider)
    options = {k: v for k, v in section.items() if k not in UNIVERSAL_OCR_FIELDS}
    cfg = _run_parse_config(cls, OcrConfig(provider=provider, options=options))
    return replace(cfg, max_concurrency=_parse_max_concurrency(section))


def _run_parse_config(cls: type[OcrProvider], config: OcrConfig) -> OcrConfig:
    """Reject unknown option keys, run ``cls.parse_config``, and check what it gave back.

    The unknown-key check runs **here** rather than only inside each provider, so
    rejecting a user's typo is guaranteed by the framework instead of being opt-in on
    whether a third party remembered to call it. Bundled providers still call it
    themselves; it is idempotent, and leaving it there keeps ``parse_config`` correct
    for anyone invoking it directly.

    Validating the return matters because the failure is otherwise silent: a
    ``parse_config`` that ends without ``return config`` yields ``None``, and the
    provider is then constructed with ``None`` as its config — no error, just a
    provider holding nothing.
    """
    cls._check_no_extra_fields(config.options)
    parsed = cls.parse_config(config)
    if not isinstance(parsed, OcrConfig):
        raise OcrConfigInvalid(
            f"{cls._describe()!r}.parse_config must return an OcrConfig, got "
            f"{type(parsed).__name__} — a parse_config that validates but forgets to "
            f"`return config` lands here."
        )
    return parsed


def _parse_max_concurrency(ocr: Mapping[str, Any]) -> int:
    """Read the optional universal ``ocr.max_concurrency`` (a positive int),
    defaulting to :data:`DEFAULT_OCR_CONCURRENCY`."""
    raw = ocr.get("max_concurrency")
    if raw is None:
        return DEFAULT_OCR_CONCURRENCY
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        raise OcrConfigInvalid(f"'ocr.max_concurrency' must be a positive integer (got {raw!r})")
    return raw


def _default_ocr_config(workspace: Workspace) -> OcrConfig:
    """Config used when the workspace declares no OCR provider.

    macOS ships a built-in on-device engine (Apple Vision), so we default
    to it — warning that we're doing so, once per workspace per process
    (see :data:`_WARNED_NO_OCR_PROVIDER`). Other platforms have no
    built-in OCR, so a missing config is an error the user must fix by
    declaring a provider.

    Built through the default provider's own parser so it stays the single
    source of truth for that provider's required fields.
    """
    if sys.platform != "darwin":
        raise OcrConfigMissing(
            "no OCR provider configured: add an 'ocr' section to config.toml "
            "with provider 'aws' or 'azure' — or your own 'module.path:ClassName' "
            "(on-device OCR is only available on macOS)"
        )
    if workspace.root not in _WARNED_NO_OCR_PROVIDER:
        _WARNED_NO_OCR_PROVIDER.add(workspace.root)
        # Names the workspace because the dedup is per workspace: a process
        # holding several must be able to tell the resulting lines apart.
        logger.warning(
            "no OCR provider configured for workspace (%s); defaulting to the "
            "on-device macOS provider (Apple Vision). Set ocr.provider in "
            "config.toml to silence this warning.",
            workspace.root,
        )
    return _parse_section(DEFAULT_OCR_PROVIDER.value, {})


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OcrPageResult:
    """A provider's OCR output for one rendered page image.

    ``words`` is the token list (``[{t, l:[left,top,right,bottom]}]``) — the
    same shape :meth:`OcrProvider.analyze_image` has always returned. ``angle``
    is the page-content skew the provider reported, in degrees, clockwise-
    positive (Azure DI's ``page.angle`` convention). ``0.0`` means "no skew" or
    "this provider doesn't report skew". A significant angle drives deskew of
    the page image and word boxes in :func:`extract_text_ocr`.

    Providers may still return a bare ``list`` of word dicts from
    ``analyze_image`` (treated as ``angle=0.0``); this dataclass is the richer
    form used by providers that report skew.
    """

    words: list[dict[str, Any]]
    angle: float = 0.0


def _as_page_result(ret: list[dict[str, Any]] | OcrPageResult) -> OcrPageResult:
    """Normalise an ``analyze_image`` return into an :class:`OcrPageResult`.

    Accepts either the richer dataclass or a bare word list (angle 0), so
    providers that don't report skew — and the fakes in the test-suite — need
    no changes.
    """
    if isinstance(ret, OcrPageResult):
        return ret
    return OcrPageResult(words=ret, angle=0.0)


class OcrProvider(ProviderConfigFields, ABC):
    """Common interface for OCR backends.

    Implementations are constructed from an :class:`OcrConfig` (which is
    where lazy SDK imports and auth setup live) and implement
    :meth:`analyze_image` for a single rendered page image. The shared
    loop in :func:`extract_text_ocr` handles filesystem I/O, per-page
    JSON output, and result aggregation — providers only need to turn
    image bytes into a list of words.

    Subclasses must declare ``config_fields`` listing the keys they accept
    under ``ocr.*`` (besides the universal keys in
    :data:`UNIVERSAL_OCR_FIELDS`, which never reach a provider); anything else
    is rejected by
    :meth:`~dgml_core.provider.ProviderConfigFields._check_no_extra_fields`
    to catch typos and stale-after-switching-provider fields. That machinery
    is shared with the ``[storage]`` and ``[workspaces]`` providers; the two
    ClassVars below bind it to this section so failures carry the ``ocr``
    vocabulary and the ``OCR_CONFIG_INVALID`` code.

    ``name`` is the provider's short name for failure messages. For the
    bundled providers it is the :class:`OcrProviderName` alias; a third
    party's can be any short identifier.
    """

    config_fields: ClassVar[frozenset[str]]
    config_section: ClassVar[str] = "ocr"
    config_error: ClassVar[type[DgmlError]] = OcrConfigInvalid

    @classmethod
    @abstractmethod
    def parse_config(cls, config: OcrConfig) -> OcrConfig:
        """Validate the provider's option fields and return the (possibly
        normalized) config.

        ``config.options`` is the ``ocr`` section minus the universal keys
        (:data:`UNIVERSAL_OCR_FIELDS`).
        Implementations should call :meth:`_check_no_extra_fields` on it first
        to reject foreign or misspelled keys, then validate the provider's own
        fields, raising :class:`OcrConfigInvalid` for missing or malformed ones.
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
    ) -> list[dict[str, Any]] | OcrPageResult:
        """Return the words found in the image (and, optionally, its skew).

        Return either a bare ``[{t: text, l: [left, top, right, bottom]}]``
        list (the historical shape; skew assumed 0) or an
        :class:`OcrPageResult` carrying the same words plus the page-content
        ``angle`` in degrees (clockwise-positive) for providers that report
        skew. When the angle is significant the shared loop deskews the page
        image and rotates these boxes to match (see :func:`extract_text_ocr`).

        Coordinates are in pixels relative to ``image_dims_px`` (top-left
        origin). Implementations may use or ignore ``image_dims_px``
        depending on whether their API returns normalized or absolute
        coordinates. ``page_num`` is for error-message context only.

        Raise :class:`OcrFailed` for provider/API errors. The shared loop
        does not retry — partial-failure semantics are the caller's
        concern (see :func:`dgml.files.FileStore._extract_text_ocr`).
        """


def make_ocr_provider(config: OcrConfig) -> OcrProvider:
    """Instantiate the :class:`OcrProvider` named by ``config`` (resolve provider →
    ``parse_config`` → construct, where the provider's lazy SDK import happens).

    Re-runs ``parse_config`` rather than trusting the caller: it is what validates a
    config built by hand (a library consumer, a test), and for one that came from
    :func:`load_ocr_config` it is a no-op — which is why ``parse_config`` is required
    to be pure and idempotent."""
    cls = resolve_provider_class(config.provider)
    return cls(_run_parse_config(cls, config))


def extract_text_ocr(
    pdf_path: Path,
    output_dir: Path,
    *,
    file_id: str,
    page_images_dir: Path,
    config: OcrConfig,
    max_concurrency: int | None = None,
    pages: Collection[int] | None = None,
) -> ExtractDigitalResult:
    """Run OCR using the configured provider and write per-page JSONs.

    ``pages`` (1-based) limits the run to those pages and leaves every other
    ``page_N.json`` in ``output_dir`` as it is: the digital path's per-page
    fallback (see :func:`recover_unusable_pages`). A requested page with no
    page image raises :class:`OcrFailed`. Without it, every page is OCR'd and
    stale page JSONs are cleared first.

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

    Pages are dispatched via :func:`dgml_core.concurrency.map_concurrent`
    with up to ``max_concurrency`` workers (when ``None``,
    ``config.max_concurrency`` — set by ``ocr.max_concurrency`` in
    config.json, default :data:`DEFAULT_OCR_CONCURRENCY`). The provider's
    ``analyze_image`` is therefore called from multiple threads; both
    shipped providers wrap stateless API calls that are safe to invoke
    concurrently against the same underlying SDK client. On the first
    per-page failure, pending pages are cancelled and the exception is
    re-raised — partial state on disk is possible (some pages may have
    written page_text JSON before the failure) and the caller is
    responsible for cleanup; today
    :meth:`dgml.files.FileStore._extract_text_ocr` records the failure
    and ``dgml check`` handles re-extraction.

    Raises :class:`OcrFailed` for provider/API errors, :class:`AuthError`
    for credential resolution failures.
    """
    provider = make_ocr_provider(config)
    workers = config.max_concurrency if max_concurrency is None else max_concurrency

    page_image_paths = sorted(page_images_dir.glob(PAGE_GLOB))
    if pages is not None:
        wanted = set(pages)
        page_image_paths = [
            p for p in page_image_paths if _page_num_from_image_name(p.name) in wanted
        ]
        found = {_page_num_from_image_name(p.name) for p in page_image_paths}
        missing = wanted - found
        if missing:
            raise OcrFailed(
                f"no page image for {format_pages(missing)} under {page_images_dir}; "
                "OCR requires rendered page images"
            )
    if not page_image_paths:
        raise OcrFailed(
            f"no page images found under {page_images_dir}; OCR requires rendered page images"
        )

    if pages is None:
        _clear_page_text(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    def _process_one_page(path: Path) -> list[dict[str, Any]] | None:
        """Read one page image, derive its pixel dims, call the provider, and
        write its page JSON — deskewing the image and word boxes first when the
        provider reports a significant page skew."""
        page_num = _page_num_from_image_name(path.name)
        if page_num is None:
            return None
        image_bytes = path.read_bytes()
        try:
            dims = _image_dimensions(image_bytes)
        except ValueError as exc:
            raise OcrFailed(f"page {page_num}: invalid PNG at {path}: {exc}") from exc
        page = _as_page_result(provider.analyze_image(image_bytes, dims, page_num))
        words = page.words
        rotation: float | None = None
        if should_rotate(page.angle):
            # Correct the skew: rotate the page image and its word boxes by the
            # same transform, then rewrite the canonical page image in place so
            # grounding / generation / export all see the deskewed page. The
            # recorded dims below come from the rotated image.
            image_bytes, dims, words = deskew_page(image_bytes, dims, words, page.angle)
            path.write_bytes(image_bytes)
            rotation = page.angle
        _write_page_json(output_dir, page_num, file_id, dims[0], dims[1], words, rotation=rotation)
        return words

    pages_written = 0
    pages_with_words = 0
    total_words = 0
    # Folded on this thread, in page order; the counters are order-independent
    # sums and each worker has already written its own page JSON.
    for words in map_concurrent(_process_one_page, page_image_paths, max_workers=workers):
        if words is None:
            continue
        pages_written += 1
        if words:
            pages_with_words += 1
            total_words += len(words)

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


def recover_unusable_pages(
    workspace: Workspace,
    pdf_path: Path,
    output_dir: Path,
    result: ExtractDigitalResult,
    *,
    file_id: str,
    pages_prefix: str,
) -> ExtractDigitalResult:
    """OCR the pages a digital extraction flagged as unusable.

    ``result`` is what :func:`~dgml_core.text_extraction.extract_text_digital`
    returned for ``output_dir``. The pages in its ``defects`` (fonts with no
    Unicode mapping), and only those, are OCR'd with the workspace's provider,
    resolved exactly as ``--text-mode ocr`` resolves it (:func:`load_ocr_config`:
    ``[ocr]`` in config.toml, or the on-device default on macOS), from the page
    images under ``pages_prefix``. A healthy result is returned untouched
    without reading any OCR config.

    When OCR cannot run (no provider, a bad ``[ocr]`` table, a missing extra, a
    provider failure) the add still lands: the ``(cid:N)`` words are dropped,
    since they match nothing a model transcribes, and the page keeps any words
    that did resolve. One WARNING per file names the pages and the remedy, and
    the returned ``defects`` / ``ocr_fallback_error`` let
    :func:`~dgml_core.text_extraction.classify_extraction_outcome` record it.
    """
    if not result.defects:
        return result
    pages = sorted(result.defects)
    # Read before OCR runs: a provider failure part way through can leave some
    # of these pages already rewritten.
    originals = {
        page: json.loads(
            (output_dir / PAGE_TEXT_FILENAME.format(page=page)).read_text(encoding="utf-8")
        )
        for page in pages
    }
    try:
        config = load_ocr_config(workspace)
        with workspace.blobs.materialize_dir(pages_prefix) as pages_dir:
            extract_text_ocr(
                pdf_path,
                output_dir,
                file_id=file_id,
                page_images_dir=pages_dir,
                config=config,
                pages=pages,
            )
    except DgmlError as exc:
        for page, data in originals.items():
            data["words"] = [w for w in data["words"] if not is_cid_word(w.get("t"))]
            (output_dir / PAGE_TEXT_FILENAME.format(page=page)).write_text(
                json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
        logger.warning(
            "file_id=%s: %s/%s pages use fonts with no Unicode mapping (%s) and OCR "
            "could not run: %s. Their (cid:N) words were dropped. Configure an OCR "
            "provider and run `dgml check --retry-errors`, or re-add the file with "
            "--text-mode ocr",
            file_id,
            len(pages),
            result.pages_written,
            format_pages(pages),
            exc,
        )
        return replace(
            _tally(output_dir, result.pages_written),
            defects=dict(result.defects),
            ocr_fallback_error=str(exc),
        )
    logger.info(
        "notice: file_id=%s: %s use fonts with no Unicode mapping; took them from OCR (%s)",
        file_id,
        format_pages(pages),
        config.provider,
    )
    return replace(_tally(output_dir, result.pages_written), ocr_fallback_pages=pages)


def _tally(output_dir: Path, pages_written: int) -> ExtractDigitalResult:
    """Recount words over the page JSONs in ``output_dir``."""
    pages_with_words = 0
    total_words = 0
    for path in output_dir.glob(PAGE_TEXT_GLOB):
        words = json.loads(path.read_text(encoding="utf-8")).get("words") or []
        if words:
            pages_with_words += 1
            total_words += len(words)
    return ExtractDigitalResult(
        pages_written=pages_written,
        pages_with_words=pages_with_words,
        total_words=total_words,
    )


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
    *,
    rotation: float | None = None,
) -> None:
    payload: dict[str, Any] = {
        "file_id": file_id,
        "page": page_num,
        "width": width_px,
        "height": height_px,
        "words": words,
    }
    # Present only when the page was deskewed: the clockwise skew (degrees) that
    # was corrected. ``width``/``height`` above are the post-rotation dims and the
    # word boxes are already in the rotated frame. It's a provenance note and the
    # signal the hybrid merge uses to rotate the digital boxes into this same
    # deskewed frame before merging (see ``_merge_into``).
    if rotation is not None:
        payload["rotation"] = round(float(rotation), 4)
    out_path = output_dir / PAGE_TEXT_FILENAME.format(page=page_num)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


# No provider registry lives here any more. Classes are resolved by dotted path
# at use time (:func:`resolve_provider_class`), which is what lets a third party
# name their own — and it also removes the import cycle the old import-time
# registry had to work around, since ocr_aws / ocr_azure / ocr_macos import this
# module rather than the other way round.
