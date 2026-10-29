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

"""Document-to-PDF conversion — abstract provider interface, config loader, dispatcher.

Source documents (docx, xlsx, pptx, …) are converted to PDF before the rest of the
pipeline, which is PDF-only. Which converter handles each *format family*
(``docx``, ``xlsx``, ``pptx``) is chosen per workspace in the ``conversion`` section of
``<workspace>/config.toml``::

    {
      "conversion": {
        "docx": {"provider": "translators_pdf.libreoffice:LibreOfficeConverter"},
        "xlsx": {"provider": "translators_pdf.xlsx:XlsxIslandsConverter", "row_gap": 4}
      }
    }

There are **no defaults**: a family with no configured converter is unsupported,
and ``.pdf`` always works because it needs no converter.

This module ships only the abstraction and the resolver — no concrete
converters. A ``provider`` is a dotted path ``"module.path:ClassName"`` that
:func:`make_converter` imports at runtime and checks is a :class:`DocConverter`
subclass. The converters DGML provides live in the separately-installable
``translators-pdf`` package and are referenced by dotted path exactly like a
user's own; there is deliberately no built-in registry of concrete classes.

Writing your own converter
--------------------------

1. ``pip install dgml`` (the wheel — no repo clone).
2. Subclass :class:`DocConverter`, implementing :meth:`~DocConverter.parse_config`
   (call :meth:`~DocConverter._check_no_extra_fields` first), ``__init__`` (lazy
   SDK/binary import — raise :class:`dgml.errors.ConversionFailed` if missing),
   and :meth:`~DocConverter.to_pdf`.
3. Make the class importable by the interpreter running dgml.
4. Set the family's ``provider`` to ``"your_pkg.mod:YourConverter"``.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from .config import load_merged_config
from .errors import ConversionConfigInvalid, UnsupportedFileType
from .models_config import ConfigSection
from .storage import Workspace


class FormatFamily(StrEnum):
    """A convertible source format family: the ``conversion.<family>`` key one
    converter is configured under. A member is its TOML key (``"docx"``)."""

    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"


# Format families and the source suffixes that belong to each. ``.pdf`` is
# absent on purpose: it needs no converter. Shared with the add-time ingestion
# gate (dgml_core.files) and the generation dispatch (dgml_core.generation.document) so
# the supported-input surface lives in one place.
FAMILY_BY_SUFFIX: dict[str, FormatFamily] = {
    ".docx": FormatFamily.DOCX,
    ".doc": FormatFamily.DOCX,
    ".xlsx": FormatFamily.XLSX,
    ".xls": FormatFamily.XLSX,
    ".pptx": FormatFamily.PPTX,
    ".ppt": FormatFamily.PPTX,
}

# The convertible suffixes as user-facing text (``.docx/.doc/…``), for the
# add-time error and the CLI help.
CONVERTIBLE_SUFFIXES_TEXT = "/".join(FAMILY_BY_SUFFIX)


def family_for_suffix(suffix: str) -> FormatFamily | None:
    """Return the conversion family for a file suffix (e.g. ``.docx`` →
    :attr:`FormatFamily.DOCX`), or ``None`` if the suffix is not a known
    convertible source format."""
    return FAMILY_BY_SUFFIX.get(suffix.lower())


@dataclass(frozen=True)
class ConverterConfig:
    """A ``conversion.<family>`` section of the workspace config.

    ``provider`` is the dotted path identifying the converter class. ``options``
    holds the section's remaining (non-``provider``) fields verbatim — e.g.
    ``row_gap`` for the xlsx island renderer, ``command`` for the generic
    command converter.

    As produced by :func:`load_conversion_config` this object is validated only
    for *shape* (``provider`` is a non-empty string); the provider's own fields
    are validated lazily by :meth:`DocConverter.parse_config` when the converter
    is built in :func:`make_converter`, so loading the config never imports the
    converter module.
    """

    provider: str
    options: Mapping[str, Any] = field(default_factory=dict)


class DocConverter(ABC):
    """Common interface for document-to-PDF converters.

    Implementations are constructed from a :class:`ConverterConfig` (where lazy
    SDK/binary imports and setup live) and implement :meth:`to_pdf` for a single
    source document.

    Subclasses declare ``config_fields`` listing the JSON keys they accept under
    ``conversion.<family>.*`` (besides the universal ``provider`` key); anything
    else is rejected by :meth:`_check_no_extra_fields` to catch typos and
    stale-after-switching-provider fields. ``input_formats`` lists the suffixes
    a converter handles, compared case-insensitively and with or without the
    leading dot; a document of any other suffix is refused before the converter
    is built (see :func:`require_converter`). An empty set means "any", for
    format-agnostic converters like the command runner.
    """

    name: ClassVar[str]
    input_formats: ClassVar[frozenset[str]]
    config_fields: ClassVar[frozenset[str]]

    @classmethod
    def _check_no_extra_fields(cls, section: Mapping[str, Any]) -> None:
        """Raise :class:`ConversionConfigInvalid` for any keys in ``section`` not
        in ``cls.config_fields`` (or the universal ``provider``)."""
        allowed = cls.config_fields | {"provider"}
        unknown = set(section.keys()) - allowed
        if unknown:
            raise ConversionConfigInvalid(
                f"unknown fields in 'conversion' for provider {cls.name!r}: "
                f"{sorted(unknown)}. Allowed: {sorted(allowed)}"
            )

    @classmethod
    @abstractmethod
    def parse_config(cls, section: Mapping[str, Any]) -> ConverterConfig:
        """Build a :class:`ConverterConfig` from a ``conversion.<family>`` section.

        Implementations should call :meth:`_check_no_extra_fields` first, then
        validate the provider's own fields. Raise :class:`ConversionConfigInvalid`
        for missing or malformed fields. The returned config's ``provider`` must
        be the section's ``provider`` string.
        """

    @abstractmethod
    def __init__(self, config: ConverterConfig) -> None:
        """Set the converter up from ``config``. Lazy-import any SDK and raise
        :class:`dgml.errors.ConversionFailed` with an actionable message if a
        required SDK or external binary is missing."""

    @abstractmethod
    def to_pdf(self, path: Path) -> bytes:
        """Convert the document at ``path`` to PDF and return the bytes.

        Raise :class:`dgml.errors.ConversionFailed` on any conversion error
        (missing binary, non-zero exit, empty/absent output)."""


def _resolve_converter_class(provider: str) -> type[DocConverter]:
    """Import and return the :class:`DocConverter` subclass named by ``provider``.

    ``provider`` must be a dotted path ``"module.path:ClassName"``. Raises
    :class:`ConversionConfigInvalid` if the string is malformed, the module or
    attribute can't be imported, or the target is not a ``DocConverter``
    subclass.
    """
    if ":" not in provider:
        raise ConversionConfigInvalid(
            f"converter provider must be a dotted path 'module.path:ClassName' "
            f"(got {provider!r}). See the translators-pdf package for ready-made "
            f"converters, e.g. 'translators_pdf.libreoffice:LibreOfficeConverter'."
        )
    module_path, _, class_name = provider.partition(":")
    if not module_path or not class_name:
        raise ConversionConfigInvalid(
            f"converter provider {provider!r} must have the form 'module.path:ClassName'"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ConversionConfigInvalid(
            f"could not import converter module {module_path!r} for provider "
            f"{provider!r}: {exc}. Is the package installed in this environment?"
        ) from exc
    try:
        obj = getattr(module, class_name)
    except AttributeError as exc:
        raise ConversionConfigInvalid(
            f"module {module_path!r} has no attribute {class_name!r} (provider {provider!r})"
        ) from exc
    if not (isinstance(obj, type) and issubclass(obj, DocConverter)):
        raise ConversionConfigInvalid(
            f"provider {provider!r} resolved to {obj!r}, which is not a DocConverter subclass"
        )
    return obj


def make_converter(config: ConverterConfig) -> DocConverter:
    """Instantiate the converter named by ``config``.

    Resolves ``config.provider`` to its class (imported here, not at config-load
    time), runs the provider's :meth:`DocConverter.parse_config` to validate and
    normalize its fields, then constructs it. Construction is where the
    converter's lazy SDK/binary import happens, so a missing tool surfaces here
    as :class:`dgml.errors.ConversionFailed`; a malformed ``conversion.<family>``
    section surfaces as :class:`ConversionConfigInvalid`.
    """
    cls = _resolve_converter_class(config.provider)
    validated = cls.parse_config({"provider": config.provider, **config.options})
    return cls(validated)


def _strip_converter_suffix(name: str) -> str:
    """Return ``name`` with a trailing ``"converter"`` token removed.

    e.g. ``"LibreOfficeConverter" -> "LibreOffice"``. A name that is *only*
    ``"converter"`` (any case) is returned unchanged, as is a name that does
    not end in ``"converter"``.
    """
    if name.lower() == "converter":
        return name
    if name.lower().endswith("converter"):
        return name[: -len("converter")].rstrip(" -_") or name
    return name


def converter_name_for_path(
    path: Path, converters: Mapping[FormatFamily, ConverterConfig]
) -> str | None:
    """Return the display name of the converter that handles ``path``.

    The name is the converter class's ``name`` ClassVar with any trailing
    ``"converter"`` suffix stripped (see :func:`_strip_converter_suffix`).
    Returns ``None`` for a ``.pdf`` source or a family with no configured
    converter — i.e. exactly the cases where :func:`convert_to_pdf_bytes` is
    not invoked.
    """
    family = family_for_suffix(path.suffix.lower())
    if family is None or family not in converters:
        return None
    cls = _resolve_converter_class(converters[family].provider)
    return _strip_converter_suffix(cls.name)


def _normalize_suffix(suffix: str) -> str:
    """``suffix`` as :data:`FAMILY_BY_SUFFIX` spells it: lowercase, with the dot —
    so a converter may declare ``input_formats`` as ``"DOCX"`` or ``"docx"``."""
    suffix = suffix.lower()
    return suffix if suffix.startswith(".") else f".{suffix}"


def _check_input_format(cls: type[DocConverter], suffix: str, family: FormatFamily) -> None:
    """Refuse a converter that declares ``input_formats`` without ``suffix``, before
    it is built: a family pointed at the wrong converter (``conversion.pptx`` at the
    xlsx renderer, say) would otherwise fail inside that converter's SDK, or hand
    back a PDF of something else."""
    handled = sorted({_normalize_suffix(s) for s in cls.input_formats})
    if handled and suffix not in handled:
        raise ConversionConfigInvalid(
            f"converter {cls.name!r} (conversion.{family}.provider) does not handle "
            f"'{suffix}' — it handles {', '.join(handled)}. Set conversion.{family}.provider "
            f"to a converter that does."
        )


def require_converter(
    suffix: str, converters: Mapping[FormatFamily, ConverterConfig]
) -> tuple[FormatFamily, ConverterConfig]:
    """The format family of a convertible ``suffix`` and the converter configured
    for it, checked to handle that suffix.

    Raises :class:`dgml.errors.UnsupportedFileType` naming the config key to set
    when no converter is configured, and :class:`ConversionConfigInvalid` when the
    configured one cannot be resolved or its ``input_formats`` excludes ``suffix``.
    Resolving the class is an import of the converter's module, not a construction
    (no binary or SDK is touched). The add-time gate (``FileStore._validate_source``),
    the CLI's bulk-add gathering and :func:`convert_to_pdf_bytes` share this, so a
    misconfigured family is refused before anything is written, with one message.
    """
    family = family_for_suffix(suffix)
    if family is None or family not in converters:
        family_hint = family or "<family>"
        raise UnsupportedFileType(
            f"no converter configured for '{suffix or '<no extension>'}'; set "
            f"conversion.{family_hint}.provider in config.toml (see the translators-pdf "
            f"package for ready-made converters)"
        )
    config = converters[family]
    _check_input_format(_resolve_converter_class(config.provider), suffix, family)
    return family, config


def convert_to_pdf_bytes(path: Path, converters: Mapping[FormatFamily, ConverterConfig]) -> bytes:
    """Convert a non-PDF source at ``path`` to PDF bytes via the configured converter.

    Looks up the converter for the file's format family in ``converters``,
    instantiates it, and runs it. Raises :class:`dgml.errors.UnsupportedFileType`
    if the suffix has no configured converter (or is not a convertible source),
    :class:`ConversionConfigInvalid` if the configured converter's
    ``input_formats`` excludes the suffix, and :class:`dgml.errors.ConversionFailed`
    on converter errors. Callers that need to handle ``.pdf`` inputs or reuse a
    previously-produced PDF do so before calling this.
    """
    _, config = require_converter(path.suffix.lower(), converters)
    return make_converter(config).to_pdf(path)


def load_conversion_config(workspace: Workspace) -> dict[FormatFamily, ConverterConfig]:
    """Read and validate the ``conversion`` section of ``<workspace>/config.toml``.

    Returns a mapping of :class:`FormatFamily` to its :class:`ConverterConfig`.
    Only families explicitly configured are present — there are no defaults, so
    an absent family means that format is unsupported. A key that is not a
    family (``conversion.doc``, ``conversion.word``) is rejected: nothing would
    ever look it up, so the converter it names would silently never run. When
    the key is a convertible suffix (``doc``, ``xls``), the error names the
    family it belongs to. A missing config file or missing ``conversion``
    section yields an empty dict.

    Validates only the *generic shape* — each family is an object with a
    non-empty string ``provider``. It deliberately does **not** import the
    provider class or run its ``parse_config``: many entry points load the
    config without ever converting (e.g. ``docset generate`` over files that
    already have their PDFs, the add-time suffix gate), and resolving the class
    here would force an otherwise-unnecessary import of the converter package.
    Provider resolution and field validation happen lazily in
    :func:`make_converter`, when a document is actually converted.

    Raises :class:`ConversionConfigInvalid` only for a malformed *shape* (the
    section or a family isn't an object, a key isn't a format family, or
    ``provider`` is missing/blank).
    """
    section = load_merged_config(workspace).get(ConfigSection.CONVERSION)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ConversionConfigInvalid("'conversion' must be a table")

    configs: dict[FormatFamily, ConverterConfig] = {}
    for key, family_section in section.items():
        family = _require_family(key)
        if not isinstance(family_section, dict):
            raise ConversionConfigInvalid(f"'conversion.{family}' must be a table")
        provider = family_section.get("provider")
        if not isinstance(provider, str) or not provider.strip():
            raise ConversionConfigInvalid(
                f"'conversion.{family}.provider' must be a non-empty string"
            )
        # Keep the section verbatim (minus provider). The class is resolved and
        # these fields validated lazily, in make_converter — see the docstring.
        options = {k: v for k, v in family_section.items() if k != "provider"}
        configs[family] = ConverterConfig(provider=provider, options=options)
    return configs


def _require_family(key: str) -> FormatFamily:
    """The :class:`FormatFamily` a ``conversion.<key>`` table configures.

    Raises :class:`ConversionConfigInvalid` for a key that is not a family.
    Converters are looked up by family only, so a converter under any other key
    (``conversion.doc``, ``conversion.word``) would load and never run — the
    file types it was meant for would be refused as having no converter. The
    message lists the families and, for a key that is a convertible suffix
    (``doc``, ``xls``), the family that suffix belongs to. A family spelled in
    the wrong case (``DOCX``) is told so, not that it is an extension.
    """
    try:
        return FormatFamily(key)
    except ValueError:
        pass
    families = ", ".join(FormatFamily)
    message = f"unknown conversion family '{key}'; expected one of {families}"
    if key.lower() in FormatFamily.__members__.values():
        message += f" (family keys are lowercase: conversion.{key.lower()})"
    else:
        suffix = _normalize_suffix(key)
        family = family_for_suffix(suffix)
        if family is not None:
            message += f" ('{suffix}' files are configured under conversion.{family})"
    raise ConversionConfigInvalid(message)
