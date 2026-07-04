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

Source documents (docx, xlsx, …) are converted to PDF before the rest of the
pipeline, which is PDF-only. Which converter handles each *format family*
(``docx``, ``xlsx``) is chosen per workspace in the ``conversion`` section of
``<workspace>/config.json``::

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
from pathlib import Path
from typing import Any, ClassVar

from .errors import ConversionConfigInvalid, UnsupportedFileType
from .storage import Workspace, read_config

# Format families and the source suffixes that belong to each. ``.pdf`` is
# absent on purpose: it needs no converter. Shared with the add-time ingestion
# gate (dgml.files) and the generation dispatch (dgml.generation.document) so
# the supported-input surface lives in one place.
FAMILY_BY_SUFFIX: dict[str, str] = {
    ".docx": "docx",
    ".doc": "docx",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
}


def family_for_suffix(suffix: str) -> str | None:
    """Return the conversion family for a file suffix (e.g. ``.docx`` → ``"docx"``),
    or ``None`` if the suffix is not a known convertible source format."""
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
    a converter handles (informational; an empty set means "any", used by
    format-agnostic converters like the command runner).
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


def converter_name_for_path(path: Path, converters: Mapping[str, ConverterConfig]) -> str | None:
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


def convert_to_pdf_bytes(path: Path, converters: Mapping[str, ConverterConfig]) -> bytes:
    """Convert a non-PDF source at ``path`` to PDF bytes via the configured converter.

    Looks up the converter for the file's format family in ``converters``,
    instantiates it, and runs it. Raises :class:`dgml.errors.UnsupportedFileType`
    if the suffix has no configured converter (or is not a convertible source),
    and :class:`dgml.errors.ConversionFailed` on converter errors. Callers that
    need to handle ``.pdf`` inputs or reuse a previously-produced PDF do so
    before calling this.
    """
    suffix = path.suffix.lower()
    family = family_for_suffix(suffix)
    if family is None or family not in converters:
        family_hint = family or "<family>"
        raise UnsupportedFileType(
            f"no converter configured for '{suffix or '<no extension>'}'; set "
            f"conversion.{family_hint}.provider in config.json (see the translators-pdf "
            f"package for ready-made converters)"
        )
    return make_converter(converters[family]).to_pdf(path)


def load_conversion_config(workspace: Workspace) -> dict[str, ConverterConfig]:
    """Read and validate the ``conversion`` section of ``<workspace>/config.json``.

    Returns a mapping of format family (``"docx"``, ``"xlsx"``) to its
    :class:`ConverterConfig`. Only families explicitly configured are present —
    there are no defaults, so an absent family means that format is unsupported.
    A missing config file or missing ``conversion`` section yields an empty dict.

    Validates only the *generic shape* — each family is an object with a
    non-empty string ``provider``. It deliberately does **not** import the
    provider class or run its ``parse_config``: many entry points load the
    config without ever converting (e.g. ``docset generate`` over files that
    already have their PDFs, the add-time suffix gate), and resolving the class
    here would force an otherwise-unnecessary import of the converter package.
    Provider resolution and field validation happen lazily in
    :func:`make_converter`, when a document is actually converted.

    Raises :class:`ConversionConfigInvalid` only for a malformed *shape* (the
    section or a family isn't an object, or ``provider`` is missing/blank).
    """
    if not workspace.config_path.exists():
        return {}

    try:
        data = read_config(workspace.config_path)
    except Exception as exc:  # CorruptMetadata and friends
        raise ConversionConfigInvalid(f"{workspace.config_path} is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ConversionConfigInvalid(f"{workspace.config_path} must contain a JSON object")

    section = data.get("conversion")
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ConversionConfigInvalid("'conversion' must be a JSON object")

    configs: dict[str, ConverterConfig] = {}
    for family, family_section in section.items():
        if not isinstance(family_section, dict):
            raise ConversionConfigInvalid(f"'conversion.{family}' must be a JSON object")
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
