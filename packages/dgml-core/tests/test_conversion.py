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

"""Tests for the core conversion abstraction: resolver + config loader + dispatch."""

from __future__ import annotations

import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import pytest
from dgml_core.conversion import (
    ConverterConfig,
    DocConverter,
    FormatFamily,
    _strip_converter_suffix,
    convert_to_pdf_bytes,
    converter_name_for_path,
    family_for_suffix,
    load_conversion_config,
    make_converter,
)
from dgml_core.errors import ConversionConfigInvalid, UnsupportedFileType
from dgml_core.generation import document
from dgml_core.storage import Workspace


class StubConverter(DocConverter):
    name: ClassVar[str] = "stub"
    input_formats: ClassVar[frozenset[str]] = frozenset({".xlsx", ".docx"})
    config_fields: ClassVar[frozenset[str]] = frozenset({"foo"})

    @classmethod
    def parse_config(cls, section: Mapping[str, Any]) -> ConverterConfig:
        cls._check_no_extra_fields(section)
        return ConverterConfig(
            provider=str(section["provider"]), options={"foo": section.get("foo")}
        )

    def __init__(self, config: ConverterConfig) -> None:
        self.config = config

    def to_pdf(self, path: Path) -> bytes:
        return b"%PDF-stub:" + Path(path).name.encode()


class AnyFormatStub(StubConverter):
    """Format-agnostic, like the command runner: an empty ``input_formats``."""

    name: ClassVar[str] = "any-format-stub"
    input_formats: ClassVar[frozenset[str]] = frozenset()


class LooseFormatStub(StubConverter):
    """Declares its formats the way a third-party converter might: no dot, mixed case."""

    name: ClassVar[str] = "loose-format-stub"
    input_formats: ClassVar[frozenset[str]] = frozenset({"DOCX", "Doc"})


class NotAConverter:
    pass


# Register a fake importable module so dotted-path resolution can find the stub.
_STUB_MODULE = types.ModuleType("conv_stub")
_STUB_MODULE.StubConverter = StubConverter  # type: ignore[attr-defined]
_STUB_MODULE.AnyFormatStub = AnyFormatStub  # type: ignore[attr-defined]
_STUB_MODULE.LooseFormatStub = LooseFormatStub  # type: ignore[attr-defined]
_STUB_MODULE.NotAConverter = NotAConverter  # type: ignore[attr-defined]
sys.modules["conv_stub"] = _STUB_MODULE

_STUB = "conv_stub:StubConverter"
_ANY_STUB = "conv_stub:AnyFormatStub"
_LOOSE_STUB = "conv_stub:LooseFormatStub"


def test_resolve_and_instantiate_dotted_path() -> None:
    conv = make_converter(ConverterConfig(provider=_STUB))
    assert isinstance(conv, StubConverter)
    assert conv.to_pdf(Path("a.xlsx")) == b"%PDF-stub:a.xlsx"


def test_provider_without_colon_raises() -> None:
    with pytest.raises(ConversionConfigInvalid):
        make_converter(ConverterConfig(provider="libreoffice"))


def test_unimportable_module_raises() -> None:
    with pytest.raises(ConversionConfigInvalid):
        make_converter(ConverterConfig(provider="nonexistent.mod:Thing"))


def test_missing_attribute_raises() -> None:
    with pytest.raises(ConversionConfigInvalid):
        make_converter(ConverterConfig(provider="conv_stub:Missing"))


def test_non_docconverter_raises() -> None:
    with pytest.raises(ConversionConfigInvalid):
        make_converter(ConverterConfig(provider="conv_stub:NotAConverter"))


def test_family_for_suffix() -> None:
    assert family_for_suffix(".DOCX") is FormatFamily.DOCX
    assert family_for_suffix(".xls") is FormatFamily.XLSX
    assert family_for_suffix(".PPTX") is FormatFamily.PPTX
    assert family_for_suffix(".ppt") is FormatFamily.PPTX
    assert family_for_suffix(".pdf") is None
    assert family_for_suffix(".txt") is None
    # A member is its TOML key, so a loaded config's string keys match it.
    assert FormatFamily.DOCX.value == "docx"


def test_converter_must_handle_the_suffix(tmp_path: Path) -> None:
    """A family pointed at a converter whose ``input_formats`` excludes that
    family's suffix is refused before the converter is built."""
    deck = tmp_path / "deck.pptx"
    deck.write_bytes(b"junk")
    converters = {FormatFamily.PPTX: ConverterConfig(provider=_STUB)}  # handles docx/xlsx
    with pytest.raises(ConversionConfigInvalid, match=r"does not handle '\.pptx'"):
        convert_to_pdf_bytes(deck, converters)
    # An empty input_formats means any suffix.
    converters = {FormatFamily.PPTX: ConverterConfig(provider=_ANY_STUB)}
    assert convert_to_pdf_bytes(deck, converters) == b"%PDF-stub:deck.pptx"


def test_input_formats_match_without_the_dot_and_case(tmp_path: Path) -> None:
    """``input_formats`` was informational before it was enforced, so a converter
    that spells its suffixes loosely (``"DOCX"``, ``"Doc"``) must keep working."""
    converters = {FormatFamily.DOCX: ConverterConfig(provider=_LOOSE_STUB)}
    for name in ("memo.docx", "MEMO.DOC"):
        doc = tmp_path / name
        doc.write_bytes(b"junk")
        assert convert_to_pdf_bytes(doc, converters) == b"%PDF-stub:" + name.encode()


def _write_config(ws: Workspace, conversion: dict[str, Any]) -> None:
    from .conftest import write_config

    write_config(ws, {"conversion": conversion})


def test_load_conversion_config_absent_returns_empty(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    assert load_conversion_config(ws) == {}


def test_load_conversion_config_parses_families(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    _write_config(
        ws,
        {
            "docx": {"provider": _STUB},
            "xlsx": {"provider": _STUB, "foo": "bar"},
        },
    )
    configs = load_conversion_config(ws)
    assert set(configs) == {FormatFamily.DOCX, FormatFamily.XLSX}
    assert all(isinstance(key, FormatFamily) for key in configs)
    assert configs[FormatFamily.XLSX].options["foo"] == "bar"


def test_unknown_field_raises_at_conversion_time(tmp_path: Path) -> None:
    """An unknown provider field is not rejected at load (that check needs
    the class); it surfaces lazily when the converter is built."""
    ws = Workspace(root=tmp_path)
    _write_config(ws, {"xlsx": {"provider": _STUB, "bogus": 1}})
    configs = load_conversion_config(ws)  # shape-only validation: no raise
    assert configs[FormatFamily.XLSX].options["bogus"] == 1
    with pytest.raises(ConversionConfigInvalid):
        make_converter(configs[FormatFamily.XLSX])


def test_load_conversion_config_does_not_import_provider(tmp_path: Path) -> None:
    """Loading the config must not import the converter module — only the actual
    conversion path does — so a provider whose module is absent still loads."""
    ws = Workspace(root=tmp_path)
    _write_config(ws, {"docx": {"provider": "totally_absent_module:Thing"}})
    configs = load_conversion_config(ws)  # no ImportError, no raise
    assert configs[FormatFamily.DOCX].provider == "totally_absent_module:Thing"
    # Resolution is deferred — it fails only when a converter is actually built.
    with pytest.raises(ConversionConfigInvalid):
        make_converter(configs[FormatFamily.DOCX])


def test_load_conversion_config_missing_provider_raises(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    _write_config(ws, {"xlsx": {"foo": "bar"}})
    with pytest.raises(ConversionConfigInvalid):
        load_conversion_config(ws)


@pytest.mark.parametrize(
    ("key", "hint"),
    [
        ("doc", "('.doc' files are configured under conversion.docx)"),
        ("xls", "('.xls' files are configured under conversion.xlsx)"),
        ("ppt", "('.ppt' files are configured under conversion.pptx)"),
    ],
)
def test_load_conversion_config_suffix_as_family_raises_with_hint(
    tmp_path: Path, key: str, hint: str
) -> None:
    """A key that is a file extension rather than a family (dgml-io/dgml#274)
    is rejected up front — nothing would ever look it up — and the message
    names the family that extension belongs to."""
    ws = Workspace(root=tmp_path)
    _write_config(ws, {key: {"provider": _STUB}})
    with pytest.raises(ConversionConfigInvalid) as excinfo:
        load_conversion_config(ws)
    message = str(excinfo.value)
    assert f"unknown conversion family '{key}'" in message
    assert "expected one of docx, xlsx, pptx" in message
    assert hint in message


def test_load_conversion_config_unknown_family_raises(tmp_path: Path) -> None:
    """A key that is neither a family nor a known extension lists the families
    and gives no extension hint."""
    ws = Workspace(root=tmp_path)
    _write_config(ws, {"docx": {"provider": _STUB}, "word": {"provider": _STUB}})
    with pytest.raises(ConversionConfigInvalid) as excinfo:
        load_conversion_config(ws)
    message = str(excinfo.value)
    assert message == "unknown conversion family 'word'; expected one of docx, xlsx, pptx"


def test_load_conversion_config_wrong_case_family_raises_with_case_hint(tmp_path: Path) -> None:
    """A family in the wrong case is told so, rather than being read as a file
    extension (``DOCX`` normalises to ``.docx``, which maps to the same family)."""
    ws = Workspace(root=tmp_path)
    _write_config(ws, {"DOCX": {"provider": _STUB}})
    with pytest.raises(ConversionConfigInvalid) as excinfo:
        load_conversion_config(ws)
    message = str(excinfo.value)
    assert "unknown conversion family 'DOCX'" in message
    assert "(family keys are lowercase: conversion.docx)" in message
    assert "files are configured under" not in message


def test_load_conversion_config_unknown_family_checked_before_shape(tmp_path: Path) -> None:
    """The family check comes first: a bad key is reported as such even when
    its table is also malformed."""
    ws = Workspace(root=tmp_path)
    _write_config(ws, {"doc": {}})
    with pytest.raises(ConversionConfigInvalid, match="unknown conversion family 'doc'"):
        load_conversion_config(ws)


def test_dispatch_pdf_passthrough(tmp_path: Path) -> None:
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 hi")
    assert document.load_document_as_pdf(pdf, converters={}) == b"%PDF-1.4 hi"


def test_dispatch_unconfigured_family_raises(tmp_path: Path) -> None:
    src = tmp_path / "x.xlsx"
    src.write_bytes(b"junk")
    with pytest.raises(UnsupportedFileType):
        document.load_document_as_pdf(src, converters={})


def test_dispatch_uses_configured_converter(tmp_path: Path) -> None:
    src = tmp_path / "x.docx"
    src.write_bytes(b"junk")
    converters = {FormatFamily.DOCX: ConverterConfig(provider=_STUB)}
    assert document.load_document_as_pdf(src, converters=converters) == b"%PDF-stub:x.docx"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("libreoffice", "libreoffice"),  # no suffix → unchanged
        ("xlsx-islands", "xlsx-islands"),
        ("LibreOfficeConverter", "LibreOffice"),  # PascalCase suffix stripped
        ("aspose-converter", "aspose"),  # separator before suffix trimmed
        ("foo_converter", "foo"),
        ("converter", "converter"),  # whole name is the suffix → kept
        ("Converter", "Converter"),
    ],
)
def test_strip_converter_suffix(raw: str, expected: str) -> None:
    assert _strip_converter_suffix(raw) == expected


def test_converter_name_for_path_pdf_is_none(tmp_path: Path) -> None:
    assert converter_name_for_path(tmp_path / "x.pdf", {}) is None


def test_converter_name_for_path_unconfigured_family_is_none(tmp_path: Path) -> None:
    assert converter_name_for_path(tmp_path / "x.docx", {}) is None


def test_converter_name_for_path_uses_converter_name() -> None:
    converters = {FormatFamily.DOCX: ConverterConfig(provider=_STUB)}
    # StubConverter.name == "stub" (no suffix to strip).
    assert converter_name_for_path(Path("x.docx"), converters) == "stub"


def test_dispatch_reuses_persisted_sibling_pdf(tmp_path: Path) -> None:
    """A converted PDF persisted next to the source is reused verbatim — no
    converter is invoked (empty converters would otherwise raise)."""
    src = tmp_path / "x.docx"
    src.write_bytes(b"original docx bytes")
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4 persisted")
    assert document.load_document_as_pdf(src, converters={}) == b"%PDF-1.4 persisted"
