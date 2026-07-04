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
    _strip_converter_suffix,
    converter_name_for_path,
    family_for_suffix,
    load_conversion_config,
    make_converter,
)
from dgml_core.errors import ConversionConfigInvalid, UnsupportedFileType
from dgml_core.generation import document
from dgml_core.storage import Workspace, write_json_atomic


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


class NotAConverter:
    pass


# Register a fake importable module so dotted-path resolution can find the stub.
_STUB_MODULE = types.ModuleType("conv_stub")
_STUB_MODULE.StubConverter = StubConverter  # type: ignore[attr-defined]
_STUB_MODULE.NotAConverter = NotAConverter  # type: ignore[attr-defined]
sys.modules["conv_stub"] = _STUB_MODULE

_STUB = "conv_stub:StubConverter"


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
    assert family_for_suffix(".DOCX") == "docx"
    assert family_for_suffix(".xls") == "xlsx"
    assert family_for_suffix(".pdf") is None
    assert family_for_suffix(".txt") is None


def _write_config(ws: Workspace, conversion: dict[str, Any]) -> None:
    write_json_atomic(ws.config_path, {"conversion": conversion})


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
    assert set(configs) == {"docx", "xlsx"}
    assert configs["xlsx"].options["foo"] == "bar"


def test_unknown_field_raises_at_conversion_time(tmp_path: Path) -> None:
    """An unknown provider field is not rejected at load (that check needs
    the class); it surfaces lazily when the converter is built."""
    ws = Workspace(root=tmp_path)
    _write_config(ws, {"xlsx": {"provider": _STUB, "bogus": 1}})
    configs = load_conversion_config(ws)  # shape-only validation: no raise
    assert configs["xlsx"].options["bogus"] == 1
    with pytest.raises(ConversionConfigInvalid):
        make_converter(configs["xlsx"])


def test_load_conversion_config_does_not_import_provider(tmp_path: Path) -> None:
    """Loading the config must not import the converter module — only the actual
    conversion path does — so a provider whose module is absent still loads."""
    ws = Workspace(root=tmp_path)
    _write_config(ws, {"docx": {"provider": "totally_absent_module:Thing"}})
    configs = load_conversion_config(ws)  # no ImportError, no raise
    assert configs["docx"].provider == "totally_absent_module:Thing"
    # Resolution is deferred — it fails only when a converter is actually built.
    with pytest.raises(ConversionConfigInvalid):
        make_converter(configs["docx"])


def test_load_conversion_config_missing_provider_raises(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    _write_config(ws, {"xlsx": {"foo": "bar"}})
    with pytest.raises(ConversionConfigInvalid):
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
    converters = {"docx": ConverterConfig(provider=_STUB)}
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
    converters = {"docx": ConverterConfig(provider=_STUB)}
    # StubConverter.name == "stub" (no suffix to strip).
    assert converter_name_for_path(Path("x.docx"), converters) == "stub"


def test_dispatch_reuses_persisted_sibling_pdf(tmp_path: Path) -> None:
    """A converted PDF persisted next to the source is reused verbatim — no
    converter is invoked (empty converters would otherwise raise)."""
    src = tmp_path / "x.docx"
    src.write_bytes(b"original docx bytes")
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.4 persisted")
    assert document.load_document_as_pdf(src, converters={}) == b"%PDF-1.4 persisted"
