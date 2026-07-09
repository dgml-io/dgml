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

"""Aspose document-to-PDF converters (commercial, user-licensed).

The Aspose SDKs (``aspose-words`` for Word documents, ``aspose-cells-python``
for spreadsheets) are proprietary and intentionally **not** declared dependencies
of any package in this Apache-2.0 repo. They are lazy-imported here; constructing a
converter without its SDK installed raises an actionable :class:`ConversionFailed`,
the same treatment LibreOffice/ghostscript get as external tools.

Both converters share a private :class:`_AsposeConverter` base: identical config
parsing, license handling, and temp-dir save scaffolding live there, and each
concrete converter only declares its SDK module and how to open/save a document.

An optional ``license`` config field points at an Aspose license file; without it
Aspose runs in evaluation mode (watermark + page cap). The same license file
typically covers both products (Aspose.Total).
"""

from __future__ import annotations

import importlib
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from dgml_core.conversion import ConverterConfig, DocConverter
from dgml_core.errors import ConversionConfigInvalid, ConversionFailed


class _AsposeConverter(DocConverter):
    """Shared base for the Aspose.Words and Aspose.Cells converters.

    Subclasses set :attr:`name`, :attr:`input_formats`, :attr:`_sdk_module`, and
    :attr:`_install_pkg`, and implement :meth:`_open` / :meth:`_save` for their
    SDK. Everything else — config parsing, lazy SDK import, license application,
    and the temp-dir save-and-read scaffolding — is shared.
    """

    config_fields: ClassVar[frozenset[str]] = frozenset({"license"})

    # Dotted import path of the SDK (e.g. "aspose.words") and the pip package
    # that provides it (e.g. "aspose-words"), for the not-installed error.
    _sdk_module: ClassVar[str]
    _install_pkg: ClassVar[str]

    @classmethod
    def parse_config(cls, section: Mapping[str, Any]) -> ConverterConfig:
        cls._check_no_extra_fields(section)
        license_path = section.get("license")
        if license_path is not None and (
            not isinstance(license_path, str) or not license_path.strip()
        ):
            raise ConversionConfigInvalid(
                "'conversion.<family>.license' must be a non-empty string if set"
            )
        options: dict[str, Any] = {}
        if license_path is not None:
            options["license"] = license_path
        return ConverterConfig(provider=str(section["provider"]), options=options)

    def __init__(self, config: ConverterConfig) -> None:
        try:
            self._sdk = importlib.import_module(self._sdk_module)
        except ImportError as exc:
            raise ConversionFailed(
                f"{self._install_pkg} is required for the {self._sdk_module} converter. "
                f"Install it (`pip install {self._install_pkg}`) and, for unwatermarked "
                "output, provide a license."
            ) from exc
        license_path = config.options.get("license")
        if license_path:
            try:
                lic = self._sdk.License()
                lic.set_license(str(license_path))
            except Exception as exc:  # aspose raises its own exception types
                raise ConversionFailed(
                    f"could not apply Aspose license {license_path!r}: {exc}"
                ) from exc

    def to_pdf(self, path: Path) -> bytes:
        path = Path(path)
        try:
            doc = self._open(path)
        except Exception as exc:
            raise ConversionFailed(f"Aspose could not open {path.name}: {exc}") from exc

        with tempfile.TemporaryDirectory(prefix="dgml-aspose-") as tmp:
            out = Path(tmp) / f"{path.stem}.pdf"
            try:
                self._save(doc, out)
            except Exception as exc:
                raise ConversionFailed(f"Aspose could not convert {path.name}: {exc}") from exc
            if not out.exists() or out.stat().st_size == 0:
                raise ConversionFailed(f"Aspose produced no PDF for {path.name}")
            return out.read_bytes()

    def _open(self, path: Path) -> Any:
        """Open the source document with the SDK, returning its document object."""
        raise NotImplementedError

    def _save(self, doc: Any, out: Path) -> None:
        """Save the document object opened by :meth:`_open` to ``out`` as PDF."""
        raise NotImplementedError


class AsposeWordsConverter(_AsposeConverter):
    name: ClassVar[str] = "aspose-words"
    input_formats: ClassVar[frozenset[str]] = frozenset({".docx", ".doc"})
    _sdk_module: ClassVar[str] = "aspose.words"
    _install_pkg: ClassVar[str] = "aspose-words"

    def _open(self, path: Path) -> Any:
        return self._sdk.Document(str(path))

    def _save(self, doc: Any, out: Path) -> None:
        doc.save(str(out))


class AsposeCellsConverter(_AsposeConverter):
    name: ClassVar[str] = "aspose-cells"
    input_formats: ClassVar[frozenset[str]] = frozenset({".xlsx", ".xls"})
    _sdk_module: ClassVar[str] = "aspose.cells"
    _install_pkg: ClassVar[str] = "aspose-cells-python"

    def _open(self, path: Path) -> Any:
        return self._sdk.Workbook(str(path))

    def _save(self, doc: Any, out: Path) -> None:
        doc.save(str(out), self._sdk.SaveFormat.PDF)
