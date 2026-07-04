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

"""Excel-to-PDF converter rendering each data "island" as its own page.

Depends on ``openpyxl`` (MIT) + ``reportlab`` (BSD), gated behind the
``translators-pdf[xlsx]`` extra and lazy-imported so importing this module never
requires them. Config fields: ``row_gap``, ``col_gap``, ``orientation``.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from dgml_core.conversion import ConverterConfig, DocConverter
from dgml_core.errors import ConversionConfigInvalid, ConversionFailed

_ORIENTATIONS = frozenset({"landscape", "portrait"})


class XlsxIslandsConverter(DocConverter):
    name: ClassVar[str] = "xlsx-islands"
    input_formats: ClassVar[frozenset[str]] = frozenset({".xlsx", ".xls"})
    config_fields: ClassVar[frozenset[str]] = frozenset({"row_gap", "col_gap", "orientation"})

    @classmethod
    def parse_config(cls, section: Mapping[str, Any]) -> ConverterConfig:
        cls._check_no_extra_fields(section)
        row_gap = cls._gap(section, "row_gap")
        col_gap = cls._gap(section, "col_gap")
        orientation = section.get("orientation", "landscape")
        if orientation not in _ORIENTATIONS:
            raise ConversionConfigInvalid(
                f"'conversion.<family>.orientation' must be one of {sorted(_ORIENTATIONS)} "
                f"(got {orientation!r})"
            )
        return ConverterConfig(
            provider=str(section["provider"]),
            options={"row_gap": row_gap, "col_gap": col_gap, "orientation": orientation},
        )

    @staticmethod
    def _gap(section: Mapping[str, Any], key: str) -> int:
        value = section.get(key, 2)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ConversionConfigInvalid(
                f"'conversion.<family>.{key}' must be a non-negative integer"
            )
        return value

    def __init__(self, config: ConverterConfig) -> None:
        try:
            import openpyxl  # noqa: F401
            import reportlab  # noqa: F401
        except ImportError as exc:
            raise ConversionFailed(
                "the xlsx island renderer requires reportlab + openpyxl. "
                "Install with `pip install translators-pdf[xlsx]`."
            ) from exc
        self._row_gap = int(config.options["row_gap"])
        self._col_gap = int(config.options["col_gap"])
        self._orientation = str(config.options["orientation"])

    def to_pdf(self, path: Path) -> bytes:
        import openpyxl

        from . import _xlsx_detector, _xlsx_renderer

        path = Path(path)
        try:
            wb = openpyxl.load_workbook(str(path), data_only=True)
        except Exception as exc:
            raise ConversionFailed(f"could not open workbook {path.name}: {exc}") from exc

        sheet_islands: dict[str, list[tuple[int, int, int, int]]] = {}
        for name in wb.sheetnames:
            islands = _xlsx_detector.find_islands(
                wb[name], row_gap=self._row_gap, col_gap=self._col_gap
            )
            if islands:
                sheet_islands[name] = islands

        with tempfile.TemporaryDirectory(prefix="dgml-xlsx-") as tmp:
            out = Path(tmp) / f"{path.stem}.pdf"
            try:
                _xlsx_renderer.render_pdf(
                    wb_path=str(path),
                    sheet_islands=sheet_islands,
                    pdf_path=str(out),
                    orientation=self._orientation,
                )
            except Exception as exc:
                raise ConversionFailed(f"could not render {path.name} to PDF: {exc}") from exc
            if not out.exists() or out.stat().st_size == 0:
                raise ConversionFailed(f"xlsx renderer produced no PDF for {path.name}")
            return out.read_bytes()
