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

"""LibreOffice document-to-PDF converter.

Converts docx/doc/xlsx/xls by driving a headless ``soffice`` subprocess. No
installable Python dependency — LibreOffice is a user-installed external tool
(like ghostscript), discovered on PATH or at the standard per-OS install path.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from dgml_core.conversion import ConverterConfig, DocConverter
from dgml_core.errors import ConversionFailed

# How long to wait for a single LibreOffice conversion before giving up. A cold
# soffice start plus a large document is slow; a wedged headless instance should
# not hang ingestion forever.
_CONVERT_TIMEOUT_S = 180


def _find_soffice() -> str | None:
    """Locate the LibreOffice executable across operating systems."""
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in (
        Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        Path("C:/Program Files/LibreOffice/program/soffice.exe"),
    ):
        if candidate.exists():
            return str(candidate)
    return None


class LibreOfficeConverter(DocConverter):
    name: ClassVar[str] = "libreoffice"
    input_formats: ClassVar[frozenset[str]] = frozenset({".docx", ".doc", ".xlsx", ".xls"})
    config_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def parse_config(cls, section: Mapping[str, Any]) -> ConverterConfig:
        cls._check_no_extra_fields(section)
        return ConverterConfig(provider=str(section["provider"]))

    def __init__(self, config: ConverterConfig) -> None:
        soffice = _find_soffice()
        if soffice is None:
            raise ConversionFailed(
                "LibreOffice ('soffice') was not found. Install it "
                "(e.g. `brew install --cask libreoffice`, `apt-get install libreoffice`) "
                "or pre-convert your file to PDF."
            )
        self._soffice = soffice

    def to_pdf(self, path: Path) -> bytes:
        path = Path(path)
        # A dedicated outdir for the result and a private UserInstallation
        # profile per call so concurrent conversions don't fight over the
        # single shared LibreOffice profile lock.
        with tempfile.TemporaryDirectory(prefix="dgml-lo-") as tmp:
            tmpdir = Path(tmp)
            outdir = tmpdir / "out"
            outdir.mkdir()
            profile = tmpdir / "profile"
            try:
                proc = subprocess.run(
                    [
                        self._soffice,
                        f"-env:UserInstallation=file://{profile}",
                        "--headless",
                        "--convert-to",
                        "pdf",
                        "--outdir",
                        str(outdir),
                        str(path),
                    ],
                    capture_output=True,
                    timeout=_CONVERT_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as exc:
                raise ConversionFailed(
                    f"LibreOffice timed out converting {path.name} after {_CONVERT_TIMEOUT_S}s"
                ) from exc

            # soffice often exits 0 even when it failed to produce output, so
            # the produced file — not the return code — is the source of truth.
            produced = sorted(outdir.glob("*.pdf"))
            if not produced:
                stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
                detail = f": {stderr}" if stderr else ""
                raise ConversionFailed(
                    f"LibreOffice did not produce a PDF for {path.name} "
                    f"(exit {proc.returncode}){detail}"
                )
            data = produced[0].read_bytes()
            if not data:
                raise ConversionFailed(f"LibreOffice produced an empty PDF for {path.name}")
            return data
