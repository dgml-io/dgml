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

"""Generic command-line document-to-PDF converter.

For users who already have a CLI converter and want zero Python. The command is
an **argv list** (not a shell string — no quoting/injection surface) with
substitution tokens:

- ``{input}``      — the source file path (required).
- ``{output}``     — the exact output PDF path the command must write.
- ``{output_dir}`` — a directory the command writes into; the produced PDF is
  discovered afterward (for tools like LibreOffice that name the file
  themselves).

Exactly one of ``{output}`` / ``{output_dir}`` must appear. Example config::

    {"provider": "translators_pdf.command:CommandConverter",
     "command": ["ssconvert", "{input}", "{output}"]}
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from dgml_core.conversion import ConverterConfig, DocConverter
from dgml_core.errors import ConversionConfigInvalid, ConversionFailed

_DEFAULT_TIMEOUT_S = 180


class CommandConverter(DocConverter):
    name: ClassVar[str] = "command"
    # Format-agnostic: it handles whatever family it is registered under.
    input_formats: ClassVar[frozenset[str]] = frozenset()
    config_fields: ClassVar[frozenset[str]] = frozenset({"command", "timeout"})

    @classmethod
    def parse_config(cls, section: Mapping[str, Any]) -> ConverterConfig:
        cls._check_no_extra_fields(section)
        command = section.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise ConversionConfigInvalid(
                "'conversion.<family>.command' must be a non-empty list of non-empty strings"
            )
        joined = " ".join(command)
        if "{input}" not in joined:
            raise ConversionConfigInvalid(
                "'conversion.<family>.command' must reference the '{input}' token"
            )
        has_output = "{output}" in joined
        has_output_dir = "{output_dir}" in joined
        if has_output == has_output_dir:
            raise ConversionConfigInvalid(
                "'conversion.<family>.command' must reference exactly one of "
                "'{output}' or '{output_dir}'"
            )
        timeout = section.get("timeout", _DEFAULT_TIMEOUT_S)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            raise ConversionConfigInvalid(
                "'conversion.<family>.timeout' must be a positive integer (seconds)"
            )
        return ConverterConfig(
            provider=str(section["provider"]),
            options={"command": list(command), "timeout": timeout},
        )

    def __init__(self, config: ConverterConfig) -> None:
        self._command: list[str] = list(config.options["command"])
        self._timeout: int = int(config.options["timeout"])
        self._uses_output = any("{output}" in part for part in self._command)

    def to_pdf(self, path: Path) -> bytes:
        path = Path(path)
        with tempfile.TemporaryDirectory(prefix="dgml-cmd-") as tmp:
            outdir = Path(tmp)
            output = outdir / f"{path.stem}.pdf"
            argv = [
                part.replace("{input}", str(path))
                .replace("{output_dir}", str(outdir))
                .replace("{output}", str(output))
                for part in self._command
            ]
            try:
                proc = subprocess.run(argv, capture_output=True, timeout=self._timeout)
            except FileNotFoundError as exc:
                raise ConversionFailed(f"converter command not found: {argv[0]!r}") from exc
            except subprocess.TimeoutExpired as exc:
                raise ConversionFailed(
                    f"converter command timed out after {self._timeout}s on {path.name}"
                ) from exc

            produced = output if self._uses_output else _single_pdf(outdir)
            if produced is None or not produced.exists() or produced.stat().st_size == 0:
                stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
                detail = f": {stderr}" if stderr else ""
                raise ConversionFailed(
                    f"converter command produced no PDF for {path.name} "
                    f"(exit {proc.returncode}){detail}"
                )
            return produced.read_bytes()


def _single_pdf(outdir: Path) -> Path | None:
    """The lone ``*.pdf`` written into ``outdir``, or ``None`` if not exactly one."""
    pdfs = sorted(outdir.glob("*.pdf"))
    return pdfs[0] if len(pdfs) == 1 else None
