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

"""The CLI writes UTF-8 to a redirected stdout or stderr whatever the locale
encoding, and never crashes on a character a stream cannot encode.

``dgml --help`` crashed with ``UnicodeEncodeError`` on a stock Windows machine
whenever its output was piped or captured: a redirected stream gets the locale
encoding (cp1252) with the ``strict`` handler, and the help text carries
characters outside it. ``PYTHONIOENCODING`` forces that starting encoding on
any platform, so the regression is reproducible everywhere.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest
from dgml.cli import _configure_stream_encodings, _emit

_CP1252_STRICT = {"PYTHONIOENCODING": "cp1252:strict", "PYTHONUTF8": "0"}
_UTF8_STRICT = {"PYTHONIOENCODING": "utf-8:strict"}


def _run(argv: list[str], env_overrides: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env.update(env_overrides)
    code = f"from dgml.cli import main; raise SystemExit(main({argv!r}))"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, timeout=120)


def test_help_is_utf8_on_a_cp1252_redirected_stdout() -> None:
    proc = _run(["--help"], _CP1252_STRICT)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    text = proc.stdout.decode("utf-8")  # strict: the bytes really are UTF-8
    assert "usage:" in text
    assert any(ord(ch) > 127 for ch in text)  # the help text that used to crash


def test_help_bytes_match_a_stdout_that_is_utf8_already() -> None:
    """A reconfigured pipe writes exactly what a UTF-8 pipe writes."""
    native = _run(["--help"], _UTF8_STRICT)
    assert native.returncode == 0, native.stderr.decode("utf-8", "replace")
    assert native.stdout == _run(["--help"], _CP1252_STRICT).stdout


def test_an_argparse_error_is_utf8_on_a_cp1252_redirected_stderr() -> None:
    """The error path writes to stderr; an argument the parser rejects is
    echoed there, and one outside cp1252 used to crash the report itself."""
    proc = _run(["\u2192"], _CP1252_STRICT)
    assert proc.returncode == 2, proc.stderr.decode("utf-8", "replace")
    assert "\u2192" in proc.stderr.decode("utf-8")


class _Tty(io.TextIOWrapper):
    def isatty(self) -> bool:
        return True


def test_a_terminal_keeps_its_code_page_and_escapes_what_it_lacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tty is not switched to UTF-8 (its reader is a person on a code page);
    it gets ``backslashreplace`` so nothing crashes."""
    raw = io.BytesIO()
    stream = _Tty(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)
    _configure_stream_encodings()
    assert stream.encoding == "cp1252"
    assert stream.errors == "backslashreplace"
    stream.write("a \u2192 b")  # an arrow cp1252 lacks: no crash, an escape
    stream.flush()
    assert raw.getvalue() == b"a \\u2192 b"


def test_a_stream_whose_isatty_fails_is_treated_as_redirected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Broken(io.TextIOWrapper):
        def isatty(self) -> bool:
            raise OSError("no such device")

    raw = io.BytesIO()
    stream = _Broken(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)
    _configure_stream_encodings()  # must not raise
    assert stream.encoding == "utf-8"
    assert stream.errors == "backslashreplace"


def test_emit_keeps_json_valid_on_a_stream_that_is_not_utf8() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="backslashreplace")
    _emit({"name": "a \u2192 b \U0001f600"}, "json", stream)
    stream.flush()
    assert json.loads(raw.getvalue().decode("ascii")) == {"name": "a \u2192 b \U0001f600"}

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8")
    _emit({"name": "a \u2192 b"}, "json", stream)
    stream.flush()
    assert "\u2192".encode() in raw.getvalue()  # a UTF-8 stream gets the character itself
