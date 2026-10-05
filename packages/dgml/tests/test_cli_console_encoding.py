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
characters outside it. The tests below build that stream in-process (a
``TextIOWrapper`` over bytes, ``isatty`` false, like a pipe) and run ``main``
on it, so they see the fix on every platform without depending on the
environment. One subprocess test pins the decision that an explicit
``PYTHONIOENCODING`` does not change the contract for a pipe.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest
from dgml.cli import _configure_stream_encodings, _emit


def _pipe(encoding: str) -> tuple[io.BytesIO, io.TextIOWrapper]:
    """A non-tty text stream over bytes, the shape of a redirected stdout."""
    raw = io.BytesIO()
    return raw, io.TextIOWrapper(raw, encoding=encoding, errors="strict")


def _run_main(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], encoding: str
) -> tuple[int, bytes, bytes]:
    """Run ``main(argv)`` with stdout and stderr on strict ``encoding`` pipes.

    Returns the exit code and the raw bytes each stream received."""
    from dgml.cli import main

    out_raw, out = _pipe(encoding)
    err_raw, err = _pipe(encoding)
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    with pytest.raises(SystemExit) as exc:
        main(argv)
    out.flush()
    err.flush()
    code = exc.value.code
    return (code if isinstance(code, int) else 1), out_raw.getvalue(), err_raw.getvalue()


def test_help_is_utf8_on_a_cp1252_redirected_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    code, out, err = _run_main(monkeypatch, ["--help"], "cp1252")
    assert code == 0, err.decode("utf-8", "replace")
    text = out.decode("utf-8")  # strict: the bytes really are UTF-8
    assert "usage:" in text
    assert any(ord(ch) > 127 for ch in text)  # the help text that used to crash


def test_help_bytes_match_a_stdout_that_is_utf8_already(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reconfigured pipe writes exactly what a UTF-8 pipe writes."""
    _, native, _ = _run_main(monkeypatch, ["--help"], "utf-8")
    _, switched, _ = _run_main(monkeypatch, ["--help"], "cp1252")
    assert native == switched


def test_an_argparse_error_is_utf8_on_a_cp1252_redirected_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error path writes to stderr; an argument the parser rejects is
    echoed there, and one outside cp1252 used to crash the report itself."""
    code, _, err = _run_main(monkeypatch, ["→"], "cp1252")
    assert code == 2, err.decode("utf-8", "replace")
    assert "→" in err.decode("utf-8")


def test_a_pipe_is_utf8_even_with_an_explicit_pythonioencoding() -> None:
    """The one subprocess test, and the decision it pins: a pipe or file gets
    UTF-8 even when ``PYTHONIOENCODING`` names another encoding, so a wrapper
    always decodes captured output as UTF-8 and never has to know the
    environment the command ran in."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env.update({"PYTHONIOENCODING": "cp1252:strict", "PYTHONUTF8": "0"})
    code = "from dgml.cli import main; raise SystemExit(main(['--help']))"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    text = proc.stdout.decode("utf-8")  # strict: really UTF-8 now
    assert "usage:" in text
    assert any(ord(ch) > 127 for ch in text)


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
    stream.write("a → b")  # an arrow cp1252 lacks: no crash, an escape
    stream.flush()
    assert raw.getvalue() == b"a \\u2192 b"


def test_a_utf8_stream_keeps_its_encoding_and_gets_backslashreplace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream that is UTF-8 already is not skipped: it keeps its encoding
    and gets ``backslashreplace``, so a lone surrogate (a non-UTF-8 file name
    is one source) is written as an escape instead of raising after part of
    the payload is already out. The escape is one JSON accepts."""
    raw, stream = _pipe("utf-8")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)
    _configure_stream_encodings()
    assert stream.encoding == "utf-8"
    assert stream.errors == "backslashreplace"
    _emit({"n": "bad\udcff"}, "json", stream)
    stream.flush()
    text = raw.getvalue().decode("utf-8")  # strict: nothing invalid reached the bytes
    assert json.loads(text) == {"n": "bad\udcff"}


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
    _emit({"name": "a → b \U0001f600"}, "json", stream)
    stream.flush()
    assert json.loads(raw.getvalue().decode("ascii")) == {"name": "a → b \U0001f600"}

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8")
    _emit({"name": "a → b"}, "json", stream)
    stream.flush()
    assert "→".encode() in raw.getvalue()  # a UTF-8 stream gets the character itself
