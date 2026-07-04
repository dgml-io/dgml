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

"""Tests for `dgml_core.errors` helpers."""

from __future__ import annotations

from dgml_core.errors import short_error_message


def test_short_error_message_includes_type_and_text() -> None:
    msg = short_error_message(RuntimeError("network down"))
    assert msg == "RuntimeError: network down"


def test_short_error_message_collapses_whitespace() -> None:
    msg = short_error_message(ValueError("line one\n\n   line two\t  end"))
    assert msg == "ValueError: line one line two end"


def test_short_error_message_truncates_long_text() -> None:
    msg = short_error_message(RuntimeError("x" * 1000))
    assert len(msg) == 300
    assert msg.endswith("...")
    assert msg.startswith("RuntimeError: ")


def test_short_error_message_respects_custom_limit() -> None:
    msg = short_error_message(RuntimeError("x" * 1000), limit=50)
    assert len(msg) == 50
    assert msg.endswith("...")


def test_short_error_message_bare_exception_is_type_name() -> None:
    assert short_error_message(RuntimeError()) == "RuntimeError"
