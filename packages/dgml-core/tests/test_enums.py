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

"""``parse_enum``: the tolerant reader for persisted enum values."""

from __future__ import annotations

from enum import StrEnum

import pytest
from dgml_core.enums import parse_enum


class _Color(StrEnum):
    RED = "red"
    BLUE = "blue"


def test_a_known_value_is_its_member() -> None:
    assert parse_enum(_Color, "red") is _Color.RED
    assert parse_enum(_Color, _Color.BLUE) is _Color.BLUE


@pytest.mark.parametrize("raw", ["green", "", "RED", None, 3, 1.5, True, ["red"], {"red": 1}])
def test_anything_else_is_none_and_never_raises(raw: object) -> None:
    assert parse_enum(_Color, raw) is None
