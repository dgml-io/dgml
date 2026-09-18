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

from __future__ import annotations

import pytest
from dgml_core.ids import ID_ALPHABET, ID_LENGTH, is_record_id, new_id


def test_new_id_format() -> None:
    for _ in range(100):
        i = new_id()
        assert len(i) == ID_LENGTH
        assert all(c in ID_ALPHABET for c in i)
        # The subset invariant: everything new_id has ever generated stays valid
        # under the wider caller-supplied grammar, so no id needs migrating.
        assert is_record_id(i)


def test_new_id_collisions_rare() -> None:
    s = {new_id() for _ in range(10_000)}
    assert len(s) == 10_000


@pytest.mark.parametrize(
    "value",
    [
        "abc",  # the 3-char floor
        "a" * 40,  # the 40-char ceiling
        "a-b",
        "a_b",
        "0aa",
        "9-9_9",
        "invoice-2024-q1",
        "invoice_2024_q1",
    ],
)
def test_is_record_id_accepts(value: str) -> None:
    assert is_record_id(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "ab",  # one short of the floor
        "a" * 41,  # one past the ceiling
        "Z" * ID_LENGTH,
        "AbC",  # uppercase: would collide on a case-insensitive filesystem
        "-abc",  # leading separator
        "_abc",
        "a.b",
        "a/b",  # would break layout.pair_id and the dgmlx:// scheme
        "a b",
        "abc\n",  # proves the pattern is \Z-anchored, not $
        ".",
        "..",
    ],
)
def test_is_record_id_rejects(value: str) -> None:
    assert not is_record_id(value)
