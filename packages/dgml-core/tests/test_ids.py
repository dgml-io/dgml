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

from dgml_core.ids import ID_LENGTH, is_valid_id, new_id


def test_new_id_format() -> None:
    for _ in range(100):
        i = new_id()
        assert len(i) == ID_LENGTH
        assert is_valid_id(i)


def test_new_id_collisions_rare() -> None:
    s = {new_id() for _ in range(10_000)}
    assert len(s) == 10_000


def test_is_valid_id_rejects() -> None:
    assert not is_valid_id("abc")
    assert not is_valid_id("Z" * ID_LENGTH)
    assert not is_valid_id("a-b-c-d-e-f-g")
    assert not is_valid_id("")
