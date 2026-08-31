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

"""Workspace id minting and the id-vs-path shape test."""

from __future__ import annotations

import pytest
from dgml_core.workspace_id import ID_PREFIX, is_workspace_id, new_workspace_id


def test_new_workspace_id_shape_and_uniqueness() -> None:
    ids = {new_workspace_id() for _ in range(50)}
    assert len(ids) == 50
    for wid in ids:
        assert wid.startswith(ID_PREFIX)
        assert is_workspace_id(wid), wid


@pytest.mark.parametrize(
    "value",
    [
        "ws_abcdefghijklmnop",
        "ws_2345672345672345",
        "ws_fixturexxxxxxxxx",  # the CLI test fixture's id — must stay addressable
    ],
)
def test_is_workspace_id_accepts_real_ids(value: str) -> None:
    assert is_workspace_id(value)


@pytest.mark.parametrize(
    "value",
    [
        "ws_abcdefghijklmno",  # 15 chars — one short
        "ws_abcdefghijklmnopq",  # 17 chars — one long
        "ws_ABCDEFGHIJKLMNOP",  # uppercase is not in the base32-lower alphabet
        "ws_abcdefghijklmn0p",  # 0 and 1 are outside [a-z2-7]
        "ws_abcdefghijklmn1p",
        "./ws_abcdefghijklmnop",  # the documented escape for a same-named directory
        "ws_abcdefghijklmnop/",  # a trailing separator makes it a path
        "ws_abcdefghij/klmnop",
        "ws_abcdefghijklmnop.bak",
        "workspace_abcdefghijklmnop",  # wrong prefix
        "dgml-workspace",
        "",
        ".",
    ],
)
def test_is_workspace_id_rejects_paths_and_near_misses(value: str) -> None:
    assert not is_workspace_id(value)


def test_is_workspace_id_is_anchored_at_both_ends() -> None:
    """A regex without both anchors would accept an id with anything appended, so a
    path that merely *contains* an id would resolve as one."""
    wid = new_workspace_id()
    assert not is_workspace_id(f"/tmp/{wid}")
    assert not is_workspace_id(f"{wid}\n")
    assert not is_workspace_id(f"{wid}{wid}")
