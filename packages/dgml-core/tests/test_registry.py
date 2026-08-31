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

"""The legacy per-machine index: reading what an older dgml left behind.

``workspaces.json`` is no longer written and nothing resolves through it — the list of
workspaces now lives in a :class:`~dgml_core.workspaces_store.WorkspacesStore`, whose
own tests are in ``test_workspaces_store.py``. What remains here is one job: read a
file this machine may already have, well enough for ``dgml workspace import`` and for
the migration that lifts a pre-upgrade ``storage`` snapshot into a workspace's own
config.

So these tests are about **tolerance**, not behaviour: every shape an older dgml (or a
hand edit) could have left must be readable without raising, because the file is
someone's only record of workspaces they still want.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dgml_core import registry
from dgml_core.errors import CorruptMetadata
from dgml_core.registry import RegistryEntry


def _write_index(rows: dict[str, dict[str, object]]) -> Path:
    """Write a legacy ``workspaces.json`` directly — the only way one comes into
    existence now, since nothing in dgml writes this file any more."""
    path = registry.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------- reading


def test_absent_index_reads_as_empty(tmp_path: Path) -> None:
    """The overwhelmingly common case now: a machine that never had one, or deleted it
    after importing. Not an error."""
    assert registry.read_registry() == {}
    assert registry.list_entries() == []
    assert registry.raw_entry_by_root(tmp_path / "ws") is None


def test_rows_are_read_and_sorted_by_id(tmp_path: Path) -> None:
    _write_index(
        {
            "ws_bbbbbbbbbbbbbbbb": {
                "name": "B",
                "organization": "beta",
                "root": str(tmp_path / "b"),
            },
            "ws_aaaaaaaaaaaaaaaa": {
                "name": "A",
                "organization": "acme",
                "root": str(tmp_path / "a"),
            },
        }
    )
    entries = registry.list_entries()
    assert [e.workspace_id for e in entries] == ["ws_aaaaaaaaaaaaaaaa", "ws_bbbbbbbbbbbbbbbb"]
    assert entries[0].name == "A"
    assert entries[1].organization == "beta"


def test_raw_entry_by_root_finds_the_row_and_names_its_id(tmp_path: Path) -> None:
    """``raw`` on purpose: the migration needs the pre-upgrade ``storage`` snapshot,
    which the dataclass deliberately drops."""
    root = tmp_path / "ws"
    _write_index(
        {
            "ws_aaaaaaaaaaaaaaaa": {
                "name": "W",
                "root": str(root),
                "storage": {"blobs": {"provider": "x:Y", "bucket": "b"}},
            }
        }
    )
    row = registry.raw_entry_by_root(root)
    assert row is not None
    assert row["workspace_id"] == "ws_aaaaaaaaaaaaaaaa"
    assert row["storage"] == {"blobs": {"provider": "x:Y", "bucket": "b"}}


def test_raw_entry_by_root_resolves_both_sides(tmp_path: Path) -> None:
    """A recorded root and the root being looked up may spell the same directory
    differently (a symlinked temp dir, a trailing ``/.``)."""
    root = tmp_path / "ws"
    root.mkdir()
    _write_index({"ws_aaaaaaaaaaaaaaaa": {"root": str(root) + "/."}})
    assert registry.raw_entry_by_root(root) is not None


# ------------------------------------------------------------------- tolerance


def test_legacy_storage_keys_are_read_as_noise(tmp_path: Path) -> None:
    """A pre-upgrade row carried the storage binding inline. Reading one must not raise
    — those keys are simply not this dataclass's business, and the binding they describe
    now lives in the workspace's own config."""
    entry = RegistryEntry.from_dict(
        "ws_legacyaaaaaaaaaa",
        {
            "name": "W",
            "organization": "acme",
            "root": str(tmp_path / "ws"),
            "storage_service": "acme",
            "storage": {"blobs": {"provider": "x:Y"}},
            "storage_fingerprint": "sha256:dead",
            "created_at": "2026-01-01T00:00:00Z",
            "schema_version": 1,
        },
    )
    assert entry.workspace_id == "ws_legacyaaaaaaaaaa"
    assert entry.name == "W"
    assert entry.root == str(tmp_path / "ws")
    assert not hasattr(entry, "storage")


@pytest.mark.parametrize(
    "row",
    [
        {},  # nothing at all
        {"name": 17, "organization": None},  # wrong types
        {"root": None},  # a row with no local root
        {"schema_version": "one"},  # unparseable version
        {"unknown_future_key": {"nested": True}},
    ],
)
def test_a_malformed_row_is_readable(row: dict[str, object]) -> None:
    """Refusing to parse would take the import command down with it, stranding every
    other workspace in the file. A row is best-effort, not validated."""
    entry = RegistryEntry.from_dict("ws_aaaaaaaaaaaaaaaa", row)
    assert entry.workspace_id == "ws_aaaaaaaaaaaaaaaa"


def test_a_non_dict_row_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    _write_index(
        {
            "ws_aaaaaaaaaaaaaaaa": "not-a-row",  # type: ignore[dict-item]
            "ws_bbbbbbbbbbbbbbbb": {"root": str(root)},
        }
    )
    assert registry.raw_entry_by_root(root) is not None


def test_unparseable_json_raises_corrupt_metadata(tmp_path: Path) -> None:
    """The one thing that *is* fatal, because there is nothing to salvage and silently
    reporting "no workspaces" would look like the file was already imported."""
    path = registry.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        registry.read_registry()


def test_a_non_object_top_level_raises(tmp_path: Path) -> None:
    path = registry.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        registry.read_registry()


# ------------------------------------------------------------------- read-only


def test_nothing_here_writes_the_index(tmp_path: Path) -> None:
    """The module is read-only now. This pins that: an index written by hand is
    byte-identical after every read path has run over it, and a machine with no index
    never grows one."""
    root = tmp_path / "ws"
    rows: dict[str, dict[str, object]] = {"ws_aaaaaaaaaaaaaaaa": {"name": "W", "root": str(root)}}
    path = _write_index(rows)
    before = path.read_bytes()

    registry.read_registry()
    registry.list_entries()
    registry.raw_entry_by_root(root)

    assert path.read_bytes() == before
    assert not [name for name in dir(registry) if name in {"register", "remove", "index_workspace"}]


def test_the_index_path_follows_the_user_config(tmp_path: Path) -> None:
    """It sits beside the user ``config.toml``, so the test isolation that redirects
    ``XDG_CONFIG_HOME`` covers this file too."""
    assert registry.registry_path().parent == (tmp_path / "xdg-home" / "dgml")
    assert registry.registry_path().name == "workspaces.json"
