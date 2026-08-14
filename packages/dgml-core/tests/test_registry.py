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

"""Tests for the per-machine workspace registry + id-or-path addressing.

The autouse ``_isolate_user_config`` fixture (conftest) points
``XDG_CONFIG_HOME`` at a per-test tmp dir, so ``registry_path()`` is sandboxed and
the developer's real ``~/.config/dgml/workspaces.json`` is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dgml_core import registry
from dgml_core.errors import CorruptMetadata
from dgml_core.registry import RegistryEntry
from dgml_core.storage import Workspace


def _entry(workspace_id: str, root: Path, *, name: str = "W", org: str = "acme") -> RegistryEntry:
    return RegistryEntry(
        workspace_id=workspace_id,
        name=name,
        organization=org,
        root=str(root),
        storage_service="default",
        storage={
            "blobs": {"provider": "dgml_core.storage_local:LocalStore"},
            "docs": {"provider": "dgml_core.storage_local:LocalStore"},
        },
        storage_fingerprint="sha256:deadbeef",
        created_at="2026-08-05T12:00:00Z",
        schema_version=1,
    )


# ---------------------------------------------------------------- new_workspace_id


def test_new_workspace_id_shape_and_uniqueness() -> None:
    ids = {registry.new_workspace_id() for _ in range(50)}
    assert len(ids) == 50  # no collisions in a small sample
    for i in ids:
        assert i.startswith("ws_")
        assert len(i) == 19  # "ws_" + 16 base32 chars
        assert i[3:].isalnum() and i[3:].islower()


def test_mint_workspace_id_skips_a_registered_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mint re-rolls when the freshly-generated id already exists in the registry."""
    taken = "ws_takentakentaken1"
    registry.register(_entry(taken, tmp_path / "a"))

    # First roll collides with the registered id, second is free.
    rolls = iter([taken, "ws_freefreefreefre1"])
    monkeypatch.setattr(registry, "new_workspace_id", lambda: next(rolls))
    assert registry.mint_workspace_id() == "ws_freefreefreefre1"


# ---------------------------------------------------------------- registry I/O


def test_read_registry_absent_is_empty(tmp_path: Path) -> None:
    assert not registry.registry_path().exists()
    assert registry.read_registry() == {}
    assert registry.list_entries() == []
    assert registry.get("ws_nope") is None


def test_register_get_list_remove_roundtrip(tmp_path: Path) -> None:
    a = _entry("ws_aaaaaaaaaaaaaaaa", tmp_path / "a", name="A")
    b = _entry("ws_bbbbbbbbbbbbbbbb", tmp_path / "b", name="B")
    registry.register(a)
    registry.register(b)

    assert registry.get("ws_aaaaaaaaaaaaaaaa") == a
    assert {e.workspace_id for e in registry.list_entries()} == {a.workspace_id, b.workspace_id}
    # list is id-sorted (stable output)
    assert [e.workspace_id for e in registry.list_entries()] == [a.workspace_id, b.workspace_id]

    assert registry.remove("ws_aaaaaaaaaaaaaaaa") is True
    assert registry.get("ws_aaaaaaaaaaaaaaaa") is None
    assert registry.remove("ws_aaaaaaaaaaaaaaaa") is False  # already gone


def test_register_is_idempotent_upsert(tmp_path: Path) -> None:
    registry.register(_entry("ws_cccccccccccccccc", tmp_path / "c", name="Old"))
    registry.register(_entry("ws_cccccccccccccccc", tmp_path / "c", name="New"))
    entries = registry.list_entries()
    assert len(entries) == 1
    assert entries[0].name == "New"


def test_get_by_root_reverse_lookup(tmp_path: Path) -> None:
    registry.register(_entry("ws_dddddddddddddddd", tmp_path / "d"))
    hit = registry.get_by_root(tmp_path / "d")
    assert hit is not None and hit.workspace_id == "ws_dddddddddddddddd"
    assert registry.get_by_root(tmp_path / "nowhere") is None


def test_corrupt_registry_raises(tmp_path: Path) -> None:
    p = registry.registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        registry.read_registry()


def test_non_object_registry_raises(tmp_path: Path) -> None:
    p = registry.registry_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        registry.read_registry()


# ---------------------------------------------------------------- id-or-path resolution


def test_resolve_by_id_uses_registered_root(tmp_path: Path) -> None:
    root = (tmp_path / "acme-ws").resolve()
    wid = registry.new_workspace_id()
    registry.register(_entry(wid, root))
    assert Workspace.resolve(wid).root == root


def test_resolve_unregistered_token_is_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A bare token that is NOT a registered id resolves as a (relative) path.
    monkeypatch.chdir(tmp_path)
    assert Workspace.resolve("some_dir").root == (tmp_path / "some_dir").resolve()


def test_resolve_absolute_path_is_never_an_id(tmp_path: Path) -> None:
    p = tmp_path / "ws"
    assert Workspace.resolve(p).root == p.resolve()
    assert Workspace.resolve(str(p)).root == p.resolve()


def test_resolve_path_typed_id_round_trips(tmp_path: Path) -> None:
    # --workspace is argparse type=Path, so an id arrives as Path("ws_...").
    root = (tmp_path / "ws").resolve()
    wid = registry.new_workspace_id()
    registry.register(_entry(wid, root))
    assert Workspace.resolve(Path(wid)).root == root


def test_workspace_constructed_by_root_has_no_id(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    assert ws.workspace_id is None  # no registry / no workspace.json required


# ---------------------------------------------------------------- integrity seal


def test_verify_storage_seal_passes_and_no_ops(tmp_path: Path) -> None:
    from dgml_core.errors import StorageBackendMismatch

    ws = Workspace(root=tmp_path / "ws")
    # Unregistered → trust-on-first-use no-op.
    registry.verify_storage_seal(ws)
    # A consistent snapshot/fingerprint pair passes.
    registry.seal_entry(
        ws,
        workspace_id=registry.mint_workspace_id(),
        name="W",
        organization="acme",
        service="default",
    )
    registry.verify_storage_seal(ws)  # does not raise

    # An entry with an empty fingerprint is treated as unsealed (no raise).
    stub = _entry("ws_unsealedxxxxxxxx", tmp_path / "u")
    registry.register(RegistryEntry(**{**stub.__dict__, "storage_fingerprint": ""}))
    registry.verify_storage_seal(Workspace(root=tmp_path / "u"))

    # Hand-edit the sealed snapshot without fixing the fingerprint → mismatch.
    entry = registry.get_by_root(ws.root)
    assert entry is not None
    tampered = RegistryEntry(
        **{
            **entry.__dict__,
            "storage": {"blobs": {"provider": "other:Store"}, "docs": entry.storage["docs"]},
        }
    )
    registry.register(tampered)
    with pytest.raises(StorageBackendMismatch):
        registry.verify_storage_seal(ws)
