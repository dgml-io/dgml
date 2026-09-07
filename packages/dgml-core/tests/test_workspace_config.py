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

"""The workspace's own ``config.toml``: identity block, storage binding, and seal.

Two properties carry most of the weight here and are worth stating up front, because
the rest of the file is mostly their consequences:

1. **Identity does not layer.** The ``[workspace]`` block is read straight from the
   workspace's file, never through the merged-config loader — otherwise a stray key in
   the user-level config would apply to every workspace on the machine.
2. **Storage does not layer either.** A service the workspace defines itself is used
   whole. This is what makes a workspace self-describing, and what keeps a user-config
   edit from silently moving another workspace's data.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from dgml_core import workspace_config as wc
from dgml_core.errors import CorruptMetadata, StorageBackendMismatch, StorageConfigInvalid
from dgml_core.storage import Workspace
from dgml_core.storage_resolve import (
    resolve_store_configs,
    storage_fingerprint_pair,
    verify_storage_fingerprint,
)
from dgml_core.workspace_id import mint_workspace_id
from dgml_core.workspaces_resolve import default_workspaces_store

LOCAL = "dgml_core.storage_local:LocalStore"

# Which backing the workspaces built by `_ws` use for this test run. Mutated by the
# autouse `_backing` fixture below so every test in this file runs twice — once with the
# config as a file in the workspace directory, once with it held in the machine's store
# of workspaces — without touching a single call site.
_BACKING = ["file"]


@pytest.fixture(autouse=True, params=["file", "store"])
def _backing(request: pytest.FixtureRequest) -> Iterator[None]:
    """Run the whole file against both ways a config can be backed.

    This is the highest-value coverage in the change: everything below —
    comment preservation, banner absorption, span replacement, the round-trip
    verification, the seal — is a *text* operation, and the whole point of routing it
    through :func:`~dgml_core.workspace_config.read_config_state` /
    :func:`~dgml_core.workspace_config.write_config_text` is that the behaviour cannot
    depend on where that text is kept. Asserting that once, here, is worth more than any
    new test: a backend-dependent splice would mean "your comments survive, depending on
    where your workspace lives"."""
    _BACKING[0] = request.param
    yield
    _BACKING[0] = "file"


def _ws(tmp_path: Path, text: str = "") -> Workspace:
    """A workspace whose config is a file in its own directory, or is held in the
    machine's store of workspaces (per the current parametrization).

    The store-backed variant roots the workspace at its folder in the store, which is
    what the local backend does in production — so ``config_path`` still names the file
    the store writes, and the file-level assertions below stay meaningful while the
    store code path is the one actually exercised."""
    if _BACKING[0] == "store":
        store = default_workspaces_store()
        wid = mint_workspace_id(store)
        store.write_config(wid, text)
        return Workspace(root=store.workspace_root(wid), workspaces_id=wid)
    root = tmp_path / "ws"
    root.mkdir(exist_ok=True)
    if text:
        (root / "config.toml").write_text(text)
    return Workspace(root=root)


def _reopen(ws: Workspace) -> Workspace:
    """The same workspace, freshly constructed, so nothing is memoized from before a
    write (see ``Workspace.config_text`` and ``store_configs``)."""
    return Workspace(
        root=ws.root, config_override=ws.config_override, workspaces_id=ws.workspaces_id
    )


def _seal(ws: Workspace) -> str:
    """Seal a *fresh* Workspace so no stale ``store_configs`` is memoized."""
    return wc.reseal(_reopen(ws))


# --------------------------------------------------- the parametrization is real


def test_the_backing_under_test_is_the_one_being_exercised(tmp_path: Path) -> None:
    """Guards every other test in this file against going vacuous.

    If ``_ws`` ever stopped producing a store-backed workspace, the whole suite would
    still pass — twice over the same file path — and the claim that splicing is
    backend-independent would be untested. So assert the plumbing directly: a
    store-backed workspace reads through the store, and a write lands there."""
    ws = _ws(tmp_path, "[workspace]\nname = 'W'\n")
    if _BACKING[0] == "file":
        assert ws.workspaces_id is None
        assert ws.config_path.is_file()
        return

    assert ws.workspaces_id is not None
    store = default_workspaces_store()
    found = store.read_config(ws.workspaces_id)
    assert found is not None and "name = 'W'" in found

    wc.write_identity(ws, organization="acme")
    written = store.read_config(ws.workspaces_id)
    assert written is not None and "acme" in written


# ------------------------------------------------------------------- identity


def test_machine_keys_round_trip(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    wc.write_identity(
        ws, workspace_id="ws_a", name="W", organization="Acme", storage_service="svcA"
    )
    identity = wc.read_identity(ws)
    assert (identity.workspace_id, identity.name, identity.organization) == ("ws_a", "W", "Acme")
    assert identity.storage_service == "svcA"


def test_write_identity_is_merge_preserving(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    wc.write_identity(ws, workspace_id="ws_a", organization="Acme")
    wc.write_identity(ws, storage_fingerprint="sha256:beef")
    identity = wc.read_identity(ws)
    assert identity.workspace_id == "ws_a"
    assert identity.organization == "Acme"
    assert identity.storage_fingerprint == "sha256:beef"


def test_absent_config_reads_as_empty_identity(tmp_path: Path) -> None:
    assert wc.read_identity(_ws(tmp_path)) == wc.WorkspaceIdentity()


def test_machine_keys_are_read_only_from_the_workspace_file(tmp_path: Path) -> None:
    """A ``[workspace]`` block in the *user* config must not be picked up.

    If it were, one stray key in ``~/.config/dgml/config.toml`` would give every
    workspace on the machine the same id, and ``DGML_WORKSPACE__STORAGE_FINGERPRINT``
    would let anyone silence the drift guard for a single invocation."""
    from dgml_core.storage import user_config_path

    user = user_config_path()
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text('[workspace]\nworkspace_id = "ws_from_user_config"\n')
    assert wc.read_identity(_ws(tmp_path)).workspace_id is None


# ---------------------------------------------------- storage does not layer


def test_a_self_defined_service_ignores_the_user_layer(tmp_path: Path) -> None:
    """Same service name in both files: the workspace's table wins **whole**, and no
    key from the user's table leaks in."""
    from dgml_core.storage import user_config_path

    user = user_config_path()
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text(f'[storage.acme.blobs]\nprovider = "{LOCAL}"\nprefix = "from-user"\n')

    ws = _ws(tmp_path, f'[storage.acme.blobs]\nprovider = "{LOCAL}"\nprefix = "from-workspace"\n')
    wc.write_identity(ws, storage_service="acme")
    blob_cfg, _ = resolve_store_configs(ws)
    assert blob_cfg.options["prefix"] == "from-workspace"


def test_an_undefined_service_falls_back_to_the_user_layer(tmp_path: Path) -> None:
    """A workspace that defines no table still resolves a shared template — the
    ergonomic that keeps `--storage acme` useful across many workspaces."""
    from dgml_core.storage import user_config_path

    user = user_config_path()
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text(f'[storage.acme.blobs]\nprovider = "{LOCAL}"\nprefix = "from-user"\n')

    ws = _ws(tmp_path)
    wc.write_identity(ws, storage_service="acme")
    blob_cfg, _ = resolve_store_configs(ws)
    assert blob_cfg.options["prefix"] == "from-user"


def test_no_config_at_all_resolves_to_the_bundled_local_store(tmp_path: Path) -> None:
    blob_cfg, doc_cfg = resolve_store_configs(_ws(tmp_path))
    assert blob_cfg.provider == LOCAL
    assert blob_cfg == doc_cfg  # one instance serves both roles


def test_sibling_services_do_not_leak_into_options(tmp_path: Path) -> None:
    """An inline ``[storage]`` shares the merged section with named services. A sibling
    name must not arrive as a store *option* — it would fail the provider's unknown-field
    check while naming a service the workspace has nothing to do with."""
    from dgml_core.storage import user_config_path

    user = user_config_path()
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text(f'[storage.other.blobs]\nprovider = "{LOCAL}"\n')

    ws = _ws(tmp_path, f'[storage]\nprovider = "{LOCAL}"\n')
    blob_cfg, _ = resolve_store_configs(ws)
    assert "other" not in blob_cfg.options


# ----------------------------------------------------------------- the seal


def test_verify_no_ops_when_unsealed(tmp_path: Path) -> None:
    """Trust-on-first-use: a workspace that has never been sealed opens."""
    verify_storage_fingerprint(_ws(tmp_path, f'[storage]\nprovider = "{LOCAL}"\n'))


def test_seal_matches_the_resolved_pair(tmp_path: Path) -> None:
    ws = _ws(tmp_path, f'[storage]\nprovider = "{LOCAL}"\n')
    recorded = _seal(ws)
    assert recorded == storage_fingerprint_pair(*resolve_store_configs(ws))
    verify_storage_fingerprint(Workspace(root=ws.root))


def test_verify_raises_on_edited_storage_table(tmp_path: Path) -> None:
    ws = _ws(tmp_path, f'[storage]\nprovider = "{LOCAL}"\n')
    _seal(ws)
    ws.config_path.write_text(
        ws.config_path.read_text().replace(f'provider = "{LOCAL}"', 'provider = "other:Store"', 1)
    )
    with pytest.raises(StorageBackendMismatch):
        verify_storage_fingerprint(Workspace(root=ws.root))


def test_credential_rotation_does_not_trip_the_seal(tmp_path: Path) -> None:
    """Secret-hinted options are outside the hash, so rotating one is not "the store
    moved"."""
    ws = _ws(tmp_path, f'[storage]\nprovider = "{LOCAL}"\napi_key = "old"\n')
    _seal(ws)
    ws.config_path.write_text(ws.config_path.read_text().replace('"old"', '"rotated"'))
    verify_storage_fingerprint(Workspace(root=ws.root))


def test_a_copied_workspace_keeps_its_seal(tmp_path: Path) -> None:
    """``root`` is outside the hash — where the config lives is not where the data
    lives, so ``cp -r ws ws2`` must not read as drift."""
    import shutil

    ws = _ws(tmp_path, f'[storage]\nprovider = "{LOCAL}"\n')
    _seal(ws)
    copy = tmp_path / "copy"
    shutil.copytree(ws.root, copy)
    verify_storage_fingerprint(Workspace(root=copy))


def test_user_layer_edit_does_not_trip_a_self_defined_seal(tmp_path: Path) -> None:
    """The corollary of replace-not-merge: a workspace carrying its own service is
    immune to edits in the user-level config."""
    from dgml_core.storage import user_config_path

    ws = _ws(tmp_path, f'[storage.acme.blobs]\nprovider = "{LOCAL}"\n')
    wc.write_identity(ws, storage_service="acme")
    _seal(ws)

    user = user_config_path()
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text(f'[storage.acme.blobs]\nprovider = "{LOCAL}"\nprefix = "meddling"\n')
    verify_storage_fingerprint(Workspace(root=ws.root))


# -------------------------------------------------------------- the writer


def test_reseal_preserves_comments_and_unrelated_tables(tmp_path: Path) -> None:
    original = (
        "# my own notes\n"
        "[models]\n"
        'light = "x"   # trailing comment\n'
        "\n"
        f'[storage]\nprovider = "{LOCAL}"\n'
    )
    ws = _ws(tmp_path, original)
    _seal(ws)
    text = ws.config_path.read_text()
    assert "# my own notes" in text
    assert 'light = "x"   # trailing comment' in text


def test_a_user_comment_above_a_table_survives_rewrites(tmp_path: Path) -> None:
    """Only dgml's own banner is absorbed on rewrite. Eating a user's note would make
    every reseal quietly delete part of their file."""
    ws = _ws(tmp_path, f'# KEEP ME\n[storage.acme.blobs]\nprovider = "{LOCAL}"\n')
    wc.write_storage_table(ws, "acme", {"blobs": {"provider": LOCAL, "prefix": "p"}})
    wc.write_storage_table(ws, "acme", {"blobs": {"provider": LOCAL, "prefix": "q"}})
    assert "# KEEP ME" in ws.config_path.read_text()


def test_rewriting_replaces_rather_than_appends(tmp_path: Path) -> None:
    """A table written twice must not end up declared twice — invalid TOML, and the
    round-trip check is what catches it."""
    ws = _ws(tmp_path)
    for prefix in ("a", "b", "c"):
        wc.write_storage_table(ws, "acme", {"blobs": {"provider": LOCAL, "prefix": prefix}})
    assert ws.config_path.read_text().count("[storage.acme.blobs]") == 1
    table = wc.read_storage_table(ws, "acme")
    assert table is not None and table["blobs"]["prefix"] == "c"


def test_identity_banner_is_not_stacked(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    for org in ("A", "B", "C"):
        wc.write_identity(ws, organization=org)
    assert ws.config_path.read_text().count("# Written by dgml") == 1


def test_writes_are_stable(tmp_path: Path) -> None:
    """Re-writing identical values must not churn the file, or every command would
    produce a spurious diff in a workspace under version control."""
    ws = _ws(tmp_path, f'[storage]\nprovider = "{LOCAL}"\n')
    wc.write_identity(ws, workspace_id="ws_a", name="W", organization="Acme")
    first = ws.config_path.read_text()
    wc.write_identity(ws, workspace_id="ws_a", name="W", organization="Acme")
    assert ws.config_path.read_text() == first


def test_config_without_trailing_newline_is_handled(tmp_path: Path) -> None:
    ws = _ws(tmp_path, f'[storage]\nprovider = "{LOCAL}"')  # no trailing \n
    wc.write_identity(ws, workspace_id="ws_a")
    assert wc.read_identity(ws).workspace_id == "ws_a"


def test_quotes_and_backslashes_in_organization_round_trip(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    tricky = 'Acme "Corp" \\ Ltd'
    wc.write_identity(ws, organization=tricky)
    assert wc.read_identity(ws).organization == tricky


def test_control_characters_are_refused(tmp_path: Path) -> None:
    with pytest.raises(StorageConfigInvalid):
        wc.write_identity(_ws(tmp_path), organization="bad\nname")


def test_unparseable_config_is_never_appended_to(tmp_path: Path) -> None:
    ws = _ws(tmp_path, "[models\nbroken = ")
    with pytest.raises(CorruptMetadata):
        wc.write_identity(ws, workspace_id="ws_a")


@pytest.mark.parametrize("shape", ["[[workspace]]\n", '[workspace.sub]\nx = "y"\n'])
def test_unsupported_workspace_table_shapes_are_refused(tmp_path: Path, shape: str) -> None:
    """dgml owns ``[workspace]``. A shape it never writes is a hand-edit, and guessing
    at it risks destroying data — refuse instead."""
    ws = _ws(tmp_path, shape)
    with pytest.raises(CorruptMetadata):
        wc.write_identity(ws, workspace_id="ws_a")


def test_non_scalar_option_is_refused(tmp_path: Path) -> None:
    with pytest.raises(StorageConfigInvalid):
        wc.write_storage_table(_ws(tmp_path), "acme", {"blobs": {"provider": LOCAL, "x": object()}})


# ------------------------------------------------------------ config override


def test_config_override_is_honored(tmp_path: Path) -> None:
    """A workspace whose config lives outside its directory resolves from that file —
    and writes land there, not in the workspace root."""
    external = tmp_path / "elsewhere" / "acme.toml"
    external.parent.mkdir(parents=True)
    external.write_text(f'[storage.acme.blobs]\nprovider = "{LOCAL}"\nprefix = "external"\n')

    root = tmp_path / "ws"
    root.mkdir()
    ws = Workspace(root=root, config_override=external)
    wc.write_identity(ws, storage_service="acme")

    blob_cfg, _ = resolve_store_configs(ws)
    assert blob_cfg.options["prefix"] == "external"
    assert blob_cfg.root == root  # the anchor is still the workspace, not the config
    assert not (root / "config.toml").exists()
    assert wc.read_identity(ws).storage_service == "acme"
