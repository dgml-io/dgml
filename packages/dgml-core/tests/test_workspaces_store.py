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

"""The ``WorkspacesStore`` contract, and the bundled local-directory backend.

The contract tests are written against factories rather than a concrete class so the
Mongo backend can import and re-run them unchanged. ``DerivedOnlyStore`` strips the
local backend's optimized overrides, which is what proves the base-class defaults a
third-party backend inherits actually work — the same trick ``DefaultBridgeStore``
plays for ``BlobStore``'s path bridge.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from dgml_core.errors import CorruptMetadata, StorageProviderUnresolvable, WorkspacesConfigInvalid
from dgml_core.provider import import_provider_class
from dgml_core.storage_local import LocalStore
from dgml_core.workspace_id import mint_workspace_id, new_workspace_id
from dgml_core.workspaces_local import LocalDirWorkspacesStore
from dgml_core.workspaces_resolve import (
    DEFAULT_WORKSPACES_PROVIDER,
    default_workspaces_store,
    load_workspaces_config,
    make_workspaces_store,
)
from dgml_core.workspaces_store import (
    WORKSPACES_ENV_VAR,
    WorkspacesConfig,
    WorkspacesStore,
    default_workspaces_root,
)

CONFIG = """\
# a comment the user wrote
[workspace]
name = "Acme Contracts"
organization = "acme"
storage_service = "bym"
created_at = "2026-08-26T18:04:11Z"
"""


class DerivedOnlyStore(LocalDirWorkspacesStore):
    """The local backend with its optimized overrides removed, so the contract runs
    against the base class's derived implementations.

    A third-party backend that implements only the four primitives gets exactly these,
    so if they are broken the breakage is invisible until someone else writes a store.

    ``workspace_root`` is deliberately *not* stripped: it is not an optimization here
    but this backend's actual answer (the folder is where the workspace lives), whereas
    the base default is the fallback a backend with no filesystem of its own needs."""

    exists = WorkspacesStore.exists
    list_ids = WorkspacesStore.list_ids
    list_entries = WorkspacesStore.list_entries


StoreFactory = Callable[[Path], WorkspacesStore]


def _local(root: Path) -> WorkspacesStore:
    cfg = WorkspacesConfig(provider=DEFAULT_WORKSPACES_PROVIDER, options={"root": str(root)})
    return LocalDirWorkspacesStore(LocalDirWorkspacesStore.parse_config(cfg))


def _derived_only(root: Path) -> WorkspacesStore:
    cfg = WorkspacesConfig(provider=DEFAULT_WORKSPACES_PROVIDER, options={"root": str(root)})
    return DerivedOnlyStore(DerivedOnlyStore.parse_config(cfg))


STORE_FACTORIES: list[StoreFactory] = [_local, _derived_only]


@pytest.fixture(params=STORE_FACTORIES, ids=["local", "derived-only"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> WorkspacesStore:
    factory: StoreFactory = request.param
    return factory(tmp_path / "workspaces")


# ----------------------------------------------------------------- the contract


def test_write_then_read_round_trips(store: WorkspacesStore) -> None:
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    found = store.read_config(wid)
    assert found is not None
    assert found == CONFIG


def test_read_missing_is_none(store: WorkspacesStore) -> None:
    assert store.read_config(new_workspace_id()) is None


def test_write_replaces(store: WorkspacesStore) -> None:
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    store.write_config(wid, "[workspace]\nname = 'Renamed'\n")
    found = store.read_config(wid)
    assert found is not None
    assert "Renamed" in found
    assert "Acme Contracts" not in found


def test_exists_tracks_write_and_delete(store: WorkspacesStore) -> None:
    wid = new_workspace_id()
    assert not store.exists(wid)
    store.write_config(wid, CONFIG)
    assert store.exists(wid)
    assert store.delete(wid) is True
    assert not store.exists(wid)


def test_delete_absent_is_false(store: WorkspacesStore) -> None:
    assert store.delete(new_workspace_id()) is False


def test_list_ids_is_sorted_and_complete(store: WorkspacesStore) -> None:
    ids = sorted(new_workspace_id() for _ in range(4))
    for wid in ids:
        store.write_config(wid, CONFIG)
    assert store.list_ids() == ids


def test_list_ids_empty_when_nothing_written(store: WorkspacesStore) -> None:
    """A store that has never been written to lists nothing rather than failing — the
    directory does not exist yet on a fresh machine."""
    assert store.list_ids() == []
    assert store.list_configs() == {}
    assert store.list_entries() == []


def test_list_entries_derives_identity_from_the_config(store: WorkspacesStore) -> None:
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    (entry,) = store.list_entries()
    assert entry.workspace_id == wid
    assert entry.name == "Acme Contracts"
    assert entry.organization == "acme"
    assert entry.storage_service == "bym"
    assert entry.created_at == "2026-08-26T18:04:11Z"


def test_list_entries_uses_the_store_key_over_a_disagreeing_block(
    store: WorkspacesStore,
) -> None:
    """The address a workspace was found under wins over a hand-edited id inside it,
    so a copied config can never make one workspace masquerade as another."""
    wid = new_workspace_id()
    store.write_config(wid, '[workspace]\nworkspace_id = "ws_aaaaaaaaaaaaaaaa"\n')
    (entry,) = store.list_entries()
    assert entry.workspace_id == wid


def test_list_entries_tolerates_a_config_with_no_identity_block(
    store: WorkspacesStore,
) -> None:
    wid = new_workspace_id()
    store.write_config(wid, "[storage.default]\nprovider = 'x:Y'\n")
    (entry,) = store.list_entries()
    assert entry.workspace_id == wid
    assert entry.name is None


def test_config_text_is_preserved_byte_for_byte(store: WorkspacesStore) -> None:
    """The splice machinery in ``workspace_config`` promises a user's comments, key
    order and formatting survive a write. That promise has to hold identically on every
    backend, or the guarantee is really "depends where your workspace lives"."""
    hostile = (
        "# leading comment\r\n"
        "\n"
        "[workspace]\n"
        'name = "Zed"   # trailing comment\n'
        'organization = "acme"\n'
        "\n"
        "# a comment between tables — plus non-ASCII: café ☕\n"
        "[storage.bym.blobs]\n"
        'bucket = "dgml-dev"\n'
        'endpoint_url = "http://localhost:9000"\n'
        "\n"
        "[storage.bym.docs]\n"
        "mongo_port = 27017\n"
        "a.b.c = 1\n"
        "\n"
        "[[thing]]\n"
        'label = """multi\nline"""'  # deliberately no trailing newline
    )
    wid = new_workspace_id()
    store.write_config(wid, hostile)
    found = store.read_config(wid)
    assert found is not None
    assert found == hostile


def test_read_text_round_trips_as_the_write_token(store: WorkspacesStore) -> None:
    """Whatever ``read_config`` hands back is accepted by ``write_config`` as
    ``expected_text``, so a caller never has to know whether its backend detects
    conflicts."""
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    text = store.read_config(wid)
    assert text is not None
    store.write_config(wid, text + "\n[models]\n", expected_text=text)
    assert store.read_config(wid) == CONFIG + "\n[models]\n"


def test_writing_identical_text_is_not_a_conflict(store: WorkspacesStore) -> None:
    """A no-op write succeeds rather than raising: the stored text is already what the
    writer wants, so there is nothing to lose. Backends that detect conflicts must not
    treat "unchanged" as "changed"."""
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    store.write_config(wid, CONFIG, expected_text=CONFIG)
    assert store.read_config(wid) == CONFIG


def test_expected_text_none_is_unconditional(store: WorkspacesStore) -> None:
    """``None`` means "write regardless", which is what a fresh workspace and a backend
    with one writer by construction both rely on."""
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    store.write_config(wid, "[workspace]\nname = 'Clobbered'\n", expected_text=None)
    assert store.read_config(wid) == "[workspace]\nname = 'Clobbered'\n"


def test_verbatim_text_is_accepted_back_unchanged(store: WorkspacesStore) -> None:
    """CRLF, non-ASCII, and both Unicode normal forms must survive a read and be
    *accepted* as ``expected_text``, not judged different.

    The NFC/NFD pair is the load-bearing case: it proves nothing in the path normalizes.
    A backend that did would break conflict detection, not merely rewrite a file — and
    the two forms must stay distinct values, which the last assertion pins."""
    nfc, nfd = "café", "café"
    assert nfc != nfd  # guard: the literals really are the two forms
    for label, text in (
        ("crlf", "[workspace]\r\nname = 'W'\r\n"),
        ("no-trailing-newline", "[workspace]\nname = 'W'"),
        ("nfc", f"[workspace]\nname = '{nfc}'\n"),
        ("nfd", f"[workspace]\nname = '{nfd}'\n"),
    ):
        wid = new_workspace_id()
        store.write_config(wid, text)
        found = store.read_config(wid)
        assert found == text, label
        # Accepted as the token: no spurious conflict from an encoding round trip.
        store.write_config(wid, text, expected_text=found)
        assert store.read_config(wid) == text, label

    # And the two forms are stored as distinct texts, not folded together.
    a, b = new_workspace_id(), new_workspace_id()
    store.write_config(a, nfc)
    store.write_config(b, nfd)
    assert store.read_config(a) != store.read_config(b)


def test_label_is_non_empty(store: WorkspacesStore) -> None:
    assert store.label()


# ------------------------------------------------- the local backend specifically


def test_folder_name_is_the_id_and_holds_the_config(tmp_path: Path) -> None:
    root = tmp_path / "workspaces"
    store = _local(root)
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    assert (root / wid / "config.toml").read_text(encoding="utf-8") == CONFIG


def test_workspace_root_is_the_folder(tmp_path: Path) -> None:
    store = _local(tmp_path / "workspaces")
    wid = new_workspace_id()
    assert isinstance(store, LocalDirWorkspacesStore)
    assert store.workspace_root(wid) == tmp_path / "workspaces" / wid


def test_stray_directories_are_not_workspaces(tmp_path: Path) -> None:
    """The parent is an ordinary directory a user may look inside, so anything whose
    name is not a well-formed id has to be ignored rather than half-listed."""
    root = tmp_path / "workspaces"
    store = _local(root)
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    for stray in ("notes", "ws_short", f"{wid}.bak", "ws_UPPERCASEAAAAAAAA"):
        (root / stray).mkdir(parents=True, exist_ok=True)
        (root / stray / "config.toml").write_text(CONFIG, encoding="utf-8")
    (root / "loose.toml").write_text(CONFIG, encoding="utf-8")
    assert store.list_ids() == [wid]


def test_a_folder_without_a_config_is_not_listed(tmp_path: Path) -> None:
    root = tmp_path / "workspaces"
    store = _local(root)
    (root / new_workspace_id()).mkdir(parents=True)
    assert store.list_ids() == []


def test_delete_unlists_but_keeps_workspace_data(tmp_path: Path) -> None:
    """``delete`` drops a listing entry; it is not a way to lose a corpus."""
    root = tmp_path / "workspaces"
    store = _local(root)
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    pdf = root / wid / "files" / "f1" / "source.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.7\n")

    assert store.delete(wid) is True
    assert store.list_ids() == []
    assert pdf.is_file()


def test_delete_prunes_the_folder_when_nothing_remains(tmp_path: Path) -> None:
    root = tmp_path / "workspaces"
    store = _local(root)
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    store.delete(wid)
    assert not (root / wid).exists()


# ------------------------------------------------------------------ minting ids


def test_mint_rerolls_past_an_id_the_store_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _local(tmp_path / "workspaces")
    taken = new_workspace_id()
    free = new_workspace_id()
    store.write_config(taken, CONFIG)
    rolls = iter([taken, free])
    monkeypatch.setattr("dgml_core.workspace_id.new_workspace_id", lambda: next(rolls))
    assert mint_workspace_id(store) == free


def test_mint_without_a_store_does_not_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """No store means no list to consult — an unchecked mint is the honest answer, not
    a silent fallback to some other list."""
    monkeypatch.setattr("dgml_core.workspace_id.new_workspace_id", lambda: "ws_aaaaaaaaaaaaaaaa")
    assert mint_workspace_id() == "ws_aaaaaaaaaaaaaaaa"


# ---------------------------------------------------------------- configuration


def test_unknown_option_is_rejected_naming_the_section() -> None:
    cfg = WorkspacesConfig(provider=DEFAULT_WORKSPACES_PROVIDER, options={"bucket": "x"})
    with pytest.raises(WorkspacesConfigInvalid, match="workspaces"):
        LocalDirWorkspacesStore.parse_config(cfg)


@pytest.mark.parametrize("root", ["", "   ", "relative/path", 17])
def test_bad_root_is_rejected(root: object) -> None:
    cfg = WorkspacesConfig(provider=DEFAULT_WORKSPACES_PROVIDER, options={"root": root})
    with pytest.raises(WorkspacesConfigInvalid):
        LocalDirWorkspacesStore.parse_config(cfg)


def test_default_root_honors_the_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(WORKSPACES_ENV_VAR, str(tmp_path / "elsewhere"))
    assert default_workspaces_root() == (tmp_path / "elsewhere").resolve()


def test_default_root_is_a_visible_home_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not hidden and not under an XDG base dir: it holds source PDFs and page images,
    and it reads as the machine-wide plural of ``./dgml-workspace``."""
    monkeypatch.delenv(WORKSPACES_ENV_VAR, raising=False)
    assert default_workspaces_root() == (Path.home() / "dgml-workspaces").resolve()


def test_no_user_config_resolves_to_the_bundled_local_store() -> None:
    """Zero config still has a working list of workspaces."""
    cfg = load_workspaces_config()
    assert cfg.provider == DEFAULT_WORKSPACES_PROVIDER
    assert isinstance(make_workspaces_store(cfg), LocalDirWorkspacesStore)


def test_a_table_without_a_provider_means_the_local_store(tmp_path: Path) -> None:
    """``[workspaces] root = "/data/ws"`` means what it obviously means."""
    user_config = tmp_path / "xdg-home" / "dgml" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(f'[workspaces]\nroot = "{tmp_path / "ws"}"\n', encoding="utf-8")
    cfg = load_workspaces_config()
    assert cfg.provider == DEFAULT_WORKSPACES_PROVIDER
    store = make_workspaces_store(cfg)
    assert isinstance(store, LocalDirWorkspacesStore)
    assert store.root == tmp_path / "ws"


def test_a_non_table_workspaces_section_is_rejected(tmp_path: Path) -> None:
    user_config = tmp_path / "xdg-home" / "dgml" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text('workspaces = "nope"\n', encoding="utf-8")
    with pytest.raises(WorkspacesConfigInvalid, match="must be a table"):
        load_workspaces_config()


def test_unparseable_user_config_is_corrupt_metadata(tmp_path: Path) -> None:
    user_config = tmp_path / "xdg-home" / "dgml" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text("[workspaces\n", encoding="utf-8")
    with pytest.raises(CorruptMetadata):
        load_workspaces_config()


def test_the_workspaces_section_of_a_workspace_config_cannot_redirect_resolution(
    tmp_path: Path,
) -> None:
    """A workspace must not be able to redefine the store that lists it — it was
    already used to find that workspace. Only the user config is consulted."""
    workspace_config = tmp_path / "ws" / "config.toml"
    workspace_config.parent.mkdir(parents=True)
    workspace_config.write_text('[workspaces]\nprovider = "nope:Nope"\n', encoding="utf-8")
    assert load_workspaces_config().provider == DEFAULT_WORKSPACES_PROVIDER


def test_default_store_is_memoized(tmp_path: Path) -> None:
    assert default_workspaces_store() is default_workspaces_store()


# ------------------------------------------------- provider namespaces stay apart


def test_a_storage_provider_is_not_a_workspaces_provider() -> None:
    """The base-class check is what keeps the two ``provider`` namespaces from bleeding
    into each other; without it a store could be resolved into either slot."""
    with pytest.raises(StorageProviderUnresolvable, match="WorkspacesStore"):
        import_provider_class(
            "dgml_core.storage_local:LocalStore", WorkspacesStore, kind="workspaces"
        )


def test_a_workspaces_provider_is_not_a_storage_provider() -> None:
    with pytest.raises(StorageProviderUnresolvable, match="LocalStore"):
        import_provider_class(
            "dgml_core.workspaces_local:LocalDirWorkspacesStore", LocalStore, kind="storage"
        )


def test_provider_failure_names_the_section_it_came_from() -> None:
    with pytest.raises(StorageProviderUnresolvable, match="workspaces provider"):
        import_provider_class("no-colon-here", WorkspacesStore, kind="workspaces")


def test_workspace_root_honors_a_declared_workspace_path(tmp_path: Path) -> None:
    """A workspace adopted from a directory elsewhere keeps its files there, recorded as
    `workspace_path`. The store has to report *that* as its root.

    Regression: it returned the standard folder instead, so `dgml workspace list` showed
    an imported workspace's root as a directory its data was not in. Caught by driving
    the real CLI, not by the contract tests — which is why this one exists.
    """
    root = tmp_path / "workspaces"
    store = _local(root)
    wid = new_workspace_id()
    elsewhere = tmp_path / "corpus-on-another-disk"
    store.write_config(
        wid,
        f'[workspace]\nstorage_service = "default"\n\n'
        f'[storage.default]\nprovider = "dgml_core.storage_local:LocalStore"\n'
        f'workspace_path = "{elsewhere}"\n',
    )
    assert store.workspace_root(wid) == elsewhere
    # ...and with nothing declared it is still the folder in the store.
    plain = new_workspace_id()
    store.write_config(plain, CONFIG)
    assert store.workspace_root(plain) == root / plain


def test_workspace_root_falls_back_for_an_unknown_workspace(tmp_path: Path) -> None:
    """Asked about a workspace it does not hold, the store still answers where one would
    go — `workspace create` needs that to derive a root from a freshly minted id."""
    store = _local(tmp_path / "workspaces")
    wid = new_workspace_id()
    assert store.workspace_root(wid) == tmp_path / "workspaces" / wid


def test_the_local_backend_names_its_config_file(tmp_path: Path) -> None:
    """It keeps configs as ordinary files, which is the whole point of a directory a
    user can look inside — so it says so rather than inheriting the base "not a file"
    default, and `config_location` is that path."""
    root = tmp_path / "workspaces"
    store = _local(root)
    wid = new_workspace_id()
    store.write_config(wid, CONFIG)
    expected = root / wid / "config.toml"
    assert store.config_file(wid) == expected
    assert store.config_location(wid) == str(expected)


def test_the_base_default_names_no_file(tmp_path: Path) -> None:
    """A backend with no filesystem of its own inherits "not a file", and
    `config_location` degrades to naming the store and the id — never a synthetic path
    someone might try to restore from backup."""

    class Nowhere(LocalDirWorkspacesStore):
        config_file = WorkspacesStore.config_file

    store = Nowhere(
        Nowhere.parse_config(
            WorkspacesConfig(
                provider=DEFAULT_WORKSPACES_PROVIDER,
                options={"root": str(tmp_path / "workspaces")},
            )
        )
    )
    wid = new_workspace_id()
    assert store.config_file(wid) is None
    assert store.config_location(wid) == f"{store.label()}/{wid}"
