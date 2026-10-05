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

"""``create_workspace`` as a library caller sees it — no CLI in front of it.

The CLI pre-validates ids with its own messages before ever reaching this
function, so the CLI suite cannot tell whether the *library* refuses a bad id.
These tests call it directly. The two ``*_clobber`` cases are the ones a
library-only caller would otherwise hit silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dgml_core import (
    ConflictError,
    CorruptMetadata,
    InvalidArgument,
    StorageConfigInvalid,
    StorageProviderUnresolvable,
    Workspace,
    WorkspacesUnavailable,
    WorkspacesWriteConflict,
    create_workspace,
    default_workspaces_store,
)
from dgml_core import workspace_config as wsconfig

SEED_SVCA = """\
[storage.svca.blobs]
provider = "dgml_core.storage_local:LocalStore"

[storage.svca.docs]
provider = "dgml_core.storage_local:LocalStore"
"""


# ------------------------------------------------------------------ happy paths


def test_listed_create_claims_id_binds_and_seals() -> None:
    result = create_workspace(workspace_id="acme", organization="Acme", name="Acme Docs")

    ws, ident = result.workspace, result.identity
    assert ws.workspaces_id == "acme"
    assert ident.workspace_id == "acme"
    assert ident.name == "Acme Docs"
    assert ident.organization == "Acme"
    assert ident.storage_service == "default"
    assert ident.storage_fingerprint and ident.storage_fingerprint.startswith("sha256:")
    assert ident.created_at
    assert ws.is_initialized()
    assert default_workspaces_store().list_ids() == ["acme"]
    assert result.organization_changed_from is None
    assert result.storage_service_changed_from is None


def test_listed_create_generates_an_id_when_none_given() -> None:
    result = create_workspace(organization="Acme")
    assert result.identity.workspace_id
    assert result.identity.workspace_id.startswith("ws_")
    assert default_workspaces_store().exists(result.identity.workspace_id)


def test_detached_create_at_a_path(tmp_path: Path) -> None:
    root = tmp_path / "det"
    result = create_workspace(Workspace(root=root), organization="Acme")

    assert result.workspace.root == root
    assert result.workspace.workspaces_id is None
    assert (root / "config.toml").is_file()
    assert result.identity.name == "det"  # falls back to the directory name
    # A detached workspace is not listed.
    assert default_workspaces_store().list_ids() == []


def test_detached_create_validates_under_the_id_it_will_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that namespaces by workspace id refuses a config without one, and create
    validates the binding before the config exists — so the check must run under the id
    about to be written, not under none."""
    import sys

    from dgml_core import LocalStore, StorageConfig

    seen: list[str | None] = []

    class IdRequiringStore(LocalStore):
        @classmethod
        def parse_config(cls, config: StorageConfig) -> StorageConfig:
            seen.append(config.workspace_id)
            if not config.workspace_id:
                raise StorageConfigInvalid("needs the workspace's id")
            return super().parse_config(config)

    monkeypatch.setattr(sys.modules[__name__], "IdRequiringStore", IdRequiringStore, raising=False)
    seed = f'[storage.svcb]\nprovider = "{__name__}:IdRequiringStore"\n'
    result = create_workspace(
        Workspace(root=tmp_path / "det"),
        organization="Acme",
        storage_service="svcb",
        seed_toml=seed,
    )
    assert seen and all(wid == result.identity.workspace_id for wid in seen)


def test_rerun_preserves_recorded_identity() -> None:
    first = create_workspace(workspace_id="acme", organization="Acme", name="Prod")
    # Re-run addressed, with nothing but the organization — the documented idempotent path.
    again = create_workspace(first.workspace, organization="Acme")

    assert again.identity.workspace_id == "acme"
    assert again.identity.name == "Prod"
    assert again.identity.organization == "Acme"
    assert again.identity.created_at == first.identity.created_at
    assert again.organization_changed_from is None


def test_rerun_reports_a_changed_organization_and_service() -> None:
    first = create_workspace(
        workspace_id="acme", organization="Acme", storage_service="svca", seed_toml=SEED_SVCA
    )
    again = create_workspace(first.workspace, organization="Other", storage_service="default")

    assert again.organization_changed_from == "Acme"
    assert again.storage_service_changed_from == "svca"
    assert again.identity.organization == "Other"
    assert again.identity.storage_service == "default"


# --------------------------------------------------- the id must never clobber


def test_existing_listed_id_is_a_conflict_not_an_overwrite() -> None:
    """The natural library re-run — ``create_workspace(None, workspace_id=<same>)`` —
    must refuse, not silently replace the workspace's config in the store.

    Without this guard the seeded ``[storage.svca]`` table below was wiped, the
    workspace rebound to local disk and renamed, and the result reported nothing."""
    create_workspace(
        workspace_id="my-ws",
        organization="Acme",
        name="Prod",
        storage_service="svca",
        seed_toml=SEED_SVCA,
    )

    with pytest.raises(ConflictError) as caught:
        create_workspace(workspace_id="my-ws", organization="Acme")

    assert caught.value.kind == "workspace"
    assert caught.value.existing_id == "my-ws"
    # The original is untouched.
    stored = default_workspaces_store().read_config("my-ws") or ""
    assert "[storage.svca" in stored
    ident = wsconfig.identity_from_text(stored)
    assert ident.name == "Prod"
    assert ident.storage_service == "svca"


def test_detached_rerun_with_a_different_id_is_refused(tmp_path: Path) -> None:
    """``create`` never re-identifies a workspace. Asking for a new id against a
    directory that records another must raise — not succeed and hand back the old id."""
    ws = Workspace(root=tmp_path / "det")
    create_workspace(ws, workspace_id="ws-aaaa", organization="Acme")

    with pytest.raises(InvalidArgument, match="'ws-bbbb' does not match 'ws-aaaa'"):
        create_workspace(ws, workspace_id="ws-bbbb", organization="Acme")

    assert wsconfig.read_identity(ws).workspace_id == "ws-aaaa"


def test_detached_id_that_shadows_a_listed_workspace_is_a_conflict(tmp_path: Path) -> None:
    """A detached workspace with no recorded id may not claim one the store already
    holds — it would shadow the listed workspace at ``--workspace <id>``."""
    create_workspace(workspace_id="taken", organization="Acme")

    with pytest.raises(ConflictError) as caught:
        create_workspace(Workspace(root=tmp_path / "det"), workspace_id="taken", organization="A")
    assert caught.value.existing_id == "taken"


# ------------------------------------------------ nothing written on rejection


def test_malformed_id_is_rejected_before_anything_is_written() -> None:
    with pytest.raises(InvalidArgument, match="not a well-formed workspace id"):
        create_workspace(workspace_id="Bad.Id", organization="Acme")
    assert default_workspaces_store().list_ids() == []


def test_seed_declaring_workspaces_table_is_rejected_before_anything_is_written() -> None:
    with pytest.raises(InvalidArgument, match=r"\[workspaces\] table"):
        create_workspace(
            workspace_id="acme", organization="Acme", seed_toml='[workspaces]\nprovider = "x"\n'
        )
    assert default_workspaces_store().list_ids() == []


def test_seed_that_is_not_toml_is_rejected_before_anything_is_written() -> None:
    with pytest.raises(InvalidArgument, match="not valid TOML"):
        create_workspace(workspace_id="acme", organization="Acme", seed_toml="[storage\n")
    assert default_workspaces_store().list_ids() == []


# ---------------------------------------------------------------- other guards


def test_organization_is_required_for_a_new_workspace() -> None:
    with pytest.raises(InvalidArgument, match="organization is required"):
        create_workspace(workspace_id="acme")
    assert default_workspaces_store().list_ids() == []


def test_seed_that_declares_only_other_services_is_refused() -> None:
    """A seed exists to name a backend; selecting a service it does not declare would
    silently fall through to the bundled local store."""
    with pytest.raises(InvalidArgument, match=r"declares no \[storage.default\]"):
        create_workspace(
            workspace_id="acme",
            organization="Acme",
            storage_service="default",
            seed_toml=SEED_SVCA,
        )
    assert default_workspaces_store().list_ids() == []


def test_seeded_unknown_service_is_answered_with_the_declared_ones() -> None:
    """A mistyped service must be answered with the services the seed *does* declare —
    resolving it first buried that under a bare 'no [storage.<service>] configured'."""
    with pytest.raises(InvalidArgument, match=r"does declare \[storage.svca\]"):
        create_workspace(
            workspace_id="acme",
            organization="Acme",
            storage_service="mystorage",
            seed_toml=SEED_SVCA,
        )
    assert default_workspaces_store().list_ids() == []


def test_unknown_storage_service_leaves_no_row() -> None:
    with pytest.raises(StorageConfigInvalid):
        create_workspace(workspace_id="bad-svc", organization="A", storage_service="nope")
    assert default_workspaces_store().list_ids() == []


def test_unresolvable_provider_leaves_no_row_and_retry_succeeds() -> None:
    """Resolved before anything is built, and the claimed row removed on failure — so the
    same id can be retried once the seed is fixed."""
    bad = SEED_SVCA.replace('"dgml_core.storage_local:LocalStore"', '"local"')
    with pytest.raises(StorageProviderUnresolvable):
        create_workspace(
            workspace_id="acme", organization="Acme", storage_service="svca", seed_toml=bad
        )
    assert default_workspaces_store().list_ids() == []

    result = create_workspace(
        workspace_id="acme", organization="Acme", storage_service="svca", seed_toml=SEED_SVCA
    )
    assert result.identity.workspace_id == "acme"


def test_seed_is_refused_against_a_different_existing_config(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    create_workspace(ws, organization="Acme")
    before = ws.config_text
    with pytest.raises(InvalidArgument, match="differs from the seed config"):
        create_workspace(ws, storage_service="svca", seed_toml=SEED_SVCA)
    assert Workspace(root=ws.root).config_text == before


def test_detached_seeded_create_that_fails_leaves_no_config(tmp_path: Path) -> None:
    """The seed this call wrote is removed on failure — so the documented retry with a
    fixed seed succeeds instead of being refused as 'differs from the seed'."""
    root = tmp_path / "ws"
    bad = SEED_SVCA.replace('"dgml_core.storage_local:LocalStore"', '"local"')
    with pytest.raises(StorageProviderUnresolvable):
        create_workspace(
            Workspace(root=root), organization="Acme", storage_service="svca", seed_toml=bad
        )
    assert not root.exists()

    result = create_workspace(
        Workspace(root=root), organization="Acme", storage_service="svca", seed_toml=SEED_SVCA
    )
    assert result.identity.storage_service == "svca"


def test_failed_seeded_create_keeps_a_preexisting_directory(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "notes.txt").write_text("keep me", encoding="utf-8")
    bad = SEED_SVCA.replace('"dgml_core.storage_local:LocalStore"', '"local"')
    with pytest.raises(StorageProviderUnresolvable):
        create_workspace(
            Workspace(root=root), organization="Acme", storage_service="svca", seed_toml=bad
        )
    assert not (root / "config.toml").exists()
    assert (root / "notes.txt").read_text(encoding="utf-8") == "keep me"


def test_seed_against_a_corrupt_config_reports_the_corruption(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "config.toml").write_text("[storage\nnot toml\n", encoding="utf-8")
    with pytest.raises(CorruptMetadata, match="invalid TOML"):
        create_workspace(Workspace(root=root), organization="Acme", seed_toml=SEED_SVCA)


def test_a_failing_rollback_does_not_mask_the_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(workspace_id: str) -> bool:
        raise OSError("disk went away")

    monkeypatch.setattr(default_workspaces_store(), "delete", boom)
    with pytest.raises(InvalidArgument, match="organization is required"):
        create_workspace(workspace_id="acme")


def test_addressed_rollback_holds_on_a_conflict_detecting_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resetting a row addressed by id must be conditional on the store's *current* text, not the
    stale memo of the Workspace the create started from. The build's last write goes
    through a fresh Workspace, so on a backend that enforces ``expected_text`` (Mongo,
    Postgres) a stale token makes the reset fail silently — stranding the seed, and with
    it the documented retry. The local store ignores the token, so this test teaches it
    to enforce it."""
    store = default_workspaces_store()
    real_write = store.write_config

    def conditional(workspace_id: str, text: str, *, expected_text: str | None = None) -> None:
        if expected_text is not None and store.read_config(workspace_id) != expected_text:
            raise WorkspacesWriteConflict("another writer changed it since it was read")
        real_write(workspace_id, text, expected_text=expected_text)

    monkeypatch.setattr(store, "write_config", conditional)
    store.create_config("acme", "")
    ws = Workspace(root=store.workspace_root("acme"), workspaces_id="acme")

    from dgml_core.migrations import stamp_schema_version

    calls = {"n": 0}

    def flaky_stamp(workspace: Workspace) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("the backend went away after the seal")
        stamp_schema_version(workspace)

    monkeypatch.setattr("dgml_core.workspace_create.stamp_schema_version", flaky_stamp)
    with pytest.raises(RuntimeError):
        create_workspace(ws, organization="Acme", storage_service="svca", seed_toml=SEED_SVCA)
    assert store.read_config("acme") == ""  # the seed this call wrote is gone

    retry = create_workspace(
        Workspace(root=store.workspace_root("acme"), workspaces_id="acme"),
        organization="Acme",
        storage_service="svca",
        seed_toml=SEED_SVCA,
    )
    assert retry.identity.workspace_id == "acme"
    assert retry.identity.storage_service == "svca"


def test_a_claim_that_fails_after_applying_is_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ack of a successful claim can be lost (connection drops after the server
    applied the insert). The failure must still remove the row, or every retry of the
    same id is refused as CONFLICT — the stranding this create promises away."""
    store = default_workspaces_store()
    real = store.create_config

    def applied_but_ack_lost(workspace_id: str, text: str) -> None:
        real(workspace_id, text)
        raise WorkspacesUnavailable("connection reset before the acknowledgement")

    monkeypatch.setattr(store, "create_config", applied_but_ack_lost)
    with pytest.raises(WorkspacesUnavailable):
        create_workspace(workspace_id="acme", organization="Acme")
    assert not store.exists("acme")

    monkeypatch.undo()
    assert create_workspace(workspace_id="acme", organization="Acme").identity.workspace_id


def test_a_claim_that_fails_before_applying_leaves_another_writers_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same error can mean the insert never applied. If a row already sits under
    the id, it is someone else's, and the lost-ack rollback must not delete it."""
    store = default_workspaces_store()
    theirs = create_workspace(workspace_id="acme", organization="Theirs")
    their_text = store.read_config("acme")

    def never_applied(workspace_id: str, text: str) -> None:
        raise WorkspacesUnavailable("timed out before the insert was sent")

    # `exists` is blinded so the pre-check does not refuse the id first: the claim
    # itself must be what fails, with their row already there.
    monkeypatch.setattr(store, "create_config", never_applied)
    monkeypatch.setattr(store, "exists", lambda workspace_id: False)
    with pytest.raises(WorkspacesUnavailable):
        create_workspace(workspace_id="acme", organization="Acme")

    assert store.read_config("acme") == their_text
    assert theirs.identity.workspace_id == "acme"


def test_a_conflicting_write_does_not_delete_the_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``WorkspacesWriteConflict`` during the build means another writer changed the
    claimed row — it is theirs now, so the rollback must leave it in the store."""
    store = default_workspaces_store()

    def taken(*args: object, **kwargs: object) -> None:
        raise WorkspacesWriteConflict("another writer changed it since it was read")

    monkeypatch.setattr(store, "write_config", taken)
    with pytest.raises(WorkspacesWriteConflict):
        create_workspace(workspace_id="acme", organization="Acme")
    assert store.exists("acme")


def test_losing_the_claim_race_is_a_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A claim that loses the create-if-absent race reads exactly like an id the store
    already held — the caller cannot tell them apart and should not have to."""
    store = default_workspaces_store()

    def lost(workspace_id: str, text: str) -> None:
        raise WorkspacesWriteConflict("already holds one")

    monkeypatch.setattr(store, "create_config", lost)
    with pytest.raises(ConflictError) as caught:
        create_workspace(workspace_id="acme", organization="Acme")
    assert caught.value.kind == "workspace"
    assert caught.value.existing_id == "acme"


def test_rerun_with_the_same_seed_is_a_no_op(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    first = create_workspace(ws, organization="Acme", storage_service="svca", seed_toml=SEED_SVCA)
    again = create_workspace(ws, storage_service="svca", seed_toml=SEED_SVCA)
    assert again.identity.workspace_id == first.identity.workspace_id


# --------------------------------------- the store is consulted only when needed


def _configure_unimportable_workspaces_store() -> None:
    """Point ``[workspaces]`` at a provider this machine cannot import — the shape of
    a fresh shell or cron job on a machine whose store lives in an out-of-tree module
    that ``PYTHONPATH`` does not currently reach."""
    from dgml_core.storage import user_config_path

    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[workspaces]\nprovider = "no_such_module:NoSuchStore"\n', encoding="utf-8")
    default_workspaces_store.cache_clear()


def test_rerun_with_a_seed_declaring_workspace_keys_is_a_no_op(tmp_path: Path) -> None:
    """``write_identity`` adds machine-managed keys (``workspace_id``, ``created_at``,
    ``storage_fingerprint``…) beside what the seed declared, so the no-op check must ask
    whether the keys the seed declares still stand — not whether the whole table stayed
    exactly as written, which after a successful create it never has."""
    seed = SEED_SVCA + '\n[workspace]\norganization = "Acme"\n'
    first = create_workspace(
        Workspace(root=tmp_path / "ws"), storage_service="svca", seed_toml=seed
    )
    assert first.identity.organization == "Acme"

    again = create_workspace(
        Workspace(root=tmp_path / "ws"), storage_service="svca", seed_toml=seed
    )
    assert again.identity.workspace_id == first.identity.workspace_id

    # A seed whose declared value genuinely differs is still refused.
    with pytest.raises(InvalidArgument, match="differs from the seed"):
        create_workspace(
            Workspace(root=tmp_path / "ws"),
            storage_service="svca",
            seed_toml=seed.replace('"Acme"', '"Beta"'),
        )


def test_detached_create_does_not_need_the_workspaces_store(tmp_path: Path) -> None:
    """A path-addressed workspace lists nowhere, so a broken ``[workspaces]`` table must
    not stop it being created. Regressed once when the store was built unconditionally."""
    _configure_unimportable_workspaces_store()

    result = create_workspace(Workspace(root=tmp_path / "det"), organization="Acme")

    assert result.workspace.is_initialized()
    assert result.workspace.workspaces_id is None


def test_listed_create_does_need_the_workspaces_store() -> None:
    """The counterpart: the laziness is precise. A listed create has nowhere to put the
    workspace but the store, so the same broken table must fail it."""
    from dgml_core import StorageProviderUnresolvable

    _configure_unimportable_workspaces_store()

    with pytest.raises(StorageProviderUnresolvable):
        create_workspace(organization="Acme")
