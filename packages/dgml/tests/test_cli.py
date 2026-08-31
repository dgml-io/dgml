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

import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core import layout
from dgml_core.migrations import (
    WORKSPACE_SCHEMA_VERSION,
    pending_migrations,
    stamp_schema_version,
    workspace_schema_version,
)
from dgml_core.run_clustering import DocPrediction
from dgml_core.storage import Workspace
from dgml_core.workspaces_resolve import default_workspaces_store

from .conftest import (
    _write_blank_pdf,
    _write_text_pdf,
    dump_toml,
    needs_gs,
    write_classification_config,
)


def _ws_args(ws: Path) -> list[str]:
    return ["--workspace", str(ws)]


def _write_ws_config(ws_root: Path, data: dict[str, Any]) -> None:
    """Write ``<workspace>/config.toml`` (resolution layer 3) from a dict."""
    Workspace(root=ws_root).config_path.write_text(dump_toml(data) + "\n", encoding="utf-8")


def _init_ws(ws: Path) -> None:
    """Bootstrap a usable workspace for tests the way `dgml workspace create`
    does — ``docsets/``, ``files/``, and the ``[workspace]`` identity block in
    ``config.toml`` — without emitting CLI stdout that would interleave with the
    output under test. Other config sections are written per-test when a command
    needs them (e.g. ``write_classification_config``).

    The identity block is not optional scaffolding: the CLI rejects an initialized
    workspace with no ``config.toml``, because that file names the storage backend and
    its absence is indistinguishable from a remote workspace whose config was deleted.
    """
    from dgml_core import workspace_config

    workspace = Workspace(root=ws.resolve())
    workspace.init()
    workspace_config.write_identity(
        workspace,
        workspace_id="ws_testxxxxxxxxxxxx",
        name=workspace.root.name,
        organization=workspace.root.name,
        storage_service="default",
    )


def _dp(cluster_name: str, confidence: float | None = None, review: bool = False) -> DocPrediction:
    """Shorthand for a mocked ``run_clustering_detailed`` outcome."""
    return DocPrediction(cluster_name=cluster_name, confidence=confidence, review=review)


def _read_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    out = capsys.readouterr().out
    return json.loads(out)  # type: ignore[no-any-return]


def _read_stderr(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    err = capsys.readouterr().err
    return json.loads(err)  # type: ignore[no-any-return]


def test_init_writes_user_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`dgml init` is config-only: it writes the user-level config.toml (with a
    [models] block) and does NOT create the workspace dirs. A second init
    without --force is a no-op."""
    from dgml_core.storage import user_config_path

    ws = tmp_path / "ws"
    rc = main(_ws_args(ws) + ["init"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["config_created"] is True
    assert payload["forced"] is False
    # Both dummy provider keys are set by the autouse fixture → auto-detect mixed.
    assert payload["provider"] == "mixed"
    assert set(payload["detected_keys"]) == {"ANTHROPIC_API_KEY", "GEMINI_API_KEY"}
    assert "next_action" in payload
    cfg = user_config_path()
    assert Path(payload["config_path"]) == cfg
    assert "[models]" in cfg.read_text(encoding="utf-8")
    # init is config-only: no workspace dirs.
    assert not (ws / "docsets").exists()
    assert not (ws / "config.toml").exists()

    # Second init without --force is a no-op.
    rc = main(_ws_args(ws) + ["init"])
    assert rc == 0
    assert _read_stdout(capsys)["config_created"] is False


def test_init_provider_flag_forces_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dgml_core.storage import user_config_path

    rc = main(_ws_args(tmp_path / "ws") + ["init", "--provider", "google"])
    assert rc == 0
    assert _read_stdout(capsys)["provider"] == "google"
    assert "gemini/" in user_config_path().read_text(encoding="utf-8")


def test_init_provider_without_force_does_not_clobber(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dgml_core.storage import user_config_path

    ws = tmp_path / "ws"
    main(_ws_args(ws) + ["init", "--provider", "anthropic"])
    capsys.readouterr()
    # Rerun with a different provider but no --force → no-op + a warning.
    rc = main(_ws_args(ws) + ["init", "--provider", "google"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["config_created"] is False
    assert "--force" in payload["next_action"]
    assert "anthropic/" in user_config_path().read_text(encoding="utf-8")  # unchanged


def test_init_force_overwrites_with_backup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from dgml_core.storage import user_config_path

    ws = tmp_path / "ws"
    main(_ws_args(ws) + ["init", "--provider", "anthropic"])
    capsys.readouterr()
    rc = main(_ws_args(ws) + ["init", "--provider", "google", "--force"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["forced"] is True
    cfg = user_config_path()
    assert "gemini/" in cfg.read_text(encoding="utf-8")
    assert cfg.with_suffix(".toml.bak").exists()


def test_workspace_create(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`workspace create` builds docsets/ + files/ + workspace.json **and** the
    workspace's own config.toml, which names its storage backend."""
    ws = tmp_path / "ws"
    main(_ws_args(ws) + ["init"])  # write the user config first
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["workspace", "create", "--organization", "Acme"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["initialized"] is True
    assert payload["organization"] == "Acme"
    assert payload["name"] == "ws"  # defaults to the workspace directory name
    assert payload["config_present"] is True  # user config existed (init ran)
    # A stable workspace_id is minted at create and echoed in the payload.
    workspace_id = payload["workspace_id"]
    assert workspace_id.startswith("ws_")
    # With no --storage it lands on the bundled default service.
    assert payload["storage_service"] == "default"
    assert (ws / "docsets").is_dir()
    assert (ws / "files").is_dir()
    # The workspace config is now written by create and is authoritative for storage.
    assert (ws / "config.toml").exists()
    assert payload["workspace_config_path"] == str(ws / "config.toml")
    assert payload["storage_fingerprint"].startswith("sha256:")
    meta = json.loads((ws / "workspace.json").read_text(encoding="utf-8"))
    # workspace.json also carries the layout revision the workspace was written
    # against, so an older one can be upgraded in place on first use, plus the
    # stable workspace_id that the machine index keys on.
    assert meta == {
        "name": "ws",
        "organization": "Acme",
        "workspace_id": workspace_id,
        "schema_version": WORKSPACE_SCHEMA_VERSION,
    }

    # An explicit --name overrides the directory-name default.
    rc = main(_ws_args(ws) + ["workspace", "create", "--organization", "Acme", "--name", "My WS"])
    assert rc == 0
    assert _read_stdout(capsys)["name"] == "My WS"


def test_workspace_create_positional_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`workspace create <path>` targets that directory without the redundant
    global --workspace, and a positional path overrides the global flag."""
    ws = tmp_path / "ws"
    rc = main(["workspace", "create", str(ws), "--organization", "Acme"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert Path(payload["workspace"]) == ws.resolve()
    assert (ws / "docsets").is_dir()

    # The positional wins over a (differing) global --workspace.
    other = tmp_path / "other"
    rc = main(
        [
            "--workspace",
            str(tmp_path / "ignored"),
            "workspace",
            "create",
            str(other),
            "--organization",
            "Acme",
        ]
    )
    assert rc == 0
    assert Path(_read_stdout(capsys)["workspace"]) == other.resolve()
    assert (other / "docsets").is_dir()
    assert not (tmp_path / "ignored" / "docsets").exists()


def test_create_without_a_path_is_listed_and_opens_by_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Naming no path puts the workspace in this machine's store of workspaces, and the
    minted id can then be used anywhere a path can."""
    rc = main(["workspace", "create", "--organization", "Acme", "--name", "A"])
    assert rc == 0
    created = _read_stdout(capsys)
    workspace_id = created["workspace_id"]
    assert created["listed"] is True
    # The default store keeps configs as ordinary files, so the payload names the file —
    # it is hand-editable, and reporting it as absent would deny the caller something
    # real. A backend that does *not* keep configs as files (Mongo) reports null here
    # instead of inventing a path someone might try to restore.
    store = default_workspaces_store()
    config_file = store.config_file(workspace_id)
    assert config_file is not None
    assert created["workspace_config_path"] == str(config_file)
    assert created["config_location"] == str(config_file)

    assert store.exists(workspace_id)
    assert Path(created["workspace"]) == store.workspace_root(workspace_id)

    rc = main(["--workspace", workspace_id, "status"])
    assert rc == 0
    assert Path(_read_stdout(capsys)["workspace"]) == store.workspace_root(workspace_id)


def test_create_with_a_path_is_detached_and_not_listed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Naming a path keeps today's behaviour exactly: the config is a file in that
    directory, and the workspace is addressed by path rather than listed."""
    ws = tmp_path / "ws"
    rc = main(["workspace", "create", str(ws), "--organization", "Acme"])
    assert rc == 0
    created = _read_stdout(capsys)
    assert created["listed"] is False
    assert created["workspace_config_path"] == str(ws.resolve() / "config.toml")
    assert not default_workspaces_store().exists(created["workspace_id"])

    # It opens by path, as before.
    assert main(_ws_args(ws) + ["status"]) == 0


def test_an_unknown_id_is_an_error_not_a_new_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The id test is on shape, so an id-shaped argument the store does not hold is a
    clear error. Under the old index-membership test it fell through to path
    resolution and created a `ws_…` directory in the working directory."""
    monkeypatch.chdir(tmp_path)
    rc = main(["--workspace", "ws_abcdefghijklmnop", "status"])
    assert rc != 0
    assert _read_stderr(capsys)["error"]["code"] == "WORKSPACE_NOT_FOUND"
    assert not (tmp_path / "ws_abcdefghijklmnop").exists()


def test_workspace_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`workspace list` reports every workspace the store holds, without opening any of
    their storage backends."""
    main(["workspace", "create", "--organization", "Acme", "--name", "A"])
    id_a = _read_stdout(capsys)["workspace_id"]
    main(["workspace", "create", "--organization", "Beta", "--name", "B"])
    id_b = _read_stdout(capsys)["workspace_id"]

    rc = main(["workspace", "list"])
    assert rc == 0
    payload = _read_stdout(capsys)
    by_id = {r["workspace_id"]: r for r in payload["workspaces"]}
    assert set(by_id) == {id_a, id_b}
    assert by_id[id_a]["name"] == "A"
    assert by_id[id_b]["organization"] == "Beta"
    assert by_id[id_a]["storage_service"] == "default"
    assert by_id[id_a]["created_at"]
    assert by_id[id_a]["root"] == str(default_workspaces_store().workspace_root(id_a))
    assert payload["workspaces_store"]


def test_workspace_list_does_not_show_a_workspace_addressed_by_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A detached workspace is not in the store, so it is not listed. That is the
    honest answer — the store is where being listed is recorded, and there is no
    per-machine index scanning directories any more."""
    main(["workspace", "create", str(tmp_path / "ws"), "--organization", "Acme"])
    detached_id = _read_stdout(capsys)["workspace_id"]
    main(["workspace", "list"])
    rows = _read_stdout(capsys)["workspaces"]
    assert detached_id not in {r["workspace_id"] for r in rows}


def test_open_backfills_a_missing_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A legacy workspace with no workspace_id gets one minted into workspace.json and
    mirrored into its config on first open — no manual step, idempotent on a second."""
    from dgml_core import workspace_config

    # Simulate a pre-id workspace: initialized, meta without a workspace_id, stamped at
    # an older schema version so the backfill migration runs. It has a config.toml
    # (every workspace does) but no identity block yet.
    ws = Workspace(root=tmp_path / "legacy")
    ws.init()
    ws.config_path.write_text('[storage]\nprovider = "dgml_core.storage_local:LocalStore"\n')
    ws.write_meta(name="legacy", organization="Acme")
    stamp_schema_version(ws, 0)
    assert ws.workspace_id is None

    assert main(_ws_args(ws.root) + ["status"]) == 0

    wid = ws.workspace_id
    assert wid is not None and wid.startswith("ws_")

    # The id is mirrored into config.toml so it is readable without the store. Read it
    # through a fresh Workspace: config text is memoized per object, and this one was
    # constructed before the command that wrote it (see Workspace._config_state).
    assert workspace_config.read_identity(Workspace(root=ws.root)).workspace_id == wid

    # Second open: id unchanged.
    assert main(_ws_args(ws.root) + ["status"]) == 0
    assert ws.workspace_id == wid


def test_a_cloned_directory_keeps_its_id_and_opens_by_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A workspace directory carrying a workspace_id (as if cloned from another
    machine) opens by path and keeps that id. It is not silently added to this
    machine's store: the id travels with the directory, and what a machine *lists* is
    now a deliberate choice rather than a side effect of opening something."""
    ws = Workspace(root=tmp_path / "cloned")
    ws.init()
    ws.config_path.write_text('[storage]\nprovider = "dgml_core.storage_local:LocalStore"\n')
    wid = "ws_clonedaaaaaaaaaa"
    ws.write_meta(name="Cloned", organization="Acme", workspace_id=wid)
    stamp_schema_version(ws)

    assert main(_ws_args(ws.root) + ["status"]) == 0
    assert ws.workspace_id == wid
    assert not default_workspaces_store().exists(wid)


def test_a_copied_workspace_directory_still_opens(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Copying a workspace directory elsewhere and opening it there works, with no row
    anywhere needing to be corrected. Nothing records where a detached workspace is, so
    there is nothing to go stale — which is why `workspace register` is gone."""
    src = tmp_path / "src"
    main(["workspace", "create", str(src), "--organization", "Acme"])
    wid = _read_stdout(capsys)["workspace_id"]

    # The whole directory travels, config.toml included — that file carries the storage
    # binding, so a copy without it is not a workspace.
    dest = tmp_path / "dest"
    shutil.copytree(src, dest)

    assert main(_ws_args(dest) + ["status"]) == 0
    assert Workspace(root=dest).workspace_id == wid


# A named storage service pointing at the bundled local store — a real, working
# backend exercised through the named-service path (no fake provider needed).
_LOCAL = "dgml_core.storage_local:LocalStore"


def _repoint_storage(ws_root: Path, service: str, provider: str) -> None:
    """Edit a workspace's own ``[storage.<service>]`` to name a different provider —
    the drift the seal exists to catch."""
    from dgml_core import workspace_config

    workspace_config.write_storage_table(
        Workspace(root=ws_root), service, {"blobs": {"provider": provider}}
    )


def test_workspace_create_on_named_storage_service(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`create --storage <name>` materializes that named service into the workspace's
    own config.toml, and the workspace opens through it."""
    from dgml_core import workspace_config
    from dgml_core.storage_resolve import resolve_store_configs

    ws = tmp_path / "ws"
    ws.mkdir()
    _write_ws_config(ws, {"storage": {"svcA": {"provider": _LOCAL}}})
    rc = main(["workspace", "create", str(ws), "--organization", "Acme", "--storage", "svcA"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["storage_service"] == "svcA"

    workspace = Workspace(root=ws)
    identity = workspace_config.read_identity(workspace)
    assert identity.storage_service == "svcA"
    assert identity.storage_fingerprint == payload["storage_fingerprint"]
    # This workspace's config already declared [storage.svcA], so create leaves it
    # exactly as authored — `workspace create` is documented as safe to re-run, which
    # means it must never rewrite a binding the user wrote themselves.
    assert workspace_config.read_storage_table(workspace, "svcA") == {"provider": _LOCAL}
    # The flat form serves both roles, so both resolve to the same backend.
    blob_cfg, doc_cfg = resolve_store_configs(workspace)
    assert blob_cfg.provider == doc_cfg.provider == _LOCAL

    # Opens through the sealed service.
    rc = main(_ws_args(ws) + ["status"])
    assert rc == 0
    assert Path(_read_stdout(capsys)["workspace"]) == ws.resolve()


def test_workspace_create_unconfigured_storage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--storage <name>` for a service that isn't in config fails cleanly, before
    anything is created."""
    ws = tmp_path / "ws"
    rc = main(["workspace", "create", str(ws), "--organization", "Acme", "--storage", "nope"])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "STORAGE_CONFIG_INVALID"
    assert not (ws / "docsets").exists()  # nothing built


def test_deleting_the_workspace_config_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The config now records where the data lives, so its absence must be loud.

    This inverts the old contract, deliberately. Falling back to the local default
    would let a remote-backed workspace open empty and report zero files — the config
    is the only record of its backend, and nothing else can reconstruct it."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_ws_config(ws, {"storage": {"svcA": {"provider": _LOCAL}}})
    main(["workspace", "create", str(ws), "--organization", "Acme", "--storage", "svcA"])
    capsys.readouterr()

    Workspace(root=ws).config_path.unlink()
    assert main(_ws_args(ws) + ["status"]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "STORAGE_CONFIG_INVALID"


def test_config_storage_edit_trips_the_seal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """**This inverts the old behaviour.** Previously a config edit could never trip
    the seal (the registry snapshot was authoritative and pinned); now the config *is*
    the binding, so editing [storage] is exactly what the guard watches for."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_ws_config(ws, {"storage": {"svcA": {"provider": _LOCAL}}})
    main(["workspace", "create", str(ws), "--organization", "Acme", "--storage", "svcA"])
    capsys.readouterr()

    _repoint_storage(ws, "svcA", "some.other:Store")
    assert main(_ws_args(ws) + ["status"]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "STORAGE_BACKEND_MISMATCH"

    # `workspace reseal` accepts the change (it is exempt from the guard).
    assert main(["workspace", "reseal", str(ws)]) == 0
    capsys.readouterr()
    assert main(_ws_args(ws) + ["status"]) == 0


def test_workspace_reseal_reports_both_fingerprints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _write_ws_config(ws, {"storage": {"svcA": {"provider": _LOCAL}}})
    main(["workspace", "create", str(ws), "--organization", "Acme", "--storage", "svcA"])
    before = _read_stdout(capsys)["storage_fingerprint"]

    _repoint_storage(ws, "svcA", "some.other:Store")
    assert main(["workspace", "reseal", str(ws)]) == 0
    payload = _read_stdout(capsys)
    assert payload["resealed"] is True
    assert payload["previous_fingerprint"] == before
    assert payload["storage_fingerprint"] != before
    assert payload["storage"]["blobs"]["provider"] == "some.other:Store"
    assert payload["config_location"] == str(Workspace(root=ws).config_path)


def test_workspace_reseal_check_reports_drift_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--check` is for CI and agents: it answers the question without consenting."""
    from dgml_core import workspace_config

    ws = tmp_path / "ws"
    ws.mkdir()
    _write_ws_config(ws, {"storage": {"svcA": {"provider": _LOCAL}}})
    main(["workspace", "create", str(ws), "--organization", "Acme", "--storage", "svcA"])
    before = _read_stdout(capsys)["storage_fingerprint"]

    _repoint_storage(ws, "svcA", "some.other:Store")
    assert main(["workspace", "reseal", str(ws), "--check"]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "STORAGE_BACKEND_MISMATCH"
    assert workspace_config.read_identity(Workspace(root=ws)).storage_fingerprint == before


def test_workspace_reseal_check_passes_when_sealed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    capsys.readouterr()
    assert main(["workspace", "reseal", str(ws), "--check"]) == 0
    assert _read_stdout(capsys)["resealed"] is False


def test_workspace_reseal_requires_an_initialized_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["workspace", "reseal", str(tmp_path / "nope")]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "WORKSPACE_NOT_INITIALIZED"


def test_workspace_register_is_gone(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Removed, but kept declared for one release so an existing caller gets a JSON
    envelope naming the replacement rather than an argparse usage dump on stderr."""
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    capsys.readouterr()

    assert main(["workspace", "register", str(ws)]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "INVALID_ARGUMENT"

    assert main(["workspace", "register", str(ws), "--storage", "svcA"]) == 1
    assert "reseal" in _read_stderr(capsys)["error"]["message"]


def test_workspace_list_rows_are_derived_from_each_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every field in a listing row is parsed out of that workspace's own config, so a
    row can never disagree with the workspace it describes. That is the difference from
    the old per-machine index, whose columns were a second copy nobody should trust —
    `storage_service` can be reported here precisely *because* it is not stored here."""
    seed = tmp_path / "seed.toml"
    seed.write_text(f'[storage.svcA]\nprovider = "{_LOCAL}"\n', encoding="utf-8")
    main(
        [
            "workspace",
            "create",
            "--organization",
            "Acme",
            "--storage",
            "svcA",
            "--from-config",
            str(seed),
        ]
    )
    wid = _read_stdout(capsys)["workspace_id"]

    rc = main(["workspace", "list"])
    assert rc == 0
    rows = {r["workspace_id"]: r for r in _read_stdout(capsys)["workspaces"]}
    assert set(rows[wid]) == {
        "workspace_id",
        "name",
        "organization",
        "storage_service",
        "root",
        "created_at",
    }
    assert rows[wid]["storage_service"] == "svcA"

    # Prove it is derived rather than stored: edit the config the store holds, and the
    # listing follows without anything re-indexing it.
    store = default_workspaces_store()
    found = store.read_config(wid)
    assert found is not None
    store.write_config(wid, found[0].replace('name = "', 'name = "Renamed '))
    main(["workspace", "list"])
    rows = {r["workspace_id"]: r for r in _read_stdout(capsys)["workspaces"]}
    assert rows[wid]["name"].startswith("Renamed ")


def test_workspace_create_requires_organization_for_a_new_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Still required with nothing to inherit it from — but now as a JSON envelope
    rather than an argparse usage dump, since this CLI is driven by agents too."""
    ws = tmp_path / "ws"
    assert main(_ws_args(ws) + ["workspace", "create"]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "INVALID_ARGUMENT"
    assert not (ws / "docsets").exists()  # nothing built


def test_workspace_create_inherits_organization_from_the_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Optional once the config records one. Adopting an existing workspace must not
    make you retype the value that defines its namespace URIs — retyping is exactly how
    a typo would re-organize the whole org's workspace."""
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    capsys.readouterr()

    assert main(["workspace", "create", str(ws)]) == 0
    assert _read_stdout(capsys)["organization"] == "Acme"


def test_workspace_create_warns_when_organization_differs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An explicit flag still wins, but loudly: it re-organizes the workspace for every
    consumer, and only affects *newly* generated XML — so the corpus would otherwise end
    up split across two namespaces with nothing to flag it later."""
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    capsys.readouterr()

    assert main(["workspace", "create", str(ws), "--organization", "Beta"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["organization"] == "Beta"
    assert "Warning:" in captured.err and "Acme" in captured.err


def test_workspace_create_without_prior_init_warns_but_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`workspace create` with no prior `dgml init` still creates the workspace
    (never blocks) and does NOT create the user-level config — it warns on
    stderr that credentials must be configured."""
    from dgml_core.storage import user_config_path

    ws = tmp_path / "ws"
    rc = main(_ws_args(ws) + ["workspace", "create", "--organization", "Acme"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["config_present"] is False
    assert "next_action" in payload
    # Workspace was created regardless.
    assert (ws / "docsets").is_dir()
    assert (ws / "files").is_dir()
    # The user config was NOT created by workspace create.
    assert not user_config_path().exists()
    # Warning always on stderr (no --verbose needed).
    assert "no user-level config found" in captured.err
    assert "dgml init" in captured.err


def test_status_after_workspace_create(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    main(_ws_args(ws) + ["workspace", "create", "--organization", "Acme", "--name", "My WS"])
    capsys.readouterr()
    rc = main(_ws_args(ws) + ["status"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["name"] == "My WS"
    assert payload["organization"] == "Acme"
    assert payload["docset_count"] == 0
    assert payload["file_count"] == 0


def test_global_flags_accepted_after_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--workspace`/`--format`/`--verbose` parse both before and after the
    subcommand (shared parent parser). Regression guard for the argparse
    "global flags must precede the subcommand" gotcha."""
    ws = tmp_path / "ws"
    # --workspace before, the rest after the subcommand.
    rc = main(["--workspace", str(ws), "init"])
    assert rc == 0
    capsys.readouterr()
    _init_ws(ws)  # `init` is config-only now; create the workspace status needs.

    # All three flags trailing the subcommand.
    rc = main(["status", "--workspace", str(ws), "--format", "json"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["file_count"] == 0

    # --format text after the subcommand switches the renderer.
    rc = main(["status", "--workspace", str(ws), "--format", "text"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "file_count: 0" in out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """`--version` prints `dgml <version>` and exits 0 (argparse version action)."""
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("dgml ")
    assert out.split()[1]  # a non-empty version token


def test_uninitialized_workspace_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "fresh"
    rc = main(_ws_args(ws) + ["status"])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "WORKSPACE_NOT_INITIALIZED"


def test_docset_create_show_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["docset", "create", "--name", "Contracts", "--description", "d"])
    assert rc == 0
    created = _read_stdout(capsys)
    assert created["name"] == "Contracts"
    assert created["description"] == "d"
    assert created["key_questions"] == []  # default when --key-question not given
    docset_id = created["id"]

    rc = main(_ws_args(ws) + ["docset", "show", docset_id])
    assert rc == 0
    shown = _read_stdout(capsys)
    assert shown == created

    rc = main(_ws_args(ws) + ["docset", "list"])
    assert rc == 0
    listed = _read_stdout(capsys)
    assert any(d["id"] == docset_id for d in listed["docsets"])


def test_docset_update_and_delete(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["docset", "create", "--name", "X"])
    created = _read_stdout(capsys)
    did = created["id"]

    rc = main(_ws_args(ws) + ["docset", "update", did, "--name", "Y"])
    assert rc == 0
    updated = _read_stdout(capsys)
    assert updated["name"] == "Y"

    rc = main(_ws_args(ws) + ["docset", "delete", did])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["deleted"] == did

    rc = main(_ws_args(ws) + ["docset", "show", did])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "DOCSET_NOT_FOUND"


def test_docset_delete_rejects_empty_id_preserves_other_docsets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: `dgml docset delete ""` must surface a structured error,
    not silently `shutil.rmtree` the entire docsets directory.
    """
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["docset", "create", "--name", "Keep"])
    keep = _read_stdout(capsys)

    rc = main(_ws_args(ws) + ["docset", "delete", ""])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "INVALID_ARGUMENT"

    rc = main(_ws_args(ws) + ["docset", "list"])
    assert rc == 0
    listed = _read_stdout(capsys)
    assert any(d["id"] == keep["id"] for d in listed["docsets"])


def test_docset_add_file_rejects_empty_file_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["docset", "create", "--name", "X"])
    created = _read_stdout(capsys)
    did = created["id"]

    rc = main(_ws_args(ws) + ["docset", "add-file", "", "--docset", did])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "INVALID_ARGUMENT"

    rc = main(_ws_args(ws) + ["docset", "list-files", did])
    assert rc == 0
    listed = _read_stdout(capsys)
    assert listed["file_ids"] == []


@needs_gs
def test_docset_add_file_and_remove_file_roundtrip(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plain add-file/remove-file happy paths: the assignment payloads and the
    list-files membership before/after. (No auto-extract — that surface is
    parked.)"""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["docset", "create", "--name", "X"])
    did = _read_stdout(capsys)["id"]
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]

    rc = main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    assert rc == 0
    assert _read_stdout(capsys) == {"docset_id": did, "file_id": fid, "assigned": True}

    rc = main(_ws_args(ws) + ["docset", "list-files", did])
    assert rc == 0
    assert _read_stdout(capsys)["file_ids"] == [fid]

    rc = main(_ws_args(ws) + ["docset", "remove-file", fid, "--docset", did])
    assert rc == 0
    assert _read_stdout(capsys) == {"docset_id": did, "file_id": fid, "assigned": False}

    rc = main(_ws_args(ws) + ["docset", "list-files", did])
    assert rc == 0
    assert _read_stdout(capsys)["file_ids"] == []


def test_docset_add_file_rejects_nonexistent_file_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["docset", "create", "--name", "X"])
    created = _read_stdout(capsys)
    did = created["id"]

    rc = main(_ws_args(ws) + ["docset", "add-file", "doesnotexist1", "--docset", did])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "FILE_NOT_FOUND"

    rc = main(_ws_args(ws) + ["docset", "list-files", did])
    assert rc == 0
    listed = _read_stdout(capsys)
    assert listed["file_ids"] == []


@needs_gs
def test_file_add_show_delete(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["created"] is True
    # Payload shape is part of the public CLI contract — lock the new fields.
    assert "text_extraction_error" in payload
    assert "text_extraction" in payload
    assert payload["file"]["text_mode"] == "digital"
    # Renderer provenance is recorded for a PDF source; no converter was used.
    assert payload["file"]["page_image_dpi"] == 300
    assert payload["file"]["page_image_renderer"] == "ghostscript"
    assert payload["file"]["pdf_converter"] is None
    fid = payload["file"]["id"]

    rc = main(_ws_args(ws) + ["file", "show", fid])
    assert rc == 0
    shown = _read_stdout(capsys)
    assert shown["id"] == fid
    assert shown["text_mode"] == "digital"
    assert shown["page_image_dpi"] == 300
    assert shown["page_image_renderer"] == "ghostscript"
    assert shown["pdf_converter"] is None

    rc = main(_ws_args(ws) + ["file", "delete", fid])
    assert rc == 0


def test_file_add_text_mode_default_is_digital(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["file"]["text_mode"] == "digital"
    assert payload["text_extraction_error"] is None
    summary = payload["text_extraction"]
    assert summary["mode"] == "digital"
    assert summary["pages_written"] == 2
    assert summary["pages_with_words"] == 2
    assert summary["total_words"] >= 4


def test_file_add_dpi_flag_is_recorded_and_used(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["file", "add", str(text_pdf), "--dpi", "150"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["file"]["page_image_dpi"] == 150

    # page_text/ boxes are in the render's pixel space, so the flag has to reach
    # digital extraction too — not just the rasterizer.
    text_dir = ws / "files" / payload["file"]["id"] / "page_text"
    page = json.loads((text_dir / "page_1.json").read_text())
    assert page["width"] == round(612 * 150 / 72)


def test_file_add_rejects_nonpositive_dpi(tmp_path: Path, text_pdf: Path) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    # An argparse usage error (exit 2), raised before the workspace is touched.
    for bad in ("0", "-150", "notanumber"):
        with pytest.raises(SystemExit) as exc:
            main(_ws_args(ws) + ["file", "add", str(text_pdf), "--dpi", bad])
        assert exc.value.code == 2
    assert not list((ws / "files").iterdir())


@needs_gs
def test_file_add_conflict_errors_by_default(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "CONFLICT"


@needs_gs
def test_check_returns_two_when_issues(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    add_payload = _read_stdout(capsys)
    fid = add_payload["file"]["id"]

    pdf = (tmp_path / "ws" / "files" / fid).glob("*.pdf").__next__()
    pdf.unlink()

    rc = main(_ws_args(ws) + ["check"])
    assert rc == 2
    report = _read_stdout(capsys)
    assert report["issue_count"] >= 1


@needs_gs
def test_cluster_assigns_unassigned_files_to_docsets(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )

    # Empty workspace: no unassigned files → no-op, no LLM or clusterer call.
    with (
        patch("litellm.completion") as mock_completion,
        patch("dgml_core.clustering.run_clustering_detailed") as mock_cluster,
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "embedding"])
        assert rc == 0
        payload = _read_stdout(capsys)
        assert payload["clusters"] == {}
        assert payload["failed_file_ids"] == []
        assert payload["skipped"] is False
        assert payload["mode"] == "fresh"
        # No unassigned files, so no engine ran.
        assert payload["method"] is None
        assert payload["n_new_clusters"] == 0
        mock_completion.assert_not_called()
        mock_cluster.assert_not_called()

    # One unassigned file, no existing docsets — mock run_clustering to
    # put it in "unknown_0"; clustering() asks the LLM for a name +
    # description and creates a fresh DocSet with that name.
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]

    response = _tool_response(
        "create_new_docset",
        {
            "name": "Sample Documents",
            "description": "test docs",
            "key_questions": ["What is this document about?"],
        },
    )
    with (
        patch("litellm.completion", return_value=response),
        patch(
            "dgml_core.clustering.run_clustering_detailed",
            return_value={fid: _dp("unknown_0")},
        ),
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "embedding"])
    assert rc == 0
    payload = _read_stdout(capsys)
    # Placeholder "unknown_0" from the clusterer is rewritten to the
    # actual DocSet name the file landed in (the LLM-proposed one).
    assert payload["clusters"] == {fid: "Sample Documents"}
    assert payload["failed_file_ids"] == []
    assert payload["mode"] == "fresh"
    assert payload["method"] == "embedding"
    assert payload["n_new_clusters"] == 1
    assert payload["assignments"][fid] == {
        "docset": "Sample Documents",
        "confidence": None,
        # A single naming attempt (the default) is not an agreement measurement.
        "naming_confidence": None,
        "is_new": True,
        "review": False,
    }
    assert payload["review_queue"] == []

    # The new DocSet has the LLM-proposed name and description, and the
    # file is assigned to it.
    main(_ws_args(ws) + ["docset", "list"])
    ds_list = _read_stdout(capsys)
    assert len(ds_list["docsets"]) == 1
    new_ds = ds_list["docsets"][0]
    assert new_ds["name"] == "Sample Documents"
    assert new_ds["description"] == "test docs"
    main(_ws_args(ws) + ["docset", "list-files", new_ds["id"]])
    assert _read_stdout(capsys)["file_ids"] == [fid]

    # Second run is a no-op — file is already assigned, no LLM or clusterer call.
    # A DocSet now exists, so the resolved mode is incremental.
    with (
        patch("litellm.completion") as mock_completion,
        patch("dgml_core.clustering.run_clustering_detailed") as mock_cluster,
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "embedding"])
        assert rc == 0
        payload = _read_stdout(capsys)
        assert payload["clusters"] == {}
        assert payload["failed_file_ids"] == []
        assert payload["skipped"] is False
        assert payload["mode"] == "incremental"
        mock_completion.assert_not_called()
        mock_cluster.assert_not_called()


@needs_gs
def test_cluster_defaults_to_auto_and_routes_a_small_workspace_to_the_llm(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plain `dgml cluster` on a couple of files must not reach the embedding path.

    Reported from the field: two files, and the run died inside sklearn with
    "max_df corresponds to < documents than min_df" — the tf-idf fit has no
    document-frequency signal to work with at that size. The routing that
    avoids it existed already; it just was not the default.
    """
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
    )
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]

    response = _tool_response(
        "group_documents",
        {
            "groups": [
                {
                    "name": "Sample Documents",
                    "description": "test docs",
                    "key_questions": ["What is this?"],
                    "members": ["doc_1"],
                }
            ]
        },
    )
    with (
        patch("litellm.completion", return_value=response),
        patch("dgml_core.clustering.run_clustering_detailed") as mock_cluster,
    ):
        rc = main(_ws_args(ws) + ["cluster"])

    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["method"] == "llm"
    # The statistical pipeline — the one that used to raise here — never ran.
    mock_cluster.assert_not_called()
    assert payload["clusters"] == {fid: "Sample Documents"}


@needs_gs
def test_cluster_reports_confidence_for_a_new_docset(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file placed in a DocSet created this run still carries the clusterer's
    confidence. Emergent clusters used to be hardcoded to `null` here; the
    fresh-clustering path scores documents against the cluster centroids, so
    that score has to survive into `assignments`."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
    )
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]

    response = _tool_response(
        "create_new_docset",
        {"name": "Invoices", "description": "billing docs", "key_questions": ["Total?"]},
    )
    with (
        patch("litellm.completion", return_value=response),
        patch(
            "dgml_core.clustering.run_clustering_detailed",
            return_value={fid: _dp("unknown_0", 0.42)},
        ),
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "embedding"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["assignments"][fid] == {
        "docset": "Invoices",
        "confidence": 0.42,
        # Grouping confidence and naming agreement are independent: the cluster
        # scored 0.42 against its own centroid, while a single naming attempt
        # (the default) is not an agreement measurement at all.
        "naming_confidence": None,
        "is_new": True,
        "review": False,
    }
    # Nothing was asked to be reviewed, so the queue is present but empty —
    # callers can read the key unconditionally.
    assert payload["review_queue"] == []


@needs_gs
def test_cluster_flags_a_low_confidence_assignment_for_review(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A flagged assignment still lands in its DocSet — `review` is advisory, not
    a veto — and the file id also shows up in the top-level `review_queue` so a
    caller doesn't have to scan every assignment to find the ones to confirm."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
    )
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]

    response = _tool_response(
        "create_new_docset",
        {"name": "Invoices", "description": "billing docs", "key_questions": ["Total?"]},
    )
    with (
        patch("litellm.completion", return_value=response),
        patch(
            "dgml_core.clustering.run_clustering_detailed",
            return_value={fid: _dp("unknown_0", 0.11, review=True)},
        ),
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "embedding"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["assignments"][fid]["review"] is True
    assert payload["assignments"][fid]["docset"] == "Invoices"
    assert payload["review_queue"] == [fid]


@needs_gs
def test_cluster_skip_existing_is_noop_when_all_assigned(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`cluster --skip-existing` short-circuits (no clusterer call) when every
    file is already assigned, emitting `skipped: true`. A normal run reports
    `skipped: false` so the field is always present."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["docset", "create", "--name", "X"])
    did = _read_stdout(capsys)["id"]
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    with patch("dgml_core.clustering.run_clustering_detailed") as mock_cluster:
        rc = main(_ws_args(ws) + ["cluster", "--skip-existing"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["clusters"] == {}
    assert payload["failed_file_ids"] == []
    assert payload["skipped"] is True
    # A DocSet exists, so the skip-existing no-op still reports incremental.
    assert payload["mode"] == "incremental"
    mock_cluster.assert_not_called()


@needs_gs
def test_cluster_config_flag_passes_overrides_to_run_clustering(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`cluster --config PATH` loads a standalone JSON and threads its contents
    to run_clustering as overrides (replacing the workspace clustering
    section)."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]

    cfg = tmp_path / "clustering_light.json"
    cfg.write_text(json.dumps({"scenario": {"leiden_k_neighbors": 7}}), encoding="utf-8")

    response = _tool_response(
        "create_new_docset",
        {"name": "Sample Documents", "description": "d", "key_questions": ["q?"]},
    )
    with (
        patch("litellm.completion", return_value=response),
        patch(
            "dgml_core.clustering.run_clustering_detailed",
            return_value={fid: _dp("unknown_0")},
        ) as mock_cluster,
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "embedding", "--config", str(cfg)])
    assert rc == 0
    # The file's overrides reached the clusterer. corpus_dir is injected
    # alongside, but our custom scenario value survives the deep merge.
    overrides = mock_cluster.call_args.kwargs["overrides"]
    assert overrides["scenario"]["leiden_k_neighbors"] == 7


@needs_gs
def test_cluster_config_flag_missing_file_errors(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A `--config` path that doesn't exist surfaces CLUSTERING_CONFIG_INVALID
    rather than silently falling back to defaults."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    capsys.readouterr()

    # No --method: the default must reject a bad --config too. It used to route
    # small corpora to the LLM before the config was ever resolved, so a mistyped
    # path succeeded on the default path and errored only under `--method embedding`.
    rc = main(_ws_args(ws) + ["cluster", "--config", str(tmp_path / "nope.json")])
    assert rc != 0
    assert _read_stderr(capsys)["error"]["code"] == "CLUSTERING_CONFIG_INVALID"


def test_cluster_incremental_without_docsets_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`cluster --mode incremental` on a workspace with no DocSets surfaces a
    clear INCREMENTAL_WITHOUT_CLUSTERS error rather than silently running fresh."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["cluster", "--mode", "incremental"])
    assert rc != 0
    assert _read_stderr(capsys)["error"]["code"] == "INCREMENTAL_WITHOUT_CLUSTERS"


@needs_gs
def test_cluster_config_preset_name_passes_preset_overrides(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`cluster --config medium` resolves the bundled preset by name and threads
    its overrides to the clusterer (rather than treating it as a path)."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]

    response = _tool_response(
        "create_new_docset",
        {"name": "Sample Documents", "description": "d", "key_questions": ["q?"]},
    )
    with (
        patch("litellm.completion", return_value=response),
        patch(
            "dgml_core.clustering.run_clustering_detailed",
            return_value={fid: _dp("unknown_0")},
        ) as mock_cluster,
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "embedding", "--config", "medium"])
    assert rc == 0
    overrides = mock_cluster.call_args.kwargs["overrides"]
    # The medium preset fuses a dense image encoder into the text signal —
    # distinct from the default light preset (image "dummy" / fusion "none").
    assert overrides["encoder_image"]["name"] == "qwen3_vl_embedding_2b"
    assert overrides["fusion"]["name"] == "concat_norm"


def _llm_partition(fid: str) -> Any:
    """A one-cluster LLMClusteringResult that names its emergent bucket in the
    same call (so clustering() needs no second naming round-trip)."""
    from dgml_core.classification import ClassificationDecision
    from dgml_core.llm_clustering import LLMClusteringResult

    return LLMClusteringResult(
        clusters={fid: "unknown_0"},
        proposals={
            "unknown_0": ClassificationDecision(
                decision="new",
                new_name="Sample Documents",
                new_description="test docs",
                new_key_questions=("What is this document about?",),
            )
        },
        failed_file_ids=[],
    )


@needs_gs
def test_cluster_method_llm_routes_to_llm_partitioner(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`cluster --method llm` sends the corpus to the vision-LLM partitioner
    (never the embedding pipeline) and creates DocSets from the proposals it
    returns in a single call."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]

    with (
        patch(
            "dgml_core.clustering.llm_cluster_files", return_value=_llm_partition(fid)
        ) as mock_llm,
        patch("dgml_core.clustering.run_clustering_detailed") as mock_embed,
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "llm"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["clusters"] == {fid: "Sample Documents"}
    assert payload["failed_file_ids"] == []
    assert payload["n_new_clusters"] == 1
    # The LLM partitioner ran; the embedding pipeline was never touched.
    mock_llm.assert_called_once()
    mock_embed.assert_not_called()


@needs_gs
def test_cluster_method_auto_small_corpus_uses_llm(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`cluster --method auto` routes a corpus at/below --small-corpus-threshold
    to the LLM partitioner rather than the embedding pipeline."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]

    # One clusterable file, threshold 8 → auto resolves to the LLM partitioner.
    with (
        patch(
            "dgml_core.clustering.llm_cluster_files", return_value=_llm_partition(fid)
        ) as mock_llm,
        patch("dgml_core.clustering.run_clustering_detailed") as mock_embed,
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "auto", "--small-corpus-threshold", "8"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["clusters"] == {fid: "Sample Documents"}
    mock_llm.assert_called_once()
    mock_embed.assert_not_called()


@needs_gs
def test_cluster_partial_success_when_llm_fails(
    tmp_path: Path,
    sample_pdf: Path,
    sample_pdf_alt: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When some clusters match existing DocSets and others need LLM naming,
    an LLM failure on the unmatched cluster leaves the matched files
    assigned and only the unmatched files in ``failed_file_ids``."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )

    # Existing DocSet "Foo" — mock run_clustering to put one file in "Foo"
    # (matches the existing DocSet) and the other in "unknown_0" (needs LLM
    # naming, which we make fail).
    main(_ws_args(ws) + ["docset", "create", "--name", "Foo"])
    existing_id = _read_stdout(capsys)["id"]
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid_a = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["file", "add", str(sample_pdf_alt)])
    fid_b = _read_stdout(capsys)["file"]["id"]
    capsys.readouterr()

    matched_fid, failed_fid = sorted([fid_a, fid_b])
    cluster_assignments = {matched_fid: _dp("Foo", 0.9), failed_fid: _dp("unknown_0")}
    with (
        patch("litellm.completion", side_effect=RuntimeError("network boom")),
        patch(
            "dgml_core.clustering.run_clustering_detailed",
            return_value=cluster_assignments,
        ),
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "embedding"])
    assert rc == 0
    payload = _read_stdout(capsys)

    assert payload["failed_file_ids"] == [failed_fid]
    main(_ws_args(ws) + ["docset", "list-files", existing_id])
    assert _read_stdout(capsys)["file_ids"] == [matched_fid]
    # No new DocSet was created — the LLM call failed for "unknown_0".
    main(_ws_args(ws) + ["docset", "list"])
    assert [d["name"] for d in _read_stdout(capsys)["docsets"]] == ["Foo"]


def test_cluster_without_classification_config_soft_fails(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without classification config, files needing LLM-named DocSets fall
    into ``failed_file_ids`` but the command still returns 0 — partial success
    is the contract, not fail-fast."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]

    with (
        patch("litellm.completion") as mock_completion,
        patch(
            "dgml_core.clustering.run_clustering_detailed",
            return_value={fid: _dp("unknown_0")},
        ),
    ):
        rc = main(_ws_args(ws) + ["cluster", "--method", "embedding"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["failed_file_ids"] == [fid]
    mock_completion.assert_not_called()

    # No DocSet was created — the file is still unassigned.
    main(_ws_args(ws) + ["docset", "list"])
    assert _read_stdout(capsys)["docsets"] == []


def test_check_clean_returns_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["check"])
    assert rc == 0
    report = _read_stdout(capsys)
    assert report["issue_count"] == 0


def test_format_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    rc = main(_ws_args(ws) + ["--format", "text", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "docset_count: 0" in out
    assert "file_count: 0" in out


def _tool_response(name: str, arguments: dict[str, Any]) -> SimpleNamespace:
    call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))]
    )


@needs_gs
def test_file_add_auto_classify_creates_new_docset(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    # Workspace config with classification settings — no existing DocSets, so
    # the LLM is forced to call create_new_docset.

    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )

    new_questions = [
        "What is the vendor name?",
        "What is the total amount?",
        "What is the receipt date?",
    ]
    response = _tool_response(
        "create_new_docset",
        {
            "name": "Receipts",
            "description": "expense receipts",
            "key_questions": new_questions,
        },
    )

    with patch("litellm.completion", return_value=response):
        rc = main(_ws_args(ws) + ["file", "add", str(sample_pdf), "--auto-classify"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert "classification" in payload
    cls = payload["classification"]
    assert cls["performed"] is True
    assert cls["decision"] == "new"
    assert cls["docset_created"] is True
    assert cls["docset_name"] == "Receipts"
    assert cls["docset_key_questions"] == new_questions
    assert cls["error"] is None
    assert cls["model"] == "gemini/gemini-2.5-flash-lite"

    # Persisted: the created DocSet's record carries the key_questions
    # for future classification calls to read.
    rc = main(_ws_args(ws) + ["docset", "show", cls["docset_id"]])
    assert rc == 0
    shown = _read_stdout(capsys)
    assert shown["key_questions"] == new_questions

    # Verify the docset and assignment actually landed.
    rc = main(_ws_args(ws) + ["docset", "list-files", cls["docset_id"]])
    assert rc == 0
    listed = _read_stdout(capsys)
    assert listed["file_ids"] == [payload["file"]["id"]]


@needs_gs
def test_file_add_auto_classify_assigns_existing_docset(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(
        _ws_args(ws)
        + [
            "docset",
            "create",
            "--name",
            "Contracts",
            "--key-question",
            "What is the agreement date?",
            "--key-question",
            "Who are the parties?",
        ]
    )
    docset_payload = _read_stdout(capsys)
    existing_id = docset_payload["id"]
    assert docset_payload["key_questions"] == [
        "What is the agreement date?",
        "Who are the parties?",
    ]

    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )
    response = _tool_response("assign_to_existing_docset", {"docset_id": existing_id})

    with patch("litellm.completion", return_value=response):
        rc = main(_ws_args(ws) + ["file", "add", str(sample_pdf), "--auto-classify"])
    assert rc == 0
    payload = _read_stdout(capsys)
    cls = payload["classification"]
    assert cls["performed"] is True
    assert cls["decision"] == "existing"
    assert cls["docset_id"] == existing_id
    assert cls["docset_created"] is False
    assert cls["docset_key_questions"] == [
        "What is the agreement date?",
        "Who are the parties?",
    ]
    assert cls["error"] is None


@needs_gs
def test_file_add_auto_classify_hard_fails_when_no_config(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No `classification` section in config → --auto-classify is a hard
    failure (exit 1, error envelope), not a per-file soft error. Config is a
    precondition; failing fast beats recording the same error on every file."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    with patch("litellm.completion") as mock_completion:
        rc = main(_ws_args(ws) + ["file", "add", str(sample_pdf), "--auto-classify"])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "CLASSIFICATION_CONFIG_MISSING"
    mock_completion.assert_not_called()


@needs_gs
def test_file_add_auto_classify_soft_fails_when_llm_errors(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )

    with patch("litellm.completion", side_effect=RuntimeError("API down")):
        rc = main(_ws_args(ws) + ["file", "add", str(sample_pdf), "--auto-classify"])
    assert rc == 0
    payload = _read_stdout(capsys)
    cls = payload["classification"]
    assert cls["error"].startswith("CLASSIFICATION_FAILED")
    assert "API down" in cls["error"]


@needs_gs
def test_file_add_auto_classify_skipped_on_duplicate(
    tmp_path: Path, sample_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-adding the same PDF with --auto-classify --on-conflict skip must not
    call the LLM; the existing record is returned and classification is
    reported as performed=false.
    """
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    capsys.readouterr()

    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )

    with patch("litellm.completion") as mock_completion:
        rc = main(
            _ws_args(ws)
            + [
                "file",
                "add",
                str(sample_pdf),
                "--on-conflict",
                "skip",
                "--auto-classify",
            ]
        )
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["created"] is False
    cls = payload["classification"]
    assert cls["performed"] is False
    assert "already existed" in cls["reason"]
    mock_completion.assert_not_called()


def test_file_add_without_auto_classify_omits_block(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The `classification` payload block must be absent when --auto-classify
    isn't passed — keeps the default surface unchanged.
    """
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    rc = main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert "classification" not in payload


def _init_with_docset(ws: Path, capsys: pytest.CaptureFixture[str], name: str = "X") -> str:
    """Init workspace, create one docset, return its id, drain stdout.

    Also writes a ``generation`` config section — ``docset generate`` has no
    code default and no model flags, so it reads both ``model``
    and ``label_model`` (both required) from config.toml.
    """
    _init_ws(ws)
    capsys.readouterr()
    _write_ws_config(
        ws,
        {
            "generation": {
                "model": "anthropic/claude-haiku-4-5",
                "label_model": "anthropic/claude-sonnet-4-6",
            }
        },
    )
    main(_ws_args(ws) + ["docset", "create", "--name", name])
    return str(_read_stdout(capsys)["id"])


# ---------------------------------------------------------------------------
# `dgml docset generate` — PDF→DGML pipeline
# ---------------------------------------------------------------------------


def _read_generate_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """`dgml docset generate` emits a single JSON object on stdout — progress
    lines go to stderr and only under `--verbose`. This asserts stdout is
    pure JSON (no leading progress noise), a regression guard for that fix."""
    out = capsys.readouterr().out
    assert out.lstrip().startswith("{"), f"stdout is not pure JSON:\n{out!r}"
    return json.loads(out)  # type: ignore[no-any-return]


def test_docset_generate_errors_when_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty docset → EMPTY_DOCSET."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    rc = main(_ws_args(ws) + ["docset", "generate", did])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "EMPTY_DOCSET"


@needs_gs
def test_docset_generate_rejects_malformed_style_config(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed `style` section fails fast with STYLE_CONFIG_INVALID, before
    any transcription — surfaced up front rather than per-file during grounding."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()
    # `style` present but no model and no [models].light tier -> invalid
    # (presence of the section is the switch).
    _write_ws_config(
        ws,
        {
            "generation": {
                "model": "anthropic/claude-haiku-4-5",
                "label_model": "anthropic/claude-sonnet-4-6",
            },
            "style": {"enabled": True, "max_tokens": 100},
        },
    )

    rc = main(_ws_args(ws) + ["docset", "generate", did])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "STYLE_CONFIG_INVALID"


@needs_gs
def test_docset_generate_rejects_unset_style_api_key_env(
    tmp_path: Path,
    text_pdf: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `style.api_key_env` pointing at an unset env var fails fast up front
    with AUTH_ERROR — before any transcription spend — rather than being
    swallowed by the best-effort style pass mid-grounding."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()
    monkeypatch.delenv("DGML_STYLE_KEY_MISSING", raising=False)
    _write_ws_config(
        ws,
        {
            "generation": {
                "model": "anthropic/claude-haiku-4-5",
                "label_model": "anthropic/claude-sonnet-4-6",
            },
            "style": {"enabled": True, "model": "m", "api_key_env": "DGML_STYLE_KEY_MISSING"},
        },
    )

    rc = main(_ws_args(ws) + ["docset", "generate", did])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "AUTH_ERROR"


@needs_gs
def test_docset_generate_skips_already_converted(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """If the per-(docset, file) `<stem>.dgml.xml` holds a generated document
    tree for every assigned file, the run short-circuits with
    summary.converted == 0 — convert_batch is never called. This is the
    resume-on-rerun contract. (An extraction-only file does NOT count as
    converted — see test_docset_generate_builds_tree_for_extraction_only_file.)"""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    # Seed the canonical per-file output so the file looks already converted —
    # a root with document-tree content, as generate would have written.
    _wsx = Workspace(root=ws)
    _wsx.blobs.put_blob(
        layout.dgml_xml_key(did, fid, "with-text"),
        b'<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><a>tree</a></dg:chunk>',
    )

    with patch("dgml_core.generation.pipeline.convert_batch") as mock_batch:
        rc = main(_ws_args(ws) + ["docset", "generate", did])
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    # All-skipped short-circuit emits the same unified envelope as a normal
    # run: nested summary + per-item results carrying a status.
    assert payload["summary"] == {"total": 1, "converted": 0, "skipped": 1, "failed": 0}
    (entry,) = payload["results"]
    assert entry["status"] == "skipped"
    assert entry["source"] == "with-text.pdf"
    assert entry["file_id"] == fid
    mock_batch.assert_not_called()


@needs_gs
def test_docset_generate_happy_path(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Happy path: convert_batch is mocked to return one xml string; the
    CLI writes it to the file's per-(docset, file) directory and emits a
    payload of the documented shape, and threads options through."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    # Models come from config — set a distinct label_model to check it threads.
    Workspace(root=ws).config_path.write_text(
        dump_toml(
            {
                "generation": {
                    "model": "anthropic/claude-haiku-4-5",
                    "label_model": "anthropic/claude-sonnet-4-6",
                }
            }
        ),
        encoding="utf-8",
    )
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    out_xml_key = layout.dgml_xml_key(did, fid, "with-text")
    fake_xml = "<xml><chunk>hello</chunk></xml>"

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output("with-text.pdf", fake_xml)  # stream one rendered doc to the CLI sink
        return {}

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert) as mock_batch:
        rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    assert payload["docset_id"] == did
    assert payload["summary"] == {"total": 1, "converted": 1, "skipped": 0, "failed": 0}
    # Slashless: `output_key` is the docset prefix as reported in the JSON payload.
    assert payload["output_key"] == layout.docset_prefix(did).rstrip("/")
    assert payload["coverage_report"] is None  # --no-coverage
    (entry,) = payload["results"]
    assert entry["status"] == "converted"
    assert entry["file_id"] == fid
    assert entry["source"] == "with-text.pdf"
    assert entry["output"] == out_xml_key
    # Generation grounds each file in place. "hello" doesn't match the real OCR,
    # so 0 elements are annotated, but the file is still grounded (the tree is
    # re-serialized, so it no longer byte-equals fake_xml) and the entry says so.
    assert entry["grounded"] is True
    assert "hello" in Workspace(root=ws).blobs.get_blob(out_xml_key).decode("utf-8")

    # Options threaded through to the typed-block ConvertOptions.
    _, kwargs = mock_batch.call_args
    opts = kwargs["options"]
    assert opts.label_model == "anthropic/claude-sonnet-4-6"


def test_docset_generate_cache_dir_and_debug_threading(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The cache dir is always set (it holds functional files the next run
    reloads); --debug only flips ConvertOptions.debug, which gates the
    debug-only artifacts. An explicit --cache-dir always wins."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()
    # The generation cache materializes (zero-copy on LocalStore) at this path.
    cache_dir = Workspace(root=ws).docsets_dir / did / "cache"

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output("with-text.pdf", "<xml/>")
        return {}

    def _run(global_flags: list[str], gen_flags: list[str]) -> Any:
        # Global flags (--debug) precede the subcommand; per-command flags
        # (--cache-dir) follow it.
        argv = (
            _ws_args(ws) + global_flags + ["docset", "generate", did, "--no-coverage"] + gen_flags
        )
        with patch("dgml_core.generation.convert_batch", side_effect=fake_convert) as mock_batch:
            assert main(argv) == 0
        capsys.readouterr()
        # A fresh out_xml each run, so clear the per-(docset, file) slot to avoid
        # the already-converted skip short-circuiting convert_batch.
        _wsx = Workspace(root=ws)
        _wsx.blobs.delete_blob(layout.dgml_xml_key(did, fid, "with-text"))
        return mock_batch.call_args.kwargs["options"]

    # Default: cache dir is the docset cache/, debug off (debug-only files skipped).
    default_opts = _run([], [])
    assert default_opts.cache_dir == cache_dir
    assert default_opts.debug is False
    # --debug: same cache dir, debug on (debug-only files also written).
    debug_opts = _run(["--debug"], [])
    assert debug_opts.cache_dir == cache_dir
    assert debug_opts.debug is True
    # Explicit --cache-dir always wins.
    explicit = tmp_path / "mycache"
    assert _run([], ["--cache-dir", str(explicit)]).cache_dir == explicit


def test_docset_generate_has_no_model_flags() -> None:
    """The model is config-only — there are no --model / --label-model flags, so
    which model runs is a single per-workspace choice (config.toml), matching
    every other model-consuming command. Passing the removed flags is rejected."""
    from dgml.cli import _build_parser

    args = _build_parser().parse_args(["docset", "generate", "somedocset"])
    assert not hasattr(args, "model")
    assert not hasattr(args, "label_model")
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["docset", "generate", "d", "--model", "x"])
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["docset", "generate", "d", "--label-model", "x"])


@needs_gs
def test_docset_generate_missing_config_errors(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No 'generation' section in config.toml → the run fails fast with
    GENERATION_CONFIG_MISSING. There is no code default and no flag override, so
    which model runs is never silent."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    main(_ws_args(ws) + ["docset", "create", "--name", "X"])
    did = str(_read_stdout(capsys)["id"])
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "GENERATION_CONFIG_MISSING"


@needs_gs
def test_docset_generate_models_from_config(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no model flags, transcription and labeling models are read from the
    workspace's 'generation' config section and threaded into ConvertOptions."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    Workspace(root=ws).config_path.write_text(
        dump_toml(
            {
                "generation": {
                    "model": "anthropic/claude-haiku-4-5",
                    "label_model": "anthropic/claude-sonnet-4-6",
                }
            }
        ),
        encoding="utf-8",
    )
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output("with-text.pdf", "<xml/>")
        return {}

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert) as mock_batch:
        rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert rc == 0
    capsys.readouterr()
    opts = mock_batch.call_args.kwargs["options"]
    assert opts.model == "anthropic/claude-haiku-4-5"
    assert opts.label_model == "anthropic/claude-sonnet-4-6"


@needs_gs
def test_docset_generate_schema_path_seeds_roster(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--schema-path loads a schema.json (Schema v1 `tags` map) and threads the
    full schema to ConvertOptions.schema_seed and its parent_role hierarchy to
    ConvertOptions.parent_map."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "tags": {
                    "PaymentTerms": {"name": "PaymentTerms", "role": "the payment clause"},
                    "DueDate": {
                        "name": "DueDate",
                        "role": "when payment is due",
                        "parent_role": "PaymentTerms",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output("with-text.pdf", "<xml/>")
        return {}

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert) as mock_batch:
        rc = main(
            _ws_args(ws)
            + ["docset", "generate", did, "--no-coverage", "--schema-path", str(schema_path)]
        )
    assert rc == 0
    _, kwargs = mock_batch.call_args
    seed = kwargs["options"].schema_seed
    assert seed is not None and set(seed.tags) == {"PaymentTerms", "DueDate"}
    assert seed.tags["PaymentTerms"].role == "the payment clause"
    assert seed.tags["DueDate"].parent_role == "PaymentTerms"
    assert kwargs["options"].roster_seed is None  # full-fidelity seed, no flat roster
    assert kwargs["options"].parent_map == {"DueDate": "PaymentTerms"}


def test_docset_generate_writes_schema_rnc(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After the batch (post-semlink), generate renders the docset's schema.json
    as full-schema.rnc in the docset dir. Synthetic schema only."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    # The real convert_batch writes schema.json during Pass B; the fake stands
    # in for that so the end-of-run RNC render has something to work from.
    docset_dir = ws / "docsets" / did
    docset_dir.mkdir(parents=True, exist_ok=True)
    docset_dir.joinpath("schema.json").write_text(
        json.dumps({"tags": {"SampleTag": {"name": "SampleTag", "role": "a synthetic role"}}}),
        encoding="utf-8",
    )

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output("with-text.pdf", "<xml/>")
        return {}

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert):
        rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert rc == 0
    rnc = docset_dir / "full-schema.rnc"
    assert rnc.exists()
    text = rnc.read_text(encoding="utf-8")
    assert "SampleTag = element SampleTag {" in text
    assert '# Description: "a synthetic role"' in text


def test_docset_generate_reuses_docset_roster_by_default(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An existing cache/concept_roster.json seeds labeling by default;
    --no-roster opts out."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    cache = ws / "docsets" / did / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "concept_roster.json").write_text(
        json.dumps({"client-name": "the client"}), encoding="utf-8"
    )

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output("with-text.pdf", "<xml/>")
        return {}

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert) as mock_batch:
        main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert mock_batch.call_args.kwargs["options"].roster_seed == {"ClientName": "the client"}

    for out in (ws / "docsets" / did / "files").rglob("*.dgml.xml"):
        out.unlink()  # clear outputs so the file isn't skipped on the second run
    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert) as mock_batch:
        main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage", "--no-roster"])
    assert mock_batch.call_args.kwargs["options"].roster_seed is None


def test_docset_generate_reuse_prefers_schema_json(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the docset has BOTH schema.json and cache/concept_roster.json, an
    incremental generate seeds from schema.json (full fidelity: examples, kind,
    hierarchy) and leaves the flat roster unused. Entity-container grouping
    stays a --schema-path opt-in: no parent_map is derived from the reuse."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    docset_dir = ws / "docsets" / did
    cache = docset_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "concept_roster.json").write_text(
        json.dumps({"ClientName": "the client"}), encoding="utf-8"
    )
    (docset_dir / "schema.json").write_text(
        json.dumps(
            {
                "tags": {
                    "ClientName": {
                        "name": "ClientName",
                        "role": "the client",
                        "kind": "inline",
                        "examples": ["Acme Pty Ltd"],
                        "parent_role": "PartyInformation",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output("with-text.pdf", "<xml/>")
        return {}

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert) as mock_batch:
        main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    options = mock_batch.call_args.kwargs["options"]
    seed = options.schema_seed
    assert seed is not None and seed.tags["ClientName"].examples == ["Acme Pty Ltd"]
    assert options.roster_seed is None
    assert options.parent_map is None  # grouping stays --schema-path opt-in


@needs_gs
def test_docset_generate_missing_source_is_per_file_failure(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file whose source PDF has gone missing does not abort the whole
    run — it becomes a `failed` entry in `results` and the batch exits 0
    (partial success, matching `dgml cluster`)."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    # Remove the copied-in source so generation can't find it.
    _wsx = Workspace(root=ws)
    for _k in _wsx.blobs.list_blobs(layout.file_prefix(fid)):
        if _k.endswith(".pdf"):
            _wsx.blobs.delete_blob(_k)

    with patch("dgml_core.generation.pipeline.convert_batch") as mock_batch:
        rc = main(_ws_args(ws) + ["docset", "generate", did])
    assert rc == 0  # partial success, not an aborting error envelope
    payload = _read_generate_stdout(capsys)
    assert payload["summary"] == {"total": 1, "converted": 0, "skipped": 0, "failed": 1}
    (entry,) = payload["results"]
    assert entry["status"] == "failed"
    assert entry["file_id"] == fid
    assert entry["error"]["code"] == "FILE_NOT_FOUND"
    mock_batch.assert_not_called()  # nothing convertible → no LLM call


@needs_gs
def test_docset_generate_mixed_converted_and_failed(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Partial success with convert_batch actually called: one file converts,
    one (missing source) fails — both appear in results and counts sum to total."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])  # with-text.pdf
    fid_ok = _read_stdout(capsys)["file"]["id"]
    other = tmp_path / "other.pdf"
    _write_text_pdf(other, ["Other one", "Other two"])
    main(_ws_args(ws) + ["file", "add", str(other)])
    fid_bad = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid_ok, "--docset", did])
    main(_ws_args(ws) + ["docset", "add-file", fid_bad, "--docset", did])
    capsys.readouterr()
    _wsx = Workspace(root=ws)
    for _k in _wsx.blobs.list_blobs(layout.file_prefix(fid_bad)):
        if _k.endswith(".pdf"):
            _wsx.blobs.delete_blob(_k)  # break the second file's source

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output("with-text.pdf", "<xml/>")  # only the present file converts
        return {}

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert) as mock_batch:
        rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    assert payload["summary"] == {"total": 2, "converted": 1, "skipped": 0, "failed": 1}
    by_status = {r["status"]: r for r in payload["results"]}
    assert by_status["converted"]["file_id"] == fid_ok
    assert by_status["failed"]["file_id"] == fid_bad
    assert by_status["failed"]["error"]["code"] == "FILE_NOT_FOUND"
    mock_batch.assert_called_once()


@needs_gs
def test_docset_generate_transcription_failure_is_reconciled(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """convert_batch silently drops a doc whose transcription failed (no
    on_output). The CLI reconciles it into a `failed` result instead of letting
    it vanish, keeping summary counts == total."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        return {}  # transcription failed for the only doc → on_output never called

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert) as mock_batch:
        rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    assert payload["summary"] == {"total": 1, "converted": 0, "skipped": 0, "failed": 1}
    (entry,) = payload["results"]
    assert entry["status"] == "failed"
    assert entry["file_id"] == fid
    assert entry["error"]["code"] == "GENERATION_FAILED"
    # No on_error reason captured → the generic fallback message stands.
    assert entry["error"]["message"] == "the generation pipeline produced no output for this file"
    mock_batch.assert_called_once()


@needs_gs
def test_docset_generate_surfaces_transcription_error_reason(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A transcription failure's short cause rides in the JSON `error.message`
    (without --verbose), instead of the generic "produced no output" string."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    def fake_convert(
        paths: object, *, options: object, on_output: Any, on_error: Any, **_kw: object
    ) -> dict[str, str]:
        # convert_batch reports the dropped doc's short reason via on_error.
        on_error("with-text.pdf", "InternalServerError: provider overloaded")
        return {}

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert):
        rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    assert payload["summary"] == {"total": 1, "converted": 0, "skipped": 0, "failed": 1}
    (entry,) = payload["results"]
    assert entry["error"]["code"] == "GENERATION_FAILED"
    assert entry["error"]["message"] == "InternalServerError: provider overloaded"


def test_uncaught_error_envelope_is_short_without_verbose(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unexpected (non-DgmlError) failure yields a short, single-line
    INTERNAL_ERROR envelope on stderr — no traceback dumped without --verbose,
    so stderr stays a clean JSON object."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    long = "boom " * 200  # ~1000 chars of provider-error-style noise
    with patch("dgml.cli._dispatch", side_effect=RuntimeError(long)):
        rc = main(_ws_args(ws) + ["status"])
    assert rc != 0
    err = _read_stderr(capsys)  # parses cleanly → stderr held only the envelope
    assert err["error"]["code"] == "INTERNAL_ERROR"
    msg = err["error"]["message"]
    assert msg.startswith("RuntimeError:")
    assert len(msg) <= 300
    assert msg.endswith("...")


def test_uncaught_error_full_traceback_on_stderr_with_verbose(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--verbose adds the full traceback to stderr, alongside the envelope."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    with patch("dgml.cli._dispatch", side_effect=RuntimeError("kaboom detail")):
        rc = main(_ws_args(ws) + ["--verbose", "status"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err
    assert "RuntimeError: kaboom detail" in err
    assert "INTERNAL_ERROR" in err  # the envelope is still emitted


@needs_gs
def test_docset_generate_duplicate_filename_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two files sharing a basename in one docset can't both convert (the
    pipeline keys docs by filename), so both are reported `failed` rather than
    one silently overwriting the other; convert_batch is not called."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write_text_pdf(tmp_path / "a" / "dup.pdf", ["Alpha one", "Alpha two"])
    _write_text_pdf(tmp_path / "b" / "dup.pdf", ["Bravo one", "Bravo two"])
    main(_ws_args(ws) + ["file", "add", str(tmp_path / "a" / "dup.pdf")])
    fid_a = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["file", "add", str(tmp_path / "b" / "dup.pdf")])
    fid_b = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid_a, "--docset", did])
    main(_ws_args(ws) + ["docset", "add-file", fid_b, "--docset", did])
    capsys.readouterr()

    with patch("dgml_core.generation.convert_batch") as mock_batch:
        rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    assert payload["summary"] == {"total": 2, "converted": 0, "skipped": 0, "failed": 2}
    assert {r["status"] for r in payload["results"]} == {"failed"}
    assert all(r["error"]["code"] == "GENERATION_FAILED" for r in payload["results"])
    assert {r["file_id"] for r in payload["results"]} == {fid_a, fid_b}
    mock_batch.assert_not_called()


def test_load_schema_roster_errors(tmp_path: Path) -> None:
    """_load_schema_roster rejects missing files, non-object/invalid JSON, and
    rosters that sanitize to no usable concepts — all as InvalidArgument."""
    from dgml.cli import _load_schema_roster
    from dgml_core.errors import InvalidArgument

    with pytest.raises(InvalidArgument):
        _load_schema_roster(tmp_path / "missing.json")

    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_roster(arr)

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_roster(bad)

    junk = tmp_path / "junk.json"
    junk.write_text(json.dumps({"###": "x", "!!!": "y"}), encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_roster(junk)


# ---------------------------------------------------------------------------
# `dgml file add <directory>` — bulk ingest (Option A)
# ---------------------------------------------------------------------------


@needs_gs
def test_file_add_directory_clean_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A directory of healthy PDFs: every file adds cleanly, one envelope
    with a summary and a per-file array (each entry the standard add shape)."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    src = tmp_path / "pdfs"
    src.mkdir()
    _write_text_pdf(src / "a.pdf", ["Alpha page one", "Alpha page two"])
    _write_text_pdf(src / "b.pdf", ["Bravo page one", "Bravo page two"])

    rc = main(_ws_args(ws) + ["file", "add", str(src)])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["directory"] == str(src)
    assert payload["recursive"] is False
    assert payload["summary"] == {
        "total": 2,
        "added": 2,
        "skipped": 0,
        "soft_failed": 0,
        "hard_failed": 0,
    }
    # Per-file entries are lex-sorted and carry the standard `file add` shape
    # plus a `path`. No classification block without --auto-classify.
    assert [e["path"] for e in payload["results"]] == [str(src / "a.pdf"), str(src / "b.pdf")]
    for entry in payload["results"]:
        assert entry["created"] is True
        assert entry["text_extraction_error"] is None
        assert "file" in entry
        assert "classification" not in entry


@needs_gs
def test_file_add_directory_skips_unconfigured_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no `conversion` config, convertible sources (docx/xlsx) in a bulk
    directory are silently skipped — not gathered, not counted as failures."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    src = tmp_path / "mixed"
    src.mkdir()
    _write_text_pdf(src / "a.pdf", ["Alpha"])
    (src / "notes.docx").write_bytes(b"PK\x03\x04 not really a docx")

    rc = main(_ws_args(ws) + ["file", "add", str(src)])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["summary"]["total"] == 1
    assert [e["path"] for e in payload["results"]] == [str(src / "a.pdf")]


@needs_gs
def test_file_add_directory_mixed_soft_and_hard_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bad PDF doesn't poison the run: it lands in `hard_failed` with an
    `error` entry, a text-less scan lands in `soft_failed`, and a non-PDF is
    ignored entirely. Exit code stays 0."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    src = tmp_path / "pdfs"
    src.mkdir()
    _write_text_pdf(src / "good.pdf", ["Has text page one", "Has text page two"])
    _write_blank_pdf(src / "blank.pdf", pages=1)  # no digital text → soft fail
    (src / "broken.pdf").write_bytes(b"not a pdf at all\n")  # bad magic → hard fail
    (src / "notes.txt").write_text("ignored", encoding="utf-8")  # not a .pdf

    rc = main(_ws_args(ws) + ["file", "add", str(src)])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["summary"] == {
        "total": 3,  # notes.txt is not counted
        "added": 1,
        "skipped": 0,
        "soft_failed": 1,
        "hard_failed": 1,
    }

    by_path = {Path(e["path"]).name: e for e in payload["results"]}
    # Every entry carries a `status` matching the summary buckets.
    assert by_path["good.pdf"]["status"] == "added"
    assert by_path["good.pdf"]["created"] is True
    assert by_path["good.pdf"]["text_extraction_error"] is None
    assert by_path["blank.pdf"]["status"] == "soft_failed"
    assert by_path["blank.pdf"]["created"] is True
    assert by_path["blank.pdf"]["text_extraction_error"] is not None
    # Hard-failed entry has a structured error and no `file` record.
    assert by_path["broken.pdf"]["status"] == "hard_failed"
    assert by_path["broken.pdf"]["error"]["code"] == "INVALID_PDF"
    assert "file" not in by_path["broken.pdf"]


@needs_gs
def test_file_add_directory_skip_already_imported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Re-running with --on-conflict skip against an already-imported set is
    idempotent: every file is `skipped`, nothing added."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    src = tmp_path / "pdfs"
    src.mkdir()
    _write_text_pdf(src / "a.pdf", ["Alpha page one", "Alpha page two"])
    _write_text_pdf(src / "b.pdf", ["Bravo page one", "Bravo page two"])

    rc = main(_ws_args(ws) + ["file", "add", str(src)])
    assert rc == 0
    assert _read_stdout(capsys)["summary"]["added"] == 2

    rc = main(_ws_args(ws) + ["file", "add", str(src), "--on-conflict", "skip"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["summary"] == {
        "total": 2,
        "added": 0,
        "skipped": 2,
        "soft_failed": 0,
        "hard_failed": 0,
    }
    for entry in payload["results"]:
        assert entry["created"] is False
        assert entry["conflict_kind"] == "hash"


@needs_gs
def test_file_add_directory_recursive(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--recursive descends into subdirectories; default scans top level only."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    src = tmp_path / "pdfs"
    (src / "sub").mkdir(parents=True)
    _write_text_pdf(src / "top.pdf", ["Top page one", "Top page two"])
    _write_text_pdf(src / "sub" / "nested.pdf", ["Nested page one", "Nested page two"])

    # Default: only the top-level PDF is seen.
    rc = main(_ws_args(ws) + ["file", "add", str(src)])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["summary"]["total"] == 1
    assert payload["results"][0]["path"] == str(src / "top.pdf")

    # --recursive picks up the nested PDF too; top.pdf is now a skip.
    rc = main(_ws_args(ws) + ["file", "add", str(src), "--recursive", "--on-conflict", "skip"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["recursive"] is True
    assert payload["summary"]["total"] == 2
    paths = sorted(Path(e["path"]).name for e in payload["results"])
    assert paths == ["nested.pdf", "top.pdf"]


@needs_gs
def test_file_add_directory_auto_classify_amortizes_docsets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--auto-classify across a directory: the config loads once and a DocSet
    created for the first file is visible to the second, which is assigned to
    it (no second DocSet)."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    write_classification_config(
        Workspace(root=ws), {"model": "gemini/gemini-2.5-flash-lite", "max_pages": 1}
    )

    src = tmp_path / "pdfs"
    src.mkdir()
    _write_text_pdf(src / "a.pdf", ["Alpha page one", "Alpha page two"])
    _write_text_pdf(src / "b.pdf", ["Bravo page one", "Bravo page two"])

    calls = {"n": 0}

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        calls["n"] += 1
        if calls["n"] == 1:
            return _tool_response(
                "create_new_docset",
                {
                    "name": "Docs",
                    "description": "test docs",
                    "key_questions": ["What is this?", "Who wrote it?", "When?"],
                },
            )
        # Second file: the DocSet created for the first must now be offered
        # in the assign tool's enum — proving in-run visibility.
        enum = kwargs["tools"][0]["function"]["parameters"]["properties"]["docset_id"]["enum"]
        assert len(enum) == 1
        return _tool_response("assign_to_existing_docset", {"docset_id": enum[0]})

    with patch("litellm.completion", side_effect=fake_completion):
        rc = main(_ws_args(ws) + ["file", "add", str(src), "--auto-classify"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["summary"]["added"] == 2

    first, second = payload["results"]  # lex-sorted: a.pdf, b.pdf
    assert first["classification"]["decision"] == "new"
    assert first["classification"]["docset_created"] is True
    new_id = first["classification"]["docset_id"]
    assert second["classification"]["decision"] == "existing"
    assert second["classification"]["docset_created"] is False
    assert second["classification"]["docset_id"] == new_id

    # Exactly one DocSet exists, holding both files.
    main(_ws_args(ws) + ["docset", "list"])
    docsets = _read_stdout(capsys)["docsets"]
    assert len(docsets) == 1
    main(_ws_args(ws) + ["docset", "list-files", new_id])
    assert len(_read_stdout(capsys)["file_ids"]) == 2


def test_file_add_directory_auto_classify_hard_fails_without_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Bulk --auto-classify with no classification config aborts up front
    (exit 1) — config is loaded once before the loop, so no file is added."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    src = tmp_path / "pdfs"
    src.mkdir()
    _write_text_pdf(src / "a.pdf", ["Alpha one", "Alpha two"])

    with patch("litellm.completion") as mock_completion:
        rc = main(_ws_args(ws) + ["file", "add", str(src), "--auto-classify"])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "CLASSIFICATION_CONFIG_MISSING"
    mock_completion.assert_not_called()

    # Fail-fast: the run aborted before adding any files.
    main(_ws_args(ws) + ["status"])
    assert _read_stdout(capsys)["file_count"] == 0


def test_format_text_handles_nested(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Text format must render nested lists/dicts readably, not as repr()."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    main(_ws_args(ws) + ["docset", "create", "--name", "Alpha"])
    capsys.readouterr()
    rc = main(_ws_args(ws) + ["--format", "text", "docset", "list"])
    assert rc == 0
    out = capsys.readouterr().out
    # No raw repr leaking through:
    assert "[{" not in out
    assert "{'id'" not in out
    # Hierarchy is visible:
    assert "docsets:" in out
    assert "name: Alpha" in out


# ---- docset generate: in-place grounding ------------------------------------

# A namespaced DGML doc whose Body text matches the seeded page OCR words, so
# the in-place grounding pass annotates Body with a dg:origin box.
_GROUNDABLE_XML = (
    '<dg:chunk xmlns:dg="http://dgml.io">'
    "<Body>Payment is due within 30 days of invoice</Body>"
    "</dg:chunk>"
)


def _seed_file_for_generate(
    ws_root: Path, docset_id: str, file_id: str, *, with_page_text: bool = True
) -> Workspace:
    """Seed a file (record + a placeholder PDF + optional page_text) and
    assign it to the docset, so `docset generate` — with convert_batch
    mocked — can run and ground the rendered XML in place. Returns the
    Workspace."""
    from dgml_core.docsets import DocSetStore
    from dgml_core.models import FileRecord

    ws = Workspace(root=ws_root)
    record = FileRecord(
        id=file_id,
        original_path="/fake/contract.pdf",
        original_filename="contract.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=1,
        text_mode="digital",
    )
    ws.docs.put_doc("files", file_id, record.to_json())
    # generate resolves the source PDF from the store; convert_batch is
    # mocked, so the bytes are never parsed — they just need to exist.
    ws.blobs.put_blob(layout.file_source_key(file_id, "contract.pdf"), b"%PDF-1.4\n%fake\n")
    if with_page_text:
        words = []
        x = 100
        for w in "Payment is due within 30 days of invoice".split():
            words.append({"t": w, "l": [x, 100, x + 50, 120]})
            x += 60
        ws.blobs.put_blob(
            layout.file_page_text_key(file_id, 1),
            json.dumps(
                {"file_id": file_id, "page": 1, "width": 1000, "height": 1000, "words": words}
            ).encode(),
        )
    DocSetStore(ws).add_file(docset_id, file_id)
    return ws


def _generate_with_xml(
    ws_root: Path,
    ds_id: str,
    xml: str,
    *,
    debug: bool = False,
    label_error: dict[str, str] | None = None,
) -> int:
    """Run `docset generate` with convert_batch mocked to emit one rendered
    doc (`xml`) for the seeded contract.pdf. When *label_error* is given, the
    mock also fires the labeling-failure callback for that file. Returns the
    exit code."""

    def fake_convert(
        paths: object,
        *,
        options: object,
        on_output: Any,
        on_label_error: Any = None,
        **_kw: object,
    ) -> dict[str, str]:
        if label_error is not None and on_label_error is not None:
            on_label_error("contract.pdf", label_error)
        on_output("contract.pdf", xml)
        return {}

    # generate reads the models from config.toml's 'generation' section (no flags).
    # Real-provider model strings so the pre-flight check (get_llm_provider)
    # accepts them; convert_batch is mocked, so no call is ever made. The dummy
    # ANTHROPIC_API_KEY from conftest satisfies the pre-flight key check.
    Workspace(root=ws_root).config_path.write_text(
        dump_toml(
            {
                "generation": {
                    "model": "anthropic/claude-haiku-4-5",
                    "label_model": "anthropic/claude-sonnet-4-6",
                }
            }
        ),
        encoding="utf-8",
    )
    extra = ["--debug"] if debug else []
    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert):
        return main(_ws_args(ws_root) + ["docset", "generate", ds_id, "--no-coverage", *extra])


def test_docset_generate_grounds_in_place(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Generation grounds each <stem>.dgml.xml in place: dg:origin boxes land
    in the canonical file (not a separate .grounded.xml), bound to the
    document's dg prefix. No stats sidecar without --debug."""
    ws_root = tmp_path / "ws"
    _init_ws(ws_root)
    capsys.readouterr()
    main(_ws_args(ws_root) + ["docset", "create", "--name", "Contracts"])
    ds_id = _read_stdout(capsys)["id"]
    ws = _seed_file_for_generate(ws_root, ds_id, "f1aaaaaaaaaa")
    out_xml_key = layout.dgml_xml_key(ds_id, "f1aaaaaaaaaa", "contract")

    rc = _generate_with_xml(ws_root, ds_id, _GROUNDABLE_XML)
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    assert payload["summary"] == {"total": 1, "converted": 1, "skipped": 0, "failed": 0}
    (entry,) = payload["results"]
    assert entry["status"] == "converted"
    assert entry["source"] == "contract.pdf"
    assert entry["grounded"] is True
    assert entry["matched_token_pct"] == 100.0
    # The Body leaf plus the root dg:chunk container (page-union box).
    assert entry["elements_annotated"] == 2
    # No labeling failure → no label_error field on the entry (like grounding_error).
    assert "label_error" not in entry

    content = ws.blobs.get_blob(out_xml_key).decode("utf-8")
    assert 'dg:origin="1 ' in content  # bound to the document's dg prefix
    # Grounded in place — no separate .grounded.xml, no stats sidecar by default.
    _out_dir = layout.docset_pair_prefix(ds_id, "f1aaaaaaaaaa")
    assert not ws.blobs.blob_exists(f"{_out_dir}contract.dgml.grounded.xml")
    assert not ws.blobs.blob_exists(f"{_out_dir}contract.dgml.grounding_stats.json")


def test_docset_generate_debug_writes_grounding_stats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The global --debug flag writes the per-file grounding_stats.json sidecar."""
    ws_root = tmp_path / "ws"
    _init_ws(ws_root)
    capsys.readouterr()
    main(_ws_args(ws_root) + ["docset", "create", "--name", "Contracts"])
    ds_id = _read_stdout(capsys)["id"]
    ws = _seed_file_for_generate(ws_root, ds_id, "f1aaaaaaaaaa")

    rc = _generate_with_xml(ws_root, ds_id, _GROUNDABLE_XML, debug=True)
    assert rc == 0
    assert ws.blobs.blob_exists(
        f"{layout.docset_pair_prefix(ds_id, 'f1aaaaaaaaaa')}contract.dgml.grounding_stats.json"
    )


def test_docset_generate_leaves_file_ungrounded_without_page_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file with no page_text is still converted but left ungrounded — the
    run succeeds and the result entry records grounded=False with the reason."""
    ws_root = tmp_path / "ws"
    _init_ws(ws_root)
    capsys.readouterr()
    main(_ws_args(ws_root) + ["docset", "create", "--name", "Contracts"])
    ds_id = _read_stdout(capsys)["id"]
    ws = _seed_file_for_generate(ws_root, ds_id, "f1aaaaaaaaaa", with_page_text=False)
    out_xml_key = layout.dgml_xml_key(ds_id, "f1aaaaaaaaaa", "contract")

    rc = _generate_with_xml(ws_root, ds_id, _GROUNDABLE_XML)
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    (entry,) = payload["results"]
    assert entry["status"] == "converted"
    assert entry["grounded"] is False
    assert entry["grounding_error"]["code"] == "FILE_NOT_FOUND"
    assert ws.blobs.blob_exists(out_xml_key)  # still written, just not grounded
    assert "dg:origin" not in ws.blobs.get_blob(out_xml_key).decode("utf-8")


def test_docset_generate_surfaces_label_error_but_still_converts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When labeling can't reach the model at runtime, the file still converts —
    status converted, exit 0, DGML written — and the failure is surfaced as
    label_error on the entry, so a misconfigured label_model is visible in the
    normal (non --verbose) JSON."""
    ws_root = tmp_path / "ws"
    _init_ws(ws_root)
    capsys.readouterr()
    main(_ws_args(ws_root) + ["docset", "create", "--name", "Contracts"])
    ds_id = _read_stdout(capsys)["id"]
    ws = _seed_file_for_generate(ws_root, ds_id, "f1aaaaaaaaaa")
    out_xml_key = layout.dgml_xml_key(ds_id, "f1aaaaaaaaaa", "contract")

    rc = _generate_with_xml(
        ws_root,
        ds_id,
        _GROUNDABLE_XML,
        label_error={
            "code": "LABEL_MODEL_UNREACHABLE",
            "message": "AuthenticationError: invalid x-api-key",
        },
    )
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    assert payload["summary"] == {"total": 1, "converted": 1, "skipped": 0, "failed": 0}
    (entry,) = payload["results"]
    assert entry["status"] == "converted"
    assert entry["label_error"]["code"] == "LABEL_MODEL_UNREACHABLE"
    assert "AuthenticationError" in entry["label_error"]["message"]
    assert ws.blobs.blob_exists(out_xml_key)  # transcription/DGML never discarded


def _seed_docset_with_one_file(ws_root: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """Init a workspace + docset with one hermetically-seeded file; return the
    docset id. No generation config is written — the caller sets one to exercise
    the pre-flight check."""
    _init_ws(ws_root)
    capsys.readouterr()
    main(_ws_args(ws_root) + ["docset", "create", "--name", "Contracts"])
    ds_id = str(_read_stdout(capsys)["id"])
    _seed_file_for_generate(ws_root, ds_id, "f1aaaaaaaaaa")
    return ds_id


def test_docset_generate_preflight_rejects_malformed_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pre-flight: a model string with no resolvable provider fails fast with
    GENERATION_CONFIG_INVALID before any transcription — convert_batch is never
    called."""
    ws_root = tmp_path / "ws"
    ds_id = _seed_docset_with_one_file(ws_root, capsys)
    Workspace(root=ws_root).config_path.write_text(
        dump_toml({"generation": {"model": "::::", "label_model": "anthropic/claude-sonnet-4-6"}}),
        encoding="utf-8",
    )
    with patch("dgml_core.generation.convert_batch") as mock_batch:
        rc = main(_ws_args(ws_root) + ["docset", "generate", ds_id, "--no-coverage"])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "GENERATION_CONFIG_INVALID"
    mock_batch.assert_not_called()


def test_docset_generate_preflight_rejects_missing_api_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-flight: a well-formed model whose provider key is absent fails fast
    with AUTH_ERROR before any transcription."""
    ws_root = tmp_path / "ws"
    ds_id = _seed_docset_with_one_file(ws_root, capsys)
    # Undo the conftest dummy key so the provider key is genuinely absent.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    Workspace(root=ws_root).config_path.write_text(
        dump_toml(
            {
                "generation": {
                    "model": "anthropic/claude-haiku-4-5",
                    "label_model": "anthropic/claude-sonnet-4-6",
                }
            }
        ),
        encoding="utf-8",
    )
    with patch("dgml_core.generation.convert_batch") as mock_batch:
        rc = main(_ws_args(ws_root) + ["docset", "generate", ds_id, "--no-coverage"])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "AUTH_ERROR"
    mock_batch.assert_not_called()


def test_docset_generate_preflight_skips_key_check_when_api_base_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-flight: with api_base set (proxy / self-hosted endpoint) the
    key-presence check is skipped, so a run with no provider key still proceeds —
    guards against a false abort on custom endpoints."""
    ws_root = tmp_path / "ws"
    _init_ws(ws_root)
    capsys.readouterr()
    main(_ws_args(ws_root) + ["docset", "create", "--name", "Contracts"])
    ds_id = str(_read_stdout(capsys)["id"])
    _seed_file_for_generate(ws_root, ds_id, "f1aaaaaaaaaa")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    Workspace(root=ws_root).config_path.write_text(
        dump_toml(
            {
                "generation": {
                    "model": "anthropic/claude-haiku-4-5",
                    "label_model": "anthropic/claude-sonnet-4-6",
                    "api_base": "http://localhost:8000",
                    "label_api_base": "http://localhost:8000",
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output("contract.pdf", _GROUNDABLE_XML)
        return {}

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert) as mock_batch:
        rc = main(_ws_args(ws_root) + ["docset", "generate", ds_id, "--no-coverage"])
    assert rc == 0
    mock_batch.assert_called_once()  # pre-flight did not abort


# --- dgmlx export / verify --------------------------------------------------


def _seed_file_dir(
    ws: Path,
    file_id: str,
    *,
    pages: int,
    pdf_name: str = "doc.pdf",
) -> None:
    """Build a file directory (file.json + source + page images + page text)
    directly on disk, no PDF pipeline / ghostscript needed — the
    attestation hashes bytes, not document semantics."""
    workspace = Workspace(root=ws)
    workspace.blobs.put_blob(layout.file_source_key(file_id, pdf_name), b"%PDF-1.4\n%fake\n")
    workspace.docs.put_doc(
        "files",
        file_id,
        {
            "id": file_id,
            "original_path": f"/src/{pdf_name}",
            "original_filename": pdf_name,
            "sha256": "0" * 64,
            "added_at": "2026-06-05T00:00:00Z",
            "page_count": pages,
            "text_mode": "digital",
        },
    )
    for n in range(1, pages + 1):
        workspace.blobs.put_blob(layout.file_page_image_key(file_id, n), f"img-{n}".encode())
        workspace.blobs.put_blob(
            layout.file_page_text_key(file_id, n),
            json.dumps({"file_id": file_id, "page": n, "words": []}).encode(),
        )


def test_dgmlx_export_then_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, "f0000000001a", pages=2)
    out_dir = tmp_path / "bundle"
    capsys.readouterr()  # drain the init payload

    rc = main(_ws_args(ws) + ["dgmlx", "export", "f0000000001a", "--output-dir", str(out_dir)])
    assert rc == 0
    exported = _read_stdout(capsys)
    assert exported["file_id"] == "f0000000001a"
    assert exported["docset_id"] is None
    # Default is archive-only: the <stem>.dgmlx is the sole output. No loose
    # attestation field/file, and no "manifest" field (folded into the archive).
    assert "manifest" not in exported
    assert "attestation" not in exported
    archive = out_dir / "doc.dgmlx"
    assert exported["dgmlx"] == str(archive)
    assert archive.exists()
    assert list(out_dir.iterdir()) == [archive]  # nothing loose left behind
    assert exported["slots"] == ["source", "page_image[1]", "page_image[2]"]
    assert len(exported["root"]) == 64

    # `verify` reads the .dgmlx archive directly.
    rc = main(_ws_args(ws) + ["dgmlx", "verify", str(archive)])
    assert rc == 0
    verified = _read_stdout(capsys)
    assert verified["valid"] is True
    assert verified["expected_root"] == verified["computed_root"] == exported["root"]


def test_dgmlx_export_converted_source_excludes_working_pdf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exporting a converted (non-PDF) source bundles only the original under
    `source/` — the converted working PDF is not attested and gets no `pdf` slot
    or `pdf/` part."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, "f0000000001a", pages=1, pdf_name="report.docx")
    out_dir = tmp_path / "bundle"
    capsys.readouterr()  # drain the init payload

    rc = main(
        _ws_args(ws)
        + ["dgmlx", "export", "f0000000001a", "--output-dir", str(out_dir), "--unpacked"]
    )
    assert rc == 0
    exported = _read_stdout(capsys)
    assert exported["slots"] == ["source", "page_image[1]"]
    assert (out_dir / "source" / "report.docx").exists()
    assert not (out_dir / "pdf").exists()

    rc = main(_ws_args(ws) + ["dgmlx", "verify", str(out_dir)])
    assert rc == 0
    assert _read_stdout(capsys)["valid"] is True


def test_dgmlx_export_unpacked_writes_loose_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, "f0000000001a", pages=2)
    out_dir = tmp_path / "bundle"
    capsys.readouterr()

    rc = main(
        _ws_args(ws)
        + ["dgmlx", "export", "f0000000001a", "--output-dir", str(out_dir), "--unpacked"]
    )
    assert rc == 0
    exported = _read_stdout(capsys)
    # --unpacked leaves the loose tree (and surfaces its attestation path) and
    # produces NO archive — the two modes are mutually exclusive.
    attestation_file = out_dir / "META-INF" / "dgml-attestation.xml"
    assert exported["attestation"] == str(attestation_file)
    assert "dgmlx" not in exported
    assert attestation_file.exists()
    assert not (out_dir / "doc.dgmlx").exists()
    assert (out_dir / "[Content_Types].xml").exists()
    assert (out_dir / "_rels" / ".rels").exists()
    # The attestation file carries the workspace identity.
    assert 'file-id="f0000000001a"' in attestation_file.read_text(encoding="utf-8")

    # `verify` reads the loose directory too.
    rc = main(_ws_args(ws) + ["dgmlx", "verify", str(out_dir)])
    assert rc == 0
    assert _read_stdout(capsys)["valid"] is True


def test_dgmlx_verify_detects_tamper(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, "f0000000001a", pages=1)
    out_dir = tmp_path / "bundle"
    # --unpacked so a loose artifact is on disk to tamper with.
    main(
        _ws_args(ws)
        + ["dgmlx", "export", "f0000000001a", "--output-dir", str(out_dir), "--unpacked"]
    )
    capsys.readouterr()

    (out_dir / "page_images" / "page_1.png").write_bytes(b"TAMPERED")
    rc = main(_ws_args(ws) + ["dgmlx", "verify", str(out_dir)])
    assert rc == 2  # mirrors `check`: verified-but-failed
    payload = _read_stdout(capsys)
    assert payload["valid"] is False
    assert payload["computed_root"] != payload["expected_root"]


def test_dgmlx_verify_malformed_bundle_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    empty = tmp_path / "no-manifest"
    empty.mkdir()
    rc = main(_ws_args(ws) + ["dgmlx", "verify", str(empty)])
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "ATTESTATION_INVALID"


# --- node export / prove ------------------------------------------------------


_NODE_XML = (
    b'<dg:chunk xmlns:dg="http://dgml.io" '
    b'xmlns:docset="http://example.com/ds">'
    b"<docset:Header>Ledger</docset:Header>"
    b"<docset:Entry><docset:Amount>100</docset:Amount></docset:Entry>"
    b"</dg:chunk>"
)


def _seed_node_xml(ws: Path, file_id: str, docset_id: str) -> None:
    """Add the docset dir + generated DGML XML on top of _seed_file_dir."""
    workspace = Workspace(root=ws)
    workspace.docs.put_doc(
        "docsets", docset_id, {"id": docset_id, "name": "T", "description": "", "key_questions": []}
    )
    workspace.blobs.put_blob(layout.dgml_xml_key(docset_id, file_id, "doc"), _NODE_XML)


def test_node_export_then_prove(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, "f0000000001a", pages=1)
    _seed_node_xml(ws, "f0000000001a", "ds000000001a")
    capsys.readouterr()

    rc = main(
        _ws_args(ws)
        + [
            "node",
            "export",
            "f0000000001a",
            "--docset",
            "ds000000001a",
            "--xpath",
            "/dg:chunk/docset:Entry/docset:Amount",
        ]
    )
    assert rc == 0
    exported = _read_stdout(capsys)
    assert exported["file_id"] == "f0000000001a"
    assert exported["docset_id"] == "ds000000001a"
    assert exported["xpath"] == "/dg:chunk/docset:Entry/docset:Amount"
    assert exported["leaf_index"] == 3
    assert exported["leaf_count"] == 4
    assert len(exported["node_hash"]) == 64
    assert len(exported["root_hash"]) == 64
    assert exported["proof"]["leaf_hash"] == exported["node_hash"]
    assert "100" in exported["node_xml"]

    proof_file = tmp_path / "proof.json"
    proof_file.write_text(json.dumps(exported), encoding="utf-8")
    rc = main(
        _ws_args(ws)
        + ["node", "prove", "f0000000001a", "--docset", "ds000000001a", "--proof", str(proof_file)]
    )
    assert rc == 0
    proven = _read_stdout(capsys)
    assert proven["valid"] is True
    assert proven["computed_node_hash"] == exported["node_hash"]


def test_node_export_by_leaf_matches_xpath(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, "f0000000001a", pages=1)
    _seed_node_xml(ws, "f0000000001a", "ds000000001a")
    capsys.readouterr()

    rc = main(
        _ws_args(ws) + ["node", "export", "f0000000001a", "--docset", "ds000000001a", "--leaf", "1"]
    )
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["xpath"] == "/dg:chunk/docset:Header"


def test_node_export_by_child_path_matches_xpath(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    main(_ws_args(ws) + ["init"])
    _init_ws(ws)
    _seed_file_dir(ws, "f0000000001a", pages=1)
    _seed_node_xml(ws, "f0000000001a", "ds000000001a")
    capsys.readouterr()

    # root -> Entry (2nd child) -> Amount (1st child).
    rc = main(
        _ws_args(ws)
        + [
            "node",
            "export",
            "f0000000001a",
            "--docset",
            "ds000000001a",
            "--child-path",
            "1/0",
        ]
    )
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["xpath"] == "/dg:chunk/docset:Entry/docset:Amount"
    assert payload["leaf_index"] == 3


def test_node_export_selectors_are_mutually_exclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    main(_ws_args(ws) + ["init"])
    _seed_file_dir(ws, "f0000000001a", pages=1)
    _seed_node_xml(ws, "f0000000001a", "ds000000001a")
    capsys.readouterr()

    with pytest.raises(SystemExit):
        main(
            _ws_args(ws)
            + [
                "node",
                "export",
                "f0000000001a",
                "--docset",
                "ds000000001a",
                "--leaf",
                "1",
                "--child-path",
                "1/0",
            ]
        )


def test_node_prove_detects_tamper(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, "f0000000001a", pages=1)
    _seed_node_xml(ws, "f0000000001a", "ds000000001a")
    capsys.readouterr()  # drain the init payload
    main(
        _ws_args(ws) + ["node", "export", "f0000000001a", "--docset", "ds000000001a", "--leaf", "3"]
    )
    exported = _read_stdout(capsys)

    workspace = Workspace(root=ws)
    workspace.blobs.put_blob(
        layout.dgml_xml_key("ds000000001a", "f0000000001a", "doc"),
        _NODE_XML.replace(b"100", b"999"),
    )

    proof_file = tmp_path / "proof.json"
    proof_file.write_text(json.dumps(exported), encoding="utf-8")
    rc = main(
        _ws_args(ws)
        + ["node", "prove", "f0000000001a", "--docset", "ds000000001a", "--proof", str(proof_file)]
    )
    assert rc == 2  # verified-but-failed, mirrors `dgmlx verify`
    payload = _read_stdout(capsys)
    assert payload["valid"] is False
    assert payload["computed_node_hash"] != payload["expected_node_hash"]


def test_node_prove_malformed_payload_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, "f0000000001a", pages=1)
    _seed_node_xml(ws, "f0000000001a", "ds000000001a")
    capsys.readouterr()

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"root_hash": "x"}), encoding="utf-8")
    rc = main(
        _ws_args(ws)
        + ["node", "prove", "f0000000001a", "--docset", "ds000000001a", "--proof", str(bad)]
    )
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "INVALID_ARGUMENT"


# --- on-chain attestation commands (dgml[chain]) -----------------------------


class _FakeRpc:
    """In-memory stand-in for dgml_chain.EvmRpc for CLI dispatch tests."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.broadcast: list[str] = []

    def get_balance(self, address: str, block: str = "latest") -> int:
        return 5_000_000_000_000_000_000  # 5 "ether"

    def get_transaction_count(self, address: str, block: str = "pending") -> int:
        return 3

    def estimate_gas(self, tx: dict[str, Any]) -> int:
        return 90_000

    def gas_price(self) -> int:
        return 10_000_000_000

    def max_priority_fee(self) -> int:
        return 1_000_000_000

    def send_raw_transaction(self, signed_tx_hex: str) -> str:
        self.broadcast.append(signed_tx_hex)
        return "0xfeed"

    def call(self, to: str, data: str, block: str = "latest") -> str:
        # The stake path resolves the registry NAME to its numeric id via the
        # `registries` view before building addRecord calldata. Return one
        # registry named "myreg" (id 1) with an empty nextKey so the client's
        # pagination stops after a single page.
        from dgml_chain.anchor import _output_types
        from eth_abi import encode  # type: ignore[attr-defined]

        reg = (1, "myreg", "", "", "", "")
        raw = encode(_output_types("registries"), [[reg], (b"", 1)])
        return "0x" + raw.hex()


# A throwaway key (Ganache test vector); the address is derived from it.
_TEST_KEY = "0x4f3edf983ac636a65a842ce7c78d9aa706d3b113bce9c46f30d7d21715b23b1d"


def _test_addr() -> str:
    from eth_account import Account

    return str(Account.from_key(_TEST_KEY).address)


def test_chain_list_and_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    assert main(_ws_args(ws) + ["chain", "list"]) == 0
    names = {c["name"] for c in _read_stdout(capsys)["chains"]}
    assert {"nvnm-testnet", "nvnm-mainnet"} <= names

    assert main(_ws_args(ws) + ["chain", "show", "nvnm-testnet"]) == 0
    show = _read_stdout(capsys)
    assert show["chain_id"] == 787111
    assert show["builtin"] is True


def test_chain_add_and_remove(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    rc = main(
        _ws_args(ws)
        + ["chain", "add", "--name", "local", "--rpc-url", "http://x:8545", "--chain-id", "1337"]
    )
    assert rc == 0
    assert _read_stdout(capsys)["added"]["name"] == "local"

    assert main(_ws_args(ws) + ["chain", "remove", "local"]) == 0
    assert _read_stdout(capsys)["removed"] == "local"

    # Built-ins are protected.
    assert main(_ws_args(ws) + ["chain", "remove", "nvnm-testnet"]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "CHAIN_CONFIG"


def test_wallet_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    monkeypatch.setattr("dgml_core.staking.EvmRpc", _FakeRpc)
    addr = _test_addr()
    rc = main(_ws_args(ws) + ["wallet", "status", "--chain", "nvnm-testnet", "--address", addr])
    assert rc == 0
    out = _read_stdout(capsys)
    assert out["address"] == addr
    assert out["nonce"] == 3
    assert out["funded"] is True


def test_stake_file_dry_run_does_not_broadcast(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    fake = _FakeRpc()
    monkeypatch.setattr("dgml_core.staking.EvmRpc", lambda *a, **k: fake)
    monkeypatch.setattr("dgml_chain.signer.load_key", lambda service="", account="": _TEST_KEY)

    # Stub the local export so the test needs no real PDF/artifacts. Mirrors
    # export_attestation's signature and mode contract: (attestation,
    # attestation_path, archive_path) with exactly one of the paths set —
    # the archive by default, the loose attestation path under --unpacked.
    def _fake_export(  # type: ignore[no-untyped-def]
        ws: Any,
        file_id: str,
        out_dir: Path,
        docset_id: str | None = None,
        *,
        unpacked: bool = False,
    ):
        attestation = SimpleNamespace(root="deadbeef", leaves=[1, 2, 3])
        if unpacked:
            return attestation, out_dir / "META-INF" / "dgml-attestation.xml", None
        return attestation, None, out_dir / "doc.dgmlx"

    monkeypatch.setattr("dgml_core.staking.export_attestation", _fake_export)

    rc = main(
        _ws_args(ws)
        + [
            "stake",
            "file",
            "f00000",
            "--chain",
            "nvnm-testnet",
            "--registry",
            "myreg",
            "--from",
            _test_addr(),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = _read_stdout(capsys)
    assert out["broadcast"] is False
    assert out["checksum"] == "deadbeef"
    assert out["uri"] == "dgmlx://f00000"
    assert out["signed_tx"].startswith("0x")
    assert "unsigned_tx" in out
    # Default is the portable archive: payload carries `dgmlx`, not `attestation`.
    assert out["dgmlx"].endswith("doc.dgmlx")
    assert "attestation" not in out
    # Crucially: nothing was sent to the chain.
    assert fake.broadcast == []


def test_stake_file_unpacked_reports_loose_attestation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()

    fake = _FakeRpc()
    monkeypatch.setattr("dgml_core.staking.EvmRpc", lambda *a, **k: fake)
    monkeypatch.setattr("dgml_chain.signer.load_key", lambda service="", account="": _TEST_KEY)

    def _fake_export(  # type: ignore[no-untyped-def]
        ws: Any,
        file_id: str,
        out_dir: Path,
        docset_id: str | None = None,
        *,
        unpacked: bool = False,
    ):
        attestation = SimpleNamespace(root="deadbeef", leaves=[1, 2, 3])
        if unpacked:
            return attestation, out_dir / "META-INF" / "dgml-attestation.xml", None
        return attestation, None, out_dir / "doc.dgmlx"

    monkeypatch.setattr("dgml_core.staking.export_attestation", _fake_export)

    rc = main(
        _ws_args(ws)
        + [
            "stake",
            "file",
            "f00000",
            "--chain",
            "nvnm-testnet",
            "--registry",
            "myreg",
            "--from",
            _test_addr(),
            "--dry-run",
            "--unpacked",
        ]
    )
    assert rc == 0
    out = _read_stdout(capsys)
    # --unpacked surfaces the loose attestation path and no archive.
    assert out["attestation"].endswith("dgml-attestation.xml")
    assert "dgmlx" not in out


def test_prove_file_missing_record_json_is_structured_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    rc = main(
        _ws_args(ws)
        + ["prove", "file", "--chain", "nvnm-testnet", "--record-json", str(tmp_path / "nope.json")]
    )
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "RECORD_NOT_FOUND"


def test_prove_file_bad_uri_is_invalid_argument(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    rec = tmp_path / "rec.json"
    rec.write_text(json.dumps({"checksum": "ab", "uri": "not-a-uri"}), encoding="utf-8")
    rc = main(
        _ws_args(ws) + ["prove", "file", "--chain", "nvnm-testnet", "--record-json", str(rec)]
    )
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "INVALID_ARGUMENT"


# --- discover ----------------------------------------------------------------

_DISCOVER_XML = (
    b"<?xml version='1.0' encoding='utf-8'?>"
    b'<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"'
    b' xmlns:docset="http://dgml.io/acme-corp/msa#"'
    b' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    b"<docset:IndemnificationClause>"
    b'<docset:IndemnifyingParty dg:origin="2 460 310 1800 355">Vendor</docset:IndemnifyingParty>'
    b'<docset:LiabilityCap xsi:type="decimal" dg:value="500000"'
    b' dg:origin="2 460 410 1800 455">$500,000</docset:LiabilityCap>'
    b'<docset:EffectiveDate xsi:type="date" dg:value="2024-01-01"'
    b' dg:origin="2 998 710 1466 755">January 1, 2024</docset:EffectiveDate>'
    b"</docset:IndemnificationClause>"
    b"<docset:PaymentTerms>"
    b'<docset:InvoiceCycle dg:origin="4 460 139 1800 184">Net 30</docset:InvoiceCycle>'
    b'<docset:LatePaymentPenalty xsi:type="decimal" dg:value="0.015"'
    b' dg:origin="4 460 190 1800 235">1.5% per month</docset:LatePaymentPenalty>'
    b"</docset:PaymentTerms>"
    b"</dg:chunk>"
)


def _seed_discover_xml(ws: Path, file_id: str, docset_id: str) -> None:
    workspace = Workspace(root=ws)
    workspace.docs.put_doc(
        "docsets", docset_id, {"id": docset_id, "name": "T", "description": "", "key_questions": []}
    )
    workspace.blobs.put_blob(layout.dgml_xml_key(docset_id, file_id, "doc"), _DISCOVER_XML)


_DISC_FILE = "f1000000001a"
_DISC_DS = "ds100000001a"


def test_discover_all(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=1)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["discover", _DISC_FILE, "--docset", _DISC_DS])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["file_id"] == _DISC_FILE
    assert payload["docset_id"] == _DISC_DS
    assert payload["filter"] == "All"
    found = {t["tag"] for t in payload["tags"]}
    assert "IndemnificationClause" in found
    assert "LiabilityCap" in found
    assert "PaymentTerms" in found
    assert payload["tag_count"] == len(payload["tags"])


def test_discover_values_filter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=1)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["discover", _DISC_FILE, "--docset", _DISC_DS, "--filter", "values"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["filter"] == "Values"
    found = {t["tag"] for t in payload["tags"]}
    # Leaf-value tags must appear.
    assert "LiabilityCap" in found
    assert "EffectiveDate" in found
    # Container (section-level) tags should not pass Values.
    assert "IndemnificationClause" not in found
    assert "PaymentTerms" not in found


def test_discover_samples_limit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=1)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["discover", _DISC_FILE, "--docset", _DISC_DS, "--samples", "1"])
    assert rc == 0
    payload = _read_stdout(capsys)
    for tag in payload["tags"]:
        assert len(tag["samples"]) <= 1


def test_discover_page_from_origin(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=4)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(
        _ws_args(ws)
        + ["discover", _DISC_FILE, "--docset", _DISC_DS, "--filter", "values", "--full"]
    )
    assert rc == 0
    payload = _read_stdout(capsys)
    # LiabilityCap has dg:origin="2 460 410 1800 455" → page 2
    liab = next((t for t in payload["tags"] if t["tag"] == "LiabilityCap"), None)
    assert liab is not None
    assert liab["samples"][0]["page"] == 2
    # InvoiceCycle has dg:origin="4 460 139 1800 184" → page 4
    inv = next((t for t in payload["tags"] if t["tag"] == "InvoiceCycle"), None)
    assert inv is not None
    assert inv["samples"][0]["page"] == 4


def test_discover_depth_first_addressable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=1)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(
        _ws_args(ws)
        + ["discover", _DISC_FILE, "--docset", _DISC_DS, "--filter", "values", "--full"]
    )
    assert rc == 0
    disc = _read_stdout(capsys)

    # Pick the first sample of the first tag and verify node export accepts its depth_first.
    first_sample = disc["tags"][0]["samples"][0]
    leaf = first_sample["depth_first"]

    rc2 = main(
        _ws_args(ws) + ["node", "export", _DISC_FILE, "--docset", _DISC_DS, "--leaf", str(leaf)]
    )
    assert rc2 == 0
    node_payload = _read_stdout(capsys)
    assert node_payload["leaf_index"] == leaf
    # The XPath from discover should match what node export computes.
    assert node_payload["xpath"] == first_sample["xpath"]


def test_discover_no_xml_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=1)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["discover", _DISC_FILE, "--docset", _DISC_DS])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert "error" not in payload
    assert isinstance(payload["tags"], list)


def test_discover_default_strips_attributes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=1)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["discover", _DISC_FILE, "--docset", _DISC_DS, "--filter", "all"])
    assert rc == 0
    payload = _read_stdout(capsys)
    for tag in payload["tags"]:
        for sample in tag["samples"]:
            xml = sample["xml"]
            assert "=" not in xml.split(">")[0], (
                f"attributes found in default snippet for tag {tag['tag']!r}: {xml[:120]}"
            )


def test_discover_default_shape(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=1)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["discover", _DISC_FILE, "--docset", _DISC_DS, "--filter", "all"])
    assert rc == 0
    payload = _read_stdout(capsys)
    for tag in payload["tags"]:
        assert set(tag.keys()) == {"tag", "count", "samples"}
        for sample in tag["samples"]:
            assert set(sample.keys()) == {"xpath", "xml"}


def test_discover_full_shape(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=1)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(
        _ws_args(ws) + ["discover", _DISC_FILE, "--docset", _DISC_DS, "--filter", "all", "--full"]
    )
    assert rc == 0
    payload = _read_stdout(capsys)
    for tag in payload["tags"]:
        assert set(tag.keys()) == {"tag", "count", "role", "filters", "samples"}
        for sample in tag["samples"]:
            assert set(sample.keys()) == {"depth_first", "xpath", "page", "xml"}


def test_discover_search_tag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=1)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(
        _ws_args(ws)
        + ["discover", _DISC_FILE, "--docset", _DISC_DS, "--filter", "all", "--search", "cycle"]
    )
    assert rc == 0
    payload = _read_stdout(capsys)
    assert all("cycle" in t["tag"].lower() for t in payload["tags"])


def test_discover_search_content(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    _seed_file_dir(ws, _DISC_FILE, pages=1)
    _seed_discover_xml(ws, _DISC_FILE, _DISC_DS)
    capsys.readouterr()

    rc = main(
        _ws_args(ws)
        + [
            "discover",
            _DISC_FILE,
            "--docset",
            _DISC_DS,
            "--filter",
            "all",
            "--search-content",
            "liability",
        ]
    )
    assert rc == 0
    payload = _read_stdout(capsys)
    assert all(any("liability" in s["xml"].lower() for s in t["samples"]) for t in payload["tags"])


def test_discover_cases(tmp_path: Path) -> None:
    from pathlib import Path as _Path

    import dgml_core
    from dgml_core.discovery import run_cases

    cases_path = (
        _Path(dgml_core.__file__).parents[2] / "tests/fixtures/subtree_discovery_cases.json"
    )
    assert cases_path.exists(), f"fixture not found: {cases_path}"
    results = run_cases(cases_path)
    failed = [r for r in results if not r["passed"]]
    assert not failed, "\n".join(f"  FAIL [{r['description']}]: {r['message']}" for r in failed)


def test_load_schema_seed_json_builds_roster_and_parent_map(tmp_path: Path) -> None:
    from dgml.cli import _load_schema_seed

    p = tmp_path / "schema.json"
    p.write_text(
        json.dumps(
            {
                "tags": {
                    "PartyInformation": {
                        "name": "PartyInformation",
                        "role": "party block",
                        "kind": "section",
                    },
                    "PartyAddress": {
                        "name": "PartyAddress",
                        "role": "address",
                        "parent_role": "PartyInformation",
                    },
                    "OrderDate": {"name": "OrderDate", "role": "order date"},
                }
            }
        ),
        encoding="utf-8",
    )
    schema, parent_map = _load_schema_seed(p)
    assert {"PartyInformation", "PartyAddress", "OrderDate"} <= set(schema.tags)
    assert schema.tags["PartyAddress"].role == "address"
    assert schema.tags["PartyInformation"].kind == "section"  # fidelity kept, not flattened
    assert parent_map["PartyAddress"] == "PartyInformation"  # via parent_role
    assert "OrderDate" not in parent_map  # top-level, no container


def test_load_schema_seed_accepts_rnc(tmp_path: Path) -> None:
    """--schema-path also accepts the lossless full-schema.rnc render — the
    `# Field: value` comment contract reconstructs the same roster/parent_map."""
    from dgml.cli import _load_schema_seed

    p = tmp_path / "full-schema.rnc"
    p.write_text(
        "# " + "-" * 20 + "\n"
        '# Description: "party block"\n'
        "# Kind: section\n"
        "PartyInformation = element PartyInformation {\n  common.atts,\n"
        "  mixed { any.docset* }\n}\n\n"
        "# " + "-" * 20 + "\n"
        '# Description: "address"\n'
        "# Kind: field\n"
        "# Parent: PartyInformation\n"
        "PartyAddress = element PartyAddress {\n  common.atts,\n  text\n}\n",
        encoding="utf-8",
    )
    schema, parent_map = _load_schema_seed(p)
    assert {tag.name: tag.role for tag in schema.tags.values()} == {
        "PartyInformation": "party block",
        "PartyAddress": "address",
    }
    assert parent_map == {"PartyAddress": "PartyInformation"}


def test_load_schema_seed_rejects_non_schema_input(tmp_path: Path) -> None:
    """--schema-path accepts only an exported schema (a `tags` map); a flat
    {concept: description} mapping, non-schema text, or a missing file is rejected."""
    from dgml.cli import _load_schema_seed
    from dgml_core.errors import InvalidArgument

    flat = tmp_path / "roster.json"  # old concept_roster shape — no `tags`
    flat.write_text(json.dumps({"BuyerName": "bill-to org"}), encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_seed(flat)

    junk = tmp_path / "seed.txt"  # arbitrary non-schema text
    junk.write_text("concepts:\n  BuyerName: bill-to org\n", encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_seed(junk)

    with pytest.raises(InvalidArgument):
        _load_schema_seed(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# `dgml extraction` — schema-driven value extraction
# ---------------------------------------------------------------------------

_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "definitions": {"grounded_field": {"type": "object"}},
    "properties": {
        "vendor_name": {"$ref": "#/definitions/grounded_field"},
        "liability_cap": {"$ref": "#/definitions/grounded_field"},
    },
}

# The typed field tree the schema-generation LLM now submits (rendered straight
# to RNC by generate_schema — no JSON Schema hop).
_FIELD_TREE = [
    {"name": "vendor_name", "kind": "field", "datatype": "text"},
    {"name": "liability_cap", "kind": "field", "datatype": "decimal"},
    {"name": "effective_date", "kind": "field", "datatype": "date"},
]

_RNC_SCHEMA = """\
namespace dg = "http://dgml.io/ns/dg#"
namespace docset = "http://www.dgml.io/ws/Contracts"

start =
  element dg:chunk {
    (text | VendorName)*
  }

VendorName =
  element docset:VendorName {
    text
  }
"""


def _write_grounded_config(ws: Path) -> None:
    Workspace(root=ws).config_path.write_text(
        dump_toml(
            {
                "grounded": {
                    "schema_model": "anthropic/claude-opus-4-7",
                    "values_model": "gemini/gemini-2.5-pro",
                }
            }
        ),
        encoding="utf-8",
    )


def _new_docset(ws: Path, capsys: pytest.CaptureFixture[str], name: str = "Contracts") -> str:
    main(_ws_args(ws) + ["docset", "create", "--name", name])
    ds_id: str = _read_stdout(capsys)["id"]
    return ds_id


def test_extraction_set_schema_from_json_stores_rnc(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    ds_id = _new_docset(ws, capsys)

    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps(_JSON_SCHEMA), encoding="utf-8")
    rc = main(_ws_args(ws) + ["extraction", "set-schema", ds_id, "--schema-file", str(schema_file)])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["schema_format"] == "rnc"
    assert "element docset:VendorName" in payload["schema"]
    # JSON in → RNC at rest: only extraction-schema.rnc is written, never a .json schema.
    _wsx = Workspace(root=ws)
    assert layout.docset_extraction_schema_key(ds_id).endswith("extraction-schema.rnc")
    assert _wsx.blobs.blob_exists(layout.docset_extraction_schema_key(ds_id))


def test_extraction_set_and_get_schema_rnc(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    ds_id = _new_docset(ws, capsys)

    schema_file = tmp_path / "schema.rnc"
    schema_file.write_text(_RNC_SCHEMA, encoding="utf-8")
    main(_ws_args(ws) + ["extraction", "set-schema", ds_id, "--schema-file", str(schema_file)])
    capsys.readouterr()

    # get-schema rnc returns the stored text verbatim.
    rc = main(_ws_args(ws) + ["extraction", "get-schema", ds_id])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["schema_format"] == "rnc"
    assert payload["schema"] == _RNC_SCHEMA

    # get-schema json returns the engine's extracted_value JSON Schema projection.
    rc = main(_ws_args(ws) + ["extraction", "get-schema", ds_id, "--schema-format", "json"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["schema_format"] == "json"
    assert payload["schema"]["properties"]["VendorName"] == {
        "$ref": "#/definitions/extracted_value"
    }
    assert "extracted_value" in payload["schema"]["definitions"]


def test_extraction_get_schema_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    ds_id = _new_docset(ws, capsys)
    rc = main(_ws_args(ws) + ["extraction", "get-schema", ds_id])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "SCHEMA_NOT_FOUND"


def test_extraction_set_and_get_guidance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    ds_id = _new_docset(ws, capsys)

    guidance_file = tmp_path / "guidance.md"
    guidance_file.write_text("# Rules\nClassify by behavior, not name.\n", encoding="utf-8")
    rc = main(
        _ws_args(ws) + ["extraction", "set-guidance", ds_id, "--guidance-file", str(guidance_file)]
    )
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload == {
        "docset_id": ds_id,
        "guidance": "# Rules\nClassify by behavior, not name.\n",
    }
    # Stored beside the schema as extraction-guidance.md.
    assert (ws / "docsets" / ds_id / "extraction-guidance.md").exists()

    rc = main(_ws_args(ws) + ["extraction", "get-guidance", ds_id])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["guidance"].startswith("# Rules")


def test_extraction_get_guidance_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    ds_id = _new_docset(ws, capsys)
    rc = main(_ws_args(ws) + ["extraction", "get-guidance", ds_id])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "GUIDANCE_NOT_FOUND"


def test_extraction_get_values_json_and_xml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """get-values projects the dg:extraction in the core dgml file to values
    JSON (default) or returns the raw XML with --as xml."""
    from dgml_core.extraction_schema import parse_rnc
    from dgml_core.extraction_xml import standalone_extraction_doc

    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    ds_id = _new_docset(ws, capsys)

    schema_file = tmp_path / "schema.rnc"
    schema_file.write_text(_RNC_SCHEMA, encoding="utf-8")
    main(_ws_args(ws) + ["extraction", "set-schema", ds_id, "--schema-file", str(schema_file)])
    capsys.readouterr()

    # Drop a core <stem>.dgml.xml (with a dg:extraction element) where
    # extraction would have written it — get-values globs *.dgml.xml.
    wsx = Workspace(root=ws)
    vocab = parse_rnc(_RNC_SCHEMA)
    values = {
        "VendorName": {
            "text": "Acme",
            "locations": [{"page_number": 1, "bounding_box": [10, 20, 30, 40]}],
        }
    }
    wsx.blobs.put_blob(
        layout.dgml_xml_key(ds_id, "fileabc", "doc"),
        standalone_extraction_doc(values, vocab=vocab).encode(),
    )

    rc = main(_ws_args(ws) + ["extraction", "get-values", ds_id, "fileabc"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["format"] == "values"
    assert payload["values"]["VendorName"]["text"] == "Acme"
    assert payload["values"]["VendorName"]["locations"][0]["bounding_box"] == [10, 20, 30, 40]

    rc = main(_ws_args(ws) + ["extraction", "get-values", ds_id, "fileabc", "--as", "xml"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["format"] == "xml"
    assert "<dg:extraction>" in payload["xml"]
    assert "<docset:VendorName" in payload["xml"]


def test_extraction_get_values_not_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    ds_id = _new_docset(ws, capsys)
    rc = main(_ws_args(ws) + ["extraction", "get-values", ds_id, "nofile"])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "VALUES_NOT_FOUND"


def test_extraction_generate_schema_no_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    _write_grounded_config(ws)
    ds_id = _new_docset(ws, capsys)
    # No files and no --from-file → NO_FILES, and the LLM is never called.
    with patch("litellm.completion") as mock_completion:
        rc = main(_ws_args(ws) + ["extraction", "generate-schema", ds_id])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "NO_FILES"
    mock_completion.assert_not_called()


def test_extraction_generate_schema_happy_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """generate-schema renders the LLM's typed field tree straight to RNC and
    stores it — datatypes preserved, no JSON Schema hop. The sample PDF is placed
    directly (no ghostscript needed for schema-gen)."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    _write_grounded_config(ws)
    ds_id = _new_docset(ws, capsys)

    # Seed a source PDF where generation expects it (files/<id>/*.pdf), written
    # through the store's staging bridge (zero-copy on LocalStore).
    fid = "filexyz12345"
    _wsx = Workspace(root=ws)
    with _wsx.blobs.staged_write(layout.file_prefix(fid)) as _stage:
        _write_blank_pdf(_stage / "doc.pdf", 1)

    response = _tool_response("submit_schema", {"fields": _FIELD_TREE})
    with patch("litellm.completion", return_value=response):
        rc = main(_ws_args(ws) + ["extraction", "generate-schema", ds_id, "--from-file", fid])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["schema_format"] == "rnc"
    assert "element docset:VendorName" in payload["schema"]
    # Datatypes chosen by the model are carried into the RNC leaves.
    assert "element docset:LiabilityCap {\n    xsd:decimal" in payload["schema"]
    assert "element docset:EffectiveDate {\n    xsd:date" in payload["schema"]
    assert payload["from_file_ids"] == [fid]
    _wsx = Workspace(root=ws)
    assert (
        _wsx.blobs.get_blob(layout.docset_extraction_schema_key(ds_id)).decode("utf-8")
        == payload["schema"]
    )


def test_extraction_extract_schema_not_found(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    _write_grounded_config(ws)
    ds_id = _new_docset(ws, capsys)
    # Schema absent → SchemaNotFound before any LLM call.
    with patch("litellm.completion") as mock_completion:
        rc = main(_ws_args(ws) + ["extraction", "extract", ds_id, "somefile"])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "SCHEMA_NOT_FOUND"
    mock_completion.assert_not_called()


def test_extraction_extract_records_usage_under_debug(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`extraction extract --debug` forwards debug into extract_values, so the
    LLM cost/token row lands in usage.jsonl — same gating as every other LLM
    op and the auto-extract path. Without --debug, no row is written."""
    from dgml_core.usage import read_events

    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    _write_grounded_config(ws)
    ds_id = _new_docset(ws, capsys)
    schema_file = tmp_path / "schema.rnc"
    schema_file.write_text(_RNC_SCHEMA, encoding="utf-8")
    main(_ws_args(ws) + ["extraction", "set-schema", ds_id, "--schema-file", str(schema_file)])
    capsys.readouterr()

    def _fresh_file() -> str:
        _seed_file_dir(ws, "fileusage0001", pages=1)
        return "fileusage0001"

    values = {"VendorName": {"text": "Acme", "locations": []}}  # empty locs → no phase 3
    response = _tool_response("submit_values", {"values": values})
    response._hidden_params = {"response_cost": 0.004}
    response.usage = SimpleNamespace(prompt_tokens=300, completion_tokens=40, total_tokens=340)

    # Without --debug: no usage row.
    fid = _fresh_file()
    with patch("litellm.completion", return_value=response):
        assert main(_ws_args(ws) + ["extraction", "extract", ds_id, fid]) == 0
    assert read_events(Workspace(root=ws)) == []

    # With --debug (global flag, precedes the subcommand): one extract_values row.
    _wsx = Workspace(root=ws)
    _wsx.blobs.delete_blob(layout.dgml_xml_key(ds_id, fid, "doc"))  # re-extract cleanly
    with patch("litellm.completion", return_value=response):
        assert main(_ws_args(ws) + ["--debug", "extraction", "extract", ds_id, fid]) == 0
    events = read_events(Workspace(root=ws))
    assert len(events) == 1
    assert events[0]["operation"] == "extract_values"
    assert events[0]["cost_usd"] == 0.004


def test_docset_add_file_auto_extracts_when_schema_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """docset add-file on a DocSet with an extraction schema fires value
    extraction: the payload gains an `extraction` block and the values land
    as a dg:extraction element in the file's <stem>.dgml.xml."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    _write_grounded_config(ws)
    ds_id = _new_docset(ws, capsys)
    schema_file = tmp_path / "schema.rnc"
    schema_file.write_text(_RNC_SCHEMA, encoding="utf-8")
    main(_ws_args(ws) + ["extraction", "set-schema", ds_id, "--schema-file", str(schema_file)])
    capsys.readouterr()
    _seed_file_dir(ws, "fileauto0001", pages=1)

    # Empty locations → phase 2 has nothing to match, no phase-3 call needed.
    values = {"VendorName": {"text": "Acme", "locations": []}}
    response = _tool_response("submit_values", {"values": values})
    with patch("litellm.completion", return_value=response):
        rc = main(_ws_args(ws) + ["docset", "add-file", "fileauto0001", "--docset", ds_id])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["assigned"] is True
    assert payload["extraction"]["performed"] is True
    assert payload["extraction"]["error"] is None
    assert payload["extraction"]["model"] == "gemini/gemini-2.5-pro"

    _wsx = Workspace(root=ws)
    xml = _wsx.blobs.get_blob(layout.dgml_xml_key(ds_id, "fileauto0001", "doc")).decode("utf-8")
    assert "<dg:extraction>" in xml
    assert "Acme" in xml


def test_docset_add_file_without_schema_has_no_extraction_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No extraction schema on the DocSet → plain assignment, no `extraction`
    key in the payload, and no LLM call."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    ds_id = _new_docset(ws, capsys)
    _seed_file_dir(ws, "fileauto0002", pages=1)

    with patch("litellm.completion") as mock_completion:
        rc = main(_ws_args(ws) + ["docset", "add-file", "fileauto0002", "--docset", ds_id])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["assigned"] is True
    assert "extraction" not in payload
    mock_completion.assert_not_called()


def test_docset_add_file_extraction_failure_is_soft(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An extraction failure (here: no `grounded` config) lands in
    `extraction.error` with exit 0 — the assignment itself stands."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    ds_id = _new_docset(ws, capsys)
    schema_file = tmp_path / "schema.rnc"
    schema_file.write_text(_RNC_SCHEMA, encoding="utf-8")
    main(_ws_args(ws) + ["extraction", "set-schema", ds_id, "--schema-file", str(schema_file)])
    capsys.readouterr()
    _seed_file_dir(ws, "fileauto0003", pages=1)

    rc = main(_ws_args(ws) + ["docset", "add-file", "fileauto0003", "--docset", ds_id])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["assigned"] is True
    assert "GROUNDED_CONFIG_MISSING" in payload["extraction"]["error"]
    # Assignment is on disk despite the failed extraction.
    main(_ws_args(ws) + ["docset", "list-files", ds_id])
    assert "fileauto0003" in _read_stdout(capsys)["file_ids"]


@needs_gs
def test_docset_generate_builds_tree_for_extraction_only_file(
    tmp_path: Path, text_pdf: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Extract-first must not wedge a file: an extraction-only <stem>.dgml.xml
    (no document tree) is NOT treated as already-converted — generate builds
    the tree and carries the existing dg:extraction over into the fresh
    render (spec mode full-extraction)."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    main(_ws_args(ws) + ["file", "add", str(text_pdf)])
    fid = _read_stdout(capsys)["file"]["id"]
    main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
    capsys.readouterr()

    # Simulate a prior `extraction extract` with no tree: extraction-only file.
    _wsx = Workspace(root=ws)
    _wsx.blobs.put_blob(
        layout.dgml_xml_key(did, fid, "with-text"),
        b'<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" xmlns:docset="http://www.dgml.io/ws/T">'
        b"<dg:extraction>"
        b'<docset:VendorName dg:origin="1 10 20 30 40">Acme</docset:VendorName>'
        b"</dg:extraction></dg:chunk>",
    )

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output(
            "with-text.pdf",
            '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><a>the tree</a></dg:chunk>',
        )
        return {}

    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert):
        rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage", "--no-semlinks"])
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    assert payload["summary"] == {"total": 1, "converted": 1, "skipped": 0, "failed": 0}

    final = _wsx.blobs.get_blob(layout.dgml_xml_key(did, fid, "with-text")).decode("utf-8")
    assert "the tree" in final  # document tree generated
    assert "<dg:extraction" in final  # prior extraction carried over
    assert ">Acme</docset:VendorName>" in final
    assert 'dg:origin="1 10 20 30 40"' in final  # grounding survived verbatim


def _two_files_in_docset(
    ws: Path, tmp_path: Path, did: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Add two documents to *did*. They must differ in both name and content:
    generate reports same-named files as a duplicate-filename failure, and
    `file add` dedupes on content hash. The tree each one yields is supplied by
    the fake convert_batch, so the PDFs themselves need only be distinct."""
    for name, text in (("one.pdf", "First Document"), ("two.pdf", "Second Document")):
        pdf = tmp_path / name
        _write_text_pdf(pdf, pages_text=[text])
        main(_ws_args(ws) + ["file", "add", str(pdf)])
        fid = _read_stdout(capsys)["file"]["id"]
        main(_ws_args(ws) + ["docset", "add-file", fid, "--docset", did])
        capsys.readouterr()  # drain, so the next iteration reads only its own JSON


# Two sibling leaves, so a link between them is one apply_plan will keep — a
# leaf linked to its enclosing chunk is its own ancestor and gets dropped.
_SAME_TREE = (
    '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><a>same tree</a><b>and again</b></dg:chunk>'
)


def _fake_convert_emitting(tree: str, *, reverse: bool = False) -> Any:
    """Stand in for convert_batch: hand *tree* to the sink for every input."""

    def fake_convert(paths: Any, *, options: Any, on_output: Any, **_kw: Any) -> dict[str, str]:
        names = [Path(p).name for p in paths]
        for name in reversed(names) if reverse else names:
            on_output(name, tree)
        return {}

    return fake_convert


def _counting_linker(calls: list[str]) -> Any:
    """Stand in for plan_links, recording every XML the model was asked about.

    Only the model call is faked; the real apply_plan writes the links, so these
    tests cover the path that actually runs.
    """

    def fake_plan_links(xml: str, config: Any, *, verify: bool = True) -> list[dict[str, Any]]:
        calls.append(xml)
        # element 0 is the enclosing dg:chunk, 1 is <a>, 2 is <b>
        return [{"subject": 1, "objects": [2], "predicate": "references", "value": ""}]

    return fake_plan_links


@needs_gs
def test_semlink_cache_replays_result_for_an_identical_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two documents that read the same to the model share one cache entry, so
    the second replays the stored links instead of paying for a second call —
    and reports the same `links` count."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)

    calls: list[str] = []
    with (
        patch("dgml_core.generation.convert_batch", side_effect=_fake_convert_emitting(_SAME_TREE)),
        patch("dgml_core.generation.links.plan_links", side_effect=_counting_linker(calls)),
    ):
        rc = main(
            _ws_args(ws) + ["docset", "generate", did, "--no-coverage", "--max-parallel-calls", "1"]
        )
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    converted = [r for r in payload["results"] if r["status"] == "converted"]
    assert len(converted) == 2
    # One model call for two identical trees; both documents report its result.
    assert len(calls) == 1
    assert {r["links"] for r in converted} == {1}
    stored = {Workspace(root=ws).blobs.get_blob(r["output"]).decode("utf-8") for r in converted}
    assert len(stored) == 1  # cache hit wrote the same bytes, not a re-derivation
    assert 'dg:itemprop="references"' in stored.pop()


@needs_gs
def test_link_usage_is_recorded_per_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two documents linked at the same time get one usage row EACH, named.

    `record_usage_for` marks its open aggregation scope on the LLMConfig object,
    so one config shared across the emit pool meant the second document to start
    saw a scope already open, folded its tokens into the first document's row,
    and wrote no row of its own — leaving the pass readable only in aggregate,
    and misattributed even then. The barrier holds both documents inside
    `plan_links` at once, which is the only state in which that could happen.
    """
    from dgml_core.usage import read_events

    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)

    both_proposing = threading.Barrier(2, timeout=10)

    def fake_call(config: Any, **kwargs: Any) -> str:
        if "reviewer" in str(kwargs["system_prompt"]):
            return json.dumps({"verdicts": [{"i": 0, "keep": True}]})
        both_proposing.wait()  # exactly one propose call per document
        return json.dumps(
            {"links": [{"subject": "e0001", "object": "e0002", "predicate": "references"}]}
        )

    def fake_convert(paths: Any, *, options: Any, on_output: Any, **_kw: Any) -> dict[str, str]:
        # Mirror convert_batch's emit pool: the sink runs concurrently per doc.
        names = [Path(p).name for p in paths]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda n: on_output(n, _SAME_TREE), names))
        return {}

    with (
        patch("dgml_core.generation.convert_batch", side_effect=fake_convert),
        patch("dgml_core.llm.call_continued", side_effect=fake_call),
    ):
        rc = main(_ws_args(ws) + ["--debug", "docset", "generate", did, "--no-coverage"])
    assert rc == 0

    rows = [e for e in read_events(Workspace(root=ws)) if e["operation"] == "links"]
    assert len(rows) == 2  # one per document, not one covering both
    assert {r["context"]["doc"] for r in rows} == {"one.pdf", "two.pdf"}


@needs_gs
def test_no_semlink_cache_always_calls_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--no-semlink-cache` opts out: identical trees each pay for their own call."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)

    calls: list[str] = []
    with (
        patch("dgml_core.generation.convert_batch", side_effect=_fake_convert_emitting(_SAME_TREE)),
        patch("dgml_core.generation.links.plan_links", side_effect=_counting_linker(calls)),
    ):
        rc = main(
            _ws_args(ws)
            + [
                "docset",
                "generate",
                did,
                "--no-coverage",
                "--no-semlink-cache",
                "--max-parallel-calls",
                "1",
            ]
        )
    assert rc == 0
    _read_generate_stdout(capsys)
    assert len(calls) == 2


@needs_gs
def test_generate_results_order_is_independent_of_completion_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Documents are converted on a pool, so the sink can fire in any order. The
    `results` array is part of the CLI contract and must stay in queued order."""

    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)

    main(_ws_args(ws) + ["docset", "list-files", did])
    queued = _read_stdout(capsys)["file_ids"]
    assert len(queued) == 2

    # The sink fires in reverse; the payload must still follow the queue.
    with (
        patch(
            "dgml_core.generation.convert_batch",
            side_effect=_fake_convert_emitting(_SAME_TREE, reverse=True),
        ),
        patch("dgml_core.generation.links.plan_links", side_effect=_counting_linker([])),
    ):
        rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    converted = [r["file_id"] for r in payload["results"] if r["status"] == "converted"]
    assert converted == queued


# ------------------------------------------------------- workspace migration


def test_legacy_workspace_migrates_on_first_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The path a real pre-`assignment.json` user hits: an existing workspace
    with directory-shaped assignments upgrades itself on the next command, with
    no migrate step to run and no change to stdout's JSON contract."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    workspace = Workspace(root=ws.resolve())

    rc = main(_ws_args(ws) + ["docset", "create", "--name", "Contracts"])
    assert rc == 0
    docset_id = _read_stdout(capsys)["id"]

    # Fabricate the legacy shape: a file, and an assignment as a bare directory.
    file_id = "aaaaaaaaaaaa"
    workspace.docs.put_doc("files", file_id, {"id": file_id})
    # Legacy bare-marker assignment (pre-assignment.json) — a LocalStore on-disk
    # state the migration upgrades; there is no store-API way to make an empty dir.
    (workspace.docsets_dir / docset_id / "files" / file_id).mkdir(parents=True)
    # Roll the stamp back so the workspace looks like it predates the change.
    stamp_schema_version(workspace, 0)

    rc = main(_ws_args(ws) + ["docset", "list-files", docset_id, "--verbose"])
    assert rc == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["file_ids"] == [file_id]
    assert "upgraded workspace" in captured.err  # announced, but never on stdout
    assert workspace.docs.get_doc("assignments", f"{docset_id}/{file_id}") is not None

    # Second command: already current, nothing announced even under --verbose.
    rc = main(_ws_args(ws) + ["docset", "list-files", docset_id, "--verbose"])
    assert rc == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["file_ids"] == [file_id]
    assert "upgraded workspace" not in captured.err


def test_migration_notice_never_breaks_the_stderr_error_envelope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --verbose, stderr stays a single parseable JSON error envelope
    even when the command both migrated the workspace and then failed."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    workspace = Workspace(root=ws.resolve())
    (workspace.docsets_dir / "ds000000001a" / "files" / "f0000000001a").mkdir(parents=True)
    stamp_schema_version(workspace, 0)
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["docset", "list-files", "nosuchdocset"])
    assert rc != 0
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "DOCSET_NOT_FOUND"
    # the migration still ran
    assert workspace_schema_version(workspace) == WORKSPACE_SCHEMA_VERSION


def test_workspace_create_stamps_schema_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A new workspace is stamped current, so it is never re-scanned."""
    ws = tmp_path / "ws"
    rc = main(_ws_args(ws) + ["workspace", "create", "--name", "W", "--organization", "acme"])
    assert rc == 0
    capsys.readouterr()
    workspace = Workspace(root=ws.resolve())
    assert workspace_schema_version(workspace) == WORKSPACE_SCHEMA_VERSION
    assert pending_migrations(workspace) == []
    # identity survives the stamp
    assert workspace.organization == "acme"
    assert workspace.display_name == "W"


@needs_gs
def test_no_semlink_verify_skips_the_review_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--no-semlink-verify` reaches the library as verify=False, and its result
    is cached separately from a reviewed one (the two are different answers)."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)

    seen: list[bool] = []

    def spy(xml: str, config: Any, *, verify: bool = True) -> list[dict[str, Any]]:
        seen.append(verify)
        return [{"subject": 1, "objects": [0], "predicate": "references", "value": ""}]

    with (
        patch("dgml_core.generation.convert_batch", side_effect=_fake_convert_emitting(_SAME_TREE)),
        patch("dgml_core.generation.links.plan_links", side_effect=spy),
    ):
        rc = main(
            _ws_args(ws)
            + [
                "docset",
                "generate",
                did,
                "--no-coverage",
                "--no-semlink-verify",
                "--max-parallel-calls",
                "1",
            ]
        )
    assert rc == 0
    assert seen == [False]  # one call for two identical trees, review skipped


@needs_gs
def test_link_failure_is_reported_and_keeps_the_dgml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A link-pass failure must not look like "this document has no links": the
    document keeps its unlinked DGML and the entry carries `link_error`."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)

    def boom(xml: str, config: Any, *, verify: bool = True) -> list[dict[str, Any]]:
        raise RuntimeError("429 rate limit exceeded")

    with (
        patch("dgml_core.generation.convert_batch", side_effect=_fake_convert_emitting(_SAME_TREE)),
        patch("dgml_core.generation.links.plan_links", side_effect=boom),
    ):
        rc = main(_ws_args(ws) + ["docset", "generate", did, "--no-coverage"])
    assert rc == 0
    converted = [r for r in _read_generate_stdout(capsys)["results"] if r["status"] == "converted"]
    assert converted
    for entry in converted:
        assert "rate limit" in entry["link_error"]
        assert entry["links"] == 0
        # the tree survived, just unlinked
        stored = Workspace(root=ws).blobs.get_blob(entry["output"]).decode("utf-8")
        assert "same tree" in stored and "dg:itemprop" not in stored


# ------------------------------------------------ the workspace config as a handle


def test_workspace_create_writes_default_local_storage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no --storage, create still writes a binding — the workspace must be
    self-describing even on the bundled default, or nothing distinguishes it from a
    remote workspace whose config went missing."""
    from dgml_core import workspace_config

    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    capsys.readouterr()

    table = workspace_config.read_storage_table(Workspace(root=ws), "default")
    assert table == {"blobs": {"provider": _LOCAL}, "docs": {"provider": _LOCAL}}


def test_workspace_create_does_not_clobber_an_existing_binding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`workspace create` is documented as safe to re-run."""
    from dgml_core import workspace_config

    ws = tmp_path / "ws"
    ws.mkdir()
    _write_ws_config(
        ws, {"storage": {"default": {"blobs": {"provider": _LOCAL, "prefix": "keep"}}}}
    )
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    capsys.readouterr()
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    capsys.readouterr()

    table = workspace_config.read_storage_table(Workspace(root=ws), "default")
    assert table is not None and table["blobs"]["prefix"] == "keep"


def test_create_from_config_copies_the_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--from-config` is a template, not an adopted file: its contents are copied into
    the config the workspace owns, verbatim, and the source is then forgotten."""
    from dgml_core import workspace_config

    seed = tmp_path / "cfg" / "acme.toml"
    seed.parent.mkdir(parents=True)
    seed.write_text(
        f'# authored by hand\n[storage.default.blobs]\nprovider = "{_LOCAL}"\n',
        encoding="utf-8",
    )

    ws = tmp_path / "ws"
    rc = main(
        ["workspace", "create", str(ws), "--organization", "Acme", "--from-config", str(seed)]
    )
    assert rc == 0
    payload = _read_stdout(capsys)

    # The workspace owns its config now — it is not pointing at the seed.
    own = ws / "config.toml"
    assert payload["workspace_config_path"] == str(own)
    assert own.is_file()
    assert "# authored by hand" in own.read_text(encoding="utf-8")
    assert workspace_config.read_identity(Workspace(root=ws)).organization == "Acme"

    # The seed is not tracked: deleting it changes nothing, where an *adopted* config
    # would have taken the workspace with it.
    seed.unlink()
    assert main(_ws_args(ws) + ["status"]) == 0


def test_create_from_config_must_exist(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "workspace",
            "create",
            str(tmp_path / "ws"),
            "--organization",
            "Acme",
            "--from-config",
            str(tmp_path / "missing.toml"),
        ]
    )
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "INVALID_ARGUMENT"


def test_create_from_config_refuses_a_workspaces_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`[workspaces]` selects the machine's store of workspaces and is read only from
    the user config, so it would be silently inert in a seed. A config that looks like it
    redirects where workspaces are listed but does not is worse than an error."""
    seed = tmp_path / "seed.toml"
    seed.write_text('[workspaces]\nprovider = "x:Y"\n', encoding="utf-8")
    rc = main(
        [
            "workspace",
            "create",
            str(tmp_path / "ws"),
            "--organization",
            "Acme",
            "--from-config",
            str(seed),
        ]
    )
    assert rc == 1
    error = _read_stderr(capsys)["error"]
    assert error["code"] == "INVALID_ARGUMENT"
    assert "[workspaces]" in error["message"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--workspace-config", "/tmp/x.toml", "status"],
        ["status", "--workspace-config", "/tmp/x.toml"],
    ],
)
def test_workspace_config_flag_is_retired(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Still declared, in both positions, so an existing caller gets the JSON error
    envelope naming the replacement rather than an argparse usage dump."""
    assert main(argv) == 1
    error = _read_stderr(capsys)["error"]
    assert error["code"] == "INVALID_ARGUMENT"
    assert "--from-config" in error["message"]


def test_dgml_config_env_var_is_retired(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """It only ever worked as an address because the machine index remembered the
    location and handed it back; with nothing recording that, the variable would have to
    be set forever or the workspace would appear to have no config at all."""
    monkeypatch.setenv("DGML_CONFIG", "/tmp/x.toml")
    assert main(["status"]) == 1
    error = _read_stderr(capsys)["error"]
    assert error["code"] == "INVALID_ARGUMENT"
    assert "--from-config" in error["message"]


def test_cluster_config_flag_still_means_the_clustering_preset() -> None:
    """`dgml cluster --config` predates the global flag. A global `--config` would
    collide with it on the shared parent parser and take `dgml --help` down at
    construction time, which is why the global one is spelled --workspace-config."""
    import argparse

    from dgml.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["cluster", "--config", "light", "--workspace-config", "/tmp/x.toml"])
    assert args.config == "light"
    assert args.workspace_config == Path("/tmp/x.toml")
    assert isinstance(parser, argparse.ArgumentParser)


def test_a_deleted_workspace_config_names_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The config names the workspace's storage backend and cannot be reconstructed, so
    losing it is a hard, specific error rather than a silent fall-through to local disk."""
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    capsys.readouterr()

    (ws / "config.toml").unlink()
    assert main(_ws_args(ws) + ["status"]) == 1
    message = _read_stderr(capsys)["error"]["message"]
    assert str(ws / "config.toml") in message
    assert "STORAGE_CONFIG_INVALID" not in message  # the code is a sibling field


def test_an_unlisted_id_names_the_store_it_was_looked_for_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A workspace dropped from the store is reported against the store, not against a
    path that was never a file. Note this is caught during *resolution* — an id-shaped
    argument the store does not hold cannot become a workspace — which is why the
    equivalent branch of the missing-config message is defensive rather than a path a
    user reaches."""
    main(["workspace", "create", "--organization", "Acme"])
    wid = _read_stdout(capsys)["workspace_id"]

    # Unlisting leaves the workspace's docsets/ and files/ in place — `delete` drops a
    # listing entry, not a corpus.
    store = default_workspaces_store()
    assert store.delete(wid) is True

    assert main(["--workspace", wid, "status"]) == 1
    error = _read_stderr(capsys)["error"]
    assert error["code"] == "WORKSPACE_NOT_FOUND"
    assert store.label() in error["message"]
    assert wid in error["message"]


def test_from_config_binds_to_the_named_service(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--from-config` and `--storage` are orthogonal: the first supplies a config to
    start from, the second says which `[storage.<name>]` table in it to bind to.

    Regression: they were briefly mutually exclusive, so a config declaring only
    `[storage.acme]` bound to the undeclared `default` service and fell through to the
    bundled local store — building the workspace somewhere the caller never asked for.
    """
    from dgml_core import workspace_config
    from dgml_core.storage_resolve import resolve_store_configs

    seed = tmp_path / "cfg" / "acme.toml"
    seed.parent.mkdir(parents=True)
    seed.write_text(f'[storage.acme.blobs]\nprovider = "{_LOCAL}"\nprefix = "acme"\n')

    ws = tmp_path / "ws"
    rc = main(
        [
            "workspace",
            "create",
            str(ws),
            "--organization",
            "Acme",
            "--from-config",
            str(seed),
            "--storage",
            "acme",
        ]
    )
    assert rc == 0
    assert _read_stdout(capsys)["storage_service"] == "acme"

    workspace = Workspace(root=ws)
    assert workspace_config.read_identity(workspace).storage_service == "acme"
    blob_cfg, _ = resolve_store_configs(workspace)
    assert blob_cfg.options["prefix"] == "acme"


def test_create_seals_the_service_it_actually_bound_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The seal must be computed *after* `storage_service` is written.

    Regression: resolution reads that pointer to choose a table, so computing the
    fingerprint before writing it sealed the workspace to the local default — and the
    very next command failed STORAGE_BACKEND_MISMATCH on a workspace that had just
    been created successfully.
    """
    seed = tmp_path / "acme.toml"
    seed.write_text(f'[storage.acme.blobs]\nprovider = "{_LOCAL}"\nprefix = "acme"\n')

    ws = tmp_path / "ws"
    main(
        [
            "workspace",
            "create",
            str(ws),
            "--organization",
            "Acme",
            "--from-config",
            str(seed),
            "--storage",
            "acme",
        ]
    )
    capsys.readouterr()

    # The workspace must be usable immediately, with no reseal.
    assert main(_ws_args(ws) + ["status"]) == 0


def test_create_seals_a_listed_workspace_it_can_open_immediately(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same ordering requirement for a workspace whose config lives in the store:
    the binding has to be written there before the seal is computed, or the very next
    command reports drift on a workspace that was just created."""
    main(["workspace", "create", "--organization", "Acme"])
    wid = _read_stdout(capsys)["workspace_id"]
    assert main(["--workspace", wid, "status"]) == 0


def test_from_config_without_the_selected_service_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Omitting --storage for a config that names only `acme` must fail loudly rather
    than silently creating the workspace on local disk. The message names what the
    config does declare, since that is the flag value the caller needs."""
    seed = tmp_path / "acme.toml"
    seed.write_text(f'[storage.acme.blobs]\nprovider = "{_LOCAL}"\n')

    rc = main(
        [
            "workspace",
            "create",
            str(tmp_path / "ws"),
            "--organization",
            "Acme",
            "--from-config",
            str(seed),
        ]
    )
    assert rc == 1
    error = _read_stderr(capsys)["error"]
    assert error["code"] == "INVALID_ARGUMENT"
    assert "[storage.acme]" in error["message"]
    assert not (tmp_path / "ws" / "docsets").exists()  # nothing built


def test_create_is_idempotent_and_preserves_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`workspace create` is documented as safe to re-run. That means it must reuse the
    identity its config already records, not mint a fresh one.

    Regression: it minted unconditionally, so re-running forked `workspace_id` and left
    two index rows for one directory — and run on a second machine against a shared
    config it changed the whole organization's workspace identity, including the
    `workspace` record in the remote doc store.
    """
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme", "--name", "Acme Contracts"])
    first = _read_stdout(capsys)

    main(["workspace", "create", str(ws), "--organization", "Acme"])
    second = _read_stdout(capsys)

    assert second["workspace_id"] == first["workspace_id"]
    # --name omitted on the re-run must not rename the workspace after its directory.
    assert second["name"] == "Acme Contracts"


def test_create_reuses_a_listed_workspaces_identity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The multi-machine flow, which is what this whole mechanism is for: pointed at the
    same store of workspaces, a second machine addresses the workspace by id and gets the
    recorded identity rather than forking it.

    This replaces the old "hand the next developer your config.toml file" flow. That
    worked by *adopting* a shared file, which meant the identity of the org's workspace
    lived in whatever copy of that file each machine happened to have."""
    main(["workspace", "create", "--organization", "Acme", "--name", "Acme Contracts"])
    original = _read_stdout(capsys)
    wid = original["workspace_id"]

    # As a second machine would: same store, addressed by id, no --name and no config
    # file passed between machines.
    main(["--workspace", wid, "workspace", "create", "--organization", "Acme"])
    reconnected = _read_stdout(capsys)

    assert reconnected["workspace_id"] == wid
    assert reconnected["name"] == "Acme Contracts"
    assert reconnected["listed"] is True


# --------------------------------------------------------- import / delete


def _legacy_index(rows: dict[str, dict[str, object]]) -> Path:
    """A legacy ``workspaces.json``, as an older dgml would have left it."""
    from dgml_core import registry

    path = registry.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_import_sweeps_the_legacy_index_without_moving_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The migration path off the per-machine index, and the reason it is a command
    rather than automatic: adopting a machine's old workspaces is the owner's decision."""
    a, b = tmp_path / "a", tmp_path / "b"
    main(["workspace", "create", str(a), "--organization", "Acme", "--name", "A"])
    id_a = _read_stdout(capsys)["workspace_id"]
    main(["workspace", "create", str(b), "--organization", "Beta", "--name", "B"])
    id_b = _read_stdout(capsys)["workspace_id"]
    _legacy_index({id_a: {"name": "A", "root": str(a)}, id_b: {"name": "B", "root": str(b)}})

    assert main(["workspace", "import"]) == 0
    payload = _read_stdout(capsys)
    assert {r["workspace_id"] for r in payload["imported"]} == {id_a, id_b}
    assert payload["failed"] == []
    # The legacy file stays, so a half-finished sweep can simply be repeated.
    assert payload["source_removed"] is False

    # Both are listed now, and neither corpus moved.
    main(["workspace", "list"])
    assert {r["workspace_id"] for r in _read_stdout(capsys)["workspaces"]} == {id_a, id_b}
    assert (a / "config.toml").is_file()
    assert (b / "docsets").is_dir()

    # ...and each opens by id, from the directory it was already in.
    assert main(["--workspace", id_a, "status"]) == 0
    assert Path(_read_stdout(capsys)["workspace"]) == a.resolve()


def test_import_records_where_the_data_already_is_without_resealing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`workspace_path` pins the existing directory, and because it is excluded from the
    storage fingerprint the workspace is usable immediately — no reseal, and no
    STORAGE_BACKEND_MISMATCH on the very next command."""
    from dgml_core import workspace_config

    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    created = _read_stdout(capsys)
    wid, sealed = created["workspace_id"], created["storage_fingerprint"]

    assert main(["workspace", "import", str(ws)]) == 0
    row = _read_stdout(capsys)["imported"][0]
    assert row["workspace_path_recorded"] is True
    assert row["moved"] is False

    imported = Workspace.resolve(wid)
    assert imported.root == ws.resolve()
    table = workspace_config.read_storage_table(imported, "default")
    assert table is not None and table["workspace_path"] == str(ws.resolve())
    # The seal is untouched: a workspace that has not moved is the same workspace on
    # the same backend.
    assert workspace_config.read_identity(imported).storage_fingerprint == sealed
    assert main(["--workspace", wid, "status"]) == 0


def test_import_move_relocates_into_the_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Opt-in, because relocating a corpus of page images is not something to do on the
    caller's behalf."""
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    wid = _read_stdout(capsys)["workspace_id"]

    assert main(["workspace", "import", "--move", str(ws)]) == 0
    assert _read_stdout(capsys)["imported"][0]["moved"] is True
    assert not ws.exists()
    assert Workspace.resolve(wid).root == default_workspaces_store().workspace_root(wid)
    assert main(["--workspace", wid, "status"]) == 0


def test_import_is_idempotent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    wid = _read_stdout(capsys)["workspace_id"]
    main(["workspace", "import", str(ws)])
    capsys.readouterr()

    assert main(["workspace", "import", str(ws)]) == 0
    payload = _read_stdout(capsys)
    assert payload["imported"] == []
    assert payload["skipped"][0]["workspace_id"] == wid


def test_import_on_conflict_fail_stops(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    capsys.readouterr()
    main(["workspace", "import", str(ws)])
    capsys.readouterr()

    assert main(["workspace", "import", "--on-conflict", "fail", str(ws)]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "CONFLICT"


def test_import_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    wid = _read_stdout(capsys)["workspace_id"]

    assert main(["workspace", "import", "--dry-run", str(ws)]) == 0
    payload = _read_stdout(capsys)
    assert payload["dry_run"] is True
    assert payload["imported"][0]["status"] == "would-import"
    assert not default_workspaces_store().exists(wid)


def test_import_reports_a_bad_row_and_keeps_going(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One dead directory in an old index must not strand every other workspace in it."""
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    wid = _read_stdout(capsys)["workspace_id"]
    _legacy_index(
        {
            "ws_goneaaaaaaaaaaaa": {"name": "Gone", "root": str(tmp_path / "vanished")},
            wid: {"name": "Live", "root": str(ws)},
        }
    )

    assert main(["workspace", "import"]) == 2  # partial success
    payload = _read_stdout(capsys)
    assert [r["workspace_id"] for r in payload["imported"]] == [wid]
    assert payload["failed"][0]["root"] == str(tmp_path / "vanished")
    assert "does not exist" in payload["failed"][0]["reason"]


def test_import_refuses_a_directory_with_no_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory with the right shape is not a workspace. Import can reconstruct a
    missing config — assuming local disk when nothing recorded a binding — but it cannot
    invent an *identity*: with no `workspace.json` and no legacy row there is nothing to
    import this as, and minting an id would adopt an arbitrary directory as a workspace."""
    bare = tmp_path / "bare"
    (bare / "docsets").mkdir(parents=True)
    (bare / "files").mkdir()

    assert main(["workspace", "import", str(bare)]) == 2
    reason = _read_stdout(capsys)["failed"][0]["reason"]
    assert "no workspace identity" in reason
    assert "workspace.json" in reason
    assert "dgml workspace create" in reason  # names the way forward


def test_a_workspaces_table_in_a_workspace_config_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`[workspaces]` selects the machine's store of workspaces and is read only from the
    user config, so in a workspace's own config it does nothing. Warn rather than stay
    silent: it *looks* like it redirects where workspaces are listed, and a user who put
    it in the wrong file has no other way to find out it is inert."""
    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    capsys.readouterr()
    with (ws / "config.toml").open("a", encoding="utf-8") as fh:
        fh.write('\n[workspaces]\nprovider = "dgml_storage_mongo:MongoWorkspacesStore"\n')

    assert main(_ws_args(ws) + ["status"]) == 0
    captured = capsys.readouterr()
    assert "[workspaces]" in captured.err
    assert "ignored" in captured.err
    # The advisory must not corrupt the JSON contract on stdout.
    assert json.loads(captured.out)["organization"] == "Acme"


def test_the_workspaces_table_warning_is_once_per_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Several commands read the config more than once; repeating the paragraph each
    time is noise rather than emphasis."""
    import dgml.cli as cli

    ws = tmp_path / "ws"
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    with (ws / "config.toml").open("a", encoding="utf-8") as fh:
        fh.write('\n[workspaces]\nprovider = "x:Y"\n')

    cli._WARNED_WORKSPACES_TABLE = False
    main(_ws_args(ws) + ["status"])
    first = capsys.readouterr().err
    main(_ws_args(ws) + ["status"])
    second = capsys.readouterr().err
    assert "[workspaces]" in first
    assert "[workspaces]" not in second


def test_re_running_create_keeps_the_recorded_storage_service(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`create` is documented as safe to re-run, so it must not silently rebind where
    the workspace's data goes.

    Regression, found by driving the documented workflow against real backends: a
    workspace created with `--storage acme` (S3 blobs + Mongo docs) and then re-created
    without `--storage` was rebound to the local-disk `default` service **and re-sealed**,
    so the next `file add` wrote to local disk while the existing corpus sat in S3. No
    error, no drift warning — the re-seal made it look intentional.
    """
    from dgml_core import workspace_config

    seed = tmp_path / "seed.toml"
    seed.write_text(f'[storage.acme.blobs]\nprovider = "{_LOCAL}"\nprefix = "acme"\n')
    ws = tmp_path / "ws"
    main(
        [
            "workspace",
            "create",
            str(ws),
            "--organization",
            "Acme",
            "--storage",
            "acme",
            "--from-config",
            str(seed),
        ]
    )
    sealed = _read_stdout(capsys)["storage_fingerprint"]

    # The re-run the docs promise is safe: no --storage.
    main(["workspace", "create", str(ws), "--organization", "Acme"])
    second = _read_stdout(capsys)

    assert second["storage_service"] == "acme"
    assert second["storage_fingerprint"] == sealed
    identity = workspace_config.read_identity(Workspace(root=ws))
    assert identity.storage_service == "acme"


def test_changing_the_storage_service_warns_loudly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rebinding is allowed — it is how a workspace moves backend — but it changes where
    data goes, and dgml does not move what is already written. That has to be said on
    stderr, not left for the user to discover as missing files."""
    seed = tmp_path / "seed.toml"
    seed.write_text(f'[storage.acme.blobs]\nprovider = "{_LOCAL}"\n')
    ws = tmp_path / "ws"
    main(
        [
            "workspace",
            "create",
            str(ws),
            "--organization",
            "Acme",
            "--storage",
            "acme",
            "--from-config",
            str(seed),
        ]
    )
    capsys.readouterr()

    assert (
        main(["workspace", "create", str(ws), "--organization", "Acme", "--storage", "default"])
        == 0
    )
    captured = capsys.readouterr()
    assert "--storage 'default'" in captured.err
    assert "'acme'" in captured.err
    assert "does not move data" in captured.err
    assert json.loads(captured.out)["storage_service"] == "default"


# ------------------------------------------- the two ways of addressing a workspace
#
# Every test here chdirs into tmp_path. Without that they would resolve
# `Path.cwd() / "dgml-workspace"` to the pytest invocation directory — and there is a
# gitignored `dgml-workspace/` at the repo root, which would silently absorb the bare
# commands below and make these tests pass for the wrong reason. That masking is why the
# gap these cover shipped in the first place.


def test_bare_create_then_a_bare_command_fails_with_a_remedy_that_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`create` with no path puts the workspace in the store, so a bare next command
    resolves `./dgml-workspace` and finds nothing. That is expected — what must not
    happen is the old advice, "run 'dgml workspace create'", which is the very
    invocation that just failed to produce a workspace here: following it makes a
    second workspace elsewhere and leaves this command failing identically.
    """
    monkeypatch.chdir(tmp_path)
    assert main(["workspace", "create", "--organization", "Acme"]) == 0
    capsys.readouterr()

    assert main(["status"]) == 1
    error = _read_stderr(capsys)["error"]
    assert error["code"] == "WORKSPACE_NOT_INITIALIZED"
    message = error["message"]
    # Both remedies present, and each one runnable as printed.
    assert f"dgml workspace create {tmp_path / 'dgml-workspace'}" in message
    assert "dgml workspace list" in message
    # ...and it says why dgml looked there, since the caller named no path.
    assert "$DGML_HOME" in message


def test_create_with_a_path_then_a_bare_command_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The path-shaped flow, which is what the `./dgml-workspace` fallback exists for.
    Nothing covered it end to end before, so the fallback could have rotted unnoticed."""
    monkeypatch.chdir(tmp_path)
    assert main(["workspace", "create", "./dgml-workspace", "--organization", "Acme"]) == 0
    capsys.readouterr()

    assert main(["status"]) == 0
    assert _read_stdout(capsys)["organization"] == "Acme"


def test_a_listed_workspace_is_addressable_both_documented_ways(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two remedies `create` prints. Asserted rather than assumed, because they are
    the entire answer to "I ran create and now nothing works"."""
    monkeypatch.chdir(tmp_path)
    main(["workspace", "create", "--organization", "Acme", "--name", "Listed"])
    wid = _read_stdout(capsys)["workspace_id"]

    assert main(["--workspace", wid, "status"]) == 0
    assert _read_stdout(capsys)["name"] == "Listed"

    monkeypatch.setenv("DGML_HOME", wid)
    assert main(["status"]) == 0
    assert _read_stdout(capsys)["name"] == "Listed"


def test_create_tells_you_how_to_address_a_listed_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    main(["init"])  # the documented first step, so next_action is free for this hint
    capsys.readouterr()
    main(["workspace", "create", "--organization", "Acme"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    wid = payload["workspace_id"]

    assert wid in payload["next_action"]
    assert "--workspace" in payload["next_action"]
    # On stderr too: this is the likeliest place to get stuck, and one field among
    # thirteen in a JSON blob is not where a human looks.
    assert f"--workspace {wid}" in captured.err
    assert f"export DGML_HOME={wid}" in captured.err


def test_a_missing_user_config_keeps_the_more_urgent_next_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Creating before `dgml init` leaves two things to say. The payload's single
    `next_action` goes to the one that blocks every LLM command, not just this one —
    while the addressing hint still reaches the user on stderr, so neither is lost."""
    monkeypatch.chdir(tmp_path)
    main(["workspace", "create", "--organization", "Acme"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert "dgml init" in payload["next_action"]
    assert f"export DGML_HOME={payload['workspace_id']}" in captured.err


def test_create_does_not_hint_for_a_workspace_addressed_by_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A detached workspace's handle is its path, which the caller just typed. No
    addressing hint, and no stderr noise about ids."""
    monkeypatch.chdir(tmp_path)
    main(["workspace", "create", "./ws", "--organization", "Acme"])
    captured = capsys.readouterr()
    assert "--workspace" not in json.loads(captured.out).get("next_action", "")
    assert "export DGML_HOME" not in captured.err
    assert "store of workspaces" not in captured.err


def test_create_never_sets_dgml_home_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It prints the export line; it must not perform it. The caller's environment is
    theirs, and a command that quietly repoints $DGML_HOME would change what every
    later command in that shell resolves."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DGML_HOME", raising=False)
    main(["workspace", "create", "--organization", "Acme"])
    assert "DGML_HOME" not in os.environ


def test_status_reports_where_the_config_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "Where is this workspace's config?" had no answer for an existing workspace —
    only `create` said — so anything wanting to edit one had to reconstruct the path.
    It cannot be reconstructed: a listed workspace's config is in the store, not under
    the data root."""
    monkeypatch.chdir(tmp_path)
    main(["workspace", "create", "--organization", "Acme"])
    wid = _read_stdout(capsys)["workspace_id"]

    main(["--workspace", wid, "status"])
    status = _read_stdout(capsys)
    assert Path(status["workspace_config_path"]).is_file()
    assert status["config_location"] == status["workspace_config_path"]
    # It is genuinely the config, not merely a path that exists.
    assert "[workspace]" in Path(status["workspace_config_path"]).read_text(encoding="utf-8")


# ----------------------------------------------- importing a workspace with no config


def _legacy_workspace(
    root: Path, *, with_storage_snapshot: bool, workspace_id: str | None = None
) -> str:
    """A workspace as an older dgml left it: initialized, no `config.toml`, its binding
    (or not) recorded only in the per-machine index. Returns the id it was given.

    The id is **minted**, not written as a literal: a hand-typed id is easy to get subtly
    wrong (16 characters from [a-z2-7] exactly), and one character off silently produces a
    workspace nothing can address — which is the very failure the malformed-id test below
    covers. Pass ``workspace_id`` only to construct that bad case deliberately."""
    from dgml_core import registry
    from dgml_core.storage import write_json_atomic
    from dgml_core.workspace_id import new_workspace_id

    workspace_id = workspace_id or new_workspace_id()

    ws = Workspace(root=root)
    ws.init()
    ws.write_meta(name="Legacy", organization="Acme", workspace_id=workspace_id)
    ws.config_path.unlink(missing_ok=True)

    row: dict[str, Any] = {
        "name": "Legacy",
        "organization": "Acme",
        "root": str(root),
        "created_at": "2026-01-01T00:00:00Z",
        "schema_version": 1,
    }
    if with_storage_snapshot:
        row["storage"] = {
            "blobs": {"provider": _LOCAL},
            "docs": {"provider": _LOCAL},
        }
        row["storage_service"] = "default"
    path = registry.registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, {workspace_id: row})
    return workspace_id


def test_import_reconstructs_a_config_from_the_legacy_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The era just before `config.toml` was required kept the binding in the index, so
    import can put back exactly the backend the workspace is already on."""
    root = tmp_path / "legacy"
    wid = _legacy_workspace(root, with_storage_snapshot=True)

    assert main(["workspace", "import"]) == 0
    (imported,) = _read_stdout(capsys)["imported"]
    assert imported["workspace_id"] == wid
    assert imported["assumed_local_storage"] is False
    assert "Migrated by dgml" in (root / "config.toml").read_text(encoding="utf-8")


def test_import_assumes_local_when_nothing_recorded_a_binding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No config and no snapshot: either the workspace predates the binding being
    recorded, or its config was deleted — indistinguishable, and in the second case the
    binding is unrecoverable anyway, so refusing preserves nothing. Local disk was the
    only backend such a workspace could have used.

    The assumption is reported, in the payload and on stderr, because only the caller can
    confirm where the data actually is."""
    root = tmp_path / "ancient"
    wid = _legacy_workspace(root, with_storage_snapshot=False)

    assert main(["workspace", "import"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    (imported,) = payload["imported"]
    assert imported["assumed_local_storage"] is True
    assert "local disk was assumed" in captured.err
    assert "reseal" in captured.err
    config = (root / "config.toml").read_text(encoding="utf-8")
    assert "local disk was assumed" in config  # the banner says so in the file too
    assert _LOCAL in config

    # And it is genuinely usable afterwards, by id, from anywhere.
    assert main(["--workspace", wid, "status"]) == 0
    assert _read_stdout(capsys)["organization"] == "Acme"


def test_a_plain_command_never_invents_a_binding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`import` may assume local disk; the per-command path must not. Inventing one on an
    ordinary `file add` would seal the workspace to local disk and write the file there,
    so a workspace whose config was deleted would quietly start a second corpus while
    reporting success."""
    root = tmp_path / "ancient"
    _legacy_workspace(root, with_storage_snapshot=False)

    assert main(_ws_args(root) + ["status"]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "STORAGE_CONFIG_INVALID"
    assert not (root / "config.toml").exists()


def test_import_refuses_a_malformed_workspace_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An id that is not `ws_` + 16 base32 chars addresses nothing: the local backend
    filters its folders by that same test, so importing would write a config into a
    directory `workspace list` never looks at and `--workspace <id>` never resolves.
    Reporting "imported" for that would be a silent no-op dressed as success.

    dgml's generator only ever emits well-formed ids, so this is hand-edited — hence a
    refusal that names both places to correct, and leaves the legacy index in place."""
    root = tmp_path / "handedited"
    _legacy_workspace(root, workspace_id="ws_tooshort", with_storage_snapshot=True)  # 9 chars

    assert main(["workspace", "import"]) == 2
    payload = _read_stdout(capsys)
    (failed,) = payload["failed"]
    assert "not well-formed" in failed["reason"]
    assert "workspace.json" in failed["reason"]
    # The index survives, so the caller can inspect and fix it.
    assert payload["source_removed"] is False
    from dgml_core import registry

    assert registry.registry_path().is_file()
    assert payload["imported"] == []
