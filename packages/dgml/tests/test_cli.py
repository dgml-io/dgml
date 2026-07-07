from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core.run_clustering import DocPrediction
from dgml_core.storage import Workspace

from .conftest import (
    _write_blank_pdf,
    _write_text_pdf,
    needs_gs,
    write_classification_config,
)


def _ws_args(ws: Path) -> list[str]:
    return ["--workspace", str(ws)]


def _init_ws(ws: Path) -> None:
    """Bootstrap a usable workspace for tests the way `dgml workspace create`
    does — create ``docsets/`` and ``files/`` — without emitting CLI stdout that
    would interleave with the output under test. Config is written per-test when
    a command needs it (e.g. ``write_classification_config``)."""
    Workspace(root=ws.resolve()).init()


def _dp(cluster_name: str, confidence: float | None = None) -> DocPrediction:
    """Shorthand for a mocked ``run_clustering_detailed`` outcome."""
    return DocPrediction(cluster_name=cluster_name, confidence=confidence)


def _read_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    out = capsys.readouterr().out
    return json.loads(out)  # type: ignore[no-any-return]


def _read_stderr(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    err = capsys.readouterr().err
    return json.loads(err)  # type: ignore[no-any-return]


def test_init_creates_shared_local_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`dgml init` is config-only: it creates the peer local_config.json and does
    NOT create the workspace dirs or any workspace config.json. A second init is
    a no-op."""
    ws = tmp_path / "ws"
    rc = main(_ws_args(ws) + ["init"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["local_config_created"] is True
    assert payload["refreshed"] is False
    assert "next_action" in payload
    local_config = tmp_path / "local_config.json"
    assert Path(payload["local_config_path"]) == local_config
    assert local_config.exists()
    # No workspace was created.
    assert not (ws / "docsets").exists()
    assert not (ws / "files").exists()
    assert not (ws / "config.json").exists()

    # Second init is a no-op.
    rc = main(_ws_args(ws) + ["init"])
    assert rc == 0
    assert _read_stdout(capsys)["local_config_created"] is False


def test_init_refresh_overwrites_local_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    local_config = tmp_path / "local_config.json"
    local_config.write_text("{}\n", encoding="utf-8")

    rc = main(_ws_args(ws) + ["init", "--refresh"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["refreshed"] is True
    assert payload["local_config_created"] is False
    # Overwritten from the bundled template (models restored), with a backup.
    assert "classification" in local_config.read_text(encoding="utf-8")
    assert (tmp_path / "local_config.json.bak").exists()


def test_workspace_create_from_local_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`dgml workspace create` creates docsets/ + files/ and copies the peer
    local_config.json (comments intact) to <workspace>/config.json. A second
    run does not clobber; --force overwrites."""
    ws = tmp_path / "ws"
    main(_ws_args(ws) + ["init"])  # seed the shared local_config.json
    capsys.readouterr()

    rc = main(_ws_args(ws) + ["workspace", "create", "--organization", "Acme"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["initialized"] is True
    assert payload["config_written"] is True
    assert payload["organization"] == "Acme"
    assert payload["name"] == "ws"  # defaults to the workspace directory name
    # local_config.json already existed (init ran), so create didn't seed it.
    assert payload["local_config_created"] is False
    assert Path(payload["config_path"]) == ws / "config.json"
    assert (ws / "docsets").is_dir()
    assert (ws / "files").is_dir()
    # Identity was persisted for later namespace generation.
    meta = json.loads((ws / "workspace.json").read_text(encoding="utf-8"))
    assert meta == {"name": "ws", "organization": "Acme"}
    config_text = (ws / "config.json").read_text(encoding="utf-8")
    assert "//" in config_text  # comments survived the verbatim copy

    # An explicit --name overrides the directory-name default.
    rc = main(_ws_args(ws) + ["workspace", "create", "--organization", "Acme", "--name", "My WS"])
    assert rc == 0
    assert _read_stdout(capsys)["name"] == "My WS"
    assert ws.joinpath("workspace.json").read_text(encoding="utf-8")

    # Second run does not clobber the existing config.json.
    rc = main(_ws_args(ws) + ["workspace", "create", "--organization", "Acme"])
    assert rc == 0
    assert _read_stdout(capsys)["config_written"] is False

    # --force re-syncs the shared config into this workspace.
    (tmp_path / "local_config.json").write_text('{"grounded": {}}\n', encoding="utf-8")
    rc = main(_ws_args(ws) + ["workspace", "create", "--organization", "Acme", "--force"])
    assert rc == 0
    assert _read_stdout(capsys)["config_written"] is True
    assert (ws / "config.json").read_text(encoding="utf-8") == '{"grounded": {}}\n'


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


def test_workspace_create_requires_organization(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--organization` is required; omitting it is an argparse usage error."""
    ws = tmp_path / "ws"
    with pytest.raises(SystemExit) as exc:
        main(_ws_args(ws) + ["workspace", "create"])
    assert exc.value.code != 0


def test_workspace_create_without_prior_init_seeds_local_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`workspace create` with no peer local_config.json seeds it from the
    bundled template (no prior `dgml init` needed), copies it to config.json,
    creates the dirs, and tells the caller how to edit the config."""
    ws = tmp_path / "ws"
    rc = main(_ws_args(ws) + ["workspace", "create", "--organization", "Acme"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["local_config_created"] is True
    assert payload["config_written"] is True
    assert payload["organization"] == "Acme"
    assert "next_action" in payload
    # The shared local_config.json was created as a peer of the workspace.
    assert (tmp_path / "local_config.json").exists()
    assert (ws / "docsets").is_dir()
    assert (ws / "config.json").exists()
    # It came from the bundled template (has the model placeholders).
    assert "classification" in (ws / "config.json").read_text(encoding="utf-8")


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
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
    )

    # Empty workspace: no unassigned files → no-op, no LLM or clusterer call.
    with (
        patch("litellm.completion") as mock_completion,
        patch("dgml_core.clustering.run_clustering_detailed") as mock_cluster,
    ):
        rc = main(_ws_args(ws) + ["cluster"])
        assert rc == 0
        payload = _read_stdout(capsys)
        assert payload["clusters"] == {}
        assert payload["failed_file_ids"] == []
        assert payload["skipped"] is False
        assert payload["mode"] == "fresh"
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
        rc = main(_ws_args(ws) + ["cluster"])
    assert rc == 0
    payload = _read_stdout(capsys)
    # Placeholder "unknown_0" from the clusterer is rewritten to the
    # actual DocSet name the file landed in (the LLM-proposed one).
    assert payload["clusters"] == {fid: "Sample Documents"}
    assert payload["failed_file_ids"] == []
    assert payload["mode"] == "fresh"
    assert payload["n_new_clusters"] == 1
    assert payload["assignments"][fid] == {
        "docset": "Sample Documents",
        "confidence": None,
        "is_new": True,
    }

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
        rc = main(_ws_args(ws) + ["cluster"])
        assert rc == 0
        payload = _read_stdout(capsys)
        assert payload["clusters"] == {}
        assert payload["failed_file_ids"] == []
        assert payload["skipped"] is False
        assert payload["mode"] == "incremental"
        mock_completion.assert_not_called()
        mock_cluster.assert_not_called()


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
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
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
        rc = main(_ws_args(ws) + ["cluster", "--config", str(cfg)])
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
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
    )
    main(_ws_args(ws) + ["file", "add", str(sample_pdf)])
    capsys.readouterr()

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
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
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
        rc = main(_ws_args(ws) + ["cluster", "--config", "medium"])
    assert rc == 0
    overrides = mock_cluster.call_args.kwargs["overrides"]
    # The medium preset uses the dense bge text encoder.
    assert overrides["encoder_text"]["name"] == "bge"


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
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
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
        rc = main(_ws_args(ws) + ["cluster"])
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
        rc = main(_ws_args(ws) + ["cluster"])
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
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
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
    assert cls["model"] == "gemini/gemini-3.1-flash-lite"

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
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
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
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
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
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
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
    and ``label_model`` (both required) from config.json.
    """
    _init_ws(ws)
    capsys.readouterr()
    Workspace(root=ws).config_path.write_text(
        json.dumps(
            {
                "generation": {
                    "model": "anthropic/claude-haiku-4-5",
                    "label_model": "anthropic/claude-sonnet-4-6",
                }
            }
        ),
        encoding="utf-8",
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
    # `style` present but no model -> invalid (presence of the section is the switch).
    (ws / "config.json").write_text(json.dumps({"style": {"max_tokens": 100}}), encoding="utf-8")

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
    (ws / "config.json").write_text(
        json.dumps({"style": {"model": "m", "api_key_env": "DGML_STYLE_KEY_MISSING"}}),
        encoding="utf-8",
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
    out_xml = Workspace(root=ws).file_dgml_xml_path(did, fid, "with-text")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><a>tree</a></dg:chunk>', encoding="utf-8"
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
        json.dumps(
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

    docset_dir = Workspace(root=ws).docset_dir(did)
    out_xml = Workspace(root=ws).file_dgml_xml_path(did, fid, "with-text")
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
    assert payload["output_dir"] == str(docset_dir)
    assert payload["coverage_report"] is None  # --no-coverage
    (entry,) = payload["results"]
    assert entry["status"] == "converted"
    assert entry["file_id"] == fid
    assert entry["source"] == "with-text.pdf"
    assert entry["output"] == str(out_xml)
    # Generation grounds each file in place. "hello" doesn't match the real OCR,
    # so 0 elements are annotated, but the file is still grounded (the tree is
    # re-serialized, so it no longer byte-equals fake_xml) and the entry says so.
    assert entry["grounded"] is True
    assert "hello" in out_xml.read_text(encoding="utf-8")

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
    cache_dir = Workspace(root=ws).docset_dir(did) / "cache"

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
        Workspace(root=ws).file_dgml_xml_path(did, fid, "with-text").unlink()
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
    which model runs is a single per-workspace choice (config.json), matching
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
    """No 'generation' section in config.json → the run fails fast with
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
        json.dumps(
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
    """--schema-path loads a schema.json (Schema v1 `tags` map) and threads its
    roles to ConvertOptions.roster_seed and its parent_role hierarchy to
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
    assert kwargs["options"].roster_seed == {
        "PaymentTerms": "the payment clause",
        "DueDate": "when payment is due",
    }
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
    for src in Workspace(root=ws).file_dir(fid).glob("*.pdf"):
        src.unlink()

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
    for src in Workspace(root=ws).file_dir(fid_bad).glob("*.pdf"):
        src.unlink()  # break the second file's source

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
        Workspace(root=ws), {"model": "gemini/gemini-3.1-flash-lite", "max_pages": 1}
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
    from dgml_core.storage import write_json_atomic

    ws = Workspace(root=ws_root)
    ws.file_dir(file_id).mkdir(parents=True, exist_ok=True)
    record = FileRecord(
        id=file_id,
        original_path="/fake/contract.pdf",
        original_filename="contract.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=1,
        text_mode="digital",
    )
    write_json_atomic(ws.file_json_path(file_id), record.to_json())
    # generate resolves the source PDF from the file dir; convert_batch is
    # mocked, so the bytes are never parsed — they just need to exist.
    (ws.file_dir(file_id) / "contract.pdf").write_bytes(b"%PDF-1.4\n%fake\n")
    if with_page_text:
        words = []
        x = 100
        for w in "Payment is due within 30 days of invoice".split():
            words.append({"t": w, "l": [x, 100, x + 50, 120]})
            x += 60
        ws.file_text_dir(file_id).mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            ws.file_text_dir(file_id) / "page_1.json",
            {"file_id": file_id, "page": 1, "width": 1000, "height": 1000, "words": words},
        )
    DocSetStore(ws).add_file(docset_id, file_id)
    return ws


def _generate_with_xml(ws_root: Path, ds_id: str, xml: str, *, debug: bool = False) -> int:
    """Run `docset generate` with convert_batch mocked to emit one rendered
    doc (`xml`) for the seeded contract.pdf. Returns the exit code."""

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        on_output("contract.pdf", xml)
        return {}

    # generate reads the models from config.json's 'generation' section (no flags).
    Workspace(root=ws_root).config_path.write_text(
        json.dumps({"generation": {"model": "test/model", "label_model": "test/label-model"}}),
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
    out_xml = ws.file_dgml_xml_path(ds_id, "f1aaaaaaaaaa", "contract")

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

    content = out_xml.read_text(encoding="utf-8")
    assert 'dg:origin="1 ' in content  # bound to the document's dg prefix
    # Grounded in place — no separate .grounded.xml, no stats sidecar by default.
    assert not (out_xml.parent / "contract.dgml.grounded.xml").exists()
    assert not (out_xml.parent / "contract.dgml.grounding_stats.json").exists()


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
    out_xml = ws.file_dgml_xml_path(ds_id, "f1aaaaaaaaaa", "contract")

    rc = _generate_with_xml(ws_root, ds_id, _GROUNDABLE_XML, debug=True)
    assert rc == 0
    assert (out_xml.parent / "contract.dgml.grounding_stats.json").exists()


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
    out_xml = ws.file_dgml_xml_path(ds_id, "f1aaaaaaaaaa", "contract")

    rc = _generate_with_xml(ws_root, ds_id, _GROUNDABLE_XML)
    assert rc == 0
    payload = _read_generate_stdout(capsys)
    (entry,) = payload["results"]
    assert entry["status"] == "converted"
    assert entry["grounded"] is False
    assert entry["grounding_error"]["code"] == "FILE_NOT_FOUND"
    assert out_xml.exists()  # still written, just not grounded
    assert "dg:origin" not in out_xml.read_text(encoding="utf-8")


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
    file_dir = workspace.file_dir(file_id)
    file_dir.mkdir(parents=True)
    (file_dir / pdf_name).write_bytes(b"%PDF-1.4\n%fake\n")
    workspace.file_json_path(file_id).write_text(
        json.dumps(
            {
                "id": file_id,
                "original_path": f"/src/{pdf_name}",
                "original_filename": pdf_name,
                "sha256": "0" * 64,
                "added_at": "2026-06-05T00:00:00Z",
                "page_count": pages,
                "text_mode": "digital",
            }
        ),
        encoding="utf-8",
    )
    pages_dir = workspace.file_pages_dir(file_id)
    pages_dir.mkdir(parents=True)
    text_dir = workspace.file_text_dir(file_id)
    text_dir.mkdir(parents=True)
    for n in range(1, pages + 1):
        (pages_dir / f"page_{n}.png").write_bytes(f"img-{n}".encode())
        (text_dir / f"page_{n}.json").write_text(
            json.dumps({"file_id": file_id, "page": n, "words": []}), encoding="utf-8"
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
    workspace.docset_dir(docset_id).mkdir(parents=True)
    (workspace.docset_dir(docset_id) / "docset.json").write_text(
        json.dumps({"id": docset_id, "name": "T", "description": "", "key_questions": []}),
        encoding="utf-8",
    )
    xml_path = workspace.file_dgml_xml_path(docset_id, file_id, "doc")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_bytes(_NODE_XML)


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
    xml_path = workspace.file_dgml_xml_path("ds000000001a", "f0000000001a", "doc")
    xml_path.write_bytes(_NODE_XML.replace(b"100", b"999"))

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
    # export_attestation's signature: (attestation, attestation_path, archive_path);
    # staking calls it with unpacked=True, so the loose attestation path is set.
    def _fake_export(  # type: ignore[no-untyped-def]
        ws: Any,
        file_id: str,
        out_dir: Path,
        docset_id: str | None = None,
        *,
        unpacked: bool = False,
    ):
        attestation = SimpleNamespace(root="deadbeef", leaves=[1, 2, 3])
        return attestation, out_dir / "META-INF" / "dgml-attestation.xml", None

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
    # Crucially: nothing was sent to the chain.
    assert fake.broadcast == []


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
    workspace.docset_dir(docset_id).mkdir(parents=True)
    (workspace.docset_dir(docset_id) / "docset.json").write_text(
        json.dumps({"id": docset_id, "name": "T", "description": "", "key_questions": []}),
        encoding="utf-8",
    )
    xml_path = workspace.file_dgml_xml_path(docset_id, file_id, "doc")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_bytes(_DISCOVER_XML)


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
    roster, parent_map = _load_schema_seed(p)
    assert {"PartyInformation", "PartyAddress", "OrderDate"} <= set(roster)
    assert roster["PartyAddress"] == "address"
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
    roster, parent_map = _load_schema_seed(p)
    assert roster == {"PartyInformation": "party block", "PartyAddress": "address"}
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
        json.dumps(
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
    assert (Workspace(root=ws).docset_schema_path(ds_id)).name == "extraction-schema.rnc"
    assert Workspace(root=ws).docset_schema_path(ds_id).exists()


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

    # get-schema json returns the engine's grounded_field JSON Schema projection.
    rc = main(_ws_args(ws) + ["extraction", "get-schema", ds_id, "--schema-format", "json"])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["schema_format"] == "json"
    assert payload["schema"]["properties"]["VendorName"]["anyOf"] == [
        {"$ref": "#/definitions/grounded_field"},
        {"$ref": "#/definitions/computed_field"},
    ]


def test_extraction_get_schema_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    ds_id = _new_docset(ws, capsys)
    rc = main(_ws_args(ws) + ["extraction", "get-schema", ds_id])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "SCHEMA_NOT_FOUND"


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
    xml_path = wsx.file_dgml_xml_path(ds_id, "fileabc", "doc")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(standalone_extraction_doc(values, vocab=vocab), encoding="utf-8")

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
    """generate-schema converts the LLM's JSON Schema to RNC and stores it.
    The sample PDF is placed directly (no ghostscript needed for schema-gen)."""
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    _write_grounded_config(ws)
    ds_id = _new_docset(ws, capsys)

    # Seed a source PDF where _pdf_path expects it (files/<id>/*.pdf).
    fid = "filexyz12345"
    file_dir = Workspace(root=ws).file_dir(fid)
    file_dir.mkdir(parents=True, exist_ok=True)
    _write_blank_pdf(file_dir / "doc.pdf", 1)

    response = _tool_response("submit_schema", {"schema": _JSON_SCHEMA})
    with patch("litellm.completion", return_value=response):
        rc = main(_ws_args(ws) + ["extraction", "generate-schema", ds_id, "--from-file", fid])
    assert rc == 0
    payload = _read_stdout(capsys)
    assert payload["schema_format"] == "rnc"
    assert "element docset:VendorName" in payload["schema"]
    assert payload["from_file_ids"] == [fid]
    assert (
        Workspace(root=ws).docset_schema_path(ds_id).read_text(encoding="utf-8")
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
    Workspace(root=ws).file_dgml_xml_path(ds_id, fid, "doc").unlink()  # re-extract cleanly
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

    xml_path = Workspace(root=ws).file_dgml_xml_path(ds_id, "fileauto0001", "doc")
    xml = xml_path.read_text(encoding="utf-8")
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
    out_xml = Workspace(root=ws).file_dgml_xml_path(did, fid, "with-text")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" xmlns:docset="http://www.dgml.io/ws/T">'
        "<dg:extraction>"
        '<docset:VendorName dg:origin="1 10 20 30 40">Acme</docset:VendorName>'
        "</dg:extraction></dg:chunk>",
        encoding="utf-8",
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

    final = out_xml.read_text(encoding="utf-8")
    assert "the tree" in final  # document tree generated
    assert "<dg:extraction" in final  # prior extraction carried over
    assert ">Acme</docset:VendorName>" in final
    assert 'dg:origin="1 10 20 30 40"' in final  # grounding survived verbatim
