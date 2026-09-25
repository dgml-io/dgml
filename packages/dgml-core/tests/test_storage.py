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

import tomllib
from pathlib import Path

import pytest
from dgml_core.default_config import PROVIDER_MODELS
from dgml_core.errors import InvalidArgument, WorkspaceNotFound
from dgml_core.storage import (
    Workspace,
    canonical_provider,
    detect_provider,
    detected_api_keys,
    read_json,
    render_config_toml,
    user_config_path,
    write_json_atomic,
    write_text_atomic,
    write_user_config,
)
from dgml_core.workspace_id import new_workspace_id
from dgml_core.workspaces_resolve import default_workspaces_store


def test_resolve_explicit(tmp_path: Path) -> None:
    ws = Workspace.resolve(tmp_path / "x")
    assert ws.root == (tmp_path / "x").resolve()


def test_resolve_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DGML_HOME", str(tmp_path / "envws"))
    ws = Workspace.resolve()
    assert ws.root == (tmp_path / "envws").resolve()


def test_resolve_default_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DGML_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    ws = Workspace.resolve()
    assert ws.root == (tmp_path / "dgml-workspace").resolve()


# --------------------------------------------------------------- id versus path
#
# An id no longer carries a distinguishing prefix, so `--workspace my-workspace` could
# mean a listed workspace or a directory. These pin the four-step rule that decides.


def _list_workspace(workspace_id: str) -> Path:
    """Register ``workspace_id`` in the (tmp-dir-isolated) store and return its root."""
    store = default_workspaces_store()
    store.write_config(workspace_id, "")
    return store.workspace_root(workspace_id)


def test_a_listed_id_resolves_to_that_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 2, and the point of the whole feature: a prefix-free id addresses a
    workspace from any directory, where the same string used to mean `./my-workspace`
    in whichever one you happened to be standing in."""
    root = _list_workspace("my-workspace")
    monkeypatch.chdir(tmp_path)
    ws = Workspace.resolve("my-workspace")
    assert ws.workspaces_id == "my-workspace"
    assert ws.root == root.resolve()


def test_a_minted_id_resolves_the_same_way(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The `ws_` prefix buys nothing at resolution any more — a generated id goes through
    exactly the steps a custom one does."""
    wid = new_workspace_id()
    root = _list_workspace(wid)
    monkeypatch.chdir(tmp_path)
    assert Workspace.resolve(wid).root == root.resolve()
    with pytest.raises(WorkspaceNotFound):
        Workspace.resolve(new_workspace_id())


def test_an_unheld_name_with_a_directory_is_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 3. `--workspace notes` has always meant `./notes`, and still does when the
    store holds nothing by that name — the cwd-relative reading is unchanged."""
    (tmp_path / "notes").mkdir()
    monkeypatch.chdir(tmp_path)
    ws = Workspace.resolve("notes")
    assert ws.workspaces_id is None
    assert ws.root == (tmp_path / "notes").resolve()


def test_an_unheld_name_with_no_directory_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 4. Both places were looked in and neither answered, so the likeliest
    explanation is a typo'd id — which must not become a new directory here."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(WorkspaceNotFound) as exc:
        Workspace.resolve("my-workspace")
    assert "my-workspace" in str(exc.value)
    assert not (tmp_path / "my-workspace").exists()


def test_a_file_of_that_name_is_not_a_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`is_dir`, not `exists`: resolving to a regular file only defers the failure to a
    confusing message about an uninitialized workspace."""
    (tmp_path / "notes").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(WorkspaceNotFound):
        Workspace.resolve("notes")


def test_a_listed_id_beats_a_same_named_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 2 runs before step 3, so the answer cannot change under someone's `mkdir`.
    The escape for the shadowed directory is the leading `./`."""
    root = _list_workspace("my-workspace")
    (tmp_path / "my-workspace").mkdir()
    monkeypatch.chdir(tmp_path)
    assert Workspace.resolve("my-workspace").root == root.resolve()

    shadowed = Workspace.resolve("./my-workspace")
    assert shadowed.workspaces_id is None
    assert shadowed.root == (tmp_path / "my-workspace").resolve()


def test_the_env_var_takes_an_id_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _list_workspace("my-workspace")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DGML_HOME", "my-workspace")
    assert Workspace.resolve().root == root.resolve()


def test_a_workspace_config_is_refused_for_an_id_but_allowed_for_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listed workspace's config lives in the store, so pointing elsewhere can only be
    a mistake. The check has to come *after* the id/path decision, or `--workspace-config`
    alongside an ordinary directory — the case the flag exists for — would fail too."""
    _list_workspace("my-workspace")
    (tmp_path / "notes").mkdir()
    monkeypatch.chdir(tmp_path)
    override = tmp_path / "elsewhere.toml"

    with pytest.raises(InvalidArgument):
        Workspace.resolve("my-workspace", config=override)

    ws = Workspace.resolve("notes", config=override)
    assert ws.config_override == override


def test_is_initialized_follows_the_config_not_the_directories(tmp_path: Path) -> None:
    """A workspace is one because it has a config, not because two directories exist.

    The old test asserted the reverse — that ``init()`` created ``files/`` and
    ``docsets/`` and that their existence meant "initialized". That described
    ``LocalStore``'s layout rather than a workspace, so a remote-backed workspace
    could satisfy it only by scaffolding directories it never wrote to.
    """
    ws = Workspace(root=tmp_path / "ws")
    ws.root.mkdir(parents=True)
    assert not ws.is_initialized()

    # Directories alone do not make a workspace.
    ws.docsets_dir.mkdir()
    ws.files_dir.mkdir()
    assert not ws.is_initialized()

    # A config does, with no directories needed.
    for d in (ws.docsets_dir, ws.files_dir):
        d.rmdir()
    ws.config_path.write_text(
        '[storage.default.blobs]\nprovider = "dgml_core.storage_local:LocalStore"\n',
        encoding="utf-8",
    )
    assert Workspace(root=ws.root).is_initialized()


def test_local_store_creates_its_directories_on_write(tmp_path: Path) -> None:
    """Nothing scaffolds ``files/``/``docsets/`` any more, so the write paths must.

    This is what replaced ``init()``: ``LocalStore``'s writes ``mkdir`` their own
    parents. If that ever moved, every local workspace would break — hence a test
    on the property rather than on the removed method.
    """
    from dgml_core import layout
    from dgml_core.storage_local import LocalStore
    from dgml_core.storage_service import StorageConfig

    root = tmp_path / "ws"
    root.mkdir()
    store = LocalStore(LocalStore.parse_config(StorageConfig(provider="x", root=root)))
    assert not (root / layout.FILES_DIR).exists()

    store.put_blob(layout.file_source_key("f1", "a.pdf"), b"bytes")
    store.put_doc(layout.Collection.DOCSETS, "d1", {"id": "d1"})

    assert (root / layout.FILES_DIR).is_dir()
    assert (root / layout.DOCSETS_DIR).is_dir()


def test_atomic_write_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    write_json_atomic(p, {"x": 1, "y": [1, 2, 3]})
    assert read_json(p) == {"x": 1, "y": [1, 2, 3]}
    assert not p.with_suffix(p.suffix + ".tmp").exists()


def test_read_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    """Hand-edited JSON with duplicate keys (the OCR 'two providers'
    footgun) must surface as CorruptMetadata rather than silently
    resolving to the last value."""
    from dgml_core.errors import CorruptMetadata

    p = tmp_path / "dup.json"
    p.write_text('{"provider": "azure", "provider": "aws"}', encoding="utf-8")
    with pytest.raises(CorruptMetadata, match="duplicate key"):
        read_json(p)


def test_read_json_rejects_duplicate_keys_nested(tmp_path: Path) -> None:
    """Duplicate keys at any nesting level are rejected — the hook fires
    on every JSON object the parser builds."""
    from dgml_core.errors import CorruptMetadata

    p = tmp_path / "dup-nested.json"
    p.write_text('{"ocr": {"provider": "azure", "provider": "aws"}}', encoding="utf-8")
    with pytest.raises(CorruptMetadata, match="duplicate key"):
        read_json(p)


def test_user_config_path_honors_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert user_config_path() == tmp_path / "cfg" / "dgml" / "config.toml"


def test_user_config_path_defaults_per_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    got = user_config_path()
    assert got.parts[-2:] == ("dgml", "config.toml")
    if sys.platform != "win32":
        assert got == Path.home() / ".config" / "dgml" / "config.toml"


def test_user_config_path_windows_uses_appdata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr("dgml_core.storage.sys.platform", "win32")
    monkeypatch.setenv("APPDATA", "C:\\Users\\dev\\AppData\\Roaming")
    got = user_config_path()
    assert got.parts[-2:] == ("dgml", "config.toml")
    assert "Roaming" in str(got)


def test_user_config_path_xdg_wins_on_every_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dgml_core.storage.sys.platform", "win32")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert user_config_path() == tmp_path / "cfg" / "dgml" / "config.toml"


def test_detect_provider() -> None:
    assert detect_provider({"ANTHROPIC_API_KEY": "x", "GEMINI_API_KEY": "y"}) == "mixed"
    assert detect_provider({"ANTHROPIC_API_KEY": "x"}) == "anthropic"
    assert detect_provider({"GEMINI_API_KEY": "y"}) == "google"
    assert detect_provider({"OPENAI_API_KEY": "z"}) == "openai"
    assert detect_provider({}) is None
    # Blank values do not count as set.
    assert detect_provider({"ANTHROPIC_API_KEY": "   "}) is None
    assert detect_provider({"OPENAI_API_KEY": "   "}) is None


def test_detect_provider_checks_openai_last() -> None:
    """An OPENAI_API_KEY never changes what the other two keys would detect.

    The detection order is a contract: a machine that grew an OpenAI key must
    keep resolving to the provider it resolved to before, or an existing
    workspace silently re-provisions onto different models. `--provider openai`
    is the way to ask for OpenAI where the other keys are also present.
    """
    assert detect_provider({"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "z"}) == "anthropic"
    assert detect_provider({"GEMINI_API_KEY": "y", "OPENAI_API_KEY": "z"}) == "google"
    assert (
        detect_provider({"ANTHROPIC_API_KEY": "x", "GEMINI_API_KEY": "y", "OPENAI_API_KEY": "z"})
        == "mixed"
    )


def test_detected_api_keys_report_order() -> None:
    got = detected_api_keys(
        {
            "OPENAI_API_KEY": "z",
            "GEMINI_API_KEY": "y",
            "ANTHROPIC_API_KEY": "x",
            "IGNORED": "q",
        }
    )
    assert got == ["ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"]


def test_every_provider_is_reachable_by_auto_detection_or_a_flag() -> None:
    """Each PROVIDER_MODELS key must be a legal `--provider` value.

    `--provider` takes its choices from PROVIDER_MODELS, and canonical_provider
    is what validates them, so a provider added to the table without a
    round-trip through canonical_provider would be offered by argparse and then
    rejected.
    """
    for provider in PROVIDER_MODELS:
        assert canonical_provider(provider) == provider


def test_canonical_provider_validates() -> None:
    assert canonical_provider("google") == "google"
    assert canonical_provider("mixed") == "mixed"
    with pytest.raises(KeyError):
        canonical_provider("gemini")
    with pytest.raises(KeyError):
        canonical_provider("bogus")


def test_render_config_toml_is_valid_and_complete() -> None:
    for provider in PROVIDER_MODELS:
        data = tomllib.loads(render_config_toml(provider))
        assert set(data["models"]) == {"light", "standard", "advanced", "expert"}
    # Placeholder (no keys): the [models] block is commented out.
    placeholder = render_config_toml(None)
    assert "models" not in tomllib.loads(placeholder)
    assert "# [models]" in placeholder


def test_default_models_are_recognized_by_the_provider_router() -> None:
    """Every shipped default must be an id litellm knows.

    `dgml init --provider X` writes these verbatim, and the LLM layer
    pre-flights each model through `litellm.get_model_info`
    (`llm._require_supported_model`). A stale or mistyped default therefore
    produces a config that only fails on the user's first LLM call, with a
    ModelNotSupported naming a model they never chose — so pin the check here
    rather than discovering it downstream.
    """
    from dgml_core.llm import model_max_output_tokens

    for provider, tiers in PROVIDER_MODELS.items():
        for tier, model in tiers.items():
            assert model_max_output_tokens(model) is not None, (
                f"default {provider}/{tier} = {model!r} is not a model id litellm "
                "recognizes; `dgml init` would write a config that fails on first use"
            )


def test_render_config_toml_ships_opt_in_features_disabled() -> None:
    """Both opt-in features are named (so `dgml init` advertises them) but off.

    They ship as real sections rather than commented out so the user only flips
    the flag; shipping them *enabled* would silently start charging for a vision
    call per page.
    """
    for provider in [*PROVIDER_MODELS, None]:
        data = tomllib.loads(render_config_toml(provider))
        assert data["style"] == {"enabled": False}
        assert data["text_extraction"] == {"enabled": False}


def test_write_user_config_create_then_refresh_with_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    path = user_config_path()

    written, backup = write_user_config("anthropic", overwrite=False)
    assert written is True and backup is None
    assert "anthropic/claude" in path.read_text(encoding="utf-8")

    # Without --refresh a present file is never clobbered.
    written2, backup2 = write_user_config("google", overwrite=False)
    assert written2 is False and backup2 is None
    assert "anthropic/claude" in path.read_text(encoding="utf-8")

    # --refresh overwrites and backs up first.
    written3, backup3 = write_user_config("google", overwrite=True)
    assert written3 is True
    assert backup3 == path.with_suffix(".toml.bak")
    assert "gemini/" in path.read_text(encoding="utf-8")
    assert "anthropic/claude" in backup3.read_text(encoding="utf-8")


def test_write_user_config_backup_is_the_same_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The .bak is the file it backs up, byte for byte, on every platform."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    original = b'# mine\r\n[models]\r\nschema = "x/y"\r\n'
    path.write_bytes(original)
    _, backup = write_user_config("google", overwrite=True)
    assert backup is not None
    assert backup.read_bytes() == original


def test_has_legacy_json_config(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    ws.root.mkdir(parents=True)  # nothing scaffolds the root now that init() is gone
    assert ws.has_legacy_json_config() is False
    (ws.root / "config.json").write_text("{}", encoding="utf-8")
    assert ws.has_legacy_json_config() is True
    # A new-format config.toml alongside it wins — no longer "legacy only".
    ws.config_path.write_text("[models]\n", encoding="utf-8")
    assert ws.has_legacy_json_config() is False


def test_workspace_meta_roundtrip_and_org_fallback(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "dgml-workspace")
    # No workspace.json yet: organization/name fall back to the directory name,
    # preserving the namespaces of pre-workspace.json workspaces.
    assert ws.read_meta() == {}
    assert ws.organization == "dgml-workspace"
    assert ws.display_name == "dgml-workspace"

    ws.write_meta(name="My Workspace", organization="Acme")
    assert ws.read_meta() == {"name": "My Workspace", "organization": "Acme"}
    assert ws.organization == "Acme"
    assert ws.display_name == "My Workspace"


def test_write_text_atomic_keeps_the_given_newlines(tmp_path: Path) -> None:
    """Mixed CRLF and LF text comes back byte for byte (Windows text mode used
    to double the carriage returns; on Linux the next test pins the argument)."""
    path = tmp_path / "config.toml"
    text = '[storage]\r\nprovider = "local"\r\n\n[models]\n'
    write_text_atomic(path, text)
    assert path.read_bytes() == text.encode("utf-8")


def test_atomic_text_writers_switch_off_newline_translation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both atomic writers pass newline="" (Linux CI cannot see the translation)."""
    from dgml_core import storage_local

    seen: list[object] = []
    real_write_text = Path.write_text

    def _spy(self: Path, data: str, *args: object, **kwargs: object) -> int:
        seen.append(kwargs.get("newline", "absent"))
        return real_write_text(self, data, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", _spy)
    write_text_atomic(tmp_path / "a.txt", "x\n")
    storage_local._write_text_atomic(tmp_path / "b.txt", "y\n")
    write_json_atomic(tmp_path / "c.json", {"k": 1})
    assert seen == ["", "", ""]


def test_write_json_atomic_writes_bare_lf(tmp_path: Path) -> None:
    """The JSON writer puts down the LF it renders, not the platform's line ending."""
    path = tmp_path / "stats.json"
    write_json_atomic(path, {"k": [1, 2]})
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"}\n")
