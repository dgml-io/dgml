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
from pathlib import Path

import pytest
from dgml_core.storage import (
    Workspace,
    bundled_default_config_text,
    read_config,
    read_json,
    strip_jsonc_line_comments,
    write_json_atomic,
)


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


def test_init_creates_dirs(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    assert not ws.is_initialized()
    ws.init()
    assert ws.is_initialized()
    assert ws.docsets_dir.is_dir()
    assert ws.files_dir.is_dir()


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


def test_strip_jsonc_line_comments_blanks_full_line_only() -> None:
    """Full-line `//` comments are blanked (line numbers preserved); `//`
    inside a string value — e.g. an https:// endpoint — is left untouched."""
    text = (
        "{\n"
        "  // a leading comment\n"
        '  "endpoint": "https://example.cognitiveservices.azure.com/",\n'
        '  "x": 1\n'
        "}\n"
    )
    stripped = strip_jsonc_line_comments(text)
    # The comment line is blanked but still present (line count unchanged).
    assert stripped.split("\n")[1] == ""
    assert len(stripped.split("\n")) == len(text.split("\n"))
    parsed = json.loads(stripped)
    assert parsed["endpoint"] == "https://example.cognitiveservices.azure.com/"
    assert parsed["x"] == 1


def test_read_config_parses_bundled_template_and_rejects_dupes(tmp_path: Path) -> None:
    from dgml_core.errors import CorruptMetadata

    p = tmp_path / "config.json"
    p.write_text(bundled_default_config_text(), encoding="utf-8")
    data = read_config(p)
    assert isinstance(data, dict)
    assert data["ocr"]["endpoint"].startswith("https://")

    p.write_text('{\n  // c\n  "a": 1,\n  "a": 2\n}\n', encoding="utf-8")
    with pytest.raises(CorruptMetadata, match="duplicate key"):
        read_config(p)


def test_bundled_default_config_shape() -> None:
    """The bundled template parses and carries exactly the intended sections:
    the model/endpoint decisions, and none of the code-defaulted knobs."""
    data = json.loads(strip_jsonc_line_comments(bundled_default_config_text()))
    assert set(data) == {"classification", "generation", "grounded", "ocr"}
    assert data["classification"]["model"]
    assert data["grounded"]["schema_model"]
    assert data["grounded"]["values_model"]
    assert data["ocr"]["provider"] == "azure"
    # No api_key_env prefilled in any section, and no code-defaulted / opt-in
    # sections (text_extraction is opt-in; clustering/conversion default in code).
    assert all("api_key_env" not in section for section in data.values())
    assert "text_extraction" not in data
    assert "clustering" not in data
    assert "conversion" not in data


def test_local_config_path_is_workspace_peer(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "dgml-workspace")
    assert ws.local_config_path == tmp_path / "local_config.json"


def test_ensure_local_config_creates_then_noops(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "dgml-workspace")
    assert ws.ensure_local_config() is True
    assert ws.local_config_path.exists()
    assert "classification" in ws.local_config_path.read_text(encoding="utf-8")
    # Idempotent: a second call leaves the file untouched.
    ws.local_config_path.write_text("edited\n", encoding="utf-8")
    assert ws.ensure_local_config() is False
    assert ws.local_config_path.read_text(encoding="utf-8") == "edited\n"


def test_refresh_local_config_overwrites_with_backup(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "dgml-workspace")
    ws.local_config_path.write_text("old\n", encoding="utf-8")
    backup = ws.refresh_local_config()
    assert backup == ws.local_config_path.with_suffix(".json.bak")
    assert backup.read_text(encoding="utf-8") == "old\n"
    assert "classification" in ws.local_config_path.read_text(encoding="utf-8")


def test_workspace_meta_roundtrip_and_org_fallback(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "dgml-workspace")
    # No workspace.json yet: organization/name fall back to the directory name,
    # preserving the namespaces of pre-workspace.json workspaces.
    assert ws.read_meta() == {}
    assert ws.organization == "dgml-workspace"
    assert ws.display_name == "dgml-workspace"

    ws.write_meta(name="My Workspace", organization="Acme")
    assert ws.meta_path == ws.root / "workspace.json"
    assert ws.read_meta() == {"name": "My Workspace", "organization": "Acme"}
    assert ws.organization == "Acme"
    assert ws.display_name == "My Workspace"


def test_write_config_from_local_copies_verbatim_and_guards(tmp_path: Path) -> None:
    from dgml_core.errors import LocalConfigMissing

    ws = Workspace(root=tmp_path / "dgml-workspace")
    # No peer file yet → raises.
    with pytest.raises(LocalConfigMissing):
        ws.write_config_from_local(overwrite=False)

    ws.local_config_path.write_text('{\n  // keep me\n  "grounded": {}\n}\n', encoding="utf-8")
    assert ws.write_config_from_local(overwrite=False) is True
    assert "// keep me" in ws.config_path.read_text(encoding="utf-8")  # comments survive

    # Existing config.json is not clobbered without overwrite.
    assert ws.write_config_from_local(overwrite=False) is False
    # overwrite re-syncs the edited shared config.
    ws.local_config_path.write_text('{"ocr": {}}\n', encoding="utf-8")
    assert ws.write_config_from_local(overwrite=True) is True
    assert ws.config_path.read_text(encoding="utf-8") == '{"ocr": {}}\n'
