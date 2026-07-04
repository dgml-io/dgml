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

"""Tests for the optional ``text_extraction`` LLM-merge config section."""

from __future__ import annotations

import json

import pytest
from dgml_core.errors import AuthError, TextExtractionConfigInvalid
from dgml_core.storage import Workspace
from dgml_core.text_extraction_config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    load_text_extraction_config,
    resolve_api_key,
)


def _write_config(workspace: Workspace, body: dict[str, object]) -> None:
    workspace.config_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def test_returns_none_when_no_config_file(workspace: Workspace) -> None:
    assert not workspace.config_path.exists()
    assert load_text_extraction_config(workspace) is None


def test_returns_none_when_section_absent(workspace: Workspace) -> None:
    _write_config(workspace, {"ocr": {"provider": "macos"}})
    assert load_text_extraction_config(workspace) is None


def test_minimal_section_applies_defaults(workspace: Workspace) -> None:
    _write_config(workspace, {"text_extraction": {"model": "ollama_chat/gemma4:latest"}})
    cfg = load_text_extraction_config(workspace)
    assert cfg is not None
    assert cfg.model == "ollama_chat/gemma4:latest"
    assert cfg.api_base is None
    assert cfg.temperature == DEFAULT_TEMPERATURE
    assert cfg.max_tokens == DEFAULT_MAX_TOKENS


def test_full_section_parsed(workspace: Workspace) -> None:
    _write_config(
        workspace,
        {
            "text_extraction": {
                "model": "ollama_chat/gemma4:latest",
                "api_base": "http://localhost:11434",
                "temperature": 0.2,
                "max_tokens": 8000,
            }
        },
    )
    cfg = load_text_extraction_config(workspace)
    assert cfg is not None
    assert cfg.api_base == "http://localhost:11434"
    assert cfg.temperature == pytest.approx(0.2)
    assert cfg.max_tokens == 8000


def test_missing_model_is_invalid(workspace: Workspace) -> None:
    _write_config(workspace, {"text_extraction": {"api_base": "http://x"}})
    with pytest.raises(TextExtractionConfigInvalid, match="model"):
        load_text_extraction_config(workspace)


def test_section_not_object_is_invalid(workspace: Workspace) -> None:
    _write_config(workspace, {"text_extraction": "ollama_chat/gemma4:latest"})
    with pytest.raises(TextExtractionConfigInvalid, match="must be a JSON object"):
        load_text_extraction_config(workspace)


def test_both_api_key_and_env_is_invalid(workspace: Workspace) -> None:
    _write_config(
        workspace,
        {
            "text_extraction": {
                "model": "gemini/gemini-3.1-flash-lite",
                "api_key": "literal",
                "api_key_env": "SOME_VAR",
            }
        },
    )
    with pytest.raises(TextExtractionConfigInvalid, match="at most one"):
        load_text_extraction_config(workspace)


def test_bad_temperature_is_invalid(workspace: Workspace) -> None:
    _write_config(workspace, {"text_extraction": {"model": "m", "temperature": "hot"}})
    with pytest.raises(TextExtractionConfigInvalid, match="temperature"):
        load_text_extraction_config(workspace)


def test_bad_max_tokens_is_invalid(workspace: Workspace) -> None:
    _write_config(workspace, {"text_extraction": {"model": "m", "max_tokens": 0}})
    with pytest.raises(TextExtractionConfigInvalid, match="max_tokens"):
        load_text_extraction_config(workspace)


def test_resolve_api_key_literal(workspace: Workspace) -> None:
    _write_config(
        workspace,
        {"text_extraction": {"model": "gemini/x", "api_key": "secret"}},
    )
    cfg = load_text_extraction_config(workspace)
    assert cfg is not None
    assert resolve_api_key(cfg) == "secret"


def test_resolve_api_key_from_env(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_MERGE_KEY", "from-env")
    _write_config(
        workspace,
        {"text_extraction": {"model": "gemini/x", "api_key_env": "MY_MERGE_KEY"}},
    )
    cfg = load_text_extraction_config(workspace)
    assert cfg is not None
    assert resolve_api_key(cfg) == "from-env"


def test_resolve_api_key_missing_env_raises(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MY_MERGE_KEY", raising=False)
    _write_config(
        workspace,
        {"text_extraction": {"model": "gemini/x", "api_key_env": "MY_MERGE_KEY"}},
    )
    cfg = load_text_extraction_config(workspace)
    assert cfg is not None
    with pytest.raises(AuthError, match="MY_MERGE_KEY"):
        resolve_api_key(cfg)


def test_resolve_api_key_none_when_unset(workspace: Workspace) -> None:
    """Local providers (Ollama) set no key; resolution returns None."""
    _write_config(workspace, {"text_extraction": {"model": "ollama_chat/gemma4:latest"}})
    cfg = load_text_extraction_config(workspace)
    assert cfg is not None
    assert resolve_api_key(cfg) is None
