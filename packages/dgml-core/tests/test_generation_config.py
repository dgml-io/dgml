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

import pytest
from dgml_core.errors import (
    AuthError,
    GenerationConfigInvalid,
    GenerationConfigMissing,
)
from dgml_core.generation import (
    GenerationConfig,
    load_generation_config,
    resolve_generation_api_key,
)
from dgml_core.storage import Workspace

MODEL = "anthropic/claude-haiku-4-5"
LABEL_MODEL = "anthropic/claude-sonnet-4-6"


def _write(workspace: Workspace, section: object) -> None:
    workspace.config_path.write_text(json.dumps({"generation": section}), encoding="utf-8")


# ---------------------------------------------------------------------------
# load_generation_config
# ---------------------------------------------------------------------------


def test_missing_when_no_config_file(workspace: Workspace) -> None:
    with pytest.raises(GenerationConfigMissing):
        load_generation_config(workspace)


def test_missing_when_no_generation_section(workspace: Workspace) -> None:
    workspace.config_path.write_text(json.dumps({"ocr": {}}), encoding="utf-8")
    with pytest.raises(GenerationConfigMissing):
        load_generation_config(workspace)


def test_invalid_when_top_level_not_object(workspace: Workspace) -> None:
    workspace.config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config(workspace)


def test_invalid_when_section_not_object(workspace: Workspace) -> None:
    _write(workspace, "haiku")
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config(workspace)


def test_invalid_when_model_missing_or_empty(workspace: Workspace) -> None:
    _write(workspace, {"label_model": LABEL_MODEL})
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config(workspace)

    _write(workspace, {"model": "   ", "label_model": LABEL_MODEL})
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config(workspace)


def test_invalid_when_label_model_missing_or_empty(workspace: Workspace) -> None:
    # label_model is required — no fallback to model.
    _write(workspace, {"model": MODEL})
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config(workspace)

    _write(workspace, {"model": MODEL, "label_model": "   "})
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config(workspace)


def test_minimal_config_requires_both_models(workspace: Workspace) -> None:
    _write(workspace, {"model": MODEL, "label_model": LABEL_MODEL})
    cfg = load_generation_config(workspace)
    assert cfg == GenerationConfig(model=MODEL, label_model=LABEL_MODEL)
    assert cfg.api_key is None
    assert cfg.api_base is None


def test_full_config_round_trips(workspace: Workspace) -> None:
    _write(
        workspace,
        {
            "model": MODEL,
            "label_model": LABEL_MODEL,
            "api_key_env": "MY_KEY",
            "api_base": "http://localhost:11434",
        },
    )
    cfg = load_generation_config(workspace)
    assert cfg.model == MODEL
    assert cfg.label_model == LABEL_MODEL
    assert cfg.api_key_env == "MY_KEY"
    assert cfg.api_base == "http://localhost:11434"


def test_invalid_when_api_key_and_env_both_set(workspace: Workspace) -> None:
    _write(
        workspace,
        {
            "model": MODEL,
            "label_model": LABEL_MODEL,
            "api_key": "sk-x",
            "api_key_env": "MY_KEY",
        },
    )
    with pytest.raises(GenerationConfigInvalid):
        load_generation_config(workspace)


# ---------------------------------------------------------------------------
# resolve_generation_api_key
# ---------------------------------------------------------------------------


def test_resolve_prefers_literal_key() -> None:
    cfg = GenerationConfig(model=MODEL, label_model=LABEL_MODEL, api_key="sk-literal")
    assert resolve_generation_api_key(cfg) == "sk-literal"


def test_resolve_none_when_unset() -> None:
    cfg = GenerationConfig(model=MODEL, label_model=LABEL_MODEL)
    assert resolve_generation_api_key(cfg) is None


def test_resolve_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_GEN_KEY", "sk-from-env")
    cfg = GenerationConfig(model=MODEL, label_model=LABEL_MODEL, api_key_env="MY_GEN_KEY")
    assert resolve_generation_api_key(cfg) == "sk-from-env"


def test_resolve_raises_when_env_var_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_GEN_KEY", raising=False)
    cfg = GenerationConfig(model=MODEL, label_model=LABEL_MODEL, api_key_env="MISSING_GEN_KEY")
    with pytest.raises(AuthError):
        resolve_generation_api_key(cfg)
