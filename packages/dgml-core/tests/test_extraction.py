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

"""`extract_file` — the library's `dgml extraction extract`.

Two contracts: a file must be assigned to the docset before it can be extracted
into it (the artifact lives under the pair's prefix, which only the assignment
makes reachable), and the call resolves its own config (overriding only the
values model/effort) and defaults stats off.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core import (
    DocSetStore,
    FileNotFound,
    GroundedConfigInvalid,
    SchemaNotFound,
    Workspace,
    extract_file,
)
from dgml_core.errors import ValuesExtractionFailed
from dgml_core.extraction import add_file_and_extract
from dgml_core.grounded import ExtractionResult, GroundedConfig

_RNC = """\
namespace docset = "http://dgml.io/test/T"

Title =
  element docset:Title {
    text
  }
"""

_CONFIG = GroundedConfig(schema_model="test/schema-model", values_model="test/values-model")


def _docset(workspace: Workspace, *, schema: bool = True) -> str:
    store = DocSetStore(workspace)
    ds = store.create(name="T")
    if schema:
        store.set_schema(ds.id, _RNC)
    return ds.id


def _use_config(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stand in for the loader; applies a ``values_model`` override like the real
    one and records the kwargs it was called with."""
    seen: dict[str, Any] = {}

    def fake_load(ws: Workspace, **kw: Any) -> GroundedConfig:
        seen.update(kw)
        model = kw.get("values_model")
        return replace(_CONFIG, values_model=model) if model else _CONFIG

    monkeypatch.setattr("dgml_core.grounded.load_grounded_config", fake_load)
    return seen


def _capture_extract(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_extract(ws: Workspace, d: str, f: str, **kw: Any) -> ExtractionResult:
        seen.update(kw)
        return ExtractionResult(
            values={}, tool_calls=0, xml_key="k", mode="extraction", model=kw["config"].values_model
        )

    monkeypatch.setattr("dgml_core.grounded.extract_values", fake_extract)
    return seen


def test_unassigned_file_is_refused_before_any_llm_call(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    with patch("litellm.completion") as completion:
        with pytest.raises(FileNotFound, match="not assigned to docset"):
            extract_file(workspace, ds_id, "fileaaaaaaaa")
    completion.assert_not_called()


def test_no_schema_is_refused_first(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_id = _docset(workspace, schema=False)
    _use_config(monkeypatch)
    with pytest.raises(SchemaNotFound):
        extract_file(workspace, ds_id, "fileaaaaaaaa")


def test_uses_workspace_config_and_defaults_stats_and_usage_off(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    seen = _capture_extract(monkeypatch)

    result = extract_file(workspace, ds_id, "fileaaaaaaaa")

    assert seen == {"config": _CONFIG, "write_stats": False, "record_usage": False}
    assert result.model == "test/values-model"


def test_model_and_effort_overrides(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_id = _docset(workspace)
    loaded = _use_config(monkeypatch)
    seen = _capture_extract(monkeypatch)

    result = extract_file(
        workspace, ds_id, "fileaaaaaaaa", values_model="test/other", values_effort="low"
    )
    # The override goes through the loader so credentials resolve for the new
    # model, rather than being swapped in after the fact.
    assert loaded == {"values_model": "test/other"}
    assert seen["config"].values_model == "test/other"
    assert seen["config"].values_reasoning_effort == "low"
    assert result.model == "test/other"

    extract_file(workspace, ds_id, "fileaaaaaaaa", values_effort="default")
    assert seen["config"].values_reasoning_effort is None


def test_invalid_effort_is_refused(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    with pytest.raises(GroundedConfigInvalid, match="values_effort"):
        extract_file(workspace, ds_id, "fileaaaaaaaa", values_effort="turbo")


def test_locations_effort_override(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    seen = _capture_extract(monkeypatch)

    extract_file(workspace, ds_id, "fileaaaaaaaa", locations_effort="low")
    assert seen["config"].locations_reasoning_effort == "low"
    # The values effort is untouched by a locations override.
    assert seen["config"].values_reasoning_effort == _CONFIG.values_reasoning_effort

    extract_file(workspace, ds_id, "fileaaaaaaaa", locations_effort="default")
    assert seen["config"].locations_reasoning_effort is None

    extract_file(workspace, ds_id, "fileaaaaaaaa")
    assert seen["config"].locations_reasoning_effort == _CONFIG.locations_reasoning_effort


def test_invalid_locations_effort_is_refused(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    with pytest.raises(GroundedConfigInvalid, match="locations_effort"):
        extract_file(workspace, ds_id, "fileaaaaaaaa", locations_effort="turbo")


# --- Logging: the steps of `extract_file` and `add_file_and_extract` -----------

_FID = "fileaaaaaaaa"


def _file(workspace: Workspace) -> str:
    workspace.docs.put_doc("files", _FID, {"id": _FID})
    return _FID


def _extract_returning_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_extract(ws: Workspace, d: str, f: str, **kw: Any) -> ExtractionResult:
        return ExtractionResult(
            values={},
            tool_calls=3,
            xml_key="k",
            mode="extraction",
            model=kw["config"].values_model,
            usage={"duration_s": 2.5, "cost_usd": 0.0123},
        )

    monkeypatch.setattr("dgml_core.grounded.extract_values", fake_extract)


def _extract_failing(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def fake_extract(ws: Workspace, d: str, f: str, **kw: Any) -> ExtractionResult:
        raise exc

    monkeypatch.setattr("dgml_core.grounded.extract_values", fake_extract)


def _messages(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "dgml_core.extraction" and r.levelno == level
    ]


def test_extract_file_leaves_start_and_finish_to_extract_values(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # extract_values logs them (see test_grounded); a line here would double them.
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    _extract_returning_usage(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="dgml_core")

    extract_file(workspace, ds_id, _FID)

    assert [r for r in caplog.records if r.name == "dgml_core.extraction"] == []


def test_extract_file_logs_failure_at_info_and_raises(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ds_id = _docset(workspace)
    _use_config(monkeypatch)
    _extract_failing(monkeypatch, ValuesExtractionFailed("model gave up"))
    caplog.set_level(logging.DEBUG, logger="dgml_core")

    with pytest.raises(ValuesExtractionFailed):
        extract_file(workspace, ds_id, _FID)

    # Raised, so the caller owns the loud report; the log only narrates.
    assert _messages(caplog, logging.INFO)[-1] == (
        f"extraction of file {_FID} into docset {ds_id} failed: "
        "VALUES_EXTRACTION_FAILED: model gave up"
    )
    assert _messages(caplog, logging.WARNING) == []


def test_add_file_and_extract_logs_nothing_of_its_own_on_success(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    ds_id = _docset(workspace)
    _file(workspace)
    _use_config(monkeypatch)
    _extract_returning_usage(monkeypatch)
    caplog.set_level(logging.DEBUG, logger="dgml_core")

    block = add_file_and_extract(workspace, ds_id, _FID)

    assert block is not None and block["error"] is None
    # The assignment is DocSetStore.add_file's line, and start and finish are
    # extract_values' (see test_docsets, test_grounded); none is repeated here.
    assert [r for r in caplog.records if r.name == "dgml_core.extraction"] == []
    assert [r.getMessage() for r in caplog.records if r.name == "dgml_core.docsets"] == [
        f"assigned file {_FID} to docset {ds_id}"
    ]


def test_add_file_and_extract_without_schema_logs_the_skip(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    ds_id = _docset(workspace, schema=False)
    _file(workspace)
    caplog.set_level(logging.DEBUG, logger="dgml_core")

    assert add_file_and_extract(workspace, ds_id, _FID) is None

    # A decision not to extract, so INFO (like "phase 3 not needed"): --verbose
    # then says why the file has no values.
    assert _messages(caplog, logging.INFO) == [
        f"docset {ds_id} has no extraction schema; not extracting file {_FID}",
    ]


@pytest.mark.parametrize(
    ("exc", "error", "traceback"),
    [
        (ValuesExtractionFailed("model gave up"), "VALUES_EXTRACTION_FAILED: model gave up", False),
        (IndexError("list index out of range"), "IndexError: list index out of range", True),
    ],
)
def test_add_file_and_extract_soft_failure_logs_at_info(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    exc: Exception,
    error: str,
    traceback: bool,
) -> None:
    ds_id = _docset(workspace)
    _file(workspace)
    _use_config(monkeypatch)
    _extract_failing(monkeypatch, exc)
    caplog.set_level(logging.DEBUG, logger="dgml_core")

    block = add_file_and_extract(workspace, ds_id, _FID)

    assert block is not None and block["error"] == error
    # The block carries the failure to the caller, so the log only narrates
    # (under --verbose) — nothing reaches the CLI's default stderr.
    assert _messages(caplog, logging.WARNING) == []
    infos = [
        r for r in caplog.records if r.name == "dgml_core.extraction" and r.levelno == logging.INFO
    ]
    assert [r.getMessage() for r in infos] == [
        f"extraction of file {_FID} into docset {ds_id} failed; the file stays assigned "
        f"without extracted values: {error}"
    ]
    assert bool(infos[0].exc_info) is traceback
