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

"""Extraction-schema generation through a provider's batch API.

The batch path (:func:`generate_schema_batch`) and the synchronous one
(:func:`generate_schema`) are fed the same canned reply, so everything they
leave behind — the RNC, the error, the usage row — can be compared directly.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest
from dgml_core import llm
from dgml_core.batch import BatchExecutor, BatchItemError, FakeBackend
from dgml_core.errors import SchemaGenerationFailed
from dgml_core.grounded import (
    GroundedConfig,
    generate_schema,
    generate_schema_batch,
    generate_schema_steps,
    schema_rnc_from_result,
)
from dgml_core.storage import Workspace
from dgml_core.usage import TIER_BATCH, TIER_STANDARD, read_events

from .conftest import FakeLLMResponse
from .test_grounded import DEFAULT_SCHEMA_MODEL, DEFAULT_VALUES_MODEL, _seed_file

FID = "f1aaaaaaaaaa"

_FIELDS = [
    {"name": "due_date", "kind": "field", "datatype": "date"},
    {
        "name": "line_items",
        "kind": "collection",
        "item": {
            "name": "line_item",
            "kind": "container",
            "fields": [
                {"name": "description", "kind": "field", "datatype": "text"},
                {"name": "amount", "kind": "field", "datatype": "decimal"},
            ],
        },
    },
]


def _submit(name: str, arguments: dict[str, Any]) -> FakeLLMResponse:
    """A canned response carrying one tool call, in the shape
    ``_parse_submit_call`` reads (``.function.name`` / ``.arguments``)."""
    call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )
    return FakeLLMResponse(
        "", tool_calls=[call], cost=0.09, prompt_tokens=9000, completion_tokens=2000
    )


def _config(**kw: Any) -> GroundedConfig:
    return GroundedConfig(
        schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL, **kw
    )


def _rows(workspace: Workspace) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in row.items() if k not in {"at", "duration_s"}}
        for row in read_events(workspace)
    ]


def _executor(backend: FakeBackend, **kw: Any) -> BatchExecutor:
    kw.setdefault("sleep", lambda _s: None)
    return BatchExecutor(backend, **kw)


def test_batch_matches_sync_except_tier(workspace: Workspace, capture_kwargs: Any) -> None:
    _seed_file(workspace, FID)
    captured = capture_kwargs([_submit("submit_schema", {"fields": _FIELDS})])
    sync_rnc = generate_schema(workspace, [FID], config=_config(), docset_name="Inv", debug=True)

    sent: list[dict[str, Any]] = []

    def respond(req: Any) -> Any:
        sent.append(req.kwargs)
        return _submit("submit_schema", {"fields": _FIELDS})

    backend = FakeBackend(respond)
    rnc = generate_schema_batch(
        workspace,
        [FID],
        config=_config(),
        docset_name="Inv",
        executor=_executor(backend),
        debug=True,
    )

    assert rnc == sync_rnc
    assert sent == captured.kwargs  # the batch request is the sync request
    assert len(backend.submitted) == 1  # one request, one batch, one round trip
    sync_row, batch_row = _rows(workspace)
    assert (sync_row["tier"], batch_row["tier"]) == (TIER_STANDARD, TIER_BATCH)
    assert {k: v for k, v in sync_row.items() if k != "tier"} == {
        k: v for k, v in batch_row.items() if k != "tier"
    }


def test_batch_failure_has_the_sync_message(workspace: Workspace) -> None:
    """A request the batch cannot serve falls back to a synchronous call; when
    that fails too, the error reads exactly as the synchronous path's does."""
    _seed_file(workspace, FID)
    backend = FakeBackend(lambda req: BatchItemError(req.custom_id, "invalid", "bad"))

    def sync_execute(_kwargs: dict[str, Any]) -> Any:
        raise RuntimeError("provider exploded")

    with pytest.raises(SchemaGenerationFailed) as excinfo:
        generate_schema_batch(
            workspace,
            [FID],
            config=_config(),
            docset_name="D",
            executor=_executor(backend, sync_execute=sync_execute),
            debug=True,
        )
    assert str(excinfo.value) == "schema generation call failed: RuntimeError: provider exploded"
    (row,) = read_events(workspace)
    assert row["outcome"] == "error" and row["error"] == "RuntimeError: provider exploded"


def test_batch_parse_error_is_the_sync_error(workspace: Workspace) -> None:
    _seed_file(workspace, FID)
    backend = FakeBackend(lambda req: _submit("not_the_right_tool", {"fields": []}))
    with pytest.raises(SchemaGenerationFailed, match="model called unexpected tool"):
        generate_schema_batch(
            workspace, [FID], config=_config(), docset_name="D", executor=_executor(backend)
        )


def test_preparation_fails_before_any_request(workspace: Workspace) -> None:
    """No files (or an unreadable PDF) fails when the request is prepared, so a
    batch driver never submits anything."""
    backend = FakeBackend(lambda req: pytest.fail("no request should be sent"))
    with pytest.raises(SchemaGenerationFailed, match="at least one example"):
        generate_schema_batch(
            workspace, [], config=_config(), docset_name="D", executor=_executor(backend)
        )
    assert backend.submitted == []


def test_steps_yield_one_request_and_return_the_call_result(workspace: Workspace) -> None:
    _seed_file(workspace, FID)
    config, steps = generate_schema_steps(workspace, [FID], config=_config())
    step = next(steps)
    assert step["tool_choice"]["function"]["name"] == "submit_schema"
    with pytest.raises(StopIteration) as done:
        steps.send(_submit("submit_schema", {"fields": _FIELDS}))
    assert isinstance(done.value.value, llm.CallResult)
    assert inspect.getgeneratorstate(steps) == inspect.GEN_CLOSED
    rnc = schema_rnc_from_result(done.value.value, workspace=workspace, docset_name="Inv")
    assert "element docset:DueDate" in rnc
    assert config.model == _config().schema_model
