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

"""The one-request stages as steps: schema generation, classification, and
Pass B roster planning (through ``single_calls``)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dgml_core import classification, llm
from dgml_core.classification import ClassificationConfig
from dgml_core.errors import ClassificationFailed, SchemaGenerationFailed
from dgml_core.generation import label as label_mod
from dgml_core.generation.blocks import Block
from dgml_core.generation.single_calls import STAGE_PLAN, single_calls_through
from dgml_core.grounded import GroundedConfig, generate_schema
from dgml_core.storage import Workspace
from dgml_core.usage import read_events

from .conftest import FakeLLMResponse
from .test_classification import _seed_file as _seed_classified_file
from .test_classification import _seed_page_image
from .test_grounded import DEFAULT_SCHEMA_MODEL, DEFAULT_VALUES_MODEL, _seed_file

EPHEMERAL = {"type": "ephemeral"}


def tool_call(name: str, arguments: dict[str, Any], **usage: Any) -> FakeLLMResponse:
    call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )
    return FakeLLMResponse("", tool_calls=[call], **usage)


# -- schema generation ---------------------------------------------------------


def schema_config() -> GroundedConfig:
    return GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)


def test_schema_request_forces_submit_schema_with_every_pdf(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")
    _seed_file(workspace, "f2bbbbbbbbbb")
    fields = [{"name": "due_date", "kind": "field", "datatype": "date"}]
    captured = capture_kwargs([tool_call("submit_schema", {"fields": fields})])
    rnc = generate_schema(
        workspace, ["f1aaaaaaaaaa", "f2bbbbbbbbbb"], config=schema_config(), docset_name="Invoice"
    )

    (request,) = captured.kwargs
    assert request["tool_choice"] == {"type": "function", "function": {"name": "submit_schema"}}
    assert "reasoning_effort" not in request
    assert [b["type"] for b in request["messages"][1]["content"]] == ["text", "file", "file"]
    assert "DueDate" in rnc


def test_schema_provider_failure_keeps_its_message_and_error_row(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    _seed_file(workspace, "f1aaaaaaaaaa")

    def boom(_request: dict[str, Any]) -> Any:
        raise RuntimeError("provider exploded")

    capture_kwargs(boom)
    with pytest.raises(SchemaGenerationFailed) as failed:
        generate_schema(
            workspace, ["f1aaaaaaaaaa"], config=schema_config(), docset_name="D", debug=True
        )
    assert str(failed.value) == "schema generation call failed: RuntimeError: provider exploded"
    (row,) = read_events(workspace)
    assert (row["outcome"], row["error"]) == ("error", "RuntimeError: provider exploded")


# -- classification ------------------------------------------------------------


def test_classify_request_requires_a_tool_and_carries_the_page_images(
    workspace: Workspace,
) -> None:
    _seed_classified_file(workspace, "fid1", filename="po-1.pdf")
    _seed_page_image(workspace, "fid1", 1, b"\x89PNG\r\n\x1a\npage-a")
    cfg, steps = classification.classify_steps(
        workspace,
        ["fid1"],
        config=ClassificationConfig(model="gemini/gemini-2.5-flash-lite"),
        prompt=classification._build_prompt_new_only(),
        tools=[classification._create_new_docset_tool()],
    )
    request = next(steps)

    assert cfg.operation == "classify"
    assert request["tool_choice"] == "required"
    assert [b["type"] for b in request["messages"][0]["content"]] == ["text", "image_url"]
    reply = tool_call("create_new_docset", {"name": "Orders", "description": "d"})
    with pytest.raises(StopIteration) as done:
        steps.send(reply)
    assert done.value.value.response is reply
    assert read_events(workspace) == []  # only a driver records usage


def test_classify_fails_before_any_request_without_page_images(workspace: Workspace) -> None:
    _seed_classified_file(workspace, "blankfid", filename="blank.pdf")
    with pytest.raises(ClassificationFailed, match="no page images"):
        classification.classify_steps(
            workspace,
            ["blankfid"],
            config=ClassificationConfig(model="gemini/gemini-2.5-flash-lite"),
            prompt="p",
            tools=[],
        )


# -- Pass B roster planning ----------------------------------------------------


def two_docs() -> dict[str, list[Block]]:
    return {
        "a.pdf": [Block(id="b0001", structure="heading", text="Payment Terms", level=1)],
        "b.pdf": [Block(id="b0001", structure="heading", text="Fees", level=1)],
    }


DRAFT = {"concepts": {"PaymentTerms": "when paid"}}
REFINED = {"concepts": {"PaymentTerms": "when invoices are paid", "Fees": "the fees"}}


def plan(tmp_path: Path) -> dict[str, str]:
    return label_mod.plan_concept_roster(
        two_docs(),
        config=llm.LLMConfig(model="anthropic/claude-sonnet-4-6"),
        cache_dir=tmp_path,
        debug=False,
        log=lambda _m: None,
    )


def test_roster_planning_drafts_then_refines_with_cached_skeletons(
    capture_kwargs: Any, tmp_path: Path
) -> None:
    captured = capture_kwargs(
        [FakeLLMResponse(json.dumps(DRAFT)), FakeLLMResponse(json.dumps(REFINED))]
    )
    assert plan(tmp_path) == REFINED["concepts"]

    draft, refine = captured.kwargs
    assert draft["messages"][0]["content"][0]["cache_control"] == EPHEMERAL
    assert draft["messages"][1]["content"][0]["cache_control"] == EPHEMERAL
    assert refine["messages"][:2] == draft["messages"]
    assert refine["messages"][2] == {"role": "assistant", "content": json.dumps(DRAFT)}
    assert refine["messages"][3]["role"] == "user"


def test_a_runner_sees_the_same_requests_under_the_plan_stage(
    capture_kwargs: Any, tmp_path: Path
) -> None:
    replies = [FakeLLMResponse(json.dumps(DRAFT)), FakeLLMResponse(json.dumps(REFINED))]
    sync = capture_kwargs(list(replies))
    plan(tmp_path / "sync")

    stages: list[str] = []

    def runner(stage: str, config: llm.LLMConfig, steps: Any) -> Any:
        stages.append(stage)
        return llm.drive(steps, config)

    through_runner = capture_kwargs(list(replies))
    with single_calls_through(runner):
        assert plan(tmp_path / "runner") == REFINED["concepts"]
    assert stages == [STAGE_PLAN]
    assert through_runner.kwargs == sync.kwargs


def test_a_failed_plan_is_best_effort(capture_kwargs: Any, tmp_path: Path) -> None:
    def down(_request: dict[str, Any]) -> Any:
        raise RuntimeError("provider down")

    capture_kwargs(down)
    assert plan(tmp_path) == {}
