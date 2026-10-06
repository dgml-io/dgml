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

"""Pass B (labeling) as a request generator: ``label_document_steps``.

Errors reach the generator by ``throw()`` at the pending request, so the
labeling policy (retry once, bisect on bad JSON, stop on an unreachable model)
lives in the generator and works the same under any driver.
"""

from __future__ import annotations

import json
import re
from typing import Any

import litellm
import pytest
from dgml_core import llm
from dgml_core.generation import label as label_mod
from dgml_core.generation.blocks import Block
from dgml_core.generation.label import RosterEntry
from dgml_core.generation.vocab import OPEN_VOCAB

from .conftest import FakeLLMResponse

HAIKU = "anthropic/claude-haiku-4-5"
EPHEMERAL = {"type": "ephemeral"}


def document() -> list[Block]:
    return [
        Block(id="b0001", structure="heading", text="Payment Terms", level=2, lim="4.2"),
        Block(id="b0002", structure="p", text="Invoices are payable within 30 days."),
        Block(id="b0003", structure="p", text="Late payments accrue interest."),
    ]


def steps(blocks: list[Block]) -> Any:
    return label_mod.label_document_steps(
        "a.pdf",
        blocks,
        {"PaymentTerms": RosterEntry(description="the payment clause")},
        config=llm.LLMConfig(model=HAIKU),
        cache_dir=None,
        debug=False,
        log=lambda _m: None,
        vocab=OPEN_VOCAB,
    )


def listing(request: dict[str, Any]) -> str:
    return str(request["messages"][1]["content"][-1]["text"])


def block_ids(request: dict[str, Any]) -> list[str]:
    return re.findall(r"(?m)^(b\d+) ", listing(request))


def labels(ids: list[str]) -> FakeLLMResponse:
    return FakeLLMResponse(json.dumps({"labels": {i: {"concept": "PaymentTerms"} for i in ids}}))


def finish(gen: Any, value: Any, *, throw: bool = False) -> Any:
    with pytest.raises(StopIteration) as done:
        gen.throw(value) if throw else gen.send(value)
    return done.value.value


def test_request_caches_system_prompt_and_roster_but_not_the_listing() -> None:
    request = next(steps(document()))

    system, user = request["messages"]
    assert system["content"][0]["cache_control"] == EPHEMERAL
    roster, blocks = user["content"]
    assert roster["cache_control"] == EPHEMERAL and roster["text"].startswith("PLANNED CONCEPTS")
    assert "cache_control" not in blocks
    assert block_ids(request) == ["b0001", "b0002", "b0003"]


def test_one_good_reply_labels_every_block() -> None:
    blocks = document()
    gen = steps(blocks)
    request = next(gen)
    assert finish(gen, labels(block_ids(request))) == ([], None, [])
    assert [b.concept for b in blocks] == ["PaymentTerms"] * 3


def test_a_thrown_error_retries_the_same_request_once() -> None:
    gen = steps(document())
    first = next(gen)
    assert gen.throw(RuntimeError("provider hiccup")) == first


def test_two_thrown_errors_give_up_softly_then_retry_the_headings() -> None:
    gen = steps(document())
    next(gen)
    gen.throw(RuntimeError("down"))
    section_retry = gen.throw(RuntimeError("down"))
    assert "left unlabeled" in listing(section_retry)

    warnings, label_error, _ = finish(gen, RuntimeError("down"), throw=True)
    assert label_error is None
    assert sum("labeling failed" in w for w in warnings) == 1


def test_an_unreachable_model_is_not_retried_and_is_reported() -> None:
    auth = litellm.exceptions.AuthenticationError(
        message="invalid key", llm_provider="anthropic", model=HAIKU
    )
    gen = steps(document())
    next(gen)
    assert "left unlabeled" in listing(gen.throw(auth))  # no second chunk attempt
    _, label_error, _ = finish(gen, auth, throw=True)
    assert label_error is not None and label_error["code"] == "LABEL_MODEL_UNREACHABLE"


def test_an_unparseable_reply_bisects_the_chunk() -> None:
    gen = steps(document())
    whole = next(gen)
    half = gen.send(FakeLLMResponse("{ not valid json ,,,"))
    assert block_ids(half) == block_ids(whole)[:1]


def test_keyboard_interrupt_is_not_swallowed() -> None:
    gen = steps(document())
    next(gen)
    with pytest.raises(KeyboardInterrupt):
        gen.throw(KeyboardInterrupt())


def test_sync_labeling_retries_a_failed_request(capture_kwargs: Any) -> None:
    replies: list[Any] = [RuntimeError("provider hiccup")]

    def respond(request: dict[str, Any]) -> Any:
        if replies:
            raise replies.pop()
        return labels(block_ids(request))

    captured = capture_kwargs(respond)
    blocks = document()
    warnings, label_error, _ = label_mod._label_one_document(
        "a.pdf",
        blocks,
        {},
        config=llm.LLMConfig(model=HAIKU),
        cache_dir=None,
        debug=False,
        log=lambda _m: None,
        vocab=OPEN_VOCAB,
    )
    assert len(captured.kwargs) == 2 and captured.kwargs[0] == captured.kwargs[1]
    assert (warnings, label_error) == ([], None)
    assert all(b.concept for b in blocks)
