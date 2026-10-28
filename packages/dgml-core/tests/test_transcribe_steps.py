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

"""Pass A (transcription) as a request generator: ``transcribe_steps``.

Gate retry and window split are covered in ``test_generation.py``; here we pin
the window request and the order windows are asked in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dgml_core import llm
from dgml_core.generation import document as document_mod
from dgml_core.generation import transcribe as transcribe_mod

from .conftest import FakeLLMResponse

HAIKU = "anthropic/claude-haiku-4-5"
EPHEMERAL = {"type": "ephemeral"}


@pytest.fixture
def three_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """A three-page PDF whose window slices are just their page indices."""
    monkeypatch.setattr(transcribe_mod, "_count_pages", lambda _b: 3)
    monkeypatch.setattr(document_mod, "slice_pdf", lambda _b, idx, **_kw: bytes(idx))


def instruction(request: dict[str, Any]) -> str:
    return str(request["messages"][1]["content"][0]["text"])


def steps(cache_dir: Path) -> Any:
    return transcribe_mod.transcribe_steps(
        b"%PDF-fake",
        doc_name="doc.pdf",
        config=llm.LLMConfig(model=HAIKU),
        window_size=1,
        cache_dir=cache_dir,
    )


@pytest.mark.usefixtures("three_pages")
def test_window_request_caches_system_prompt_and_window_pdf(tmp_path: Path) -> None:
    request = next(steps(tmp_path))

    system, user = request["messages"]
    assert system["content"][0]["cache_control"] == EPHEMERAL
    text, document = user["content"]
    assert "cache_control" not in text and "pages 1-1 of 3" in text["text"]
    assert document["type"] == "file" and document["cache_control"] == EPHEMERAL


@pytest.mark.usefixtures("three_pages")
def test_windows_are_asked_in_order_each_carrying_the_previous_tail(tmp_path: Path) -> None:
    gen = steps(tmp_path)
    assert "pages 1-1" in instruction(next(gen))
    second = gen.send(FakeLLMResponse("P\tfirst window tail"))
    assert "pages 2-2" in instruction(second)
    assert "first window tail" in instruction(second)
    third = gen.send(FakeLLMResponse("P\tsecond"))
    assert "pages 3-3" in instruction(third)

    with pytest.raises(StopIteration) as done:
        gen.send(FakeLLMResponse("P\tthird"))
    assert [b.text for b in done.value.value] == ["first window tail", "second", "third"]
    assert (tmp_path / "doc_blocks.json").exists()


@pytest.mark.usefixtures("three_pages")
def test_a_truncated_window_is_continued_with_an_assistant_prefill(tmp_path: Path) -> None:
    gen = steps(tmp_path)
    first = next(gen)
    continued = gen.send(FakeLLMResponse("P\tHello wo", finish_reason="length"))

    assert continued["messages"] == [
        *first["messages"],
        {"role": "assistant", "content": "P\tHello wo"},
    ]


@pytest.mark.usefixtures("three_pages")
def test_closing_mid_document_writes_no_blocks_cache(tmp_path: Path) -> None:
    gen = steps(tmp_path)
    next(gen)
    gen.send(FakeLLMResponse("P\tfirst"))
    gen.close()
    assert not (tmp_path / "doc_blocks.json").exists()


def test_a_cached_document_yields_no_request(tmp_path: Path) -> None:
    from dgml_core.generation.blocks import Block

    (tmp_path / "doc_blocks.json").write_text(
        transcribe_mod.blocks_to_json([Block(id="b0001", structure="p", text="cached")]),
        encoding="utf-8",
    )
    with pytest.raises(StopIteration) as done:
        next(steps(tmp_path))
    assert [b.text for b in done.value.value] == ["cached"]
