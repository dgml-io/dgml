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

"""Per-document batch labeling is the synchronous labeling, byte for byte.

Open-vocabulary labeling under ``--batch`` runs each document as one
``run_stage`` (its chunks fan out into one wave). These tests pin that it is
the sync labeling exactly: the same requests per document, the same result,
labeled blocks, roster — after EVERY document, so the next one is prompted
identically — warnings and usage rows (modulo the tier). Synthetic data only.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest
from dgml_core import llm
from dgml_core.batch import BatchExecutor, BatchItemError, FakeBackend, Unit, run_stage
from dgml_core.generation import label as label_mod
from dgml_core.generation.blocks import Block
from dgml_core.generation.label import RosterEntry
from dgml_core.generation.vocab import OPEN_VOCAB, TagVocab
from dgml_core.storage import Workspace
from dgml_core.usage import read_events

from .conftest import FakeLLMResponse

MODEL = "anthropic/claude-haiku-4-5"
_LINE_RE = re.compile(r"(?m)^(b\d+) (\w+)(?: \[[^]]*\])?: (.*)$")


def _listing(kwargs: dict[str, Any]) -> str:
    return str(kwargs["messages"][1]["content"][-1]["text"])


def _ids(kwargs: dict[str, Any]) -> list[str]:
    return [i for i, _s, _t in _LINE_RE.findall(_listing(kwargs))]


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=repr))


def _canon(kwargs: list[dict[str, Any]]) -> list[str]:
    return sorted(json.dumps(k, sort_keys=True, default=repr) for k in kwargs)


def _corpus(n: int = 7) -> dict[str, list[Block]]:
    """*n* documents (more than the pilot holds by default), each a heading
    plus 2-5 paragraphs whose first word becomes the coined concept."""
    docs: dict[str, list[Block]] = {}
    for d in range(n):
        blocks = [Block(id="b0001", structure="heading", text=f"Section {d}", level=2)]
        for i in range(2 + d % 4):
            topic = ("Payment", "Delivery", "Warranty", "Termination")[(d + i) % 4]
            blocks.append(Block(id=f"b{i + 2:04d}", structure="p", text=f"{topic} {d}.{i} ok."))
        docs[f"doc{d}.pdf"] = blocks
    return docs


def _answer(kwargs: dict[str, Any]) -> Any:
    """Every labeling path at once: a heading left for the section retry, a
    listing of more than two blocks that comes back unparseable (bisect), and
    concepts coined from the text (the roster grows document by document)."""
    lines = _LINE_RE.findall(_listing(kwargs))
    if "left unlabeled" in _listing(kwargs):
        labels = {i: {"concept": "SectionTitle"} for i, _s, _t in lines}
        return FakeLLMResponse(json.dumps({"labels": labels}), cost=0.0011, prompt_tokens=70)
    if len(lines) > 2:
        return FakeLLMResponse("{ not valid json ,,,", cost=0.0007, prompt_tokens=90)
    labels = {i: {"concept": f"{t.split()[0]}Clause"} for i, s, t in lines if s != "heading"}
    return FakeLLMResponse(
        json.dumps({"labels": labels}), cost=0.0013, prompt_tokens=100, completion_tokens=10
    )


def _executor() -> tuple[FakeBackend, BatchExecutor]:
    def script(request: Any) -> Any:
        try:
            return _answer(request.kwargs)
        except Exception as exc:  # pragma: no cover - the answer never raises
            return BatchItemError(request.custom_id, "invalid", str(exc))

    backend = FakeBackend(script, provider="anthropic")
    executor = BatchExecutor(backend, sleep=lambda _s: None, min_wave_size=1, sync_execute=_answer)
    return backend, executor


def _label_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, batch: bool, n: int = 7
) -> dict[str, Any]:
    monkeypatch.setattr(label_mod, "_MAX_CHUNK_CHARS", 70)  # 1-3 chunks per document
    # Planning is fixed here so the labeling row holds only labeling responses.
    monkeypatch.setattr(
        label_mod, "plan_concept_roster", lambda docs, **_kw: {"PaymentClause": "a payment clause"}
    )
    sync_sent: list[dict[str, Any]] = []

    def sync_completion(kwargs: dict[str, Any], **_kw: Any) -> Any:
        sync_sent.append(kwargs)
        return _answer(kwargs)

    monkeypatch.setattr(llm, "_completion_with_retry", sync_completion)
    backend, executor = _executor()
    ws = Workspace(root=tmp_path / ("batch" if batch else "sync"))
    ws.root.mkdir(parents=True)
    cfg = llm.LLMConfig(model=MODEL, workspace=ws, debug=True, operation="label")
    docs = _corpus(n)
    order: list[str] = []
    rosters: list[Any] = []
    requests: dict[str, list[Any]] = {}

    def submitted() -> list[dict[str, Any]]:
        return [r.kwargs for b in backend.submitted for r in b]

    def label_document(
        name: str, blocks: list[Block], roster: dict[str, RosterEntry], vocab: TagVocab
    ) -> Any:
        before = len(submitted()) if batch else len(sync_sent)
        if batch:
            unit = Unit(
                name,
                cfg,
                label_mod.label_document_steps(
                    name,
                    blocks,
                    roster,
                    config=cfg,
                    cache_dir=None,
                    debug=True,
                    log=lambda _m: None,
                    vocab=vocab,
                ),
            )
            outcome = run_stage([unit], executor, stage="label")[name]
            assert outcome.error is None
            out = outcome.result
            sent = submitted()[before:]
        else:
            out = label_mod._label_one_document(
                name,
                blocks,
                roster,
                config=cfg,
                cache_dir=None,
                debug=True,
                log=lambda _m: None,
                vocab=vocab,
            )
            sent = sync_sent[before:]
        order.append(name)
        requests[name] = _jsonable(sent)
        rosters.append(_jsonable({k: dataclasses.asdict(v) for k, v in roster.items()}))
        return out

    with llm.record_usage_for(cfg):
        warnings = label_mod.label_documents(
            docs, config=cfg, vocab=OPEN_VOCAB, label_document=label_document
        )
    return {
        "order": order,
        "requests": requests,
        "rosters": rosters,
        "warnings": warnings,
        "blocks": _jsonable({n: [dataclasses.asdict(b) for b in bl] for n, bl in docs.items()}),
        "rows": read_events(ws),
        "waves": [[_listing(r.kwargs).splitlines()[0] for r in b] for b in backend.submitted],
    }


def _row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in {"at", "duration_s", "tier"}}


@pytest.mark.parametrize("n", [7, 3], ids=["pilot-staged", "unstaged"])
def test_per_document_batch_labeling_is_the_sync_labeling_document_by_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    sync = _label_corpus(tmp_path, monkeypatch, batch=False, n=n)
    batch = _label_corpus(tmp_path, monkeypatch, batch=True, n=n)

    # The corpus really hits every path.
    pilot = n > label_mod._PILOT_MAX_DOCS
    assert (sync["order"] != sorted(sync["order"])) is pilot  # the pilot reorders
    every = [k for reqs in sync["requests"].values() for k in reqs]
    assert any("left unlabeled" in _listing(k) for k in every)  # section retry
    assert any(len(_ids(k)) > 2 for k in every)  # an unparseable chunk (bisect)
    assert any(len(reqs) > 1 for reqs in sync["requests"].values())  # multi-chunk
    assert len(sync["rosters"][0]) < len(sync["rosters"][-1])  # the roster grows

    assert batch["order"] == sync["order"]  # same documents, same (pilot) order
    for name in sync["order"]:
        assert _canon(batch["requests"][name]) == _canon(sync["requests"][name]), name
    assert batch["rosters"] == sync["rosters"]  # after EVERY document
    assert batch["blocks"] == sync["blocks"]
    assert batch["warnings"] == sync["warnings"]
    assert {r["tier"] for r in batch["rows"]} == {"batch"}
    assert {r["tier"] for r in sync["rows"]} == {"standard"}
    assert [_row(r) for r in batch["rows"]] == [_row(r) for r in sync["rows"]]
    # One stage per document, in order; a document's chunks share a wave.
    assert all(len(set(w)) == 1 for w in batch["waves"])
    assert max(len(w) for w in batch["waves"]) > 1


def test_a_documents_chunks_fan_into_one_wave(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(label_mod, "_MAX_CHUNK_CHARS", 30)  # several chunks
    backend = FakeBackend(
        lambda req: FakeLLMResponse(
            json.dumps({"labels": {i: {"concept": "X"} for i in _ids(req.kwargs)}})
        ),
        provider="anthropic",
    )
    executor = BatchExecutor(backend, sleep=lambda _s: None, min_wave_size=1)
    blocks = [Block(id=f"b{i:04d}", structure="p", text=f"Para {i} text.") for i in (1, 2, 3)]
    roster: dict[str, RosterEntry] = {}
    cfg = llm.LLMConfig(model=MODEL)
    unit = Unit(
        "a.pdf",
        cfg,
        label_mod.label_document_steps(
            "a.pdf",
            blocks,
            roster,
            config=cfg,
            cache_dir=None,
            debug=False,
            log=lambda _m: None,
            vocab=OPEN_VOCAB,
        ),
    )
    out = run_stage([unit], executor)["a.pdf"]
    assert out.ok
    (wave,) = backend.submitted  # every chunk in the one wave
    assert len(wave) > 1
    assert [i for r in wave for i in _ids(r.kwargs)] == ["b0001", "b0002", "b0003"]
    # Roster observations applied after the join (one entry for the concept).
    assert set(roster) == {"X"}
