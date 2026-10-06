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

"""``convert_batch`` in batch mode against the synchronous pipeline.

Every test here runs the same documents twice over the same canned model
replies — once synchronously, once through the batch driver with a
:class:`~dgml_core.batch.FakeBackend` standing in for the provider — and
asserts the two runs are indistinguishable apart from the usage rows' ``tier``.
That is the property the before/after evaluation relies on: batching changes
the price of a request, never its bytes or its outcome.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from dgml_core import llm
from dgml_core.batch import BatchItemError, BatchRequest, FakeBackend, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.errors import GenerationConfigInvalid
from dgml_core.generation import document as document_mod
from dgml_core.generation import label as label_mod
from dgml_core.generation import transcribe as transcribe_mod
from dgml_core.generation.config import load_generation_batch
from dgml_core.generation.pipeline import (
    BATCH_LABEL_OFF,
    BatchOptions,
    ConvertOptions,
    convert_batch,
)
from dgml_core.generation.prompts import get as prompt
from dgml_core.generation.schema import parse_authored_schema
from dgml_core.generation.to_semantic import build_header
from dgml_core.generation.vocab import TagVocab
from dgml_core.storage import Workspace
from dgml_core.usage import read_events

from .conftest import FakeLLMResponse

TRANSCRIBE_MODEL = "anthropic/claude-haiku-4-5"
LABEL_MODEL = "anthropic/claude-sonnet-4-6"
_HEADER = build_header("TestOrg", "TestDocSet")
# Page counts per document: a one-window doc next to a three-window one, so the
# batch run needs several waves and the short document finishes early.
_PAGES = {"alpha.pdf": 1, "bravo.pdf": 3}
_HEADER_RE = re.compile(r"pages (\d+)-(\d+) of (\d+)")
_BLOCK_ID_RE = re.compile(r"\bb\d{4}\b")


def _system_text(kwargs: dict[str, Any]) -> str:
    content = kwargs["messages"][0]["content"]
    if isinstance(content, str):
        return content
    return "".join(str(block.get("text", "")) for block in content)


def _user_text(kwargs: dict[str, Any]) -> str:
    content = kwargs["messages"][1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict))


def _reply(text: str) -> FakeLLMResponse:
    return FakeLLMResponse(text, cost=0.01, prompt_tokens=100, completion_tokens=20)


def make_answer(*, fail_total: int | None = None) -> Callable[[dict[str, Any]], Any]:
    """One deterministic model: every request's reply depends only on its bytes.

    Transcription windows answer with text naming their document (by page
    total) and page; labeling tags every listed block ``Greeting``; roster
    planning and concept description answer with fixed JSON. ``fail_total``
    makes every transcription request of the document with that page count
    raise, as an unreachable provider would.
    """

    def answer(kwargs: dict[str, Any]) -> Any:
        system = _system_text(kwargs)
        user = _user_text(kwargs)
        if system == transcribe_mod.SYSTEM_PROMPT:
            match = _HEADER_RE.search(user)
            assert match is not None, user
            first, _last, total = (int(g) for g in match.groups())
            if fail_total is not None and total == fail_total:
                raise RuntimeError("provider down")
            blocks = [{"structure": "p", "text": f"Hello from doc{total} page{first}."}]
            return _reply(json.dumps({"continues": "", "blocks": blocks}))
        if system == label_mod.SYSTEM_PROMPT:
            ids = sorted(set(_BLOCK_ID_RE.findall(user)))
            labels = {block_id: {"concept": "Greeting"} for block_id in ids}
            return _reply(json.dumps({"labels": labels}))
        if system == label_mod.PLAN_SYSTEM_PROMPT:
            return _reply(json.dumps({"concepts": {"Greeting": "a greeting line"}}))
        if system == prompt("describe_concepts"):
            return _reply(json.dumps({"descriptions": {}}))
        raise AssertionError(f"unexpected request with system prompt {system[:60]!r}")

    return answer


@pytest.fixture(autouse=True)
def _stub_pdf_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Page counting and slicing stubbed: a window's payload is its page indices."""
    monkeypatch.setattr(
        document_mod,
        "load_document_as_pdf",
        lambda path, *, converters: f"%PDF-{Path(path).name}".encode(),
    )
    monkeypatch.setattr(
        transcribe_mod,
        "_count_pages",
        lambda pdf_bytes: _PAGES[pdf_bytes.decode().removeprefix("%PDF-")],
    )
    monkeypatch.setattr(document_mod, "slice_pdf", lambda _b, idx, **_kw: bytes(idx))


class _Provider:
    """The fake batch provider a test runs against: which model answers, and
    every backend the pipeline built (one per executor)."""

    def __init__(self) -> None:
        self.answer: Callable[[dict[str, Any]], Any] = make_answer()
        self.built: list[FakeBackend] = []

    def script(self, request: BatchRequest) -> Any:
        try:
            return self.answer(request.kwargs)
        except RuntimeError as exc:
            # Not retryable: the executor serves it synchronously, where the
            # same answer raises again — the outcome the sync path sees.
            return BatchItemError(request.custom_id, "invalid", str(exc))


@pytest.fixture
def fake_provider() -> Iterator[_Provider]:
    """Stand a FakeBackend in for the Anthropic batch backend; restore after."""
    saved = dict(batch_registry._REGISTRY)
    provider = _Provider()

    def factory(_cfg: Any) -> FakeBackend:
        backend = FakeBackend(provider.script, provider="anthropic")
        provider.built.append(backend)
        return backend

    register_backend("anthropic", factory)
    try:
        yield provider
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


def _run(
    tmp: Path,
    answer: Callable[[dict[str, Any]], Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    batch: _Provider | None,
    vocab: TagVocab | None = None,
    schema_text: str | None = None,
    logs: list[str] | None = None,
    label: bool = True,
    pages: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run convert_batch once; return everything observable about the run."""
    tmp.mkdir(parents=True, exist_ok=True)
    ws = Workspace(root=tmp / "ws")
    ws.root.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp / "cache"
    monkeypatch.setattr(llm, "_completion_with_retry", lambda kwargs, **_kw: answer(kwargs))
    if batch is not None:
        batch.answer = answer
    schema = parse_authored_schema(schema_text)[0] if schema_text else None
    batch_opts = (
        BatchOptions(poll_interval_s=0, min_wave_size=1, label=label) if batch is not None else None
    )
    outputs: dict[str, str] = {}
    errors: dict[str, str] = {}
    convert_batch(
        [Path(name) for name in (pages or _PAGES)],
        options=ConvertOptions(
            model=TRANSCRIBE_MODEL,
            label_model=LABEL_MODEL,
            dgml_header=_HEADER,
            cache_dir=cache_dir,
            debug=True,
            workspace=ws,
            max_parallel_docs=1,
            window_size=1,  # one page per window: multi-page docs need several waves
            schema_seed=schema,
            vocab=vocab,
            progress=(logs.append if logs is not None else None),
            batch=batch_opts,
        ),
        on_output=outputs.__setitem__,
        on_error=errors.__setitem__,
    )
    files = {
        str(p.relative_to(cache_dir)): p.read_bytes()
        for p in sorted(cache_dir.rglob("*"))
        if p.is_file()
    }
    schema_json = (tmp / "schema.json").read_bytes() if (tmp / "schema.json").exists() else None
    return {
        "outputs": outputs,
        "errors": errors,
        "files": files,
        "schema_json": schema_json,
        "rows": read_events(ws),
        "stats": batch_opts.stats if batch_opts is not None else None,
    }


def _comparable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Usage rows minus what legitimately differs between the two paths (the
    tier, and a scope's split into one row per tier: see :func:`_rejoined`)."""
    drop = {"at", "duration_s", "tier"}
    out = [{k: v for k, v in row.items() if k not in drop} for row in _rejoined(rows)]
    return sorted(out, key=lambda r: json.dumps(r, sort_keys=True))


def _rejoined(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """*rows* with every tier-split scope folded back into the one row the
    synchronous run writes for it (its parts summed, in order).

    A scope served by two tiers — the labeling pass's row when roster planning
    ran as a batch stage and labeling stayed synchronous — is written as one
    row per tier (``usage.scope_events``), so only the sum is comparable."""
    sums = ("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens")
    out: list[dict[str, Any]] = []
    for row in rows:
        context = dict(row.get("context") or {})
        if not context.pop("tier_split", False):
            out.append(row)
            continue
        part = {**row, "context": context}
        prev = out[-1] if out else None
        if prev is not None and prev.get("_split") and prev["context"] == context:
            for key in (*sums, "cache_read_tokens", "cache_creation_tokens"):
                if prev.get(key) is None and part.get(key) is None:
                    continue
                prev[key] = (prev.get(key) or 0) + (part.get(key) or 0)
            continue
        out.append({**part, "_split": True})
    return [{k: v for k, v in row.items() if k != "_split"} for row in out]


_CLOSED_SCHEMA = '{"Greeting": "a greeting line"}'


def _closed_vocab() -> TagVocab:
    schema = parse_authored_schema(_CLOSED_SCHEMA)[0]
    return TagVocab.build(schema.tags, closed=True, authored=True)


def test_batch_run_is_indistinguishable_from_sync_under_closed_vocab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    answer = make_answer()
    sync = _run(
        tmp_path / "sync",
        answer,
        monkeypatch,
        batch=None,
        vocab=_closed_vocab(),
        schema_text=_CLOSED_SCHEMA,
    )
    batch = _run(
        tmp_path / "batch",
        answer,
        monkeypatch,
        batch=fake_provider,
        vocab=_closed_vocab(),
        schema_text=_CLOSED_SCHEMA,
    )

    assert batch["outputs"] == sync["outputs"]
    assert set(sync["outputs"]) == set(_PAGES)
    assert "Greeting" in sync["outputs"]["bravo.pdf"]
    assert batch["files"] == sync["files"]  # blocks, raw windows, label raws, roster
    assert any(name.endswith("_blocks.json") for name in sync["files"])
    assert batch["schema_json"] == sync["schema_json"]
    assert batch["errors"] == sync["errors"] == {}

    # Same rows, same order-independent content; only the tier differs.
    assert _comparable(batch["rows"]) == _comparable(sync["rows"])
    assert {row["tier"] for row in sync["rows"]} == {"standard"}
    assert {row["tier"] for row in batch["rows"]} == {"batch"}
    transcribe_rows = [r for r in batch["rows"] if r["operation"] == "transcribe"]
    assert sorted(r["context"]["doc"] for r in transcribe_rows) == sorted(_PAGES)

    stats = batch["stats"]
    assert stats["transcribe"]["requests"] == sum(_PAGES.values())
    assert stats["transcribe"]["batch_ok"] == sum(_PAGES.values())
    assert stats["transcribe"]["waves"] == max(_PAGES.values())
    assert stats["label"]["requests"] == len(_PAGES)  # labeling batched: roster fixed
    assert fake_provider.built and all(b.submitted for b in fake_provider.built)


def test_no_batch_label_keeps_labeling_synchronous_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    """``label=False`` (``--no-batch-label``): open-vocabulary labeling runs with
    synchronous calls; everything else still batches."""
    answer = make_answer()
    sync = _run(tmp_path / "sync", answer, monkeypatch, batch=None)
    logs: list[str] = []
    batch = _run(
        tmp_path / "batch", answer, monkeypatch, batch=fake_provider, logs=logs, label=False
    )

    assert batch["outputs"] == sync["outputs"]
    assert batch["files"] == sync["files"]
    assert _comparable(batch["rows"]) == _comparable(sync["rows"])
    assert batch["stats"]["label"] == {"skipped": BATCH_LABEL_OFF, "mode": "sync"}
    assert any("labeling stays synchronous under batch mode" in line for line in logs)
    # Transcription and roster planning (draft, then refine) batched; the
    # labeling calls stay standard-tier, so the pass's row splits by tier.
    assert batch["stats"]["plan"]["requests"] == batch["stats"]["plan"]["batch_ok"] == 2
    assert batch["stats"]["plan"]["waves"] == 2
    tiers = {(row["operation"], row["tier"]) for row in batch["rows"]}
    assert ("transcribe", "batch") in tiers and ("transcribe", "standard") not in tiers
    assert {t for op, t in tiers if op == "label"} == {"batch", "standard"}


def _coining(answer: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
    """*answer*, except labeling tags every block ``Salutation`` — a concept
    the planner never named, so the run has to describe it."""

    def coin(kwargs: dict[str, Any]) -> Any:
        if _system_text(kwargs) == label_mod.SYSTEM_PROMPT:
            ids = sorted(set(_BLOCK_ID_RE.findall(_user_text(kwargs))))
            return _reply(json.dumps({"labels": {i: {"concept": "Salutation"} for i in ids}}))
        if _system_text(kwargs) == prompt("describe_concepts"):
            return _reply(json.dumps({"descriptions": {"Salutation": "an opening greeting"}}))
        return answer(kwargs)

    return coin


def test_concept_descriptions_batch_and_match_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    """A concept coined during labeling is described by one more call, which
    batches as its own one-unit stage: same request, same schema.json."""
    answer = _coining(make_answer())
    sync = _run(tmp_path / "sync", answer, monkeypatch, batch=None)
    batch = _run(tmp_path / "batch", answer, monkeypatch, batch=fake_provider)

    assert batch["outputs"] == sync["outputs"]
    assert batch["files"] == sync["files"]
    assert batch["schema_json"] == sync["schema_json"]
    assert b"an opening greeting" in (sync["schema_json"] or b"")
    assert _comparable(batch["rows"]) == _comparable(sync["rows"])
    describe = [
        r.kwargs
        for backend in fake_provider.built
        for wave in backend.submitted
        for r in wave
        if _system_text(r.kwargs) == prompt("describe_concepts")
    ]
    assert len(describe) == 1
    assert batch["stats"]["describe"]["requests"] == batch["stats"]["describe"]["batch_ok"] == 1


def test_extend_schema_gap_planning_batches_and_matches_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    """``--extend-schema``'s gap-planning call batches as a one-unit stage; the
    vocabulary it bounds, the DGML and the cache files are the sync run's."""
    base = make_answer()

    def answer(kwargs: dict[str, Any]) -> Any:
        if _system_text(kwargs) == prompt("plan_gaps_system"):
            return _reply(json.dumps({"concepts": {"Signature": "the signing line"}}))
        return base(kwargs)

    schema = parse_authored_schema(_CLOSED_SCHEMA)[0]
    vocab = TagVocab.build(schema.tags, closed=False, authored=True)
    assert vocab.extends
    logs_s: list[str] = []
    logs_b: list[str] = []
    sync = _run(
        tmp_path / "sync",
        answer,
        monkeypatch,
        batch=None,
        vocab=vocab,
        schema_text=_CLOSED_SCHEMA,
        logs=logs_s,
    )
    batch = _run(
        tmp_path / "batch",
        answer,
        monkeypatch,
        batch=fake_provider,
        vocab=TagVocab.build(schema.tags, closed=False, authored=True),
        schema_text=_CLOSED_SCHEMA,
        logs=logs_b,
    )
    assert any("1 planned tag(s)" in line for line in logs_s)
    assert batch["outputs"] == sync["outputs"]
    assert batch["files"] == sync["files"]
    assert batch["schema_json"] == sync["schema_json"]
    assert _comparable(batch["rows"]) == _comparable(sync["rows"])
    assert batch["stats"]["plan_gaps"]["requests"] == batch["stats"]["plan_gaps"]["batch_ok"] == 1
    gap_rows = [r for r in batch["rows"] if r["operation"] == "label" and r["tier"] == "batch"]
    assert gap_rows  # the gap-planning scope's row, billed at batch price


def test_a_pilot_staged_run_labels_per_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    """An unseeded run over more documents than the pilot holds is never
    labeled all at once (its roster changes between the two stages), so it
    labels one document per batch stage — and says so."""
    monkeypatch.setattr(label_mod, "_PILOT_MAX_DOCS", 1)  # two documents: staged
    logs: list[str] = []
    batch = _run(tmp_path / "batch", make_answer(), monkeypatch, batch=fake_provider, logs=logs)
    assert any("pilot stage" in line for line in logs)
    label = batch["stats"]["label"]
    assert label["mode"] == "per-document" and label["documents"] == len(_PAGES)


@pytest.mark.parametrize("closed", [False, True], ids=["open", "closed"])
def test_no_batch_label_labels_synchronously_under_every_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider, closed: bool
) -> None:
    """``label=False`` turns batch labeling off even where it would batch every
    document at once (a closed vocabulary): no labeling request reaches the
    batch endpoint, the output is the synchronous run's, and the stage reports
    ``mode: "sync"``."""
    kwargs: dict[str, Any] = (
        {"vocab": _closed_vocab(), "schema_text": _CLOSED_SCHEMA} if closed else {}
    )
    sync = _run(tmp_path / "sync", make_answer(), monkeypatch, batch=None, **kwargs)
    batch = _run(
        tmp_path / "batch", make_answer(), monkeypatch, batch=fake_provider, label=False, **kwargs
    )
    assert batch["outputs"] == sync["outputs"]
    assert batch["files"] == sync["files"]
    assert batch["stats"]["label"] == {"skipped": BATCH_LABEL_OFF, "mode": "sync"}
    batched = [r for b in fake_provider.built for w in b.submitted for r in w]
    assert not [r for r in batched if _system_text(r.kwargs) == label_mod.SYSTEM_PROMPT]
    assert batched  # transcription still batched


def _rich_answer(kwargs: dict[str, Any]) -> Any:
    """:func:`make_answer` with every labeling path in play: page 1 of each
    document opens with a heading the first labeling pass leaves untagged (so
    the section retry runs), a listing of more than two blocks comes back
    unparseable (so the chunk bisects), and concepts are coined from the text
    (so each document's labels change the roster the next one is shown)."""
    system, user = _system_text(kwargs), _user_text(kwargs)
    if system == transcribe_mod.SYSTEM_PROMPT:
        match = _HEADER_RE.search(user)
        assert match is not None, user
        first, _last, total = (int(g) for g in match.groups())
        blocks: list[dict[str, Any]] = [
            {"structure": "p", "text": f"Fee{total} applies on page {first}."},
            {"structure": "p", "text": f"Term{first} lasts {total} months."},
        ]
        if first == 1:
            blocks.insert(0, {"structure": "heading", "text": f"Part {total}", "level": 2})
        return _reply(json.dumps({"continues": "", "blocks": blocks}))
    if system == label_mod.SYSTEM_PROMPT:
        lines = _LISTING_RE.findall(user)
        if "left unlabeled" in user:
            labels = {i: {"concept": "PartTitle"} for i, _s, _t in lines}
            return _reply(json.dumps({"labels": labels}))
        if len(lines) > 2:
            return _reply("{ not valid json ,,,")
        labels = {
            i: {"concept": t.split()[0].rstrip("0123456789")}
            for i, kind, t in lines
            if kind != "heading"
        }
        return _reply(json.dumps({"labels": labels}))
    return make_answer()(kwargs)


_LISTING_RE = re.compile(r"(?m)^(b\d{4}) (\w+)(?: \[[^]]*\])?: (.*)$")


def _label_requests(sent: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """``(document, canonical request)`` for every labeling request, in order."""
    out = []
    for kwargs in sent:
        if _system_text(kwargs) == label_mod.SYSTEM_PROMPT:
            doc = re.search(r"== (.+?) ==", _user_text(kwargs))
            assert doc is not None
            out.append((doc.group(1), json.dumps(kwargs, sort_keys=True, default=repr)))
    return out


def _merged_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Rows per operation with tier-split parts summed back together: what a
    batch run must agree with the sync run on (the split itself is the tier)."""
    numeric = ("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        context = {k: v for k, v in row["context"].items() if k != "tier_split"}
        key = row["operation"] + json.dumps(context, sort_keys=True)
        if key not in out:
            out[key] = dict.fromkeys(numeric, 0)
        for k in numeric:
            out[key][k] += row[k] or 0
    return {k: {f: pytest.approx(v) for f, v in vals.items()} for k, vals in out.items()}


def _by_doc(reqs: list[tuple[str, str]]) -> list[tuple[str, list[str]]]:
    """Requests grouped into consecutive runs per document (each run sorted)."""
    grouped: list[tuple[str, list[str]]] = []
    for doc, req in reqs:
        if not grouped or grouped[-1][0] != doc:
            grouped.append((doc, []))
        grouped[-1][1].append(req)
    return [(doc, sorted(r)) for doc, r in grouped]


@pytest.mark.parametrize("pilot", [False, True], ids=["unstaged", "pilot-staged"])
def test_per_document_batch_labeling_is_indistinguishable_from_sync_under_open_vocab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider, pilot: bool
) -> None:
    """Batch labeling (the default) under an open vocabulary: through the batch
    endpoint, one document per stage, ends byte-identical to the sync run —
    with several chunks per document, a bisect, a section retry, a growing
    roster and (parametrized) the pilot stage."""
    monkeypatch.setattr(label_mod, "_MAX_CHUNK_CHARS", 70)  # several chunks per document
    pages = {"alpha.pdf": 1, "bravo.pdf": 3, "charlie.pdf": 2}
    monkeypatch.setattr(sys.modules[__name__], "_PAGES", pages)
    if pilot:
        monkeypatch.setattr(label_mod, "_PILOT_MAX_DOCS", 2)
    sent: list[dict[str, Any]] = []

    def answer(kwargs: dict[str, Any]) -> Any:
        sent.append(kwargs)
        return _rich_answer(kwargs)

    sync = _run(tmp_path / "sync", answer, monkeypatch, batch=None)
    sync_label = _label_requests(sent)
    sent.clear()
    logs: list[str] = []
    batch = _run(
        tmp_path / "batch",
        answer,
        monkeypatch,
        batch=fake_provider,
        logs=logs,
    )
    batch_requests = [
        r.kwargs for backend in fake_provider.built for wave in backend.submitted for r in wave
    ]
    batch_label = _label_requests(batch_requests)

    # The run exercised what it claims to.
    assert any("left unlabeled" in req for _doc, req in sync_label)  # section retry
    assert any(name.endswith("_unparseable.txt") for name in sync["files"])  # bisect
    assert any("pilot stage" in line for line in logs) is pilot

    # Same documents in the same order, the same requests per document.
    assert _by_doc(batch_label) == _by_doc(sync_label)
    assert len(_by_doc(sync_label)) == len(pages)  # each document labeled once, contiguously

    assert batch["outputs"] == sync["outputs"]
    assert batch["files"] == sync["files"]  # label raws and inputs, roster, blocks
    assert batch["schema_json"] == sync["schema_json"]
    assert batch["errors"] == sync["errors"] == {}
    assert _merged_rows(batch["rows"]) == _merged_rows(sync["rows"])
    assert "batch" in {r["tier"] for r in batch["rows"] if r["operation"] == "label"}

    stats = batch["stats"]["label"]
    assert stats["mode"] == "per-document" and stats["documents"] == len(pages)
    assert stats["requests"] == stats["batch_ok"] == len(batch_label)
    assert stats["waves"] >= len(pages)  # at least one wave per document
    assert stats["waves"] < len(batch_label)  # a document's chunks shared a wave


def test_per_document_labeling_composes_with_batched_planning_and_descriptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    """Every Pass B call batched at once: roster planning (draft, refine), each
    document's labeling as its own stage, then the description of the concept
    labeling coined — in that order, each stage reported, the whole labeling
    pass one ``batch`` row (no tier split), and every byte the sync run's."""
    answer = _coining(make_answer())
    sync = _run(tmp_path / "sync", answer, monkeypatch, batch=None)
    batch = _run(tmp_path / "batch", answer, monkeypatch, batch=fake_provider)
    assert batch["outputs"] == sync["outputs"]
    assert "Salutation" in sync["outputs"]["bravo.pdf"]
    assert batch["files"] == sync["files"]
    assert batch["schema_json"] == sync["schema_json"]
    assert b"an opening greeting" in (batch["schema_json"] or b"")
    assert _comparable(batch["rows"]) == _comparable(sync["rows"])
    label_rows = [r for r in batch["rows"] if r["operation"] == "label"]
    assert len(label_rows) == 1 and label_rows[0]["tier"] == "batch"
    assert "tier_split" not in label_rows[0]["context"]

    names = {
        transcribe_mod.SYSTEM_PROMPT: "transcribe",
        label_mod.PLAN_SYSTEM_PROMPT: "plan",
        label_mod.SYSTEM_PROMPT: "label",
        prompt("describe_concepts"): "describe",
    }
    waves = [
        {names[_system_text(r.kwargs)] for r in wave}
        for backend in fake_provider.built
        for wave in backend.submitted
    ]
    stage_waves = [w for w in waves if w != {"transcribe"}]
    assert stage_waves == [{"plan"}, {"plan"}, *([{"label"}] * len(_PAGES)), {"describe"}]

    stats = batch["stats"]
    assert set(stats) >= {"transcribe", "plan", "label", "describe"}
    assert stats["plan"]["batch_ok"] == stats["plan"]["requests"] == 2
    assert stats["label"]["mode"] == "per-document"
    assert stats["label"]["documents"] == len(_PAGES)
    assert stats["describe"]["batch_ok"] == stats["describe"]["requests"] == 1
    assert all(s.get("sync_fallbacks", 0) == 0 for s in stats.values())


def test_batch_labeling_batches_a_closed_vocab_all_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    batch = _run(
        tmp_path / "batch",
        make_answer(),
        monkeypatch,
        batch=fake_provider,
        vocab=_closed_vocab(),
        schema_text=_CLOSED_SCHEMA,
    )
    label = batch["stats"]["label"]
    assert label["mode"] == "all-at-once"
    assert label["requests"] == len(_PAGES) and label["waves"] == 1


def test_per_document_label_stage_failure_leaves_every_document_unlabeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    """A batch-level failure on the first document's stage is a label error for
    it and for every later document (no further attempt), not a lost run."""

    def factory(_cfg: Any) -> FakeBackend:
        backend = _Outage(fake_provider.script, provider="anthropic")
        # Executors in creation order: transcription, roster planning, then
        # labeling — the third and later ones lose the provider.
        backend.accept = 0 if len(fake_provider.built) >= 2 else 10**6
        fake_provider.built.append(backend)
        return backend

    register_backend("anthropic", factory)
    batch = _run(tmp_path / "batch", make_answer(), monkeypatch, batch=fake_provider)
    assert set(batch["outputs"]) == set(_PAGES)
    assert all("Greeting" not in xml for xml in batch["outputs"].values())
    assert batch["stats"]["label"]["documents"] == 1  # one try, then every doc gives up
    assert batch["stats"]["plan"]["batch_ok"] == 2  # planning was served; labeling failed


def test_batch_options_label_is_a_strict_bool() -> None:
    assert BatchOptions().label is True
    for bad in ("sequential", "true", 1, None):
        with pytest.raises(GenerationConfigInvalid, match=r"BatchOptions\.label must be"):
            BatchOptions(label=bad)  # type: ignore[arg-type]


def test_generation_batch_label_config_key(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dgml_core.generation.config import load_generation_batch_label

    config = workspace.root.joinpath("config.toml")
    assert load_generation_batch_label(workspace) is True  # absent: batch labeling on
    for raw, expected in (("false", False), ("true", True), ('"no"', False)):
        config.write_text(f"[generation]\nbatch_label = {raw}\n")
        assert load_generation_batch_label(workspace) is expected
    for bad in ('"sequential"', '"sync"', "1", '"parallel"'):
        config.write_text(f"[generation]\nbatch_label = {bad}\n")
        with pytest.raises(GenerationConfigInvalid, match=r"generation\.batch_label must be"):
            load_generation_batch_label(workspace)
    config.write_text("")
    monkeypatch.setenv("DGML_GENERATION__BATCH_LABEL", " False ")
    assert load_generation_batch_label(workspace) is False


def test_failed_transcription_is_dropped_with_the_sync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    answer = make_answer(fail_total=3)  # bravo.pdf's every window fails
    sync = _run(tmp_path / "sync", answer, monkeypatch, batch=None)
    batch = _run(tmp_path / "batch", answer, monkeypatch, batch=fake_provider)

    assert set(sync["outputs"]) == set(batch["outputs"]) == {"alpha.pdf"}
    assert batch["errors"] == sync["errors"]
    assert batch["errors"]["bravo.pdf"]  # a short reason, as the sync path records
    assert batch["outputs"] == sync["outputs"]
    assert _comparable(batch["rows"]) == _comparable(sync["rows"])
    assert batch["stats"]["transcribe"]["sync_fallbacks"] >= 1


def test_cached_documents_make_no_request_and_write_no_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    answer = make_answer()
    first = _run(tmp_path / "run", answer, monkeypatch, batch=fake_provider)
    assert [r for r in first["rows"] if r["operation"] == "transcribe"]

    def no_transcription(kwargs: dict[str, Any]) -> Any:
        assert _system_text(kwargs) != transcribe_mod.SYSTEM_PROMPT, "transcribed a cached doc"
        return answer(kwargs)

    # Same cache dir: every document's _blocks.json is already there.
    ws_usage = tmp_path / "run" / "ws" / "usage.jsonl"
    before = ws_usage.read_text(encoding="utf-8").count("\n")
    second = _run(tmp_path / "run", no_transcription, monkeypatch, batch=fake_provider)
    new_rows = second["rows"][before:]
    assert not [r for r in new_rows if r["operation"] == "transcribe"]
    assert second["stats"]["transcribe"]["requests"] == 0  # every document was cached
    assert second["outputs"] == first["outputs"]


class _Outage(FakeBackend):
    """Accepts its first ``accept`` batches, then the provider goes away."""

    accept = 1

    def submit(self, requests: Any) -> Any:
        if len(self.submitted) >= self.accept:
            raise RuntimeError("provider down")
        return super().submit(requests)


def test_transcribe_stage_failure_keeps_the_documents_that_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    """Wave 1 (every document's first window) is served; wave 2 fails at
    batch level. alpha.pdf (one window) finished in wave 1 and is kept;
    bravo.pdf is reported failed, not the whole run."""

    def factory(_cfg: Any) -> FakeBackend:
        backend = _Outage(fake_provider.script, provider="anthropic")
        fake_provider.built.append(backend)
        return backend

    register_backend("anthropic", factory)
    batch = _run(tmp_path / "batch", make_answer(), monkeypatch, batch=fake_provider)
    assert set(batch["outputs"]) == {"alpha.pdf"}
    assert "provider down" in batch["errors"]["bravo.pdf"]


def test_label_stage_failure_keeps_every_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_provider: _Provider
) -> None:
    """A batch-level failure of the labeling stage is a per-document label
    error (as for an unreachable label model), not a lost run."""

    def factory(_cfg: Any) -> FakeBackend:
        backend = _Outage(fake_provider.script, provider="anthropic")
        if fake_provider.built:  # the 2nd executor is the label stage's: it gets nothing
            backend.accept = 0
        else:
            backend.accept = 10**6
        fake_provider.built.append(backend)
        return backend

    register_backend("anthropic", factory)
    batch = _run(
        tmp_path / "batch",
        make_answer(),
        monkeypatch,
        batch=fake_provider,
        vocab=_closed_vocab(),
        schema_text=_CLOSED_SCHEMA,
    )
    assert set(batch["outputs"]) == set(_PAGES)
    assert all("Greeting" not in xml for xml in batch["outputs"].values())


def test_generation_batch_config_key(workspace: Workspace) -> None:
    assert load_generation_batch(workspace) is False
    workspace.root.joinpath("config.toml").write_text("[generation]\nbatch = true\n")
    assert load_generation_batch(workspace) is True
    workspace.root.joinpath("config.toml").write_text("[generation]\nbatch = false\n")
    assert load_generation_batch(workspace) is False
    for bad in ('"maybe"', "1.5", "[true]"):
        workspace.root.joinpath("config.toml").write_text(f"[generation]\nbatch = {bad}\n")
        with pytest.raises(
            GenerationConfigInvalid, match=r"generation\.batch must be true or false"
        ):
            load_generation_batch(workspace)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("TRUE", True), (" yes ", True), ("1", True)]
    + [("false", False), ("False", False), ("no", False), ("0", False)],
)
def test_generation_batch_env_var_strings(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    """The DGML_ env-var layer delivers strings, never bools."""
    monkeypatch.setenv("DGML_GENERATION__BATCH", raw)
    assert load_generation_batch(workspace) is expected


def test_generation_batch_env_var_rejects_a_non_boolean(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DGML_GENERATION__BATCH", "on-ish")
    with pytest.raises(GenerationConfigInvalid, match="on-ish"):
        load_generation_batch(workspace)


def test_preflight_and_decision_share_one_batchability_rule() -> None:
    """The CLI pre-flight (vocabulary only, no roster yet) and the pipeline's
    decision (vocabulary and roster) come from one rule, so a vocabulary the
    pre-flight clears is never refused later for a vocabulary reason, and a
    vocabulary it refuses is never batched."""
    from dgml_core.generation.label import (
        _seed_entries_from_schema,
        is_batchable_vocab,
        vocab_batch_blocker,
    )
    from dgml_core.generation.pipeline import ROSTER_NOT_FIXED, unbatchable_label_reason
    from dgml_core.generation.vocab import OPEN_VOCAB

    schema = parse_authored_schema(_CLOSED_SCHEMA)[0]
    closed = _closed_vocab()
    extends = TagVocab.build(schema.tags, closed=False, authored=True)
    frozen_roster = _seed_entries_from_schema(schema)
    for vocab in (OPEN_VOCAB, extends, closed):
        blocker = vocab_batch_blocker(vocab)
        assert unbatchable_label_reason(vocab) == blocker  # the pre-flight's answer
        if blocker is not None:
            assert not is_batchable_vocab(vocab, frozen_roster)
            assert unbatchable_label_reason(vocab, frozen_roster) == blocker
    # Closed vocabulary: pre-flight says yes; the decision then depends on the roster.
    assert unbatchable_label_reason(closed) is None
    assert unbatchable_label_reason(closed, frozen_roster) is None
    assert is_batchable_vocab(closed, frozen_roster)
    unfrozen = {name: label_mod.RosterEntry(description="x") for name in frozen_roster}
    assert unbatchable_label_reason(closed, unfrozen) == ROSTER_NOT_FIXED
    assert not is_batchable_vocab(closed, unfrozen)


def test_batch_package_stays_off_the_default_import_path() -> None:
    """Upstream #160 keeps the lightweight import path free of litellm; batch
    mode must not add ``dgml_core.batch`` (which loads every provider backend)
    to any import that does not ask for it. Checked in a fresh interpreter so
    this process's own imports cannot mask a regression."""
    code = (
        "import sys\n"
        "import dgml_core, dgml_core.generation\n"
        "light = ('litellm' in sys.modules, 'dgml_core.batch' in sys.modules)\n"
        "import dgml_core.generation.pipeline, dgml.cli\n"
        "heavy = 'dgml_core.batch' in sys.modules\n"
        "print(light[0], light[1], heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    litellm_on_light_path, batch_on_light_path, batch_on_pipeline_path = result.stdout.split()
    assert litellm_on_light_path == "False"
    assert batch_on_light_path == "False"
    assert batch_on_pipeline_path == "False"
