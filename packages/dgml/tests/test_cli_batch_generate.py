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

"""``dgml docset generate --batch`` end to end, through the real pipeline.

Every scenario runs the same command twice over one deterministic model — once
synchronously, once with ``--batch`` against a :class:`FakeBackend` standing
in for Anthropic Message Batches — and asserts the two leave the same DGML,
cache files and usage rows (apart from ``tier``), and that the batch run's
payload reports each stage in its ``batch`` block.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core import layout, style_llm
from dgml_core.batch import FakeBackend, fake_model_response, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.docsets import DocSetStore
from dgml_core.generation import document as document_mod
from dgml_core.generation import label as label_mod
from dgml_core.generation import links as links_mod
from dgml_core.generation import transcribe as transcribe_mod
from dgml_core.generation.prompts import get as prompt
from dgml_core.storage import Workspace
from dgml_core.usage import read_events

from .test_cli import _init_ws, _read_stderr, _read_stdout, _seed_file_dir, _write_ws_config

# ---- a deterministic model ----------------------------------------------------------

_PAGES = {"alpha.pdf": 1, "bravo.pdf": 3}
_HEADER_RE = re.compile(r"pages (\d+)-(\d+) of (\d+)")
_BLOCK_ID_RE = re.compile(r"\bb\d{4}\b")
_MODELS = {"model": "anthropic/claude-haiku-4-5", "label_model": "anthropic/claude-sonnet-4-6"}


def _system(kwargs: dict[str, Any]) -> str:
    content = kwargs["messages"][0]["content"]
    if isinstance(content, str):
        return content
    return "".join(str(b.get("text", "")) for b in content)


def _user(kwargs: dict[str, Any]) -> str:
    content = kwargs["messages"][1]["content"]
    if isinstance(content, str):
        return content
    return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, dict))


def _reply(text: str) -> Any:
    return fake_model_response(
        text, usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, cost=0.01
    )


def _answer(kwargs: dict[str, Any]) -> Any:
    """Every reply depends only on its request's bytes."""
    system, user = _system(kwargs), _user(kwargs)
    if system == transcribe_mod.SYSTEM_PROMPT:
        match = _HEADER_RE.search(user)
        assert match is not None, user
        first, _last, total = (int(g) for g in match.groups())
        blocks = [{"structure": "p", "text": f"Hello from doc{total} page{first}."}]
        return _reply(json.dumps({"continues": "", "blocks": blocks}))
    if system == label_mod.SYSTEM_PROMPT:
        ids = sorted(set(_BLOCK_ID_RE.findall(user)))
        return _reply(json.dumps({"labels": {i: {"concept": "Greeting"} for i in ids}}))
    if system == label_mod.PLAN_SYSTEM_PROMPT:
        return _reply(json.dumps({"concepts": {"Greeting": "a greeting line"}}))
    if system == prompt("describe_concepts"):
        return _reply(json.dumps({"descriptions": {}}))
    if system == links_mod.SYSTEM_PROMPT:
        # One candidate, so the verify wave runs too.
        link = {"subject": "e0001", "object": "e0002", "predicate": "references"}
        return _reply(json.dumps({"links": [link]}))
    if system == links_mod.VERIFY_SYSTEM_PROMPT:
        return _reply(json.dumps({"verdicts": [{"i": 0, "keep": True}]}))
    if system == style_llm._SYSTEM_PROMPT:
        return _reply(json.dumps({"styles": [{"index": 0, "style": "font-weight: bold"}]}))
    raise AssertionError(f"unexpected request: {system[:60]!r}")


# ---- the provider and the workspace ---------------------------------------------------


class _Provider:
    """One FakeBackend for the whole test, answering with *answer*."""

    def __init__(self) -> None:
        self.backend: FakeBackend | None = None

    def install(self, answer: Callable[[dict[str, Any]], Any]) -> FakeBackend:
        backend = FakeBackend(lambda request: answer(request.kwargs), provider="anthropic")
        self.backend = backend
        register_backend("anthropic", lambda _cfg: backend)
        return backend


@pytest.fixture
def provider() -> Iterator[_Provider]:
    saved = dict(batch_registry._REGISTRY)
    try:
        yield _Provider()
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


@pytest.fixture(autouse=True)
def pdf_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Page counting and slicing stubbed: a document's pages come from its name."""
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


def _ws_args(ws: Path) -> list[str]:
    return ["--workspace", str(ws)]


def _seed(
    root: Path, capsys: pytest.CaptureFixture[str], config: dict[str, Any] | None = None
) -> tuple[Path, str]:
    """A workspace with one DocSet holding the two documents of :data:`_PAGES`.
    With a ``style`` section the files are OCR files whose page words are the
    text their transcription will hold, so grounding places them and the
    image-style pass runs."""
    ws = root / "ws"
    _init_ws(ws)
    capsys.readouterr()
    config = config or {"generation": _MODELS}
    _write_ws_config(ws, config)
    main(_ws_args(ws) + ["docset", "create", "--name", "Letters"])
    did = str(_read_stdout(capsys)["id"])
    wsx = Workspace(root=ws)
    store = DocSetStore(wsx)
    ocr = "style" in config
    for index, (name, pages) in enumerate(_PAGES.items()):
        fid = f"fgen0000000{index}"
        _seed_file_dir(ws, fid, pages=pages, pdf_name=name)
        if ocr:
            record = wsx.docs.get_doc("files", fid)
            assert record is not None
            wsx.docs.put_doc("files", fid, {**record, "text_mode": "ocr"})
        for n in range(1, pages + 1):
            words = ["Hello", "from", f"doc{pages}", f"page{n}."] if ocr else []
            boxes = [
                {"t": w, "l": [100 + 90 * i, 100, 180 + 90 * i, 120]} for i, w in enumerate(words)
            ]
            page = {"file_id": fid, "page": n, "width": 1000, "height": 1000, "words": boxes}
            wsx.blobs.put_blob(layout.file_page_text_key(fid, n), json.dumps(page).encode())
        store.add_file(did, fid)
    return ws, did


def _argv(ws: Path, did: str, *extra: str) -> list[str]:
    return _ws_args(ws) + [
        "--debug",
        "docset",
        "generate",
        did,
        "--no-coverage",
        "--max-parallel-calls",
        "1",
        "--window-size",
        "1",
        *extra,
    ]


_BATCH = ("--batch", "--batch-poll-interval", "0.01")


def _rows(ws: Path) -> list[dict[str, Any]]:
    """Usage rows minus timing, the tier, and each workspace's own DocSet id,
    with every scope a batch run split by tier folded back into one row."""
    sums = ("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens")
    rows: list[dict[str, Any]] = []
    joined: dict[str, dict[str, Any]] = {}
    for row in read_events(Workspace(root=ws)):
        row = {k: v for k, v in row.items() if k not in {"at", "duration_s", "tier"}}
        context = dict(row.get("context") or {})
        if "docset_id" in context:
            context["docset_id"] = "<ds>"
        split = context.pop("tier_split", False)
        row["context"] = context
        if not split:
            rows.append(row)
            continue
        key = json.dumps({k: v for k, v in row.items() if k not in sums}, sort_keys=True)
        if key not in joined:
            joined[key] = row
            rows.append(row)
            continue
        for name in sums:
            joined[key][name] = (joined[key].get(name) or 0) + (row.get(name) or 0)
    for row in rows:
        if row.get("cost_usd") is not None:
            row["cost_usd"] = round(row["cost_usd"], 9)
    return sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))


def _state(ws: Path, did: str) -> dict[str, Any]:
    """Everything a generate run leaves behind that the two modes must share."""
    wsx = Workspace(root=ws)
    docset_dir = ws / layout.DOCSETS_DIR / did
    files: dict[str, Any] = {}
    for p in sorted(docset_dir.rglob("*")):
        if not p.is_file() or p.name in ("docset.json", "assignment.json"):
            continue
        data: Any = p.read_bytes()
        if p.name.endswith(".grounding_stats.json"):
            volatile = ("completed_at", "source", "output")
            data = {k: v for k, v in json.loads(data).items() if k not in volatile}
        files[str(p.relative_to(docset_dir))] = data
    outputs = {}
    for fid in DocSetStore(wsx).list_files(did):
        name = wsx.docs.get_doc("files", fid)["original_filename"]  # type: ignore[index]
        key = layout.dgml_xml_key(did, fid, Path(name).stem)
        outputs[name] = wsx.blobs.get_blob(key)
    return {"files": files, "outputs": outputs, "rows": _rows(ws)}


def _payload_shape(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload minus what names the workspace (its DocSet id)."""
    out = {k: v for k, v in payload.items() if k not in ("docset_id", "output_key")}
    out["results"] = [{k: v for k, v in r.items() if k != "output"} for r in out["results"]]
    return out


def _run_both(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    *extra: str,
    config: dict[str, Any] | None = None,
    batch_extra: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[Any]]:
    """(sync payload, sync state, batch payload, batch state, batch-run sync calls)."""
    ws_s, did_s = _seed(tmp_path / "sync", capsys, config)
    with patch("litellm.completion", side_effect=lambda **kw: _answer(kw)):
        assert main(_argv(ws_s, did_s, *extra)) == 0
    sync_payload = _read_stdout(capsys)

    ws_b, did_b = _seed(tmp_path / "batch", capsys, config)
    provider.install(_answer)
    batch_sync: list[dict[str, Any]] = []

    def record(**kwargs: Any) -> Any:
        batch_sync.append(kwargs)
        return _answer(kwargs)

    with patch("litellm.completion", side_effect=record):
        assert main(_argv(ws_b, did_b, *extra, *_BATCH, *batch_extra)) == 0
    batch_payload = _read_stdout(capsys)
    return (
        sync_payload,
        _state(ws_s, did_s),
        batch_payload,
        _state(ws_b, did_b),
        batch_sync,
    )


_COST_FIELDS = ("cost_usd", "standard_cost_usd", "saved_usd")


def test_open_vocab_batches_every_stage_and_labels_per_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    """The default under --batch: an open vocabulary labels one document per
    batch stage, in the sync order, and ends byte-identical to the sync run."""
    sync_payload, sync_state, payload, state, batch_sync = _run_both(tmp_path, capsys, provider)

    assert "batch" not in sync_payload
    assert state["outputs"] == sync_state["outputs"]
    assert b"Greeting" in state["outputs"]["bravo.pdf"]
    assert state["files"] == sync_state["files"]
    assert state["rows"] == sync_state["rows"]
    rest = {k: v for k, v in payload.items() if k != "batch"}
    assert _payload_shape(rest) == _payload_shape(sync_payload)

    stages = payload["batch"]["stages"]
    assert set(stages) == {"transcribe", "plan", "label", "links"}
    label = stages["label"]
    assert label["mode"] == "per-document"
    assert label["documents"] == len(_PAGES)
    assert label["waves"] >= len(_PAGES)  # at least one round trip per document
    assert label["requests"] == label["batch_ok"] > 0
    assert label["sync_fallbacks"] == 0
    assert label["cost_usd"] == pytest.approx(label["standard_cost_usd"] / 2)
    assert batch_sync == []  # every request went through the batch


def test_no_batch_label_batches_every_stage_but_labeling_and_matches_sync(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    """``--no-batch-label``: today's behavior — labeling stays synchronous."""
    sync_payload, sync_state, payload, state, batch_sync = _run_both(
        tmp_path, capsys, provider, batch_extra=("--no-batch-label",)
    )

    assert "batch" not in sync_payload  # no --batch: the payload is unchanged
    assert state["outputs"] == sync_state["outputs"]
    assert b"Greeting" in state["outputs"]["bravo.pdf"]
    assert state["files"] == sync_state["files"]
    assert state["rows"] == sync_state["rows"]
    rest = {k: v for k, v in payload.items() if k != "batch"}
    assert _payload_shape(rest) == _payload_shape(sync_payload)

    block = payload["batch"]
    assert block["enabled"] is True
    stages = block["stages"]
    assert set(stages) == {"transcribe", "plan", "label", "links"}
    for name in ("transcribe", "plan", "links"):
        stage = stages[name]
        assert stage["requests"] == stage["batch_ok"] > 0
        assert stage["sync_fallbacks"] == 0
        assert stage["batch_ids"]
        assert stage["cost_usd"] == pytest.approx(stage["standard_cost_usd"] / 2)
        assert stage["saved_usd"] == pytest.approx(stage["cost_usd"])
    assert stages["transcribe"]["waves"] == max(_PAGES.values())
    assert stages["plan"]["waves"] == 2  # the draft, then the refine turn
    assert stages["links"]["waves"] == 2  # propose, then verify
    assert stages["label"]["mode"] == "sync"
    assert "batch_label = false" in stages["label"]["skipped"]
    # Only labeling reached the synchronous seam.
    assert batch_sync and {_system(c) for c in batch_sync} == {label_mod.SYSTEM_PROMPT}


def test_closed_vocab_labels_every_document_in_one_wave(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"Greeting": "a greeting line"}), encoding="utf-8")
    _sp, sync_state, payload, state, batch_sync = _run_both(
        tmp_path, capsys, provider, "--schema-path", str(schema)
    )

    assert state["outputs"] == sync_state["outputs"]
    assert b"Greeting" in state["outputs"]["alpha.pdf"]
    assert state["files"] == sync_state["files"]
    assert state["rows"] == sync_state["rows"]
    label = payload["batch"]["stages"]["label"]
    assert label["mode"] == "all-at-once"
    assert label["requests"] == label["batch_ok"] == len(_PAGES)
    assert label["waves"] == 1
    assert label["cost_usd"] == pytest.approx(label["standard_cost_usd"] / 2)
    assert batch_sync == []  # every request went through the batch


def test_no_semlinks_reports_the_link_stage_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    _sp, sync_state, payload, state, _calls = _run_both(tmp_path, capsys, provider, "--no-semlinks")
    assert state["outputs"] == sync_state["outputs"]
    assert payload["batch"]["stages"]["links"] == {"skipped": "--no-semlinks"}


_STYLE_CONFIG = {
    "generation": _MODELS,
    "style": {"enabled": True, "model": "anthropic/claude-haiku-4-5"},
}


def test_style_batches_as_one_stage_and_matches_sync(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    _sp, sync_state, payload, state, batch_sync = _run_both(
        tmp_path, capsys, provider, config=_STYLE_CONFIG
    )
    assert all(b"font-weight: bold" in xml for xml in sync_state["outputs"].values())
    assert state["outputs"] == sync_state["outputs"]
    assert state["files"] == sync_state["files"]
    assert state["rows"] == sync_state["rows"]
    assert not [c for c in batch_sync if _system(c) == style_llm._SYSTEM_PROMPT]
    assert provider.backend is not None
    style_waves = [
        wave
        for wave in provider.backend.submitted
        if _system(wave[0].kwargs) == style_llm._SYSTEM_PROMPT
    ]
    assert [len(w) for w in style_waves] == [sum(_PAGES.values())]  # every page, one wave
    stage = payload["batch"]["stages"]["style"]
    assert stage["requests"] == stage["batch_ok"] == sum(_PAGES.values())


@pytest.mark.parametrize(
    ("config", "stage"),
    [
        ({"generation": {**_MODELS, "model": "mistral/mistral-large-latest"}}, "transcribe"),
        ({"generation": {**_MODELS, "label_model": "mistral/mistral-large-latest"}}, "label"),
        (
            {**_STYLE_CONFIG, "style": {"enabled": True, "model": "mistral/pixtral-large-latest"}},
            "style",
        ),
    ],
    ids=["transcribe", "label", "style"],
)
def test_a_stage_without_a_batch_backend_is_rejected_up_front(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    stage: str,
) -> None:
    ws, did = _seed(tmp_path, capsys, config)
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")  # never used: no call is made
    with patch("litellm.completion", side_effect=AssertionError("no model call expected")):
        assert main(_argv(ws, did, *_BATCH)) == 1
    err = _read_stderr(capsys)["error"]
    assert err["code"] == "BATCH_UNAVAILABLE"
    assert f"stage {stage!r}" in err["message"]
    assert not list((ws / layout.DOCSETS_DIR / did).rglob("*.dgml.xml"))


def test_batch_poll_interval_must_be_positive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws, did = _seed(tmp_path, capsys)
    with pytest.raises(SystemExit) as exc:
        main(_argv(ws, did, "--batch", "--batch-poll-interval", "0"))
    assert exc.value.code == 2
