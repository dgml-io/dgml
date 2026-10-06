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

"""Batch job mode through the CLI: ``--no-wait``, ``--job`` and ``dgml batch``.

Every scenario runs twice over the same deterministic model — once as a plain
blocking ``--batch`` run, once as a ``--no-wait`` job driven to completion
with ``dgml batch resume`` — and asserts the two leave the same outputs, cache
files and usage rows. The provider is one :class:`FakeBackend` instance shared
by every run, the way a real provider's batches outlive the process that
submitted them; ``polls_until_ended=1`` makes each batch still be running at
its first status check, so a ``--no-wait`` run pauses after every wave.
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
from dgml_core import layout
from dgml_core.batch import FakeBackend, fake_model_response, list_jobs, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.batch.types import BatchSubmitUncertain
from dgml_core.docsets import DocSetStore
from dgml_core.generation import document as document_mod
from dgml_core.generation import label as label_mod
from dgml_core.generation import links as links_mod
from dgml_core.generation import transcribe as transcribe_mod
from dgml_core.generation.pipeline import BATCH_LABEL_OFF
from dgml_core.generation.prompts import get as prompt
from dgml_core.storage import Workspace
from dgml_core.usage import read_events

from .conftest import needs_gs
from .test_cli import _init_ws, _read_stderr, _read_stdout, _seed_file_dir, _write_ws_config
from .test_cli_batch_extraction import _seed_classify_ws, _seed_extraction, _set_schema

# ---- a deterministic model ----------------------------------------------------------

_PAGES = {"alpha.pdf": 1, "bravo.pdf": 3}
_HEADER_RE = re.compile(r"pages (\d+)-(\d+) of (\d+)")
_BLOCK_ID_RE = re.compile(r"\bb\d{4}\b")


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


def _tool_reply(name: str, arguments: dict[str, Any]) -> Any:
    return fake_model_response(
        "",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
        usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
        cost=0.02,
    )


def _generation_answer(kwargs: dict[str, Any]) -> Any:
    """Every reply depends only on its request's bytes (see the module doc)."""
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
    raise AssertionError(f"unexpected request: {system[:60]!r}")


def _values_answer(_kwargs: dict[str, Any]) -> Any:
    values = {"VendorName": {"text": "Acme", "locations": []}}
    return _tool_reply("submit_values", {"values": values})


# ---- the provider -----------------------------------------------------------------------


class _Provider:
    """One FakeBackend per provider for the whole test, answering with *answer*."""

    def __init__(self) -> None:
        self.backends: dict[str, FakeBackend] = {}

    def install(
        self, provider: str, answer: Callable[[dict[str, Any]], Any], *, polls: int
    ) -> FakeBackend:
        backend = FakeBackend(
            lambda request: answer(request.kwargs), provider=provider, polls_until_ended=polls
        )
        self.backends[provider] = backend
        register_backend(provider, lambda _cfg: backend)
        return backend


@pytest.fixture
def provider() -> Iterator[_Provider]:
    saved = dict(batch_registry._REGISTRY)
    try:
        yield _Provider()
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


@pytest.fixture
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


def _drive_job(
    capsys: pytest.CaptureFixture[str], ws: Path, first: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resume the job *first* named until it finishes; return every pending
    payload seen (the first included) and the final payload."""
    pendings = [first]
    job_id = first["batch_job"]["job_id"]
    for _ in range(20):
        assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
        out = _read_stdout(capsys)
        if "batch_job" not in out:
            return pendings, out
        assert out["batch_job"]["job_id"] == job_id
        pendings.append(out)
    raise AssertionError("the job never completed")


def _rows(ws: Path) -> list[dict[str, Any]]:
    """Usage rows minus timing and each workspace's own DocSet id."""
    drop = {"at", "duration_s"}
    rows = []
    for row in read_events(Workspace(root=ws)):
        row = {k: v for k, v in row.items() if k not in drop}
        if "docset_id" in row.get("context", {}):
            row["context"] = {**row["context"], "docset_id": "<ds>"}
        rows.append(row)
    return sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))


# ---- docset generate ------------------------------------------------------------------


def _seed_generate_ws(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, str]:
    ws = root / "ws"
    _init_ws(ws)
    capsys.readouterr()
    _write_ws_config(
        ws,
        {
            "generation": {
                "model": "anthropic/claude-haiku-4-5",
                "label_model": "anthropic/claude-sonnet-4-6",
            }
        },
    )
    main(_ws_args(ws) + ["docset", "create", "--name", "Letters"])
    did = str(_read_stdout(capsys)["id"])
    wsx = Workspace(root=ws)
    store = DocSetStore(wsx)
    for index, (name, pages) in enumerate(_PAGES.items()):
        fid = f"fgen0000000{index}"
        _seed_file_dir(ws, fid, pages=pages, pdf_name=name)
        for n in range(1, pages + 1):  # grounding reads the page's dimensions
            page = {"file_id": fid, "page": n, "width": 1000, "height": 1000, "words": []}
            wsx.blobs.put_blob(layout.file_page_text_key(fid, n), json.dumps(page).encode())
        store.add_file(did, fid)
    return ws, did


def _generate_argv(ws: Path, did: str, *extra: str) -> list[str]:
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
        "--batch",
        "--batch-poll-interval",
        "0.01",
        *extra,
    ]


def _generate_state(ws: Path, did: str) -> dict[str, Any]:
    """Everything a generate run leaves behind that the two modes must share."""
    wsx = Workspace(root=ws)
    docset_dir = ws / layout.DOCSETS_DIR / did
    cache: dict[str, Any] = {}
    for p in sorted(docset_dir.rglob("*")):
        # Documents (docset.json, assignment.json) carry each workspace's own
        # creation timestamps; they are inputs here, not generate's output.
        if not p.is_file() or p.name in ("docset.json", "assignment.json"):
            continue
        data: Any = p.read_bytes()
        if p.name.endswith(".grounding_stats.json"):
            # Per-run telemetry: when grounding ran and which working copy it
            # read and wrote. Everything else in the sidecar must match.
            volatile = ("completed_at", "source", "output")
            data = {k: v for k, v in json.loads(data).items() if k not in volatile}
        cache[str(p.relative_to(docset_dir))] = data
    outputs = {}
    for fid in DocSetStore(wsx).list_files(did):
        name = wsx.docs.get_doc("files", fid)["original_filename"]  # type: ignore[index]
        key = layout.dgml_xml_key(did, fid, Path(name).stem)
        if wsx.blobs.blob_exists(key):
            outputs[name] = wsx.blobs.get_blob(key)
    return {"files": cache, "outputs": outputs, "rows": _rows(ws)}


def _comparable_payload(payload: dict[str, Any]) -> dict[str, Any]:
    drop = {"batch", "docset_id", "output_key"}
    out = {k: v for k, v in payload.items() if k not in drop}
    out["results"] = sorted(
        ({k: v for k, v in r.items() if k != "output"} for r in payload["results"]),
        key=lambda r: r["source"],
    )
    return out


def test_generate_no_wait_job_matches_a_blocking_batch_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
) -> None:
    """The headline guarantee: a job paused after every wave and resumed with
    `dgml batch resume` ends exactly where one blocking --batch run ends —
    same DGML, same cache files, same usage rows — and each resume advances
    exactly one wave. (``--no-batch-label``: labeling runs synchronously and
    is replayed on every resume; the default per-document batch labeling is
    covered by the next test.)"""
    # Blocking reference run.
    ws_b, did_b = _seed_generate_ws(tmp_path / "blocking", capsys)
    provider.install("anthropic", _generation_answer, polls=0)
    with patch("litellm.completion", side_effect=lambda **kw: _generation_answer(kw)):
        assert main(_generate_argv(ws_b, did_b, "--no-batch-label")) == 0
    blocking_payload = _read_stdout(capsys)
    blocking = _generate_state(ws_b, did_b)

    # The same documents as a --no-wait job.
    ws_j, did_j = _seed_generate_ws(tmp_path / "job", capsys)
    backend = provider.install("anthropic", _generation_answer, polls=1)
    sync_calls: list[dict[str, Any]] = []

    def sync(**kwargs: Any) -> Any:
        sync_calls.append(kwargs)
        return _generation_answer(kwargs)

    with patch("litellm.completion", side_effect=sync):
        assert main(_generate_argv(ws_j, did_j, "--no-wait", "--no-batch-label")) == 0
        first = _read_stdout(capsys)
        assert first["batch_job"]["status"] == "pending"
        assert first["batch_job"]["command"] == "docset generate"
        assert first["batch_job"]["resume"] == f"dgml batch resume {first['batch_job']['job_id']}"
        # Nothing but job state written by a paused run: no DGML, no rows.
        paused = _generate_state(ws_j, did_j)
        assert paused["rows"] == [] and paused["outputs"] == {}
        pendings, final = _drive_job(capsys, ws_j, first)
    job = _generate_state(ws_j, did_j)

    # One wave per call: three transcription windows (bravo.pdf), roster
    # planning's draft and refine, then the link pass's propose and verify
    # waves. Labeling runs synchronously (--no-batch-label) — once, and
    # replayed on every later resume.
    assert len(pendings) == 3 + 2 + 2
    labeling_calls = [c for c in sync_calls if _system(c) == label_mod.SYSTEM_PROMPT]
    assert len(labeling_calls) == len(_PAGES)
    assert not [c for c in sync_calls if _system(c) == label_mod.PLAN_SYSTEM_PROMPT]
    assert len(backend.submitted) == 3 + 2 + 2  # every wave submitted exactly once

    assert job["outputs"] == blocking["outputs"]
    assert job["files"] == blocking["files"]
    assert job["rows"] == blocking["rows"]
    assert {r["tier"] for r in job["rows"] if r["operation"] == "transcribe"} == {"batch"}
    assert _comparable_payload(final) == _comparable_payload(blocking_payload)
    assert final["batch"]["stages"]["transcribe"]["replayed"] >= 1
    assert "replayed" not in json.dumps(blocking_payload["batch"])

    (manifest,) = list_jobs(Workspace(root=ws_j))
    assert manifest.status == "completed"
    assert manifest.runs == len(pendings) + 1


# ---- extraction extract ---------------------------------------------------------------


def _label_rows_merged(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The labeling pass's usage with tier-split parts summed back together."""
    out: dict[str, Any] = {}
    for row in rows:
        if row["operation"] != "label":
            continue
        for key in ("cost_usd", "prompt_tokens", "completion_tokens", "total_tokens"):
            out[key] = out.get(key, 0) + (row[key] or 0)
    return {k: pytest.approx(v) for k, v in out.items()}


def test_generate_per_document_label_job_matches_a_blocking_batch_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
) -> None:
    """Batch labeling (the default) under an open vocabulary, as a --no-wait
    job: labeling goes through the batch API one document per wave — each
    resume advances one document — and the job ends with exactly the DGML and
    cache files of a blocking --batch --no-batch-label run that labeled
    synchronously (which are the synchronous run's)."""
    ws_b, did_b = _seed_generate_ws(tmp_path / "blocking", capsys)
    provider.install("anthropic", _generation_answer, polls=0)
    with patch("litellm.completion", side_effect=lambda **kw: _generation_answer(kw)):
        assert main(_generate_argv(ws_b, did_b, "--no-batch-label")) == 0
    blocking_payload = _read_stdout(capsys)
    blocking = _generate_state(ws_b, did_b)
    assert blocking_payload["batch"]["stages"]["label"] == {
        "skipped": BATCH_LABEL_OFF,
        "mode": "sync",
    }

    ws_j, did_j = _seed_generate_ws(tmp_path / "job", capsys)
    backend = provider.install("anthropic", _generation_answer, polls=1)
    sync_calls: list[dict[str, Any]] = []

    def sync(**kwargs: Any) -> Any:
        sync_calls.append(kwargs)
        return _generation_answer(kwargs)

    with patch("litellm.completion", side_effect=sync):
        argv = _generate_argv(ws_j, did_j, "--no-wait")
        assert main(argv) == 0
        first = _read_stdout(capsys)
        assert first["batch_job"]["status"] == "pending"
        pendings, final = _drive_job(capsys, ws_j, first)
    job = _generate_state(ws_j, did_j)

    # Three transcription waves, roster planning's draft and refine, ONE
    # labeling wave per document (each is a single chunk with nothing to
    # retry), then the link pass's two waves.
    assert len(pendings) == 3 + 2 + len(_PAGES) + 2
    assert len(backend.submitted) == 3 + 2 + len(_PAGES) + 2  # every wave submitted once
    assert not [c for c in sync_calls if _system(c) == label_mod.SYSTEM_PROMPT]
    assert not [c for c in sync_calls if _system(c) == label_mod.PLAN_SYSTEM_PROMPT]
    # Stage order: plan (draft, refine), then labeling document by document.
    kinds = {label_mod.PLAN_SYSTEM_PROMPT: "plan", label_mod.SYSTEM_PROMPT: "label"}
    wave_kinds = [{kinds.get(_system(r.kwargs), "other") for r in w} for w in backend.submitted]
    assert wave_kinds[3 : 5 + len(_PAGES)] == [{"plan"}] * 2 + [{"label"}] * len(_PAGES)
    labeled = [
        r
        for wave in backend.submitted
        for r in wave
        if _system(r.kwargs) == label_mod.SYSTEM_PROMPT
    ]
    assert len(labeled) == len(_PAGES)

    assert job["outputs"] == blocking["outputs"]
    assert job["files"] == blocking["files"]
    assert [r for r in job["rows"] if r["operation"] != "label"] == [
        r for r in blocking["rows"] if r["operation"] != "label"
    ]
    assert _label_rows_merged(job["rows"]) == _label_rows_merged(blocking["rows"])
    assert "batch" in {r["tier"] for r in job["rows"] if r["operation"] == "label"}
    label = final["batch"]["stages"]["label"]
    assert label["mode"] == "per-document" and label["documents"] == len(_PAGES)
    assert label["requests"] == len(_PAGES) and label["replayed"] >= 1
    assert _comparable_payload(final) == _comparable_payload(blocking_payload)


def _coining_answer(kwargs: dict[str, Any]) -> Any:
    """:func:`_generation_answer`, except labeling coins ``Salutation`` — a
    concept planning never named — so the run describes it afterwards."""
    if _system(kwargs) == label_mod.SYSTEM_PROMPT:
        ids = sorted(set(_BLOCK_ID_RE.findall(_user(kwargs))))
        return _reply(json.dumps({"labels": {i: {"concept": "Salutation"} for i in ids}}))
    if _system(kwargs) == prompt("describe_concepts"):
        return _reply(json.dumps({"descriptions": {"Salutation": "an opening greeting"}}))
    return _generation_answer(kwargs)


def test_a_per_document_label_job_replays_planning_labeling_and_descriptions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
) -> None:
    """Every Pass B stage batched in one --no-wait job — planning's two waves,
    one labeling wave per document, the description wave — then the links.
    Each resume replays every earlier stage from the job's store (each wave is
    submitted exactly once), and the job ends with a synchronous run's DGML,
    cache files (schema.json with the description included) and usage sums."""
    ws_s, did_s = _seed_generate_ws(tmp_path / "sync", capsys)
    with patch("litellm.completion", side_effect=lambda **kw: _coining_answer(kw)):
        sync_argv = [a for a in _generate_argv(ws_s, did_s) if a != "--batch"]
        assert main([*sync_argv, "--no-batch"]) == 0
    _read_stdout(capsys)
    reference = _generate_state(ws_s, did_s)

    ws, did = _seed_generate_ws(tmp_path / "job", capsys)
    backend = provider.install("anthropic", _coining_answer, polls=1)
    with patch("litellm.completion", side_effect=AssertionError("no full-price fallback")):
        argv = _generate_argv(ws, did, "--no-wait")
        assert main(argv) == 0
        pendings, final = _drive_job(capsys, ws, _read_stdout(capsys))
    names = {
        transcribe_mod.SYSTEM_PROMPT: "transcribe",
        label_mod.PLAN_SYSTEM_PROMPT: "plan",
        label_mod.SYSTEM_PROMPT: "label",
        prompt("describe_concepts"): "describe",
    }
    waves = [
        "+".join(sorted({names.get(_system(r.kwargs), "links") for r in w}))
        for w in backend.submitted
    ]
    expected = ["transcribe"] * 3 + ["plan"] * 2 + ["label"] * len(_PAGES) + ["describe"]
    assert waves == [*expected, "links", "links"]
    assert len(pendings) == len(waves)  # one pause per wave, each submitted once

    job = _generate_state(ws, did)
    assert job["outputs"] == reference["outputs"]
    assert job["files"] == reference["files"]
    assert any(
        b"an opening greeting" in v for k, v in job["files"].items() if k.endswith("schema.json")
    )
    assert _label_rows_merged(job["rows"]) == _label_rows_merged(reference["rows"])
    assert {r["tier"] for r in job["rows"]} == {"batch"}
    stages = final["batch"]["stages"]
    assert {"transcribe", "plan", "label", "describe", "links"} <= set(stages)
    assert stages["label"]["mode"] == "per-document"
    assert all(v.get("sync_fallbacks", 0) == 0 for v in stages.values())


@pytest.mark.parametrize("mode", ["blocking", "no-wait", "no-wait-last-stage"])
def test_a_per_document_label_stage_failure_resumes_from_the_failed_document(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
    mode: str,
) -> None:
    """A stage-wide failure while labeling per document leaves that document
    and every later one unlabeled (a ``label_error``, no further attempt in
    that run). Whatever the run does next, the job ends with exactly the DGML
    and cache files of an uninterrupted synchronous run:

    - ``blocking``: the run finishes, the payload names the failed job (a
      blocking run's silent job is kept, store and all), and ``dgml batch
      resume`` relabels from the failed document onward.
    - ``no-wait`` (links on): the link pass's wave is still in flight when the
      run pauses, so the job is still pending; its next resume relabels the
      failed document and the job completes as usual.
    - ``no-wait-last-stage`` (``--no-semlinks``): labeling was the last stage,
      so the run ends the job failed; ``batch resume`` relabels from there.

    Every resume replays what was already served — transcription, planning
    and the documents labeled before the failure are never submitted again."""
    no_links = ["--no-semlinks"] if mode == "no-wait-last-stage" else []
    ws_s, did_s = _seed_generate_ws(tmp_path / "sync", capsys)
    with patch("litellm.completion", side_effect=lambda **kw: _generation_answer(kw)):
        sync_argv = [a for a in _generate_argv(ws_s, did_s, *no_links) if a != "--batch"]
        assert main([*sync_argv, "--no-batch"]) == 0
    _read_stdout(capsys)
    reference = _generate_state(ws_s, did_s)

    label_waves: list[int] = []

    def outage(batch: list[Any]) -> Exception | None:
        if not any(_system(r.kwargs) == label_mod.SYSTEM_PROMPT for r in batch):
            return None
        label_waves.append(len(batch))
        # The SECOND document's labeling wave fails as a whole, once.
        return RuntimeError("503 service unavailable") if len(label_waves) == 2 else None

    ws, did = _seed_generate_ws(tmp_path / "job", capsys)
    backend = FakeBackend(
        lambda request: _generation_answer(request.kwargs),
        provider="anthropic",
        polls_until_ended=0 if mode == "blocking" else 1,
        fail_submit=outage,
    )
    register_backend("anthropic", lambda _cfg: backend)
    extra = [*no_links]
    if mode != "blocking":
        extra.append("--no-wait")

    def kinds(waves: list[list[Any]]) -> list[str]:
        names = {
            transcribe_mod.SYSTEM_PROMPT: "transcribe",
            label_mod.PLAN_SYSTEM_PROMPT: "plan",
            label_mod.SYSTEM_PROMPT: "label",
        }
        return ["+".join(sorted({names.get(_system(r.kwargs), "links") for r in w})) for w in waves]

    with patch("litellm.completion", side_effect=AssertionError("no full-price fallback")):
        assert main(_generate_argv(ws, did, *extra)) == 0
        out = _read_stdout(capsys)
        # Drive the job until the run that hit the failure has ended (or paused).
        job_id = out.get("batch_job", {}).get("job_id")
        while "batch_job" in out and len(label_waves) < 2:
            assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
            out = _read_stdout(capsys)
        assert len(label_waves) == 2  # the failure happened
        before = len(backend.submitted)
        if mode == "no-wait":
            assert out["batch_job"]["status"] == "pending"  # links still in flight
        else:
            # The run ended: the failed document carries the stage's error, the
            # job is failed and resumable.
            errors = [r for r in out["results"] if r.get("label_error")]
            assert len(errors) == 1 == len(_PAGES) - 1
            assert out["batch"]["stages"]["label"]["documents"] == 2
            assert out["batch"]["job"]["status"] == "failed"
            (job,) = list_jobs(Workspace(root=ws))
            assert job.status == "failed" and "a batch stage failed" in (job.error or "")
            job_id = job.job_id
            assert _generate_state(ws, did)["outputs"] != reference["outputs"]
        assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
        resumed = _read_stdout(capsys)
        if "batch_job" in resumed:
            _pendings, resumed = _drive_job(capsys, ws, resumed)

    assert not [r for r in resumed["results"] if r.get("label_error")]
    assert "job" not in resumed["batch"]  # completed: nothing left to resume
    assert resumed["batch"]["stages"]["label"]["mode"] == "per-document"
    # Only the failed document's labeling and what follows it went out again.
    again = kinds(backend.submitted[before:])
    assert again[0] == "label" and set(again) <= {"label", "links"}
    assert len(label_waves) == 3

    final = _generate_state(ws, did)
    assert final["outputs"] == reference["outputs"]
    assert final["files"] == reference["files"]


_MODELS = {"model": "anthropic/claude-haiku-4-5", "label_model": "anthropic/claude-sonnet-4-6"}


def test_batch_label_config_key_and_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
) -> None:
    """Batch labeling is on by default; `[generation] batch_label = false` and
    `--no-batch-label` turn it off; `--batch-label` overrides a false config;
    `--no-batch-label` is refused without batch mode."""
    provider.install("anthropic", _generation_answer, polls=0)
    modes = {}
    cases: dict[str, tuple[bool | None, list[str]]] = {
        "default": (None, []),
        "config-off": (False, []),
        "flag-off": (None, ["--no-batch-label"]),
        "flag-on-over-config": (False, ["--batch-label"]),
    }
    for name, (config, extra) in cases.items():
        ws, did = _seed_generate_ws(tmp_path / name, capsys)
        section: dict[str, Any] = dict(_MODELS)
        if config is not None:
            section["batch_label"] = config
        _write_ws_config(ws, {"generation": section})
        with patch("litellm.completion", side_effect=lambda **kw: _generation_answer(kw)):
            if name == "flag-off":
                no_batch = [a for a in _generate_argv(ws, did) if a != "--batch"]
                assert main([*no_batch, "--no-batch", "--no-batch-label"]) == 1
                error = _read_stderr(capsys)["error"]
                assert error["code"] == "BATCH_JOB_INVALID"
                assert "--no-batch-label" in error["message"]
            assert main(_generate_argv(ws, did, *extra)) == 0
        modes[name] = _read_stdout(capsys)["batch"]["stages"]["label"]["mode"]
    assert modes == {
        "default": "per-document",
        "config-off": "sync",
        "flag-off": "sync",
        "flag-on-over-config": "per-document",
    }


@pytest.mark.parametrize("config", [True, False])
def test_a_job_pins_the_batch_label_choice_it_started_with(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
    config: bool,
) -> None:
    """The batch-label choice that came from `[generation] batch_label` is
    recorded on the job as `--batch-label` / `--no-batch-label`, as `--batch`
    is, so editing the config between resumes cannot switch a job half-way."""
    ws, did = _seed_generate_ws(tmp_path, capsys)
    _write_ws_config(ws, {"generation": {**_MODELS, "batch_label": config}})
    provider.install("anthropic", _generation_answer, polls=1)
    with patch("litellm.completion", side_effect=lambda **kw: _generation_answer(kw)):
        assert main(_generate_argv(ws, did, "--no-wait")) == 0
        first = _read_stdout(capsys)
        (manifest,) = list_jobs(Workspace(root=ws))
        assert manifest.argv[-1] == ("--batch-label" if config else "--no-batch-label")
        _write_ws_config(ws, {"generation": {**_MODELS, "batch_label": not config}})
        _pendings, final = _drive_job(capsys, ws, first)
    assert final["batch"]["stages"]["label"]["mode"] == ("per-document" if config else "sync")


def test_batch_label_rejects_an_invalid_config_value(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,
    pdf_stubs: None,
) -> None:
    provider.install("anthropic", _generation_answer, polls=0)
    for bad in ("parallel", "sequential", 1):
        ws, did = _seed_generate_ws(tmp_path / str(bad), capsys)
        _write_ws_config(ws, {"generation": {**_MODELS, "batch_label": bad}})
        assert main(_generate_argv(ws, did)) == 1
        error = _read_stderr(capsys)["error"]
        assert error["code"] == "GENERATION_CONFIG_INVALID" and "batch_label" in error["message"]


def _extract_argv(ws: Path, ds_id: str, *extra: str) -> list[str]:
    return _ws_args(ws) + [
        "--debug",
        "extraction",
        "extract",
        ds_id,
        "--all",
        "--batch",
        "--batch-poll-interval",
        "0.01",
        *extra,
    ]


def test_extraction_no_wait_job_matches_a_blocking_batch_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    fids = ["fjob00000001", "fjob00000002"]
    ws_b, ds_b = _seed_extraction(tmp_path / "blocking", capsys, fids)
    provider.install("anthropic", _values_answer, polls=0)
    assert main(_extract_argv(ws_b, ds_b)) == 0
    blocking = _read_stdout(capsys)

    ws_j, ds_j = _seed_extraction(tmp_path / "job", capsys, fids)
    backend = provider.install("anthropic", _values_answer, polls=1)
    with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
        assert main(_extract_argv(ws_j, ds_j, "--no-wait")) == 0
        first = _read_stdout(capsys)
        assert first["batch_job"]["requests_in_flight"] == 2
        pendings, final = _drive_job(capsys, ws_j, first)
    assert len(pendings) == 1  # one phase-1 wave (no locations: no phase 3)
    assert len(backend.submitted) == 1

    def results(payload: dict[str, Any], ds_id: str) -> list[dict[str, Any]]:
        # xml_key names each workspace's own DocSet id.
        return [{**r, "xml_key": r["xml_key"].replace(ds_id, "<ds>")} for r in payload["results"]]

    assert results(final, ds_j) == results(blocking, ds_b)
    assert final["summary"] == blocking["summary"]
    # The resume collected the open batch; nothing was stored yet to replay.
    assert "replayed" not in final["batch"]
    assert _rows(ws_j) == _rows(ws_b)


def test_no_wait_and_job_need_batch_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    for extra in (["--no-wait"], ["--job", "bj_000000000000"]):
        rc = main(_ws_args(ws) + ["extraction", "extract", ds_id, "--all", *extra])
        assert rc == 1
        assert _read_stderr(capsys)["error"]["code"] == "BATCH_JOB_INVALID"


def test_resuming_an_unknown_job_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    for sub in ("resume", "status", "cancel"):
        assert main(_ws_args(ws) + ["batch", sub, "bj_000000000000"]) == 1
        assert _read_stderr(capsys)["error"]["code"] == "BATCH_JOB_NOT_FOUND"


# ---- dgml batch status / list / cancel ----------------------------------------------------


def test_status_list_and_cancel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    fids = ["fjob00000001"]
    ws, ds_id = _seed_extraction(tmp_path, capsys, fids)
    backend = provider.install("anthropic", _values_answer, polls=2)
    assert main(_extract_argv(ws, ds_id, "--no-wait")) == 0
    job_id = _read_stdout(capsys)["batch_job"]["job_id"]

    assert main(_ws_args(ws) + ["batch", "status", job_id]) == 0
    status = _read_stdout(capsys)
    assert status["status"] == "pending"  # second poll: still running
    assert status["command"] == "extraction extract"
    (entry,) = status["batches"]
    assert entry["state"] == "open" and entry["done"] is False and entry["requests"] == 1

    assert main(_ws_args(ws) + ["batch", "status", job_id]) == 0
    assert _read_stdout(capsys)["status"] == "ready"  # third poll: ended

    assert main(_ws_args(ws) + ["batch", "list"]) == 0
    (listed,) = _read_stdout(capsys)["jobs"]
    # status is read-only: `ready` is derived when asked, the stored status stays.
    assert listed["job_id"] == job_id and listed["status"] == "pending"

    assert main(_ws_args(ws) + ["batch", "cancel", job_id]) == 0
    canceled = _read_stdout(capsys)
    assert canceled["status"] == "failed" and canceled["batches"][0]["state"] == "dropped"
    assert backend.canceled

    # A canceled job resumes by resubmitting its requests at batch price.
    fresh = provider.install("anthropic", _values_answer, polls=0)
    with patch("litellm.completion", side_effect=AssertionError("no full-price fallback")):
        assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
    final = _read_stdout(capsys)
    assert final["summary"] == {"total": 1, "ok": 1, "failed": 0}
    assert len(fresh.submitted) == 1


def test_a_finished_blocking_batch_run_leaves_no_job_behind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    """Its crash-recovery job is silent: payload unchanged, and deleted with
    all its stored responses once the run finishes."""
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    provider.install("anthropic", _values_answer, polls=0)
    assert main(_extract_argv(ws, ds_id)) == 0
    payload = _read_stdout(capsys)
    assert "batch_job" not in payload and "replayed" not in payload["batch"]
    assert "job" not in payload["batch"]  # nothing left behind, nothing to name
    assert list_jobs(Workspace(root=ws)) == []
    assert not (ws / layout.BATCHES_DIR).exists()  # not even an empty batches/


# ---- file add <dir> ----------------------------------------------------------------------


@needs_gs
def test_file_add_no_wait_job_never_ingests_twice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    """The pre-LLM phase (ingest) runs once: a resume restores its results from
    the job, so every file is still reported 'added' and gets classified."""
    payloads = {}
    for mode in ("blocking", "job"):
        ws, contracts, _, src = _seed_classify_ws(tmp_path, capsys, mode, grounded=True)
        _set_schema(ws, tmp_path, contracts, capsys)

        def answer(kwargs: dict[str, Any], contracts: str = contracts) -> Any:
            name = kwargs["tools"][0]["function"]["name"]
            if name == "assign_to_existing_docset":
                return _tool_reply(name, {"docset_id": contracts})
            return _values_answer(kwargs)

        provider.install("anthropic", answer, polls=0 if mode == "blocking" else 1)
        argv = _ws_args(ws) + [
            "file",
            "add",
            str(src),
            "--auto-classify",
            "existing",
            "--batch",
            "--batch-poll-interval",
            "0.01",
        ]
        if mode == "job":
            argv.append("--no-wait")
        with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
            assert main(argv) == 0
            out = _read_stdout(capsys)
            if mode == "job":
                pendings, out = _drive_job(capsys, ws, out)
                # Classification wave, then the auto-extraction's phase-1 wave.
                assert len(pendings) == 2
        payloads[mode] = out
        main(_ws_args(ws) + ["file", "list"])
        assert len(_read_stdout(capsys)["files"]) == 2  # ingested once, not twice

    def comparable(payload: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for entry in payload["results"]:
            block = dict(entry["classification"])
            block.pop("docset_id", None)
            out.append({"status": entry["status"], "classification": block})
        return out

    assert payloads["job"]["summary"] == payloads["blocking"]["summary"]
    assert payloads["job"]["summary"]["added"] == 2
    assert comparable(payloads["job"]) == comparable(payloads["blocking"])


@needs_gs
def test_file_add_default_mode_job_creates_each_docset_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    """The default mode as a --no-wait job: one pause per file's wave, and
    every resume replays the earlier files' decisions without creating their
    DocSets again — the job ends with the blocking run's DocSets."""

    def answer(kwargs: dict[str, Any]) -> Any:
        text = kwargs["messages"][0]["content"][0]["text"]
        letters = re.search(r"- id=(\S+)\n  name: Letters\n", text)
        if letters is None:
            return _tool_reply(
                "create_new_docset",
                {"name": "Letters", "description": "letters", "key_questions": ["Who?"]},
            )
        return _tool_reply("assign_to_existing_docset", {"docset_id": letters.group(1)})

    results: dict[str, Any] = {}
    for mode in ("blocking", "job"):
        ws, _contracts, _, src = _seed_classify_ws(tmp_path, capsys, mode)
        backend = provider.install("anthropic", answer, polls=0 if mode == "blocking" else 1)
        argv = _ws_args(ws) + ["file", "add", str(src), "--auto-classify", "--batch"]
        argv += ["--batch-poll-interval", "0.01"]
        if mode == "job":
            argv.append("--no-wait")
        with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
            assert main(argv) == 0
            out = _read_stdout(capsys)
            if mode == "job":
                pendings, out = _drive_job(capsys, ws, out)
                assert len(pendings) == 2  # one wave per file
        assert [len(w) for w in backend.submitted] == [1, 1]  # each submitted once
        main(_ws_args(ws) + ["docset", "list"])
        names = sorted(d["name"] for d in _read_stdout(capsys)["docsets"])
        results[mode] = (out, names)

    (blocking, names_b), (job, names_j) = results["blocking"], results["job"]
    assert names_j == names_b == ["Contracts", "Letters", "Safety Datasheets"]
    for payload in (blocking, job):
        decisions = [e["classification"]["decision"] for e in payload["results"]]
        assert decisions == ["new", "existing"]
        ids = {e["classification"]["docset_id"] for e in payload["results"]}
        assert len(ids) == 1  # the second file joined the DocSet the first created
    assert job["summary"] == blocking["summary"]


# ---- hardening: uncertain creates and stage outages --------------------------------------


def _install_failing(provider_name: str, fail: Any) -> FakeBackend:
    backend = FakeBackend(
        lambda request: _values_answer(request.kwargs), provider=provider_name, fail_submit=fail
    )
    register_backend(provider_name, lambda _cfg: backend)
    return backend


def _error_envelope(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """The error envelope on stderr, past any log lines before it."""
    err = capsys.readouterr().err
    start = max(i for i, c in enumerate(err) if c == "{" and (i == 0 or err[i - 1] == "\n"))
    return json.loads(err[start:])  # type: ignore[no-any-return]


def test_a_job_with_an_uncertain_create_refuses_to_resume_until_canceled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    backend = _install_failing("anthropic", BatchSubmitUncertain("read timeout after send"))
    with patch("litellm.completion", side_effect=AssertionError("no full-price fallback")):
        main(_extract_argv(ws, ds_id, "--no-wait"))
    capsys.readouterr()
    (job,) = list_jobs(Workspace(root=ws))
    assert job.status == "failed"
    assert [r["state"] for r in job.provider_batches] == ["uncertain"]

    # Neither `batch resume` nor `--job` resubmits the requests that may exist.
    assert main(_ws_args(ws) + ["batch", "resume", job.job_id]) == 1
    error = _error_envelope(capsys)["error"]
    assert error["code"] == "BATCH_JOB_INVALID"
    assert f"dgml batch cancel {job.job_id}" in error["message"]
    assert main(_extract_argv(ws, ds_id, "--job", job.job_id)) == 1
    assert _error_envelope(capsys)["error"]["code"] == "BATCH_JOB_INVALID"
    assert len(backend.attempted) == 1

    # `batch cancel` acknowledges it; the resume then submits those requests again.
    assert main(_ws_args(ws) + ["batch", "cancel", job.job_id]) == 0
    canceled = _read_stdout(capsys)
    (entry,) = canceled["batches"]
    assert entry["state"] == "dropped" and entry["acknowledged"] is True
    assert "may exist" in entry["error"]
    assert canceled["canceled"] is True and canceled["status"] == "failed"

    fresh = provider.install("anthropic", _values_answer, polls=0)
    with patch("litellm.completion", side_effect=AssertionError("no full-price fallback")):
        assert main(_ws_args(ws) + ["batch", "resume", job.job_id]) == 0
    assert _read_stdout(capsys)["summary"] == {"total": 1, "ok": 1, "failed": 0}
    assert len(fresh.submitted) == 1


def test_prune_keeps_a_job_with_an_unacknowledged_uncertain_create(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    _install_failing("anthropic", BatchSubmitUncertain("503 after send"))
    main(_extract_argv(ws, ds_id, "--no-wait"))
    capsys.readouterr()
    (job,) = list_jobs(Workspace(root=ws))
    assert main(_ws_args(ws) + ["batch", "prune"]) == 0
    assert _read_stdout(capsys) == {"deleted": [], "kept": [job.job_id]}
    main(_ws_args(ws) + ["batch", "cancel", job.job_id])
    capsys.readouterr()
    assert main(_ws_args(ws) + ["batch", "prune"]) == 0
    assert _read_stdout(capsys)["deleted"] == [job.job_id]


@pytest.mark.parametrize("mode", ["blocking", "no-wait"])
def test_a_stage_outage_leaves_a_resumable_failed_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], provider: _Provider, mode: str
) -> None:
    """A wave that fails as a whole soft-fails its files (exit 0, errors in the
    payload, as before) — but the job must end failed with its store kept, not
    completed and trimmed, so a resume picks the work up."""
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fjob00000001"])
    _install_failing("anthropic", RuntimeError("503 service unavailable"))
    extra = ["--no-wait"] if mode == "no-wait" else []
    with patch("litellm.completion", side_effect=AssertionError("no full-price fallback")):
        assert main(_extract_argv(ws, ds_id, *extra)) == 0
    payload = _read_stdout(capsys)
    assert payload["summary"] == {"total": 1, "ok": 0, "failed": 1}
    assert "BATCH_EXECUTION_FAILED" in json.dumps(payload["results"])

    (job,) = list_jobs(Workspace(root=ws))  # even the blocking run's silent job is kept
    assert job.status == "failed"
    assert job.error is not None and "a batch stage failed" in job.error
    # The job left behind is named in the payload, not only in a log line.
    assert payload["batch"]["job"] == {
        "job_id": job.job_id,
        "status": "failed",
        "resume": f"dgml batch resume {job.job_id}",
    }

    fresh = provider.install("anthropic", _values_answer, polls=0)
    with patch("litellm.completion", side_effect=AssertionError("no full-price fallback")):
        assert main(_ws_args(ws) + ["batch", "resume", job.job_id]) == 0
    resumed = _read_stdout(capsys)
    assert resumed["summary"] == {"total": 1, "ok": 1, "failed": 0}
    assert "job" not in resumed["batch"]  # completed: nothing left to resume
    assert len(fresh.submitted) == 1
