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

"""`generate_docset` — the library entry point behind `dgml docset generate`.

The CLI suite (packages/dgml/tests/test_cli.py) exercises the orchestration
end to end through `main()` and pins the JSON payload; these tests cover what
is library-specific — the typed report's serialization contract, the raised
preconditions, and the schema-seed loaders that moved here from `cli.py`.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core import (
    DocSetStore,
    EmptyDocSet,
    FileStore,
    GenerateFileResult,
    GenerateFileStatus,
    GenerateReport,
    JobRequest,
    Workspace,
)
from dgml_core.docset_generate import (
    _load_schema_roster,
    _load_schema_seed,
    batch_stage_models,
    generate_docset,
)
from dgml_core.errors import BatchJobInvalid, BatchUnavailable, DocSetNotFound, InvalidArgument
from dgml_core.generation import BatchOptions, resolve_generation_config
from dgml_core.style_config import StyleConfig

from .conftest import needs_gs, write_config

_LABEL_MODEL = "anthropic/claude-sonnet-4-6"


def _batch_seed(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: str = "anthropic/claude-haiku-4-5",
    pdf: Path | None = None,
) -> str:
    """A docset under batchable-or-not generation models, holding *pdf*."""
    write_config(workspace, {"generation": {"model": model, "label_model": _LABEL_MODEL}})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-dummy")  # never used: no call is made
    ds = DocSetStore(workspace).create(name="Batch")
    if pdf is not None:
        added = FileStore(workspace).add(pdf)
        DocSetStore(workspace).add_file(ds.id, added.record.id)
    return ds.id


# ---------------------------------------------------------------------------
# Raised preconditions
# ---------------------------------------------------------------------------


def test_generate_docset_raises_for_missing_docset(workspace: Workspace) -> None:
    with pytest.raises(DocSetNotFound):
        generate_docset(workspace, "ds_nope")


def test_generate_docset_raises_empty_docset(workspace: Workspace) -> None:
    """A DocSet with no files assigned is a precondition failure, not an empty
    report — EMPTY_DOCSET, the code the CLI has always emitted for it."""
    ds = DocSetStore(workspace).create(name="Empty")
    with pytest.raises(EmptyDocSet) as excinfo:
        generate_docset(workspace, ds.id)
    assert excinfo.value.code == "EMPTY_DOCSET"
    assert ds.id in str(excinfo.value)


@needs_gs
def test_batch_rejects_an_unbatchable_stage_before_any_spend(
    workspace: Workspace, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A library caller gets the same pre-flight the CLI does: a stage whose
    model has no batch backend is BATCH_UNAVAILABLE, naming the stage, before
    transcription runs."""
    did = _batch_seed(
        workspace, monkeypatch, model="openrouter/anthropic/claude-sonnet-4", pdf=sample_pdf
    )
    with (
        patch("dgml_core.generation.convert_batch") as convert,
        pytest.raises(BatchUnavailable) as excinfo,
    ):
        generate_docset(workspace, did, batch=BatchOptions())
    assert "stage 'transcribe'" in str(excinfo.value)
    convert.assert_not_called()


def test_batch_preflight_runs_after_the_offline_checks(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch pre-flight is the last check: an empty docset is EMPTY_DOCSET
    whatever its models could batch."""
    did = _batch_seed(workspace, monkeypatch, model="openrouter/anthropic/claude-sonnet-4")
    with pytest.raises(EmptyDocSet):
        generate_docset(workspace, did, batch=BatchOptions())


@needs_gs
def test_pipeline_knobs_default_to_convert_options(
    workspace: Workspace, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unset, the four pipeline knobs take ConvertOptions' own defaults, and
    an explicit value still wins — the numbers live in one place."""
    from dgml_core.generation import ConvertOptions

    did = _batch_seed(workspace, monkeypatch, pdf=sample_pdf)
    with patch("dgml_core.generation.convert_batch", return_value={}) as convert:
        generate_docset(workspace, did)
    opts = convert.call_args.kwargs["options"]
    assert (opts.window_size, opts.temperature, opts.max_tokens, opts.max_parallel_docs) == (
        ConvertOptions.window_size,
        ConvertOptions.temperature,
        ConvertOptions.max_tokens,
        ConvertOptions.max_parallel_docs,
    )
    with patch("dgml_core.generation.convert_batch", return_value={}) as convert:
        generate_docset(workspace, did, window_size=3, max_parallel_docs=1)
    opts = convert.call_args.kwargs["options"]
    assert (opts.window_size, opts.max_parallel_docs) == (3, 1)


@needs_gs
def test_progress_logs_from_worker_threads_reach_the_module_logger(
    workspace: Workspace,
    sample_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The per-document sink runs on the pipeline's worker pool; its lines
    still land on ``dgml_core.docset_generate`` at INFO, as a caller's handler
    sees them, whichever thread emitted them."""
    did = _batch_seed(workspace, monkeypatch, pdf=sample_pdf)

    def fake_convert(
        paths: object, *, options: object, on_output: Any, **_kw: object
    ) -> dict[str, str]:
        worker = threading.Thread(
            target=on_output, args=(sample_pdf.name, "<xml><chunk>hello</chunk></xml>")
        )
        worker.start()
        worker.join()
        return {}

    caplog.set_level(logging.INFO, logger="dgml_core.docset_generate")
    with patch("dgml_core.generation.convert_batch", side_effect=fake_convert):
        generate_docset(workspace, did, semlinks=False)  # no link model call to block
    ground = [r for r in caplog.records if r.getMessage().startswith("[ground] ")]
    assert ground, [r.getMessage() for r in caplog.records]
    assert ground[0].name == "dgml_core.docset_generate"
    assert ground[0].threadName != threading.main_thread().name


@needs_gs
def test_batch_credentials_default_from_the_resolved_config(
    workspace: Workspace, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline gets the resolved key references under the caller's own,
    on a copy: the caller's BatchOptions is left as it was, so one object
    can serve a second run (with other models) without carrying this run's
    credentials or stats into it."""
    did = _batch_seed(workspace, monkeypatch, pdf=sample_pdf)
    mine = {"section": "generation", "field": "label", "env": "MY_LABEL_KEY"}
    opts = BatchOptions(credentials={"label": dict(mine)})
    with patch("dgml_core.generation.convert_batch", return_value={}) as convert:
        generate_docset(workspace, did, batch=opts)
    received = convert.call_args.kwargs["options"].batch
    assert received is not opts
    assert received.credentials == {
        "transcribe": {"section": "generation", "field": "transcribe", "env": None},
        "label": mine,
    }
    assert opts.credentials == {"label": mine}
    assert opts.stats == {}


def test_an_unknown_mode_is_rejected_before_anything_is_read(workspace: Workspace) -> None:
    """A misspelled thinking or transcription mode is INVALID_ARGUMENT up
    front (the docset is not even looked up), never a silent text-mode run or
    a per-window provider rejection after the pre-flights."""
    with pytest.raises(InvalidArgument, match="thinking must be one of"):
        generate_docset(workspace, "nope", thinking="off")  # type: ignore[arg-type]
    with pytest.raises(InvalidArgument, match="transcription must be one of"):
        generate_docset(workspace, "nope", transcription="line")  # type: ignore[arg-type]


@needs_gs
def test_job_request_opens_only_once_the_preflights_pass(
    workspace: Workspace, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run rejected at pre-flight never opened its job: nothing to resume,
    nothing left in the workspace."""
    from dgml_core.batch import list_jobs

    did = _batch_seed(
        workspace, monkeypatch, model="openrouter/anthropic/claude-sonnet-4", pdf=sample_pdf
    )
    request = JobRequest()
    with pytest.raises(BatchUnavailable):
        generate_docset(workspace, did, batch=BatchOptions(), job=request)
    assert request.session is None
    assert list_jobs(workspace) == []


@needs_gs
def test_job_request_runs_under_a_job_the_run_opens_and_closes(
    workspace: Workspace, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller holds no session: the run opens the job after its
    pre-flights, records it on the request, and closes it on the way out — a
    finished blocking run's job is deleted with it."""
    from dgml_core.batch import list_jobs
    from dgml_core.batch.jobs import JobStatus, active_session

    did = _batch_seed(workspace, monkeypatch, pdf=sample_pdf)
    request = JobRequest()
    with patch("dgml_core.generation.convert_batch", return_value={}):
        report = generate_docset(workspace, did, batch=BatchOptions(), job=request)
    assert report.total == 1
    assert request.session is not None
    assert request.session.manifest.command == "docset generate"
    assert request.session.manifest.status == JobStatus.COMPLETED
    assert active_session() is None  # closed, not left active for the caller
    assert list_jobs(workspace) == []


def test_a_job_needs_batch_mode(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    did = _batch_seed(workspace, monkeypatch)
    with pytest.raises(BatchJobInvalid):
        generate_docset(workspace, did, job=JobRequest())


@needs_gs
def test_batch_stage_models_gate_style_and_links(
    workspace: Workspace, sample_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Style is a stage only when a file is OCR; links only with semlinks on."""
    did = _batch_seed(workspace, monkeypatch, pdf=sample_pdf)
    store = DocSetStore(workspace)
    file_ids = store.list_files(did)
    gen_cfg, _source = resolve_generation_config(workspace)
    style = StyleConfig(model="anthropic/claude-haiku-4-5")
    files = FileStore(workspace)

    assert batch_stage_models(files, file_ids, gen_cfg, None, semlinks=False) == {
        "transcribe": "anthropic/claude-haiku-4-5",
        "label": _LABEL_MODEL,
    }
    assert batch_stage_models(files, file_ids, gen_cfg, style, semlinks=True) == {
        "transcribe": "anthropic/claude-haiku-4-5",
        "label": _LABEL_MODEL,
        "links": _LABEL_MODEL,
    }
    record = workspace.docs.get_doc("files", file_ids[0])
    assert record is not None
    workspace.docs.put_doc("files", file_ids[0], {**record, "text_mode": "ocr"})
    assert batch_stage_models(files, file_ids, gen_cfg, style, semlinks=True) == {
        "transcribe": "anthropic/claude-haiku-4-5",
        "label": _LABEL_MODEL,
        "style": "anthropic/claude-haiku-4-5",
        "links": _LABEL_MODEL,
    }


# ---------------------------------------------------------------------------
# Serialization contract — `to_json` must render exactly the CLI's rows
# ---------------------------------------------------------------------------


def test_file_result_rows_match_the_cli_shape() -> None:
    """Conditional keys appear exactly when their pass ran or failed, in the
    CLI's key order — the report is the payload, so the two cannot drift."""
    skipped = GenerateFileResult(
        status=GenerateFileStatus.SKIPPED, file_id="f1", source="a.pdf", output="k/a"
    )
    assert skipped.to_json() == {
        "status": "skipped",
        "file_id": "f1",
        "source": "a.pdf",
        "output": "k/a",
    }

    failed = GenerateFileResult(
        status=GenerateFileStatus.FAILED,
        file_id="f2",
        source="b.pdf",
        error={"code": "X", "message": "m"},
    )
    assert failed.to_json() == {
        "status": "failed",
        "file_id": "f2",
        "source": "b.pdf",
        "error": {"code": "X", "message": "m"},
    }

    grounded = GenerateFileResult(
        status=GenerateFileStatus.CONVERTED,
        file_id="f3",
        source="c.pdf",
        output="k/c",
        links=2,
        grounded=True,
        matched_token_pct=98.5,
        elements_annotated=40,
    )
    assert grounded.to_json() == {
        "status": "converted",
        "file_id": "f3",
        "source": "c.pdf",
        "output": "k/c",
        "links": 2,
        "grounded": True,
        "matched_token_pct": 98.5,
        "elements_annotated": 40,
    }

    ungrounded = GenerateFileResult(
        status=GenerateFileStatus.CONVERTED,
        file_id="f4",
        source="d.pdf",
        output="k/d",
        links=0,
        grounded=False,
        grounding_error={"code": "GROUNDING_FAILED", "message": "no page_text"},
        label_error={"code": "LABEL_MODEL_UNREACHABLE", "message": "401"},
        link_error="rate limited",
    )
    row = ungrounded.to_json()
    assert row["grounded"] is False
    assert row["grounding_error"] == {"code": "GROUNDING_FAILED", "message": "no page_text"}
    assert "matched_token_pct" not in row and "elements_annotated" not in row
    assert row["label_error"] == {"code": "LABEL_MODEL_UNREACHABLE", "message": "401"}
    assert row["link_error"] == "rate limited"


def test_off_schema_concepts_key_follows_extend_schema() -> None:
    """The same facts render as `unmatched_concepts` under a strict schema
    (refused) and `added_concepts` under extend-schema (coined and used)."""
    tally = {"count": 3, "distinct": 2, "examples": ["Foo", "Bar"]}
    result = GenerateFileResult(
        status=GenerateFileStatus.CONVERTED,
        file_id="f1",
        source="a.pdf",
        output="k/a",
        links=0,
        grounded=True,
        matched_token_pct=100.0,
        elements_annotated=1,
        off_schema_concepts=tally,
    )
    assert result.to_json(extend_schema=False)["unmatched_concepts"] == tally
    assert result.to_json(extend_schema=True)["added_concepts"] == tally


def test_report_to_json_builds_the_generate_envelope(workspace: Workspace) -> None:
    """The envelope the CLI prints: summary counts come from the three lists,
    results concatenate in skipped + failed + converted order."""
    ds = DocSetStore(workspace).create(name="Invoices")
    skipped = GenerateFileResult(
        status=GenerateFileStatus.SKIPPED, file_id="f1", source="a.pdf", output="k/a"
    )
    failed = GenerateFileResult(
        status=GenerateFileStatus.FAILED,
        file_id="f2",
        source="b.pdf",
        error={"code": "X", "message": "m"},
    )
    converted = GenerateFileResult(
        status=GenerateFileStatus.CONVERTED,
        file_id="f3",
        source="c.pdf",
        output="k/c",
        links=1,
        grounded=True,
        matched_token_pct=99.0,
        elements_annotated=7,
    )
    report = GenerateReport(
        docset=ds,
        total=3,
        skipped=[skipped],
        failed=[failed],
        converted=[converted],
        rerendered=["a.pdf"],
        output_key=f"docsets/{ds.id}",
        coverage_report_key=None,
        model="anthropic/claude-haiku-4-5",
        label_model="anthropic/claude-sonnet-5",
        model_source="workspace config",
        schema_extended=False,
    )
    assert report.results == [skipped, failed, converted]
    assert report.to_json() == {
        "docset_id": ds.id,
        "docset_name": "Invoices",
        "summary": {"total": 3, "converted": 1, "skipped": 1, "failed": 1},
        "models": {
            "model": "anthropic/claude-haiku-4-5",
            "label_model": "anthropic/claude-sonnet-5",
            "source": "workspace config",
        },
        "output_key": f"docsets/{ds.id}",
        "coverage_report": None,
        "results": [skipped.to_json(), failed.to_json(), converted.to_json()],
        "rerendered": ["a.pdf"],
    }


# ---------------------------------------------------------------------------
# Schema-seed loaders (moved from cli.py with `docset generate`)
# ---------------------------------------------------------------------------


def test_load_schema_roster_errors(tmp_path: Path) -> None:
    """_load_schema_roster rejects missing files, non-object/invalid JSON, and
    rosters that sanitize to no usable concepts — all as InvalidArgument."""
    with pytest.raises(InvalidArgument):
        _load_schema_roster(tmp_path / "missing.json")

    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_roster(arr)

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_roster(bad)

    junk = tmp_path / "junk.json"
    junk.write_text(json.dumps({"###": "x", "!!!": "y"}), encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_roster(junk)


def test_load_schema_seed_json_builds_roster_and_parent_map(tmp_path: Path) -> None:
    p = tmp_path / "schema.json"
    p.write_text(
        json.dumps(
            {
                "tags": {
                    "PartyInformation": {
                        "name": "PartyInformation",
                        "role": "party block",
                        "kind": "section",
                    },
                    "PartyAddress": {
                        "name": "PartyAddress",
                        "role": "address",
                        "parent_role": "PartyInformation",
                    },
                    "OrderDate": {"name": "OrderDate", "role": "order date"},
                }
            }
        ),
        encoding="utf-8",
    )
    schema, parent_map, _notes = _load_schema_seed(p)
    assert {"PartyInformation", "PartyAddress", "OrderDate"} <= set(schema.tags)
    assert schema.tags["PartyAddress"].role == "address"
    assert schema.tags["PartyInformation"].kind == "section"  # fidelity kept, not flattened
    assert parent_map["PartyAddress"] == "PartyInformation"  # via parent_role
    assert "OrderDate" not in parent_map  # top-level, no container


def test_load_schema_seed_accepts_rnc(tmp_path: Path) -> None:
    """--schema-path also accepts the lossless full-schema.rnc render — the
    `# Field: value` comment contract reconstructs the same roster/parent_map."""
    p = tmp_path / "full-schema.rnc"
    p.write_text(
        "# " + "-" * 20 + "\n"
        '# Description: "party block"\n'
        "# Kind: section\n"
        "PartyInformation = element PartyInformation {\n  common.atts,\n"
        "  mixed { any.docset* }\n}\n\n"
        "# " + "-" * 20 + "\n"
        '# Description: "address"\n'
        "# Kind: inline\n"
        "# Parent: PartyInformation\n"
        "PartyAddress = element PartyAddress {\n  common.atts,\n  text\n}\n",
        encoding="utf-8",
    )
    schema, parent_map, _notes = _load_schema_seed(p)
    assert {tag.name: tag.role for tag in schema.tags.values()} == {
        "PartyInformation": "party block",
        "PartyAddress": "address",
    }
    assert parent_map == {"PartyAddress": "PartyInformation"}


def test_load_schema_seed_accepts_a_plain_tag_list(tmp_path: Path) -> None:
    """Form A — one bare tag name per line, `#` comments and blanks ignored.

    Names are taken VERBATIM: `Notes` and `Details` survive (`sanitize_concept`
    would fold both to ''), and only XML validity is enforced, which turns
    `Line Items` into `Line_Items` and says so in the notes."""
    p = tmp_path / "tags.txt"
    p.write_text(
        "# Liquor distribution purchase orders\n\nCustomerName\nNotes\n"
        "Details\nLine Items\nAgreementSummary\n",
        encoding="utf-8",
    )
    schema, parent_map, notes = _load_schema_seed(p)
    assert list(schema.tags) == [
        "CustomerName",
        "Notes",
        "Details",
        "Line_Items",
        "AgreementSummary",
    ]
    assert all(tag.kind == "inline" for tag in schema.tags.values())
    assert parent_map == {}
    assert any("Line Items" in note and "Line_Items" in note for note in notes)
    assert any("kind=inline" in note for note in notes)


def test_load_schema_seed_accepts_a_name_to_description_mapping(tmp_path: Path) -> None:
    """Form B — the recommended shape. Previously rejected outright."""
    p = tmp_path / "schema.json"
    p.write_text(
        json.dumps({"BuyerName": "bill-to org", "OrderDate": "date the order was placed"}),
        encoding="utf-8",
    )
    schema, parent_map, _notes = _load_schema_seed(p)
    assert {n: t.role for n, t in schema.tags.items()} == {
        "BuyerName": "bill-to org",
        "OrderDate": "date the order was placed",
    }
    assert parent_map == {}


def test_load_schema_seed_rejects_ambiguous_input(tmp_path: Path) -> None:
    """What cannot be read UNAMBIGUOUSLY fails at load, never as a tag that
    quietly failed to appear in the output hours later."""

    def _reject(name: str, text: str) -> str:
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        with pytest.raises(InvalidArgument) as excinfo:
            _load_schema_seed(p)
        return str(excinfo.value)

    # A `key: value` file (YAML, or Form B written as text) would otherwise
    # load as tags named "concepts_" and "BuyerName__bill-to_org".
    assert "':'" in _reject("seed.txt", "concepts:\n  BuyerName: bill-to org\n")
    # Two names that differ only in case/punctuation are one tag, twice.
    assert "differ only in case" in _reject("dup.txt", "BuyerName\nbuyer_name\n")
    # A kind outside VALID_KINDS is a typo, not a silent coercion to `inline`.
    assert "kind" in _reject("k.json", json.dumps({"tags": {"A": {"role": "x", "kind": "field"}}}))
    # A parent_role naming a tag that does not exist would synthesize a
    # container outside the vocabulary.
    assert "parent_role" in _reject(
        "p.json", json.dumps({"tags": {"A": {"role": "x", "parent_role": "Nope"}}})
    )
    # An empty `tags` map, and a JSON array (neither form).
    assert "no tags" in _reject("e.json", json.dumps({"tags": {}}))
    assert "array" in _reject("a.json", json.dumps(["A", "B"]))

    with pytest.raises(InvalidArgument):
        _load_schema_seed(tmp_path / "missing.json")
