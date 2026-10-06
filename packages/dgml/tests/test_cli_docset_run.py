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

"""``dgml docset run``: the whole docset pipeline in one command.

The model is deterministic (every reply depends only on its request's bytes),
so a ``run`` can be compared with the three standalone commands it chains, a
``--batch`` run with a synchronous one, and a ``--no-wait`` job driven to
completion with ``dgml batch resume`` with one blocking ``--batch`` run. The
providers are :class:`FakeBackend` instances shared by every run of a test,
the way a real provider's batches outlive the process that submitted them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import (
    _RUN_BOOLEAN_OPTIONAL,
    _RUN_PASSTHROUGH,
    _build_parser,
    _run_step_argv,
    main,
)
from dgml_core import layout
from dgml_core.batch import jobs as jobs_mod
from dgml_core.batch import list_jobs
from dgml_core.docsets import DocSetStore
from dgml_core.generation import transcribe as transcribe_mod
from dgml_core.storage import Workspace

from .test_cli import (
    _RNC_SCHEMA,
    _init_ws,
    _read_stderr,
    _read_stdout,
    _seed_file_dir,
    _write_ws_config,
)
from .test_cli_batch_jobs import (
    _PAGES,
    _generate_state,
    _generation_answer,
    _Provider,
    _system,
    _values_answer,
    _ws_args,
    pdf_stubs,  # noqa: F401  (fixture)
    provider,  # noqa: F401  (fixture)
)
from .test_cli_batch_schema import _schema_reply

_FIDS = [f"fgen0000000{i}" for i in range(len(_PAGES))]
# The last (longest: 3 pages) document is the schema sample.
_SAMPLE = _FIDS[-1]


def _answer(kwargs: dict[str, Any]) -> Any:
    """One model for every step: schema, generation, extraction."""
    names = [t["function"]["name"] for t in kwargs.get("tools") or []]
    if "submit_schema" in names:
        return _schema_reply(**kwargs)
    if "submit_values" in names:
        return _values_answer(kwargs)
    return _generation_answer(kwargs)


def _sync(**kwargs: Any) -> Any:
    return _answer(kwargs)


def _seed(
    root: Path, capsys: pytest.CaptureFixture[str], *, schema: bool = False
) -> tuple[Path, str]:
    ws = root / "ws"
    _init_ws(ws)
    capsys.readouterr()
    _write_ws_config(
        ws,
        {
            "generation": {
                "model": "anthropic/claude-haiku-4-5",
                "label_model": "anthropic/claude-sonnet-4-6",
            },
            "grounded": {
                "schema_model": "anthropic/claude-opus-4-7",
                "values_model": "gemini/gemini-2.5-pro",
            },
        },
    )
    main(_ws_args(ws) + ["docset", "create", "--name", "Letters"])
    did = str(_read_stdout(capsys)["id"])
    wsx = Workspace(root=ws)
    store = DocSetStore(wsx)
    for fid, (name, pages) in zip(_FIDS, _PAGES.items(), strict=True):
        _seed_file_dir(ws, fid, pages=pages, pdf_name=name)
        for n in range(1, pages + 1):  # grounding reads the page's dimensions
            page = {"file_id": fid, "page": n, "width": 1000, "height": 1000, "words": []}
            wsx.blobs.put_blob(layout.file_page_text_key(fid, n), json.dumps(page).encode())
        store.add_file(did, fid)
    if schema:
        store.set_schema(did, _RNC_SCHEMA)
    return ws, did


_GEN_FLAGS = ["--no-coverage", "--max-parallel-calls", "1", "--window-size", "1"]
_POLL = ["--batch-poll-interval", "0.01"]


def _run_argv(ws: Path, did: str, *extra: str) -> list[str]:
    return _ws_args(ws) + ["--debug", "docset", "run", did, *_GEN_FLAGS, *extra]


def _timeless(value: Any) -> Any:
    """*value* without per-run timing (``completed_at``, ``duration_s``)."""
    if isinstance(value, dict):
        return {
            k: _timeless(v) for k, v in value.items() if k not in ("completed_at", "duration_s")
        }
    if isinstance(value, list):
        return [_timeless(v) for v in value]
    return value


def _state(ws: Path, did: str) -> dict[str, Any]:
    state = _generate_state(ws, did)
    for name, data in state["files"].items():
        if name.endswith("extraction_stats.json"):  # per-run telemetry: when, how long
            state["files"][name] = _timeless(json.loads(data))
    state["schema"] = DocSetStore(Workspace(root=ws)).get_schema(did)
    return state


def _assert_same_state(a: dict[str, Any], b: dict[str, Any]) -> None:
    """Equal, with a failure naming the first differing file."""
    assert sorted(a["files"]) == sorted(b["files"])
    for name in a["files"]:
        assert a["files"][name] == b["files"][name], name
    assert a["outputs"] == b["outputs"]
    assert a["rows"] == b["rows"]
    assert a["schema"] == b["schema"]


def _untiered(state: dict[str, Any]) -> dict[str, Any]:
    """*state* with its usage rows' tiers dropped, and every scope a batch run
    split into one row per tier (``usage.scope_events``) folded back into the
    one row the sync run writes — the labeling pass's, whose roster planning
    batches while open-vocabulary labeling stays synchronous."""
    sums = (
        "cost_usd",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
    )
    rows: list[dict[str, Any]] = []
    joined: dict[str, dict[str, Any]] = {}
    for row in state["rows"]:
        row = {k: v for k, v in row.items() if k != "tier"}
        context = dict(row["context"])
        if not context.pop("tier_split", False):
            rows.append(row)
            continue
        row["context"] = context
        key = json.dumps({k: v for k, v in row.items() if k not in sums}, sort_keys=True)
        if key not in joined:
            joined[key] = row
            rows.append(row)
            continue
        into = joined[key]
        for name in sums:
            if into.get(name) is not None or row.get(name) is not None:
                into[name] = (into.get(name) or 0) + (row.get(name) or 0)
    for row in rows:  # a re-associated float sum differs only past the 9th digit
        if row.get("cost_usd") is not None:
            row["cost_usd"] = round(row["cost_usd"], 9)
    rows.sort(key=lambda r: json.dumps(r, sort_keys=True))
    return {**state, "rows": rows}


def _neutral(payload: Any, did: str) -> Any:
    """*payload* with this workspace's DocSet id replaced, for cross-workspace
    comparison."""
    return json.loads(json.dumps(payload).replace(did, "<ds>"))


def _step(payload: dict[str, Any], name: str) -> dict[str, Any]:
    step = dict(payload["steps"][name])
    step.pop("batch", None)
    return step


def _install(provider: _Provider, *, polls: int) -> None:  # noqa: F811
    provider.install("anthropic", _answer, polls=polls)
    provider.install("gemini", _answer, polls=polls)


# ---- run == the three standalone commands ------------------------------------------------


def test_sync_run_equals_the_three_commands_in_sequence(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pdf_stubs: None,  # noqa: F811
) -> None:
    ws_r, did_r = _seed(tmp_path / "run", capsys)
    with patch("litellm.completion", side_effect=_sync):
        assert main(_run_argv(ws_r, did_r, "--schema-from", _SAMPLE)) == 0
    run = _read_stdout(capsys)
    assert list(run) == ["docset_id", "steps"]  # no `batch` block on a sync run
    assert list(run["steps"]) == ["schema", "generate", "extract"]

    ws_c, did_c = _seed(tmp_path / "commands", capsys)
    with patch("litellm.completion", side_effect=_sync):
        head = _ws_args(ws_c) + ["--debug"]
        assert main(head + ["extraction", "generate-schema", did_c, "--from-file", _SAMPLE]) == 0
        schema = _read_stdout(capsys)
        assert main(head + ["docset", "generate", did_c, *_GEN_FLAGS]) == 0
        generate = _read_stdout(capsys)
        assert main(head + ["extraction", "extract", did_c, "--all"]) == 0
        extract = _read_stdout(capsys)

    assert _neutral(run["steps"]["schema"], did_r) == _neutral(schema, did_c)
    assert _neutral(run["steps"]["generate"], did_r) == _neutral(generate, did_c)
    assert _neutral(run["steps"]["extract"], did_r) == _neutral(extract, did_c)
    _assert_same_state(_state(ws_r, did_r), _state(ws_c, did_c))


@pytest.mark.parametrize("closed", [False, True], ids=["open-vocab", "closed-vocab"])
def test_batch_run_equals_sync_run_except_tiers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,  # noqa: F811
    pdf_stubs: None,  # noqa: F811
    closed: bool,
) -> None:
    """Open vocabulary: roster planning batches, labeling stays synchronous.
    A closed vocabulary (`--schema-path`, passed through to `docset generate`)
    labels every document in one batch wave instead."""
    extra: list[str] = []
    if closed:
        vocab = tmp_path / "vocab.json"
        vocab.write_text(json.dumps({"Greeting": "a greeting line"}), encoding="utf-8")
        extra = ["--schema-path", str(vocab)]
    ws_s, did_s = _seed(tmp_path / "sync", capsys)
    with patch("litellm.completion", side_effect=_sync):
        assert main(_run_argv(ws_s, did_s, "--schema-from", _SAMPLE, *extra)) == 0
    sync = _read_stdout(capsys)

    ws_b, did_b = _seed(tmp_path / "batch", capsys)
    _install(provider, polls=0)
    with patch("litellm.completion", side_effect=_sync):
        argv = _run_argv(ws_b, did_b, "--schema-from", _SAMPLE, "--batch", *_POLL, *extra)
        assert main(argv) == 0
    batch = _read_stdout(capsys)

    for name in ("schema", "generate", "extract"):
        assert _neutral(_step(batch, name), did_b) == _neutral(_step(sync, name), did_s)
        assert "batch" in batch["steps"][name]
    assert _untiered(_state(ws_b, did_b)) == _untiered(_state(ws_s, did_s))
    tiers: dict[str, set[str]] = {}
    for r in _state(ws_b, did_b)["rows"]:
        tiers.setdefault(r["operation"], set()).add(r["tier"])
    assert tiers["schema_generate"] == tiers["transcribe"] == tiers["extract_values"] == {"batch"}
    assert tiers["label"] == ({"batch"} if closed else {"batch", "standard"})
    label_stage = batch["steps"]["generate"]["batch"]["stages"]["label"]
    assert label_stage["mode"] == ("all-at-once" if closed else "sync")

    summary = batch["batch"]
    assert summary["enabled"] is True
    assert list(summary["steps"]) == ["schema", "generate", "extract"]
    assert summary["steps"]["schema"]["waves"] == 1
    # Transcription: one wave per page window of the 3-page document; then
    # roster planning's draft + refine (open) or one labeling wave (closed);
    # links: propose + verify.
    generate_waves = 3 + (1 if closed else 2) + 2
    assert summary["steps"]["generate"]["waves"] == generate_waves
    assert summary["steps"]["extract"]["waves"] == 1
    assert summary["waves"] == 1 + generate_waves + 1
    assert summary["sync_fallbacks"] == 0
    for name in ("cost_usd", "standard_cost_usd", "saved_usd"):
        steps_total = sum(s[name] for s in summary["steps"].values())
        assert summary[name] == pytest.approx(steps_total)
    assert list_jobs(Workspace(root=ws_b)) == []  # the silent job is gone


# ---- one job across every step ----------------------------------------------------------


def test_no_wait_job_spans_every_step_and_matches_a_blocking_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,  # noqa: F811
    pdf_stubs: None,  # noqa: F811
) -> None:
    """One job id carries the whole pipeline: each `batch resume` advances one
    wave — the schema, each transcription window of the long document, the
    link waves, then extraction — and the finished job leaves exactly what a
    blocking --batch run leaves, every response billed once."""
    ws_b, did_b = _seed(tmp_path / "blocking", capsys)
    _install(provider, polls=0)
    with patch("litellm.completion", side_effect=_sync):
        assert main(_run_argv(ws_b, did_b, "--schema-from", _SAMPLE, "--batch", *_POLL)) == 0
    blocking = _read_stdout(capsys)
    blocking_state = _state(ws_b, did_b)

    ws_j, did_j = _seed(tmp_path / "job", capsys)
    _install(provider, polls=1)
    argv = _run_argv(ws_j, did_j, "--schema-from", _SAMPLE, "--batch", *_POLL, "--no-wait")
    steps_seen: list[str] = []
    with patch("litellm.completion", side_effect=_sync):
        assert main(argv) == 0
        pending = _read_stdout(capsys)
        assert pending["batch_job"]["command"] == "docset run"
        job_id = pending["batch_job"]["job_id"]
        # A paused run wrote nothing but job state.
        assert DocSetStore(Workspace(root=ws_j)).has_schema(did_j) is False
        for _ in range(20):
            assert main(_ws_args(ws_j) + ["batch", "status", job_id]) == 0
            steps_seen.append(_read_stdout(capsys)["step"])
            assert main(_ws_args(ws_j) + ["batch", "resume", job_id]) == 0
            out = _read_stdout(capsys)
            if "batch_job" not in out:
                break
            assert out["batch_job"]["job_id"] == job_id
        else:
            raise AssertionError("the job never completed")

    # One pause per wave: schema, 3 transcription windows, roster planning's
    # draft and refine, 2 link waves, extraction.
    assert steps_seen == ["schema"] + ["generate"] * 7 + ["extract"]
    assert len(provider.backends["anthropic"].submitted) == 1 + 3 + 2 + 2  # once each
    assert len(provider.backends["gemini"].submitted) == 1

    for name in ("schema", "generate", "extract"):
        assert _neutral(_step(out, name), did_j) == _neutral(_step(blocking, name), did_b)
    # Outputs, cache files, usage rows: each response billed exactly once,
    # including the link waves' (their plan cache was rewound, not hit).
    _assert_same_state(_state(ws_j, did_j), blocking_state)
    assert {r["operation"] for r in blocking_state["rows"]} >= {"links", "schema_generate"}
    (manifest,) = list_jobs(Workspace(root=ws_j))
    assert manifest.status == "completed"
    assert main(_ws_args(ws_j) + ["batch", "list"]) == 0
    assert _read_stdout(capsys)["jobs"][0]["step"] == "completed"  # kept through compaction
    # Later steps replayed earlier ones' responses from the job, at no cost.
    assert out["batch"]["steps"]["schema"]["waves"] == 1
    # The run's cost is the blocking run's, and the sum of its steps'.
    for name in ("cost_usd", "standard_cost_usd", "saved_usd"):
        assert out["batch"][name] == pytest.approx(blocking["batch"][name])
        steps_total = sum(s[name] for s in out["batch"]["steps"].values() if "skipped" not in s)
        assert out["batch"][name] == pytest.approx(steps_total)


# ---- skipped steps --------------------------------------------------------------------


def test_extraction_is_skipped_with_a_reason_without_a_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pdf_stubs: None,  # noqa: F811
) -> None:
    ws, did = _seed(tmp_path, capsys)
    with patch("litellm.completion", side_effect=_sync):
        assert main(_run_argv(ws, did)) == 0
    steps = _read_stdout(capsys)["steps"]
    assert "no --schema-from" in steps["schema"]["skipped"]
    assert steps["extract"] == {"skipped": "the docset has no extraction schema"}
    assert steps["generate"]["summary"]["converted"] == len(_PAGES)


def test_existing_schema_is_reused_and_steps_can_be_skipped(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    pdf_stubs: None,  # noqa: F811
) -> None:
    ws, did = _seed(tmp_path, capsys, schema=True)
    with patch("litellm.completion", side_effect=_sync):
        assert main(_run_argv(ws, did, "--no-generate")) == 0
    steps = _read_stdout(capsys)["steps"]
    assert steps["schema"] == {"skipped": "the docset already has an extraction schema"}
    assert steps["generate"] == {"skipped": "--no-generate"}
    assert steps["extract"]["summary"] == {"total": len(_FIDS), "ok": len(_FIDS), "failed": 0}

    with patch("litellm.completion", side_effect=_sync):
        assert main(_run_argv(ws, did, "--no-extract")) == 0
    steps = _read_stdout(capsys)["steps"]
    assert steps["schema"] == steps["extract"] == {"skipped": "--no-extract"}


# ---- failures --------------------------------------------------------------------------


def test_batch_unavailable_names_the_offending_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws, did = _seed(tmp_path, capsys, schema=True)
    bedrock = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    for flag, step in (("--values-model", "extract"), ("--model", "generate.transcribe")):
        with patch("litellm.completion", side_effect=AssertionError("no model call expected")):
            rc = main(_run_argv(ws, did, flag, bedrock, "--batch"))
        assert rc == 1
        err = _read_stderr(capsys)["error"]
        assert err["code"] == "BATCH_UNAVAILABLE"
        assert f"stage '{step}'" in err["message"]
    assert list_jobs(Workspace(root=ws)) == []  # rejected before any job existed


def test_unknown_values_effort_fails_the_run_before_any_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the extract step reads `--values-effort`, and it runs last: a bad
    value is refused up front, not after generate has paid."""
    ws, did = _seed(tmp_path, capsys, schema=True)
    with patch("litellm.completion", side_effect=AssertionError("no model call expected")):
        rc = main(_run_argv(ws, did, "--values-effort", "turbo"))
    assert rc == 1
    err = _read_stderr(capsys)["error"]
    assert err["code"] == "GROUNDED_CONFIG_INVALID"
    assert "--values-effort" in err["message"]


def test_a_step_that_fails_as_a_whole_fails_the_run_naming_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    main(_ws_args(ws) + ["docset", "create", "--name", "Empty"])
    did = str(_read_stdout(capsys)["id"])
    assert main(_ws_args(ws) + ["docset", "run", did]) == 1
    err = _read_stderr(capsys)["error"]
    assert err["code"] == "EMPTY_DOCSET"
    assert err["details"] == {"step": "generate", "completed_steps": ["schema"]}


def test_job_flags_need_batch_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws, did = _seed(tmp_path, capsys, schema=True)
    assert main(_ws_args(ws) + ["docset", "run", did, "--no-wait"]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "BATCH_JOB_INVALID"


# ---- the nondeterminism guard is fatal to the run ----------------------------------------


def test_the_drift_guard_stops_the_run_at_its_step_with_one_outcome(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,  # noqa: F811
    pdf_stubs: None,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resume whose transcription requests drifted (rebuilt differently on
    every run) while their batch is still in flight: `docset generate`
    soft-fails stage errors per file, which used to let the run print a
    success payload, go on to extraction and submit new billed batches. The
    guard is fatal instead: one outcome (its error envelope, exit 1, nothing
    on stdout), the run stops at `generate`, and nothing more is submitted."""
    ws, did = _seed(tmp_path, capsys)
    _install(provider, polls=1)
    argv = _run_argv(ws, did, "--schema-from", _SAMPLE, "--batch", *_POLL, "--no-wait")
    with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
        assert main(argv) == 0
        job_id = _read_stdout(capsys)["batch_job"]["job_id"]
        # Resume once: the schema wave is collected, a transcription wave submitted.
        assert main(_ws_args(ws) + ["batch", "resume", job_id]) == 0
        assert _read_stdout(capsys)["batch_job"]["job_id"] == job_id
    (manifest,) = list_jobs(Workspace(root=ws))
    assert manifest.state["run.step"] == "generate"
    anthropic = provider.backends["anthropic"]
    gemini = provider.backends["gemini"]
    submitted = (len(anthropic.submitted), len(gemini.submitted))

    digest = jobs_mod.request_digest

    def drifting(kwargs: Any) -> str:
        salt = "-drifted" if _system(kwargs) == transcribe_mod.SYSTEM_PROMPT else ""
        return digest(kwargs) + salt

    monkeypatch.setattr(jobs_mod, "request_digest", drifting)
    with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
        rc = main(_ws_args(ws) + ["batch", "resume", job_id])
    out = capsys.readouterr()
    assert rc == 1
    assert out.out == ""  # no success payload alongside the error
    error = json.loads(out.err[out.err.index('{\n  "error"') :])["error"]
    assert error["code"] == "BATCH_JOB_NONDETERMINISTIC"
    assert error["details"]["step"] == "generate"
    assert error["details"]["batch"]["job"] == {
        "job_id": job_id,
        "status": "failed",
        "resume": f"dgml batch resume {job_id}",
    }
    # Nothing new was paid for: not the drifted wave, not a later step.
    assert (len(anthropic.submitted), len(gemini.submitted)) == submitted
    assert not anthropic.canceled  # the in-flight batch is kept for the recovery
    (manifest,) = list_jobs(Workspace(root=ws))
    assert manifest.status == "failed" and manifest.state["run.step"] == "generate"
    assert "BatchJobNondeterministic" in (manifest.error or "")


# ---- pass-through options are validated before any step -------------------------------


@pytest.mark.parametrize(("flag", "value"), [("--window-size", "abc"), ("--thinking", "sometimes")])
def test_a_bad_pass_through_value_fails_before_any_step(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,  # noqa: F811
    flag: str,
    value: str,
) -> None:
    """Checked up front, as a JSON envelope — not argparse usage text and exit
    2 from the generate step after the schema step already paid."""
    ws, did = _seed(tmp_path, capsys)
    _install(provider, polls=0)
    argv = _ws_args(ws) + ["docset", "run", did, "--schema-from", _SAMPLE, flag, value]
    with patch("litellm.completion", side_effect=AssertionError("no model call expected")):
        assert main([*argv, "--batch", *_POLL]) == 1
    error = _read_stderr(capsys)["error"]
    assert error["code"] == "INVALID_ARGUMENT"
    assert "step 'generate'" in error["message"] and flag in error["message"]
    assert provider.backends["anthropic"].submitted == []
    assert provider.backends["gemini"].submitted == []
    assert list_jobs(Workspace(root=ws)) == []  # rejected before any job existed
    assert DocSetStore(Workspace(root=ws)).has_schema(did) is False


def _subparser(parser: argparse.ArgumentParser, *path: str) -> argparse.ArgumentParser:
    for name in path:
        (sub,) = (a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        parser = sub.choices[name]
    return parser


def test_run_pass_through_matches_the_step_parsers() -> None:
    """`_RUN_PASSTHROUGH` must name every option of each step's own command
    (bar the ones `docset run` sets itself), with the same dest and arity — so
    a flag added to `docset generate` or `extraction extract` cannot silently
    be missing from `docset run`, and a pass-through cannot name a flag its
    step no longer has."""
    root = _build_parser()
    steps = {
        "generate": ("docset", "generate"),
        "schema": ("extraction", "generate-schema"),
        "extract": ("extraction", "extract"),
    }
    # Common options, positionals, and what `docset run` passes itself.
    run_owned = {
        "help",
        "workspace",
        "workspace_config",
        "format",
        "verbose",
        "debug",
        "docset_id",
        "file_ids",
        "all",
        "from_files",
        "batch",
        "batch_poll_interval",
        "no_wait",
        "batch_deadline",
        "job",
    }
    run_flags = {a.dest: a for a in _subparser(root, "docset", "run")._actions if a.option_strings}
    for step, path in steps.items():
        actions = {a.dest: a for a in _subparser(root, *path)._actions}
        declared = {
            dest: (flag, takes_value)
            for flag, dest, takes_value, owner in _RUN_PASSTHROUGH
            if owner == step
        }
        assert set(declared) == set(actions) - run_owned, step
        for dest, (flag, takes_value) in declared.items():
            action = actions[dest]
            assert flag in action.option_strings, (step, flag)
            assert (action.nargs != 0) == takes_value, (step, flag)
            assert flag in run_flags[dest].option_strings, (step, flag)
            # A --flag/--no-flag pair stays one on both sides (and is declared so).
            boolean_optional = isinstance(action, argparse.BooleanOptionalAction)
            assert boolean_optional == (flag in _RUN_BOOLEAN_OPTIONAL), (step, flag)
            assert isinstance(run_flags[dest], argparse.BooleanOptionalAction) == boolean_optional
    # The pairs are a subset of the table (none today; F1 adds --batch-label).
    assert _RUN_BOOLEAN_OPTIONAL <= {f for f, _d, _v, _o in _RUN_PASSTHROUGH}


def test_run_step_argv_hands_each_step_only_its_own_options() -> None:
    root = _build_parser()
    args = root.parse_args(
        [
            "docset",
            "run",
            "ds",
            "--window-size",
            "2",
            "--no-semlinks",
            "--thinking",
            "adaptive",
            "--schema-model",
            "m1",
            "--values-model",
            "m2",
            "--values-effort",
            "low",
        ]
    )
    assert _run_step_argv(args, "generate") == [
        "--window-size",
        "2",
        "--thinking",
        "adaptive",
        "--no-semlinks",
    ]
    assert _run_step_argv(args, "schema") == ["--schema-model", "m1"]
    assert _run_step_argv(args, "extract") == ["--values-model", "m2", "--values-effort", "low"]


def test_a_non_dgml_step_exception_names_the_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import dgml.cli as cli

    ws, did = _seed(tmp_path, capsys, schema=True)

    def boom(*_a: object, **_kw: object) -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "_extraction_cmd", boom)
    assert main(_run_argv(ws, did, "--no-generate")) == 1
    err = _read_stderr(capsys)["error"]
    assert err["code"] == "INTERNAL_ERROR"
    assert "step 'extract'" in err["message"] and "kaboom" in err["message"]
    assert err["details"] == {"step": "extract", "completed_steps": ["generate", "schema"]}


def test_batch_steps_report_skipped_as_a_reason(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,  # noqa: F811
    pdf_stubs: None,  # noqa: F811
) -> None:
    ws, did = _seed(tmp_path, capsys, schema=True)
    _install(provider, polls=0)
    with patch("litellm.completion", side_effect=_sync):
        assert main(_run_argv(ws, did, "--no-generate", "--batch", *_POLL)) == 0
    steps = _read_stdout(capsys)["batch"]["steps"]
    assert steps["schema"] == {"skipped": "the docset already has an extraction schema"}
    assert steps["generate"] == {"skipped": "--no-generate"}
    assert "skipped" not in steps["extract"] and steps["extract"]["requests"] == len(_FIDS)


def test_batch_status_reports_the_step_of_a_run_job(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    provider: _Provider,  # noqa: F811
    pdf_stubs: None,  # noqa: F811
) -> None:
    """`batch list` names the step too; a non-run job carries no `step`."""
    ws, did = _seed(tmp_path, capsys, schema=True)
    _install(provider, polls=1)
    with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
        argv = _run_argv(ws, did, "--no-generate", "--batch", *_POLL, "--no-wait")
        assert main(argv) == 0
    job_id = _read_stdout(capsys)["batch_job"]["job_id"]
    assert main(_ws_args(ws) + ["batch", "list"]) == 0
    (summary,) = _read_stdout(capsys)["jobs"]
    assert summary["job_id"] == job_id and summary["step"] == "extract"
    assert main(_ws_args(ws) + ["batch", "cancel", job_id]) == 0
