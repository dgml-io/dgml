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

"""``dgml docset generate --batch``: pre-flight rejection, the batched link
pass, and the additive ``batch`` payload block.

``convert_batch`` is faked (as in ``test_cli.py``) so these tests exercise
what the CLI owns: when batch mode is on, what it rejects before any work,
and that the semantic-link pass run as one batch stage leaves the same DGML
and per-file results as the synchronous pass.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core.batch import FakeBackend, fake_model_response, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.generation import links as links_mod
from dgml_core.storage import Workspace

from .conftest import needs_gs
from .test_cli import (
    _init_with_docset,
    _read_generate_stdout,
    _read_stderr,
    _two_files_in_docset,
    _write_ws_config,
    _ws_args,
)

# Two sibling leaves, so the proposed link survives apply_plan.
_TREE = '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><a>same tree</a><b>and again</b></dg:chunk>'


def _tree_for(name: str) -> str:
    """A tree that reads differently per document (its own link-cache entry)."""
    return _TREE.replace("same tree", f"tree of {name}")


_LINKS_REPLY = {"links": [{"subject": "e0001", "object": "e0002", "predicate": "references"}]}
_VERDICTS_REPLY = {"verdicts": [{"i": 0, "keep": True}]}


def _link_answer(kwargs: dict[str, Any]) -> Any:
    """The link model: propose one link, keep it on review."""
    system = kwargs["messages"][0]["content"]
    if not isinstance(system, str):
        system = "".join(str(b.get("text", "")) for b in system)
    if system == links_mod.SYSTEM_PROMPT:
        text = json.dumps(_LINKS_REPLY)
    elif system == links_mod.VERIFY_SYSTEM_PROMPT:
        text = json.dumps(_VERDICTS_REPLY)
    else:
        raise AssertionError(f"unexpected request: {system[:60]!r}")
    return fake_model_response(text, cost=0.001)


def _fake_convert_with(tree: Any) -> Any:
    """Stand in for convert_batch: input *name* renders to ``tree(name)``; a
    batch run also reports a transcription stage, as the real pipeline does."""

    def fake(paths: Any, *, options: Any, on_output: Any, **_kw: Any) -> dict[str, str]:
        if options.batch is not None:
            options.batch.stats["transcribe"] = {"waves": 1, "requests": len(paths)}
        for p in paths:
            on_output(Path(p).name, tree(Path(p).name))
        return {}

    return fake


_fake_convert = _fake_convert_with(_tree_for)


@pytest.fixture
def fake_anthropic_batches() -> Iterator[list[FakeBackend]]:
    """A FakeBackend standing in for Anthropic Message Batches; restored after."""
    saved = dict(batch_registry._REGISTRY)
    built: list[FakeBackend] = []

    def factory(_cfg: Any) -> FakeBackend:
        backend = FakeBackend(lambda request: _link_answer(request.kwargs), provider="anthropic")
        built.append(backend)
        return backend

    register_backend("anthropic", factory)
    try:
        yield built
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


def _generate(ws: Path, did: str, *flags: str) -> int:
    base = ["docset", "generate", did, "--no-coverage", "--max-parallel-calls", "1"]
    return main(_ws_args(ws) + base + list(flags))


def test_batch_rejects_a_provider_with_no_batch_backend_before_any_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-dummy")
    _write_ws_config(
        ws,
        {
            "generation": {
                "model": "openrouter/anthropic/claude-sonnet-4",
                "label_model": "anthropic/claude-sonnet-4-6",
            }
        },
    )
    capsys.readouterr()
    with patch("dgml_core.generation.convert_batch") as convert:
        rc = _generate(ws, did, "--batch")
    assert rc == 1
    err = _read_stderr(capsys)
    assert err["error"]["code"] == "BATCH_UNAVAILABLE"
    assert "stage 'transcribe'" in err["error"]["message"]
    assert "openrouter" in err["error"]["message"]
    convert.assert_not_called()


def test_batch_checks_the_label_model_under_an_open_vocabulary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Roster planning and concept descriptions batch over the label model
    under every vocabulary, so even an open-vocabulary run without the link
    pass is rejected up front when that model has no batch backend."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")  # never used: no call is made
    _write_ws_config(
        ws,
        {
            "generation": {
                "model": "anthropic/claude-haiku-4-5",
                "label_model": "mistral/mistral-large-latest",
            }
        },
    )
    capsys.readouterr()
    with patch("dgml_core.generation.convert_batch") as convert:
        rc = _generate(ws, did, "--batch", "--no-semlinks")
    assert rc == 1
    err = _read_stderr(capsys)["error"]
    assert err["code"] == "BATCH_UNAVAILABLE"
    assert "stage 'label'" in err["message"]
    convert.assert_not_called()


def test_config_batch_is_overridden_by_no_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``[generation] batch = true`` turns batch mode on; ``--no-batch`` wins."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-dummy")
    _write_ws_config(
        ws,
        {
            "generation": {
                "model": "openrouter/anthropic/claude-sonnet-4",
                "label_model": "anthropic/claude-sonnet-4-6",
                "batch": True,
            }
        },
    )
    capsys.readouterr()
    with patch("dgml_core.generation.convert_batch") as convert:
        assert _generate(ws, did, "--no-semlinks") == 1  # config says batch: rejected
    assert _read_stderr(capsys)["error"]["code"] == "BATCH_UNAVAILABLE"
    convert.assert_not_called()
    with patch("dgml_core.generation.convert_batch", return_value={}) as convert:
        _generate(ws, did, "--no-semlinks", "--no-batch")
    convert.assert_called_once()
    assert convert.call_args.kwargs["options"].batch is None


def _run_both(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], convert: Any
) -> dict[str, dict[str, Any]]:
    """Generate the same two documents sync and under --batch; collect results."""
    runs: dict[str, dict[str, Any]] = {}
    for mode in ("sync", "batch"):
        root = tmp_path / mode
        root.mkdir()
        ws = root / "ws"
        did = _init_with_docset(ws, capsys)
        _two_files_in_docset(ws, root, did, capsys)
        capsys.readouterr()
        flags = ["--batch", "--batch-poll-interval", "0.01"] if mode == "batch" else []
        calls: list[dict[str, Any]] = []

        def model(kwargs: dict[str, Any], *, calls: list[Any] = calls, **_kw: Any) -> Any:
            calls.append(kwargs)
            return _link_answer(kwargs)

        with (
            patch("dgml_core.generation.convert_batch", side_effect=convert),
            patch("dgml_core.llm._completion_with_retry", side_effect=model),
        ):
            assert _generate(ws, did, *flags) == 0
        payload = _read_generate_stdout(capsys)
        store = Workspace(root=ws).blobs
        runs[mode] = {
            "payload": payload,
            "sync_calls": len(calls),
            "xml": sorted(
                store.get_blob(r["output"]).decode("utf-8")
                for r in payload["results"]
                if r["status"] == "converted"
            ),
            "results": sorted(
                (r["source"], r["links"], r.get("link_error")) for r in payload["results"]
            ),
        }
    return runs


def _without_batch(payload: dict[str, Any]) -> dict[str, Any]:
    """The payload minus the batch block and the per-workspace ids."""
    drop = {"batch", "results", "docset_id", "output_key"}
    return {k: v for k, v in payload.items() if k not in drop}


@needs_gs
def test_batched_link_pass_matches_the_synchronous_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_anthropic_batches: list[FakeBackend],
) -> None:
    """Same documents, same link model: the batch link stage writes the same
    DGML and reports the same per-file results as the per-document sync pass,
    and only the batch run's payload carries the ``batch`` block."""
    runs = _run_both(tmp_path, capsys, _fake_convert)
    sync, batch = runs["sync"], runs["batch"]
    assert "batch" not in sync["payload"]
    assert batch["results"] == sync["results"]
    assert all(links == 1 for _src, links, _err in sync["results"])
    assert batch["xml"] == sync["xml"]
    assert all('dg:itemprop="references"' in xml for xml in batch["xml"])
    assert _without_batch(batch["payload"]) == _without_batch(sync["payload"])

    block = batch["payload"]["batch"]
    assert block["enabled"] is True
    assert block["stages"]["transcribe"] == {"waves": 1, "requests": 2}
    links_stage = block["stages"]["links"]
    # Both documents' propose requests in one wave, then both verify requests.
    assert links_stage["waves"] == 2
    assert links_stage["requests"] == 4
    assert links_stage["batch_ok"] == 4
    assert batch["sync_calls"] == 0  # every link request went through the batch
    assert sync["sync_calls"] == 4
    assert fake_anthropic_batches and fake_anthropic_batches[0].submitted


@needs_gs
def test_identical_documents_share_one_link_plan_under_batch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_anthropic_batches: list[FakeBackend],
) -> None:
    """Two documents that read the same to the model cost one plan in the sync
    pass (the second hits the cache entry the first wrote). The batch stage
    looks the cache up before either plan exists, so it must de-duplicate on
    the cache key itself — or batch mode would pay twice for what sync pays
    once."""
    runs = _run_both(tmp_path, capsys, _fake_convert_with(lambda _name: _TREE))
    sync, batch = runs["sync"], runs["batch"]
    assert batch["results"] == sync["results"]
    assert batch["xml"] == sync["xml"]
    assert sync["sync_calls"] == 2  # one propose + one verify, then a cache hit
    links_stage = batch["payload"]["batch"]["stages"]["links"]
    assert links_stage["requests"] == 2  # one shared plan, not one per document


@needs_gs
def test_batch_with_no_semlinks_reports_the_link_stage_skipped(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_anthropic_batches: list[FakeBackend],
) -> None:
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)
    capsys.readouterr()
    with patch("dgml_core.generation.convert_batch", side_effect=_fake_convert):
        assert _generate(ws, did, "--batch", "--no-semlinks") == 0
    payload = _read_generate_stdout(capsys)
    assert payload["batch"]["stages"]["links"] == {"skipped": "--no-semlinks"}
    assert {r["links"] for r in payload["results"] if r["status"] == "converted"} == {0}
    assert not any(b.submitted for b in fake_anthropic_batches)


_PRIOR_EXTRACTION = (
    '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" xmlns:docset="http://www.dgml.io/ws/T">'
    "<dg:extraction>"
    '<docset:VendorName dg:origin="1 10 20 30 40">Acme</docset:VendorName>'
    "</dg:extraction></dg:chunk>"
)


def _seed_prior_extraction(ws: Path, did: str) -> list[str]:
    """Give every file in *did* an extraction-only DGML (a prior `extraction
    extract`), which generate must carry over into its fresh render."""
    from dgml_core import layout
    from dgml_core.docsets import DocSetStore
    from dgml_core.files import FileStore

    wsx = Workspace(root=ws)
    keys = []
    for fid in DocSetStore(wsx).list_files(did):
        stem = Path(FileStore(wsx).get(fid).original_filename).stem
        key = layout.dgml_xml_key(did, fid, stem)
        wsx.blobs.put_blob(key, _PRIOR_EXTRACTION.encode())
        keys.append(key)
    return keys


class _PollFails(FakeBackend):
    """A provider whose batch was accepted but can no longer be polled."""

    def poll(self, job: Any) -> Any:
        raise RuntimeError("provider unreachable while polling")


def _boom(_kwargs: dict[str, Any], **_kw: Any) -> Any:
    raise RuntimeError("provider unreachable while polling")


@needs_gs
def test_a_failed_link_stage_loses_no_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A batch-level failure of the link stage (here: polling raises) must
    behave like a failed sync link pass: every document keeps its DGML, its
    prior dg:extraction is carried over, it reports a link_error, and the
    command still emits its payload with rc 0. Before the fix the failure
    escaped after grounding had overwritten each blob and before the
    extraction was re-embedded, so the extracted values were lost."""
    saved = dict(batch_registry._REGISTRY)
    register_backend(
        "anthropic",
        lambda _cfg: _PollFails(lambda r: _link_answer(r.kwargs), provider="anthropic"),
    )
    try:
        runs: dict[str, dict[str, Any]] = {}
        for mode in ("sync", "batch"):
            root = tmp_path / mode
            root.mkdir()
            ws = root / "ws"
            did = _init_with_docset(ws, capsys)
            _two_files_in_docset(ws, root, did, capsys)
            keys = _seed_prior_extraction(ws, did)
            capsys.readouterr()
            flags = ["--batch", "--batch-poll-interval", "0.01"] if mode == "batch" else []
            with (
                patch("dgml_core.generation.convert_batch", side_effect=_fake_convert),
                patch("dgml_core.llm._completion_with_retry", side_effect=_boom),
            ):
                rc = _generate(ws, did, *flags)
            payload = _read_generate_stdout(capsys)
            store = Workspace(root=ws).blobs
            runs[mode] = {
                "rc": rc,
                "payload": payload,
                "xml": [store.get_blob(k).decode("utf-8") for k in keys],
            }
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)

    sync, batch = runs["sync"], runs["batch"]
    assert batch["rc"] == sync["rc"] == 0
    for run in (sync, batch):
        converted = [r for r in run["payload"]["results"] if r["status"] == "converted"]
        assert len(converted) == 2
        assert all(r["links"] == 0 and r["link_error"] for r in converted)
        for xml in run["xml"]:
            assert "tree of" in xml  # the fresh render is there
            assert ">Acme</docset:VendorName>" in xml  # the extraction survived
    assert all(
        "provider unreachable while polling" in r["link_error"] for r in batch["payload"]["results"]
    )


@needs_gs
def test_a_follower_is_not_stranded_when_the_owner_request_cannot_be_built(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fake_anthropic_batches: list[FakeBackend],
) -> None:
    """Two documents share one link-cache key. If building the first one's
    request raises, the second must not be left waiting on a request that
    never exists: it becomes the owner and is linked, as in the sync pass,
    where each document tries on its own."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)
    capsys.readouterr()
    real_steps = links_mod.plan_links_steps
    built: list[str] = []

    def flaky_steps(source: str, config: Any, **kw: Any) -> Any:
        built.append(config.context["doc"])
        if len(built) == 1:
            raise RuntimeError("could not build the request")
        return real_steps(source, config, **kw)

    same_tree = _fake_convert_with(lambda _name: _TREE)
    with (
        patch("dgml_core.generation.convert_batch", side_effect=same_tree),
        patch.object(links_mod, "plan_links_steps", side_effect=flaky_steps),
    ):
        assert _generate(ws, did, "--batch", "--batch-poll-interval", "0.01") == 0
    payload = _read_generate_stdout(capsys)
    by_source = {r["source"]: r for r in payload["results"] if r["status"] == "converted"}
    first, second = built
    assert "could not build the request" in by_source[first]["link_error"]
    assert by_source[second]["links"] == 1
    assert "link_error" not in by_source[second]


def test_batch_env_var_turns_batch_mode_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DGML_GENERATION__BATCH`` arrives as a string; ``true`` must enable batch
    mode (proven by the up-front BATCH_UNAVAILABLE for an unbatchable model),
    and ``False`` must leave it off."""
    ws = tmp_path / "ws"
    did = _init_with_docset(ws, capsys)
    _two_files_in_docset(ws, tmp_path, did, capsys)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-dummy")
    _write_ws_config(
        ws,
        {
            "generation": {
                "model": "openrouter/anthropic/claude-sonnet-4",
                "label_model": "anthropic/claude-sonnet-4-6",
            }
        },
    )
    capsys.readouterr()
    monkeypatch.setenv("DGML_GENERATION__BATCH", "true")
    with patch("dgml_core.generation.convert_batch") as convert:
        assert _generate(ws, did, "--no-semlinks") == 1
    assert _read_stderr(capsys)["error"]["code"] == "BATCH_UNAVAILABLE"
    convert.assert_not_called()
    monkeypatch.setenv("DGML_GENERATION__BATCH", "False")
    with patch("dgml_core.generation.convert_batch", return_value={}) as convert:
        _generate(ws, did, "--no-semlinks")
    assert convert.call_args.kwargs["options"].batch is None
