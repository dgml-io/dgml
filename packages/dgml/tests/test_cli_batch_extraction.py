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

"""``--batch`` on ``extraction extract`` and ``file add <dir>``.

The provider's batch backend is replaced by a :class:`FakeBackend` answering
with the same function the synchronous path's patched ``litellm.completion``
uses, so a batch run and a sync run can be compared entry for entry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core import layout
from dgml_core.batch import FakeBackend, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.docsets import DocSetStore
from dgml_core.storage import Workspace
from dgml_core.usage import TIER_BATCH, read_events

from .conftest import _write_text_pdf, dump_toml, needs_gs
from .test_cli import (
    _RNC_SCHEMA,
    _init_ws,
    _new_docset,
    _read_stderr,
    _read_stdout,
    _seed_file_dir,
    _tool_response,
    _ws_args,
)

Respond = Callable[..., Any]


class _Fakes:
    """The fake backends the tests installed, by provider."""

    def __init__(self) -> None:
        self.by_provider: dict[str, list[FakeBackend]] = {}

    def install(self, provider: str, respond: Respond) -> None:
        def factory(_cfg: Any) -> FakeBackend:
            backend = FakeBackend(lambda req: respond(**req.kwargs), provider=provider)
            self.by_provider.setdefault(provider, []).append(backend)
            return backend

        register_backend(provider, factory)

    def submitted(self, provider: str) -> list[int]:
        return [len(b) for backend in self.by_provider.get(provider, []) for b in backend.submitted]


@pytest.fixture
def fakes() -> Iterator[_Fakes]:
    saved = dict(batch_registry._REGISTRY)
    yield _Fakes()
    batch_registry._REGISTRY.clear()
    batch_registry._REGISTRY.update(saved)


def _write_config(ws: Path, *, classification: bool = False, grounded: bool = True) -> None:
    data: dict[str, Any] = {}
    if grounded:
        data["grounded"] = {
            "schema_model": "anthropic/claude-opus-4-7",
            "values_model": "anthropic/claude-sonnet-4-6",
        }
    if classification:
        data["classification"] = {"model": "anthropic/claude-haiku-4-5", "max_pages": 1}
    Workspace(root=ws).config_path.write_text(dump_toml(data), encoding="utf-8")


def _set_schema(ws: Path, tmp_path: Path, ds_id: str, capsys: pytest.CaptureFixture[str]) -> None:
    schema_file = tmp_path / "schema.rnc"
    schema_file.write_text(_RNC_SCHEMA, encoding="utf-8")
    assert (
        main(_ws_args(ws) + ["extraction", "set-schema", ds_id, "--schema-file", str(schema_file)])
        == 0
    )
    capsys.readouterr()


def _values_response(**_kwargs: Any) -> SimpleNamespace:
    # Empty locations → nothing for phase 2/3 to locate: one request per file.
    return _tool_response(
        "submit_values", {"values": {"VendorName": {"text": "Acme", "locations": []}}}
    )


def _seed_extraction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fids: list[str]
) -> tuple[Path, str]:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    _write_config(ws)
    ds_id = _new_docset(ws, capsys)
    _set_schema(ws, tmp_path, ds_id, capsys)
    store = DocSetStore(Workspace(root=ws))
    for fid in fids:
        _seed_file_dir(ws, fid, pages=1)
        store.add_file(ds_id, fid)  # the store, not the CLI: no auto-extract
    return ws, ds_id


# ---- extraction extract ---------------------------------------------------------


def test_single_file_form_is_unchanged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, [])
    _seed_file_dir(ws, "fsingle00001", pages=1)
    with patch("litellm.completion", side_effect=_values_response):
        assert main(_ws_args(ws) + ["extraction", "extract", ds_id, "fsingle00001"]) == 0
    payload = _read_stdout(capsys)
    assert list(payload) == [
        "docset_id",
        "file_id",
        "model",
        "mode",
        "tool_calls",
        "field_count",
        "xml_key",
    ]


def test_all_sync_reports_each_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fids = ["fall00000001", "fall00000002"]
    ws, ds_id = _seed_extraction(tmp_path, capsys, fids)
    with patch("litellm.completion", side_effect=_values_response):
        assert main(_ws_args(ws) + ["extraction", "extract", ds_id, "--all"]) == 0
    payload = _read_stdout(capsys)
    assert "batch" not in payload
    assert payload["summary"] == {"total": 2, "ok": 2, "failed": 0}
    assert [r["file_id"] for r in payload["results"]] == fids
    assert all(r["status"] == "ok" and r["field_count"] == 1 for r in payload["results"])


def test_all_batch_matches_sync_and_adds_a_batch_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fakes: _Fakes
) -> None:
    fids = ["fall00000001", "fall00000002", "fall00000003"]
    ws, ds_id = _seed_extraction(tmp_path, capsys, fids)
    with patch("litellm.completion", side_effect=_values_response):
        assert main(_ws_args(ws) + ["extraction", "extract", ds_id, *fids]) == 0
    sync = _read_stdout(capsys)

    fakes.install("anthropic", _values_response)
    with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
        assert (
            main(_ws_args(ws) + ["--debug", "extraction", "extract", ds_id, "--all", "--batch"])
            == 0
        )
    batch = _read_stdout(capsys)
    assert batch["results"] == sync["results"]
    assert batch["summary"] == sync["summary"]
    assert batch["batch"]["provider"] == "anthropic"
    assert batch["batch"]["requests"] == 3 and batch["batch"]["sync_fallbacks"] == 0
    assert fakes.submitted("anthropic") == [3]
    rows = read_events(Workspace(root=ws))
    assert [r["tier"] for r in rows] == [TIER_BATCH] * 3


def test_batch_isolates_a_failing_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fakes: _Fakes
) -> None:
    fids = ["fgood0000001", "fbad00000002"]
    ws, ds_id = _seed_extraction(tmp_path, capsys, fids)
    Workspace(root=ws).blobs.delete_blob(layout.file_source_key("fbad00000002", "doc.pdf"))
    fakes.install("anthropic", _values_response)
    # With the bad file out at setup, the good file's phase-1 wave is a single
    # request. Batch mode batches even a lone request (min_wave_size defaults to
    # 1), so it is served by the batch backend and never by the synchronous path.
    with patch("litellm.completion", side_effect=_values_response) as sync_call:
        assert main(_ws_args(ws) + ["extraction", "extract", ds_id, *fids, "--batch"]) == 0
    assert sync_call.call_count == 0
    payload = _read_stdout(capsys)
    assert payload["batch"]["sync_fallbacks"] == 0
    assert payload["batch"]["batch_ok"] >= 1
    good, bad = payload["results"]
    assert good["status"] == "ok"
    assert bad["status"] == "failed" and set(bad["error"]) == {"code", "message"}
    assert payload["summary"] == {"total": 2, "ok": 1, "failed": 1}


def test_batch_unavailable_is_rejected_before_any_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fall00000001"])
    with patch("litellm.completion") as mock_completion:
        rc = main(
            _ws_args(ws)
            + [
                "extraction",
                "extract",
                ds_id,
                "--all",
                "--batch",
                "--values-model",
                "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
            ]
        )
    assert rc == 1
    err = _read_stderr(capsys)["error"]
    assert err["code"] == "BATCH_UNAVAILABLE"
    assert "extraction" in err["message"]
    mock_completion.assert_not_called()


@pytest.mark.parametrize("extra", [[], ["fall00000001", "--all"]])
def test_extract_argument_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fall00000001"])
    assert main(_ws_args(ws) + ["extraction", "extract", ds_id, *extra]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "INVALID_ARGUMENT"


# ---- file add <dir> --batch ---------------------------------------------------


def _seed_classify_ws(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], name: str, *, grounded: bool = False
) -> tuple[Path, str, str, Path]:
    ws = tmp_path / name
    _init_ws(ws)
    capsys.readouterr()
    _write_config(ws, classification=True, grounded=grounded)
    ids = []
    for ds_name in ("Contracts", "Safety Datasheets"):
        main(_ws_args(ws) + ["docset", "create", "--name", ds_name, "--key-question", "What?"])
        ids.append(_read_stdout(capsys)["id"])
    src = tmp_path / f"{name}-pdfs"
    src.mkdir()
    _write_text_pdf(src / "a.pdf", ["Alpha page one"])
    _write_text_pdf(src / "b.pdf", ["Bravo page one"])
    return ws, ids[0], ids[1], src


def _assign_to(docset_id: str) -> Respond:
    def respond(**kwargs: Any) -> SimpleNamespace:
        assert [t["function"]["name"] for t in kwargs["tools"]] == ["assign_to_existing_docset"]
        return _tool_response("assign_to_existing_docset", {"docset_id": docset_id})

    return respond


def _classification_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-entry classification blocks, with the run-specific DocSet id stripped."""
    out = []
    for entry in payload["results"]:
        block = dict(entry["classification"])
        block.pop("docset_id", None)
        out.append(block)
    return out


@needs_gs
def test_directory_batch_classifies_in_one_wave_with_payload_parity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fakes: _Fakes
) -> None:
    ws_s, contracts_s, _, src_s = _seed_classify_ws(tmp_path, capsys, "sync")
    with patch("litellm.completion", side_effect=_assign_to(contracts_s)):
        assert (
            main(_ws_args(ws_s) + ["file", "add", str(src_s), "--auto-classify", "existing"]) == 0
        )
    sync = _read_stdout(capsys)

    ws_b, contracts_b, _, src_b = _seed_classify_ws(tmp_path, capsys, "batch")
    fakes.install("anthropic", _assign_to(contracts_b))
    with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
        rc = main(
            _ws_args(ws_b) + ["file", "add", str(src_b), "--auto-classify", "existing", "--batch"]
        )
    assert rc == 0
    batch = _read_stdout(capsys)
    assert _classification_blocks(batch) == _classification_blocks(sync)
    assert all(e["classification"]["docset_id"] == contracts_b for e in batch["results"])
    assert batch["summary"] == sync["summary"]
    assert "batch" not in sync
    assert batch["batch"]["classification"]["requests"] == 2
    assert fakes.submitted("anthropic") == [2]
    main(_ws_args(ws_b) + ["docset", "list-files", contracts_b])
    assert len(_read_stdout(capsys)["file_ids"]) == 2


@needs_gs
def test_directory_batch_auto_extracts_as_a_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fakes: _Fakes
) -> None:
    ws, contracts, _, src = _seed_classify_ws(tmp_path, capsys, "extract", grounded=True)
    _set_schema(ws, tmp_path, contracts, capsys)

    assign = _assign_to(contracts)

    def respond(**kwargs: Any) -> SimpleNamespace:
        name = kwargs["tools"][0]["function"]["name"]
        return assign(**kwargs) if name == "assign_to_existing_docset" else _values_response()

    fakes.install("anthropic", respond)
    rc = main(_ws_args(ws) + ["file", "add", str(src), "--auto-classify", "existing", "--batch"])
    assert rc == 0
    payload = _read_stdout(capsys)
    for entry in payload["results"]:
        block = entry["classification"]
        assert list(block)[-1] == "extraction"  # the key lands last, as in the sync path
        assert block["extraction"] == {
            "performed": True,
            "model": "anthropic/claude-sonnet-4-6",
            "tool_calls": 0,
            "error": None,
        }
    extraction = payload["batch"]["extraction"]
    assert extraction["docset_ids"] == [contracts] and extraction["requests"] == 2


@needs_gs
def test_directory_batch_requires_auto_classify_before_adding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws, _, _, src = _seed_classify_ws(tmp_path, capsys, "mode")
    assert main(_ws_args(ws) + ["file", "add", str(src), "--batch"]) == 1
    assert _read_stderr(capsys)["error"]["code"] == "INVALID_ARGUMENT"
    main(_ws_args(ws) + ["file", "list"])
    assert _read_stdout(capsys)["files"] == []  # nothing was added


def _new_or_existing(**kwargs: Any) -> SimpleNamespace:
    """a.pdf founds a "Letters" DocSet; b.pdf, classified after it, must be
    offered that DocSet (by id) and joins it."""
    import base64
    import re

    names = [t["function"]["name"] for t in kwargs["tools"]]
    assert names == ["assign_to_existing_docset", "create_new_docset"]
    text = kwargs["messages"][0]["content"][0]["text"]
    letters = re.search(r"- id=(\S+)\n  name: Letters\n", text)
    if letters is None:
        return _tool_response(
            "create_new_docset",
            {"name": "Letters", "description": "letters", "key_questions": ["Who wrote it?"]},
        )
    image = kwargs["messages"][0]["content"][1]["image_url"]["url"]
    assert base64.b64decode(image.split(",", 1)[1])  # a rendered page
    return _tool_response("assign_to_existing_docset", {"docset_id": letters.group(1)})


@needs_gs
def test_directory_batch_default_mode_classifies_in_order_like_sync(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fakes: _Fakes
) -> None:
    """The default mode under --batch: one wave per file, in order, each
    request built after the DocSets the files before it created exist — the
    same decisions, DocSets and payload as the synchronous run."""
    ws_s, _, _, src_s = _seed_classify_ws(tmp_path, capsys, "sync")
    with patch("litellm.completion", side_effect=_new_or_existing):
        assert main(_ws_args(ws_s) + ["file", "add", str(src_s), "--auto-classify"]) == 0
    sync = _read_stdout(capsys)

    ws_b, _, _, src_b = _seed_classify_ws(tmp_path, capsys, "batch")
    fakes.install("anthropic", _new_or_existing)
    with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
        rc = main(_ws_args(ws_b) + ["file", "add", str(src_b), "--auto-classify", "--batch"])
    assert rc == 0
    batch = _read_stdout(capsys)
    assert _classification_blocks(batch) == _classification_blocks(sync)
    assert [e["classification"]["decision"] for e in batch["results"]] == ["new", "existing"]
    letters = batch["results"][0]["classification"]["docset_id"]
    assert batch["results"][1]["classification"]["docset_id"] == letters
    assert batch["summary"] == sync["summary"]
    assert fakes.submitted("anthropic") == [1, 1]  # one wave per file
    assert batch["batch"]["classification"]["waves"] == 2
    main(_ws_args(ws_b) + ["docset", "list-files", letters])
    assert len(_read_stdout(capsys)["file_ids"]) == 2


@needs_gs
def test_directory_batch_unavailable_before_adding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws, _, _, src = _seed_classify_ws(tmp_path, capsys, "unavail")
    Workspace(root=ws).config_path.write_text(
        dump_toml(
            {"classification": {"model": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"}}
        ),
        encoding="utf-8",
    )
    rc = main(_ws_args(ws) + ["file", "add", str(src), "--auto-classify", "existing", "--batch"])
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "BATCH_UNAVAILABLE"
    main(_ws_args(ws) + ["file", "list"])
    assert _read_stdout(capsys)["files"] == []


@needs_gs
def test_single_file_add_rejects_batch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws, _, _, src = _seed_classify_ws(tmp_path, capsys, "single")
    rc = main(
        _ws_args(ws) + ["file", "add", str(src / "a.pdf"), "--auto-classify", "existing", "--batch"]
    )
    assert rc == 1
    assert _read_stderr(capsys)["error"]["code"] == "INVALID_ARGUMENT"


# ---- review fixes: unset credentials soft-fail per file, exactly as sync ---------

_UNSET = "DGML_TEST_UNSET_BATCH_KEY"


def _config_with(ws: Path, *, classification_env: bool = False, values_env: bool = False) -> None:
    classification: dict[str, Any] = {"model": "anthropic/claude-haiku-4-5", "max_pages": 1}
    grounded: dict[str, Any] = {
        "schema_model": "anthropic/claude-opus-4-7",
        "values_model": "anthropic/claude-sonnet-4-6",
    }
    if classification_env:
        classification["api_key_env"] = _UNSET
    if values_env:
        grounded["values_api_key_env"] = _UNSET
    Workspace(root=ws).config_path.write_text(
        dump_toml({"classification": classification, "grounded": grounded}), encoding="utf-8"
    )


def _no_request(**_kwargs: Any) -> Any:
    raise AssertionError("no request expected")


@needs_gs
def test_directory_batch_unset_classification_key_soft_fails_like_sync(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fakes: _Fakes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_UNSET, raising=False)
    payloads = []
    for name, extra in (("sync", []), ("batch", ["--batch"])):
        ws, _, _, src = _seed_classify_ws(tmp_path, capsys, name)
        _config_with(ws, classification_env=True)
        fakes.install("anthropic", _no_request)
        with patch("litellm.completion", side_effect=_no_request):
            rc = main(
                _ws_args(ws) + ["file", "add", str(src), "--auto-classify", "existing", *extra]
            )
        assert rc == 0
        payloads.append(_read_stdout(capsys))
    sync, batch = payloads
    assert _classification_blocks(batch) == _classification_blocks(sync)
    assert all(e["classification"]["error"].startswith("AUTH_") for e in batch["results"])
    assert batch["summary"] == sync["summary"] and batch["summary"]["added"] == 2


@needs_gs
def test_directory_batch_unset_values_key_soft_fails_extraction_like_sync(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fakes: _Fakes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_UNSET, raising=False)
    extraction_blocks = []
    for name, extra in (("sync", []), ("batch", ["--batch"])):
        ws, contracts, _, src = _seed_classify_ws(tmp_path, capsys, name)
        _config_with(ws, values_env=True)
        _set_schema(ws, tmp_path, contracts, capsys)
        assign = _assign_to(contracts)
        fakes.install("anthropic", assign)
        with patch("litellm.completion", side_effect=assign):
            rc = main(
                _ws_args(ws) + ["file", "add", str(src), "--auto-classify", "existing", *extra]
            )
        assert rc == 0
        payload = _read_stdout(capsys)
        assert all(e["classification"]["docset_id"] == contracts for e in payload["results"])
        extraction_blocks.append([e["classification"]["extraction"] for e in payload["results"]])
    sync, batch = extraction_blocks
    assert batch == sync
    assert all(b["error"].startswith("AUTH_") and b["model"] for b in batch)


def test_extract_batch_unset_values_key_reports_each_file_like_sync(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_UNSET, raising=False)
    ws, ds_id = _seed_extraction(tmp_path, capsys, ["fall00000001", "fall00000002"])
    _config_with(ws, values_env=True)
    results = []
    for extra in ([], ["--batch"]):
        with patch("litellm.completion", side_effect=_no_request):
            assert main(_ws_args(ws) + ["extraction", "extract", ds_id, "--all", *extra]) == 0
        results.append(_read_stdout(capsys)["results"])
    assert results[1] == results[0]
    assert all(
        r["status"] == "failed" and r["error"]["code"].startswith("AUTH") for r in results[1]
    )
