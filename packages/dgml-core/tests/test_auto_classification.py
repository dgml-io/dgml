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

""":mod:`dgml_core.auto_classification`: classify-and-assign as a library.

The batch path (:func:`prepare_bulk_classify` + :func:`classify_bulk_batch`)
must write, for every file, the ``classification`` block the synchronous
:func:`auto_classify` writes in assign-only mode — in one wave.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from dgml_core import auto_classify, classify_bulk_batch, prepare_bulk_classify
from dgml_core.auto_classification import classification_block, soft_error
from dgml_core.batch import FakeBackend, provider_of, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.classification import ClassificationConfig
from dgml_core.docsets import DocSetStore
from dgml_core.errors import BatchUnavailable, ClassificationFailed
from dgml_core.files import AddFileResult, FileStore
from dgml_core.storage import Workspace

from .test_classification import DEFAULT_TEST_MODEL, _tool_call_response
from .test_classification_batch import _seed_many


@pytest.fixture(autouse=True)
def _registry() -> Iterator[None]:
    saved = dict(batch_registry._REGISTRY)
    try:
        yield
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


def _ws(root: Path) -> Workspace:
    ws = Workspace(root=root)
    ws.root.mkdir(parents=True, exist_ok=True)
    return ws


def _added(ws: Workspace, fids: list[str]) -> list[AddFileResult]:
    files = FileStore(ws)
    return [AddFileResult(record=files.get(fid), created=True) for fid in fids]


def _neutral(blocks: list[dict[str, Any]], docset_id: str) -> Any:
    return json.loads(json.dumps(blocks).replace(docset_id, "<ds>"))


def test_batch_blocks_match_sync_assign_only(tmp_path: Path, capture_kwargs: Any) -> None:
    cfg = ClassificationConfig(model=DEFAULT_TEST_MODEL)

    ws_s = _ws(tmp_path / "sync")
    invoices_s, fids = _seed_many(ws_s, 3)
    docsets_s = DocSetStore(ws_s).list_all()
    capture_kwargs(
        lambda _k: _tool_call_response("assign_to_existing_docset", {"docset_id": invoices_s})
    )
    sync = [
        auto_classify(ws_s, res, config=cfg, docsets=docsets_s, allow_new=False)
        for res in _added(ws_s, fids)
    ]

    ws_b = _ws(tmp_path / "batch")
    invoices_b, fids_b = _seed_many(ws_b, 3)
    assert fids_b == fids
    backend = FakeBackend(
        lambda _r: _tool_call_response("assign_to_existing_docset", {"docset_id": invoices_b}),
        provider=provider_of(DEFAULT_TEST_MODEL),
    )
    register_backend(provider_of(DEFAULT_TEST_MODEL), lambda _cfg: backend)
    docsets_b = DocSetStore(ws_b).list_all()
    state = prepare_bulk_classify(ws_b, config=cfg, docsets=docsets_b, poll_interval_s=0)
    assert state.classifier is not None and state.grounded is None  # no schema anywhere
    entries: list[dict[str, Any]] = [{} for _ in fids]
    state.pending.extend(zip(entries, _added(ws_b, fids), strict=True))
    stats = classify_bulk_batch(ws_b, state, config=cfg, docsets=docsets_b)

    assert [len(wave) for wave in backend.submitted] == [3]  # one wave
    assert list(stats) == ["classification"]
    batch = [e["classification"] for e in entries]
    assert _neutral(batch, invoices_b) == _neutral(sync, invoices_s)
    assert DocSetStore(ws_b).list_files(invoices_b) == DocSetStore(ws_s).list_files(invoices_s)


def test_an_existing_file_is_reported_not_classified(tmp_path: Path) -> None:
    ws = _ws(tmp_path / "ws")
    _invoices, fids = _seed_many(ws, 1)
    (res,) = _added(ws, fids)
    res.created = False
    assert auto_classify(ws, res, config=ClassificationConfig(model=DEFAULT_TEST_MODEL)) == {
        "performed": False,
        "reason": "file already existed; classification skipped",
    }


def test_prepare_rejects_a_model_with_no_batch_backend(tmp_path: Path) -> None:
    ws = _ws(tmp_path / "ws")
    _seed_many(ws, 1)
    with pytest.raises(BatchUnavailable):
        prepare_bulk_classify(
            ws,
            config=ClassificationConfig(model="ollama/llama3"),
            docsets=DocSetStore(ws).list_all(),
        )


def test_soft_errors_keep_the_sync_format() -> None:
    assert soft_error(ClassificationFailed("no answer")) == "CLASSIFICATION_FAILED: no answer"
    assert soft_error(RuntimeError("boom")) == "RuntimeError: boom"
    block = classification_block(ClassificationConfig(model=DEFAULT_TEST_MODEL))
    assert block["performed"] is True and block["error"] is None
