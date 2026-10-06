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

"""``classify_files_batch``: assign-only classification, one batch wave."""

from __future__ import annotations

from typing import Any

import pytest
from dgml_core.batch import BatchExecutor, BatchItemError, BatchRequest, FakeBackend
from dgml_core.classification import (
    ClassificationConfig,
    ClassificationDecision,
    classify_file,
    classify_files_batch,
)
from dgml_core.docsets import DocSetStore
from dgml_core.errors import BatchExecutionFailed, ClassificationFailed, NoExistingDocSets
from dgml_core.storage import Workspace
from dgml_core.usage import TIER_BATCH, read_events

from .test_classification import (
    DEFAULT_TEST_MODEL,
    _seed_file,
    _seed_for_classify,
    _seed_page_image,
    _tool_call_response,
)


def _cfg() -> ClassificationConfig:
    return ClassificationConfig(model=DEFAULT_TEST_MODEL)


def _seed_many(workspace: Workspace, n: int) -> tuple[str, list[str]]:
    invoices_id, first = _seed_for_classify(workspace)
    fids = [first]
    for i in range(1, n):
        fid = f"newfid{i}"
        _seed_file(workspace, fid, filename=f"incoming{i}.pdf")
        _seed_page_image(workspace, fid, 1, b"\x89PNG\r\n\x1a\n" + f"fake-{i}".encode())
        fids.append(fid)
    return invoices_id, fids


def _assign(docset_id: str) -> Any:
    return _tool_call_response("assign_to_existing_docset", {"docset_id": docset_id})


def _executor(script: Any) -> tuple[BatchExecutor, FakeBackend]:
    backend = FakeBackend(script)
    return BatchExecutor(backend, min_wave_size=1, sleep=lambda _s: None), backend


def test_batch_decisions_match_sync_in_one_wave(workspace: Workspace, capture_kwargs: Any) -> None:
    invoices_id, fids = _seed_many(workspace, 3)
    docsets = DocSetStore(workspace).list_all()

    captured = capture_kwargs(lambda _k: _assign(invoices_id))
    sync = {
        fid: classify_file(workspace, fid, config=_cfg(), docsets=docsets, allow_new=False)
        for fid in fids
    }
    sync_requests = [dict(k) for k in captured.kwargs]

    executor, backend = _executor(lambda _r: _assign(invoices_id))
    batch = classify_files_batch(workspace, fids, config=_cfg(), docsets=docsets, executor=executor)
    assert batch == sync
    assert [len(b) for b in backend.submitted] == [3]
    # The requests the batch sends are the sync requests, byte for byte.
    assert [r.kwargs for r in backend.submitted[0]] == sync_requests


def test_usage_rows_are_batch_tier_with_context(workspace: Workspace) -> None:
    invoices_id, fids = _seed_many(workspace, 2)
    docsets = DocSetStore(workspace).list_all()
    executor, _ = _executor(lambda _r: _assign(invoices_id))
    classify_files_batch(
        workspace, fids, config=_cfg(), docsets=docsets, executor=executor, debug=True
    )
    rows = read_events(workspace)
    assert [r["tier"] for r in rows] == [TIER_BATCH, TIER_BATCH]
    assert [r["context"]["file_ids"] for r in rows] == [[f] for f in fids]


def test_failures_are_isolated_and_match_sync_messages(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    invoices_id, fids = _seed_many(workspace, 3)
    _seed_file(workspace, "nopages", filename="blank.pdf")  # no page images
    docsets = DocSetStore(workspace).list_all()
    bad, raising = fids[1], fids[2]

    def sync_respond(kwargs: dict[str, Any]) -> Any:
        blob = str(kwargs["messages"])
        if "fake-1" in _decoded(blob):
            return _tool_call_response("assign_to_existing_docset", {"docset_id": "nope"})
        if "fake-2" in _decoded(blob):
            raise RuntimeError("provider down")
        return _assign(invoices_id)

    capture_kwargs(sync_respond)
    order = [fids[0], bad, raising, "nopages"]
    sync: dict[str, Any] = {}
    for fid in order:
        try:
            sync[fid] = classify_file(
                workspace, fid, config=_cfg(), docsets=docsets, allow_new=False
            )
        except Exception as exc:
            sync[fid] = exc

    def script(req: BatchRequest) -> Any:
        if "fake-2" in _decoded(str(req.kwargs["messages"])):
            return BatchItemError(req.custom_id, "invalid", "rejected")  # → sync fallback raises
        return sync_respond(req.kwargs)

    executor, _ = _executor(script)
    batch = classify_files_batch(
        workspace, order, config=_cfg(), docsets=docsets, executor=executor
    )
    assert batch[fids[0]] == sync[fids[0]]
    for fid in (bad, raising, "nopages"):
        assert type(batch[fid]) is type(sync[fid]) is ClassificationFailed
        assert str(batch[fid]) == str(sync[fid])
    assert str(batch[raising]) == "LLM call failed: RuntimeError: provider down"


def _decoded(blob: str) -> str:
    """The seeded PNG bytes travel base64-encoded in a data URL."""
    import base64
    import re

    out = []
    for chunk in re.findall(r"base64,([A-Za-z0-9+/=]+)", blob):
        out.append(base64.b64decode(chunk).decode("latin-1"))
    return " ".join(out)


def test_single_docset_needs_no_request(workspace: Workspace) -> None:
    only = DocSetStore(workspace).create(name="Invoices", description="d", key_questions=["q?"])
    _seed_file(workspace, "a")
    executor, backend = _executor(lambda _r: pytest.fail("no request expected"))
    batch = classify_files_batch(workspace, ["a"], config=_cfg(), docsets=[only], executor=executor)
    assert batch == {"a": ClassificationDecision(decision="existing", existing_docset_id=only.id)}
    assert backend.submitted == []


def test_no_docsets_raises(workspace: Workspace) -> None:
    executor, _ = _executor(lambda _r: None)
    with pytest.raises(NoExistingDocSets):
        classify_files_batch(workspace, ["a"], config=_cfg(), docsets=[], executor=executor)


def test_stage_failure_is_every_files_entry(workspace: Workspace) -> None:
    invoices_id, fids = _seed_many(workspace, 2)
    docsets = DocSetStore(workspace).list_all()
    backend = FakeBackend(lambda _r: _assign(invoices_id), fail_submit=RuntimeError("refused"))
    executor = BatchExecutor(backend, min_wave_size=1, sleep=lambda _s: None)
    batch = classify_files_batch(workspace, fids, config=_cfg(), docsets=docsets, executor=executor)
    assert all(isinstance(batch[f], BatchExecutionFailed) for f in fids)
