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

""":class:`dgml_core.generation.link_stage.LinkStage`: one link stage, two
drivers. The batch driver must leave exactly what the sync driver leaves —
the linked XML, the cached plans, the per-document outcomes — while making one
request per distinct cache entry, in one wave per step."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from dgml_core import LinkOutcome, LinkStage, StagedDocument, layout
from dgml_core.batch import FakeBackend, fake_model_response, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.generation import links as links_mod
from dgml_core.generation.pipeline import BatchOptions
from dgml_core.storage import Workspace

_MODEL = "anthropic/claude-haiku-4-5"
_DOCSET = "ds_links"

# element order under root: e0000=chunk, 1=Commencement, 2=Adjustment, 3=BaseRent
_XML = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">'
    "<dg:CommencementDate>November 1, 2024</dg:CommencementDate>"
    "<dg:AdjustmentDate>each anniversary of the Commencement Date</dg:AdjustmentDate>"
    "<dg:BaseRent>{rent}</dg:BaseRent>"
    "</dg:chunk>"
)
_LINK = {"subject": "e0002", "object": "e0001", "predicate": "relativeTo"}
#: The plan the model's answer above becomes (positions in document order).
_PLAN = [{"subject": 2, "objects": [1], "predicate": "relativeTo", "value": ""}]


def _key(name: str) -> str:
    stem = Path(name).stem
    return layout.dgml_xml_key(_DOCSET, f"f_{stem}", stem)


def _doc(rent: int) -> str:
    return _XML.format(rent=rent)


def _system(kwargs: dict[str, Any]) -> str:
    content = kwargs["messages"][0]["content"]
    if isinstance(content, str):
        return content
    return "".join(str(b.get("text", "")) for b in content)


def _answer(kwargs: dict[str, Any]) -> Any:
    usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    if _system(kwargs) == links_mod.VERIFY_SYSTEM_PROMPT:
        return fake_model_response(json.dumps({"verdicts": [{"i": 0, "keep": True}]}), usage=usage)
    return fake_model_response(json.dumps({"links": [_LINK]}), usage=usage)


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


def _stage(ws: Workspace, **kw: Any) -> LinkStage:
    return LinkStage(workspace=ws, docset_id=_DOCSET, model=_MODEL, api_key="sk-test", **kw)


def _blobs(ws: Workspace) -> dict[str, bytes]:
    return {k: ws.blobs.get_blob(k) for k in sorted(ws.blobs.list_blobs(""))}


_DOCS = {"a.pdf": _doc(100), "b.pdf": _doc(100), "c.pdf": _doc(200)}  # a and b: one plan


def test_batch_driver_leaves_what_the_sync_driver_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws_s = _ws(tmp_path / "sync")
    sync_calls: list[str] = []

    def plan(xml: str, config: Any, *, verify: bool = True) -> list[dict[str, Any]]:
        sync_calls.append(config.context["doc"])
        return [dict(p) for p in _PLAN]

    monkeypatch.setattr(links_mod, "plan_links", plan)
    stage_s = _stage(ws_s)
    sync: dict[str, LinkOutcome] = {}
    for name, xml in _DOCS.items():
        ws_s.blobs.put_blob(_key(name), xml.encode())
        sync[name] = stage_s.link_document(name, _key(name))
    assert sync_calls == ["a.pdf", "c.pdf"]  # b.pdf hit a.pdf's cached plan
    assert all(o == LinkOutcome(links=1) for o in sync.values())
    monkeypatch.undo()

    ws_b = _ws(tmp_path / "batch")
    backend = FakeBackend(lambda r: _answer(r.kwargs), provider="anthropic")
    register_backend("anthropic", lambda _cfg: backend)
    opts = BatchOptions(poll_interval_s=0)
    staged = {
        name: StagedDocument(_key(name), {_key(name): xml.encode()}) for name, xml in _DOCS.items()
    }
    batch = _stage(ws_b).link_documents(staged, batch=opts, max_workers=2)

    assert batch == sync
    # One propose wave and one verify wave, each one request per distinct plan.
    assert [len(wave) for wave in backend.submitted] == [2, 2]
    assert opts.stats["links"]["waves"] == 2
    assert all(not item.writes for item in staged.values())  # held trees written
    assert _blobs(ws_b) == _blobs(ws_s)


def test_every_plan_cached_makes_no_request(tmp_path: Path) -> None:
    ws = _ws(tmp_path / "ws")
    stage = _stage(ws)
    source = _doc(1)
    ws.blobs.put_blob(f"{stage.cache_key(source)}.json", json.dumps(_PLAN).encode())
    opts = BatchOptions(poll_interval_s=0)
    out = stage.link_documents(
        {"a.pdf": StagedDocument(_key("a.pdf"), {_key("a.pdf"): source.encode()})},
        batch=opts,
        max_workers=1,
    )
    assert out == {"a.pdf": LinkOutcome(links=1)}
    assert opts.stats["links"] == {"skipped": "every link plan was cached"}
    assert b"dg:itemprop" in ws.blobs.get_blob(_key("a.pdf"))


def test_a_failed_stage_soft_fails_every_document_and_keeps_its_tree(tmp_path: Path) -> None:
    """No batch backend at all: every document gets a link error, and the
    grounded trees held back are still written."""
    ws = _ws(tmp_path / "ws")
    batch_registry._REGISTRY.clear()
    out = _stage(ws).link_documents(
        {
            name: StagedDocument(_key(name), {_key(name): xml.encode()})
            for name, xml in _DOCS.items()
        },
        batch=BatchOptions(poll_interval_s=0),
        max_workers=1,
    )
    assert set(out) == set(_DOCS)
    assert all(o.links == 0 and o.error for o in out.values())
    for name, xml in _DOCS.items():
        assert ws.blobs.get_blob(_key(name)) == xml.encode()


def test_sync_failure_and_disabled_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _ws(tmp_path / "ws")
    ws.blobs.put_blob(_key("a.pdf"), _doc(1).encode())

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("rate limited")

    monkeypatch.setattr(links_mod, "plan_links", boom)
    outcome = _stage(ws).link_document("a.pdf", _key("a.pdf"))
    assert outcome == LinkOutcome(links=0, error="RuntimeError: rate limited")
    assert ws.blobs.get_blob(_key("a.pdf")) == _doc(1).encode()  # DGML kept

    # Disabled: nothing read, nothing written, nothing asked.
    assert _stage(ws, enabled=False).link_document("a.pdf", "missing") == LinkOutcome()
    staged = {"a.pdf": StagedDocument(_key("b.pdf"), {_key("b.pdf"): b"<x/>"})}
    out = _stage(ws, enabled=False).link_documents(
        staged, batch=BatchOptions(poll_interval_s=0), max_workers=1
    )
    assert out == {"a.pdf": LinkOutcome()} and ws.blobs.get_blob(_key("b.pdf")) == b"<x/>"


def test_cache_key_tracks_model_and_review_pass(tmp_path: Path) -> None:
    ws = _ws(tmp_path / "ws")
    source = _doc(1)
    base = _stage(ws).cache_key(source)
    assert base.startswith(f"docsets/{_DOCSET}/")
    assert _stage(ws, verify=False).cache_key(source) != base
    other_model = LinkStage(workspace=ws, docset_id=_DOCSET, model="openai/gpt-5")
    assert other_model.cache_key(source) != base
    # Attributes are not part of the key: grounding a document still hits.
    grounded = source.replace("<dg:BaseRent>", '<dg:BaseRent dg:origin="1 2 3 4">')
    assert _stage(ws).cache_key(grounded) == base


def test_no_cache_batch_plans_every_document_like_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--no-semlink-cache``: the sync pass plans every document afresh, so
    the batch pass must too — no document rides another's request as
    ``(cached)``, even when their plans share a cache key."""
    ws_s = _ws(tmp_path / "sync")
    sync_calls: list[str] = []

    def plan(xml: str, config: Any, *, verify: bool = True) -> list[dict[str, Any]]:
        sync_calls.append(config.context["doc"])
        return [dict(p) for p in _PLAN]

    monkeypatch.setattr(links_mod, "plan_links", plan)
    sync_log: list[str] = []
    stage_s = _stage(ws_s, use_cache=False, log=sync_log.append)
    for name, xml in _DOCS.items():
        ws_s.blobs.put_blob(_key(name), xml.encode())
        stage_s.link_document(name, _key(name))
    assert sync_calls == list(_DOCS)  # b.pdf planned afresh, not a cache hit
    monkeypatch.undo()

    ws_b = _ws(tmp_path / "batch")
    backend = FakeBackend(lambda r: _answer(r.kwargs), provider="anthropic")
    register_backend("anthropic", lambda _cfg: backend)
    staged = {
        name: StagedDocument(_key(name), {_key(name): xml.encode()}) for name, xml in _DOCS.items()
    }
    batch_log: list[str] = []
    _stage(ws_b, use_cache=False, log=batch_log.append).link_documents(
        staged, batch=BatchOptions(poll_interval_s=0), max_workers=1
    )
    assert [len(wave) for wave in backend.submitted] == [3, 3]  # every document its own unit
    semlinks = sorted(line for line in batch_log if line.startswith("[semlinks]"))
    assert semlinks == sorted(sync_log)
    assert not any("(cached)" in line for line in semlinks)
    assert _blobs(ws_b) == _blobs(ws_s)
