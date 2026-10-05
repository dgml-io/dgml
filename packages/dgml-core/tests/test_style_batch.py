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

"""The image-based ``dg:style`` pass (OCR files) under batch mode.

Grounding an OCR file in a workspace whose ``style`` section is enabled sends
one vision request per page. Under ``docset generate --batch`` grounding
prepares those requests instead (``ground_dgml_xml(defer_style=True)``) and
every document's pages go out as ONE batch stage
(:func:`dgml_core.style_llm.style_documents_batch`). This module proves the
batch path sends exactly the synchronous requests and produces exactly the
synchronous grounded XML, stats and usage rows (apart from ``tier``).

Both paths run here over the same canned replies and are compared directly.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from dgml_core import layout, llm
from dgml_core.models import FileRecord
from dgml_core.storage import Workspace
from dgml_core.usage import read_events

MODEL = "anthropic/claude-haiku-4-5"

#: file id → its pages' words (one word list per page) and its DGML.
DOCS: dict[str, tuple[list[list[str]], str]] = {
    "fsty00000001": (
        [["TITLE"], ["Body", "text"]],
        "<Title>TITLE</Title><Para>Body text</Para>",
    ),
    "fsty00000002": (
        [["Heading"], ["Unreadable"], ["Footer"]],
        "<Heading>Heading</Heading><Note>Unreadable</Note><Foot>Footer</Foot>",
    ),
}


def _reply(text: str) -> Any:
    from litellm import ModelResponse

    response = ModelResponse(
        choices=[
            {"message": {"role": "assistant", "content": text}, "finish_reason": "stop", "index": 0}
        ],
        usage={"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340},
        model=MODEL,
    )
    response._hidden_params["response_cost"] = 0.02
    return response


def answer(kwargs: dict[str, Any]) -> Any:
    """Bold every page's first snippet; the page showing "Unreadable" fails."""
    prompt = kwargs["messages"][1]["content"][0]["text"]
    if "Unreadable" in prompt:
        raise RuntimeError("provider down")
    style = "font-weight: bold" if "TITLE" in prompt or "Heading" in prompt else "color: red"
    return _reply(json.dumps({"styles": [{"index": 0, "style": style}]}))


def _seed(tmp_path: Path) -> tuple[Workspace, dict[str, Path]]:
    ws = Workspace(root=tmp_path / "ws")
    ws.root.mkdir(parents=True, exist_ok=True)
    ws.config_path.write_text('[style]\nenabled = true\nmodel = "' + MODEL + '"\n')
    sources: dict[str, Path] = {}
    for fid, (pages, body) in DOCS.items():
        ws.docs.put_doc(
            "files",
            fid,
            FileRecord(
                id=fid,
                original_path=f"/{fid}.pdf",
                original_filename=f"{fid}.pdf",
                sha256="0" * 64,
                added_at="2026-01-01T00:00:00Z",
                page_count=len(pages),
                text_mode="ocr",
            ).to_json(),
        )
        for n, words in enumerate(pages, start=1):
            boxes = [
                {"t": w, "l": [100 + 80 * i, 100, 170 + 80 * i, 120]} for i, w in enumerate(words)
            ]
            page = {"file_id": fid, "page": n, "width": 1000, "height": 1000, "words": boxes}
            ws.blobs.put_blob(layout.file_page_text_key(fid, n), json.dumps(page).encode())
            ws.blobs.put_blob(layout.file_page_image_key(fid, n), f"PNG-{fid}-{n}".encode())
        src = tmp_path / f"{fid}.dgml.xml"
        src.write_text(
            f'<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">{body}</dg:chunk>', encoding="utf-8"
        )
        sources[fid] = src
    return ws, sources


def _key(kwargs: dict[str, Any]) -> str:
    return json.dumps(kwargs, sort_keys=True, default=repr)


def _rows(ws: Workspace, tier: str | None) -> list[dict[str, Any]]:
    out = []
    for row in read_events(ws):
        assert row.pop("tier", None) == tier, row
        out.append({k: v for k, v in row.items() if k not in {"at", "duration_s"}})
    return sorted(out, key=lambda r: json.dumps(r, sort_keys=True))


def _stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stats.items() if k not in ("completed_at", "source", "output")}


def _sync_style(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Synchronous grounding with the image-style pass: what it sent and left."""
    from dgml_core.xml_grounding import ground_dgml_xml

    ws, sources = _seed(tmp_path)
    sent: list[dict[str, Any]] = []

    def respond(kwargs: dict[str, Any], **_kw: Any) -> Any:
        sent.append(copy.deepcopy(kwargs))
        return answer(kwargs)

    monkeypatch.setattr(llm, "_completion_with_retry", respond)
    xml: dict[str, str] = {}
    stats: dict[str, Any] = {}
    for fid, src in sources.items():
        res = ground_dgml_xml(
            ws, fid, src, output_path=src, force=True, write_stats=False, debug=True
        )
        xml[fid] = src.read_text(encoding="utf-8")
        stats[fid] = _stats(res.stats)
    return {
        # Pages are requested concurrently: compare them as a set.
        "kwargs": json.loads(json.dumps(sorted(sent, key=_key), default=repr)),
        "xml": xml,
        "stats": json.loads(json.dumps(stats)),
        "usage_rows": _rows(ws, "standard"),
    }


@pytest.fixture
def anthropic_backend() -> Iterator[list[Any]]:
    from dgml_core.batch import BatchItemError, FakeBackend, register_backend
    from dgml_core.batch import registry as batch_registry

    saved = dict(batch_registry._REGISTRY)
    built: list[Any] = []

    def script(request: Any) -> Any:
        try:
            return answer(request.kwargs)
        except RuntimeError as exc:
            return BatchItemError(request.custom_id, "invalid", str(exc))

    def factory(_cfg: Any) -> Any:
        backend = FakeBackend(script, provider="anthropic")
        built.append(backend)
        return backend

    register_backend("anthropic", factory)
    try:
        yield built
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


def test_batch_style_matches_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, anthropic_backend: list[Any]
) -> None:
    """Deferred grounding + one batch stage over both documents' pages: the
    same requests (as submitted), the same grounded XML and stats, the same
    usage rows apart from ``tier``."""
    from dgml_core.style_config import load_style_config
    from dgml_core.style_llm import style_batch_executor, style_documents_batch
    from dgml_core.xml_grounding import ground_dgml_xml

    sync = _sync_style(tmp_path / "sync", monkeypatch)
    assert "font-weight: bold" in sync["xml"]["fsty00000001"]
    ws, sources = _seed(tmp_path / "batch")
    # The failed page's synchronous fallback reaches the sync seam.
    monkeypatch.setattr(llm, "_completion_with_retry", lambda kwargs, **_kw: answer(kwargs))
    pending: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    for fid, src in sources.items():
        res = ground_dgml_xml(
            ws,
            fid,
            src,
            output_path=src,
            force=True,
            write_stats=False,
            debug=True,
            defer_style=True,
        )
        assert res.pending_style is not None
        pending[fid] = res.pending_style
        stats[fid] = _stats(res.stats)
    assert read_events(ws) == []  # nothing sent while grounding
    config = load_style_config(ws)
    assert config is not None
    executor = style_batch_executor(config, poll_interval_s=0)
    xml = {
        fid: data.decode("utf-8") for fid, data in style_documents_batch(pending, executor).items()
    }

    (backend,) = anthropic_backend
    assert [len(wave) for wave in backend.submitted] == [5]  # every page, one wave
    submitted = [req.kwargs for wave in backend.submitted for req in wave]
    assert json.loads(json.dumps(sorted(submitted, key=_key), default=repr)) == sync["kwargs"]
    assert xml == sync["xml"]
    assert json.loads(json.dumps(stats)) == sync["stats"]
    assert _rows(ws, "batch") == sync["usage_rows"]
    # The pages showing "Unreadable" (the root lists every snippet on page 1).
    assert executor.stats.to_json()["sync_fallbacks"] == 2


def test_defer_style_is_inert_without_a_style_pass(tmp_path: Path) -> None:
    """A digital file (or no ``style`` section) has nothing to defer: the
    grounding is the synchronous one and ``pending_style`` stays unset."""
    from dgml_core.xml_grounding import ground_dgml_xml

    ws, sources = _seed(tmp_path)
    ws.config_path.write_text("")
    fid, src = next(iter(sources.items()))
    res = ground_dgml_xml(
        ws, fid, src, output_path=src, force=True, write_stats=False, defer_style=True
    )
    assert res.pending_style is None
