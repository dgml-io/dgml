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

"""``extract_values_batch``: many files, each LLM phase one batch wave.

Every test answers requests with one content-keyed responder, used both as the
synchronous seam (``capture_kwargs``) and as the :class:`FakeBackend` script, so
the sync and batch runs see identical replies and their outputs can be compared
field for field. Both runs share a workspace, so DocSet ids — which appear in the
usage rows and XML keys — are the same too.
"""

from __future__ import annotations

import base64
import copy
import json
import re
from collections.abc import Callable
from typing import Any

import pytest
from dgml_core import layout
from dgml_core.batch import BatchExecutor, BatchItemError, BatchRequest, FakeBackend
from dgml_core.errors import (
    BatchExecutionFailed,
    SchemaInvalid,
    SchemaNotFound,
    ValuesExtractionFailed,
)
from dgml_core.grounded import (
    ExtractionResult,
    GroundedConfig,
    extract_values,
    extract_values_batch,
)
from dgml_core.storage import Workspace
from dgml_core.usage import TIER_BATCH, TIER_STANDARD, read_events

from .test_grounded import (
    _TITLE_RNC,
    DEFAULT_SCHEMA_MODEL,
    DEFAULT_VALUES_MODEL,
    _seed_docset_with_schema,
    _seed_file,
    _seed_page_image,
    _seed_page_text,
    _tool_call_response,
    _truncated_response,
)

Responder = Callable[[dict[str, Any]], Any]

MATCHED = "fmatched0001"  # phase 2 matches everything → no phase 3
ONE_PAGE = "fonepage0002"  # one unmatched page → one phase-3 call
TWO_PAGES = "ftwopages003"  # unmatched on two pages → two phase-3 calls


def _config() -> GroundedConfig:
    return GroundedConfig(schema_model=DEFAULT_SCHEMA_MODEL, values_model=DEFAULT_VALUES_MODEL)


def _pdf(fid: str) -> bytes:
    return f"%PDF-1.4 {fid}\n".encode()


def _file_of(kwargs: dict[str, Any]) -> str | None:
    """Which seeded file a phase-1 request carries (its PDF names it)."""
    blob = json.dumps(kwargs["messages"])
    for fid in (MATCHED, ONE_PAGE, TWO_PAGES, "fbroken00004", "fchunked0005", "fwide0000006"):
        if base64.b64encode(_pdf(fid)).decode() in blob:
            return fid
    return None


def _tool_name(kwargs: dict[str, Any]) -> str:
    return str(kwargs["tools"][0]["function"]["name"])


def _locate_all(kwargs: dict[str, Any]) -> Any:
    """Answer a phase-3 page with one box for every id its tool enumerates."""
    params = json.dumps(kwargs["tools"][0]["function"]["parameters"])
    ids = sorted(set(re.findall(r'"enum": \[([^\]]*)\]', params)[0].replace('"', "").split(", ")))
    page = int(re.search(r"page (\d+)", json.dumps(kwargs["messages"])).group(1))  # type: ignore[union-attr]
    return _tool_call_response(
        "submit_locations",
        {"locations": [{"id": i, "bounding_boxes": [[10 * page, 20, 30, 40]]} for i in ids]},
        cost_usd=0.002 * page,
        prompt_tokens=40 * page,
        completion_tokens=5,
    )


_PHASE1_VALUES = {
    MATCHED: {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}},
    ONE_PAGE: {"title": {"text": "Goodnight", "locations": [{"page_number": 1}]}},
    TWO_PAGES: {
        "title": {"text": "Goodnight", "locations": [{"page_number": 1}, {"page_number": 2}]}
    },
}


def _responder(overrides: dict[str, Responder] | None = None) -> Responder:
    """The canned model: phase 1 per file (by PDF), phase 3 per page (by ids)."""

    def respond(kwargs: dict[str, Any]) -> Any:
        if _tool_name(kwargs) == "submit_locations":
            return _locate_all(kwargs)
        fid = _file_of(kwargs)
        if overrides and fid in overrides:
            return overrides[fid](kwargs)
        assert fid in _PHASE1_VALUES, f"unexpected phase-1 request for {fid}"
        return _tool_call_response(
            "submit_values",
            {"values": _PHASE1_VALUES[fid]},
            cost_usd=0.01,
            prompt_tokens=100,
            completion_tokens=50,
            cache_read_tokens=7,
        )

    return respond


def _seed_three(workspace: Workspace) -> str:
    from dgml_core.docsets import DocSetStore

    _seed_file(workspace, MATCHED, pdf_bytes=_pdf(MATCHED), filename="matched.pdf")
    _seed_page_text(workspace, MATCHED, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, MATCHED)
    store = DocSetStore(workspace)
    for fid, pages, name in ((ONE_PAGE, 1, "one.pdf"), (TWO_PAGES, 2, "two.pdf")):
        _seed_file(workspace, fid, pdf_bytes=_pdf(fid), page_count=pages, filename=name)
        store.add_file(ds_id, fid)
        for page in range(1, pages + 1):
            _seed_page_text(workspace, fid, page=page)
            _seed_page_image(workspace, fid, page)
    return ds_id


def _executor(respond: Responder, **fake: Any) -> tuple[BatchExecutor, FakeBackend]:
    backend = FakeBackend(lambda req: respond(req.kwargs), **fake)
    # min_wave_size=1: every wave goes through the batch backend, so the test
    # exercises the batch path even for a one-request wave.
    return BatchExecutor(backend, min_wave_size=1, sleep=lambda _s: None), backend


def _stats(workspace: Workspace, ds_id: str, fid: str) -> dict[str, Any]:
    stats = workspace.docs.get_doc(layout.Collection.EXTRACTION_STATS, layout.pair_id(ds_id, fid))
    assert stats is not None, f"no extraction_stats for {fid}"
    return stats


def _snapshot(workspace: Workspace, ds_id: str, fids: list[str]) -> dict[str, Any]:
    """Everything a run leaves behind, minus wall-clock fields."""
    out: dict[str, Any] = {}
    names = {MATCHED: "matched", ONE_PAGE: "one", TWO_PAGES: "two"}
    for fid in fids:
        key = layout.dgml_xml_key(ds_id, fid, names[fid])
        out[f"xml:{fid}"] = workspace.blobs.get_blob(key).decode()
        stats = _stats(workspace, ds_id, fid)
        stats.pop("completed_at")
        for phase in stats["phases"].values():
            phase.pop("duration_s")
        out[f"stats:{fid}"] = stats
    return out


def _rows(workspace: Workspace) -> list[dict[str, Any]]:
    rows = []
    for row in read_events(workspace):
        rows.append({k: v for k, v in row.items() if k not in {"at", "duration_s"}})
    return rows


def _run_sync(
    workspace: Workspace, ds_id: str, fids: list[str], capture_kwargs: Any, respond: Responder
) -> dict[str, ExtractionResult | Exception]:
    capture_kwargs(respond)
    out: dict[str, ExtractionResult | Exception] = {}
    for fid in fids:
        try:
            out[fid] = extract_values(
                workspace, ds_id, fid, config=_config(), write_stats=True, debug=True
            )
        except Exception as exc:
            out[fid] = exc
    return out


# ---- parity ---------------------------------------------------------------------


def test_batch_matches_sync_values_xml_stats_and_rows(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    ds_id = _seed_three(workspace)
    fids = [MATCHED, ONE_PAGE, TWO_PAGES]
    respond = _responder()

    sync = _run_sync(workspace, ds_id, fids, capture_kwargs, respond)
    sync_snapshot = _snapshot(workspace, ds_id, fids)
    sync_rows = _rows(workspace)

    executor, backend = _executor(respond)
    batch = extract_values_batch(
        workspace, ds_id, fids, config=_config(), executor=executor, write_stats=True, debug=True
    )
    batch_snapshot = _snapshot(workspace, ds_id, fids)
    batch_rows = _rows(workspace)[len(sync_rows) :]

    assert list(batch) == fids
    for fid in fids:
        s, b = sync[fid], batch[fid]
        assert isinstance(s, ExtractionResult) and isinstance(b, ExtractionResult)
        assert b == s
    assert batch_snapshot == sync_snapshot
    # One row per file, identical but for the tier.
    assert len(batch_rows) == len(sync_rows) == 3
    for s_row, b_row in zip(sync_rows, batch_rows, strict=True):
        assert s_row["tier"] == TIER_STANDARD and b_row["tier"] == TIER_BATCH
        assert {**b_row, "tier": TIER_STANDARD} == s_row
    # Phase 3 really ran (one page + two pages), and the phase-3 cost is in the rows.
    assert sync_snapshot[f"stats:{TWO_PAGES}"]["phases"]["phase3"]["page_calls"] == 2
    assert sync_snapshot[f"stats:{ONE_PAGE}"]["phases"]["phase3"]["page_calls"] == 1
    # Two waves: phase 1 (3 files), then phase 3 (1 + 2 pages) — every request batched.
    assert [len(batch_) for batch_ in backend.submitted] == [3, 3]
    assert executor.stats.sync_fallbacks == 0


def test_batch_locates_a_page_without_words_on_the_grid_like_sync(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    """Upstream #198 through the batch path: a page with no OCR words is asked
    for 0-1000 grid boxes at reasoning effort "medium", and the boxes are
    scaled to page pixels — the same requests, values, XML, stats and rows as
    the synchronous run (the phase-3 generator is shared, so both get it)."""
    from dgml_core.docsets import DocSetStore

    from .test_grounded import _png_header

    _seed_file(workspace, MATCHED, pdf_bytes=_pdf(MATCHED), filename="matched.pdf")
    _seed_page_text(workspace, MATCHED, page=1)
    ds_id, _ = _seed_docset_with_schema(workspace, MATCHED)
    _seed_file(workspace, ONE_PAGE, pdf_bytes=_pdf(ONE_PAGE), page_count=1, filename="one.pdf")
    DocSetStore(workspace).add_file(ds_id, ONE_PAGE)
    _seed_page_text(workspace, ONE_PAGE, page=1, width=2550, height=3300, words=[])
    workspace.blobs.put_blob(layout.file_page_image_key(ONE_PAGE, 1), _png_header(2550, 3300))
    fids = [ONE_PAGE]
    respond = _responder()

    sync_seen: list[dict[str, Any]] = []

    def sync_respond(kwargs: dict[str, Any]) -> Any:
        sync_seen.append(kwargs)
        return respond(kwargs)

    sync = _run_sync(workspace, ds_id, fids, capture_kwargs, sync_respond)
    sync_snapshot = _snapshot(workspace, ds_id, fids)
    sync_rows = _rows(workspace)

    executor, backend = _executor(respond)
    batch = extract_values_batch(
        workspace, ds_id, fids, config=_config(), executor=executor, write_stats=True, debug=True
    )
    assert batch[ONE_PAGE] == sync[ONE_PAGE]
    assert _snapshot(workspace, ds_id, fids) == sync_snapshot
    batch_rows = _rows(workspace)[len(sync_rows) :]
    assert [{**r, "tier": TIER_STANDARD} for r in batch_rows] == sync_rows

    batched = [req.kwargs for wave in backend.submitted for req in wave]
    phase3 = [kw for kw in batched if _tool_name(kw) == "submit_locations"]
    assert phase3 == [kw for kw in sync_seen if _tool_name(kw) == "submit_locations"]
    (request,) = phase3
    assert "0-1000 grid" in request["messages"][0]["content"]
    assert request["reasoning_effort"] == "medium"
    result = batch[ONE_PAGE]
    assert isinstance(result, ExtractionResult)
    # [10, 20, 30, 40] on the grid, scaled to the 2550 x 3300 page.
    assert result.values["title"]["locations"] == [
        {"page_number": 1, "bounding_box": [26, 66, 76, 132]}
    ]
    phase3_stats = sync_snapshot[f"stats:{ONE_PAGE}"]["phases"]["phase3"]
    assert (phase3_stats["grid_pages"], phase3_stats["boxes_dropped"]) == (1, 0)


def test_phase3_fans_out_across_files_and_pages_in_one_wave(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    ds_id = _seed_three(workspace)
    capture_kwargs(_responder())  # nothing should reach the sync seam
    executor, backend = _executor(_responder())
    extract_values_batch(
        workspace, ds_id, [ONE_PAGE, TWO_PAGES], config=_config(), executor=executor
    )
    phase3 = backend.submitted[1]
    # Unit names are "<docset>/<file>/p<page>"; ids carry them sanitized.
    stems = sorted(r.custom_id.rsplit("-", 1)[0].split("_", 1)[1] for r in phase3)
    assert stems == sorted(
        [f"{ds_id}_{ONE_PAGE}_p1", f"{ds_id}_{TWO_PAGES}_p1", f"{ds_id}_{TWO_PAGES}_p2"]
    )
    assert executor.stats.waves == 2


@pytest.mark.parametrize("effort", ["low", "none", None])
def test_configured_values_effort_batch_matches_sync(
    workspace: Workspace, capture_kwargs: Any, effort: str | None
) -> None:
    """``grounded.values_reasoning_effort`` reaches the batch requests exactly as
    it reaches the synchronous ones: phase 1 carries the configured effort
    (absent for ``None``, the config's ``"default"``), phase 3 keeps its own
    fixed effort, and every other request field matches too."""
    from dataclasses import replace

    ds_id = _seed_three(workspace)
    fids = [MATCHED, ONE_PAGE, TWO_PAGES]
    config = replace(_config(), values_reasoning_effort=effort)
    respond = _responder()

    captured = capture_kwargs(respond)
    for fid in fids:
        extract_values(workspace, ds_id, fid, config=config)
    sync_requests = list(captured.kwargs)

    # Copied as the backend answers them: a multi-turn generator appends to its
    # message list after the request was submitted.
    batch_requests: list[dict[str, Any]] = []

    def respond_batch(kwargs: dict[str, Any]) -> Any:
        batch_requests.append(copy.deepcopy(kwargs))
        return respond(kwargs)

    executor, backend = _executor(respond_batch)
    extract_values_batch(workspace, ds_id, fids, config=config, executor=executor)
    assert executor.stats.sync_fallbacks == 0
    assert sum(len(wave) for wave in backend.submitted) == len(batch_requests) == 6

    def _key(kwargs: dict[str, Any]) -> str:
        return json.dumps(kwargs, sort_keys=True, default=str)

    assert sorted(map(_key, batch_requests)) == sorted(map(_key, sync_requests))
    phase1 = [k for k in sync_requests if _tool_name(k) != "submit_locations"]
    phase3 = [k for k in sync_requests if _tool_name(k) == "submit_locations"]
    assert len(phase1) == 3 and len(phase3) == 3
    for kwargs in phase1:
        if effort is None:
            assert "reasoning_effort" not in kwargs
        else:
            assert kwargs["reasoning_effort"] == effort
    assert all(k["reasoning_effort"] == "high" for k in phase3)


# ---- isolation and failures ------------------------------------------------


def test_one_file_failing_phase1_is_isolated_with_the_sync_message(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    ds_id = _seed_three(workspace)
    broken = "fbroken00004"
    from dgml_core.docsets import DocSetStore

    _seed_file(workspace, broken, pdf_bytes=_pdf(broken), filename="broken.pdf")
    _seed_page_text(workspace, broken, page=1)
    DocSetStore(workspace).add_file(ds_id, broken)

    def no_tool_call(_kwargs: dict[str, Any]) -> Any:
        return _tool_call_response("not_a_tool", {})

    respond = _responder({broken: no_tool_call})
    fids = [MATCHED, broken, ONE_PAGE]

    sync = _run_sync(workspace, ds_id, fids, capture_kwargs, respond)
    executor, _ = _executor(respond)
    batch = extract_values_batch(
        workspace, ds_id, fids, config=_config(), executor=executor, debug=True
    )

    sync_err, batch_err = sync[broken], batch[broken]
    assert isinstance(sync_err, ValuesExtractionFailed)
    assert type(batch_err) is type(sync_err)
    assert str(batch_err) == str(sync_err)
    assert isinstance(batch[MATCHED], ExtractionResult)
    assert isinstance(batch[ONE_PAGE], ExtractionResult)
    # The failed file still wrote its error row, like the sync path.
    rows = [r for r in _rows(workspace) if r["tier"] == TIER_BATCH]
    assert [r["outcome"] for r in rows if r["context"]["file_id"] == broken] == ["error"]


def test_executor_failure_on_a_request_gets_the_sync_message(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    """A request the batch cannot serve and whose sync fallback fails reaches the
    generator by throw(), and surfaces with the sync path's single prefix."""
    ds_id = _seed_three(workspace)

    def boom(kwargs: dict[str, Any]) -> Any:
        if _file_of(kwargs) == ONE_PAGE:
            raise RuntimeError("provider down")
        return _responder()(kwargs)

    sync = _run_sync(workspace, ds_id, [ONE_PAGE], capture_kwargs, boom)

    def script(req: BatchRequest) -> Any:
        if _file_of(req.kwargs) == ONE_PAGE:
            return BatchItemError(req.custom_id, "invalid", "rejected")
        return _responder()(req.kwargs)

    executor = BatchExecutor(FakeBackend(script), min_wave_size=1, sleep=lambda _s: None)
    batch = extract_values_batch(
        workspace, ds_id, [MATCHED, ONE_PAGE], config=_config(), executor=executor
    )
    assert str(batch[ONE_PAGE]) == str(sync[ONE_PAGE])
    assert str(batch[ONE_PAGE]) == "extraction call failed: RuntimeError: provider down"
    assert isinstance(batch[MATCHED], ExtractionResult)


def test_setup_failure_is_the_files_entry_and_writes_nothing(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    ds_id = _seed_three(workspace)
    capture_kwargs(_responder())
    executor, _ = _executor(_responder())
    missing = "fmissing0009"  # no file record, no PDF
    batch = extract_values_batch(
        workspace, ds_id, [missing, MATCHED], config=_config(), executor=executor, debug=True
    )
    assert isinstance(batch[missing], Exception)
    assert isinstance(batch[MATCHED], ExtractionResult)
    assert [r["context"]["file_id"] for r in _rows(workspace)] == [MATCHED]


def test_no_schema_is_a_setup_failure_for_every_file(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    from dgml_core.docsets import DocSetStore

    ds = DocSetStore(workspace).create(name="NoSchema")
    _seed_file(workspace, MATCHED, pdf_bytes=_pdf(MATCHED))
    executor, backend = _executor(_responder())
    batch = extract_values_batch(workspace, ds.id, [MATCHED], config=_config(), executor=executor)
    assert isinstance(batch[MATCHED], SchemaNotFound)
    assert backend.submitted == []


def test_unresolvable_invariant_is_a_setup_failure_like_sync(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    """A stored schema whose ``## Invariant:`` path can never resolve (written
    around ``set_schema``, which now refuses it) fails extraction before any
    model call — under batch exactly as in sync, since both load the schema
    through :class:`_FileExtraction` (upstream ac40172)."""
    _seed_file(workspace, MATCHED, pdf_bytes=_pdf(MATCHED))
    ds_id, _ = _seed_docset_with_schema(workspace, MATCHED)
    bad = _TITLE_RNC.replace("title =\n", "## Invariant: count(Missing)\ntitle =\n", 1)
    assert bad != _TITLE_RNC
    workspace.blobs.put_blob(layout.docset_extraction_schema_key(ds_id), bad.encode("utf-8"))
    sync = _run_sync(workspace, ds_id, [MATCHED], capture_kwargs, _responder())
    executor, backend = _executor(_responder())
    batch = extract_values_batch(workspace, ds_id, [MATCHED], config=_config(), executor=executor)
    for out in (sync[MATCHED], batch[MATCHED]):
        assert isinstance(out, SchemaInvalid)
        assert "names no collection" in str(out)
    assert str(batch[MATCHED]) == str(sync[MATCHED])
    assert backend.submitted == []
    assert _rows(workspace) == []


def test_stage_failure_is_recorded_against_every_file_in_flight(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    ds_id = _seed_three(workspace)
    capture_kwargs(_responder())
    executor, _ = _executor(_responder(), fail_submit=RuntimeError("provider refused"))
    batch = extract_values_batch(
        workspace, ds_id, [MATCHED, ONE_PAGE], config=_config(), executor=executor, debug=True
    )
    for fid in (MATCHED, ONE_PAGE):
        assert isinstance(batch[fid], BatchExecutionFailed)
    assert [r["outcome"] for r in _rows(workspace)] == ["error", "error"]


def test_duplicate_file_ids_are_rejected(workspace: Workspace) -> None:
    executor, _ = _executor(_responder())
    with pytest.raises(ValueError, match="duplicated"):
        extract_values_batch(
            workspace, "ds", [MATCHED, MATCHED], config=_config(), executor=executor
        )


# ---- multi-turn phase 1 ---------------------------------------------------------

_CHUNK_VALUES = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}


def test_truncated_phase1_retries_across_extra_waves(
    workspace: Workspace, capture_kwargs: Any
) -> None:
    """finish_reason='length' restarts phase 1 with the chunking directive: under
    batch that is just another wave for that file, and the result matches sync."""
    ds_id = _seed_three(workspace)
    chunked = "fchunked0005"
    from dgml_core.docsets import DocSetStore

    _seed_file(workspace, chunked, pdf_bytes=_pdf(chunked), filename="chunked.pdf")
    _seed_page_text(workspace, chunked, page=1)
    DocSetStore(workspace).add_file(ds_id, chunked)

    def turns(kwargs: dict[str, Any]) -> Any:
        # The first attempt truncates; the chunked retry (its user content
        # carries the extra directive block) submits in one call.
        if len(kwargs["messages"][1]["content"]) == 2:
            return _truncated_response()
        return _tool_call_response("submit_values", {"values": _CHUNK_VALUES}, call_id="r1")

    respond = _responder({chunked: turns})
    sync = _run_sync(workspace, ds_id, [chunked], capture_kwargs, respond)
    executor, backend = _executor(respond)
    batch = extract_values_batch(
        workspace, ds_id, [chunked, MATCHED], config=_config(), executor=executor
    )
    assert batch[chunked] == sync[chunked]
    # Wave 1: both files; wave 2: only the retry for the truncated file.
    assert [len(b) for b in backend.submitted] == [2, 1]


def test_permissive_schema_fallback_under_batch(workspace: Workspace, capture_kwargs: Any) -> None:
    """A 'too many states' refusal of the inlined schema retries phase 1 with the
    permissive parameter — inside the generator, so batch gets it for free."""
    ds_id = _seed_three(workspace)
    wide = "fwide0000006"
    from dgml_core.docsets import DocSetStore

    _seed_file(workspace, wide, pdf_bytes=_pdf(wide), filename="wide.pdf")
    _seed_page_text(workspace, wide, page=1)
    DocSetStore(workspace).add_file(ds_id, wide)

    def inlined(kwargs: dict[str, Any]) -> bool:
        params = kwargs["tools"][0]["function"]["parameters"]
        return "title" in json.dumps(params)

    def sync_respond(kwargs: dict[str, Any]) -> Any:
        if _file_of(kwargs) == wide:
            if inlined(kwargs):
                raise RuntimeError("BadRequestError: schema produces too many states for serving")
            return _tool_call_response("submit_values", {"values": _CHUNK_VALUES})
        return _responder()(kwargs)

    sync = _run_sync(workspace, ds_id, [wide], capture_kwargs, sync_respond)
    assert isinstance(sync[wide], ExtractionResult)
    sync_stats = _stats(workspace, ds_id, wide)

    def script(req: BatchRequest) -> Any:
        if _file_of(req.kwargs) == wide and inlined(req.kwargs):
            return BatchItemError(req.custom_id, "invalid", "too many states")
        return sync_respond(req.kwargs)

    executor = BatchExecutor(FakeBackend(script), min_wave_size=1, sleep=lambda _s: None)
    batch = extract_values_batch(
        workspace, ds_id, [wide, MATCHED], config=_config(), executor=executor, write_stats=True
    )
    assert batch[wide] == sync[wide]
    stats = _stats(workspace, ds_id, wide)
    assert stats["phase1_tool_schema"] == sync_stats["phase1_tool_schema"] == "permissive"


def test_import_discipline_sync_path_does_not_load_the_batch_package() -> None:
    """Upstream keeps litellm (which the batch package imports) off the
    deterministic path; grounded must only import dgml_core.batch lazily."""
    import subprocess
    import sys

    code = (
        "import sys, dgml_core.grounded, dgml_core.classification;"
        "print('dgml_core.batch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ---- review fixes: per-file isolation of every step, cross-DocSet waves ----------


def test_usage_write_failure_is_isolated_to_its_file(
    workspace: Workspace, capture_kwargs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure writing one file's usage row becomes that file's entry — as the
    sync path's ``finally`` would raise it — and the other files still succeed."""
    from dgml_core import grounded

    ds_id = _seed_three(workspace)
    real_record = grounded.record_usage  # type: ignore[attr-defined]

    def flaky_record(ws: Workspace, event: Any) -> None:
        if event.context["file_id"] == ONE_PAGE:
            raise OSError("disk full")
        real_record(ws, event)

    monkeypatch.setattr(grounded, "record_usage", flaky_record)

    capture_kwargs(_responder())
    with pytest.raises(OSError, match="disk full"):
        extract_values(workspace, ds_id, ONE_PAGE, config=_config(), debug=True)

    executor, _ = _executor(_responder())
    batch = extract_values_batch(
        workspace,
        ds_id,
        [MATCHED, ONE_PAGE, TWO_PAGES],
        config=_config(),
        executor=executor,
        debug=True,
    )
    assert isinstance(batch[ONE_PAGE], OSError) and str(batch[ONE_PAGE]) == "disk full"
    assert isinstance(batch[MATCHED], ExtractionResult)
    assert isinstance(batch[TWO_PAGES], ExtractionResult)


def test_pairs_across_docsets_share_two_waves(workspace: Workspace, capture_kwargs: Any) -> None:
    """Files in three DocSets: one phase-1 wave and one phase-3 wave in total,
    not two per DocSet."""
    from dgml_core.docsets import DocSetStore
    from dgml_core.grounded import extract_values_batch_pairs

    from .test_grounded import _TITLE_RNC

    store = DocSetStore(workspace)
    pairs = []
    for fid, pages, name in (
        (MATCHED, 1, "matched.pdf"),
        (ONE_PAGE, 1, "one.pdf"),
        (TWO_PAGES, 2, "two.pdf"),
    ):
        _seed_file(workspace, fid, pdf_bytes=_pdf(fid), page_count=pages, filename=name)
        for page in range(1, pages + 1):
            _seed_page_text(workspace, fid, page=page)
            _seed_page_image(workspace, fid, page)
        ds = store.create(name=f"DS {fid}")
        store.set_schema(ds.id, _TITLE_RNC)
        store.add_file(ds.id, fid)
        pairs.append((ds.id, fid))

    capture_kwargs(_responder())
    executor, backend = _executor(_responder())
    results = extract_values_batch_pairs(workspace, pairs, config=_config(), executor=executor)
    assert list(results) == pairs
    assert all(isinstance(r, ExtractionResult) for r in results.values())
    assert executor.stats.waves == 2
    # Wave 1: three files' phase 1; wave 2: the unmatched pages of two DocSets' files.
    assert [len(b) for b in backend.submitted] == [3, 3]


def test_envelope_repair_and_refusal_match_sync(workspace: Workspace, capture_kwargs: Any) -> None:
    """#172 under --batch: a ``submit_values`` envelope repeated one level down
    is unwrapped and counted, and a tree that fits no schema root is refused
    after phase 1's tool calls and layout are recorded, exactly as in sync."""
    ds_id = _seed_three(workspace)
    fids = [MATCHED, ONE_PAGE]
    layout_ = {"title": {"kind": "free_form"}}

    def nested(_kwargs: dict[str, Any]) -> Any:
        values = {"values": _PHASE1_VALUES[MATCHED], "layout": layout_}
        return _tool_call_response("submit_values", {"values": values}, cost_usd=0.01)

    def lookup_then_unfit(kwargs: dict[str, Any]) -> Any:
        if not any(m.get("role") == "tool" for m in kwargs["messages"]):
            return _tool_call_response("get_page_words", {"page": 1}, call_id="w1")
        bogus = {"invoice": {"text": "Goodnight", "locations": [{"page_number": 1}]}}
        return _tool_call_response("submit_values", {"values": bogus, "layout": layout_})

    respond = _responder({MATCHED: nested, ONE_PAGE: lookup_then_unfit})

    def outcome() -> dict[str, Any]:
        return {fid: _stats(workspace, ds_id, fid) for fid in fids}

    sync = _run_sync(workspace, ds_id, fids, capture_kwargs, respond)
    sync_stats, sync_rows = outcome(), _rows(workspace)
    executor, _ = _executor(respond)
    batch = extract_values_batch(
        workspace, ds_id, fids, config=_config(), executor=executor, write_stats=True, debug=True
    )
    batch_stats, batch_rows = outcome(), _rows(workspace)[len(sync_rows) :]

    sync_ok = sync[MATCHED]
    assert isinstance(sync_ok, ExtractionResult) and batch[MATCHED] == sync_ok
    assert sync_ok.values["title"]["text"] == "Hello world"
    sync_err, batch_err = sync[ONE_PAGE], batch[ONE_PAGE]
    assert isinstance(sync_err, ValuesExtractionFailed)
    assert type(batch_err) is type(sync_err) and str(batch_err) == str(sync_err)
    assert "nothing that fits the schema" in str(sync_err) and "'invoice'" in str(sync_err)
    for stats in (sync_stats, batch_stats):
        for fid in fids:
            stats[fid].pop("completed_at")
            for phase in stats[fid]["phases"].values():
                phase.pop("duration_s")
        assert stats[MATCHED]["phases"]["phase1"]["envelope_repairs"] == 1
        refused = stats[ONE_PAGE]
        assert refused["outcome"] == "error"
        assert refused["phases"]["phase1"]["envelope_repairs"] == 0
        # Kept before the refusal raised.
        assert refused["phase1_layout"] == layout_
    assert batch_stats == sync_stats
    assert len(batch_rows) == len(sync_rows) == 2
    by_file = {row["context"]["file_id"]: row for row in sync_rows}
    # Batch rows land in completion order; compare them per file.
    for b_row in batch_rows:
        assert {**b_row, "tier": TIER_STANDARD} == by_file[b_row["context"]["file_id"]]
    # The refused run's row counts the get_page_words call phase 1 made.
    assert by_file[ONE_PAGE]["outcome"] == "error"
    assert by_file[ONE_PAGE]["context"]["tool_calls"] == 1
