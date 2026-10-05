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

"""Extraction phase 1 on a PDF too large for one request.

The fake provider below rejects any request whose messages exceed a byte limit,
raising what litellm raises for Anthropic's 413 ``request_too_large``. Phase 1
must then split the document by page range, run once per part, and merge.
"""

from __future__ import annotations

import base64
import io
import json
import random
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core import layout
from dgml_core.docsets import DocSetStore
from dgml_core.errors import ValuesExtractionFailed
from dgml_core.grounded import GroundedConfig, extract_values
from dgml_core.models import FileRecord
from dgml_core.storage import Workspace
from litellm.exceptions import BadRequestError, ContextWindowExceededError

from .conftest import _write_text_pdf, write_config

_MODEL = "anthropic/claude-sonnet-4-5"
_PAGES = 8
# PDFium compresses the slices: a 2-page part is ~40 KB base64, a 4-page one ~80 KB,
# and the rest of a request a few KB. So quarters fit under this and halves do not.
_LIMIT = 70_000

# What Anthropic returns for a body over its request limit (HTTP 413). litellm
# maps it to a BadRequestError carrying the body text.
_TOO_LARGE_BODY = (
    '{"type":"error","error":{"type":"request_too_large",'
    '"message":"Request exceeds the maximum allowed number of bytes."}}'
)

_BILL_RNC = """\
namespace docset = "http://dgml.io/x/bill"

Bill =
  element docset:Bill {
    (text | Total | Items)*
  }

Total =
  element docset:Total {
    xsd:decimal
  }

Items =
  element docset:Items {
    Item*
  }

Item =
  element docset:Item {
    (text | Name)*
  }

Name =
  element docset:Name {
    text
  }
"""


def _too_large_error() -> Exception:
    return BadRequestError(
        message=f"AnthropicException - {_TOO_LARGE_BODY}", model=_MODEL, llm_provider="anthropic"
    )


def _big_pdf(tmp_path: Path, pages: int = _PAGES) -> bytes:
    """A PDF whose pages are ~30 KB each of incompressible-ish text, every page
    starting with a ``PAGEMARK<n>`` marker the fake model reads back."""
    rng = random.Random(0)
    texts = [
        f"PAGEMARK{n} " + "".join(rng.choice("abcdefghij") for _ in range(30_000))
        for n in range(1, pages + 1)
    ]
    path = tmp_path / "big.pdf"
    _write_text_pdf(path, texts)
    return path.read_bytes()


def _attached_pdf(kwargs: Mapping[str, Any]) -> bytes:
    for message in kwargs["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "file":
                    data = block["file"]["file_data"]
                    return base64.b64decode(data.split(",", 1)[1])
    raise AssertionError("no PDF in the request")


def _pages_in(pdf: bytes) -> list[int]:
    """The document page numbers the attached PDF holds, in order."""
    from pdfminer.high_level import extract_text

    return [int(m) for m in re.findall(r"PAGEMARK(\d+)", extract_text(io.BytesIO(pdf)))]


def _submit(values: dict[str, Any]) -> SimpleNamespace:
    call = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="submit_values", arguments=json.dumps(values))
    )
    msg = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])


def _bill_model(kwargs: dict[str, Any]) -> SimpleNamespace:
    """Answers like a model reading the attached PDF: one item per page, the
    total only where page 8 is, page numbers relative to the attached PDF."""
    pages = _pages_in(_attached_pdf(kwargs))
    bill: dict[str, Any] = {
        "Items": [
            {"Name": {"text": f"item{n}", "locations": [{"page_number": i}]}}
            for i, n in enumerate(pages, start=1)
        ]
    }
    if _PAGES in pages:
        bill["Total"] = {
            "text": "$8",
            "value": "8",
            "locations": [{"page_number": pages.index(_PAGES) + 1}],
        }
    return _submit({"values": {"Bill": bill}})


def _rejecting(
    limit: int, respond: Callable[[dict[str, Any]], Any], sizes: list[int]
) -> Callable[..., Any]:
    """A ``litellm.completion`` that refuses requests over *limit* bytes."""

    def fake(**kwargs: Any) -> Any:
        size = len(json.dumps(kwargs["messages"]).encode())
        sizes.append(size)
        if size > limit:
            raise _too_large_error()
        return respond(kwargs)

    return fake


def _seed(workspace: Workspace, pdf: bytes) -> tuple[str, str]:
    write_config(workspace, {"pdf": {"provider": "pypdfium2"}})
    fid = "f1aaaaaaaaaa"
    record = FileRecord(
        id=fid,
        original_path="/fake/big.pdf",
        original_filename="big.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=_PAGES,
        text_mode="digital",
    )
    workspace.docs.put_doc("files", fid, record.to_json())
    workspace.blobs.put_blob(layout.file_source_key(fid, "big.pdf"), pdf)
    for n in range(1, _PAGES + 1):
        words = [{"t": f"item{n}", "l": [100, 100 + n, 200, 140 + n]}]
        if n == _PAGES:
            words.append({"t": "$8", "l": [300, 300, 340, 340]})
        payload = {"file_id": fid, "page": n, "width": 1000, "height": 1000, "words": words}
        workspace.blobs.put_blob(layout.file_page_text_key(fid, n), json.dumps(payload).encode())
    store = DocSetStore(workspace)
    ds = store.create(name="Bill")
    store.set_schema(ds.id, _BILL_RNC)
    store.add_file(ds.id, fid)
    return ds.id, fid


def _config() -> GroundedConfig:
    return GroundedConfig(schema_model=_MODEL, values_model=_MODEL)


# ---- detection -------------------------------------------------------------


def test_is_request_too_large_recognizes_provider_errors() -> None:
    from dgml_core.llm import is_request_too_large

    assert is_request_too_large(_too_large_error())
    gemini = ContextWindowExceededError(
        message="GeminiException - 400 Request payload size exceeds the limit: 20971520 bytes.",
        model="gemini/gemini-2.5-pro",
        llm_provider="gemini",
    )
    assert is_request_too_large(gemini)

    class Http413(Exception):
        status_code = 413

    assert is_request_too_large(Http413("Request Entity Too Large"))
    # Found through the chain, as extraction wraps the provider error.
    try:
        try:
            raise _too_large_error()
        except Exception as inner:
            raise ValuesExtractionFailed("extraction call failed") from inner
    except ValuesExtractionFailed as wrapped:
        assert is_request_too_large(wrapped)


def test_is_request_too_large_ignores_other_errors() -> None:
    from dgml_core.llm import is_request_too_large

    bad = BadRequestError(
        message="AnthropicException - invalid tool schema", model=_MODEL, llm_provider="anthropic"
    )
    assert not is_request_too_large(bad)
    assert not is_request_too_large(RuntimeError("too many states for serving"))
    assert not is_request_too_large(RuntimeError("network down"))


# ---- phase 1 split ---------------------------------------------------------


def test_oversized_pdf_is_split_and_merged(workspace: Workspace, tmp_path: Path) -> None:
    pdf = _big_pdf(tmp_path)
    ds_id, fid = _seed(workspace, pdf)
    sizes: list[int] = []
    # Two pages fit, four do not: the whole document is refused, then each
    # half, then the quarters go through.
    limit = _LIMIT
    with patch("litellm.completion", side_effect=_rejecting(limit, _bill_model, sizes)) as m:
        result = extract_values(workspace, ds_id, fid, config=_config())

    first_request = m.call_args_list[0].kwargs
    assert _attached_pdf(first_request) == pdf  # the whole document was tried first
    assert sizes[0] > limit
    assert m.call_count == 1 + 2 + 4  # whole, two halves refused, four quarters

    bill = result.values["Bill"]
    assert [item["Name"]["text"] for item in bill["Items"]] == [
        f"item{n}" for n in range(1, _PAGES + 1)
    ]
    # Page numbers are back in the document's numbering, and phase 2 found
    # each value on that page.
    for n, item in enumerate(bill["Items"], start=1):
        (loc,) = item["Name"]["locations"]
        assert loc["page_number"] == n
        assert loc["bounding_box"] == [100, 100 + n, 200, 140 + n]
    assert bill["Total"]["locations"][0]["page_number"] == _PAGES

    stats = workspace.docs.get_doc("extraction_stats", f"{ds_id}/{fid}")
    assert stats is not None
    assert stats["outcome"] == "ok"
    assert stats["phases"]["phase1"]["pdf_parts"] == [[1, 2], [3, 4], [5, 6], [7, 8]]


def test_part_requests_say_which_pages_they_hold(workspace: Workspace, tmp_path: Path) -> None:
    ds_id, fid = _seed(workspace, _big_pdf(tmp_path))
    with patch("litellm.completion", side_effect=_rejecting(_LIMIT, _bill_model, [])) as m:
        extract_values(workspace, ds_id, fid, config=_config())
    last = m.call_args_list[-1].kwargs["messages"][1]["content"]
    texts = [b["text"] for b in last if b.get("type") == "text"]
    assert any(re.search(r"pages\s+7-8 of the document", t) for t in texts)


def test_page_words_lookup_reads_the_document_page(workspace: Workspace, tmp_path: Path) -> None:
    """In a part, the model numbers pages from 1; a ``get_page_words`` call
    for its page 1 reads the document page the part starts at."""
    ds_id, fid = _seed(workspace, _big_pdf(tmp_path))
    lookups: list[dict[str, Any]] = []

    def respond(kwargs: dict[str, Any]) -> Any:
        messages = kwargs["messages"]
        pages = _pages_in(_attached_pdf(kwargs))
        if pages[0] == 7 and messages[-1]["role"] != "tool":
            call = SimpleNamespace(
                id="w1",
                function=SimpleNamespace(name="get_page_words", arguments='{"page": 1}'),
            )
            msg = SimpleNamespace(content=None, tool_calls=[call])
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool")])
        if messages[-1]["role"] == "tool":
            lookups.append(json.loads(messages[-1]["content"]))
        return _bill_model(kwargs)

    with patch("litellm.completion", side_effect=_rejecting(_LIMIT, respond, [])):
        extract_values(workspace, ds_id, fid, config=_config())
    (lookup,) = lookups
    assert lookup["page"] == 1
    assert [w["text"] for w in lookup["words"]] == ["item7"]


def test_single_page_too_large_fails_clearly(workspace: Workspace, tmp_path: Path) -> None:
    ds_id, fid = _seed(workspace, _big_pdf(tmp_path))
    with (
        patch("litellm.completion", side_effect=_rejecting(10_000, _bill_model, [])),
        pytest.raises(ValuesExtractionFailed, match="page 1 alone"),
    ):
        extract_values(workspace, ds_id, fid, config=_config())


def test_accepted_request_is_unchanged(workspace: Workspace, tmp_path: Path) -> None:
    """A request the provider accepts is sent once, whole, exactly as before:
    the schema block then the original PDF, with no part note and no split in
    the stats."""
    pdf = _big_pdf(tmp_path)
    ds_id, fid = _seed(workspace, pdf)
    with patch("litellm.completion", side_effect=_rejecting(10**9, _bill_model, [])) as m:
        extract_values(workspace, ds_id, fid, config=_config())
    assert m.call_count == 1
    content = m.call_args_list[0].kwargs["messages"][1]["content"]
    assert [b["type"] for b in content] == ["text", "file"]
    assert content[1] == {
        "type": "file",
        "file": {"file_data": "data:application/pdf;base64," + base64.b64encode(pdf).decode()},
    }
    stats = workspace.docs.get_doc("extraction_stats", f"{ds_id}/{fid}")
    assert stats is not None
    assert "pdf_parts" not in stats["phases"]["phase1"]


def test_other_provider_errors_do_not_split(workspace: Workspace, tmp_path: Path) -> None:
    ds_id, fid = _seed(workspace, _big_pdf(tmp_path))
    with (
        patch("litellm.completion", side_effect=RuntimeError("invalid request")) as m,
        pytest.raises(ValuesExtractionFailed, match="invalid request"),
    ):
        extract_values(workspace, ds_id, fid, config=_config())
    assert m.call_count == 1


# ---- merging ---------------------------------------------------------------


def test_merge_part_values_concatenates_and_rebases_refs() -> None:
    from dgml_core.grounded import _merge_part_values

    base: dict[str, Any] = {
        "Bill": {
            "Items": [{"Name": {"text": "a"}}, {"Name": {"text": "b"}}],
            "Vendor": {"text": "Acme"},
            "Note": {"text": ""},
        }
    }
    extra: dict[str, Any] = {
        "Bill": {
            "Items": [{"Name": {"text": "c"}}],
            "Vendor": {"text": "Other"},
            "Note": {"text": "late"},
            "Count": {"text": "1", "computed": True, "derived_from": ["Bill.Items[0].Name"]},
        }
    }
    _merge_part_values(base, extra)
    bill = base["Bill"]
    assert [i["Name"]["text"] for i in bill["Items"]] == ["a", "b", "c"]
    assert bill["Vendor"]["text"] == "Acme"  # first non-empty wins
    assert bill["Note"]["text"] == "late"  # an empty earlier value does not
    assert bill["Count"]["derived_from"] == ["Bill.Items[2].Name"]
