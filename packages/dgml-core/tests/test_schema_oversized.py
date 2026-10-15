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

"""Schema generation when the planning documents are too large for one request.

The fake provider below rejects any request whose messages exceed a byte limit,
raising what litellm raises for Anthropic's 413 ``request_too_large``. Schema
generation must then send leading pages of the largest samples and retry.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import random
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core import layout
from dgml_core.errors import SchemaGenerationFailed
from dgml_core.grounded import GroundedConfig, generate_schema
from dgml_core.models import FileRecord
from dgml_core.storage import Workspace
from litellm.exceptions import BadRequestError

from .conftest import _write_text_pdf, write_config

_MODEL = "anthropic/claude-opus-4-7"
# The big sample is ~320 KB base64 and the small ones ~1-2 KB. PDFium compresses
# slices, so a leading-pages excerpt of the big one is well under this.
_LIMIT = 150_000

_TOO_LARGE_BODY = (
    '{"type":"error","error":{"type":"request_too_large",'
    '"message":"Request exceeds the maximum allowed number of bytes."}}'
)

_FIELDS = [{"name": "Title", "kind": "field", "datatype": "text"}]


def _pdf(tmp_path: Path, name: str, pages: int, chars_per_page: int, seed: int) -> bytes:
    rng = random.Random(seed)
    texts = [
        f"PAGEMARK{n} " + "".join(rng.choice("abcdefghij") for _ in range(chars_per_page))
        for n in range(1, pages + 1)
    ]
    path = tmp_path / f"{name}.pdf"
    _write_text_pdf(path, texts)
    return path.read_bytes()


def _seed(workspace: Workspace, fid: str, pdf: bytes, pages: int) -> None:
    record = FileRecord(
        id=fid,
        original_path=f"/fake/{fid}.pdf",
        original_filename=f"{fid}.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=pages,
        text_mode="digital",
    )
    workspace.docs.put_doc("files", fid, record.to_json())
    workspace.blobs.put_blob(layout.file_source_key(fid, f"{fid}.pdf"), pdf)


def _attached_pdfs(kwargs: Mapping[str, Any]) -> list[bytes]:
    content = kwargs["messages"][1]["content"]
    return [
        base64.b64decode(b["file"]["file_data"].split(",", 1)[1])
        for b in content
        if b.get("type") == "file"
    ]


def _texts(kwargs: Mapping[str, Any]) -> str:
    content = kwargs["messages"][1]["content"]
    return "\n".join(b["text"] for b in content if b.get("type") == "text")


def _pages_in(pdf: bytes) -> list[int]:
    from pdfminer.high_level import extract_text

    return [int(m) for m in re.findall(r"PAGEMARK(\d+)", extract_text(io.BytesIO(pdf)))]


def _schema_response() -> SimpleNamespace:
    call = SimpleNamespace(
        id="s1",
        function=SimpleNamespace(name="submit_schema", arguments=json.dumps({"fields": _FIELDS})),
    )
    msg = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])


def _rejecting(limit: int) -> Callable[..., Any]:
    def fake(**kwargs: Any) -> Any:
        if len(json.dumps(kwargs["messages"]).encode()) > limit:
            raise BadRequestError(
                message=f"AnthropicException - {_TOO_LARGE_BODY}",
                model=_MODEL,
                llm_provider="anthropic",
            )
        return _schema_response()

    return fake


def _config() -> GroundedConfig:
    return GroundedConfig(schema_model=_MODEL, values_model=_MODEL)


@pytest.fixture
def samples(workspace: Workspace, tmp_path: Path) -> dict[str, bytes]:
    """Two small samples around one big one: ``a`` (2 pages), ``big`` (8
    pages, ~240 KB), ``b`` (2 pages)."""
    write_config(workspace, {"pdf": {"provider": "pypdfium2"}})
    pdfs = {
        "fa0000000000": _pdf(tmp_path, "a", 2, 200, 1),
        "fbig00000000": _pdf(tmp_path, "big", 8, 30_000, 2),
        "fb0000000000": _pdf(tmp_path, "b", 2, 200, 3),
    }
    for fid, pdf in pdfs.items():
        _seed(workspace, fid, pdf, len(_pages_in(pdf)))
    return pdfs


def test_oversized_samples_are_trimmed_not_dropped(
    workspace: Workspace, samples: dict[str, bytes], caplog: pytest.LogCaptureFixture
) -> None:
    ids = list(samples)
    with (
        caplog.at_level(logging.WARNING, logger="dgml_core"),
        patch("litellm.completion", side_effect=_rejecting(_LIMIT)) as m,
    ):
        rnc = generate_schema(workspace, ids, config=_config(), docset_name="Leases")
    assert "Title" in rnc

    assert m.call_count == 2  # refused whole, accepted trimmed
    assert _attached_pdfs(m.call_args_list[0].kwargs) == list(samples.values())

    retry = m.call_args_list[1].kwargs
    small_a, big, small_b = _attached_pdfs(retry)
    # Every sample is still there; the small ones are untouched.
    assert small_a == samples["fa0000000000"]
    assert small_b == samples["fb0000000000"]
    # The big one is its leading pages.
    pages = _pages_in(big)
    assert pages == list(range(1, len(pages) + 1))
    assert 1 <= len(pages) < 8
    # The model is told which sample is partial.
    note = _texts(retry)
    assert re.search(rf"sample 2 .*pages 1-{len(pages)} of 8", note, re.IGNORECASE)
    assert "3 attached PDFs" in note
    assert "fbig00000000" in caplog.text


def test_sample_whose_first_page_does_not_fit_is_dropped(
    workspace: Workspace, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_config(workspace, {"pdf": {"provider": "pypdfium2"}})
    pdfs = {
        "fa0000000000": _pdf(tmp_path, "a", 2, 200, 1),
        "fbig00000000": _pdf(tmp_path, "big", 8, 30_000, 2),
        "fhuge0000000": _pdf(tmp_path, "huge", 1, 250_000, 4),  # one ~250 KB page
    }
    for fid, pdf in pdfs.items():
        _seed(workspace, fid, pdf, len(_pages_in(pdf)))
    with (
        caplog.at_level(logging.WARNING, logger="dgml_core"),
        patch("litellm.completion", side_effect=_rejecting(_LIMIT)) as m,
    ):
        generate_schema(workspace, list(pdfs), config=_config(), docset_name="Leases")

    final = m.call_args_list[-1].kwargs
    attached = _attached_pdfs(final)
    assert len(attached) == 2
    assert attached[0] == pdfs["fa0000000000"]
    assert "2 attached PDFs" in _texts(final)
    dropped = [r for r in caplog.records if "fhuge0000000" in r.getMessage()]
    assert dropped and dropped[0].levelno == logging.WARNING


def test_fails_when_no_sample_fits(workspace: Workspace, samples: dict[str, bytes]) -> None:
    with (
        patch("litellm.completion", side_effect=_rejecting(1_000)),
        pytest.raises(SchemaGenerationFailed, match="even trimmed"),
    ):
        generate_schema(workspace, list(samples), config=_config(), docset_name="Leases")


def test_accepted_request_is_unchanged(workspace: Workspace, samples: dict[str, bytes]) -> None:
    """A request the provider accepts goes once, exactly as before: the intro
    text then every sample whole, with no excerpt note."""
    with patch("litellm.completion", side_effect=_rejecting(10**9)) as m:
        generate_schema(workspace, list(samples), config=_config(), docset_name="Leases")
    assert m.call_count == 1
    content = m.call_args_list[0].kwargs["messages"][1]["content"]
    assert [b["type"] for b in content] == ["text", "file", "file", "file"]
    assert _attached_pdfs(m.call_args_list[0].kwargs) == list(samples.values())
    assert "excerpt" not in content[0]["text"].lower()


def test_other_provider_errors_are_not_retried(
    workspace: Workspace, samples: dict[str, bytes]
) -> None:
    with (
        patch("litellm.completion", side_effect=RuntimeError("invalid request")) as m,
        pytest.raises(SchemaGenerationFailed, match="invalid request"),
    ):
        generate_schema(workspace, list(samples), config=_config(), docset_name="Leases")
    assert m.call_count == 1


def test_is_request_too_large() -> None:
    from dgml_core.llm import is_request_too_large

    too_large = BadRequestError(
        message=f"AnthropicException - {_TOO_LARGE_BODY}", model=_MODEL, llm_provider="anthropic"
    )
    assert is_request_too_large(too_large)
    assert is_request_too_large(RuntimeError("400 Request payload size exceeds the limit"))
    assert not is_request_too_large(RuntimeError("invalid request"))
