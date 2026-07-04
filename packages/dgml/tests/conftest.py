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

"""Shared fixtures for the dgml test suite."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from dgml_core.pages import GS_BINARIES
from dgml_core.storage import Workspace

PAGE_WIDTH_PTS = 612
PAGE_HEIGHT_PTS = 792


def _write_blank_pdf(path: Path, pages: int) -> None:
    """Write an ``pages``-page PDF with no text, via the dependency-free
    hand-constructed writer (empty content stream per page)."""
    _write_text_pdf(path, pages_text=[""] * pages)


def _write_text_pdf(path: Path, pages_text: list[str]) -> None:
    """Write a minimal PDF where each page has one line of Helvetica text
    rendered at (100, 700) in PDF points (origin bottom-left).

    Hand-constructed bytes so digital-text extraction has a stable, dependency-
    light fixture. Each page is US Letter (612x792 pts); single Type1 Helvetica
    font; one BT/ET text block per page.
    """
    out = bytearray()
    offsets: list[int] = []

    def add_object(body: bytes) -> int:
        offsets.append(len(out))
        obj_num = len(offsets)
        out.extend(f"{obj_num} 0 obj\n".encode())
        out.extend(body)
        out.extend(b"\nendobj\n")
        return obj_num

    out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    n_pages = len(pages_text)
    catalog_id = 1
    pages_id = 2
    font_id = 3
    page_obj_ids = list(range(font_id + 1, font_id + 1 + n_pages))
    content_obj_ids = list(range(page_obj_ids[-1] + 1, page_obj_ids[-1] + 1 + n_pages))

    add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    add_object(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_id, content_id in zip(page_obj_ids, content_obj_ids, strict=True):
        body = (
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 {PAGE_WIDTH_PTS} {PAGE_HEIGHT_PTS}] "
            f"/Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode()
        # The page object we just emitted lands at `page_id` because we add
        # objects in declaration order.
        assert add_object(body) == page_id

    for text, content_id in zip(pages_text, content_obj_ids, strict=True):
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_body = f"BT /F1 24 Tf 100 700 Td ({escaped}) Tj ET\n".encode()
        body = f"<< /Length {len(stream_body)} >>\nstream\n".encode() + stream_body + b"endstream"
        assert add_object(body) == content_id

    xref_offset = len(out)
    n_objects = len(offsets)
    out.extend(f"xref\n0 {n_objects + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        (
            f"trailer\n<< /Size {n_objects + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )

    path.write_bytes(bytes(out))


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    return ws


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    out = tmp_path / "sample.pdf"
    _write_blank_pdf(out, pages=2)
    return out


@pytest.fixture
def sample_pdf_alt(tmp_path: Path) -> Path:
    out = tmp_path / "alt.pdf"
    _write_blank_pdf(out, pages=1)
    return out


@pytest.fixture
def text_pdf(tmp_path: Path) -> Path:
    """Two-page PDF with embedded Helvetica text on each page."""
    out = tmp_path / "with-text.pdf"
    _write_text_pdf(out, pages_text=["Hello World", "Second Page Text"])
    return out


@pytest.fixture
def mixed_pdf(tmp_path: Path) -> Path:
    """Two-page PDF: page 1 has Helvetica text, page 2 has no glyphs.

    Exercises the partial-empty digital-extraction path — pdfminer succeeds
    overall but `pages_with_words < pages_written`.
    """
    out = tmp_path / "mixed.pdf"
    _write_text_pdf(out, pages_text=["Only the first page has text", ""])
    return out


def has_ghostscript() -> bool:
    return any(shutil.which(name) is not None for name in GS_BINARIES)


needs_gs = pytest.mark.skipif(
    not has_ghostscript(),
    reason="ghostscript not installed",
)


def write_ocr_config(workspace: Workspace, ocr: dict[str, object]) -> None:
    """Write ``<workspace>/config.json`` with the given ``ocr`` section."""
    workspace.config_path.write_text(json.dumps({"ocr": ocr}, indent=2) + "\n", encoding="utf-8")


def write_classification_config(workspace: Workspace, classification: dict[str, object]) -> None:
    """Write ``<workspace>/config.json`` with the given ``classification`` section."""
    workspace.config_path.write_text(
        json.dumps({"classification": classification}, indent=2) + "\n",
        encoding="utf-8",
    )


def make_fake_png(width: int, height: int, payload: bytes = b"") -> bytes:
    """Build a minimal but valid PNG so ``dgml_core.ocr._image_dimensions``
    can parse the width/height. ``payload`` is appended after the IHDR
    chunk and before IEND; tests use it to embed a routing marker that
    fake provider clients look up to return the right per-page response.

    Structure: 8-byte PNG signature, IHDR chunk (with valid CRC32 over
    type+data so strict PNG consumers don't reject it), payload bytes,
    IEND chunk. No actual image data — but our dim parser stops after
    IHDR, and consumers scanning for the routing marker still find it.
    """
    import zlib

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR payload: width(4) height(4) bitdepth(1) colortype(1) comp(1) filter(1) interlace(1)
    ihdr_data = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"  # 8-bit, color type 2 (RGB), no interlace
    )
    ihdr_chunk = (
        (13).to_bytes(4, "big")
        + b"IHDR"
        + ihdr_data
        + zlib.crc32(b"IHDR" + ihdr_data).to_bytes(4, "big")
    )
    iend_chunk = b"\x00\x00\x00\x00IEND" + zlib.crc32(b"IEND").to_bytes(4, "big")
    return sig + ihdr_chunk + payload + iend_chunk


@pytest.fixture
def azure_config(workspace: Workspace) -> Workspace:
    write_ocr_config(
        workspace,
        {
            "provider": "azure",
            "endpoint": "https://example.cognitiveservices.azure.com/",
            "api_key_env": "TEST_AZURE_KEY",
        },
    )
    return workspace


@pytest.fixture
def aws_config(workspace: Workspace) -> Workspace:
    write_ocr_config(
        workspace,
        {
            "provider": "aws",
            "region": "us-east-1",
            "profile": "test-profile",
        },
    )
    return workspace


@pytest.fixture(autouse=True)
def _stub_add_links(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite hermetic: `generate`'s final semantic-link step calls the
    LLM, so stub it to a no-op (returns the XML unchanged, no links)."""
    monkeypatch.setattr(
        "dgml_core.generation.links.add_links",
        lambda xml, config, **kw: (xml, []),
    )
