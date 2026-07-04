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

"""AwsProvider tests with mocked Textract client."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from dgml_core.errors import OcrFailed
from dgml_core.ocr import OcrConfig, OcrProviderName, extract_text_ocr
from dgml_core.storage import Workspace

from .conftest import make_fake_png


class _FakeTextract:
    """Returns a queued response per call, keyed by a unique substring in
    the input bytes so the test is order-independent (the production
    code calls pages concurrently)."""

    def __init__(self, responses_by_marker: dict[bytes, dict[str, Any]]) -> None:
        self._responses = responses_by_marker
        self._lock = threading.Lock()
        self.call_count = 0

    def detect_document_text(self, *, Document: dict[str, bytes]) -> dict[str, Any]:  # noqa: N803
        with self._lock:
            self.call_count += 1
        bytes_in = Document["Bytes"]
        for marker, response in self._responses.items():
            if marker in bytes_in:
                return response
        raise KeyError(f"no fake response for bytes {bytes_in[:32]!r}")


class _FakeBoto3Session:
    def __init__(self, client: _FakeTextract, profile_name: str | None, region_name: str | None):
        self.profile_name = profile_name
        self.region_name = region_name
        self._client = client

    def client(self, service_name: str) -> _FakeTextract:
        assert service_name == "textract"
        return self._client


def test_aws_missing_sdk_raises_ocr_failed(
    workspace: Workspace, text_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "boto3", None)
    cfg = OcrConfig(provider=OcrProviderName.AWS, region="us-east-1")
    with pytest.raises(OcrFailed, match="pip install dgml\\[aws\\]"):
        extract_text_ocr(
            text_pdf,
            workspace.file_text_dir("does-not-matter"),
            file_id="does-not-matter",
            page_images_dir=workspace.file_pages_dir("does-not-matter"),
            config=cfg,
        )


def test_aws_extract_writes_per_page_json(
    aws_config: Workspace,
    text_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Pretend ghostscript rendered two 1000x1000 page images. Real PNG
    # headers so _image_dimensions can read the dims; the embedded marker
    # routes the fake response.
    pages_dir = tmp_path / "page_images"
    pages_dir.mkdir()
    (pages_dir / "page_1.png").write_bytes(make_fake_png(1000, 1000, b"page-1-marker"))
    (pages_dir / "page_2.png").write_bytes(make_fake_png(1000, 1000, b"page-2-marker"))

    responses: dict[bytes, dict[str, Any]] = {
        b"page-1-marker": {
            "Blocks": [
                {
                    "BlockType": "WORD",
                    "Text": "hello",
                    "Geometry": {
                        "BoundingBox": {"Left": 0.1, "Top": 0.1, "Width": 0.2, "Height": 0.05}
                    },
                }
            ]
        },
        b"page-2-marker": {"Blocks": []},
    }
    fake_client = _FakeTextract(responses)

    captured: dict[str, Any] = {}

    def fake_session(*, profile_name: str | None, region_name: str | None) -> _FakeBoto3Session:
        captured["profile_name"] = profile_name
        captured["region_name"] = region_name
        return _FakeBoto3Session(fake_client, profile_name, region_name)

    import boto3

    monkeypatch.setattr(boto3, "Session", fake_session)

    out_dir = tmp_path / "page_text"
    cfg = OcrConfig(
        provider=OcrProviderName.AWS,
        region="us-east-1",
        profile="test-profile",
    )
    result = extract_text_ocr(
        text_pdf,
        out_dir,
        file_id="fid",
        page_images_dir=pages_dir,
        config=cfg,
    )

    assert result.pages_written == 2
    assert result.pages_with_words == 1
    assert result.total_words == 1
    assert captured == {"profile_name": "test-profile", "region_name": "us-east-1"}
    assert fake_client.call_count == 2

    p1 = json.loads((out_dir / "page_1.json").read_text())
    assert p1["page"] == 1
    assert p1["file_id"] == "fid"
    assert len(p1["words"]) == 1
    assert p1["words"][0]["t"] == "hello"
    # Bbox should be positive integers inside the image bounds.
    left, top, right, bottom = p1["words"][0]["l"]
    assert 0 < left < right < p1["width"]
    assert 0 < top < bottom < p1["height"]


def test_aws_extract_requires_page_images(
    aws_config: Workspace, text_pdf: Path, tmp_path: Path
) -> None:
    cfg = OcrConfig(provider=OcrProviderName.AWS, region="us-east-1")
    with pytest.raises(OcrFailed, match="no page images"):
        extract_text_ocr(
            text_pdf,
            tmp_path / "page_text",
            file_id="fid",
            page_images_dir=tmp_path / "page_images",  # does not exist
            config=cfg,
        )
