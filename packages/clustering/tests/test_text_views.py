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

"""Tests for the structure-aware text views and the TF-IDF lexical encoder.

These exercise the word-box → text assembly logic (``_build_text``) and the
corpus-fitted ``tfidf`` encoder on a tiny synthetic workspace — no network,
no model weights.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from clustering.config.schema import EncoderConfig
from clustering.encoders import build_encoder
from clustering.example import _build_text


def _write_page(
    file_dir: Path, page_no: int, words: list[dict[str, object]], *, height: int = 1000
) -> None:
    """Write one ``page_text/page_N.json`` with the given word boxes."""
    page_dir = file_dir / "page_text"
    page_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "file_id": file_dir.name,
        "page": page_no,
        "width": 800,
        "height": height,
        "words": words,
    }
    (page_dir / f"page_{page_no}.json").write_text(json.dumps(payload), encoding="utf-8")


def _w(t: str, x0: int, y0: int, x1: int, y1: int) -> dict[str, object]:
    return {"t": t, "l": [x0, y0, x1, y1]}


@pytest.fixture
def doc_dir(tmp_path: Path) -> Path:
    """A 2-page doc: a big-font title at the top of page 1, then small body text."""
    d = tmp_path / "files" / "doc1"
    # Page 1: tall (big-font) title near the top, then small body words.
    _write_page(
        d,
        1,
        [
            _w("RENT", 100, 20, 200, 80),  # tall (height 60) + top band
            _w("ROLL", 210, 20, 300, 80),  # tall + top band
            _w("tenant", 100, 500, 160, 515),  # small body (height 15)
            _w("12345", 200, 500, 260, 560),  # tall NUMBER → must be excluded from salient
        ],
    )
    # Page 2: only small body words (no salient).
    _write_page(d, 2, [_w("rent", 100, 100, 140, 115), _w("paid", 150, 100, 190, 115)])
    return d


def test_full_view_concatenates_all_pages(doc_dir: Path) -> None:
    text = _build_text(doc_dir, view="full")
    assert text == "RENT ROLL tenant 12345 rent paid"


def test_page1_view_is_first_page_only(doc_dir: Path) -> None:
    text = _build_text(doc_dir, view="page1")
    assert text == "RENT ROLL tenant 12345"
    assert "rent paid" not in text


def test_headers_view_keeps_title_drops_numbers(doc_dir: Path) -> None:
    text = _build_text(doc_dir, view="headers")
    # Big-font / top-band words kept; the tall *number* is excluded.
    assert text == "RENT ROLL"
    assert "12345" not in text


def test_salient_boost_prepends_repeated_headers(doc_dir: Path) -> None:
    text = _build_text(doc_dir, view="salient_boost")
    # Salient text repeated ahead of the full body so type tokens dominate.
    assert text.startswith("RENT ROLL RENT ROLL RENT ROLL ")
    assert text.endswith("RENT ROLL tenant 12345 rent paid")


def test_missing_page_text_returns_empty(tmp_path: Path) -> None:
    empty = tmp_path / "files" / "nodoc"
    empty.mkdir(parents=True)
    assert _build_text(empty, view="full") == ""


def test_headers_falls_back_to_full_when_no_salient(tmp_path: Path) -> None:
    # A pure body (uniform small font, no top-band words) has no salient signal.
    d = tmp_path / "files" / "body_only"
    _write_page(
        d,
        1,
        [_w(f"row{i}", 100, 500 + i, 140, 515 + i) for i in range(5)],
    )
    headers = _build_text(d, view="headers")
    full = _build_text(d, view="full")
    assert headers == full  # graceful degradation


def test_tfidf_encoder_fits_and_encodes(tmp_path: Path) -> None:
    files = tmp_path / "files"
    # Three docs with distinct vocabularies so TF-IDF has signal to learn.
    for i, body in enumerate(
        ["rent roll tenant lease unit occupancy"] * 3
        + ["balance sheet assets liabilities equity"] * 3
        + ["capital call notice commitment drawdown"] * 3
    ):
        d = files / f"doc{i}"
        _write_page(
            d,
            1,
            [_w(tok, 100 + 50 * j, 100, 140 + 50 * j, 115) for j, tok in enumerate(body.split())],
        )
    cfg = EncoderConfig(
        name="tfidf",
        model_id="tfidf",
        embedding_dim=8,
        extra={"corpus_dir": str(files), "text_view": "full"},
    )
    enc = build_encoder(cfg)
    out = enc.encode(["rent roll tenant", "balance sheet assets"])
    assert out.pooled.shape == (2, 8)
    # Rows are L2-normalized.
    norms = out.pooled.norm(dim=-1)
    assert all(abs(float(n) - 1.0) < 1e-4 for n in norms)
