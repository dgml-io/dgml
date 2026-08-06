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

import numpy as np
import pytest
from clustering.config.schema import EncoderConfig
from clustering.encoders import build_encoder
from clustering.example import VIEW_TEXT_SEP, _build_text, split_view_spec


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


def _tfidf_cfg(files: Path) -> EncoderConfig:
    return EncoderConfig(
        name="tfidf",
        model_id="tfidf",
        embedding_dim=8,
        extra={"corpus_dir": str(files), "text_view": "full"},
    )


def test_tfidf_encoder_names_ocr_when_the_corpus_has_no_text(tmp_path: Path) -> None:
    # Scanned PDFs: the files were added fine and have page_text/ on disk, but
    # every page came back wordless. Distinct from "no page_text at all".
    files = tmp_path / "files"
    for i in range(3):
        _write_page(files / f"scan{i}", 1, [])

    with pytest.raises(ValueError, match="none contain extracted text") as exc:
        build_encoder(_tfidf_cfg(files))
    message = str(exc.value)
    assert "3 files" in message, "says how many were looked at"
    assert "--text-mode ocr|hybrid" in message, "names the fix, not just the symptom"


def test_tfidf_encoder_distinguishes_an_absent_corpus_from_a_wordless_one(tmp_path: Path) -> None:
    files = tmp_path / "files"
    files.mkdir(parents=True)
    with pytest.raises(ValueError, match="found no page_text"):
        build_encoder(_tfidf_cfg(files))


def test_tfidf_encoder_accepts_a_corpus_where_only_some_files_are_wordless(tmp_path: Path) -> None:
    # A partial OCR gap is not this encoder's problem to refuse — TF-IDF fits
    # over whatever text exists, and the empty rows simply carry no signal.
    files = tmp_path / "files"
    for i, body in enumerate(["rent roll tenant lease"] * 2 + ["balance sheet assets equity"] * 2):
        _write_page(
            files / f"doc{i}",
            1,
            [_w(tok, 100 + 50 * j, 100, 140 + 50 * j, 115) for j, tok in enumerate(body.split())],
        )
    _write_page(files / "scan", 1, [])

    out = build_encoder(_tfidf_cfg(files)).encode(["rent roll tenant"])
    assert out.pooled.shape == (1, 8)


# --- multi-view specs -------------------------------------------------------


def test_split_view_spec_single_and_multi() -> None:
    assert split_view_spec("full") == ("full",)
    assert split_view_spec("page1+full+salient_boost") == ("page1", "full", "salient_boost")


@pytest.mark.parametrize("spec", ["bogus", "page1+bogus", "page1+", "", "page1+page1"])
def test_split_view_spec_rejects_bad_specs(spec: str) -> None:
    # `TextView` is a bare `str` alias, so the spec is user input with no
    # `Literal` behind it — every rejection path is worth pinning.
    with pytest.raises(ValueError):
        split_view_spec(spec)


def test_build_text_multi_view_joins_each_view(doc_dir: Path) -> None:
    parts = _build_text(doc_dir, view="page1+full+headers").split(VIEW_TEXT_SEP)
    assert parts == [
        _build_text(doc_dir, view="page1"),
        _build_text(doc_dir, view="full"),
        _build_text(doc_dir, view="headers"),
    ]


def test_build_text_multi_view_keeps_part_count_when_empty(tmp_path: Path) -> None:
    # A file with no page_text still has to yield one part per view, or the
    # encoder can't line its blocks up.
    empty = tmp_path / "files" / "nodoc"
    empty.mkdir(parents=True)
    assert _build_text(empty, view="page1+full").split(VIEW_TEXT_SEP) == ["", ""]


def test_build_text_multi_view_strips_stray_separator(tmp_path: Path) -> None:
    # The separator can't come out of a PDF text layer, but if it ever did it
    # would desynchronize every subsequent view, so it's normalized away.
    d = tmp_path / "files" / "weird"
    _write_page(d, 1, [_w(f"a{VIEW_TEXT_SEP}b", 100, 500, 160, 515)])
    assert _build_text(d, view="page1+full").split(VIEW_TEXT_SEP) == ["a b", "a b"]


def _tiny_corpus(files: Path) -> None:
    """Nine 2-page docs, three vocabularies, page 1 discriminating and page 2 shared."""
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
        # A page-2-only term, on two thirds of the corpus so it survives both
        # `min_df=2` and `max_df=0.9` and can be looked for in the `full` block.
        tail = "boilerplate" if i < 6 else "appendix"
        _write_page(d, 2, [_w(tail, 100, 100, 200, 115)])


def _encoder(files: Path, view: str, *, dim: int = 8) -> object:
    return build_encoder(
        EncoderConfig(
            name="tfidf",
            model_id="tfidf",
            embedding_dim=dim,
            extra={"corpus_dir": str(files), "text_view": view},
        )
    )


@pytest.mark.parametrize(
    ("dim", "view"),
    [
        (8, "page1+full"),  # divides exactly
        (7, "page1+full"),  # doesn't divide — 3 per view, 6 of 7 used, 1 padded
        (6, "page1+full+salient_boost"),  # 2 per view, exactly at the floor
        (256, "page1+full+salient_boost"),  # the shipped width, three views
        (768, "page1+full+salient_boost"),  # the measured configuration
    ],
)
def test_tfidf_multi_view_keeps_declared_width(tmp_path: Path, dim: int, view: str) -> None:
    files = tmp_path / "files"
    _tiny_corpus(files)
    enc = _encoder(files, view, dim=dim)
    batch = [_build_text(files / "doc0", view=view)]
    out = enc.encode(batch)  # type: ignore[attr-defined]
    # Views share the configured width instead of multiplying it — the fusion and
    # manifold dims are sized against `embedding_dim`, so this is a contract, and
    # it has to hold for widths that don't divide by the view count as well as
    # ones that do. `encode` only pads, so a width that overflowed would ship a
    # tensor wider than the fusion expects.
    assert out.pooled.shape == (1, dim)
    assert abs(float(out.pooled.norm(dim=-1)[0]) - 1.0) < 1e-4


def test_tfidf_rejects_width_too_small_for_the_views(tmp_path: Path) -> None:
    files = tmp_path / "files"
    _tiny_corpus(files)
    # Each block needs >= 2 SVD components, so 3 views need >= 6 columns. Below
    # that there is no valid split and the encoder would emit a row wider than it
    # declares; it has to refuse instead.
    with pytest.raises(ValueError, match="too small for the 3 text views"):
        _encoder(files, "page1+full+salient_boost", dim=5)


def test_tfidf_warns_when_named_views_collapse_to_the_same_text(tmp_path: Path) -> None:
    files = tmp_path / "files"
    # Single-page docs with uniform font and nothing in the top band: `page1` is
    # the whole document and there is no salient signal, so all three views
    # assemble identical text despite being three different names.
    for i, body in enumerate(["rent roll tenant lease"] * 3 + ["balance sheet assets"] * 3):
        _write_page(
            files / f"doc{i}",
            1,
            [_w(tok, 100 + 50 * j, 500, 140 + 50 * j, 515) for j, tok in enumerate(body.split())],
        )
    with pytest.warns(RuntimeWarning, match="assemble identical text"):
        _encoder(files, "page1+full+salient_boost", dim=12)


def test_tfidf_names_the_empty_view_in_the_error(tmp_path: Path) -> None:
    files = tmp_path / "files"
    # Page 1 of every doc is stop words only, so the `page1` block has no
    # vocabulary while `full` does. sklearn's own message names neither.
    for i in range(4):
        d = files / f"doc{i}"
        _write_page(d, 1, [_w(tok, 100, 500, 140, 515) for tok in ("the", "and", "of")])
        _write_page(d, 2, [_w(tok, 100, 100, 140, 115) for tok in ("rent", "roll", "tenant")])
    with pytest.raises(ValueError, match=r"text view 'page1' .* yields no usable vocabulary"):
        _encoder(files, "page1+full", dim=8)


def test_tfidf_multi_view_all_empty_corpus_names_ocr(tmp_path: Path) -> None:
    # The scanned-PDF guard has to survive multi-view: each corpus entry is the
    # views joined by VIEW_TEXT_SEP, and the separator is not whitespace, so a
    # naive `.strip()` would let an all-empty corpus fall through to the less
    # specific per-view error instead of the OCR message.
    files = tmp_path / "files"
    for i in range(3):
        _write_page(files / f"scan{i}", 1, [])
    with pytest.raises(ValueError, match="none contain extracted text") as exc:
        _encoder(files, "page1+full+salient_boost", dim=8)
    assert "--text-mode ocr|hybrid" in str(exc.value)


def test_tfidf_multi_view_blocks_are_independently_fitted(tmp_path: Path) -> None:
    files = tmp_path / "files"
    _tiny_corpus(files)
    enc = _encoder(files, "page1+full")
    # One fitted (vectorizer, SVD) per view, and they saw different text: page 1
    # alone never contains the page-2 boilerplate term.
    vocabs = [v.vocabulary_ for v, _ in enc._blocks]  # type: ignore[attr-defined]
    assert len(vocabs) == 2
    assert "boilerplate" not in vocabs[0]
    assert "boilerplate" in vocabs[1]


def test_tfidf_multi_view_encodes_arbitrary_strings(tmp_path: Path) -> None:
    files = tmp_path / "files"
    _tiny_corpus(files)
    enc = _encoder(files, "page1+full")
    # S4 encodes category-name prototypes and the scenarios encode all-empty
    # placeholder rows: strings with no document (and so no views) behind them.
    # They must still produce a full-width row rather than raising.
    out = enc.encode(["rent roll", ""])  # type: ignore[attr-defined]
    assert out.pooled.shape == (2, 8)
    assert abs(float(out.pooled.norm(dim=-1)[0]) - 1.0) < 1e-4


def test_tfidf_single_view_output_unchanged_by_multi_view_support(tmp_path: Path) -> None:
    # The single-view path must be untouched: it is what every shipped config
    # uses, so it is pinned against a hand-rolled fit of the same pipeline.
    pytest.importorskip("sklearn")
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    files = tmp_path / "files"
    _tiny_corpus(files)
    corpus = [_build_text(files / f"doc{i}", view="page1") for i in range(9)]
    vec = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.9
    )
    tfidf = vec.fit_transform(corpus)
    # Both bounds spelled out rather than assumed: SVD rank is capped by the
    # vocabulary *and* by the corpus size, and hardcoding either would make the
    # reference silently diverge if `_tiny_corpus` changed size.
    rank = max(min(8, tfidf.shape[1] - 1, tfidf.shape[0] - 1), 2)
    svd = TruncatedSVD(n_components=rank, random_state=0)
    svd.fit(tfidf)
    batch = corpus[:2]
    expected = svd.transform(vec.transform(batch))
    expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)

    pooled = _encoder(files, "page1").encode(batch).pooled.numpy()  # type: ignore[attr-defined]
    assert pooled.shape == (2, 8)
    assert np.allclose(pooled[:, :rank], expected, atol=1e-6)
    # The rest is padding, and asserting it is zero is what would catch a wrong
    # component count hiding inside a right-width tensor.
    assert np.array_equal(pooled[:, rank:], np.zeros((2, 8 - rank), dtype=pooled.dtype))
