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

"""The tfidf encoder must fit corpora too small for its own pruning bounds.

Two independent failures, both reached from real corpora while measuring:

``TfidfVectorizer(min_df=2, max_df=0.9)`` mixes an absolute document count with
a fraction, so the two bounds cross whenever ``0.9 * n < 2`` — every corpus of
two documents, whatever it contains. Above that the window ``[2, 0.9n]`` is
narrow enough that a homogeneous corpus can leave it empty. Both cases used to
surface as a raw sklearn ``ValueError`` suggesting the caller tune ``min_df`` /
``max_df``, which dgml does not expose. Sampling the four internal corpora 20
times per size showed the relaxation ladder engaging only at n <= 5 and never
at n >= 6, which is what lets this ship without re-scoring anything.

Separately, a corpus whose vocabulary prunes down to a *single term* reaches
``TruncatedSVD`` with one feature, which it refuses ("a minimum of 2 is
required"). That one needs no ladder to happen — it fires at the shipped
bounds, with nothing else out of the ordinary.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from clustering.config.schema import EncoderConfig
from clustering.encoders.lexical import _MAX_DF, _MIN_DF, TfidfEncoder


def _corpus(root: Path, docs: list[str]) -> Path:
    """Stage ``docs`` as a workspace ``files/`` dir the encoder can fit on."""
    files_dir = root / "files"
    for i, text in enumerate(docs):
        page_dir = files_dir / f"f{i}" / "page_text"
        page_dir.mkdir(parents=True)
        (page_dir / "page_1.json").write_text(
            json.dumps({"words": [{"t": w, "l": [0, 0, 10, 10]} for w in text.split()]}),
            encoding="utf-8",
        )
    return files_dir


def _encoder(files_dir: Path, dim: int = 256) -> TfidfEncoder:
    return TfidfEncoder(
        EncoderConfig(
            name="tfidf",
            model_id="tfidf",
            embedding_dim=dim,
            extra={"corpus_dir": str(files_dir), "text_view": "full"},
        )
    )


def _only_block(enc: TfidfEncoder) -> tuple[object, object, int]:
    """The single fitted block of a single-view encoder."""
    (block,) = enc._blocks
    return block


# A corpus large and varied enough that the shipped bounds are satisfiable:
# every term below appears in 2 of the 8 documents, inside the [2, 7.2] window.
_ORDINARY = [
    "acme invoice total amount due",
    "acme invoice total amount due",
    "borough lease rent roll schedule",
    "borough lease rent roll schedule",
    "custody statement account holdings",
    "custody statement account holdings",
    "delta capital call notice",
    "delta capital call notice",
]


def test_ordinary_corpus_is_untouched(tmp_path: Path) -> None:
    """The relaxation must be invisible wherever the encoder already worked.

    This is the whole no-regression argument, and it is what lets the change
    ship without re-scoring: the first rung of the ladder *is* today's
    parameter pair, so a corpus that fits today fits on it, with an unchanged
    vocabulary and no warning. If a future edit reorders the rungs or widens
    the reducer's rank rule, this fails.
    """
    files_dir = _corpus(tmp_path, _ORDINARY)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="tfidf encoder:", category=RuntimeWarning)
        enc = _encoder(files_dir)

    vectorizer, svd, dim = _only_block(enc)
    assert vectorizer.min_df == _MIN_DF  # type: ignore[attr-defined]
    assert vectorizer.max_df == _MAX_DF  # type: ignore[attr-defined]
    # The rank rule the encoder shipped with, unchanged: bounded by the
    # vocabulary and the corpus, floored at 2.
    assert dim == max(min(256, len(vectorizer.vocabulary_) - 1, len(_ORDINARY) - 1), 2)  # type: ignore[attr-defined]
    assert svd is not None


def test_two_document_corpus_fits(tmp_path: Path) -> None:
    """``0.9 * 2 < 2``: sklearn refused before ever looking at the text.

    This is the reported bug. Measured on all four internal corpora — 20 random
    two-document samples from each, 100% failure, message "max_df corresponds
    to < documents than min_df".
    """
    docs = [
        "invoice total amount due payment terms",
        "invoice total amount payable payment schedule",
    ]
    files_dir = _corpus(tmp_path, docs)
    with pytest.warns(RuntimeWarning, match="left no vocabulary"):
        enc = _encoder(files_dir)

    out = enc.encode(docs)
    assert out.pooled.shape == (2, 256)
    # Both documents are represented — neither row was zeroed on the way out.
    assert float(out.pooled[0].abs().sum()) > 0
    assert float(out.pooled[1].abs().sum()) > 0


def test_two_documents_sharing_nothing_stay_distinct(tmp_path: Path) -> None:
    """Relaxing ``max_df`` alone is not enough when the documents overlap in nothing.

    With ``min_df=2`` and no shared term, the second rung still prunes
    everything; only dropping ``min_df`` to 1 leaves a vocabulary. Both
    documents must come out with their own content, which is the whole point of
    preferring this over a clean failure — a representation that keeps only one
    of the two would satisfy "not equal" while being useless.
    """
    docs = ["aardvark badger capybara", "dolphin elephant flamingo"]
    files_dir = _corpus(tmp_path, docs)
    with pytest.warns(RuntimeWarning, match=r"refitted with min_df=1, max_df=1\.0"):
        enc = _encoder(files_dir)

    out = enc.encode(docs)
    assert out.pooled.shape == (2, 256)
    assert not out.pooled[0].equal(out.pooled[1])
    assert float(out.pooled[0].abs().sum()) > 0
    assert float(out.pooled[1].abs().sum()) > 0
    # Disjoint vocabularies must not be mapped onto each other.
    assert abs(float(out.pooled[0] @ out.pooled[1])) < 0.5


def test_identical_documents_fit(tmp_path: Path) -> None:
    """Every term is in 100% of the corpus, so ``max_df=0.9`` prunes all of them.

    A different trigger from the two-document case: the bounds are satisfiable
    here, the corpus just has nothing outside them.
    """
    files_dir = _corpus(tmp_path, ["rent roll schedule"] * 3)
    with pytest.warns(RuntimeWarning, match="near-identical"):
        enc = _encoder(files_dir)

    assert enc.encode(["rent roll schedule"]).pooled.shape == (1, 256)


def test_single_term_vocabulary_skips_the_reducer(tmp_path: Path) -> None:
    """A one-term vocabulary is not something TruncatedSVD will reduce.

    This corpus prunes to exactly one term (``aaa``) **at the shipped bounds**,
    so the ladder never engages and nothing warns about pruning — the old code
    went straight to "Found array with 1 feature(s) ... a minimum of 2 is
    required by TruncatedSVD" and died. Verified against the pre-change rank
    rule. Reached on a four-document sample of a real corpus.
    """
    files_dir = _corpus(tmp_path, ["aaa bbb", "aaa ccc", "ddd eee", "fff ggg", "iii hhh"])
    with pytest.warns(RuntimeWarning, match="vocabulary of one term"):
        enc = _encoder(files_dir)

    vectorizer, svd, dim = _only_block(enc)
    assert len(vectorizer.vocabulary_) == 1  # type: ignore[attr-defined]
    assert svd is None
    # Width 1, so a degenerate view can never widen the stacked output.
    assert dim == 1
    out = enc.encode(["aaa bbb", "ddd eee"])
    assert out.pooled.shape == (2, 256)
    # The single column survives to the output; the rest is the usual padding.
    assert int((out.pooled[0] != 0).sum()) == 1


def test_single_document_corpus_fits(tmp_path: Path) -> None:
    """One document is a legal workspace, and `dgml cluster` reaches the encoder.

    ``min_df=2`` cannot be met by one document, so the vectorizer refused.
    """
    files_dir = _corpus(tmp_path, ["quarterly rent roll for the borough property"])
    with pytest.warns(RuntimeWarning, match="left no vocabulary"):
        enc = _encoder(files_dir)

    out = enc.encode(["quarterly rent roll"])
    assert out.pooled.shape == (1, 256)
    assert float(out.pooled[0].abs().sum()) > 0


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 8])
def test_every_small_corpus_emits_the_declared_width(tmp_path: Path, n: int) -> None:
    """The width contract must hold at every size, not just the ones with a test.

    The rest of the pipeline — fusion, the manifold, the reducer — is configured
    for `embedding_dim` and does not check. This caught a real off-by-one: on a
    one-document corpus sklearn returns fewer components than were requested, and
    padding against the *requested* count emitted 255 columns instead of 256.
    """
    docs = [f"document {i} rent roll schedule borough lease term {i}" for i in range(n)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        enc = _encoder(_corpus(tmp_path, docs))

    out = enc.encode(docs)
    assert out.pooled.shape == (n, 256)
    assert bool(out.pooled.isfinite().all())


def test_wordless_corpus_still_fails_loudly(tmp_path: Path) -> None:
    """No rung may turn a textless corpus into all-zero embeddings.

    A corpus with no text at all is caught by the scanned-PDF guard above the
    ladder, which names OCR — the more specific answer. The guarantee this
    pins is that it is still an error and not a run that silently means
    nothing.
    """
    files_dir = _corpus(tmp_path, ["", "", ""])
    with pytest.raises(ValueError, match="none contain extracted text"):
        _encoder(files_dir)


def test_failure_message_names_the_view_and_the_remedy(tmp_path: Path) -> None:
    """Text that survives ``strip()`` but not the stop list must fail actionably.

    Nothing is pruned on the last rung, so a failure there is the corpus and
    not the bounds. sklearn's own message points at ``min_df``/``max_df``,
    which dgml does not expose; the wrapper has to name the view that died, say
    the pruning is not what is left to blame, and point at OCR.
    """
    files_dir = _corpus(tmp_path, ["the and of", "the and of"])
    with pytest.raises(ValueError) as excinfo:
        _encoder(files_dir)

    message = str(excinfo.value)
    assert "'full'" in message
    assert "pruning disabled" in message
    assert "--text-mode ocr|hybrid" in message
