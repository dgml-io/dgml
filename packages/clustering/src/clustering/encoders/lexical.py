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

"""Lexical (TF-IDF + LSA) text encoder.

A sparse, corpus-fitted counterpoint to the dense transformer embedders. Dense
document embeddings smear visually/semantically similar financial documents
together (a balance sheet *is* a financial statement); a TF-IDF representation
instead keys on the *characteristic vocabulary* of each document type — "rent
roll", "capital account", "schedule of investments" — which is exactly the
discriminative signal those families differ on.

TF-IDF needs corpus-global document frequencies, which the per-batch
:meth:`Encoder.encode` contract can't supply. So this encoder fits once at
construction over the whole workspace corpus (path + text view passed through
``cfg.extra``), reduces the sparse matrix to ``cfg.embedding_dim`` dense
components with Truncated SVD (i.e. LSA), and then ``encode`` just transforms
each batch through the frozen vectorizer + SVD. Output vectors are L2-normalized
so they drop into the same spherical/UMAP pipeline as the dense encoders.

``cfg.extra['text_view']`` may name several views joined by ``+``
(``"page1+full+salient_boost"``), in which case one TF-IDF + SVD block is fitted
**per view** and the blocks are stacked into one row, each view taking an equal
share of ``cfg.embedding_dim`` so the declared output width is unchanged.

That is deliberately not the same thing as concatenating the view *texts*. One
fit over merged text yields a single vocabulary in which the words the views
share merely have a higher term frequency — which ``sublinear_tf=True`` then
dampens on purpose. Separate fits give the reducer each view's own vocabulary,
document-frequency statistics and SVD basis, so it can use whichever view
separates a given pair of documents. The two shapes were measured against each
other on four real corpora: separate fits improved on every one, merged text on
two, so only the first is offered here.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder

# Document-frequency pruning bounds, named because the fit has to reason about
# whether they are satisfiable on the corpus in front of it — see
# :meth:`TfidfEncoder._fit_view`.
_MIN_DF = 2  # an absolute document count
_MAX_DF = 0.9  # a fraction of the documents


class TfidfEncoder(Encoder[str]):
    """Corpus-fitted TF-IDF → Truncated-SVD (LSA) text encoder.

    One fitted block per view named in ``cfg.extra['text_view']``. A single view
    — the default — is the one-block case and takes the identical code path.
    """

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
        try:
            # The vectorizer is imported where it is built (``_fit_view``); one
            # sklearn import is enough to turn a missing install into this
            # message rather than a traceback from halfway down the fit.
            from sklearn.decomposition import TruncatedSVD
        except ImportError as exc:  # pragma: no cover - exercised only without sklearn
            raise ImportError(
                "scikit-learn is required for the 'tfidf' encoder. It ships with the "
                "clustering deps (used by the reducers); run `uv sync`."
            ) from exc

        self.cfg = cfg
        self.multi_vector = False
        corpus_dir = cfg.extra.get("corpus_dir")
        if not corpus_dir:
            raise ValueError(
                "tfidf encoder requires cfg.extra['corpus_dir'] (the workspace files/ dir) "
                "so it can fit document frequencies over the whole corpus."
            )
        # Imported here, not at module scope: `clustering.example` reaches
        # `clustering.scenarios`, which reaches this module.
        from clustering.example import VIEW_TEXT_SEP, split_view_spec

        text_view = str(cfg.extra.get("text_view", "full"))
        self._views = split_view_spec(text_view)  # raises on an unknown view name
        corpus = self._read_corpus(Path(corpus_dir), text_view)
        if not corpus:
            raise ValueError(f"tfidf encoder found no page_text under {corpus_dir!r}.")
        # An all-empty corpus is the scanned-PDF case, and it is worth its own
        # message: sklearn's own failure here is "empty vocabulary; perhaps the
        # documents only contain stop words", which names neither the cause nor
        # the fix. The files exist and were added successfully — they just have
        # no digital text layer to read.
        # In multi-view mode each entry is the views joined by ``VIEW_TEXT_SEP``;
        # strip that out first so an all-empty multi-view corpus still trips the
        # scanned-PDF message rather than falling through to a per-view error.
        if not any(doc.replace(VIEW_TEXT_SEP, "").strip() for doc in corpus):
            raise ValueError(
                f"tfidf encoder found {len(corpus)} files under {corpus_dir!r} but none "
                "contain extracted text. Scanned/image PDFs need OCR: configure an 'ocr' "
                "provider in config.toml (Apple Vision on macOS, or azure/aws) and re-add "
                "the files with --text-mode ocr|hybrid so page_text/ is populated."
            )

        # Views share the declared width rather than multiplying it, so a
        # multi-view spec emits exactly `cfg.embedding_dim` like every other
        # encoder and needs no coordinated change to the fusion/manifold dims.
        # Anyone wanting the wider representation raises `embedding_dim`.
        #
        # Every block below is capped at `embedding_dim // n_views` — the SVD
        # width by its rank bound, and a reducer-less block is a single column —
        # so the total can only fall short of `embedding_dim`, never exceed it,
        # and `encode` pads the difference. What the check below buys is the
        # floor, not the ceiling: a view whose share is under 2 components
        # cannot carry a direction, and silently emitting one is worse than
        # saying the width is too small for the spec.
        floor = 2 * len(self._views)
        if cfg.embedding_dim < floor:
            raise ValueError(
                f"tfidf encoder: embedding_dim={cfg.embedding_dim} is too small for the "
                f"{len(self._views)} text views in {text_view!r}. The views share the "
                "configured width and each needs at least 2 SVD components, so "
                f"embedding_dim must be >= {floor}. A multi-view spec generally wants a "
                "width per view comparable to a single-view run (e.g. embedding_dim=768 "
                "for three views), not the single-view width split three ways."
            )
        per_view_dim = cfg.embedding_dim // len(self._views)
        columns = self._split_columns(corpus)
        self._warn_collapsed_views(columns, corpus_dir=str(corpus_dir), spec=text_view)
        self._blocks: list[tuple[Any, Any, int]] = []
        n_components = 0
        for name, texts in zip(self._views, columns, strict=True):
            vectorizer, tfidf = self._fit_view(
                texts, view=name, spec=text_view, corpus_dir=str(corpus_dir)
            )
            if tfidf.shape[1] < 2:
                # TruncatedSVD requires at least two features, so the floor
                # below asks a one-term matrix for two components and raises.
                # Nothing is lost by skipping it: a single term already is its
                # own coordinate. Width 1, so this can never widen the output.
                # The view carries no signal at all in this state — warn,
                # because in a multi-view spec it still takes its full share of
                # the stacked row and nothing downstream would ever say so.
                warnings.warn(
                    f"tfidf encoder: text view {name!r} (of {text_view!r}) has a "
                    f"vocabulary of one term over the {len(texts)} files under "
                    f"{corpus_dir!r}, so it separates no two documents. Every document "
                    "gets the same value for this view.",
                    RuntimeWarning,
                    stacklevel=1,
                )
                dim, svd = 1, None
            else:
                # SVD rank is bounded by both the vocabulary and the corpus size.
                dim = max(min(per_view_dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1), 2)
                svd = TruncatedSVD(n_components=dim, random_state=0)
                svd.fit(tfidf)
                # Read the width back rather than trusting the request: on a
                # corpus that spans fewer dimensions than were asked for — one
                # document, say — sklearn returns the components it could find,
                # and padding against the requested count emits a row one column
                # short of `embedding_dim`.
                dim = int(svd.components_.shape[0])
            self._blocks.append((vectorizer, svd, dim))
            n_components += dim
        # The pipeline expects a fixed embedding width; pad SVD output up to the
        # configured dim with zeros if the rank was capped below it.
        self.embedding_dim = cfg.embedding_dim
        self._n_components = n_components

    @staticmethod
    def _fit_view(texts: list[str], *, view: str, spec: str, corpus_dir: str) -> tuple[Any, Any]:
        """Fit one view's vectorizer, relaxing the pruning bounds only on failure.

        Returns the fitted vectorizer and its document-term matrix.

        The document-frequency bounds are a noise heuristic, not a correctness
        requirement, and on a short or homogeneous corpus they can leave no
        vocabulary at all — or cross outright, since ``_MIN_DF`` is an absolute
        document count and ``_MAX_DF`` a fraction, so ``_MAX_DF * n < _MIN_DF``
        holds for *every* corpus of two documents whatever it contains.
        sklearn's answer is to raise and suggest tuning ``min_df``/``max_df``,
        which dgml does not expose, leaving the caller nothing to act on.

        So step the bounds down instead, cheapest concession first. Both bounds
        were swept on the four internal corpora on the shipped path: relaxing
        ``max_df`` to 1.0 moved NMI by exactly 0.0000 on three of the four and
        left the vocabulary byte-identical on three (4 terms out of 4265 on the
        fourth), while ``min_df=1`` cost 0.066 NMI on the smallest corpus. So
        ``max_df`` goes first and ``min_df`` last: a price worth paying to run
        at all, and not worth paying otherwise. The first rung is exactly the
        shipped pair, so a corpus that fits today fits on it, with an unchanged
        vocabulary and no warning; the later rungs are reachable only where the
        encoder used to raise.

        This mirrors the reduce-method fallback in
        :mod:`clustering.scenarios.clustering`, which likewise degrades with a
        ``RuntimeWarning`` rather than failing the run.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        ladder = ((_MIN_DF, _MAX_DF), (_MIN_DF, 1.0), (1, 1.0))
        for rung, (min_df, max_df) in enumerate(ladder):
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=min_df,
                max_df=max_df,
            )
            try:
                tfidf = vectorizer.fit_transform(texts)
            except ValueError as exc:
                if rung < len(ladder) - 1:
                    continue
                # Nothing is pruned on the last rung, so the bounds are not what
                # is left to blame: this view really has no usable text. sklearn's
                # own message names neither the view nor the fix, and with several
                # fitted blocks any one of them can be the one that killed the run.
                raise ValueError(
                    f"tfidf encoder: text view {view!r} (of {spec!r}) yields no usable "
                    f"vocabulary over the {len(texts)} files under {corpus_dir!r}, even "
                    f"with document-frequency pruning disabled — sklearn reports: {exc}. "
                    "That view is empty or stop-words-only for this corpus; drop it from "
                    "text_view, or populate page_text/ via OCR (--text-mode ocr|hybrid) "
                    "if the files are scanned."
                ) from exc
            if rung > 0:
                warnings.warn(
                    f"tfidf encoder: document-frequency pruning (min_df={_MIN_DF}, "
                    f"max_df={_MAX_DF}) left no vocabulary for text view {view!r} over the "
                    f"{len(texts)} files under {corpus_dir!r}; refitted with min_df={min_df}, "
                    f"max_df={max_df}. Expect a weaker text signal — the corpus is small or "
                    "its documents are near-identical. A corpus this size is better served "
                    "by `dgml cluster --method llm`.",
                    RuntimeWarning,
                    # Config-driven dispatch (`build_encoder` -> `_REGISTRY[name]`
                    # -> `__init__` -> here), so no caller frame belongs to the
                    # user; point at the check itself, as _warn_collapsed_views does.
                    stacklevel=1,
                )
            return vectorizer, tfidf
        raise AssertionError("unreachable: the last rung either returns or raises")

    @staticmethod
    def _read_corpus(files_dir: Path, text_view: str) -> list[str]:
        """Read every workspace file's text under ``text_view`` (sorted by id)."""
        from clustering.example import _build_text

        if not files_dir.is_dir():
            return []
        texts: list[str] = []
        for file_dir in sorted(p for p in files_dir.iterdir() if p.is_dir()):
            texts.append(_build_text(file_dir, view=text_view))
        return texts

    def _warn_collapsed_views(
        self, columns: list[list[str]], *, corpus_dir: str, spec: str
    ) -> None:
        """Warn when two *differently named* views assembled identical text.

        ``split_view_spec`` rejects a spec that names a view twice, but distinct
        names can still produce the same text for a given corpus: ``_build_text``
        degrades ``headers`` and ``salient_boost`` to the full body for a document
        with no layout signal (uniform font sizes, nothing in the top band — a pure
        table), and ``page1`` *is* the full body for a single-page document. When
        that happens across the whole corpus the two blocks are bit-identical, so
        the spec spends part of its width restating one view and the L2 over the
        stacked row then discounts that content relative to a genuine extra view.
        That is worse than not asking for the second view at all, and nothing
        downstream would ever say so — it just scores lower.
        """
        if len(columns) < 2:
            return
        # At most a handful of views, so compare exactly rather than by hash.
        groups: list[tuple[list[str], list[str]]] = []  # (texts, view names)
        for name, texts in zip(self._views, columns, strict=True):
            for known, names in groups:
                if texts == known:
                    names.append(name)
                    break
            else:
                groups.append((texts, [name]))
        collapsed = [names for _, names in groups if len(names) > 1]
        if not collapsed:
            return
        which = "; ".join(" == ".join(names) for names in collapsed)
        warnings.warn(
            f"tfidf encoder: text view spec {spec!r} names views that assemble identical "
            f"text for every file under {corpus_dir!r} ({which}), so their blocks are "
            f"duplicates — {len(columns) - len(groups)} of {len(columns)} blocks restate "
            "another view, and normalizing the stacked row then weighs that content down "
            "rather than adding a view. 'headers' and 'salient_boost' fall back to the full "
            "body for files with no layout signal, and 'page1' is the full body for "
            "single-page files. Drop the duplicated view from text_view.",
            RuntimeWarning,
            # Config-driven dispatch (`build_encoder` -> `_REGISTRY[name]` -> here),
            # so no caller frame belongs to the user; point at the check itself.
            stacklevel=1,
        )

    def _split_columns(self, texts: Sequence[str]) -> list[list[str]]:
        """Transpose view-joined per-document text into one text list per view.

        A string carrying no separator was not assembled by ``_build_text`` — it
        is a caller encoding an arbitrary string, which the scenarios legitimately
        do for S4's category-name prototypes and for the all-empty placeholder
        rows that stand in for a skipped modality. Such a string has no per-view
        decomposition, so it is fed to *every* block: the result is that same text
        under each view's vocabulary, which keeps its width and geometry
        comparable with the documents it will be matched against.
        """
        from clustering.example import VIEW_TEXT_SEP

        n = len(self._views)
        if n == 1:
            return [list(texts)]
        columns: list[list[str]] = [[] for _ in range(n)]
        for text in texts:
            parts = text.split(VIEW_TEXT_SEP)
            if len(parts) != n:
                parts = [text.replace(VIEW_TEXT_SEP, " ")] * n
            for column, part in zip(columns, parts, strict=True):
                column.append(part)
        return columns

    def encode(self, batch: Sequence[str]) -> EncoderOutput:
        import numpy as np

        def l2(x: np.ndarray) -> np.ndarray:
            """L2-normalize rows so cosine/spherical geometry matches the dense encoders."""
            return np.asarray(x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None))

        def project(texts: Sequence[str], vectorizer: Any, svd: Any) -> np.ndarray:
            sparse = vectorizer.transform(texts)
            # No SVD ⇒ a one-term vocabulary, which is already its own single
            # coordinate (see __init__); the column passes straight through.
            return l2(svd.transform(sparse) if svd is not None else sparse.toarray())

        blocks = [
            project(texts, vectorizer, svd)
            for texts, (vectorizer, svd, _) in zip(
                self._split_columns(batch), self._blocks, strict=True
            )
        ]
        # Normalize each view's block before stacking, so a view is weighted by
        # its own geometry and not by how much raw SVD magnitude it happens to
        # carry; then normalize the stacked row, because the unit norm is what
        # the rest of the pipeline reads off this encoder.
        reduced = blocks[0] if len(blocks) == 1 else l2(np.hstack(blocks))
        pooled = torch.from_numpy(reduced).float()
        if self._n_components < self.embedding_dim:
            pad = torch.zeros((pooled.shape[0], self.embedding_dim - self._n_components))
            pooled = torch.cat([pooled, pad], dim=-1)
        return EncoderOutput(pooled=pooled)


@register_encoder("tfidf")
def _factory_tfidf(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return TfidfEncoder(cfg, device=device)
