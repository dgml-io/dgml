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
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder


class TfidfEncoder(Encoder[str]):
    """Corpus-fitted TF-IDF → Truncated-SVD (LSA) text encoder."""

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
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
        text_view = str(cfg.extra.get("text_view", "full"))
        corpus = self._read_corpus(Path(corpus_dir), text_view)
        if not corpus:
            raise ValueError(f"tfidf encoder found no page_text under {corpus_dir!r}.")

        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
        )
        tfidf = self._vectorizer.fit_transform(corpus)
        # SVD rank is bounded by both the vocabulary and the corpus size.
        n_components = min(cfg.embedding_dim, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        n_components = max(n_components, 2)
        self._svd = TruncatedSVD(n_components=n_components, random_state=0)
        self._svd.fit(tfidf)
        # The pipeline expects a fixed embedding width; pad SVD output up to the
        # configured dim with zeros if the rank was capped below it.
        self.embedding_dim = cfg.embedding_dim
        self._n_components = n_components

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

    def encode(self, batch: Sequence[str]) -> EncoderOutput:
        import numpy as np

        tfidf = self._vectorizer.transform(list(batch))
        reduced = self._svd.transform(tfidf)  # [B, n_components]
        # L2-normalize so cosine/spherical geometry matches the dense encoders.
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        reduced = reduced / np.clip(norms, 1e-12, None)
        pooled = torch.from_numpy(reduced).float()
        if self._n_components < self.embedding_dim:
            pad = torch.zeros((pooled.shape[0], self.embedding_dim - self._n_components))
            pooled = torch.cat([pooled, pad], dim=-1)
        return EncoderOutput(pooled=pooled)


@register_encoder("tfidf")
def _factory_tfidf(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return TfidfEncoder(cfg, device=device)
