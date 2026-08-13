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

"""OpenAI text-embedding encoder.

Embeds text with an OpenAI embedding model through ``litellm`` — already a
dependency (used by the LLM clustering / classification paths), provider-
agnostic, so this adds no new package or license. Mirrors the Gemini encoder;
select with ``encoder_text=openai``.

Single-vector text encoder. Set ``embedding_dim`` to the output width you want.
``text-embedding-3-large`` is 3072 natively; the ``text-embedding-3`` models are
Matryoshka-trained, so a *smaller* ``embedding_dim`` works only when you also
pass ``extra.dimensions`` (forwarded to the API) — otherwise the model returns
its native width and :meth:`encode` rejects the mismatch rather than corrupting
downstream fusion silently.

Three things separate an API encoder from the local ones, each a knob here:

* **Credentials.** Read from ``extra.api_key``, else the env var named by
  ``extra.api_key_env`` (default ``OPENAI_API_KEY``). Neither found is not an
  error: ``api_key`` stays ``None`` and litellm resolves it from its own
  environment; a genuinely absent credential surfaces as litellm's own
  authentication error on the first call.
* **Input length.** ``text-embedding-3-*`` documents an 8191-token limit. A
  multi-page document can exceed it, so ``cfg.max_length`` (a token budget
  converted to characters via :data:`_CHARS_PER_TOKEN`) cuts the text here —
  reproducible and bounds the bill. ``None`` sends the text through untouched.
* **Failure.** One rate-limit response mid-corpus would otherwise discard every
  embedding already paid for in the same run, so transient failures are retried
  with exponential backoff (``extra.num_retries``, default
  :data:`_DEFAULT_NUM_RETRIES`). The loop lives here, in
  :meth:`OpenAIEncoder._embed_chunk`: litellm's retry dispatch is keyed on call
  type and covers ``completion`` / ``responses`` only, so a ``num_retries=``
  kwarg is accepted and ignored for ``embedding``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import torch

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder

_DEFAULT_MODEL = "openai/text-embedding-3-large"

# Retries for the transient failures a corpus-sized sweep provokes (429s
# especially). Three rides out a burst without turning a permanent failure into
# a long hang; the backoff below is what gives the quota time to refill.
_DEFAULT_NUM_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_CAP_SECONDS = 30.0

# ``cfg.max_length`` is a token count; the API takes text. 4 chars/token is the
# usual English average — an average, not a bound — so the resulting window is
# approximate: a budget, not a guarantee.
_CHARS_PER_TOKEN = 4


class OpenAIEncoder(Encoder[str]):
    """Text embeddings from an OpenAI model, via ``litellm.embedding``."""

    multi_vector = False

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
        del device  # API-based: no local device
        self.embedding_dim = cfg.embedding_dim
        self.model = cfg.model_id or _DEFAULT_MODEL
        self.batch_size = max(1, int(cfg.extra.get("batch_size", 100)))
        self.timeout = float(cfg.extra.get("timeout", 60.0))
        self.num_retries = max(0, int(cfg.extra.get("num_retries", _DEFAULT_NUM_RETRIES)))
        self.max_chars = (
            None if cfg.max_length is None else max(1, int(cfg.max_length) * _CHARS_PER_TOKEN)
        )
        # Optional passthrough, only sent when set so the provider default (and
        # any measurement taken against it) stays in force. ``dimensions`` asks
        # a Matryoshka model for a narrower output, and is what makes an
        # ``embedding_dim`` below the model's native width work at all.
        self.extra_kwargs: dict[str, Any] = {}
        dimensions = cfg.extra.get("dimensions")
        if dimensions is not None:
            self.extra_kwargs["dimensions"] = dimensions
        api_key: Any = cfg.extra.get("api_key")
        if not api_key:
            env = str(cfg.extra.get("api_key_env", "OPENAI_API_KEY"))
            api_key = os.environ.get(env)
        # ``None`` on purpose when nothing was found — litellm then applies its
        # own credential resolution instead of us pre-emptively failing.
        self.api_key: str | None = str(api_key) if api_key else None

    def _prepare(self, batch: Sequence[str]) -> list[str]:
        if self.max_chars is None:
            return list(batch)
        return [t[: self.max_chars] for t in batch]

    def _embed_chunk(self, litellm: Any, chunk: list[str]) -> Any:
        """One request, retried on the transient classes only.

        ``litellm.num_retries`` does not cover this call (its retry dispatch is
        keyed on call type and only ``completion`` / ``responses`` reach it), so
        the loop lives here. Auth / bad-request failures are deterministic and
        not retried. The module is passed in because the import is deferred to
        :meth:`encode` (litellm is an optional dependency for this package).
        """
        transient = (
            litellm.RateLimitError,
            litellm.Timeout,
            litellm.APIConnectionError,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
        )
        for attempt in range(self.num_retries + 1):
            try:
                return litellm.embedding(
                    model=self.model,
                    input=chunk,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    **self.extra_kwargs,
                )
            except transient:
                if attempt == self.num_retries:
                    raise
                time.sleep(min(_BACKOFF_BASE_SECONDS * 2**attempt, _BACKOFF_CAP_SECONDS))
        raise AssertionError("unreachable")  # pragma: no cover — loop returns or raises

    @torch.no_grad()
    def encode(self, batch: Sequence[str]) -> EncoderOutput:
        # Nothing to embed: answer with a correctly shaped empty tensor rather
        # than sending a request, to keep the caller's ``[N, D]`` contract.
        if not batch:
            return EncoderOutput(pooled=torch.zeros((0, self.embedding_dim), dtype=torch.float32))

        try:
            import litellm
        except ImportError as exc:  # pragma: no cover — optional dep
            raise ImportError(
                "The OpenAI encoder requires litellm. Install it with "
                "`pip install dgml-clustering[openai]` (or `dgml[clustering]`, which "
                "already includes it)."
            ) from exc

        texts = self._prepare(batch)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            resp = self._embed_chunk(litellm, chunk)
            for item in resp.data:
                vec = getattr(item, "embedding", None)
                if vec is None:
                    vec = item["embedding"]
                if len(vec) != self.embedding_dim:
                    raise ValueError(
                        f"OpenAI model {self.model!r} returned {len(vec)}-d embeddings but "
                        f"encoder_text.embedding_dim={self.embedding_dim}; set embedding_dim to "
                        "match the model's output width, or pass extra.dimensions for a "
                        "Matryoshka model to request a narrower output."
                    )
                rows.append(list(vec))
        if len(rows) != len(texts):
            # Row order carries the doc<->vector alignment for the whole
            # pipeline; a short or long response would misattribute every
            # embedding after the gap.
            raise ValueError(
                f"OpenAI returned {len(rows)} embedding(s) for {len(texts)} input(s); "
                "the response cannot be aligned back to the batch."
            )
        return EncoderOutput(pooled=torch.tensor(rows, dtype=torch.float32))


def _factory(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return OpenAIEncoder(cfg, device=device)


register_encoder("openai")(_factory)
