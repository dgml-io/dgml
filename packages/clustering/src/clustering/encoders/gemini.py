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

"""Gemini text-embedding encoder.

Embeds text with a Google Gemini embedding model through ``litellm`` — already
a dependency (used by the LLM clustering / classification paths), provider-
agnostic, so this adds no new package or license.

Single-vector text encoder. Set ``embedding_dim`` to the model's output width
(3072 for ``gemini/gemini-embedding-001``); a mismatch is caught at encode
time rather than corrupting downstream fusion silently.

Three things separate an API encoder from the local ones, and each is a knob
here rather than an assumption:

* **Credentials.** Read from ``extra.api_key``, else the env var named by
  ``extra.api_key_env`` (default ``GEMINI_API_KEY``). Finding neither is *not*
  an error: ``api_key`` stays ``None`` and litellm resolves it from its own
  environment, which is how Vertex service-account and ``GOOGLE_API_KEY``
  setups work. A genuinely absent credential surfaces as litellm's own
  authentication error on the first call.
* **Input length.** ``gemini-embedding-001`` documents a 2048-token input
  limit, and a multi-page document blows past it. What the endpoint does with
  the excess is *its* choice, not ours — in our own benchmark runs it accepted
  every over-limit request without error — which is the reason to cut the text
  ourselves: an explicit window is reproducible and bounds what we pay for,
  where an implicit one leaves both to the provider. ``cfg.max_length`` sets
  it, as a token budget converted to characters (see :data:`_CHARS_PER_TOKEN`);
  ``None`` sends the text through untouched, for callers that already
  truncated.
* **Failure.** One rate-limit response mid-corpus would otherwise discard every
  embedding already paid for in the same run, so transient failures are retried
  with exponential backoff (``extra.num_retries``, default
  :data:`_DEFAULT_NUM_RETRIES`). The retry loop lives *here*, in
  :meth:`GeminiEncoder._embed_chunk`, and not in a ``num_retries=`` kwarg:
  litellm's retry dispatch is keyed on call type and covers ``completion`` /
  ``responses`` only, so that kwarg is accepted and ignored for ``embedding``.

Select with ``encoder_text=gemini``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import torch

from clustering.config.schema import EncoderConfig, EncoderName
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder

_DEFAULT_MODEL = "gemini/gemini-embedding-001"

# Retries for the transient failures a corpus-sized sweep provokes (429s
# especially). Three rides out a burst without turning a permanent failure into
# a long hang; the backoff below is what actually gives the quota time to refill.
_DEFAULT_NUM_RETRIES = 3
_BACKOFF_BASE_SECONDS = 2.0
_BACKOFF_CAP_SECONDS = 30.0

# ``cfg.max_length`` is a token count; the API takes text. Converting needs a
# chars-per-token figure, and 4 is the usual English average — an *average*,
# not a bound. Measured on 36 documents from the corpora this encoder was
# benchmarked on (first 3 pages, cut at 8000 characters): median 3.7
# chars/token, minimum 2.1, and 8% of them still over 2048 tokens after the
# cut. So treat the resulting window as approximate. It is not load-bearing for
# correctness — the endpoint accepted those over-limit inputs — but it does mean
# ``max_length`` is a budget, not a guarantee, and a token-dense corpus lands
# nearer 2 chars/token than 4.
_CHARS_PER_TOKEN = 4


class GeminiEncoder(Encoder[str]):
    """Text embeddings from a Gemini model, via ``litellm.embedding``."""

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
        # Optional passthroughs. Only sent when the caller sets them, so the
        # provider's own defaults (and any measurement taken against them)
        # stay in force. ``dimensions`` asks the model for a narrower output via
        # Matryoshka representation learning, and is what makes an
        # ``embedding_dim`` below the model's native width work at all — the
        # check in :meth:`encode` rejects the mismatch you get without it.
        self.extra_kwargs: dict[str, Any] = {}
        for field in ("task_type", "dimensions"):
            value = cfg.extra.get(field)
            if value is not None:
                self.extra_kwargs[field] = value
        api_key: Any = cfg.extra.get("api_key")
        if not api_key:
            env = str(cfg.extra.get("api_key_env", "GEMINI_API_KEY"))
            api_key = os.environ.get(env)
        # ``None`` on purpose when nothing was found — litellm then applies its
        # own credential resolution (GOOGLE_API_KEY, Vertex ADC, …) instead of
        # us pre-emptively failing a setup that would have worked.
        self.api_key: str | None = str(api_key) if api_key else None

    def _prepare(self, batch: Sequence[str]) -> list[str]:
        if self.max_chars is None:
            return list(batch)
        return [t[: self.max_chars] for t in batch]

    def _embed_chunk(self, litellm: Any, chunk: list[str]) -> Any:
        """One request, retried on the failures that are worth retrying.

        ``litellm.num_retries`` does not cover this call: litellm's retry
        dispatch is keyed on call type and only ``completion`` / ``responses``
        reach it, so the kwarg is accepted and ignored for ``embedding``. Hence
        the loop. Only transient classes are retried — an auth or bad-request
        failure is deterministic, and sleeping 2/4/8s before repeating it just
        delays the traceback.

        The module comes in as an argument because the import is deferred to
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
        # than sending a request, because ``torch.tensor([])`` would be 1-D and
        # break the caller's ``[N, D]`` contract. No shipped caller passes an
        # empty batch today (CachingEncoder forwards only its misses, under an
        # ``if missing:``), so this is a guard on the ABC's contract, not a hot
        # path.
        if not batch:
            return EncoderOutput(pooled=torch.zeros((0, self.embedding_dim), dtype=torch.float32))

        try:
            import litellm
        except ImportError as exc:  # pragma: no cover — optional dep
            raise ImportError(
                "The Gemini encoder requires litellm. Install it with "
                "`pip install dgml-clustering[gemini]` (or `dgml[clustering]`, which "
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
                        f"Gemini model {self.model!r} returned {len(vec)}-d embeddings but "
                        f"encoder_text.embedding_dim={self.embedding_dim}; set embedding_dim "
                        "to match the model's output width."
                    )
                rows.append(list(vec))
        if len(rows) != len(texts):
            # Row order carries the doc↔vector alignment for the whole
            # pipeline. A short (or long) response would silently misattribute
            # every embedding after the gap.
            raise ValueError(
                f"Gemini returned {len(rows)} embedding(s) for {len(texts)} input(s); "
                "the response cannot be aligned back to the batch."
            )
        return EncoderOutput(pooled=torch.tensor(rows, dtype=torch.float32))


def _factory(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return GeminiEncoder(cfg, device=device)


register_encoder(EncoderName.GEMINI)(_factory)
