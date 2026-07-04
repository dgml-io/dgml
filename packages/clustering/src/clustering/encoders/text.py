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

"""Text encoders.

A single :class:`SentenceTransformerEncoder` backs every text embedder
that's exposed through the ``sentence_transformers`` API: the original
``st_minilm`` plus the modern instruction-tuned / Matryoshka frontier:

- **E5** (``intfloat/e5-large-v2`` and friends; Wang et al., Microsoft) —
  instruction-tuned, query/passage prefix convention, strong on MTEB.
- **BGE** (``BAAI/bge-large-en-v1.5``) — query-side instruction prefix
  for retrieval; BGE-M3 (multi-vector + sparse) lives behind a separate
  config when we add it.
- **GTE** (``Alibaba-NLP/gte-large-en-v1.5``) — long-context
  open-weights embedder. The smaller v1.5 doesn't require a prefix; the
  bigger ``gte-Qwen2`` family does.
- **Stella** (``dunzhang/stella_en_400M_v5``) — Matryoshka representations,
  smaller checkpoint than NV-Embed/E5-Mistral with competitive MTEB.
- **Jina v3** (``jinaai/jina-embeddings-v3``) — task-aware LoRA adapters,
  Matryoshka down to 32 dim. We use the default task here; per-task
  routing is a future extension.

All share one wrapper so any new ST-compatible checkpoint can be added
with just a Hydra config — no Python required. The wrapper prepends
``cfg.doc_prefix`` to every input (corpus-encoding case); call sites
that need asymmetric retrieval can read ``cfg.query_prefix`` directly.

LayoutLM-text is a near-future add-on; its tokenizer / layout-aware path
lives alongside the image side in PROJECT_PLAN.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder
from clustering.utils import resolve_device


class SentenceTransformerEncoder(Encoder[str]):
    """Thin wrapper over ``sentence_transformers.SentenceTransformer``.

    Honors ``cfg.doc_prefix``: if set, every input is prefixed with it
    before being sent to the underlying model. This is how
    instruction-tuned embedders (E5, BGE, Stella, …) signal the
    *passage / document* side of their query-vs-passage convention.
    """

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Add the 'encoders' extra "
                "with: `uv sync --extra encoders`."
            ) from exc
        if cfg.model_id is None:
            raise ValueError(f"Encoder {cfg.name!r} requires a model_id.")

        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim
        self.multi_vector = cfg.multi_vector
        self.doc_prefix: str = cfg.doc_prefix or ""
        # Encoding batch size. Long-context models (Jina v3, GTE) fall back to a
        # dense O(seq²) attention when flash-attention is unavailable (CPU / MPS),
        # so a smaller batch keeps the materialized score tensor in check. Tunable
        # via ``cfg.extra["batch_size"]``.
        self.batch_size: int = int(cfg.extra.get("batch_size", 32))
        # Optional task adapter (Jina v3 LoRA). Jina v3 ships task-specific
        # adapters — ``separation`` (clustering / reranking), ``classification``,
        # ``text-matching``, ``retrieval.query`` / ``retrieval.passage``. Passing
        # ``task`` to ``encode`` activates the matching LoRA; the model card
        # recommends ``separation`` for clustering. Left ``None`` (no adapter) for
        # models that don't support it. Tunable via ``cfg.extra["task"]``.
        task = cfg.extra.get("task")
        self.task: str | None = str(task) if task else None
        info = resolve_device(device)
        self.device = info.torch_device
        # SentenceTransformer accepts ``trust_remote_code`` on some newer
        # checkpoints (Jina v3, GTE long-context). We pass it through any
        # ``cfg.extra["trust_remote_code"]`` opt-in to keep the default safe.
        st_kwargs: dict[str, Any] = {"device": str(self.device)}
        if cfg.extra.get("trust_remote_code"):
            st_kwargs["trust_remote_code"] = True
        self.model = SentenceTransformer(cfg.model_id, **st_kwargs)
        # Cap the input sequence length. Long-context embedders (Jina v3 ≈ 8k
        # tokens) default ``max_seq_length`` to their full context window; with
        # no flash-attention the attention scores tensor is materialized densely
        # as ``[batch, heads, seq, seq]``, which blows up to tens of GB and OOMs
        # on a long document. ``cfg.max_length`` clamps it (never *above* the
        # model's native context — that would index past its position table).
        if cfg.max_length is not None:
            native = getattr(self.model, "max_seq_length", None)
            self.model.max_seq_length = (
                cfg.max_length if native is None else min(cfg.max_length, native)
            )

    def _apply_prefix(self, batch: Sequence[str]) -> list[str]:
        if not self.doc_prefix:
            return list(batch)
        return [f"{self.doc_prefix}{x}" for x in batch]

    @torch.no_grad()
    def encode(self, batch: Sequence[str]) -> EncoderOutput:
        inputs = self._apply_prefix(batch)
        # ``task`` only applies to models with LoRA adapters (Jina v3). When set,
        # SentenceTransformer routes it to the Transformer module's forward (other
        # modules filter it out by signature), activating the matching adapter.
        encode_kwargs: dict[str, Any] = {}
        if self.task is not None:
            encode_kwargs["task"] = self.task
        embeddings = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            **encode_kwargs,
        )
        # `embeddings` is a tensor [B, D] on `self.device`. Move to CPU so
        # downstream code is device-agnostic; callers can re-`.to(...)` as needed.
        return EncoderOutput(pooled=embeddings.detach().cpu())


def _factory_st(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    """Shared factory for every ST-compatible text encoder.

    Registered under each model family's canonical short name so users
    select via ``encoder_text=e5`` (etc.) and the Hydra config supplies
    ``model_id`` and the prefix templates.
    """
    return SentenceTransformerEncoder(cfg, device=device)


# ── Registry ────────────────────────────────────────────────────────────
# All five new families share the wrapper above; their Hydra YAMLs
# provide the canonical ``model_id`` / ``doc_prefix`` / ``query_prefix``.
register_encoder("st_minilm")(_factory_st)
register_encoder("e5")(_factory_st)
register_encoder("bge")(_factory_st)
register_encoder("gte")(_factory_st)
register_encoder("stella")(_factory_st)
register_encoder("jina")(_factory_st)
