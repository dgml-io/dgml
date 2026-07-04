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

"""Qwen3-VL-Embedding image encoder (``Qwen/Qwen3-VL-Embedding-8B`` / ``-2B``).

Unlike :mod:`clustering.encoders.qwen_vl` — which borrows the *vision tower*
of a generative Qwen2/2.5-VL checkpoint and mean-pools its patches — this is a
purpose-built **multimodal embedding** model (base: Qwen3-VL-8B-Instruct). It
emits a single, already-normalized vector per input in a shared text+image
space, so we use it directly as a page-image encoder: each page image is sent
through the model's ``sentence_transformers`` interface and the returned vector
is ``pooled``.

The model is instruction-aware and Matryoshka-trained (MRL): the native 4096-d
embedding can be truncated to any width in ``[64, 4096]``. We pass
``cfg.embedding_dim`` as the SentenceTransformer ``truncate_dim`` so the encoder
honors the dim the rest of the pipeline (fusion / manifold) is configured for,
and re-normalize after truncation.

Single-vector only: the model exposes one pooled embedding, so
``multi_vector=True`` (i.e. ``fusion=late_interaction``) is rejected — use
ColPali for multi-vector page retrieval.

Notes:
    This is an 8B-parameter checkpoint that ships custom modeling code, so
    ``trust_remote_code`` is required (defaulted ``True`` here, overridable via
    ``cfg.extra['trust_remote_code']``) and a GPU is strongly recommended. The
    checkpoint's native dtype is used unless ``cfg.extra['torch_dtype']`` (e.g.
    ``"bfloat16"``) overrides it — forcing float32 would need ~32 GB just for
    weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder
from clustering.utils import resolve_device


class Qwen3VLEmbeddingEncoder(Encoder[Image.Image]):
    """Qwen3-VL-Embedding page-image encoder. Single-vector (shared space)."""

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
        if cfg.multi_vector:
            raise ValueError(
                "Qwen3-VL-Embedding is single-vector only: it emits one pooled "
                "embedding per input, not per-patch tokens. Use "
                "encoder_image=colpali for fusion=late_interaction."
            )

        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim
        self.multi_vector = False
        # Custom-code checkpoint; trust_remote_code is required to load it.
        self.trust_remote_code: bool = bool(cfg.extra.get("trust_remote_code", True))
        # 8B model — keep the per-call image batch small by default to bound
        # activation memory; tune via ``cfg.extra['batch_size']``.
        self.batch_size: int = int(cfg.extra.get("batch_size", 8))
        # Optional instruction. The model wraps inputs with a default
        # ("Represent the user's input.") when no prompt is given; override via
        # ``cfg.extra['prompt']`` to steer the embedding toward document type.
        prompt = cfg.extra.get("prompt")
        self.prompt: str | None = str(prompt) if prompt else None
        info = resolve_device(device)
        self.device = info.torch_device

        st_kwargs: dict[str, Any] = {
            "device": str(self.device),
            "trust_remote_code": self.trust_remote_code,
            # MRL: truncate the native 4096-d output to the configured width.
            "truncate_dim": cfg.embedding_dim,
        }
        model_kwargs: dict[str, Any] = {}
        # Honor an explicit dtype (e.g. "bfloat16") without forcing one — the
        # checkpoint's native dtype is the sane default.
        torch_dtype = cfg.extra.get("torch_dtype")
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = str(torch_dtype)
        # MPS (Metal) can't run this model's grouped-query attention through the
        # fused/SDPA path: ``mps_matmul`` rejects the 16-query-vs-8-KV-head
        # broadcast ("incompatible dimensions"), aborting the process. Eager
        # attention does the explicit repeat_kv and works. Force it on MPS unless
        # overridden; leave the fast default (SDPA/flash) on CUDA/CPU.
        attn = cfg.extra.get("attn_implementation")
        if attn is None and self.device.type == "mps":
            attn = "eager"
        if attn is not None:
            model_kwargs["attn_implementation"] = str(attn)
        if model_kwargs:
            st_kwargs["model_kwargs"] = model_kwargs
        self.model = SentenceTransformer(cfg.model_id, **st_kwargs)

    @torch.no_grad()
    def encode(self, batch: Sequence[Image.Image]) -> EncoderOutput:
        # The multimodal interface takes one dict per input; an image-only
        # document is ``{"image": <PIL.Image>}``.
        inputs: list[dict[str, Image.Image]] = [{"image": image} for image in batch]
        encode_kwargs: dict[str, Any] = {}
        if self.prompt is not None:
            encode_kwargs["prompt"] = self.prompt
        embeddings = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **encode_kwargs,
        )
        # .float(): the checkpoint runs in bf16, but the encoder contract is
        # float32 (numpy/UMAP downstream can't consume bfloat16).
        return EncoderOutput(pooled=embeddings.detach().cpu().float())


def _factory(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return Qwen3VLEmbeddingEncoder(cfg, device=device)


# One model family, one wrapper: both checkpoints share the same
# sentence-transformers interface and differ only in ``model_id`` (and native
# dim — 4096 for 8B, 2048 for 2B, both MRL-truncatable). Registered under a
# size-explicit name each so callers select via ``encoder_image=...`` and the
# config supplies the matching ``model_id`` / ``embedding_dim``.
register_encoder("qwen3_vl_embedding")(_factory)  # Qwen/Qwen3-VL-Embedding-8B
register_encoder("qwen3_vl_embedding_2b")(_factory)  # Qwen/Qwen3-VL-Embedding-2B
