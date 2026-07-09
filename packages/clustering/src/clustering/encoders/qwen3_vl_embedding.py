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

"""Qwen3-VL-Embedding encoder (``Qwen/Qwen3-VL-Embedding-8B`` / ``-2B``).

Unlike :mod:`clustering.encoders.qwen_vl` — which borrows the *vision tower*
of a generative Qwen2/2.5-VL checkpoint and mean-pools its patches — this is a
purpose-built **multimodal embedding** model (base: Qwen3-VL-*-Instruct). It
emits a single, already-normalized vector per input in a *shared text+image
space*, so the same encoder serves both sides of the pipeline:

* as ``encoder_image`` it embeds a page render (``page_1.png``);
* as ``encoder_text`` it embeds the page's OCR text;

and because both land in one space, the fusion layer combines genuinely
comparable vectors. An item is a ``str`` (text), a ``PIL.Image`` (page image),
or — if a caller ever passes one — a ``{"text", "image"}`` dict for a single
mixed input.

**MRL (Matryoshka).** The native embedding (4096-d for 8B, 2048-d for 2B) can
be truncated to any width in ``[64, native]`` and re-normalized. We honor
``cfg.embedding_dim`` as the target width so the rest of the pipeline (fusion /
manifold) gets the dimensionality it's configured for. See
:mod:`clustering.encoders.mrl` for the sweep helper that picks a good width.

**Pluggable backend** (``cfg.extra['backend']``):

* ``"local"`` (default) — load the checkpoint in-process via
  ``sentence_transformers`` and truncate with its ``truncate_dim``. Simplest;
  a GPU is strongly recommended (the 8B in float32 needs ~32 GB just for
  weights, so its native dtype / bf16 is used).
* ``"server"`` — POST to an OpenAI-compatible embeddings endpoint (a vLLM
  ``--task embed`` or SGLang server hosting the same checkpoint). Keeps torch
  and the weights out of this process; MRL truncation is applied client-side
  via :func:`clustering.encoders.mrl.mrl_truncate` so the configured width is
  honored regardless of whether the server did its own truncation.

Single-vector only: the model exposes one pooled embedding, so
``multi_vector=True`` (``fusion=late_interaction``) is rejected — use ColPali
for multi-vector page retrieval.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder
from clustering.encoders.mrl import mrl_truncate
from clustering.utils import resolve_device

# One item into the model: text, a page image, or a single mixed input.
Qwen3VLInput = str | Image.Image | dict[str, Any]


def _to_mm_input(item: Qwen3VLInput) -> dict[str, Any]:
    """Normalize one encode input into the model's ``{"text"?, "image"?}`` dict."""
    if isinstance(item, str):
        return {"text": item}
    if isinstance(item, Image.Image):
        return {"image": item}
    if isinstance(item, dict):
        if not ("text" in item or "image" in item):
            raise ValueError(f"Mixed input dict needs a 'text' and/or 'image' key; got {item!r}.")
        return item
    raise TypeError(f"Unsupported Qwen3-VL input type: {type(item).__name__}.")


def _image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG ``data:`` URL for the server backend."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _server_input_element(item: dict[str, Any]) -> Any:
    """Map a ``{"text"?, "image"?}`` item to one OpenAI ``input`` element.

    Text-only inputs become a bare string (the vanilla embeddings contract);
    anything with an image becomes an OpenAI chat-style content list mixing
    ``image_url`` and ``text`` parts (the multimodal-embeddings convention).
    """
    text = item.get("text")
    image = item.get("image")
    if image is None:
        return text if text is not None else ""
    url = _image_to_data_url(image) if isinstance(image, Image.Image) else str(image)
    content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": url}}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return content


def build_server_payload(
    items: Sequence[dict[str, Any]],
    *,
    model: str,
    dim: int | None,
    prompt: str | None,
) -> dict[str, Any]:
    """Build the JSON body for an OpenAI-compatible ``/v1/embeddings`` request.

    Pure (no I/O) so it can be unit-tested offline. ``dim`` is forwarded as the
    OpenAI ``dimensions`` field *as a hint*; the client still truncates the
    returned vectors defensively, since not every server honors it for a
    multimodal pooling model.
    """
    payload: dict[str, Any] = {
        "model": model,
        "input": [_server_input_element(it) for it in items],
        "encoding_format": "float",
    }
    if dim is not None:
        payload["dimensions"] = dim
    if prompt is not None:
        payload["instruction"] = prompt
    return payload


def parse_server_response(obj: dict[str, Any]) -> list[list[float]]:
    """Extract embeddings from an OpenAI-compatible response, in request order.

    Pure (no I/O). Sorts by ``index`` when present so an out-of-order server
    response still lines up with the inputs.
    """
    data = obj.get("data")
    if not isinstance(data, list):
        raise ValueError(
            f"Malformed embeddings response: missing 'data' list (got keys {list(obj)})."
        )
    rows = sorted(data, key=lambda d: int(d.get("index", 0)))
    return [[float(x) for x in row["embedding"]] for row in rows]


class _ServerBackend:
    """Thin OpenAI-compatible embeddings client (stdlib ``urllib``, no new deps)."""

    def __init__(self, cfg: EncoderConfig) -> None:
        base_url = cfg.extra.get("base_url") or cfg.extra.get("endpoint")
        if not base_url:
            raise ValueError(
                "Qwen3-VL-Embedding server backend requires extra['base_url'] "
                "(e.g. 'http://localhost:8000/v1'). Point it at a vLLM --task embed "
                "or SGLang server hosting the checkpoint."
            )
        base = str(base_url).rstrip("/")
        self.url = base if base.endswith("/embeddings") else f"{base}/embeddings"
        # The name the server registered the model under; defaults to the HF id.
        self.model = str(cfg.extra.get("served_model_name") or cfg.model_id)
        api_key = cfg.extra.get("api_key")
        self.api_key: str | None = str(api_key) if api_key else None
        self.timeout: float = float(cfg.extra.get("timeout", 60.0))

    def embed(
        self, items: Sequence[dict[str, Any]], *, dim: int | None, prompt: str | None
    ) -> torch.Tensor:
        payload = build_server_payload(items, model=self.model, dim=dim, prompt=prompt)
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Qwen3-VL embeddings request to {self.url} failed: {exc}") from exc
        vectors = parse_server_response(obj)
        return torch.tensor(vectors, dtype=torch.float32)


class Qwen3VLEmbeddingEncoder(Encoder[Qwen3VLInput]):
    """Qwen3-VL-Embedding encoder. Single-vector, shared text+image space."""

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
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
        self.backend_name: str = str(cfg.extra.get("backend", "local"))
        # Optional instruction. The model wraps inputs with a default
        # ("Represent the user's input.") when no prompt is given; override via
        # ``cfg.extra['prompt']`` to steer the embedding toward document type.
        prompt = cfg.extra.get("prompt")
        self.prompt: str | None = str(prompt) if prompt else None
        # Per-call batch size; the 8B is memory-hungry, so keep it small.
        self.batch_size: int = int(cfg.extra.get("batch_size", 8))

        if self.backend_name == "server":
            self._server: _ServerBackend | None = _ServerBackend(cfg)
            self.model: Any = None
            return
        if self.backend_name != "local":
            raise ValueError(
                f"Unknown backend {self.backend_name!r} for Qwen3-VL-Embedding; "
                "expected 'local' or 'server'."
            )

        self._server = None
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Add the 'encoders' extra "
                "with: `uv sync --extra encoders`."
            ) from exc
        # Custom-code checkpoint; trust_remote_code is required to load it.
        self.trust_remote_code: bool = bool(cfg.extra.get("trust_remote_code", True))
        info = resolve_device(device)
        self.device = info.torch_device
        st_kwargs: dict[str, Any] = {
            "device": str(self.device),
            "trust_remote_code": self.trust_remote_code,
            # MRL: truncate the native output to the configured width.
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
        # broadcast, aborting the process. Eager attention does the explicit
        # repeat_kv and works. Force it on MPS unless overridden.
        attn = cfg.extra.get("attn_implementation")
        if attn is None and self.device.type == "mps":
            attn = "eager"
        if attn is not None:
            model_kwargs["attn_implementation"] = str(attn)
        if model_kwargs:
            st_kwargs["model_kwargs"] = model_kwargs
        self.model = SentenceTransformer(cfg.model_id, **st_kwargs)

    @torch.no_grad()
    def encode(self, batch: Sequence[Qwen3VLInput]) -> EncoderOutput:
        inputs = [_to_mm_input(item) for item in batch]
        if self._server is not None:
            raw = self._server.embed(inputs, dim=self.embedding_dim, prompt=self.prompt)
            # The server may or may not have truncated; enforce the width + renorm.
            pooled = mrl_truncate(raw, self.embedding_dim)
            return EncoderOutput(pooled=pooled.detach().cpu().float())

        assert self.model is not None  # local backend: model is loaded
        encode_kwargs: dict[str, Any] = {}
        if self.prompt is not None:
            encode_kwargs["prompt"] = self.prompt
        embeddings = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,  # renormalize after truncate_dim
            show_progress_bar=False,
            **encode_kwargs,
        )
        # .float(): the checkpoint runs in bf16, but the encoder contract is
        # float32 (numpy / UMAP downstream can't consume bfloat16).
        return EncoderOutput(pooled=embeddings.detach().cpu().float())


def _factory(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return Qwen3VLEmbeddingEncoder(cfg, device=device)


# One model family, one wrapper: both checkpoints share the same interface and
# differ only in ``model_id`` (and native dim — 4096 for 8B, 2048 for 2B, both
# MRL-truncatable). Registered under a size-explicit name each; the same name
# works as ``encoder_text`` or ``encoder_image`` because the encoder embeds
# both modalities into one shared space.
register_encoder("qwen3_vl_embedding")(_factory)  # Qwen/Qwen3-VL-Embedding-8B
register_encoder("qwen3_vl_embedding_2b")(_factory)  # Qwen/Qwen3-VL-Embedding-2B
