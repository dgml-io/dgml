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

"""Qwen-VL image encoder (Qwen2-VL / Qwen2.5-VL).

Qwen-VL is a generative vision-language model; we use only its vision tower
to turn a page image into an embedding. Qwen-VL processes images at
*dynamic* resolution, so different images yield different numbers of visual
tokens. Each image is therefore encoded independently and its per-patch
visual tokens are mean-pooled into a single ``pooled`` vector — the signal
clustering consumes.

Because the per-image token grids are ragged (no common ``N``), this encoder
is single-vector only: ``multi_vector=True`` is rejected.

Notes:
    The Qwen-VL transformers classes have been renamed across releases: the
    2.5 line adds ``Qwen2_5_VLForConditionalGeneration`` alongside the 2.x
    ``Qwen2VLForConditionalGeneration``. The class is chosen from the
    checkpoint's own ``model_type`` (see :func:`_load_qwen_vl_class`) so a
    2.x checkpoint loads into the 2.x class and a 2.5 checkpoint into the 2.5
    class — mixing them yields a vision-tower size mismatch. The rest of the
    framework consumes only the common :class:`EncoderOutput` and is
    unaffected.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from PIL import Image

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput, register_encoder
from clustering.utils import resolve_device


def _load_qwen_vl_class(model_id: str) -> Any:
    """Return the Qwen-VL conditional-generation class matching ``model_id``.

    The class is selected from the *checkpoint's own* ``model_type`` rather
    than by import availability: a Qwen2-VL checkpoint (e.g.
    ``Qwen2-VL-2B-Instruct``) must load into the 2.x class and a Qwen2.5-VL
    checkpoint into the 2.5 class. Forcing the 2.5 class onto a 2.x checkpoint
    triggers a vision-tower ``size mismatch`` because the hidden dims differ.
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError(
            "transformers with Qwen-VL support is not installed. Add the "
            "'encoders' extra with: `uv sync --extra encoders` (needs "
            "transformers>=4.49 for Qwen2.5-VL)."
        ) from exc

    config = AutoConfig.from_pretrained(model_id)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen2_5_vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                f"{model_id!r} is a Qwen2.5-VL checkpoint but this transformers "
                "version lacks Qwen2_5_VLForConditionalGeneration. Upgrade with "
                "`uv sync --extra encoders` (needs transformers>=4.49)."
            ) from exc

    try:
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "transformers with Qwen-VL support is not installed. Add the "
            "'encoders' extra with: `uv sync --extra encoders` (needs "
            "transformers>=4.49 for Qwen2.5-VL)."
        ) from exc


class QwenVLEncoder(Encoder[Image.Image]):
    """Qwen-VL vision-tower image encoder. Single-vector (mean-pooled patches)."""

    def __init__(self, cfg: EncoderConfig, *, device: str = "auto") -> None:
        try:
            from transformers import AutoImageProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Add the 'encoders' extra with: "
                "`uv sync --extra encoders`."
            ) from exc
        if cfg.model_id is None:
            raise ValueError(f"Encoder {cfg.name!r} requires a model_id.")
        if cfg.multi_vector:
            raise ValueError(
                "Qwen-VL is single-vector only: its dynamic-resolution visual "
                "tokens form ragged per-image grids that cannot be stacked into "
                "a uniform [B, N, D] tensor."
            )

        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim
        self.multi_vector = False
        info = resolve_device(device)
        self.device = info.torch_device

        # Cap the dynamic-resolution pixel budget. Qwen-VL emits one visual
        # token per 28x28 px block (patch 14, spatial-merge 2), and attention
        # over the patch grid is O(N^2): an uncapped high-res page image yields
        # tens of thousands of tokens and tries to allocate a multi-hundred-GB
        # attention buffer. Bounding ``max_pixels`` to ~1280 tokens keeps the
        # vision tower tractable on CPU while preserving enough detail for a
        # pooled page embedding. Override via ``encoder.extra['max_pixels']``.
        token_px = 28 * 28
        max_pixels = int(cfg.extra.get("max_pixels", 1280 * token_px))
        min_pixels = int(cfg.extra.get("min_pixels", 4 * token_px))

        model_cls = _load_qwen_vl_class(cfg.model_id)
        self.processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            cfg.model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.model: Any = (
            model_cls.from_pretrained(cfg.model_id, torch_dtype=torch.float32)
            .to(self.device)
            .eval()
        )
        # The vision tower lives at ``.visual`` on transformers 4.x but moved
        # under ``.model.visual`` in 5.x (the ForConditionalGeneration backbone
        # was nested). Resolve once so encode() works on either.
        self._visual: Any = getattr(self.model, "visual", None)
        if self._visual is None:
            self._visual = self.model.model.visual

    @torch.no_grad()
    def encode(self, batch: Sequence[Image.Image]) -> EncoderOutput:
        # Encode each image on its own: dynamic resolution → variable token
        # counts, so a single batched vision-tower call can't be split back
        # into per-image vectors without the grid metadata anyway.
        pooled_rows: list[torch.Tensor] = []
        for image in batch:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            grid_thw = inputs["image_grid_thw"].to(self.device)
            vis_out = self._visual(pixel_values, grid_thw=grid_thw)
            # 4.x returned the merged-patch tensor [N_patches, D] directly; 5.x
            # wraps it in a BaseModelOutputWithPooling whose ``pooler_output`` is
            # that same merged-patch sequence (the merger output, out_hidden_size
            # dim) — not last_hidden_state, which is the pre-merger tokens.
            embeds = getattr(vis_out, "pooler_output", vis_out)  # [N_patches, D]
            pooled_rows.append(embeds.mean(dim=0))
        pooled = torch.stack(pooled_rows, dim=0)  # [B, D]
        return EncoderOutput(pooled=pooled.detach().cpu())


@register_encoder("qwen_vl")
def _factory(cfg: EncoderConfig, *, device: str = "auto") -> Encoder[Any]:
    return QwenVLEncoder(cfg, device=device)
