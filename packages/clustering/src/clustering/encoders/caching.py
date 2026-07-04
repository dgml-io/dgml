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

"""Disk-backed embedding cache — a transparent :class:`Encoder` wrapper.

The encoder forward pass is the only step in the pipeline that is both
expensive *and* deterministic: encoders run in ``eval()`` under
``torch.no_grad()``, so a given (encoder, input) pair always yields the
same vector. Everything downstream (fusion, manifold projection,
clustering) depends on per-run config / training and is deliberately left
uncached.

:class:`CachingEncoder` wraps any encoder and persists each
:class:`EncoderOutput` to ``<cache_dir>/<fingerprint>/<input_hash>.pt``:

* **fingerprint** — a stable digest of the :class:`EncoderConfig`. Any
  field that changes the output vector (``name``, ``model_id``,
  ``embedding_dim``, ``multi_vector``, ``doc_prefix``, ``extra`` …) is in
  the config, so a config change lands in a fresh namespace instead of
  serving stale vectors.
* **input hash** — a content hash of each item (text bytes / image
  pixels). Content-keying means re-ingests and renames reuse vectors, and
  identical pages dedupe across documents.

Caching is opt-in: it engages only when :func:`build_encoder` is given a
``cache_dir`` (driven by ``Config.cache_dir``). The wrapper is otherwise
invisible — it implements the same :class:`Encoder` ABC and returns the
same batched :class:`EncoderOutput`.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from clustering.config.schema import EncoderConfig
from clustering.encoders.base import Encoder, EncoderOutput


def encoder_fingerprint(cfg: EncoderConfig) -> str:
    """Stable, output-determining digest of an encoder config.

    Built from the canonical JSON dump of the (frozen) config so two
    configs that differ in any field — even ``extra`` — get distinct
    cache namespaces. Prefixed with the encoder name for human-readable
    cache directories.
    """
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{cfg.name}-{digest}"


def _content_hash(item: Any) -> str:
    """Content hash of a single encode input (text or image)."""
    h = hashlib.sha256()
    if isinstance(item, str):
        h.update(b"str\0")
        h.update(item.encode("utf-8"))
    elif isinstance(item, Image.Image):
        # Pixels + geometry + mode fully determine what the vision tower sees.
        h.update(b"img\0")
        h.update(f"{item.mode}|{item.size}\0".encode())
        h.update(item.tobytes())
    else:  # pragma: no cover — defensive; encoders only take str | Image.
        h.update(b"repr\0")
        h.update(repr(item).encode("utf-8"))
    return h.hexdigest()


class CachingEncoder(Encoder[Any]):
    """Wrap ``inner`` so identical (encoder, input) pairs encode once.

    Per-item granularity: a batch is served from cache where it can be and
    only the misses are forwarded to ``inner``, so overlapping batches
    across runs share work. Cached tensors are stored on CPU (encoders
    already return ``.cpu()`` outputs) and reloaded with
    ``map_location="cpu"``.
    """

    def __init__(self, inner: Encoder[Any], cache_dir: Path, fingerprint: str) -> None:
        self.inner = inner
        self.embedding_dim = inner.embedding_dim
        self.multi_vector = inner.multi_vector
        self._dir = Path(cache_dir) / fingerprint
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.pt"

    def _load(self, key: str) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            blob = torch.load(path, map_location="cpu")
        except Exception:  # pragma: no cover — corrupt/partial file → recompute
            return None
        return blob["pooled"], blob.get("tokens")

    def _store(self, key: str, pooled: torch.Tensor, tokens: torch.Tensor | None) -> None:
        blob: dict[str, torch.Tensor] = {"pooled": pooled.contiguous()}
        if tokens is not None:
            blob["tokens"] = tokens.contiguous()
        # Write to a temp file in the same dir + atomic rename so a crash or
        # concurrent reader never sees a half-written cache entry.
        fd = tempfile.NamedTemporaryFile(dir=self._dir, suffix=".tmp", delete=False)
        try:
            with fd:
                torch.save(blob, fd)
            Path(fd.name).replace(self._path(key))
        except BaseException:
            Path(fd.name).unlink(missing_ok=True)
            raise

    def encode(self, batch: Sequence[Any]) -> EncoderOutput:
        items = list(batch)
        if not items:
            return self.inner.encode(items)

        keys = [_content_hash(x) for x in items]
        pooled: list[torch.Tensor | None] = [None] * len(items)
        tokens: list[torch.Tensor | None] = [None] * len(items)
        missing: list[int] = []
        for i, key in enumerate(keys):
            hit = self._load(key)
            if hit is None:
                missing.append(i)
            else:
                pooled[i], tokens[i] = hit

        if missing:
            out = self.inner.encode([items[i] for i in missing])
            for j, i in enumerate(missing):
                p = out.pooled[j]
                t = out.tokens[j] if out.tokens is not None else None
                self._store(keys[i], p, t)
                pooled[i], tokens[i] = p, t

        stacked_pooled = torch.stack([p for p in pooled if p is not None])
        stacked_tokens = (
            torch.stack([t for t in tokens if t is not None]) if self.multi_vector else None
        )
        return EncoderOutput(pooled=stacked_pooled, tokens=stacked_tokens)
