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

"""Tests for the disk-backed :class:`CachingEncoder`.

These run without network access or model weights — they wrap the
deterministic ``dummy`` encoder, whose ``(input → vector)`` map is fixed,
so the cache's hit/miss behaviour and round-trips are exactly assertable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from clustering.config.schema import EncoderConfig
from clustering.encoders import build_encoder, encoder_fingerprint
from clustering.encoders.base import Encoder, EncoderOutput
from clustering.encoders.caching import CachingEncoder
from clustering.encoders.dummy import DummyEncoder
from PIL import Image


class _CountingEncoder(Encoder[Any]):
    """Wrap an encoder and count how many *items* it actually encodes."""

    def __init__(self, inner: Encoder[Any]) -> None:
        self.inner = inner
        self.embedding_dim = inner.embedding_dim
        self.multi_vector = inner.multi_vector
        self.encoded = 0

    def encode(self, batch: Sequence[Any]) -> EncoderOutput:
        items = list(batch)
        self.encoded += len(items)
        return self.inner.encode(items)


def _dummy(dim: int = 16, *, multi_vector: bool = False) -> EncoderConfig:
    return EncoderConfig(name="dummy", embedding_dim=dim, multi_vector=multi_vector)


def test_fingerprint_is_stable_and_config_sensitive() -> None:
    cfg = _dummy()
    assert encoder_fingerprint(cfg) == encoder_fingerprint(_dummy())
    # Any output-determining field change → a different namespace.
    assert encoder_fingerprint(cfg) != encoder_fingerprint(_dummy(dim=32))
    assert encoder_fingerprint(cfg) != encoder_fingerprint(
        EncoderConfig(name="dummy", embedding_dim=16, extra={"x": 1})
    )
    # Human-readable: prefixed with the encoder name.
    assert encoder_fingerprint(cfg).startswith("dummy-")


def test_cached_output_matches_uncached(tmp_path: Path) -> None:
    cfg = _dummy()
    direct = build_encoder(cfg)
    cached = build_encoder(cfg, cache_dir=tmp_path)
    texts = ["alpha", "beta", "gamma"]

    expected = direct.encode(texts)
    # First call populates the cache; second is served entirely from disk.
    first = cached.encode(texts)
    second = cached.encode(texts)

    assert torch.equal(first.pooled, expected.pooled)
    assert torch.equal(second.pooled, expected.pooled)


def test_second_call_is_a_full_cache_hit(tmp_path: Path) -> None:
    fp = encoder_fingerprint(_dummy())
    counter = _CountingEncoder(DummyEncoder(_dummy()))
    cached = CachingEncoder(counter, tmp_path, fp)

    cached.encode(["a", "b", "c"])
    assert counter.encoded == 3  # cold cache → all three encoded
    cached.encode(["a", "b", "c"])
    assert counter.encoded == 3  # warm cache → nothing re-encoded


def test_partial_overlap_only_encodes_misses(tmp_path: Path) -> None:
    fp = encoder_fingerprint(_dummy())
    counter = _CountingEncoder(DummyEncoder(_dummy()))
    cached = CachingEncoder(counter, tmp_path, fp)

    cached.encode(["a", "b"])
    assert counter.encoded == 2
    # "b" is cached; only "c" should hit the inner encoder.
    out = cached.encode(["b", "c"])
    assert counter.encoded == 3

    direct = DummyEncoder(_dummy()).encode(["b", "c"])
    assert torch.equal(out.pooled, direct.pooled)


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    fp = encoder_fingerprint(_dummy())
    first = CachingEncoder(_CountingEncoder(DummyEncoder(_dummy())), tmp_path, fp)
    first.encode(["x", "y"])

    counter = _CountingEncoder(DummyEncoder(_dummy()))
    second = CachingEncoder(counter, tmp_path, fp)
    second.encode(["x", "y"])
    assert counter.encoded == 0  # served from the on-disk cache the first run wrote


def test_distinct_fingerprints_do_not_collide(tmp_path: Path) -> None:
    a = CachingEncoder(_CountingEncoder(DummyEncoder(_dummy(16))), tmp_path, "dummy-aaaa")
    b_inner = _CountingEncoder(DummyEncoder(_dummy(16)))
    b = CachingEncoder(b_inner, tmp_path, "dummy-bbbb")

    a.encode(["same-input"])
    b.encode(["same-input"])
    assert b_inner.encoded == 1  # different namespace → no cross-talk


def test_multi_vector_tokens_round_trip(tmp_path: Path) -> None:
    cfg = _dummy(multi_vector=True)
    fp = encoder_fingerprint(cfg)
    counter = _CountingEncoder(DummyEncoder(cfg))
    cached = CachingEncoder(counter, tmp_path, fp)

    cold = cached.encode(["doc"])
    warm = cached.encode(["doc"])
    assert counter.encoded == 1
    assert cold.tokens is not None and warm.tokens is not None
    assert torch.equal(cold.tokens, warm.tokens)
    assert torch.equal(cold.pooled, warm.pooled)


def test_image_inputs_are_content_keyed(tmp_path: Path) -> None:
    fp = encoder_fingerprint(_dummy())
    counter = _CountingEncoder(DummyEncoder(_dummy()))
    cached = CachingEncoder(counter, tmp_path, fp)

    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    same = Image.new("RGB", (8, 8), color=(10, 20, 30))
    other = Image.new("RGB", (8, 8), color=(40, 50, 60))

    cached.encode([img])
    assert counter.encoded == 1
    cached.encode([same])  # identical pixels → cache hit
    assert counter.encoded == 1
    cached.encode([other])  # different pixels → miss
    assert counter.encoded == 2
