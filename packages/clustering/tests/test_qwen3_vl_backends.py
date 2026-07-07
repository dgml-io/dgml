"""Backend + input-handling tests for the Qwen3-VL-Embedding encoder.

These run offline: the pure request/response/​input builders never touch the
network, and the ``server`` backend is exercised with a monkeypatched embed
call so no server (and no model weights) are needed. The ``local`` backend's
construction guards fire before any ``from_pretrained``.
"""

from __future__ import annotations

import pytest
import torch
from clustering.config.schema import EncoderConfig
from clustering.encoders import build_encoder
from clustering.encoders.qwen3_vl_embedding import (
    Qwen3VLEmbeddingEncoder,
    _to_mm_input,
    build_server_payload,
    parse_server_response,
)


# ── input normalization ───────────────────────────────────────────────────
def test_to_mm_input_text_and_dict() -> None:
    assert _to_mm_input("hello") == {"text": "hello"}
    assert _to_mm_input({"text": "a", "image": "u"}) == {"text": "a", "image": "u"}


def test_to_mm_input_rejects_empty_dict() -> None:
    with pytest.raises(ValueError, match="text' and/or 'image'"):
        _to_mm_input({"foo": "bar"})


def test_to_mm_input_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="Unsupported"):
        _to_mm_input(42)  # type: ignore[arg-type]


# ── server payload / response builders (pure) ─────────────────────────────
def test_build_server_payload_text_only() -> None:
    payload = build_server_payload([{"text": "a"}, {"text": "b"}], model="m", dim=256, prompt=None)
    assert payload["model"] == "m"
    assert payload["input"] == ["a", "b"]
    assert payload["dimensions"] == 256
    assert "instruction" not in payload


def test_build_server_payload_image_becomes_content_list() -> None:
    payload = build_server_payload(
        [{"image": "http://x/y.png", "text": "cap"}],
        model="m",
        dim=None,
        prompt="Represent it.",
    )
    element = payload["input"][0]
    assert element[0]["type"] == "image_url"
    assert element[0]["image_url"]["url"] == "http://x/y.png"
    assert element[1] == {"type": "text", "text": "cap"}
    assert payload["instruction"] == "Represent it."
    assert "dimensions" not in payload


def test_parse_server_response_sorts_by_index() -> None:
    obj = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
    }
    assert parse_server_response(obj) == [[1.0, 0.0], [0.0, 1.0]]


def test_parse_server_response_rejects_malformed() -> None:
    with pytest.raises(ValueError, match="missing 'data'"):
        parse_server_response({"oops": True})


# ── server backend construction + encode (no network) ─────────────────────
def _server_cfg(**extra: object) -> EncoderConfig:
    base: dict[str, object] = {"backend": "server", "base_url": "http://localhost:8000/v1"}
    base.update(extra)
    return EncoderConfig(
        name="qwen3_vl_embedding",
        model_id="Qwen/Qwen3-VL-Embedding-8B",
        embedding_dim=4,
        extra=base,
    )


def test_server_backend_requires_base_url() -> None:
    cfg = EncoderConfig(
        name="qwen3_vl_embedding",
        model_id="Qwen/Qwen3-VL-Embedding-8B",
        extra={"backend": "server"},
    )
    with pytest.raises(ValueError, match="requires extra\\['base_url'\\]"):
        build_encoder(cfg, device="cpu")


def test_server_backend_builds_without_loading_weights() -> None:
    # No sentence_transformers import, no model download — .model stays None.
    enc = build_encoder(_server_cfg(), device="cpu")
    assert isinstance(enc, Qwen3VLEmbeddingEncoder)
    assert enc.model is None
    assert enc._server is not None


def test_server_encode_truncates_and_renormalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    enc = build_encoder(_server_cfg(), device="cpu")
    assert isinstance(enc, Qwen3VLEmbeddingEncoder)

    # Native width 5; embedding_dim is 4 ⇒ encode must return [B, 4], unit-norm.
    def fake_embed(items, *, dim, prompt):  # type: ignore[no-untyped-def]
        return torch.tensor([[3.0, 4.0, 0.0, 0.0, 99.0], [0.0, 0.0, 6.0, 8.0, -1.0]])

    monkeypatch.setattr(enc._server, "embed", fake_embed)
    out = enc.encode(["doc one", "doc two"])
    assert out.pooled.shape == (2, 4)
    norms = out.pooled.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-6)


def test_unknown_backend_raises() -> None:
    cfg = EncoderConfig(
        name="qwen3_vl_embedding",
        model_id="Qwen/Qwen3-VL-Embedding-8B",
        extra={"backend": "banana"},
    )
    with pytest.raises(ValueError, match="Unknown backend"):
        build_encoder(cfg, device="cpu")
