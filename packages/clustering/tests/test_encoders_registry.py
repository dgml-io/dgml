"""Registration + construction-guard tests for the SigLIP / Qwen-VL /
Qwen3-VL-Embedding encoders.

These run without network access or model weights: every assertion fires
before any ``from_pretrained`` call (registry lookup, schema validation, and
the ``model_id`` / ``multi_vector`` guards in ``__init__``).
"""

from __future__ import annotations

import pytest
from clustering.config.schema import EncoderConfig
from clustering.encoders import build_encoder, registered_encoders


@pytest.mark.parametrize(
    "name", ["siglip", "qwen_vl", "qwen3_vl_embedding", "qwen3_vl_embedding_2b"]
)
def test_encoder_is_registered(name: str) -> None:
    assert name in registered_encoders()


@pytest.mark.parametrize(
    "name", ["siglip", "qwen_vl", "qwen3_vl_embedding", "qwen3_vl_embedding_2b"]
)
def test_schema_accepts_name(name: str) -> None:
    cfg = EncoderConfig(name=name, model_id="some/model")  # type: ignore[arg-type]
    assert cfg.name == name


@pytest.mark.parametrize(
    "name", ["siglip", "qwen_vl", "qwen3_vl_embedding", "qwen3_vl_embedding_2b"]
)
def test_missing_model_id_raises(name: str) -> None:
    cfg = EncoderConfig(name=name, model_id=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires a model_id"):
        build_encoder(cfg, device="cpu")


def test_qwen_vl_rejects_multi_vector() -> None:
    cfg = EncoderConfig(name="qwen_vl", model_id="Qwen/Qwen2.5-VL-3B-Instruct", multi_vector=True)
    with pytest.raises(ValueError, match="single-vector only"):
        build_encoder(cfg, device="cpu")


@pytest.mark.parametrize(
    ("name", "model_id"),
    [
        ("qwen3_vl_embedding", "Qwen/Qwen3-VL-Embedding-8B"),
        ("qwen3_vl_embedding_2b", "Qwen/Qwen3-VL-Embedding-2B"),
    ],
)
def test_qwen3_vl_embedding_rejects_multi_vector(name: str, model_id: str) -> None:
    cfg = EncoderConfig(name=name, model_id=model_id, multi_vector=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="single-vector only"):
        build_encoder(cfg, device="cpu")
