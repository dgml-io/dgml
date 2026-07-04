"""Config system — Hydra config groups validated through pydantic schemas.

The Hydra config groups under ``configs/`` produce ``OmegaConf`` ``DictConfig``
objects at run time. ``resolve()`` validates them into typed pydantic models
so the rest of the code gets type safety + IDE support, and computes the
deterministic ``run_id`` used by the loggers and parquet artifacts.
"""

from __future__ import annotations

from clustering.config.resolve import resolve
from clustering.config.schema import (
    Config,
    CorpusConfig,
    EncoderConfig,
    FusionConfig,
    LoggerConfig,
    ManifoldConfig,
    ScenarioConfig,
    TrainingConfig,
)

__all__ = [
    "Config",
    "CorpusConfig",
    "EncoderConfig",
    "FusionConfig",
    "LoggerConfig",
    "ManifoldConfig",
    "ScenarioConfig",
    "TrainingConfig",
    "resolve",
]
