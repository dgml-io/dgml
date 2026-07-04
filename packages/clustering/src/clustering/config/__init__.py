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
