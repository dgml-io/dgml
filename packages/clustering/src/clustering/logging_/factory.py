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

"""Factory: build the configured logger, with graceful no-op fallback.

Resolution rules (in order):

1. ``cfg.name == "none"`` → :class:`NoopLogger`.
2. ``cfg.name == "clearml"`` → :class:`ClearMLLogger`, or
   :class:`NoopLogger` if clearml is not installed / no credentials.
3. ``cfg.name == "wandb"`` → :class:`WandbLogger`, or
   :class:`NoopLogger` if wandb is not installed / no credentials.
4. ``cfg.name == "multi"`` → :class:`MultiLogger` of every backend that
   could be successfully constructed. If both fall back, returns
   :class:`NoopLogger`.

Construction failures are reported via the stdlib ``logging`` module (not
raised), so a missing credential never crashes a research run.
"""

from __future__ import annotations

import logging
from typing import assert_never

from clustering.config.schema import LoggerConfig, LoggerName
from clustering.logging_.base import Logger, NoopLogger
from clustering.logging_.multi import MultiLogger

_log = logging.getLogger(__name__)


def _try_clearml(cfg: LoggerConfig, *, run_id: str) -> Logger | None:
    try:
        from clustering.logging_.clearml_logger import ClearMLLogger

        return ClearMLLogger(project=cfg.project, run_id=run_id, tags=list(cfg.tags))
    except Exception as exc:
        _log.warning("ClearML logger unavailable, falling back to noop: %s", exc)
        return None


def _try_wandb(cfg: LoggerConfig, *, run_id: str) -> Logger | None:
    try:
        from clustering.logging_.wandb_logger import WandbLogger

        return WandbLogger(
            project=cfg.project,
            run_id=run_id,
            entity=cfg.entity,
            tags=list(cfg.tags),
        )
    except Exception as exc:
        _log.warning("Weights & Biases logger unavailable, falling back to noop: %s", exc)
        return None


def build_logger(cfg: LoggerConfig, *, run_id: str) -> Logger:
    """Build the configured logger, falling back to noop on any error.

    Raises:
        ValueError: If ``cfg.name`` is not a :class:`LoggerName` (possible only
            for a config built with ``model_construct``, which skips validation).
    """
    try:
        LoggerName(cfg.name)
    except ValueError:
        raise ValueError(f"Unknown logger name: {cfg.name!r}") from None
    if cfg.name == LoggerName.NONE:
        return NoopLogger()

    if cfg.name == LoggerName.CLEARML:
        return _try_clearml(cfg, run_id=run_id) or NoopLogger()

    if cfg.name == LoggerName.WANDB:
        return _try_wandb(cfg, run_id=run_id) or NoopLogger()

    if cfg.name == LoggerName.MULTI:
        backends: list[Logger] = []
        c = _try_clearml(cfg, run_id=run_id)
        if c is not None:
            backends.append(c)
        w = _try_wandb(cfg, run_id=run_id)
        if w is not None:
            backends.append(w)
        if not backends:
            return NoopLogger()
        return MultiLogger(backends)

    assert_never(cfg.name)
