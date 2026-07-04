"""Logger framework — ClearML / W&B / Multi / Noop behind a common ABC.

``build_logger`` reads the resolved :class:`~clustering.config.LoggerConfig`
and returns the configured backend, gracefully falling back to
:class:`NoopLogger` if credentials are missing. This means runs never
crash because an experiment tracker is unreachable.

Module is named ``logging_`` (trailing underscore) to avoid shadowing the
stdlib ``logging`` module.
"""

from __future__ import annotations

from clustering.logging_.base import Logger, NoopLogger
from clustering.logging_.factory import build_logger
from clustering.logging_.multi import MultiLogger

__all__ = [
    "Logger",
    "MultiLogger",
    "NoopLogger",
    "build_logger",
]
