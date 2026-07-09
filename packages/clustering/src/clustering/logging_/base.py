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

"""Logger ABC + the always-available :class:`NoopLogger`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Logger(ABC):
    """Common interface for experiment-tracker backends.

    All methods are no-throw — backends should fail loudly *at construction*
    if their credentials are missing (so we fall back to noop), but never
    crash a run because a metric couldn't be flushed.
    """

    name: str

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log a flat dict of hyperparameters (config snapshot)."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        """Log a dict of scalar metrics (optionally at ``step``)."""

    @abstractmethod
    def log_tags(self, tags: list[str]) -> None:
        """Attach run-level tags (e.g. ``["scenario:s1", "fusion:gated"]``)."""

    @abstractmethod
    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """Upload an on-disk artifact (parquet, plot, …)."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release any backend resources."""

    # Context-manager sugar.
    def __enter__(self) -> Logger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class NoopLogger(Logger):
    """No-op logger. The default fallback when no backend is configured."""

    name = "none"

    def log_params(self, params: dict[str, Any]) -> None:
        del params

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        del metrics, step

    def log_tags(self, tags: list[str]) -> None:
        del tags

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        del path, name

    def close(self) -> None:
        return None
