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

"""Fan-out logger that broadcasts every call to multiple backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clustering.logging_.base import Logger


class MultiLogger(Logger):
    """Wraps a list of loggers; each call is fanned out to all of them.

    Designed for the "log to both ClearML and W&B" case from the brief.
    Empty / one-element lists are accepted — the latter just acts as a
    pass-through.
    """

    name = "multi"

    def __init__(self, loggers: list[Logger]) -> None:
        self.loggers = list(loggers)

    def log_params(self, params: dict[str, Any]) -> None:
        for lg in self.loggers:
            lg.log_params(params)

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        for lg in self.loggers:
            lg.log_metrics(metrics, step=step)

    def log_tags(self, tags: list[str]) -> None:
        for lg in self.loggers:
            lg.log_tags(tags)

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        for lg in self.loggers:
            lg.log_artifact(path, name=name)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()
