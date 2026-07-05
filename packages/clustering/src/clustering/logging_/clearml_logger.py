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

"""ClearML logger adapter.

Lazy-imports :mod:`clearml` so the framework doesn't require it as a hard
dep. Raises :class:`ImportError` / :class:`RuntimeError` from the factory
when the package is missing or credentials are unset; :func:`build_logger`
catches and falls back to :class:`NoopLogger` with a warning.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from clustering.logging_.base import Logger


class ClearMLLogger(Logger):
    """Wraps a single :class:`clearml.Task`."""

    name = "clearml"

    def __init__(
        self,
        *,
        project: str,
        run_id: str,
        tags: list[str] | None = None,
    ) -> None:
        try:
            from clearml import Task
        except ImportError as exc:
            raise ImportError(
                "clearml is not installed. Install the 'logging' extra: `uv sync --extra logging`."
            ) from exc

        if not os.environ.get("CLEARML_API_ACCESS_KEY"):
            raise RuntimeError(
                "ClearML credentials not found. Set CLEARML_API_ACCESS_KEY / "
                "CLEARML_API_SECRET_KEY in your .env, or use a clearml.conf."
            )

        self._task: Any = Task.init(
            project_name=project,
            task_name=run_id,
            tags=tags or [],
            auto_connect_frameworks=False,
            reuse_last_task_id=False,
            output_uri=False,
        )

    def log_params(self, params: dict[str, Any]) -> None:
        self._task.connect(params)

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        logger = self._task.get_logger()
        s = step if step is not None else 0
        for key, value in metrics.items():
            logger.report_scalar(title=key, series=key, value=float(value), iteration=s)

    def log_tags(self, tags: list[str]) -> None:
        self._task.add_tags(tags)

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        self._task.upload_artifact(name or path.name, artifact_object=str(path))

    def close(self) -> None:
        self._task.close()
