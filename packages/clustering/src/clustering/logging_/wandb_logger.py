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

"""Weights & Biases logger adapter.

Lazy-imports :mod:`wandb` so the framework doesn't require it as a hard
dep. Raises from the factory when the package is missing or credentials
are unset; :func:`build_logger` catches and falls back to
:class:`NoopLogger` with a warning.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

from clustering.logging_.base import Logger

WandbMode = Literal["online", "offline", "disabled", "shared"]
_VALID_WANDB_MODES: frozenset[str] = frozenset(("online", "offline", "disabled", "shared"))


class WandbLogger(Logger):
    name = "wandb"

    def __init__(
        self,
        *,
        project: str,
        run_id: str,
        entity: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install the 'logging' extra: `uv sync --extra logging`."
            ) from exc

        mode_env = os.environ.get("WANDB_MODE", "online")
        if mode_env not in _VALID_WANDB_MODES:
            raise ValueError(f"WANDB_MODE={mode_env!r} is not one of {sorted(_VALID_WANDB_MODES)}.")
        mode = cast(WandbMode, mode_env)
        if mode != "disabled" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY not set. Either log in via `wandb login`, set "
                "WANDB_API_KEY in your .env, or set WANDB_MODE=disabled."
            )

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_id,
            tags=tags or [],
            mode=mode,
            reinit=True,
        )
        self._wandb = wandb

    def log_params(self, params: dict[str, Any]) -> None:
        # wandb.config.update is untyped in wandb's stubs.
        self._run.config.update(params, allow_val_change=True)  # type: ignore[no-untyped-call]

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        self._run.log({k: float(v) for k, v in metrics.items()}, step=step)

    def log_tags(self, tags: list[str]) -> None:
        # W&B exposes tags as a settable list on the run.
        existing = list(getattr(self._run, "tags", []) or [])
        merged = list(dict.fromkeys(existing + tags))
        self._run.tags = merged

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        artifact = self._wandb.Artifact(name=name or path.stem, type="result")
        artifact.add_file(str(path))
        self._run.log_artifact(artifact)

    def close(self) -> None:
        self._run.finish()
