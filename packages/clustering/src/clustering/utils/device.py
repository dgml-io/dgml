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

"""Device selection helper.

The framework auto-selects the best available compute target:

    CUDA → MPS → CPU

This module is import-safe and intentionally cheap: it must not trigger
heavy CUDA initialization at import time. Device probing is lazy and
cached on first call.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch

DeviceKind = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class DeviceInfo:
    """Result of a device-resolution query.

    Attributes:
        kind: One of ``"cuda" | "mps" | "cpu"``.
        torch_device: A concrete ``torch.device`` to hand to tensor / module
            ``.to(...)`` calls.
        name: Human-readable label (GPU name, ``"Apple Silicon (MPS)"`` or
            ``"CPU"``). Logged at run start.
    """

    kind: DeviceKind
    torch_device: torch.device
    name: str

    def __str__(self) -> str:
        return f"{self.kind} ({self.name})"


def _mps_available() -> bool:
    """Return True if Apple-Silicon MPS is both built into torch and usable."""
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is None:
        return False
    is_available = getattr(mps, "is_available", lambda: False)
    return bool(is_available())


@lru_cache(maxsize=1)
def auto_select_device() -> DeviceInfo:
    """Pick the best available torch device.

    Order of preference: ``cuda`` → ``mps`` → ``cpu``. The result is cached
    so repeated calls don't re-query the driver.
    """
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        return DeviceInfo(
            kind="cuda",
            torch_device=torch.device(f"cuda:{idx}"),
            name=torch.cuda.get_device_name(idx),
        )
    if _mps_available():
        return DeviceInfo(
            kind="mps",
            torch_device=torch.device("mps"),
            name="Apple Silicon (MPS)",
        )
    return DeviceInfo(
        kind="cpu",
        torch_device=torch.device("cpu"),
        name="CPU",
    )


def resolve_device(spec: str | None = None) -> DeviceInfo:
    """Resolve a user-provided device string, falling back to auto-select.

    Args:
        spec: One of ``None`` / ``"auto"`` / ``"cuda"`` / ``"cuda:N"`` /
            ``"mps"`` / ``"cpu"``. ``None`` and ``"auto"`` both auto-select.

    Raises:
        RuntimeError: If the user explicitly requests a device that is not
            available on this host. We intentionally fail loud rather than
            silently fall back — silent fallback masks bugs in
            reproducibility-critical runs.
        ValueError: If ``spec`` is not a recognised device string.
    """
    if spec is None or spec == "auto":
        return auto_select_device()

    if spec == "cuda" or spec.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device {spec!r} but CUDA is not available.")
        dev = torch.device(spec)
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        return DeviceInfo(
            kind="cuda",
            torch_device=dev,
            name=torch.cuda.get_device_name(idx),
        )

    if spec == "mps":
        if not _mps_available():
            raise RuntimeError("Requested device 'mps' but MPS is not available.")
        return DeviceInfo(
            kind="mps",
            torch_device=torch.device("mps"),
            name="Apple Silicon (MPS)",
        )

    if spec == "cpu":
        return DeviceInfo(kind="cpu", torch_device=torch.device("cpu"), name="CPU")

    raise ValueError(f"Unknown device spec: {spec!r}")
