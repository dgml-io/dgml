"""Utilities — leaf module with no internal dependencies.

Exports the small set of cross-cutting helpers used everywhere else:
deterministic seeding, device auto-selection, and run-id hashing.
"""

from __future__ import annotations

from clustering.utils.device import DeviceInfo, DeviceKind, auto_select_device, resolve_device
from clustering.utils.runid import run_id_for
from clustering.utils.seed import seed_everything

__all__ = [
    "DeviceInfo",
    "DeviceKind",
    "auto_select_device",
    "resolve_device",
    "run_id_for",
    "seed_everything",
]
