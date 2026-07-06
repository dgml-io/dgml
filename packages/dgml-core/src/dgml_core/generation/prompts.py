"""Prompt text for the generation pipeline, loaded from ``resources/prompts.yaml``.

Keeping every prompt in one YAML file — rather than inline in the Python
modules — makes the wording easy to read, diff, and tune without touching code.
Use :func:`get` to fetch a prompt by name.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from typing import Any

import yaml


@lru_cache(maxsize=1)
def _prompts() -> dict[str, str]:
    resource = files("dgml_core.generation.resources").joinpath("prompts.yaml")
    text = resource.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(text)
    return {str(k): str(v) for k, v in data.items()}


def get(name: str) -> str:
    """Return the named prompt. Raises ``KeyError`` if it is not defined."""
    try:
        return _prompts()[name]
    except KeyError:
        raise KeyError(f"unknown prompt {name!r}; defined: {sorted(_prompts())}") from None
