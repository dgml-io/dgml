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

"""Prompt text for the generation pipeline, loaded from ``resources/prompts.yaml``.

Keeping every prompt in one YAML file — rather than inline in the Python
modules — makes the wording easy to read, diff, and tune without touching code.
Use :func:`get` to fetch a prompt by name.

**Variants.** ``prompts.yaml`` may carry an optional ``variants`` block holding
named, partial overrides::

    variants:
      fewshot:
        label_system: |
          ...alternative wording...

Selecting one with ``$DGML_PROMPT_VARIANT`` swaps just those prompts, leaving
the rest at baseline. This exists so prompt wording can be A/B-tested as a
measured axis (the same way the model is) instead of being changed blind: the
benchmark harness runs the same corpus under each variant and compares.

Deliberately strict — an unknown variant name, or a variant that overrides a
prompt that does not exist, raises rather than silently falling back. A run
that quietly ignored the variant it was asked for yields results that look
valid and are not.
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.resources import files
from typing import Any

import yaml

BASELINE = "baseline"
_VARIANT_ENV = "DGML_PROMPT_VARIANT"
_VARIANTS_KEY = "variants"


@lru_cache(maxsize=1)
def _raw() -> dict[str, Any]:
    resource = files("dgml_core.generation.resources").joinpath("prompts.yaml")
    data: dict[str, Any] = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return data


@lru_cache(maxsize=1)
def _prompts() -> dict[str, str]:
    """The baseline prompts — every top-level key except ``variants``."""
    return {str(k): str(v) for k, v in _raw().items() if k != _VARIANTS_KEY}


@lru_cache(maxsize=1)
def _variants() -> dict[str, dict[str, str]]:
    """``{variant name: {prompt name: text}}`` from the optional ``variants`` block.

    A variant overrides only the prompts it names; everything else falls back
    to the baseline, so a variant stays a small diff rather than a fork.
    """
    raw = _raw().get(_VARIANTS_KEY) or {}
    out: dict[str, dict[str, str]] = {}
    for variant, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"prompt variant {variant!r} must be a mapping of prompt -> text")
        unknown = sorted(set(body) - set(_prompts()))
        if unknown:
            # A typo here would silently do nothing, which is the failure mode
            # this whole mechanism exists to avoid. Fail at load instead.
            raise ValueError(
                f"prompt variant {variant!r} overrides unknown prompt(s): {unknown}"
            )
        out[str(variant)] = {str(k): str(v) for k, v in body.items()}
    return out


def available_variants() -> list[str]:
    """Every selectable variant name, ``baseline`` first."""
    return [BASELINE, *sorted(_variants())]


def active_variant() -> str:
    """The variant selected by ``$DGML_PROMPT_VARIANT`` (default ``baseline``).

    Validated here rather than at use time: an unrecognised name must be a hard
    error, never a silent fall back to baseline — a run that quietly ignored the
    variant it was asked for produces results that look valid and are not.
    """
    name = os.environ.get(_VARIANT_ENV, "").strip() or BASELINE
    if name != BASELINE and name not in _variants():
        raise ValueError(
            f"unknown prompt variant {name!r} from ${_VARIANT_ENV}; "
            f"available: {available_variants()}"
        )
    return name


def describe(variant: str | None = None) -> str:
    """One-line provenance string: which variant is live and what it changes."""
    name = variant or active_variant()
    if name == BASELINE:
        return "prompts: baseline"
    overrides = sorted(_variants().get(name, {}))
    return f"prompts: variant={name} overrides={overrides}"


def get(name: str, *, variant: str | None = None) -> str:
    """Return the named prompt, honouring the active variant's overrides."""
    selected = variant if variant is not None else active_variant()
    if selected != BASELINE:
        override = _variants().get(selected, {}).get(name)
        if override is not None:
            return override
    try:
        return _prompts()[name]
    except KeyError:
        raise KeyError(f"unknown prompt {name!r}; defined: {sorted(_prompts())}") from None
