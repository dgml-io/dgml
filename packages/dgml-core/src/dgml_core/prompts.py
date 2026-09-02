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

"""Prompt text for core (non-generation) LLM features, loaded from ``resources/prompts.yaml``.

Keeping every prompt in one YAML file — rather than inline in the Python
modules — makes the wording easy to read, diff, and tune without touching code.
This is the same pattern the generation pipeline uses
(:mod:`dgml_core.generation.prompts`); the two YAMLs are kept separate so each
subsystem's prompts live next to the code that reads them.

Fetch prompts through :func:`get` with a :class:`PromptKey` member rather than a
bare string, so a mistyped name is a resolution error at the call site instead
of a ``KeyError`` reached only when that code path runs. ``PromptKey`` is a
``StrEnum``, so members are usable anywhere a ``str`` is; the test suite asserts
the enum and the YAML keys stay in one-to-one correspondence, which is what
catches a member whose prompt was renamed or removed.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from importlib.resources import files
from typing import Any

import yaml


class PromptKey(StrEnum):
    """Every prompt defined in ``resources/prompts.yaml``."""

    # Hybrid text extraction
    MERGE_SYSTEM = "merge_system_prompt"

    # Extraction: schema generation
    SCHEMA_SYSTEM = "extraction_schema_system"
    SCHEMA_USER_INTRO_SINGLE = "extraction_schema_user_intro_single"
    SCHEMA_USER_INTRO_MULTI = "extraction_schema_user_intro_multi"
    SCHEMA_USER_BODY = "extraction_schema_user_body"

    # Extraction: value extraction, phase 1 (values + page numbers)
    VALUES_PHASE1_SYSTEM = "extraction_values_phase1_system"
    VALUES_PHASE1_USER = "extraction_values_phase1_user"
    VALUES_PHASE1_GUIDANCE = "extraction_values_phase1_guidance"
    VALUES_PHASE1_RETRY_CHUNKED = "extraction_values_phase1_retry_chunked"

    # Extraction: value extraction, phase 3 (attach bounding boxes)
    VALUES_PHASE3_SYSTEM = "extraction_values_phase3_system"
    VALUES_PHASE3_USER = "extraction_values_phase3_user"

    # LLM clustering
    CLUSTER_GROUPING_INTRO = "cluster_grouping_intro"
    CLUSTER_GROUPING_DOC_MANIFEST = "cluster_grouping_doc_manifest"
    CLUSTER_GROUPING_EXISTING_INTRO = "cluster_grouping_existing_intro"
    CLUSTER_GROUPING_INSTRUCTIONS = "cluster_grouping_instructions"

    # Shared LLM dispatch
    CONTINUE_TRUNCATED = "llm_continue_truncated"


@lru_cache(maxsize=1)
def _prompts() -> dict[str, str]:
    resource = files("dgml_core.resources").joinpath("prompts.yaml")
    text = resource.read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(text)
    return {str(k): str(v) for k, v in data.items()}


def get(name: PromptKey | str) -> str:
    """Return the named prompt. Raises ``KeyError`` if it is not defined.

    Accepts a bare string as well as a :class:`PromptKey` so external callers
    aren't forced to import the enum; first-party code should pass the member.
    """
    try:
        return _prompts()[str(name)]
    except KeyError:
        raise KeyError(f"unknown prompt {str(name)!r}; defined: {sorted(_prompts())}") from None
