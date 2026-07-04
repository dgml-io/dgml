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

"""Tests for the shared core-prompt loader (:mod:`dgml_core.prompts`)."""

from __future__ import annotations

import pytest
from dgml_core.prompts import get

# Every prompt key that live code fetches by name. If a key is renamed in
# resources/prompts.yaml without updating the caller (or vice versa), this
# list — checked below — is the tripwire.
_REFERENCED_KEYS = [
    "merge_system_prompt",
    "extraction_schema_system",
    "extraction_schema_user_intro_single",
    "extraction_schema_user_intro_multi",
    "extraction_schema_user_body",
    "extraction_values_phase1_system",
    "extraction_values_phase1_user",
    "extraction_values_phase3_system",
    "extraction_values_phase3_user",
]


@pytest.mark.parametrize("key", _REFERENCED_KEYS)
def test_referenced_prompts_are_defined_and_non_empty(key: str) -> None:
    assert get(key).strip()


def test_unknown_prompt_raises_keyerror_listing_defined_names() -> None:
    with pytest.raises(KeyError) as exc:
        get("does_not_exist")
    # The message lists the available names so a typo is easy to fix.
    assert "does_not_exist" in str(exc.value)
    assert "merge_system_prompt" in str(exc.value)


def test_templated_prompts_accept_their_placeholders() -> None:
    """The .format()-templated prompts interpolate cleanly — no stray braces
    that would raise KeyError/IndexError, and the placeholder is substituted."""
    assert "5" in get("extraction_schema_user_intro_multi").format(n_files=5)
    assert "MY_SCHEMA" in get("extraction_values_phase1_user").format(schema="MY_SCHEMA")
    filled = get("extraction_values_phase3_user").format(
        page_number=7,
        ocr_words="[]",
        known_locations="(none)",
        needs_locating="- id: x",
    )
    assert "page 7" in filled and "- id: x" in filled


def test_verbatim_prompts_keep_literal_braces() -> None:
    """The schema body and phase-1 system prompt carry literal JSON braces and
    are used without str.format — the braces must survive intact."""
    assert '{"$ref": "#/definitions/grounded_field"}' in get("extraction_schema_user_body")
    assert '"kind": "table"' in get("extraction_values_phase1_system")
