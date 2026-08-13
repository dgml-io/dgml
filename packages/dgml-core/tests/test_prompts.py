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
from dgml_core.prompts import PromptKey, _prompts, get


def test_prompt_keys_and_yaml_are_one_to_one() -> None:
    """``PromptKey`` is only typo-protection while it stays in lockstep with the
    YAML: a member whose prompt was renamed still KeyErrors at call time, and a
    prompt with no member is reachable only via the bare-string escape hatch.
    This supersedes hand-maintaining a list of referenced keys — it covers every
    key in both directions."""
    assert {str(key) for key in PromptKey} == set(_prompts())


@pytest.mark.parametrize("key", list(PromptKey), ids=lambda k: str(k))
def test_every_prompt_is_defined_and_non_empty(key: PromptKey) -> None:
    assert get(key).strip()


def test_get_accepts_enum_and_bare_string() -> None:
    """The enum is a StrEnum, so members interoperate with the string form the
    public loader still accepts for external callers."""
    assert get(PromptKey.MERGE_SYSTEM) == get("merge_system_prompt")


def test_unknown_prompt_raises_keyerror_listing_defined_names() -> None:
    with pytest.raises(KeyError) as exc:
        get("does_not_exist")
    # The message lists the available names so a typo is easy to fix.
    assert "does_not_exist" in str(exc.value)
    assert "merge_system_prompt" in str(exc.value)


def test_templated_prompts_accept_their_placeholders() -> None:
    """The .format()-templated prompts interpolate cleanly — no stray braces
    that would raise KeyError/IndexError, and the placeholder is substituted."""
    assert "5" in get(PromptKey.SCHEMA_USER_INTRO_MULTI).format(n_files=5)
    assert "MY_SCHEMA" in get(PromptKey.VALUES_PHASE1_USER).format(schema="MY_SCHEMA")
    assert "MY_RULES" in get(PromptKey.VALUES_PHASE1_GUIDANCE).format(guidance="MY_RULES")
    filled = get(PromptKey.VALUES_PHASE3_USER).format(
        page_number=7,
        ocr_words="[]",
        known_locations="(none)",
        needs_locating="- id: x",
    )
    assert "page 7" in filled and "- id: x" in filled


def test_verbatim_prompts_keep_literal_braces() -> None:
    """The schema body and phase-1 system prompt carry literal JSON braces and
    are used without str.format — the braces must survive intact."""
    # The field-tree node example in the schema body carries literal braces.
    body = get(PromptKey.SCHEMA_USER_BODY)
    assert '"name": "DueDate"' in body
    assert '"kind": "field"' in body
    assert '"kind": "table"' in get(PromptKey.VALUES_PHASE1_SYSTEM)


def test_chunked_protocol_lives_only_in_the_retry_prompt() -> None:
    """The chunked-submission tools are offered only after a truncated attempt
    (see grounded._phase1_tools), so the always-on system prompt must not
    describe a protocol whose tools aren't on the request — and the retry
    prompt must be self-contained about it."""
    assert "append_entries" not in get(PromptKey.VALUES_PHASE1_SYSTEM)
    retry = get(PromptKey.VALUES_PHASE1_RETRY_CHUNKED)
    assert "append_entries" in retry
    assert "done: false" in retry and "done: true" in retry
