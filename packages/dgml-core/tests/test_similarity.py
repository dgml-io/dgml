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

"""Unit tests for the character-class weighted edit-distance scoring
primitive (dgml_core.textmatch.similarity and friends).

These pin the *cost model* — what each kind of edit is worth — so the
future fuzzy span matcher built on top has a stable, well-understood
foundation. The headline contract: punctuation differences are nearly
free, letter slips cost a unit, and digit edits are expensive enough to
keep numbers/dates/money distinct. The behavioral pairs here mirror the
precision guardrail suite in test_matching.py.
"""

from __future__ import annotations

import pytest
from dgml_core.textmatch import (
    CASE_EDIT_COST as _CASE_EDIT_COST,
)
from dgml_core.textmatch import (
    DIGIT_EDIT_COST as _DIGIT_EDIT_COST,
)
from dgml_core.textmatch import (
    LETTER_EDIT_COST as _LETTER_EDIT_COST,
)
from dgml_core.textmatch import (
    PUNCT_EDIT_COST as _PUNCT_EDIT_COST,
)
from dgml_core.textmatch import (
    char_class as _char_class,
)
from dgml_core.textmatch import (
    indel_cost as _indel_cost,
)
from dgml_core.textmatch import (
    similarity as _similarity,
)
from dgml_core.textmatch import (
    sub_cost as _sub_cost,
)
from dgml_core.textmatch import (
    weighted_edit_distance as _weighted_edit_distance,
)

# The recommended acceptance threshold the fuzzy matcher will use. The
# pairs below are checked relative to it so the cost weights and the
# threshold stay coherent.
THRESHOLD = 0.9


def test_char_class() -> None:
    assert _char_class("7") == "digit"
    assert _char_class("a") == "alpha"
    assert _char_class("Q") == "alpha"
    assert _char_class(" ") == "space"
    assert _char_class(":") == "punct"
    assert _char_class("-") == "punct"


def test_sub_cost_by_class() -> None:
    assert _sub_cost("a", "a") == 0.0
    assert _sub_cost("a", "b") == _LETTER_EDIT_COST
    assert _sub_cost("a", "A") == _CASE_EDIT_COST  # case-only: cheap, non-zero
    assert _sub_cost(",", ".") == _PUNCT_EDIT_COST
    assert _sub_cost(":", ";") == _PUNCT_EDIT_COST
    assert _sub_cost("1", "2") == _DIGIT_EDIT_COST
    # A digit on either side of a non-identical edit is expensive — incl.
    # OCR shape confusions, which we deliberately do not bridge.
    assert _sub_cost("O", "0") == _DIGIT_EDIT_COST
    assert _sub_cost("l", "1") == _DIGIT_EDIT_COST


def test_indel_cost_by_class() -> None:
    assert _indel_cost("x") == _LETTER_EDIT_COST
    assert _indel_cost(":") == _PUNCT_EDIT_COST
    assert _indel_cost(" ") == _PUNCT_EDIT_COST
    assert _indel_cost("5") == _DIGIT_EDIT_COST


def test_distance_identical_and_empty() -> None:
    assert _weighted_edit_distance("abc", "abc") == 0.0
    assert _weighted_edit_distance("", "") == 0.0
    assert _weighted_edit_distance("ab", "") == 2 * _LETTER_EDIT_COST
    assert _weighted_edit_distance("12", "") == 2 * _DIGIT_EDIT_COST


def test_similarity_bounds() -> None:
    assert _similarity("", "") == 1.0
    assert _similarity("abc", "abc") == 1.0
    assert _similarity("a", "") == 0.0


# --- Punctuation / boundary noise must stay above threshold (recall) ------
@pytest.mark.parametrize(
    "a,b",
    [
        ("receiving party", "receiving party:"),  # trailing colon
        ("INC.", "INC:"),  # period vs colon
        ("Terrill", "Terrill:"),  # trailing colon
        ("confidential", "confiden-tial"),  # line-break hyphen
        ('"as is"', "“as is”"),  # smart vs straight quotes
    ],
)
def test_punctuation_noise_scores_high(a: str, b: str) -> None:
    assert _similarity(a, b) > THRESHOLD


def test_case_difference_is_cheap_but_ranked_below_exact() -> None:
    # A case variant scores high (still a match candidate) but strictly
    # below an exact-case match, so the matcher can prefer the exact one.
    assert _similarity("Mutual", "mutual") > THRESHOLD
    assert _similarity("Mutual", "mutual") < _similarity("Mutual", "Mutual")


# --- Meaning-changing differences must fall below threshold (precision) ---
@pytest.mark.parametrize(
    "a,b",
    [
        ("1001", "1000"),  # one digit
        ("5/6/2018", "5/16/2018"),  # inserted digit (date)
        ("03:45", "03:46"),  # digit behind a colon
        ("$1,250.00", "$1,350.00"),  # currency digit
        ("98033", "98303"),  # transposed digits
        ("Smith", "Smithson"),  # target is only a prefix
        ("shall not disclose", "shall disclose"),  # dropped negation
        ("O", "0"),  # OCR shape confusion across classes
    ],
)
def test_meaning_changes_score_below_threshold(a: str, b: str) -> None:
    assert _similarity(a, b) < THRESHOLD


def test_digit_edit_dominates_short_field_but_not_long_clause() -> None:
    """The same single-digit change is fatal in a short numeric field yet
    negligible in a long clause — length normalization, not a special
    case, is what produces that."""
    short = _similarity("100", "200")
    clause_a = "the term of this agreement is 5 years from the effective date"
    clause_b = "the term of this agreement is 6 years from the effective date"
    long = _similarity(clause_a, clause_b)
    assert short < 0.7
    assert long > THRESHOLD
    assert long > short
