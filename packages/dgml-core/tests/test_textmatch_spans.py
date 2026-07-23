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

"""Unit tests for span search over page words (dgml_core.textmatch):
the exact matcher's normalization table and the boundary-punctuation-
lenient fallback.

The real-document motivations (a PSE utility bill):

- an amount typeset as ``($14.77`` tokenizes to ``($`` + digit
  fragments — the ``$`` the target starts with is glued behind a ``(``;
- a column's amounts render without their currency symbol (``127.22``
  on the page, ``$127.22`` in the generated XML);
- a ``Label —`` separator dash exists only in the XML — the page
  renders the gap as whitespace;
- a digital PDF writes negative amounts with U+2212 while another
  extractor reads ASCII ``-`` for the same glyph.
"""

from __future__ import annotations

from dgml_core.textmatch import Word, find_spans, find_spans_lenient, fuzzy_norm


def _words(*texts: str) -> list[Word]:
    """One-line page: 50px words, 10px gaps, in the given stream order."""
    out = []
    x = 100
    for i, t in enumerate(texts):
        out.append(
            Word(
                idx=i,
                text=t,
                text_norm=fuzzy_norm(t),
                left=x,
                top=100,
                right=x + 50,
                bottom=120,
            )
        )
        x += 60
    return out


# ---- fuzzy_norm: minus/hyphen codepoint variants ---------------------------


def test_fuzzy_norm_minus_sign_variants_collapse_to_hyphen() -> None:
    # U+2212 (minus), U+2010 (hyphen), U+2011 (non-breaking hyphen),
    # U+FF0D (fullwidth) all normalize to ASCII hyphen-minus, so word
    # identity can't hinge on which codepoint the extractor produced.
    # (Escapes, not literals — the codepoint distinction IS the test.)
    assert fuzzy_norm("\u2212" + "13.29") == "-1329"
    assert fuzzy_norm("\u2010" + "13.29") == "-1329"
    assert fuzzy_norm("\u2011" + "13.29") == "-1329"
    assert fuzzy_norm("\uff0d" + "13.29") == "-1329"
    assert fuzzy_norm("-13.29") == "-1329"


def test_exact_span_matches_across_minus_codepoints() -> None:
    # XML says U+2212 "13.29"; the page word was read with ASCII "-".
    spans = find_spans("\u2212" + "13.29", _words("Charge", "-13.29", "Total"))
    assert spans == [(1, 2)]


# ---- find_spans_lenient: boundary punctuation ------------------------------


def test_lenient_first_word_sheds_glued_leading_punct() -> None:
    # "($" + "7" + "." + "89": exact search can't start inside "($".
    words = _words("Tax", "($", "7", ".", "89", "included")
    assert find_spans("$7.89", words) == []
    assert find_spans_lenient("$7.89", words) == [(1, 5)]


def test_lenient_target_missing_currency_symbol_on_page() -> None:
    # The page column renders bare "127.22"; the XML carries "$127.22".
    words = _words("Electric", "381", ".", "47", "127", ".", "22", "508")
    assert find_spans("$127.22", words) == []
    assert find_spans_lenient("$127.22", words) == [(4, 7)]


def test_lenient_target_trailing_separator_dash() -> None:
    # "Days in billing cycle — " where the page has no dash word at all.
    words = _words("Days", "in", "billing", "cycle", "Average", "temperature")
    assert find_spans("Days in billing cycle — ", words) == []
    assert find_spans_lenient("Days in billing cycle — ", words) == [(0, 4)]


def test_lenient_returns_all_candidates_for_duplicated_text() -> None:
    # Unlike the fuzzy matcher (which refuses ambiguity), the lenient
    # search returns every location — the caller scores candidates by
    # claim state and expected position, exactly like exact spans.
    words = _words("Days", "in", "billing", "cycle", "then", "Days", "in", "billing", "cycle")
    spans = find_spans_lenient("Days in billing cycle — ", words)
    assert spans == [(0, 4), (5, 9)]


def test_lenient_never_trims_letters_or_digits() -> None:
    # A changed digit still refuses — leniency is confined to boundary
    # punctuation, identity-bearing characters must match exactly.
    words = _words("128", ".", "22")
    assert find_spans_lenient("$127.22", words) == []
    # Prefix-only content never matches ("$127" is not "$127.22"
    # boundary noise — the trailing ".22" is digits).
    assert find_spans_lenient("$127.22", _words("127")) == []


def test_lenient_keeps_interior_punctuation_load_bearing() -> None:
    # "03:45" must not match "03 45" — the colon is interior, not
    # boundary, and stays load-bearing exactly as in the exact search.
    words = _words("03", "45")
    assert find_spans_lenient("03:45", words) == []
