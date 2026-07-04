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

"""Tests for the coverage text tokenizer (``dgml_core.generation.coverage``)."""

from __future__ import annotations

from dgml_core.generation.coverage import _tokenize, merge_coverage_documents


def test_merge_coverage_documents_keeps_prior_and_replaces_by_source() -> None:
    existing = [{"source": "a.pdf", "rouge1_pct": 90.0}, {"source": "b.pdf", "rouge1_pct": 80.0}]
    new = [{"source": "b.pdf", "rouge1_pct": 85.0}, {"source": "c.pdf", "rouge1_pct": 70.0}]
    merged = {d["source"]: d["rouge1_pct"] for d in merge_coverage_documents(existing, new)}
    assert merged == {"a.pdf": 90.0, "b.pdf": 85.0, "c.pdf": 70.0}  # a kept, b updated, c added


def test_hyphenated_and_spaced_forms_tokenize_identically() -> None:
    # A hyphenated compound and its spaced/hyphen-with-spaces variants must
    # produce the same tokens, so coverage doesn't flag spurious "missing"
    # words when the model and the PDF differ only in hyphenation.
    expected = ["all", "inclusive"]
    assert _tokenize("All-Inclusive") == expected
    assert _tokenize("All - Inclusive") == expected
    assert _tokenize("all inclusive") == expected


def test_currency_and_thousands_separator_stripped() -> None:
    assert _tokenize("$3,995") == ["3995"]


def test_single_char_and_punctuation_only_tokens_dropped() -> None:
    # Lone letters, dot leaders, and signature underscores are noise.
    assert _tokenize("a b cd") == ["cd"]
    assert _tokenize("____ ... -") == []


def test_apostrophes_preserved() -> None:
    assert _tokenize("Explorer's guide") == ["explorer's", "guide"]
