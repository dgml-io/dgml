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

"""Tests for the phase-3 TOON word-listing codec and its selector flag."""

from __future__ import annotations

from typing import Any

import pytest
from dgml_core.grounded import _compact_extraction_words_enabled
from dgml_core.toon import decode_phase3_words, encode_phase3_words, toon_scalar

_PAGE = 4


def _word(idx: int, text: str, box: tuple[int, int, int, int]) -> dict[str, Any]:
    """A single ``get_page_words``-shaped word record on page ``_PAGE``."""
    return {
        "idx": idx,
        "text": text,
        "location": {"page_number": _PAGE, "bounding_box": list(box)},
    }


# ---------------------------------------------------------------------------
# toon_scalar quoting rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (42, "42"),
        (-5, "-5"),
        (True, "true"),
        (False, "false"),
        (None, "null"),
        ("BIOPLEX", "BIOPLEX"),  # plain word: bare
        ("Hello", "Hello"),
        ("307", '"307"'),  # numberlike text quoted to keep it a string
        ("-cap", '"-cap"'),  # leading dash is list-marker ambiguous
        (" x", '" x"'),  # leading space forces quoting; space kept verbatim
        ("true", '"true"'),  # bool-like text quoted
        ("a,b", '"a,b"'),  # delimiter forces quoting
        ('he said "hi"', '"he said \\"hi\\""'),  # quote escaped
        ("back\\slash", '"back\\\\slash"'),  # backslash escaped
        ("line1\nline2", '"line1\\nline2"'),  # newline escaped
        ("col1\tcol2", '"col1\\tcol2"'),  # tab escaped
    ],
)
def test_toon_scalar(value: Any, expected: str) -> None:
    assert toon_scalar(value) == expected


# ---------------------------------------------------------------------------
# encode structure + constant-column hoist
# ---------------------------------------------------------------------------


def test_encode_header_and_rows() -> None:
    words = [_word(0, "BIOPLEX", (307, 307, 666, 398)), _word(1, "INC", (728, 307, 883, 398))]
    encoded = encode_phase3_words(words)
    lines = encoded.split("\n")
    assert lines[0] == "words[2]{idx,text,left,top,right,bottom}:"
    # One row per word, indented two spaces, comma-separated columns.
    assert lines[1] == "  0,BIOPLEX,307,307,666,398"
    assert lines[2] == "  1,INC,728,307,883,398"
    assert len(lines) == 3


def test_encode_hoists_constant_page_number() -> None:
    """page_number is dropped from every row (hoisted to the prompt prose):
    it appears nowhere in the table, yet decode restores it from the page arg."""
    words = [_word(0, "A", (1, 2, 3, 4)), _word(1, "B", (5, 6, 7, 8))]
    encoded = encode_phase3_words(words)
    assert "page_number" not in encoded
    # Six columns only (idx,text,left,top,right,bottom) — no page_number column.
    assert encoded.split("\n")[0] == "words[2]{idx,text,left,top,right,bottom}:"
    for row in encoded.split("\n")[1:]:
        assert len(row.split(",")) == 6
    restored = decode_phase3_words(encoded, _PAGE)
    assert all(w["location"]["page_number"] == _PAGE for w in restored)


def test_encode_empty_words() -> None:
    encoded = encode_phase3_words([])
    assert encoded == "words[0]{idx,text,left,top,right,bottom}:"
    assert decode_phase3_words(encoded, _PAGE) == []


# ---------------------------------------------------------------------------
# round-trip losslessness (the core lever guarantee)
# ---------------------------------------------------------------------------


def test_roundtrip_special_characters() -> None:
    """Words containing comma, quote, newline, tab and unicode round-trip
    byte-exactly through encode -> decode."""
    words = [
        _word(0, "plain", (0, 0, 10, 10)),
        _word(1, ",", (10, 0, 20, 10)),  # bare comma word
        _word(2, 'quote"inside', (20, 0, 30, 10)),
        _word(3, "tab\there", (30, 0, 40, 10)),
        _word(4, "new\nline", (40, 0, 50, 10)),
        _word(5, "back\\slash", (50, 0, 60, 10)),
        _word(6, "café—naïve — €5 ½", (60, 0, 70, 10)),  # unicode
        _word(7, "307", (70, 0, 80, 10)),  # numberlike text stays text
        _word(8, "  leading+trailing  ", (80, 0, 90, 10)),
        _word(9, "true", (90, 0, 100, 10)),  # bool-like text stays text
        _word(10, 'a,b,c\td\ne"f', (100, 0, 110, 10)),  # everything at once
    ]
    encoded = encode_phase3_words(words)
    # The table stays one physical line per word even with embedded newlines.
    assert len(encoded.split("\n")) == 1 + len(words)
    assert decode_phase3_words(encoded, _PAGE) == words


def test_roundtrip_preserves_int_types_and_coords() -> None:
    words = [_word(123, "Word", (11, 22, 333, 4444))]
    restored = decode_phase3_words(encode_phase3_words(words), _PAGE)
    assert restored == words
    assert restored[0]["idx"] == 123 and isinstance(restored[0]["idx"], int)
    assert restored[0]["location"]["bounding_box"] == [11, 22, 333, 4444]


# ---------------------------------------------------------------------------
# decoder guards
# ---------------------------------------------------------------------------


def test_decode_rejects_bad_header() -> None:
    with pytest.raises(ValueError, match="not a phase-3 TOON words block"):
        decode_phase3_words("[]", _PAGE)


def test_decode_rejects_wrong_columns() -> None:
    with pytest.raises(ValueError, match="unexpected TOON columns"):
        decode_phase3_words("words[1]{idx,text,x0,y0,x1,y1}:\n  0,a,1,2,3,4", _PAGE)


def test_decode_rejects_row_count_mismatch() -> None:
    with pytest.raises(ValueError, match="declared 2 rows"):
        decode_phase3_words("words[2]{idx,text,left,top,right,bottom}:\n  0,a,1,2,3,4", _PAGE)


def test_decode_rejects_wrong_cell_count() -> None:
    with pytest.raises(ValueError, match="cells, expected 6"):
        decode_phase3_words("words[1]{idx,text,left,top,right,bottom}:\n  0,a,1,2,3", _PAGE)


# ---------------------------------------------------------------------------
# flag resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Off: unset, empty/whitespace, and the accepted false spellings.
        (None, False),
        ("", False),
        ("  ", False),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("OFF", False),  # case-insensitive
        # On: the accepted true spellings, case-insensitive and trimmed.
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("On", True),
        ("  TRUE  ", True),
    ],
)
def test_flag_resolves(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: bool) -> None:
    if raw is None:
        monkeypatch.delenv("DGML_COMPACT_EXTRACTION_WORDS", raising=False)
    else:
        monkeypatch.setenv("DGML_COMPACT_EXTRACTION_WORDS", raw)
    assert _compact_extraction_words_enabled() is expected


def test_flag_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DGML_COMPACT_EXTRACTION_WORDS", "yaml")
    with pytest.raises(
        ValueError, match="DGML_COMPACT_EXTRACTION_WORDS='yaml' is not a valid boolean"
    ):
        _compact_extraction_words_enabled()
