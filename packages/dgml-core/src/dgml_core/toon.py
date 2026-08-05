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

"""TOON (Token-Oriented Object Notation, https://toonformat.dev) encoding of the
phase-3 OCR word listing.

Phase-3 grounding sends the model every OCR word on a page with its bounding box
so it can union word boxes into the bbox for each value it must locate. On
``main`` that listing is serialised as ``json.dumps(words, indent=2)`` — the most
verbose possible rendering (~60 tokens/word, mostly whitespace and repeated
keys). TOON renders the *same* words as a compact tabular block::

    words[N]{idx,text,left,top,right,bottom}:
      0,BIOPLEX,307,307,666,398
      1,",",666,307,717,398
      ...

one header naming the columns, one comma-separated row per word. Offline
measurement over the 8-file bench corpus (69 pages / 42,501 words) put this at
**-72.2% input tokens** versus the ``indent=2`` JSON on the exact same words.

The encoding is **lossless** for every field the JSON path carries: each ``idx``,
word ``text`` (verbatim — commas, quotes, tabs and newlines are backslash-escaped
so a word round-trips exactly), and integer coordinate is preserved. The one
column that is *dropped* is ``location.page_number``: it is constant across a page
(always equal to the page the surrounding prompt is already about — see
``get_page_words``), so it is hoisted into the prompt prose rather than repeated on
every row. That hoist is the only difference from the JSON payload's information
content.

:func:`decode_phase3_words` is the exact inverse of :func:`encode_phase3_words`
and exists so the losslessness property is testable / auditable offline; it is not
used on the production hot path — the model reads the table, it never sends it
back (phase 3's response contract is unchanged ``submit_locations`` bboxes).
"""

from __future__ import annotations

import re
from typing import Any

# Column order matches the ``[left, top, right, bottom]`` bbox vocabulary the
# phase-3 system prompt already documents; ``idx``/``text`` mirror the JSON keys.
_FIELDS: tuple[str, ...] = ("idx", "text", "left", "top", "right", "bottom")

# A bare (unquoted) scalar must be unambiguous on round-trip. Quote whenever the
# text could be misread as a number/bool/null, carries the ``,`` delimiter, or
# contains any character that would break the one-row-per-line table structure.
_NUMBERLIKE_RE = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_NEEDS_QUOTE_CHARS = frozenset(',"\\:\n\r\t[]{}#')

# Escapes emitted inside a quoted cell (and their inverse on decode). Only these
# five are ever produced; the surrounding quotes already protect ``,`` ``:`` etc.
_ESCAPE: dict[str, str] = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}
_UNESCAPE: dict[str, str] = {"\\": "\\", '"': '"', "n": "\n", "r": "\r", "t": "\t"}

_HEADER_RE = re.compile(r"^words\[(\d+)\]\{([^}]*)\}:$")


def _needs_quoting(s: str) -> bool:
    """True when ``s`` cannot be emitted as a bare cell without ambiguity."""
    return (
        s == ""
        or s != s.strip()
        or s in ("true", "false", "null")
        or _NUMBERLIKE_RE.match(s) is not None
        or s.startswith("-")
        or any(c in _NEEDS_QUOTE_CHARS for c in s)
    )


def toon_scalar(value: Any) -> str:
    """Render one scalar cell per TOON quoting rules.

    Ints/floats/bools/None emit bare; strings emit bare when unambiguous, else
    double-quoted with backslash escaping (``\\`` ``"`` ``\\n`` ``\\r`` ``\\t``).
    """
    if value is None:
        return "null"
    if isinstance(value, bool):  # bool is an int subclass — check first
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    s = str(value)
    if _needs_quoting(s):
        return '"' + "".join(_ESCAPE.get(c, c) for c in s) + '"'
    return s


def encode_phase3_words(words: list[dict[str, Any]]) -> str:
    """Encode ``get_page_words``-shaped word dicts as a TOON table.

    Each ``word`` is ``{"idx": int, "text": str, "location": {"page_number": int,
    "bounding_box": [left, top, right, bottom]}}`` — the exact objects the JSON
    path serialises. ``page_number`` is dropped (hoisted to the prompt prose,
    see the module docstring); everything else is preserved verbatim.
    """
    lines = [f"words[{len(words)}]{{{','.join(_FIELDS)}}}:"]
    for w in words:
        left, top, right, bottom = w["location"]["bounding_box"]
        row = (
            toon_scalar(w["idx"]),
            toon_scalar(w["text"]),
            toon_scalar(left),
            toon_scalar(top),
            toon_scalar(right),
            toon_scalar(bottom),
        )
        lines.append("  " + ",".join(row))
    return "\n".join(lines)


def _split_row(row: str) -> list[str]:
    """Split one TOON data row into its decoded cell strings.

    Top-level ``,`` separate cells; a cell may be double-quoted, in which case
    the quotes are stripped and escapes resolved. Returns the raw (bare) or
    unescaped (quoted) text of each cell — numeric conversion is the caller's job.
    """
    cells: list[str] = []
    i, n = 0, len(row)
    while True:
        if i < n and row[i] == '"':
            j = i + 1
            buf: list[str] = []
            closed = False
            while j < n:
                c = row[j]
                if c == "\\":
                    if j + 1 >= n:
                        raise ValueError(f"dangling escape in TOON row: {row!r}")
                    buf.append(_UNESCAPE.get(row[j + 1], row[j + 1]))
                    j += 2
                    continue
                if c == '"':
                    closed = True
                    break
                buf.append(c)
                j += 1
            if not closed:
                raise ValueError(f"unterminated quote in TOON row: {row!r}")
            cells.append("".join(buf))
            i = j + 1
            if i == n:
                break
            if row[i] != ",":
                raise ValueError(f"expected ',' after quoted cell in TOON row: {row!r}")
            i += 1
        else:
            k = row.find(",", i)
            if k == -1:
                cells.append(row[i:])
                break
            cells.append(row[i:k])
            i = k + 1
    return cells


def decode_phase3_words(encoded: str, page_number: int) -> list[dict[str, Any]]:
    """Inverse of :func:`encode_phase3_words`.

    Reconstructs the full ``get_page_words`` word dicts, re-attaching the hoisted
    ``page_number``. Used by tests / offline validation to prove losslessness.
    """
    lines = encoded.split("\n")
    header = lines[0] if lines else ""
    m = _HEADER_RE.match(header)
    if m is None:
        raise ValueError(f"not a phase-3 TOON words block: {header!r}")
    count = int(m.group(1))
    fields = tuple(m.group(2).split(",")) if m.group(2) else ()
    if fields != _FIELDS:
        raise ValueError(f"unexpected TOON columns {list(fields)}; expected {list(_FIELDS)}")
    rows = lines[1 : 1 + count]
    if len(rows) != count:
        raise ValueError(f"TOON header declared {count} rows but {len(rows)} present")
    out: list[dict[str, Any]] = []
    for row in rows:
        body = row[2:] if row[:2] == "  " else row
        cells = _split_row(body)
        if len(cells) != len(_FIELDS):
            raise ValueError(f"TOON row has {len(cells)} cells, expected {len(_FIELDS)}: {row!r}")
        left, top, right, bottom = (int(c) for c in cells[2:])
        out.append(
            {
                "idx": int(cells[0]),
                "text": cells[1],
                "location": {
                    "page_number": page_number,
                    "bounding_box": [left, top, right, bottom],
                },
            }
        )
    return out
