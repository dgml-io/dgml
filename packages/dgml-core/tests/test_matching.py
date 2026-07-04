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

from __future__ import annotations

import json
from typing import Any

import pytest
from dgml_core.matching import (
    _array_layout_key,
    _walk_leaves,
    parse_path,
    path_to_str,
    run_phase2_matching,
    walk_computed_leaves,
)
from dgml_core.models import FileRecord
from dgml_core.storage import Workspace, write_json_atomic


def _seed_file_and_text(
    workspace: Workspace,
    file_id: str,
    page: int,
    words: list[dict[str, Any]],
    *,
    width: int = 1000,
    height: int = 1000,
) -> None:
    """Set up the minimum on-disk state the matcher reads: file.json
    (so storage helpers resolve a path) + a page_text/page_N.json."""
    workspace.file_dir(file_id).mkdir(parents=True, exist_ok=True)
    record = FileRecord(
        id=file_id,
        original_path="/fake/x.pdf",
        original_filename="x.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=page,
        text_mode="digital",
    )
    write_json_atomic(workspace.file_json_path(file_id), record.to_json())
    workspace.file_text_dir(file_id).mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        workspace.file_text_dir(file_id) / f"page_{page}.json",
        {"file_id": file_id, "page": page, "width": width, "height": height, "words": words},
    )


def test_single_candidate_match_fills_bbox(workspace: Workspace) -> None:
    """A unique span on the page becomes a bbox in integer image-pixel
    space, with no LLM in the loop."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            {"t": "Hello", "l": [100, 210, 182, 242]},
            {"t": "world", "l": [190, 210, 290, 242]},
        ],
    )
    phase1 = {"title": {"text": "Hello world", "locations": [{"page_number": 1}]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)

    assert result.stats.matched_locations == 1
    assert result.stats.unmatched_locations == 0
    assert result.unmatched == []
    # Integer image pixels [left, top, right, bottom]: union of the two
    # words → left=100, top=210, right=290, bottom=242.
    assert result.values["title"]["locations"] == [
        {"page_number": 1, "bounding_box": [100, 210, 290, 242]}
    ]


def test_no_candidates_leaves_unmatched(workspace: Workspace) -> None:
    """A value whose text doesn't appear on the page becomes an
    ``UnmatchedItem`` for phase 3, with a short page-unique id."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[{"t": "Hello", "l": [100, 100, 200, 120]}],
    )
    phase1 = {"title": {"text": "Goodbye", "locations": [{"page_number": 1}]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)

    assert result.stats.matched_locations == 0
    assert result.stats.unmatched_locations == 1
    assert len(result.unmatched) == 1
    item = result.unmatched[0]
    assert item.text == "Goodbye"
    assert item.page_number == 1
    assert item.id == "a"  # first id on this page
    # The leaf's location stays as a page-only placeholder so the caller
    # can later splice in the phase-3 result.
    assert result.values["title"]["locations"] == [{"page_number": 1}]


def test_ambiguous_match_disambiguated_by_sibling_row(workspace: Workspace) -> None:
    """When two OCR spans match a value's text, the candidate within a
    row of a matched sibling wins. The unmatched sibling 'Jul 01, 2025'
    appears in two columns; 'Bob' uniquely matches the row at y=110,
    so the right 'Jul 01, 2025' is the one at y≈110."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Row 1 (y≈110): Bob | Jul 01, 2025
            {"t": "Bob", "l": [50, 100, 80, 120]},
            {"t": "Jul", "l": [100, 100, 120, 120]},
            {"t": "01,", "l": [125, 100, 145, 120]},
            {"t": "2025", "l": [150, 100, 180, 120]},
            # Row 2 (y≈210): Carol | Jul 01, 2025
            {"t": "Carol", "l": [50, 200, 90, 220]},
            {"t": "Jul", "l": [100, 200, 120, 220]},
            {"t": "01,", "l": [125, 200, 145, 220]},
            {"t": "2025", "l": [150, 200, 180, 220]},
        ],
    )
    phase1 = {
        "rows": [
            {
                "name": {"text": "Bob", "locations": [{"page_number": 1}]},
                "date": {"text": "Jul 01, 2025", "locations": [{"page_number": 1}]},
            }
        ]
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)

    # Both rows[0].name and rows[0].date resolved — the date's two
    # candidates were disambiguated by name's row anchor.
    assert result.stats.matched_locations == 2
    assert result.stats.unmatched_locations == 0
    bob_box = result.values["rows"][0]["name"]["locations"][0]["bounding_box"]
    date_box = result.values["rows"][0]["date"]["locations"][0]["bounding_box"]
    # Both should sit on row 1: top=100, bottom=120 (box is [left, top,
    # right, bottom] image pixels).
    assert bob_box[1] == 100 and bob_box[3] == 120
    assert date_box[1] == 100 and date_box[3] == 120


def test_ambiguous_match_without_anchor_stays_unmatched(workspace: Workspace) -> None:
    """When a value has multiple candidates and no matched sibling
    provides a row anchor, the matcher defers to phase 3 rather than
    guessing wrong."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # 'Acme' appears twice, no sibling to anchor against.
            {"t": "Acme", "l": [100, 100, 140, 120]},
            {"t": "Acme", "l": [100, 300, 140, 320]},
        ],
    )
    phase1 = {"name": {"text": "Acme", "locations": [{"page_number": 1}]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)

    assert result.stats.unmatched_locations == 1
    assert result.unmatched[0].text == "Acme"


def test_whitespace_collapse_in_text_and_words(workspace: Workspace) -> None:
    """Internal whitespace doesn't dictate matches — '$1,123.03' matches
    whether the OCR token kept the spaces or split."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            {"t": "$", "l": [100, 100, 110, 120]},
            {"t": "1,123.03", "l": [115, 100, 180, 120]},
        ],
    )
    phase1 = {"balance": {"text": "$1,123.03", "locations": [{"page_number": 1}]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)

    assert result.stats.matched_locations == 1
    assert result.unmatched == []


def test_no_page_text_falls_through_to_unmatched(workspace: Workspace) -> None:
    """A file without page_text (file was added without text extraction)
    can't be matched in code — every leaf becomes unmatched and waits
    for phase 3 to look at the page image."""
    # Seed file but no page_text.
    workspace.file_dir("f1aaaaaaaaaa").mkdir(parents=True, exist_ok=True)
    record = FileRecord(
        id="f1aaaaaaaaaa",
        original_path="/fake/x.pdf",
        original_filename="x.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=1,
        text_mode="digital",
    )
    write_json_atomic(workspace.file_json_path("f1aaaaaaaaaa"), record.to_json())

    phase1 = {"title": {"text": "Hello", "locations": [{"page_number": 1}]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)
    assert result.stats.unmatched_locations == 1


def test_stats_persisted_dict_does_not_mutate_input(workspace: Workspace) -> None:
    """The matcher returns a fresh values dict — patching its locations
    doesn't leak back into the caller's phase-1 result."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[{"t": "Hi", "l": [10, 10, 30, 30]}],
    )
    phase1 = {"x": {"text": "Hi", "locations": [{"page_number": 1}]}}
    original = json.loads(json.dumps(phase1))
    run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)
    assert phase1 == original


# ---------------------------------------------------------------------------
# Column anchor disambiguation
# ---------------------------------------------------------------------------


def test_column_anchor_resolves_left_aligned_collision(workspace: Workspace) -> None:
    """A same-row collision (two cells with identical text on one row)
    that the row anchor can't break gets resolved by the column anchor
    derived from the array's other rows.

    Layout: three transactions; rows 1 and 2 have unique post_date /
    due_date values that match unambiguously, establishing the column
    positions. Row 0's post_date and due_date happen to share the same
    text — the column anchor (left edge of each column) picks the right
    candidate for each."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Row 0 (y≈100): post=Jun 16, due=Jun 16 — collide.
            {"t": "Jun", "l": [50, 100, 80, 120]},
            {"t": "16,", "l": [82, 100, 95, 120]},
            {"t": "2025", "l": [97, 100, 130, 120]},
            {"t": "Jun", "l": [200, 100, 230, 120]},
            {"t": "16,", "l": [232, 100, 245, 120]},
            {"t": "2025", "l": [247, 100, 280, 120]},
            # Row 1 (y≈200): post=May 27, due=Jun 30 — unique values.
            {"t": "May", "l": [50, 200, 80, 220]},
            {"t": "27,", "l": [82, 200, 95, 220]},
            {"t": "2025", "l": [97, 200, 130, 220]},
            {"t": "Jun", "l": [200, 200, 230, 220]},
            {"t": "30,", "l": [232, 200, 245, 220]},
            {"t": "2025", "l": [247, 200, 280, 220]},
            # Row 2 (y≈300): post=Apr 29, due=Apr 30 — unique values.
            {"t": "Apr", "l": [50, 300, 80, 320]},
            {"t": "29,", "l": [82, 300, 95, 320]},
            {"t": "2025", "l": [97, 300, 130, 320]},
            {"t": "Apr", "l": [200, 300, 230, 320]},
            {"t": "30,", "l": [232, 300, 245, 320]},
            {"t": "2025", "l": [247, 300, 280, 320]},
        ],
    )
    phase1 = {
        "transactions": [
            {
                "post_date": {"text": "Jun 16, 2025", "locations": [{"page_number": 1}]},
                "due_date": {"text": "Jun 16, 2025", "locations": [{"page_number": 1}]},
            },
            {
                "post_date": {"text": "May 27, 2025", "locations": [{"page_number": 1}]},
                "due_date": {"text": "Jun 30, 2025", "locations": [{"page_number": 1}]},
            },
            {
                "post_date": {"text": "Apr 29, 2025", "locations": [{"page_number": 1}]},
                "due_date": {"text": "Apr 30, 2025", "locations": [{"page_number": 1}]},
            },
        ]
    }
    layout = {"transactions": {"kind": "table", "columns": ["post_date", "due_date"]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    # All 6 locations resolved by code; phase 3 has nothing to do.
    assert result.stats.matched_locations == 6
    assert result.unmatched == []
    # The colliding row's post_date sits in the left column (xmin=50),
    # its due_date in the right column (xmin=200) — column anchor did
    # the disambiguation.
    pd = result.values["transactions"][0]["post_date"]["locations"][0]["bounding_box"]
    dd = result.values["transactions"][0]["due_date"]["locations"][0]["bounding_box"]
    assert pd[0] == 50  # left
    assert dd[0] == 200


def test_column_anchor_handles_right_aligned_currency(workspace: Workspace) -> None:
    """Right-aligned currency: peer xmins vary (different number
    widths), peer xmaxs are tight. The matcher must pick the xmax edge
    so the column anchor stays usable."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Row 0: amount=$1,234.56 — unique value (right-aligned).
            {"t": "$1,234.56", "l": [220, 100, 300, 120]},
            # Row 1: amount=$50.00 — unique. Narrower; xmin shifts right.
            {"t": "$50.00", "l": [255, 200, 300, 220]},
            # Row 2: amount=$10.00 — collides with another "$10.00" we
            # also place in a non-amount cell to force ambiguity.
            {"t": "$10.00", "l": [263, 300, 300, 320]},
            {"t": "$10.00", "l": [400, 300, 437, 320]},  # bogus second "10.00" on the same row
        ],
    )
    phase1 = {
        "rows": [
            {"amount": {"text": "$1,234.56", "locations": [{"page_number": 1}]}},
            {"amount": {"text": "$50.00", "locations": [{"page_number": 1}]}},
            {"amount": {"text": "$10.00", "locations": [{"page_number": 1}]}},
        ]
    }
    layout = {"rows": {"kind": "table", "columns": ["amount"]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    assert result.stats.matched_locations == 3
    # Row 2's $10.00 had two candidates; right-edge anchor (xmax≈300)
    # picks the one in the currency column, not the bogus one at xmax=437.
    box = result.values["rows"][2]["amount"]["locations"][0]["bounding_box"]
    assert box[2] == 300  # right


def test_column_anchor_handles_center_aligned(workspace: Workspace) -> None:
    """Center-aligned dates: xmin and xmax both wander as widths change
    but x_center stays put. The picker must select the center edge."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Rows 0-2: unique date texts of varying widths, all
            # centered at x=200. xmin and xmax both spread, x_center
            # is the only tight edge — that's what the heuristic must
            # latch onto.
            {"t": "ABC", "l": [180, 100, 220, 120]},  # cx=200, w=40
            {"t": "DEFG", "l": [170, 200, 230, 220]},  # cx=200, w=60
            {"t": "HI", "l": [190, 300, 210, 320]},  # cx=200, w=20
            # Row 3's date "JK" appears in the date column AND as stray
            # text off-column on the same row, creating ambiguity that
            # row anchor (no siblings here) can't break.
            {"t": "JK", "l": [185, 400, 215, 420]},  # cx=200, in col
            {"t": "JK", "l": [400, 400, 430, 420]},  # cx=415, off col
        ],
    )
    phase1 = {
        "rows": [
            {"date": {"text": "ABC", "locations": [{"page_number": 1}]}},
            {"date": {"text": "DEFG", "locations": [{"page_number": 1}]}},
            {"date": {"text": "HI", "locations": [{"page_number": 1}]}},
            {"date": {"text": "JK", "locations": [{"page_number": 1}]}},
        ]
    }
    layout = {"rows": {"kind": "table", "columns": ["date"]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    assert result.stats.matched_locations == 4
    # Row 3 had two candidates for "JK"; the centered anchor (cx=200)
    # picked the in-column candidate over the off-column one (cx=415).
    box = result.values["rows"][3]["date"]["locations"][0]["bounding_box"]
    assert (box[0] + box[2]) / 2 == 200.0  # x_center (left+right)/2


def test_column_anchor_declines_for_scattered_layout(workspace: Workspace) -> None:
    """Free-form list of signatures: each name sits at a wildly
    different x. The spread is too high to claim a column, so the
    heuristic declines and the ambiguous case stays unmatched
    (deferring to phase 3)."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Three signature blocks at scattered positions.
            {"t": "Alice", "l": [50, 100, 100, 120]},  # cx=75
            {"t": "Bob", "l": [400, 200, 430, 220]},  # cx=415
            {"t": "Carol", "l": [700, 300, 750, 320]},  # cx=725
            # Ambiguous row: "Dave" appears twice, far apart.
            {"t": "Dave", "l": [60, 400, 100, 420]},
            {"t": "Dave", "l": [600, 400, 640, 420]},
        ],
    )
    phase1 = {
        "signatures": [
            {"name": {"text": "Alice", "locations": [{"page_number": 1}]}},
            {"name": {"text": "Bob", "locations": [{"page_number": 1}]}},
            {"name": {"text": "Carol", "locations": [{"page_number": 1}]}},
            {"name": {"text": "Dave", "locations": [{"page_number": 1}]}},
        ]
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)
    # Three unique names matched; "Dave" stays ambiguous because the
    # column anchor correctly refuses to claim a column for this layout.
    assert result.stats.matched_locations == 3
    assert result.stats.unmatched_locations == 1
    assert result.unmatched[0].text == "Dave"


def test_column_anchor_skipped_when_no_array_in_path(workspace: Workspace) -> None:
    """Column anchors only apply to array-of-object schemas. A
    non-array path has no peers, so the heuristic shouldn't try to
    anchor anything."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            {"t": "Acme", "l": [50, 100, 100, 120]},
            {"t": "Acme", "l": [200, 100, 250, 120]},
        ],
    )
    # ``issuer.name`` has no array index in the path; both "Acme"
    # candidates are on the same row, no sibling to anchor.
    phase1 = {
        "issuer": {"name": {"text": "Acme", "locations": [{"page_number": 1}]}},
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)
    assert result.stats.unmatched_locations == 1


# ---------------------------------------------------------------------------
# Layout-driven row assignment
# ---------------------------------------------------------------------------


def test_layout_resolves_all_values_collide_pathology(workspace: Workspace) -> None:
    """The case neither row nor column anchor can fix: every row of an
    array has post_date == due_date, so phase 2 never bootstraps a
    column anchor for either field. The layout hint ('post_date is
    column 0, due_date is column 1') breaks the tie by sorting the
    row's unresolved candidates left-to-right."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Row 0 (y=100): both dates "Jun 16, 2025"; a third unique
            # column 'id' anchors the row.
            {"t": "Jun", "l": [50, 100, 80, 120]},
            {"t": "16,", "l": [82, 100, 95, 120]},
            {"t": "2025", "l": [97, 100, 130, 120]},
            {"t": "Jun", "l": [200, 100, 230, 120]},
            {"t": "16,", "l": [232, 100, 245, 120]},
            {"t": "2025", "l": [247, 100, 280, 120]},
            {"t": "AAA", "l": [400, 100, 430, 120]},
            # Row 1 (y=200): dates collide again on a different date.
            {"t": "Jul", "l": [50, 200, 80, 220]},
            {"t": "01,", "l": [82, 200, 95, 220]},
            {"t": "2025", "l": [97, 200, 130, 220]},
            {"t": "Jul", "l": [200, 200, 230, 220]},
            {"t": "01,", "l": [232, 200, 245, 220]},
            {"t": "2025", "l": [247, 200, 280, 220]},
            {"t": "BBB", "l": [400, 200, 430, 220]},
        ],
    )
    phase1 = {
        "transactions": [
            {
                "post_date": {"text": "Jun 16, 2025", "locations": [{"page_number": 1}]},
                "due_date": {"text": "Jun 16, 2025", "locations": [{"page_number": 1}]},
                "id": {"text": "AAA", "locations": [{"page_number": 1}]},
            },
            {
                "post_date": {"text": "Jul 01, 2025", "locations": [{"page_number": 1}]},
                "due_date": {"text": "Jul 01, 2025", "locations": [{"page_number": 1}]},
                "id": {"text": "BBB", "locations": [{"page_number": 1}]},
            },
        ]
    }
    layout = {
        "transactions": {"kind": "table", "columns": ["post_date", "due_date", "id"]},
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    assert result.stats.matched_locations == 6
    assert result.unmatched == []
    # The layout pass put post_date at left=50 and due_date at left=200
    # in BOTH rows — exactly the column order the layout dictated.
    for i in range(2):
        pd = result.values["transactions"][i]["post_date"]["locations"][0]["bounding_box"]
        dd = result.values["transactions"][i]["due_date"]["locations"][0]["bounding_box"]
        assert pd[0] == 50
        assert dd[0] == 200
        assert pd != dd


def test_layout_skipped_for_free_form_arrays(workspace: Workspace) -> None:
    """A ``free_form`` array layout means the matcher must NOT apply
    column ordering — the items aren't column-shaped."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            {"t": "Acme", "l": [50, 100, 100, 120]},
            {"t": "Acme", "l": [400, 100, 450, 120]},
        ],
    )
    phase1 = {
        "signatures": [
            {"name": {"text": "Acme", "locations": [{"page_number": 1}]}},
        ]
    }
    layout = {"signatures": {"kind": "free_form"}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    # Two candidates, no sibling to anchor, layout says free-form ⇒
    # phase 2 correctly defers the lone task to phase 3.
    assert result.stats.unmatched_locations == 1


# ---------------------------------------------------------------------------
# Row-band enforcement
# ---------------------------------------------------------------------------


def test_row_band_rejects_off_row_candidate(workspace: Workspace) -> None:
    """Once a row has at least one matched sibling, candidates outside
    that sibling's y-band are rejected even when the off-row candidate
    is a perfectly valid text match.

    Setup: row 0 has three fields. The page has the legit tokens at
    y=100 plus a stray "C0" token at y=600. With the row band
    enforced, row 0's `c` matches at y=100, not y=600 — even though
    y=600 is also a valid text match."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Row 0 (y=100-120): three unique tokens packed at the same y.
            {"t": "A0", "l": [50, 100, 80, 120]},
            {"t": "B0", "l": [200, 100, 230, 120]},
            {"t": "C0", "l": [400, 100, 430, 120]},
            # Stray duplicate "C0" off-row at y=600 — same text as a
            # legit cell to force ambiguity.
            {"t": "C0", "l": [400, 600, 430, 620]},
        ],
    )
    phase1 = {
        "rows": [
            {
                "a": {"text": "A0", "locations": [{"page_number": 1}]},
                "b": {"text": "B0", "locations": [{"page_number": 1}]},
                "c": {"text": "C0", "locations": [{"page_number": 1}]},
            },
        ]
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)
    # `c` resolved at y=100 (in-row), not y=600 (stray).
    box = result.values["rows"][0]["c"]["locations"][0]["bounding_box"]
    assert box[1] == 100  # top
    assert box[0] == 400  # left column confirms it's the right token


def test_row_band_blocks_wrong_row_when_first_field_matches_first(
    workspace: Workspace,
) -> None:
    """The order in which a row's fields get processed matters: as
    soon as one matches, the band shrinks, and subsequent same-row
    fields that have a *different-row* unique candidate stay
    unmatched (rather than locking the row to the wrong y).

    Setup: row 0 has two fields. The page only has one A0 (correct
    row, y=100). The page has TWO B0 instances — one at the correct
    row y=100 and one off-row at y=400. When a matches first at
    y=100, b's off-row candidate is filtered out and b matches at
    y=100 too. If a hadn't existed, b's first candidate (y=100, in
    page-reading order) would have been picked greedily and the
    wrong-row scenario would have to be cleaned up by the audit."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            {"t": "A0", "l": [50, 100, 80, 120]},
            {"t": "B0", "l": [200, 100, 230, 120]},
            {"t": "B0", "l": [200, 400, 230, 420]},
        ],
    )
    phase1 = {
        "rows": [
            {
                "a": {"text": "A0", "locations": [{"page_number": 1}]},
                "b": {"text": "B0", "locations": [{"page_number": 1}]},
            },
        ]
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)
    a_box = result.values["rows"][0]["a"]["locations"][0]["bounding_box"]
    b_box = result.values["rows"][0]["b"]["locations"][0]["bounding_box"]
    assert a_box[1] == 100  # top
    assert b_box[1] == 100


def test_audit_demotes_misordered_single_candidate(workspace: Workspace) -> None:
    """When the first task in a row picks a single-candidate match
    that turns out to be on the wrong row (because its only candidate
    is off-row), the band gets locked to the wrong y and the rest of
    the row's fields can't match. The audit detects this — most
    candidates' y-centers cluster at the row's real y — and demotes
    the lone misfit so the row can re-resolve correctly.

    Setup: row 0's fields are a, b, c, d. The page has legit B0/C0/D0
    at y=100 and a stray A0 only at y=600 (no legit A0 in the
    correct row). a's only candidate is off-row."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Row 0 legit fields at y=100, EXCEPT no A0 there.
            {"t": "B0", "l": [200, 100, 230, 120]},
            {"t": "C0", "l": [400, 100, 430, 120]},
            {"t": "D0", "l": [600, 100, 630, 120]},
            # Stray A0 at y=600 — a's only candidate.
            {"t": "A0", "l": [50, 600, 80, 620]},
        ],
    )
    phase1 = {
        "rows": [
            {
                "a": {"text": "A0", "locations": [{"page_number": 1}]},
                "b": {"text": "B0", "locations": [{"page_number": 1}]},
                "c": {"text": "C0", "locations": [{"page_number": 1}]},
                "d": {"text": "D0", "locations": [{"page_number": 1}]},
            },
        ]
    }
    # Row-coherence audit is now phase-1-gated: it fires only for arrays
    # marked ``kind:"table"``. Pass the hint so this tabular scenario
    # exercises the audit path.
    layout = {"rows": {"kind": "table", "columns": ["a", "b", "c", "d"]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    # In iteration 1 a got matched at y=600 (its only candidate),
    # locking the band away from the real row. b/c/d ended up
    # unmatched. The audit demoted a (single-leaf row + outlier wrt
    # the unmatched candidates' y), freeing iteration 2 to match
    # b/c/d at y=100. a stays unmatched (no legit candidate left).
    row = result.values["rows"][0]
    assert row["b"]["locations"][0]["bounding_box"][1] == 100  # top
    assert row["c"]["locations"][0]["bounding_box"][1] == 100
    assert row["d"]["locations"][0]["bounding_box"][1] == 100
    # a couldn't recover — its only candidate is off-row. Goes to phase 3.
    a_locs = row["a"]["locations"]
    assert all("bounding_box" not in loc for loc in a_locs)


def test_column_anchor_filters_single_candidate_in_wrong_column(
    workspace: Workspace,
) -> None:
    """A single-candidate match that falls outside its column anchor's
    tolerance should be rejected, not committed. This covers the OCR
    misspelling case: ``charge_code`` for one row only matches a
    sub-span of that row's description (because the actual charge_code
    cell is OCR'd as ``Utilitles``), but its candidate's x is far from
    the column's settled position. We'd rather leave it unmatched and
    let phase 3 see the page image."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Rows 0-2: clean charge_code at xmin=340, unique values.
            {"t": "Alpha", "l": [340, 100, 400, 120]},
            {"t": "Beta", "l": [340, 200, 400, 220]},
            {"t": "Gamma", "l": [340, 300, 400, 320]},
            # Row 3: charge_code OCR is garbage, only the description-col
            # prefix ``Delta`` is reachable as a text match.
            {"t": "DelTta", "l": [340, 400, 400, 420]},  # typo in charge col
            {"t": "Delta", "l": [550, 400, 610, 420]},  # delta inside description col
            # Sibling anchoring the rows.
            {"t": "row0", "l": [700, 100, 750, 120]},
            {"t": "row1", "l": [700, 200, 750, 220]},
            {"t": "row2", "l": [700, 300, 750, 320]},
            {"t": "row3", "l": [700, 400, 750, 420]},
        ],
    )
    phase1 = {
        "rows": [
            {
                "code": {"text": "Alpha", "locations": [{"page_number": 1}]},
                "id": {"text": "row0", "locations": [{"page_number": 1}]},
            },
            {
                "code": {"text": "Beta", "locations": [{"page_number": 1}]},
                "id": {"text": "row1", "locations": [{"page_number": 1}]},
            },
            {
                "code": {"text": "Gamma", "locations": [{"page_number": 1}]},
                "id": {"text": "row2", "locations": [{"page_number": 1}]},
            },
            # Row 3's code matches only the description-col token; the
            # filter should reject it as off-column.
            {
                "code": {"text": "Delta", "locations": [{"page_number": 1}]},
                "id": {"text": "row3", "locations": [{"page_number": 1}]},
            },
        ]
    }
    layout = {"rows": {"kind": "table", "columns": ["code", "id"]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    # First 3 rows resolve at the correct column (left=340).
    for i in range(3):
        box = result.values["rows"][i]["code"]["locations"][0]["bounding_box"]
        assert box[0] == 340, f"row {i} code should be at left=340, got {box[0]}"
    # Row 3's code stays unmatched — the lone candidate at left=550
    # is outside the column anchor, so the filter rejects it.
    row3_code_locs = result.values["rows"][3]["code"]["locations"]
    assert all("bounding_box" not in loc for loc in row3_code_locs), (
        f"row 3 code should be unmatched but got {row3_code_locs}"
    )


def test_fuzzy_punctuation_matches_dropped_comma(workspace: Workspace) -> None:
    """OCR sometimes drops the comma from a date entirely (the cell
    reads ``Jun 16 2025`` instead of ``Jun 16, 2025``). The fuzzy
    canonicalization collapses commas (and periods) to nothing so
    phase 2 still resolves the cell — without it, only the column
    with a clean OCR read would be a candidate, and phase 2 would
    pin the wrong field there."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Post col with comma intact; Due col missing the comma.
            {"t": "Jun", "l": [50, 100, 80, 120]},
            {"t": "16,", "l": [82, 100, 95, 120]},
            {"t": "2025", "l": [97, 100, 130, 120]},
            {"t": "Jun", "l": [200, 100, 230, 120]},
            {"t": "16", "l": [232, 100, 245, 120]},  # missing comma
            {"t": "2025", "l": [247, 100, 280, 120]},
            {"t": "ROW1", "l": [400, 100, 450, 120]},
        ],
    )
    phase1 = {
        "transactions": [
            {
                "post_date": {"text": "Jun 16, 2025", "locations": [{"page_number": 1}]},
                "due_date": {"text": "Jun 16, 2025", "locations": [{"page_number": 1}]},
                "id": {"text": "ROW1", "locations": [{"page_number": 1}]},
            }
        ]
    }
    layout = {
        "transactions": {"kind": "table", "columns": ["post_date", "due_date", "id"]},
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    assert result.unmatched == []
    pd_box = result.values["transactions"][0]["post_date"]["locations"][0]["bounding_box"]
    dd_box = result.values["transactions"][0]["due_date"]["locations"][0]["bounding_box"]
    # post_date pinned to the Post col, due_date to the Due col —
    # despite the Due-col OCR missing its comma.
    assert pd_box[0] == 50  # left
    assert dd_box[0] == 200


def test_fuzzy_punctuation_matches_comma_vs_period(workspace: Workspace) -> None:
    """When phase 1 says ``Jun 01, 2025`` but OCR reads the cell as
    ``Jun 01. 2025`` (period instead of comma — a common OCR
    confusion), the matcher should still find it. The same fuzziness
    lets phase 2 disambiguate ledger date columns that suffer the
    occasional misread without polluting itself with cross-row
    drift."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Row 0: Post col has correct comma, Due col has period (OCR misread).
            {"t": "Jun", "l": [50, 100, 80, 120]},
            {"t": "01,", "l": [82, 100, 95, 120]},
            {"t": "2025", "l": [97, 100, 130, 120]},
            {"t": "Jun", "l": [200, 100, 230, 120]},
            {"t": "01.", "l": [232, 100, 245, 120]},
            {"t": "2025", "l": [247, 100, 280, 120]},
            # Sibling that anchors the row.
            {"t": "ROWX", "l": [400, 100, 450, 120]},
        ],
    )
    phase1 = {
        "transactions": [
            {
                "post_date": {"text": "Jun 01, 2025", "locations": [{"page_number": 1}]},
                "due_date": {"text": "Jun 01, 2025", "locations": [{"page_number": 1}]},
                "id": {"text": "ROWX", "locations": [{"page_number": 1}]},
            }
        ]
    }
    layout = {
        "transactions": {"kind": "table", "columns": ["post_date", "due_date", "id"]},
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    # All three resolved by phase 2 — fuzzy gave due_date a Due-col
    # candidate it would otherwise have been blind to.
    assert result.unmatched == []
    pd_box = result.values["transactions"][0]["post_date"]["locations"][0]["bounding_box"]
    dd_box = result.values["transactions"][0]["due_date"]["locations"][0]["bounding_box"]
    assert pd_box[0] == 50  # Post col left
    assert dd_box[0] == 200  # Due col left


def test_cell_reordering_handles_sub_pixel_y_jitter(workspace: Workspace) -> None:
    """OCR can report two same-line words with sub-pixel-different
    tops (1402 vs 1403), and a naive ``(top, left)`` sort would flip
    them — which breaks ``_find_spans`` for any phrase that crosses
    the swap (e.g. "Pet Deposit" reordering to "Deposit Pet"). The
    fix is to sort within a cell by visual line first, then by x
    within the line. This test exercises that path directly."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # "Pet Deposit" with Deposit reading 1 unit higher than
            # Pet — same visual line in practice, but trips up a naive
            # (top, left) sort.
            {"t": "Pet", "l": [100, 103, 150, 130]},
            {"t": "Deposit", "l": [160, 102, 240, 130]},
        ],
    )
    phase1 = {
        "title": {"text": "Pet Deposit", "locations": [{"page_number": 1}]},
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)
    # Found and matched (single unique candidate), not unmatched.
    assert result.stats.matched_locations == 1
    box = result.values["title"]["locations"][0]["bounding_box"]
    # bbox spans both words horizontally — sanity check the span is
    # the whole "Pet Deposit", not just one token.
    assert box[0] == 100  # left (Pet's left edge)
    assert box[2] == 240  # right (Deposit's right edge)


def test_sibling_overlap_filter_rejects_substring_inside_resolved_field(
    workspace: Workspace,
) -> None:
    """A leaf's text may also appear *inside* another field's text on
    the same row (charge_code "Payment" appearing inside description
    "eCheck Payment ID ..."). Once description matches, its span owns
    those OCR words; the sibling-overlap filter then rejects the
    "Payment" candidate that's a sub-span of description and forces
    charge_code onto the standalone token in its own column."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # The row's transaction id — its own cell anchoring the row.
            {"t": "14611067", "l": [50, 100, 150, 120]},
            # Standalone "Payment" in the charge_code column. The wide
            # gap to the next token keeps it a distinct cell from the
            # description (cell-merger threshold is ~16 units here).
            {"t": "Payment", "l": [340, 100, 430, 120]},
            # Description cell: "eCheck Payment ID 1649463089 Captured".
            # Inter-word gaps are small enough that they all collapse
            # into one cell, separate from charge_code's cell.
            {"t": "eCheck", "l": [470, 100, 540, 120]},
            {"t": "Payment", "l": [550, 100, 640, 120]},
            {"t": "ID", "l": [660, 100, 680, 120]},
            {"t": "1649463089", "l": [700, 100, 850, 120]},
            {"t": "Captured", "l": [870, 100, 970, 120]},
        ],
    )
    phase1 = {
        "transactions": [
            {
                "transaction_id": {"text": "14611067", "locations": [{"page_number": 1}]},
                "charge_code": {"text": "Payment", "locations": [{"page_number": 1}]},
                "description": {
                    "text": "eCheck Payment ID 1649463089 Captured",
                    "locations": [{"page_number": 1}],
                },
            }
        ]
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)
    assert result.unmatched == []
    cc_box = result.values["transactions"][0]["charge_code"]["locations"][0]["bounding_box"]
    desc_box = result.values["transactions"][0]["description"]["locations"][0]["bounding_box"]
    # charge_code lands on the standalone token at left=340, NOT the
    # one inside the description span (which starts at left=470).
    assert cc_box[0] == 340
    # And description's span is the whole "eCheck ... Captured" range.
    assert desc_box[0] == 470


def test_layout_handles_non_contiguous_unresolved_columns(workspace: Workspace) -> None:
    """Layout = [A, B, C] where B resolves uniquely in the main loop and
    A, C collide on identical text. The layout pass must still pair
    A → leftmost surviving cluster and C → rightmost, even though the
    unresolved set isn't a contiguous prefix of the columns list."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            {"t": "Q", "l": [50, 100, 80, 120]},
            {"t": "B-uniq", "l": [200, 100, 260, 120]},
            {"t": "Q", "l": [400, 100, 430, 120]},
        ],
    )
    phase1 = {
        "rows": [
            {
                "a": {"text": "Q", "locations": [{"page_number": 1}]},
                "b": {"text": "B-uniq", "locations": [{"page_number": 1}]},
                "c": {"text": "Q", "locations": [{"page_number": 1}]},
            }
        ]
    }
    layout = {"rows": {"kind": "table", "columns": ["a", "b", "c"]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    assert result.stats.matched_locations == 3
    assert result.unmatched == []
    a_left = result.values["rows"][0]["a"]["locations"][0]["bounding_box"][0]
    c_left = result.values["rows"][0]["c"]["locations"][0]["bounding_box"][0]
    assert a_left == 50
    assert c_left == 400


def test_layout_skipped_when_field_not_in_columns(workspace: Workspace) -> None:
    """If a row has an unresolved task whose field isn't in the
    layout's ``columns`` list, the layout pass declines the whole
    row — partial layout info would risk mispicks for the listed
    columns."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            # Both X's on the same row, different columns. No sibling
            # anchors. Layout names only one of two unresolved fields.
            {"t": "X", "l": [50, 100, 80, 120]},
            {"t": "X", "l": [200, 100, 230, 120]},
        ],
    )
    phase1 = {
        "rows": [
            {
                "a": {"text": "X", "locations": [{"page_number": 1}]},
                "b": {"text": "X", "locations": [{"page_number": 1}]},
            }
        ]
    }
    # Layout omits 'b'.
    layout = {"rows": {"kind": "table", "columns": ["a"]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1, layout=layout)
    # Layout pass declines because 'b' isn't in columns; both stay
    # unmatched and roll over to phase 3.
    assert result.stats.unmatched_locations == 2


def test_array_layout_key_handles_path_shapes() -> None:
    assert _array_layout_key(("transactions", 0, "post_date")) == "transactions"
    assert _array_layout_key(("co", "contacts", 3, "name")) == "co.contacts"
    assert _array_layout_key(("outer", 0, "inner", 2, "field")) == "outer.inner"
    assert _array_layout_key(("scalar",)) is None
    assert _array_layout_key(("a", "b", "c")) is None


# A party block: name + entity_type share the first visual line (y≈110),
# the address sits on the next line (y≈160), well outside the first line's
# row band. Used by both free_form tests below.
_PARTY_BLOCK_WORDS = [
    {"t": "Acme", "l": [50, 100, 100, 120]},
    {"t": "Inc", "l": [110, 100, 160, 120]},
    {"t": "corporation", "l": [170, 100, 300, 120]},
    {"t": "123", "l": [50, 150, 90, 170]},
    {"t": "Main", "l": [100, 150, 160, 170]},
    {"t": "St", "l": [170, 150, 200, 170]},
]
_PARTY_BLOCK_PHASE1 = {
    "parties": [
        {
            "name": {"text": "Acme Inc", "locations": [{"page_number": 1}]},
            "entity_type": {"text": "corporation", "locations": [{"page_number": 1}]},
            "address": {"text": "123 Main St", "locations": [{"page_number": 1}]},
        }
    ]
}


def test_free_form_array_matches_fields_on_separate_lines(workspace: Workspace) -> None:
    """A free_form array element whose fields stack across visual lines —
    a party block with name/type on one line and the address on the next —
    resolves every field. The single-line row band and the row-coherence
    audit are both suspended for arrays the layout marks free_form, so the
    off-line address is neither band-filtered before matching nor demoted
    after (demotion would strip its only candidate; see _demote)."""
    _seed_file_and_text(workspace, "f1aaaaaaaaaa", page=1, words=_PARTY_BLOCK_WORDS)
    layout = {"parties": {"kind": "free_form"}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", _PARTY_BLOCK_PHASE1, layout=layout)

    assert result.stats.unmatched_locations == 0
    party = result.values["parties"][0]
    # Address resolved to its own line (top 150, bottom 170, left 50,
    # right 200), not the name's line. Box is [left, top, right, bottom].
    assert party["address"]["locations"] == [
        {"page_number": 1, "bounding_box": [50, 150, 200, 170]}
    ]


def test_non_table_array_reading_order_resolves_wrap_line(workspace: Workspace) -> None:
    """Without an explicit ``kind:"table"`` hint, the matcher treats the
    array as non-tabular: column-x logic is off, the row-coherence audit
    doesn't fire, and the reading-order region picks up off-line content
    inside the element. So a party block whose ``address`` sits on the
    line below ``name``/``entity_type`` resolves cleanly — the wrap line
    falls inside the element's region. (Contrast with the explicit
    ``free_form`` test above: same outcome, but the hint there spells
    out the intent rather than relying on the default.)"""
    _seed_file_and_text(workspace, "f1aaaaaaaaaa", page=1, words=_PARTY_BLOCK_WORDS)
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", _PARTY_BLOCK_PHASE1, layout=None)

    assert result.stats.unmatched_locations == 0
    assert result.values["parties"][0]["address"]["locations"] == [
        {"page_number": 1, "bounding_box": [50, 150, 200, 170]}
    ]


# ---- Precision guardrail -------------------------------------------------
#
# These pin the matcher's precision: each case is OCR that is *close to but
# not* the extracted value, where the only correct move is to leave the
# value unmatched (defer to phase 3) rather than commit a wrong bbox. They
# pass against today's strict exact-span matcher and exist to stay green
# when span matching is loosened (weighted edit-distance / fuzzy recall).
# A regression here means looser matching has started inventing matches —
# trading correctness for coverage, which is never the trade we want.
#
# Two failure modes are guarded:
#   - *content* differences a fuzzy matcher might paper over (a changed
#     digit, a dropped negation, a longer word the target only prefixes);
#   - *ambiguity*: when the same text appears twice with nothing to break
#     the tie, the unambiguous-only-commit rule must defer, not guess.

_MUST_NOT_MATCH = [
    pytest.param("1001", [{"t": "1000", "l": [100, 100, 160, 120]}], id="different-number"),
    pytest.param("5/6/2018", [{"t": "5/16/2018", "l": [100, 100, 220, 120]}], id="different-date"),
    pytest.param("03:45", [{"t": "03:46", "l": [100, 100, 160, 120]}], id="different-time"),
    pytest.param(
        "$1,250.00", [{"t": "$1,350.00", "l": [100, 100, 220, 120]}], id="different-amount"
    ),
    pytest.param("98033", [{"t": "98303", "l": [100, 100, 180, 120]}], id="transposed-digits"),
    pytest.param(
        "Smith", [{"t": "Smithson", "l": [100, 100, 220, 120]}], id="prefix-not-whole-word"
    ),
    pytest.param(
        "shall not disclose",
        [
            {"t": "shall", "l": [100, 100, 150, 120]},
            {"t": "disclose", "l": [160, 100, 260, 120]},
        ],
        id="dropped-negation",
    ),
    pytest.param(
        # Same value text appears twice, far apart, with no sibling/column
        # anchor to disambiguate — the matcher must defer, not pick one.
        "Confidential",
        [
            {"t": "Confidential", "l": [100, 100, 260, 120]},
            {"t": "Confidential", "l": [100, 500, 260, 520]},
        ],
        id="ambiguous-occurrences-defer",
    ),
]


@pytest.mark.parametrize("text,words", _MUST_NOT_MATCH)
def test_phase2_must_not_match(
    workspace: Workspace, text: str, words: list[dict[str, Any]]
) -> None:
    """The matcher must leave ``text`` unmatched against near-miss OCR —
    no wrong bbox committed. See the precision-guardrail note above."""
    _seed_file_and_text(workspace, "f1aaaaaaaaaa", page=1, words=words)
    phase1 = {"field": {"text": text, "locations": [{"page_number": 1}]}}
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)

    assert result.stats.matched_locations == 0
    assert result.values["field"]["locations"] == [{"page_number": 1}]


# ---------------------------------------------------------------------------
# computed leaves + path helpers


def test_computed_leaf_skipped_by_matcher(workspace: Workspace) -> None:
    """A computed leaf (no locations; computed/derived_from keys) is invisible
    to phase 2: it creates no task, no unmatched item, and survives the copy
    untouched."""
    _seed_file_and_text(
        workspace,
        "f1aaaaaaaaaa",
        page=1,
        words=[
            {"t": "Hello", "l": [100, 210, 182, 242]},
            {"t": "world", "l": [190, 210, 290, 242]},
        ],
    )
    phase1 = {
        "title": {"text": "Hello world", "locations": [{"page_number": 1}]},
        "word_count": {
            "text": "2",
            "value": "2",
            "computed": True,
            "derived_from": ["title"],
        },
    }
    result = run_phase2_matching(workspace, "f1aaaaaaaaaa", phase1)
    assert result.stats.total_locations == 1  # only the grounded leaf
    assert result.stats.matched_locations == 1
    assert result.unmatched == []
    assert result.values["word_count"] == phase1["word_count"]


def test_walk_computed_leaves_complements_walk_leaves() -> None:
    values = {
        "title": {"text": "Hello", "locations": [{"page_number": 1}]},
        "items": [
            {"qty": {"text": "3", "locations": [{"page_number": 2}]}},
            {"qty": {"text": "2", "computed": True, "derived_from": ["items[0].qty"]}},
        ],
    }
    grounded = {path for path, _ in _walk_leaves(values)}
    computed = {path for path, _ in walk_computed_leaves(values)}
    assert grounded == {("title",), ("items", 0, "qty")}
    assert computed == {("items", 1, "qty")}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("title", ("title",)),
        ("Invoice.LineItems[0].Quantity", ("Invoice", "LineItems", 0, "Quantity")),
        ("a.b[12].c[0]", ("a", "b", 12, "c", 0)),
        ("", None),
        ("not a [valid path", None),
        ("a..b", None),
    ],
)
def test_parse_path(text: str, expected: tuple[object, ...] | None) -> None:
    assert parse_path(text) == expected
    if expected is not None:
        assert parse_path(path_to_str(expected)) == expected
