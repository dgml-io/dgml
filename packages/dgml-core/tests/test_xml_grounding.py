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

"""Tests for whole-document DGML XML grounding (dgml_core.xml_grounding).

The fixtures fabricate the two streams the aligner consumes: page_text
JSONs with deterministic word boxes, and a small DGML-shaped XML tree.
Boxes are emitted in the project-wide pixel convention — each box is
``<page> <x1> <y1> <x2> <y2>`` (space-separated) in integer image pixels.
Pages are 1000x1000 px here, so a word at x=100 reads back as left=100.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dgml_core.errors import FileNotFound, GroundingFailed
from dgml_core.models import FileRecord
from dgml_core.storage import Workspace, write_json_atomic
from dgml_core.xml_grounding import (
    _lis_pairs,
    ground_dgml_xml,
    grounded_output_path,
)
from lxml import etree  # type: ignore[import-untyped]

FILE_ID = "f1aaaaaaaaaa"

# Word geometry: width 50px, gap 10px, line height 20px, lines start at
# x=100. Boxes are integer pixels, so a word at x=100 reads back as 100.
_WORD_W = 50
_WORD_GAP = 10
_LINE_H = 20


def _line(words: str, top: int) -> list[dict[str, Any]]:
    out = []
    x = 100
    for w in words.split():
        out.append({"t": w, "l": [x, top, x + _WORD_W, top + _LINE_H]})
        x += _WORD_W + _WORD_GAP
    return out


def _seed_pages(workspace: Workspace, pages: dict[int, list[dict[str, Any]]]) -> None:
    workspace.file_dir(FILE_ID).mkdir(parents=True, exist_ok=True)
    record = FileRecord(
        id=FILE_ID,
        original_path="/fake/contract.pdf",
        original_filename="contract.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=len(pages),
        text_mode="digital",
    )
    write_json_atomic(workspace.file_json_path(FILE_ID), record.to_json())
    workspace.file_text_dir(FILE_ID).mkdir(parents=True, exist_ok=True)
    for page, words in pages.items():
        write_json_atomic(
            workspace.file_text_dir(FILE_ID) / f"page_{page}.json",
            {"file_id": FILE_ID, "page": page, "width": 1000, "height": 1000, "words": words},
        )


def _standard_pages(workspace: Workspace) -> None:
    _seed_pages(
        workspace,
        {
            1: (
                _line("Master Services Agreement", 100)
                + _line("This agreement is between Acme Corporation and Zenith Ltd", 200)
                + _line("1. Payment Terms", 300)
                + _line("Payment is due within 30 days of invoice", 400)
            ),
            2: (
                _line("2. Termination", 100)
                + _line("Either party may terminate with 60 days notice", 200)
                + _line("Signed by John Smith CEO", 300)
            ),
        },
    )


_STANDARD_XML = """\
<root>
  <Title>Master Services Agreement</Title>
  <Intro>This agreement is between <Company>Acme Corporation</Company> \
and <Client>Zenith Ltd</Client></Intro>
  <Section>
    <lim>1.</lim>
    <Heading>Payment Terms</Heading>
    <Body>Payment is due within 30 days of invoice</Body>
  </Section>
  <Section>
    <lim>2.</lim>
    <Heading>Termination</Heading>
    <Body>Either party may terminate with 60 days notice</Body>
  </Section>
  <Signature>Signed by John Smith CEO</Signature>
</root>
"""


def _ground(workspace: Workspace, tmp_path: Path, xml: str, **kwargs: Any) -> Any:
    src = tmp_path / "contract.dgml.xml"
    src.write_text(xml, encoding="utf-8")
    return ground_dgml_xml(workspace, FILE_ID, src, **kwargs)


def _boxes(root: Any, tag: str) -> list[str]:
    return [el.get("origin") for el in root.iter(tag)]


def _parse_box(box: str) -> tuple[int, int, int, int, int]:
    """(page, left, top, right, bottom) ints from one pixel box string
    ``<page> <x1> <y1> <x2> <y2>`` (space-separated)."""
    page, left, top, right, bottom = (int(n) for n in box.split())
    return page, left, top, right, bottom


def test_grounds_full_document(workspace: Workspace, tmp_path: Path) -> None:
    _standard_pages(workspace)
    res = _ground(workspace, tmp_path, _STANDARD_XML)

    assert res.output_path == tmp_path / "contract.dgml.grounded.xml"
    assert res.output_path.exists()
    assert res.stats_path == tmp_path / "contract.dgml.grounding_stats.json"
    assert res.stats_path.exists()
    assert res.stats["matched_token_pct"] == 100.0
    assert res.stats["text_nodes"]["ungrounded"] == 0

    root = etree.parse(str(res.output_path)).getroot()

    # Title: leaf on page 1, line at top=100 spanning 3 words from x=100.
    (title,) = _boxes(root, "Title")
    # 3 words: 100..150, 160..210, 220..270 → left=100, right=270; top=100, bottom=120
    assert title == "1 100 100 270 120"

    # Section 2's Body grounds on page 2.
    bodies = _boxes(root, "Body")
    assert bodies[0].startswith("1 ")
    assert bodies[1].startswith("2 ")

    # Signature: page 2, line at top=300.
    (sig,) = _boxes(root, "Signature")
    assert sig.startswith("2 ")
    assert _parse_box(sig)[2] == 300  # top


def test_write_stats_false_skips_sidecar(workspace: Workspace, tmp_path: Path) -> None:
    """write_stats=False (set by the CLI unless --debug) produces the grounded
    XML but no stats sidecar, and leaves stats_path None."""
    _standard_pages(workspace)
    res = _ground(workspace, tmp_path, _STANDARD_XML, write_stats=False)

    assert res.output_path.exists()
    assert res.stats_path is None
    assert not (tmp_path / "contract.dgml.grounding_stats.json").exists()
    # The stats dict is still returned in-memory for callers that want it.
    assert res.stats["matched_token_pct"] == 100.0


def test_mixed_content_parent_gets_line_boxes(workspace: Workspace, tmp_path: Path) -> None:
    """An element with both text and element children is annotated with
    per-line boxes covering its whole subtree, same as a leaf."""
    _standard_pages(workspace)
    res = _ground(workspace, tmp_path, _STANDARD_XML)
    root = etree.parse(str(res.output_path)).getroot()

    (intro,) = _boxes(root, "Intro")
    # Whole intro line: 9 words at top=200 → left=100, top=200, bottom=220,
    # right = 100 + 9*50 + 8*10 = 630px. One visual line → one box.
    assert intro == "1 100 200 630 220"
    # Inline children are leaves with their own tighter boxes.
    (company,) = _boxes(root, "Company")
    assert company.startswith("1 ")
    assert company != intro


def test_multiline_mixed_content_gets_one_box_per_line(
    workspace: Workspace, tmp_path: Path
) -> None:
    """Regression: a wrapped paragraph with inline children must carry
    one box per visual line (uniform with childless paragraphs), not a
    single page-level union rectangle — and its tail text (the words
    between/after inline children) must ground."""
    _seed_pages(
        workspace,
        {
            1: (
                _line("This agreement continues until December 31 2025 and renews", 100)
                + _line("annually for one year terms unless notice is given", 200)
                + _line("by the Company before the end of the term", 300)
            )
        },
    )
    xml = """\
<root>
  <Clause>This agreement continues until <Date>December 31 2025</Date> and renews \
annually for <Term>one year terms</Term> unless notice is given by the Company \
before the end of the term</Clause>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    assert res.stats["matched_token_pct"] == 100.0
    root = etree.parse(str(res.output_path)).getroot()

    (clause,) = _boxes(root, "Clause")
    rects = clause.split("; ")
    # Three visual lines → three boxes, each one line-height tall (20px),
    # not one tall union covering the whole paragraph block.
    assert len(rects) == 3
    assert [_parse_box(r)[2] for r in rects] == [100, 200, 300]  # top per line
    assert all(_parse_box(r)[4] - _parse_box(r)[2] == 20 for r in rects)  # one line tall
    # Inline children still carry their own tighter boxes.
    (date,) = _boxes(root, "Date")
    assert date.startswith("1 ")


def test_pure_container_elements_get_page_union_boxes(workspace: Workspace, tmp_path: Path) -> None:
    """Elements with no text-node children (Section, root) are annotated
    with one union box per page covering their grounded subtree."""
    _standard_pages(workspace)
    res = _ground(workspace, tmp_path, _STANDARD_XML)
    root = etree.parse(str(res.output_path)).getroot()

    # Section 1 sits entirely on page 1: lines at top=300 and top=400.
    sections = list(root.iter("Section"))
    s1 = sections[0].get("origin")
    assert s1 is not None and "; " not in s1  # one page → one union box
    page, _left, top, _right, bottom = _parse_box(s1)
    assert (page, top, bottom) == (1, 300, 420)
    # Section 2 is on page 2.
    assert sections[1].get("origin").startswith("2 ")

    # The root spans both pages: one union box per page.
    root_boxes = root.get("origin").split("; ")
    assert [_parse_box(b)[0] for b in root_boxes] == [1, 2]
    assert res.stats["containers_annotated"] >= 3  # root + two Sections


def test_dg_namespace_attribute_when_declared(workspace: Workspace, tmp_path: Path) -> None:
    """A document binding the ``dg`` prefix gets ``dg:origin`` (qualified to
    whatever URI the prefix maps to — here the open dgml.io scheme that
    generated DGML uses); the plain fixture above gets a no-namespace
    ``origin`` attribute."""
    _standard_pages(workspace)
    xml = _STANDARD_XML.replace("<root>", '<root xmlns:dg="http://dgml.io">')
    res = _ground(workspace, tmp_path, xml)
    content = res.output_path.read_text(encoding="utf-8")
    assert 'dg:origin="1 ' in content
    assert " origin=" not in content


def test_existing_output_requires_force(workspace: Workspace, tmp_path: Path) -> None:
    _standard_pages(workspace)
    res = _ground(workspace, tmp_path, _STANDARD_XML)
    first = res.output_path.read_bytes()

    with pytest.raises(GroundingFailed):
        _ground(workspace, tmp_path, _STANDARD_XML)

    res2 = _ground(workspace, tmp_path, _STANDARD_XML, force=True)
    assert res2.output_path.read_bytes() == first


def test_missing_page_text_raises(workspace: Workspace, tmp_path: Path) -> None:
    workspace.file_dir(FILE_ID).mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFound):
        _ground(workspace, tmp_path, _STANDARD_XML)


def test_hallucinated_text_stays_ungrounded(workspace: Workspace, tmp_path: Path) -> None:
    """Text the page doesn't show gets no box and lands in the stats."""
    _standard_pages(workspace)
    xml = _STANDARD_XML.replace(
        "<Signature>Signed by John Smith CEO</Signature>",
        "<Signature>Signed by John Smith CEO</Signature>"
        "\n  <Fabricated>completely invented sentence nowhere on any page</Fabricated>",
    )
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()

    (fab,) = list(root.iter("Fabricated"))
    assert fab.get("origin") is None
    assert res.stats["text_nodes"]["ungrounded"] == 1
    assert any("Fabricated" in u["path"] for u in res.stats["top_ungrounded"])
    # Everything else still grounds.
    (sig,) = _boxes(root, "Signature")
    assert sig.startswith("2 ")


def test_ocr_letter_slip_recovered(workspace: Workspace, tmp_path: Path) -> None:
    """An OCR letter error ('Aqreement') breaks exact token equality but
    the gap-recovery similarity pass still grounds the run."""
    _seed_pages(
        workspace,
        {
            1: (
                _line("Master Services Aqreement of January", 100)
                + _line("Payment is due within 30 days of invoice", 200)
            )
        },
    )
    xml = """\
<root>
  <Title>Master Services Agreement of January</Title>
  <Body>Payment is due within 30 days of invoice</Body>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()
    (title,) = _boxes(root, "Title")
    assert title is not None and title.startswith("1 ")
    assert res.stats["recovered_tokens"] >= 1


def test_recovery_run_spanning_cells_does_not_merge_columns(
    workspace: Workspace, tmp_path: Path
) -> None:
    """Regression (the Hatfield ledger column merge): when one unmatched
    run covers two adjacent table cells (both words OCR-slipped), gap
    recovery must give each cell its own word — not smear the whole gap
    onto both, which painted one cell's box across its neighbor's
    column."""
    _seed_pages(
        workspace,
        {
            1: (
                _line("Code Description Charge Balance", 100)
                # OCR letter slips in BOTH adjacent description cells:
                + _line("Pet Deposlt Darnages 1,063.97", 200)
            )
        },
    )
    xml = """\
<root>
  <Header>Code Description Charge Balance</Header>
  <Row><A>Pet</A> <B>Deposit</B> <C>Damages</C> <D>1,063.97</D></Row>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()
    assert res.stats["recovered_tokens"] == 2

    # Word x-ranges on line 2: Pet=100-150, Deposlt=160-210, Darnages=220-270.
    (b,) = _boxes(root, "B")
    (c,) = _boxes(root, "C")
    _, bl, _, br, _ = _parse_box(b)
    _, cl, _, cr, _ = _parse_box(c)
    assert (bl, br) == (160, 210)  # B covers only "Deposlt"
    assert (cl, cr) == (220, 270)  # C covers only "Darnages", not both
    assert b != c


def test_repeated_block_rescued(workspace: Workspace, tmp_path: Path) -> None:
    """Content the XML repeats but the page shows once: the aligner can
    consume the page words only once; the rescue pass grounds the second
    copy to the same region."""
    _standard_pages(workspace)
    xml = _STANDARD_XML.replace(
        "<Title>Master Services Agreement</Title>",
        "<Title>Master Services Agreement</Title>"
        "\n  <CoverTitle>Master Services Agreement</CoverTitle>",
    )
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()

    (title,) = _boxes(root, "Title")
    (cover,) = _boxes(root, "CoverTitle")
    assert cover == title  # both point at the one place the page shows it
    assert res.stats["text_nodes"]["ungrounded"] == 0


def test_rescue_prefers_unclaimed_occurrence(workspace: Workspace, tmp_path: Path) -> None:
    """Regression (the page-7 'Software' bug): a one-word table cell
    whose row serializes in a different local order than OCR falls out
    of the monotonic alignment and reaches the rescue pass. The page
    shows the same word twice — once in prose (already claimed by the
    paragraph) and once as the actual table label. Rescue must ground
    the cell to the unclaimed table-label occurrence, not the first
    span on the page."""
    _seed_pages(
        workspace,
        {
            1: (
                _line("The purpose of this SOW is to describe the Software and", 100)
                + _line("Software MagicSoft Mobile fifty seats", 200)
                + _line("Total 4500 dollars", 300)
            )
        },
    )
    # XML serializes the fee row item-before-label — locally reordered
    # vs the OCR line, so the aligner can't place "Software" in order.
    xml = """\
<root>
  <Body>The purpose of this SOW is to describe the Software and</Body>
  <Fees>
    <Item>MagicSoft Mobile fifty seats</Item>
    <Category>Software</Category>
    <Total>4500 dollars</Total>
  </Fees>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()

    # The table label sits at line 2 word 0: left=100, top=200. The prose
    # occurrence (word 9 of line 1) would be at left=640, top=100.
    (category,) = _boxes(root, "Category")
    assert category == "1 100 200 150 220"
    assert res.stats["rescued_tokens"] >= 1


def test_feature_table_punct_cells_and_row_pinning(workspace: Workspace, tmp_path: Path) -> None:
    """Regression (the page-8 feature-comparison table): '?' cells have
    no alphanumeric tokens, so they're invisible to alignment — the
    positional pass must still ground them to the '?' word in their own
    row. And the repetitive Yes/No cells must stay pinned to their rows
    (unique row labels anchor every window level)."""
    _seed_pages(
        workspace,
        {
            1: (
                _line("Product Grammar Spell Math", 100)
                + _line("CopyDesk No ? No", 200)
                + _line("FrameMaker No Yes ?", 300)
                + _line("TeXmacs Yes No Yes", 400)
            )
        },
    )
    xml = """\
<root>
  <Header>Product Grammar Spell Math</Header>
  <Row><Name>CopyDesk</Name> <G>No</G> <S>?</S> <M>No</M></Row>
  <Row><Name>FrameMaker</Name> <G>No</G> <S>Yes</S> <M>?</M></Row>
  <Row><Name>TeXmacs</Name> <G>Yes</G> <S>No</S> <M>Yes</M></Row>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()

    # Each row's Grammar cell grounds in its OWN row's y band
    # (y = 20/30/40%), even though 'No' repeats everywhere.
    g_ys = [_parse_box(b)[2] for b in _boxes(root, "G")]
    assert g_ys == [200, 300, 400]
    m_ys = [_parse_box(b)[2] for b in _boxes(root, "M")]
    assert m_ys == [200, 300, 400]

    # The '?' cells ground positionally to their own row and column.
    s_boxes = _boxes(root, "S")
    assert s_boxes[0] is not None and _parse_box(s_boxes[0])[2] == 200  # CopyDesk ?
    (fm_m,) = [_boxes(root, "M")[1]]
    assert _parse_box(fm_m)[2] == 300  # FrameMaker row's ? in Math col
    assert res.stats["punct_grounded_nodes"] == 2
    # All token-bearing nodes grounded too.
    assert res.stats["text_nodes"]["ungrounded"] == 0


def test_interleaved_multiline_cells_ground_via_subsequence(
    workspace: Workspace, tmp_path: Path
) -> None:
    """Adjacent two-line table cells whose first-line words nearly touch
    merge into one component in reading-order cell building, so the OCR
    stream interleaves their lines ("Monthly Base Annual Base / Rent
    Fee") and neither cell's text is contiguous. The row-context
    subsequence pass must still ground both cells — each to its own
    words, in column order — and the boxes must stay per-cell."""
    _seed_pages(
        workspace,
        {
            1: [
                # Title and footer give the aligner solid anchors.
                *_line("Rent Roll Header", 20),
                # Header row: "Unit" is its own cell; the next two cells
                # are two lines each, and their line-1 words are 10px
                # apart (under the ~16px h-gap tolerance) so the cell
                # builder merges them into one component.
                {"t": "Unit", "l": [20, 100, 70, 120]},
                {"t": "Monthly", "l": [100, 100, 150, 120]},
                {"t": "Base", "l": [160, 100, 210, 120]},
                {"t": "Annual", "l": [220, 100, 270, 120]},
                {"t": "Base", "l": [280, 100, 330, 120]},
                {"t": "Rent", "l": [100, 125, 150, 145]},
                {"t": "Fee", "l": [220, 125, 270, 145]},
                *_line("End of report", 300),
            ]
        },
    )
    xml = """\
<root>
  <Title>Rent Roll Header</Title>
  <Table>
    <Row>
      <Unit>Unit</Unit>
      <ColA>Monthly Base Rent</ColA>
      <ColB>Annual Base Fee</ColB>
    </Row>
  </Table>
  <Footer>End of report</Footer>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    assert res.stats["matched_token_pct"] == 100.0
    root = etree.parse(str(res.output_path)).getroot()

    (col_a,) = _boxes(root, "ColA")
    (col_b,) = _boxes(root, "ColB")
    # Each cell: two visual lines → two boxes, confined to its own column.
    a_boxes = [_parse_box(b) for b in col_a.split("; ")]
    b_boxes = [_parse_box(b) for b in col_b.split("; ")]
    assert [(b[2], b[4]) for b in a_boxes] == [(100, 120), (125, 145)]
    assert [(b[2], b[4]) for b in b_boxes] == [(100, 120), (125, 145)]
    assert max(b[3] for b in a_boxes) == 210  # ColA never crosses into ColB
    assert min(b[1] for b in b_boxes) == 220
    assert res.stats["interleaved_cell_tokens"] > 0


def test_digit_discrepant_date_grounds_via_row_context(
    workspace: Workspace, tmp_path: Path
) -> None:
    """Regression (the Hatfield missing Post/Due dates): the generator
    transcribed a date digit differently than OCR read it ("Jun 18" vs
    page "Jun 16"). Search passes rightly refuse digit mismatches, but
    within the already-grounded row the location isn't in question —
    the digit-masked row-context pass must ground both date cells to
    their own columns."""
    _seed_pages(
        workspace,
        {
            1: (
                _line("Post Due Month Id Code Charge", 100)
                + _line("Jun 16, 2025 Jun 16, 2025 06/2025 15617476 Utilities $5.07", 200)
            )
        },
    )
    xml = """\
<root>
  <Header>Post Due Month Id Code Charge</Header>
  <Row><Post>Jun 18, 2025</Post> <Due>Jun 18, 2025</Due> <Month>06/2025</Month> \
<Id>15617476</Id> <Code>Utilities</Code> <Charge>$5.07</Charge></Row>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()

    # Each date was partially matched (jun/2025 aligned, the digit
    # didn't) — the stat counts newly-gained tokens: one per date.
    assert res.stats["shape_matched_tokens"] == 2
    # Line-2 word x positions: Jun=100, 16,=160, 2025=220 | Jun=280, 16,=340, 2025=400.
    (post,) = _boxes(root, "Post")
    (due,) = _boxes(root, "Due")
    assert post is not None and due is not None
    assert _parse_box(post)[1] == 100  # first date occurrence
    assert _parse_box(due)[1] == 280  # second — not the same one
    assert res.stats["text_nodes"]["ungrounded"] == 0


def test_pure_numeric_grounds_only_when_shape_unique(workspace: Workspace, tmp_path: Path) -> None:
    """Pure-numeric text (a transaction id, an amount) has no letters
    to pin identity, so digit-masked matching applies only when the row
    window offers exactly ONE unclaimed same-shape word — uniqueness is
    the anchor (the Hatfield page-1 transaction ids: XML '19817479' vs
    page '15617479')."""
    _seed_pages(
        workspace,
        {1: (_line("Id Code Charge Balance", 100) + _line("15617479 Utilities 5.07 22.65", 200))},
    )
    # XML id and charge each differ from the page by a digit; each has
    # a unique shape in the row (8 digits / N.NN), so both ground.
    xml = """\
<root>
  <Header>Id Code Charge Balance</Header>
  <Row><Id>19817479</Id> <Code>Utilities</Code> <Charge>9.99</Charge> \
<Balance>22.65</Balance></Row>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()

    # Line-2 x: 15617479=100, Utilities=160, 5.07=220, 22.65=280.
    (idbox,) = _boxes(root, "Id")
    (charge,) = _boxes(root, "Charge")
    assert idbox is not None and _parse_box(idbox)[1] == 100
    assert charge is not None and _parse_box(charge)[1] == 220
    assert res.stats["text_nodes"]["ungrounded"] == 0


def test_pure_numeric_refuses_ambiguous_shapes(workspace: Workspace, tmp_path: Path) -> None:
    """Precision guard: two same-shape amounts in the row (charge vs
    balance) — a wrong amount staying ungrounded beats it pointing at
    the other column."""
    _seed_pages(
        workspace,
        {1: (_line("Code Charge Balance", 100) + _line("Utilities 666.32 688.97", 200))},
    )
    # BOTH amounts transcribed wrong — neither page word is claimed, so
    # each seg sees two unclaimed same-shape candidates and must refuse.
    xml = """\
<root>
  <Header>Code Charge Balance</Header>
  <Row><Code>Utilities</Code> <Charge>999.99</Charge> <Balance>111.11</Balance></Row>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()

    (charge,) = _boxes(root, "Charge")
    (balance,) = _boxes(root, "Balance")
    assert charge is None  # 666.32 and 688.97 share the masked shape — refuse
    assert balance is None
    assert res.stats["text_nodes"]["ungrounded"] == 2


def test_numeric_shape_keeps_currency_structure(workspace: Workspace, tmp_path: Path) -> None:
    """Regression (the Hatfield $22.85→'2025' mis-ground): a dollar
    amount must not shape-match a bare year just because both are four
    digits — the '$' is part of the shape. The fragmented amount on the
    page ($22 / . / 65) is assembled as a multi-word span instead."""
    _seed_pages(
        workspace,
        {
            1: (
                _line("Date Code Balance", 100)
                # The amount is OCR-fragmented; '2025' is a same-digit-count trap.
                + _line("2025 Utilities $22 . 65", 200)
            )
        },
    )
    xml = """\
<root>
  <Header>Date Code Balance</Header>
  <Row><Date>2025</Date> <Code>Utilities</Code> <Balance>$22.85</Balance></Row>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()

    # Line-2 x: 2025=100, Utilities=160, $22=220, .=280, 65=340.
    (balance,) = _boxes(root, "Balance")
    assert balance is not None
    assert _parse_box(balance)[1] == 220  # the $22…65 span, NOT the year at left=100


def test_fragmented_amount_box_covers_currency_symbols(
    workspace: Workspace, tmp_path: Path
) -> None:
    """Regression (the Hatfield $38.70 split box): shattering OCR
    splits an amount into fragments; the numerals match but the '$'
    and '.' fragments aren't in the alignment stream, so the box
    clipped them. The absorption pass must pull adjacent punctuation
    fragments into the box — without grabbing the next column's."""
    _seed_pages(
        workspace,
        {
            1: (
                _line("Code Charge Balance", 100)
                # Amount shattered into fragments: $ 38 . 70 — and a far
                # parenthesis fragment belonging to the balance.
                + _line("Utilities $ 38 . 70 ($ 1272 )", 200)
            )
        },
    )
    xml = """\
<root>
  <Header>Code Charge Balance</Header>
  <Row><Code>Utilities</Code> <Charge>$38.70</Charge> <Balance>($1272)</Balance></Row>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    root = etree.parse(str(res.output_path)).getroot()

    # Line-2 x: Utilities=100, $=160, 38=220, .=280, 70=340, ($=400, 1272=460, )=520.
    (charge,) = _boxes(root, "Charge")
    assert charge is not None
    _, left, _, right, _ = _parse_box(charge)
    assert left == 160  # box starts at the '$' fragment
    assert right <= 400  # and does NOT absorb the balance's '($'
    (balance,) = _boxes(root, "Balance")
    assert _parse_box(balance)[1] == 400  # balance box starts at its '($'
    assert res.stats["absorbed_fragments"] >= 3


def test_short_cell_outlier_words_pruned(workspace: Workspace, tmp_path: Path) -> None:
    """Unit: a cell-sized segment keeps only its dominant x-cluster of
    box words — a stray fragment a column away is attribution noise."""
    from dgml_core.textmatch import PageDims, Word
    from dgml_core.xml_grounding import _prune_outlier_words, _TextSeg

    def word(idx: int, left: int, right: int, top: int = 100) -> Word:
        return Word(
            idx=idx, text="x", text_norm="x", left=left, top=top, right=right, bottom=top + 20
        )

    seg = _TextSeg(owner=None, raw="$35.00", n_tokens=1, matched_tokens=1)
    seg.matched_words = [(1, word(0, 620, 650)), (1, word(1, 655, 680)), (1, word(2, 740, 760))]
    pruned = _prune_outlier_words([seg], {1: PageDims(1000, 1000)})
    assert pruned == 1
    assert [w.idx for _p, w in seg.matched_words] == [0, 1]  # stray at x74 dropped

    # No dominant cluster (1 vs 1) — leave ambiguity alone.
    seg2 = _TextSeg(owner=None, raw="ab", n_tokens=2, matched_tokens=2)
    seg2.matched_words = [(1, word(0, 100, 150)), (1, word(1, 700, 750))]
    assert _prune_outlier_words([seg2], {1: PageDims(1000, 1000)}) == 0
    assert len(seg2.matched_words) == 2

    # Long segments are untouched even when spread wide.
    seg3 = _TextSeg(owner=None, raw="a b c d e f", n_tokens=6, matched_tokens=6)
    seg3.matched_words = [(1, word(i, 100 + i * 150, 140 + i * 150)) for i in range(6)]
    assert _prune_outlier_words([seg3], {1: PageDims(1000, 1000)}) == 0

    # A wrapped inline span jumps from line-end to line-start: its
    # continuation words sit far away in x but on the NEXT line —
    # layout, not noise. Nothing may be pruned.
    seg4 = _TextSeg(owner=None, raw="Office and Laboratory", n_tokens=3, matched_tokens=3)
    seg4.matched_words = [
        (1, word(0, 800, 880)),  # "Office" at line end
        (1, word(1, 100, 140, top=125)),  # "and" at next line start
        (1, word(2, 150, 260, top=125)),  # "Laboratory"
    ]
    assert _prune_outlier_words([seg4], {1: PageDims(1000, 1000)}) == 0
    assert len(seg4.matched_words) == 3


def test_signature_rule_does_not_capture_name_below(workspace: Workspace, tmp_path: Path) -> None:
    """A signature rule (underscore run) x-overlaps the typed name
    below it; as a decorative word it must not union with that name —
    otherwise the name is pulled into the rule's earlier band and
    "Julian Kase" serializes reversed, ungroundable by any pass."""
    _seed_pages(
        workspace,
        {
            1: [
                *_line("Witness block follows", 20),
                {"t": "Name", "l": [100, 200, 150, 220]},
                {"t": ":", "l": [150, 200, 155, 220]},
                {"t": "Julian", "l": [300, 200, 350, 220]},
                # Rule 2px above the last name, x-overlapping it.
                {"t": "___________", "l": [500, 178, 800, 198]},
                {"t": "Kase", "l": [700, 200, 750, 220]},
                *_line("End of block", 300),
            ]
        },
    )
    xml = """\
<root>
  <Header>Witness block follows</Header>
  <L>Name:</L>
  <Sig>Julian Kase</Sig>
  <Footer>End of block</Footer>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    assert res.stats["matched_token_pct"] == 100.0
    root = etree.parse(str(res.output_path)).getroot()
    (sig,) = _boxes(root, "Sig")
    page, left, top, right, bottom = _parse_box(sig)
    assert (page, left, top, right, bottom) == (1, 300, 200, 750, 220)


def test_partial_cell_completed_along_its_own_line(workspace: Workspace, tmp_path: Path) -> None:
    """A name typeset with a wide mid-line gap ("Tarek <gap> Vargas")
    whose first word merges into the label block below it: cell
    connectivity can't accept the pick (the words are far apart), but
    the matched word pins the visual line and the same-line completion
    grounds the rest. The pinned result must survive outlier pruning."""
    _seed_pages(
        workspace,
        {
            1: [
                *_line("Signature page follows", 20),
                {"t": "Name", "l": [100, 200, 150, 220]},
                {"t": ":", "l": [150, 200, 155, 220]},
                {"t": "Tarek", "l": [400, 200, 450, 220]},
                # Title line below; "Ops" x-overlaps "Tarek", merging it
                # into the left block so "Tarek Vargas" is never
                # contiguous in reading order.
                {"t": "Title", "l": [100, 225, 150, 245]},
                {"t": ":", "l": [150, 225, 155, 245]},
                {"t": "Director", "l": [160, 225, 240, 245]},
                {"t": "of", "l": [250, 225, 330, 245]},
                {"t": "Ops", "l": [340, 225, 440, 245]},
                {"t": "Vargas", "l": [700, 200, 760, 220]},
                *_line("End of block", 320),
            ]
        },
    )
    xml = """\
<root>
  <Header>Signature page follows</Header>
  <L>Name:</L>
  <Sig>Tarek Vargas</Sig>
  <T>Title: Director of Ops</T>
  <Footer>End of block</Footer>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    assert res.stats["matched_token_pct"] == 100.0
    root = etree.parse(str(res.output_path)).getroot()
    (sig,) = _boxes(root, "Sig")
    page, left, top, right, bottom = _parse_box(sig)
    # One visual line box spanning both name words, not clipped to either.
    assert (page, left, top, right, bottom) == (1, 400, 200, 760, 220)


def test_fragment_absorption_precedes_outlier_pruning(workspace: Workspace, tmp_path: Path) -> None:
    """ "Cash & Liquid Credits": the '&' is invisible to alignment, so
    before absorption the Cash→Liquid gap spans it and looks like an
    outlier split — pruning must run after absorption fills the gap,
    keeping "Cash" in the box."""
    _seed_pages(
        workspace,
        {
            1: [
                *_line("Balance sheet follows", 20),
                {"t": "Cash", "l": [100, 200, 150, 220]},
                {"t": "&", "l": [160, 200, 180, 220]},
                {"t": "Liquid", "l": [190, 200, 240, 220]},
                {"t": "Credits", "l": [250, 200, 300, 220]},
                # A far column on the same line (real gutter).
                {"t": "142.50", "l": [700, 200, 760, 220]},
                *_line("End of statement", 300),
            ]
        },
    )
    xml = """\
<root>
  <Header>Balance sheet follows</Header>
  <Row><C>Cash &amp; Liquid Credits</C><V>142.50</V></Row>
  <Footer>End of statement</Footer>
</root>
"""
    res = _ground(workspace, tmp_path, xml)
    assert res.stats["matched_token_pct"] == 100.0
    root = etree.parse(str(res.output_path)).getroot()
    (cell,) = _boxes(root, "C")
    page, left, _top, right, _bottom = _parse_box(cell)
    assert (page, left, right) == (1, 100, 300)  # Cash through Credits, & included


def test_words_form_one_cell_row_gap_bound() -> None:
    """Unit: vertical connectivity accepts wrapped-cell leading (well
    under one word height) and rejects table-row padding (a full height
    or more) — the bound that keeps a header cell from unioning with
    the data row below it."""
    from dgml_core.textmatch import Word, words_form_one_cell

    def word(idx: int, top: int, left: int = 100) -> Word:
        return Word(
            idx=idx, text="x", text_norm="x", left=left, top=top, right=left + 60, bottom=top + 20
        )

    # Two lines 8px apart (0.4 * height): one wrapped cell.
    assert words_form_one_cell([word(0, 100), word(1, 128)])
    # Two lines 24px apart (1.2 * height): separate rows.
    assert not words_form_one_cell([word(0, 100), word(1, 144)])


def test_grounded_output_path_naming() -> None:
    assert grounded_output_path(Path("/x/a.dgml.xml")) == Path("/x/a.dgml.grounded.xml")
    assert grounded_output_path(Path("/x/a.xml")) == Path("/x/a.grounded.xml")


def test_lis_pairs_drops_non_monotonic() -> None:
    # (x, y) pairs sorted by x; the y=500 outlier breaks monotonicity
    # and is the unique pair every longest chain must drop.
    pairs = [(1, 10), (2, 500), (3, 20), (4, 30)]
    assert _lis_pairs(pairs) == [(1, 10), (3, 20), (4, 30)]


def test_lis_pairs_empty_and_single() -> None:
    assert _lis_pairs([]) == []
    assert _lis_pairs([(7, 7)]) == [(7, 7)]


def _styled_line(
    words: str, top: int, *, style: dict[str, Any] | None = None, size: float | None = None
) -> list[dict[str, Any]]:
    """``_line`` plus a per-word ``"s"`` style object (the digital path's shape)."""
    out = _line(words, top)
    s = dict(style or {})
    if size is not None:
        s["sz"] = size
    if s:
        for w in out:
            w["s"] = s
    return out


def test_dg_style_emitted_from_word_facts(workspace: Workspace, tmp_path: Path) -> None:
    """A bold, larger, uppercase heading gets dg:style from the matched words'
    facts; the plain body line (baseline size) gets none."""
    _seed_pages(
        workspace,
        {
            1: (
                _styled_line("BOLD BIG TITLE", 100, style={"b": 1}, size=24.0)
                + _styled_line("normal body paragraph text goes here", 200, size=12.0)
            )
        },
    )
    xml = (
        "<root>\n"
        "  <Heading>BOLD BIG TITLE</Heading>\n"
        "  <Body>normal body paragraph text goes here</Body>\n"
        "</root>\n"
    )
    res = _ground(workspace, tmp_path, xml, write_stats=False)
    root = etree.parse(str(res.output_path)).getroot()

    (heading,) = list(root.iter("Heading"))
    style = heading.get("style")
    assert style is not None
    assert "font-weight: bold" in style
    assert "font-size: 2em" in style  # 24pt vs 12pt baseline
    assert "text-transform: uppercase" in style

    (body,) = list(root.iter("Body"))
    assert body.get("style") is None  # nothing observable -> omitted


def test_dg_style_uses_dg_namespace_when_declared(workspace: Workspace, tmp_path: Path) -> None:
    """dg:style is qualified to the document's dg URI, like dg:origin."""
    _seed_pages(workspace, {1: _styled_line("BOLDWORD HERE NOW", 100, style={"b": 1}, size=12.0)})
    xml = (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><Heading>BOLDWORD HERE NOW</Heading></dg:chunk>'
    )
    res = _ground(workspace, tmp_path, xml, write_stats=False)
    content = res.output_path.read_text(encoding="utf-8")
    assert 'dg:style="font-weight: bold' in content
    assert " style=" not in content  # never the bare attribute when dg is bound


def test_dg_style_child_does_not_repeat_inherited_color(
    workspace: Workspace, tmp_path: Path
) -> None:
    """A whole line rendered red: color lands on the enclosing paragraph (its
    own tail text is red too), and the inner span does NOT restate it — the
    child inherits it. Mirrors the 03_colors regression."""
    _seed_pages(
        workspace, {1: _styled_line("Red warning text", 100, style={"c": "red"}, size=12.0)}
    )
    xml = "<root><Warning><ColorName>Red</ColorName> warning text</Warning></root>"
    res = _ground(workspace, tmp_path, xml, write_stats=False)
    root = etree.parse(str(res.output_path)).getroot()
    (warning,) = list(root.iter("Warning"))
    (color_name,) = list(root.iter("ColorName"))
    assert warning.get("style") == "color: red"  # kept where it first appears
    assert color_name.get("style") is None  # redundant inherited value dropped


def test_dg_style_child_overrides_inherited_bold(workspace: Workspace, tmp_path: Path) -> None:
    """A normal child inside a bold-styled mixed-content element keeps an
    explicit ``font-weight: normal`` so it overrides the inherited bold, rather
    than rendering bold under the copy-verbatim-into-HTML contract. Regression
    for the absolute defaults-drop in build_style."""
    _seed_pages(
        workspace,
        {
            1: (
                _styled_line("Bold intro text", 100, style={"b": 1}, size=12.0)
                + _styled_line("plain normal child words", 200, size=12.0)
            )
        },
    )
    xml = "<root><Para>Bold intro text <Note>plain normal child words</Note></Para></root>"
    res = _ground(workspace, tmp_path, xml, write_stats=False)
    root = etree.parse(str(res.output_path)).getroot()

    (para,) = list(root.iter("Para"))
    (note,) = list(root.iter("Note"))
    assert para.get("style") == "font-weight: bold"  # introduces bold
    # The child restates normal *because it differs from the inherited bold* —
    # kept, not dropped. Sparse elsewhere: no redundant font-size/text-transform.
    assert note.get("style") == "font-weight: normal"


def test_dg_style_plain_body_stays_sparse(workspace: Workspace, tmp_path: Path) -> None:
    """Emitting observed defaults during aggregation must not leak into output:
    a plain body line under a plain root ends up with no dg:style at all, since
    every value matches the inherited default and is suppressed."""
    _seed_pages(workspace, {1: _styled_line("just some plain body text here", 100, size=12.0)})
    xml = "<root><Body>just some plain body text here</Body></root>"
    res = _ground(workspace, tmp_path, xml, write_stats=False)
    root = etree.parse(str(res.output_path)).getroot()
    (body,) = list(root.iter("Body"))
    assert body.get("style") is None


def test_dg_style_color_needs_char_weight_majority(workspace: Workspace, tmp_path: Path) -> None:
    """color is a char-weighted majority, not a plurality: a mostly-black
    paragraph with one stray colored word stays uncolored. (Black/near-black
    words never enter the color counter but still count toward the total.)"""
    _seed_pages(
        workspace,
        {
            1: (
                _styled_line("plain black body words here now", 100, size=12.0)
                + _styled_line("red", 200, style={"c": "red"}, size=12.0)
            )
        },
    )
    xml = "<root><Para>plain black body words here now red</Para></root>"
    res = _ground(workspace, tmp_path, xml, write_stats=False)
    root = etree.parse(str(res.output_path)).getroot()
    (para,) = list(root.iter("Para"))
    # One 3-char red word among ~26 black chars is not a majority -> no color.
    assert "color" not in (para.get("style") or "")


def test_dg_style_color_emitted_when_majority(workspace: Workspace, tmp_path: Path) -> None:
    """The positive control: when the dominant color IS a char-weight majority
    it is emitted (a whole line rendered red)."""
    _seed_pages(
        workspace,
        {1: _styled_line("entirely red heading line", 100, style={"c": "red"}, size=12.0)},
    )
    xml = "<root><Heading>entirely red heading line</Heading></root>"
    res = _ground(workspace, tmp_path, xml, write_stats=False)
    root = etree.parse(str(res.output_path)).getroot()
    (heading,) = list(root.iter("Heading"))
    assert "color: red" in (heading.get("style") or "")


def test_dg_style_never_emits_text_align_for_digital(workspace: Workspace, tmp_path: Path) -> None:
    """text-align is not derived deterministically (page-relative geometry can't
    tell right-aligned text from a left-aligned column) — it is left to the OCR
    image path. A right-shifted digital line still gets its font facts but no
    text-align."""
    words = [
        {"t": w, "l": [1728 + i * 60, 100, 1728 + i * 60 + 50, 120], "s": {"b": 1}}
        for i, w in enumerate("Right shifted heading".split())
    ]
    _seed_pages(workspace, {1: words})
    xml = (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">'
        "<Heading>Right shifted heading</Heading></dg:chunk>"
    )
    res = _ground(workspace, tmp_path, xml, write_stats=False)
    content = res.output_path.read_text(encoding="utf-8")
    assert "font-weight: bold" in content  # font facts still land
    assert "text-align" not in content  # never derived for digital/hybrid


def test_dg_style_regrounding_clears_stale_style(workspace: Workspace, tmp_path: Path) -> None:
    """Grounding owns dg:style: a stale value from a prior run (e.g. a
    text-align a since-removed heuristic emitted) must be cleared, not kept,
    when the element has no observable style now. Guards idempotency."""
    _seed_pages(workspace, {1: _line("plain body paragraph text here", 100)})
    # Input already carries a dg:style, as a re-grounded file would.
    xml = (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">'
        '<Body dg:style="text-align: right">plain body paragraph text here</Body>'
        "</dg:chunk>"
    )
    res = _ground(workspace, tmp_path, xml, write_stats=False)
    content = res.output_path.read_text(encoding="utf-8")
    assert "dg:style" not in content  # stale attribute cleared
    assert "text-align" not in content


def test_dg_style_child_keeps_differing_value(workspace: Workspace, tmp_path: Path) -> None:
    """A child whose value differs from the ancestor keeps its own declaration."""
    _seed_pages(
        workspace,
        {
            1: (
                _styled_line("Red", 100, style={"c": "red"}, size=12.0)
                + _styled_line("blue tail here", 100, style={"c": "blue"}, size=12.0)
            )
        },
    )
    xml = "<root><Line><Inner>Red</Inner> blue tail here</Line></root>"
    res = _ground(workspace, tmp_path, xml, write_stats=False)
    root = etree.parse(str(res.output_path)).getroot()
    (inner,) = list(root.iter("Inner"))
    # Parent majority is blue (3 blue words vs 1 red); the red inner differs, so
    # it keeps its own color rather than being suppressed.
    assert inner.get("style") == "color: red"
