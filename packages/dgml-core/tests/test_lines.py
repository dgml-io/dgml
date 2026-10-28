"""Structure over text-layer lines (generation.lines) and its hook in Pass A."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest
from dgml_core import llm
from dgml_core.generation import lines as lines_mod
from dgml_core.textmatch import PageDims, Word, fuzzy_norm

from .test_generation import _route_window_calls

W, H = 1000, 1400
LINE_H = 20


def _word(idx: int, text: str, left: int, top: int, *, bold: bool = False) -> Word:
    return Word(
        idx=idx,
        text=text,
        text_norm=fuzzy_norm(text),
        left=left,
        top=top,
        right=left + 9 * len(text),
        bottom=top + LINE_H,
        bold=bold,
    )


def _page(rows: list[list[tuple[str, int]]], *, bold_rows: Collection[int] = ()) -> list[Word]:
    """Rows of (text, left) at successive baselines."""
    out: list[Word] = []
    for r, row in enumerate(rows):
        for text, left in row:
            out.append(_word(len(out), text, left, 100 + r * 2 * LINE_H, bold=r in bold_rows))
    return out


def _prose(words: str, left: int = 100) -> list[tuple[str, int]]:
    x, out = left, []
    for w in words.split():
        out.append((w, x))
        x += 9 * len(w) + 6
    return out


PAGE1 = _page(
    [
        [("Annual", 100), ("Report", 170)],  # L1 running header
        _prose("ITEM 1. BUSINESS"),  # L2 heading
        _prose("We are a leading pro-"),  # L3 paragraph
        _prose("vider of services to clients."),  # L4
        [("(in millions)", 100), ("2013", 600), ("2012", 800)],  # L5 table header
        [("Revenue", 100), ("10,180", 600), ("9,337", 800)],  # L6 row
        [("Total expenses", 100), ("6,323", 600), ("5,813", 800)],  # L7 row
        _prose("(a) First item of a list."),  # L8
        _prose("(b) Second item of a list."),  # L9
        [("12", 500)],  # L10 page number
    ],
    bold_rows={1},
)
PAGE2 = _page(
    [
        [("Annual", 100), ("Report", 170)],  # L11 running header
        _prose("continues the sentence from before."),  # L12
        _prose("Another paragraph here."),  # L13
        [("13", 500)],  # L14 page number
    ]
)
PAGES = {1: (PAGE1, PageDims(W, H)), 2: (PAGE2, PageDims(W, H))}


def test_window_lines_are_numbered_across_pages_with_cells() -> None:
    lines = lines_mod.window_lines(PAGES, [1, 2])
    assert [ln.id for ln in lines] == list(range(1, 15))
    assert [ln.page for ln in lines] == [1] * 10 + [2] * 4
    assert lines[4].cells() == ["(in millions)", "2013", "2012"]
    assert lines[2].text == "We are a leading pro-"
    listing = lines_mod.render_listing(lines)
    assert listing.startswith("--- page 1 ---\nL1\tAnnual Report\n")
    assert "L5\t(in millions) | 2013 | 2012" in listing
    assert "--- page 2 ---\nL11\t" in listing


@pytest.mark.parametrize(
    ("field", "ids"),
    [
        ("L3", [3]),
        ("L3-L4", [3, 4]),
        ("L3-4", [3, 4]),
        ("L3\u2013L5", [3, 4, 5]),
        ("L6+L7", [6, 7]),
        ("l9", [9]),
        ("Hello", None),
        ("", None),
    ],
)
def test_parse_refs(field: str, ids: list[int] | None) -> None:
    assert lines_mod.parse_refs(field) == ids


REPLY = "\n".join(
    [
        "X\tL1",
        "H1\tITEM 1.\tL2",
        "P\tL3-L4",
        "R\tL5",
        "R\tL6",
        "R\tL7",
        "I\t(a)\tL8",
        "I\t(b)\tL9",
        "X\tL10",
        "X\tL11",
        "C\tL12",
        "P\tL13",
        "X\tL14",
    ]
)


def test_build_payload_copies_text_from_the_lines() -> None:
    lines = lines_mod.window_lines(PAGES, [1, 2])
    res = lines_mod.build_payload(REPLY, lines)
    blocks = res.payload["blocks"]
    assert [b["structure"] for b in blocks] == [
        "heading",
        "p",
        "row",
        "row",
        "row",
        "item",
        "item",
        "p",
    ]
    assert blocks[0]["lim"] == "ITEM 1." and blocks[0]["text"] == "BUSINESS"
    # end-of-line hyphenation mended, lines joined
    assert blocks[1]["text"] == "We are a leading provider of services to clients."
    assert blocks[2]["cells"] == ["(in millions)", "2013", "2012"]
    assert blocks[3]["cells"] == ["Revenue", "10,180", "9,337"]
    assert blocks[5] == {
        "structure": "item",
        "lim": "(a)",
        "text": "First item of a list.",
        "lines": [8],
    }
    assert res.payload["continues"] == "continues the sentence from before."
    assert res.decorations == {1, 10, 11, 14}
    assert res.recall(lines) == 1.0
    assert res.unassigned(lines) == []
    assert (res.malformed, res.unknown_refs, res.literal_fields) == (0, 0, 0)


def test_wrapped_row_merges_cellwise_and_literal_fields_pass_through() -> None:
    page = _page(
        [
            [("Long label that", 100), ("1,234", 600), ("5,678", 800)],
            [("wraps", 100)],
        ]
    )
    lines = lines_mod.window_lines({1: (page, PageDims(W, H))}, [1])
    res = lines_mod.build_payload("R\tL1+L2\nP\t[Chart: revenue by region]", lines)
    assert res.payload["blocks"][0]["cells"] == ["Long label that wraps", "1,234", "5,678"]
    assert res.payload["blocks"][1]["text"] == "[Chart: revenue by region]"
    assert res.literal_fields == 1


def test_unaccounted_lines_are_filled_in_place() -> None:
    lines = lines_mod.window_lines(PAGES, [1, 2])
    # The model forgot the table (L5-L7) and the page number.
    reply = (
        "X\tL1\nH1\tITEM 1.\tL2\nP\tL3-L4\nI\t(a)\tL8\nI\t(b)\tL9\nX\tL11\nC\tL12\nP\tL13\nX\tL14"
    )
    res = lines_mod.build_payload(reply, lines)
    assert lines_mod.describe_unassigned(res.unassigned(lines)) == "L5-L7, L10"
    assert res.recall(lines) < 1.0
    filled = lines_mod.complete_payload(res, lines)
    assert filled == 4
    kinds = [(b["structure"], b.get("fallback", False)) for b in res.payload["blocks"]]
    # table rows land between the paragraph and the list, as rows
    assert kinds == [
        ("heading", False),
        ("p", False),
        ("row", True),
        ("row", True),
        ("row", True),
        ("item", False),
        ("item", False),
        ("p", True),  # the page number, a short paragraph at the end of page 1
        ("p", False),
    ]
    assert res.recall(lines) == 1.0


def test_unknown_and_duplicate_refs_are_counted_not_fatal() -> None:
    lines = lines_mod.window_lines(PAGES, [1])
    res = lines_mod.build_payload("P\tL3\nP\tL3\nP\tL99\nQ\tL4", lines)
    assert [b["text"] for b in res.payload["blocks"]] == ["We are a leading pro-", "", ""]
    assert res.unknown_refs == 2
    assert res.malformed == 1


def _write_page_text(tmp_path: Path, pages: dict[int, tuple[list[Word], PageDims]]) -> Path:
    d = tmp_path / "page_text"
    d.mkdir()
    for n, (words, dims) in pages.items():
        payload = {
            "page": n,
            "width": dims.width,
            "height": dims.height,
            "words": [
                {"t": w.text, "l": [w.left, w.top, w.right, w.bottom], "s": {"b": w.bold}}
                for w in words
            ],
        }
        (d / f"page_{n}.json").write_text(json.dumps(payload))
    return d


def test_transcribe_document_in_lines_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """End to end through Pass A: the model is shown the numbered listing with
    the lines system prompt, answers with ids, and the blocks carry the page's
    text; a forgotten line is filled before the window is kept."""
    from dgml_core.generation import document as document_mod
    from dgml_core.generation import transcribe as transcribe_mod

    page_text = _write_page_text(tmp_path, {1: PAGES[1]})
    seen: list[dict[str, Any]] = []

    def fake_call_continued(config: llm.LLMConfig, **kwargs: object) -> str:
        seen.append(dict(kwargs))
        # forgets L7 (a table row) and L10 (page number)
        return "X\tL1\nH1\tITEM 1.\tL2\nP\tL3-L4\nR\tL5\nR\tL6\nI\t(a)\tL8\nI\t(b)\tL9"

    _route_window_calls(monkeypatch, fake_call_continued)
    monkeypatch.setattr(transcribe_mod, "_count_pages", lambda _b: 1)
    monkeypatch.setattr(document_mod, "slice_pdf", lambda b, _pages, **_kw: b)

    blocks = transcribe_mod.transcribe_document(
        b"%PDF-fake",
        doc_name="doc.pdf",
        config=llm.LLMConfig(model="anthropic/claude-haiku-5-5"),
        cache_dir=tmp_path,
        debug=True,
        page_text_dir=page_text,
        transcription="lines",
        complete=False,
    )
    assert seen[0]["system_prompt"] == transcribe_mod.SYSTEM_PROMPT_LINES
    user_text = seen[0]["user_content"][0]["text"]
    assert "L5\t(in millions) | 2013 | 2012" in user_text
    assert (tmp_path / "doc_w01_lines.txt").exists()
    texts = [(b.structure, b.text or " | ".join(b.cells)) for b in blocks]
    assert texts == [
        ("p", "Annual Report"),  # X refused: a single page cannot show a repeat
        ("heading", "BUSINESS"),
        # "provider" occurs nowhere on its own in this one-page document, so
        # the line-break hyphen is kept rather than mended into a new word.
        ("p", "We are a leading pro-vider of services to clients."),
        ("row", "(in millions) | 2013 | 2012"),
        ("row", "Revenue | 10,180 | 9,337"),
        ("row", "Total expenses | 6,323 | 5,813"),
        ("item", "First item of a list."),
        ("item", "Second item of a list."),
        ("p", "12"),
    ]
    assert blocks[1].lim == "ITEM 1."


def test_x_is_only_accepted_for_short_edge_lines() -> None:
    """A model that marks a page's body as decorations gets it filled instead."""
    lines = lines_mod.window_lines(PAGES, [1, 2])
    reply = "X\tL1-L14"
    res = lines_mod.build_payload(reply, lines)
    # Running header (repeated on page 2) and page numbers only; the body is not.
    assert res.decorations == {1, 10, 11, 14}
    assert [ln.id for ln in res.unassigned(lines)] == list(range(2, 10)) + [12, 13]


def test_literal_text_claims_the_matching_lines() -> None:
    lines = lines_mod.window_lines(PAGES, [1])
    reply = "P\tWe are a leading pro- vider of services to clients.\nH1\t\tITEM 1. BUSINESS"
    res = lines_mod.build_payload(reply, lines)
    assert res.literal_fields == 2
    assert res.assigned == {2, 3, 4}
    assert res.payload["blocks"][0]["text"] == "We are a leading provider of services to clients."
    assert res.payload["blocks"][1]["text"] == "ITEM 1. BUSINESS"


def test_restarted_reply_keeps_the_better_draft() -> None:
    lines = lines_mod.window_lines(PAGES, [1, 2])
    reply = "X\tL1\nP\tL1-L2\nWait, that was wrong. Restarting:\n" + REPLY
    res = lines_mod.build_payload(reply, lines)
    assert res.restarts == 1
    assert res.recall(lines) == 1.0
    assert res.payload["blocks"][0]["structure"] == "heading"


def test_continuation_that_starts_a_sentence_becomes_a_paragraph() -> None:
    lines = lines_mod.window_lines(PAGES, [1, 2])
    # L12 "continues the sentence…" is a real continuation; L2 "ITEM 1. BUSINESS" is not.
    res = lines_mod.build_payload("C\tL2\nC\tL12", lines)
    assert res.payload["continues"] == "continues the sentence from before."
    assert res.payload["blocks"][0] == {"structure": "p", "text": "ITEM 1. BUSINESS", "lines": [2]}
    assert res.demoted_continuations == 1


def test_blocks_are_put_back_in_page_order() -> None:
    lines = lines_mod.window_lines(PAGES, [1])
    res = lines_mod.build_payload("I\t(b)\tL9\nI\t(a)\tL8\nP\tL3-L4\nH1\tITEM 1.\tL2", lines)
    moved = lines_mod.order_by_lines(res)
    assert moved == 4
    assert [min(b["lines"]) for b in res.payload["blocks"]] == [2, 3, 8, 9]


def test_report_only_pass_does_not_fill(tmp_path: Path) -> None:
    from dgml_core.generation import completeness
    from dgml_core.generation.blocks import Block

    d = _write_page_text(tmp_path, {1: PAGES[1]})
    lines = lines_mod.window_lines({1: PAGES[1]}, [1])
    # Only the first paragraph is transcribed; the rest of the page is missing.
    blocks = [Block(id="b1", structure="p", text=lines[2].text + " " + lines[3].text)]
    out, rep = completeness.ensure_complete(blocks, d, fill=False)
    assert [b.id for b in out] == ["b1"]
    assert rep["filled"] is False
    assert rep["regions"] and rep["final_coverage_pct"] == rep["model_coverage_pct"] < 50


def test_a_row_range_is_one_row_per_line() -> None:
    page = _page(
        [
            [("(in millions)", 100), ("2013", 600), ("2012", 800)],
            [("Revenue", 100), ("10,180", 600), ("9,337", 800)],
            [("Expenses", 100), ("6,323", 600), ("5,813", 800)],
        ]
    )
    lines = lines_mod.window_lines({1: (page, PageDims(W, H))}, [1])
    res = lines_mod.build_payload("R\tL1-L3", lines)
    assert [b["cells"] for b in res.payload["blocks"]] == [
        ["(in millions)", "2013", "2012"],
        ["Revenue", "10,180", "9,337"],
        ["Expenses", "6,323", "5,813"],
    ]
    assert res.recall(lines) == 1.0


def test_text_format_reply_in_lines_mode_is_used_as_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dgml_core.generation import document as document_mod
    from dgml_core.generation import transcribe as transcribe_mod

    page_text = _write_page_text(tmp_path, {1: PAGES[1]})
    reply = json.dumps(
        {
            "continues": "",
            "blocks": [
                {"structure": "p", "text": "We are a leading provider of services to clients."}
            ],
        }
    )
    _route_window_calls(monkeypatch, lambda config, **kw: reply)
    monkeypatch.setattr(transcribe_mod, "_count_pages", lambda _b: 1)
    monkeypatch.setattr(document_mod, "slice_pdf", lambda b, _pages, **_kw: b)
    blocks = transcribe_mod.transcribe_document(
        b"%PDF-fake",
        doc_name="doc.pdf",
        config=llm.LLMConfig(model="anthropic/claude-haiku-5-5"),
        cache_dir=tmp_path,
        page_text_dir=page_text,
        transcription="lines",
        complete=False,
    )
    assert [b.text for b in blocks] == ["We are a leading provider of services to clients."]


def test_two_column_page_is_read_column_by_column() -> None:
    rows = []
    for i in range(6):
        rows.append(
            _prose(f"left column line {i} with enough words here", left=60)
            + _prose(f"right column line {i} with enough words too", left=560)
        )
    rows.append(_prose("A full width closing sentence spans the whole page width.", left=60))
    page = _page(rows)
    lines = lines_mod.window_lines({1: (page, PageDims(W, H))}, [1])
    texts = [ln.text for ln in lines]
    assert texts[:6] == [f"left column line {i} with enough words here" for i in range(6)]
    assert texts[6:12] == [f"right column line {i} with enough words too" for i in range(6)]
    assert texts[12].startswith("A full width")


def test_table_is_not_mistaken_for_columns() -> None:
    rows = [[("Revenue", 100), ("10,180", 600), ("9,337", 800)] for _ in range(6)]
    page = _page(rows)
    lines = lines_mod.window_lines({1: (page, PageDims(W, H))}, [1])
    assert len(lines) == 6 and lines[0].cells() == ["Revenue", "10,180", "9,337"]


def test_lim_written_as_a_ref_resolves_to_text() -> None:
    lines = lines_mod.window_lines(PAGES, [1])
    res = lines_mod.build_payload("H1\tL2\tL3", lines)
    assert res.payload["blocks"][0]["lim"] == "ITEM 1. BUSINESS"
    assert res.payload["blocks"][0]["text"] == "We are a leading pro-"


def test_hyphen_is_mended_only_into_a_known_word() -> None:
    page = _page(
        [
            _prose("a former wholly-"),
            _prose("owned subsidiary and a pro-"),
            _prose("vider of services"),
        ]
    )
    lines = lines_mod.window_lines({1: (page, PageDims(W, H))}, [1])
    vocab = {"provider", "wholly", "owned", "subsidiary", "services"}
    res = lines_mod.build_payload("P\tL1-L3", lines, vocab)
    assert (
        res.payload["blocks"][0]["text"]
        == "a former wholly-owned subsidiary and a provider of services"
    )
    # without a vocabulary every hyphen is mended
    res2 = lines_mod.build_payload("P\tL1-L3", lines)
    assert (
        res2.payload["blocks"][0]["text"]
        == "a former whollyowned subsidiary and a provider of services"
    )


def test_textless_page_gets_its_own_text_call_in_a_lines_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Page 1 has a text layer, page 2 has none: the window is still asked in
    lines mode, then page 2 alone is asked in the text format and spliced in
    after page 1's blocks."""
    from dgml_core.generation import document as document_mod
    from dgml_core.generation import transcribe as transcribe_mod

    page_text = _write_page_text(tmp_path, {1: PAGES[1], 2: ([], PageDims(W, H))})
    calls: list[str] = []

    def fake(config: llm.LLMConfig, **kw: object) -> str:
        calls.append(str(kw["system_prompt"]))
        if kw["system_prompt"] == transcribe_mod.SYSTEM_PROMPT_LINES:
            return (
                "X\tL1\nH1\tITEM 1.\tL2\nP\tL3-L4\nR\tL5\nR\tL6\nR\tL7\n"
                "I\t(a)\tL8\nI\t(b)\tL9\nX\tL10"
            )
        return "P\tScanned map of the network."

    _route_window_calls(monkeypatch, fake)
    monkeypatch.setattr(transcribe_mod, "_count_pages", lambda _b: 2)
    monkeypatch.setattr(document_mod, "slice_pdf", lambda b, _pages, **_kw: b)
    blocks = transcribe_mod.transcribe_document(
        b"%PDF-fake",
        doc_name="doc.pdf",
        config=llm.LLMConfig(model="anthropic/claude-haiku-5-5"),
        cache_dir=tmp_path,
        page_text_dir=page_text,
        transcription="lines",
        complete=False,
    )
    assert calls == [transcribe_mod.SYSTEM_PROMPT_LINES, transcribe_mod.SYSTEM_PROMPT]
    assert blocks[-1].text == "Scanned map of the network."
    assert blocks[0].text == "Annual Report"  # page 1 content first


def test_report_only_pass_fills_the_pages_it_is_told_to(tmp_path: Path) -> None:
    from dgml_core.generation import completeness
    from dgml_core.generation.blocks import Block

    d = _write_page_text(tmp_path, {1: PAGES[1], 2: PAGES[2]})
    lines = lines_mod.window_lines({1: PAGES[1]}, [1])
    blocks = [Block(id="b1", structure="p", text=lines[2].text + " " + lines[3].text)]
    out, rep = completeness.ensure_complete(blocks, d, fill=False, fill_pages={2})
    pages_filled = {r["page"] for r in rep["regions"] if r.get("source")}
    assert 2 in pages_filled and len(out) > 1
    assert any(
        b.text.startswith("continues the sentence") or "Another paragraph" in b.text for b in out
    )
