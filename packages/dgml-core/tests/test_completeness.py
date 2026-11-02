"""The completeness pass: inverse grounding of transcribed blocks against the
page text. No model, no network -- synthetic pages with boxes."""

import json
from pathlib import Path
from typing import Any

from dgml_core.generation import completeness
from dgml_core.generation.blocks import Block
from dgml_core.generation.render import render_xml

W, H = 1000, 1400
LINE = 20  # word height
GAP_CELL = 60  # gap between table cells (> 1.5 * LINE)


def _words(lines: list[list[str]], *, top: int = 100, bold: bool = False) -> list[dict[str, Any]]:
    """Page words from a list of lines; a line is a list of cells, each a string.
    Cells are separated by a column gap, words within a cell by a space."""
    out: list[dict[str, Any]] = []
    y = top
    for cells in lines:
        x = 50
        for cell in cells:
            for tok in cell.split():
                wdt = 9 * len(tok)
                out.append({"t": tok, "l": [x, y, x + wdt, y + LINE], "s": {"b": 1 if bold else 0}})
                x += wdt + 8
            x += GAP_CELL
        y += int(LINE * 1.4)
    return out


def _write_pages(tmp_path: Path, pages: list[list[dict[str, Any]]]) -> Path:
    d = tmp_path / "page_text"
    d.mkdir()
    for n, words in enumerate(pages, 1):
        payload = {"page": n, "width": W, "height": H, "words": words}
        (d / f"page_{n}.json").write_text(json.dumps(payload))
    return d


def _prose(page: int, n: int = 30) -> list[list[str]]:
    return [[" ".join(f"p{page}w{i:02d}" for i in range(k, k + 10))] for k in range(0, n, 10)]


def _p(text: str, bid: str = "b1") -> Block:
    return Block(id=bid, structure="p", text=text)


def _page_text(words: list[dict[str, Any]]) -> str:
    return " ".join(w["t"] for w in words)


def test_lost_first_page_is_filled_at_the_front(tmp_path: Path) -> None:
    pages = [_words(_prose(1)), _words(_prose(2))]
    d = _write_pages(tmp_path, pages)
    blocks = [_p(_page_text(pages[1]), "b2")]  # page 1 never transcribed
    out, rep = completeness.ensure_complete(blocks, d)
    assert rep["regions"][0]["page"] == 1 and rep["regions"][0]["kind"] == "page"
    assert rep["regions"][0]["source"] == "text-layer"
    assert out[-1].id == "b2"  # original block stays last
    assert " ".join(b.flat_text() for b in out[:-1]) == _page_text(pages[0])
    assert rep["final_coverage_pct"] == 100.0 and rep["pages"][0]["unclaimed_after"] == 0


def test_lost_middle_page_lands_between_its_neighbours(tmp_path: Path) -> None:
    pages = [_words(_prose(1)), _words(_prose(2)), _words(_prose(3))]
    d = _write_pages(tmp_path, pages)
    blocks = [_p(_page_text(pages[0]), "b1"), _p(_page_text(pages[2]), "b3")]
    out, rep = completeness.ensure_complete(blocks, d)
    assert out[0].id == "b1" and out[-1].id == "b3"
    assert " ".join(b.flat_text() for b in out[1:-1]) == _page_text(pages[1])
    assert rep["model_coverage_pct"] < 70 < rep["final_coverage_pct"] == 100.0


def test_column_major_table_is_not_a_region(tmp_path: Path) -> None:
    # the text layer reads the table column by column; the model wrote rows
    grid = [[f"r{i}c{j}" for j in range(4)] for i in range(6)]
    column_major = [[c] for j in range(4) for c in (grid[i][j] for i in range(6))]
    d = _write_pages(tmp_path, [_words(column_major)])
    blocks = [Block(id=f"b{i}", structure="row", cells=row) for i, row in enumerate(grid)]
    out, rep = completeness.ensure_complete(blocks, d)
    assert rep["regions"] == [] and len(out) == 6
    assert rep["model_coverage_pct"] == 100.0


def test_lost_table_inside_a_present_page_becomes_rows(tmp_path: Path) -> None:
    prose = _prose(1, 20)
    table = [["Revenue", "1,267", "1,356"], ["Operating income", "203", "208"]]
    words = _words(prose) + _words(table, top=100 + 3 * int(LINE * 1.4) + 60)
    d = _write_pages(tmp_path, [words])
    blocks = [_p(" ".join(c for ln in prose for c in ln), "b1")]
    out, rep = completeness.ensure_complete(blocks, d)
    assert len(rep["regions"]) == 1 and rep["regions"][0]["kind"] == "block"
    rows = [b for b in out if b.structure == "row"]
    assert [r.cells for r in rows] == table
    assert out[0].id == "b1"  # table placed after the prose it follows
    assert rep["pages"][0]["unclaimed_figures_after"] == 0


def test_overlap_duplicate_is_dropped(tmp_path: Path) -> None:
    pages = [_words(_prose(1)), _words(_prose(2))]
    d = _write_pages(tmp_path, pages)
    t2 = _page_text(pages[1])
    # page 2 transcribed twice
    blocks = [_p(_page_text(pages[0]), "b1"), _p(t2, "b2"), _p(t2, "b2dup")]
    out, rep = completeness.ensure_complete(blocks, d)
    assert [b.id for b in out] in (["b1", "b2"], ["b1", "b2dup"])
    assert rep["dropped"] == {"duplicates": 1, "fabricated": 0}


def test_fabricated_block_is_dropped(tmp_path: Path) -> None:
    pages = [_words(_prose(1))]
    d = _write_pages(tmp_path, pages)
    blocks = [
        _p("trading could limit the ability to attract clients into covered funds", "fake"),
        _p(_page_text(pages[0]), "b1"),
    ]
    out, rep = completeness.ensure_complete(blocks, d)
    assert [b.id for b in out] == ["b1"]
    assert rep["dropped"]["fabricated"] == 1


def test_rescue_replaces_fallback_only_when_it_claims_the_page(tmp_path: Path) -> None:
    pages = [_words(_prose(1)), _words(_prose(2))]
    d = _write_pages(tmp_path, pages)
    blocks = [_p(_page_text(pages[1]), "b2")]
    good = lambda idx: [Block(id="rescued", structure="p", text=_page_text(pages[idx]))]  # noqa: E731
    out, rep = completeness.ensure_complete(blocks, d, rescue=good)
    assert out[0].id == "rescued" and rep["regions"][0]["source"] == "model-rescue"
    bad = lambda idx: [Block(id="wrong", structure="p", text="nothing from that page at all here")]  # noqa: E731
    out, rep = completeness.ensure_complete(blocks, d, rescue=bad)
    assert all(b.id != "wrong" for b in out) and rep["regions"][0]["source"] == "text-layer"


def test_ligatures_do_not_look_missing(tmp_path: Path) -> None:
    words = _words([["net proﬁt rose signiﬁcantly in the ﬁrst quarter of the year"]])
    d = _write_pages(tmp_path, [words])
    blocks = [_p("net profit rose significantly in the first quarter of the year")]
    _, rep = completeness.ensure_complete(blocks, d)
    assert rep["model_coverage_pct"] == 100.0 and rep["regions"] == []


def test_text_layer_fill_is_xml_safe(tmp_path: Path) -> None:
    """A font with no ToUnicode entry for a ligature extracts it as NUL. A page
    filled from the text layer must come out without it, or rendering the
    document raises (lxml refuses control characters in element text)."""
    lost = [
        ["the con\x00guration is de\x00ned in the o\x00cial speci\x00cation of the system"],
        ["each modi\x00cation is veri\x00ed before the \x00nal certi\x00cation is issued"],
        ["Pro\x00t", "1,267", "1,356"],
    ]
    pages = [_words(lost), _words(_prose(2))]
    d = _write_pages(tmp_path, pages)
    blocks = [_p(_page_text(pages[1]), "b2")]  # page 1 never transcribed
    out, rep = completeness.ensure_complete(blocks, d)
    assert rep["regions"] and rep["regions"][0]["source"] == "text-layer"
    text = " ".join(b.flat_text() for b in out)
    assert "\x00" not in text and "conguration" in text and "1,267" in text
    render_xml(out, doc_name="example.pdf")  # raised ValueError before the fix


def test_block_written_before_the_pages_first_paragraph_is_claimed(tmp_path: Path) -> None:
    """The model wrote a page's table before the paragraph the text layer
    reads first. The paragraph aligns at the page's first word; the table
    must still be claimed on that page, not dropped and re-filled."""
    pages = [_words(_prose(1)), _words(_prose(2))]
    d = _write_pages(tmp_path, pages)
    first = _page_text(pages[1]).split()
    para, table = " ".join(first[: len(first) // 2]), " ".join(first[len(first) // 2 :])
    blocks = [_p(_page_text(pages[0]), "b1"), _p(table, "b2_table"), _p(para, "b3_para")]
    out, rep = completeness.ensure_complete(blocks, d)
    assert [b.id for b in out] == ["b1", "b2_table", "b3_para"]
    assert rep["dropped"] == {"duplicates": 0, "fabricated": 0}
    assert rep["regions"] == []


def test_overlap_duplicate_is_dropped_by_window_position(tmp_path: Path) -> None:
    """With window starts known, the second window's leading blocks that
    repeat the first window's trailing blocks are the overlap page and go,
    whichever copy the ordered diff happened to credit."""
    pages = [_words(_prose(1)), _words(_prose(2)), _words(_prose(3))]
    d = _write_pages(tmp_path, pages)
    t2 = _page_text(pages[1])
    blocks = [
        _p(_page_text(pages[0]), "b1"),
        _p(t2, "b2"),
        _p(t2, "b2again"),
        _p(_page_text(pages[2]), "b3"),
    ]
    out, rep = completeness.ensure_complete(blocks, d, window_starts=[0, 2])
    assert [b.id for b in out] == ["b1", "b2", "b3"]
    assert rep["dropped"] == {"duplicates": 1, "fabricated": 0}
    assert rep["regions"] == []
