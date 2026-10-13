"""The completeness pass: inverse grounding of transcribed blocks against the
page text. No model, no network -- synthetic pages with boxes."""

import json

from dgml_core.generation import completeness
from dgml_core.generation.blocks import Block

W, H = 1000, 1400
LINE = 20  # word height
GAP_CELL = 60  # gap between table cells (> 1.5 * LINE)


def _words(lines, *, top=100, bold=False):
    """Page words from a list of lines; a line is a list of cells, each a string.
    Cells are separated by a column gap, words within a cell by a space."""
    out = []
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


def _write_pages(tmp_path, pages):
    d = tmp_path / "page_text"
    d.mkdir()
    for n, words in enumerate(pages, 1):
        payload = {"page": n, "width": W, "height": H, "words": words}
        (d / f"page_{n}.json").write_text(json.dumps(payload))
    return d


def _prose(page, n=30):
    return [[" ".join(f"p{page}w{i:02d}" for i in range(k, k + 10))] for k in range(0, n, 10)]


def _p(text, bid="b1"):
    return Block(id=bid, structure="p", text=text)


def _page_text(words):
    return " ".join(w["t"] for w in words)


def test_lost_first_page_is_filled_at_the_front(tmp_path):
    pages = [_words(_prose(1)), _words(_prose(2))]
    d = _write_pages(tmp_path, pages)
    blocks = [_p(_page_text(pages[1]), "b2")]  # page 1 never transcribed
    out, rep = completeness.ensure_complete(blocks, d)
    assert rep["regions"][0]["page"] == 1 and rep["regions"][0]["kind"] == "page"
    assert rep["regions"][0]["source"] == "text-layer"
    assert out[-1].id == "b2"  # original block stays last
    assert " ".join(b.flat_text() for b in out[:-1]) == _page_text(pages[0])
    assert rep["final_coverage_pct"] == 100.0 and rep["pages"][0]["unclaimed_after"] == 0


def test_lost_middle_page_lands_between_its_neighbours(tmp_path):
    pages = [_words(_prose(1)), _words(_prose(2)), _words(_prose(3))]
    d = _write_pages(tmp_path, pages)
    blocks = [_p(_page_text(pages[0]), "b1"), _p(_page_text(pages[2]), "b3")]
    out, rep = completeness.ensure_complete(blocks, d)
    assert out[0].id == "b1" and out[-1].id == "b3"
    assert " ".join(b.flat_text() for b in out[1:-1]) == _page_text(pages[1])
    assert rep["model_coverage_pct"] < 70 < rep["final_coverage_pct"] == 100.0


def test_column_major_table_is_not_a_region(tmp_path):
    # the text layer reads the table column by column; the model wrote rows
    grid = [[f"r{i}c{j}" for j in range(4)] for i in range(6)]
    column_major = [[c] for j in range(4) for c in (grid[i][j] for i in range(6))]
    d = _write_pages(tmp_path, [_words(column_major)])
    blocks = [Block(id=f"b{i}", structure="row", cells=row) for i, row in enumerate(grid)]
    out, rep = completeness.ensure_complete(blocks, d)
    assert rep["regions"] == [] and len(out) == 6
    assert rep["model_coverage_pct"] == 100.0


def test_lost_table_inside_a_present_page_becomes_rows(tmp_path):
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


def test_overlap_duplicate_is_dropped(tmp_path):
    pages = [_words(_prose(1)), _words(_prose(2))]
    d = _write_pages(tmp_path, pages)
    t2 = _page_text(pages[1])
    # page 2 transcribed twice
    blocks = [_p(_page_text(pages[0]), "b1"), _p(t2, "b2"), _p(t2, "b2dup")]
    out, rep = completeness.ensure_complete(blocks, d)
    assert [b.id for b in out] in (["b1", "b2"], ["b1", "b2dup"])
    assert rep["dropped"] == {"duplicates": 1, "fabricated": 0}


def test_fabricated_block_is_dropped(tmp_path):
    pages = [_words(_prose(1))]
    d = _write_pages(tmp_path, pages)
    blocks = [
        _p("trading could limit the ability to attract clients into covered funds", "fake"),
        _p(_page_text(pages[0]), "b1"),
    ]
    out, rep = completeness.ensure_complete(blocks, d)
    assert [b.id for b in out] == ["b1"]
    assert rep["dropped"]["fabricated"] == 1


def test_rescue_replaces_fallback_only_when_it_claims_the_page(tmp_path):
    pages = [_words(_prose(1)), _words(_prose(2))]
    d = _write_pages(tmp_path, pages)
    blocks = [_p(_page_text(pages[1]), "b2")]
    good = lambda idx: [Block(id="rescued", structure="p", text=_page_text(pages[idx]))]  # noqa: E731
    out, rep = completeness.ensure_complete(blocks, d, rescue=good)
    assert out[0].id == "rescued" and rep["regions"][0]["source"] == "model-rescue"
    bad = lambda idx: [Block(id="wrong", structure="p", text="nothing from that page at all here")]  # noqa: E731
    out, rep = completeness.ensure_complete(blocks, d, rescue=bad)
    assert all(b.id != "wrong" for b in out) and rep["regions"][0]["source"] == "text-layer"


def test_ligatures_do_not_look_missing(tmp_path):
    words = _words([["net proﬁt rose signiﬁcantly in the ﬁrst quarter of the year"]])
    d = _write_pages(tmp_path, [words])
    blocks = [_p("net profit rose significantly in the first quarter of the year")]
    _, rep = completeness.ensure_complete(blocks, d)
    assert rep["model_coverage_pct"] == 100.0 and rep["regions"] == []
