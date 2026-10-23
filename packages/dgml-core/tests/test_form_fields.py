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

"""Values filled into PDF form fields reach digital page_text.

The fixtures are small hand-built AcroForm PDFs: printed labels in the page
content and filled text fields and combo boxes whose values exist only in the
widget appearance streams, next to fields that must add nothing (empty,
hidden, no appearance, a check box, a list box, a password field, a widget
with no field name, a value the page content already prints). The field words
go into the page in reading order, next to their labels.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from dgml_core import form_fields
from dgml_core.form_fields import (
    FieldWords,
    _display_text,
    _prorate,
    form_field_words,
    merge_form_field_words,
)
from dgml_core.text_extraction import PageTextDefect, extract_text_digital
from dgml_core.textmatch import Word, fuzzy_norm
from dgml_core.xml_grounding import _page_baseline

from .conftest import _write_text_pdf

DPI = 300
SCALE = DPI / 72
PAGE_W, PAGE_H = 612, 792
Box = tuple[float, float, float, float]
COMBO = 1 << 17  # choice-field flag: a combo box; without it, a list box
PASSWORD = 1 << 13  # text-field flag: a password field


@dataclass
class Field:
    """One field with one widget. ``ap`` is the appearance content stream
    (``None``: no /AP), drawn in a BBox of the rect's own size unless ``bbox``
    says otherwise, under ``matrix``. ``states`` makes ``/AP /N`` a
    dictionary of appearance states (state -> content) and ``state`` the
    ``/AS`` that picks one. The appearance can use ``/Helv`` (Helvetica) or
    ``/NoMap`` (a font whose glyphs pdfminer cannot map to text). ``opt`` is a
    raw ``/Opt`` array for a choice field. ``named=False`` leaves out the field
    name ``/T``."""

    rect: Box
    value: str | None
    ap: str | None
    ft: str = "Tx"
    flags: int = 4  # Print
    ff: int = 0
    on_parent: bool = False  # T, FT, V, Ff and Opt on a parent field, widget is a kid
    opt: str = ""
    named: bool = True
    bbox: Box | None = None
    matrix: str = ""
    states: dict[str, str] | None = None
    state: str = ""


@dataclass
class Page:
    content: str = ""
    fields: list[Field] = field(default_factory=list)
    rotate: int = 0
    mediabox: Box = (0, 0, PAGE_W, PAGE_H)
    scan: bool = False  # a full-page image under the content: a scanned page


def _write_form_pdf(path: Path, pages: list[Page], *, acroform: bool = True) -> None:
    """Hand-built PDF: Helvetica page content plus AcroForm fields."""
    objs: list[bytes] = []

    def add(body: str | bytes) -> int:
        objs.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(objs)

    def stream(text: str | bytes, extra: str = "") -> int:
        data = text.encode("latin-1") if isinstance(text, str) else text
        return add(
            b"<< /Length %d %s >>\nstream\n" % (len(data), extra.encode()) + data + b"\nendstream"
        )

    catalog = add("")  # filled in last
    pages_id = add("")
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    # Identity-H codes with no /ToUnicode: pdfminer reads each glyph as (cid:N).
    no_map = add(
        "<< /Type /Font /Subtype /Type0 /BaseFont /NoMap /Encoding /Identity-H "
        "/DescendantFonts [<< /Type /Font /Subtype /CIDFontType2 /BaseFont /NoMap "
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        "/FontDescriptor << /Type /FontDescriptor /FontName /NoMap /Flags 4 "
        "/FontBBox [0 -200 1000 800] /ItalicAngle 0 /Ascent 800 /Descent -200 "
        "/CapHeight 700 /StemV 80 >> /DW 600 >>] >>"
    )
    image = stream(
        bytes([200, 200, 200, 200]),
        "/Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray "
        "/BitsPerComponent 8",
    )
    field_ids: list[int] = []
    page_ids: list[int] = []
    for page in pages:
        mx0, my0, mx1, my1 = page.mediabox
        scan = f"q {mx1 - mx0} 0 0 {my1 - my0} {mx0} {my0} cm /Im0 Do Q " if page.scan else ""
        content = stream(scan + page.content)
        annot_ids: list[int] = []
        page_id = add("")  # filled below, once its widgets exist
        for f in page.fields:
            x0, y0, x1, y1 = f.rect
            bx0, by0, bx1, by1 = f.bbox or (0, 0, x1 - x0, y1 - y0)
            form = (
                f"/Type /XObject /Subtype /Form /BBox [{bx0} {by0} {bx1} {by1}] "
                + (f"/Matrix [{f.matrix}] " if f.matrix else "")
                + f"/Resources << /Font << /Helv {font} 0 R /NoMap {no_map} 0 R >> >>"
            )
            ap = ""
            if f.states is not None:
                normal = " ".join(f"/{s} {stream(c, form)} 0 R" for s, c in f.states.items())
                ap = f"/AP << /N << {normal} >> >> /AS /{f.state}"
            elif f.ap is not None:
                ap = f"/AP << /N {stream(f.ap, form)} 0 R >>"
            v = "" if f.value is None else f"/V ({f.value})"
            kind = f"/FT /{f.ft} {v} /Ff {f.ff}" + (f" /Opt {f.opt}" if f.opt else "")
            common = (
                f"/Type /Annot /Subtype /Widget /Rect [{x0} {y0} {x1} {y1}] "
                f"/F {f.flags} /P {page_id} 0 R {ap}"
            )
            if f.on_parent:
                parent = add("")
                kid = add(f"<< {common} /Parent {parent} 0 R >>")
                name = f"/T (p{parent})" if f.named else ""
                objs[parent - 1] = f"<< {name} {kind} /Kids [{kid} 0 R] >>".encode()
                field_ids.append(parent)
                annot_ids.append(kid)
            else:
                name = f"/T (f{len(objs) + 1})" if f.named else ""
                wid = add(f"<< {common} {name} {kind} >>")
                field_ids.append(wid)
                annot_ids.append(wid)
        annots = " ".join(f"{a} 0 R" for a in annot_ids)
        objs[page_id - 1] = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [{mx0} {my0} {mx1} {my1}] "
            f"/Rotate {page.rotate} /Contents {content} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> /XObject << /Im0 {image} 0 R >> >> "
            f"/Annots [{annots}] >>"
        ).encode()
        page_ids.append(page_id)
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objs[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    form_dict = ""
    if acroform:
        fields = " ".join(f"{i} 0 R" for i in field_ids)
        form_dict = f"/AcroForm << /Fields [{fields}] /DR << /Font << /Helv {font} 0 R >> >> >>"
    objs[catalog - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R {form_dict} >>".encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objs) + 1,
        catalog,
        xref,
    )
    path.write_bytes(bytes(out))


def _ap(text: str, x: float = 2, y: float = 6, size: int = 12) -> str:
    return f"/Tx BMC BT /Helv {size} Tf {x} {y} Td ({text}) Tj ET EMC"


def _ap_no_map(glyphs: int) -> str:
    """An appearance that draws ``glyphs`` glyphs pdfminer cannot read."""
    codes = "".join(f"{i:04X}" for i in range(1, glyphs + 1))
    return f"/Tx BMC BT /NoMap 12 Tf 2 6 Td <{codes}> Tj ET EMC"


def _px(rect: Box, *, rotate: int = 0, mediabox: Box = (0, 0, PAGE_W, PAGE_H)) -> Box:
    """Where the PDF-point ``rect`` lands in the page image, in pixels (left,
    top, right, bottom). ``/Rotate`` turns the page clockwise for display."""
    x0, y0, x1, y1 = mediabox

    def point(x: float, y: float) -> tuple[float, float]:
        if rotate == 90:
            return y - y0, x - x0
        if rotate == 180:
            return x1 - x, y - y0
        if rotate == 270:
            return y1 - y, x1 - x
        return x - x0, y1 - y

    corners = [point(x, y) for x in (rect[0], rect[2]) for y in (rect[1], rect[3])]
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return min(xs) * SCALE, min(ys) * SCALE, max(xs) * SCALE, max(ys) * SCALE


def _extract(pdf: Path, tmp_path: Path) -> tuple[Any, dict[int, list[dict[str, Any]]]]:
    out = tmp_path / "page_text"
    result = extract_text_digital(pdf, out, file_id="f", dpi=DPI)
    pages = {}
    for f in sorted(out.glob("page_*.json")):
        payload = json.loads(f.read_text())
        pages[payload["page"]] = payload["words"]
    return result, pages


def _inside(box: list[int], rect_px: Box, tol: float = 12) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return rect_px[0] - tol <= cx <= rect_px[2] + tol and rect_px[1] - tol <= cy <= rect_px[3] + tol


NAME = (150.0, 690.0, 400.0, 712.0)
AMOUNT = (150.0, 650.0, 400.0, 672.0)


def test_filled_text_fields_become_words_in_their_widget(tmp_path: Path) -> None:
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                content=(
                    "BT /F1 12 Tf 72 696 Td (Name:) Tj ET BT /F1 12 Tf 72 656 Td (Amount:) Tj ET"
                ),
                fields=[
                    Field(NAME, "Jane Example", _ap("Jane Example")),
                    # Right-aligned: the appearance draws the amount at the right end.
                    Field(AMOUNT, "1,250.00", _ap("1,250.00", x=190)),
                ],
            )
        ],
    )
    result, pages = _extract(pdf, tmp_path)

    words = pages[1]
    texts = [w["t"] for w in words]
    # Each value comes right after its label on the same line.
    assert texts == ["Name", ":", "Jane", "Example", "Amount", ":", "1", ",", "250", ".", "00"]
    assert result.form_field_words == 7
    assert result.total_words == 11
    assert result.to_summary()["form_field_words"] == 7

    jane = words[2]
    assert _inside(jane["l"], _px(NAME))
    # The glyph box starts where the appearance draws the text (2 pt in).
    assert abs(jane["l"][0] - (NAME[0] + 2) * SCALE) <= 2
    # The amount box sits where it is drawn (right part of the field), not
    # spread over the whole field.
    amount_left = words[6]["l"][0]
    assert amount_left > (NAME[0] + 150) * SCALE
    assert all(_inside(w["l"], _px(AMOUNT)) for w in words[6:])
    # Page content words carry their style facts; form words carry none.
    content = [w for w in words if w["t"] in {"Name", "Amount"}]
    assert all(w["s"]["sz"] == 12.0 for w in content)
    form = words[2:4] + words[6:]
    assert not any("s" in w for w in form)


def _baseline(words: list[dict[str, Any]]) -> float | None:
    """The page body size grounding sizes ``dg:style`` against, from page_text
    words loaded as :func:`dgml_core.textmatch.load_page_words` does."""
    return _page_baseline(
        [
            Word(
                idx=i,
                text=w["t"],
                text_norm=fuzzy_norm(w["t"]),
                left=w["l"][0],
                top=w["l"][1],
                right=w["l"][2],
                bottom=w["l"][3],
                size=(w.get("s") or {}).get("sz"),
            )
            for i, w in enumerate(words)
        ]
    )


def test_form_words_do_not_move_the_page_body_size(tmp_path: Path) -> None:
    """Labels at 9 pt, long values at 12 pt: the filled form and the same page
    without its form give the labels the same body size."""
    content = (
        "BT /F1 9 Tf 72 696 Td (Name:) Tj ET "
        "BT /F1 9 Tf 72 656 Td (Notes:) Tj ET "
        "BT /F1 9 Tf 72 616 Td (Signed by the applicant) Tj ET"
    )
    fields = [
        Field(NAME, "Jane Alexandra Example", _ap("Jane Alexandra Example")),
        Field(AMOUNT, "Paid in full in March", _ap("Paid in full in March")),
    ]
    filled, blank = tmp_path / "filled.pdf", tmp_path / "blank.pdf"
    _write_form_pdf(filled, [Page(content=content, fields=fields)])
    _write_form_pdf(blank, [Page(content=content, fields=fields)], acroform=False)
    result, pages = _extract(filled, tmp_path / "a")
    _, blank_pages = _extract(blank, tmp_path / "b")
    assert result.form_field_words == 8
    assert _baseline(pages[1]) == _baseline(blank_pages[1]) == 9.0


def test_fields_that_show_nothing_new_add_no_words(tmp_path: Path) -> None:
    dup = (150.0, 600.0, 400.0, 622.0)
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                # The page content already prints the value inside the field.
                content="BT /F1 12 Tf 152 606 Td (ACME Corp) Tj ET",
                fields=[
                    Field(NAME, None, _ap("")),  # empty value
                    Field(NAME, "   ", _ap("   ")),  # blank value
                    Field(AMOUNT, "Hidden", _ap("Hidden"), flags=2 | 4),  # Hidden flag
                    Field(AMOUNT, "Not printed", _ap("Not printed"), flags=0),  # no Print flag
                    Field(AMOUNT, "No appearance", None),  # nothing drawn
                    Field(AMOUNT, "Nothing drawn", "/Tx BMC EMC"),  # appearance draws no glyphs
                    Field(AMOUNT, "Yes", _ap("4"), ft="Btn"),  # a check box
                    Field(AMOUNT, "B", _ap("A B C"), ft="Ch"),  # a list box
                    Field(dup, "ACME Corp", _ap("ACME Corp")),  # already in page content
                ],
            )
        ],
    )
    result, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == ["ACME", "Corp"]
    assert result.form_field_words == 0
    assert "form_field_words" not in result.to_summary()


def test_password_fields_add_no_words(tmp_path: Path) -> None:
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                fields=[
                    # These appearances draw the value. A password field's value
                    # is a secret (readers must not show or store it), so it
                    # never goes into page_text.
                    Field(NAME, "s3cret", _ap("s3cret"), ff=PASSWORD),
                    # The password flag on the parent field.
                    Field(AMOUNT, "hunter2", _ap("hunter2"), ff=PASSWORD, on_parent=True),
                    Field((150.0, 610.0, 400.0, 632.0), "Visible", _ap("Visible")),
                ]
            )
        ],
    )
    result, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == ["Visible"]
    assert result.form_field_words == 1


def test_widgets_without_a_field_name_add_no_words(tmp_path: Path) -> None:
    """Ghostscript does not draw a widget that has no /T on itself or on a
    parent field, so the page image does not show its value."""
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                fields=[
                    Field(NAME, "Nameless", _ap("Nameless"), named=False),
                    Field(AMOUNT, "Orphan kid", _ap("Orphan kid"), named=False, on_parent=True),
                    # A kid widget without its own /T, under a named parent: drawn.
                    Field((150.0, 610.0, 400.0, 632.0), "Kid", _ap("Kid"), on_parent=True),
                ]
            )
        ],
    )
    result, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == ["Kid"]
    assert result.form_field_words == 1


def test_inherited_value_multiline_and_clipping(tmp_path: Path) -> None:
    notes = (100.0, 400.0, 400.0, 440.0)
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                fields=[
                    # T, FT and V on the parent field, the widget is its kid.
                    Field(NAME, "Inherited Value", _ap("Inherited Value"), on_parent=True),
                    # Two lines, plus a third line scrolled out of the field.
                    Field(
                        notes,
                        "Line one\rLine two",
                        "/Tx BMC BT /Helv 10 Tf 2 26 Td (Line one) Tj 0 -14 Td (Line two) Tj "
                        "0 -40 Td (Clipped) Tj ET EMC",
                        ff=1 << 12,
                    ),
                ]
            )
        ],
    )
    _, pages = _extract(pdf, tmp_path)
    words = pages[1]
    assert [w["t"] for w in words] == ["Inherited", "Value", "Line", "one", "Line", "two"]
    line_one, line_two = words[2], words[4]
    assert line_one["l"][3] <= line_two["l"][1] + 2  # first line above the second
    assert all(_inside(w["l"], _px(notes)) for w in words[2:])


OFFSET_MEDIABOX = (100.0, 100.0, 712.0, 892.0)


@pytest.mark.parametrize("rotate", [0, 90, 180, 270])
def test_rotated_page_with_offset_mediabox_uses_the_page_image_space(
    tmp_path: Path, rotate: int
) -> None:
    """The field words land where the page image shows the field, in the same
    space as the page content words, whatever the /Rotate and wherever the
    MediaBox starts."""
    label_rect = (170.0, 790.0, 210.0, 812.0)
    value_rect = (250.0, 790.0, 500.0, 812.0)
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                content="BT /F1 12 Tf 172 796 Td (Label:) Tj ET",
                fields=[Field(value_rect, "ROTATED", _ap("ROTATED"))],
                rotate=rotate,
                mediabox=OFFSET_MEDIABOX,
            )
        ],
    )
    result, pages = _extract(pdf, tmp_path)
    words = pages[1]
    # pdfminer lays text on a page turned by 90 or 270 out one glyph per word
    # (page content and field values alike).
    label = [w for w in words if w["t"] in set("Label:") | {"Label"}]
    value = [w for w in words if w not in label]
    assert sorted("".join(w["t"] for w in label)) == sorted("Label:")
    assert sorted("".join(w["t"] for w in value)) == sorted("ROTATED")
    assert result.form_field_words == len(value)
    label_px = _px(label_rect, rotate=rotate, mediabox=OFFSET_MEDIABOX)
    value_px = _px(value_rect, rotate=rotate, mediabox=OFFSET_MEDIABOX)
    assert all(_inside(w["l"], label_px, tol=4) for w in label)
    assert all(_inside(w["l"], value_px, tol=4) for w in value)


def test_appearance_bbox_and_matrix_map_onto_the_widget(tmp_path: Path) -> None:
    """PDF 32000-1 12.5.5: the appearance BBox, transformed by its Matrix, is
    fitted onto the widget Rect."""
    shifted = (150.0, 690.0, 400.0, 712.0)
    upright = (300.0, 300.0, 322.0, 500.0)  # a narrow, tall field written bottom to top
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                fields=[
                    # The BBox does not start at 0, 0: the text 2 pt in from its corner.
                    Field(
                        shifted,
                        "Shifted",
                        "/Tx BMC BT /Helv 12 Tf 102 106 Td (Shifted) Tj ET EMC",
                        bbox=(100.0, 100.0, 350.0, 122.0),
                    ),
                    # A field turned by 90 degrees (/MK /R 90): the BBox is the
                    # unturned size, the Matrix turns it.
                    Field(
                        upright,
                        "Turned",
                        _ap("Turned"),
                        bbox=(0.0, 0.0, 200.0, 22.0),
                        matrix="0 1 -1 0 0 0",
                    ),
                ]
            )
        ],
    )
    _, pages = _extract(pdf, tmp_path)
    words = pages[1]
    assert words[0]["t"] == "Shifted"
    assert _inside(words[0]["l"], _px(shifted), tol=0)
    assert abs(words[0]["l"][0] - (shifted[0] + 2) * SCALE) <= 2
    turned = words[1:]
    assert sorted("".join(w["t"] for w in turned)) == sorted("Turned")
    assert all(_inside(w["l"], _px(upright), tol=0) for w in turned)
    # Written upwards from 2 pt above the bottom of the field.
    assert max(w["l"][3] for w in turned) <= (PAGE_H - upright[1] - 1) * SCALE
    assert min(w["l"][1] for w in turned) >= (PAGE_H - upright[1] - 60) * SCALE


def test_appearance_state_picks_the_drawn_appearance(tmp_path: Path) -> None:
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                fields=[
                    Field(
                        NAME,
                        "Chosen",
                        None,
                        states={"Off": _ap("Other"), "On": _ap("Chosen")},
                        state="On",
                    )
                ]
            )
        ],
    )
    _, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == ["Chosen"]


def test_pdf_without_acroform_is_unchanged(tmp_path: Path) -> None:
    plain = tmp_path / "plain.pdf"
    _write_text_pdf(plain, ["Hello world"])
    assert form_field_words(plain) == {}

    # Widgets present but no /AcroForm in the catalog: not read.
    orphan = tmp_path / "orphan.pdf"
    _write_form_pdf(orphan, [Page(fields=[Field(NAME, "Orphan", _ap("Orphan"))])], acroform=False)
    assert form_field_words(orphan) == {}
    result, pages = _extract(orphan, tmp_path)
    assert pages[1] == [] and result.form_field_words == 0


def test_broken_form_does_not_fail_extraction(tmp_path: Path) -> None:
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                content="BT /F1 12 Tf 72 696 Td (Name:) Tj ET",
                fields=[Field(NAME, "Good", _ap("Good"))],
            )
        ],
    )
    # Point the first field's /AP at an object that does not exist.
    data = pdf.read_bytes().replace(b"/AP << /N ", b"/AP << /N 999", 1)
    pdf.write_bytes(data)
    result, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == ["Name", ":"]
    assert result.form_field_words == 0


@pytest.mark.parametrize("failures", [1, 2])
def test_a_widget_that_cannot_be_read_costs_only_that_widget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failures: int,
) -> None:
    """The first ``failures`` appearances raise. The other fields are still
    added, extraction succeeds, and the document logs one warning that
    counts the skipped widgets (each one is logged at DEBUG)."""
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                content="BT /F1 12 Tf 72 696 Td (Name:) Tj ET",
                fields=[
                    Field(NAME, "First", _ap("First")),
                    Field(AMOUNT, "Second", _ap("Second")),
                    Field((150.0, 610.0, 400.0, 632.0), "Third", _ap("Third")),
                ],
            )
        ],
    )
    real_render = form_fields._AppearanceRenderer.render
    calls = 0

    def flaky_render(self: Any, *args: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls <= failures:
            raise RuntimeError("unreadable appearance")
        return real_render(self, *args)

    monkeypatch.setattr(form_fields._AppearanceRenderer, "render", flaky_render)
    with caplog.at_level(logging.DEBUG, logger="dgml_core.form_fields"):
        result, pages = _extract(pdf, tmp_path)

    kept = ["First", "Second", "Third"][failures:]
    assert [w["t"] for w in pages[1]] == ["Name", ":", *kept]
    assert result.form_field_words == len(kept)
    records = [r for r in caplog.records if r.name == "dgml_core.form_fields"]
    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage().startswith(f"{failures} form field widgets of form.pdf")
    assert len([r for r in records if r.levelno == logging.DEBUG]) == failures


def test_a_parse_error_keeps_the_fields_read_before_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Reading the form stops at page 2: page 1 keeps its field words, and
    the document logs one warning."""
    label = "BT /F1 12 Tf 72 696 Td (Name:) Tj ET"
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(content=label, fields=[Field(NAME, "First", _ap("First"))]),
            Page(content=label, fields=[Field(NAME, "Second", _ap("Second"))]),
        ],
    )
    real_page_ctm = form_fields._page_ctm
    calls = 0

    def page_ctm_then_fail(pdf_page: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("broken page")
        return real_page_ctm(pdf_page)

    monkeypatch.setattr(form_fields, "_page_ctm", page_ctm_then_fail)
    with caplog.at_level(logging.WARNING, logger="dgml_core.form_fields"):
        result, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == ["Name", ":", "First"]
    assert [w["t"] for w in pages[2]] == ["Name", ":"]
    assert result.form_field_words == 1
    [warning] = [r for r in caplog.records if r.name == "dgml_core.form_fields"]
    assert warning.levelno == logging.WARNING and "parse error" in warning.getMessage()


def test_a_failing_merge_keeps_the_page_content_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                content="BT /F1 12 Tf 72 696 Td (Name:) Tj ET",
                fields=[Field(NAME, "Jane", _ap("Jane"))],
            )
        ],
    )

    def broken_merge(words: list[dict[str, Any]], *_: Any, **__: Any) -> int:
        words.append({"t": "half-merged", "l": [0, 0, 1, 1]})
        raise RuntimeError("merge failed")

    monkeypatch.setattr(form_fields, "merge_form_field_words", broken_merge)
    with caplog.at_level(logging.WARNING, logger="dgml_core.text_extraction"):
        result, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == ["Name", ":"]
    assert result.form_field_words == 0
    assert result.total_words == 2
    [warning] = [r for r in caplog.records if r.name == "dgml_core.text_extraction"]
    assert warning.levelno == logging.WARNING
    assert "page 1 of form.pdf" in warning.getMessage()


def test_scanned_page_with_filled_fields_is_still_flagged(tmp_path: Path) -> None:
    """The form words join the page after the scan check: a full-page image
    with no text layer stays a scanned page for the OCR fallback, however
    many words its fields hold."""
    twelve = "1 2 3 4 5 6 7 8 9 10 11 12"
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(pdf, [Page(scan=True, fields=[Field(NAME, twelve, _ap(twelve, size=10))])])
    result, pages = _extract(pdf, tmp_path)
    assert result.defects == {1: PageTextDefect.IMAGE_ONLY}
    assert [w["t"] for w in pages[1]] == twelve.split()
    assert result.form_field_words == 12


def test_prorated_words_when_the_appearance_text_is_unreadable() -> None:
    rect = (100.0, 500.0, 300.0, 520.0)
    words = _prorate("AB 1234", rect, multiline=False)
    assert [w[0] for w in words] == ["AB", "1234"]
    (_, ab), (_, num) = words
    assert ab[0] == 100.0 and num[2] == 300.0
    assert ab[1] == 500.0 and ab[3] == 520.0

    multi = _prorate("one\rtwo", rect, multiline=True)
    assert [w[0] for w in multi] == ["one", "two"]
    assert multi[0][1][1] == multi[1][1][3]  # line two right below line one


def test_merge_skips_a_value_the_page_already_prints() -> None:
    rect = (100.0, 500.0, 300.0, 520.0)
    fw = FieldWords(rect=rect, words=[("ACME", (102.0, 504.0, 140.0, 516.0))])
    printed = [{"t": "acme", "l": [430, 1137, 580, 1183]}]
    assert merge_form_field_words(printed, [fw], page_h_pts=792, dpi=DPI) == 0
    elsewhere = [{"t": "ACME", "l": [10, 10, 50, 50]}]
    assert merge_form_field_words(elsewhere, [fw], page_h_pts=792, dpi=DPI) == 1
    assert elsewhere[-1]["t"] == "ACME"
    # On the field's line, but left of the field: not inside it.
    beside = [{"t": "ACME", "l": [10, 1137, 300, 1183]}]
    assert merge_form_field_words(beside, [fw], page_h_pts=792, dpi=DPI) == 1
    # A second widget drawing the same value at the same place adds nothing.
    twice = [{"t": "ACME", "l": [10, 10, 50, 50]}]
    assert merge_form_field_words(twice, [fw, fw], page_h_pts=792, dpi=DPI) == 1


LIST_BOX = (150.0, 560.0, 400.0, 600.0)


def test_combo_box_value_is_read_and_list_box_is_skipped(tmp_path: Path) -> None:
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                fields=[
                    # /V holds the export value; the appearance draws the option text.
                    Field(NAME, "TX", _ap("Texas"), ft="Ch", ff=COMBO, opt="[[(TX) (Texas)]]"),
                    # FT, V and the Combo flag on the parent field.
                    Field(
                        AMOUNT, "Monthly", _ap("Monthly", x=60), ft="Ch", ff=COMBO, on_parent=True
                    ),
                    # A list box draws all its options, so its words are not the value.
                    Field(
                        LIST_BOX,
                        "Beta",
                        "/Tx BMC BT /Helv 10 Tf 2 28 Td (Alpha) Tj 0 -12 Td (Beta) Tj "
                        "0 -12 Td (Gamma) Tj ET EMC",
                        ft="Ch",
                    ),
                ]
            )
        ],
    )
    result, pages = _extract(pdf, tmp_path)
    words = pages[1]
    assert [w["t"] for w in words] == ["Texas", "Monthly"]
    assert result.form_field_words == 2
    texas, monthly = words
    assert _inside(texas["l"], _px(NAME))
    assert abs(texas["l"][0] - (NAME[0] + 2) * SCALE) <= 2  # the drawn glyph box
    assert _inside(monthly["l"], _px(AMOUNT))
    assert abs(monthly["l"][0] - (AMOUNT[0] + 60) * SCALE) <= 2


def test_combo_box_fallback_spreads_the_display_text(tmp_path: Path) -> None:
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                fields=[
                    # The appearance text is unreadable: the words come from /V,
                    # shown as the display text /Opt pairs with the export value.
                    Field(
                        NAME,
                        "TX",
                        _ap_no_map(5),
                        ft="Ch",
                        ff=COMBO,
                        opt="[[(CA) (California)] [(TX) (Texas)]]",
                    ),
                    # Options without export values: /V is the text shown.
                    Field(
                        AMOUNT, "Weekly", _ap_no_map(6), ft="Ch", ff=COMBO, opt="[(Daily) (Weekly)]"
                    ),
                ]
            )
        ],
    )
    (texas, weekly) = form_field_words(pdf)[1]
    assert texas.prorated and weekly.prorated
    assert [t for t, _ in texas.words] == ["Texas"]
    assert [t for t, _ in weekly.words] == ["Weekly"]
    _, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == ["Texas", "Weekly"]
    assert _inside(pages[1][0]["l"], _px(NAME))

    # An export value with no /Opt entry stays as it is.
    assert _display_text({"Opt": [[b"CA", b"California"]]}, "NV") == "NV"
    assert _display_text({}, "NV") == "NV"


def test_value_under_its_label_follows_the_whole_label_line(tmp_path: Path) -> None:
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                # Labels over their boxes; each value is alone on its line.
                content=(
                    "BT /F1 10 Tf 72 720 Td (Applicant name) Tj ET "
                    "BT /F1 10 Tf 72 660 Td (Mailing address) Tj ET"
                ),
                fields=[
                    Field((72.0, 630.0, 400.0, 652.0), "1 Main Street", _ap("1 Main Street")),
                    # Only "Applicant" is over the first value word.
                    Field((72.0, 690.0, 400.0, 712.0), "Jane Example", _ap("Jane Example")),
                ],
            )
        ],
    )
    _, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == [
        "Applicant",
        "name",
        "Jane",
        "Example",
        "Mailing",
        "address",
        "1",
        "Main",
        "Street",
    ]


def test_value_with_no_content_before_it_goes_to_the_end(tmp_path: Path) -> None:
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            # The only page content is below the field.
            Page(
                content="BT /F1 10 Tf 72 100 Td (Page footer) Tj ET",
                fields=[Field(NAME, "Top value", _ap("Top value"))],
            ),
            # No page content at all; the widgets are listed bottom first.
            Page(
                fields=[Field(AMOUNT, "Second", _ap("Second")), Field(NAME, "First", _ap("First"))]
            ),
        ],
    )
    result, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == ["Page", "footer", "Top", "value"]
    assert [w["t"] for w in pages[2]] == ["First", "Second"]
    assert result.form_field_words == 4


def test_fields_that_follow_the_same_word_read_left_to_right_then_down(tmp_path: Path) -> None:
    pdf = tmp_path / "form.pdf"
    _write_form_pdf(
        pdf,
        [
            Page(
                content=(
                    "BT /F1 12 Tf 72 696 Td (Phone:) Tj ET BT /F1 10 Tf 72 620 Td (Address) Tj ET"
                ),
                fields=[
                    # Two fields on the label's line, listed right one first.
                    Field((220.0, 690.0, 300.0, 712.0), "0100", _ap("0100")),
                    Field((150.0, 690.0, 210.0, 712.0), "555", _ap("555")),
                    # Two lines of one address under its label, listed bottom first.
                    Field((72.0, 570.0, 400.0, 592.0), "Springfield", _ap("Springfield")),
                    Field((72.0, 595.0, 400.0, 617.0), "1 Main St", _ap("1 Main St")),
                ],
            )
        ],
    )
    _, pages = _extract(pdf, tmp_path)
    assert [w["t"] for w in pages[1]] == [
        "Phone",
        ":",
        "555",
        "0100",
        "Address",
        "1",
        "Main",
        "St",
        "Springfield",
    ]


def _field_at(text: str, box_px: list[int]) -> FieldWords:
    """A one-word field drawn at ``box_px`` on a page read at 72 dpi (one
    pixel per point) that is 1000 pt tall."""
    left, top, right, bottom = box_px
    box = (float(left), 1000.0 - bottom, float(right), 1000.0 - top)
    return FieldWords(rect=box, words=[(text, box)])


def _merged(words: list[dict[str, Any]], fields: list[FieldWords]) -> list[str]:
    merge_form_field_words(words, fields, page_h_pts=1000, dpi=72)
    return [w["t"] for w in words]


def test_value_follows_the_nearest_label_on_its_line() -> None:
    """Two labels and two values on one line, and pdfminer lists the right
    label first: each value still follows the label just before it. One label
    is taller than its value and one is shorter; a colon may end slightly past
    the value's left edge."""
    words = [
        {"t": "Phone", "l": [300, 96, 350, 114]},
        {"t": ":", "l": [350, 96, 354, 114]},
        {"t": "Name", "l": [10, 102, 50, 110]},
        {"t": ":", "l": [50, 102, 61, 110]},
        {"t": "Footer", "l": [10, 300, 60, 312]},
    ]
    fields = [_field_at("Jane", [60, 100, 90, 112]), _field_at("5550100", [360, 100, 420, 112])]
    assert _merged(words, fields) == ["Phone", ":", "5550100", "Name", ":", "Jane", "Footer"]


def test_value_follows_the_nearest_line_above_it() -> None:
    """A heading far above the field comes later in the list than the label
    right over the field, and a word in another column ends lower than the
    label: the value follows the label line. The label's descent reaches
    slightly below the value's top."""
    words = [
        {"t": "Applicant", "l": [10, 200, 80, 222]},
        {"t": "name", "l": [84, 200, 120, 222]},
        {"t": "Other", "l": [300, 205, 350, 223]},
        {"t": "Section", "l": [10, 100, 70, 115]},
        {"t": "A", "l": [74, 100, 84, 115]},
    ]
    fields = [_field_at("Jane", [10, 220, 100, 232])]
    assert _merged(words, fields) == ["Applicant", "name", "Jane", "Other", "Section", "A"]


def test_page_content_words_keep_their_order(tmp_path: Path) -> None:
    content = (
        "BT /F1 14 Tf 72 750 Td (Order form) Tj ET "
        "BT /F1 12 Tf 72 696 Td (Name:) Tj ET "
        "BT /F1 12 Tf 72 670 Td (Date:) Tj ET "
        "BT /F1 10 Tf 72 640 Td (Company) Tj ET "
        "BT /F1 10 Tf 72 400 Td (Signature) Tj ET"
    )
    fields = [
        Field((110.0, 664.0, 300.0, 686.0), "2021-01-11", _ap("2021-01-11")),
        Field((110.0, 690.0, 300.0, 712.0), "Jane Example", _ap("Jane Example")),
        Field((72.0, 610.0, 400.0, 632.0), "ACME Corp", _ap("ACME Corp")),
        Field((72.0, 760.0, 300.0, 782.0), "Copy 2", _ap("Copy 2")),  # above all content
    ]
    with_form, plain = tmp_path / "form.pdf", tmp_path / "plain.pdf"
    _write_form_pdf(with_form, [Page(content=content, fields=fields)])
    _write_form_pdf(plain, [Page(content=content, fields=fields)], acroform=False)
    result, pages = _extract(with_form, tmp_path / "a")
    _, plain_pages = _extract(plain, tmp_path / "b")

    field_texts = {"2021", "-", "01", "11", "Jane", "Example", "ACME", "Corp", "Copy", "2"}
    words = pages[1]
    assert [w for w in words if w["t"] not in field_texts] == plain_pages[1]
    assert len(words) - len(plain_pages[1]) == result.form_field_words == 11
    assert [w["t"] for w in words] == [
        "Order",
        "form",
        "Name",
        ":",
        "Jane",
        "Example",
        "Date",
        ":",
        "2021",
        "-",
        "01",
        "-",
        "11",
        "Company",
        "ACME",
        "Corp",
        "Signature",
        "Copy",
        "2",
    ]


def test_ocr_fallback_keeps_the_form_word_count_of_pages_it_does_not_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OCR fallback rewrites only the scanned pages; the summary keeps the
    form-field words of the other pages and drops those of the OCR'd pages."""
    import contextlib
    from collections.abc import Iterator

    from dgml_core import ocr
    from dgml_core.text_extraction import (
        PAGE_TEXT_FILENAME,
        ExtractDigitalResult,
    )

    out = tmp_path / "page_text"
    out.mkdir()
    for page, n in ((1, 3), (2, 1)):
        words = [{"t": f"w{i}", "l": [0, 0, 10, 10]} for i in range(n)]
        (out / PAGE_TEXT_FILENAME.format(page=page)).write_text(json.dumps({"words": words}))

    def fake_ocr(_pdf: Path, output_dir: Path, **kwargs: object) -> None:
        for page in kwargs["pages"]:  # type: ignore[attr-defined]
            words = [{"t": "ocr", "l": [0, 0, 10, 10]}]
            path = output_dir / PAGE_TEXT_FILENAME.format(page=page)
            path.write_text(json.dumps({"words": words}))

    class _Blobs:
        @contextlib.contextmanager
        def materialize_dir(self, _prefix: str) -> Iterator[Path]:
            yield tmp_path

    class _Workspace:
        blobs = _Blobs()

    import types

    config = types.SimpleNamespace(provider="fake")
    monkeypatch.setattr(ocr, "load_ocr_config", lambda _ws: config)
    monkeypatch.setattr(ocr, "extract_text_ocr", fake_ocr)
    result = ExtractDigitalResult(
        pages_written=2,
        pages_with_words=2,
        total_words=4,
        defects={2: next(iter(PageTextDefect))},
        form_field_words=3,
        form_field_words_by_page={1: 2, 2: 1},
    )
    recovered = ocr.recover_unusable_pages(
        _Workspace(),  # type: ignore[arg-type]
        tmp_path / "x.pdf",
        out,
        result,
        file_id="f1",
        pages_prefix="p",
    )
    assert recovered.ocr_fallback_pages == [2]
    assert recovered.form_field_words == 2
    assert recovered.to_summary()["form_field_words"] == 2
