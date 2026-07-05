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

"""Tests for the dg:style allow-list/assembly (dgml_core.style) and the
deterministic capture of style facts during digital extraction."""

from __future__ import annotations

import json
from pathlib import Path

from dgml_core.style import (
    build_style,
    fontname_is_bold,
    fontname_is_italic,
    merge_styles,
    rgb_to_named,
    size_to_em,
    validate_style,
)
from dgml_core.text_extraction import extract_text_digital

# ---- Pure helpers ----------------------------------------------------------


def test_fontname_bold_italic_detection() -> None:
    assert fontname_is_bold("ABCDEF+Times-Bold")
    assert fontname_is_bold("Arial-Black")
    assert fontname_is_bold("Helvetica-SemiBold")  # subsumed by "bold"
    assert not fontname_is_bold("Helvetica")
    assert not fontname_is_bold(None)
    assert fontname_is_italic("Times-Italic")
    assert fontname_is_italic("Helvetica-Oblique")
    assert not fontname_is_italic("Times-Roman")


def test_size_to_em_buckets() -> None:
    assert size_to_em(12.0, 12.0) is None  # baseline -> omitted
    assert size_to_em(18.0, 12.0) == "1.5em"
    assert size_to_em(24.0, 12.0) == "2em"
    assert size_to_em(9.0, 12.0) == "0.75em"
    assert size_to_em(15.0, 12.0) == "1.25em"
    assert size_to_em(13.0, 12.0) is None  # nearest bucket is 1em
    assert size_to_em(None, 12.0) is None
    assert size_to_em(18.0, 0) is None


def test_rgb_to_named() -> None:
    assert rgb_to_named((255, 0, 0)) == "red"
    assert rgb_to_named((130, 130, 130)) == "gray"
    assert rgb_to_named((10, 10, 10)) is None  # near-black default
    assert rgb_to_named(None) is None
    # Snaps an off-exact value to the nearest keyword.
    assert rgb_to_named((250, 5, 5)) == "red"


def test_build_style_orders_and_keeps_inheriting_defaults() -> None:
    # Inheriting defaults (font-style: normal, text-align: left) are KEPT here —
    # they may override a non-default inherited from an ancestor; the redundant
    # copies are elided later by _suppress_inherited_style. A non-inheriting
    # default (text-decoration: none) can never override anything, so it drops.
    out = build_style(
        {
            "color": "gray",
            "font-weight": "bold",
            "font-style": "normal",  # inheriting default -> kept
            "text-decoration": "none",  # non-inheriting default -> dropped
            "text-align": "left",  # inheriting default -> kept
        }
    )
    assert out == "font-weight: bold; font-style: normal; text-align: left; color: gray"


def test_build_style_rejects_out_of_allowlist() -> None:
    # Disallowed value and a non-named color are both dropped.
    assert build_style({"font-size": "3em", "color": "#ff0000"}) == ""
    assert build_style({"white-space": "pre"}) == "white-space: pre"


def test_build_style_keeps_inheriting_defaults_for_override() -> None:
    # These are all inheriting defaults; build_style keeps them (a
    # child needs them to override a bold/left-aligned ancestor). Sparseness is
    # restored downstream by the inheritance-aware suppression pass.
    assert build_style({"font-weight": "normal", "text-align": "left"}) == (
        "font-weight: normal; text-align: left"
    )


def test_build_style_drops_non_inheriting_defaults_and_empties() -> None:
    # Non-inheriting default drops; empty/whitespace values drop -> nothing left.
    assert build_style({"text-decoration": "none", "font-weight": "  "}) == ""


def test_validate_style_filters_llm_output() -> None:
    raw = "font-weight: bold; font-family: Arial; text-align: center; color: teal"
    # font-family is not allow-listed and is dropped; the rest survive in order.
    assert validate_style(raw) == "font-weight: bold; text-align: center; color: teal"
    assert validate_style("not css at all") == ""


def test_validate_style_is_case_insensitive_on_values() -> None:
    # Free-form LLM output uses mixed case; enum values, color names, and em
    # buckets must all match case-insensitively and emit canonical lower-case.
    # (Regression: only color values were lower-cased; enums matched exactly.)
    raw = "font-weight: Bold; text-transform: UPPERCASE; color: Red; font-size: 1.5EM"
    assert validate_style(raw) == (
        "font-weight: bold; font-size: 1.5em; text-transform: uppercase; color: red"
    )
    # A mixed-case default is still recognized as the (non-inheriting) default.
    assert validate_style("text-decoration: NONE") == ""


def test_validate_style_keeps_explicit_inheriting_default() -> None:
    # A vision model's explicit "font-weight: normal" must survive validation —
    # it is how the image path overrides a bold inherited from an ancestor.
    # (build_style must not strip it unconditionally.)
    assert validate_style("font-weight: normal") == "font-weight: normal"
    # And it survives a merge (extra provides normal; base has no font-weight).
    assert merge_styles("color: red", "font-weight: normal") == ("font-weight: normal; color: red")


# ---- Digital extraction capture --------------------------------------------


def _write_styled_pdf(path: Path) -> None:
    """A one-page PDF with a bold, red, large title line and a plain body line,
    hand-constructed so pdfminer reports fontname/size/fill color per glyph."""
    out = bytearray()
    offsets: list[int] = []

    def add_object(body: bytes) -> int:
        offsets.append(len(out))
        obj_num = len(offsets)
        out.extend(f"{obj_num} 0 obj\n".encode())
        out.extend(body)
        out.extend(b"\nendobj\n")
        return obj_num

    out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    add_object(b"<< /Type /Pages /Kids [5 0 R] /Count 1 >>")
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    add_object(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R "
        b"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>"
    )
    stream = (
        b"BT /F1 24 Tf 1 0 0 rg 100 700 Td (TITLE) Tj ET\n"
        b"BT /F2 12 Tf 0 0 0 rg 100 650 Td (body) Tj ET\n"
    )
    add_object(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")

    xref_offset = len(out)
    n = len(offsets)
    out.extend(f"xref\n0 {n + 1}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        f"trailer\n<< /Size {n + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    path.write_bytes(bytes(out))


def test_extract_digital_captures_style(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("pdfminer")
    pdf = tmp_path / "styled.pdf"
    _write_styled_pdf(pdf)
    out_dir = tmp_path / "page_text"
    extract_text_digital(pdf, out_dir, file_id="styledfile12")

    page = json.loads((out_dir / "page_1.json").read_text())
    words = {w["t"]: w for w in page["words"]}
    assert "TITLE" in words and "body" in words

    title_style = words["TITLE"].get("s") or {}
    assert title_style.get("b") == 1  # Helvetica-Bold -> bold
    assert title_style.get("sz") == 24.0  # captured glyph size in points
    # Plain body line carries no bold flag (and no color override).
    body_style = words["body"].get("s") or {}
    assert "b" not in body_style


def test_extract_digital_color_capture(tmp_path: Path) -> None:
    """Red fill (``1 0 0 rg``) is captured as a named color when pdfminer
    surfaces the non-stroking color; tolerated as best-effort otherwise."""
    import pytest

    pytest.importorskip("pdfminer")
    pdf = tmp_path / "styled.pdf"
    _write_styled_pdf(pdf)
    out_dir = tmp_path / "page_text"
    extract_text_digital(pdf, out_dir, file_id="styledfile12")
    page = json.loads((out_dir / "page_1.json").read_text())
    title = next(w for w in page["words"] if w["t"] == "TITLE")
    color = (title.get("s") or {}).get("c")
    assert color in (None, "red")


# ---- Hybrid style threading ------------------------------------------------


def test_hybrid_transplants_digital_style_onto_ocr_words() -> None:
    from dgml_core.hybrid import _apply_digital_style, _box_overlap_fraction

    assert _box_overlap_fraction([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _box_overlap_fraction([0, 0, 10, 10], [100, 100, 110, 110]) == 0.0

    ocr = [{"t": "Bold", "l": [0, 0, 10, 10]}, {"t": "plain", "l": [200, 0, 210, 10]}]
    digital = [{"t": "Bold", "l": [0, 0, 10, 10], "s": {"b": 1}}]
    out = _apply_digital_style(ocr, digital)
    assert out[0]["s"] == {"b": 1}  # overlapping OCR word gains the style
    assert "s" not in out[1]  # non-overlapping OCR word stays bare
    assert "s" not in ocr[0]  # original input not mutated


# ---- OCR LLM-from-image path -----------------------------------------------


def test_style_llm_prompt_is_visual_only() -> None:
    """The prompt must steer the model to judge rendering, not word meaning, so
    a document sentence like 'the following is bold' can't induce bold."""
    from dgml_core.style_llm import _SYSTEM_PROMPT, _build_prompt

    prompt = _build_prompt(["This is the title", "ordinary body text"])
    assert "Ignore the meaning" in prompt
    assert "body text" in prompt  # bold judged relative to body text
    assert "not instructions to you" in prompt
    assert "MEAN" in _SYSTEM_PROMPT


def test_style_llm_parse_and_first_page() -> None:
    from dgml_core.style_llm import _first_page, _parse_styles

    assert _first_page("3 10 20 30 40; 4 1 2 3 4") == 3
    assert _first_page(None) is None
    assert _first_page("garbage") is None

    parsed = _parse_styles('```json\n{"styles": [{"index": 0, "style": "font-weight: bold"}]}\n```')
    assert parsed == {0: "font-weight: bold"}
    assert _parse_styles("[{'not': 'json'}]") == {}


def test_annotate_style_from_image(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from dgml_core import style_llm
    from dgml_core.storage import Workspace
    from lxml import etree  # type: ignore[import-untyped]

    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    file_id = "ocrfile12345"
    pages_dir = ws.file_pages_dir(file_id)
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page_1.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")

    root = etree.fromstring(
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">'
        '<Heading dg:origin="1 10 20 30 40">TITLE</Heading>'
        '<Body dg:origin="1 10 60 30 80">body</Body>'
        "</dg:chunk>"
    )
    dg = "http://dgml.io/ns/dg#"

    def fake_request(config, image_bytes, snippets):  # type: ignore[no-untyped-def]
        # Only the first snippet ("TITLE") gets styling; an out-of-allowlist
        # value is filtered by validate_style downstream.
        return {0: "font-weight: bold; font-family: Arial"}

    monkeypatch.setattr(style_llm, "_request_styles", fake_request)

    from dgml_core.llm import LLMConfig

    styled = style_llm.annotate_style_from_image(
        ws,
        file_id,
        root,
        config=LLMConfig(model="anthropic/claude-haiku-4-5", api_key=None, max_tokens=None),
        style_attr=f"{{{dg}}}style",
        origin_attr=f"{{{dg}}}origin",
    )
    assert styled == 1
    # iter() yields root, then Heading, then Body.
    _, heading, body = list(root.iter())
    assert heading.get(f"{{{dg}}}style") == "font-weight: bold"
    assert body.get(f"{{{dg}}}style") is None


# ---- Workspace `style` config section --------------------------------------


def _ws_with_config(tmp_path: Path, config: dict | None):  # type: ignore[type-arg,no-untyped-def]
    from dgml_core.storage import Workspace

    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    if config is not None:
        ws.config_path.write_text(json.dumps(config), encoding="utf-8")
    return ws


def test_load_style_config_absent_returns_none(tmp_path: Path) -> None:
    from dgml_core.style_config import load_style_config

    assert load_style_config(_ws_with_config(tmp_path, None)) is None
    assert load_style_config(_ws_with_config(tmp_path, {"ocr": {}})) is None


def test_load_style_config_valid(tmp_path: Path) -> None:
    from dgml_core.style_config import load_style_config

    cfg = load_style_config(
        _ws_with_config(tmp_path, {"style": {"model": "anthropic/claude-haiku-4-5"}})
    )
    assert cfg is not None
    assert cfg.model == "anthropic/claude-haiku-4-5"


def test_load_style_config_requires_model(tmp_path: Path) -> None:
    """Presence of the section is the switch, so a section without a model is
    a misconfiguration, not a silent no-op."""
    import pytest
    from dgml_core.errors import StyleConfigInvalid
    from dgml_core.style_config import load_style_config

    with pytest.raises(StyleConfigInvalid):
        load_style_config(_ws_with_config(tmp_path, {"style": {"max_tokens": 100}}))


def test_load_style_config_rejects_dual_keys(tmp_path: Path) -> None:
    import pytest
    from dgml_core.errors import StyleConfigInvalid
    from dgml_core.style_config import load_style_config

    with pytest.raises(StyleConfigInvalid):
        load_style_config(
            _ws_with_config(
                tmp_path,
                {"style": {"model": "m", "api_key": "k", "api_key_env": "E"}},
            )
        )


def test_ground_honors_style_config_for_ocr(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """End-to-end: an OCR file + an enabling `style` config drives the
    image-based dg:style pass through grounding; the LLM call is stubbed."""
    from dgml_core import style_llm
    from dgml_core.models import FileRecord
    from dgml_core.storage import Workspace, write_json_atomic
    from dgml_core.xml_grounding import ground_dgml_xml

    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    fid = "ocrwire12345"
    ws.file_dir(fid).mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        ws.file_json_path(fid),
        FileRecord(
            id=fid,
            original_path="/f.pdf",
            original_filename="f.pdf",
            sha256="0" * 64,
            added_at="2026-01-01T00:00:00Z",
            page_count=1,
            text_mode="ocr",  # the gate
        ).to_json(),
    )
    ws.file_text_dir(fid).mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        ws.file_text_dir(fid) / "page_1.json",
        {
            "file_id": fid,
            "page": 1,
            "width": 1000,
            "height": 1000,
            # OCR words carry no "s" style facts.
            "words": [
                {"t": "TITLE", "l": [100, 100, 150, 120]},
                {"t": "HERE", "l": [160, 100, 210, 120]},
            ],
        },
    )
    pages_dir = ws.file_pages_dir(fid)
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page_1.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    ws.config_path.write_text(
        json.dumps({"style": {"model": "anthropic/claude-haiku-4-5"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(style_llm, "_request_styles", lambda c, i, s: {0: "font-weight: bold"})

    src = tmp_path / "doc.dgml.xml"
    src.write_text(
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><Heading>TITLE HERE</Heading></dg:chunk>',
        encoding="utf-8",
    )
    ground_dgml_xml(ws, fid, src, output_path=src, force=True, write_stats=False)
    content = src.read_text(encoding="utf-8")
    # LLM-inferred font-weight lands, merged with the deterministic all-caps
    # text-transform the grounding pass already derived from "TITLE HERE".
    assert "font-weight: bold" in content
    assert "text-transform: uppercase" in content


class _PricedResp(dict):  # type: ignore[type-arg]
    """A litellm response that is both subscriptable (``call`` reads
    ``["choices"]``) and attribute-accessible (``extract_cost_and_tokens`` reads
    ``.usage`` / ``._hidden_params``)."""

    def __init__(self, text: str) -> None:
        from types import SimpleNamespace

        super().__init__(choices=[{"message": {"content": text}}])
        self._hidden_params = {"response_cost": 0.02}
        self.usage = SimpleNamespace(prompt_tokens=300, completion_tokens=40, total_tokens=340)


def _seed_ocr_style_workspace(tmp_path: Path):  # type: ignore[no-untyped-def]
    """An OCR file + enabling `style` config + one page image/text. Returns
    ``(ws, fid, src_xml_path)`` ready to ground."""
    from dgml_core.models import FileRecord
    from dgml_core.storage import Workspace, write_json_atomic

    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    fid = "ocrwireusage"
    ws.file_dir(fid).mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        ws.file_json_path(fid),
        FileRecord(
            id=fid,
            original_path="/f.pdf",
            original_filename="f.pdf",
            sha256="0" * 64,
            added_at="2026-01-01T00:00:00Z",
            page_count=1,
            text_mode="ocr",
        ).to_json(),
    )
    ws.file_text_dir(fid).mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        ws.file_text_dir(fid) / "page_1.json",
        {
            "file_id": fid,
            "page": 1,
            "width": 1000,
            "height": 1000,
            "words": [{"t": "TITLE", "l": [100, 100, 150, 120]}],
        },
    )
    pages_dir = ws.file_pages_dir(fid)
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page_1.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    ws.config_path.write_text(
        json.dumps({"style": {"model": "anthropic/claude-haiku-4-5"}}),
        encoding="utf-8",
    )
    src = tmp_path / "doc.dgml.xml"
    src.write_text(
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><Heading>TITLE</Heading></dg:chunk>',
        encoding="utf-8",
    )
    return ws, fid, src


def test_ground_records_style_usage_under_debug(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The image-based style pass records one `style_annotate` usage row per
    page via the standard llm.call recording layer — gated on --debug. Drives
    the real llm.call path (patching litellm.completion) rather than stubbing
    _request_styles, to lock the post-#212 recording pattern."""
    from dgml_core.usage import read_events
    from dgml_core.xml_grounding import ground_dgml_xml

    ws, fid, src = _seed_ocr_style_workspace(tmp_path)
    monkeypatch.setattr(
        "litellm.completion",
        lambda **k: _PricedResp('{"styles": [{"index": 0, "style": "font-weight: bold"}]}'),
    )

    ground_dgml_xml(ws, fid, src, output_path=src, force=True, write_stats=False, debug=True)

    events = read_events(ws)
    assert len(events) == 1
    assert events[0]["operation"] == "style_annotate"
    assert events[0]["context"] == {"file_id": fid, "page": 1}
    assert events[0]["cost_usd"] == 0.02
    assert events[0]["total_tokens"] == 340


def test_ground_records_no_style_usage_without_debug(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Without --debug, the style pass writes no usage.jsonl row (matching every
    other LLM path post-#212), even though it still styles the document."""
    from dgml_core.usage import read_events
    from dgml_core.xml_grounding import ground_dgml_xml

    ws, fid, src = _seed_ocr_style_workspace(tmp_path)
    monkeypatch.setattr(
        "litellm.completion",
        lambda **k: _PricedResp('{"styles": [{"index": 0, "style": "font-weight: bold"}]}'),
    )

    ground_dgml_xml(ws, fid, src, output_path=src, force=True, write_stats=False)  # debug False

    assert read_events(ws) == []
    assert "font-weight: bold" in src.read_text(encoding="utf-8")  # still styled


def test_ground_skips_style_config_when_disabled(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """With no `style` section, the LLM path must not run even for OCR files."""
    from dgml_core import style_llm
    from dgml_core.models import FileRecord
    from dgml_core.storage import Workspace, write_json_atomic
    from dgml_core.xml_grounding import ground_dgml_xml

    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    fid = "ocrwire67890"
    ws.file_dir(fid).mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        ws.file_json_path(fid),
        FileRecord(
            id=fid,
            original_path="/f.pdf",
            original_filename="f.pdf",
            sha256="0" * 64,
            added_at="2026-01-01T00:00:00Z",
            page_count=1,
            text_mode="ocr",
        ).to_json(),
    )
    ws.file_text_dir(fid).mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        ws.file_text_dir(fid) / "page_1.json",
        {
            "file_id": fid,
            "page": 1,
            "width": 1000,
            "height": 1000,
            "words": [{"t": "TITLE", "l": [100, 100, 150, 120]}],
        },
    )

    def _boom(*args: object, **kwargs: object) -> dict[int, str]:
        raise AssertionError("_request_styles must not be called when disabled")

    monkeypatch.setattr(style_llm, "_request_styles", _boom)
    # No `style` section at all -> the LLM path must not run (_boom guards it).
    # "TITLE" is all-caps, so the deterministic text-transform branch WOULD fire
    # — but an OCR file with no style config must carry no dg:style whatsoever
    # (storage-layout.md / cli-reference.md / SKILL.md all promise OCR is empty
    # unless the workspace opts in). dg:origin is still emitted.
    src = tmp_path / "doc.dgml.xml"
    src.write_text('<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><Heading>TITLE</Heading></dg:chunk>')
    ground_dgml_xml(ws, fid, src, output_path=src, force=True, write_stats=False)
    content = src.read_text(encoding="utf-8")
    assert "dg:style" not in content
    assert "text-transform" not in content
    assert "dg:origin" in content  # grounding itself is unaffected


def test_style_credential_failure_preserves_grounding(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A style-only credential failure must NOT discard the deterministic
    grounding: `dg:origin` still lands and the file is written, even though the
    image-based style pass can't run (its `api_key_env` is unset)."""
    from dgml_core.models import FileRecord
    from dgml_core.storage import Workspace, write_json_atomic
    from dgml_core.xml_grounding import ground_dgml_xml

    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    fid = "ocrwireauth1"
    ws.file_dir(fid).mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        ws.file_json_path(fid),
        FileRecord(
            id=fid,
            original_path="/f.pdf",
            original_filename="f.pdf",
            sha256="0" * 64,
            added_at="2026-01-01T00:00:00Z",
            page_count=1,
            text_mode="ocr",  # the gate
        ).to_json(),
    )
    ws.file_text_dir(fid).mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        ws.file_text_dir(fid) / "page_1.json",
        {
            "file_id": fid,
            "page": 1,
            "width": 1000,
            "height": 1000,
            "words": [
                {"t": "TITLE", "l": [100, 100, 150, 120]},
                {"t": "HERE", "l": [160, 100, 210, 120]},
            ],
        },
    )
    pages_dir = ws.file_pages_dir(fid)
    pages_dir.mkdir(parents=True, exist_ok=True)
    (pages_dir / "page_1.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
    # `api_key_env` points at an env var we make sure is unset -> resolve_api_key
    # raises AuthError inside the style pass at grounding time.
    monkeypatch.delenv("DGML_STYLE_KEY_MISSING", raising=False)
    ws.config_path.write_text(
        json.dumps(
            {
                "style": {
                    "model": "anthropic/claude-haiku-4-5",
                    "api_key_env": "DGML_STYLE_KEY_MISSING",
                }
            }
        ),
        encoding="utf-8",
    )

    src = tmp_path / "doc.dgml.xml"
    src.write_text(
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><Heading>TITLE HERE</Heading></dg:chunk>',
        encoding="utf-8",
    )
    # Must not raise, and must still write the grounded tree with dg:origin.
    res = ground_dgml_xml(ws, fid, src, output_path=src, force=True, write_stats=False)
    content = src.read_text(encoding="utf-8")
    assert "dg:origin" in content
    assert res.stats["elements_annotated"] >= 1
    # Deterministic style survives; the LLM-only pass simply didn't run.
    assert "text-transform: uppercase" in content
