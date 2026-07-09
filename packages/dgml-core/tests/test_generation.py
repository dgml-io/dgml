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

"""Tests for the generation pipeline (typed blocks + batch labeling)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from dgml_core import llm
from dgml_core.generation import blocks as blocks_mod
from dgml_core.generation.blocks import (
    Block,
    Span,
    build_tree,
    parse_block,
    sanitize_concept,
)
from dgml_core.generation.label import (
    apply_labels,
    label_documents,
    propagate_table_consistency,
    render_block_listing,
    wrap_detected_values,
)
from dgml_core.generation.render import render_xml
from dgml_core.generation.transcribe import (
    _append_continuation,
    _parse_window_json,
    loads_tolerant,
)
from lxml import etree  # type: ignore[import-untyped]


def _patch_roster(monkeypatch: pytest.MonkeyPatch, roster: dict[str, str]) -> dict[str, str]:
    """Stub ``llm.call_with_refinement`` (Pass B.1, the default refine path).

    Returns *roster* from both the draft and refined turns. Returns a dict the
    caller can read ``["listing"]`` from after the run.
    """
    captured: dict[str, str] = {}
    payload = json.dumps({"concepts": roster})

    def fake_refine(config: llm.LLMConfig, **kwargs: object) -> tuple[str, str]:
        captured["listing"] = kwargs["user_content"][0]["text"]  # type: ignore[index]
        return (payload, payload)

    monkeypatch.setattr(llm, "call_with_refinement", fake_refine)
    return captured


# ── block model ──────────────────────────────────────────────────────────────


def test_parse_block_coerces_and_drops_empty() -> None:
    assert parse_block({"structure": "bogus", "text": "x"}, "b1") is not None
    parsed = parse_block({"structure": "bogus", "text": "x"}, "b1")
    assert parsed is not None and parsed.structure == "p"
    assert parse_block({"structure": "p", "text": ""}, "b2") is None
    row = parse_block({"structure": "row", "cells": ["a", 2]}, "b3")
    assert row is not None and row.cells == ["a", "2"]


def test_sanitize_concept_pascal_case() -> None:
    assert sanitize_concept("payment-terms") == "PaymentTerms"
    assert sanitize_concept("Payment Terms!") == "PaymentTerms"
    assert sanitize_concept("DefinitionOfTerm") == "DefinitionOfTerm"
    assert sanitize_concept("  --GST--  ") == "GST"
    assert sanitize_concept("###") == ""


def test_org_ns_segment_sanitizes_but_preserves_valid_segments() -> None:
    from dgml_core.generation.semantic_transform import org_ns_segment

    # Spaces (the "Andrew Corp" case that broke extraction) collapse to hyphens.
    assert org_ns_segment("Andrew Corp") == "Andrew-Corp"
    assert org_ns_segment("  Andrew   Corp  ") == "Andrew-Corp"
    # URI-illegal characters are dropped.
    assert org_ns_segment("A&B/Co.") == "ABCo."
    # Already-valid segments are unchanged — notably the workspace-dir-name
    # fallback used by pre-workspace.json workspaces, so their namespaces hold.
    assert org_ns_segment("dgml-workspace") == "dgml-workspace"
    assert org_ns_segment("Acme") == "Acme"
    # Degenerate input still yields a legal segment.
    assert org_ns_segment("   ") == "org"


def test_build_header_embeds_sanitized_org_in_namespace() -> None:
    from dgml_core.generation.to_semantic import build_header

    header = build_header("Andrew Corp", "NAV REIT Property")
    assert 'xmlns:docset="http://dgml.io/Andrew-Corp/NavReitProperty"' in header
    assert "Andrew Corp" not in header  # no raw space leaks into the URI


def test_sanitize_concept_strips_structural_suffixes() -> None:
    assert sanitize_concept("PaymentTermsClause") == "PaymentTerms"
    assert sanitize_concept("DeliveryRulesSection") == "DeliveryRules"
    assert sanitize_concept("DefinitionItem") == "Definition"
    assert sanitize_concept("SummaryParagraph2") == ""  # Summary is structural too
    # Purely structural concepts normalize away entirely → unlabeled.
    assert sanitize_concept("Item") == ""
    assert sanitize_concept("Heading") == ""
    assert sanitize_concept("section-title") == ""


def test_wrap_detected_values_isolates_currency_and_percent() -> None:
    b = Block(id="b1", structure="p", text="from $13,995 per person, save 90% today")
    wrap_detected_values([b])
    vals = [b.text[s.start : s.end] for s in b.entities]
    assert "$13,995" in vals
    assert "90%" in vals
    assert all(s.concept == "" for s in b.entities)


def test_wrap_detected_values_skips_model_span_overlap() -> None:
    b = Block(id="b1", structure="p", text="was $16,995 now $13,995")
    b.entities = [Span(start=4, end=11, concept="OriginalPrice")]  # "$16,995"
    wrap_detected_values([b])
    spans = sorted(b.entities, key=lambda s: s.start)
    assert [b.text[s.start : s.end] for s in spans] == ["$16,995", "$13,995"]
    assert spans[0].concept == "OriginalPrice"
    assert spans[1].concept == ""


def test_loads_tolerant_repairs_unescaped_quotes() -> None:
    # The real failure: verbatim text with inner quotes breaks the JSON string.
    raw = (
        '{"continues": "", "blocks": [{"structure": "p", '
        '"text": "maintains or "clamps" glucose to a constant target."}]}'
    )
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw)
    out = loads_tolerant(raw)
    assert out["blocks"][0]["text"] == 'maintains or "clamps" glucose to a constant target.'


def test_loads_tolerant_leaves_valid_json_untouched() -> None:
    raw = '{"continues": "", "blocks": [{"structure": "p", "text": "plain, with: punctuation"}]}'
    assert loads_tolerant(raw) == json.loads(raw)


def test_sanitize_concept_strips_leaked_concept_prefix() -> None:
    assert sanitize_concept("ConceptClientName") == "ClientName"
    assert sanitize_concept("ConceptCompanySignatory") == "CompanySignatory"
    assert sanitize_concept("Conception") == "Conception"  # real word, no strip
    assert sanitize_concept("Concept") == "Concept"  # nothing follows, no strip


def _b(structure: str, bid: str, **kw: object) -> Block:
    return Block(id=bid, structure=structure, **kw)  # type: ignore[arg-type]


def test_build_tree_headings_nest_by_level() -> None:
    tree = build_tree(
        [
            _b("heading", "b1", text="PART ONE", level=1),
            _b("p", "b2", text="intro"),
            _b("heading", "b3", text="Sub", level=2),
            _b("p", "b4", text="body"),
            _b("heading", "b5", text="PART TWO", level=1),
        ]
    )
    assert [c.kind for c in tree.children] == ["section", "section"]
    part_one = tree.children[0]
    assert [c.kind for c in part_one.children] == ["h", "p", "section"]
    assert part_one.children[2].children[0].block.text == "Sub"  # type: ignore[union-attr]


def test_build_tree_groups_runs() -> None:
    tree = build_tree(
        [
            _b("item", "b1", text="first", lim="(a)"),
            _b("item", "b2", text="second", lim="(b)"),
            _b("p", "b3", text="break"),
            _b("row", "b4", cells=["x", "1"]),
            _b("row", "b5", cells=["y", "2"]),
            _b("field", "b6", lim="ITEM 3", label="Date", value="1 July 2025"),
        ]
    )
    kinds = [c.kind for c in tree.children]
    assert kinds == ["list", "p", "table", "form"]
    assert len(tree.children[0].children) == 2
    assert len(tree.children[2].children) == 2


# ── transcription helpers ────────────────────────────────────────────────────


def test_parse_window_json_tolerates_fences_and_noise() -> None:
    raw = 'Sure! ```json\n{"continues": "", "blocks": []}\n``` done'
    assert _parse_window_json(raw) == {"continues": "", "blocks": []}


def test_append_continuation_targets_last_text_block() -> None:
    seq = [
        _b("p", "b1", text="The fee is payable"),
        _b("row", "b2", cells=["Oven", "2"]),
    ]
    _append_continuation(seq, "within 30 days.")
    assert seq[1].cells == ["Oven", "2 within 30 days."]
    seq2 = [_b("p", "b1", text="The fee is payable")]
    _append_continuation(seq2, "within 30 days.")
    assert seq2[0].text == "The fee is payable within 30 days."


# ── labeling ─────────────────────────────────────────────────────────────────


def _docs() -> dict[str, list[Block]]:
    return {
        "a.pdf": [
            _b("heading", "b0001", text="Payment Terms", level=2, lim="4.2"),
            _b("p", "b0002", text="Invoices are payable within 30 days of receipt."),
        ],
        "b.pdf": [
            _b("heading", "b0001", text="Payment Terms", level=2, lim="3.1"),
            _b("p", "b0002", text="Invoices are payable within 45 days of receipt."),
        ],
    }


def test_apply_labels_validates_and_applies() -> None:
    blocks = _docs()["a.pdf"]
    warnings = apply_labels(
        blocks,
        {
            "b0001": {"concept": "Payment Terms"},
            "b0002": {
                "concept": "payment-terms",
                "entities": [
                    {"quote": "30 days", "concept": "payment-due-period"},
                    {"quote": "days of rec", "concept": "overlap-dropped"},
                    {"quote": "not in the text", "concept": "missing-quote"},
                ],
            },
            "b9999": {"concept": "ghost"},
        },
        doc_name="a.pdf",
    )
    assert blocks[0].concept == "PaymentTerms"
    # The quote is located by the pipeline; the model never counts positions.
    assert blocks[1].entities == [Span(start=28, end=35, concept="PaymentDuePeriod")]
    assert blocks[1].text[28:35] == "30 days"
    assert len(warnings) == 3  # overlapping quote, missing quote, unknown block


def test_apply_labels_quote_occurrence_picks_the_right_match() -> None:
    block = _b("p", "b1", text="pay 5% now and 5% later")
    warnings = apply_labels(
        [block],
        {"b1": {"entities": [{"quote": "5%", "occurrence": 2, "concept": "later-rate"}]}},
    )
    assert warnings == []
    (span,) = block.entities
    assert (span.start, span.end) == (15, 17)
    assert block.text[span.start : span.end] == "5%"


def test_apply_labels_offsets_without_quote_are_rejected() -> None:
    block = _b("p", "b1", text="payable within 30 days")
    warnings = apply_labels(
        [block],
        {"b1": {"entities": [{"start": 15, "end": 22, "concept": "due-period"}]}},
    )
    assert block.entities == []
    assert any("not found verbatim" in w for w in warnings)


def test_apply_labels_value_heading_signaled_by_whole_text_entity() -> None:
    block = _b("heading", "b1", text="Acme Pty Ltd", level=1)
    apply_labels(
        [block],
        {
            "b1": {
                "concept": "SellerName",
                "entities": [{"quote": "Acme Pty Ltd", "concept": "SellerName"}],
            }
        },
    )
    assert block.concept == ""  # section stays generic
    assert block.value_concept == "SellerName"  # value-kind tags the heading
    assert block.entities == []  # not also wrapped as an inline span


def test_render_dgml_value_heading_names_header_not_section() -> None:
    from dgml_core.generation.to_semantic import render_dgml

    seq = [_b("heading", "b1", text="Acme Pty Ltd", level=1, value_concept="SupplierName")]
    out = render_dgml(seq, header="<dg:chunk>")
    assert '<dg:chunk dg:structure="section">' in out
    assert '<docset:SupplierName dg:structure="header">Acme Pty Ltd</docset:SupplierName>' in out


def test_label_documents_plans_then_labels_with_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = _docs()
    calls: list[tuple[str, str]] = []
    plan = _patch_roster(monkeypatch, {"PaymentTerms": "payment obligations section"})

    def fake_call(config: llm.LLMConfig, **kwargs: object) -> str:
        system = str(kwargs["system_prompt"])
        text = kwargs["user_content"][0]["text"]  # type: ignore[index]
        calls.append((system, text))
        # Label both the heading and the paragraph so no section retry fires.
        return json.dumps(
            {
                "labels": {
                    "b0001": {"concept": "PaymentSection"},
                    "b0002": {"concept": "PaymentTerms"},
                }
            }
        )

    monkeypatch.setattr(llm, "call", fake_call)
    warnings = label_documents(docs, config=llm.LLMConfig(model="anthropic/claude-haiku-4-5"))
    assert warnings == []
    # Pass B.1 plans the roster over EVERY document's skeleton side by side (one
    # call_with_refinement); then one labeling call per document carries it.
    assert len(calls) == 2
    assert "== a.pdf ==" in plan["listing"] and "== b.pdf ==" in plan["listing"]
    for _system, text in calls:
        assert "CONCEPTS ALREADY IN USE" in text
        assert "- PaymentTerms" in text
    assert docs["a.pdf"][1].concept == docs["b.pdf"][1].concept == "PaymentTerms"


def test_label_documents_roster_seed_skips_planning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A roster_seed (from --schema-path) is used as-is; Pass B.1 is skipped."""
    docs = _docs()
    label_inputs: list[str] = []

    def no_plan(config: llm.LLMConfig, **kwargs: object) -> tuple[str, str]:
        raise AssertionError("planning must be skipped when roster_seed is given")

    def fake_call(config: llm.LLMConfig, **kwargs: object) -> str:
        label_inputs.append(kwargs["user_content"][0]["text"])  # type: ignore[index]
        return json.dumps({"labels": {"b0001": {"concept": "PaymentTerms"}}})

    monkeypatch.setattr(llm, "call_with_refinement", no_plan)  # would raise if planning ran
    monkeypatch.setattr(llm, "call", fake_call)
    label_documents(
        docs,
        config=llm.LLMConfig(model="anthropic/claude-haiku-4-5"),
        roster_seed={"PaymentTerms": "the payment clause"},
    )
    # The seeded concept is carried into the per-document labeling calls.
    assert all("- PaymentTerms" in text for text in label_inputs)
    assert docs["a.pdf"][0].concept == "PaymentTerms"


def test_label_documents_failure_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    docs = _docs()

    def boom(config: llm.LLMConfig, **kwargs: object) -> str:
        raise RuntimeError("provider down")

    def boom_refine(config: llm.LLMConfig, **kwargs: object) -> tuple[str, str]:
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm, "call", boom)
    monkeypatch.setattr(llm, "call_with_refinement", boom_refine)
    warnings = label_documents(docs, config=llm.LLMConfig(model="anthropic/claude-haiku-4-5"))
    # Planning fails soft (empty roster); each doc chunk fails soft too.
    assert sum("labeling failed" in w for w in warnings) == 2
    assert all(not b.concept for blocks in docs.values() for b in blocks)


@pytest.mark.parametrize("max_parallel_docs", [1, 2])
def test_convert_batch_skips_failed_document(
    monkeypatch: pytest.MonkeyPatch, max_parallel_docs: int
) -> None:
    """One document failing transcription must not sink the batch — the rest are
    still transcribed, labeled, and rendered (serial and parallel paths)."""
    from dgml_core.generation import pipeline as pl

    monkeypatch.setattr(
        "dgml_core.generation.document.load_document_as_pdf",
        lambda path, *, converters: b"%PDF-",
    )

    def fake_transcribe(pdf_bytes: bytes, *, doc_name: str, **kw: object) -> list[Block]:
        if doc_name == "bad.pdf":
            raise RuntimeError("provider down")
        return [Block(id="b1", structure="p", text="hello", concept="Greeting")]

    monkeypatch.setattr(pl, "transcribe_document", fake_transcribe)
    monkeypatch.setattr(pl, "label_documents", lambda docs, **kw: [])  # no LLM in this test

    out = pl.convert_batch(
        ["good.pdf", "bad.pdf"],
        options=pl.ConvertOptions(
            model="anthropic/claude-haiku-4-5",
            dgml_header="<dg:chunk>",
            max_parallel_docs=max_parallel_docs,
        ),
    )
    assert set(out) == {"good.pdf"}  # bad.pdf dropped; the batch did not abort
    assert "hello" in out["good.pdf"]


def test_convert_batch_streams_to_sink_without_accumulating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With on_output, each rendered doc is handed to the sink and the returned
    dict stays empty (no accumulation of the whole batch's DGML)."""
    from dgml_core.generation import pipeline as pl

    monkeypatch.setattr(
        "dgml_core.generation.document.load_document_as_pdf",
        lambda path, *, converters: b"%PDF-",
    )
    monkeypatch.setattr(
        pl,
        "transcribe_document",
        lambda pdf_bytes, *, doc_name, **kw: [Block(id="b1", structure="p", text="hi")],
    )
    monkeypatch.setattr(pl, "label_documents", lambda docs, **kw: [])

    seen: list[str] = []
    out = pl.convert_batch(
        ["a.pdf", "b.pdf"],
        options=pl.ConvertOptions(model="anthropic/claude-haiku-4-5", dgml_header="<dg:chunk>"),
        on_output=lambda name, xml: seen.append(name),
    )
    assert out == {}  # nothing accumulated when a sink is provided
    assert sorted(seen) == ["a.pdf", "b.pdf"]  # each doc streamed exactly once


def test_load_labeled_docs_from_cache_roundtrip(tmp_path: Path) -> None:
    from dgml_core.generation.pipeline import load_labeled_docs_from_cache

    (tmp_path / "doc_blocks.json").write_text(
        json.dumps([{"id": "b1", "structure": "p", "text": "Acme owes $5"}]),
        encoding="utf-8",
    )
    (tmp_path / "label_doc_c01_raw.json").write_text(
        json.dumps({"labels": {"b1": {"concept": "PaymentObligation"}}}),
        encoding="utf-8",
    )
    docs = load_labeled_docs_from_cache(tmp_path, ["doc", "missing"])
    assert set(docs) == {"doc"}  # 'missing' has no _blocks.json → skipped
    assert docs["doc"][0].concept == "PaymentObligation"


def test_schema_load_rejects_unknown_keys(tmp_path: Path) -> None:
    """A stale or typo'd field in schema.json is a hard failure, never a silent
    drop — a caller must not think a field was set
    when it wasn't. The CLI maps this to INVALID_ARGUMENT for --schema-path."""
    from dgml_core.generation.schema import Schema

    payload = {
        "tags": {"Foo": {"name": "Foo", "role": "a foo", "kind": "inline", "exmaple": "typo"}},
        "notes": "",
    }
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(TypeError):
        Schema.load(path)


def test_cache_write_gates_on_debug_flag(tmp_path: Path) -> None:
    """cache_write(debug=False) is a no-op; the default (debug=True) and
    explicit debug=True always write when a cache dir is set."""
    from dgml_core.generation.transcribe import cache_write

    cache_write(tmp_path, "skip.json", "x", debug=False)
    assert not (tmp_path / "skip.json").exists()

    cache_write(tmp_path, "keep.json", "x", debug=True)
    assert (tmp_path / "keep.json").exists()

    cache_write(tmp_path, "functional.json", "x")
    assert (tmp_path / "functional.json").exists()


def test_convert_batch_concepts_always_docset_namespaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every concept is docset:-namespaced regardless of how many docs use it,
    and adding a doc never flips a prior doc's namespaces — so nothing semantic
    lands in dg: and unchanged priors aren't re-rendered."""
    from dgml_core.generation import pipeline as pl
    from dgml_core.generation.to_semantic import render_dgml

    monkeypatch.setattr(
        "dgml_core.generation.document.load_document_as_pdf",
        lambda path, *, converters: b"%PDF-",
    )
    monkeypatch.setattr(
        pl,
        "transcribe_document",
        lambda pdf_bytes, *, doc_name, **kw: [
            Block(id="b1", structure="p", text="x", concept="Shared")
        ],
    )
    monkeypatch.setattr(pl, "label_documents", lambda docs, **kw: [])

    prior = [
        Block(id="p1", structure="p", text="y", concept="Shared"),
        Block(id="p2", structure="p", text="z", concept="OnlyOld"),
    ]
    # A concept seen in a single document is still docset:, never dg:.
    prior_outputs = {"old.pdf": render_dgml(prior, header="<dg:chunk>")}
    assert "docset:Shared" in prior_outputs["old.pdf"]
    assert "docset:OnlyOld" in prior_outputs["old.pdf"]
    assert "dg:OnlyOld" not in prior_outputs["old.pdf"]  # nothing semantic in dg:

    seen: dict[str, str] = {}
    pl.convert_batch(
        ["new.pdf"],
        options=pl.ConvertOptions(model="anthropic/claude-haiku-4-5", dgml_header="<dg:chunk>"),
        on_output=lambda name, xml: seen.__setitem__(name, xml),
        prior_docs={"old.pdf": prior},
        prior_outputs=prior_outputs,
    )
    # New doc's concept is docset:; the prior doc is NOT re-rendered because
    # a second occurrence of "Shared" no longer flips any prefix.
    assert "docset:Shared" in seen["new.pdf"]
    assert "old.pdf" not in seen


def test_plan_concept_roster_caps_to_largest_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over the cap, only the largest-skeleton docs feed the planning call."""
    from dgml_core.generation.label import _PLAN_MAX_DOCS, plan_concept_roster

    # _PLAN_MAX_DOCS + 2 docs; doc00/doc01 are large, the rest are tiny.
    docs: dict[str, list[Block]] = {
        f"doc{i:02d}.pdf": [
            _b("heading", f"{i}-{j}", text=f"H{j}", level=1) for j in range(50 if i < 2 else 2)
        ]
        for i in range(_PLAN_MAX_DOCS + 2)
    }

    captured = _patch_roster(monkeypatch, {"Foo": "a foo role"})
    roster = plan_concept_roster(docs, config=llm.LLMConfig(model="anthropic/claude-haiku-4-5"))

    listing = captured["listing"]
    assert listing.count("== doc") == _PLAN_MAX_DOCS  # only the cap many docs planned
    assert "== doc00.pdf ==" in listing and "== doc01.pdf ==" in listing  # largest included
    assert "== doc21.pdf ==" not in listing  # a tiny doc dropped from planning
    assert roster == {"Foo": "a foo role"}


def test_render_block_listing_one_line_per_block() -> None:
    listing = render_block_listing("a.pdf", _docs()["a.pdf"])
    assert listing.startswith("== a.pdf ==")
    assert "b0001 heading [4.2]: 4.2 Payment Terms" in listing


# ── rendering ────────────────────────────────────────────────────────────────


def test_render_xml_structure_and_concepts() -> None:
    seq = [
        _b(
            "heading",
            "b1",
            text="Payment Terms",
            level=1,
            lim="4.2",
            concept="payment-terms",
        ),
        _b(
            "p",
            "b2",
            text="Invoices are payable within 30 days of receipt.",
            entities=[Span(start=28, end=35, concept="payment-due-period")],
        ),
        _b("item", "b3", text="use the loading dock", lim="12.1"),
        _b("row", "b4", cells=["Oven", "2", "$180"]),
        _b(
            "field",
            "b5",
            lim="ITEM 3",
            label="Commencement Date",
            value="1 July 2025",
            concept="commencement-date",
        ),
    ]
    xml = render_xml(seq, doc_name="a.pdf")
    root = etree.fromstring(xml.encode())
    sec = root.find("sec")
    assert sec is not None and sec.get("concept") == "payment-terms"  # lifted from heading
    assert sec.findtext("h/lim") == "4.2"
    v = sec.find(".//v")
    assert v is not None and v.text == "30 days" and v.get("concept") == "payment-due-period"
    li = sec.find(".//li")
    assert li is not None and li.findtext("lim") == "12.1"
    assert "".join(li.itertext()) == "12.1use the loading dock"
    assert [td.text for td in sec.findall(".//tr/td")] == ["Oven", "2", "$180"]
    fld = sec.find(".//fld")
    assert fld is not None and fld.get("concept") is None
    assert fld.findtext("label") == "Commencement Date"
    value = fld.find("value")
    # The concept names the VALUE's role, so it marks the value element only.
    assert value is not None and value.get("concept") == "commencement-date"
    assert value.text == "1 July 2025"


def test_render_xml_text_is_verbatim() -> None:
    text = "Late payments accrue interest at 2% per month."
    seq = [
        _b(
            "p",
            "b1",
            text=text,
            entities=[Span(start=33, end=45, concept="interest-rate")],
        ),
    ]
    root = etree.fromstring(render_xml(seq).encode())
    p = root.find("p")
    assert p is not None
    assert "".join(p.itertext()) == text  # tags inserted around spans only


def test_render_xml_unlabeled_blocks_have_no_concept() -> None:
    root = etree.fromstring(render_xml([_b("p", "b1", text="boilerplate")]).encode())
    p = root.find("p")
    assert p is not None and p.get("concept") is None  # Option I: no label is legal


# ── flat structures constant stays in sync ───────────────────────────────────


def test_flat_structures_match_parser() -> None:
    content: dict[str, dict[str, object]] = {
        "row": {"cells": ["x"]},
        "field": {"label": "l", "value": "v"},
    }
    for s in blocks_mod.FLAT_STRUCTURES:
        kw = content.get(s, {"text": "t"})
        parsed = parse_block({"structure": s, **kw}, "b1")
        assert parsed is not None and parsed.structure == s


# ── debug cache artifacts ────────────────────────────────────────────────────


def test_label_documents_writes_cache_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docs = _docs()
    _patch_roster(monkeypatch, {"PaymentTerms": "payment section"})

    def fake_call(config: llm.LLMConfig, **kw: object) -> str:
        return json.dumps({"labels": {"b0001": {"concept": "PaymentTerms"}}})

    monkeypatch.setattr(llm, "call", fake_call)
    cfg = llm.LLMConfig(model="anthropic/claude-haiku-4-5")

    # Default (debug off): only the functional files the next run reloads are
    # written — the concept roster and the per-chunk RAW label outputs.
    cache = tmp_path / "cache"
    label_documents(docs, config=cfg, cache_dir=cache)
    names = sorted(p.name for p in cache.iterdir())
    assert "concept_roster.json" in names
    assert "label_a_c01_raw.json" in names and "label_b_c01_raw.json" in names
    # Debug-only artifacts are skipped.
    assert not any(n.endswith("_input.txt") for n in names)
    assert "plan_roster_raw.json" not in names and "plan_roster_draft_raw.json" not in names
    roster = json.loads((cache / "concept_roster.json").read_text())
    assert "PaymentTerms" in roster

    # debug=True: the input listings and roster-planning dumps are captured too.
    dbg = tmp_path / "cache_debug"
    label_documents(docs, config=cfg, cache_dir=dbg, debug=True)
    dbg_names = sorted(p.name for p in dbg.iterdir())
    assert "plan_roster_input.txt" in dbg_names and "plan_roster_raw.json" in dbg_names
    assert "plan_roster_draft_raw.json" in dbg_names  # the refine draft turn is captured
    assert "label_a_c01_input.txt" in dbg_names and "label_b_c01_input.txt" in dbg_names


def test_render_xml_lim_precedes_text_in_reading_order() -> None:
    """The printed enumerator serializes BEFORE the text, as on the page —
    text after a <lim> child must live in its tail, never in el.text."""
    seq = [
        _b("heading", "b1", text="Definitions", lim="1.1", level=2),
        _b(
            "p",
            "b2",
            text="payable within 30 days of receipt",
            lim="(a)",
            entities=[Span(start=15, end=22, concept="due-period")],
        ),
    ]
    xml = render_xml(seq)
    root = etree.fromstring(xml.encode())
    h = root.find(".//h")
    assert h is not None and (h.text or "").strip() == ""  # nothing before <lim>
    assert "".join(h.itertext()) == "1.1Definitions"
    p = root.find(".//p")
    assert p is not None
    # Reading order: lim, then text, with the entity span in place.
    assert "".join(p.itertext()) == "(a)payable within 30 days of receipt"
    assert p.findtext("v") == "30 days"
    assert re.search(r"<lim>\(a\)</lim>payable", xml)


# ── to_semantic: blocks → structure-attribute semantic XML ───────────────────


def test_render_semantic_xml_concept_tags_only_where_labeled() -> None:
    from dgml_core.generation.to_semantic import render_semantic_xml

    seq = [
        _b(
            "heading",
            "b1",
            text="Payment Terms",
            level=1,
            lim="4.2",
            concept="PaymentTerms",
        ),
        _b(
            "p",
            "b2",
            text="payable within 30 days of receipt",
            entities=[Span(start=15, end=22, concept="DuePeriod")],
        ),
        _b("p", "b3", text="connective prose, unlabeled"),
        _b("item", "b4", text="use the loading dock", lim="(a)", concept="DeliveryRule"),
        _b(
            "field",
            "b5",
            lim="ITEM 3",
            label="Date",
            value="1 July 2025",
            concept="CommencementDate",
        ),
    ]
    xml = render_semantic_xml(seq)
    root = etree.fromstring(xml.encode())
    assert root.tag == "xml"
    sec = root.find("PaymentTerms")  # concept tag on the section
    assert sec is not None and sec.get("structure") == "section"
    h = sec.find("header")
    assert h is not None and h.get("structure") == "header"
    assert "".join(h.itertext()) == "4.2Payment Terms"  # lim precedes text
    # Unlabeled paragraph keeps the plain structural name.
    assert sec.find("p") is not None and sec.find("p").get("structure") == "p"
    # Inline entity is a PascalCase element with no structure attribute.
    due = sec.find(".//DuePeriod")
    assert due is not None and due.text == "30 days" and due.get("structure") is None
    # Labeled list item gets its concept tag; the list container is plain ol.
    ol = sec.find("ol")
    assert ol is not None and ol.find("DeliveryRule") is not None
    # Form field: value wrapped in the concept inline element.
    li = sec.find(".//ul/li")
    assert li is not None
    assert li.findtext("lim") == "ITEM 3"
    assert li.find("p/CommencementDate").text == "1 July 2025"


def test_render_semantic_xml_value_heading_keeps_concept_on_header() -> None:
    """A value-heading carries its concept on value_concept (apply_labels moves
    it there and clears block.concept). The semantic view must tag the header
    element with it — keeping the label around the value — rather than dropping
    it to a plain <header> and hoisting nothing to the section."""
    from dgml_core.generation.to_semantic import render_semantic_xml

    seq = [
        _b("heading", "b1", text="LECH WAŁĘSA", level=1, value_concept="ExpertName"),
        _b("p", "b2", text="Guest Speaker", concept="ExpertRole"),
    ]
    xml = render_semantic_xml(seq)
    root = etree.fromstring(xml.encode())
    expert = root.find(".//ExpertName")
    assert expert is not None
    assert expert.get("structure") == "header"  # label stays on the header (the value)
    assert "".join(expert.itertext()) == "LECH WAŁĘSA"
    # concept lived on value_concept, not block.concept → section stays generic
    assert root.find(".//ExpertName[@structure='section']") is None
    assert root.find("section[@structure='section']") is not None


def test_render_semantic_xml_feeds_pass4(tmp_path: Path) -> None:
    """The transformation output round-trips through the real Pass-4 encoder."""
    import json as _json

    from dgml_core.generation.semantic_transform import transform_docset
    from dgml_core.generation.to_semantic import render_semantic_xml

    seq = [
        _b(
            "heading",
            "b1",
            text="Payment Terms",
            level=1,
            lim="4.2",
            concept="PaymentTerms",
        ),
        _b("p", "b2", text="payable within 30 days"),
    ]
    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()
    (semantic_dir / "doc.xml").write_text(render_semantic_xml(seq), encoding="utf-8")
    docset_json = tmp_path / "docset.json"
    docset_json.write_text(_json.dumps({"id": "ds1", "name": "Test Set"}), encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    n = transform_docset(semantic_dir, docset_json, "test-ws", out_dir)
    assert n == 1
    dgml = (out_dir / "doc.dgml.xml").read_text(encoding="utf-8")
    assert "<dg:chunk" in dgml and 'xmlns:docset="' in dgml
    assert ":PaymentTerms" in dgml  # concept survives into the namespaced format


def test_apply_labels_container_named_after_value_keeps_all_leaves() -> None:
    # The block carries a value concept (Supplier) it ALSO emits inline, PLUS a
    # different value (SupplierAddress) — a multi-value container the labeler
    # named after one of its values. It is demoted to dg:chunk and every value
    # is kept as an inline leaf (a leaf concept must not wrap other leaves).
    block = _b("p", "b1", text="Acme Pty Ltd of 1 Example Street, Springfield")
    warnings = apply_labels(
        [block],
        {
            "b1": {
                "concept": "Supplier",
                "entities": [
                    {"quote": "Acme Pty Ltd", "concept": "Supplier"},
                    {
                        "quote": "1 Example Street, Springfield",
                        "concept": "SupplierAddress",
                    },
                ],
            }
        },
    )
    assert block.concept == ""  # demoted to a generic container
    assert [s.concept for s in block.entities] == ["Supplier", "SupplierAddress"]
    assert not any("equals the block concept" in w for w in warnings)


def test_skeleton_listing_includes_first_paragraph_after_heading() -> None:
    from dgml_core.generation.label import render_skeleton_listing

    docs = {
        "a.pdf": [
            _b("heading", "b1", text="Supplier", level=2),
            _b("p", "b2", text="Acme Pty Ltd of 1 Example Street"),
            _b("p", "b3", text="second paragraph stays out"),
        ]
    }
    listing = render_skeleton_listing(docs)
    assert "Acme Pty Ltd of 1 Example Street" in listing  # planner sees a value line
    assert "second paragraph stays out" not in listing


# ── final dgml conversion (render_dgml) ──────────────────────────────────────


def test_render_dgml_namespacing_no_hx_and_typing() -> None:
    from dgml_core.generation.to_semantic import render_dgml

    seq = [
        _b("heading", "b1", text="Charges", level=1, lim="4.", concept="ChargesTerms"),
        _b(
            "p",
            "b2",
            text="Payable by 1 July 2025.",
            entities=[Span(start=11, end=22, concept="DueDate")],
        ),
        _b("p", "b3", text="connective prose, unlabeled"),
        _b("item", "b4", text="first rule", lim="(a)"),
    ]
    out = render_dgml(seq, header="<dg:chunk>")
    # well-formed once the placeholder header declares the prefixes
    etree.fromstring(
        out.encode().replace(
            b"<dg:chunk>", b'<dg:chunk xmlns:dg="d" xmlns:docset="s" xmlns:xsi="x">'
        )
    )
    # no depth-based hx anywhere
    assert 'dg:structure="h1"' not in out and 'dg:structure="h2"' not in out
    # concept → docset:, real structural type in `dg:structure`
    assert '<docset:ChargesTerms dg:structure="section">' in out
    # unlabeled scaffolding → dg:chunk with its real type
    assert '<dg:chunk dg:structure="header">' in out
    assert '<dg:chunk dg:structure="p">connective prose' in out
    assert '<dg:chunk dg:structure="ol">' in out and '<dg:chunk dg:structure="li">' in out
    assert '<dg:chunk dg:structure="lim">(a)</dg:chunk>' in out
    # every concept → docset:, even a single-doc one, with value typing on the date
    assert (
        '<docset:DueDate xsi:type="date" dg:value="2025-07-01">1 July 2025</docset:DueDate>' in out
    )
    # envelope
    assert out.startswith("<?xml version='1.0' encoding='utf-8'?>\n<dg:chunk")
    assert out.rstrip().endswith("</dg:chunk>")


def test_label_documents_section_retry_forces_untagged_headings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unlabeled heading after the first pass triggers a targeted retry."""
    docs = {
        "a.pdf": [
            _b("heading", "b0001", text="Charges", level=1),
            _b("p", "b0002", text="connective prose"),
        ]
    }
    calls: list[str] = []
    _patch_roster(monkeypatch, {})  # empty roster; this test is about the retry

    def fake_call(config: llm.LLMConfig, **kwargs: object) -> str:
        text = kwargs["user_content"][0]["text"]  # type: ignore[index]
        calls.append(text)
        if "left unlabeled" in text:  # the retry call
            return json.dumps({"labels": {"b0001": {"concept": "ChargesTerms"}}})
        return json.dumps({"labels": {}})  # first pass labels nothing

    monkeypatch.setattr(llm, "call", fake_call)
    label_documents(docs, config=llm.LLMConfig(model="anthropic/claude-haiku-4-5"))
    # plan + first-pass + section retry
    assert sum("left unlabeled" in c for c in calls) == 1
    assert docs["a.pdf"][0].concept == "ChargesTerms"
    assert not docs["a.pdf"][1].concept  # paragraph not forced


def test_needs_label_only_unlabeled_headings() -> None:
    from dgml_core.generation.label import _needs_label

    assert _needs_label(_b("heading", "b1", text="X"))
    assert not _needs_label(_b("heading", "b2", text="X", concept="Already"))
    assert not _needs_label(_b("p", "b3", text="prose"))


def test_build_header_dgml_io_scheme() -> None:
    from dgml_core.generation.to_semantic import build_header

    h = build_header("my-workspace", "Org / Lease Set")
    assert h.startswith("<dg:chunk")
    assert 'xmlns:dg="http://dgml.io/ns/dg#"' in h
    assert 'xmlns:docset="http://dgml.io/my-workspace/' in h
    assert 'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"' in h
    assert 'xmlns:xhtml="http://www.w3.org/1999/xhtml"' in h
    assert h.rstrip().endswith(">")
    # cp:version / addedChunks namespaces are not emitted
    assert "cp:version" not in h and "addedChunks" not in h


# ── table & list consistency ─────────────────────────────────────────────────


def test_propagate_table_record_uniform_columns() -> None:

    # Record table: column 0 = name, column 1 = price across rows; row 2 left
    # the columns unlabeled — propagation fills them.
    rows = [
        Block(
            id="r1",
            structure="row",
            cells=["Oven", "$180"],
            concept="Entry",
            group_concept="EquipmentSchedule",
            cell_concepts=["ItemName", "ItemPrice"],
        ),
        Block(
            id="r2",
            structure="row",
            cells=["Fridge", "$95"],
            concept="Entry",
            group_concept="",
            cell_concepts=["ItemName", "ItemPrice"],
        ),
        Block(id="r3", structure="row", cells=["Mixer", "$40"]),
    ]
    propagate_table_consistency(rows)
    assert all(r.group_concept == "EquipmentSchedule" for r in rows)  # table name shared
    assert all(r.concept == "Entry" for r in rows)  # row concept shared
    assert rows[2].cell_concepts == ["ItemName", "ItemPrice"]  # columns propagated


def test_propagate_table_keyvalue_columns_left_per_row() -> None:

    # Key-value table: column 1 holds a different value-kind each row, so the
    # varying column is NOT propagated.
    rows = [
        Block(
            id="r1",
            structure="row",
            cells=["Date", "1 July 2025"],
            cell_concepts=["FieldLabel", "CommencementDate"],
        ),
        Block(
            id="r2",
            structure="row",
            cells=["Term", "5 years"],
            cell_concepts=["FieldLabel", "LeaseTerm"],
        ),
    ]
    propagate_table_consistency(rows)
    # Column 0 repeats (FieldLabel) → uniform; column 1 varies → kept per-row.
    assert rows[0].cell_concepts == ["FieldLabel", "CommencementDate"]
    assert rows[1].cell_concepts == ["FieldLabel", "LeaseTerm"]


def test_propagate_list_short_items_share_concept() -> None:
    from dgml_core.generation.label import propagate_list_consistency

    items = [
        Block(id="i1", structure="item", text="apples", concept="Produce"),
        Block(id="i2", structure="item", text="pears", concept="Produce"),
        Block(id="i3", structure="item", text="plums"),  # unlabeled short item
    ]
    propagate_list_consistency(items)
    assert all(b.concept == "Produce" for b in items)


def test_propagate_list_prose_items_keep_individual_concepts() -> None:
    from dgml_core.generation.label import propagate_list_consistency

    long = "The Lessee must do a great many things " * 3
    items = [
        Block(id="i1", structure="item", text=long, concept="MaintenanceDuty"),
        Block(id="i2", structure="item", text=long, concept="InsuranceDuty"),
        Block(id="i3", structure="item", text=long),
    ]
    propagate_list_consistency(items)
    assert items[0].concept == "MaintenanceDuty"
    assert items[1].concept == "InsuranceDuty"
    assert items[2].concept == ""  # prose item not force-shared


def test_render_dgml_record_table_names_table_and_columns() -> None:
    from dgml_core.generation.to_semantic import render_dgml

    rows = [
        Block(
            id="r1",
            structure="row",
            cells=["Oven", "$180"],
            group_concept="EquipmentSchedule",
            concept="Equipment",
            cell_concepts=["ItemName", "ItemPrice"],
        ),
        Block(
            id="r2",
            structure="row",
            cells=["Fridge", "$95"],
            group_concept="EquipmentSchedule",
            concept="Equipment",
            cell_concepts=["ItemName", "ItemPrice"],
        ),
    ]
    out = render_dgml(rows, header="<dg:chunk>")
    assert '<docset:EquipmentSchedule dg:structure="table">' in out
    assert '<docset:Equipment dg:structure="tr">' in out
    # concept-tagged cells still carry the td layout role (HTML-render contract)
    assert '<docset:ItemName dg:structure="td">Oven</docset:ItemName>' in out
    assert '<docset:ItemPrice dg:structure="td" xsi:type="decimal"' in out  # price typed


def test_multi_value_cell_splits_on_count_mismatch() -> None:
    """A merged cell labeled with more concepts than physical cells splits inline.

    Synthetic data only. The transcriber emitted 5 cells but the labeler returned
    6 column concepts (it split one cell into an id + a name). The old
    all-or-nothing guard dropped every concept; now the row's entities tag each
    value inside the cell.
    """
    from dgml_core.generation.to_semantic import render_dgml

    row = Block(
        id="r1",
        structure="row",
        cells=["7", "AAA111   Gamma Item Zeta", "3", "$11.00", "$33.00"],
    )
    payload = {
        "concept": "SampleRecord",
        "table": "SampleTable",
        "cells": [  # 6 concepts vs 5 physical cells → positional path is skipped
            "Alpha",
            "Beta",
            "Gamma",
            "Delta",
            "Epsilon",
            "Zeta",
        ],
        "entities": [
            {"quote": "7", "concept": "Alpha"},
            {"quote": "AAA111", "concept": "Beta"},
            {"quote": "Gamma Item Zeta", "concept": "Gamma"},
            {"quote": "3", "concept": "Delta"},
            {"quote": "$11.00", "concept": "Epsilon"},
            {"quote": "$33.00", "concept": "Zeta"},
        ],
    }
    apply_labels([row], {"r1": payload}, doc_name="synthetic.pdf")
    # positional concepts NOT trusted (count mismatch); per-cell entities resolved
    assert row.cell_concepts == []
    # the merged cell carries two inline spans; the "3" quote lands in its own cell
    assert [s.concept for s in row.cell_entities[1]] == ["Beta", "Gamma"]
    assert [s.concept for s in row.cell_entities[0]] == ["Alpha"]
    assert [s.concept for s in row.cell_entities[2]] == ["Delta"]

    out = render_dgml([row], header="<dg:chunk>")
    # merged cell → generic td container holding two concept spans (namespace
    # prefix depends on shared-vocabulary membership; assert role-agnostically).
    assert ">AAA111</" in out and "Beta>" in out
    assert ">Gamma Item Zeta</" in out and "Gamma>" in out
    # single-value cells become whole-cell tags, not bare td
    assert "Alpha" in out and "Delta" in out
    # every cell carries the td layout role — 4 whole-cell concept tags + the
    # 1 generic merged-cell container = 5 cells, all with dg:structure="td"
    assert out.count('dg:structure="td"') == 5
    # the merged cell is the only GENERIC (dg:chunk) td container in the row
    assert out.count('<dg:chunk dg:structure="td"') == 1


def test_matched_table_ignores_entities_no_short_quote_corruption() -> None:
    """When cell count matches, positional concepts win and entities don't corrupt.

    Synthetic data. A short quote like "2" must not be spliced into a cell such
    as "2X9" — and on a matched row the entities are ignored entirely.
    """
    row = Block(
        id="r1",
        structure="row",
        cells=["40", "AAA222", "2X9", "Gamma Name", "2", "$12.00", "$24.00"],
    )
    payload = {
        "concept": "SampleRecord",
        "cells": [  # 7 concepts == 7 cells → positional path trusted
            "Alpha",
            "Beta",
            "Gamma",
            "Delta",
            "Epsilon",
            "Zeta",
            "Eta",
        ],
        "entities": [  # redundant; must be ignored on the matched path
            {"quote": "2X9", "concept": "Gamma"},
            {"quote": "2", "concept": "Epsilon"},
        ],
    }
    apply_labels([row], {"r1": payload}, doc_name="synthetic.pdf")
    assert row.cell_concepts[2] == "Gamma"
    assert row.cell_entities == []  # not layered on a matched row
    # the "2X9" cell stays intact — no "2" spliced out of it
    assert row.cells[2] == "2X9"


def test_find_verbatim_respects_token_boundaries() -> None:
    from dgml_core.generation.label import _find_verbatim

    assert _find_verbatim("2X9Z", "2", 0) == -1  # embedded in a token → no match
    assert _find_verbatim("2 2X9Z", "2", 0) == 0  # standalone token → matches
    assert _find_verbatim("zz 2", "2", 0) == 3


def test_salvage_window_json_recovers_complete_blocks() -> None:
    """A truncated transcription window keeps every block before the cut."""
    from dgml_core.generation.transcribe import _salvage_window_json

    truncated = (
        '```json\n{"continues": "", "blocks": [\n'
        '  {"structure": "heading", "text": "Title"},\n'
        '  {"structure": "p", "text": "complete"},\n'
        '  {"structure": "p", "text": "cut off mid str'  # truncated, no closing
    )
    out = _salvage_window_json(truncated)
    assert out is not None
    assert [b["text"] for b in out["blocks"]] == ["Title", "complete"]


def test_salvage_window_json_none_without_blocks() -> None:
    from dgml_core.generation.transcribe import _salvage_window_json

    assert _salvage_window_json("not json at all") is None
    assert _salvage_window_json('{"continues": "", "blocks": [') is None


# ── entity-container grouping (meta-concept "B") — synthetic data only ────────


def test_build_tree_groups_entity_leaves_under_container() -> None:
    """Contiguous leaves sharing a container-parent wrap in one section node."""
    blocks = [
        Block(
            id="b1",
            structure="p",
            text="Acme Distributing",
            concept="PartyOrganizationName",
        ),
        Block(id="b2", structure="p", text="1 Sample St, Town ST", concept="PartyAddress"),
        Block(id="b3", structure="p", text="(555) 000-0000", concept="PartyPhone"),
        Block(id="b4", structure="p", text="unrelated body text", concept=""),
    ]
    pm = {
        "PartyOrganizationName": "PartyInformation",
        "PartyAddress": "PartyInformation",
        "PartyPhone": "PartyInformation",
    }
    tree = build_tree(blocks, pm)
    section = tree.children[0]
    assert section.kind == "section" and section.concept == "PartyInformation"
    child_concepts: list[str] = []
    for c in section.children:
        assert c.block is not None
        child_concepts.append(c.block.concept)
    assert child_concepts == ["PartyOrganizationName", "PartyAddress", "PartyPhone"]
    # the non-family block ends the run → sibling, not swallowed
    assert tree.children[1].kind == "p"
    # without a parent_map, no grouping happens
    assert all(c.kind == "p" for c in build_tree(blocks).children)


def test_render_dgml_entity_container_wraps_leaves() -> None:
    from dgml_core.generation.to_semantic import render_dgml

    blocks = [
        Block(
            id="b1",
            structure="p",
            text="Acme Distributing",
            concept="PartyOrganizationName",
        ),
        Block(id="b2", structure="p", text="1 Sample St, Town ST", concept="PartyAddress"),
        Block(id="b3", structure="p", text="unrelated body text", concept=""),
    ]
    pm = {
        "PartyOrganizationName": "PartyInformation",
        "PartyAddress": "PartyInformation",
    }
    out = render_dgml(blocks, header="<dg:chunk>", parent_map=pm)
    assert '<docset:PartyInformation dg:structure="section">' in out
    inner = out[out.index("<docset:PartyInformation") : out.index("</docset:PartyInformation>")]
    assert "<docset:PartyOrganizationName" in inner
    assert "<docset:PartyAddress" in inner
    assert "unrelated body text" not in inner  # non-family leaf stays outside


def test_apply_labels_single_role_clause_keeps_concept() -> None:
    """A substantive block with ONLY its own concept duplicated keeps the block
    concept and drops the redundant span (unchanged clause behaviour)."""
    text = "The parties agree to the following warranty disclaimer in full herein."
    b = Block(id="b1", structure="p", text=text, concept="WarrantyDisclaimer")
    payload = {
        "concept": "WarrantyDisclaimer",
        "entities": [{"quote": "warranty disclaimer", "concept": "WarrantyDisclaimer"}],
    }
    apply_labels([b], {"b1": payload}, doc_name="d.pdf")
    assert b.concept == "WarrantyDisclaimer"  # kept — no other-concept values
    assert b.entities == []  # redundant duplicate dropped


def test_entity_quote_matching_lim_tags_the_lim() -> None:
    """A quote that is the list marker itself (a date/number in the lim) is not
    dropped — the concept moves onto the lim and renders as a tagged lim."""
    from dgml_core.generation.to_semantic import render_dgml

    b = Block(id="b1", structure="heading", lim="JAN 05", text="ARRIVE SAMPLE CITY")
    payload = {
        "concept": "TravelDay",
        "entities": [{"quote": "JAN 05", "concept": "TravelDayDate"}],
    }
    warnings = apply_labels([b], {"b1": payload}, doc_name="d.pdf")
    assert b.lim_concept == "TravelDayDate"
    assert b.entities == []  # no inline span — the value lives in the lim
    assert not any("not found verbatim" in w for w in warnings)

    out = render_dgml([b], header="<dg:chunk>")
    assert '<docset:TravelDayDate dg:structure="lim">JAN 05</docset:TravelDayDate>' in out


def test_entity_quote_not_in_text_or_lim_still_dropped() -> None:
    """The lim fallback only fires on an exact lim match — a hallucinated quote
    is still dropped with a warning."""
    b = Block(id="b1", structure="p", lim="(a)", text="some body text")
    payload = {"entities": [{"quote": "NOWHERE", "concept": "SomeConcept"}]}
    warnings = apply_labels([b], {"b1": payload}, doc_name="d.pdf")
    assert b.lim_concept == ""
    assert b.entities == []
    assert any("not found verbatim" in w for w in warnings)


def test_lim_quote_resolves_on_field_blocks_too() -> None:
    """The lim rule applies to every structure — a form field whose numbered
    marker is the labeled value (e.g. an item number) keeps that concept, even
    though field blocks skip the inline-entity loop."""
    b = Block(id="b1", structure="field", lim="7", label="Qty", value="12 units")
    payload = {
        "concept": "OrderedQuantity",
        "entities": [{"quote": "7", "concept": "ItemNumber"}],
    }
    apply_labels([b], {"b1": payload}, doc_name="d.pdf")
    assert b.lim_concept == "ItemNumber"
    assert b.concept == "OrderedQuantity"  # field concept untouched


def test_column_propagation_never_buries_cell_entity_evidence() -> None:
    """A row whose cell carries a resolved whole-cell entity keeps it: the
    cross-row uniform column concept must not fill that slot (the renderer
    prefers positional concepts, so filling it would bury the verbatim tag)."""
    from dgml_core.generation.label import propagate_table_consistency
    from dgml_core.generation.to_semantic import render_dgml

    rows = [
        Block(id="r1", structure="row", cells=["10", "x"], cell_concepts=["Amount", ""]),
        Block(id="r2", structure="row", cells=["20", "y"], cell_concepts=["Amount", ""]),
        Block(
            id="r3",
            structure="row",
            cells=["ZZ", "z"],
            cell_concepts=[],
            cell_entities=[[Span(start=0, end=2, concept="RegionCode")], []],
        ),
    ]
    propagate_table_consistency(rows)
    assert rows[0].cell_concepts[0] == "Amount"
    assert rows[2].cell_concepts[0] == ""  # entity evidence preserved the slot
    out = render_dgml(rows, header="<dg:chunk>")
    assert '<docset:RegionCode dg:structure="td">ZZ</docset:RegionCode>' in out


def test_count_matched_row_keeps_positional_and_partial_entities() -> None:
    """When the labeler's column model matches the physical cells, positional
    concepts win whole-cell — but PARTIAL entity spans (sub-values packed in a
    cell) are still resolved and render inside the positionally-tagged td."""
    from dgml_core.generation.to_semantic import render_dgml

    b = Block(id="r1", structure="row", cells=["WidgetKit Pro - 25 seats", "$500"])
    payload = {
        "concept": "ProductLine",
        "cells": ["ProductName", "LinePrice"],
        "entities": [
            {"quote": "25 seats", "concept": "SeatCount"},  # partial → kept
            {
                "quote": "$500",
                "concept": "OriginalPrice",
            },  # whole-cell conflict → dropped
        ],
    }
    apply_labels([b], {"r1": payload}, doc_name="d.pdf")
    assert b.cell_concepts == ["ProductName", "LinePrice"]
    assert [s.concept for s in b.cell_entities[0]] == ["SeatCount"]
    assert b.cell_entities[1] == []  # whole-cell span: positional wins

    out = render_dgml([b], header="<dg:chunk>")
    # positional tag kept on the split cell, sub-value wrapped inside it
    assert '<docset:ProductName dg:structure="td">' in out
    assert "<docset:SeatCount>25 seats</docset:SeatCount>" in out
    assert '<docset:LinePrice dg:structure="td"' in out  # untouched whole-cell column


def test_field_secondary_entities_render_inside_value() -> None:
    """A field whose value packs several sub-values keeps them: partial entity
    spans resolve inside block.value and render as inline concept spans within
    the concept-wrapped value; a whole-value span is dropped (block concept
    wins whole-value, as for table cells)."""
    from dgml_core.generation.to_semantic import render_dgml

    b = Block(
        id="f1",
        structure="field",
        label="License",
        value="No. 0000 - Sample Org LLC",
        concept="LicenseNumber",
    )
    payload = {
        "concept": "LicenseNumber",
        "entities": [
            {
                "quote": "Sample Org LLC",
                "concept": "OrganizationName",
            },  # partial → kept
            {
                "quote": "No. 0000 - Sample Org LLC",
                "concept": "LicenseLine",
            },  # whole → drop
        ],
    }
    apply_labels([b], {"f1": payload}, doc_name="d.pdf")
    assert [s.concept for s in b.entities] == ["OrganizationName"]
    assert b.concept == "LicenseNumber"

    out = render_dgml([b], header="<dg:chunk>")
    assert "<docset:LicenseNumber>No. 0000 - " in out.replace("\n", "")
    assert "<docset:OrganizationName>Sample Org LLC</docset:OrganizationName>" in out


def test_field_label_entities_render_inside_label() -> None:
    """A packed 'label' carrying real values (a code and a name in one line)
    keeps them as inline spans inside the rendered label header — whole and
    partial label quotes both survive (no conflict: the block concept wraps
    the value, never the label)."""
    from dgml_core.generation.to_semantic import render_dgml

    b = Block(id="f1", structure="field", label="No. 0000 - Sample Org LLC", value="")
    payload = {
        "concept": "",
        "entities": [
            {"quote": "0000", "concept": "LicenseCode"},
            {"quote": "Sample Org LLC", "concept": "OrganizationName"},
        ],
    }
    apply_labels([b], {"f1": payload}, doc_name="d.pdf")
    assert [s.concept for s in b.label_entities] == ["LicenseCode", "OrganizationName"]
    assert b.entities == []  # value empty — nothing resolved there

    out = render_dgml([b], header="<dg:chunk>")
    assert "<docset:LicenseCode" in out and ">0000</docset:LicenseCode>" in out
    assert "<docset:OrganizationName>Sample Org LLC</docset:OrganizationName>" in out


def test_row_cells_accept_object_entries() -> None:
    """The labeler sometimes answers a cell with an object ({"concept": ...,
    "entities": [...]}) instead of a concept string. The concept is read and
    the nested entities fold into per-cell resolution — never str()-ified into
    a garbage mega-tag."""
    b = Block(id="r1", structure="row", cells=["WidgetKit Pro x 25 seats", "$500"])
    payload = {
        "concept": "ProductLine",
        "cells": [
            {
                "concept": "ProductName",
                "entities": [
                    {"quote": "WidgetKit Pro", "concept": "ProductName"},
                    {"quote": "25 seats", "concept": "SeatCount"},
                ],
            },
            "LinePrice",
        ],
    }
    apply_labels([b], {"r1": payload}, doc_name="d.pdf")
    assert b.cell_concepts == ["ProductName", "LinePrice"]
    assert [s.concept for s in b.cell_entities[0]] == ["ProductName", "SeatCount"]
    assert all(len(c) < 30 for c in b.cell_concepts)  # no dict-repr mega-tag


def test_sanitize_concept_rejects_garbage_length() -> None:
    """A concept that normalizes past the length cap is garbage (a str()-ified
    payload or a sentence), not a tag — dropped, while long legit names live."""
    assert sanitize_concept("IndependentlyDevelopedIntellectualProperty") != ""
    assert sanitize_concept("x" * 100) == ""
    mangled = str(
        {
            "concept": "ProductName",
            "entities": [
                {"quote": "WidgetKit Mobile", "concept": "ProductName"},
                {"quote": "50 seats", "concept": "ProductSeatCount"},
            ],
        }
    )
    assert sanitize_concept(mangled) == ""


def test_header_row_labeled_as_data_is_demoted() -> None:
    """A first row whose cells are bare words while every typed column below
    carries digit-bearing values is a printed header row — its concepts are
    cleared so the column titles never become the columns' first values."""
    from dgml_core.generation.label import propagate_table_consistency

    rows = [
        Block(
            id="r0",
            structure="row",
            cells=["Code", "Qty", "Price"],
            cell_concepts=["ItemCode", "ItemQty", "ItemPrice"],  # labeler leak
            concept="LineItem",
        ),
        Block(
            id="r1",
            structure="row",
            cells=["A100", "2", "$10.00"],
            cell_concepts=["ItemCode", "ItemQty", "ItemPrice"],
            concept="LineItem",
        ),
        Block(
            id="r2",
            structure="row",
            cells=["B200", "5", "$25.00"],
            cell_concepts=["ItemCode", "ItemQty", "ItemPrice"],
            concept="LineItem",
        ),
    ]
    propagate_table_consistency(rows)
    assert rows[0].cell_concepts == [] and rows[0].concept == ""  # header demoted
    assert rows[1].cell_concepts == ["ItemCode", "ItemQty", "ItemPrice"]  # data intact


def test_all_text_table_keeps_its_first_row() -> None:
    """A table whose columns are textual (no typed columns) has no basis for
    header detection — the first row keeps its concepts."""
    from dgml_core.generation.label import propagate_table_consistency

    rows = [
        Block(
            id="r0",
            structure="row",
            cells=["Alpha Corp", "Berlin"],
            cell_concepts=["PartyName", "PartyCity"],
        ),
        Block(
            id="r1",
            structure="row",
            cells=["Beta GmbH", "Munich"],
            cell_concepts=["PartyName", "PartyCity"],
        ),
        Block(
            id="r2",
            structure="row",
            cells=["Gamma LLC", "Hamburg"],
            cell_concepts=["PartyName", "PartyCity"],
        ),
    ]
    propagate_table_consistency(rows)
    assert rows[0].cell_concepts == ["PartyName", "PartyCity"]  # untouched
