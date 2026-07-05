"""Tests for dgml2jsonld XAST-convention JSON-LD export."""

from __future__ import annotations

import textwrap
from pathlib import Path

from dgml2jsonld import xml_to_jsonld


def parse(xml: str) -> dict:
    tmp = Path(__file__).parent / "_tmp.dgml.xml"
    tmp.write_text(textwrap.dedent(xml).strip(), encoding="utf-8")
    try:
        return xml_to_jsonld(tmp)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 1. Full canonical example — exact output match
# ---------------------------------------------------------------------------

FULL_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
          xmlns:docset="http://dgml.io/acme-corp/master-services-agreements#"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <docset:IndemnificationClause dg:origin="3 180 400 2360 1200">
    <dg:chunk dg:structure="header" dg:style="font-size: 1.5em; text-transform: uppercase"
              dg:origin="3 180 400 2360 450">
      <dg:chunk dg:structure="lim">3.1</dg:chunk>
      Indemnification
    </dg:chunk>
    <docset:LiabilityCap xsi:type="decimal" dg:value="500000"
        dg:origin="3 180 500 2360 540">$500,000</docset:LiabilityCap>
    <docset:EffectiveDate xsi:type="date" dg:value="2024-01-01"
        dg:origin="3 180 550 2360 590">January 1, 2024</docset:EffectiveDate>
    <docset:VendorRef dg:itemprop="docset:paidBy" dg:href="#vendor-co"/>
  </docset:IndemnificationClause>
  <docset:VendorName xml:id="vendor-co"
      dg:origin="1 220 150 680 200">MagicSoft, Inc.</docset:VendorName>
</dg:chunk>
"""

FULL_EXPECTED = {
    "@context": {
        "xast": "http://dgml.io/ns/xast#",
        "children": {"@id": "xast:children", "@container": "@list"},
        "attributes": {"@id": "xast:attributes", "@container": "@index"},
        "nodeType": {"@id": "xast:nodeType"},
        "value": {"@id": "xast:value"},
        "dg": "http://dgml.io/ns/dg#",
        "docset": "http://dgml.io/acme-corp/master-services-agreements#",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    },
    "@graph": [
        {
            "nodeType": "xast:element",
            "@type": "dg:chunk",
            "children": [
                {
                    "nodeType": "xast:element",
                    "@type": "docset:IndemnificationClause",
                    "attributes": {"dg:origin": "3 180 400 2360 1200"},
                    "children": [
                        {
                            "nodeType": "xast:element",
                            "@type": "dg:chunk",
                            "attributes": {
                                "dg:structure": "header",
                                "dg:style": "font-size: 1.5em; text-transform: uppercase",
                                "dg:origin": "3 180 400 2360 450",
                            },
                            "children": [
                                {
                                    "nodeType": "xast:element",
                                    "@type": "dg:chunk",
                                    "attributes": {"dg:structure": "lim"},
                                    "children": [{"nodeType": "xast:text", "value": "3.1"}],
                                },
                                {"nodeType": "xast:text", "value": "Indemnification"},
                            ],
                        },
                        {
                            "nodeType": "xast:element",
                            "@type": "docset:LiabilityCap",
                            "attributes": {
                                "xsi:type": "decimal",
                                "dg:value": "500000",
                                "dg:origin": "3 180 500 2360 540",
                            },
                            "children": [{"nodeType": "xast:text", "value": "$500,000"}],
                        },
                        {
                            "nodeType": "xast:element",
                            "@type": "docset:EffectiveDate",
                            "attributes": {
                                "xsi:type": "date",
                                "dg:value": "2024-01-01",
                                "dg:origin": "3 180 550 2360 590",
                            },
                            "children": [{"nodeType": "xast:text", "value": "January 1, 2024"}],
                        },
                        {
                            "nodeType": "xast:element",
                            "@type": "docset:VendorRef",
                            "docset:paidBy": {"@id": "#vendor-co"},
                            "children": [],
                        },
                    ],
                },
                {
                    "nodeType": "xast:element",
                    "@type": "docset:VendorName",
                    "@id": "#vendor-co",
                    "attributes": {"dg:origin": "1 220 150 680 200"},
                    "children": [{"nodeType": "xast:text", "value": "MagicSoft, Inc."}],
                },
            ],
        }
    ],
}


def test_full_example() -> None:
    tmp = Path(__file__).parent / "_full.dgml.xml"
    tmp.write_text(FULL_XML, encoding="utf-8")
    try:
        result = xml_to_jsonld(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    assert result == FULL_EXPECTED


# ---------------------------------------------------------------------------
# 2. Plain id → prefixed with #
# ---------------------------------------------------------------------------


def test_plain_id_prefixed() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#">
          <docset:Clause xmlns:docset="http://dgml.io/x/y#" id="clause-1">text</docset:Clause>
        </dg:chunk>
    """)
    clause = result["@graph"][0]["children"][0]
    assert clause["@id"] == "#clause-1"


# ---------------------------------------------------------------------------
# 3. URN id → unchanged
# ---------------------------------------------------------------------------


def test_urn_id_unchanged() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#">
          <docset:Party xmlns:docset="http://dgml.io/x/y#"
              id="urn:dgml:org:acme">Acme</docset:Party>
        </dg:chunk>
    """)
    party = result["@graph"][0]["children"][0]
    assert party["@id"] == "urn:dgml:org:acme"


# ---------------------------------------------------------------------------
# 4. dg:itemprop/dg:href → named link property on the element itself
# ---------------------------------------------------------------------------


def test_itemprop_href_on_element() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
                  xmlns:docset="http://dgml.io/x/y#">
          <docset:SignatoryName xml:id="sig-1"
              dg:itemprop="signatoryOf" dg:href="#vendor-co"
              dg:origin="11 460 2430 1200 2475">Jane Smith</docset:SignatoryName>
        </dg:chunk>
    """)
    signatory = result["@graph"][0]["children"][0]
    assert signatory["@type"] == "docset:SignatoryName"
    assert signatory["@id"] == "#sig-1"
    assert signatory["signatoryOf"] == {"@id": "#vendor-co"}
    assert signatory["children"] == [{"nodeType": "xast:text", "value": "Jane Smith"}]


def test_itemprop_href_multi_valued() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
                  xmlns:docset="http://dgml.io/x/y#">
          <docset:QuarterlyRevenue
              dg:itemprop="aggregates"
              dg:href="#monthly-1 #monthly-2 #monthly-3">$300,000</docset:QuarterlyRevenue>
        </dg:chunk>
    """)
    quarterly = result["@graph"][0]["children"][0]
    assert quarterly["aggregates"] == [
        {"@id": "#monthly-1"},
        {"@id": "#monthly-2"},
        {"@id": "#monthly-3"},
    ]


def test_itemprop_href_empty_element() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
                  xmlns:docset="http://dgml.io/x/y#">
          <docset:PaymentTerms>
            <docset:VendorRef dg:itemprop="docset:paidBy" dg:href="#vendor-co"/>
          </docset:PaymentTerms>
        </dg:chunk>
    """)
    terms = result["@graph"][0]["children"][0]
    assert len(terms["children"]) == 1
    ref = terms["children"][0]
    assert ref["@type"] == "docset:VendorRef"
    assert ref["docset:paidBy"] == {"@id": "#vendor-co"}
    assert ref["children"] == []


# ---------------------------------------------------------------------------
# 5. Text-only element → text node in children
# ---------------------------------------------------------------------------


def test_text_only_element() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
                  xmlns:docset="http://dgml.io/x/y#">
          <docset:InvoiceCycle>Net 30</docset:InvoiceCycle>
        </dg:chunk>
    """)
    cycle = result["@graph"][0]["children"][0]
    assert cycle["children"] == [{"nodeType": "xast:text", "value": "Net 30"}]


# ---------------------------------------------------------------------------
# 6. Mixed content — correct children order preserved
# ---------------------------------------------------------------------------


def test_mixed_content_order() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
                  xmlns:docset="http://dgml.io/x/y#">
          <docset:Clause>This is <docset:Party>Acme</docset:Party> and done.</docset:Clause>
        </dg:chunk>
    """)
    clause = result["@graph"][0]["children"][0]
    children = clause["children"]
    assert children[0] == {"nodeType": "xast:text", "value": "This is"}
    assert children[1]["@type"] == "docset:Party"
    assert children[2] == {"nodeType": "xast:text", "value": "and done."}


# ---------------------------------------------------------------------------
# 7. Comment and PI nodes are skipped
# ---------------------------------------------------------------------------


def test_comments_and_pi_skipped() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
                  xmlns:docset="http://dgml.io/x/y#">
          <!-- this is a comment -->
          <?pi target?>
          <docset:Value>hello</docset:Value>
        </dg:chunk>
    """)
    children = result["@graph"][0]["children"]
    assert len(children) == 1
    assert children[0]["@type"] == "docset:Value"


# ---------------------------------------------------------------------------
# 8. Default namespace → @vocab
# ---------------------------------------------------------------------------


def test_default_namespace_vocab() -> None:
    result = parse("""\
        <chunk xmlns="http://dgml.io/x/y#">
          <Value>hello</Value>
        </chunk>
    """)
    assert result["@context"]["@vocab"] == "http://dgml.io/x/y#"
    assert result["@graph"][0]["@type"] == "chunk"


# ---------------------------------------------------------------------------
# 9. dg:origin always present in attributes when set
# ---------------------------------------------------------------------------


def test_dg_origin_in_attributes() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
                  xmlns:docset="http://dgml.io/x/y#">
          <docset:LiabilityCap dg:origin="2 100 200 300 400">$500,000</docset:LiabilityCap>
        </dg:chunk>
    """)
    cap = result["@graph"][0]["children"][0]
    assert cap["attributes"]["dg:origin"] == "2 100 200 300 400"


# ---------------------------------------------------------------------------
# 10. Empty attributes → attributes key omitted
# ---------------------------------------------------------------------------


def test_empty_attributes_omitted() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
                  xmlns:docset="http://dgml.io/x/y#">
          <docset:Value>hello</docset:Value>
        </dg:chunk>
    """)
    value = result["@graph"][0]["children"][0]
    assert "attributes" not in value


# ---------------------------------------------------------------------------
# 11. dg:structure passes through as regular attribute
# ---------------------------------------------------------------------------


def test_dg_structure_passthrough() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#">
          <dg:chunk dg:structure="header">Heading</dg:chunk>
        </dg:chunk>
    """)
    header = result["@graph"][0]["children"][0]
    assert header["attributes"]["dg:structure"] == "header"


# ---------------------------------------------------------------------------
# 12. dg:style passes through as regular attribute
# ---------------------------------------------------------------------------


def test_dg_style_passthrough() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#">
          <dg:chunk dg:style="font-weight: bold; color: gray">Bold</dg:chunk>
        </dg:chunk>
    """)
    styled = result["@graph"][0]["children"][0]
    assert styled["attributes"]["dg:style"] == "font-weight: bold; color: gray"


# ---------------------------------------------------------------------------
# 13. xast: terms always present in @context
# ---------------------------------------------------------------------------


def test_xast_context_always_present() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#">
          <dg:chunk>hello</dg:chunk>
        </dg:chunk>
    """)
    ctx = result["@context"]
    assert ctx["xast"] == "http://dgml.io/ns/xast#"
    assert ctx["children"] == {"@id": "xast:children", "@container": "@list"}
    assert ctx["attributes"] == {"@id": "xast:attributes", "@container": "@index"}
    assert ctx["nodeType"] == {"@id": "xast:nodeType"}
    assert ctx["value"] == {"@id": "xast:value"}


# ---------------------------------------------------------------------------
# 14. nodeType is xast:element for elements, xast:text for text nodes
# ---------------------------------------------------------------------------


def test_node_types() -> None:
    result = parse("""\
        <dg:chunk xmlns:dg="http://dgml.io/ns/dg#"
                  xmlns:docset="http://dgml.io/x/y#">
          <docset:Value>hello</docset:Value>
        </dg:chunk>
    """)
    elem = result["@graph"][0]
    assert elem["nodeType"] == "xast:element"
    child_elem = elem["children"][0]
    assert child_elem["nodeType"] == "xast:element"
    text_node = child_elem["children"][0]
    assert text_node["nodeType"] == "xast:text"
    assert text_node["value"] == "hello"
