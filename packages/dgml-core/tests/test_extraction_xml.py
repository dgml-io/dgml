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

from dgml_core.extraction_schema import json_schema_to_rnc, parse_rnc
from dgml_core.extraction_xml import (
    carry_extraction_over,
    count_dropped_refs,
    dgml_xml_to_values,
    embed_extraction_into,
    has_extraction,
    standalone_extraction_doc,
    unattributed_computed_fields,
)

_SCHEMA = {
    "definitions": {"grounded_field": {"type": "object"}},
    "properties": {
        "vendor_name": {"$ref": "#/definitions/grounded_field"},
        "liability_cap": {"$ref": "#/definitions/grounded_field"},
        "indemnification": {
            "type": "object",
            "properties": {"indemnifying_party": {"$ref": "#/definitions/grounded_field"}},
        },
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_name": {"$ref": "#/definitions/grounded_field"},
                    "unit_price": {"$ref": "#/definitions/grounded_field"},
                },
            },
        },
    },
}


def _vocab() -> object:
    return parse_rnc(json_schema_to_rnc(_SCHEMA, workspace="acme", docset_name="MSA"))


def test_standalone_doc_wraps_fields_in_dg_extraction() -> None:
    vocab = _vocab()
    values = {
        "LiabilityCap": {
            "text": "$500,000",
            "locations": [{"page_number": 2, "bounding_box": [460, 310, 1800, 355]}],
        }
    }
    xml = standalone_extraction_doc(values, vocab=vocab)  # type: ignore[arg-type]
    # extracted values live inside dg:extraction under the root dg:chunk
    assert "<dg:extraction>" in xml
    assert "<dg:chunk" in xml
    assert has_extraction(xml)
    # typed value: decimal normalization + dg:value, plus dg:origin
    assert 'xsi:type="decimal"' in xml
    assert 'dg:value="500000"' in xml
    assert 'dg:origin="2 460 310 1800 355"' in xml
    assert "<docset:LiabilityCap" in xml


def test_multibox_origin_joined_with_semicolons() -> None:
    vocab = _vocab()
    values = {
        "VendorName": {
            "text": "Acme",
            "locations": [
                {"page_number": 1, "bounding_box": [1, 2, 3, 4]},
                {"page_number": 1, "bounding_box": [5, 6, 7, 8]},
            ],
        }
    }
    xml = standalone_extraction_doc(values, vocab=vocab)  # type: ignore[arg-type]
    assert 'dg:origin="1 1 2 3 4; 1 5 6 7 8"' in xml


def test_roundtrip_values_xml_values() -> None:
    vocab = _vocab()
    values = {
        "VendorName": {
            "text": "Acme",
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        },
        "Indemnification": {
            "IndemnifyingParty": {
                "text": "Vendor",
                "locations": [{"page_number": 3, "bounding_box": [180, 450, 900, 490]}],
            }
        },
        "LineItems": [
            {
                "ProductName": {"text": "Widget", "locations": []},
                "UnitPrice": {"text": "9", "locations": []},
            }
        ],
    }
    xml = standalone_extraction_doc(values, vocab=vocab)  # type: ignore[arg-type]
    back = dgml_xml_to_values(xml, vocab=vocab)  # type: ignore[arg-type]

    # Leaf + container survive exactly (no dg:value for non-normalizable text).
    assert back["VendorName"] == values["VendorName"]
    assert back["Indemnification"] == values["Indemnification"]
    # Single-item collection stays a list thanks to the vocab-guided projection.
    assert isinstance(back["LineItems"], list)
    assert len(back["LineItems"]) == 1
    # "9" normalizes to an integer, so the projection gains a value field.
    assert back["LineItems"][0]["UnitPrice"]["value"] == "9"


def test_embed_into_existing_document_tree() -> None:
    """full-extraction: dg:extraction is added as a sibling of the doc tree,
    and re-embedding replaces the prior dg:extraction (no duplicate)."""
    vocab = _vocab()
    core = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"\n'
        '          xmlns:docset="http://www.dgml.io/acme/MSA"\n'
        '          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        "  <docset:Body>the whole document tree</docset:Body>\n"
        "</dg:chunk>\n"
    )
    values = {"VendorName": {"text": "Acme", "locations": []}}
    out = embed_extraction_into(core, values, vocab=vocab)  # type: ignore[arg-type]
    assert "the whole document tree" in out  # doc tree preserved
    assert out.count("<dg:extraction>") == 1
    # Re-embedding replaces rather than appends.
    out2 = embed_extraction_into(out, values, vocab=vocab)  # type: ignore[arg-type]
    assert out2.count("<dg:extraction>") == 1
    assert has_extraction(out2)


def test_collection_of_text_leaves_roundtrip() -> None:
    """A list of grounded text values serializes as repeated leaf item elements
    (each with text + dg:origin) and projects back to a list of leaf dicts."""
    schema = {
        "definitions": {"grounded_field": {"type": "object"}},
        "properties": {
            "learning_outcomes": {
                "type": "array",
                "items": {"$ref": "#/definitions/grounded_field"},
            }
        },
    }
    vocab = parse_rnc(json_schema_to_rnc(schema, workspace="ws", docset_name="d"))
    values = {
        "LearningOutcomes": [
            {
                "text": "Analyze data",
                "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
            },
            {"text": "Communicate findings", "locations": []},
        ]
    }
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert xml.count("<docset:LearningOutcome ") + xml.count("<docset:LearningOutcome>") == 2
    assert "Analyze data" in xml
    back = dgml_xml_to_values(xml, vocab=vocab)
    assert back["LearningOutcomes"] == values["LearningOutcomes"]


_CHOICE_RNC = """\
namespace docset = "http://www.dgml.io/acme/programs#"

TotalCredits =
  element docset:TotalCredits {
    ( xsd:integer | ( MinTotalCredits, MaxTotalCredits ) )
  }

MinTotalCredits =
  element docset:MinTotalCredits {
    xsd:integer
  }

MaxTotalCredits =
  element docset:MaxTotalCredits {
    xsd:integer
  }
"""


def test_choice_scalar_branch() -> None:
    """The scalar alternative: TotalCredits carries an integer directly, typed
    from the schema (xsd:integer wins over heuristics on '181 CREDITS')."""
    vocab = parse_rnc(_CHOICE_RNC)
    values = {"TotalCredits": {"text": "181 CREDITS", "locations": []}}
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert '<docset:TotalCredits xsi:type="integer" dg:value="181"' in xml
    assert "MinTotalCredits" not in xml
    assert dgml_xml_to_values(xml, vocab=vocab)["TotalCredits"]["value"] == "181"


def test_choice_range_branch() -> None:
    """The group alternative: a MinTotalCredits/MaxTotalCredits pair of integers."""
    vocab = parse_rnc(_CHOICE_RNC)
    values = {
        "TotalCredits": {
            "MinTotalCredits": {"text": "180", "locations": []},
            "MaxTotalCredits": {"text": "182", "locations": []},
        }
    }
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert '<docset:MinTotalCredits xsi:type="integer" dg:value="180"' in xml
    assert '<docset:MaxTotalCredits xsi:type="integer" dg:value="182"' in xml
    back = dgml_xml_to_values(xml, vocab=vocab)["TotalCredits"]
    assert back["MinTotalCredits"]["value"] == "180"
    assert back["MaxTotalCredits"]["value"] == "182"


def test_has_extraction_false_for_generate_only_file() -> None:
    core = (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" '
        'xmlns:docset="http://www.dgml.io/acme/MSA">'
        "<docset:Body>tree</docset:Body></dg:chunk>"
    )
    assert has_extraction(core) is False


def test_single_item_collection_inferred_as_container_without_vocab() -> None:
    """The schema-less inference path cannot tell a one-item collection from a
    container — the vocab-guided path (used by the CLI) is what disambiguates."""
    vocab = _vocab()
    values = {"LineItems": [{"ProductName": {"text": "W", "locations": []}}]}
    xml = standalone_extraction_doc(values, vocab=vocab)  # type: ignore[arg-type]
    inferred = dgml_xml_to_values(xml)  # no vocab
    assert isinstance(inferred["LineItems"], dict)  # mis-inferred as container
    guided = dgml_xml_to_values(xml, vocab=vocab)  # type: ignore[arg-type]
    assert isinstance(guided["LineItems"], list)  # correct with the schema


def test_missing_field_is_omitted() -> None:
    vocab = _vocab()
    xml = standalone_extraction_doc({"VendorName": {"text": "Acme", "locations": []}}, vocab=vocab)  # type: ignore[arg-type]
    assert "LiabilityCap" not in xml  # not extracted ⇒ not emitted


# ── computed (reasoned) fields — spec §7/§13 ──────────────────────────────────

# The spec's §13 invoice: line items are read off the page; InvoiceTotal is
# derived from them (`## Prompt:` carries the rule).
_INVOICE_RNC = """\
namespace docset = "http://www.dgml.io/acme/invoices#"

Invoice =
  element docset:Invoice {
    (text | VendorName | LineItems | InvoiceTotal)*
  }

VendorName =
  element docset:VendorName {
    text
  }

LineItems =
  element docset:LineItems {
    LineItem*
  }

LineItem =
  element docset:LineItem {
    (text | Quantity | UnitPrice)*
  }

Quantity =
  element docset:Quantity {
    xsd:integer
  }

UnitPrice =
  element docset:UnitPrice {
    xsd:decimal
  }

## Prompt: Compute as sum of Quantity times UnitPrice for each LineItem
InvoiceTotal =
  element docset:InvoiceTotal {
    xsd:decimal
  }
"""


def _invoice_values() -> dict[str, object]:
    return {
        "Invoice": {
            "VendorName": {
                "text": "MagicSoft, Inc.",
                "locations": [{"page_number": 1, "bounding_box": [220, 150, 680, 200]}],
            },
            "LineItems": [
                {
                    "Quantity": {
                        "text": "3",
                        "locations": [{"page_number": 2, "bounding_box": [900, 400, 1100, 440]}],
                    },
                    "UnitPrice": {
                        "text": "$49.95",
                        "locations": [{"page_number": 2, "bounding_box": [1100, 400, 1400, 440]}],
                    },
                },
                {
                    "Quantity": {
                        "text": "1",
                        "locations": [{"page_number": 2, "bounding_box": [900, 450, 1100, 490]}],
                    },
                    "UnitPrice": {
                        "text": "$200.00",
                        "locations": [{"page_number": 2, "bounding_box": [1100, 450, 1400, 490]}],
                    },
                },
            ],
            "InvoiceTotal": {
                "text": "$349.85",
                "value": "349.85",
                "computed": True,
                "derived_from": [
                    "Invoice.LineItems[0].Quantity",
                    "Invoice.LineItems[0].UnitPrice",
                    "Invoice.LineItems[1].Quantity",
                    "Invoice.LineItems[1].UnitPrice",
                ],
            },
        }
    }


def test_computed_field_emits_spec_attribute_set() -> None:
    """A computed leaf carries dg:origin="computed", a mandatory dg:value,
    and dg:itemprop/dg:href naming the sources; each source element gains
    an xml:id derived from its path — the spec §13 InvoiceTotal shape."""
    vocab = parse_rnc(_INVOICE_RNC)
    xml = standalone_extraction_doc(_invoice_values(), vocab=vocab)
    assert 'dg:origin="computed"' in xml
    assert 'dg:itemprop="computedFrom"' in xml
    assert (
        'dg:href="#invoice-line-items-0-quantity; #invoice-line-items-0-unit-price; '
        '#invoice-line-items-1-quantity; #invoice-line-items-1-unit-price"' in xml
    )
    # Schema-declared xsd:decimal + the model's canonical value.
    assert 'xsi:type="decimal" dg:value="349.85"' in xml
    assert ">$349.85</docset:InvoiceTotal>" in xml
    # Sources carry the referenced ids; unreferenced elements carry none.
    assert 'xml:id="invoice-line-items-0-quantity"' in xml
    assert 'xml:id="invoice-line-items-1-unit-price"' in xml
    assert "VendorName xml:id" not in xml


def test_computed_field_roundtrip() -> None:
    """XML → values reconstructs the computed leaf, mapping #id hrefs back
    to dotted paths; grounded leaves are untouched."""
    vocab = parse_rnc(_INVOICE_RNC)
    values = _invoice_values()
    back = dgml_xml_to_values(standalone_extraction_doc(values, vocab=vocab), vocab=vocab)
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    assert back["Invoice"]["InvoiceTotal"] == invoice["InvoiceTotal"]
    assert back["Invoice"]["VendorName"] == invoice["VendorName"]
    # Grounded line-item leaves gain their normalized dg:value on the way back.
    assert back["Invoice"]["LineItems"][0]["UnitPrice"]["value"] == "49.95"


def test_computed_dangling_refs_dropped() -> None:
    """Malformed or dangling derived_from entries lose their href; when none
    survive, the element keeps dg:origin="computed" + dg:value but no
    itemprop/href pair."""
    vocab = parse_rnc(_INVOICE_RNC)
    values = _invoice_values()
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    invoice["InvoiceTotal"]["derived_from"] = [
        "Invoice.LineItems[9].Quantity",  # index out of range
        "Invoice.NoSuchField",  # unknown key
        "not a [valid path",  # unparseable
    ]
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert 'dg:origin="computed"' in xml
    assert 'dg:value="349.85"' in xml
    assert "dg:itemprop" not in xml
    assert "dg:href" not in xml
    assert "xml:id" not in xml


def test_computed_partial_dangling_keeps_resolvable_refs() -> None:
    vocab = parse_rnc(_INVOICE_RNC)
    values = _invoice_values()
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    invoice["InvoiceTotal"]["derived_from"] = [
        "Invoice.LineItems[0].Quantity",
        "Invoice.Bogus",
    ]
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert 'dg:href="#invoice-line-items-0-quantity"' in xml


def test_computed_without_canonical_value_falls_back_to_text() -> None:
    """No model-provided ``value``: dg:value comes from the schema-typed
    normalization of the display text (spec: computed always carries dg:value)."""
    vocab = parse_rnc(_INVOICE_RNC)
    values = _invoice_values()
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    del invoice["InvoiceTotal"]["value"]
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert 'xsi:type="decimal" dg:value="349.85"' in xml  # normalized from "$349.85"


def test_count_dropped_refs() -> None:
    values = _invoice_values()
    assert count_dropped_refs(values) == 0
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    invoice["InvoiceTotal"]["derived_from"] = [
        "Invoice.LineItems[0].Quantity",  # resolves
        "Invoice.LineItems[9].Quantity",  # dangles
        "not a [valid path",  # malformed
        42,  # non-string
    ]
    assert count_dropped_refs(values) == 3


def test_unattributed_computed_fields() -> None:
    """The consistency-check helper names computed elements with no dg:href;
    attributed computed fields and grounded fields don't trip it."""
    vocab = parse_rnc(_INVOICE_RNC)
    xml = standalone_extraction_doc(_invoice_values(), vocab=vocab)
    assert unattributed_computed_fields(xml) == []

    values = _invoice_values()
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    invoice["InvoiceTotal"]["derived_from"] = ["Invoice.Bogus"]  # all refs dangle
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert unattributed_computed_fields(xml) == ["InvoiceTotal"]


def test_computed_crossfile_href_stays_raw_on_parse() -> None:
    """A dg:href target outside this file (fileid#id form) can't be mapped to
    a values path — it survives the projection as the raw reference."""
    xml = (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" '
        'xmlns:docset="http://www.dgml.io/acme/invoices#">'
        "<dg:extraction>"
        '<docset:InvoiceTotal dg:origin="computed" dg:value="10" '
        'dg:itemprop="computedFrom" dg:href="5kqt9r5fowno#notice-1; #unknown-local">'
        "$10</docset:InvoiceTotal>"
        "</dg:extraction></dg:chunk>"
    )
    back = dgml_xml_to_values(xml)
    total = back["InvoiceTotal"]
    assert total["computed"] is True
    assert total["derived_from"] == ["5kqt9r5fowno#notice-1", "#unknown-local"]
    assert "locations" not in total


def test_carry_extraction_over_moves_element_verbatim() -> None:
    """The dg:extraction element (values, origins, hrefs, xml:ids) survives a
    fresh tree render byte-identically in content."""
    vocab = parse_rnc(_INVOICE_RNC)
    prior = standalone_extraction_doc(_invoice_values(), vocab=vocab)
    fresh_tree = (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">'
        '<docset2:Body xmlns:docset2="http://other/ns">the generated tree</docset2:Body>'
        "</dg:chunk>"
    )
    merged = carry_extraction_over(prior, fresh_tree)
    assert "the generated tree" in merged
    back = dgml_xml_to_values(merged, vocab=vocab)
    values = _invoice_values()
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    assert back["Invoice"]["InvoiceTotal"] == invoice["InvoiceTotal"]
    assert back["Invoice"]["VendorName"]["text"] == "MagicSoft, Inc."


def test_carry_extraction_over_replaces_existing_extraction() -> None:
    vocab = parse_rnc(_INVOICE_RNC)
    prior = standalone_extraction_doc(_invoice_values(), vocab=vocab)
    target_with_stale = (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" xmlns:docset="http://www.dgml.io/acme/invoices#">'
        "<a>tree</a>"
        "<dg:extraction><docset:VendorName>Stale Corp</docset:VendorName></dg:extraction>"
        "</dg:chunk>"
    )
    merged = carry_extraction_over(prior, target_with_stale)
    assert "Stale Corp" not in merged
    assert "MagicSoft, Inc." in merged
    assert merged.count("<dg:extraction") == 1


def test_carry_extraction_over_noop_without_prior_extraction() -> None:
    prior = '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><a>only a tree</a></dg:chunk>'
    fresh = '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><b>new tree</b></dg:chunk>'
    assert carry_extraction_over(prior, fresh) == fresh
