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

import pytest
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


def test_integer_strips_thousands_separator() -> None:
    """An xsd:integer with a grouped source like '8,500' normalizes to '8500',
    not '8' (regression: the integer branch used to match \\d+ before the comma)."""
    vocab = parse_rnc(_CHOICE_RNC)
    values = {"TotalCredits": {"text": "8,500", "locations": []}}
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert '<docset:TotalCredits xsi:type="integer" dg:value="8500"' in xml
    assert dgml_xml_to_values(xml, vocab=vocab)["TotalCredits"]["value"] == "8500"


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


# ── Enum fields and model-returned normalized values ─────────────────────────

_ENUM_RNC = """\
namespace docset = "http://dgml.io/acme/utility-bills"

MeterType =
  element docset:MeterType {
    ( "electric" | "natural_gas" | "water" )
  }

TotalUsage =
  element docset:TotalUsage {
    xsd:decimal
  }

Notes =
  element docset:Notes {
    text
  }
"""


def _enum_vocab() -> object:
    return parse_rnc(_ENUM_RNC)


def test_enum_field_valid_token_becomes_dg_value() -> None:
    values = {
        "MeterType": {
            "text": "Electric Service",
            "value": "electric",
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        }
    }
    xml = standalone_extraction_doc(values, vocab=_enum_vocab())  # type: ignore[arg-type]
    assert 'dg:value="electric"' in xml
    assert ">Electric Service</docset:MeterType>" in xml
    # Enum tokens aren't XSD types — no xsi:type on the element.
    assert "xsi:type" not in xml


def test_enum_field_invalid_token_stays_text_only() -> None:
    values = {
        "MeterType": {
            "text": "Mystery Commodity",
            "value": "plasma",  # not in the enum — never guessed into dg:value
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        }
    }
    xml = standalone_extraction_doc(values, vocab=_enum_vocab())  # type: ignore[arg-type]
    assert "dg:value" not in xml
    assert ">Mystery Commodity</docset:MeterType>" in xml


def test_enum_field_missing_value_stays_text_only() -> None:
    values = {
        "MeterType": {
            "text": "Electric Service",
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        }
    }
    xml = standalone_extraction_doc(values, vocab=_enum_vocab())  # type: ignore[arg-type]
    assert "dg:value" not in xml


def test_enum_dg_value_round_trips_to_values_json() -> None:
    vocab = _enum_vocab()
    values = {
        "MeterType": {
            "text": "Electric Service",
            "value": "electric",
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        }
    }
    xml = standalone_extraction_doc(values, vocab=vocab)  # type: ignore[arg-type]
    assert dgml_xml_to_values(xml, vocab=vocab) == values  # type: ignore[arg-type]


def test_typed_field_model_value_wins_when_it_validates() -> None:
    values = {
        "TotalUsage": {
            "text": "18,808.674 kWh",
            "value": "18808.674",
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        }
    }
    xml = standalone_extraction_doc(values, vocab=_enum_vocab())  # type: ignore[arg-type]
    assert 'xsi:type="decimal"' in xml
    assert 'dg:value="18808.674"' in xml


def test_typed_field_bad_model_value_falls_back_to_text_heuristics() -> None:
    values = {
        "TotalUsage": {
            "text": "18,808.674 kWh",
            "value": "lots",  # garbage — the verbatim text still normalizes
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        }
    }
    xml = standalone_extraction_doc(values, vocab=_enum_vocab())  # type: ignore[arg-type]
    assert 'dg:value="18808.674"' in xml


def test_untyped_field_model_value_kept_without_xsi_type() -> None:
    values = {
        "Notes": {
            "text": "See rider B",
            "value": "rider-b",
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        }
    }
    xml = standalone_extraction_doc(values, vocab=_enum_vocab())  # type: ignore[arg-type]
    assert 'dg:value="rider-b"' in xml
    assert "xsi:type" not in xml


def test_computed_enum_field_uses_valid_token() -> None:
    values = {
        "Notes": {
            "text": "electric bill",
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        },
        "MeterType": {
            "text": "Electric",
            "value": "electric",
            "computed": True,
            "derived_from": ["Notes"],
        },
    }
    xml = standalone_extraction_doc(values, vocab=_enum_vocab())  # type: ignore[arg-type]
    assert 'dg:origin="computed"' in xml
    assert 'dg:value="electric"' in xml


def test_count_unnormalized_enum_values() -> None:
    from dgml_core.extraction_xml import count_unnormalized_enum_values

    vocab = _enum_vocab()
    values = {
        "MeterType": {
            "text": "Mystery",
            "value": "plasma",
            "locations": [{"page_number": 1}],
        },
        "TotalUsage": {"text": "5", "locations": [{"page_number": 1}]},
    }
    assert count_unnormalized_enum_values(values, vocab) == 1  # type: ignore[arg-type]
    values["MeterType"]["value"] = "electric"
    assert count_unnormalized_enum_values(values, vocab) == 0  # type: ignore[arg-type]


# ── Derivation recompute (report-only) ────────────────────────────────────────


def test_check_derivations_sum_match_and_mismatch() -> None:
    from dgml_core.extraction_xml import check_derivations

    def bill(total: str) -> dict[str, object]:
        return {
            "Items": [
                {"Amount": {"text": "$2.00", "locations": [{"page_number": 1}]}},
                {"Amount": {"text": "$3.01", "locations": [{"page_number": 1}]}},
            ],
            "Total": {
                "text": f"${total}",
                "value": total,
                "computed": True,
                "derived_from": ["Items[0].Amount", "Items[1].Amount"],
            },
        }

    assert check_derivations(bill("5.01")) == (1, 0)  # exact
    assert check_derivations(bill("5.05")) == (1, 0)  # within $0.05
    assert check_derivations(bill("6.50")) == (1, 1)  # off


def test_check_derivations_count_and_passthrough_accepted() -> None:
    from dgml_core.extraction_xml import check_derivations

    values = {
        "MeterIds": [
            {"text": "1010016194", "locations": [{"page_number": 1}]},
            {"text": "2020032388", "locations": [{"page_number": 1}]},
        ],
        # A count derivation: value equals len(inputs), not their sum.
        "NMeters": {
            "text": "2",
            "value": "2",
            "computed": True,
            "derived_from": ["MeterIds[0]", "MeterIds[1]"],
        },
        # A passthrough/max-style derivation: value equals one input.
        "LargestCharge": {
            "text": "$9.00",
            "value": "9.00",
            "computed": True,
            "derived_from": ["Charges[0]", "Charges[1]"],
        },
        "Charges": [
            {"text": "$4.00", "locations": [{"page_number": 1}]},
            {"text": "$9.00", "locations": [{"page_number": 1}]},
        ],
    }
    assert check_derivations(values) == (2, 0)


def test_check_derivations_skips_non_numeric_and_dangling() -> None:
    from dgml_core.extraction_xml import check_derivations

    values = {
        "UtilityProvider": {"text": "PECO Energy", "locations": [{"page_number": 1}]},
        # Inferred currency: input not numeric → skipped, not counted.
        "Currency": {
            "text": "USD",
            "value": "USD",
            "computed": True,
            "derived_from": ["UtilityProvider"],
        },
        # Dangling ref → skipped.
        "Total": {
            "text": "$5",
            "value": "5",
            "computed": True,
            "derived_from": ["Missing.Path"],
        },
    }
    assert check_derivations(values) == (0, 0)


# ── Schema-declared invariants (## Invariant:) ───────────────────────────────

_INVARIANT_RNC = """\
namespace docset = "http://dgml.io/acme/bills"

## Number of distinct meters
## Invariant: count(MeterReadings)
NMeters =
  element docset:NMeters {
    xsd:integer
  }

## Invariant: sum(MeterReadings[].TotalCost)
TotalNewCharges =
  element docset:TotalNewCharges {
    xsd:decimal
  }

MeterReadings =
  element docset:MeterReadings {
    MeterReading*
  }

MeterReading =
  element docset:MeterReading {
    (text | TotalCost)*
  }

TotalCost =
  element docset:TotalCost {
    xsd:decimal
  }
"""


def _inv_vocab() -> object:
    return parse_rnc(_INVARIANT_RNC)


def _readings(*costs: str) -> list[dict[str, object]]:
    return [{"TotalCost": {"text": f"${c}", "value": c}} for c in costs]


def test_count_invariant_catches_disagreement() -> None:
    """The bug this exists for: a count field that disagrees with the
    collection it counts, where the derivation itself is self-consistent."""
    from dgml_core.extraction_xml import check_invariants

    vocab = _inv_vocab()
    ok = {"NMeters": {"text": "2", "value": "2"}, "MeterReadings": _readings("1", "2")}
    assert check_invariants(ok, vocab) == (1, [])  # type: ignore[arg-type]

    bad = {"NMeters": {"text": "1", "value": "1"}, "MeterReadings": _readings("1", "2")}
    checked, violations = check_invariants(bad, vocab)  # type: ignore[arg-type]
    assert checked == 1
    assert len(violations) == 1
    assert "NMeters" in violations[0] and "count(MeterReadings)" in violations[0]


def test_sum_invariant_uses_money_tolerance() -> None:
    from dgml_core.extraction_xml import check_invariants

    vocab = _inv_vocab()
    for total, expect_violation in (("3.00", False), ("3.05", False), ("3.50", True)):
        values = {
            "TotalNewCharges": {"text": f"${total}", "value": total},
            "MeterReadings": _readings("1.00", "2.00"),
        }
        _, violations = check_invariants(values, vocab)  # type: ignore[arg-type]
        assert bool(violations) is expect_violation, (total, violations)


def test_invariants_skip_absent_and_non_numeric_fields() -> None:
    """Every field is nullable, so 'not extracted' must never read as a
    violation — and a non-numeric value has nothing to compare."""
    from dgml_core.extraction_xml import check_invariants

    vocab = _inv_vocab()
    # Field absent entirely.
    assert check_invariants({"MeterReadings": _readings("1")}, vocab) == (0, [])  # type: ignore[arg-type]
    # Collection absent.
    assert check_invariants({"NMeters": {"text": "1", "value": "1"}}, vocab) == (0, [])  # type: ignore[arg-type]
    # Field present but non-numeric.
    values = {"NMeters": {"text": "several"}, "MeterReadings": _readings("1")}
    assert check_invariants(values, vocab) == (0, [])  # type: ignore[arg-type]


def test_unsupported_invariant_form_is_a_schema_error() -> None:
    from dgml_core.errors import SchemaInvalid

    bad = _INVARIANT_RNC.replace("## Invariant: count(MeterReadings)", "## Invariant: nMeters > 0")
    with pytest.raises(SchemaInvalid, match="Invariant"):
        parse_rnc(bad)


_INVOICE_INVARIANT_RNC = """\
namespace docset = "http://www.dgml.io/acme/invoices#"

## Total invoice amount
## Invariant: sum(LineItems[].Amount)
InvoiceTotal =
  element docset:InvoiceTotal {
    xsd:decimal
  }

LineItems =
  element docset:LineItems {
    LineItem*
  }

LineItem =
  element docset:LineItem {
    (text | Amount)*
  }

Amount =
  element docset:Amount {
    xsd:decimal
  }
"""


def test_sum_invariant_covers_the_spec_invoice_case() -> None:
    """The DGML spec's own §13 example — an invoice total composed of its line
    items — is a single root-level collection, which the sum form expresses."""
    from dgml_core.extraction_xml import check_invariants

    vocab = parse_rnc(_INVOICE_INVARIANT_RNC)
    items = [
        {"Amount": {"text": "$149.85", "value": "149.85"}},
        {"Amount": {"text": "$200.00", "value": "200.00"}},
    ]
    ok = {"InvoiceTotal": {"text": "$349.85", "value": "349.85"}, "LineItems": items}
    assert check_invariants(ok, vocab) == (1, [])

    bad = {"InvoiceTotal": {"text": "$349.85", "value": "500.00"}, "LineItems": items}
    _, violations = check_invariants(bad, vocab)
    assert len(violations) == 1 and "sum(LineItems[].Amount)" in violations[0]


def test_invariant_cannot_reach_a_sibling_collection_inside_an_entry() -> None:
    """Documented limit: paths resolve from the root through dict hops, so a
    field inside a collection entry cannot reference a collection alongside it.
    Such an invariant is skipped (not counted, not violated) rather than
    resolving to an arbitrary entry."""
    from dgml_core.extraction_xml import check_invariants

    rnc = _INVOICE_INVARIANT_RNC.replace(
        "## Invariant: sum(LineItems[].Amount)",
        "## Invariant: sum(LineItems.Nested[].Amount)",
    )
    vocab = parse_rnc(rnc)
    values = {
        "InvoiceTotal": {"text": "$1", "value": "1"},
        "LineItems": [{"Amount": {"text": "$1", "value": "1"}}],
    }
    assert check_invariants(values, vocab) == (0, [])
