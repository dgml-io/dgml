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

from typing import Any

import pytest
from dgml_core.extraction_schema import Tag, Vocabulary, json_schema_to_rnc, parse_rnc
from dgml_core.extraction_xml import (
    carry_extraction_over,
    count_dropped_inputs,
    dgml_xml_to_values,
    drop_unboxed_values,
    embed_extraction_into,
    has_extraction,
    provenance_issues,
    standalone_extraction_doc,
    verify_input_citations,
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


def _string_id_vocab() -> Vocabulary:
    """A vocab with one explicitly string-typed id field and one untyped field."""
    return Vocabulary(
        namespace_uri="http://acme.example/ns#",
        roots=[
            Tag(name="PremiseId", kind="field", value_type="string"),
            Tag(name="Note", kind="field"),  # untyped → heuristic detection
        ],
    )


def test_string_type_emits_verbatim_id_without_coercion() -> None:
    """An explicitly string-typed field emits its text verbatim — no xsi:type, no
    dg:value — so an all-digit id keeps its leading zero instead of being coerced
    to an integer. An untyped sibling with the same text still detects as integer."""
    vocab = _string_id_vocab()
    values = {
        "PremiseId": {
            "text": "0819510000",
            "locations": [{"page_number": 1, "bounding_box": [280, 898, 488, 935]}],
        },
        "Note": {"text": "0819510000", "locations": []},
    }
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert '<docset:PremiseId dg:origin="1 280 898 488 935">0819510000</docset:PremiseId>' in xml
    assert "xsi:type" not in xml.split("<docset:Note")[0]  # PremiseId carries no type
    # Contrast: the untyped sibling still heuristically types as integer — the
    # semantic the string type exists to avoid on ids.
    assert '<docset:Note xsi:type="integer" dg:value="0819510000"' in xml
    # Round-trips as plain text (no value projection for the string id).
    back = dgml_xml_to_values(xml, vocab=vocab)
    assert back["PremiseId"]["text"] == "0819510000"
    assert "value" not in back["PremiseId"]


def test_string_type_unprinted_keeps_value_without_type() -> None:
    """An unprinted string-typed field keeps its mandatory dg:value (verbatim)
    but still carries no xsi:type."""
    vocab = _string_id_vocab()
    values = {
        "PremiseId": {
            "text": "0819510000",
            "value": "0819510000",
            "method": "defaulted",
        },
    }
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert 'dg:method="defaulted"' in xml
    assert "dg:origin" not in xml and "dg:inputs" not in xml
    assert 'dg:value="0819510000"' in xml
    assert "xsi:type" not in xml


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


# ── values the page doesn't print: inputs + method (DGML 0.71) ────────────────

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
                "inputs": [
                    "Invoice.LineItems[0].Quantity",
                    "Invoice.LineItems[0].UnitPrice",
                    "Invoice.LineItems[1].Quantity",
                    "Invoice.LineItems[1].UnitPrice",
                ],
                "method": "calculated",
            },
        }
    }


def test_calculated_field_emits_inputs_and_method() -> None:
    """A calculated leaf carries a mandatory dg:value, dg:inputs naming the
    elements it used and dg:method; each input element gains an xml:id
    derived from its path. No dg:origin — the page doesn't print it — and
    none of the retired 0.7 attributes."""
    vocab = parse_rnc(_INVOICE_RNC)
    xml = standalone_extraction_doc(_invoice_values(), vocab=vocab)
    assert 'dg:method="calculated"' in xml
    assert (
        'dg:inputs="#invoice-line-items-0-quantity; #invoice-line-items-0-unit-price; '
        '#invoice-line-items-1-quantity; #invoice-line-items-1-unit-price"' in xml
    )
    assert "computed" not in xml
    assert "dg:itemprop" not in xml and "dg:href" not in xml
    assert "<docset:InvoiceTotal dg:origin" not in xml
    # Schema-declared xsd:decimal + the model's canonical value.
    assert 'xsi:type="decimal" dg:value="349.85"' in xml
    assert ">$349.85</docset:InvoiceTotal>" in xml
    # Sources carry the referenced ids; unreferenced elements carry none.
    assert 'xml:id="invoice-line-items-0-quantity"' in xml
    assert 'xml:id="invoice-line-items-1-unit-price"' in xml
    assert "VendorName xml:id" not in xml


def test_calculated_field_roundtrip() -> None:
    """XML → values reconstructs the calculated leaf, mapping #id inputs back
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


def test_dangling_inputs_dropped() -> None:
    """Malformed or dangling inputs entries are dropped; when none survive, a
    calculated value can't keep its method (calculated requires inputs), so
    the element is left with only its dg:value — an unsupported value the
    consistency check reports."""
    vocab = parse_rnc(_INVOICE_RNC)
    values = _invoice_values()
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    invoice["InvoiceTotal"]["inputs"] = [
        "Invoice.LineItems[9].Quantity",  # index out of range
        "Invoice.NoSuchField",  # unknown key
        "not a [valid path",  # unparseable
    ]
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert 'dg:value="349.85"' in xml
    assert "dg:inputs" not in xml
    assert "dg:method" not in xml
    assert "xml:id" not in xml


def test_partial_dangling_inputs_keep_resolvable_refs() -> None:
    vocab = parse_rnc(_INVOICE_RNC)
    values = _invoice_values()
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    invoice["InvoiceTotal"]["inputs"] = [
        "Invoice.LineItems[0].Quantity",
        "Invoice.Bogus",
    ]
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert 'dg:inputs="#invoice-line-items-0-quantity" dg:method="calculated"' in xml


def test_unprinted_without_canonical_value_falls_back_to_text() -> None:
    """No model-provided ``value``: dg:value comes from the schema-typed
    normalization of the display text (an unprinted value always carries
    dg:value)."""
    vocab = parse_rnc(_INVOICE_RNC)
    values = _invoice_values()
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    del invoice["InvoiceTotal"]["value"]
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert 'xsi:type="decimal" dg:value="349.85"' in xml  # normalized from "$349.85"


def test_count_dropped_inputs() -> None:
    values = _invoice_values()
    assert count_dropped_inputs(values) == 0
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    invoice["InvoiceTotal"]["inputs"] = [
        "Invoice.LineItems[0].Quantity",  # resolves
        "Invoice.LineItems[9].Quantity",  # dangles
        "not a [valid path",  # malformed
        42,  # non-string
    ]
    assert count_dropped_inputs(values) == 3


def test_inputs_into_a_leafs_own_keys_are_dropped() -> None:
    """``Total.value`` or ``X.evidence[0]`` resolve in the values tree but name
    no element, so they'd be stamped as a dg:inputs target no xml:id answers;
    they are dropped (and counted) instead."""
    vocab = parse_rnc(_INVOICE_RNC)
    values = _invoice_values()
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    invoice["InvoiceTotal"]["inputs"] = [
        "Invoice.LineItems[0].Quantity",  # resolves
        "Invoice.LineItems[0].Quantity.value",  # inside a leaf
        "Invoice.LineItems[0].Quantity.locations[0]",  # inside a leaf
    ]
    assert count_dropped_inputs(values) == 2
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert provenance_issues(xml) == []


def test_bare_value_is_not_flagged_by_the_xml_check() -> None:
    """A calculated value whose inputs all dangle is written bare by the
    serializer (no dg:method — calculated requires inputs). In the XML that
    looks the same as any other bare value, so provenance_issues leaves it
    alone; extraction drops such values before writing (drop_unboxed_values)."""
    vocab = parse_rnc(_INVOICE_RNC)
    values = _invoice_values()
    invoice = values["Invoice"]
    assert isinstance(invoice, dict)
    invoice["InvoiceTotal"]["inputs"] = ["Invoice.Bogus"]  # all inputs dangle
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert provenance_issues(xml) == []
    result = drop_unboxed_values(values)
    assert result.unsupported == 1 and "InvoiceTotal" not in invoice


def _ext(body: str) -> str:
    return (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" '
        'xmlns:docset="http://www.dgml.io/acme/invoices#">'
        f"<dg:extraction>{body}</dg:extraction></dg:chunk>"
    )


@pytest.mark.parametrize(
    ("element", "detail"),
    [
        (
            '<docset:A dg:origin="1 1 1 2 2" dg:method="inferred" dg:value="x">x</docset:A>',
            "dg:method on an element with dg:origin",
        ),
        (
            '<docset:A xml:id="b">b</docset:A>'
            '<docset:A dg:method="defaulted" dg:inputs="#b" dg:value="x">x</docset:A>',
            "defaulted value with dg:inputs",
        ),
        (
            '<docset:A dg:method="calculated" dg:value="x">x</docset:A>',
            "calculated value with no dg:inputs",
        ),
        ('<docset:A dg:method="guessed" dg:value="x">x</docset:A>', "unknown dg:method 'guessed'"),
    ],
)
def test_provenance_issues_flags_misplaced_method(element: str, detail: str) -> None:
    found = provenance_issues(_ext(element))
    assert ("method_misplaced", "A", detail) in found


def test_provenance_issues_flags_dangling_and_cyclic_inputs() -> None:
    xml = _ext(
        '<docset:A xml:id="a" dg:value="1" dg:inputs="#b" dg:method="calculated">1</docset:A>'
        '<docset:B xml:id="b" dg:value="2" dg:inputs="#a" dg:method="calculated">2</docset:B>'
        '<docset:C xml:id="c" dg:value="3" dg:inputs="#c" dg:method="inferred">3</docset:C>'
        '<docset:D dg:value="4" dg:inputs="#gone; f1#elsewhere" dg:method="inferred">4</docset:D>'
    )
    found = provenance_issues(xml)
    assert ("inputs_dangling", "D", "#gone") in found
    assert not any(detail == "f1#elsewhere" for _, _, detail in found)  # cross-file: unchecked
    cycles = {tag for kind, tag, _ in found if kind == "inputs_cycle"}
    assert cycles == {"A", "B", "C"}


def test_provenance_issues_reads_legacy_computed() -> None:
    """A 0.7 computed element with sources checks cleanly; one without reads
    as unsupported (it was computed_field_unattributed before)."""
    ok = _ext(
        '<docset:P xml:id="p1" dg:origin="1 1 1 2 2">5</docset:P>'
        '<docset:Total dg:origin="computed" dg:value="10" '
        'dg:itemprop="computedFrom" dg:href="#p1">10</docset:Total>'
    )
    assert provenance_issues(ok) == []
    bare = _ext('<docset:Total dg:origin="computed" dg:value="10">10</docset:Total>')
    assert [k for k, _, _ in provenance_issues(bare)] == ["value_unsupported"]


def test_legacy_computed_crossfile_href_reads_as_inputs() -> None:
    """A 0.7 dg:origin="computed" element reads in its 0.71 form: no boxes,
    its dg:href targets as inputs. A target outside this file (fileid#id)
    can't be mapped to a values path, so it survives as the raw reference."""
    xml = _ext(
        '<docset:InvoiceTotal dg:origin="computed" dg:value="10" '
        'dg:itemprop="computedFrom" dg:href="5kqt9r5fowno#notice-1; #unknown-local">'
        "$10</docset:InvoiceTotal>"
    )
    back = dgml_xml_to_values(xml)
    total = back["InvoiceTotal"]
    assert total["inputs"] == ["5kqt9r5fowno#notice-1", "#unknown-local"]
    assert total["method"] == "inferred"  # no dg:method stated: the safe reading
    assert "locations" not in total and "computed" not in total


def test_partly_printed_value_keeps_boxes_and_inputs() -> None:
    """The lease "3/1" case: one element holds the printed text's boxes, the
    completed date as dg:value, and dg:inputs naming what supplied the year.
    No dg:method — the boxes already say it was read."""
    vocab = Vocabulary(
        namespace_uri="http://www.dgml.io/acme/leases#",
        roots=[
            Tag(name="CommencementDate", kind="field", value_type="date"),
            Tag(name="PeriodStart", kind="field", value_type="date"),
        ],
    )
    values = {
        "CommencementDate": {
            "text": "March 1, 2026",
            "value": "2026-03-01",
            "locations": [{"page_number": 1, "bounding_box": [100, 200, 300, 230]}],
        },
        "PeriodStart": {
            "text": "3/1",
            "value": "2026-03-01",
            "locations": [{"page_number": 2, "bounding_box": [410, 980, 520, 1025]}],
            "inputs": ["CommencementDate"],
        },
    }
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert (
        '<docset:PeriodStart xsi:type="date" dg:value="2026-03-01" '
        'dg:origin="2 410 980 520 1025" dg:inputs="#commencement-date">3/1</docset:PeriodStart>'
    ) in xml
    assert 'xml:id="commencement-date"' in xml
    assert "dg:method" not in xml
    assert provenance_issues(xml) == []
    back = dgml_xml_to_values(xml, vocab=vocab)
    assert back["PeriodStart"]["inputs"] == ["CommencementDate"]
    assert back["PeriodStart"]["locations"] == values["PeriodStart"]["locations"]
    assert "method" not in back["PeriodStart"]


def test_currency_symbol_boxed_twice_with_an_input() -> None:
    """Several boxes mean the same text printed more than once."""
    vocab = Vocabulary(
        namespace_uri="http://www.dgml.io/acme/leases#",
        roots=[Tag(name="PremisesAddress", kind="field"), Tag(name="Currency", kind="field")],
    )
    values = {
        "PremisesAddress": {
            "text": "100 King St W, Toronto",
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        },
        "Currency": {
            "text": "$",
            "value": "CAD",
            "locations": [
                {"page_number": 2, "bounding_box": [1510, 880, 1528, 915]},
                {"page_number": 2, "bounding_box": [1510, 1240, 1528, 1275]},
            ],
            "inputs": ["PremisesAddress"],
        },
    }
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert 'dg:origin="2 1510 880 1528 915; 2 1510 1240 1528 1275"' in xml
    assert 'dg:inputs="#premises-address"' in xml


def test_defaulted_value_never_carries_inputs() -> None:
    vocab = Vocabulary(
        namespace_uri="http://www.dgml.io/acme/leases#",
        roots=[
            Tag(name="Notes", kind="field"),
            Tag(name="LateFeeGraceDays", kind="field", value_type="integer"),
        ],
    )
    values = {
        "Notes": {"text": "n", "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}]},
        "LateFeeGraceDays": {
            "text": "0",
            "value": "0",
            "method": "defaulted",
            "inputs": ["Notes"],  # a sloppy producer: defaulted forbids inputs
        },
    }
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert '<docset:LateFeeGraceDays xsi:type="integer" dg:value="0" dg:method="defaulted">' in xml
    assert "dg:inputs" not in xml
    back = dgml_xml_to_values(xml, vocab=vocab)
    assert back["LateFeeGraceDays"] == {"text": "0", "value": "0", "method": "defaulted"}


# ── evidence: inputs that aren't schema fields become dg:chunk (rule 7) ──────


# ── input citations ──────────────────────────────────────────────────────────


def _bill() -> dict[str, Any]:
    return {
        "Items": [
            {"Name": {"text": "Customer Charge"}, "Amount": {"text": "$12.75", "value": "12.75"}},
            {"Name": {"text": "Gas Delivery"}, "Amount": {"text": "$318.40"}},
            {"Name": {"text": "Gas Delivery"}, "Amount": {"text": "$27.05"}},
            {"Name": {"text": "City Utility Tax"}, "Amount": {"text": "$58.20"}},
        ],
        "Tax": {"text": "58.20", "method": "calculated", "inputs": []},
    }


def _verify(inputs: list[Any]) -> tuple[Any, tuple[int, int]]:
    values = _bill()
    values["Tax"]["inputs"] = inputs
    result = verify_input_citations(values)
    return values["Tax"].get("inputs"), (result.repaired, result.dropped)


def test_citation_that_matches_is_kept_bare() -> None:
    """Money formatting doesn't matter: "58.20" cites "$58.20"."""
    assert _verify(["Items[3].Amount=58.20"]) == (["Items[3].Amount"], (0, 0))


def test_citation_matches_the_normalized_value_too() -> None:
    assert _verify(["Items[0].Amount=12.75"]) == (["Items[0].Amount"], (0, 0))


def test_numeric_citation_matches_by_value() -> None:
    """A leading-zero or thousands-separator difference is the same number."""
    values = _bill()
    values["Items"][3]["Amount"] = {"text": "$0.45", "value": "0.45"}
    values["Tax"]["inputs"] = ["Items[3].Amount=.45"]
    assert verify_input_citations(values).dropped == 0
    assert values["Tax"]["inputs"] == ["Items[3].Amount"]


def test_punctuation_only_citation_compares_bare_text() -> None:
    """A currency printed only as "$": the citation normalizes to nothing,
    so it is compared as written rather than dropped."""
    values: dict[str, Any] = {
        "Currency": {"text": "$", "value": "USD", "method": "inferred", "inputs": ["Hq"]},
        "MeterCurrency": {"text": "$", "method": "inferred", "inputs": ["Currency=$"]},
        "Hq": {"text": "Springfield"},
    }
    assert verify_input_citations(values).dropped == 0
    assert values["MeterCurrency"]["inputs"] == ["Currency"]
    values["MeterCurrency"]["inputs"] = ["Currency=€"]
    assert verify_input_citations(values).dropped == 1


def test_miscounted_index_is_repaired_to_the_one_matching_sibling() -> None:
    assert _verify(["Items[2].Amount=$58.20"]) == (["Items[3].Amount"], (1, 0))


def test_out_of_range_index_is_repaired_too() -> None:
    assert _verify(["Items[9].Amount=58.20"]) == (["Items[3].Amount"], (1, 0))


def test_ambiguous_citation_is_dropped_not_guessed() -> None:
    """Two rows print "Gas Delivery"; a wrong index citing that name
    can't be re-pointed with confidence."""
    assert _verify(["Items[0].Name=Gas Delivery"]) == (None, (0, 1))


def test_citation_matching_nothing_is_dropped() -> None:
    inputs, counts = _verify(["Items[3].Amount=99.99", "Items[3].Amount=58.20"])
    assert inputs == ["Items[3].Amount"] and counts == (0, 1)


def test_uncited_and_non_leaf_inputs_are_kept() -> None:
    """No citation: nothing to check. A whole row can't be compared to text,
    so it is kept with the citation stripped."""
    assert _verify(["Items[3].Amount", "Items[3]=City Utility Tax"]) == (
        ["Items[3].Amount", "Items[3]"],
        (0, 0),
    )


def _meter() -> dict[str, Any]:
    """A bill's shape: line items and their summary nested in a meter reading."""
    bill = _bill()
    item = bill["Items"][3]
    item["Category"] = {"text": "tax", "method": "inferred", "inputs": ["Name=City Utility Tax"]}
    return {"Meters": [bill]}


def test_path_relative_to_the_value_is_anchored_by_its_citation() -> None:
    """The model wrote "Items[3].Amount" from inside Meters[0] — a path that
    resolves nowhere from the root. The citation finds it under the value's
    own meter reading, and it is rewritten as a root path."""
    values = _meter()
    values["Meters"][0]["Tax"]["inputs"] = ["Items[3].Amount=58.20"]
    result = verify_input_citations(values)
    assert values["Meters"][0]["Tax"]["inputs"] == ["Meters[0].Items[3].Amount"]
    assert values["Meters"][0]["Items"][3]["Category"]["inputs"] == ["Meters[0].Items[3].Name"]
    assert (result.repaired, result.dropped) == (2, 0)


def test_relative_path_with_a_miscounted_index_is_repaired() -> None:
    values = _meter()
    values["Meters"][0]["Tax"]["inputs"] = ["Items[2].Amount=58.20"]
    verify_input_citations(values)
    assert values["Meters"][0]["Tax"]["inputs"] == ["Meters[0].Items[3].Amount"]


def test_relative_path_whose_citation_matches_nothing_is_dropped() -> None:
    values = _meter()
    values["Meters"][0]["Tax"]["inputs"] = ["Items[3].Amount=99.99"]
    result = verify_input_citations(values)
    assert "inputs" not in values["Meters"][0]["Tax"]
    assert result.dropped == 1


def test_uncited_relative_path_is_anchored_too() -> None:
    values = _meter()
    values["Meters"][0]["Tax"]["inputs"] = ["Items[3].Amount"]
    assert verify_input_citations(values).repaired == 2  # Tax's input and Category's
    assert values["Meters"][0]["Tax"]["inputs"] == ["Meters[0].Items[3].Amount"]


# ── printed values grounding couldn't box ─────────────────────────────────────

_BOX = [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}]
_NO_BOX = [{"page_number": 1}]


def test_unboxed_printed_value_is_dropped_and_later_inputs_reindexed() -> None:
    values: dict[str, Any] = {
        "Items": [
            {"Amount": {"text": "1.00", "locations": _BOX}},
            {"Amount": {"text": "2.00", "locations": _NO_BOX}},  # row empties: gone
            {"Amount": {"text": "3.00", "locations": _BOX}},
        ],
        "Total": {
            "text": "4.00",
            "method": "calculated",
            "inputs": ["Items[0].Amount", "Items[1].Amount", "Items[2].Amount"],
        },
    }
    result = drop_unboxed_values(values)
    assert (result.values, result.inputs) == (1, 1)
    assert result.paths == ["Items[1].Amount=2.00"]
    assert [i["Amount"]["text"] for i in values["Items"]] == ["1.00", "3.00"]
    # Items[2] is now Items[1]; the input naming the dropped row is gone.
    assert values["Total"]["inputs"] == ["Items[0].Amount", "Items[1].Amount"]


def test_unboxed_partly_printed_value_is_dropped_not_read_as_inferred() -> None:
    values: dict[str, Any] = {
        "CommencementDate": {"text": "March 1, 2026", "locations": _BOX},
        "PeriodStart": {
            "text": "3/1",
            "value": "2026-03-01",
            "locations": _NO_BOX,
            "inputs": ["CommencementDate"],
        },
    }
    assert drop_unboxed_values(values).values == 1
    assert "PeriodStart" not in values


def test_unprinted_and_partly_boxed_values_are_kept() -> None:
    values: dict[str, Any] = {
        "Name": {"text": "Acme", "locations": _BOX},
        # One of two printed spots boxed: still tied to the page.
        "Currency": {"text": "$", "locations": _BOX + _NO_BOX, "inputs": ["Name"]},
        "Region": {"text": "EMEA", "method": "inferred", "inputs": ["Name"]},
        "Grace": {"text": "0", "method": "defaulted"},
    }
    before = {k: dict(v) for k, v in values.items()}
    assert drop_unboxed_values(values) == (0, 0, [], 0)
    assert values == before


def test_value_whose_inputs_were_all_dropped_is_dropped_too() -> None:
    """Dropping Hq leaves Region inferred from nothing, so it goes as well."""
    values: dict[str, Any] = {
        "Hq": {"text": "Berlin", "locations": _NO_BOX},
        "Region": {"text": "EMEA", "method": "inferred", "inputs": ["Hq"]},
    }
    result = drop_unboxed_values(values)
    assert (result.values, result.unsupported, result.inputs) == (1, 1, 1)
    assert result.paths == ["Hq=Berlin", "Region=EMEA (unsupported)"]
    assert values == {}


def test_value_with_no_box_and_no_method_is_dropped() -> None:
    """The model returned it bare, or claimed a calculation naming nothing."""
    values: dict[str, Any] = {
        "Name": {"text": "Acme", "locations": _BOX},
        "Bare": {"text": "ACME"},
        "Months": {"text": "12", "method": "calculated", "inputs": ["Nowhere"]},
        "Grace": {"text": "0", "method": "defaulted"},
    }
    result = drop_unboxed_values(values)
    assert (result.values, result.unsupported) == (0, 2)
    assert set(values) == {"Name", "Grace"}


def _end_date_vocab() -> Vocabulary:
    return Vocabulary(
        namespace_uri="http://www.dgml.io/acme/contracts#",
        roots=[
            Tag(name="Party", kind="field"),
            Tag(name="EndDate", kind="field", value_type="date"),
        ],
    )


def _end_date_values() -> dict[str, object]:
    return {
        "Party": {
            "text": "Acme",
            "locations": [{"page_number": 1, "bounding_box": [10, 10, 90, 40]}],
        },
        "EndDate": {
            "text": "2028-01-14",
            "value": "2028-01-14",
            "method": "calculated",
            "evidence": [
                {
                    "text": "January 15, 2026",
                    "locations": [{"page_number": 1, "bounding_box": [300, 410, 620, 450]}],
                },
                {
                    "text": "a term of twenty-four (24) months",
                    "locations": [{"page_number": 1, "bounding_box": [300, 520, 980, 560]}],
                },
            ],
        },
    }


def test_evidence_emits_anonymous_chunks_named_in_inputs() -> None:
    """The spec's EndDate example: a calculated value whose inputs are
    printed text the schema doesn't extract. Each becomes a dg:chunk with an
    xml:id, its boxes and its text, ahead of the field that names it."""
    xml = standalone_extraction_doc(_end_date_values(), vocab=_end_date_vocab())
    assert (
        '<dg:chunk xml:id="end-date--in-0" dg:origin="1 300 410 620 450">'
        "January 15, 2026</dg:chunk>"
    ) in xml
    assert (
        '<dg:chunk xml:id="end-date--in-1" dg:origin="1 300 520 980 560">'
        "a term of twenty-four (24) months</dg:chunk>"
    ) in xml
    assert (
        '<docset:EndDate xsi:type="date" dg:value="2028-01-14" '
        'dg:inputs="#end-date--in-0; #end-date--in-1" dg:method="calculated">'
    ) in xml
    assert xml.index("end-date--in-1") < xml.index("<docset:EndDate")
    assert provenance_issues(xml) == []


def test_evidence_round_trips() -> None:
    vocab = _end_date_vocab()
    values = _end_date_values()
    back = dgml_xml_to_values(standalone_extraction_doc(values, vocab=vocab), vocab=vocab)
    assert back == values
    # Inferred from the XML alone, too: the chunks are not mistaken for fields.
    assert dgml_xml_to_values(standalone_extraction_doc(values, vocab=vocab)) == back


def test_evidence_mixes_with_field_inputs() -> None:
    vocab = _end_date_vocab()
    values = _end_date_values()
    end = values["EndDate"]
    assert isinstance(end, dict)
    end["inputs"] = ["Party"]
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert 'dg:inputs="#party; #end-date--in-0; #end-date--in-1"' in xml
    back = dgml_xml_to_values(xml, vocab=vocab)
    assert back["EndDate"]["inputs"] == ["Party"]
    assert [e["text"] for e in back["EndDate"]["evidence"]] == [
        "January 15, 2026",
        "a term of twenty-four (24) months",
    ]


def test_evidence_without_a_box_is_dropped_and_counted() -> None:
    """Evidence phases 2/3 couldn't box has nothing to crop: no chunk. With
    no evidence left the value can't keep its method."""
    vocab = _end_date_vocab()
    values = _end_date_values()
    end = values["EndDate"]
    assert isinstance(end, dict)
    for item in end["evidence"]:
        item["locations"] = [{"page_number": 1}]
    assert count_dropped_inputs(values) == 2
    xml = standalone_extraction_doc(values, vocab=vocab)
    assert "<dg:chunk xml:id" not in xml
    assert "dg:inputs" not in xml and "dg:method" not in xml


def test_structural_chunks_are_not_read_as_evidence() -> None:
    """A document tree's own dg:chunks (unnamed scaffolding) stay out of the
    projection's evidence; only chunks named in dg:inputs are evidence."""
    vocab = _end_date_vocab()
    core = (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">'
        '<dg:chunk xml:id="c1" dg:origin="1 1 1 2 2">a heading</dg:chunk>'
        "</dg:chunk>"
    )
    xml = embed_extraction_into(core, _end_date_values(), vocab=vocab)
    back = dgml_xml_to_values(xml, vocab=vocab)
    assert back == _end_date_values()


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


def test_inferred_enum_field_uses_valid_token() -> None:
    values = {
        "Notes": {
            "text": "electric bill",
            "locations": [{"page_number": 1, "bounding_box": [1, 2, 3, 4]}],
        },
        "MeterType": {
            "text": "Electric",
            "value": "electric",
            "inputs": ["Notes"],
            "method": "inferred",
        },
    }
    xml = standalone_extraction_doc(values, vocab=_enum_vocab())  # type: ignore[arg-type]
    assert 'dg:inputs="#notes" dg:method="inferred"' in xml
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
                "inputs": ["Items[0].Amount", "Items[1].Amount"],
                "method": "calculated",
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
            "inputs": ["MeterIds[0]", "MeterIds[1]"],
            "method": "calculated",
        },
        # A passthrough/max-style derivation: value equals one input.
        "LargestCharge": {
            "text": "$9.00",
            "value": "9.00",
            "inputs": ["Charges[0]", "Charges[1]"],
            "method": "calculated",
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
        # Inferred currency: a judgment, never recomputed.
        "Currency": {
            "text": "USD",
            "value": "USD",
            "inputs": ["UtilityProvider"],
            "method": "inferred",
        },
        # Calculated but not numeric → skipped, not counted.
        "Region": {
            "text": "EMEA",
            "value": "EMEA",
            "inputs": ["UtilityProvider"],
            "method": "calculated",
        },
        # Dangling ref → skipped.
        "Total": {
            "text": "$5",
            "value": "5",
            "inputs": ["Missing.Path"],
            "method": "calculated",
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


_SINGLE_ROOT_INVOICE_RNC = """\
namespace docset = "http://www.dgml.io/acme/invoices#"

CommercialInvoice =
  element docset:CommercialInvoice {
    (text | DocumentTotal | LineItems)*
  }

## Total of the invoice
## Invariant: sum(LineItems[].LineAmount)
DocumentTotal =
  element docset:DocumentTotal {
    xsd:decimal
  }

LineItems =
  element docset:LineItems {
    LineItem*
  }

LineItem =
  element docset:LineItem {
    (text | LineAmount)*
  }

LineAmount =
  element docset:LineAmount {
    xsd:decimal
  }
"""


def test_invariant_under_a_single_root_is_checked_unqualified() -> None:
    """The schema's one root is ``CommercialInvoice`` and the invariant says
    ``LineItems``, not ``CommercialInvoice.LineItems``. That path never
    resolved from the submission root, so every run reported
    ``invariants_checked: 0``; it is now read under the root."""
    from dgml_core.extraction_xml import check_invariants

    vocab = parse_rnc(_SINGLE_ROOT_INVOICE_RNC)
    items = [
        {"LineAmount": {"text": "$149.85", "value": "149.85"}},
        {"LineAmount": {"text": "$200.00", "value": "200.00"}},
    ]
    ok = {
        "CommercialInvoice": {
            "DocumentTotal": {"text": "$349.85", "value": "349.85"},
            "LineItems": items,
        }
    }
    assert check_invariants(ok, vocab) == (1, [])
    bad = {
        "CommercialInvoice": {
            "DocumentTotal": {"text": "$500.00", "value": "500.00"},
            "LineItems": items,
        }
    }
    checked, violations = check_invariants(bad, vocab)
    assert checked == 1
    assert violations == [
        "DocumentTotal: expected sum(LineItems[].LineAmount) = 349.85, got 500.00"
    ]


def test_input_that_resolves_to_the_citing_value_is_dropped() -> None:
    """A relative path anchored back onto the citing value would make it its
    own input — a dg:inputs cycle."""
    values: dict[str, Any] = {
        "Rows": [{"Start": {"text": "3/1", "locations": [], "inputs": ["Start=3/1"]}}]
    }
    assert verify_input_citations(values).dropped == 1
    assert "inputs" not in values["Rows"][0]["Start"]


def test_citation_numbers_are_plain_decimals() -> None:
    """An exponent or NaN spelling is text, not a number: "1e3" doesn't cite
    "1000", and a signalling NaN doesn't raise."""
    values: dict[str, Any] = {
        "A": {"text": "1000"},
        "B": {"text": "sNaN"},
        "T": {"text": "x", "method": "inferred", "inputs": ["A=1e3", "B=snan"]},
    }
    assert verify_input_citations(values).dropped == 1
    assert values["T"]["inputs"] == ["B"]


def test_legacy_computed_without_sources_reads_as_a_bare_value() -> None:
    """A 0.7 computed element with no dg:href has no boxes: it reads with no
    locations, not as a printed value with an empty box list."""
    back = dgml_xml_to_values(
        _ext('<docset:Total dg:origin="computed" dg:value="10">10</docset:Total>')
    )
    assert back["Total"] == {"text": "10", "value": "10"}
