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
from dgml_core.errors import SchemaInvalid
from dgml_core.extraction_schema import (
    json_schema_to_rnc,
    parse_rnc,
    rnc_to_json_schema,
    validate_rnc,
    vocabulary_to_rnc,
)

# A grounded_field JSON Schema exercising every shape: leaf, container,
# collection, and the description/example metadata.
_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "definitions": {"grounded_field": {"type": "object"}},
    "properties": {
        "vendor_name": {
            "$ref": "#/definitions/grounded_field",
            "description": "Legal name of the vendor",
        },
        "liability_cap": {
            "$ref": "#/definitions/grounded_field",
            "example": "$500,000",
        },
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


def _rnc() -> str:
    return json_schema_to_rnc(_SCHEMA, workspace="acme-corp", docset_name="Master Services")


def test_json_to_rnc_basic_shape() -> None:
    rnc = _rnc()
    assert 'namespace docset = "http://dgml.io/acme-corp/MasterServices"' in rnc
    # snake_case property names become PascalCase docset tags (no suffix stripping)
    assert "element docset:VendorName" in rnc
    assert "element docset:LineItems" in rnc
    # collection emits a plural element holding repeated singular items
    assert "LineItem*" in rnc
    assert "element docset:LineItem" in rnc
    # doc comments carry through
    assert "## Legal name of the vendor" in rnc
    assert "## Example: $500,000" in rnc


def test_rnc_roundtrip_is_stable() -> None:
    rnc = _rnc()
    # RNC -> Vocabulary -> RNC reproduces the input byte-for-byte.
    assert vocabulary_to_rnc(parse_rnc(rnc)) == rnc


def test_rnc_to_json_schema_preserves_structure() -> None:
    js = rnc_to_json_schema(_rnc())
    props = js["properties"]
    assert "grounded_field" in js["definitions"]
    assert "computed_field" in js["definitions"]
    assert set(props) == {"VendorName", "LiabilityCap", "Indemnification", "LineItems"}
    # leaf is the grounded/computed union
    assert props["VendorName"]["anyOf"] == [
        {"$ref": "#/definitions/grounded_field"},
        {"$ref": "#/definitions/computed_field"},
    ]
    # container nests properties
    assert props["Indemnification"]["type"] == "object"
    assert "IndemnifyingParty" in props["Indemnification"]["properties"]
    # collection is an array of objects
    assert props["LineItems"]["type"] == "array"
    assert set(props["LineItems"]["items"]["properties"]) == {"ProductName", "UnitPrice"}


def test_rnc_json_rnc_roundtrip_through_converted_schema() -> None:
    rnc = _rnc()
    js = rnc_to_json_schema(rnc)
    again = json_schema_to_rnc(js, workspace="acme-corp", docset_name="Master Services")
    assert again == rnc


def test_validate_rnc_accepts_generated() -> None:
    validate_rnc(_rnc())  # does not raise


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "not an rnc schema",  # no namespace / start
        "{}",  # JSON, not RNC
        'namespace docset = "x"\n',  # namespace but no start rule
    ],
)
def test_validate_rnc_rejects_out_of_subset(bad: str) -> None:
    with pytest.raises(SchemaInvalid):
        validate_rnc(bad)


def test_json_schema_without_properties_rejected() -> None:
    with pytest.raises(SchemaInvalid):
        json_schema_to_rnc({"type": "object"}, workspace="ws", docset_name="d")


def test_attribute_in_element_body_rejected() -> None:
    """The RNC subset has no attributes in element bodies — a stray one is a
    hard failure."""
    rnc = (
        'namespace docset = "http://dgml.io/x/y#"\n\n'
        "VendorName =\n"
        "  element docset:VendorName {\n"
        '    attribute anyAttr { "true" },\n'
        "    text\n"
        "  }\n"
    )
    with pytest.raises(SchemaInvalid):
        parse_rnc(rnc)


# Spec §13 form: no `start`/`dg:chunk` rule, a single root concept, `## Prompt:`
# annotations. The parser must accept it and round-trip it byte-for-byte.
_SPEC_RNC = """\
namespace docset = "http://www.dgml.io/acme/invoices#"

## Invoice root
Invoice =
  element docset:Invoice {
    (text | VendorName | LineItems)*
  }

## Legal name of the vendor
## Example: MagicSoft, Inc.
## Prompt: Look for the company name at the top of the invoice
VendorName =
  element docset:VendorName {
    text
  }

## Collection of line items
LineItems =
  element docset:LineItems {
    LineItem*
  }

## Single line item
LineItem =
  element docset:LineItem {
    (text | ProductName)*
  }

## Product or service name
## Prompt: The description column of the line item
ProductName =
  element docset:ProductName {
    text
  }
"""


def test_spec_form_rnc_no_start_rule_round_trips() -> None:
    vocab = parse_rnc(_SPEC_RNC)
    # The single unreferenced element is the root.
    assert [t.name for t in vocab.roots] == ["Invoice"]
    # `## Prompt:` is preserved on fields and survives a byte-for-byte round-trip.
    vendor = next(t for t in vocab.roots[0].children if t.name == "VendorName")
    assert vendor.prompt == "Look for the company name at the top of the invoice"
    assert vocabulary_to_rnc(vocab) == _SPEC_RNC


def test_prompt_carried_into_json_schema() -> None:
    js = rnc_to_json_schema(_SPEC_RNC)
    vendor = js["properties"]["Invoice"]["properties"]["VendorName"]
    assert vendor["prompt"] == "Look for the company name at the top of the invoice"


_CHOICE_RNC = """\
namespace docset = "http://www.dgml.io/acme/programs#"

## Total credits — a single integer or a min/max range
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


def test_choice_and_typed_leaves_round_trip() -> None:
    vocab = parse_rnc(_CHOICE_RNC)
    assert [t.name for t in vocab.roots] == ["TotalCredits"]
    tc = vocab.roots[0]
    assert tc.kind == "choice"
    assert tc.value_type == "integer"  # the scalar alternative is xsd:integer
    assert [c.name for c in tc.children] == ["MinTotalCredits", "MaxTotalCredits"]
    assert tc.children[0].kind == "field" and tc.children[0].value_type == "integer"
    # byte-for-byte round-trip
    assert vocabulary_to_rnc(vocab) == _CHOICE_RNC
    # engine JSON models the choice as anyOf(grounded_field, computed_field, object)
    js = rnc_to_json_schema(_CHOICE_RNC)
    node = js["properties"]["TotalCredits"]
    assert "anyOf" in node
    assert node["anyOf"][0]["$ref"] == "#/definitions/grounded_field"
    assert node["anyOf"][1]["$ref"] == "#/definitions/computed_field"
    assert set(node["anyOf"][2]["properties"]) == {"MinTotalCredits", "MaxTotalCredits"}


def test_collection_of_text_leaves() -> None:
    """A list of grounded text values (spec's uniform short-item list): the
    array's items are a grounded_field, not a container of sub-fields."""
    schema = {
        "definitions": {"grounded_field": {"type": "object"}},
        "properties": {
            "learning_outcomes": {
                "type": "array",
                "items": {"$ref": "#/definitions/grounded_field"},
            }
        },
    }
    rnc = json_schema_to_rnc(schema, workspace="ws", docset_name="d")
    # plural collection + singular leaf item whose content model is bare `text`
    assert "LearningOutcomes*" not in rnc  # container isn't self-referential
    assert "LearningOutcome*" in rnc
    assert "element docset:LearningOutcome {" in rnc
    # round-trips, and the JSON projection keeps the leaf item as a leaf union
    assert vocabulary_to_rnc(parse_rnc(rnc)) == rnc
    js = rnc_to_json_schema(rnc)
    lo = js["properties"]["LearningOutcomes"]
    assert lo["type"] == "array"
    assert lo["items"]["anyOf"][0]["$ref"] == "#/definitions/grounded_field"
    assert lo["items"]["anyOf"][1]["$ref"] == "#/definitions/computed_field"
