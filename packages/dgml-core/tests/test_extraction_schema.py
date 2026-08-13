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
    field_tree_to_rnc,
    field_tree_to_vocabulary,
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
    assert "extracted_value" in js["definitions"]
    assert set(props) == {"VendorName", "LiabilityCap", "Indemnification", "LineItems"}
    # leaf is the merged extracted_value ref
    assert props["VendorName"]["$ref"] == "#/definitions/extracted_value"
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
    # engine JSON models the choice as anyOf(extracted_value, object)
    js = rnc_to_json_schema(_CHOICE_RNC)
    node = js["properties"]["TotalCredits"]
    assert "anyOf" in node
    assert node["anyOf"][0]["$ref"] == "#/definitions/extracted_value"
    assert set(node["anyOf"][1]["properties"]) == {"MinTotalCredits", "MaxTotalCredits"}


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
    # round-trips, and the JSON projection keeps the leaf item as a leaf ref
    assert vocabulary_to_rnc(parse_rnc(rnc)) == rnc
    js = rnc_to_json_schema(rnc)
    lo = js["properties"]["LearningOutcomes"]
    assert lo["type"] == "array"
    assert lo["items"]["$ref"] == "#/definitions/extracted_value"


# ── Typed field tree → RNC (the schema-generation path) ──────────────────────

# What the schema-generation LLM submits: a typed field tree with a datatype on
# every leaf, containers, and a collection (both the explicit-`item` form and
# the implicit-`fields` form).
_FIELD_TREE = [
    {"name": "vendor_name", "kind": "field", "datatype": "text"},
    {
        "name": "due_date",
        "kind": "field",
        "datatype": "date",
        "description": "When payment is due",
    },
    {
        "name": "bill_summary",
        "kind": "container",
        "fields": [
            {"name": "total_amount_due", "kind": "field", "datatype": "decimal"},
            {"name": "invoice_count", "kind": "field", "datatype": "integer"},
        ],
    },
    {
        "name": "line_items",
        "kind": "collection",
        "item": {
            "name": "line_item",
            "kind": "container",
            "fields": [
                {"name": "product_name", "kind": "field", "datatype": "text"},
                {"name": "unit_price", "kind": "field", "datatype": "decimal"},
            ],
        },
    },
]


def test_field_tree_to_rnc_carries_datatypes() -> None:
    rnc = field_tree_to_rnc(_FIELD_TREE, workspace="acme-corp", docset_name="Master Services")
    # Same namespace convention as the JSON-schema path.
    assert 'namespace docset = "http://dgml.io/acme-corp/MasterServices"' in rnc
    # Datatypes land as xsd: typed leaves; untyped stays `text`.
    assert "element docset:DueDate {\n    xsd:date" in rnc
    assert "element docset:TotalAmountDue {\n    xsd:decimal" in rnc
    assert "element docset:InvoiceCount {\n    xsd:integer" in rnc
    assert "element docset:VendorName {\n    text" in rnc
    # Container + collection expand as usual.
    assert "element docset:LineItem {" in rnc
    assert "LineItem*" in rnc
    # Doc comment preserved.
    assert "## When payment is due" in rnc


def test_field_tree_to_rnc_round_trips() -> None:
    rnc = field_tree_to_rnc(_FIELD_TREE, workspace="ws", docset_name="d")
    # Parses as valid RNC and re-serializes identically, and the datatypes
    # survive the round-trip on the parsed vocabulary.
    vocab = parse_rnc(rnc)
    assert vocabulary_to_rnc(vocab) == rnc
    by_name = {t.name: t for t in vocab.roots}
    assert by_name["DueDate"].value_type == "date"
    assert by_name["VendorName"].value_type is None
    assert [t.name for t in vocab.roots] == [
        "VendorName",
        "DueDate",
        "BillSummary",
        "LineItems",
    ]


def test_field_tree_collection_implicit_fields() -> None:
    """A collection may carry the item's fields directly (no explicit `item`);
    the singular item element is synthesized."""
    tree = [
        {
            "name": "readings",
            "kind": "collection",
            "fields": [{"name": "value", "kind": "field", "datatype": "integer"}],
        }
    ]
    rnc = field_tree_to_rnc(tree, workspace="ws", docset_name="d")
    assert "Reading*" in rnc
    assert "element docset:Reading {" in rnc
    assert "element docset:Value {\n    xsd:integer" in rnc


def test_field_tree_bad_datatype_rejected() -> None:
    tree = [{"name": "x", "kind": "field", "datatype": "money"}]
    with pytest.raises(SchemaInvalid):
        field_tree_to_vocabulary(tree, namespace_uri="urn:x")


def test_field_tree_unknown_kind_rejected() -> None:
    tree = [{"name": "x", "kind": "widget"}]
    with pytest.raises(SchemaInvalid):
        field_tree_to_vocabulary(tree, namespace_uri="urn:x")


def test_field_tree_missing_name_rejected() -> None:
    with pytest.raises(SchemaInvalid):
        field_tree_to_vocabulary([{"kind": "field"}], namespace_uri="urn:x")


def test_field_tree_empty_rejected() -> None:
    with pytest.raises(SchemaInvalid):
        field_tree_to_vocabulary([], namespace_uri="urn:x")


def test_field_tree_xsd_prefixed_datatype_accepted() -> None:
    """The model may write `xsd:date` or bare `date`; both normalize."""
    tree = [{"name": "d", "kind": "field", "datatype": "xsd:date"}]
    rnc = field_tree_to_rnc(tree, workspace="ws", docset_name="d")
    assert "element docset:D {\n    xsd:date" in rnc


# ── Enums (RNC value enumeration) ─────────────────────────────────────────────

_METER_TYPES = [
    "electric",
    "natural_gas",
    "water",
    "steam",
    "district_heating",
    "district_cooling",
    "fuel_oil",
    "propane",
    "irrigation",
    "lighting",
    "chilled_water",
    "sewer",
    "unknown",
]


def _enum_tree(values: list[str]) -> list[dict[str, object]]:
    return [
        {
            "name": "meter_type",
            "kind": "field",
            "enum": values,
            "prompt": "Classify the commodity being billed",
        }
    ]


def test_enum_field_emits_rnc_value_enumeration() -> None:
    rnc = field_tree_to_rnc(_enum_tree(["electric", "water"]), workspace="ws", docset_name="d")
    # short enums pack onto one line
    assert '( "electric" | "water" )' in rnc
    vocab = parse_rnc(rnc)
    tag = vocab.roots[0]
    assert tag.kind == "field"
    assert tag.enum_values == ["electric", "water"]
    assert tag.value_type is None


def test_enum_field_round_trips_byte_for_byte() -> None:
    for values in (["a"], ["electric", "water"], _METER_TYPES, [f"token_{i}" for i in range(27)]):
        rnc = field_tree_to_rnc(_enum_tree(values), workspace="ws", docset_name="d")
        vocab = parse_rnc(rnc)
        assert vocab.roots[0].enum_values == values
        assert vocabulary_to_rnc(vocab) == rnc


def test_long_enum_wraps_one_token_per_line() -> None:
    rnc = field_tree_to_rnc(_enum_tree(_METER_TYPES), workspace="ws", docset_name="d")
    assert '( "electric"\n      | "natural_gas"' in rnc
    assert '| "unknown" )' in rnc
    assert all(len(line) <= 100 for line in rnc.splitlines())
    assert vocabulary_to_rnc(parse_rnc(rnc)) == rnc


def test_enum_survives_json_projection_round_trip() -> None:
    rnc = field_tree_to_rnc(_enum_tree(_METER_TYPES), workspace="ws", docset_name="d")
    js = rnc_to_json_schema(rnc)
    node = js["properties"]["MeterType"]
    assert node["$ref"] == "#/definitions/extracted_value"
    assert node["value_enum"] == _METER_TYPES
    assert node["prompt"] == "Classify the commodity being billed"
    again = json_schema_to_rnc(js, workspace="ws", docset_name="d")
    assert again == rnc


def test_datatype_survives_json_projection_round_trip() -> None:
    """Typed leaves carry `datatype` through the JSON projection so
    RNC → JSON → RNC is lossless for xsd-typed fields too."""
    rnc = field_tree_to_rnc(_FIELD_TREE, workspace="ws", docset_name="d")
    js = rnc_to_json_schema(rnc)
    assert js["properties"]["DueDate"]["datatype"] == "date"
    assert "datatype" not in js["properties"]["VendorName"]
    assert json_schema_to_rnc(js, workspace="ws", docset_name="d") == rnc


def test_enum_plus_datatype_rejected() -> None:
    tree = [{"name": "x", "kind": "field", "enum": ["a"], "datatype": "date"}]
    with pytest.raises(SchemaInvalid):
        field_tree_to_vocabulary(tree, namespace_uri="urn:x")


@pytest.mark.parametrize(
    "bad",
    [
        [],  # empty
        ["a", "a"],  # duplicate
        [""],  # empty token
        ['with"quote'],  # breaks the RNC quoting
        [42],  # non-string
    ],
)
def test_bad_enum_values_rejected(bad: list[object]) -> None:
    tree = [{"name": "x", "kind": "field", "enum": bad}]
    with pytest.raises(SchemaInvalid):
        field_tree_to_vocabulary(tree, namespace_uri="urn:x")


def test_collection_of_enum_leaves_round_trips() -> None:
    tree = [
        {
            "name": "meter_types_present",
            "kind": "collection",
            "item": {"name": "meter_type_present", "kind": "field", "enum": ["electric", "gas"]},
        }
    ]
    rnc = field_tree_to_rnc(tree, workspace="ws", docset_name="d")
    assert "MeterTypePresent*" in rnc
    assert '( "electric" | "gas" )' in rnc
    vocab = parse_rnc(rnc)
    assert vocab.roots[0].item is not None
    assert vocab.roots[0].item.enum_values == ["electric", "gas"]
    assert vocabulary_to_rnc(vocab) == rnc
    # JSON projection keeps the enum on the array items and round-trips.
    js = rnc_to_json_schema(rnc)
    items = js["properties"]["MeterTypesPresent"]["items"]
    assert items["value_enum"] == ["electric", "gas"]
    assert json_schema_to_rnc(js, workspace="ws", docset_name="d") == rnc


# ── Duplicate-name definitions ────────────────────────────────────────────────


def test_identical_shared_definition_is_reused() -> None:
    """The same structure referenced from two levels emits one definition and
    round-trips."""
    line_items = {
        "name": "charge_line_items",
        "kind": "collection",
        "fields": [
            {"name": "line_item_name", "kind": "field"},
            {"name": "amount", "kind": "field", "datatype": "decimal"},
        ],
    }
    tree = [
        {"name": "meter_readings", "kind": "collection", "fields": [dict(line_items)]},
        {"name": "account_level_section", "kind": "container", "fields": [dict(line_items)]},
    ]
    rnc = field_tree_to_rnc(tree, workspace="ws", docset_name="d")
    assert rnc.count("ChargeLineItems =") == 1
    assert rnc.count("element docset:Amount {") == 1
    assert vocabulary_to_rnc(parse_rnc(rnc)) == rnc


def test_same_name_different_content_is_an_error() -> None:
    """Two same-named fields with different guidance/content must not silently
    collapse into one definition (first-wins used to drop one side's prompt)."""
    tree = [
        {
            "name": "document",
            "kind": "container",
            "fields": [
                {"name": "currency", "kind": "field", "prompt": "ISO code for the whole bill"}
            ],
        },
        {
            "name": "meter",
            "kind": "container",
            "fields": [{"name": "currency", "kind": "field", "prompt": "ISO code for this meter"}],
        },
    ]
    with pytest.raises(SchemaInvalid, match="'Currency'"):
        field_tree_to_rnc(tree, workspace="ws", docset_name="d")


# ── Standard JSON Schema dialect ingestion (root $ref, $defs, titles, leaves) ─

# A synthetic schema in the same dialect as externally-generated extraction
# schemas: root $ref, shared $defs, `title` as the DGML element
# name, and a merged {text, value, locations, derived_from} leaf shape with
# typed/enum `value` subschemas.
_DIALECT_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Utility Bill",
    "$ref": "#/$defs/BillDocument",
    "$defs": {
        "BillDocument": {
            "type": "object",
            "properties": {
                "statement_date": {
                    "title": "StatementDate",
                    "prompt": "Look for 'Statement Date' or 'Bill Date'",
                    "$ref": "#/$defs/value.date",
                },
                "meter_type": {"title": "MeterType", "$ref": "#/$defs/value.MeterType"},
                "total_cost": {"title": "TotalCost", "$ref": "#/$defs/value.Decimal"},
                "meter_readings": {
                    "title": "MeterReadings",
                    "type": "array",
                    "items": {"$ref": "#/$defs/Reading"},
                },
            },
        },
        "Reading": {
            "type": "object",
            "description": "One meter reading",
            "properties": {
                "usage": {"title": "Usage", "$ref": "#/$defs/value.Decimal"},
                "notes": {"title": "Notes", "$ref": "#/$defs/value.str"},
            },
        },
        "value.date": {
            "type": "object",
            "description": "An extracted value: shared leaf mechanics boilerplate.",
            "properties": {
                "text": {"type": "string"},
                "value": {"type": "string", "format": "date"},
                "locations": {"type": "array", "items": {"$ref": "#/$defs/loc"}},
                "derived_from": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text"],
        },
        "value.MeterType": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "value": {"type": "string", "enum": ["electric", "water"]},
                "locations": {"type": "array", "items": {"$ref": "#/$defs/loc"}},
                "derived_from": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text"],
        },
        "value.Decimal": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "value": {"type": "string", "pattern": "^-?\\d+(\\.\\d+)?$"},
                "locations": {"type": "array", "items": {"$ref": "#/$defs/loc"}},
                "derived_from": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text"],
        },
        "value.str": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "value": {"type": "string"},
                "locations": {"type": "array", "items": {"$ref": "#/$defs/loc"}},
                "derived_from": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["text"],
        },
        "loc": {
            "type": "object",
            "properties": {
                "page": {"type": "integer"},
                "bounding_box": {"type": "array", "items": {"type": "integer"}},
            },
        },
    },
}


def test_standard_dialect_schema_ingests() -> None:
    from dgml_core.extraction_schema import json_schema_to_vocabulary

    vocab = json_schema_to_vocabulary(_DIALECT_SCHEMA, namespace_uri="urn:x")
    by_name = {t.name: t for t in vocab.roots}
    assert set(by_name) == {"StatementDate", "MeterType", "TotalCost", "MeterReadings"}
    # `title` names the element; value subschemas map to types.
    assert by_name["StatementDate"].value_type == "date"
    assert by_name["StatementDate"].prompt == "Look for 'Statement Date' or 'Bill Date'"
    assert by_name["MeterType"].enum_values == ["electric", "water"]
    assert by_name["TotalCost"].value_type == "decimal"
    # Shared-def boilerplate description is NOT copied onto fields.
    assert by_name["StatementDate"].description is None
    # Nested collection through a $def resolves.
    readings = by_name["MeterReadings"]
    assert readings.kind == "collection"
    assert {c.name for c in readings.children} == {"Usage", "Notes"}
    assert next(c for c in readings.children if c.name == "Usage").value_type == "decimal"


def test_standard_dialect_schema_renders_and_round_trips() -> None:
    rnc = json_schema_to_rnc(_DIALECT_SCHEMA, workspace="ws", docset_name="d")
    assert 'element docset:MeterType {\n    ( "electric" | "water" )' in rnc
    assert "element docset:StatementDate {\n    xsd:date" in rnc
    assert vocabulary_to_rnc(parse_rnc(rnc)) == rnc


def test_dialect_recursive_ref_rejected() -> None:
    from dgml_core.extraction_schema import json_schema_to_vocabulary

    schema = {
        "$ref": "#/$defs/A",
        "$defs": {
            "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/A"}}},
        },
    }
    with pytest.raises(SchemaInvalid, match="recursive"):
        json_schema_to_vocabulary(schema, namespace_uri="urn:x")


def test_dialect_dangling_ref_rejected() -> None:
    from dgml_core.extraction_schema import json_schema_to_vocabulary

    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/Missing"}}}
    with pytest.raises(SchemaInvalid):
        json_schema_to_vocabulary(schema, namespace_uri="urn:x")


def test_root_that_is_also_referenced_gets_explicit_start_rule() -> None:
    """A top-level field whose identical definition is also nested (a
    document-level address reused per section) must stay a root across the
    RNC round-trip — via an explicit start rule."""
    tree = [
        {"name": "service_address", "kind": "field"},
        {
            "name": "meters",
            "kind": "collection",
            "fields": [{"name": "service_address", "kind": "field"}],
        },
    ]
    rnc = field_tree_to_rnc(tree, workspace="ws", docset_name="d")
    assert "start =" in rnc
    vocab = parse_rnc(rnc)
    assert [t.name for t in vocab.roots] == ["ServiceAddress", "Meters"]
    assert vocabulary_to_rnc(vocab) == rnc


def test_simple_schema_emits_no_start_rule() -> None:
    rnc = field_tree_to_rnc([{"name": "title", "kind": "field"}], workspace="ws", docset_name="d")
    assert "start =" not in rnc
