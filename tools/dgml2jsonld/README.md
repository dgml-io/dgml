# dgml2jsonld

Converts a DGML semantic XML file to a JSON-LD document following the
[XAST (XML Abstract Syntax Tree)](https://github.com/syntax-tree/xast) convention —
a DOM-style tree representation of XML in JSON.

The XML file is the system of record. The JSON-LD is a delivery vehicle derived
mechanically from it, optimized for consumption by JSON-LD processors and graph tools.

## Install

```bash
uv pip install .
```

## Usage

### CLI

```bash
dgml2jsonld path/to/document.dgml.xml
```

Writes JSON-LD to stdout.

### Python API

```python
from dgml2jsonld import xml_to_jsonld, xml_to_jsonld_string

# Returns a dict
result = xml_to_jsonld("document.dgml.xml")

# Returns a formatted JSON string
json_str = xml_to_jsonld_string("document.dgml.xml")
```

## Output format

Every XML element becomes a JSON object:

```json
{
  "nodeType": "xast:element",
  "@type": "docset:LiabilityCap",
  "@id": "#clause-1",
  "attributes": {
    "xsi:type": "decimal",
    "dg:value": "500000",
    "dg:origin": "3 180 500 2360 540"
  },
  "children": [
    {"nodeType": "xast:text", "value": "$500,000"}
  ]
}
```

- `nodeType` — `"xast:element"` for elements, `"xast:text"` for text nodes
- `@type` — the element tag as a CURIE (e.g. `"docset:LiabilityCap"`, `"dg:chunk"`)
- `@id` — present only if the element has an `id` or `xml:id` attribute; plain local ids are prefixed with `#`
- `attributes` — all XML attributes except `id`, `xml:id`, `dg:itemprop`, and `dg:href`; omitted if empty
- `children` — ordered array of child nodes, always present

If an element carries both `dg:itemprop` and `dg:href`, those become a named link
property on the element: `{ "<itemprop_value>": {"@id": "<href_value>"} }`.
If `dg:href` holds more than one space-separated id (an aggregating property
with multiple targets), the property becomes a list of link objects instead:
`{ "<itemprop_value>": [{"@id": "<id_1>"}, {"@id": "<id_2>"}, ...] }`.

The `@context` always includes the `xast:` namespace terms inline — no external fetch required.
All `xmlns:` declarations on the root element are mapped to `@context` prefix entries.

## Sample

```bash
dgml2jsonld samples/sample.dgml.xml
```

See [`samples/sample.dgml.xml`](samples/sample.dgml.xml) and the pre-generated
[`samples/sample.dgml.jsonld`](samples/sample.dgml.jsonld).

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
