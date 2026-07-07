# rnc2jsonld

Converts a DGML docset schema (`schema.rnc`, RELAX NG Compact) to a JSON-LD
document that declares the vocabulary used by that docset.

The output `@context` namespace URI matches the one used by
[dgml2jsonld](../dgml2jsonld) for the same docset, making the two JSON-LD
documents directly joinable by a standard JSON-LD processor.

## Install

```bash
uv pip install .
```

## Usage

### CLI

```bash
rnc2jsonld path/to/schema.rnc
```

Writes JSON-LD to stdout.

### Python API

```python
from rnc2jsonld import rnc_to_jsonld, rnc_to_jsonld_string

# Returns a dict
result = rnc_to_jsonld("schema.rnc")

# Returns a formatted JSON string
json_str = rnc_to_jsonld_string("schema.rnc")
```

## Output format

The output contains a fixed `@context` and a `@graph` of tag and group definitions.

```json
{
  "@context": {
    "docset": "<namespace URI from schema>",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "Tag":          "docset:Tag",
    "TagGroup":     "docset:TagGroup",
    "members":      { "@id": "docset:members", "@type": "@id", "@container": "@set" },
    "description":  "docset:description",
    "example":      "docset:example"
  },
  "@graph": [
    { "@id": "docset:ClauseItemTag", "@type": "docset:TagGroup", "members": ["..."] },
    { "@id": "docset:LiabilityCap",  "@type": "docset:Tag", "description": "...", "example": "..." },
    { "@id": "docset:VendorName",    "@type": "docset:Tag", "description": "..." }
  ]
}
```

### RNC concept mapping

| RNC construct | JSON-LD result |
|---|---|
| `namespace docset = "…"` | `"docset": "…"` entry in `@context` |
| Element definition | Node with `"@type": "docset:Tag"` |
| Named group (alternation of element refs) | Node with `"@type": "docset:TagGroup"` and `"members"` array |
| `## description line` doc comment | `"description"` property on the tag node |
| `## Example: …` doc comment line | `"example"` property on the tag node |

## Sample

```bash
rnc2jsonld samples/sample.schema.rnc
```

See [`samples/sample.schema.rnc`](samples/sample.schema.rnc) and the pre-generated
[`samples/sample.schema.jsonld`](samples/sample.schema.jsonld).

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
