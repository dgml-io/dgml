# dgml4models

Strips spatial and layout attributes from a DGML XML file before sending it to a language model.

## Usage

```bash
python tools/dgml4models/dgml4models input.dgml.xml          # writes to stdout
python tools/dgml4models/dgml4models input.dgml.xml -o out.xml
cat input.dgml.xml | python tools/dgml4models/dgml4models -
```

No dependencies beyond the Python standard library.

## What Gets Stripped

| Attribute      | Purpose in DGML                        |
|----------------|----------------------------------------|
| `dg:origin`    | Pixel bounding box on source page image |
| `dg:structure` | Visual/layout role (`section`, `p`, …) |
| `dg:style`     | Inline CSS styling                     |

These attributes are for grounding and rendering — not for reasoning. Removing them reduces token count without losing any semantic content.

## What Is Preserved

Everything a model needs to reason over the document:

- Element tags (`docset:LiabilityCap`, `dg:chunk`, …)
- `dg:value` — normalized machine-readable values
- `xsi:type` — type annotations (`date`, `decimal`, `integer`, …)
- `dg:itemprop` / `dg:href` — named semantic links between elements
- `xml:id` — element identifiers used by cross-references
- All text content
- All namespace declarations

## Samples

The `samples/` directory contains a before/after example:

- `sample.dgml.xml` — input DGML with all attributes present
- `sample.dgml.m.xml` — output ready for models with `dg:origin`, `dg:structure`, and `dg:style` removed
