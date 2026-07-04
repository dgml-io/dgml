# dgml2html

Renders a DGML XML file as a styled HTML page.

## Usage

```bash
python tools/dgml2html/dgml2html input.dgml.xml          # writes to stdout
python tools/dgml2html/dgml2html input.dgml.xml -o out.html
cat input.dgml.xml | python tools/dgml2html/dgml2html -
```

No dependencies beyond the Python standard library.

## How It Works

### `dg:structure` → HTML tag

Every DGML element with a `dg:structure` attribute maps directly to an HTML element:

| `dg:structure` | HTML tag   |
|----------------|------------|
| `section`      | `<section>`|
| `header`       | `<header>` |
| `p`            | `<p>`      |
| `ol`           | `<ol>`     |
| `ul`           | `<ul>`     |
| `li`           | `<li>`     |
| `table`        | `<table>`  |
| `tr`           | `<tr>`     |
| `td`           | `<td>`     |
| `span`         | `<span>`   |
| `lim`          | *(text only, no wrapper tag)* |

`lim` (enumerator markers such as `1.`, `A.`) emit their text content inline followed by a space, with no surrounding element.

### `dg:style` → HTML `style` attribute

The value of `dg:style` is copied verbatim into the HTML `style` attribute. Per the DGML spec, valid values are standard CSS declarations (e.g. `font-weight: bold; text-align: right`).

### Semantic elements without `dg:structure`

Semantic elements (e.g. `<docset:LiabilityCap>`) may not carry a `dg:structure` attribute, especially in AI-generated DGML. In that case the tool infers the appropriate HTML wrapper using the following fallback rules in order:

1. **Parent is an inline container** (`p`, `li`, `td`, `header`, `span`, `lim`) → `<span>`
2. **Parent is a block container** (`section`, `ol`, `ul`, `table`, `tr`) → `<div>`
3. **Any direct child has a block `dg:structure`** → `<div>`
4. **Total text content exceeds 80 characters** → `<div>`
5. **Default** → `<span>`

### Output formatting

- Block elements (`section`, `header`, `p`, `ol`, `ul`, `li`, `table`, `tr`, `td`, `div`) are indented by nesting depth.
- Text nodes are whitespace-normalized (leading/trailing stripped, internal runs collapsed to a single space).
- After a `lim` marker, the following text stays on the same line.

### Base stylesheet

The output includes an inline stylesheet for readable out-of-the-box rendering:

- White page on a light gray background with a drop shadow
- Palatino serif font, comfortable line height
- Green accents for section borders and headers
- Striped table rows, green header row with white text

## Samples

The `samples/` directory contains a comprehensive example exercising all DGML features:

- `sample.dgml.xml` — input DGML with sections, typed values, tables (record and key-value), ordered lists, `lim` markers, `dg:style`, `dg:origin`, `dg:itemprop`/`dg:href`, and `xml:id`
- `sample.dgml.html` — the rendered output; open in a browser to preview
