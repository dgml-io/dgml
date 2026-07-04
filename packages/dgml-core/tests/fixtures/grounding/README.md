# Grounding fixtures (real documents)

Inputs for `tests/test_grounding_real_docs.py`, which exercises the
**code-based** grounding matcher (`dgml.matching.run_phase2_matching`,
"phase 2") against OCR captured from real documents. No LLM, no network.

## Cases

| dir                | document                                                              |
| ------------------ | --------------------------------------------------------------------- |
| `raul_terrill/`    | Raul Terrill — synthetic mutual NDA (legal contract)                  |
| `interior_design/` | Interior Design, BAA — Bellevue College program of study (public)    |

## Files per case

- `inputs.json` — the matcher's input: `phase1` (real extracted values
  with bounding boxes stripped and locations collapsed to one entry per
  `(leaf, page)` — the shape phase 1 emits before phase-2 materializes
  per-line boxes), the `layout` hint, and source metadata.
- `page_text/page_N.json` — the document's real OCR word boxes, verbatim
  from the workspace (`files/<id>/page_text/`). This is all the matcher
  reads.
- `expected_phase2.json` — the golden: every `(leaf, page)` the matcher
  currently resolves, mapped to its pixel-exact bounding boxes.

## Regenerating the golden snapshot

After an intentional change to the matcher, re-run with the update flag
and review the diff (added lines = new matches; removed/changed lines =
losses to scrutinize):

```bash
DGML_UPDATE_GROUNDING_SNAPSHOTS=1 uv run pytest \
    packages/dgml/tests/test_grounding_real_docs.py
```

## Re-bootstrapping the inputs (`inputs.json` + `page_text/`)

These are derived once from an extracted `dgml-workspace` (the doc must
be added, assigned to a docset, and `dgml file extract`-ed). The
derivation strips bboxes from the docset's `values.json`, collapses
locations to phase-1 shape, and copies `page_text/`. It is a manual,
infrequent step — only needed if a document is re-extracted or a new
case is added — so the script is not committed; see the test module
docstring for the shape, or reconstruct from `dgml-workspace`.
