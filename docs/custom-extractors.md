# Custom extractors

`dgml extraction extract` runs three phases: a model reads the values and
their page numbers, code matches each value to the page's words, and a model
boxes what code could not. A **custom extractor** replaces all three: the
`[extractor]` section names a class, dgml hands it one file at a time (or the
whole run under `--batch`), and it returns the values with their boxes. dgml
then runs what it runs after its own phases: provenance normalization, box
verification against the text layer, the drop of values no box supports, the
`dg:extraction` write, and the usage row and stats.

There is no bundled extractor. dgml's own pipeline is what runs when none is
configured, and what a run falls back to when the configured one fails.

## Configuring an extractor

```toml
[extractor]
provider = "my_pkg.extractors:InvoiceExtractor"   # module.path:ClassName
fallback = true                                   # default; false fails the file instead
max_concurrency = 5                               # default; files in flight under --batch
model = "my-model"                                # the class's own options follow
```

`provider`, `fallback` and `max_concurrency` are dgml's keys. Every other key
reaches the class's `parse_config` as `config.options`; the class lists the
keys it accepts in `config_fields`, and any other is refused with
`EXTRACTOR_CONFIG_INVALID`. The section is validated when the config is
loaded, before any file is read, by every command that extracts:
`extraction extract` and the auto-extraction of `docset add-file`,
`file add --auto-classify`, `cluster` and `docset run`.

`--values-model` and `--values-effort` override the values model, which only
dgml's own pipeline uses, so with an extractor configured they affect the
fallback alone. An extractor's model is its own option in the table.

## Writing your own extractor

1. `pip install dgml-core` (or `pip install dgml`, which pulls it in).
2. Subclass `ValueExtractor`:

   ```python
   from typing import ClassVar

   from dgml_core import ExtractionRequest, ExtractorConfig, ExtractorOutput, ValueExtractor
   from dgml_core.errors import ExtractorConfigInvalid, ExtractorFailed


   class InvoiceExtractor(ValueExtractor):
       name: ClassVar[str] = "invoice"
       config_fields: ClassVar[frozenset[str]] = frozenset({"model"})

       @classmethod
       def parse_config(cls, config: ExtractorConfig) -> ExtractorConfig:
           if not isinstance(config.options.get("model"), str):
               raise ExtractorConfigInvalid("extractor.model is required")
           return config  # must return it

       def __init__(self, config: ExtractorConfig) -> None:
           self.model = config.options["model"]  # lazy-import any SDK here

       def extract(self, request: ExtractionRequest) -> ExtractorOutput:
           pdf = request.pdf()
           words = request.page_words(1)["words"]
           ...
           return ExtractorOutput(values=values, model=self.model)
   ```

3. Make the class importable by the interpreter running `dgml`.
4. Point `extractor.provider` at it.

A dotted `provider` path runs code from your config, as you, the same trust
model as `[ocr]`, `[conversion]` and `[storage]`.

### API keys

An extractor that calls a model provider can use the key already configured
for dgml's own models instead of taking its own. List the providers in
`api_key_providers`, and read each key in `__init__` with `config.api_key`:

```python
from dgml_core import Provider


class InvoiceExtractor(ValueExtractor):
    api_key_providers: ClassVar[frozenset[Provider]] = frozenset({Provider.ANTHROPIC})

    def __init__(self, config: ExtractorConfig) -> None:
        self.client = anthropic.Anthropic(api_key=config.api_key(Provider.ANTHROPIC))
```

The keys come from the `[models]` provider keys, `anthropic_api_key`,
`google_api_key` and `openai_api_key`, or their `_api_key_env` forms
(`Models(anthropic_api_key=...)` in a `Configuration.build`):

- dgml passes only the providers the class lists. A provider outside
  `anthropic`, `google` and `openai` is refused with
  `EXTRACTOR_CONFIG_INVALID` when the config is loaded.
- An `_api_key_env` form is read from the environment when the extractor is
  built. A variable that is not set passes no key and logs a WARNING.
- A provider with no key configured is left out, and `config.api_key`
  returns `None`. Passing `None` to the provider's SDK makes it read its own
  environment variable. An extractor that cannot run without a key raises
  `ExtractorFailed`.
- `parse_config` never sees a key; `config.api_keys` is empty there.
- Keys never appear in the config's `repr`, the stats file, the usage rows,
  the DGML or a batch job's records.

A key goes only to its provider's own API. Send it nowhere else, and take a
key for any other service as one of the class's own options.

### The request

An `ExtractionRequest` is one file against one docset:

| Field or method | What it is |
| --- | --- |
| `docset_id` | The docset the file is extracted against. |
| `file_id`, `original_filename`, `page_count` | The file. |
| `schema_rnc` | The docset schema as stored (RELAX NG compact). |
| `schema` | The same schema as JSON Schema, the shape the values tree follows. |
| `vocab` | The parsed schema dgml validates the output against. |
| `guidance` | The docset's extraction guidance, or `None`. |
| `pdf()` | The stored PDF bytes. |
| `page_words(page)` | The page's words with boxes in page-image pixels, as the `get_page_words` tool returns them; empty `words` when the page has no text layer. |
| `page_image(page)` | `(key, bytes)` of the rendered page image, or `None`. |
| `page_size(page)` | `(width, height)` in pixels of that image, the space every box is in; `None` when unknown. |

The accessors read the workspace on each call and cache nothing. They are the
whole content contract: there is no `Workspace` on the request, and nothing
on it writes.

### The output

`ExtractorOutput(values, model=None, usage=None, stats=None, tool_calls=0)`.

`values` is the values tree, the shape `dgml extraction get-values` prints
and `dgml_core.extraction_xml.dgml_xml_to_values` reads back (an extractor
that produces DGML XML can convert with it). A field is a leaf dict, a
container a dict, a collection a list of entries. Every leaf is:

```json
{
  "text": "$1,234.56",
  "value": "1234.56",
  "locations": [{"page_number": 2, "bounding_box": [412, 1088, 530, 1112]}]
}
```

- `text` is the span as printed. `value` is optional, the normalized form (an
  enum token, an ISO date, a plain number), written as `dg:value` when it fits
  the field's declared type.
- `locations` lists where the text is printed: a 1-based `page_number` and a
  `bounding_box` of four integers `[left, top, right, bottom]` in the pixel
  space of `page_size(page)`. Every printed value must be boxed; dgml never
  boxes a value from a page number alone.
- A value the page does not print has no `locations` and a `method`:
  `calculated` or `inferred` with `inputs` naming the values used
  (`"LineItems[2].Amount=$27.05"`) or `evidence` listing printed text used
  (`{"text": ..., "locations": [...]}`), or `defaulted` with nothing else.
  A value printed in part carries both `locations` and `inputs`.

`usage` is the extractor's cost and token totals in the `usage.jsonl` keys
(`cost_usd`, `prompt_tokens`, `completion_tokens`, `total_tokens`,
`cache_read_tokens`, `cache_creation_tokens`), merged into the run's totals
even after a fallback. `stats` is free-form JSON carried as
`extractor.details` in the stats file. Neither can fail the file: a malformed
block is dropped with one WARNING.

### What dgml checks

The output is validated before anything else touches it; a failure is
`EXTRACTOR_OUTPUT_INVALID`, listing up to ten problems with their paths:

1. It is an `ExtractorOutput` whose `values` is a JSON-serializable dict.
2. The tree follows the vocabulary strictly: root keys are schema roots, a
   field is a dict, a container a dict, a collection a list, a choice one of
   its alternatives, and an unknown key is an error. An empty tree, or roots
   set to `null`, is a legal "nothing found".
3. Each leaf: `text` is a string; `value`, if present, a string; `method` one
   of `calculated`, `inferred`, `defaulted`; `inputs` a non-empty list of
   non-empty strings; `evidence` a list of `{text, locations}`.
4. Each location: `page_number` in `1..page_count`; `bounding_box` four
   integers with `left < right` and `top < bottom`, inside the page's pixel
   size when it is known.

The tree then gets the normalization dgml's own phase-1 output gets, and the
box verification: where the page has a text layer, the words under each box
must read as the leaf's `text` (letters and digits, case-insensitive); a box
that does not is moved to the one exact match on its line or rejected, and a
printed value left with no box is dropped. The stats count `snapped_boxes`,
`rejected_boxes` and `dropped_unboxed_values`. An extractor that checks its
own boxes, and may return a normalized `text` the page does not print, sets
`verifies_boxes: ClassVar[bool] = True`: dgml then runs the check on a copy,
keeps the counts, and writes the boxes as returned (`boxes_verified:
applied | reported` in the stats).

### `extract_many` and `--batch`

Under `--batch` dgml calls `extract_many(requests, max_concurrency=...)` once
with every file of the run and batches only the model requests it still makes
itself, phase 1 and 3 for the files that fell back. The default `extract_many`
runs `extract` on a thread pool of `max_concurrency` workers, so `extract`
must be thread-safe. Override it to submit the run to your own provider's
batch API, returning one outcome per request in request order: an output, or
the exception that request failed with. The call blocks; under `--no-wait`
the command exits only after it returns.

With `fallback = false` dgml makes no model request of its own, so `--batch`
does not need the values model to have a batch backend or a key, and the
`batch` block is all zeros with `"provider": null`.

In job mode each accepted output is recorded under the job
(`inputs/extractor/<docset>/<file>.json`), so a resumed run replays it and
never calls the extractor for that file again. A failed or rejected output is
not recorded: the resume calls the extractor again, and takes its tree if it
succeeds or replays the fallback's phase 1 if not.

### Failure and fallback

- A config error (`EXTRACTOR_CONFIG_INVALID`): a bad dotted path, a class that
  is not a `ValueExtractor` or is abstract, `parse_config` raising or returning
  `None`, an unknown option key. Raised at config load. Never falls back.
  Schema generation does not read `[extractor]`, so it is not affected.
- The extractor raising (`EXTRACTOR_FAILED`), from `__init__` or `extract`,
  whatever the exception. Raise `ExtractorFailed` for a failure you detect.
- Invalid output (`EXTRACTOR_OUTPUT_INVALID`).

With `fallback = true` (the default) the last two run the file through dgml's
own pipeline instead, logged as a WARNING naming the provider and the files
(`--verbose` names every file). With `fallback = false`, or
`--no-extractor-fallback` on `extraction extract`, the file fails with
`extractor <provider> failed: <detail>`, where the detail is the
`ExtractorFailed` or `ExtractorOutputInvalid` message, or the type and
message of any other exception; the error carries the run's usage and stats.
An interrupt never falls back.

With `fallback = true` the values model's key is resolved before the extractor
runs, so a `values_api_key_env` naming an unset variable fails every file with
`AUTH_ERROR` before the extractor is called, even the files it would have
handled: a run whose fallback cannot work never pays for the extractor. With
`fallback = false` the key is not read.

### What the result says

- `extractor`, on the CLI payloads, the `extraction` block of the assignment
  commands and `ExtractionResult.extractor`, names the extractor whose values
  were written, `null` after a fallback. The key is present only when an
  extractor is configured.
- `model` is what produced the values: the extractor's reported model (or its
  provider path), or the values model after a fallback. On a failed run, in
  the stats file, the usage row, a failed multi-file entry or auto-extraction
  block and the error's `stats`, it is what ran: the values model when the
  fallback ran, whatever its outcome, and `null` when the extractor failed and
  no fallback ran. A multi-file run's top-level `model` is the values model.
- The `dg:extraction` element carries `dg:extractor="<provider>"` when a
  custom extractor produced it.
- The stats file (`--debug`) gains an `extractor` block: `provider`, `model`,
  `outcome` (`ok`, `failed`, `invalid`), `error`, `fell_back`, `duration_s`,
  `tool_calls`, `locations`, `boxes_verified`, `usage` and `details`. When the
  extractor's values were taken, `phases.phase1` holds the extractor's time
  and usage, the phase-3 counters are zero, and `locations` counts the boxes
  in its tree rather than phase-1 value-page entries, so the count is not
  comparable with dgml's own runs.

### Logging

Log through `self.log()`, the `dgml_core.extractor.<name>` logger the CLI
routes to stderr (WARNING by default, INFO under `--verbose`). A logger under
your own module's name is not shown. Never write to stdout, which is the CLI's
JSON surface, and never configure handlers or levels.

## Troubleshooting

| Message | Cause |
| --- | --- |
| `extractor provider must be a dotted path` | `provider` has no `:`. Use `module.path:ClassName`. |
| `could not import extractor module …` | The module is not importable by the interpreter running `dgml`. |
| `… is not a ValueExtractor subclass` | The path resolved to something else. |
| `… resolved to abstract class …` | The class misses `parse_config`, `__init__` or `extract`. |
| `….parse_config must return an ExtractorConfig` | Your `parse_config` validated but did not `return config`. |
| `unknown fields in 'extractor' for provider …` | An option not in `config_fields`. |
| `extractor … failed: extractor output is invalid (N problems): …` | See "What dgml checks"; each problem names its path. |
| `extractor … failed: <Type>: …` | The extractor raised that exception, with no fallback. |
| `extractor … failed … extracted with dgml's own pipeline instead` | A fallback. The error is in the message and in the stats block. |
