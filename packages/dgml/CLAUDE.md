# `dgml` package

The **CLI** package. Owns the public `dgml` command surface, which is consumed
by humans **and** LLM agents. It is a thin front end: every command resolves a
workspace and calls into the **`dgml-core`** library (import `dgml_core`) — the
pipeline, OCR, page rendering, generation, grounding, and storage all live
there, not here. This package holds only `src/dgml/cli.py` (plus a CLI-only
`__init__.py`) and depends on `dgml-core`. The library API lives in
`dgml_core`: library code uses `from dgml_core import …`; the CLI package
re-exports nothing. See [../dgml-core/CLAUDE.md](../dgml-core/CLAUDE.md)
for the library (including how to add OCR providers).

## CLI surface is a public contract

The CLI's commands, flags, JSON output schema, exit codes, and error
codes are part of `dgml`'s API. Schema changes are breaking.

**When you change the CLI surface, five files must move together:**

1. [src/dgml/cli.py](src/dgml/cli.py) — implementation
2. [tests/test_cli.py](tests/test_cli.py) — locks the JSON shape; update assertions for the new payload
3. [../../docs/cli-reference.md](../../docs/cli-reference.md) — human and agent reference
4. [../../.claude/skills/dgml/SKILL.md](../../.claude/skills/dgml/SKILL.md) — Claude skill that teaches agents *when* and *how* to invoke the CLI
5. [../../get-started/get-started.md](../../get-started/get-started.md) — the end-to-end tutorial; update any command invocation the change touches (e.g. new required flags, renamed args, changed setup steps) so a reader following it top-to-bottom doesn't hit a broken command

Also grep the rest of `docs/` for the affected command — reference pages like [../../docs/storage-layout.md](../../docs/storage-layout.md) may show the same invocation or its on-disk output.

A "CLI surface change" includes any of:

- adding, removing, or renaming a command or subcommand
- adding, removing, or renaming a flag or positional argument
- changing the JSON output shape (field names, types, nesting, enum values like `conflict_kind`)
- changing exit codes or error code identifiers (`FILE_NOT_FOUND`, `WORKSPACE_NOT_INITIALIZED`, …)
- changing the default behavior of an existing command or flag
- moving output between **stdout and stderr**, or changing what a command prints by default — stdout is the JSON contract; progress/diagnostics belong on stderr behind `--verbose`. A consumer that streams or parses stdout can break even when the JSON schema is unchanged.

For SKILL.md specifically: update the relevant workflow example, not just a reference table — the skill's value is in showing agents the *right pattern* (e.g. `--on-conflict skip` in bulk loops), not just listing flags.

### Downstream CLI consumers

The four files above are the in-package contract. Some sample tools also
**shell out to the `dgml` CLI** and parse its output, so a surface change can
break them even though they live outside this package. When you change
command syntax, flags, payload shapes, or stdout/stderr behavior, grep the
repo for CLI spawns and update any that are affected:

```bash
grep -rIl "subprocess" app-sample/ tools/ 2>/dev/null | xargs grep -lI "dgml"
```

Known consumer today:
[app-sample/dgml-app-sample-server.py](../../app-sample/dgml-app-sample-server.py)
— spawns `dgml` (`init`, `file add`, `cluster [--skip-existing]`, `docset
list`, `docset generate`, `docset ground`), parses **`docset list`** output,
and streams the child's stdout **and stderr** into its live log (so a
stdout↔stderr move changes what the UI shows). It passes global flags
(`--workspace`, `--verbose`) *before* the subcommand.
