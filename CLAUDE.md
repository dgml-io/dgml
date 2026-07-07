# DGML

**DGML** (Document Graph Markup Language) is a semantic XML representation of business documents. Where raw source files give you layout and pixels, DGML gives you meaning: tags that describe what each element *is* in the document's domain — a contract clause, an invoice line item, a policy definition — not how it appeared on the page.

The headline feature is **cross-document tag consistency**: documents of the same kind share the same semantic vocabulary — what separates DGML from a raw extraction or structural transcription, and what makes it suitable for reasoning over a corpus rather than a single file.

The second property is **complete semantic preservation**. Traditional extraction pipelines choose fields upfront and discard the rest — a decision that fails the moment a new use case emerges and needs a field no one thought to extract. DGML preserves the full semantic structure instead — every element, relationship, and typed value — so a document processed once stays fully queryable without returning to the source.

The third is **document order with graph semantics**. Most graph formats treat documents as unordered collections of facts, but in business documents order is meaning: definitions precede usage, clause sequence governs interpretation, provenance depends on position. DGML preserves document order as a first-class property while also representing relationships across elements and documents as a graph.

The fourth is **attestation**: **Proof of Origin at the data-element level**. Every DGMLX package is tamper-evident — any alteration to its content breaks its cryptographic hash. The deeper innovation is that this hashing isn't limited to the whole document: because the semantic tree is structured, any XML element subtree — a single data point, a payment term, a liability cap — can be hashed and anchored on an external chain independently, proving its origin without producing the entire document.

License: **Apache 2.0**.

## Stack

- Python 3.11+ as the primary language.
- [`uv`](https://docs.astral.sh/uv/) for environment and package management.
- `ruff` for lint + format, `mypy` (strict) for type checking, `pytest` for tests.
- Distribution: a UV **workspace** of multiple installable packages that all
  resolve into a single shared venv with local source references between them.
  The split is deliberate: `dgml` is the **CLI** (the `dgml` command), and
  `dgml-core` (import `dgml_core`) is the **library** it drives — the PDF→DGML
  pipeline, OCR, page rendering, generation, grounding, and workspace storage.
  `dgml` depends on `dgml-core`, so `pip install dgml` still ships the command;
  library consumers `pip install dgml-core` and `import dgml_core`. Optional
  extras (`aws`, `azure`, `macos`, `clustering`, `chain`) live on `dgml-core`
  and are mirrored as pass-throughs on `dgml`, so `pip install dgml[aws]` works.

### System dependencies (not Python packages)

- **Ghostscript** (`gs`) — required for PDF page-image rendering. Install
  via the OS package manager (`brew install ghostscript`,
  `apt-get install ghostscript`, etc.). Ghostscript is AGPL, but DGML
  invokes it as a *subprocess* (like `git` or `ffmpeg`); the AGPL applies
  to the user's ghostscript install, not to the dgml wheel. The permissive-license
  policy below governs Python deps that ship inside our wheel.

## Repository layout

```text
.
├── .github/workflows/        # CI (lint, type-check, test, license-audit)
├── docs/                     # Long-form docs (CLI reference, storage layout, etc.)
├── examples/                 # Runnable end-to-end examples (PDF → DGML)
├── packages/                 # UV workspace members (one dir per package)
│   ├── dgml/                 # CLI package — the `dgml` command only
│   │   ├── pyproject.toml     #   depends on dgml-core; ships `dgml.cli:main`
│   │   ├── src/dgml/          # `src/` layout — import as `dgml` (CLI only)
│   │   │   ├── __init__.py
│   │   │   ├── cli.py
│   │   │   └── py.typed       # Marker: package ships type information
│   │   └── tests/
│   └── dgml-core/            # Library — PDF→DGML pipeline, OCR, image
│       ├── pyproject.toml     #   rendering, generation, grounding, storage
│       ├── src/dgml_core/     # `src/` layout — import as `dgml_core`
│       │   ├── __init__.py    #   the public library API
│       │   └── py.typed
│       └── tests/
├── pyproject.toml            # Workspace root + shared tool config
├── LICENSE                   # Apache 2.0
├── README.md
└── CLAUDE.md
```

Everything Python-shaped lives under `packages/<name>/`. The repo root holds
only workspace-wide configuration and meta files.

## Working with the workspace

```bash
# Install everything (all workspace members + dev deps) into one venv
uv sync

# Run the test suite for the whole workspace
uv run pytest

# Lint / format / type-check
uv run ruff check .
uv run ruff format .
uv run mypy packages
```

`uv sync` resolves *all* workspace members together — they must be jointly
solvable. If a new dep in one package conflicts with another, fix it; do not
bypass with separate venvs.

### Verify locally before pushing

CI runs four gates: `ruff check`, `ruff format --check`, `mypy packages`,
`pytest`, and a `pip-licenses` audit (see `.github/workflows/ci.yml`).
`scripts/verify.sh` runs the same things, in the same order, against
your local venv — use it before `git push` to avoid the push → CI fails
→ fix → push loop:

```bash
scripts/verify.sh              # everything CI runs (~15s on a warm venv)
scripts/verify.sh --no-sync    # skip `uv sync` if pyproject/uv.lock are
                               # unchanged (faster repeats)
scripts/verify.sh --fast       # lint + format + types only (skip tests
                               # and license audit — quick feedback loop)
```

Mirror, don't drift — if you change CI, update `verify.sh`, and vice versa.

## Adding a new package

1. `mkdir -p packages/<name>/src/<import_name> packages/<name>/tests`
2. Add `packages/<name>/pyproject.toml` (mirror `packages/dgml-core/pyproject.toml`
   for a library, or `packages/dgml/pyproject.toml` for a CLI/thin wrapper).
3. If it depends on another workspace package, declare a normal dependency
   (e.g. `dependencies = ["dgml"]`) — UV will wire it as a local source ref
   automatically because both are workspace members. **Do not** publish or
   reference the dep from PyPI for in-repo cross-package use.
4. Drop a `py.typed` marker next to `__init__.py` so type info propagates.
5. `uv lock && uv sync` to refresh the lockfile and venv.

Whenever you change runtime or dev dependencies (in any package's
`pyproject.toml` or the workspace root), run `uv lock` and **commit the
updated `uv.lock`** in the same change. CI runs `uv sync --locked` and
will fail if the lockfile drifted.

Naming: distribution name is hyphenated and namespaced under `dgml-`
(e.g. `dgml-clustering`); the import name uses dotted form (`dgml.clustering`).
Subpackages should sit under the `dgml.` namespace via PEP 420 implicit
namespace packages — i.e. **no** `__init__.py` at `src/dgml/` in subpackages,
only at the leaf (`src/dgml/clustering/__init__.py`).

## Coding conventions

- **Type hints required** on all public functions, methods, and module-level
  attributes. `mypy --strict` must pass.
- Standard PEP 8 / PEP 257 via `ruff`. Line length 100.
- Prefer the standard library; pull in a dep only when it earns its weight.
- Public API of each package is what's exported from its top-level
  `__init__.py`. Anything else is internal and may change without notice
  pre-1.0.
- Tests live next to the package they cover (`packages/<name>/tests/`).
  Integration tests that span packages can live in a top-level `tests/` if
  one is later added.

## Workspaces and the `dgml` CLI

A DGML *workspace* is a directory holding `docsets/` and `files/`. The
`dgml` CLI (entry point: [`dgml.cli:main`](../packages/dgml/src/dgml/cli.py))
manages CRUD over these primitives.

- Workspace root resolution: `--workspace` flag → `$DGML_HOME` →
  `./dgml-workspace`.
- The CLI is designed for both humans and LLM agents — JSON-default
  output, structured error envelopes on stderr, no interactive prompts.
- Full on-disk format: [docs/storage-layout.md](../docs/storage-layout.md).
- CLI command reference: [docs/cli-reference.md](../docs/cli-reference.md).

When extending the CLI, preserve those contracts. JSON output is part of
the API surface — schema changes need to be considered breaking.

## License compatibility (important)

The project is Apache-2.0-licensed. Strong copyleft (GPL/LGPL/AGPL/SSPL/EUPL/
CC-BY-SA) would be viral against our wheel and is banned outright —
including transitively. Weak copyleft (MPL-2.0) is file-level only and
allowed only as a *transitive* dep; do not add it as a direct dep.

**Direct dependencies** (anything you put in a `pyproject.toml`
`dependencies = […]` or `optional-dependencies` list):

- ✅ MIT, BSD (2/3-clause), Apache-2.0, ISC, PSF, Unlicense, 0BSD
- ❌ GPL (any version), LGPL, AGPL, MPL, SSPL, EUPL, CC-BY-SA, "source-available"

**Transitive dependencies** (what `pip` drags in to satisfy the direct
deps you picked):

- ✅ Everything in the direct-dep allow-list, plus MPL-2.0 (weak,
  file-level). In practice `certifi`, `tqdm`, and `pathspec` arrive
  this way and are unavoidable for any HTTP-using stack.
- ❌ Same strong-copyleft list as above.

If you're considering pulling MPL into the direct-dep list anyway,
don't — find an alternative or open an issue before merging.

PDF-space gotchas to watch for:

- `PyMuPDF` / `fitz` — **AGPL**, do not use.
- `pdf2image` depends on `poppler` (GPL via system binary). Do not use
  it or any other poppler-backed wrapper. The generation pipeline reuses
  workspace `page_images/` rendered by ghostscript at file-add time (or
  renders into a tempdir via the same canonical helper for non-workspace
  inputs), so no poppler-backed rasterizer is needed.
- ✅ acceptable PDF libs: `pypdf` (BSD-3), `pdfminer.six` (MIT),
  `pdfplumber` (MIT). `pikepdf` is MPL-2.0 and therefore borderline —
  acceptable transitively but not as a direct dep; prefer alternatives.

Run an audit when in doubt — `--partial-match` is required for the
deny tokens to match real license strings, and MPL is intentionally
omitted (we accept it transitively):

```bash
uv run pip-licenses --from=mixed --partial-match \
  --fail-on="GPL;LGPL;AGPL;SSPL;EUPL;CC-BY-SA"
```

## Format / spec notes for future work

- DGML is a **semantic** XML representation. Tags should describe what an
  element *is* in the document's domain (e.g. invoice line item, contract
  clause, table header), not how it looked on the page. Layout details are a
  means, not the output.
- "Consistent tags across similar documents" is the headline feature —
  changes that hurt cross-document tag stability need an explicit rationale.
- The format spec lives in the parallel [`dgml-spec`](https://github.com/dgml-io/dgml-spec)
  repo, not in `docs/`. Keep it versioned alongside changes here that affect
  the format.

# Fork-based contribution workflow (REQUIRED)

## Git & PR policy — do NOT push branches to the upstream repo

This org uses a **fork-based** workflow for all day-to-day code changes. Never create
a branch on or push to the upstream `dgml-io/*` repository, even if you have write access.
(Claude Code's built-in default is to branch on the current repo — that default does NOT
apply here; this instruction overrides it.)

When you need to open a PR:

1. Ensure a fork under the current user's account exists: `gh repo fork --remote=false` (idempotent).
2. Add/verify a `fork` remote pointing at that fork; leave `origin` pointing at upstream for pulls.
3. Push the topic branch to **`fork`**, never to `origin`.
4. Open the PR from the fork: `gh pr create --repo dgml-io/<repo> --head <user>:<branch>`.

Do **not** run `git push origin <branch>` or a bare `git push` when `origin` resolves to
`github.com/dgml-io/*`. If you're unsure whether a remote is the upstream or a fork,
check `git remote -v` first.
