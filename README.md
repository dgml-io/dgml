# DGML

**DGML** (Document Graph Markup Language) is a semantic XML representation of business documents. Where raw source files give you layout and pixels, DGML gives you meaning: tags that describe what each element *is* in the document's domain — a contract clause, an invoice line item, a policy definition — not how it appeared on the page.

The headline feature is **cross-document tag consistency**: documents of the same kind share the same semantic vocabulary — what separates DGML from a raw extraction or structural transcription, and what makes it suitable for reasoning over a corpus rather than a single file.

The second property is **complete semantic preservation**. Traditional extraction pipelines choose fields upfront and discard the rest — a decision that fails the moment a new use case emerges and needs a field no one thought to extract. DGML preserves the full semantic structure instead — every element, relationship, and typed value — so a document processed once stays fully queryable without returning to the source.

The third is **document order with graph semantics**. Most graph formats treat documents as unordered collections of facts, but in business documents order is meaning: definitions precede usage, clause sequence governs interpretation, provenance depends on position. DGML preserves document order as a first-class property while also representing relationships across elements and documents as a graph.

The fourth is **attestation**: **Proof of Origin at the data-element level**. Every DGMLX package is tamper-evident — any alteration to its content breaks its cryptographic hash. The deeper innovation is that this hashing isn't limited to the whole document: because the semantic tree is structured, any XML element subtree — a single data point, a payment term, a liability cap — can be hashed and anchored on an external chain independently, proving its origin without producing the entire document.

This repository is the **Python reference implementation**: the CLI, the PDF→DGML pipeline, ML clustering, on-chain attestation, DOCX/XLSX→PDF translators, and evaluation tooling. The format specification itself lives in the parallel [`dgml-spec`](https://github.com/dgml-io/dgml-spec) repo.

License: **Apache 2.0**.

## Get started

New here? Start with **[`get-started/getstarted.md`](get-started/getstarted.md)** — a hands-on walkthrough that takes you from zero to a cryptographically staked, tamper-verified document using the real sample PDFs in this repo. It covers all four phases of the toolchain:

1. Workspace setup and file ingestion
2. Automated document clustering
3. Anchoring (staking) to the NVNM blockchain
4. Integrity verification and proof validation

It's the fastest way to see the whole system work end to end.

## Install

```bash
uv sync              # install the full workspace into one venv
uv run dgml --help   # or: pip install dgml
```

See [CLAUDE.md](CLAUDE.md) for the full workspace/package layout and contributor conventions.

## Repository layout

| Path | What's there |
|---|---|
| [`get-started/`](get-started) | Hands-on walkthrough guide — **start here**. |
| [`packages/`](packages) | UV workspace members: `dgml` (the CLI), `dgml-core` (the library — PDF→DGML pipeline, OCR, rendering, generation, grounding, storage), `clustering` (`dgml-clustering`, ML document clustering), `dgml-chain` (blockchain staking/attestation), `translators-pdf` (translates other formats, e.g. DOCX, XLSX, to PDF so they can flow through the same PDF→DGML pipeline). |
| [`tools/`](tools) | Standalone, dependency-light CLIs for working with DGML XML directly: [`dgml2html`](tools/dgml2html) (render as styled HTML), [`dgml2jsonld`](tools/dgml2jsonld) (convert to JSON-LD/XAST), [`dgml4models`](tools/dgml4models) (strip layout attributes before sending to an LLM), [`rnc2jsonld`](tools/rnc2jsonld) (convert a docset's RNC schema to a JSON-LD vocabulary). |
| [`docs/`](docs) | Long-form docs: CLI reference, on-disk storage layout, conversion, Merkle attestation, blockchain chaining quickstart, clustering quickstarts. |
| [`app-sample/`](app-sample) | Single-file sample web app for browsing and rendering DGML documents. |
| [`scripts/`](scripts) | Repo-wide dev scripts — `verify.sh` mirrors what CI runs; also rendering/grounding utilities. |
| [`.github/workflows/`](.github/workflows) | CI: lint, type-check, test, license audit. |

## The spec

DGML the *format* — its schema, semantics, blockchain anchoring model, and versioning — is specified independently of this implementation, in [**dgml-io/dgml-spec**](https://github.com/dgml-io/dgml-spec). This repo tracks that spec and implements it; format questions, proposals, and discussions belong in the spec repo, not here.

## Participate

DGML is an open initiative, and it's still early — the format and the implementation are both being shaped right now, in the open, by whoever shows up.

- **Build.** This repo is the reference implementation. Use it, break it, improve it. The implementations that emerge from real use cases are what harden a standard, and the people who build them are the ones who end up shaping its direction. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.
- **Shape.** The format itself — schema, semantics, versioning — is specified in [`dgml-spec`](https://github.com/dgml-io/dgml-spec). Open an issue there to propose a new subject, challenge a design decision, or bring a use case the spec hasn't addressed yet. What gets raised now gets considered now.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
