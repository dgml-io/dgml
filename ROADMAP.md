# Roadmap

This roadmap covers the **`dgml` reference implementation** — the CLI, the PDF→DGML pipeline, and integrations built on top of it. The format specification's own roadmap lives in [dgml-io/dgml-spec](https://github.com/dgml-io/dgml-spec).
It reflects our priorities for the next 3–6 months.

## Themes
### 1. Third-party integrations
Expand DGML reach with integration with other solutions:
 * LangChain, LlamaIndex
  * Expose DGML capabilities through MCPs to enable agentic-based scenarios.

Tracked in [#11](https://github.com/dgml-io/dgml/issues/11).

### 2. Validate with small/open-source LLMs
Prove the generation/tagging/extraction pipeline works well on small, open-weight LLMs (e.g. Llama, Mistral) — not just frontier proprietary ones. This may also includes proposing open-weight LLMs to the community as a supported path, or adapter weights built on top of open-weight models. The goal is to enable more cost-sensitive and self-hosting scenarios.

Tracked in [#12](https://github.com/dgml-io/dgml/issues/12).

### 3. Real-world examples, more samples, and multi-chain attestation
Grow the set of industry examples showing DGML in production-style use — more real-world document types, contributed to [`dgml-spec/samples/`](https://github.com/dgml-io/dgml-spec/tree/main/samples) — and exercise attestation across multiple blockchain networks. The focus here is breadth of real usage and industry examples.

Tracked in [#13](https://github.com/dgml-io/dgml/issues/13).

## How to influence this roadmap
Open an [Issue](../../issues) or [Discussion](../../discussions). If your use case doesn't fit one of the three themes above, that's useful signal too — say so.
