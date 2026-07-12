# Get Started with DGML: Hands-on Walkthrough Guide

Welcome to the **DGML Get Started Guide**. 

**DGML** (Document Graph Markup Language) is a semantic XML representation of business documents. Where raw source files give you layout and pixels, DGML gives you meaning: tags that describe what each element *is* in the document's domain — a contract clause, an invoice line item, a policy definition — not how it appeared on the page. Spec can be found in the [DGML-Spec Repo](https://github.com/dgml-io/dgml-spec).

The headline feature is **cross-document tag consistency**: documents of the same kind share the same semantic vocabulary — what separates DGML from a raw extraction or structural transcription, and what makes it suitable for reasoning over a corpus rather than a single file.

The second property is **complete semantic preservation**. Traditional extraction pipelines choose fields upfront and discard the rest — a decision that fails the moment a new use case emerges and needs a field no one thought to extract. DGML preserves the full semantic structure instead — every element, relationship, and typed value — so a document processed once stays fully queryable without returning to the source.

The third is **document order with graph semantics**. Most graph formats treat documents as unordered collections of facts, but in business documents order is meaning: definitions precede usage, clause sequence governs interpretation, provenance depends on position. DGML preserves document order as a first-class property while also representing relationships across elements and documents as a graph.

The fourth is **attestation**: **Proof of Origin at the data-element level**. Every DGMLX package is tamper-evident — any alteration to its content breaks its cryptographic hash. The deeper innovation is that this hashing isn't limited to the whole document: because the semantic tree is structured, any XML element subtree — a single data point, a payment term, a liability cap — can be hashed and anchored on an external chain independently, proving its origin without producing the entire document.


This guide walks you through the four core phases of the DGML toolchain from absolute scratch, using the real PDF files published in the [`dgml-io/dgml-spec`](https://github.com/dgml-io/dgml-spec) repository under its `samples/` directory. Section 1.2 shows you how to fetch them.

---
## 4 Sample Scenarios
### 1. Non-Traded Real Estate Funds
Marguerite Halloran, CFO of a non-traded real estate fund, must publish a monthly net asset value (NAV) — the per-share value of everything the fund owns — that investors rely on to subscribe or redeem. Her valuation advisor, Theo Ellison, won’t stand behind the number unless he can trace each property’s value back to the underlying signed leases — which today sit in scattered systems he cannot independently verify.

DGML encodes each lease as a structured document: base rent, escalation schedule, free-rent concessions, operating-expense recoveries, renewal options, co-tenancy clauses — every data element typed, traced to the source page, and anchorable on-chain. Theo follows the trail from NAV to clause without a phone call.

### 2. Commercial Mortgage Lenders
Russell Carrow, CFO of a publicly traded commercial mortgage lender, needs investors to believe the buildings backing his loans are still worth enough to cover them. Analyst Naomi Brandt has openly questioned whether the company is marking its troubled office loans honestly — and she wants independently verifiable evidence, not management assurances.

DGML encodes each loan as a structured document: principal, covenants, reported performance, tenant concentration, reserve balances, default triggers — every data element typed, traced to the source page, and anchorable on-chain. Russell produces a loan-by-loan trail Naomi can check herself.

### 3. Private Credit Funds
Daniel Roth, Head of Private Wealth Solutions at a private credit manager, wants to make his fund’s net asset value (NAV) verifiable enough to offer a liquid, eventually tokenized interest. Evelyn Castellano, Chair of a public pension plan’s investment committee, will not commit capital to a fund whose NAV she cannot independently verify down to individual loans.

DGML encodes each loan as a structured document: principal, interest terms, maintenance covenants, fair-value assumptions, collateral — every data element typed, traced to the source page, and anchorable on-chain. Evelyn traces the fund’s NAV down to one borrower’s covenant breach or a single recovery assumption.

### 4. Infrastructure Funds
Margit Dahl, CFO of an infrastructure fund, wants to give institutional investors a way to value, transfer, or borrow against positions in toll roads, regulated utilities, and renewable-energy assets. Conrad Boyle, Head of Real Assets at a public pension plan, will not increase, transfer, or finance his stake without independently verified operating performance across decades-long horizons.

DGML encodes each concession agreement and operating report as a structured document: tariff mechanisms, performance obligations, debt-service covenants, capital expenditure commitments, ESG metrics — every data element typed, traced to the source page, and anchorable on-chain. Conrad traces a valuation down to a single toll road’s traffic figures.

---
## The 4-Phase Walkthrough

1. **Phase 1: Initial Setup, Ingestion, & Workspace Control** — Learn how to set up a workspace, ingest files from `dgml-spec/samples/1-NonTraded-NAV-REITs`, and organize them using **file** and **docset** verbs.
2. **Phase 2: Automated Document Clustering** — Ingest the larger document collection in `dgml-spec/samples/4-Infrastructure-Funds` and let the ML clusterer automatically group and name DocSets.
3. **Phase 3: Anchoring (Staking) to the NVNM Blockchain** — Learn how to generate deterministic SHA-256 Merkle roots of your documents and anchor them on the **NVNM Chain testnet** for tamper-proof trust.
4. **Phase 4: Integrity Verification & Proof Validation** — Use the on-chain state to prove that your local document is authentic and has not been altered, down to a single clause or table cell.

---

## Phase 1: Setup, File Ingestion, and Workspace Control

In this phase, we initialize a DGML workspace and ingest our first set of documents from `dgml-spec/samples/1-NonTraded-NAV-REITs/files/`. This directory contains a mix of rent rolls and primary property documents (like `Arcadia Biotech.pdf`, `Elysium Dome.pdf`, `Olympus Plaza.pdf`, etc.).

### 1.1 Install System Dependencies & CLI
First, ensure you have **Ghostscript** (`gs`) installed. It is required for page-image rendering:

```bash
# macOS
brew install ghostscript

# Debian/Ubuntu
sudo apt-get install ghostscript
```

Next, sync your environment. Since this repository uses `uv` for package management, you can synchronize all workspace packages directly:

```bash
# From the repository root
uv sync
```

Alternatively, to run individual commands without a manual environment setup, prepend `uv run` to your CLI commands:
```bash
uv run dgml --help
```

### 1.2 Get the Sample Documents
The sample PDFs used throughout this guide are published in the [`dgml-io/dgml-spec`](https://github.com/dgml-io/dgml-spec) repository. Clone it alongside your working directory so the paths below resolve:

```bash
git clone https://github.com/dgml-io/dgml-spec.git
```

This gives you `dgml-spec/samples/<scenario>/files/`, where each scenario's raw source documents live (e.g. `dgml-spec/samples/1-NonTraded-NAV-REITs/files/`). The commands in this guide assume you run them from the parent directory that now contains `dgml-spec/`.

### 1.3 Initialize your Workspace
All DGML documents, metadata, and schemas reside in a single directory called the **Workspace**. By default, DGML looks at the `DGML_HOME` environment variable, or falls back to `./dgml-workspace`.

Let's initialize a clean workspace. `--organization` is required — it is
embedded in this workspace's docset namespace URIs
(`http://dgml.io/<organization>/<DocSetSlug>`), so pick a stable identifier for
your org. `--name` is an optional human-readable label (defaults to the
workspace directory name):

```bash
export DGML_HOME=./my-dgml-workspace
uv run dgml workspace create --organization "Acme" --name "Getting Started"
```
*Note: `workspace create` is idempotent and safe to re-run. It creates the
workspace (`docsets/` + `files/`), records its identity in `workspace.json`,
and seeds the shared `local_config.json` (a peer of the workspace) from the
bundled template if it doesn't exist yet, copying it in as `config.json` — so
no separate `dgml init` is needed. The response's `next_action` tells you where
to edit the models / OCR endpoint. Run `dgml init` first only if you'd rather
review and edit that shared config before any workspace is created; re-run
`dgml workspace create --force` to re-sync an edited config into an existing
workspace.*

### 1.4 Configure Models and API Keys
Ingesting files needs no configuration, but every LLM-backed command you will
run later in this guide — `dgml docset generate` (Phase 1), the cluster
auto-naming (Phase 2), and value extraction — reads its model and credentials
from `<workspace>/config.json`. There are **no in-code model defaults**: an
unconfigured section fails the command with an error like
`GENERATION_CONFIG_MISSING` rather than silently making a paid LLM call you
didn't set up. Configure it now so the later phases run through cleanly.

Open `<workspace>/config.json` (seeded by `workspace create` in the previous
step) and review these sections:

```json
{
  "generation": {
    "model": "anthropic/claude-haiku-4-5",
    "label_model": "anthropic/claude-sonnet-4-6",
    "api_key_env": "ANTHROPIC_API_KEY"
  },
  "classification": {
    "model": "gemini/gemini-2.5-flash",
    "api_key_env": "GEMINI_API_KEY"
  }
}
```

- **`generation`** — required by `dgml docset generate`. `model` runs the
  per-page transcription (the bulk of the calls); `label_model` runs the single
  batch-wide semantic-labeling call. Any provider-prefixed litellm model id
  works.
- **`classification`** — a vision-capable model used to auto-name clusters in
  Phase 2 (and by `dgml file add --auto-classify`).
- **`grounded`** — only needed for schema-driven value extraction
  (`dgml extraction …`, see the interlude before Phase 3).

Each section names its API key indirectly via `api_key_env` — the **name** of
an environment variable, never the secret itself. Export the matching key in
your shell before running the commands:

```bash
export ANTHROPIC_API_KEY="sk-ant-…"
export GEMINI_API_KEY="…"
```

*Tip: edits to `<workspace>/config.json` apply to this workspace only. To make
your configuration the default for future workspaces, edit the shared
`local_config.json` (a peer of the workspace directory) and re-run
`dgml workspace create --force` to re-sync it. The full schema of every config
section is in [docs/storage-layout.md](../docs/storage-layout.md).*

### 1.5 Ingest Sample Documents
Let's add the non-traded REIT PDF documents from `dgml-spec/samples/1-NonTraded-NAV-REITs/files`. We use `--recursive` to walk folders and `--on-conflict skip` to ensure our run is idempotent (safely resuming if interrupted):

```bash
uv run dgml file add "dgml-spec/samples/1-NonTraded-NAV-REITs/files" --recursive --on-conflict skip
```

Behind the scenes, DGML performs the following tasks per PDF:
1. Copies the file into `<workspace>/files/<file_id>/`.
2. Computes the SHA-256 hash of the bytes.
3. Renders each page into a high-resolution (300 DPI) PNG under `page_images/`.
4. Extracts per-page text layout word boxes using `pdfminer.six` and stores them as JSON under `page_text/`.

You can inspect the summary of this ingestion pass by piping the output to `jq`:
```bash
uv run dgml file add "dgml-spec/samples/1-NonTraded-NAV-REITs/files" --recursive --on-conflict skip | jq .summary
```

### 1.6 Workspace Control: Navigating Files and DocSets
DGML organizes your workspace around two primary entities: **Files** and **DocSets**.

- **Files** are the raw, physical documents.
- **DocSets** are logical groupings (types) of files that share the same structural schema (e.g., "Rent Rolls" or "Operating Statements").

Let's use our CLI verbs to inspect and control the workspace:

#### Check Workspace Status
Print a summary showing the workspace path, count of DocSets, and total files:
```bash
uv run dgml status
```

#### List Ingested Files
Retrieve all files in the workspace with their unique 12-char base-36 IDs (e.g., `k7q3xb91pmrf`):
```bash
uv run dgml file list
```

#### Inspect a Specific File
View the metadata of an individual file using its file ID:
```bash
uv run dgml file show <file_id>
```

#### Create a Custom DocSet
Let's manually create a DocSet for Non-Traded REIT main files:
```bash
uv run dgml docset create --name "NonTraded_REITs" --description "Primary property descriptions and assets"
```
This returns a DocSet ID (e.g., `p9pjusnwg50l`).

#### Assign a File to a DocSet
To link an ingested file to your newly created DocSet:
```bash
uv run dgml docset add-file <file_id> --docset <docset_id>
```

#### Generate DGML XML for the DocSet
With files assigned, you can run the PDF → DGML semantic pipeline. It uses the
`generation` models you configured in §1.4:
```bash
uv run dgml docset generate <docset_id>
```
This transcribes pages window-by-window, plans a shared semantic labeling schema, and writes namespaced `.dgml.xml` files for each document into `<workspace>/docsets/<docset_id>/files/<file_id>/<stem>.dgml.xml`.

---

## Phase 2: Automated Document Clustering

Manually categorizing and labeling hundreds of incoming files is tedious. DGML solves this with **automated clustering**, which groups files by the similarity of their content, then auto-names the groups using a vision LLM.

To demonstrate, we will use the `dgml-spec/samples/4-Infrastructure-Funds/files` directory, which contains a larger set of files (including loan agreements, quarterly reports, and valuation memos across multiple sets).

### 2.1 Set Up Clustering Dependencies
Clustering requires the `clustering` extra, which installs machine learning and embedding libraries. DGML is not published to PyPI yet, so install the extra from your repository checkout:

```bash
# From the repository root
uv sync --extra clustering
```

*(Once DGML is published to PyPI, this will become `pip install "dgml[clustering]"`.)*

Additionally, auto-naming the resulting clusters requires a vision-capable LLM in the `classification` section of your `<workspace>/config.json` — you set this up in §1.4. Make sure the matching API key (e.g. `GEMINI_API_KEY`) is exported in your terminal environment.

### 2.2 Ingest the Infrastructure Funds
First, let's ingest the large batch of Infrastructure Fund PDFs into our workspace:

```bash
uv run dgml file add "dgml-spec/samples/4-Infrastructure-Funds/files" --recursive --on-conflict skip
```

The folder holds three document kinds for each of eleven deals: loan agreements (`deal_<name>_la.pdf`), quarterly reports (`deal_<name>_qr.pdf`), and valuation memos (`deal_<name>_vm.pdf`).

*Note: one deal's documents (`deal_aegis_*`) ship as Word/Excel files rather than PDFs. Ingesting those requires a configured document converter (see [docs/conversion.md](../docs/conversion.md)); without one they are recorded with a conversion error and skipped by the clusterer. The walkthrough works fine with just the 30 PDFs of the other ten deals.*

### 2.3 Run the Clusterer
Now, execute the clustering command over the unassigned files. Because Phase 1 already created a DocSet, the default `--mode auto` would run *incremental* clustering and pull the new files toward that existing DocSet — so pass `--mode fresh` to cluster the unassigned files from scratch into new groups:

```bash
uv run dgml cluster --mode fresh
```

Under the hood, the clusterer:
1. Embeds each unassigned document from its first-page text (the default is a corpus-fitted TF-IDF text encoder; a file also needs a rendered first-page image to be eligible).
2. Reduces the embeddings with UMAP and runs the **Leiden community detection algorithm** to identify distinct clusters.
3. Assigns any cluster whose name matches an existing DocSet to that DocSet.
4. For each remaining cluster, collects a few sample page images and sends them to the vision LLM configured in `classification` (§1.4), which proposes a cohesive **Name** (e.g., "Loan Agreements", "Valuation Memos") and **Description** for the group.
5. Creates the new DocSets in your workspace and assigns the respective files to them.

Partial success is the contract: if the `classification` config is missing or an LLM call fails, the affected files land in `failed_file_ids` while every other cluster is still assigned — fix the config and re-run.

Verify the auto-created DocSets and assignments:
```bash
uv run dgml docset list
```
You will find that files of the same kind — for example the loan agreements `deal_alpha_la.pdf`, `deal_apex_la.pdf`, and `deal_titan_la.pdf` — have been grouped under a single, auto-labeled DocSet, with the quarterly reports and valuation memos in DocSets of their own.

---

## Interlude: Generation, Extraction, and the DGMLX Bundle

Before anchoring anything, it helps to understand the two LLM-backed processing passes DGML offers, because what you run determines what ends up in the tamper-evident bundle you anchor.

**Generation** (`dgml docset generate`, used in Phase 1) transcribes the *whole* document into a semantic XML tree — every clause, table, and value typed, labeled with the DocSet's shared vocabulary, and grounded to its source-page position (`dg:origin`). Use it when the full document needs to stay queryable and verifiable.

**Extraction** (`dgml extraction …`) pulls a *defined set of fields* out of a document against an extraction schema, and grounds each value back to the source page. The schema can be proposed by an LLM (`dgml extraction generate-schema <docset_id>`) or supplied by you (`set-schema`); `dgml extraction extract <docset_id> <file_id>` then writes the values as a `dg:extraction` element **inside the file's core `<stem>.dgml.xml`** — there is no separate values file. It uses the `grounded` config section (§1.4). See the [extraction commands](../docs/cli-reference.md#extraction-commands) for details.

When to use which:

| You run… | You get… |
|---|---|
| **Generation only** | The complete semantic tree. Any value can be located and proven later, without deciding fields upfront. |
| **Extraction only** | A minimal core XML holding just the schema's fields, each grounded. Cheaper when you only ever need a known field set. |
| **Both** (either order) | The full tree *plus* the `dg:extraction` element alongside it — full queryability and a stable, schema-shaped view of the key fields. |
| **Neither** | The file is still ingested, hashed, organized, and page-rendered — and can still be anchored at whole-file granularity. |

**The DGMLX bundle** is the portable, tamper-evident export of everything DGML knows about a file: the source document, its page images, and — when a DocSet is named — the DocSet schema(s) and the generated `<stem>.dgml.xml`. All of it is rolled up into a single SHA-256 Merkle root recorded in the bundle's `META-INF/dgml-attestation.xml`, so any alteration to any artifact breaks verification. A bundle exists in two forms — zipped into a single `<stem>.dgmlx` archive, or *unpacked* as a plain directory with the same layout — and `dgml dgmlx verify` accepts either. The bundle format is specified in Part II of the [DGML spec](https://github.com/dgml-io/dgml-spec/blob/main/spec.md).

DGMLX bundles are **optional**, and the two commands that build one produce the two forms:

- `dgml dgmlx export` writes the portable `.dgmlx` archive — use it when you want to hand a self-contained, verifiable document package to another party.
- `dgml stake file` (Phase 3) builds the bundle implicitly, because its Merkle root is exactly what gets anchored on-chain — but it materializes the bundle **unpacked**, as a directory under `dgmlx-bundles/<file_id>[-<docset_id>]/` in your workspace, so the exact artifacts that were hashed stay directly inspectable and the on-chain receipt (`record.json`) is saved alongside them. It never writes a `.dgmlx` archive; if you also want the portable file, run `dgml dgmlx export` — both forms carry the identical Merkle root.

What the root covers follows from what you ran:

- **Neither pass:** the root covers the source document and page images only — whole-file anchoring still works.
- **Generation** adds the DocSet's `full-schema.rnc` and the semantic XML to the root — and enables *element-level* anchoring (`stake node`), which needs the generated tree.
- **Extraction** adds `extraction-schema.rnc` and the `dg:extraction` values (inside the core XML) to what is attested.

A missing artifact is simply an absent slot (a smaller bundle), never an error. The flip side: re-running generation or extraction changes the Merkle root, so anchor *after* the document has reached the state you want to attest — and re-stake if you deliberately reprocess it.

---

## Phase 3: Anchoring (Staking) to the NVNM Blockchain

Once a file is in your workspace — ideally with generated DGML XML — you can **stake (anchor)** it to a blockchain. This establishes a permanent, cryptographic proof of the document's content without uploading the actual PDF or XML contents to the public ledger.

DGML achieves this using a **SHA-256 Merkle Tree**:
- Every element node in the generated XML is canonicalized using **Exclusive XML Canonicalization (C14N)** and hashed.
- These hashes are paired bottom-up to produce a single **Merkle Root Hash**.
- Only the 64-character Merkle Root Hash is written on-chain.
- Off-chain, you maintain the XML elements and an **Inclusion Proof**, which allows you or a third party to prove that a specific sentence, table cell, or clause was part of the original staked document.

Rather than duplicate the walkthrough here, follow the dedicated
**[chaining quickstart](../docs/chaining/quickstart-chaining.md)**. It takes
you from absolute zero to an anchored document on the **NVNM Chain testnet**:

1. Install MetaMask, create a throwaway wallet, add the NVNM testnet, and fund it from the faucet (steps 1–7).
2. Install the chain extra — from this repo checkout that is `uv sync --extra chain` (the quickstart's `pip install "dgml[chain]"` applies once DGML is on PyPI) — and point the CLI at your workspace (steps 8–10).
3. Export your private key and store it in your OS keyring (steps 11–12).
4. Pick a file with generated DGML from your workspace, create a registry, and stake the file — or a single XML element of it (steps 13–15).

Staking saves a `record.json` receipt (or `record-node-<leaf>.json` for a single element) in the bundle directory — keep it, Phase 4 uses it.

---

## Phase 4: Integrity Verification and Proof Validation

Now that your document root is anchored in the blockchain, you can mathematically prove its absolute integrity, or the integrity of specific extracted values, to any external auditor.

This verification is highly robust:
- If someone edits a single number in the local PDF,
- If a word box changes coordinate slightly,
- If the schema or XML tag naming shifts even by one character,
- **The resulting Merkle Root will change, causing verification to fail.**

### 4.1 Validate using the Local Record file
You can perform offline validation using your saved `record.json` receipt. It compares the hashes of your current local workspace files against the cryptographic parameters saved in the receipt:

```bash
uv run dgml prove file --chain nvnm-testnet --record-json record.json
```

If the document is authentic and unmodified, the CLI will output:
```json
{
  "valid": true,
  "checksum": "...",
  "uri": "..."
}
```
And exit with code `0`.

### 4.2 Validate by Querying the Blockchain Registry
If you do not have the local `record.json` file, you can fetch the anchored state directly from the NVNM blockchain by referencing its registered checksum:

```bash
uv run dgml prove file --chain nvnm-testnet \
  --registry "<your-registry-name>" --checksum <checksum_from_staking_phase>
```

### 4.3 Witness Tamper-Proofing in Action
To witness the power of Merkle attestation, try making a trivial modification to the generated XML. For example, open the XML file:
`<workspace>/docsets/<docset_id>/files/<file_id>/<stem>.dgml.xml`

Change a single letter inside any tag or text node, then save the file and re-run the proof command:
```bash
uv run dgml prove file --chain nvnm-testnet --record-json record.json
```

The CLI will detect the tamper instantly, outputting `"valid": false` and exiting with code `2`.

---

## Summary & Next Steps

Congratulations! You have completed the comprehensive getting started walkthrough for DGML. You have:
1. Initialized a workspace and mastered **files** and **docsets**.
2. Automated classification with **ML clustering** on a larger sample set.
3. Created secure on-chain **registries** using MetaMask, `keyring`, and NVNM Testnet.
4. Cryptographically staked and validated documents using **SHA-256 Merkle proofs**.

For deep architectural details and command references, read:
- **`docs/cli-reference.md`** — Comprehensive flag listings and subcommands.
- **`docs/merkle-attestation.md`** — In-depth explanation of the Merkle tree construction.
- **`docs/storage-layout.md`** — Layout description of how files and caches are stored on-disk.
