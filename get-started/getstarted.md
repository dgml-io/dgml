# Get Started with DGML: Hands-on Walkthrough Guide

Welcome to the **DGML Get Started Guide**. 

**DGML** (Document Graph Markup Language) is a semantic XML representation of business documents. Where raw source files give you layout and pixels, DGML gives you meaning: tags that describe what each element *is* in the document's domain — a contract clause, an invoice line item, a policy definition — not how it appeared on the page. Spec can be found in the [DGML Repo](https://github.com/dgml-io/dgml-spec).

)

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

### 1.4 Ingest Sample Documents
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

### 1.5 Workspace Control: Navigating Files and DocSets
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
With files assigned, you can run the PDF → DGML semantic pipeline:
```bash
uv run dgml docset generate <docset_id>
```
This transcribes pages window-by-window, plans a shared semantic labeling schema, and writes namespaced `.dgml.xml` files for each document into `<workspace>/docsets/<docset_id>/files/<file_id>/<stem>.dgml.xml`.

---

## Phase 2: Automated Document Clustering

Manually categorizing and labeling hundreds of incoming files is tedious. DGML solves this with **automated clustering**, which groups files based on visual layout and text semantics, then auto-names the groups using an LLM.

To demonstrate, we will use the `dgml-spec/samples/4-Infrastructure-Funds/files` directory, which contains a larger set of files (including loan agreements, quarterly reports, and valuation memos across multiple sets).

### 2.1 Set Up Clustering Dependencies
Clustering requires the `clustering` extra, which installs machine learning and embedding libraries:

```bash
pip install "dgml[clustering]"
```

Additionally, auto-naming the resulting clusters requires a Vision LLM configuration in your `<workspace>/config.json`. Add a `classification` section using your preferred LLM provider:

```json
{
  "classification": {
    "model": "gemini/gemini-2.5-flash",
    "api_key_env": "GEMINI_API_KEY"
  }
}
```
*Make sure your `GEMINI_API_KEY` (or chosen provider key) is exported in your terminal environment.*

### 2.2 Ingest the Infrastructure Funds
First, let's ingest the large batch of Infrastructure Fund PDFs into our workspace:

```bash
uv run dgml file add "dgml-spec/samples/4-Infrastructure-Funds/files" --recursive --on-conflict skip
```

### 2.3 Run the Clusterer
Now, execute the clustering command. This operates on all unassigned files in the workspace:

```bash
uv run dgml cluster --skip-existing
```

Under the hood, the clusterer runs an **unsupervised (S1)** pipeline:
1. It extracts visual features from the first page of each document using a vision-aware encoder.
2. It combines these with semantic text embeddings of the document's content.
3. It maps the files into a joint manifold space and runs the **Leiden community detection algorithm** to identify distinct clusters.
4. For each detected cluster, it collects a few sample page images and sends them to your Vision LLM.
5. The LLM analyzes the layouts and proposes a cohesive **Name** (e.g., "Loan Agreements", "Valuation Memos") and **Description** for each group.
6. The CLI automatically creates the DocSets in your workspace and assigns the respective files to them!

Verify the auto-created DocSets and assignments:
```bash
uv run dgml docset list
```
You will find that files like `set1_loan_agreement.pdf` and `set2_loan_agreement.pdf` have been grouped under a single, auto-labeled DocSet!

---

## Phase 3: Anchoring (Staking) to the NVNM Blockchain

Once you have generated DGML XML for a file, you can **stake (anchor)** it to a blockchain. This establishes a permanent, cryptographic proof of the document's layout and content without uploading the actual PDF or XML contents to the public ledger.

DGML achieves this using a **SHA-256 Merkle Tree**:
- Every element node in the generated XML is canonicalized using **Exclusive XML Canonicalization (C14N)** and hashed.
- These hashes are paired bottom-up to produce a single **Merkle Root Hash**.
- Only the 64-character Merkle Root Hash is written on-chain.
- Off-chain, you maintain the XML elements and an **Inclusion Proof**, which allows you or a third party to prove that a specific sentence, table cell, or clause was part of the original staked document.

Let's anchor a document on the **NVNM Chain testnet**!

### 3.1 Network Details
We will connect to the EVM-compatible NVNM Chain testnet:

- **Network name:** `NVNM testnet`
- **Chain ID:** `787111`
- **RPC URL:** `https://evm.testnet.nvnmchain.io`
- **Block explorer:** `https://explorer.evm.testnet.nvnmchain.io/`
- **Faucet:** `https://faucet.testnet.nvnmchain.io/`

### 3.2 Enable Chaining and Setup MetaMask Wallet
1. Install MetaMask in your browser.
2. Create a fresh, throwaway wallet and add the **NVNM testnet** as a custom network using the parameters above.
3. Visit the **faucet** (`faucet.testnet.nvnmchain.io`) and enter your EVM address (starts with `0x`) to receive **10 $ (wmantraUSD)** of testnet gas.
4. Install the CLI chain extension:
   ```bash
   pip install "dgml[chain]"
   ```

### 3.3 Securely Store your Private Key
To sign transactions, the CLI needs access to your account's private key. Export the **Ethereum private key** from MetaMask.

We use the cross-platform `keyring` library bundled with DGML to store the key in your operating system's native secure credential manager (macOS Keychain, Windows Credential Manager, or GNOME Keyring). This ensures your key never leaks into terminal logs or history:

```bash
keyring set nvnm-wallet default
# When prompted, paste your MetaMask private key and hit Enter (characters will not echo)
```

Confirm that the CLI can securely resolve your address and check your balance:
```bash
uv run dgml wallet status --chain nvnm-testnet
```
This should print your wallet address and show a balance of `10` ($) with `"funded": true`.

### 3.4 Create an On-Chain Registry
Staked documents reside in registries on the chain. Let's create a unique registry. Replace `my-unique-registry` with a unique name of your choice (it must be globally unique):

```bash
uv run dgml registry create --chain nvnm-testnet \
  --name "my-unique-registry-12345" \
  --description "Registry for REIT and Infrastructure document anchors"
```

### 3.5 Stake the Document
Now let's anchor one of our processed files. Pick a DocSet ID (`ds`) and a File ID (`fid`) from your workspace:

```bash
uv run dgml stake file "<file_id>" --docset "<docset_id>" \
  --chain nvnm-testnet --registry "my-unique-registry-12345"
```

This command:
1. Formulates the Merkle Tree of the entire document bundle (PDF, images, schema, and XML) and writes it as a single portable `<stem>.dgmlx` archive (pass `--unpacked` to write the loose bundle tree instead).
2. Broadcasts a transaction pinning the Merkle Root Hash under the URI schema `dgmlx://<file_id>/<docset_id>`.
3. Waits for the blockchain transaction to commit.
4. Saves a `record.json` file in the document's workspace directory (alongside the archive) containing the block receipt, on-chain checksum, and the Merkle structure.

### 3.6 Stake a Single Node (Optional)
You can also anchor just **one specific XML element** (e.g., a critical lease rate or termination clause). Identify the element path (using `--xpath`) and stake it:

```bash
uv run dgml stake node "<file_id>" --docset "<docset_id>" \
  --xpath '/dg:chunk/docset:Entry[2]/docset:Amount' \
  --chain nvnm-testnet --registry "my-unique-registry-12345"
```
The metadata will preserve the document's overall Merkle Root alongside a compact inclusion proof, saving the receipt as `record-node-<leaf_index>.json`.

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
  --registry "my-unique-registry-12345" --checksum <checksum_from_staking_phase>
```

### 4.3 Witness Tamper-Proofing in Action
To witness the power of Merkle attestation, try making a trivial modification to the generated XML. For example, open the XML file:
`<workspace>/docsets/<docset_id>/files/<file_id>/<stem>.dgml.xml`

Change a single letter inside any tag or text node, then save the file and re-run the proof command:
```bash
uv run dgml prove file --chain nvnm-testnet --record-json record.json
```

The CLI will detect the tamper instantly, outputting `"valid": false` and exiting with code `2`!

---

## Summary & Next Steps

Congratulations! You have completed the comprehensive getting started walkthrough for DGML. You have:
1. Initialized a workspace and mastered **files** and **docsets**.
2. Automated classification with **multimodal ML clustering** on a larger sample set.
3. Created secure on-chain **registries** using MetaMask, `keyring`, and NVNM Testnet.
4. Cryptographically staked and validated documents using **SHA-256 Merkle proofs**.

For deep architectural details and command references, read:
- **`docs/cli-reference.md`** — Comprehensive flag listings and subcommands.
- **`docs/merkle-attestation.md`** — In-depth explanation of the Merkle tree construction.
- **`docs/storage-layout.md`** — Layout description of how files and caches are stored on-disk.
