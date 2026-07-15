---
name: attestation
description: Use when the user wants to stake, anchor, or attest DGML artifacts on NVNM Chain (or another configured EVM chain), or prove a previously anchored record — a whole file's DGMLX bundle or a single DGML XML node. Triggers on phrases like "stake this file", "anchor on chain", "attest this document", "stake a node", "prove the record", "verify against the chain", "NVNM". Uses the `dgml[chain]` CLI; no MCP server required.
---

# Attesting DGML artifacts on a chain

NVNM Chain is an EVM L2 used as a neutral notary: records hold a SHA-256
checksum + a URI + JSON metadata inside a *registry*. DGML talks to it
(and to any compatible EVM chain) **directly** through the `dgml`
CLI's chain commands — no MCP server. DGML anchors two granularities:

- **Bundle (DGMLX)** — checksum = the bundle's Merkle root over every
  artifact of a file (PDF, page images, page text, schema, DGML XML).
- **Node** — checksum = the canonical hash of ONE element of the DGML
  XML; the record metadata carries the document tree's Merkle root and
  the RFC 6962 inclusion path, so the node can be proven against the
  tree without revealing the other nodes.

Anchored checksums, URIs, and metadata are **public on-chain** — never
put document content in them (node records expose only hashes; the
node's text stays off-chain).

## Hard rules

- **All chain work goes through the `dgml` CLI** (`dgml stake|prove|
  registry|wallet|chain`). These require the `chain` extra — if a
  command returns a `MISSING_EXTRA` envelope, tell the user to
  `pip install dgml[chain]` (or `uv sync` in the repo).
- **Never ask for, read, or print the private key.** The signing key
  lives in the OS keyring (service `nvnm-wallet`, account `default`;
  override with `NVNM_KEY_SERVICE` / `NVNM_KEY_ACCOUNT`). The CLI reads
  it only at signing time and refuses to sign if it doesn't control the
  `--from` address.
- **Writes broadcast by default.** `stake` and `registry create` build,
  sign, and send in one step. Use `--dry-run` first when the user wants
  to inspect the transaction before spending gas.
- **Save the record JSON.** `dgml stake` writes the fetched record into
  the bundle dir (`record.json` for a file bundle, `record-node-<leaf>.json`
  for a node) and reports the path in `record_path`; keep it — proving
  works from that file plus the workspace, with no chain access.

## Configuration

- Chain: `--chain <name>` (env `NVNM_CHAIN`, default `nvnm-testnet`).
  `dgml chain list` shows configured chains; `dgml chain add` registers
  a custom EVM chain.
- Registry **name** (not id): `--registry` (env `NVNM_REGISTRY`).
- Sender address: `--from` (env `NVNM_FROM_ADDRESS`); defaults to the
  keyring key's address.
- Before any write, check the wallet has gas:

```bash
dgml wallet status --chain "$chain"   # balance + pending nonce
```

## Stake a file (DGMLX bundle)

```bash
dgml stake file "$fid" --docset "$ds" \
  --chain "$chain" --registry "$reg" --workspace "$ws"
```

One command exports the bundle, anchors its Merkle root (URI
`dgmlx://<fid>/<ds>`), broadcasts, waits for the receipt, then fetches
and saves the record. The result JSON carries `checksum` (the root),
`uri`, `tx_hash`, `receipt_status`, `record`, `record_path`, and
`explorer_url`. By default the bundle is written as a single portable
`<stem>.dgmlx` archive (path in `dgmlx`); add `--unpacked` to write the
loose bundle tree instead (path in `attestation`, no `dgmlx`). Add
`--dry-run` to stop after signing (emits `unsigned_tx` + `signed_tx`,
broadcasts nothing).

## Stake a node (one DGML XML element)

```bash
# Selector: --xpath (from the UX tree view's "Copy XPath") or --leaf <n>.
dgml stake node "$fid" --docset "$ds" \
  --xpath '/dg:chunk/docset:Entry[2]/docset:Amount' \
  --chain "$chain" --registry "$reg" --workspace "$ws"
```

The URI gains a `#<leaf>` fragment and the metadata embeds
`{kind: "dgml-node", root_hash, proof}`. Node attestation needs the
docset-scoped DGML XML at its canonical location
(`docsets/<ds>/files/<fid>/<stem>.dgml.xml`).

## Prove an anchored record

```bash
# Look the record up on-chain by checksum…
dgml prove file --chain "$chain" --registry "$reg" --checksum "$sum" --workspace "$ws"
# …or prove from a saved record file (no chain access needed):
dgml prove file --chain "$chain" --record-json record.json --workspace "$ws"
```

Use `prove file` for a bundle URI (no `#fragment`) and `prove node` for
a node URI (`#leaf`). Both exit `0` proven / `2` mismatch
(`valid: false` — report it, never paper over it) / `1` structural
error. On a node mismatch, compare `computed_node_hash` vs
`expected_node_hash` to tell "this node changed" from "the tree around
it changed".

## Gotchas

- The `--registry` parameter is the registry **NAME**. Create one first
  with `dgml registry create --chain … --name …` (same sign/broadcast
  flow) if the user has none — the creator becomes admin.
- A `reverted` receipt (CLI error `CHAIN_TX_REVERTED`) usually means a
  role/permission problem on the registry, not a bug.
- Re-anchoring the same `(registry, record)` adds a **version**
  (`index` increments, `is_latest` moves). Re-running generation or
  adding artifacts changes roots — the old record version then stops
  proving against the workspace, by design; anchor a new version and
  tell the user.
- An empty `dgml wallet status` balance means the wallet needs gas
  (wmantraUSD on NVNM testnet) before any write.
- Other notaries/transports are future extensions; today the anchor
  interface is assumed identical across configured chains.
