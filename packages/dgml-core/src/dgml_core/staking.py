# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrate on-chain attestation: the local half + direct chain calls.

This module glues the workspace-local Merkle export (``file_attestation`` /
``node_attestation``) to the direct EVM transport in ``dgml_chain``. The
``dgml stake|prove|registry|wallet|chain`` CLI commands are thin wrappers over
the functions here.

Importing this module requires the ``dgml[chain]`` extra (it imports
``dgml_chain`` at module load); the CLI imports it lazily and reports
``MISSING_EXTRA`` when absent.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from dgml_chain import (
    CHECKSUM_ALGO,
    AnchorContract,
    ChainConfig,
    ChainStore,
    EvmRpc,
    RpcError,
    build_node_uri,
    build_unsigned_tx,
    build_uri,
    bundle_metadata,
    encode_add_record,
    encode_add_registry,
    keyring_address,
    node_metadata,
    parse_uri,
    sign_tx,
)
from dgml_chain.signer import DEFAULT_ACCOUNT, DEFAULT_SERVICE

from .errors import (
    ChainConfigError,
    ChainRpcFailed,
    ChainTxReverted,
    InvalidArgument,
    RecordNotFound,
    WalletKeyMissing,
)
from .file_attestation import attest_file, export_attestation
from .merkle import proof_from_json, proof_to_json
from .node_attestation import export_node, prove_node
from .storage import Workspace, read_json

# Receipt polling: how long to wait for inclusion before giving up.
_POLL_TIMEOUT_S = 120
_POLL_INTERVAL_S = 2.0


# --- configuration -----------------------------------------------------------


def resolve_chain_config_path(ws: Workspace, explicit: Path | None) -> Path:
    """Resolve where custom chains live: flag → ``$DGML_CHAINS`` → workspace."""
    if explicit is not None:
        return explicit
    env = os.environ.get("DGML_CHAINS")
    if env:
        return Path(env)
    return ws.root / "chains.json"


def _chain_store(ws: Workspace, config_path: Path | None) -> ChainStore:
    try:
        return ChainStore(resolve_chain_config_path(ws, config_path))
    except ValueError as exc:
        raise ChainConfigError(str(exc)) from exc


def get_chain(ws: Workspace, name: str, config_path: Path | None) -> ChainConfig:
    try:
        return _chain_store(ws, config_path).get(name)
    except KeyError as exc:
        raise ChainConfigError(str(exc).strip("\"'")) from exc


def _chain_entry(store: ChainStore, cfg: ChainConfig) -> dict[str, Any]:
    entry = cfg.to_json()
    entry["builtin"] = store.is_builtin(cfg.name)
    return entry


def chain_list(ws: Workspace, config_path: Path | None) -> dict[str, Any]:
    store = _chain_store(ws, config_path)
    chains = [_chain_entry(store, c) for _, c in sorted(store.all().items())]
    return {"chains": chains}


def chain_show(ws: Workspace, name: str, config_path: Path | None) -> dict[str, Any]:
    store = _chain_store(ws, config_path)
    try:
        return _chain_entry(store, store.get(name))
    except KeyError as exc:
        raise ChainConfigError(str(exc).strip("\"'")) from exc


def chain_add(
    ws: Workspace,
    *,
    name: str,
    rpc_url: str,
    chain_id: int,
    anchor_address: str,
    explorer: str | None,
    native_token: str | None,
    config_path: Path | None,
) -> dict[str, Any]:
    store = _chain_store(ws, config_path)
    cfg = ChainConfig(
        name=name,
        rpc_url=rpc_url,
        chain_id=chain_id,
        anchor_address=anchor_address,
        explorer=explorer,
        native_token=native_token,
    )
    try:
        store.add(cfg)
    except ValueError as exc:
        raise ChainConfigError(str(exc)) from exc
    return {"added": _chain_entry(store, cfg), "config_path": str(store.config_path)}


def chain_remove(ws: Workspace, name: str, config_path: Path | None) -> dict[str, Any]:
    store = _chain_store(ws, config_path)
    try:
        store.remove(name)
    except ValueError as exc:
        raise ChainConfigError(str(exc)) from exc
    except KeyError as exc:
        raise ChainConfigError(str(exc).strip("\"'")) from exc
    return {"removed": name, "config_path": str(store.config_path)}


# --- shared helpers ----------------------------------------------------------


def _clients(chain: ChainConfig) -> tuple[EvmRpc, AnchorContract]:
    rpc = EvmRpc(chain.rpc_url)
    return rpc, AnchorContract(rpc, chain.anchor_address)


def _resolve_from(from_address: str | None, service: str, account: str) -> str:
    addr = from_address or keyring_address(service, account)
    if not addr:
        raise WalletKeyMissing(
            "no sender address: pass --from, or store a key in the keyring "
            f"(service={service!r} account={account!r})"
        )
    return addr


def _explorer_tx_url(chain: ChainConfig, tx_hash: str) -> str | None:
    return f"{chain.explorer.rstrip('/')}/tx/{tx_hash}" if chain.explorer else None


def _wei_to_eth(wei: int) -> str:
    """Format a wei amount as an exact decimal string (no float rounding)."""
    whole, frac = divmod(wei, 10**18)
    return f"{whole}.{frac:018d}".rstrip("0").rstrip(".")


def _wait_for_receipt(rpc: EvmRpc, tx_hash: str) -> dict[str, Any]:
    """Poll until the transaction is mined; return its receipt.

    Raises ``ChainTxReverted`` if the receipt reports a failed status (the
    precompile rejected the write — typically a role/permission problem).
    """
    deadline = time.monotonic() + _POLL_TIMEOUT_S
    receipt: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        receipt = rpc.get_transaction_receipt(tx_hash)
        if receipt is not None:
            break
        time.sleep(_POLL_INTERVAL_S)
    if receipt is None:
        raise ChainRpcFailed(
            f"transaction {tx_hash} not mined within {_POLL_TIMEOUT_S}s "
            "(it may still confirm later; re-check with `dgml prove`)"
        )
    # Modern EVM receipts carry status 0x1 (success) / 0x0 (failure). Treat
    # only an explicit zero as a revert; a missing status (some non-standard
    # nodes omit it) means the tx was mined, so we accept it rather than
    # falsely reporting a successful anchor as reverted.
    status = receipt.get("status")
    if status is not None:
        try:
            status_int = int(status, 16) if isinstance(status, str) else int(status)
        except (TypeError, ValueError):
            status_int = None
        if status_int == 0:
            raise ChainTxReverted(
                f"transaction {tx_hash} reverted (status {status!r}) — the precompile "
                "rejected the write; check that the sender has a role on the registry"
            )
    return receipt


def _prepare_and_sign(
    chain: ChainConfig,
    rpc: EvmRpc,
    *,
    from_address: str,
    data: str,
    service: str,
    account: str,
    legacy: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the unsigned tx for ``data`` and sign it. Returns (unsigned, signed)."""
    try:
        unsigned = build_unsigned_tx(
            rpc,
            from_address=from_address,
            to=chain.anchor_address,
            data=data,
            chain_id=chain.chain_id,
            legacy=legacy,
        )
    except RpcError as exc:
        raise ChainRpcFailed(str(exc)) from exc
    try:
        signed = sign_tx(unsigned, service=service, account=account, expected_from=from_address)
    except ValueError as exc:
        raise WalletKeyMissing(str(exc)) from exc
    return unsigned, signed


def _broadcast(rpc: EvmRpc, signed: dict[str, Any]) -> dict[str, Any]:
    try:
        tx_hash = rpc.send_raw_transaction(signed["signed_tx"])
    except RpcError as exc:
        raise ChainRpcFailed(str(exc)) from exc
    return _wait_for_receipt(rpc, tx_hash)


def _find_record(records: list[dict[str, Any]], checksum: str, uri: str) -> dict[str, Any] | None:
    """Pick the record matching this checksum (and URI) — usually the latest."""
    matches = [r for r in records if r.get("checksum") == checksum and r.get("uri") == uri]
    if not matches:
        return None
    latest = [r for r in matches if r.get("is_latest")]
    return (latest or matches)[-1]


# --- wallet ------------------------------------------------------------------


def wallet_status(
    ws: Workspace,
    *,
    chain_name: str,
    address: str | None,
    config_path: Path | None,
    service: str = DEFAULT_SERVICE,
    account: str = DEFAULT_ACCOUNT,
) -> dict[str, Any]:
    chain = get_chain(ws, chain_name, config_path)
    addr = _resolve_from(address, service, account)
    rpc, _ = _clients(chain)
    try:
        balance = rpc.get_balance(addr)
        nonce = rpc.get_transaction_count(addr, "pending")
    except RpcError as exc:
        raise ChainRpcFailed(str(exc)) from exc
    return {
        "chain": chain.name,
        "address": addr,
        "balance_wei": str(balance),
        "balance_eth": _wei_to_eth(balance),
        "native_token": chain.native_token,
        "nonce": nonce,
        "funded": balance > 0,
    }


# --- registry ----------------------------------------------------------------


def registry_create(
    ws: Workspace,
    *,
    chain_name: str,
    name: str,
    description: str,
    metadata: str,
    from_address: str | None,
    config_path: Path | None,
    dry_run: bool,
    legacy: bool,
    service: str = DEFAULT_SERVICE,
    account: str = DEFAULT_ACCOUNT,
) -> dict[str, Any]:
    chain = get_chain(ws, chain_name, config_path)
    sender = _resolve_from(from_address, service, account)
    rpc, _ = _clients(chain)
    data = encode_add_registry(name, description, metadata)
    unsigned, signed = _prepare_and_sign(
        chain, rpc, from_address=sender, data=data, service=service, account=account, legacy=legacy
    )
    out: dict[str, Any] = {
        "chain": chain.name,
        "registry": name,
        "from": signed["from"],
        "tx_hash": signed["tx_hash"],
        "broadcast": not dry_run,
    }
    if dry_run:
        out["unsigned_tx"] = unsigned
        out["signed_tx"] = signed["signed_tx"]
        return out
    receipt = _broadcast(rpc, signed)
    out["receipt_status"] = "success"
    out["block_number"] = receipt.get("blockNumber")
    out["explorer_url"] = _explorer_tx_url(chain, signed["tx_hash"])
    return out


def registry_list(
    ws: Workspace,
    *,
    chain_name: str,
    name: str | None,
    config_path: Path | None,
) -> dict[str, Any]:
    chain = get_chain(ws, chain_name, config_path)
    _, anchor = _clients(chain)
    try:
        registries = anchor.get_registries(name=name or "")
    except RpcError as exc:
        raise ChainRpcFailed(str(exc)) from exc
    return {"chain": chain.name, "registries": registries}


# --- stake -------------------------------------------------------------------


def _default_bundle_dir(ws: Workspace, file_id: str, docset_id: str | None) -> Path:
    stem = file_id + (f"-{docset_id}" if docset_id else "")
    return ws.root / "dgmlx-bundles" / stem


def stake_file(
    ws: Workspace,
    *,
    file_id: str,
    docset_id: str | None,
    chain_name: str,
    registry: str,
    from_address: str | None,
    output_dir: Path | None,
    config_path: Path | None,
    dry_run: bool,
    legacy: bool,
    unpacked: bool = False,
    service: str = DEFAULT_SERVICE,
    account: str = DEFAULT_ACCOUNT,
) -> dict[str, Any]:
    chain = get_chain(ws, chain_name, config_path)
    sender = _resolve_from(from_address, service, account)
    rpc, anchor = _clients(chain)

    out_dir = output_dir or _default_bundle_dir(ws, file_id, docset_id)
    # Default writes only the portable <stem>.dgmlx archive into out_dir;
    # --unpacked materializes the loose bundle tree there instead. Exactly one of
    # attestation_path (loose META-INF/dgml-attestation.xml) / archive_path is set,
    # matching the mode. record.json is saved into out_dir either way.
    attestation, attestation_path, archive_path = export_attestation(
        ws, file_id, out_dir, docset_id, unpacked=unpacked
    )
    extra: dict[str, Any] = {
        "kind": "dgmlx",
        "file_id": file_id,
        "docset_id": docset_id,
        "bundle_dir": str(out_dir),
    }
    if unpacked:
        extra["attestation"] = str(attestation_path)
    else:
        extra["dgmlx"] = str(archive_path)
    return _anchor_record(
        rpc,
        anchor,
        chain,
        sender=sender,
        registry=registry,
        uri=build_uri(file_id, docset_id),
        checksum=attestation.root,
        metadata=bundle_metadata(len(attestation.leaves)),
        extra=extra,
        save_dir=out_dir,
        record_name="record.json",
        dry_run=dry_run,
        legacy=legacy,
        service=service,
        account=account,
    )


def stake_node(
    ws: Workspace,
    *,
    file_id: str,
    docset_id: str,
    leaf_index: int | None,
    xpath: str | None,
    chain_name: str,
    registry: str,
    from_address: str | None,
    output_dir: Path | None,
    config_path: Path | None,
    dry_run: bool,
    legacy: bool,
    service: str = DEFAULT_SERVICE,
    account: str = DEFAULT_ACCOUNT,
) -> dict[str, Any]:
    chain = get_chain(ws, chain_name, config_path)
    sender = _resolve_from(from_address, service, account)
    rpc, anchor = _clients(chain)

    att = export_node(ws, file_id, docset_id, leaf_index=leaf_index, xpath=xpath)
    return _anchor_record(
        rpc,
        anchor,
        chain,
        sender=sender,
        registry=registry,
        uri=build_node_uri(file_id, docset_id, att.leaf_index),
        checksum=att.node_hash,
        metadata=node_metadata(att.root_hash, proof_to_json(att.proof)),
        extra={
            "kind": "dgml-node",
            "file_id": file_id,
            "docset_id": docset_id,
            "leaf_index": att.leaf_index,
            "leaf_count": att.leaf_count,
            "xpath": att.xpath,
            "root_hash": att.root_hash,
        },
        save_dir=output_dir or _default_bundle_dir(ws, file_id, docset_id),
        record_name=f"record-node-{att.leaf_index}.json",
        dry_run=dry_run,
        legacy=legacy,
        service=service,
        account=account,
    )


def _anchor_record(
    rpc: EvmRpc,
    anchor: AnchorContract,
    chain: ChainConfig,
    *,
    sender: str,
    registry: str,
    uri: str,
    checksum: str,
    metadata: str,
    extra: dict[str, Any],
    save_dir: Path,
    record_name: str,
    dry_run: bool,
    legacy: bool,
    service: str,
    account: str,
) -> dict[str, Any]:
    """Shared stake spine: encode the addRecord call, sign, and either return
    the unsigned+signed tx (dry run) or broadcast and persist the record.

    ``extra`` carries the granularity-specific fields (bundle vs node); the
    chain/registry/uri/checksum/tx fields are common to both."""
    data = encode_add_record(
        registry=registry,
        uri=uri,
        checksum=checksum,
        checksum_algo=CHECKSUM_ALGO,
        metadata=metadata,
    )
    unsigned, signed = _prepare_and_sign(
        chain, rpc, from_address=sender, data=data, service=service, account=account, legacy=legacy
    )
    out: dict[str, Any] = {
        **extra,
        "chain": chain.name,
        "registry": registry,
        "uri": uri,
        "checksum": checksum,
        "checksum_algo": CHECKSUM_ALGO,
        "from": signed["from"],
        "tx_hash": signed["tx_hash"],
        "broadcast": not dry_run,
    }
    if dry_run:
        out["unsigned_tx"] = unsigned
        out["signed_tx"] = signed["signed_tx"]
        return out
    return _finish_stake(
        rpc, anchor, chain, signed, out, registry, checksum, uri, save_dir, record_name
    )


def _finish_stake(
    rpc: EvmRpc,
    anchor: AnchorContract,
    chain: ChainConfig,
    signed: dict[str, Any],
    out: dict[str, Any],
    registry: str,
    checksum: str,
    uri: str,
    save_dir: Path,
    record_name: str,
) -> dict[str, Any]:
    """Broadcast, confirm, fetch the anchored record, and persist it to disk.

    ``record_name`` keeps bundle and per-node records in distinct files within
    the shared bundle dir so they do not clobber each other.
    """
    receipt = _broadcast(rpc, signed)
    out["receipt_status"] = "success"
    out["block_number"] = receipt.get("blockNumber")
    out["explorer_url"] = _explorer_tx_url(chain, signed["tx_hash"])

    try:
        records = anchor.get_records(registry, checksum=checksum)
    except RpcError as exc:
        raise ChainRpcFailed(str(exc)) from exc
    record = _find_record(records, checksum, uri)
    out["record"] = record
    if record is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        record_path = save_dir / record_name
        from .storage import write_json_atomic

        write_json_atomic(record_path, record)
        out["record_path"] = str(record_path)
    return out


# --- prove -------------------------------------------------------------------


def _load_record_arg(record_json: str) -> dict[str, Any]:
    """Read a record from a path or stdin ('-'); accept a bare record or a
    ``{"records": [...]}`` envelope holding exactly one."""
    import json
    import sys

    loaded: Any
    try:
        if record_json == "-":
            loaded = json.load(sys.stdin)
        else:
            loaded = read_json(Path(record_json))
    except FileNotFoundError as exc:
        raise RecordNotFound(f"record file not found: {record_json}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidArgument(f"cannot read record JSON {record_json!r}: {exc}") from exc
    if isinstance(loaded, dict) and "records" in loaded:
        records = loaded["records"]
        if len(records) != 1:
            raise InvalidArgument(
                f"expected exactly one record in the envelope, got {len(records)}"
            )
        loaded = records[0]
    if not isinstance(loaded, dict) or "checksum" not in loaded or "uri" not in loaded:
        raise InvalidArgument("record is missing 'checksum' or 'uri'")
    record: dict[str, Any] = loaded
    return record


def _fetch_record(anchor: AnchorContract, registry: str, checksum: str) -> dict[str, Any]:
    try:
        records = anchor.get_records(registry, checksum=checksum)
    except RpcError as exc:
        raise ChainRpcFailed(str(exc)) from exc
    if not records:
        raise RecordNotFound(
            f"no record with checksum {checksum} in registry {registry!r} on this chain"
        )
    # Prefer the current version, mirroring _find_record's selection at stake
    # time, so prove verifies against the same record that was anchored.
    latest = [r for r in records if r.get("is_latest")]
    return (latest or records)[-1]


def _resolve_record(
    ws: Workspace,
    *,
    chain_name: str,
    registry: str | None,
    checksum: str | None,
    record_json: str | None,
    config_path: Path | None,
) -> dict[str, Any]:
    if record_json:
        return _load_record_arg(record_json)
    if checksum and registry:
        chain = get_chain(ws, chain_name, config_path)
        _, anchor = _clients(chain)
        return _fetch_record(anchor, registry, checksum)
    raise InvalidArgument("prove needs either --record-json, or both --registry and --checksum")


def _parse_record_uri(uri: str) -> dict[str, Any]:
    """parse_uri, but a malformed URI is a structured INVALID_ARGUMENT."""
    try:
        return parse_uri(uri)
    except ValueError as exc:
        raise InvalidArgument(str(exc)) from exc


def prove_file(
    ws: Workspace,
    *,
    chain_name: str,
    registry: str | None,
    checksum: str | None,
    record_json: str | None,
    config_path: Path | None,
) -> tuple[dict[str, Any], bool]:
    record = _resolve_record(
        ws,
        chain_name=chain_name,
        registry=registry,
        checksum=checksum,
        record_json=record_json,
        config_path=config_path,
    )
    algo = record.get("checksum_algo", CHECKSUM_ALGO)
    if algo != CHECKSUM_ALGO:
        raise InvalidArgument(
            f"record uses checksum_algo {algo!r}; only {CHECKSUM_ALGO!r} is provable"
        )
    ids = _parse_record_uri(record["uri"])
    expected = record["checksum"]
    # Only the Merkle root is needed; attest_file computes it without copying
    # the bundle artifacts to disk (which export_attestation would).
    computed = attest_file(ws, ids["file_id"], ids["docset_id"]).root
    valid = computed == expected
    return {
        "uri": record["uri"],
        "file_id": ids["file_id"],
        "docset_id": ids["docset_id"],
        "checksum_algo": CHECKSUM_ALGO,
        "expected_checksum": expected,
        "computed_checksum": computed,
        "valid": valid,
    }, valid


def prove_node_record(
    ws: Workspace,
    *,
    chain_name: str,
    registry: str | None,
    checksum: str | None,
    record_json: str | None,
    config_path: Path | None,
) -> tuple[dict[str, Any], bool]:
    record = _resolve_record(
        ws,
        chain_name=chain_name,
        registry=registry,
        checksum=checksum,
        record_json=record_json,
        config_path=config_path,
    )
    ids = _parse_record_uri(record["uri"])
    if ids["docset_id"] is None or ids["leaf_index"] is None:
        raise InvalidArgument(f"record URI {record['uri']!r} is not a node URI")
    algo = record.get("checksum_algo", CHECKSUM_ALGO)
    if algo != CHECKSUM_ALGO:
        raise InvalidArgument(
            f"record uses checksum_algo {algo!r}; only {CHECKSUM_ALGO!r} is provable"
        )

    import json

    try:
        meta = json.loads(record.get("metadata") or "{}")
    except json.JSONDecodeError as exc:
        raise InvalidArgument(f"record metadata is not valid JSON: {exc}") from exc
    root_hash, proof_payload = meta.get("root_hash"), meta.get("proof")
    if not root_hash or not isinstance(proof_payload, dict):
        raise InvalidArgument("record metadata lacks 'root_hash'/'proof' (not a dgml-node record?)")
    # The record must be self-consistent before the workspace is consulted.
    if proof_payload.get("leaf_hash") != record["checksum"]:
        raise InvalidArgument("record checksum does not match the metadata proof's leaf_hash")
    if proof_payload.get("leaf_index") != ids["leaf_index"]:
        raise InvalidArgument("record URI #leaf does not match the metadata proof's leaf_index")

    try:
        proof = proof_from_json(proof_payload)
    except ValueError as exc:
        raise InvalidArgument(f"malformed proof in record metadata: {exc}") from exc
    result = prove_node(ws, ids["file_id"], ids["docset_id"], root_hash, proof)
    return {
        "uri": record["uri"],
        "file_id": ids["file_id"],
        "docset_id": ids["docset_id"],
        "leaf_index": ids["leaf_index"],
        "xpath": result.xpath,
        "checksum_algo": CHECKSUM_ALGO,
        "expected_node_hash": result.expected_node_hash,
        "computed_node_hash": result.computed_node_hash,
        "expected_root": result.expected_root,
        "valid": result.valid,
    }, result.valid
