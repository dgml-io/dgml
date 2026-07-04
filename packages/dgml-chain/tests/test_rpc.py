from __future__ import annotations

import io
import json
from typing import Any

import pytest
from dgml_chain.rpc import _USER_AGENT, EvmRpc, RpcError


def test_user_agent_is_not_the_urllib_default() -> None:
    # The public NVNM RPC WAF returns 403 for `Python-urllib/*`; we must send
    # our own UA. This guards against a regression to the default.
    assert not _USER_AGENT.lower().startswith("python-urllib")
    assert _USER_AGENT.startswith("dgml-chain")


def _fake_urlopen(captured: dict[str, Any], result: Any):  # type: ignore[no-untyped-def]
    def _open(req: Any, timeout: int = 0):  # type: ignore[no-untyped-def]
        captured["user_agent"] = req.get_header("User-agent")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return io.BytesIO(json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}).encode())

    return _open


def test_call_sets_user_agent_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(captured, "0xc02a7"))
    rpc = EvmRpc("https://example.test")
    assert rpc.chain_id() == 0xC02A7
    assert captured["user_agent"] == _USER_AGENT
    assert captured["body"]["method"] == "eth_chainId"


def test_rpc_error_envelope_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _open(req: Any, timeout: int = 0):  # type: ignore[no-untyped-def]
        return io.BytesIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"message": "boom"}}).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", _open)
    with pytest.raises(RpcError, match="boom"):
        EvmRpc("https://example.test").gas_price()
