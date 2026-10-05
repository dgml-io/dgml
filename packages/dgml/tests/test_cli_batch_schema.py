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

"""``--batch`` on ``extraction generate-schema``.

The schema model's provider is replaced by a :class:`FakeBackend` answering
with the same reply the synchronous run's patched ``litellm.completion``
returns, so the two runs' payloads and usage rows compare directly.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from dgml.cli import main
from dgml_core.batch import FakeBackend, fake_model_response, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.docsets import DocSetStore
from dgml_core.storage import Workspace
from dgml_core.usage import TIER_BATCH, TIER_STANDARD, read_events

from .test_cli import _init_ws, _new_docset, _read_stderr, _read_stdout, _seed_file_dir, _ws_args
from .test_cli_batch_extraction import _write_config

_FIELDS = [
    {"name": "vendor_name", "kind": "field", "datatype": "text"},
    {"name": "invoice_date", "kind": "field", "datatype": "date"},
]
_FIDS = ["fschema00001", "fschema00002"]


def _schema_reply(**_kwargs: Any) -> Any:
    return fake_model_response(
        "",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "submit_schema", "arguments": json.dumps({"fields": _FIELDS})},
            }
        ],
        usage={"prompt_tokens": 900, "completion_tokens": 200, "total_tokens": 1100},
        cost=0.05,
    )


def _install(polls: int = 0) -> FakeBackend:
    """Register one fake Anthropic batch backend (the schema model is
    ``anthropic/claude-opus-4-7``). ``polls`` status checks report it still
    running before it ends."""
    fake = FakeBackend(
        lambda req: _schema_reply(**req.kwargs), provider="anthropic", polls_until_ended=polls
    )
    register_backend("anthropic", lambda _cfg: fake)
    return fake


@pytest.fixture
def registry() -> Iterator[None]:
    """Restore the real batch backends after a test replaces one."""
    saved = dict(batch_registry._REGISTRY)
    try:
        yield
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


def _seed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, str]:
    ws = tmp_path / "ws"
    _init_ws(ws)
    capsys.readouterr()
    _write_config(ws)
    ds_id = _new_docset(ws, capsys)
    store = DocSetStore(Workspace(root=ws))
    for fid in _FIDS:
        _seed_file_dir(ws, fid, pages=1)
        store.add_file(ds_id, fid)
    return ws, ds_id


def _rows(ws: Path) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in r.items() if k not in {"at", "duration_s", "tier"}}
        for r in read_events(Workspace(root=ws))
    ]


def test_sync_payload_is_unchanged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws, ds_id = _seed(tmp_path, capsys)
    with patch("litellm.completion", side_effect=_schema_reply):
        assert main(_ws_args(ws) + ["extraction", "generate-schema", ds_id]) == 0
    payload = _read_stdout(capsys)
    assert list(payload) == ["docset_id", "schema_format", "schema", "from_file_ids", "model"]


@pytest.mark.usefixtures("registry")
def test_batch_matches_sync_and_adds_a_batch_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    backend = _install()
    ws, ds_id = _seed(tmp_path, capsys)
    with patch("litellm.completion", side_effect=_schema_reply):
        assert main(_ws_args(ws) + ["--debug", "extraction", "generate-schema", ds_id]) == 0
    sync = _read_stdout(capsys)
    sync_rows = _rows(ws)

    with patch("litellm.completion", side_effect=AssertionError("no sync call expected")):
        assert (
            main(
                _ws_args(ws)
                + ["--debug", "extraction", "generate-schema", ds_id, "--batch"]
                + ["--batch-poll-interval", "0.01"]
            )
            == 0
        )
    batch = _read_stdout(capsys)
    assert {k: v for k, v in batch.items() if k != "batch"} == sync
    assert batch["batch"]["provider"] == "anthropic"
    assert batch["batch"]["requests"] == 1 and batch["batch"]["waves"] == 1
    assert batch["batch"]["sync_fallbacks"] == 0
    assert len(backend.submitted) == 1
    tiers = [r["tier"] for r in read_events(Workspace(root=ws))]
    assert tiers == [TIER_STANDARD, TIER_BATCH]
    assert _rows(ws) == sync_rows * 2  # identical rows apart from the tier


def test_unsupported_provider_is_rejected_before_any_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws, ds_id = _seed(tmp_path, capsys)
    with patch("litellm.completion") as mock_completion:
        rc = main(
            _ws_args(ws)
            + ["extraction", "generate-schema", ds_id, "--batch"]
            + ["--schema-model", "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"]
        )
    assert rc == 1
    err = _read_stderr(capsys)["error"]
    assert err["code"] == "BATCH_UNAVAILABLE"
    assert "schema" in err["message"]
    mock_completion.assert_not_called()
    assert not DocSetStore(Workspace(root=ws)).has_schema(ds_id)
