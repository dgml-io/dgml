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

""":func:`dgml_core.extract_values_many`: several files, sync or batch, each
file's failure its own outcome."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from dgml_core import extract_values_many
from dgml_core.batch import FakeBackend, provider_of, register_backend
from dgml_core.batch import registry as batch_registry
from dgml_core.errors import AuthError, BatchUnavailable
from dgml_core.grounded import ExtractionResult
from dgml_core.storage import Workspace

from .test_extraction_batch import (
    MATCHED,
    ONE_PAGE,
    TWO_PAGES,
    _config,
    _responder,
    _seed_three,
    _snapshot,
)
from .test_grounded import DEFAULT_VALUES_MODEL

_FIDS = [MATCHED, ONE_PAGE, TWO_PAGES]


@pytest.fixture(autouse=True)
def _registry() -> Iterator[None]:
    saved = dict(batch_registry._REGISTRY)
    try:
        yield
    finally:
        batch_registry._REGISTRY.clear()
        batch_registry._REGISTRY.update(saved)


def test_batch_matches_sync(tmp_path: Path, capture_kwargs: Any) -> None:
    ws_s = Workspace(root=tmp_path / "sync")
    ws_s.root.mkdir(parents=True)
    ds_s = _seed_three(ws_s)
    capture_kwargs(_responder())
    sync = extract_values_many(ws_s, ds_s, _FIDS, config=_config(), write_stats=True, debug=True)
    assert sync.batch is None
    assert all(isinstance(o, ExtractionResult) for o in sync.outcomes.values())

    ws_b = Workspace(root=tmp_path / "batch")
    ws_b.root.mkdir(parents=True)
    ds_b = _seed_three(ws_b)
    respond = _responder()
    backend = FakeBackend(lambda req: respond(req.kwargs))
    register_backend(provider_of(DEFAULT_VALUES_MODEL), lambda _cfg: backend)
    many = extract_values_many(
        ws_b,
        ds_b,
        _FIDS,
        config=_config(),
        batch=True,
        poll_interval_s=0,
        write_stats=True,
        debug=True,
    )
    assert list(many.outcomes) == _FIDS
    assert many.batch is not None and many.batch["provider"] == backend.provider
    assert len(backend.submitted) == 2  # phase 1, then phase 3
    assert _snapshot(ws_b, ds_b, _FIDS) == _snapshot(ws_s, ds_s, _FIDS)


def test_unset_key_env_runs_sync_and_reports_per_file(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds_id = _seed_three(workspace)
    monkeypatch.delenv("DGML_TEST_UNSET_VALUES_KEY", raising=False)
    config = replace(_config(), values_api_key_env="DGML_TEST_UNSET_VALUES_KEY")
    backend = FakeBackend(lambda _req: pytest.fail("no request expected"))
    register_backend(provider_of(DEFAULT_VALUES_MODEL), lambda _cfg: backend)
    many = extract_values_many(workspace, ds_id, _FIDS, config=config, batch=True)
    assert many.batch is None and backend.submitted == []
    assert all(isinstance(o, AuthError) for o in many.outcomes.values())


def test_a_model_with_no_batch_backend_is_rejected_up_front(workspace: Workspace) -> None:
    ds_id = _seed_three(workspace)
    config = replace(_config(), values_model="ollama/llama3")
    with pytest.raises(BatchUnavailable):
        extract_values_many(workspace, ds_id, _FIDS, config=config, batch=True)
