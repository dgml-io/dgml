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

"""The capture seam, and keeping the batch package off the default import path.

A backend encodes a request by running it through ``litellm.completion``
against a client that records the outgoing request and never sends it. These
tests use an Anthropic model because that is the route the seam serves; no
backend is involved.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest
from dgml_core.batch._capture import HTTPX_HANDLER, CaptureFailed, capture_sync_request

_REPLY: dict[str, Any] = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude",
    "content": [{"type": "text", "text": ""}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 0, "output_tokens": 0},
}

_KWARGS: dict[str, Any] = {
    "model": "anthropic/claude-haiku-4-5",
    "messages": [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": [{"type": "text", "text": "Say hello."}]},
    ],
    "max_tokens": 16,
}


def test_capture_returns_the_body_litellm_would_post_and_sends_nothing() -> None:
    captured = capture_sync_request(
        _KWARGS, seam=HTTPX_HANDLER, reply=_REPLY, api_key="test-key", expect_path="/v1/messages"
    )
    assert captured.url.endswith("/v1/messages")
    assert captured.body["model"] == "claude-haiku-4-5"
    assert captured.body["max_tokens"] == 16
    assert captured.body["system"] == [{"type": "text", "text": "Answer briefly."}]
    assert captured.body["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Say hello."}]}
    ]
    assert "test-key" not in str(captured.body)  # the key travels as a header only


def test_capture_leaves_the_callers_kwargs_untouched() -> None:
    before = repr(_KWARGS)
    capture_sync_request(_KWARGS, seam=HTTPX_HANDLER, reply=_REPLY, api_key="k")
    assert repr(_KWARGS) == before


def test_capture_refuses_a_request_addressed_elsewhere() -> None:
    with pytest.raises(CaptureFailed, match="not the /v1/other endpoint") as info:
        capture_sync_request(
            _KWARGS, seam=HTTPX_HANDLER, reply=_REPLY, api_key="k", expect_path="/v1/other"
        )
    assert info.value.refused_urls and info.value.refused_urls[0].endswith("/v1/messages")


def test_importing_dgml_core_does_not_import_the_batch_package() -> None:
    """Batch mode is opt-in: the library, its stages and the CLI never import
    ``dgml_core.batch`` (nor its litellm-internals coupling) on their own."""
    code = (
        "import sys, dgml_core, dgml_core.llm, dgml_core.usage, dgml_core.grounded, "
        "dgml_core.generation, dgml_core.classification, dgml.cli; "
        "print(sorted(m for m in sys.modules if m.startswith('dgml_core.batch')))"
    )
    env = {**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"
