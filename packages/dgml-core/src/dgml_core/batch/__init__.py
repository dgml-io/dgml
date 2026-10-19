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

"""Provider-agnostic batch client core.

Providers' asynchronous batch endpoints (submit many completions, poll,
collect at half price) share one shape; :class:`BatchBackend` is that shape.
This package holds the shape, the value types that cross it, the registry
that picks a backend for a model, and a scripted :class:`FakeBackend` for
tests. On top sit the :class:`BatchExecutor` (one wave of requests → one
response each, with resubmission and synchronous fallback) and
:func:`run_stage` (many ``steps_*`` generators driven side by side, wave by
wave).

Registered backends: ``anthropic`` (Message Batches,
:mod:`dgml_core.batch.anthropic`). Nothing in the package is reachable from
the CLI or imported by the rest of ``dgml_core``: importing ``dgml_core``
never imports this package.

Policy: requesting batch mode for a model with no registered backend, or a
backend whose optional dependency is not installed, is an error
(:class:`dgml_core.errors.BatchUnavailable`) raised when the backend is
resolved — before any request is sent. It never silently falls back to a
full-price synchronous call.
"""

from __future__ import annotations

# Built-in backends register themselves on import. Each needs nothing beyond
# what dgml-core already depends on, so none carries an install hint.
from dgml_core.batch.anthropic import AnthropicBatchBackend, AnthropicBatchError
from dgml_core.batch.backend import BatchBackend
from dgml_core.batch.chunking import DEFAULT_OVERHEAD_BYTES, plan_batches, request_size
from dgml_core.batch.compat import SUPPORTED_LITELLM, IncompatibleDependency
from dgml_core.batch.driver import Unit, UnitOutcome, run_stage, run_stage_sync
from dgml_core.batch.executor import TIER_MARKER, BatchExecutor, WaveStats, make_executor
from dgml_core.batch.fake import FakeBackend, fake_model_response
from dgml_core.batch.registry import (
    AvailabilityProbe,
    BackendConfig,
    BackendFactory,
    RequestProbe,
    StageRequest,
    assert_batchable,
    provider_of,
    register_backend,
    registered_providers,
    resolve_backend,
    unregister_backend,
)
from dgml_core.batch.types import (
    BatchItemError,
    BatchJob,
    BatchNotFound,
    BatchRejected,
    BatchRequest,
    BatchState,
    BatchStatus,
    BatchSubmitUncertain,
    BatchThrottled,
    ItemErrorKind,
)

__all__ = [
    "DEFAULT_OVERHEAD_BYTES",
    "SUPPORTED_LITELLM",
    "TIER_MARKER",
    "AnthropicBatchBackend",
    "AnthropicBatchError",
    "AvailabilityProbe",
    "BackendConfig",
    "BackendFactory",
    "BatchBackend",
    "BatchExecutor",
    "BatchItemError",
    "BatchJob",
    "BatchNotFound",
    "BatchRejected",
    "BatchRequest",
    "BatchState",
    "BatchStatus",
    "BatchSubmitUncertain",
    "BatchThrottled",
    "FakeBackend",
    "IncompatibleDependency",
    "ItemErrorKind",
    "RequestProbe",
    "StageRequest",
    "Unit",
    "UnitOutcome",
    "WaveStats",
    "assert_batchable",
    "fake_model_response",
    "make_executor",
    "plan_batches",
    "provider_of",
    "register_backend",
    "registered_providers",
    "request_size",
    "resolve_backend",
    "run_stage",
    "run_stage_sync",
    "unregister_backend",
]
