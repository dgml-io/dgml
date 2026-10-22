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

"""Tests for :func:`clustering.logging_.factory.build_logger`."""

from __future__ import annotations

import pytest
from clustering.config.schema import LoggerConfig
from clustering.logging_.base import NoopLogger
from clustering.logging_.factory import build_logger


def test_none_builds_a_noop_logger() -> None:
    assert isinstance(build_logger(LoggerConfig(), run_id="r"), NoopLogger)


def test_unknown_name_raises_value_error() -> None:
    # Pydantic rejects an unknown name on construction; ``model_construct``
    # skips that, and the name must still fail as a ValueError, not an
    # AssertionError from the exhaustive dispatch.
    cfg = LoggerConfig.model_construct(name="bogus")
    with pytest.raises(ValueError, match=r"Unknown logger name: 'bogus'"):
        build_logger(cfg, run_id="r")
