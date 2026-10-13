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

"""The library's logging contract, checked from outside the process.

In-process tests cannot verify the *default-silence* half of the contract:
pytest always attaches its own capturing handlers to the root logger, which
keeps Python's last-resort stderr handler from ever firing — so an in-process
assertion on empty stderr would pass even without the ``NullHandler`` in
``dgml_core/__init__.py``. A subprocess with no logging configured is the
production condition.
"""

from __future__ import annotations

import subprocess
import sys

# Triggers a WARNING-level record (the [models] tier fallback) on
# ``dgml_core.models_config`` without needing a workspace, credentials, or the
# lazy litellm import.
_WARN_SNIPPET = (
    "import dgml_core\n"
    "from dgml_core.models_config import Model, ModelsConfig, Tier\n"
    "ModelsConfig(standard=Model('s')).resolve(Tier.EXPERT)\n"
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )


def test_silent_by_default() -> None:
    """A caller that configures no logging sees nothing — no stdout, no stderr.

    This is the ``NullHandler`` contract: the tier-fallback WARNING is emitted,
    handled by the ``NullHandler`` on ``dgml_core``, and therefore never reaches
    Python's last-resort stderr handler."""
    result = _run(_WARN_SNIPPET)
    assert result.stdout == ""
    assert result.stderr == ""


def test_caller_routes_with_stdlib_logging() -> None:
    """The same event reaches a caller's handler once logging is configured,
    under the documented ``dgml_core.*`` logger name — the routing half of the
    contract (one ``basicConfig`` line, no dgml-specific API)."""
    result = _run(
        "import logging\n"
        "logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s %(message)s')\n"
        + _WARN_SNIPPET
    )
    assert result.stdout == ""
    assert "dgml_core.models_config WARNING" in result.stderr
    assert "falling back to 'standard'" in result.stderr
