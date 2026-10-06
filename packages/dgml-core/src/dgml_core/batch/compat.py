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

"""The litellm versions the built-in batch backends are verified against.

The backends build and decode batch requests with litellm INTERNALS — the
``HTTPHandler.post`` capture seam and each provider's response transformer
(``AnthropicConfig.transform_parsed_response``, and for Gemini
``GoogleAIStudioGeminiConfig._transform_google_generate_content_to_openai_model_response``)
— so that a batch request is byte-for-byte the request the synchronous path
sends. None of that is public API.

That constraint belongs to batch mode alone, so it is enforced here, at
runtime, rather than as a pin on ``dgml-core``'s litellm requirement (which
would hold every consumer of the library to one litellm minor series for a
feature most of them never turn on). Every built-in backend's availability
probe calls :func:`require_supported_litellm`; the registry turns its failure
into :class:`dgml_core.errors.BatchUnavailable` before any request is sent.
Widen :data:`SUPPORTED_LITELLM` only after the backends' wire/decode parity
tests pass on the new series.
"""

from __future__ import annotations

import re
from importlib import metadata

#: The litellm range the batch backends are verified against (a PEP 440
#: specifier, for display; :data:`_LOWER` / :data:`_UPPER` are what is checked).
SUPPORTED_LITELLM = ">=1.85,<1.86"
_LOWER = (1, 85)  # inclusive
_UPPER = (1, 86)  # exclusive
_RELEASE = re.compile(r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?")


class IncompatibleDependency(ImportError):
    """A batch backend's dependency is installed, but at a version it does not
    support. An :class:`ImportError`, so the registry's existing handling turns
    it into :class:`dgml_core.errors.BatchUnavailable` — phrased as a version
    problem rather than a missing package."""


def _release(version: str) -> tuple[int, int, int] | None:
    """``"1.85.1"`` → ``(1, 85, 1)``; pre/post/local suffixes are ignored."""
    match = _RELEASE.match(version)
    if match is None:
        return None
    major, minor, patch = (int(part) if part else 0 for part in match.groups())
    return major, minor, patch


def litellm_version() -> str:
    """The installed litellm distribution's version string."""
    return metadata.version("litellm")


def require_supported_litellm() -> None:
    """Raise :class:`IncompatibleDependency` unless the installed litellm is in
    :data:`SUPPORTED_LITELLM` (an absent litellm raises ``ImportError``)."""
    try:
        installed = litellm_version()
    except metadata.PackageNotFoundError as exc:
        raise ImportError("litellm is not installed") from exc
    release = _release(installed)
    if release is None or not (_LOWER <= release[:2] < _UPPER):
        raise IncompatibleDependency(
            f"litellm {installed} is installed, but the batch backends support "
            f"litellm{SUPPORTED_LITELLM} (they rely on litellm internals that change "
            f"between minor releases). Install a supported version, e.g. "
            f"`pip install 'litellm{SUPPORTED_LITELLM}'`, or run without --batch"
        )
