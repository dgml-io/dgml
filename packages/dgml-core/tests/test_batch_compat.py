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

"""The runtime litellm-version gate for the batch backends.

``dgml-core`` accepts a wide litellm range; only batch mode depends on litellm
internals, so only batch mode checks the version — through a backend's
availability probe, which the registry turns into BATCH_UNAVAILABLE.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from dgml_core.batch import (
    SUPPORTED_LITELLM,
    BackendConfig,
    FakeBackend,
    assert_batchable,
    register_backend,
    resolve_backend,
    unregister_backend,
)
from dgml_core.batch import compat as compat_mod
from dgml_core.batch.compat import IncompatibleDependency, require_supported_litellm
from dgml_core.errors import BatchUnavailable

_MODEL = "anthropic/claude-haiku-4-5"


@pytest.fixture
def gated_backend() -> Iterator[None]:
    """A backend registered the way a built-in one is: its availability probe
    is the litellm version gate."""

    def factory(cfg: BackendConfig) -> FakeBackend:
        return FakeBackend({}, provider="anthropic")

    register_backend("anthropic", factory, available=require_supported_litellm)
    yield
    unregister_backend("anthropic")


@pytest.mark.parametrize("version", ["1.85.0", "1.85.1", "1.85.99", "1.85.2rc1", "1.85.3.post1"])
def test_supported_versions_pass(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    monkeypatch.setattr(compat_mod, "litellm_version", lambda: version)
    require_supported_litellm()


@pytest.mark.parametrize("version", ["1.84.9", "1.86.0", "2.0.0", "1.51.0", "not-a-version"])
def test_unsupported_versions_fail(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    monkeypatch.setattr(compat_mod, "litellm_version", lambda: version)
    with pytest.raises(IncompatibleDependency) as info:
        require_supported_litellm()
    assert isinstance(info.value, ImportError)  # what the registry translates
    assert version in str(info.value)
    assert SUPPORTED_LITELLM in str(info.value)


def test_the_installed_litellm_is_supported() -> None:
    """The lockfile's litellm must be one the backends support, or every
    batch test in this suite is testing a mode users cannot turn on."""
    require_supported_litellm()


def test_a_gated_backend_rejects_an_unsupported_litellm(
    monkeypatch: pytest.MonkeyPatch, gated_backend: None
) -> None:
    monkeypatch.setattr(compat_mod, "litellm_version", lambda: "1.99.0")
    with pytest.raises(BatchUnavailable) as preflight:
        assert_batchable({"transcribe": _MODEL})
    message = str(preflight.value)
    assert "stage 'transcribe'" in message
    assert "litellm 1.99.0 is installed" in message
    assert SUPPORTED_LITELLM in message
    assert "not installed" not in message  # a version problem, not a missing package
    with pytest.raises(BatchUnavailable, match=r"litellm 1\.99\.0 is installed"):
        resolve_backend(_MODEL, api_key="k")


def test_a_gated_backend_resolves_on_the_supported_litellm(gated_backend: None) -> None:
    assert resolve_backend(_MODEL).provider == "anthropic"
