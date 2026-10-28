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

"""The public surface stays importable, and importing it stays cheap.

`dgml_core.__all__` is the library's contract. Two things must hold: every
name in it resolves, and `import dgml_core` does not load `litellm` — the
~1.4s client that deterministic callers (Workspace/FileStore only) never use.
The second is kept by resolving the LLM-reaching re-exports lazily (PEP 562,
`_LAZY_SUBMODULES`); a name added eagerly from a module that imports `.llm`
breaks it silently, which is what the subprocess test below catches.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import dgml_core
import pytest


def test_every_public_name_resolves() -> None:
    missing = [name for name in dgml_core.__all__ if not hasattr(dgml_core, name)]
    assert not missing, f"names in __all__ with no attribute: {missing}"


def test_lazy_names_are_in_all() -> None:
    not_exported = set(dgml_core._LAZY_SUBMODULES) - set(dgml_core.__all__)
    assert not not_exported, f"lazy names missing from __all__: {not_exported}"


@pytest.mark.parametrize("name", sorted(dgml_core._LAZY_SUBMODULES))
def test_lazy_name_resolves_to_its_submodule(name: str) -> None:
    submodule = importlib.import_module(dgml_core._LAZY_SUBMODULES[name], "dgml_core")
    assert getattr(dgml_core, name) is getattr(submodule, name)


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="no attribute 'no_such_name'"):
        dgml_core.no_such_name  # noqa: B018


def test_import_does_not_load_litellm() -> None:
    # A fresh interpreter: this process has already imported the lazy
    # submodules through the tests above.
    probe = (
        "import sys, dgml_core\n"
        "loaded = sorted(m for m in ('litellm', 'dgml_core.llm', "
        "'dgml_core.classification', 'dgml_core.consistency') if m in sys.modules)\n"
        "print(','.join(loaded))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert out == "", f"`import dgml_core` loaded: {out}"
