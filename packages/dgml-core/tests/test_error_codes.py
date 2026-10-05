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

"""The error-code registry and its documentation must not drift apart.

An `error.code` is part of the CLI's public contract, so a code that exists in
neither place it should is a real defect: an undocumented code leaves callers
branching on an identifier nothing describes, and a documented code with no
class is either a phantom (the class was renamed and the row stayed) or a
string literal somebody will later fail to find.

Both directions had drifted before these tests existed — eight class codes were
undocumented, and `MANIFEST_INVALID` was documented for a path that had raised
`ATTESTATION_INVALID` for some time.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import dgml_core.errors as errors_module
from dgml_core.errors import DgmlError

#: The **master** error-code table in the CLI reference — the one under the
#: "Error code reference" heading, not the per-command tables that also list a
#: code or two. Rows look like ``| `CODE` | hard | Meaning. |``.
_DOC = Path(__file__).resolve().parents[3] / "docs" / "cli-reference.md"
_HEADING = "## Error code reference"
_ROW = re.compile(r"^\| `([A-Z][A-Z0-9_]*)` \| (?:hard|soft|hard / soft) \|", re.M)

#: Codes the **CLI** emits that deliberately have no class in `dgml_core.errors`.
#: Keeping this list short is the point — see the module docstring in
#: `dgml_core/errors.py` for why each one belongs to the command line rather
#: than the library, and which are only here until the operation that raises
#: them moves into `dgml_core`.
CLI_ONLY_CODES = frozenset(
    {
        # Genuinely CLI-layer: exists for exceptions that are *not* a DgmlError,
        # so a library that raised it would contradict itself.
        "INTERNAL_ERROR",
        # Domain preconditions currently checked in `cli.py` because the
        # operations that check them live there. These become real classes when
        # `extraction` moves into `dgml_core` (as EMPTY_DOCSET became
        # `EmptyDocSet` when `docset generate` did).
        "NO_FILES",
        "VALUES_NOT_FOUND",
    }
)


def _documented_codes() -> set[str]:
    text = _DOC.read_text(encoding="utf-8")
    start = text.index(_HEADING)  # IndexError here means the heading was renamed
    end = text.find("\n## ", start + len(_HEADING))
    return set(_ROW.findall(text[start : end if end != -1 else len(text)]))


def _class_codes() -> dict[str, str]:
    """``code -> class name`` for every error class in ``dgml_core.errors``."""
    found: dict[str, str] = {}
    for name, obj in vars(errors_module).items():
        if inspect.isclass(obj) and issubclass(obj, DgmlError):
            found.setdefault(obj.code, name)
    return found


def test_every_class_code_is_documented() -> None:
    """A new error class must bring its documentation row with it."""
    undocumented = {
        code: cls for code, cls in _class_codes().items() if code not in _documented_codes()
    }
    assert not undocumented, (
        "these error classes have no row in docs/cli-reference.md's error-code table: "
        f"{sorted(undocumented.items())}"
    )


def test_every_documented_code_resolves() -> None:
    """A documented code must name a real class, or be an allowlisted CLI code.

    Catches the phantom left behind by a rename: the row keeps describing real
    behaviour while naming a code nothing can emit any more.
    """
    orphans = _documented_codes() - set(_class_codes()) - CLI_ONLY_CODES
    assert not orphans, (
        "these codes are documented but have no class in dgml_core.errors and are not in "
        f"CLI_ONLY_CODES: {sorted(orphans)}"
    )


def test_cli_only_codes_really_have_no_class() -> None:
    """Keeps the allowlist honest — an entry that gains a class must leave it,
    or the class's own documentation stops being checked."""
    redundant = CLI_ONLY_CODES & set(_class_codes())
    assert not redundant, (
        f"these are in CLI_ONLY_CODES but now have a class; drop them from the list: "
        f"{sorted(redundant)}"
    )


def test_codes_are_unique_per_class() -> None:
    """Two classes sharing a code makes the code ambiguous to branch on."""
    seen: dict[str, list[str]] = {}
    for name, obj in vars(errors_module).items():
        if inspect.isclass(obj) and issubclass(obj, DgmlError) and "code" in obj.__dict__:
            seen.setdefault(obj.code, []).append(name)
    duplicated = {code: names for code, names in seen.items() if len(names) > 1}
    assert not duplicated, f"error codes declared by more than one class: {duplicated}"


def test_every_error_class_is_exported() -> None:
    """The whole hierarchy is public surface.

    A code is already contract through the CLI, so a library caller branching on
    the same condition should be able to `except` the type instead of matching
    the string — which only works if the type is importable from `dgml_core`.
    Catching this at test time is the difference between adding one line to
    `__all__` now and a consumer reaching into `dgml_core.errors` forever.
    """
    import dgml_core

    exported = set(dgml_core.__all__)
    unexported = sorted(
        name
        for name, obj in vars(errors_module).items()
        if inspect.isclass(obj) and issubclass(obj, DgmlError) and name not in exported
    )
    assert not unexported, (
        f"these error classes are not exported from dgml_core.__init__: {unexported}"
    )
