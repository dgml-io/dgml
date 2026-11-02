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

"""Tests for `dgml_core.errors` helpers."""

from __future__ import annotations

import copy
import inspect
import pickle
from pathlib import Path

import pytest
from dgml_core.errors import (
    ConflictError,
    ConflictErrorKind,
    DgmlError,
    InvalidArgument,
    MissingExtra,
    WorkspaceNotInitialized,
    short_error_message,
)
from dgml_core.storage import Workspace


def test_short_error_message_includes_type_and_text() -> None:
    msg = short_error_message(RuntimeError("network down"))
    assert msg == "RuntimeError: network down"


def test_short_error_message_collapses_whitespace() -> None:
    msg = short_error_message(ValueError("line one\n\n   line two\t  end"))
    assert msg == "ValueError: line one line two end"


def test_short_error_message_truncates_long_text() -> None:
    msg = short_error_message(RuntimeError("x" * 1000))
    assert len(msg) == 300
    assert msg.endswith("...")
    assert msg.startswith("RuntimeError: ")


def test_short_error_message_respects_custom_limit() -> None:
    msg = short_error_message(RuntimeError("x" * 1000), limit=50)
    assert len(msg) == 50
    assert msg.endswith("...")


def test_short_error_message_bare_exception_is_type_name() -> None:
    assert short_error_message(RuntimeError()) == "RuntimeError"


def _keyword_errors() -> list[DgmlError]:
    """Fresh instances per call, so ``_raised`` never mutates a shared one."""
    return [
        WorkspaceNotInitialized("no config", workspace=Workspace(root=Path("/w"))),
        MissingExtra("need it", extra="azure", distribution="azure-identity"),
        MissingExtra("need it", extra="chain"),
        ConflictError("taken", kind=ConflictErrorKind.WORKSPACE, existing_id="acme"),
    ]


def _all_subclasses(cls: type) -> set[type]:
    found: set[type] = set()
    for sub in cls.__subclasses__():
        found.add(sub)
        found |= _all_subclasses(sub)
    return found


def test_every_keyword_error_class_is_covered_by_the_pickle_test() -> None:
    """Pins ``_keyword_errors`` to the classes whose ``__init__`` takes a required
    keyword, so a new one cannot skip the round-trip test below."""
    with_required_keywords = {
        cls
        for cls in _all_subclasses(DgmlError)
        if "__init__" in vars(cls)
        and any(
            p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
            for p in inspect.signature(vars(cls)["__init__"]).parameters.values()
        )
    }
    assert with_required_keywords == {type(e) for e in _keyword_errors()}


def _raised(exc: DgmlError) -> DgmlError:
    try:
        raise exc
    except DgmlError as caught:
        caught.add_note("from a worker")
        return caught


@pytest.mark.parametrize(
    "exc",
    [*_keyword_errors(), InvalidArgument("bad", 42), *map(_raised, _keyword_errors())],
    ids=lambda e: type(e).__name__,
)
def test_errors_survive_pickle_and_copy(exc: DgmlError) -> None:
    """``Exception.__reduce__`` rebuilds via ``cls(*args)``, so without the
    ``DgmlError`` hook the keyword-argument classes arrived from a
    ``ProcessPoolExecutor`` as a ``TypeError`` about the missing keyword. Also
    covers a plain error and ones raised with a traceback and a note."""
    for clone in (pickle.loads(pickle.dumps(exc)), copy.deepcopy(exc)):
        assert type(clone) is type(exc)
        assert clone.args == exc.args
        assert str(clone) == str(exc)
        assert clone.code == exc.code
        assert vars(clone) == vars(exc)
