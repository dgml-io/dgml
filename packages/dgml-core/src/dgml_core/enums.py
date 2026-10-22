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

"""Helpers for the ``StrEnum`` value sets used across dgml_core.

A leaf module (standard library only), so any module can import it without
an import cycle.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

E = TypeVar("E", bound=StrEnum)


def parse_enum(cls: type[E], raw: object) -> E | None:
    """*raw* as a member of *cls*, or ``None`` when it names none.

    The tolerant reader for a persisted value (a manifest field, a usage row, an
    XML attribute): a non-string, or a string this version doesn't know (a
    newer dgml wrote it), is ``None`` rather than an error, so the caller
    decides what to keep. Never raises.
    """
    if not isinstance(raw, str):
        return None
    try:
        return cls(raw)
    except ValueError:
        return None
