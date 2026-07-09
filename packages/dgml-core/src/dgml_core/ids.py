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

"""ID generation for DocSets and Files."""

from __future__ import annotations

import secrets
import string

ID_ALPHABET = string.ascii_lowercase + string.digits
ID_LENGTH = 12


def new_id() -> str:
    """Return a fresh 12-char base-36 ID (lowercase letters + digits)."""
    return "".join(secrets.choice(ID_ALPHABET) for _ in range(ID_LENGTH))


def is_valid_id(value: str) -> bool:
    if len(value) != ID_LENGTH:
        return False
    return all(c in ID_ALPHABET for c in value)
