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

"""Which config keys hold a credential, and tables with those values hidden.

Shared by the reprs of the configuration objects (so a ``Configuration`` can be logged)
and by the storage seal (which leaves credentials out of a store's identity hash).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

#: A key naming a credential carries one of these: ``api_key``,
#: ``aws_secret_access_key``, ``auth_token``. The storage seal matches them as
#: substrings; :func:`is_secret_name` as whole words (``max_tokens`` is a count).
SECRET_HINTS = ("key", "secret", "token", "password", "credential")

#: Matched as a whole ``_``-separated word, not a substring: a connection URI
#: (``mongo_uri``) carries credentials inline, but ``security_…`` is not a secret.
SECRET_WORDS = frozenset({"uri"})

#: What a hidden value prints as.
REDACTED = "***"

#: Words that make a key a secret by name alone: the seal's hints plus the one-word
#: spellings. Redaction only — the seal keeps its own, narrower match, which sealed
#: workspaces depend on.
_SECRET_NAME_WORDS = frozenset(SECRET_HINTS) | {"dsn", "apikey", "passwd", "credentials", "auth"}

#: Keys judged by value: a ``*_url`` / ``*_uri`` is a secret only when it carries
#: ``user:password@`` — ``mongo_uri = "mongodb://u:p@h/db"`` is, the S3
#: ``endpoint_url`` is not.
_URL_WORDS = SECRET_WORDS | {"url"}


def _words(name: str) -> set[str]:
    return set(name.lower().split("_"))


def is_secret_name(name: str) -> bool:
    """Whether a config key names a credential *value* by its name alone. A ``*_env``
    key names the variable holding one, which is not itself a secret."""
    return not name.lower().endswith("_env") and not _SECRET_NAME_WORDS.isdisjoint(_words(name))


def _is_url_with_userinfo(name: str, value: Any) -> bool:
    if _URL_WORDS.isdisjoint(_words(name)) or not isinstance(value, str):
        return False
    try:
        return urlsplit(value).username is not None
    except ValueError:
        return True  # unparseable: assume the worst


def redacted(table: Mapping[str, Any]) -> dict[str, Any]:
    """*table* with every credential value replaced by :data:`REDACTED`, nested tables
    and lists included. ``None`` stays ``None``: unset is not a secret."""
    return {key: _redacted_value(str(key), value) for key, value in table.items()}


def _redacted_value(key: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        return redacted(value)
    if isinstance(value, (list, tuple)):
        return [_redacted_value(key, item) for item in value]
    if value is not None and (is_secret_name(key) or _is_url_with_userinfo(key, value)):
        return REDACTED
    return value
