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

"""Credential-carrying options never move the storage seal."""

from __future__ import annotations

from pathlib import Path

from dgml_core.storage_resolve import storage_fingerprint
from dgml_core.storage_service import StorageConfig


def _fp(**options: str) -> str:
    return storage_fingerprint(
        StorageConfig(provider="m:Store", root=Path("/r"), options=options, workspace_id="w")
    )


def test_connection_uri_and_aws_credentials_are_outside_the_fingerprint() -> None:
    base = _fp(mongo_database="d")
    assert _fp(mongo_database="d", mongo_uri="mongodb://u:p@a/d") == base
    assert _fp(mongo_database="d", mongo_uri="mongodb://u:p@b/d") == base
    assert _fp(mongo_database="d", aws_access_key_id="A", aws_secret_access_key="S") == base
    assert _fp(mongo_database="other") != base


def test_uri_is_matched_as_a_word_not_a_substring() -> None:
    base = _fp(mongo_database="d")
    assert _fp(mongo_database="d", security_level="high") != base
    assert _fp(mongo_database="d", mongo_uri="mongodb://a/d") == base
