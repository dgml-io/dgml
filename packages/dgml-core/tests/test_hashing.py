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

from __future__ import annotations

import hashlib
from pathlib import Path

from dgml_core.hashing import sha256_file


def test_known_content(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    assert sha256_file(p) == hashlib.sha256(b"hello world").hexdigest()


def test_streaming_large_file(tmp_path: Path) -> None:
    p = tmp_path / "big.bin"
    chunk = b"x" * 4096
    expected = hashlib.sha256()
    with p.open("wb") as fh:
        for _ in range(64):
            fh.write(chunk)
            expected.update(chunk)
    assert sha256_file(p) == expected.hexdigest()
