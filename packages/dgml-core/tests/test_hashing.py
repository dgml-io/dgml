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
