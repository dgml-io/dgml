"""Content hashing for files."""

from __future__ import annotations

import hashlib
from pathlib import Path

CHUNK_SIZE = 1 << 16


def sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of the file at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()
