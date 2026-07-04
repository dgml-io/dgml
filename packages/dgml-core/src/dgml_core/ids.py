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
