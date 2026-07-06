"""Deterministic run-id derived from a resolved config object.

Same resolved config → same run_id. Bumping any tracked field (scenario,
fusion, manifold, encoder, seed, etc.) changes the id. This is what
makes ClearML / W&B / parquet artifact collisions predictable.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def run_id_for(resolved_config: dict[str, Any], *, length: int = 12) -> str:
    """Compute a deterministic short hash for a resolved-config dict.

    Args:
        resolved_config: A fully-resolved (no ``???`` Hydra placeholders)
            config dict. Nested dicts and lists are serialised in sorted
            order so reorderings produce the same id.
        length: Number of hex characters to return from the sha256 prefix.
            Default 12 is enough to make collisions effectively impossible
            for the ≤low-thousand runs we anticipate.

    Returns:
        Lowercase hex string of ``length`` characters.
    """
    canonical = json.dumps(
        resolved_config,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]
