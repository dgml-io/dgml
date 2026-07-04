"""Data loading: dataset wrapper.

For the low-thousand-document scale we target, everything fits in memory.
Heavy on-disk structures (ANN indexes, tile servers) are intentionally absent.
"""

from __future__ import annotations

from clustering.data.datasets import DocumentDataset, DocumentRecord

__all__ = [
    "DocumentDataset",
    "DocumentRecord",
]
