"""Torch-compatible Dataset wrappers around a :class:`Corpus`.

We *don't* inherit from ``torch.utils.data.Dataset`` so the class is usable
without importing torch (relevant for tests and for the FastAPI server).
PyTorch's ``DataLoader`` will accept any map-style object exposing
``__len__`` and ``__getitem__``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class DocumentRecord:
    """One record yielded by any :class:`DocumentDataset` implementation."""

    doc_id: str
    label: str | None
    image: Image.Image
    text: str  # OCR text — empty string in Phase 2; populated by OCR pass later.
    thumbnail_path: Path | None


class DocumentDataset(ABC):
    """Map-style lazy dataset of :class:`DocumentRecord` s.

    The scenario pipeline only consumes ``__len__`` and ``__getitem__``;
    subclasses pick the document source (a folder Corpus, a workspace's
    file IDs, a database query, …) and the strategy for materializing
    the first-page image.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> DocumentRecord: ...

    def __iter__(self) -> Iterator[DocumentRecord]:
        for i in range(len(self)):
            yield self[i]
