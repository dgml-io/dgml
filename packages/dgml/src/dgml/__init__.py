"""DGML CLI package.

This package provides the ``dgml`` command-line interface only. The library —
the PDF→DGML pipeline, OCR, page rendering, generation, grounding, attestation,
and workspace storage — lives in the ``dgml-core`` distribution and is imported
as ``dgml_core``. Library users should ``import dgml_core`` (not ``dgml``).
"""

from __future__ import annotations

__version__ = "0.1.1"

__all__ = ["__version__"]
