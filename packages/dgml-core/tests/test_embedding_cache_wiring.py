"""The workspace embedding cache is wired into the clustering config.

Regression cover for the fix that threads a per-workspace ``cache_dir`` into
``run_clustering`` so the ``CachingEncoder`` actually persists embeddings
(previously ``cache_dir`` stayed ``None`` and caching never happened).
"""

from __future__ import annotations

from pathlib import Path

from dgml_core.run_clustering import _build_config
from dgml_core.storage import Workspace


def test_workspace_exposes_embedding_cache_dir(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path / "ws")
    assert ws.embedding_cache_dir == tmp_path / "ws" / ".cache" / "embeddings"


def test_build_config_threads_cache_dir(tmp_path: Path) -> None:
    cache_dir = tmp_path / "ws" / ".cache" / "embeddings"
    cfg = _build_config(
        known_categories=[],
        all_categories_known=False,
        n_samples_per_category=0,
        cache_dir=cache_dir,
    )
    assert cfg.cache_dir == cache_dir


def test_build_config_cache_dir_defaults_none() -> None:
    cfg = _build_config(
        known_categories=[],
        all_categories_known=False,
        n_samples_per_category=0,
    )
    assert cfg.cache_dir is None
