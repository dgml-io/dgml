"""Prototype: cluster with Qwen3-VL-Embedding + an MRL dimension sweep.

Standalone (not wired into the package) end-to-end demo of the option added in
``docs/mrl-qwen3-vl.md``:

1. Build a Qwen3-VL-Embedding encoder — 2B or 8B, ``local`` or ``server``.
2. Embed a handful of inputs (built-in text demo, or the first-page renders of
   a DGML workspace) *once* at full width.
3. Run :func:`clustering.encoders.mrl.mrl_dimension_sweep` over several MRL
   prefix widths and report the cheapest width that clusters well.

Examples::

    # Text demo, local 2B checkpoint (needs the weights + ideally a GPU):
    python -m clustering.examples.mrl_qwen3_vl_clustering --size 2b

    # Page images from a DGML workspace, served over HTTP (no local weights):
    python -m clustering.examples.mrl_qwen3_vl_clustering \\
        --backend server --base-url http://localhost:8000/v1 \\
        --workspace /path/to/dgml-workspace

The heavy bits (model load / HTTP) only happen at run time; importing this
module is cheap.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from clustering.config.schema import EncoderConfig
from clustering.encoders import build_encoder
from clustering.encoders.mrl import mrl_dimension_sweep

_MODELS = {"2b": "Qwen/Qwen3-VL-Embedding-2B", "8b": "Qwen/Qwen3-VL-Embedding-8B"}

# A tiny, obviously-clustered text demo: invoices, NDAs, and resumes.
_DEMO_TEXTS: list[str] = [
    "Invoice #1042. Amount due: $3,200. Net 30. Remit to Acme LLC.",
    "Invoice #1043. Total payable $980 within thirty days of receipt.",
    "Invoice #1051. Balance $12,750 due upon delivery of goods.",
    "Mutual Non-Disclosure Agreement between the parties, effective 2026.",
    "This NDA governs confidential information exchanged during discussions.",
    "Confidentiality Agreement: recipient shall not disclose trade secrets.",
    "Jane Doe — Software Engineer. Skills: Python, distributed systems.",
    "Curriculum Vitae of John Smith, data scientist, 8 years experience.",
    "Resume: backend developer, Go and Kubernetes, seeking new role.",
]


def _build(size: str, backend: str, base_url: str | None, dim: int) -> Any:
    name = "qwen3_vl_embedding_2b" if size == "2b" else "qwen3_vl_embedding"
    extra: dict[str, Any] = {"backend": backend}
    if backend == "server":
        if not base_url:
            raise SystemExit("--base-url is required with --backend server")
        extra["base_url"] = base_url
    cfg = EncoderConfig(
        name=name,  # type: ignore[arg-type]
        model_id=_MODELS[size],
        embedding_dim=dim,
        extra=extra,
    )
    return build_encoder(cfg, device="auto")


def _workspace_images(workspace: Path, limit: int) -> list[Any]:
    from PIL import Image

    files_root = workspace / "files"
    images: list[Any] = []
    for file_dir in sorted(p for p in files_root.iterdir() if p.is_dir()):
        page1 = file_dir / "page_images" / "page_1.png"
        if page1.exists():
            images.append(Image.open(page1).convert("RGB"))
        if len(images) >= limit:
            break
    if not images:
        raise SystemExit(f"No page_images/page_1.png found under {files_root}")
    return images


def _encode_all(encoder: Any, inputs: Sequence[Any], batch: int = 8) -> np.ndarray:
    rows: list[np.ndarray] = []
    for i in range(0, len(inputs), batch):
        out = encoder.encode(list(inputs[i : i + batch]))
        rows.append(out.pooled.numpy())
    return np.vstack(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", choices=["2b", "8b"], default="2b")
    ap.add_argument("--backend", choices=["local", "server"], default="local")
    ap.add_argument("--base-url", default=None, help="Server backend endpoint (…/v1).")
    ap.add_argument("--workspace", type=Path, default=None, help="Embed page images from here.")
    ap.add_argument("--full-dim", type=int, default=1024, help="Width to embed at first.")
    ap.add_argument("--sweep", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--k", type=int, default=3, help="k for the k-means used in the sweep.")
    args = ap.parse_args()

    encoder = _build(args.size, args.backend, args.base_url, args.full_dim)

    if args.workspace is not None:
        inputs: list[Any] = _workspace_images(args.workspace, limit=60)
        print(f"Embedding {len(inputs)} page images with Qwen3-VL-Embedding-{args.size.upper()}…")
    else:
        inputs = list(_DEMO_TEXTS)
        print(f"Embedding {len(inputs)} demo texts with Qwen3-VL-Embedding-{args.size.upper()}…")

    embeddings = _encode_all(encoder, inputs)
    print(f"Full-width embeddings: {embeddings.shape}")

    from sklearn.cluster import KMeans

    def cluster_fn(feats: np.ndarray) -> np.ndarray:
        labels: np.ndarray = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit_predict(feats)
        return labels

    result = mrl_dimension_sweep(embeddings, args.sweep, cluster_fn)
    print("\nMRL dimension sweep (cosine silhouette; higher = better):")
    for d, s in zip(result.dims, result.scores, strict=True):
        marker = "  <- best" if d == result.best_dim else ""
        print(f"  dim {d:>5}:  {s:+.4f}{marker}")
    print(f"\nBest width: {result.best_dim}  (score {result.best_score:+.4f})")
    print(f"Labels @ best width: {result.best_labels.tolist()}")


if __name__ == "__main__":
    main()
