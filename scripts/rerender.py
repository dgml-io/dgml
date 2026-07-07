#!/usr/bin/env python3
"""Replay a generation run's cached data into the final dgml format.

The debug cache (written by `dgml docset generate` under the docset's cache/)
holds everything the deterministic back half needs:
  <stem>_blocks.json          Pass-A transcription (flat typed blocks)
  label_<stem>_cNN_raw.json   Pass-B labeling returns (raw model JSON)

Default mode makes NO LLM calls: cached labels are re-applied to the cached
blocks and the dgml is re-rendered directly from them (plus the plain
semantic XML at semantic/<stem>.xml) — so renderer / labeler-application
changes can be re-validated on past runs for free.

With --relabel MODEL, Pass B runs LIVE with the current prompts against the
cached transcription (~1 roster-plan call + 1 labeling call per document)
— the expensive Pass-A window calls are never re-run. Previous labeling
artifacts are moved to cache/label_prev/ first.

Usage:
    uv run python scripts/rerender.py <docset_dir> [--workspace-name NAME]
        [--relabel anthropic/claude-sonnet-4-6]

    <docset_dir> e.g. evaluation/dgml/workspaces/<workspace>/docsets/<id>
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from dgml_core import llm
from dgml_core.generation.blocks import Block, Span
from dgml_core.generation.label import _parse_labels_json, apply_labels, label_documents
from dgml_core.generation.to_semantic import (
    build_header,
    render_dgml,
    render_semantic_xml,
)


def _backup_label_cache(cache: Path) -> None:
    """Move existing Pass-B artifacts aside so a live relabel can't clobber them."""
    artifacts = (
        list(cache.glob("label_*_c*_input.txt"))
        + list(cache.glob("label_*_c*_raw.json"))
        + list(cache.glob("plan_roster_*"))
        + list(cache.glob("concept_roster.json"))
    )
    if not artifacts:
        return
    prev = cache / "label_prev"
    prev.mkdir(exist_ok=True)
    for f in artifacts:
        f.rename(prev / f.name)
    print(f"previous labeling artifacts moved to {prev}")


def rerender(
    docset_dir: Path,
    workspace_name: str | None,
    *,
    relabel_model: str | None = None,
) -> int:
    cache = docset_dir / "cache"
    if not cache.is_dir():
        raise SystemExit(
            f"no cache/ under {docset_dir} — run `dgml docset generate` first "
            "(it writes the per-window cache this script replays)."
        )

    docset_meta = json.loads((docset_dir / "docset.json").read_text(encoding="utf-8"))
    ws_name = workspace_name or docset_dir.parent.parent.name
    header = build_header(ws_name, docset_meta.get("name", ""))

    block_files = sorted(cache.glob("*_blocks.json"))
    if not block_files:
        raise SystemExit(f"no *_blocks.json in {cache}")

    docs: dict[str, list[Block]] = {}
    for blocks_file in block_files:
        stem = blocks_file.name.removesuffix("_blocks.json")
        raw_blocks = json.loads(blocks_file.read_text(encoding="utf-8"))
        docs[stem] = [
            Block(**{**b, "entities": [Span(**sp) for sp in b.get("entities", [])]})
            for b in raw_blocks
        ]

    if relabel_model:
        # Live Pass B with the CURRENT prompts on the cached transcription.
        _backup_label_cache(cache)
        config = llm.LLMConfig(model=relabel_model, temperature=0.0, max_tokens=32000)
        label_documents(docs, config=config, cache_dir=cache, log=print)
    else:
        for stem, blocks in docs.items():
            n_warn = 0
            for label_file in sorted(cache.glob(f"label_{glob.escape(stem)}_c*_raw.json")):
                payload = _parse_labels_json(label_file.read_text(encoding="utf-8"))
                n_warn += len(apply_labels(blocks, payload.get("labels", {}) or {}, doc_name=stem))
            if n_warn:
                print(f"{stem[:48]}: {n_warn} label warning(s)")

    semantic_dir = docset_dir / "semantic"
    semantic_dir.mkdir(exist_ok=True)

    for stem, blocks in docs.items():
        # Final dgml directly from blocks — the conversion, no Pass 4.
        (docset_dir / f"{stem}.dgml.xml").write_text(
            render_dgml(blocks, header=header), encoding="utf-8"
        )
        (semantic_dir / f"{stem}.xml").write_text(render_semantic_xml(blocks), encoding="utf-8")
        labeled = sum(1 for b in blocks if b.concept)
        print(f"{stem[:48]:50s} blocks={len(blocks):4d} labeled={labeled:4d}")

    print(f"converted {len(docs)} file(s) → {docset_dir}/<stem>.dgml.xml")
    return len(docs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docset_dir", type=Path)
    parser.add_argument(
        "--relabel",
        metavar="MODEL",
        default=None,
        help=(
            "Re-run Pass B (semantic labeling) LIVE with this model on the cached "
            "transcription, e.g. anthropic/claude-sonnet-4-6 — a handful of small "
            "calls instead of a full regeneration. Previous labeling artifacts "
            "move to cache/label_prev/."
        ),
    )
    parser.add_argument(
        "--workspace-name",
        default=None,
        help="Namespace label for the dg:chunk header (default: derived from the path).",
    )
    args = parser.parse_args()
    rerender(args.docset_dir, args.workspace_name, relabel_model=args.relabel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
