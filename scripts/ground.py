#!/usr/bin/env python3
"""Re-ground a file's generated DGML XML in place.

`dgml docset generate` already grounds each `<stem>.dgml.xml` in place
(writes `dg:origin` bounding-box attributes) as the last step of a run.
This script re-runs *just* that grounding pass — no LLM, no regeneration
— against the file's page OCR, so you can pick up a grounding change
without paying to regenerate the docset. Grounding is deterministic and
overwrites the source `<stem>.dgml.xml` in place.

It is intentionally NOT part of the public `dgml` CLI surface; it is a
maintenance/debug tool.

Usage:
    uv run python scripts/ground.py --docset <docset_id> [--file <file_id>]
        [--workspace <path>] [--debug]

    --file   ground only this file (must be assigned to the docset);
             default is every file in the docset.
    --debug  also write the <stem>.dgml.grounding_stats.json sidecar.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dgml_core.docsets import DocSetStore
from dgml_core.errors import DgmlError
from dgml_core.files import FileStore
from dgml_core.storage import Workspace
from dgml_core.xml_grounding import ground_dgml_xml


def _source_xml(ws: Workspace, docset_id: str, file_id: str, stem: str) -> Path | None:
    """The file's generated `<stem>.dgml.xml`: per-(docset, file) dir first,
    then the docset dir (where older runs wrote it)."""
    candidates = [
        ws.docset_file_dir(docset_id, file_id) / f"{stem}.dgml.xml",
        ws.docset_dir(docset_id) / f"{stem}.dgml.xml",
    ]
    return next((c for c in candidates if c.exists()), None)


def ground_docset(ws: Workspace, docset_id: str, *, file_id: str | None, write_stats: bool) -> int:
    ds_store = DocSetStore(ws)
    file_store = FileStore(ws)
    ds_store.get(docset_id)  # raises if the docset is unknown

    file_ids = ds_store.list_files(docset_id)
    if file_id is not None:
        if file_id not in set(file_ids):
            raise SystemExit(f"file '{file_id}' is not assigned to docset '{docset_id}'")
        file_ids = [file_id]
    if not file_ids:
        raise SystemExit(f"docset '{docset_id}' has no files assigned")

    grounded = 0
    for fid in file_ids:
        record = file_store.get(fid)
        stem = Path(record.original_filename).stem
        source = _source_xml(ws, docset_id, fid, stem)
        if source is None:
            print(f"{record.original_filename}: no .dgml.xml — run `dgml docset generate` first")
            continue
        try:
            res = ground_dgml_xml(
                ws, fid, source, output_path=source, force=True, write_stats=write_stats
            )
        except DgmlError as exc:
            print(f"{record.original_filename}: not grounded ({exc})")
            continue
        grounded += 1
        print(
            f"{record.original_filename}: {res.stats['elements_annotated']} element(s), "
            f"{res.stats['matched_token_pct']}% tokens matched → {source}"
        )
    print(f"grounded {grounded}/{len(file_ids)} file(s) in place")
    return grounded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docset", required=True, help="DocSet ID whose DGML XML to ground.")
    parser.add_argument(
        "--file",
        dest="file_id",
        default=None,
        help="Ground only this file (must be assigned to the docset). Default: every file.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace root (overrides $DGML_HOME and the default ./dgml-workspace).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also write the <stem>.dgml.grounding_stats.json sidecar.",
    )
    args = parser.parse_args()
    ws = Workspace.resolve(args.workspace)
    ground_docset(ws, args.docset, file_id=args.file_id, write_stats=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
