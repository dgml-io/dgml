# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cluster the files in a DGML workspace.

A runnable end-to-end example: walk a workspace on disk, build a
:class:`DocumentDataset` over its files, pick the right scenario based
on what DocSets already exist, and print ``{doc_id: cluster_name}``.

The workspace layout this script consumes is described in
``docs/storage-layout.md`` — in particular:

    <workspace>/
    ├── docsets/<docset_id>/
    │   ├── docset.json                # { id, name, description, ... }
    │   └── files/<file_id>/           # marker: file is assigned to this docset
    └── files/<file_id>/
        ├── file.json                  # { id, original_filename, page_count, ... }
        ├── page_images/page_1.png     # 300 dpi render of page 1
        └── page_text/page_N.json      # word boxes (one per page)

We treat each file as one document, use ``page_1.png`` as the visual
input, and concatenate ``page_text/*.json`` words into the text input.
A file's "label" — used as supervision by S2/S3/S5 — is the *name* of
the DocSet it's already assigned to, or ``None`` if it's unassigned.

Run it::

    python -m clustering.example /path/to/dgml-workspace
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from clustering.data import DocumentDataset, DocumentRecord
from clustering.scenarios import build_scenario


# ── DocumentDataset over a DGML workspace ───────────────────────────────────
class WorkspaceDataset(DocumentDataset):
    """Walks ``<workspace>/files/`` and yields one record per file.

    Records carry the file's id as ``doc_id``, the first-page render as
    ``image``, the concatenated word text from ``page_text/`` as ``text``,
    and the assigned DocSet's *name* as ``label`` (or ``None`` if the
    file is not assigned to any DocSet).
    """

    def __init__(self, workspace: Path, *, only_unassigned: bool = False) -> None:
        self.workspace = workspace
        self._assignments = _load_file_to_docset_name(workspace)
        files_root = workspace / "files"
        if not files_root.is_dir():
            raise FileNotFoundError(
                f"No files/ directory under {workspace} — is this a DGML workspace?"
            )
        file_ids = sorted(p.name for p in files_root.iterdir() if p.is_dir())
        if only_unassigned:
            file_ids = [fid for fid in file_ids if fid not in self._assignments]
        self._file_ids = file_ids

    def __len__(self) -> int:
        return len(self._file_ids)

    def __getitem__(self, index: int) -> DocumentRecord:
        file_id = self._file_ids[index]
        file_dir = self.workspace / "files" / file_id
        page1 = _page_one_image(file_dir)
        if page1 is None:
            raise FileNotFoundError(
                f"{file_id}: page_images/page_1.* is missing — "
                "ingest may have failed; run `dgml check`."
            )
        return DocumentRecord(
            doc_id=file_id,
            label=self._assignments.get(file_id),
            image=Image.open(page1).convert("RGB"),
            text=_concat_page_text(file_dir),
            thumbnail_path=page1,
        )


def _page_one_image(file_dir: Path) -> Path | None:
    """Return the page-1 image for a workspace file, or None if absent.

    The DGML renderer emits ``page_images/page_<n>.png`` (canonical); older
    workspaces may carry ``.jpg``. Resolve by extension-agnostic glob rather
    than hardcoding a suffix so this tracks the renderer's format.
    """
    matches = sorted((file_dir / "page_images").glob("page_1.*"))
    return matches[0] if matches else None


def _concat_page_text(file_dir: Path) -> str:
    """Concatenate words from every ``page_text/page_N.json`` for one file."""
    return _build_text(file_dir, view="full")


# Text "views" select which words from the OCR/word-box layer feed the text
# encoder. Document *type* is usually declared in the title and section/column
# headers (large-font or top-of-page words) — but in the flat ``full`` view
# those few tokens are drowned by the numeric table body, which is why visually
# similar financial docs (balance sheet / financial statement / rent roll)
# collapse together. The structure-aware views recover that signal using the
# per-word bounding boxes (``l = [x0, y0, x1, y1]``) already stored alongside
# each word — font size is proxied by box height (``y1 - y0``).
TextView = str  # "full" | "page1" | "headers" | "salient_boost"
_SALIENT_BOOST_REPEAT = 3  # times salient text is repeated ahead of the body


@dataclass(frozen=True)
class _Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class _Page:
    height: float | None
    words: list[_Word]


def _load_pages(file_dir: Path) -> list[_Page]:
    """Parse every ``page_text/page_N.json`` into typed :class:`_Page` records."""
    page_text_dir = file_dir / "page_text"
    if not page_text_dir.is_dir():
        return []
    pages: list[_Page] = []
    for p in sorted(page_text_dir.glob("page_*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        words: list[_Word] = []
        for w in payload.get("words", []):
            if not (isinstance(w, dict) and "t" in w):
                continue
            box = w.get("l")
            if not (isinstance(box, list) and len(box) == 4):
                continue
            x0, y0, x1, y1 = (float(v) for v in box)
            words.append(_Word(str(w["t"]), x0, y0, x1, y1))
        raw_h = payload.get("height")
        height = float(raw_h) if isinstance(raw_h, (int, float)) else None
        pages.append(_Page(height=height, words=words))
    return pages


def _salient_words(pages: list[_Page]) -> list[str]:
    """Words that look like titles / headers: large font (tall box) on any page,
    plus everything in the top band of page 1 (the document title block).

    Falls back to an empty list when a page has no usable boxes; callers decide
    what to do with an empty result (typically: use the full text instead).
    """
    salient: list[str] = []
    for page_idx, page in enumerate(pages):
        if not page.words:
            continue
        heights = sorted(w.height for w in page.words)
        # 80th-percentile box height ⇒ "large font". Guard the degenerate case
        # where every word is the same size (a pure table) so we don't tag all.
        p80 = heights[min(len(heights) - 1, int(0.8 * len(heights)))]
        median = heights[len(heights) // 2]
        large_thr = p80 if p80 > median * 1.15 else float("inf")
        top_band = 0.15 * page.height if page.height else 0.0
        for w in page.words:
            # Document-type signal lives in words, not figures: a big-font number
            # is a table value or page number, not a header. Require at least one
            # letter so salient text stays titles / section + column labels.
            if not any(ch.isalpha() for ch in w.text):
                continue
            is_large = w.height >= large_thr
            is_title = page_idx == 0 and top_band > 0 and w.y0 <= top_band
            if is_large or is_title:
                salient.append(w.text)
    return salient


def _build_text(file_dir: Path, *, view: TextView = "full") -> str:
    """Assemble the text input for one file under the requested ``view``.

    - ``full``: every word, every page, in reading order (the original behavior).
    - ``page1``: only the first page (where the type is usually declared).
    - ``headers``: only the salient title/header words (empty ⇒ falls back to full).
    - ``salient_boost``: salient words repeated ahead of the full body, so the
      type tokens dominate the mean-pooled embedding without losing the body.
    """
    pages = _load_pages(file_dir)
    if not pages:
        return ""
    if view == "page1":
        return " ".join(w.text for w in pages[0].words)

    full = " ".join(w.text for page in pages for w in page.words)
    if view == "full":
        return full

    salient = _salient_words(pages)
    if not salient:
        return full  # no layout signal — degrade gracefully to the body text
    salient_text = " ".join(salient)
    if view == "headers":
        return salient_text
    if view == "salient_boost":
        return " ".join([salient_text] * _SALIENT_BOOST_REPEAT + [full])
    raise ValueError(f"unknown text view {view!r}")


def _load_file_to_docset_name(workspace: Path) -> dict[str, str]:
    """Build ``{file_id: docset_name}`` for every assigned file.

    Reads ``<workspace>/docsets/*/docset.json`` for the name and the
    ``files/<file_id>/`` marker directories for the assignment.
    """
    docsets_root = workspace / "docsets"
    if not docsets_root.is_dir():
        return {}
    out: dict[str, str] = {}
    for docset_dir in sorted(p for p in docsets_root.iterdir() if p.is_dir()):
        meta_path = docset_dir / "docset.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = meta.get("name")
        if not isinstance(name, str):
            continue
        assignments_dir = docset_dir / "files"
        if not assignments_dir.is_dir():
            continue
        for marker in assignments_dir.iterdir():
            if marker.is_dir():
                out[marker.name] = name
    return out


# ── Driver ──────────────────────────────────────────────────────────────────
def cluster_workspace(workspace: Path) -> dict[str, str]:
    """Cluster the files in ``workspace`` and return ``{file_id: name}``.

    Scenario is picked from the workspace state:

    * No DocSets exist → **S1** unsupervised — emits ``cluster_N`` labels.
    * One or more DocSets exist → **S2** partial-labels — assigned files
      seed prototypes; unassigned ones are predicted into a known name
      or a new ``unknown_N`` bucket.
    """
    known_categories = sorted(set(_load_file_to_docset_name(workspace).values()))

    dataset = WorkspaceDataset(workspace)
    if len(dataset) == 0:
        return {}

    # Build a Config in code rather than via Hydra — keeps the example
    # dependency-free. Dims match the chosen encoders.
    from clustering.config.schema import (
        Config,
        CorpusConfig,
        EncoderConfig,
        FusionConfig,
        LoggerConfig,
        ManifoldConfig,
        ScenarioConfig,
        TrainingConfig,
    )

    if known_categories:
        scenario_cfg = ScenarioConfig(name="s2", known_categories=known_categories)
    else:
        scenario_cfg = ScenarioConfig(name="s1")

    config = Config(
        scenario=scenario_cfg,
        encoder_text=EncoderConfig(
            name="st_minilm",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dim=384,
        ),
        encoder_image=EncoderConfig(
            name="vit",
            model_id="google/vit-base-patch16-224",
            embedding_dim=768,
        ),
        fusion=FusionConfig(name="late_concat", output_dim=256),
        manifold=ManifoldConfig(name="euclidean", dim=256),
        training=TrainingConfig(epochs=0, batch_size=8),
        logger=LoggerConfig(name="none"),
        # corpus.root is required by the schema but unused here — we pass
        # the dataset to the scenario directly.
        corpus=CorpusConfig(root=workspace / "files"),
        device="auto",
        seed=0,
    )

    scenario = build_scenario(config)
    result = scenario.fit_predict(dataset)

    return {
        doc_id: pred
        for doc_id, pred in zip(result.doc_ids, result.predictions, strict=True)
        if pred is not None
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clustering.example",
        description="Cluster the files in a DGML workspace and print {file_id: cluster_name}.",
    )
    parser.add_argument(
        "workspace",
        type=Path,
        help="Path to a DGML workspace (the directory with files/ and docsets/).",
    )
    args = parser.parse_args(argv)

    if not args.workspace.is_dir():
        print(f"error: {args.workspace} is not a directory", file=sys.stderr)
        return 1

    assignments = cluster_workspace(args.workspace)
    json.dump(assignments, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
