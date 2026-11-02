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

"""Workspace-backed :class:`DocumentDataset` implementations.

Bridges between dgml's per-file storage layout and the clustering
package's dataset contract. The clustering scenarios only consume
``__len__`` / ``__getitem__`` over :class:`DocumentRecord` s; concrete
sourcing (folder Corpus, workspace file IDs, …) is the dataset class's
job.
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from clustering.data.datasets import DocumentDataset, DocumentRecord
from PIL import Image

from . import layout
from .storage import Workspace
from .utils import gather_file_pages


@contextmanager
def _file_text_dir(workspace: Workspace, file_id: str, text_view: str) -> Iterator[Path]:
    """A local directory holding one file's ``page_text/``, for ``_build_text``.

    ``_build_text`` takes a *file* directory and opens only
    ``<file_dir>/page_text/page_*.json`` (``clustering.example._load_pages``), so
    that subtree is all that has to exist locally. ``files/<id>/`` also holds the
    source document and every rendered page image — the bulk of a workspace, and
    nothing this reader opens. Materializing the whole prefix is free on
    ``LocalStore`` and a full per-record download on every other backend; the page
    images in it would also duplicate the ones ``__getitem__`` already fetched
    individually.

    Which pages ``text_view`` needs is :func:`dgml_core.utils.page_text_keys`'s
    call — shared with :func:`dgml_core.clustering._corpus_dir`, which does the
    same job for a whole corpus. Only the directory shape differs, and that is
    what stays here.
    """
    from .storage_local import LocalStore

    # Exact type, not ``isinstance``: the passthrough is only valid because
    # ``LocalStore``'s keys *are* paths under the workspace root. A subclass has
    # changed something — ``DefaultBridgeStore`` in the test suite subclasses it
    # precisely to keep the primitives but take the download bridge — so the
    # assumption no longer holds. Materializing for an unknown subclass is slower
    # and correct; short-circuiting it would be fast and wrong.
    if type(workspace.blobs) is LocalStore:
        yield workspace.files_dir / file_id
        return

    from .utils import page_text_keys

    with tempfile.TemporaryDirectory(prefix="dgml-file-text-") as tmp:
        root = Path(tmp)
        # Created even when nothing matches, so ``_load_pages`` sees the same
        # shape it sees on local disk (an empty dir, not a missing one).
        page_text = root / layout.PAGE_TEXT_DIR
        page_text.mkdir()
        prefix = layout.file_text_prefix(file_id)
        for key in page_text_keys(workspace, file_id, text_view):
            workspace.blobs.download_blob(key, page_text / key[len(prefix) :])
        yield root


class WorkspaceFileDataset(DocumentDataset):
    """Lazy :class:`DocumentDataset` over a list of dgml file IDs.

    The first-page image for each file comes from the pre-rendered
    ``<file>/page_images/page_1.{png,jpg}`` that ``dgml file add`` produced —
    no re-rendering happens at cluster time. ``text`` is assembled from
    the file's ``page_text/`` JSON under ``text_view`` (the same word-box
    → text logic the eval corpus uses); pass ``text_view`` to match the
    view the configured text encoder expects. Constructing the dataset
    reads nothing; image *and* text loading happen lazily in
    ``__getitem__``.

    ``labels`` is an optional ``{file_id: category}`` map used to build
    labeled support sets for the few-shot scenarios (S3 / S5). When
    omitted, every record's ``label`` is ``None`` — the right default
    for the unassigned-file (unknown) dataset.

    Callers are expected to filter file IDs whose page images are
    missing *before* handing the list here — :func:`dgml.clustering.clustering_internal`
    does this and routes the missing ones into ``failed_file_ids``.
    """

    def __init__(
        self,
        workspace: Workspace,
        file_ids: list[str],
        labels: dict[str, str] | None = None,
        *,
        text_view: str = "full",
        max_pages: int = 1,
    ) -> None:
        self.workspace = workspace
        self.file_ids = list(file_ids)
        self.labels = dict(labels) if labels else None
        self.text_view = text_view
        # How many leading page renders to load into ``page_images`` for optional
        # multi-page pooling. ``1`` (default) loads only page 1 — no extra I/O and
        # identical to prior behaviour. Set to match ``scenario.pooling_pages``.
        self.max_pages = max(1, max_pages)

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, index: int) -> DocumentRecord:
        # Imported lazily (and cached in sys.modules) so importing this
        # module doesn't pull in the clustering eval stack; mirrors how the
        # tfidf encoder reaches the same helper.
        from clustering.example import _build_text

        file_id = self.file_ids[index]
        ws = self.workspace
        # Up to max_pages page renders, for optional multi-page pooling.
        raw_pages = gather_file_pages(ws, file_id, self.max_pages)
        if not raw_pages:
            raise FileNotFoundError(f"no rendered page images for file '{file_id}'")
        page_images = tuple(Image.open(io.BytesIO(b)).convert("RGB") for b in raw_pages)
        # `_build_text` reads `<file_dir>/page_text/*.json` and nothing else, so
        # hand it just that (the real dir on LocalStore, zero-copy).
        with _file_text_dir(ws, file_id, self.text_view) as file_dir:
            text = _build_text(file_dir, view=self.text_view)
        return DocumentRecord(
            doc_id=file_id,
            label=self.labels.get(file_id) if self.labels else None,
            image=page_images[0],
            text=text,
            thumbnail_path=None,
            page_images=page_images,
        )
