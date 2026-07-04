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

from clustering.data.datasets import DocumentDataset, DocumentRecord
from PIL import Image

from .pages import PAGE_GLOB
from .storage import Workspace


class WorkspaceFileDataset(DocumentDataset):
    """Lazy :class:`DocumentDataset` over a list of dgml file IDs.

    The first-page image for each file comes from the pre-rendered
    ``<file>/page_images/page_1.png`` that ``dgml file add`` produced —
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
    ) -> None:
        self.workspace = workspace
        self.file_ids = list(file_ids)
        self.labels = dict(labels) if labels else None
        self.text_view = text_view

    def __len__(self) -> int:
        return len(self.file_ids)

    def __getitem__(self, index: int) -> DocumentRecord:
        # Imported lazily (and cached in sys.modules) so importing this
        # module doesn't pull in the clustering eval stack; mirrors how the
        # tfidf encoder reaches the same helper.
        from clustering.example import _build_text

        file_id = self.file_ids[index]
        pages = sorted(self.workspace.file_pages_dir(file_id).glob(PAGE_GLOB))
        if not pages:
            raise FileNotFoundError(f"no rendered page images for file '{file_id}'")
        image = Image.open(pages[0]).convert("RGB")
        text = _build_text(self.workspace.file_dir(file_id), view=self.text_view)
        return DocumentRecord(
            doc_id=file_id,
            label=self.labels.get(file_id) if self.labels else None,
            image=image,
            text=text,
            thumbnail_path=None,
        )
