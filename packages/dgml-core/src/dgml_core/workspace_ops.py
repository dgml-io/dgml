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

"""Multi-key workspace mutations — the cascades, in one place.

The store deliberately offers only **single-key atomic** primitives
(``put_doc`` / ``delete_doc`` / ``put_blob`` / ``delete_blob``) plus
**idempotent bulk** ones (``delete_docs`` / ``delete_blobs``). It offers no
transaction, because none of the backends it targets has one: neither an object
store plus a document database nor a POSIX filesystem can commit across keys.
Pretending otherwise in the interface would be a lie on every implementation.

So anything that must touch several keys is a *named operation* here, written to
be **idempotent and resumable** rather than atomic. Each is safe to re-run after
a crash, and each leaves a failure state that a later pass can finish.

The ordering rule
-----------------

Every cascade obeys one rule:

    **The authoritative record dies first.**

A crash mid-cascade must leave *orphaned bytes* — recoverable garbage that a
later sweep can identify and remove — and never *a record pointing at bytes that
are gone*, which reads as a valid entity and misleads everything downstream.

Concretely: delete the manifest/assignment document, then its dependent
documents, then the blobs. ``delete_blobs`` also runs last for a second reason —
on ``LocalStore`` it is the step that prunes the emptied directory, which it can
only do once the documents beside those blobs are gone.

Why an object rather than free functions: a cascade issues many store calls, and
``Workspace.store`` resolves the configured backend afresh on every access. One
instance binds the store once for the whole operation, which on a remote backend
is the difference between one client and dozens.
"""

from __future__ import annotations

from . import layout
from .errors import DocSetNotFound, FileNotFound, InvalidArgument
from .storage import Workspace
from .storage_service import BlobStore, DocStore


class WorkspaceOps:
    """Cascading deletes over a workspace, composed from native store calls.

    Constructed per operation (or per command); holds the workspace's blob and
    document stores for the lifetime of the instance."""

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace
        self.blobs: BlobStore = workspace.blobs
        self.docs: DocStore = workspace.docs

    # ---- assignments ----

    def unassign(self, docset_id: str, file_id: str) -> None:
        """Remove a docset↔file assignment and that pair's generated artifacts
        (the ``dgml.xml`` blob and the extraction-stats document).

        Record-first: the assignment document goes before the artifacts it
        describes, so an interrupted run leaves regenerable outputs with no
        assignment rather than an assignment whose outputs vanished."""
        pair = layout.pair_id(docset_id, file_id)
        self.docs.delete_doc(layout.Collection.ASSIGNMENTS, pair)
        self.docs.delete_doc(layout.Collection.EXTRACTION_STATS, pair)
        self.blobs.delete_blobs(layout.docset_pair_prefix(docset_id, file_id))

    # ---- entities ----

    def delete_file(self, file_id: str) -> None:
        """Delete a file: every assignment referencing it, then its own
        documents, then its blobs.

        The assignments go first because they are what makes the file reachable
        from a docset — a crash after that point leaves an unreferenced file
        directory, not docsets pointing at a file that no longer exists."""
        if not file_id.strip():
            raise InvalidArgument("file id must not be empty")
        if self.docs.get_doc(layout.Collection.FILES, file_id) is None:
            raise FileNotFound(f"file '{file_id}' not found")
        for assignment in self.docs.find_docs(layout.Collection.ASSIGNMENTS, {"file_id": file_id}):
            self.unassign(str(assignment["docset_id"]), file_id)
        self.docs.delete_doc(layout.Collection.FILES, file_id)
        self.docs.delete_doc(layout.Collection.ERRORS, file_id)
        self.blobs.delete_blobs(layout.file_prefix(file_id))

    def delete_docset(self, docset_id: str) -> None:
        """Delete a docset: every assignment to it (with each pair's outputs),
        then its own document, then its blobs.

        The files themselves under ``files/`` are deliberately untouched — a
        docset is a grouping, not an owner."""
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        if self.docs.get_doc(layout.Collection.DOCSETS, docset_id) is None:
            raise DocSetNotFound(f"docset '{docset_id}' not found")
        for assignment in self.docs.find_docs(
            layout.Collection.ASSIGNMENTS, {"docset_id": docset_id}
        ):
            self.unassign(docset_id, str(assignment["file_id"]))
        self.docs.delete_doc(layout.Collection.DOCSETS, docset_id)
        self.blobs.delete_blobs(layout.docset_prefix(docset_id))
