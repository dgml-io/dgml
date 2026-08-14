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

"""DocSet CRUD operations and File assignments."""

from __future__ import annotations

from . import layout
from .errors import (
    DocSetNotFound,
    FileNotFound,
    GuidanceNotFound,
    InvalidArgument,
    SchemaInvalid,
    SchemaNotFound,
    now_iso,
)
from .ids import new_id
from .models import DocSet
from .storage import Workspace
from .workspace_ops import WorkspaceOps


class DocSetStore:
    """CRUD for DocSets in a workspace."""

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    def _require_docset(self, docset_id: str) -> None:
        """Raise :class:`DocSetNotFound` unless the docset's manifest exists.

        Existence is defined by the ``docset.json`` document, not a directory —
        so the check works on any store (a remote backend has no ``docsets/<id>/``
        directory to stat)."""
        if self.ws.docs.get_doc(layout.Collection.DOCSETS, docset_id) is None:
            raise DocSetNotFound(f"docset '{docset_id}' not found")

    def list_all(self) -> list[DocSet]:
        # Sorted here, not by the store: ``find_docs`` has no defined ordering
        # (LocalStore returns path order, a document database returns insertion
        # order), and this list is user-visible CLI output that must not depend
        # on which backend the workspace happens to live on.
        return sorted(
            (
                DocSet.from_json(data)
                for data in self.ws.docs.find_docs(layout.Collection.DOCSETS, {})
            ),
            key=lambda ds: ds.id,
        )

    def get(self, docset_id: str) -> DocSet:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        data = self.ws.docs.get_doc(layout.Collection.DOCSETS, docset_id)
        if data is None:
            raise DocSetNotFound(f"docset '{docset_id}' not found")
        return DocSet.from_json(data)

    def create(
        self,
        name: str,
        description: str = "",
        *,
        key_questions: list[str] | None = None,
    ) -> DocSet:
        if not name.strip():
            raise InvalidArgument("docset name must not be empty")
        docset_id = new_id()
        ds = DocSet(
            id=docset_id,
            name=name,
            description=description,
            key_questions=list(key_questions or []),
        )
        # No directory is created up front — the store owns container creation
        # (put_doc writes the manifest). A fresh new_id never collides.
        self.ws.docs.put_doc(layout.Collection.DOCSETS, docset_id, ds.to_json())
        return ds

    def update(
        self,
        docset_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        key_questions: list[str] | None = None,
    ) -> DocSet:
        ds = self.get(docset_id)
        if name is not None:
            if not name.strip():
                raise InvalidArgument("docset name must not be empty")
            ds.name = name
        if description is not None:
            ds.description = description
        if key_questions is not None:
            ds.key_questions = list(key_questions)
        self.ws.docs.put_doc(layout.Collection.DOCSETS, docset_id, ds.to_json())
        return ds

    def delete(self, docset_id: str) -> None:
        """Delete the docset and every assignment to it. The underlying files
        are untouched. The cascade itself lives in :class:`WorkspaceOps`."""
        WorkspaceOps(self.ws).delete_docset(docset_id)

    def list_files(self, docset_id: str) -> list[str]:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        self._require_docset(docset_id)
        assignments = self.ws.docs.find_docs(
            layout.Collection.ASSIGNMENTS, {"docset_id": docset_id}
        )
        return sorted(str(a["file_id"]) for a in assignments)

    def add_file(self, docset_id: str, file_id: str) -> None:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        if not file_id.strip():
            raise InvalidArgument("file id must not be empty")
        self._require_docset(docset_id)
        if self.ws.docs.get_doc(layout.Collection.FILES, file_id) is None:
            raise FileNotFound(f"file '{file_id}' not found")
        # An assignment is a document keyed by the (docset, file) pair. Re-adding
        # is idempotent — it replaces the same document, refreshing assigned_at.
        self.ws.docs.put_doc(
            layout.Collection.ASSIGNMENTS,
            layout.pair_id(docset_id, file_id),
            {"docset_id": docset_id, "file_id": file_id, "assigned_at": now_iso()},
        )

    def remove_file(self, docset_id: str, file_id: str) -> None:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        if not file_id.strip():
            raise InvalidArgument("file id must not be empty")
        self._require_docset(docset_id)
        if (
            self.ws.docs.get_doc(layout.Collection.ASSIGNMENTS, layout.pair_id(docset_id, file_id))
            is None
        ):
            raise FileNotFound(f"file '{file_id}' is not assigned to docset '{docset_id}'")
        WorkspaceOps(self.ws).unassign(docset_id, file_id)

    # ---- extraction schema (docsets/<id>/extraction-schema.rnc, RELAX NG Compact) --

    def get_schema(self, docset_id: str) -> str:
        """Read the docset's extraction schema as RNC text.

        Raises :class:`SchemaNotFound` if absent. Callers that need the
        engine's grounded_field JSON Schema convert via
        :func:`dgml_core.extraction_schema.rnc_to_json_schema`.
        """
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        self._require_docset(docset_id)
        key = layout.docset_extraction_schema_key(docset_id)
        try:
            return self.ws.blobs.get_blob(key).decode("utf-8")
        except FileNotFoundError:
            raise SchemaNotFound(f"docset '{docset_id}' has no schema") from None

    def has_schema(self, docset_id: str) -> bool:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        return self.ws.blobs.blob_exists(layout.docset_extraction_schema_key(docset_id))

    def set_schema(self, docset_id: str, schema: str) -> str:
        """Write (replace) the docset's extraction schema from RNC text.

        Validates that *schema* parses within the supported RNC subset
        (:func:`dgml_core.extraction_schema.validate_rnc`); raises
        :class:`SchemaInvalid` otherwise. The CLI accepts JSON Schema input and
        converts it to RNC before calling this — RNC is the only on-disk form.
        """
        from .extraction_schema import validate_rnc

        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        self._require_docset(docset_id)
        if not isinstance(schema, str):
            raise SchemaInvalid("schema must be RNC text")
        validate_rnc(schema)  # raises SchemaInvalid on anything outside the subset
        self.ws.blobs.put_blob(
            layout.docset_extraction_schema_key(docset_id),
            schema.encode("utf-8"),
        )
        return schema

    def clear_schema(self, docset_id: str) -> bool:
        """Remove the docset's schema. Returns True if a schema was removed,
        False if there was none to remove."""
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        self._require_docset(docset_id)
        key = layout.docset_extraction_schema_key(docset_id)
        if not self.ws.blobs.blob_exists(key):
            return False
        self.ws.blobs.delete_blob(key)
        return True

    # ---- extraction guidance (docsets/<id>/extraction-guidance.md) ---------

    def get_guidance(self, docset_id: str) -> str:
        """Read the docset's extraction guidance text.

        Raises :class:`GuidanceNotFound` if absent. The guidance is free-form
        markdown/plain text injected into the phase-1 extraction prompt for
        every file extracted against this docset.
        """
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        self._require_docset(docset_id)
        key = layout.docset_guidance_key(docset_id)
        try:
            return self.ws.blobs.get_blob(key).decode("utf-8")
        except FileNotFoundError:
            raise GuidanceNotFound(f"docset '{docset_id}' has no extraction guidance") from None

    def has_guidance(self, docset_id: str) -> bool:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        return self.ws.blobs.blob_exists(layout.docset_guidance_key(docset_id))

    def set_guidance(self, docset_id: str, guidance: str) -> str:
        """Write (replace) the docset's extraction guidance text."""
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        self._require_docset(docset_id)
        if not isinstance(guidance, str) or not guidance.strip():
            raise InvalidArgument("guidance must be non-empty text")
        self.ws.blobs.put_blob(
            layout.docset_guidance_key(docset_id),
            guidance.encode("utf-8"),
        )
        return guidance

    def clear_guidance(self, docset_id: str) -> bool:
        """Remove the docset's extraction guidance. Returns True if removed,
        False if there was none to remove."""
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        self._require_docset(docset_id)
        key = layout.docset_guidance_key(docset_id)
        if not self.ws.blobs.blob_exists(key):
            return False
        self.ws.blobs.delete_blob(key)
        return True
