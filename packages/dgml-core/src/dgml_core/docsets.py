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

import shutil

from .errors import (
    CorruptMetadata,
    DocSetNotFound,
    FileNotFound,
    InvalidArgument,
    SchemaInvalid,
    SchemaNotFound,
)
from .ids import new_id
from .models import DocSet
from .storage import Workspace, read_json, write_json_atomic, write_text_atomic


class DocSetStore:
    """CRUD for DocSets in a workspace."""

    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace

    def list_all(self) -> list[DocSet]:
        if not self.ws.docsets_dir.exists():
            return []
        out: list[DocSet] = []
        for entry in sorted(self.ws.docsets_dir.iterdir()):
            if not entry.is_dir():
                continue
            json_path = entry / "docset.json"
            if not json_path.exists():
                continue
            try:
                data = read_json(json_path)
            except CorruptMetadata:
                continue
            out.append(DocSet.from_json(data))
        return out

    def get(self, docset_id: str) -> DocSet:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        json_path = self.ws.docset_json_path(docset_id)
        if not json_path.exists():
            raise DocSetNotFound(f"docset '{docset_id}' not found")
        return DocSet.from_json(read_json(json_path))

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
        self.ws.docset_dir(docset_id).mkdir(parents=True, exist_ok=False)
        self.ws.docset_files_dir(docset_id).mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.ws.docset_json_path(docset_id), ds.to_json())
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
        write_json_atomic(self.ws.docset_json_path(docset_id), ds.to_json())
        return ds

    def delete(self, docset_id: str) -> None:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        if not self.ws.docset_dir(docset_id).exists():
            raise DocSetNotFound(f"docset '{docset_id}' not found")
        shutil.rmtree(self.ws.docset_dir(docset_id))

    def list_files(self, docset_id: str) -> list[str]:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        if not self.ws.docset_dir(docset_id).exists():
            raise DocSetNotFound(f"docset '{docset_id}' not found")
        files_dir = self.ws.docset_files_dir(docset_id)
        if not files_dir.exists():
            return []
        return sorted(p.name for p in files_dir.iterdir() if p.is_dir())

    def add_file(self, docset_id: str, file_id: str) -> None:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        if not file_id.strip():
            raise InvalidArgument("file id must not be empty")
        if not self.ws.docset_dir(docset_id).exists():
            raise DocSetNotFound(f"docset '{docset_id}' not found")
        if not self.ws.file_dir(file_id).exists():
            raise FileNotFound(f"file '{file_id}' not found")
        ref = self.ws.docset_files_dir(docset_id) / file_id
        ref.mkdir(parents=True, exist_ok=True)

    def remove_file(self, docset_id: str, file_id: str) -> None:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        if not file_id.strip():
            raise InvalidArgument("file id must not be empty")
        if not self.ws.docset_dir(docset_id).exists():
            raise DocSetNotFound(f"docset '{docset_id}' not found")
        ref = self.ws.docset_files_dir(docset_id) / file_id
        if not ref.exists():
            raise FileNotFound(f"file '{file_id}' is not assigned to docset '{docset_id}'")
        shutil.rmtree(ref)

    # ---- extraction schema (docsets/<id>/extraction-schema.rnc, RELAX NG Compact) --

    def get_schema(self, docset_id: str) -> str:
        """Read the docset's extraction schema as RNC text.

        Raises :class:`SchemaNotFound` if absent. Callers that need the
        engine's grounded_field JSON Schema convert via
        :func:`dgml_core.extraction_schema.rnc_to_json_schema`.
        """
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        if not self.ws.docset_dir(docset_id).exists():
            raise DocSetNotFound(f"docset '{docset_id}' not found")
        path = self.ws.docset_schema_path(docset_id)
        if not path.exists():
            raise SchemaNotFound(f"docset '{docset_id}' has no schema")
        return path.read_text(encoding="utf-8")

    def has_schema(self, docset_id: str) -> bool:
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        return self.ws.docset_schema_path(docset_id).exists()

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
        if not self.ws.docset_dir(docset_id).exists():
            raise DocSetNotFound(f"docset '{docset_id}' not found")
        if not isinstance(schema, str):
            raise SchemaInvalid("schema must be RNC text")
        validate_rnc(schema)  # raises SchemaInvalid on anything outside the subset
        write_text_atomic(self.ws.docset_schema_path(docset_id), schema)
        return schema

    def clear_schema(self, docset_id: str) -> bool:
        """Remove the docset's schema. Returns True if a schema was removed,
        False if there was none to remove."""
        if not docset_id.strip():
            raise InvalidArgument("docset id must not be empty")
        if not self.ws.docset_dir(docset_id).exists():
            raise DocSetNotFound(f"docset '{docset_id}' not found")
        path = self.ws.docset_schema_path(docset_id)
        if not path.exists():
            return False
        path.unlink()
        return True
