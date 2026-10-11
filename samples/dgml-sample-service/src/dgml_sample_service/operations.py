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

"""The DGML operations the service performs — every one a ``dgml_core`` library call.

Nothing here shells out to the ``dgml`` CLI. The CLI's composite flows (``file add
--auto-classify``, ``extraction get-values``) are re-composed from the same public
library calls, the way an embedding host has to.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from dgml_core import (
    AddFileResult,
    ClassifyMode,
    DocSetStore,
    FileStore,
    TextMode,
    Workspace,
    add_file_and_extract,
    classify_file,
    extract_file,
    layout,
    load_classification_config,
)


def add_file(ws: Workspace, upload: Path, *, text_mode: str) -> AddFileResult:
    """``FileStore.add`` from an uploaded temp file. DGML copies the source into the
    workspace's blob store (S3), renders page images with PDFium and extracts page
    text; soft failures come back on the result rather than raising. A file the
    workspace already holds (same content, or same name) is a ``ConflictError``."""
    return FileStore(ws).add(upload, text_mode=TextMode(text_mode))


def add_result_json(result: AddFileResult) -> dict[str, Any]:
    return {
        "file": result.record.to_json(),
        "created": result.created,
        "conflict_kind": result.conflict_kind,
        "page_render_error": result.page_render_error,
        "page_count_error": result.page_count_error,
        "text_extraction_error": result.text_extraction_error,
        "conversion_error": result.conversion_error,
        "note": result.note,
    }


def stage_upload(filename: str, data: bytes) -> Path:
    """Write an upload to ``<tempdir>/<filename>``. DGML records the source's own name
    as the file's ``original_filename``, so the temp file keeps the upload's."""
    safe = Path(filename).name or "upload.pdf"
    directory = Path(tempfile.mkdtemp(prefix="dgml-upload-"))
    path = directory / safe
    path.write_bytes(data)
    return path


def discard_upload(path: Path) -> None:
    shutil.rmtree(path.parent, ignore_errors=True)


def classify(ws: Workspace, file_id: str, *, mode: str, extract: bool = True) -> dict[str, Any]:
    """Classify a file into a docset, then assign it (and extract, when the docset
    has an extraction schema) — the library form of ``dgml file add --auto-classify``.

    ``mode`` is a :class:`ClassifyMode`: ``existing`` may decline (``"none"``) and
    leave the file unassigned; ``existing-or-new`` creates the proposed docset when
    nothing fits."""
    config = load_classification_config(ws)
    decision = classify_file(ws, file_id, config=config, mode=ClassifyMode(mode))
    out: dict[str, Any] = {
        "decision": decision.decision,
        "reason": decision.reason,
        "docset_id": None,
        "created_docset": False,
        "extraction": None,
    }
    store = DocSetStore(ws)
    if decision.decision == "none":
        return out
    if decision.decision == "new":
        ds = store.create(
            decision.new_name or "Untitled",
            decision.new_description or "",
            key_questions=list(decision.new_key_questions),
        )
        docset_id = ds.id
        out["created_docset"] = True
    else:
        assert decision.existing_docset_id is not None
        docset_id = decision.existing_docset_id
    out["docset_id"] = docset_id
    if extract:
        out["extraction"] = add_file_and_extract(ws, docset_id, file_id)
    else:
        store.add_file(docset_id, file_id)
    return out


def extract(ws: Workspace, docset_id: str, file_id: str) -> dict[str, Any]:
    """Extract (or re-extract) a file's values against the docset's schema. Re-running
    rewrites the ``dg:extraction`` element of the pair's ``.dgml.xml``."""
    result = extract_file(ws, docset_id, file_id)
    return {
        "docset_id": docset_id,
        "file_id": file_id,
        "model": result.model,
        "mode": result.mode,
        "tool_calls": result.tool_calls,
        "field_count": len(result.values),
        "xml_key": result.xml_key,
    }


def generate_schema(ws: Workspace, docset_id: str, file_ids: list[str]) -> dict[str, Any]:
    """Propose an extraction schema (RELAX NG Compact) from sample files and store it."""
    from dgml_core.grounded import generate_schema as _generate
    from dgml_core.grounded import load_grounded_config

    store = DocSetStore(ws)
    ds = store.get(docset_id)
    config = load_grounded_config(ws)
    rnc = _generate(ws, file_ids, config=config, docset_name=ds.name)
    store.set_schema(docset_id, rnc)
    return {"docset_id": docset_id, "schema": rnc, "model": config.schema_model}


def dgml_xml_key(ws: Workspace, docset_id: str, file_id: str) -> str | None:
    """The pair's ``<stem>.dgml.xml`` blob key, if one exists. Mirrors the CLI's
    ``extraction get-values``: the single ``*.dgml.xml`` blob in the pair's prefix
    (a ``.dgml.grounded.xml`` sibling does not end in ``.dgml.xml``)."""
    keys = sorted(
        k
        for k in ws.blobs.list_blobs(layout.docset_pair_prefix(docset_id, file_id))
        if k.endswith(layout.DGML_XML_SUFFIX)
    )
    return keys[0] if keys else None


def files_with_dgml(ws: Workspace, docset_id: str) -> set[str]:
    """The ids of the docset's files that have a ``.dgml.xml`` — one listing for the
    whole docset rather than one per file."""
    prefix = layout.docset_files_prefix(docset_id)
    found = set()
    for key in ws.blobs.list_blobs(prefix):
        file_id, _, name = key[len(prefix) :].partition("/")
        if name.endswith(layout.DGML_XML_SUFFIX) and "/" not in name:
            found.add(file_id)
    return found


def read_pair(ws: Workspace, docset_id: str, file_id: str) -> dict[str, Any]:
    """A pair's DGML XML and its extracted values (``{tag: {text, value?, locations}}``,
    projected through the docset schema's vocabulary when it has one)."""
    from dgml_core.extraction_schema import parse_rnc
    from dgml_core.extraction_xml import dgml_xml_to_values, has_document_tree, has_extraction

    store = DocSetStore(ws)
    key = dgml_xml_key(ws, docset_id, file_id)
    xml = ws.blobs.get_blob(key).decode("utf-8") if key else None
    extracted = bool(xml and has_extraction(xml))
    values = None
    if xml and extracted:
        vocab = parse_rnc(store.get_schema(docset_id)) if store.has_schema(docset_id) else None
        values = dgml_xml_to_values(xml, vocab=vocab)
    return {
        "docset_id": docset_id,
        "file_id": file_id,
        "xml_key": key,
        "xml": xml,
        "has_extraction": extracted,
        "has_document_tree": bool(xml and has_document_tree(xml)),
        "values": values,
    }
