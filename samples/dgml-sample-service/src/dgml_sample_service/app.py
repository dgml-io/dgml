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

"""The FastAPI application.

Every route is a plain ``def``: dgml-core is synchronous, so FastAPI runs each
request on its thread pool. Short CRUD calls run inline; slow work (adding a
file, classification, schema generation, extraction) is queued as a job.

Routes are scoped to an organisation (``/api/orgs/{org_id}/…``): one organisation
is one DGML workspace, opened from that organisation's settings.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from dgml_core import (
    ConflictError,
    DgmlError,
    DocSetStore,
    FileRecord,
    FileStore,
    InvalidArgument,
    InvalidPDF,
    NotFoundError,
    SchemaInvalid,
    UnsupportedFileType,
    Workspace,
    layout,
)
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import explorer
from . import operations as ops
from . import settings_store as st
from .config import ServiceConfig
from .db import create_schema, make_engine
from .docstore import PostgresDocStore
from .jobs import JobFn, JobRunner
from .tenancy import TEXT_MODES, SettingsMissing, WorkspaceRegistry

logger = logging.getLogger(__name__)

CLASSIFY_MODES = ("existing", "existing-or-new")


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class OrgCreate(BaseModel):
    name: str
    slug: str


class SettingsBody(BaseModel):
    llm_family: str | None = None
    text_mode: str | None = None
    ocr_provider: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    s3_bucket: str | None = None
    blob_folder: str | None = None
    #: kind -> new value; "" clears it; a missing kind is left as is.
    secrets: dict[str, str] = Field(default_factory=dict)
    #: Create the bucket when the probe finds it missing.
    create_bucket: bool = False

    def update(self) -> st.SettingsUpdate:
        values = self.model_dump(exclude={"secrets", "create_bucket"}, exclude_unset=True)
        return st.SettingsUpdate(values=values, secrets=dict(self.secrets))


class DocSetCreate(BaseModel):
    name: str
    description: str = ""
    key_questions: list[str] = Field(default_factory=list)


class DocSetPatch(BaseModel):
    name: str | None = None
    description: str | None = None
    key_questions: list[str] | None = None


class TextBody(BaseModel):
    text: str


class AssignBody(BaseModel):
    #: Extract right away when the docset has a schema (queued as a job).
    extract: bool = True


class ClassifyBody(BaseModel):
    mode: str = "existing"
    extract: bool = True


class GenerateSchemaBody(BaseModel):
    #: Sample files to learn from; defaults to (up to 3 of) the docset's files.
    file_ids: list[str] | None = None


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------


class Services:
    """What every request needs: the engine, the workspace registry, the job runner."""

    def __init__(self, config: ServiceConfig, engine: Engine) -> None:
        self.config = config
        self.engine = engine
        self.registry = WorkspaceRegistry(engine)
        self.jobs = JobRunner(engine, workers=config.job_workers)


def _services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(_services)]


def _org(org_id: uuid.UUID, services: ServicesDep) -> uuid.UUID:
    st.get_org(services.engine, org_id)  # raises OrgNotFound -> 404
    return org_id


OrgDep = Annotated[uuid.UUID, Depends(_org)]


def _workspace(org_id: OrgDep, services: ServicesDep) -> Workspace:
    return services.registry.open(org_id)


WorkspaceDep = Annotated[Workspace, Depends(_workspace)]


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _dgml_status(exc: DgmlError) -> int:
    """DGML errors map to HTTP by class, never by message text."""
    if isinstance(exc, NotFoundError):
        return 404
    if isinstance(exc, ConflictError):
        return 409
    if isinstance(exc, InvalidArgument | UnsupportedFileType | InvalidPDF | SchemaInvalid):
        return 422
    return 500


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app(config: ServiceConfig | None = None, *, engine: Engine | None = None) -> FastAPI:
    config = config or ServiceConfig.from_env()
    engine = engine or make_engine(config.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        create_schema(engine)
        PostgresDocStore.bind_engine(engine)
        services = Services(config, engine)
        recovered = services.jobs.recover()
        if recovered:
            logger.warning("marked %d unfinished job(s) from a previous run as failed", recovered)
        app.state.services = services
        yield
        services.jobs.shutdown()

    app = FastAPI(title="DGML sample service", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(DgmlError)
    def _on_dgml(_: Request, exc: DgmlError) -> JSONResponse:
        status = _dgml_status(exc)
        if status >= 500:
            logger.error("DGML error %s: %s", exc.code, exc)
        return _error(status, exc.code, str(exc))

    @app.exception_handler(StarletteHTTPException)
    def _on_http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return _error(exc.status_code, code, str(exc.detail))

    @app.exception_handler(explorer.TableNotFound)
    def _on_table(_: Request, exc: explorer.TableNotFound) -> JSONResponse:
        return _error(404, "TABLE_NOT_FOUND", str(exc))

    @app.exception_handler(st.OrgNotFound)
    def _on_org(_: Request, exc: st.OrgNotFound) -> JSONResponse:
        return _error(404, "ORG_NOT_FOUND", str(exc))

    @app.exception_handler(SettingsMissing)
    def _on_settings_missing(_: Request, exc: SettingsMissing) -> JSONResponse:
        return _error(409, "SETTINGS_MISSING", str(exc))

    @app.exception_handler(st.SettingsInvalid)
    def _on_settings_invalid(_: Request, exc: st.SettingsInvalid) -> JSONResponse:
        return _error(422, "SETTINGS_INVALID", str(exc))

    @app.exception_handler(st.SettingsLocked)
    def _on_settings_locked(_: Request, exc: st.SettingsLocked) -> JSONResponse:
        return _error(409, "STORAGE_LOCKED", str(exc))

    @app.exception_handler(st.StorageProbeFailed)
    def _on_probe(_: Request, exc: st.StorageProbeFailed) -> JSONResponse:
        return _error(422, "STORAGE_PROBE_FAILED", str(exc))

    _routes(app)
    if config.frontend_dist is not None:
        _frontend(app, config.frontend_dist)
    return app


def _routes(app: FastAPI) -> None:
    # ---- health + organisations ----

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/orgs")
    def list_orgs(services: ServicesDep) -> list[dict[str, Any]]:
        return st.list_orgs(services.engine)

    @app.post("/api/orgs", status_code=201)
    def create_org(body: OrgCreate, services: ServicesDep) -> dict[str, Any]:
        return st.create_org(services.engine, services.config, name=body.name, slug=body.slug)

    @app.get("/api/orgs/{org_id}")
    def get_org(org_id: OrgDep, services: ServicesDep) -> dict[str, Any]:
        return st.get_org(services.engine, org_id)

    @app.delete("/api/orgs/{org_id}", status_code=204)
    def delete_org(org_id: OrgDep, services: ServicesDep) -> Response:
        # DGML first — that is what deletes the blobs — then every row the org owns.
        try:
            ws = services.registry.open(org_id)
        except (DgmlError, SettingsMissing) as exc:
            # Settings that no longer open a workspace: its blobs are unreachable, so
            # only the rows go (st.delete_org clears the dgml_* tables too).
            logger.warning("deleting org %s without its DGML blobs: %s", org_id, exc)
        else:
            files, docsets = FileStore(ws), DocSetStore(ws)
            for record in files.list_all():
                with services.registry.file_lock(org_id, record.id):  # wait out its jobs
                    files.delete(record.id)
            for ds in docsets.list_all():
                docsets.delete(ds.id)
        services.registry.forget(org_id)
        st.delete_org(services.engine, org_id)
        return Response(status_code=204)

    # ---- settings ----

    @app.get("/api/orgs/{org_id}/settings")
    def get_settings(org_id: OrgDep, services: ServicesDep) -> dict[str, Any]:
        return st.settings_json(services.engine, org_id)

    @app.put("/api/orgs/{org_id}/settings")
    def put_settings(org_id: OrgDep, body: SettingsBody, services: ServicesDep) -> dict[str, Any]:
        update = body.update()
        st.validate_update(update)
        current = services.registry.settings(org_id)
        new = st.merged_settings(current, update)
        if st.storage_location_changed(current, new) and st.file_count(services.engine, org_id):
            raise st.SettingsLocked(
                "the storage connection and folder are locked once the workspace holds "
                "files: changing them would orphan every stored blob"
            )
        if st.storage_changed(current, new) or body.create_bucket:
            st.probe_storage(new, create_bucket=body.create_bucket)
        # No cache to clear: the registry reopens the workspace once the Configuration
        # built from the saved settings differs from the cached one.
        st.save_settings(services.engine, new, update)
        return st.settings_json(services.engine, org_id)

    @app.post("/api/orgs/{org_id}/settings/test-storage")
    def test_storage(
        org_id: OrgDep, services: ServicesDep, create_bucket: bool = False
    ) -> dict[str, Any]:
        return st.probe_storage(services.registry.settings(org_id), create_bucket=create_bucket)

    # ---- docsets ----

    @app.get("/api/orgs/{org_id}/docsets")
    def list_docsets(ws: WorkspaceDep) -> list[dict[str, Any]]:
        store = DocSetStore(ws)
        return [docset_json(store, ds.id) for ds in store.list_all()]

    @app.post("/api/orgs/{org_id}/docsets", status_code=201)
    def create_docset(body: DocSetCreate, ws: WorkspaceDep) -> dict[str, Any]:
        store = DocSetStore(ws)
        ds = store.create(body.name, body.description, key_questions=body.key_questions)
        return docset_json(store, ds.id)

    @app.get("/api/orgs/{org_id}/docsets/{docset_id}")
    def get_docset(docset_id: str, ws: WorkspaceDep) -> dict[str, Any]:
        out = docset_json(DocSetStore(ws), docset_id)
        files = FileStore(ws)
        extracted = ops.files_with_dgml(ws, docset_id)
        out["files"] = []
        for fid in out["file_ids"]:
            entry: dict[str, Any] = {"id": fid, "has_dgml": fid in extracted}
            try:
                entry |= files.get(fid).to_json()
            except NotFoundError:
                entry["missing"] = True
            out["files"].append(entry)
        return out

    @app.patch("/api/orgs/{org_id}/docsets/{docset_id}")
    def update_docset(docset_id: str, body: DocSetPatch, ws: WorkspaceDep) -> dict[str, Any]:
        store = DocSetStore(ws)
        store.update(
            docset_id,
            name=body.name,
            description=body.description,
            key_questions=body.key_questions,
        )
        return docset_json(store, docset_id)

    @app.delete("/api/orgs/{org_id}/docsets/{docset_id}", status_code=204)
    def delete_docset(docset_id: str, ws: WorkspaceDep) -> Response:
        DocSetStore(ws).delete(docset_id)
        return Response(status_code=204)

    @app.get("/api/orgs/{org_id}/docsets/{docset_id}/schema")
    def get_schema(docset_id: str, ws: WorkspaceDep) -> dict[str, Any]:
        return {"docset_id": docset_id, "schema": DocSetStore(ws).get_schema(docset_id)}

    @app.put("/api/orgs/{org_id}/docsets/{docset_id}/schema")
    def set_schema(docset_id: str, body: TextBody, ws: WorkspaceDep) -> dict[str, Any]:
        store = DocSetStore(ws)
        store.get(docset_id)
        return {"docset_id": docset_id, "schema": store.set_schema(docset_id, body.text)}

    @app.delete("/api/orgs/{org_id}/docsets/{docset_id}/schema", status_code=204)
    def clear_schema(docset_id: str, ws: WorkspaceDep) -> Response:
        DocSetStore(ws).clear_schema(docset_id)
        return Response(status_code=204)

    @app.post("/api/orgs/{org_id}/docsets/{docset_id}/schema/generate", status_code=202)
    def generate_schema(
        docset_id: str,
        body: GenerateSchemaBody,
        org_id: OrgDep,
        ws: WorkspaceDep,
        services: ServicesDep,
    ) -> dict[str, Any]:
        store = DocSetStore(ws)
        ds = store.get(docset_id)
        file_ids = body.file_ids or store.list_files(docset_id)[:3]
        if not file_ids:
            raise InvalidArgument("the docset has no files to learn a schema from; add some")
        return services.jobs.submit(
            org_id,
            "generate_schema",
            lambda: ops.generate_schema(ws, docset_id, file_ids),
            docset_id=docset_id,
            label=f"Generate schema for {ds.name}",
            params={"file_ids": file_ids},
        )

    @app.get("/api/orgs/{org_id}/docsets/{docset_id}/guidance")
    def get_guidance(docset_id: str, ws: WorkspaceDep) -> dict[str, Any]:
        return {"docset_id": docset_id, "guidance": DocSetStore(ws).get_guidance(docset_id)}

    @app.put("/api/orgs/{org_id}/docsets/{docset_id}/guidance")
    def set_guidance(docset_id: str, body: TextBody, ws: WorkspaceDep) -> dict[str, Any]:
        store = DocSetStore(ws)
        store.get(docset_id)
        return {"docset_id": docset_id, "guidance": store.set_guidance(docset_id, body.text)}

    @app.delete("/api/orgs/{org_id}/docsets/{docset_id}/guidance", status_code=204)
    def clear_guidance(docset_id: str, ws: WorkspaceDep) -> Response:
        DocSetStore(ws).clear_guidance(docset_id)
        return Response(status_code=204)

    # ---- assignments, extraction and the DGML viewer ----

    @app.post("/api/orgs/{org_id}/docsets/{docset_id}/files/{file_id}")
    def assign(
        docset_id: str,
        file_id: str,
        body: AssignBody,
        org_id: OrgDep,
        ws: WorkspaceDep,
        services: ServicesDep,
    ) -> dict[str, Any]:
        store = DocSetStore(ws)
        FileStore(ws).get(file_id)
        store.get(docset_id)
        if body.extract and store.has_schema(docset_id):
            job = _extract_job(services, org_id, ws, docset_id, file_id, assign=True)
            return {"assigned": False, "job": job}
        store.add_file(docset_id, file_id)
        return {"assigned": True, "job": None}

    @app.delete("/api/orgs/{org_id}/docsets/{docset_id}/files/{file_id}", status_code=204)
    def unassign(docset_id: str, file_id: str, ws: WorkspaceDep) -> Response:
        DocSetStore(ws).remove_file(docset_id, file_id)
        return Response(status_code=204)

    @app.post("/api/orgs/{org_id}/docsets/{docset_id}/files/{file_id}/extract", status_code=202)
    def extract(
        docset_id: str, file_id: str, org_id: OrgDep, ws: WorkspaceDep, services: ServicesDep
    ) -> dict[str, Any]:
        store = DocSetStore(ws)
        if not store.is_assigned(docset_id, file_id):
            raise HTTPException(404, f"file {file_id!r} is not assigned to docset {docset_id!r}")
        store.get_schema(docset_id)  # SchemaNotFound -> 404 now, not in the job
        return _extract_job(services, org_id, ws, docset_id, file_id, assign=False)

    @app.get("/api/orgs/{org_id}/docsets/{docset_id}/files/{file_id}/dgml")
    def read_dgml(docset_id: str, file_id: str, ws: WorkspaceDep) -> dict[str, Any]:
        if not DocSetStore(ws).is_assigned(docset_id, file_id):
            raise HTTPException(404, f"file {file_id!r} is not assigned to docset {docset_id!r}")
        return ops.read_pair(ws, docset_id, file_id)

    @app.get("/api/orgs/{org_id}/docsets/{docset_id}/files/{file_id}/dgml.xml")
    def download_dgml(docset_id: str, file_id: str, ws: WorkspaceDep) -> Response:
        key = ops.dgml_xml_key(ws, docset_id, file_id)
        if key is None:
            raise HTTPException(404, "no DGML for this file in this docset yet")
        return Response(
            ws.blobs.get_blob(key),
            media_type="application/xml",
            headers={"Content-Disposition": f'attachment; filename="{Path(key).name}"'},
        )

    # ---- files ----

    @app.get("/api/orgs/{org_id}/files")
    def list_files(ws: WorkspaceDep) -> list[dict[str, Any]]:
        # One pass over the docsets, not a docsets_for_file call per file.
        store = DocSetStore(ws)
        docsets_of: dict[str, list[dict[str, str]]] = {}
        for ds in store.list_all():
            for fid in store.list_files(ds.id):
                docsets_of.setdefault(fid, []).append({"id": ds.id, "name": ds.name})
        return [file_json(r, docsets_of.get(r.id, [])) for r in FileStore(ws).list_all()]

    @app.post("/api/orgs/{org_id}/files", status_code=202)
    def upload_files(
        org_id: OrgDep,
        ws: WorkspaceDep,
        services: ServicesDep,
        files: Annotated[list[UploadFile], File()],
        classify: Annotated[str, Form()] = "none",
        text_mode: Annotated[str | None, Form()] = None,
    ) -> list[dict[str, Any]]:
        """Queue one add-file job per upload; optionally classify each afterwards
        (``classify`` = ``none`` | ``existing`` | ``existing-or-new``)."""
        if classify not in ("none", *CLASSIFY_MODES):
            raise InvalidArgument(f"classify must be 'none' or one of {list(CLASSIFY_MODES)}")
        if text_mode is not None and text_mode not in TEXT_MODES:
            raise InvalidArgument(f"text_mode must be one of {list(TEXT_MODES)}")
        mode = text_mode or services.registry.settings(org_id).text_mode
        queued = []
        for upload in files:
            filename = upload.filename or "upload.pdf"
            staged = ops.stage_upload(filename, upload.file.read())
            queued.append(
                services.jobs.submit(
                    org_id,
                    "add_file",
                    _add_file_fn(services, org_id, ws, staged, mode, classify),
                    label=f"Add {filename}",
                    params={"filename": filename, "text_mode": mode, "classify": classify},
                )
            )
        return queued

    @app.get("/api/orgs/{org_id}/files/{file_id}")
    def get_file(file_id: str, org_id: OrgDep, ws: WorkspaceDep, services: ServicesDep) -> Any:
        record = FileStore(ws).get(file_id)
        docsets = [
            {"id": a.docset.id, "name": a.docset.name}
            for a in DocSetStore(ws).docsets_for_file(file_id)
        ]
        out = file_json(record, docsets)
        out["jobs"] = services.jobs.list_jobs(org_id, file_id=file_id, limit=20)
        return out

    @app.delete("/api/orgs/{org_id}/files/{file_id}", status_code=204)
    def delete_file(
        file_id: str, org_id: OrgDep, ws: WorkspaceDep, services: ServicesDep
    ) -> Response:
        with services.registry.file_lock(org_id, file_id):
            FileStore(ws).delete(file_id)
        return Response(status_code=204)

    @app.get("/api/orgs/{org_id}/files/{file_id}/source")
    def file_source(file_id: str, ws: WorkspaceDep) -> Response:
        record = FileStore(ws).get(file_id)
        key = layout.file_pdf_key(file_id, record.original_filename)
        if not ws.blobs.blob_exists(key):
            raise HTTPException(404, "the file's PDF is not in the workspace")
        return Response(
            ws.blobs.get_blob(key),
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{Path(key).name}"'},
        )

    @app.get("/api/orgs/{org_id}/files/{file_id}/pages/{page}")
    def page_image(file_id: str, page: int, ws: WorkspaceDep) -> Response:
        key = layout.file_page_image_key(file_id, page)
        try:
            data = ws.blobs.get_blob(key)
        except FileNotFoundError:
            raise HTTPException(404, f"no page image {page} for file {file_id!r}") from None
        return Response(data, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/orgs/{org_id}/files/{file_id}/classify", status_code=202)
    def classify_file(
        file_id: str, body: ClassifyBody, org_id: OrgDep, ws: WorkspaceDep, services: ServicesDep
    ) -> dict[str, Any]:
        if body.mode not in CLASSIFY_MODES:
            raise InvalidArgument(f"mode must be one of {list(CLASSIFY_MODES)}")
        record = FileStore(ws).get(file_id)

        def run() -> dict[str, Any]:
            with services.registry.file_lock(org_id, file_id):
                return ops.classify(ws, file_id, mode=body.mode, extract=body.extract)

        return services.jobs.submit(
            org_id,
            "classify",
            run,
            file_id=file_id,
            label=f"Classify {record.original_filename}",
            params={"mode": body.mode, "extract": body.extract},
        )

    # ---- data explorer (read-only) ----

    @app.get("/api/orgs/{org_id}/explore/s3")
    def explore_s3(
        org_id: OrgDep, services: ServicesDep, prefix: str = "", recursive: bool = False
    ) -> dict[str, Any]:
        return explorer.list_s3(services.registry.settings(org_id), prefix, recursive=recursive)

    @app.get("/api/orgs/{org_id}/explore/s3/object")
    def explore_s3_object(org_id: OrgDep, services: ServicesDep, key: str) -> Response:
        try:
            data, ctype = explorer.get_s3_object(services.registry.settings(org_id), key)
        except FileNotFoundError:
            raise HTTPException(404, f"no object {key!r} in the workspace") from None
        # Shown in the browser, never executed as a page of this origin.
        return Response(
            data,
            media_type=ctype,
            headers={
                "Content-Disposition": f'inline; filename="{Path(key).name}"',
                "Content-Security-Policy": "sandbox",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/orgs/{org_id}/explore/db")
    def explore_tables(org_id: OrgDep, services: ServicesDep) -> list[dict[str, Any]]:
        return explorer.list_tables(services.engine, org_id)

    @app.get("/api/orgs/{org_id}/explore/db/{table}")
    def explore_table(
        table: str, org_id: OrgDep, services: ServicesDep, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        return explorer.table_rows(services.engine, org_id, table, limit=limit, offset=offset)

    # ---- jobs ----

    @app.get("/api/orgs/{org_id}/jobs")
    def list_jobs(
        org_id: OrgDep,
        services: ServicesDep,
        file_id: str | None = None,
        docset_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return services.jobs.list_jobs(org_id, file_id=file_id, docset_id=docset_id, limit=limit)

    @app.get("/api/orgs/{org_id}/jobs/{job_id}")
    def get_job(job_id: uuid.UUID, org_id: OrgDep, services: ServicesDep) -> dict[str, Any]:
        job = services.jobs.get(org_id, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job


def docset_json(store: DocSetStore, docset_id: str) -> dict[str, Any]:
    """A docset with its file ids and whether it has a schema / guidance."""
    ds = store.get(docset_id)
    return {
        **ds.to_json(),
        "file_ids": store.list_files(docset_id),
        "has_schema": store.has_schema(docset_id),
        "has_guidance": store.has_guidance(docset_id),
    }


def file_json(record: FileRecord, docsets: list[dict[str, str]]) -> dict[str, Any]:
    """A file record plus the docsets (``{id, name}``) it is assigned to."""
    return {**record.to_json(), "docsets": docsets}


def _extract_job(
    services: Services,
    org_id: uuid.UUID,
    ws: Workspace,
    docset_id: str,
    file_id: str,
    *,
    assign: bool,
) -> dict[str, Any]:
    record = FileStore(ws).get(file_id)

    def run() -> dict[str, Any]:
        with services.registry.file_lock(org_id, file_id):
            if assign:
                DocSetStore(ws).add_file(docset_id, file_id)
            return ops.extract(ws, docset_id, file_id)

    return services.jobs.submit(
        org_id,
        "extract",
        run,
        file_id=file_id,
        docset_id=docset_id,
        label=f"Extract {record.original_filename}",
    )


def _add_file_fn(
    services: Services,
    org_id: uuid.UUID,
    ws: Workspace,
    staged: Path,
    text_mode: str,
    classify: str,
) -> JobFn:
    def run() -> dict[str, Any]:
        try:
            result = ops.add_file(ws, staged, text_mode=text_mode)
        finally:
            ops.discard_upload(staged)
        out = ops.add_result_json(result)
        file_id = result.record.id
        if classify != "none" and result.page_render_error is None:
            # Classification is a vision call over the page images, so it needs the
            # render to have succeeded. A classify failure does not undo the add.
            try:
                with services.registry.file_lock(org_id, file_id):
                    out["classification"] = ops.classify(ws, file_id, mode=classify)
            except DgmlError as exc:
                out["classification"] = {"error": f"{exc.code}: {exc}"}
        return out

    return run


def _frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built single-page app, falling back to ``index.html`` for its routes."""
    index = dist / "index.html"

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> Response:
        target = (dist / path).resolve()
        if path and target.is_file() and dist.resolve() in target.parents:
            return FileResponse(target)
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        return FileResponse(index)
