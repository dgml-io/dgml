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

"""Background jobs for the slow DGML calls.

Adding a file (page rendering, text extraction, OCR), classification, schema
generation and value extraction take seconds to minutes, so the API queues them
and returns a job id the UI polls. dgml-core is synchronous, so jobs run on a
thread pool; each job's state lives in the ``jobs`` table.

The work itself is a closure, so it does not survive a restart: on startup any
job still ``queued`` or ``running`` is marked failed (``INTERRUPTED``) rather
than left spinning forever. A production service would put jobs on a durable
queue instead.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import sqlalchemy as sa
from dgml_core import DgmlError
from sqlalchemy.engine import Engine

from .db import iso, jobs, utcnow

logger = logging.getLogger(__name__)

JobFn = Callable[[], dict[str, Any]]

QUEUED, RUNNING, SUCCEEDED, FAILED = "queued", "running", "succeeded", "failed"


def job_json(row: Any) -> dict[str, Any]:
    """A ``jobs`` row as the API returns it."""
    return {
        "id": str(row["id"]),
        "kind": row["kind"],
        "status": row["status"],
        "file_id": row["file_id"],
        "docset_id": row["docset_id"],
        "label": row["label"],
        "params": row["params"],
        "result": row["result"],
        "error": row["error"],
        "error_code": row["error_code"],
        "created_at": iso(row["created_at"]),
        "started_at": iso(row["started_at"]),
        "finished_at": iso(row["finished_at"]),
    }


class JobRunner:
    """Runs job closures on a thread pool and records their state."""

    def __init__(self, engine: Engine, *, workers: int = 4) -> None:
        self._engine = engine
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dgml-job")

    def recover(self) -> int:
        """Fail every job a previous process left unfinished. Returns the count."""
        with self._engine.begin() as conn:
            result = conn.execute(
                sa.update(jobs)
                .where(jobs.c.status.in_([QUEUED, RUNNING]))
                .values(
                    status=FAILED,
                    error="the service restarted before this job finished; run it again",
                    error_code="INTERRUPTED",
                    finished_at=utcnow(),
                )
            )
            return int(result.rowcount)

    def submit(
        self,
        organisation_id: uuid.UUID,
        kind: str,
        fn: JobFn,
        *,
        file_id: str | None = None,
        docset_id: str | None = None,
        label: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a queued job, schedule ``fn``, and return the job as JSON."""
        job_id = uuid.uuid4()
        with self._engine.begin() as conn:
            conn.execute(
                sa.insert(jobs).values(
                    id=job_id,
                    organisation_id=organisation_id,
                    kind=kind,
                    status=QUEUED,
                    file_id=file_id,
                    docset_id=docset_id,
                    label=label,
                    params=params or {},
                    created_at=utcnow(),
                )
            )
        self._pool.submit(self._run, job_id, fn)
        found = self.get(organisation_id, job_id)
        assert found is not None
        return found

    def _set(self, job_id: uuid.UUID, **values: Any) -> None:
        with self._engine.begin() as conn:
            conn.execute(sa.update(jobs).where(jobs.c.id == job_id).values(**values))

    def _fail(self, job_id: uuid.UUID, error: str, code: str) -> None:
        self._set(job_id, status=FAILED, error=error, error_code=code, finished_at=utcnow())

    def _run(self, job_id: uuid.UUID, fn: JobFn) -> None:
        # Nothing may escape: the pool would swallow the exception and leave the job
        # ``running`` forever. Recording the result is inside the ``try`` too, so a
        # result the JSON column cannot store fails the job instead.
        try:
            self._set(job_id, status=RUNNING, started_at=utcnow())
            result = fn()
            self._set(job_id, status=SUCCEEDED, result=result, finished_at=utcnow())
        except DgmlError as exc:
            logger.warning("job %s failed: %s: %s", job_id, exc.code, exc)
            self._fail(job_id, str(exc), exc.code)
        except Exception as exc:
            logger.exception("job %s crashed", job_id)
            self._fail(job_id, f"{type(exc).__name__}: {exc}", "INTERNAL")

    def get(self, organisation_id: uuid.UUID, job_id: uuid.UUID) -> dict[str, Any] | None:
        with self._engine.connect() as conn:
            row = (
                conn.execute(
                    sa.select(jobs).where(
                        jobs.c.id == job_id, jobs.c.organisation_id == organisation_id
                    )
                )
                .mappings()
                .first()
            )
        return job_json(row) if row is not None else None

    def list_jobs(
        self,
        organisation_id: uuid.UUID,
        *,
        file_id: str | None = None,
        docset_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        stmt = sa.select(jobs).where(jobs.c.organisation_id == organisation_id)
        if file_id is not None:
            stmt = stmt.where(jobs.c.file_id == file_id)
        if docset_id is not None:
            stmt = stmt.where(jobs.c.docset_id == docset_id)
        stmt = stmt.order_by(jobs.c.created_at.desc()).limit(limit)
        with self._engine.connect() as conn:
            return [job_json(row) for row in conn.execute(stmt).mappings()]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
