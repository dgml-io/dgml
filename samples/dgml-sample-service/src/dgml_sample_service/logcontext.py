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

"""The workspace id on every log line — this service's own and ``dgml_core``'s.

dgml-core logs through stdlib ``logging`` and never configures it; the host
decides what a line carries. This is how a host adds an id of its own (here the
workspace id, which is the organisation's UUID) to every record, dgml-core's
included, without dgml-core knowing about it:

- a :class:`~contextvars.ContextVar` holds the id for the work in progress. It is
  bound per request (:class:`WorkspaceLogContextMiddleware`) and per background
  job (:class:`~dgml_sample_service.jobs.JobRunner`), with
  :func:`bind_workspace_id`;
- a :class:`WorkspaceIdFilter` on the *handler* copies it onto every record the
  handler emits, so the format can use ``%(workspace_id)s``. On a handler, not a
  logger: a logger's filters run only for records created on that very logger,
  so one on ``dgml_core`` would miss ``dgml_core.extraction`` and the rest.

Context variables follow the work: into each request's task, into the thread
FastAPI runs a sync endpoint on, into a job (the runner copies the submitting
context), and into dgml-core's own worker threads (phase-3 pages, transcription),
which carry the caller's context too. Any other id — a job id, a request id —
works the same way with a variable and filter of its own.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

#: Shown on records logged outside any workspace (startup, the org list).
NO_WORKSPACE = "-"

#: The format :func:`configure_logging` installs.
LOG_FORMAT = "%(asctime)s %(levelname)s [%(workspace_id)s] %(name)s: %(message)s"

_WORKSPACE_ID: ContextVar[str | None] = ContextVar("dgml_sample_workspace_id", default=None)

# The organisation-scoped routes: /api/orgs/<uuid> and everything under it.
_ORG_PATH = re.compile(r"^/api/orgs/([^/]+)(?:/|$)")


def current_workspace_id() -> str | None:
    """The workspace id bound for the work in progress, if any."""
    return _WORKSPACE_ID.get()


@contextmanager
def bind_workspace_id(workspace_id: uuid.UUID | str | None) -> Iterator[None]:
    """Stamp *workspace_id* on every record logged inside the block."""
    token = _WORKSPACE_ID.set(str(workspace_id) if workspace_id is not None else None)
    try:
        yield
    finally:
        _WORKSPACE_ID.reset(token)


class WorkspaceIdFilter(logging.Filter):
    """Copies the bound workspace id onto each record as ``record.workspace_id``
    (:data:`NO_WORKSPACE` when none is bound). Never drops a record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.workspace_id = _WORKSPACE_ID.get() or NO_WORKSPACE
        return True


def workspace_id_from_path(path: str) -> str | None:
    """The organisation id in an ``/api/orgs/<id>/…`` path, when it is a UUID.

    A path with something else there is left unbound; the route rejects it."""
    match = _ORG_PATH.match(path)
    if match is None:
        return None
    try:
        return str(uuid.UUID(match.group(1)))
    except ValueError:
        return None


class WorkspaceLogContextMiddleware:
    """Binds the workspace id of an organisation-scoped request for its duration.

    Pure ASGI rather than a dependency: a sync dependency runs on a worker thread
    in a context of its own, so a value it binds never reaches the endpoint. This
    runs in the request's task, which the endpoint's thread inherits from."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        workspace_id = workspace_id_from_path(scope["path"]) if scope["type"] == "http" else None
        if workspace_id is None:
            await self.app(scope, receive, send)
            return
        with bind_workspace_id(workspace_id):
            await self.app(scope, receive, send)


def configure_logging(level: int = logging.INFO) -> None:
    """Route every logger to stderr with the workspace id on each line."""
    handler = logging.StreamHandler()
    handler.addFilter(WorkspaceIdFilter())
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.basicConfig(level=level, handlers=[handler])
