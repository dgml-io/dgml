"""The workspace id on every log line, dgml-core's included (see ``logcontext``).

The end-to-end tests read real dgml-core records — the assignment line
``DocSetStore.add_file`` logs, and auto-extraction's no-schema line — through a
handler set up the way :func:`~dgml_sample_service.logcontext.configure_logging`
sets one up: with :class:`WorkspaceIdFilter` on the handler, not a logger.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest
from dgml_core import ClassificationDecision, DecisionKind
from dgml_sample_service import operations
from dgml_sample_service.logcontext import (
    NO_WORKSPACE,
    WorkspaceIdFilter,
    bind_workspace_id,
    workspace_id_from_path,
)
from fastapi.testclient import TestClient

ORG = "0b9c8d39-2f4c-4c3a-9a7e-1d2f3a4b5c6d"


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []
        self.addFilter(WorkspaceIdFilter())

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def stamped(self, logger: str, contains: str) -> list[str]:
        """The workspace id on each record from *logger* whose message has *contains*."""
        return [
            r.workspace_id  # type: ignore[attr-defined]
            for r in self.records
            if r.name == logger and contains in r.getMessage()
        ]


@pytest.fixture
def dgml_lines() -> Iterator[_Collect]:
    """dgml-core's records, through a handler on the parent logger only — so the
    filter must catch records from its children (dgml_core.docsets, …)."""
    handler = _Collect()
    lib = logging.getLogger("dgml_core")
    level = lib.level
    lib.addHandler(handler)
    lib.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        lib.removeHandler(handler)
        lib.setLevel(level)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (f"/api/orgs/{ORG}", ORG),
        (f"/api/orgs/{ORG}/docsets/ds1/files/f1", ORG),
        (f"/api/orgs/{ORG.upper()}/files", ORG),  # normalised, as the route parses it
        ("/api/orgs", None),
        ("/api/orgs/not-a-uuid/files", None),
        ("/api/health", None),
    ],
)
def test_workspace_id_from_path(path: str, expected: str | None) -> None:
    assert workspace_id_from_path(path) == expected


def test_filter_stamps_the_bound_id_and_a_placeholder_outside_one() -> None:
    def stamp() -> str:
        record = logging.LogRecord("dgml_core.x", logging.INFO, __file__, 1, "m", None, None)
        assert WorkspaceIdFilter().filter(record) is True  # never drops a record
        return str(record.workspace_id)  # type: ignore[attr-defined]

    assert stamp() == NO_WORKSPACE
    with bind_workspace_id(ORG):
        assert stamp() == ORG
    assert stamp() == NO_WORKSPACE


def test_request_lines_carry_the_workspace_id(
    client: TestClient, org: dict[str, Any], pdf: Any, wait_job: Any, dgml_lines: _Collect
) -> None:
    """A sync endpoint runs on a worker thread; the id the middleware bound for
    the request still reaches dgml-core's lines logged there."""
    org_id = org["id"]
    ds = client.post(f"/api/orgs/{org_id}/docsets", json={"name": "Leases"}).json()
    [job] = client.post(
        f"/api/orgs/{org_id}/files",
        files=[("files", ("a.pdf", pdf("a.pdf", ["Lease"]).read_bytes(), "application/pdf"))],
    ).json()
    file_id = wait_job(org_id, job["id"])["result"]["file"]["id"]

    resp = client.post(
        f"/api/orgs/{org_id}/docsets/{ds['id']}/files/{file_id}", json={"extract": False}
    )
    assert resp.status_code == 200, resp.text

    assert dgml_lines.stamped("dgml_core.docsets", f"assigned file {file_id}") == [org_id]


def test_job_lines_carry_the_workspace_id(
    client: TestClient,
    org: dict[str, Any],
    pdf: Any,
    wait_job: Any,
    dgml_lines: _Collect,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job runs on the runner's own pool; dgml-core's lines from it — here the
    assignment and auto-extraction's no-schema line — carry the job's org."""
    org_id = org["id"]
    ds = client.post(f"/api/orgs/{org_id}/docsets", json={"name": "Leases"}).json()
    [job] = client.post(
        f"/api/orgs/{org_id}/files",
        files=[("files", ("a.pdf", pdf("a.pdf", ["Lease"]).read_bytes(), "application/pdf"))],
    ).json()
    file_id = wait_job(org_id, job["id"])["result"]["file"]["id"]
    monkeypatch.setattr(
        operations,
        "classify_file",
        lambda ws, fid, **kw: ClassificationDecision(
            decision=DecisionKind.EXISTING, existing_docset_id=ds["id"]
        ),
    )

    job = client.post(
        f"/api/orgs/{org_id}/files/{file_id}/classify", json={"mode": "existing"}
    ).json()
    assert wait_job(org_id, job["id"])["status"] == "succeeded"

    assert dgml_lines.stamped("dgml_core.docsets", f"assigned file {file_id}") == [org_id]
    assert dgml_lines.stamped("dgml_core.extraction", "has no extraction schema") == [org_id]
