"""The job runner records every outcome, including a result it cannot store."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from dgml_core import InvalidArgument
from dgml_sample_service.jobs import JobRunner
from sqlalchemy.engine import Engine


def _finished(runner: JobRunner, org: uuid.UUID, job: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        found = runner.get(org, uuid.UUID(job["id"]))
        assert found is not None
        if found["status"] in ("succeeded", "failed"):
            return found
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_outcomes_are_recorded(engine: Engine) -> None:
    runner = JobRunner(engine, workers=1)
    org = uuid.uuid4()
    try:
        ok = _finished(runner, org, runner.submit(org, "extract", lambda: {"n": 1}))
        assert ok["status"] == "succeeded" and ok["result"] == {"n": 1}

        def dgml_error() -> dict[str, Any]:
            raise InvalidArgument("bad input")

        failed = _finished(runner, org, runner.submit(org, "extract", dgml_error))
        assert failed["status"] == "failed" and failed["error_code"] == "INVALID_ARGUMENT"

        # A result the JSON column cannot hold fails the job instead of leaving it running.
        unstorable = runner.submit(org, "extract", lambda: {"at": datetime.now(UTC)})
        crashed = _finished(runner, org, unstorable)
        assert crashed["status"] == "failed" and crashed["error_code"] == "INTERNAL"
    finally:
        runner.shutdown()
