"""Fixtures for the sample service.

Everything runs offline and in-process, like the rest of the suite:

- **SQLite** (a file per test) stands in for Postgres. The DocStore and service tables use
  only portable SQLAlchemy Core — ``text[]`` falls back to JSON and the upsert picks
  the dialect's ``ON CONFLICT`` — so the same code paths run.
- **moto** stands in for S3 (the bucket the default organisation settings name).
- LLM calls are monkeypatched by the tests that need them.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from dgml_sample_service import PostgresDocStore, ServiceConfig, create_app
from dgml_sample_service.db import create_schema, make_engine
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

BUCKET = "dgml-sample-test"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """No user config, no ambient model keys — the service passes everything in."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))
    monkeypatch.delenv("DGML_HOME", raising=False)
    for var in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _fake_s3(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from moto import mock_aws

    for var, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
    }.items():
        monkeypatch.setenv(var, value)
    with mock_aws():
        import boto3

        boto3.client("s3").create_bucket(Bucket=BUCKET)
        yield


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    # A file, not :memory: — the job threads need connections of their own.
    eng = make_engine(f"sqlite+pysqlite:///{tmp_path / 'service.db'}")
    create_schema(eng)
    PostgresDocStore.bind_engine(eng)
    yield eng
    PostgresDocStore.bind_engine(None)
    eng.dispose()


@pytest.fixture
def client(engine: Engine) -> Iterator[TestClient]:
    config = ServiceConfig(
        database_url="sqlite://",  # unused: the engine is passed in
        default_s3_endpoint=None,  # moto intercepts the default AWS endpoint
        default_s3_bucket=BUCKET,
        job_workers=2,
    )
    with TestClient(create_app(config, engine=engine)) as c:
        yield c


@pytest.fixture
def org(client: TestClient) -> dict[str, Any]:
    resp = client.post("/api/orgs", json={"name": "Acme Corp", "slug": "acme"})
    assert resp.status_code == 201, resp.text
    org: dict[str, Any] = resp.json()
    return org


def wait_for_job(client: TestClient, org_id: str, job_id: str, timeout: float = 30) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/orgs/{org_id}/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def make_pdf(path: Path, lines: list[str]) -> Path:
    """A one-page PDF with real (extractable) text, written by hand."""
    text = "\n".join(
        f"BT /F1 14 Tf 72 {720 - 24 * i} Td ({line}) Tj ET" for i, line in enumerate(lines)
    )
    stream = text.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    path.write_bytes(bytes(out))
    return path


@pytest.fixture
def wait_job(client: TestClient) -> Any:
    """``wait_job(org_id, job_id)`` — poll a job until it finishes."""
    return lambda org_id, job_id: wait_for_job(client, org_id, job_id)


@pytest.fixture
def pdf(tmp_path: Path) -> Any:
    """``pdf(name, lines)`` — write a one-page text PDF and return its path."""
    return lambda name, lines: make_pdf(tmp_path / name, lines)
