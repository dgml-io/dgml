"""The HTTP API end to end: organisations, settings, docsets, files, jobs, viewer data.

LLM-backed calls (classification, extraction) are monkeypatched at the
service's seam (:mod:`dgml_sample_service.operations`), so no model is called.
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from dgml_core import ClassificationDecision, layout
from dgml_core.grounded import ExtractionResult
from dgml_sample_service import operations
from dgml_sample_service.db import dgml_settings
from dgml_sample_service.settings_store import ORG_TABLES
from fastapi.testclient import TestClient
from sqlalchemy.engine import Engine

RNC = """\
namespace docset = "http://dgml.io/acme/Leases"

MonthlyRent =
  element docset:MonthlyRent {
    text
  }
"""


def _upload(client: TestClient, org_id: str, path: Any, **form: str) -> list[dict[str, Any]]:
    with path.open("rb") as fh:
        resp = client.post(
            f"/api/orgs/{org_id}/files",
            files=[("files", (path.name, fh, "application/pdf"))],
            data=form,
        )
    assert resp.status_code == 202, resp.text
    jobs: list[dict[str, Any]] = resp.json()
    return jobs


def _add_file(client: TestClient, org_id: str, pdf: Any, wait_job: Any) -> str:
    [job] = _upload(client, org_id, pdf("lease.pdf", ["Lease agreement", "Rent: 1,000 USD"]))
    done = wait_job(org_id, job["id"])
    assert done["status"] == "succeeded", done
    file_id: str = done["result"]["file"]["id"]
    return file_id


def test_orgs(client: TestClient, org: dict[str, Any]) -> None:
    assert [o["slug"] for o in client.get("/api/orgs").json()] == ["acme"]
    bad = client.post("/api/orgs", json={"name": "X", "slug": "Not A Slug"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "SETTINGS_INVALID"
    dup = client.post("/api/orgs", json={"name": "Again", "slug": "acme"})
    assert dup.status_code == 422
    missing = client.get("/api/orgs/00000000-0000-0000-0000-000000000000/docsets")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "ORG_NOT_FOUND"


def test_settings_secrets_are_write_only(client: TestClient, org: dict[str, Any]) -> None:
    url = f"/api/orgs/{org['id']}/settings"
    settings = client.get(url).json()
    assert settings["llm_family"] == "anthropic"
    assert settings["secrets"]["llm_api_key"]["set"] is False
    assert settings["storage_path"] == f"s3://dgml-sample-test/{org['id']}/"

    resp = client.put(url, json={"llm_family": "openai", "secrets": {"llm_api_key": "sk-123"}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["llm_family"] == "openai"
    assert body["secrets"]["llm_api_key"]["set"] is True
    assert "sk-123" not in resp.text

    assert client.put(url, json={"llm_family": "nope"}).status_code == 422
    cleared = client.put(url, json={"secrets": {"llm_api_key": ""}}).json()
    assert cleared["secrets"]["llm_api_key"]["set"] is False


def test_settings_reach_the_workspace_configuration(
    client: TestClient, org: dict[str, Any]
) -> None:
    client.put(
        f"/api/orgs/{org['id']}/settings",
        json={"llm_family": "google", "secrets": {"llm_api_key": "g-key"}},
    )
    services = client.app.state.services  # type: ignore[attr-defined]
    import uuid

    ws = services.registry.open(uuid.UUID(org["id"]))
    models = ws.config.models
    assert models.light.model.startswith("gemini/")
    assert models.light.api_key == "g-key"
    assert ws.organization == "acme"


def test_storage_probe_and_bucket_creation(client: TestClient, org: dict[str, Any]) -> None:
    url = f"/api/orgs/{org['id']}/settings"
    missing = client.put(url, json={"s3_bucket": "not-there"})
    assert missing.status_code == 422 and missing.json()["error"]["code"] == "STORAGE_PROBE_FAILED"
    created = client.put(url, json={"s3_bucket": "made-now", "create_bucket": True})
    assert created.status_code == 200, created.text
    assert created.json()["s3_bucket"] == "made-now"


def test_docset_crud_schema_and_guidance(client: TestClient, org: dict[str, Any]) -> None:
    base = f"/api/orgs/{org['id']}/docsets"
    ds = client.post(base, json={"name": "Leases", "key_questions": ["What is the rent?"]}).json()
    assert ds["has_schema"] is False and ds["file_ids"] == []

    patched = client.patch(f"{base}/{ds['id']}", json={"description": "Commercial leases"}).json()
    assert patched["description"] == "Commercial leases"
    assert patched["key_questions"] == ["What is the rent?"]

    assert client.get(f"{base}/{ds['id']}/schema").status_code == 404
    bad = client.put(f"{base}/{ds['id']}/schema", json={"text": "not rnc"})
    assert bad.status_code == 422, bad.text
    ok = client.put(f"{base}/{ds['id']}/schema", json={"text": RNC})
    assert ok.status_code == 200, ok.text
    assert client.get(f"{base}/{ds['id']}").json()["has_schema"] is True

    client.put(f"{base}/{ds['id']}/guidance", json={"text": "Rent is monthly."})
    assert client.get(f"{base}/{ds['id']}/guidance").json()["guidance"] == "Rent is monthly."

    assert [d["name"] for d in client.get(base).json()] == ["Leases"]
    assert client.delete(f"{base}/{ds['id']}").status_code == 204
    assert client.get(f"{base}/{ds['id']}").status_code == 404


def test_upload_view_and_storage_lock(
    client: TestClient, org: dict[str, Any], pdf: Any, wait_job: Any
) -> None:
    org_id = org["id"]
    file_id = _add_file(client, org_id, pdf, wait_job)

    files = client.get(f"/api/orgs/{org_id}/files").json()
    assert [f["id"] for f in files] == [file_id]
    assert files[0]["page_count"] == 1 and files[0]["page_image_renderer"] == "pypdfium2"

    page = client.get(f"/api/orgs/{org_id}/files/{file_id}/pages/1")
    assert page.status_code == 200 and page.content.startswith(b"\x89PNG")
    assert client.get(f"/api/orgs/{org_id}/files/{file_id}/pages/9").status_code == 404
    source = client.get(f"/api/orgs/{org_id}/files/{file_id}/source")
    assert source.status_code == 200 and source.content.startswith(b"%PDF")

    detail = client.get(f"/api/orgs/{org_id}/files/{file_id}").json()
    assert detail["jobs"] == []  # the add job ran before the file had an id

    # Storage locks once the workspace holds a file; credentials and models do not.
    url = f"/api/orgs/{org_id}/settings"
    assert client.get(url).json()["storage_locked"] is True
    locked = client.put(url, json={"blob_folder": "elsewhere"})
    assert locked.status_code == 409 and locked.json()["error"]["code"] == "STORAGE_LOCKED"
    assert client.put(url, json={"llm_family": "google"}).status_code == 200

    assert client.delete(f"/api/orgs/{org_id}/files/{file_id}").status_code == 204
    assert client.get(f"/api/orgs/{org_id}/files").json() == []
    assert client.get(url).json()["storage_locked"] is False


def test_upload_validates_its_form_before_queueing(
    client: TestClient, org: dict[str, Any], pdf: Any
) -> None:
    path = pdf("a.pdf", ["Lease"])
    for form in ({"text_mode": "pdf"}, {"classify": "maybe"}):
        with path.open("rb") as fh:
            resp = client.post(
                f"/api/orgs/{org['id']}/files",
                files=[("files", (path.name, fh, "application/pdf"))],
                data=form,
            )
        assert resp.status_code == 422 and resp.json()["error"]["code"] == "INVALID_ARGUMENT"
    assert client.get(f"/api/orgs/{org['id']}/jobs").json() == []


def test_upload_rejects_non_pdf(
    client: TestClient, org: dict[str, Any], tmp_path: Any, wait_job: Any
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    [job] = _upload(client, org["id"], path)
    done = wait_job(org["id"], job["id"])
    assert done["status"] == "failed"
    assert done["error_code"] == "UNSUPPORTED_FILE_TYPE"


def test_classify_assigns_and_extracts(
    client: TestClient,
    org: dict[str, Any],
    pdf: Any,
    wait_job: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id = org["id"]
    base = f"/api/orgs/{org_id}/docsets"
    ds = client.post(base, json={"name": "Leases", "description": "Commercial leases"}).json()
    client.put(f"{base}/{ds['id']}/schema", json={"text": RNC})
    file_id = _add_file(client, org_id, pdf, wait_job)

    seen: dict[str, Any] = {}

    def fake_classify(ws: Any, fid: str, *, config: Any, mode: Any) -> ClassificationDecision:
        seen["mode"], seen["model"] = mode, config.model
        return ClassificationDecision(decision="existing", existing_docset_id=ds["id"])

    def fake_extract(ws: Any, docset_id: str, fid: str, **_: Any) -> ExtractionResult:
        key = layout.dgml_xml_key(docset_id, fid, "lease")
        ws.blobs.put_blob(key, b"<Lease xmlns='http://dgml.io/acme/Leases'/>")
        return ExtractionResult(
            values={"MonthlyRent": {}},
            tool_calls=2,
            xml_key=key,
            mode="extraction",
            model="anthropic/claude-sonnet-5",
        )

    monkeypatch.setattr(operations, "classify_file", fake_classify)
    monkeypatch.setattr("dgml_core.extraction.extract_file", fake_extract)
    monkeypatch.setattr(
        "dgml_core.extraction._auto_extract",
        lambda ws, d, f, **kw: {
            "performed": True,
            "error": None,
            "model": fake_extract(ws, d, f).model,
            "tool_calls": 2,
        },
    )

    job = client.post(
        f"/api/orgs/{org_id}/files/{file_id}/classify", json={"mode": "existing"}
    ).json()
    done = wait_job(org_id, job["id"])
    assert done["status"] == "succeeded", done
    assert done["result"]["decision"] == "existing"
    assert done["result"]["docset_id"] == ds["id"]
    assert done["result"]["extraction"]["performed"] is True
    assert str(seen["mode"]) == "existing"
    assert seen["model"] == "anthropic/claude-haiku-4-5"  # the light tier of the family

    detail = client.get(f"{base}/{ds['id']}").json()
    assert detail["file_ids"] == [file_id] and detail["files"][0]["has_dgml"] is True
    files = client.get(f"/api/orgs/{org_id}/files").json()
    assert files[0]["docsets"][0]["id"] == ds["id"]

    pair = client.get(f"{base}/{ds['id']}/files/{file_id}/dgml").json()
    assert pair["xml"].startswith("<Lease") and pair["has_extraction"] is False
    xml = client.get(f"{base}/{ds['id']}/files/{file_id}/dgml.xml")
    assert xml.headers["content-type"].startswith("application/xml")

    # Explicit (re-)extraction goes through extract_file.
    monkeypatch.setattr(operations, "extract_file", fake_extract)
    job = client.post(f"{base}/{ds['id']}/files/{file_id}/extract").json()
    done = wait_job(org_id, job["id"])
    assert done["status"] == "succeeded", done
    assert done["result"]["tool_calls"] == 2

    jobs = client.get(f"/api/orgs/{org_id}/jobs", params={"file_id": file_id}).json()
    assert {j["kind"] for j in jobs} == {"classify", "extract"}

    assert client.delete(f"{base}/{ds['id']}/files/{file_id}").status_code == 204
    assert client.get(f"{base}/{ds['id']}").json()["file_ids"] == []


def test_classify_none_leaves_file_unassigned(
    client: TestClient,
    org: dict[str, Any],
    pdf: Any,
    wait_job: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id = org["id"]
    client.post(f"/api/orgs/{org_id}/docsets", json={"name": "Invoices"})
    file_id = _add_file(client, org_id, pdf, wait_job)
    monkeypatch.setattr(
        operations,
        "classify_file",
        lambda *a, **k: ClassificationDecision(decision="none", reason="a lease, not an invoice"),
    )
    job = client.post(f"/api/orgs/{org_id}/files/{file_id}/classify", json={}).json()
    done = wait_job(org_id, job["id"])
    assert done["result"] == {
        "decision": "none",
        "reason": "a lease, not an invoice",
        "docset_id": None,
        "created_docset": False,
        "extraction": None,
    }
    assert client.get(f"/api/orgs/{org_id}/files").json()[0]["docsets"] == []


def test_classify_new_creates_docset(
    client: TestClient,
    org: dict[str, Any],
    pdf: Any,
    wait_job: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    org_id = org["id"]
    monkeypatch.setattr(
        operations,
        "classify_file",
        lambda *a, **k: ClassificationDecision(
            decision="new",
            new_name="Lease",
            new_description="A lease",
            new_key_questions=("What is the rent?",),
        ),
    )
    [job] = _upload(client, org_id, pdf("a.pdf", ["Lease"]), classify="existing-or-new")
    done = wait_job(org_id, job["id"])
    assert done["status"] == "succeeded", done
    result = done["result"]["classification"]
    assert result["decision"] == "new" and result["created_docset"] is True
    docsets = client.get(f"/api/orgs/{org_id}/docsets").json()
    assert [(d["name"], d["key_questions"]) for d in docsets] == [("Lease", ["What is the rent?"])]
    assert docsets[0]["file_ids"] == [done["result"]["file"]["id"]]


def test_classify_without_docsets_fails_the_step_not_the_add(
    client: TestClient, org: dict[str, Any], pdf: Any, wait_job: Any
) -> None:
    [job] = _upload(client, org["id"], pdf("a.pdf", ["Lease"]), classify="existing")
    done = wait_job(org["id"], job["id"])
    assert done["status"] == "succeeded"
    assert done["result"]["classification"]["error"].startswith("NO_EXISTING_DOCSETS")


def test_delete_org_removes_its_data(
    client: TestClient, org: dict[str, Any], pdf: Any, wait_job: Any
) -> None:
    _add_file(client, org["id"], pdf, wait_job)
    assert client.delete(f"/api/orgs/{org['id']}").status_code == 204
    assert client.get("/api/orgs").json() == []
    import boto3

    listing = boto3.client("s3").list_objects_v2(Bucket="dgml-sample-test", Prefix=org["id"])
    assert listing.get("KeyCount", 0) == 0


def test_delete_org_clears_dgml_rows_when_storage_is_gone(
    client: TestClient, engine: Engine, org: dict[str, Any], pdf: Any, wait_job: Any
) -> None:
    _add_file(client, org["id"], pdf, wait_job)
    client.post(f"/api/orgs/{org['id']}/docsets", json={"name": "Leases"})
    # The workspace can no longer be opened (its settings row is gone) ...
    with engine.begin() as conn:
        conn.execute(sa.delete(dgml_settings))
    # ... yet deleting the organisation still leaves none of its rows behind.
    assert client.delete(f"/api/orgs/{org['id']}").status_code == 204
    with engine.connect() as conn:
        for table in ORG_TABLES:
            count = conn.execute(sa.select(sa.func.count()).select_from(table)).scalar_one()
            assert count == 0, table.name
