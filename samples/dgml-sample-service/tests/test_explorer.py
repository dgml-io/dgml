"""The read-only data explorer: S3 browsing and Postgres tables, scoped to one org."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def _add(client: TestClient, org_id: str, pdf: Any, wait_job: Any) -> str:
    path = pdf("lease.pdf", ["Lease agreement"])
    with path.open("rb") as fh:
        [job] = client.post(
            f"/api/orgs/{org_id}/files", files=[("files", (path.name, fh, "application/pdf"))]
        ).json()
    done = wait_job(org_id, job["id"])
    assert done["status"] == "succeeded", done
    file_id: str = done["result"]["file"]["id"]
    return file_id


def test_s3_browse_and_preview(
    client: TestClient, org: dict[str, Any], pdf: Any, wait_job: Any
) -> None:
    org_id = org["id"]
    file_id = _add(client, org_id, pdf, wait_job)
    base = f"/api/orgs/{org_id}/explore/s3"

    root = client.get(base).json()
    assert root["bucket"] == "dgml-sample-test"
    assert root["root"] == f"{org_id}/"
    assert [f["name"] for f in root["folders"]] == ["files"]

    files = client.get(base, params={"prefix": "files"}).json()
    assert files["folders"] == [{"name": file_id, "prefix": f"files/{file_id}"}]

    one = client.get(base, params={"prefix": f"files/{file_id}"}).json()
    assert {f["name"] for f in one["folders"]} >= {"page_images", "page_text"}
    [source] = [o for o in one["objects"] if o["name"].endswith(".pdf")]
    assert source["size"] > 0

    deep = client.get(base, params={"recursive": True}).json()
    assert deep["folders"] == []
    assert f"files/{file_id}/page_text/page_1.json" in {o["key"] for o in deep["objects"]}

    pages = client.get(base, params={"prefix": f"files/{file_id}/page_images"}).json()
    png = client.get(f"{base}/object", params={"key": pages["objects"][0]["key"]})
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert png.headers["content-security-policy"] == "sandbox"

    text_key = f"files/{file_id}/page_text/page_1.json"
    text = client.get(f"{base}/object", params={"key": text_key})
    assert text.headers["content-type"] == "application/json"
    assert text.json()["page"] == 1

    assert client.get(f"{base}/object", params={"key": "files/nope.pdf"}).status_code == 404


def test_s3_stays_inside_the_workspace_prefix(
    client: TestClient, org: dict[str, Any], pdf: Any, wait_job: Any
) -> None:
    import boto3

    boto3.client("s3").put_object(
        Bucket="dgml-sample-test", Key="other-tenant/secret.txt", Body=b"x"
    )
    base = f"/api/orgs/{org['id']}/explore/s3"
    escaped = client.get(f"{base}/object", params={"key": "../other-tenant/secret.txt"})
    assert escaped.status_code == 404
    listing = client.get(base, params={"prefix": "../.."}).json()
    assert listing["prefix"] == "" and listing["root"] == f"{org['id']}/"


def test_tables_are_scoped_and_secrets_redacted(
    client: TestClient, org: dict[str, Any], pdf: Any, wait_job: Any
) -> None:
    org_id = org["id"]
    other = client.post("/api/orgs", json={"name": "Other", "slug": "other"}).json()
    _add(client, org_id, pdf, wait_job)
    _add(client, other["id"], pdf, wait_job)
    client.put(f"/api/orgs/{org_id}/settings", json={"secrets": {"llm_api_key": "sk-live-123"}})

    tables = {t["name"]: t for t in client.get(f"/api/orgs/{org_id}/explore/db").json()}
    assert tables["dgml_files"]["rows"] == 1
    assert tables["organisations"]["rows"] == 1
    assert tables["dgml_workspaces"]["group"] == "DGML DocStore"
    kq = next(c for c in tables["dgml_docsets"]["columns"] if c["name"] == "key_questions")
    assert kq["type"] == "TEXT[]"

    files = client.get(f"/api/orgs/{org_id}/explore/db/dgml_files").json()
    assert files["total"] == 1 and files["rows"][0]["page_image_renderer"] == "pypdfium2"

    creds = client.get(f"/api/orgs/{org_id}/explore/db/service_credentials")
    assert "sk-live-123" not in creds.text and "dgml-secret" not in creds.text
    assert {r["secret"] for r in creds.json()["rows"]} == {"•••• redacted"}

    jobs = client.get(f"/api/orgs/{org_id}/explore/db/jobs", params={"limit": 1}).json()
    assert jobs["total"] == 1 and len(jobs["rows"]) == 1

    missing = client.get(f"/api/orgs/{org_id}/explore/db/pg_user")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "TABLE_NOT_FOUND"
