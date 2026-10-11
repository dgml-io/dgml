"""PostgresDocStore: the strict, typed mapping of DGML's state collections."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from dgml_core import (
    Configuration,
    DocSet,
    DocSetStore,
    FileRecord,
    FileStore,
    Identity,
    InvalidArgument,
    ProviderSpec,
    Storage,
    StorageConfig,
    StorageConfigInvalid,
    Workspace,
)
from dgml_core.configuration import Pdf
from dgml_core.layout import Collection
from dgml_sample_service import PostgresDocStore
from sqlalchemy.engine import Engine

PROVIDER = "dgml_sample_service.docstore:PostgresDocStore"


def _store(org: uuid.UUID) -> PostgresDocStore:
    cfg = StorageConfig(PROVIDER, Path("."), {"organisation_id": str(org)}, str(org))
    return PostgresDocStore(PostgresDocStore.parse_config(cfg))


@pytest.fixture
def store(engine: Engine) -> PostgresDocStore:
    return _store(uuid.uuid4())


FILE = FileRecord(
    id="f-1",
    original_path="../files/report.pdf",
    original_filename="report.pdf",
    sha256="ab" * 32,
    added_at="2026-10-09T12:34:56Z",
    page_count=3,
    text_mode="digital",
    page_image_dpi=300,
    page_image_renderer="pypdfium2",
    pdf_converter=None,
).to_json()


def test_state_records_round_trip_exactly(store: PostgresDocStore) -> None:
    docset = DocSet(id="ds-1", name="Leases", description="d", key_questions=["Who?"]).to_json()
    assignment = {"docset_id": "ds-1", "file_id": "f-1", "assigned_at": "2026-10-09T01:02:03Z"}
    store.put_doc(Collection.FILES, "f-1", FILE)
    store.put_doc(Collection.DOCSETS, "ds-1", docset)
    store.put_doc(Collection.ASSIGNMENTS, "ds-1/f-1", assignment)

    assert store.get_doc(Collection.FILES, "f-1") == FILE
    assert store.get_doc(Collection.DOCSETS, "ds-1") == docset
    assert store.get_doc(Collection.ASSIGNMENTS, "ds-1/f-1") == assignment
    assert store.find_docs(Collection.FILES, {}) == [FILE]
    assert store.find_docs(Collection.ASSIGNMENTS, {"file_id": "f-1"}) == [assignment]
    assert store.find_docs(Collection.ASSIGNMENTS, {"docset_id": "nope"}) == []


def test_put_replaces_not_merges(store: PostgresDocStore) -> None:
    store.put_doc(
        Collection.DOCSETS,
        "ds-1",
        {"id": "ds-1", "name": "A", "description": "x", "key_questions": ["q"]},
    )
    store.put_doc(
        Collection.DOCSETS,
        "ds-1",
        {"id": "ds-1", "name": "B", "description": "", "key_questions": []},
    )
    assert store.get_doc(Collection.DOCSETS, "ds-1") == {
        "id": "ds-1",
        "name": "B",
        "description": "",
        "key_questions": [],
    }


def test_assignment_without_timestamp(store: PostgresDocStore) -> None:
    # DGML's layout migration writes assignments with no assigned_at.
    store.put_doc(Collection.ASSIGNMENTS, "d/f", {"docset_id": "d", "file_id": "f"})
    assert store.get_doc(Collection.ASSIGNMENTS, "d/f") == {
        "docset_id": "d",
        "file_id": "f",
        "assigned_at": None,
    }


def test_workspace_meta(store: PostgresDocStore) -> None:
    org = str(store._org)
    store.put_doc(
        Collection.WORKSPACE,
        "workspace",
        {"name": "Acme", "organization": "acme", "workspace_id": org},
    )
    assert store.get_doc(Collection.WORKSPACE, "workspace") == {
        "name": "Acme",
        "organization": "acme",
        "workspace_id": org,
    }
    store.put_doc(
        Collection.WORKSPACE,
        "workspace",
        {"name": "Acme", "organization": "acme", "workspace_id": org, "schema_version": 4},
    )
    assert store.get_doc(Collection.WORKSPACE, "workspace")["schema_version"] == 4  # type: ignore[index]
    with pytest.raises(InvalidArgument, match="not this store's organisation"):
        store.put_doc(
            Collection.WORKSPACE,
            "workspace",
            {"name": "x", "organization": "y", "workspace_id": "someone-else"},
        )


def test_delete(store: PostgresDocStore) -> None:
    for fid in ("f-1", "f-2"):
        store.put_doc(Collection.ASSIGNMENTS, f"d/{fid}", {"docset_id": "d", "file_id": fid})
    store.delete_doc(Collection.ASSIGNMENTS, "d/f-1")
    store.delete_doc(Collection.ASSIGNMENTS, "d/missing")  # no-op
    assert store.delete_docs(Collection.ASSIGNMENTS, {"docset_id": "d"}) == 1
    assert store.find_docs(Collection.ASSIGNMENTS, {}) == []


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda s: s.put_doc(Collection.FILES, "f-1", {**FILE, "new_field": 1}), "unknown field"),
        (
            lambda s: s.put_doc(
                Collection.FILES, "f-1", {k: v for k, v in FILE.items() if k != "sha256"}
            ),
            "missing field",
        ),
        (lambda s: s.put_doc("mystery", "x", {}), "does not hold collection"),
        (lambda s: s.find_docs(Collection.FILES, {"sha256": "x"}), "cannot query"),
        (
            lambda s: s.put_doc(Collection.FILES, "f-1", {**FILE, "added_at": "2026-10-09"}),
            "round-trip",
        ),
        (
            lambda s: s.put_doc(
                Collection.ASSIGNMENTS, "no-slash", {"docset_id": "a", "file_id": "b"}
            ),
            "not '<",
        ),
        (lambda s: s.put_doc(Collection.FILES, "f-2", FILE), "does not match"),
        (lambda s: s.append_doc(Collection.FILES, {}), "append-only"),
    ],
)
def test_strict_mapping_raises(store: PostgresDocStore, call: Any, match: str) -> None:
    with pytest.raises(InvalidArgument, match=match):
        call(store)


def test_outlets_are_write_only(store: PostgresDocStore, caplog: pytest.LogCaptureFixture) -> None:
    store.put_doc(Collection.ERRORS, "f-1", {"errors": [{"operation": "render"}]})
    store.put_doc(Collection.EXTRACTION_STATS, "d/f", {"phase1_layout": "x"})
    store.append_doc(Collection.USAGE, {"model": "m", "tokens": 3})
    assert store.get_doc(Collection.ERRORS, "f-1") is None
    assert store.find_docs(Collection.USAGE, {}) == []
    assert store.delete_docs(Collection.EXTRACTION_STATS, {}) == 0
    assert sum("dgml errors" in r.getMessage() for r in caplog.records) == 1


def test_organisations_are_isolated(engine: Engine) -> None:
    a, b = _store(uuid.uuid4()), _store(uuid.uuid4())
    a.put_doc(Collection.FILES, "f-1", FILE)
    assert b.get_doc(Collection.FILES, "f-1") is None
    assert b.find_docs(Collection.FILES, {}) == []
    assert b.delete_docs(Collection.ASSIGNMENTS, {}) == 0


def test_config_requires_org_uuid_matching_workspace(engine: Engine) -> None:
    org = str(uuid.uuid4())
    with pytest.raises(StorageConfigInvalid, match="UUID"):
        PostgresDocStore.parse_config(
            StorageConfig(PROVIDER, Path("."), {"organisation_id": "x"}, org)
        )
    with pytest.raises(StorageConfigInvalid, match="must be the organisation_id"):
        PostgresDocStore.parse_config(
            StorageConfig(PROVIDER, Path("."), {"organisation_id": org}, "other-ws")
        )
    with pytest.raises(StorageConfigInvalid, match="unknown fields"):
        PostgresDocStore.parse_config(
            StorageConfig(PROVIDER, Path("."), {"organisation_id": org, "dsn": "x"}, org)
        )


def test_unbound_engine_is_actionable(engine: Engine) -> None:
    PostgresDocStore.bind_engine(None)
    org = str(uuid.uuid4())
    cfg = StorageConfig(PROVIDER, Path("."), {"organisation_id": org}, org)
    with pytest.raises(StorageConfigInvalid, match="bind_engine"):
        PostgresDocStore(cfg)


def test_workspace_on_postgres_docs_and_s3_blobs(engine: Engine, pdf: Any) -> None:
    """The whole stack DGML-side: an in-memory Configuration, docs in the typed tables,
    blobs in (moto) S3, pages rendered with PDFium."""
    org = str(uuid.uuid4())
    cfg = Configuration.build(
        identity=Identity(workspace_id=org, name="Acme", organization="acme"),
        storage=Storage(
            blobs=ProviderSpec(
                "dgml_storage_s3:S3BlobStore", {"bucket": "dgml-sample-test", "prefix": "tenants"}
            ),
            docs=ProviderSpec(PROVIDER, {"organisation_id": org}),
        ),
        pdf=Pdf(provider="pypdfium2"),
    )
    ws = Workspace.open(configuration=cfg)
    result = FileStore(ws).add(pdf("lease.pdf", ["Lease agreement", "Rent: 1,000 USD"]))
    assert result.page_render_error is None and result.text_extraction_error is None
    assert result.record.page_count == 1
    assert result.record.page_image_renderer == "pypdfium2"

    ds = DocSetStore(ws).create("Leases", "Commercial leases", key_questions=["What is the rent?"])
    DocSetStore(ws).add_file(ds.id, result.record.id)
    assert [a.docset.id for a in DocSetStore(ws).docsets_for_file(result.record.id)] == [ds.id]

    import boto3

    keys = [
        o["Key"]
        for o in boto3.client("s3").list_objects_v2(
            Bucket="dgml-sample-test", Prefix=f"tenants/{org}/"
        )["Contents"]
    ]
    assert any(k.endswith("page_images/page_1.png") for k in keys)

    # Re-opening reads the meta document back and agrees on the identity.
    again = Workspace.open(configuration=cfg)
    assert [f.id for f in FileStore(again).list_all()] == [result.record.id]

    FileStore(ws).delete(result.record.id)
    assert DocSetStore(ws).list_files(ds.id) == []
