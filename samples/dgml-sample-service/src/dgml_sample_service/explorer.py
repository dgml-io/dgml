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

"""Read-only views of the backend data, for the UI's Data page.

- **S3**: browse the organisation's workspace prefix (``<folder>/<org id>/``) in its
  bucket, folder by folder, and fetch an object to preview it. Everything DGML stored
  for the workspace is there: source PDFs, page images, page text, schemas, ``.dgml.xml``.
- **Postgres**: every table the service and its DocStore own, filtered to the
  organisation's rows. Secret values are redacted, never returned.

Both are scoped to one organisation and cannot reach outside it: S3 keys are always
joined under the workspace prefix, and tables come from a fixed allow-list, never from
the request.
"""

from __future__ import annotations

import mimetypes
import uuid
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine

from .db import dgml_settings, iso, jobs, organisations, service_credentials
from .docstore import dgml_assignments, dgml_docsets, dgml_files, dgml_workspaces
from .tenancy import OrgSettings

REDACTED = "•••• redacted"

#: Column types are shown as Postgres spells them, whatever the engine.
_PG = postgresql.dialect()  # type: ignore[no-untyped-call]

#: name -> (table, the column that holds the organisation id, group)
TABLES: dict[str, tuple[sa.Table, str, str]] = {
    "dgml_workspaces": (dgml_workspaces, "organisation_id", "DGML DocStore"),
    "dgml_docsets": (dgml_docsets, "organisation_id", "DGML DocStore"),
    "dgml_files": (dgml_files, "organisation_id", "DGML DocStore"),
    "dgml_assignments": (dgml_assignments, "organisation_id", "DGML DocStore"),
    "organisations": (organisations, "id", "Service"),
    "dgml_settings": (dgml_settings, "organisation_id", "Service"),
    "service_credentials": (service_credentials, "organisation_id", "Service"),
    "jobs": (jobs, "organisation_id", "Service"),
}

#: Columns whose values are never sent to the browser.
SECRET_COLUMNS = {("service_credentials", "secret")}

#: DGML writes these with no extension that ``mimetypes`` knows.
_TEXT_SUFFIXES = {".rnc": "text/plain", ".md": "text/markdown", ".jsonl": "application/json"}


class TableNotFound(LookupError):
    pass


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------


def _relative(key: str) -> str:
    """A client-supplied key or prefix, normalized to be relative to the workspace."""
    parts = [p for p in key.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    return "/".join(parts)


def list_s3(settings: OrgSettings, prefix: str = "", *, recursive: bool = False) -> dict[str, Any]:
    """One "folder" of the workspace prefix: its sub-folders and objects. With
    ``recursive``, every object below the folder instead, and no sub-folders."""
    root = settings.workspace_prefix()
    rel = _relative(prefix)
    rel = f"{rel}/" if rel else ""
    client = settings.s3_client()
    folders: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    truncated = False
    paginator = client.get_paginator("list_objects_v2")
    delimiter = {} if recursive else {"Delimiter": "/"}
    for page in paginator.paginate(
        Bucket=settings.s3_bucket,
        Prefix=root + rel,
        **delimiter,
        PaginationConfig={"MaxItems": 1000},
    ):
        for cp in page.get("CommonPrefixes", []):
            sub = cp["Prefix"][len(root) :]
            folders.append({"name": sub[len(rel) :].rstrip("/"), "prefix": sub.rstrip("/")})
        for obj in page.get("Contents", []):
            key = obj["Key"][len(root) :]
            objects.append(
                {
                    "key": key,
                    "name": key[len(rel) :],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                }
            )
        truncated = truncated or bool(page.get("IsTruncated"))
    return {
        "bucket": settings.s3_bucket,
        "endpoint_url": settings.s3_endpoint_url,
        "root": root,
        "prefix": rel.rstrip("/"),
        "folders": folders,
        "objects": objects,
        "truncated": truncated,
    }


def content_type(key: str) -> str:
    for suffix, ctype in _TEXT_SUFFIXES.items():
        if key.endswith(suffix):
            return ctype
    if key.endswith(".dgml.xml"):
        return "application/xml"
    return mimetypes.guess_type(key)[0] or "application/octet-stream"


def get_s3_object(settings: OrgSettings, key: str) -> tuple[bytes, str]:
    """An object's bytes and content type. Raises ``FileNotFoundError``."""
    from botocore.exceptions import ClientError

    rel = _relative(key)
    if not rel:
        raise FileNotFoundError("no object key given")
    try:
        obj = settings.s3_client().get_object(
            Bucket=settings.s3_bucket, Key=settings.workspace_prefix() + rel
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            raise FileNotFoundError(rel) from exc
        raise
    return obj["Body"].read(), content_type(rel)


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------


def _json_value(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return iso(value)
    return value


def _columns(table: sa.Table) -> list[dict[str, Any]]:
    return [
        {
            "name": c.name,
            "type": str(c.type.compile(dialect=_PG)),
            "primary_key": c.primary_key,
            "nullable": c.nullable,
        }
        for c in table.columns
    ]


def _order(table: sa.Table) -> list[Any]:
    """Newest first where a table records creation, else primary-key order."""
    if "created_at" in table.c:
        return [table.c.created_at.desc()]
    return list(table.primary_key.columns)


def list_tables(engine: Engine, org_id: uuid.UUID) -> list[dict[str, Any]]:
    out = []
    with engine.connect() as conn:
        for name, (table, org_col, group) in TABLES.items():
            count = conn.execute(
                sa.select(sa.func.count()).select_from(table).where(table.c[org_col] == org_id)
            ).scalar_one()
            out.append(
                {"name": name, "group": group, "rows": int(count), "columns": _columns(table)}
            )
    return out


def table_rows(
    engine: Engine, org_id: uuid.UUID, name: str, *, limit: int = 50, offset: int = 0
) -> dict[str, Any]:
    if name not in TABLES:
        raise TableNotFound(f"no table {name!r} (known: {sorted(TABLES)})")
    table, org_col, group = TABLES[name]
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    where = table.c[org_col] == org_id
    with engine.connect() as conn:
        total = conn.execute(
            sa.select(sa.func.count()).select_from(table).where(where)
        ).scalar_one()
        rows = conn.execute(
            sa.select(table).where(where).order_by(*_order(table)).limit(limit).offset(offset)
        ).mappings()
        out_rows = [
            {
                c: (REDACTED if (name, c) in SECRET_COLUMNS else _json_value(v))
                for c, v in row.items()
            }
            for row in rows
        ]
    return {
        "name": name,
        "group": group,
        "columns": _columns(table),
        "rows": out_rows,
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }
