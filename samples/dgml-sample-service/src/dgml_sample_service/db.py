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

"""The service's database: one SQLAlchemy ``MetaData`` for every table.

Two groups of tables share it:

- the service's own — ``organisations``, ``dgml_settings`` (each organisation's
  DGML settings, the source its workspace ``Configuration`` is built from),
  ``service_credentials`` (a stand-in for a secrets vault) and ``jobs``;
- DGML's state tables (``dgml_workspaces``, ``dgml_docsets``, ``dgml_files``,
  ``dgml_assignments``), declared next to the store that owns them in
  :mod:`dgml_sample_service.docstore`.

A sample runs ``create_all`` at startup; a real service would own these tables
through its migrations (Alembic) instead.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Engine

metadata = sa.MetaData()


def now_column(name: str) -> sa.Column[Any]:
    """A ``timestamptz NOT NULL DEFAULT now()`` column."""
    return sa.Column(name, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(value: datetime | None) -> str | None:
    """A timestamp column as ISO-8601 for the API. SQLite (the tests) returns naive
    datetimes; everything is stored as UTC, so a missing zone is UTC."""
    if value is None:
        return None
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).isoformat()


organisations = sa.Table(
    "organisations",
    metadata,
    sa.Column("id", sa.Uuid(), primary_key=True),
    sa.Column("name", sa.Text(), nullable=False),
    # DGML's `organization`: the namespace URI segment in http://dgml.io/<slug>/...
    # Fixed at creation — changing it would split the corpus across two namespaces.
    sa.Column("slug", sa.Text(), nullable=False, unique=True),
    now_column("created_at"),
)

dgml_settings = sa.Table(
    "dgml_settings",
    metadata,
    sa.Column("organisation_id", sa.Uuid(), primary_key=True),
    # 'anthropic' | 'google' | 'openai' | 'anthropic_google' — DGML's [models].family.
    sa.Column("llm_family", sa.Text(), nullable=False),
    # How FileStore.add reads text: 'digital' | 'ocr' | 'hybrid'.
    sa.Column("text_mode", sa.Text(), nullable=False, server_default="digital"),
    # The OCR provider for 'ocr' / 'hybrid' text ('macos' only, in this sample).
    sa.Column("ocr_provider", sa.Text()),
    # The organisation's S3 connection. Credentials live in service_credentials.
    sa.Column("s3_endpoint_url", sa.Text()),
    sa.Column("s3_region", sa.Text()),
    sa.Column("s3_bucket", sa.Text(), nullable=False),
    # A path in the bucket; '' is the bucket root. Objects land under
    # <blob_folder>/<organisation_id>/.
    sa.Column("blob_folder", sa.Text(), nullable=False, server_default=""),
    now_column("created_at"),
    now_column("updated_at"),
)

#: Secrets, one row per (organisation, kind). A **sample stand-in for a vault**: a
#: real deployment keeps these in Vault / a KMS and stores only a reference here.
#: The API never returns a secret, only whether one is set and when.
service_credentials = sa.Table(
    "service_credentials",
    metadata,
    sa.Column("organisation_id", sa.Uuid(), primary_key=True),
    # 'llm_api_key' | 'llm_api_key_google' | 's3_access_key_id' | 's3_secret_access_key'
    sa.Column("kind", sa.Text(), primary_key=True),
    sa.Column("secret", sa.Text(), nullable=False),
    now_column("updated_at"),
)

jobs = sa.Table(
    "jobs",
    metadata,
    sa.Column("id", sa.Uuid(), primary_key=True),
    sa.Column("organisation_id", sa.Uuid(), nullable=False, index=True),
    # 'add_file' | 'classify' | 'extract' | 'generate_schema'
    sa.Column("kind", sa.Text(), nullable=False),
    # 'queued' | 'running' | 'succeeded' | 'failed'
    sa.Column("status", sa.Text(), nullable=False),
    sa.Column("file_id", sa.Text()),
    sa.Column("docset_id", sa.Text()),
    sa.Column("label", sa.Text()),
    sa.Column("params", sa.JSON(), nullable=False),
    sa.Column("result", sa.JSON()),
    sa.Column("error", sa.Text()),
    sa.Column("error_code", sa.Text()),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("started_at", sa.DateTime(timezone=True)),
    sa.Column("finished_at", sa.DateTime(timezone=True)),
)


def make_engine(url: str) -> Engine:
    """An engine for ``url``. SQLite (the tests) waits on its file lock rather than
    failing when a job thread and a request write at once."""
    kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"timeout": 30, "check_same_thread": False}
    return sa.create_engine(url, **kwargs)


def create_schema(engine: Engine) -> None:
    """Create every table that does not exist yet."""
    # Imported for its side effect: declares the dgml_* tables on ``metadata``.
    from . import docstore  # noqa: F401

    metadata.create_all(engine)
