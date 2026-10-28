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

"""Organisation settings -> a DGML ``Configuration`` -> an open ``Workspace``.

There is no ``config.toml`` anywhere in this service. Each organisation's
``dgml_settings`` row plus its secrets is materialized into a
:class:`dgml_core.Configuration` in memory, and the workspace is opened with
``Workspace.open(configuration=…)``:

- **identity** — the workspace id is the organisation's UUID, the name its display
  name, and DGML's ``organization`` its slug;
- **storage** — docs in Postgres (:class:`~dgml_sample_service.docstore.PostgresDocStore`),
  blobs in the organisation's S3 bucket under ``<blob_folder>/<organisation_id>/``;
- **models** — DGML's ``[models]`` preset for the chosen family, with the API key
  passed by value (never through the process environment);
- **pdf** — PDFium (``pypdfium2``), so no Ghostscript binary is needed.

Opened workspaces are cached per organisation and reused until the organisation's
settings change; ``Workspace`` objects are cheap, but each one builds its own store
clients.
"""

from __future__ import annotations

import re
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from dgml_core import Configuration, Identity, ProviderSpec, Storage, Workspace
from dgml_core.configuration import Models, Ocr, Pdf
from sqlalchemy.engine import Engine

from .db import dgml_settings, organisations, service_credentials

#: The DGML ``[models].family`` presets this service offers.
LLM_FAMILIES = ("anthropic", "google", "openai", "anthropic_google")
TEXT_MODES = ("digital", "ocr", "hybrid")
OCR_PROVIDERS = ("macos",)

#: Secret kinds held in ``service_credentials``.
LLM_API_KEY = "llm_api_key"
LLM_API_KEY_GOOGLE = "llm_api_key_google"  # the second key the anthropic_google family needs
S3_ACCESS_KEY_ID = "s3_access_key_id"
S3_SECRET_ACCESS_KEY = "s3_secret_access_key"
SECRET_KINDS = (LLM_API_KEY, LLM_API_KEY_GOOGLE, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY)

DOCSTORE_PROVIDER = "dgml_sample_service.docstore:PostgresDocStore"
BLOBSTORE_PROVIDER = "dgml_storage_s3:S3BlobStore"

_SLUG_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?\Z")


class SettingsMissing(Exception):
    """The organisation has no DGML settings yet, so it has no workspace."""


def is_slug(value: str) -> bool:
    """Whether ``value`` can be DGML's ``organization`` (a namespace URI segment)."""
    return _SLUG_RE.match(value) is not None


def workspace_prefix(blob_folder: str, organisation_id: uuid.UUID) -> str:
    """Where a workspace's objects live in its bucket: ``<folder>/<org id>/``."""
    folder = blob_folder.strip("/")
    return f"{folder + '/' if folder else ''}{organisation_id}/"


@dataclass(frozen=True)
class OrgSettings:
    """Everything needed to build one organisation's ``Configuration``."""

    organisation_id: uuid.UUID
    org_name: str
    org_slug: str
    llm_family: str
    text_mode: str
    ocr_provider: str | None
    s3_endpoint_url: str | None
    s3_region: str | None
    s3_bucket: str
    blob_folder: str
    secrets: dict[str, str]

    def s3_options(self) -> dict[str, Any]:
        """The ``S3BlobStore`` options. ``prefix`` is the folder: the store appends the
        workspace id (the organisation id) itself."""
        opts: dict[str, Any] = {"bucket": self.s3_bucket, "prefix": self.blob_folder.strip("/")}
        if self.s3_endpoint_url:
            opts["endpoint_url"] = self.s3_endpoint_url
        if self.s3_region:
            opts["region"] = self.s3_region
        if self.secrets.get(S3_ACCESS_KEY_ID):
            opts["aws_access_key_id"] = self.secrets[S3_ACCESS_KEY_ID]
        if self.secrets.get(S3_SECRET_ACCESS_KEY):
            opts["aws_secret_access_key"] = self.secrets[S3_SECRET_ACCESS_KEY]
        return opts

    def workspace_prefix(self) -> str:
        return workspace_prefix(self.blob_folder, self.organisation_id)

    def s3_client(self) -> Any:
        """A boto3 S3 client for this organisation's connection (for the service's own
        probe and browser; DGML builds its own inside ``S3BlobStore``)."""
        import boto3

        opts = self.s3_options()
        kwargs: dict[str, Any] = {}
        if opts.get("endpoint_url"):
            kwargs["endpoint_url"] = opts["endpoint_url"]
        if opts.get("region"):
            kwargs["region_name"] = opts["region"]
        for name in ("aws_access_key_id", "aws_secret_access_key"):
            if opts.get(name):
                kwargs[name] = opts[name]
        return boto3.client("s3", **kwargs)

    def models(self) -> Models:
        key = self.secrets.get(LLM_API_KEY)
        if self.llm_family == "anthropic":
            return Models(family="anthropic", anthropic_api_key=key)
        if self.llm_family == "google":
            return Models(family="google", google_api_key=key)
        if self.llm_family == "openai":
            return Models(family="openai", openai_api_key=key)
        return Models(
            family="anthropic_google",
            anthropic_api_key=key,
            google_api_key=self.secrets.get(LLM_API_KEY_GOOGLE),
        )

    def configuration(self) -> Configuration:
        return Configuration.build(
            identity=Identity(
                workspace_id=str(self.organisation_id),
                name=self.org_name,
                organization=self.org_slug,
            ),
            storage=Storage(
                blobs=ProviderSpec(BLOBSTORE_PROVIDER, self.s3_options()),
                docs=ProviderSpec(
                    DOCSTORE_PROVIDER, {"organisation_id": str(self.organisation_id)}
                ),
            ),
            models=self.models(),
            pdf=Pdf(provider="pypdfium2"),
            ocr=Ocr(provider=self.ocr_provider) if self.ocr_provider else None,
        )


def load_settings(engine: Engine, organisation_id: uuid.UUID) -> OrgSettings:
    """The organisation's settings and secrets. Raises :class:`SettingsMissing`."""
    with engine.connect() as conn:
        row = (
            conn.execute(
                sa.select(dgml_settings, organisations.c.name, organisations.c.slug)
                .join(organisations, organisations.c.id == dgml_settings.c.organisation_id)
                .where(dgml_settings.c.organisation_id == organisation_id)
            )
            .mappings()
            .first()
        )
        if row is None:
            raise SettingsMissing(f"organisation {organisation_id} has no DGML settings")
        secrets = {
            r.kind: r.secret
            for r in conn.execute(
                sa.select(service_credentials.c.kind, service_credentials.c.secret).where(
                    service_credentials.c.organisation_id == organisation_id
                )
            )
        }
    return OrgSettings(
        organisation_id=organisation_id,
        org_name=row["name"],
        org_slug=row["slug"],
        llm_family=row["llm_family"],
        text_mode=row["text_mode"],
        ocr_provider=row["ocr_provider"],
        s3_endpoint_url=row["s3_endpoint_url"],
        s3_region=row["s3_region"],
        s3_bucket=row["s3_bucket"],
        blob_folder=row["blob_folder"],
        secrets=secrets,
    )


class WorkspaceRegistry:
    """Opens each organisation's workspace from its settings, and caches it.

    The cache is keyed on the organisation and checked against the freshly built
    ``Configuration`` on every call, so a settings change (a new API key, a new
    model family) takes effect on the next request without a restart.

    Also hands out a per-(organisation, file) lock: two jobs writing the same
    file's artifacts — a retry racing a running extraction, say — would overwrite
    each other's ``.dgml.xml``. The locks are a fixed set, picked by hashing the
    (organisation, file) pair, so they never need cleaning up; two files that share
    one only wait on each other. In-process locks suffice for a single-process
    sample; several service replicas would need a Postgres advisory lock instead.
    """

    FILE_LOCK_COUNT = 64

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._cache: dict[uuid.UUID, tuple[Configuration, Workspace]] = {}
        self._file_locks = [threading.Lock() for _ in range(self.FILE_LOCK_COUNT)]

    def settings(self, organisation_id: uuid.UUID) -> OrgSettings:
        return load_settings(self._engine, organisation_id)

    def open(self, organisation_id: uuid.UUID) -> Workspace:
        cfg = self.settings(organisation_id).configuration()
        with self._lock:
            cached = self._cache.get(organisation_id)
            if cached is not None and cached[0] == cfg:
                return cached[1]
        ws = Workspace.open(configuration=cfg)
        with self._lock:
            self._cache[organisation_id] = (cfg, ws)
        return ws

    def forget(self, organisation_id: uuid.UUID) -> None:
        """Drop the organisation's cached workspace (when the organisation is deleted)."""
        with self._lock:
            self._cache.pop(organisation_id, None)

    @contextmanager
    def file_lock(self, organisation_id: uuid.UUID, file_id: str) -> Iterator[None]:
        """Serialize writes to one file's artifacts. Not re-entrant: never nest two."""
        with self._file_locks[hash((organisation_id, file_id)) % self.FILE_LOCK_COUNT]:
            yield
