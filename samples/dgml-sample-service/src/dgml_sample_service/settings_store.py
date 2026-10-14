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

"""Organisations and their DGML settings: reads, validated writes, and the storage probe.

Settings rules (mirroring the design for a hosted DGML integration):

- the LLM family, API keys, text mode and OCR provider can change at any time; the
  next DGML call picks them up;
- the storage connection (bucket, endpoint, region, folder) **locks once the
  workspace holds a file** — changing it would orphan every blob, since an
  in-memory ``Configuration`` has no storage seal to catch a moved store;
- secrets are write-only: reads say whether one is set and when, never its value;
- a storage change is probed (write + delete an object under the workspace prefix)
  before it is saved.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from .config import ServiceConfig
from .db import dgml_settings, iso, jobs, organisations, service_credentials, utcnow
from .docstore import dgml_assignments, dgml_docsets, dgml_files, dgml_workspaces
from .tenancy import (
    LLM_FAMILIES,
    OCR_PROVIDERS,
    S3_ACCESS_KEY_ID,
    S3_SECRET_ACCESS_KEY,
    SECRET_KINDS,
    TEXT_MODES,
    OrgSettings,
    is_slug,
    workspace_prefix,
)

STORAGE_FIELDS = ("s3_endpoint_url", "s3_region", "s3_bucket", "blob_folder")
SETTINGS_FIELDS = ("llm_family", "text_mode", "ocr_provider", *STORAGE_FIELDS)


class OrgNotFound(Exception):
    pass


class SettingsInvalid(ValueError):
    pass


class SettingsLocked(Exception):
    pass


class StorageProbeFailed(Exception):
    pass


def org_json(row: Any) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "name": row["name"],
        "slug": row["slug"],
        "created_at": iso(row["created_at"]),
    }


def list_orgs(engine: Engine) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(sa.select(organisations).order_by(organisations.c.name)).mappings()
        return [org_json(r) for r in rows]


def get_org(engine: Engine, org_id: uuid.UUID) -> dict[str, Any]:
    with engine.connect() as conn:
        row = (
            conn.execute(sa.select(organisations).where(organisations.c.id == org_id))
            .mappings()
            .first()
        )
    if row is None:
        raise OrgNotFound(f"organisation {org_id} not found")
    return org_json(row)


def create_org(engine: Engine, config: ServiceConfig, *, name: str, slug: str) -> dict[str, Any]:
    """Create an organisation with default settings pointing at the service's default
    S3 (the local SeaweedFS in development) and no LLM key yet."""
    name, slug = name.strip(), slug.strip()
    if not name:
        raise SettingsInvalid("name must not be empty")
    if not is_slug(slug):
        raise SettingsInvalid(
            "slug must be 1-64 lowercase letters, digits or hyphens, starting and ending "
            "with a letter or digit (it becomes the DGML namespace segment)"
        )
    org_id = uuid.uuid4()
    try:
        with engine.begin() as conn:
            conn.execute(sa.insert(organisations).values(id=org_id, name=name, slug=slug))
            conn.execute(
                sa.insert(dgml_settings).values(
                    organisation_id=org_id,
                    llm_family="anthropic",
                    text_mode="digital",
                    ocr_provider=None,
                    s3_endpoint_url=config.default_s3_endpoint,
                    s3_region=None,
                    s3_bucket=config.default_s3_bucket,
                    blob_folder="",
                )
            )
            for kind, value in (
                (S3_ACCESS_KEY_ID, config.default_s3_access_key_id),
                (S3_SECRET_ACCESS_KEY, config.default_s3_secret_access_key),
            ):
                if value:
                    _put_secret(conn, org_id, kind, value)
    except sa.exc.IntegrityError as exc:
        # The id is fresh, so the only constraint a new organisation can break is the
        # unique slug. Relying on it (not a check-then-insert) is race-free.
        raise SettingsInvalid(f"slug {slug!r} is already taken") from exc
    return get_org(engine, org_id)


#: Every table with an ``organisation_id`` column: what deleting an organisation clears.
ORG_TABLES = (
    dgml_workspaces,
    dgml_docsets,
    dgml_files,
    dgml_assignments,
    service_credentials,
    dgml_settings,
    jobs,
)


def delete_org(engine: Engine, org_id: uuid.UUID) -> None:
    """Delete every row the organisation owns, then the organisation itself.

    The caller removes the DGML data through DGML first (that is what deletes the
    blobs); clearing the ``dgml_*`` tables here as well means an organisation whose
    workspace can no longer be opened still leaves no rows behind."""
    with engine.begin() as conn:
        for table in ORG_TABLES:
            conn.execute(sa.delete(table).where(table.c.organisation_id == org_id))
        conn.execute(sa.delete(organisations).where(organisations.c.id == org_id))


def file_count(engine: Engine, org_id: uuid.UUID) -> int:
    """How many files the organisation's workspace holds — read straight off the typed
    ``dgml_files`` table."""
    with engine.connect() as conn:
        return int(
            conn.execute(
                sa.select(sa.func.count())
                .select_from(dgml_files)
                .where(dgml_files.c.organisation_id == org_id)
            ).scalar_one()
        )


def settings_json(engine: Engine, org_id: uuid.UUID) -> dict[str, Any]:
    """The organisation's settings as the API returns them: secrets only as set/unset."""
    with engine.connect() as conn:
        row = (
            conn.execute(sa.select(dgml_settings).where(dgml_settings.c.organisation_id == org_id))
            .mappings()
            .first()
        )
        secrets = {
            r.kind: r.updated_at
            for r in conn.execute(
                sa.select(service_credentials.c.kind, service_credentials.c.updated_at).where(
                    service_credentials.c.organisation_id == org_id
                )
            )
        }
    if row is None:
        return {"configured": False}
    out: dict[str, Any] = {f: row[f] for f in SETTINGS_FIELDS}
    out["configured"] = True
    out["updated_at"] = iso(row["updated_at"])
    out["secrets"] = {
        kind: {"set": kind in secrets, "updated_at": iso(secrets.get(kind))}
        for kind in SECRET_KINDS
    }
    out["storage_locked"] = file_count(engine, org_id) > 0
    out["storage_path"] = f"s3://{row['s3_bucket']}/{workspace_prefix(row['blob_folder'], org_id)}"
    out["options"] = {
        "llm_families": list(LLM_FAMILIES),
        "text_modes": list(TEXT_MODES),
        "ocr_providers": list(OCR_PROVIDERS),
    }
    return out


@dataclass(frozen=True)
class SettingsUpdate:
    """A partial update. In ``values`` an absent field is left unchanged and ``None``
    clears it (e.g. ``ocr_provider``). In ``secrets`` a value sets the secret, ``""``
    clears it, and an absent kind is left alone."""

    values: dict[str, Any]
    secrets: dict[str, str]


def validate_update(update: SettingsUpdate) -> None:
    v = update.values
    unknown = set(v) - set(SETTINGS_FIELDS)
    if unknown:
        raise SettingsInvalid(f"unknown setting(s): {sorted(unknown)}")
    if "llm_family" in v and v["llm_family"] not in LLM_FAMILIES:
        raise SettingsInvalid(f"llm_family must be one of {list(LLM_FAMILIES)}")
    if "text_mode" in v and v["text_mode"] not in TEXT_MODES:
        raise SettingsInvalid(f"text_mode must be one of {list(TEXT_MODES)}")
    if v.get("ocr_provider") not in (None, *OCR_PROVIDERS):
        raise SettingsInvalid(f"ocr_provider must be empty or one of {list(OCR_PROVIDERS)}")
    if "s3_bucket" in v and not str(v["s3_bucket"] or "").strip():
        raise SettingsInvalid("s3_bucket must not be empty")
    bad = set(update.secrets) - set(SECRET_KINDS)
    if bad:
        raise SettingsInvalid(f"unknown secret kind(s): {sorted(bad)}")


def merged_settings(current: OrgSettings, update: SettingsUpdate) -> OrgSettings:
    """``current`` with ``update`` applied — what the settings *would* be."""
    fields = {f: getattr(current, f) for f in SETTINGS_FIELDS}
    for key, value in update.values.items():
        fields[key] = (value.strip() or None) if isinstance(value, str) else value
    fields["blob_folder"] = (fields.get("blob_folder") or "").strip("/")
    secrets = dict(current.secrets)
    for kind, value in update.secrets.items():
        if value:
            secrets[kind] = value
        else:
            secrets.pop(kind, None)
    return OrgSettings(
        organisation_id=current.organisation_id,
        org_name=current.org_name,
        org_slug=current.org_slug,
        secrets=secrets,
        **fields,
    )


def storage_changed(old: OrgSettings, new: OrgSettings) -> bool:
    keys = (S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY)
    return any(getattr(old, f) != getattr(new, f) for f in STORAGE_FIELDS) or any(
        old.secrets.get(k) != new.secrets.get(k) for k in keys
    )


def storage_location_changed(old: OrgSettings, new: OrgSettings) -> bool:
    """A change to *where* blobs live (credentials may rotate freely)."""
    return any(getattr(old, f) != getattr(new, f) for f in STORAGE_FIELDS)


def probe_storage(settings: OrgSettings, *, create_bucket: bool = False) -> dict[str, Any]:
    """Write and delete a probe object under the workspace's prefix. Optionally create
    the bucket first when it does not exist (handy against a fresh SeaweedFS)."""
    from botocore.exceptions import BotoCoreError, ClientError

    client = settings.s3_client()
    bucket = settings.s3_bucket
    key = f"{settings.workspace_prefix()}.probe-{uuid.uuid4().hex}"
    created = False
    try:
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError as exc:
            missing = str(exc.response.get("Error", {}).get("Code")) in {"404", "NoSuchBucket"}
            if not (missing and create_bucket):
                raise
            client.create_bucket(Bucket=bucket)
            created = True
        client.put_object(Bucket=bucket, Key=key, Body=b"dgml-sample-service probe")
        client.delete_object(Bucket=bucket, Key=key)
    except (ClientError, BotoCoreError) as exc:
        raise StorageProbeFailed(f"cannot write to s3://{bucket}/{key}: {exc}") from exc
    return {"ok": True, "bucket": bucket, "bucket_created": created, "probe_key": key}


def _put_secret(conn: Connection, org_id: uuid.UUID, kind: str, value: str) -> None:
    conn.execute(
        sa.delete(service_credentials).where(
            service_credentials.c.organisation_id == org_id, service_credentials.c.kind == kind
        )
    )
    conn.execute(
        sa.insert(service_credentials).values(
            organisation_id=org_id, kind=kind, secret=value, updated_at=utcnow()
        )
    )


def save_settings(engine: Engine, new: OrgSettings, update: SettingsUpdate) -> None:
    with engine.begin() as conn:
        conn.execute(
            sa.update(dgml_settings)
            .where(dgml_settings.c.organisation_id == new.organisation_id)
            .values(**{f: getattr(new, f) for f in SETTINGS_FIELDS}, updated_at=utcnow())
        )
        for kind, value in update.secrets.items():
            if value:
                _put_secret(conn, new.organisation_id, kind, value)
            else:
                conn.execute(
                    sa.delete(service_credentials).where(
                        service_credentials.c.organisation_id == new.organisation_id,
                        service_credentials.c.kind == kind,
                    )
                )
