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

"""The service's own process configuration, from ``DGML_SAMPLE_*`` environment variables.

This is about where the *service* runs (its database, CORS, the defaults offered to
a new organisation) — not DGML configuration. Each organisation's DGML settings live
in Postgres and become a ``Configuration`` per request (:mod:`.tenancy`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceConfig:
    database_url: str = "postgresql+pg8000://dgml:dgml@localhost:55432/dgml_sample"
    cors_origins: tuple[str, ...] = ("http://localhost:5180", "http://127.0.0.1:5180")
    #: Defaults for a new organisation's storage settings: the local SeaweedFS.
    default_s3_endpoint: str | None = "http://localhost:18333"
    default_s3_bucket: str = "dgml-sample"
    #: SeaweedFS (no identity config) accepts any signature, but boto3 refuses to sign
    #: without *some* credentials.
    default_s3_access_key_id: str | None = "dgml"
    default_s3_secret_access_key: str | None = "dgml-secret"
    job_workers: int = 4
    #: A built frontend (``frontend/dist``) to serve at ``/``; ``None`` serves the API only.
    frontend_dist: Path | None = None

    @classmethod
    def from_env(cls) -> ServiceConfig:
        env = os.environ
        default = cls()
        dist = env.get("DGML_SAMPLE_FRONTEND_DIST")
        origins = env.get("DGML_SAMPLE_CORS_ORIGINS")
        return cls(
            database_url=env.get("DGML_SAMPLE_DATABASE_URL", default.database_url),
            cors_origins=(
                tuple(o.strip() for o in origins.split(",") if o.strip())
                if origins is not None
                else default.cors_origins
            ),
            default_s3_endpoint=env.get("DGML_SAMPLE_S3_ENDPOINT", default.default_s3_endpoint)
            or None,
            default_s3_bucket=env.get("DGML_SAMPLE_S3_BUCKET", default.default_s3_bucket),
            default_s3_access_key_id=env.get(
                "DGML_SAMPLE_S3_ACCESS_KEY_ID", default.default_s3_access_key_id
            )
            or None,
            default_s3_secret_access_key=env.get(
                "DGML_SAMPLE_S3_SECRET_ACCESS_KEY", default.default_s3_secret_access_key
            )
            or None,
            job_workers=int(env.get("DGML_SAMPLE_JOB_WORKERS", default.job_workers)),
            frontend_dist=Path(dist) if dist else None,
        )
