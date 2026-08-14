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

"""DGML: semantic XML representation of documents."""

from __future__ import annotations

from . import layout
from .consistency import CheckReport, Issue, check_workspace
from .conversion import (
    ConverterConfig,
    DocConverter,
    load_conversion_config,
    make_converter,
)
from .docsets import DocSetStore
from .errors import (
    ConflictError,
    ConversionConfigInvalid,
    ConversionFailed,
    CorruptMetadata,
    DgmlError,
    DocSetNotFound,
    FileNotFound,
    GhostscriptNotFound,
    InvalidArgument,
    InvalidPDF,
    PageRenderFailed,
    UnsupportedFileType,
    WorkspaceMigrationFailed,
    WorkspaceNotInitialized,
)
from .file_attestation import (
    ArtifactKind,
    ArtifactRef,
    AttestationEntry,
    AttestationInventory,
    FileAttestation,
    FileVersion,
    VerifyResult,
    attest_file,
    attest_file_version,
    collect_file_version,
    collect_from_attestation,
    export_attestation,
    read_attestation,
    verify_attestation_dir,
    verify_bundle,
    verify_file_version,
    write_attestation,
)
from .files import AddFileResult, ConflictPolicy, FileStore
from .layout import Collection
from .migrations import (
    WORKSPACE_SCHEMA_VERSION,
    Migration,
    MigrationResult,
    migrate_workspace,
    pending_migrations,
    stamp_schema_version,
    workspace_schema_version,
)
from .models import DocSet, FileRecord
from .storage import Workspace
from .storage_local import LocalStore
from .storage_resolve import (
    DEFAULT_STORAGE_PROVIDER,
    DEFAULT_STORAGE_SERVICE,
    load_store_configs,
    make_blob_store,
    make_doc_store,
    storage_fingerprint,
)
from .storage_service import (
    BlobStore,
    DocStore,
    StorageConfig,
)
from .workspace_ops import WorkspaceOps

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_STORAGE_PROVIDER",
    "DEFAULT_STORAGE_SERVICE",
    "WORKSPACE_SCHEMA_VERSION",
    "AddFileResult",
    "ArtifactKind",
    "ArtifactRef",
    "AttestationEntry",
    "AttestationInventory",
    "BlobStore",
    "CheckReport",
    "Collection",
    "ConflictError",
    "ConflictPolicy",
    "ConversionConfigInvalid",
    "ConversionFailed",
    "ConverterConfig",
    "CorruptMetadata",
    "DgmlError",
    "DocConverter",
    "DocSet",
    "DocSetNotFound",
    "DocSetStore",
    "DocStore",
    "FileAttestation",
    "FileNotFound",
    "FileRecord",
    "FileStore",
    "FileVersion",
    "GhostscriptNotFound",
    "InvalidArgument",
    "InvalidPDF",
    "Issue",
    "LocalStore",
    "Migration",
    "MigrationResult",
    "PageRenderFailed",
    "StorageConfig",
    "UnsupportedFileType",
    "VerifyResult",
    "Workspace",
    "WorkspaceMigrationFailed",
    "WorkspaceNotInitialized",
    "WorkspaceOps",
    "__version__",
    "attest_file",
    "attest_file_version",
    "check_workspace",
    "collect_file_version",
    "collect_from_attestation",
    "export_attestation",
    "layout",
    "load_conversion_config",
    "load_store_configs",
    "make_blob_store",
    "make_converter",
    "make_doc_store",
    "migrate_workspace",
    "pending_migrations",
    "read_attestation",
    "stamp_schema_version",
    "storage_fingerprint",
    "verify_attestation_dir",
    "verify_bundle",
    "verify_file_version",
    "workspace_schema_version",
    "write_attestation",
]
