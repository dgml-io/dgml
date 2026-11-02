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

import logging
from typing import TYPE_CHECKING, Any

from . import configuration, layout
from .configuration import Configuration, Identity, ProviderSpec, Storage
from .conversion import (
    ConverterConfig,
    DocConverter,
    FormatFamily,
    load_conversion_config,
    make_converter,
)
from .dgml_output import DgmlDocument, read_dgml
from .docsets import DocSetStore

# Exported in full, and pinned so by tests/test_error_codes.py.
from .errors import (
    AttestationInvalid,
    AuthError,
    BatchExecutionFailed,
    BatchJobBusy,
    BatchJobInvalid,
    BatchJobLeaseLost,
    BatchJobNondeterministic,
    BatchJobNotFound,
    BatchUnavailable,
    ChainConfigError,
    ChainRpcFailed,
    ChainTxReverted,
    ClassificationConfigInvalid,
    ClassificationConfigMissing,
    ClassificationFailed,
    ClusteringConfigInvalid,
    ConflictError,
    ConflictErrorKind,
    ConversionConfigInvalid,
    ConversionFailed,
    CorruptMetadata,
    DgmlError,
    DgmlNotFound,
    DocSetNotFound,
    EmptyModelResponse,
    EngineNotAvailable,
    FileNotFound,
    GenerationConfigInvalid,
    GenerationConfigMissing,
    GenerationFailed,
    GhostscriptNotFound,
    GroundedConfigInvalid,
    GroundedConfigMissing,
    GroundingFailed,
    GuidanceNotFound,
    IncrementalWithoutClusters,
    InvalidArgument,
    InvalidPDF,
    LabelModelUnreachable,
    LegacyConfigPresent,
    LinkPlanFailed,
    MissingExtra,
    ModelNotSupported,
    ModelsConfigInvalid,
    NoExistingDocSets,
    NotFoundError,
    OcrConfigInvalid,
    OcrConfigMissing,
    OcrFailed,
    PageRenderFailed,
    PdfConfigInvalid,
    PdfSliceFailed,
    RecordNotFound,
    RegistryNotFound,
    SchemaGenerationFailed,
    SchemaInvalid,
    SchemaNotFound,
    StorageBackendMismatch,
    StorageConfigInvalid,
    StorageProviderUnresolvable,
    StyleConfigInvalid,
    TextExtractionConfigInvalid,
    TextExtractionFailed,
    UnsupportedFileType,
    ValuesExtractionFailed,
    ValuesNotFound,
    WalletKeyMissing,
    WorkspaceMigrationFailed,
    WorkspaceNotFound,
    WorkspaceNotInitialized,
    WorkspacesConfigInvalid,
    WorkspacesUnavailable,
    WorkspacesWriteConflict,
)

# Eager: ``.extraction`` imports only dgml_output/docsets/errors/storage at module
# scope (all already loaded above) and reaches ``.grounded`` → ``.llm`` and
# ``.extraction_xml`` inside the functions, so it costs nothing here.
from .extraction import add_file_and_extract, extract_file, get_extracted_values
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
from .files import AddFileResult, ConflictKind, ConflictPolicy, FileStore
from .ids import RECORD_ID_SHAPE, is_record_id
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
from .models import DocSet, DocSetAssignment, FileRecord
from .ocr import (
    BUILTIN_OCR_PROVIDERS,
    OcrConfig,
    OcrProvider,
    OcrProviderName,
    load_ocr_config,
    make_ocr_provider,
)
from .pages import EngineName, PdfConfig, PdfSlicer, load_pdf_config, slice_pages
from .storage import Workspace
from .storage_local import LocalStore
from .storage_resolve import (
    DEFAULT_STORAGE_PROVIDER,
    DEFAULT_STORAGE_SERVICE,
    load_store_configs,
    make_blob_store,
    make_doc_store,
    resolve_store_configs,
    storage_fingerprint,
)
from .storage_service import (
    BlobStore,
    DocStore,
    StorageConfig,
)
from .text_extraction import TextMode
from .utils import page_image_key, read_page_image
from .workspace_config import WorkspaceIdentity
from .workspace_create import CreateWorkspaceResult, create_workspace
from .workspace_id import (
    ID_SHAPE,
    generate_unique_workspace_id,
    is_workspace_id,
    new_workspace_id,
)
from .workspace_ops import WorkspaceOps
from .workspaces_local import LocalDirWorkspacesStore
from .workspaces_resolve import (
    DEFAULT_WORKSPACES_PROVIDER,
    default_workspaces_store,
    load_workspaces_config,
    make_workspaces_store,
)
from .workspaces_store import WorkspacesConfig, WorkspacesStore, default_workspaces_root

if TYPE_CHECKING:
    from .auto_classification import (
        BulkClassifyBatch,
        auto_classify,
        classify_bulk_batch,
        prepare_bulk_classify,
    )
    from .batch.jobs import (
        cancel_job,
        delete_job,
        job_status,
        list_job_summaries,
        prune_jobs,
        resume_would_wait,
        unlock_job,
    )
    from .classification import (
        ClassificationConfig,
        ClassificationDecision,
        ClassifyMode,
        DecisionKind,
        classify_file,
        load_classification_config,
    )
    from .consistency import CheckReport, Issue, IssueKind, IssueTarget, check_workspace
    from .generation.link_stage import LinkOutcome, LinkStage, StagedDocument
    from .grounded import ExtractionResult, ManyExtraction, extract_values_many

__version__ = "0.1.0"

# Logging: every module logs through ``logging.getLogger(__name__)`` (so under
# ``dgml_core.*``) and configures nothing — the caller decides where records go.
# The NullHandler keeps a caller who configured nothing silent, rather than
# falling through to Python's last-resort stderr handler.
logging.getLogger(__name__).addHandler(logging.NullHandler())

#: Names re-exported from ``.consistency``, ``.classification`` and ``.grounded``,
#: resolved on FIRST ACCESS rather than at import (PEP 562). All three reach ``.llm``
#: → ``litellm`` (consistency via ``.hybrid``; the other two directly), which costs
#: ~1.4s of the package's ~1.66s import — paid by every consumer, including the
#: ones that only ever touch ``Workspace``/``FileStore`` and never make an LLM
#: call. Deterministic CLIs that invoke this package thousands of times per
#: session (bill extraction) spend that entire budget on an unused client.
#: Importing a submodule directly, or touching any name below, still loads it
#: exactly as before. ``tests/test_public_api.py`` pins the invariant.
#: The orchestration API below (auto-classification, multi-file extraction,
#: the semantic-link stage, batch job management) is lazy for the same reason,
#: and so that ``dgml_core.batch`` is never on the default import path.
_LAZY_SUBMODULES = {
    "CheckReport": ".consistency",
    "ClassificationConfig": ".classification",
    "ClassificationDecision": ".classification",
    "ClassifyMode": ".classification",
    "DecisionKind": ".classification",
    "ExtractionResult": ".grounded",
    "Issue": ".consistency",
    "IssueKind": ".consistency",
    "IssueTarget": ".consistency",
    "BulkClassifyBatch": ".auto_classification",
    "auto_classify": ".auto_classification",
    "classify_bulk_batch": ".auto_classification",
    "prepare_bulk_classify": ".auto_classification",
    "ManyExtraction": ".grounded",
    "extract_values_many": ".grounded",
    "LinkOutcome": ".generation.link_stage",
    "LinkStage": ".generation.link_stage",
    "StagedDocument": ".generation.link_stage",
    "cancel_job": ".batch.jobs",
    "delete_job": ".batch.jobs",
    "job_status": ".batch.jobs",
    "list_job_summaries": ".batch.jobs",
    "prune_jobs": ".batch.jobs",
    "resume_would_wait": ".batch.jobs",
    "unlock_job": ".batch.jobs",
    "check_workspace": ".consistency",
    "classify_file": ".classification",
    "load_classification_config": ".classification",
}


def __getattr__(name: str) -> Any:
    """Resolve the deferred re-exports on first access (PEP 562)."""
    module = _LAZY_SUBMODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "BUILTIN_OCR_PROVIDERS",
    "DEFAULT_STORAGE_PROVIDER",
    "DEFAULT_STORAGE_SERVICE",
    "DEFAULT_WORKSPACES_PROVIDER",
    "ID_SHAPE",
    "RECORD_ID_SHAPE",
    "WORKSPACE_SCHEMA_VERSION",
    "AddFileResult",
    "ArtifactKind",
    "ArtifactRef",
    "AttestationEntry",
    "AttestationInvalid",
    "AttestationInventory",
    "AuthError",
    "BatchExecutionFailed",
    "BatchJobBusy",
    "BatchJobInvalid",
    "BatchJobLeaseLost",
    "BatchJobNondeterministic",
    "BatchJobNotFound",
    "BatchUnavailable",
    "BlobStore",
    "BulkClassifyBatch",
    "ChainConfigError",
    "ChainRpcFailed",
    "ChainTxReverted",
    "CheckReport",
    "ClassificationConfig",
    "ClassificationConfigInvalid",
    "ClassificationConfigMissing",
    "ClassificationDecision",
    "ClassificationFailed",
    "ClassifyMode",
    "ClusteringConfigInvalid",
    "Collection",
    "Configuration",
    "ConflictError",
    "ConflictErrorKind",
    "ConflictKind",
    "ConflictPolicy",
    "ConversionConfigInvalid",
    "ConversionFailed",
    "ConverterConfig",
    "CorruptMetadata",
    "CreateWorkspaceResult",
    "DecisionKind",
    "DgmlDocument",
    "DgmlError",
    "DgmlNotFound",
    "DocConverter",
    "DocSet",
    "DocSetAssignment",
    "DocSetNotFound",
    "DocSetStore",
    "DocStore",
    "EmptyModelResponse",
    "EngineName",
    "EngineNotAvailable",
    "ExtractionResult",
    "FileAttestation",
    "FileNotFound",
    "FileRecord",
    "FileStore",
    "FileVersion",
    "FormatFamily",
    "GenerationConfigInvalid",
    "GenerationConfigMissing",
    "GenerationFailed",
    "GhostscriptNotFound",
    "GroundedConfigInvalid",
    "GroundedConfigMissing",
    "GroundingFailed",
    "GuidanceNotFound",
    "Identity",
    "IncrementalWithoutClusters",
    "InvalidArgument",
    "InvalidPDF",
    "Issue",
    "IssueKind",
    "IssueTarget",
    "LabelModelUnreachable",
    "LegacyConfigPresent",
    "LinkOutcome",
    "LinkPlanFailed",
    "LinkStage",
    "LocalDirWorkspacesStore",
    "LocalStore",
    "ManyExtraction",
    "Migration",
    "MigrationResult",
    "MissingExtra",
    "ModelNotSupported",
    "ModelsConfigInvalid",
    "NoExistingDocSets",
    "NotFoundError",
    "OcrConfig",
    "OcrConfigInvalid",
    "OcrConfigMissing",
    "OcrFailed",
    "OcrProvider",
    "OcrProviderName",
    "PageRenderFailed",
    "PdfConfig",
    "PdfConfigInvalid",
    "PdfSliceFailed",
    "PdfSlicer",
    "ProviderSpec",
    "RecordNotFound",
    "RegistryNotFound",
    "SchemaGenerationFailed",
    "SchemaInvalid",
    "SchemaNotFound",
    "StagedDocument",
    "Storage",
    "StorageBackendMismatch",
    "StorageConfig",
    "StorageConfigInvalid",
    "StorageProviderUnresolvable",
    "StyleConfigInvalid",
    "TextExtractionConfigInvalid",
    "TextExtractionFailed",
    "TextMode",
    "UnsupportedFileType",
    "ValuesExtractionFailed",
    "ValuesNotFound",
    "VerifyResult",
    "WalletKeyMissing",
    "Workspace",
    "WorkspaceIdentity",
    "WorkspaceMigrationFailed",
    "WorkspaceNotFound",
    "WorkspaceNotInitialized",
    "WorkspaceOps",
    "WorkspacesConfig",
    "WorkspacesConfigInvalid",
    "WorkspacesStore",
    "WorkspacesUnavailable",
    "WorkspacesWriteConflict",
    "__version__",
    "add_file_and_extract",
    "attest_file",
    "attest_file_version",
    "auto_classify",
    "cancel_job",
    "check_workspace",
    "classify_bulk_batch",
    "classify_file",
    "collect_file_version",
    "collect_from_attestation",
    "configuration",
    "create_workspace",
    "default_workspaces_root",
    "default_workspaces_store",
    "delete_job",
    "export_attestation",
    "extract_file",
    "extract_values_many",
    "generate_unique_workspace_id",
    "get_extracted_values",
    "is_record_id",
    "is_workspace_id",
    "job_status",
    "layout",
    "list_job_summaries",
    "load_classification_config",
    "load_conversion_config",
    "load_ocr_config",
    "load_pdf_config",
    "load_store_configs",
    "load_workspaces_config",
    "make_blob_store",
    "make_converter",
    "make_doc_store",
    "make_ocr_provider",
    "make_workspaces_store",
    "migrate_workspace",
    "new_workspace_id",
    "page_image_key",
    "pending_migrations",
    "prepare_bulk_classify",
    "prune_jobs",
    "read_attestation",
    "read_dgml",
    "read_page_image",
    "resolve_store_configs",
    "resume_would_wait",
    "slice_pages",
    "stamp_schema_version",
    "storage_fingerprint",
    "unlock_job",
    "verify_attestation_dir",
    "verify_bundle",
    "verify_file_version",
    "workspace_schema_version",
    "write_attestation",
]
