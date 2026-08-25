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

"""Custom exceptions and persistent error records."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from . import layout

if TYPE_CHECKING:
    from .storage import Workspace


class DgmlError(Exception):
    """Base class for all DGML errors. Carries a stable ``code`` for the CLI."""

    code: str = "DGML_ERROR"


class WorkspaceNotInitialized(DgmlError):
    code = "WORKSPACE_NOT_INITIALIZED"


class NotFoundError(DgmlError):
    code = "NOT_FOUND"


class DocSetNotFound(NotFoundError):
    code = "DOCSET_NOT_FOUND"


class FileNotFound(NotFoundError):
    code = "FILE_NOT_FOUND"


class ConflictError(DgmlError):
    code = "CONFLICT"

    def __init__(self, message: str, *, kind: str, existing_id: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.existing_id = existing_id


class UnsupportedFileType(DgmlError):
    code = "UNSUPPORTED_FILE_TYPE"


class InvalidPDF(DgmlError):
    code = "INVALID_PDF"


class GhostscriptNotFound(DgmlError):
    code = "GHOSTSCRIPT_NOT_FOUND"


class PageRenderFailed(DgmlError):
    code = "PAGE_RENDER_FAILED"


class PdfSliceFailed(DgmlError):
    code = "PDF_SLICE_FAILED"


class TextExtractionFailed(DgmlError):
    code = "TEXT_EXTRACTION_FAILED"


class NotImplementedMode(DgmlError):
    code = "NOT_IMPLEMENTED"


class InvalidArgument(DgmlError):
    code = "INVALID_ARGUMENT"


class CorruptMetadata(DgmlError):
    code = "CORRUPT_METADATA"


class ModelsConfigInvalid(DgmlError):
    """The ``[models]`` tier block is malformed (non-string / empty tier)."""

    code = "MODELS_CONFIG_INVALID"


class LegacyConfigPresent(DgmlError):
    """A pre-migration ``config.json`` is the only config present; the format is
    now TOML. Surfaced so the user runs ``dgml init`` to migrate."""

    code = "LEGACY_CONFIG_PRESENT"


class StorageConfigInvalid(DgmlError):
    """Malformed ``storage`` section (bad shape or provider option fields)."""

    code = "STORAGE_CONFIG_INVALID"


class StorageProviderUnresolvable(DgmlError):
    """A ``storage.provider`` dotted path is malformed, not importable, or does
    not resolve to a :class:`~dgml_core.storage_service.StorageService` subclass."""

    code = "STORAGE_PROVIDER_UNRESOLVABLE"


class StorageBackendMismatch(DgmlError):
    """The live ``storage`` config differs from the store this workspace was
    created with. Changing the store is a migration, not a config edit."""

    code = "STORAGE_BACKEND_MISMATCH"


class WorkspaceMigrationFailed(DgmlError):
    """A pending workspace-schema migration could not be applied.

    Migrations run automatically on open, so this is nearly always a
    permissions problem (a read-only mount). Failing loudly is deliberate:
    an un-migrated workspace reads as structurally valid but incomplete —
    e.g. pre-``assignment.json`` DocSet assignments would silently list as
    empty — and a wrong answer is worse than a refusal."""

    code = "WORKSPACE_MIGRATION_FAILED"


class OcrConfigInvalid(DgmlError):
    code = "OCR_CONFIG_INVALID"


class OcrConfigMissing(DgmlError):
    code = "OCR_CONFIG_MISSING"


class TextExtractionConfigInvalid(DgmlError):
    code = "TEXT_EXTRACTION_CONFIG_INVALID"


class StyleConfigInvalid(DgmlError):
    code = "STYLE_CONFIG_INVALID"


class ConversionConfigInvalid(DgmlError):
    code = "CONVERSION_CONFIG_INVALID"


class ConversionFailed(DgmlError):
    code = "CONVERSION_FAILED"


class AuthError(DgmlError):
    code = "AUTH_ERROR"


class ModelNotSupported(DgmlError):
    """An LLM model id litellm doesn't recognize — usually a misspelling, a
    wrong/absent provider prefix, or a model this litellm version doesn't know.
    Checked up front so a bad id surfaces clearly instead of as a confusing
    downstream parameter error at call time."""

    code = "MODEL_NOT_SUPPORTED"


class EmptyModelResponse(DgmlError):
    """A litellm.completion call returned successfully but carried no choices
    (zero candidates / zero completion tokens). Observed intermittently with
    Gemini and normally transient — the caller retries first and only raises
    this once an empty response persists across every attempt, so it surfaces
    clearly instead of as a downstream ``choices[0]`` IndexError."""

    code = "EMPTY_MODEL_RESPONSE"


class OcrFailed(DgmlError):
    code = "OCR_FAILED"


class ClassificationConfigMissing(DgmlError):
    code = "CLASSIFICATION_CONFIG_MISSING"


class ClassificationConfigInvalid(DgmlError):
    code = "CLASSIFICATION_CONFIG_INVALID"


class ClassificationFailed(DgmlError):
    code = "CLASSIFICATION_FAILED"


class ClusteringConfigInvalid(DgmlError):
    code = "CLUSTERING_CONFIG_INVALID"


class IncrementalWithoutClusters(DgmlError):
    """``dgml cluster --mode incremental`` with no existing DocSets to grow."""

    code = "INCREMENTAL_WITHOUT_CLUSTERS"


class AttestationInvalid(DgmlError):
    code = "ATTESTATION_INVALID"


class SchemaNotFound(NotFoundError):
    code = "SCHEMA_NOT_FOUND"


class GuidanceNotFound(NotFoundError):
    """The docset has no extraction guidance (``extraction-guidance.md``)."""

    code = "GUIDANCE_NOT_FOUND"


class SchemaInvalid(DgmlError):
    code = "SCHEMA_INVALID"


class GroundedConfigMissing(DgmlError):
    code = "GROUNDED_CONFIG_MISSING"


class GroundedConfigInvalid(DgmlError):
    code = "GROUNDED_CONFIG_INVALID"


class GenerationConfigMissing(DgmlError):
    code = "GENERATION_CONFIG_MISSING"


class GenerationConfigInvalid(DgmlError):
    code = "GENERATION_CONFIG_INVALID"


class SchemaGenerationFailed(DgmlError):
    code = "SCHEMA_GENERATION_FAILED"


class GenerationFailed(DgmlError):
    code = "GENERATION_FAILED"


class ValuesExtractionFailed(DgmlError):
    code = "VALUES_EXTRACTION_FAILED"


class GroundingFailed(DgmlError):
    code = "GROUNDING_FAILED"


class LinkPlanFailed(DgmlError):
    """The semantic-link model answered, but the answer could not be used —
    undecodable even item-wise, or a reviewer reply carrying no verdict for any
    candidate. Raised rather than returned as an empty plan: the caller
    content-addresses what it gets back, so a transport-level accident returned
    as "this document has no links" would be cached under the document's key and
    never retried."""

    code = "LINK_PLAN_FAILED"


class LabelModelUnreachable(DgmlError):
    # Never raised — labeling must never propagate, or it would discard a good
    # transcription. Exists so the per-file `label_error` payload surfaced by
    # `docset generate` draws its `code` from this registry (mirroring how
    # `grounding_error` reads `code` off a raised DgmlError).
    code = "LABEL_MODEL_UNREACHABLE"


class ChainConfigError(DgmlError):
    code = "CHAIN_CONFIG"


class ChainRpcFailed(DgmlError):
    code = "CHAIN_RPC"


class ChainTxReverted(DgmlError):
    code = "CHAIN_TX_REVERTED"


class WalletKeyMissing(DgmlError):
    code = "WALLET_KEY_MISSING"


class RecordNotFound(NotFoundError):
    code = "RECORD_NOT_FOUND"


class RegistryNotFound(NotFoundError):
    code = "REGISTRY_NOT_FOUND"


@dataclass
class RecordedError:
    """A persistent record of a fatal failure for a file or docset.

    ``permanent=True`` errors are skipped by ``dgml check`` until cleared
    with ``--retry-errors``. Use this for failures re-running cannot fix
    (corrupt PDF, missing system dependency, etc.).
    """

    operation: str
    message: str
    occurred_at: str
    permanent: bool = True

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RecordedError:
        return cls(
            operation=data["operation"],
            message=data["message"],
            occurred_at=data["occurred_at"],
            permanent=bool(data.get("permanent", True)),
        )


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string with second resolution."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def short_error_message(exc: BaseException, *, limit: int = 300) -> str:
    """Compact, single-line ``Type: message`` summary of an exception.

    For machine-readable JSON payloads, where the full text — which for
    LLM-provider errors can be a wall of nested JSON — would bloat the
    output. Whitespace (including newlines) is collapsed to single spaces
    and the result is capped at ``limit`` characters. The untruncated
    detail still reaches stderr under ``--verbose``.
    """
    detail = " ".join(str(exc).split())
    label = type(exc).__name__
    summary = f"{label}: {detail}" if detail else label
    if len(summary) > limit:
        return summary[: limit - 3] + "..."
    return summary


def load_recorded_errors(workspace: Workspace, file_id: str) -> list[RecordedError]:
    try:
        doc = workspace.docs.get_doc(layout.Collection.ERRORS, file_id)
    except CorruptMetadata:
        # Graceful: a corrupt errors.json should not block the consistency
        # check that reads it. Treat as "no errors recorded" — the caller
        # will (re)record any new failures it detects.
        return []
    if doc is None:
        return []
    return [RecordedError.from_json(item) for item in doc.get("errors", [])]


def append_recorded_error(workspace: Workspace, file_id: str, err: RecordedError) -> None:
    existing = load_recorded_errors(workspace, file_id)
    existing.append(err)
    workspace.docs.put_doc(
        layout.Collection.ERRORS, file_id, {"errors": [e.to_json() for e in existing]}
    )


def clear_recorded_errors(
    workspace: Workspace, file_id: str, operations: Iterable[str] | None = None
) -> int:
    """Delete recorded errors. If ``operations`` is given, only those are
    removed. Returns the number of errors removed."""
    existing = load_recorded_errors(workspace, file_id)
    if not existing:
        return 0
    if operations is None:
        workspace.docs.delete_doc(layout.Collection.ERRORS, file_id)
        return len(existing)
    ops = set(operations)
    keep = [e for e in existing if e.operation not in ops]
    if not keep:
        workspace.docs.delete_doc(layout.Collection.ERRORS, file_id)
    else:
        workspace.docs.put_doc(
            layout.Collection.ERRORS, file_id, {"errors": [e.to_json() for e in keep]}
        )
    return len(existing) - len(keep)
