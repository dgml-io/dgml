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
from pathlib import Path
from typing import Any

from .storage import read_json, write_json_atomic


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


class LocalConfigMissing(DgmlError):
    """No peer ``local_config.json`` when ``dgml workspace create`` needs one."""

    code = "LOCAL_CONFIG_MISSING"


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


def load_recorded_errors(path: Path) -> list[RecordedError]:
    if not path.exists():
        return []
    try:
        raw = read_json(path)
    except CorruptMetadata:
        # Graceful: a corrupt errors.json should not block the consistency
        # check that reads it. Treat as "no errors recorded" — the caller
        # will (re)record any new failures it detects.
        return []
    return [RecordedError.from_json(item) for item in raw.get("errors", [])]


def append_recorded_error(path: Path, err: RecordedError) -> None:
    existing = load_recorded_errors(path)
    existing.append(err)
    write_json_atomic(path, {"errors": [e.to_json() for e in existing]})


def clear_recorded_errors(path: Path, operations: Iterable[str] | None = None) -> int:
    """Delete recorded errors. If ``operations`` is given, only those are
    removed. Returns the number of errors removed."""
    if not path.exists():
        return 0
    existing = load_recorded_errors(path)
    if operations is None:
        path.unlink()
        return len(existing)
    ops = set(operations)
    keep = [e for e in existing if e.operation not in ops]
    if not keep:
        path.unlink()
    else:
        write_json_atomic(path, {"errors": [e.to_json() for e in keep]})
    return len(existing) - len(keep)
