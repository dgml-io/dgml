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
from .models import DocSet, FileRecord
from .storage import Workspace

__version__ = "0.1.0"

__all__ = [
    "AddFileResult",
    "ArtifactKind",
    "ArtifactRef",
    "AttestationEntry",
    "AttestationInventory",
    "CheckReport",
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
    "FileAttestation",
    "FileNotFound",
    "FileRecord",
    "FileStore",
    "FileVersion",
    "GhostscriptNotFound",
    "InvalidArgument",
    "InvalidPDF",
    "Issue",
    "PageRenderFailed",
    "UnsupportedFileType",
    "VerifyResult",
    "Workspace",
    "WorkspaceNotInitialized",
    "__version__",
    "attest_file",
    "attest_file_version",
    "check_workspace",
    "collect_file_version",
    "collect_from_attestation",
    "export_attestation",
    "load_conversion_config",
    "make_converter",
    "read_attestation",
    "verify_attestation_dir",
    "verify_bundle",
    "verify_file_version",
    "write_attestation",
]
