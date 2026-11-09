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

"""Command-line interface for DGML.

Designed for both humans and LLM-agent consumption: emits JSON to stdout
by default, errors as a stable JSON envelope on stderr, and uses
non-interactive flag-driven commands.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import json
import logging
import os
import shlex
import shutil
import sys
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping
from enum import StrEnum
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from dgml_core import layout
from dgml_core.auto_classification import (
    BulkClassifyBatch,
    classify_bulk_batch,
    prepare_bulk_classify,
)
from dgml_core.auto_classification import auto_classify as run_auto_classify
from dgml_core.classification import (
    ClassificationConfig,
    ClassifyMode,
    load_classification_config,
)
from dgml_core.clustering_types import ClusterMethod, ClusterMode
from dgml_core.consistency import check_workspace
from dgml_core.conversion import (
    CONVERTIBLE_SUFFIXES_TEXT,
    FAMILY_BY_SUFFIX,
    load_conversion_config,
    require_converter,
)
from dgml_core.default_config import PROVIDER_API_KEYS, PROVIDER_MODELS
from dgml_core.docset_generate import JobRequest, batch_stage_models, generate_docset
from dgml_core.docsets import DocSetStore
from dgml_core.errors import (
    BatchJobInvalid,
    BatchJobLeaseLost,
    BatchJobNondeterministic,
    BatchPending,
    ConflictError,
    ConflictErrorKind,
    DgmlError,
    InvalidArgument,
    MissingExtra,
    NoExistingDocSets,
    StorageBackendMismatch,
    ValuesNotFound,
    WorkspaceNotInitialized,
    short_error_message,
)
from dgml_core.files import AddFileResult, ConflictPolicy, FileStore
from dgml_core.generation.config import ThinkingMode, TranscriptionMode
from dgml_core.ids import RECORD_ID_SHAPE
from dgml_core.models import DocSet
from dgml_core.pages import DEFAULT_DPI, PAGE_MAX_EDGE, PAGE_MIN_EDGE
from dgml_core.storage import (
    API_KEY_ENV_VARS,
    Workspace,
    canonical_provider,
    detect_provider,
    detected_api_keys,
    read_json,
    user_config_path,
    write_user_config,
)
from dgml_core.storage import (
    ENV_VAR as WORKSPACE_ENV_VAR,
)
from dgml_core.storage_resolve import (
    DEFAULT_STORAGE_PROVIDER,
    DEFAULT_STORAGE_SERVICE,
    storage_fingerprint_pair,
)
from dgml_core.text_extraction import TextMode
from dgml_core.workspace_create import create_workspace
from dgml_core.workspace_id import ID_SHAPE, is_workspace_id
from dgml_core.workspaces_resolve import default_workspaces_store
from dgml_core.workspaces_store import WorkspacesStore

if TYPE_CHECKING:
    from dgml_core.grounded import GroundedConfig


class _Capture:
    """What a run emitted, held back instead of printed (see
    :func:`_begin_job_capture`), or what one ``docset run`` step emitted (see
    :func:`_capturing`)."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []


#: Set while a batch-job run holds its output back until its session is
#: closed; its stdout payloads / stderr error envelopes land here. None
#: otherwise.
_CAPTURE: _Capture | None = None


@contextlib.contextmanager
def _capturing() -> Iterator[_Capture]:
    """Hold what is emitted inside the block in a fresh :class:`_Capture`
    (``docset run`` drives each step's handler unchanged and nests its
    payload), restoring the previous capture after."""
    global _CAPTURE
    prev, _CAPTURE = _CAPTURE, _Capture()
    try:
        yield _CAPTURE
    finally:
        _CAPTURE = prev


# The CLI's own logger (``dgml.cli``); routed to stderr with the library's by
# `_configure_logging`. Diagnostics only — the JSON payload and the error
# envelope go through `_emit`/`_emit_error`, never through logging.
_log = logging.getLogger(__name__)


def _emit(payload: dict[str, Any], fmt: str, stream: IO[str] | None = None) -> None:
    if _CAPTURE is not None:
        (_CAPTURE.errors if stream is sys.stderr else _CAPTURE.payloads).append(payload)
        return
    out = stream or sys.stdout
    if fmt == "json":
        # A stream on another encoding (a terminal on a code page) gets ASCII
        # with JSON escapes, so the payload stays valid JSON there too; a
        # UTF-8 stream, or one with no encoding of its own, gets the characters.
        encoding = getattr(out, "encoding", None)
        escape = isinstance(encoding, str) and not _is_utf8(encoding)
        json.dump(payload, out, indent=2, ensure_ascii=escape)
        out.write("\n")
    else:
        out.write(_render_text(payload))


def _render_text(payload: Any, indent: int = 0) -> str:
    """Render a JSON-serializable payload as YAML-ish text for humans."""
    pad = "  " * indent
    if isinstance(payload, dict):
        if not payload:
            return f"{pad}{{}}\n"
        lines: list[str] = []
        for k, v in payload.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:\n{_render_text(v, indent + 1)}")
            else:
                lines.append(f"{pad}{k}: {_format_scalar(v)}\n")
        return "".join(lines)
    if isinstance(payload, list):
        if not payload:
            return f"{pad}[]\n"
        lines = []
        for item in payload:
            if isinstance(item, (dict, list)) and item:
                # First line gets the "- " bullet; subsequent lines indent.
                rendered = _render_text(item, indent + 1)
                first, _, rest = rendered.partition("\n")
                stripped = first.lstrip()
                lines.append(f"{pad}- {stripped}\n")
                if rest:
                    lines.append(rest if rest.endswith("\n") else rest + "\n")
            else:
                lines.append(f"{pad}- {_format_scalar(item)}\n")
        return "".join(lines)
    return f"{pad}{_format_scalar(payload)}\n"


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _error_details(exc: BaseException) -> dict[str, Any] | None:
    """The ``error.details`` an error carries itself, if any."""
    details = getattr(exc, "details", None)
    return dict(details) if isinstance(details, Mapping) and details else None


def _emit_error(
    code: str,
    message: str,
    fmt: str,
    *,
    details: dict[str, Any] | None = None,
) -> int:
    envelope: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        envelope["error"]["details"] = details
    _emit(envelope, fmt, stream=sys.stderr)
    return 1


_WORKSPACE_HELP = "Workspace root (overrides $DGML_HOME and the default ./dgml-workspace)."
_WORKSPACE_CONFIG_HELP = (
    "Path to the workspace's config.toml, when it is kept outside the workspace "
    "directory (overrides $DGML_CONFIG and the default <workspace>/config.toml). "
    "This file names the workspace's storage backend. Distinct from `cluster "
    "--config`, which selects a clustering preset."
)
_FORMAT_HELP = "Output format. Default 'json' for machine/agent consumption."
_VERBOSE_HELP = (
    "Emit informational diagnostics to stderr: INFO-level log lines from dgml "
    "and its library (hybrid text-mode warnings and per-page merge summary, "
    "the `docset generate` pipeline's progress lines, workspace-upgrade "
    "notices). Default off so stderr stays reserved for error envelopes and "
    "warnings. DGML_DEBUG=1 in the environment has the same effect."
)
_DEBUG_HELP = (
    "Keep intermediate pipeline files in the workspace and record LLM cost/token "
    "telemetry to usage.jsonl: the `docset generate` cache/ and "
    "coverage_report.json, the in-place grounding pass's grounding_stats.json, "
    "and a value extraction's extraction_stats.json (from `extraction extract`, "
    "`docset add-file`, auto-classifying `file add`, and `cluster`). Default off — only "
    "final files (DGML XML, page text/images, schemas, values, metadata) are "
    "kept and no usage rows are written. Does not change log output (that is "
    "--verbose / DGML_DEBUG=1)."
)


def _add_global_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Declare the global flags (`--workspace`, `--workspace-config`, `--format`,
    `--verbose`, `--debug`) in one place. ``suppress=False`` (top-level parser) gives
    them their real defaults; ``suppress=True`` (the shared parent threaded into every
    subparser) uses ``SUPPRESS`` so an omitted flag after the subcommand leaves
    the namespace untouched, letting the top-level default stand instead of
    clobbering it. Declaring both from one function keeps the two positions'
    metadata (choices/help) from drifting on a public-contract surface.

    ``--workspace-config`` is spelled in full rather than ``--config`` because
    ``dgml cluster --config`` already exists: a global ``--config`` would collide with
    it when this parent is attached to the ``cluster`` subparser, and argparse raises
    at parser-construction time, taking ``dgml --help`` down with it."""
    parser.add_argument(
        "--workspace",
        # Deliberately *not* `type=Path`: `Path("./notes")` normalizes to `notes`, and
        # that leading `./` is load-bearing. It is how a caller says "the directory, not
        # the workspace of that name" — the escape when a listed id shadows a local
        # directory — and `Workspace.resolve` can only honour it if it survives argparse.
        default=argparse.SUPPRESS if suppress else None,
        help=_WORKSPACE_HELP,
    )
    # Removed, but still declared so an existing caller gets the JSON error envelope
    # from `_reject_retired_config_flag` rather than an argparse usage dump. Same
    # treatment `workspace register` got.
    parser.add_argument(
        "--workspace-config",
        type=Path,
        default=argparse.SUPPRESS if suppress else None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default=argparse.SUPPRESS if suppress else "json",
        help=_FORMAT_HELP,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help=_VERBOSE_HELP,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=argparse.SUPPRESS if suppress else False,
        help=_DEBUG_HELP,
    )


def _dgml_version() -> str:
    """Installed `dgml` distribution version, for `--version`."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("dgml")
    except PackageNotFoundError:  # pragma: no cover - only when run from a non-installed tree
        return "unknown"


def _common_parser() -> argparse.ArgumentParser:
    """The global flags (`--workspace`, `--format`, `--verbose`), shared as a
    parent parser so they parse both *before* the subcommand
    (``dgml --format text file list``) and *after* it
    (``dgml file list --format text``).

    Defaults are ``SUPPRESS`` so that when a flag is omitted after the
    subcommand the child parser leaves the namespace untouched — the real
    default set by the top-level parser (which carries the same flags) stands
    rather than being clobbered back to its own default.
    """
    common = argparse.ArgumentParser(add_help=False)
    _add_global_flags(common, suppress=True)
    return common


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dgml",
        description="DGML — manage DocSets and Files (PDF -> DGML pipeline).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_dgml_version()}",
        help="Print the dgml version and exit.",
    )
    # Declared here with real defaults so the namespace always has them, and on
    # `common` (with SUPPRESS) so they also parse after the subcommand.
    _add_global_flags(parser, suppress=False)

    common = _common_parser()
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser(
        "init",
        parents=[common],
        help=(
            "Write the user-level config (~/.config/dgml/config.toml) with a [models] "
            "block (config only; run once per machine). Then `dgml workspace create`."
        ),
    )
    init_p.add_argument(
        "--provider",
        choices=sorted(PROVIDER_MODELS),
        default=None,
        help=(
            "Force a provider family's default models ([models] family = ...). Omit to "
            f"auto-detect from the API-key env vars that are set ({', '.join(API_KEY_ENV_VARS)})."
        ),
    )
    init_p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing config.toml (backing it up to config.toml.bak first). "
            "Without --force, init never clobbers an existing file."
        ),
    )

    workspace_sub = sub.add_parser(
        "workspace", parents=[common], help="Workspace lifecycle."
    ).add_subparsers(dest="workspace_command", required=True)
    ws_create = workspace_sub.add_parser(
        "create",
        parents=[common],
        help=(
            "Create a workspace (docsets/ + files/ + workspace.json) and its config.toml, "
            "which names the storage backend. Does not create or touch the user-level "
            "~/.config/dgml/config.toml — that is `dgml init`."
        ),
    )
    ws_create.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Directory to create the workspace in. Omit it and the workspace goes into "
            "this machine's store of workspaces instead, addressed by its ws_… id and "
            "shown by `dgml workspace list` — that is the default. Give a path here, or "
            "set the global --workspace or $DGML_HOME to one, for a workspace that lives "
            "in that directory and is addressed by path."
        ),
    )
    ws_create.add_argument(
        "--organization",
        default=None,
        help=(
            "Organization name. Embedded in this workspace's docset namespace URIs "
            "(http://dgml.io/<organization>/<DocSetSlug>) — pick a stable identifier for "
            "your org, as changing it later shifts the namespaces of newly generated XML. "
            "Required for a new workspace; optional when the config already records one "
            "(re-running create, or adopting an existing workspace on another machine), "
            "in which case passing a different value re-organizes the workspace and warns."
        ),
    )
    ws_create.add_argument(
        "--name",
        default=None,
        help=(
            "Human-readable workspace name (identity metadata, stored in workspace.json). "
            "Defaults to the workspace directory name."
        ),
    )
    ws_create.add_argument(
        "--id",
        default=None,
        metavar="WORKSPACE_ID",
        help=(
            f"Set the workspace's stable handle instead of generating one — {ID_SHAPE}, "
            "e.g. 'my-workspace'. It is what --workspace and $DGML_HOME address the "
            "workspace by, and the folder name the local store of workspaces gives it. "
            "Fails with CONFLICT if this machine's store of workspaces already holds "
            "that id. Omit it for a generated ws_… id."
        ),
    )
    ws_create.add_argument(
        "--storage",
        default=None,
        help=(
            "Name of the storage service to create this workspace on — a "
            "[storage.<name>] table in your config.toml. Its config is materialized "
            "into the new workspace's own config, which is authoritative from then on. "
            "Omit for the bundled local-disk default. Composes with --from-config: that "
            "flag supplies a config to start from, this one says which [storage.<name>] "
            "table in it to bind to."
        ),
    )
    ws_create.add_argument(
        "--from-config",
        default=None,
        metavar="PATH",
        help=(
            "Start this workspace from a config.toml you authored: its contents are "
            "copied verbatim (comments included) into the config the new workspace owns. "
            "A template, not an adopted file — the source is not tracked and later edits "
            "to it have no effect on the workspace."
        ),
    )
    workspace_sub.add_parser(
        "list",
        parents=[common],
        help=(
            "List the workspaces this machine's store of workspaces holds (id, name, "
            "organization, root). Opens no storage backend, so it works when a "
            "workspace's own blob store is unreachable. A workspace addressed only by "
            "path is not listed — 'dgml workspace import' adds one."
        ),
    )
    ws_import = workspace_sub.add_parser(
        "import",
        parents=[common],
        help=(
            "Add existing workspaces to this machine's store of workspaces. With no "
            "arguments, imports every workspace listed in the legacy "
            "~/.config/dgml/workspaces.json. Data never moves: a workspace on local disk "
            "keeps its directory, recorded as workspace_path in its own config."
        ),
    )
    ws_import.add_argument(
        "path",
        nargs="*",
        type=Path,
        help=("Workspace directories to import. Omit to sweep the legacy workspaces.json instead."),
    )
    ws_import.add_argument(
        "--move",
        action="store_true",
        help=(
            "Relocate each imported workspace's directory into the store of workspaces "
            "instead of recording where it already is. Off by default: moving a corpus "
            "of page images is not something to do on the caller's behalf."
        ),
    )
    ws_import.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be imported and write nothing.",
    )
    ws_import.add_argument(
        "--on-conflict",
        choices=[p.value for p in ImportConflictPolicy],
        default=ImportConflictPolicy.SKIP.value,
        help=(
            "What to do when the store already holds a workspace with the same id: "
            "skip it (default), fail the whole command, or replace the stored config."
        ),
    )
    ws_reseal = workspace_sub.add_parser(
        "reseal",
        parents=[common],
        help=(
            "Accept a change to a workspace's [storage] configuration: recompute its "
            "storage_fingerprint from the currently-resolved backends and record it. "
            "Run this after editing [storage] in the workspace's config.toml."
        ),
    )
    ws_reseal.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Workspace directory. Optional; defaults to the globally-resolved workspace.",
    )
    ws_reseal.add_argument(
        "--check",
        action="store_true",
        help=(
            "Report whether the seal matches without writing. Exits 1 with "
            "STORAGE_BACKEND_MISMATCH when the storage has drifted."
        ),
    )
    # Removed in favour of the self-healing index (a moved workspace is re-pointed on
    # open) and `workspace reseal` (which replaced `register --storage`). Kept declared
    # for one release so an existing caller gets a JSON error envelope naming the
    # replacement rather than an argparse usage dump on stderr.
    ws_register = workspace_sub.add_parser("register", parents=[common], help=argparse.SUPPRESS)
    ws_register.add_argument("path", nargs="?", type=Path, default=None)
    ws_register.add_argument("--storage", default=None)
    sub.add_parser("status", parents=[common], help="Show workspace summary.")

    chk = sub.add_parser(
        "check", parents=[common], help="Run a consistency check on the workspace."
    )
    chk.add_argument(
        "--retry-errors",
        action="store_true",
        help="Clear recorded permanent errors first and re-attempt failed operations.",
    )

    cluster_p = sub.add_parser(
        "cluster",
        parents=[common],
        help="Cluster files not currently assigned to any DocSet "
        "(requires `pip install dgml[clustering]`).",
    )
    cluster_p.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="No-op if all files are already assigned to a DocSet (safe to use when resuming).",
    )
    cluster_p.add_argument(
        "--config",
        dest="config",
        metavar="PRESET|PATH",
        default=None,
        help="Clustering configuration for this run. Either a bundled preset "
        "name (small | light | medium | heavy) or a path to a standalone config "
        "JSON (same shape as the 'clustering' section of <workspace>/config.toml "
        "— e.g. encoder_text, encoder_image, fusion, scenario). Replaces the "
        "workspace config's clustering section for this run. Defaults to the "
        "workspace config, or the bundled light preset when none is set.",
    )
    cluster_p.add_argument(
        "--mode",
        dest="mode",
        choices=[m.value for m in ClusterMode],
        default=ClusterMode.AUTO.value,
        help="Clustering mode. 'auto' (default) runs incremental clustering "
        "when the workspace already has DocSets (assign new files to existing "
        "clusters, open new clusters for the rest) and fresh clustering "
        "otherwise. 'fresh' always clusters from scratch; 'incremental' forces "
        "the incremental path and errors if no DocSets exist yet.",
    )
    cluster_p.add_argument(
        "--method",
        dest="method",
        choices=[m.value for m in ClusterMethod],
        default=ClusterMethod.AUTO.value,
        help="How documents are grouped, orthogonal to --mode. 'auto' (default) "
        "picks 'llm' for a FRESH run of at most --small-corpus-threshold "
        "clusterable files, and 'embedding' for everything else, incremental "
        "runs included whatever their batch size. 'embedding' forces the "
        "statistical encode → project → cluster pipeline: the right choice once "
        "a corpus is large enough for tf-idf / neighbor statistics to be "
        "meaningful, and too little signal below that. 'llm' forces sending "
        "every document's page images to the vision LLM in one call and letting "
        "it partition them. Both 'llm' and 'auto' (when it routes to the LLM) "
        "need the same `classification` config as --auto-classify. When that "
        "config is absent or unusable, 'auto' groups with 'embedding' instead "
        "and 'llm' reports the failure. The method that ran is echoed as "
        "`method` in the JSON result, null if none did.",
    )
    cluster_p.add_argument(
        "--small-corpus-threshold",
        dest="small_corpus_threshold",
        type=int,
        metavar="N",
        # Keep in sync with dgml_core.clustering.SMALL_CORPUS_MAX_FILES (8).
        default=8,
        help="With --method auto, route a fresh run of at most N clusterable "
        "files to the LLM partitioner, and larger ones to the embedding "
        "pipeline (default 8). Ignored for --method embedding / llm, and for "
        "incremental runs.",
    )

    docset = sub.add_parser("docset", parents=[common], help="DocSet management.").add_subparsers(
        dest="docset_command", required=True
    )
    _add_generate_subparser(docset, common)
    _add_run_subparser(docset, common)
    ds_create = docset.add_parser("create", parents=[common], help="Create a new DocSet.")
    ds_create.add_argument("--name", required=True)
    ds_create.add_argument(
        "--id",
        default=None,
        metavar="DOCSET_ID",
        help=(
            f"Assign this id to the new DocSet instead of generating one — {RECORD_ID_SHAPE}, "
            "e.g. 'q2-contracts'. Fails with CONFLICT if a DocSet "
            "already holds it. Omit it for a generated 12-character id."
        ),
    )
    ds_create.add_argument("--description", default="")
    ds_create.add_argument(
        "--key-question",
        dest="key_questions",
        action="append",
        default=None,
        help=(
            "Concrete question this DocSet's documents can answer from their "
            "first pages. Repeatable — pass once per question. Shown to "
            "auto-classification when deciding whether new files belong in "
            "this DocSet, so prefer type-discriminating questions over "
            "generic ones."
        ),
    )
    docset.add_parser("list", parents=[common], help="List DocSets.")
    ds_show = docset.add_parser("show", parents=[common], help="Show one DocSet.")
    ds_show.add_argument("docset_id")
    ds_update = docset.add_parser(
        "update", parents=[common], help="Update name and/or description."
    )
    ds_update.add_argument("docset_id")
    ds_update.add_argument("--name")
    ds_update.add_argument("--description")
    ds_delete = docset.add_parser(
        "delete", parents=[common], help="Delete a DocSet (does NOT delete its underlying Files)."
    )
    ds_delete.add_argument("docset_id")
    ds_addf = docset.add_parser("add-file", parents=[common], help="Assign a File to a DocSet.")
    ds_addf.add_argument("file_id")
    ds_addf.add_argument(
        "--docset",
        required=True,
        dest="docset_id",
        help="DocSet to assign the file to.",
    )
    ds_remf = docset.add_parser("remove-file", parents=[common], help="Remove a File assignment.")
    ds_remf.add_argument("file_id")
    ds_remf.add_argument(
        "--docset",
        required=True,
        dest="docset_id",
        help="DocSet to remove the file from.",
    )
    ds_lf = docset.add_parser(
        "list-files", parents=[common], help="List Files assigned to a DocSet."
    )
    ds_lf.add_argument("docset_id")

    _add_extraction_subparsers(sub, common)

    files = sub.add_parser("file", parents=[common], help="File management.").add_subparsers(
        dest="file_command", required=True
    )
    fl_add = files.add_parser(
        "add",
        parents=[common],
        help="Add a File (PDF or convertible source), or a whole directory (one JSON envelope).",
    )
    fl_add.add_argument(
        "path",
        type=Path,
        help=(
            f"Path to a file (.pdf, or a convertible source {CONVERTIBLE_SUFFIXES_TEXT} "
            "when a converter is configured), or a directory. When PATH is a "
            "directory, every ingestible file (case-insensitive) in it is added "
            "in one run and a summary envelope is returned. Use --recursive to "
            "descend into subdirectories."
        ),
    )
    fl_add.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "When PATH is a directory, descend into subdirectories. Ignored "
            "when PATH is a single file. Default: off (top-level only)."
        ),
    )
    fl_add.add_argument(
        "--id",
        default=None,
        metavar="FILE_ID",
        help=(
            f"Assign this id to the new File instead of generating one — {RECORD_ID_SHAPE}, "
            "e.g. 'invoice-2024-q1'. Fails with CONFLICT if another File already holds "
            "it with different content — no --on-conflict policy overrides that. "
            "Re-adding identical content under the same id is a no-op. Not allowed when "
            "PATH is a directory. Omit it for a generated 12-character id."
        ),
    )
    fl_add.add_argument(
        "--on-conflict",
        choices=[p.value for p in ConflictPolicy],
        default=ConflictPolicy.ERROR.value,
        help="How to react to a hash- or path-conflict with an existing File.",
    )
    fl_add.add_argument(
        "--text-mode",
        choices=[m.value for m in TextMode],
        default=TextMode.DIGITAL.value,
        help=(
            "How to extract text. 'digital' uses pdfminer.six on the PDF "
            "(default). 'ocr' uses the cloud provider configured in "
            "<workspace>/config.toml (requires `pip install dgml[aws]` or "
            "`pip install dgml[azure]`). 'hybrid' runs digital and OCR and "
            "merges them by grouping overlapping words into regions: "
            "digital wins when the two sides agree on content, OCR wins "
            "when they disagree, and digital-only regions (no overlapping "
            "OCR) are dropped as assumed-invisible. Pass --verbose to "
            "surface per-page merge decisions on stderr. Requires the "
            "same OCR config as 'ocr'."
        ),
    )
    fl_add.add_argument(
        "--dpi",
        type=_positive_int,
        default=DEFAULT_DPI,
        help=(
            f"Resolution to rasterize page images at, in dots per inch "
            f"(default: {DEFAULT_DPI}). Lower values (e.g. 150) roughly halve "
            "render time and page_images/ disk use and are usually ample for "
            "OCR and clustering; higher values cost render time. Each stored "
            f"image's longest edge is then fit between {PAGE_MIN_EDGE} and "
            f"{PAGE_MAX_EDGE} pixels, so "
            "a render outside that range is rescaled. The value is stored "
            "on the File as 'page_image_dpi' and reused by `dgml check "
            "--retry-errors`, and digital word boxes in page_text/ are written "
            "in this render's pixel space."
        ),
    )
    fl_add.add_argument(
        "--auto-classify",
        nargs="?",
        metavar="MODE",
        const=ClassifyMode.EXISTING_OR_NEW.value,
        default=None,
        choices=[m.value for m in ClassifyMode],
        help=(
            "After adding, use the configured vision LLM to assign the file "
            "to a DocSet. MODE is 'existing-or-new' (the default when the flag "
            "is passed bare): assign to an existing DocSet if one fits, "
            "otherwise create a new one. 'existing' never creates a DocSet: it "
            "assigns to an existing DocSet if one fits, otherwise leaves the "
            "file unassigned and reports decision 'none' (no auto-extraction). "
            "'existing-forced' always assigns to the closest existing DocSet, "
            "even an off-type file — use it only when every file is known to "
            "belong in one. With no DocSets to choose from, both 'existing' "
            "modes are an error (NO_EXISTING_DOCSETS, exit 1). Note MODE is consumed greedily, so "
            "put PATH before this flag (or pass MODE explicitly). Requires a "
            "'classification' section in <workspace>/config.toml; a missing or "
            "invalid config is a hard error (exit 1). Failures of the "
            "classification call itself (LLM error, auth) are reported in the "
            "'classification' field of the response payload without aborting "
            "the file add."
        ),
    )
    _add_batch_arguments(
        fl_add,
        what="the classification (and auto-extraction) of a directory add",
        extra=(
            " Requires PATH to be a directory and --auto-classify. In the 'existing' "
            "modes every file is classified in one wave; in the default mode the files are "
            "classified in order, one wave per file, because a DocSet created for one "
            "file must be visible to the next."
        ),
    )
    files.add_parser("list", parents=[common], help="List Files.")
    fl_show = files.add_parser("show", parents=[common], help="Show one File.")
    fl_show.add_argument("file_id")
    fl_delete = files.add_parser(
        "delete", parents=[common], help="Delete a File and remove all DocSet assignments to it."
    )
    fl_delete.add_argument("file_id")

    _add_dgmlx_subparser(sub, common)
    _add_node_subparser(sub, common)
    _add_discover_subparser(sub, common)
    _add_chain_subparsers(sub, common)
    _add_batch_subparser(sub, common)

    return parser


def _positive_int(raw: str) -> int:
    """argparse type: a strictly-positive integer (rejects 0 and negatives).

    Raising ``ArgumentTypeError`` keeps a bad value an argparse usage error
    (exit 2, before any workspace is touched) rather than a mid-ingest failure.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{raw!r} is not an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


# ---- batch mode: shared by every command that takes --batch -----------------
#
# A batch-capable command (1) adds its flags with _add_batch_arguments, (2)
# rejects, before any work, a stage whose model has no batch backend with
# _batch_preflight (BATCH_UNAVAILABLE, naming the stage and model) — or, for
# `docset generate`, leaves that to `generate_docset` — and (3)
# reports what its batch stages did as the payload's `batch` block: `enabled`
# plus each stage's WaveStats.to_json() (or a {"skipped": ...} entry). Without
# --batch none of this runs and the payload has no `batch` key.


def _positive_float(raw: str) -> float:
    """argparse type: a strictly-positive number of seconds."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{raw!r} is not a number") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def _add_batch_arguments(
    parser: argparse.ArgumentParser, *, what: str, extra: str = "", optional: bool = False
) -> None:
    """``--batch`` and ``--batch-poll-interval`` for a command that can send its
    *what* requests through a provider's batch API.

    ``optional=True`` makes ``--batch`` a ``--batch/--no-batch`` pair whose
    unset value is ``None``, for a command with a config key that turns batch
    mode on (the flag then overrides it either way)."""
    help_text = (
        f"Send {what} requests through the provider's batch API (half price; "
        "results usually within an hour, at most 24h) instead of one call at a "
        "time. A model whose provider has no batch backend is rejected before any "
        "work starts (BATCH_UNAVAILABLE), never silently run at full price. Adds "
        "a 'batch' block to the payload; without --batch the output is unchanged." + extra
    )
    if optional:
        parser.add_argument(
            "--batch", action=argparse.BooleanOptionalAction, default=None, help=help_text
        )
    else:
        parser.add_argument("--batch", action="store_true", help=help_text)
    parser.add_argument(
        "--batch-poll-interval",
        type=_positive_float,
        default=30.0,
        metavar="SECONDS",
        help="Seconds between batch status polls under --batch (default: 30).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help=(
            "Batch job mode: submit the batch wave and exit instead of waiting for "
            "it. Prints a 'batch_job' payload (exit 0) naming the job; continue it "
            "with `dgml batch resume <job_id>` (from a cron entry or an agent). "
            "Requires batch mode."
        ),
    )
    parser.add_argument(
        "--job",
        default=None,
        metavar="JOB_ID",
        help=(
            "Continue an existing batch job (what `dgml batch resume` passes): "
            "responses already received are replayed, batches still open are "
            "collected rather than resubmitted. Requires batch mode."
        ),
    )
    parser.add_argument(
        "--batch-deadline",
        default=None,
        metavar="DURATION",
        help=(
            "Stop waiting on the provider's batches this long after the run (or job) "
            "starts: 90m, 6h, 1d, or plain seconds. Once it passes, open batches are "
            "canceled, results the provider already produced are kept (batch price), "
            "and every remaining request of the run runs synchronously at standard "
            "price. A job stores it as an absolute UTC time on its first run; every "
            "resume honors it. Default: none (wait up to the provider's own limit). "
            "Requires batch mode."
        ),
    )


def _batch_preflight(stages: Mapping[str, Any]) -> None:
    """Reject batch mode before any work when a stage cannot batch.

    *stages* maps a stage name to its model id (or a representative request,
    see :func:`dgml_core.batch.assert_batchable`). Raises
    :class:`~dgml_core.errors.BatchUnavailable` naming the first stage whose
    provider has no batch backend. Imports the batch package lazily, so a
    command without ``--batch`` never loads it."""
    from dgml_core.batch import assert_batchable

    assert_batchable(stages)
    _log.info(f"[batch] batch mode on ({', '.join(stages)})")


def _batch_log(args: argparse.Namespace) -> Callable[[str], None]:
    """Batch-executor progress lines: INFO on the CLI logger — stderr under
    --verbose (see `_configure_logging`), else dropped (stdout carries only
    the JSON payload)."""
    return _log.info


def _parse_child_path(raw: str) -> list[int]:
    """Parse a slash-separated child-path string like ``'1/1'`` into ``[1, 1]``.

    An empty string (after stripping leading/trailing slashes) means the
    document root itself, i.e. ``[]``.
    """
    stripped = raw.strip("/")
    if not stripped:
        return []
    try:
        return [int(part) for part in stripped.split("/")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --child-path {raw!r}: must be slash-separated non-negative integers"
        ) from exc


def _add_node_subparser(
    sub: argparse._SubParsersAction,  # type: ignore[type-arg]
    common: argparse.ArgumentParser,
) -> None:
    """Register the top-level `node` command group (export + prove).

    Node-level attestation: one element of a file's generated DGML XML,
    addressed by Merkle leaf index or by XPath, with the inclusion proof
    connecting its hash to the document tree's Merkle root.
    """
    node = sub.add_parser(
        "node",
        parents=[common],
        help=(
            "Attest a single element of a File's DGML XML: export its hash, the tree's "
            "Merkle root, and the inclusion proof — or prove a previous export still holds."
        ),
    ).add_subparsers(dest="node_command", required=True)

    nd_export = node.add_parser(
        "export",
        parents=[common],
        help=(
            "Emit the attestation payload for one element: node hash, Merkle root, "
            "inclusion proof, canonical XPath, and the node's canonical XML."
        ),
    )
    nd_export.add_argument("file_id", help="ID of the File whose DGML XML holds the node.")
    nd_export.add_argument(
        "--docset",
        required=True,
        dest="docset_id",
        help="DocSet the DGML XML was generated in (node attestation is docset-scoped).",
    )
    sel = nd_export.add_mutually_exclusive_group(required=True)
    sel.add_argument(
        "--leaf",
        type=int,
        default=None,
        dest="leaf_index",
        help="0-based DFS pre-order leaf index of the element.",
    )
    sel.add_argument(
        "--xpath",
        default=None,
        help="XPath selecting exactly one element (the UX tree view's 'Copy XPath' value).",
    )
    sel.add_argument(
        "--child-path",
        default=None,
        dest="child_path",
        type=_parse_child_path,
        help=(
            "Slash-separated 0-based child-element indices from the document root "
            "(e.g. '1/1'), as a DOM tree view's Element.children would address the "
            "node. Empty string selects the root element."
        ),
    )

    nd_prove = node.add_parser(
        "prove",
        parents=[common],
        help=(
            "Re-verify a node export against the workspace's current DGML XML: the element "
            "at the proof's leaf index must still hash into the recorded Merkle root."
        ),
    )
    nd_prove.add_argument("file_id", help="ID of the File to prove against.")
    nd_prove.add_argument(
        "--docset",
        required=True,
        dest="docset_id",
        help="DocSet whose DGML XML to prove against.",
    )
    nd_prove.add_argument(
        "--proof",
        required=True,
        dest="proof_path",
        help="Path to a `node export` payload (or '-' for stdin); needs root_hash + proof.",
    )


_DISCOVER_FILTERS = [
    "all",
    "values",
    "sections",
    "density",
    "patterns",
    "who",
    "when",
    "amounts",
    "definitions",
    "rules",
]


def _add_discover_subparser(
    sub: argparse._SubParsersAction,  # type: ignore[type-arg]
    common: argparse.ArgumentParser,
) -> None:
    """Register the top-level ``discover`` command."""
    disc = sub.add_parser(
        "discover",
        parents=[common],
        help=(
            "Discover XML element subtrees in a File's generated DGML XML, "
            "grouped by tag type and filtered by structural role or semantic category."
        ),
    )
    disc.add_argument("file_id", help="ID of the File whose DGML XML to analyse.")
    disc.add_argument(
        "--docset",
        required=True,
        dest="docset_id",
        help="DocSet the DGML XML was generated in.",
    )
    disc.add_argument(
        "--filter",
        dest="filter_name",
        default="all",
        choices=_DISCOVER_FILTERS,
        metavar="FILTER",
        help=(
            "Filter to apply. Algorithmic: all (default), values, sections, density, "
            "patterns. Semantic (requires LLM config): who, when, amounts, definitions, "
            "rules."
        ),
    )
    disc.add_argument(
        "--samples",
        type=int,
        default=2,
        metavar="N",
        help="Maximum number of element samples to include per tag (default 2).",
    )
    disc.add_argument(
        "--include-structural",
        action="store_true",
        default=False,
        help="Include dg:-prefixed framework elements in the results.",
    )
    disc.add_argument(
        "--full",
        action="store_true",
        default=False,
        help=(
            "Full output: includes role, filters, depth_first, page, and XML attributes "
            "in each sample. Default strips attributes and drops role/filters/depth_first/page."
        ),
    )
    disc.add_argument(
        "--search",
        default=None,
        metavar="TERM",
        help="Case-insensitive substring filter on tag names (e.g. 'date', 'price').",
    )
    disc.add_argument(
        "--search-content",
        default=None,
        dest="search_content",
        metavar="TERM",
        help="Case-insensitive substring filter on sample XML text content.",
    )


def _add_batch_subparser(
    sub: argparse._SubParsersAction,  # type: ignore[type-arg]
    common: argparse.ArgumentParser,
) -> None:
    """Register `dgml batch status|resume|list|cancel|delete|prune|unlock`."""
    batch_group = sub.add_parser(
        "batch",
        parents=[common],
        help="Batch jobs: inspect, resume, or cancel a --batch run.",
    ).add_subparsers(dest="batch_command", required=True)
    bj_status = batch_group.add_parser(
        "status",
        parents=[common],
        help="Poll a job's provider batches once and report their state (read-only).",
    )
    bj_status.add_argument("job_id")
    bj_resume = batch_group.add_parser(
        "resume",
        parents=[common],
        help=(
            "Re-run a job's command: received responses are replayed, open batches "
            "collected, and it stops again (or finishes) exactly as the original run would."
        ),
    )
    bj_resume.add_argument("job_id")
    batch_group.add_parser("list", parents=[common], help="List batch jobs, newest first.")
    bj_cancel = batch_group.add_parser(
        "cancel",
        parents=[common],
        help=(
            "Cancel a job's open provider batches (and acknowledge any uncertain batch "
            "create); a later resume resubmits them."
        ),
    )
    bj_cancel.add_argument("job_id")
    bj_delete = batch_group.add_parser(
        "delete",
        parents=[common],
        help="Delete a job's stored state (refused while it has open provider batches).",
    )
    bj_delete.add_argument("job_id")
    bj_delete.add_argument(
        "--force",
        action="store_true",
        help="Cancel the job's open provider batches first, then delete it.",
    )
    bj_prune = batch_group.add_parser(
        "prune",
        parents=[common],
        help="Delete finished jobs (completed, or failed with no open provider batch).",
    )
    bj_prune.add_argument(
        "--older-than",
        type=float,
        default=0.0,
        metavar="DAYS",
        help="Only jobs last updated more than DAYS days ago (default: 0, every finished job).",
    )
    bj_unlock = batch_group.add_parser(
        "unlock",
        parents=[common],
        help="Break a job's lease left behind by a process that died.",
    )
    bj_unlock.add_argument("job_id")


def _add_dgmlx_subparser(
    sub: argparse._SubParsersAction,  # type: ignore[type-arg]
    common: argparse.ArgumentParser,
) -> None:
    """Register the top-level `dgmlx` command group (export + verify).

    A DGMLX bundle is the Merkle-attested, portable, filename-independent
    export of a File's DGML artifacts (the source document, page images, and
    — when a DocSet is named — its schema.json and the file's DGML XML).
    """
    dgmlx = sub.add_parser(
        "dgmlx",
        parents=[common],
        help=(
            "Export or verify a DGMLX bundle — the Merkle-attested, portable export of a "
            "File's DGML artifacts."
        ),
    ).add_subparsers(dest="dgmlx_command", required=True)

    dx_export = dgmlx.add_parser(
        "export",
        parents=[common],
        help=(
            "Write a DGMLX bundle — a single portable <stem>.dgmlx archive (a File's "
            "artifacts plus META-INF/dgml-attestation.xml carrying the Merkle root and "
            "inventory) — into --output-dir."
        ),
    )
    dx_export.add_argument("file_id", help="ID of the File to export.")
    dx_export.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        dest="output_dir",
        help="Directory to write the <stem>.dgmlx archive into.",
    )
    dx_export.add_argument(
        "--docset",
        default=None,
        dest="docset_id",
        help=(
            "Include the docset-scoped artifacts (schema.json, <stem>.dgml.xml) for this "
            "DocSet. Omit to export only the file-side artifacts (source, page images)."
        ),
    )
    dx_export.add_argument(
        "--unpacked",
        action="store_true",
        help=(
            "Write the unpacked bundle tree (source/, page_images/, META-INF/, "
            "[Content_Types].xml, _rels/, …) into --output-dir instead of the archive. "
            "By default only the .dgmlx archive is written; these two modes are mutually "
            "exclusive."
        ),
    )

    dx_verify = dgmlx.add_parser(
        "verify",
        parents=[common],
        help=(
            "Re-hash a DGMLX bundle's artifacts (ordered by its attestation file) and "
            "compare against the recorded Merkle root."
        ),
    )
    dx_verify.add_argument(
        "path",
        type=Path,
        help="A .dgmlx archive, or an unpacked bundle directory containing "
        "META-INF/dgml-attestation.xml.",
    )


def _add_keyring_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--keychain-service",
        default=os.environ.get("NVNM_KEY_SERVICE", "nvnm-wallet"),
        help="OS keyring service holding the signing key (env NVNM_KEY_SERVICE).",
    )
    p.add_argument(
        "--keychain-account",
        default=os.environ.get("NVNM_KEY_ACCOUNT", "default"),
        help="OS keyring account holding the signing key (env NVNM_KEY_ACCOUNT).",
    )


def _add_write_args(p: argparse.ArgumentParser) -> None:
    """Flags shared by every command that builds, signs, and broadcasts a tx."""
    p.add_argument(
        "--from",
        dest="from_address",
        default=os.environ.get("NVNM_FROM_ADDRESS"),
        help="Sender EVM address (env NVNM_FROM_ADDRESS); defaults to the keyring key's address.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and sign the transaction but do not broadcast; emit it for review.",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Use a legacy (type-0) transaction instead of EIP-1559.",
    )
    _add_keyring_args(p)


def _add_chain_subparsers(
    sub: argparse._SubParsersAction,  # type: ignore[type-arg]
    common: argparse.ArgumentParser,
) -> None:
    """Register the on-chain attestation command groups.

    `chain` manages chain configs; `wallet` reads balance/nonce; `registry`
    creates/lists registries; `stake` anchors a bundle or node; `prove`
    re-verifies an anchored record against the workspace. All require the
    `dgml[chain]` extra (handlers report MISSING_EXTRA when absent).
    """

    def _chain_config_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--chain-config",
            type=Path,
            default=None,
            dest="chain_config",
            help="Custom-chains JSON file (default $DGML_CHAINS or <workspace>/chains.json).",
        )

    def _chain_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--chain",
            dest="chain_name",
            default=os.environ.get("NVNM_CHAIN", "nvnm-testnet"),
            help="Configured chain to use (env NVNM_CHAIN; default nvnm-testnet).",
        )

    def _registry_arg(p: argparse.ArgumentParser, *, required: bool) -> None:
        p.add_argument(
            "--registry",
            default=os.environ.get("NVNM_REGISTRY"),
            required=required and not os.environ.get("NVNM_REGISTRY"),
            help="Registry NAME on the chain (env NVNM_REGISTRY).",
        )

    # --- chain ---------------------------------------------------------------
    chain = sub.add_parser(
        "chain", parents=[common], help="Manage chain configurations."
    ).add_subparsers(dest="chain_command", required=True)
    _chain_config_arg(
        chain.add_parser(
            "list", parents=[common], help="List configured chains (built-in + custom)."
        )
    )
    ch_show = chain.add_parser("show", parents=[common], help="Show one chain's configuration.")
    ch_show.add_argument("name")
    _chain_config_arg(ch_show)
    ch_add = chain.add_parser("add", parents=[common], help="Add (persist) a custom chain.")
    ch_add.add_argument("--name", required=True)
    ch_add.add_argument("--rpc-url", required=True, dest="rpc_url")
    ch_add.add_argument("--chain-id", required=True, type=int, dest="chain_id")
    ch_add.add_argument(
        "--anchor-address",
        dest="anchor_address",
        default="0x0000000000000000000000000000000000000A00",
        help="Anchor precompile/contract address (default the NVNM precompile).",
    )
    ch_add.add_argument("--explorer", default=None)
    ch_add.add_argument("--native-token", dest="native_token", default=None)
    _chain_config_arg(ch_add)
    ch_rm = chain.add_parser(
        "remove", parents=[common], help="Remove a custom chain (built-ins protected)."
    )
    ch_rm.add_argument("name")
    _chain_config_arg(ch_rm)

    # --- wallet --------------------------------------------------------------
    wallet = sub.add_parser(
        "wallet", parents=[common], help="Wallet status on a chain."
    ).add_subparsers(dest="wallet_command", required=True)
    wl_status = wallet.add_parser(
        "status", parents=[common], help="Show balance and pending nonce."
    )
    _chain_arg(wl_status)
    wl_status.add_argument(
        "--address", default=None, help="Address to inspect; defaults to the keyring key's address."
    )
    _chain_config_arg(wl_status)
    _add_keyring_args(wl_status)

    # --- registry ------------------------------------------------------------
    registry = sub.add_parser(
        "registry", parents=[common], help="Manage on-chain registries."
    ).add_subparsers(dest="registry_command", required=True)
    rg_create = registry.add_parser(
        "create", parents=[common], help="Create a registry (creator becomes admin)."
    )
    rg_create.add_argument("--name", required=True, help="Unique registry name.")
    rg_create.add_argument("--description", default="", help="Registry description.")
    rg_create.add_argument(
        "--metadata", default="{}", help="Registry metadata JSON (default '{}')."
    )
    _chain_arg(rg_create)
    _chain_config_arg(rg_create)
    _add_write_args(rg_create)
    rg_list = registry.add_parser(
        "list", parents=[common], help="List registries (optionally by name)."
    )
    rg_list.add_argument("--name", default=None, help="Filter to one registry name.")
    _chain_arg(rg_list)
    _chain_config_arg(rg_list)

    # --- stake ---------------------------------------------------------------
    stake = sub.add_parser(
        "stake",
        parents=[common],
        help="Anchor a DGMLX bundle or a single DGML node on a chain.",
    ).add_subparsers(dest="stake_command", required=True)
    st_file = stake.add_parser(
        "file", parents=[common], help="Anchor a file's DGMLX bundle (Merkle root)."
    )
    st_file.add_argument("file_id")
    st_file.add_argument("--docset", dest="docset_id", default=None)
    _registry_arg(st_file, required=True)
    _chain_arg(st_file)
    st_file.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help=(
            "Directory to write the <stem>.dgmlx archive (and record.json) into "
            "(default <workspace>/dgmlx-bundles/<ids>)."
        ),
    )
    st_file.add_argument(
        "--unpacked",
        action="store_true",
        help=(
            "Write the unpacked bundle tree (source/, page_images/, META-INF/, "
            "[Content_Types].xml, _rels/, …) into --output-dir instead of the archive. "
            "By default only the .dgmlx archive is written; these two modes are mutually "
            "exclusive."
        ),
    )
    _chain_config_arg(st_file)
    _add_write_args(st_file)

    st_node = stake.add_parser(
        "node", parents=[common], help="Anchor one element of a file's DGML XML."
    )
    st_node.add_argument("file_id")
    st_node.add_argument("--docset", dest="docset_id", required=True)
    st_node_sel = st_node.add_mutually_exclusive_group(required=True)
    st_node_sel.add_argument("--leaf", type=int, default=None, dest="leaf_index")
    st_node_sel.add_argument("--xpath", default=None, help="XPath from the UX 'Copy XPath'.")
    _registry_arg(st_node, required=True)
    _chain_arg(st_node)
    st_node.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help="Dir to save the fetched record.json (default <workspace>/dgmlx-bundles/<ids>).",
    )
    _chain_config_arg(st_node)
    _add_write_args(st_node)

    # --- prove ---------------------------------------------------------------
    prove = sub.add_parser(
        "prove",
        parents=[common],
        help="Re-verify an anchored record against the current workspace.",
    ).add_subparsers(dest="prove_command", required=True)
    for kind, helptext in (
        ("file", "Re-export the bundle and compare its Merkle root to the anchored checksum."),
        ("node", "Re-hash the element and re-walk its proof against the recorded root."),
    ):
        pv = prove.add_parser(kind, parents=[common], help=helptext)
        _chain_arg(pv)
        _registry_arg(pv, required=False)
        pv.add_argument("--checksum", default=None, help="Anchored checksum to look up on-chain.")
        pv.add_argument(
            "--record-json",
            default=None,
            dest="record_json",
            help="Saved record JSON (path or '-' for stdin) instead of a chain lookup.",
        )
        _chain_config_arg(pv)


def _verbose_enabled(args: argparse.Namespace) -> bool:
    """True when verbose diagnostics are on: the ``--verbose`` flag, or its
    env-var alias ``DGML_DEBUG`` set to ``1`` or ``true`` (any case).

    For the env var, ``1`` is the documented spelling; ``true``/``True`` is
    accepted because ``str(True)`` is how a Python caller naturally writes an
    env var. Strict otherwise, on purpose: a plain truthiness check would
    treat ``DGML_DEBUG=0`` — a CI line that means "off" — as *on*."""
    if getattr(args, "verbose", False):
        return True
    return os.environ.get("DGML_DEBUG", "").strip().lower() in ("1", "true")


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """A stream handler bound to whatever ``sys.stderr`` is *at emit time*.

    A plain ``StreamHandler(sys.stderr)`` captures the stream object it was built
    with; ``main()`` runs many times in one process under pytest, and each run's
    ``sys.stderr`` is a different capture buffer — closed by the time a later run
    writes to it."""

    @property
    def stream(self) -> IO[str]:
        return sys.stderr

    @stream.setter
    def stream(self, _value: IO[str]) -> None:
        pass


def _configure_logging(args: argparse.Namespace) -> None:
    """Route the ``dgml_core`` (library) and ``dgml`` (CLI) loggers to stderr.

    The CLI is just another caller of the library: it owns routing, so this is
    the one place logging is configured. Level is WARNING by default and INFO
    under ``--verbose``. ``DGML_DEBUG=1`` is an env-var alias for ``--verbose``
    (it long predates this switch, may be retired, and gets no role of its own);
    DEBUG has no CLI switch — records at that level exist for library callers.

    The WARNING default is load-bearing: stderr carries the JSON error envelope,
    and a plain-text INFO line ahead of a failing command would leave stderr
    holding text *and* JSON, breaking every agent that parses it. Under
    ``--verbose`` stderr is already non-JSON (the traceback goes there too).

    Idempotent: ``main()`` runs repeatedly in one process (tests), so a previous
    run's handler is replaced, not stacked.

    Deliberately process-owning: calling ``main()`` in-process adopts the CLI's
    routing — levels are set on ``dgml_core``/``dgml`` (clobbering the host's),
    the stderr handler is added, and ``propagate`` is left on (a host with a
    root handler sees records twice). ``main()`` is an entry point, not an
    embedding API; a program that wants dgml in-process with its own logging
    should call ``dgml_core`` directly."""
    level = logging.INFO if _verbose_enabled(args) else logging.WARNING
    for name in ("dgml_core", "dgml"):
        log = logging.getLogger(name)
        for old in [h for h in log.handlers if isinstance(h, _StderrHandler)]:
            log.removeHandler(old)
        handler = _StderrHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(level)


def _is_utf8(encoding: object) -> bool:
    """Whether ``encoding`` (a stream's ``encoding`` attribute) names UTF-8."""
    if not isinstance(encoding, str):
        return False
    try:
        return codecs.lookup(encoding).name == "utf-8"
    except LookupError:
        return False


def _configure_stream_encodings() -> None:
    """Never let a stream's encoding kill a command.

    Help text, error envelopes and JSON payloads carry characters outside
    cp1252. An interactive Windows console already gets UTF-8 (Python writes
    to it through the wide console API), but a redirected or piped stream
    gets the locale encoding, and its default ``strict`` handler turned the
    first such character into a ``UnicodeEncodeError`` from inside argparse:
    ``dgml --help | more`` and every wrapper that captured the output crashed
    on a stock Windows machine.

    A piped or redirected stdout or stderr gets UTF-8, since its reader is a
    program that wants the payload intact. A pipe or file gets UTF-8 even
    when ``PYTHONIOENCODING`` names another encoding; a terminal keeps its
    own. That is the contract for wrappers: what they capture is always
    UTF-8, whatever the locale or the environment says. A terminal keeps its
    code page, since UTF-8 bytes would show as mojibake there. Every stream,
    including one that is UTF-8 already, gets ``backslashreplace``, so a
    character the encoding lacks, or a lone surrogate from a file name,
    prints as an escape rather than crashing or silently becoming ``?``;
    :func:`_emit` keeps JSON valid on a non-UTF-8 stream by escaping
    non-ASCII itself, and a lone surrogate on a UTF-8 stream lands as a
    backslash escape that JSON accepts. A stream without ``reconfigure`` is
    left alone; one whose ``isatty`` is missing or fails is treated as
    redirected. stdin is not touched. ``main`` is the console
    entry point, so the change lasts for the process; a host that calls
    ``main()`` in-process on its own streams will find them reconfigured.
    ``PYTHONUTF8=1`` has the same effect and stays the way to get it for
    other tools in the same shell.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        utf8 = _is_utf8(getattr(stream, "encoding", None))
        try:
            tty = bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            tty = False
        try:
            if tty or utf8:
                reconfigure(errors="backslashreplace")
            else:
                reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):
            pass  # a closed or detached stream; nothing to fix


def main(argv: list[str] | None = None) -> int:
    _configure_stream_encodings()
    parser = _build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    # What a batch job replays on `dgml batch resume` (see _start_batch_job).
    args.raw_argv = raw_argv
    fmt: str = args.format
    _configure_logging(args)

    try:
        _reject_retired_config_flag(args)
        # `init` manages only the user-level config; `workspace create` is what
        # actually builds the workspace — so both run before the workspace exists
        # and get `resolve` (which only answers "which workspace") rather than
        # `open` (which also migrates it, and requires it to be initialized).
        if args.command in ("init", "workspace"):
            ws = Workspace.resolve(args.workspace)
        else:
            try:
                ws = Workspace.open(args.workspace)
            except WorkspaceNotInitialized as exc:
                # Re-raised with remedies the library cannot offer: every one of them
                # is a `dgml` command, and which of them applies depends on how this
                # invocation addressed the workspace (see `_uninitialized_message`).
                raise WorkspaceNotInitialized(
                    _uninitialized_message(
                        exc.workspace, from_default=_root_is_the_cwd_default(args)
                    ),
                    workspace=exc.workspace,
                ) from exc
            _warn_if_config_declares_workspaces(ws)
        return _dispatch_with_job(args, ws, fmt)
    except DgmlError as exc:
        return _emit_error(exc.code, str(exc), fmt)
    except Exception as exc:
        import traceback

        # The JSON error envelope carries a short, single-line cause so an
        # agent parsing it isn't handed a wall of provider error text. The full
        # traceback goes to stderr under --verbose (or DGML_DEBUG=1) — stderr is
        # already non-JSON under --verbose, so it can't corrupt the envelope.
        if _verbose_enabled(args):
            traceback.print_exc()
        return _emit_error("INTERNAL_ERROR", short_error_message(exc), fmt)


# Tier → the tasks it drives. Shown as inline comments in `dgml init`'s stderr
# report ONLY — never written into the generated config.toml, since the mapping
# may change without a config rewrite.
_TIER_CAPABILITIES = {
    "light": "classification, style",
    "standard": "transcription, text extraction",
    "advanced": "labeling, value extraction",
    "expert": "schema generation",
}

# Rendered into the `dgml init` advisories as `--provider <a|b|c>`. Derived from
# PROVIDER_MODELS so a provider added there shows up in the help text too.
_PROVIDER_CHOICES = "|".join(sorted(PROVIDER_MODELS))

# How to go beyond the family-only config `dgml init` writes.
_INIT_OVERRIDE_HINT = (
    '[dgml init] override a task with its own field (e.g. [generation] label_model = "..."); '
    f"switch families with dgml init --provider <{_PROVIDER_CHOICES}>."
)


def _init_models_report(provider: str) -> str:
    """What ``[models] family = "<provider>"`` expands to, with tier→capability
    comments — for the stderr advisory only. The config file names just the
    family, so this report is the user's view of the expansion."""
    tiers = PROVIDER_MODELS[provider]
    width = max(len(t) for t in _TIER_CAPABILITIES)
    lines = [f'  [models] family = "{provider}" expands to:']
    for tier in ("light", "standard", "advanced", "expert"):
        lines.append(f'  {tier.ljust(width)} = "{tiers[tier]}"    # {_TIER_CAPABILITIES[tier]}')
    return "\n".join(lines)


def _init_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Write the user-level ``~/.config/dgml/config.toml`` with a ``[models]``
    block (config only; does not create a workspace).

    The provider comes from ``--provider`` or, absent that, auto-detection from
    the API-key env vars that are set. Overwrites an existing file only with
    ``--force`` (backing it up to ``config.toml.bak``); otherwise a present
    file is left untouched.

    The stdout JSON payload is the contract; the human-readable report
    (detected keys, the [models] block, next steps) goes to stderr only under
    ``--verbose``.
    """

    environ = dict(os.environ)
    detected = detected_api_keys(environ)
    provider = args.provider if args.provider is not None else detect_provider(environ)
    canonical = canonical_provider(provider) if provider is not None else None
    path = user_config_path()

    written, backup = write_user_config(provider, overwrite=args.force)

    payload: dict[str, Any] = {
        "config_path": str(path),
        "config_created": written,
        "provider": canonical,
        "detected_keys": detected,
        "forced": bool(args.force and written),
    }

    if not written:
        payload["next_action"] = (
            f"{path} already exists — pass 'dgml init --force' to overwrite it "
            "(backs up to config.toml.bak), or edit it directly"
        )
        _emit(payload, fmt)
        return 0

    if backup is not None:
        _log.info(f"[dgml init] previous config backed up to {backup}.")

    if canonical is None:
        checked = ", ".join(API_KEY_ENV_VARS)
        payload["next_action"] = (
            f"set an API key, then rerun: dgml init --provider <{_PROVIDER_CHOICES}>"
        )
        _log.info(
            f"[dgml init] no API keys detected (checked {checked}).\n"
            f"[dgml init] wrote {path} with a commented-out [models] placeholder."
        )
        _emit(payload, fmt)
        return 0

    payload["next_action"] = "dgml workspace create --organization <org>"
    if args.provider is not None:
        keys = " and ".join(PROVIDER_API_KEYS[canonical])
        _log.info(
            f"[dgml init] wrote {path} (provider: {canonical}).\n"
            f"{_init_models_report(canonical)}\n"
            f"[dgml init] make sure {keys} is set before running dgml commands.\n"
            f"{_INIT_OVERRIDE_HINT}"
        )
    else:
        keys_line = "  ".join(f"[x] {k}" for k in detected) if detected else "(none)"
        _log.info(
            f"[dgml init] detected API keys: {keys_line}\n"
            f"[dgml init] wrote {path} (provider: {canonical}).\n"
            f"{_init_models_report(canonical)}\n"
            f"{_INIT_OVERRIDE_HINT}"
        )
    _emit(payload, fmt)
    return 0


def _root_is_the_cwd_default(args: argparse.Namespace, *, path: Path | None = None) -> bool:
    """Whether the workspace root came from the `./dgml-workspace` fallback rather than
    from something the caller named. ``path`` is a subcommand's own positional, if it has
    one."""
    return (
        path is None
        and getattr(args, "workspace", None) is None
        and not os.environ.get(WORKSPACE_ENV_VAR, "").strip()
    )


def _uninitialized_message(ws: Workspace, *, from_default: bool) -> str:
    """Why this workspace cannot be used, and remedies that actually work.

    ``is_initialized()`` is "has a config", so this one message answers two readings
    of the same fact: *never a workspace*, and *a workspace whose config is gone*.
    Nothing on disk distinguishes them for a remote-backed workspace, and for a local
    one the distinction would not change the advice — so both remedies are offered
    rather than guessed between.

    Addressed by id, the root is meaningless (the workspace lives in a store, not at a
    path), so that case names the store and the id instead.

    "Run 'dgml workspace create'" alone would be a loop: a bare `create` puts the
    workspace in the store of workspaces, so following it literally creates one
    *somewhere else* and leaves the next command failing identically. Every suggestion
    below resolves to the workspace the caller was actually asking about.
    """
    if ws.workspaces_id is not None:
        return (
            f"{ws.config_location} holds no config for {ws.workspaces_id}. It names this "
            f"workspace's storage backend and cannot be reconstructed — restore it from "
            f"backup, or run 'dgml workspace list' to see what this machine's store of "
            f"workspaces does hold."
        )
    looked = (
        " (dgml looked there because neither --workspace nor $DGML_HOME was set)"
        if from_default
        else ""
    )
    return (
        f"no workspace at {ws.root}: {ws.config_path} is missing{looked}. The config "
        f"names the storage backend and cannot be reconstructed — create a workspace "
        f"there with 'dgml workspace create {ws.root} --organization <org>', restore the "
        f"config from backup, or, if you already have a workspace, find it with "
        f"'dgml workspace list' and pass --workspace <ws_id>."
    )


#: One advisory per process, following the same pattern as models_config's
#: ``_WARNED_DISABLED``: several commands read the config more than once, and repeating
#: the same paragraph per read is noise rather than emphasis.
_WARNED_WORKSPACES_TABLE = False


def _warn_if_config_declares_workspaces(ws: Workspace) -> None:
    """Warn when a *workspace's* config declares ``[workspaces]``.

    That table selects the machine's store of workspaces and is read only from the user
    config, so here it does nothing at all. Silence would be the wrong answer: the table
    looks like it redirects where workspaces are listed, and a user who put it in the
    wrong file has no way to tell it is inert.

    Cannot be honoured even in principle — that store is what dgml used to *fetch* the
    file the table appears in."""
    global _WARNED_WORKSPACES_TABLE
    if _WARNED_WORKSPACES_TABLE:
        return
    text = ws.config_text
    if not text:
        return
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return  # the ordinary config read reports this properly, with a label
    if "workspaces" not in parsed:
        return
    _WARNED_WORKSPACES_TABLE = True
    _log.warning(
        f"Warning: {ws.config_location} declares a [workspaces] table, which is "
        f"ignored.\n\n"
        f"That table selects the machine's store of workspaces and is read only from "
        f"{user_config_path()} — it cannot be set per workspace, because that store is "
        f"what dgml used to find this workspace in the first place.\n\n"
        f"Move it to the user config, or delete it."
    )


def _workspace_config_file(ws: Workspace) -> str | None:
    """The workspace's config as a filesystem path, or ``None`` when it is not a file."""
    if ws.workspaces_id is None:
        return str(ws.config_path)
    found = default_workspaces_store().config_file(ws.workspaces_id)
    return str(found) if found is not None else None


def _reject_retired_config_flag(args: argparse.Namespace) -> None:
    """Refuse ``--workspace-config`` / ``$DGML_CONFIG``, naming the replacement.

    The flag conflated two things. As an *address* — "this workspace's config lives over
    there" — it only ever worked because the machine index recorded the location and
    handed it back on the next open; with the index gone as an authority there is nothing
    to remember it, so the flag would have to be repeated on every single invocation or
    the workspace would appear to have no config at all. As a *template* it is genuinely
    useful, and that is now ``workspace create --from-config``.

    Raised rather than silently ignored, and declared with a suppressed help string
    rather than removed, so an existing caller gets the ordinary JSON error envelope
    naming the replacement instead of an argparse usage dump."""
    if getattr(args, "workspace_config", None) is not None:
        raise InvalidArgument(
            "--workspace-config has been removed. A workspace's config now lives either "
            "in its own directory or in the machine's store of workspaces. To start a "
            "workspace from a config you authored, use "
            "'dgml workspace create --from-config <path>'."
        )
    if os.environ.get("DGML_CONFIG", "").strip():
        raise InvalidArgument(
            "$DGML_CONFIG has been removed, for the same reason as --workspace-config. "
            "Unset it; to start a workspace from a config you authored, use "
            "'dgml workspace create --from-config <path>'."
        )


def _read_seed_config(args: argparse.Namespace) -> str | None:
    """The text of ``--from-config``, or ``None``.

    A **template**, not an adopted file: its contents are copied into the config the new
    workspace owns and the source is then forgotten, so later edits to it have no effect.
    Copied verbatim, comments and key order included, since the point is that a config a
    user authored survives as written.

    A ``[workspaces]`` table in it is refused rather than ignored. That table selects the
    machine's store of workspaces, is read only from the user config, and would be
    silently inert here — a config that looks like it redirects where workspaces are
    listed but does not is worse than an error."""
    raw = getattr(args, "from_config", None)
    if raw is None:
        return None
    path = Path(raw).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InvalidArgument(
            f"--from-config {path} does not exist. Author it first (it should declare the "
            f"[storage] table this workspace will use), or omit the flag to have dgml "
            f"write the workspace's config for you."
        ) from exc
    except OSError as exc:
        raise InvalidArgument(f"could not read --from-config {path}: {exc}") from exc

    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise InvalidArgument(f"--from-config {path} is not valid TOML: {exc}") from exc
    if "workspaces" in parsed:
        raise InvalidArgument(
            f"--from-config {path} declares a [workspaces] table. That table selects the "
            f"machine's store of workspaces and is read only from the user config "
            f"({user_config_path()}), so it would have no effect here. Remove it."
        )
    return text


class ImportConflictPolicy(StrEnum):
    """``dgml workspace import --on-conflict``: what to do when the store of
    workspaces already holds the id being imported.

    ``SKIP`` (the default) leaves the stored one alone and reports the row
    skipped; ``FAIL`` aborts the command; ``REPLACE`` overwrites its stored config.
    """

    SKIP = "skip"
    FAIL = "fail"
    REPLACE = "replace"


class ImportStatus(StrEnum):
    """One workspace's outcome in ``dgml workspace import`` (a report row's ``status``).

    ``IMPORTED`` was written to the store; ``WOULD_IMPORT`` is ``IMPORTED`` under
    ``--dry-run``; ``SKIPPED`` was already in the store (``--on-conflict skip``);
    ``FAILED`` could not be imported (the row's ``reason`` says why).
    """

    IMPORTED = "imported"
    WOULD_IMPORT = "would-import"
    SKIPPED = "skipped"
    FAILED = "failed"


def _import_one(
    root: Path,
    *,
    store: WorkspacesStore,
    move: bool,
    dry_run: bool,
    on_conflict: ImportConflictPolicy,
    legacy_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """Import the workspace at ``root`` into ``store``. Returns one report row.

    Never raises for a bad workspace: a sweep of an old index must report what it could
    not take and keep going, or one dead directory strands every other workspace in the
    file."""
    from dgml_core import workspace_config as wsconfig
    from dgml_core.migrations import migrate_workspace_config

    row: dict[str, Any] = {"root": str(root)}
    if not root.is_dir():
        return {**row, "status": ImportStatus.FAILED.value, "reason": "directory does not exist"}

    source = Workspace(root=root)
    had_config = source.config_present
    if not dry_run:
        # Run the storage migration first: a pre-upgrade workspace kept its binding in
        # the legacy index row, so without this the config we are about to import would
        # not name a backend at all. `assume_local_when_unbound` covers the workspace that
        # recorded a binding nowhere — see the flag's docstring for why import may assume
        # and the per-command path may not.
        migrate_workspace_config(source, assume_local_when_unbound=True)
        source = Workspace(root=root)
    # Mirrors the migration's own condition (see `assume_local_when_unbound`): a binding
    # was assumed only when *neither* a config nor a legacy snapshot recorded one. Having
    # no config is not enough — the snapshot case reconstructs the real backend.
    had_snapshot = isinstance((legacy_row or {}).get("storage"), dict)
    row["assumed_local_storage"] = not had_config and not had_snapshot and not dry_run

    text = source.config_text
    if text is None:
        return {
            **row,
            "status": ImportStatus.FAILED.value,
            "reason": "no config.toml, and no workspace layout to infer a local one from",
        }

    identity = wsconfig.read_identity(source)
    legacy_id = (legacy_row or {}).get("workspace_id")
    workspace_id = identity.workspace_id or (legacy_id if isinstance(legacy_id, str) else None)
    if not workspace_id:
        # No identity anywhere: no `[workspace] workspace_id`, no `workspace.json`, and no
        # legacy index row. That is not a workspace dgml ever created — a directory with
        # `docsets/` and `files/` in it is not enough — so there is nothing to import it
        # *as*, and generating an id here would adopt an arbitrary directory as a workspace.
        return {
            **row,
            "status": ImportStatus.FAILED.value,
            "reason": (
                f"no workspace identity found — neither a [workspace] block, nor "
                f"{root / layout.WORKSPACE_FILE}, nor a row in the legacy index records a "
                f"workspace_id. If this directory is not a workspace dgml created, make "
                f"one with 'dgml workspace create {root} --organization <org>'."
            ),
        }
    row["workspace_id"] = workspace_id

    if not is_workspace_id(workspace_id):
        # Refuse rather than adopt it. A malformed id addresses nothing: the local backend
        # filters its folders by this same test, so the workspace would be written into a
        # directory `workspace_list` never looks at and `--workspace <id>` never resolves —
        # "imported" would mean "invisible". dgml's own generator only ever produces
        # well-formed ids, so this is a hand-edited or corrupted value, and the caller is
        # the one who can say what it should be.
        return {
            **row,
            "status": ImportStatus.FAILED.value,
            "reason": (
                f"workspace_id {workspace_id!r} is not well-formed — it must be "
                f"{ID_SHAPE}, or nothing can address "
                f"or list this workspace. Correct it in {source.config_location} (the "
                f"[workspace] block) and in {root / layout.WORKSPACE_FILE}, then re-run. "
                f"The legacy index is left in place, so nothing is lost meanwhile."
            ),
        }

    if store.exists(workspace_id):
        if on_conflict is ImportConflictPolicy.SKIP:
            return {
                **row,
                "status": ImportStatus.SKIPPED.value,
                "reason": "already in the store of workspaces",
            }
        if on_conflict is ImportConflictPolicy.FAIL:
            raise ConflictError(
                f"{store.label()} already holds {workspace_id}. Pass --on-conflict "
                f"replace to overwrite its stored config, or skip to leave it alone.",
                kind=ConflictErrorKind.WORKSPACE,
                existing_id=workspace_id,
            )

    # Where the data is, and whether that needs recording. `workspace_path` is a
    # LocalStore option, so it is only meaningful — and only accepted — when this
    # workspace's blobs actually live on local disk.
    target = store.workspace_root(workspace_id)
    blob_cfg, _doc_cfg = source.store_configs
    is_local = blob_cfg.provider == DEFAULT_STORAGE_PROVIDER
    needs_path = is_local and root.resolve() != target.resolve() and not move
    row["moved"] = bool(move and root.resolve() != target.resolve())
    row["workspace_path_recorded"] = needs_path

    if dry_run:
        return {**row, "status": ImportStatus.WOULD_IMPORT.value}

    if move and root.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(root), str(target))
        imported = Workspace(root=target, workspaces_id=workspace_id)
        store.write_config(workspace_id, text)
    else:
        store.write_config(workspace_id, text)
        imported = Workspace(root=root, workspaces_id=workspace_id)
        if needs_path:
            # Pin the data where it already is. This does *not* re-seal: `workspace_path`
            # is excluded from the storage fingerprint (see `_LOCATION_HINTS`), because a
            # workspace that has not moved is the same workspace on the same backend.
            service = identity.storage_service or DEFAULT_STORAGE_SERVICE
            table = wsconfig.read_storage_table(imported, service) or {}
            wsconfig.write_storage_table(
                imported,
                service,
                {**table, "workspace_path": str(root.resolve())},
            )

    row["config_location"] = imported.config_location
    return {**row, "status": ImportStatus.IMPORTED.value}


def _workspace_import(args: argparse.Namespace, fmt: str) -> int:
    """``dgml workspace import`` — adopt existing workspaces into the store.

    With paths, imports those directories. With none, sweeps the legacy
    ``workspaces.json`` an older dgml left behind, which is the migration path off the
    per-machine index: nothing happens automatically, so a machine adopts its old
    workspaces when its owner asks and not before."""
    from dgml_core import registry

    store = default_workspaces_store()
    roots: list[tuple[Path, dict[str, Any] | None]] = []
    source_label: str | None = None

    if args.path:
        # Look the legacy row up for a named path as well, not just for a sweep: for a
        # pre-upgrade workspace that row is the only record of its id and its binding, and
        # without it importing by path would fail on a workspace the sweep could take.
        roots = [
            (resolved, registry.raw_entry_by_root(resolved))
            for resolved in (p.expanduser().resolve() for p in args.path)
        ]
    else:
        source_label = str(registry.registry_path())
        for entry in registry.list_entries():
            if entry.root is None:
                continue
            root = Path(entry.root)
            roots.append((root, registry.raw_entry_by_root(root)))

    rows = [
        _import_one(
            root,
            store=store,
            move=args.move,
            dry_run=args.dry_run,
            on_conflict=ImportConflictPolicy(args.on_conflict),
            legacy_row=legacy,
        )
        for root, legacy in roots
    ]

    payload: dict[str, Any] = {
        "workspaces_store": store.label(),
        "imported": [
            r for r in rows if r["status"] in (ImportStatus.IMPORTED, ImportStatus.WOULD_IMPORT)
        ],
        "skipped": [r for r in rows if r["status"] == ImportStatus.SKIPPED],
        "failed": [r for r in rows if r["status"] == ImportStatus.FAILED],
        "dry_run": args.dry_run,
    }
    assumed = [
        r for r in rows if r.get("assumed_local_storage") and r["status"] == ImportStatus.IMPORTED
    ]
    if assumed:
        # Loud, and not behind --verbose: this is an assumption about which backend holds
        # the workspace's data, and only the caller can confirm it.
        listed = "\n".join(f"  {r['root']}" for r in assumed)
        _log.warning(
            f"Note: {len(assumed)} workspace(s) recorded no storage binding, so local disk "
            f"was assumed:\n{listed}\n\n"
            f"That is the only backend they could have used at the time. If any of them "
            f"actually kept its data on a remote backend, edit [storage] in its config and "
            f"run 'dgml workspace reseal <id>'."
        )
    if source_label is not None:
        payload["source"] = source_label
        # The legacy file is deliberately left in place, so import is re-runnable and a
        # half-finished sweep can simply be repeated.
        payload["source_removed"] = False
    _emit(payload, fmt)
    return 0 if not payload["failed"] else 2


def _requested_workspace_id(args: argparse.Namespace, ws: Workspace, *, listed: bool) -> str | None:
    """``workspace create --id``, validated against the workspace being created.

    Returns the id to use, or ``None`` when the caller passed none and one should be
    generated. Everything here runs before the workspace's config, directory or store row
    exists, so a rejected ``--id`` leaves nothing behind.

    Three ways it can fail, and they are different errors on purpose: a malformed id is
    the caller's typo (``INVALID_ARGUMENT``); an id that disagrees with one this
    workspace already records is a re-run that would *re-identify* an existing
    workspace, which ``create`` never does, and is also the caller's mistake; an id
    another workspace already holds is a genuine collision (``CONFLICT``), because
    proceeding would overwrite that workspace's config in the store.

    ``listed`` says a **new** store-listed workspace is being created, in which case
    ``ws`` is not it — it is whatever ``Workspace.resolve`` fell back to, and the caller
    discards it. See the comment on ``known`` below.
    """
    from dgml_core import workspace_config as wsconfig

    requested: str | None = args.id
    if requested is None:
        return None
    if not is_workspace_id(requested):
        raise InvalidArgument(
            f"--id {requested!r} is not a well-formed workspace id: it must be {ID_SHAPE}."
        )

    # What this workspace is *already* called, if anything: the id it is listed under,
    # else the one its own config records. `create` is documented as safe to re-run, so
    # an --id that agrees with it is a no-op rather than a conflict — including for a
    # detached workspace that has since been imported into the store, where the naive
    # `store.exists` check below would otherwise report the workspace colliding with
    # itself.
    #
    # Except when a new listed workspace is being created: then `ws` is only what
    # `Workspace.resolve` fell back to — `./dgml-workspace` in the working directory —
    # and the caller replaces it wholesale with one rooted at the new id. Reading an
    # identity off it would make `create --id` fail wherever a `./dgml-workspace`
    # happens to sit, complaining that "this workspace" has a different id, while the
    # same command without `--id` cheerfully generates one and ignores that directory.
    if listed and ws.workspaces_id is None:
        known = None
    else:
        known = ws.workspaces_id or wsconfig.read_identity(ws).workspace_id
    if known is not None:
        if known != requested:
            # Name *how* this workspace came to be addressed. Someone passing --id has
            # almost always come to create a new workspace and not realized something is
            # pointing at an existing one — easy when $DGML_HOME is set once and then
            # forgotten — so the message names that thing rather than describing "this
            # workspace" and leaving them to guess what to change.
            if args.path is not None:
                addressed = f"the path {str(args.path)!r} you gave"
                stop = "drop that path argument"
            elif getattr(args, "workspace", None) is not None:
                addressed = f"--workspace {str(args.workspace)!r}"
                stop = "drop --workspace"
            elif os.environ.get(WORKSPACE_ENV_VAR, "").strip():
                addressed = f"${WORKSPACE_ENV_VAR}"
                stop = f"unset {WORKSPACE_ENV_VAR}"
            else:  # pragma: no cover - defensive; one of the three is always set here
                addressed = "the workspace this command resolved"
                stop = "stop addressing it"
            raise InvalidArgument(
                f"--id {requested!r} does not match {known!r}, the id of the workspace "
                f"addressed by {addressed}.\n\n"
                f"To create a *new* workspace called {requested!r}, {stop} — it is what "
                f"points this command at the existing one."
            )
        return requested

    store = default_workspaces_store()
    if store.exists(requested):
        raise ConflictError(
            f"{store.label()} already holds a workspace {requested}. If it is this "
            f"workspace, re-run addressed to it — dgml --workspace {requested} workspace "
            f"create … (create is safe to re-run). Otherwise pick another --id.",
            kind=ConflictErrorKind.WORKSPACE,
            existing_id=requested,
        )
    return requested


def _workspace_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Workspace lifecycle: create, list, reseal.

    ``create`` writes the workspace directory *and* its ``config.toml``. The
    user-level config stays owned by ``dgml init``: this command does not create or
    touch it, and when it is absent the workspace is still created (never blocked)
    with a warning on stderr.
    """
    from dgml_core import workspace_config as wsconfig

    sub = args.workspace_command
    if sub == "create":
        # Where does this workspace live? A path the caller named puts it there
        # ("detached"); naming none puts it in the machine's store of workspaces, which
        # is the new default and the one behaviour change in this command. $DGML_HOME
        # still means "this directory is my workspace", so a setup that relies on it
        # keeps working untouched.
        listed = ws.workspaces_id is not None or not (
            args.path is not None
            or getattr(args, "workspace", None) is not None
            or os.environ.get(WORKSPACE_ENV_VAR, "").strip()
        )
        if args.path is not None:
            # A positional path overrides the globally-resolved root, so
            # `dgml workspace create ./ws …` reads without doubling --workspace.
            ws = Workspace(root=Path(args.path).expanduser().resolve())

        # --id, settled before anything is written. A rejected id must not leave a
        # half-built workspace behind, and for a listed workspace the id decides the
        # root, so there is no later point at which this could be checked.
        requested_id = _requested_workspace_id(args, ws, listed=listed)

        seed = _read_seed_config(args)

        result = create_workspace(
            None if (listed and ws.workspaces_id is None) else ws,
            workspace_id=requested_id,
            organization=args.organization,
            name=args.name,
            storage_service=args.storage,
            seed_toml=seed,
        )
        ws = result.workspace
        workspace_id = result.identity.workspace_id or ""
        name = result.identity.name or ""
        organization = result.identity.organization or ""
        service = result.identity.storage_service or DEFAULT_STORAGE_SERVICE

        if result.organization_changed_from is not None:
            # Loud, and not behind --verbose: this rewrites the organization for every
            # consumer of the workspace, and only affects *newly* generated XML, so the
            # corpus ends up split across two namespaces with nothing to flag it later.
            _log.warning(
                f"Warning: --organization {args.organization!r} differs from the "
                f"{result.organization_changed_from!r} recorded in {ws.config_location}.\n\n"
                f"The workspace is now organization {args.organization!r}. Docset "
                f"namespace URIs generated from here on will use it, while XML already "
                f"generated keeps the old namespace.\n\n"
                f"If this was a typo, re-run with --organization "
                f"{result.organization_changed_from!r}."
            )
        if result.storage_service_changed_from is not None:
            # Loud, and not behind --verbose: this rebinds where the workspace's data
            # lives. Artifacts already written stay on the old backend, so the corpus
            # ends up split across two with nothing to flag it later.
            _log.warning(
                f"Warning: --storage {args.storage!r} differs from the "
                f"{result.storage_service_changed_from!r} recorded in "
                f"{ws.config_location}.\n\n"
                f"This workspace's data now resolves through "
                f"[storage.{args.storage}]. Anything already written stays on "
                f"[storage.{result.storage_service_changed_from}] — dgml does not move "
                f"data.\n\n"
                f"If this was not intended, re-run with --storage "
                f"{result.storage_service_changed_from!r}."
            )

        upath = user_config_path()
        config_present = upath.exists()
        payload: dict[str, Any] = {
            "workspace": str(ws.root),
            "workspace_id": workspace_id,
            "name": name,
            "organization": organization,
            "storage_service": service,
            "initialized": True,
            # The config as a file a caller could open, or null when it is not one.
            # A workspace addressed by path always has one; a listed workspace has one
            # only if its store keeps configs as files (the local backend does, a
            # networked one does not) — and inventing a path for that case would invite
            # someone to try to restore it.
            "workspace_config_path": _workspace_config_file(ws),
            "config_location": ws.config_location,
            "listed": ws.workspaces_id is not None,
            "storage_fingerprint": result.identity.storage_fingerprint,
            "config_path": str(upath),
            "config_present": config_present,
        }
        if not config_present:
            # Succeed but warn — LLM-backed commands will fail until the user
            # configures credentials. Always on stderr (no --verbose needed).
            keys = " / ".join(API_KEY_ENV_VARS)
            payload["next_action"] = f"run `dgml init` and set one of {keys}"
            _log.warning(
                "Warning: no user-level config found.\n\n"
                "Some commands will fail until credentials are configured.\n\n"
                f"Run `dgml init` and set one of {keys}."
            )
        if ws.workspaces_id is not None:
            # A listed workspace has no path to use as a handle, and a bare next command
            # resolves `./dgml-workspace` and fails — so say what the handle is. Always on
            # stderr, because this is the likeliest place to get stuck and one field among
            # thirteen in a JSON blob is not where a human looks.
            #
            # dgml only ever *prints* the export line. It does not set the variable and
            # does not touch a shell profile: the caller's environment is theirs.
            #
            # `setdefault`, so a missing user config — which stops every LLM command, not
            # just this one — keeps the more urgent `next_action`. The stderr line below
            # still reaches the user either way.
            payload.setdefault(
                "next_action",
                f"address it with --workspace {workspace_id} (or: export DGML_HOME={workspace_id})",
            )
            _log.warning(
                f"Workspace {workspace_id} is in this machine's store of workspaces, not a "
                f"directory here.\n\n"
                f"Use it with:  dgml --workspace {workspace_id} <command>\n"
                f"or, for this shell:  export DGML_HOME={workspace_id}\n\n"
                f"'dgml workspace list' shows it again later."
            )
        _emit(payload, fmt)
        return 0

    if sub == "list":
        store = default_workspaces_store()
        _emit(
            {
                "workspaces": [
                    {
                        "workspace_id": e.workspace_id,
                        "name": e.name,
                        "organization": e.organization,
                        "storage_service": e.storage_service,
                        # Computed on *this* machine (the declared `workspace_path`, else
                        # the standard folder) and never stored: where a workspace's files
                        # sit is per-machine, so a shared store recording it would be the
                        # per-machine index mistake all over again.
                        "root": str(store.workspace_root(e.workspace_id or "")),
                        "created_at": e.created_at,
                    }
                    for e in store.list_entries()
                ],
                "workspaces_store": store.label(),
            },
            fmt,
        )
        return 0

    if sub == "import":
        return _workspace_import(args, fmt)

    if sub == "reseal":
        if args.path is not None:
            # Takes a path or a workspace id, through the same resolution every other
            # command uses — a workspace listed in the store has no path to name.
            ws = Workspace.resolve(args.path)
        if not ws.is_initialized():
            raise WorkspaceNotInitialized(
                _uninitialized_message(
                    ws, from_default=_root_is_the_cwd_default(args, path=args.path)
                ),
                workspace=ws,
            )
        previous = wsconfig.read_identity(ws).storage_fingerprint
        blob_cfg, doc_cfg = ws.store_configs
        current = storage_fingerprint_pair(blob_cfg, doc_cfg)
        if args.check:
            if previous and previous != current:
                raise StorageBackendMismatch(
                    f"the [storage] configuration this workspace resolves no longer "
                    f"matches the storage_fingerprint recorded in {ws.config_location}. "
                    f"Run 'dgml workspace reseal {ws.root}' to accept the change."
                )
        else:
            wsconfig.write_identity(ws, storage_fingerprint=current)
        _emit(
            {
                "workspace": str(ws.root),
                "workspace_id": wsconfig.read_identity(ws).workspace_id,
                "config_location": ws.config_location,
                "storage": {
                    "blobs": {"provider": blob_cfg.provider},
                    "docs": {"provider": doc_cfg.provider},
                },
                "storage_fingerprint": current,
                "previous_fingerprint": previous,
                "resealed": not args.check,
            },
            fmt,
        )
        return 0

    if sub == "register":
        # Removed. Declared only so an existing caller gets an error envelope naming
        # the replacement instead of an argparse usage dump.
        if args.storage is not None:
            raise InvalidArgument(
                "`dgml workspace register --storage` has been removed. A workspace's "
                "storage now lives in its own config.toml: edit the [storage] table "
                "there, then run `dgml workspace reseal <path>` to accept the change."
            )
        raise InvalidArgument(
            "`dgml workspace register` has been removed. A workspace created without a "
            "path is listed in this machine's store of workspaces automatically; one that "
            "already exists in a directory is added with `dgml workspace import <path>`. "
            "Use `dgml workspace list` to confirm."
        )

    raise AssertionError(f"unhandled workspace subcommand: {sub}")  # unreachable (required=True)


# ---- batch jobs: one session per --batch run --------------------------------
#
# Every --batch run opens a batch job session (_start_batch_job): a plain
# blocking run silently, for crash recovery; --no-wait / --job explicitly.
# `docset generate` instead hands the library a JobRequest, which opens the
# session after its pre-flights and closes it itself (the command leaves it
# on its args for the payload). _dispatch_with_job closes a session the command opened
# exactly once with how the run ended, and only then emits what the run
# printed (one outcome per run).


def _active_batch_job() -> Any:
    """The running batch job session, if any — without importing the batch
    package (litellm stays off the default import path): no session can exist
    unless ``dgml_core.batch.jobs`` was already imported."""
    jobs = sys.modules.get("dgml_core.batch.jobs")
    return None if jobs is None else jobs.active_session()


#: `args` attribute a command sets to the session the library opened and closed
#: for its run (`docset generate` under a JobRequest): no longer active, still
#: what the payload reports on. On the parsed args, not module state, so a
#: nested dispatch (`batch resume` replaying the command) never reports on
#: the inner run's job a second time.
_CLOSED_JOB_ATTR = "closed_batch_job"


def _run_batch_job(args: argparse.Namespace) -> Any:
    """This run's batch job session, running or already closed by the library."""
    active = _active_batch_job()
    return active if active is not None else getattr(args, _CLOSED_JOB_ATTR, None)


def _begin_job_capture() -> None:
    """Hold back what this run emits until its batch job session is closed.

    How a job run ends is only known once :func:`_dispatch_with_job` closes the
    session, after the command has built its payload: the nondeterminism guard
    may still have to replace a success payload with its error envelope (one
    outcome per run), and a job left behind is reported inside the payload's
    ``batch`` block."""
    global _CAPTURE
    if _CAPTURE is None:
        _CAPTURE = _Capture()


def _end_job_capture() -> _Capture | None:
    """Stop holding back output; the held output, or None if nothing was held."""
    global _CAPTURE
    capture, _CAPTURE = _CAPTURE, None
    return capture


def _flush_job_capture(
    capture: _Capture | None,
    fmt: str,
    *,
    job: dict[str, Any] | None,
    payloads: bool = True,
    deadline: dict[str, Any] | None = None,
) -> None:
    """Emit the output a job run held back. *job* (a job the run left behind)
    and *deadline* (the job's batch deadline block, when it has one) go into
    the last payload's ``batch.job`` / ``batch.deadline`` — or, when the run
    emitted no payload, into its last error envelope's ``details.batch``."""
    if capture is None:
        return
    fields = {name: value for name, value in (("job", job), ("deadline", deadline)) if value}
    if payloads and capture.payloads:
        for name, value in fields.items():
            _attach_batch_field(capture.payloads[-1], name, value)
    elif fields and capture.errors:
        error = capture.errors[-1].get("error")
        if isinstance(error, dict):
            batch = error.setdefault("details", {}).setdefault("batch", {})
            if isinstance(batch, dict):
                batch.update(fields)
    if payloads:
        for payload in capture.payloads:
            _emit(payload, fmt)
    for envelope in capture.errors:
        _emit(envelope, fmt, stream=sys.stderr)


def _attach_batch_field(payload: dict[str, Any], name: str, value: dict[str, Any]) -> None:
    """Set *payload*'s ``batch.<name>`` (the block created if absent)."""
    block = payload.setdefault("batch", {})
    if isinstance(block, dict):
        block[name] = value


def _deadline_block(session: Any) -> dict[str, Any] | None:
    """The just-closed *session*'s ``batch.deadline`` block, if it had one."""
    if session is None:
        return None
    try:
        block = session.deadline_json()
    except Exception:
        return None
    return block if isinstance(block, dict) else None


def _left_behind(session: Any) -> dict[str, Any] | None:
    """The resumable job the just-closed *session* left in the workspace, as
    a payload block — or None when the run finished its job (completed, or a
    silent job deleted with the run)."""
    if session is None:
        return None
    # A session exists only once ``dgml_core.batch.jobs`` is imported, so this
    # import is free here — and litellm stays off the error path of a run
    # that never opened one.
    from dgml_core.batch.jobs import JobStatus

    if session.manifest.status != JobStatus.FAILED:
        return None
    try:
        if not session.store.exists():
            return None
    except Exception:
        return None
    return {
        "job_id": session.job_id,
        "status": JobStatus.FAILED.value,
        "resume": f"dgml batch resume {session.job_id}",
    }


def _emit_nondeterministic(guard: BatchJobNondeterministic, session: Any, fmt: str) -> int:
    """The nondeterminism guard's error envelope, naming the job it left
    (failed, its in-flight batches kept) and, in `docset run`, the step."""
    details: dict[str, Any] = {}
    if session is not None:
        from dgml_core.batch.jobs import RUN_STEP_STATE

        step = session.state.get(RUN_STEP_STATE)
        if step is not None:
            details["step"] = step
        job = _left_behind(session)
        if job is not None:
            details["batch"] = {"job": job}
    return _emit_error(guard.code, str(guard), fmt, details=details or None)


def _emit_lease_lost(lost: BatchJobLeaseLost, session: Any, fmt: str) -> int:
    """The error envelope of a run that lost its job's lease mid-run
    (``BATCH_JOB_BUSY``): the run's one outcome, whatever it emitted before —
    the job now belongs to the lease's holder, so this run's output is void."""
    details: dict[str, Any] = {"lease_lost": True}
    if session is not None:
        details["batch"] = {
            "job": {"job_id": session.job_id, "resume": f"dgml batch resume {session.job_id}"}
        }
    return _emit_error(lost.code, str(lost), fmt, details=details)


def _fenced(session: Any) -> BatchJobLeaseLost | None:
    """The lease loss that fenced *session* (after its close), if any."""
    return getattr(session, "lease_lost", None) if session is not None else None


def _dispatch_with_job(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """:func:`_dispatch`, closing the batch job session a command opened.

    A command that runs in batch job mode opens its session itself (see
    :func:`_start_batch_job`) or has the library open and close it (a
    :class:`JobRequest`); this is the one place every exit path passes, so
    the session is closed exactly once with how the run ended (a close is
    idempotent), and the output the run held back is emitted only then. A
    run that paused
    (:class:`BatchPending`, ``--no-wait``) is not an error: it prints the job's
    ``batch_job`` payload on stdout and exits 0. A run the nondeterminism guard
    stopped (:class:`BatchJobNondeterministic`) or that lost its lease has
    exactly one outcome — its error envelope, exit 1 — whatever the command
    emitted before. A run that leaves a failed, resumable job behind names it
    in its payload's ``batch.job`` (``error.details.batch.job`` of an error
    envelope)."""
    try:
        rc = _dispatch(args, ws, fmt)
    except BatchPending as pending:
        session = _run_batch_job(args)
        command = session.manifest.command if session is not None else None
        if session is not None:
            session.close(pending)
        if (fenced := _fenced(session)) is not None:
            _end_job_capture()
            return _emit_lease_lost(fenced, session, fmt)
        # A paused run's one result is the job to resume.
        _flush_job_capture(_end_job_capture(), fmt, job=None, payloads=False)
        from dgml_core.batch.jobs import RUN_STEP_STATE, deadline_summary, pending_job_payload

        block = pending_job_payload(
            pending.job_id,
            command=command,
            submitted_batches=pending.submitted_batches,
            requests_in_flight=pending.requests_in_flight,
            deadline=deadline_summary(session.manifest) if session is not None else None,
            step=session.state.get(RUN_STEP_STATE) if session is not None else None,
        )
        _emit({"batch_job": block}, fmt)
        return 0
    except BatchJobNondeterministic as guard:
        session = _run_batch_job(args)
        if session is not None:
            session.close(guard)
        _end_job_capture()  # whatever the run emitted before is void
        return _emit_nondeterministic(guard, session, fmt)
    except BatchJobLeaseLost as lost:
        # Another process drives the job now: this run stopped, fenced.
        session = _run_batch_job(args)
        if session is not None:
            session.close(lost)
        _end_job_capture()
        return _emit_lease_lost(lost, session, fmt)
    except BaseException as exc:
        session = _run_batch_job(args)
        if session is not None:
            session.close(exc)
        if (fenced := _fenced(session)) is not None:
            _end_job_capture()
            return _emit_lease_lost(fenced, session, fmt)
        job = _left_behind(session)
        deadline = _deadline_block(session)
        _flush_job_capture(_end_job_capture(), fmt, job=None, deadline=deadline)
        if isinstance(exc, DgmlError) and (job is not None or deadline is not None):
            # The failure's envelope names the job left behind and the deadline
            # block (possibly_double_billed, a settling job's id): with no
            # payload, it is the only place an agent can learn them.
            batch = {name: value for name, value in (("job", job), ("deadline", deadline)) if value}
            return _emit_error(exc.code, str(exc), fmt, details={"batch": batch})
        raise
    session = _run_batch_job(args)
    job = None
    if session is not None:
        drift = session.nondeterministic
        session.close(None, ok=rc == 0)
        if (fenced := _fenced(session)) is not None and drift is None:
            _end_job_capture()
            return _emit_lease_lost(fenced, session, fmt)
        if drift is not None:
            # Backstop: the guard is a BaseException, so this is reached only
            # if something caught it on the way up. Still one outcome.
            _end_job_capture()
            return _emit_nondeterministic(drift, session, fmt)
        job = _left_behind(session)
    _flush_job_capture(_end_job_capture(), fmt, job=job, deadline=_deadline_block(session))
    return rc


def _start_batch_job(
    args: argparse.Namespace,
    ws: Workspace,
    *,
    command: str,
    pinned: Mapping[str, str | bool] | None = None,
) -> Any:
    """Open this run's batch job session (every ``--batch`` run has one).

    A plain blocking ``--batch`` run gets a fresh job too, silently: it records
    every response, so a run that crashes mid-way can be resumed with ``dgml
    batch resume`` instead of paying again — and its payload is unchanged.
    ``--no-wait`` makes the run stop once a wave is submitted; ``--job``
    continues an existing job. Closed by :func:`_dispatch_with_job`."""
    if getattr(args, "run_step", False):
        # A step of `docset run`: the run opened the job (one job spans every
        # step, so `dgml batch resume` replays the whole pipeline).
        session = _active_batch_job()
        if session is not None:
            return session
    from dgml_core.batch import start_session

    request = _job_request(args, command=command, pinned=pinned)
    session = start_session(
        ws,
        command=command,
        argv=request.argv,
        job_id=request.job_id,
        wait=request.wait,
        cwd=request.cwd,
        log=request.log or _batch_log(args),
        deadline_s=request.deadline_s,
    )
    # From here the run's output waits for the session's close (one outcome
    # per run; a job left behind is named in the payload).
    _begin_job_capture()
    if not session.resumed:
        # Name the job up front: if this run is interrupted (crash, kill,
        # reboot, lost network), this id is what picks it up again. A run that
        # finishes cleanly deletes its job, so the id only matters on failure.
        _batch_log(args)(
            f"[batch-job] job {session.job_id}: if this run is interrupted, "
            f"continue it with `dgml batch resume {session.job_id}` "
            "(`dgml batch list` also shows it)"
        )
    return session


def _job_request(
    args: argparse.Namespace, *, command: str, pinned: Mapping[str, str | bool] | None = None
) -> JobRequest:
    """This run's batch job as the library opens it: the job to continue
    (``--job``), the run's mode (``--no-wait``, ``--batch-deadline``), and the
    command line a resume replays."""
    # Stored canonical (abbreviated flags spelled in full): resume and the
    # `--no-wait` short-circuit inspect it token by token.
    argv = _canonical_argv(list(getattr(args, "raw_argv", [])))
    # Batch mode may have come from config ([generation] batch = true); record
    # it explicitly so a resume replays in batch mode whatever the config says.
    if "--batch" not in argv:
        argv.append("--batch")
    # Likewise every *pinned* option the run resolved from config (flag →
    # value), unless the command line already gave it. A bool value pins a
    # ``--flag/--no-flag`` pair: ``--flag`` for True, ``--no-flag`` for False.
    for flag, value in (pinned or {}).items():
        negated = f"--no-{flag[2:]}"
        if any(t in (flag, negated) or t.startswith(f"{flag}=") for t in argv):
            continue
        if isinstance(value, bool):
            argv.append(flag if value else negated)
        else:
            argv += [flag, value]
    return JobRequest(
        job_id=getattr(args, "job", None),
        wait=not getattr(args, "no_wait", False),
        deadline_s=_batch_deadline_seconds(args),
        command=command,
        argv=argv,
        cwd=os.getcwd(),
        log=_batch_log(args),
    )


def _reject_job_flags_without_batch(args: argparse.Namespace) -> None:
    if getattr(args, "no_wait", False) or getattr(args, "job", None):
        raise BatchJobInvalid("--no-wait and --job are batch job options; they need --batch")
    batch_label = getattr(args, "batch_label", None)
    if batch_label is not None:
        flag = "--batch-label" if batch_label else "--no-batch-label"
        raise BatchJobInvalid(f"{flag} is a batch option; it needs --batch")
    if getattr(args, "batch_deadline", None) is not None:
        raise BatchJobInvalid("--batch-deadline is a batch option; it needs --batch")


def _batch_deadline_seconds(args: argparse.Namespace) -> float | None:
    """``--batch-deadline`` in seconds, or None when not given
    (INVALID_ARGUMENT for a value that is not a positive duration)."""
    spec = getattr(args, "batch_deadline", None)
    if spec is None:
        return None
    from dgml_core.batch import parse_duration

    try:
        return parse_duration(spec)
    except ValueError as exc:
        raise InvalidArgument(f"--batch-deadline: {exc}") from None


def _subcommands(parser: argparse.ArgumentParser) -> Mapping[str, argparse.ArgumentParser]:
    """*parser*'s subcommand parsers by name (empty for a leaf command)."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def _canonical_argv(argv: list[str]) -> list[str]:
    """*argv* with every abbreviated long option spelled in full.

    argparse accepts any unambiguous prefix (``--form`` for ``--format``,
    ``--no-w`` for ``--no-wait``), so the command line a batch job stores may
    carry one; anything that inspects the stored argv by token (resume's
    stripping of ``--workspace`` / ``--format`` / ``--job``, the ``--no-wait``
    short-circuit) must see the canonical flag. Each option is resolved
    against the parser of the (sub)command it follows, as argparse does;
    option values and positionals are kept as they are, and an unknown or
    ambiguous prefix is left for argparse to judge."""
    current = _build_parser()
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        i += 1
        if token == "--":
            out.extend(argv[i - 1 :])
            break
        if not token.startswith("-") or token == "-":
            # A positional: a subcommand word descends into its parser.
            current = _subcommands(current).get(token, current)
            out.append(token)
            continue
        name, eq, value = token.partition("=")
        options = current._option_string_actions
        action = options.get(name)
        if action is None and name.startswith("--"):
            matches = {options[o] for o in options if o.startswith(name) and o.startswith("--")}
            if len(matches) == 1:
                (action,) = matches
                name = next(o for o in action.option_strings if o.startswith(name))
        out.append(f"{name}{eq}{value}" if action is not None else token)
        if action is None or eq or action.nargs == 0:
            continue
        # Its value(s): one, or (for ?, *, +) what argparse would consume.
        if action.nargs in (None, 1) or isinstance(action.nargs, int):
            count = action.nargs if isinstance(action.nargs, int) else 1
            out.extend(argv[i : i + count])
            i += count
        else:
            while i < len(argv) and not argv[i].startswith("-"):
                out.append(argv[i])
                i += 1
                if action.nargs == "?":
                    break
    return out


def _strip_flag(argv: list[str], flag: str) -> list[str]:
    """*argv* without ``flag VALUE`` / ``flag=VALUE``."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == flag:
            skip = True
            continue
        if token.startswith(f"{flag}="):
            continue
        out.append(token)
    return out


class _ArgvRejected(Exception):
    """:func:`_parse_quietly`'s failure: argparse's one-line reason."""


def _parse_quietly(argv: list[str]) -> argparse.Namespace:
    """Parse *argv* with the real parser, raising :class:`_ArgvRejected` (the
    reason argparse gave) instead of printing usage text and exiting 2."""
    import io

    usage = io.StringIO()
    try:
        with contextlib.redirect_stderr(usage), contextlib.redirect_stdout(io.StringIO()):
            return _build_parser().parse_args(argv)
    except SystemExit:
        lines = [line for line in usage.getvalue().splitlines() if line.strip()]
        reason = lines[-1] if lines else "invalid arguments"
        _prog, sep, rest = reason.partition(": error: ")
        raise _ArgvRejected(rest if sep else reason) from None


def _batch_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """``dgml batch status|resume|list|cancel|delete|prune|unlock``: each is one
    call into :mod:`dgml_core.batch.jobs`, whose return value is the payload."""
    from dgml_core.batch import jobs

    sub = args.batch_command
    if sub == "list":
        _emit({"jobs": jobs.list_job_summaries(ws)}, fmt)
    elif sub == "prune":
        _emit(jobs.prune_jobs(ws, older_than_days=args.older_than), fmt)
    elif sub == "unlock":
        _emit(jobs.unlock_job(ws, args.job_id), fmt)
    elif sub == "resume":
        return _batch_resume(args, ws, fmt)
    elif sub == "status":
        _emit(jobs.job_status(ws, args.job_id), fmt)
    elif sub == "delete":
        _emit(jobs.delete_job(ws, args.job_id, force=args.force), fmt)
    elif sub == "cancel":
        _emit(jobs.cancel_job(ws, args.job_id), fmt)
    else:  # unreachable — argparse `required=True` rejects unknown subcommands
        raise AssertionError(f"unhandled batch subcommand: {sub}")
    return 0


def _batch_resume(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Re-run the job's command from the directory it was first run in (so
    relative paths in its argv resolve as they did), continuing the job.

    Short-circuited when re-running could only pause again: a ``--no-wait``
    job whose open provider batches are all still running (polled read-only,
    as ``batch status`` does) reports the same pending payload a pausing run
    prints, without re-running anything (see
    :func:`dgml_core.batch.jobs.resume_would_wait`)."""
    from dgml_core.batch import jobs

    waiting = jobs.resume_would_wait(ws, args.job_id)  # BATCH_JOB_NOT_FOUND
    if waiting is not None:
        _emit({"batch_job": waiting}, fmt)
        return 0
    manifest = jobs.BatchJobStore(ws, args.job_id).load()
    # Reopened the way this run opened it: a workspace held in the store of
    # workspaces by its id (its config may not be under its root at all, as in
    # MongoDB), any other by its absolute path.
    where = ws.workspaces_id or str(Path(ws.root).resolve())
    # Canonical first (a job stored by a library caller may carry
    # abbreviations), so the stripping below sees every spelling.
    argv = _canonical_argv(manifest.argv)
    argv = _strip_flag(_strip_flag(argv, "--job"), "--workspace")
    argv = _strip_flag(argv, "--format")
    prefix = ["--workspace", where, "--format", fmt]
    if getattr(args, "verbose", False) and "--verbose" not in argv:
        prefix.append("--verbose")
    replay = [*prefix, *argv, "--job", manifest.job_id]
    try:
        # A stored command line that no longer parses (a job created by a
        # library caller, or by a dgml version whose flags since changed) is
        # the job's problem, not a usage error: one structured envelope.
        _parse_quietly(replay)
    except _ArgvRejected as exc:
        raise BatchJobInvalid(
            f"batch job {manifest.job_id} cannot be resumed: its stored command line "
            f"no longer parses ({exc}): {shlex.join(manifest.argv)}"
        ) from None
    if manifest.cwd and not Path(manifest.cwd).is_dir():
        # Relative paths in the stored command line resolve against it; running
        # from here instead could read (or write) the wrong files.
        raise BatchJobInvalid(
            f"batch job {manifest.job_id} cannot be resumed: the directory it was first "
            f"run in ({manifest.cwd}) no longer exists, and relative paths in its command "
            "line resolve against it. Recreate that directory and resume again, or "
            f"`dgml batch cancel {manifest.job_id}` to give it up"
        )
    here = os.getcwd()
    if manifest.cwd:
        os.chdir(manifest.cwd)
    try:
        return main(replay)
    finally:
        os.chdir(here)


def _dispatch(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    cmd = args.command

    if cmd == "init":
        return _init_cmd(args, ws, fmt)

    if cmd == "workspace":
        return _workspace_cmd(args, ws, fmt)

    if cmd == "status":
        docsets = DocSetStore(ws).list_all()
        files = FileStore(ws).list_all()
        _emit(
            {
                "workspace": str(ws.root),
                "name": ws.display_name,
                "organization": ws.organization,
                # Where this workspace's config is. Reported here because `create` used to
                # be the only command that said, so anything wanting to edit an existing
                # workspace's config had to reconstruct the path — and it cannot be
                # reconstructed reliably: a listed workspace's config sits in the store,
                # which is not under the data root when `workspace_path` relocates it, and
                # is not a file at all on a networked backend (null, with config_location
                # naming it instead).
                "workspace_config_path": _workspace_config_file(ws),
                "config_location": ws.config_location,
                "docset_count": len(docsets),
                "file_count": len(files),
            },
            fmt,
        )
        return 0

    if cmd == "check":
        report = check_workspace(ws, retry_errors=args.retry_errors, record_usage=args.debug)
        _emit(report.to_json(), fmt)
        return 0 if report.ok else 2

    if cmd == "cluster":
        try:
            from dgml_core.clustering import clustering
        except ImportError as exc:
            raise MissingExtra(
                "The 'clustering' extra is not installed. Run: pip install dgml[clustering]",
                extra="clustering",
            ) from exc
        # `clustering` owns the `skipped` key and the skip-existing no-op
        # short-circuit (which avoids re-scanning the workspace). `config`
        # is passed through raw — it may be a preset name or a path.
        _emit(
            clustering(
                ws,
                skip_existing=getattr(args, "skip_existing", False),
                config=getattr(args, "config", None),
                mode=ClusterMode(getattr(args, "mode", ClusterMode.AUTO)),
                method=ClusterMethod(getattr(args, "method", ClusterMethod.AUTO)),
                small_corpus_threshold=getattr(args, "small_corpus_threshold", 8),
                write_stats=args.debug,
                record_usage=args.debug,
            ),
            fmt,
        )
        return 0

    if cmd == "docset":
        return _docset_cmd(args, ws, fmt)
    if cmd == "extraction":
        return _extraction_cmd(args, ws, fmt)
    if cmd == "file":
        return _file_cmd(args, ws, fmt)
    if cmd == "dgmlx":
        return _dgmlx_cmd(args, ws, fmt)
    if cmd == "node":
        return _node_cmd(args, ws, fmt)
    if cmd == "discover":
        return _discover_cmd(args, ws, fmt)
    if cmd == "batch":
        return _batch_cmd(args, ws, fmt)
    if cmd in ("chain", "wallet", "registry", "stake", "prove"):
        return _chain_cmd(args, ws, fmt)

    # Unreachable: the subparsers are `required=True`, so argparse rejects an
    # unknown command before dispatch. Assert the invariant rather than carry a
    # phantom error code in the public surface.
    raise AssertionError(f"unhandled command: {cmd}")


def _discover_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Discover XML element subtrees in a File's generated DGML XML."""
    from dgml_core.discovery import (
        SEMANTIC_FILTER_NAMES,
        classify_tags_with_llm,
        discover_subtrees,
        load_subtree_root,
    )

    filter_name: str = args.filter_name.title()
    samples: int = args.samples
    include_structural: bool = args.include_structural
    full: bool = args.full
    strip_attributes: bool = not full
    search: str | None = args.search
    search_content: str | None = args.search_content

    # Semantic filters need an LLM config; fall back to All if unavailable.
    semantic_map: dict[str, str] | None = None
    if filter_name in SEMANTIC_FILTER_NAMES:
        try:
            from dgml_core.discovery import compute_tag_metrics
            from dgml_core.generation import load_generation_config, resolve_generation_api_key
            from dgml_core.llm import LLMConfig

            gen_cfg = load_generation_config(ws)
            llm_cfg = LLMConfig(
                model=gen_cfg.model,
                api_key=resolve_generation_api_key(gen_cfg),
                api_base=gen_cfg.api_base,
            )
            root_for_tags = load_subtree_root(ws, args.file_id, args.docset_id)
            metrics = compute_tag_metrics(root_for_tags, include_structural=include_structural)
            tag_names = [m.name for m in metrics]
            semantic_map = classify_tags_with_llm(tag_names, llm_cfg)
        except Exception as exc:
            _log.warning(
                f"[dgml discover] semantic filter unavailable ({exc}), falling back to All"
            )
            filter_name = "All"

    root = load_subtree_root(ws, args.file_id, args.docset_id)
    tags = discover_subtrees(
        root,
        filter_name=filter_name,
        samples=samples,
        semantic_map=semantic_map,
        include_structural=include_structural,
        strip_attributes=strip_attributes,
    )

    # Apply --search and --search-content filters.
    if search:
        term = search.lower()
        tags = [t for t in tags if term in t.tag.lower()]
    if search_content:
        term_c = search_content.lower()
        tags = [t for t in tags if any(term_c in s.xml.lower() for s in t.samples)]

    _emit(
        {
            "file_id": args.file_id,
            "docset_id": args.docset_id,
            "filter": filter_name,
            "tag_count": len(tags),
            "tags": [t.to_json(full=full) for t in tags],
        },
        fmt,
    )
    return 0


def _chain_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Dispatch the on-chain attestation commands (gated behind dgml[chain])."""
    import importlib.util

    # Only a genuinely-absent extra is MISSING_EXTRA; a real import error inside
    # staking/dgml_chain (broken transitive dep, code bug) must surface as
    # INTERNAL_ERROR rather than be masked as "extra not installed".
    if importlib.util.find_spec("dgml_chain") is None:
        raise MissingExtra(
            "The 'chain' extra is not installed. Run: pip install dgml[chain]",
            extra="chain",
            distribution="dgml-chain",
        )
    from dgml_core import staking

    cmd = args.command
    cfg = args.chain_config

    if cmd == "chain":
        if args.chain_command == "list":
            _emit(staking.chain_list(ws, cfg), fmt)
        elif args.chain_command == "show":
            _emit(staking.chain_show(ws, args.name, cfg), fmt)
        elif args.chain_command == "add":
            _emit(
                staking.chain_add(
                    ws,
                    name=args.name,
                    rpc_url=args.rpc_url,
                    chain_id=args.chain_id,
                    anchor_address=args.anchor_address,
                    explorer=args.explorer,
                    native_token=args.native_token,
                    config_path=cfg,
                ),
                fmt,
            )
        else:  # remove
            _emit(staking.chain_remove(ws, args.name, cfg), fmt)
        return 0

    if cmd == "wallet":
        _emit(
            staking.wallet_status(
                ws,
                chain_name=args.chain_name,
                address=args.address,
                config_path=cfg,
                service=args.keychain_service,
                account=args.keychain_account,
            ),
            fmt,
        )
        return 0

    if cmd == "registry":
        if args.registry_command == "create":
            _emit(
                staking.registry_create(
                    ws,
                    chain_name=args.chain_name,
                    name=args.name,
                    description=args.description,
                    metadata=args.metadata,
                    from_address=args.from_address,
                    config_path=cfg,
                    dry_run=args.dry_run,
                    legacy=args.legacy,
                    service=args.keychain_service,
                    account=args.keychain_account,
                ),
                fmt,
            )
        else:  # list
            _emit(
                staking.registry_list(
                    ws, chain_name=args.chain_name, name=args.name, config_path=cfg
                ),
                fmt,
            )
        return 0

    if cmd == "stake":
        if args.stake_command == "file":
            payload = staking.stake_file(
                ws,
                file_id=args.file_id,
                docset_id=args.docset_id,
                chain_name=args.chain_name,
                registry=args.registry,
                from_address=args.from_address,
                output_dir=args.output_dir,
                config_path=cfg,
                dry_run=args.dry_run,
                legacy=args.legacy,
                unpacked=args.unpacked,
                service=args.keychain_service,
                account=args.keychain_account,
            )
        else:  # node
            payload = staking.stake_node(
                ws,
                file_id=args.file_id,
                docset_id=args.docset_id,
                leaf_index=args.leaf_index,
                xpath=args.xpath,
                chain_name=args.chain_name,
                registry=args.registry,
                from_address=args.from_address,
                output_dir=args.output_dir,
                config_path=cfg,
                dry_run=args.dry_run,
                legacy=args.legacy,
                service=args.keychain_service,
                account=args.keychain_account,
            )
        _emit(payload, fmt)
        return 0

    # prove
    prover = staking.prove_file if args.prove_command == "file" else staking.prove_node_record
    payload, valid = prover(
        ws,
        chain_name=args.chain_name,
        registry=args.registry,
        checksum=args.checksum,
        record_json=args.record_json,
        config_path=cfg,
    )
    _emit(payload, fmt)
    # Mirror `dgmlx verify` / `node prove`: 0 proven, 2 computed-but-mismatched.
    return 0 if valid else 2


def _node_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Export a node attestation payload, or prove one against the workspace."""
    from dgml_core.merkle import proof_from_json, proof_to_json
    from dgml_core.node_attestation import export_node, prove_node

    sub = args.node_command
    if sub == "export":
        att = export_node(
            ws,
            args.file_id,
            args.docset_id,
            leaf_index=args.leaf_index,
            xpath=args.xpath,
            child_path=args.child_path,
        )
        _emit(
            {
                "file_id": att.file_id,
                "docset_id": att.docset_id,
                "leaf_index": att.leaf_index,
                "leaf_count": att.leaf_count,
                "xpath": att.xpath,
                "node_hash": att.node_hash,
                "root_hash": att.root_hash,
                "proof": proof_to_json(att.proof),
                "node_xml": att.node_xml,
            },
            fmt,
        )
        return 0
    if sub == "prove":
        if args.proof_path == "-":
            payload = json.load(sys.stdin)
        else:
            try:
                payload = read_json(Path(args.proof_path))
            except OSError as exc:
                return _emit_error("INVALID_ARGUMENT", f"cannot read proof file: {exc}", fmt)
        if not isinstance(payload, dict) or "root_hash" not in payload or "proof" not in payload:
            return _emit_error(
                "INVALID_ARGUMENT",
                "proof payload must be a JSON object with 'root_hash' and 'proof' "
                "(the `node export` output)",
                fmt,
            )
        try:
            proof = proof_from_json(payload["proof"])
        except ValueError as exc:
            return _emit_error("INVALID_ARGUMENT", f"malformed proof: {exc}", fmt)
        result = prove_node(ws, args.file_id, args.docset_id, payload["root_hash"], proof)
        _emit(
            {
                "file_id": result.file_id,
                "docset_id": result.docset_id,
                "leaf_index": result.leaf_index,
                "xpath": result.xpath,
                "expected_root": result.expected_root,
                "expected_node_hash": result.expected_node_hash,
                "computed_node_hash": result.computed_node_hash,
                "valid": result.valid,
            },
            fmt,
        )
        # Mirror `dgmlx verify`: 0 proven, 2 computed-but-mismatched.
        return 0 if result.valid else 2

    raise AssertionError(f"unhandled node subcommand: {sub}")  # unreachable (required=True)


def _dgmlx_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Export a DGMLX bundle, or verify one against its attestation file."""
    from dgml_core.file_attestation import export_attestation, verify_bundle

    sub = args.dgmlx_command
    if sub == "export":
        attestation, attestation_path, archive_path = export_attestation(
            ws, args.file_id, args.output_dir, args.docset_id, unpacked=args.unpacked
        )
        payload: dict[str, Any] = {
            "file_id": attestation.file_id,
            "docset_id": attestation.docset_id,
            "output_dir": str(args.output_dir),
            "root": attestation.root,
            "slots": [a.slot_id for a in attestation.leaves],
        }
        # Exactly one output mode: the .dgmlx archive (default) or the loose
        # attestation file (--unpacked). Surface whichever was produced.
        if archive_path is not None:
            payload["dgmlx"] = str(archive_path)
        if attestation_path is not None:
            payload["attestation"] = str(attestation_path)
        _emit(payload, fmt)
        return 0
    if sub == "verify":
        result = verify_bundle(args.path)
        _emit(
            {
                "path": str(args.path),
                "file_id": result.file_id,
                "docset_id": result.docset_id,
                "valid": result.valid,
                "expected_root": result.expected_root,
                "computed_root": result.computed_root,
                "slots": list(result.slot_ids),
            },
            fmt,
        )
        # Mirror `check`: 0 when sound, 2 when the bundle verifies-but-fails
        # (tampered/altered artifact). Malformed bundles raise → exit 1.
        return 0 if result.valid else 2

    raise AssertionError(f"unhandled dgmlx subcommand: {sub}")  # unreachable (required=True)


def _add_extraction_subparsers(
    sub: argparse._SubParsersAction,  # type: ignore[type-arg]
    common: argparse.ArgumentParser,
) -> None:
    """Register the `extraction` command group.

    Schema-driven value extraction: generate or set a docset's extraction
    schema (RELAX NG Compact at rest, JSON Schema accepted on input), extract
    grounded values into a compact ``extracted.dgml.xml`` fragment, and read
    them back as values-shape JSON or raw DGML XML.
    """
    extraction = sub.add_parser(
        "extraction",
        parents=[common],
        help="Schema-driven value extraction (RNC schema → grounded DGML values).",
    ).add_subparsers(dest="extraction_command", required=True)

    ex_gen = extraction.add_parser(
        "generate-schema",
        parents=[common],
        help="Generate an extraction schema (RNC) from sample files via the configured LLM.",
    )
    ex_gen.add_argument("docset_id")
    ex_gen.add_argument(
        "--from-file",
        dest="from_files",
        action="append",
        default=None,
        help="File id to sample (repeatable). Defaults to every file in the DocSet.",
    )
    ex_gen.add_argument(
        "--schema-model",
        default=None,
        help=(
            "Override grounded.schema_model for this call (LiteLLM model string). "
            "It takes the configured [models] provider key for its provider, else "
            "LiteLLM's per-provider env var (e.g. ANTHROPIC_API_KEY); no other "
            "configured key or api_base carries over."
        ),
    )
    _add_batch_arguments(
        ex_gen,
        what="the schema-generation",
        extra=" Schema generation is one request, so batch mode is one round trip.",
    )

    ex_set = extraction.add_parser(
        "set-schema",
        parents=[common],
        help="Set the extraction schema from a file. Accepts .rnc or .json; stored as RNC.",
    )
    ex_set.add_argument("docset_id")
    ex_set.add_argument(
        "--schema-file",
        required=True,
        type=Path,
        help="Path to a RELAX NG Compact (.rnc) or JSON Schema (.json) document.",
    )

    ex_get_schema = extraction.add_parser(
        "get-schema",
        parents=[common],
        help="Show a DocSet's extraction schema.",
    )
    ex_get_schema.add_argument("docset_id")
    ex_get_schema.add_argument(
        "--schema-format",
        choices=["rnc", "json"],
        default="rnc",
        help="Representation to return: rnc (canonical, default) or json (JSON Schema projection).",
    )

    ex_set_guidance = extraction.add_parser(
        "set-guidance",
        parents=[common],
        help="Set docset-level extraction guidance (free-form text shown to the extraction LLM).",
    )
    ex_set_guidance.add_argument("docset_id")
    ex_set_guidance.add_argument(
        "--guidance-file",
        required=True,
        type=Path,
        help="Path to a markdown/plain-text file with domain rules for this document kind.",
    )

    ex_get_guidance = extraction.add_parser(
        "get-guidance",
        parents=[common],
        help="Show a DocSet's extraction guidance.",
    )
    ex_get_guidance.add_argument("docset_id")

    ex_extract = extraction.add_parser(
        "extract",
        parents=[common],
        help="Extract grounded values from a file against its DocSet schema.",
    )
    ex_extract.add_argument("docset_id")
    ex_extract.add_argument(
        "file_ids",
        nargs="*",
        metavar="file_id",
        help=(
            "File(s) to extract. One id alone (no --all) keeps the single-file "
            "output (plus a `batch` block under --batch). Several ids, or --all, "
            "return one result per file and a failed file is reported in its entry "
            "instead of aborting the run."
        ),
    )
    ex_extract.add_argument(
        "--all",
        action="store_true",
        help="Extract every file in the DocSet (instead of naming file ids).",
    )
    ex_extract.add_argument(
        "--values-model",
        default=None,
        help=(
            "Override grounded.values_model for this call (LiteLLM model string). "
            "It takes the configured [models] provider key for its provider, else "
            "LiteLLM's per-provider env var (e.g. ANTHROPIC_API_KEY); no other "
            "configured key or api_base carries over."
        ),
    )
    ex_extract.add_argument(
        "--values-effort",
        default=None,
        help=(
            "Override grounded.values_reasoning_effort for this call: none, minimal, "
            "low, medium, high, xhigh, or 'default' to send no reasoning effort."
        ),
    )
    ex_extract.add_argument(
        "--locations-effort",
        default=None,
        help=(
            "Override grounded.locations_reasoning_effort (location grounding on pages "
            "with OCR words) for this call: none, minimal, low, medium, high, xhigh, or "
            "'default' to send no reasoning effort."
        ),
    )
    _add_batch_arguments(ex_extract, what="extraction")

    ex_get_values = extraction.add_parser(
        "get-values",
        parents=[common],
        help="Return extracted values as JSON (default) or the raw DGML XML fragment.",
    )
    ex_get_values.add_argument("docset_id")
    ex_get_values.add_argument("file_id")
    ex_get_values.add_argument(
        "--as",
        dest="as_form",
        choices=["values", "xml"],
        default="values",
        help="values: values-shape JSON projection (default). xml: the stored DGML fragment.",
    )


def _coerce_schema_to_rnc(raw: str, path: Path, workspace_name: str, docset_name: str) -> str:
    """Normalize a user-supplied schema file to RNC text.

    Accepts both formats (the CLI contract): a ``.json`` file (or one whose
    content begins with ``{``) is parsed as a grounded_field JSON Schema and
    converted; anything else is treated as RNC and validated. RNC is the only
    on-disk form.
    """
    from dgml_core.errors import SchemaInvalid
    from dgml_core.extraction_schema import json_schema_to_rnc, validate_rnc

    suffix = path.suffix.lower()
    looks_json = suffix == ".json" or (suffix != ".rnc" and raw.lstrip().startswith("{"))
    if looks_json:
        try:
            schema = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaInvalid(f"schema file is not valid JSON: {exc}") from exc
        if not isinstance(schema, dict):
            raise SchemaInvalid("JSON schema must be a JSON object")
        return json_schema_to_rnc(schema, workspace=workspace_name, docset_name=docset_name)
    validate_rnc(raw)  # raises SchemaInvalid if outside the supported subset
    return raw


def _error_block(exc: BaseException) -> dict[str, str]:
    """A per-item ``{code, message}`` block: a DgmlError keeps its code, anything
    else is an INTERNAL_ERROR with the same one-line message the top-level
    envelope would carry."""
    if isinstance(exc, DgmlError):
        return {"code": exc.code, "message": str(exc)}
    return {"code": "INTERNAL_ERROR", "message": short_error_message(exc)}


def _extract_many(
    args: argparse.Namespace,
    ws: Workspace,
    fmt: str,
    *,
    config: GroundedConfig,
    file_ids: list[str],
    single: bool = False,
) -> int:
    """``extraction extract`` over several files (ids, or ``--all``), or over
    one file id under ``--batch`` (*single*: the single-file payload plus a
    ``batch`` block, and the single-file form's errors).

    Each file is extracted independently and reported in its own ``results``
    entry — a failure is that entry's ``error``, never the run's. Under
    ``--batch`` both LLM phases go through the provider's batch API
    (:func:`dgml_core.grounded.extract_values_many`) and the payload gains a
    ``batch`` block; the per-file entries are the same either way.

    Exits ``0`` once the run completes, unless EVERY file failed: then the
    payload is still emitted and the exit is ``2`` (as ``dgml check``: the run
    finished, the outcome did not), so a script can detect a total failure
    without parsing the summary. An empty run (``--all`` on an empty DocSet)
    exits ``0``.
    """
    from dgml_core.grounded import extract_values_many, values_batch_stages

    store = DocSetStore(ws)
    store.get(args.docset_id)  # raises DocSetNotFound
    if args.all:
        file_ids = store.list_files(args.docset_id)
    if len(set(file_ids)) != len(file_ids):
        raise InvalidArgument("file ids must be unique")

    if args.batch:
        # Checked before any file is read, and before the job opens, so a
        # rejected run leaves no job behind.
        _batch_preflight(values_batch_stages(config.values_model, config.values_reasoning_effort))
        _start_batch_job(args, ws, command="extraction extract")
    many = extract_values_many(
        ws,
        args.docset_id,
        file_ids,
        config=config,
        batch=args.batch,
        poll_interval_s=args.batch_poll_interval,
        log=_batch_log(args),
        write_stats=args.debug,
        record_usage=args.debug,
    )
    outcomes = many.outcomes
    batch_block = many.batch

    if single:
        # One file id under --batch: the single-file form's payload and errors,
        # plus the run's `batch` block.
        (fid,) = file_ids
        outcome = outcomes[fid]
        if isinstance(outcome, BaseException):
            raise outcome
        one: dict[str, Any] = {
            "docset_id": args.docset_id,
            "file_id": fid,
            "model": config.values_model,
            "mode": outcome.mode,
            "tool_calls": outcome.tool_calls,
            "field_count": len(outcome.values),
            "xml_key": outcome.xml_key,
        }
        if batch_block is not None:
            one["batch"] = batch_block
        _emit(one, fmt)
        return 0

    results: list[dict[str, Any]] = []
    for fid in file_ids:
        outcome = outcomes[fid]
        if isinstance(outcome, BaseException):
            results.append({"file_id": fid, "status": "failed", "error": _error_block(outcome)})
        else:
            results.append(
                {
                    "file_id": fid,
                    "status": "ok",
                    "mode": outcome.mode,
                    "tool_calls": outcome.tool_calls,
                    "field_count": len(outcome.values),
                    "xml_key": outcome.xml_key,
                }
            )
    payload: dict[str, Any] = {
        "docset_id": args.docset_id,
        "model": config.values_model,
        "summary": {
            "total": len(results),
            "ok": sum(r["status"] == "ok" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
        },
        "results": results,
    }
    if batch_block is not None:
        payload["batch"] = batch_block
    _emit(payload, fmt)
    summary = payload["summary"]
    return 2 if summary["total"] and summary["failed"] == summary["total"] else 0


def _extraction_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Dispatch the `extraction` command group."""
    from dgml_core.extraction import extract_file, extraction_config
    from dgml_core.extraction_schema import rnc_to_json_schema
    from dgml_core.grounded import (
        generate_schema,
        load_grounded_config,
        parse_reasoning_effort,
        schema_batch_request,
    )

    store = DocSetStore(ws)
    sub = args.extraction_command

    if sub == "generate-schema":
        ds = store.get(args.docset_id)  # raises DocSetNotFound
        config = load_grounded_config(ws, schema_model=args.schema_model or None)
        if args.batch:
            # Before any PDF is read: an unsupported provider is BATCH_UNAVAILABLE,
            # never a silent full-price call.
            _batch_preflight({"schema": schema_batch_request(config.schema_model)})
        else:
            _reject_job_flags_without_batch(args)
        file_ids = args.from_files or store.list_files(args.docset_id)
        if not file_ids:
            return _emit_error(
                "NO_FILES",
                f"docset '{args.docset_id}' has no files; pass --from-file or add files first",
                fmt,
            )
        batch_block: dict[str, Any] | None = None
        if args.batch:
            from dgml_core.grounded import generate_schema_batch, schema_batch_executor

            # Replay-safe for job mode: nothing is written before the one request.
            _start_batch_job(args, ws, command="extraction generate-schema")
            executor = schema_batch_executor(
                config, poll_interval_s=args.batch_poll_interval, log=_batch_log(args)
            )
            rnc = generate_schema_batch(
                ws,
                file_ids,
                config=config,
                docset_name=ds.name,
                executor=executor,
                record_usage=args.debug,
            )
            batch_block = {"provider": executor.backend.provider, **executor.stats.to_json()}
        else:
            rnc = generate_schema(
                ws, file_ids, config=config, docset_name=ds.name, record_usage=args.debug
            )
        store.set_schema(args.docset_id, rnc)
        payload: dict[str, Any] = {
            "docset_id": args.docset_id,
            "schema_format": "rnc",
            "schema": rnc,
            "from_file_ids": list(file_ids),
            "model": config.schema_model,
        }
        if batch_block is not None:
            payload["batch"] = batch_block
        _emit(payload, fmt)
        return 0

    if sub == "set-schema":
        ds = store.get(args.docset_id)  # raises DocSetNotFound
        raw = args.schema_file.read_text(encoding="utf-8")
        rnc = _coerce_schema_to_rnc(raw, args.schema_file, ws.organization, ds.name)
        store.set_schema(args.docset_id, rnc)  # validates the RNC subset
        _emit({"docset_id": args.docset_id, "schema_format": "rnc", "schema": rnc}, fmt)
        return 0

    if sub == "get-schema":
        rnc = store.get_schema(args.docset_id)  # raises SchemaNotFound
        if args.schema_format == "json":
            _emit(
                {
                    "docset_id": args.docset_id,
                    "schema_format": "json",
                    "schema": rnc_to_json_schema(rnc),
                },
                fmt,
            )
        else:
            _emit({"docset_id": args.docset_id, "schema_format": "rnc", "schema": rnc}, fmt)
        return 0

    if sub == "set-guidance":
        store.get(args.docset_id)  # raises DocSetNotFound
        guidance = args.guidance_file.read_text(encoding="utf-8")
        store.set_guidance(args.docset_id, guidance)
        _emit({"docset_id": args.docset_id, "guidance": guidance}, fmt)
        return 0

    if sub == "get-guidance":
        guidance = store.get_guidance(args.docset_id)  # raises GuidanceNotFound
        _emit({"docset_id": args.docset_id, "guidance": guidance}, fmt)
        return 0

    if sub == "extract":
        extract_ids: list[str] = list(args.file_ids)
        if args.all and extract_ids:
            raise InvalidArgument("pass file ids or --all, not both")
        if not args.all and not extract_ids:
            raise InvalidArgument("name at least one file id, or pass --all")
        if not args.batch:
            _reject_job_flags_without_batch(args)
        if args.values_effort is not None:
            # Validated here too so the error names the flag, not the library kwarg.
            parse_reasoning_effort(args.values_effort, source="--values-effort")
        if args.locations_effort is not None:
            parse_reasoning_effort(args.locations_effort, source="--locations-effort")
        if len(extract_ids) == 1 and not args.all and not args.batch:
            # The original single-file form: output and errors unchanged.
            result = extract_file(
                ws,
                args.docset_id,
                extract_ids[0],
                values_model=args.values_model,
                values_effort=args.values_effort,
                locations_effort=args.locations_effort,
                write_stats=args.debug,
                record_usage=args.debug,
            )
            _emit(
                {
                    "docset_id": args.docset_id,
                    "file_id": extract_ids[0],
                    "model": result.model,
                    "mode": result.mode,
                    "tool_calls": result.tool_calls,
                    "field_count": len(result.values),
                    "xml_key": result.xml_key,
                },
                fmt,
            )
            return 0
        return _extract_many(
            args,
            ws,
            fmt,
            config=extraction_config(
                ws,
                values_model=args.values_model,
                values_effort=args.values_effort,
                locations_effort=args.locations_effort,
            ),
            file_ids=extract_ids,
            single=len(extract_ids) == 1 and not args.all,
        )

    if sub == "get-values":
        from dgml_core.dgml_output import read_dgml
        from dgml_core.extraction import get_extracted_values

        # Extracted values live as a dg:extraction element inside the pair's
        # core <stem>.dgml.xml; the library raises FILE_NOT_FOUND (unassigned),
        # DGML_NOT_FOUND (nothing generated or extracted yet) and VALUES_NOT_FOUND
        # (a tree with no extraction) through the usual envelope.
        if args.as_form == "xml":
            from dgml_core.extraction_xml import has_extraction

            doc = read_dgml(ws, args.docset_id, args.file_id)
            if not has_extraction(doc.xml):
                raise ValuesNotFound(
                    f"no extracted values for file '{args.file_id}' in docset "
                    f"'{args.docset_id}'; run 'dgml extraction extract' first"
                )
            _emit(
                {
                    "docset_id": args.docset_id,
                    "file_id": args.file_id,
                    "format": "xml",
                    "xml": doc.xml.decode("utf-8"),
                },
                fmt,
            )
        else:
            _emit(
                {
                    "docset_id": args.docset_id,
                    "file_id": args.file_id,
                    "format": "values",
                    "values": get_extracted_values(ws, args.docset_id, args.file_id),
                },
                fmt,
            )
        return 0

    raise AssertionError(f"unhandled extraction command: {sub}")


def _add_generate_subparser(
    docset_subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    common: argparse.ArgumentParser,
) -> None:
    """Register the `docset generate` subcommand."""
    gen = docset_subparsers.add_parser(
        "generate",
        parents=[common],
        help=(
            "Convert every file in a DocSet to DGML XML (typed-block pipeline; "
            "base install), then ground each <stem>.dgml.xml in place with "
            "dg:origin bounding-box attributes."
        ),
    )
    gen.add_argument("docset_id", help="ID of the DocSet whose files will be converted.")
    # Model selection. The default source is the workspace's 'generation' config
    # section, so the model stays one visible, deliberate choice per workspace
    # (see load_generation_config). These flags let a run name an explicit model
    # config without hand-editing config.json; the effective models and their
    # source are echoed into the JSON output's `models` block so the choice
    # remains visible/recorded. Mirrors `dgml cluster --config PRESET|PATH`.
    gen.add_argument(
        "--generation-config",
        dest="generation_config",
        metavar="PROFILE|PATH",
        default=None,
        help=(
            "Model config for this run. Either a bundled profile name "
            "(fast | balanced | quality) or a path to a standalone config JSON "
            "(same shape as the 'generation' section of <workspace>/config.json — "
            "'model', 'label_model', optional 'api_key'/'api_key_env'/'api_base'). "
            "Replaces the workspace config's generation section for this run. "
            "Defaults to the workspace config."
        ),
    )
    gen.add_argument(
        "--model",
        dest="model",
        default=None,
        help=(
            "Override the per-page transcription model for this run (e.g. "
            "'anthropic/claude-haiku-5-5'). Layers on top of --generation-config / "
            "the workspace config. It takes the configured [models] provider key "
            "for its provider, else LiteLLM's per-provider env var (e.g. "
            "ANTHROPIC_API_KEY); no other configured key or api_base carries over."
        ),
    )
    gen.add_argument(
        "--label-model",
        dest="label_model",
        default=None,
        help=(
            "Override the batch-wide semantic-labeling model for this run (e.g. "
            "'anthropic/claude-sonnet-4-6'). Layers on top of --generation-config / "
            "the workspace config. It takes the configured [models] provider key "
            "for its provider, else LiteLLM's per-provider env var (e.g. "
            "ANTHROPIC_API_KEY); no other configured key or api_base carries over."
        ),
    )
    # These four default to None: generate_docset then uses ConvertOptions'
    # own values, so the numbers are written down once, in the library.
    gen.add_argument(
        "--window-size", type=int, default=None, help="Pages per transcription window."
    )
    gen.add_argument("--temperature", type=float, default=None)
    gen.add_argument("--max-tokens", type=int, default=None)
    gen.add_argument(
        "--transcription",
        choices=[m.value for m in TranscriptionMode],
        default=None,
        help=(
            "How Pass A is asked for a window. 'text': the model copies the text "
            "into typed block lines. 'lines': the model assigns structure to the "
            "numbered lines of the page's text layer and the text is copied from "
            "the page (a tenth of the output tokens, nothing to hallucinate); "
            "pages without a text layer fall back to 'text'. Overrides "
            "[generation] transcription, whose default is 'text'."
        ),
    )
    gen.add_argument(
        "--thinking",
        choices=[m.value for m in ThinkingMode],
        default=None,
        help=(
            "Anthropic extended thinking for both generation passes. Overrides "
            "[generation] thinking, whose default is 'disabled'. Omitting the "
            "field on the wire is NOT the same as turning thinking off: Claude "
            "4.6+/5 models reason adaptively unless told not to. Claude Sonnet "
            "5.5 rejects 'disabled'; 'between_tools' is its off switch (thinking "
            "only between tool calls, so none for these text replies). Ignored "
            "for non-Anthropic models."
        ),
    )
    gen.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip word-coverage metrics.",
    )
    gen.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for per-window debug snapshots (transcription + labeling).",
    )
    gen.add_argument(
        "--max-parallel-calls",
        type=int,
        default=None,
        help=(
            "Max documents transcribed concurrently. Windows within a document "
            "stay serial. Set to 1 to disable parallelism. Tune to your provider's "
            "RPM tier."
        ),
    )
    gen.add_argument(
        "--schema-path",
        type=Path,
        default=None,
        help=(
            "Tag schema to label against. Four forms, detected by content: a plain "
            "newline-delimited list of tag names (blank lines and `#` comments ignored); "
            "a JSON {name: one-line description} object (RECOMMENDED — descriptions are "
            "what the model matches content against); an exported docsets/<id>/schema.json "
            "(a `tags` map of name -> {role, kind, examples, parent_role}); or its RELAX NG "
            "Compact render docsets/<id>/full-schema.rnc. Supplying a schema means the "
            "generated DGML uses THOSE tag names and no others: the planning pass is "
            "skipped and the vocabulary is closed. Content whose role has no matching tag "
            "is NOT dropped — it renders as dg:chunk with its text, structure, and "
            "dg:origin intact. To let labeling invent its own vocabulary instead, do not "
            "supply a schema."
        ),
    )
    gen.add_argument(
        "--extend-schema",
        action="store_true",
        help=(
            "Treat the supplied schema as a foundation rather than the whole "
            "vocabulary: labeling reuses your tag names wherever one fits, and may "
            "coin a new name for a recurring role your schema does not cover. Every "
            "coined name is reported per file under `added_concepts`, so it can be "
            "folded into the next revision of your schema. Requires a supplied "
            "schema (--schema-path, or one a previous run remembered); without this "
            "flag a supplied schema is used strictly and nothing else is emitted."
        ),
    )
    gen.add_argument(
        "--no-roster",
        action="store_true",
        help=(
            "Disable automatic roster reuse. By default an incremental generate "
            "seeds labeling with the docset's own authored-schema.json (if a previous run "
            "supplied one), else schema.json, else cache/concept_roster.json, so added "
            "documents stay tag-consistent; this labels them in isolation. A remembered "
            "authored schema closes the vocabulary the same way --schema-path does; a "
            "schema the pipeline derived itself only seeds."
        ),
    )
    gen.add_argument(
        "--no-semlinks",
        action="store_true",
        help=(
            "Skip the final semantic-link pass. By default each grounded "
            "<stem>.dgml.xml gets dg:itemprop/dg:href links (references, relative "
            "dates, derived values) added in place, using the labeling model."
        ),
    )
    gen.add_argument(
        "--no-semlink-cache",
        action="store_true",
        help=(
            "Always call the model for the semantic-link pass. By default the pass "
            "is cached on what the model actually reads (tag names and text, plus "
            "the labeling model and the link prompts), so a repeat run replays the "
            "cached links instead of paying for them again."
        ),
    )
    gen.add_argument(
        "--no-semlink-verify",
        action="store_true",
        help=(
            "Skip the second, skeptical pass that reviews each proposed link. "
            "Halves the model calls the link pass makes and cuts its wall-clock "
            "time by about 60%%, and keeps roughly twice as many links — including "
            "the weaker ones the review would have dropped."
        ),
    )
    _add_batch_arguments(
        gen,
        what="the generation pipeline's model",
        optional=True,
        extra=(
            " Batches transcription, roster planning, concept descriptions, the "
            "semantic-link pass and (OCR files) the image-style pass. Labeling "
            "batches every document at once under a closed --schema-path vocabulary, "
            "otherwise one document at a time in the synchronous order (identical "
            "output; --no-batch-label labels synchronously instead). Overrides "
            "[generation] batch in the config; --no-batch forces synchronous."
        ),
    )
    gen.add_argument(
        "--batch-label",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "--no-batch-label: under --batch, label with ordinary synchronous calls "
            "(everything else still batches). Batch labeling is the default: all "
            "documents at once under a closed --schema-path vocabulary, otherwise "
            "one document at a time in the synchronous order (identical output at "
            "batch price, about one batch round trip per document — pair it with "
            "--no-wait and `dgml batch resume`); --no-batch-label trades that price "
            "for latency. Overrides [generation] batch_label (default true). "
            "Requires batch mode."
        ),
    )


#: `docset run` options handed through, verbatim, to the step that owns them:
#: (flag, argparse dest, takes a value, step). Declared once so the run parser
#: and the argv each step is parsed from cannot drift apart (a test pins this
#: table to the step parsers); every value is re-parsed by the step's own
#: parser, so types and defaults stay the step's. A valueless flag is a
#: ``store_true`` switch, unless it is in :data:`_RUN_BOOLEAN_OPTIONAL` (a
#: ``--flag/--no-flag`` pair whose unset value is None and is not passed on).
_RUN_PASSTHROUGH: tuple[tuple[str, str, bool, str], ...] = (
    ("--generation-config", "generation_config", True, "generate"),
    ("--model", "model", True, "generate"),
    ("--label-model", "label_model", True, "generate"),
    ("--window-size", "window_size", True, "generate"),
    ("--temperature", "temperature", True, "generate"),
    ("--max-tokens", "max_tokens", True, "generate"),
    ("--transcription", "transcription", True, "generate"),
    ("--thinking", "thinking", True, "generate"),
    ("--no-coverage", "no_coverage", False, "generate"),
    ("--cache-dir", "cache_dir", True, "generate"),
    ("--max-parallel-calls", "max_parallel_calls", True, "generate"),
    ("--schema-path", "schema_path", True, "generate"),
    ("--extend-schema", "extend_schema", False, "generate"),
    ("--no-roster", "no_roster", False, "generate"),
    ("--no-semlinks", "no_semlinks", False, "generate"),
    ("--no-semlink-cache", "no_semlink_cache", False, "generate"),
    ("--no-semlink-verify", "no_semlink_verify", False, "generate"),
    ("--batch-label", "batch_label", False, "generate"),
    ("--schema-model", "schema_model", True, "schema"),
    ("--values-model", "values_model", True, "extract"),
    ("--values-effort", "values_effort", True, "extract"),
    ("--locations-effort", "locations_effort", True, "extract"),
)
_RUN_BOOLEAN_OPTIONAL: frozenset[str] = frozenset({"--batch-label"})

_RUN_STEP_COMMANDS = {
    "generate": "`docset generate`",
    "schema": "`extraction generate-schema`",
    "extract": "`extraction extract`",
}


def _add_run_subparser(
    docset_subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    common: argparse.ArgumentParser,
) -> None:
    """Register `docset run`: the whole docset pipeline in one command."""
    run = docset_subparsers.add_parser(
        "run",
        parents=[common],
        help=(
            "Run the whole docset pipeline: generate the extraction schema (when "
            "--schema-from is given and the docset has none), `docset generate`, "
            "then `extraction extract --all`. With --batch every step uses the "
            "provider's batch API under one batch job."
        ),
    )
    run.add_argument("docset_id", help="ID of the DocSet to run the pipeline over.")
    run.add_argument(
        "--schema-from",
        dest="schema_from",
        action="append",
        default=None,
        metavar="FILE_ID",
        help=(
            "Sample file for generating the extraction schema, repeatable. Used only "
            "when the docset has no extraction schema yet; without it, and with no "
            "schema, extraction is skipped (the payload says why)."
        ),
    )
    run.add_argument("--no-generate", action="store_true", help="Skip the `docset generate` step.")
    run.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction (and so schema generation).",
    )
    for flag, dest, takes_value, step in _RUN_PASSTHROUGH:
        what = f"Passed to {_RUN_STEP_COMMANDS[step]}."
        if takes_value:
            run.add_argument(flag, dest=dest, default=None, help=what)
        elif flag in _RUN_BOOLEAN_OPTIONAL:
            run.add_argument(
                flag, dest=dest, action=argparse.BooleanOptionalAction, default=None, help=what
            )
        else:
            run.add_argument(flag, dest=dest, action="store_true", help=what)
    _add_batch_arguments(
        run,
        what="every step's model",
        optional=True,
        extra=(
            " The whole pipeline is ONE batch job: with --no-wait, `dgml batch resume "
            "<job_id>` advances it step by step. A model with no batch backend is "
            "rejected before any step runs (BATCH_UNAVAILABLE, naming the stage). "
            "Defaults to [generation] batch; --no-batch forces synchronous."
        ),
    )


def _docset_generate_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Convert every file in a DocSet to DGML XML via the typed-block pipeline.

    A thin wrapper now: the whole orchestration — transcription, labeling,
    rendering, in-place grounding, semantic links, coverage, schema artifacts —
    is :func:`dgml_core.generate_docset`, which this maps the parsed flags
    onto. Progress is diagnostic, not part of the JSON contract: stdout stays a
    single JSON object (the report), and the library logs progress at INFO
    under ``dgml_core.*``, which `_configure_logging` shows on stderr only
    under ``--verbose``.

    Batch mode (--batch / --no-batch, else [generation] batch) is the CLI's to
    turn on; the library does the rest as it does for any caller: it
    pre-flights the stages it will batch (BATCH_UNAVAILABLE before any work,
    never a silent full-price fallback), then opens the run's batch job from
    the `JobRequest` built here and closes it however the run ends. A step of
    `docset run` passes the run's own session instead.
    """
    from dgml_core.generation import (
        BatchOptions,
        load_generation_batch,
        load_generation_batch_label,
    )

    batch_opts: BatchOptions | None = None
    job: Any = None
    batch_enabled = args.batch if args.batch is not None else load_generation_batch(ws)
    if batch_enabled:
        batch_label = (
            args.batch_label if args.batch_label is not None else load_generation_batch_label(ws)
        )
        batch_opts = BatchOptions(
            poll_interval_s=args.batch_poll_interval, label=batch_label, log=_batch_log(args)
        )
        if getattr(args, "run_step", False):
            job = _active_batch_job()  # one job spans every step of `docset run`
        if job is None:
            job = _job_request(
                args, command="docset generate", pinned={"--batch-label": batch_label}
            )
            # The run's output waits for the job's close (one outcome per run;
            # a job left behind is named in the payload).
            _begin_job_capture()
    else:
        _reject_job_flags_without_batch(args)

    try:
        report = _generate(args, ws, batch=batch_opts, job=job)
    finally:
        if isinstance(job, JobRequest):
            setattr(args, _CLOSED_JOB_ATTR, job.session)
    _emit(report.to_json(), fmt)
    return 0


def _generate(args: argparse.Namespace, ws: Workspace, *, batch: Any, job: Any) -> Any:
    """`generate_docset` with the parsed flags mapped onto its parameters."""
    return generate_docset(
        ws,
        args.docset_id,
        generation_config=args.generation_config,
        model=args.model,
        label_model=args.label_model,
        window_size=args.window_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=None if args.thinking is None else ThinkingMode(args.thinking),
        transcription=(
            None if args.transcription is None else TranscriptionMode(args.transcription)
        ),
        max_parallel_docs=args.max_parallel_calls,
        cache_dir=args.cache_dir,
        schema_path=args.schema_path,
        extend_schema=args.extend_schema,
        reuse_roster=not args.no_roster,
        coverage=not args.no_coverage,
        semlinks=not args.no_semlinks,
        semlink_cache=not args.no_semlink_cache,
        semlink_verify=not args.no_semlink_verify,
        debug=args.debug,
        batch=batch,
        job=job,
    )


#: Job state recording whether a `docset run` job had to generate the
#: extraction schema when it STARTED (the schema step writes the schema, so a
#: resumed run must not re-decide from the store — it replays the step instead).
_RUN_SCHEMA_NEEDED_STATE = "run.schema_needed"

#: Counters summed across steps for `docset run`'s aggregate `batch` block.
_RUN_BATCH_COUNTERS = (
    "waves",
    "batches",
    "requests",
    "batch_ok",
    "sync_fallbacks",
    "resubmitted",
    "failed",
)


def _run_step_argv(args: argparse.Namespace, step: str) -> list[str]:
    """The options `docset run` hands to *step*, as command-line tokens."""
    out: list[str] = []
    for flag, dest, takes_value, owner in _RUN_PASSTHROUGH:
        if owner != step:
            continue
        value = getattr(args, dest)
        if takes_value and value is not None:
            out += [flag, str(value)]
        elif flag in _RUN_BOOLEAN_OPTIONAL and value is not None:
            out.append(flag if value else f"--no-{flag[2:]}")
        elif not takes_value and value:
            out.append(flag)
    return out


def _run_step_argvs(args: argparse.Namespace, *, batch: bool) -> dict[str, list[str]]:
    """Every step's command line, as `docset run` runs it (the schema step's
    only when there are ``--schema-from`` files to generate it from)."""
    batch_tokens = ["--batch", "--batch-poll-interval", str(args.batch_poll_interval)]
    argvs: dict[str, list[str]] = {}
    if args.schema_from:
        argv = ["extraction", "generate-schema", args.docset_id]
        for fid in args.schema_from:
            argv += ["--from-file", fid]
        argv += _run_step_argv(args, "schema")
        argvs["schema"] = argv + (batch_tokens if batch else [])
    argv = ["docset", "generate", args.docset_id, *_run_step_argv(args, "generate")]
    # --no-batch: a sync run stays sync whatever [generation] batch says.
    argvs["generate"] = argv + (batch_tokens if batch else ["--no-batch"])
    argv = ["extraction", "extract", args.docset_id, "--all", *_run_step_argv(args, "extract")]
    argvs["extract"] = argv + (batch_tokens if batch else [])
    return argvs


def _parse_run_step(step: str, argv: list[str]) -> argparse.Namespace:
    """Parse one `docset run` step's command line with the step's own parser.

    A bad pass-through value (``--window-size abc``) raises INVALID_ARGUMENT
    naming the step and the flag, instead of argparse's usage text and exit 2."""
    try:
        return _parse_quietly(argv)
    except _ArgvRejected as exc:
        raise InvalidArgument(f"docset run: invalid option for step '{step}': {exc}") from None


def _sum_cost_fields(parts: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    """Add up blocks' cost fields; ``null`` if any block's is unknown."""
    blocks = list(parts)
    if any(b.get("cost_usd") is None for b in blocks):
        return {"cost_usd": None, "standard_cost_usd": None, "saved_usd": None}
    cost = sum(float(b["cost_usd"]) for b in blocks)
    standard = sum(float(b["standard_cost_usd"]) for b in blocks)
    return {
        "cost_usd": round(cost, 6),
        "standard_cost_usd": round(standard, 6),
        "saved_usd": round(standard - cost, 6),
    }


def _run_step_batch_stats(step: str, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """One step's batch counters (``generate`` sums its stages), or None when
    the step ran no batch."""
    block = payload.get("batch")
    if not isinstance(block, Mapping):
        return None
    parts: list[Mapping[str, Any]]
    if step == "generate":
        stages = block.get("stages") or {}
        parts = [v for v in stages.values() if isinstance(v, Mapping) and "waves" in v]
    else:
        parts = [block]
    out: dict[str, Any] = {
        k: sum(int(p.get(k, 0) or 0) for p in parts) for k in _RUN_BATCH_COUNTERS
    }
    out.update(_sum_cost_fields(parts))
    return out


def _docset_run_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """``docset run``: extraction schema (if needed), ``docset generate``, then
    ``extraction extract --all`` — each step is the standalone command's own
    handler, run unchanged, its payload nested under ``steps``.

    Under ``--batch`` the whole run is ONE batch job: every step's executor
    replays from and records into it, so ``--no-wait`` pauses at whichever
    step first has a wave in flight and ``dgml batch resume`` re-runs the
    pipeline, replaying completed steps' responses from the job at no cost.
    Which steps a job runs is decided when it starts and replayed after, so a
    step's own writes (the schema it stores, the DGML generate writes) never
    turn a resumed run into a different one."""
    from dgml_core.generation import load_generation_batch, resolve_generation_config
    from dgml_core.grounded import (
        load_grounded_config,
        parse_reasoning_effort,
        schema_batch_request,
        values_batch_stages,
    )

    ds_store = DocSetStore(ws)
    ds_store.get(args.docset_id)  # raises DocSetNotFound
    batch = args.batch if args.batch is not None else load_generation_batch(ws)
    if not batch:
        _reject_job_flags_without_batch(args)

    want_extract = not args.no_extract
    has_schema = ds_store.has_schema(args.docset_id)
    schema_needed = want_extract and not has_schema

    prefix = ["--format", "json"]
    if args.verbose:
        prefix.append("--verbose")
    if args.debug:
        prefix.append("--debug")
    # Every step that may run is parsed now, before any step (or batch job)
    # starts: a bad pass-through value fails the run up front as
    # INVALID_ARGUMENT, not after an earlier step already paid.
    step_argvs = _run_step_argvs(args, batch=batch)
    may_run = {"schema": want_extract, "generate": not args.no_generate, "extract": want_extract}
    for name, argv in step_argvs.items():
        if may_run[name]:
            _parse_run_step(name, prefix + argv)
    # Their values are only checked by the extract step's handler, which runs
    # last: refuse a bad one (GROUNDED_CONFIG_INVALID) before generate pays.
    values_effort = (
        parse_reasoning_effort(args.values_effort, source="--values-effort")
        if want_extract and args.values_effort is not None
        else None
    )
    locations_effort = (
        parse_reasoning_effort(args.locations_effort, source="--locations-effort")
        if want_extract and args.locations_effort is not None
        else None
    )

    # Only an extraction that will actually run needs the grounded config, and
    # it is resolved now in either mode: a missing or invalid one
    # (GROUNDED_CONFIG_MISSING / _INVALID) fails the run before generate pays.
    # The flag models go in as the loader's overrides, as the steps pass them:
    # they satisfy resolution and carry their own credentials.
    extracting = want_extract and (has_schema or bool(args.schema_from))
    grounded = (
        load_grounded_config(
            ws,
            schema_model=args.schema_model or None,
            values_model=args.values_model or None,
        )
        if extracting
        else None
    )

    session = None
    if batch:
        # Every model a planned step will send to a batch, checked before any
        # work: the first offending stage is named (BATCH_UNAVAILABLE).
        from dataclasses import replace

        stages: dict[str, Any] = {}
        pinned: dict[str, str | bool] = {}  # resolved from config, recorded with the job
        if grounded is not None and args.values_effort is not None:
            grounded = replace(grounded, values_reasoning_effort=values_effort)
        if grounded is not None and args.locations_effort is not None:
            grounded = replace(grounded, locations_reasoning_effort=locations_effort)
        # A resumed job may already have written the schema it set out to
        # generate, so check the schema model whenever the run could make one.
        if grounded is not None and args.schema_from and (schema_needed or args.job):
            stages["schema"] = schema_batch_request(grounded.schema_model)
        if not args.no_generate:
            gen_cfg, _source = resolve_generation_config(
                ws, config=args.generation_config, model=args.model, label_model=args.label_model
            )
            from dgml_core.style_config import load_style_config

            # The stages `docset generate` batches, named as its step.
            generate_stages = batch_stage_models(
                FileStore(ws),
                ds_store.list_files(args.docset_id),
                gen_cfg,
                load_style_config(ws),
                semlinks=not args.no_semlinks,
            )
            stages.update({f"generate.{name}": model for name, model in generate_stages.items()})
            from dgml_core.generation import load_generation_batch_label

            # As `docset generate` pins it: a resume labels the same way
            # whatever [generation] batch_label says by then.
            pinned["--batch-label"] = (
                args.batch_label
                if args.batch_label is not None
                else load_generation_batch_label(ws)
            )
        if grounded is not None and (has_schema or args.schema_from):
            stages.update(
                values_batch_stages(
                    grounded.values_model, grounded.values_reasoning_effort, stage="extract"
                )
            )
        _batch_preflight(stages)
        session = _start_batch_job(args, ws, command="docset run", pinned=pinned)
        schema_needed = bool(session.state.setdefault(_RUN_SCHEMA_NEEDED_STATE, schema_needed))
        session.persist()

    steps: dict[str, Any] = {}
    completed: list[str] = []  # steps that ran, in run order

    def run_step(
        name: str, argv: list[str], handler: Callable[[argparse.Namespace, Workspace, str], int]
    ) -> int:
        """Run one step's command handler; 0 with its payload recorded under
        ``steps[name]``, 2 likewise when the step ran but failed on every file
        (as its standalone command exits 2), else the step's error re-emitted
        naming the step."""
        if session is not None:
            from dgml_core.batch.jobs import RUN_STEP_STATE

            session.state[RUN_STEP_STATE] = name
            session.persist()
        sub_args = _parse_run_step(name, prefix + argv)
        sub_args.raw_argv = []
        sub_args.run_step = True  # reuse this run's batch job (see _start_batch_job)
        error: dict[str, Any] | None = None
        own: dict[str, Any] = {}  # details the raised error carries itself
        with _capturing() as captured:
            try:
                rc = handler(sub_args, ws, "json")
            except DgmlError as exc:
                rc, error = 1, {"code": exc.code, "message": str(exc)}
                own = _error_details(exc) or {}
            except Exception as exc:
                # Not a DgmlError (a bug, an unexpected provider error): still
                # INTERNAL_ERROR, but naming the step like any step failure.
                if _verbose_enabled(args):
                    import traceback

                    traceback.print_exc()
                rc, error = 1, {"code": "INTERNAL_ERROR", "message": short_error_message(exc)}
        if rc in (0, 2) and captured.payloads:
            steps[name] = captured.payloads[-1]
            completed.append(name)
            return rc
        if error is None:
            envelope = captured.errors[-1] if captured.errors else {}
            error = envelope.get("error") or {
                "code": "INTERNAL_ERROR",
                "message": f"step '{name}' produced no result",
            }
            own = dict(error.get("details") or {})  # an emitted error's own details
        skipped = [n for n, v in steps.items() if n not in completed]
        return _emit_error(
            str(error["code"]),
            f"docset run: step '{name}' failed: {error['message']}",
            fmt,
            details={**own, "step": name, "completed_steps": completed, "skipped_steps": skipped},
        )

    # A step that exits 2 ran but failed on every file: the run stops there and
    # reports its payload with exit 2 (the later steps are skipped, naming it).
    failed_all: str | None = None

    # 1. Extraction schema.
    if not want_extract:
        steps["schema"] = {"skipped": "--no-extract"}
    elif schema_needed and not args.schema_from:
        steps["schema"] = {
            "skipped": (
                "the docset has no extraction schema and no --schema-from sample files were given"
            )
        }
    elif schema_needed:
        rc = run_step("schema", step_argvs["schema"], _extraction_cmd)
        if rc not in (0, 2):
            return rc
        if rc == 2:
            failed_all = "schema"
    else:
        steps["schema"] = {"skipped": "the docset already has an extraction schema"}

    # 2. DGML generation.
    if failed_all is not None:
        steps["generate"] = {"skipped": f"step '{failed_all}' failed on every file"}
    elif args.no_generate:
        steps["generate"] = {"skipped": "--no-generate"}
    else:
        rc = run_step("generate", step_argvs["generate"], _docset_cmd)
        if rc not in (0, 2):
            return rc
        if rc == 2:
            failed_all = "generate"

    # 3. Extraction.
    if failed_all is not None:
        steps["extract"] = {"skipped": f"step '{failed_all}' failed on every file"}
    elif not want_extract:
        steps["extract"] = {"skipped": "--no-extract"}
    elif schema_needed and "skipped" in steps["schema"]:
        steps["extract"] = {"skipped": "the docset has no extraction schema"}
    else:
        rc = run_step("extract", step_argvs["extract"], _extraction_cmd)
        if rc not in (0, 2):
            return rc
        if rc == 2:
            failed_all = "extract"

    payload: dict[str, Any] = {"docset_id": args.docset_id, "steps": steps}
    if batch:
        per_step: dict[str, Any] = {}
        for name, step_payload in steps.items():
            stats = _run_step_batch_stats(name, step_payload)
            if stats is None:
                # A string reason, as every other `skipped` in a batch block.
                reason = step_payload.get("skipped")
                stats = {"skipped": str(reason) if reason else "the step ran no batch"}
            per_step[name] = stats
        ran = [v for v in per_step.values() if "skipped" not in v]
        totals = {k: sum(int(v.get(k, 0)) for v in ran) for k in _RUN_BATCH_COUNTERS}
        payload["batch"] = {"enabled": True, **totals, **_sum_cost_fields(ran), "steps": per_step}
        if session is not None:
            from dgml_core.batch.jobs import RUN_STEP_STATE

            session.state[RUN_STEP_STATE] = "completed"
    _emit(payload, fmt)
    # As `dgml check` and a multi-file `extraction extract`: the payload is
    # printed, but the outcome is not clean.
    return 2 if failed_all is not None else 0


def _docset_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    store = DocSetStore(ws)
    sub = args.docset_command
    if sub == "generate":
        return _docset_generate_cmd(args, ws, fmt)
    if sub == "run":
        return _docset_run_cmd(args, ws, fmt)
    if sub == "create":
        ds = store.create(
            name=args.name,
            description=args.description,
            key_questions=args.key_questions,
            docset_id=args.id,
        )
        _emit(ds.to_json(), fmt)
    elif sub == "list":
        _emit({"docsets": [d.to_json() for d in store.list_all()]}, fmt)
    elif sub == "show":
        _emit(store.get(args.docset_id).to_json(), fmt)
    elif sub == "update":
        ds = store.update(args.docset_id, name=args.name, description=args.description)
        _emit(ds.to_json(), fmt)
    elif sub == "delete":
        store.delete(args.docset_id)
        _emit({"deleted": args.docset_id}, fmt)
    elif sub == "add-file":
        # Assign, then auto-extract when the DocSet has an extraction schema
        # set. Extraction soft-fails into the payload's `extraction.error`;
        # the assignment itself always stands. No schema → no block.
        from dgml_core.extraction import add_file_and_extract

        extraction_block = add_file_and_extract(
            ws, args.docset_id, args.file_id, write_stats=args.debug, record_usage=args.debug
        )
        payload: dict[str, Any] = {
            "docset_id": args.docset_id,
            "file_id": args.file_id,
            "assigned": True,
        }
        if extraction_block is not None:
            payload["extraction"] = extraction_block
        _emit(payload, fmt)
    elif sub == "remove-file":
        store.remove_file(args.docset_id, args.file_id)
        _emit(
            {"docset_id": args.docset_id, "file_id": args.file_id, "assigned": False},
            fmt,
        )
    elif sub == "list-files":
        _emit(
            {"docset_id": args.docset_id, "file_ids": store.list_files(args.docset_id)},
            fmt,
        )
    else:  # unreachable — argparse `required=True` rejects unknown subcommands
        raise AssertionError(f"unhandled docset subcommand: {sub}")
    return 0


def _file_add_payload(result: AddFileResult) -> dict[str, Any]:
    """The standard ``dgml file add`` success payload for one File.

    Shared by the single-file path and each entry in a bulk run, so callers
    parse the same shape either way (part of the public CLI contract). The
    shape itself is :meth:`AddFileResult.to_json`.
    """
    return result.to_json()


def _ingestible_suffixes(ws: Workspace) -> frozenset[str]:
    """Suffixes a directory bulk-add will collect: ``.pdf`` always, plus the
    convertible source extensions whose format family has a converter
    configured in the workspace. Unconfigured source types are skipped (not
    gathered), so a folder of PDFs with stray Office docs doesn't produce a
    pile of per-file failures. Raises ``ConversionConfigInvalid`` on a
    malformed ``conversion`` config or a key that is not a format family, or
    a family whose converter cannot be resolved or does not handle the
    family's suffixes (a real error worth surfacing once, up front, rather
    than once per gathered file). Deciding what to gather needs the config, so
    this fires even for a folder of only PDFs — unlike a single ``.pdf`` add,
    which never reads the section."""
    configured = load_conversion_config(ws)
    extra: set[str] = set()
    for sfx, family in FAMILY_BY_SUFFIX.items():
        if family in configured:
            require_converter(sfx, configured)
            extra.add(sfx)
    return frozenset({".pdf"} | extra)


def _gather_pdfs(directory: Path, *, recursive: bool, suffixes: frozenset[str]) -> list[Path]:
    """Return the ingestible files under ``directory`` (case-insensitive).

    ``suffixes`` is the accepted extension set (see :func:`_ingestible_suffixes`).
    ``recursive`` descends into subdirectories; otherwise only the top level
    is scanned (matching the ``find -maxdepth 1`` variant the skill calls
    out). Hidden files and directories below ``directory`` (names starting with
    ``.``) and Office lock files (``~$…``) are skipped; hidden directories are
    pruned, not walked. Results are lex-sorted so a bulk run is deterministic.
    """
    if recursive:
        candidates: list[Path] = []
        for root, dirnames, filenames in os.walk(directory):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            candidates.extend(Path(root) / f for f in filenames)
    else:
        candidates = list(directory.iterdir())
    return sorted(
        p
        for p in candidates
        if not p.name.startswith((".", "~$")) and p.suffix.lower() in suffixes and p.is_file()
    )


def _require_existing_docsets(docsets: list[DocSet]) -> None:
    """Guard the ``--auto-classify existing`` / ``existing-forced`` precondition.

    Those modes can only route into existing DocSets, so in an empty workspace
    there is nothing to ask the LLM: every file would come back unassigned.
    Raising up front surfaces the misconfiguration instead of a run of
    ``"none"`` decisions (or, for ``existing-forced``, no possible answer).
    """
    if not docsets:
        raise NoExistingDocSets(
            "no DocSets to assign to; create one with `dgml docset create`, "
            f"or use `--auto-classify {ClassifyMode.EXISTING_OR_NEW}`"
        )


class FileAddStatus(StrEnum):
    """One file's outcome in a directory ``dgml file add`` (an entry's ``status``,
    and the matching ``summary`` count key).

    ``ADDED`` created a record cleanly; ``SKIPPED`` returned an existing one (a
    conflict under ``--on-conflict skip``/``replace``); ``SOFT_FAILED`` created a
    record but page counting, rendering, text extraction or conversion failed;
    ``HARD_FAILED`` raised, so no record was created.
    """

    ADDED = "added"
    SKIPPED = "skipped"
    SOFT_FAILED = "soft_failed"
    HARD_FAILED = "hard_failed"


def _file_add_bulk(args: argparse.Namespace, ws: Workspace, store: FileStore, fmt: str) -> int:
    """Add every PDF under a directory in one run, emitting a single envelope.

    Each file commits independently (same as adding them one at a time): a
    per-file failure is recorded in its entry and the run continues. The
    payload carries a ``summary`` count block plus a per-file ``results``
    array; every entry carries a ``status`` (``added`` / ``skipped`` /
    ``soft_failed`` / ``hard_failed``) matching the summary counts. With
    ``--auto-classify`` the summary also counts ``unassigned``: files this run
    classified that still have no DocSet (declined under ``existing``, or a
    soft-failed classification) — an overlay on the status counts, so callers
    can tell from the summary alone whether any file needs routing. The
    command exits 0 as long as the run completes — individual soft- or
    hard-failures are reported, not raised (partial success is the contract,
    matching ``dgml cluster``). Only a run-level abort (workspace not
    initialized, directory unreadable) surfaces as an error envelope.
    """
    directory: Path = args.path
    recursive: bool = args.recursive
    pdfs = _gather_pdfs(directory, recursive=recursive, suffixes=_ingestible_suffixes(ws))

    classify_mode = _classify_mode(args)
    config: ClassificationConfig | None = None
    docsets: list[DocSet] | None = None
    if classify_mode is not None:
        # Load the classification config once, up front: a missing/invalid
        # config is a hard failure that aborts the run before any file is
        # added, rather than recording the same error on every file.
        config = load_classification_config(ws)
        # Read existing DocSets once; auto_classify appends newly-created
        # ones so similar PDFs cluster within the run without re-scanning.
        docsets = DocSetStore(ws).list_all()
        if classify_mode != ClassifyMode.EXISTING_OR_NEW:
            # Checked here so the run aborts before any file is added rather
            # than on the first one.
            _require_existing_docsets(docsets)

    # --batch is validated (mode, batch backends) before any file is added.
    batch: BulkClassifyBatch | None = None
    job: Any = None
    if getattr(args, "batch", False):
        if classify_mode is None:
            # Classification (and the auto-extraction after it) is all a
            # directory add sends to a model that a batch can serve.
            raise InvalidArgument(
                "--batch on a directory add batches its classification: pass "
                "--auto-classify (any mode)"
            )
        assert config is not None and docsets is not None
        # The session first: the executors prepare_bulk_classify builds are
        # bound to it. A run rejected there leaves no job behind (a fresh job
        # that failed before any request removes itself on close).
        job = _start_batch_job(args, ws, command="file add")
        batch = prepare_bulk_classify(
            ws,
            config=config,
            docsets=docsets,
            poll_interval_s=args.batch_poll_interval,
            log=_batch_log(args),
        )
    else:
        _reject_job_flags_without_batch(args)

    on_conflict = ConflictPolicy(args.on_conflict)
    text_mode = TextMode(args.text_mode)

    counts = {str(status): 0 for status in FileAddStatus}
    entries: list[dict[str, Any]] = []
    # A resumed job never ingests again: every file is already in, and adding
    # it a second time would report it "skipped" and skip its classification.
    # The first run's ingest results are restored from the job instead.
    restored = job.state.get("file_add") if job is not None and job.resumed else None
    if restored is not None:
        assert batch is not None
        counts = dict(restored["counts"])
        entries = [dict(e) for e in restored["entries"]]
        store_files = FileStore(ws)
        for index, file_id, created in restored["pending"]:
            batch.pending.append(
                (
                    entries[index],
                    AddFileResult(record=store_files.get(file_id), created=bool(created)),
                )
            )
        pdfs = [Path(p) for p in restored["pdfs"]]
        if docsets is not None:
            # Classify against the DocSets the job started with: those its
            # earlier runs created for earlier files are added back, in order,
            # as their replayed decisions are applied (see classify_bulk_batch).
            ds_store = DocSetStore(ws)
            docsets[:] = [ds_store.get(docset_id) for docset_id in restored["docset_ids"]]
    for pdf in pdfs if restored is None else []:
        try:
            result = store.add(
                pdf,
                on_conflict=on_conflict,
                text_mode=text_mode,
                dpi=args.dpi,
                record_usage=args.debug,
            )
        except DgmlError as exc:
            counts[FileAddStatus.HARD_FAILED.value] += 1
            entries.append(
                {
                    "status": FileAddStatus.HARD_FAILED.value,
                    "path": str(pdf),
                    "error": {"code": exc.code, "message": str(exc)},
                }
            )
            continue

        if not result.created:
            status = FileAddStatus.SKIPPED
        elif (
            result.page_render_error
            or result.page_count_error
            or result.text_extraction_error
            or result.conversion_error
        ):
            status = FileAddStatus.SOFT_FAILED
        else:
            status = FileAddStatus.ADDED
        counts[str(status)] += 1

        entry: dict[str, Any] = {
            "status": str(status),
            "path": str(pdf),
            **_file_add_payload(result),
        }
        if batch is not None:
            # Classified together once every file is in (see classify_bulk_batch).
            batch.pending.append((entry, result))
        elif classify_mode is not None:
            entry["classification"] = run_auto_classify(
                ws,
                result,
                config=config,
                docsets=docsets,
                mode=classify_mode,
                write_stats=args.debug,
                record_usage=args.debug,
            )
        entries.append(entry)

    if job is not None and restored is None:
        # Everything before the first LLM call, kept so a resume skips it.
        assert batch is not None
        by_id = {id(entry): i for i, entry in enumerate(entries)}
        job.state["file_add"] = {
            "pdfs": [str(p) for p in pdfs],
            "counts": dict(counts),
            "entries": json.loads(json.dumps(entries)),
            "pending": [
                [by_id[id(entry)], res.record.id, res.created] for entry, res in batch.pending
            ],
            # The DocSets the classifications start from (their ids are in the
            # requests): a resume must not see the ones this job creates.
            "docset_ids": [ds.id for ds in docsets or []],
        }
        job.persist()

    batch_block: dict[str, Any] | None = None
    if batch is not None:
        assert config is not None and docsets is not None and classify_mode is not None
        batch_block = classify_bulk_batch(
            ws,
            batch,
            config=config,
            docsets=docsets,
            write_stats=args.debug,
            record_usage=args.debug,
            mode=classify_mode,
        )

    summary: dict[str, Any] = {"total": len(pdfs), **counts}
    if classify_mode is not None:
        # Counted once every file is classified (the batch path classifies
        # after the loop): files this run classified that have no DocSet.
        summary["unassigned"] = sum(
            1
            for e in entries
            if (block := e.get("classification")) is not None
            and block["performed"]
            and block["docset_id"] is None
        )
    payload: dict[str, Any] = {
        "directory": str(directory),
        "recursive": recursive,
        "summary": summary,
        "results": entries,
    }
    if batch_block is not None:
        payload["batch"] = batch_block
    _emit(payload, fmt)
    return 0


def _file_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    store = FileStore(ws)
    sub = args.file_command
    if sub == "add":
        if args.path.is_dir():
            # Checked here rather than in FileStore.add: "directory" is a
            # CLI-surface concept the store never sees (its is_file() check
            # would fail first, with a misleading "does not exist").
            if args.id is not None:
                raise InvalidArgument(
                    f"--id names one File and cannot be used when PATH is a directory "
                    f"({args.path}) — a bulk run adds many. Add the files one at a time "
                    f"to choose each id."
                )
            return _file_add_bulk(args, ws, store, fmt)
        if getattr(args, "batch", False):
            raise InvalidArgument(
                "--batch applies to a directory add (PATH is a directory): one file "
                "is one request, which a batch cannot make cheaper to wait for"
            )
        _reject_job_flags_without_batch(args)
        classify_mode = _classify_mode(args)
        config: ClassificationConfig | None = None
        docsets: list[DocSet] | None = None
        if classify_mode is not None and classify_mode != ClassifyMode.EXISTING_OR_NEW:
            # The existing-only modes' preconditions are checked *before*
            # ingesting, so a misconfigured request errors out without leaving
            # a stray file behind. Same order as the bulk path.
            config = load_classification_config(ws)
            docsets = DocSetStore(ws).list_all()
            _require_existing_docsets(docsets)
        result = store.add(
            args.path,
            file_id=args.id,
            on_conflict=ConflictPolicy(args.on_conflict),
            text_mode=TextMode(args.text_mode),
            dpi=args.dpi,
            record_usage=args.debug,
        )
        payload: dict[str, Any] = _file_add_payload(result)
        if classify_mode is not None:
            # In the default mode auto_classify loads the classification
            # config itself; a missing/invalid one raises straight through to
            # an error envelope.
            payload["classification"] = run_auto_classify(
                ws,
                result,
                config=config,
                docsets=docsets,
                mode=classify_mode,
                write_stats=args.debug,
                record_usage=args.debug,
            )
        _emit(payload, fmt)
    elif sub == "list":
        _emit({"files": [f.to_json() for f in store.list_all()]}, fmt)
    elif sub == "show":
        _emit(store.get(args.file_id).to_json(), fmt)
    elif sub == "delete":
        store.delete(args.file_id)
        _emit({"deleted": args.file_id}, fmt)
    else:  # unreachable — argparse `required=True` rejects unknown subcommands
        raise AssertionError(f"unhandled file subcommand: {sub}")
    return 0


def _classify_mode(args: argparse.Namespace) -> ClassifyMode | None:
    """The ``--auto-classify`` MODE, or ``None`` when the flag wasn't passed."""
    raw = getattr(args, "auto_classify", None)
    return None if raw is None else ClassifyMode(raw)


if __name__ == "__main__":
    sys.exit(main())
