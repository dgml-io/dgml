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
import json
import os
import sys
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from dgml_core.classification import (
    ClassificationConfig,
    classify_file,
    load_classification_config,
)
from dgml_core.consistency import check_workspace
from dgml_core.conversion import FAMILY_BY_SUFFIX, load_conversion_config
from dgml_core.docsets import DocSetStore
from dgml_core.errors import DgmlError, WorkspaceNotInitialized, short_error_message
from dgml_core.files import AddFileResult, ConflictPolicy, FileStore
from dgml_core.models import DocSet
from dgml_core.storage import Workspace, read_json
from dgml_core.text_extraction import TextMode

if TYPE_CHECKING:
    from dgml_core.generation.schema import Schema


def _emit(payload: dict[str, Any], fmt: str, stream: IO[str] | None = None) -> None:
    out = stream or sys.stdout
    if fmt == "json":
        json.dump(payload, out, indent=2, ensure_ascii=False)
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
_FORMAT_HELP = "Output format. Default 'json' for machine/agent consumption."
_VERBOSE_HELP = (
    "Emit informational diagnostics to stderr. Controls hybrid text-mode "
    "warnings (digital/OCR conflicts, OCR misses) and the per-page merge "
    "summary, and the `docset generate` pipeline's progress lines; default "
    "off so stderr stays reserved for error envelopes."
)
_DEBUG_HELP = (
    "Keep intermediate pipeline files in the workspace: the `docset generate` "
    "cache/ and coverage_report.json, the `docset ground` grounding_stats.json, "
    "and the `file extract` extraction_stats.json. Default off — only final "
    "files (DGML XML, page text/images, schemas, values, metadata) are kept."
)


def _add_global_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Declare the three global flags (`--workspace`/`--format`/`--verbose`) in
    one place. ``suppress=False`` (top-level parser) gives them their real
    defaults; ``suppress=True`` (the shared parent threaded into every
    subparser) uses ``SUPPRESS`` so an omitted flag after the subcommand leaves
    the namespace untouched, letting the top-level default stand instead of
    clobbering it. Declaring both from one function keeps the two positions'
    metadata (choices/help) from drifting on a public-contract surface."""
    parser.add_argument(
        "--workspace",
        type=Path,
        default=argparse.SUPPRESS if suppress else None,
        help=_WORKSPACE_HELP,
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
            "Create the shared local_config.json (config only; run once). Edit it, "
            "then `dgml workspace create`."
        ),
    )
    init_p.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Overwrite local_config.json from the bundled default template (pull the "
            "latest baseline / new knobs). Back up local_config.json first if needed."
        ),
    )

    workspace_sub = sub.add_parser(
        "workspace", parents=[common], help="Workspace lifecycle."
    ).add_subparsers(dest="workspace_command", required=True)
    ws_create = workspace_sub.add_parser(
        "create",
        parents=[common],
        help=(
            "Create a workspace (docsets/ + files/) and seed its config.json from the "
            "shared local_config.json (auto-created if `dgml init` was not run first)."
        ),
    )
    ws_create.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Directory to create the workspace in. Optional; when omitted the root "
            "resolves in the usual order (the global --workspace, then $DGML_HOME, then "
            "./dgml-workspace). A path given here overrides that."
        ),
    )
    ws_create.add_argument(
        "--organization",
        required=True,
        help=(
            "Organization name. Embedded in this workspace's docset namespace URIs "
            "(http://dgml.io/<organization>/<DocSetSlug>) — pick a stable identifier for "
            "your org, as changing it later shifts the namespaces of newly generated XML."
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
        "--force",
        action="store_true",
        help=(
            "Overwrite an existing config.json with the current local_config.json "
            "(re-sync edited shared config into this workspace)."
        ),
    )

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
        "JSON (same shape as the 'clustering' section of <workspace>/config.json "
        "— e.g. encoder_text, encoder_image, fusion, scenario). Replaces the "
        "workspace config's clustering section for this run. Defaults to the "
        "workspace config, or the bundled light preset when none is set.",
    )
    cluster_p.add_argument(
        "--mode",
        dest="mode",
        choices=("auto", "fresh", "incremental"),
        default="auto",
        help="Clustering mode. 'auto' (default) runs incremental clustering "
        "when the workspace already has DocSets (assign new files to existing "
        "clusters, open new clusters for the rest) and fresh clustering "
        "otherwise. 'fresh' always clusters from scratch; 'incremental' forces "
        "the incremental path and errors if no DocSets exist yet.",
    )
    cluster_p.add_argument(
        "--method",
        dest="method",
        choices=("auto", "embedding", "llm"),
        default="embedding",
        help="How documents are grouped, orthogonal to --mode. 'embedding' "
        "(default) uses the statistical encode → project → cluster pipeline — "
        "the right choice once a corpus is large enough for tf-idf / neighbor "
        "statistics to be meaningful. 'llm' sends every document's page images "
        "to the vision LLM in one call and lets it partition them — built for "
        "very small corpora where the embedding pipeline has too little signal. "
        "'auto' picks 'llm' when at most --small-corpus-threshold files are "
        "clusterable, else 'embedding'. Both 'llm' and 'auto' (when it routes "
        "to the LLM) need the same `classification` config as --auto-classify.",
    )
    cluster_p.add_argument(
        "--small-corpus-threshold",
        dest="small_corpus_threshold",
        type=int,
        metavar="N",
        # Keep in sync with dgml_core.clustering.SMALL_CORPUS_MAX_FILES (8).
        default=8,
        help="With --method auto, route corpora of at most N clusterable files "
        "to the LLM partitioner, and larger ones to the embedding pipeline "
        "(default 8). Ignored for --method embedding / llm.",
    )

    docset = sub.add_parser("docset", parents=[common], help="DocSet management.").add_subparsers(
        dest="docset_command", required=True
    )
    _add_generate_subparser(docset, common)
    ds_create = docset.add_parser("create", parents=[common], help="Create a new DocSet.")
    ds_create.add_argument("--name", required=True)
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
            "Path to a file (.pdf, or a convertible source .docx/.doc/.xlsx/.xls "
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
            "<workspace>/config.json (requires `pip install dgml[aws]` or "
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
        "--auto-classify",
        action="store_true",
        help=(
            "After adding, use the configured vision LLM to assign the file "
            "to a DocSet — assigning to an existing DocSet if one fits, "
            "otherwise creating a new one. Requires a 'classification' section "
            "in <workspace>/config.json; a missing or invalid config is a hard "
            "error (exit 1). Failures of the classification call itself (LLM "
            "error, auth) are reported in the 'classification' field of the "
            "response payload without aborting the file add."
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

    return parser


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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    fmt: str = args.format

    try:
        ws = Workspace.resolve(args.workspace)
        # `init` manages only the peer local_config.json; `workspace create`
        # is what actually builds the workspace — so both run before the
        # workspace exists.
        if args.command not in ("init", "workspace") and not ws.is_initialized():
            raise WorkspaceNotInitialized(
                f"workspace at {ws.root} is not initialized — run 'dgml workspace create'"
            )
        return _dispatch(args, ws, fmt)
    except DgmlError as exc:
        return _emit_error(exc.code, str(exc), fmt)
    except Exception as exc:
        import os
        import traceback

        # The JSON error envelope carries a short, single-line cause so an
        # agent parsing it isn't handed a wall of provider error text. The full
        # traceback goes to stderr under --verbose (or DGML_DEBUG) — stderr is
        # already non-JSON under --verbose, so it can't corrupt the envelope.
        if getattr(args, "verbose", False) or os.environ.get("DGML_DEBUG"):
            traceback.print_exc()
        return _emit_error("INTERNAL_ERROR", short_error_message(exc), fmt)


def _init_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Manage the shared peer ``local_config.json`` (config only).

    Does not create ``docsets/``, ``files/``, or any workspace ``config.json``
    — that is ``dgml workspace create``. Resolves the workspace only to locate
    the peer file at ``<workspace-parent>/local_config.json``.
    """
    path = ws.local_config_path

    if args.refresh:
        backup = ws.refresh_local_config()
        sys.stderr.write(f"[dgml init] refreshed {path} from the bundled default template.\n")
        if backup is not None:
            sys.stderr.write(f"[dgml init] previous config backed up to {backup}.\n")
        _emit(
            {"local_config_path": str(path), "local_config_created": False, "refreshed": True},
            fmt,
        )
        return 0

    created = ws.ensure_local_config()
    payload: dict[str, Any] = {
        "local_config_path": str(path),
        "local_config_created": created,
        "refreshed": False,
    }
    if created:
        payload["next_action"] = f"edit {path} then run dgml workspace create"
        sys.stderr.write(
            f"[dgml init] created {path}. Review/edit the models and OCR endpoint, "
            "then run `dgml workspace create`.\n"
        )
    _emit(payload, fmt)
    return 0


def _workspace_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Create a workspace from the shared ``local_config.json``."""
    sub = args.workspace_command
    if sub == "create":
        # A positional path overrides the globally-resolved root, so
        # `dgml workspace create ./ws …` reads without doubling --workspace.
        if args.path is not None:
            ws = Workspace.resolve(args.path)
        # `create` does not require a prior `init`: seed the shared
        # local_config.json from the bundled template if it is absent, so a
        # first-time user can create a workspace in one step.
        local_config_created = ws.ensure_local_config()
        existed = ws.config_path.exists()
        written = ws.write_config_from_local(overwrite=args.force)
        ws.init()
        name = args.name or ws.root.name
        ws.write_meta(name=name, organization=args.organization)
        if existed and written:
            sys.stderr.write(
                f"[dgml workspace create] overwrote {ws.config_path} from {ws.local_config_path}.\n"
            )
        payload: dict[str, Any] = {
            "workspace": str(ws.root),
            "name": name,
            "organization": args.organization,
            "initialized": True,
            "config_path": str(ws.config_path),
            "config_written": written,
            "local_config_path": str(ws.local_config_path),
            "local_config_created": local_config_created,
        }
        if local_config_created:
            # We just seeded local_config.json (and copied it to config.json)
            # from the template's placeholder models/OCR. Tell the caller how to
            # make it real.
            payload["next_action"] = (
                f"edit {ws.config_path} to set the models and OCR endpoint "
                f"(or edit the shared {ws.local_config_path} and re-run "
                "'dgml workspace create --force' to re-sync it)"
            )
            sys.stderr.write(
                f"[dgml workspace create] seeded {ws.local_config_path} from the bundled "
                f"template and copied it to {ws.config_path}. Edit the models and OCR "
                "endpoint before running LLM-backed commands.\n"
            )
        _emit(payload, fmt)
        return 0

    raise AssertionError(f"unhandled workspace subcommand: {sub}")  # unreachable (required=True)


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
                "docset_count": len(docsets),
                "file_count": len(files),
            },
            fmt,
        )
        return 0

    if cmd == "check":
        report = check_workspace(
            ws, retry_errors=args.retry_errors, verbose=args.verbose, debug=args.debug
        )
        _emit(report.to_json(), fmt)
        return 0 if report.ok else 2

    if cmd == "cluster":
        try:
            from dgml_core.clustering import clustering
        except ImportError:
            return _emit_error(
                "MISSING_EXTRA",
                "The 'clustering' extra is not installed. Run: pip install dgml[clustering]",
                fmt,
            )
        # `clustering` owns the `skipped` key and the skip-existing no-op
        # short-circuit (which avoids re-scanning the workspace). `config`
        # is passed through raw — it may be a preset name or a path.
        _emit(
            clustering(
                ws,
                skip_existing=getattr(args, "skip_existing", False),
                config=getattr(args, "config", None),
                mode=getattr(args, "mode", "auto"),
                method=getattr(args, "method", "embedding"),
                small_corpus_threshold=getattr(args, "small_corpus_threshold", 8),
                debug=args.debug,
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
            sys.stderr.write(
                f"[dgml discover] semantic filter unavailable ({exc}), falling back to All\n"
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
        return _emit_error(
            "MISSING_EXTRA",
            "The 'chain' extra is not installed. Run: pip install dgml[chain]",
            fmt,
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
        help="Override grounded.schema_model for this call (LiteLLM model string).",
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

    ex_extract = extraction.add_parser(
        "extract",
        parents=[common],
        help="Extract grounded values from a file against its DocSet schema.",
    )
    ex_extract.add_argument("docset_id")
    ex_extract.add_argument("file_id")
    ex_extract.add_argument(
        "--values-model",
        default=None,
        help="Override grounded.values_model for this call (LiteLLM model string).",
    )

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


def _extraction_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Dispatch the `extraction` command group."""
    from dataclasses import replace

    from dgml_core.extraction_schema import parse_rnc, rnc_to_json_schema
    from dgml_core.extraction_xml import dgml_xml_to_values
    from dgml_core.grounded import extract_values, generate_schema, load_grounded_config

    store = DocSetStore(ws)
    sub = args.extraction_command

    if sub == "generate-schema":
        ds = store.get(args.docset_id)  # raises DocSetNotFound
        config = load_grounded_config(ws)
        if args.schema_model:
            config = replace(config, schema_model=args.schema_model)
        file_ids = args.from_files or store.list_files(args.docset_id)
        if not file_ids:
            return _emit_error(
                "NO_FILES",
                f"docset '{args.docset_id}' has no files; pass --from-file or add files first",
                fmt,
            )
        rnc = generate_schema(ws, file_ids, config=config, docset_name=ds.name)
        store.set_schema(args.docset_id, rnc)
        _emit(
            {
                "docset_id": args.docset_id,
                "schema_format": "rnc",
                "schema": rnc,
                "from_file_ids": list(file_ids),
                "model": config.schema_model,
            },
            fmt,
        )
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

    if sub == "extract":
        config = load_grounded_config(ws)
        if args.values_model:
            config = replace(config, values_model=args.values_model)
        result = extract_values(
            ws,
            args.docset_id,
            args.file_id,
            config=config,
            write_stats=args.debug,
            debug=args.debug,
        )
        _emit(
            {
                "docset_id": args.docset_id,
                "file_id": args.file_id,
                "model": config.values_model,
                "mode": result.mode,
                "tool_calls": result.tool_calls,
                "field_count": len(result.values),
                "xml_path": str(result.xml_path),
            },
            fmt,
        )
        return 0

    if sub == "get-values":
        from dgml_core.extraction_xml import has_extraction

        # Extracted values live as a dg:extraction element inside the file's
        # core <stem>.dgml.xml — the single *.dgml.xml in the marker dir.
        candidates = sorted(ws.docset_file_dir(args.docset_id, args.file_id).glob("*.dgml.xml"))
        xml = candidates[0].read_text(encoding="utf-8") if candidates else ""
        if not xml or not has_extraction(xml):
            return _emit_error(
                "VALUES_NOT_FOUND",
                f"no extracted values for file '{args.file_id}' in docset '{args.docset_id}'; "
                "run 'dgml extraction extract' first",
                fmt,
            )
        if args.as_form == "xml":
            _emit(
                {"docset_id": args.docset_id, "file_id": args.file_id, "format": "xml", "xml": xml},
                fmt,
            )
        else:
            vocab = (
                parse_rnc(store.get_schema(args.docset_id))
                if store.has_schema(args.docset_id)
                else None
            )
            _emit(
                {
                    "docset_id": args.docset_id,
                    "file_id": args.file_id,
                    "format": "values",
                    "values": dgml_xml_to_values(xml, vocab=vocab),
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
    # The transcription and labeling models are NOT CLI flags: like every other
    # model-consuming command (schema generate, file extract, discover), they are
    # read solely from the workspace's 'generation' config section, so the model
    # is one visible, deliberate choice per workspace. See load_generation_config.
    gen.add_argument("--window-size", type=int, default=10, help="Pages per transcription window.")
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--max-tokens", type=int, default=32000)
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
        default=4,
        help=(
            "Max documents transcribed concurrently. Windows within a document "
            "stay serial. Set to 1 to disable parallelism. Tune to your provider's "
            "RPM tier (default: 4)."
        ),
    )
    gen.add_argument(
        "--schema-path",
        type=Path,
        default=None,
        help=(
            "Exported schema to seed labeling with — either docsets/<id>/schema.json "
            "(Schema v1: a `tags` map of name -> {role, kind, parent_role, ...}) or its "
            "RELAX NG Compact render docsets/<id>/full-schema.rnc. When given, this vocabulary "
            "is used as-is and the planning pass is skipped (making labels deterministic), "
            "and the tag hierarchy seeds entity-container grouping; per-document labeling "
            "still extends it for roles the schema does not cover."
        ),
    )
    gen.add_argument(
        "--no-roster",
        action="store_true",
        help=(
            "Disable automatic roster reuse. By default an incremental generate "
            "seeds labeling with the docset's existing cache/concept_roster.json so "
            "added documents stay tag-consistent; this labels them in isolation."
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


def _load_schema_roster(path: Path) -> dict[str, str]:
    """Load a flat ``{concept: description}`` JSON roster (the shape emitted at
    ``cache/concept_roster.json``) into a roster seed.

    Used for automatic roster reuse on an incremental generate, so newly-added
    documents stay tag-consistent with the docset's existing vocabulary. Concept
    keys are sanitized to PascalCase; descriptions are truncated to the roster
    hint length. Raises ``InvalidArgument`` on a missing / malformed file or one
    with no usable concepts.
    """
    from dgml_core.errors import InvalidArgument
    from dgml_core.generation.blocks import sanitize_concept

    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InvalidArgument(f"roster file not found: {path}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidArgument(f"roster is not valid JSON ({path}): {exc}") from exc
    if not isinstance(raw, dict):
        raise InvalidArgument(
            f"roster must be a JSON {{concept: description}} object, got "
            f"{type(raw).__name__} ({path})"
        )

    roster: dict[str, str] = {}
    for name, description in raw.items():
        concept = sanitize_concept(str(name))
        if concept:
            roster[concept] = str(description)[:60]
    if not roster:
        raise InvalidArgument(f"roster produced no usable concepts ({path})")
    return roster


def _load_schema_seed(path: Path) -> tuple[Schema, dict[str, str]]:
    """Load an exported schema — ``schema.json`` or ``full-schema.rnc`` — into
    ``(schema, parent_map)``.

    ``--schema-path`` accepts the two formats ``docset generate`` emits at the
    docset root: ``schema.json`` (Schema v1: a ``tags`` map of ``name ->
    {role, kind, parent_role, ...}``) or its lossless RELAX NG Compact render
    ``full-schema.rnc`` (a ``.rnc`` suffix; the ``# Field: value`` comment contract
    carries the same fields). The full schema seeds the labeling vocabulary —
    role descriptions, curated examples, kind, hierarchy (via
    ``ConvertOptions.schema_seed``); each tag's ``parent_role`` also becomes
    the leaf → container ``parent_map`` that drives entity-container grouping
    in ``render_dgml``.

    Raises ``InvalidArgument`` on a missing / malformed file, or one with no
    tags (e.g. a flat ``{concept: description}`` mapping — that shape is not
    accepted).
    """
    from dgml_core.errors import InvalidArgument
    from dgml_core.generation.blocks import sanitize_concept
    from dgml_core.generation.schema import Schema

    try:
        if Path(path).suffix.lower() == ".rnc":
            from dgml_core.generation.rnc import rnc_to_schema_dict

            schema = Schema.from_dict(rnc_to_schema_dict(Path(path).read_text(encoding="utf-8")))
        else:
            schema = Schema.load(path)
    except FileNotFoundError as exc:
        raise InvalidArgument(f"--schema-path file not found: {path}") from exc
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        raise InvalidArgument(f"--schema-path is not a valid schema ({path}): {exc}") from exc

    parent_map: dict[str, str] = {}
    for tag in schema.tags.values():
        concept = sanitize_concept(tag.name)
        parent = sanitize_concept(tag.parent_role or "")
        if concept and parent:
            parent_map[concept] = parent
    if not schema.tags:
        raise InvalidArgument(
            f"--schema-path has no tags — expected an exported schema.json or full-schema.rnc "
            f"(a flat {{concept: description}} mapping is not accepted) ({path})"
        )
    return schema, parent_map


def _file_result(status: str, file_id: str, source: str, **extra: Any) -> dict[str, Any]:
    """One entry in a batch command's ``results`` array: always ``status`` /
    ``file_id`` / ``source``, plus ``output`` (success) or ``error`` (failure).
    Centralizes the per-file shape so every producer stays in lockstep."""
    return {"status": status, "file_id": file_id, "source": source, **extra}


def _has_generated_tree(xml_path: Path) -> bool:
    """True when a ``<stem>.dgml.xml`` holds a generated document tree — the
    `docset generate` skip test. An extraction-only file (whose root has just a
    ``dg:extraction`` child) or an unparseable one returns False so generation
    proceeds and (re)builds the tree."""
    from dgml_core.extraction_xml import has_document_tree

    try:
        return has_document_tree(xml_path.read_text(encoding="utf-8"))
    except Exception:
        return False


def _generate_payload(
    ds: DocSet,
    total: int,
    skipped: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    converted: list[dict[str, Any]],
    output_dir: Path,
    coverage_report: Path | None,
) -> dict[str, Any]:
    """The single `docset generate` envelope, built the same way whether or not
    any file actually needed converting (so the two paths can't drift)."""
    return {
        "docset_id": ds.id,
        "docset_name": ds.name,
        "summary": {
            "total": total,
            "converted": len(converted),
            "skipped": len(skipped),
            "failed": len(failed),
        },
        "output_dir": str(output_dir),
        "coverage_report": str(coverage_report) if coverage_report is not None else None,
        "results": skipped + failed + converted,
    }


def _docset_generate_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    """Convert every file in a DocSet to DGML XML via the typed-block pipeline.

    Per window: flat JSON block transcription (``generation.model``); then ONE
    batch-wide semantic-labeling call across all documents
    (``generation.label_model``); then deterministic ``dg:chunk`` rendering. Word
    coverage is measured on the rendered XML unless ``--no-coverage``.

    Each rendered ``<stem>.dgml.xml`` is then grounded in place against the
    file's page OCR — ``dg:origin`` boxes are written onto every element with
    text content (deterministic, no LLM). A file with no ``page_text/`` is
    left ungrounded with a warning rather than failing the run. ``--debug``
    additionally writes the per-file ``<stem>.dgml.grounding_stats.json``.
    """
    from dgml_core import llm
    from dgml_core.errors import InvalidArgument
    from dgml_core.extraction_xml import carry_extraction_over, has_extraction
    from dgml_core.generation import (
        ConvertOptions,
        convert_batch,
        load_generation_config,
        resolve_generation_api_key,
        validate_generation_models,
    )
    from dgml_core.generation import coverage as cov_mod
    from dgml_core.generation.blocks import Block
    from dgml_core.generation.links import add_links
    from dgml_core.generation.pipeline import load_labeled_docs_from_cache
    from dgml_core.generation.rnc import write_docset_rnc
    from dgml_core.generation.to_semantic import build_header
    from dgml_core.usage import OPERATION_LINKS
    from dgml_core.xml_grounding import ground_dgml_xml

    def _diag(msg: str) -> None:
        # Progress is diagnostic, not part of the JSON contract: keep stdout a
        # single JSON object and surface progress on stderr only under --verbose.
        if args.verbose:
            print(msg, file=sys.stderr, flush=True)

    ds_store = DocSetStore(ws)
    file_store = FileStore(ws)

    ds = ds_store.get(args.docset_id)
    file_ids = ds_store.list_files(args.docset_id)
    if not file_ids:
        return _emit_error(
            "EMPTY_DOCSET",
            f"DocSet '{args.docset_id}' has no files assigned.",
            fmt,
        )

    # Validate the optional `style` config up front — before any LLM
    # transcription — rather than surfacing per-file during grounding. A
    # malformed section fails fast with STYLE_CONFIG_INVALID; a referenced-but-
    # unset `api_key_env` fails fast with AUTH_ERROR (the grounding-time style
    # pass is best-effort and would otherwise swallow this silently, after the
    # transcription spend).
    from dgml_core.style_config import load_style_config, resolve_api_key

    try:
        style_config = load_style_config(ws)
        if style_config is not None:
            resolve_api_key(style_config)
    except DgmlError as exc:
        return _emit_error(exc.code, str(exc), fmt)

    # Resolve the LLM models from the workspace 'generation' config — there are
    # no --model flags. load_generation_config raises GENERATION_CONFIG_MISSING
    # when the section is absent; both 'model' and 'label_model'
    # are required, so every model is a deliberate, visible choice.
    gen_cfg = load_generation_config(ws)
    gen_model = gen_cfg.model
    label_model = gen_cfg.label_model
    gen_api_key = resolve_generation_api_key(gen_cfg)
    gen_api_base = gen_cfg.api_base

    # Pre-flight — fail fast BEFORE any transcription spend on the two model
    # misconfigurations detectable offline: a malformed model string, or a
    # missing API key for either model's provider. A present-but-wrong key or a
    # well-formed-but-nonexistent model id can't be caught here; those surface
    # per file as label_error (see _on_label_error below). Mirrors the style-
    # config pre-flight above.
    try:
        validate_generation_models(gen_cfg, gen_api_key)
    except DgmlError as exc:
        return _emit_error(exc.code, str(exc), fmt)

    # The semantic-link pass runs on the labeling model.
    link_config = llm.LLMConfig(
        model=label_model,
        api_key=gen_api_key,
        api_base=gen_api_base,
        workspace=ws,
        debug=args.debug,
        operation=OPERATION_LINKS,
    )

    # The docset directory is always the output base — schema.json,
    # coverage_report.json, cache/, and semantic/ live here. Each file's
    # final .dgml.xml lands in its per-(docset, file) directory (see
    # ws.file_dgml_xml_path) so placement is deterministic and stable.
    output_dir = ws.docset_dir(args.docset_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve each assigned file into exactly one bucket so the summary counts
    # always sum to `total`: skipped (already converted), failed (source
    # missing, or a duplicate filename the pipeline can't disambiguate), or a
    # to-convert candidate. Partial success — a per-file problem is recorded
    # and the run continues (exit 0), matching `dgml cluster`.
    skipped_results: list[dict[str, Any]] = []
    failed_results: list[dict[str, Any]] = []
    # original_filename → list of (file_id, pdf_path, out_xml, page_text_dir).
    # Grouped by filename to detect collisions: convert_batch keys documents by
    # filename, so two files sharing a basename can't both convert in one run.
    candidates: dict[str, list[tuple[str, Path, Path, Path | None]]] = {}
    # Already-generated docs, for whole-docset roster reuse + namespacing recompute.
    prior_stems: dict[str, str] = {}  # cache stem → original_filename
    prior_out_paths: dict[str, Path] = {}  # original_filename → existing .dgml.xml
    # original_filename → file id for grounding (resolves the file's page OCR).
    # Spans candidates *and* re-rendered prior docs; kept separate from
    # filename_to_fid so the failure-reconciliation loop stays candidate-only.
    name_to_fid: dict[str, str] = {}
    for fid in file_ids:
        record = file_store.get(fid)
        name = record.original_filename
        pdf_path = ws.file_dir(fid) / name
        if not pdf_path.exists():
            failed_results.append(
                _file_result(
                    "failed",
                    fid,
                    name,
                    error={
                        "code": "FILE_NOT_FOUND",
                        "message": f"source not found at {pdf_path} for file '{fid}'",
                    },
                )
            )
            _diag(f"Source missing for {name} (file '{fid}') — reported as failed")
            continue
        out_xml = ws.file_dgml_xml_path(args.docset_id, fid, pdf_path.stem)
        if out_xml.exists() and _has_generated_tree(out_xml):
            # Skip only when a generated document tree is present. An
            # extraction-only file (`extraction extract` ran before
            # `generate`) falls through and gets its tree built; _on_output
            # carries the existing dg:extraction over into the fresh render.
            skipped_results.append(_file_result("skipped", fid, name, output=str(out_xml)))
            prior_stems[pdf_path.stem] = name
            prior_out_paths[name] = out_xml
            name_to_fid[name] = fid  # in case it re-renders below and needs re-grounding
            _diag(f"Skipping {name} (already converted)")
            continue
        ptd = ws.file_text_dir(fid)
        candidates.setdefault(name, []).append(
            (fid, pdf_path, out_xml, ptd if ptd.is_dir() else None)
        )

    # Same-basename collision: the typed-block pipeline keys documents by
    # filename, so it can't tell two same-named files apart in one batch.
    # Fail them explicitly instead of silently dropping/misattributing output.
    pdf_paths: list[Path | str] = []
    dgml_xml_paths: dict[str, Path] = {}
    filename_to_fid: dict[str, str] = {}
    page_text_dirs: dict[str, Path] = {}
    for name, group in candidates.items():
        if len(group) > 1:
            for fid, _pdf, _out, _ptd in group:
                failed_results.append(
                    _file_result(
                        "failed",
                        fid,
                        name,
                        error={
                            "code": "GENERATION_FAILED",
                            "message": (
                                f"duplicate filename '{name}' within the docset; the "
                                "generation pipeline keys documents by filename, so give "
                                "each file a unique name before converting"
                            ),
                        },
                    )
                )
            _diag(f"Duplicate filename '{name}' across {len(group)} files — reported as failed")
            continue
        fid, pdf_path, out_xml, pt_dir = group[0]
        pdf_paths.append(pdf_path)
        dgml_xml_paths[name] = out_xml
        filename_to_fid[name] = fid
        name_to_fid[name] = fid
        if pt_dir is not None:
            page_text_dirs[name] = pt_dir

    # Coverage is computed (and its per-file summary printed) whenever the user
    # didn't pass --no-coverage, but the coverage_report.json *file* is an
    # intermediate artifact persisted only under --debug.
    compute_cov = not args.no_coverage
    cov_path = output_dir / "coverage_report.json" if (compute_cov and args.debug) else None
    written: list[dict[str, Any]] = []
    rerendered: list[str] = []
    cov_results: list[dict[str, Any]] = []
    # Already-generated docs reloaded from cache (populated below when there is
    # new work) so namespacing spans the whole docset and flipped originals
    # re-render. _on_output reads prior_outputs to route/flag them.
    prior_docs: dict[str, list[Block]] = {}
    prior_outputs: dict[str, str] = {}

    # name → short reason for a per-document transcription failure, so the
    # reconciliation loop below can name the cause in the JSON payload instead
    # of the generic "produced no output" message. The full error still goes to
    # stderr under --verbose via _diag (convert_batch's progress log).
    gen_errors: dict[str, str] = {}
    # name → {code, message} when a file's labeling couldn't reach the model at
    # all (bad model id, wrong/absent key, network). Surfaced as label_error on
    # the converted entry so a misconfigured label_model is visible without
    # --verbose; the document still renders (unlabeled). Labeling completes
    # before any _on_output fires, so the entry below can read this.
    label_errors: dict[str, dict[str, str]] = {}

    def _on_error(name: str, message: str) -> None:
        gen_errors[name] = message

    def _on_label_error(name: str, err: dict[str, str]) -> None:
        label_errors[name] = err
        _diag(f"[label] {name}: model unreachable ({err.get('message', '')})")

    def _on_output(name: str, xml: str) -> None:
        out_xml = dgml_xml_paths[name]
        out_xml.parent.mkdir(parents=True, exist_ok=True)
        # A file already at this path may carry extracted values — an
        # extraction-only file getting its tree now, or a full-extraction
        # file being re-rendered. Capture its dg:extraction before the fresh
        # render overwrites it; re-embedded below after grounding + semlinks.
        prior_with_extraction: str | None = None
        if out_xml.exists():
            try:
                prior_text = out_xml.read_text(encoding="utf-8")
                if has_extraction(prior_text):
                    prior_with_extraction = prior_text
            except Exception:
                prior_with_extraction = None  # unparseable prior — nothing to carry
        out_xml.write_text(xml, encoding="utf-8")
        # Ground in place: re-parse the just-written tree, align it against the
        # file's page OCR, and rewrite <stem>.dgml.xml with dg:origin boxes.
        # Deterministic and free; a file with no page_text is left ungrounded.
        # Runs for re-rendered prior docs too — their fresh XML would otherwise
        # lose the boxes a previous run grounded in.
        grounding: dict[str, Any]
        try:
            res = ground_dgml_xml(
                ws,
                name_to_fid[name],
                out_xml,
                output_path=out_xml,
                force=True,
                write_stats=args.debug,
                debug=args.debug,
            )
        except DgmlError as exc:
            grounding = {
                "grounded": False,
                "grounding_error": {"code": exc.code, "message": str(exc)},
            }
            _diag(f"[ground] {name}: not grounded ({exc})")
        else:
            grounding = {
                "grounded": True,
                "matched_token_pct": res.stats["matched_token_pct"],
                "elements_annotated": res.stats["elements_annotated"],
            }
            _diag(
                f"[ground] {name}: {res.stats['elements_annotated']} element(s), "
                f"{res.stats['matched_token_pct']}% tokens matched"
            )
        # Final step: add semantic links in place (dg:itemprop/dg:href). Runs on
        # re-rendered priors too — their fresh XML would otherwise lose the links.
        links_added = 0
        if not args.no_semlinks:
            try:
                linked, applied = add_links(out_xml.read_text(encoding="utf-8"), link_config)
                out_xml.write_text(linked, encoding="utf-8")
                links_added = len(applied)
                _diag(f"[semlinks] {name}: {links_added} link(s)")
            except Exception as exc:  # a link-pass failure must not lose the DGML
                _diag(f"[semlinks] {name}: skipped ({exc})")
        # Re-embed the prior dg:extraction last, after grounding + semlinks
        # have finished rewriting the tree, so the extraction subtree is
        # spliced in verbatim and never run through those passes.
        if prior_with_extraction is not None:
            try:
                merged = carry_extraction_over(
                    prior_with_extraction, out_xml.read_text(encoding="utf-8")
                )
                out_xml.write_text(merged, encoding="utf-8")
                _diag(f"[extraction] {name}: carried dg:extraction over into the fresh render")
            except Exception as exc:  # never lose the fresh DGML over the merge
                _diag(f"[extraction] {name}: dg:extraction NOT carried over ({exc})")
        if name in prior_outputs:
            rerendered.append(name)  # an already-generated doc whose namespacing flipped
            return
        pt_dir = page_text_dirs.get(name)
        if compute_cov and pt_dir is not None:
            result = cov_mod.compute_coverage(xml, name, page_text_dir=pt_dir)
            _diag(cov_mod.coverage_summary_line(result))
            cov_results.append(result)
        # Present only on files whose labeling couldn't reach the model — like
        # grounding_error, which appears only when grounded is False.
        label_error = label_errors.get(name)
        label_extra = {"label_error": label_error} if label_error is not None else {}
        written.append(
            _file_result(
                "converted",
                filename_to_fid[name],
                name,
                output=str(out_xml),
                links=links_added,
                **grounding,
                **label_extra,
            )
        )

    if pdf_paths:
        # The cache always exists — it holds functional files the next run
        # reloads (blocks, per-chunk labels, concept_roster.json). --debug only
        # controls whether the extra debug-only artifacts are also written
        # (threaded via ConvertOptions.debug below).
        cache_dir = args.cache_dir or output_dir / "cache"
        roster_path = Path(cache_dir) / "concept_roster.json"
        schema_seed = None
        roster_seed: dict[str, str] | None = None
        parent_map_seed: dict[str, str] = {}
        if args.schema_path:
            schema_seed, parent_map_seed = _load_schema_seed(Path(args.schema_path))
            _diag(
                f"Loaded schema: {len(schema_seed.tags)} concept(s), "
                f"{len(parent_map_seed)} container link(s) from {args.schema_path}"
            )
        elif not args.no_roster:
            # Incremental reuse prefers the docset's own schema.json — full
            # fidelity (role descriptions, observed examples, kind, hierarchy)
            # — over the flat cache/concept_roster.json fallback. Unlike
            # --schema-path, no parent_map is derived here: entity-container
            # grouping stays an explicit opt-in.
            from dgml_core.generation.schema import Schema

            schema_json_path = Path(cache_dir).parent / "schema.json"
            if schema_json_path.exists():
                try:
                    schema_seed = Schema.load(schema_json_path)
                    _diag(f"Reusing docset schema: {len(schema_seed.tags)} tag(s)")
                except (json.JSONDecodeError, TypeError, ValueError, OSError):
                    schema_seed = None
            if schema_seed is None and roster_path.exists():
                try:
                    roster_seed = _load_schema_roster(roster_path)
                    _diag(f"Reusing docset roster: {len(roster_seed)} concept(s)")
                except InvalidArgument:
                    roster_seed = None

        # Reload already-generated docs from cache so the whole docset stays
        # consistent as its schema/roster grows; changed originals re-render
        # (no re-LLM).
        for stem, blocks in load_labeled_docs_from_cache(cache_dir, list(prior_stems)).items():
            nm = prior_stems[stem]
            prior_docs[nm] = blocks
            prior_outputs[nm] = prior_out_paths[nm].read_text(encoding="utf-8")
            dgml_xml_paths[nm] = prior_out_paths[nm]

        options = ConvertOptions(
            model=gen_model,
            label_model=label_model,
            api_key=gen_api_key,
            api_base=gen_api_base,
            window_size=args.window_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_parallel_docs=args.max_parallel_calls,
            cache_dir=cache_dir,
            debug=args.debug,
            page_text_dirs=page_text_dirs,
            workspace=ws,
            dgml_header=build_header(ws.organization, ds.name),
            converters=load_conversion_config(ws),
            roster_seed=roster_seed,
            schema_seed=schema_seed,
            parent_map=parent_map_seed or None,
            progress=_diag,
        )
        convert_batch(
            pdf_paths,
            options=options,
            on_output=_on_output,
            on_error=_on_error,
            on_label_error=_on_label_error,
            prior_docs=prior_docs,
            prior_outputs=prior_outputs,
        )
        # convert_batch silently drops documents whose transcription failed, so
        # `_on_output` never fires for them. Reconcile: any queued file with no
        # output is a per-file failure, not a vanished row (keeps counts summing
        # to `total`).
        produced = {entry["source"] for entry in written}
        for name, fid in filename_to_fid.items():
            if name not in produced:
                message = gen_errors.get(
                    name, "the generation pipeline produced no output for this file"
                )
                failed_results.append(
                    _file_result(
                        "failed",
                        fid,
                        name,
                        error={"code": "GENERATION_FAILED", "message": message},
                    )
                )
        if cov_path is not None and cov_results:
            # Merge into any existing report so an incremental run keeps the
            # already-generated docs' coverage instead of overwriting it.
            existing_docs: list[dict[str, Any]] = []
            if cov_path.exists():
                try:
                    existing_docs = json.loads(cov_path.read_text(encoding="utf-8")).get(
                        "documents", []
                    )
                except (OSError, json.JSONDecodeError):
                    existing_docs = []
            cov_mod.save_coverage_report(
                cov_mod.merge_coverage_documents(existing_docs, cov_results), cov_path
            )
    else:
        _diag("Nothing to convert — every file is already converted, missing, or a duplicate name.")

    # Final step, after every file is converted, grounded and semlinked:
    # refresh the docset's full-schema.rnc (schema.json rendered as RELAX NG
    # Compact, with data types observed in the final XML). Best-effort — a
    # schema render failure must not fail the generate.
    try:
        rnc_path = write_docset_rnc(output_dir)
        if rnc_path is not None:
            _diag(f"[schema] wrote {rnc_path.name}")
    except Exception as exc:
        _diag(f"[schema] full-schema.rnc skipped ({exc})")

    # Report the coverage path only if a report was actually written.
    coverage_report = cov_path if cov_results else None
    payload = _generate_payload(
        ds, len(file_ids), skipped_results, failed_results, written, output_dir, coverage_report
    )
    payload["rerendered"] = rerendered
    _emit(payload, fmt)
    return 0


def _docset_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    store = DocSetStore(ws)
    sub = args.docset_command
    if sub == "generate":
        return _docset_generate_cmd(args, ws, fmt)
    if sub == "create":
        ds = store.create(
            name=args.name,
            description=args.description,
            key_questions=args.key_questions,
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
            ws, args.docset_id, args.file_id, write_stats=args.debug, debug=args.debug
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
    parse the same shape either way (part of the public CLI contract).
    """
    return {
        "file": result.record.to_json(),
        "created": result.created,
        "conflict_kind": result.conflict_kind,
        "page_render_error": result.page_render_error,
        "page_count_error": result.page_count_error,
        "text_extraction_error": result.text_extraction_error,
        "conversion_error": result.conversion_error,
        "text_extraction": result.text_extraction,
        "note": result.note,
    }


def _ingestible_suffixes(ws: Workspace) -> frozenset[str]:
    """Suffixes a directory bulk-add will collect: ``.pdf`` always, plus the
    convertible source extensions whose format family has a converter
    configured in the workspace. Unconfigured source types are skipped (not
    gathered), so a folder of PDFs with stray Office docs doesn't produce a
    pile of per-file failures. Raises ``ConversionConfigInvalid`` on a
    malformed ``conversion`` config (a real error worth surfacing once)."""
    configured = load_conversion_config(ws)
    extra = {sfx for sfx, family in FAMILY_BY_SUFFIX.items() if family in configured}
    return frozenset({".pdf"} | extra)


def _gather_pdfs(directory: Path, *, recursive: bool, suffixes: frozenset[str]) -> list[Path]:
    """Return the ingestible files under ``directory`` (case-insensitive).

    ``suffixes`` is the accepted extension set (see :func:`_ingestible_suffixes`).
    ``recursive`` descends into subdirectories; otherwise only the top level
    is scanned (matching the ``find -maxdepth 1`` variant the skill calls
    out). Results are lex-sorted so a bulk run is deterministic.
    """
    candidates = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in suffixes)


def _file_add_bulk(args: argparse.Namespace, ws: Workspace, store: FileStore, fmt: str) -> int:
    """Add every PDF under a directory in one run, emitting a single envelope.

    Each file commits independently (same as adding them one at a time): a
    per-file failure is recorded in its entry and the run continues. The
    payload carries a ``summary`` count block plus a per-file ``results``
    array; every entry carries a ``status`` (``added`` / ``skipped`` /
    ``soft_failed`` / ``hard_failed``) matching the summary counts. The
    command exits 0 as long as the run completes — individual soft- or
    hard-failures are reported, not raised (partial success is the contract,
    matching ``dgml cluster``). Only a run-level abort (workspace not
    initialized, directory unreadable) surfaces as an error envelope.
    """
    directory: Path = args.path
    recursive: bool = args.recursive
    pdfs = _gather_pdfs(directory, recursive=recursive, suffixes=_ingestible_suffixes(ws))

    auto_classify = getattr(args, "auto_classify", False)
    config: ClassificationConfig | None = None
    docsets: list[DocSet] | None = None
    if auto_classify:
        # Load the classification config once, up front: a missing/invalid
        # config is a hard failure that aborts the run before any file is
        # added, rather than recording the same error on every file.
        config = load_classification_config(ws)
        # Read existing DocSets once; _auto_classify appends newly-created
        # ones so similar PDFs cluster within the run without re-scanning.
        docsets = DocSetStore(ws).list_all()

    on_conflict = ConflictPolicy(args.on_conflict)
    text_mode = TextMode(args.text_mode)

    counts = {"added": 0, "skipped": 0, "soft_failed": 0, "hard_failed": 0}
    entries: list[dict[str, Any]] = []
    for pdf in pdfs:
        try:
            result = store.add(pdf, on_conflict=on_conflict, text_mode=text_mode, debug=args.debug)
        except DgmlError as exc:
            counts["hard_failed"] += 1
            entries.append(
                {
                    "status": "hard_failed",
                    "path": str(pdf),
                    "error": {"code": exc.code, "message": str(exc)},
                }
            )
            continue

        if not result.created:
            status = "skipped"
        elif (
            result.page_render_error
            or result.page_count_error
            or result.text_extraction_error
            or result.conversion_error
        ):
            status = "soft_failed"
        else:
            status = "added"
        counts[status] += 1

        entry: dict[str, Any] = {"status": status, "path": str(pdf), **_file_add_payload(result)}
        if auto_classify:
            entry["classification"] = _auto_classify(
                ws, result, config=config, docsets=docsets, debug=args.debug
            )
        entries.append(entry)

    payload: dict[str, Any] = {
        "directory": str(directory),
        "recursive": recursive,
        "summary": {"total": len(pdfs), **counts},
        "results": entries,
    }
    _emit(payload, fmt)
    return 0


def _file_cmd(args: argparse.Namespace, ws: Workspace, fmt: str) -> int:
    store = FileStore(ws)
    sub = args.file_command
    if sub == "add":
        if args.path.is_dir():
            return _file_add_bulk(args, ws, store, fmt)
        result = store.add(
            args.path,
            on_conflict=ConflictPolicy(args.on_conflict),
            text_mode=TextMode(args.text_mode),
            verbose=args.verbose,
            debug=args.debug,
        )
        payload: dict[str, Any] = _file_add_payload(result)
        if getattr(args, "auto_classify", False):
            # _auto_classify loads the classification config itself; a
            # missing/invalid one raises straight through to an error envelope.
            payload["classification"] = _auto_classify(ws, result, debug=args.debug)
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


def _auto_classify(
    ws: Workspace,
    result: AddFileResult,
    *,
    config: ClassificationConfig | None = None,
    docsets: list[DocSet] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run LLM auto-classification on a freshly added File and assign it.

    Returns the ``classification`` block embedded in ``dgml file add`` output.

    A missing or invalid classification config is a **hard** failure: when
    ``config`` is not supplied it is loaded here via
    :func:`load_classification_config`, whose error propagates straight to the
    CLI error envelope (exit 1) rather than soft-failing per file. Bulk callers
    load the config once up front and pass it in, so the run aborts before any
    file is processed when it's missing. Failures *after* config is in hand —
    the LLM/classify call, auth — stay soft: the File record is already on
    disk, so they land in ``classification.error`` with exit 0.

    ``docsets``, when supplied, is a mutable list the caller maintains across
    a bulk run: it is forwarded to :func:`classify_file` so the LLM sees
    DocSets created earlier in the same run, and any freshly-created DocSet
    is appended to it here so later files can be assigned to it.

    Skipped (and reported as ``performed: false``) when the add returned an
    existing record rather than creating a new one — re-runs stay idempotent,
    and we neither require config nor burn an LLM call on a duplicate.
    """
    if not result.created:
        return {
            "performed": False,
            "reason": "file already existed; classification skipped",
        }

    if config is None:
        config = load_classification_config(ws)

    file_id = result.record.id
    block: dict[str, Any] = {
        "performed": True,
        "model": config.model,
        "decision": None,
        "docset_id": None,
        "docset_created": False,
        "docset_name": None,
        "docset_key_questions": [],
        "error": None,
    }

    try:
        decision = classify_file(ws, file_id, config=config, docsets=docsets, debug=debug)
    except DgmlError as exc:
        block["error"] = f"{exc.code}: {exc}"
        return block

    docset_store = DocSetStore(ws)
    try:
        if decision.decision == "existing":
            assert decision.existing_docset_id is not None
            # Assign, and auto-extract when the target DocSet has an
            # extraction schema set (soft-fail — the extraction block
            # carries any error; the assignment itself stands).
            from dgml_core.extraction import add_file_and_extract

            extraction_block = add_file_and_extract(
                ws, decision.existing_docset_id, file_id, write_stats=debug, debug=debug
            )
            existing = docset_store.get(decision.existing_docset_id)
            block.update(
                decision="existing",
                docset_id=existing.id,
                docset_name=existing.name,
                docset_key_questions=list(existing.key_questions),
            )
            if extraction_block is not None:
                block["extraction"] = extraction_block
        else:
            assert decision.new_name is not None and decision.new_description is not None
            created = docset_store.create(
                name=decision.new_name,
                description=decision.new_description,
                key_questions=list(decision.new_key_questions),
            )
            docset_store.add_file(created.id, file_id)
            if docsets is not None:
                docsets.append(created)
            block.update(
                decision="new",
                docset_id=created.id,
                docset_created=True,
                docset_name=created.name,
                docset_key_questions=list(created.key_questions),
            )
    except DgmlError as exc:
        block["error"] = f"{exc.code}: {exc}"
    return block


if __name__ == "__main__":
    sys.exit(main())
