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

"""Value extraction for a DocSet file: on demand, or automatically on assignment.

:func:`extract_file` backs ``dgml extraction extract``: a hard-failing call on a
file already assigned to the DocSet. The *auto-extract-on-assignment* path
(:func:`add_file_and_extract`) backs every CLI assignment: ``docset add-file``,
``file add --auto-classify`` (existing-DocSet decisions), and ``cluster``
(existing-DocSet matches — a DocSet created mid-run can't have a schema yet).

:func:`add_file_and_extract` assigns the file and, if the target DocSet has an
extraction schema set, fires value extraction. Soft-fail semantics: an
extraction failure surfaces in the returned block's ``error`` field; the
assignment itself still succeeds. No schema → plain assignment, no block.

:func:`get_extracted_values` is the read side: the extracted values of one pair as
values-shape JSON, as ``dgml extraction get-values`` prints them.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .dgml_output import read_dgml
from .docsets import DocSetStore
from .errors import DgmlError, ValuesNotFound
from .storage import Workspace

if TYPE_CHECKING:
    from .grounded import ExtractionResult, GroundedConfig

logger = logging.getLogger(__name__)


def auto_extract(
    ws: Workspace,
    docset_id: str,
    file_id: str,
    *,
    write_stats: bool = False,
    record_usage: bool = False,
) -> dict[str, Any]:
    """Run LLM-backed value extraction on a freshly-assigned file.

    Returns the ``extraction`` block embedded in JSON payloads of
    commands that assign files to DocSets (``dgml docset add-file``,
    ``dgml file add --auto-classify``, ``dgml cluster``). Soft-fails:
    the assignment is already on disk; any failure here surfaces in
    ``extraction.error`` without aborting the parent command. A config
    load error is captured into ``error`` the same way.

    ``usage`` is the run's cost/token totals (``ExtractionResult.usage``),
    present whether or not ``record_usage`` persisted a row, so a host can
    account for auto-extraction without reading ``usage.jsonl``. A failed run
    reports what it spent before failing (the error's ``usage``); it is
    ``None`` when nothing was spent (a config error) or the totals could not
    be built.
    """
    from .grounded import extract_values, load_grounded_config

    block: dict[str, Any] = {
        "performed": True,
        "model": None,
        "tool_calls": None,
        "usage": None,
        "error": None,
    }
    try:
        config = load_grounded_config(ws)
    except DgmlError as exc:
        block["error"] = f"{exc.code}: {exc}"
        _log_auto_extract_failed(file_id, docset_id, block["error"])
        return block
    block["model"] = config.values_model
    # extract_values logs the run's start and finish; only the soft failure,
    # which is this caller's call, is logged here.
    try:
        result = extract_values(
            ws,
            docset_id,
            file_id,
            config=config,
            write_stats=write_stats,
            record_usage=record_usage,
        )
    except DgmlError as exc:
        block["error"] = f"{exc.code}: {exc}"
        block["usage"] = exc.usage
        _log_auto_extract_failed(file_id, docset_id, block["error"])
        return block
    except Exception as exc:
        # Bug-grade exception from extraction (e.g. IndexError in a phase).
        # The assignment is already persisted, and clustering/bulk callers
        # need to emit their summary for the *other* files, so we soft-fail
        # here too — the error type/message lands in `extraction.error` and
        # the parent command exits 0. The traceback goes to the log, since
        # the block can only carry the message.
        block["error"] = f"{type(exc).__name__}: {exc}"
        _log_auto_extract_failed(file_id, docset_id, block["error"], exc_info=True)
        return block
    block["tool_calls"] = result.tool_calls
    block["usage"] = result.usage or None
    return block


def _log_auto_extract_failed(
    file_id: str, docset_id: str, error: str, *, exc_info: bool = False
) -> None:
    """Log a soft-failed auto-extraction: the assignment stands, but the file has
    no extracted values. INFO, not WARNING: the failure already reaches the caller
    in the block's ``error``, and extraction was incidental to the assignment, so
    the line (and any traceback) belongs under --verbose. A WARNING would also
    repeat once per file in a bulk run."""
    logger.info(
        "extraction of file %s into docset %s failed; the file stays assigned "
        "without extracted values: %s",
        file_id,
        docset_id,
        error,
        exc_info=exc_info,
    )


def add_file_and_extract(
    ws: Workspace,
    docset_id: str,
    file_id: str,
    *,
    write_stats: bool = False,
    record_usage: bool = False,
) -> dict[str, Any] | None:
    """Assign ``file_id`` to ``docset_id``; auto-extract if the DocSet has a schema.

    Returns the extraction block (same shape :func:`auto_extract` produces)
    when the DocSet has a schema set, else ``None`` — the omit-when-no-schema
    policy matches the existing ``dgml docset add-file`` contract.

    ``write_stats`` and ``record_usage`` are forwarded to
    :func:`dgml_core.grounded.extract_values`: ``extraction_stats.json`` is
    stored only when ``write_stats`` is set, and the LLM usage row only when
    ``record_usage`` is (both off by default; the CLI sets both under
    ``--debug``).
    """
    store = DocSetStore(ws)
    store.add_file(docset_id, file_id)  # logs the assignment itself
    if not store.has_schema(docset_id):
        logger.info(
            "docset %s has no extraction schema; not extracting file %s", docset_id, file_id
        )
        return None
    return auto_extract(ws, docset_id, file_id, write_stats=write_stats, record_usage=record_usage)


def extract_file(
    ws: Workspace,
    docset_id: str,
    file_id: str,
    *,
    values_model: str | None = None,
    values_effort: str | None = None,
    locations_effort: str | None = None,
    write_stats: bool = False,
    record_usage: bool = False,
) -> ExtractionResult:
    """Extract ``file_id`` against ``docset_id``'s schema now — ``dgml extraction extract``.

    The workspace's grounded config is loaded first (:class:`GroundedConfigMissing`
    / :class:`GroundedConfigInvalid`). Then the docset must have a schema
    (:class:`SchemaNotFound`) and the file must already be assigned to it
    (:class:`FileNotFound`); schema and assignment can be set in either order
    before this runs. Unlike :func:`add_file_and_extract` this is a hard-failing
    call that returns the :class:`ExtractionResult`.

    ``values_model`` and ``values_effort`` override the workspace's
    ``grounded.values_model`` / ``grounded.values_reasoning_effort`` for this call;
    ``values_effort`` takes the config's values (``"default"`` sends none).
    ``locations_effort`` does the same for ``grounded.locations_reasoning_effort``,
    the effort of the location-grounding call on pages with OCR words (the no-words
    grid path keeps its own ``"medium"``).
    ``values_model`` resolves as a layer above ``[grounded]``: it replaces the
    model's whole credential group, so neither the section's ``values_*`` keys nor
    a tier's carry over to the new model
    (see :func:`~dgml_core.grounded.load_grounded_config`).
    ``write_stats`` and ``record_usage`` (both off by default) are forwarded
    to :func:`~dgml_core.grounded.extract_values`; the result's ``usage`` and
    ``stats`` carry the same numbers either way.
    """
    from .grounded import extract_values

    config = extraction_config(
        ws,
        values_model=values_model,
        values_effort=values_effort,
        locations_effort=locations_effort,
    )
    try:
        return extract_values(
            ws,
            docset_id,
            file_id,
            config=config,
            write_stats=write_stats,
            record_usage=record_usage,
        )
    except DgmlError as exc:
        # Raised to the caller, which decides how loud a failure is; INFO keeps
        # the run's story complete under --verbose without doubling the CLI's
        # error envelope by default.
        logger.info(
            "extraction of file %s into docset %s failed: %s: %s",
            file_id,
            docset_id,
            exc.code,
            exc,
        )
        raise


def extraction_config(
    ws: Workspace,
    *,
    values_model: str | None = None,
    values_effort: str | None = None,
    locations_effort: str | None = None,
) -> GroundedConfig:
    """The workspace's grounded config with :func:`extract_file`'s per-call
    overrides applied — the config every ``dgml extraction extract`` form
    (one file, several, ``--batch``) extracts with. See :func:`extract_file`
    for what each override means and how ``values_model`` resolves its
    credentials."""
    from .grounded import load_grounded_config, parse_reasoning_effort

    config = load_grounded_config(ws, values_model=values_model)
    if values_effort is not None:
        config = replace(
            config,
            values_reasoning_effort=parse_reasoning_effort(values_effort, source="values_effort"),
        )
    if locations_effort is not None:
        config = replace(
            config,
            locations_reasoning_effort=parse_reasoning_effort(
                locations_effort, source="locations_effort"
            ),
        )
    return config


def get_extracted_values(workspace: Workspace, docset_id: str, file_id: str) -> dict[str, Any]:
    """Return the pair's extracted values as ``{tag: {text, value?, locations}}``.

    Projects the ``dg:extraction`` element of the core DGML file, classifying
    each element by the docset's extraction schema when one is set (so a
    single-item collection stays a list), else inferring structure from the XML.

    Raises what :func:`read_dgml` raises, plus :class:`ValuesNotFound` when the
    DGML has no ``dg:extraction`` element (a generated tree only).
    """
    from .extraction_schema import parse_rnc
    from .extraction_xml import dgml_xml_to_values, has_extraction

    doc = read_dgml(workspace, docset_id, file_id)
    if not has_extraction(doc.xml):
        raise ValuesNotFound(
            f"no extracted values for file '{file_id}' in docset '{docset_id}'; "
            "run 'dgml extraction extract' first"
        )
    store = DocSetStore(workspace)
    vocab = parse_rnc(store.get_schema(docset_id)) if store.has_schema(docset_id) else None
    return dgml_xml_to_values(doc.xml, vocab=vocab)
