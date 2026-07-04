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

"""Helpers that assign a file to a DocSet and (optionally) auto-extract values.

The explicit extraction surface is live on the CLI as the ``dgml extraction``
command group (``generate-schema`` / ``set-schema`` / ``get-schema`` /
``extract`` / ``get-values``), which calls :func:`dgml_core.grounded.extract_values`
directly. This module's *auto-extract-on-assignment* path
(:func:`add_file_and_extract`) backs every CLI assignment: ``docset add-file``,
``file add --auto-classify`` (existing-DocSet decisions), and ``cluster``
(existing-DocSet matches — a DocSet created mid-run can't have a schema yet).

:func:`add_file_and_extract` assigns the file and, if the target DocSet has an
extraction schema set, fires value extraction. Soft-fail semantics: an
extraction failure surfaces in the returned block's ``error`` field; the
assignment itself still succeeds. No schema → plain assignment, no block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .docsets import DocSetStore
from .errors import DgmlError
from .storage import Workspace

if TYPE_CHECKING:
    from .grounded import GroundedConfig


def _auto_extract(
    ws: Workspace,
    docset_id: str,
    file_id: str,
    *,
    config: GroundedConfig | None = None,
    write_stats: bool = True,
    debug: bool = False,
) -> dict[str, Any]:
    """Run LLM-backed value extraction on a freshly-assigned file.

    Returns the ``extraction`` block embedded in JSON payloads of
    commands that assign files to DocSets (``dgml docset add-file``,
    ``dgml file add --auto-classify``, ``dgml cluster``). Soft-fails:
    the assignment is already on disk; any failure here surfaces in
    ``extraction.error`` without aborting the parent command.

    When ``config`` is provided, the caller is responsible for loading
    it — useful in bulk paths that want one config load amortized over
    many files. When ``None``, :func:`load_grounded_config` is invoked
    here and its error (if any) is captured into ``error``.
    """
    from .grounded import extract_values, load_grounded_config

    block: dict[str, Any] = {
        "performed": True,
        "model": None,
        "tool_calls": None,
        "error": None,
    }
    if config is None:
        try:
            config = load_grounded_config(ws)
        except DgmlError as exc:
            block["error"] = f"{exc.code}: {exc}"
            return block
    block["model"] = config.values_model
    try:
        result = extract_values(
            ws, docset_id, file_id, config=config, write_stats=write_stats, debug=debug
        )
    except DgmlError as exc:
        block["error"] = f"{exc.code}: {exc}"
        return block
    except Exception as exc:
        # Bug-grade exception from extraction (e.g. IndexError in a phase).
        # The assignment is already persisted, and clustering/bulk callers
        # need to emit their summary for the *other* files, so we soft-fail
        # here too — the error type/message lands in `extraction.error` and
        # the parent command exits 0.
        block["error"] = f"{type(exc).__name__}: {exc}"
        return block
    block["tool_calls"] = result.tool_calls
    return block


def add_file_and_extract(
    ws: Workspace,
    docset_id: str,
    file_id: str,
    *,
    config: GroundedConfig | None = None,
    write_stats: bool = True,
    debug: bool = False,
) -> dict[str, Any] | None:
    """Assign ``file_id`` to ``docset_id``; auto-extract if the DocSet has a schema.

    Returns the extraction block (same shape :func:`_auto_extract` produces)
    when the DocSet has a schema set, else ``None`` — the omit-when-no-schema
    policy matches the existing ``dgml docset add-file`` contract.

    ``write_stats`` and ``debug`` are forwarded to
    :func:`dgml_core.grounded.extract_values` — CLI callers pass
    ``args.debug`` for both, so ``extraction_stats.json`` and LLM usage rows
    are persisted only under ``--debug``.
    """
    store = DocSetStore(ws)
    store.add_file(docset_id, file_id)
    if not store.has_schema(docset_id):
        return None
    return _auto_extract(
        ws, docset_id, file_id, config=config, write_stats=write_stats, debug=debug
    )
