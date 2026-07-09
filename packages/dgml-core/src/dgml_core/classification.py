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

"""LLM-based auto-classification of newly added Files into DocSets.

When ``dgml file add --auto-classify`` is used, this module:

1. Loads the ``classification`` section of ``<workspace>/config.json``.
2. Gathers a small number of rendered page images from the new file plus
   the id/name/description of each existing DocSet.
3. Calls the configured vision LLM via :mod:`litellm`, forcing a choice
   between two tools: assign to an existing DocSet, or propose a new one.

The CLI treats every failure path here as a *soft fail*: the file record
is kept, ``classification.error`` is populated in the response payload,
and exit code stays 0. We surface failures by raising the exception types
below; the CLI layer is responsible for converting them to soft-fail
payload fields.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .docsets import DocSetStore
from .errors import (
    AuthError,
    ClassificationConfigInvalid,
    ClassificationConfigMissing,
    ClassificationFailed,
    CorruptMetadata,
)
from .llm import LLMConfig, call_with_tools
from .models import DocSet
from .storage import Workspace, read_config
from .usage import OPERATION_CLASSIFY
from .utils import gather_file_pages, image_to_data_url

DEFAULT_MAX_PAGES = 3

_TOOL_ASSIGN = "assign_to_existing_docset"
_TOOL_CREATE = "create_new_docset"

_NEW_DOCSET_INSTRUCTION_BULLETS = "\n".join(
    [
        "  - a short, document-type-specific name (2-5 words; prefer the "
        'document\'s own type, e.g. "Property Tax Bill" or "PILOT '
        'Agreement", not a topical bucket like "Property Tax Records"),',
        "  - a one-sentence description of what kind of document this is, and",
        "  - a list of 3-7 concrete questions answerable from the first "
        "pages of this kind of document. These define the DocSet for "
        "future classification — prefer specific, type-discriminating "
        "questions over generic ones.",
    ]
)


@dataclass(frozen=True)
class ClassificationConfig:
    """Parsed ``classification`` section of the workspace config.

    By construction this object is well-formed: :func:`load_classification_config`
    validates each field before returning.

    API key resolution: literal ``api_key`` > env-name lookup via
    ``api_key_env`` > litellm's per-provider default env var
    (``GEMINI_API_KEY`` for ``gemini/...``, etc.). Setting both
    ``api_key`` and ``api_key_env`` is a config error.
    """

    model: str
    max_pages: int = DEFAULT_MAX_PAGES
    api_key: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class ClassificationDecision:
    """The LLM's decision: assign to an existing DocSet, or create a new one.

    Exactly one of ``existing_docset_id`` or (``new_name``, ``new_description``,
    ``new_key_questions``) is populated. Validated at construction by
    :func:`classify_file`.
    """

    decision: str  # "existing" | "new"
    existing_docset_id: str | None = None
    new_name: str | None = None
    new_description: str | None = None
    new_key_questions: tuple[str, ...] = ()


def load_classification_config(workspace: Workspace) -> ClassificationConfig:
    """Read and validate the ``classification`` section of ``<workspace>/config.json``.

    Raises :class:`ClassificationConfigMissing` when no config file or no
    ``classification`` section is present; :class:`ClassificationConfigInvalid`
    when the section exists but is malformed.
    """
    if not workspace.config_path.exists():
        raise ClassificationConfigMissing(
            f"no config.json at {workspace.config_path}; "
            "auto-classification requires a workspace config with a 'classification' section"
        )

    try:
        data = read_config(workspace.config_path)
    except CorruptMetadata as exc:
        raise ClassificationConfigInvalid(
            f"{workspace.config_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ClassificationConfigInvalid(f"{workspace.config_path} must contain a JSON object")

    section = data.get("classification")
    if section is None:
        raise ClassificationConfigMissing(
            f"{workspace.config_path} has no 'classification' section"
        )
    if not isinstance(section, dict):
        raise ClassificationConfigInvalid("'classification' must be a JSON object")

    model = section.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ClassificationConfigInvalid(
            "'classification.model' must be a non-empty string "
            "(e.g. 'gemini/gemini-3.1-flash-lite')"
        )

    max_pages_raw = section.get("max_pages", DEFAULT_MAX_PAGES)
    if not isinstance(max_pages_raw, int) or isinstance(max_pages_raw, bool) or max_pages_raw < 1:
        raise ClassificationConfigInvalid(
            "'classification.max_pages' must be a positive integer if set"
        )

    api_key = section.get("api_key")
    if api_key is not None and (not isinstance(api_key, str) or not api_key):
        raise ClassificationConfigInvalid(
            "'classification.api_key' must be a non-empty string if set"
        )

    api_key_env = section.get("api_key_env")
    if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
        raise ClassificationConfigInvalid(
            "'classification.api_key_env' must be a non-empty env var name if set"
        )

    if api_key is not None and api_key_env is not None:
        raise ClassificationConfigInvalid(
            "set at most one of 'classification.api_key' / 'classification.api_key_env', not both"
        )

    return ClassificationConfig(
        model=model,
        max_pages=max_pages_raw,
        api_key=api_key,
        api_key_env=api_key_env,
    )


def classify_file(
    workspace: Workspace,
    file_id: str,
    *,
    config: ClassificationConfig,
    docsets: list[DocSet] | None = None,
    debug: bool = False,
) -> ClassificationDecision:
    """Ask the configured vision LLM to classify ``file_id`` into a DocSet.

    The LLM picks exactly one of two tools: assign the file to an existing
    DocSet, or propose a new one (name + description).

    ``docsets`` is the list of existing DocSets to classify against. When
    omitted it is read fresh from the workspace. Bulk callers (e.g.
    ``dgml file add <dir> --auto-classify``) pass an explicit list they
    maintain across the run — reading the store once and appending each
    newly-created DocSet — so per-file disk scans are avoided while
    DocSets created earlier in the run stay visible to later files.

    Raises :class:`ClassificationFailed` for any non-auth failure
    (missing images, malformed LLM response, network error).
    Raises :class:`AuthError` when ``config.api_key_env`` names an env var
    that isn't set.
    """
    if docsets is None:
        docsets = DocSetStore(workspace).list_all()
    response = _vision_tool_call(
        workspace,
        [file_id],
        config=config,
        prompt=_build_prompt(docsets),
        tools=_build_tools(docsets),
        debug=debug,
    )
    return _parse_response(response, docsets)


def propose_new_docset_for_files(
    workspace: Workspace,
    file_ids: list[str],
    *,
    config: ClassificationConfig,
    debug: bool = False,
) -> ClassificationDecision:
    """Ask the configured vision LLM to propose a new DocSet (name,
    description, and key questions) that ``file_ids`` should anchor.

    Unlike :func:`classify_file`, the caller has already decided a new
    DocSet is warranted — only the ``create_new_docset`` tool is offered,
    so the LLM doesn't have to choose between assign-vs-create. Pages
    from every file in ``file_ids`` are sent (up to ``config.max_pages``
    per file), so the LLM can name a DocSet anchored on the cluster as a
    whole rather than a single example. The caller is responsible for
    capping ``file_ids`` if cost/context is a concern.

    Same failure contract as :func:`classify_file`: raises
    :class:`ClassificationFailed` for missing SDK / no page images on
    any of the files / malformed response / provider error, and
    :class:`AuthError` when ``config.api_key_env`` is set but the env
    var isn't.
    """
    response = _vision_tool_call(
        workspace,
        file_ids,
        config=config,
        prompt=_build_prompt_new_only(),
        tools=[_create_new_docset_tool()],
        debug=debug,
    )
    name, args = _extract_single_tool_call(response)
    return _parse_new_docset_args(name, args)


def _vision_tool_call(
    workspace: Workspace,
    file_ids: list[str],
    *,
    config: ClassificationConfig,
    prompt: str,
    tools: list[dict[str, Any]],
    debug: bool = False,
) -> Any:
    """Call the configured vision LLM with ``prompt`` + the rendered page
    images of every file in ``file_ids`` (up to ``config.max_pages`` per
    file), forcing it to invoke exactly one of ``tools``. Returns the raw
    litellm response so callers can run their own tool-call parsing.

    Shared between :func:`classify_file` (always a single file) and
    :func:`propose_new_docset_for_files` (a cluster of files).
    """

    page_bytes: list[bytes] = []
    for fid in file_ids:
        page_bytes.extend(gather_file_pages(workspace, fid, config.max_pages))
    if not page_bytes:
        raise ClassificationFailed(
            f"no page images found for files {file_ids!r}; "
            "auto-classification requires successfully rendered pages"
        )

    api_key = _resolve_api_key(config)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in page_bytes:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(img)}})

    # The call records its own usage row (gated on --debug) from the context
    # carried on the config; no wrapper needed for this single call.
    llm_config = LLMConfig(
        model=config.model,
        api_key=api_key,
        max_tokens=None,
        workspace=workspace,
        debug=debug,
        operation=OPERATION_CLASSIFY,
        context={"file_ids": file_ids},
    )
    try:
        result = call_with_tools(
            llm_config,
            messages=[{"role": "user", "content": content}],
            tools=tools,
            tool_choice="required",
        )
    except Exception as exc:
        # litellm normalizes provider errors but we never want a raw
        # provider exception bubbling past — wrap unconditionally with
        # the type name so the CLI's soft-fail message stays informative.
        raise ClassificationFailed(f"LLM call failed: {type(exc).__name__}: {exc}") from exc
    return result.response


def _resolve_api_key(config: ClassificationConfig) -> str | None:
    """Resolve the API key.

    Precedence: literal ``config.api_key`` > env-name lookup via
    ``config.api_key_env`` > ``None`` (litellm falls back to its own
    per-provider env var). Mutual exclusion of the two config fields
    is enforced upstream in :func:`load_classification_config`.
    """
    if config.api_key:
        return config.api_key
    if not config.api_key_env:
        return None
    key = os.environ.get(config.api_key_env)
    if not key:
        raise AuthError(
            f"environment variable ${config.api_key_env} is not set "
            "(referenced by classification.api_key_env in config.json)"
        )
    return key


def _build_prompt_new_only() -> str:
    """Prompt for :func:`propose_new_docset_for_files`. The caller has already
    decided a new DocSet is needed; the LLM is just being asked to name and
    describe it."""
    return "\n".join(
        [
            "You are proposing a DocSet — a named grouping of semantically "
            "similar documents in a DGML workspace — anchored on a newly "
            "ingested document.",
            "",
            "The rendered first pages of the new file are attached as images.",
            "",
            f"Call `{_TOOL_CREATE}` with:",
            _NEW_DOCSET_INSTRUCTION_BULLETS,
        ]
    )


def _build_prompt(docsets: list[DocSet]) -> str:
    lines = [
        "You are classifying a newly ingested document into a DocSet.",
        "",
        "A DocSet groups documents of the **same document type** — documents "
        "that could plausibly share a single extraction schema. Two documents "
        "belong in the same DocSet if, and only if, the same set of "
        'structured questions ("what is X?", "when did Y happen?") could be '
        "answered from each of them.",
        "",
        "Topical similarity is NOT enough. A property tax bill and a tax "
        "abatement (PILOT) agreement both concern property taxes, but they "
        "answer different questions (tax owed vs. abatement terms), so they "
        "belong in **different** DocSets. Use the document type, not the topic.",
        "",
        "The rendered first pages of the new file are attached as images.",
        "",
    ]
    if docsets:
        lines.append("Existing DocSets:")
        for ds in docsets:
            lines.append(f"- id={ds.id}")
            lines.append(f"  name: {ds.name}")
            if ds.description:
                lines.append(f"  description: {ds.description}")
            if ds.key_questions:
                lines.append("  key questions this DocSet's documents answer:")
                for q in ds.key_questions:
                    lines.append(f"    - {q}")
    else:
        lines.append("There are no existing DocSets in this workspace yet.")
    lines.extend(
        [
            "",
            f"Call `{_TOOL_ASSIGN}` only if the new file's first pages plausibly "
            "answer the same key questions as one of the existing DocSets above "
            "(i.e. a single extraction schema would work for both). Otherwise "
            f"call `{_TOOL_CREATE}` with:",
            "  - a short, document-type-specific name (2-5 words; prefer the "
            'document\'s own type, e.g. "Property Tax Bill" or "PILOT '
            'Agreement", not a topical bucket like "Property Tax Records"),',
            "  - a one-sentence description of what kind of document this is, and",
            "  - a list of 3-7 concrete questions answerable from the first "
            "pages of this kind of document. These define the DocSet for "
            "future classification — prefer specific, type-discriminating "
            "questions over generic ones.",
            "",
            "Call exactly one tool.",
        ]
    )
    return "\n".join(lines)


def _build_tools(docsets: list[DocSet]) -> list[dict[str, Any]]:
    valid_ids = [ds.id for ds in docsets]
    assign_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "docset_id": {
                "type": "string",
                "description": "The id of the existing DocSet that best fits the new file.",
            }
        },
        "required": ["docset_id"],
        "additionalProperties": False,
    }
    if valid_ids:
        assign_schema["properties"]["docset_id"]["enum"] = valid_ids

    return [
        {
            "type": "function",
            "function": {
                "name": _TOOL_ASSIGN,
                "description": "Assign the new file to one of the existing DocSets.",
                "parameters": assign_schema,
            },
        },
        _create_new_docset_tool(),
    ]


def _create_new_docset_tool() -> dict[str, Any]:
    """Litellm tool schema for proposing a new DocSet (name + description).

    Used by :func:`classify_file` as one of two tool options, and by
    :func:`propose_new_docset_for_files` as the only tool option.
    """
    return {
        "type": "function",
        "function": {
            "name": _TOOL_CREATE,
            "description": (
                "Create a new DocSet for this file when no existing DocSet fits. "
                "The new DocSet should describe a single document type — "
                "documents that could share one extraction schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Short document-type name (2-5 words). Prefer the "
                            "document's own type (e.g. 'Property Tax Bill', "
                            "'PILOT Agreement') over topical buckets "
                            "(e.g. 'Property Tax Records')."
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "One sentence describing what kind of document this DocSet groups."
                        ),
                    },
                    "key_questions": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 7,
                        "items": {"type": "string"},
                        "description": (
                            "3-7 concrete, type-discriminating questions "
                            "answerable from the first pages of this kind "
                            "of document. These will be shown to future "
                            "classifications to decide whether new files "
                            "belong in this DocSet, so prefer specific "
                            "questions over generic ones."
                        ),
                    },
                },
                "required": ["name", "description", "key_questions"],
                "additionalProperties": False,
            },
        },
    }


def _parse_response(response: Any, docsets: list[DocSet]) -> ClassificationDecision:
    """Parse the LLM's tool call into a :class:`ClassificationDecision`."""
    name, args = _extract_single_tool_call(response)

    if name == _TOOL_ASSIGN:
        docset_id = args.get("docset_id")
        if not isinstance(docset_id, str) or not docset_id.strip():
            raise ClassificationFailed(f"{_TOOL_ASSIGN} call missing a non-empty 'docset_id'")
        valid_ids = {ds.id for ds in docsets}
        if valid_ids and docset_id not in valid_ids:
            raise ClassificationFailed(
                f"{_TOOL_ASSIGN} returned unknown docset_id {docset_id!r}; "
                f"valid ids: {sorted(valid_ids)}"
            )
        return ClassificationDecision(decision="existing", existing_docset_id=docset_id)

    if name == _TOOL_CREATE:
        return _parse_new_docset_args(name, args)

    raise ClassificationFailed(f"LLM returned unexpected tool name: {name!r}")


def _extract_single_tool_call(response: Any) -> tuple[str | None, dict[str, Any]]:
    """Pull the single ``(tool_name, arguments)`` pair out of a litellm
    response made with ``tool_choice="required"``.

    litellm returns OpenAI-compatible objects regardless of provider, so the
    attribute path ``response.choices[0].message.tool_calls[0].function`` is
    stable across Claude / GPT-4o / Gemini.

    Raises :class:`ClassificationFailed` if the response is malformed or
    has no tool calls.
    """
    try:
        choices = response.choices
        message = choices[0].message
        tool_calls = message.tool_calls
    except (AttributeError, IndexError, TypeError) as exc:
        raise ClassificationFailed(
            f"LLM response missing tool_calls: {type(exc).__name__}: {exc}"
        ) from exc

    if not tool_calls:
        raise ClassificationFailed("LLM response contained no tool calls")

    call = tool_calls[0]
    name = getattr(getattr(call, "function", None), "name", None)
    raw_args = getattr(getattr(call, "function", None), "arguments", None)
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
    except (json.JSONDecodeError, TypeError) as exc:
        raise ClassificationFailed(f"LLM tool-call arguments not valid JSON: {exc}") from exc
    return name, args


def _parse_new_docset_args(name: str | None, args: dict[str, Any]) -> ClassificationDecision:
    """Validate a ``create_new_docset`` tool call and return a
    ``ClassificationDecision`` with ``decision="new"`` populated.
    Reused by :func:`_parse_response` (after dispatching on tool name) and
    :func:`propose_new_docset_for_files` (where ``create_new_docset`` is the
    only valid tool, so we also reject any other tool name here).
    """
    if name == _TOOL_CREATE:
        new_name = args.get("name")
        new_description = args.get("description")
        raw_questions = args.get("key_questions")
        if not isinstance(new_name, str) or not new_name.strip():
            raise ClassificationFailed(f"{_TOOL_CREATE} call missing a non-empty 'name'")
        if not isinstance(new_description, str):
            raise ClassificationFailed(f"{_TOOL_CREATE} call missing a string 'description'")
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ClassificationFailed(
                f"{_TOOL_CREATE} call missing a non-empty 'key_questions' array"
            )
        cleaned_questions: list[str] = []
        for q in raw_questions:
            if not isinstance(q, str):
                raise ClassificationFailed(
                    f"{_TOOL_CREATE} 'key_questions' must be strings; got {type(q).__name__}"
                )
            stripped = q.strip()
            if stripped:
                cleaned_questions.append(stripped)
        if not cleaned_questions:
            raise ClassificationFailed(
                f"{_TOOL_CREATE} 'key_questions' contained no non-empty entries"
            )
        return ClassificationDecision(
            decision="new",
            new_name=new_name.strip(),
            new_description=new_description.strip(),
            new_key_questions=tuple(cleaned_questions),
        )
    raise ClassificationFailed(f"LLM returned unexpected tool name: {name!r}")
