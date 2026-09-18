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

1. Loads the ``classification`` section of ``<workspace>/config.toml``.
2. Gathers a small number of rendered page images from the new file plus
   the id/name/description of each existing DocSet.
3. Calls the configured vision LLM via :mod:`litellm` with
   ``tool_choice="required"``. What it may choose depends on the mode: by
   default (``existing-or-new``) it picks between assign-to-existing and
   propose-a-new-one; in ``existing`` mode only the assign tool is offered,
   so the LLM must place the file in the best-fitting existing DocSet even
   when the fit is imperfect. That mode skips step 3 entirely when the
   workspace holds a single DocSet — the answer is already determined.

The CLI treats every failure path here as a *soft fail*: the file record
is kept, ``classification.error`` is populated in the response payload,
and exit code stays 0. We surface failures by raising the exception types
below; the CLI layer is responsible for converting them to soft-fail
payload fields.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from .config import load_merged_config
from .docsets import DocSetStore
from .errors import (
    AuthError,
    ClassificationConfigInvalid,
    ClassificationConfigMissing,
    ClassificationFailed,
    NoExistingDocSets,
)
from .llm import LLMConfig, call_with_tools
from .models import DocSet
from .models_config import ConfigSection, Tier, resolve_tiered_model
from .storage import Workspace
from .usage import OPERATION_CLASSIFY
from .utils import gather_file_pages, image_to_data_url

DEFAULT_MAX_PAGES = 3
DEFAULT_NAMING_ATTEMPTS = 1

_TOOL_ASSIGN = "assign_to_existing_docset"
_TOOL_CREATE = "create_new_docset"


class ClassifyMode(StrEnum):
    """Which DocSets auto-classification is allowed to route a file into.

    ``EXISTING_OR_NEW`` is the historical (and default) behavior: assign to an
    existing DocSet when one fits, otherwise create one.

    ``EXISTING`` restricts the LLM to DocSets that already exist and requires
    it to pick one — the best available, even when the fit is imperfect. It
    never creates a DocSet and never declines, so every file lands somewhere.
    That makes it the right mode only when the caller already knows the files
    belong in the workspace's existing DocSets: given an off-type document it
    will produce a confident wrong answer rather than flagging it. With no
    DocSets to choose from there is no outcome it could produce, so it raises
    :class:`~dgml_core.errors.NoExistingDocSets`.
    """

    EXISTING = "existing"
    EXISTING_OR_NEW = "existing-or-new"


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

    ``naming_attempts`` is the default ``attempts`` for
    :func:`propose_new_docset_for_files` — how many independent proposals to
    request before returning the modal one. It costs tokens linearly, so it
    stays at 1 unless the workspace opts in.
    """

    model: str
    max_pages: int = DEFAULT_MAX_PAGES
    api_key: str | None = None
    api_key_env: str | None = None
    api_base: str | None = None
    naming_attempts: int = DEFAULT_NAMING_ATTEMPTS


@dataclass(frozen=True)
class ClassificationDecision:
    """The LLM's decision: assign to an existing DocSet, or create a new one.

    Exactly one of ``existing_docset_id`` or (``new_name``, ``new_description``,
    ``new_key_questions``) is populated. Validated at construction by
    :func:`classify_file`. Under :class:`ClassifyMode.EXISTING` only the former
    is reachable — that mode always assigns.

    ``confidence`` is populated only by the multi-attempt path — see
    :func:`propose_new_docset_for_files` with ``attempts >= 2``. It is the share
    of independent attempts that landed on the *returned* name, in ``(0, 1]``:
    an ordinal robustness signal about the naming, not a calibrated probability,
    and unrelated to how confident the clusterer was about the grouping.
    """

    decision: str  # "existing" | "new"
    existing_docset_id: str | None = None
    new_name: str | None = None
    new_description: str | None = None
    new_key_questions: tuple[str, ...] = ()
    confidence: float | None = None


def load_classification_config(workspace: Workspace) -> ClassificationConfig:
    """Resolve the classification model (``classification.model`` override →
    ``[models].light`` tier) and its credentials from the merged config.

    Raises :class:`ClassificationConfigMissing` when neither the override nor a
    tier names a model; :class:`ClassificationConfigInvalid` when the section is
    malformed.
    """
    merged = load_merged_config(workspace)
    rm = resolve_tiered_model(
        merged,
        section_name=ConfigSection.CLASSIFICATION,
        tier=Tier.LIGHT,
        invalid=ClassificationConfigInvalid,
        missing=ClassificationConfigMissing,
    )

    section = merged.get(ConfigSection.CLASSIFICATION)
    sec: dict[str, Any] = section if isinstance(section, dict) else {}
    max_pages_raw = sec.get("max_pages", DEFAULT_MAX_PAGES)
    if not isinstance(max_pages_raw, int) or isinstance(max_pages_raw, bool) or max_pages_raw < 1:
        raise ClassificationConfigInvalid(
            "'classification.max_pages' must be a positive integer if set"
        )

    attempts_raw = sec.get("naming_attempts", DEFAULT_NAMING_ATTEMPTS)
    if not isinstance(attempts_raw, int) or isinstance(attempts_raw, bool) or attempts_raw < 1:
        raise ClassificationConfigInvalid(
            "'classification.naming_attempts' must be a positive integer if set"
        )

    return ClassificationConfig(
        model=rm.model,
        max_pages=max_pages_raw,
        api_key=rm.api_key,
        api_key_env=rm.api_key_env,
        api_base=rm.api_base,
        naming_attempts=attempts_raw,
    )


def classify_file(
    workspace: Workspace,
    file_id: str,
    *,
    config: ClassificationConfig,
    docsets: list[DocSet] | None = None,
    allow_new: bool = True,
    debug: bool = False,
) -> ClassificationDecision:
    """Ask the configured vision LLM to classify ``file_id`` into a DocSet.

    The LLM picks exactly one of two tools: assign the file to an existing
    DocSet, or propose a new one (name + description).

    ``allow_new=False`` (:class:`ClassifyMode.EXISTING`) offers only the assign
    tool, so the decision is always ``"existing"``: the LLM must return the
    best-fitting DocSet even when the fit is imperfect. Use it only when the
    file is known to belong in one of them — nothing here detects an off-type
    document, it just picks the least-bad home for it.

    With exactly one DocSet that mode has only one answer available, so the
    LLM is not called at all.

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
    Raises :class:`NoExistingDocSets` when ``allow_new=False`` and the
    workspace has no DocSets — there is no decision to make, and the assign
    tool would have no valid id to enumerate.
    """
    if docsets is None:
        docsets = DocSetStore(workspace).list_all()
    if not allow_new:
        if not docsets:
            raise NoExistingDocSets(
                "no DocSets to assign to; create one first, or allow "
                "classification to propose a new DocSet"
            )
        if len(docsets) == 1:
            # Only one possible answer, so there is nothing to decide.
            return ClassificationDecision(decision="existing", existing_docset_id=docsets[0].id)
    response = _vision_tool_call(
        workspace,
        [file_id],
        config=config,
        prompt=_build_prompt(docsets, allow_new=allow_new),
        tools=_build_tools(docsets, allow_new=allow_new),
        debug=debug,
    )
    return _parse_response(response, docsets, allow_new=allow_new)


def propose_new_docset_for_files(
    workspace: Workspace,
    file_ids: list[str],
    *,
    config: ClassificationConfig,
    debug: bool = False,
    attempts: int | None = None,
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

    ``attempts`` buys a robustness signal at a linear cost in tokens. When
    omitted it falls back to ``config.naming_attempts``, so a workspace can turn
    the signal on for every cluster without any caller passing it; an explicit
    argument still wins. At ``1`` a single call is made and the decision carries
    no ``confidence``. With ``attempts >= 2`` the proposal is requested that many
    times independently, and the **modal** proposal is returned — the one the
    plurality of attempts agreed on, not whichever came back first — carrying a
    ``confidence`` equal to that plurality's share. A name three attempts out of
    three settled on is one you can create without a human looking; a 2-1 split
    is a coin toss worth surfacing.

    Same failure contract as :func:`classify_file`: raises
    :class:`ClassificationFailed` for missing SDK / no page images on
    any of the files / malformed response / provider error, and
    :class:`AuthError` when ``config.api_key_env`` is set but the env
    var isn't. With ``attempts >= 2`` an individual attempt is allowed to fail:
    agreement is computed over the attempts that succeeded, and the error is
    re-raised only if *every* attempt failed. Otherwise asking for more opinions
    would multiply the exposure to one transient provider hiccup, making the
    robustness feature less robust than not using it.
    """
    if attempts is None:
        attempts = config.naming_attempts
    if attempts < 1:
        raise ValueError(f"attempts must be at least 1, got {attempts}")

    decisions: list[ClassificationDecision] = []
    first_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = _vision_tool_call(
                workspace,
                file_ids,
                config=config,
                prompt=_build_prompt_new_only(),
                tools=[_create_new_docset_tool()],
                debug=debug,
            )
            name, args = _extract_single_tool_call(response)
            decisions.append(_parse_new_docset_args(name, args))
        except (ClassificationFailed, AuthError) as exc:
            if attempts == 1:
                raise
            first_error = first_error or exc

    if not decisions:
        assert first_error is not None  # attempts >= 1, so we either got one or failed
        raise first_error
    if attempts == 1:
        return decisions[0]
    return _modal_decision(decisions)


def _modal_decision(decisions: list[ClassificationDecision]) -> ClassificationDecision:
    """The proposal the plurality of ``decisions`` agreed on, plus its share.

    Names are compared case- and whitespace-insensitively, so "PILOT Agreement"
    and "pilot   agreement" count as agreement. Ties go to the earliest attempt.
    The share is over the attempts that produced a name — which, since
    :func:`_parse_new_docset_args` rejects a proposal without one, is every
    attempt that did not fail outright.
    """
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[_normalize_name(decision.new_name)] = (
            counts.get(_normalize_name(decision.new_name), 0) + 1
        )
    winner = max(counts, key=lambda key: counts[key])
    # The *returned* decision has to be one from the winning group, or the
    # reported confidence would describe a name we didn't return.
    modal = next(d for d in decisions if _normalize_name(d.new_name) == winner)
    return replace(modal, confidence=counts[winner] / len(decisions))


def _normalize_name(name: str | None) -> str:
    """A name reduced to what matters for agreement: lowercase, single-spaced."""
    return " ".join((name or "").lower().split())


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
        api_base=config.api_base,
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
            "(referenced by classification.api_key_env in the config)"
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


def _build_prompt(docsets: list[DocSet], *, allow_new: bool = True) -> str:
    lines = [
        "You are classifying a newly ingested document into a DocSet.",
        "",
        "A DocSet groups documents of the **same document type** — documents "
        "that could plausibly share a single extraction schema. Two documents "
        "belong in the same DocSet if, and only if, the same set of "
        'structured questions ("what is X?", "when did Y happen?") could be '
        "answered from each of them.",
        "",
    ]
    if allow_new:
        lines.append(
            "Topical similarity is NOT enough. A property tax bill and a tax "
            "abatement (PILOT) agreement both concern property taxes, but they "
            "answer different questions (tax owed vs. abatement terms), so they "
            "belong in **different** DocSets. Use the document type, not the topic."
        )
    else:
        # Same rubric, but as a matter of degree: this mode has no
        # create-a-DocSet option, so the strict gate above would only tell the
        # LLM to refuse a choice it is required to make.
        lines.append(
            "Treat that as a matter of degree, not a pass/fail gate: you will "
            "be asked to choose the closest DocSet from a fixed list, so judge "
            "by document type rather than by topic. A property tax bill and a "
            "tax abatement (PILOT) agreement both concern property taxes but "
            "answer different questions, so a DocSet of one is a poor home for "
            "the other — prefer a DocSet whose own questions the new document "
            "actually answers."
        )
    lines.extend(
        [
            "",
            "The rendered first pages of the new file are attached as images.",
            "",
        ]
    )
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
    lines.append("")
    if allow_new:
        lines.append(
            f"Call `{_TOOL_ASSIGN}` only if the new file's first pages plausibly "
            "answer the same key questions as one of the existing DocSets above "
            "(i.e. a single extraction schema would work for both). Otherwise "
            f"call `{_TOOL_CREATE}` with:"
        )
        lines.append(_NEW_DOCSET_INSTRUCTION_BULLETS)
    else:
        lines.append(
            f"Call `{_TOOL_ASSIGN}` with the existing DocSet that **best** "
            "fits the new file. You must choose one: there is no option to "
            "create a DocSet and no option to decline. A perfect fit is not "
            "required — if none of them matches the new file's document type "
            "exactly, pick whichever is closest rather than refusing. Apply "
            "the criterion above: prefer the DocSet whose key questions the "
            "new file can best answer."
        )
    lines.extend(["", "Call exactly one tool."])
    return "\n".join(lines)


def _build_tools(docsets: list[DocSet], *, allow_new: bool = True) -> list[dict[str, Any]]:
    docset_id_schema: dict[str, Any] = {
        "type": "string",
        "description": "The id of the existing DocSet that best fits the new file.",
    }
    if docsets:
        docset_id_schema["enum"] = [ds.id for ds in docsets]

    if allow_new:
        assign_description = "Assign the new file to one of the existing DocSets."
    else:
        assign_description = (
            "Assign the new file to whichever existing DocSet fits it best. "
            "This is the only available action, so a choice is required even "
            "when no DocSet is a perfect fit."
        )

    assign_tool = {
        "type": "function",
        "function": {
            "name": _TOOL_ASSIGN,
            "description": assign_description,
            "parameters": {
                "type": "object",
                "properties": {"docset_id": docset_id_schema},
                "required": ["docset_id"],
                "additionalProperties": False,
            },
        },
    }
    # In assign-only mode this is the *sole* tool, which is what forces a
    # choice under tool_choice="required".
    return [assign_tool, _create_new_docset_tool()] if allow_new else [assign_tool]


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


def _parse_response(
    response: Any, docsets: list[DocSet], *, allow_new: bool = True
) -> ClassificationDecision:
    """Parse the LLM's tool call into a :class:`ClassificationDecision`.

    With ``allow_new=False`` a ``create_new_docset`` call is rejected rather
    than honored: the tool wasn't offered, so acting on it would create the
    DocSet the caller explicitly ruled out.
    """
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
        if not allow_new:
            raise ClassificationFailed(
                f"LLM called {_TOOL_CREATE}, which was not offered; "
                f"{_TOOL_ASSIGN} is the only tool available in "
                f"'{ClassifyMode.EXISTING}' mode"
            )
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
