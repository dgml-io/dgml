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

"""Pass A — verbatim transcription into flat typed blocks.

Per document: disjoint page windows, one LLM call each. The model is always
told to emit the compact tab-delimited ("TL") line format; the reply parser
still content-sniffs, so legacy JSON replies (e.g. cached ``*_wNN_raw.json``
artifacts from older runs) keep parsing. Merging is list concatenation; a
window that starts mid-sentence returns the remainder in `continues`, which
is appended to the previous window's last text block.
"""

from __future__ import annotations

import dataclasses
import json
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dgml_core import llm
from dgml_core.generation import coverage, document
from dgml_core.generation import lines as lines_mod
from dgml_core.generation.blocks import (
    Block,
    Span,
    anchor_heading_levels,
    normalize_enumerated_paragraphs,
    parse_block,
    xml_safe_block_dict,
)
from dgml_core.generation.prompts import get as prompt
from dgml_core.pages import PdfConfig, pdf_page_count
from dgml_core.utils import xml_safe

_UNSAFE_FNAME_RE = re.compile(r'[<>:"/\\|?*]')


def _count_pages(pdf_bytes: bytes) -> int:
    """Page count for an in-memory PDF, via ``pages.pdf_page_count``.

    It takes a path, so the bytes are spilled to a short-lived temp file.
    Written with ``delete=False`` and closed before ``pdf_page_count`` reopens
    it by path: Windows refuses a second handle on a file while the first
    (the still-open ``NamedTemporaryFile``) holds it.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        return pdf_page_count(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def cache_write(
    cache_dir: Path | str | None, name: str, content: str, *, debug: bool = True
) -> None:
    """Write one cache artifact into *cache_dir* (no-op when *cache_dir* is unset).

    *debug* gates the write so debug-only artifacts can share this function:

    - The *functional* cache files the next ``docset generate`` run reloads
      (``<stem>_blocks.json``, ``label_<stem>_cNN_raw.json``,
      ``concept_roster.json``) call with the default ``debug=True`` and are
      always written when a cache dir is set.
    - *Debug-only* artifacts (raw model returns, intermediate XML renders,
      prompt listings) — never read back, kept only for tracing — pass
      ``debug=<the --debug flag>`` so they are emitted only under ``--debug``.
    """
    if not debug or cache_dir is None:
        return
    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / _UNSAFE_FNAME_RE.sub("_", name)).write_text(content, encoding="utf-8")


def blocks_to_json(blocks: list[Block]) -> str:
    """Serialize blocks (with labels/entities) for snapshot files."""
    return json.dumps([dataclasses.asdict(b) for b in blocks], indent=2, ensure_ascii=False)


def _load_cached_blocks(cache_dir: Path | str | None, doc_name: str) -> list[Block] | None:
    """Reload a document's transcription from ``<stem>_blocks.json`` if present.

    The blocks cache is written right after transcription (pre-labeling), so
    reloading it replays Pass A exactly: a re-run whose outputs were removed
    (or that crashed after transcription) re-labels and re-renders without
    paying for transcription again. Delete the file to force a fresh Pass A.
    ``None`` (no/unreadable cache) means transcribe normally.
    """
    if cache_dir is None:
        return None
    stem = _UNSAFE_FNAME_RE.sub("_", f"{Path(doc_name).stem}_blocks.json")
    blocks_file = Path(cache_dir) / stem
    if not blocks_file.exists():
        return None
    try:
        raw_blocks = json.loads(blocks_file.read_text(encoding="utf-8"))
        # xml_safe_block_dict, not just Block(**b): a cache predating the
        # sanitizer (or written by anything that bypasses parse_block) can hold
        # a character XML cannot represent, and the re-render this feeds runs
        # outside the per-file error boundary — one such character would abort
        # the whole docset rather than fail its own document.
        return [
            Block(
                **{
                    **(safe := xml_safe_block_dict(b)),
                    "entities": [Span(**sp) for sp in safe.get("entities", [])],
                }
            )
            for b in raw_blocks
        ]
    except (json.JSONDecodeError, TypeError, ValueError):
        return None  # unreadable cache — fall through to a fresh transcription


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def strip_fences(text: str) -> str:
    """Unwrap a ```json … ``` code fence so cached raw JSON is valid JSON.

    Returns the fenced body when a complete fence is present; otherwise returns
    the text unchanged (stripped). Truncated/damaged output — which has no
    closing fence — is left as-is so failures stay inspectable in the cache.
    """
    match = _JSON_FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


# Pass-A transcription always instructs the model to emit the compact
# tab-delimited ("TL") line grammar. The reply parser still content-sniffs
# (:func:`parse_window_any`), so legacy JSON replies — including cached raw
# ``*_wNN_raw.json`` artifacts from older runs — keep parsing unchanged.
SYSTEM_PROMPT = prompt("transcribe_system_compact")
SYSTEM_PROMPT_LINES = prompt("transcribe_system_lines")


def _heading_breadcrumb(blocks: list[Block]) -> list[str]:
    """The chain of still-open headings (a level stack, like ``build_tree``)."""
    stack: list[tuple[int, str]] = []
    for b in blocks:
        if b.structure == "heading":
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            label = f"{b.lim} {b.text}".strip() if b.lim else b.text
            if label:
                stack.append((b.level, label))
    return [label for _, label in stack]


def _open_table_header(blocks: list[Block]) -> list[str]:
    """Cells of the first row of a table still open at the window's end, else []."""
    if not blocks or blocks[-1].structure != "row":
        return []
    i = len(blocks) - 1
    while i > 0 and blocks[i - 1].structure == "row":
        i -= 1
    return [c for c in blocks[i].cells if c]


def _window_context(blocks: list[Block], tail_chars: int = 300) -> str:
    """Structural continuity for the next window: section path, an open table's
    columns, and the trailing text — richer than a bare text tail so a clause or
    table spanning a page seam keeps its heading levels and column model."""
    if not blocks:
        return ""
    parts: list[str] = []
    crumb = _heading_breadcrumb(blocks)
    if crumb:
        parts.append("Section path: " + " > ".join(crumb))
    header = _open_table_header(blocks)
    if header:
        parts.append("Open table columns: " + " | ".join(header))
    tail = blocks[-1].flat_text()[-tail_chars:]
    if tail:
        parts.append(f'Ends with: "{tail}"')
    return "\n".join(parts)


def _window_instruction(first_page: int, last_page: int, total: int, context: str) -> str:
    parts = [
        prompt("transcribe_window_header").format(
            first=first_page + 1, last=last_page + 1, total=total
        )
    ]
    if context:
        parts.append(prompt("transcribe_window_context").format(context=context))
    parts.append(prompt("transcribe_window_compact"))
    return "\n\n".join(parts)


def _lines_instruction(
    first_page: int, last_page: int, total: int, context: str, listing: str
) -> str:
    parts = [
        prompt("transcribe_lines_header").format(
            first=first_page + 1,
            last=last_page + 1,
            total=total,
            n_lines=listing.count("\nL") + (1 if listing.startswith("L") else 0),
        )
    ]
    if context:
        parts.append(prompt("transcribe_window_context").format(context=context))
    parts.append("Numbered lines:\n" + listing)
    return "\n\n".join(parts)


def _escape_inner_quotes(text: str) -> str:
    # Escape quotes inside string values; verbatim text copies them unescaped.
    out: list[str] = []
    in_str, i, n = False, 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\" and in_str:
            out.append(text[i : i + 2])
            i += 2
            continue
        if c == '"':
            if not in_str:
                in_str = True
            else:
                j = i + 1
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j >= n or text[j] in ":,}]":
                    in_str = False
                else:
                    out.append("\\")
        out.append(c)
        i += 1
    return "".join(out)


def loads_tolerant(text: str) -> Any:
    """Decode a model's JSON reply, tolerating the ways models bend the grammar.

    In order: strict; then the outermost ``{...}`` span, dropping any prose the
    model wrapped around it; then ``strict=False``, which permits literal
    control characters inside strings; then quote-escaping.

    ``strict=False`` earns its place on quoted source text. Asked to echo a
    span of a document, a model reproduces it with its line breaks intact —

        {"quote": "Acme Holdings Ltd\n1 Example Way\nSpringfield"}

    with a real newline rather than the ``\\n`` escape RFC 8259 requires. The
    JSON is otherwise well-formed, so rejecting it discards a whole chunk's
    labels over a character the model had no reason to think mattered.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                text = text[start : end + 1]
        try:
            return json.loads(text, strict=False)
        except json.JSONDecodeError:
            return json.loads(_escape_inner_quotes(text))


def _parse_window_json(raw: str) -> dict[str, Any]:
    match = _JSON_FENCE_RE.search(raw)
    cleaned = (match.group(1) if match else raw).strip()
    out = loads_tolerant(cleaned)
    return out if isinstance(out, dict) else {}


def _salvage_window_json(raw: str) -> dict[str, Any] | None:
    """Recover the complete blocks from a truncated window reply.

    A length/stream truncation leaves the trailing block object incomplete and
    the whole JSON unparseable, which would otherwise discard the entire window.
    Decode whole block objects from inside the ``blocks`` array until the first
    incomplete one and keep them, dropping only that partial tail. Returns
    ``None`` if nothing parseable can be recovered.
    """
    match = _JSON_FENCE_RE.search(raw)
    text = (match.group(1) if match else raw).strip()
    start = text.find('"blocks"')
    start = text.find("[", start) if start != -1 else -1
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    blocks: list[Any] = []
    pos, end = start + 1, len(text)
    while pos < end:
        while pos < end and text[pos] in " \n\r\t,":
            pos += 1
        if pos >= end or text[pos] == "]":
            break
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break  # the truncated trailing object — stop, keep what's complete
        blocks.append(obj)
    return {"continues": "", "blocks": blocks} if blocks else None


# Compact ("TL") wire format — sigil-prefixed tab-delimited lines (H/P/I/R/F/O
# plus a leading C continues-line), the 4-sequence escape set (``\\ \t \n \r``)
# and ``\*`` for an option whose printed text starts with the checked-marker
# asterisk. See ``transcribe_system_compact`` in prompts.yaml for the grammar
# the model is shown.
_TL_UNESCAPE_RE = re.compile(r"\\([\\tnr*])")
_TL_UNESCAPE_MAP = {"\\": "\\", "t": "\t", "n": "\n", "r": "\r", "*": "*"}
_HEADING_SIGIL_RE = re.compile(r"^H(\d*)$")
# The grammar in prompts.yaml writes the separator as the placeholder
# ``<TAB>``, and models sometimes copy the placeholder instead of emitting a
# real tab, for a whole window at a time. Such a line has no real tab, so
# splitting it on tabs yields an unknown sigil and the line would be dropped.
_TL_TAB_PLACEHOLDER = "<TAB>"
_TL_PLACEHOLDER_LINE_RE = re.compile(r"^(?:H\d*|[PIRFOC])<TAB>")


def _tl_unescape(text: str) -> str:
    """Decode the wire escapes (``\\\\ \\t \\n \\r \\*``); anything else is literal."""
    return _TL_UNESCAPE_RE.sub(lambda m: _TL_UNESCAPE_MAP[m.group(1)], text)


def _tl_field(rest: list[str], i: int) -> str:
    """Field *i* of a split line, unescaped ('' when absent)."""
    return _tl_unescape(rest[i]) if i < len(rest) else ""


def _tl_tail(rest: list[str], i: int) -> str:
    """Fields *i*.. re-joined with tabs, then unescaped.

    The free-text field is the LAST field on its line, so a model-emitted
    unescaped tab inside it is re-joined rather than truncating the text.
    """
    return _tl_unescape("\t".join(rest[i:]))


def _parse_window_compact(text: str) -> tuple[dict[str, Any], int]:
    """Decode a tab-delimited Pass-A reply into the JSON-shaped window payload.

    Grammar (fields separated by real tabs; one block per line)::

        C<TAB>text                       — continues-preamble (first line)
        H<level><TAB>lim<TAB>text        — heading, level 1-6 ("H2")
        P<TAB>text                       — paragraph
        I<TAB>lim<TAB>text               — list item
        R<TAB>cell<TAB>cell…             — table row
        F<TAB>lim<TAB>label<TAB>value    — label/value form line
        O<TAB>lim<TAB>label<TAB>value<TAB>opt… — choice group; checked opts
                                                 prefixed "*" (literal "\\*")

    A line that uses the literal placeholder ``<TAB>`` where a real tab
    belongs (and holds no real tab) is decoded as if the placeholders were
    tabs.

    Per-line tolerant: a malformed line (unknown sigil, a half-written
    truncation tail) is dropped and counted — the natural analogue of
    ``_salvage_window_json``, which likewise keeps the complete blocks and
    drops the partial trailing one. Returns ``(payload, dropped)`` where
    *payload* is exactly the ``{"continues": str, "blocks": [dict, ...]}``
    shape that ``parse_block`` (and ``_payload_text`` / ``_window_recall`` /
    ``_payload_tail`` / ``_merge_payloads``) consume today.
    """
    continues = ""
    blocks: list[dict[str, Any]] = []
    dropped = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        if "\t" not in line and _TL_PLACEHOLDER_LINE_RE.match(line):
            # Only a line that would otherwise be dropped is repaired: a line
            # with a real tab keeps any literal "<TAB>" in its text.
            line = line.replace(_TL_TAB_PLACEHOLDER, "\t")
        fields = line.split("\t")
        sigil, rest = fields[0], fields[1:]
        heading = _HEADING_SIGIL_RE.match(sigil)
        if heading is not None:
            level = int(heading.group(1)) if heading.group(1) else 1
            blocks.append(
                {
                    "structure": "heading",
                    "level": level,
                    "lim": _tl_field(rest, 0),
                    "text": _tl_tail(rest, 1),
                }
            )
        elif sigil == "P":
            blocks.append({"structure": "p", "text": _tl_tail(rest, 0)})
        elif sigil == "I":
            blocks.append(
                {"structure": "item", "lim": _tl_field(rest, 0), "text": _tl_tail(rest, 1)}
            )
        elif sigil == "R":
            blocks.append({"structure": "row", "cells": [_tl_unescape(c) for c in rest]})
        elif sigil == "F":
            blocks.append(
                {
                    "structure": "field",
                    "lim": _tl_field(rest, 0),
                    "label": _tl_field(rest, 1),
                    "value": _tl_tail(rest, 2),
                }
            )
        elif sigil == "O":
            options: list[str] = []
            checked: list[str] = []
            for tok in rest[3:]:
                starred = tok.startswith("*")
                body = _tl_unescape(tok[1:] if starred else tok)
                options.append(body)
                if starred:
                    checked.append(body)
            blocks.append(
                {
                    "structure": "field",
                    "lim": _tl_field(rest, 0),
                    "label": _tl_field(rest, 1),
                    "value": _tl_field(rest, 2),
                    "options": options,
                    "checked": checked,
                }
            )
        elif sigil == "C":
            continues = _tl_tail(rest, 0)
        else:
            dropped += 1
    return {"continues": continues, "blocks": blocks}, dropped


def parse_window_any(raw: str, *, log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """Parse a window reply in EITHER wire format, by content sniffing.

    After fence-stripping, a reply starting with ``{`` takes today's JSON path
    (:func:`_parse_window_json`) unchanged — including raising
    ``json.JSONDecodeError`` on damaged JSON, so the caller's
    ``_salvage_window_json`` fallback still runs exactly as before. Anything
    else decodes as the compact tab-delimited format, which is per-line
    tolerant and never raises. The CONTENT decides, not how the model was
    prompted, so legacy JSON caches and today's compact replies parse through
    one entry point.
    """
    if strip_fences(raw).lstrip().startswith("{"):
        return _parse_window_json(raw)
    # Compact branch: unwrap a fence WITHOUT strip_fences' whole-text strip —
    # a trailing tab on the final line is an empty trailing cell/value and is
    # significant (stripping it would shift the last row's column count).
    match = _JSON_FENCE_RE.search(raw)
    body = match.group(1) if match else raw
    payload, dropped = _parse_window_compact(body)
    if dropped:
        log(f"compact reply: dropped {dropped} malformed line(s)")
    return payload


def _append_continuation(blocks: list[Block], continuation: str) -> None:
    """Splice mid-element continuation text onto the last text-bearing block.

    ``xml_safe`` here for the same reason as in :func:`parse_block`: this text
    comes straight off the model and is written into a block, so it is the one
    other path by which an unserializable character could reach the renderer.
    """
    text = xml_safe(continuation).strip()
    if not text:
        return
    for block in reversed(blocks):
        if block.structure in ("p", "item", "heading"):
            joiner = "" if (block.text and block.text[-1].isspace()) else " "
            block.text = f"{block.text}{joiner}{text}" if block.text else text
            return
        if block.structure == "field":
            joiner = "" if (block.value and block.value[-1].isspace()) else " "
            block.value = f"{block.value}{joiner}{text}" if block.value else text
            return
        if block.structure == "row" and block.cells:
            block.cells[-1] = f"{block.cells[-1]} {text}".strip()
            return


# ── window completeness gate ────────────────────────────────────────────────
# A window can come back as VALID JSON that simply stops partway through its
# pages: a clean end_turn — no length cut for call_continued to resume, no
# parse error to salvage — silently losing whole pages (and, because windows
# are fixed page ranges, the next window never re-covers the hole). When the
# caller passes the file's page_text/ directory (per-page words written by
# digital extraction/OCR before generation), every window's output is checked
# against the words its pages actually contain, and re-requested when recall
# falls below _GATE_RECALL. Healthy windows recall nearly all of their pages'
# words; windows that stopped early fall far short. Windows whose pages carry
# fewer than _GATE_MIN_TOKENS extractable words (image-only scans) are never
# gated.
_GATE_RECALL = 0.85
_GATE_MIN_TOKENS = 50
_GATE_RETRIES = 1
# Lines mode: a window whose best attempt still leaves this many lines
# unaccounted for gets one rescue call over just those lines.
_LINES_RESCUE_MIN = 6


def _tokens(text: str) -> list[str]:
    """The coverage tokenizer over NFKC-normalised text: PDF text layers carry
    ligatures ('proﬁt') where a model writes plain letters, and without the
    normalisation every such word reads as missing."""
    return coverage._tokenize(unicodedata.normalize("NFKC", text))


def _page_token_lists(page_text_dir: Path | str | None) -> list[list[str]]:
    """Per-page token lists from a workspace page_text/ dir ([] without one)."""
    if page_text_dir is None:
        return []
    pages = coverage.read_workspace_page_texts(Path(page_text_dir))
    return [_tokens(p) for p in pages]


def _payload_text(payload: dict[str, Any]) -> str:
    """Every character a parsed window payload contributes to the document."""
    parts = [str(payload.get("continues", "") or "")]
    for b in payload.get("blocks", []) or []:
        if not isinstance(b, dict):
            continue
        for key in ("text", "lim", "label", "value"):
            parts.append(str(b.get(key, "") or ""))
        parts.extend(str(c) for c in b.get("cells", []) or [])
    return " ".join(parts)


def _window_recall(payload: dict[str, Any], expected: list[str]) -> float:
    """Multiset recall of the window's page words within the window's output."""
    if not expected:
        return 1.0
    have = Counter(_tokens(_payload_text(payload)))
    matched = 0
    for tok in expected:
        if have[tok] > 0:
            have[tok] -= 1
            matched += 1
    return matched / len(expected)


def _payload_tail(payload: dict[str, Any]) -> str:
    """Last ~300 chars a payload contributes — the next slice's continuation tail."""
    for b in reversed(payload.get("blocks", []) or []):
        if not isinstance(b, dict):
            continue
        parts = [
            str(b.get("lim", "") or ""),
            str(b.get("text", "") or ""),
            str(b.get("label", "") or ""),
            str(b.get("value", "") or ""),
        ]
        parts += [str(c) for c in b.get("cells", []) or []]
        text = " ".join(p for p in parts if p).strip()
        if text:
            return text[-300:]
    return ""


def _merge_payloads(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two half-window payloads into one window-shaped payload.

    ``b``'s ``continues`` finishes ``a``'s final block (the same rule
    _append_continuation applies at the blocks level), so the merged dict
    parses identically to a single window that produced both halves.
    """
    blocks_a = [dict(x) for x in a.get("blocks", []) or [] if isinstance(x, dict)]
    blocks_b = [x for x in b.get("blocks", []) or [] if isinstance(x, dict)]
    cont_b = str(b.get("continues", "") or "").strip()
    if cont_b and blocks_a:
        last = blocks_a[-1]
        if last.get("cells"):
            cells = [str(c) for c in last["cells"]]
            cells[-1] = f"{cells[-1]} {cont_b}".strip()
            last["cells"] = cells
        elif last.get("value") or last.get("label"):
            last["value"] = f"{last.get('value', '') or ''!s} {cont_b}".strip()
        else:
            last["text"] = f"{last.get('text', '') or ''!s} {cont_b}".strip()
    return {"continues": str(a.get("continues", "") or ""), "blocks": blocks_a + blocks_b}


def transcribe_steps(
    pdf_bytes: bytes,
    *,
    doc_name: str,
    config: llm.LLMConfig,
    window_size: int = 10,
    cache_dir: Path | str | None = None,
    debug: bool = False,
    log: Callable[[str], None] = lambda _m: None,
    page_text_dir: Path | str | None = None,
    pdf_config: PdfConfig | None = None,
    window_overlap: int = 1,
    complete: bool = True,
    transcription: str = "text",
) -> llm.LLMSteps[list[Block]]:
    """Pass A for one document as a state machine: yields each window request in
    order, receives its response, and returns the flat block list.

    This is the single copy of the transcription logic. :func:`transcribe_document`
    drives it synchronously; a batch driver advances many documents' generators a
    wave at a time. Either way the requests, gate decisions, cache artifacts, and
    blocks are the same, because they are all decided here. The generator is pure
    in the :data:`dgml_core.llm.LLMSteps` sense: every model request is a yield of
    litellm kwargs (composed from :func:`dgml_core.llm.steps_continued`) and it
    does no usage accounting; the driver owns that.

    Windows are strictly sequential within a document: window N+1's instruction
    carries window N's structural tail (:func:`_window_context`), so the next
    window is never yielded before the previous one's response has been sent in.
    The gate retry, the stage-2 split, and a length continuation are further
    yields of the same generator.

    Cache behavior matches :func:`transcribe_document`: a cached
    ``<stem>_blocks.json`` returns immediately without yielding; raw window
    artifacts are written as each response lands (``debug``-gated); the blocks
    cache is written only once the whole document is assembled, so a generator
    closed mid-document leaves no partial functional cache behind.

    *transcription* picks the wire format: ``"text"`` has the model copy the
    text into typed block lines; ``"lines"`` shows the model the numbered
    lines of the pages' text layer and has it assign structure to line ids,
    the text being copied from the page (:mod:`dgml_core.generation.lines`).
    ``lines`` needs *page_text_dir*; a window with a page that has no usable
    text layer is asked in the text format instead. In lines mode windows do
    not overlap (line ids make the join exact) and a line the model left
    unaccounted for is filled in as a plain block before the window is kept.

    Windows overlap by *window_overlap* page(s): the page a window skips at
    its start is then also the previous window's last page, and the
    completeness pass drops whichever copy the alignment did not claim.

    With *complete* set (and *page_text_dir* available), the assembled blocks
    are aligned against the page words after the windows, and whatever the
    model left out is filled from the text layer at its reading-order position
    -- a whole page is first offered back to the model as a single-page
    rescue. See :mod:`dgml_core.generation.completeness`; the report is
    written as ``<stem>_completeness.json`` beside the blocks cache.
    """
    cached = _load_cached_blocks(cache_dir, doc_name)
    if cached is not None:
        log(f"{doc_name}: reusing cached transcription ({len(cached)} block(s))")
        return cached
    total = _count_pages(pdf_bytes)
    lines_mode = transcription == "lines" and page_text_dir is not None
    page_words: dict[int, Any] = {}
    if lines_mode:
        from dgml_core.generation import completeness

        assert page_text_dir is not None  # lines_mode implies it
        page_words = completeness.load_page_words_dir(page_text_dir)
        doc_vocab = lines_mod.document_vocabulary(page_words)
        if not page_words:
            log(f"{doc_name}: no text layer; transcribing in the text format")
            lines_mode = False
    overlap = 0 if lines_mode else max(0, min(window_overlap, window_size - 1))
    windows = document.iter_windows(total, window_size, overlap=overlap)
    log(
        f"{doc_name}: {total} pages → {len(windows)} window(s)"
        + (" [structure over text-layer lines]" if lines_mode else "")
    )
    page_tokens = _page_token_lists(page_text_dir)
    # Index in `blocks` at which each window's output begins; the completeness
    # pass uses it to remove the overlap page's second transcription.
    window_starts: list[int] = []
    # In lines mode, the 1-based pages that were nevertheless transcribed in
    # the text format (no text layer, or a text-format reply): the pass may
    # fill those, and only those.
    text_format_pages: set[int] = set()

    # One usage row per document, aggregating every window's call (gated on
    # ``record_usage`` via the config): the driver opens a ``record_usage_for`` scope on
    # this config and the row it writes carries this context. ``config`` is
    # fresh per document in the pipeline, so setting it here is thread-local.
    config.context = {"doc": doc_name}
    blocks: list[Block] = []
    counter = 0
    stem = Path(doc_name).stem

    def run_attempts(
        pages: list[int], context: str, wlog: str, wfile: str
    ) -> llm.LLMSteps[tuple[float, str, dict[str, Any]] | None]:
        """Gated attempt loop for one page range; best (recall, raw, payload)."""
        # `total` was counted once for this document above; passing it keeps
        # each window from re-walking the whole page tree.
        pdf_slice = document.slice_pdf(pdf_bytes, pages, config=pdf_config, total_pages=total)
        exp = [t for p in pages if p < len(page_tokens) for t in page_tokens[p]]
        # Lines mode for this window when a page has a text layer; a page
        # without one (a scan, a map) gets a text-format call of its own
        # after the lines reply, so one such page does not drag the whole
        # window into the text format.
        wlines: list[lines_mod.TextLine] = []
        textless: list[int] = []
        if lines_mode:
            with_text = [
                p
                for p in pages
                if len(page_words.get(p + 1, ([], None))[0]) >= lines_mod.MIN_PAGE_WORDS
            ]
            textless = [p for p in pages if p not in with_text]
            if with_text:
                wlines = lines_mod.window_lines(page_words, [p + 1 for p in with_text])
        if wlines:
            listing = lines_mod.render_listing(wlines)
            cache_write(cache_dir, f"{stem}_{wfile}_lines.txt", listing, debug=debug)
            instr = _lines_instruction(pages[0], pages[-1], total, context, listing)
            system_prompt = SYSTEM_PROMPT_LINES
        else:
            if lines_mode:
                log(f"{doc_name} {wlog}: no page has a text layer; asking in the text format")
            instr = _window_instruction(pages[0], pages[-1], total, context)
            system_prompt = SYSTEM_PROMPT
        n_attempts = 1 + (_GATE_RETRIES if len(exp) >= _GATE_MIN_TOKENS else 0)
        found: tuple[float, str, dict[str, Any]] | None = None
        found_lines: lines_mod.LinesResult | None = None
        for attempt in range(n_attempts):
            # Retry-nudge: at temperature 0 an identical retry tends to
            # reproduce the same early stop, so tell the model what its
            # previous attempt missed instead of re-rolling the same call.
            attempt_instr = instr
            if attempt > 0 and wlines and found_lines is not None:
                attempt_instr = (
                    instr
                    + "\n\n"
                    + prompt("transcribe_lines_retry").format(
                        missing=lines_mod.describe_unassigned(found_lines.unassigned(wlines)),
                        n_lines=len(wlines),
                    )
                )
            elif attempt > 0:
                attempt_instr = (
                    instr
                    + "\n\n"
                    + prompt("transcribe_window_retry").format(
                        pct=round(100 * (found[0] if found else 0.0)),
                        first=pages[0] + 1,
                        last=pages[-1] + 1,
                    )
                )
            raw = yield from llm.steps_continued(
                config,
                system_prompt=system_prompt,
                user_content=llm.build_user_content(
                    instruction_text=attempt_instr, pdf_bytes=pdf_slice
                ),
                cache=True,  # cache the static system prefix across windows (Anthropic)
            )
            suffix = "" if attempt == 0 else f"_retry{attempt}"
            cache_write(
                cache_dir,
                f"{stem}_{wfile}{suffix}_raw.json",
                strip_fences(raw),
                debug=debug,
            )
            if wlines and strip_fences(raw).lstrip().startswith("{"):
                # The model answered in the text format (copied text as
                # JSON) despite the lines prompt. Keep the window as a
                # text-format window: parse it as such and gate it on
                # word recall; no line accounting is possible.
                log(f"{doc_name} {wlog}: text-format reply to the lines prompt; kept as text")
                wlines_attempt_text = True
            else:
                wlines_attempt_text = False
            if wlines and not wlines_attempt_text:
                res = lines_mod.build_payload(strip_fences(raw), wlines, doc_vocab)
                recall = res.recall(wlines)
                if res.malformed or res.unknown_refs or res.literal_fields:
                    log(
                        f"{doc_name} {wlog}: lines reply: {res.malformed} malformed line(s), "
                        f"{res.unknown_refs} unknown/duplicate ref(s), "
                        f"{res.literal_fields} literal field(s)"
                    )
                if found is None or recall > found[0]:
                    found = (recall, raw, res.payload)
                    found_lines = res
                if recall >= _GATE_RECALL:
                    break
                log(
                    f"{doc_name} {wlog}: {len(res.unassigned(wlines))} of {len(wlines)} "
                    f"lines unaccounted for ({recall:.0%} of the words)"
                    + ("; retrying window" if attempt + 1 < n_attempts else "")
                )
                continue
            try:
                payload = parse_window_any(raw, log=lambda m: log(f"{doc_name} {wlog}: {m}"))
            except json.JSONDecodeError as exc:
                # Continuation should normally close the JSON; if a window
                # still arrives truncated (e.g. a stream cut the provider
                # didn't flag as length), salvage the complete blocks
                # instead of dropping it all.
                salvaged = _salvage_window_json(raw)
                if salvaged is None:
                    log(f"{doc_name} {wlog}: unparseable JSON ({exc})")
                    continue
                payload = salvaged
                log(
                    f"{doc_name} {wlog}: truncated JSON; "
                    f"salvaged {len(payload.get('blocks', []))} block(s)"
                )
            recall = _window_recall(payload, exp)
            if found is None or recall > found[0]:
                found = (recall, raw, payload)
            if recall >= _GATE_RECALL:
                break
            log(
                f"{doc_name} {wlog}: transcription covers only "
                f"{recall:.0%} of the pages' words"
                + ("; retrying window" if attempt + 1 < n_attempts else "")
            )
        lines_won = (
            found is not None and found_lines is not None and found[2] is found_lines.payload
        )
        if wlines and found is not None and not lines_won:
            # The kept reply was in the text format (the model ignored the
            # lines prompt): these pages were not accounted for line by
            # line, so the completeness pass must be allowed to fill them.
            text_format_pages.update(p + 1 for p in pages)
        if wlines and lines_won:
            assert found is not None and found_lines is not None  # lines_won implies both
            # A span the model skipped is asked for on its own: the
            # sub-listing of just those lines, same pages attached. A
            # tiny reply, and precise where a page split is not.
            missing = found_lines.unassigned(wlines)
            if len(missing) >= _LINES_RESCUE_MIN:
                sub = lines_mod.render_listing(missing)
                rescue_instr = (
                    prompt("transcribe_lines_remaining").format(
                        first=pages[0] + 1,
                        last=pages[-1] + 1,
                        total=total,
                        n_lines=len(missing),
                        missing=lines_mod.describe_unassigned(missing),
                    )
                    + "\n\nNumbered lines:\n"
                    + sub
                )
                raw = yield from llm.steps_continued(
                    config,
                    system_prompt=SYSTEM_PROMPT_LINES,
                    user_content=llm.build_user_content(
                        instruction_text=rescue_instr, pdf_bytes=pdf_slice
                    ),
                    cache=True,
                )
                cache_write(
                    cache_dir, f"{stem}_{wfile}_rescue_raw.json", strip_fences(raw), debug=debug
                )
                extra = lines_mod.build_payload(strip_fences(raw), missing, doc_vocab)
                got = len(extra.assigned | extra.decorations)
                log(
                    f"{doc_name} {wlog}: rescue over {len(missing)} skipped line(s) "
                    f"accounted for {got}"
                )
                lines_mod.merge_results(found_lines, extra)
            # Whatever is still unaccounted for is filled from the text
            # layer here, so the window is complete by construction.
            filled = lines_mod.complete_payload(found_lines, wlines, doc_vocab)
            moved = lines_mod.order_by_lines(found_lines)
            for p in textless:
                # The page without a text layer, transcribed in the text
                # format on its own and spliced in at its place.
                p_instr = _window_instruction(p, p, total, _window_context(blocks))
                p_slice = document.slice_pdf(pdf_bytes, [p], config=pdf_config, total_pages=total)
                p_raw = yield from llm.steps_continued(
                    config,
                    system_prompt=SYSTEM_PROMPT,
                    user_content=llm.build_user_content(
                        instruction_text=p_instr, pdf_bytes=p_slice
                    ),
                    cache=True,
                )
                cache_write(
                    cache_dir,
                    f"{stem}_{wfile}_p{p + 1:03d}_raw.json",
                    strip_fences(p_raw),
                    debug=debug,
                )
                p_tag = f"{doc_name} {wlog} p{p + 1}"

                def _page_log(m: str, _t: str = p_tag) -> None:
                    log(f"{_t}: {m}")

                try:
                    p_payload = parse_window_any(p_raw, log=_page_log)
                except json.JSONDecodeError:
                    p_payload = _salvage_window_json(p_raw) or {"blocks": []}
                p_blocks = [b for b in p_payload.get("blocks", []) or [] if isinstance(b, dict)]
                lines_mod.insert_page_blocks(found_lines, wlines, p + 1, p_blocks)
                log(
                    f"{doc_name} {wlog}: page {p + 1} has no text layer; "
                    f"{len(p_blocks)} block(s) from a text-format call"
                )
                text_format_pages.add(p + 1)
            if moved:
                log(f"{doc_name} {wlog}: {moved} block(s) put back in page order")
            n_dec = len(found_lines.decorations)
            log(
                f"{doc_name} {wlog}: {len(wlines)} lines → "
                f"{len(found_lines.payload['blocks'])} block(s), {n_dec} decoration(s)"
                + (f", {filled} line(s) filled from the text layer" if filled else "")
            )
            found = (1.0 if filled else found[0], found[1], found_lines.payload)
        return found

    for w_idx, page_indices in enumerate(windows):
        context = _window_context(blocks)
        wlog, wfile = f"w{w_idx + 1}", f"w{w_idx + 1:02d}"
        expected = [t for p in page_indices if p < len(page_tokens) for t in page_tokens[p]]
        gate_on = len(expected) >= _GATE_MIN_TOKENS
        best = yield from run_attempts(page_indices, context, wlog, wfile)
        if best is None:
            log(f"{doc_name} {wlog}: window skipped")
            continue
        # Stage-2 fallback: a retry that reproduces the same early stop is
        # anchored in the window's CONTENT, so change the INPUT — split
        # the page range and transcribe the halves.
        if gate_on and best[0] < _GATE_RECALL and len(page_indices) >= 2:
            log(f"{doc_name} {wlog}: still short after retry; splitting the window")
            mid = (len(page_indices) + 1) // 2
            half_a = yield from run_attempts(page_indices[:mid], context, f"{wlog}a", f"{wfile}a")
            context_b = _payload_tail(half_a[2]) if half_a else context
            half_b = yield from run_attempts(page_indices[mid:], context_b, f"{wlog}b", f"{wfile}b")
            if half_a and half_b:
                merged = _merge_payloads(half_a[2], half_b[2])
                merged_recall = _window_recall(merged, expected)
                if merged_recall > best[0]:
                    best = (merged_recall, json.dumps(merged, ensure_ascii=False), merged)
                    log(
                        f"{doc_name} {wlog}: split halves cover "
                        f"{merged_recall:.0%} — keeping the split"
                    )
        recall, raw, payload = best
        if gate_on and recall < _GATE_RECALL:
            log(f"{doc_name} {wlog}: keeping best attempt at {recall:.0%} page-word coverage")
        # The kept content always lives at the unsuffixed name the caches
        # and debug tooling expect (a no-retry run writes it exactly once;
        # a kept split writes the merged payload).
        cache_write(
            cache_dir,
            f"{stem}_{wfile}_raw.json",
            strip_fences(raw),
            debug=debug,
        )
        # With overlap, a window's first page is also the previous window's
        # last, so a "continues" tail would complete text that is already
        # there in full.
        if not (overlap and w_idx):
            _append_continuation(blocks, str(payload.get("continues", "") or ""))
        kept = 0
        window_starts.append(len(blocks))
        for raw_block in payload.get("blocks", []) or []:
            if not isinstance(raw_block, dict):
                continue
            counter += 1
            block = parse_block(raw_block, block_id=f"b{counter:04d}")
            if block is not None:
                blocks.append(block)
                kept += 1
        log(f"{doc_name} {wlog}: {kept} block(s)")

    if complete and page_text_dir is not None and blocks:
        from dgml_core.generation import completeness

        # The block list as the windows produced it, before the pass, so a
        # completeness report can be replayed offline (debug only).
        cache_write(
            cache_dir,
            f"{stem}_blocks_prepass.json",
            json.dumps(
                {
                    "window_starts": window_starts,
                    "blocks": [dataclasses.asdict(b) for b in blocks],
                },
                ensure_ascii=False,
            ),
            debug=debug,
        )
        # The pass is split around its model calls. The plan (align, drop
        # duplicates, find the regions) is pure; a page the model left out
        # entirely is then offered back to it as a single-page rescue — a
        # further yield of this generator, which a callback could not make;
        # the fill (a rescue that claims its page's words, else the text
        # layer) is pure again. Structure over lines is complete by
        # construction; the pass then only checks, reports and removes
        # duplicates.
        plan = completeness.plan_completion(
            blocks,
            page_text_dir,
            window_starts=window_starts,
            fill=not lines_mode,
            fill_pages=text_format_pages if lines_mode else None,
        )
        rescued: dict[int, list[Block] | None] = {}
        for page_idx in plan.rescue_pages:
            tag = f"p{page_idx + 1:03d}"
            got = yield from run_attempts(
                [page_idx], _window_context(blocks), f"p{page_idx + 1}", tag
            )
            page_blocks: list[Block] | None = None
            if got is not None:
                page_blocks = []
                for k, raw_block in enumerate(got[2].get("blocks", []) or []):
                    if isinstance(raw_block, dict):
                        b = parse_block(raw_block, block_id=f"r{page_idx + 1:03d}_{k + 1:03d}")
                        if b is not None:
                            page_blocks.append(b)
            rescued[page_idx] = page_blocks
        blocks, report = completeness.finish_completion(
            plan, rescued, log=lambda m: log(f"{doc_name}: {m}")
        )
        cache_write(
            cache_dir,
            f"{stem}_completeness.json",
            json.dumps(report, ensure_ascii=False, indent=1),
            debug=True,
        )
    # Deterministic normalization: printed enumerators already encode the
    # answer, so remove the model's per-run degrees of freedom (p-vs-item on
    # sequential "(a)…" runs; heading depth of dotted numbering) before the
    # blocks become the document of record.
    normalize_enumerated_paragraphs(blocks)
    anchor_heading_levels(blocks)
    # Functional file the next run reloads — written regardless of --debug.
    cache_write(cache_dir, f"{Path(doc_name).stem}_blocks.json", blocks_to_json(blocks), debug=True)
    return blocks


def transcribe_document(
    pdf_bytes: bytes,
    *,
    doc_name: str,
    config: llm.LLMConfig,
    window_size: int = 10,
    cache_dir: Path | str | None = None,
    debug: bool = False,
    log: Callable[[str], None] = lambda _m: None,
    page_text_dir: Path | str | None = None,
    pdf_config: PdfConfig | None = None,
    window_overlap: int = 1,
    complete: bool = True,
    transcription: str = "text",
) -> list[Block]:
    """Transcribe one document into a flat block list (Pass A), synchronously.

    The sync driver over :func:`transcribe_steps`: each window request the
    generator yields is executed immediately, in order. *window_overlap*,
    *complete* and *transcription* are passed through; see there.

    With *cache_dir* set, the assembled flat blocks are written as
    ``<stem>_blocks.json`` (a functional file the next run reloads). With
    *debug* additionally set, each window's RAW model return is written as
    ``<stem>_wNN_raw.json`` (pre-parse, so truncation/JSON damage is
    inspectable).

    With *page_text_dir* set (the file's per-page word JSONs, written by
    digital extraction/OCR before generation), each window's output is
    completeness-checked against the words its pages contain and retried up
    to ``_GATE_RETRIES`` times when recall falls below ``_GATE_RECALL`` — the
    guard against silent window early-stops. Without it behavior is unchanged.

    A cached ``<stem>_blocks.json`` short-circuits the whole pass: the blocks
    are reloaded verbatim and no LLM call is made — so a re-run only pays for
    labeling and rendering. Delete the cache file to force re-transcription.
    """
    # Short-circuit before opening the usage scope, exactly as before: a cached
    # document makes no call and leaves no usage row. transcribe_steps repeats
    # the (cheap) check and returns without yielding, and the batch driver
    # (dgml_core.batch.run_stage) opens a unit's usage scope only once its
    # generator has yielded a request — so a cached document leaves no row
    # there either. (llm.drive would write an empty row for such a generator,
    # which is why this sync path checks the cache before calling it.)
    cached = _load_cached_blocks(cache_dir, doc_name)
    if cached is not None:
        log(f"{doc_name}: reusing cached transcription ({len(cached)} block(s))")
        return cached
    steps = transcribe_steps(
        pdf_bytes,
        doc_name=doc_name,
        config=config,
        window_size=window_size,
        cache_dir=cache_dir,
        debug=debug,
        log=log,
        page_text_dir=page_text_dir,
        pdf_config=pdf_config,
        window_overlap=window_overlap,
        complete=complete,
        transcription=transcription,
    )
    # One usage row per document: drive() records every request into this
    # scope, which writes the single aggregated row on exit.
    with llm.record_usage_for(config):
        return llm.drive(steps, config)
