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

"""Pass A — verbatim transcription into flat typed blocks (JSON).

Per document: disjoint page windows, one LLM call each, JSON out. Merging is
list concatenation; a window that starts mid-sentence returns the remainder
in `continues`, which is appended to the previous window's last text block.
No overlap pages, no dedup, no XML repair — the contract makes those
concepts inapplicable.
"""

from __future__ import annotations

import dataclasses
import json
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dgml_core import llm
from dgml_core.generation import document
from dgml_core.generation.blocks import Block, parse_block
from dgml_core.generation.prompts import get as prompt
from dgml_core.pages import pdf_page_count

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


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def strip_fences(text: str) -> str:
    """Unwrap a ```json … ``` code fence so cached raw JSON is valid JSON.

    Returns the fenced body when a complete fence is present; otherwise returns
    the text unchanged (stripped). Truncated/damaged output — which has no
    closing fence — is left as-is so failures stay inspectable in the cache.
    """
    match = _JSON_FENCE_RE.search(text)
    return (match.group(1) if match else text).strip()


SYSTEM_PROMPT = prompt("transcribe_system")


def _window_instruction(first_page: int, last_page: int, total: int, tail: str) -> str:
    parts = [
        prompt("transcribe_window_header").format(
            first=first_page + 1, last=last_page + 1, total=total
        )
    ]
    if tail:
        parts.append(prompt("transcribe_window_tail").format(tail=tail))
    parts.append(prompt("transcribe_window_json"))
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                text = text[start : end + 1]
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


def _append_continuation(blocks: list[Block], continuation: str) -> None:
    """Splice mid-element continuation text onto the last text-bearing block."""
    text = continuation.strip()
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


def transcribe_document(
    pdf_bytes: bytes,
    *,
    doc_name: str,
    config: llm.LLMConfig,
    window_size: int = 10,
    cache_dir: Path | str | None = None,
    debug: bool = False,
    log: Callable[[str], None] = lambda _m: None,
) -> list[Block]:
    """Transcribe one document into a flat block list (Pass A).

    With *cache_dir* set, the assembled flat blocks are written as
    ``<stem>_blocks.json`` (a functional file the next run reloads). With
    *debug* additionally set, each window's RAW model return is written as
    ``<stem>_wNN_raw.json`` (pre-parse, so truncation/JSON damage is
    inspectable).
    """
    total = _count_pages(pdf_bytes)
    windows = document.iter_windows(total, window_size, overlap=0)
    log(f"{doc_name}: {total} pages → {len(windows)} window(s)")

    # One usage row per document, aggregating every window's call (gated on
    # --debug via the config). ``config`` is fresh per document in the pipeline,
    # so setting the context here is safe and thread-local.
    config.context = {"doc": doc_name}
    blocks: list[Block] = []
    counter = 0
    with llm.record_usage_for(config):
        for w_idx, page_indices in enumerate(windows):
            tail = blocks[-1].flat_text()[-300:] if blocks else ""
            instruction = _window_instruction(page_indices[0], page_indices[-1], total, tail)
            window_pdf = document.slice_pdf(pdf_bytes, page_indices)
            raw = llm.call_continued(
                config,
                system_prompt=SYSTEM_PROMPT,
                user_content=llm.build_user_content(
                    instruction_text=instruction, pdf_bytes=window_pdf
                ),
                cache=True,  # cache the static system prefix across windows (Anthropic)
            )
            stem = Path(doc_name).stem
            cache_write(
                cache_dir, f"{stem}_w{w_idx + 1:02d}_raw.json", strip_fences(raw), debug=debug
            )
            try:
                payload = _parse_window_json(raw)
            except json.JSONDecodeError as exc:
                # Continuation should normally close the JSON; if a window still
                # arrives truncated (e.g. a stream cut the provider didn't flag as
                # length), salvage the complete blocks instead of dropping it all.
                salvaged = _salvage_window_json(raw)
                if salvaged is None:
                    log(f"{doc_name} w{w_idx + 1}: unparseable JSON ({exc}); window skipped")
                    continue
                payload = salvaged
                log(
                    f"{doc_name} w{w_idx + 1}: truncated JSON; "
                    f"salvaged {len(payload.get('blocks', []))} block(s)"
                )
            _append_continuation(blocks, str(payload.get("continues", "") or ""))
            kept = 0
            for raw_block in payload.get("blocks", []) or []:
                if not isinstance(raw_block, dict):
                    continue
                counter += 1
                block = parse_block(raw_block, block_id=f"b{counter:04d}")
                if block is not None:
                    blocks.append(block)
                    kept += 1
            log(f"{doc_name} w{w_idx + 1}: {kept} block(s)")
    # Functional file the next run reloads — written regardless of --debug.
    cache_write(cache_dir, f"{Path(doc_name).stem}_blocks.json", blocks_to_json(blocks), debug=True)
    return blocks
