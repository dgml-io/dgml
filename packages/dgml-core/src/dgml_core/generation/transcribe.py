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
"""

from __future__ import annotations

import dataclasses
import json
import re
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dgml_core import llm
from dgml_core.generation import coverage, document
from dgml_core.generation.blocks import (
    Block,
    Span,
    anchor_heading_levels,
    normalize_enumerated_paragraphs,
    parse_block,
)
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
        return [
            Block(**{**b, "entities": [Span(**sp) for sp in b.get("entities", [])]})
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


# ── window completeness gate ────────────────────────────────────────────────
# A window can come back as VALID JSON that simply stops partway through its
# pages: a clean end_turn — no length cut for call_continued to resume, no
# parse error to salvage — silently losing whole pages (and, because windows
# are fixed page ranges, the next window never re-covers the hole). When the
# caller passes the file's page_text/ directory (per-page words written by
# digital extraction/OCR before generation), every window's output is checked
# against the words its pages actually contain, and re-requested when recall
# falls below _GATE_RECALL. Healthy windows measure >=0.96 on real corpora;
# observed failures measure 0.13-0.66. Windows whose pages carry fewer than
# _GATE_MIN_TOKENS extractable words (image-only scans) are never gated.
_GATE_RECALL = 0.85
_GATE_MIN_TOKENS = 50
_GATE_RETRIES = 1


def _page_token_lists(page_text_dir: Path | str | None) -> list[list[str]]:
    """Per-page token lists from a workspace page_text/ dir ([] without one)."""
    if page_text_dir is None:
        return []
    pages = coverage.read_workspace_page_texts(Path(page_text_dir))
    return [coverage._tokenize(p) for p in pages]


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
    have = Counter(coverage._tokenize(_payload_text(payload)))
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
) -> list[Block]:
    """Transcribe one document into a flat block list (Pass A).

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
    cached = _load_cached_blocks(cache_dir, doc_name)
    if cached is not None:
        log(f"{doc_name}: reusing cached transcription ({len(cached)} block(s))")
        return cached
    total = _count_pages(pdf_bytes)
    windows = document.iter_windows(total, window_size, overlap=0)
    log(f"{doc_name}: {total} pages → {len(windows)} window(s)")
    page_tokens = _page_token_lists(page_text_dir)

    # One usage row per document, aggregating every window's call (gated on
    # --debug via the config). ``config`` is fresh per document in the pipeline,
    # so setting the context here is safe and thread-local.
    config.context = {"doc": doc_name}
    blocks: list[Block] = []
    counter = 0
    stem = Path(doc_name).stem
    with llm.record_usage_for(config):

        def run_attempts(
            pages: list[int], page_tail: str, wlog: str, wfile: str
        ) -> tuple[float, str, dict[str, Any]] | None:
            """Gated attempt loop for one page range; best (recall, raw, payload)."""
            pdf_slice = document.slice_pdf(pdf_bytes, pages)
            instr = _window_instruction(pages[0], pages[-1], total, page_tail)
            exp = [t for p in pages if p < len(page_tokens) for t in page_tokens[p]]
            n_attempts = 1 + (_GATE_RETRIES if len(exp) >= _GATE_MIN_TOKENS else 0)
            found: tuple[float, str, dict[str, Any]] | None = None
            for attempt in range(n_attempts):
                # Retry-nudge: at temperature 0 an identical retry tends to
                # reproduce the same early stop, so tell the model what its
                # previous attempt missed instead of re-rolling the same call.
                attempt_instr = instr
                if attempt > 0:
                    attempt_instr = (
                        instr
                        + "\n\n"
                        + prompt("transcribe_window_retry").format(
                            pct=round(100 * (found[0] if found else 0.0)),
                            first=pages[0] + 1,
                            last=pages[-1] + 1,
                        )
                    )
                raw = llm.call_continued(
                    config,
                    system_prompt=SYSTEM_PROMPT,
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
                try:
                    payload = _parse_window_json(raw)
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
            return found

        for w_idx, page_indices in enumerate(windows):
            tail = blocks[-1].flat_text()[-300:] if blocks else ""
            wlog, wfile = f"w{w_idx + 1}", f"w{w_idx + 1:02d}"
            expected = [t for p in page_indices if p < len(page_tokens) for t in page_tokens[p]]
            gate_on = len(expected) >= _GATE_MIN_TOKENS
            best = run_attempts(page_indices, tail, wlog, wfile)
            if best is None:
                log(f"{doc_name} {wlog}: window skipped")
                continue
            # Stage-2 fallback: a retry that reproduces the same early stop is
            # anchored in the window's CONTENT, so change the INPUT — split
            # the page range and transcribe the halves.
            if gate_on and best[0] < _GATE_RECALL and len(page_indices) >= 2:
                log(f"{doc_name} {wlog}: still short after retry; splitting the window")
                mid = (len(page_indices) + 1) // 2
                half_a = run_attempts(page_indices[:mid], tail, f"{wlog}a", f"{wfile}a")
                tail_b = _payload_tail(half_a[2]) if half_a else tail
                half_b = run_attempts(page_indices[mid:], tail_b, f"{wlog}b", f"{wfile}b")
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
            log(f"{doc_name} {wlog}: {kept} block(s)")
    # Deterministic normalization: printed enumerators already encode the
    # answer, so remove the model's per-run degrees of freedom (p-vs-item on
    # sequential "(a)…" runs; heading depth of dotted numbering) before the
    # blocks become the document of record.
    normalize_enumerated_paragraphs(blocks)
    anchor_heading_levels(blocks)
    # Functional file the next run reloads — written regardless of --debug.
    cache_write(cache_dir, f"{Path(doc_name).stem}_blocks.json", blocks_to_json(blocks), debug=True)
    return blocks
