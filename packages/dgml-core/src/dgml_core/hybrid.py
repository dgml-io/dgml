"""Hybrid text extraction: run digital + OCR and merge by bounding-box overlap.

The output JSON shape matches :func:`dgml.text_extraction.extract_text_digital`
and :func:`dgml.ocr.extract_text_ocr` so downstream code (``dgml check``,
consumers) doesn't care which mode produced the text.

Merge rules (per page, applied to word bounding boxes):

- **CID guard**: if digital text has more than
  :data:`MAX_CID_WORDS_PER_PAGE` words containing ``"(cid:"`` — i.e.
  pdfminer couldn't resolve glyphs to Unicode — we treat the whole page's
  digital output as garbage, log a "unicode error", and use OCR for the
  entire page.
- Otherwise, group digital + OCR words that cover the same region into
  **regions** (connected components over a "boxes overlap" graph; see
  :func:`_boxes_overlap`, which matches on IoU *or* containment so a
  merged box and the split boxes it contains land in one region). Each
  region is resolved as a unit, so split / merge tokenization — one side
  splitting a span the other keeps whole — is one decision instead of
  per-word matching:

    * OCR-only region (no digital) → keep OCR; log it.
    * Digital-only region (no OCR) → assumed invisible to the human eye
      (white-on-white, off-page, hidden form layer, …) and **dropped**
      with a warning.
    * Mixed region → concatenate each side's text in reading order and
      compute the (dash-normalized) Levenshtein distance:

        - within :data:`LEVENSHTEIN_THRESHOLD` → the two agree on content,
          so keep **digital** — its character codes come straight from the
          PDF font, which is more reliable than OCR even when OCR's
          tokenization is finer.
        - beyond the threshold → the texts disagree, so take **OCR** (the
          authority on what's actually visible) and warn.

Text comparison folds dash-family code points (PDFs love U+2212 / en-dash
where OCR returns ASCII hyphen) via :func:`_normalize_for_compare`, but the
stored word keeps its original characters.

**LLM-driven merge (optional).** When the workspace declares a
``text_extraction`` section (see :mod:`dgml.text_extraction_config`), the
per-region decision is delegated to that LLM instead of the Levenshtein
rule above. The decision tree changes to:

- digital-only region → dropped (no LLM call, as before);
- OCR-only region → keep OCR (no LLM call);
- mixed region whose digital and OCR token sequences are *identical*
  (dash-normalized) → keep digital (no LLM call);
- every other (differing) mixed region → sent to the LLM, which returns the
  final token list (take digital / take OCR / a combination).

A page's to-decide regions go out in batches of :data:`MERGE_BATCH_SIZE`
regions per call (one page-sized call can overrun the model's output-token
budget and come back truncated). A batch whose call fails (model unreachable,
timeout, unparseable / structurally-invalid response) falls back to the
heuristic for *that batch's* regions only; other batches keep their LLM
result, so a flaky local model degrades gracefully. See :func:`_llm_emit_plan`
for the request/response contract.

Warnings are emitted to stderr — stdout is reserved for the CLI's JSON
payload contract — and are gated behind a ``verbose`` flag. By default
hybrid mode is silent; pass ``verbose=True`` (set by ``dgml --verbose``
on the CLI) to see the per-page warnings and summary.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .errors import TextExtractionFailed
from .llm import LLMConfig, build_user_content, call
from .ocr import OcrConfig, _write_page_json, extract_text_ocr
from .prompts import get as prompt
from .storage import Workspace
from .text_extraction import (
    PAGE_TEXT_GLOB,
    ExtractDigitalResult,
    extract_text_digital,
)
from .text_extraction_config import TextExtractionConfig, resolve_api_key
from .usage import OPERATION_HYBRID_MERGE

OVERLAP_THRESHOLD = 0.5
# Two boxes also count as overlapping when one is at least this fraction
# contained in the other (intersection / smaller-box area), even if IoU is
# low. Catches split/merge tokenization where a merged box dwarfs each split
# box, so their IoU never reaches OVERLAP_THRESHOLD.
COVERAGE_THRESHOLD = 0.8
# Words within this many edits are treated as "the same word" (digital wins).
# 2 covers common OCR mistakes (l vs 1, O vs 0, ASCII hyphen vs the
# Unicode minus or en-dash variants PDFs often use) without flattening
# genuinely different short tokens.
LEVENSHTEIN_THRESHOLD = 2
# If digital text on a page has more than this many words containing the
# pdfminer "(cid:N)" sentinel — meaning the PDF's font CMap didn't resolve
# glyph IDs to Unicode — we treat the page's digital output as unusable
# and fall back to OCR for that page.
MAX_CID_WORDS_PER_PAGE = 10
# Regions needing an LLM decision go out in batches of this many per request.
# One call for the whole page can overrun the model's output-token limit (or a
# local model's num_ctx) on dense pages — the reply is truncated mid-JSON and
# the page falls back to the heuristic. Batching keeps each reply small enough
# to complete, and isolates a failure to its own batch (only that batch's
# regions fall back, not the page). Tune down for small-context local models,
# up to cut per-call overhead.
MERGE_BATCH_SIZE = 40

# One region queued for the LLM: (sort_key, region_id, digital_idxs,
# ocr_idxs, payload_entry). Sorted by sort_key into page reading order.
_RegionToSend = tuple[tuple[int, int], str, list[int], list[int], dict[str, Any]]


class _LLMMergeError(ValueError):
    """LLM merge failure that carries the raw model output for logging.

    Raised once the model has replied but its reply is unparseable or
    structurally invalid. ``raw_output`` is the text (or a JSON dump of the
    parsed-but-invalid object) so the fall-back warning can show what the
    model actually returned. Transport failures (no reply) raise the
    underlying exception instead, with no ``raw_output``.
    """

    def __init__(self, message: str, *, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


def extract_text_hybrid(
    pdf_path: Path,
    output_dir: Path,
    *,
    file_id: str,
    page_images_dir: Path,
    config: OcrConfig,
    text_extraction_config: TextExtractionConfig | None = None,
    workspace: Workspace | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> ExtractDigitalResult:
    """Run digital extraction then OCR and merge the per-page results.

    If digital extraction itself fails (pdfminer can't parse the PDF), log
    a stderr warning (when ``verbose``) and continue with OCR-only output
    rather than aborting — OCR is the authoritative source in hybrid mode.
    An OCR failure is propagated to the caller (same as ``--text-mode ocr``).

    ``text_extraction_config`` (when not ``None``) switches the per-region
    merge from the Levenshtein heuristic to the configured LLM;
    ``workspace`` is used only to record LLM usage telemetry. Both are
    optional so non-workspace callers keep the heuristic behaviour.
    """
    with tempfile.TemporaryDirectory(prefix="dgml-hybrid-") as tmp_str:
        tmp = Path(tmp_str)
        digital_dir = tmp / "digital"
        ocr_dir = tmp / "ocr"

        digital_failed = False
        try:
            extract_text_digital(pdf_path, digital_dir, file_id=file_id)
        except TextExtractionFailed as exc:
            digital_failed = True
            if verbose:
                print(
                    f"warning: file_id={file_id}: digital extraction failed; "
                    f"falling back to OCR-only output: {exc}",
                    file=sys.stderr,
                )

        extract_text_ocr(
            pdf_path,
            ocr_dir,
            file_id=file_id,
            page_images_dir=page_images_dir,
            config=config,
        )

        return _merge_into(
            None if digital_failed else digital_dir,
            ocr_dir,
            output_dir,
            file_id=file_id,
            text_extraction_config=text_extraction_config,
            workspace=workspace,
            verbose=verbose,
            debug=debug,
        )


def _merge_into(
    digital_dir: Path | None,
    ocr_dir: Path,
    output_dir: Path,
    *,
    file_id: str,
    text_extraction_config: TextExtractionConfig | None = None,
    workspace: Workspace | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> ExtractDigitalResult:
    """Merge per-page JSONs from ``digital_dir`` and ``ocr_dir`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob(PAGE_TEXT_GLOB):
        existing.unlink()

    digital_pages = _index_page_files(digital_dir) if digital_dir is not None else {}
    ocr_pages = _index_page_files(ocr_dir)

    pages_written = 0
    pages_with_words = 0
    total_words = 0

    for page_num in sorted(set(digital_pages) | set(ocr_pages)):
        d_payload = _read_page(digital_pages.get(page_num))
        o_payload = _read_page(ocr_pages.get(page_num))

        # Prefer OCR's reported dimensions — the PNG IHDR is the source of
        # truth they were measured against. Digital fills in only when OCR
        # didn't process this page.
        if o_payload is not None:
            width = int(o_payload["width"])
            height = int(o_payload["height"])
        else:
            assert d_payload is not None  # at least one side processed this page
            width = int(d_payload["width"])
            height = int(d_payload["height"])

        digital_words = list(d_payload["words"]) if d_payload is not None else []
        ocr_words = list(o_payload["words"]) if o_payload is not None else []
        merged = _merge_words(
            digital_words,
            ocr_words,
            file_id=file_id,
            page_num=page_num,
            text_extraction_config=text_extraction_config,
            workspace=workspace,
            verbose=verbose,
            debug=debug,
        )

        _write_page_json(output_dir, page_num, file_id, width, height, merged)
        pages_written += 1
        if merged:
            pages_with_words += 1
            total_words += len(merged)

    return ExtractDigitalResult(
        pages_written=pages_written,
        pages_with_words=pages_with_words,
        total_words=total_words,
    )


def _merge_words(
    digital_words: list[dict[str, Any]],
    ocr_words: list[dict[str, Any]],
    *,
    file_id: str,
    page_num: int,
    text_extraction_config: TextExtractionConfig | None = None,
    workspace: Workspace | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """Apply the merge rules described in the module docstring.

    When ``text_extraction_config`` is ``None`` the deterministic
    Levenshtein heuristic resolves each region (:func:`_heuristic_emit_plan`).
    Otherwise the configured LLM does (:func:`_llm_emit_plan`), with a
    fall-back to the heuristic for the page on any failure. Per-page warnings
    and summary go to stderr only when ``verbose`` is set.
    """
    cid_count = _count_cid_words(digital_words)
    if cid_count > MAX_CID_WORDS_PER_PAGE:
        if verbose:
            print(
                f"unicode error: file_id={file_id} page={page_num}: digital text "
                f"has {cid_count} words containing '(cid:' (pdfminer could not "
                f"resolve glyphs); using OCR for the entire page",
                file=sys.stderr,
            )
            print(
                f"hybrid: file_id={file_id} page={page_num}: "
                f"digital_words={len(digital_words)} ocr_words={len(ocr_words)} "
                f"merged={len(ocr_words)} cid_guard=true",
                file=sys.stderr,
            )
        return list(ocr_words)

    regions = _region_overlaps(digital_words, ocr_words)

    emit_plan: dict[int, list[dict[str, Any]]] | None = None
    if text_extraction_config is not None:
        try:
            emit_plan = _llm_emit_plan(
                regions,
                digital_words,
                ocr_words,
                file_id=file_id,
                page_num=page_num,
                config=text_extraction_config,
                workspace=workspace,
                verbose=verbose,
                debug=debug,
            )
        except Exception as exc:
            # Safety net: per-batch failures are handled (and logged) inside
            # _llm_emit_plan, so reaching here means an unexpected error
            # escaped it — leave emit_plan unset and fall back to the heuristic
            # for the whole page below.
            if verbose:
                print(
                    f"warning: file_id={file_id} page={page_num}: LLM merge failed "
                    f"({type(exc).__name__}: {exc}); falling back to heuristic",
                    file=sys.stderr,
                )

    # Heuristic merge when no LLM is configured, or the LLM merge failed above.
    if emit_plan is None:
        emit_plan = _heuristic_emit_plan(
            regions,
            digital_words,
            ocr_words,
            file_id=file_id,
            page_num=page_num,
            verbose=verbose,
        )

    merged: list[dict[str, Any]] = []
    for j, o_word in enumerate(ocr_words):
        merged.extend(emit_plan[j] if j in emit_plan else [o_word])

    # OCR words carry no style; transplant the observed style facts ("s") from
    # the spatially-coincident digital word so digital/hybrid feed the same
    # deterministic dg:style path. Digital words that survived the merge keep
    # their own "s" untouched.
    merged = _apply_digital_style(merged, digital_words)

    if verbose:
        print(
            f"hybrid: file_id={file_id} page={page_num}: "
            f"digital_words={len(digital_words)} ocr_words={len(ocr_words)} "
            f"merged={len(merged)}",
            file=sys.stderr,
        )
    return merged


def _apply_digital_style(
    merged: list[dict[str, Any]], digital_words: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Copy each digital word's observed style (``"s"``) onto the merged word
    that occupies the same region, when the merged word has no style of its own.

    Hybrid keeps OCR geometry/text but OCR carries no font facts; the digital
    word at the same place does. A merged word is matched to the styled digital
    word with the greatest overlap of the merged box, accepted at ≥ 50%."""
    styled = [w for w in digital_words if w.get("s") and w.get("l")]
    if not styled:
        return merged
    out: list[dict[str, Any]] = []
    for w in merged:
        box = w.get("l")
        if w.get("s") or not box:
            out.append(w)
            continue
        best_s: dict[str, Any] | None = None
        best_frac = 0.5
        for d in styled:
            frac = _box_overlap_fraction(box, d["l"])
            if frac > best_frac:
                best_frac = frac
                best_s = d["s"]
        if best_s is not None:
            w = {**w, "s": best_s}
        out.append(w)
    return out


def _box_overlap_fraction(a: list[int], b: list[int]) -> float:
    """Intersection area of boxes ``a`` and ``b`` as a fraction of ``a``'s area.
    Boxes are ``[left, top, right, bottom]``."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    return inter / area_a


# `emit_plan[o_idx]` overrides what an OCR word emits: a list of words to
# splice in at that position, or `[]` to drop it. OCR words absent from the
# plan (and those with malformed boxes, which never region) emit themselves —
# so OCR reading order is preserved. Both resolvers below return one.
def _heuristic_emit_plan(
    regions: list[tuple[list[int], list[int]]],
    digital_words: list[dict[str, Any]],
    ocr_words: list[dict[str, Any]],
    *,
    file_id: str,
    page_num: int,
    verbose: bool,
) -> dict[int, list[dict[str, Any]]]:
    """Resolve regions with the Levenshtein heuristic (the default merge)."""
    emit_plan: dict[int, list[dict[str, Any]]] = {}

    for d_idxs, o_idxs in regions:
        if not o_idxs:
            # Digital-only region: no visual counterpart, assume invisible.
            if verbose:
                for di in d_idxs:
                    dw = digital_words[di]
                    print(
                        f"warning: file_id={file_id} page={page_num}: digital text "
                        f"{dw.get('t', '')!r} at {dw.get('l')} was not detected by "
                        f"OCR; assumed invisible to human eye, dropping",
                        file=sys.stderr,
                    )
            continue

        if not d_idxs:
            # OCR-only region: keep, log which tokens OCR contributed.
            if verbose:
                for oi in o_idxs:
                    ow = ocr_words[oi]
                    print(
                        f"info: file_id={file_id} page={page_num}: OCR text "
                        f"{ow.get('t', '')!r} at {ow.get('l')} has no matching "
                        f"digital text; keeping OCR",
                        file=sys.stderr,
                    )
            continue

        # Mixed region. Compare concatenated text (dash-normalized) to decide
        # whether the two sides agree, then keep the finer tokenization.
        d_concat = _concat_reading_order([digital_words[i] for i in d_idxs])
        o_concat = _concat_reading_order([ocr_words[j] for j in o_idxs])
        dist = _levenshtein_distance(
            _normalize_for_compare(d_concat), _normalize_for_compare(o_concat)
        )
        agree = dist <= LEVENSHTEIN_THRESHOLD
        # Agree → use digital (character codes come straight from the PDF font,
        # so they're more reliable than OCR even when OCR's tokenization is
        # finer). Disagree → OCR is the authority on what's actually visible.
        take_digital = agree

        if take_digital:
            anchor = min(o_idxs)
            emit_plan[anchor] = _reading_order([digital_words[i] for i in d_idxs])
            for oi in o_idxs:
                if oi != anchor:
                    emit_plan[oi] = []

        if verbose:
            _log_region_decision(
                file_id=file_id,
                page_num=page_num,
                digital_words=digital_words,
                ocr_words=ocr_words,
                d_idxs=d_idxs,
                o_idxs=o_idxs,
                d_text=d_concat,
                o_text=o_concat,
                dist=dist,
                agree=agree,
                take_digital=take_digital,
            )

    return emit_plan


def _merge_system_prompt() -> str:
    """Return the hybrid-merge system prompt, loaded from ``resources/prompts.yaml``.

    Fetched via :func:`dgml_core.prompts.get` (the shared loader used by every
    core LLM feature), so wording can be tuned without touching this module.
    The loader caches after the first read.
    """
    return prompt("merge_system_prompt").rstrip()


def _llm_emit_plan(
    regions: list[tuple[list[int], list[int]]],
    digital_words: list[dict[str, Any]],
    ocr_words: list[dict[str, Any]],
    *,
    file_id: str,
    page_num: int,
    config: TextExtractionConfig,
    workspace: Workspace | None,
    verbose: bool,
    debug: bool = False,
) -> dict[int, list[dict[str, Any]]]:
    """Resolve regions with the configured LLM, batched across requests.

    Decision tree (see module docstring): digital-only regions are dropped,
    OCR-only regions keep OCR, and token-identical mixed regions keep digital
    — all without any LLM call. Only differing mixed regions are sent to the
    model, in batches of :data:`MERGE_BATCH_SIZE`. A batch whose request
    fails (transport / parse / structural-validation) falls back to the
    heuristic for *that batch's* regions only — disjoint from the other
    batches, so no region is resolved twice. Successful batches keep their
    LLM result; unexpected errors propagate so :func:`_merge_words` can fall
    back to the heuristic for the whole page.
    """
    emit_plan: dict[int, list[dict[str, Any]]] = {}
    to_send: list[_RegionToSend] = []

    for n, (d_idxs, o_idxs) in enumerate(regions):
        if not o_idxs:
            # Digital-only region: dropped, same as the heuristic. No LLM call.
            if verbose:
                for di in d_idxs:
                    dw = digital_words[di]
                    print(
                        f"warning: file_id={file_id} page={page_num}: digital text "
                        f"{dw.get('t', '')!r} at {dw.get('l')} was not detected by "
                        f"OCR; assumed invisible to human eye, dropping",
                        file=sys.stderr,
                    )
            continue

        if not d_idxs:
            # OCR-only region: keep, same as the heuristic. No LLM call — the
            # OCR words emit themselves (absent from emit_plan).
            if verbose:
                for oi in o_idxs:
                    ow = ocr_words[oi]
                    print(
                        f"info: file_id={file_id} page={page_num}: OCR text "
                        f"{ow.get('t', '')!r} at {ow.get('l')} has no matching "
                        f"digital text; keeping OCR",
                        file=sys.stderr,
                    )
            continue

        if _tokens_identical(d_idxs, o_idxs, digital_words, ocr_words):
            # Mixed region whose tokenization + text already agree → digital.
            anchor = min(o_idxs)
            emit_plan[anchor] = _reading_order([digital_words[i] for i in d_idxs])
            for oi in o_idxs:
                if oi != anchor:
                    emit_plan[oi] = []
            continue

        # Only differing mixed regions reach the LLM.
        rid = f"r{n}"
        entry: dict[str, Any] = {"id": rid, "kind": "mixed"}
        entry["digital"] = [
            {"id": f"d{i}", "t": str(digital_words[i].get("t", ""))}
            for i in _reading_order_indices(d_idxs, digital_words)
        ]
        entry["ocr"] = [
            {"id": f"o{j}", "t": str(ocr_words[j].get("t", ""))}
            for j in _reading_order_indices(o_idxs, ocr_words)
        ]
        sort_key = _region_sort_key(d_idxs, o_idxs, digital_words, ocr_words)
        to_send.append((sort_key, rid, d_idxs, o_idxs, entry))

    if not to_send:
        return emit_plan

    to_send.sort(key=lambda e: e[0])  # regions in page reading order
    for start in range(0, len(to_send), MERGE_BATCH_SIZE):
        batch = to_send[start : start + MERGE_BATCH_SIZE]
        try:
            response = _call_merge_llm(
                {"regions": [e[4] for e in batch]},
                config=config,
                workspace=workspace,
                file_id=file_id,
                page_num=page_num,
                debug=debug,
            )
            # Resolve into a fragment first: a malformed entry raises before
            # any of the batch's decisions touch emit_plan, so the heuristic
            # fall-back below re-resolves the whole batch without colliding
            # with regions that happened to parse.
            batch_plan = _resolve_batch_decisions(
                batch,
                response,
                digital_words=digital_words,
                ocr_words=ocr_words,
                file_id=file_id,
                page_num=page_num,
                verbose=verbose,
            )
        except Exception as exc:
            if verbose:
                batch_num = start // MERGE_BATCH_SIZE + 1
                print(
                    f"warning: file_id={file_id} page={page_num}: LLM merge failed "
                    f"for batch {batch_num} ({len(batch)} regions) "
                    f"({type(exc).__name__}: {exc}); falling back to heuristic "
                    f"for this batch",
                    file=sys.stderr,
                )
                raw_output = getattr(exc, "raw_output", None)
                if raw_output is not None:
                    print(
                        f"warning: file_id={file_id} page={page_num}: LLM merge raw "
                        f"output was: {raw_output!r}",
                        file=sys.stderr,
                    )
            batch_plan = _heuristic_emit_plan(
                [(d_idxs, o_idxs) for _key, _rid, d_idxs, o_idxs, _entry in batch],
                digital_words,
                ocr_words,
                file_id=file_id,
                page_num=page_num,
                verbose=verbose,
            )
        emit_plan.update(batch_plan)

    return emit_plan


def _resolve_batch_decisions(
    batch: list[_RegionToSend],
    response: dict[str, Any],
    *,
    digital_words: list[dict[str, Any]],
    ocr_words: list[dict[str, Any]],
    file_id: str,
    page_num: int,
    verbose: bool,
) -> dict[int, list[dict[str, Any]]]:
    """Turn one batch's LLM response into an emit-plan fragment.

    Validates every region in the batch, raising :class:`_LLMMergeError` on
    the first malformed entry *before* returning, so the caller can fall back
    to the heuristic for the whole batch without double-resolving the regions
    that did parse.
    """
    batch_plan: dict[int, list[dict[str, Any]]] = {}
    for _key, rid, d_idxs, o_idxs, _entry in batch:
        decision = response.get(rid)
        if not isinstance(decision, list):
            raise _LLMMergeError(
                f"response missing or malformed entry for region {rid!r}",
                raw_output=json.dumps(response, ensure_ascii=False),
            )
        resolved = _resolve_llm_decision(decision, d_idxs, o_idxs, digital_words, ocr_words)
        anchor = min(o_idxs)
        batch_plan[anchor] = resolved
        for oi in o_idxs:
            if oi != anchor:
                batch_plan[oi] = []
        if verbose:
            d_texts = [
                str(digital_words[i].get("t", ""))
                for i in _reading_order_indices(d_idxs, digital_words)
            ]
            o_texts = [
                str(ocr_words[j].get("t", "")) for j in _reading_order_indices(o_idxs, ocr_words)
            ]
            accepted = [str(w.get("t", "")) for w in resolved]
            print(
                f"info: file_id={file_id} page={page_num}: LLM resolved mixed "
                f"region {rid} ({len(d_idxs)} digital, {len(o_idxs)} OCR); "
                f"digital={d_texts!r} ocr={o_texts!r} "
                f"-> {len(resolved)} token(s): {accepted!r}",
                file=sys.stderr,
            )

    return batch_plan


def _call_merge_llm(
    payload: dict[str, Any],
    *,
    config: TextExtractionConfig,
    workspace: Workspace | None,
    file_id: str,
    page_num: int,
    debug: bool = False,
) -> dict[str, Any]:
    """Send one page's regions to the LLM and return the parsed JSON object.

    The call records its own usage row (gated on ``--debug``) from the context
    carried on the config.
    """
    llm_config = LLMConfig(
        model=config.model,
        api_key=resolve_api_key(config),
        api_base=config.api_base,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        workspace=workspace,
        debug=debug,
        operation=OPERATION_HYBRID_MERGE,
        context={"file_id": file_id, "page": page_num},
    )
    user_content = build_user_content(instruction_text=json.dumps(payload, ensure_ascii=False))
    text = call(
        llm_config,
        system_prompt=_merge_system_prompt(),
        user_content=user_content,
    )
    return _parse_merge_response(text)


def _parse_merge_response(text: str) -> dict[str, Any]:
    """Parse the model's reply into a ``{region_id: [tokens]}`` dict.

    Tolerates a ```` ```json ```` / ```` ``` ```` code fence around the JSON.
    Raises ``ValueError`` on anything that isn't a JSON object.
    """
    try:
        data = json.loads(_strip_code_fences(text))
    except json.JSONDecodeError as exc:
        raise _LLMMergeError(f"LLM response was not valid JSON: {exc}", raw_output=text) from exc
    if not isinstance(data, dict):
        raise _LLMMergeError("LLM response was not a JSON object", raw_output=text)
    return data


def _strip_code_fences(text: str) -> str:
    """Remove a single surrounding Markdown code fence if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Drop the opening fence line (``` or ```json) and a trailing fence line.
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _resolve_llm_decision(
    decision: list[Any],
    d_idxs: list[int],
    o_idxs: list[int],
    digital_words: list[dict[str, Any]],
    ocr_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn one region's ``[{ref, t?}]`` decision into output word dicts.

    Each output word inherits the box of the input token named by ``ref``;
    ``t`` overrides the text. An unknown ``ref`` falls back to the nearest
    box (the region's first reading-order token) rather than emitting a
    box-less word — the model can lose box *precision* but never produce an
    invalid word dict. Tokens with no resolvable text are skipped. Raises
    ``ValueError`` on a structurally invalid item (caller → page fallback).
    """
    id_to_word: dict[str, dict[str, Any]] = {}
    for i in d_idxs:
        id_to_word[f"d{i}"] = digital_words[i]
    for j in o_idxs:
        id_to_word[f"o{j}"] = ocr_words[j]

    region_words = [digital_words[i] for i in d_idxs] + [ocr_words[j] for j in o_idxs]
    ordered = _reading_order(region_words)
    fallback_box = ordered[0]["l"] if ordered else None

    out: list[dict[str, Any]] = []
    for item in decision:
        if not isinstance(item, dict) or "ref" not in item:
            raise ValueError(f"malformed token in decision: {item!r}")
        override = item.get("t")
        if override is not None and not isinstance(override, str):
            raise ValueError(f"non-string 't' in decision token: {item!r}")

        ref = item["ref"]
        src = id_to_word.get(ref) if isinstance(ref, str) else None
        if src is not None:
            box = src["l"]
            base_text = str(src.get("t", ""))
        else:
            box = fallback_box
            base_text = ""

        text = override if override is not None else base_text
        if not text or box is None:
            # Unknown ref with no usable text, or no box to borrow → drop it.
            continue
        out.append({"t": text, "l": box})
    return out


def _tokens_identical(
    d_idxs: list[int],
    o_idxs: list[int],
    digital_words: list[dict[str, Any]],
    ocr_words: list[dict[str, Any]],
) -> bool:
    """True if the two sides have identical token sequences (dash-normalized).

    Reading-order token texts compared one-for-one. Such regions skip the
    LLM and keep digital (its character codes come straight from the font).
    """
    if len(d_idxs) != len(o_idxs):
        return False
    d_texts = [
        _normalize_for_compare(str(digital_words[i].get("t", "")))
        for i in _reading_order_indices(d_idxs, digital_words)
    ]
    o_texts = [
        _normalize_for_compare(str(ocr_words[j].get("t", "")))
        for j in _reading_order_indices(o_idxs, ocr_words)
    ]
    return d_texts == o_texts


def _reading_order_indices(idxs: list[int], words: list[dict[str, Any]]) -> list[int]:
    """Sort word indices top-to-bottom then left-to-right (see :func:`_reading_order`)."""
    if not idxs:
        return []
    band = max(1, min(max(1, words[i]["l"][3] - words[i]["l"][1]) for i in idxs) // 2)
    return sorted(idxs, key=lambda i: (words[i]["l"][1] // band, words[i]["l"][0]))


def _region_sort_key(
    d_idxs: list[int],
    o_idxs: list[int],
    digital_words: list[dict[str, Any]],
    ocr_words: list[dict[str, Any]],
) -> tuple[int, int]:
    """Top-left ``(top, left)`` of a region, for ordering regions in the payload."""
    boxes = [digital_words[i]["l"] for i in d_idxs] + [ocr_words[j]["l"] for j in o_idxs]
    top = min(b[1] for b in boxes)
    left = min(b[0] for b in boxes)
    return (top, left)


def _log_region_decision(
    *,
    file_id: str,
    page_num: int,
    digital_words: list[dict[str, Any]],
    ocr_words: list[dict[str, Any]],
    d_idxs: list[int],
    o_idxs: list[int],
    d_text: str,
    o_text: str,
    dist: int,
    agree: bool,
    take_digital: bool,
) -> None:
    """Emit the stderr line(s) describing how one mixed region was resolved."""
    if len(d_idxs) == 1 and len(o_idxs) == 1:
        # Clean 1:1 — digital silently wins when the texts agree; only the
        # disagreement (OCR wins) is worth a warning.
        if not agree:
            dw, ow = digital_words[d_idxs[0]], ocr_words[o_idxs[0]]
            print(
                f"warning: file_id={file_id} page={page_num}: digital text "
                f"{dw.get('t', '')!r} and OCR text {ow.get('t', '')!r} overlap at "
                f"digital={dw.get('l')} ocr={ow.get('l')} but differ "
                f"(levenshtein={dist} > {LEVENSHTEIN_THRESHOLD}); using OCR",
                file=sys.stderr,
            )
        return

    # Split / merge region.
    kept = f"digital's {len(d_idxs)}" if take_digital else f"OCR's {len(o_idxs)}"
    print(
        f"info: file_id={file_id} page={page_num}: tokenization mismatch "
        f"({len(d_idxs)} digital vs {len(o_idxs)} OCR words; "
        f"text {'agrees' if agree else 'differs'}, levenshtein={dist}); "
        f"digital={d_text!r} ocr={o_text!r}; "
        f"keeping {kept} tokens",
        file=sys.stderr,
    )


def _count_cid_words(words: list[dict[str, Any]]) -> int:
    """Count words whose text contains the pdfminer ``(cid:N)`` sentinel."""
    count = 0
    for w in words:
        text = w.get("t", "")
        if isinstance(text, str) and "(cid:" in text:
            count += 1
    return count


# Dash-family code points that PDFs and OCR engines disagree about. Folding
# them to ASCII hyphen-minus before measuring Levenshtein distance keeps
# (e.g.) a digital MINUS-SIGN-separated ``out-of-pocket`` from looking
# different from OCR's ASCII-hyphen ``out-of-pocket``.
#
# Code points referenced explicitly so this file stays free of ambiguous
# Unicode characters (ruff RUF001).
_DASH_CODEPOINTS: tuple[int, ...] = (
    0x002D,  # HYPHEN-MINUS (the target — included so the lookup is idempotent)
    0x00AD,  # SOFT HYPHEN
    0x2010,  # HYPHEN
    0x2011,  # NON-BREAKING HYPHEN
    0x2012,  # FIGURE DASH
    0x2013,  # EN DASH
    0x2014,  # EM DASH
    0x2015,  # HORIZONTAL BAR
    0x2043,  # HYPHEN BULLET
    0x2212,  # MINUS SIGN
    0xFE58,  # SMALL EM DASH
    0xFE63,  # SMALL HYPHEN-MINUS
    0xFF0D,  # FULLWIDTH HYPHEN-MINUS
)
_DASH_TABLE = {cp: "-" for cp in _DASH_CODEPOINTS}


def _normalize_for_compare(text: str) -> str:
    """Fold dash-family characters to ASCII ``-`` for comparison purposes only.

    Used solely on the Levenshtein input — the stored word text keeps the
    original code points so downstream consumers see whatever the source
    actually contained.
    """
    return text.translate(_DASH_TABLE)


def _region_overlaps(
    digital_words: list[dict[str, Any]],
    ocr_words: list[dict[str, Any]],
) -> list[tuple[list[int], list[int]]]:
    """Group digital & OCR word indices into connected overlap regions.

    Union-Find over a bipartite graph whose edges are "boxes overlap" (see
    :func:`_boxes_overlap`). Returns a list of ``(digital_indices,
    ocr_indices)``: only-OCR means an OCR-only region, only-digital a
    digital-only region, both a shared region (possibly with differing
    tokenization). Words with malformed boxes are excluded; the caller keeps
    such OCR words as-is.
    """
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x: tuple[str, int]) -> tuple[str, int]:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: tuple[str, int], b: tuple[str, int]) -> None:
        parent[find(a)] = find(b)

    d_valid = [(i, w["l"]) for i, w in enumerate(digital_words) if _is_box(w.get("l"))]
    o_valid = [(j, w["l"]) for j, w in enumerate(ocr_words) if _is_box(w.get("l"))]

    for i, _ in d_valid:
        find(("d", i))
    for j, _ in o_valid:
        find(("o", j))

    for i, d_box in d_valid:
        for j, o_box in o_valid:
            if _boxes_overlap(d_box, o_box):
                union(("d", i), ("o", j))

    groups: dict[tuple[str, int], tuple[list[int], list[int]]] = {}
    for i, _ in d_valid:
        groups.setdefault(find(("d", i)), ([], []))[0].append(i)
    for j, _ in o_valid:
        groups.setdefault(find(("o", j)), ([], []))[1].append(j)
    return list(groups.values())


def _boxes_overlap(a: list[int], b: list[int]) -> bool:
    """True if two boxes cover the same text region.

    Matches when IoU exceeds :data:`OVERLAP_THRESHOLD` (clean 1:1) **or** one
    box is mostly contained in the other (coverage = intersection / smaller
    area >= :data:`COVERAGE_THRESHOLD`). Coverage is what catches split/merge:
    a merged box dwarfs each split box, so IoU stays low but coverage is high.
    """
    inter = _intersection_area(a, b)
    if inter == 0:
        return False
    area_a, area_b = _area(a), _area(b)
    union = area_a + area_b - inter
    if union > 0 and inter / union > OVERLAP_THRESHOLD:
        return True
    smaller = min(area_a, area_b)
    return smaller > 0 and inter / smaller >= COVERAGE_THRESHOLD


def _reading_order(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort words top-to-bottom then left-to-right.

    Top is bucketed by half the smallest box height so same-line jitter (a
    1-2px difference in box tops) doesn't reorder words on one line.
    """
    if not words:
        return []
    band = max(1, min(max(1, w["l"][3] - w["l"][1]) for w in words) // 2)
    return sorted(words, key=lambda w: (w["l"][1] // band, w["l"][0]))


def _concat_reading_order(words: list[dict[str, Any]]) -> str:
    """Concatenate word text in reading order (no separators)."""
    return "".join(str(w.get("t", "")) for w in _reading_order(words))


def _is_box(box: Any) -> bool:
    return isinstance(box, list) and len(box) == 4


def _area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection_area(a: list[int], b: list[int]) -> int:
    il, it = max(a[0], b[0]), max(a[1], b[1])
    ir, ib = min(a[2], b[2]), min(a[3], b[3])
    if ir <= il or ib <= it:
        return 0
    return (ir - il) * (ib - it)


def _levenshtein_distance(a: str, b: str) -> int:
    """Classic Levenshtein edit distance (single-row DP, O(len(a) * len(b)))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insertions = curr_row[j - 1] + 1
            deletions = prev_row[j] + 1
            substitutions = prev_row[j - 1] + (0 if ca == cb else 1)
            curr_row[j] = min(insertions, deletions, substitutions)
        prev_row = curr_row
    return prev_row[-1]


def _iou(a: list[int], b: list[int]) -> float:
    """IoU of two ``[left, top, right, bottom]`` pixel boxes."""
    inter = _intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _index_page_files(dir_: Path) -> dict[int, Path]:
    """Map page number → ``page_<N>.json`` path under ``dir_``."""
    if not dir_.exists():
        return {}
    out: dict[int, Path] = {}
    for path in dir_.glob(PAGE_TEXT_GLOB):
        num = _page_num_from_text_name(path.name)
        if num is not None:
            out[num] = path
    return out


def _page_num_from_text_name(name: str) -> int | None:
    """Parse ``page_<N>.json`` → ``N``."""
    prefix = "page_"
    suffix = ".json"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    try:
        return int(name[len(prefix) : -len(suffix)])
    except ValueError:
        return None


def _read_page(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


__all__ = [
    "OVERLAP_THRESHOLD",
    "extract_text_hybrid",
]
