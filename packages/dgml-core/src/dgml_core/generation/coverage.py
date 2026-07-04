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

"""Word-coverage metrics between a source PDF and its generated DGML XML.

Three metrics, all computed on normalised tokens:

  unique_lexicon  — set-based recall  (each word counted once)
  rouge1          — frequency-aware unigram recall  (ROUGE-1 recall)
  rouge2          — bigram recall                   (ROUGE-2 recall)

All three have per-page variants, measuring whether each PDF page's content
was captured anywhere in the final XML.

PDF text is extracted from the workspace page_text JSON files in the supplied
page_text_dir. These files contain word-level OCR output that handles
non-standard fonts correctly (unlike naive text extraction, which can produce
garbled tokens for ASCII-shifted or symbol-encoded text).

If the PDF is image-based (fewer than 5 real words/page on average),
`digital_pdf` is False and no metrics are computed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TOP_MISSING_DOC = 50
_TOP_MISSING_PAGE = 15


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip currency symbols, de-comma numbers, strip punctuation.

    Hyphens are treated as word separators (not kept), so a hyphenated form
    and its spaced equivalent tokenize the same way: ``All-Inclusive`` and
    ``All - Inclusive`` both become ``all inclusive``. This avoids spurious
    "missing word" reports when the model writes ``All-Inclusive`` but the PDF
    renders it as ``All - Inclusive`` (or vice versa). Apostrophes are kept so
    contractions/possessives stay intact.
    """
    text = text.lower()
    text = re.sub(r"[$€£¥₹]", "", text)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)  # 1,200 → 1200
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    # Drop tokens with no alphanumeric content (e.g. signature underscore lines,
    # dot leaders).  `_` is a word char to Python regex so those tokens survive
    # _normalize without this guard.
    return [w for w in _normalize(text).split() if len(w) > 1 and any(c.isalnum() for c in w)]


def _ngrams(words: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))


# ---------------------------------------------------------------------------
# XML text extraction
# ---------------------------------------------------------------------------

_BARE_AMP_RE = re.compile(r"&(?!(?:#\d+|#x[\da-fA-F]+|[A-Za-z]\w*);)")
# (cid:N) appears when a character cannot be decoded from a CID font.
# Strip the whole token (including bare cid:N without parens) before tokenising
# so these encoding artifacts never appear in coverage metrics.
_CID_TOKEN_RE = re.compile(r"\(?cid:\d+\)?", re.IGNORECASE)


def _xml_words(xml_text: str) -> list[str]:
    try:
        from lxml import etree  # type: ignore[import-untyped]

        cleaned = _BARE_AMP_RE.sub("&amp;", xml_text)
        # Try strict parser first; recover parser can silently drop content.
        try:
            root = etree.fromstring(cleaned.encode())
        except etree.XMLSyntaxError:
            recover_parser = etree.XMLParser(recover=True, encoding="utf-8")
            root = etree.fromstring(cleaned.encode(), parser=recover_parser)
        if root is None or not any(True for _ in root.iter()):
            raise ValueError("empty tree after recovery")
        text = " ".join(root.itertext())
    except Exception:
        text = re.sub(r"<[^>]+>", " ", xml_text)
    return _tokenize(text)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _rouge_n_pct(ref: Counter[tuple[str, ...]], hyp: Counter[tuple[str, ...]]) -> float:
    total = sum(ref.values())
    if total == 0:
        return 0.0
    matched = sum(min(ref[ng], hyp[ng]) for ng in ref)
    return round(matched / total * 100, 1)


def _unique_pct(ref_words: list[str], hyp_vocab: set[str]) -> tuple[float, list[str]]:
    ref_vocab = set(ref_words)
    missing = sorted(ref_vocab - hyp_vocab)
    pct = round(len(ref_vocab & hyp_vocab) / len(ref_vocab) * 100, 1) if ref_vocab else 0.0
    return pct, missing


# ---------------------------------------------------------------------------
# Workspace page text reader
# ---------------------------------------------------------------------------


def read_workspace_page_texts(page_text_dir: Path) -> list[str]:
    """Read per-page text from workspace JSON files in *page_text_dir*.

    Files are named ``page_1.json``, ``page_2.json``, … Each file contains a
    JSON object with a ``words`` array of ``{"t": "<token>", "l": [...]}``
    entries in reading order.  Words are joined with spaces to produce a plain
    text string per page.

    Pages are returned in order; missing files for intermediate pages are
    returned as empty strings so that page numbering stays aligned.
    """
    if not page_text_dir.is_dir():
        return []
    page_files = sorted(
        page_text_dir.glob("page_*.json"),
        key=lambda p: int(re.search(r"page_(\d+)", p.stem).group(1)),  # type: ignore[union-attr]
    )
    if not page_files:
        return []
    max_page = int(re.search(r"page_(\d+)", page_files[-1].stem).group(1))  # type: ignore[union-attr]
    page_texts: list[str] = [""] * max_page
    for pf in page_files:
        idx = int(re.search(r"page_(\d+)", pf.stem).group(1)) - 1  # type: ignore[union-attr]
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            words = [w["t"] for w in data.get("words", []) if isinstance(w.get("t"), str)]
            page_texts[idx] = " ".join(words)
        except Exception:
            pass
    return page_texts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_coverage(
    xml_text: str,
    source_name: str,
    *,
    page_text_dir: Path,
) -> dict[str, Any]:
    """Return a coverage dict for one document.

    Page texts are read from the workspace page_text JSON files in
    *page_text_dir*. This produces accurate metrics for PDFs with
    non-standard fonts.

    If the PDF is image-based (fewer than 5 real words/page on average),
    returns ``{"source": ..., "digital_pdf": False}``.
    """
    page_texts = [_CID_TOKEN_RE.sub(" ", t) for t in read_workspace_page_texts(page_text_dir)]

    # Guard: detect image-only or undecodeable PDFs. We check avg words (not
    # chars) because undecodeable pages produce many short CID tokens that
    # would pass a char test.
    all_words = [_tokenize(t) for t in page_texts]
    avg_words = sum(len(w) for w in all_words) / max(len(all_words), 1)
    if avg_words < 5:  # fewer than 5 real words/page on average → skip
        return {"source": source_name, "digital_pdf": False}
    # Replace page_texts with the already-tokenised version to avoid re-tokenising.
    # We also blank out pages whose word list is suspiciously small so they
    # don't drag down the per-page stats as noise.
    page_texts = [" ".join(w) if len(w) >= 3 else "" for w in all_words]

    # --- XML side (computed once) ---
    xml_words = _xml_words(xml_text)
    xml_vocab = set(xml_words)
    xml_uni = _ngrams(xml_words, 1)
    xml_bi = _ngrams(xml_words, 2)

    # --- Full-document PDF side ---
    pdf_words = _tokenize(" ".join(page_texts))
    pdf_counter = Counter(pdf_words)
    pdf_uni = _ngrams(pdf_words, 1)
    pdf_bi = _ngrams(pdf_words, 2)

    unique_pct, missing_words = _unique_pct(pdf_words, xml_vocab)
    top_missing = [
        {"word": w, "pdf_count": pdf_counter[w]}
        for w in sorted(missing_words, key=lambda w: -pdf_counter[w])[:_TOP_MISSING_DOC]
    ]

    # --- Per-page ---
    per_page: list[dict[str, Any]] = []
    for i, page_text in enumerate(page_texts):
        pw = _tokenize(page_text)
        if not pw:
            per_page.append({"page": i + 1, "skipped": True})
            continue
        p_uni = _ngrams(pw, 1)
        p_bi = _ngrams(pw, 2)
        upct, p_missing = _unique_pct(pw, xml_vocab)
        p_counter = Counter(pw)
        per_page.append(
            {
                "page": i + 1,
                "pdf_words": len(pw),
                "unique_lexicon_pct": upct,
                "rouge1_pct": _rouge_n_pct(p_uni, xml_uni),
                "rouge2_pct": _rouge_n_pct(p_bi, xml_bi),
                "top_missing": [
                    w for w in sorted(p_missing, key=lambda w: -p_counter[w])[:_TOP_MISSING_PAGE]
                ],
            }
        )

    return {
        "source": source_name,
        "digital_pdf": True,
        "pdf_total_words": len(pdf_words),
        "pdf_unique_words": len(set(pdf_words)),
        "xml_total_words": len(xml_words),
        "xml_unique_words": len(xml_vocab),
        "unique_lexicon_pct": unique_pct,
        "rouge1_pct": _rouge_n_pct(pdf_uni, xml_uni),
        "rouge2_pct": _rouge_n_pct(pdf_bi, xml_bi),
        "top_missing_words": top_missing,
        "per_page": per_page,
    }


def merge_coverage_documents(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge coverage entries by ``source``; *new* replaces same-source *existing*."""
    by_source = {d["source"]: d for d in existing if "source" in d}
    for r in new:
        by_source[r["source"]] = r
    return list(by_source.values())


def save_coverage_report(results: list[dict[str, Any]], path: Path) -> None:
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "documents": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def coverage_summary_line(result: dict[str, Any]) -> str:
    """One-line summary for the progress log."""
    if not result.get("digital_pdf"):
        return f"{result['source']}: image-based PDF — coverage skipped"
    return (
        f"{result['source']}: "
        f"unique {result['unique_lexicon_pct']}% | "
        f"ROUGE-1 {result['rouge1_pct']}% | "
        f"ROUGE-2 {result['rouge2_pct']}%"
    )
