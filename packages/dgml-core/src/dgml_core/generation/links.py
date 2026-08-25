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

"""Post-generation semantic links.

A semantic link records a relationship the XML tree's nesting does not capture:
subject ``xml:id`` → ``dg:itemprop`` (predicate) → ``dg:href`` (``#id`` of the
object, or a space-separated list of ``#id``s when a value derives from several).
This pass covers three families:

- **references / relationships** — one element points to another it refers to,
  amends, incorporates, is a signatory of, describes, etc. (often non-local).
- **relative dates** — a date defined by another date/event ("each anniversary
  of the Commencement Date", "effective on signature"); offset in ``dg:value``.
- **derived values** — a value that means nothing on its own: a lesser/greater-of
  formula (multiple objects), a CPI-escalated rent, a value stated by reference.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lxml import etree  # type: ignore[import-untyped]

from dgml_core import llm
from dgml_core.errors import LinkPlanFailed
from dgml_core.generation.prompts import get as prompt
from dgml_core.generation.transcribe import loads_tolerant, strip_fences

_DG = "http://dgml.io/ns/dg#"
_XML = "http://www.w3.org/XML/1998/namespace"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"


def _salvage_items(raw: str, key: str) -> list[Any]:
    """The complete entries of the ``key`` array in a reply that will not decode.

    One truncated or malformed entry makes the whole reply unparseable, which
    would discard every good entry alongside the broken one. Decode whole
    objects out of the array until the first that will not decode, keep those,
    and drop the tail. Mirrors :func:`transcribe._salvage_window_json`.
    """
    text = strip_fences(raw)
    start = text.find(f'"{key}"')
    start = text.find("[", start) if start != -1 else -1
    if start == -1:
        return []
    decoder = json.JSONDecoder()
    items: list[Any] = []
    pos, end = start + 1, len(text)
    while pos < end:
        while pos < end and text[pos] in " \n\r\t,":
            pos += 1
        if pos >= end or text[pos] == "]":
            break
        try:
            obj, pos = decoder.raw_decode(text, pos)
        except ValueError:
            break  # the damaged entry — stop here, keep the complete prefix
        items.append(obj)
    return items


def _parse_items(raw: str, key: str) -> list[Any]:
    """The ``key`` array of a model JSON reply (fences, prose, unescaped quotes).

    A clean decode wins; otherwise the reply is salvaged entry-wise. An empty
    array is a real answer and is returned as one, but a reply nothing can be
    recovered from raises :class:`LinkPlanFailed` instead of reading as an empty
    one — see that class for why the difference matters here.
    """
    try:
        payload = loads_tolerant(strip_fences(raw))
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict):
        items = payload.get(key)
        return items if isinstance(items, list) else []
    salvaged = _salvage_items(raw, key)
    if not salvaged:
        raise LinkPlanFailed(f"the model's {key} reply could not be decoded")
    return salvaged


SYSTEM_PROMPT = prompt("link_system")

VERIFY_SYSTEM_PROMPT = prompt("link_verify")


@dataclass
class Link:
    subject: str
    objects: list[str]
    predicate: str
    value: str = ""
    href: str = field(default="")


def _elements(root: etree._Element) -> list[etree._Element]:
    return [el for el in root.iter() if isinstance(el.tag, str)]


# Indentation is capped so a deeply nested tree cannot spend most of a line on
# leading spaces, and each element's own text is capped so one long paragraph
# cannot crowd out the rest of the document.
_MAX_INDENT = 8
_MAX_OWN_TEXT = 220


def _own_text(el: etree._Element) -> str:
    """The text belonging to *el* itself — never a descendant's.

    That is ``el.text`` plus the tail of each child, which together are the
    character data directly under this element.
    """
    parts = [el.text or ""]
    for child in el:
        parts.append(child.tail or "")
    # Joined with a space, not concatenated: the child's own text sits on its
    # own line, so running the words either side of it together would invent a
    # word that is not in the document ("provide" + "monthly").
    return " ".join(" ".join(parts).split())


def _depth(el: etree._Element) -> int:
    """How many ancestors *el* has; 0 for the root."""
    return sum(1 for _ in el.iterancestors())


def _listing(elements: list[etree._Element]) -> str:
    """One line per element: id, nesting depth, tag name, and its own text.

    Each element contributes only the text directly under it. Using the whole
    subtree instead would repeat a clause's opening words on the clause, on its
    section, and on the root — the same characters billed once per level of
    nesting, which on a deep document is most of the prompt.

    Every element still gets a line, including one with no text of its own, so
    both ends of a link stay addressable by id. Elements are listed in document
    order and the indent shows nesting, so the model can still read a clause
    across the lines it is split over.

    (``_snip``, which labels an already-chosen candidate for the review pass,
    deliberately keeps using the full subtree: there the point is to identify
    one element in a short phrase, not to lay out the whole document.)
    """
    lines = []
    for i, el in enumerate(elements):
        name = etree.QName(el).localname
        pad = "  " * min(_depth(el), _MAX_INDENT)
        lines.append(f"e{i:04d} {pad}<{name}>: {_own_text(el)[:_MAX_OWN_TEXT]}")
    return "\n".join(lines)


def _slug(el: etree._Element) -> str:
    name = str(etree.QName(el).localname)
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "el"


def _ensure_id(el: etree._Element, used: set[str]) -> str:
    existing = el.get(f"{{{_XML}}}id")
    if existing:
        return str(existing)
    base = _slug(el)
    xid, n = base, 1
    while xid in used:
        n += 1
        xid = f"{base}-{n}"
    used.add(xid)
    el.set(f"{{{_XML}}}id", xid)
    return xid


@dataclass
class _Candidate:
    subject: int  # element index
    objects: list[int]
    predicate: str
    value: str


def _idx_resolver(n: int) -> Callable[[object], int | None]:
    def _idx(eid: object) -> int | None:
        m = re.fullmatch(r"e(\d+)", str(eid).strip())
        i = int(m.group(1)) if m else -1
        return i if 0 <= i < n else None

    return _idx


def _propose(elements: list[etree._Element], config: llm.LLMConfig) -> list[_Candidate]:
    # call_continued, not call: a length-truncated proposal resumes from where
    # it stopped instead of being lost whole. Headroom is comfortable today, so
    # this normally costs exactly one call.
    raw = llm.call_continued(
        config,
        system_prompt=SYSTEM_PROMPT,
        user_content=[{"type": "text", "text": _listing(elements)}],
        cache=True,
    )
    idx = _idx_resolver(len(elements))
    cands: list[_Candidate] = []
    for item in _parse_items(raw, "links"):
        if not isinstance(item, dict):
            continue  # salvage can hand back a non-object entry
        si = idx(item.get("subject", ""))
        raw_objs = item.get("object", "")
        obj_eids = raw_objs if isinstance(raw_objs, list) else [raw_objs]
        obj_idxs = [oi for e in obj_eids if (oi := idx(e)) is not None and oi != si]
        if si is None or not obj_idxs:
            continue
        cands.append(
            _Candidate(
                si,
                obj_idxs,
                str(item.get("predicate") or "references"),
                str(item.get("value") or ""),
            )
        )
    return cands


def _snip(el: etree._Element) -> str:
    return " ".join("".join(el.itertext()).split())[:90]


def _verify(
    elements: list[etree._Element], cands: list[_Candidate], config: llm.LLMConfig
) -> list[_Candidate]:
    lines = []
    for i, c in enumerate(cands):
        subj = f'<{etree.QName(elements[c.subject]).localname}> "{_snip(elements[c.subject])}"'
        objs = "; ".join(
            f'<{etree.QName(elements[o]).localname}> "{_snip(elements[o])}"' for o in c.objects
        )
        val = f" value={c.value}" if c.value else ""
        lines.append(f"L{i}: {subj} --{c.predicate}{val}--> {objs}")
    raw = llm.call_continued(
        config,
        system_prompt=VERIFY_SYSTEM_PROMPT,
        user_content=[{"type": "text", "text": "\n".join(lines)}],
        cache=True,
    )
    verdicts = _parse_items(raw, "verdicts")
    if not verdicts:
        # Not a review that rejected everything — a review that never happened.
        # Dropping every candidate here is what used to get cached as "this
        # document has no links".
        raise LinkPlanFailed(f"the reviewer returned no verdict on {len(cands)} candidate link(s)")
    kept = {v.get("i") for v in verdicts if isinstance(v, dict) and v.get("keep")}
    return [c for i, c in enumerate(cands) if i in kept]


def listing_digest(xml: str) -> str:
    """Hash of what a link plan for *xml* actually depends on: its text and shape.

    Two things the model *is* shown are deliberately left out.

    Attributes never appear in the prompt at all, so grounding a document with
    ``dg:origin`` boxes or moving concepts to another namespace prefix must not
    make the pass run again.

    Tag names do appear in the prompt, but a plan names neither end of a link —
    :func:`apply_plan` addresses both by position in document order — so a plan
    computed before a concept was renamed still lands on exactly the same
    elements. Keying on them meant every roster change re-linked each
    already-generated document it re-rendered, at full price: the Nth file added
    to a docset paid for N link passes rather than one. The trade is that a
    rename reuses the plan the old names produced instead of asking again.
    """
    root = etree.fromstring(xml.encode())
    shape = "\n".join(
        f"{min(_depth(el), _MAX_INDENT)}\t{_own_text(el)[:_MAX_OWN_TEXT]}" for el in _elements(root)
    )
    return hashlib.sha256(shape.encode("utf-8")).hexdigest()


def plan_links(xml: str, config: llm.LLMConfig, *, verify: bool = True) -> list[dict[str, Any]]:
    """Ask the model which links *xml* should carry.

    Every model call for the pass happens here. The answer comes back as plain
    data, with both ends of a link given as positions in document order, so a
    caller can store it and apply it later with :func:`apply_plan`.

    An empty plan means the model found nothing to link. A call it could not
    read an answer out of raises :class:`LinkPlanFailed` rather than returning
    one, so a caller that caches the result never stores a failure as a fact.
    """
    root = etree.fromstring(xml.encode())
    elements = _elements(root)
    # Propose + verify fold into one usage row (gated on --debug via the config).
    with llm.record_usage_for(config):
        cands = _propose(elements, config)
        if verify and cands:
            cands = _verify(elements, cands, config)
    return [
        {
            "subject": c.subject,
            "objects": list(c.objects),
            "predicate": c.predicate,
            "value": c.value,
        }
        for c in cands
    ]


def apply_plan(xml: str, plan: list[dict[str, Any]]) -> tuple[str, list[Link]]:
    """Write a plan from :func:`plan_links` into *xml*. Makes no model call.

    Entries pointing outside the tree are skipped, so a plan applied to a
    document it was not built for produces fewer links instead of raising.
    """
    root = etree.fromstring(xml.encode())
    elements = _elements(root)
    used: set[str] = {i for el in elements if (i := el.get(f"{{{_XML}}}id"))}
    applied: list[Link] = []
    count = len(elements)
    for item in plan:
        subject_idx = item.get("subject")
        obj_idxs = [
            o
            for o in (item.get("objects") or [])
            if isinstance(o, int) and 0 <= o < count and o != subject_idx
        ]
        if not isinstance(subject_idx, int) or not 0 <= subject_idx < count or not obj_idxs:
            continue
        predicate = str(item.get("predicate") or "references")
        obj_ids = [_ensure_id(elements[o], used) for o in obj_idxs]
        subj_id = _ensure_id(elements[subject_idx], used)
        href = " ".join(f"#{oid}" for oid in obj_ids)
        subject = elements[subject_idx]
        subject.set(f"{{{_DG}}}itemprop", predicate)
        subject.set(f"{{{_DG}}}href", href)
        # On a TYPED element (xsi:type present) dg:value already holds the
        # normalized typed value — writing the link payload over it would make
        # the xsi:type/dg:value pair inconsistent (e.g. decimal + "$2,500,000").
        # The typed value wins; the link keeps itemprop + href.
        value = "" if subject.get(f"{{{_XSI}}}type") else str(item.get("value") or "")
        if value:
            subject.set(f"{{{_DG}}}value", value)
        applied.append(Link(subj_id, obj_ids, predicate, value, href))

    body = etree.tostring(root, encoding="unicode")
    return f"<?xml version='1.0' encoding='utf-8'?>\n{body}\n", applied


def add_links(xml: str, config: llm.LLMConfig, *, verify: bool = True) -> tuple[str, list[Link]]:
    """Add semantic links to *xml*; return (linked xml, applied links).

    Proposes links, verifies them with a skeptical second pass (unless
    *verify* is False), then applies the survivors.
    """
    return apply_plan(xml, plan_links(xml, config, verify=verify))
