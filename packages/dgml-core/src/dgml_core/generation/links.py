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

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from lxml import etree  # type: ignore[import-untyped]

from dgml_core import llm
from dgml_core.generation.prompts import get as prompt
from dgml_core.generation.transcribe import loads_tolerant, strip_fences

_DG = "http://dgml.io/ns/dg#"
_XML = "http://www.w3.org/XML/1998/namespace"
_XSI = "http://www.w3.org/2001/XMLSchema-instance"


def _parse_json(raw: str) -> dict[str, Any]:
    """Tolerantly parse a model JSON reply (fences, prose, unescaped quotes)."""
    try:
        out = loads_tolerant(strip_fences(raw))
    except (ValueError, TypeError):
        return {}
    return out if isinstance(out, dict) else {}


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


def _listing(elements: list[etree._Element]) -> str:
    lines = []
    for i, el in enumerate(elements):
        name = etree.QName(el).localname
        text = " ".join("".join(el.itertext()).split())[:220]
        lines.append(f"e{i:04d} <{name}>: {text}")
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
    raw = llm.call(
        config,
        system_prompt=SYSTEM_PROMPT,
        user_content=[{"type": "text", "text": _listing(elements)}],
        cache=True,
    )
    payload = _parse_json(raw)
    idx = _idx_resolver(len(elements))
    cands: list[_Candidate] = []
    for item in payload.get("links", []) or []:
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
    raw = llm.call(
        config,
        system_prompt=VERIFY_SYSTEM_PROMPT,
        user_content=[{"type": "text", "text": "\n".join(lines)}],
        cache=True,
    )
    payload = _parse_json(raw)
    kept = {v.get("i") for v in payload.get("verdicts", []) or [] if v.get("keep")}
    return [c for i, c in enumerate(cands) if i in kept]


def add_links(xml: str, config: llm.LLMConfig, *, verify: bool = True) -> tuple[str, list[Link]]:
    """Add semantic links to *xml*; return (linked xml, applied links).

    Proposes links, verifies them with a skeptical second pass (unless
    *verify* is False), then applies the survivors.
    """
    root = etree.fromstring(xml.encode())
    elements = _elements(root)
    # Propose + verify fold into one usage row (gated on --debug via the config).
    with llm.record_usage_for(config):
        cands = _propose(elements, config)
        if verify and cands:
            cands = _verify(elements, cands, config)

    used: set[str] = {i for el in elements if (i := el.get(f"{{{_XML}}}id"))}
    applied: list[Link] = []
    for c in cands:
        obj_ids = [_ensure_id(elements[o], used) for o in c.objects]
        subj_id = _ensure_id(elements[c.subject], used)
        href = " ".join(f"#{oid}" for oid in obj_ids)
        subject = elements[c.subject]
        subject.set(f"{{{_DG}}}itemprop", c.predicate)
        subject.set(f"{{{_DG}}}href", href)
        # On a TYPED element (xsi:type present) dg:value already holds the
        # normalized typed value — writing the link payload over it would make
        # the xsi:type/dg:value pair inconsistent (e.g. decimal + "$2,500,000").
        # The typed value wins; the link keeps itemprop + href.
        value = "" if subject.get(f"{{{_XSI}}}type") else c.value
        if value:
            subject.set(f"{{{_DG}}}value", value)
        applied.append(Link(subj_id, obj_ids, c.predicate, value, href))

    body = etree.tostring(root, encoding="unicode")
    return f"<?xml version='1.0' encoding='utf-8'?>\n{body}\n", applied
