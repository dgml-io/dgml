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


def _parent_positions(elements: list[etree._Element]) -> list[int]:
    """Each element's parent, as a position in *elements* (-1 for the root).

    Derived from nesting depth in one pass rather than from element identity:
    document order is DFS pre-order, so the parent of the element at depth *d*
    is the most recent element seen at depth *d-1*. Positions are also the
    coordinate a plan speaks in, which keeps the whole module on one of them.
    """
    parents = [-1] * len(elements)
    open_at_depth: list[int] = []
    for i, el in enumerate(elements):
        depth = _depth(el)
        del open_at_depth[depth:]  # close everything the new element is not under
        parents[i] = open_at_depth[depth - 1] if depth else -1
        open_at_depth.append(i)
    return parents


def _is_ancestor(parents: list[int], ancestor: int, node: int) -> bool:
    at = parents[node]
    while at != -1:
        if at == ancestor:
            return True
        at = parents[at]
    return False


def _shares_a_path(parents: list[int], a: int, b: int) -> bool:
    """True when one of *a*, *b* is an ancestor of the other.

    A link between two elements on the same root-to-leaf path states a
    relationship the nesting already states. ``link_system`` opens by defining
    a link as a relationship "that the tree's nesting does not capture", so this
    is not a judgement about link quality — it is the one case the format
    excludes by definition.
    """
    return _is_ancestor(parents, a, b) or _is_ancestor(parents, b, a)


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


# Two elements within this many positions of each other in document order are
# neighbours: close enough that a reader meets them together, so a link between
# them is unlikely to be carrying information the layout does not.
_NEIGHBOUR_WITHIN = 3


@dataclass(frozen=True)
class LinkAudit:
    """What a document's own XML says about the links it carries.

    No reference output and no annotation: these are absolute properties of a
    single document, so run-to-run churn — which makes semantic-link output hard
    to score against itself — does not enter.

    Counts are over subject→object *pairs*, not links: a value derived from
    three elements is one link and three pairs, and each pair is judged
    separately.

    Two of the four are defects. ``dangling`` is a broken document: a
    ``dg:href`` that resolves to nothing. ``nested`` contradicts the format
    outright — ``link_system`` defines a link as a relationship the nesting does
    not capture, and a link to one's own ancestor is nothing else.

    The other two are measurements, not verdicts. ``sibling`` and ``neighbour``
    count links between elements a reader meets together, which the prompt asks
    the model to skip as "recoverable from layout alone" — but a document of
    short consecutive clauses can legitimately link neighbours, so a high count
    is a question, not a finding. ``mean_distance`` is context for both.
    """

    links: int
    pairs: int
    nested: int
    sibling: int
    neighbour: int
    dangling: int
    mean_distance: float
    nested_subjects: list[str]
    dangling_hrefs: list[str]


def audit_links(xml: str | bytes) -> LinkAudit:
    """Audit the semantic links in *xml* structurally. Makes no model call."""
    root = etree.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    elements = _elements(root)
    parents = _parent_positions(elements)
    position_of = {xid: i for i, el in enumerate(elements) if (xid := el.get(f"{{{_XML}}}id"))}
    links = pairs = nested = sibling = neighbour = dangling = 0
    distance_total = 0
    nested_subjects: list[str] = []
    dangling_hrefs: list[str] = []
    for subject_idx, subject in enumerate(elements):
        href = subject.get(f"{{{_DG}}}href")
        if not href:
            continue
        links += 1
        for token in href.split():
            pairs += 1
            object_idx = position_of.get(token.lstrip("#"))
            if object_idx is None:
                dangling += 1
                dangling_hrefs.append(token)
                continue
            if _shares_a_path(parents, subject_idx, object_idx):
                nested += 1
                nested_subjects.append(subject.get(f"{{{_XML}}}id") or f"e{subject_idx:04d}")
            if parents[subject_idx] == parents[object_idx]:
                sibling += 1
            distance = abs(subject_idx - object_idx)
            if distance <= _NEIGHBOUR_WITHIN:
                neighbour += 1
            distance_total += distance
    resolved = pairs - dangling
    return LinkAudit(
        links=links,
        pairs=pairs,
        nested=nested,
        sibling=sibling,
        neighbour=neighbour,
        dangling=dangling,
        mean_distance=distance_total / resolved if resolved else 0.0,
        nested_subjects=nested_subjects,
        dangling_hrefs=dangling_hrefs,
    )


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


@dataclass(frozen=True)
class PlanLosses:
    """The plan entries that do not become links, by cause.

    - ``nested`` — every object is on the subject's own root-to-leaf path, so
      the link would state what the nesting states (:func:`_shares_a_path`). An
      entry naming three objects that loses only one to nesting still lands, and
      is not counted here.
    - ``overwritten`` — ``dg:itemprop``/``dg:href`` are attributes *on the
      subject element*, so a second link on one subject displaces the first.
      That is a limit of the format rather than a decision made here, which is
      exactly why it is worth surfacing: without it, a pass silently discards
      about a tenth of what it accepted.
    - ``off_tree`` — the subject, or every object, is not an element of this
      document. Expected when a cached plan meets a re-rendered tree.

    Every entry either becomes a link or lands in one of these, so
    ``len(plan) == len(applied) + total``.
    """

    nested: int
    overwritten: int
    off_tree: int

    @property
    def total(self) -> int:
        return self.nested + self.overwritten + self.off_tree


def _resolve_plan(
    elements: list[etree._Element], plan: list[dict[str, Any]]
) -> tuple[list[tuple[int, list[int], str, str]], PlanLosses]:
    """The plan entries that will reach the XML, in write order, and what did not.

    One walk, shared by :func:`apply_plan` and :func:`plan_losses`, so what gets
    written and what gets counted cannot disagree.
    """
    count = len(elements)
    parents = _parent_positions(elements)
    resolved: list[tuple[int, list[int], str, str]] = []
    at_subject: dict[int, int] = {}
    off_tree = nested = overwritten = 0
    for item in plan:
        subject_idx = item.get("subject")
        objects = [o for o in (item.get("objects") or []) if isinstance(o, int)]
        if not isinstance(subject_idx, int) or not 0 <= subject_idx < count:
            off_tree += 1
            continue
        on_tree = [o for o in objects if 0 <= o < count and o != subject_idx]
        if not on_tree:
            off_tree += 1
            continue
        # A nested object is dropped on its own: a lesser-of formula naming
        # three elements, one of them an ancestor, keeps the other two rather
        # than losing the whole link.
        keep = [o for o in on_tree if not _shares_a_path(parents, subject_idx, o)]
        if not keep:
            nested += 1
            continue
        entry = (
            subject_idx,
            keep,
            str(item.get("predicate") or "references"),
            str(item.get("value") or ""),
        )
        previous = at_subject.get(subject_idx)
        if previous is None:
            at_subject[subject_idx] = len(resolved)
            resolved.append(entry)
        else:
            resolved[previous] = entry  # the XML keeps only the last one
            overwritten += 1
    return resolved, PlanLosses(nested=nested, overwritten=overwritten, off_tree=off_tree)


def plan_losses(xml: str, plan: list[dict[str, Any]]) -> PlanLosses:
    """Which of *plan*'s entries applying it to *xml* would not land. Writes nothing."""
    return _resolve_plan(_elements(etree.fromstring(xml.encode())), plan)[1]


def apply_plan(xml: str, plan: list[dict[str, Any]]) -> tuple[str, list[Link]]:
    """Write a plan from :func:`plan_links` into *xml*. Makes no model call.

    Entries pointing outside the tree are skipped, so a plan applied to a
    document it was not built for produces fewer links instead of raising. So
    are links between an element and its own ancestor or descendant, which the
    format excludes by definition (:func:`_shares_a_path`).

    The returned links are the ones a reader will find in the XML — a link a
    later entry on the same subject overwrote is not among them. Use
    :func:`plan_losses` for what did not make it, and why.
    """
    root = etree.fromstring(xml.encode())
    elements = _elements(root)
    used: set[str] = {i for el in elements if (i := el.get(f"{{{_XML}}}id"))}
    resolved, _ = _resolve_plan(elements, plan)
    applied: list[Link] = []
    for subject_idx, obj_idxs, predicate, raw_value in resolved:
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
        value = "" if subject.get(f"{{{_XSI}}}type") else raw_value
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
