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

"""Tests for the semantic-link pass (dgml_core.generation.links)."""

from __future__ import annotations

import json

import pytest
from dgml_core import llm
from dgml_core.errors import LinkPlanFailed
from dgml_core.generation.links import _parse_items, add_links
from lxml import etree  # type: ignore[import-untyped]

_DG = "http://dgml.io/ns/dg#"
_XMLID = "{http://www.w3.org/XML/1998/namespace}id"

# element order under root: e0000=chunk, 1=Commencement, 2=Adjustment, 3=BaseRent, 4=Escalation
_XML = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">'
    "<dg:CommencementDate>November 1, 2024</dg:CommencementDate>"
    "<dg:AdjustmentDate>each anniversary of the Commencement Date</dg:AdjustmentDate>"
    "<dg:BaseRent>100</dg:BaseRent>"
    "<dg:Escalation>the greater of (a) or (b)</dg:Escalation>"
    "</dg:chunk>"
)


def _fake_llm(
    monkeypatch: pytest.MonkeyPatch, links: list[dict[str, object]], keep: list[bool]
) -> None:
    def fake_call(config: llm.LLMConfig, **kwargs: object) -> str:
        if "reviewer" in str(kwargs["system_prompt"]):
            return json.dumps({"verdicts": [{"i": i, "keep": k} for i, k in enumerate(keep)]})
        return json.dumps({"links": links})

    monkeypatch.setattr(llm, "call_continued", fake_call)


def test_add_links_applies_relative_and_multi_target_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_llm(
        monkeypatch,
        links=[
            {"subject": "e0002", "object": "e0001", "predicate": "relativeTo", "value": "P1Y"},
            {"subject": "e0004", "object": ["e0001", "e0003"], "predicate": "greaterOf"},
        ],
        keep=[True, True],
    )
    linked, applied = add_links(_XML, llm.LLMConfig(model="x"))
    root = etree.fromstring(linked.encode())
    by = {etree.QName(e).localname: e for e in root.iter() if isinstance(e.tag, str)}
    ids = {e.get(_XMLID) for e in root.iter() if e.get(_XMLID)}

    adj = by["AdjustmentDate"]
    assert adj.get(f"{{{_DG}}}itemprop") == "relativeTo"
    assert adj.get(f"{{{_DG}}}value") == "P1Y"
    assert adj.get(f"{{{_DG}}}href") == "#" + by["CommencementDate"].get(_XMLID)

    esc = by["Escalation"]
    targets = [t.lstrip("#") for t in esc.get(f"{{{_DG}}}href").split()]
    assert len(targets) == 2 and all(t in ids for t in targets)  # multi-target href resolves
    assert len(applied) == 2


def test_verify_drops_rejected_links(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_llm(
        monkeypatch,
        links=[
            {"subject": "e0002", "object": "e0001", "predicate": "relativeTo", "value": "P1Y"},
            {"subject": "e0004", "object": "e0003", "predicate": "greaterOf"},
        ],
        keep=[True, False],
    )
    linked, applied = add_links(_XML, llm.LLMConfig(model="x"))
    assert len(applied) == 1 and applied[0].predicate == "relativeTo"
    root = etree.fromstring(linked.encode())
    esc = next(e for e in root.iter() if etree.QName(e).localname == "Escalation")
    assert esc.get(f"{{{_DG}}}itemprop") is None  # dropped link left unlinked


def test_parse_items_tolerates_fences_and_prose() -> None:
    assert _parse_items('```json\n{"links": []}\n```', "links") == []
    assert _parse_items('sure: {"links": [{"predicate": "x"}]} done', "links") == [
        {"predicate": "x"}
    ]
    # Nothing recoverable is an error, not an empty answer: returning [] here
    # would be cached as "this document has no links" and never retried.
    with pytest.raises(LinkPlanFailed):
        _parse_items("not json at all", "links")


def test_link_value_never_clobbers_a_typed_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a TYPED subject (xsi:type present) dg:value holds the normalized typed
    value; the link payload must not overwrite it (else xsi:type/dg:value turn
    inconsistent, e.g. decimal + "$100"). Untyped subjects still take the payload."""
    xml = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dg:CapAmount>100</dg:CapAmount>"
        '<dg:FeeAmount xsi:type="decimal" dg:value="100">$100</dg:FeeAmount>'
        "<dg:DueDate>seven days after the cap is set</dg:DueDate>"
        "</dg:chunk>"
    )
    _fake_llm(
        monkeypatch,
        links=[
            {"subject": "e0002", "object": "e0001", "predicate": "valueFrom", "value": "$100"},
            {"subject": "e0003", "object": "e0001", "predicate": "relativeTo", "value": "P7D"},
        ],
        keep=[True, True],
    )
    linked, applied = add_links(xml, llm.LLMConfig(model="x"))
    root = etree.fromstring(linked.encode())
    by = {etree.QName(e).localname: e for e in root.iter() if isinstance(e.tag, str)}

    fee = by["FeeAmount"]  # typed: link applied, but typed dg:value kept
    assert fee.get(f"{{{_DG}}}itemprop") == "valueFrom"
    assert fee.get(f"{{{_DG}}}value") == "100"
    due = by["DueDate"]  # untyped: link payload lands in dg:value
    assert due.get(f"{{{_DG}}}value") == "P7D"
    assert [ln.value for ln in applied] == ["", "P7D"]  # reported value mirrors the XML


def test_listing_digest_ignores_attributes() -> None:
    """The prompt shows tag names and text, never attributes — so grounding a
    document or moving concepts to another namespace prefix must not change its
    cache key, or every re-render would pay for the pass again."""
    from dgml_core.generation.links import listing_digest

    grounded = _XML.replace("<dg:BaseRent>", '<dg:BaseRent dg:origin="1,2,3,4" dg:style="bold">')
    assert listing_digest(_XML) == listing_digest(grounded)

    renamed = _XML.replace("dg:BaseRent", "docset:BaseRent").replace(
        'xmlns:dg="http://dgml.io/ns/dg#"',
        'xmlns:dg="http://dgml.io/ns/dg#" xmlns:docset="http://dgml.io/ns/docset#"',
    )
    assert listing_digest(_XML) == listing_digest(renamed)

    # Changing text the model DOES see must change the key.
    assert listing_digest(_XML) != listing_digest(_XML.replace("100", "250"))


def _renamed(xml: str) -> str:
    """Every concept tag in the fixture renamed — what a roster revision does."""
    for old, new in (
        ("CommencementDate", "StartDate"),
        ("AdjustmentDate", "ReviewDate"),
        ("BaseRent", "AnnualRent"),
        ("Escalation", "RentIncrease"),
    ):
        xml = xml.replace(f"dg:{old}", f"dg:{new}")
    return xml


def _linked_positions(xml: str, plan: list[dict[str, object]]) -> list[tuple[int, str, list[int]]]:
    """Every applied link as (subject position, predicate, object positions).

    Positions, not ids: `_ensure_id` mints an id from the tag name, so a renamed
    tree correctly gets different id strings for the same elements.
    """
    from dgml_core.generation.links import _elements, apply_plan

    linked, _ = apply_plan(xml, plan)
    elements = _elements(etree.fromstring(linked.encode()))
    at = {el.get(_XMLID): i for i, el in enumerate(elements)}
    out = []
    for i, el in enumerate(elements):
        predicate = el.get(f"{{{_DG}}}itemprop")
        if predicate:
            href = el.get(f"{{{_DG}}}href") or ""
            out.append((i, predicate, [at[h.lstrip("#")] for h in href.split()]))
    return out


def test_listing_digest_ignores_tag_names() -> None:
    """Renaming concepts must not re-link the document. A plan addresses both
    ends of a link by position in document order, so the plan the old names
    produced still lands on exactly the same elements — and keying on tag names
    made every roster revision re-link each document it re-rendered."""
    from dgml_core.generation.links import listing_digest

    renamed = _renamed(_XML)
    assert renamed != _XML  # the fixture really did change
    assert listing_digest(_XML) == listing_digest(renamed)

    plan = [
        {"subject": 2, "objects": [1], "predicate": "relativeTo", "value": "P1Y"},
        {"subject": 4, "objects": [1, 3], "predicate": "greaterOf", "value": ""},
    ]
    assert _linked_positions(_XML, plan) == _linked_positions(renamed, plan)


def test_listing_digest_still_tracks_nesting() -> None:
    """Shape is half of what the key covers: moving an element under another
    changes the plan the model would give, so it must change the key."""
    from dgml_core.generation.links import listing_digest

    flat = (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><dg:A>alpha</dg:A><dg:B>beta</dg:B></dg:chunk>'
    )
    nested = (
        '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#"><dg:A>alpha<dg:B>beta</dg:B></dg:A></dg:chunk>'
    )
    assert listing_digest(flat) != listing_digest(nested)


def test_apply_plan_is_deterministic_and_needs_no_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached plan replays without a model call, and lands the same links on a
    re-grounded copy of the same tree."""
    from dgml_core.generation.links import apply_plan

    def explode(*_a: object, **_k: object) -> str:
        raise AssertionError("apply_plan must not call the model")

    monkeypatch.setattr(llm, "call_continued", explode)

    plan = [{"subject": 2, "objects": [1], "predicate": "relativeTo", "value": "P1Y"}]
    first, applied = apply_plan(_XML, plan)
    grounded = _XML.replace("<dg:BaseRent>", '<dg:BaseRent dg:origin="1,2,3,4">')
    second, applied_again = apply_plan(grounded, plan)

    assert [(ln.subject, ln.predicate, ln.href) for ln in applied] == [
        (ln.subject, ln.predicate, ln.href) for ln in applied_again
    ]
    assert 'dg:itemprop="relativeTo"' in first and 'dg:itemprop="relativeTo"' in second


def test_apply_plan_skips_entries_outside_the_tree() -> None:
    """A plan applied to a document it wasn't built for loses links rather than
    raising, so a stale cache entry can never sink a file."""
    from dgml_core.generation.links import apply_plan

    _, applied = apply_plan(
        _XML,
        [
            {"subject": 999, "objects": [1], "predicate": "references", "value": ""},
            {"subject": 2, "objects": [999], "predicate": "references", "value": ""},
            {"subject": 2, "objects": [1], "predicate": "relativeTo", "value": "P1Y"},
        ],
    )
    assert [ln.predicate for ln in applied] == ["relativeTo"]


def test_verify_false_keeps_every_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-semlink-verify path: one model call, nothing filtered."""
    calls: list[str] = []

    def fake_call(config: llm.LLMConfig, **kwargs: object) -> str:
        calls.append(str(kwargs["system_prompt"]))
        return json.dumps(
            {
                "links": [
                    {"subject": "e0002", "object": "e0001", "predicate": "relativeTo"},
                    {"subject": "e0004", "object": "e0003", "predicate": "greaterOf"},
                ]
            }
        )

    monkeypatch.setattr(llm, "call_continued", fake_call)
    _, applied = add_links(_XML, llm.LLMConfig(model="x"), verify=False)
    assert len(applied) == 2
    assert len(calls) == 1 and "reviewer" not in calls[0]


_NESTED = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    '<dg:chunk xmlns:dg="http://dgml.io/ns/dg#">'
    "<dg:Section>TERMS"
    "<dg:Clause>Company shall provide<dg:Item>widgets</dg:Item>monthly</dg:Clause>"
    "</dg:Section>"
    "</dg:chunk>"
)


def _lines(xml: str) -> list[str]:
    from dgml_core.generation.links import _elements, _listing

    root = etree.fromstring(xml.encode())
    return _listing(_elements(root)).splitlines()


def test_listing_does_not_repeat_descendant_text() -> None:
    """A clause's words must appear on the clause, not again on its section and
    the root. Repeating them bills the same characters once per nesting level."""
    lines = _lines(_NESTED)
    section = next(ln for ln in lines if "<Section>" in ln)
    clause = next(ln for ln in lines if "<Clause>" in ln)

    assert section.split(": ", 1)[1] == "TERMS"  # its own text only
    assert "Company" not in section and "widgets" not in section
    assert "widgets" not in clause  # the Item's text lives on the Item's line


def test_listing_keeps_every_element_addressable() -> None:
    """Both ends of a link are named by id, so an element with no text of its
    own still needs a line — otherwise it could never be a subject or object."""
    from dgml_core.generation.links import _elements

    root = etree.fromstring(_NESTED.encode())
    lines = _lines(_NESTED)
    assert len(lines) == len(_elements(root))
    chunk = next(ln for ln in lines if "<chunk>" in ln)
    assert chunk.startswith("e0000") and chunk.endswith(": ")


def test_listing_indent_shows_nesting() -> None:
    """With descendant text gone, the indent is what tells the model a clause
    sits inside a section."""
    lines = _lines(_NESTED)
    indents = [len(ln.split("<")[0]) - len("e0000 ") for ln in lines]
    assert indents == [0, 2, 4, 6]  # chunk, Section, Clause, Item


def test_listing_does_not_run_words_together_across_a_child() -> None:
    """In mixed content the child's text sits on its own line. Joining the text
    either side of it without a space would invent a word ("providemonthly")."""
    clause = next(ln for ln in _lines(_NESTED) if "<Clause>" in ln)
    assert clause.split(": ", 1)[1] == "Company shall provide monthly"


def test_listing_carries_every_word_once() -> None:
    """Nothing is dropped and nothing is duplicated: each word of the document
    appears in the listing exactly once."""
    import collections

    lines = _lines(_NESTED)
    listed = collections.Counter(w for ln in lines for w in ln.split(": ", 1)[1].split())
    assert listed == collections.Counter(
        ["TERMS", "Company", "shall", "provide", "monthly", "widgets"]
    )


# A well-formed proposal, and the two ways the model has been seen to damage
# one: a bare (unquoted) element id inside an object list, observed in
# production; and a length truncation, which loses the tail of the array.
_GOOD_REPLY = json.dumps(
    {
        "links": [
            {"subject": "e0002", "object": "e0001", "predicate": "relativeTo", "value": "P1Y"},
            {"subject": "e0004", "object": ["e0001", "e0003"], "predicate": "greaterOf"},
            {"subject": "e0003", "object": "e0001", "predicate": "valueFrom"},
        ]
    }
)


def test_salvage_keeps_the_links_before_a_malformed_one() -> None:
    """One bad entry used to cost the whole document's links: the reply would
    not decode, so the pass planned zero links and that empty plan was cached."""
    damaged = _GOOD_REPLY.replace('["e0001", "e0003"]', '[e0001, "e0003"]')
    with pytest.raises(ValueError):
        json.loads(damaged)  # the whole reply really is unparseable

    items = _parse_items(damaged, "links")
    assert [i["predicate"] for i in items] == ["relativeTo"]


def test_salvage_keeps_the_complete_prefix_of_a_truncated_reply() -> None:
    """Cut the reply at every offset: never raise, never invent an entry, and
    always recover exactly the entries that survived the cut whole."""
    complete = json.loads(_GOOD_REPLY)["links"]
    seen_counts = set()
    for cut in range(len(_GOOD_REPLY)):
        head = _GOOD_REPLY[:cut]
        try:
            items = _parse_items(head, "links")
        except LinkPlanFailed:
            continue  # nothing recoverable yet — the array has not opened
        assert items == complete[: len(items)], f"invented an entry at cut {cut}"
        seen_counts.add(len(items))
    # Every non-empty prefix is reachable, so the walk really did stop
    # entry-wise rather than all-or-nothing. Zero is never a *result* here: a
    # cut that leaves nothing intact is a failed call, not an empty answer.
    assert seen_counts == {1, 2, 3}


def test_clean_reply_parses_exactly_as_before() -> None:
    """The regression half: salvage is a fallback, never a re-interpretation."""
    assert _parse_items(_GOOD_REPLY, "links") == json.loads(_GOOD_REPLY)["links"]
    assert _parse_items(f"```json\n{_GOOD_REPLY}\n```", "links") == json.loads(_GOOD_REPLY)["links"]


def test_unparseable_proposal_raises_rather_than_planning_no_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan is content-addressed by its document, so "no links" from a
    garbled reply would be cached under that key and never asked again. The
    pass has to fail loudly instead."""
    monkeypatch.setattr(llm, "call_continued", lambda *_a, **_k: "I could not do that.")
    with pytest.raises(LinkPlanFailed):
        add_links(_XML, llm.LLMConfig(model="x"))


def test_reviewer_returning_no_verdicts_raises_rather_than_dropping_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A review that never happened is not a review that rejected everything.
    Unparsed verdicts drop every candidate, which used to be indistinguishable
    from a document with nothing to link — and got cached as one."""

    def fake_call(config: llm.LLMConfig, **kwargs: object) -> str:
        if "reviewer" in str(kwargs["system_prompt"]):
            return "The candidates look mostly fine to me."
        return _GOOD_REPLY

    monkeypatch.setattr(llm, "call_continued", fake_call)
    with pytest.raises(LinkPlanFailed):
        add_links(_XML, llm.LLMConfig(model="x"))

    # An explicit rejection of every candidate is a real answer, and stands.
    def reject_all(config: llm.LLMConfig, **kwargs: object) -> str:
        if "reviewer" in str(kwargs["system_prompt"]):
            return json.dumps({"verdicts": [{"i": i, "keep": False} for i in range(3)]})
        return _GOOD_REPLY

    monkeypatch.setattr(llm, "call_continued", reject_all)
    _, applied = add_links(_XML, llm.LLMConfig(model="x"))
    assert applied == []
