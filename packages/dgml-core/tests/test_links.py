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
from dgml_core.generation.links import _parse_json, add_links
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

    monkeypatch.setattr(llm, "call", fake_call)


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


def test_parse_json_tolerates_fences_and_prose() -> None:
    assert _parse_json('```json\n{"links": []}\n```') == {"links": []}
    assert _parse_json('sure: {"links": [{"predicate": "x"}]} done')["links"][0]["predicate"] == "x"
    assert _parse_json("not json at all") == {}


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
