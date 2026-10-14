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

import pytest
from dgml_core.dg_refs import (
    Ref,
    effective_inputs,
    effective_method,
    effective_origin,
    format_refs,
    parse_ref,
    parse_refs,
    split_refs,
)
from lxml import etree  # type: ignore[import-untyped]

_NS = 'xmlns:dg="http://dgml.io/ns/dg#"'


def _el(attrs: str) -> etree._Element:
    return etree.fromstring(f"<dg:F {_NS} {attrs}>x</dg:F>")


@pytest.mark.parametrize(
    "attr",
    ["#a; #b", "#a #b", "#a;#b", " #a ;\n #b; "],
)
def test_split_refs_accepts_semicolons_and_whitespace(attr: str) -> None:
    assert split_refs(attr) == ["#a", "#b"]


def test_split_refs_of_nothing() -> None:
    assert split_refs(None) == [] and split_refs("") == []


def test_parse_ref_forms() -> None:
    assert parse_ref("#a") == Ref(None, "a")
    assert parse_ref("5kqt9r5fowno#notice-1") == Ref("5kqt9r5fowno", "notice-1")
    assert parse_ref("bare") is None
    assert parse_ref("#") is None
    assert parse_ref("f#a#b") is None


def test_parse_refs_skips_malformed_and_format_round_trips() -> None:
    refs = parse_refs("#a; junk; f1#b")
    assert refs == [Ref(None, "a"), Ref("f1", "b")]
    assert refs[0].local and not refs[1].local
    assert format_refs(refs) == "#a; f1#b"


def test_boxed_value_has_no_method() -> None:
    el = _el('dg:origin="2 410 980 520 1025" dg:inputs="#commencement" dg:value="2026-03-01"')
    assert effective_origin(el) == "2 410 980 520 1025"
    assert effective_inputs(el) == ["#commencement"]
    assert effective_method(el) is None


def test_stated_method_wins() -> None:
    assert effective_method(_el('dg:value="0" dg:method="defaulted"')) == "defaulted"
    el = _el('dg:value="12" dg:inputs="#s; #e" dg:method="calculated"')
    assert effective_method(el) == "calculated"


def test_missing_method_with_inputs_reads_as_inferred() -> None:
    assert effective_method(_el('dg:value="EMEA" dg:inputs="#hq"')) == "inferred"


def test_no_origin_inputs_or_method_is_unsupported() -> None:
    assert effective_method(_el('dg:value="EMEA"')) is None
    assert effective_method(_el('dg:value="EMEA" dg:method="guessed"')) is None


def test_legacy_computed_origin_reads_href_as_inputs() -> None:
    el = _el('dg:origin="computed" dg:value="10" dg:itemprop="computedFrom" dg:href="#p1; #p2"')
    assert effective_origin(el) is None
    assert effective_inputs(el) == ["#p1", "#p2"]
    assert effective_method(el) == "inferred"


def test_semantic_href_is_not_an_input() -> None:
    el = _el('dg:origin="1 1 1 2 2" dg:itemprop="references" dg:href="#a"')
    assert effective_inputs(el) == []
