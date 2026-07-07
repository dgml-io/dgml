"""Tests for rnc2jsonld DGML schema converter."""

from __future__ import annotations

import textwrap
from pathlib import Path

from rnc2jsonld import rnc_to_jsonld


def parse(rnc: str) -> dict:
    tmp = Path(__file__).parent / "_tmp.schema.rnc"
    tmp.write_text(textwrap.dedent(rnc).strip(), encoding="utf-8")
    try:
        return rnc_to_jsonld(tmp)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 1. @context is always present with fixed keys
# ---------------------------------------------------------------------------


def test_context_fixed_keys() -> None:
    result = parse("""
        namespace docset = "http://dgml.io/acme-corp/msa#"
        Foo = element docset:Foo { text* }
    """)
    ctx = result["@context"]
    assert ctx["docset"] == "http://dgml.io/acme-corp/msa#"  # URI used as-is from RNC
    assert ctx["xsd"] == "http://www.w3.org/2001/XMLSchema#"
    assert ctx["Tag"] == "docset:Tag"
    assert ctx["TagGroup"] == "docset:TagGroup"
    assert ctx["members"] == {"@id": "docset:members", "@type": "@id", "@container": "@set"}
    assert ctx["description"] == "docset:description"
    assert ctx["example"] == "docset:example"


# ---------------------------------------------------------------------------
# 2. Element definition → Tag node
# ---------------------------------------------------------------------------


def test_element_becomes_tag() -> None:
    result = parse("""
        namespace docset = "http://dgml.io/x/y#"
        LiabilityCap = element docset:LiabilityCap { text* }
    """)
    node = result["@graph"][0]
    assert node["@id"] == "docset:LiabilityCap"
    assert node["@type"] == "docset:Tag"


# ---------------------------------------------------------------------------
# 4. Doc comment → description and example
# ---------------------------------------------------------------------------


def test_doc_comment_description_and_example() -> None:
    result = parse("""
        namespace docset = "http://dgml.io/x/y#"
        ## Agreed maximum liability exposure
        ## Example: $500,000
        LiabilityCap = element docset:LiabilityCap { text* }
    """)
    node = result["@graph"][0]
    assert node["description"] == "Agreed maximum liability exposure"
    assert node["example"] == "$500,000"


# ---------------------------------------------------------------------------
# 5. No doc comment → description and example absent
# ---------------------------------------------------------------------------


def test_no_doc_comment() -> None:
    result = parse("""
        namespace docset = "http://dgml.io/x/y#"
        Foo = element docset:Foo { text* }
    """)
    node = result["@graph"][0]
    assert "description" not in node
    assert "example" not in node


# ---------------------------------------------------------------------------
# 6. Group definition → TagGroup with members
# ---------------------------------------------------------------------------


def test_group_becomes_tag_group() -> None:
    result = parse("""
        namespace docset = "http://dgml.io/x/y#"
        ClauseItemTag = LiabilityCap | EffectiveDate
        LiabilityCap = element docset:LiabilityCap { text* }
        EffectiveDate = element docset:EffectiveDate { text* }
    """)
    groups = [n for n in result["@graph"] if n.get("@type") == "docset:TagGroup"]
    assert len(groups) == 1
    g = groups[0]
    assert g["@id"] == "docset:ClauseItemTag"
    assert set(g["members"]) == {"docset:LiabilityCap", "docset:EffectiveDate"}


# ---------------------------------------------------------------------------
# 7. Namespace URI trailing # is normalised
# ---------------------------------------------------------------------------


def test_namespace_trailing_hash_normalised() -> None:
    result = parse("""
        namespace docset = "http://dgml.io/x/y#"
        Foo = element docset:Foo { text* }
    """)
    assert result["@context"]["docset"] == "http://dgml.io/x/y#"


# ---------------------------------------------------------------------------
# 8. Full sample schema — tag count and group count
# ---------------------------------------------------------------------------


def test_full_sample_schema() -> None:
    sample = Path(__file__).parent.parent / "samples" / "sample.schema.rnc"
    result = rnc_to_jsonld(sample)
    tags = [n for n in result["@graph"] if n.get("@type") == "docset:Tag"]
    groups = [n for n in result["@graph"] if n.get("@type") == "docset:TagGroup"]
    assert len(tags) == 15  # 15 element definitions
    assert (
        len(groups) == 6
    )  # ContractLevelTag, ClauseItemTag, TableRowTag, TableCellTag, InlineTag, AnyTag
