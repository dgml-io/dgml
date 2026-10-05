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

"""`generate_docset` — the library entry point behind `dgml docset generate`.

The CLI suite (packages/dgml/tests/test_cli.py) exercises the orchestration
end to end through `main()` and pins the JSON payload; these tests cover what
is library-specific — the typed report's serialization contract, the raised
preconditions, and the schema-seed loaders that moved here from `cli.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from dgml_core import DocSetStore, EmptyDocSet, GenerateFileResult, GenerateReport, Workspace
from dgml_core.docset_generate import _load_schema_roster, _load_schema_seed, generate_docset
from dgml_core.errors import DocSetNotFound, InvalidArgument

# ---------------------------------------------------------------------------
# Raised preconditions
# ---------------------------------------------------------------------------


def test_generate_docset_raises_for_missing_docset(workspace: Workspace) -> None:
    with pytest.raises(DocSetNotFound):
        generate_docset(workspace, "ds_nope")


def test_generate_docset_raises_empty_docset(workspace: Workspace) -> None:
    """A DocSet with no files assigned is a precondition failure, not an empty
    report — EMPTY_DOCSET, the code the CLI has always emitted for it."""
    ds = DocSetStore(workspace).create(name="Empty")
    with pytest.raises(EmptyDocSet) as excinfo:
        generate_docset(workspace, ds.id)
    assert excinfo.value.code == "EMPTY_DOCSET"
    assert ds.id in str(excinfo.value)


# ---------------------------------------------------------------------------
# Serialization contract — `to_json` must render exactly the CLI's rows
# ---------------------------------------------------------------------------


def test_file_result_rows_match_the_cli_shape() -> None:
    """Conditional keys appear exactly when their pass ran or failed, in the
    CLI's key order — the report is the payload, so the two cannot drift."""
    skipped = GenerateFileResult(status="skipped", file_id="f1", source="a.pdf", output="k/a")
    assert skipped.to_json() == {
        "status": "skipped",
        "file_id": "f1",
        "source": "a.pdf",
        "output": "k/a",
    }

    failed = GenerateFileResult(
        status="failed", file_id="f2", source="b.pdf", error={"code": "X", "message": "m"}
    )
    assert failed.to_json() == {
        "status": "failed",
        "file_id": "f2",
        "source": "b.pdf",
        "error": {"code": "X", "message": "m"},
    }

    grounded = GenerateFileResult(
        status="converted",
        file_id="f3",
        source="c.pdf",
        output="k/c",
        links=2,
        grounded=True,
        matched_token_pct=98.5,
        elements_annotated=40,
    )
    assert grounded.to_json() == {
        "status": "converted",
        "file_id": "f3",
        "source": "c.pdf",
        "output": "k/c",
        "links": 2,
        "grounded": True,
        "matched_token_pct": 98.5,
        "elements_annotated": 40,
    }

    ungrounded = GenerateFileResult(
        status="converted",
        file_id="f4",
        source="d.pdf",
        output="k/d",
        links=0,
        grounded=False,
        grounding_error={"code": "GROUNDING_FAILED", "message": "no page_text"},
        label_error={"code": "LABEL_MODEL_UNREACHABLE", "message": "401"},
        link_error="rate limited",
    )
    row = ungrounded.to_json()
    assert row["grounded"] is False
    assert row["grounding_error"] == {"code": "GROUNDING_FAILED", "message": "no page_text"}
    assert "matched_token_pct" not in row and "elements_annotated" not in row
    assert row["label_error"] == {"code": "LABEL_MODEL_UNREACHABLE", "message": "401"}
    assert row["link_error"] == "rate limited"


def test_off_schema_concepts_key_follows_extend_schema() -> None:
    """The same facts render as `unmatched_concepts` under a strict schema
    (refused) and `added_concepts` under extend-schema (coined and used)."""
    tally = {"count": 3, "distinct": 2, "examples": ["Foo", "Bar"]}
    result = GenerateFileResult(
        status="converted",
        file_id="f1",
        source="a.pdf",
        output="k/a",
        links=0,
        grounded=True,
        matched_token_pct=100.0,
        elements_annotated=1,
        off_schema_concepts=tally,
    )
    assert result.to_json(extend_schema=False)["unmatched_concepts"] == tally
    assert result.to_json(extend_schema=True)["added_concepts"] == tally


def test_report_to_json_builds_the_generate_envelope(workspace: Workspace) -> None:
    """The envelope the CLI prints: summary counts come from the three lists,
    results concatenate in skipped + failed + converted order."""
    ds = DocSetStore(workspace).create(name="Invoices")
    skipped = GenerateFileResult(status="skipped", file_id="f1", source="a.pdf", output="k/a")
    failed = GenerateFileResult(
        status="failed", file_id="f2", source="b.pdf", error={"code": "X", "message": "m"}
    )
    converted = GenerateFileResult(
        status="converted",
        file_id="f3",
        source="c.pdf",
        output="k/c",
        links=1,
        grounded=True,
        matched_token_pct=99.0,
        elements_annotated=7,
    )
    report = GenerateReport(
        docset=ds,
        total=3,
        skipped=[skipped],
        failed=[failed],
        converted=[converted],
        rerendered=["a.pdf"],
        output_key=f"docsets/{ds.id}",
        coverage_report_key=None,
        model="anthropic/claude-haiku-4-5",
        label_model="anthropic/claude-sonnet-5",
        model_source="workspace config",
        schema_extended=False,
    )
    assert report.results == [skipped, failed, converted]
    assert report.to_json() == {
        "docset_id": ds.id,
        "docset_name": "Invoices",
        "summary": {"total": 3, "converted": 1, "skipped": 1, "failed": 1},
        "models": {
            "model": "anthropic/claude-haiku-4-5",
            "label_model": "anthropic/claude-sonnet-5",
            "source": "workspace config",
        },
        "output_key": f"docsets/{ds.id}",
        "coverage_report": None,
        "results": [skipped.to_json(), failed.to_json(), converted.to_json()],
        "rerendered": ["a.pdf"],
    }


# ---------------------------------------------------------------------------
# Schema-seed loaders (moved from cli.py with `docset generate`)
# ---------------------------------------------------------------------------


def test_load_schema_roster_errors(tmp_path: Path) -> None:
    """_load_schema_roster rejects missing files, non-object/invalid JSON, and
    rosters that sanitize to no usable concepts — all as InvalidArgument."""
    with pytest.raises(InvalidArgument):
        _load_schema_roster(tmp_path / "missing.json")

    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_roster(arr)

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_roster(bad)

    junk = tmp_path / "junk.json"
    junk.write_text(json.dumps({"###": "x", "!!!": "y"}), encoding="utf-8")
    with pytest.raises(InvalidArgument):
        _load_schema_roster(junk)


def test_load_schema_seed_json_builds_roster_and_parent_map(tmp_path: Path) -> None:
    p = tmp_path / "schema.json"
    p.write_text(
        json.dumps(
            {
                "tags": {
                    "PartyInformation": {
                        "name": "PartyInformation",
                        "role": "party block",
                        "kind": "section",
                    },
                    "PartyAddress": {
                        "name": "PartyAddress",
                        "role": "address",
                        "parent_role": "PartyInformation",
                    },
                    "OrderDate": {"name": "OrderDate", "role": "order date"},
                }
            }
        ),
        encoding="utf-8",
    )
    schema, parent_map, _notes = _load_schema_seed(p)
    assert {"PartyInformation", "PartyAddress", "OrderDate"} <= set(schema.tags)
    assert schema.tags["PartyAddress"].role == "address"
    assert schema.tags["PartyInformation"].kind == "section"  # fidelity kept, not flattened
    assert parent_map["PartyAddress"] == "PartyInformation"  # via parent_role
    assert "OrderDate" not in parent_map  # top-level, no container


def test_load_schema_seed_accepts_rnc(tmp_path: Path) -> None:
    """--schema-path also accepts the lossless full-schema.rnc render — the
    `# Field: value` comment contract reconstructs the same roster/parent_map."""
    p = tmp_path / "full-schema.rnc"
    p.write_text(
        "# " + "-" * 20 + "\n"
        '# Description: "party block"\n'
        "# Kind: section\n"
        "PartyInformation = element PartyInformation {\n  common.atts,\n"
        "  mixed { any.docset* }\n}\n\n"
        "# " + "-" * 20 + "\n"
        '# Description: "address"\n'
        "# Kind: inline\n"
        "# Parent: PartyInformation\n"
        "PartyAddress = element PartyAddress {\n  common.atts,\n  text\n}\n",
        encoding="utf-8",
    )
    schema, parent_map, _notes = _load_schema_seed(p)
    assert {tag.name: tag.role for tag in schema.tags.values()} == {
        "PartyInformation": "party block",
        "PartyAddress": "address",
    }
    assert parent_map == {"PartyAddress": "PartyInformation"}


def test_load_schema_seed_accepts_a_plain_tag_list(tmp_path: Path) -> None:
    """Form A — one bare tag name per line, `#` comments and blanks ignored.

    Names are taken VERBATIM: `Notes` and `Details` survive (`sanitize_concept`
    would fold both to ''), and only XML validity is enforced, which turns
    `Line Items` into `Line_Items` and says so in the notes."""
    p = tmp_path / "tags.txt"
    p.write_text(
        "# Liquor distribution purchase orders\n\nCustomerName\nNotes\n"
        "Details\nLine Items\nAgreementSummary\n",
        encoding="utf-8",
    )
    schema, parent_map, notes = _load_schema_seed(p)
    assert list(schema.tags) == [
        "CustomerName",
        "Notes",
        "Details",
        "Line_Items",
        "AgreementSummary",
    ]
    assert all(tag.kind == "inline" for tag in schema.tags.values())
    assert parent_map == {}
    assert any("Line Items" in note and "Line_Items" in note for note in notes)
    assert any("kind=inline" in note for note in notes)


def test_load_schema_seed_accepts_a_name_to_description_mapping(tmp_path: Path) -> None:
    """Form B — the recommended shape. Previously rejected outright."""
    p = tmp_path / "schema.json"
    p.write_text(
        json.dumps({"BuyerName": "bill-to org", "OrderDate": "date the order was placed"}),
        encoding="utf-8",
    )
    schema, parent_map, _notes = _load_schema_seed(p)
    assert {n: t.role for n, t in schema.tags.items()} == {
        "BuyerName": "bill-to org",
        "OrderDate": "date the order was placed",
    }
    assert parent_map == {}


def test_load_schema_seed_rejects_ambiguous_input(tmp_path: Path) -> None:
    """What cannot be read UNAMBIGUOUSLY fails at load, never as a tag that
    quietly failed to appear in the output hours later."""

    def _reject(name: str, text: str) -> str:
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        with pytest.raises(InvalidArgument) as excinfo:
            _load_schema_seed(p)
        return str(excinfo.value)

    # A `key: value` file (YAML, or Form B written as text) would otherwise
    # load as tags named "concepts_" and "BuyerName__bill-to_org".
    assert "':'" in _reject("seed.txt", "concepts:\n  BuyerName: bill-to org\n")
    # Two names that differ only in case/punctuation are one tag, twice.
    assert "differ only in case" in _reject("dup.txt", "BuyerName\nbuyer_name\n")
    # A kind outside VALID_KINDS is a typo, not a silent coercion to `inline`.
    assert "kind" in _reject("k.json", json.dumps({"tags": {"A": {"role": "x", "kind": "field"}}}))
    # A parent_role naming a tag that does not exist would synthesize a
    # container outside the vocabulary.
    assert "parent_role" in _reject(
        "p.json", json.dumps({"tags": {"A": {"role": "x", "parent_role": "Nope"}}})
    )
    # An empty `tags` map, and a JSON array (neither form).
    assert "no tags" in _reject("e.json", json.dumps({"tags": {}}))
    assert "array" in _reject("a.json", json.dumps(["A", "B"]))

    with pytest.raises(InvalidArgument):
        _load_schema_seed(tmp_path / "missing.json")
