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

"""Tests for the LLM-based small-corpus clustering method.

The vision LLM is never really called: ``litellm.completion`` (the
innermost dispatch inside :mod:`dgml_core.llm`) is patched to return
hand-built OpenAI-shaped stubs, exactly as ``test_classification.py`` does.
This file covers the standalone :func:`llm_cluster_files` partitioner plus
its integration through :func:`dgml_core.clustering.clustering_internal` /
:func:`dgml_core.clustering.clustering`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from dgml_core import layout
from dgml_core.classification import ClassificationConfig
from dgml_core.clustering import (
    _resolve_method,
    clustering,
    clustering_internal,
)
from dgml_core.docsets import DocSetStore
from dgml_core.errors import (
    AuthError,
    ClassificationConfigInvalid,
    ClassificationConfigMissing,
    ClassificationFailed,
    ClusteringConfigInvalid,
)
from dgml_core.llm_clustering import (
    DEFAULT_MAX_FILES,
    LLMClusteringResult,
    _group_confidence,
    llm_cluster_files,
)
from dgml_core.models import DocSet, FileRecord
from dgml_core.storage import Workspace

from .conftest import make_fake_png, write_classification_config

DEFAULT_TEST_MODEL = "gemini/gemini-2.5-flash-lite"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_file(workspace: Workspace, file_id: str) -> None:
    """Materialize a minimal File record on disk (no real PDF ingest)."""
    record = FileRecord(
        id=file_id,
        original_path=f"/fake/{file_id}.pdf",
        original_filename=f"{file_id}.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=1,
        text_mode="digital",
    )
    workspace.docs.put_doc("files", file_id, record.to_json())


def _seed_page_image(workspace: Workspace, file_id: str) -> None:
    """Write a valid single-page PNG so ``gather_file_pages`` finds one page."""
    workspace.blobs.put_blob(layout.file_page_image_key(file_id, 1), make_fake_png(8, 8))


def _seed(workspace: Workspace, file_id: str, *, with_page: bool = True) -> None:
    _seed_file(workspace, file_id)
    if with_page:
        _seed_page_image(workspace, file_id)


def _config(model: str = DEFAULT_TEST_MODEL, **kw: Any) -> ClassificationConfig:
    return ClassificationConfig(model=model, **kw)


def _group_response(groups: list[dict[str, Any]]) -> SimpleNamespace:
    """A litellm.completion stub carrying one ``group_documents`` tool call."""
    call = SimpleNamespace(
        function=SimpleNamespace(name="group_documents", arguments=json.dumps({"groups": groups}))
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=[call], content=None),
                finish_reason="tool_calls",
            )
        ]
    )


def _raw_response(tool_name: str | None, arguments: Any) -> SimpleNamespace:
    """A stub with an arbitrary tool name / raw (already-serialized) arguments."""
    call = SimpleNamespace(function=SimpleNamespace(name=tool_name, arguments=arguments))
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=[call], content=None),
                finish_reason="tool_calls",
            )
        ]
    )


def _empty_tool_calls_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[], content=None))]
    )


def _new_group(name: str, members: list[str], **extra: Any) -> dict[str, Any]:
    group: dict[str, Any] = {
        "name": name,
        "description": f"{name} documents",
        "key_questions": ["q1?", "q2?", "q3?"],
        "members": members,
    }
    group.update(extra)
    return group


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path / "ws")
    ws.init()
    return ws


# ---------------------------------------------------------------------------
# llm_cluster_files — happy paths
# ---------------------------------------------------------------------------


def test_fresh_partition_into_new_clusters(workspace: Workspace) -> None:
    for fid in ("a", "b", "c"):
        _seed(workspace, fid)

    groups = [
        _new_group("Invoice", ["doc_1", "doc_2"]),
        _new_group("Contract", ["doc_3"]),
    ]
    with patch("litellm.completion", return_value=_group_response(groups)) as completion:
        result = llm_cluster_files(workspace, ["a", "b", "c"], config=_config())

    assert isinstance(result, LLMClusteringResult)
    # Two emergent clusters, named unknown_0 / unknown_1 in group order.
    assert result.clusters == {"a": "unknown_0", "b": "unknown_0", "c": "unknown_1"}
    assert set(result.proposals) == {"unknown_0", "unknown_1"}
    assert result.proposals["unknown_0"].new_name == "Invoice"
    assert result.proposals["unknown_0"].new_key_questions == ("q1?", "q2?", "q3?")
    assert result.proposals["unknown_1"].new_name == "Contract"
    assert result.failed_file_ids == []

    # A single forced tool call to the configured model.
    assert completion.call_count == 1
    kwargs = completion.call_args.kwargs
    assert kwargs["model"] == DEFAULT_TEST_MODEL
    assert kwargs["tool_choice"] == "required"
    assert kwargs["temperature"] == 0.0  # greedy decoding for reproducible partitions
    assert [t["function"]["name"] for t in kwargs["tools"]] == ["group_documents"]


def test_content_carries_doc_markers_and_images(workspace: Workspace) -> None:
    for fid in ("a", "b"):
        _seed(workspace, fid)

    with patch(
        "litellm.completion", return_value=_group_response([_new_group("T", ["doc_1", "doc_2"])])
    ) as completion:
        llm_cluster_files(workspace, ["a", "b"], config=_config())

    content = completion.call_args.kwargs["messages"][0]["content"]
    texts = [b["text"] for b in content if b.get("type") == "text"]
    images = [b for b in content if b.get("type") == "image_url"]
    assert "=== Document doc_1 ===" in texts
    assert "=== Document doc_2 ===" in texts
    assert len(images) == 2  # one page image per file
    assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_assign_to_existing_docset_uses_docset_name(workspace: Workspace) -> None:
    for fid in ("a", "b"):
        _seed(workspace, fid)
    docsets = [
        DocSet(id="ds1", name="Invoice", description="", key_questions=["total?"]),
        DocSet(id="ds2", name="Contract", description="", key_questions=["parties?"]),
    ]

    groups = [
        {"existing_docset_id": "ds1", "members": ["doc_1"]},
        _new_group("Receipt", ["doc_2"]),
    ]
    with patch("litellm.completion", return_value=_group_response(groups)) as completion:
        result = llm_cluster_files(workspace, ["a", "b"], config=_config(), docsets=docsets)

    # Existing-docset group keyed by the docset *name*; new group is unknown_0.
    assert result.clusters == {"a": "Invoice", "b": "unknown_0"}
    assert set(result.proposals) == {"unknown_0"}
    assert "unknown_0" not in {"Invoice", "Contract"}

    # The tool exposes the existing docset ids as an enum only when docsets exist.
    tool = completion.call_args.kwargs["tools"][0]
    enum = tool["function"]["parameters"]["properties"]["groups"]["items"]["properties"][
        "existing_docset_id"
    ]["enum"]
    assert enum == ["ds1", "ds2"]


def test_no_existing_docset_id_property_when_no_docsets(workspace: Workspace) -> None:
    _seed(workspace, "a")
    with patch(
        "litellm.completion", return_value=_group_response([_new_group("T", ["doc_1"])])
    ) as completion:
        llm_cluster_files(workspace, ["a"], config=_config())

    props = completion.call_args.kwargs["tools"][0]["function"]["parameters"]["properties"][
        "groups"
    ]["items"]["properties"]
    assert "existing_docset_id" not in props


# ---------------------------------------------------------------------------
# llm_cluster_files — partial / failure handling
# ---------------------------------------------------------------------------


def test_file_without_page_image_is_failed_not_sent(workspace: Workspace) -> None:
    _seed(workspace, "a")
    _seed(workspace, "b", with_page=False)  # no rendered page

    with patch(
        "litellm.completion", return_value=_group_response([_new_group("T", ["doc_1"])])
    ) as completion:
        result = llm_cluster_files(workspace, ["a", "b"], config=_config())

    assert result.clusters == {"a": "unknown_0"}
    assert result.failed_file_ids == ["b"]
    # Only "a" (doc_1) was sent — a single image, one doc marker.
    content = completion.call_args.kwargs["messages"][0]["content"]
    assert sum(1 for b in content if b.get("type") == "image_url") == 1


def test_member_labels_tolerant_matching(workspace: Workspace) -> None:
    """The model may echo labels loosely — a bare int, 'doc2', or canonical
    'doc_3'. All should still resolve to the right file."""
    for fid in ("a", "b", "c"):
        _seed(workspace, fid)
    groups = [
        {
            "name": "T",
            "description": "d",
            "key_questions": ["q1?", "q2?", "q3?"],
            "members": [1, "doc2", "doc_3"],  # int, no-underscore, canonical
        }
    ]
    with patch("litellm.completion", return_value=_group_response(groups)):
        result = llm_cluster_files(workspace, ["a", "b", "c"], config=_config())
    assert result.clusters == {"a": "unknown_0", "b": "unknown_0", "c": "unknown_0"}
    assert result.failed_file_ids == []


def test_no_members_resolve_raises(workspace: Workspace) -> None:
    """Groups whose members map to nothing must surface a loud error, not a
    silent all-failed result."""
    for fid in ("a", "b"):
        _seed(workspace, fid)
    groups = [_new_group("T", ["nonsense_x", "other_y"])]  # no digits → no match
    with patch("litellm.completion", return_value=_group_response(groups)):
        with pytest.raises(ClassificationFailed, match="none could be placed"):
            llm_cluster_files(workspace, ["a", "b"], config=_config())


def test_all_files_without_pages_raises(workspace: Workspace) -> None:
    _seed(workspace, "a", with_page=False)
    _seed(workspace, "b", with_page=False)
    with patch("litellm.completion") as completion:
        with pytest.raises(ClassificationFailed, match="no page images"):
            llm_cluster_files(workspace, ["a", "b"], config=_config())
    completion.assert_not_called()


def test_overflow_beyond_max_files_is_failed(workspace: Workspace) -> None:
    for fid in ("a", "b", "c"):
        _seed(workspace, fid)
    with patch(
        "litellm.completion",
        return_value=_group_response([_new_group("T", ["doc_1", "doc_2"])]),
    ) as completion:
        result = llm_cluster_files(workspace, ["a", "b", "c"], config=_config(), max_files=2)

    assert result.clusters == {"a": "unknown_0", "b": "unknown_0"}
    assert result.failed_file_ids == ["c"]
    # "c" was never sent.
    content = completion.call_args.kwargs["messages"][0]["content"]
    assert sum(1 for b in content if b.get("type") == "image_url") == 2


def test_max_files_must_be_positive(workspace: Workspace) -> None:
    _seed(workspace, "a")
    with pytest.raises(ValueError, match="max_files"):
        llm_cluster_files(workspace, ["a"], config=_config(), max_files=0)


def test_unplaced_document_becomes_failed(workspace: Workspace) -> None:
    for fid in ("a", "b"):
        _seed(workspace, fid)
    # Model groups only doc_1; doc_2 is omitted entirely.
    with patch("litellm.completion", return_value=_group_response([_new_group("T", ["doc_1"])])):
        result = llm_cluster_files(workspace, ["a", "b"], config=_config())
    assert result.clusters == {"a": "unknown_0"}
    assert result.failed_file_ids == ["b"]


def test_duplicate_placement_first_group_wins(workspace: Workspace) -> None:
    _seed(workspace, "a")
    groups = [_new_group("First", ["doc_1"]), _new_group("Second", ["doc_1"])]
    with patch("litellm.completion", return_value=_group_response(groups)):
        result = llm_cluster_files(workspace, ["a"], config=_config())
    assert result.clusters == {"a": "unknown_0"}
    # The second (empty-after-dedup) group produced no proposal.
    assert set(result.proposals) == {"unknown_0"}
    assert result.proposals["unknown_0"].new_name == "First"


def test_nameless_new_group_still_clusters(workspace: Workspace) -> None:
    """A group with valid members but no name is still a cluster — membership
    is what matters. It becomes an unnamed ``unknown_N`` bucket (no proposal);
    naming falls back to clustering()'s pass-2."""
    for fid in ("a", "b"):
        _seed(workspace, fid)
    groups = [
        {"members": ["doc_1"]},  # no name — kept as an unnamed cluster
        _new_group("Real", ["doc_2"]),
    ]
    with patch("litellm.completion", return_value=_group_response(groups)):
        result = llm_cluster_files(workspace, ["a", "b"], config=_config())
    assert result.clusters == {"a": "unknown_0", "b": "unknown_1"}
    assert result.failed_file_ids == []
    # Only the named group carries a proposal; the nameless one does not.
    assert set(result.proposals) == {"unknown_1"}
    assert result.proposals["unknown_1"].new_name == "Real"


def test_fresh_mode_requires_naming_fields(workspace: Workspace) -> None:
    """With no existing DocSets every group must be new, so the tool schema
    forces the model to name each group in the same call."""
    _seed(workspace, "a")
    with patch(
        "litellm.completion", return_value=_group_response([_new_group("T", ["doc_1"])])
    ) as completion:
        llm_cluster_files(workspace, ["a"], config=_config())
    item = completion.call_args.kwargs["tools"][0]["function"]["parameters"]["properties"][
        "groups"
    ]["items"]
    assert set(item["required"]) == {"members", "name", "description", "key_questions"}


def test_unknown_docset_id_treated_as_new_group(workspace: Workspace) -> None:
    _seed(workspace, "a")
    docsets = [DocSet(id="ds1", name="Invoice")]
    # Model references a docset id that isn't offered → falls back to new-group
    # parsing, which needs a name.
    groups = [_new_group("Invoice", ["doc_1"], existing_docset_id="bogus")]
    with patch("litellm.completion", return_value=_group_response(groups)):
        result = llm_cluster_files(workspace, ["a"], config=_config(), docsets=docsets)
    assert result.clusters == {"a": "unknown_0"}


def test_missing_key_questions_degrades_gracefully(workspace: Workspace) -> None:
    _seed(workspace, "a")
    groups = [{"name": "Invoice", "members": ["doc_1"]}]  # no description / key_questions
    with patch("litellm.completion", return_value=_group_response(groups)):
        result = llm_cluster_files(workspace, ["a"], config=_config())
    proposal = result.proposals["unknown_0"]
    assert proposal.new_name == "Invoice"
    assert proposal.new_description == ""
    assert proposal.new_key_questions == ()


# ---------------------------------------------------------------------------
# llm_cluster_files — malformed responses
# ---------------------------------------------------------------------------


def test_provider_exception_wrapped(workspace: Workspace) -> None:
    _seed(workspace, "a")
    with patch("litellm.completion", side_effect=RuntimeError("boom")):
        with pytest.raises(ClassificationFailed, match="LLM call failed: RuntimeError"):
            llm_cluster_files(workspace, ["a"], config=_config())


def test_empty_tool_calls_raises(workspace: Workspace) -> None:
    _seed(workspace, "a")
    with patch("litellm.completion", return_value=_empty_tool_calls_response()):
        with pytest.raises(ClassificationFailed, match="no tool calls"):
            llm_cluster_files(workspace, ["a"], config=_config())


def test_wrong_tool_name_raises(workspace: Workspace) -> None:
    _seed(workspace, "a")
    with patch("litellm.completion", return_value=_raw_response("something_else", "{}")):
        with pytest.raises(ClassificationFailed, match="unexpected tool name"):
            llm_cluster_files(workspace, ["a"], config=_config())


def test_arguments_not_valid_json_raises(workspace: Workspace) -> None:
    _seed(workspace, "a")
    with patch("litellm.completion", return_value=_raw_response("group_documents", "{not json")):
        with pytest.raises(ClassificationFailed, match="not valid JSON"):
            llm_cluster_files(workspace, ["a"], config=_config())


def test_missing_groups_array_raises(workspace: Workspace) -> None:
    _seed(workspace, "a")
    with patch("litellm.completion", return_value=_raw_response("group_documents", json.dumps({}))):
        with pytest.raises(ClassificationFailed, match="missing a 'groups' array"):
            llm_cluster_files(workspace, ["a"], config=_config())


# ---------------------------------------------------------------------------
# Self-reported confidence
# ---------------------------------------------------------------------------


def test_group_confidence_propagates_to_every_member(workspace: Workspace) -> None:
    for fid in ("a", "b", "c"):
        _seed(workspace, fid)
    groups = [
        _new_group("Invoice", ["doc_1", "doc_2"], confidence=0.9),
        _new_group("Contract", ["doc_3"], confidence=0.4),
    ]
    with patch("litellm.completion", return_value=_group_response(groups)):
        result = llm_cluster_files(workspace, ["a", "b", "c"], config=_config())

    # The model judges the group, so every member inherits the group's number.
    assert result.confidences == {"a": 0.9, "b": 0.9, "c": 0.4}


def test_group_confidence_absent_is_none(workspace: Workspace) -> None:
    for fid in ("a", "b"):
        _seed(workspace, fid)
    # Neither group carries `confidence` — the field is optional and lite
    # models routinely omit it. Reporting None beats inventing a default.
    groups = [_new_group("Invoice", ["doc_1"]), _new_group("Contract", ["doc_2"])]
    with patch("litellm.completion", return_value=_group_response(groups)):
        result = llm_cluster_files(workspace, ["a", "b"], config=_config())

    assert result.confidences == {"a": None, "b": None}


def test_unplaceable_members_get_no_confidence_entry(workspace: Workspace) -> None:
    """The confidence map covers exactly the files that were actually placed."""
    for fid in ("a", "b"):
        _seed(workspace, fid)
    # doc_2 is placed; doc_99 does not resolve to any seeded file.
    groups = [_new_group("Invoice", ["doc_2", "doc_99"], confidence=0.7)]
    with patch("litellm.completion", return_value=_group_response(groups)):
        result = llm_cluster_files(workspace, ["a", "b"], config=_config())

    assert set(result.confidences) == set(result.clusters)
    assert result.failed_file_ids == ["a"]


def test_confidence_is_an_optional_tool_property(workspace: Workspace) -> None:
    _seed(workspace, "a")
    with patch(
        "litellm.completion", return_value=_group_response([_new_group("T", ["doc_1"])])
    ) as completion:
        llm_cluster_files(workspace, ["a"], config=_config())

    group_schema = completion.call_args.kwargs["tools"][0]["function"]["parameters"]["properties"][
        "groups"
    ]["items"]
    conf = group_schema["properties"]["confidence"]
    assert conf["type"] == "number"
    assert (conf["minimum"], conf["maximum"]) == (0.0, 1.0)
    # Never required: forcing it would make a model that cannot produce the
    # field fail the whole partition call, which is a bad trade for a signal.
    assert "confidence" not in group_schema["required"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.73, 0.73),
        (1, 1.0),  # a bare int is what models usually answer
        (0, 0.0),
        (1.5, 1.0),  # out of range clamps rather than discards
        (-0.2, 0.0),
        (None, None),
        ("0.9", None),  # a numeric string is not silently coerced
        (True, None),  # bool is an int subclass; must not read as 1.0
        (float("nan"), None),  # would poison every downstream comparison
    ],
)
def test_group_confidence_parsing(raw: Any, expected: float | None) -> None:
    assert _group_confidence({"members": ["doc_1"], "confidence": raw}) == expected


def test_group_confidence_missing_key() -> None:
    assert _group_confidence({"members": ["doc_1"]}) is None


def test_clustering_llm_assignment_carries_confidence(workspace: Workspace) -> None:
    """End to end: the self-report reaches each assignment in the CLI payload.

    This is the column that was previously always null for method="llm".
    """
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    groups = [_new_group("Invoice", ["doc_1", "doc_2"], confidence=0.85)]
    with patch("litellm.completion", return_value=_group_response(groups)):
        out = clustering(workspace, method="llm")

    assert out["assignments"]["a"]["confidence"] == 0.85
    assert out["assignments"]["b"]["confidence"] == 0.85


# ---------------------------------------------------------------------------
# _resolve_method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "n", "threshold", "mode", "expected"),
    [
        ("auto", 3, 8, "fresh", "llm"),
        ("auto", 8, 8, "fresh", "llm"),  # boundary: <= threshold
        ("auto", 9, 8, "fresh", "embedding"),
        # An incremental batch is scored against prototypes that already exist,
        # so its size says nothing about whether the statistics hold — it stays
        # on the embedding path however small it is.
        ("auto", 1, 8, "incremental", "embedding"),
        ("auto", 8, 8, "incremental", "embedding"),
        ("llm", 100, 8, "fresh", "llm"),  # explicit wins regardless of size
        ("llm", 2, 8, "incremental", "llm"),  # ... and regardless of mode
        ("embedding", 1, 8, "fresh", "embedding"),
    ],
)
def test_resolve_method(method: str, n: int, threshold: int, mode: str, expected: str) -> None:
    assert _resolve_method(method, n_usable=n, threshold=threshold, mode=mode) == expected


# ---------------------------------------------------------------------------
# clustering_internal / clustering integration (method="llm")
# ---------------------------------------------------------------------------


def test_clustering_internal_llm_returns_proposals(workspace: Workspace) -> None:
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    groups = [_new_group("Invoice", ["doc_1", "doc_2"])]
    with patch("litellm.completion", return_value=_group_response(groups)):
        internal = clustering_internal(workspace, method="llm")

    assert internal.method == "llm"
    assert internal.mode == "fresh"
    assert internal.clusters == {"a": "unknown_0", "b": "unknown_0"}
    assert internal.proposals["unknown_0"].new_name == "Invoice"


def test_clustering_llm_creates_docsets_without_second_call(workspace: Workspace) -> None:
    """The llm method partitions *and* names in one call, so pass-2 must not
    issue a second (naming) LLM request."""
    for fid in ("a", "b", "c"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    groups = [
        _new_group("Invoice", ["doc_1", "doc_2"]),
        _new_group("Contract", ["doc_3"]),
    ]
    with patch("litellm.completion", return_value=_group_response(groups)) as completion:
        out = clustering(workspace, method="llm")

    assert completion.call_count == 1  # single grouping call, no naming pass
    assert out["mode"] == "fresh"
    assert out["n_new_clusters"] == 2
    assert out["failed_file_ids"] == []

    store = DocSetStore(workspace)
    by_name = {d.name: d for d in store.list_all()}
    assert set(by_name) == {"Invoice", "Contract"}
    assert sorted(store.list_files(by_name["Invoice"].id)) == ["a", "b"]
    assert store.list_files(by_name["Contract"].id) == ["c"]
    # The returned cluster map reflects the real DocSet names, not placeholders.
    assert out["clusters"] == {"a": "Invoice", "b": "Invoice", "c": "Contract"}


def test_clustering_auto_small_corpus_uses_llm(workspace: Workspace) -> None:
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    groups = [_new_group("Invoice", ["doc_1", "doc_2"])]
    with patch("litellm.completion", return_value=_group_response(groups)) as completion:
        out = clustering(workspace, method="auto", small_corpus_threshold=8)

    assert completion.call_count == 1
    assert out["n_new_clusters"] == 1


def test_clustering_llm_incremental_assigns_to_existing(workspace: Workspace) -> None:
    store = DocSetStore(workspace)
    existing = store.create("Invoice", key_questions=["total?"])
    # A member gives the DocSet a page-backed prototype (not strictly needed
    # for the llm path, but mirrors a realistic workspace).
    _seed(workspace, "seed")
    store.add_file(existing.id, "seed")

    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    groups = [
        {"existing_docset_id": existing.id, "members": ["doc_1"]},
        _new_group("Contract", ["doc_2"]),
    ]
    with patch("litellm.completion", return_value=_group_response(groups)):
        out = clustering(workspace, method="llm", mode="incremental")

    assert out["mode"] == "incremental"
    assert out["clusters"]["a"] == "Invoice"
    assert out["clusters"]["b"] == "Contract"
    assert sorted(store.list_files(existing.id)) == ["a", "seed"]


def test_clustering_llm_missing_classification_config_raises(workspace: Workspace) -> None:
    _seed(workspace, "a")
    # No classification config written → the llm method can't run.
    with patch("litellm.completion"):
        with pytest.raises(ClassificationConfigMissing):
            clustering(workspace, method="llm")


def test_clustering_llm_runtime_failure_soft_fails(workspace: Workspace) -> None:
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    with patch("litellm.completion", side_effect=RuntimeError("boom")):
        out = clustering(workspace, method="llm")

    # Provider blew up → every file soft-fails, no DocSet created, no raise.
    assert out["n_new_clusters"] == 0
    assert sorted(out["failed_file_ids"]) == ["a", "b"]
    assert DocSetStore(workspace).list_all() == []


def test_unknown_method_rejected(workspace: Workspace) -> None:
    from dgml_core.errors import ClusteringConfigInvalid

    with pytest.raises(ClusteringConfigInvalid, match="unknown clustering method"):
        clustering(workspace, method="telepathy")


def test_default_max_files_is_reasonable() -> None:
    assert DEFAULT_MAX_FILES >= 8


# ---------------------------------------------------------------------------
# The default method, and what `auto` does when the LLM partitioner can't run
# ---------------------------------------------------------------------------


def test_default_method_routes_a_small_corpus_to_the_llm(workspace: Workspace) -> None:
    """`dgml cluster` with no --method must not hand a tiny corpus to embeddings.

    Reported from the field: two files, and the run died inside sklearn with
    "max_df corresponds to < documents than min_df". The routing that avoids
    that already existed — it was just never the default, so it never fired.
    """
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    groups = [_new_group("Invoice", ["doc_1", "doc_2"])]
    with patch("litellm.completion", return_value=_group_response(groups)) as completion:
        out = clustering(workspace)

    assert completion.call_count == 1
    assert out["method"] == "llm"
    assert out["n_new_clusters"] == 1


def test_default_method_leaves_a_large_corpus_on_the_embedding_path(
    workspace: Workspace,
) -> None:
    """The other side of the same default: above the threshold nothing moves.

    This is what keeps the change off every existing large-corpus run.
    """
    for i in range(9):
        _seed(workspace, f"f{i}")
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    with (
        patch("litellm.completion") as completion,
        patch(
            "dgml_core.clustering.run_clustering_detailed",
            return_value={f"f{i}": _prediction("unknown_0") for i in range(9)},
        ) as run_embedding,
    ):
        internal = clustering_internal(workspace)

    assert internal.method == "embedding"
    assert run_embedding.call_count == 1
    assert completion.call_count == 0


def test_auto_falls_back_to_embedding_without_classification_config(
    workspace: Workspace,
) -> None:
    """A workspace with no `classification` section must still cluster.

    Under `auto` the *routing* chose the LLM, not the caller, so a partitioner
    that cannot run is the routing's problem to solve — the statistical path is
    a weaker answer on a corpus this small, but it is an answer. Contrast
    ``test_clustering_llm_missing_classification_config_raises``: someone who
    typed --method llm named the engine and gets the error.
    """
    for fid in ("a", "b"):
        _seed(workspace, fid)
    # No classification config written.

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"a": _prediction("unknown_0"), "b": _prediction("unknown_0")},
    ) as run_embedding:
        # Silent on purpose: a workspace that never configured a model is not
        # LLM-enabled, and saying so on every small run would be noise.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            internal = clustering_internal(workspace, method="auto")

    assert internal.method == "embedding"
    assert run_embedding.call_count == 1


def test_auto_retries_a_failed_partitioner_call_on_the_embedding_engine(
    workspace: Workspace,
) -> None:
    """A partitioner call that fails once started must not lose the run.

    Some of these failures belong to the partitioner and not to the model. It
    sends every file's pages in one request under its own grouping tool, so it
    can exceed a provider's per-request image cap, or get back members named by
    filename instead of the ``doc_N`` labels the tool defines. The naming pass
    the embedding path uses has neither problem: one small request per cluster,
    a different tool. Measured on a workspace where the model answers with
    filenames, ``--method embedding`` names the corpus and creates a DocSet
    while the partitioner produces nothing, so under ``auto`` — where the
    caller did not choose the partitioner — the run continues on the other
    engine. Pinned ``llm`` still soft-fails; see
    ``test_clustering_llm_runtime_failure_soft_fails``.
    """
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    # Members given as filenames, which `_assemble_result` rejects. This is the
    # partitioner's own tool contract failing, not the model being unreachable.
    groups = [_new_group("Invoice", ["a.pdf", "b.pdf"])]
    with (
        patch("litellm.completion", return_value=_group_response(groups)),
        patch(
            "dgml_core.clustering.run_clustering_detailed",
            return_value={"a": _prediction("unknown_0"), "b": _prediction("unknown_0")},
        ) as run_embedding,
    ):
        internal = clustering_internal(workspace, method="auto")

    assert internal.method == "embedding"
    assert run_embedding.call_count == 1


def test_pinned_llm_never_falls_back(workspace: Workspace) -> None:
    """The pin has to be a pin, or --method llm cannot be used to test the LLM."""
    for fid in ("a", "b"):
        _seed(workspace, fid)

    with pytest.raises(ClassificationConfigMissing):
        clustering(workspace, method="llm")


def test_result_reports_the_method_that_ran(workspace: Workspace) -> None:
    """`method` is the only way a caller can tell which engine produced a result.

    Under the default the caller does not choose it, and an `auto` run that fell
    back off the LLM is otherwise indistinguishable from one that never routed
    there.
    """
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    groups = [_new_group("Invoice", ["doc_1", "doc_2"])]
    with patch("litellm.completion", return_value=_group_response(groups)):
        assert clustering(workspace, method="llm")["method"] == "llm"

    # A resolved embedding run reports embedding — through the public payload,
    # and on a run that actually reached the engine rather than short-circuiting.
    for fid in ("c", "d"):
        _seed(workspace, fid)
    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"c": _prediction("unknown_0"), "d": _prediction("unknown_0")},
    ):
        assert clustering(workspace, method="embedding")["method"] == "embedding"


def test_no_engine_ran_reports_a_null_method(workspace: Workspace) -> None:
    """An empty workspace resolves no engine, so the field must not claim one.

    Echoing the request would put ``"auto"`` — the default — into a field
    documented as the engine that grouped the files.
    """
    assert clustering(workspace, skip_existing=True)["method"] is None
    assert clustering(workspace)["method"] is None


def _prediction(cluster_name: str) -> Any:
    """A stub ``run_clustering_detailed`` outcome, so the fallback tests can
    assert *which engine ran* without also depending on a fittable corpus."""
    from dgml_core.run_clustering import DocPrediction

    return DocPrediction(cluster_name=cluster_name, confidence=None)


def test_auto_routes_around_a_malformed_classification_section_but_says_so(
    workspace: Workspace,
) -> None:
    """ "Configured wrong" must neither break the run nor pass unmentioned.

    ``clustering()`` documents that it does not raise and the CLI that it exits
    0, and the caller who typed nothing did not ask for the LLM, so a bad
    section cannot be allowed to fail the run. But quietly clustering by
    another engine would hide the mistake and leave the config inert forever,
    so it warns. Pinned ``llm`` still raises — see the test below.
    """
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL, "max_pages": 0})

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"a": _prediction("unknown_0"), "b": _prediction("unknown_0")},
    ) as run_embedding:
        with pytest.warns(RuntimeWarning, match="cannot be used"):
            internal = clustering_internal(workspace, method="auto")

    assert internal.method == "embedding"
    assert run_embedding.call_count == 1


def test_pinned_llm_still_raises_on_a_malformed_section(workspace: Workspace) -> None:
    """The pin is what turns the fallback back into an error."""
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL, "max_pages": 0})

    with pytest.raises(ClassificationConfigInvalid):
        clustering_internal(workspace, method="llm")


def test_pinned_llm_still_raises_on_an_absent_credential(workspace: Workspace) -> None:
    """Ditto for a credential the environment does not hold.

    Under ``auto`` this is routed around; naming the engine has to keep meaning
    "tell me when it cannot run", or --method llm is untestable.
    """
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(
        workspace, {"model": DEFAULT_TEST_MODEL, "api_key_env": "DGML_TEST_ABSENT_KEY"}
    )

    with pytest.raises(AuthError):
        clustering_internal(workspace, method="llm")


def test_incremental_batch_stays_on_the_embedding_path(workspace: Workspace) -> None:
    """A small *incremental* batch must not be routed away from the prototypes.

    This is the regression that made the `auto` default unsafe on its first
    cut: the routing counted the unassigned batch, so a mature workspace plus
    two freshly added files — the ordinary `file add` -> `cluster` loop — took
    the one-shot partitioner and skipped the nearest-prototype assignment, the
    novelty gate and the review queue entirely. Verified by hand on a 40-file
    workspace before the fix: `method` came back `llm` and
    `n_assigned_existing` was 0.
    """
    store = DocSetStore(workspace)
    existing = store.create("Invoice", key_questions=["total?"])
    for fid in ("m1", "m2", "m3"):
        _seed(workspace, fid)
        store.add_file(existing.id, fid)
    # Two new arrivals — under SMALL_CORPUS_MAX_FILES, so the batch-size rule
    # would have sent them to the LLM.
    for fid in ("new1", "new2"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    with (
        patch("litellm.completion") as completion,
        patch(
            "dgml_core.clustering.run_clustering_detailed",
            return_value={"new1": _prediction("Invoice"), "new2": _prediction("Invoice")},
        ) as run_embedding,
    ):
        internal = clustering_internal(workspace)

    assert internal.mode == "incremental"
    assert internal.method == "embedding"
    assert run_embedding.call_count == 1
    assert completion.call_count == 0


def test_auto_validates_the_config_before_routing(workspace: Workspace) -> None:
    """`--config` is ignored on the LLM path, but it must still be *validated*.

    Otherwise a mistyped preset name or config path is an error under
    `--method embedding` and silently accepted under the default, on exactly
    the small corpora where the default routes away from the config.
    """
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(workspace, {"model": DEFAULT_TEST_MODEL})

    with patch("litellm.completion") as completion:
        with pytest.raises(ClusteringConfigInvalid):
            clustering_internal(workspace, method="auto", config="no-such-preset")
    assert completion.call_count == 0


def test_auto_falls_back_when_the_credential_is_missing(workspace: Workspace) -> None:
    """A configured model with no key behind it must not break the no-raise contract.

    `classification.api_key_env` naming an unset variable raises `AuthError`
    from inside the partitioner's setup, before any call. Under `auto` that is
    the routing's problem, and `clustering()` documents that it does not raise.
    """
    for fid in ("a", "b"):
        _seed(workspace, fid)
    write_classification_config(
        workspace, {"model": DEFAULT_TEST_MODEL, "api_key_env": "DGML_TEST_ABSENT_KEY"}
    )

    with patch(
        "dgml_core.clustering.run_clustering_detailed",
        return_value={"a": _prediction("unknown_0"), "b": _prediction("unknown_0")},
    ) as run_embedding:
        with pytest.warns(RuntimeWarning, match="cannot be used"):
            internal = clustering_internal(workspace, method="auto")

    assert internal.method == "embedding"
    assert run_embedding.call_count == 1
