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

"""Tests for the litellm-backed :class:`LLMAdjudicator` (reassign + repartition).

``litellm.completion`` is patched with hand-built OpenAI-shaped stubs, exactly
as ``test_classification`` / ``test_llm_clustering`` do — so nothing here needs
a provider or an API key.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from clustering.config.schema import ConsolidationMode
from clustering.consolidation import AdjudicationRequest
from clustering.data.datasets import DocumentDataset, DocumentRecord
from dgml_core.classification import ClassificationConfig
from dgml_core.consolidation import LLMAdjudicator
from PIL import Image

DEFAULT_TEST_MODEL = "gemini/gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# Response stubs
# ---------------------------------------------------------------------------
def _tool_response(name: str, args: dict[str, Any]) -> SimpleNamespace:
    call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=[call], content=None),
                finish_reason="tool_calls",
            )
        ]
    )


def _adjudicate_response(choice: str, confidence: float | None = 0.8) -> SimpleNamespace:
    args: dict[str, Any] = {"choice": choice, "rationale": "because"}
    if confidence is not None:
        args["confidence"] = confidence
    return _tool_response("adjudicate", args)


def _regroup_response(groups: list[dict[str, Any]]) -> SimpleNamespace:
    return _tool_response("regroup_documents", {"groups": groups})


def _no_tool_call_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(tool_calls=None, content="sorry, I'd rather chat"),
                finish_reason="stop",
            )
        ]
    )


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
class _MemDataset(DocumentDataset):
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> DocumentRecord:
        if index >= self._n:
            raise IndexError(index)
        return DocumentRecord(
            doc_id=f"doc_{index}",
            label=None,
            image=Image.new("RGB", (8, 8), color=(index * 20 % 255, 0, 0)),
            text=f"document {index}",
            thumbnail_path=None,
        )


def _config() -> ClassificationConfig:
    return ClassificationConfig(model=DEFAULT_TEST_MODEL)


def _request(index: int = 0, candidates: list[str] | None = None) -> AdjudicationRequest:
    return AdjudicationRequest(
        doc_id=f"doc_{index}",
        doc_index=index,
        current_label="B",
        candidate_labels=["A", "B"] if candidates is None else candidates,
    )


# ---------------------------------------------------------------------------
# reassign
# ---------------------------------------------------------------------------
def test_adjudicator_reassigns_to_candidate() -> None:
    adj = LLMAdjudicator(_config(), attempts=2)
    with patch("litellm.completion", return_value=_adjudicate_response("A", 0.9)):
        verdicts = adj(_MemDataset(2), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    assert set(verdicts) == {"doc_0"}
    v = verdicts["doc_0"]
    assert v.assignment == "A"
    # Both attempts agree (rotation only reorders candidates) ⇒ agreement 1.0,
    # so confidence is the mean self-report (0.9) x 1.0.
    assert v.confidence is not None and abs(v.confidence - 0.9) < 1e-6
    assert v.rationale == "because"


def test_disagreement_across_attempts_halves_the_confidence() -> None:
    adj = LLMAdjudicator(_config(), attempts=2)
    with patch(
        "litellm.completion",
        side_effect=[_adjudicate_response("A", 1.0), _adjudicate_response("B", 1.0)],
    ):
        verdicts = adj(_MemDataset(1), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    # A 2-way split ⇒ agreement 0.5, so a maximally self-assured model still
    # only earns 0.5. That is the whole point of running it twice.
    v = verdicts["doc_0"]
    assert v.confidence is not None and abs(v.confidence - 0.5) < 1e-6


def test_agreement_falls_back_to_the_confidence_when_none_is_reported() -> None:
    adj = LLMAdjudicator(_config(), attempts=2)
    with patch("litellm.completion", return_value=_adjudicate_response("A", confidence=None)):
        verdicts = adj(_MemDataset(1), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    # No self-report to average, so the agreement rate is the signal.
    assert verdicts["doc_0"].confidence == 1.0


def test_a_failed_attempt_does_not_penalize_the_survivor() -> None:
    adj = LLMAdjudicator(_config(), attempts=2)
    with patch(
        "litellm.completion",
        side_effect=[RuntimeError("transient"), _adjudicate_response("A", 0.8)],
    ):
        verdicts = adj(_MemDataset(1), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    # Agreement is over the attempts that *succeeded*: a provider hiccup
    # degrades to a single-attempt answer rather than halving its confidence.
    v = verdicts["doc_0"]
    assert v.assignment == "A"
    assert v.confidence is not None and abs(v.confidence - 0.8) < 1e-6


def test_adjudicator_novel_verdict() -> None:
    adj = LLMAdjudicator(_config(), attempts=1)
    with patch("litellm.completion", return_value=_adjudicate_response("__novel__", 0.6)):
        verdicts = adj(
            _MemDataset(1),
            [_request(candidates=["A"])],
            mode=ConsolidationMode.REASSIGN,
            batch_size=40,
        )
    assert verdicts["doc_0"].assignment is None  # novel


def test_adjudicator_soft_fails_per_document() -> None:
    adj = LLMAdjudicator(_config(), attempts=1)
    with patch("litellm.completion", side_effect=RuntimeError("down")):
        verdicts = adj(_MemDataset(1), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    assert verdicts == {}  # failed doc simply drops out; no raise


def test_a_reply_with_no_tool_call_yields_no_verdict() -> None:
    adj = LLMAdjudicator(_config(), attempts=1)
    with patch("litellm.completion", return_value=_no_tool_call_response()):
        verdicts = adj(_MemDataset(1), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    assert verdicts == {}


def test_a_missing_document_is_skipped_not_fatal() -> None:
    adj = LLMAdjudicator(_config(), attempts=1)
    # doc_index 5 is past the end of a 1-document dataset.
    with patch("litellm.completion", return_value=_adjudicate_response("A")):
        verdicts = adj(
            _MemDataset(1), [_request(index=5)], mode=ConsolidationMode.REASSIGN, batch_size=40
        )
    assert verdicts == {}


def test_adjudicator_empty_requests_makes_no_call() -> None:
    adj = LLMAdjudicator(_config())
    with patch("litellm.completion") as completion:
        assert adj(_MemDataset(0), [], mode=ConsolidationMode.REASSIGN, batch_size=40) == {}
    completion.assert_not_called()


def test_novel_sentinel_is_always_offered_as_a_choice() -> None:
    adj = LLMAdjudicator(_config(), attempts=1)
    with patch("litellm.completion", return_value=_adjudicate_response("A")) as completion:
        adj(_MemDataset(1), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    tools = completion.call_args.kwargs["tools"]
    choices = tools[0]["function"]["parameters"]["properties"]["choice"]["enum"]
    # Without the sentinel the model is forced to pick a candidate it may not
    # believe in, which is how a genuinely new document type gets buried.
    assert choices[-1] == "__novel__"
    assert set(choices) == {"A", "B", "__novel__"}


def test_candidate_order_rotates_between_attempts() -> None:
    adj = LLMAdjudicator(_config(), attempts=2)
    with patch("litellm.completion", return_value=_adjudicate_response("A")) as completion:
        adj(_MemDataset(1), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    enums = [
        call.kwargs["tools"][0]["function"]["parameters"]["properties"]["choice"]["enum"]
        for call in completion.call_args_list
    ]
    # Rotating means agreement measures robustness, not fixed position bias.
    assert enums[0] == ["A", "B", "__novel__"]
    assert enums[1] == ["B", "A", "__novel__"]


def test_out_of_range_self_confidence_is_clamped() -> None:
    adj = LLMAdjudicator(_config(), attempts=1)
    with patch("litellm.completion", return_value=_adjudicate_response("A", 7.5)):
        verdicts = adj(_MemDataset(1), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    assert verdicts["doc_0"].confidence == 1.0


# ---------------------------------------------------------------------------
# repartition
# ---------------------------------------------------------------------------
def _batch(n: int) -> list[AdjudicationRequest]:
    return [
        AdjudicationRequest(
            doc_id=f"doc_{i}", doc_index=i, current_label="x", candidate_labels=["A"]
        )
        for i in range(n)
    ]


def test_adjudicator_repartition_maps_groups() -> None:
    adj = LLMAdjudicator(_config())
    groups = [
        {"members": ["doc_1", "doc_2"], "existing_label": "A", "confidence": 0.9},
        {"members": ["doc_3"], "name": "NewType", "confidence": 0.5},
    ]
    with patch("litellm.completion", return_value=_regroup_response(groups)):
        verdicts = adj(_MemDataset(3), _batch(3), mode=ConsolidationMode.REPARTITION, batch_size=40)
    # The model sees 1-based ``doc_N`` tags, which map back onto the requests in
    # order — so its "doc_1" is our first request.
    assert verdicts["doc_0"].assignment == "A"
    assert verdicts["doc_1"].assignment == "A"
    # A group that only proposes a *name* is a novel verdict: minting the
    # bucket is the framework's job, since it owns the numbering.
    assert verdicts["doc_2"].assignment is None


def test_repartition_ties_a_multi_member_novel_group_together() -> None:
    adj = LLMAdjudicator(_config())
    groups = [
        # Two documents the model judged to be the *same* new type.
        {"members": ["doc_1", "doc_2"], "name": "Lease Addendum", "confidence": 0.6},
        # A second, distinct new type.
        {"members": ["doc_3"], "name": "Estoppel", "confidence": 0.5},
    ]
    with patch("litellm.completion", return_value=_regroup_response(groups)):
        verdicts = adj(_MemDataset(3), _batch(3), mode=ConsolidationMode.REPARTITION, batch_size=40)
    # All novel (no existing_label).
    assert verdicts["doc_0"].assignment is None
    assert verdicts["doc_1"].assignment is None
    assert verdicts["doc_2"].assignment is None
    # The two members of the first group share a novel-group token so they mint
    # one bucket downstream; the second group's token is distinct. Without this,
    # each novel doc would split into its own unknown_N.
    assert verdicts["doc_0"].novel_group == verdicts["doc_1"].novel_group
    assert verdicts["doc_0"].novel_group is not None
    assert verdicts["doc_2"].novel_group != verdicts["doc_0"].novel_group


def test_repartition_is_a_single_call() -> None:
    adj = LLMAdjudicator(_config())
    groups = [{"members": ["doc_1", "doc_2", "doc_3"], "existing_label": "A"}]
    with patch("litellm.completion", return_value=_regroup_response(groups)) as completion:
        adj(_MemDataset(3), _batch(3), mode=ConsolidationMode.REPARTITION, batch_size=40)
    # The whole point of repartition is one grouping call over the region.
    assert completion.call_count == 1


def test_repartition_respects_the_batch_size() -> None:
    adj = LLMAdjudicator(_config())
    groups = [{"members": ["doc_1", "doc_2"], "existing_label": "A"}]
    with patch("litellm.completion", return_value=_regroup_response(groups)):
        verdicts = adj(_MemDataset(5), _batch(5), mode=ConsolidationMode.REPARTITION, batch_size=2)
    # Only the first two requests were sent, so only those can get verdicts.
    assert set(verdicts) == {"doc_0", "doc_1"}


def test_repartition_ignores_unknown_members_and_malformed_groups() -> None:
    adj = LLMAdjudicator(_config())
    groups: list[Any] = [
        "not a group",
        {"no_members_key": True},
        {"members": "doc_1"},  # not a list
        {"members": ["doc_99"], "existing_label": "A"},  # tag we never issued
        {"members": ["doc_1"], "existing_label": "A"},
    ]
    with patch("litellm.completion", return_value=_regroup_response(groups)):
        verdicts = adj(_MemDataset(2), _batch(2), mode=ConsolidationMode.REPARTITION, batch_size=40)
    assert set(verdicts) == {"doc_0"}


def test_repartition_soft_fails() -> None:
    adj = LLMAdjudicator(_config())
    with patch("litellm.completion", side_effect=RuntimeError("down")):
        assert (
            adj(_MemDataset(2), _batch(2), mode=ConsolidationMode.REPARTITION, batch_size=40) == {}
        )


def test_auto_mode_uses_the_per_document_path() -> None:
    adj = LLMAdjudicator(_config(), attempts=1)
    with patch("litellm.completion", return_value=_adjudicate_response("A")) as completion:
        verdicts = adj(_MemDataset(3), _batch(3), mode=ConsolidationMode.AUTO, batch_size=40)
    # A per-document candidate pick is the safe, bounded default: three
    # documents ⇒ three calls, not one batch grouping.
    assert completion.call_count == 3
    assert set(verdicts) == {"doc_0", "doc_1", "doc_2"}


def test_adjudication_model_override_is_honoured() -> None:
    # A real, litellm-recognized id that differs from the default: the LLM path
    # now checks the model up front (`ModelNotSupported`), and the adjudicator
    # soft-fails on any error, so an unrecognized id would never reach the call
    # and the override would look honoured only because nothing ran.
    adj = LLMAdjudicator(ClassificationConfig(model="gemini/gemini-2.5-pro"), attempts=1)
    with patch("litellm.completion", return_value=_adjudicate_response("A")) as completion:
        adj(_MemDataset(1), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    assert completion.call_args.kwargs["model"] == "gemini/gemini-2.5-pro"


def test_adjudication_forwards_the_custom_endpoint() -> None:
    # A configured `api_base` must reach the LLM call: it routes to the custom
    # endpoint and skips litellm's up-front model-id check, so an id litellm has
    # no metadata for (a self-hosted model) still runs instead of soft-failing.
    config = ClassificationConfig(model="my-proxy/local-model", api_base="http://localhost:11434")
    adj = LLMAdjudicator(config, attempts=1)
    with patch("litellm.completion", return_value=_adjudicate_response("A")) as completion:
        verdicts = adj(_MemDataset(1), [_request()], mode=ConsolidationMode.REASSIGN, batch_size=40)
    assert completion.call_args.kwargs["api_base"] == "http://localhost:11434"
    assert set(verdicts) == {"doc_0"}
