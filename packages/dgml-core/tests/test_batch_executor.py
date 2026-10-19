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

"""``BatchExecutor.run_wave``: every request gets a usable response."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from dgml_core.batch import (
    TIER_MARKER,
    BatchExecutor,
    BatchItemError,
    BatchJob,
    BatchRequest,
    FakeBackend,
    fake_model_response,
)
from dgml_core.batch.executor import DEFAULT_MAX_POLL_S, POLL_MARGIN_S, default_max_poll_s
from dgml_core.batch.types import BatchRejected, BatchState, BatchStatus, BatchSubmitUncertain
from dgml_core.errors import BatchExecutionFailed
from dgml_core.usage import TIER_BATCH, TIER_STANDARD


def _step(tag: str) -> dict[str, Any]:
    return {"model": "fake/model", "messages": [{"role": "user", "content": tag}]}


def _text(response: Any) -> str:
    return str(response.choices[0].message.content)


def _ok_script(ids: list[str]) -> dict[str, Any]:
    return {cid: fake_model_response(f"reply-{cid}") for cid in ids}


class _SyncRecorder:
    """Stand-in for ``llm._completion_with_retry``: records kwargs, answers."""

    def __init__(self, text: str = "sync") -> None:
        self.calls: list[dict[str, Any]] = []
        self.text = text

    def __call__(self, kwargs: dict[str, Any]) -> Any:
        self.calls.append(kwargs)
        return fake_model_response(f"{self.text}:{kwargs['messages'][0]['content']}")


def _executor(backend: FakeBackend, **kw: Any) -> tuple[BatchExecutor, _SyncRecorder]:
    sync = _SyncRecorder()
    kw.setdefault("sleep", lambda _s: None)
    kw.setdefault("min_wave_size", 1)
    return BatchExecutor(backend, sync_execute=sync, **kw), sync


def test_every_id_gets_its_response_regardless_of_result_order() -> None:
    ids = [f"u{i}" for i in range(6)]
    backend = FakeBackend(_ok_script(ids), shuffle=True, seed=7)
    ex, sync = _executor(backend)

    out = ex.run_wave({cid: _step(cid) for cid in ids})

    assert set(out) == set(ids)
    assert all(_text(out[cid]) == f"reply-{cid}" for cid in ids)
    assert all(out[cid]._hidden_params[TIER_MARKER] == TIER_BATCH for cid in ids)
    assert sync.calls == []
    assert len(backend.submitted) == 1
    assert [r.custom_id for r in backend.submitted[0]] == ids
    assert ex.stats.waves == 1 and ex.stats.batches == 1 and ex.stats.requests == 6
    assert ex.stats.batch_ok == 6 and ex.stats.sync_fallbacks == 0
    assert ex.stats.batch_ids == ["fake_batch_0001"]


def test_errored_item_is_resubmitted_once_then_run_synchronously() -> None:
    def script(req: BatchRequest) -> Any:
        if req.custom_id == "bad":
            return BatchItemError("bad", "errored", "provider hiccup")
        return fake_model_response(f"reply-{req.custom_id}")

    backend = FakeBackend(script)
    ex, sync = _executor(backend, max_item_retries=1)

    out = ex.run_wave({"good": _step("good"), "bad": _step("bad")})

    # The retryable error was resubmitted alone once, failed again, then fell back.
    assert [[r.custom_id for r in b] for b in backend.submitted] == [["good", "bad"], ["bad"]]
    assert sync.calls == [_step("bad")]
    assert _text(out["bad"]) == "sync:bad"
    assert out["bad"]._hidden_params[TIER_MARKER] == TIER_STANDARD
    assert _text(out["good"]) == "reply-good"
    assert ex.stats.resubmitted == 1 and ex.stats.sync_fallbacks == 1
    assert ex.stats.batch_ok == 1 and ex.stats.batches == 2


def test_non_retryable_item_falls_back_immediately() -> None:
    def script(req: BatchRequest) -> Any:
        if req.custom_id == "bad":
            return BatchItemError("bad", "invalid", "schema rejected")
        return fake_model_response("ok")

    backend = FakeBackend(script)
    ex, sync = _executor(backend, max_item_retries=3)
    ex.run_wave({"good": _step("good"), "bad": _step("bad")})
    assert len(backend.submitted) == 1
    assert sync.calls == [_step("bad")]
    assert ex.stats.resubmitted == 0 and ex.stats.sync_fallbacks == 1


def test_expired_item_is_resubmitted_and_served_by_the_batch() -> None:
    seen: dict[str, int] = {}

    def script(req: BatchRequest) -> Any:
        seen[req.custom_id] = seen.get(req.custom_id, 0) + 1
        if req.custom_id == "late" and seen["late"] == 1:
            return BatchItemError("late", "expired", "24h window passed")
        return fake_model_response(f"reply-{req.custom_id}")

    backend = FakeBackend(script)
    ex, sync = _executor(backend, max_item_retries=1)
    out = ex.run_wave({"a": _step("a"), "late": _step("late")})

    assert _text(out["late"]) == "reply-late"
    assert out["late"]._hidden_params[TIER_MARKER] == TIER_BATCH
    assert sync.calls == []
    assert seen == {"a": 1, "late": 2}
    assert ex.stats.resubmitted == 1 and ex.stats.sync_fallbacks == 0 and ex.stats.batches == 2


def test_zero_retries_means_any_failure_falls_back_at_once() -> None:
    def script(req: BatchRequest) -> Any:
        return BatchItemError(req.custom_id, "expired", "expired")

    backend = FakeBackend(script)
    ex, sync = _executor(backend, max_item_retries=0)
    out = ex.run_wave({"a": _step("a"), "b": _step("b")})
    assert len(backend.submitted) == 1
    assert [c["messages"][0]["content"] for c in sync.calls] == ["a", "b"]
    assert {cid: _text(r) for cid, r in out.items()} == {"a": "sync:a", "b": "sync:b"}
    assert ex.stats.resubmitted == 0 and ex.stats.sync_fallbacks == 2


def test_response_without_choices_counts_as_an_item_error() -> None:
    def script(req: BatchRequest) -> Any:
        if req.custom_id == "empty":
            return {"choices": []}
        return fake_model_response("ok")

    backend = FakeBackend(script)
    ex, sync = _executor(backend, max_item_retries=0)
    out = ex.run_wave({"ok": _step("ok"), "empty": _step("empty")})
    assert sync.calls == [_step("empty")]
    assert _text(out["empty"]) == "sync:empty"


def test_result_missing_from_the_provider_is_treated_as_errored() -> None:
    class Forgetful(FakeBackend):
        def results(self, job: BatchJob) -> Iterator[tuple[str, Any]]:
            for cid, outcome in super().results(job):
                if cid != "lost":
                    yield cid, outcome

    backend = Forgetful(_ok_script(["kept", "lost"]))
    ex, sync = _executor(backend, max_item_retries=0)
    out = ex.run_wave({"kept": _step("kept"), "lost": _step("lost")})
    assert _text(out["kept"]) == "reply-kept"
    assert _text(out["lost"]) == "sync:lost"
    assert sync.calls == [_step("lost")]


def test_wave_below_min_size_runs_synchronously_without_a_batch() -> None:
    backend = FakeBackend(_ok_script(["only"]))
    ex, sync = _executor(backend, min_wave_size=2)
    out = ex.run_wave({"only": _step("only")})
    assert backend.submitted == []
    assert sync.calls == [_step("only")]
    assert out["only"]._hidden_params[TIER_MARKER] == TIER_STANDARD
    assert ex.stats.waves == 1 and ex.stats.batches == 0 and ex.stats.sync_fallbacks == 1


def test_empty_wave_is_a_no_op() -> None:
    backend = FakeBackend({})
    ex, sync = _executor(backend)
    assert ex.run_wave({}) == {}
    assert backend.submitted == [] and sync.calls == []
    assert ex.stats.waves == 1 and ex.stats.requests == 0


def test_polling_sleeps_between_polls_until_the_batch_ends() -> None:
    backend = FakeBackend(_ok_script(["a", "b"]), polls_until_ended=3)
    sleeps: list[float] = []
    ex = BatchExecutor(
        backend, sync_execute=_SyncRecorder(), sleep=sleeps.append, poll_interval_s=7.5
    )
    ex.run_wave({"a": _step("a"), "b": _step("b")})
    assert sleeps == [7.5, 7.5, 7.5]
    assert backend.polls == 4


def test_poll_deadline_cancels_the_job_and_raises() -> None:
    backend = FakeBackend(_ok_script(["a", "b"]), polls_until_ended=10**6)
    ex, _sync = _executor(backend, max_poll_s=0.0)
    with pytest.raises(BatchExecutionFailed, match="not finished after 0s; canceled"):
        ex.run_wave({"a": _step("a"), "b": _step("b")})
    assert backend.canceled == ["fake_batch_0001"]


def test_submission_failure_raises_batch_execution_failed() -> None:
    backend = FakeBackend({}, fail_submit=RuntimeError("413 payload too large"))
    ex, sync = _executor(backend)
    with pytest.raises(BatchExecutionFailed, match="413 payload too large"):
        ex.run_wave({"a": _step("a"), "b": _step("b")})
    assert sync.calls == []


def test_large_waves_are_split_by_the_backend_limits() -> None:
    ids = [f"u{i}" for i in range(5)]
    backend = FakeBackend(_ok_script(ids), max_requests=2)
    ex, _sync = _executor(backend)
    out = ex.run_wave({cid: _step(cid) for cid in ids})
    assert len(out) == 5
    assert [len(b) for b in backend.submitted] == [2, 2, 1]
    assert ex.stats.batches == 3 and len(ex.stats.batch_ids) == 3


def test_default_sync_executor_is_the_llm_retry_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from dgml_core import llm

    seen: list[dict[str, Any]] = []

    def fake_retry(kwargs: dict[str, Any]) -> Any:
        seen.append(kwargs)
        return fake_model_response("via-llm")

    monkeypatch.setattr(llm, "_completion_with_retry", fake_retry)
    backend = FakeBackend({}, max_requests=100)
    ex = BatchExecutor(backend, min_wave_size=5, sleep=lambda _s: None)
    out = ex.run_wave({"a": _step("a")})
    assert seen == [_step("a")]
    assert _text(out["a"]) == "via-llm"


def test_stats_to_json_round_trip_shape() -> None:
    backend = FakeBackend(_ok_script(["a", "b"]))
    ex, _sync = _executor(backend)
    ex.run_wave({"a": _step("a"), "b": _step("b")})
    assert ex.stats.to_json() == {
        "waves": 1,
        "batches": 1,
        "requests": 2,
        "batch_ok": 2,
        "sync_fallbacks": 0,
        "resubmitted": 0,
        "failed": 0,
        "batch_ids": ["fake_batch_0001"],
        # The fake responses carry no price, so the cost is unknown.
        "cost_usd": None,
        "standard_cost_usd": None,
        "saved_usd": None,
    }


def test_cost_fields_price_batch_results_and_fallbacks_at_their_tiers() -> None:
    def script(req: BatchRequest) -> Any:
        if req.custom_id == "bad":
            return BatchItemError("bad", "invalid", "rejected")
        return fake_model_response("ok", cost=0.25)  # batch price

    def sync(_kwargs: dict[str, Any]) -> Any:
        return fake_model_response("sync", cost=0.5)  # standard price

    ex = BatchExecutor(FakeBackend(script), sync_execute=sync, sleep=lambda _s: None)
    ex.run_wave({"a": _step("a"), "b": _step("b"), "bad": _step("bad")})
    stats = ex.stats.to_json()
    assert stats["cost_usd"] == 0.25 + 0.25 + 0.5
    assert stats["standard_cost_usd"] == 0.5 + 0.5 + 0.5  # batch x2, fallback as-is
    assert stats["saved_usd"] == 0.5


def test_failing_sync_fallback_is_returned_as_that_items_outcome() -> None:
    def script(req: BatchRequest) -> Any:
        if req.custom_id == "bad":
            return BatchItemError("bad", "invalid", "rejected")
        return fake_model_response("ok")

    def sync(_kwargs: dict[str, Any]) -> Any:
        raise ConnectionError("network down")

    backend = FakeBackend(script)
    ex = BatchExecutor(backend, sync_execute=sync, min_wave_size=1, sleep=lambda _s: None)
    out = ex.run_wave({"good": _step("good"), "bad": _step("bad")})
    assert _text(out["good"]) == "ok"
    assert isinstance(out["bad"], ConnectionError)
    assert ex.stats.sync_fallbacks == 1 and ex.stats.failed == 1


# ---- backend failures inside a wave ---------------------------------------


def test_request_that_cannot_be_encoded_runs_synchronously_alone() -> None:
    class PickyEncoder(FakeBackend):
        def encode(self, request: BatchRequest) -> dict[str, Any]:
            if request.custom_id == "odd":
                raise TypeError("unsupported content block")
            return super().encode(request)

    backend = PickyEncoder(_ok_script(["a", "b", "odd"]))
    ex, sync = _executor(backend)
    out = ex.run_wave({cid: _step(cid) for cid in ["a", "b", "odd"]})
    assert [r.custom_id for r in backend.submitted[0]] == ["a", "b"]
    assert sync.calls == [_step("odd")]
    assert _text(out["odd"]) == "sync:odd" and _text(out["a"]) == "reply-a"
    assert ex.stats.sync_fallbacks == 1 and ex.stats.resubmitted == 0


def test_poll_failure_cancels_the_job_and_raises_chained() -> None:
    class BrokenPoll(FakeBackend):
        def poll(self, job: BatchJob) -> Any:
            raise ConnectionResetError("status endpoint down")

    backend = BrokenPoll(_ok_script(["a", "b"]))
    ex, _sync = _executor(backend)
    with pytest.raises(BatchExecutionFailed, match="polling batch fake_batch_0001") as info:
        ex.run_wave({"a": _step("a"), "b": _step("b")})
    assert isinstance(info.value.__cause__, ConnectionResetError)
    assert backend.canceled == ["fake_batch_0001"]


def test_results_failure_cancels_open_jobs_and_raises_chained() -> None:
    class BrokenResults(FakeBackend):
        def results(self, job: BatchJob) -> Iterator[tuple[str, Any]]:
            raise ValueError("malformed results file")

    backend = BrokenResults(_ok_script(["a", "b"]))
    ex, _sync = _executor(backend)
    with pytest.raises(BatchExecutionFailed, match="collecting results") as info:
        ex.run_wave({"a": _step("a"), "b": _step("b")})
    assert isinstance(info.value.__cause__, ValueError)


class _PerJobPolls(FakeBackend):
    """Each submitted batch takes its own number of polls to end."""

    def __init__(self, *args: Any, polls_by_batch: list[int], **kw: Any) -> None:
        super().__init__(*args, **kw)
        self._by_batch = polls_by_batch
        self.polled: dict[str, int] = {}

    def poll(self, job: BatchJob) -> Any:
        index = int(job.job_id.rsplit("_", 1)[1]) - 1
        self._polls_until_ended = self._by_batch[index]
        self.polled[job.job_id] = self.polled.get(job.job_id, 0) + 1
        return super().poll(job)


def test_split_wave_batches_are_polled_together() -> None:
    ids = ["a", "b", "c"]
    backend = _PerJobPolls(_ok_script(ids), max_requests=1, polls_by_batch=[2, 5, 0])
    sleeps: list[float] = []
    ex = BatchExecutor(backend, sync_execute=_SyncRecorder(), min_wave_size=1, sleep=sleeps.append)
    out = ex.run_wave({cid: _step(cid) for cid in ids})
    assert {cid: _text(r) for cid, r in out.items()} == {cid: f"reply-{cid}" for cid in ids}
    # All three submitted before any poll; wall time = the slowest batch.
    assert len(backend.submitted) == 3
    assert len(sleeps) == max(2, 5, 0)
    assert backend.polled == {"fake_batch_0001": 3, "fake_batch_0002": 6, "fake_batch_0003": 1}


def test_one_refused_submission_keeps_the_other_batches_results() -> None:
    class RefuseThird(FakeBackend):
        def submit(self, requests: Sequence[BatchRequest]) -> BatchJob:
            if [r.custom_id for r in requests] == ["c"]:
                raise RuntimeError("413 payload too large")
            return super().submit(requests)

    ids = ["a", "b", "c"]
    backend = RefuseThird(_ok_script(ids), max_requests=1)
    ex, sync = _executor(backend)
    out = ex.run_wave({cid: _step(cid) for cid in ids})
    assert _text(out["a"]) == "reply-a" and _text(out["b"]) == "reply-b"
    assert out["a"]._hidden_params[TIER_MARKER] == TIER_BATCH
    assert sync.calls == [_step("c")] and _text(out["c"]) == "sync:c"
    assert ex.stats.batches == 2 and ex.stats.sync_fallbacks == 1


def test_shared_deadline_cancels_every_open_batch() -> None:
    backend = _PerJobPolls(_ok_script(["a", "b"]), max_requests=1, polls_by_batch=[10**6, 10**6])
    ex, _sync = _executor(backend, max_poll_s=0.0)
    with pytest.raises(BatchExecutionFailed, match="not finished"):
        ex.run_wave({"a": _step("a"), "b": _step("b")})
    assert sorted(backend.canceled) == ["fake_batch_0001", "fake_batch_0002"]


def test_default_batches_a_lone_request_rather_than_paying_full_price() -> None:
    """Batch mode is a cost mode: with default settings a one-request wave (the
    tail window of a long document, a one-document link stage) still goes
    through the batch backend at batch price instead of running synchronously."""
    backend = FakeBackend(_ok_script(["only"]))
    sync = _SyncRecorder()
    ex = BatchExecutor(backend, sync_execute=sync, sleep=lambda _s: None)

    out = ex.run_wave({"only": _step("only")})

    assert _text(out["only"]) == "reply-only"
    assert out["only"]._hidden_params[TIER_MARKER] == TIER_BATCH
    assert sync.calls == []
    assert len(backend.submitted) == 1
    assert ex.stats.sync_fallbacks == 0


# ---- batch-level rejection: bisect, never the whole wave at full price -------


def test_batch_rejected_at_submit_is_bisected_until_it_fits() -> None:
    ids = [f"u{i}" for i in range(8)]

    def too_big(batch: list[BatchRequest]) -> Exception | None:
        return BatchRejected("over the enqueued-token limit") if len(batch) > 2 else None

    backend = FakeBackend(_ok_script(ids), fail_submit=too_big)
    ex, sync = _executor(backend)

    out = ex.run_wave({cid: _step(cid) for cid in ids})

    assert all(_text(out[cid]) == f"reply-{cid}" for cid in ids)
    assert all(out[cid]._hidden_params[TIER_MARKER] == TIER_BATCH for cid in ids)
    assert sync.calls == []
    # 8 → 4 + 4 → (2 + 2) + (2 + 2), halves in request order.
    assert [len(b) for b in backend.attempted] == [8, 4, 2, 2, 4, 2, 2]
    assert [[r.custom_id for r in b] for b in backend.submitted] == [
        ["u0", "u1"],
        ["u2", "u3"],
        ["u4", "u5"],
        ["u6", "u7"],
    ]
    assert ex.stats.bisections == 3 and ex.stats.batches == 4
    assert ex.stats.to_json()["bisections"] == 3


def test_single_request_rejected_at_batch_level_runs_synchronously() -> None:
    backend = FakeBackend({}, fail_submit=BatchRejected("queue full"))
    ex, _sync = _executor(backend)

    out = ex.run_wave({"a": _step("a"), "b": _step("b")})

    # Split down to single requests, each still refused: those run sync —
    # no BatchExecutionFailed, since each was refused alone.
    assert [len(b) for b in backend.attempted] == [2, 1, 1]
    assert _text(out["a"]) == "sync:a" and _text(out["b"]) == "sync:b"
    assert ex.stats.bisections == 1 and ex.stats.sync_fallbacks == 2


def test_accepted_batch_failed_whole_at_batch_level_is_bisected() -> None:
    ids = ["a", "b", "c", "d"]
    backend = FakeBackend(_ok_script(ids), reject_batch=lambda batch: len(batch) > 2)
    ex, sync = _executor(backend)

    out = ex.run_wave({cid: _step(cid) for cid in ids})

    assert all(_text(out[cid]) == f"reply-{cid}" for cid in ids)
    assert sync.calls == []
    assert [[r.custom_id for r in b] for b in backend.submitted] == [ids, ["a", "b"], ["c", "d"]]
    assert ex.stats.bisections == 1 and ex.stats.batches == 3
    assert ex.stats.batch_ok == 4 and ex.stats.resubmitted == 0
    # Every ended, collected batch is cleaned up — the rejected one included.
    assert backend.cleaned == ["fake_batch_0001", "fake_batch_0002", "fake_batch_0003"]


def test_single_request_batch_failed_at_batch_level_falls_back() -> None:
    backend = FakeBackend(_ok_script(["a"]), reject_batch=lambda _batch: True)
    ex, _sync = _executor(backend)

    out = ex.run_wave({"a": _step("a")})

    assert _text(out["a"]) == "sync:a"
    assert len(backend.submitted) == 1 and ex.stats.bisections == 0


def test_other_submit_errors_still_fail_a_wave_nothing_was_accepted_for() -> None:
    backend = FakeBackend({}, fail_submit=RuntimeError("401 bad key"))
    ex, sync = _executor(backend)
    with pytest.raises(BatchExecutionFailed, match="401 bad key"):
        ex.run_wave({"a": _step("a"), "b": _step("b")})
    assert len(backend.attempted) == 1 and sync.calls == []


# ---- uncertain create: never resubmitted, never paid at full price -----------


def test_uncertain_create_fails_the_wave_and_cancels_its_other_batches() -> None:
    def second_times_out(batch: list[BatchRequest]) -> Exception | None:
        if batch[0].custom_id == "b":
            return BatchSubmitUncertain("read timeout after send")
        return None

    backend = FakeBackend(_ok_script(["a", "b", "c"]), max_requests=1, fail_submit=second_times_out)
    ex, sync = _executor(backend)

    with pytest.raises(BatchExecutionFailed, match="may exist") as info:
        ex.run_wave({"a": _step("a"), "b": _step("b"), "c": _step("c")})

    assert isinstance(info.value.__cause__, BatchSubmitUncertain)
    assert [b[0].custom_id for b in backend.attempted] == ["a", "b"]  # c never sent
    assert backend.canceled == ["fake_batch_0001"]
    assert sync.calls == []
    # c was encoded for planning but never submitted: its encoding is released too.
    assert sorted(cid for ids in backend.released for cid in ids) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "failure",
    [BatchSubmitUncertain("read timeout after send"), "throttled"],
    ids=["uncertain", "throttled"],
)
def test_a_failed_wave_releases_the_encodings_it_never_submitted(failure: Any) -> None:
    """F9: planning encoded every batch of the wave; the ones never submitted
    after the wave failed mid-plan must not stay cached in the backend."""
    from dgml_core.batch.types import BatchThrottled

    error = BatchThrottled("HTTP 429") if failure == "throttled" else failure

    def second_fails(batch: list[BatchRequest]) -> Exception | None:
        return error if batch[0].custom_id == "b" else None

    ids = ["a", "b", "c", "d"]
    backend = FakeBackend(_ok_script(ids), max_requests=1, fail_submit=second_fails)
    ex, _sync = _executor(backend)
    with pytest.raises(BatchExecutionFailed):
        ex.run_wave({cid: _step(cid) for cid in ids})
    released = sorted(cid for batch in backend.released for cid in batch)
    assert released == ids


def test_uncertain_create_while_bisecting_cancels_every_open_batch() -> None:
    def halves_time_out(batch: list[BatchRequest]) -> Exception | None:
        return BatchSubmitUncertain("502 after send") if len(batch) == 1 else None

    class SecondRuns(FakeBackend):
        """Batch 2 (c, d) keeps running while batch 1 (a, b) ends rejected."""

        def poll(self, job: BatchJob) -> Any:
            if job.job_id == "fake_batch_0002":
                return BatchStatus(state=BatchState.RUNNING, processing=2)
            return super().poll(job)

    backend = SecondRuns(
        _ok_script(["a", "b", "c", "d"]),
        max_requests=2,
        reject_batch=lambda batch: [r.custom_id for r in batch] == ["a", "b"],
        fail_submit=halves_time_out,
    )
    ex, sync = _executor(backend)
    with pytest.raises(BatchExecutionFailed, match="may exist"):
        ex.run_wave({cid: _step(cid) for cid in ["a", "b", "c", "d"]})
    assert sync.calls == []
    assert backend.canceled == ["fake_batch_0002"]
    assert [len(b) for b in backend.attempted] == [2, 2, 1]  # no second half sent


# ---- rate-limited / over-quota create: fail loudly, never bisect or sync (F2) ---


def test_throttled_create_fails_the_wave_and_cancels_its_other_batches() -> None:
    from dgml_core.batch.types import BatchThrottled

    def second_throttled(batch: list[BatchRequest]) -> Exception | None:
        if batch[0].custom_id == "b":
            return BatchThrottled("HTTP 429: insufficient_quota")
        return None

    backend = FakeBackend(_ok_script(["a", "b", "c"]), max_requests=1, fail_submit=second_throttled)
    ex, sync = _executor(backend)

    with pytest.raises(BatchExecutionFailed, match="rate limit or quota") as info:
        ex.run_wave({"a": _step("a"), "b": _step("b"), "c": _step("c")})

    assert "billing" in str(info.value) and "later" in str(info.value)
    assert isinstance(info.value.__cause__, BatchThrottled)
    assert [b[0].custom_id for b in backend.attempted] == ["a", "b"]  # c never sent
    assert backend.canceled == ["fake_batch_0001"]
    assert sync.calls == [] and ex.stats.bisections == 0 and ex.stats.sync_fallbacks == 0


def test_throttled_create_of_a_multi_request_batch_is_not_bisected() -> None:
    from dgml_core.batch.types import BatchThrottled

    backend = FakeBackend({}, fail_submit=BatchThrottled("HTTP 429: rate_limit_error"))
    ex, sync = _executor(backend)
    with pytest.raises(BatchExecutionFailed, match="rate limit or quota"):
        ex.run_wave({"a": _step("a"), "b": _step("b")})
    assert [len(b) for b in backend.attempted] == [2]
    assert sync.calls == [] and ex.stats.bisections == 0


def test_throttled_create_while_bisecting_cancels_every_open_batch() -> None:
    from dgml_core.batch.types import BatchThrottled

    def halves_throttled(batch: list[BatchRequest]) -> Exception | None:
        return BatchThrottled("HTTP 429") if len(batch) == 1 else None

    class SecondRuns(FakeBackend):
        def poll(self, job: BatchJob) -> Any:
            if job.job_id == "fake_batch_0002":
                return BatchStatus(state=BatchState.RUNNING, processing=2)
            return super().poll(job)

    backend = SecondRuns(
        _ok_script(["a", "b", "c", "d"]),
        max_requests=2,
        reject_batch=lambda batch: [r.custom_id for r in batch] == ["a", "b"],
        fail_submit=halves_throttled,
    )
    ex, sync = _executor(backend)
    with pytest.raises(BatchExecutionFailed, match="rate limit or quota"):
        ex.run_wave({cid: _step(cid) for cid in ["a", "b", "c", "d"]})
    assert sync.calls == []
    assert backend.canceled == ["fake_batch_0002"]
    assert [len(b) for b in backend.attempted] == [2, 2, 1]


# ---- polling deadline from the provider's expiry window ---------------------


def test_default_poll_deadline_follows_the_backend_expiry_window() -> None:
    ex, _sync = _executor(FakeBackend({}, max_wait_s=48 * 3600))
    assert ex.max_poll_s == 48 * 3600 + POLL_MARGIN_S


def test_explicit_poll_deadline_wins_over_the_backend_window() -> None:
    ex, _sync = _executor(FakeBackend({}, max_wait_s=48 * 3600), max_poll_s=60.0)
    assert ex.max_poll_s == 60.0


def test_backend_without_an_expiry_window_keeps_the_24h_default() -> None:
    assert default_max_poll_s(object()) == DEFAULT_MAX_POLL_S
    assert DEFAULT_MAX_POLL_S == 24 * 3600


# ---- release and cleanup hooks ------------------------------------------------


def test_release_follows_every_submit_and_cleanup_every_collected_batch() -> None:
    backend = FakeBackend(_ok_script(["a", "b", "c"]), max_requests=2)
    ex, _sync = _executor(backend)
    ex.run_wave({"a": _step("a"), "b": _step("b"), "c": _step("c")})
    assert backend.released == [("a", "b"), ("c",)]
    assert backend.cleaned == ["fake_batch_0001", "fake_batch_0002"]


def test_release_follows_a_refused_submit_too() -> None:
    backend = FakeBackend({}, fail_submit=RuntimeError("nope"))
    ex, _sync = _executor(backend)
    with pytest.raises(BatchExecutionFailed):
        ex.run_wave({"a": _step("a")})
    assert backend.released == [("a",)] and backend.cleaned == []


def test_a_failing_cleanup_never_fails_the_wave() -> None:
    class CleanupFails(FakeBackend):
        def cleanup(self, job: BatchJob) -> None:
            raise RuntimeError("delete failed")

    logs: list[str] = []
    backend = CleanupFails(_ok_script(["a"]))
    ex, _sync = _executor(backend, log=logs.append)
    out = ex.run_wave({"a": _step("a")})
    assert _text(out["a"]) == "reply-a"
    assert any("cleanup of fake_batch_0001 failed" in line for line in logs)


def test_a_backend_without_the_optional_hooks_still_works() -> None:
    class NoHooks:
        provider = "bare"
        max_requests = 10
        max_bytes = 10**6

        def __init__(self) -> None:
            self.inner = FakeBackend(_ok_script(["a"]))

        def encode(self, request: BatchRequest) -> dict[str, Any]:
            return self.inner.encode(request)

        def submit(self, requests: Sequence[BatchRequest]) -> BatchJob:
            return self.inner.submit(requests)

        def poll(self, job: BatchJob) -> Any:
            return self.inner.poll(job)

        def results(self, job: BatchJob) -> Iterator[tuple[str, Any]]:
            return self.inner.results(job)

        def cancel(self, job: BatchJob) -> None:
            self.inner.cancel(job)

    # The protocol now declares the hooks; the executor still tolerates a
    # third-party backend that predates them (``getattr`` lookups).
    bare: Any = NoHooks()
    ex = BatchExecutor(bare, sync_execute=_SyncRecorder(), sleep=lambda _s: None)
    assert ex.max_poll_s == DEFAULT_MAX_POLL_S
    assert _text(ex.run_wave({"a": _step("a")})["a"]) == "reply-a"
