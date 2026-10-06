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

"""``run_stage`` / ``run_stage_sync``: many ``steps_*`` generators, wave by wave."""

from __future__ import annotations

import inspect
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from dgml_core import llm
from dgml_core.batch import (
    BatchExecutor,
    BatchItemError,
    BatchRequest,
    FakeBackend,
    Unit,
    fake_model_response,
    run_stage,
    run_stage_sync,
)
from dgml_core.batch.driver import CUSTOM_ID_PATTERN
from dgml_core.errors import BatchExecutionFailed, EmptyModelResponse
from dgml_core.storage import Workspace
from dgml_core.usage import TIER_BATCH, TIER_STANDARD, read_events

from .conftest import FakeLLMResponse

MODEL = "anthropic/claude-haiku-4-5"


def _cfg(ws: Workspace | None = None, **kw: Any) -> llm.LLMConfig:
    return llm.LLMConfig(
        model=MODEL,
        workspace=ws,
        debug=ws is not None,
        operation="transcribe",
        **kw,
    )


def _text(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": content}]


def _call_unit(name: str, cfg: llm.LLMConfig) -> Unit:
    return Unit(name, cfg, llm.steps_call(cfg, system_prompt="S", user_content=_text(name)))


def _continued_unit(name: str, cfg: llm.LLMConfig) -> Unit:
    return Unit(name, cfg, llm.steps_continued(cfg, system_prompt="S", user_content=_text(name)))


def _key(custom_id: str) -> str:
    """``u<i>_<stem>-<n>`` → ``<stem>#<n>``: the tests script and assert by unit
    name and step number (their unit names are plain alphanumerics)."""
    prefix, n = custom_id.rsplit("-", 1)
    stem = prefix.split("_", 1)[1] if "_" in prefix else ""
    return f"{stem}#{n}"


def _script(table: dict[str, Any]) -> Any:
    """Script keyed by ``<unit name>#<n>``; anything else is invalid."""

    def script(req: BatchRequest) -> Any:
        try:
            return table[_key(req.custom_id)]
        except KeyError:
            return BatchItemError(req.custom_id, "invalid", f"unscripted {req.custom_id}")

    return script


def _executor(backend: FakeBackend, **kw: Any) -> BatchExecutor:
    kw.setdefault("sleep", lambda _s: None)
    kw.setdefault("min_wave_size", 1)
    kw.setdefault("sync_execute", _sync_from_kwargs)
    return BatchExecutor(backend, **kw)


def _sync_from_kwargs(kwargs: dict[str, Any]) -> Any:
    """Answer with the unit's tag: the text of the first user turn (a
    continuation round ends with the assistant prefill, not a user turn)."""
    user = next(m for m in kwargs["messages"] if m["role"] == "user")
    return FakeLLMResponse(f"sync:{user['content'][0]['text']}")


def test_results_land_on_the_right_unit_under_shuffled_delivery() -> None:
    cfg = _cfg()
    backend = FakeBackend(
        _script({f"u{i}#1": FakeLLMResponse(f"reply-u{i}") for i in range(5)}),
        shuffle=True,
        seed=3,
    )
    ex = _executor(backend)
    out = run_stage([_call_unit(f"u{i}", cfg) for i in range(5)], ex)

    assert list(out) == [f"u{i}" for i in range(5)]
    assert all(out[f"u{i}"].result == f"reply-u{i}" for i in range(5))
    assert all(o.ok and o.steps == 1 and o.batch_steps == 1 for o in out.values())
    assert ex.stats.waves == 1
    assert [_key(r.custom_id) for r in backend.submitted[0]] == [f"u{i}#1" for i in range(5)]


def test_continuation_spans_two_waves_and_prefills_the_partial() -> None:
    cfg = _cfg()
    backend = FakeBackend(
        _script(
            {
                "long#1": FakeLLMResponse("part-1", finish_reason="length"),
                "long#2": FakeLLMResponse("part-2"),
                "short#1": FakeLLMResponse("done"),
            }
        )
    )
    ex = _executor(backend)
    out = run_stage([_continued_unit("long", cfg), _call_unit("short", cfg)], ex)

    assert out["long"].result == "part-1part-2" and out["long"].steps == 2
    assert out["short"].result == "done" and out["short"].steps == 1
    assert ex.stats.waves == 2
    assert [[_key(r.custom_id) for r in b] for b in backend.submitted] == [
        ["long#1", "short#1"],
        ["long#2"],
    ]
    # Wave 2 carries the Anthropic prefill continuation built by steps_continued.
    second = backend.submitted[1][0].kwargs["messages"]
    assert second[-1] == {"role": "assistant", "content": "part-1"}


def test_straggler_wave_runs_synchronously_under_the_default_min_size() -> None:
    cfg = _cfg()
    backend = FakeBackend(
        _script(
            {
                "long#1": FakeLLMResponse("part-1", finish_reason="length"),
                "short#1": FakeLLMResponse("done"),
            }
        )
    )
    ex = _executor(backend, min_wave_size=2)
    out = run_stage([_continued_unit("long", cfg), _call_unit("short", cfg)], ex)
    assert out["long"].result == "part-1sync:long"
    assert out["long"].batch_steps == 1 and out["long"].sync_steps == 1
    assert len(backend.submitted) == 1 and ex.stats.sync_fallbacks == 1


def test_one_units_parse_failure_is_isolated(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)

    def boom() -> Generator[dict[str, Any], Any, str]:
        _ = yield llm._build_completion_kwargs(cfg, messages=[{"role": "user", "content": "x"}])
        raise ValueError("bad parse")

    backend = FakeBackend(
        _script(
            {
                "ok#1": FakeLLMResponse("fine", cost=0.01, prompt_tokens=10, completion_tokens=5),
                "bad#1": FakeLLMResponse("junk", cost=0.02, prompt_tokens=20, completion_tokens=7),
            }
        )
    )
    out = run_stage([_call_unit("ok", cfg), Unit("bad", cfg, boom())], _executor(backend))

    assert out["ok"].ok and out["ok"].result == "fine"
    assert isinstance(out["bad"].error, ValueError) and out["bad"].result is None
    assert out["bad"].steps == 1

    rows = {r["context"].get("unit", i): r for i, r in enumerate(read_events(ws))}
    assert len(rows) == 2
    by_outcome = {r["outcome"]: r for r in rows.values()}
    assert by_outcome["error"]["error"] == "ValueError: bad parse"
    # The failed unit's row still carries what it spent before parsing failed.
    assert by_outcome["error"]["cost_usd"] == pytest.approx(0.02)
    assert by_outcome["error"]["total_tokens"] == 27
    assert by_outcome["ok"]["cost_usd"] == pytest.approx(0.01)


def test_empty_choices_reaching_a_generator_is_an_isolated_unit_error() -> None:
    cfg = _cfg()
    backend = FakeBackend(_script({"a#1": {"choices": []}, "b#1": FakeLLMResponse("ok")}))
    # The executor's fallback also returns an empty response here, so the
    # generator's own guard is what fails the unit.
    ex = _executor(backend, sync_execute=lambda _k: {"choices": []}, max_item_retries=0)
    out = run_stage([_call_unit("a", cfg), _call_unit("b", cfg)], ex)
    assert isinstance(out["a"].error, EmptyModelResponse)
    assert out["a"].sync_steps == 1 and out["a"].batch_steps == 0
    assert out["b"].result == "ok"


def test_item_error_fallback_is_counted_on_the_unit() -> None:
    cfg = _cfg()

    def script(req: BatchRequest) -> Any:
        if _key(req.custom_id) == "flaky#1":
            return BatchItemError(req.custom_id, "errored", "hiccup")
        return FakeLLMResponse("ok")

    seen: list[dict[str, Any]] = []

    def sync(kwargs: dict[str, Any]) -> Any:
        seen.append(kwargs)
        return FakeLLMResponse("sync-ok")

    ex = _executor(FakeBackend(script), sync_execute=sync, max_item_retries=0)
    out = run_stage([_call_unit("flaky", cfg), _call_unit("solid", cfg)], ex)
    assert out["flaky"].result == "sync-ok"
    assert out["flaky"].sync_steps == 1 and out["flaky"].batch_steps == 0
    assert out["solid"].batch_steps == 1 and out["solid"].sync_steps == 0
    assert len(seen) == 1 and seen[0]["messages"][-1]["content"] == _text("flaky")


def test_one_batch_tier_usage_row_per_unit_with_summed_totals(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws, context={"stage": "pass-a"})
    backend = FakeBackend(
        _script(
            {
                "a#1": FakeLLMResponse(
                    "p1", finish_reason="length", cost=0.01, prompt_tokens=100, completion_tokens=50
                ),
                "a#2": FakeLLMResponse(
                    "p2", cost=0.02, prompt_tokens=150, completion_tokens=25, cache_read_tokens=40
                ),
                "b#1": FakeLLMResponse("only", cost=0.05, prompt_tokens=10, completion_tokens=1),
            }
        )
    )
    out = run_stage([_continued_unit("a", cfg), _call_unit("b", cfg)], _executor(backend))
    assert out["a"].result == "p1p2" and out["b"].result == "only"

    rows = read_events(ws)
    assert len(rows) == 2
    assert {r["tier"] for r in rows} == {TIER_BATCH}
    assert {r["operation"] for r in rows} == {"transcribe"}
    assert all(r["context"] == {"stage": "pass-a"} and r["outcome"] == "ok" for r in rows)
    by_cost = sorted(rows, key=lambda r: r["cost_usd"])
    assert by_cost[0]["cost_usd"] == pytest.approx(0.03)
    assert by_cost[0]["prompt_tokens"] == 250
    assert by_cost[0]["completion_tokens"] == 75
    assert by_cost[0]["cache_read_tokens"] == 40
    assert by_cost[1]["cost_usd"] == pytest.approx(0.05)
    assert by_cost[1]["total_tokens"] == 11


def test_units_inside_record_usage_for_fold_into_the_scope_row(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws, tier=TIER_BATCH)
    backend = FakeBackend(
        _script(
            {
                "a#1": FakeLLMResponse("x", cost=0.01, prompt_tokens=10, completion_tokens=1),
                "b#1": FakeLLMResponse("y", cost=0.02, prompt_tokens=20, completion_tokens=2),
            }
        )
    )
    with llm.record_usage_for(cfg):
        run_stage([_call_unit("a", cfg), _call_unit("b", cfg)], _executor(backend))

    rows = read_events(ws)
    assert len(rows) == 1
    assert rows[0]["cost_usd"] == pytest.approx(0.03)
    assert rows[0]["prompt_tokens"] == 30 and rows[0]["completion_tokens"] == 3
    # The scope row carries the scope config's own tier — set it to batch there.
    assert rows[0]["tier"] == TIER_BATCH


def test_run_stage_sync_produces_the_same_results_at_standard_tier(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)
    responses = iter(
        [
            FakeLLMResponse("p1", finish_reason="length", cost=0.01, prompt_tokens=1),
            FakeLLMResponse("p2", cost=0.02, prompt_tokens=2),
            FakeLLMResponse("only", cost=0.05, prompt_tokens=5),
        ]
    )
    seen: list[dict[str, Any]] = []

    def execute(kwargs: dict[str, Any]) -> Any:
        seen.append(kwargs)
        return next(responses)

    out = run_stage_sync([_continued_unit("a", cfg), _call_unit("b", cfg)], execute)
    assert out["a"].result == "p1p2" and out["a"].steps == 2 and out["a"].sync_steps == 2
    assert out["b"].result == "only" and out["b"].steps == 1
    assert all(o.batch_steps == 0 for o in out.values())
    assert len(seen) == 3

    rows = read_events(ws)
    assert len(rows) == 2 and {r["tier"] for r in rows} == {TIER_STANDARD}
    assert sorted(r["cost_usd"] for r in rows) == [pytest.approx(0.03), pytest.approx(0.05)]


def test_run_stage_sync_isolates_unit_errors_and_uses_the_llm_seam(
    capture_kwargs: Any,
) -> None:
    cfg = _cfg()
    capture_kwargs([FakeLLMResponse("first"), {"choices": []}])
    out = run_stage_sync([_call_unit("a", cfg), _call_unit("b", cfg)])
    assert out["a"].result == "first"
    assert isinstance(out["b"].error, EmptyModelResponse)


def test_duplicate_unit_names_are_rejected() -> None:
    cfg = _cfg()
    units = [_call_unit("dup", cfg), _call_unit("dup", cfg)]
    with pytest.raises(ValueError, match="duplicated: \\['dup'\\]"):
        run_stage(units, _executor(FakeBackend({})))
    with pytest.raises(ValueError, match="duplicated"):
        run_stage_sync([_call_unit("dup", cfg), _call_unit("dup", cfg)])


def test_every_custom_id_matches_the_strictest_provider_rule() -> None:
    """Anthropic accepts only ``^[a-zA-Z0-9_-]{1,64}$``; ids must satisfy it for
    any unit name — dots, spaces, unicode, very long names — at any step."""
    cfg = _cfg()
    names = [
        "Big Sky - 12.1.pdf",
        "résumé \u2013 ñandú 見積書.pdf",  # en dash, accents, CJK
        "x" * 200,
        "a.b.c",
        "...",
        "",
        "#weird#name#",
    ]
    backend = FakeBackend(lambda _r: FakeLLMResponse("p", finish_reason="length"))
    units = [
        Unit(n, cfg, llm.steps_continued(cfg, system_prompt="S", user_content=_text("t")))
        for n in names
    ]
    out = run_stage(units, _executor(backend))
    assert all(o.ok for o in out.values())
    ids = [r.custom_id for batch in backend.submitted for r in batch]
    assert len(ids) == len(names) * 4  # max_rounds rounds of continuation each
    assert all(CUSTOM_ID_PATTERN.match(i) for i in ids), ids
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", i) for i in ids)
    assert len(set(ids)) == len(ids)
    assert ids[0] == "u0_Big_Sky_12_1_pdf-1"


def test_sanitised_names_can_never_collide() -> None:
    """'a b', 'a_b' and 'a_b_2' sanitise into each other's suffixed forms; the
    index in every id keeps them distinct."""
    cfg = _cfg()
    backend = FakeBackend(lambda _r: FakeLLMResponse("ok"))
    names = ["a b", "a_b", "a_b_2", "a-b", "a.b"]
    out = run_stage([_call_unit(n, cfg) for n in names], _executor(backend))
    assert [o.result for o in out.values()] == ["ok"] * 5
    ids = [r.custom_id for r in backend.submitted[0]]
    assert len(set(ids)) == len(ids) == 5
    assert ids == ["u0_a_b-1", "u1_a_b-1", "u2_a_b_2-1", "u3_a_b-1", "u4_a_b-1"]


def test_unit_that_returns_before_yielding_makes_no_request() -> None:
    cfg = _cfg()

    def instant() -> Generator[dict[str, Any], Any, str]:
        return "cached"
        yield {}  # pragma: no cover - makes this a generator

    backend = FakeBackend(_script({"live#1": FakeLLMResponse("ok")}))
    ex = _executor(backend)
    out = run_stage([Unit("skip", cfg, instant()), _call_unit("live", cfg)], ex)
    assert out["skip"].result == "cached" and out["skip"].steps == 0
    assert out["live"].result == "ok"
    assert [_key(r.custom_id) for r in backend.submitted[0]] == ["live#1"]


def test_executor_failure_becomes_every_unfinished_units_outcome(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)
    backend = FakeBackend({}, fail_submit=RuntimeError("provider down"))
    units = [_call_unit("a", cfg), _call_unit("b", cfg)]
    out = run_stage(units, _executor(backend))
    assert all(isinstance(o.error, BatchExecutionFailed) and o.stage_error for o in out.values())
    assert all("provider down" in str(o.error) for o in out.values())
    assert all(inspect.getgeneratorstate(u.steps) == inspect.GEN_CLOSED for u in units)
    rows = read_events(ws)
    assert len(rows) == 2
    assert all(r["outcome"] == "error" and "provider down" in r["error"] for r in rows)
    assert {r["tier"] for r in rows} == {TIER_BATCH}


def _multi_step_unit(name: str, cfg: llm.LLMConfig, steps: int) -> Unit:
    """A unit that makes *steps* sequential requests and returns their texts."""

    def gen() -> Generator[dict[str, Any], Any, list[str]]:
        texts: list[str] = []
        for n in range(1, steps + 1):
            response = yield {
                "model": MODEL,
                "messages": [{"role": "user", "content": [{"type": "text", "text": name}]}],
                "n": n,
            }
            texts.append(response.choices[0].message.content)
        return texts

    return Unit(name, cfg, gen())


class _FailSecondWave(FakeBackend):
    """Serves the first submitted batch; refuses every later submit."""

    def submit(self, requests: Any) -> Any:
        if self.submitted:
            raise RuntimeError("provider down")
        return super().submit(requests)


def test_stage_failure_keeps_the_results_of_units_that_finished(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)
    backend = _FailSecondWave(lambda req: fake_model_response(f"r:{_key(req.custom_id)}"))
    units = [_multi_step_unit("done", cfg, 1), _multi_step_unit("long", cfg, 2)]

    out = run_stage(units, _executor(backend))

    assert out["done"].ok and out["done"].result == ["r:done#1"]
    assert not out["done"].stage_error
    assert isinstance(out["long"].error, BatchExecutionFailed) and out["long"].stage_error
    assert out["long"].steps == 1
    rows = {r["outcome"] for r in read_events(ws)}
    assert rows == {"ok", "error"}


def _wave_keys(backend: FakeBackend) -> list[list[str]]:
    return [[_key(r.custom_id) for r in batch] for batch in backend.submitted]


def test_every_unit_is_admitted_into_the_first_wave() -> None:
    cfg = _cfg()
    backend = FakeBackend(lambda req: fake_model_response("ok"))
    run_stage([_call_unit(f"u{i}", cfg) for i in range(4)], _executor(backend))
    assert _wave_keys(backend) == [["u0#1", "u1#1", "u2#1", "u3#1"]]


def test_interrupt_during_a_wave_closes_units_and_propagates() -> None:
    cfg = _cfg()

    class Interrupting(FakeBackend):
        def poll(self, job: Any) -> Any:
            raise KeyboardInterrupt

    units = [_call_unit("a", cfg), _call_unit("b", cfg)]
    with pytest.raises(KeyboardInterrupt):
        run_stage(units, _executor(Interrupting({})))
    assert all(inspect.getgeneratorstate(u.steps) == inspect.GEN_CLOSED for u in units)


def _failing_executor(backend: FakeBackend, exc: Exception) -> BatchExecutor:
    def sync(_kwargs: dict[str, Any]) -> Any:
        raise exc

    return _executor(backend, sync_execute=sync, max_item_retries=0)


def test_unserved_request_is_thrown_into_the_generator_which_retries(tmp_path: Path) -> None:
    """A stage generator that catches around ``yield from`` — as the sync code
    catches around ``llm.call`` — gets the failure at its pending yield and can
    retry; the retry is the next wave."""
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)
    caught: list[str] = []

    def retry_once() -> Generator[dict[str, Any], Any, str]:
        for attempt in range(2):
            try:
                return (yield from llm.steps_call(cfg, system_prompt="S", user_content=_text("r")))
            except ConnectionError as exc:
                caught.append(str(exc))
                if attempt:
                    raise
        raise AssertionError("unreachable")

    table: dict[str, Any] = {
        "r#1": BatchItemError("r#1", "invalid", "rejected"),
        "r#2": FakeLLMResponse("second-try", cost=0.02, prompt_tokens=5, completion_tokens=1),
        "other#1": FakeLLMResponse("fine", cost=0.01, prompt_tokens=1, completion_tokens=1),
    }
    backend = FakeBackend(_script(table))
    ex = _failing_executor(backend, ConnectionError("unreachable host"))
    out = run_stage([Unit("r", cfg, retry_once()), _call_unit("other", cfg)], ex)

    assert out["r"].ok and out["r"].result == "second-try"
    assert caught == ["unreachable host"]
    assert out["r"].steps == 2 and out["r"].failed_steps == 1 and out["r"].batch_steps == 1
    assert [[_key(r.custom_id) for r in b] for b in backend.submitted] == [
        ["r#1", "other#1"],
        ["r#2"],
    ]
    rows = read_events(ws)
    assert {r["outcome"] for r in rows} == {"ok"}
    # The failed attempt contributes no usage.
    assert sorted(r["cost_usd"] for r in rows) == [pytest.approx(0.01), pytest.approx(0.02)]


def test_uncaught_thrown_failure_fails_only_that_unit(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)
    table: dict[str, Any] = {
        "a#1": BatchItemError("a#1", "invalid", "rejected"),
        "b#1": FakeLLMResponse("fine", cost=0.01, prompt_tokens=1),
    }
    units = [_call_unit("a", cfg), _call_unit("b", cfg)]
    ex = _failing_executor(FakeBackend(_script(table)), ConnectionError("down"))
    out = run_stage(units, ex)
    assert isinstance(out["a"].error, ConnectionError) and out["a"].failed_steps == 1
    assert out["b"].result == "fine"
    assert inspect.getgeneratorstate(units[0].steps) == inspect.GEN_CLOSED
    rows = {r["outcome"]: r for r in read_events(ws)}
    assert rows["error"]["error"] == "ConnectionError: down"
    assert rows["ok"]["cost_usd"] == pytest.approx(0.01)


def test_generator_may_return_a_fallback_value_from_a_thrown_failure() -> None:
    cfg = _cfg()

    def tolerant() -> Generator[dict[str, Any], Any, str]:
        try:
            return (yield from llm.steps_call(cfg, system_prompt="S", user_content=_text("t")))
        except ConnectionError:
            return "degraded"

    table = {
        "t#1": BatchItemError("t#1", "invalid", "rejected"),
        "x#1": FakeLLMResponse("x"),
    }
    ex = _failing_executor(FakeBackend(_script(table)), ConnectionError("down"))
    out = run_stage([Unit("t", cfg, tolerant()), _call_unit("x", cfg)], ex)
    assert out["t"].ok and out["t"].result == "degraded"


def test_run_stage_sync_throws_failures_into_the_generator_too() -> None:
    cfg = _cfg()
    outcomes: list[Any] = [ConnectionError("blip"), FakeLLMResponse("ok")]

    def execute(_kwargs: dict[str, Any]) -> Any:
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o

    def retry_once() -> Generator[dict[str, Any], Any, str]:
        try:
            return (yield from llm.steps_call(cfg, system_prompt="S", user_content=_text("s")))
        except ConnectionError:
            return (yield from llm.steps_call(cfg, system_prompt="S", user_content=_text("s")))

    out = run_stage_sync([Unit("s", cfg, retry_once())], execute)
    assert out["s"].result == "ok" and out["s"].steps == 2 and out["s"].failed_steps == 1


def _context_setting_unit(name: str, cfg: llm.LLMConfig) -> Unit:
    """Like transcribe_steps: the generator sets context on the config it holds."""

    def gen() -> Generator[dict[str, Any], Any, str]:
        cfg.context = {"doc": f"{name}.pdf"}
        return (yield from llm.steps_call(cfg, system_prompt="S", user_content=_text(name)))

    return Unit(name, cfg, gen())


def test_row_carries_context_the_generator_set_and_batch_tier(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg_a, cfg_b = _cfg(ws), _cfg(ws)
    backend = FakeBackend(lambda _r: FakeLLMResponse("ok", cost=0.01, prompt_tokens=1))
    run_stage(
        [_context_setting_unit("a", cfg_a), _context_setting_unit("b", cfg_b)],
        _executor(backend),
    )
    rows = read_events(ws)
    assert sorted(r["context"]["doc"] for r in rows) == ["a.pdf", "b.pdf"]
    assert {r["tier"] for r in rows} == {TIER_BATCH}
    # The configs' own tier is restored once the stage is over.
    assert cfg_a.tier == cfg_b.tier == TIER_STANDARD


def test_enclosing_standard_scope_row_is_marked_batch(tmp_path: Path) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)  # standard tier, as a stage caller would build it
    backend = FakeBackend(lambda _r: FakeLLMResponse("ok", cost=0.01, prompt_tokens=2))
    with llm.record_usage_for(cfg):
        run_stage([_call_unit("a", cfg), _call_unit("b", cfg)], _executor(backend))
    rows = read_events(ws)
    assert len(rows) == 1
    assert rows[0]["tier"] == TIER_BATCH
    assert rows[0]["cost_usd"] == pytest.approx(0.02)
    assert cfg.tier == TIER_STANDARD


def test_sync_scope_row_stays_standard(tmp_path: Path) -> None:
    """The sink marker is the driver's: an ordinary sync scope is unaffected."""
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)
    with llm.record_usage_for(cfg):
        run_stage_sync([_call_unit("a", cfg)], lambda _k: FakeLLMResponse("ok", cost=0.01))
    assert [r["tier"] for r in read_events(ws)] == [TIER_STANDARD]


def test_unit_that_makes_no_request_writes_no_row(tmp_path: Path) -> None:
    """Parity with the sync cached-transcription path, which short-circuits
    before any accounting scope exists."""
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)

    def cached() -> Generator[dict[str, Any], Any, str]:
        return "cached blocks"
        yield {}  # pragma: no cover - makes this a generator

    backend = FakeBackend(_script({"live#1": FakeLLMResponse("ok", cost=0.01)}))
    out = run_stage([Unit("skip", cfg, cached()), _call_unit("live", cfg)], _executor(backend))
    assert out["skip"].result == "cached blocks"
    assert len(read_events(ws)) == 1  # only the live unit's row


def test_unit_failing_before_its_first_request_still_writes_an_error_row(
    tmp_path: Path,
) -> None:
    ws = Workspace(root=tmp_path)
    cfg = _cfg(ws)

    def broken() -> Generator[dict[str, Any], Any, str]:
        raise ValueError("bad input")
        yield {}  # pragma: no cover - makes this a generator

    backend = FakeBackend(_script({"live#1": FakeLLMResponse("ok")}))
    out = run_stage([Unit("broken", cfg, broken()), _call_unit("live", cfg)], _executor(backend))
    assert isinstance(out["broken"].error, ValueError)
    outcomes = sorted(r["outcome"] for r in read_events(ws))
    assert outcomes == ["error", "ok"]
