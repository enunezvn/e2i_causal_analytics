"""Every step result carries its outcome class, attempts, cache hit and error type (spec §5.3).

The learning loop records one row per finished step and counts reliability from these fields,
so each ``_execute_step`` exit must say what happened from the exception type it already
caught, never from error text:

- succeeded / cache_hit;
- refused (``ToolRefusalError``) / input_rejected (``ToolInputError``): the data could not
  answer, not a tool fault;
- timeout (``SyncToolTimeout``, or the async ``wait_for`` on the last attempt) / error: health
  failures;
- plan_defect / dependency_unmet / circuit_open / not_registered: the tool never ran.

``execute(..., on_step_result=cb)`` hands each ``StepResult`` to the recorder the moment it
exists, inside the parallel-group task, so a step that finished before a cancel or an escaping
exception in its group is not lost.

Refusals and the async input rejection come from REAL tool guard code (``tool_registrations``)
over real frames. The sync input rejection is the one exception: the only guard on this branch
raising ``ToolInputError`` is ``counterfactual_simulator``, which #2015 turns async, so a local
sync probe stands in for it on that arm. Otherwise the probe callables only count calls, sleep
or raise.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.cache import get_cache_manager
from src.agents.tool_composer.executor import ExecutionError, PlanExecutor
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    StepResult,
    SubQuestion,
    ToolMapping,
)
from src.tool_registry.registry import ToolSchema, get_registry


@pytest.fixture(autouse=True)
def _clean_bounded_pool():
    """Sync tools run on the #1592 bounded pool; give every case a fresh one."""
    from src.api.dependencies.compute import _reset_limiter_cache_for_tests

    _reset_limiter_cache_for_tests()
    yield
    _reset_limiter_cache_for_tests()


@pytest.fixture
def registry():
    """The process registry, emptied for the test and restored exactly afterwards (#782)."""
    live = get_registry()
    snapshot = live.snapshot()
    live.clear()
    try:
        yield live
    finally:
        live.restore_snapshot(snapshot)


def _register(registry, name: str, fn: Any) -> None:
    registry.register(
        schema=ToolSchema(
            name=name,
            description="outcome-class probe tool.",
            source_agent="gap_analyzer",
            tier=2,
            input_parameters=[],
            output_schema="Dict[str, Any]",
            avg_execution_ms=10,
        ),
        callable=fn,
    )


def _step(
    step_id: str,
    tool: str,
    *,
    depends_on: Sequence[str] = (),
    input_mapping: Optional[Dict[str, Any]] = None,
) -> ExecutionStep:
    return ExecutionStep(
        step_id=step_id,
        sub_question_id="sq_1",
        tool_name=tool,
        source_agent="gap_analyzer",
        input_mapping=input_mapping or {},
        dependency_type=DependencyType.SEQUENTIAL,
        depends_on_steps=list(depends_on),
    )


def _plan(steps: List[ExecutionStep], groups: List[List[str]]) -> ExecutionPlan:
    return ExecutionPlan(
        decomposition=DecompositionResult(
            original_query="outcome classes?",
            sub_questions=[
                SubQuestion(id="sq_1", question="q1", intent="CAUSAL", entities=[], depends_on=[])
            ],
            decomposition_reasoning="t",
            timestamp=datetime.now(timezone.utc),
        ),
        steps=steps,
        tool_mappings=[
            ToolMapping(
                sub_question_id="sq_1",
                tool_name=s.tool_name,
                source_agent="gap_analyzer",
                confidence=0.9,
                reasoning="t",
            )
            for s in steps
        ],
        parallel_groups=groups,
        planning_reasoning="t",
    )


def _one(step_id: str, tool: str, **kw: Any) -> ExecutionPlan:
    return _plan([_step(step_id, tool, **kw)], [[step_id]])


def _executor(registry, **overrides: Any) -> PlanExecutor:
    settings: Dict[str, Any] = {
        "tool_registry": registry,
        "max_retries": 2,
        "backoff_base_delay": 0.0,
        "backoff_max_delay": 0.0,
        "enable_caching": False,
    }
    settings.update(overrides)
    return PlanExecutor(**settings)


def _classes(result: StepResult) -> Tuple[Optional[str], int, bool, Optional[str]]:
    return (result.outcome_class, result.attempts, result.cache_hit, result.error_type)


def _single_brand_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"brand": ["Kisqali"] * 6, "market_share": [0.72, 0.71, 0.74, 0.73, 0.7, 0.72]}
    )


def _refusing_gap_calculator(**_: Any) -> Any:
    # REAL guard (#1574): one brand group cannot support a brand-vs-brand gap.
    return tr.gap_calculator(
        metric="market_share",
        entity_type="brand",
        entities=["Kisqali", "competitor"],
        estimation_data=_single_brand_frame(),
    )


# ---------------------------------------------------------------------------
# One class per exit of _execute_step
# ---------------------------------------------------------------------------


async def test_succeeded_class(registry):
    _register(registry, "ok_tool", lambda **_: {"ok": True})
    trace = await _executor(registry).execute(_one("s", "ok_tool"))
    assert _classes(trace.step_results[0]) == ("succeeded", 1, False, None)


async def test_refusal_class(registry):
    _register(registry, "gap_calculator", _refusing_gap_calculator)
    trace = await _executor(registry).execute(_one("s", "gap_calculator"))
    result = trace.step_results[0]
    assert result.status == ExecutionStatus.FAILED
    assert _classes(result) == ("refused", 1, False, "ToolRefusalError")


async def test_input_rejected_class(registry):
    calls: List[int] = []

    async def rejecting(**_: Any) -> Any:
        calls.append(1)
        # REAL guard, and deliberately one that BOTH tool contracts refuse, because this call
        # has to survive a signature change landing from another branch:
        #   here (#1573)  - expected_effect=None is no usable effect estimate;
        #   after #2015   - "speaker program" is not a digital-twin intervention (the catalog
        #                   holds speaker_program_invitation), and brand is required.
        # Each signature absorbs the other's extra argument through **kwargs, so one call fits
        # both. #2015 also makes the tool async, hence awaiting an awaitable result: a
        # coroutine returned from a sync probe would never raise and the case would silently
        # pass as a success.
        result = tr.counterfactual_simulator(
            intervention="speaker program",
            brand="Kisqali",
            target_entities=["NE"],
            expected_effect=None,
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    _register(registry, "counterfactual_simulator", rejecting)
    trace = await _executor(registry).execute(_one("s", "counterfactual_simulator"))
    assert _classes(trace.step_results[0]) == ("input_rejected", 1, False, "ToolInputError")
    # Reported attempts is what the step SAYS; the call count is what happened.
    assert calls == [1]


@pytest.mark.parametrize(
    "message",
    [
        "declined: the probe's inputs cannot support this calculation",
        "invalid sample size",
        "refused: this cohort cannot answer that",
    ],
    ids=["says_declined", "neutral_wording", "wording_of_a_refusal"],
)
async def test_input_rejected_class_on_the_async_path(registry, message):
    """The same wording independence, on the ASYNC arm.

    The sync matrix below does not pin this arm. An implementation that keyed on the word
    "declined" for coroutine callables and on the exception type otherwise would satisfy every
    sync case AND the real-guard case above — whose #1573 message happens to say "declined" —
    while an async ``ToolInputError("invalid sample size")`` was recorded as a refusal.

    The real-guard case above is left alone deliberately: it exercises the actual tool contract
    across the #2015 transition, which a synthetic probe cannot. This one varies only the message.
    """
    from src.agents.tool_composer.errors import ToolInputError

    calls: List[int] = []

    async def declining_async(**_: Any) -> Any:
        calls.append(1)
        raise ToolInputError(message)

    assert asyncio.iscoroutinefunction(declining_async), "this case must take the async arm"

    _register(registry, "power_calculator", declining_async)
    trace = await _executor(registry).execute(_one("s", "power_calculator"))

    assert _classes(trace.step_results[0]) == ("input_rejected", 1, False, "ToolInputError")
    assert calls == [1], "a retry would run it again; reported attempts alone is not proof"


@pytest.mark.parametrize(
    "message",
    [
        "declined: the probe's inputs cannot support this calculation",
        "invalid sample size",
        "refused: this cohort cannot answer that",
    ],
    ids=["says_declined", "neutral_wording", "wording_of_a_refusal"],
)
async def test_input_rejected_class_on_the_sync_path(registry, message):
    """The same class through the executor's SYNC arm (``executor.py`` iscoroutinefunction split).

    The case above has to be async, because #2015 turns ``counterfactual_simulator`` async. The
    sync arm has its own transport — ``_run_sync_tool`` on the #1592 bounded pool — and only then
    reaches the shared ``except (ToolInputError, ToolRefusalError)`` handler, so a change that
    turned a sync ``ToolInputError`` into a refusal would be invisible to the async case alone.
    That arm does not go away at the merge: #2015's ``power_calculator`` raises ``ToolInputError``
    from its own input guards, and #2016's binary-treatment guard reaches the same handler with a
    ``ToolRefusalError`` — the shared handler stays live either way.

    The message is parametrized because the class must come from the exception TYPE, never from
    what the message happens to say (spec §5.3). Every other input-rejection probe in this repo
    says "declined", so classifying on that word alone would keep all of them green while a real
    ``ToolInputError("invalid sample size")`` was recorded as a refusal.
    """
    from src.agents.tool_composer.errors import ToolInputError

    calls: List[int] = []

    def declining_sync(**_: Any) -> Any:
        calls.append(1)
        raise ToolInputError(message)

    assert not asyncio.iscoroutinefunction(declining_sync), "this case must take the sync arm"

    _register(registry, "power_calculator", declining_sync)
    trace = await _executor(registry).execute(_one("s", "power_calculator"))

    assert _classes(trace.step_results[0]) == ("input_rejected", 1, False, "ToolInputError")
    assert calls == [1], "a retry would run it again; reported attempts alone is not proof"


async def test_plan_defect_class(registry):
    calls: List[int] = []
    _register(registry, "ok_tool", lambda **_: calls.append(1) or {"ok": True})
    plan = _one("s", "ok_tool", input_mapping={"x": "$missing.field"})
    trace = await _executor(registry).execute(plan)
    assert _classes(trace.step_results[0]) == ("plan_defect", 0, False, "ReferenceResolutionError")
    assert calls == []


async def test_dependency_unmet_class(registry):
    calls: List[int] = []
    _register(registry, "gap_calculator", _refusing_gap_calculator)
    _register(registry, "ok_tool", lambda **_: calls.append(1) or {"ok": True})
    plan = _plan(
        [_step("a", "gap_calculator"), _step("b", "ok_tool", depends_on=["a"])], [["a"], ["b"]]
    )
    trace = await _executor(registry).execute(plan)
    assert [r.outcome_class for r in trace.step_results] == ["refused", "dependency_unmet"]
    assert _classes(trace.step_results[1]) == ("dependency_unmet", 0, False, None)
    assert calls == []


async def test_cache_hit_class(registry):
    calls: List[int] = []
    _register(registry, "det_tool", lambda **_: calls.append(1) or {"value": 42})
    executor = _executor(registry, enable_caching=True)
    get_cache_manager().register_deterministic_tool("det_tool")
    plan = _plan([_step("first", "det_tool"), _step("second", "det_tool")], [["first"], ["second"]])
    trace = await executor.execute(plan)
    assert [_classes(r) for r in trace.step_results] == [
        ("succeeded", 1, False, None),
        ("cache_hit", 0, True, None),
    ]
    assert calls == [1]


async def test_not_registered_class(registry):
    trace = await _executor(registry).execute(_one("s", "ghost_tool"))
    assert _classes(trace.step_results[0]) == ("not_registered", 0, False, None)


async def test_sync_timeout_class(registry):
    _register(registry, "slow_sync", lambda **_: time.sleep(1.5) or {"late": True})
    trace = await _executor(registry, timeout_seconds=1).execute(_one("s", "slow_sync"))
    assert _classes(trace.step_results[0]) == ("timeout", 1, False, "SyncToolTimeout")


async def test_async_timeout_class(registry):
    async def slow_async(**_: Any) -> Any:
        await asyncio.sleep(10)
        return {"late": True}

    _register(registry, "slow_async", slow_async)
    executor = _executor(registry, timeout_seconds=1, max_retries=1)
    trace = await executor.execute(_one("s", "slow_async"))
    assert _classes(trace.step_results[0]) == ("timeout", 2, False, "TimeoutError")


async def test_retry_then_success_attempts(registry):
    attempts: List[int] = []

    def flaky(**_: Any) -> Any:
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError("transient")
        return {"ok": True}

    _register(registry, "flaky", flaky)
    trace = await _executor(registry).execute(_one("s", "flaky"))
    assert _classes(trace.step_results[0]) == ("succeeded", 2, False, None)


@pytest.mark.parametrize(
    "raised, expected",
    [
        ([TimeoutError("first"), KeyError("last")], ("error", "KeyError")),
        ([KeyError("first"), TimeoutError("tool's own timeout")], ("timeout", "TimeoutError")),
        ([ValueError("a"), ValueError("b")], ("error", "ValueError")),
    ],
)
async def test_error_class_keeps_last_exception_type(registry, raised, expected):
    remaining = list(raised)

    def failing(**_: Any) -> Any:
        raise remaining.pop(0)

    _register(registry, "failing", failing)
    trace = await _executor(registry, max_retries=1).execute(_one("s", "failing"))
    result = trace.step_results[0]
    assert (result.outcome_class, result.error_type) == expected
    assert result.attempts == 2


async def test_circuit_open_class(registry):
    calls: List[int] = []

    def broken(**_: Any) -> Any:
        calls.append(1)
        raise RuntimeError("down")

    _register(registry, "broken", broken)
    steps = [_step(f"s{i}", "broken") for i in range(4)]
    plan = _plan(steps, [[s.step_id] for s in steps])
    trace = await _executor(registry, max_retries=0).execute(plan)
    assert [r.outcome_class for r in trace.step_results] == [
        "error",
        "error",
        "error",
        "circuit_open",
    ]
    assert _classes(trace.step_results[3]) == ("circuit_open", 0, False, None)
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# The per-step callback
# ---------------------------------------------------------------------------


async def test_callback_called_per_step_single_and_parallel(registry):
    _register(registry, "ok_tool", lambda **_: {"ok": True})
    steps = [_step("solo", "ok_tool"), _step("par_a", "ok_tool"), _step("par_b", "ok_tool")]
    plan = _plan(steps, [["solo"], ["par_a", "par_b"]])
    seen: List[Tuple[int, StepResult]] = []
    trace = await _executor(registry).execute(plan, on_step_result=lambda n, r: seen.append((n, r)))
    assert sorted(n for n, _ in seen) == [0, 1, 2]
    for n, result in seen:
        assert result.step_id == plan.steps[n].step_id
    assert {id(r) for _, r in seen} == {id(r) for r in trace.step_results}


async def test_callback_step_number_is_plan_position_not_completion_order(registry):
    fast_reported = asyncio.Event()

    async def slow(**_: Any) -> Any:
        # Finishes only after the fast step's result was reported: ordering by event, not time.
        await asyncio.wait_for(fast_reported.wait(), timeout=5)
        return {"ok": True}

    async def fast(**_: Any) -> Any:
        return {"ok": True}

    _register(registry, "slow", slow)
    _register(registry, "fast", fast)
    plan = _plan([_step("p0_slow", "slow"), _step("p1_fast", "fast")], [["p0_slow", "p1_fast"]])
    seen: List[Tuple[int, str]] = []

    def record(n: int, result: StepResult) -> None:
        seen.append((n, result.step_id))
        if result.step_id == "p1_fast":
            fast_reported.set()

    await _executor(registry).execute(plan, on_step_result=record)
    assert seen == [(1, "p1_fast"), (0, "p0_slow")]


async def test_failing_callback_never_fails_the_step(registry, caplog):
    _register(registry, "ok_tool", lambda **_: {"ok": True})

    def exploding(n: int, result: StepResult) -> None:
        raise RuntimeError("recorder bug")

    trace = await _executor(registry).execute(_one("s", "ok_tool"), on_step_result=exploding)
    assert trace.tools_succeeded == 1
    assert any("recorder bug" in r.getMessage() for r in caplog.records)


async def test_cancel_keeps_finished_parallel_sibling(registry):
    fast_done = asyncio.Event()

    async def fast(**_: Any) -> Any:
        return {"ok": True}

    async def slow(**_: Any) -> Any:
        await asyncio.sleep(30)
        return {"late": True}

    _register(registry, "fast", fast)
    _register(registry, "slow", slow)
    plan = _plan(
        [_step("fast_step", "fast"), _step("slow_step", "slow")], [["fast_step", "slow_step"]]
    )
    seen: List[int] = []

    def record(n: int, result: StepResult) -> None:
        seen.append(n)
        if result.step_id == "fast_step":
            fast_done.set()

    task = asyncio.create_task(_executor(registry).execute(plan, on_step_result=record))
    await asyncio.wait_for(fast_done.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == [0]


class _Exploding:
    """A producer output whose attribute access raises something other than a resolution error."""

    @property
    def edge_list(self) -> Any:
        raise TypeError("not subscriptable here")


async def test_escaping_exception_keeps_finished_sibling(registry):
    async def producer(**_: Any) -> Any:
        return {"graph": _Exploding()}

    async def sibling(**_: Any) -> Any:
        return {"ok": True}

    _register(registry, "producer", producer)
    _register(registry, "sibling", sibling)
    _register(registry, "consumer", sibling)
    plan = _plan(
        [
            _step("p", "producer"),
            _step("a_first", "sibling"),
            _step(
                "b_second", "consumer", depends_on=["p"], input_mapping={"g": "$p.graph.edge_list"}
            ),
        ],
        [["p"], ["a_first", "b_second"]],
    )
    seen: List[int] = []
    # One slot: a_first (listed first) runs to completion before b_second starts, so the
    # ordering is deterministic, not timing-based.
    executor = _executor(registry, max_parallel=1)
    with pytest.raises(ExecutionError):
        await executor.execute(plan, on_step_result=lambda n, r: seen.append(n))
    assert seen == [0, 1]
