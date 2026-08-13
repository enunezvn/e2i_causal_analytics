"""#1592 — SYNC composer tools must run on the BOUNDED heavy-compute pool,
under a timeout envelope, and must NOT be re-dispatched after that timeout.

Before this fix ``PlanExecutor._execute_step`` dispatched sync tools with

    await asyncio.get_event_loop().run_in_executor(None, lambda: tool(**inputs))

which has two defects:

1. ``None`` = the loop's DEFAULT executor (``min(32, cpu+4)`` threads; 12 on
   the prod box) — it bypasses the bounded heavy-compute pool that exists
   precisely to keep concurrent in-process compute inside the api container's
   5G cgroup (``src/api/dependencies/compute.py``).
2. No timeout at all — the async branch immediately above wraps its call in
   ``asyncio.wait_for(..., timeout=self.timeout_seconds)``; the sync branch
   had none, so a hung sync tool pinned its thread and its plan step forever.

Residual (pinned by ``test_timed_out_sync_tool_is_not_re_dispatched``): a
``wait_for`` around a thread future cancels the FUTURE, never the thread. The
abandoned thread runs to completion holding a bounded-pool slot, so re-running
the same compute on retry is strictly harmful — the sync branch must fail the
step once instead.

Falsifiability: each test names the exact pre-fix behavior it would observe.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.agents.tool_composer.executor import PlanExecutor
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    SubQuestion,
    ToolMapping,
)
from src.tool_registry.registry import ToolRegistry, ToolSchema


@pytest.fixture(autouse=True)
def _clean_bounded_pool():
    """Start every case with a fresh limiter/executor bound to this test's loop."""
    from src.api.dependencies.compute import _reset_limiter_cache_for_tests

    _reset_limiter_cache_for_tests()
    yield
    _reset_limiter_cache_for_tests()


def _registry_with(name: str, fn: Any) -> ToolRegistry:
    registry = ToolRegistry()
    registry.clear()
    registry.register(
        schema=ToolSchema(
            name=name,
            description="1592 probe tool.",
            source_agent="causal_impact",
            tier=2,
            input_parameters=[],
            output_schema="Dict[str, Any]",
            avg_execution_ms=10,
        ),
        callable=fn,
    )
    return registry


def _single_step_plan(tool_name: str, input_mapping: Optional[Dict[str, Any]] = None):
    decomposition = DecompositionResult(
        original_query="1592?",
        sub_questions=[
            SubQuestion(id="sq_1", question="q1", intent="CAUSAL", entities=[], depends_on=[]),
        ],
        decomposition_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )
    step = ExecutionStep(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name=tool_name,
        source_agent="causal_impact",
        input_mapping=input_mapping or {},
        dependency_type=DependencyType.SEQUENTIAL,
        depends_on_steps=[],
    )
    return ExecutionPlan(
        decomposition=decomposition,
        steps=[step],
        tool_mappings=[
            ToolMapping(
                sub_question_id="sq_1",
                tool_name=tool_name,
                source_agent="causal_impact",
                confidence=0.9,
                reasoning="t",
            )
        ],
        estimated_duration_ms=10,
        parallel_groups=[["step_1"]],
        planning_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# (a) sync dispatch goes through the bounded heavy-compute pool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_tool_dispatches_through_bounded_executor(monkeypatch) -> None:
    """Falsifiability: pre-fix the sync branch called
    ``loop.run_in_executor(None, ...)`` directly and NEVER touched
    ``run_in_bounded_executor``, so ``calls`` stays empty."""
    import src.api.dependencies.compute as compute

    calls: List[str] = []
    real = compute.run_in_bounded_executor

    async def spy(func, *args, **kwargs):
        calls.append("bounded")
        return await real(func, *args, **kwargs)

    monkeypatch.setattr(compute, "run_in_bounded_executor", spy)

    registry = _registry_with("sync_probe", lambda **kw: {"ok": True})
    executor = PlanExecutor(tool_registry=registry, enable_caching=False, max_retries=0)

    trace = await executor.execute(_single_step_plan("sync_probe"), context={})

    assert trace.get_result("step_1").status == ExecutionStatus.COMPLETED
    assert calls == ["bounded"], (
        "sync tool must be dispatched via run_in_bounded_executor; "
        f"observed bounded-pool calls={calls}"
    )


@pytest.mark.asyncio
async def test_sync_tool_does_not_use_the_loop_default_executor() -> None:
    """Independent pin of the same defect, at the loop seam rather than the
    import seam. Falsifiability: pre-fix the recorded executor argument is
    ``None`` (the loop's default pool); post-fix it is the shared bounded
    ``heavy-compute`` ThreadPoolExecutor."""
    loop = asyncio.get_running_loop()
    seen: List[Any] = []
    real_run_in_executor = loop.run_in_executor

    def recording_run_in_executor(executor, func, *args):
        seen.append(executor)
        return real_run_in_executor(executor, func, *args)

    loop.run_in_executor = recording_run_in_executor  # type: ignore[method-assign]
    try:
        registry = _registry_with("sync_probe", lambda **kw: {"ok": True})
        executor = PlanExecutor(tool_registry=registry, enable_caching=False, max_retries=0)
        trace = await executor.execute(_single_step_plan("sync_probe"), context={})
    finally:
        loop.run_in_executor = real_run_in_executor  # type: ignore[method-assign]

    assert trace.get_result("step_1").status == ExecutionStatus.COMPLETED
    assert seen, "the sync tool never reached run_in_executor at all"
    assert all(e is not None for e in seen), (
        f"sync tools must not run on the loop's DEFAULT executor (recorded executors={seen})"
    )
    pools = [e for e in seen if isinstance(e, ThreadPoolExecutor)]
    assert pools and all(getattr(p, "_thread_name_prefix", "") == "heavy-compute" for p in pools), (
        f"expected the shared bounded 'heavy-compute' pool; got {pools}"
    )


# ---------------------------------------------------------------------------
# (b) the timeout envelope + its honest post-timeout semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slow_sync_tool_trips_the_timeout_envelope() -> None:
    """Falsifiability: pre-fix the sync branch had NO timeout, so this step
    blocks until the stub returns (i.e. the test hangs / only completes when
    the release event fires) instead of failing at ``timeout_seconds``."""
    released = threading.Event()
    started = threading.Event()

    def slow_tool(**_kwargs: Any) -> Dict[str, Any]:
        started.set()
        released.wait(timeout=30)
        return {"ok": True}

    registry = _registry_with("slow_sync_probe", slow_tool)
    executor = PlanExecutor(
        tool_registry=registry,
        enable_caching=False,
        max_retries=0,
        timeout_seconds=0.25,
    )

    try:
        trace = await asyncio.wait_for(
            executor.execute(_single_step_plan("slow_sync_probe"), context={}),
            timeout=10,
        )
    finally:
        released.set()

    result = trace.get_result("step_1")
    assert started.is_set(), "the stub tool never ran"
    assert result.status == ExecutionStatus.FAILED
    assert result.output.is_success is False
    assert "timed out" in (result.output.error or "").lower(), (
        f"expected an honest timeout error; got {result.output.error!r}"
    )


@pytest.mark.asyncio
async def test_timed_out_sync_tool_is_not_re_dispatched() -> None:
    """A ``wait_for`` around a thread future cancels the FUTURE, not the
    thread: the first call is still burning a bounded-pool slot. Retrying
    stacks a second copy of the same compute behind it and cannot finish any
    sooner, so the step must fail once.

    Falsifiability: with the default retry loop applied to timeouts the stub
    is invoked ``max_retries + 1`` == 3 times.
    """
    released = threading.Event()
    invocations: List[int] = []
    lock = threading.Lock()

    def slow_tool(**_kwargs: Any) -> Dict[str, Any]:
        with lock:
            invocations.append(1)
        released.wait(timeout=30)
        return {"ok": True}

    registry = _registry_with("slow_sync_probe", slow_tool)
    executor = PlanExecutor(
        tool_registry=registry,
        enable_caching=False,
        max_retries=2,
        backoff_base_delay=0.01,
        backoff_max_delay=0.02,
        timeout_seconds=0.25,
    )

    try:
        trace = await asyncio.wait_for(
            executor.execute(_single_step_plan("slow_sync_probe"), context={}),
            timeout=10,
        )
    finally:
        released.set()

    assert trace.get_result("step_1").status == ExecutionStatus.FAILED
    assert len(invocations) == 1, (
        "a timed-out sync tool must NOT be re-dispatched while its abandoned "
        f"thread still holds a pool slot; invoked {len(invocations)} times"
    )


# ---------------------------------------------------------------------------
# (c) the async branch is untouched
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_tool_never_touches_the_bounded_pool(monkeypatch) -> None:
    """Async tools already run on the loop under ``wait_for``; the fix must not
    move them into the thread pool."""
    import src.api.dependencies.compute as compute

    calls: List[str] = []
    real = compute.run_in_bounded_executor

    async def spy(func, *args, **kwargs):
        calls.append("bounded")
        return await real(func, *args, **kwargs)

    monkeypatch.setattr(compute, "run_in_bounded_executor", spy)

    async def async_tool(**_kwargs: Any) -> Dict[str, Any]:
        return {"ok": "async"}

    registry = _registry_with("async_probe", async_tool)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False, max_retries=0)

    trace = await executor.execute(_single_step_plan("async_probe"), context={})

    result = trace.get_result("step_1")
    assert result.status == ExecutionStatus.COMPLETED
    assert result.output.result == {"ok": "async"}
    assert calls == [], "async tools must stay on the loop, not the bounded pool"


@pytest.mark.asyncio
async def test_async_tool_timeout_still_retries() -> None:
    """The async branch's established semantics are unchanged: ``wait_for``
    CANCELS the coroutine, so a retry starts from a clean slate and the
    existing retry loop still applies (``max_retries + 1`` invocations)."""
    invocations: List[int] = []

    async def slow_async_tool(**_kwargs: Any) -> Dict[str, Any]:
        invocations.append(1)
        await asyncio.sleep(5)
        return {"ok": True}

    registry = _registry_with("slow_async_probe", slow_async_tool)
    executor = PlanExecutor(
        tool_registry=registry,
        enable_caching=False,
        max_retries=1,
        backoff_base_delay=0.01,
        backoff_max_delay=0.02,
        timeout_seconds=0.25,
    )

    trace = await asyncio.wait_for(
        executor.execute(_single_step_plan("slow_async_probe"), context={}), timeout=10
    )

    assert trace.get_result("step_1").status == ExecutionStatus.FAILED
    assert len(invocations) == 2, (
        "async-branch retry behavior must be byte-identical to pre-fix; "
        f"invoked {len(invocations)} times"
    )


# ---------------------------------------------------------------------------
# (d) a fast sync tool still produces identical results through the new path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fast_sync_tool_result_is_unchanged() -> None:
    """The bounded pool must be a transport change only: same kwargs in, same
    result dict out, step COMPLETED."""
    seen_kwargs: Dict[str, Any] = {}

    def fast_tool(**kwargs: Any) -> Dict[str, Any]:
        seen_kwargs.update(kwargs)
        return {"effect": 0.15, "ci_lower": 0.12, "note": kwargs.get("metric")}

    registry = _registry_with("fast_sync_probe", fast_tool)
    executor = PlanExecutor(tool_registry=registry, enable_caching=False, max_retries=0)

    trace = await executor.execute(
        _single_step_plan("fast_sync_probe", {"metric": "rx_volume"}), context={}
    )

    result = trace.get_result("step_1")
    assert result.status == ExecutionStatus.COMPLETED
    assert result.output.result == {"effect": 0.15, "ci_lower": 0.12, "note": "rx_volume"}
    assert seen_kwargs["metric"] == "rx_volume"
    assert result.duration_ms is not None and result.duration_ms >= 0


@pytest.mark.asyncio
async def test_sync_tool_raising_is_still_retried_and_failed() -> None:
    """Ordinary sync-tool exceptions keep their existing retry semantics — the
    fix must special-case ONLY the timeout."""
    invocations: List[int] = []

    def boom(**_kwargs: Any) -> Dict[str, Any]:
        invocations.append(1)
        raise RuntimeError("sync boom")

    registry = _registry_with("boom_probe", boom)
    executor = PlanExecutor(
        tool_registry=registry,
        enable_caching=False,
        max_retries=1,
        backoff_base_delay=0.01,
        backoff_max_delay=0.02,
    )

    trace = await executor.execute(_single_step_plan("boom_probe"), context={})

    result = trace.get_result("step_1")
    assert result.status == ExecutionStatus.FAILED
    assert "sync boom" in (result.output.error or "")
    assert len(invocations) == 2
