"""Red-first pins for #1600 — deterministic tool refusals must not be retried.

``PlanExecutor._execute_step``'s retry loop special-cases ``ToolInputError``
(#1573) and ``SyncToolTimeout`` (#1592); everything else falls into the generic
``except Exception`` arm and is re-dispatched ``max_retries`` more times. The
``tool_registrations`` fail-closed guards raise descriptive ``RuntimeError``s
that are fully deterministic over the step's resolved inputs, so those extra
dispatches re-run the same compute over the same frame and fail with the same
message — live-corroborated as 3 identical ``gap_calculator`` refusal attempts
in the 2026-08-14 forced q08 replay. Since #1598 they also serialize behind the
one-slot heavy-compute pool.

What these tests pin:

* A deterministic guard refusal is attempted EXACTLY ONCE (pre-fix: 1 + 2).
* A genuinely transient failure (a plain ``RuntimeError`` from a tool body that
  is not a deterministic guard) still retries — the fix must not convert
  "non-retryable" into "never retry anything".
* The refusal's own message reaches ``ToolOutput.error`` VERBATIM. #1574's
  ``estimation_data_scope`` disclosure rides that string into synthesis, and
  the composer truncates from the END, so a prefix is not free.
* A refusal is NOT recorded against the circuit breaker, for the same reason
  ``ToolInputError`` is not: a plan/data defect says nothing about the tool's
  health, and must not open the circuit for other valid steps using that tool.
* ``ToolInputError``'s existing behavior (fail once, "input contract
  violation: " prefix) is unchanged.

The refusals under test come from REAL production guard code — the step's tool
calls the real ``tool_registrations.gap_calculator`` over a real DataFrame.
Only the invocation COUNTING is test scaffolding.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr
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
    """Sync tools dispatch through the #1592 bounded pool; keep it per-test."""
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
            description="1600 probe tool.",
            source_agent="gap_analyzer",
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
        original_query="1600?",
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
        source_agent="gap_analyzer",
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
                source_agent="gap_analyzer",
                confidence=0.9,
                reasoning="t",
            )
        ],
        estimated_duration_ms=10,
        parallel_groups=[["step_1"]],
        planning_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )


def _executor(registry: ToolRegistry) -> PlanExecutor:
    """max_retries=2 (the production default); zero backoff so tests are fast."""
    return PlanExecutor(
        tool_registry=registry,
        max_retries=2,
        backoff_base_delay=0.0,
        backoff_max_delay=0.0,
        enable_caching=False,
    )


def _single_brand_frame() -> pd.DataFrame:
    """The #1574 shape — one distinct brand group, so gap_calculator refuses."""
    return pd.DataFrame(
        {
            "brand": ["Kisqali"] * 6,
            "market_share": [0.72, 0.71, 0.74, 0.73, 0.70, 0.72],
        }
    )


# ---------------------------------------------------------------------------
# (d) a deterministic guard refusal is attempted EXACTLY ONCE
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deterministic_guard_refusal_is_attempted_once():
    """Pre-fix: 3 attempts (1 + max_retries), each re-running the same compute."""
    attempts: List[int] = []

    def refusing_tool(**kwargs: Any) -> Any:
        attempts.append(1)
        # REAL guard: a single-brand frame cannot support a gap comparison.
        return tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=["Kisqali", "competitor"],
            estimation_data=_single_brand_frame(),
        )

    executor = _executor(_registry_with("gap_calculator", refusing_tool))
    trace = await executor.execute(_single_step_plan("gap_calculator"))

    assert len(attempts) == 1, f"deterministic refusal re-dispatched {len(attempts)} times"
    assert trace.step_results[0].status == ExecutionStatus.FAILED
    assert trace.step_results[0].output.success is False


@pytest.mark.asyncio
async def test_refusal_reason_reaches_tool_output_verbatim():
    """#1574's estimation_data_scope rides this string into synthesis."""
    reason_seen: List[str] = []

    def refusing_tool(**kwargs: Any) -> Any:
        try:
            return tr.gap_calculator(
                metric="market_share",
                entity_type="brand",
                entities=["Kisqali", "competitor"],
                estimation_data=_single_brand_frame(),
            )
        except RuntimeError as exc:
            reason_seen.append(str(exc))
            raise

    executor = _executor(_registry_with("gap_calculator", refusing_tool))
    trace = await executor.execute(_single_step_plan("gap_calculator"))

    error = trace.step_results[0].output.error
    assert error == reason_seen[0], "the tool's own reason must not be rewritten"
    assert "estimation_data_scope=" in error
    assert not error.startswith("input contract violation")


@pytest.mark.asyncio
async def test_refusal_is_not_recorded_against_the_circuit_breaker():
    """A plan/data defect is not a tool-health signal (same call as #1573)."""

    def refusing_tool(**kwargs: Any) -> Any:
        return tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=["Kisqali", "competitor"],
            estimation_data=_single_brand_frame(),
        )

    executor = _executor(_registry_with("gap_calculator", refusing_tool))
    await executor.execute(_single_step_plan("gap_calculator"))

    stats = executor.failure_tracker._stats.get("gap_calculator")
    assert stats is None or stats.total_failures == 0
    assert executor.failure_tracker.can_execute("gap_calculator") is True


# ---------------------------------------------------------------------------
# (e) PRESERVED — a genuinely transient failure still retries
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_transient_runtime_error_still_retries():
    """A plain RuntimeError from a tool body is NOT a deterministic refusal."""
    attempts: List[int] = []

    def flaky_tool(**kwargs: Any) -> Any:
        attempts.append(1)
        raise RuntimeError("upstream service connection reset")

    executor = _executor(_registry_with("flaky", flaky_tool))
    trace = await executor.execute(_single_step_plan("flaky"))

    assert len(attempts) == 3, f"transient failure must retry; got {len(attempts)} attempts"
    assert trace.step_results[0].status == ExecutionStatus.FAILED
    assert executor.failure_tracker._stats["flaky"].total_failures == 1


@pytest.mark.asyncio
async def test_transient_runtime_error_that_recovers_is_reported_successful():
    """The retry path must still be able to succeed on a later attempt."""
    attempts: List[int] = []

    def recovering_tool(**kwargs: Any) -> Any:
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("transient: connection reset")
        return {"ok": True}

    executor = _executor(_registry_with("recovering", recovering_tool))
    trace = await executor.execute(_single_step_plan("recovering"))

    assert len(attempts) == 2
    assert trace.step_results[0].status == ExecutionStatus.COMPLETED
    assert trace.step_results[0].output.result == {"ok": True}


# ---------------------------------------------------------------------------
# (f) PRESERVED — ToolInputError keeps its #1573 behavior exactly
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_tool_input_error_still_fails_once_with_its_prefix():
    from src.agents.tool_composer.errors import ToolInputError

    attempts: List[int] = []

    def declining_tool(**kwargs: Any) -> Any:
        attempts.append(1)
        raise ToolInputError("declined: expected_effect is None — cannot simulate a lift")

    executor = _executor(_registry_with("counterfactual_simulator", declining_tool))
    trace = await executor.execute(_single_step_plan("counterfactual_simulator"))

    assert len(attempts) == 1
    error = trace.step_results[0].output.error
    assert error.startswith("input contract violation: ")
    assert "expected_effect is None" in error
    stats = executor.failure_tracker._stats.get("counterfactual_simulator")
    assert stats is None or stats.total_failures == 0


# ---------------------------------------------------------------------------
# Type contract for the new exception
# ---------------------------------------------------------------------------
def test_tool_refusal_error_is_a_runtime_error():
    """Subclassing RuntimeError keeps every documented fail-closed contract
    (and every existing ``pytest.raises(RuntimeError)`` pin) truthful."""
    from src.agents.tool_composer.errors import ToolInputError, ToolRefusalError

    assert issubclass(ToolRefusalError, RuntimeError)
    # Distinct concepts: an input-contract violation vs a refusal to produce a
    # result from inputs that are structurally fine but not analyzable.
    assert not issubclass(ToolRefusalError, ToolInputError)
    assert not issubclass(ToolInputError, ToolRefusalError)


@pytest.mark.asyncio
async def test_explicit_tool_refusal_error_is_attempted_once():
    from src.agents.tool_composer.errors import ToolRefusalError

    attempts: List[int] = []

    def refusing_tool(**kwargs: Any) -> Any:
        attempts.append(1)
        raise ToolRefusalError("probe_tool: refusing — deterministic over inputs.")

    executor = _executor(_registry_with("probe_tool", refusing_tool))
    trace = await executor.execute(_single_step_plan("probe_tool"))

    assert len(attempts) == 1
    assert trace.step_results[0].output.error == (
        "probe_tool: refusing — deterministic over inputs."
    )
