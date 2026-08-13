"""
E2I Tool Composer - Phase 3: Executor
Version: 4.3
Purpose: Execute tool chains according to the execution plan

Implements:
- Exponential backoff retry strategy
- Circuit breaker pattern for failing tools
- Per-tool failure tracking
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from src.data.adaptive_validity_repository import query_active_role_attributions
from src.data.role_attribution import RoleAttribution, should_act
from src.tool_registry.registry import ToolRegistry

# Import tool_registrations to ensure tools are registered before execution
# The @composable_tool decorators register tools when the module is imported
from . import tool_registrations as _tool_registrations  # noqa: F401
from .cache import get_cache_manager
from .errors import ReferenceResolutionError, ToolInputError
from .models.composition_models import (
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    ExecutionTrace,
    StepResult,
    ToolInput,
    ToolOutput,
)
from .tool_registrations import _DATAFRAME_KWARGS_KEYS

logger = logging.getLogger(__name__)


# ============================================================================
# EXPONENTIAL BACKOFF
# ============================================================================


@dataclass
class ExponentialBackoff:
    """
    Exponential backoff strategy for retries.

    Delay calculation: min(max_delay, base_delay * (factor ** attempt))
    With optional jitter to prevent thundering herd.
    """

    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    factor: float = 2.0
    jitter: float = 0.1  # Random variation factor (0-1)

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt (0-indexed)."""
        import random

        delay = min(self.max_delay, self.base_delay * (self.factor**attempt))

        # Add jitter to prevent thundering herd
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)  # Never negative


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================


class CircuitState(str, Enum):
    """State of the circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    When a tool fails repeatedly, the circuit opens and blocks further
    calls until a reset timeout has passed, at which point it enters
    half-open state to test recovery.
    """

    failure_threshold: int = 3  # Failures before opening
    reset_timeout: float = 60.0  # Seconds before half-open
    half_open_max_calls: int = 1  # Max calls in half-open state

    # Internal state
    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    success_count: int = field(default=0)
    last_failure_time: Optional[float] = field(default=None)
    half_open_calls: int = field(default=0)

    def can_execute(self) -> bool:
        """Check if the circuit allows execution."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if self.last_failure_time is not None:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.reset_timeout:
                    # Transition to half-open
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit breaker entering HALF_OPEN state after {elapsed:.1f}s")
                    return True
            return False

        # Half-open: allow limited calls to test recovery
        return self.half_open_calls < self.half_open_max_calls

    def record_success(self) -> None:
        """Record a successful execution."""
        self.success_count += 1

        if self.state == CircuitState.HALF_OPEN:
            # Service recovered, close the circuit
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
            logger.info("Circuit breaker CLOSED - service recovered")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success (sliding window would be better for prod)
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery test, reopen circuit
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker re-OPENED - recovery test failed")
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")

    def get_state_info(self) -> Dict[str, Any]:
        """Get current circuit breaker state for observability."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }


# ============================================================================
# TOOL FAILURE TRACKER
# ============================================================================


@dataclass
class ToolFailureStats:
    """
    Statistics for a single tool's execution history.

    Implements:
    - Exponential moving average (EMA) for latency tracking (G8)
    - Sliding window for recent success rate calculation (G8)
    """

    total_calls: int = 0
    total_failures: int = 0
    total_successes: int = 0
    total_latency_ms: int = 0
    last_failure_reason: Optional[str] = None
    last_success_time: Optional[float] = None
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    # G8: Exponential moving average for latency
    ema_latency_ms: float = 0.0
    ema_alpha: float = 0.2  # Weight for new observations (0.2 = responsive)

    # G8: Sliding window for recent success rate
    recent_results: List[bool] = field(default_factory=list)
    sliding_window_size: int = 50  # Track last 50 calls

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate as a percentage."""
        if self.total_calls == 0:
            return 1.0  # Assume success if never called
        return self.total_successes / self.total_calls

    @property
    def recent_success_rate(self) -> float:
        """Calculate success rate for recent calls in sliding window (G8)."""
        if not self.recent_results:
            return 1.0  # Assume success if no recent calls
        return sum(1 for r in self.recent_results if r) / len(self.recent_results)

    @property
    def avg_latency_ms(self) -> float:
        """Calculate simple average latency in milliseconds."""
        if self.total_successes == 0:
            return 0.0
        return self.total_latency_ms / self.total_successes

    def update_ema_latency(self, latency_ms: int) -> None:
        """Update exponential moving average latency (G8)."""
        if self.ema_latency_ms == 0.0:
            # First observation
            self.ema_latency_ms = float(latency_ms)
        else:
            # EMA formula: new_ema = alpha * new_value + (1 - alpha) * old_ema
            self.ema_latency_ms = (
                self.ema_alpha * latency_ms + (1 - self.ema_alpha) * self.ema_latency_ms
            )

    def record_result(self, success: bool) -> None:
        """Record a result in the sliding window (G8)."""
        self.recent_results.append(success)
        # Trim to window size
        if len(self.recent_results) > self.sliding_window_size:
            self.recent_results = self.recent_results[-self.sliding_window_size :]


class ToolFailureTracker:
    """
    Tracks failure statistics and circuit breaker state per tool.

    Provides centralized tracking for:
    - Per-tool circuit breakers
    - Failure/success rates
    - Latency statistics
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._stats: Dict[str, ToolFailureStats] = {}

    def _get_or_create_stats(self, tool_name: str) -> ToolFailureStats:
        """Get or create stats for a tool."""
        if tool_name not in self._stats:
            self._stats[tool_name] = ToolFailureStats(
                circuit_breaker=CircuitBreaker(
                    failure_threshold=self.failure_threshold,
                    reset_timeout=self.reset_timeout,
                )
            )
        return self._stats[tool_name]

    def can_execute(self, tool_name: str) -> bool:
        """Check if tool's circuit breaker allows execution."""
        stats = self._get_or_create_stats(tool_name)
        return stats.circuit_breaker.can_execute()

    def record_success(self, tool_name: str, latency_ms: int) -> None:
        """Record a successful tool execution."""
        stats = self._get_or_create_stats(tool_name)
        stats.total_calls += 1
        stats.total_successes += 1
        stats.total_latency_ms += latency_ms
        stats.last_success_time = time.time()
        stats.circuit_breaker.record_success()

        # G8: Update performance learning metrics
        stats.update_ema_latency(latency_ms)
        stats.record_result(success=True)

    def record_failure(self, tool_name: str, reason: str) -> None:
        """Record a failed tool execution."""
        stats = self._get_or_create_stats(tool_name)
        stats.total_calls += 1
        stats.total_failures += 1
        stats.last_failure_reason = reason
        stats.circuit_breaker.record_failure()

        # G8: Update sliding window for recent success rate
        stats.record_result(success=False)

    def get_stats(self, tool_name: str) -> Optional[ToolFailureStats]:
        """Get stats for a specific tool."""
        return self._stats.get(tool_name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all tracked tools."""
        return {
            name: {
                "total_calls": stats.total_calls,
                "success_rate": stats.success_rate,
                "avg_latency_ms": stats.avg_latency_ms,
                # G8: Include performance learning metrics
                "ema_latency_ms": stats.ema_latency_ms,
                "recent_success_rate": stats.recent_success_rate,
                "circuit_breaker": stats.circuit_breaker.get_state_info(),
            }
            for name, stats in self._stats.items()
        }

    def reset(self, tool_name: Optional[str] = None) -> None:
        """Reset stats for a tool or all tools."""
        if tool_name:
            self._stats.pop(tool_name, None)
        else:
            self._stats.clear()

    def get_circuit_breaker_summary(self) -> Dict[str, Any]:
        """Get circuit breaker summary for observability (V4.3).

        Returns a summary suitable for Opik tracing with:
        - Total circuit trips across all tools
        - Number of currently open circuits
        - List of open/half-open circuits
        - Per-tool circuit state
        """
        total_trips = 0
        open_circuits = []
        half_open_circuits = []
        per_tool_state = {}

        for tool_name, stats in self._stats.items():
            cb = stats.circuit_breaker
            state_info = cb.get_state_info()
            per_tool_state[tool_name] = state_info

            # Count trips (approximated by failure count when opened)
            if state_info["state"] in ("open", "half_open"):
                total_trips += 1
                if state_info["state"] == "open":
                    open_circuits.append(tool_name)
                else:
                    half_open_circuits.append(tool_name)

        return {
            "total_trips": total_trips,
            "open_circuits": len(open_circuits),
            "half_open_circuits": len(half_open_circuits),
            "open_circuit_tools": open_circuits,
            "half_open_circuit_tools": half_open_circuits,
            "per_tool_state": per_tool_state,
        }


# ============================================================================
# EXECUTOR CLASS
# ============================================================================


class PlanExecutor:
    """
    Executes tool chains according to the execution plan.

    This is Phase 3 of the Tool Composer pipeline.

    Features:
    - Executes tools in dependency order
    - Supports parallel execution of independent tools
    - Passes outputs from prior steps as inputs to dependent steps
    - Handles retries with exponential backoff
    - Circuit breaker pattern for failing tools
    - Per-tool failure tracking and statistics
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        max_parallel: int = 3,
        max_retries: int = 2,
        timeout_seconds: int = 120,
        # Exponential backoff configuration
        backoff_base_delay: float = 1.0,
        backoff_max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        # Circuit breaker configuration
        circuit_failure_threshold: int = 3,
        circuit_reset_timeout: float = 60.0,
        # Caching configuration
        enable_caching: bool = True,
    ):
        self.registry = tool_registry or ToolRegistry()
        self.max_parallel = max_parallel
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.enable_caching = enable_caching

        # Initialize exponential backoff strategy
        self.backoff = ExponentialBackoff(
            base_delay=backoff_base_delay,
            max_delay=backoff_max_delay,
            factor=backoff_factor,
        )

        # Initialize per-tool failure tracker with circuit breakers
        self.failure_tracker = ToolFailureTracker(
            failure_threshold=circuit_failure_threshold,
            reset_timeout=circuit_reset_timeout,
        )

        # Initialize cache manager for deterministic tool output caching (G6)
        self._cache_manager = get_cache_manager() if enable_caching else None

    async def execute(
        self, plan: ExecutionPlan, context: Optional[Dict[str, Any]] = None
    ) -> ExecutionTrace:
        """
        Execute the plan and return a trace of all executions.

        Args:
            plan: The execution plan from Phase 2
            context: Optional additional context (e.g., data, filters)

        Returns:
            ExecutionTrace with all step results
        """
        logger.info(f"Executing plan {plan.plan_id} with {plan.step_count} steps")

        trace = ExecutionTrace(plan_id=plan.plan_id, started_at=datetime.now(timezone.utc))

        # Store outputs for dependency resolution
        outputs: Dict[str, Any] = {}
        # F5: track step_ids that FAILED (or were skipped) so dependents can
        # be short-circuited instead of crashing on a missing upstream output.
        failed_step_ids: set[str] = set()
        context = context or {}

        try:
            # Get execution order (groups of parallel steps)
            execution_groups = plan.get_execution_order()

            for group_idx, group in enumerate(execution_groups):
                logger.info(f"Executing group {group_idx + 1}/{len(execution_groups)}: {group}")

                # Execute steps in this group (potentially in parallel)
                if len(group) == 1:
                    # Single step, execute directly
                    step = plan.get_step(group[0])
                    if step:
                        result = await self._execute_step(step, outputs, context, failed_step_ids)
                        trace.add_result(result)
                        if result.output.is_success:
                            outputs[step.step_id] = result.output.result
                        else:
                            failed_step_ids.add(step.step_id)
                else:
                    # Multiple steps, execute in parallel
                    results = await self._execute_parallel(
                        [step for sid in group if (step := plan.get_step(sid)) is not None],
                        outputs,
                        context,
                        failed_step_ids,
                    )
                    for result in results:
                        trace.add_result(result)
                        if result.output.is_success:
                            outputs[result.step_id] = result.output.result
                        else:
                            failed_step_ids.add(result.step_id)
                    trace.parallel_executions += 1

            trace.completed_at = datetime.now(timezone.utc)
            logger.info(
                f"Execution complete: {trace.tools_succeeded}/{trace.tools_executed} succeeded"
            )

        except asyncio.TimeoutError:
            logger.error(f"Execution timed out after {self.timeout_seconds}s")
            trace.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            trace.completed_at = datetime.now(timezone.utc)
            raise ExecutionError(f"Plan execution failed: {e}") from e

        return trace

    async def _execute_step(
        self,
        step: ExecutionStep,
        prior_outputs: Dict[str, Any],
        context: Dict[str, Any],
        failed_step_ids: Optional[set[str]] = None,
    ) -> StepResult:
        """Execute a single step with circuit breaker and exponential backoff.

        F5: if any of this step's ``depends_on_steps`` is in
        ``failed_step_ids`` (an upstream step that already failed), the step
        is SKIPPED — its tool is NOT invoked — and a clear dependency-unmet
        error is recorded. This prevents dependents from crashing on a None
        upstream output (which never lands in ``prior_outputs``).
        """
        started_at = datetime.now(timezone.utc)

        logger.debug(f"Executing step {step.step_id}: {step.tool_name}")

        # F5: short-circuit dependents of failed upstream steps.
        unmet = sorted(set(step.depends_on_steps) & (failed_step_ids or set()))
        if unmet:
            logger.warning(
                f"Skipping step {step.step_id} ({step.tool_name}): "
                f"dependency unmet: {', '.join(unmet)}"
            )
            completed_at = datetime.now(timezone.utc)
            return StepResult(
                step_id=step.step_id,
                sub_question_id=step.sub_question_id,
                tool_name=step.tool_name,
                input=ToolInput(tool_name=step.tool_name, parameters={}, context=context),
                output=ToolOutput(
                    tool_name=step.tool_name,
                    success=False,
                    error=f"dependency unmet: {', '.join(unmet)}",
                ),
                status=ExecutionStatus.SKIPPED,
                started_at=started_at,
                completed_at=completed_at,
            )

        # Resolve input parameters.
        # #1573: an unresolvable TOP-LEVEL reference is a plan defect that
        # dooms this step deterministically — fail fast with an explicit
        # reason (which reaches synthesis via StepResult.output.error), never
        # invoke the tool, and never retry. The tool's circuit breaker is NOT
        # penalized: the tool never ran, and opening a healthy tool's circuit
        # over a planner defect would block other, valid steps that use the
        # same tool.
        try:
            resolved_inputs = self._resolve_inputs(step.input_mapping, prior_outputs, context)
        except ReferenceResolutionError as e:
            logger.warning(f"Step {step.step_id} ({step.tool_name}) failed before execution: {e}")
            return StepResult(
                step_id=step.step_id,
                sub_question_id=step.sub_question_id,
                tool_name=step.tool_name,
                input=ToolInput(tool_name=step.tool_name, parameters={}, context=context),
                output=ToolOutput(
                    tool_name=step.tool_name,
                    success=False,
                    error=f"unresolvable reference: {e}",
                ),
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

        # Phase 7.2 + S14: tool composer auto-population hook
        # ===========================================================
        # When the planned tool accepts a ``confounders`` parameter AND
        # the caller did NOT supply an explicit value AND the context
        # carries an ``experiment_id``, pre-fill ``confounders`` from
        # the adaptive_validity_verdicts mirror. The C1 trust-gate
        # (``should_act``) is rechecked at the consumer boundary for
        # defense-in-depth.
        #
        # Plan: .claude/plans/causal_role_propagation_FINAL.md §7.2.
        # ===========================================================
        autopop_confounders = self._maybe_autopopulate_confounders(step, resolved_inputs, context)
        if autopop_confounders is not None:
            # Build a new params dict rather than mutating caller's resolved_inputs
            # (codex audit PR #367 INVARIANT 8 — preserve caller's dict identity).
            resolved_inputs = {**resolved_inputs, "confounders": autopop_confounders}

        # S14 (issue #360): propagate ``experiment_id`` from the context
        # carrier into tool kwargs when the tool's schema declares the
        # parameter AND the caller did NOT supply an explicit value. This
        # mirrors the confounders auto-pop pattern above. Explicit caller
        # value always wins (C1 trust-gate parity — key presence, not
        # truthiness, is the explicit-caller signal).
        autopop_experiment_id = self._maybe_autopopulate_experiment_id(
            step, resolved_inputs, context
        )
        if autopop_experiment_id is not None:
            resolved_inputs = {**resolved_inputs, "experiment_id": autopop_experiment_id}

        # F2-core: thread a context-carried DataFrame into tool kwargs under
        # the canonical ``estimation_data`` key (only when the caller did not
        # already supply one). All composable tools accept **kwargs.
        autopop_dataframe = self._maybe_autopopulate_dataframe(step, resolved_inputs, context)
        if autopop_dataframe is not None:
            resolved_inputs = {**resolved_inputs, "estimation_data": autopop_dataframe}

        tool_input = ToolInput(
            tool_name=step.tool_name, parameters=resolved_inputs, context=context
        )

        # G6: Check cache for deterministic tool outputs
        if self._cache_manager:
            cached_output = self._cache_manager.get_tool_output(step.tool_name, resolved_inputs)
            if cached_output is not None:
                logger.debug(f"Cache hit for tool '{step.tool_name}'")
                completed_at = datetime.now(timezone.utc)
                duration_ms = int((completed_at - started_at).total_seconds() * 1000)
                return StepResult(
                    step_id=step.step_id,
                    sub_question_id=step.sub_question_id,
                    tool_name=step.tool_name,
                    input=tool_input,
                    output=ToolOutput(
                        tool_name=step.tool_name,
                        success=True,
                        result=cached_output,
                        execution_time_ms=duration_ms,
                    ),
                    status=ExecutionStatus.COMPLETED,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                )

        # Check circuit breaker before attempting execution
        if not self.failure_tracker.can_execute(step.tool_name):
            logger.warning(f"Circuit breaker OPEN for tool '{step.tool_name}', skipping execution")
            return StepResult(
                step_id=step.step_id,
                sub_question_id=step.sub_question_id,
                tool_name=step.tool_name,
                input=tool_input,
                output=ToolOutput(
                    tool_name=step.tool_name,
                    success=False,
                    error=f"Circuit breaker open for tool '{step.tool_name}'",
                ),
                status=ExecutionStatus.SKIPPED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

        # Get the tool callable
        tool_callable = self.registry.get_callable(step.tool_name)

        if not tool_callable:
            error_msg = f"Tool '{step.tool_name}' not found in registry"
            self.failure_tracker.record_failure(step.tool_name, error_msg)
            return StepResult(
                step_id=step.step_id,
                sub_question_id=step.sub_question_id,
                tool_name=step.tool_name,
                input=tool_input,
                output=ToolOutput(
                    tool_name=step.tool_name,
                    success=False,
                    error=error_msg,
                ),
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

        # Execute with retries and exponential backoff
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Execute the tool
                if asyncio.iscoroutinefunction(tool_callable):
                    result = await asyncio.wait_for(
                        tool_callable(**resolved_inputs), timeout=self.timeout_seconds
                    )
                else:
                    result = await self._run_sync_tool(
                        tool_callable, resolved_inputs, step.tool_name
                    )

                completed_at = datetime.now(timezone.utc)
                duration_ms = int((completed_at - started_at).total_seconds() * 1000)

                # Convert result to dict if needed
                if hasattr(result, "model_dump"):
                    result_dict = result.model_dump()
                elif hasattr(result, "dict"):
                    result_dict = result.dict()
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {"value": result}

                # Record success with the failure tracker
                self.failure_tracker.record_success(step.tool_name, duration_ms)

                # G6: Cache result for deterministic tools
                if self._cache_manager:
                    self._cache_manager.cache_tool_output(
                        step.tool_name, resolved_inputs, result_dict
                    )

                return StepResult(
                    step_id=step.step_id,
                    sub_question_id=step.sub_question_id,
                    tool_name=step.tool_name,
                    input=tool_input,
                    output=ToolOutput(
                        tool_name=step.tool_name,
                        success=True,
                        result=result_dict,
                        execution_time_ms=duration_ms,
                    ),
                    status=ExecutionStatus.COMPLETED,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                )

            except ToolInputError as e:
                # #1573: deterministic input-contract violation — retrying with
                # identical inputs cannot succeed, so fail once with the tool's
                # stated reason. Not recorded against the circuit breaker: a
                # rejected input says nothing about the tool's health, and a
                # plan defect must not open the circuit for other, valid steps
                # that use the same tool.
                logger.warning(
                    f"Step {step.step_id} input rejected by '{step.tool_name}': {e} "
                    f"— not retrying (deterministic input-contract violation)"
                )
                return StepResult(
                    step_id=step.step_id,
                    sub_question_id=step.sub_question_id,
                    tool_name=step.tool_name,
                    input=tool_input,
                    output=ToolOutput(
                        tool_name=step.tool_name,
                        success=False,
                        error=f"input contract violation: {e}",
                    ),
                    status=ExecutionStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )

            except SyncToolTimeout as e:
                # #1592: the sync branch's timeout envelope fired. Unlike the
                # async branch — where ``wait_for`` CANCELS the coroutine, so a
                # retry starts from a clean slate — the timed-out sync call is
                # still running on its bounded-pool thread (see
                # :meth:`_run_sync_tool`). Re-dispatching the same compute over
                # the same inputs would queue a second copy BEHIND the first on
                # a pool sized to the heavy-compute budget: it cannot finish any
                # sooner and it delays every other heavy op. So: fail the step
                # once, with the honest reason.
                #
                # Recorded against the circuit breaker (unlike ToolInputError):
                # a tool that exceeds the step budget IS a health signal, and
                # opening the circuit after repeated timeouts stops the plan
                # from feeding more abandoned work into the pool.
                logger.warning(
                    f"Step {step.step_id} tool '{step.tool_name}' {e} — not retrying "
                    "(the abandoned thread still holds a bounded-pool slot)"
                )
                self.failure_tracker.record_failure(step.tool_name, str(e))
                return StepResult(
                    step_id=step.step_id,
                    sub_question_id=step.sub_question_id,
                    tool_name=step.tool_name,
                    input=tool_input,
                    output=ToolOutput(
                        tool_name=step.tool_name,
                        success=False,
                        error=str(e),
                    ),
                    status=ExecutionStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Step {step.step_id} attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    # Use exponential backoff with jitter
                    delay = self.backoff.get_delay(attempt)
                    logger.debug(f"Retrying in {delay:.2f}s (attempt {attempt + 2})")
                    await asyncio.sleep(delay)

        # All retries exhausted - record failure
        self.failure_tracker.record_failure(step.tool_name, last_error or "Unknown error")

        completed_at = datetime.now(timezone.utc)
        return StepResult(
            step_id=step.step_id,
            sub_question_id=step.sub_question_id,
            tool_name=step.tool_name,
            input=tool_input,
            output=ToolOutput(
                tool_name=step.tool_name, success=False, error=last_error or "Unknown error"
            ),
            status=ExecutionStatus.FAILED,
            started_at=started_at,
            completed_at=completed_at,
        )

    async def _run_sync_tool(
        self, tool_callable: Any, resolved_inputs: Dict[str, Any], tool_name: str
    ) -> Any:
        """Run a SYNC tool off the loop: bounded pool + timeout envelope (#1592).

        Two guarantees the previous ``run_in_executor(None, ...)`` did not give:

        1. **Bounded.** ``None`` meant the loop's DEFAULT executor
           (``min(32, cpu+4)`` threads — 12 on the prod box), which sidesteps the
           per-worker heavy-compute bound that exists to keep concurrent
           in-process compute inside the api container's 5G cgroup. 16 of the 20
           registered composable tools are sync, and several of them (
           ``causal_effect_estimator``, ``refutation_runner``, ``cate_analyzer``,
           ``propensity_estimator``, ``cohort_builder``) fit real models over the
           in-context frame. They now share the process-global bounded pool
           (default: ONE worker) with the two other paths that use it —
           ``rank_drivers``' SHAP (#1590) and ``POST /api/digital-twin/simulate``
           — so those can no longer run concurrently with a composer sync tool.
           A composer-private pool was rejected: it would re-multiply exactly the
           in-flight memory ``compute.py`` exists to bound.

           NOT bounded against: ``POST /api/explain/predict``, which holds a
           ``heavy_compute_slot`` but runs SHAP on its OWN pool
           (``src/api/routes/explain.py``), so it can still overlap with a
           composer sync tool. That overlap is unchanged from before this fix
           (the two were already on different pools) — this change only ever
           REMOVES concurrency. Closing it would mean taking a
           ``heavy_compute_slot`` here, which is deliberately not done: the slot
           is reject-fast for API entry points that can answer 503 +
           ``Retry-After``, and a chat turn's tool step should queue briefly
           (bounded by the envelope below) rather than fail outright — the same
           call #1590 made for ``rank_drivers``.
        2. **Time-boxed**, matching the async branch's ``self.timeout_seconds``.
           Measured headroom on the largest frame observed in prod (37,515x12,
           #1548): ``refutation_runner`` 26.0s, ``cate_analyzer`` 0.03s on a
           realistic low-cardinality segment (35.3s when a planner binds a
           high-cardinality column — the pathology the envelope is FOR),
           ``causal_effect_estimator`` 0.32s, ``propensity_estimator`` 0.10s.

        RESIDUAL — what this does NOT fix. ``wait_for`` cancels the FUTURE, never
        the thread: on timeout the tool keeps running to completion and its
        result is discarded. So (a) the abandoned call still occupies a bounded
        pool slot until it returns, delaying other heavy compute (never the
        loop); and (b) if that compute holds the GIL across a single C call —
        #1548's ``dense_tree_shap`` held it for 1240.5s — the loop is still
        starved for its duration and gunicorn can still murder the worker. The
        envelope bounds PLAN latency and gives the step an honest failure; it is
        not a substitute for bounding the compute itself.

        Raises:
            SyncToolTimeout: when the envelope fires. Any exception raised by the
                tool itself propagates unchanged (including a ``TimeoutError`` of
                its own, which is a tool failure, not an envelope timeout, and
                keeps the ordinary retry semantics).
        """
        # Function-local import (mirrors the #1590 precedent in
        # ``tool_registry/tools/causal_discovery.py``): keeps the
        # ``src.api.dependencies`` package out of this module's import path for
        # non-API consumers of the composer.
        from src.api.dependencies.compute import run_in_bounded_executor

        async def bounded_call() -> tuple[Any, Exception | None]:
            # Capture the tool's own failure instead of letting it reach
            # ``wait_for``, so a TimeoutError raised BY the tool can never be
            # mislabeled as an envelope timeout. ``except Exception`` (not
            # BaseException) deliberately lets CancelledError propagate.
            try:
                return await run_in_bounded_executor(lambda: tool_callable(**resolved_inputs)), None
            except Exception as exc:  # noqa: BLE001 - re-raised by the caller
                return None, exc

        try:
            value, tool_error = await asyncio.wait_for(bounded_call(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise SyncToolTimeout(
                f"timed out after {self.timeout_seconds}s (sync tool, still running "
                "on the bounded heavy-compute pool — a thread cannot be cancelled)"
            ) from exc

        if tool_error is not None:
            raise tool_error
        return value

    async def _execute_parallel(
        self,
        steps: List[ExecutionStep],
        prior_outputs: Dict[str, Any],
        context: Dict[str, Any],
        failed_step_ids: Optional[set[str]] = None,
    ) -> List[StepResult]:
        """Execute multiple steps in parallel"""
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute_with_semaphore(step: ExecutionStep) -> StepResult:
            async with semaphore:
                return await self._execute_step(step, prior_outputs, context, failed_step_ids)

        tasks = [execute_with_semaphore(step) for step in steps]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def _resolve_inputs(
        self,
        input_mapping: Dict[str, Any],
        prior_outputs: Dict[str, Any],
        context: Dict[str, Any],
        _nested: bool = False,
    ) -> Dict[str, Any]:
        """
        Resolve input parameters, substituting references to prior outputs.

        References use the syntax: $step_X.field or $step_X.nested.field

        Reference contract (#1573):

        - A TOP-LEVEL string reference (``"param": "$step_X.field"``) that
          cannot be resolved raises :class:`ReferenceResolutionError`: the
          planner declared that argument IS the referenced value, so an
          unresolvable reference means the argument is missing and the step
          is deterministically doomed — it must fail fast with an explicit
          reason instead of silently receiving ``None`` (the q08
          ``NoneType * float`` crash).
        - References NESTED inside dict/list values degrade to ``None`` with
          a warning (unchanged behavior): nested containers are planner
          CONSTRUCTIONS — e.g. the ``discover_dag`` ``data={'col': '$ref'}``
          artifact — with a test-pinned degradation contract (F7 DataFrame
          auto-injection repairs them downstream). Blanket strictness here
          measurably breaks steps that succeed today.

        Raises:
            ReferenceResolutionError: on an unresolvable top-level reference.
        """
        resolved = {}

        for param, value in input_mapping.items():
            if isinstance(value, str) and value.startswith("$"):
                # This is a reference to a prior output
                if _nested:
                    resolved[param] = self._resolve_reference_lenient(value, prior_outputs, context)
                else:
                    resolved[param] = self._resolve_reference(value, prior_outputs, context)
            elif isinstance(value, dict):
                # Recursively resolve nested dicts (lenient — see docstring)
                resolved[param] = self._resolve_inputs(value, prior_outputs, context, _nested=True)
            elif isinstance(value, list):
                # Resolve each list item (lenient — list literals are
                # constructions, same rationale as nested dicts)
                resolved[param] = [
                    (
                        self._resolve_reference_lenient(v, prior_outputs, context)
                        if isinstance(v, str) and v.startswith("$")
                        else v
                    )
                    for v in value
                ]
            else:
                resolved[param] = value

        return resolved

    # Cap for listing available fields/sources in error messages — keeps the
    # synthesis-visible reason informative without ballooning on wide outputs.
    _MAX_LISTED_FIELDS = 25

    def _resolve_reference(
        self, reference: str, prior_outputs: Dict[str, Any], context: Dict[str, Any]
    ) -> Any:
        """
        Resolve a reference like $step_1.field.nested_field

        Special references:
        - $context.field: Access context dictionary
        - $step_X.field: Access output from step X

        Raises:
            ReferenceResolutionError: when the source is unknown (e.g. a
                planner-invented ``$dataset``) or the field path does not
                exist on the source (#1573). This method NEVER silently
                returns ``None`` for an unresolvable reference — the caller
                decides whether to fail fast (top-level argument) or degrade
                (:meth:`_resolve_reference_lenient`, nested constructions).
        """
        # Remove the $ prefix and split by dots
        parts = reference[1:].split(".")

        # Determine the source
        source_key = parts[0]
        field_path = parts[1:]

        if source_key == "context":
            source = context
        elif source_key in prior_outputs:
            source = prior_outputs[source_key]
        else:
            available = sorted(prior_outputs.keys())[: self._MAX_LISTED_FIELDS]
            raise ReferenceResolutionError(
                reference=reference,
                reason=(
                    f"unknown source '{source_key}' — valid sources are '$context' or "
                    f"the id of a prior step with a successful output "
                    f"(available step outputs: {available!r})"
                ),
            )

        # Navigate the field path
        current = source
        for field in field_path:  # noqa: F402
            if isinstance(current, dict) and field in current:
                current = current[field]
            elif hasattr(current, field):
                current = getattr(current, field)
            else:
                available = (
                    sorted(str(k) for k in current.keys())[: self._MAX_LISTED_FIELDS]
                    if isinstance(current, dict)
                    else []
                )
                raise ReferenceResolutionError(
                    reference=reference,
                    reason=(
                        f"field '{field}' not found on source '{source_key}'"
                        + (f" (available fields: {available!r})" if available else "")
                    ),
                )

        return current

    def _resolve_reference_lenient(
        self, reference: str, prior_outputs: Dict[str, Any], context: Dict[str, Any]
    ) -> Any:
        """Resolve a NESTED reference, degrading to ``None`` with a warning.

        Preserves the pre-#1573 behavior for references inside dict/list
        constructions (the F7 ``discover_dag`` data-dict contract depends on
        it). Top-level references go through the strict
        :meth:`_resolve_reference` instead.
        """
        try:
            return self._resolve_reference(reference, prior_outputs, context)
        except ReferenceResolutionError as e:
            logger.warning(f"Nested reference degraded to None: {e}")
            return None

    # ------------------------------------------------------------------
    # Phase 7.2 + S14: causal-role auto-population
    # ------------------------------------------------------------------

    def _maybe_autopopulate_confounders(
        self,
        step: ExecutionStep,
        resolved_inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[List[str]]:
        """Pre-fill ``confounders`` from the role-attribution repository.

        Fires when ALL of the following hold:

        1. ``context["experiment_id"]`` is a non-empty string (S14
           propagation — the caller carries the experiment id through
           the existing context dict).
        2. The tool's schema declares a ``confounders`` parameter (we
           don't inject a kwarg the tool can't accept).
        3. The caller did NOT supply an explicit ``confounders`` value
           in the resolved inputs (caller-explicit always wins per the
           C1 trust-gate — the most-authoritative source).

        The repository's default ``only_evaluator_satisfied=True``
        already filters unverified LLM rows at the SQL layer; this
        method ALSO re-checks ``should_act`` at the consumer boundary
        so a future caller passing ``only_evaluator_satisfied=False``
        cannot leak unverified roles into tool calls.

        Plan: ``.claude/plans/causal_role_propagation_FINAL.md`` §7.2.
        """
        # Gate 1: experiment_id present in context (S14)
        experiment_id = context.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            return None

        # Gate 2: tool schema declares a ``confounders`` parameter
        schema = self.registry.get_schema(step.tool_name)
        if schema is None:
            return None
        param_names = {p.name for p in schema.input_parameters}
        if "confounders" not in param_names:
            return None

        # Gate 3: caller did not supply explicit confounders.
        # Codex audit (PR #367): key presence — not non-None value — is the
        # explicit-caller signal. An explicit ``confounders=None`` is still
        # a caller decision and must NOT be auto-populated.
        if "confounders" in resolved_inputs:
            return None

        # Query + filter + assign — Phase 7.2 is an enhancement, not a hard
        # dependency for tool execution. Codex audit (PR #367): the broad
        # try/except must envelope the FULL hook (query, filter, assignment),
        # not only the SQL query — a malformed attribution row would otherwise
        # raise KeyError outside the try and fail tool execution.
        try:
            attributions: list[RoleAttribution] = query_active_role_attributions(experiment_id)
        except Exception as e:  # noqa: BLE001 — broad on purpose
            logger.warning(
                f"Role-attribution auto-pop query failed for "
                f"experiment_id={experiment_id!r}: {e}. "
                f"Proceeding without confounder pre-fill."
            )
            return None

        # Filter: causal_role == 'confounder' AND C1 trust-gate (consumer-
        # boundary defense-in-depth). Each row is wrapped in its own
        # try/except so a single malformed row doesn't discard ALL valid
        # rows in the same batch (codex iter-2 LOW finding).
        confounder_features: list[str] = []
        for attr in attributions:
            if not isinstance(attr, dict):
                continue
            try:
                if attr.get("causal_role") != "confounder":
                    continue
                feature = attr.get("feature")
                if not isinstance(feature, str):
                    continue
                if not should_act(attr):
                    continue
                confounder_features.append(feature)
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    f"Auto-pop skipped malformed attribution row "
                    f"for experiment_id={experiment_id!r}: {e}"
                )
                continue

        logger.debug(
            f"Auto-populated {len(confounder_features)} confounder(s) for "
            f"tool '{step.tool_name}' from experiment_id={experiment_id!r}"
        )
        return confounder_features

    def _maybe_autopopulate_experiment_id(
        self,
        step: ExecutionStep,
        resolved_inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Propagate ``experiment_id`` from context into tool kwargs.

        Issue #360 (S14). Mirrors the
        ``_maybe_autopopulate_confounders`` triple-gate pattern:

        1. ``context["experiment_id"]`` is a non-empty string.
        2. The tool's schema declares an ``experiment_id`` parameter
           (we don't inject a kwarg the tool can't accept).
        3. The caller did NOT supply an explicit value in the
           resolved inputs. Key presence — not truthiness — is the
           explicit-caller signal (C1 trust-gate parity with the
           confounders hook). An explicit ``experiment_id=None`` is
           still a caller decision and must NOT be overridden.

        Returns the value to inject, or ``None`` to skip auto-pop
        entirely.
        """
        # Gate 1: experiment_id present in context (S14)
        experiment_id = context.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            return None

        # Gate 2: tool schema declares an ``experiment_id`` parameter
        schema = self.registry.get_schema(step.tool_name)
        if schema is None:
            return None
        param_names = {p.name for p in schema.input_parameters}
        if "experiment_id" not in param_names:
            return None

        # Gate 3: caller did not supply explicit experiment_id.
        if "experiment_id" in resolved_inputs:
            return None

        logger.debug(f"Auto-populated experiment_id={experiment_id!r} for tool '{step.tool_name}'")
        return experiment_id

    @staticmethod
    def _is_explicit_dataframe_input(value: Any) -> bool:
        """True iff ``value`` is a genuine caller-explicit frame or data dict.

        Gate-1 (caller-explicit-wins) must distinguish a REAL explicit input
        from the planner's ``discover_dag`` artifact ``data={'col': '$ref'}``
        (a column->reference-string dict). The latter is NOT a usable frame and
        NOT a valid ``Dict[str, List]`` discovery dict — so it must NOT block
        auto-injection of the in-context real DataFrame (F7).

        Counts as explicit when ``value`` is either:

        * a pandas-like DataFrame (duck-typed: has ``.columns`` and ``__len__``),
          OR
        * a non-empty ``dict`` whose every value is a list/tuple (the legacy
          ``DataFrame.to_dict('list')`` contract that ``discover_dag`` accepts).
        """
        if value is None:
            return False
        if hasattr(value, "columns") and hasattr(value, "__len__"):
            return True
        if isinstance(value, dict) and value:
            return all(isinstance(v, (list, tuple)) for v in value.values())
        return False

    def _maybe_autopopulate_dataframe(
        self,
        step: ExecutionStep,
        resolved_inputs: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[Any]:
        """Thread a context-carried DataFrame into tool kwargs (F2-core, F7).

        The composable tools read their working frame from ``**kwargs`` via
        ``_extract_dataframe_from_kwargs`` — but the planner's
        ``input_mapping`` never names ``data``/``dataframe``/``estimation_data``,
        so until now NO production path delivered a frame. This hook mirrors
        the ``_maybe_autopopulate_experiment_id`` pattern and fires when:

        1. ``context`` carries a pandas-like DataFrame under ANY canonical
           key in ``_DATAFRAME_KWARGS_KEYS`` (duck-typed: has ``.columns``).
        2. The caller did NOT already supply a GENUINE explicit frame / valid
           data dict under any of those keys (caller-explicit always wins —
           C1 parity).

        F7 refinement: ``discover_dag`` / ``rank_drivers`` consume the SAME
        real DataFrame as the other causal tools. The planner emits
        ``data={'col': '$step.field'}`` for ``discover_dag`` — a column->ref
        dict that is NOT a usable frame and NOT a valid ``Dict[str, List]``.
        Such a value must NOT count as "caller-explicit" (otherwise the real
        frame would never be injected and the tool would fail). Gate-1 now
        treats a canonical key as explicit ONLY when its value is a real frame
        or a valid data dict (see :meth:`_is_explicit_dataframe_input`); the
        broken planner dict falls through and injection proceeds.

        Returns the frame to inject (under the canonical ``estimation_data``
        key — NOT ``data``, since ``discover_dag``'s ``DiscoverDagInput.data``
        is a Dict), or ``None`` to skip injection entirely. All composable
        tools accept ``**kwargs`` so an unused ``estimation_data`` is harmless.
        """
        # Gate 1: caller-explicit wins — but ONLY for a GENUINE explicit frame
        # or valid data dict. A present-but-unusable value (the planner's
        # column->reference dict for discover_dag) does NOT block injection.
        for key in _DATAFRAME_KWARGS_KEYS:
            if key in resolved_inputs and self._is_explicit_dataframe_input(resolved_inputs[key]):
                return None

        # Gate 2: context carries a duck-typed DataFrame under a canonical key.
        for key in _DATAFRAME_KWARGS_KEYS:
            candidate = context.get(key)
            if candidate is None:
                continue
            if hasattr(candidate, "columns") and hasattr(candidate, "__len__"):
                logger.debug(
                    f"Auto-injected context DataFrame (from context[{key!r}]) "
                    f"into tool '{step.tool_name}' as 'estimation_data'."
                )
                return candidate
        return None

    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get execution statistics for all tools.

        Returns a dictionary with per-tool stats including:
        - total_calls: Number of executions
        - success_rate: Success percentage (0.0 - 1.0)
        - avg_latency_ms: Average execution time
        - circuit_breaker: Current circuit breaker state
        """
        return self.failure_tracker.get_all_stats()

    def reset_tool_stats(self, tool_name: Optional[str] = None) -> None:
        """
        Reset tool statistics.

        Args:
            tool_name: Specific tool to reset, or None to reset all
        """
        self.failure_tracker.reset(tool_name)

    def update_tool_performance(
        self,
        tool_name: Optional[str] = None,
        min_calls: int = 10,
    ) -> Dict[str, bool]:
        """
        Sync learned performance metrics back to the tool registry (G8).

        Updates the registry's avg_execution_ms for tools that have been
        executed enough times to provide reliable EMA latency estimates.

        Args:
            tool_name: Specific tool to update, or None to update all
            min_calls: Minimum number of calls required before updating

        Returns:
            Dictionary mapping tool names to update success status
        """
        results: Dict[str, bool] = {}

        # Get tools to update
        if tool_name:
            tools_to_update = [tool_name] if tool_name in self.failure_tracker._stats else []
        else:
            tools_to_update = list(self.failure_tracker._stats.keys())

        for name in tools_to_update:
            stats = self.failure_tracker.get_stats(name)
            if not stats:
                results[name] = False
                continue

            # Only update if we have enough observations
            if stats.total_calls < min_calls:
                logger.debug(f"Skipping {name}: only {stats.total_calls} calls (need {min_calls})")
                results[name] = False
                continue

            # Only update if we have a valid EMA latency
            if stats.ema_latency_ms <= 0:
                logger.debug(f"Skipping {name}: no valid EMA latency")
                results[name] = False
                continue

            # Get the registered tool and update its schema
            registered = self.registry.get(name)
            if not registered:
                logger.warning(f"Tool {name} not found in registry")
                results[name] = False
                continue

            # Update the schema's avg_execution_ms with learned EMA
            old_latency = registered.schema.avg_execution_ms
            new_latency = int(round(stats.ema_latency_ms))
            registered.schema.avg_execution_ms = new_latency

            logger.info(
                f"G8: Updated {name} latency: {old_latency}ms → {new_latency}ms "
                f"(EMA from {stats.total_calls} calls, "
                f"recent success rate: {stats.recent_success_rate:.1%})"
            )
            results[name] = True

        return results


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ExecutionError(Exception):
    """Error during plan execution"""

    pass


class SyncToolTimeout(Exception):
    """A SYNC tool exceeded the step's timeout envelope (#1592).

    Distinct from a plain ``TimeoutError`` so the retry loop can tell "the
    envelope fired around a thread we cannot cancel" (do not re-dispatch) from
    "the tool itself reported a timeout" (ordinary retryable failure). Raised
    only by :meth:`PlanExecutor._run_sync_tool`.
    """

    pass


# ============================================================================
# SYNC WRAPPER
# ============================================================================


def execute_sync(
    plan: ExecutionPlan, context: Optional[Dict[str, Any]] = None, **kwargs
) -> ExecutionTrace:
    """
    Synchronous wrapper for execution.

    Handles event loop conflicts when called from async contexts.
    """
    import asyncio

    executor = PlanExecutor(**kwargs)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(executor.execute(plan, context))
    else:
        return asyncio.run(executor.execute(plan, context))
