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

from src.tool_registry.registry import ToolRegistry

# Import tool_registrations to ensure tools are registered before execution
# The @composable_tool decorators register tools when the module is imported
from . import tool_registrations as _tool_registrations  # noqa: F401
from .cache import get_cache_manager
from .models.composition_models import (
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    ExecutionTrace,
    StepResult,
    ToolInput,
    ToolOutput,
)

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
                        result = await self._execute_step(step, outputs, context)
                        trace.add_result(result)
                        if result.output.is_success:
                            outputs[step.step_id] = result.output.result
                else:
                    # Multiple steps, execute in parallel
                    results = await self._execute_parallel(
                        [step for sid in group if (step := plan.get_step(sid)) is not None],
                        outputs,
                        context,
                    )
                    for result in results:
                        trace.add_result(result)
                        if result.output.is_success:
                            outputs[result.step_id] = result.output.result
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
        self, step: ExecutionStep, prior_outputs: Dict[str, Any], context: Dict[str, Any]
    ) -> StepResult:
        """Execute a single step with circuit breaker and exponential backoff."""
        started_at = datetime.now(timezone.utc)

        logger.debug(f"Executing step {step.step_id}: {step.tool_name}")

        # Resolve input parameters
        resolved_inputs = self._resolve_inputs(step.input_mapping, prior_outputs, context)

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
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: tool_callable(**resolved_inputs)
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

    async def _execute_parallel(
        self, steps: List[ExecutionStep], prior_outputs: Dict[str, Any], context: Dict[str, Any]
    ) -> List[StepResult]:
        """Execute multiple steps in parallel"""
        # Limit concurrency
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute_with_semaphore(step: ExecutionStep) -> StepResult:
            async with semaphore:
                return await self._execute_step(step, prior_outputs, context)

        tasks = [execute_with_semaphore(step) for step in steps]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def _resolve_inputs(
        self, input_mapping: Dict[str, Any], prior_outputs: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve input parameters, substituting references to prior outputs.

        References use the syntax: $step_X.field or $step_X.nested.field
        """
        resolved = {}

        for param, value in input_mapping.items():
            if isinstance(value, str) and value.startswith("$"):
                # This is a reference to a prior output
                resolved[param] = self._resolve_reference(value, prior_outputs, context)
            elif isinstance(value, dict):
                # Recursively resolve nested dicts
                resolved[param] = self._resolve_inputs(value, prior_outputs, context)
            elif isinstance(value, list):
                # Resolve each list item
                resolved[param] = [
                    (
                        self._resolve_reference(v, prior_outputs, context)
                        if isinstance(v, str) and v.startswith("$")
                        else v
                    )
                    for v in value
                ]
            else:
                resolved[param] = value

        return resolved

    def _resolve_reference(
        self, reference: str, prior_outputs: Dict[str, Any], context: Dict[str, Any]
    ) -> Any:
        """
        Resolve a reference like $step_1.field.nested_field

        Special references:
        - $context.field: Access context dictionary
        - $step_X.field: Access output from step X
        """
        # Remove the $ prefix
        ref = reference[1:]

        # Split by dots
        parts = ref.split(".")

        if not parts:
            return None

        # Determine the source
        source_key = parts[0]
        field_path = parts[1:]

        if source_key == "context":
            source = context
        elif source_key in prior_outputs:
            source = prior_outputs[source_key]
        else:
            logger.warning(f"Unknown reference source: {source_key}")
            return None

        # Navigate the field path
        current = source
        for field in field_path:  # noqa: F402
            if isinstance(current, dict) and field in current:
                current = current[field]
            elif hasattr(current, field):
                current = getattr(current, field)
            else:
                logger.warning(f"Could not resolve field '{field}' in reference '{reference}'")
                return None

        return current

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
