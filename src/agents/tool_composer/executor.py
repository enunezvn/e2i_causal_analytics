"""
E2I Tool Composer - Phase 3: Executor
Version: 4.2
Purpose: Execute tool chains according to the execution plan
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.tool_registry.registry import ToolRegistry

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
    - Handles retries and failures gracefully
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        max_parallel: int = 3,
        max_retries: int = 2,
        timeout_seconds: int = 120,
    ):
        self.registry = tool_registry or ToolRegistry()
        self.max_parallel = max_parallel
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

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
                        [plan.get_step(sid) for sid in group if plan.get_step(sid)],
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
        """Execute a single step"""
        started_at = datetime.now(timezone.utc)

        logger.debug(f"Executing step {step.step_id}: {step.tool_name}")

        # Resolve input parameters
        resolved_inputs = self._resolve_inputs(step.input_mapping, prior_outputs, context)

        tool_input = ToolInput(
            tool_name=step.tool_name, parameters=resolved_inputs, context=context
        )

        # Get the tool callable
        tool_callable = self.registry.get_callable(step.tool_name)

        if not tool_callable:
            return StepResult(
                step_id=step.step_id,
                sub_question_id=step.sub_question_id,
                tool_name=step.tool_name,
                input=tool_input,
                output=ToolOutput(
                    tool_name=step.tool_name,
                    success=False,
                    error=f"Tool '{step.tool_name}' not found in registry",
                ),
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

        # Execute with retries
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
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

        # All retries exhausted
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
        for field in field_path:
            if isinstance(current, dict) and field in current:
                current = current[field]
            elif hasattr(current, field):
                current = getattr(current, field)
            else:
                logger.warning(f"Could not resolve field '{field}' in reference '{reference}'")
                return None

        return current


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
    """
    import asyncio

    executor = PlanExecutor(**kwargs)
    return asyncio.run(executor.execute(plan, context))
