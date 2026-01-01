"""
E2I Tool Composer - Phase 2: Planner
Version: 4.4
Purpose: Map sub-questions to tools and create execution plan

V4.4 Updates:
- Added causal discovery tool hints (discover_dag, rank_drivers, detect_structural_drift)
- Added tool chaining guidance for discover_dag → rank_drivers pipeline
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.tool_registry.registry import ToolRegistry

from .cache import get_cache_manager
from .memory_hooks import ToolComposerMemoryHooks, get_tool_composer_memory_hooks
from .models.composition_models import (
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStep,
    ToolMapping,
)

logger = logging.getLogger(__name__)


# ============================================================================
# PLANNING PROMPT
# ============================================================================

PLANNING_SYSTEM_PROMPT = """You are a tool planning specialist for a pharmaceutical analytics platform.

Your task is to map sub-questions to available tools and create an execution plan.

## Available Tools:
{tools_description}

## Causal Discovery Tool Hints:
Use these mappings for causal discovery queries:
- "discover causal structure" / "learn DAG" / "causal graph" / "causal relationships" → discover_dag
- "causal vs predictive" / "driver ranking" / "feature importance" / "which features cause" → rank_drivers
- "structural drift" / "causal structure changed" / "DAG stability" → detect_structural_drift

Tool Chaining for Causal Discovery:
- discover_dag produces `edge_list` → rank_drivers can consume it as `dag_edge_list`
- Example chain: discover_dag (step_1) → rank_drivers (step_2) with input_mapping: {{"dag_edge_list": "$step_1.edge_list"}}

## Guidelines:
1. Match each sub-question to the most appropriate tool based on:
   - The question's intent (CAUSAL, COMPARATIVE, etc.)
   - Required inputs and outputs
   - Tool capabilities
2. Identify which tool outputs feed into which tool inputs
3. Determine execution order based on dependencies
4. Group independent tools for parallel execution

## Output Format:
Return a JSON object with:
{{
  "reasoning": "Your step-by-step reasoning for the plan",
  "tool_mappings": [
    {{
      "sub_question_id": "sq_1",
      "tool_name": "tool_name_here",
      "confidence": 0.95,
      "reasoning": "Why this tool fits"
    }}
  ],
  "execution_steps": [
    {{
      "step_id": "step_1",
      "sub_question_id": "sq_1",
      "tool_name": "tool_name_here",
      "input_mapping": {{
        "param_name": "value or $step_X.field for prior outputs"
      }},
      "depends_on_steps": []
    }}
  ],
  "parallel_groups": [
    ["step_1", "step_2"],  // Steps that can run in parallel
    ["step_3"]             // Must wait for group 1
  ]
}}

## Important:
- Every sub-question must map to exactly one tool
- Use $step_X.field syntax to reference prior step outputs
- Parallel groups should be ordered by execution wave"""


PLANNING_USER_TEMPLATE = """Create an execution plan for these sub-questions:

SUB-QUESTIONS:
{sub_questions}

Map each to the best tool and create an execution plan.
Return valid JSON only."""


# ============================================================================
# PLANNER CLASS
# ============================================================================


class ToolPlanner:
    """
    Maps sub-questions to tools and creates execution plans.

    This is Phase 2 of the Tool Composer pipeline.

    Memory Integration (G1, G2):
    - Uses episodic memory to find similar past compositions
    - Leverages successful patterns to optimize tool selection
    """

    def __init__(
        self,
        llm_client: Any,
        tool_registry: Optional[ToolRegistry] = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.2,
        max_tools_per_plan: int = 8,
        memory_hooks: Optional[ToolComposerMemoryHooks] = None,
        use_episodic_memory: bool = True,
        enable_caching: bool = True,
    ):
        self.llm_client = llm_client
        self.registry = tool_registry or ToolRegistry()
        self.model = model
        self.temperature = temperature
        self.max_tools_per_plan = max_tools_per_plan
        self.memory_hooks = memory_hooks or get_tool_composer_memory_hooks()
        self.use_episodic_memory = use_episodic_memory
        self.enable_caching = enable_caching

        # G6: Initialize cache manager for plan similarity matching
        self._cache_manager = get_cache_manager() if enable_caching else None

    async def plan(self, decomposition: DecompositionResult) -> ExecutionPlan:
        """
        Create an execution plan from decomposed sub-questions.

        Args:
            decomposition: Result from Phase 1 (Decomposer)

        Returns:
            ExecutionPlan with tool mappings and execution steps
        """
        logger.info(f"Planning execution for {decomposition.question_count} sub-questions")

        try:
            # G6: Check for similar cached plan
            if self._cache_manager:
                cached_result = self._cache_manager.get_similar_plan(decomposition)
                if cached_result:
                    cached_plan, similarity = cached_result
                    logger.info(
                        f"Found similar cached plan (similarity: {similarity:.2f})"
                    )
                    # Adapt cached plan to current decomposition
                    adapted_plan = self._adapt_cached_plan(cached_plan, decomposition)
                    if adapted_plan:
                        return adapted_plan

            # Get available tools for planning
            tools_description = self._format_tools_for_prompt()

            # G1/G2: Check episodic memory for similar past compositions
            similar_compositions = await self._check_episodic_memory(
                decomposition.original_query
            )

            # Call LLM for planning (with episodic context if available)
            response = await self._call_llm(
                decomposition, tools_description, similar_compositions
            )

            # Parse response
            parsed = self._parse_response(response)

            # Build plan components
            tool_mappings = self._build_tool_mappings(parsed)
            execution_steps = self._build_execution_steps(parsed, decomposition)
            parallel_groups = parsed.get("parallel_groups", [])

            # Validate plan
            self._validate_plan(tool_mappings, execution_steps, decomposition)

            # Calculate estimated duration
            estimated_duration = self._estimate_duration(execution_steps)

            plan = ExecutionPlan(
                decomposition=decomposition,
                steps=execution_steps,
                tool_mappings=tool_mappings,
                estimated_duration_ms=estimated_duration,
                parallel_groups=parallel_groups,
                planning_reasoning=parsed.get("reasoning", ""),
                timestamp=datetime.now(timezone.utc),
            )

            # G6: Cache the plan for future similarity matching
            if self._cache_manager:
                self._cache_manager.cache_plan(decomposition, plan)
                logger.debug("Cached plan for future similarity matching")

            logger.info(f"Created plan with {len(execution_steps)} steps")
            return plan

        except Exception as e:
            logger.error(f"Planning failed: {e}")
            raise PlanningError(f"Failed to create execution plan: {e}") from e

    def _adapt_cached_plan(
        self, cached_plan: ExecutionPlan, decomposition: DecompositionResult
    ) -> Optional[ExecutionPlan]:
        """
        Adapt a cached plan to a new decomposition if possible.

        Returns None if adaptation is not feasible.
        """
        try:
            # Check if sub-question counts match
            if len(cached_plan.steps) != decomposition.question_count:
                logger.debug("Cached plan step count doesn't match, skipping adaptation")
                return None

            # Create new plan with same structure but updated decomposition
            # Only adapt if tool sequences match the intent patterns
            new_plan = ExecutionPlan(
                decomposition=decomposition,
                steps=cached_plan.steps,  # Reuse steps structure
                tool_mappings=cached_plan.tool_mappings,
                estimated_duration_ms=cached_plan.estimated_duration_ms,
                parallel_groups=cached_plan.parallel_groups,
                planning_reasoning=f"Adapted from cached plan: {cached_plan.planning_reasoning}",
                timestamp=datetime.now(timezone.utc),
            )
            return new_plan
        except Exception as e:
            logger.debug(f"Failed to adapt cached plan: {e}")
            return None

    async def _check_episodic_memory(
        self, query: str, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Check episodic memory for similar past compositions (G1, G2).

        Uses vector search to find successful compositions that can inform
        tool selection and execution planning.

        Args:
            query: The original query to find similar compositions for
            limit: Maximum number of similar compositions to retrieve

        Returns:
            List of similar compositions with their tool sequences
        """
        if not self.use_episodic_memory or not self.memory_hooks:
            return []

        try:
            similar = await self.memory_hooks.find_similar_compositions(
                query=query, limit=limit
            )

            if similar:
                logger.info(
                    f"Found {len(similar)} similar compositions in episodic memory"
                )
                # Log the tool sequences for debugging
                for comp in similar:
                    raw = comp.get("raw_content", {})
                    logger.debug(
                        f"  Similar: tools={raw.get('tool_sequence', [])}, "
                        f"confidence={raw.get('confidence', 0):.2f}"
                    )
            return similar
        except Exception as e:
            logger.warning(f"Failed to check episodic memory: {e}")
            return []

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for the planning prompt"""
        schemas = self.registry.get_schemas_for_planning()

        if not schemas:
            raise PlanningError("No tools available in registry")

        lines = []
        for tool in schemas:
            lines.append(f"### {tool['name']} ({tool['source']})")
            lines.append(f"Description: {tool['description']}")
            lines.append(f"Inputs: {', '.join(tool['inputs'])}")
            lines.append(f"Output: {tool['output']}")
            lines.append(f"Avg execution: {tool['avg_ms']}ms")
            lines.append("")

        return "\n".join(lines)

    async def _call_llm(
        self,
        decomposition: DecompositionResult,
        tools_description: str,
        similar_compositions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Call the LLM for planning with optional episodic context."""
        # Format sub-questions
        sq_text = "\n".join(
            [
                f"- {sq.id}: {sq.question} [Intent: {sq.intent}] [Depends on: {sq.depends_on}]"
                for sq in decomposition.sub_questions
            ]
        )

        system_prompt = PLANNING_SYSTEM_PROMPT.format(tools_description=tools_description)

        # G1/G2: Include similar compositions as context
        episodic_context = ""
        if similar_compositions:
            episodic_context = self._format_episodic_context(similar_compositions)

        user_message = PLANNING_USER_TEMPLATE.format(sub_questions=sq_text)
        if episodic_context:
            user_message = f"{episodic_context}\n\n{user_message}"

        response = await self.llm_client.messages.create(
            model=self.model,
            max_tokens=3000,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

    def _format_episodic_context(
        self, similar_compositions: List[Dict[str, Any]]
    ) -> str:
        """Format similar compositions as context for the LLM."""
        if not similar_compositions:
            return ""

        lines = [
            "## Similar Past Compositions (Use as Reference)",
            "The following successful compositions may inform your planning:",
            "",
        ]

        for i, comp in enumerate(similar_compositions, 1):
            raw = comp.get("raw_content", {})
            tool_seq = raw.get("tool_sequence", [])
            confidence = raw.get("confidence", 0)
            duration = raw.get("total_duration_ms", 0)

            lines.append(f"### Reference {i}")
            lines.append(f"- Tools used: {', '.join(tool_seq)}")
            lines.append(f"- Success confidence: {confidence:.2f}")
            lines.append(f"- Execution time: {duration}ms")
            lines.append("")

        lines.append("Consider similar tool sequences if they match the current query's intent.")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse planning JSON: {response[:200]}...")
            raise PlanningError(f"Invalid JSON in LLM response: {e}") from e

    def _build_tool_mappings(self, parsed: Dict[str, Any]) -> List[ToolMapping]:
        """Build ToolMapping objects from parsed response"""
        raw_mappings = parsed.get("tool_mappings", [])

        mappings = []
        for m in raw_mappings:
            # Validate tool exists
            tool_name = m["tool_name"]
            if not self.registry.validate_tool_exists(tool_name):
                logger.warning(f"Tool '{tool_name}' not in registry, skipping")
                continue

            schema = self.registry.get_schema(tool_name)

            mappings.append(
                ToolMapping(
                    sub_question_id=m["sub_question_id"],
                    tool_name=tool_name,
                    source_agent=schema.source_agent if schema else "unknown",
                    confidence=m.get("confidence", 0.8),
                    reasoning=m.get("reasoning", ""),
                )
            )

        return mappings

    def _build_execution_steps(
        self, parsed: Dict[str, Any], decomposition: DecompositionResult
    ) -> List[ExecutionStep]:
        """Build ExecutionStep objects from parsed response"""
        raw_steps = parsed.get("execution_steps", [])

        # Build mapping from sub_question_id to dependencies
        {sq.id: sq.depends_on for sq in decomposition.sub_questions}

        steps = []
        for s in raw_steps:
            tool_name = s["tool_name"]

            # Validate tool exists
            if not self.registry.validate_tool_exists(tool_name):
                raise PlanningError(f"Unknown tool in plan: {tool_name}")

            schema = self.registry.get_schema(tool_name)

            # Determine dependency type
            dep_type = DependencyType.SEQUENTIAL
            if not s.get("depends_on_steps"):
                dep_type = DependencyType.PARALLEL

            steps.append(
                ExecutionStep(
                    step_id=s.get("step_id", f"step_{len(steps)+1}"),
                    sub_question_id=s["sub_question_id"],
                    tool_name=tool_name,
                    source_agent=schema.source_agent if schema else "unknown",
                    input_mapping=s.get("input_mapping", {}),
                    dependency_type=dep_type,
                    depends_on_steps=s.get("depends_on_steps", []),
                )
            )

        return steps

    def _validate_plan(
        self,
        mappings: List[ToolMapping],
        steps: List[ExecutionStep],
        decomposition: DecompositionResult,
    ) -> None:
        """Validate the execution plan"""
        # Check all sub-questions are mapped
        sq_ids = {sq.id for sq in decomposition.sub_questions}
        mapped_ids = {m.sub_question_id for m in mappings}

        missing = sq_ids - mapped_ids
        if missing:
            raise PlanningError(f"Sub-questions not mapped to tools: {missing}")

        # Check all steps reference valid tools
        for step in steps:
            if not self.registry.validate_tool_exists(step.tool_name):
                raise PlanningError(f"Step references unknown tool: {step.tool_name}")

        # Check step dependencies are valid
        step_ids = {s.step_id for s in steps}
        for step in steps:
            for dep in step.depends_on_steps:
                if dep not in step_ids:
                    raise PlanningError(f"Step {step.step_id} depends on unknown step {dep}")

        # Check for dependency cycles
        self._check_cycles(steps)

    def _check_cycles(self, steps: List[ExecutionStep]) -> None:
        """Check for cycles in step dependencies"""
        visited = set()
        rec_stack = set()

        step_map = {s.step_id: s for s in steps}

        def has_cycle(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)

            step = step_map.get(step_id)
            if step:
                for dep in step.depends_on_steps:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(step_id)
            return False

        for step in steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id):
                    raise PlanningError("Cycle detected in execution plan")

    def _estimate_duration(self, steps: List[ExecutionStep]) -> int:
        """Estimate total execution duration"""
        total_ms = 0
        for step in steps:
            schema = self.registry.get_schema(step.tool_name)
            if schema:
                total_ms += schema.avg_execution_ms
            else:
                total_ms += 1000  # Default estimate

        return total_ms


# ============================================================================
# EXCEPTIONS
# ============================================================================


class PlanningError(Exception):
    """Error during execution planning"""

    pass


# ============================================================================
# SYNC WRAPPER
# ============================================================================


def plan_sync(decomposition: DecompositionResult, llm_client: Any, **kwargs) -> ExecutionPlan:
    """
    Synchronous wrapper for planning.

    Handles event loop conflicts when called from async contexts.
    """
    import asyncio

    planner = ToolPlanner(llm_client=llm_client, **kwargs)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(planner.plan(decomposition))
    else:
        return asyncio.run(planner.plan(decomposition))
