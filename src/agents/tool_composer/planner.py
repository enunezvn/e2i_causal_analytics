"""
E2I Tool Composer - Phase 2: Planner
Version: 4.2
Purpose: Map sub-questions to tools and create execution plan
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models.composition_models import (
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStep,
    SubQuestion,
    ToolMapping,
)
from src.tool_registry.registry import ToolRegistry, ToolSchema

logger = logging.getLogger(__name__)


# ============================================================================
# PLANNING PROMPT
# ============================================================================

PLANNING_SYSTEM_PROMPT = """You are a tool planning specialist for a pharmaceutical analytics platform.

Your task is to map sub-questions to available tools and create an execution plan.

## Available Tools:
{tools_description}

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
    """
    
    def __init__(
        self,
        llm_client: Any,
        tool_registry: Optional[ToolRegistry] = None,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.2,
        max_tools_per_plan: int = 8
    ):
        self.llm_client = llm_client
        self.registry = tool_registry or ToolRegistry()
        self.model = model
        self.temperature = temperature
        self.max_tools_per_plan = max_tools_per_plan
    
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
            # Get available tools for planning
            tools_description = self._format_tools_for_prompt()
            
            # Call LLM for planning
            response = await self._call_llm(decomposition, tools_description)
            
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
                timestamp=datetime.utcnow()
            )
            
            logger.info(f"Created plan with {len(execution_steps)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            raise PlanningError(f"Failed to create execution plan: {e}") from e
    
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
        tools_description: str
    ) -> str:
        """Call the LLM for planning"""
        # Format sub-questions
        sq_text = "\n".join([
            f"- {sq.id}: {sq.question} [Intent: {sq.intent}] [Depends on: {sq.depends_on}]"
            for sq in decomposition.sub_questions
        ])
        
        system_prompt = PLANNING_SYSTEM_PROMPT.format(tools_description=tools_description)
        user_message = PLANNING_USER_TEMPLATE.format(sub_questions=sq_text)
        
        response = await self.llm_client.messages.create(
            model=self.model,
            max_tokens=3000,
            temperature=self.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        return response.content[0].text
    
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
            raise PlanningError(f"Invalid JSON in LLM response: {e}")
    
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
            
            mappings.append(ToolMapping(
                sub_question_id=m["sub_question_id"],
                tool_name=tool_name,
                source_agent=schema.source_agent if schema else "unknown",
                confidence=m.get("confidence", 0.8),
                reasoning=m.get("reasoning", "")
            ))
        
        return mappings
    
    def _build_execution_steps(
        self,
        parsed: Dict[str, Any],
        decomposition: DecompositionResult
    ) -> List[ExecutionStep]:
        """Build ExecutionStep objects from parsed response"""
        raw_steps = parsed.get("execution_steps", [])
        
        # Build mapping from sub_question_id to dependencies
        sq_deps = {sq.id: sq.depends_on for sq in decomposition.sub_questions}
        
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
            
            steps.append(ExecutionStep(
                step_id=s.get("step_id", f"step_{len(steps)+1}"),
                sub_question_id=s["sub_question_id"],
                tool_name=tool_name,
                source_agent=schema.source_agent if schema else "unknown",
                input_mapping=s.get("input_mapping", {}),
                dependency_type=dep_type,
                depends_on_steps=s.get("depends_on_steps", [])
            ))
        
        return steps
    
    def _validate_plan(
        self,
        mappings: List[ToolMapping],
        steps: List[ExecutionStep],
        decomposition: DecompositionResult
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
                    raise PlanningError(
                        f"Step {step.step_id} depends on unknown step {dep}"
                    )
        
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

def plan_sync(
    decomposition: DecompositionResult,
    llm_client: Any,
    **kwargs
) -> ExecutionPlan:
    """
    Synchronous wrapper for planning.
    """
    import asyncio
    
    planner = ToolPlanner(llm_client=llm_client, **kwargs)
    return asyncio.run(planner.plan(decomposition))
