"""
E2I Tool Composer - Main Orchestrator
Version: 4.2
Purpose: Orchestrate the 4-phase tool composition pipeline

The Tool Composer enables dynamic composition of analytical tools to answer
complex, multi-faceted queries that span multiple agent capabilities.

Pipeline:
    Phase 1: DECOMPOSE - Break query into atomic sub-questions
    Phase 2: PLAN     - Map sub-questions to tools, create execution plan
    Phase 3: EXECUTE  - Run tools in dependency order
    Phase 4: SYNTHESIZE - Combine results into coherent response
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .decomposer import QueryDecomposer, DecompositionError
from .planner import ToolPlanner, PlanningError
from .executor import PlanExecutor, ExecutionError
from .synthesizer import ResponseSynthesizer
from .models.composition_models import (
    CompositionPhase,
    CompositionResult,
    SynthesisInput,
)
from src.tool_registry.registry import ToolRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL COMPOSER CLASS
# ============================================================================

class ToolComposer:
    """
    Orchestrates the 4-phase tool composition pipeline.
    
    The Tool Composer is invoked by the Orchestrator when a query is
    classified as MULTI_FACETED - requiring capabilities from multiple
    agents combined in novel ways.
    
    Usage:
        composer = ToolComposer(llm_client=anthropic_client)
        result = await composer.compose(
            query="Compare causal impact of X vs Y and predict outcome Z"
        )
        print(result.response.answer)
    """
    
    def __init__(
        self,
        llm_client: Any,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Tool Composer.
        
        Args:
            llm_client: Anthropic client or compatible LLM client
            tool_registry: Optional custom tool registry (uses global if not provided)
            config: Optional configuration overrides
        """
        self.llm_client = llm_client
        self.registry = tool_registry or ToolRegistry()
        self.config = config or {}
        
        # Initialize phase handlers
        self._init_phase_handlers()
        
        logger.info(f"ToolComposer initialized with {self.registry.tool_count} tools")
    
    def _init_phase_handlers(self) -> None:
        """Initialize handlers for each phase"""
        decompose_config = self.config.get("phases", {}).get("decompose", {})
        plan_config = self.config.get("phases", {}).get("plan", {})
        execute_config = self.config.get("phases", {}).get("execute", {})
        synthesize_config = self.config.get("phases", {}).get("synthesize", {})
        
        self.decomposer = QueryDecomposer(
            llm_client=self.llm_client,
            model=decompose_config.get("model", "claude-sonnet-4-20250514"),
            temperature=decompose_config.get("temperature", 0.3),
            max_sub_questions=decompose_config.get("max_sub_questions", 6),
            min_sub_questions=decompose_config.get("min_sub_questions", 2)
        )
        
        self.planner = ToolPlanner(
            llm_client=self.llm_client,
            tool_registry=self.registry,
            model=plan_config.get("model", "claude-sonnet-4-20250514"),
            temperature=plan_config.get("temperature", 0.2),
            max_tools_per_plan=plan_config.get("max_tools_per_plan", 8)
        )
        
        self.executor = PlanExecutor(
            tool_registry=self.registry,
            max_parallel=execute_config.get("parallel_execution_limit", 3),
            max_retries=execute_config.get("max_retries", 2),
            timeout_seconds=execute_config.get("max_execution_time_seconds", 120)
        )
        
        self.synthesizer = ResponseSynthesizer(
            llm_client=self.llm_client,
            model=synthesize_config.get("model", "claude-sonnet-4-20250514"),
            temperature=synthesize_config.get("temperature", 0.4),
            max_tokens=synthesize_config.get("max_tokens", 2000)
        )
    
    async def compose(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> CompositionResult:
        """
        Execute the full composition pipeline.
        
        Args:
            query: The user's multi-faceted query
            context: Optional context (filters, data references, etc.)
            
        Returns:
            CompositionResult with the synthesized response and full trace
        """
        started_at = datetime.now(timezone.utc)
        phase_durations: Dict[str, int] = {}
        context = context or {}
        
        logger.info(f"Starting composition for query: {query[:100]}...")
        
        try:
            # ================================================================
            # PHASE 1: DECOMPOSE
            # ================================================================
            phase_start = datetime.now(timezone.utc)
            logger.info("Phase 1: Decomposing query...")
            
            decomposition = await self.decomposer.decompose(query)
            
            phase_durations["decompose"] = self._elapsed_ms(phase_start)
            logger.info(
                f"Phase 1 complete: {decomposition.question_count} sub-questions "
                f"({phase_durations['decompose']}ms)"
            )
            
            # ================================================================
            # PHASE 2: PLAN
            # ================================================================
            phase_start = datetime.now(timezone.utc)
            logger.info("Phase 2: Creating execution plan...")
            
            plan = await self.planner.plan(decomposition)
            
            phase_durations["plan"] = self._elapsed_ms(phase_start)
            logger.info(
                f"Phase 2 complete: {plan.step_count} steps planned "
                f"({phase_durations['plan']}ms)"
            )
            
            # ================================================================
            # PHASE 3: EXECUTE
            # ================================================================
            phase_start = datetime.now(timezone.utc)
            logger.info("Phase 3: Executing tool chain...")
            
            execution_trace = await self.executor.execute(plan, context)
            
            phase_durations["execute"] = self._elapsed_ms(phase_start)
            logger.info(
                f"Phase 3 complete: {execution_trace.tools_succeeded}/"
                f"{execution_trace.tools_executed} tools succeeded "
                f"({phase_durations['execute']}ms)"
            )
            
            # ================================================================
            # PHASE 4: SYNTHESIZE
            # ================================================================
            phase_start = datetime.now(timezone.utc)
            logger.info("Phase 4: Synthesizing response...")
            
            synthesis_input = SynthesisInput(
                original_query=query,
                decomposition=decomposition,
                execution_trace=execution_trace
            )
            
            response = await self.synthesizer.synthesize(synthesis_input)
            
            phase_durations["synthesize"] = self._elapsed_ms(phase_start)
            logger.info(
                f"Phase 4 complete: confidence={response.confidence} "
                f"({phase_durations['synthesize']}ms)"
            )
            
            # ================================================================
            # BUILD RESULT
            # ================================================================
            completed_at = datetime.now(timezone.utc)
            total_duration = int((completed_at - started_at).total_seconds() * 1000)
            
            result = CompositionResult(
                query=query,
                decomposition=decomposition,
                plan=plan,
                execution=execution_trace,
                response=response,
                total_duration_ms=total_duration,
                phase_durations=phase_durations,
                success=True,
                started_at=started_at,
                completed_at=completed_at
            )
            
            logger.info(f"Composition complete in {total_duration}ms")
            return result
            
        except DecompositionError as e:
            return self._create_error_result(
                query, started_at, phase_durations, 
                f"Decomposition failed: {e}", CompositionPhase.DECOMPOSE
            )
            
        except PlanningError as e:
            return self._create_error_result(
                query, started_at, phase_durations,
                f"Planning failed: {e}", CompositionPhase.PLAN
            )
            
        except ExecutionError as e:
            return self._create_error_result(
                query, started_at, phase_durations,
                f"Execution failed: {e}", CompositionPhase.EXECUTE
            )
            
        except Exception as e:
            logger.exception(f"Unexpected error during composition: {e}")
            return self._create_error_result(
                query, started_at, phase_durations,
                f"Unexpected error: {e}", None
            )
    
    def _elapsed_ms(self, start: datetime) -> int:
        """Calculate elapsed milliseconds since start"""
        return int((datetime.now(timezone.utc) - start).total_seconds() * 1000)
    
    def _create_error_result(
        self,
        query: str,
        started_at: datetime,
        phase_durations: Dict[str, int],
        error: str,
        failed_phase: Optional[CompositionPhase]
    ) -> CompositionResult:
        """Create an error result when composition fails"""
        from .models.composition_models import (
            ComposedResponse,
            DecompositionResult,
            ExecutionPlan,
            ExecutionTrace,
        )
        
        # Create minimal placeholder objects
        decomposition = DecompositionResult(
            original_query=query,
            sub_questions=[],
            decomposition_reasoning=error
        )
        
        plan = ExecutionPlan(
            decomposition=decomposition,
            steps=[],
            tool_mappings=[],
            planning_reasoning=error
        )
        
        execution = ExecutionTrace(plan_id=plan.plan_id)
        
        response = ComposedResponse(
            answer=f"Unable to complete analysis: {error}",
            confidence=0.0,
            caveats=[error],
            failed_components=[failed_phase.value if failed_phase else "unknown"]
        )
        
        completed_at = datetime.now(timezone.utc)
        
        return CompositionResult(
            query=query,
            decomposition=decomposition,
            plan=plan,
            execution=execution,
            response=response,
            total_duration_ms=int((completed_at - started_at).total_seconds() * 1000),
            phase_durations=phase_durations,
            success=False,
            error=error,
            started_at=started_at,
            completed_at=completed_at
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def compose_query(
    query: str,
    llm_client: Any,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> CompositionResult:
    """
    Convenience function to compose a query.
    
    Args:
        query: The user's multi-faceted query
        llm_client: LLM client for the composition phases
        context: Optional context dictionary
        **kwargs: Additional arguments for ToolComposer
        
    Returns:
        CompositionResult with the synthesized response
    """
    composer = ToolComposer(llm_client=llm_client, **kwargs)
    return await composer.compose(query, context)


def compose_query_sync(
    query: str,
    llm_client: Any,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> CompositionResult:
    """
    Synchronous wrapper for query composition.
    """
    import asyncio
    return asyncio.run(compose_query(query, llm_client, context, **kwargs))


# ============================================================================
# INTEGRATION WITH ORCHESTRATOR
# ============================================================================

class ToolComposerIntegration:
    """
    Integration helper for the Orchestrator to use Tool Composer.
    
    This class provides the interface that the Orchestrator uses
    to invoke the Tool Composer for MULTI_FACETED queries.
    """
    
    def __init__(self, composer: ToolComposer):
        self.composer = composer
    
    async def handle_multi_faceted_query(
        self,
        query: str,
        extracted_entities: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle a multi-faceted query from the Orchestrator.
        
        Args:
            query: The classified MULTI_FACETED query
            extracted_entities: Entities extracted by the NLP layer
            user_context: User context (filters, permissions, etc.)
            
        Returns:
            Response dictionary in the format expected by Orchestrator
        """
        # Merge context
        context = {
            **extracted_entities,
            **user_context
        }
        
        # Run composition
        result = await self.composer.compose(query, context)
        
        # Format for Orchestrator
        return {
            "success": result.success,
            "response": result.response.answer,
            "confidence": result.response.confidence,
            "supporting_data": result.response.supporting_data,
            "citations": result.response.citations,
            "caveats": result.response.caveats,
            "metadata": {
                "composition_id": result.composition_id,
                "sub_questions": result.decomposition.question_count,
                "tools_executed": result.execution.tools_executed,
                "total_duration_ms": result.total_duration_ms,
                "phase_durations": result.phase_durations
            }
        }
